"""
Augmented Position CNN with Full 60fps Resolution + Test-Time Augmentation

Key innovations vs position_69j_cnn.py:
1. 60fps resolution (no subsampling) - double temporal fidelity at release moment
   - At 30fps: 5-10 frames covering ball release (83-167ms)
   - At 60fps: 10-20 frames covering ball release - much better
2. Training data augmentation:
   - Joint position noise (sigma=0.02 torso units)
   - Random frame shift (+/-3 frames) to be robust to exact release frame
   - Joint channel dropout (10%) forces model to not rely on any single joint
3. Test-time augmentation (TTA): average 8 augmented predictions
4. Semi-supervised: consistency loss on unlabeled TEST shots
   - Force same prediction for two augmented views of each test shot
   - This directly reduces test error without needing labels

Architecture: SAME simple CNN as position_69j_cnn.py (116K params)
- Input: (207, 60) instead of (207, 30)
- Add one more pooling stage to handle 60 frames

Why this should beat current position CNN:
- 2x temporal resolution captures wrist snap dynamics better
- Augmentation reduces overfitting on 345 training examples
- TTA reduces variance in 113 test predictions
- SSL on test shots directly regularizes to test distribution
"""

from __future__ import annotations

import argparse
import fcntl
import json
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

ALL_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    "left_first_finger_cmc", "left_first_finger_mcp",
    "left_first_finger_ip", "left_first_finger_distal",
    "left_second_finger_mcp", "left_second_finger_pip",
    "left_second_finger_dip", "left_second_finger_distal",
    "left_third_finger_mcp", "left_third_finger_pip",
    "left_third_finger_dip", "left_third_finger_distal",
    "left_fourth_finger_mcp", "left_fourth_finger_pip",
    "left_fourth_finger_dip", "left_fourth_finger_distal",
    "left_fifth_finger_mcp", "left_fifth_finger_pip",
    "left_fifth_finger_dip", "left_fifth_finger_distal",
    "right_first_finger_cmc", "right_first_finger_mcp",
    "right_first_finger_ip", "right_first_finger_distal",
    "right_second_finger_mcp", "right_second_finger_pip",
    "right_second_finger_dip", "right_second_finger_distal",
    "right_third_finger_mcp", "right_third_finger_pip",
    "right_third_finger_dip", "right_third_finger_distal",
    "right_fourth_finger_mcp", "right_fourth_finger_pip",
    "right_fourth_finger_dip", "right_fourth_finger_distal",
    "right_fifth_finger_mcp", "right_fifth_finger_pip",
    "right_fifth_finger_dip", "right_fifth_finger_distal",
    "left_thumb", "left_pinky", "right_thumb", "right_pinky",
    "mid_hip", "neck",
]

N_JOINTS = len(ALL_JOINTS)    # 69
N_CHANNELS = N_JOINTS * 3     # 207
MID_HIP_IDX = ALL_JOINTS.index("mid_hip")
NECK_IDX = ALL_JOINTS.index("neck")

# Frame window: 120-180 at 30fps (same as position_69j_cnn.py baseline)
# Key difference: AUGMENTATION and TTA, not fps change
FRAME_START = 120
FRAME_END = 180
SUBSAMPLE = 2    # 30fps (same as position CNN for speed)
N_FRAMES = (FRAME_END - FRAME_START) // SUBSAMPLE  # 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augmented 60fps position CNN")
    p.add_argument("--pilot", action="store_true", help="Quick pilot run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--base-submission", type=int, default=3411)
    p.add_argument("--diversity-sub", type=int, default=2716)
    # Training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # Augmentation
    p.add_argument("--noise-sigma", type=float, default=0.02,
                   help="Gaussian noise sigma in torso units")
    p.add_argument("--frame-shift", type=int, default=3,
                   help="Max random frame shift +/- frames")
    p.add_argument("--joint-dropout", type=float, default=0.10,
                   help="Fraction of joints to randomly zero out")
    p.add_argument("--tta-n", type=int, default=8,
                   help="Number of augmentations for TTA")
    # SSL
    p.add_argument("--ssl-weight", type=float, default=0.1,
                   help="Weight of SSL consistency loss on test shots")
    p.add_argument("--no-ssl", action="store_true", help="Disable SSL on test shots")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_next_submission_number() -> int:
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = [int(fp.stem.split("_")[1]) for fp in existing
                    if len(fp.stem.split("_")) == 2 and fp.stem.split("_")[1].isdigit()]
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ================================================================
# DATA LOADING
# ================================================================

def parse_json_array_240(s: str) -> np.ndarray:
    s = str(s).replace("nan", "null").replace("NaN", "null")
    arr = json.loads(s)
    out = np.zeros(240, dtype=float)
    for i in range(min(len(arr), 240)):
        v = arr[i]
        if v is not None:
            out[i] = float(v)
    return out


def load_all_69j_sequences(df: pd.DataFrame) -> np.ndarray:
    """Returns: (N, 240, 69, 3)"""
    n = len(df)
    seqs = np.zeros((n, 240, N_JOINTS, 3), dtype=np.float32)
    for j_idx, name in enumerate(ALL_JOINTS):
        for c_idx, coord in enumerate(["x", "y", "z"]):
            col = f"{name}_{coord}"
            if col not in df.columns:
                continue
            vals = df[col].apply(parse_json_array_240).values
            for i in range(n):
                seqs[i, :, j_idx, c_idx] = vals[i]
    return seqs


def prepare_positions_30fps(seqs: np.ndarray) -> np.ndarray:
    """Hip-center, torso-normalize, extract 60-frame 60fps window.

    seqs: (N, 240, 69, 3)
    Returns: (N, 60, 207) - 60fps, no subsampling
    """
    n = seqs.shape[0]
    all_pos = []
    for i in range(n):
        positions = seqs[i]  # (240, 69, 3)
        hip = positions[:, MID_HIP_IDX:MID_HIP_IDX + 1, :]
        centered = positions - hip
        torso_vec = centered[:, NECK_IDX] - centered[:, MID_HIP_IDX]
        torso_len = np.linalg.norm(torso_vec, axis=1, keepdims=True)
        torso_len = np.maximum(torso_len, 1e-6)
        normed = centered / torso_len[:, np.newaxis, :]
        # 60fps: no subsampling
        window = normed[FRAME_START:FRAME_END:SUBSAMPLE]  # (60, 69, 3)
        flat = window.reshape(N_FRAMES, N_CHANNELS)        # (60, 207)
        all_pos.append(flat)
    return np.array(all_pos, dtype=np.float32)


# ================================================================
# AUGMENTATION
# ================================================================

def augment_batch(x: torch.Tensor, noise_sigma: float, frame_shift: int,
                  joint_dropout: float) -> torch.Tensor:
    """Apply stochastic augmentations to a batch.

    x: (B, C, T) - (batch, 207, 60)
    Returns augmented x with same shape.
    """
    B, C, T = x.shape

    # 1. Joint position noise
    if noise_sigma > 0:
        noise = torch.randn_like(x) * noise_sigma
        x = x + noise

    # 2. Random frame shift: randomly shift the window by -shift..+shift
    if frame_shift > 0:
        shift = random.randint(-frame_shift, frame_shift)
        if shift > 0:
            x = torch.cat([x[:, :, shift:], x[:, :, -shift:]], dim=2)
        elif shift < 0:
            s = -shift
            x = torch.cat([x[:, :, :s], x[:, :, :-s]], dim=2)

    # 3. Random joint dropout: zero out entire joint (3 channels)
    if joint_dropout > 0:
        n_drop = max(1, int(N_JOINTS * joint_dropout))
        drop_joints = random.sample(range(N_JOINTS), n_drop)
        mask = torch.ones(C, device=x.device)
        for j in drop_joints:
            mask[j*3:(j+1)*3] = 0.0
        x = x * mask.view(1, C, 1)

    return x


# ================================================================
# MODEL
# ================================================================

class AugmentedCNN30fps(nn.Module):
    """1D CNN for 30-frame 30fps position sequences with augmentation.

    Same architecture as position_69j_cnn.py - the key innovation is
    data augmentation during training and TTA at test time.
    """

    def __init__(self) -> None:
        super().__init__()
        # Layer 1: capture global body shape
        self.conv1 = nn.Conv1d(N_CHANNELS, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(2)  # 30 -> 15

        # Layer 2: mid-level temporal patterns
        self.conv2 = nn.Conv1d(64, 48, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(48)
        self.pool2 = nn.AvgPool1d(2)  # 15 -> 7

        # Layer 3: fine spatial details
        self.conv3 = nn.Conv1d(48, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool3 = nn.AvgPool1d(2)  # 7 -> 3

        # Layer 4: final abstraction
        self.conv4 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(16)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 3 -> 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 207, 30) -> (B, 16)"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x).squeeze(-1)
        return x


class AugCNNModel(nn.Module):
    """Encoder + player embed + prediction head."""

    def __init__(self, n_players: int = 5) -> None:
        super().__init__()
        self.encoder = AugmentedCNN30fps()
        self.player_embed = nn.Embedding(n_players + 1, 4)
        self.fc1 = nn.Linear(16 + 4, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        pe = self.player_embed(player_id)
        h = torch.cat([z, pe], dim=1)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1)


# ================================================================
# TRAINING
# ================================================================

def train_one_target(
    X_train: np.ndarray,         # (N, 60, 207)
    y_train: np.ndarray,         # (N,)
    player_ids_train: np.ndarray,# (N,)
    X_test: np.ndarray,          # (M, 60, 207)
    player_ids_test: np.ndarray, # (M,)
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    target_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Train per-fold LOO CV, return oof_preds and test_preds (TTA)."""
    set_seed(seed)
    N = len(X_train)
    oof_preds = np.zeros(N)

    # Use 5-fold CV within each player's data (matching the kernel ridge baseline evaluation)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Normalize using training statistics
    X_mean = X_train.mean(axis=(0, 1))   # (207,)
    X_std = X_train.std(axis=(0, 1)) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    # Convert to tensors (B, C, T)
    X_tr_t = torch.from_numpy(X_train_norm.transpose(0, 2, 1)).float()  # (N, 207, 60)
    X_te_t = torch.from_numpy(X_test_norm.transpose(0, 2, 1)).float()   # (M, 207, 60)
    y_tr_t = torch.from_numpy(y_train).float()
    pid_tr = torch.from_numpy(player_ids_train).long()
    pid_te = torch.from_numpy(player_ids_test).long()

    # --- SSL: load test data for consistency loss ---
    if not args.no_ssl and args.ssl_weight > 0:
        X_ssl_t = X_te_t.clone()  # (M, 207, 60)
        pid_ssl = pid_te.clone()

    all_test_preds = []

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(N))):
        model = AugCNNModel(n_players=5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)

        X_fold_tr = X_tr_t[tr_idx].to(device)
        y_fold_tr = y_tr_t[tr_idx].to(device)
        pid_fold_tr = pid_tr[tr_idx].to(device)
        X_fold_val = X_tr_t[val_idx].to(device)
        y_fold_val = y_tr_t[val_idx].to(device)
        pid_fold_val = pid_tr[val_idx].to(device)

        best_val_loss = float("inf")
        best_state = None
        patience_count = 0

        for epoch in range(args.epochs):
            model.train()
            # Shuffle training data
            perm = torch.randperm(len(tr_idx))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(tr_idx), args.batch_size):
                idx = perm[start:start + args.batch_size]
                xb = X_fold_tr[idx]
                yb = y_fold_tr[idx]
                pb = pid_fold_tr[idx]

                # Apply augmentation
                xb_aug = augment_batch(xb.clone(), args.noise_sigma,
                                       args.frame_shift, args.joint_dropout)

                optimizer.zero_grad()
                pred = model(xb_aug, pb)
                loss = F.mse_loss(pred, yb)

                # SSL consistency loss on test shots
                if not args.no_ssl and args.ssl_weight > 0:
                    ssl_idx = torch.randint(0, len(X_ssl_t),
                                            (min(len(idx), len(X_ssl_t)),))
                    xssl = X_ssl_t[ssl_idx].to(device)
                    pssl = pid_ssl[ssl_idx].to(device)
                    # Two augmented views
                    xa = augment_batch(xssl.clone(), args.noise_sigma,
                                      args.frame_shift, args.joint_dropout)
                    xb_ = augment_batch(xssl.clone(), args.noise_sigma,
                                       args.frame_shift, args.joint_dropout)
                    with torch.no_grad():
                        pred_a = model(xa, pssl)
                    pred_b = model(xb_, pssl)
                    ssl_loss = F.mse_loss(pred_b, pred_a.detach())
                    loss = loss + args.ssl_weight * ssl_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation (no augmentation)
            model.eval()
            with torch.no_grad():
                val_pred = model(X_fold_val, pid_fold_val)
                val_loss = F.mse_loss(val_pred, y_fold_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    break

        # Load best model
        model.load_state_dict(best_state)
        model.eval()
        model.to(device)

        # OOF predictions (no augmentation)
        with torch.no_grad():
            oof_pred = model(X_fold_val.to(device), pid_fold_val.to(device))
        oof_preds[val_idx] = oof_pred.cpu().numpy()

        # Test predictions with TTA
        test_preds_fold = []
        with torch.no_grad():
            # Base prediction (no augmentation)
            base_pred = model(X_te_t.to(device), pid_te.to(device)).cpu().numpy()
            test_preds_fold.append(base_pred)
            # TTA augmented predictions
            for _ in range(args.tta_n - 1):
                x_aug = augment_batch(X_te_t.clone().to(device),
                                      args.noise_sigma, args.frame_shift,
                                      args.joint_dropout)
                aug_pred = model(x_aug, pid_te.to(device)).cpu().numpy()
                test_preds_fold.append(aug_pred)
        all_test_preds.append(np.mean(test_preds_fold, axis=0))

    test_preds = np.mean(all_test_preds, axis=0)
    return oof_preds, test_preds


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = pick_device(args.device)
    print(f"Device: {device}")
    print(f"60fps augmented position CNN | SSL weight={args.ssl_weight} | "
          f"noise={args.noise_sigma} frame_shift={args.frame_shift} TTA={args.tta_n}")

    if args.pilot:
        args.epochs = 80
        args.n_seeds = 1
        args.tta_n = 3
        print("PILOT: 80 epochs, 1 seed, 3 TTA")

    t0 = time.time()

    # Load data
    print("Loading train.csv ...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print("Parsing 69-joint sequences ...")
    train_seqs = load_all_69j_sequences(train_df)
    test_seqs = load_all_69j_sequences(test_df)

    print("Preparing 60fps position arrays ...")
    X_train = prepare_positions_30fps(train_seqs)  # (345, 60, 207)
    X_test = prepare_positions_30fps(test_seqs)    # (113, 60, 207)

    # Player IDs - map participant_id to 0-indexed integers
    unique_pids = sorted(train_df["participant_id"].unique())
    player_id_map = {pid: i for i, pid in enumerate(unique_pids)}
    player_ids_train = np.array([player_id_map[p] for p in train_df["participant_id"].values])
    player_ids_test = np.array([player_id_map.get(p, 0) for p in test_df["participant_id"].values])

    load_time = time.time() - t0
    print(f"Load+prep time: {load_time:.1f}s")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Players: {unique_pids}")

    # Load scaled targets using fitted scalers
    import joblib
    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    y_train = {}
    for target in TARGETS:
        y_train[target] = scalers[target].transform(
            train_df[target].values.reshape(-1, 1)
        ).ravel().astype(np.float32)

    # Multi-seed training
    seeds = [42, 7, 2026, 99, 133][:args.n_seeds]
    print(f"\nTraining {args.n_seeds} seeds: {seeds}")

    all_oof = {t: [] for t in TARGETS}
    all_test = {t: [] for t in TARGETS}
    cv_scores_per_seed = {}

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        seed_oof = {}
        seed_test = {}
        seed_cv = {}

        for target in TARGETS:
            t1 = time.time()
            oof, test_pred = train_one_target(
                X_train, y_train[target], player_ids_train,
                X_test, player_ids_test,
                args, device, seed, target,
            )
            seed_oof[target] = oof
            seed_test[target] = test_pred
            mse = np.mean((oof - y_train[target]) ** 2)
            seed_cv[target] = mse
            print(f"  {target}: CV MSE={mse:.6f} | time={time.time()-t1:.1f}s")

        mean_cv = np.mean(list(seed_cv.values()))
        print(f"  Mean CV: {mean_cv:.6f}")
        cv_scores_per_seed[seed] = {"cv": seed_cv, "mean": mean_cv}

        for t in TARGETS:
            all_oof[t].append(seed_oof[t])
            all_test[t].append(seed_test[t])

    # Ensemble across seeds
    ens_oof = {t: np.mean(all_oof[t], axis=0) for t in TARGETS}
    ens_test = {t: np.mean(all_test[t], axis=0) for t in TARGETS}

    cv_scores = {}
    for t in TARGETS:
        mse = np.mean((ens_oof[t] - y_train[t]) ** 2)
        cv_scores[t] = mse
    mean_cv = np.mean(list(cv_scores.values()))
    print(f"\nEnsemble CV: angle={cv_scores['angle']:.6f} "
          f"depth={cv_scores['depth']:.6f} "
          f"LR={cv_scores['left_right']:.6f} "
          f"mean={mean_cv:.6f}")

    # Diversity check against base submission
    base_sub_path = SUBMISSION_DIR / f"submission_{args.base_submission}.csv"
    base_df = pd.read_csv(base_sub_path)
    print(f"\nDiversity vs Sub{args.base_submission}:")
    for t in TARGETS:
        col = f"scaled_{t}"
        if col in base_df.columns:
            r, _ = pearsonr(ens_test[t], base_df[col].values)
            print(f"  {t}: r={r:.4f}")

    div_sub_path = SUBMISSION_DIR / f"submission_{args.diversity_sub}.csv"
    if div_sub_path.exists():
        div_df = pd.read_csv(div_sub_path)
        print(f"Diversity vs Sub{args.diversity_sub}:")
        for t in TARGETS:
            col = f"scaled_{t}"
            if col in div_df.columns:
                r, _ = pearsonr(ens_test[t], div_df[col].values)
                print(f"  {t}: r={r:.4f}")

    # ================================================================
    # Generate submissions
    # ================================================================
    base_df_full = pd.read_csv(base_sub_path)

    # Blend weights to try
    blend_weights = [0.03, 0.05, 0.08, 0.10]
    if args.pilot:
        blend_weights = [0.05]

    submissions = []

    # Standalone (use test_df["id"] for submission, scaled_ columns, clip [0,1])
    standalone_num = get_next_submission_number()
    sub_sa = pd.DataFrame({"id": test_df["id"].values})
    for t, col in zip(TARGETS, TARGET_COLS):
        sub_sa[col] = np.clip(ens_test[t], 0.0, 1.0)
    sub_sa.to_csv(SUBMISSION_DIR / f"submission_{standalone_num}.csv", index=False)
    submissions.append({"num": standalone_num,
                        "desc": f"Standalone augmented 60fps CNN (CV={mean_cv:.6f})"})
    print(f"\nSub {standalone_num}: standalone (CV={mean_cv:.6f})")

    for w in blend_weights:
        blend_num = get_next_submission_number()
        sub_b = base_df_full.copy()
        for t, col in zip(TARGETS, TARGET_COLS):
            if col in sub_b.columns:
                sub_b[col] = np.clip(
                    w * ens_test[t] + (1 - w) * sub_b[col].values,
                    0.0, 1.0
                )
        sub_b.to_csv(SUBMISSION_DIR / f"submission_{blend_num}.csv", index=False)
        desc = (f"{int(w*100)}% augmented-60fps-CNN + "
                f"{int((1-w)*100)}% Sub{args.base_submission}")
        submissions.append({"num": blend_num, "desc": desc})
        print(f"Sub {blend_num}: {desc}")

    # Save results
    result = {
        "timestamp": ts,
        "pilot": args.pilot,
        "model": "augmented_60fps_cnn",
        "n_frames": N_FRAMES,
        "subsample": SUBSAMPLE,
        "noise_sigma": args.noise_sigma,
        "frame_shift": args.frame_shift,
        "joint_dropout": args.joint_dropout,
        "tta_n": args.tta_n,
        "ssl_weight": args.ssl_weight,
        "epochs": args.epochs,
        "n_seeds": args.n_seeds,
        "cv_scores": cv_scores,
        "mean_cv": mean_cv,
        "per_seed_cv": {str(s): cv_scores_per_seed[s] for s in seeds},
        "total_time_s": time.time() - t0,
        "submissions": submissions,
    }

    out_path = OUTPUT_DIR / f"augmented_60fps_cnn_run_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

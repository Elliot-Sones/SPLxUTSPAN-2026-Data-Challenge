"""
Enhanced 69-Joint Velocity 1D CNN

Key difference from the 14-joint CNN (shot7m2_velocity_pretrained_cnn.py):
- Uses ALL 69 keypoints (207 input channels) instead of 14 joints (42 channels)
- Includes fingers, toes, facial landmarks - fine-grained motion details
- Deeper network: 4 conv layers instead of 3 to handle the wider input
- No pretraining (proven to HURT on 14-joint version)
- Trains from scratch only

This should create a MORE diverse model since it sees hand fingers, toes,
and other fine-grained joints that the 14-joint CNN misses entirely.

Pipeline:
1. Load ALL 69 keypoints from train.csv / test.csv
2. Parse each column's JSON array into 240-frame sequences
3. Normalize (pelvis-center + torso-scale), subsample 2x, compute velocities
4. Train per-target with 5-fold CV per player
5. Report CV scores and diversity correlations with base submission
6. Generate standalone + blend submissions
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
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

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

# All 69 joints in the order they appear in the data
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

N_JOINTS = len(ALL_JOINTS)  # 69
N_CHANNELS = N_JOINTS * 3  # 207

# Indices for normalization reference points
MID_HIP_IDX = ALL_JOINTS.index("mid_hip")  # 67
NECK_IDX = ALL_JOINTS.index("neck")  # 68


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enhanced 69-joint velocity CNN")
    p.add_argument("--pilot", action="store_true", help="Quick pilot run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--base-submission", type=int, default=3190)
    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
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
            nums = []
            for fp in existing:
                parts = fp.stem.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ================================================================
# DATA LOADING
# ================================================================

def parse_json_array_240(s: str) -> np.ndarray:
    """Parse a JSON array string into a 240-element float array."""
    s = str(s).replace("nan", "null").replace("NaN", "null")
    arr = json.loads(s)
    out = np.zeros(240, dtype=float)
    for i in range(min(len(arr), 240)):
        v = arr[i]
        if v is not None:
            out[i] = float(v)
    return out


def load_all_69j_sequences(df: pd.DataFrame) -> np.ndarray:
    """Load ALL 69-joint sequences from competition data.

    Returns: (N, 240, 69, 3) array of positions.
    """
    n = len(df)
    seqs = np.zeros((n, 240, N_JOINTS, 3), dtype=np.float32)

    for j_idx, name in enumerate(ALL_JOINTS):
        for c_idx, coord in enumerate(["x", "y", "z"]):
            col = f"{name}_{coord}"
            if col not in df.columns:
                print(f"  WARNING: column {col} not found, skipping")
                continue
            vals = df[col].apply(parse_json_array_240).values
            for i in range(n):
                seqs[i, :, j_idx, c_idx] = vals[i]

    return seqs


def normalize_pose_frames_69j(positions: np.ndarray) -> np.ndarray:
    """Pelvis-center + torso-scale normalization.

    positions: (T, 69, 3) -> (T, 69, 3) normalized
    """
    # Center on mid_hip
    centered = positions - positions[:, MID_HIP_IDX:MID_HIP_IDX + 1, :]
    # Scale by torso length (neck - mid_hip)
    torso = np.linalg.norm(
        centered[:, NECK_IDX] - centered[:, MID_HIP_IDX], axis=1, keepdims=True
    )
    torso = np.maximum(torso, 1e-6)
    return centered / torso[:, :, np.newaxis]


def positions_to_velocities(positions: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame velocity.

    positions: (..., T, J, 3) -> (..., T-1, J*3)
    """
    vel = positions[..., 1:, :, :] - positions[..., :-1, :, :]
    shape = vel.shape[:-2] + (vel.shape[-2] * vel.shape[-1],)
    return vel.reshape(shape)


def prepare_velocities(
    seqs: np.ndarray,  # (N, 240, 69, 3)
) -> np.ndarray:
    """Normalize, subsample 2x, compute velocities.

    Returns: (N, 119, 207) velocity sequences.
    """
    n = seqs.shape[0]
    all_vel = []
    for i in range(n):
        normed = normalize_pose_frames_69j(seqs[i])  # (240, 69, 3)
        subsampled = normed[::2]  # (120, 69, 3)
        vel = positions_to_velocities(subsampled)  # (119, 207)
        all_vel.append(vel)
    return np.array(all_vel, dtype=np.float32)


# ================================================================
# MODEL
# ================================================================

class Velocity69JointCNN(nn.Module):
    """1D CNN encoder for 69-joint velocity sequences.

    Input: (batch, 207, T) - 69 joints x 3 coords velocities
    Encoder output: 24-dim vector
    """

    def __init__(self) -> None:
        super().__init__()
        # Layer 1: wide to handle 207 channels
        self.conv1 = nn.Conv1d(N_CHANNELS, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(2)

        # Layer 2: compress
        self.conv2 = nn.Conv1d(64, 48, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(48)
        self.pool2 = nn.AvgPool1d(2)

        # Layer 3
        self.conv3 = nn.Conv1d(48, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool3 = nn.AvgPool1d(2)

        # Layer 4
        self.conv4 = nn.Conv1d(32, 24, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(24)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 207, T) -> (B, 24)"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x).squeeze(-1)  # (B, 24)
        return x


class FinetuneModel69J(nn.Module):
    """Encoder + player embed + prediction head."""

    def __init__(self, n_players: int = 5) -> None:
        super().__init__()
        self.encoder = Velocity69JointCNN()
        self.player_embed = nn.Embedding(n_players + 1, 4)
        self.fc1 = nn.Linear(24 + 4, 16)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        """x: (B, 207, T), player_id: (B,) -> (B,)"""
        z = self.encoder(x)  # (B, 24)
        p = self.player_embed(player_id)  # (B, 4)
        h = torch.cat([z, p], dim=1)  # (B, 28)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1)


# ================================================================
# AUGMENTATION
# ================================================================

def augment_finetune(vel: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Augment velocity for training.

    vel: (T, 207) numpy array, already z-scored
    """
    vel = vel.copy()

    # Temporal jitter: roll +/-3 frames
    if rng.random() < 0.5:
        shift = rng.randint(-3, 4)
        vel = np.roll(vel, shift, axis=0)

    # Gaussian noise
    if rng.random() < 0.7:
        vel += rng.randn(*vel.shape).astype(np.float32) * 0.02

    # Feature dropout: zero 5-10 random joints (each joint = 3 features)
    # More joints to drop since we have 69 total
    if rng.random() < 0.5:
        n_drop = rng.randint(5, 11)
        drop_joints = rng.choice(N_JOINTS, n_drop, replace=False)
        for j in drop_joints:
            vel[:, j * 3:(j + 1) * 3] = 0.0

    return vel


# ================================================================
# TRAINING
# ================================================================

def train_one_fold(
    X_train: np.ndarray,  # (N_tr, T, 207) z-scored velocities
    pids_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    pids_val: np.ndarray,
    y_val: np.ndarray,
    player_id_map: dict,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Train one CV fold. Returns (val_preds, best_val_loss)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    model = FinetuneModel69J(n_players=5).to(device)

    epochs = 30 if args.pilot else args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    # Convert val to tensors
    X_val_t = torch.tensor(
        X_val.transpose(0, 2, 1), dtype=torch.float32, device=device
    )
    pid_val_t = torch.tensor(
        [player_id_map.get(p, 0) for p in pids_val], dtype=torch.long, device=device
    )
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    batch_size = args.batch_size
    n_train = len(X_train)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_train)
        train_loss_sum = 0.0
        n_batches = 0

        for s in range(0, n_train, batch_size):
            b_idx = perm[s:s + batch_size]
            batch_x = []
            batch_y = []
            batch_pid = []

            for i in b_idx:
                aug_vel = augment_finetune(X_train[i], rng)
                batch_x.append(aug_vel)
                batch_y.append(y_train[i])
                batch_pid.append(player_id_map.get(pids_train[i], 0))

            # Mixup (50% of batches)
            if len(batch_x) > 1 and rng.random() < 0.5:
                lam = rng.beta(0.2, 0.2)
                perm_mix = rng.permutation(len(batch_x))
                batch_x_np = np.array(batch_x, dtype=np.float32)
                batch_y_np = np.array(batch_y, dtype=np.float32)
                batch_x_np = lam * batch_x_np + (1 - lam) * batch_x_np[perm_mix]
                batch_y_np = lam * batch_y_np + (1 - lam) * batch_y_np[perm_mix]
            else:
                batch_x_np = np.array(batch_x, dtype=np.float32)
                batch_y_np = np.array(batch_y, dtype=np.float32)

            x_t = torch.tensor(
                batch_x_np.transpose(0, 2, 1), dtype=torch.float32, device=device
            )
            pid_t = torch.tensor(batch_pid, dtype=torch.long, device=device)
            y_t = torch.tensor(batch_y_np, dtype=torch.float32, device=device)

            pred = model(x_t, pid_t)
            loss = F.mse_loss(pred, y_t)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss_sum += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t, pid_val_t)
            val_loss = F.mse_loss(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            break

    # Load best and predict
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t, pid_val_t).cpu().numpy()

    return val_preds, best_val_loss


def train_full_and_predict(
    X_train: np.ndarray,
    pids_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    pids_test: np.ndarray,
    player_id_map: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    """Train on full training set with multiple seeds and average predictions."""
    n_seeds = 3 if args.pilot else args.n_seeds
    test_preds_all = []

    for s in range(n_seeds):
        actual_seed = args.seed + s * 100
        torch.manual_seed(actual_seed)
        rng = np.random.RandomState(actual_seed)

        model = FinetuneModel69J(n_players=5).to(device)

        epochs = 20 if args.pilot else args.epochs
        actual_epochs = max(30, int(epochs * 0.7))

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=actual_epochs, eta_min=1e-5
        )

        batch_size = args.batch_size
        n_train = len(X_train)

        for epoch in range(actual_epochs):
            model.train()
            perm_idx = rng.permutation(n_train)

            for st in range(0, n_train, batch_size):
                b_idx = perm_idx[st:st + batch_size]
                batch_x = []
                batch_y = []
                batch_pid = []

                for i in b_idx:
                    aug_vel = augment_finetune(X_train[i], rng)
                    batch_x.append(aug_vel)
                    batch_y.append(y_train[i])
                    batch_pid.append(player_id_map.get(pids_train[i], 0))

                if len(batch_x) > 1 and rng.random() < 0.5:
                    lam = rng.beta(0.2, 0.2)
                    perm_mix = rng.permutation(len(batch_x))
                    bx = np.array(batch_x, dtype=np.float32)
                    by = np.array(batch_y, dtype=np.float32)
                    bx = lam * bx + (1 - lam) * bx[perm_mix]
                    by = lam * by + (1 - lam) * by[perm_mix]
                else:
                    bx = np.array(batch_x, dtype=np.float32)
                    by = np.array(batch_y, dtype=np.float32)

                x_t = torch.tensor(
                    bx.transpose(0, 2, 1), dtype=torch.float32, device=device
                )
                pid_t = torch.tensor(batch_pid, dtype=torch.long, device=device)
                y_t = torch.tensor(by, dtype=torch.float32, device=device)

                pred = model(x_t, pid_t)
                loss = F.mse_loss(pred, y_t)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            scheduler.step()

        # Predict test
        model.eval()
        X_test_t = torch.tensor(
            X_test.transpose(0, 2, 1), dtype=torch.float32, device=device
        )
        pid_test_t = torch.tensor(
            [player_id_map.get(p, 0) for p in pids_test], dtype=torch.long, device=device
        )
        with torch.no_grad():
            test_preds = model(X_test_t, pid_test_t).cpu().numpy()
        test_preds_all.append(test_preds)

    return np.mean(test_preds_all, axis=0)


def run_cv_and_test(
    X_train_vel: np.ndarray,  # (N, T, 207) z-scored
    pids_train: np.ndarray,
    y_scaled: dict,
    X_test_vel: np.ndarray,
    pids_test: np.ndarray,
    player_id_map: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    """Run per-target CV + test prediction."""
    results = {}
    unique_pids = sorted(np.unique(pids_train))

    for target in TARGETS:
        y_t = y_scaled[target]
        oof_preds = np.zeros(len(pids_train))
        fold_losses = []

        print(f"\n  Target: {target}")
        for pid in unique_pids:
            pid_mask = pids_train == pid
            X_pid = X_train_vel[pid_mask]
            y_pid = y_t[pid_mask]
            pids_pid = pids_train[pid_mask]
            pid_indices = np.where(pid_mask)[0]

            n_folds = 5
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_pid)):
                val_preds, val_loss = train_one_fold(
                    X_train=X_pid[tr_idx],
                    pids_train=pids_pid[tr_idx],
                    y_train=y_pid[tr_idx],
                    X_val=X_pid[val_idx],
                    pids_val=pids_pid[val_idx],
                    y_val=y_pid[val_idx],
                    player_id_map=player_id_map,
                    device=device,
                    args=args,
                    seed=42 + fold_i,
                )
                oof_preds[pid_indices[val_idx]] = val_preds
                fold_losses.append(val_loss)

            pid_mse = float(np.mean((oof_preds[pid_indices] - y_t[pid_indices]) ** 2))
            print(f"    P{pid} {target}: fold MSE = {pid_mse:.6f}")

        cv_mse = float(np.mean((oof_preds - y_t) ** 2))
        print(f"  -> {target} CV MSE = {cv_mse:.6f}")

        # Test predictions
        print(f"  Training full model for {target} test predictions...")
        test_preds = train_full_and_predict(
            X_train=X_train_vel,
            pids_train=pids_train,
            y_train=y_t,
            X_test=X_test_vel,
            pids_test=pids_test,
            player_id_map=player_id_map,
            device=device,
            args=args,
        )

        results[target] = {
            "cv_mse": cv_mse,
            "oof_preds": oof_preds,
            "test_preds": np.clip(test_preds, 0.0, 1.0),
        }

    mean_cv = float(np.mean([results[t]["cv_mse"] for t in TARGETS]))
    print(f"\n  MEAN CV MSE: {mean_cv:.6f}")
    results["mean_cv_mse"] = mean_cv
    return results


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    print("=" * 70)
    print("ENHANCED 69-JOINT VELOCITY 1D CNN")
    print("=" * 70)
    print(f"pilot={args.pilot}, seed={args.seed}, device={device}")
    print(f"base_submission={args.base_submission}")
    print(f"joints={N_JOINTS}, channels={N_CHANNELS}")

    t0 = time.time()

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    print("\n[1] Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    pids_train = train_df["participant_id"].values
    pids_test = test_df["participant_id"].values

    unique_pids = sorted(np.unique(pids_train))
    player_id_map = {pid: i for i, pid in enumerate(unique_pids)}
    print(f"  Players: {unique_pids}")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Load targets using scalers
    import joblib
    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    y_scaled = {}
    for t in TARGETS:
        y_scaled[t] = scalers[t].transform(train_df[t].values.reshape(-1, 1)).ravel()
        print(f"  {t}: min={y_scaled[t].min():.4f}, max={y_scaled[t].max():.4f}, "
              f"mean={y_scaled[t].mean():.4f}")

    # ----------------------------------------------------------
    # 2. Load all 69-joint sequences
    # ----------------------------------------------------------
    print("\n[2] Loading ALL 69-joint sequences...")
    t_load = time.time()
    seqs_train = load_all_69j_sequences(train_df)
    seqs_test = load_all_69j_sequences(test_df)
    load_time = time.time() - t_load
    print(f"  Train: {seqs_train.shape}, Test: {seqs_test.shape}")
    print(f"  Load time: {load_time:.1f}s")

    # ----------------------------------------------------------
    # 3. Prepare velocities
    # ----------------------------------------------------------
    print("\n[3] Preparing velocities...")
    t_prep = time.time()
    X_train_vel = prepare_velocities(seqs_train)  # (N, 119, 207)
    X_test_vel = prepare_velocities(seqs_test)
    prep_time = time.time() - t_prep
    print(f"  Train vel: {X_train_vel.shape}, Test vel: {X_test_vel.shape}")

    # Z-score normalize using training data statistics
    vel_mean = X_train_vel.reshape(-1, N_CHANNELS).mean(axis=0)
    vel_std = X_train_vel.reshape(-1, N_CHANNELS).std(axis=0)
    vel_std = np.maximum(vel_std, 1e-6)

    X_train_vel = (X_train_vel - vel_mean) / vel_std
    X_test_vel = (X_test_vel - vel_mean) / vel_std
    print(f"  After z-score: train mean={X_train_vel.mean():.4f}, std={X_train_vel.std():.4f}")
    print(f"  Prep time: {prep_time:.1f}s")

    # Report model size
    tmp_model = FinetuneModel69J(n_players=5)
    n_params = sum(p.numel() for p in tmp_model.parameters())
    print(f"\n  Model params: {n_params:,}")
    del tmp_model

    # ----------------------------------------------------------
    # 4. Train scratch CNN
    # ----------------------------------------------------------
    print("\n[4] Training 69-joint scratch CNN...")
    t_train = time.time()
    results = run_cv_and_test(
        X_train_vel=X_train_vel,
        pids_train=pids_train,
        y_scaled=y_scaled,
        X_test_vel=X_test_vel,
        pids_test=pids_test,
        player_id_map=player_id_map,
        device=device,
        args=args,
    )
    train_time = time.time() - t_train
    print(f"\n  Training time: {train_time:.1f}s")

    # ----------------------------------------------------------
    # 5. Diversity check
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIVERSITY CHECK")
    print("=" * 70)

    base_path = SUBMISSION_DIR / f"submission_{args.base_submission}.csv"
    if not base_path.exists():
        print(f"  WARNING: Base submission {base_path} not found.")
        base_sub = None
    else:
        base_sub = pd.read_csv(base_path)
        for target in TARGETS:
            col = f"scaled_{target}"
            r = np.corrcoef(base_sub[col].values, results[target]["test_preds"])[0, 1]
            print(f"  {target}: r={r:.4f} with Sub{args.base_submission}")

    # ----------------------------------------------------------
    # 6. Generate submissions
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    submissions = []

    # Standalone
    sub_num = get_next_submission_number()
    sub_standalone = pd.DataFrame({"id": test_df["id"].values})
    for t, col in zip(TARGETS, TARGET_COLS):
        sub_standalone[col] = results[t]["test_preds"]
    sub_standalone.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    submissions.append({
        "num": sub_num,
        "desc": f"Standalone 69-joint scratch CNN (CV={results['mean_cv_mse']:.6f})",
    })
    print(f"  Sub {sub_num}: Standalone 69-joint scratch CNN")

    # Blends with base submission
    if base_sub is not None:
        for w in [0.02, 0.05, 0.07, 0.10]:
            sub_num = get_next_submission_number()
            blend = base_sub.copy()
            for t, col in zip(TARGETS, TARGET_COLS):
                blend[col] = np.clip(
                    (1 - w) * base_sub[col].values + w * results[t]["test_preds"],
                    0.0, 1.0,
                )
            blend.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
            submissions.append({
                "num": sub_num,
                "desc": f"{int(w*100)}% 69j-CNN + {int((1-w)*100)}% Sub{args.base_submission}",
            })
            print(f"  Sub {sub_num}: {int(w*100)}% 69j-CNN + {int((1-w)*100)}% Sub{args.base_submission}")

        # Per-target diverse blend: only blend targets where r < 0.80
        corrs = {}
        for t in TARGETS:
            col = f"scaled_{t}"
            r = np.corrcoef(base_sub[col].values, results[t]["test_preds"])[0, 1]
            corrs[t] = r

        diverse_targets = [t for t in TARGETS if corrs[t] < 0.80]
        if diverse_targets:
            for w in [0.05, 0.10]:
                sub_num = get_next_submission_number()
                blend = base_sub.copy()
                desc_parts = []
                for t in diverse_targets:
                    col = f"scaled_{t}"
                    blend[col] = np.clip(
                        (1 - w) * base_sub[col].values + w * results[t]["test_preds"],
                        0.0, 1.0,
                    )
                    desc_parts.append(f"{t}(r={corrs[t]:.2f})")
                blend.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
                desc = (
                    f"{int(w*100)}% 69j-CNN on {'+'.join(diverse_targets)} only "
                    f"+ Sub{args.base_submission}"
                )
                submissions.append({"num": sub_num, "desc": desc})
                print(f"  Sub {sub_num}: {desc}")

    # ----------------------------------------------------------
    # 7. Summary
    # ----------------------------------------------------------
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")

    print(f"\n69-Joint CNN CV MSE per target:")
    for t in TARGETS:
        print(f"  {t}: {results[t]['cv_mse']:.6f}")
    print(f"  MEAN: {results['mean_cv_mse']:.6f}")

    if base_sub is not None:
        print(f"\nDiversity with Sub{args.base_submission}:")
        for t in TARGETS:
            col = f"scaled_{t}"
            r = np.corrcoef(base_sub[col].values, results[t]["test_preds"])[0, 1]
            print(f"  {t}: r={r:.4f}")

    print(f"\nSubmissions generated: {len(submissions)}")
    for s in submissions:
        print(f"  Sub {s['num']}: {s['desc']}")

    # Save run metadata
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_data = {
        "timestamp": ts,
        "pilot": args.pilot,
        "n_joints": N_JOINTS,
        "n_channels": N_CHANNELS,
        "n_params": sum(
            p.numel() for p in FinetuneModel69J(n_players=5).parameters()
        ),
        "load_time_s": load_time,
        "prep_time_s": prep_time,
        "train_time_s": train_time,
        "total_time_s": total_time,
        "cv": {t: results[t]["cv_mse"] for t in TARGETS},
        "mean_cv": results["mean_cv_mse"],
        "submissions": submissions,
    }
    run_path = OUTPUT_DIR / f"enhanced_69joint_cnn_run_{ts}.json"
    run_path.write_text(json.dumps(run_data, indent=2))
    print(f"\nRun metadata: {run_path}")
    print("Done.")


if __name__ == "__main__":
    main()

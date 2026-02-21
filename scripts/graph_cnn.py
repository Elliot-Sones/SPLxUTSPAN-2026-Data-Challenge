"""
Graph-Aware Position CNN for Basketball Shot Prediction

Innovation: Graph Diffusion preprocessing before the 1D CNN
- One step of graph diffusion: x_diffused = A @ x (neighbor-aggregated features)
- Concat original + diffused: 207 + 207 = 414 input channels
- Then standard 1D CNN on 414-channel input

Why this beats standard CNN:
- The CNN sees both raw joint positions AND neighbor-aggregated positions
- This encodes the skeleton structure: wrist position vs its average neighbor (elbow, fingers)
- The arm chain coordination is captured in the diffusion:
  elbow sees shoulder+wrist average, wrist sees elbow+fingertip average
- No MPS compatibility issues (just a matmul + concat in preprocessing)

Graph structure: 69-joint skeleton adjacency (same as planned ST-GCN)

Additional innovations:
- Training augmentation (noise + frame shift + joint dropout)
- TTA at test time (average 8 augmented predictions)
- SSL consistency loss on unlabeled test shots

Speed: Similar to position CNN (414 vs 207 channels, but same architecture)
Expected gain: 10-20% better than position CNN due to spatial structure capture
"""

from __future__ import annotations

import argparse
import fcntl
import json
import random
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
N_COORDS = 3                   # x, y, z
N_POS_CHANNELS = N_JOINTS * N_COORDS  # 207
N_INPUT_CHANNELS = N_POS_CHANNELS * 2  # 414 (pos + graph-diffused pos)

MID_HIP_IDX = ALL_JOINTS.index("mid_hip")
NECK_IDX = ALL_JOINTS.index("neck")

FRAME_START = 120
FRAME_END = 180
SUBSAMPLE = 2
N_FRAMES = (FRAME_END - FRAME_START) // SUBSAMPLE  # 30


# ================================================================
# SKELETON ADJACENCY
# ================================================================

def build_adjacency() -> np.ndarray:
    """Build skeleton adjacency matrix for 69 joints.

    Returns normalized (V, V) adjacency with self-loops.
    """
    V = N_JOINTS
    A = np.zeros((V, V), dtype=np.float32)

    def add_edge(j1: str, j2: str) -> None:
        if j1 not in ALL_JOINTS or j2 not in ALL_JOINTS:
            return
        i1, i2 = ALL_JOINTS.index(j1), ALL_JOINTS.index(j2)
        A[i1, i2] = 1.0
        A[i2, i1] = 1.0

    # Face
    add_edge("nose", "left_eye"); add_edge("nose", "right_eye")
    add_edge("left_eye", "left_ear"); add_edge("right_eye", "right_ear")

    # Spine
    add_edge("nose", "neck"); add_edge("neck", "mid_hip")

    # Arms
    add_edge("neck", "left_shoulder"); add_edge("left_shoulder", "left_elbow")
    add_edge("left_elbow", "left_wrist")
    add_edge("neck", "right_shoulder"); add_edge("right_shoulder", "right_elbow")
    add_edge("right_elbow", "right_wrist")

    # Left hand fingers
    for fb in ["left_first_finger_cmc", "left_second_finger_mcp",
                "left_third_finger_mcp", "left_fourth_finger_mcp",
                "left_fifth_finger_mcp"]:
        add_edge("left_wrist", fb)
    for pref in ["left_first_finger", "left_second_finger", "left_third_finger",
                  "left_fourth_finger", "left_fifth_finger"]:
        parts = sorted([n for n in ALL_JOINTS if n.startswith(pref)],
                       key=lambda x: ["cmc","mcp","pip","ip","dip","distal"].index(x.split("_")[-1])
                       if x.split("_")[-1] in ["cmc","mcp","pip","ip","dip","distal"] else 9)
        for k in range(len(parts) - 1):
            add_edge(parts[k], parts[k+1])
    add_edge("left_wrist", "left_thumb"); add_edge("left_wrist", "left_pinky")

    # Right hand fingers (THE SHOOTING HAND)
    for fb in ["right_first_finger_cmc", "right_second_finger_mcp",
                "right_third_finger_mcp", "right_fourth_finger_mcp",
                "right_fifth_finger_mcp"]:
        add_edge("right_wrist", fb)
    for pref in ["right_first_finger", "right_second_finger", "right_third_finger",
                  "right_fourth_finger", "right_fifth_finger"]:
        parts = sorted([n for n in ALL_JOINTS if n.startswith(pref)],
                       key=lambda x: ["cmc","mcp","pip","ip","dip","distal"].index(x.split("_")[-1])
                       if x.split("_")[-1] in ["cmc","mcp","pip","ip","dip","distal"] else 9)
        for k in range(len(parts) - 1):
            add_edge(parts[k], parts[k+1])
    add_edge("right_wrist", "right_thumb"); add_edge("right_wrist", "right_pinky")

    # Legs
    add_edge("mid_hip", "left_hip"); add_edge("left_hip", "left_knee")
    add_edge("left_knee", "left_ankle"); add_edge("left_ankle", "left_heel")
    add_edge("left_ankle", "left_big_toe"); add_edge("left_ankle", "left_small_toe")
    add_edge("mid_hip", "right_hip"); add_edge("right_hip", "right_knee")
    add_edge("right_knee", "right_ankle"); add_edge("right_ankle", "right_heel")
    add_edge("right_ankle", "right_big_toe"); add_edge("right_ankle", "right_small_toe")

    # Self-loops
    A = A + np.eye(V, dtype=np.float32)

    # Normalize
    d = A.sum(axis=1, keepdims=True)
    A_norm = A / np.maximum(d, 1e-6)

    return A_norm


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


def prepare_positions(seqs: np.ndarray) -> np.ndarray:
    """Hip-center, torso-normalize, extract 30-frame window.

    seqs: (N, 240, 69, 3)
    Returns: (N, 30, 207) position data
    """
    n = seqs.shape[0]
    all_pos = []
    for i in range(n):
        positions = seqs[i]
        hip = positions[:, MID_HIP_IDX:MID_HIP_IDX + 1, :]
        centered = positions - hip
        torso_vec = centered[:, NECK_IDX] - centered[:, MID_HIP_IDX]
        torso_len = np.linalg.norm(torso_vec, axis=1, keepdims=True)
        torso_len = np.maximum(torso_len, 1e-6)
        normed = centered / torso_len[:, np.newaxis, :]
        window = normed[FRAME_START:FRAME_END:SUBSAMPLE]  # (30, 69, 3)
        flat = window.reshape(N_FRAMES, N_JOINTS * 3)     # (30, 207)
        all_pos.append(flat)
    return np.array(all_pos, dtype=np.float32)


def apply_graph_diffusion(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Apply one step of graph diffusion: X_diff = X_flat @ A^T

    X: (N, 30, 207) where 207 = 69 joints * 3 coords
    A: (69, 69) normalized adjacency

    For each frame, reshape to (69, 3), multiply by A (neighbor aggregation),
    then flatten back to (207,).

    Returns: (N, 30, 207) graph-diffused positions
    """
    N, T, C = X.shape
    # Reshape to (N, T, 69, 3)
    X_joint = X.reshape(N, T, N_JOINTS, N_COORDS)
    # Apply adjacency: (N, T, 69, 3) @ A^T = (N, T, 69, 3)
    # For each (n, t): X_joint[n,t] is (69, 3), A @ X_joint[n,t] = (69, 3)
    X_diffused = np.einsum("ntvc,wv->ntwc", X_joint, A)  # (N, T, 69, 3)
    return X_diffused.reshape(N, T, N_JOINTS * N_COORDS)


# ================================================================
# AUGMENTATION
# ================================================================

def augment_batch(x: torch.Tensor, noise_sigma: float, frame_shift: int,
                  joint_dropout: float) -> torch.Tensor:
    """x: (B, C=414, T=30) - augment position channels only"""
    B, C, T = x.shape

    if noise_sigma > 0:
        # Add noise only to position channels (first 207), diffusion inherits noise
        noise = torch.randn_like(x[:, :N_POS_CHANNELS, :]) * noise_sigma
        x = x.clone()
        x[:, :N_POS_CHANNELS, :] = x[:, :N_POS_CHANNELS, :] + noise
        # Recompute diffusion? No - just add noise to both for simplicity
        x[:, N_POS_CHANNELS:, :] = x[:, N_POS_CHANNELS:, :] + noise

    if frame_shift > 0:
        shift = random.randint(-frame_shift, frame_shift)
        if shift > 0:
            x = torch.cat([x[:, :, shift:], x[:, :, -shift:]], dim=2)
        elif shift < 0:
            s = -shift
            x = torch.cat([x[:, :, :s], x[:, :, :-s]], dim=2)

    if joint_dropout > 0:
        n_drop = max(1, int(N_JOINTS * joint_dropout))
        drop_joints = random.sample(range(N_JOINTS), n_drop)
        mask = torch.ones(C, device=x.device)
        for j in drop_joints:
            mask[j*3:(j+1)*3] = 0.0
            mask[N_POS_CHANNELS + j*3:N_POS_CHANNELS + (j+1)*3] = 0.0
        x = x * mask.view(1, C, 1)

    return x


# ================================================================
# MODEL
# ================================================================

class GraphCNN(nn.Module):
    """1D CNN on graph-augmented position features.

    Input: (B, 414, 30) - original (207) + graph-diffused (207) positions
    """

    def __init__(self) -> None:
        super().__init__()
        C_in = N_INPUT_CHANNELS  # 414

        # Layer 1: process original + neighbor features jointly
        self.conv1 = nn.Conv1d(C_in, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(2)  # 30 -> 15

        # Layer 2
        self.conv2 = nn.Conv1d(64, 48, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(48)
        self.pool2 = nn.AvgPool1d(2)  # 15 -> 7

        # Layer 3
        self.conv3 = nn.Conv1d(48, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool3 = nn.AvgPool1d(2)  # 7 -> 3

        # Layer 4
        self.conv4 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(16)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x).squeeze(-1)
        return x


class GraphCNNModel(nn.Module):
    def __init__(self, n_players: int = 5) -> None:
        super().__init__()
        self.encoder = GraphCNN()
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
# ARGS AND HELPERS
# ================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph-aware CNN for basketball shot prediction")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--base-submission", type=int, default=3411)
    p.add_argument("--diversity-sub", type=int, default=2716)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--noise-sigma", type=float, default=0.02)
    p.add_argument("--frame-shift", type=int, default=3)
    p.add_argument("--joint-dropout", type=float, default=0.10)
    p.add_argument("--tta-n", type=int, default=8)
    p.add_argument("--ssl-weight", type=float, default=0.05)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "cpu": return torch.device("cpu")
    if name == "cuda": return torch.device("cuda")
    if name == "mps": return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    # MPS has issues with Metal compiler; default to CPU for stability
    # if torch.backends.mps.is_available(): return torch.device("mps")
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
# TRAINING
# ================================================================

def train_one_target(
    X_train: np.ndarray,         # (N, 30, 414) graph-diffused
    y_train: np.ndarray,
    pids_train: np.ndarray,
    X_test: np.ndarray,          # (M, 30, 414)
    pids_test: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    N = len(X_train)
    oof_preds = np.zeros(N)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Normalize
    X_mean = X_train.mean(axis=(0, 1))
    X_std = X_train.std(axis=(0, 1)) + 1e-8
    Xn_train = (X_train - X_mean) / X_std
    Xn_test = (X_test - X_mean) / X_std

    # (N, T, C) -> (N, C, T) for Conv1d
    X_tr = torch.from_numpy(Xn_train.transpose(0, 2, 1)).float()
    X_te = torch.from_numpy(Xn_test.transpose(0, 2, 1)).float()
    y_tr = torch.from_numpy(y_train).float()
    pid_tr = torch.from_numpy(pids_train).long()
    pid_te = torch.from_numpy(pids_test).long()

    all_test_preds = []

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(N))):
        model = GraphCNNModel(n_players=5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)

        X_ftr = X_tr[tr_idx].to(device)
        y_ftr = y_tr[tr_idx].to(device)
        pid_ftr = pid_tr[tr_idx].to(device)
        X_fval = X_tr[val_idx].to(device)
        y_fval = y_tr[val_idx].to(device)
        pid_fval = pid_tr[val_idx].to(device)

        best_val = float("inf")
        best_state = None
        pat = 0

        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(tr_idx))
            for start in range(0, len(tr_idx), args.batch_size):
                idx = perm[start:start + args.batch_size]
                xb = augment_batch(X_ftr[idx].clone(), args.noise_sigma,
                                   args.frame_shift, args.joint_dropout)
                yb = y_ftr[idx]
                pb = pid_ftr[idx]

                optimizer.zero_grad()
                pred = model(xb, pb)
                loss = F.mse_loss(pred, yb)

                # SSL consistency on test shots
                if args.ssl_weight > 0:
                    ssl_idx = torch.randint(0, len(X_te),
                                            (min(len(idx), len(X_te)),))
                    xssl = X_te[ssl_idx].to(device)
                    pssl = pid_te[ssl_idx].to(device)
                    xa = augment_batch(xssl.clone(), args.noise_sigma,
                                      args.frame_shift, args.joint_dropout)
                    xb2 = augment_batch(xssl.clone(), args.noise_sigma,
                                        args.frame_shift, args.joint_dropout)
                    with torch.no_grad():
                        pa = model(xa, pssl)
                    pb2 = model(xb2, pssl)
                    loss = loss + args.ssl_weight * F.mse_loss(pb2, pa.detach())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            model.eval()
            with torch.no_grad():
                vp = model(X_fval, pid_fval)
                vl = F.mse_loss(vp, y_fval).item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= args.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        model.to(device)

        with torch.no_grad():
            oof_preds[val_idx] = model(X_fval, pid_fval).cpu().numpy()

        # TTA for test
        fold_test = []
        with torch.no_grad():
            fold_test.append(model(X_te.to(device), pid_te.to(device)).cpu().numpy())
            for _ in range(args.tta_n - 1):
                xa = augment_batch(X_te.clone().to(device), args.noise_sigma,
                                   args.frame_shift, args.joint_dropout)
                fold_test.append(model(xa, pid_te.to(device)).cpu().numpy())
        all_test_preds.append(np.mean(fold_test, axis=0))

    return oof_preds, np.mean(all_test_preds, axis=0)


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = pick_device(args.device)
    print(f"Device: {device}")
    print("Graph-Aware CNN: pos(207) + graph-diffused-pos(207) = 414 channels")

    if args.pilot:
        args.epochs = 80
        args.n_seeds = 1
        args.tta_n = 3
        print("PILOT: 80 epochs, 1 seed")

    t0 = time.time()

    # Build adjacency
    print("Building skeleton adjacency ...")
    A_np = build_adjacency()
    print(f"  Adjacency {A_np.shape}, sparsity={(A_np == 0).mean():.2%}")

    # Count params
    tmp = GraphCNNModel()
    n_params = sum(p.numel() for p in tmp.parameters())
    print(f"  Graph CNN params: {n_params:,}")
    del tmp

    # Load data
    print("Loading data ...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print("Parsing sequences ...")
    train_seqs = load_all_69j_sequences(train_df)
    test_seqs = load_all_69j_sequences(test_df)

    print("Preparing positions ...")
    X_pos_train = prepare_positions(train_seqs)  # (345, 30, 207)
    X_pos_test = prepare_positions(test_seqs)    # (113, 30, 207)

    print("Applying graph diffusion ...")
    X_diff_train = apply_graph_diffusion(X_pos_train, A_np)  # (345, 30, 207)
    X_diff_test = apply_graph_diffusion(X_pos_test, A_np)

    # Concatenate: original + diffused = 414 channels
    X_train = np.concatenate([X_pos_train, X_diff_train], axis=2)  # (345, 30, 414)
    X_test = np.concatenate([X_pos_test, X_diff_test], axis=2)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Player IDs
    unique_pids = sorted(train_df["participant_id"].unique())
    player_id_map = {pid: i for i, pid in enumerate(unique_pids)}
    pids_train = np.array([player_id_map[p] for p in train_df["participant_id"].values])
    pids_test = np.array([player_id_map.get(p, 0) for p in test_df["participant_id"].values])

    # Targets
    import joblib
    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    y_train = {}
    for t in TARGETS:
        y_train[t] = scalers[t].transform(
            train_df[t].values.reshape(-1, 1)
        ).ravel().astype(np.float32)

    seeds = [42, 7, 2026, 99, 133][:args.n_seeds]
    print(f"\nTraining {args.n_seeds} seeds x 3 targets x 5 folds ...")

    all_oof = {t: [] for t in TARGETS}
    all_test = {t: [] for t in TARGETS}
    cv_per_seed = {}

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        seed_cv = {}
        for target in TARGETS:
            t1 = time.time()
            oof, tp = train_one_target(
                X_train, y_train[target], pids_train,
                X_test, pids_test,
                args, device, seed,
            )
            mse = float(np.mean((oof - y_train[target]) ** 2))
            seed_cv[target] = mse
            all_oof[target].append(oof)
            all_test[target].append(tp)
            print(f"  {target}: CV={mse:.6f} | {time.time()-t1:.1f}s")
        mean_cv = np.mean(list(seed_cv.values()))
        print(f"  Mean: {mean_cv:.6f}")
        cv_per_seed[seed] = seed_cv

    # Ensemble
    ens_oof = {t: np.mean(all_oof[t], axis=0) for t in TARGETS}
    ens_test = {t: np.mean(all_test[t], axis=0) for t in TARGETS}
    cv_scores = {t: float(np.mean((ens_oof[t] - y_train[t]) ** 2)) for t in TARGETS}
    mean_cv = np.mean(list(cv_scores.values()))
    print(f"\nEnsemble CV: angle={cv_scores['angle']:.6f} "
          f"depth={cv_scores['depth']:.6f} LR={cv_scores['left_right']:.6f} "
          f"mean={mean_cv:.6f}")

    # Diversity
    base_df = pd.read_csv(SUBMISSION_DIR / f"submission_{args.base_submission}.csv")
    print(f"\nDiversity vs Sub{args.base_submission}:")
    for t in TARGETS:
        col = f"scaled_{t}"
        if col in base_df.columns:
            r, _ = pearsonr(ens_test[t], base_df[col].values)
            print(f"  {t}: r={r:.4f}")

    div_path = SUBMISSION_DIR / f"submission_{args.diversity_sub}.csv"
    if div_path.exists():
        div_df = pd.read_csv(div_path)
        print(f"Diversity vs Sub{args.diversity_sub}:")
        for t in TARGETS:
            col = f"scaled_{t}"
            if col in div_df.columns:
                r, _ = pearsonr(ens_test[t], div_df[col].values)
                print(f"  {t}: r={r:.4f}")

    # Submissions
    base_full = pd.read_csv(SUBMISSION_DIR / f"submission_{args.base_submission}.csv")
    blend_weights = [0.03, 0.05, 0.08, 0.10] if not args.pilot else [0.05]
    submissions = []

    # Standalone
    snum = get_next_submission_number()
    sa = pd.DataFrame({"id": test_df["id"].values})
    for t in TARGETS:
        sa[f"scaled_{t}"] = np.clip(ens_test[t], 0.0, 1.0)
    sa.to_csv(SUBMISSION_DIR / f"submission_{snum}.csv", index=False)
    submissions.append({"num": snum, "desc": f"Standalone graph-CNN (CV={mean_cv:.6f})"})
    print(f"\nSub {snum}: standalone graph-CNN (CV={mean_cv:.6f})")

    for w in blend_weights:
        bnum = get_next_submission_number()
        sb = base_full.copy()
        for t in TARGETS:
            col = f"scaled_{t}"
            if col in sb.columns:
                sb[col] = np.clip(w * ens_test[t] + (1-w) * sb[col].values, 0.0, 1.0)
        sb.to_csv(SUBMISSION_DIR / f"submission_{bnum}.csv", index=False)
        desc = f"{int(w*100)}% graph-CNN + {int((1-w)*100)}% Sub{args.base_submission}"
        submissions.append({"num": bnum, "desc": desc})
        print(f"Sub {bnum}: {desc}")

    result = {
        "timestamp": ts, "pilot": args.pilot, "model": "graph_cnn",
        "n_params": n_params, "n_frames": N_FRAMES, "n_channels": N_INPUT_CHANNELS,
        "noise_sigma": args.noise_sigma, "frame_shift": args.frame_shift,
        "joint_dropout": args.joint_dropout, "tta_n": args.tta_n,
        "ssl_weight": args.ssl_weight, "epochs": args.epochs, "n_seeds": args.n_seeds,
        "cv_scores": cv_scores, "mean_cv": mean_cv,
        "per_seed_cv": {str(s): cv_per_seed[s] for s in seeds},
        "total_time_s": time.time() - t0, "submissions": submissions,
    }
    out_path = OUTPUT_DIR / f"graph_cnn_run_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults: {out_path}")
    print(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

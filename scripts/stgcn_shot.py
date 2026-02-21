"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) for Basketball Shot Prediction

Based on: "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition"
(Yan et al., AAAI 2018) https://arxiv.org/abs/1801.07455

Why this is the RIGHT architecture for this problem:
- Standard 1D CNN treats all 207 channels independently (no spatial structure)
- ST-GCN explicitly models the SKELETON GRAPH: adjacent joints are connected
- Spatial GCN captures how the arm chain coordinates (shoulder->elbow->wrist->fingertip)
- This captures the kinematic chain that determines release angle and depth

Key innovation: The skeleton adjacency tells the model WHICH joints are physically connected.
For basketball shooting, the critical chain is:
  neck -> right_shoulder -> right_elbow -> right_wrist -> right_fingers
This chain determines the ball release velocity, which determines angle/depth/LR.

Architecture:
  Input: (B, T, V, C_in) = (batch, 60frames, 69joints, 3coords)
  -> ST-GCN block 1: spatial GCN (neighbor aggregation) + temporal 1D CNN
  -> ST-GCN block 2: another ST-GCN block with more channels
  -> Global temporal pool + player embed -> per-target prediction head

Expected improvement over position CNN:
- ST-GCN original paper: 5-20% improvement over flat CNN for skeleton actions
- For shooting (fine-grained motion), spatial structure should matter more
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
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",         # 0-4
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",   # 5-8
    "left_wrist", "right_wrist", "left_hip", "right_hip",             # 9-12
    "left_knee", "right_knee", "left_ankle", "right_ankle",           # 13-16
    "left_big_toe", "left_small_toe", "left_heel",                    # 17-19
    "right_big_toe", "right_small_toe", "right_heel",                 # 20-22
    "left_first_finger_cmc", "left_first_finger_mcp",                 # 23-24
    "left_first_finger_ip", "left_first_finger_distal",               # 25-26
    "left_second_finger_mcp", "left_second_finger_pip",               # 27-28
    "left_second_finger_dip", "left_second_finger_distal",            # 29-30
    "left_third_finger_mcp", "left_third_finger_pip",                 # 31-32
    "left_third_finger_dip", "left_third_finger_distal",              # 33-34
    "left_fourth_finger_mcp", "left_fourth_finger_pip",               # 35-36
    "left_fourth_finger_dip", "left_fourth_finger_distal",            # 37-38
    "left_fifth_finger_mcp", "left_fifth_finger_pip",                 # 39-40
    "left_fifth_finger_dip", "left_fifth_finger_distal",              # 41-42
    "right_first_finger_cmc", "right_first_finger_mcp",               # 43-44
    "right_first_finger_ip", "right_first_finger_distal",             # 45-46
    "right_second_finger_mcp", "right_second_finger_pip",             # 47-48
    "right_second_finger_dip", "right_second_finger_distal",          # 49-50
    "right_third_finger_mcp", "right_third_finger_pip",               # 51-52
    "right_third_finger_dip", "right_third_finger_distal",            # 53-54
    "right_fourth_finger_mcp", "right_fourth_finger_pip",             # 55-56
    "right_fourth_finger_dip", "right_fourth_finger_distal",          # 57-58
    "right_fifth_finger_mcp", "right_fifth_finger_pip",               # 59-60
    "right_fifth_finger_dip", "right_fifth_finger_distal",            # 61-62
    "left_thumb", "left_pinky", "right_thumb", "right_pinky",         # 63-66
    "mid_hip", "neck",                                                 # 67-68
]

N_JOINTS = len(ALL_JOINTS)   # 69
MID_HIP_IDX = ALL_JOINTS.index("mid_hip")   # 67
NECK_IDX = ALL_JOINTS.index("neck")         # 68

# Frame window: 120-180 at 60fps = 60 frames
FRAME_START = 120
FRAME_END = 180
SUBSAMPLE = 1
N_FRAMES = (FRAME_END - FRAME_START) // SUBSAMPLE  # 60


def build_adjacency() -> np.ndarray:
    """Build skeleton adjacency matrix for 69 joints.

    Returns: (V, V) binary adjacency (self-loops included via identity).
    The edges follow the human body skeleton structure.
    """
    V = N_JOINTS
    A = np.zeros((V, V), dtype=np.float32)

    def add_edge(j1: str, j2: str) -> None:
        i1, i2 = ALL_JOINTS.index(j1), ALL_JOINTS.index(j2)
        A[i1, i2] = 1.0
        A[i2, i1] = 1.0

    # ----- Core skeleton connections -----
    # Face
    add_edge("nose", "left_eye")
    add_edge("nose", "right_eye")
    add_edge("left_eye", "left_ear")
    add_edge("right_eye", "right_ear")

    # Spine
    add_edge("nose", "neck")
    add_edge("neck", "mid_hip")

    # Left arm
    add_edge("neck", "left_shoulder")
    add_edge("left_shoulder", "left_elbow")
    add_edge("left_elbow", "left_wrist")

    # Right arm (the shooting arm - most important chain!)
    add_edge("neck", "right_shoulder")
    add_edge("right_shoulder", "right_elbow")
    add_edge("right_elbow", "right_wrist")

    # Left hand fingers
    for finger_base in ["left_first_finger_cmc", "left_second_finger_mcp",
                         "left_third_finger_mcp", "left_fourth_finger_mcp",
                         "left_fifth_finger_mcp"]:
        add_edge("left_wrist", finger_base)

    # Left finger chains
    for prefix in ["left_first_finger", "left_second_finger", "left_third_finger",
                    "left_fourth_finger", "left_fifth_finger"]:
        joints = [n for n in ALL_JOINTS if n.startswith(prefix)]
        joints.sort(key=lambda x: ["cmc", "mcp", "pip", "ip", "dip", "distal"].index(x.split("_")[-1])
                    if x.split("_")[-1] in ["cmc", "mcp", "pip", "ip", "dip", "distal"] else 99)
        for k in range(len(joints) - 1):
            add_edge(joints[k], joints[k + 1])

    # Right hand (THE SHOOTING HAND)
    for finger_base in ["right_first_finger_cmc", "right_second_finger_mcp",
                         "right_third_finger_mcp", "right_fourth_finger_mcp",
                         "right_fifth_finger_mcp"]:
        add_edge("right_wrist", finger_base)

    # Right finger chains
    for prefix in ["right_first_finger", "right_second_finger", "right_third_finger",
                    "right_fourth_finger", "right_fifth_finger"]:
        joints = [n for n in ALL_JOINTS if n.startswith(prefix)]
        joints.sort(key=lambda x: ["cmc", "mcp", "pip", "ip", "dip", "distal"].index(x.split("_")[-1])
                    if x.split("_")[-1] in ["cmc", "mcp", "pip", "ip", "dip", "distal"] else 99)
        for k in range(len(joints) - 1):
            add_edge(joints[k], joints[k + 1])

    # Thumb and pinky
    add_edge("left_wrist", "left_thumb")
    add_edge("left_wrist", "left_pinky")
    add_edge("right_wrist", "right_thumb")
    add_edge("right_wrist", "right_pinky")

    # Hips and legs
    add_edge("mid_hip", "left_hip")
    add_edge("mid_hip", "right_hip")
    add_edge("left_hip", "left_knee")
    add_edge("left_knee", "left_ankle")
    add_edge("left_ankle", "left_heel")
    add_edge("left_ankle", "left_big_toe")
    add_edge("left_ankle", "left_small_toe")
    add_edge("right_hip", "right_knee")
    add_edge("right_knee", "right_ankle")
    add_edge("right_ankle", "right_heel")
    add_edge("right_ankle", "right_big_toe")
    add_edge("right_ankle", "right_small_toe")

    # Self-loops
    A = A + np.eye(V, dtype=np.float32)

    # Normalize: D^{-1/2} A D^{-1/2}
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-6)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return A_norm


# ================================================================
# ST-GCN LAYERS
# ================================================================

class STGCNBlock(nn.Module):
    """One Spatial-Temporal GCN block.

    Spatial part: Graph convolution over joints (each frame independently)
    Temporal part: 1D convolution over frames (each joint independently)

    Input: (B, C_in, T, V) = (batch, channels, frames, joints)
    Output: (B, C_out, T', V)
    """

    def __init__(self, c_in: int, c_out: int, A: torch.Tensor,
                 t_kernel: int = 5, t_stride: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.register_buffer("A", A)

        # Spatial graph convolution: (B, C_in, T, V) -> (B, C_out, T, V)
        self.gc_weight = nn.Parameter(torch.randn(c_in, c_out) * 0.01)
        self.gc_bn = nn.BatchNorm2d(c_out)

        # Temporal convolution: (B, C_out, T, V) -> (B, C_out, T', V)
        padding = (t_kernel - 1) // 2
        self.tc = nn.Conv2d(c_out, c_out, kernel_size=(t_kernel, 1),
                            stride=(t_stride, 1), padding=(padding, 0))
        self.tc_bn = nn.BatchNorm2d(c_out)
        self.dropout = nn.Dropout(dropout)

        # Residual
        if c_in != c_out or t_stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=(t_stride, 1)),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, T, V) -> (B, C_out, T', V)

        Spatial GCN using matmul to ensure MPS/CUDA compatibility:
          1. Neighbor aggregation: x @ A^T -> (B, C, T, V)
          2. Linear channel transform via Conv2d(1x1)
        """
        res = self.residual(x)

        # Step 1: Graph neighbor aggregation over joint dimension V
        # x: (B, C, T, V), A: (V, V)
        # x @ A^T means: for each (b,c,t), output[v] = sum_u x[u] * A[u,v]
        # This aggregates neighbor features (A is normalized adjacency)
        x_agg = torch.matmul(x, self.A.T).contiguous()  # (B, C_in, T, V)

        # Step 2: Linear channel transform: C_in -> C_out
        # x_agg: (B, C_in, T, V), gc_weight: (C_in, C_out)
        # Rearrange to apply linear: (B, T, V, C_in) @ (C_in, C_out) -> (B, T, V, C_out)
        B, C, T, V = x_agg.shape
        x_2d = x_agg.permute(0, 2, 3, 1).contiguous()  # (B, T, V, C_in)
        x_2d = x_2d.reshape(B * T * V, C)               # (B*T*V, C_in)
        x_2d = x_2d @ self.gc_weight                     # (B*T*V, C_out)
        C_out = self.gc_weight.shape[1]
        x_sp = x_2d.reshape(B, T, V, C_out).permute(0, 3, 1, 2).contiguous()  # (B, C_out, T, V)
        x_sp = F.relu(self.gc_bn(x_sp))

        # Temporal convolution
        x_tc = self.tc(x_sp)       # (B, C_out, T', V)
        x_tc = self.tc_bn(x_tc)
        x_tc = self.dropout(x_tc)

        return F.relu(x_tc + res)


class STGCNModel(nn.Module):
    """Full ST-GCN model for shot prediction.

    Input: (B, T, V, C_in=3) -> (B,) predictions for one target
    """

    def __init__(self, A: torch.Tensor, n_players: int = 5, dropout: float = 0.2) -> None:
        super().__init__()

        # Initial projection: 3 -> 16 spatial channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # ST-GCN blocks: progressively reduce temporal dimension
        # Block 1: 16 channels, no temporal stride
        self.block1 = STGCNBlock(16, 32, A, t_kernel=5, t_stride=1, dropout=dropout)
        # Block 2: 32 channels, temporal stride=2 (60->30)
        self.block2 = STGCNBlock(32, 32, A, t_kernel=5, t_stride=2, dropout=dropout)
        # Block 3: 32 channels, temporal stride=2 (30->15)
        self.block3 = STGCNBlock(32, 16, A, t_kernel=3, t_stride=2, dropout=dropout)

        # Global pooling over time and joints -> embedding
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 16, 1, 1)

        # Player embedding + head
        self.player_embed = nn.Embedding(n_players + 1, 4)
        self.fc1 = nn.Linear(16 + 4, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        """x: (B, T, V, 3), player_id: (B,) -> (B,)"""
        B, T, V, C = x.shape

        # Rearrange to (B, C_in, T, V) for Conv2d
        x = x.permute(0, 3, 1, 2)  # (B, 3, T, V)

        # Input projection
        x = self.input_proj(x)      # (B, 16, T, V)

        # ST-GCN blocks
        x = self.block1(x)          # (B, 32, T, V)
        x = self.block2(x)          # (B, 32, T/2, V)
        x = self.block3(x)          # (B, 16, T/4, V)

        # Global pool
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # (B, 16)

        # Player embed
        pe = self.player_embed(player_id)
        h = torch.cat([x, pe], dim=1)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1)


# ================================================================
# DATA LOADING (reuse from position_69j_cnn.py logic)
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


def prepare_stgcn_input(seqs: np.ndarray) -> np.ndarray:
    """Hip-center, torso-normalize, extract 60-frame window.

    seqs: (N, 240, 69, 3)
    Returns: (N, 60, 69, 3) - (samples, time, joints, coords)
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
        window = normed[FRAME_START:FRAME_END:SUBSAMPLE]  # (60, 69, 3)
        all_pos.append(window)
    return np.array(all_pos, dtype=np.float32)


def augment_stgcn(x: torch.Tensor, noise_sigma: float = 0.02,
                   frame_shift: int = 3) -> torch.Tensor:
    """Augment ST-GCN input. x: (B, T, V, 3)"""
    B, T, V, C = x.shape
    if noise_sigma > 0:
        x = x + torch.randn_like(x) * noise_sigma
    if frame_shift > 0:
        shift = random.randint(-frame_shift, frame_shift)
        if shift > 0:
            x = torch.cat([x[:, shift:, :, :], x[:, -shift:, :, :]], dim=1)
        elif shift < 0:
            s = -shift
            x = torch.cat([x[:, :s, :, :], x[:, :-s, :, :]], dim=1)
    return x


# ================================================================
# ARGUMENT PARSING
# ================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ST-GCN for basketball shot prediction")
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
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--noise-sigma", type=float, default=0.02)
    p.add_argument("--frame-shift", type=int, default=3)
    p.add_argument("--tta-n", type=int, default=5)
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
    if torch.backends.mps.is_available(): return torch.device("mps")
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

def train_one_target_stgcn(
    X_train: np.ndarray,          # (N, 60, 69, 3)
    y_train: np.ndarray,
    player_ids_train: np.ndarray,
    X_test: np.ndarray,           # (M, 60, 69, 3)
    player_ids_test: np.ndarray,
    A_tensor: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    N = len(X_train)
    oof_preds = np.zeros(N)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Normalize per-coordinate across training set
    X_mean = X_train.mean(axis=(0, 1, 2))   # (3,)
    X_std = X_train.std(axis=(0, 1, 2)) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    X_tr_t = torch.from_numpy(X_train_norm).float()   # (N, 60, 69, 3)
    X_te_t = torch.from_numpy(X_test_norm).float()    # (M, 60, 69, 3)
    y_tr_t = torch.from_numpy(y_train).float()
    pid_tr = torch.from_numpy(player_ids_train).long()
    pid_te = torch.from_numpy(player_ids_test).long()

    all_test_preds = []

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(N))):
        model = STGCNModel(A=A_tensor.to(device), n_players=5, dropout=args.dropout).to(device)
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
            perm = torch.randperm(len(tr_idx))

            for start in range(0, len(tr_idx), args.batch_size):
                idx = perm[start:start + args.batch_size]
                xb = X_fold_tr[idx]
                yb = y_fold_tr[idx]
                pb = pid_fold_tr[idx]

                xb_aug = augment_stgcn(xb.clone(), args.noise_sigma, args.frame_shift)

                optimizer.zero_grad()
                pred = model(xb_aug, pb)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

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

        model.load_state_dict(best_state)
        model.eval()
        model.to(device)

        with torch.no_grad():
            oof_pred = model(X_fold_val.to(device), pid_fold_val.to(device))
        oof_preds[val_idx] = oof_pred.cpu().numpy()

        # TTA for test predictions
        test_preds_fold = []
        with torch.no_grad():
            base = model(X_te_t.to(device), pid_te.to(device)).cpu().numpy()
            test_preds_fold.append(base)
            for _ in range(args.tta_n - 1):
                xa = augment_stgcn(X_te_t.clone().to(device), args.noise_sigma, args.frame_shift)
                ap = model(xa, pid_te.to(device)).cpu().numpy()
                test_preds_fold.append(ap)
        all_test_preds.append(np.mean(test_preds_fold, axis=0))

    return oof_preds, np.mean(all_test_preds, axis=0)


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = pick_device(args.device)
    print(f"Device: {device}")

    if args.pilot:
        args.epochs = 60
        args.n_seeds = 1
        args.tta_n = 3
        print("PILOT: 60 epochs, 1 seed")

    # Build adjacency matrix
    print("Building skeleton adjacency matrix for 69 joints ...")
    A_np = build_adjacency()
    A_tensor = torch.from_numpy(A_np).float()
    print(f"Adjacency: {A_np.shape}, sparsity={1-(A_np > 0).sum()/(69*69):.2%}")

    t0 = time.time()

    # Count parameters
    tmp_model = STGCNModel(A_tensor, n_players=5)
    n_params = sum(p.numel() for p in tmp_model.parameters())
    print(f"ST-GCN params: {n_params:,}")
    del tmp_model

    # Load data
    print("Loading data ...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print("Parsing sequences ...")
    train_seqs = load_all_69j_sequences(train_df)
    test_seqs = load_all_69j_sequences(test_df)

    print("Preparing ST-GCN input ...")
    X_train = prepare_stgcn_input(train_seqs)   # (345, 60, 69, 3)
    X_test = prepare_stgcn_input(test_seqs)     # (113, 60, 69, 3)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    unique_pids = sorted(train_df["participant_id"].unique())
    player_id_map = {pid: i for i, pid in enumerate(unique_pids)}
    player_ids_train = np.array([player_id_map[p] for p in train_df["participant_id"].values])
    player_ids_test = np.array([player_id_map.get(p, 0) for p in test_df["participant_id"].values])

    import joblib
    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    y_train = {}
    for t in TARGETS:
        y_train[t] = scalers[t].transform(
            train_df[t].values.reshape(-1, 1)
        ).ravel().astype(np.float32)

    seeds = [42, 7, 2026, 99, 133][:args.n_seeds]
    print(f"\nTraining {args.n_seeds} seeds x 3 targets x 5 folds")

    all_oof = {t: [] for t in TARGETS}
    all_test = {t: [] for t in TARGETS}
    cv_per_seed = {}

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        seed_cv = {}
        for target in TARGETS:
            t1 = time.time()
            oof, test_pred = train_one_target_stgcn(
                X_train, y_train[target], player_ids_train,
                X_test, player_ids_test,
                A_tensor, args, device, seed,
            )
            mse = np.mean((oof - y_train[target]) ** 2)
            seed_cv[target] = mse
            all_oof[target].append(oof)
            all_test[target].append(test_pred)
            print(f"  {target}: CV={mse:.6f} | {time.time()-t1:.1f}s")
        mean_cv = np.mean(list(seed_cv.values()))
        print(f"  Mean CV: {mean_cv:.6f}")
        cv_per_seed[seed] = seed_cv

    # Ensemble
    ens_oof = {t: np.mean(all_oof[t], axis=0) for t in TARGETS}
    ens_test = {t: np.mean(all_test[t], axis=0) for t in TARGETS}
    cv_scores = {t: float(np.mean((ens_oof[t] - y_train[t]) ** 2)) for t in TARGETS}
    mean_cv = np.mean(list(cv_scores.values()))
    print(f"\nEnsemble CV: {mean_cv:.6f} "
          f"(angle={cv_scores['angle']:.6f} depth={cv_scores['depth']:.6f} "
          f"LR={cv_scores['left_right']:.6f})")

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

    # Generate submissions
    base_df_full = pd.read_csv(SUBMISSION_DIR / f"submission_{args.base_submission}.csv")

    blend_weights = [0.03, 0.05, 0.08, 0.10] if not args.pilot else [0.05]
    submissions = []

    standalone_num = get_next_submission_number()
    sa = pd.DataFrame({"id": test_df["id"].values})
    for t in TARGETS:
        sa[f"scaled_{t}"] = np.clip(ens_test[t], 0.0, 1.0)
    sa.to_csv(SUBMISSION_DIR / f"submission_{standalone_num}.csv", index=False)
    submissions.append({"num": standalone_num,
                        "desc": f"Standalone ST-GCN (CV={mean_cv:.6f})"})
    print(f"\nSub {standalone_num}: standalone ST-GCN (CV={mean_cv:.6f})")

    for w in blend_weights:
        num = get_next_submission_number()
        sub = base_df_full.copy()
        for t in TARGETS:
            col = f"scaled_{t}"
            if col in sub.columns:
                sub[col] = np.clip(
                    w * ens_test[t] + (1 - w) * sub[col].values,
                    0.0, 1.0
                )
        sub.to_csv(SUBMISSION_DIR / f"submission_{num}.csv", index=False)
        desc = f"{int(w*100)}% ST-GCN + {int((1-w)*100)}% Sub{args.base_submission}"
        submissions.append({"num": num, "desc": desc})
        print(f"Sub {num}: {desc}")

    result = {
        "timestamp": ts,
        "pilot": args.pilot,
        "model": "stgcn",
        "n_params": n_params,
        "n_frames": N_FRAMES,
        "epochs": args.epochs,
        "n_seeds": args.n_seeds,
        "cv_scores": cv_scores,
        "mean_cv": mean_cv,
        "per_seed_cv": {str(s): cv_per_seed[s] for s in seeds},
        "total_time_s": time.time() - t0,
        "submissions": submissions,
    }
    out_path = OUTPUT_DIR / f"stgcn_shot_run_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

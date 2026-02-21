"""
Improved CNN for Basketball Free Throw Prediction

Key improvements over existing CNNs:
1. Multi-scale temporal convolutions (k=3, k=7, k=15) in parallel branches
2. Temporal attention mechanism to learn which frames matter per target
3. Combined position + velocity + acceleration as multi-channel input
4. Multi-task learning: shared backbone predicts all 3 targets jointly
5. Residual connections for better gradient flow
6. LOO cross-validation matching the Ridge baseline evaluation
7. Per-target frame windows (angle=153, depth=150, LR=170) centered

Architecture:
  Input: (B, C_in, T) where C_in = pos_channels + vel_channels + acc_channels
  -> Multi-scale conv block (3 parallel branches with k=3/7/15)
  -> Temporal attention (self-attention over time dimension)
  -> Residual conv blocks x2
  -> Global pool + player embed -> per-target prediction heads

Goal: CV < 0.006, diversity r < 0.75 vs Sub 2716 on depth
"""

from __future__ import annotations

import argparse
import copy
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
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
# Release frame centers per target
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

# All 69 joints
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
N_POS_CHANNELS = N_JOINTS * 3  # 207

MID_HIP_IDX = ALL_JOINTS.index("mid_hip")
NECK_IDX = ALL_JOINTS.index("neck")

# Frame window: centered around release with some padding
FRAME_HALF_WINDOW = 30  # +/- 30 frames around target frame (1 second each side)
SUBSAMPLE = 2  # 60fps -> 30fps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Improved multi-scale attention CNN")
    p.add_argument("--pilot", action="store_true", help="Quick pilot run (100ep, 1 seed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--base-submission", type=int, default=3411)
    p.add_argument("--diversity-submission", type=int, default=2716,
                   help="Submission to check diversity against")
    # Architecture
    p.add_argument("--hidden-dim", type=int, default=48,
                   help="Hidden channel dimension in conv layers")
    p.add_argument("--embed-dim", type=int, default=24,
                   help="Final embedding dimension before heads")
    p.add_argument("--n-attn-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.25)
    # Training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=5e-3)
    # Blend weights to generate
    p.add_argument("--blend-weights", type=str, default="0.03,0.05,0.08,0.10")
    # Ablation flags
    p.add_argument("--no-velocity", action="store_true", help="Positions only")
    p.add_argument("--no-accel", action="store_true", help="No acceleration channel")
    p.add_argument("--no-attention", action="store_true", help="No temporal attention")
    p.add_argument("--single-task", action="store_true", help="Single-task instead of multi-task")
    p.add_argument("--target", type=str, default=None,
                   help="Single target to train (for single-task mode)")
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
    s = str(s).replace("nan", "null").replace("NaN", "null")
    arr = json.loads(s)
    out = np.zeros(240, dtype=float)
    for i in range(min(len(arr), 240)):
        v = arr[i]
        if v is not None:
            out[i] = float(v)
    return out


def load_all_69j_sequences(df: pd.DataFrame) -> np.ndarray:
    """Load ALL 69-joint sequences. Returns (N, 240, 69, 3)."""
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


def prepare_multi_channel_input(
    seqs: np.ndarray,  # (N, 240, 69, 3)
    target_frame: int,
    half_window: int = FRAME_HALF_WINDOW,
    subsample: int = SUBSAMPLE,
    include_velocity: bool = True,
    include_accel: bool = True,
) -> np.ndarray:
    """Create multi-channel input: position + velocity + acceleration.

    Args:
        seqs: (N, 240, 69, 3) raw positions
        target_frame: center frame for the window
        half_window: frames each side of center
        subsample: subsample factor

    Returns:
        (N, T, C) where C = 207 (pos) + 207 (vel) + 207 (acc) = up to 621
    """
    n = seqs.shape[0]

    # Frame window (clamped to valid range)
    f_start = max(0, target_frame - half_window)
    f_end = min(240, target_frame + half_window)

    all_features = []
    for i in range(n):
        positions = seqs[i]  # (240, 69, 3)

        # Center on mid_hip
        hip = positions[:, MID_HIP_IDX:MID_HIP_IDX + 1, :]
        centered = positions - hip

        # Scale by torso length
        torso_vec = centered[:, NECK_IDX] - centered[:, MID_HIP_IDX]
        torso_len = np.linalg.norm(torso_vec, axis=1, keepdims=True)
        torso_len = np.maximum(torso_len, 1e-6)
        normed = centered / torso_len[:, np.newaxis, :]

        # Extract window and subsample
        window = normed[f_start:f_end:subsample]  # (T, 69, 3)
        T = window.shape[0]

        # Flatten position channels: (T, 207)
        pos_flat = window.reshape(T, N_POS_CHANNELS)

        channels = [pos_flat]

        if include_velocity:
            # Velocity: frame-to-frame diff, pad first frame with zeros
            vel = np.zeros_like(pos_flat)
            vel[1:] = pos_flat[1:] - pos_flat[:-1]
            channels.append(vel)

        if include_accel:
            # Acceleration: second derivative, pad first 2 frames
            acc = np.zeros_like(pos_flat)
            if T > 2:
                acc[2:] = pos_flat[2:] - 2 * pos_flat[1:-1] + pos_flat[:-2]
            channels.append(acc)

        # Concatenate along channel dimension: (T, C)
        combined = np.concatenate(channels, axis=1)
        all_features.append(combined)

    return np.array(all_features, dtype=np.float32)


# ================================================================
# MODEL
# ================================================================

class MultiScaleConvBlock(nn.Module):
    """Parallel convolutions with different kernel sizes.
    Captures fine (k=3), medium (k=7), and coarse (k=15) temporal patterns.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Each branch gets out_channels//3 filters
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2  # remaining

        self.branch_fine = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm1d(c1),
            nn.GELU(),
        )
        self.branch_medium = nn.Sequential(
            nn.Conv1d(in_channels, c2, kernel_size=7, padding=3),
            nn.BatchNorm1d(c2),
            nn.GELU(),
        )
        self.branch_coarse = nn.Sequential(
            nn.Conv1d(in_channels, c3, kernel_size=15, padding=7),
            nn.BatchNorm1d(c3),
            nn.GELU(),
        )
        self.pool = nn.AvgPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, T) -> (B, C_out, T//2)"""
        b1 = self.branch_fine(x)
        b2 = self.branch_medium(x)
        b3 = self.branch_coarse(x)
        out = torch.cat([b1, b2, b3], dim=1)  # (B, C_out, T)
        return self.pool(out)  # (B, C_out, T//2)


class ResidualConvBlock(nn.Module):
    """Residual 1D conv block with skip connection."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class TemporalAttention(nn.Module):
    """Self-attention over the temporal dimension.
    Learns which frames are most important for prediction.
    """

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        assert channels % n_heads == 0

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C, T) with attention-weighted features."""
        B, C, T = x.shape
        # Transpose to (B, T, C) for attention
        x_t = x.permute(0, 2, 1)  # (B, T, C)

        # Compute Q, K, V
        qkv = self.qkv(x_t).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)  # (B, T, C)
        out = self.proj(out)

        # Residual + LayerNorm
        out = self.norm(out + x_t)

        return out.permute(0, 2, 1)  # back to (B, C, T)


class ImprovedCNNEncoder(nn.Module):
    """Multi-scale temporal CNN encoder with attention.

    Architecture:
    1. Multi-scale conv block (k=3/7/15) with pooling
    2. Residual conv block
    3. Temporal attention
    4. Residual conv block
    5. Down-conv + global pool -> embedding
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 48,
        embed_dim: int = 24,
        n_attn_heads: int = 4,
        use_attention: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention

        # Stage 1: Multi-scale conv
        self.ms_conv = MultiScaleConvBlock(in_channels, hidden_dim)

        # Stage 2: Residual conv
        self.res1 = ResidualConvBlock(hidden_dim)
        self.pool1 = nn.AvgPool1d(2)

        # Stage 3: Temporal attention (optional)
        if use_attention:
            self.attention = TemporalAttention(hidden_dim, n_heads=n_attn_heads)

        # Stage 4: Second residual + down-conv
        self.res2 = ResidualConvBlock(hidden_dim)
        self.down_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, T) -> (B, embed_dim)"""
        x = self.ms_conv(x)       # (B, hidden_dim, T//2)
        x = self.res1(x)          # (B, hidden_dim, T//2)
        x = self.pool1(x)         # (B, hidden_dim, T//4)

        if self.use_attention:
            x = self.attention(x)  # (B, hidden_dim, T//4)

        x = self.res2(x)          # (B, hidden_dim, T//4)
        x = self.down_conv(x)     # (B, embed_dim, T//4)
        x = self.global_pool(x).squeeze(-1)  # (B, embed_dim)
        x = self.dropout(x)
        return x


class MultiTaskModel(nn.Module):
    """Multi-task model: shared encoder + per-target heads.
    Predicts all 3 targets jointly with a shared backbone.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 48,
        embed_dim: int = 24,
        n_attn_heads: int = 4,
        n_players: int = 5,
        use_attention: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = ImprovedCNNEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            n_attn_heads=n_attn_heads,
            use_attention=use_attention,
            dropout=dropout,
        )
        self.player_embed = nn.Embedding(n_players + 1, 8)

        head_input_dim = embed_dim + 8

        # Per-target prediction heads
        self.head_angle = nn.Sequential(
            nn.Linear(head_input_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self.head_depth = nn.Sequential(
            nn.Linear(head_input_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self.head_lr = nn.Sequential(
            nn.Linear(head_input_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

        self.heads = {"angle": self.head_angle, "depth": self.head_depth, "left_right": self.head_lr}

    def forward(
        self,
        x: torch.Tensor,
        player_id: torch.Tensor,
        target: str | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        x: (B, C_in, T), player_id: (B,)
        If target is None, returns dict of all targets.
        If target is specified, returns single tensor.
        """
        z = self.encoder(x)  # (B, embed_dim)
        p = self.player_embed(player_id)  # (B, 8)
        h = torch.cat([z, p], dim=1)  # (B, embed_dim + 8)

        if target is not None:
            return self.heads[target](h).squeeze(-1)

        return {
            "angle": self.head_angle(h).squeeze(-1),
            "depth": self.head_depth(h).squeeze(-1),
            "left_right": self.head_lr(h).squeeze(-1),
        }


class SingleTaskModel(nn.Module):
    """Single-task model: encoder + player embed + one head."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 48,
        embed_dim: int = 24,
        n_attn_heads: int = 4,
        n_players: int = 5,
        use_attention: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = ImprovedCNNEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            n_attn_heads=n_attn_heads,
            use_attention=use_attention,
            dropout=dropout,
        )
        self.player_embed = nn.Embedding(n_players + 1, 8)
        head_input_dim = embed_dim + 8
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        p = self.player_embed(player_id)
        h = torch.cat([z, p], dim=1)
        return self.head(h).squeeze(-1)


# ================================================================
# AUGMENTATION
# ================================================================

def augment_sequence(seq: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Augment multi-channel sequence.
    seq: (T, C) - z-scored position+velocity+acceleration channels
    """
    seq = seq.copy()

    # Temporal jitter: roll +/-2 frames
    if rng.random() < 0.5:
        shift = rng.randint(-2, 3)
        seq = np.roll(seq, shift, axis=0)

    # Gaussian noise (small)
    if rng.random() < 0.7:
        noise_scale = rng.uniform(0.01, 0.03)
        seq += rng.randn(*seq.shape).astype(np.float32) * noise_scale

    # Feature dropout: zero random joints (affects pos+vel+acc channels for that joint)
    if rng.random() < 0.5:
        n_joint_channels = N_JOINTS * 3  # 207 per representation
        n_reps = seq.shape[1] // n_joint_channels  # 1, 2, or 3
        n_drop = rng.randint(3, 10)
        drop_joints = rng.choice(N_JOINTS, n_drop, replace=False)
        for j in drop_joints:
            for rep in range(n_reps):
                offset = rep * n_joint_channels
                seq[:, offset + j * 3:offset + (j + 1) * 3] = 0.0

    # Scale augmentation (body size variation)
    if rng.random() < 0.3:
        scale = rng.uniform(0.92, 1.08)
        seq *= scale

    # Temporal scaling: slight speed variation via interpolation
    if rng.random() < 0.3:
        T = seq.shape[0]
        speed = rng.uniform(0.9, 1.1)
        new_T = int(T * speed)
        if new_T >= 3 and new_T != T:
            old_indices = np.linspace(0, T - 1, new_T)
            new_seq = np.zeros((new_T, seq.shape[1]), dtype=np.float32)
            for c in range(seq.shape[1]):
                new_seq[:, c] = np.interp(old_indices, np.arange(T), seq[:, c])
            # Pad/crop back to original length
            if new_T < T:
                padded = np.zeros_like(seq)
                padded[:new_T] = new_seq
                padded[new_T:] = new_seq[-1:]
                seq = padded
            else:
                seq = new_seq[:T]

    return seq


# ================================================================
# TRAINING - MULTI-TASK
# ================================================================

def train_multitask_loo(
    X_data: dict[str, np.ndarray],  # {target: (N, T, C)} per-target windows
    pids: np.ndarray,
    y_scaled: dict[str, np.ndarray],
    player_id_map: dict,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, np.ndarray]:
    """LOO cross-validation with multi-task model.
    Returns OOF predictions per target.
    """
    set_seed(seed)
    rng = np.random.RandomState(seed)
    n = len(pids)

    # Use angle window for shared encoder (middle ground)
    # Actually, use per-target windows in single-task mode, or merge for multi-task
    # For multi-task: average the per-target windows to get shared input
    # Better approach: use the widest window that covers all targets
    # Target frames: angle=153, depth=150, LR=170
    # Use depth window (centered at 150) as shared - closest to angle too
    X_shared = X_data["depth"]  # Use depth as shared

    in_channels = X_shared.shape[2]
    epochs = 100 if args.pilot else args.epochs

    oof_preds = {t: np.zeros(n) for t in TARGETS}

    # LOO: train on N-1, predict 1
    for i in range(n):
        if i % 50 == 0:
            print(f"    LOO fold {i}/{n}...")

        # Split
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        X_tr = X_shared[train_mask]
        X_val = X_shared[i:i+1]
        pids_tr = pids[train_mask]
        pid_val = pids[i:i+1]

        model = MultiTaskModel(
            in_channels=in_channels,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            n_attn_heads=args.n_attn_heads,
            use_attention=not args.no_attention,
            dropout=args.dropout,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

        # Prepare validation tensors
        X_val_t = torch.tensor(X_val.transpose(0, 2, 1), dtype=torch.float32, device=device)
        pid_val_t = torch.tensor([player_id_map.get(p, 0) for p in pid_val], dtype=torch.long, device=device)

        y_tr = {t: y_scaled[t][train_mask] for t in TARGETS}
        batch_size = args.batch_size
        n_tr = len(X_tr)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            perm = rng.permutation(n_tr)

            for s in range(0, n_tr, batch_size):
                b_idx = perm[s:s + batch_size]
                batch_x = []
                batch_pid = []
                batch_y = {t: [] for t in TARGETS}

                for bi in b_idx:
                    aug = augment_sequence(X_tr[bi], rng)
                    batch_x.append(aug)
                    batch_pid.append(player_id_map.get(pids_tr[bi], 0))
                    for t in TARGETS:
                        batch_y[t].append(y_tr[t][bi])

                # Mixup
                batch_x_np = np.array(batch_x, dtype=np.float32)
                batch_y_np = {t: np.array(batch_y[t], dtype=np.float32) for t in TARGETS}

                if len(batch_x) > 1 and rng.random() < 0.5:
                    lam = rng.beta(0.3, 0.3)
                    perm_mix = rng.permutation(len(batch_x))
                    batch_x_np = lam * batch_x_np + (1 - lam) * batch_x_np[perm_mix]
                    for t in TARGETS:
                        batch_y_np[t] = lam * batch_y_np[t] + (1 - lam) * batch_y_np[t][perm_mix]

                x_t = torch.tensor(batch_x_np.transpose(0, 2, 1), dtype=torch.float32, device=device)
                pid_t = torch.tensor(batch_pid, dtype=torch.long, device=device)
                y_t = {t: torch.tensor(batch_y_np[t], dtype=torch.float32, device=device) for t in TARGETS}

                preds = model(x_t, pid_t)  # dict of 3 targets

                # Multi-task loss: equal weight for each target
                loss = sum(F.mse_loss(preds[t], y_t[t]) for t in TARGETS) / 3.0

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            scheduler.step()

            # Quick validation (on held-out point) - check if training is converging
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t, pid_val_t)
                val_loss = sum(
                    (val_preds[t].item() - y_scaled[t][i]) ** 2 for t in TARGETS
                ) / 3.0

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= args.patience:
                break

        # Predict with best model
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            final_preds = model(X_val_t, pid_val_t)
            for t in TARGETS:
                oof_preds[t][i] = final_preds[t].item()

    return oof_preds


def train_singletask_loo(
    X_data: dict[str, np.ndarray],  # {target: (N, T, C)} per-target windows
    pids: np.ndarray,
    y_scaled: dict[str, np.ndarray],
    player_id_map: dict,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    target: str,
) -> np.ndarray:
    """LOO CV for a single target. Uses target-specific window."""
    set_seed(seed)
    rng = np.random.RandomState(seed)
    n = len(pids)

    X = X_data[target]
    y = y_scaled[target]
    in_channels = X.shape[2]
    epochs = 100 if args.pilot else args.epochs

    oof_preds = np.zeros(n)

    for i in range(n):
        if i % 50 == 0:
            print(f"    LOO fold {i}/{n} ({target})...")

        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        X_tr = X[train_mask]
        X_val = X[i:i+1]
        pids_tr = pids[train_mask]
        pid_val = pids[i:i+1]
        y_tr = y[train_mask]

        model = SingleTaskModel(
            in_channels=in_channels,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            n_attn_heads=args.n_attn_heads,
            use_attention=not args.no_attention,
            dropout=args.dropout,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

        X_val_t = torch.tensor(X_val.transpose(0, 2, 1), dtype=torch.float32, device=device)
        pid_val_t = torch.tensor([player_id_map.get(p, 0) for p in pid_val], dtype=torch.long, device=device)

        batch_size = args.batch_size
        n_tr = len(X_tr)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            perm = rng.permutation(n_tr)

            for s in range(0, n_tr, batch_size):
                b_idx = perm[s:s + batch_size]
                batch_x = []
                batch_pid = []
                batch_y = []

                for bi in b_idx:
                    aug = augment_sequence(X_tr[bi], rng)
                    batch_x.append(aug)
                    batch_pid.append(player_id_map.get(pids_tr[bi], 0))
                    batch_y.append(y_tr[bi])

                batch_x_np = np.array(batch_x, dtype=np.float32)
                batch_y_np = np.array(batch_y, dtype=np.float32)

                if len(batch_x) > 1 and rng.random() < 0.5:
                    lam = rng.beta(0.3, 0.3)
                    perm_mix = rng.permutation(len(batch_x))
                    batch_x_np = lam * batch_x_np + (1 - lam) * batch_x_np[perm_mix]
                    batch_y_np = lam * batch_y_np + (1 - lam) * batch_y_np[perm_mix]

                x_t = torch.tensor(batch_x_np.transpose(0, 2, 1), dtype=torch.float32, device=device)
                pid_t = torch.tensor(batch_pid, dtype=torch.long, device=device)
                y_t = torch.tensor(batch_y_np, dtype=torch.float32, device=device)

                pred = model(x_t, pid_t)
                loss = F.mse_loss(pred, y_t)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t, pid_val_t)
                val_loss = (val_pred.item() - y[i]) ** 2

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= args.patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof_preds[i] = model(X_val_t, pid_val_t).item()

    return oof_preds


# ================================================================
# TRAINING - KFOLD (faster for iteration)
# ================================================================

def train_kfold_cv(
    X_data: dict[str, np.ndarray],
    pids: np.ndarray,
    y_scaled: dict[str, np.ndarray],
    player_id_map: dict,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, np.ndarray]:
    """5-fold CV per player with multi-task model (faster than LOO)."""
    from sklearn.model_selection import KFold

    set_seed(seed)
    rng = np.random.RandomState(seed)

    X_shared = X_data["depth"]
    in_channels = X_shared.shape[2]
    epochs = 100 if args.pilot else args.epochs
    n = len(pids)

    oof_preds = {t: np.zeros(n) for t in TARGETS}
    unique_pids = sorted(np.unique(pids))

    fold_count = 0
    for pid in unique_pids:
        pid_mask = pids == pid
        pid_indices = np.where(pid_mask)[0]
        X_pid = X_shared[pid_mask]
        pids_pid = pids[pid_mask]
        y_pid = {t: y_scaled[t][pid_mask] for t in TARGETS}

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_pid)):
            fold_count += 1
            if fold_count % 5 == 0:
                print(f"    KFold: P{pid} fold {fold_i+1}/5...")

            X_tr = X_pid[tr_idx]
            X_val = X_pid[val_idx]
            pids_tr = pids_pid[tr_idx]
            pids_val = pids_pid[val_idx]

            model = MultiTaskModel(
                in_channels=in_channels,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                n_attn_heads=args.n_attn_heads,
                use_attention=not args.no_attention,
                dropout=args.dropout,
            ).to(device)

            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

            X_val_t = torch.tensor(X_val.transpose(0, 2, 1), dtype=torch.float32, device=device)
            pid_val_t = torch.tensor(
                [player_id_map.get(p, 0) for p in pids_val], dtype=torch.long, device=device
            )

            y_tr = {t: y_pid[t][tr_idx] for t in TARGETS}
            y_val_np = {t: y_pid[t][val_idx] for t in TARGETS}
            n_tr = len(X_tr)
            batch_size = args.batch_size

            best_val_loss = float("inf")
            best_state = None
            no_improve = 0

            for epoch in range(epochs):
                model.train()
                perm = rng.permutation(n_tr)

                for s in range(0, n_tr, batch_size):
                    b_idx = perm[s:s + batch_size]
                    batch_x = []
                    batch_pid = []
                    batch_y = {t: [] for t in TARGETS}

                    for bi in b_idx:
                        aug = augment_sequence(X_tr[bi], rng)
                        batch_x.append(aug)
                        batch_pid.append(player_id_map.get(pids_tr[bi], 0))
                        for t in TARGETS:
                            batch_y[t].append(y_tr[t][bi])

                    batch_x_np = np.array(batch_x, dtype=np.float32)
                    batch_y_np = {t: np.array(batch_y[t], dtype=np.float32) for t in TARGETS}

                    if len(batch_x) > 1 and rng.random() < 0.5:
                        lam = rng.beta(0.3, 0.3)
                        perm_mix = rng.permutation(len(batch_x))
                        batch_x_np = lam * batch_x_np + (1 - lam) * batch_x_np[perm_mix]
                        for t in TARGETS:
                            batch_y_np[t] = lam * batch_y_np[t] + (1 - lam) * batch_y_np[t][perm_mix]

                    x_t = torch.tensor(batch_x_np.transpose(0, 2, 1), dtype=torch.float32, device=device)
                    pid_t = torch.tensor(batch_pid, dtype=torch.long, device=device)
                    y_t = {t: torch.tensor(batch_y_np[t], dtype=torch.float32, device=device) for t in TARGETS}

                    pred_all = model(x_t, pid_t)
                    loss = sum(F.mse_loss(pred_all[t], y_t[t]) for t in TARGETS) / 3.0

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                scheduler.step()

                model.eval()
                with torch.no_grad():
                    val_pred_all = model(X_val_t, pid_val_t)
                    val_loss = sum(
                        F.mse_loss(
                            val_pred_all[t],
                            torch.tensor(y_val_np[t], dtype=torch.float32, device=device)
                        ).item()
                        for t in TARGETS
                    ) / 3.0

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= args.patience:
                    break

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                final_preds = model(X_val_t, pid_val_t)
                for t in TARGETS:
                    oof_preds[t][pid_indices[val_idx]] = final_preds[t].cpu().numpy()

    return oof_preds


def train_singletask_kfold_cv(
    X_data: dict[str, np.ndarray],
    pids: np.ndarray,
    y_scaled: dict[str, np.ndarray],
    player_id_map: dict,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    target: str,
) -> np.ndarray:
    """5-fold CV per player for a single target (uses target-specific window)."""
    from sklearn.model_selection import KFold

    set_seed(seed)
    rng = np.random.RandomState(seed)

    X = X_data[target]
    y = y_scaled[target]
    in_channels = X.shape[2]
    epochs = 100 if args.pilot else args.epochs
    n = len(pids)

    oof_preds = np.zeros(n)
    unique_pids = sorted(np.unique(pids))

    for pid in unique_pids:
        pid_mask = pids == pid
        pid_indices = np.where(pid_mask)[0]
        X_pid = X[pid_mask]
        pids_pid = pids[pid_mask]
        y_pid = y[pid_mask]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_pid)):
            X_tr = X_pid[tr_idx]
            X_val = X_pid[val_idx]
            pids_tr = pids_pid[tr_idx]
            pids_val = pids_pid[val_idx]
            y_tr = y_pid[tr_idx]
            y_val = y_pid[val_idx]

            model = SingleTaskModel(
                in_channels=in_channels,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                n_attn_heads=args.n_attn_heads,
                use_attention=not args.no_attention,
                dropout=args.dropout,
            ).to(device)

            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

            X_val_t = torch.tensor(X_val.transpose(0, 2, 1), dtype=torch.float32, device=device)
            pid_val_t = torch.tensor(
                [player_id_map.get(p, 0) for p in pids_val], dtype=torch.long, device=device
            )
            y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

            n_tr = len(X_tr)
            batch_size = args.batch_size

            best_val_loss = float("inf")
            best_state = None
            no_improve = 0

            for epoch in range(epochs):
                model.train()
                perm = rng.permutation(n_tr)

                for s in range(0, n_tr, batch_size):
                    b_idx = perm[s:s + batch_size]
                    batch_x = []
                    batch_pid = []
                    batch_y = []

                    for bi in b_idx:
                        aug = augment_sequence(X_tr[bi], rng)
                        batch_x.append(aug)
                        batch_pid.append(player_id_map.get(pids_tr[bi], 0))
                        batch_y.append(y_tr[bi])

                    batch_x_np = np.array(batch_x, dtype=np.float32)
                    batch_y_np = np.array(batch_y, dtype=np.float32)

                    if len(batch_x) > 1 and rng.random() < 0.5:
                        lam = rng.beta(0.3, 0.3)
                        perm_mix = rng.permutation(len(batch_x))
                        batch_x_np = lam * batch_x_np + (1 - lam) * batch_x_np[perm_mix]
                        batch_y_np = lam * batch_y_np + (1 - lam) * batch_y_np[perm_mix]

                    x_t = torch.tensor(batch_x_np.transpose(0, 2, 1), dtype=torch.float32, device=device)
                    pid_t = torch.tensor(batch_pid, dtype=torch.long, device=device)
                    y_t = torch.tensor(batch_y_np, dtype=torch.float32, device=device)

                    pred = model(x_t, pid_t)
                    loss = F.mse_loss(pred, y_t)

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                scheduler.step()

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

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                oof_preds[pid_indices[val_idx]] = model(X_val_t, pid_val_t).cpu().numpy()

        pid_mse = float(np.mean((oof_preds[pid_indices] - y[pid_indices]) ** 2))
        print(f"    P{pid} {target}: CV MSE = {pid_mse:.6f}")

    return oof_preds


# ================================================================
# TEST PREDICTION
# ================================================================

def train_full_predict(
    X_train: np.ndarray,  # (N, T, C)
    pids_train: np.ndarray,
    y_train: dict[str, np.ndarray],  # {target: (N,)}
    X_test: np.ndarray,
    pids_test: np.ndarray,
    player_id_map: dict,
    device: torch.device,
    args: argparse.Namespace,
    multi_task: bool = True,
) -> dict[str, np.ndarray]:
    """Train on all data, predict test. Multi-seed ensemble."""
    n_seeds = 1 if args.pilot else args.n_seeds
    test_preds_all = {t: [] for t in TARGETS}

    in_channels = X_train.shape[2]
    epochs = 100 if args.pilot else args.epochs
    actual_epochs = max(50, int(epochs * 0.7))

    for s in range(n_seeds):
        actual_seed = args.seed + s * 100
        set_seed(actual_seed)
        rng = np.random.RandomState(actual_seed)

        if multi_task:
            model = MultiTaskModel(
                in_channels=in_channels,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                n_attn_heads=args.n_attn_heads,
                use_attention=not args.no_attention,
                dropout=args.dropout,
            ).to(device)
        else:
            # Single task: need separate models per target
            pass

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=actual_epochs, eta_min=1e-5)

        n_train = len(X_train)
        batch_size = args.batch_size

        for epoch in range(actual_epochs):
            model.train()
            perm = rng.permutation(n_train)

            for st in range(0, n_train, batch_size):
                b_idx = perm[st:st + batch_size]
                batch_x = []
                batch_pid = []
                batch_y = {t: [] for t in TARGETS}

                for bi in b_idx:
                    aug = augment_sequence(X_train[bi], rng)
                    batch_x.append(aug)
                    batch_pid.append(player_id_map.get(pids_train[bi], 0))
                    for t in TARGETS:
                        batch_y[t].append(y_train[t][bi])

                batch_x_np = np.array(batch_x, dtype=np.float32)
                batch_y_np = {t: np.array(batch_y[t], dtype=np.float32) for t in TARGETS}

                if len(batch_x) > 1 and rng.random() < 0.5:
                    lam = rng.beta(0.3, 0.3)
                    perm_mix = rng.permutation(len(batch_x))
                    batch_x_np = lam * batch_x_np + (1 - lam) * batch_x_np[perm_mix]
                    for t in TARGETS:
                        batch_y_np[t] = lam * batch_y_np[t] + (1 - lam) * batch_y_np[t][perm_mix]

                x_t = torch.tensor(batch_x_np.transpose(0, 2, 1), dtype=torch.float32, device=device)
                pid_t = torch.tensor(batch_pid, dtype=torch.long, device=device)
                y_t = {t: torch.tensor(batch_y_np[t], dtype=torch.float32, device=device) for t in TARGETS}

                pred_all = model(x_t, pid_t)
                loss = sum(F.mse_loss(pred_all[t], y_t[t]) for t in TARGETS) / 3.0

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            scheduler.step()

        # Predict test
        model.eval()
        X_test_t = torch.tensor(X_test.transpose(0, 2, 1), dtype=torch.float32, device=device)
        pid_test_t = torch.tensor(
            [player_id_map.get(p, 0) for p in pids_test], dtype=torch.long, device=device
        )
        with torch.no_grad():
            test_pred = model(X_test_t, pid_test_t)
            for t in TARGETS:
                test_preds_all[t].append(test_pred[t].cpu().numpy())

    return {t: np.clip(np.mean(test_preds_all[t], axis=0), 0, 1) for t in TARGETS}


def train_full_predict_singletask(
    X_data_train: dict[str, np.ndarray],
    pids_train: np.ndarray,
    y_train: dict[str, np.ndarray],
    X_data_test: dict[str, np.ndarray],
    pids_test: np.ndarray,
    player_id_map: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Train single-task models per target, predict test."""
    n_seeds = 1 if args.pilot else args.n_seeds
    results = {}

    for target in TARGETS:
        print(f"    Training full model for {target}...")
        X_train = X_data_train[target]
        X_test = X_data_test[target]
        y = y_train[target]
        in_channels = X_train.shape[2]
        epochs = 100 if args.pilot else args.epochs
        actual_epochs = max(50, int(epochs * 0.7))

        test_preds_seeds = []

        for s in range(n_seeds):
            actual_seed = args.seed + s * 100
            set_seed(actual_seed)
            rng = np.random.RandomState(actual_seed)

            model = SingleTaskModel(
                in_channels=in_channels,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                n_attn_heads=args.n_attn_heads,
                use_attention=not args.no_attention,
                dropout=args.dropout,
            ).to(device)

            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=actual_epochs, eta_min=1e-5)

            n_train = len(X_train)
            batch_size = args.batch_size

            for epoch in range(actual_epochs):
                model.train()
                perm = rng.permutation(n_train)

                for st in range(0, n_train, batch_size):
                    b_idx = perm[st:st + batch_size]
                    batch_x = []
                    batch_pid = []
                    batch_y = []

                    for bi in b_idx:
                        aug = augment_sequence(X_train[bi], rng)
                        batch_x.append(aug)
                        batch_pid.append(player_id_map.get(pids_train[bi], 0))
                        batch_y.append(y[bi])

                    batch_x_np = np.array(batch_x, dtype=np.float32)
                    batch_y_np = np.array(batch_y, dtype=np.float32)

                    if len(batch_x) > 1 and rng.random() < 0.5:
                        lam = rng.beta(0.3, 0.3)
                        perm_mix = rng.permutation(len(batch_x))
                        batch_x_np = lam * batch_x_np + (1 - lam) * batch_x_np[perm_mix]
                        batch_y_np = lam * batch_y_np + (1 - lam) * batch_y_np[perm_mix]

                    x_t = torch.tensor(batch_x_np.transpose(0, 2, 1), dtype=torch.float32, device=device)
                    pid_t = torch.tensor(batch_pid, dtype=torch.long, device=device)
                    y_t = torch.tensor(batch_y_np, dtype=torch.float32, device=device)

                    pred = model(x_t, pid_t)
                    loss = F.mse_loss(pred, y_t)

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                scheduler.step()

            model.eval()
            X_test_t = torch.tensor(X_test.transpose(0, 2, 1), dtype=torch.float32, device=device)
            pid_test_t = torch.tensor(
                [player_id_map.get(p, 0) for p in pids_test], dtype=torch.long, device=device
            )
            with torch.no_grad():
                test_preds = model(X_test_t, pid_test_t).cpu().numpy()
            test_preds_seeds.append(test_preds)

        results[target] = np.clip(np.mean(test_preds_seeds, axis=0), 0, 1)

    return results


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    print("=" * 70)
    print("IMPROVED MULTI-SCALE ATTENTION CNN")
    print("=" * 70)
    print(f"pilot={args.pilot}, seed={args.seed}, device={device}")
    print(f"base_submission={args.base_submission}")
    print(f"hidden_dim={args.hidden_dim}, embed_dim={args.embed_dim}")
    print(f"attention={not args.no_attention}, multi_task={not args.single_task}")
    print(f"include_velocity={not args.no_velocity}, include_accel={not args.no_accel}")
    print(f"epochs={args.epochs}, lr={args.lr}, wd={args.weight_decay}")

    if args.pilot:
        print("PILOT MODE: 100 epochs, 1 seed")

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

    import joblib
    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    y_scaled = {}
    for t in TARGETS:
        y_scaled[t] = scalers[t].transform(train_df[t].values.reshape(-1, 1)).ravel()
        print(f"  {t}: mean={y_scaled[t].mean():.4f}, std={y_scaled[t].std():.4f}")

    # ----------------------------------------------------------
    # 2. Load sequences
    # ----------------------------------------------------------
    print("\n[2] Loading 69-joint sequences...")
    t_load = time.time()
    seqs_train = load_all_69j_sequences(train_df)
    seqs_test = load_all_69j_sequences(test_df)
    print(f"  Train: {seqs_train.shape}, Test: {seqs_test.shape}")
    print(f"  Load time: {time.time() - t_load:.1f}s")

    # ----------------------------------------------------------
    # 3. Prepare per-target input windows
    # ----------------------------------------------------------
    print("\n[3] Preparing multi-channel input data per target...")
    t_prep = time.time()

    include_vel = not args.no_velocity
    include_acc = not args.no_accel

    X_train_data = {}
    X_test_data = {}

    for target in TARGETS:
        center_frame = TARGET_FRAMES[target]
        X_train_data[target] = prepare_multi_channel_input(
            seqs_train, center_frame,
            include_velocity=include_vel,
            include_accel=include_acc,
        )
        X_test_data[target] = prepare_multi_channel_input(
            seqs_test, center_frame,
            include_velocity=include_vel,
            include_accel=include_acc,
        )
        print(f"  {target} (frame {center_frame}): train={X_train_data[target].shape}, "
              f"test={X_test_data[target].shape}")

    # Z-score normalize per target window
    z_stats = {}
    for target in TARGETS:
        X_tr = X_train_data[target]
        n_ch = X_tr.shape[2]
        mean = X_tr.reshape(-1, n_ch).mean(axis=0)
        std = X_tr.reshape(-1, n_ch).std(axis=0)
        std = np.maximum(std, 1e-6)
        X_train_data[target] = (X_tr - mean) / std
        X_test_data[target] = (X_test_data[target] - mean) / std
        z_stats[target] = (mean, std)

    in_channels = X_train_data["depth"].shape[2]
    print(f"  Input channels: {in_channels}")
    print(f"  Prep time: {time.time() - t_prep:.1f}s")

    # Model size
    if not args.single_task:
        tmp_model = MultiTaskModel(
            in_channels=in_channels,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            n_attn_heads=args.n_attn_heads,
            use_attention=not args.no_attention,
            dropout=args.dropout,
        )
    else:
        tmp_model = SingleTaskModel(
            in_channels=in_channels,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            n_attn_heads=args.n_attn_heads,
            use_attention=not args.no_attention,
            dropout=args.dropout,
        )
    n_params = sum(p.numel() for p in tmp_model.parameters())
    print(f"  Model params: {n_params:,}")
    del tmp_model

    # ----------------------------------------------------------
    # 4. Cross-validation
    # ----------------------------------------------------------
    print("\n[4] Running cross-validation...")
    t_cv = time.time()

    if args.single_task:
        oof_preds = {}
        for target in TARGETS:
            print(f"\n  Training single-task for {target}...")
            oof_preds[target] = train_singletask_kfold_cv(
                X_data=X_train_data,
                pids=pids_train,
                y_scaled=y_scaled,
                player_id_map=player_id_map,
                device=device,
                args=args,
                seed=args.seed,
                target=target,
            )
    else:
        print("  Multi-task KFold CV...")
        oof_preds = train_kfold_cv(
            X_data=X_train_data,
            pids=pids_train,
            y_scaled=y_scaled,
            player_id_map=player_id_map,
            device=device,
            args=args,
            seed=args.seed,
        )

    cv_time = time.time() - t_cv
    print(f"\n  CV time: {cv_time:.1f}s")

    # Compute CV MSE
    cv_scores = {}
    for t in TARGETS:
        mse = float(np.mean((oof_preds[t] - y_scaled[t]) ** 2))
        cv_scores[t] = mse
        print(f"  {t}: CV MSE = {mse:.6f}")
    mean_cv = float(np.mean(list(cv_scores.values())))
    print(f"  MEAN CV MSE: {mean_cv:.6f}")

    # ----------------------------------------------------------
    # 5. Test predictions
    # ----------------------------------------------------------
    print("\n[5] Generating test predictions...")
    t_test = time.time()

    if args.single_task:
        test_preds = train_full_predict_singletask(
            X_data_train=X_train_data,
            pids_train=pids_train,
            y_train=y_scaled,
            X_data_test=X_test_data,
            pids_test=pids_test,
            player_id_map=player_id_map,
            device=device,
            args=args,
        )
    else:
        # Use depth window for multi-task
        test_preds = train_full_predict(
            X_train=X_train_data["depth"],
            pids_train=pids_train,
            y_train=y_scaled,
            X_test=X_test_data["depth"],
            pids_test=pids_test,
            player_id_map=player_id_map,
            device=device,
            args=args,
        )

    test_time = time.time() - t_test
    print(f"  Test prediction time: {test_time:.1f}s")

    # ----------------------------------------------------------
    # 6. Diversity check
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIVERSITY CHECK")
    print("=" * 70)

    div_path = SUBMISSION_DIR / f"submission_{args.diversity_submission}.csv"
    div_sub = None
    if div_path.exists():
        div_sub = pd.read_csv(div_path)
        print(f"\n  vs Sub {args.diversity_submission} (diversity reference):")
        for t in TARGETS:
            col = f"scaled_{t}"
            r, _ = pearsonr(div_sub[col].values, test_preds[t])
            print(f"    {t}: r={r:.4f}")

    base_path = SUBMISSION_DIR / f"submission_{args.base_submission}.csv"
    base_sub = None
    if base_path.exists():
        base_sub = pd.read_csv(base_path)
        print(f"\n  vs Sub {args.base_submission} (blend base):")
        for t in TARGETS:
            col = f"scaled_{t}"
            r, _ = pearsonr(base_sub[col].values, test_preds[t])
            print(f"    {t}: r={r:.4f}")

    # ----------------------------------------------------------
    # 7. Generate submissions
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    submissions = []

    # Standalone
    sub_num = get_next_submission_number()
    sub_df = pd.DataFrame({"id": test_df["id"].values})
    for t, col in zip(TARGETS, TARGET_COLS):
        sub_df[col] = test_preds[t]
    sub_df.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    submissions.append({
        "num": sub_num,
        "desc": f"Standalone improved CNN (CV={mean_cv:.6f})",
    })
    print(f"  Sub {sub_num}: Standalone improved CNN (CV={mean_cv:.6f})")

    # Blends with base submission
    if base_sub is not None:
        blend_weights = [float(w) for w in args.blend_weights.split(",")]
        for w in blend_weights:
            sub_num = get_next_submission_number()
            blend = base_sub.copy()
            for t, col in zip(TARGETS, TARGET_COLS):
                blend[col] = np.clip(
                    (1 - w) * base_sub[col].values + w * test_preds[t],
                    0.0, 1.0,
                )
            blend.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
            desc = f"{int(w*100)}% improved-CNN + {int((1-w)*100)}% Sub{args.base_submission}"
            submissions.append({"num": sub_num, "desc": desc})
            print(f"  Sub {sub_num}: {desc}")

        # Per-target diverse blend: only diverse targets
        if div_sub is not None:
            corrs = {}
            for t in TARGETS:
                col = f"scaled_{t}"
                r, _ = pearsonr(div_sub[col].values, test_preds[t])
                corrs[t] = r

            diverse_targets = [t for t in TARGETS if corrs[t] < 0.80]
            if diverse_targets:
                for w in [0.05, 0.10]:
                    sub_num = get_next_submission_number()
                    blend = base_sub.copy()
                    for t in diverse_targets:
                        col = f"scaled_{t}"
                        blend[col] = np.clip(
                            (1 - w) * base_sub[col].values + w * test_preds[t],
                            0.0, 1.0,
                        )
                    blend.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
                    desc = (
                        f"{int(w*100)}% improved-CNN on {'+'.join(diverse_targets)} "
                        f"+ Sub{args.base_submission}"
                    )
                    submissions.append({"num": sub_num, "desc": desc})
                    print(f"  Sub {sub_num}: {desc}")

    # ----------------------------------------------------------
    # 8. Summary
    # ----------------------------------------------------------
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")

    mode = "single-task" if args.single_task else "multi-task"
    attn = "with-attention" if not args.no_attention else "no-attention"
    channels = "pos"
    if not args.no_velocity:
        channels += "+vel"
    if not args.no_accel:
        channels += "+acc"

    print(f"\nConfig: {mode}, {attn}, channels={channels}")
    print(f"Architecture: hidden={args.hidden_dim}, embed={args.embed_dim}, "
          f"attn_heads={args.n_attn_heads}")

    print(f"\nCV MSE per target:")
    for t in TARGETS:
        print(f"  {t}: {cv_scores[t]:.6f}")
    print(f"  MEAN: {mean_cv:.6f}")

    if div_sub is not None:
        print(f"\nDiversity with Sub{args.diversity_submission}:")
        for t in TARGETS:
            col = f"scaled_{t}"
            r, _ = pearsonr(div_sub[col].values, test_preds[t])
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
        "mode": mode,
        "attention": not args.no_attention,
        "channels": channels,
        "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim,
        "n_attn_heads": args.n_attn_heads,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "n_params": n_params,
        "in_channels": in_channels,
        "cv_time_s": cv_time,
        "test_time_s": test_time,
        "total_time_s": total_time,
        "cv_scores": cv_scores,
        "mean_cv": mean_cv,
        "submissions": submissions,
    }
    run_path = OUTPUT_DIR / f"improved_cnn_run_{ts}.json"
    run_path.write_text(json.dumps(run_data, indent=2))
    print(f"\nRun metadata: {run_path}")
    print("Done.")


if __name__ == "__main__":
    main()

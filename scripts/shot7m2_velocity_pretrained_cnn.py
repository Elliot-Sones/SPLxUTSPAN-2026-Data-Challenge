"""
SHOT7M2 Velocity-Pretrained 1D CNN

Key insight: wrist VELOCITIES are similar between SHOT7M2 and competition data
(0.0798 vs 0.0741 torso-lengths/frame) even though positions differ massively.
We pretrain a small 1D CNN on SHOT7M2 velocity sequences, then fine-tune on
competition data. This creates a NEW MODEL FAMILY for ensemble diversity.

Pipeline:
1. Load SHOT7M2 shooting events -> compute velocities
2. Load competition data (14 shared joints) -> subsample 2x -> compute velocities
3. Pretrain CNN encoder on SHOT7M2 (masked recon + moveme + shoot confidence)
4. Fine-tune per-target on competition data with heavy augmentation
5. Generate standalone + blend submissions
6. Also train scratch baseline (no pretraining) for comparison
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
SHOT7M2_DIR = PROJECT_DIR / "external_data" / "shot7m2_sample"

TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

# 14 shared joints between SHOT7M2 and competition
JOINT_MAP = {
    "mid_hip": 0,
    "left_hip": 1,
    "left_knee": 2,
    "left_ankle": 3,
    "right_hip": 6,
    "right_knee": 7,
    "right_ankle": 8,
    "left_shoulder": 16,
    "left_elbow": 17,
    "left_wrist": 18,
    "right_shoulder": 23,
    "right_elbow": 24,
    "right_wrist": 25,
    "neck": 19,
}
MAPPED_JOINTS = list(JOINT_MAP.keys())
SHOT7M2_INDICES = [JOINT_MAP[j] for j in MAPPED_JOINTS]

# Moveme joints: indices into the 14-joint array
MOVEME_JOINTS = {
    "right_wrist": MAPPED_JOINTS.index("right_wrist"),
    "left_wrist": MAPPED_JOINTS.index("left_wrist"),
    "right_elbow": MAPPED_JOINTS.index("right_elbow"),
    "left_elbow": MAPPED_JOINTS.index("left_elbow"),
    "left_knee": MAPPED_JOINTS.index("left_knee"),
    "right_knee": MAPPED_JOINTS.index("right_knee"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHOT7M2 velocity-pretrained CNN")
    p.add_argument("--pilot", action="store_true", help="Quick pilot run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--base-submission", type=int, default=2716)
    p.add_argument("--shot-threshold", type=float, default=0.3)
    p.add_argument("--window-radius", type=int, default=16)
    # Pretraining
    p.add_argument("--pretrain-epochs", type=int, default=20)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--pretrain-batch", type=int, default=64)
    p.add_argument("--mask-ratio", type=float, default=0.30)
    # Fine-tuning
    p.add_argument("--finetune-epochs", type=int, default=200)
    p.add_argument("--finetune-lr", type=float, default=1e-3)
    p.add_argument("--finetune-batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--n-seeds", type=int, default=5)
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


def normalize_pose_frames(positions: np.ndarray) -> np.ndarray:
    """Pelvis-center + torso-scale normalization. positions: (T, 14, 3)"""
    pelvis_idx = MAPPED_JOINTS.index("mid_hip")
    neck_idx = MAPPED_JOINTS.index("neck")
    centered = positions - positions[:, pelvis_idx:pelvis_idx + 1, :]
    torso = np.linalg.norm(centered[:, neck_idx] - centered[:, pelvis_idx], axis=1, keepdims=True)
    torso = np.maximum(torso, 1e-6)
    return centered / torso[:, :, np.newaxis]


def apply_axis_transform(positions: np.ndarray, perm=(0, 2, 1), signs=(1, 1, 1)) -> np.ndarray:
    x = positions[:, :, list(perm)]
    return x * np.asarray(signs, dtype=float)[np.newaxis, np.newaxis, :]


def load_competition_14j_sequences(df: pd.DataFrame) -> np.ndarray:
    """Load 14-joint sequences from competition data. Returns (N, 240, 14, 3)."""
    n = len(df)
    seqs = np.zeros((n, 240, 14, 3), dtype=float)
    for j_idx, name in enumerate(MAPPED_JOINTS):
        for c_idx, coord in enumerate(["x", "y", "z"]):
            col = f"{name}_{coord}"
            vals = df[col].apply(parse_json_array_240).values
            for i in range(n):
                seqs[i, :, j_idx, c_idx] = vals[i]
    # Apply axis transform to align with SHOT7M2 coordinate system
    seqs_flat = seqs.reshape(-1, 14, 3)
    seqs_flat = apply_axis_transform(seqs_flat, perm=(0, 2, 1), signs=(1, 1, 1))
    return seqs_flat.reshape(n, 240, 14, 3)


def positions_to_velocities(positions: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame velocity. positions: (..., T, 14, 3) -> (..., T-1, 42)"""
    vel = positions[..., 1:, :, :] - positions[..., :-1, :, :]
    # Flatten joint*coord dimension
    shape = vel.shape[:-2] + (vel.shape[-2] * vel.shape[-1],)  # (..., T-1, 42)
    return vel.reshape(shape)


# ================================================================
# SHOT7M2 DATA
# ================================================================

def collect_shot7m2_data(args: argparse.Namespace) -> dict:
    """Load SHOT7M2 shooting windows with velocities and labels."""
    poses = np.load(SHOT7M2_DIR / "train" / "train_dictionary_poses.npy", allow_pickle=True).item()
    actions = np.load(SHOT7M2_DIR / "train" / "train_dictionary_actions.npy", allow_pickle=True).item()

    labels = actions["label_array"]
    vocab = actions["vocabulary"]
    fnm = actions["frame_number_map"]

    shoot_idx = vocab.index("action_Shoot")
    dribble_idx = vocab.index("action_Dribble") if "action_Dribble" in vocab else None
    move_idx = vocab.index("action_Move") if "action_Move" in vocab else None
    sprint_idx = vocab.index("action_Sprint") if "action_Sprint" in vocab else None

    radius = args.window_radius
    window_len = 2 * radius + 1  # 33 frames

    all_vel_windows = []  # velocity windows: (32, 42)
    all_shoot_confs = []  # action_Shoot confidence at center
    all_moveme_labels = []  # 6-joint activity labels

    for key, seq in poses["sequences"]["keypoints"].items():
        start, end = fnm[key]
        shoot = labels[shoot_idx, start:end]
        shoot_frames = np.where(shoot > args.shot_threshold)[0]
        if len(shoot_frames) == 0:
            continue

        ep_poses = seq[:, 0, SHOT7M2_INDICES, :]  # (T, 14, 3)

        for f in shoot_frames:
            # Extract position window
            t_start = max(0, f - radius)
            t_end = min(ep_poses.shape[0], f + radius + 1)
            pos_window = ep_poses[t_start:t_end]  # (L, 14, 3)

            # Pad if needed
            if pos_window.shape[0] < window_len:
                padded = np.zeros((window_len, 14, 3), dtype=float)
                offset = radius - (f - t_start)
                padded[offset:offset + pos_window.shape[0]] = pos_window
                # Edge-pad
                if offset > 0:
                    padded[:offset] = pos_window[0:1]
                end_idx = offset + pos_window.shape[0]
                if end_idx < window_len:
                    padded[end_idx:] = pos_window[-1:]
                pos_window = padded

            # Normalize
            pos_window = normalize_pose_frames(pos_window)

            # Compute velocities: (32, 42)
            vel = positions_to_velocities(pos_window)  # (32, 42)
            all_vel_windows.append(vel)

            # Shoot confidence
            all_shoot_confs.append(float(shoot[f]))

            # Moveme labels: joint activity = mean absolute velocity in window
            vel_3d = pos_window[1:] - pos_window[:-1]  # (32, 14, 3)
            joint_activity = np.mean(np.abs(vel_3d), axis=(0, 2))  # (14,)
            moveme = np.array([
                joint_activity[MOVEME_JOINTS["right_wrist"]],
                joint_activity[MOVEME_JOINTS["left_wrist"]],
                joint_activity[MOVEME_JOINTS["right_elbow"]],
                joint_activity[MOVEME_JOINTS["left_elbow"]],
                joint_activity[MOVEME_JOINTS["left_knee"]],
                joint_activity[MOVEME_JOINTS["right_knee"]],
            ], dtype=float)
            # Normalize to [0, 1] range (clip at reasonable max)
            moveme = np.clip(moveme / 0.3, 0.0, 1.0)
            all_moveme_labels.append(moveme)

    vel_windows = np.array(all_vel_windows, dtype=np.float32)
    shoot_confs = np.array(all_shoot_confs, dtype=np.float32)
    moveme_labels = np.array(all_moveme_labels, dtype=np.float32)

    print(f"  SHOT7M2: {len(vel_windows)} shooting windows, vel shape={vel_windows.shape}")
    print(f"  Shoot conf: mean={shoot_confs.mean():.4f}, moveme mean={moveme_labels.mean():.4f}")

    return {
        "vel_windows": vel_windows,  # (N, 32, 42)
        "shoot_confs": shoot_confs,  # (N,)
        "moveme_labels": moveme_labels,  # (N, 6)
    }


def prepare_competition_velocities(
    seqs: np.ndarray,  # (N, 240, 14, 3)
) -> np.ndarray:
    """Normalize, subsample 2x, compute velocities. Returns (N, 119, 42)."""
    n = seqs.shape[0]
    all_vel = []
    for i in range(n):
        # Normalize each shot
        normed = normalize_pose_frames(seqs[i])  # (240, 14, 3)
        # Subsample 2x: 60fps -> 30fps equivalent
        subsampled = normed[::2]  # (120, 14, 3)
        # Compute velocities
        vel = positions_to_velocities(subsampled)  # (119, 42)
        all_vel.append(vel)
    return np.array(all_vel, dtype=np.float32)


# ================================================================
# MODEL
# ================================================================

class Velocity1DCNN(nn.Module):
    """Small 1D CNN encoder for velocity sequences.

    Input: (batch, 42, T) - 14 joints x 3 coords velocities
    Encoder output: 16-dim vector
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(42, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.AvgPool1d(2)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.AvgPool1d(2)

        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 42, T) -> (B, 16)"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)  # (B, 16)
        return x


class PretrainModel(nn.Module):
    """Pretraining wrapper with reconstruction + moveme + shoot heads."""

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.encoder = Velocity1DCNN()
        self.recon_head = nn.Linear(16, seq_len * 42)
        self.moveme_head = nn.Linear(16, 6)
        self.shoot_head = nn.Linear(16, 1)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, 42, T)"""
        z = self.encoder(x)  # (B, 16)
        recon = self.recon_head(z).view(-1, self.seq_len, 42)  # (B, T, 42)
        moveme = self.moveme_head(z)  # (B, 6)
        shoot = self.shoot_head(z).squeeze(-1)  # (B,)
        return z, recon, moveme, shoot


class FinetuneModel(nn.Module):
    """Fine-tuning wrapper: pretrained encoder + player embed + prediction head."""

    def __init__(self, encoder: Velocity1DCNN, n_players: int = 5) -> None:
        super().__init__()
        self.encoder = encoder
        self.player_embed = nn.Embedding(n_players + 1, 4)
        self.fc1 = nn.Linear(16 + 4, 16)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        """x: (B, 42, T), player_id: (B,) -> (B,)"""
        z = self.encoder(x)  # (B, 16)
        p = self.player_embed(player_id)  # (B, 4)
        h = torch.cat([z, p], dim=1)  # (B, 20)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1)


# ================================================================
# PRETRAINING
# ================================================================

def augment_pretrain(vel: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Augment velocity windows for pretraining.
    vel: (T, 42) numpy array
    """
    vel = vel.copy()

    # Temporal crop: random sub-window (80-100% of length)
    if rng.random() < 0.5:
        T = vel.shape[0]
        crop_len = rng.randint(int(T * 0.8), T + 1)
        start = rng.randint(0, T - crop_len + 1)
        cropped = vel[start:start + crop_len]
        # Resize back to T via linear interpolation
        if crop_len != T:
            indices = np.linspace(0, crop_len - 1, T)
            vel = np.array([np.interp(indices, np.arange(crop_len), cropped[:, j])
                           for j in range(42)]).T

    # Gaussian noise
    if rng.random() < 0.7:
        vel += rng.randn(*vel.shape).astype(np.float32) * 0.02

    # Speed perturbation: scale velocities
    if rng.random() < 0.3:
        scale = rng.uniform(0.8, 1.2)
        vel *= scale

    return vel


def pretrain(
    shot7m2_data: dict,
    args: argparse.Namespace,
    device: torch.device,
    vel_mean: np.ndarray,
    vel_std: np.ndarray,
) -> PretrainModel:
    """Pretrain on SHOT7M2 shooting velocity windows."""
    vel_windows = shot7m2_data["vel_windows"]  # (N, 32, 42)
    shoot_confs = shot7m2_data["shoot_confs"]  # (N,)
    moveme_labels = shot7m2_data["moveme_labels"]  # (N, 6)

    seq_len = vel_windows.shape[1]
    n = len(vel_windows)

    max_windows = 200 if args.pilot else n
    if max_windows < n:
        idx = np.random.choice(n, max_windows, replace=False)
        vel_windows = vel_windows[idx]
        shoot_confs = shoot_confs[idx]
        moveme_labels = moveme_labels[idx]
        n = max_windows

    epochs = 3 if args.pilot else args.pretrain_epochs

    model = PretrainModel(seq_len=seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    rng = np.random.RandomState(args.seed)
    batch_size = args.pretrain_batch

    print(f"  Pretraining: {n} windows, {epochs} epochs, seq_len={seq_len}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params}")

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        total_loss = 0.0
        n_batches = 0

        for s in range(0, n, batch_size):
            b_idx = perm[s:s + batch_size]
            batch_vel = []
            for i in b_idx:
                aug = augment_pretrain(vel_windows[i], rng)
                # Z-score normalize
                aug = (aug - vel_mean) / vel_std
                batch_vel.append(aug)
            batch_vel = np.array(batch_vel, dtype=np.float32)

            # (B, T, 42) -> (B, 42, T) for Conv1d
            x = torch.tensor(batch_vel.transpose(0, 2, 1), dtype=torch.float32, device=device)

            # Mask for reconstruction
            B, C, T = x.shape
            mask_frames = torch.rand(B, T, device=device) < args.mask_ratio
            x_masked = x.clone()
            x_masked[:, :, mask_frames.any(dim=0)] = 0.0  # Zero out masked frames
            # Per-sample masking
            for bi in range(B):
                x_masked[bi, :, mask_frames[bi]] = 0.0

            z, recon, moveme_pred, shoot_pred = model(x_masked)

            # Reconstruction loss (on masked frames only)
            # recon: (B, T, 42), x transposed: (B, T, 42)
            x_target = x.transpose(1, 2)  # (B, T, 42)
            mask_expanded = mask_frames.unsqueeze(-1).expand(-1, -1, 42).float()
            denom = torch.clamp(mask_expanded.sum(), min=1.0)
            loss_recon = ((recon - x_target) ** 2 * mask_expanded).sum() / denom

            # Moveme loss (BCE)
            moveme_target = torch.tensor(moveme_labels[b_idx], dtype=torch.float32, device=device)
            loss_moveme = F.binary_cross_entropy_with_logits(moveme_pred, moveme_target)

            # Shoot confidence loss (MSE)
            shoot_target = torch.tensor(shoot_confs[b_idx], dtype=torch.float32, device=device)
            loss_shoot = F.mse_loss(shoot_pred, shoot_target)

            loss = 1.0 * loss_recon + 0.3 * loss_moveme + 0.5 * loss_shoot

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}")

    return model


# ================================================================
# FINE-TUNING
# ================================================================

def augment_finetune(vel: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Augment velocity for fine-tuning.
    vel: (T, 42) numpy array, already z-scored
    """
    vel = vel.copy()

    # Temporal jitter: roll +/-3 frames
    if rng.random() < 0.5:
        shift = rng.randint(-3, 4)
        vel = np.roll(vel, shift, axis=0)

    # Gaussian noise
    if rng.random() < 0.7:
        vel += rng.randn(*vel.shape).astype(np.float32) * 0.02

    # Feature dropout: zero 2-4 random joints (each joint = 3 features)
    if rng.random() < 0.5:
        n_drop = rng.randint(2, 5)
        drop_joints = rng.choice(14, n_drop, replace=False)
        for j in drop_joints:
            vel[:, j * 3:(j + 1) * 3] = 0.0

    return vel


def train_one_fold_finetune(
    encoder: Velocity1DCNN,
    X_train: np.ndarray,  # (N_tr, T, 42) z-scored velocities
    pids_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    pids_val: np.ndarray,
    y_val: np.ndarray,
    player_id_map: dict,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    pretrained: bool = True,
) -> tuple[np.ndarray, float]:
    """Train one CV fold. Returns (val_preds, best_val_loss)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    # Build model
    if pretrained:
        import copy
        enc = copy.deepcopy(encoder)
    else:
        enc = Velocity1DCNN()
    model = FinetuneModel(enc, n_players=5).to(device)

    epochs = 30 if args.pilot else args.finetune_epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    # Convert val to tensors (no augmentation)
    X_val_t = torch.tensor(X_val.transpose(0, 2, 1), dtype=torch.float32, device=device)
    pid_val_t = torch.tensor([player_id_map.get(p, 0) for p in pids_val], dtype=torch.long, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    batch_size = args.finetune_batch
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

            x_t = torch.tensor(batch_x_np.transpose(0, 2, 1), dtype=torch.float32, device=device)
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
    encoder: Velocity1DCNN,
    X_train: np.ndarray,
    pids_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    pids_test: np.ndarray,
    player_id_map: dict,
    device: torch.device,
    args: argparse.Namespace,
    pretrained: bool = True,
) -> np.ndarray:
    """Train on full training set with multiple seeds and average predictions."""
    n_seeds = 3 if args.pilot else args.n_seeds
    test_preds_all = []

    for s in range(n_seeds):
        actual_seed = args.seed + s * 100
        torch.manual_seed(actual_seed)
        rng = np.random.RandomState(actual_seed)

        if pretrained:
            import copy
            enc = copy.deepcopy(encoder)
        else:
            enc = Velocity1DCNN()
        model = FinetuneModel(enc, n_players=5).to(device)

        epochs = 20 if args.pilot else args.finetune_epochs
        actual_epochs = max(30, int(epochs * 0.7))

        opt = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=actual_epochs, eta_min=1e-5)

        batch_size = args.finetune_batch
        n_train = len(X_train)

        for epoch in range(actual_epochs):
            model.train()
            perm = rng.permutation(n_train)

            for st in range(0, n_train, batch_size):
                b_idx = perm[st:st + batch_size]
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

                x_t = torch.tensor(bx.transpose(0, 2, 1), dtype=torch.float32, device=device)
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
        X_test_t = torch.tensor(X_test.transpose(0, 2, 1), dtype=torch.float32, device=device)
        pid_test_t = torch.tensor([player_id_map.get(p, 0) for p in pids_test], dtype=torch.long, device=device)
        with torch.no_grad():
            test_preds = model(X_test_t, pid_test_t).cpu().numpy()
        test_preds_all.append(test_preds)

    return np.mean(test_preds_all, axis=0)


def run_cv_and_test(
    encoder: Velocity1DCNN | None,
    X_train_vel: np.ndarray,  # (N, T, 42) z-scored
    pids_train: np.ndarray,
    y_scaled: dict,
    X_test_vel: np.ndarray,
    pids_test: np.ndarray,
    player_id_map: dict,
    device: torch.device,
    args: argparse.Namespace,
    pretrained: bool,
    label: str,
) -> dict:
    """Run per-target CV + test prediction."""
    results = {}
    unique_pids = sorted(np.unique(pids_train))

    for target in TARGETS:
        y_t = y_scaled[target]
        oof_preds = np.zeros(len(pids_train))
        fold_losses = []

        for pid in unique_pids:
            pid_mask = pids_train == pid
            X_pid = X_train_vel[pid_mask]
            y_pid = y_t[pid_mask]
            pids_pid = pids_train[pid_mask]
            pid_indices = np.where(pid_mask)[0]

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_pid)):
                val_preds, val_loss = train_one_fold_finetune(
                    encoder=encoder,
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
                    pretrained=pretrained,
                )
                oof_preds[pid_indices[val_idx]] = val_preds
                fold_losses.append(val_loss)

        cv_mse = float(np.mean((oof_preds - y_t) ** 2))
        print(f"    {label} {target}: CV MSE = {cv_mse:.6f}")

        # Test predictions
        test_preds = train_full_and_predict(
            encoder=encoder,
            X_train=X_train_vel,
            pids_train=pids_train,
            y_train=y_t,
            X_test=X_test_vel,
            pids_test=pids_test,
            player_id_map=player_id_map,
            device=device,
            args=args,
            pretrained=pretrained,
        )

        results[target] = {
            "cv_mse": cv_mse,
            "oof_preds": oof_preds,
            "test_preds": np.clip(test_preds, 0.0, 1.0),
        }

    mean_cv = float(np.mean([results[t]["cv_mse"] for t in TARGETS]))
    print(f"    {label} MEAN CV MSE: {mean_cv:.6f}")
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
    print("SHOT7M2 VELOCITY-PRETRAINED 1D CNN")
    print("=" * 70)
    print(f"pilot={args.pilot}, seed={args.seed}, device={device}")
    print(f"base_submission={args.base_submission}")

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

    # Load targets (scaled)
    import joblib
    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    y_scaled = {}
    for t in TARGETS:
        y_scaled[t] = scalers[t].transform(train_df[t].values.reshape(-1, 1)).ravel()

    # Load competition 14-joint sequences
    print("  Loading competition 14-joint sequences...")
    seqs_train = load_competition_14j_sequences(train_df)
    seqs_test = load_competition_14j_sequences(test_df)
    print(f"  Train: {seqs_train.shape}, Test: {seqs_test.shape}")

    # ----------------------------------------------------------
    # 2. Load SHOT7M2 and compute velocities
    # ----------------------------------------------------------
    print("\n[2] Loading SHOT7M2 shooting windows...")
    shot7m2_data = collect_shot7m2_data(args)

    # Compute velocity statistics from SHOT7M2 for z-scoring
    shot_vel = shot7m2_data["vel_windows"]  # (N, 32, 42)
    vel_mean = shot_vel.reshape(-1, 42).mean(axis=0)
    vel_std = shot_vel.reshape(-1, 42).std(axis=0)
    vel_std = np.maximum(vel_std, 1e-6)
    print(f"  Velocity stats: mean_abs={np.abs(vel_mean).mean():.6f}, std_mean={vel_std.mean():.6f}")

    # ----------------------------------------------------------
    # 3. Prepare competition velocities
    # ----------------------------------------------------------
    print("\n[3] Preparing competition velocities...")
    X_train_vel = prepare_competition_velocities(seqs_train)  # (N, 119, 42)
    X_test_vel = prepare_competition_velocities(seqs_test)    # (N, 119, 42)
    print(f"  Train vel: {X_train_vel.shape}, Test vel: {X_test_vel.shape}")

    # Z-score normalize using SHOT7M2 velocity statistics (domain alignment)
    X_train_vel = (X_train_vel - vel_mean) / vel_std
    X_test_vel = (X_test_vel - vel_mean) / vel_std
    print(f"  After z-score: train mean={X_train_vel.mean():.4f}, std={X_train_vel.std():.4f}")

    # ----------------------------------------------------------
    # 4. Pretrain on SHOT7M2
    # ----------------------------------------------------------
    print("\n[4] Pretraining on SHOT7M2...")
    t_pretrain = time.time()
    pretrain_model = pretrain(shot7m2_data, args, device, vel_mean, vel_std)
    pretrain_time = time.time() - t_pretrain
    print(f"  Pretrain time: {pretrain_time:.1f}s")

    # Extract pretrained encoder
    pretrained_encoder = pretrain_model.encoder

    # ----------------------------------------------------------
    # 5. Fine-tune (pretrained) + CV
    # ----------------------------------------------------------
    print("\n[5] Fine-tuning pretrained model...")
    t_ft = time.time()
    pretrained_results = run_cv_and_test(
        encoder=pretrained_encoder,
        X_train_vel=X_train_vel,
        pids_train=pids_train,
        y_scaled=y_scaled,
        X_test_vel=X_test_vel,
        pids_test=pids_test,
        player_id_map=player_id_map,
        device=device,
        args=args,
        pretrained=True,
        label="PRETRAINED",
    )
    ft_time = time.time() - t_ft
    print(f"  Fine-tune time: {ft_time:.1f}s")

    # ----------------------------------------------------------
    # 6. Scratch baseline (no pretraining)
    # ----------------------------------------------------------
    print("\n[6] Training scratch baseline (no pretraining)...")
    t_scratch = time.time()
    scratch_results = run_cv_and_test(
        encoder=None,
        X_train_vel=X_train_vel,
        pids_train=pids_train,
        y_scaled=y_scaled,
        X_test_vel=X_test_vel,
        pids_test=pids_test,
        player_id_map=player_id_map,
        device=device,
        args=args,
        pretrained=False,
        label="SCRATCH",
    )
    scratch_time = time.time() - t_scratch
    print(f"  Scratch time: {scratch_time:.1f}s")

    # ----------------------------------------------------------
    # 7. Kill criteria check
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("KILL CRITERIA CHECK")
    print("=" * 70)

    pretrained_cv = pretrained_results["mean_cv_mse"]
    scratch_cv = scratch_results["mean_cv_mse"]

    pretrain_delta = (pretrained_cv - scratch_cv) / scratch_cv * 100
    print(f"  Pretrained mean CV: {pretrained_cv:.6f}")
    print(f"  Scratch mean CV:    {scratch_cv:.6f}")
    print(f"  Pretrain delta:     {pretrain_delta:+.2f}%")

    # Use the better model for submissions
    use_pretrained = pretrained_cv < scratch_cv
    best_results = pretrained_results if use_pretrained else scratch_results
    best_label = "PRETRAINED" if use_pretrained else "SCRATCH"
    print(f"  Using: {best_label}")

    all_above_015 = all(best_results[t]["cv_mse"] > 0.015 for t in TARGETS)
    if all_above_015:
        print("  WARNING: All target CV MSE > 0.015 - KILL CRITERIA MET")
        print("  Proceeding anyway to generate submissions for analysis...")

    # ----------------------------------------------------------
    # 8. Diversity check
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIVERSITY CHECK")
    print("=" * 70)

    base_path = SUBMISSION_DIR / f"submission_{args.base_submission}.csv"
    if not base_path.exists():
        print(f"  WARNING: Base submission {base_path} not found. Skipping diversity check.")
        base_sub = None
    else:
        base_sub = pd.read_csv(base_path)
        for target in TARGETS:
            col = f"scaled_{target}"
            r_pre = np.corrcoef(base_sub[col].values, pretrained_results[target]["test_preds"])[0, 1]
            r_scr = np.corrcoef(base_sub[col].values, scratch_results[target]["test_preds"])[0, 1]
            print(f"  {target}: pretrained r={r_pre:.4f}, scratch r={r_scr:.4f}")

    # ----------------------------------------------------------
    # 9. Generate submissions
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    submissions = []

    # Standalone pretrained
    sub_num = get_next_submission_number()
    sub_standalone = pd.DataFrame({"id": test_df["id"].values})
    for t, col in zip(TARGETS, TARGET_COLS):
        sub_standalone[col] = pretrained_results[t]["test_preds"]
    sub_standalone.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    submissions.append({"num": sub_num, "desc": "Standalone pretrained CNN"})
    print(f"  Sub {sub_num}: Standalone pretrained CNN")

    # Standalone scratch
    sub_num = get_next_submission_number()
    sub_scratch = pd.DataFrame({"id": test_df["id"].values})
    for t, col in zip(TARGETS, TARGET_COLS):
        sub_scratch[col] = scratch_results[t]["test_preds"]
    sub_scratch.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    submissions.append({"num": sub_num, "desc": "Standalone scratch CNN"})
    print(f"  Sub {sub_num}: Standalone scratch CNN")

    # Blends with base submission
    if base_sub is not None:
        for model_name, model_results in [("pretrained", pretrained_results), ("scratch", scratch_results)]:
            for w in [0.01, 0.02, 0.03, 0.05]:
                sub_num = get_next_submission_number()
                blend = base_sub.copy()
                for t, col in zip(TARGETS, TARGET_COLS):
                    blend[col] = np.clip(
                        (1 - w) * base_sub[col].values + w * model_results[t]["test_preds"],
                        0.0, 1.0,
                    )
                blend.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
                submissions.append({
                    "num": sub_num,
                    "desc": f"{int(w*100)}% {model_name} CNN + {int((1-w)*100)}% Sub{args.base_submission}",
                })
                print(f"  Sub {sub_num}: {int(w*100)}% {model_name} CNN + Sub{args.base_submission}")

        # Per-target blend: use CNN only on targets with lowest correlation
        if base_sub is not None:
            for model_name, model_results in [("pretrained", pretrained_results), ("scratch", scratch_results)]:
                # Find which targets have lowest correlation
                corrs = {}
                for t in TARGETS:
                    col = f"scaled_{t}"
                    r = np.corrcoef(base_sub[col].values, model_results[t]["test_preds"])[0, 1]
                    corrs[t] = r

                # Blend only targets where r < 0.80
                diverse_targets = [t for t in TARGETS if corrs[t] < 0.80]
                if diverse_targets:
                    for w in [0.03, 0.05]:
                        sub_num = get_next_submission_number()
                        blend = base_sub.copy()
                        desc_parts = []
                        for t in diverse_targets:
                            col = f"scaled_{t}"
                            blend[col] = np.clip(
                                (1 - w) * base_sub[col].values + w * model_results[t]["test_preds"],
                                0.0, 1.0,
                            )
                            desc_parts.append(f"{t}(r={corrs[t]:.2f})")
                        blend.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
                        desc = f"{int(w*100)}% {model_name} on {'+'.join(diverse_targets)} + Sub{args.base_submission}"
                        submissions.append({"num": sub_num, "desc": desc})
                        print(f"  Sub {sub_num}: {desc}")

    # ----------------------------------------------------------
    # 10. Summary
    # ----------------------------------------------------------
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")
    print(f"\nPretrained CV MSE per target:")
    for t in TARGETS:
        print(f"  {t}: {pretrained_results[t]['cv_mse']:.6f}")
    print(f"  MEAN: {pretrained_results['mean_cv_mse']:.6f}")

    print(f"\nScratch CV MSE per target:")
    for t in TARGETS:
        print(f"  {t}: {scratch_results[t]['cv_mse']:.6f}")
    print(f"  MEAN: {scratch_results['mean_cv_mse']:.6f}")

    print(f"\nPretrain benefit: {pretrain_delta:+.2f}%")
    print(f"\nSubmissions generated: {len(submissions)}")
    for s in submissions:
        print(f"  Sub {s['num']}: {s['desc']}")

    # Save run metadata
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_data = {
        "timestamp": ts,
        "pilot": args.pilot,
        "pretrain_time_s": pretrain_time,
        "finetune_time_s": ft_time,
        "scratch_time_s": scratch_time,
        "total_time_s": total_time,
        "pretrained_cv": {t: pretrained_results[t]["cv_mse"] for t in TARGETS},
        "pretrained_mean_cv": pretrained_results["mean_cv_mse"],
        "scratch_cv": {t: scratch_results[t]["cv_mse"] for t in TARGETS},
        "scratch_mean_cv": scratch_results["mean_cv_mse"],
        "pretrain_delta_pct": pretrain_delta,
        "submissions": submissions,
    }
    run_path = OUTPUT_DIR / f"shot7m2_velocity_cnn_run_{ts}.json"
    run_path.write_text(json.dumps(run_data, indent=2))
    print(f"\nRun metadata: {run_path}")
    print("Done.")


if __name__ == "__main__":
    main()

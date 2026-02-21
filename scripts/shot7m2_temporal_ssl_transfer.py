"""
SHOT7M2 Temporal SSL Transfer

Pipeline:
1. Build aligned 14-joint temporal windows from SHOT7M2 shooting frames.
2. Build aligned 14-joint temporal windows from our unlabeled train+test shots.
3. Pretrain a masked temporal autoencoder (SSL) on combined windows.
4. Extract latent embeddings for target windows (angle/depth/left_right frames).
5. Evaluate honest per-player LOO with hoop features vs hoop+SSL.
6. Optionally generate submission candidates blended with a base submission.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
SHOT7M2_DIR = PROJECT_DIR / "external_data" / "shot7m2_sample"

TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
HOOP = np.array([5.25, -25.0, 10.0], dtype=float)

# 14-joint mapping used throughout SHOT7M2 transfer work
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


def parse_triplet_int(s: str) -> tuple[int, int, int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 integers, got: {s}")
    return vals[0], vals[1], vals[2]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHOT7M2 temporal SSL transfer")
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260215)
    p.add_argument("--shot-threshold", type=float, default=0.3)
    p.add_argument("--context-frames", type=int, default=30)
    p.add_argument("--shot-windows-per-scale", type=int, default=1500)
    p.add_argument("--axis-perm", type=str, default="0,2,1")
    p.add_argument("--axis-signs", type=str, default="1,1,1")
    p.add_argument("--window-radius", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--ssl-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mask-ratio", type=float, default=0.35)
    p.add_argument("--velocity-loss-weight", type=float, default=0.25)
    p.add_argument("--weight-dribble", type=float, default=1.0)
    p.add_argument("--weight-move", type=float, default=1.0)
    p.add_argument("--weight-sprint", type=float, default=1.0)
    p.add_argument("--weight-temperature", type=float, default=0.3)
    p.add_argument("--min-shot-weight", type=float, default=0.05)
    p.add_argument("--n-pls-hoop", type=int, default=15)
    p.add_argument("--n-pls-ssl-grid", type=str, default="1,3,5")
    p.add_argument("--bw-quantile", type=float, default=0.45)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--skip-submissions", action="store_true")
    p.add_argument("--base-submission", type=int, default=2503)
    p.add_argument("--blend-weights", type=str, default="0.05,0.10,0.15,0.20,0.30")
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


def parse_json_safe_value(s: str, frame: int) -> float:
    s = str(s).replace("nan", "null").replace("NaN", "null")
    arr = json.loads(s)
    v = arr[frame]
    if v is None:
        for offset in range(1, 10):
            if frame - offset >= 0 and arr[frame - offset] is not None:
                return float(arr[frame - offset])
            if frame + offset < len(arr) and arr[frame + offset] is not None:
                return float(arr[frame + offset])
        return 0.0
    return float(v)


def parse_json_array_240(s: str) -> np.ndarray:
    s = str(s).replace("nan", "null").replace("NaN", "null")
    arr = json.loads(s)
    out = np.zeros(240, dtype=float)
    for i in range(min(len(arr), 240)):
        v = arr[i]
        if v is None:
            continue
        out[i] = float(v)
    return out


def apply_axis_transform(
    positions: np.ndarray,
    perm: tuple[int, int, int],
    signs: tuple[int, int, int],
) -> np.ndarray:
    x = positions[:, :, perm]
    return x * np.asarray(signs, dtype=float)[np.newaxis, np.newaxis, :]


def normalize_pose_frames(positions: np.ndarray) -> np.ndarray:
    """
    positions: (T, 14, 3)
    normalize each frame by pelvis center and torso length.
    """
    pelvis_idx = MAPPED_JOINTS.index("mid_hip")
    neck_idx = MAPPED_JOINTS.index("neck")

    centered = positions - positions[:, pelvis_idx : pelvis_idx + 1, :]
    torso = np.linalg.norm(centered[:, neck_idx] - centered[:, pelvis_idx], axis=1, keepdims=True)
    torso = np.maximum(torso, 1e-6)
    return centered / torso[:, :, np.newaxis]


def extract_window_flat(
    seq_14x3: np.ndarray,
    center: int,
    radius: int,
) -> np.ndarray:
    """
    seq_14x3: (T, 14, 3)
    returns: (L, 42) where L=2*radius+1, with edge padding.
    """
    l = 2 * radius + 1
    out = np.zeros((l, 14, 3), dtype=float)
    t = seq_14x3.shape[0]
    for i, rel in enumerate(range(-radius, radius + 1)):
        idx = center + rel
        if idx < 0:
            idx = 0
        elif idx >= t:
            idx = t - 1
        out[i] = seq_14x3[idx]
    out = normalize_pose_frames(out)
    return out.reshape(l, -1)


def load_competition_14j_sequences(
    df: pd.DataFrame,
    axis_perm: tuple[int, int, int],
    axis_signs: tuple[int, int, int],
) -> np.ndarray:
    """
    returns shape (N, 240, 14, 3)
    """
    n = len(df)
    seqs = np.zeros((n, 240, 14, 3), dtype=float)
    for j_idx, name in enumerate(MAPPED_JOINTS):
        x_col = f"{name}_x"
        y_col = f"{name}_y"
        z_col = f"{name}_z"
        x_vals = df[x_col].apply(parse_json_array_240).values
        y_vals = df[y_col].apply(parse_json_array_240).values
        z_vals = df[z_col].apply(parse_json_array_240).values
        for i in range(n):
            seqs[i, :, j_idx, 0] = x_vals[i]
            seqs[i, :, j_idx, 1] = y_vals[i]
            seqs[i, :, j_idx, 2] = z_vals[i]

    # apply axis transform sample-wise
    seqs_flat = seqs.reshape(-1, 14, 3)
    seqs_flat = apply_axis_transform(seqs_flat, axis_perm, axis_signs)
    return seqs_flat.reshape(n, 240, 14, 3)


def collect_shot7m2_windows(args: argparse.Namespace, radius: int) -> tuple[np.ndarray, dict]:
    poses = np.load(SHOT7M2_DIR / "train" / "train_dictionary_poses.npy", allow_pickle=True).item()
    actions = np.load(SHOT7M2_DIR / "train" / "train_dictionary_actions.npy", allow_pickle=True).item()

    labels = actions["label_array"]
    vocab = actions["vocabulary"]
    fnm = actions["frame_number_map"]

    shoot_idx = vocab.index("action_Shoot")
    dribble_idx = vocab.index("action_Dribble")
    move_idx = vocab.index("action_Move")
    sprint_idx = vocab.index("action_Sprint")

    windows = []
    weights = []
    confs = []
    mobilities = []

    for key, seq in poses["sequences"]["keypoints"].items():
        start, end = fnm[key]
        shoot = labels[shoot_idx, start:end]
        drib = labels[dribble_idx, start:end]
        move = labels[move_idx, start:end]
        sprint = labels[sprint_idx, start:end]
        shoot_frames = np.where(shoot > args.shot_threshold)[0]
        if len(shoot_frames) == 0:
            continue

        ep_poses = seq[:, 0, SHOT7M2_INDICES, :]  # (1800, 14, 3)
        for f in shoot_frames:
            w = extract_window_flat(ep_poses, int(f), radius)

            c0 = max(0, f - args.context_frames)
            c1 = min(len(shoot), f + args.context_frames + 1)
            d_mean = float(np.clip(np.mean(drib[c0:c1]), 0.0, 1.0))
            m_mean = float(np.clip(np.mean(move[c0:c1]), 0.0, 1.0))
            s_mean = float(np.clip(np.mean(sprint[c0:c1]), 0.0, 1.0))
            mobility = (
                args.weight_dribble * d_mean
                + args.weight_move * m_mean
                + args.weight_sprint * s_mean
            )
            ft_like = math.exp(-mobility / max(args.weight_temperature, 1e-8))
            conf = max(float(shoot[f]), 0.0)
            frame_weight = conf * (args.min_shot_weight + (1.0 - args.min_shot_weight) * ft_like)

            windows.append(w)
            weights.append(frame_weight)
            confs.append(conf)
            mobilities.append(mobility)

    windows = np.asarray(windows, dtype=float)
    weights = np.asarray(weights, dtype=float)
    confs = np.asarray(confs, dtype=float)
    mobilities = np.asarray(mobilities, dtype=float)

    max_keep = min(len(windows), args.scale * args.shot_windows_per_scale)
    order = np.argsort(-weights)
    keep = order[:max_keep]
    kept_windows = windows[keep]
    kept_weights = weights[keep]
    kept_conf = confs[keep]
    kept_mobility = mobilities[keep]

    stats = {
        "total_shoot_frames": int(len(windows)),
        "kept_shot_windows": int(len(kept_windows)),
        "kept_ratio": float(len(kept_windows) / max(len(windows), 1)),
        "weight_mean_all": float(weights.mean()) if len(weights) else 0.0,
        "weight_mean_kept": float(kept_weights.mean()) if len(kept_weights) else 0.0,
        "conf_mean_kept": float(kept_conf.mean()) if len(kept_conf) else 0.0,
        "mobility_mean_kept": float(kept_mobility.mean()) if len(kept_mobility) else 0.0,
    }
    return kept_windows, stats


def collect_competition_ssl_windows(
    seqs_train: np.ndarray,
    seqs_test: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, dict]:
    centers = [150, 153, 170]
    all_seqs = np.concatenate([seqs_train, seqs_test], axis=0)
    windows = []
    for i in range(len(all_seqs)):
        seq = all_seqs[i]
        for c in centers:
            windows.append(extract_window_flat(seq, c, radius))
    windows = np.asarray(windows, dtype=float)
    stats = {
        "competition_ssl_windows": int(len(windows)),
        "competition_ssl_shots_total": int(len(all_seqs)),
    }
    return windows, stats


class TemporalMaskedAE(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, seq_len * input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x)
        pooled = h.mean(dim=1)
        return self.to_latent(pooled)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decoder(z).view(-1, self.seq_len, self.input_dim)
        return recon, z


def train_ssl(
    windows: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TemporalMaskedAE, np.ndarray, np.ndarray, list[float]]:
    """
    windows: (N, L, D)
    returns model, feature_mean, feature_std, epoch_losses
    """
    n, l, d = windows.shape
    f_mean = windows.reshape(-1, d).mean(axis=0)
    f_std = windows.reshape(-1, d).std(axis=0)
    f_std = np.maximum(f_std, 1e-6)

    x_all = (windows - f_mean[np.newaxis, np.newaxis, :]) / f_std[np.newaxis, np.newaxis, :]
    model = TemporalMaskedAE(input_dim=d, seq_len=l, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    losses = []
    batch_size = args.batch_size
    model.train()
    rng = np.random.RandomState(args.seed)

    for epoch in range(args.ssl_epochs):
        idx = rng.permutation(n)
        total_loss = 0.0
        total_count = 0
        for s in range(0, n, batch_size):
            b_idx = idx[s : s + batch_size]
            xb_np = x_all[b_idx]
            xb = torch.tensor(xb_np, dtype=torch.float32, device=device)

            mask_frames = (torch.rand((xb.shape[0], xb.shape[1]), device=device) < args.mask_ratio)
            mask = mask_frames.unsqueeze(-1).expand(-1, -1, xb.shape[2])

            x_masked = xb.clone()
            x_masked[mask] = 0.0

            recon, _ = model(x_masked)
            mask_f = mask.float()
            denom = torch.clamp(mask_f.sum(), min=1.0)
            loss_recon = ((recon - xb) ** 2 * mask_f).sum() / denom

            # Velocity reconstruction consistency around masked frames
            vel_true = xb[:, 1:] - xb[:, :-1]
            vel_pred = recon[:, 1:] - recon[:, :-1]
            vel_mask = (mask_frames[:, 1:] | mask_frames[:, :-1]).unsqueeze(-1).float()
            vel_denom = torch.clamp(vel_mask.sum(), min=1.0)
            loss_vel = ((vel_pred - vel_true) ** 2 * vel_mask).sum() / vel_denom

            loss = loss_recon + args.velocity_loss_weight * loss_vel
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item()) * len(b_idx)
            total_count += len(b_idx)

        epoch_loss = total_loss / max(total_count, 1)
        losses.append(epoch_loss)
        print(f"  SSL epoch {epoch+1}/{args.ssl_epochs}: loss={epoch_loss:.12f}")

    return model, f_mean, f_std, losses


def encode_windows(
    model: TemporalMaskedAE,
    windows: np.ndarray,
    f_mean: np.ndarray,
    f_std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    x = (windows - f_mean[np.newaxis, np.newaxis, :]) / f_std[np.newaxis, np.newaxis, :]
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            xb = torch.tensor(x[s : s + batch_size], dtype=torch.float32, device=device)
            z = model.encode(xb)
            out.append(z.cpu().numpy())
    return np.vstack(out)


def build_hoop_features(df: pd.DataFrame, frame: int) -> np.ndarray:
    n = len(df)
    kp_names = sorted(
        set(c.rsplit("_", 1)[0] for c in df.columns if c.endswith(("_x", "_y", "_z")))
    )
    positions = np.zeros((n, len(kp_names), 3), dtype=float)
    for k_idx, kp in enumerate(kp_names):
        for c_idx, coord in enumerate(["x", "y", "z"]):
            col = f"{kp}_{coord}"
            positions[:, k_idx, c_idx] = df[col].apply(lambda v: parse_json_safe_value(v, frame)).values
    hoop_rel = positions - HOOP[np.newaxis, np.newaxis, :]
    return hoop_rel.reshape(n, -1)


def run_per_player_loo(
    hoop_features: np.ndarray,
    ssl_features: np.ndarray | None,
    y: np.ndarray,
    player_ids: np.ndarray,
    use_ssl: bool,
    n_pls_hoop: int,
    n_pls_ssl: int,
    bw_quantile: float,
    alpha: float,
) -> np.ndarray:
    preds = np.zeros(len(y), dtype=float)
    for pid in sorted(np.unique(player_ids)):
        mask = player_ids == pid
        idx = np.where(mask)[0]
        xh = hoop_features[mask]
        ys = y[mask]
        xs = ssl_features[mask] if (use_ssl and ssl_features is not None) else None

        for i_local, i_global in enumerate(idx):
            loo = np.ones(len(idx), dtype=bool)
            loo[i_local] = False
            xh_tr, xh_te = xh[loo], xh[~loo]
            y_tr = ys[loo]

            sh = StandardScaler()
            xh_tr_s = sh.fit_transform(xh_tr)
            xh_te_s = sh.transform(xh_te)

            parts_tr = [xh_tr_s]
            parts_te = [xh_te_s]

            nph = min(n_pls_hoop, xh_tr_s.shape[1], xh_tr_s.shape[0] - 1)
            if nph > 0:
                plsh = PLSRegression(n_components=nph)
                plsh.fit(xh_tr_s, y_tr)
                parts_tr.append(plsh.transform(xh_tr_s))
                parts_te.append(plsh.transform(xh_te_s))

            if use_ssl and xs is not None:
                xs_tr, xs_te = xs[loo], xs[~loo]
                ss = StandardScaler()
                xs_tr_s = ss.fit_transform(xs_tr)
                xs_te_s = ss.transform(xs_te)
                nps = min(n_pls_ssl, xs_tr_s.shape[1], xs_tr_s.shape[0] - 1)
                if nps > 0:
                    plss = PLSRegression(n_components=nps)
                    plss.fit(xs_tr_s, y_tr)
                    parts_tr.append(plss.transform(xs_tr_s))
                    parts_te.append(plss.transform(xs_te_s))

            x_tr = np.hstack(parts_tr)
            x_te = np.hstack(parts_te)
            d = np.linalg.norm(x_tr - x_te, axis=1)
            bw = np.quantile(d, bw_quantile)
            w = np.exp(-0.5 * (d / max(bw, 1e-8)) ** 2)

            wm = np.diag(np.sqrt(w))
            ridge = Ridge(alpha=alpha)
            ridge.fit(wm @ x_tr, wm @ y_tr)
            preds[i_global] = ridge.predict(x_te)[0]
    return preds


def predict_test_target(
    xh_train: np.ndarray,
    xs_train: np.ndarray,
    y_train: np.ndarray,
    train_pids: np.ndarray,
    xh_test: np.ndarray,
    xs_test: np.ndarray,
    test_pids: np.ndarray,
    n_pls_hoop: int,
    n_pls_ssl: int,
    bw_quantile: float,
    alpha: float,
) -> np.ndarray:
    preds = np.zeros(len(xh_test), dtype=float)
    for pid in sorted(np.unique(train_pids)):
        tr_mask = train_pids == pid
        te_mask = test_pids == pid
        te_idx = np.where(te_mask)[0]
        if len(te_idx) == 0:
            continue

        xh = xh_train[tr_mask]
        xs = xs_train[tr_mask]
        y = y_train[tr_mask]
        xh_te = xh_test[te_mask]
        xs_te = xs_test[te_mask]

        sh = StandardScaler()
        xh_s = sh.fit_transform(xh)
        xh_te_s = sh.transform(xh_te)
        ss = StandardScaler()
        xs_s = ss.fit_transform(xs)
        xs_te_s = ss.transform(xs_te)

        parts_tr = [xh_s]
        parts_te = [xh_te_s]

        nph = min(n_pls_hoop, xh_s.shape[1], xh_s.shape[0] - 1)
        if nph > 0:
            plsh = PLSRegression(n_components=nph)
            plsh.fit(xh_s, y)
            parts_tr.append(plsh.transform(xh_s))
            parts_te.append(plsh.transform(xh_te_s))

        nps = min(n_pls_ssl, xs_s.shape[1], xs_s.shape[0] - 1)
        if nps > 0:
            plss = PLSRegression(n_components=nps)
            plss.fit(xs_s, y)
            parts_tr.append(plss.transform(xs_s))
            parts_te.append(plss.transform(xs_te_s))

        x_tr = np.hstack(parts_tr)
        x_te = np.hstack(parts_te)
        for i in range(len(x_te)):
            d = np.linalg.norm(x_tr - x_te[i], axis=1)
            bw = np.quantile(d, bw_quantile)
            w = np.exp(-0.5 * (d / max(bw, 1e-8)) ** 2)
            wm = np.diag(np.sqrt(w))
            ridge = Ridge(alpha=alpha)
            ridge.fit(wm @ x_tr, wm @ y)
            preds[te_idx[i]] = ridge.predict(x_te[i : i + 1])[0]
    return preds


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    axis_perm = parse_triplet_int(args.axis_perm)
    axis_signs = parse_triplet_int(args.axis_signs)
    n_pls_ssl_grid = parse_int_list(args.n_pls_ssl_grid)
    blend_weights = parse_float_list(args.blend_weights)
    radius = args.window_radius
    seq_len = 2 * radius + 1
    device = pick_device(args.device)

    print("=" * 80)
    print("SHOT7M2 TEMPORAL SSL TRANSFER")
    print("=" * 80)
    print(f"scale={args.scale}, seed={args.seed}, device={device}")
    print(f"axis_perm={axis_perm}, axis_signs={axis_signs}")
    print(f"seq_len={seq_len}, latent_dim={args.latent_dim}, ssl_epochs={args.ssl_epochs}")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_pids = train_df["participant_id"].values
    test_pids = test_df["participant_id"].values

    print("\nBuilding 14-joint temporal sequences for competition data...")
    seqs_train = load_competition_14j_sequences(train_df, axis_perm, axis_signs)
    seqs_test = load_competition_14j_sequences(test_df, axis_perm, axis_signs)

    print("Collecting SHOT7M2 weighted windows...")
    shot_windows, shot_stats = collect_shot7m2_windows(args, radius)
    print(f"  total_shoot_frames={shot_stats['total_shoot_frames']}")
    print(f"  kept_shot_windows={shot_stats['kept_shot_windows']}")
    print(f"  kept_ratio={shot_stats['kept_ratio']:.12f}")
    print(f"  conf_mean_kept={shot_stats['conf_mean_kept']:.12f}")
    print(f"  mobility_mean_kept={shot_stats['mobility_mean_kept']:.12f}")

    print("Collecting competition unlabeled windows...")
    comp_windows, comp_stats = collect_competition_ssl_windows(seqs_train, seqs_test, radius)
    print(f"  competition_ssl_windows={comp_stats['competition_ssl_windows']}")

    ssl_windows = np.concatenate([shot_windows, comp_windows], axis=0)
    print(f"  ssl_training_windows_total={len(ssl_windows)}")

    print("\nTraining temporal SSL encoder...")
    t_ssl = time.time()
    model, f_mean, f_std, ssl_losses = train_ssl(ssl_windows, args, device)
    ssl_train_time = time.time() - t_ssl
    print(f"  ssl_train_time_s={ssl_train_time:.12f}")

    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    scaled_targets = {
        t: scalers[t].transform(train_df[t].values.reshape(-1, 1)).ravel() for t in TARGETS
    }

    train_hoop = {}
    test_hoop = {}
    train_ssl_feats = {}
    test_ssl_feats = {}

    print("\nExtracting hoop + SSL embedding features per target...")
    for t in TARGETS:
        frame = TARGET_FRAMES[t]
        train_hoop[t] = build_hoop_features(train_df, frame)
        test_hoop[t] = build_hoop_features(test_df, frame)

        tr_w = np.asarray([extract_window_flat(seqs_train[i], frame, radius) for i in range(len(seqs_train))])
        te_w = np.asarray([extract_window_flat(seqs_test[i], frame, radius) for i in range(len(seqs_test))])
        train_ssl_feats[t] = encode_windows(model, tr_w, f_mean, f_std, args.batch_size, device)
        test_ssl_feats[t] = encode_windows(model, te_w, f_mean, f_std, args.batch_size, device)
        print(
            f"  {t}: hoop={train_hoop[t].shape}, ssl={train_ssl_feats[t].shape}"
        )

    print("\n" + "=" * 80)
    print("HONEST LOO EVALUATION")
    print("=" * 80)
    configs: list[tuple[str, bool, int]] = [("baseline_hoop_only", False, 0)]
    for n in n_pls_ssl_grid:
        configs.append((f"hoop_plus_temporal_ssl_{n}pls", True, n))

    all_results: dict[str, dict] = {}
    for name, use_ssl, n_pls_ssl in configs:
        t0 = time.time()
        vals = {}
        print(f"\nConfig: {name}")
        for t in TARGETS:
            preds = run_per_player_loo(
                hoop_features=train_hoop[t],
                ssl_features=train_ssl_feats[t] if use_ssl else None,
                y=scaled_targets[t],
                player_ids=train_pids,
                use_ssl=use_ssl,
                n_pls_hoop=args.n_pls_hoop,
                n_pls_ssl=n_pls_ssl,
                bw_quantile=args.bw_quantile,
                alpha=args.alpha,
            )
            mse = float(np.mean((preds - scaled_targets[t]) ** 2))
            vals[t] = mse
            print(f"  {t}: {mse:.12f}")
        vals["mean"] = float(np.mean([vals[t] for t in TARGETS]))
        vals["time_s"] = time.time() - t0
        all_results[name] = vals
        print(f"  mean: {vals['mean']:.12f} [{vals['time_s']:.3f}s]")

    baseline = all_results["baseline_hoop_only"]["mean"]
    print("\nComparison vs baseline:")
    for name in all_results:
        delta_pct = (all_results[name]["mean"] - baseline) / baseline * 100.0
        all_results[name]["delta_pct"] = float(delta_pct)
        print(f"  {name}: mean={all_results[name]['mean']:.12f}, delta_pct={delta_pct:+.12f}")

    best_name = min(
        [k for k in all_results.keys() if k != "baseline_hoop_only"],
        key=lambda k: all_results[k]["mean"],
    )
    best_n_pls_ssl = int(best_name.split("_")[-1].replace("pls", ""))
    print(f"\nBest config: {best_name} (n_pls_ssl={best_n_pls_ssl})")

    submissions_created = []
    if not args.skip_submissions:
        base_path = SUBMISSION_DIR / f"submission_{args.base_submission}.csv"
        if not base_path.exists():
            raise FileNotFoundError(f"Base submission not found: {base_path}")
        base_sub = pd.read_csv(base_path)
        standalone = base_sub.copy()

        print("\n" + "=" * 80)
        print("GENERATING SUBMISSIONS")
        print("=" * 80)
        for t, col in zip(TARGETS, TARGET_COLS):
            preds = predict_test_target(
                xh_train=train_hoop[t],
                xs_train=train_ssl_feats[t],
                y_train=scaled_targets[t],
                train_pids=train_pids,
                xh_test=test_hoop[t],
                xs_test=test_ssl_feats[t],
                test_pids=test_pids,
                n_pls_hoop=args.n_pls_hoop,
                n_pls_ssl=best_n_pls_ssl,
                bw_quantile=args.bw_quantile,
                alpha=args.alpha,
            )
            standalone[col] = np.clip(preds, 0.0, 1.0)

        sub_num = get_next_submission_number()
        standalone_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        standalone.to_csv(standalone_path, index=False)
        submissions_created.append(
            {
                "submission_number": sub_num,
                "path": str(standalone_path),
                "type": "standalone_temporal_ssl",
                "blend_weight_ssl": 1.0,
            }
        )
        print(f"  standalone: submission_{sub_num}.csv")

        for bw in blend_weights:
            blend = base_sub.copy()
            for col in TARGET_COLS:
                blend[col] = np.clip(
                    bw * standalone[col].values + (1.0 - bw) * base_sub[col].values,
                    0.0,
                    1.0,
                )
            bnum = get_next_submission_number()
            bpath = SUBMISSION_DIR / f"submission_{bnum}.csv"
            blend.to_csv(bpath, index=False)
            submissions_created.append(
                {
                    "submission_number": bnum,
                    "path": str(bpath),
                    "type": "blend_with_base",
                    "blend_weight_ssl": float(bw),
                    "blend_weight_base": float(1.0 - bw),
                    "base_submission": int(args.base_submission),
                }
            )
            print(f"  blend: submission_{bnum}.csv (ssl={bw:.6f}, base={1.0-bw:.6f})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_json = OUTPUT_DIR / f"shot7m2_temporal_ssl_transfer_run_{ts}.json"
    run_md = OUTPUT_DIR / f"shot7m2_temporal_ssl_transfer_details_{ts}.md"

    payload = {
        "timestamp": ts,
        "command": " ".join(sys.argv),
        "config": {
            "scale": args.scale,
            "seed": args.seed,
            "shot_threshold": args.shot_threshold,
            "context_frames": args.context_frames,
            "shot_windows_per_scale": args.shot_windows_per_scale,
            "axis_perm": axis_perm,
            "axis_signs": axis_signs,
            "window_radius": args.window_radius,
            "seq_len": seq_len,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "ssl_epochs": args.ssl_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "mask_ratio": args.mask_ratio,
            "velocity_loss_weight": args.velocity_loss_weight,
            "weight_dribble": args.weight_dribble,
            "weight_move": args.weight_move,
            "weight_sprint": args.weight_sprint,
            "weight_temperature": args.weight_temperature,
            "min_shot_weight": args.min_shot_weight,
            "n_pls_hoop": args.n_pls_hoop,
            "n_pls_ssl_grid": n_pls_ssl_grid,
            "bw_quantile": args.bw_quantile,
            "alpha": args.alpha,
            "device": str(device),
            "skip_submissions": args.skip_submissions,
            "base_submission": args.base_submission,
            "blend_weights": blend_weights,
        },
        "shot7m2_stats": shot_stats,
        "competition_ssl_stats": comp_stats,
        "ssl_training_windows_total": int(len(ssl_windows)),
        "ssl_losses": [float(x) for x in ssl_losses],
        "ssl_train_time_s": float(ssl_train_time),
        "results": all_results,
        "best_config_name": best_name,
        "best_n_pls_ssl": best_n_pls_ssl,
        "submissions_created": submissions_created,
    }
    run_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# SHOT7M2 Temporal SSL Transfer Run",
        "",
        f"- Timestamp: `{ts}`",
        f"- Command: `{payload['command']}`",
        "",
        "## SHOT7M2 Window Stats",
    ]
    for k, v in shot_stats.items():
        lines.append(f"- {k}: `{v}`")
    lines += [
        "",
        "## Competition SSL Window Stats",
    ]
    for k, v in comp_stats.items():
        lines.append(f"- {k}: `{v}`")
    lines += [
        "",
        "## SSL Training",
        f"- ssl_training_windows_total: `{len(ssl_windows)}`",
        f"- ssl_train_time_s: `{ssl_train_time}`",
        f"- ssl_losses: `{ssl_losses}`",
        "",
        "## LOO Results",
    ]
    for cfg, vals in all_results.items():
        lines.append(
            f"- {cfg}: angle={vals['angle']}, depth={vals['depth']}, "
            f"left_right={vals['left_right']}, mean={vals['mean']}, delta_pct={vals['delta_pct']}"
        )
    lines += [
        "",
        "## Best Config",
        f"- name: `{best_name}`",
        f"- n_pls_ssl: `{best_n_pls_ssl}`",
        "",
        "## Submissions",
    ]
    if submissions_created:
        for x in submissions_created:
            lines.append(f"- {x}")
    else:
        lines.append("- None (`--skip-submissions` used)")
    run_md.write_text("\n".join(lines) + "\n")

    print("\nArtifacts:")
    print(f"  {run_json}")
    print(f"  {run_md}")
    print("\nDone.")


if __name__ == "__main__":
    main()

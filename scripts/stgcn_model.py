"""
ST-GCN (Spatial-Temporal Graph Convolutional Network) for basketball shot prediction.

Architecture:
- Skeleton graph adjacency matrix built from 69 keypoint connectivity
- ST-GCN blocks: spatial graph convolution + temporal 1D convolution
- Adaptive graph learning (data-driven edge importance)
- Global average pooling + player embedding + prediction head
- Per-target models, multi-seed ensemble (5 seeds)

Key insight: treats body as a GRAPH, not flat features. Connected joints
(shoulder->elbow->wrist) share information through graph convolution,
capturing biomechanical relationships that flat feature extraction misses.
"""

import json
import fcntl
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from pathlib import Path
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]

# ============================================================
# Skeleton Graph Definition (69 keypoints)
# ============================================================

# Keypoint indices (from column ordering in data)
KP_NAMES = [
    'nose',                        # 0
    'left_eye',                    # 1
    'right_eye',                   # 2
    'left_ear',                    # 3
    'right_ear',                   # 4
    'left_shoulder',               # 5
    'right_shoulder',              # 6
    'left_elbow',                  # 7
    'right_elbow',                 # 8
    'left_wrist',                  # 9
    'right_wrist',                 # 10
    'left_hip',                    # 11
    'right_hip',                   # 12
    'left_knee',                   # 13
    'right_knee',                  # 14
    'left_ankle',                  # 15
    'right_ankle',                 # 16
    'left_big_toe',                # 17
    'left_small_toe',              # 18
    'left_heel',                   # 19
    'right_big_toe',               # 20
    'right_small_toe',             # 21
    'right_heel',                  # 22
    'left_first_finger_cmc',       # 23
    'left_first_finger_mcp',       # 24
    'left_first_finger_ip',        # 25
    'left_first_finger_distal',    # 26
    'left_second_finger_mcp',      # 27
    'left_second_finger_pip',      # 28
    'left_second_finger_dip',      # 29
    'left_second_finger_distal',   # 30
    'left_third_finger_mcp',       # 31
    'left_third_finger_pip',       # 32
    'left_third_finger_dip',       # 33
    'left_third_finger_distal',    # 34
    'left_fourth_finger_mcp',      # 35
    'left_fourth_finger_pip',      # 36
    'left_fourth_finger_dip',      # 37
    'left_fourth_finger_distal',   # 38
    'left_fifth_finger_mcp',       # 39
    'left_fifth_finger_pip',       # 40
    'left_fifth_finger_dip',       # 41
    'left_fifth_finger_distal',    # 42
    'right_first_finger_cmc',      # 43
    'right_first_finger_mcp',      # 44
    'right_first_finger_ip',       # 45
    'right_first_finger_distal',   # 46
    'right_second_finger_mcp',     # 47
    'right_second_finger_pip',     # 48
    'right_second_finger_dip',     # 49
    'right_second_finger_distal',  # 50
    'right_third_finger_mcp',      # 51
    'right_third_finger_pip',      # 52
    'right_third_finger_dip',      # 53
    'right_third_finger_distal',   # 54
    'right_fourth_finger_mcp',     # 55
    'right_fourth_finger_pip',     # 56
    'right_fourth_finger_dip',     # 57
    'right_fourth_finger_distal',  # 58
    'right_fifth_finger_mcp',      # 59
    'right_fifth_finger_pip',      # 60
    'right_fifth_finger_dip',      # 61
    'right_fifth_finger_distal',   # 62
    'left_thumb',                  # 63
    'left_pinky',                  # 64
    'right_thumb',                 # 65
    'right_pinky',                 # 66
    'mid_hip',                     # 67
    'neck',                        # 68
]

NUM_NODES = 69


def build_skeleton_edges():
    """Build skeleton graph edges from anatomical connectivity."""
    edges = []

    # --- Spine / torso ---
    edges.append((67, 68))   # mid_hip -> neck
    edges.append((68, 0))    # neck -> nose
    edges.append((67, 11))   # mid_hip -> left_hip
    edges.append((67, 12))   # mid_hip -> right_hip
    edges.append((68, 5))    # neck -> left_shoulder
    edges.append((68, 6))    # neck -> right_shoulder

    # --- Head ---
    edges.append((0, 1))     # nose -> left_eye
    edges.append((0, 2))     # nose -> right_eye
    edges.append((1, 3))     # left_eye -> left_ear
    edges.append((2, 4))     # right_eye -> right_ear

    # --- Left arm ---
    edges.append((5, 7))     # left_shoulder -> left_elbow
    edges.append((7, 9))     # left_elbow -> left_wrist

    # --- Right arm ---
    edges.append((6, 8))     # right_shoulder -> right_elbow
    edges.append((8, 10))    # right_elbow -> right_wrist

    # --- Left leg ---
    edges.append((11, 13))   # left_hip -> left_knee
    edges.append((13, 15))   # left_knee -> left_ankle
    edges.append((15, 17))   # left_ankle -> left_big_toe
    edges.append((15, 18))   # left_ankle -> left_small_toe
    edges.append((15, 19))   # left_ankle -> left_heel

    # --- Right leg ---
    edges.append((12, 14))   # right_hip -> right_knee
    edges.append((14, 16))   # right_knee -> right_ankle
    edges.append((16, 20))   # right_ankle -> right_big_toe
    edges.append((16, 21))   # right_ankle -> right_small_toe
    edges.append((16, 22))   # right_ankle -> right_heel

    # --- Left hand (from wrist=9) ---
    # Thumb chain
    edges.append((9, 23))    # left_wrist -> left_first_finger_cmc
    edges.append((23, 24))   # cmc -> mcp
    edges.append((24, 25))   # mcp -> ip
    edges.append((25, 26))   # ip -> distal
    edges.append((26, 63))   # distal -> left_thumb (tip)

    # Index finger (second)
    edges.append((9, 27))    # left_wrist -> left_second_finger_mcp
    edges.append((27, 28))   # mcp -> pip
    edges.append((28, 29))   # pip -> dip
    edges.append((29, 30))   # dip -> distal

    # Middle finger (third)
    edges.append((9, 31))    # left_wrist -> left_third_finger_mcp
    edges.append((31, 32))   # mcp -> pip
    edges.append((32, 33))   # pip -> dip
    edges.append((33, 34))   # dip -> distal

    # Ring finger (fourth)
    edges.append((9, 35))    # left_wrist -> left_fourth_finger_mcp
    edges.append((35, 36))   # mcp -> pip
    edges.append((36, 37))   # pip -> dip
    edges.append((37, 38))   # dip -> distal

    # Pinky finger (fifth)
    edges.append((9, 39))    # left_wrist -> left_fifth_finger_mcp
    edges.append((39, 40))   # mcp -> pip
    edges.append((40, 41))   # pip -> dip
    edges.append((41, 42))   # dip -> distal
    edges.append((42, 64))   # distal -> left_pinky (tip)

    # --- Right hand (from wrist=10) ---
    # Thumb chain
    edges.append((10, 43))   # right_wrist -> right_first_finger_cmc
    edges.append((43, 44))   # cmc -> mcp
    edges.append((44, 45))   # mcp -> ip
    edges.append((45, 46))   # ip -> distal
    edges.append((46, 65))   # distal -> right_thumb (tip)

    # Index finger (second)
    edges.append((10, 47))   # right_wrist -> right_second_finger_mcp
    edges.append((47, 48))   # mcp -> pip
    edges.append((48, 49))   # pip -> dip
    edges.append((49, 50))   # dip -> distal

    # Middle finger (third)
    edges.append((10, 51))   # right_wrist -> right_third_finger_mcp
    edges.append((51, 52))   # mcp -> pip
    edges.append((52, 53))   # pip -> dip
    edges.append((53, 54))   # dip -> distal

    # Ring finger (fourth)
    edges.append((10, 55))   # right_wrist -> right_fourth_finger_mcp
    edges.append((55, 56))   # mcp -> pip
    edges.append((56, 57))   # pip -> dip
    edges.append((57, 58))   # dip -> distal

    # Pinky finger (fifth)
    edges.append((10, 59))   # right_wrist -> right_fifth_finger_mcp
    edges.append((59, 60))   # mcp -> pip
    edges.append((60, 61))   # pip -> dip
    edges.append((61, 62))   # dip -> distal
    edges.append((62, 66))   # distal -> right_pinky (tip)

    return edges


def build_adjacency_matrix():
    """Build normalized adjacency matrix for graph convolution.

    Uses the partition strategy from ST-GCN:
    - Self-connections (identity)
    - Neighbor connections (adjacency)

    Returns: A_hat of shape (2, N, N) for 2-subset partitioning:
      A_hat[0] = identity (self)
      A_hat[1] = normalized adjacency (neighbors)
    """
    edges = build_skeleton_edges()

    # Build raw adjacency
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    for (i, j) in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Normalize: D^{-1/2} A D^{-1/2}
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.zeros_like(D)
    nonzero = D > 0
    D_inv_sqrt[nonzero] = 1.0 / np.sqrt(D[nonzero])
    D_mat = np.diag(D_inv_sqrt)
    A_norm = D_mat @ A @ D_mat

    # Identity for self-connections
    I = np.eye(NUM_NODES, dtype=np.float32)

    # Stack: (2, N, N) - partition into self and neighbors
    A_hat = np.stack([I, A_norm], axis=0)
    return A_hat


# ============================================================
# ST-GCN Model
# ============================================================

class SpatialGraphConv(nn.Module):
    """Spatial graph convolution with learnable edge importance."""

    def __init__(self, in_channels, out_channels, A_shape, residual=True):
        super().__init__()
        self.num_subsets = A_shape[0]  # 2 (self + neighbors)

        # One conv per subset
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(self.num_subsets)
        ])

        # Learnable edge importance weights
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(A_shape[1], A_shape[2]))
            for _ in range(self.num_subsets)
        ])

        self.bn = nn.BatchNorm2d(out_channels)

        if residual and in_channels == out_channels:
            self.residual = nn.Identity()
        elif residual:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = None

    def forward(self, x, A):
        """
        x: (B, C, T, N) - batch, channels, time, nodes
        A: (K, N, N) - adjacency subsets
        """
        res = self.residual(x) if self.residual is not None else 0

        out = 0
        for k in range(self.num_subsets):
            # Apply edge importance
            A_k = A[k] * self.edge_importance[k]
            # Graph convolution: x @ A_k
            # x: (B, C, T, N), A_k: (N, N)
            z = self.convs[k](x)  # (B, C_out, T, N)
            z = torch.einsum('bctn,nm->bctm', z, A_k)
            out = out + z

        out = self.bn(out)
        out = F.relu(out + res)
        return out


class TemporalConv(nn.Module):
    """Temporal convolution block."""

    def __init__(self, channels, kernel_size=9, stride=1, residual=True):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                      padding=(pad, 0), stride=(stride, 1)),
            nn.BatchNorm2d(channels)
        )

        if residual:
            if stride == 1:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1,
                              stride=(stride, 1)),
                    nn.BatchNorm2d(channels)
                )
        else:
            self.residual = None

    def forward(self, x):
        res = self.residual(x) if self.residual is not None else 0
        return F.relu(self.conv(x) + res)


class STGCNBlock(nn.Module):
    """One ST-GCN block: spatial graph conv + temporal conv + dropout."""

    def __init__(self, in_channels, out_channels, A_shape,
                 temporal_kernel=9, stride=1, dropout=0.2):
        super().__init__()
        self.sgc = SpatialGraphConv(in_channels, out_channels, A_shape,
                                     residual=(in_channels == out_channels))
        self.tc = TemporalConv(out_channels, kernel_size=temporal_kernel,
                               stride=stride)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        x = self.sgc(x, A)
        x = self.tc(x)
        x = self.dropout(x)
        return x


class STGCN(nn.Module):
    """Full ST-GCN model for basketball shot regression."""

    def __init__(self, in_channels=3, hidden_dim=20, num_blocks=3,
                 n_players=5, temporal_kernel=9, dropout=0.3):
        super().__init__()

        A_hat = build_adjacency_matrix()
        self.register_buffer('A', torch.FloatTensor(A_hat))
        A_shape = A_hat.shape  # (2, 69, 69)

        # Input batch norm
        self.input_bn = nn.BatchNorm1d(in_channels * NUM_NODES)

        # ST-GCN blocks with channel progression
        channels = [in_channels] + [hidden_dim] * num_blocks
        # Use stride=2 on blocks 1+ to reduce temporal dimension
        strides = [1] + [2] * (num_blocks - 1)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                STGCNBlock(channels[i], channels[i+1], A_shape,
                           temporal_kernel=temporal_kernel,
                           stride=strides[i], dropout=dropout)
            )

        # Player embedding
        self.player_embed = nn.Embedding(n_players + 1, 4)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 4, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x, player_ids):
        """
        x: (B, C, T, N) - batch, 3 coords, time frames, 69 nodes
        player_ids: (B,) - player indices
        """
        B, C, T, N = x.shape

        # Input normalization - reshape to (B, C*N) for BN then back
        x_flat = x.reshape(B, C * N, T).permute(0, 2, 1)  # (B, T, C*N)
        x_flat = x_flat.reshape(B * T, C * N)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(B, T, C, N).permute(0, 2, 1, 3)  # (B, C, T, N)

        # ST-GCN blocks
        for block in self.blocks:
            x = block(x, self.A)

        # Global average pooling over time and nodes
        x = x.mean(dim=[2, 3])  # (B, hidden_dim)

        # Add player embedding
        pe = self.player_embed(player_ids)
        x = torch.cat([x, pe], dim=1)

        return self.head(x).squeeze(-1)


# ============================================================
# Data Loading
# ============================================================

def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = [int(fp.stem.split('_')[1]) for fp in existing
                    if fp.stem.split('_')[1].isdigit()]
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_data():
    """Load and parse 3D keypoint data into (N, 3, T, 69) tensors."""
    print("Loading data...", flush=True)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    def process(df, is_train=True):
        n = len(df)
        # Shape: (N, 3, 240, 69) - channels first for conv
        X_3d = np.zeros((n, 3, 240, NUM_NODES), dtype=np.float32)
        ids, pids = [], []
        targets = []

        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                node_idx = col_i // 3
                coord_idx = col_i % 3
                arr = parse_array_string(row[col])
                X_3d[idx, coord_idx, :, node_idx] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

        result = {'X_3d': X_3d, 'pids': np.array(pids), 'ids': np.array(ids)}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    train_data = process(train_df, True)
    test_data = process(test_df, False)

    return train_data, test_data


def preprocess_data(X_3d, subsample_frames=80):
    """Preprocess: interpolate NaNs, subsample frames, normalize per-shot.

    Args:
        X_3d: (N, 3, 240, 69) raw data
        subsample_frames: number of frames to keep

    Returns:
        (N, 3, subsample_frames, 69) preprocessed data
    """
    N, C, T, V = X_3d.shape

    # Frame range focused on the shot (frames 60-220)
    frame_idx = np.linspace(60, 220, subsample_frames, dtype=int)

    X_out = np.zeros((N, C, subsample_frames, V), dtype=np.float32)

    for i in range(N):
        for c in range(C):
            for v in range(V):
                traj = X_3d[i, c, :, v].astype(np.float64)
                bad = np.isnan(traj) | np.isinf(traj)
                if np.all(bad):
                    traj[:] = 0.0
                elif np.any(bad):
                    good = ~bad
                    traj[bad] = np.interp(np.where(bad)[0],
                                          np.where(good)[0], traj[good])
                X_out[i, c, :, v] = traj[frame_idx]

        # Per-shot normalization: zero mean, unit variance per coordinate
        for c in range(C):
            vals = X_out[i, c, :, :]
            mu = vals.mean()
            std = vals.std()
            if std > 1e-8:
                X_out[i, c, :, :] = (vals - mu) / std

    return X_out


# ============================================================
# Training
# ============================================================

def train_model(X_train, y_train, pids_train, X_val, pids_val,
                target_idx, seed, hidden_dim=20, num_blocks=3,
                epochs=150, lr=0.002, weight_decay=0.05, dropout=0.3):
    """Train one ST-GCN model. Returns val predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    pids_tr = torch.LongTensor(pids_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    pids_v = torch.LongTensor(pids_val).to(device)

    model = STGCN(in_channels=3, hidden_dim=hidden_dim, num_blocks=num_blocks,
                  dropout=dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience_limit = 40

    for epoch in range(epochs):
        model.train()

        # Data augmentation
        X_aug = X_tr.clone()
        # Random noise
        if np.random.random() < 0.5:
            noise = torch.randn_like(X_aug) * 0.01
            X_aug = X_aug + noise
        # Random temporal shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-2, 3)
            if shift > 0:
                X_aug[:, :, shift:, :] = X_aug[:, :, :-shift, :].clone()
            elif shift < 0:
                X_aug[:, :, :shift, :] = X_aug[:, :, -shift:, :].clone()

        pred = model(X_aug, pids_tr)
        loss = F.mse_loss(pred, y_tr)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience_limit:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_v, pids_v).cpu().numpy()
    return val_pred, best_loss


def honest_loo_pilot(X_all, y_all, pids_all, target_idx, seed=42,
                     hidden_dim=20, num_blocks=3, epochs=100):
    """Run LOO on a subsample to estimate generalization error.

    For speed, run LOO on 20 random samples (quick estimate).
    """
    n = len(X_all)
    np.random.seed(seed)
    sample_idx = np.random.choice(n, min(20, n), replace=False)

    loo_preds = np.zeros(len(sample_idx))
    loo_true = np.zeros(len(sample_idx))

    for ii, test_i in enumerate(sample_idx):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_i] = False

        X_tr = X_all[train_mask]
        y_tr = y_all[train_mask]
        pids_tr = pids_all[train_mask]

        X_te = X_all[test_i:test_i+1]
        pids_te = pids_all[test_i:test_i+1]

        pred, _ = train_model(X_tr, y_tr, pids_tr, X_te, pids_te,
                              target_idx, seed=seed,
                              hidden_dim=hidden_dim, num_blocks=num_blocks,
                              epochs=epochs)
        loo_preds[ii] = np.clip(pred[0], 0, 1)
        loo_true[ii] = y_all[test_i]

        if (ii + 1) % 5 == 0:
            mse_so_far = np.mean((loo_preds[:ii+1] - loo_true[:ii+1])**2)
            print(f"    LOO {ii+1}/{len(sample_idx)}: MSE so far = {mse_so_far:.6f}", flush=True)

    mse = np.mean((loo_preds - loo_true)**2)
    return mse


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("ST-GCN MODEL - Spatial-Temporal Graph Convolution", flush=True)
    print("Skeleton graph structure for basketball shot prediction", flush=True)
    print("=" * 60, flush=True)

    # Verify skeleton graph
    edges = build_skeleton_edges()
    A_hat = build_adjacency_matrix()
    print(f"\nSkeleton graph: {NUM_NODES} nodes, {len(edges)} edges", flush=True)
    print(f"Adjacency shape: {A_hat.shape}", flush=True)
    print(f"Mean degree: {A_hat[1].sum(axis=1).mean():.2f}", flush=True)

    # Load data
    train_data, test_data = load_data()
    print(f"Train: {len(train_data['X_3d'])} shots", flush=True)
    print(f"Test: {len(test_data['X_3d'])} shots", flush=True)

    # Preprocess
    SUBSAMPLE_FRAMES = 80
    print(f"\nPreprocessing (subsample to {SUBSAMPLE_FRAMES} frames)...", flush=True)
    X_train = preprocess_data(train_data['X_3d'], SUBSAMPLE_FRAMES)
    X_test = preprocess_data(test_data['X_3d'], SUBSAMPLE_FRAMES)
    print(f"  Train shape: {X_train.shape}", flush=True)
    print(f"  Test shape: {X_test.shape}", flush=True)

    # Scale targets to [0, 1]
    y_raw = train_data['y']  # (345, 3)
    # Already [0,1] in the data? Check
    for t, tname in enumerate(TARGETS):
        print(f"  {tname}: min={y_raw[:, t].min():.4f}, max={y_raw[:, t].max():.4f}", flush=True)

    # Map player IDs
    unique_pids = np.unique(train_data['pids'])
    pid_map = {p: i for i, p in enumerate(unique_pids)}
    pids_train = np.array([pid_map[p] for p in train_data['pids']])
    pids_test = np.array([pid_map[p] for p in test_data['pids']])

    # ============================================================
    # Pilot: quick LOO on angle (1 seed, 20 samples)
    # ============================================================
    HIDDEN_DIM = 20
    NUM_BLOCKS = 3
    PILOT_EPOCHS = 100
    FULL_EPOCHS = 150

    print("\n" + "-" * 60, flush=True)
    print("PILOT: Quick LOO estimate (20 samples, angle target)", flush=True)
    print(f"Config: hidden_dim={HIDDEN_DIM}, blocks={NUM_BLOCKS}, epochs={PILOT_EPOCHS}", flush=True)
    print("-" * 60, flush=True)

    pilot_mse = honest_loo_pilot(X_train, y_raw[:, 0], pids_train,
                                  target_idx=0, seed=42,
                                  hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS,
                                  epochs=PILOT_EPOCHS)
    print(f"\nPilot LOO MSE (angle, 20 samples): {pilot_mse:.6f}", flush=True)

    if pilot_mse > 0.015:
        print("\n*** KILL CRITERION: LOO > 0.015 ***", flush=True)
        print(f"    Pilot MSE = {pilot_mse:.6f} > 0.015", flush=True)
        print("    Stopping. Model is too weak for this task.", flush=True)
        elapsed = time.time() - t0
        print(f"\nRuntime: {elapsed:.1f}s", flush=True)
        return

    print(f"  Pilot passed (MSE {pilot_mse:.6f} < 0.015). Proceeding to full run.", flush=True)

    # ============================================================
    # Full run: per-target, multi-seed ensemble
    # ============================================================
    N_SEEDS = 5
    SEEDS = [42, 7, 99, 133, 2026]

    print("\n" + "=" * 60, flush=True)
    print(f"FULL RUN: {N_SEEDS} seeds x 3 targets = {N_SEEDS * 3} models", flush=True)
    print("=" * 60, flush=True)

    all_test_preds = {t: [] for t in range(3)}

    for t, target in enumerate(TARGETS):
        print(f"\n--- Target: {target} ---", flush=True)
        for s_i, seed in enumerate(SEEDS):
            pred, best_loss = train_model(
                X_train, y_raw[:, t], pids_train,
                X_test, pids_test,
                target_idx=t, seed=seed,
                hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS,
                epochs=FULL_EPOCHS, lr=0.002, weight_decay=0.05,
                dropout=0.3
            )
            all_test_preds[t].append(pred)
            print(f"  Seed {s_i+1}/{N_SEEDS} (seed={seed}): train loss = {best_loss:.6f}", flush=True)

    # Average predictions across seeds
    test_preds = np.zeros((len(test_data['ids']), 3))
    for t in range(3):
        test_preds[:, t] = np.clip(np.mean(all_test_preds[t], axis=0), 0, 1)

    # ============================================================
    # Diversity check
    # ============================================================
    print("\n" + "-" * 60, flush=True)
    print("DIVERSITY CHECK", flush=True)
    print("-" * 60, flush=True)

    anchor = pd.read_csv(SUBMISSION_DIR / "submission_2716.csv")
    corrs = []
    for t, target in enumerate(TARGETS):
        col = f"scaled_{target}"
        r = np.corrcoef(test_preds[:, t], anchor[col].values)[0, 1]
        corrs.append(r)
        print(f"  {target}: r = {r:.4f} with Sub 2716", flush=True)
    mean_corr = np.mean(corrs)
    print(f"  Mean correlation: {mean_corr:.4f}", flush=True)

    if all(c > 0.90 for c in corrs):
        print("\n*** KILL CRITERION: All correlations > 0.90 (no diversity) ***", flush=True)
        print("    Stopping. Predictions too similar to anchor.", flush=True)
        elapsed = time.time() - t0
        print(f"\nRuntime: {elapsed:.1f}s", flush=True)
        return

    # ============================================================
    # Prediction stats
    # ============================================================
    print("\nPrediction statistics:", flush=True)
    for t, target in enumerate(TARGETS):
        vals = test_preds[:, t]
        print(f"  {target}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}", flush=True)

    # Seed diversity (how different are individual seed predictions)
    print("\nSeed diversity (std across seeds):", flush=True)
    for t, target in enumerate(TARGETS):
        preds_stack = np.array(all_test_preds[t])  # (n_seeds, n_test)
        seed_std = preds_stack.std(axis=0).mean()
        print(f"  {target}: mean seed std = {seed_std:.6f}", flush=True)

    # ============================================================
    # Save submissions
    # ============================================================
    print("\n" + "=" * 60, flush=True)
    print("GENERATING SUBMISSIONS", flush=True)
    print("=" * 60, flush=True)

    # Standalone
    sub_num = get_next_submission_number()
    sub_df = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': test_preds[:, 0],
        'scaled_depth': test_preds[:, 1],
        'scaled_left_right': test_preds[:, 2]
    })
    sub_df.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    standalone_num = sub_num
    print(f"Sub {sub_num}: ST-GCN standalone ({N_SEEDS} seeds, hidden={HIDDEN_DIM}, blocks={NUM_BLOCKS})", flush=True)

    # Blends with Sub 2716
    anchor_preds = anchor[['scaled_angle', 'scaled_depth', 'scaled_left_right']].values
    blend_subs = {}
    for w in [0.02, 0.05]:
        blend = (1 - w) * anchor_preds + w * test_preds
        blend = np.clip(blend, 0, 1)
        sn = get_next_submission_number()
        df = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': blend[:, 0],
            'scaled_depth': blend[:, 1],
            'scaled_left_right': blend[:, 2]
        })
        df.to_csv(SUBMISSION_DIR / f"submission_{sn}.csv", index=False)
        blend_subs[w] = sn
        print(f"Sub {sn}: {int(w*100)}% ST-GCN + {int((1-w)*100)}% Sub 2716", flush=True)

    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Architecture: ST-GCN ({NUM_BLOCKS} blocks, hidden_dim={HIDDEN_DIM})", flush=True)
    print(f"Input: 69 nodes x 3 coords x {SUBSAMPLE_FRAMES} frames", flush=True)
    print(f"Skeleton graph: {len(edges)} edges, {NUM_NODES} nodes", flush=True)
    print(f"Training: {FULL_EPOCHS} epochs, lr=0.002, wd=0.05, dropout=0.3", flush=True)
    print(f"Ensemble: {N_SEEDS} seeds x 3 targets", flush=True)
    print(f"Pilot LOO MSE (angle): {pilot_mse:.6f}", flush=True)
    print(f"Diversity vs Sub 2716: angle={corrs[0]:.4f}, depth={corrs[1]:.4f}, LR={corrs[2]:.4f}", flush=True)
    print(f"Submissions: standalone={standalone_num}, blends={blend_subs}", flush=True)
    print(f"Total runtime: {elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

"""
1D Temporal CNN for basketball shot prediction.

Key design:
- SMALL model (10K-50K params) to avoid overfitting on 345 samples
- HEAVY augmentation: temporal shift, Gaussian noise, keypoint dropout, mixup
- Per-target training (not multi-task)
- MPS GPU acceleration (Apple Silicon)
- Hoop-relative coordinates
- Per-player 5-fold CV with early stopping
- Diversity check vs Sub 1350 and Sub 784
"""

import json
import time
import fcntl
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048

# Use frames from the best pipeline
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}


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


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def safe_savgol(x, window, polyorder, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


# ==============================================================
# DATA LOADING (same as per_example_pipeline.py)
# ==============================================================

def load_data():
    """Load and parse all data."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}

    def process(df, is_train=True):
        n = len(df)
        n_kp = len(kp_names)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        ids, pids, targets = [], [], []

        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_3d[idx, :, col_i // 3, col_i % 3] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        result = {'X_3d': X_3d, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    return process(train_df, True), process(test_df, False)


def compute_hoop_transform(ts_3d, kp_index):
    mid_hip_idx = kp_index.get('mid_hip', 0)
    player_pos = ts_3d[120, mid_hip_idx, :].copy()
    player_pos[2] = 0

    forward = HOOP_POS[:2] - player_pos[:2]
    fn = np.linalg.norm(forward)
    if fn > 1e-6:
        forward /= fn
    else:
        forward = np.array([0.0, -1.0])
    lateral = np.array([-forward[1], forward[0]])

    R = np.eye(3, dtype=np.float32)
    R[0, 0] = forward[0]; R[0, 1] = forward[1]
    R[1, 0] = lateral[0]; R[1, 1] = lateral[1]

    centered = ts_3d - player_pos.reshape(1, 1, 3)
    return np.einsum('ij,fkj->fki', R, centered)


# ==============================================================
# CNN MODEL
# ==============================================================

class SmallTemporalCNN(nn.Module):
    """Small 1D CNN for temporal pose data.

    Input: (batch, channels=207, time=240)

    Architecture:
    - Conv1d(207->32, k=7) -> BN -> ReLU -> AvgPool(4)  => (32, 58)
    - Conv1d(32->32, k=5) -> BN -> ReLU -> AvgPool(4)   => (32, 13)
    - Conv1d(32->16, k=3) -> BN -> ReLU -> AdaptiveAvgPool(1) => (16, 1)
    - FC(16+1) -> 16 -> 1  (the +1 is for player_id embedding)

    Total params: ~25K
    """
    def __init__(self, n_channels=207, n_players=5):
        super().__init__()

        # Temporal convolutions
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.AvgPool1d(4)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.AvgPool1d(4)

        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Player embedding (small, just 4-dim)
        self.player_embed = nn.Embedding(n_players + 1, 4)  # +1 for safety

        # FC head
        self.fc1 = nn.Linear(16 + 4, 16)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x, player_id):
        # x: (B, C, T) where C=207, T=240
        # player_id: (B,) long tensor

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)  # (B, 16)

        # Player embedding
        p = self.player_embed(player_id)  # (B, 4)

        # Concat and predict
        z = torch.cat([x, p], dim=1)  # (B, 20)
        z = F.relu(self.fc1(z))
        z = self.dropout(z)
        z = self.fc2(z)  # (B, 1)
        return z.squeeze(-1)


# ==============================================================
# DATASET WITH AUGMENTATION
# ==============================================================

class ShotDataset(Dataset):
    """Dataset with heavy augmentation for training."""

    def __init__(self, X_hr, pids, y=None, augment=False, player_id_map=None):
        """
        X_hr: (N, 240, n_kp, 3) hoop-relative coordinates
        pids: (N,) participant ids
        y: (N,) scaled targets (optional for test)
        augment: whether to apply augmentation
        player_id_map: dict mapping participant_id -> integer index
        """
        self.X_hr = X_hr  # (N, 240, 69, 3)
        self.pids = pids
        self.y = y
        self.augment = augment
        self.player_id_map = player_id_map or {}

    def __len__(self):
        return len(self.X_hr)

    def __getitem__(self, idx):
        # Shape: (240, 69, 3)
        x = self.X_hr[idx].copy()
        pid = self.player_id_map.get(self.pids[idx], 0)

        if self.augment:
            x = self._augment(x)

        # Reshape to (C, T) = (207, 240) for Conv1d
        # From (240, 69, 3) -> flatten keypoints -> (240, 207) -> transpose -> (207, 240)
        x_flat = x.reshape(240, -1)  # (240, 207)
        x_tensor = torch.from_numpy(x_flat.T).float()  # (207, 240)

        pid_tensor = torch.tensor(pid, dtype=torch.long)

        if self.y is not None:
            y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_tensor, pid_tensor, y_tensor
        else:
            return x_tensor, pid_tensor

    def _augment(self, x):
        """Apply heavy augmentation to a single shot.
        x: (240, 69, 3) numpy array
        """
        # 1. Temporal shift (roll by 1-5 frames)
        if np.random.random() < 0.5:
            shift = np.random.randint(-5, 6)
            x = np.roll(x, shift, axis=0)

        # 2. Gaussian noise on keypoints
        if np.random.random() < 0.7:
            noise_std = np.random.uniform(0.01, 0.05)
            noise = np.random.randn(*x.shape).astype(np.float32) * noise_std
            # Don't add noise to NaN/zero regions
            mask = np.abs(x) > 1e-6
            x = x + noise * mask

        # 3. Random keypoint dropout (zero out entire joints)
        if np.random.random() < 0.5:
            n_drop = np.random.randint(1, 10)  # Drop 1-9 joints out of 69
            drop_idx = np.random.choice(69, n_drop, replace=False)
            x[:, drop_idx, :] = 0.0

        # 4. Temporal scaling (stretch/compress time slightly)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.95, 1.05)
            orig_len = x.shape[0]
            new_len = int(orig_len * scale)
            new_len = max(10, new_len)
            # Simple linear interpolation
            old_indices = np.linspace(0, orig_len - 1, new_len)
            new_x = np.zeros_like(x)
            target_indices = np.linspace(0, new_len - 1, orig_len)
            for kp in range(x.shape[1]):
                for coord in range(3):
                    series = x[:, kp, coord]
                    interp_series = np.interp(old_indices, np.arange(orig_len), series)
                    new_x[:, kp, coord] = np.interp(target_indices, np.arange(new_len), interp_series)
            x = new_x

        # 5. Small spatial jitter (simulate measurement noise)
        if np.random.random() < 0.3:
            # Add consistent offset to all joints (simulates slight camera shift)
            offset = np.random.randn(1, 1, 3).astype(np.float32) * 0.02
            mask = np.abs(x) > 1e-6
            x = x + offset * mask.any(axis=(0, 2), keepdims=True)

        return x


def do_mixup(x1, pid1, y1, x2, pid2, y2, alpha=0.2):
    """Mixup between two samples. Only mix within same player."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = lam * y1 + (1 - lam) * y2
    return x_mix, pid1, y_mix


# ==============================================================
# TRAINING LOOP
# ==============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_fold(X_train_hr, pids_train, y_train,
                   X_val_hr, pids_val, y_val,
                   player_id_map, device,
                   n_channels=207, n_epochs=200, lr=1e-3,
                   batch_size=32, patience=30, seed=42):
    """Train one fold and return val predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create datasets
    train_ds = ShotDataset(X_train_hr, pids_train, y_train, augment=True, player_id_map=player_id_map)
    val_ds = ShotDataset(X_val_hr, pids_val, y_val, augment=False, player_id_map=player_id_map)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    # Model
    model = SmallTemporalCNN(n_channels=n_channels, n_players=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        # Training with mixup
        model.train()
        train_losses = []
        for batch in train_loader:
            x, pid, y = batch
            x, pid, y = x.to(device), pid.to(device), y.to(device)

            # Mixup within batch (intra-player when possible)
            if len(x) > 1 and np.random.random() < 0.5:
                # Shuffle for mixup
                perm = torch.randperm(len(x))
                lam = np.random.beta(0.2, 0.2)
                x = lam * x + (1 - lam) * x[perm]
                y = lam * y + (1 - lam) * y[perm]

            pred = model(x, pid)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, pid, y = batch
                x, pid, y = x.to(device), pid.to(device), y.to(device)
                pred = model(x, pid)
                val_loss = F.mse_loss(pred, y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Load best model and predict
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x, pid = batch[0].to(device), batch[1].to(device)
            val_preds = model(x, pid).cpu().numpy()

    return val_preds, best_val_loss, model


def train_full_and_predict(X_train_hr, pids_train, y_train,
                           X_test_hr, pids_test,
                           player_id_map, device,
                           n_channels=207, n_epochs=200, lr=1e-3,
                           batch_size=32, patience=30, seed=42,
                           n_ensemble=5):
    """Train on full training set with multiple seeds and average predictions."""
    test_preds_all = []

    for s in range(n_ensemble):
        actual_seed = seed + s * 100
        torch.manual_seed(actual_seed)
        np.random.seed(actual_seed)

        train_ds = ShotDataset(X_train_hr, pids_train, y_train, augment=True, player_id_map=player_id_map)
        test_ds = ShotDataset(X_test_hr, pids_test, augment=False, player_id_map=player_id_map)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

        model = SmallTemporalCNN(n_channels=n_channels, n_players=5).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

        # Train for fixed epochs (no early stopping on full train)
        # Use ~80% of max epochs since we saw early stopping around there in CV
        actual_epochs = max(50, int(n_epochs * 0.7))

        for epoch in range(actual_epochs):
            model.train()
            for batch in train_loader:
                x, pid, y = batch
                x, pid, y = x.to(device), pid.to(device), y.to(device)

                if len(x) > 1 and np.random.random() < 0.5:
                    perm = torch.randperm(len(x))
                    lam = np.random.beta(0.2, 0.2)
                    x = lam * x + (1 - lam) * x[perm]
                    y = lam * y + (1 - lam) * y[perm]

                pred = model(x, pid)
                loss = F.mse_loss(pred, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        # Predict on test
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x, pid = batch[0].to(device), batch[1].to(device)
                test_preds = model(x, pid).cpu().numpy()

        test_preds_all.append(test_preds)

    return np.mean(test_preds_all, axis=0)


# ==============================================================
# NORMALIZATION HELPERS
# ==============================================================

def normalize_per_player(X_hr, pids):
    """Compute per-player mean/std for normalization.
    X_hr: (N, 240, 69, 3)
    Returns: normalized X_hr, stats dict
    """
    unique_pids = sorted(np.unique(pids))
    stats = {}
    X_norm = X_hr.copy()

    for pid in unique_pids:
        mask = pids == pid
        # Flatten to (N_p, 240*69*3) to get stats
        data = X_hr[mask].reshape(mask.sum(), -1)
        # Per-feature mean/std
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        std[std < 1e-6] = 1.0

        # Normalize
        normalized = (data - mean) / std
        X_norm[mask] = normalized.reshape(mask.sum(), 240, X_hr.shape[2], 3)
        stats[pid] = (mean, std)

    return X_norm, stats


def apply_normalization(X_hr, pids, stats):
    """Apply pre-computed normalization stats."""
    X_norm = X_hr.copy()
    for pid in np.unique(pids):
        mask = pids == pid
        mean, std = stats[pid]
        data = X_hr[mask].reshape(mask.sum(), -1)
        normalized = (data - mean) / std
        X_norm[mask] = normalized.reshape(mask.sum(), 240, X_hr.shape[2], 3)
    return X_norm


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("1D TEMPORAL CNN WITH HEAVY AUGMENTATION")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    # Load data
    train_data, test_data = load_data()
    y_train_raw = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']
    n_kp = len(train_data['kp_names'])

    print(f"Train: {len(pids_train)}, Test: {len(pids_test)}")
    print(f"Keypoints: {n_kp}, Channels: {n_kp * 3}")

    # Player ID mapping
    unique_pids = sorted(np.unique(pids_train))
    player_id_map = {pid: i for i, pid in enumerate(unique_pids)}
    print(f"Players: {unique_pids}")

    # Convert to hoop-relative
    print("Computing hoop-relative coordinates...")
    X_train_hr = np.zeros_like(train_data['X_3d'])
    X_test_hr = np.zeros_like(test_data['X_3d'])

    for i in range(len(pids_train)):
        X_train_hr[i] = compute_hoop_transform(train_data['X_3d'][i], kp_index)
    for i in range(len(pids_test)):
        X_test_hr[i] = compute_hoop_transform(test_data['X_3d'][i], kp_index)

    # NaN handling
    X_train_hr = np.nan_to_num(X_train_hr, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_hr = np.nan_to_num(X_test_hr, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize per player
    print("Normalizing per player...")
    X_train_norm, norm_stats = normalize_per_player(X_train_hr, pids_train)
    X_test_norm = apply_normalization(X_test_hr, pids_test, norm_stats)

    # Load scalers for targets
    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train_raw[:, target_idx[target]].reshape(-1, 1)).ravel()

    n_channels = n_kp * 3

    # ===========================================================
    # HYPERPARAMETERS
    # ===========================================================
    config = {
        'n_epochs': 200,
        'lr': 1e-3,
        'batch_size': 32,
        'patience': 30,
        'n_ensemble': 5,
    }

    print(f"\nConfig: {config}")
    print(f"Model params: ~{sum(p.numel() for p in SmallTemporalCNN(n_channels).parameters())} params")

    # ===========================================================
    # PER-TARGET TRAINING WITH 5-FOLD CV
    # ===========================================================

    results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()}")
        print(f"{'=' * 70}")

        y_t = y_scaled[target]

        # Per-player 5-fold CV
        oof_preds = np.zeros(len(pids_train))
        fold_losses = []

        for pid in unique_pids:
            pid_mask = pids_train == pid
            X_pid = X_train_norm[pid_mask]
            y_pid = y_t[pid_mask]
            pids_pid = pids_train[pid_mask]
            pid_indices = np.where(pid_mask)[0]
            n_pid = len(y_pid)

            print(f"\n  Player {pid}: {n_pid} samples")

            # 5-fold CV within player
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_pid)):
                X_fold_tr = X_pid[tr_idx]
                y_fold_tr = y_pid[tr_idx]
                pids_fold_tr = pids_pid[tr_idx]

                X_fold_val = X_pid[val_idx]
                y_fold_val = y_pid[val_idx]
                pids_fold_val = pids_pid[val_idx]

                val_preds, val_loss, _ = train_one_fold(
                    X_fold_tr, pids_fold_tr, y_fold_tr,
                    X_fold_val, pids_fold_val, y_fold_val,
                    player_id_map, device,
                    n_channels=n_channels,
                    n_epochs=config['n_epochs'],
                    lr=config['lr'],
                    batch_size=config['batch_size'],
                    patience=config['patience'],
                    seed=42 + fold_i,
                )

                oof_preds[pid_indices[val_idx]] = val_preds
                fold_losses.append(val_loss)
                print(f"    Fold {fold_i+1}: val_loss={val_loss:.6f}")

        cv_mse = np.mean((oof_preds - y_t) ** 2)
        print(f"\n  {target} CV MSE: {cv_mse:.6f}")
        print(f"  Mean fold loss: {np.mean(fold_losses):.6f}")

        # Train on full data and predict test (multi-seed ensemble)
        print(f"  Training full model ensemble for test predictions...")
        test_preds = train_full_and_predict(
            X_train_norm, pids_train, y_t,
            X_test_norm, pids_test,
            player_id_map, device,
            n_channels=n_channels,
            n_epochs=config['n_epochs'],
            lr=config['lr'],
            batch_size=config['batch_size'],
            patience=config['patience'],
            seed=42,
            n_ensemble=config['n_ensemble'],
        )

        results[target] = {
            'cv_mse': cv_mse,
            'oof_preds': oof_preds,
            'test_preds': test_preds,
        }

    # ===========================================================
    # OVERALL RESULTS
    # ===========================================================
    print(f"\n{'=' * 70}")
    print("OVERALL CV RESULTS")
    print(f"{'=' * 70}")

    total_mse = 0
    for target in TARGETS:
        mse = results[target]['cv_mse']
        total_mse += mse
        print(f"  {target}: {mse:.6f}")
    mean_mse = total_mse / 3
    print(f"  MEAN: {mean_mse:.6f}")

    # ===========================================================
    # DIVERSITY CHECK
    # ===========================================================
    print(f"\n{'=' * 70}")
    print("DIVERSITY CHECK")
    print(f"{'=' * 70}")

    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")

    print("  Correlation with Sub 784:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_784[col].values, results[target]['test_preds'])[0, 1]
        print(f"    {target}: r={r:.4f}")

    print("  Correlation with Sub 1350:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_1350[col].values, results[target]['test_preds'])[0, 1]
        print(f"    {target}: r={r:.4f}")

    # ===========================================================
    # GENERATE SUBMISSIONS
    # ===========================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Standalone CNN submission
    sub_num = get_next_submission_number()
    sub_cnn = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': results['angle']['test_preds'],
        'scaled_depth': results['depth']['test_preds'],
        'scaled_left_right': results['left_right']['test_preds'],
    })
    sub_cnn.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: CNN STANDALONE")
    print(f"    angle_std={sub_cnn['scaled_angle'].std():.6f}, depth_mean={sub_cnn['scaled_depth'].mean():.6f}")

    # Blends with Sub 784
    for w, desc in [(0.10, "10% CNN"), (0.20, "20% CNN"), (0.30, "30% CNN"), (0.50, "50% CNN")]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_784[col] + w * results[target]['test_preds']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {desc} + Sub 784")
        print(f"    angle_std={blended['scaled_angle'].std():.6f}, depth_mean={blended['scaled_depth'].mean():.6f}")

    # Blends with Sub 1350 (our best)
    for w, desc in [(0.10, "10% CNN"), (0.20, "20% CNN"), (0.30, "30% CNN")]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_1350[col] + w * results[target]['test_preds']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {desc} + Sub 1350")
        print(f"    angle_std={blended['scaled_angle'].std():.6f}, depth_mean={blended['scaled_depth'].mean():.6f}")

    # Target-specific blends (only for targets where CNN is diverse)
    for target in TARGETS:
        col = f'scaled_{target}'
        r_784 = np.corrcoef(sub_784[col].values, results[target]['test_preds'])[0, 1]
        r_1350 = np.corrcoef(sub_1350[col].values, results[target]['test_preds'])[0, 1]

        if r_1350 < 0.85:  # If diverse enough
            for w in [0.15, 0.25]:
                sub_num = get_next_submission_number()
                blended = sub_1350.copy()
                blended[col] = (1 - w) * sub_1350[col] + w * results[target]['test_preds']
                blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
                print(f"  Sub {sub_num}: {target}-only {int(w*100)}% CNN + Sub 1350 (r={r_1350:.3f})")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    results = main()

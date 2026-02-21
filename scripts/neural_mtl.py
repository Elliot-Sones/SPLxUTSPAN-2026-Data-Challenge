"""
Neural Multi-Task Learning with Soft Parameter Sharing

Approaches:
1. Shared MLP encoder + task-specific heads
2. Cross-stitch networks (learn combination of shared/specific features)
3. Attention-based task interaction
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048

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
        X_raw = np.zeros((n, len(keypoint_cols) * 240), dtype=np.float32)
        ids, pids, targets = [], [], []

        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_3d[idx, :, col_i // 3, col_i % 3] = arr
                X_raw[idx, col_i * 240:(col_i + 1) * 240] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        X_raw = np.nan_to_num(X_raw, nan=0.0)
        result = {'X_3d': X_3d, 'X_raw': X_raw, 'pids': np.array(pids),
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


def detect_release_frame(ts_3d, kp_index):
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is None:
        return 120
    wrist_traj = ts_3d[:, rw_idx, :].copy()
    for ax in range(3):
        vals = wrist_traj[:, ax]
        bad = np.isnan(vals) | np.isinf(vals)
        if np.all(bad):
            return 120
        if np.any(bad):
            good = ~bad
            vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
        wrist_traj[:, ax] = vals

    wrist_z_smooth = safe_savgol(wrist_traj[:, 2], 11, 3)
    wrist_peak = 80 + np.argmax(wrist_z_smooth[80:200])

    ft_keys = ['right_second_finger_distal', 'right_third_finger_distal',
               'right_fourth_finger_distal']
    ft_trajs = [ts_3d[:, kp_index[k], :] for k in ft_keys if k in kp_index]
    if ft_trajs:
        ft_center = np.nanmean(ft_trajs, axis=0)
        for ax in range(3):
            ft_center[:, ax] = safe_savgol(ft_center[:, ax], 15, 3)
        ball = wrist_traj + 0.6 * (ft_center - wrist_traj)
    else:
        ball = wrist_traj.copy()
    for ax in range(3):
        ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)

    vel = np.zeros_like(ball * FEET_TO_METERS)
    for ax in range(3):
        vel[:, ax] = safe_savgol(ball[:, ax] * FEET_TO_METERS, 9, 3, deriv=1, delta=DT)
    speed = np.linalg.norm(vel, axis=1)

    s, e = max(80, wrist_peak - 40), min(wrist_peak + 5, 200)
    return int(np.clip(s + np.argmax(speed[s:e]), 80, 200))


def extract_features_unified(ts_3d, ts_hr, kp_index, release_frame):
    """Extract features at all three target-specific frames and concatenate."""
    all_feats = []

    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        f = int(np.clip(frame, 0, 239))
        feats = []

        key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                      'left_wrist', 'left_shoulder',
                      'right_hip', 'left_hip', 'mid_hip',
                      'right_knee', 'left_knee', 'neck', 'nose']

        for jname in key_joints:
            idx = kp_index.get(jname)
            if idx is None:
                feats.extend([0.0] * 6)
                continue
            for coord in range(3):
                feats.append(ts_hr[f, idx, coord])
                vel = np.gradient(ts_hr[:, idx, coord], DT)
                feats.append(vel[f])

        for jname in key_joints:
            idx = kp_index.get(jname)
            if idx is None:
                feats.extend([0.0] * 9)
                continue
            for coord in range(3):
                series = ts_hr[:, idx, coord]
                feats.append(np.nanmean(series))
                feats.append(np.nanstd(series))
                feats.append(np.nanmax(series) - np.nanmin(series))

        all_feats.extend(feats)

    rw = kp_index.get('right_wrist')
    all_feats.append(release_frame)

    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            all_feats.append(np.nanmean(series[140:180]))
            all_feats.append(np.nanmax(vel[140:180]))
    else:
        all_feats.extend([0.0] * 6)

    return np.array(all_feats, dtype=np.float32)


def extract_all_features_unified(data):
    """Extract unified feature set for all shots."""
    n = len(data['pids'])
    kp_index = data['kp_index']

    all_feats = []
    release_frames = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames.append(rf)
        feats = extract_features_unified(ts_3d, ts_hr, kp_index, rf)
        all_feats.append(feats)

    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(release_frames)


# ==============================================================
# NEURAL MULTI-TASK MODELS
# ==============================================================

class SharedEncoderMTL(nn.Module):
    """Shared encoder with task-specific heads."""
    def __init__(self, input_dim, hidden_dims=[128, 64], n_tasks=3, dropout=0.3):
        super().__init__()

        # Shared encoder
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        # Task-specific heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 32),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 1)
            ) for _ in range(n_tasks)
        ])

    def forward(self, x):
        shared = self.encoder(x)
        outputs = [head(shared) for head in self.heads]
        return torch.cat(outputs, dim=1)


class CrossStitchMTL(nn.Module):
    """Cross-stitch network with learnable combination of shared/task-specific features."""
    def __init__(self, input_dim, hidden_dim=64, n_tasks=3, dropout=0.3):
        super().__init__()
        self.n_tasks = n_tasks

        # Shared and task-specific first layers
        self.shared_layer1 = nn.Linear(input_dim, hidden_dim)
        self.task_layers1 = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(n_tasks)])

        # Cross-stitch units (learnable combination)
        self.cross_stitch1 = nn.Parameter(torch.eye(n_tasks + 1))  # +1 for shared

        # Second layers
        self.shared_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.task_layers2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim // 2) for _ in range(n_tasks)])

        # Output heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(n_tasks)
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

    def forward(self, x):
        # First layer
        shared_h1 = self.relu(self.bn1(self.shared_layer1(x)))
        task_h1 = [self.relu(self.bn1(layer(x))) for layer in self.task_layers1]

        # Cross-stitch combination
        all_h1 = [shared_h1] + task_h1
        combined_h1 = []
        for i in range(self.n_tasks):
            combined = sum(self.cross_stitch1[i, j] * all_h1[j] for j in range(self.n_tasks + 1))
            combined_h1.append(self.dropout(combined))

        # Second layer
        shared_h2 = self.relu(self.bn2(self.shared_layer2(combined_h1[0])))
        task_h2 = [self.relu(self.bn2(self.task_layers2[i](combined_h1[i])))
                   for i in range(self.n_tasks)]

        # Output
        outputs = [head(h2) for head, h2 in zip(self.heads, task_h2)]
        return torch.cat(outputs, dim=1)


def train_neural_mtl(X_train, y_train, X_test, pids_train, pids_test,
                     model_class, model_kwargs, epochs=50, batch_size=16, lr=0.001):
    """Train neural MTL model per player."""
    unique_pids = sorted(np.unique(pids_train))
    n_targets = y_train.shape[1]

    oof = np.zeros((len(X_train), n_targets))
    test_preds = np.zeros((len(X_test), n_targets))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid

        X_p = X_train[tr_mask]
        y_p = y_train[tr_mask]
        indices = np.where(tr_mask)[0]

        # Standardize
        scaler_x = StandardScaler()
        X_s = scaler_x.fit_transform(X_p)

        # 5-fold CV for OOF
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(X_s):
            X_tr_t = torch.FloatTensor(X_s[tr_idx]).to(device)
            y_tr_t = torch.FloatTensor(y_p[tr_idx]).to(device)
            X_val_t = torch.FloatTensor(X_s[val_idx]).to(device)

            # Train model
            model = model_class(input_dim=X_s.shape[1], **model_kwargs).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.MSELoss()

            train_dataset = TensorDataset(X_tr_t, y_tr_t)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            model.train()
            for epoch in range(epochs):
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Predict
            model.eval()
            with torch.no_grad():
                oof[indices[val_idx]] = model(X_val_t).cpu().numpy()

        # Test predictions
        if np.any(te_mask):
            X_te_s = scaler_x.transform(X_test[te_mask])
            X_te_t = torch.FloatTensor(X_te_s).to(device)
            y_tr_t = torch.FloatTensor(y_p).to(device)
            X_tr_t = torch.FloatTensor(X_s).to(device)

            model = model_class(input_dim=X_s.shape[1], **model_kwargs).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.MSELoss()

            train_dataset = TensorDataset(X_tr_t, y_tr_t)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            model.train()
            for epoch in range(epochs):
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                test_preds[te_mask] = model(X_te_t).cpu().numpy()

    return oof, test_preds


def main():
    t0 = time.time()
    print("=" * 70)
    print("NEURAL MULTI-TASK LEARNING")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    # Load scalers
    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # Scale targets
    y_scaled = np.zeros_like(y_train)
    for i, target in enumerate(TARGETS):
        y_scaled[:, i] = scalers[target].transform(
            y_train[:, i].reshape(-1, 1)).ravel()

    # Extract features
    print("\nExtracting features...")
    X_train, _ = extract_all_features_unified(train_data)
    X_test, _ = extract_all_features_unified(test_data)
    print(f"  Features: {X_train.shape[1]}")

    # Add PLS components
    print("\nAdding PLS components...")
    pls_train_all = []
    pls_test_all = []

    for t_idx, target in enumerate(TARGETS):
        unique_pids = sorted(np.unique(pids_train))
        max_nc = 15
        pls_train = np.zeros((len(pids_train), max_nc), dtype=np.float32)
        pls_test = np.zeros((len(pids_test), max_nc), dtype=np.float32)

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            n_p = tr_mask.sum()

            scaler = StandardScaler()
            raw_tr = scaler.fit_transform(train_data['X_raw'][tr_mask])
            raw_te = scaler.transform(test_data['X_raw'][te_mask])

            nc = min(10, n_p - n_p // 5 - 1)
            nc = max(3, nc)

            pls = PLSRegression(n_components=nc)
            pls.fit(raw_tr, y_train[tr_mask, t_idx])
            pls_train[tr_mask, :nc] = pls.transform(raw_tr)
            pls_test[te_mask, :nc] = pls.transform(raw_te)

        pls_train_all.append(pls_train)
        pls_test_all.append(pls_test)

    pls_train_concat = np.hstack(pls_train_all)
    pls_test_concat = np.hstack(pls_test_all)

    X_train_aug = np.hstack([X_train, pls_train_concat])
    X_test_aug = np.hstack([X_test, pls_test_concat])
    print(f"  Augmented features: {X_train_aug.shape[1]}")

    # ============================================================
    # NEURAL MTL MODELS
    # ============================================================

    results = {}

    # 1. Shared Encoder MTL
    print("\n[1] Shared Encoder MTL...")
    print("  Training (epochs=50, batch=16, lr=0.001)...")
    oof_shared, test_shared = train_neural_mtl(
        X_train_aug, y_scaled, X_test_aug, pids_train, pids_test,
        SharedEncoderMTL, {'hidden_dims': [128, 64], 'dropout': 0.3},
        epochs=50, batch_size=16, lr=0.001)

    mse_per_target = []
    for i, target in enumerate(TARGETS):
        mse = np.mean((oof_shared[:, i] - y_scaled[:, i]) ** 2)
        mse_per_target.append(mse)
        print(f"  {target}: MSE={mse:.6f}")
    mean_mse = np.mean(mse_per_target)
    print(f"  MEAN: {mean_mse:.6f}")

    results['neural_shared'] = {
        'oof': oof_shared,
        'test': test_shared,
        'mse': mean_mse,
        'mse_per_target': mse_per_target,
    }

    # 2. Cross-Stitch MTL
    print("\n[2] Cross-Stitch MTL...")
    print("  Training (epochs=50, batch=16, lr=0.001)...")
    oof_cross, test_cross = train_neural_mtl(
        X_train_aug, y_scaled, X_test_aug, pids_train, pids_test,
        CrossStitchMTL, {'hidden_dim': 64, 'dropout': 0.3},
        epochs=50, batch_size=16, lr=0.001)

    mse_per_target = []
    for i, target in enumerate(TARGETS):
        mse = np.mean((oof_cross[:, i] - y_scaled[:, i]) ** 2)
        mse_per_target.append(mse)
        print(f"  {target}: MSE={mse:.6f}")
    mean_mse = np.mean(mse_per_target)
    print(f"  MEAN: {mean_mse:.6f}")

    results['neural_cross'] = {
        'oof': oof_cross,
        'test': test_cross,
        'mse': mean_mse,
        'mse_per_target': mse_per_target,
    }

    # ============================================================
    # DIVERSITY ANALYSIS
    # ============================================================

    print("\n" + "=" * 70)
    print("DIVERSITY ANALYSIS")
    print("=" * 70)

    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")

    for approach_name, res in results.items():
        print(f"\n{approach_name}:")
        for i, target in enumerate(TARGETS):
            col = f'scaled_{target}'
            r_784 = np.corrcoef(sub_784[col].values, res['test'][:, i])[0, 1]
            r_1350 = np.corrcoef(sub_1350[col].values, res['test'][:, i])[0, 1]
            print(f"  {target}: r(Sub784)={r_784:.4f}, r(Sub1350)={r_1350:.4f}")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================

    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    # Standalone submissions
    for approach_name, res in results.items():
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': res['test'][:, 0],
            'scaled_depth': res['test'][:, 1],
            'scaled_left_right': res['test'][:, 2],
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {approach_name} (standalone)")
        print(f"    CV MSE: {res['mse']:.6f}")

    # Blends with Sub 784
    print("\n  Blending with Sub 784:")

    for approach_name, res in results.items():
        for aw, dw, lw, desc in [
            (0.00, 0.30, 0.50, "Sub 784 weights"),
            (0.00, 0.20, 0.30, "conservative"),
        ]:
            sub_num = get_next_submission_number()
            blended = sub_784.copy()
            blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*res['test'][:, 0]
            blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*res['test'][:, 1]
            blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*res['test'][:, 2]

            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
            print(f"  Sub {sub_num}: {approach_name} blend aw={aw:.2f} dw={dw:.2f} lw={lw:.2f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

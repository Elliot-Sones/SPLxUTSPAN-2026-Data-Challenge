"""
BiGRU Velocity Variant - Train on VELOCITIES instead of positions.

Key insight: velocity captures DYNAMICS of the motion (how fast joints move),
while positions capture STATICS (where joints are). These are complementary information.
By training on velocities, we get predictions that are diverse from both:
1. The Ridge model (position-based features)
2. The position-based BiGRU (different input features)

Features: 15 key joints x 3 coords x velocity = 45 features per frame
Also includes accelerations (2nd derivative) = 45 more features
Total: 90 velocity+acceleration features, 120 subsampled frames

Multi-seed ensemble with aggressive regularization.
"""

import json
import fcntl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]

KEY_JOINTS = [
    'right_wrist', 'right_elbow', 'right_shoulder',
    'left_wrist', 'left_elbow', 'left_shoulder',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'mid_hip', 'nose', 'neck'
]


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


class VelocityBiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=20, n_layers=1, n_players=5, dropout=0.4):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, 32)
        self.gru = nn.GRU(32, hidden_dim, num_layers=n_layers,
                          batch_first=True, bidirectional=True, dropout=0.0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.player_embed = nn.Embedding(n_players + 1, 4)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, 24),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1)
        )

    def forward(self, x, player_ids):
        x = self.input_norm(x)
        x = torch.relu(self.input_proj(x))
        gru_out, _ = self.gru(x)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = (attn_weights * gru_out).sum(dim=1)
        pe = self.player_embed(player_ids)
        combined = torch.cat([context, pe], dim=1)
        return self.head(combined).squeeze(-1)


def extract_velocity_features(X_3d, kp_index):
    """Extract velocity and acceleration features from trajectories."""
    n = len(X_3d)
    joint_indices = [kp_index.get(j) for j in KEY_JOINTS if j in kp_index]
    n_feats = len(joint_indices) * 3  # joints x 3 coords

    # Subsample frames
    frame_idx = np.linspace(60, 220, 120, dtype=int)

    features = np.zeros((n, 120, n_feats * 2), dtype=np.float32)  # velocity + acceleration

    for i in range(n):
        feat_list = []
        vel_list = []
        for ji in joint_indices:
            for ax in range(3):
                traj = X_3d[i, :, ji, ax].astype(np.float64)
                # Interpolate NaN
                bad = np.isnan(traj) | np.isinf(traj)
                if np.all(bad):
                    traj[:] = 0.0
                elif np.any(bad):
                    good = ~bad
                    traj[bad] = np.interp(np.where(bad)[0], np.where(good)[0], traj[good])

                # Velocity (1st derivative)
                vel = np.gradient(traj)
                # Acceleration (2nd derivative)
                acc = np.gradient(vel)

                feat_list.append(vel[frame_idx])
                vel_list.append(acc[frame_idx])

        features[i, :, :n_feats] = np.column_stack(feat_list)
        features[i, :, n_feats:] = np.column_stack(vel_list)

    return features


def train_model(X_train, y_train, pids_train, X_test, pids_test,
                target_idx, seed, epochs=100, lr=0.003):
    """Train one BiGRU model with given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train[:, target_idx]).to(device)
    pids_tr = torch.LongTensor(pids_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)
    pids_te = torch.LongTensor(pids_test).to(device)

    model = VelocityBiGRU(input_dim=X_train.shape[2], hidden_dim=20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        # Augmentation
        X_aug = X_tr.clone()
        if np.random.random() < 0.5:
            noise = torch.randn_like(X_aug) * 0.02
            X_aug = X_aug + noise
        if np.random.random() < 0.3:
            # Temporal jitter
            shift = np.random.randint(-3, 4)
            if shift > 0:
                X_aug[:, shift:, :] = X_aug[:, :-shift, :].clone()
            elif shift < 0:
                X_aug[:, :shift, :] = X_aug[:, -shift:, :].clone()

        pred = model(X_aug, pids_tr)
        loss = nn.functional.mse_loss(pred, y_tr)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience > 30:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_te, pids_te).cpu().numpy()
    return test_pred


def main():
    import time
    t0 = time.time()

    print("=" * 60, flush=True)
    print("BiGRU VELOCITY VARIANT", flush=True)
    print("Velocity + acceleration features for diverse predictions", flush=True)
    print("=" * 60, flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    kp_names = [col[:-2] for col in keypoint_cols if col.endswith("_x")]
    kp_index = {name: i for i, name in enumerate(kp_names)}

    def process(df, is_train=True):
        n = len(df)
        n_kp = len(kp_names)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        ids, pids = [], []
        targets = []
        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                X_3d[idx, :, col_i // 3, col_i % 3] = parse_array_string(row[col])
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

    # Extract features
    print("Extracting velocity features...", flush=True)
    X_train = extract_velocity_features(train_data['X_3d'], kp_index)
    X_test = extract_velocity_features(test_data['X_3d'], kp_index)
    print(f"  Feature shape: {X_train.shape}", flush=True)

    # Scale targets
    scalers = {}
    y_scaled = np.zeros_like(train_data['y'])
    for t in range(3):
        sc = MinMaxScaler()
        y_scaled[:, t] = sc.fit_transform(train_data['y'][:, t:t+1]).ravel()
        scalers[t] = sc

    # Map player IDs to 0-indexed
    unique_pids = np.unique(train_data['pids'])
    pid_map = {p: i for i, p in enumerate(unique_pids)}
    pids_train = np.array([pid_map[p] for p in train_data['pids']])
    pids_test = np.array([pid_map[p] for p in test_data['pids']])

    # Multi-seed training
    n_seeds = 10
    print(f"\nGenerating test predictions ({n_seeds} seeds)...", flush=True)

    all_test_preds = {t: [] for t in range(3)}
    seeds = [42, 7, 99, 133, 2026, 314, 17, 555, 888, 1234]

    for t, target in enumerate(TARGETS):
        for s_i, seed in enumerate(seeds[:n_seeds]):
            pred = train_model(X_train, y_scaled, pids_train,
                             X_test, pids_test, t, seed, epochs=100)
            all_test_preds[t].append(pred)
            print(f"  {target}: seed {s_i+1}/{n_seeds} done", flush=True)

    # Average predictions
    test_preds = np.zeros((len(test_data['ids']), 3))
    for t in range(3):
        test_preds[:, t] = np.clip(np.mean(all_test_preds[t], axis=0), 0, 1)

    # Diversity check
    print("\n" + "-" * 60, flush=True)
    print("DIVERSITY CHECK: Correlation with Sub 2716", flush=True)
    print("-" * 60, flush=True)
    anchor = pd.read_csv(SUBMISSION_DIR / "submission_2716.csv")
    corrs = []
    for t, target in enumerate(TARGETS):
        col = f"scaled_{target}"
        r = np.corrcoef(test_preds[:, t], anchor[col].values)[0, 1]
        corrs.append(r)
        print(f"  {target}: r = {r:.4f}", flush=True)
    print(f"  Mean correlation: {np.mean(corrs):.4f}", flush=True)

    # Also check diversity with position-based BiGRU (Sub 2887)
    print("\nDiversity vs Position BiGRU (Sub 2887):", flush=True)
    bigru_pos = pd.read_csv(SUBMISSION_DIR / "submission_2887.csv")
    for t, target in enumerate(TARGETS):
        col = f"scaled_{target}"
        r = np.corrcoef(test_preds[:, t], bigru_pos[col].values)[0, 1]
        print(f"  {target}: r = {r:.4f}", flush=True)

    # Save submissions
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
    print(f"Sub {sub_num}: Velocity BiGRU standalone ({n_seeds} seeds)", flush=True)

    # Blends with Sub 2716
    anchor_preds = anchor[['scaled_angle', 'scaled_depth', 'scaled_left_right']].values
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
        print(f"Sub {sn}: {int(w*100)}% velocity BiGRU + {int((1-w)*100)}% Sub 2716", flush=True)

    # Average of position and velocity BiGRUs
    pos_preds = bigru_pos[['scaled_angle', 'scaled_depth', 'scaled_left_right']].values
    bigru_avg = 0.5 * test_preds + 0.5 * pos_preds

    # Check diversity of averaged BiGRU
    print("\nDiversity of averaged pos+vel BiGRU vs Sub 2716:", flush=True)
    for t, target in enumerate(TARGETS):
        col = f"scaled_{target}"
        r = np.corrcoef(bigru_avg[:, t], anchor[col].values)[0, 1]
        print(f"  {target}: r = {r:.4f}", flush=True)

    # Blends of averaged BiGRU
    for w in [0.02, 0.05, 0.10]:
        blend = (1 - w) * anchor_preds + w * bigru_avg
        blend = np.clip(blend, 0, 1)
        sn = get_next_submission_number()
        df = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': blend[:, 0],
            'scaled_depth': blend[:, 1],
            'scaled_left_right': blend[:, 2]
        })
        df.to_csv(SUBMISSION_DIR / f"submission_{sn}.csv", index=False)
        print(f"Sub {sn}: {int(w*100)}% avg(pos+vel)BiGRU + {int((1-w)*100)}% Sub 2716", flush=True)

    # Depth-only blend of averaged BiGRU
    for w in [0.02, 0.05]:
        blend = anchor_preds.copy()
        blend[:, 1] = (1 - w) * anchor_preds[:, 1] + w * bigru_avg[:, 1]
        blend = np.clip(blend, 0, 1)
        sn = get_next_submission_number()
        df = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': blend[:, 0],
            'scaled_depth': blend[:, 1],
            'scaled_left_right': blend[:, 2]
        })
        df.to_csv(SUBMISSION_DIR / f"submission_{sn}.csv", index=False)
        print(f"Sub {sn}: {int(w*100)}% avg(pos+vel)BiGRU depth-only + Sub 2716", flush=True)

    total = time.time() - t0
    print(f"\nTotal runtime: {total:.1f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

"""
GATv2 Skeleton Graph Network

Model the human skeleton as a graph where:
- 69 nodes = keypoints (joints)
- Edges = anatomical connections (bones)
- Node features = 3D coordinates + velocities at release window

Use Graph Attention Network v2 (GATv2) to learn which joint relationships
matter for each target. This preserves spatial topology that flat feature
vectors destroy.

Produces diverse predictions for blending with existing Ridge + CNN pipeline.
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
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0

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
    print("Loading data...", flush=True)
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


# ============================================================
# SKELETON GRAPH DEFINITION
# ============================================================

def build_skeleton_edges(kp_index):
    """Build skeleton adjacency from anatomical connections.
    Returns edge_index tensor of shape [2, num_edges].
    """
    # Define anatomical connections
    connections = [
        # Spine
        ('mid_hip', 'spine_navel'), ('spine_navel', 'spine_chest'),
        ('spine_chest', 'neck'), ('neck', 'nose'),
        # Right arm
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('neck', 'right_shoulder'),
        # Left arm
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('neck', 'left_shoulder'),
        # Right leg
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('mid_hip', 'right_hip'),
        # Left leg
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('mid_hip', 'left_hip'),
        # Face
        ('nose', 'left_eye'), ('nose', 'right_eye'),
        ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
        # Right hand
        ('right_wrist', 'right_thumb_proximal'), ('right_thumb_proximal', 'right_thumb_distal'),
        ('right_wrist', 'right_index_finger_proximal'), ('right_index_finger_proximal', 'right_index_finger_intermediate'),
        ('right_index_finger_intermediate', 'right_index_finger_distal'),
        ('right_wrist', 'right_second_finger_proximal'), ('right_second_finger_proximal', 'right_second_finger_intermediate'),
        ('right_second_finger_intermediate', 'right_second_finger_distal'),
        ('right_wrist', 'right_third_finger_proximal'), ('right_third_finger_proximal', 'right_third_finger_intermediate'),
        ('right_third_finger_intermediate', 'right_third_finger_distal'),
        ('right_wrist', 'right_fourth_finger_proximal'), ('right_fourth_finger_proximal', 'right_fourth_finger_intermediate'),
        ('right_fourth_finger_intermediate', 'right_fourth_finger_distal'),
        # Left hand
        ('left_wrist', 'left_thumb_proximal'), ('left_thumb_proximal', 'left_thumb_distal'),
        ('left_wrist', 'left_index_finger_proximal'), ('left_index_finger_proximal', 'left_index_finger_intermediate'),
        ('left_index_finger_intermediate', 'left_index_finger_distal'),
        ('left_wrist', 'left_second_finger_proximal'), ('left_second_finger_proximal', 'left_second_finger_intermediate'),
        ('left_second_finger_intermediate', 'left_second_finger_distal'),
        ('left_wrist', 'left_third_finger_proximal'), ('left_third_finger_proximal', 'left_third_finger_intermediate'),
        ('left_third_finger_intermediate', 'left_third_finger_distal'),
        ('left_wrist', 'left_fourth_finger_proximal'), ('left_fourth_finger_proximal', 'left_fourth_finger_intermediate'),
        ('left_fourth_finger_intermediate', 'left_fourth_finger_distal'),
        # Feet
        ('right_ankle', 'right_foot'), ('left_ankle', 'left_foot'),
    ]

    edges = []
    for a, b in connections:
        if a in kp_index and b in kp_index:
            i, j = kp_index[a], kp_index[b]
            edges.append([i, j])
            edges.append([j, i])  # Undirected

    # Also add self-loops
    n_kp = len(kp_index)
    for i in range(n_kp):
        edges.append([i, i])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def extract_node_features(ts_3d, ts_hr, kp_index, frame, n_kp):
    """Extract per-node features at a given frame.
    For each keypoint: position (3) + velocity (3) + acceleration (3) = 9 features.
    Also add multi-frame context: positions at frame-5 and frame+5.
    Total: 15 features per node.
    """
    f = int(np.clip(frame, 5, 234))
    n_feats = 15
    node_feats = np.zeros((n_kp, n_feats), dtype=np.float32)

    for idx in range(n_kp):
        # Position at frame (hoop-relative)
        pos = ts_hr[f, idx, :]
        node_feats[idx, 0:3] = pos

        # Velocity (central difference)
        vel = np.zeros(3)
        for c in range(3):
            series = ts_hr[:, idx, c]
            bad = np.isnan(series) | np.isinf(series)
            if not np.all(bad):
                if np.any(bad):
                    good = ~bad
                    series[bad] = np.interp(np.where(bad)[0], np.where(good)[0], series[good])
                g = np.gradient(series, DT)
                vel[c] = g[f]
        node_feats[idx, 3:6] = vel

        # Acceleration
        acc = np.zeros(3)
        for c in range(3):
            series = ts_hr[:, idx, c]
            bad = np.isnan(series) | np.isinf(series)
            if not np.all(bad):
                if np.any(bad):
                    good = ~bad
                    series[bad] = np.interp(np.where(bad)[0], np.where(good)[0], series[good])
                g2 = np.gradient(np.gradient(series, DT), DT)
                acc[c] = g2[f]
        node_feats[idx, 6:9] = acc

        # Position at frame-5
        f_prev = max(0, f - 5)
        node_feats[idx, 9:12] = ts_hr[f_prev, idx, :]

        # Position at frame+5
        f_next = min(239, f + 5)
        node_feats[idx, 12:15] = ts_hr[f_next, idx, :]

    # Clean NaNs
    node_feats = np.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
    return node_feats


# ============================================================
# GATv2 MODEL
# ============================================================

class SkeletonGATv2(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, num_heads=4,
                 n_players=5, player_embed_dim=8, dropout=0.3):
        super().__init__()

        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads,
                               dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)

        self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels,
                               heads=num_heads, dropout=dropout, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_channels * num_heads)

        self.conv3 = GATv2Conv(hidden_channels * num_heads, hidden_channels,
                               heads=1, dropout=dropout, concat=False)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # Player embedding
        self.player_embed = nn.Embedding(n_players, player_embed_dim)

        # Prediction head (mean pool + max pool = 2 * hidden_channels)
        fc_in = hidden_channels * 2 + player_embed_dim
        self.fc1 = nn.Linear(fc_in, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        player_id = data.player_id

        # GATv2 layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)

        # Graph-level pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1)

        # Player embedding
        p_embed = self.player_embed(player_id)
        x_combined = torch.cat([x_graph, p_embed], dim=1)

        # Prediction
        x_combined = F.elu(self.fc1(x_combined))
        x_combined = self.dropout(x_combined)
        x_combined = F.elu(self.fc2(x_combined))
        x_combined = self.fc3(x_combined)

        return x_combined.squeeze(-1)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_graph_data(data, kp_index, edge_index, frame, y_target=None, y_scaled=None):
    """Convert skeleton data to PyG Data objects."""
    n = len(data['pids'])
    n_kp = len(kp_index)

    # Map player IDs to 0-4
    unique_pids = sorted(np.unique(data['pids']))
    pid_map = {pid: i for i, pid in enumerate(unique_pids)}

    graphs = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        node_feats = extract_node_features(ts_3d, ts_hr, kp_index, frame, n_kp)

        g = Data(
            x=torch.tensor(node_feats, dtype=torch.float),
            edge_index=edge_index.clone(),
            player_id=torch.tensor([pid_map[data['pids'][i]]], dtype=torch.long),
        )

        if y_target is not None:
            g.y = torch.tensor([y_scaled[i]], dtype=torch.float)

        graphs.append(g)

    return graphs


def train_and_predict_loo(graphs_train, graphs_test, y_scaled,
                          n_epochs=200, lr=0.005, weight_decay=0.01,
                          seed=42, patience=30):
    """Train GATv2 with LOO cross-validation. Returns OOF and test predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_train = len(graphs_train)
    oof_preds = np.zeros(n_train)

    # Use 5-fold CV instead of LOO for speed
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(range(n_train))):
        print(f"    Fold {fold+1}/5...", flush=True)

        tr_graphs = [graphs_train[i] for i in tr_idx]
        val_graphs = [graphs_train[i] for i in val_idx]

        tr_loader = DataLoader(tr_graphs, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=len(val_graphs))

        in_channels = tr_graphs[0].x.shape[1]
        model = SkeletonGATv2(in_channels=in_channels, hidden_channels=32,
                              num_heads=4, dropout=0.3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_loss = float('inf')
        best_state = None
        patience_count = 0

        for epoch in range(n_epochs):
            # Train
            model.train()
            total_loss = 0
            for batch in tr_loader:
                optimizer.zero_grad()
                pred = model(batch)
                loss = F.mse_loss(pred, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            scheduler.step()

            # Validate
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    pred = model(batch)
                    val_loss = F.mse_loss(pred, batch.y).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= patience:
                break

        # Predict validation fold
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch)
                oof_preds[val_idx] = pred.numpy()

        print(f"      Val MSE: {best_val_loss:.6f} (epoch {epoch - patience_count + 1})", flush=True)

    # Train full model for test predictions
    print("    Training full model for test predictions...", flush=True)
    test_preds_all = []

    for s in range(3):  # Multi-seed
        torch.manual_seed(seed + s)
        full_loader = DataLoader(graphs_train, batch_size=32, shuffle=True)
        test_loader = DataLoader(graphs_test, batch_size=len(graphs_test))

        in_channels = graphs_train[0].x.shape[1]
        model = SkeletonGATv2(in_channels=in_channels, hidden_channels=32,
                              num_heads=4, dropout=0.3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        for epoch in range(n_epochs):
            model.train()
            for batch in full_loader:
                optimizer.zero_grad()
                pred = model(batch)
                loss = F.mse_loss(pred, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                pred = model(batch)
                test_preds_all.append(pred.numpy())

    test_preds = np.mean(test_preds_all, axis=0)
    return oof_preds, test_preds


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("GATv2 SKELETON GRAPH NETWORK", flush=True)
    print("=" * 70, flush=True)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']

    # Load scalers
    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    # Build skeleton graph
    edge_index = build_skeleton_edges(kp_index)
    print(f"Skeleton graph: {len(kp_index)} nodes, {edge_index.shape[1]} edges", flush=True)

    # Load Sub 3411 for diversity comparison
    sub_3411 = pd.read_csv(SUBMISSION_DIR / "submission_3411.csv")

    results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}", flush=True)
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})", flush=True)
        print(f"{'=' * 70}", flush=True)

        frame = TARGET_FRAMES[target]
        y_target = y_scaled[target]

        # Prepare graph data
        print("  Preparing graph data...", flush=True)
        graphs_train = prepare_graph_data(train_data, kp_index, edge_index,
                                           frame, y_train, y_target)
        graphs_test = prepare_graph_data(test_data, kp_index, edge_index,
                                          frame)

        # Also try multi-frame: average predictions from multiple frames
        frames_to_try = [frame, frame - 5, frame + 5]
        all_oof = []
        all_test = []

        for f in frames_to_try:
            print(f"\n  Frame {f}:", flush=True)
            g_train = prepare_graph_data(train_data, kp_index, edge_index,
                                          f, y_train, y_target)
            g_test = prepare_graph_data(test_data, kp_index, edge_index, f)

            oof, test_pred = train_and_predict_loo(
                g_train, g_test, y_target,
                n_epochs=200, lr=0.005, weight_decay=0.01,
                seed=42, patience=30)
            mse = np.mean((oof - y_target) ** 2)
            print(f"    CV MSE: {mse:.6f}", flush=True)

            all_oof.append(oof)
            all_test.append(test_pred)

        # Average across frames
        oof_avg = np.mean(all_oof, axis=0)
        test_avg = np.mean(all_test, axis=0)
        mse_avg = np.mean((oof_avg - y_target) ** 2)
        print(f"\n  Multi-frame average CV MSE: {mse_avg:.6f}", flush=True)

        # Pick best: single frame or average
        best_mse = mse_avg
        best_oof = oof_avg
        best_test = test_avg
        best_desc = "multi-frame avg"

        for i, f in enumerate(frames_to_try):
            mse_f = np.mean((all_oof[i] - y_target) ** 2)
            if mse_f < best_mse:
                best_mse = mse_f
                best_oof = all_oof[i]
                best_test = all_test[i]
                best_desc = f"frame {f}"

        # Diversity vs Sub 3411
        col = f'scaled_{target}'
        r = np.corrcoef(sub_3411[col].values, best_test)[0, 1]
        print(f"\n  BEST {target}: {best_desc} (MSE={best_mse:.6f})", flush=True)
        print(f"  Diversity vs Sub3411: r={r:.4f}", flush=True)

        results[target] = {
            'best_test': best_test,
            'best_oof': best_oof,
            'best_mse': float(best_mse),
            'best_desc': best_desc,
            'diversity_r': float(r),
        }

    # ============================================================
    # OVERALL RESULTS
    # ============================================================
    print(f"\n{'=' * 70}", flush=True)
    print("OVERALL RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)

    total_mse = 0
    for target in TARGETS:
        mse = results[target]['best_mse']
        total_mse += mse
        r = results[target]['diversity_r']
        print(f"  {target}: MSE={mse:.6f}, diversity r={r:.4f} [{results[target]['best_desc']}]", flush=True)
    print(f"  MEAN: {total_mse/3:.6f}", flush=True)

    # ============================================================
    # SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}", flush=True)
    print("GENERATING SUBMISSIONS", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Standalone
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': results['angle']['best_test'],
        'scaled_depth': results['depth']['best_test'],
        'scaled_left_right': results['left_right']['best_test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: GATv2 STANDALONE", flush=True)

    # Blends with Sub 3411
    for w in [0.03, 0.05, 0.10, 0.15]:
        sub_num = get_next_submission_number()
        blended = sub_3411.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_3411[col] + w * results[target]['best_test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(w*100)}% GATv2 + {int((1-w)*100)}% Sub3411", flush=True)

    # Save results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': {t: {'best_mse': results[t]['best_mse'],
                        'diversity_r': results[t]['diversity_r'],
                        'best_desc': results[t]['best_desc']}
                    for t in TARGETS},
        'mean_mse': float(total_mse / 3),
    }
    with open(OUTPUT_DIR / "gatv2_skeleton_graph_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    # Save OOF predictions for stacking
    oof_df = pd.DataFrame({
        'oof_angle': results['angle']['best_oof'],
        'oof_depth': results['depth']['best_oof'],
        'oof_left_right': results['left_right']['best_oof'],
    })
    oof_df.to_csv(OUTPUT_DIR / "gatv2_oof_predictions.csv", index=False)

    test_df = pd.DataFrame({
        'test_angle': results['angle']['best_test'],
        'test_depth': results['depth']['best_test'],
        'test_left_right': results['left_right']['best_test'],
    })
    test_df.to_csv(OUTPUT_DIR / "gatv2_test_predictions.csv", index=False)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s", flush=True)
    print(f"Params: {sum(p.numel() for p in SkeletonGATv2(15).parameters())}", flush=True)


if __name__ == "__main__":
    main()

"""
DTW (Dynamic Time Warping) Similarity Model

Instead of measuring similarity using Euclidean distance on single frames
(what our Ridge pipeline does), use DTW distance on temporal sequences.

Key insight: Two shots with similar DYNAMICS but slightly different
positions at a specific frame would be poorly weighted by our Ridge,
but well-weighted by DTW. This captures temporal similarity that
static features miss.

Steps:
1. Extract key joint trajectories (shooting arm: wrist, elbow, shoulder)
2. Compute DTW distances between all pairs of shots
3. Use DTW-based kernel weights for locally-weighted Ridge regression
4. Per-player per-target models

This is fundamentally different from our base pipeline because:
- Distance metric is temporal (not spatial snapshot)
- Captures motion PATTERNS not just positions
- Different shots will get different weights
"""

import json
import fcntl
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])


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


def load_data():
    """Load keypoint data and extract trajectories."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    # Shooting arm joints for DTW
    dtw_joints = ["right_wrist", "right_elbow", "right_shoulder"]
    dtw_coords = ['x', 'y', 'z']

    # Feature extraction joints for regression
    feature_joints = [
        "right_wrist", "right_elbow", "right_shoulder",
        "left_wrist", "left_elbow", "left_shoulder",
        "mid_hip", "neck", "right_hip", "left_hip",
    ]
    target_frames = {"angle": 153, "depth": 150, "left_right": 170}

    def extract(df, is_train=True):
        n = len(df)
        ids, pids, targets = [], [], []

        # DTW trajectories: (N, T, n_dtw_joints*3)
        # Use frames 100-200 (the shot motion window)
        frame_range = range(100, 200)
        n_frames = len(frame_range)
        n_dtw_feats = len(dtw_joints) * len(dtw_coords)
        trajectories = np.zeros((n, n_frames, n_dtw_feats), dtype=np.float32)

        # Regression features: hoop-relative positions at target frames
        n_reg_feats = len(feature_joints) * 3 * len(target_frames) + 2  # + hoop distance, angle
        features = np.zeros((n, n_reg_feats), dtype=np.float32)

        for idx in range(n):
            row = df.iloc[idx]
            kp = {}
            for col in keypoint_cols:
                kp[col] = parse_array_string(row[col])

            # DTW trajectories (normalized per-shot)
            feat_idx = 0
            for jname in dtw_joints:
                for coord in dtw_coords:
                    ts = kp.get(f"{jname}_{coord}", np.zeros(240))
                    segment = ts[100:200]
                    segment = np.nan_to_num(segment, nan=0.0)
                    trajectories[idx, :, feat_idx] = segment
                    feat_idx += 1

            # Normalize trajectory per shot (remove positional offset)
            mean = trajectories[idx].mean(axis=0, keepdims=True)
            std = trajectories[idx].std(axis=0, keepdims=True)
            std = np.maximum(std, 1e-6)
            trajectories[idx] = (trajectories[idx] - mean) / std

            # Regression features
            reg_idx = 0
            for fname, fnum in target_frames.items():
                for jname in feature_joints:
                    for coord in dtw_coords:
                        ts = kp.get(f"{jname}_{coord}", np.zeros(240))
                        features[idx, reg_idx] = ts[fnum] if fnum < 240 else 0
                        reg_idx += 1

            # Hoop features
            mx = kp.get("mid_hip_x", np.zeros(240))[120]
            my = kp.get("mid_hip_y", np.zeros(240))[120]
            mz = kp.get("mid_hip_z", np.zeros(240))[120]
            pos = np.array([mx, my, mz])
            features[idx, reg_idx] = np.linalg.norm(HOOP_POS - pos)
            features[idx, reg_idx + 1] = np.arctan2(HOOP_POS[1] - pos[1], HOOP_POS[0] - pos[0])

            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        result = {
            'trajectories': trajectories,  # (N, 100, 9) for DTW
            'features': features,           # (N, n_reg_feats) for Ridge
            'pids': np.array(pids),
            'ids': np.array(ids),
        }
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    return extract(train_df, True), extract(test_df, False)


def fast_dtw_distance(seq1, seq2):
    """Fast approximate DTW using downsampled Euclidean on windows.

    Instead of full DTW (O(T^2)), use a fast approximation:
    1. Divide each sequence into K equal windows
    2. Compute mean feature vector per window
    3. Compute Euclidean distance on the windowed features

    This captures temporal alignment while being O(K) per pair.
    """
    K = 10  # Number of windows
    T = seq1.shape[0]
    window_size = T // K

    # Mean features per window
    w1 = np.array([seq1[i*window_size:(i+1)*window_size].mean(axis=0) for i in range(K)])
    w2 = np.array([seq2[i*window_size:(i+1)*window_size].mean(axis=0) for i in range(K)])

    return np.sqrt(np.sum((w1 - w2) ** 2))


def compute_dtw_distances(trajs):
    """Compute pairwise DTW distances."""
    n = len(trajs)
    D = np.zeros((n, n))

    # Vectorized windowed approach (much faster)
    K = 10
    T = trajs.shape[1]
    ws = T // K

    # Reshape to windows: (N, K, n_feats)
    windowed = np.zeros((n, K, trajs.shape[2]))
    for k in range(K):
        windowed[:, k, :] = trajs[:, k*ws:(k+1)*ws, :].mean(axis=1)

    # Flatten windows: (N, K*n_feats)
    flat = windowed.reshape(n, -1)

    # Euclidean distance matrix
    D = cdist(flat, flat, metric='euclidean')

    return D


def dtw_weighted_ridge(train_data, test_data, bw_quantile=0.35, alpha=10.0):
    """DTW-distance weighted Ridge regression per player per target."""
    y_raw = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    n_train = len(pids_train)
    n_test = len(pids_test)

    # Scale targets
    target_scalers = {}
    y_scaled = np.zeros_like(y_raw)
    for t in range(3):
        scaler = MinMaxScaler()
        y_scaled[:, t] = scaler.fit_transform(y_raw[:, t:t+1]).ravel()
        target_scalers[t] = scaler

    unique_pids = sorted(set(pids_train.tolist()))
    test_preds = np.zeros((n_test, 3))
    loo_preds = np.zeros((n_train, 3))

    for pid in unique_pids:
        train_mask = pids_train == pid
        test_mask = pids_test == pid
        n_p = train_mask.sum()

        # DTW distances within player
        trajs_p = train_data['trajectories'][train_mask]
        feats_p = train_data['features'][train_mask]
        y_p = y_scaled[train_mask]

        # Scale features
        feat_scaler = StandardScaler()
        feats_p_scaled = feat_scaler.fit_transform(feats_p)

        # Compute DTW distance matrix for this player
        D_train = compute_dtw_distances(trajs_p)

        # Bandwidth from quantile
        triu_idx = np.triu_indices(n_p, k=1)
        all_dists = D_train[triu_idx]
        bw = np.quantile(all_dists, bw_quantile)
        bw = max(bw, 1e-6)

        for t in range(3):
            target_name = TARGETS[t]

            # LOO predictions
            train_idx = np.where(train_mask)[0]
            for i in range(n_p):
                loo_mask = np.ones(n_p, dtype=bool)
                loo_mask[i] = False

                # DTW-based kernel weights
                dists = D_train[i, loo_mask]
                weights = np.exp(-0.5 * (dists / bw) ** 2)
                weights = np.maximum(weights, 1e-10)

                # Weighted Ridge
                W = np.diag(weights)
                X_loo = feats_p_scaled[loo_mask]
                y_loo = y_p[loo_mask, t]

                try:
                    model = Ridge(alpha=alpha)
                    model.fit(X_loo, y_loo, sample_weight=weights)
                    loo_preds[train_idx[i], t] = model.predict(feats_p_scaled[i:i+1])[0]
                except Exception:
                    loo_preds[train_idx[i], t] = np.mean(y_p[loo_mask, t])

            # Test predictions
            if test_mask.any():
                trajs_test_p = test_data['trajectories'][test_mask]
                feats_test_p = feat_scaler.transform(test_data['features'][test_mask])

                # Compute DTW distances: test vs train
                all_trajs = np.concatenate([trajs_p, trajs_test_p], axis=0)
                D_all = compute_dtw_distances(all_trajs)
                D_test_train = D_all[n_p:, :n_p]  # (n_test_p, n_train_p)

                for j in range(test_mask.sum()):
                    dists = D_test_train[j]
                    weights = np.exp(-0.5 * (dists / bw) ** 2)
                    weights = np.maximum(weights, 1e-10)

                    try:
                        model = Ridge(alpha=alpha)
                        model.fit(feats_p_scaled, y_p[:, t], sample_weight=weights)
                        test_idx = np.where(test_mask)[0][j]
                        test_preds[test_idx, t] = model.predict(feats_test_p[j:j+1])[0]
                    except Exception:
                        test_idx = np.where(test_mask)[0][j]
                        test_preds[test_idx, t] = np.mean(y_p[:, t])

        print(f"  Player {pid}: {n_p} train, {test_mask.sum()} test, bw={bw:.3f}")

    # Clip
    test_preds = np.clip(test_preds, 0, 1)
    loo_preds = np.clip(loo_preds, 0, 1)

    # LOO metrics
    print("\n=== DTW-Weighted Ridge LOO Results ===")
    for t in range(3):
        mse = np.mean((loo_preds[:, t] - y_scaled[:, t]) ** 2)
        print(f"  {TARGETS[t]}: MSE = {mse:.6f}")
    mean_mse = np.mean([(np.mean((loo_preds[:, t] - y_scaled[:, t]) ** 2)) for t in range(3)])
    print(f"  Mean MSE: {mean_mse:.6f}")

    return test_preds, loo_preds, mean_mse


def main():
    train_data, test_data = load_data()
    print(f"DTW trajectories: {train_data['trajectories'].shape}")
    print(f"Regression features: {train_data['features'].shape}")

    # Run DTW-weighted Ridge
    test_preds, loo_preds, loo_mse = dtw_weighted_ridge(train_data, test_data)

    # Diversity check
    anchor = pd.read_csv(SUBMISSION_DIR / "submission_2716.csv")
    print("\nDiversity vs Sub 2716:")
    for t, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
        r = np.corrcoef(test_preds[:, t], anchor[col].values)[0, 1]
        print(f"  {TARGETS[t]}: r = {r:.3f}")

    # Check vs Sub 2503 too
    anchor_2503 = pd.read_csv(SUBMISSION_DIR / "submission_2503.csv")
    print("\nDiversity vs Sub 2503:")
    for t, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
        r = np.corrcoef(test_preds[:, t], anchor_2503[col].values)[0, 1]
        print(f"  {TARGETS[t]}: r = {r:.3f}")

    # Kill criteria
    if loo_mse > 0.015:
        print(f"\nKILL: LOO too weak ({loo_mse:.6f} > 0.015)")
        # Still save and check diversity
    else:
        print(f"\nPASS: LOO = {loo_mse:.6f}")

    # Save standalone
    sub_num = get_next_submission_number()
    sub_df = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': test_preds[:, 0],
        'scaled_depth': test_preds[:, 1],
        'scaled_left_right': test_preds[:, 2],
    })
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"\nStandalone: {sub_path}")

    # Blends
    for w in [0.02, 0.05, 0.10]:
        blend_num = get_next_submission_number()
        blend_df = anchor.copy()
        for t, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
            blend_df[col] = (1 - w) * anchor[col].values + w * test_preds[:, t]
            blend_df[col] = blend_df[col].clip(0, 1)
        blend_path = SUBMISSION_DIR / f"submission_{blend_num}.csv"
        blend_df.to_csv(blend_path, index=False)
        print(f"  Sub {blend_num}: {int(w*100)}% DTW + {int((1-w)*100)}% Sub 2716")

    # Also try bandwidth sweep
    print("\n=== Bandwidth sweep ===")
    for bw_q in [0.2, 0.5]:
        test_preds_v, _, mse_v = dtw_weighted_ridge(train_data, test_data, bw_quantile=bw_q)
        test_preds_v = np.clip(test_preds_v, 0, 1)

        print(f"  bw_q={bw_q}: LOO MSE = {mse_v:.6f}")
        for t, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
            r = np.corrcoef(test_preds_v[:, t], anchor[col].values)[0, 1]
            print(f"    {TARGETS[t]}: r = {r:.3f}")

        sub_num = get_next_submission_number()
        sub_df = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': test_preds_v[:, 0],
            'scaled_depth': test_preds_v[:, 1],
            'scaled_left_right': test_preds_v[:, 2],
        })
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"  Standalone ({bw_q}): {sub_path}")

        # Blend
        for w in [0.02, 0.05]:
            blend_num = get_next_submission_number()
            blend_df = anchor.copy()
            for t, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
                blend_df[col] = (1 - w) * anchor[col].values + w * test_preds_v[:, t]
                blend_df[col] = blend_df[col].clip(0, 1)
            blend_path = SUBMISSION_DIR / f"submission_{blend_num}.csv"
            blend_df.to_csv(blend_path, index=False)
            print(f"  Sub {blend_num}: {int(w*100)}% DTW(bw={bw_q}) + {int((1-w)*100)}% Sub 2716")

    print("\nDone!")


if __name__ == "__main__":
    main()

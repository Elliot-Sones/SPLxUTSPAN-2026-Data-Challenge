"""
Learned Similarity Model - a creative approach for diversity.

Instead of using a fixed distance metric (like Gaussian kernel in our Ridge),
learn which features matter most for predicting each target.

Approach:
1. LASSO feature selection: find which features are most predictive per target
2. Use only those features for distance computation
3. Weighted k-NN with target-specific learned distance metrics
4. Ensemble multiple distance metrics

This should produce very different predictions from Ridge because:
- Different features are used per target
- Non-linear distance weighting
- No regression coefficients (pure similarity-based prediction)

Also: ElasticNet-selected features fed into a small MLP.
"""

import json
import fcntl
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
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


def load_rich_features():
    """Extract rich features including temporal statistics."""
    print("Loading data and extracting rich features...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])

    key_joints = [
        "right_wrist", "right_elbow", "right_shoulder",
        "left_wrist", "left_elbow", "left_shoulder",
        "right_hip", "left_hip", "mid_hip",
        "right_knee", "left_knee", "neck", "nose",
        "right_ankle", "left_ankle",
    ]

    # Frame windows
    target_frames = {"angle": 153, "depth": 150, "left_right": 170}
    windows = [(120, 155), (145, 175), (80, 200)]  # pre-release, release, full

    def extract(df, is_train=True):
        n = len(df)
        features_list = []
        ids, pids, targets = [], [], []

        for idx in range(n):
            row = df.iloc[idx]
            kp_data = {}
            for col in keypoint_cols:
                kp_data[col] = parse_array_string(row[col])

            feat = []

            # 1. Positions at target frames
            for jname in key_joints:
                for coord in ['x', 'y', 'z']:
                    col_name = f"{jname}_{coord}"
                    ts = kp_data.get(col_name, np.zeros(240))
                    for fname, fnum in target_frames.items():
                        feat.append(ts[fnum] if fnum < 240 else 0)

            # 2. Velocities at target frames
            for jname in key_joints:
                for coord in ['x', 'y', 'z']:
                    col_name = f"{jname}_{coord}"
                    ts = kp_data.get(col_name, np.zeros(240))
                    for fname, fnum in target_frames.items():
                        if 0 < fnum < 240:
                            feat.append((ts[fnum] - ts[fnum-1]) * 60)
                        else:
                            feat.append(0)

            # 3. Window statistics (mean, std, range) for key joints
            for jname in key_joints[:8]:  # Top 8 joints only for compactness
                for coord in ['x', 'y', 'z']:
                    col_name = f"{jname}_{coord}"
                    ts = kp_data.get(col_name, np.zeros(240))
                    for ws, we in windows:
                        segment = ts[ws:we]
                        segment = segment[~np.isnan(segment)]
                        if len(segment) > 0:
                            feat.extend([np.mean(segment), np.std(segment),
                                        np.max(segment) - np.min(segment)])
                        else:
                            feat.extend([0, 0, 0])

            # 4. Inter-joint distances at release
            rw = kp_data.get("right_wrist_x", np.zeros(240))
            rwy = kp_data.get("right_wrist_y", np.zeros(240))
            rwz = kp_data.get("right_wrist_z", np.zeros(240))
            for jname2 in ["right_elbow", "right_shoulder", "neck", "mid_hip"]:
                for fnum in [150, 153, 170]:
                    j2x = kp_data.get(f"{jname2}_x", np.zeros(240))
                    j2y = kp_data.get(f"{jname2}_y", np.zeros(240))
                    j2z = kp_data.get(f"{jname2}_z", np.zeros(240))
                    dist = np.sqrt((rw[fnum]-j2x[fnum])**2 + (rwy[fnum]-j2y[fnum])**2 + (rwz[fnum]-j2z[fnum])**2)
                    feat.append(dist if np.isfinite(dist) else 0)

            # 5. Hoop-relative global features
            mid_x = kp_data.get("mid_hip_x", np.zeros(240))
            mid_y = kp_data.get("mid_hip_y", np.zeros(240))
            mid_z = kp_data.get("mid_hip_z", np.zeros(240))
            pos = np.array([mid_x[120], mid_y[120], mid_z[120]])
            feat.append(np.linalg.norm(HOOP_POS - pos))
            feat.append(np.arctan2(HOOP_POS[1] - pos[1], HOOP_POS[0] - pos[0]))

            features_list.append(feat)
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

        X = np.array(features_list, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        result = {'X': X, 'pids': np.array(pids), 'ids': np.array(ids)}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    return extract(train_df, True), extract(test_df, False)


def lasso_weighted_knn(X_train, y_train, pids_train, X_test, pids_test):
    """LASSO-selected features + weighted k-NN per player per target."""
    # Scale targets
    target_scalers = {}
    y_scaled = np.zeros_like(y_train)
    for t in range(3):
        scaler = MinMaxScaler()
        y_scaled[:, t] = scaler.fit_transform(y_train[:, t:t+1]).ravel()
        target_scalers[t] = scaler

    unique_pids = sorted(set(pids_train.tolist()))
    n_test = len(X_test)
    test_preds = np.zeros((n_test, 3))
    loo_preds = np.zeros((len(X_train), 3))

    for pid in unique_pids:
        train_mask = pids_train == pid
        test_mask = pids_test == pid
        X_p = X_train[train_mask]
        y_p = y_scaled[train_mask]
        n_p = len(X_p)

        scaler = StandardScaler()
        X_ps = scaler.fit_transform(X_p)
        X_test_ps = scaler.transform(X_test[test_mask]) if test_mask.any() else None

        for t in range(3):
            # LASSO to find important features
            lasso = LassoCV(cv=min(5, n_p-1), max_iter=5000, n_alphas=50)
            lasso.fit(X_ps, y_p[:, t])
            feat_importance = np.abs(lasso.coef_)

            # Use top features for distance weighting
            n_top = max(10, int(np.sum(feat_importance > 0)))
            top_idx = np.argsort(feat_importance)[-n_top:]

            # Also try PCA-reduced features
            n_pca = min(15, n_p - 2, X_ps.shape[1])
            pca = PCA(n_components=n_pca)
            X_pca_train = pca.fit_transform(X_ps)
            X_pca_test = pca.transform(X_test_ps) if test_mask.any() else None

            # Multi-k ensemble
            k_values = [3, 5, 7, min(11, n_p - 1)]
            k_values = [k for k in k_values if k < n_p]

            # Method 1: LASSO-weighted k-NN
            preds_lasso = []
            for k in k_values:
                X_sel = X_ps[:, top_idx]
                knn = KNeighborsRegressor(n_neighbors=k, weights='distance',
                                           metric='euclidean')
                knn.fit(X_sel, y_p[:, t])
                if test_mask.any():
                    preds_lasso.append(knn.predict(X_test_ps[:, top_idx]))

            # Method 2: PCA k-NN
            preds_pca = []
            for k in k_values:
                knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
                knn.fit(X_pca_train, y_p[:, t])
                if test_mask.any():
                    preds_pca.append(knn.predict(X_pca_test))

            # Average all predictions
            if test_mask.any():
                all_preds = preds_lasso + preds_pca
                test_preds[test_mask, t] = np.mean(all_preds, axis=0)

            # LOO
            train_idx = np.where(train_mask)[0]
            for i, idx in enumerate(train_idx):
                loo_mask = np.ones(n_p, dtype=bool)
                loo_mask[i] = False

                loo_list = []
                for k in k_values:
                    if k < loo_mask.sum():
                        # LASSO features
                        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
                        knn.fit(X_ps[loo_mask][:, top_idx], y_p[loo_mask, t])
                        loo_list.append(knn.predict(X_ps[i:i+1, top_idx])[0])

                        # PCA features
                        knn2 = KNeighborsRegressor(n_neighbors=k, weights='distance')
                        knn2.fit(X_pca_train[loo_mask], y_p[loo_mask, t])
                        loo_list.append(knn2.predict(X_pca_train[i:i+1])[0])

                if loo_list:
                    loo_preds[idx, t] = np.mean(loo_list)

        print(f"  Player {pid}: {n_p} train, {test_mask.sum()} test, LASSO selected {n_top} features")

    test_preds = np.clip(test_preds, 0, 1)
    loo_preds = np.clip(loo_preds, 0, 1)

    return test_preds, loo_preds


def main():
    train_data, test_data = load_rich_features()
    print(f"Features: {train_data['X'].shape[1]}")

    print("\n=== LASSO-Weighted k-NN ===")
    test_preds, loo_preds = lasso_weighted_knn(
        train_data['X'], train_data['y'], train_data['pids'],
        test_data['X'], test_data['pids']
    )

    # Scale targets for LOO eval
    y_scaled = np.zeros_like(train_data['y'])
    for t in range(3):
        scaler = MinMaxScaler()
        y_scaled[:, t] = scaler.fit_transform(train_data['y'][:, t:t+1]).ravel()

    print("\nLOO Results:")
    for t in range(3):
        mse = np.mean((loo_preds[:, t] - y_scaled[:, t]) ** 2)
        print(f"  {TARGETS[t]}: MSE = {mse:.6f}")
    mean_mse = np.mean([(np.mean((loo_preds[:, t] - y_scaled[:, t]) ** 2)) for t in range(3)])
    print(f"  Mean MSE: {mean_mse:.6f}")

    # Diversity check
    anchor = pd.read_csv(SUBMISSION_DIR / "submission_2716.csv")
    print("\nDiversity vs Sub 2716:")
    for t, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
        r = np.corrcoef(test_preds[:, t], anchor[col].values)[0, 1]
        print(f"  {TARGETS[t]}: r = {r:.3f}")

    # Kill criteria
    if mean_mse > 0.015:
        print("\nKILL: LOO too weak")
        return

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
        print(f"  Sub {blend_num}: {int(w*100)}% LASSO-kNN + {int((1-w)*100)}% Sub 2716")

    print("\nDone!")


if __name__ == "__main__":
    main()

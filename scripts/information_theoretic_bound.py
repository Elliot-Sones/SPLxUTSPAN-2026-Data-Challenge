"""
Information-Theoretic Lower Bound on Prediction MSE

Estimates the aleatoric noise floor - the irreducible prediction error
due to outcome variability among biomechanically-similar shots.

Approach: For each training shot, find k nearest neighbors (same player)
in feature space. The variance of outcomes within these groups estimates
the noise that no model can overcome.
"""

import json
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0

TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

KEY_JOINTS = [
    'right_wrist', 'right_elbow', 'right_shoulder',
    'left_wrist', 'left_shoulder',
    'right_hip', 'left_hip', 'mid_hip',
    'right_knee', 'left_knee', 'neck', 'nose'
]


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


def load_train_data():
    print("Loading training data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}

    n = len(train_df)
    n_kp = len(kp_names)
    X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
    pids = []
    targets = []

    for idx, row in train_df.iterrows():
        for col_i, col in enumerate(keypoint_cols):
            arr = parse_array_string(row[col])
            X_3d[idx, :, col_i // 3, col_i % 3] = arr
        pids.append(row['participant_id'])
        targets.append([row['angle'], row['depth'], row['left_right']])

    pids = np.array(pids)
    y = np.array(targets, dtype=np.float32)
    print(f"  Loaded {n} shots, {len(np.unique(pids))} players")
    return X_3d, pids, y, kp_names, kp_index


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


def extract_features_for_target(X_3d, kp_index, target_name):
    """Extract hoop-relative position + velocity features at target frame."""
    frame = TARGET_FRAMES[target_name]
    n = X_3d.shape[0]
    all_feats = []

    for i in range(n):
        ts_hr = compute_hoop_transform(X_3d[i], kp_index)
        feats = []
        f = int(np.clip(frame, 0, 239))

        # Position + velocity at target frame for key joints
        for jname in KEY_JOINTS:
            idx = kp_index.get(jname)
            if idx is None:
                feats.extend([0.0] * 6)
                continue
            for coord in range(3):
                feats.append(float(ts_hr[f, idx, coord]))
                vel = np.gradient(ts_hr[:, idx, coord], DT)
                feats.append(float(vel[f]))

        # Summary stats (mean, std, range) for key joints
        for jname in KEY_JOINTS:
            idx = kp_index.get(jname)
            if idx is None:
                feats.extend([0.0] * 9)
                continue
            for coord in range(3):
                series = ts_hr[:, idx, coord]
                feats.append(float(np.nanmean(series)))
                feats.append(float(np.nanstd(series)))
                feats.append(float(np.nanmax(series) - np.nanmin(series)))

        all_feats.append(feats)

    X = np.array(all_feats, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def estimate_noise_floor():
    X_3d, pids, y, kp_names, kp_index = load_train_data()

    # Scale targets to [0,1]
    import joblib
    y_scaled = np.zeros_like(y)
    for t_idx, tname in enumerate(TARGETS):
        scaler = joblib.load(DATA_DIR / f"scaler_{tname}.pkl")
        y_scaled[:, t_idx] = scaler.transform(y[:, t_idx].reshape(-1, 1)).ravel()

    unique_players = np.unique(pids)
    k_values = [1, 2, 3, 5, 7, 10, 15, 20]

    results = {}

    for t_idx, tname in enumerate(TARGETS):
        print(f"\n=== Target: {tname} (frame={TARGET_FRAMES[tname]}) ===")
        X_feat = extract_features_for_target(X_3d, kp_index, tname)

        # Per-player analysis
        all_knn_mse = {k: [] for k in k_values}
        all_knn_var = {k: [] for k in k_values}
        all_neighbor_dists = []

        for pid in unique_players:
            mask = pids == pid
            X_p = X_feat[mask]
            y_p = y_scaled[mask, t_idx]
            n_p = len(y_p)

            # Standardize features per player
            scaler = StandardScaler()
            X_ps = scaler.fit_transform(X_p)

            # Compute pairwise distances
            D = cdist(X_ps, X_ps, metric='euclidean')
            np.fill_diagonal(D, np.inf)  # exclude self

            for k in k_values:
                if k >= n_p:
                    continue
                for i in range(n_p):
                    # k nearest neighbors
                    nn_idx = np.argsort(D[i])[:k]
                    nn_y = y_p[nn_idx]
                    nn_mean = np.mean(nn_y)

                    # kNN prediction MSE (LOO)
                    all_knn_mse[k].append((nn_mean - y_p[i]) ** 2)

                    # Variance within kNN group (aleatoric noise estimate)
                    group = np.append(nn_y, y_p[i])
                    all_knn_var[k].append(np.var(group))

                    if k == 5:
                        all_neighbor_dists.append(np.mean(D[i, nn_idx]))

        print(f"\n  k  |  kNN LOO MSE  |  Mean kNN Var  |  kNN Var / MSE")
        print(f"  ---|---------------|----------------|----------------")
        for k in k_values:
            if all_knn_mse[k]:
                mse = np.mean(all_knn_mse[k])
                var = np.mean(all_knn_var[k])
                ratio = var / mse if mse > 0 else 0
                print(f"  {k:2d} |  {mse:.6f}     |  {var:.6f}      |  {ratio:.3f}")
                results.setdefault(tname, {})[k] = {
                    'knn_loo_mse': float(mse),
                    'mean_knn_var': float(var),
                }

        # Detailed analysis for k=5
        if all_neighbor_dists:
            print(f"\n  k=5 neighbor distance stats:")
            dists = np.array(all_neighbor_dists)
            print(f"    mean={dists.mean():.3f}, median={np.median(dists):.3f}, "
                  f"p10={np.percentile(dists,10):.3f}, p90={np.percentile(dists,90):.3f}")

    # Aggregate across targets
    print("\n" + "=" * 70)
    print("AGGREGATE NOISE FLOOR ESTIMATES (mean across 3 targets)")
    print("=" * 70)

    for k in k_values:
        mses = []
        vars_ = []
        for tname in TARGETS:
            if tname in results and k in results[tname]:
                mses.append(results[tname][k]['knn_loo_mse'])
                vars_.append(results[tname][k]['mean_knn_var'])
        if mses:
            mean_mse = np.mean(mses)
            mean_var = np.mean(vars_)
            print(f"  k={k:2d}: kNN LOO MSE = {mean_mse:.6f}, "
                  f"Aleatoric Var = {mean_var:.6f}")

    # Best estimate: use k=5 variance as noise floor
    noise_floors = []
    for tname in TARGETS:
        if tname in results and 5 in results[tname]:
            nf = results[tname][5]['mean_knn_var']
            noise_floors.append(nf)
            print(f"\n  {tname} noise floor (k=5 var): {nf:.6f}")

    overall_noise_floor = np.mean(noise_floors) if noise_floors else 0
    print(f"\n  OVERALL NOISE FLOOR: {overall_noise_floor:.6f}")
    print(f"  Current best LB:    0.006148")
    print(f"  Gap above floor:    {0.006148 - overall_noise_floor:.6f}")
    if overall_noise_floor > 0:
        pct_above = (0.006148 - overall_noise_floor) / overall_noise_floor * 100
        print(f"  Pct above floor:    {pct_above:.1f}%")

    # Additional: conditional variance analysis
    # Split shots into "easy" (low variance neighbors) and "hard" (high variance)
    print("\n" + "=" * 70)
    print("CONDITIONAL ANALYSIS: Easy vs Hard shots (k=5)")
    print("=" * 70)

    for t_idx, tname in enumerate(TARGETS):
        X_feat = extract_features_for_target(X_3d, kp_index, tname)
        shot_variances = []

        for pid in unique_players:
            mask = pids == pid
            X_p = X_feat[mask]
            y_p = y_scaled[mask, t_idx]
            n_p = len(y_p)
            if n_p < 6:
                continue

            scaler = StandardScaler()
            X_ps = scaler.fit_transform(X_p)
            D = cdist(X_ps, X_ps, metric='euclidean')
            np.fill_diagonal(D, np.inf)

            for i in range(n_p):
                nn_idx = np.argsort(D[i])[:5]
                group = np.append(y_p[nn_idx], y_p[i])
                shot_variances.append(np.var(group))

        shot_variances = np.array(shot_variances)
        median_var = np.median(shot_variances)
        easy = shot_variances[shot_variances <= median_var]
        hard = shot_variances[shot_variances > median_var]

        print(f"\n  {tname}:")
        print(f"    Easy half (low local var): mean var = {easy.mean():.6f}")
        print(f"    Hard half (high local var): mean var = {hard.mean():.6f}")
        print(f"    Ratio hard/easy: {hard.mean()/easy.mean():.1f}x")

    # Per-player noise floor
    print("\n" + "=" * 70)
    print("PER-PLAYER NOISE FLOOR (k=5)")
    print("=" * 70)

    for pid in sorted(unique_players):
        mask = pids == pid
        n_p = mask.sum()
        player_nf = []
        for t_idx, tname in enumerate(TARGETS):
            X_feat = extract_features_for_target(X_3d, kp_index, tname)
            X_p = X_feat[mask]
            y_p = y_scaled[mask, t_idx]
            if n_p < 6:
                continue

            scaler = StandardScaler()
            X_ps = scaler.fit_transform(X_p)
            D = cdist(X_ps, X_ps, metric='euclidean')
            np.fill_diagonal(D, np.inf)

            vars_ = []
            for i in range(n_p):
                nn_idx = np.argsort(D[i])[:5]
                group = np.append(y_p[nn_idx], y_p[i])
                vars_.append(np.var(group))
            player_nf.append(np.mean(vars_))

        if player_nf:
            print(f"  P{pid}: {' | '.join(f'{TARGETS[i]}: {v:.6f}' for i,v in enumerate(player_nf))}"
                  f" | mean: {np.mean(player_nf):.6f} (n={n_p})")

    return results, overall_noise_floor


if __name__ == "__main__":
    results, noise_floor = estimate_noise_floor()

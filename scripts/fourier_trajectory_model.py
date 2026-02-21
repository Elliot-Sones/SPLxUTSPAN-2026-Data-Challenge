"""
Fourier Trajectory Model

A fundamentally different approach to the SPLxUTSPAN basketball shot prediction challenge.
Instead of extracting position/velocity features at specific frames, this model:

1. Takes the FFT of key joint trajectories over the shooting motion window (frames 60-200)
2. Uses the magnitudes of the first N Fourier coefficients as features
3. This captures FREQUENCY content (rhythm, tempo, periodicity) of the shooting motion
4. These frequency features are orthogonal to single-frame position features

Pipeline:
- 15 key joints x 3 coords x 10 Fourier coefficients = 450 raw features
- Per-player StandardScaler
- Per-player, per-target PLS compression (honest LOO: PLS refit per fold)
- Locally weighted Ridge regression (same kernel approach as baseline)

Goal: DIVERSITY - low correlation with existing Ridge predictions (Sub 2716).
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
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
DT = 1.0 / 60.0

# 15 key joints (replacing 'head' with 'nose' since 'head' doesn't exist in the data)
KEY_JOINTS = [
    'right_wrist', 'right_elbow', 'right_shoulder',
    'left_wrist', 'left_elbow', 'left_shoulder',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'mid_hip', 'nose', 'neck',
]

# Fourier parameters
FRAME_START = 60   # Start of shooting motion window
FRAME_END = 200    # End of shooting motion window
N_COEFFS = 10      # Number of Fourier coefficients to keep

# Per-target bandwidth (reuse proven values from existing pipeline)
BEST_BW = {"angle": 0.80, "depth": 0.55, "left_right": 0.30}

# Player-specific overrides
PLAYER_OVERRIDES = {(1, "left_right"): {"bw": 0.15}}

# PLS components for Fourier features
FOURIER_PLS_COMPONENTS = 5

# Number of PLS components for raw timeseries (standard augmentation)
RAW_PLS_COMPONENTS = 10


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


def safe_interpolate(x):
    """Interpolate NaN/inf values in a 1D array."""
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return x


def load_data():
    print("Loading data...", flush=True)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    n_kp_cols = len(keypoint_cols)
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


def extract_fourier_features(ts_3d, kp_index):
    """
    Extract Fourier features from key joint trajectories.

    For each of 15 key joints x 3 coords:
    - Take the trajectory over frames FRAME_START:FRAME_END
    - Interpolate NaN/inf values
    - Apply Hann window (reduces spectral leakage)
    - Compute FFT
    - Take magnitudes of first N_COEFFS coefficients (skip DC component)

    Returns: array of shape (15 * 3 * N_COEFFS,) = (450,)
    """
    n_frames = FRAME_END - FRAME_START
    window = np.hanning(n_frames)
    feats = []

    for joint_name in KEY_JOINTS:
        idx = kp_index.get(joint_name)
        if idx is None:
            # Joint not found - zero features
            feats.extend([0.0] * (3 * N_COEFFS))
            continue

        for coord in range(3):
            # Get trajectory slice
            traj = ts_3d[FRAME_START:FRAME_END, idx, coord].copy()

            # Interpolate bad values
            traj = safe_interpolate(traj)

            # Remove mean (we want AC components, not DC)
            traj = traj - np.mean(traj)

            # Apply Hann window to reduce spectral leakage
            traj_windowed = traj * window

            # FFT
            fft_vals = np.fft.rfft(traj_windowed)

            # Take magnitudes of first N_COEFFS (skip index 0 = DC since we removed mean)
            magnitudes = np.abs(fft_vals[1:N_COEFFS + 1])

            # Pad if not enough coefficients (shouldn't happen with 140 frames)
            if len(magnitudes) < N_COEFFS:
                magnitudes = np.pad(magnitudes, (0, N_COEFFS - len(magnitudes)))

            feats.extend(magnitudes.tolist())

    return np.array(feats, dtype=np.float32)


def extract_fourier_features_multiscale(ts_3d, kp_index):
    """
    Multi-scale Fourier features: extract FFT at different temporal windows.
    - Full motion window: frames 60-200 (captures overall rhythm)
    - Pre-release window: frames 100-170 (captures shot preparation)
    - Release window: frames 130-190 (captures release mechanics)

    Each window provides 15 joints * 3 coords * N_COEFFS features.
    Total: 3 windows * 450 = 1350 features (before PLS compression).
    """
    windows = [
        (60, 200),    # Full shooting motion
        (100, 170),   # Pre-release preparation
        (130, 190),   # Release phase
    ]

    all_feats = []
    for start, end in windows:
        n_frames = end - start
        hann = np.hanning(n_frames)

        for joint_name in KEY_JOINTS:
            idx = kp_index.get(joint_name)
            if idx is None:
                all_feats.extend([0.0] * (3 * N_COEFFS))
                continue

            for coord in range(3):
                traj = ts_3d[start:end, idx, coord].copy()
                traj = safe_interpolate(traj)
                traj = traj - np.mean(traj)
                traj_w = traj * hann
                fft_vals = np.fft.rfft(traj_w)
                mags = np.abs(fft_vals[1:N_COEFFS + 1])
                if len(mags) < N_COEFFS:
                    mags = np.pad(mags, (0, N_COEFFS - len(mags)))
                all_feats.extend(mags.tolist())

    return np.array(all_feats, dtype=np.float32)


def honest_loo_pipeline(X_fourier_tr, y_scaled, pid_bw, alpha=10.0, n_pls=5):
    """
    Honest LOO: for each held-out sample, refit PLS on the remaining samples,
    then predict with locally-weighted Ridge.

    This avoids PLS data leakage that inflates LOO scores.
    """
    n = len(X_fourier_tr)
    oof = np.zeros(n)

    for i in range(n):
        # Leave one out
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_loo = X_fourier_tr[mask]
        y_loo = y_scaled[mask]

        # Fit PLS on LOO data
        nc = min(n_pls, X_loo.shape[1], len(X_loo) - 2)
        nc = max(1, nc)

        pls_sc = StandardScaler()
        X_loo_sc = pls_sc.fit_transform(X_loo)
        X_i_sc = pls_sc.transform(X_fourier_tr[i:i+1])

        pls = PLSRegression(n_components=nc)
        pls.fit(X_loo_sc, y_loo)
        X_loo_pls = pls.transform(X_loo_sc)
        X_i_pls = pls.transform(X_i_sc)

        # StandardScaler on PLS features
        feat_sc = StandardScaler()
        X_loo_final = feat_sc.fit_transform(X_loo_pls)
        X_i_final = feat_sc.transform(X_i_pls)

        # Distance-based kernel
        dists = cdist(X_i_final, X_loo_final, 'euclidean')[0]
        all_dists_loo = cdist(X_loo_final, X_loo_final, 'euclidean')
        triu = all_dists_loo[np.triu_indices(len(X_loo_final), k=1)]
        sigma = np.quantile(triu, pid_bw) if len(triu) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        weights = np.exp(-dists**2 / (2 * sigma**2))
        if weights.sum() < 1e-10:
            oof[i] = np.mean(y_loo)
            continue

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_loo_final, y_loo, sample_weight=weights)
        oof[i] = ridge.predict(X_i_final)[0]

    return oof


def fast_loo_pipeline(X_fourier_tr, y_scaled, pid_bw, alpha=10.0, n_pls=5):
    """
    Fast (leaky) LOO for quick screening. PLS fit on all data.
    Only used for rapid iteration, not final evaluation.
    """
    n = len(X_fourier_tr)

    # Fit PLS on all data (leaky but fast)
    nc = min(n_pls, X_fourier_tr.shape[1], n - 2)
    nc = max(1, nc)

    pls_sc = StandardScaler()
    X_sc = pls_sc.fit_transform(X_fourier_tr)
    pls = PLSRegression(n_components=nc)
    pls.fit(X_sc, y_scaled)
    X_pls = pls.transform(X_sc)

    feat_sc = StandardScaler()
    X_final = feat_sc.fit_transform(X_pls)

    D = cdist(X_final, X_final, 'euclidean')
    triu = D[np.triu_indices(n, k=1)]
    sigma = np.quantile(triu, pid_bw) if len(triu) > 0 else 1.0
    sigma = max(sigma, 1e-6)

    oof = np.zeros(n)
    for i in range(n):
        weights = np.exp(-D[i, :]**2 / (2 * sigma**2))
        weights[i] = 0
        if weights.sum() < 1e-10:
            oof[i] = np.mean(y_scaled)
            continue
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_final, y_scaled, sample_weight=weights)
        oof[i] = ridge.predict(X_final[i:i+1])[0]

    return oof


def predict_test(X_fourier_tr, X_fourier_te, y_scaled, pid_bw, alpha=10.0, n_pls=5):
    """
    Generate test predictions (PLS fit on all training data for this player).
    """
    n_tr = len(X_fourier_tr)
    n_te = len(X_fourier_te)
    if n_te == 0:
        return np.zeros(0)

    nc = min(n_pls, X_fourier_tr.shape[1], n_tr - 2)
    nc = max(1, nc)

    pls_sc = StandardScaler()
    X_tr_sc = pls_sc.fit_transform(X_fourier_tr)
    X_te_sc = pls_sc.transform(X_fourier_te)

    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr_sc, y_scaled)
    X_tr_pls = pls.transform(X_tr_sc)
    X_te_pls = pls.transform(X_te_sc)

    feat_sc = StandardScaler()
    X_tr_final = feat_sc.fit_transform(X_tr_pls)
    X_te_final = feat_sc.transform(X_te_pls)

    D_tr = cdist(X_tr_final, X_tr_final, 'euclidean')
    triu = D_tr[np.triu_indices(n_tr, k=1)]
    sigma = np.quantile(triu, pid_bw) if len(triu) > 0 else 1.0
    sigma = max(sigma, 1e-6)

    D_te = cdist(X_te_final, X_tr_final, 'euclidean')
    preds = np.zeros(n_te)
    for j in range(n_te):
        weights = np.exp(-D_te[j, :]**2 / (2 * sigma**2))
        if weights.sum() < 1e-10:
            preds[j] = np.mean(y_scaled)
            continue
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr_final, y_scaled, sample_weight=weights)
        preds[j] = ridge.predict(X_te_final[j:j+1])[0]

    return preds


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("FOURIER TRAJECTORY MODEL", flush=True)
    print("Frequency-domain features from joint trajectories", flush=True)
    print("=" * 70, flush=True)

    # Load data
    train_data, test_data = load_data()
    y_train = train_data['y']
    kp_index = train_data['kp_index']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    unique_pids = sorted(np.unique(pids_train))
    n_train = len(pids_train)
    n_test = len(pids_test)

    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}

    # ====================================================================
    # Step 1: Extract Fourier features for all samples
    # ====================================================================
    print("\nExtracting multi-scale Fourier features...", flush=True)
    t1 = time.time()

    fourier_train = np.zeros((n_train, len(KEY_JOINTS) * 3 * N_COEFFS * 3), dtype=np.float32)
    fourier_test = np.zeros((n_test, len(KEY_JOINTS) * 3 * N_COEFFS * 3), dtype=np.float32)

    for i in range(n_train):
        fourier_train[i] = extract_fourier_features_multiscale(train_data['X_3d'][i], kp_index)
    for i in range(n_test):
        fourier_test[i] = extract_fourier_features_multiscale(test_data['X_3d'][i], kp_index)

    # Clean NaN/inf
    fourier_train = np.nan_to_num(fourier_train, nan=0.0, posinf=0.0, neginf=0.0)
    fourier_test = np.nan_to_num(fourier_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Fourier features shape: {fourier_train.shape} ({time.time()-t1:.1f}s)", flush=True)
    print(f"  Non-zero features per sample: {np.mean(np.sum(fourier_train != 0, axis=1)):.0f} / {fourier_train.shape[1]}", flush=True)

    # ====================================================================
    # Step 2: Quick screening with fast (leaky) LOO
    # ====================================================================
    print("\n" + "-" * 70, flush=True)
    print("QUICK SCREEN: Fast (leaky) LOO", flush=True)
    print("-" * 70, flush=True)

    fast_loo_mses = {}
    for target in TARGETS:
        tidx = target_idx[target]
        y_raw = y_train[:, tidx]
        y_sc = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()

        oof = np.zeros(n_train)
        for pid in unique_pids:
            tr_mask = pids_train == pid
            tr_idx = np.where(tr_mask)[0]
            override = PLAYER_OVERRIDES.get((pid, target), {})
            bw = override.get("bw", BEST_BW[target])

            oof_player = fast_loo_pipeline(
                fourier_train[tr_mask], y_sc[tr_mask], bw,
                alpha=10.0, n_pls=FOURIER_PLS_COMPONENTS)
            oof[tr_idx] = oof_player

        mse = np.mean((oof - y_sc) ** 2)
        fast_loo_mses[target] = mse
        print(f"  {target}: fast LOO = {mse:.6f}", flush=True)

    fast_mean = np.mean(list(fast_loo_mses.values()))
    print(f"  Mean fast LOO: {fast_mean:.6f}", flush=True)
    print(f"  NOTE: This is inflated by PLS leakage. Honest LOO will be higher.", flush=True)

    # ====================================================================
    # Step 3: Honest LOO (PLS refit per fold) - the real evaluation
    # ====================================================================
    print("\n" + "-" * 70, flush=True)
    print("HONEST LOO: PLS refit per held-out sample", flush=True)
    print("-" * 70, flush=True)

    honest_oof = {}
    honest_mses = {}
    test_preds = {}

    for target in TARGETS:
        t2 = time.time()
        tidx = target_idx[target]
        y_raw = y_train[:, tidx]
        y_sc = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()

        oof = np.zeros(n_train)
        tpred = np.zeros(n_test)

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            tr_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]

            override = PLAYER_OVERRIDES.get((pid, target), {})
            bw = override.get("bw", BEST_BW[target])

            # Honest LOO
            oof_player = honest_loo_pipeline(
                fourier_train[tr_mask], y_sc[tr_mask], bw,
                alpha=10.0, n_pls=FOURIER_PLS_COMPONENTS)
            oof[tr_idx] = oof_player

            # Test predictions
            if len(te_idx) > 0:
                tpred_player = predict_test(
                    fourier_train[tr_mask], fourier_test[te_mask],
                    y_sc[tr_mask], bw,
                    alpha=10.0, n_pls=FOURIER_PLS_COMPONENTS)
                tpred[te_idx] = tpred_player

        mse = np.mean((oof - y_sc) ** 2)
        honest_oof[target] = oof
        honest_mses[target] = mse
        test_preds[target] = tpred

        print(f"\n  {target}: honest LOO = {mse:.6f} [{time.time()-t2:.1f}s]", flush=True)
        for pid in unique_pids:
            mask = pids_train == pid
            pmse = np.mean((oof[mask] - y_sc[mask]) ** 2)
            extra = ""
            if (pid, target) in PLAYER_OVERRIDES:
                extra = f" [override: {PLAYER_OVERRIDES[(pid, target)]}]"
            print(f"    P{pid}: {pmse:.6f}{extra}", flush=True)

    honest_mean = np.mean(list(honest_mses.values()))
    print(f"\n  Mean honest LOO: {honest_mean:.6f}", flush=True)

    # ====================================================================
    # Kill check 1: Is honest LOO > 0.015?
    # ====================================================================
    if honest_mean > 0.015:
        print(f"\n*** KILL: honest LOO {honest_mean:.6f} > 0.015 threshold ***", flush=True)
        print("Model too weak. Aborting.", flush=True)
        return

    # ====================================================================
    # Step 4: Diversity check vs Sub 2716
    # ====================================================================
    print("\n" + "-" * 70, flush=True)
    print("DIVERSITY CHECK: Correlation with Sub 2716", flush=True)
    print("-" * 70, flush=True)

    sub2716 = pd.read_csv(SUBMISSION_DIR / "submission_2716.csv")
    # Align by id
    sub2716_indexed = sub2716.set_index("id")

    corrs = {}
    for target in TARGETS:
        col = f"scaled_{target}"
        # Get predictions in same order as test_data ids
        anchor_vals = np.array([sub2716_indexed.loc[tid, col] for tid in test_data['ids']])
        our_vals = test_preds[target]
        r = np.corrcoef(anchor_vals, our_vals)[0, 1]
        corrs[target] = r
        print(f"  {target}: r = {r:.4f}", flush=True)

    mean_corr = np.mean(list(corrs.values()))
    print(f"  Mean correlation: {mean_corr:.4f}", flush=True)

    all_high = all(c > 0.90 for c in corrs.values())
    if all_high:
        print(f"\n*** KILL: All correlations > 0.90. No diversity. ***", flush=True)
        print("Predictions too similar to Sub 2716. Aborting.", flush=True)
        return

    print(f"\n  PROCEED: At least one target has r < 0.90 (diversity exists)", flush=True)

    # ====================================================================
    # Step 5: Generate submissions
    # ====================================================================
    print("\n" + "=" * 70, flush=True)
    print("GENERATING SUBMISSIONS", flush=True)
    print("=" * 70, flush=True)

    # Standalone submission
    sub_num = get_next_submission_number()
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df = pd.DataFrame({
        "id": test_data['ids'],
        "scaled_angle": test_preds['angle'],
        "scaled_depth": test_preds['depth'],
        "scaled_left_right": test_preds['left_right'],
    })
    sub_df.to_csv(sub_path, index=False)
    print(f"Sub {sub_num}: Fourier standalone (honest LOO {honest_mean:.6f})", flush=True)

    # Load Sub 2716 anchor values aligned to test ids
    anchor_angle = np.array([sub2716_indexed.loc[tid, "scaled_angle"] for tid in test_data['ids']])
    anchor_depth = np.array([sub2716_indexed.loc[tid, "scaled_depth"] for tid in test_data['ids']])
    anchor_lr = np.array([sub2716_indexed.loc[tid, "scaled_left_right"] for tid in test_data['ids']])

    # 2% blend - all targets
    w = 0.02
    sub_num_blend = get_next_submission_number()
    blend_path = SUBMISSION_DIR / f"submission_{sub_num_blend}.csv"
    blend_df = pd.DataFrame({
        "id": test_data['ids'],
        "scaled_angle": w * test_preds['angle'] + (1 - w) * anchor_angle,
        "scaled_depth": w * test_preds['depth'] + (1 - w) * anchor_depth,
        "scaled_left_right": w * test_preds['left_right'] + (1 - w) * anchor_lr,
    })
    blend_df.to_csv(blend_path, index=False)
    print(f"Sub {sub_num_blend}: 2% Fourier + 98% Sub 2716 (all targets)", flush=True)

    # Per-target 2% blends (use Fourier only for the target with lowest correlation)
    for target in TARGETS:
        sub_num_t = get_next_submission_number()
        t_path = SUBMISSION_DIR / f"submission_{sub_num_t}.csv"
        t_df = pd.DataFrame({
            "id": test_data['ids'],
            "scaled_angle": (w * test_preds['angle'] + (1 - w) * anchor_angle) if target == 'angle' else anchor_angle,
            "scaled_depth": (w * test_preds['depth'] + (1 - w) * anchor_depth) if target == 'depth' else anchor_depth,
            "scaled_left_right": (w * test_preds['left_right'] + (1 - w) * anchor_lr) if target == 'left_right' else anchor_lr,
        })
        t_df.to_csv(t_path, index=False)
        print(f"Sub {sub_num_t}: 2% Fourier {target} only + 98% Sub 2716", flush=True)

    # Also try 5% and 10% blends for the most diverse target
    most_diverse_target = min(corrs, key=corrs.get)
    print(f"\nMost diverse target: {most_diverse_target} (r={corrs[most_diverse_target]:.4f})", flush=True)

    for blend_w in [0.05, 0.10]:
        sub_num_w = get_next_submission_number()
        w_path = SUBMISSION_DIR / f"submission_{sub_num_w}.csv"
        w_df = pd.DataFrame({
            "id": test_data['ids'],
            "scaled_angle": (blend_w * test_preds['angle'] + (1 - blend_w) * anchor_angle)
                if most_diverse_target == 'angle' else anchor_angle,
            "scaled_depth": (blend_w * test_preds['depth'] + (1 - blend_w) * anchor_depth)
                if most_diverse_target == 'depth' else anchor_depth,
            "scaled_left_right": (blend_w * test_preds['left_right'] + (1 - blend_w) * anchor_lr)
                if most_diverse_target == 'left_right' else anchor_lr,
        })
        w_df.to_csv(w_path, index=False)
        print(f"Sub {sub_num_w}: {int(blend_w*100)}% Fourier {most_diverse_target} + {int((1-blend_w)*100)}% Sub 2716", flush=True)

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Fourier features: {fourier_train.shape[1]} raw -> {FOURIER_PLS_COMPONENTS} PLS per player", flush=True)
    print(f"Fast (leaky) LOO:  {fast_mean:.6f}", flush=True)
    print(f"Honest LOO:        {honest_mean:.6f}", flush=True)
    print(f"PLS leakage ratio: {honest_mean / max(fast_mean, 1e-10):.2f}x", flush=True)
    print(f"\nDiversity vs Sub 2716:", flush=True)
    for target in TARGETS:
        print(f"  {target}: r = {corrs[target]:.4f}, honest LOO = {honest_mses[target]:.6f}", flush=True)
    print(f"\nTotal runtime: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

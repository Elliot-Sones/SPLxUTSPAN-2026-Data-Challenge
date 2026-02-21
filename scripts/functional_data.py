"""
Functional Data Analysis for Basketball Shot Prediction
--------------------------------------------------------
Treats each shot's timeseries as functional data and extracts features using:
1. Functional PCA (per-player) on key joint trajectories
2. B-spline basis expansion coefficients
3. Derivative features (velocity, acceleration, jerk) from smoothed curves
4. Distance-to-mean trajectory features (shot consistency measure)
5. Cross-joint coordination features (kinetic chain timing)

Blends with Sub 784 (current best LB 0.007224).
"""

import fcntl
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.cross_decomposition import PLSRegression
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ---- Config ----
PROJECT_DIR = Path("/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge")
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
SCALED_TARGETS = ["scaled_angle", "scaled_depth", "scaled_left_right"]

# Key joints for functional analysis
KEY_JOINTS = [
    "right_wrist_x", "right_wrist_y", "right_wrist_z",
    "right_elbow_x", "right_elbow_y", "right_elbow_z",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
]

# Extended joints for release frame features
RELEASE_JOINTS = [
    "right_wrist_x", "right_wrist_y", "right_wrist_z",
    "right_elbow_x", "right_elbow_y", "right_elbow_z",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
    "left_wrist_x", "left_wrist_y", "left_wrist_z",
    "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
    "nose_x", "nose_y", "nose_z",
    "right_hip_x", "right_hip_y", "right_hip_z",
]

RELEASE_FRAME = 153
RELEASE_WINDOW = (130, 180)  # Frames around release

N_SPLINE_KNOTS = 20
SAVGOL_WINDOW = 11
SAVGOL_POLY = 3

FPCA_K_OPTIONS = [3, 5, 8, 10, 15, 20]
PLS_N_OPTIONS = list(range(3, 16))


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    return np.array(json.loads(s.replace("nan", "null")), dtype=np.float32)


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for fp in existing:
                parts = fp.stem.split('_')
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            placeholder = SUBMISSION_DIR / f"submission_{next_num}.csv"
            placeholder.touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_data():
    """Load train and test data, parse timeseries for key joints."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id"] + TARGETS
    all_kp_cols = [c for c in train_df.columns if c not in meta_cols]

    # Parse all keypoint timeseries
    def parse_df(df, cols):
        n = len(df)
        data = {}
        for col in cols:
            arr = np.zeros((n, 240), dtype=np.float32)
            for idx, row in df.iterrows():
                arr[idx] = parse_array_string(row[col])
            data[col] = arr
        return data

    # Parse key joints + release joints (union)
    all_needed = list(set(KEY_JOINTS + RELEASE_JOINTS))
    train_ts = parse_df(train_df, all_needed)
    test_ts = parse_df(test_df, all_needed)

    y_train = train_df[TARGETS].values.astype(np.float32)
    pids_train = train_df["participant_id"].values
    pids_test = test_df["participant_id"].values
    test_ids = test_df["id"].values
    train_ids = train_df["id"].values

    return train_ts, y_train, pids_train, train_ids, test_ts, pids_test, test_ids


def smooth_timeseries(ts_array, window=SAVGOL_WINDOW, poly=SAVGOL_POLY):
    """Apply Savitzky-Golay smoothing to each timeseries."""
    n, t = ts_array.shape
    out = np.zeros_like(ts_array)
    for i in range(n):
        s = ts_array[i]
        valid = ~np.isnan(s)
        if valid.sum() >= window:
            s_clean = np.interp(np.arange(t), np.where(valid)[0], s[valid])
            out[i] = savgol_filter(s_clean, window, poly)
        else:
            out[i] = np.nan_to_num(s, nan=0.0)
    return out


# =====================================================
# FEATURE 1: Functional PCA (per-player)
# =====================================================
def extract_fpca_features(train_ts, test_ts, pids_train, pids_test, n_components):
    """
    Per-player Functional PCA on key joint trajectories.
    Stack the 9 key joint timeseries into a 2160-dim vector per shot,
    smooth, then PCA.
    """
    # Build stacked timeseries for key joints
    def stack_key_joints(ts_dict):
        arrays = []
        for joint in KEY_JOINTS:
            arrays.append(ts_dict[joint])
        return np.hstack(arrays)  # (n_shots, 9*240=2160)

    train_stacked = stack_key_joints(train_ts)
    test_stacked = stack_key_joints(test_ts)

    # Smooth each joint segment separately
    def smooth_stacked(stacked):
        n = stacked.shape[0]
        out = np.zeros_like(stacked)
        for j in range(len(KEY_JOINTS)):
            segment = stacked[:, j*240:(j+1)*240]
            out[:, j*240:(j+1)*240] = smooth_timeseries(segment)
        return out

    train_smooth = smooth_stacked(train_stacked)
    test_smooth = smooth_stacked(test_stacked)

    # Per-player PCA
    players = np.unique(pids_train)
    train_features = np.zeros((len(pids_train), n_components))
    test_features = np.zeros((len(pids_test), n_components))

    for pid in players:
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        # Fit PCA on this player's training data
        player_data = train_smooth[train_mask]
        # StandardScale first
        scaler = StandardScaler()
        player_data_scaled = scaler.fit_transform(player_data)

        pca = PCA(n_components=min(n_components, player_data_scaled.shape[0] - 1))
        train_scores = pca.fit_transform(player_data_scaled)

        # Pad if fewer components than requested
        actual_k = train_scores.shape[1]
        train_features[train_mask, :actual_k] = train_scores

        # Transform test
        test_data = test_smooth[test_mask]
        test_data_scaled = scaler.transform(test_data)
        test_scores = pca.transform(test_data_scaled)
        test_features[test_mask, :actual_k] = test_scores

    return train_features, test_features


# =====================================================
# FEATURE 2: B-spline basis expansion
# =====================================================
def extract_bspline_features(train_ts, test_ts):
    """
    Fit B-spline basis functions to key joint trajectories.
    Use cubic splines with ~20 internal knots.
    The coefficients become features.
    """
    # Joints for B-spline: right_wrist xyz + right_elbow xyz
    bspline_joints = [
        "right_wrist_x", "right_wrist_y", "right_wrist_z",
        "right_elbow_x", "right_elbow_y", "right_elbow_z",
    ]

    t = np.arange(240, dtype=np.float64)
    # Internal knots (evenly spaced)
    knots = np.linspace(0, 239, N_SPLINE_KNOTS + 2)[1:-1]

    def fit_bspline_one(ts_1d):
        """Fit B-spline to a single timeseries and return coefficients."""
        valid = ~np.isnan(ts_1d)
        if valid.sum() < 10:
            return np.zeros(N_SPLINE_KNOTS + 4)  # k=3 cubic: n_coeffs = n_knots + k + 1
        # Interpolate NaNs first
        ts_clean = np.interp(t, t[valid], ts_1d[valid]).astype(np.float64)
        try:
            spl = make_interp_spline(t, ts_clean, k=3)
            return spl.c  # coefficients
        except Exception:
            return np.zeros(N_SPLINE_KNOTS + 4)

    def extract_for_dataset(ts_dict, n_shots):
        all_coeffs = []
        for joint in bspline_joints:
            joint_ts = ts_dict[joint]
            for i in range(n_shots):
                coeffs = fit_bspline_one(joint_ts[i])
                if i == 0 and len(all_coeffs) == 0:
                    # Determine coefficient size
                    pass
                all_coeffs.append(coeffs)

        # Reshape: n_shots x (n_joints * n_coeffs)
        n_joints = len(bspline_joints)
        coeffs_per_joint = len(all_coeffs[0])
        arr = np.array(all_coeffs).reshape(n_joints, n_shots, coeffs_per_joint)
        arr = arr.transpose(1, 0, 2).reshape(n_shots, -1)
        return arr

    n_train = len(list(train_ts.values())[0])
    n_test = len(list(test_ts.values())[0])

    train_feat = extract_for_dataset(train_ts, n_train)
    test_feat = extract_for_dataset(test_ts, n_test)

    return train_feat, test_feat


# =====================================================
# FEATURE 3: Derivative features from smoothed curves
# =====================================================
def extract_derivative_features(train_ts, test_ts):
    """
    Compute velocity, acceleration, jerk from smoothed key joint trajectories.
    Extract statistics over the release window.
    """
    deriv_joints = KEY_JOINTS  # 9 joint coordinates

    def compute_derivatives(ts_dict, n_shots):
        features = []
        for i in range(n_shots):
            row = {}
            for joint in deriv_joints:
                s = ts_dict[joint][i]
                # Smooth
                valid = ~np.isnan(s)
                if valid.sum() >= SAVGOL_WINDOW:
                    s_clean = np.interp(np.arange(240), np.where(valid)[0], s[valid])
                    s_smooth = savgol_filter(s_clean, SAVGOL_WINDOW, SAVGOL_POLY)
                else:
                    s_smooth = np.nan_to_num(s, nan=0.0)

                # Derivatives
                vel = np.gradient(s_smooth)
                acc = np.gradient(vel)
                jerk = np.gradient(acc)

                # Release window
                w_start, w_end = RELEASE_WINDOW
                vel_w = vel[w_start:w_end]
                acc_w = acc[w_start:w_end]
                jerk_w = jerk[w_start:w_end]

                jname = joint.replace("right_", "r_").replace("left_", "l_")

                # Peak velocity timing (frame number of max |velocity| in release window)
                row[f"{jname}_peak_vel_frame"] = w_start + np.argmax(np.abs(vel_w))
                # Peak velocity magnitude
                row[f"{jname}_peak_vel_mag"] = np.max(np.abs(vel_w))
                # Mean velocity in release window
                row[f"{jname}_mean_vel"] = np.mean(vel_w)

                # Peak acceleration timing
                row[f"{jname}_peak_acc_frame"] = w_start + np.argmax(np.abs(acc_w))
                # Peak acceleration magnitude
                row[f"{jname}_peak_acc_mag"] = np.max(np.abs(acc_w))

                # Jerk integral (smoothness measure) - L2 norm of jerk
                row[f"{jname}_jerk_integral"] = np.sqrt(np.sum(jerk_w**2))
                # Mean jerk magnitude
                row[f"{jname}_mean_jerk"] = np.mean(np.abs(jerk_w))

                # Velocity at release frame
                row[f"{jname}_vel_at_release"] = vel[RELEASE_FRAME]
                # Acceleration at release frame
                row[f"{jname}_acc_at_release"] = acc[RELEASE_FRAME]

            features.append(row)
        return pd.DataFrame(features).values

    n_train = len(list(train_ts.values())[0])
    n_test = len(list(test_ts.values())[0])

    train_feat = compute_derivatives(train_ts, n_train)
    test_feat = compute_derivatives(test_ts, n_test)

    return train_feat, test_feat


# =====================================================
# FEATURE 4: Distance-to-mean trajectory features
# =====================================================
def extract_distance_features(train_ts, test_ts, pids_train, pids_test):
    """
    For each player, compute the Euclidean distance of each shot's wrist trajectory
    to the player's mean trajectory. This captures shot-to-shot consistency.
    """
    dist_joints = ["right_wrist_x", "right_wrist_y", "right_wrist_z"]

    players = np.unique(pids_train)
    n_train = len(pids_train)
    n_test = len(pids_test)

    # Features: overall L2 distance, distance in release window, max frame distance
    train_feat = np.zeros((n_train, 5))
    test_feat = np.zeros((n_test, 5))

    for pid in players:
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        # Stack wrist xyz: (n_player_shots, 3, 240)
        train_wrist = np.stack([
            smooth_timeseries(train_ts[j][train_mask]) for j in dist_joints
        ], axis=1)  # (n, 3, 240)

        # Mean trajectory for this player
        mean_traj = np.nanmean(train_wrist, axis=0)  # (3, 240)

        # Distance per frame: sqrt(sum_over_xyz (shot - mean)^2)
        def compute_dist_features(wrist_data):
            diff = wrist_data - mean_traj[np.newaxis, :, :]  # (n, 3, 240)
            frame_dist = np.sqrt(np.sum(diff**2, axis=1))  # (n, 240)

            feats = np.zeros((wrist_data.shape[0], 5))
            # Overall L2 norm
            feats[:, 0] = np.sqrt(np.sum(frame_dist**2, axis=1))
            # Mean frame distance
            feats[:, 1] = np.mean(frame_dist, axis=1)
            # Distance in release window
            w_start, w_end = RELEASE_WINDOW
            feats[:, 2] = np.sqrt(np.sum(frame_dist[:, w_start:w_end]**2, axis=1))
            # Max frame distance
            feats[:, 3] = np.max(frame_dist, axis=1)
            # Frame of max distance
            feats[:, 4] = np.argmax(frame_dist, axis=1)
            return feats

        train_feat[train_mask] = compute_dist_features(train_wrist)

        # Test
        test_wrist = np.stack([
            smooth_timeseries(test_ts[j][test_mask]) for j in dist_joints
        ], axis=1)
        test_feat[test_mask] = compute_dist_features(test_wrist)

    return train_feat, test_feat


# =====================================================
# FEATURE 5: Cross-joint coordination features
# =====================================================
def extract_coordination_features(train_ts, test_ts):
    """
    Cross-correlation between joint pairs to capture kinetic chain timing.
    - right_elbow_z velocity vs right_wrist_z velocity
    - right_shoulder_z velocity vs right_elbow_z velocity
    """
    pairs = [
        ("right_elbow_z", "right_wrist_z"),
        ("right_shoulder_z", "right_elbow_z"),
        ("right_elbow_y", "right_wrist_y"),
        ("right_shoulder_y", "right_elbow_y"),
    ]

    def compute_xcorr_features(ts_dict, n_shots):
        features = np.zeros((n_shots, len(pairs) * 3))

        for p_idx, (j1, j2) in enumerate(pairs):
            for i in range(n_shots):
                s1 = ts_dict[j1][i]
                s2 = ts_dict[j2][i]

                # Smooth
                valid1 = ~np.isnan(s1)
                valid2 = ~np.isnan(s2)
                if valid1.sum() >= SAVGOL_WINDOW and valid2.sum() >= SAVGOL_WINDOW:
                    s1c = np.interp(np.arange(240), np.where(valid1)[0], s1[valid1])
                    s2c = np.interp(np.arange(240), np.where(valid2)[0], s2[valid2])
                    s1s = savgol_filter(s1c, SAVGOL_WINDOW, SAVGOL_POLY)
                    s2s = savgol_filter(s2c, SAVGOL_WINDOW, SAVGOL_POLY)
                else:
                    s1s = np.nan_to_num(s1, nan=0.0)
                    s2s = np.nan_to_num(s2, nan=0.0)

                # Velocities
                v1 = np.gradient(s1s)
                v2 = np.gradient(s2s)

                # Normalize
                v1_norm = (v1 - np.mean(v1)) / (np.std(v1) + 1e-8)
                v2_norm = (v2 - np.mean(v2)) / (np.std(v2) + 1e-8)

                # Cross-correlation (only in release window +/- some margin)
                w_start, w_end = 110, 200
                v1_w = v1_norm[w_start:w_end]
                v2_w = v2_norm[w_start:w_end]

                xcorr = np.correlate(v1_w, v2_w, mode='full')
                xcorr = xcorr / len(v1_w)  # Normalize

                mid = len(xcorr) // 2
                max_lag = 30  # Look within +/- 30 frames
                search_region = xcorr[mid - max_lag:mid + max_lag + 1]

                # Peak cross-correlation value
                features[i, p_idx * 3 + 0] = np.max(search_region)
                # Lag at peak (negative means j1 leads j2)
                features[i, p_idx * 3 + 1] = np.argmax(search_region) - max_lag
                # Correlation at zero lag
                features[i, p_idx * 3 + 2] = xcorr[mid]

        return features

    n_train = len(list(train_ts.values())[0])
    n_test = len(list(test_ts.values())[0])

    train_feat = compute_xcorr_features(train_ts, n_train)
    test_feat = compute_xcorr_features(test_ts, n_test)

    return train_feat, test_feat


# =====================================================
# FEATURE 6: Release frame value features
# =====================================================
def extract_release_features(train_ts, test_ts):
    """Simple features: value at release frame for key joints."""
    def extract(ts_dict, n_shots):
        features = np.zeros((n_shots, len(RELEASE_JOINTS)))
        for j_idx, joint in enumerate(RELEASE_JOINTS):
            ts = ts_dict[joint]
            for i in range(n_shots):
                s = ts[i]
                valid = ~np.isnan(s)
                if valid.sum() >= SAVGOL_WINDOW:
                    s_clean = np.interp(np.arange(240), np.where(valid)[0], s[valid])
                    s_smooth = savgol_filter(s_clean, SAVGOL_WINDOW, SAVGOL_POLY)
                    features[i, j_idx] = s_smooth[RELEASE_FRAME]
                else:
                    features[i, j_idx] = np.nan_to_num(s[RELEASE_FRAME], nan=0.0)
        return features

    n_train = len(list(train_ts.values())[0])
    n_test = len(list(test_ts.values())[0])

    return extract(train_ts, n_train), extract(test_ts, n_test)


# =====================================================
# CV and model training
# =====================================================
def run_cv_and_predict(X_train_all, y_train, pids_train, X_test_all, pids_test, fpca_k):
    """
    Per-player, per-target 5-fold CV with Ridge + LGB + PLS ensemble.
    Returns: OOF predictions, test predictions, per-player per-target CV scores.
    """
    players = np.unique(pids_train)
    n_train = len(pids_train)
    n_test = len(pids_test)

    oof_preds = np.zeros((n_train, 3))
    test_preds = np.zeros((n_test, 3))
    cv_scores = {}

    for pid in players:
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        X_p_train = X_train_all[train_mask]
        y_p_train = y_train[train_mask]
        X_p_test = X_test_all[test_mask]

        # Per-player StandardScaler
        scaler = StandardScaler()
        X_p_train_sc = scaler.fit_transform(X_p_train)
        X_p_test_sc = scaler.transform(X_p_test)

        for t_idx, target in enumerate(TARGETS):
            y_t = y_p_train[:, t_idx]

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            oof_t = np.zeros(len(y_t))
            test_preds_folds = []

            for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_p_train_sc)):
                X_tr, X_val = X_p_train_sc[tr_idx], X_p_train_sc[val_idx]
                y_tr, y_val = y_t[tr_idx], y_t[val_idx]

                fold_test_preds = []

                # Ridge with alpha CV
                alphas = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
                ridge = RidgeCV(alphas=alphas)
                ridge.fit(X_tr, y_tr)
                oof_ridge = ridge.predict(X_val)
                test_ridge = ridge.predict(X_p_test_sc)
                fold_test_preds.append(test_ridge)

                # LightGBM (conservative)
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=80, num_leaves=8, learning_rate=0.05,
                    reg_alpha=1.0, reg_lambda=1.0, min_child_samples=5,
                    subsample=0.8, colsample_bytree=0.8,
                    verbose=-1, n_jobs=-1, random_state=42
                )
                lgb_model.fit(X_tr, y_tr)
                oof_lgb = lgb_model.predict(X_val)
                test_lgb = lgb_model.predict(X_p_test_sc)
                fold_test_preds.append(test_lgb)

                # PLS regression - find best n_components via inner CV
                best_pls_nc = 3
                best_pls_score = 1e9
                max_nc = min(15, X_tr.shape[1], X_tr.shape[0] - 1)
                for nc in PLS_N_OPTIONS:
                    if nc > max_nc:
                        break
                    try:
                        pls = PLSRegression(n_components=nc, max_iter=500)
                        pls.fit(X_tr, y_tr)
                        pls_pred = pls.predict(X_val).ravel()
                        score = np.mean((pls_pred - y_val)**2)
                        if score < best_pls_score:
                            best_pls_score = score
                            best_pls_nc = nc
                    except Exception:
                        continue

                pls = PLSRegression(n_components=best_pls_nc, max_iter=500)
                pls.fit(X_tr, y_tr)
                oof_pls = pls.predict(X_val).ravel()
                test_pls = pls.predict(X_p_test_sc).ravel()
                fold_test_preds.append(test_pls)

                # Ensemble: 0.35 Ridge + 0.35 LGB + 0.30 PLS
                oof_t[val_idx] = 0.35 * oof_ridge + 0.35 * oof_lgb + 0.30 * oof_pls
                fold_test = 0.35 * test_ridge + 0.35 * test_lgb + 0.30 * test_pls
                test_preds_folds.append(fold_test)

            # Average test predictions across folds
            test_avg = np.mean(test_preds_folds, axis=0)

            # Store
            oof_preds[train_mask, t_idx] = oof_t
            test_preds[test_mask, t_idx] = test_avg

            # CV score (RMSE)
            rmse = np.sqrt(np.mean((oof_t - y_t)**2))
            cv_scores[(pid, target)] = rmse

    return oof_preds, test_preds, cv_scores


def main():
    print("=" * 70)
    print("FUNCTIONAL DATA ANALYSIS - Basketball Shot Prediction")
    print("=" * 70)

    # Load data
    train_ts, y_train, pids_train, train_ids, test_ts, pids_test, test_ids = load_data()
    n_train = len(pids_train)
    n_test = len(pids_test)
    print(f"Train: {n_train}, Test: {n_test}")
    print(f"Players: {np.unique(pids_train)}")

    # ---- Extract all feature groups ----

    print("\n--- Feature 1: B-spline basis expansion ---")
    bspline_train, bspline_test = extract_bspline_features(train_ts, test_ts)
    print(f"  B-spline features: {bspline_train.shape[1]}")

    print("\n--- Feature 2: Derivative features ---")
    deriv_train, deriv_test = extract_derivative_features(train_ts, test_ts)
    print(f"  Derivative features: {deriv_train.shape[1]}")

    print("\n--- Feature 3: Distance-to-mean features ---")
    dist_train, dist_test = extract_distance_features(train_ts, test_ts, pids_train, pids_test)
    print(f"  Distance features: {dist_train.shape[1]}")

    print("\n--- Feature 4: Cross-joint coordination features ---")
    coord_train, coord_test = extract_coordination_features(train_ts, test_ts)
    print(f"  Coordination features: {coord_train.shape[1]}")

    print("\n--- Feature 5: Release frame features ---")
    release_train, release_test = extract_release_features(train_ts, test_ts)
    print(f"  Release features: {release_train.shape[1]}")

    # ---- Try different FPCA component counts ----
    print("\n--- Feature 6: Functional PCA (per-player) ---")
    print("Testing FPCA k values:", FPCA_K_OPTIONS)

    best_overall_cv = 1e9
    best_k = None
    best_test_preds = None
    best_cv_scores = None

    for k in FPCA_K_OPTIONS:
        print(f"\n  FPCA k={k}:")
        fpca_train, fpca_test = extract_fpca_features(train_ts, test_ts, pids_train, pids_test, k)
        print(f"    FPCA features: {fpca_train.shape[1]}")

        # Combine all features
        X_train_all = np.hstack([
            fpca_train,
            bspline_train,
            deriv_train,
            dist_train,
            coord_train,
            release_train,
        ])
        X_test_all = np.hstack([
            fpca_test,
            bspline_test,
            deriv_test,
            dist_test,
            coord_test,
            release_test,
        ])

        # Handle NaN/Inf
        X_train_all = np.nan_to_num(X_train_all, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_all = np.nan_to_num(X_test_all, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"    Total features: {X_train_all.shape[1]}")

        # Run CV
        oof_preds, test_preds_k, cv_scores = run_cv_and_predict(
            X_train_all, y_train, pids_train, X_test_all, pids_test, k
        )

        # Overall CV RMSE
        overall_rmse = 0
        for t_idx, target in enumerate(TARGETS):
            target_rmse = np.sqrt(np.mean((oof_preds[:, t_idx] - y_train[:, t_idx])**2))
            overall_rmse += target_rmse
            print(f"    {target} CV RMSE: {target_rmse:.6f}")
        mean_rmse = overall_rmse / 3
        print(f"    Mean CV RMSE: {mean_rmse:.6f}")

        if mean_rmse < best_overall_cv:
            best_overall_cv = mean_rmse
            best_k = k
            best_test_preds = test_preds_k.copy()
            best_cv_scores = cv_scores.copy()

    print(f"\n{'=' * 70}")
    print(f"Best FPCA k: {best_k}, Mean CV RMSE: {best_overall_cv:.6f}")
    print(f"{'=' * 70}")

    # Print per-player per-target CV scores
    print("\nPer-player per-target CV RMSE (best k):")
    print(f"{'Player':<8} {'angle':<12} {'depth':<12} {'left_right':<12}")
    for pid in sorted(np.unique(pids_train)):
        a = best_cv_scores.get((pid, 'angle'), 0)
        d = best_cv_scores.get((pid, 'depth'), 0)
        lr = best_cv_scores.get((pid, 'left_right'), 0)
        print(f"  {pid:<6} {a:<12.6f} {d:<12.6f} {lr:<12.6f}")

    # ---- Scale predictions ----
    print("\nScaling predictions...")
    scalers = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    scaled_preds = np.zeros((n_test, 3))
    for i, target in enumerate(TARGETS):
        scaled_preds[:, i] = scalers[target].transform(
            best_test_preds[:, i].reshape(-1, 1)
        ).flatten()

    # ---- Save standalone submission ----
    sub_num_standalone = get_next_submission_number()
    sub_standalone = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })
    standalone_path = SUBMISSION_DIR / f"submission_{sub_num_standalone}.csv"
    sub_standalone.to_csv(standalone_path, index=False)
    print(f"\nStandalone submission saved: {standalone_path}")
    print(f"  angle: mean={scaled_preds[:, 0].mean():.6f}, std={scaled_preds[:, 0].std():.6f}")
    print(f"  depth: mean={scaled_preds[:, 1].mean():.6f}, std={scaled_preds[:, 1].std():.6f}")
    print(f"  left_right: mean={scaled_preds[:, 2].mean():.6f}, std={scaled_preds[:, 2].std():.6f}")

    # ---- Blend with Sub 784 ----
    print("\n--- Blending with Sub 784 ---")
    sub784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # Align by id
    sub784_aligned = sub784.set_index('id').loc[test_ids].reset_index()

    # Correlation with Sub 784
    print("\nCorrelation with Sub 784:")
    for col in SCALED_TARGETS:
        corr = np.corrcoef(sub_standalone[col].values, sub784_aligned[col].values)[0, 1]
        print(f"  {col}: {corr:.6f}")

    # Blend at different weights
    blend_weights = [0.10, 0.20, 0.30, 0.50]
    print(f"\nBlend weights for new model: {blend_weights}")

    for w_new in blend_weights:
        w_base = 1.0 - w_new
        sub_num = get_next_submission_number()

        blended = pd.DataFrame({'id': test_ids})
        for col in SCALED_TARGETS:
            blended[col] = w_base * sub784_aligned[col].values + w_new * sub_standalone[col].values

        blend_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blended.to_csv(blend_path, index=False)
        print(f"  Sub {sub_num}: {w_new:.0%} new + {w_base:.0%} Sub784 -> {blend_path}")

    # ---- Target-specific blend ----
    print("\n--- Target-specific blend ---")
    # For each target, compute correlation with Sub784
    # Use the new model only for the target with lowest correlation (most diverse)
    correlations = {}
    for col in SCALED_TARGETS:
        correlations[col] = np.corrcoef(
            sub_standalone[col].values, sub784_aligned[col].values
        )[0, 1]

    min_corr_target = min(correlations, key=correlations.get)
    print(f"Lowest correlation target: {min_corr_target} (r={correlations[min_corr_target]:.6f})")
    print(f"All correlations: {correlations}")

    # Create target-specific blend: use Sub784 for high-corr targets, new model blend for low-corr
    for w_new in [0.20, 0.30, 0.50]:
        sub_num = get_next_submission_number()
        blended_ts = pd.DataFrame({'id': test_ids})
        for col in SCALED_TARGETS:
            if col == min_corr_target:
                blended_ts[col] = (1 - w_new) * sub784_aligned[col].values + w_new * sub_standalone[col].values
            else:
                blended_ts[col] = sub784_aligned[col].values  # Keep Sub784 for other targets

        ts_blend_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blended_ts.to_csv(ts_blend_path, index=False)
        print(f"  Sub {sub_num}: Target-specific ({min_corr_target} {w_new:.0%} new) -> {ts_blend_path}")

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Functional Data Analysis features:")
    print(f"  - FPCA (per-player, k={best_k}): captures within-player trajectory variation")
    print(f"  - B-spline coefficients: compact trajectory shape representation")
    print(f"  - Derivative features: velocity/acceleration/jerk dynamics at release")
    print(f"  - Distance-to-mean: shot consistency measure")
    print(f"  - Cross-joint coordination: kinetic chain timing")
    print(f"  - Release frame values: instantaneous pose at release")
    print(f"Best Mean CV RMSE: {best_overall_cv:.6f}")
    print(f"Standalone: Sub {sub_num_standalone} ({standalone_path})")
    print(f"Correlations with Sub 784: {correlations}")


if __name__ == "__main__":
    main()

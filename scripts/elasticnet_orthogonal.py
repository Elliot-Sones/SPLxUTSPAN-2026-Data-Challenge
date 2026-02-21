"""
ElasticNet Orthogonal Model

Tests Gemini's ElasticNet approach (7500+ statistical features) as an
orthogonal model for blending with our locally-weighted Ridge pipeline.

Key value proposition: DIVERSITY. Different model class + different features
= different errors = potential blend improvement even if standalone is weaker.

Based on Gemini's super_model_submission.py with fixes:
- Fixed KC feature indexing bug (column index vs keypoint index)
- LOO evaluation for fair comparison with our pipeline
- Correlation analysis against Sub 2475 predictions
- Blend submission generation

Features: ~7500 per sample
- 69 keypoints x 3 coords x (5 angle frames + 5 depth frames + 4 LR frames
  + 2 velocities per angle frame + window mean/std + phase mean/std/range/vel_max)
- 40 kinetic chain features
- 5 player one-hot

Model: Per-player ElasticNet with L1/L2 regularization
"""

import json
import time
import fcntl
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGET_NAMES = ["angle", "depth", "left_right"]
DT = 1.0 / 60.0

import joblib
TARGET_SCALERS = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGET_NAMES}


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


def smooth_signal(x, window=5):
    if len(x) < window:
        return x
    return uniform_filter1d(x.astype(np.float64), size=window, mode='nearest')


def compute_velocity(pos, dt=DT):
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel


def safe_savgol(x, window=7, polyorder=2):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder)


# ---------------------------------------------------------------------------
# ElasticNet feature extraction (~7500 features)
# ---------------------------------------------------------------------------
def extract_elasticnet_features(ts, participant_id, kp_names, n_kp):
    """Extract statistical features from time series at key frames/windows/phases."""
    features = {}

    # Participant one-hot
    for i in range(1, 6):
        features[f"participant_{i}"] = 1.0 if participant_id == i else 0.0

    # Key frames per target
    angle_frames = [148, 150, 153, 155, 158]
    angle_window = (145, 165)
    depth_frames = [100, 102, 105, 108, 110]
    depth_window = (95, 115)
    lr_frames = [220, 225, 230, 235]
    lr_window = (215, 240)
    phases = [("setup", 0, 60), ("load", 60, 120), ("release", 120, 180), ("follow", 180, 240)]

    for kp_i in range(n_kp):
        kp_name = kp_names[kp_i]
        for c, coord in enumerate(["x", "y", "z"]):
            pos = ts[:, kp_i, c]
            pos_smooth = smooth_signal(pos)
            vel = compute_velocity(pos_smooth)

            # Angle target features
            for f in angle_frames:
                features[f"angle_f{f}_{kp_name}_{coord}"] = pos_smooth[f]
                features[f"angle_f{f}_{kp_name}_v{coord}"] = vel[f]
            w = pos_smooth[angle_window[0]:angle_window[1]]
            features[f"angle_window_{kp_name}_{coord}_mean"] = np.nanmean(w)
            features[f"angle_window_{kp_name}_{coord}_std"] = np.nanstd(w)

            # Depth target features
            for f in depth_frames:
                features[f"depth_f{f}_{kp_name}_{coord}"] = pos_smooth[f]
            w = pos_smooth[depth_window[0]:depth_window[1]]
            features[f"depth_window_{kp_name}_{coord}_mean"] = np.nanmean(w)

            # LR target features
            for f in lr_frames:
                features[f"lr_f{f}_{kp_name}_{coord}"] = pos_smooth[f]

            # Phase features
            for phase_name, start, end in phases:
                phase_pos = pos_smooth[start:end]
                phase_vel = vel[start:end]
                features[f"phase_{phase_name}_{kp_name}_{coord}_mean"] = np.nanmean(phase_pos)
                features[f"phase_{phase_name}_{kp_name}_{coord}_std"] = np.nanstd(phase_pos)
                features[f"phase_{phase_name}_{kp_name}_{coord}_range"] = np.nanmax(phase_pos) - np.nanmin(phase_pos)
                features[f"phase_{phase_name}_{kp_name}_vel_max_{coord}"] = np.nanmax(np.abs(phase_vel))

    return features


# ---------------------------------------------------------------------------
# Kinetic chain features (fixed indexing)
# ---------------------------------------------------------------------------
def extract_kc_features(ts_3d, kp_name_to_idx):
    """Extract kinetic chain features from (240, n_kp, 3) array.

    Fixed version: uses proper keypoint index mapping instead of column index.
    """
    features = {}

    # Get joint positions
    joint_kps = {
        "ankle": "right_ankle",
        "knee": "right_knee",
        "hip": "right_hip",
        "shoulder": "right_shoulder",
        "elbow": "right_elbow",
        "wrist": "right_wrist",
    }

    joints = {}
    for name, kp_name in joint_kps.items():
        if kp_name not in kp_name_to_idx:
            return features
        idx = kp_name_to_idx[kp_name]
        data = ts_3d[:, idx, :]  # (240, 3)
        # Smooth each coordinate
        smoothed = np.column_stack([safe_savgol(data[:, c]) for c in range(3)])
        joints[name] = smoothed

    # Velocities
    velocities = {}
    vert_velocities = {}
    for k, v in joints.items():
        vel = np.gradient(v, DT, axis=0)
        velocities[k] = np.linalg.norm(vel, axis=1)
        vert_velocities[k] = vel[:, 2]

    # Release frame (peak wrist velocity in 2nd half)
    search_start = 120
    release_frame = search_start + np.argmax(velocities["wrist"][search_start:])
    features["kc_release_frame"] = release_frame

    # Propulsion phase
    prop_start = max(0, release_frame - 60)
    prop_end = min(240, release_frame + 5)

    # Peak analysis
    chain_order = ["knee", "hip", "shoulder", "elbow", "wrist"]
    peak_times = {}
    peak_mags = {}

    def compute_joint_angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        dot = np.sum(v1 * v2, axis=1)
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        denom = n1 * n2
        denom[denom == 0] = 1e-10
        return np.arccos(np.clip(dot / denom, -1, 1)) * 180 / np.pi

    for joint in chain_order:
        window_vel = velocities[joint][prop_start:prop_end]
        if len(window_vel) == 0:
            continue

        if joint == "knee":
            angle = compute_joint_angle(joints["hip"], joints["knee"], joints["ankle"])
            ext_rate = np.gradient(angle, DT)
            window_ext = ext_rate[prop_start:prop_end]
            if len(window_ext) > 0:
                idx_max = np.argmax(window_ext)
                peak_val = window_ext[idx_max]
                peak_frame = prop_start + idx_max
            else:
                continue
        elif joint == "hip":
            window_vert = vert_velocities["hip"][prop_start:prop_end]
            if len(window_vert) > 0:
                idx_max = np.argmax(window_vert)
                peak_val = window_vert[idx_max]
                peak_frame = prop_start + idx_max
            else:
                continue
        else:
            idx_max = np.argmax(window_vel)
            peak_val = window_vel[idx_max]
            peak_frame = prop_start + idx_max

        peak_times[joint] = peak_frame
        peak_mags[joint] = peak_val
        features[f"kc_{joint}_peak_frame"] = peak_frame
        features[f"kc_{joint}_peak_mag"] = peak_val
        features[f"kc_{joint}_time_to_release"] = (peak_frame - release_frame) * DT

    # Sequencing
    pairs = [("knee", "hip"), ("hip", "shoulder"), ("shoulder", "elbow"), ("elbow", "wrist")]
    for prev, curr in pairs:
        if prev in peak_times and curr in peak_times:
            delta = peak_times[curr] - peak_times[prev]
            features[f"kc_seq_{prev}_to_{curr}_frames"] = delta
            features[f"kc_seq_{prev}_to_{curr}_time"] = delta * DT
            if abs(peak_mags[prev]) > 1e-6:
                features[f"kc_amp_{prev}_to_{curr}_ratio"] = peak_mags[curr] / peak_mags[prev]

    # Integrated energy
    for joint in chain_order:
        if joint in velocities:
            energy = np.sum(velocities[joint][prop_start:release_frame] ** 2)
            features[f"kc_{joint}_energy_propulsion"] = energy

    # Release mechanics
    v_rel = np.gradient(joints["wrist"], DT, axis=0)[release_frame]
    features["kc_wrist_vx_release"] = v_rel[0]
    features["kc_wrist_vy_release"] = v_rel[1]
    features["kc_wrist_vz_release"] = v_rel[2]
    features["kc_wrist_vmag_release"] = np.linalg.norm(v_rel)

    v_horiz = np.sqrt(v_rel[0] ** 2 + v_rel[1] ** 2)
    features["kc_release_angle_elev"] = np.arctan2(v_rel[2], v_horiz) * 180 / np.pi
    features["kc_release_angle_azi"] = np.arctan2(v_rel[0], v_rel[1]) * 180 / np.pi

    # Jerk
    acc = np.gradient(np.gradient(joints["wrist"], DT, axis=0), DT, axis=0)
    jerk = np.gradient(acc, DT, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    features["kc_wrist_jerk_release"] = jerk_mag[release_frame]
    features["kc_wrist_jerk_mean"] = np.mean(jerk_mag[prop_start:release_frame])

    return features


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right",
                 "scaled_angle", "scaled_depth", "scaled_left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    # Build keypoint name -> index mapping
    kp_names = []
    kp_name_to_idx = {}
    for col in keypoint_cols:
        parts = col.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("x", "y", "z"):
            kp_name = parts[0]
            if kp_name not in kp_name_to_idx:
                kp_name_to_idx[kp_name] = len(kp_names)
                kp_names.append(kp_name)

    n_kp = len(kp_names)
    print(f"  {n_kp} keypoints, {len(keypoint_cols)} columns")

    def process_df(df, is_train):
        n = len(df)
        all_features = []
        pids = np.zeros(n, dtype=int)
        ids = []

        for i, (_, row) in enumerate(df.iterrows()):
            # Parse time series into (240, n_kp, 3)
            ts_3d = np.zeros((240, n_kp, 3), dtype=np.float32)
            for j, col in enumerate(keypoint_cols):
                kp_i = j // 3
                c = j % 3
                ts_3d[:, kp_i, c] = parse_array_string(row[col])

            # Also build (240, n_cols) for compatibility
            ts_flat = ts_3d.reshape(240, -1)

            pid = row["participant_id"]
            pids[i] = pid
            ids.append(row["id"])

            # Extract features
            enet_feats = extract_elasticnet_features(ts_3d, pid, kp_names, n_kp)
            kc_feats = extract_kc_features(ts_3d, kp_name_to_idx)

            combined = {**enet_feats, **kc_feats}
            all_features.append(combined)

        # Build feature matrix
        X_df = pd.DataFrame(all_features).fillna(0)

        # Get targets if train (scale raw -> [0,1])
        y = None
        if is_train:
            y = np.column_stack([
                TARGET_SCALERS[t].transform(df[t].values.reshape(-1, 1)).flatten()
                for t in TARGET_NAMES
            ])

        return X_df, y, pids, ids

    print("Processing train...")
    X_train_df, y_train, train_pids, train_ids = process_df(train_df, True)
    print(f"  {X_train_df.shape[1]} features extracted")

    print("Processing test...")
    X_test_df, _, test_pids, test_ids = process_df(test_df, False)

    # Align columns
    all_cols = sorted(set(X_train_df.columns) | set(X_test_df.columns))
    X_train_df = X_train_df.reindex(columns=all_cols, fill_value=0)
    X_test_df = X_test_df.reindex(columns=all_cols, fill_value=0)

    # Replace any remaining NaN/inf
    X_train_df = X_train_df.replace([np.inf, -np.inf], 0).fillna(0)
    X_test_df = X_test_df.replace([np.inf, -np.inf], 0).fillna(0)

    print(f"  Final: {X_train_df.shape[1]} aligned features")

    return {
        "X_train": X_train_df.values.astype(np.float64),
        "y_train": y_train,
        "train_pids": train_pids,
        "train_ids": train_ids,
        "X_test": X_test_df.values.astype(np.float64),
        "test_pids": test_pids,
        "test_ids": test_ids,
        "feature_names": all_cols,
    }


# ---------------------------------------------------------------------------
# LOO evaluation
# ---------------------------------------------------------------------------
def loo_evaluate(data, alpha=1.0, l1_ratio=0.5):
    """Leave-one-out evaluation with per-player ElasticNet models."""
    X = data["X_train"]
    y = data["y_train"]
    pids = data["train_pids"]
    unique_pids = sorted(np.unique(pids))
    n = len(X)

    loo_preds = np.zeros((n, 3))

    print(f"\nLOO evaluation (alpha={alpha}, l1_ratio={l1_ratio}):")

    for pid in unique_pids:
        pid_mask = pids == pid
        idx = np.where(pid_mask)[0]
        X_player = X[idx]
        y_player = y[idx]
        n_player = len(idx)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)

        for target_idx, target_name in enumerate(TARGET_NAMES):
            y_target = y_player[:, target_idx]

            # LOO for this player-target
            for li in range(n_player):
                train_mask = np.ones(n_player, dtype=bool)
                train_mask[li] = False

                X_tr = X_scaled[train_mask]
                y_tr = y_target[train_mask]
                X_te = X_scaled[li:li+1]

                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)
                model.fit(X_tr, y_tr)
                loo_preds[idx[li], target_idx] = model.predict(X_te)[0]

        # Per-player MSE
        player_mse = np.mean((loo_preds[idx] - y[idx]) ** 2, axis=0)
        print(f"  P{pid} ({n_player}): angle={player_mse[0]:.6f}  depth={player_mse[1]:.6f}  LR={player_mse[2]:.6f}")

    # Overall MSE
    mse_per_target = np.mean((loo_preds - y) ** 2, axis=0)
    mean_mse = np.mean(mse_per_target)
    print(f"\n  LOO MSE: angle={mse_per_target[0]:.6f}  depth={mse_per_target[1]:.6f}  LR={mse_per_target[2]:.6f}")
    print(f"  Mean LOO: {mean_mse:.6f}")

    return loo_preds, mse_per_target, mean_mse


def quick_cv_evaluate(data, alpha=1.0, l1_ratio=0.5, n_splits=5):
    """Quick 5-fold CV (faster than LOO for sweeps)."""
    X = data["X_train"]
    y = data["y_train"]
    pids = data["train_pids"]
    unique_pids = sorted(np.unique(pids))
    n = len(X)

    cv_preds = np.zeros((n, 3))

    for pid in unique_pids:
        pid_mask = pids == pid
        idx = np.where(pid_mask)[0]
        X_player = X[idx]
        y_player = y[idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for target_idx in range(3):
            y_target = y_player[:, target_idx]
            for train_idx, val_idx in kf.split(X_scaled):
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)
                model.fit(X_scaled[train_idx], y_target[train_idx])
                cv_preds[idx[val_idx], target_idx] = model.predict(X_scaled[val_idx])

    mse_per_target = np.mean((cv_preds - y) ** 2, axis=0)
    mean_mse = np.mean(mse_per_target)
    return cv_preds, mse_per_target, mean_mse


# ---------------------------------------------------------------------------
# Generate test predictions
# ---------------------------------------------------------------------------
def generate_test_predictions(data, alpha=1.0, l1_ratio=0.5):
    """Train on all data, predict test."""
    X = data["X_train"]
    y = data["y_train"]
    pids = data["train_pids"]
    X_test = data["X_test"]
    test_pids = data["test_pids"]
    unique_pids = sorted(np.unique(pids))

    test_preds = np.zeros((len(X_test), 3))

    for pid in unique_pids:
        train_mask = pids == pid
        test_mask = test_pids == pid

        X_train_p = X[train_mask]
        X_test_p = X_test[test_mask]
        y_train_p = y[train_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_p)
        X_test_scaled = scaler.transform(X_test_p)

        for target_idx in range(3):
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)
            model.fit(X_train_scaled, y_train_p[:, target_idx])
            test_preds[test_mask, target_idx] = model.predict(X_test_scaled)

    return test_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    data = load_data()

    # ====================================================================
    # Phase 1: Quick alpha/l1_ratio sweep with 5-fold CV
    # ====================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Quick hyperparameter sweep (5-fold CV)")
    print("=" * 70)

    configs = [
        (0.01, 0.5),
        (0.05, 0.5),
        (0.1, 0.5),
        (0.5, 0.5),
        (1.0, 0.5),
        (2.0, 0.5),
        (0.1, 0.1),   # more L2
        (0.1, 0.9),   # more L1
        (0.5, 0.1),
        (0.5, 0.9),
        (0.01, 0.1),
        (0.01, 0.9),
    ]

    results = []
    for alpha, l1 in configs:
        _, mse, mean_mse = quick_cv_evaluate(data, alpha=alpha, l1_ratio=l1)
        results.append((alpha, l1, mean_mse, mse))
        print(f"  alpha={alpha:.2f}, l1={l1:.1f}: mean={mean_mse:.6f} "
              f"(a={mse[0]:.6f}, d={mse[1]:.6f}, lr={mse[2]:.6f})")

    # Sort by mean MSE
    results.sort(key=lambda x: x[2])
    best_alpha, best_l1, best_cv, _ = results[0]
    print(f"\nBest: alpha={best_alpha}, l1_ratio={best_l1}, CV={best_cv:.6f}")

    # ====================================================================
    # Phase 2: LOO with best config
    # ====================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 2: LOO evaluation (alpha={best_alpha}, l1={best_l1})")
    print("=" * 70)

    loo_preds, loo_mse, loo_mean = loo_evaluate(data, alpha=best_alpha, l1_ratio=best_l1)

    # ====================================================================
    # Phase 3: Diversity analysis
    # ====================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Diversity analysis vs Sub 2475")
    print("=" * 70)

    # Load Sub 2475 (our best)
    sub_2475 = pd.read_csv(SUBMISSION_DIR / "submission_2475.csv")
    # We need LOO preds from our pipeline for correlation - use train LOO
    # Compare test predictions instead since that's what matters for blending

    # Generate test predictions
    test_preds = generate_test_predictions(data, alpha=best_alpha, l1_ratio=best_l1)

    # Correlation with Sub 2475 on test set
    for i, target in enumerate(TARGET_NAMES):
        sub_vals = sub_2475[f"scaled_{target}"].values
        enet_vals = test_preds[:, i]
        corr = np.corrcoef(sub_vals, enet_vals)[0, 1]
        print(f"  {target}: correlation with Sub 2475 = {corr:.4f}")

    # Also check prediction range
    print("\n  Prediction ranges (ElasticNet):")
    for i, target in enumerate(TARGET_NAMES):
        vals = test_preds[:, i]
        print(f"  {target}: [{vals.min():.4f}, {vals.max():.4f}] mean={vals.mean():.4f}")

    print("\n  Prediction ranges (Sub 2475):")
    for target in TARGET_NAMES:
        vals = sub_2475[f"scaled_{target}"].values
        print(f"  {target}: [{vals.min():.4f}, {vals.max():.4f}] mean={vals.mean():.4f}")

    # ====================================================================
    # Phase 4: Clipped predictions and per-target sweep
    # ====================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Clipping and per-target analysis")
    print("=" * 70)

    # Clip predictions to training target range
    y = data["y_train"]
    for i in range(3):
        lo, hi = y[:, i].min(), y[:, i].max()
        print(f"  {TARGET_NAMES[i]}: clip [{lo:.4f}, {hi:.4f}]")
        test_preds[:, i] = np.clip(test_preds[:, i], lo, hi)
        loo_preds[:, i] = np.clip(loo_preds[:, i], lo, hi)

    # Re-check correlation after clipping
    print("\n  Correlations after clipping:")
    for i, target in enumerate(TARGET_NAMES):
        sub_vals = sub_2475[f"scaled_{target}"].values
        enet_vals = test_preds[:, i]
        corr = np.corrcoef(sub_vals, enet_vals)[0, 1]
        print(f"  {target}: r={corr:.4f}")

    # Check LOO after clipping
    mse_clipped = np.mean((loo_preds - y) ** 2, axis=0)
    print(f"\n  LOO after clip: angle={mse_clipped[0]:.6f}  depth={mse_clipped[1]:.6f}  LR={mse_clipped[2]:.6f}")
    print(f"  Mean: {np.mean(mse_clipped):.6f}")

    # ====================================================================
    # Phase 5: Per-target optimal alpha sweep
    # ====================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Per-target optimal alpha (LOO)")
    print("=" * 70)

    from sklearn.linear_model import Ridge as SkRidge

    X_all = data["X_train"]
    y_all = data["y_train"]
    pids_all = data["train_pids"]

    best_per_target = {}
    for target_idx, target_name in enumerate(TARGET_NAMES):
        print(f"\n  {target_name}:")
        best_mse = 999
        best_cfg = None
        for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
            for model_type in ["elasticnet", "ridge"]:
                # Quick LOO per target
                preds_t = np.zeros(len(X_all))
                for pid in sorted(np.unique(pids_all)):
                    pid_mask = pids_all == pid
                    idx_p = np.where(pid_mask)[0]
                    X_p = X_all[idx_p]
                    y_p = y_all[idx_p, target_idx]

                    scaler_p = StandardScaler()
                    X_ps = scaler_p.fit_transform(X_p)

                    for li in range(len(idx_p)):
                        mask = np.ones(len(idx_p), dtype=bool)
                        mask[li] = False
                        if model_type == "elasticnet":
                            m = ElasticNet(alpha=alpha, l1_ratio=0.1, max_iter=10000, random_state=42)
                        else:
                            m = SkRidge(alpha=alpha)
                        m.fit(X_ps[mask], y_p[mask])
                        preds_t[idx_p[li]] = m.predict(X_ps[li:li+1])[0]

                mse_t = np.mean((preds_t - y_all[:, target_idx]) ** 2)
                if mse_t < best_mse:
                    best_mse = mse_t
                    best_cfg = (model_type, alpha)
                    best_preds = preds_t.copy()

        print(f"    Best: {best_cfg[0]} alpha={best_cfg[1]}, LOO={best_mse:.6f}")
        best_per_target[target_name] = {
            "model_type": best_cfg[0], "alpha": best_cfg[1],
            "loo_mse": best_mse, "loo_preds": best_preds
        }

    # Generate per-target optimal test predictions
    print("\n  Generating per-target optimal test predictions...")
    test_preds_opt = np.zeros((len(data["X_test"]), 3))
    X_test_all = data["X_test"]
    test_pids_all = data["test_pids"]

    for target_idx, target_name in enumerate(TARGET_NAMES):
        cfg = best_per_target[target_name]
        for pid in sorted(np.unique(pids_all)):
            train_mask = pids_all == pid
            test_mask = test_pids_all == pid

            scaler_p = StandardScaler()
            X_train_s = scaler_p.fit_transform(X_all[train_mask])
            X_test_s = scaler_p.transform(X_test_all[test_mask])

            if cfg["model_type"] == "elasticnet":
                m = ElasticNet(alpha=cfg["alpha"], l1_ratio=0.1, max_iter=10000, random_state=42)
            else:
                m = SkRidge(alpha=cfg["alpha"])
            m.fit(X_train_s, y_all[train_mask, target_idx])
            test_preds_opt[test_mask, target_idx] = m.predict(X_test_s)

    # Clip
    for i in range(3):
        lo, hi = y_all[:, i].min(), y_all[:, i].max()
        test_preds_opt[:, i] = np.clip(test_preds_opt[:, i], lo, hi)

    # Diversity check for per-target optimal
    print("\n  Per-target optimal diversity vs Sub 2475:")
    for i, target in enumerate(TARGET_NAMES):
        sub_vals = sub_2475[f"scaled_{target}"].values
        corr = np.corrcoef(sub_vals, test_preds_opt[:, i])[0, 1]
        print(f"  {target}: r={corr:.4f}")

    # ====================================================================
    # Phase 6: Generate submissions
    # ====================================================================
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    anchor = sub_2475[["scaled_angle", "scaled_depth", "scaled_left_right"]].values

    # Clipped ElasticNet blends
    for w in [0.05, 0.10, 0.20]:
        blended = w * test_preds + (1 - w) * anchor
        sub_num = get_next_submission_number()
        blend_df = pd.DataFrame({
            "id": data["test_ids"],
            "scaled_angle": blended[:, 0],
            "scaled_depth": blended[:, 1],
            "scaled_left_right": blended[:, 2],
        })
        blend_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blend_df.to_csv(blend_path, index=False)
        print(f"Sub {sub_num}: {int(w*100)}% clipped ElasticNet + {int((1-w)*100)}% Sub 2475")

    # Per-target optimal blends
    for w in [0.05, 0.10, 0.20]:
        blended = w * test_preds_opt + (1 - w) * anchor
        sub_num = get_next_submission_number()
        blend_df = pd.DataFrame({
            "id": data["test_ids"],
            "scaled_angle": blended[:, 0],
            "scaled_depth": blended[:, 1],
            "scaled_left_right": blended[:, 2],
        })
        blend_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blend_df.to_csv(blend_path, index=False)
        print(f"Sub {sub_num}: {int(w*100)}% per-target optimal + {int((1-w)*100)}% Sub 2475")

    # Per-target selective blend: only blend targets where ElasticNet adds diversity
    # Use ElasticNet for angle/LR (r<0.4), use Sub 2475 for depth (r=0.87)
    for w in [0.05, 0.10]:
        blended = anchor.copy()
        blended[:, 0] = w * test_preds_opt[:, 0] + (1 - w) * anchor[:, 0]  # angle
        blended[:, 2] = w * test_preds_opt[:, 2] + (1 - w) * anchor[:, 2]  # LR
        # depth stays 100% Sub 2475
        sub_num = get_next_submission_number()
        blend_df = pd.DataFrame({
            "id": data["test_ids"],
            "scaled_angle": blended[:, 0],
            "scaled_depth": blended[:, 1],
            "scaled_left_right": blended[:, 2],
        })
        blend_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blend_df.to_csv(blend_path, index=False)
        print(f"Sub {sub_num}: {int(w*100)}% angle+LR only (per-target opt) + Sub 2475")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

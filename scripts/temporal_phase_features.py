"""
Temporal Phase Features - Multi-resolution temporal statistics for shooting motion.

KEY INSIGHT: Current pipeline uses ONE frame per target. But the shooting motion has
distinct phases (preparation, loading, release, follow-through) and different temporal
scales carry different information:
- Position at release frame: WHERE the body is
- Velocity profile: HOW FAST the motion happens
- Acceleration peaks: the EXPLOSIVENESS of the shot
- Phase transitions: WHEN the motion shifts

This script extracts features at multiple time scales and phases, then uses PLS
to compress them into useful predictors for the per-player Ridge pipeline.

Features extracted:
1. Position statistics per phase (mean, std, range, skew) for key joints
2. Velocity statistics per phase
3. Peak velocity timing and magnitude
4. Inter-phase transitions (change in statistics between phases)
5. Temporal symmetry features (left vs right hand timing)
"""

import json
import fcntl
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP = np.array([5.25, -25, 10])

# Key joints for biomechanics
KEY_JOINTS = [
    'right_wrist', 'right_elbow', 'right_shoulder',
    'left_wrist', 'left_elbow', 'left_shoulder',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'mid_hip', 'nose', 'neck'
]

# Shooting phases (at 60fps, 240 frames total = 4 seconds)
PHASES = {
    'preparation': (0, 90),      # 0-1.5s: getting set
    'loading': (90, 140),        # 1.5-2.3s: dipping/loading
    'acceleration': (140, 160),  # 2.3-2.67s: upward acceleration
    'release': (155, 175),       # 2.58-2.92s: ball release window
    'follow_through': (175, 240) # 2.92-4s: follow through
}


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


def parse_array(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    return np.array(json.loads(s.replace("nan", "null")), dtype=np.float32)


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {'id', 'shot_id', 'participant_id', 'angle', 'depth', 'left_right'}
    kp_cols = [c for c in train.columns if c not in meta_cols]

    # Find available key joints
    available_joints = []
    for j in KEY_JOINTS:
        if f"{j}_x" in kp_cols:
            available_joints.append(j)

    print(f"  Available key joints: {len(available_joints)}")

    # Parse all keypoint arrays for key joints
    def extract_trajectories(df):
        n = len(df)
        n_joints = len(available_joints)
        # Full 240-frame trajectories: (n_samples, 240, n_joints*3)
        trajs = np.zeros((n, 240, n_joints * 3), dtype=np.float32)

        for i in range(n):
            for j_idx, joint in enumerate(available_joints):
                for c_idx, coord in enumerate(['x', 'y', 'z']):
                    col = f"{joint}_{coord}"
                    vals = parse_array(df.iloc[i][col])
                    trajs[i, :, j_idx * 3 + c_idx] = vals

            # NaN interpolation per feature
            for f in range(trajs.shape[2]):
                ts = trajs[i, :, f]
                nans = np.isnan(ts)
                if nans.any() and not nans.all():
                    valid = np.where(~nans)[0]
                    trajs[i, nans, f] = np.interp(
                        np.where(nans)[0], valid, ts[valid])
                elif nans.all():
                    trajs[i, :, f] = 0.0

        return trajs, available_joints

    print("  Parsing train trajectories...")
    train_trajs, joints = extract_trajectories(train)
    print("  Parsing test trajectories...")
    test_trajs, _ = extract_trajectories(test)

    # Get targets and player IDs
    train_targets = {}
    for t in TARGETS:
        col = f"scaled_{t}" if f"scaled_{t}" in train.columns else t
        train_targets[t] = train[col].values if col in train.columns else train[t].values

    # Map participant IDs to player indices
    all_pids = pd.concat([train['participant_id'], test['participant_id']]).unique()
    pid_map = {pid: idx for idx, pid in enumerate(sorted(all_pids))}
    train_players = train['participant_id'].map(pid_map).values
    test_players = test['participant_id'].map(pid_map).values

    return (train_trajs, test_trajs, train_targets,
            train_players, test_players, joints, test)


def make_hoop_relative(trajs, joints):
    """Convert to hoop-relative coordinates."""
    n_joints = len(joints)
    for j_idx in range(n_joints):
        base = j_idx * 3
        trajs[:, :, base] -= HOOP[0]
        trajs[:, :, base + 1] -= HOOP[1]
        trajs[:, :, base + 2] -= HOOP[2]
    return trajs


def extract_phase_features(trajs, joints):
    """Extract multi-resolution temporal features from trajectories.

    For each key joint and coordinate:
    1. Position statistics per phase (mean, std, range)
    2. Velocity statistics per phase (mean, std, max_abs)
    3. Acceleration statistics per phase (mean, std, max_abs)
    4. Peak velocity timing (frame index of max velocity)
    5. Inter-phase transitions (delta between phase means)
    """
    n_samples = trajs.shape[0]
    n_joints = len(joints)
    n_coords = 3

    feature_list = []
    feature_names = []

    # Compute velocities and accelerations
    velocities = np.gradient(trajs, axis=1)  # (n, 240, features)
    accelerations = np.gradient(velocities, axis=1)

    phase_names = list(PHASES.keys())

    for j_idx in range(n_joints):
        for c_idx in range(n_coords):
            feat_idx = j_idx * 3 + c_idx
            coord_name = ['x', 'y', 'z'][c_idx]
            joint_name = joints[j_idx]

            pos = trajs[:, :, feat_idx]  # (n, 240)
            vel = velocities[:, :, feat_idx]
            acc = accelerations[:, :, feat_idx]

            # Per-phase statistics
            for phase_name, (start, end) in PHASES.items():
                p_pos = pos[:, start:end]
                p_vel = vel[:, start:end]
                p_acc = acc[:, start:end]

                # Position stats
                feature_list.append(np.nanmean(p_pos, axis=1))
                feature_names.append(f"{joint_name}_{coord_name}_pos_mean_{phase_name}")

                feature_list.append(np.nanstd(p_pos, axis=1))
                feature_names.append(f"{joint_name}_{coord_name}_pos_std_{phase_name}")

                feature_list.append(np.nanmax(p_pos, axis=1) - np.nanmin(p_pos, axis=1))
                feature_names.append(f"{joint_name}_{coord_name}_pos_range_{phase_name}")

                # Velocity stats
                feature_list.append(np.nanmean(p_vel, axis=1))
                feature_names.append(f"{joint_name}_{coord_name}_vel_mean_{phase_name}")

                feature_list.append(np.nanstd(p_vel, axis=1))
                feature_names.append(f"{joint_name}_{coord_name}_vel_std_{phase_name}")

                feature_list.append(np.nanmax(np.abs(p_vel), axis=1))
                feature_names.append(f"{joint_name}_{coord_name}_vel_maxabs_{phase_name}")

                # Acceleration stats (only for dynamic phases)
                if phase_name in ('loading', 'acceleration', 'release'):
                    feature_list.append(np.nanmax(np.abs(p_acc), axis=1))
                    feature_names.append(f"{joint_name}_{coord_name}_acc_maxabs_{phase_name}")

            # Peak velocity timing (frame of max absolute velocity in release window)
            release_vel = np.abs(vel[:, 130:180])
            peak_frame = np.nanargmax(release_vel, axis=1) + 130
            feature_list.append(peak_frame.astype(np.float32))
            feature_names.append(f"{joint_name}_{coord_name}_peak_vel_frame")

            # Peak velocity magnitude
            feature_list.append(np.nanmax(np.abs(vel[:, 130:180]), axis=1))
            feature_names.append(f"{joint_name}_{coord_name}_peak_vel_mag")

            # Inter-phase transitions
            for i_phase in range(len(phase_names) - 1):
                p1_name = phase_names[i_phase]
                p2_name = phase_names[i_phase + 1]
                s1, e1 = PHASES[p1_name]
                s2, e2 = PHASES[p2_name]
                delta = np.nanmean(pos[:, s2:e2], axis=1) - np.nanmean(pos[:, s1:e1], axis=1)
                feature_list.append(delta)
                feature_names.append(f"{joint_name}_{coord_name}_delta_{p1_name}_to_{p2_name}")

    # Cross-joint features: hand separation timing
    if 'right_wrist' in joints and 'left_wrist' in joints:
        rw_idx = joints.index('right_wrist')
        lw_idx = joints.index('left_wrist')

        for c in range(3):
            rw = trajs[:, :, rw_idx * 3 + c]
            lw = trajs[:, :, lw_idx * 3 + c]
            sep = rw - lw  # hand separation

            # Separation velocity (how fast hands move apart)
            sep_vel = np.gradient(sep, axis=1)

            # Max separation rate in release window
            feature_list.append(np.nanmax(sep_vel[:, 130:180], axis=1))
            feature_names.append(f"hand_sep_vel_max_{['x','y','z'][c]}")

            # Frame of max separation
            peak_sep = np.nanargmax(np.abs(sep_vel[:, 100:200]), axis=1) + 100
            feature_list.append(peak_sep.astype(np.float32))
            feature_names.append(f"hand_sep_peak_frame_{['x','y','z'][c]}")

    # Shooting arm kinetic chain timing
    if all(j in joints for j in ['right_wrist', 'right_elbow', 'right_shoulder']):
        chain_joints = ['right_shoulder', 'right_elbow', 'right_wrist']
        chain_indices = [joints.index(j) for j in chain_joints]

        # Peak upward (z) velocity timing for each joint
        for j_name, j_idx in zip(chain_joints, chain_indices):
            z_vel = velocities[:, :, j_idx * 3 + 2]  # z velocity
            peak_z = np.nanargmax(z_vel[:, 120:180], axis=1) + 120
            feature_list.append(peak_z.astype(np.float32))
            feature_names.append(f"{j_name}_peak_z_vel_timing")

        # Kinetic chain delay: wrist peaks AFTER elbow AFTER shoulder
        shoulder_z = velocities[:, :, chain_indices[0] * 3 + 2]
        elbow_z = velocities[:, :, chain_indices[1] * 3 + 2]
        wrist_z = velocities[:, :, chain_indices[2] * 3 + 2]

        sh_peak = np.nanargmax(shoulder_z[:, 120:180], axis=1)
        el_peak = np.nanargmax(elbow_z[:, 120:180], axis=1)
        wr_peak = np.nanargmax(wrist_z[:, 120:180], axis=1)

        feature_list.append((el_peak - sh_peak).astype(np.float32))
        feature_names.append("kinetic_chain_delay_elbow_shoulder")

        feature_list.append((wr_peak - el_peak).astype(np.float32))
        feature_names.append("kinetic_chain_delay_wrist_elbow")

        feature_list.append((wr_peak - sh_peak).astype(np.float32))
        feature_names.append("kinetic_chain_delay_wrist_shoulder")

    features = np.column_stack(feature_list)

    # Replace any remaining NaN with 0
    features = np.nan_to_num(features, nan=0.0)

    print(f"  Phase features shape: {features.shape} ({len(feature_names)} features)")
    return features, feature_names


def run_pipeline(train_features, test_features, train_targets,
                 train_players, test_players, n_pls=10):
    """Per-player Ridge with PLS-compressed phase features."""

    unique_players = sorted(set(train_players))
    n_test = len(test_features)

    predictions = {t: np.zeros(n_test) for t in TARGETS}
    loo_preds = {t: np.full(len(train_features), np.nan) for t in TARGETS}

    for target in TARGETS:
        y_all = train_targets[target]

        for player in unique_players:
            train_mask = train_players == player
            test_mask = test_players == player

            X_train_p = train_features[train_mask]
            y_train_p = y_all[train_mask]
            X_test_p = test_features[test_mask]
            n_train_p = len(X_train_p)

            if n_train_p < 5:
                # Too few samples, use player mean
                mean_val = np.mean(y_train_p)
                predictions[target][test_mask] = mean_val
                loo_preds[target][train_mask] = mean_val
                continue

            # Scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_p)
            X_test_scaled = scaler.transform(X_test_p) if test_mask.any() else None

            # PLS compression
            n_comp = min(n_pls, n_train_p - 2, X_train_scaled.shape[1])
            if n_comp < 2:
                n_comp = 2

            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_train_scaled, y_train_p)
            X_train_pls = pls.transform(X_train_scaled)

            if X_test_scaled is not None:
                X_test_pls = pls.transform(X_test_scaled)

            # Compute distance matrix for kernel weights
            from scipy.spatial.distance import cdist
            D = cdist(X_train_pls, X_train_pls, 'euclidean')

            # Bandwidth from quantile
            bw = np.quantile(D[D > 0], 0.45) if (D > 0).any() else 1.0

            # Gaussian kernel
            K = np.exp(-0.5 * (D / bw) ** 2)

            # LOO predictions
            train_indices = np.where(train_mask)[0]
            for i_local in range(n_train_p):
                weights = K[i_local].copy()
                weights[i_local] = 0  # leave one out

                if weights.sum() < 1e-10:
                    loo_preds[target][train_indices[i_local]] = np.mean(y_train_p)
                    continue

                W = np.diag(weights)
                ridge = Ridge(alpha=10.0, fit_intercept=True)
                ridge.fit(X_train_pls, y_train_p, sample_weight=weights)
                pred = ridge.predict(X_train_pls[i_local:i_local+1])[0]
                loo_preds[target][train_indices[i_local]] = np.clip(pred, 0, 1)

            # Test predictions
            if X_test_scaled is not None and test_mask.any():
                D_test = cdist(X_test_pls, X_train_pls, 'euclidean')

                for i_test in range(D_test.shape[0]):
                    weights = np.exp(-0.5 * (D_test[i_test] / bw) ** 2)

                    if weights.sum() < 1e-10:
                        test_indices = np.where(test_mask)[0]
                        predictions[target][test_indices[i_test]] = np.mean(y_train_p)
                        continue

                    ridge = Ridge(alpha=10.0, fit_intercept=True)
                    ridge.fit(X_train_pls, y_train_p, sample_weight=weights)
                    test_indices = np.where(test_mask)[0]
                    pred = ridge.predict(X_test_pls[i_test:i_test+1])[0]
                    predictions[target][test_indices[i_test]] = np.clip(pred, 0, 1)

    return predictions, loo_preds


def main():
    print("=" * 60)
    print("TEMPORAL PHASE FEATURES")
    print("Multi-resolution temporal statistics for shooting motion")
    print("=" * 60)

    print("\nLoading data...")
    (train_trajs, test_trajs, train_targets,
     train_players, test_players, joints, test_df) = load_data()

    # Make hoop-relative
    print("Making hoop-relative...")
    train_trajs = make_hoop_relative(train_trajs, joints)
    test_trajs = make_hoop_relative(test_trajs, joints)

    # Extract phase features
    print("Extracting phase features...")
    print("  Train:")
    train_features, feature_names = extract_phase_features(train_trajs, joints)
    print("  Test:")
    test_features, _ = extract_phase_features(test_trajs, joints)

    # Run pipeline
    print("\nRunning per-player Ridge with PLS...")
    predictions, loo_preds = run_pipeline(
        train_features, test_features, train_targets,
        train_players, test_players, n_pls=10
    )

    # Evaluate LOO
    print("\nLOO Results:")
    loo_mses = {}
    for target in TARGETS:
        y_true = train_targets[target]
        y_pred = loo_preds[target]
        valid = ~np.isnan(y_pred)
        mse = np.mean((y_true[valid] - y_pred[valid]) ** 2)
        loo_mses[target] = mse
        print(f"  {target}: LOO MSE = {mse:.6f}")
    mean_loo = np.mean(list(loo_mses.values()))
    print(f"  Mean LOO MSE: {mean_loo:.6f}")

    # Load anchor for diversity analysis
    anchor_path = SUBMISSION_DIR / "submission_2716.csv"
    if anchor_path.exists():
        anchor = pd.read_csv(anchor_path)
        print("\nDiversity vs Sub 2716:")
        for target in TARGETS:
            col = f"scaled_{target}"
            r = np.corrcoef(predictions[target], anchor[col].values)[0, 1]
            print(f"  {target}: r = {r:.4f}")

    # Generate submissions
    print("\nGenerating submissions...")

    # 1. Standalone
    sub_num = get_next_submission_number()
    sub_df = test_df[['id']].copy()
    for target in TARGETS:
        sub_df[f"scaled_{target}"] = np.clip(predictions[target], 0, 1)
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"  Sub {sub_num}: Standalone temporal phase features")
    standalone_num = sub_num

    # 2-5. Blends with Sub 2716
    if anchor_path.exists():
        anchor = pd.read_csv(anchor_path)
        for w in [0.02, 0.05, 0.10, 0.20]:
            sub_num = get_next_submission_number()
            sub_df = test_df[['id']].copy()
            for target in TARGETS:
                col = f"scaled_{target}"
                blended = w * predictions[target] + (1 - w) * anchor[col].values
                sub_df[col] = np.clip(blended, 0, 1)
            sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
            sub_df.to_csv(sub_path, index=False)
            print(f"  Sub {sub_num}: {int(w*100)}% phase + {int((1-w)*100)}% Sub 2716")

    # 6. Per-target blend: use phase features where they're most diverse
    if anchor_path.exists():
        sub_num = get_next_submission_number()
        sub_df = test_df[['id']].copy()
        # Per-target weights based on diversity
        for target in TARGETS:
            col = f"scaled_{target}"
            r = np.corrcoef(predictions[target], anchor[col].values)[0, 1]
            # More weight when more diverse (lower correlation)
            w = max(0.02, min(0.30, 0.5 * (1 - r)))
            blended = w * predictions[target] + (1 - w) * anchor[col].values
            sub_df[col] = np.clip(blended, 0, 1)
            print(f"    {target}: w={w:.3f} (r={r:.4f})")
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"  Sub {sub_num}: Per-target diversity-weighted phase blend")

    print("\nDone!")


if __name__ == "__main__":
    main()

"""
Follow-up diagnostic: per-player velocity correlations at FIXED frames.

We know:
- Global r(wrist_speed @ f153, angle) = -0.53 (strong)
- But this might be between-player variance
- Our model is per-player, so we need WITHIN-player signal

Also sweep frames 130-170 for best per-player velocity signal per target.
And build a quick Ridge-LOO model using velocity features to check diversity.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

HOOP_FEET = np.array([5.25, -25.0, 10.0])
FEET_TO_M = 0.3048
DT = 1.0 / 60.0
N_FRAMES = 240


def safe_savgol(arr, window=11, poly=3, deriv=0, delta=1.0):
    arr = np.array(arr, dtype=np.float64)
    bad = np.isnan(arr) | np.isinf(arr)
    if np.all(bad):
        return np.zeros_like(arr)
    if np.any(bad):
        good = ~bad
        arr[bad] = np.interp(np.where(bad)[0], np.where(good)[0], arr[good])
    if len(arr) < window:
        return arr
    return savgol_filter(arr, window, poly, deriv=deriv, delta=delta)


def parse_array_string(s):
    s = str(s).replace("nan", "NaN").replace("null", "NaN")
    return np.nan_to_num(np.array(json.loads(s), dtype=np.float64), nan=0.0)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    kp_cols = [c for c in df.columns if c not in meta_cols]
    kp_names = [c[:-2] for c in kp_cols if c.endswith("_x")]
    kp_index = {name: i for i, name in enumerate(kp_names)}
    n_kp = len(kp_names)
    n = len(df)
    X_3d = np.zeros((n, N_FRAMES, n_kp, 3), dtype=np.float32)
    for idx, row in df.iterrows():
        for col_i, col in enumerate(kp_cols):
            X_3d[idx, :, col_i // 3, col_i % 3] = parse_array_string(row[col])
        if (idx + 1) % 100 == 0:
            print(f"  Loaded {idx+1}/{n}...")
    return X_3d, kp_names, kp_index, df


def get_joint(ts_3d, kp_index, name, smooth=True, window=11):
    idx = kp_index.get(name)
    if idx is None:
        return np.zeros((N_FRAMES, 3))
    traj = ts_3d[:, idx, :].copy()
    for ax in range(3):
        bad = np.isnan(traj[:, ax]) | np.isinf(traj[:, ax])
        if np.any(bad) and not np.all(bad):
            good = ~bad
            traj[bad, ax] = np.interp(np.where(bad)[0], np.where(good)[0], traj[good, ax])
    if smooth:
        for ax in range(3):
            traj[:, ax] = safe_savgol(traj[:, ax], window, 3)
    return traj


def compute_velocity(positions_feet, window=7):
    vel = np.zeros((N_FRAMES, 3))
    for ax in range(3):
        pos_m = positions_feet[:, ax] * FEET_TO_M
        vel[:, ax] = safe_savgol(pos_m, window, 3, deriv=1, delta=DT)
    return vel


def decompose_velocity(vel_3d, player_pos):
    """Decompose into forward/lateral/up relative to hoop direction."""
    to_hoop = HOOP_FEET[:2] - player_pos[:2]
    d = np.linalg.norm(to_hoop)
    if d > 1e-6:
        fwd = to_hoop / d
    else:
        fwd = np.array([0, -1.0])
    lat = np.array([-fwd[1], fwd[0]])

    v_fwd = vel_3d[0]*fwd[0] + vel_3d[1]*fwd[1]
    v_lat = vel_3d[0]*lat[0] + vel_3d[1]*lat[1]
    v_up = vel_3d[2]
    return v_fwd, v_lat, v_up


def main():
    print("=" * 70)
    print("VELOCITY PER-PLAYER SIGNAL ANALYSIS")
    print("=" * 70)

    print("\nLoading data...")
    X_3d, kp_names, kp_index, df = load_data(DATA_DIR / "train.csv")
    pids = df["participant_id"].values
    n = len(df)

    import joblib
    targets = {}
    for t in ["angle", "depth", "left_right"]:
        scaler_path = DATA_DIR / f"scaler_{t}.pkl"
        if scaler_path.exists():
            s = joblib.load(scaler_path)
            targets[t] = s.transform(df[t].values.reshape(-1, 1)).ravel()
        else:
            v = df[t].values
            targets[t] = (v - v.min()) / (v.max() - v.min() + 1e-10)

    # ================================================================
    # Precompute velocity profiles for wrist + 3 fingertips
    # ================================================================
    print("\nPrecomputing velocity profiles...")

    joints_to_track = {
        'right_wrist': None,
        'right_elbow': None,
        'right_shoulder': None,
        'right_second_finger_distal': None,
        'right_third_finger_distal': None,
    }

    velocities = {}
    for jname in joints_to_track:
        vel_all = np.zeros((n, N_FRAMES, 3))
        for i in range(n):
            traj = get_joint(X_3d[i], kp_index, jname, smooth=True, window=11)
            vel_all[i] = compute_velocity(traj, window=7)
        velocities[jname] = vel_all

    # Player positions for hoop-relative decomposition (at stance)
    player_pos = np.zeros((n, 3))
    for i in range(n):
        hip = get_joint(X_3d[i], kp_index, 'mid_hip', smooth=True)
        player_pos[i] = hip[20]
        player_pos[i, 2] = 0

    # ================================================================
    # SWEEP: per-player correlations at each frame
    # ================================================================
    print("\n" + "=" * 50)
    print("FRAME SWEEP: per-player correlations for wrist velocity")
    print("=" * 50)

    frames_to_check = list(range(100, 200, 2))

    best_per_target = {}

    for tname in ["angle", "depth", "left_right"]:
        y = targets[tname]
        print(f"\n  {tname}:")
        best_r = -999
        best_feat = ""
        best_frame = 0

        for frame in frames_to_check:
            # Compute features at this frame
            wrist_vel = velocities['right_wrist'][:, frame, :]
            ft_vel = velocities['right_second_finger_distal'][:, frame, :]

            # Decompose
            w_fwd = np.zeros(n)
            w_lat = np.zeros(n)
            w_up = np.zeros(n)
            f_fwd = np.zeros(n)
            f_lat = np.zeros(n)
            f_up = np.zeros(n)

            for i in range(n):
                w_fwd[i], w_lat[i], w_up[i] = decompose_velocity(wrist_vel[i], player_pos[i])
                f_fwd[i], f_lat[i], f_up[i] = decompose_velocity(ft_vel[i], player_pos[i])

            w_speed = np.linalg.norm(wrist_vel, axis=1)
            f_speed = np.linalg.norm(ft_vel, axis=1)
            w_horiz = np.sqrt(w_fwd**2 + w_lat**2)
            w_angle_est = np.degrees(np.arctan2(w_up, w_horiz + 1e-8))

            # Per-player correlations
            features = {
                'w_speed': w_speed, 'w_fwd': w_fwd, 'w_lat': w_lat, 'w_up': w_up,
                'w_angle': w_angle_est,
                'f_speed': f_speed, 'f_fwd': f_fwd, 'f_lat': f_lat, 'f_up': f_up,
            }

            for fname, fvals in features.items():
                pp_r_list = []
                for pid in sorted(np.unique(pids)):
                    mask = pids == pid
                    if np.sum(mask) < 10:
                        continue
                    v = fvals[mask]
                    t = y[mask]
                    if np.std(v) < 1e-10:
                        continue
                    r = np.corrcoef(v, t)[0, 1]
                    if np.isfinite(r):
                        pp_r_list.append(r)

                if pp_r_list:
                    mean_pp_r = np.mean(pp_r_list)
                    if abs(mean_pp_r) > abs(best_r):
                        best_r = mean_pp_r
                        best_feat = fname
                        best_frame = frame

        print(f"    Best: {best_feat} @ frame {best_frame}: mean per-player r = {best_r:+.4f}")
        best_per_target[tname] = (best_feat, best_frame, best_r)

    # ================================================================
    # DETAILED: show per-player correlations at best frames
    # ================================================================
    print("\n" + "=" * 50)
    print("DETAILED: per-player correlations at best frames")
    print("=" * 50)

    for tname in ["angle", "depth", "left_right"]:
        y = targets[tname]
        best_feat, best_frame, _ = best_per_target[tname]
        print(f"\n  {tname} ({best_feat} @ f{best_frame}):")

        jname = 'right_wrist' if best_feat.startswith('w_') else 'right_second_finger_distal'
        vel = velocities[jname][:, best_frame, :]

        for pid in sorted(np.unique(pids)):
            mask = pids == pid
            n_p = np.sum(mask)
            v = vel[mask]
            t = y[mask]

            # Decompose
            fwd = np.zeros(n_p)
            lat = np.zeros(n_p)
            up = np.zeros(n_p)
            for j, idx in enumerate(np.where(mask)[0]):
                fwd[j], lat[j], up[j] = decompose_velocity(v[j], player_pos[idx])

            speed = np.linalg.norm(v, axis=1)

            feats = {'speed': speed, 'fwd': fwd, 'lat': lat, 'up': up}
            horiz = np.sqrt(fwd**2 + lat**2)
            feats['angle_est'] = np.degrees(np.arctan2(up, horiz + 1e-8))

            r_strs = []
            for fn, fv in feats.items():
                if np.std(fv) > 1e-10:
                    r = np.corrcoef(fv, t)[0, 1]
                    r_strs.append(f"{fn}={r:+.3f}")

            print(f"    P{pid} (n={n_p}): {', '.join(r_strs)}")

    # ================================================================
    # BROADER SWEEP: check ALL joints, not just wrist/finger
    # ================================================================
    print("\n" + "=" * 50)
    print("BROADER: Check ALL available joints for best per-player signal")
    print("=" * 50)

    # Check some key joints at a few frames
    broad_joints = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'right_second_finger_distal', 'right_third_finger_distal',
        'left_wrist', 'left_shoulder',
        'right_knee', 'right_ankle', 'mid_hip', 'neck',
    ]

    for tname in ["angle", "depth", "left_right"]:
        y = targets[tname]
        print(f"\n  {tname}:")
        results = []

        for jname in broad_joints:
            if jname not in kp_index:
                continue

            # Compute velocity for this joint
            vel_j = np.zeros((n, N_FRAMES, 3))
            for i in range(n):
                traj = get_joint(X_3d[i], kp_index, jname, smooth=True, window=11)
                vel_j[i] = compute_velocity(traj, window=7)

            for frame in [130, 140, 145, 150, 153, 155, 160, 170]:
                v = vel_j[:, frame, :]

                # Decompose per shot
                fwd_all = np.zeros(n)
                lat_all = np.zeros(n)
                up_all = np.zeros(n)
                for i in range(n):
                    fwd_all[i], lat_all[i], up_all[i] = decompose_velocity(v[i], player_pos[i])
                speed_all = np.linalg.norm(v, axis=1)

                for comp_name, comp_vals in [('speed', speed_all), ('fwd', fwd_all),
                                              ('lat', lat_all), ('up', up_all)]:
                    pp_r = []
                    for pid in sorted(np.unique(pids)):
                        mask = pids == pid
                        if np.sum(mask) < 10:
                            continue
                        cv = comp_vals[mask]
                        ct = y[mask]
                        if np.std(cv) < 1e-10:
                            continue
                        r = np.corrcoef(cv, ct)[0, 1]
                        if np.isfinite(r):
                            pp_r.append(r)

                    if pp_r:
                        mean_r = np.mean(pp_r)
                        results.append((abs(mean_r), mean_r, jname, frame, comp_name))

        results.sort(reverse=True)
        for _, r, jname, frame, comp in results[:10]:
            print(f"    r={r:+.4f}  {jname:35s} f{frame:3d} {comp}")

    # ================================================================
    # QUICK TEST: build features and test with Ridge LOO
    # ================================================================
    print("\n" + "=" * 50)
    print("QUICK MODEL: Velocity features at 3 fixed frames -> Ridge LOO")
    print("Frames: 140 (acceleration), 150 (near-release), 160 (follow-through)")
    print("=" * 50)

    from sklearn.linear_model import Ridge
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import StandardScaler

    feature_frames = [130, 140, 150, 160]
    feature_joints = ['right_wrist', 'right_elbow', 'right_second_finger_distal']

    # Build feature matrix
    print("\nBuilding velocity feature matrix...")
    feat_list = []
    for i in range(n):
        row = []
        for jname in feature_joints:
            traj = get_joint(X_3d[i], kp_index, jname, smooth=True, window=11)
            vel = compute_velocity(traj, window=7)
            for ff in feature_frames:
                # Decompose velocity
                v_fwd, v_lat, v_up = decompose_velocity(vel[ff], player_pos[i])
                row.extend([v_fwd, v_lat, v_up])
                # Also add position at this frame (hoop-relative)
                pos_rel = traj[ff] - HOOP_FEET
                row.extend([pos_rel[0], pos_rel[1], pos_rel[2]])
        feat_list.append(row)

    X_vel = np.array(feat_list, dtype=np.float32)
    X_vel = np.nan_to_num(X_vel, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Feature matrix: {X_vel.shape}")

    # Simple LOO per-player Ridge (no PLS to avoid leakage)
    for tname in ["angle", "depth", "left_right"]:
        y = targets[tname]
        preds = np.full(n, np.nan)

        for pid in sorted(np.unique(pids)):
            mask = pids == pid
            idx = np.where(mask)[0]
            X_p = X_vel[mask]
            y_p = y[mask]
            n_p = len(y_p)

            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_p)

            D = pairwise_distances(X_s)
            bw = np.quantile(D[np.triu_indices(n_p, k=1)], 0.3)
            if bw < 1e-10:
                bw = 1.0

            for j in range(n_p):
                w = np.exp(-0.5 * (D[j] / bw) ** 2)
                w[j] = 0
                if w.sum() < 1e-10:
                    preds[idx[j]] = np.mean(np.delete(y_p, j))
                    continue
                sqrt_w = np.sqrt(w)
                X_w = X_s * sqrt_w[:, None]
                y_w = y_p * sqrt_w
                ridge = Ridge(alpha=10.0, fit_intercept=True)
                ridge.fit(np.delete(X_w, j, 0), np.delete(y_w, j))
                preds[idx[j]] = ridge.predict(X_s[j:j+1])[0]

        mse = np.mean((preds - y) ** 2)
        print(f"  {tname}: LOO MSE = {mse:.6f}")

    # Overall
    mean_mse = np.mean([
        np.mean((np.where(np.isnan(preds), y.mean(), preds) - targets[t]) ** 2)
        for t, y in zip(["angle", "depth", "left_right"],
                        [targets["angle"], targets["depth"], targets["left_right"]])
    ])

    print(f"\n  (Note: compare to honest LOO baseline ~0.006830)")
    print(f"  (Compare to sparse physics LOO = 0.013831)")

    print("\nDone.")


if __name__ == "__main__":
    main()

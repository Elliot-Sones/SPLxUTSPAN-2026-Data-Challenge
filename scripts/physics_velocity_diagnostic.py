"""
Diagnostic: WHY are ball velocity measurements physically implausible?

Known problem: measured release speed = 2.01 m/s (expected 6.5-7.5 m/s for free throw)
Known problem: measured release angle = 16.4 deg (expected 50-55 deg)
Known problem: all correlations with targets near zero

Hypotheses to test:
1. Ball position proxy (wrist + 0.6*(fingertip-wrist)) is too slow - the blend dampens velocity
2. Savgol window too wide - smooths out the fast release event
3. Force-based release detection finds wrong frame
4. Units error somewhere

Approach: measure velocity from multiple sources, check physical plausibility at every frame.
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
G = 9.81
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


def get_joint(ts_3d, kp_index, name, smooth=True, window=15):
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


def compute_velocity_profile(positions_feet, method='savgol', window=9):
    """Compute velocity in m/s from position in feet."""
    vel = np.zeros((N_FRAMES, 3))
    for ax in range(3):
        pos_m = positions_feet[:, ax] * FEET_TO_M
        vel[:, ax] = safe_savgol(pos_m, window, 3, deriv=1, delta=DT)
    speed = np.linalg.norm(vel, axis=1)
    return vel, speed


def main():
    print("=" * 70)
    print("PHYSICS VELOCITY DIAGNOSTIC")
    print("=" * 70)

    print("\nLoading training data...")
    X_3d, kp_names, kp_index, df = load_data(DATA_DIR / "train.csv")
    pids = df["participant_id"].values

    # Scale targets to [0,1]
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

    n = len(df)

    # ================================================================
    # TEST 1: Compare velocity sources at EVERY frame
    # ================================================================
    print("\n" + "=" * 50)
    print("TEST 1: Velocity source comparison")
    print("Which body part gives physically plausible ball speed?")
    print("Expected free throw release speed: 6.5-7.5 m/s (21-25 ft/s)")
    print("=" * 50)

    # Track peak speeds from different sources
    sources = {
        'wrist': 'right_wrist',
        'fingertip_2nd': 'right_second_finger_distal',
        'fingertip_3rd': 'right_third_finger_distal',
        'fingertip_4th': 'right_fourth_finger_distal',
        'elbow': 'right_elbow',
    }

    peak_speeds_per_source = {s: [] for s in sources}
    peak_frames_per_source = {s: [] for s in sources}

    for i in range(n):
        for sname, kp_name in sources.items():
            traj = get_joint(X_3d[i], kp_index, kp_name, smooth=True, window=15)
            _, speed = compute_velocity_profile(traj, window=9)
            # Peak speed in frames 100-200 (shot phase)
            peak_f = 100 + np.argmax(speed[100:200])
            peak_s = speed[peak_f]
            peak_speeds_per_source[sname].append(peak_s)
            peak_frames_per_source[sname].append(peak_f)

    print("\nPeak speed in frames 100-200 (m/s):")
    print(f"  {'Source':<18} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6}  {'Peak frame (mean)':>18}")
    for sname in sources:
        speeds = peak_speeds_per_source[sname]
        frames = peak_frames_per_source[sname]
        print(f"  {sname:<18} {np.mean(speeds):6.2f} {np.std(speeds):6.2f} "
              f"{np.min(speeds):6.2f} {np.max(speeds):6.2f}  {np.mean(frames):18.1f}")

    # Ball proxy (blended)
    ball_proxy_peak_speeds = []
    for i in range(n):
        wrist = get_joint(X_3d[i], kp_index, 'right_wrist', smooth=True, window=15)
        ft_keys = ['right_second_finger_distal', 'right_third_finger_distal',
                   'right_fourth_finger_distal']
        ft_trajs = [get_joint(X_3d[i], kp_index, k, smooth=True, window=15)
                    for k in ft_keys if k in kp_index]
        ft_center = np.mean(ft_trajs, axis=0) if ft_trajs else wrist.copy()
        ball = wrist + 0.6 * (ft_center - wrist)
        for ax in range(3):
            ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)
        _, speed = compute_velocity_profile(ball, window=9)
        peak_s = np.max(speed[100:200])
        ball_proxy_peak_speeds.append(peak_s)

    print(f"  {'ball_proxy(0.6)':<18} {np.mean(ball_proxy_peak_speeds):6.2f} "
          f"{np.std(ball_proxy_peak_speeds):6.2f} "
          f"{np.min(ball_proxy_peak_speeds):6.2f} {np.max(ball_proxy_peak_speeds):6.2f}")

    # ================================================================
    # TEST 2: Effect of smoothing window on velocity magnitude
    # ================================================================
    print("\n" + "=" * 50)
    print("TEST 2: Savgol window effect on peak fingertip speed")
    print("=" * 50)

    windows = [5, 7, 9, 11, 15, 21]
    smooth_windows = [5, 7, 11, 15, 21]  # For initial position smoothing

    # Just test first 50 shots for speed
    n_test = min(50, n)
    for sw in [7, 15]:  # position smooth window
        print(f"\n  Position smooth window = {sw}:")
        for vw in windows:
            speeds = []
            for i in range(n_test):
                traj = get_joint(X_3d[i], kp_index, 'right_second_finger_distal',
                               smooth=True, window=sw)
                _, speed = compute_velocity_profile(traj, window=vw)
                speeds.append(np.max(speed[100:200]))
            print(f"    Velocity window={vw:2d}: peak speed = {np.mean(speeds):.2f} +/- {np.std(speeds):.2f} m/s")

    # ================================================================
    # TEST 3: Raw (unsmoothed) velocity to check noise vs signal
    # ================================================================
    print("\n" + "=" * 50)
    print("TEST 3: Raw vs smoothed fingertip velocity")
    print("=" * 50)

    raw_peak = []
    smooth_peak = []
    for i in range(n_test):
        # Raw: no position smoothing, smallest possible velocity window
        traj_raw = get_joint(X_3d[i], kp_index, 'right_second_finger_distal',
                            smooth=False)
        _, speed_raw = compute_velocity_profile(traj_raw, window=5)

        # Smooth: heavy smoothing
        traj_smooth = get_joint(X_3d[i], kp_index, 'right_second_finger_distal',
                               smooth=True, window=15)
        _, speed_smooth = compute_velocity_profile(traj_smooth, window=9)

        raw_peak.append(np.max(speed_raw[100:200]))
        smooth_peak.append(np.max(speed_smooth[100:200]))

    print(f"  Raw peak speed:     {np.mean(raw_peak):.2f} +/- {np.std(raw_peak):.2f} m/s (range {np.min(raw_peak):.2f} - {np.max(raw_peak):.2f})")
    print(f"  Smoothed peak speed: {np.mean(smooth_peak):.2f} +/- {np.std(smooth_peak):.2f} m/s (range {np.min(smooth_peak):.2f} - {np.max(smooth_peak):.2f})")

    # ================================================================
    # TEST 4: Physical plausibility of position data
    # ================================================================
    print("\n" + "=" * 50)
    print("TEST 4: Physical plausibility of position data")
    print("Are positions in feet? Check wrist height, distance from hoop")
    print("=" * 50)

    wrist_heights = []
    hoop_dists = []
    for i in range(n_test):
        wrist = get_joint(X_3d[i], kp_index, 'right_wrist', smooth=True)
        # Wrist height at frame 150 (near release)
        wrist_heights.append(wrist[150, 2])
        # Horizontal distance to hoop at frame 150
        d_xy = np.sqrt((wrist[150, 0] - HOOP_FEET[0])**2 +
                        (wrist[150, 1] - HOOP_FEET[1])**2)
        hoop_dists.append(d_xy)

    print(f"  Wrist height at f150: {np.mean(wrist_heights):.2f} +/- {np.std(wrist_heights):.2f} feet")
    print(f"    Expected: ~7-8 feet (shot release height)")
    print(f"  Distance to hoop at f150: {np.mean(hoop_dists):.2f} +/- {np.std(hoop_dists):.2f} feet")
    print(f"    Expected: ~13-15 feet (free throw line distance)")

    # ================================================================
    # TEST 5: Frame-by-frame velocity at fingertip vs wrist
    # Show one example shot profile
    # ================================================================
    print("\n" + "=" * 50)
    print("TEST 5: Example velocity profile (shot 0)")
    print("=" * 50)

    i = 0
    wrist = get_joint(X_3d[i], kp_index, 'right_wrist', smooth=True, window=11)
    ft = get_joint(X_3d[i], kp_index, 'right_second_finger_distal', smooth=True, window=11)

    _, wrist_speed = compute_velocity_profile(wrist, window=7)
    _, ft_speed = compute_velocity_profile(ft, window=7)

    ball_proxy = wrist + 0.6 * (ft - wrist)
    _, proxy_speed = compute_velocity_profile(ball_proxy, window=7)

    print(f"  Frame  | Wrist m/s | Finger m/s | Proxy m/s | Wrist_z ft | Finger_z ft")
    for f in range(100, 220, 5):
        print(f"  {f:5d}  | {wrist_speed[f]:9.3f} | {ft_speed[f]:10.3f} | {proxy_speed[f]:9.3f} | "
              f"{wrist[f, 2]:10.3f} | {ft[f, 2]:11.3f}")

    print(f"\n  Peak wrist speed: {np.max(wrist_speed[100:200]):.3f} m/s at frame {100+np.argmax(wrist_speed[100:200])}")
    print(f"  Peak finger speed: {np.max(ft_speed[100:200]):.3f} m/s at frame {100+np.argmax(ft_speed[100:200])}")
    print(f"  Peak proxy speed: {np.max(proxy_speed[100:200]):.3f} m/s at frame {100+np.argmax(proxy_speed[100:200])}")

    # ================================================================
    # TEST 6: Correlation of fingertip velocity at peak with targets
    # ================================================================
    print("\n" + "=" * 50)
    print("TEST 6: Correlation with targets using FINGERTIP velocity at peak frame")
    print("=" * 50)

    ft_vel_at_peak = np.zeros((n, 3))
    ft_speed_at_peak = np.zeros(n)
    peak_frame_per_shot = np.zeros(n, dtype=int)

    for i in range(n):
        ft = get_joint(X_3d[i], kp_index, 'right_second_finger_distal',
                      smooth=True, window=11)
        vel, speed = compute_velocity_profile(ft, window=7)
        peak_f = 100 + np.argmax(speed[100:200])
        ft_vel_at_peak[i] = vel[peak_f]
        ft_speed_at_peak[i] = speed[peak_f]
        peak_frame_per_shot[i] = peak_f

    print(f"  Peak frame: {np.mean(peak_frame_per_shot):.0f} +/- {np.std(peak_frame_per_shot):.0f} "
          f"(range {np.min(peak_frame_per_shot)}-{np.max(peak_frame_per_shot)})")
    print(f"  Peak speed: {np.mean(ft_speed_at_peak):.2f} +/- {np.std(ft_speed_at_peak):.2f} m/s")

    # Compute hoop direction per shot for proper velocity decomposition
    for i in range(n):
        wrist_pos = get_joint(X_3d[i], kp_index, 'right_wrist', smooth=True)[150]
        to_hoop = HOOP_FEET[:2] - wrist_pos[:2]
        d = np.linalg.norm(to_hoop)
        if d > 1e-6:
            forward = to_hoop / d
        else:
            forward = np.array([0, -1.0])
        lateral = np.array([-forward[1], forward[0]])

        v = ft_vel_at_peak[i]
        v_forward_i = v[0] * forward[0] + v[1] * forward[1]
        v_lateral_i = v[0] * lateral[0] + v[1] * lateral[1]
        v_up_i = v[2]
        v_horiz = np.sqrt(v_forward_i**2 + v_lateral_i**2)

        ft_vel_at_peak[i] = [v_forward_i, v_lateral_i, v_up_i]

    # Correlations
    v_forward = ft_vel_at_peak[:, 0]
    v_lateral = ft_vel_at_peak[:, 1]
    v_up = ft_vel_at_peak[:, 2]
    v_horiz = np.sqrt(v_forward**2 + v_lateral**2)
    release_angle_est = np.degrees(np.arctan2(v_up, v_horiz + 1e-8))

    print(f"\n  Velocity components (m/s):")
    print(f"    v_forward: {np.mean(v_forward):.2f} +/- {np.std(v_forward):.2f}")
    print(f"    v_lateral: {np.mean(v_lateral):.2f} +/- {np.std(v_lateral):.2f}")
    print(f"    v_up:      {np.mean(v_up):.2f} +/- {np.std(v_up):.2f}")
    print(f"    speed:     {np.mean(ft_speed_at_peak):.2f} +/- {np.std(ft_speed_at_peak):.2f}")
    print(f"    angle_est: {np.mean(release_angle_est):.1f} +/- {np.std(release_angle_est):.1f} deg")

    print(f"\n  Correlations with targets:")
    for tname, expected_corr in [("angle", "release_angle_est should be positive"),
                                   ("depth", "speed should be positive"),
                                   ("left_right", "v_lateral should be positive")]:
        y = targets[tname]
        print(f"\n  {tname} ({expected_corr}):")
        for feat_name, feat_vals in [("v_forward", v_forward),
                                      ("v_lateral", v_lateral),
                                      ("v_up", v_up),
                                      ("speed", ft_speed_at_peak),
                                      ("release_angle_est", release_angle_est)]:
            r = np.corrcoef(feat_vals, y)[0, 1]
            print(f"    {feat_name:20s}: r = {r:+.4f}")

    # ================================================================
    # TEST 7: Per-player correlations (signal may be masked by player effects)
    # ================================================================
    print("\n" + "=" * 50)
    print("TEST 7: Per-player correlations (removing between-player variance)")
    print("=" * 50)

    for tname in ["angle", "depth", "left_right"]:
        y = targets[tname]
        print(f"\n  {tname}:")
        for feat_name, feat_vals in [("speed", ft_speed_at_peak),
                                      ("release_angle_est", release_angle_est),
                                      ("v_lateral", v_lateral)]:
            per_player_r = []
            for pid in sorted(np.unique(pids)):
                mask = pids == pid
                if np.sum(mask) < 10:
                    continue
                r = np.corrcoef(feat_vals[mask], y[mask])[0, 1]
                per_player_r.append(r)
            mean_r = np.mean(per_player_r) if per_player_r else 0
            print(f"    {feat_name:20s}: per-player r = {mean_r:+.4f} "
                  f"(range {np.min(per_player_r):+.4f} to {np.max(per_player_r):+.4f})")

    # ================================================================
    # TEST 8: Use FIXED frames (our known best frames) instead of force detection
    # ================================================================
    print("\n" + "=" * 50)
    print("TEST 8: Fingertip velocity at FIXED frames vs force-detected")
    print("Compare known best frames (150-153) vs force detection")
    print("=" * 50)

    fixed_frames = [140, 145, 150, 153, 155, 160, 170]

    for ff in fixed_frames:
        ft_vel_fixed = np.zeros((n, 3))
        ft_speed_fixed = np.zeros(n)
        for i in range(n):
            ft = get_joint(X_3d[i], kp_index, 'right_second_finger_distal',
                          smooth=True, window=11)
            vel, speed = compute_velocity_profile(ft, window=7)
            ft_vel_fixed[i] = vel[ff]
            ft_speed_fixed[i] = speed[ff]

        # Decompose into forward/lateral/up
        for i in range(n):
            wrist_pos = get_joint(X_3d[i], kp_index, 'right_wrist', smooth=True)[ff]
            to_hoop = HOOP_FEET[:2] - wrist_pos[:2]
            d = np.linalg.norm(to_hoop)
            if d > 1e-6:
                fwd = to_hoop / d
            else:
                fwd = np.array([0, -1.0])
            lat = np.array([-fwd[1], fwd[0]])
            v = ft_vel_fixed[i]
            ft_vel_fixed[i] = [v[0]*fwd[0] + v[1]*fwd[1],
                               v[0]*lat[0] + v[1]*lat[1],
                               v[2]]

        v_up_f = ft_vel_fixed[:, 2]
        v_fwd_f = ft_vel_fixed[:, 0]
        v_lat_f = ft_vel_fixed[:, 1]
        v_horiz_f = np.sqrt(v_fwd_f**2 + v_lat_f**2)
        angle_est_f = np.degrees(np.arctan2(v_up_f, v_horiz_f + 1e-8))

        r_angle = np.corrcoef(angle_est_f, targets["angle"])[0, 1]
        r_depth = np.corrcoef(ft_speed_fixed, targets["depth"])[0, 1]
        r_lr = np.corrcoef(v_lat_f, targets["left_right"])[0, 1]

        print(f"  Frame {ff:3d}: speed={np.mean(ft_speed_fixed):.2f} m/s, "
              f"angle_est={np.mean(angle_est_f):.1f} deg, "
              f"r(angle)={r_angle:+.4f}, r(depth)={r_depth:+.4f}, r(LR)={r_lr:+.4f}")

    # ================================================================
    # TEST 9: What does the WRIST velocity look like at known frames?
    # ================================================================
    print("\n" + "=" * 50)
    print("TEST 9: Wrist velocity at fixed frames")
    print("=" * 50)

    for ff in fixed_frames:
        w_vel = np.zeros((n, 3))
        w_speed = np.zeros(n)
        for i in range(n):
            w = get_joint(X_3d[i], kp_index, 'right_wrist', smooth=True, window=11)
            vel, speed = compute_velocity_profile(w, window=7)
            w_vel[i] = vel[ff]
            w_speed[i] = speed[ff]

        # Decompose
        for i in range(n):
            wrist_pos = get_joint(X_3d[i], kp_index, 'right_wrist', smooth=True)[ff]
            to_hoop = HOOP_FEET[:2] - wrist_pos[:2]
            d = np.linalg.norm(to_hoop)
            if d > 1e-6:
                fwd = to_hoop / d
            else:
                fwd = np.array([0, -1.0])
            lat = np.array([-fwd[1], fwd[0]])
            v = w_vel[i]
            w_vel[i] = [v[0]*fwd[0] + v[1]*fwd[1],
                        v[0]*lat[0] + v[1]*lat[1],
                        v[2]]

        v_up_w = w_vel[:, 2]
        v_fwd_w = w_vel[:, 0]
        v_lat_w = w_vel[:, 1]
        v_horiz_w = np.sqrt(v_fwd_w**2 + v_lat_w**2)
        angle_est_w = np.degrees(np.arctan2(v_up_w, v_horiz_w + 1e-8))

        r_angle = np.corrcoef(angle_est_w, targets["angle"])[0, 1]
        r_depth = np.corrcoef(w_speed, targets["depth"])[0, 1]
        r_lr = np.corrcoef(v_lat_w, targets["left_right"])[0, 1]

        print(f"  Frame {ff:3d}: speed={np.mean(w_speed):.2f} m/s, "
              f"angle_est={np.mean(angle_est_w):.1f} deg, "
              f"r(angle)={r_angle:+.4f}, r(depth)={r_depth:+.4f}, r(LR)={r_lr:+.4f}")

    # ================================================================
    # TEST 10: Sanity check - position at frame 0 vs frame 150
    # ================================================================
    print("\n" + "=" * 50)
    print("TEST 10: Position sanity check")
    print("=" * 50)

    for kp in ['right_wrist', 'right_second_finger_distal', 'mid_hip', 'right_ankle']:
        pos_0 = []
        pos_150 = []
        for i in range(n_test):
            traj = get_joint(X_3d[i], kp_index, kp, smooth=False)
            pos_0.append(traj[0])
            pos_150.append(traj[150])
        pos_0 = np.mean(pos_0, axis=0)
        pos_150 = np.mean(pos_150, axis=0)
        disp = np.linalg.norm(pos_150 - pos_0)
        print(f"  {kp:30s}: f0={pos_0} -> f150={pos_150}  displacement={disp:.2f} ft")

    print("\nDone.")


if __name__ == "__main__":
    main()

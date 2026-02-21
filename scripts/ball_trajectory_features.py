"""
Ball Trajectory Features

Instead of 207 skeleton features per frame, reduce to 3D ball trajectory.
Use PLS on the ball trajectory to extract compact features.
This removes player body proportions and shot form, keeping only ball dynamics.

Two approaches:
A) Ball trajectory PLS: estimate ball position (wrist+offset) across all 240 frames,
   apply PLS to the 720-dim trajectory to get compact features
B) Player-normalized ball: express ball trajectory relative to player body
   (remove body size, standing position) then PLS
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
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
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
    print("Loading data...")
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


def estimate_ball_trajectory(ts_3d, ts_hr, kp_index):
    """Estimate ball position at each frame from hand keypoints.
    Returns ball trajectory in hoop-relative coordinates."""
    rw = kp_index.get('right_wrist')
    ft_keys = ['right_first_finger_distal', 'right_second_finger_distal',
               'right_third_finger_distal', 'right_fourth_finger_distal',
               'right_fifth_finger_distal']
    ft_indices = [kp_index[k] for k in ft_keys if k in kp_index]

    if rw is None:
        return np.zeros((240, 3), dtype=np.float32)

    # Ball = wrist + 0.6 * (fingertip_center - wrist)
    wrist_hr = ts_hr[:, rw, :]  # (240, 3)
    if ft_indices:
        ft_center = np.nanmean(ts_hr[:, ft_indices, :], axis=1)  # (240, 3)
        ball = wrist_hr + 0.6 * (ft_center - wrist_hr)
    else:
        ball = wrist_hr.copy()

    # Smooth
    for ax in range(3):
        ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)

    return ball


def extract_ball_features(ball_traj, ts_hr, kp_index, frame):
    """Extract ball-centric features at a specific frame."""
    f = int(np.clip(frame, 0, 239))
    feats = []

    # Ball position at target frame
    feats.extend(ball_traj[f, :])

    # Ball velocity at target frame
    for ax in range(3):
        vel = np.gradient(ball_traj[:, ax], DT)
        feats.append(vel[f])

    # Ball acceleration at target frame
    for ax in range(3):
        vel = np.gradient(ball_traj[:, ax], DT)
        acc = np.gradient(vel, DT)
        feats.append(acc[f])

    # Ball trajectory summary stats (frames 80-180)
    for ax in range(3):
        series = ball_traj[80:180, ax]
        feats.append(np.nanmean(series))
        feats.append(np.nanstd(series))
        feats.append(np.nanmax(series))
        feats.append(np.nanmin(series))
        # Peak velocity
        vel = np.gradient(ball_traj[:, ax], DT)
        feats.append(np.nanmax(vel[80:180]))
        feats.append(np.nanmin(vel[80:180]))

    # Ball height profile (key frames)
    for kf in [100, 110, 120, 130, 140, 150, 160, 170]:
        feats.append(ball_traj[kf, 2])  # z height

    # Ball speed profile
    vel_3d = np.zeros_like(ball_traj)
    for ax in range(3):
        vel_3d[:, ax] = np.gradient(ball_traj[:, ax], DT)
    speed = np.linalg.norm(vel_3d, axis=1)
    for kf in [100, 110, 120, 130, 140, 150]:
        feats.append(speed[kf])

    # Ball-to-hoop distance over time
    hoop_hr = np.array([0.0, 0.0, HOOP_POS[2]])  # hoop in HR coords (forward=0 after transform)
    # Actually hoop distance in original coords
    for kf in [120, 140, 160]:
        d = np.linalg.norm(ball_traj[kf, :] - hoop_hr)
        feats.append(d)

    # Elevation angle (ball direction toward hoop)
    for kf in [120, 130, 140]:
        dx = -ball_traj[kf, 0]  # forward toward hoop
        dz = HOOP_POS[2] - ball_traj[kf, 2]
        feats.append(np.arctan2(dz, max(abs(dx), 1e-6)))

    return np.array(feats, dtype=np.float32)


def extract_all_ball_features(data, target):
    """Extract ball trajectory features for all shots."""
    frame = TARGET_FRAMES[target]
    n = len(data['pids'])
    kp_index = data['kp_index']

    all_feats = []
    all_trajs = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        ball_traj = estimate_ball_trajectory(ts_3d, ts_hr, kp_index)
        all_trajs.append(ball_traj)
        feats = extract_ball_features(ball_traj, ts_hr, kp_index, frame)
        all_feats.append(feats)

    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    trajs = np.array(all_trajs)  # (n, 240, 3)
    return X, trajs


def ball_trajectory_pls(trajs_train, y_train, pids_train, trajs_test, pids_test, max_nc=20):
    """Apply PLS to ball trajectories (720-dim) to get compact features."""
    unique_pids = sorted(np.unique(pids_train))

    # Flatten trajectories: (n, 240, 3) -> (n, 720)
    # Use frames 80-200 only (shooting motion): 120 frames x 3 = 360 dim
    X_tr_flat = trajs_train[:, 80:200, :].reshape(len(trajs_train), -1)
    X_te_flat = trajs_test[:, 80:200, :].reshape(len(trajs_test), -1)

    pls_train = np.zeros((len(pids_train), max_nc), dtype=np.float32)
    pls_test = np.zeros((len(pids_test), max_nc), dtype=np.float32)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        n_p = tr_mask.sum()

        scaler = StandardScaler()
        raw_tr = scaler.fit_transform(X_tr_flat[tr_mask])
        raw_te = scaler.transform(X_te_flat[te_mask])

        nc = min(max_nc, n_p - n_p // 5 - 1)
        nc = max(3, nc)

        best_nc, best_mse = 3, float('inf')
        for c in [3, 5, 8, 10, 15, 20]:
            if c > nc:
                break
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mses = []
            for tr_idx, val_idx in kf.split(raw_tr):
                pls = PLSRegression(n_components=c)
                pls.fit(raw_tr[tr_idx], y_train[tr_mask][tr_idx])
                pred = pls.predict(raw_tr[val_idx]).flatten()
                mses.append(np.mean((pred - y_train[tr_mask][val_idx]) ** 2))
            if np.mean(mses) < best_mse:
                best_mse = np.mean(mses)
                best_nc = c

        pls = PLSRegression(n_components=best_nc)
        pls.fit(raw_tr, y_train[tr_mask])
        pls_train[tr_mask, :best_nc] = pls.transform(raw_tr)
        pls_test[te_mask, :best_nc] = pls.transform(raw_te)

    return pls_train, pls_test


def locally_weighted_prediction(X_train, y_train, X_test, pids_train, pids_test,
                                bandwidth_quantile=0.5, alpha=10.0):
    """Per-example locally weighted Ridge regression."""
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]
        n_tr = len(X_tr)
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        D_tr_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        D_te_tr = cdist(X_te_s, X_tr_s, metric='euclidean') if len(X_te) > 0 else np.zeros((0, n_tr))

        all_dists = D_tr_tr[np.triu_indices(n_tr, k=1)]
        sigma = np.quantile(all_dists, bandwidth_quantile) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        for i in range(n_tr):
            dists = D_tr_tr[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        for j in range(len(X_te)):
            dists = D_te_tr[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

    return oof_preds, test_preds


def main():
    t0 = time.time()
    print("=" * 70)
    print("BALL TRAJECTORY FEATURES")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    results = {}
    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        y_target = y_scaled[target]
        y_raw = y_train[:, target_idx[target]]

        # Extract ball trajectory features
        print("  Extracting ball trajectory features...")
        X_train_ball, trajs_train = extract_all_ball_features(train_data, target)
        X_test_ball, trajs_test = extract_all_ball_features(test_data, target)
        print(f"  Ball features: {X_train_ball.shape[1]}")

        # Ball trajectory PLS
        print("  Computing ball trajectory PLS...")
        pls_train, pls_test = ball_trajectory_pls(
            trajs_train, y_raw, pids_train, trajs_test, pids_test, max_nc=20)
        print(f"  PLS features: {pls_train.shape[1]}")

        # Feature sets to test:
        feature_sets = {
            'ball_handcrafted': (X_train_ball, X_test_ball),
            'ball_pls': (pls_train, pls_test),
            'ball_combined': (np.hstack([X_train_ball, pls_train]),
                             np.hstack([X_test_ball, pls_test])),
        }

        target_results = {}
        for name, (X_tr, X_te) in feature_sets.items():
            print(f"\n  [{name}] ({X_tr.shape[1]} features)")

            oof, test = locally_weighted_prediction(
                X_tr, y_target, X_te, pids_train, pids_test,
                bandwidth_quantile=0.5, alpha=10.0)
            mse = np.mean((oof - y_target) ** 2)
            print(f"    LOO MSE: {mse:.6f}")

            # Correlation with Sub 1350
            sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")
            col = f'scaled_{target}'
            r = np.corrcoef(sub_1350[col].values, test)[0, 1]
            print(f"    Correlation with Sub 1350: r={r:.4f}")

            target_results[name] = {'oof': oof, 'test': test, 'mse': mse, 'r_1350': r}

        results[target] = target_results

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    sub_1350_baseline = {'angle': 0.002511, 'depth': 0.004510, 'left_right': 0.004209}

    for name in ['ball_handcrafted', 'ball_pls', 'ball_combined']:
        mses = [results[t][name]['mse'] for t in TARGETS]
        mean_mse = np.mean(mses)
        base_mean = np.mean(list(sub_1350_baseline.values()))
        delta = (mean_mse - base_mean) / base_mean * 100
        print(f"\n  {name}:")
        for t in TARGETS:
            r = results[t][name]['r_1350']
            m = results[t][name]['mse']
            b = sub_1350_baseline[t]
            d = (m - b) / b * 100
            print(f"    {t}: MSE={m:.6f} (vs baseline {b:.6f}, {d:+.1f}%), r_1350={r:.4f}")
        print(f"    MEAN: {mean_mse:.6f} (vs baseline {base_mean:.6f}, {delta:+.1f}%)")

    # Generate submissions
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")

    for name in ['ball_handcrafted', 'ball_pls', 'ball_combined']:
        # Standalone
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': results['angle'][name]['test'],
            'scaled_depth': results['depth'][name]['test'],
            'scaled_left_right': results['left_right'][name]['test'],
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {name} standalone")

        # Blend with Sub 784 (standard weights)
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = sub_784['scaled_angle']
        blended['scaled_depth'] = 0.70 * sub_784['scaled_depth'] + 0.30 * results['depth'][name]['test']
        blended['scaled_left_right'] = 0.50 * sub_784['scaled_left_right'] + 0.50 * results['left_right'][name]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {name} blend with Sub 784 (aw=0, dw=0.30, lw=0.50)")

        # 10% blend with Sub 1350
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for col, t in [('scaled_angle', 'angle'), ('scaled_depth', 'depth'), ('scaled_left_right', 'left_right')]:
            blended[col] = 0.90 * sub_1350[col] + 0.10 * results[t][name]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: 10% {name} + 90% Sub 1350")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

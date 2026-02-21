"""
Shot Deviation Model

Novel approach: instead of using absolute positions/velocities at specific frames,
compute how each shot DEVIATES from the player's average trajectory across all frames.
The deviation pattern captures what's UNIQUE about each shot - which should predict
the unique outcome.

Features:
- Per-joint temporal deviation profiles (max, mean, std of deviation)
- Frame of maximum deviation (timing of key differences)
- Directional deviations (vertical = angle, lateral = LR, forward = depth)
- Follow-through deviation (frames after release)
- Preparation deviation (frames before release)
- Symmetry of deviation (left vs right body side)

This produces genuinely different features from single-frame position extraction.
"""

import json
import fcntl
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])

PLAYER_FRAMES = {
    "angle": {1: 140, 2: 155, 3: 165, 4: 150, 5: 150},
    "depth": {1: 155, 2: 140, 3: 130, 4: 180, 5: 140},
    "left_right": {1: 185, 2: 130, 3: 140, 4: 180, 5: 145},
}

# Key joints for deviation analysis
KEY_JOINTS = [
    'right_wrist', 'right_elbow', 'right_shoulder',
    'left_wrist', 'left_elbow', 'left_shoulder',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'mid_hip', 'head', 'neck'
]


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


def safe_smooth(x, window=11, polyorder=3):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder)


def load_data():
    print("Loading data...", flush=True)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    kp_names = [col[:-2] for col in keypoint_cols if col.endswith("_x")]
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
        result = {
            'X_3d': X_3d, 'pids': np.array(pids),
            'ids': np.array(ids), 'kp_index': kp_index
        }
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


def extract_deviation_features(X_3d_all, pids_all, kp_index, n_train):
    """Extract shot deviation features for all shots."""
    print("Extracting deviation features...", flush=True)
    n_total = len(X_3d_all)
    unique_players = np.unique(pids_all)

    # Phase 1: Compute hoop-relative coordinates and smooth trajectories
    all_hoop_rel = []
    for i in range(n_total):
        hoop_rel = compute_hoop_transform(X_3d_all[i], kp_index)
        # Smooth each joint trajectory
        n_kp = hoop_rel.shape[1]
        for j in range(n_kp):
            for ax in range(3):
                hoop_rel[:, j, ax] = safe_smooth(hoop_rel[:, j, ax])
        all_hoop_rel.append(hoop_rel)
    all_hoop_rel = np.array(all_hoop_rel)  # (n_total, 240, n_kp, 3)

    # Phase 2: Compute per-player mean trajectories (using training data only)
    player_means = {}
    for pid in unique_players:
        train_mask = (pids_all[:n_train] == pid)
        if np.sum(train_mask) > 0:
            player_means[pid] = np.mean(all_hoop_rel[:n_train][train_mask], axis=0)

    # Phase 3: Extract deviation features
    key_joint_indices = [kp_index.get(j) for j in KEY_JOINTS if j in kp_index]
    key_joint_names = [j for j in KEY_JOINTS if j in kp_index]

    features = []
    for i in range(n_total):
        pid = pids_all[i]
        shot = all_hoop_rel[i]  # (240, n_kp, 3)
        mean_traj = player_means.get(pid, np.mean(all_hoop_rel[:n_train], axis=0))

        # Deviation from player mean
        deviation = shot - mean_traj  # (240, n_kp, 3)

        feats = []

        # Temporal windows
        prep_frames = slice(60, 120)    # preparation phase
        release_frames = slice(120, 170) # release phase
        follow_frames = slice(170, 220)  # follow-through

        for ji, jname in zip(key_joint_indices, key_joint_names):
            dev_j = deviation[:, ji, :]  # (240, 3)

            # Per-axis deviation statistics
            for ax, ax_name in enumerate(['forward', 'lateral', 'vertical']):
                dev_ax = dev_j[:, ax]

                # Global stats
                feats.append(np.nanmean(dev_ax[60:200]))   # mean deviation
                feats.append(np.nanstd(dev_ax[60:200]))    # std of deviation
                feats.append(np.nanmax(np.abs(dev_ax[60:200])))  # max absolute deviation

                # Phase-specific deviations
                feats.append(np.nanmean(dev_ax[prep_frames]))    # prep phase mean
                feats.append(np.nanmean(dev_ax[release_frames])) # release phase mean
                feats.append(np.nanmean(dev_ax[follow_frames]))  # follow-through mean

                # Frame of maximum deviation in release window
                rel_dev = np.abs(dev_ax[120:170])
                if np.any(~np.isnan(rel_dev)):
                    feats.append(120 + np.nanargmax(rel_dev))  # timing
                else:
                    feats.append(145)

            # Magnitude of 3D deviation
            dev_mag = np.linalg.norm(dev_j, axis=1)  # (240,)
            feats.append(np.nanmean(dev_mag[release_frames]))  # mean 3D deviation at release
            feats.append(np.nanmax(dev_mag[60:200]))  # peak 3D deviation

            # Rate of change of deviation (is the deviation growing or shrinking?)
            dev_rate = np.gradient(dev_mag)
            feats.append(np.nanmean(dev_rate[release_frames]))  # growing = shot diverging from average

        # Cross-joint deviation patterns
        # Symmetry: compare left vs right body side
        for pair in [('right_wrist', 'left_wrist'), ('right_elbow', 'left_elbow'),
                     ('right_shoulder', 'left_shoulder'), ('right_hip', 'left_hip')]:
            ri = kp_index.get(pair[0])
            li = kp_index.get(pair[1])
            if ri is not None and li is not None:
                r_dev_mag = np.linalg.norm(deviation[:, ri, :], axis=1)
                l_dev_mag = np.linalg.norm(deviation[:, li, :], axis=1)
                asymmetry = r_dev_mag - l_dev_mag
                feats.append(np.nanmean(asymmetry[release_frames]))
                feats.append(np.nanstd(asymmetry[release_frames]))

        # Global shot deviation score (how different is this shot from average?)
        overall_dev = np.nanmean(np.linalg.norm(
            deviation[:, key_joint_indices, :].reshape(240, -1), axis=1
        )[60:200])
        feats.append(overall_dev)

        # Deviation trajectory shape (is deviation concentrated or distributed?)
        dev_profile = np.nanmean(np.abs(deviation[:, key_joint_indices, :]), axis=(1, 2))
        if np.nanmax(dev_profile[60:200]) > 1e-8:
            normalized_profile = dev_profile[60:200] / np.nanmax(dev_profile[60:200])
            feats.append(np.nanargmax(normalized_profile))  # frame of peak deviation
            feats.append(np.nanstd(normalized_profile))     # deviation spread
        else:
            feats.append(70)
            feats.append(0.3)

        features.append(np.array(feats, dtype=np.float32))

    features = np.array(features)
    # Clean NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Deviation features shape: {features.shape}", flush=True)
    return features


def honest_loo(X, y, pids, target_name, n_pls=5, bw_quantile=0.45, alpha=10.0):
    """Honest LOO with PLS refit per fold."""
    from scipy.spatial.distance import cdist

    n = len(y)
    predictions = np.zeros(n)
    unique_players = np.unique(pids)

    for pid in unique_players:
        mask = pids == pid
        X_player = X[mask]
        y_player = y[mask]
        n_player = len(y_player)

        player_overrides = {}
        if pid == 1 and target_name == 'left_right':
            player_overrides = {'bw': 0.15}

        for i in range(n_player):
            # Leave one out
            train_mask = np.ones(n_player, dtype=bool)
            train_mask[i] = False
            X_train = X_player[train_mask]
            y_train = y_player[train_mask]
            X_test = X_player[i:i+1]

            # PLS on training only
            n_comp = min(n_pls, min(X_train.shape[0] - 1, X_train.shape[1]))
            if n_comp < 1:
                predictions[np.where(mask)[0][i]] = np.mean(y_train)
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X_train_s, y_train)
            X_train_pls = pls.transform(X_train_s)
            X_test_pls = pls.transform(X_test_s)

            # Kernel-weighted Ridge
            D = cdist(X_test_pls, X_train_pls, metric='euclidean')
            bw = player_overrides.get('bw', bw_quantile)
            sigma = np.quantile(D, bw) + 1e-8
            weights = np.exp(-0.5 * (D[0] / sigma) ** 2)
            weights /= weights.sum() + 1e-10

            W = np.diag(np.sqrt(weights))
            Xw = W @ X_train_pls
            yw = W @ y_train

            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(Xw, yw)
            predictions[np.where(mask)[0][i]] = ridge.predict(X_test_pls)[0]

    return predictions


def main():
    import time
    t0 = time.time()

    print("=" * 60, flush=True)
    print("SHOT DEVIATION MODEL", flush=True)
    print("Trajectory deviation features for diverse predictions", flush=True)
    print("=" * 60, flush=True)

    train_data, test_data = load_data()

    # Combine train and test for feature extraction
    X_3d_all = np.concatenate([train_data['X_3d'], test_data['X_3d']], axis=0)
    pids_all = np.concatenate([train_data['pids'], test_data['pids']])
    n_train = len(train_data['pids'])

    features = extract_deviation_features(X_3d_all, pids_all, train_data['kp_index'], n_train)

    X_train = features[:n_train]
    X_test = features[n_train:]

    # Scale targets to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    y_raw = train_data['y']
    scalers = {}
    y_scaled = np.zeros_like(y_raw)
    for t in range(3):
        sc = MinMaxScaler()
        y_scaled[:, t] = sc.fit_transform(y_raw[:, t:t+1]).ravel()
        scalers[t] = sc

    # Quick LOO screen (with leakage for fast check)
    print("\n" + "-" * 60, flush=True)
    print("QUICK SCREEN: Fast (leaky) LOO", flush=True)
    print("-" * 60, flush=True)
    from scipy.spatial.distance import cdist
    fast_loo_mses = []
    for t, target in enumerate(TARGETS):
        unique_players = np.unique(train_data['pids'])
        preds_fast = np.zeros(n_train)
        for pid in unique_players:
            mask = train_data['pids'] == pid
            Xp = X_train[mask]
            yp = y_scaled[mask, t]
            n_p = len(yp)

            n_comp = min(5, min(n_p - 2, Xp.shape[1]))
            scaler = StandardScaler()
            Xps = scaler.fit_transform(Xp)
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(Xps, yp)
            Xpls = pls.transform(Xps)

            D = cdist(Xpls, Xpls, metric='euclidean')
            for i in range(n_p):
                sigma = np.quantile(D[i], 0.45) + 1e-8
                w = np.exp(-0.5 * (D[i] / sigma) ** 2)
                w[i] = 0
                w /= w.sum() + 1e-10
                preds_fast[np.where(mask)[0][i]] = np.sum(w * yp)

        mse = np.mean((preds_fast - y_scaled[:, t]) ** 2)
        fast_loo_mses.append(mse)
        print(f"  {target}: fast LOO = {mse:.6f}", flush=True)
    print(f"  Mean fast LOO: {np.mean(fast_loo_mses):.6f}", flush=True)

    # Honest LOO
    print("\n" + "-" * 60, flush=True)
    print("HONEST LOO: PLS refit per held-out sample", flush=True)
    print("-" * 60, flush=True)
    honest_mses = []
    all_loo_preds = {}
    for t, target in enumerate(TARGETS):
        t1 = time.time()
        preds = honest_loo(X_train, y_scaled[:, t], train_data['pids'], target)
        mse = np.mean((preds - y_scaled[:, t]) ** 2)
        honest_mses.append(mse)
        all_loo_preds[target] = preds

        # Per-player breakdown
        print(f"\n  {target}: honest LOO = {mse:.6f} [{time.time()-t1:.1f}s]", flush=True)
        for pid in sorted(np.unique(train_data['pids'])):
            mask = train_data['pids'] == pid
            p_mse = np.mean((preds[mask] - y_scaled[mask, t]) ** 2)
            print(f"    P{pid}: {p_mse:.6f}", flush=True)

    mean_honest = np.mean(honest_mses)
    print(f"\n  Mean honest LOO: {mean_honest:.6f}", flush=True)

    # Kill criteria
    if mean_honest > 0.015:
        print(f"\n  KILLED: honest LOO {mean_honest:.6f} > 0.015", flush=True)
        return

    # Generate test predictions
    print("\n" + "-" * 60, flush=True)
    print("GENERATING TEST PREDICTIONS", flush=True)
    print("-" * 60, flush=True)

    test_preds = np.zeros((len(test_data['ids']), 3))
    for t, target in enumerate(TARGETS):
        unique_players = np.unique(train_data['pids'])
        for pid in unique_players:
            train_mask = train_data['pids'] == pid
            test_mask = test_data['pids'] == pid
            if not np.any(test_mask):
                continue

            X_tr = X_train[train_mask]
            y_tr = y_scaled[train_mask, t]
            X_te = X_test[test_mask]

            n_comp = min(5, min(X_tr.shape[0] - 1, X_tr.shape[1]))
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X_tr_s, y_tr)
            X_tr_pls = pls.transform(X_tr_s)
            X_te_pls = pls.transform(X_te_s)

            bw_q = 0.15 if (pid == 1 and target == 'left_right') else 0.45

            for i in range(len(X_te_pls)):
                D = cdist(X_te_pls[i:i+1], X_tr_pls, metric='euclidean')
                sigma = np.quantile(D, bw_q) + 1e-8
                w = np.exp(-0.5 * (D[0] / sigma) ** 2)
                w /= w.sum() + 1e-10

                W = np.diag(np.sqrt(w))
                Xw = W @ X_tr_pls
                yw = W @ y_tr
                ridge = Ridge(alpha=10.0, fit_intercept=True)
                ridge.fit(Xw, yw)
                test_preds[np.where(test_mask)[0][i], t] = ridge.predict(X_te_pls[i:i+1])[0]

    test_preds = np.clip(test_preds, 0, 1)

    # Diversity check
    print("\n" + "-" * 60, flush=True)
    print("DIVERSITY CHECK: Correlation with Sub 2716", flush=True)
    print("-" * 60, flush=True)
    anchor = pd.read_csv(SUBMISSION_DIR / "submission_2716.csv")
    corrs = []
    for t, target in enumerate(TARGETS):
        col = f"scaled_{target}"
        r = np.corrcoef(test_preds[:, t], anchor[col].values)[0, 1]
        corrs.append(r)
        print(f"  {target}: r = {r:.4f}", flush=True)
    print(f"  Mean correlation: {np.mean(corrs):.4f}", flush=True)

    if min(corrs) > 0.90:
        print(f"\n  WARNING: Low diversity (min r = {min(corrs):.4f})", flush=True)
    else:
        print(f"\n  PROCEED: At least one target has r < 0.90 (diversity exists)", flush=True)

    # Save submissions
    print("\n" + "=" * 60, flush=True)
    print("GENERATING SUBMISSIONS", flush=True)
    print("=" * 60, flush=True)

    # Standalone
    sub_num = get_next_submission_number()
    sub_df = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': test_preds[:, 0],
        'scaled_depth': test_preds[:, 1],
        'scaled_left_right': test_preds[:, 2]
    })
    sub_df.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"Sub {sub_num}: Deviation model standalone (honest LOO {mean_honest:.6f})", flush=True)

    # Blends with Sub 2716
    anchor_preds = anchor[['scaled_angle', 'scaled_depth', 'scaled_left_right']].values
    for w in [0.02, 0.05]:
        blend = (1 - w) * anchor_preds + w * test_preds
        blend = np.clip(blend, 0, 1)
        sn = get_next_submission_number()
        blend_df = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': blend[:, 0],
            'scaled_depth': blend[:, 1],
            'scaled_left_right': blend[:, 2]
        })
        blend_df.to_csv(SUBMISSION_DIR / f"submission_{sn}.csv", index=False)
        print(f"Sub {sn}: {int(w*100)}% deviation + {int((1-w)*100)}% Sub 2716", flush=True)

    # Per-target blends (best diversity target only)
    best_div_target = TARGETS[np.argmin(corrs)]
    best_div_idx = np.argmin(corrs)
    for w in [0.02, 0.05]:
        blend = anchor_preds.copy()
        blend[:, best_div_idx] = (1 - w) * anchor_preds[:, best_div_idx] + w * test_preds[:, best_div_idx]
        blend = np.clip(blend, 0, 1)
        sn = get_next_submission_number()
        blend_df = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': blend[:, 0],
            'scaled_depth': blend[:, 1],
            'scaled_left_right': blend[:, 2]
        })
        blend_df.to_csv(SUBMISSION_DIR / f"submission_{sn}.csv", index=False)
        print(f"Sub {sn}: {int(w*100)}% deviation ({best_div_target} only) + Sub 2716", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Total runtime: {time.time()-t0:.1f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

"""
Temporal Stability of Player Control Channels

Tests whether player-specific channels are temporally stable "motor signatures"
or statistical artifacts. Method:
1. Split each player's shots into first half / second half by shot_id
2. Compute channel correlations (|r| for each feature) on first half
3. Test whether those same features predict second half outcomes
4. If channels are stable (rank correlation of |r| vectors between halves is high),
   this is a genuine motor signature.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr

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


def load_data():
    print("Loading data...")
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
    shot_ids = []
    targets_raw = []

    for idx, row in train_df.iterrows():
        for col_i, col in enumerate(keypoint_cols):
            arr = parse_array_string(row[col])
            X_3d[idx, :, col_i // 3, col_i % 3] = arr
        pids.append(row['participant_id'])
        shot_ids.append(row['shot_id'])
        targets_raw.append([row['angle'], row['depth'], row['left_right']])
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(train_df)}")

    return X_3d, np.array(pids), np.array(shot_ids), np.array(targets_raw, dtype=np.float32), kp_names, kp_index


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


def extract_features_for_frame(ts_3d, ts_hr, kp_index, frame):
    f = int(np.clip(frame, 1, 238))
    feats = []
    feat_names = []

    for jname in KEY_JOINTS:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            for ax in ['x', 'y', 'z']:
                feat_names.append(f"{jname}_pos_{ax}")
            for ax in ['x', 'y', 'z']:
                feat_names.append(f"{jname}_vel_{ax}")
            continue
        pos = np.nan_to_num(ts_hr[f, idx, :], nan=0.0)
        feats.extend(pos.tolist())
        for ax in ['x', 'y', 'z']:
            feat_names.append(f"{jname}_pos_{ax}")
        pos_prev = np.nan_to_num(ts_hr[f - 1, idx, :], nan=0.0)
        pos_next = np.nan_to_num(ts_hr[min(f + 1, 239), idx, :], nan=0.0)
        vel = (pos_next - pos_prev) / (2 * DT)
        feats.extend(vel.tolist())
        for ax in ['x', 'y', 'z']:
            feat_names.append(f"{jname}_vel_{ax}")

    # Joint angles
    angle_defs = [
        ('elbow_angle', 'right_shoulder', 'right_elbow', 'right_wrist'),
        ('shoulder_angle', 'right_hip', 'right_shoulder', 'right_elbow'),
        ('knee_angle_r', 'right_hip', 'right_knee', 'left_knee'),
    ]
    for aname, j1, j2, j3 in angle_defs:
        i1, i2, i3 = kp_index.get(j1), kp_index.get(j2), kp_index.get(j3)
        if i1 is not None and i2 is not None and i3 is not None:
            v1 = np.nan_to_num(ts_hr[f, i1, :] - ts_hr[f, i2, :], nan=0.0)
            v2 = np.nan_to_num(ts_hr[f, i3, :] - ts_hr[f, i2, :], nan=0.0)
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            feats.append(float(np.arccos(np.clip(cos_a, -1, 1))))
        else:
            feats.append(0.0)
        feat_names.append(aname)

    rw = kp_index.get('right_wrist')
    if rw is not None:
        wrist_pos = np.nan_to_num(ts_hr[f, rw, :], nan=0.0)
        feats.append(float(np.linalg.norm(HOOP_POS - wrist_pos)))
    else:
        feats.append(0.0)
    feat_names.append('wrist_hoop_dist')

    return np.array(feats, dtype=np.float32), feat_names


def run_temporal_stability():
    X_3d, pids, shot_ids, y_raw, kp_names, kp_index = load_data()
    n_shots = len(pids)
    players = sorted(np.unique(pids))

    # Scale targets
    scaled_targets = np.zeros((n_shots, 3), dtype=np.float32)
    for ti, tname in enumerate(TARGETS):
        scaler = joblib.load(DATA_DIR / f"scaler_{tname}.pkl")
        scaled_targets[:, ti] = scaler.transform(y_raw[:, ti].reshape(-1, 1)).ravel()

    print(f"\nData: {n_shots} shots, {len(players)} players")

    # Extract features for all frames needed
    features_by_target = {}
    feat_names = None
    for tname in TARGETS:
        frame = TARGET_FRAMES[tname]
        all_feats = []
        for i in range(n_shots):
            ts_hr = compute_hoop_transform(X_3d[i], kp_index)
            feats, names = extract_features_for_frame(X_3d[i], ts_hr, kp_index, frame)
            all_feats.append(feats)
            if feat_names is None:
                feat_names = names
        features_by_target[tname] = np.array(all_feats)

    n_feats = len(feat_names)
    print(f"Features: {n_feats}")

    # Run temporal stability test
    print(f"\n{'='*70}")
    print("TEMPORAL STABILITY OF PLAYER CHANNELS")
    print(f"{'='*70}")

    all_stability_rho = []
    all_top10_overlap = []
    all_predictive_r = []

    for ti, tname in enumerate(TARGETS):
        X = features_by_target[tname]
        print(f"\n--- {tname} (frame {TARGET_FRAMES[tname]}) ---")

        for pid in players:
            mask = pids == pid
            X_p = X[mask]
            y_p = scaled_targets[mask, ti]
            sids_p = shot_ids[mask]
            n_p = mask.sum()

            if n_p < 20:
                continue

            # Split by shot_id (temporal ordering)
            sorted_idx = np.argsort(sids_p)
            half = n_p // 2
            first_half = sorted_idx[:half]
            second_half = sorted_idx[half:]

            X_h1, y_h1 = X_p[first_half], y_p[first_half]
            X_h2, y_h2 = X_p[second_half], y_p[second_half]

            # Compute |r| on each half
            abs_r_h1 = np.zeros(n_feats)
            abs_r_h2 = np.zeros(n_feats)
            for fi in range(n_feats):
                if np.std(X_h1[:, fi]) > 1e-10:
                    r, _ = pearsonr(X_h1[:, fi], y_h1)
                    abs_r_h1[fi] = abs(r)
                if np.std(X_h2[:, fi]) > 1e-10:
                    r, _ = pearsonr(X_h2[:, fi], y_h2)
                    abs_r_h2[fi] = abs(r)

            # Test 1: Rank correlation of |r| vectors between halves
            valid = (abs_r_h1 > 0) | (abs_r_h2 > 0)
            if valid.sum() < 10:
                continue
            rho_stability, p_stab = spearmanr(abs_r_h1[valid], abs_r_h2[valid])

            # Test 2: Top-10 feature overlap
            top10_h1 = set(np.argsort(abs_r_h1)[-10:])
            top10_h2 = set(np.argsort(abs_r_h2)[-10:])
            overlap = len(top10_h1 & top10_h2)

            # Test 3: Predictive transfer
            # Use top-10 features from h1 to predict h2 outcomes
            top10_h1_idx = np.argsort(abs_r_h1)[-10:]
            # Simple: mean |r| of h1-selected features when evaluated on h2
            predictive_r = abs_r_h2[top10_h1_idx].mean()
            # Compare to: mean |r| of h2's own top-10
            own_top_r = abs_r_h2[np.argsort(abs_r_h2)[-10:]].mean()
            # And random baseline: mean |r| of random 10 features on h2
            random_r = abs_r_h2[valid].mean()

            all_stability_rho.append(rho_stability)
            all_top10_overlap.append(overlap)
            all_predictive_r.append(predictive_r)

            print(f"\n  Player {pid} (n={n_p}, split {len(first_half)}/{len(second_half)})")
            print(f"    Stability rho (rank corr of |r| vectors): {rho_stability:.3f} (p={p_stab:.4f})")
            print(f"    Top-10 feature overlap: {overlap}/10")
            print(f"    Predictive transfer: h1-selected mean |r| on h2 = {predictive_r:.3f}")
            print(f"      vs h2's own top-10 mean |r| = {own_top_r:.3f}")
            print(f"      vs random baseline mean |r| = {random_r:.3f}")
            print(f"      Transfer ratio: {predictive_r/random_r:.2f}x random" if random_r > 0 else "")

            # Show which features are stable across halves
            stable_top = top10_h1 & top10_h2
            if stable_top:
                print(f"    Stable top features (in both halves):")
                for fi in stable_top:
                    print(f"      {feat_names[fi]:30s}: h1 |r|={abs_r_h1[fi]:.3f}, h2 |r|={abs_r_h2[fi]:.3f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  N player-target pairs tested: {len(all_stability_rho)}")
    print(f"\n  Stability rho (rank corr between halves):")
    print(f"    Mean: {np.mean(all_stability_rho):.3f}")
    print(f"    Median: {np.median(all_stability_rho):.3f}")
    print(f"    Positive: {sum(1 for x in all_stability_rho if x > 0)}/{len(all_stability_rho)}")
    print(f"    >0.3: {sum(1 for x in all_stability_rho if x > 0.3)}/{len(all_stability_rho)}")
    print(f"    >0.5: {sum(1 for x in all_stability_rho if x > 0.5)}/{len(all_stability_rho)}")

    print(f"\n  Top-10 overlap:")
    print(f"    Mean: {np.mean(all_top10_overlap):.1f}/10")
    print(f"    Median: {np.median(all_top10_overlap):.1f}/10")
    print(f"    By chance (hypergeometric): ~1.3/10")

    print(f"\n  Predictive transfer (h1-selected mean |r| on h2):")
    print(f"    Mean: {np.mean(all_predictive_r):.3f}")

    # Permutation test: is the mean stability rho significantly above zero?
    # Bootstrap 95% CI
    rhos = np.array(all_stability_rho)
    boot_means = []
    for _ in range(10000):
        idx = np.random.choice(len(rhos), len(rhos), replace=True)
        boot_means.append(rhos[idx].mean())
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"\n  Bootstrap 95% CI for mean stability rho: [{ci_low:.3f}, {ci_high:.3f}]")
    if ci_low > 0:
        print(f"    SIGNIFICANT: channels are temporally stable (CI excludes zero)")
    else:
        print(f"    NOT significant at 95% level (CI includes zero)")

    return all_stability_rho, all_top10_overlap


if __name__ == "__main__":
    run_temporal_stability()

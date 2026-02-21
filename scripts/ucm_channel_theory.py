"""
UCM (Uncontrolled Manifold) Theory Analysis

Tests whether movement variability decomposition matches our channel discovery:
- HIGH |r| features (task-relevant, ORT) should have LOWER variance (tightly controlled)
- LOW |r| features (task-irrelevant, UCM) should have HIGHER variance (uncontrolled)

UCM Index = (V_UCM - V_ORT) / V_total
Positive = motor system stabilizes outcome-relevant dimensions

Reference: Scholz & Schoner (1999), "The Uncontrolled Manifold Concept"
"""

import json
import numpy as np
import pandas as pd
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
FEET_TO_METERS = 0.3048

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
    targets_raw = []

    for idx, row in train_df.iterrows():
        for col_i, col in enumerate(keypoint_cols):
            arr = parse_array_string(row[col])
            X_3d[idx, :, col_i // 3, col_i % 3] = arr
        pids.append(row['participant_id'])
        targets_raw.append([row['angle'], row['depth'], row['left_right']])
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(train_df)}")

    return X_3d, np.array(pids), np.array(targets_raw, dtype=np.float32), kp_names, kp_index


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
    """Extract hoop-relative position + velocity features at a given frame."""
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

        pos = ts_hr[f, idx, :]
        pos = np.nan_to_num(pos, nan=0.0)
        feats.extend(pos.tolist())
        for ax in ['x', 'y', 'z']:
            feat_names.append(f"{jname}_pos_{ax}")

        # Velocity via finite difference
        pos_prev = np.nan_to_num(ts_hr[f - 1, idx, :], nan=0.0)
        pos_next = np.nan_to_num(ts_hr[min(f + 1, 239), idx, :], nan=0.0)
        vel = (pos_next - pos_prev) / (2 * DT)
        feats.extend(vel.tolist())
        for ax in ['x', 'y', 'z']:
            feat_names.append(f"{jname}_vel_{ax}")

    # Joint angles (elbow, shoulder, knee)
    angle_defs = [
        ('elbow_angle', 'right_shoulder', 'right_elbow', 'right_wrist'),
        ('shoulder_angle', 'right_hip', 'right_shoulder', 'right_elbow'),
        ('knee_angle_r', 'right_hip', 'right_knee', 'left_knee'),  # approximate
    ]
    for aname, j1, j2, j3 in angle_defs:
        i1 = kp_index.get(j1)
        i2 = kp_index.get(j2)
        i3 = kp_index.get(j3)
        if i1 is not None and i2 is not None and i3 is not None:
            v1 = np.nan_to_num(ts_hr[f, i1, :] - ts_hr[f, i2, :], nan=0.0)
            v2 = np.nan_to_num(ts_hr[f, i3, :] - ts_hr[f, i2, :], nan=0.0)
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            feats.append(float(np.arccos(np.clip(cos_a, -1, 1))))
        else:
            feats.append(0.0)
        feat_names.append(aname)

    # Distances: wrist-to-hoop, elbow height, shoulder height
    rw = kp_index.get('right_wrist')
    if rw is not None:
        wrist_pos = np.nan_to_num(ts_hr[f, rw, :], nan=0.0)
        hoop_rel = HOOP_POS - wrist_pos  # approximate
        feats.append(float(np.linalg.norm(hoop_rel)))
        feat_names.append('wrist_hoop_dist')
    else:
        feats.append(0.0)
        feat_names.append('wrist_hoop_dist')

    return np.array(feats, dtype=np.float32), feat_names


def run_ucm_analysis():
    X_3d, pids, y_raw, kp_names, kp_index = load_data()
    n_shots = len(pids)
    players = sorted(np.unique(pids))

    # Scale targets to [0,1]
    import joblib
    scaled_targets = np.zeros((n_shots, 3), dtype=np.float32)
    for ti, tname in enumerate(TARGETS):
        scaler = joblib.load(DATA_DIR / f"scaler_{tname}.pkl")
        scaled_targets[:, ti] = scaler.transform(y_raw[:, ti].reshape(-1, 1)).ravel()

    print(f"\nData loaded: {n_shots} shots, {len(players)} players")
    print(f"Players: {players}")

    # Extract features per target frame
    # We'll do per-target analysis since different targets use different frames
    results = {}

    for ti, tname in enumerate(TARGETS):
        frame = TARGET_FRAMES[tname]
        print(f"\n{'='*60}")
        print(f"TARGET: {tname} (frame {frame})")
        print(f"{'='*60}")

        # Extract features for all shots at this target's frame
        all_feats = []
        feat_names = None
        for i in range(n_shots):
            ts_hr = compute_hoop_transform(X_3d[i], kp_index)
            feats, names = extract_features_for_frame(X_3d[i], ts_hr, kp_index, frame)
            all_feats.append(feats)
            if feat_names is None:
                feat_names = names
        X = np.array(all_feats)
        n_feats = X.shape[1]
        print(f"Features extracted: {n_feats} features x {n_shots} shots")

        target_results = {}

        for pid in players:
            mask = pids == pid
            X_p = X[mask]
            y_p = scaled_targets[mask, ti]
            n_p = mask.sum()

            if n_p < 10:
                continue

            # Compute |r| for each feature
            abs_r = np.zeros(n_feats)
            p_vals = np.zeros(n_feats)
            for fi in range(n_feats):
                col = X_p[:, fi]
                if np.std(col) < 1e-10:
                    abs_r[fi] = 0.0
                    p_vals[fi] = 1.0
                    continue
                r, p = pearsonr(col, y_p)
                abs_r[fi] = abs(r)
                p_vals[fi] = p

            # Compute variance of each feature (across shots)
            feat_var = np.var(X_p, axis=0)

            # Standardize variance by feature mean^2 to make comparable (coefficient of variation^2)
            feat_mean = np.mean(X_p, axis=0)
            # Use relative variance (CV^2) where possible, raw variance otherwise
            cv2 = np.zeros(n_feats)
            for fi in range(n_feats):
                if abs(feat_mean[fi]) > 1e-6:
                    cv2[fi] = feat_var[fi] / (feat_mean[fi] ** 2)
                else:
                    cv2[fi] = feat_var[fi]

            # Split into HIGH |r| (top 20%) and LOW |r| (bottom 20%)
            # Only use features with nonzero variance
            valid = feat_var > 1e-12
            n_valid = valid.sum()
            if n_valid < 10:
                continue

            threshold_high = np.percentile(abs_r[valid], 80)
            threshold_low = np.percentile(abs_r[valid], 20)

            high_r_mask = valid & (abs_r >= threshold_high)
            low_r_mask = valid & (abs_r <= threshold_low)

            n_high = high_r_mask.sum()
            n_low = low_r_mask.sum()

            if n_high < 2 or n_low < 2:
                continue

            # Raw variance comparison
            var_high = feat_var[high_r_mask].mean()  # ORT (task-relevant)
            var_low = feat_var[low_r_mask].mean()     # UCM (task-irrelevant)
            var_total = feat_var[valid].mean()

            # CV^2 comparison (scale-invariant)
            cv2_high = cv2[high_r_mask].mean()
            cv2_low = cv2[low_r_mask].mean()
            cv2_total = cv2[valid].mean()

            # UCM Index using raw variance
            ucm_index_raw = (var_low - var_high) / (var_total + 1e-15)

            # UCM Index using CV^2 (scale-invariant)
            ucm_index_cv2 = (cv2_low - cv2_high) / (cv2_total + 1e-15)

            # Rank-based analysis: Spearman correlation between |r| and variance
            # If UCM holds: higher |r| -> lower variance -> negative correlation
            valid_idx = np.where(valid)[0]
            rho_r_var, p_r_var = spearmanr(abs_r[valid], feat_var[valid])

            # Additional: per-feature analysis for top features
            top_feats_idx = np.argsort(abs_r)[::-1][:10]
            bot_feats_idx = np.argsort(abs_r)[:10]

            print(f"\n  Player {pid} (n={n_p})")
            print(f"    HIGH |r| feats (ORT, n={n_high}): mean |r|={abs_r[high_r_mask].mean():.3f}, mean var={var_high:.6f}, mean CV2={cv2_high:.4f}")
            print(f"    LOW  |r| feats (UCM, n={n_low}):  mean |r|={abs_r[low_r_mask].mean():.3f}, mean var={var_low:.6f}, mean CV2={cv2_low:.4f}")
            print(f"    UCM Index (raw): {ucm_index_raw:.4f} {'POSITIVE (UCM supported)' if ucm_index_raw > 0 else 'NEGATIVE (UCM violated)'}")
            print(f"    UCM Index (CV2): {ucm_index_cv2:.4f} {'POSITIVE (UCM supported)' if ucm_index_cv2 > 0 else 'NEGATIVE (UCM violated)'}")
            print(f"    Spearman(|r|, var): rho={rho_r_var:.3f}, p={p_r_var:.4f} (expect NEGATIVE if UCM holds)")

            print(f"    Top 5 HIGH |r| features:")
            for rank, fi in enumerate(top_feats_idx[:5]):
                print(f"      {feat_names[fi]:30s}: |r|={abs_r[fi]:.3f}, var={feat_var[fi]:.6f}")

            print(f"    Top 5 LOW |r| features:")
            for rank, fi in enumerate(bot_feats_idx[:5]):
                if feat_var[fi] > 1e-12:
                    print(f"      {feat_names[fi]:30s}: |r|={abs_r[fi]:.3f}, var={feat_var[fi]:.6f}")

            target_results[pid] = {
                'n_shots': int(n_p),
                'n_high_r': int(n_high),
                'n_low_r': int(n_low),
                'mean_r_high': float(abs_r[high_r_mask].mean()),
                'mean_r_low': float(abs_r[low_r_mask].mean()),
                'var_high_raw': float(var_high),
                'var_low_raw': float(var_low),
                'var_total_raw': float(var_total),
                'cv2_high': float(cv2_high),
                'cv2_low': float(cv2_low),
                'cv2_total': float(cv2_total),
                'ucm_index_raw': float(ucm_index_raw),
                'ucm_index_cv2': float(ucm_index_cv2),
                'spearman_r_var_rho': float(rho_r_var),
                'spearman_r_var_p': float(p_r_var),
                'top5_high_r': [(feat_names[fi], float(abs_r[fi]), float(feat_var[fi])) for fi in top_feats_idx[:5]],
                'top5_low_r': [(feat_names[fi], float(abs_r[fi]), float(feat_var[fi])) for fi in bot_feats_idx[:5] if feat_var[fi] > 1e-12],
            }

        results[tname] = target_results

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY: UCM Theory Test Results")
    print(f"{'='*60}")

    all_ucm_raw = []
    all_ucm_cv2 = []
    all_spearman = []

    for tname in TARGETS:
        print(f"\n  {tname}:")
        for pid in sorted(results[tname].keys()):
            r = results[tname][pid]
            all_ucm_raw.append(r['ucm_index_raw'])
            all_ucm_cv2.append(r['ucm_index_cv2'])
            all_spearman.append(r['spearman_r_var_rho'])
            sign_raw = "+" if r['ucm_index_raw'] > 0 else "-"
            sign_cv2 = "+" if r['ucm_index_cv2'] > 0 else "-"
            print(f"    P{pid}: UCM_raw={r['ucm_index_raw']:+.4f} [{sign_raw}], UCM_cv2={r['ucm_index_cv2']:+.4f} [{sign_cv2}], Spearman(|r|,var)={r['spearman_r_var_rho']:+.3f} (p={r['spearman_r_var_p']:.4f})")

    n_positive_raw = sum(1 for x in all_ucm_raw if x > 0)
    n_positive_cv2 = sum(1 for x in all_ucm_cv2 if x > 0)
    n_negative_spearman = sum(1 for x in all_spearman if x < 0)
    n_total = len(all_ucm_raw)

    print(f"\n  OVERALL:")
    print(f"    UCM Index (raw) positive: {n_positive_raw}/{n_total} ({100*n_positive_raw/n_total:.0f}%)")
    print(f"    UCM Index (CV2) positive: {n_positive_cv2}/{n_total} ({100*n_positive_cv2/n_total:.0f}%)")
    print(f"    Spearman(|r|,var) negative: {n_negative_spearman}/{n_total} ({100*n_negative_spearman/n_total:.0f}%)")
    print(f"    Mean UCM Index (raw): {np.mean(all_ucm_raw):.4f}")
    print(f"    Mean UCM Index (CV2): {np.mean(all_ucm_cv2):.4f}")
    print(f"    Mean Spearman: {np.mean(all_spearman):.3f}")

    # Additional: Cross-player UCM consistency
    # For each target, check if the SAME features are high-|r| across players
    print(f"\n{'='*60}")
    print("CROSS-PLAYER CHANNEL CONSISTENCY")
    print(f"{'='*60}")

    for ti, tname in enumerate(TARGETS):
        frame = TARGET_FRAMES[tname]
        print(f"\n  {tname} (frame {frame}):")

        # Recompute features once more to get per-player rankings
        all_feats = []
        for i in range(n_shots):
            ts_hr = compute_hoop_transform(X_3d[i], kp_index)
            feats, names = extract_features_for_frame(X_3d[i], ts_hr, kp_index, frame)
            all_feats.append(feats)
        X = np.array(all_feats)
        n_f = X.shape[1]

        # For each player, rank features by |r|
        player_rankings = {}
        for pid in players:
            mask = pids == pid
            X_p = X[mask]
            y_p = scaled_targets[mask, ti]
            abs_r = np.zeros(n_f)
            for fi in range(n_f):
                col = X_p[:, fi]
                if np.std(col) < 1e-10:
                    continue
                r, _ = pearsonr(col, y_p)
                abs_r[fi] = abs(r)
            player_rankings[pid] = abs_r

        # Compute pairwise Spearman correlation of |r| vectors between players
        from itertools import combinations
        player_list = sorted(player_rankings.keys())
        if len(player_list) >= 2:
            pair_corrs = []
            for p1, p2 in combinations(player_list, 2):
                rho, _ = spearmanr(player_rankings[p1], player_rankings[p2])
                pair_corrs.append((p1, p2, rho))
                print(f"    P{p1} vs P{p2}: rank corr of |r| vectors = {rho:.3f}")
            mean_pair = np.mean([x[2] for x in pair_corrs])
            print(f"    Mean cross-player rank correlation: {mean_pair:.3f}")
            print(f"    {'HIGH' if mean_pair > 0.5 else 'MODERATE' if mean_pair > 0.3 else 'LOW'}: "
                  f"{'Same features matter for all players' if mean_pair > 0.5 else 'Some player-specific channels' if mean_pair > 0.3 else 'Highly player-specific channels'}")

    return results


if __name__ == "__main__":
    results = run_ucm_analysis()

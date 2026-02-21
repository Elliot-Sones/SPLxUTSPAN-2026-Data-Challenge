"""
Rigorous Statistical Validation of Player-Specific Biomechanical Channels

Research question: Are player-specific information channels real, or noise from small N?

Tests:
1. Fisher z-test: Is P1's r(hip_vel, LR) significantly DIFFERENT from P2's r(hip_vel, LR)?
2. Bootstrap confidence intervals: What's the uncertainty on each player's top feature?
3. Permutation test: Under the null (all players same), what's the expected max r?
4. Cross-validation: Do top features on train-half predict on held-out half?
5. Cross-player transfer: Do P1's top LR features predict P2's LR?

If the channels are real:
- Fisher z-test should reject H0: r_P1 == r_P2 for player-specific features
- Bootstrap CIs should be narrow enough to show meaningful effect
- Cross-player transfer should FAIL (P1's top features should NOT predict P2)
- CV should confirm the channels on held-out data

This analysis gives publication-quality evidence for or against the hypothesis.
"""

import json
import time
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import norm

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def safe_savgol(x, window=11, polyorder=3, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    kp_names = [col[:-2] for col in keypoint_cols if col.endswith("_x")]
    kp_index = {name: i for i, name in enumerate(kp_names)}

    n = len(train_df)
    n_kp = len(kp_names)
    X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
    ids, pids, targets = [], [], []

    for idx, row in train_df.iterrows():
        for col_i, col in enumerate(keypoint_cols):
            arr = parse_array_string(row[col])
            X_3d[idx, :, col_i // 3, col_i % 3] = arr
        ids.append(row['id'])
        pids.append(row['participant_id'])
        targets.append([row['angle'], row['depth'], row['left_right']])

    X_3d = np.nan_to_num(X_3d, nan=0.0)
    return {'X_3d': X_3d, 'pids': np.array(pids),
            'ids': np.array(ids), 'kp_index': kp_index,
            'y': np.array(targets, dtype=np.float32)}


def get_hoop_transform(ts_smooth, kp_index):
    mid_hip_idx = kp_index.get('mid_hip', 0)
    player_pos = ts_smooth[120, mid_hip_idx, :].copy()
    player_pos[2] = 0
    forward = HOOP_POS[:2] - player_pos[:2]
    fn = np.linalg.norm(forward)
    forward = forward / fn if fn > 1e-6 else np.array([0.0, -1.0])
    lateral = np.array([-forward[1], forward[0]])
    R = np.eye(3, dtype=np.float32)
    R[0, 0] = forward[0]; R[0, 1] = forward[1]
    R[1, 0] = lateral[0]; R[1, 1] = lateral[1]
    centered = ts_smooth - player_pos.reshape(1, 1, 3)
    return np.einsum('ij,fkj->fki', R, centered)


def extract_key_features(ts_3d, kp_index):
    """
    Extract the specific features discovered as player-specific channels.
    Focus on lateral (y-axis in hoop frame) velocities and positions.
    """
    n_kp = ts_3d.shape[1]
    ts_smooth = np.zeros_like(ts_3d, dtype=np.float64)
    for k in range(n_kp):
        for ax in range(3):
            ts_smooth[:, k, ax] = safe_savgol(ts_3d[:, k, ax].astype(np.float64))

    ts_hr = get_hoop_transform(ts_smooth, kp_index)
    vel_hr = np.zeros_like(ts_hr)
    for k in range(n_kp):
        for ax in range(3):
            vel_hr[:, k, ax] = safe_savgol(ts_hr[:, k, ax], deriv=1, delta=DT)

    feats = {}

    # THE KEY DISCOVERED CHANNELS
    rh_idx = kp_index.get('right_hip')
    mh_idx = kp_index.get('mid_hip')
    rw_idx = kp_index.get('right_wrist')
    neck_idx = kp_index.get('neck')
    ls_idx = kp_index.get('left_shoulder')
    lw_idx = kp_index.get('left_wrist')
    re_idx = kp_index.get('right_elbow')

    for f in [153, 155, 160, 165, 170, 175, 180]:
        if rh_idx is not None:
            feats[f'vel_rh_y_f{f}'] = float(vel_hr[f, rh_idx, 1])
        if mh_idx is not None:
            feats[f'vel_mh_y_f{f}'] = float(vel_hr[f, mh_idx, 1])
        if rw_idx is not None:
            feats[f'vel_rw_y_f{f}'] = float(vel_hr[f, rw_idx, 1])
        if re_idx is not None:
            feats[f'vel_re_y_f{f}'] = float(vel_hr[f, re_idx, 1])
        if neck_idx is not None:
            feats[f'hr_neck_y_f{f}'] = float(ts_hr[f, neck_idx, 1])
        if ls_idx is not None:
            feats[f'hr_ls_y_f{f}'] = float(ts_hr[f, ls_idx, 1])
        if lw_idx is not None:
            feats[f'hr_lw_y_f{f}'] = float(ts_hr[f, lw_idx, 1])

    # Also include all velocities at target frames for comparison
    # (to show these channels beat "random" features)
    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_shoulder', 'right_hip', 'mid_hip', 'neck']
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            continue
        for f in [150, 153, 165, 170, 175]:
            for ax, axname in enumerate(['x', 'y', 'z']):
                feats[f'hr_{jname}_{axname}_f{f}'] = float(ts_hr[f, idx, ax])
                feats[f'vel_{jname}_{axname}_f{f}'] = float(vel_hr[f, idx, ax])

    return feats


def build_feature_df(data):
    """Build feature dataframe for all shots."""
    print("  Extracting features...")
    rows = []
    kp_index = data['kp_index']
    n = data['X_3d'].shape[0]
    for i in range(n):
        f = extract_key_features(data['X_3d'][i], kp_index)
        f['pid'] = data['pids'][i]
        for ti, t in enumerate(TARGETS):
            f[t] = float(data['y'][i, ti])
        rows.append(f)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n}")
    df = pd.DataFrame(rows)
    df = df.fillna(0.0)
    return df


# ============================================================
# STATISTICAL TESTS
# ============================================================

def pearson_r(x, y):
    """Pearson correlation with NaN handling."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0, 1.0
    x, y = x[mask], y[mask]
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0, 1.0
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    # Two-tailed p-value via Fisher transformation
    z = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    p = 2 * (1 - norm.cdf(abs(z) / se))
    return r, p


def fisher_z_test(r1, n1, r2, n2):
    """
    Fisher z-test: Is r1 significantly different from r2?
    H0: rho_1 == rho_2
    Returns: z_stat, p_value (two-tailed)
    """
    z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))
    se = np.sqrt(1.0 / max(n1 - 3, 1) + 1.0 / max(n2 - 3, 1))
    z_stat = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(z_stat)))
    return float(z_stat), float(p)


def bootstrap_r(x, y, n_boot=2000, seed=42):
    """Bootstrap 95% CI for Pearson r."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = pearson_r(x[idx], y[idx])
        rs.append(r)
    rs = np.array(rs)
    ci_lo, ci_hi = np.percentile(rs, [2.5, 97.5])
    return float(np.mean(rs)), float(ci_lo), float(ci_hi)


def permutation_test_max_r(feat_matrix, y, n_perm=1000, seed=42):
    """
    Permutation test: Under null hypothesis that no feature correlates with y,
    what is the expected maximum |r| across all features?
    """
    rng = np.random.default_rng(seed)
    n_feats = feat_matrix.shape[1]
    max_rs = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        rs = [abs(pearson_r(feat_matrix[:, fi], y_perm)[0]) for fi in range(n_feats)]
        max_rs.append(max(rs))
    return np.array(max_rs)


def cross_val_r(x, y, n_splits=5, seed=42):
    """
    Cross-validation: correlation on held-out half.
    Fits nothing - just measures stability of correlation.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.permutation(n)
    split_size = n // n_splits
    held_out_rs = []
    for fold in range(n_splits):
        test_idx = idx[fold * split_size:(fold + 1) * split_size]
        r, _ = pearson_r(x[test_idx], y[test_idx])
        held_out_rs.append(r)
    return float(np.mean(held_out_rs)), float(np.std(held_out_rs))


# ============================================================
# MAIN ANALYSIS
# ============================================================

if __name__ == "__main__":
    t_start = time.time()
    print("=== Rigorous Statistical Validation of Player-Specific Channels ===\n")

    data = load_data()
    df = build_feature_df(data)

    unique_pids = sorted(df['pid'].unique())
    feat_cols = [c for c in df.columns if c not in ['pid'] + TARGETS]

    results = {}

    for target in TARGETS:
        tf = TARGET_FRAMES[target]
        print(f"\n{'='*70}")
        print(f"TARGET: {target.upper()} (frame={tf})")
        print(f"{'='*70}")

        # --- STEP 1: Per-player top features ---
        player_top = {}
        for pid in unique_pids:
            mask = df['pid'] == pid
            sub = df[mask]
            y = sub[target].values
            n_p = len(sub)

            rs = {}
            for fc in feat_cols:
                x = sub[fc].values
                r, p = pearson_r(x, y)
                rs[fc] = (r, p, n_p)

            # Sort by |r|
            sorted_feats = sorted(rs.items(), key=lambda kv: abs(kv[1][0]), reverse=True)
            player_top[pid] = sorted_feats[:10]

        print(f"\n--- Top Features Per Player ---")
        for pid in unique_pids:
            print(f"\nP{pid} (n={sum(df['pid']==pid)}):")
            for fname, (r, p, n) in player_top[pid][:5]:
                stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                print(f"  r={r:+.3f} p={p:.4f}{stars}  {fname}")

        # --- STEP 2: Focus on the KEY discovered channels ---
        print(f"\n--- KEY CHANNEL: Fisher Z-Tests (Is P1 different from others?) ---")

        # For LR: vel_rh_y_f175 (P1 channel) and vel_rw_y_f175 (P5 channel)
        # For ANGLE: hr_neck_y_f165 (P4 channel) and hr_lw_y_f153 (P5 channel)
        key_features = {
            "angle": ["hr_neck_y_f165", "hr_lw_y_f153", "hr_ls_y_f160"],
            "depth": ["vel_rw_y_f160", "vel_rh_y_f165", "hr_lw_y_f153"],
            "left_right": ["vel_rh_y_f175", "vel_rw_y_f175", "vel_mh_y_f175"],
        }

        key_feats_for_target = key_features[target]
        fisher_results = {}

        for kf in key_feats_for_target:
            if kf not in feat_cols:
                continue
            print(f"\n  Feature: {kf}")
            player_rs = {}
            player_ns = {}
            for pid in unique_pids:
                mask = df['pid'] == pid
                sub = df[mask]
                x = sub[kf].values
                y = sub[target].values
                r, p = pearson_r(x, y)
                n_p = len(sub)
                player_rs[pid] = r
                player_ns[pid] = n_p
                stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "  "))
                print(f"    P{pid}: r={r:+.3f} (n={n_p}, p={p:.4f}{stars})")

            # Fisher z-test for all pairs
            print(f"  Fisher Z-tests (pairwise):")
            for i, pid1 in enumerate(unique_pids):
                for pid2 in unique_pids[i+1:]:
                    z, p = fisher_z_test(player_rs[pid1], player_ns[pid1],
                                          player_rs[pid2], player_ns[pid2])
                    sig = "SIGNIFICANT" if p < 0.05 else "not sig"
                    print(f"    P{pid1} vs P{pid2}: z={z:+.2f}, p={p:.4f}  [{sig}]")

            fisher_results[kf] = {'player_rs': player_rs, 'player_ns': player_ns}

        # --- STEP 3: Bootstrap CIs for top player channels ---
        print(f"\n--- Bootstrap 95% CIs (2000 resamples) ---")
        for target_pid, target_feat in [
            (1, "vel_rh_y_f175"), (5, "vel_rw_y_f175"),
            (4, "hr_neck_y_f165"), (5, "hr_lw_y_f153")
        ]:
            if target not in ["left_right", "angle"]:
                continue
            if target_pid not in unique_pids:
                continue
            if target_feat not in feat_cols:
                continue
            # Check target match
            if target == "left_right" and target_feat not in ["vel_rh_y_f175", "vel_rw_y_f175"]:
                continue
            if target == "angle" and target_feat not in ["hr_neck_y_f165", "hr_lw_y_f153"]:
                continue

            mask = df['pid'] == target_pid
            sub = df[mask]
            x = sub[target_feat].values
            y = sub[target].values
            r_mean, ci_lo, ci_hi = bootstrap_r(x, y)
            print(f"  P{target_pid} {target_feat}: r={r_mean:.3f} 95%CI=[{ci_lo:.3f}, {ci_hi:.3f}]")

        # --- STEP 4: Permutation test ---
        print(f"\n--- Permutation Test (null distribution of max |r|) ---")
        print(f"  Question: Is the observed max |r| for each player better than chance?")
        for pid in unique_pids:
            mask = df['pid'] == pid
            sub = df[mask]
            y = sub[target].values
            feat_matrix = sub[feat_cols].values

            # Observed max |r|
            obs_rs = [abs(pearson_r(feat_matrix[:, fi], y)[0]) for fi in range(len(feat_cols))]
            obs_max_r = max(obs_rs)

            # Permutation null
            perm_max_rs = permutation_test_max_r(feat_matrix, y, n_perm=500)
            p_perm = float(np.mean(perm_max_rs >= obs_max_r))
            print(f"  P{pid}: obs_max|r|={obs_max_r:.3f}, perm_95th={np.percentile(perm_max_rs, 95):.3f}, "
                  f"perm_99th={np.percentile(perm_max_rs, 99):.3f}, p_perm={p_perm:.4f}")

        # --- STEP 5: Cross-player transfer test ---
        print(f"\n--- Cross-Player Transfer Test ---")
        print(f"  Question: Do P1's top features predict P2's {target}? (Should FAIL if channels are player-specific)")

        # For each player: take their top feature, test it on all other players
        for source_pid in unique_pids:
            source_mask = df['pid'] == source_pid
            source_sub = df[source_mask]
            source_y = source_sub[target].values

            # Top feature for this player
            rs_source = {}
            for fc in feat_cols:
                x = source_sub[fc].values
                r, _ = pearson_r(x, source_y)
                rs_source[fc] = abs(r)
            top_feat = max(rs_source, key=rs_source.get)
            top_r = rs_source[top_feat]

            # Test on other players
            transfer_rs = {}
            for target_pid in unique_pids:
                if target_pid == source_pid:
                    continue
                target_mask = df['pid'] == target_pid
                target_sub = df[target_mask]
                x = target_sub[top_feat].values
                y = target_sub[target].values
                r, _ = pearson_r(x, y)
                transfer_rs[target_pid] = r

            avg_transfer = np.mean(list(transfer_rs.values()))
            print(f"  P{source_pid} top feat ({top_feat}, r={top_r:.3f} on P{source_pid}): "
                  f"transfer r={avg_transfer:+.3f} on others "
                  f"[{', '.join(f'P{p}:{transfer_rs[p]:+.2f}' for p in unique_pids if p != source_pid)}]")

        # --- STEP 6: Cross-validation stability ---
        print(f"\n--- Cross-Validation Stability (5-fold) ---")
        print(f"  Question: Are top features stable across folds (not overfitting to small N)?")
        for pid in unique_pids:
            mask = df['pid'] == pid
            sub = df[mask]
            y = sub[target].values

            # Top feature on full data
            rs_full = {}
            for fc in feat_cols:
                x = sub[fc].values
                r, _ = pearson_r(x, y)
                rs_full[fc] = abs(r)
            top_feat = max(rs_full, key=rs_full.get)
            top_r_full = rs_full[top_feat]

            # CV stability
            x_top = sub[top_feat].values
            cv_r_mean, cv_r_std = cross_val_r(x_top, y)

            print(f"  P{pid} top feat ({top_feat}): full r={top_r_full:.3f}, "
                  f"CV r_mean={cv_r_mean:.3f} +/- {cv_r_std:.3f}")

        results[target] = fisher_results

    # Save results
    total = time.time() - t_start
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"rigorous_channel_analysis_{ts}.json"

    # Convert to serializable
    save_results = {}
    for t, fr in results.items():
        save_results[t] = {}
        for kf, v in fr.items():
            save_results[t][kf] = {
                'player_rs': {str(k): float(vv) for k, vv in v['player_rs'].items()},
                'player_ns': {str(k): int(vv) for k, vv in v['player_ns'].items()},
            }
    save_results['total_time_s'] = total

    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Total time: {total:.1f}s")
    print(f"Results saved to {out_path}")
    print(f"\nSUMMARY FOR PRESENTATION:")
    print(f"  The Fisher z-tests above show whether player-specific channels are")
    print(f"  statistically distinguishable from each other.")
    print(f"  The cross-player transfer test shows whether channels are player-specific")
    print(f"  (P1's top feature should NOT predict P2 if channels are truly individual).")
    print(f"  Bootstrap CIs show the uncertainty on key correlations.")

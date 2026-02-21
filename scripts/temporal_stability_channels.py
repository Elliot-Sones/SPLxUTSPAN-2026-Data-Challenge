"""
Temporal Stability of Player-Specific Biomechanical Control Channels

Test: Split each player's shots by shot_id (ascending) into first/second half.
Compute feature-target correlations independently on each half.
If the same features correlate in both halves, channels are temporally stable.
"""

import numpy as np
import pandas as pd
import ast
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

# Constants
HOOP = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
DATA_DIR = "/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/data"

# Load data
print("Loading data...")
df = pd.read_csv(f"{DATA_DIR}/train.csv")
print(f"Shape: {df.shape}, {df['shot_id'].nunique()} shots")

# Get keypoint names
cols = df.columns.tolist()
kp_x_cols = [c for c in cols if c.endswith('_x') and c not in ['id']]
keypoint_names = [c[:-2] for c in kp_x_cols]
print(f"Found {len(keypoint_names)} keypoints")

targets = ['angle', 'depth', 'left_right']

def safe_savgol(x, window, polyorder, **kwargs):
    if len(x) < window:
        return x
    return savgol_filter(x, window, polyorder, **kwargs)

def parse_sequence(val):
    """Parse a string representation of a list into numpy array. Handles nan values."""
    if isinstance(val, str):
        # Replace nan with NaN that numpy understands
        cleaned = val.replace('nan', 'float("nan")')
        arr = np.array(eval(cleaned))
        return arr
    return np.array(val)

# Parse all shots
print("Parsing keypoint sequences...")
shots_data = []
for idx, row in df.iterrows():
    shot_id = row['shot_id']
    player_id = row['participant_id']

    # Parse positions: (240, n_kp, 3)
    n_kp = len(keypoint_names)
    # Parse first keypoint to get n_frames
    first_seq = parse_sequence(row[f"{keypoint_names[0]}_x"])
    n_frames = len(first_seq)

    positions = np.zeros((n_frames, n_kp, 3))
    for i, kp in enumerate(keypoint_names):
        positions[:, i, 0] = parse_sequence(row[f"{kp}_x"])
        positions[:, i, 1] = parse_sequence(row[f"{kp}_y"])
        positions[:, i, 2] = parse_sequence(row[f"{kp}_z"])

    tgt = {}
    for t in targets:
        tgt[t] = float(row[t])

    shots_data.append({
        'shot_id': shot_id,
        'player_id': player_id,
        'positions': positions,
        'targets': tgt,
    })

print(f"Parsed {len(shots_data)} shots, frames per shot: {shots_data[0]['positions'].shape[0]}")

# Build keypoint index
kp_idx = {kp: i for i, kp in enumerate(keypoint_names)}

def get_hoop_relative(positions, kp_name, frame, axis):
    """Get hoop-relative coordinate for a keypoint at a specific frame."""
    ax_map = {'x': 0, 'y': 1, 'z': 2}
    ax = ax_map[axis]
    ki = kp_idx[kp_name]
    traj = positions[:, ki, ax].copy()
    traj_smooth = safe_savgol(traj, 11, 3)
    return traj_smooth[frame] - HOOP[ax]

def get_velocity(positions, kp_name, frame, axis):
    """Get savgol velocity for a keypoint at a specific frame."""
    ax_map = {'x': 0, 'y': 1, 'z': 2}
    ax = ax_map[axis]
    ki = kp_idx[kp_name]
    traj = positions[:, ki, ax].copy()
    vel = safe_savgol(traj, 11, 3, deriv=1, delta=DT)
    return vel[frame]

# Define the channels to test
channels = [
    (5, "vel_left_shoulder_z_f153", lambda pos: get_velocity(pos, 'left_shoulder', 153, 'z'), 'depth',
     "P5 depth: left shoulder z-velocity at frame 153"),
    (2, "hr_right_wrist_y_f150", lambda pos: get_hoop_relative(pos, 'right_wrist', 150, 'y'), 'left_right',
     "P2 LR: right wrist y-position (hoop-rel) at frame 150"),
    (1, "vel_right_hip_y_f175", lambda pos: get_velocity(pos, 'right_hip', 175, 'y'), 'left_right',
     "P1 LR: right hip y-velocity at frame 175"),
    (4, "hr_neck_y_f165", lambda pos: get_hoop_relative(pos, 'neck', 165, 'y'), 'angle',
     "P4 angle: neck y-position (hoop-rel) at frame 165"),
    (5, "hr_left_wrist_y_f153", lambda pos: get_hoop_relative(pos, 'left_wrist', 153, 'y'), 'angle',
     "P5 angle: left wrist y-position (hoop-rel) at frame 153"),
    (3, "hr_left_shoulder_y_f170", lambda pos: get_hoop_relative(pos, 'left_shoulder', 170, 'y'), 'left_right',
     "P3 LR: left shoulder y-position (hoop-rel) at frame 170"),
]

print("\n" + "="*80)
print("PART 1: Testing pre-identified channels")
print("="*80)

results = []

for player_id, feat_name, extractor, target, description in channels:
    player_shots = sorted([s for s in shots_data if s['player_id'] == player_id],
                         key=lambda s: s['shot_id'])
    n = len(player_shots)
    half = n // 2
    first_half = player_shots[:half]
    second_half = player_shots[half:]

    feat_first = np.array([extractor(s['positions']) for s in first_half])
    tgt_first = np.array([s['targets'][target] for s in first_half])
    feat_second = np.array([extractor(s['positions']) for s in second_half])
    tgt_second = np.array([s['targets'][target] for s in second_half])

    feat_all = np.concatenate([feat_first, feat_second])
    tgt_all = np.concatenate([tgt_first, tgt_second])
    r_full, p_full = pearsonr(feat_all, tgt_all)
    r_first, p_first = pearsonr(feat_first, tgt_first)
    r_second, p_second = pearsonr(feat_second, tgt_second)

    same_sign = (r_first > 0) == (r_second > 0)
    magnitude_ratio = min(abs(r_first), abs(r_second)) / max(abs(r_first), abs(r_second)) if max(abs(r_first), abs(r_second)) > 0 else 0

    result = {
        'player': player_id,
        'feature': feat_name,
        'target': target,
        'description': description,
        'n_first': len(first_half),
        'n_second': len(second_half),
        'r_full': r_full,
        'p_full': p_full,
        'r_first': r_first,
        'p_first': p_first,
        'r_second': r_second,
        'p_second': p_second,
        'same_sign': same_sign,
        'magnitude_ratio': magnitude_ratio,
        'stable': same_sign and magnitude_ratio > 0.3,
    }
    results.append(result)

    status = "STABLE" if result['stable'] else "UNSTABLE"
    print(f"\n{description}")
    print(f"  N: {len(first_half)} / {len(second_half)}")
    print(f"  r_full={r_full:.4f} (p={p_full:.4f})")
    print(f"  r_first={r_first:.4f} (p={p_first:.4f})")
    print(f"  r_second={r_second:.4f} (p={p_second:.4f})")
    print(f"  Same sign: {same_sign}, Magnitude ratio: {magnitude_ratio:.3f}")
    print(f"  --> {status}")


# PART 2: Discovery validation
print("\n" + "="*80)
print("PART 2: Discovery validation - find top features on first half, validate on second half")
print("="*80)

scan_keypoints = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'neck', 'mid_hip', 'nose',
    'left_knee', 'right_knee',
    'right_first_finger_distal', 'right_second_finger_distal',
    'right_third_finger_distal', 'right_fourth_finger_distal',
    'right_fifth_finger_distal',
]

scan_frames = [140, 145, 150, 153, 155, 160, 165, 170, 175, 180]
scan_axes = ['x', 'y', 'z']
scan_types = ['position', 'velocity']

discovery_results = []

for player_id in sorted(df['participant_id'].unique()):
    player_shots = sorted([s for s in shots_data if s['player_id'] == player_id],
                         key=lambda s: s['shot_id'])
    n = len(player_shots)
    half = n // 2
    first_half = player_shots[:half]
    second_half = player_shots[half:]

    for target in targets:
        best_r_first = 0
        best_feat = None
        best_extractor = None

        tgt_first = np.array([s['targets'][target] for s in first_half])
        tgt_second = np.array([s['targets'][target] for s in second_half])

        for kp in scan_keypoints:
            if kp not in kp_idx:
                continue
            for frame in scan_frames:
                for axis in scan_axes:
                    for feat_type in scan_types:
                        if feat_type == 'position':
                            ext = lambda pos, k=kp, f=frame, a=axis: get_hoop_relative(pos, k, f, a)
                            fname = f"hr_{kp}_{axis}_f{frame}"
                        else:
                            ext = lambda pos, k=kp, f=frame, a=axis: get_velocity(pos, k, f, a)
                            fname = f"vel_{kp}_{axis}_f{frame}"

                        feat_vals = np.array([ext(s['positions']) for s in first_half])
                        if np.std(feat_vals) < 1e-10:
                            continue
                        r, p = pearsonr(feat_vals, tgt_first)
                        if abs(r) > abs(best_r_first):
                            best_r_first = r
                            best_feat = fname
                            best_extractor = ext

        if best_extractor is None:
            continue

        feat_second = np.array([best_extractor(s['positions']) for s in second_half])
        r_second, p_second = pearsonr(feat_second, tgt_second)

        same_sign = (best_r_first > 0) == (r_second > 0)
        mag_ratio = min(abs(best_r_first), abs(r_second)) / max(abs(best_r_first), abs(r_second)) if max(abs(best_r_first), abs(r_second)) > 0 else 0

        disc = {
            'player': player_id,
            'target': target,
            'best_feature': best_feat,
            'n_first': len(first_half),
            'n_second': len(second_half),
            'r_first': best_r_first,
            'r_second': r_second,
            'p_second': p_second,
            'same_sign': same_sign,
            'magnitude_ratio': mag_ratio,
            'stable': same_sign and mag_ratio > 0.3,
        }
        discovery_results.append(disc)

        status = "STABLE" if disc['stable'] else "UNSTABLE"
        print(f"\nP{player_id} {target}: best on 1st half = {best_feat}")
        print(f"  r_first={best_r_first:.4f}, r_second={r_second:.4f} (p={p_second:.4f})")
        print(f"  Same sign: {same_sign}, Mag ratio: {mag_ratio:.3f} --> {status}")


# PART 3: Permutation test
print("\n" + "="*80)
print("PART 3: Permutation test - is observed stability better than chance?")
print("="*80)

np.random.seed(42)
n_perms = 1000

perm_results = []
for player_id, feat_name, extractor, target, description in channels:
    player_shots = sorted([s for s in shots_data if s['player_id'] == player_id],
                         key=lambda s: s['shot_id'])
    n = len(player_shots)
    half = n // 2

    first_half = player_shots[:half]
    second_half = player_shots[half:]

    feat_first = np.array([extractor(s['positions']) for s in first_half])
    tgt_first = np.array([s['targets'][target] for s in first_half])
    feat_second = np.array([extractor(s['positions']) for s in second_half])
    tgt_second = np.array([s['targets'][target] for s in second_half])

    r_first_true, _ = pearsonr(feat_first, tgt_first)
    r_second_true, _ = pearsonr(feat_second, tgt_second)
    stability_true = r_first_true * r_second_true

    all_feat = np.concatenate([feat_first, feat_second])
    all_tgt = np.concatenate([tgt_first, tgt_second])

    perm_stabilities = []
    for _ in range(n_perms):
        idx = np.random.permutation(n)
        pf = all_feat[idx[:half]]
        tf = all_tgt[idx[:half]]
        ps = all_feat[idx[half:]]
        ts = all_tgt[idx[half:]]
        r1, _ = pearsonr(pf, tf)
        r2, _ = pearsonr(ps, ts)
        perm_stabilities.append(r1 * r2)

    perm_stabilities = np.array(perm_stabilities)
    p_value = np.mean(perm_stabilities >= stability_true)

    perm_results.append({
        'description': description,
        'stability_metric': stability_true,
        'perm_mean': np.mean(perm_stabilities),
        'perm_std': np.std(perm_stabilities),
        'p_value': p_value,
    })

    print(f"\n{description}")
    print(f"  Stability metric (r1*r2): {stability_true:.4f}")
    print(f"  Permutation mean: {np.mean(perm_stabilities):.4f}, std: {np.std(perm_stabilities):.4f}")
    print(f"  Permutation p-value: {p_value:.4f}")
    print(f"  --> {'SIGNIFICANT (p<0.05)' if p_value < 0.05 else 'NOT SIGNIFICANT'}")


# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

n_stable = sum(1 for r in results if r['stable'])
print(f"\nPre-identified channels: {n_stable}/{len(results)} temporally stable")

n_disc_stable = sum(1 for r in discovery_results if r['stable'])
print(f"Discovery channels: {n_disc_stable}/{len(discovery_results)} temporally stable")

print("\nDetailed results:")
for r in results:
    status = "STABLE" if r['stable'] else "UNSTABLE"
    print(f"  P{r['player']} {r['target']}: r_1st={r['r_first']:.4f} r_2nd={r['r_second']:.4f} [{status}]")

print("\nPermutation test results:")
for pr in perm_results:
    sig = "SIG" if pr['p_value'] < 0.05 else "NS"
    print(f"  {pr['description']}: r1*r2={pr['stability_metric']:.4f}, p={pr['p_value']:.4f} [{sig}]")

# Write results
out_dir = "/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/Research"
out_path = f"{out_dir}/TEMPORAL_STABILITY_CHANNELS_2026-02-20.md"

with open(out_path, 'w') as f:
    f.write("# Temporal Stability of Player-Specific Biomechanical Control Channels\n\n")
    f.write("**Date**: 2026-02-20\n")
    f.write("**Script**: scripts/temporal_stability_channels.py\n\n")

    f.write("## Method\n\n")
    f.write("Split each player's shots by shot_id (ascending) into first half / second half.\n")
    f.write("Computed feature-target Pearson correlations independently on each half.\n")
    f.write("A channel is 'temporally stable' if correlations have the same sign AND magnitude ratio > 0.3.\n")
    f.write("Permutation test (1000 shuffles) assesses whether temporal stability exceeds chance.\n\n")

    f.write("## Part 1: Pre-identified Channel Results\n\n")
    f.write("| Player | Feature | Target | N (1st/2nd) | r_full | r_first | r_second | Same Sign | Mag Ratio | Status |\n")
    f.write("|--------|---------|--------|-------------|--------|---------|----------|-----------|-----------|--------|\n")
    for r in results:
        status = "STABLE" if r['stable'] else "UNSTABLE"
        f.write(f"| P{r['player']} | {r['feature']} | {r['target']} | {r['n_first']}/{r['n_second']} | "
                f"{r['r_full']:.4f} | {r['r_first']:.4f} | {r['r_second']:.4f} | "
                f"{'Yes' if r['same_sign'] else 'No'} | {r['magnitude_ratio']:.3f} | {status} |\n")

    f.write(f"\n**Summary**: {n_stable}/{len(results)} channels temporally stable\n\n")

    f.write("## Part 2: Discovery Validation Results\n\n")
    f.write("Best feature found on first half only, then validated on second half (unseen data).\n\n")
    f.write("| Player | Target | Best Feature (1st half) | N (1st/2nd) | r_first | r_second | p_second | Same Sign | Mag Ratio | Status |\n")
    f.write("|--------|--------|------------------------|-------------|---------|----------|----------|-----------|-----------|--------|\n")
    for r in discovery_results:
        status = "STABLE" if r['stable'] else "UNSTABLE"
        f.write(f"| P{r['player']} | {r['target']} | {r['best_feature']} | {r['n_first']}/{r['n_second']} | "
                f"{r['r_first']:.4f} | {r['r_second']:.4f} | {r['p_second']:.4f} | "
                f"{'Yes' if r['same_sign'] else 'No'} | {r['magnitude_ratio']:.3f} | {status} |\n")

    f.write(f"\n**Summary**: {n_disc_stable}/{len(discovery_results)} discovery channels temporally stable\n\n")

    f.write("## Part 3: Permutation Test Results\n\n")
    f.write("Stability metric = r_first * r_second. Positive = same sign, larger = stronger.\n")
    f.write("P-value = fraction of 1000 random temporal shuffles with equal or greater stability.\n\n")
    f.write("| Channel | r1*r2 (observed) | Perm mean | Perm std | p-value | Significant? |\n")
    f.write("|---------|-----------------|-----------|----------|---------|-------------|\n")
    for pr in perm_results:
        sig = "YES (p<0.05)" if pr['p_value'] < 0.05 else "No"
        f.write(f"| {pr['description']} | {pr['stability_metric']:.4f} | "
                f"{pr['perm_mean']:.4f} | {pr['perm_std']:.4f} | {pr['p_value']:.4f} | {sig} |\n")

    f.write("\n## Interpretation\n\n")

    stability_rate = n_stable / len(results) if results else 0
    disc_rate = n_disc_stable / len(discovery_results) if discovery_results else 0

    if stability_rate >= 0.7:
        f.write("**Strong evidence for temporal stability**: The majority of pre-identified channels\n")
        f.write("show consistent correlations across temporal halves. These are genuine motor signatures,\n")
        f.write("not artifacts of overfitting to the full training set.\n\n")
    elif stability_rate >= 0.4:
        f.write("**Mixed evidence**: Some channels are temporally stable while others are not.\n")
        f.write("The stable channels likely represent genuine motor signatures, while unstable ones\n")
        f.write("may be artifacts or context-dependent strategies.\n\n")
    else:
        f.write("**Weak evidence for temporal stability**: Most channels do not replicate across\n")
        f.write("temporal halves. The correlations found on the full dataset may be artifacts.\n\n")

    if disc_rate < stability_rate - 0.1:
        f.write("Note: Discovery validation rate is lower than pre-identified rate, suggesting\n")
        f.write("some overfitting in the feature selection process (selecting the best feature\n")
        f.write("on half the data and expecting it to hold on the other half is a hard test).\n\n")

    # Practical implications
    f.write("## Practical Implications for Competition\n\n")
    f.write("If channels are stable:\n")
    f.write("- Player-specific feature selection is NOT overfitting - it captures real motor patterns\n")
    f.write("- Per-player models with different features per target are justified\n")
    f.write("- These channels could be used as primary features in test-time predictions\n\n")
    f.write("If channels are unstable:\n")
    f.write("- The full-dataset correlations are inflated by temporal non-stationarity\n")
    f.write("- Player-specific feature selection carries overfitting risk\n")
    f.write("- Should prefer generic features that work across all time periods\n")

print(f"\nResults written to {out_path}")

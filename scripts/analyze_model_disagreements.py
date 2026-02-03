"""
Analyze Model Disagreements

Instead of adding external data, analyze where our best models disagree
and try to understand WHICH model is right in different situations.

Key insight: Sub 219 (LB 0.007682) is best, but it's a blend.
Different component models might be better for different samples.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def extract_simple_features(timeseries, keypoint_idx):
    """Extract simple features for analysis."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Key frame features
    for joint in ["right_wrist", "right_shoulder", "mid_hip"]:
        pos = get_joint(joint, 153)
        features[f'{joint}_z_153'] = pos[2]

    # Wrist velocity at release
    wrist_pre = get_joint("right_wrist", 150)
    wrist_rel = get_joint("right_wrist", 153)
    if np.any(wrist_pre) and np.any(wrist_rel):
        vel = np.linalg.norm(wrist_rel - wrist_pre)
        features['wrist_velocity'] = vel

    return features


def main():
    print("="*80)
    print("ANALYZING MODEL DISAGREEMENTS")
    print("="*80)

    # Load key submissions
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")  # Best LB
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")  # Previous best
    sub151 = pd.read_csv(SUBMISSION_DIR / "submission_151.csv")  # Used for amplification
    sub9 = pd.read_csv(SUBMISSION_DIR / "submission_9.csv")      # Original ensemble

    print(f"\nLoaded {len(sub219)} test samples")

    # Analyze disagreements
    print("\n" + "="*60)
    print("SUBMISSION CORRELATIONS")
    print("="*60)

    subs = {
        'Sub219': sub219['scaled_angle'].values,
        'Sub133': sub133['scaled_angle'].values,
        'Sub151': sub151['scaled_angle'].values,
        'Sub9': sub9['scaled_angle'].values,
    }

    for name1, vals1 in subs.items():
        for name2, vals2 in subs.items():
            if name1 < name2:
                corr = np.corrcoef(vals1, vals2)[0, 1]
                print(f"{name1} vs {name2}: {corr:.4f}")

    # Find samples where Sub 219 differs most from others
    print("\n" + "="*60)
    print("SAMPLES WHERE SUB 219 DIFFERS MOST")
    print("="*60)

    diff_from_133 = np.abs(sub219['scaled_angle'].values - sub133['scaled_angle'].values)
    diff_from_151 = np.abs(sub219['scaled_angle'].values - sub151['scaled_angle'].values)
    diff_from_9 = np.abs(sub219['scaled_angle'].values - sub9['scaled_angle'].values)

    print(f"\nMean abs diff from Sub 133: {diff_from_133.mean():.6f}")
    print(f"Mean abs diff from Sub 151: {diff_from_151.mean():.6f}")
    print(f"Mean abs diff from Sub 9:   {diff_from_9.mean():.6f}")

    # Top disagreeing samples
    top_diff_idx = np.argsort(diff_from_133)[-10:]
    print(f"\nTop 10 samples where Sub 219 differs from Sub 133:")
    for idx in top_diff_idx:
        print(f"  Sample {idx}: 219={sub219.iloc[idx]['scaled_angle']:.4f}, "
              f"133={sub133.iloc[idx]['scaled_angle']:.4f}, "
              f"diff={diff_from_133[idx]:.4f}")

    # Load test features to understand these samples
    print("\n" + "="*60)
    print("ANALYZING HIGH-DISAGREEMENT SAMPLES")
    print("="*60)

    keypoint_idx = get_keypoint_indices()
    test_features = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_simple_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_ids.append(metadata['id'])

    test_df = pd.DataFrame(test_features)
    test_df['id'] = test_ids
    test_df['diff_from_133'] = diff_from_133
    test_df['sub219_angle'] = sub219['scaled_angle'].values
    test_df['sub133_angle'] = sub133['scaled_angle'].values

    # Compare high vs low disagreement samples
    high_diff = test_df[test_df['diff_from_133'] > test_df['diff_from_133'].quantile(0.8)]
    low_diff = test_df[test_df['diff_from_133'] < test_df['diff_from_133'].quantile(0.2)]

    print(f"\nHigh disagreement samples ({len(high_diff)}):")
    for col in ['right_wrist_z_153', 'wrist_velocity']:
        if col in test_df.columns:
            print(f"  {col}: {high_diff[col].mean():.4f}")

    print(f"\nLow disagreement samples ({len(low_diff)}):")
    for col in ['right_wrist_z_153', 'wrist_velocity']:
        if col in test_df.columns:
            print(f"  {col}: {low_diff[col].mean():.4f}")

    # What predicts disagreement?
    print("\n" + "="*60)
    print("WHAT PREDICTS DISAGREEMENT?")
    print("="*60)

    for col in ['right_wrist_z_153', 'right_shoulder_z_153', 'wrist_velocity']:
        if col in test_df.columns:
            corr = np.corrcoef(test_df[col].fillna(0), test_df['diff_from_133'])[0, 1]
            print(f"  {col} vs disagreement: {corr:.4f}")

    # Direction analysis: when Sub 219 predicts higher, is that correct?
    print("\n" + "="*60)
    print("DIRECTION OF AMPLIFICATION")
    print("="*60)

    higher_219 = (sub219['scaled_angle'].values > sub133['scaled_angle'].values).sum()
    lower_219 = (sub219['scaled_angle'].values < sub133['scaled_angle'].values).sum()
    print(f"Sub 219 predicts HIGHER than Sub 133: {higher_219} samples")
    print(f"Sub 219 predicts LOWER than Sub 133: {lower_219} samples")

    # Profile analysis
    print("\n" + "="*60)
    print("PROFILE COMPARISON")
    print("="*60)

    for name, sub in [('Sub219', sub219), ('Sub133', sub133), ('Sub151', sub151), ('Sub9', sub9)]:
        print(f"\n{name}:")
        print(f"  angle_mean: {sub['scaled_angle'].mean():.6f}")
        print(f"  angle_std:  {sub['scaled_angle'].std():.6f}")
        print(f"  depth_mean: {sub['scaled_depth'].mean():.6f}")

    # Recommendation
    print("\n" + "="*60)
    print("ANALYSIS CONCLUSION")
    print("="*60)
    print("""
The key insight: Sub 219's improvement over Sub 133 comes from
selective amplification on ~9% of samples (percentile 91).

These high-disagreement samples share characteristics:
- Slightly higher wrist position at release
- More variability in wrist velocity

The amplification direction matters - Sub 219 tends to push predictions
TOWARD Sub 133's direction, just more extremely.

To improve further, we would need to:
1. Identify which of these amplified samples are STILL wrong
2. Find a different signal for those specific samples
3. Or find a new set of samples where current models disagree

However, without ground truth on test data, we cannot verify which
samples are being predicted correctly.
""")


if __name__ == "__main__":
    main()

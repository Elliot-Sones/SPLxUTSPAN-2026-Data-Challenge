"""
Generate Final Candidate Submissions

Create the most promising candidates based on all our findings.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Load key submissions
subs = {}
for num in [9, 10, 25, 111, 113, 133, 147, 149, 151, 219]:
    path = SUBMISSION_DIR / f"submission_{num}.csv"
    if path.exists():
        subs[num] = pd.read_csv(path)

test_ids = subs[219]['id'].values

# Get next submission number
existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
next_num = max(nums) + 1

TARGET_DEPTH_MEAN = 0.5055
TARGET_ANGLE_STD = 0.137162  # Sub 219's angle_std

def save_submission(angle, depth, lr, name):
    global next_num
    depth = depth - np.mean(depth) + TARGET_DEPTH_MEAN
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': np.clip(angle, 0, 1),
        'scaled_depth': np.clip(depth, 0, 1),
        'scaled_left_right': np.clip(lr, 0, 1)
    })
    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"Saved: {output_file} ({name}, angle_std={np.std(angle):.6f})")
    next_num += 1

print("="*80)
print("GENERATING FINAL CANDIDATES")
print("="*80)

# Candidate 1: Fine-tuned blend around best pair (0.6*Sub149 + 0.4*Sub219)
print("\n--- Fine-tuned pairs around 0.6*Sub149 + 0.4*Sub219 ---")
for w in [0.55, 0.58, 0.60, 0.62, 0.65]:
    angle = w * subs[149]['scaled_angle'].values + (1-w) * subs[219]['scaled_angle'].values
    depth = w * subs[149]['scaled_depth'].values + (1-w) * subs[219]['scaled_depth'].values
    lr = w * subs[149]['scaled_left_right'].values + (1-w) * subs[219]['scaled_left_right'].values
    save_submission(angle, depth, lr, f"{w:.2f}*Sub149 + {1-w:.2f}*Sub219")

# Candidate 2: Blend with Sub 133 variations
print("\n--- Sub 133 variations ---")
for w in [0.65, 0.70, 0.75, 0.80]:
    angle = w * subs[133]['scaled_angle'].values + (1-w) * subs[25]['scaled_angle'].values
    depth = w * subs[133]['scaled_depth'].values + (1-w) * subs[25]['scaled_depth'].values
    lr = w * subs[133]['scaled_left_right'].values + (1-w) * subs[25]['scaled_left_right'].values
    save_submission(angle, depth, lr, f"{w:.2f}*Sub133 + {1-w:.2f}*Sub25")

# Candidate 3: Three-way blends
print("\n--- Three-way blends ---")
configs = [
    (0.5, 0.3, 0.2, 133, 219, 25),
    (0.6, 0.2, 0.2, 133, 219, 9),
    (0.4, 0.4, 0.2, 133, 149, 25),
    (0.5, 0.25, 0.25, 133, 219, 149),
]

for w1, w2, w3, s1, s2, s3 in configs:
    angle = w1 * subs[s1]['scaled_angle'].values + w2 * subs[s2]['scaled_angle'].values + w3 * subs[s3]['scaled_angle'].values
    depth = w1 * subs[s1]['scaled_depth'].values + w2 * subs[s2]['scaled_depth'].values + w3 * subs[s3]['scaled_depth'].values
    lr = w1 * subs[s1]['scaled_left_right'].values + w2 * subs[s2]['scaled_left_right'].values + w3 * subs[s3]['scaled_left_right'].values
    save_submission(angle, depth, lr, f"{w1}*Sub{s1} + {w2}*Sub{s2} + {w3}*Sub{s3}")

# Candidate 4: Amplification with finer control
print("\n--- Fine-tuned amplification ---")
diff_angle = subs[133]['scaled_angle'].values - subs[151]['scaled_angle'].values
diff_depth = subs[133]['scaled_depth'].values - subs[151]['scaled_depth'].values
diff_lr = subs[133]['scaled_left_right'].values - subs[151]['scaled_left_right'].values
total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

for pctl in [89, 90, 91]:
    for alpha in [0.95, 1.0, 1.05, 1.1]:
        threshold = np.percentile(total_diff, pctl)
        mask = total_diff > threshold

        new_angle = subs[133]['scaled_angle'].values.copy()
        new_depth = subs[133]['scaled_depth'].values.copy()
        new_lr = subs[133]['scaled_left_right'].values.copy()

        new_angle[mask] += alpha * diff_angle[mask]
        new_depth[mask] += alpha * diff_depth[mask]
        new_lr[mask] += alpha * diff_lr[mask]

        save_submission(new_angle, new_depth, new_lr, f"amp_pctl{pctl}_alpha{alpha}")

# Candidate 5: Use Sub 147 as contrast (found to be better)
print("\n--- Sub 147 contrast amplification ---")
diff_angle = subs[133]['scaled_angle'].values - subs[147]['scaled_angle'].values
diff_depth = subs[133]['scaled_depth'].values - subs[147]['scaled_depth'].values
diff_lr = subs[133]['scaled_left_right'].values - subs[147]['scaled_left_right'].values
total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

for pctl in [87, 88, 89, 90]:
    for alpha in [0.7, 0.8, 0.9]:
        threshold = np.percentile(total_diff, pctl)
        mask = total_diff > threshold

        new_angle = subs[133]['scaled_angle'].values.copy()
        new_depth = subs[133]['scaled_depth'].values.copy()
        new_lr = subs[133]['scaled_left_right'].values.copy()

        new_angle[mask] += alpha * diff_angle[mask]
        new_depth[mask] += alpha * diff_depth[mask]
        new_lr[mask] += alpha * diff_lr[mask]

        save_submission(new_angle, new_depth, new_lr, f"amp147_pctl{pctl}_alpha{alpha}")

# Candidate 6: Combine best techniques
print("\n--- Combined best techniques ---")

# Start with best pair, then apply small amplification
base_angle = 0.6 * subs[149]['scaled_angle'].values + 0.4 * subs[219]['scaled_angle'].values
base_depth = 0.6 * subs[149]['scaled_depth'].values + 0.4 * subs[219]['scaled_depth'].values
base_lr = 0.6 * subs[149]['scaled_left_right'].values + 0.4 * subs[219]['scaled_left_right'].values

# Small adjustment toward Sub 133
for w in [0.05, 0.1]:
    angle = (1-w) * base_angle + w * subs[133]['scaled_angle'].values
    depth = (1-w) * base_depth + w * subs[133]['scaled_depth'].values
    lr = (1-w) * base_lr + w * subs[133]['scaled_left_right'].values
    save_submission(angle, depth, lr, f"combined_{w:.2f}_133")

print("\nDone! Generated all final candidates.")

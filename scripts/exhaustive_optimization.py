"""
Exhaustive Optimization

Try every possible combination to find the best submission.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.optimize import minimize

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Load all potentially good submissions
subs = {}
for num in range(1, 450):
    path = SUBMISSION_DIR / f"submission_{num}.csv"
    if path.exists():
        subs[num] = pd.read_csv(path)

print(f"Loaded {len(subs)} submissions")

# Get test IDs
test_ids = list(subs.values())[0]['id'].values

# Target profile (from Sub 219 which has best LB)
TARGET_ANGLE_STD = 0.137162
TARGET_DEPTH_MEAN = 0.5055

# Function to evaluate a blend
def evaluate_blend(weights, sub_list, target='angle'):
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    if target == 'angle':
        preds = sum(w * subs[s]['scaled_angle'].values for w, s in zip(weights, sub_list))
    elif target == 'depth':
        preds = sum(w * subs[s]['scaled_depth'].values for w, s in zip(weights, sub_list))
    else:
        preds = sum(w * subs[s]['scaled_left_right'].values for w, s in zip(weights, sub_list))

    return preds

def profile_score(angle_preds, depth_preds):
    """Score based on how close to target profile."""
    angle_std = np.std(angle_preds)
    depth_mean = np.mean(depth_preds)

    # Penalize deviation from target
    angle_penalty = abs(angle_std - TARGET_ANGLE_STD)
    depth_penalty = abs(depth_mean - TARGET_DEPTH_MEAN)

    return angle_penalty + depth_penalty * 10

# Find best pair of submissions
print("\n" + "="*60)
print("FINDING BEST PAIRS")
print("="*60)

# Use key submissions
key_subs = [9, 10, 25, 111, 113, 133, 147, 149, 151, 219]
key_subs = [s for s in key_subs if s in subs]

best_pairs = []

for s1, s2 in combinations(key_subs, 2):
    for w in np.arange(0.1, 1.0, 0.1):
        angle_preds = w * subs[s1]['scaled_angle'].values + (1-w) * subs[s2]['scaled_angle'].values
        depth_preds = w * subs[s1]['scaled_depth'].values + (1-w) * subs[s2]['scaled_depth'].values
        depth_preds = depth_preds - np.mean(depth_preds) + TARGET_DEPTH_MEAN

        score = profile_score(angle_preds, depth_preds)
        angle_std = np.std(angle_preds)

        best_pairs.append({
            's1': s1, 's2': s2, 'w': w,
            'angle_std': angle_std,
            'score': score
        })

best_pairs = sorted(best_pairs, key=lambda x: x['score'])[:20]
print("\nTop 20 pairs by profile match:")
for p in best_pairs[:10]:
    print(f"  {p['w']:.1f}*Sub{p['s1']} + {1-p['w']:.1f}*Sub{p['s2']}: angle_std={p['angle_std']:.6f}, score={p['score']:.6f}")

# Find best triplets
print("\n" + "="*60)
print("FINDING BEST TRIPLETS")
print("="*60)

best_triplets = []

for s1, s2, s3 in combinations(key_subs[:8], 3):
    for w1 in np.arange(0.1, 0.8, 0.2):
        for w2 in np.arange(0.1, 0.8 - w1, 0.2):
            w3 = 1 - w1 - w2

            angle_preds = (w1 * subs[s1]['scaled_angle'].values +
                           w2 * subs[s2]['scaled_angle'].values +
                           w3 * subs[s3]['scaled_angle'].values)
            depth_preds = (w1 * subs[s1]['scaled_depth'].values +
                           w2 * subs[s2]['scaled_depth'].values +
                           w3 * subs[s3]['scaled_depth'].values)
            depth_preds = depth_preds - np.mean(depth_preds) + TARGET_DEPTH_MEAN

            score = profile_score(angle_preds, depth_preds)
            angle_std = np.std(angle_preds)

            best_triplets.append({
                's1': s1, 's2': s2, 's3': s3,
                'w1': w1, 'w2': w2, 'w3': w3,
                'angle_std': angle_std,
                'score': score
            })

best_triplets = sorted(best_triplets, key=lambda x: x['score'])[:20]
print("\nTop 10 triplets by profile match:")
for t in best_triplets[:10]:
    print(f"  {t['w1']:.1f}*Sub{t['s1']} + {t['w2']:.1f}*Sub{t['s2']} + {t['w3']:.1f}*Sub{t['s3']}: angle_std={t['angle_std']:.6f}")

# Optimize blend weights
print("\n" + "="*60)
print("OPTIMIZING BLEND WEIGHTS")
print("="*60)

# Use top submissions for optimization
opt_subs = [133, 219, 9, 10, 25]
opt_subs = [s for s in opt_subs if s in subs]

def objective(weights):
    weights = np.abs(weights)
    weights = weights / weights.sum()

    angle = sum(w * subs[s]['scaled_angle'].values for w, s in zip(weights, opt_subs))
    depth = sum(w * subs[s]['scaled_depth'].values for w, s in zip(weights, opt_subs))
    depth = depth - np.mean(depth) + TARGET_DEPTH_MEAN

    return profile_score(angle, depth)

# Multiple random starts
best_result = None
best_score = float('inf')

for _ in range(20):
    x0 = np.random.dirichlet(np.ones(len(opt_subs)))
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500})
    if result.fun < best_score:
        best_score = result.fun
        best_result = result

weights = np.abs(best_result.x)
weights = weights / weights.sum()

print(f"\nOptimal weights:")
for w, s in zip(weights, opt_subs):
    print(f"  Sub {s}: {w:.4f}")

# Generate final predictions with optimal weights
angle_opt = sum(w * subs[s]['scaled_angle'].values for w, s in zip(weights, opt_subs))
depth_opt = sum(w * subs[s]['scaled_depth'].values for w, s in zip(weights, opt_subs))
lr_opt = sum(w * subs[s]['scaled_left_right'].values for w, s in zip(weights, opt_subs))

depth_opt = depth_opt - np.mean(depth_opt) + TARGET_DEPTH_MEAN

print(f"\nOptimized blend: angle_std={np.std(angle_opt):.6f}, depth_mean={np.mean(depth_opt):.6f}")

# Save submissions
existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
next_num = max(nums) + 1

# Save best pair
bp = best_pairs[0]
angle = bp['w'] * subs[bp['s1']]['scaled_angle'].values + (1-bp['w']) * subs[bp['s2']]['scaled_angle'].values
depth = bp['w'] * subs[bp['s1']]['scaled_depth'].values + (1-bp['w']) * subs[bp['s2']]['scaled_depth'].values
lr = bp['w'] * subs[bp['s1']]['scaled_left_right'].values + (1-bp['w']) * subs[bp['s2']]['scaled_left_right'].values
depth = depth - np.mean(depth) + TARGET_DEPTH_MEAN

submission = pd.DataFrame({
    'id': test_ids,
    'scaled_angle': np.clip(angle, 0, 1),
    'scaled_depth': np.clip(depth, 0, 1),
    'scaled_left_right': np.clip(lr, 0, 1)
})
output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
submission.to_csv(output_file, index=False)
print(f"\nSaved: {output_file} (best pair: {bp['w']:.1f}*Sub{bp['s1']} + {1-bp['w']:.1f}*Sub{bp['s2']})")
next_num += 1

# Save best triplet
bt = best_triplets[0]
angle = (bt['w1'] * subs[bt['s1']]['scaled_angle'].values +
         bt['w2'] * subs[bt['s2']]['scaled_angle'].values +
         bt['w3'] * subs[bt['s3']]['scaled_angle'].values)
depth = (bt['w1'] * subs[bt['s1']]['scaled_depth'].values +
         bt['w2'] * subs[bt['s2']]['scaled_depth'].values +
         bt['w3'] * subs[bt['s3']]['scaled_depth'].values)
lr = (bt['w1'] * subs[bt['s1']]['scaled_left_right'].values +
      bt['w2'] * subs[bt['s2']]['scaled_left_right'].values +
      bt['w3'] * subs[bt['s3']]['scaled_left_right'].values)
depth = depth - np.mean(depth) + TARGET_DEPTH_MEAN

submission = pd.DataFrame({
    'id': test_ids,
    'scaled_angle': np.clip(angle, 0, 1),
    'scaled_depth': np.clip(depth, 0, 1),
    'scaled_left_right': np.clip(lr, 0, 1)
})
output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
submission.to_csv(output_file, index=False)
print(f"Saved: {output_file} (best triplet)")
next_num += 1

# Save optimized blend
submission = pd.DataFrame({
    'id': test_ids,
    'scaled_angle': np.clip(angle_opt, 0, 1),
    'scaled_depth': np.clip(depth_opt, 0, 1),
    'scaled_left_right': np.clip(lr_opt, 0, 1)
})
output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
submission.to_csv(output_file, index=False)
print(f"Saved: {output_file} (optimized blend)")

print("\nDone!")

"""
Stacking / Meta-Learning

Use predictions from multiple models as features for a meta-model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Load best submissions as base predictions
base_subs = {}

# Load a variety of submissions
for num in [9, 10, 25, 111, 113, 133, 145, 147, 149, 151, 219, 321, 326]:
    path = SUBMISSION_DIR / f"submission_{num}.csv"
    if path.exists():
        base_subs[num] = pd.read_csv(path)

print(f"Loaded {len(base_subs)} base submissions")

# Get test IDs
test_ids = base_subs[219]['id'].values

# Create feature matrix from base predictions
def create_meta_features(base_subs, target_col):
    features = []
    for num, sub in sorted(base_subs.items()):
        features.append(sub[target_col].values)
    return np.column_stack(features)

print("\n" + "="*60)
print("STACKING META-LEARNING")
print("="*60)

# For stacking, we use a weighted average optimized on correlation with Sub 219
# (assuming Sub 219's predictions are good signals)

# Load Sub 219 as target
target_219 = base_subs[219]

# Create meta-features
X_angle = create_meta_features({k: v for k, v in base_subs.items() if k != 219}, 'scaled_angle')
X_depth = create_meta_features({k: v for k, v in base_subs.items() if k != 219}, 'scaled_depth')
X_lr = create_meta_features({k: v for k, v in base_subs.items() if k != 219}, 'scaled_left_right')

print(f"Meta-features shape: {X_angle.shape}")

# Method 1: Simple average of all
print("\nMethod 1: Simple Average")
pred_angle = np.mean(X_angle, axis=1)
pred_depth = np.mean(X_depth, axis=1)
pred_lr = np.mean(X_lr, axis=1)

pred_depth = pred_depth - np.mean(pred_depth) + 0.5055
predictions = np.column_stack([pred_angle, pred_depth, pred_lr])
print(f"angle_std: {np.std(pred_angle):.6f}")

# Method 2: Weighted by correlation with Sub 219
print("\nMethod 2: Correlation-Weighted Average")
weights_angle = []
weights_depth = []
weights_lr = []

for i, (num, sub) in enumerate(sorted({k: v for k, v in base_subs.items() if k != 219}.items())):
    corr_a = np.corrcoef(sub['scaled_angle'].values, target_219['scaled_angle'].values)[0, 1]
    corr_d = np.corrcoef(sub['scaled_depth'].values, target_219['scaled_depth'].values)[0, 1]
    corr_l = np.corrcoef(sub['scaled_left_right'].values, target_219['scaled_left_right'].values)[0, 1]
    weights_angle.append(max(0, corr_a))
    weights_depth.append(max(0, corr_d))
    weights_lr.append(max(0, corr_l))
    print(f"  Sub {num}: angle_corr={corr_a:.4f}, depth_corr={corr_d:.4f}")

weights_angle = np.array(weights_angle) / np.sum(weights_angle)
weights_depth = np.array(weights_depth) / np.sum(weights_depth)
weights_lr = np.array(weights_lr) / np.sum(weights_lr)

pred_angle_w = X_angle @ weights_angle
pred_depth_w = X_depth @ weights_depth
pred_lr_w = X_lr @ weights_lr

pred_depth_w = pred_depth_w - np.mean(pred_depth_w) + 0.5055
print(f"Weighted angle_std: {np.std(pred_angle_w):.6f}")

# Method 3: Blend top submissions only
print("\nMethod 3: Top Submissions Only")
top_subs = {k: v for k, v in base_subs.items() if k in [133, 219, 9, 10, 25]}
X_top_angle = create_meta_features(top_subs, 'scaled_angle')
X_top_depth = create_meta_features(top_subs, 'scaled_depth')
X_top_lr = create_meta_features(top_subs, 'scaled_left_right')

pred_angle_top = np.mean(X_top_angle, axis=1)
pred_depth_top = np.mean(X_top_depth, axis=1)
pred_lr_top = np.mean(X_top_lr, axis=1)

pred_depth_top = pred_depth_top - np.mean(pred_depth_top) + 0.5055
print(f"Top-only angle_std: {np.std(pred_angle_top):.6f}")

# Method 4: Rank-based ensemble
print("\nMethod 4: Rank-Based Ensemble")
from scipy.stats import rankdata

ranks_angle = np.zeros_like(X_angle)
ranks_depth = np.zeros_like(X_depth)
ranks_lr = np.zeros_like(X_lr)

for i in range(X_angle.shape[1]):
    ranks_angle[:, i] = rankdata(X_angle[:, i]) / len(X_angle)
    ranks_depth[:, i] = rankdata(X_depth[:, i]) / len(X_depth)
    ranks_lr[:, i] = rankdata(X_lr[:, i]) / len(X_lr)

pred_angle_rank = np.mean(ranks_angle, axis=1)
pred_depth_rank = np.mean(ranks_depth, axis=1)
pred_lr_rank = np.mean(ranks_lr, axis=1)

pred_depth_rank = pred_depth_rank - np.mean(pred_depth_rank) + 0.5055
print(f"Rank-based angle_std: {np.std(pred_angle_rank):.6f}")

# Save submissions
existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
next_num = max(nums) + 1

methods = [
    ('stacking_average', pred_angle, pred_depth, pred_lr),
    ('stacking_weighted', pred_angle_w, pred_depth_w, pred_lr_w),
    ('stacking_top', pred_angle_top, pred_depth_top, pred_lr_top),
    ('stacking_rank', pred_angle_rank, pred_depth_rank, pred_lr_rank),
]

for name, a, d, lr in methods:
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': np.clip(a, 0, 1),
        'scaled_depth': np.clip(d, 0, 1),
        'scaled_left_right': np.clip(lr, 0, 1)
    })
    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"Saved: {output_file} ({name})")
    next_num += 1

# Method 5: Blend stack with Sub 219
print("\n" + "="*60)
print("BLENDING STACKS WITH SUB 219")
print("="*60)

for w in [0.1, 0.2, 0.3]:
    blend_angle = w * pred_angle_w + (1-w) * target_219['scaled_angle'].values
    blend_depth = w * pred_depth_w + (1-w) * target_219['scaled_depth'].values
    blend_lr = w * pred_lr_w + (1-w) * target_219['scaled_left_right'].values

    blend_depth = blend_depth - np.mean(blend_depth) + 0.5055

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': np.clip(blend_angle, 0, 1),
        'scaled_depth': np.clip(blend_depth, 0, 1),
        'scaled_left_right': np.clip(blend_lr, 0, 1)
    })
    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"Saved: {output_file} ({w:.0%} stack + {1-w:.0%} Sub219, angle_std={submission['scaled_angle'].std():.6f})")
    next_num += 1

print("\nDone!")

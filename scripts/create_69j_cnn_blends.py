#!/usr/bin/env python3
"""
Create 69j-CNN blend submissions at 7% and 10% weights on Sub2716.

Blends:
- Sub 3297: 7% Sub3287 + 93% Sub2716
- Sub 3298: 10% Sub3287 + 90% Sub2716
"""

import pandas as pd
from pathlib import Path

# Paths
submission_dir = Path("/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/submission")
sub_3287_path = submission_dir / "submission_3287.csv"
sub_2716_path = submission_dir / "submission_2716.csv"

# Read submissions
print("Reading Sub 3287 (69j-CNN standalone)...")
sub_3287 = pd.read_csv(sub_3287_path)
print(f"  Shape: {sub_3287.shape}")

print("Reading Sub 2716 (anchor)...")
sub_2716 = pd.read_csv(sub_2716_path)
print(f"  Shape: {sub_2716.shape}")

# Verify same structure
assert list(sub_3287.columns) == list(sub_2716.columns), "Column mismatch!"
assert len(sub_3287) == len(sub_2716), "Row count mismatch!"
assert list(sub_3287["id"]) == list(sub_2716["id"]), "ID mismatch!"
print("\n✓ Both submissions aligned")

# Create blends
targets = ["scaled_angle", "scaled_depth", "scaled_left_right"]

# 7% blend
print("\nCreating 7% Sub3287 + 93% Sub2716...")
blend_7 = sub_2716.copy()
for target in targets:
    blend_7[target] = 0.07 * sub_3287[target] + 0.93 * sub_2716[target]

# 10% blend
print("Creating 10% Sub3287 + 90% Sub2716...")
blend_10 = sub_2716.copy()
for target in targets:
    blend_10[target] = 0.10 * sub_3287[target] + 0.90 * sub_2716[target]

# Save submissions
sub_3297_path = submission_dir / "submission_3297.csv"
sub_3298_path = submission_dir / "submission_3298.csv"

blend_7.to_csv(sub_3297_path, index=False)
print(f"✓ Saved Sub 3297 to {sub_3297_path}")

blend_10.to_csv(sub_3298_path, index=False)
print(f"✓ Saved Sub 3298 to {sub_3298_path}")

# Print summary
print("\n" + "="*70)
print("SUBMISSION SUMMARY")
print("="*70)
print(f"\nSub 3297: 7% Sub3287 (69j-CNN) + 93% Sub2716")
print(f"  Angle   mean: {blend_7['scaled_angle'].mean():.8f}")
print(f"  Depth   mean: {blend_7['scaled_depth'].mean():.8f}")
print(f"  LR      mean: {blend_7['scaled_left_right'].mean():.8f}")

print(f"\nSub 3298: 10% Sub3287 (69j-CNN) + 90% Sub2716")
print(f"  Angle   mean: {blend_10['scaled_angle'].mean():.8f}")
print(f"  Depth   mean: {blend_10['scaled_depth'].mean():.8f}")
print(f"  LR      mean: {blend_10['scaled_left_right'].mean():.8f}")

print("\n" + "="*70)
print("Files ready for Kaggle submission (do NOT submit yet)")
print("="*70)

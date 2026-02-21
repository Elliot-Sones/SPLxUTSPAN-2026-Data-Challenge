"""
Adaptive Oracle Blend: Variable per-player per-target blend weights

Uses oracle LOO quality (R2) to determine how much oracle to use per cell.
High-R2 cells (e.g. P5 depth R2=0.716) get more oracle.
Low-R2 cells get less.

Blend formula:
  pred_blended[player, target] = (1 - w) * pred_base + w * pred_oracle
  
  where w = R2_oracle * max_blend_ratio
  
  E.g. P5 depth R2=0.716, max_blend_ratio=0.30 -> w=0.215
       P2 LR R2=0.582 -> w=0.175
       P4 angle R2=0.366 -> w=0.110
"""

import fcntl
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = PROJECT_DIR / "data"

TARGETS = ["angle", "depth", "left_right"]

# LOO R2 from player_channel_oracle.py
ORACLE_R2 = {
    "angle": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.366, 5: 0.168},
    "depth": {1: 0.422, 2: 0.101, 3: 0.237, 4: 0.533, 5: 0.716},
    "left_right": {1: 0.469, 2: 0.582, 3: 0.435, 4: 0.444, 5: 0.449},
}


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


def main():
    # Load test participant IDs
    test_df = pd.read_csv(DATA_DIR / "test.csv")[["id", "participant_id"]]
    
    # Load base submission
    base_sub_path = SUBMISSION_DIR / "submission_3558.csv"
    base_df = pd.read_csv(base_sub_path)
    
    # Load oracle submission (3626 = 3% fixed blend)
    # Actually we want the pure oracle predictions, not the blended
    # The oracle submissions are: 3626 (3%), 3627 (5%), 3628 (10%), 3629 (20%)
    # We need the oracle-only predictions
    # Reconstruct: oracle_pred = (sub3626 - 0.97*base) / 0.03
    sub3626 = pd.read_csv(SUBMISSION_DIR / "submission_3626.csv")
    
    # Column mapping
    col_map = {}
    for t in TARGETS:
        if f"scaled_{t}" in base_df.columns:
            col_map[t] = f"scaled_{t}"
        else:
            col_map[t] = t
    
    # Reconstruct oracle-only predictions
    oracle_df = base_df.copy()
    for t in TARGETS:
        col = col_map[t]
        oracle_vals = (sub3626[col].values - 0.97 * base_df[col].values) / 0.03
        oracle_df[col] = oracle_vals
    
    # Check reconstruction
    print("Oracle reconstruction check:")
    for t in TARGETS:
        col = col_map[t]
        print(f"  {t}: min={oracle_df[col].min():.3f}, max={oracle_df[col].max():.3f}")
    
    # Merge with participant IDs
    oracle_df = oracle_df.merge(test_df, on='id', how='left')
    base_df2 = base_df.merge(test_df, on='id', how='left')
    
    # Adaptive blend: w per player per target
    for max_ratio in [0.15, 0.25, 0.40]:
        blend_df = base_df.copy()
        
        weights_used = {}
        for t in TARGETS:
            col = col_map[t]
            weights_used[t] = {}
            for pid in [1, 2, 3, 4, 5]:
                r2 = ORACLE_R2[t].get(pid, 0.0)
                w = r2 * max_ratio
                weights_used[t][pid] = float(w)
                
                # Apply to matching rows
                pid_mask = base_df2['participant_id'] == pid
                if pid_mask.sum() == 0:
                    continue
                base_vals = base_df[col].values.copy()
                oracle_vals = oracle_df[col].values.copy()
                
                blend_vals = base_vals.copy()
                blend_vals[pid_mask] = (1 - w) * base_vals[pid_mask] + w * oracle_vals[pid_mask]
                blend_df[col] = blend_vals
        
        sub_num = get_next_submission_number()
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        # Drop participant_id column if added
        if 'participant_id' in blend_df.columns:
            blend_df = blend_df.drop(columns=['participant_id'])
        blend_df.to_csv(sub_path, index=False)
        
        print(f"\nSub {sub_num}: Adaptive oracle blend (max_ratio={max_ratio})")
        print(f"  Weights:")
        for t in TARGETS:
            for pid in [1, 2, 3, 4, 5]:
                w = weights_used[t][pid]
                if w > 0:
                    print(f"    P{pid} {t}: w={w:.3f} (R2={ORACLE_R2[t][pid]:.3f})")


if __name__ == "__main__":
    main()

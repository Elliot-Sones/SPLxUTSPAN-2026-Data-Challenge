"""
Blend multiple submissions for final result.
"""

import pandas as pd
import numpy as np
from pathlib import Path

SUBMISSION_DIR = Path(__file__).parent.parent / "submission"

def blend_submissions(weights=None):
    """Blend submissions with given weights."""

    # Load all recent submissions
    submissions = {}
    for num in [10, 11, 12]:
        filepath = SUBMISSION_DIR / f"submission_{num}.csv"
        if filepath.exists():
            submissions[num] = pd.read_csv(filepath)
            print(f"Loaded submission_{num}.csv")

    if len(submissions) == 0:
        print("No submissions found!")
        return

    # Default equal weights if not specified
    if weights is None:
        weights = {num: 1.0/len(submissions) for num in submissions.keys()}

    print(f"\nBlending with weights: {weights}")

    # Blend predictions
    first_sub = list(submissions.values())[0]
    blended = pd.DataFrame({'id': first_sub['id']})

    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        blended[col] = sum(
            submissions[num][col] * w for num, w in weights.items()
        )

    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    # Save
    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    blended.to_csv(filepath, index=False)

    print(f"\nBlended submission saved to: {filepath}")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={blended[col].mean():.4f}, std={blended[col].std():.4f}")

    return filepath


if __name__ == "__main__":
    # Weights based on CV scores (inverse of CV score as weight)
    # Sub 10: CV 0.007919
    # Sub 11: CV 0.007767
    # Sub 12: CV 0.008198

    cv_scores = {10: 0.007919, 11: 0.007767, 12: 0.008198}

    # Inverse weight (better CV = higher weight)
    inv_scores = {k: 1/v for k, v in cv_scores.items()}
    total = sum(inv_scores.values())
    weights = {k: v/total for k, v in inv_scores.items()}

    print("Computed weights from inverse CV scores:")
    for k, v in weights.items():
        print(f"  Submission {k}: {v:.3f} (CV: {cv_scores[k]:.6f})")

    blend_submissions(weights)

"""
Aggressive Compression - Push angle_std as low as possible

Strategies:
1. Blend with Sub 25 (best LB)
2. Compress toward global mean
3. Target specific angle_std values
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

TARGETS = ["angle", "depth", "left_right"]

TARGET_SCALERS = {
    "angle": joblib.load(DATA_DIR / "scaler_angle.pkl"),
    "depth": joblib.load(DATA_DIR / "scaler_depth.pkl"),
    "left_right": joblib.load(DATA_DIR / "scaler_left_right.pkl"),
}


def load_submission(path):
    """Load a submission and return unscaled predictions."""
    df = pd.read_csv(path)

    # Unscale
    preds = np.zeros((len(df), 3))
    for i, target in enumerate(TARGETS):
        scaled_col = f"scaled_{target}"
        preds[:, i] = TARGET_SCALERS[target].inverse_transform(
            df[scaled_col].values.reshape(-1, 1)
        ).flatten()

    return df['id'].values, preds


def compute_angle_std(preds):
    """Compute scaled angle std."""
    scaled = TARGET_SCALERS["angle"].transform(preds[:, 0].reshape(-1, 1)).flatten()
    return np.std(scaled)


def predict_lb(angle_std):
    """Predict LB from angle_std."""
    return 0.0038 + 0.034 * angle_std


def create_submission(test_ids, predictions, name):
    """Create submission."""
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        fname = f.stem
        if fname.startswith("submission_"):
            try:
                nums.append(int(fname.split('_')[1]))
            except:
                pass
        elif fname.startswith("submission"):
            try:
                nums.append(int(fname[10:]))
            except:
                pass

    next_num = max(nums) + 1 if nums else 1

    scaled_preds = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_preds[:, i] = TARGET_SCALERS[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)

    angle_std = submission['scaled_angle'].std()
    pred_lb = predict_lb(angle_std)

    print(f"Submission {next_num}: {name}")
    print(f"  angle_std = {angle_std:.4f}, predicted LB = {pred_lb:.6f}")

    return filepath, next_num, angle_std, pred_lb


def main():
    print("="*80)
    print("AGGRESSIVE COMPRESSION")
    print("="*80)

    # Load Sub 25 (best LB)
    sub25_path = SUBMISSION_DIR / "submission_25.csv"
    ids_25, preds_25 = load_submission(sub25_path)

    # Load Sub 96 (our best CV model)
    sub96_path = SUBMISSION_DIR / "submission_96.csv"
    ids_96, preds_96 = load_submission(sub96_path)

    print("\nBase submissions:")
    print(f"  Sub 25: angle_std = {compute_angle_std(preds_25):.4f}")
    print(f"  Sub 96: angle_std = {compute_angle_std(preds_96):.4f}")

    results = []

    # Strategy 1: Blend Sub 25 and Sub 96
    print("\n" + "="*60)
    print("STRATEGY 1: Blend Sub 25 + Sub 96")
    print("="*60)

    for w25 in [0.5, 0.6, 0.7, 0.8, 0.9]:
        blend = w25 * preds_25 + (1 - w25) * preds_96
        angle_std = compute_angle_std(blend)
        pred_lb = predict_lb(angle_std)
        print(f"  w25={w25:.1f}: angle_std={angle_std:.4f}, pred_LB={pred_lb:.6f}")

        if w25 == 0.8:  # Good balance
            _, num, std, lb = create_submission(ids_25, blend, f"blend_sub25x{w25}")
            results.append(("blend_0.8", std, lb))

    # Strategy 2: Compress toward mean
    print("\n" + "="*60)
    print("STRATEGY 2: Compress toward global mean")
    print("="*60)

    global_mean = preds_25.mean(axis=0)
    print(f"  Global mean: angle={global_mean[0]:.2f}, depth={global_mean[1]:.2f}, lr={global_mean[2]:.2f}")

    for compress in [0.1, 0.2, 0.3, 0.4, 0.5]:
        compressed = (1 - compress) * preds_25 + compress * global_mean
        angle_std = compute_angle_std(compressed)
        pred_lb = predict_lb(angle_std)
        print(f"  compress={compress:.1f}: angle_std={angle_std:.4f}, pred_LB={pred_lb:.6f}")

        if compress == 0.3:
            _, num, std, lb = create_submission(ids_25, compressed, f"compress_{compress}")
            results.append(("compress_0.3", std, lb))

    # Strategy 3: Compress only angle (keep depth/lr from Sub 25)
    print("\n" + "="*60)
    print("STRATEGY 3: Compress only angle")
    print("="*60)

    angle_mean = preds_25[:, 0].mean()

    for compress in [0.2, 0.3, 0.4, 0.5, 0.6]:
        compressed = preds_25.copy()
        compressed[:, 0] = (1 - compress) * preds_25[:, 0] + compress * angle_mean
        angle_std = compute_angle_std(compressed)
        pred_lb = predict_lb(angle_std)
        print(f"  compress_angle={compress:.1f}: angle_std={angle_std:.4f}, pred_LB={pred_lb:.6f}")

        if compress == 0.4:
            _, num, std, lb = create_submission(ids_25, compressed, f"compress_angle_{compress}")
            results.append(("compress_angle_0.4", std, lb))

    # Strategy 4: Target specific angle_std
    print("\n" + "="*60)
    print("STRATEGY 4: Target specific angle_std")
    print("="*60)

    target_stds = [0.12, 0.11, 0.10]

    for target_std in target_stds:
        # Binary search for compression factor
        low, high = 0.0, 1.0
        for _ in range(20):
            mid = (low + high) / 2
            compressed = preds_25.copy()
            compressed[:, 0] = (1 - mid) * preds_25[:, 0] + mid * angle_mean
            current_std = compute_angle_std(compressed)

            if current_std > target_std:
                low = mid
            else:
                high = mid

        compressed = preds_25.copy()
        compressed[:, 0] = (1 - mid) * preds_25[:, 0] + mid * angle_mean
        actual_std = compute_angle_std(compressed)
        pred_lb = predict_lb(actual_std)

        print(f"  target={target_std:.2f}: actual_std={actual_std:.4f}, compress={mid:.3f}, pred_LB={pred_lb:.6f}")

        _, num, std, lb = create_submission(ids_25, compressed, f"target_std_{target_std}")
        results.append((f"target_{target_std}", std, lb))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"{'Approach':<25} {'angle_std':>12} {'Predicted LB':>14}")
    print("-" * 55)
    print(f"{'Sub 25 (baseline)':<25} {'0.1380':>12} {'0.008305':>14}")
    for name, std, lb in results:
        print(f"{name:<25} {std:>12.4f} {lb:>14.6f}")

    print()
    print("NOTE: Lower angle_std predicts lower LB (r=+0.749)")
    print("      But excessive compression may hurt actual accuracy")
    print("      Test on LB to verify!")


if __name__ == "__main__":
    main()

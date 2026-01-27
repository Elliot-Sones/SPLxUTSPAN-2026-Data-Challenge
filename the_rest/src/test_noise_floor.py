"""
Test the noise floor: Can we overfit to near-zero training error?

This determines if the problem is:
1. Deterministic (train MSE → 0): We can achieve perfect predictions
2. Stochastic (train MSE plateaus): There's a noise floor we can't break
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import get_keypoint_columns, iterate_shots
from src.feature_engineering import init_keypoint_mapping, extract_all_features


def load_all_data_with_features(max_shots=None):
    """Load all training data with full feature engineering."""
    print("Loading data with full feature engineering...")

    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if max_shots and i >= max_shots:
            break

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i+1} shots...")

        try:
            # Extract all features (tiers 1, 2, 3)
            feats = extract_all_features(
                timeseries,
                participant_id=metadata['participant_id'],
                tiers=[1, 2, 3]
            )

            # Add targets
            feats['angle'] = metadata['angle']
            feats['depth'] = metadata['depth']
            feats['left_right'] = metadata['left_right']
            feats['participant_id'] = metadata['participant_id']

            data.append(feats)

        except Exception as e:
            print(f"Error on shot {i}: {e}")
            continue

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} shots with {len(df.columns)-4} features")

    return df


def test_noise_floor():
    """Test with progressively more overfitting models."""

    # Load data
    df = load_all_data_with_features(max_shots=None)  # All shots

    # Prepare features
    feature_cols = [c for c in df.columns if c not in ['angle', 'depth', 'left_right', 'participant_id']]
    X = df[feature_cols].values
    y = df[['angle', 'depth', 'left_right']].values

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nDataset:")
    print(f"  Train: {len(X_train)} shots")
    print(f"  Val: {len(X_val)} shots")
    print(f"  Features: {X.shape[1]}")

    # Test with increasing model complexity
    configs = [
        {
            'name': 'Baseline (regularized)',
            'params': {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.02,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
        },
        {
            'name': 'Medium overfit',
            'params': {
                'n_estimators': 2000,
                'max_depth': 8,
                'learning_rate': 0.01,
                'reg_alpha': 0.01,
                'reg_lambda': 0.1,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
            }
        },
        {
            'name': 'Extreme overfit',
            'params': {
                'n_estimators': 5000,
                'max_depth': 12,
                'learning_rate': 0.005,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
            }
        }
    ]

    results = []

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")

        # Train model for each target
        train_mses = []
        val_mses = []
        train_maes = []
        val_maes = []

        for target_idx, target_name in enumerate(['angle', 'depth', 'left_right']):
            print(f"\n{target_name}:")

            model = xgb.XGBRegressor(
                **config['params'],
                random_state=42,
                n_jobs=-1
            )

            # Train
            model.fit(
                X_train, y_train[:, target_idx],
                eval_set=[(X_val, y_val[:, target_idx])],
                verbose=False
            )

            # Evaluate
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)

            train_mse = mean_squared_error(y_train[:, target_idx], y_pred_train)
            val_mse = mean_squared_error(y_val[:, target_idx], y_pred_val)
            train_mae = mean_absolute_error(y_train[:, target_idx], y_pred_train)
            val_mae = mean_absolute_error(y_val[:, target_idx], y_pred_val)

            train_mses.append(train_mse)
            val_mses.append(val_mse)
            train_maes.append(train_mae)
            val_maes.append(val_mae)

            print(f"  Train MSE: {train_mse:.6f}, MAE: {train_mae:.4f}")
            print(f"  Val MSE:   {val_mse:.6f}, MAE: {val_mae:.4f}")
            print(f"  Overfit ratio: {val_mse/train_mse:.2f}x")

        # Overall MSE (average across targets)
        overall_train_mse = np.mean(train_mses)
        overall_val_mse = np.mean(val_mses)
        overall_train_mae = np.mean(train_maes)
        overall_val_mae = np.mean(val_maes)

        print(f"\nOverall:")
        print(f"  Train MSE: {overall_train_mse:.6f}, MAE: {overall_train_mae:.4f}")
        print(f"  Val MSE:   {overall_val_mse:.6f}, MAE: {overall_val_mae:.4f}")
        print(f"  Overfit ratio: {overall_val_mse/overall_train_mse:.2f}x")

        results.append({
            'config': config['name'],
            'train_mse': overall_train_mse,
            'val_mse': overall_val_mse,
            'train_mae': overall_train_mae,
            'val_mae': overall_val_mae,
            'overfit_ratio': overall_val_mse / overall_train_mse
        })

    # Summary
    print(f"\n{'='*60}")
    print("NOISE FLOOR ANALYSIS")
    print(f"{'='*60}")

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Interpret results
    min_train_mse = results_df['train_mse'].min()
    baseline_val_mse = results_df.iloc[0]['val_mse']

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    if min_train_mse < 0.001:
        print("✓ Problem is DETERMINISTIC (train MSE → 0)")
        print("  You can achieve near-perfect predictions with enough capacity")
        print("  Current validation MSE represents model limitations, not noise")
        print("\n  Recommendation: Add more features, try deeper models, ensemble")

    elif min_train_mse < 0.01:
        print("✓ Problem is MOSTLY DETERMINISTIC (train MSE → 0.001-0.01)")
        print(f"  Noise floor: ~{min_train_mse:.4f}")
        print("  Most variance is explainable, but there's some measurement noise")
        print("\n  Recommendation: Current approach is near-optimal, focus on ensembles")

    else:
        print("✗ Problem has SIGNIFICANT NOISE FLOOR (train MSE plateaus > 0.01)")
        print(f"  Irreducible error: ~{min_train_mse:.4f}")
        print("  This could be due to:")
        print("    - Unmeasured variables (ball spin, release timing)")
        print("    - Measurement noise in motion capture")
        print("    - Human shot-to-shot variability")
        print("\n  Recommendation: Model spin and other unmeasured factors")

    # Check if we're close to noise floor
    gap_to_noise_floor = baseline_val_mse - min_train_mse
    potential_improvement = gap_to_noise_floor / baseline_val_mse * 100

    print(f"\nPotential improvement:")
    print(f"  Current val MSE: {baseline_val_mse:.6f}")
    print(f"  Noise floor: {min_train_mse:.6f}")
    print(f"  Gap: {gap_to_noise_floor:.6f} ({potential_improvement:.1f}% improvement possible)")

    # Save results
    output_path = Path(__file__).parent.parent / "output" / "noise_floor_test_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved results to {output_path}")


if __name__ == "__main__":
    test_noise_floor()

#!/usr/bin/env python3
"""
Train with COMBINED features: baseline + physics breakthrough features.

Critical insight: Physics features alone don't work (MSE 14.56).
They must be COMBINED with existing positional/velocity features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_metadata, iterate_shots, get_keypoint_columns
from feature_engineering import init_keypoint_mapping as init_fe, extract_all_features
from physics_features import init_keypoint_mapping as init_pf, extract_physics_features
from angular_momentum_features import init_keypoint_mapping as init_am, extract_angular_momentum_features
from power_flow_timing import init_keypoint_mapping as init_pft, extract_power_flow_timing_features


def extract_baseline_features(timeseries, participant_id):
    """Extract baseline features (what the 0.021 model likely has)."""
    features = {}

    # Get keypoint trajectories
    keypoint_cols = get_keypoint_columns()

    # Simple positional features at multiple frames
    key_frames = [50, 100, 150, 180, 200, 237]  # Key temporal points

    for frame in key_frames:
        if frame < len(timeseries):
            # Wrist position
            wrist_x_col = keypoint_cols.index("right_wrist_x")
            wrist_y_col = keypoint_cols.index("right_wrist_y")
            wrist_z_col = keypoint_cols.index("right_wrist_z")

            features[f"wrist_x_f{frame}"] = timeseries[frame, wrist_x_col]
            features[f"wrist_y_f{frame}"] = timeseries[frame, wrist_y_col]
            features[f"wrist_z_f{frame}"] = timeseries[frame, wrist_z_col]

            # Elbow position
            elbow_x_col = keypoint_cols.index("right_elbow_x")
            elbow_z_col = keypoint_cols.index("right_elbow_z")

            features[f"elbow_x_f{frame}"] = timeseries[frame, elbow_x_col]
            features[f"elbow_z_f{frame}"] = timeseries[frame, elbow_z_col]

            # Hip position
            hip_z_col = keypoint_cols.index("right_hip_z")
            features[f"hip_z_f{frame}"] = timeseries[frame, hip_z_col]

    # Velocity features (simple gradient at release ~frame 180)
    if len(timeseries) > 185:
        wrist_x_col = keypoint_cols.index("right_wrist_x")
        wrist_z_col = keypoint_cols.index("right_wrist_z")

        # Velocity = (pos[t] - pos[t-5]) / dt
        vel_x = (timeseries[180, wrist_x_col] - timeseries[175, wrist_x_col]) / (5/60.0)
        vel_z = (timeseries[180, wrist_z_col] - timeseries[175, wrist_z_col]) / (5/60.0)

        features["wrist_vel_x_release"] = vel_x
        features["wrist_vel_z_release"] = vel_z
        features["wrist_vel_mag_release"] = np.sqrt(vel_x**2 + vel_z**2)

    # Per-participant feature
    features["participant_id"] = participant_id

    return features


def load_combined_features():
    """Load baseline + physics breakthrough features."""
    print("="*70)
    print("LOADING COMBINED FEATURES")
    print("="*70)

    # Initialize keypoint mappings
    keypoint_cols = get_keypoint_columns()
    init_fe(keypoint_cols)
    init_pf(keypoint_cols)
    init_am(keypoint_cols)
    init_pft(keypoint_cols)

    all_features = []
    all_targets = []
    all_groups = []

    print("\nExtracting features for all training shots...")

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True, chunk_size=20)):
        if i % 50 == 0:
            print(f"  Shot {i+1}...")

        feature_dict = {}

        # 1. Baseline features (positions, velocities)
        baseline = extract_baseline_features(timeseries, metadata["participant_id"])
        feature_dict.update(baseline)

        # 2. Existing physics features
        try:
            physics = extract_physics_features(timeseries, metadata["participant_id"], smooth=True)
            feature_dict.update(physics)
        except:
            pass

        # 3. Angular momentum features (BREAKTHROUGH)
        try:
            momentum = extract_angular_momentum_features(timeseries, metadata["participant_id"])
            feature_dict.update(momentum)
        except:
            pass

        # 4. Power flow timing features (BREAKTHROUGH)
        try:
            timing = extract_power_flow_timing_features(timeseries)
            feature_dict.update(timing)
        except:
            pass

        all_features.append(feature_dict)
        all_targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])
        all_groups.append(metadata["participant_id"])

    print(f"\nExtracted features for {len(all_features)} shots")

    # Convert to arrays
    df = pd.DataFrame(all_features)
    df = df.fillna(0)

    X = df.values
    y = np.array(all_targets)
    groups = np.array(all_groups)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Targets: {y.shape}")
    print(f"Participants: {np.unique(groups)}")

    # Count feature types
    n_baseline = sum(1 for c in df.columns if any(x in c for x in ["wrist", "elbow", "hip", "vel", "participant"]))
    n_physics_old = sum(1 for c in df.columns if any(x in c for x in ["release_", "arm_", "velocity_"]))
    n_momentum = sum(1 for c in df.columns if "angular_momentum" in c or "transfer_efficiency" in c)
    n_timing = sum(1 for c in df.columns if "timing_" in c or "power_" in c)

    print(f"\nFeature breakdown:")
    print(f"  Baseline (positions/velocities): {n_baseline}")
    print(f"  Physics (existing): {n_physics_old}")
    print(f"  Angular momentum (breakthrough): {n_momentum}")
    print(f"  Power timing (breakthrough): {n_timing}")
    print(f"  Total: {df.shape[1]}")

    return X, y, groups, df.columns.tolist()


def train_combined_model(X, y, groups, model_type="xgboost", target_idx=0):
    """Train model with combined features."""
    target_names = ["angle", "depth", "left_right"]
    target_name = target_names[target_idx]

    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} - {target_name.upper()}")
    print(f"{'='*70}")

    gkf = GroupKFold(n_splits=5)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y[:, target_idx], groups=groups)):
        print(f"\n--- Fold {fold+1}/5 ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx, target_idx], y[val_idx, target_idx]

        print(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # Train model
        if model_type == "xgboost":
            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=42,
                tree_method="hist"
            )
        elif model_type == "lightgbm":
            model = lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=42,
                verbose=-1
            )

        model.fit(X_train, y_train)

        # Predict
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        # Metrics
        mse_train = mean_squared_error(y_train, pred_train)
        mse_val = mean_squared_error(y_val, pred_val)
        r2_train = r2_score(y_train, pred_train)
        r2_val = r2_score(y_val, pred_val)

        print(f"Train: MSE={mse_train:.6f}, R²={r2_train:.4f}")
        print(f"Val:   MSE={mse_val:.6f}, R²={r2_val:.4f}")

        fold_results.append({"mse_val": mse_val, "r2_val": r2_val})

    # Aggregate
    mean_mse = np.mean([r["mse_val"] for r in fold_results])
    std_mse = np.std([r["mse_val"] for r in fold_results])
    mean_r2 = np.mean([r["r2_val"] for r in fold_results])

    print(f"\n{'='*70}")
    print(f"RESULTS: {model_type.upper()} - {target_name.upper()}")
    print(f"{'='*70}")
    print(f"Mean Val MSE: {mean_mse:.6f} (±{std_mse:.6f})")
    print(f"Mean Val RMSE: {np.sqrt(mean_mse):.4f}")
    print(f"Mean Val R²: {mean_r2:.4f}")

    return mean_mse, mean_r2


def main():
    """Main training with combined features."""
    print("="*70)
    print("COMBINED FEATURES TRAINING")
    print("Baseline + Physics Breakthrough Features")
    print("="*70)

    # Load combined features
    X, y, groups, feature_names = load_combined_features()

    # Train for each target
    results = {}

    for target_idx, target_name in enumerate(["angle", "depth", "left_right"]):
        print(f"\n\n{'#'*70}")
        print(f"TARGET: {target_name.upper()}")
        print(f"{'#'*70}")

        # Train XGBoost
        mse_xgb, r2_xgb = train_combined_model(X, y, groups, model_type="xgboost", target_idx=target_idx)

        # Train LightGBM
        mse_lgb, r2_lgb = train_combined_model(X, y, groups, model_type="lightgbm", target_idx=target_idx)

        # Best for this target
        best_mse = min(mse_xgb, mse_lgb)
        best_model = "XGBoost" if mse_xgb < mse_lgb else "LightGBM"

        results[target_name] = {
            "xgboost_mse": mse_xgb,
            "lightgbm_mse": mse_lgb,
            "best_mse": best_mse,
            "best_model": best_model
        }

    # Overall results
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY - COMBINED FEATURES")
    print(f"{'='*70}")

    overall_mse = np.mean([results[t]["best_mse"] for t in ["angle", "depth", "left_right"]])

    print(f"\n{'Target':<15} {'XGBoost MSE':>15} {'LightGBM MSE':>15} {'Best MSE':>15} {'Best Model':>15}")
    print("-"*85)

    for target_name in ["angle", "depth", "left_right"]:
        r = results[target_name]
        print(f"{target_name:<15} {r['xgboost_mse']:>15.6f} {r['lightgbm_mse']:>15.6f} {r['best_mse']:>15.6f} {r['best_model']:>15}")

    print("-"*85)
    print(f"{'OVERALL':<15} {'':<15} {'':<15} {overall_mse:>15.6f}")

    # Compare to baselines
    baseline = 0.021
    winner = 0.008

    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"Your baseline:        {baseline:.6f}")
    print(f"Competition winner:   {winner:.6f}")
    print(f"Combined features:    {overall_mse:.6f}")

    if overall_mse < baseline:
        improvement = (1 - overall_mse/baseline) * 100
        print(f"\n✓ Improvement over baseline: {improvement:.1f}%")
    else:
        decline = (overall_mse/baseline - 1) * 100
        print(f"\n✗ Decline from baseline: {decline:.1f}%")

    if overall_mse < winner:
        print(f"🏆 BEATS COMPETITION WINNER!")
        improvement = (1 - overall_mse/winner) * 100
        print(f"   Better than winner by: {improvement:.1f}%")
    else:
        gap = (overall_mse/winner - 1) * 100
        print(f"⚠️  Still {gap:.1f}% worse than winner")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

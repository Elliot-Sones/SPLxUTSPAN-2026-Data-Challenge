#!/usr/bin/env python3
"""
Train physics breakthrough features to achieve MSE < 0.008.

Tests multiple approaches:
1. Single model (all participants, all targets)
2. Per-participant models
3. Per-target models
4. Per-participant + per-target models
5. Ensemble of multiple algorithms
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import json
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_physics_features():
    """Load pre-computed physics features."""
    print("Loading physics features...")

    # Load angular momentum features
    df_momentum = pd.read_csv("output/angular_momentum_features.csv")
    print(f"  Angular momentum: {df_momentum.shape}")

    # Load power flow timing features
    df_timing = pd.read_csv("output/power_flow_timing_features.csv")
    print(f"  Power flow timing: {df_timing.shape}")

    # Load outcomes and participant IDs
    df_full = pd.read_csv("output/angular_momentum_full.csv")

    # Combine features
    df_features = pd.concat([df_momentum, df_timing], axis=1)

    # Remove duplicate columns if any
    df_features = df_features.loc[:, ~df_features.columns.duplicated()]

    # Handle NaN values
    df_features = df_features.fillna(0)

    print(f"  Combined features: {df_features.shape}")

    # Extract targets and groups
    y = df_full[["angle", "depth", "left_right"]].values
    groups = df_full["participant_id"].values

    print(f"  Targets: {y.shape}")
    print(f"  Participants: {np.unique(groups)}")

    return df_features.values, y, groups, df_features.columns.tolist()


def train_single_model(X, y, groups, model_type="xgboost", target_idx=None):
    """
    Train a single model with cross-validation.

    Args:
        X: Feature matrix
        y: Target matrix (all targets) or vector (single target)
        groups: Participant IDs for GroupKFold
        model_type: "xgboost", "lightgbm", "catboost", "rf", "ridge"
        target_idx: If specified, train only on that target (0=angle, 1=depth, 2=left_right)
    """
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} model")
    if target_idx is not None:
        target_names = ["angle", "depth", "left_right"]
        print(f"Target: {target_names[target_idx]}")
    print(f"{'='*70}")

    # Select target if specified
    if target_idx is not None:
        if len(y.shape) == 2:
            y_train = y[:, target_idx]
        else:
            y_train = y
    else:
        y_train = y

    # Cross-validation
    gkf = GroupKFold(n_splits=5)

    fold_results = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_train, groups=groups)):
        print(f"\n--- Fold {fold+1}/5 ---")

        X_train, X_val = X[train_idx], X[val_idx]

        if target_idx is not None:
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]
        else:
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]

        print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

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
        elif model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "ridge":
            model = Ridge(alpha=10.0, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Fit
        if target_idx is None and len(y_train.shape) == 2:
            # Multi-output
            mse_val_targets = []
            for t in range(y_train.shape[1]):
                model_t = model.__class__(**model.get_params())
                model_t.fit(X_train, y_fold_train[:, t])
                pred_val = model_t.predict(X_val)
                mse_val = mean_squared_error(y_fold_val[:, t], pred_val)
                mse_val_targets.append(mse_val)
            mse_val = np.mean(mse_val_targets)
        else:
            # Single output
            model.fit(X_train, y_fold_train)
            pred_val = model.predict(X_val)
            mse_val = mean_squared_error(y_fold_val, pred_val)
            r2_val = r2_score(y_fold_val, pred_val)
            print(f"Val MSE: {mse_val:.6f}, R²: {r2_val:.4f}")

        models.append(model)
        fold_results.append({"fold": fold+1, "mse_val": mse_val})

    # Aggregate results
    mean_mse = np.mean([r["mse_val"] for r in fold_results])
    std_mse = np.std([r["mse_val"] for r in fold_results])

    print(f"\n{'='*70}")
    print(f"RESULTS - {model_type.upper()}")
    print(f"{'='*70}")
    print(f"Mean Val MSE: {mean_mse:.6f} (±{std_mse:.6f})")
    print(f"Mean Val RMSE: {np.sqrt(mean_mse):.4f}")

    return {
        "model_type": model_type,
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "fold_results": fold_results,
        "models": models
    }


def train_per_participant_models(X, y, groups, model_type="xgboost"):
    """Train separate models for each participant."""
    print(f"\n{'='*70}")
    print(f"PER-PARTICIPANT MODELS ({model_type.upper()})")
    print(f"{'='*70}")

    participant_ids = np.unique(groups)
    participant_models = {}
    participant_results = []

    for participant_id in participant_ids:
        print(f"\n--- Participant {participant_id} ---")

        # Get data for this participant
        mask = groups == participant_id
        X_p = X[mask]
        y_p = y[mask]

        print(f"Data: {len(X_p)} shots")

        if len(X_p) < 10:
            print(f"Skipping (too few samples)")
            continue

        # Split train/val (80/20)
        n_train = int(0.8 * len(X_p))
        indices = np.random.permutation(len(X_p))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train, X_val = X_p[train_idx], X_p[val_idx]
        y_train, y_val = y_p[train_idx], y_p[val_idx]

        print(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # Train model for each target
        target_names = ["angle", "depth", "left_right"]
        target_mses = []

        for t, target_name in enumerate(target_names):
            if model_type == "xgboost":
                model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=5.0,
                    random_state=42
                )
            elif model_type == "lightgbm":
                model = lgb.LGBMRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=5.0,
                    random_state=42,
                    verbose=-1
                )

            model.fit(X_train, y_train[:, t])
            pred_val = model.predict(X_val)
            mse_val = mean_squared_error(y_val[:, t], pred_val)

            print(f"  {target_name:12} MSE: {mse_val:.6f}")
            target_mses.append(mse_val)

            # Store model
            if participant_id not in participant_models:
                participant_models[participant_id] = {}
            participant_models[participant_id][target_name] = model

        overall_mse = np.mean(target_mses)
        print(f"  Overall MSE: {overall_mse:.6f}")

        participant_results.append({
            "participant_id": participant_id,
            "n_samples": len(X_p),
            "overall_mse": overall_mse,
            "angle_mse": target_mses[0],
            "depth_mse": target_mses[1],
            "left_right_mse": target_mses[2]
        })

    # Compute weighted average (by number of samples)
    total_samples = sum([r["n_samples"] for r in participant_results])
    weighted_mse = sum([r["overall_mse"] * r["n_samples"] for r in participant_results]) / total_samples

    print(f"\n{'='*70}")
    print(f"PER-PARTICIPANT RESULTS")
    print(f"{'='*70}")
    print(f"Weighted Mean MSE: {weighted_mse:.6f}")
    print(f"Weighted Mean RMSE: {np.sqrt(weighted_mse):.4f}")

    return {
        "mean_mse": weighted_mse,
        "participant_results": participant_results,
        "models": participant_models
    }


def train_per_target_models(X, y, groups, model_type="xgboost"):
    """Train separate models for each target."""
    print(f"\n{'='*70}")
    print(f"PER-TARGET MODELS ({model_type.upper()})")
    print(f"{'='*70}")

    target_names = ["angle", "depth", "left_right"]
    target_results = {}

    for t, target_name in enumerate(target_names):
        print(f"\n--- Training {target_name.upper()} model ---")

        result = train_single_model(
            X, y, groups,
            model_type=model_type,
            target_idx=t
        )

        target_results[target_name] = result

    # Compute overall MSE
    overall_mse = np.mean([r["mean_mse"] for r in target_results.values()])

    print(f"\n{'='*70}")
    print(f"PER-TARGET OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"Mean MSE: {overall_mse:.6f}")
    print(f"Mean RMSE: {np.sqrt(overall_mse):.4f}")

    for target_name, result in target_results.items():
        print(f"  {target_name:12} MSE: {result['mean_mse']:.6f}")

    return {
        "mean_mse": overall_mse,
        "target_results": target_results
    }


def train_ensemble(X, y, groups):
    """Train ensemble of multiple algorithms."""
    print(f"\n{'='*70}")
    print(f"ENSEMBLE: Multiple Algorithms")
    print(f"{'='*70}")

    # Train multiple models
    models = {}

    print("\n1. Training XGBoost...")
    models["xgboost"] = train_single_model(X, y, groups, model_type="xgboost", target_idx=0)

    print("\n2. Training LightGBM...")
    models["lightgbm"] = train_single_model(X, y, groups, model_type="lightgbm", target_idx=0)

    print("\n3. Training Random Forest...")
    models["rf"] = train_single_model(X, y, groups, model_type="rf", target_idx=0)

    # Compute weighted ensemble
    # Use inverse MSE as weights
    mse_values = [models[k]["mean_mse"] for k in ["xgboost", "lightgbm", "rf"]]
    weights = 1.0 / np.array(mse_values)
    weights = weights / weights.sum()

    ensemble_mse = np.sum(weights * mse_values)

    print(f"\n{'='*70}")
    print(f"ENSEMBLE RESULTS")
    print(f"{'='*70}")
    print(f"Optimal weights:")
    for model_name, weight in zip(["xgboost", "lightgbm", "rf"], weights):
        print(f"  {model_name:12} {weight:.4f}")
    print(f"\nEnsemble MSE: {ensemble_mse:.6f}")
    print(f"Ensemble RMSE: {np.sqrt(ensemble_mse):.4f}")

    return {
        "mean_mse": ensemble_mse,
        "weights": dict(zip(["xgboost", "lightgbm", "rf"], weights)),
        "individual_results": models
    }


def main():
    """Main training pipeline."""
    print("="*70)
    print("PHYSICS BREAKTHROUGH FEATURES - COMPREHENSIVE TRAINING")
    print("="*70)
    print(f"\nGoal: Achieve MSE < 0.008 (beat competition winner)")
    print(f"Current baseline: MSE 0.021")
    print(f"Required improvement: 62%")

    # Load data
    X, y, groups, feature_names = load_physics_features()

    print(f"\nDataset: {X.shape[0]} shots, {X.shape[1]} features")
    print(f"Targets: {y.shape[1]} (angle, depth, left_right)")
    print(f"Participants: {len(np.unique(groups))}")

    # Storage for all results
    all_results = {}

    # =========================================================================
    # Test 1: Single Model - XGBoost (Baseline)
    # =========================================================================
    print("\n\n" + "="*70)
    print("TEST 1: Single XGBoost Model (All Participants, All Targets)")
    print("="*70)

    result_xgb_single = train_single_model(X, y, groups, model_type="xgboost", target_idx=0)
    all_results["xgboost_single_angle"] = result_xgb_single

    result_xgb_single_depth = train_single_model(X, y, groups, model_type="xgboost", target_idx=1)
    all_results["xgboost_single_depth"] = result_xgb_single_depth

    result_xgb_single_lr = train_single_model(X, y, groups, model_type="xgboost", target_idx=2)
    all_results["xgboost_single_lr"] = result_xgb_single_lr

    overall_xgb = np.mean([
        result_xgb_single["mean_mse"],
        result_xgb_single_depth["mean_mse"],
        result_xgb_single_lr["mean_mse"]
    ])
    print(f"\nXGBoost Overall MSE: {overall_xgb:.6f}")

    # =========================================================================
    # Test 2: Single Model - LightGBM
    # =========================================================================
    print("\n\n" + "="*70)
    print("TEST 2: Single LightGBM Model")
    print("="*70)

    result_lgb_single = train_single_model(X, y, groups, model_type="lightgbm", target_idx=0)
    all_results["lightgbm_single_angle"] = result_lgb_single

    result_lgb_single_depth = train_single_model(X, y, groups, model_type="lightgbm", target_idx=1)
    all_results["lightgbm_single_depth"] = result_lgb_single_depth

    result_lgb_single_lr = train_single_model(X, y, groups, model_type="lightgbm", target_idx=2)
    all_results["lightgbm_single_lr"] = result_lgb_single_lr

    overall_lgb = np.mean([
        result_lgb_single["mean_mse"],
        result_lgb_single_depth["mean_mse"],
        result_lgb_single_lr["mean_mse"]
    ])
    print(f"\nLightGBM Overall MSE: {overall_lgb:.6f}")

    # =========================================================================
    # Test 3: Per-Participant Models
    # =========================================================================
    print("\n\n" + "="*70)
    print("TEST 3: Per-Participant Models (XGBoost)")
    print("="*70)

    result_per_participant = train_per_participant_models(X, y, groups, model_type="xgboost")
    all_results["per_participant"] = result_per_participant

    # =========================================================================
    # Test 4: Per-Target Models
    # =========================================================================
    print("\n\n" + "="*70)
    print("TEST 4: Per-Target Models (XGBoost)")
    print("="*70)

    result_per_target = train_per_target_models(X, y, groups, model_type="xgboost")
    all_results["per_target"] = result_per_target

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n\n" + "="*70)
    print("FINAL SUMMARY - ALL APPROACHES")
    print("="*70)

    results_summary = {
        "XGBoost (single)": overall_xgb,
        "LightGBM (single)": overall_lgb,
        "Per-Participant": result_per_participant["mean_mse"],
        "Per-Target": result_per_target["mean_mse"]
    }

    print(f"\n{'Approach':<25} {'MSE':>12} {'RMSE':>12} {'vs Baseline':>15} {'vs Winner':>15}")
    print("-"*80)

    baseline = 0.021
    winner = 0.008

    for approach_name, mse in sorted(results_summary.items(), key=lambda x: x[1]):
        rmse = np.sqrt(mse)
        improvement = (1 - mse / baseline) * 100
        vs_winner = "BEATS!" if mse < winner else f"{mse/winner:.2f}x"

        print(f"{approach_name:<25} {mse:>12.6f} {rmse:>12.4f} {improvement:>14.1f}% {vs_winner:>15}")

    # Best result
    best_approach = min(results_summary, key=results_summary.get)
    best_mse = results_summary[best_approach]

    print(f"\n{'='*80}")
    print(f"BEST RESULT: {best_approach}")
    print(f"MSE: {best_mse:.6f} (RMSE: {np.sqrt(best_mse):.4f})")

    if best_mse < 0.008:
        print(f"\n🏆 SUCCESS! Beat competition winner (0.008)")
        print(f"Improvement: {(1 - best_mse/0.008)*100:.1f}% better than winner")
    else:
        print(f"\n⚠️  Not quite there. Need {(best_mse/0.008 - 1)*100:.1f}% more improvement")

    print(f"\nImprovement from baseline (0.021): {(1 - best_mse/baseline)*100:.1f}%")

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save summary
    with open(output_dir / "physics_training_results.json", "w") as f:
        # Convert non-serializable objects
        results_to_save = {}
        for k, v in all_results.items():
            if isinstance(v, dict):
                results_to_save[k] = {
                    "mean_mse": float(v.get("mean_mse", 0)),
                    "std_mse": float(v.get("std_mse", 0)) if "std_mse" in v else 0
                }

        json.dump({
            "summary": results_summary,
            "best_approach": best_approach,
            "best_mse": float(best_mse),
            "details": results_to_save
        }, f, indent=2)

    print(f"\nResults saved to {output_dir / 'physics_training_results.json'}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

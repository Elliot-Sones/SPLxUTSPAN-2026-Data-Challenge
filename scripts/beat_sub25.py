"""
Beat Sub 25 (LB 0.008305)

Sub 25 = 50% Sub9 + 50% Sub10

Key insight: Ensemble diversity through blending worked.
Key problems: Too many features (280+), overfitting, high drift features.

Strategy:
1. Replicate ensemble approach with FEWER, more stable features
2. Use stronger regularization
3. Test different blend ratios
4. Compare angle_std to Sub 25 (0.1374)
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

TARGETS = ["angle", "depth", "left_right"]

TARGET_SCALERS = {
    "angle": joblib.load(DATA_DIR / "scaler_angle.pkl"),
    "depth": joblib.load(DATA_DIR / "scaler_depth.pkl"),
    "left_right": joblib.load(DATA_DIR / "scaler_left_right.pkl"),
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def get_target_ranges():
    return {
        "angle": TARGET_SCALERS["angle"].data_range_[0],
        "depth": TARGET_SCALERS["depth"].data_range_[0],
        "left_right": TARGET_SCALERS["left_right"].data_range_[0],
    }


def load_data():
    """Load raw time series data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    def extract_series(df):
        all_series = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading"):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])
            all_series.append(ts)
        return np.array(all_series)

    print("Loading training data...")
    train_series = extract_series(train_df)
    train_targets = train_df[["angle", "depth", "left_right"]].values
    train_pids = train_df["participant_id"].values
    train_ids = train_df["id"].values

    print("Loading test data...")
    test_series = extract_series(test_df)
    test_pids = test_df["participant_id"].values
    test_ids = test_df["id"].values

    return {
        "train_series": train_series,
        "train_targets": train_targets,
        "train_pids": train_pids,
        "train_ids": train_ids,
        "test_series": test_series,
        "test_pids": test_pids,
        "test_ids": test_ids,
        "keypoint_cols": keypoint_cols,
    }


def compute_metrics(oof_preds, train_targets, test_preds):
    """Compute CV score and angle_std."""
    ranges = get_target_ranges()

    total_mse = 0
    target_scores = {}
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        target_scores[target] = scaled_mse
        total_mse += scaled_mse

    cv_score = total_mse / 3

    scaled_angle = TARGET_SCALERS["angle"].transform(
        test_preds[:, 0].reshape(-1, 1)
    ).flatten()
    angle_std = np.std(scaled_angle)

    scaled_depth = TARGET_SCALERS["depth"].transform(
        test_preds[:, 1].reshape(-1, 1)
    ).flatten()
    depth_mean = np.mean(scaled_depth)

    return {
        "cv": cv_score,
        "angle_std": angle_std,
        "depth_mean": depth_mean,
        "target_scores": target_scores,
    }


# ============================================================
# APPROACH 1: Simple Ridge Ensemble (like Sub 25 but simpler)
# ============================================================
def approach_simple_ensemble(data, frames_config, alpha_config, n_pca_config):
    """
    Simple ensemble of Ridge models with different configurations.
    Mimics the diversity of Sub 9/10 blend but simpler.
    """
    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    unique_pids = sorted(np.unique(train_pids))

    # Multiple configurations for diversity
    configs = []
    for frames in frames_config:
        for alpha in alpha_config:
            for n_pca in n_pca_config:
                configs.append((frames, alpha, n_pca))

    n_configs = len(configs)
    all_oof = np.zeros((len(train_series), 3, n_configs))
    all_test = np.zeros((len(test_series), 3, n_configs))

    for cfg_idx, (frames, alpha, n_pca) in enumerate(configs):
        def flatten(series):
            flat = series[:, frames, :].reshape(len(series), -1)
            return np.nan_to_num(flat, nan=0.0)

        train_flat = flatten(train_series)
        test_flat = flatten(test_series)

        oof_preds = np.zeros_like(train_targets)
        predictions = np.zeros((len(test_series), 3))

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_train = train_flat[train_mask]
            y_train = train_targets[train_mask]
            X_test = test_flat[test_mask]

            player_indices = np.where(train_mask)[0]

            n_comp = min(n_pca, len(X_train) - 1)
            pca = PCA(n_components=n_comp)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_pca)
            X_test_scaled = scaler.transform(X_test_pca)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for t_idx in range(3):
                fold_test_preds = []

                for train_idx, val_idx in kf.split(X_train_scaled):
                    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                    y_tr = y_train[train_idx, t_idx]

                    model = Ridge(alpha=alpha)
                    model.fit(X_tr, y_tr)

                    oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val)
                    fold_test_preds.append(model.predict(X_test_scaled))

                predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

        all_oof[:, :, cfg_idx] = oof_preds
        all_test[:, :, cfg_idx] = predictions

    # Simple average blend
    oof_blend = np.mean(all_oof, axis=2)
    test_blend = np.mean(all_test, axis=2)

    return oof_blend, test_blend


# ============================================================
# APPROACH 2: LightGBM + Ridge Ensemble
# ============================================================
def approach_lgb_ridge_ensemble(data):
    """
    Blend LightGBM with Ridge for model diversity.
    """
    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    frames = list(range(140, 170))

    def flatten(series):
        flat = series[:, frames, :].reshape(len(series), -1)
        return np.nan_to_num(flat, nan=0.0)

    train_flat = flatten(train_series)
    test_flat = flatten(test_series)

    unique_pids = sorted(np.unique(train_pids))

    # Ridge predictions
    oof_ridge = np.zeros_like(train_targets)
    test_ridge = np.zeros((len(test_series), 3))

    # LightGBM predictions
    oof_lgb = np.zeros_like(train_targets)
    test_lgb = np.zeros((len(test_series), 3))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        player_indices = np.where(train_mask)[0]

        # PCA for Ridge
        n_comp = min(15, len(X_train) - 1)
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for t_idx in range(3):
            fold_test_ridge = []
            fold_test_lgb = []

            for train_idx, val_idx in kf.split(X_train):
                # Ridge
                X_tr_r, X_val_r = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr = y_train[train_idx, t_idx]

                ridge = Ridge(alpha=100)
                ridge.fit(X_tr_r, y_tr)
                oof_ridge[player_indices[val_idx], t_idx] = ridge.predict(X_val_r)
                fold_test_ridge.append(ridge.predict(X_test_scaled))

                # LightGBM on raw features
                X_tr_l, X_val_l = X_train[train_idx], X_train[val_idx]

                lgb_model = lgb.LGBMRegressor(
                    n_estimators=100,
                    num_leaves=10,
                    learning_rate=0.05,
                    reg_alpha=1.0,
                    reg_lambda=1.0,
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1,
                )
                lgb_model.fit(X_tr_l, y_tr)
                oof_lgb[player_indices[val_idx], t_idx] = lgb_model.predict(X_val_l)
                fold_test_lgb.append(lgb_model.predict(X_test))

            test_ridge[test_mask, t_idx] = np.mean(fold_test_ridge, axis=0)
            test_lgb[test_mask, t_idx] = np.mean(fold_test_lgb, axis=0)

    # Blend 50-50 like Sub 25
    oof_blend = 0.5 * oof_ridge + 0.5 * oof_lgb
    test_blend = 0.5 * test_ridge + 0.5 * test_lgb

    return oof_blend, test_blend, oof_ridge, test_ridge, oof_lgb, test_lgb


# ============================================================
# APPROACH 3: LightGBM + CatBoost + Ridge (Like Sub 9/10)
# ============================================================
def approach_full_ensemble(data):
    """
    Full ensemble like Sub 9/10 but with simpler features.
    """
    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    frames = list(range(140, 170))

    def flatten(series):
        flat = series[:, frames, :].reshape(len(series), -1)
        return np.nan_to_num(flat, nan=0.0)

    train_flat = flatten(train_series)
    test_flat = flatten(test_series)

    unique_pids = sorted(np.unique(train_pids))

    # Storage for each model type
    oof_ridge = np.zeros_like(train_targets)
    test_ridge = np.zeros((len(test_series), 3))
    oof_lgb = np.zeros_like(train_targets)
    test_lgb = np.zeros((len(test_series), 3))
    oof_cat = np.zeros_like(train_targets)
    test_cat = np.zeros((len(test_series), 3))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        player_indices = np.where(train_mask)[0]

        # PCA for Ridge
        n_comp = min(15, len(X_train) - 1)
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for t_idx in range(3):
            fold_test_ridge = []
            fold_test_lgb = []
            fold_test_cat = []

            for train_idx, val_idx in kf.split(X_train):
                y_tr = y_train[train_idx, t_idx]

                # Ridge
                X_tr_r, X_val_r = X_train_scaled[train_idx], X_train_scaled[val_idx]
                ridge = Ridge(alpha=100)
                ridge.fit(X_tr_r, y_tr)
                oof_ridge[player_indices[val_idx], t_idx] = ridge.predict(X_val_r)
                fold_test_ridge.append(ridge.predict(X_test_scaled))

                # LightGBM
                X_tr_l, X_val_l = X_train[train_idx], X_train[val_idx]
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=100, num_leaves=10, learning_rate=0.05,
                    reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1
                )
                lgb_model.fit(X_tr_l, y_tr)
                oof_lgb[player_indices[val_idx], t_idx] = lgb_model.predict(X_val_l)
                fold_test_lgb.append(lgb_model.predict(X_test))

                # CatBoost
                cat_model = CatBoostRegressor(
                    iterations=100, depth=4, learning_rate=0.05,
                    l2_leaf_reg=3.0, random_state=42, verbose=False
                )
                cat_model.fit(X_tr_l, y_tr)
                oof_cat[player_indices[val_idx], t_idx] = cat_model.predict(X_val_l)
                fold_test_cat.append(cat_model.predict(X_test))

            test_ridge[test_mask, t_idx] = np.mean(fold_test_ridge, axis=0)
            test_lgb[test_mask, t_idx] = np.mean(fold_test_lgb, axis=0)
            test_cat[test_mask, t_idx] = np.mean(fold_test_cat, axis=0)

    return {
        "ridge": (oof_ridge, test_ridge),
        "lgb": (oof_lgb, test_lgb),
        "cat": (oof_cat, test_cat),
    }


def find_optimal_blend(models_dict, train_targets):
    """Find optimal blend weights using CV."""
    ranges = get_target_ranges()

    best_cv = float('inf')
    best_weights = None
    best_blend = None

    # Try different weight combinations
    weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    model_names = list(models_dict.keys())
    n_models = len(model_names)

    if n_models == 2:
        for w1 in weight_options:
            w2 = 1.0 - w1
            weights = {model_names[0]: w1, model_names[1]: w2}

            oof_blend = (w1 * models_dict[model_names[0]][0] +
                        w2 * models_dict[model_names[1]][0])
            test_blend = (w1 * models_dict[model_names[0]][1] +
                         w2 * models_dict[model_names[1]][1])

            total_mse = 0
            for t_idx, target in enumerate(TARGETS):
                mse = np.mean((oof_blend[:, t_idx] - train_targets[:, t_idx])**2)
                scaled_mse = mse / (ranges[target]**2)
                total_mse += scaled_mse
            cv = total_mse / 3

            if cv < best_cv:
                best_cv = cv
                best_weights = weights
                best_blend = (oof_blend, test_blend)

    elif n_models == 3:
        for w1 in [0.0, 0.2, 0.33, 0.4, 0.5]:
            for w2 in [0.0, 0.2, 0.33, 0.4, 0.5]:
                w3 = 1.0 - w1 - w2
                if w3 < 0 or w3 > 1:
                    continue

                weights = {model_names[0]: w1, model_names[1]: w2, model_names[2]: w3}

                oof_blend = (w1 * models_dict[model_names[0]][0] +
                            w2 * models_dict[model_names[1]][0] +
                            w3 * models_dict[model_names[2]][0])
                test_blend = (w1 * models_dict[model_names[0]][1] +
                             w2 * models_dict[model_names[1]][1] +
                             w3 * models_dict[model_names[2]][1])

                total_mse = 0
                for t_idx, target in enumerate(TARGETS):
                    mse = np.mean((oof_blend[:, t_idx] - train_targets[:, t_idx])**2)
                    scaled_mse = mse / (ranges[target]**2)
                    total_mse += scaled_mse
                cv = total_mse / 3

                if cv < best_cv:
                    best_cv = cv
                    best_weights = weights
                    best_blend = (oof_blend, test_blend)

    return best_weights, best_cv, best_blend


def main():
    print("=" * 80)
    print("SYSTEMATIC TESTING TO BEAT SUB 25 (LB 0.008305)")
    print("=" * 80)
    print("\nSub 25 profile: angle_std=0.1374, depth_mean=0.5055")
    print("Target: LB < 0.008305")

    data = load_data()
    train_targets = data["train_targets"]

    results = []

    # ============================================================
    # TEST 1: Simple Ridge Ensemble
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: Simple Ridge Ensemble (varying configs)")
    print("=" * 60)

    frames_configs = [
        list(range(140, 170)),
        list(range(145, 165)),
        list(range(135, 175)),
    ]
    alpha_configs = [50, 100, 200]
    n_pca_configs = [10, 15, 20]

    oof, test = approach_simple_ensemble(
        data, frames_configs, alpha_configs, n_pca_configs
    )
    metrics = compute_metrics(oof, train_targets, test)

    print(f"  CV: {metrics['cv']:.6f}")
    print(f"  angle_std: {metrics['angle_std']:.4f} (Sub25: 0.1374)")
    print(f"  depth_mean: {metrics['depth_mean']:.4f} (Sub25: 0.5055)")

    results.append({
        "name": "Simple Ridge Ensemble",
        "cv": metrics["cv"],
        "angle_std": metrics["angle_std"],
        "oof": oof,
        "test": test,
    })

    # ============================================================
    # TEST 2: LightGBM + Ridge Blend
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: LightGBM + Ridge Blend (50-50)")
    print("=" * 60)

    oof_blend, test_blend, oof_ridge, test_ridge, oof_lgb, test_lgb = \
        approach_lgb_ridge_ensemble(data)

    metrics = compute_metrics(oof_blend, train_targets, test_blend)
    print(f"  Blend 50-50:")
    print(f"    CV: {metrics['cv']:.6f}")
    print(f"    angle_std: {metrics['angle_std']:.4f}")

    results.append({
        "name": "LGB+Ridge 50-50",
        "cv": metrics["cv"],
        "angle_std": metrics["angle_std"],
        "oof": oof_blend,
        "test": test_blend,
    })

    # Try optimal blend
    models_2 = {"ridge": (oof_ridge, test_ridge), "lgb": (oof_lgb, test_lgb)}
    best_weights, best_cv, best_blend = find_optimal_blend(models_2, train_targets)

    print(f"\n  Optimal blend weights: {best_weights}")
    metrics_opt = compute_metrics(best_blend[0], train_targets, best_blend[1])
    print(f"    CV: {metrics_opt['cv']:.6f}")
    print(f"    angle_std: {metrics_opt['angle_std']:.4f}")

    results.append({
        "name": f"LGB+Ridge optimal",
        "cv": metrics_opt["cv"],
        "angle_std": metrics_opt["angle_std"],
        "oof": best_blend[0],
        "test": best_blend[1],
    })

    # ============================================================
    # TEST 3: Full Ensemble (LGB + Cat + Ridge)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Full Ensemble (LGB + CatBoost + Ridge)")
    print("=" * 60)

    models_3 = approach_full_ensemble(data)

    # Find optimal blend
    best_weights, best_cv, best_blend = find_optimal_blend(models_3, train_targets)

    print(f"  Optimal blend weights: {best_weights}")
    metrics_full = compute_metrics(best_blend[0], train_targets, best_blend[1])
    print(f"    CV: {metrics_full['cv']:.6f}")
    print(f"    angle_std: {metrics_full['angle_std']:.4f}")

    results.append({
        "name": "Full Ensemble optimal",
        "cv": metrics_full["cv"],
        "angle_std": metrics_full["angle_std"],
        "oof": best_blend[0],
        "test": best_blend[1],
    })

    # Also try 33-33-33 blend
    oof_333 = (models_3["ridge"][0] + models_3["lgb"][0] + models_3["cat"][0]) / 3
    test_333 = (models_3["ridge"][1] + models_3["lgb"][1] + models_3["cat"][1]) / 3
    metrics_333 = compute_metrics(oof_333, train_targets, test_333)

    print(f"\n  Equal blend (33-33-33):")
    print(f"    CV: {metrics_333['cv']:.6f}")
    print(f"    angle_std: {metrics_333['angle_std']:.4f}")

    results.append({
        "name": "Full Ensemble 33-33-33",
        "cv": metrics_333["cv"],
        "angle_std": metrics_333["angle_std"],
        "oof": oof_333,
        "test": test_333,
    })

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY - SORTED BY CV")
    print("=" * 80)

    print(f"\n{'Approach':<30} {'CV':>10} {'angle_std':>12} {'vs Sub25 std':>12}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: x["cv"]):
        diff = r["angle_std"] - 0.1374
        sign = "+" if diff > 0 else ""
        print(f"{r['name']:<30} {r['cv']:>10.6f} {r['angle_std']:>12.4f} {sign}{diff:>11.4f}")

    # ============================================================
    # BEST CANDIDATE
    # ============================================================
    print("\n" + "=" * 80)
    print("BEST CANDIDATE FOR SUBMISSION")
    print("=" * 80)

    # Sort by CV
    best = min(results, key=lambda x: x["cv"])

    print(f"\nBest by CV: {best['name']}")
    print(f"  CV: {best['cv']:.6f}")
    print(f"  angle_std: {best['angle_std']:.4f}")

    # Compare to Sub 25
    print(f"\nComparison to Sub 25:")
    print(f"  Sub 25 angle_std: 0.1374")
    print(f"  This angle_std:   {best['angle_std']:.4f}")
    print(f"  Difference:       {best['angle_std'] - 0.1374:+.4f}")

    # Check if angle_std is close enough to Sub 25
    if abs(best['angle_std'] - 0.1374) < 0.01:
        print(f"\n  GOOD: angle_std is close to Sub 25!")
    else:
        print(f"\n  WARNING: angle_std differs significantly from Sub 25")

    return results, best


if __name__ == "__main__":
    results, best = main()

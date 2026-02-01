"""
Match Sub 25's Profile More Closely

Sub 25 profile:
- angle_std: 0.1374
- depth_mean: 0.5055
- LB: 0.008305

Our best so far has angle_std=0.1437 (too high).
Need to reduce angle variability while maintaining CV.

Strategies:
1. Try different blend weights to lower angle_std
2. Add regularization specifically for angle
3. Use conservative predictions for angle
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
from sklearn.linear_model import Ridge
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


def train_per_target_ensemble(data, target_configs):
    """
    Train ensemble with target-specific configurations.

    target_configs: dict mapping target name to config dict with:
        - models: list of model types to use
        - weights: blend weights
        - alpha: Ridge alpha (for angle, use higher for more conservative)
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

    oof_preds = np.zeros_like(train_targets)
    predictions = np.zeros((len(test_series), 3))

    for t_idx, target in enumerate(TARGETS):
        config = target_configs[target]

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_train = train_flat[train_mask]
            y_train = train_targets[train_mask, t_idx]
            X_test = test_flat[test_mask]

            player_indices = np.where(train_mask)[0]

            # PCA
            n_comp = min(15, len(X_train) - 1)
            pca = PCA(n_components=n_comp)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_pca)
            X_test_scaled = scaler.transform(X_test_pca)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # Collect predictions from different models
            model_oof = {m: np.zeros(len(X_train)) for m in config["models"]}
            model_test = {m: [] for m in config["models"]}

            for train_idx, val_idx in kf.split(X_train):
                X_tr_pca, X_val_pca = X_train_scaled[train_idx], X_train_scaled[val_idx]
                X_tr_raw, X_val_raw = X_train[train_idx], X_train[val_idx]
                y_tr = y_train[train_idx]

                for model_name in config["models"]:
                    if model_name == "ridge":
                        model = Ridge(alpha=config.get("ridge_alpha", 100))
                        model.fit(X_tr_pca, y_tr)
                        model_oof["ridge"][val_idx] = model.predict(X_val_pca)
                        model_test["ridge"].append(model.predict(X_test_scaled))

                    elif model_name == "lgb":
                        model = lgb.LGBMRegressor(
                            n_estimators=100, num_leaves=10, learning_rate=0.05,
                            reg_alpha=config.get("lgb_alpha", 1.0),
                            reg_lambda=config.get("lgb_lambda", 1.0),
                            random_state=42, verbose=-1
                        )
                        model.fit(X_tr_raw, y_tr)
                        model_oof["lgb"][val_idx] = model.predict(X_val_raw)
                        model_test["lgb"].append(model.predict(X_test))

                    elif model_name == "cat":
                        model = CatBoostRegressor(
                            iterations=100, depth=4, learning_rate=0.05,
                            l2_leaf_reg=config.get("cat_l2", 3.0),
                            random_state=42, verbose=False
                        )
                        model.fit(X_tr_raw, y_tr)
                        model_oof["cat"][val_idx] = model.predict(X_val_raw)
                        model_test["cat"].append(model.predict(X_test))

            # Blend models
            weights = config["weights"]
            oof_blend = sum(w * model_oof[m] for m, w in weights.items())
            test_blend = sum(
                w * np.mean(model_test[m], axis=0) for m, w in weights.items()
            )

            oof_preds[player_indices, t_idx] = oof_blend
            predictions[test_mask, t_idx] = test_blend

    return oof_preds, predictions


def main():
    print("=" * 80)
    print("MATCHING SUB 25 PROFILE")
    print("=" * 80)
    print("\nTarget: angle_std=0.1374, depth_mean=0.5055")

    data = load_data()
    train_targets = data["train_targets"]

    results = []

    # ============================================================
    # Config 1: More conservative angle (higher Ridge weight)
    # ============================================================
    print("\n" + "=" * 60)
    print("Config 1: Conservative angle (60% Ridge for angle)")
    print("=" * 60)

    config1 = {
        "angle": {
            "models": ["ridge", "lgb", "cat"],
            "weights": {"ridge": 0.6, "lgb": 0.2, "cat": 0.2},
            "ridge_alpha": 200,  # Higher = more conservative
        },
        "depth": {
            "models": ["ridge", "lgb", "cat"],
            "weights": {"ridge": 0.2, "lgb": 0.4, "cat": 0.4},
            "ridge_alpha": 100,
        },
        "left_right": {
            "models": ["ridge", "lgb", "cat"],
            "weights": {"ridge": 0.2, "lgb": 0.4, "cat": 0.4},
            "ridge_alpha": 100,
        },
    }

    oof1, test1 = train_per_target_ensemble(data, config1)
    m1 = compute_metrics(oof1, train_targets, test1)
    print(f"  CV: {m1['cv']:.6f}, angle_std: {m1['angle_std']:.4f}")
    results.append(("Config1_conservative_angle", m1, oof1, test1))

    # ============================================================
    # Config 2: Even more conservative angle (80% Ridge)
    # ============================================================
    print("\n" + "=" * 60)
    print("Config 2: Very conservative angle (80% Ridge for angle)")
    print("=" * 60)

    config2 = {
        "angle": {
            "models": ["ridge", "lgb"],
            "weights": {"ridge": 0.8, "lgb": 0.2},
            "ridge_alpha": 300,
        },
        "depth": {
            "models": ["ridge", "lgb", "cat"],
            "weights": {"ridge": 0.2, "lgb": 0.4, "cat": 0.4},
            "ridge_alpha": 100,
        },
        "left_right": {
            "models": ["ridge", "lgb", "cat"],
            "weights": {"ridge": 0.2, "lgb": 0.4, "cat": 0.4},
            "ridge_alpha": 100,
        },
    }

    oof2, test2 = train_per_target_ensemble(data, config2)
    m2 = compute_metrics(oof2, train_targets, test2)
    print(f"  CV: {m2['cv']:.6f}, angle_std: {m2['angle_std']:.4f}")
    results.append(("Config2_very_conservative", m2, oof2, test2))

    # ============================================================
    # Config 3: Balanced with higher regularization everywhere
    # ============================================================
    print("\n" + "=" * 60)
    print("Config 3: High regularization everywhere")
    print("=" * 60)

    config3 = {
        "angle": {
            "models": ["ridge", "lgb", "cat"],
            "weights": {"ridge": 0.4, "lgb": 0.3, "cat": 0.3},
            "ridge_alpha": 200,
            "lgb_alpha": 2.0,
            "lgb_lambda": 2.0,
            "cat_l2": 5.0,
        },
        "depth": {
            "models": ["ridge", "lgb", "cat"],
            "weights": {"ridge": 0.4, "lgb": 0.3, "cat": 0.3},
            "ridge_alpha": 200,
            "lgb_alpha": 2.0,
            "lgb_lambda": 2.0,
            "cat_l2": 5.0,
        },
        "left_right": {
            "models": ["ridge", "lgb", "cat"],
            "weights": {"ridge": 0.4, "lgb": 0.3, "cat": 0.3},
            "ridge_alpha": 200,
            "lgb_alpha": 2.0,
            "lgb_lambda": 2.0,
            "cat_l2": 5.0,
        },
    }

    oof3, test3 = train_per_target_ensemble(data, config3)
    m3 = compute_metrics(oof3, train_targets, test3)
    print(f"  CV: {m3['cv']:.6f}, angle_std: {m3['angle_std']:.4f}")
    results.append(("Config3_high_reg", m3, oof3, test3))

    # ============================================================
    # Config 4: Replicate Sub 25 approach (LGB+Cat heavy blend)
    # ============================================================
    print("\n" + "=" * 60)
    print("Config 4: Sub9/10 style (LGB + Cat heavy)")
    print("=" * 60)

    config4 = {
        "angle": {
            "models": ["ridge", "lgb", "cat"],
            "weights": {"ridge": 0.1, "lgb": 0.45, "cat": 0.45},
            "ridge_alpha": 100,
        },
        "depth": {
            "models": ["ridge", "lgb", "cat"],
            "weights": {"ridge": 0.1, "lgb": 0.45, "cat": 0.45},
            "ridge_alpha": 100,
        },
        "left_right": {
            "models": ["ridge", "lgb", "cat"],
            "weights": {"ridge": 0.1, "lgb": 0.45, "cat": 0.45},
            "ridge_alpha": 100,
        },
    }

    oof4, test4 = train_per_target_ensemble(data, config4)
    m4 = compute_metrics(oof4, train_targets, test4)
    print(f"  CV: {m4['cv']:.6f}, angle_std: {m4['angle_std']:.4f}")
    results.append(("Config4_sub25_style", m4, oof4, test4))

    # ============================================================
    # Config 5: Blend of configs to target exact angle_std
    # ============================================================
    print("\n" + "=" * 60)
    print("Config 5: Blend configs to target angle_std=0.1374")
    print("=" * 60)

    # Blend config2 (conservative) with config4 (aggressive)
    for blend_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        oof_blend = blend_w * oof2 + (1 - blend_w) * oof4
        test_blend = blend_w * test2 + (1 - blend_w) * test4
        m_blend = compute_metrics(oof_blend, train_targets, test_blend)

        diff = abs(m_blend['angle_std'] - 0.1374)
        print(f"  w={blend_w:.1f}: CV={m_blend['cv']:.6f}, "
              f"angle_std={m_blend['angle_std']:.4f} (diff={diff:.4f})")

        if diff < 0.005:  # Close enough
            results.append((f"Config5_blend_{blend_w}", m_blend, oof_blend, test_blend))

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Config':<30} {'CV':>10} {'angle_std':>12} {'diff':>10}")
    print("-" * 65)

    for name, metrics, _, _ in sorted(results, key=lambda x: abs(x[1]['angle_std'] - 0.1374)):
        diff = metrics['angle_std'] - 0.1374
        print(f"{name:<30} {metrics['cv']:>10.6f} {metrics['angle_std']:>12.4f} {diff:>+10.4f}")

    # Find best match to Sub 25 profile
    best_match = min(results, key=lambda x: abs(x[1]['angle_std'] - 0.1374))
    print(f"\nBest match to Sub 25 angle_std: {best_match[0]}")
    print(f"  angle_std: {best_match[1]['angle_std']:.4f} (target: 0.1374)")
    print(f"  CV: {best_match[1]['cv']:.6f}")

    # Find best CV
    best_cv = min(results, key=lambda x: x[1]['cv'])
    print(f"\nBest CV: {best_cv[0]}")
    print(f"  CV: {best_cv[1]['cv']:.6f}")
    print(f"  angle_std: {best_cv[1]['angle_std']:.4f}")

    return results


if __name__ == "__main__":
    results = main()

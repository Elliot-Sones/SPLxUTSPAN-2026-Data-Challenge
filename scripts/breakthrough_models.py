"""
Breakthrough Models - Creative approaches for step-change improvement.

Three strategies:
1. Feature-selected tree model (aim LOO < 0.008)
2. k-NN regression (completely different model family)
3. Stacked residual correction (predict what Sub 2716 gets wrong)

Goal: produce models with correlation < 0.70 vs Sub 2716 AND LOO < 0.008.
"""

import json
import time
import fcntl
import argparse
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0

KEY_JOINTS = [
    "right_wrist", "right_elbow", "right_shoulder",
    "left_wrist", "left_elbow", "left_shoulder",
    "right_hip", "left_hip", "mid_hip",
    "right_knee", "left_knee",
    "neck", "nose",
    "right_ankle", "left_ankle",
]

WINDOWS = {
    "early": (60, 120),
    "pre_release": (120, 150),
    "release": (145, 170),
    "follow": (170, 210),
    "full": (80, 200),
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


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_data(csv_path):
    """Load and parse keypoint data."""
    df = pd.read_csv(csv_path)
    n = len(df)
    kp_cols = [c for c in df.columns if c.startswith(("right_", "left_", "mid_", "neck", "nose"))
               and c not in ("left_right",)]
    n_kp = len(kp_cols)

    raw = np.zeros((n, n_kp, 240), dtype=np.float32)
    for j, col in enumerate(kp_cols):
        for i in range(n):
            raw[i, j, :] = parse_array_string(df[col].iloc[i])

    n_joints = n_kp // 3
    kp3d = np.zeros((n, 240, n_joints, 3), dtype=np.float32)
    joint_names = []
    for j in range(n_joints):
        base = kp_cols[j * 3].rsplit("_", 1)[0]
        joint_names.append(base)
        for c in range(3):
            kp3d[:, :, j, c] = raw[:, j * 3 + c, :]

    nan_mask = np.isnan(kp3d) | np.isinf(kp3d)
    kp3d[nan_mask] = 0.0

    targets = {}
    target_scalers = {}
    for t in TARGETS:
        for col in [f"scaled_{t}", t]:
            if col in df.columns:
                raw_vals = df[col].values.astype(np.float32)
                if col.startswith("scaled_"):
                    targets[t] = raw_vals
                else:
                    scaler = MinMaxScaler()
                    targets[t] = scaler.fit_transform(raw_vals.reshape(-1, 1)).ravel().astype(np.float32)
                    target_scalers[t] = scaler
                break

    players = df["participant_id"].values if "participant_id" in df.columns else np.ones(n, dtype=int)
    ids = df["id"].values if "id" in df.columns else np.arange(n)

    return kp3d, joint_names, targets, players, ids, target_scalers


def compute_hoop_relative(kp3d, joint_names):
    n, T, n_joints, _ = kp3d.shape
    hoop = HOOP_POS.astype(np.float32)
    joint_idx = {name: i for i, name in enumerate(joint_names)}
    kp_hoop = kp3d.copy()
    for i in range(n):
        for t in range(T):
            kp_hoop[i, t, :, :] -= hoop
    return kp_hoop, joint_idx


def extract_features_full(kp3d, kp_hoop, joint_names, joint_idx):
    """Extract comprehensive features for tree models."""
    n = kp3d.shape[0]
    features = []
    feature_names = []

    key_joint_indices = []
    for jname in KEY_JOINTS:
        if jname in joint_idx:
            key_joint_indices.append(joint_idx[jname])

    for ji, jname in zip(key_joint_indices, KEY_JOINTS):
        for wname, (t_start, t_end) in WINDOWS.items():
            for coord in range(3):
                coord_name = ["x", "y", "z"][coord]
                prefix = f"{jname}_{coord_name}_{wname}"

                vals = kp_hoop[:, t_start:t_end, ji, coord]
                features.append(np.nanmean(vals, axis=1))
                feature_names.append(f"{prefix}_mean")
                features.append(np.nanstd(vals, axis=1))
                feature_names.append(f"{prefix}_std")
                features.append(np.nanmin(vals, axis=1))
                feature_names.append(f"{prefix}_min")
                features.append(np.nanmax(vals, axis=1))
                feature_names.append(f"{prefix}_max")
                features.append(np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1))
                feature_names.append(f"{prefix}_range")

                vel = np.diff(vals, axis=1) / DT
                if vel.shape[1] > 0:
                    features.append(np.nanmean(vel, axis=1))
                    feature_names.append(f"{prefix}_vel_mean")
                    features.append(np.nanstd(vel, axis=1))
                    feature_names.append(f"{prefix}_vel_std")
                    features.append(np.nanmax(np.abs(vel), axis=1))
                    feature_names.append(f"{prefix}_vel_peak")

    # Cross-joint distances at key frames
    for frame_offset, frame_name in [(150, "release"), (170, "post")]:
        for j1_name, j2_name in [
            ("right_wrist", "right_elbow"),
            ("right_elbow", "right_shoulder"),
            ("right_shoulder", "right_hip"),
            ("left_shoulder", "right_shoulder"),
            ("mid_hip", "neck"),
        ]:
            if j1_name in joint_idx and j2_name in joint_idx:
                j1 = joint_idx[j1_name]
                j2 = joint_idx[j2_name]
                dist = np.linalg.norm(
                    kp_hoop[:, frame_offset, j1, :] - kp_hoop[:, frame_offset, j2, :],
                    axis=1
                )
                features.append(dist)
                feature_names.append(f"dist_{j1_name}_{j2_name}_{frame_name}")

    for frame_offset, frame_name in [(140, "pre"), (150, "mid"), (160, "post"), (170, "late")]:
        if "right_wrist" in joint_idx:
            ji = joint_idx["right_wrist"]
            dist = np.linalg.norm(kp_hoop[:, frame_offset, ji, :], axis=1)
            features.append(dist)
            feature_names.append(f"wrist_hoop_dist_{frame_name}")

    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, feature_names


def extract_compact_features(kp_hoop, joint_idx, frames=[140, 150, 155, 160, 170]):
    """Extract compact hoop-relative features at specific frames for k-NN."""
    n = kp_hoop.shape[0]
    features = []

    key_joints = ["right_wrist", "right_elbow", "right_shoulder",
                  "left_wrist", "left_shoulder", "mid_hip", "neck"]

    for frame in frames:
        for jname in key_joints:
            if jname in joint_idx:
                ji = joint_idx[jname]
                for c in range(3):
                    features.append(kp_hoop[:, frame, ji, c])

    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ============================================================
# MODEL 1: Feature-Selected XGBoost
# ============================================================

def feature_selected_xgb(X_train, y_dict, X_test, players_train, players_test,
                         target, n_top=80, n_seeds=5):
    """XGBoost with feature selection via mutual information."""
    from xgboost import XGBRegressor

    y = y_dict[target]
    n = len(y)

    # Step 1: Feature selection via mutual information (on all data)
    mi_scores = mutual_info_regression(X_train, y, random_state=42)
    top_idx = np.argsort(mi_scores)[-n_top:]

    X_sel_train = X_train[:, top_idx]
    X_sel_test = X_test[:, top_idx]

    # Step 2: LOO with selected features
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        player_i = players_train[i]
        player_mask = players_train[mask] == player_i

        if np.sum(player_mask) < 8:
            X_tr = X_sel_train[mask]
            y_tr = y[mask]
        else:
            full_idx = np.where(mask)[0]
            X_tr = X_sel_train[full_idx[player_mask]]
            y_tr = y[full_idx[player_mask]]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_sel_train[i:i+1])

        model = XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=2,
            reg_alpha=0.5,
            reg_lambda=2.0,
            verbosity=0,
            random_state=42,
        )
        model.fit(X_tr_s, y_tr)
        loo_preds[i] = model.predict(X_te_s)[0]

    loo_mse = np.mean((loo_preds - y) ** 2)

    # Step 3: Multi-seed test predictions
    test_preds_all = []
    unique_players = np.unique(players_test)

    for seed in [42, 7, 99, 133, 2026][:n_seeds]:
        test_preds = np.zeros(len(X_sel_test))

        for p in unique_players:
            train_mask = players_train == p
            test_mask = players_test == p

            if np.sum(train_mask) < 8:
                X_tr = X_sel_train
                y_tr = y
            else:
                X_tr = X_sel_train[train_mask]
                y_tr = y[train_mask]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_sel_test[test_mask])

            model = XGBRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=2,
                reg_alpha=0.5,
                reg_lambda=2.0,
                verbosity=0,
                random_state=seed,
            )
            model.fit(X_tr_s, y_tr)
            test_preds[test_mask] = model.predict(X_te_s)

        test_preds_all.append(test_preds)

    return np.clip(np.mean(test_preds_all, axis=0), 0, 1), loo_preds, loo_mse, top_idx


# ============================================================
# MODEL 2: k-NN Regression
# ============================================================

def knn_model(X_train, y_dict, X_test, players_train, players_test, target,
              k_values=[5, 7, 10, 15, 20]):
    """k-NN regression with optimal k selection via LOO."""
    y = y_dict[target]
    n = len(y)

    # Test different k values via LOO
    best_k = None
    best_mse = float('inf')

    for k in k_values:
        loo_preds = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            player_i = players_train[i]
            player_mask = players_train[mask] == player_i

            if np.sum(player_mask) < k + 2:
                X_tr = X_train[mask]
                y_tr = y[mask]
                k_use = min(k, len(y_tr) - 1)
            else:
                full_idx = np.where(mask)[0]
                X_tr = X_train[full_idx[player_mask]]
                y_tr = y[full_idx[player_mask]]
                k_use = min(k, len(y_tr) - 1)

            if k_use < 1:
                loo_preds[i] = np.mean(y_tr)
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_train[i:i+1])

            knn = KNeighborsRegressor(n_neighbors=k_use, weights='distance')
            knn.fit(X_tr_s, y_tr)
            loo_preds[i] = knn.predict(X_te_s)[0]

        mse = np.mean((loo_preds - y) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_k = k
            best_loo = loo_preds.copy()

    # Test predictions with best k
    test_preds = np.zeros(X_test.shape[0])
    unique_players = np.unique(players_test)

    for p in unique_players:
        train_mask = players_train == p
        test_mask = players_test == p

        if np.sum(train_mask) < best_k + 2:
            X_tr, y_tr = X_train, y
        else:
            X_tr = X_train[train_mask]
            y_tr = y[train_mask]

        k_use = min(best_k, len(y_tr) - 1)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_test[test_mask])

        knn = KNeighborsRegressor(n_neighbors=k_use, weights='distance')
        knn.fit(X_tr_s, y_tr)
        test_preds[test_mask] = knn.predict(X_te_s)

    return np.clip(test_preds, 0, 1), best_loo, best_mse, best_k


# ============================================================
# MODEL 3: Stacked Residual Correction
# ============================================================

def residual_correction(X_train, y_dict, X_test, players_train, players_test,
                        target, base_loo_preds):
    """Train a model to predict residuals of the base model."""
    from xgboost import XGBRegressor

    y = y_dict[target]
    residuals = y - base_loo_preds  # what the base model gets wrong
    n = len(y)

    # Use multiple approaches to predict residuals
    loo_corrections = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        player_i = players_train[i]
        player_mask = players_train[mask] == player_i

        if np.sum(player_mask) < 8:
            X_tr = X_train[mask]
            r_tr = residuals[mask]
        else:
            full_idx = np.where(mask)[0]
            X_tr = X_train[full_idx[player_mask]]
            r_tr = residuals[full_idx[player_mask]]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_train[i:i+1])

        # Light model to predict residuals
        model = XGBRegressor(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            min_child_weight=5,
            reg_alpha=1.0,
            reg_lambda=5.0,
            verbosity=0,
            random_state=42,
        )
        model.fit(X_tr_s, r_tr)
        loo_corrections[i] = model.predict(X_te_s)[0]

    # Evaluate: does correction help?
    corrected_preds = base_loo_preds + loo_corrections
    original_mse = np.mean((base_loo_preds - y) ** 2)
    corrected_mse = np.mean((corrected_preds - y) ** 2)

    # Test predictions
    test_corrections = np.zeros(X_test.shape[0])
    unique_players = np.unique(players_test)

    for p in unique_players:
        train_mask = players_train == p
        test_mask = players_test == p

        if np.sum(train_mask) < 8:
            X_tr = X_train
            r_tr = residuals
        else:
            X_tr = X_train[train_mask]
            r_tr = residuals[train_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_test[test_mask])

        model = XGBRegressor(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            min_child_weight=5,
            reg_alpha=1.0,
            reg_lambda=5.0,
            verbosity=0,
            random_state=42,
        )
        model.fit(X_tr_s, r_tr)
        test_corrections[test_mask] = model.predict(X_te_s)

    return test_corrections, loo_corrections, original_mse, corrected_mse


# ============================================================
# MODEL 4: Per-Player Blend Weight Optimization
# ============================================================

def optimize_per_player_weights(anchor_sub, diverse_subs, train_targets, players_train,
                                test_ids, players_test):
    """Find optimal blend weights per player using honest LOO from anchor."""
    # This uses the actual predictions and targets to optimize per-player weights
    # For each player, find the blend that minimizes LOO MSE

    # Load anchor LOO predictions (we need these from the pipeline)
    # For now, use a simulated approach: test weight combinations per player

    anchor_df = pd.read_csv(SUBMISSION_DIR / f"submission_{anchor_sub}.csv")
    diverse_dfs = {s: pd.read_csv(SUBMISSION_DIR / f"submission_{s}.csv") for s in diverse_subs}

    # For each target, try different weights per player
    results = {}
    for target in TARGETS:
        col = f"scaled_{target}"
        results[target] = {}

        for player in np.unique(players_test):
            test_mask = players_test == player
            n_test_player = np.sum(test_mask)

            best_weights = {anchor_sub: 1.0}

            # For each diverse source, try small weights
            for div_sub in diverse_subs:
                for w in [0.01, 0.02, 0.03, 0.05]:
                    # Store config
                    pass

            results[target][player] = best_weights

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=int, default=2716)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-top-features", type=int, default=80)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("BREAKTHROUGH MODELS - Creative Approaches")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    kp3d_train, joint_names, targets_train, players_train, ids_train, scalers = load_data(
        DATA_DIR / "train.csv"
    )
    kp3d_test, _, _, players_test, ids_test, _ = load_data(
        DATA_DIR / "test.csv"
    )
    n_train = kp3d_train.shape[0]
    n_test = kp3d_test.shape[0]
    print(f"  Train: {n_train}, Test: {n_test}")

    # Compute hoop-relative
    kp_hoop_train, joint_idx = compute_hoop_relative(kp3d_train, joint_names)
    kp_hoop_test, _ = compute_hoop_relative(kp3d_test, joint_names)

    # Extract features
    print("\nExtracting full features (for tree models)...")
    X_full_train, feat_names = extract_features_full(kp3d_train, kp_hoop_train, joint_names, joint_idx)
    X_full_test, _ = extract_features_full(kp3d_test, kp_hoop_test, joint_names, joint_idx)
    print(f"  Full features: {X_full_train.shape[1]}")

    print("Extracting compact features (for k-NN)...")
    X_compact_train = extract_compact_features(kp_hoop_train, joint_idx)
    X_compact_test = extract_compact_features(kp_hoop_test, joint_idx)
    print(f"  Compact features: {X_compact_train.shape[1]}")

    # Load anchor for diversity check
    anchor_df = pd.read_csv(SUBMISSION_DIR / f"submission_{args.anchor}.csv")

    # ====================================================================
    # MODEL 1: Feature-Selected XGBoost
    # ====================================================================
    print("\n" + "=" * 50)
    print("MODEL 1: Feature-Selected XGBoost")
    print("=" * 50)

    fsxgb_test = {}
    fsxgb_loo = {}
    fsxgb_mse = {}

    for n_top in [50, 80, 120]:
        print(f"\n  n_top_features = {n_top}")
        test_preds_t = {}
        loo_mse_t = {}

        for target in TARGETS:
            test_p, loo_p, mse, top_idx = feature_selected_xgb(
                X_full_train, targets_train, X_full_test,
                players_train, players_test, target,
                n_top=n_top, n_seeds=args.n_seeds
            )
            test_preds_t[target] = test_p
            loo_mse_t[target] = mse
            print(f"    {target}: LOO={mse:.6f}")

        mean_mse = np.mean(list(loo_mse_t.values()))
        print(f"    Mean LOO: {mean_mse:.6f}")

        # Diversity check
        for target in TARGETS:
            corr = np.corrcoef(test_preds_t[target], anchor_df[f"scaled_{target}"].values)[0, 1]
            print(f"    {target} corr: {corr:.4f}")

        # Keep best
        if not fsxgb_mse or mean_mse < np.mean(list(fsxgb_mse.values())):
            fsxgb_test = test_preds_t
            fsxgb_mse = loo_mse_t
            best_n_top = n_top

    print(f"\n  Best n_top: {best_n_top}")

    # Save feature-selected XGB
    sub_df = pd.DataFrame({
        "id": ids_test,
        "scaled_angle": fsxgb_test["angle"],
        "scaled_depth": fsxgb_test["depth"],
        "scaled_left_right": fsxgb_test["left_right"],
    })
    fsxgb_num = get_next_submission_number()
    sub_df.to_csv(SUBMISSION_DIR / f"submission_{fsxgb_num}.csv", index=False)
    print(f"  Standalone: Sub {fsxgb_num}")

    # ====================================================================
    # MODEL 2: k-NN Regression
    # ====================================================================
    print("\n" + "=" * 50)
    print("MODEL 2: k-NN Regression")
    print("=" * 50)

    knn_test = {}
    knn_mse = {}

    # Try both compact and full features
    for feat_name, X_tr, X_te in [
        ("compact", X_compact_train, X_compact_test),
        ("full_selected", X_full_train, X_full_test),
    ]:
        print(f"\n  Features: {feat_name} ({X_tr.shape[1]} dims)")
        test_t = {}
        mse_t = {}

        for target in TARGETS:
            test_p, loo_p, mse, best_k = knn_model(
                X_tr, targets_train, X_te,
                players_train, players_test, target
            )
            test_t[target] = test_p
            mse_t[target] = mse
            print(f"    {target}: LOO={mse:.6f}, best_k={best_k}")

        mean_mse = np.mean(list(mse_t.values()))
        print(f"    Mean LOO: {mean_mse:.6f}")

        for target in TARGETS:
            corr = np.corrcoef(test_t[target], anchor_df[f"scaled_{target}"].values)[0, 1]
            print(f"    {target} corr: {corr:.4f}")

        # Keep best
        if not knn_mse or mean_mse < np.mean(list(knn_mse.values())):
            knn_test = test_t
            knn_mse = mse_t

    # Save k-NN
    sub_df = pd.DataFrame({
        "id": ids_test,
        "scaled_angle": knn_test["angle"],
        "scaled_depth": knn_test["depth"],
        "scaled_left_right": knn_test["left_right"],
    })
    knn_num = get_next_submission_number()
    sub_df.to_csv(SUBMISSION_DIR / f"submission_{knn_num}.csv", index=False)
    print(f"  Standalone: Sub {knn_num}")

    # ====================================================================
    # MODEL 3: Multi-Model Mega Average
    # ====================================================================
    print("\n" + "=" * 50)
    print("MODEL 3: Multi-Model Mega Average (FS-XGB + k-NN + Original Tree)")
    print("=" * 50)

    # Load original tree models
    xgb_df = pd.read_csv(SUBMISSION_DIR / "submission_2679.csv")
    lgbm_df = pd.read_csv(SUBMISSION_DIR / "submission_2704.csv")

    mega_test = {}
    for target in TARGETS:
        col = f"scaled_{target}"
        # Average 4 diverse tree/knn models
        mega = (fsxgb_test[target] + knn_test[target] +
                xgb_df[col].values + lgbm_df[col].values) / 4.0
        mega_test[target] = np.clip(mega, 0, 1)

        corr = np.corrcoef(mega, anchor_df[col].values)[0, 1]
        print(f"  {target}: corr with Sub {args.anchor} = {corr:.4f}")

    sub_df = pd.DataFrame({
        "id": ids_test,
        "scaled_angle": mega_test["angle"],
        "scaled_depth": mega_test["depth"],
        "scaled_left_right": mega_test["left_right"],
    })
    mega_num = get_next_submission_number()
    sub_df.to_csv(SUBMISSION_DIR / f"submission_{mega_num}.csv", index=False)
    print(f"  Mega Average Standalone: Sub {mega_num}")

    # ====================================================================
    # Generate blend candidates
    # ====================================================================
    print("\n" + "=" * 50)
    print("GENERATING BLEND CANDIDATES")
    print("=" * 50)

    anchor_vals = {t: anchor_df[f"scaled_{t}"].values for t in TARGETS}

    # Blend each new model with anchor
    for model_name, model_test, model_num in [
        ("FS-XGB", fsxgb_test, fsxgb_num),
        ("KNN", knn_test, knn_num),
        ("MegaAvg", mega_test, mega_num),
    ]:
        print(f"\n  {model_name} blends:")
        for w in [0.02, 0.03, 0.05, 0.08]:
            blend_df = pd.DataFrame({"id": ids_test})
            for t in TARGETS:
                blend_df[f"scaled_{t}"] = (1 - w) * anchor_vals[t] + w * model_test[t]
            blend_num = get_next_submission_number()
            blend_df.to_csv(SUBMISSION_DIR / f"submission_{blend_num}.csv", index=False)
            print(f"    {w*100:.0f}%: Sub {blend_num}")

    # Mega blend: anchor + all new models at small weights
    print("\n  Combined mega blends:")
    for fw, kw, mw in [(0.02, 0.01, 0.01), (0.02, 0.02, 0.02), (0.03, 0.02, 0.01)]:
        total = fw + kw + mw
        aw = 1 - total
        blend_df = pd.DataFrame({"id": ids_test})
        for t in TARGETS:
            blend_df[f"scaled_{t}"] = (aw * anchor_vals[t] +
                                       fw * fsxgb_test[t] +
                                       kw * knn_test[t] +
                                       mw * mega_test[t])
        blend_num = get_next_submission_number()
        blend_df.to_csv(SUBMISSION_DIR / f"submission_{blend_num}.csv", index=False)
        print(f"    {fw*100:.0f}%FS + {kw*100:.0f}%KNN + {mw*100:.0f}%Mega: Sub {blend_num}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print(f"{'='*70}")

    # Summary
    print(f"\nFS-XGB (n_top={best_n_top}): LOO = {np.mean(list(fsxgb_mse.values())):.6f}")
    print(f"k-NN: LOO = {np.mean(list(knn_mse.values())):.6f}")
    print(f"\nStandalone subs: FS-XGB={fsxgb_num}, KNN={knn_num}, MegaAvg={mega_num}")


if __name__ == "__main__":
    main()

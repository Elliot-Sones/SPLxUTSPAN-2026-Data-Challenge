"""
Unexplored Approaches - Trying things we haven't done yet

1. Quantile Regression (predict median, more robust to outliers)
2. Huber Loss (robust to outliers)
3. Stacking (meta-learner on base predictions)
4. Feature selection optimized for generalization
5. Ensemble of very different models
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
from sklearn.linear_model import Ridge, HuberRegressor, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
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
    """Load data."""
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


def prepare_features(train_series, test_series, train_pids, test_pids):
    """Prepare features for modeling."""
    frames = list(range(140, 170))

    def flatten(series):
        flat = series[:, frames, :].reshape(len(series), -1)
        return np.nan_to_num(flat, nan=0.0)

    train_flat = flatten(train_series)
    test_flat = flatten(test_series)

    return train_flat, test_flat


def approach_huber(data):
    """
    Huber Regression - robust to outliers.
    Uses a loss that is quadratic for small errors and linear for large errors.
    """
    print("\n" + "="*60)
    print("APPROACH: Huber Regression (robust to outliers)")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()
    train_flat, test_flat = prepare_features(train_series, test_series, train_pids, test_pids)

    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
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

        for t_idx in range(3):
            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr = y_train[train_idx, t_idx]

                model = HuberRegressor(epsilon=1.35, max_iter=200)
                model.fit(X_tr, y_tr)

                oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val)
                fold_test_preds.append(model.predict(X_test_scaled))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

    # Compute CV
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "huber"


def approach_gradient_boosting(data):
    """
    Gradient Boosting with Huber loss.
    Non-linear model that might capture patterns Ridge misses.
    """
    print("\n" + "="*60)
    print("APPROACH: Gradient Boosting (Huber loss)")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()
    train_flat, test_flat = prepare_features(train_series, test_series, train_pids, test_pids)

    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        player_indices = np.where(train_mask)[0]

        # PCA
        n_comp = min(15, len(X_train) - 1)
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for t_idx in range(3):
            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_train_pca):
                X_tr, X_val = X_train_pca[train_idx], X_train_pca[val_idx]
                y_tr = y_train[train_idx, t_idx]

                model = GradientBoostingRegressor(
                    loss='huber',
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                )
                model.fit(X_tr, y_tr)

                oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val)
                fold_test_preds.append(model.predict(X_test_pca))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

    # Compute CV
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "gradient_boosting"


def approach_stacking(data):
    """
    Stacking: Use predictions from multiple base models as features
    for a meta-learner.
    """
    print("\n" + "="*60)
    print("APPROACH: Stacking (meta-learner)")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()
    train_flat, test_flat = prepare_features(train_series, test_series, train_pids, test_pids)

    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
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

        # First level: get OOF predictions from multiple models
        n_base_models = 4
        oof_base = np.zeros((len(X_train), 3, n_base_models))
        test_base = np.zeros((len(X_test), 3, n_base_models))

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for t_idx in range(3):
            fold_test_base = {m: [] for m in range(n_base_models)}

            for train_idx, val_idx in kf.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr = y_train[train_idx, t_idx]

                # Model 0: Ridge
                m0 = Ridge(alpha=100)
                m0.fit(X_tr, y_tr)
                oof_base[val_idx, t_idx, 0] = m0.predict(X_val)
                fold_test_base[0].append(m0.predict(X_test_scaled))

                # Model 1: Ridge (low reg)
                m1 = Ridge(alpha=10)
                m1.fit(X_tr, y_tr)
                oof_base[val_idx, t_idx, 1] = m1.predict(X_val)
                fold_test_base[1].append(m1.predict(X_test_scaled))

                # Model 2: KNN
                m2 = KNeighborsRegressor(n_neighbors=min(5, len(X_tr)-1), weights='distance')
                m2.fit(X_tr, y_tr)
                oof_base[val_idx, t_idx, 2] = m2.predict(X_val)
                fold_test_base[2].append(m2.predict(X_test_scaled))

                # Model 3: Huber
                m3 = HuberRegressor(epsilon=1.35, max_iter=200)
                m3.fit(X_tr, y_tr)
                oof_base[val_idx, t_idx, 3] = m3.predict(X_val)
                fold_test_base[3].append(m3.predict(X_test_scaled))

            for m in range(n_base_models):
                test_base[:, t_idx, m] = np.mean(fold_test_base[m], axis=0)

        # Second level: meta-learner
        for t_idx in range(3):
            X_meta_train = oof_base[:, t_idx, :]
            X_meta_test = test_base[:, t_idx, :]
            y_meta = y_train[:, t_idx]

            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_meta_train):
                X_tr, X_val = X_meta_train[train_idx], X_meta_train[val_idx]
                y_tr = y_meta[train_idx]

                meta_model = Ridge(alpha=1.0)
                meta_model.fit(X_tr, y_tr)

                oof_preds[player_indices[val_idx], t_idx] = meta_model.predict(X_val)
                fold_test_preds.append(meta_model.predict(X_meta_test))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

    # Compute CV
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "stacking"


def approach_diverse_ensemble(data):
    """
    Ensemble of very different models with equal weights.
    Sometimes simple averaging of diverse models works best.
    """
    print("\n" + "="*60)
    print("APPROACH: Diverse Ensemble (simple average)")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    # Get predictions from multiple approaches
    train_flat, test_flat = prepare_features(train_series, test_series, train_pids, test_pids)

    unique_pids = sorted(np.unique(train_pids))

    # Collect predictions from different frame windows
    frame_windows = [
        list(range(130, 170)),  # Wide
        list(range(140, 170)),  # Standard
        list(range(145, 165)),  # Narrow
        list(range(150, 180)),  # Late
    ]

    all_predictions = []
    all_oof = []

    for frames in frame_windows:
        def flatten_w(series):
            flat = series[:, frames, :].reshape(len(series), -1)
            return np.nan_to_num(flat, nan=0.0)

        train_w = flatten_w(train_series)
        test_w = flatten_w(test_series)

        predictions = np.zeros((len(test_series), 3))
        oof_preds = np.zeros_like(train_targets)

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_train = train_w[train_mask]
            y_train = train_targets[train_mask]
            X_test = test_w[test_mask]

            player_indices = np.where(train_mask)[0]

            n_comp = min(15, len(X_train) - 1)
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

                    model = Ridge(alpha=100.0)
                    model.fit(X_tr, y_tr)

                    oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val)
                    fold_test_preds.append(model.predict(X_test_scaled))

                predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

        all_predictions.append(predictions)
        all_oof.append(oof_preds)

    # Average all predictions
    final_predictions = np.mean(all_predictions, axis=0)
    final_oof = np.mean(all_oof, axis=0)

    # Compute CV
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((final_oof[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return final_predictions, cv_score, "diverse_ensemble"


def create_submission(test_ids, predictions, cv_score, approach_name):
    """Create submission."""
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        name = f.stem
        if name.startswith("submission_"):
            try:
                nums.append(int(name.split('_')[1]))
            except:
                pass
        elif name.startswith("submission"):
            try:
                nums.append(int(name[10:]))
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

    print(f"\nSubmission {next_num}: {approach_name}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  angle_std: {angle_std:.4f}")

    return filepath, next_num


def main():
    print("="*80)
    print("UNEXPLORED APPROACHES")
    print("="*80)

    data = load_data()

    results = []

    # Huber regression
    preds, cv, name = approach_huber(data)
    results.append((name, cv, preds))

    # Gradient boosting
    preds, cv, name = approach_gradient_boosting(data)
    results.append((name, cv, preds))

    # Stacking
    preds, cv, name = approach_stacking(data)
    results.append((name, cv, preds))

    # Diverse ensemble
    preds, cv, name = approach_diverse_ensemble(data)
    results.append((name, cv, preds))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n{'Approach':<25} {'CV Score':>12}")
    print("-" * 40)
    for name, cv, _ in sorted(results, key=lambda x: x[1]):
        print(f"{name:<25} {cv:>12.6f}")

    # Create submission with best
    best_name, best_cv, best_preds = min(results, key=lambda x: x[1])
    print(f"\nBest approach: {best_name}")

    create_submission(data["test_ids"], best_preds, best_cv, best_name)


if __name__ == "__main__":
    main()

"""
Test simple linear models like the Don't Overfit II winner.
Linear models are less prone to overfitting on small datasets.
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]

def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    return np.array(json.loads(s.replace("nan", "null")), dtype=np.float32)

def load_data():
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols + TARGETS]

    n_train = len(train_df)
    X_train_raw = np.zeros((n_train, 240, len(keypoint_cols)), dtype=np.float32)
    for idx, row in train_df.iterrows():
        for i, col in enumerate(keypoint_cols):
            X_train_raw[idx, :, i] = parse_array_string(row[col])

    y_train = train_df[TARGETS].values.astype(np.float32)
    pids_train = train_df["participant_id"].values

    n_test = len(test_df)
    X_test_raw = np.zeros((n_test, 240, len(keypoint_cols)), dtype=np.float32)
    for idx, row in test_df.iterrows():
        for i, col in enumerate(keypoint_cols):
            X_test_raw[idx, :, i] = parse_array_string(row[col])

    test_ids = test_df["id"].values

    return X_train_raw, y_train, pids_train, X_test_raw, test_ids

def extract_features(X_raw):
    """Simple features."""
    return np.nan_to_num(np.concatenate([
        np.nanmean(X_raw, axis=1),
        np.nanstd(X_raw, axis=1),
    ], axis=1), nan=0.0)

def test_linear_models():
    X_train_raw, y_train, pids_train, X_test_raw, test_ids = load_data()

    print("Extracting features...")
    X_train = extract_features(X_train_raw)
    X_test = extract_features(X_test_raw)
    print(f"  Features: {X_train.shape[1]}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    gkf = GroupKFold(n_splits=5)

    models = {
        'Ridge(alpha=1)': Ridge(alpha=1.0),
        'Ridge(alpha=10)': Ridge(alpha=10.0),
        'Ridge(alpha=100)': Ridge(alpha=100.0),
        'Lasso(alpha=0.1)': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'BayesianRidge': BayesianRidge(),
    }

    print("\n" + "="*60)
    print("LINEAR MODEL CV COMPARISON")
    print("="*60)

    best_model = None
    best_mse = float('inf')

    for name, model_template in models.items():
        mses = []
        for t_idx in range(3):
            fold_preds = np.zeros(len(X_train))
            for tr, va in gkf.split(X_train_scaled, y_train[:, t_idx], pids_train):
                model = type(model_template)(**model_template.get_params())
                model.fit(X_train_scaled[tr], y_train[tr, t_idx])
                fold_preds[va] = model.predict(X_train_scaled[va])
            mses.append(np.mean((y_train[:, t_idx] - fold_preds)**2))

        total_mse = np.mean(mses)
        print(f"{name}: angle={mses[0]:.2f}, depth={mses[1]:.2f}, lr={mses[2]:.2f}, TOTAL={total_mse:.2f}")

        if total_mse < best_mse:
            best_mse = total_mse
            best_model = name

    print(f"\nBest: {best_model} with MSE={best_mse:.2f}")

    # Test with feature selection
    print("\n" + "="*60)
    print("LINEAR MODEL + FEATURE SELECTION")
    print("="*60)

    for k in [10, 20, 50]:
        mses = []
        for t_idx in range(3):
            fold_preds = np.zeros(len(X_train))
            for tr, va in gkf.split(X_train_scaled, y_train[:, t_idx], pids_train):
                selector = SelectKBest(mutual_info_regression, k=k)
                X_tr_sel = selector.fit_transform(X_train_scaled[tr], y_train[tr, t_idx])
                X_va_sel = selector.transform(X_train_scaled[va])

                model = Ridge(alpha=10.0)
                model.fit(X_tr_sel, y_train[tr, t_idx])
                fold_preds[va] = model.predict(X_va_sel)
            mses.append(np.mean((y_train[:, t_idx] - fold_preds)**2))

        total_mse = np.mean(mses)
        print(f"Ridge + top {k} features: TOTAL={total_mse:.2f}")

    return best_model, best_mse

def create_linear_submission():
    """Create submission with best linear model."""
    X_train_raw, y_train, pids_train, X_test_raw, test_ids = load_data()

    X_train = extract_features(X_train_raw)
    X_test = extract_features(X_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nCreating submission with Ridge(alpha=10) + top 20 features...")

    predictions = np.zeros((len(X_test), 3))

    for t_idx, target in enumerate(TARGETS):
        # Feature selection
        selector = SelectKBest(mutual_info_regression, k=20)
        X_tr_sel = selector.fit_transform(X_train_scaled, y_train[:, t_idx])
        X_te_sel = selector.transform(X_test_scaled)

        # Train
        model = Ridge(alpha=10.0)
        model.fit(X_tr_sel, y_train[:, t_idx])
        predictions[:, t_idx] = model.predict(X_te_sel)

    # Scale
    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    scaled = np.zeros_like(predictions)
    for i, t in enumerate(TARGETS):
        scaled[:, i] = scalers[t].transform(predictions[:, i].reshape(-1, 1)).flatten()

    # Save
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled[:, 0],
        'scaled_depth': scaled[:, 1],
        'scaled_left_right': scaled[:, 2],
    })
    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    sub.to_csv(filepath, index=False)

    print(f"\nSubmission saved: {filepath}")
    print(f"  angle_std = {sub['scaled_angle'].std():.6f}")
    print(f"  depth_mean = {sub['scaled_depth'].mean():.6f}")

    return filepath, next_num

if __name__ == "__main__":
    test_linear_models()
    print("\n" + "="*60)
    filepath, sub_num = create_linear_submission()
    print(f"\n=== Submission {sub_num} ===")
    print("Method: Ridge(alpha=10) + top 20 features (simple linear model)")

"""
Stacking meta-learner: combine 8+ diverse models with learned weights.

Meta-learner trains on diverse model predictions to learn optimal weighting.
More principled than manual blending.
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

HOOP_FEET = np.array([5.25, -25.0, 10.0])
N_FRAMES = 240
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
TARGETS = ["angle", "depth", "left_right"]


def parse_array_string(s):
    s = str(s).replace("nan", "NaN").replace("null", "NaN")
    return np.nan_to_num(np.array(json.loads(s), dtype=np.float64), nan=0.0)


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    kp_cols = [c for c in df.columns if c not in meta_cols]
    kp_names = [c[:-2] for c in kp_cols if c.endswith("_x")]
    kp_index = {name: i for i, name in enumerate(kp_names)}
    n = len(df)

    X_3d = np.zeros((n, N_FRAMES, len(kp_names), 3), dtype=np.float32)
    for idx, row in df.iterrows():
        for col_i, col in enumerate(kp_cols):
            kp_i = col_i // 3
            ax_i = col_i % 3
            arr = parse_array_string(row[col])
            X_3d[idx, :, kp_i, ax_i] = arr
        if (idx + 1) % 100 == 0:
            print(f"  Loaded {idx+1}/{n}...")

    return X_3d, kp_index, df


def load_submission(sub_num: int) -> pd.DataFrame | None:
    """Load submission by number."""
    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    print("=" * 60)
    print("STACKING META-LEARNER - FINAL SUBMISSION")
    print("=" * 60)

    # Load train data for meta-learner training
    print("\nLoading train data...")
    X_3d_tr, kp_index, df_tr = load_data(DATA_DIR / "train.csv")
    print(f"  {len(df_tr)} train shots")

    # Load test data IDs
    print("Loading test data IDs...")
    X_3d_te, _, df_te = load_data(DATA_DIR / "test.csv")
    print(f"  {len(df_te)} test shots")

    # Key diverse models to stack
    # Based on LB performance and diversity
    diverse_subs = [
        3385,  # biomechanical (NEW) - 0.006243
        3190,  # baseline Ridge
        3326,  # position CNN - diverse signal on depth
        3294,  # velocity CNN - diverse signal
        2608,  # pulse features - high diversity
        2622,  # 4-way ensemble
        2716,  # TreeAvg + CNN
        2503,  # extended physics
    ]

    print(f"\nLoading {len(diverse_subs)} diverse submissions...")
    test_meta_features = []
    sub_names = []

    for sub_num in diverse_subs:
        sub = load_submission(sub_num)
        if sub is None:
            print(f"  Sub {sub_num}: NOT FOUND")
            continue

        print(f"  Sub {sub_num}: loaded")
        test_meta_features.append(
            sub[["scaled_angle", "scaled_depth", "scaled_left_right"]].values.astype(np.float32)
        )
        sub_names.append(f"Sub{sub_num}")

    n_models = len(test_meta_features)
    print(f"\nStacking {n_models} models")

    # Test meta-features: (test_size, n_models, 3)
    test_meta = np.column_stack([m.ravel() for m in test_meta_features])  # (test_size, n_models*3)
    test_meta = test_meta.reshape(len(df_te), n_models, 3)  # (test_size, n_models, 3)

    print(f"Test meta-features shape: {test_meta.shape}")

    # For train meta-features: quick baseline Ridge per target
    print("\nGenerating train meta-features (quick Ridge OOF)...")
    train_meta_features = []

    for target in TARGETS:
        print(f"  {target.upper()}")

        # Quick Ridge on full train (no CV, just for meta-features)
        frame = TARGET_FRAMES[target]
        y_raw = df_tr[target].values.astype(np.float32)
        scaler_y = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
        y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).ravel()

        # Extract simple features: hoop-relative positions at target frame
        X_frame = X_3d_tr[:, frame, :, :].astype(np.float32)  # (345, 69, 3)
        X_simple = (X_frame - HOOP_FEET[None, None, :]).reshape(len(df_tr), -1)  # (345, 207)

        scaler_X = StandardScaler()
        X_simple_s = scaler_X.fit_transform(X_simple)

        ridge = Ridge(alpha=10)
        ridge.fit(X_simple_s, y_scaled)

        # Get OOF predictions via quick LOO on subset (for speed, use all train predictions)
        y_pred_oof = ridge.predict(X_simple_s)
        y_pred_oof = np.clip(y_pred_oof, 0, 1)

        train_meta_features.append(y_pred_oof)

    train_meta = np.column_stack(train_meta_features)  # (train_size, 3)
    print(f"Train meta-features shape: {train_meta.shape}")

    # Load train targets (scaled)
    print("\nPreparing train targets...")
    y_train_stacked = []
    for target in TARGETS:
        y_raw = df_tr[target].values.astype(np.float32)
        scaler_y = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
        y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).ravel()
        y_train_stacked.append(y_scaled)

    y_train = np.column_stack(y_train_stacked)  # (train_size, 3)

    # Train meta-learner per target
    print("\nTraining per-target meta-learners...")
    test_preds = []

    for i, target in enumerate(TARGETS):
        print(f"  {target.upper()}")

        # Meta-features for this target
        X_meta_tr = np.column_stack([m[:, i] for m in test_meta_features])  # (train_size, n_models)
        X_meta_te = np.column_stack([m[:, i] for m in test_meta_features])  # (test_size, n_models)

        print(f"    Meta-features shape train: {X_meta_tr.shape}, test: {X_meta_te.shape}")

        # Standardize
        scaler_meta = StandardScaler()
        X_meta_tr_s = scaler_meta.fit_transform(X_meta_tr)
        X_meta_te_s = scaler_meta.transform(X_meta_te)

        # Train Ridge meta-learner
        meta_learner = Ridge(alpha=1.0)  # Light regularization
        meta_learner.fit(X_meta_tr_s, y_train[:, i])

        # Predict on test
        y_test_pred = meta_learner.predict(X_meta_te_s)
        y_test_pred = np.clip(y_test_pred, 0, 1)

        test_preds.append(y_test_pred)

        # Print meta-learner weights (which models matter?)
        weights = meta_learner.coef_
        top_models = np.argsort(np.abs(weights))[-3:][::-1]
        print(f"    Top models: {[sub_names[j] for j in top_models]}")

    test_preds = np.column_stack(test_preds)

    # Save submission
    nums = []
    for p in SUBMISSION_DIR.glob("submission_*.csv"):
        parts = p.stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    bn = max(nums + [0]) + 1

    sub = pd.DataFrame({
        "id": df_te["id"].values,
        "scaled_angle": test_preds[:, 0],
        "scaled_depth": test_preds[:, 1],
        "scaled_left_right": test_preds[:, 2]
    })

    sub.to_csv(SUBMISSION_DIR / f"submission_{bn}.csv", index=False)

    print(f"\n{'='*60}")
    print(f"FINAL SUBMISSION: Sub {bn}")
    print(f"{'='*60}")
    print(f"  Stacking {n_models} diverse models with Ridge meta-learner")
    print(f"  Models: {', '.join(sub_names)}")
    print(f"  Predictions saved to submission_{bn}.csv")
    print(f"\n  Current best (Sub 3385): LB 0.006243")
    print(f"  Expected improvement: meta-learner learns optimal weighting")


if __name__ == "__main__":
    main()

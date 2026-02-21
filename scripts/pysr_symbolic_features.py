"""
PySR Symbolic Regression for Basketball Free Throw Prediction

Uses genetic programming (PySR) to discover mathematical formulas that map
keypoint features to targets. The key idea:
- PySR evolves symbolic expressions (like "sin(x1) + x2^2 / x3")
- These discovered formulas become NEW FEATURES for the Ridge pipeline
- Unlike PLS/hand-engineered features, these can capture nonlinear physics

Strategy:
1. Extract compact feature set per shot (key joints at release frame)
2. Run PySR per target to discover 3-5 best symbolic expressions
3. Evaluate those expressions on all shots as new features
4. Add to Ridge pipeline and measure LOO improvement + diversity

This is genuinely novel - symbolic regression has NOT been tried in this competition.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

# Key joints for symbolic regression - focus on shooting arm + body alignment
KEY_JOINTS = [
    "right_shoulder", "right_elbow", "right_wrist",
    "right_first_finger_mcp", "right_second_finger_mcp",
    "left_shoulder", "left_wrist",
    "right_hip", "left_hip",
    "nose",
]

HOOP_POS = np.array([5.25, -25.0, 10.0])


def load_data():
    """Load train and test data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, test_df


def extract_compact_features(df, frame_idx, joints=KEY_JOINTS):
    """Extract compact feature vector per shot at a specific frame.

    Returns position + velocity + angles for key joints, relative to hoop.
    ~60-80 features per shot.
    """
    shots = df.groupby("shot_id")
    feature_rows = []
    shot_ids = []
    player_ids = []

    for shot_id, group in shots:
        group = group.sort_values("frame_idx")
        frames = group["frame_idx"].values

        # Get hip center for normalization
        hip_cols = []
        for side in ["right_hip", "left_hip"]:
            for c in ["x", "y", "z"]:
                hip_cols.append(f"{side}_{c}")

        hip_data = group[hip_cols].values
        hip_center = (hip_data[:, :3] + hip_data[:, 3:6]) / 2  # (T, 3)

        # Find frame index
        if frame_idx in frames:
            fidx = np.where(frames == frame_idx)[0][0]
        else:
            fidx = np.argmin(np.abs(frames - frame_idx))

        features = {}

        # Position features at release frame (hoop-relative, hip-centered)
        for joint in joints:
            pos = np.array([
                group[f"{joint}_x"].values[fidx],
                group[f"{joint}_y"].values[fidx],
                group[f"{joint}_z"].values[fidx],
            ])
            pos_centered = pos - hip_center[fidx]
            pos_hoop = pos - HOOP_POS

            features[f"{joint}_cx"] = pos_centered[0]
            features[f"{joint}_cy"] = pos_centered[1]
            features[f"{joint}_cz"] = pos_centered[2]
            features[f"{joint}_hx"] = pos_hoop[0]
            features[f"{joint}_hy"] = pos_hoop[1]
            features[f"{joint}_hz"] = pos_hoop[2]

        # Velocity features (frame diff)
        if fidx > 0:
            dt = 1.0 / 60.0
            for joint in joints[:5]:  # Only shooting arm for velocity
                for c in ["x", "y", "z"]:
                    v1 = group[f"{joint}_{c}"].values[fidx]
                    v0 = group[f"{joint}_{c}"].values[fidx - 1]
                    features[f"{joint}_v{c}"] = (v1 - v0) / dt
        else:
            for joint in joints[:5]:
                for c in ["x", "y", "z"]:
                    features[f"{joint}_v{c}"] = 0.0

        # Derived angle features
        # Elbow angle
        rs = np.array([group[f"right_shoulder_{c}"].values[fidx] for c in "xyz"])
        re = np.array([group[f"right_elbow_{c}"].values[fidx] for c in "xyz"])
        rw = np.array([group[f"right_wrist_{c}"].values[fidx] for c in "xyz"])

        v1 = rs - re
        v2 = rw - re
        cos_elbow = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        features["elbow_angle"] = np.arccos(np.clip(cos_elbow, -1, 1))

        # Shoulder elevation
        shoulder_vec = re - rs
        features["shoulder_elev"] = np.arctan2(shoulder_vec[2], np.sqrt(shoulder_vec[0]**2 + shoulder_vec[1]**2))

        # Wrist-to-hoop direction alignment
        wrist_to_hoop = HOOP_POS - rw
        wrist_vel = np.array([features.get(f"right_wrist_v{c}", 0) for c in "xyz"])
        if np.linalg.norm(wrist_vel) > 1e-6:
            cos_align = np.dot(wrist_to_hoop, wrist_vel) / (np.linalg.norm(wrist_to_hoop) * np.linalg.norm(wrist_vel) + 1e-8)
            features["wrist_hoop_align"] = cos_align
        else:
            features["wrist_hoop_align"] = 0.0

        # Hip rotation (shoulder-hip relative angle)
        ls = np.array([group[f"left_shoulder_{c}"].values[fidx] for c in "xyz"])
        shoulder_vec_lr = rs - ls
        features["shoulder_width_angle"] = np.arctan2(shoulder_vec_lr[0], -shoulder_vec_lr[1])

        # Distance to hoop
        features["dist_to_hoop"] = np.linalg.norm(rw - HOOP_POS)
        features["dist_to_hoop_xy"] = np.linalg.norm((rw - HOOP_POS)[:2])
        features["height_above_hoop"] = rw[2] - HOOP_POS[2]

        feature_rows.append(features)
        shot_ids.append(shot_id)
        player_ids.append(group["player_id"].iloc[0])

    feat_df = pd.DataFrame(feature_rows)
    feat_df["shot_id"] = shot_ids
    feat_df["player_id"] = player_ids
    return feat_df


def run_pysr_per_target(X, y, target_name, n_iterations=40, maxsize=20,
                         populations=15, ncycles=300):
    """Run PySR to discover symbolic expressions for one target."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("Installing PySR...", flush=True)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pysr"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from pysr import PySRRegressor

    # PySR configuration
    model = PySRRegressor(
        niterations=n_iterations,
        populations=populations,
        ncycles_per_iteration=ncycles,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "sqrt_abs", "square", "abs"],
        maxsize=maxsize,
        # Parsimony to prefer simpler expressions
        parsimony=0.001,
        # Use multiple complexity levels
        select_k_features=min(15, X.shape[1]),  # PySR's built-in feature selection
        # Speed settings
        turbo=True,
        bumper=True,
        # Deterministic-ish
        random_state=42,
        # Output
        temp_equation_file=True,
        verbosity=1,
        progress=False,
    )

    print(f"  Running PySR for {target_name} with {X.shape[1]} features, "
          f"{n_iterations} iterations...", flush=True)

    t0 = time.time()
    model.fit(X, y)
    elapsed = time.time() - t0

    print(f"  PySR {target_name} done in {elapsed:.1f}s", flush=True)

    # Get the Pareto front of expressions
    equations = model.equations_
    if equations is not None and len(equations) > 0:
        print(f"  Found {len(equations)} equations on Pareto front", flush=True)
        # Get top 5 by score (complexity-penalized loss)
        top_eqs = equations.nsmallest(5, "loss")
        for _, row in top_eqs.iterrows():
            print(f"    complexity={row['complexity']}, loss={row['loss']:.6f}: {row['equation']}", flush=True)

    return model


def evaluate_symbolic_features(X_sym_train, X_base_train, y_train,
                                X_sym_test, X_base_test,
                                player_ids_train, player_ids_test,
                                target_name, target_idx):
    """Evaluate symbolic features added to the base Ridge pipeline via LOO."""

    n_train = len(y_train)

    # Per-player LOO evaluation
    loo_preds = np.zeros(n_train)
    loo_preds_base = np.zeros(n_train)

    for i in range(n_train):
        mask = np.ones(n_train, dtype=bool)
        mask[i] = False

        player = player_ids_train[i]
        player_mask = player_ids_train == player
        train_mask = mask & player_mask

        if train_mask.sum() < 3:
            train_mask = mask  # Fall back to all players

        # Base features only
        X_tr = X_base_train[train_mask]
        y_tr = y_train[train_mask]
        X_te = X_base_train[i:i+1]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Gaussian kernel weighting
        dists = cdist(X_te_s, X_tr_s, metric='euclidean').flatten()
        bw = np.quantile(dists, 0.3)
        if bw < 1e-8:
            bw = 1.0
        weights = np.exp(-0.5 * (dists / bw) ** 2)

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr_s, y_tr, sample_weight=weights)
        loo_preds_base[i] = np.clip(ridge.predict(X_te_s)[0], 0, 1)

        # Base + symbolic features
        X_tr_aug = np.hstack([X_base_train[train_mask], X_sym_train[train_mask]])
        X_te_aug = np.hstack([X_base_train[i:i+1], X_sym_train[i:i+1]])

        scaler2 = StandardScaler()
        X_tr_s2 = scaler2.fit_transform(X_tr_aug)
        X_te_s2 = scaler2.transform(X_te_aug)

        dists2 = cdist(X_te_s2, X_tr_s2, metric='euclidean').flatten()
        bw2 = np.quantile(dists2, 0.3)
        if bw2 < 1e-8:
            bw2 = 1.0
        weights2 = np.exp(-0.5 * (dists2 / bw2) ** 2)

        ridge2 = Ridge(alpha=10.0)
        ridge2.fit(X_tr_s2, y_tr, sample_weight=weights2)
        loo_preds[i] = np.clip(ridge2.predict(X_te_s2)[0], 0, 1)

    mse_base = np.mean((loo_preds_base - y_train) ** 2)
    mse_aug = np.mean((loo_preds - y_train) ** 2)
    improvement = (mse_aug - mse_base) / mse_base * 100

    print(f"  {target_name}: base LOO={mse_base:.6f}, +symbolic LOO={mse_aug:.6f}, "
          f"change={improvement:+.2f}%", flush=True)

    return mse_base, mse_aug, loo_preds, loo_preds_base


def main():
    print("=" * 70, flush=True)
    print("PySR Symbolic Regression Feature Discovery", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()

    # Load data
    print("\nLoading data...", flush=True)
    train_df, test_df = load_data()

    targets_train = {}
    for shot_id, group in train_df.groupby("shot_id"):
        for tc in TARGET_COLS:
            if tc not in targets_train:
                targets_train[tc] = {}
            targets_train[tc][shot_id] = group[tc].iloc[0]

    # Extract features for each target frame
    results = {}

    for tidx, (target_name, target_col) in enumerate(zip(TARGETS, TARGET_COLS)):
        print(f"\n{'='*60}", flush=True)
        print(f"TARGET: {target_name} (frame {TARGET_FRAMES[target_name]})", flush=True)
        print(f"{'='*60}", flush=True)

        frame = TARGET_FRAMES[target_name]

        # Extract compact features
        print("Extracting features...", flush=True)
        train_feats = extract_compact_features(train_df, frame)
        test_feats = extract_compact_features(test_df, frame)

        # Align shot order with targets
        shot_ids_train = train_feats["shot_id"].values
        player_ids_train = train_feats["player_id"].values
        y_train = np.array([targets_train[target_col][sid] for sid in shot_ids_train])

        # Feature matrix (drop shot_id, player_id)
        feat_cols = [c for c in train_feats.columns if c not in ["shot_id", "player_id"]]
        X_train = train_feats[feat_cols].values.astype(np.float64)
        X_test = test_feats[feat_cols].values.astype(np.float64)

        # Handle NaN/inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"Feature matrix: {X_train.shape[0]} shots x {X_train.shape[1]} features", flush=True)
        print(f"Target range: [{y_train.min():.4f}, {y_train.max():.4f}], "
              f"mean={y_train.mean():.4f}, std={y_train.std():.4f}", flush=True)

        # Standardize for PySR
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        # Run PySR
        print("\nRunning PySR...", flush=True)
        model = run_pysr_per_target(
            X_train_s, y_train, target_name,
            n_iterations=30,  # Reasonable for ~80 features
            maxsize=15,       # Don't want overly complex expressions
            populations=15,
            ncycles=200,
        )

        # Get predictions from top equations as new features
        if model.equations_ is not None and len(model.equations_) > 0:
            # Use predictions from multiple complexity levels as features
            X_test_s = scaler.transform(X_test)

            # Get predictions at different complexity levels
            sym_features_train = []
            sym_features_test = []

            eqs = model.equations_
            # Pick equations at different complexity levels
            complexities = sorted(eqs["complexity"].unique())
            selected_complexities = complexities[::max(1, len(complexities)//5)]  # ~5 spread out

            for comp in selected_complexities:
                subset = eqs[eqs["complexity"] == comp]
                best_eq = subset.loc[subset["loss"].idxmin()]

                # Predict using this equation
                try:
                    pred_train = model.predict(X_train_s, index=best_eq.name)
                    pred_test = model.predict(X_test_s, index=best_eq.name)
                    sym_features_train.append(pred_train)
                    sym_features_test.append(pred_test)
                    print(f"  Added equation (complexity={comp}): {best_eq['equation']}", flush=True)
                except Exception as e:
                    print(f"  Skipped complexity={comp}: {e}", flush=True)

            if sym_features_train:
                X_sym_train = np.column_stack(sym_features_train)
                X_sym_test = np.column_stack(sym_features_test)

                # Handle NaN/inf from symbolic expressions
                X_sym_train = np.nan_to_num(X_sym_train, nan=0.0, posinf=0.0, neginf=0.0)
                X_sym_test = np.nan_to_num(X_sym_test, nan=0.0, posinf=0.0, neginf=0.0)

                print(f"\n{X_sym_train.shape[1]} symbolic features generated", flush=True)

                # Evaluate
                mse_base, mse_aug, loo_preds, loo_preds_base = evaluate_symbolic_features(
                    X_sym_train, X_train, y_train,
                    X_sym_test, X_test,
                    player_ids_train, test_feats["player_id"].values,
                    target_name, tidx
                )

                results[target_name] = {
                    "mse_base": float(mse_base),
                    "mse_augmented": float(mse_aug),
                    "improvement_pct": float((mse_aug - mse_base) / mse_base * 100),
                    "n_sym_features": X_sym_train.shape[1],
                    "equations": [str(row["equation"]) for _, row in
                                 eqs.nsmallest(5, "loss").iterrows()],
                    "loo_preds": loo_preds.tolist(),
                    "loo_preds_base": loo_preds_base.tolist(),
                }
            else:
                print("  No valid symbolic features generated!", flush=True)
                results[target_name] = {"error": "No valid symbolic features"}
        else:
            print("  PySR found no equations!", flush=True)
            results[target_name] = {"error": "No equations found"}

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    for target_name in TARGETS:
        if target_name in results and "improvement_pct" in results[target_name]:
            r = results[target_name]
            print(f"  {target_name}: {r['improvement_pct']:+.2f}% "
                  f"(base={r['mse_base']:.6f}, +sym={r['mse_augmented']:.6f})", flush=True)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)", flush=True)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"pysr_symbolic_run_{ts}.json"

    # Remove non-serializable items from results for JSON
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: vv for kk, vv in v.items()
                          if kk not in ["loo_preds", "loo_preds_base"]}

    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

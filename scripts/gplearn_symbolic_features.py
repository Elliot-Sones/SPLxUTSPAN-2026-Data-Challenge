"""
Symbolic Regression Feature Discovery using gplearn

Uses genetic programming to evolve mathematical formulas that predict targets.
The discovered formulas become new features for the Ridge pipeline.

Key difference from hand-engineered features: gplearn searches the space of ALL
possible mathematical combinations (sin, cos, sqrt, *, /, +, -) to find the
best expressions. This can discover nonlinear physics relationships we missed.
"""

from __future__ import annotations

import fcntl
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

HOOP_POS = np.array([5.25, -25.0, 10.0])

# Focus on the most informative joints for shooting
KEY_JOINTS = [
    "right_shoulder", "right_elbow", "right_wrist",
    "right_first_finger_mcp", "right_second_finger_mcp",
    "left_shoulder", "left_wrist",
    "right_hip", "left_hip",
    "nose",
]


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_data():
    """Load and parse data into 3D arrays."""
    print("Loading data...", flush=True)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    kp_cols = [c for c in train_df.columns if c not in meta_cols]

    kp_names = []
    for col in kp_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}

    def process(df, is_train=True):
        n = len(df)
        n_kp = len(kp_names)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        pids = []
        shot_ids = []
        targets = []

        for idx in range(n):
            row = df.iloc[idx]
            for col_i, col in enumerate(kp_cols):
                arr = parse_array_string(row[col])
                X_3d[idx, :, col_i // 3, col_i % 3] = arr
            pids.append(row["participant_id"])
            shot_ids.append(row["shot_id"])
            if is_train:
                targets.append([row["angle"], row["depth"], row["left_right"]])

        result = {
            "X_3d": X_3d,
            "pids": np.array(pids),
            "shot_ids": np.array(shot_ids),
            "kp_index": kp_index,
            "kp_names": kp_names,
        }
        if is_train:
            result["y"] = np.array(targets, dtype=np.float32)
        return result

    print("  Processing train...", flush=True)
    train = process(train_df, True)
    print("  Processing test...", flush=True)
    test = process(test_df, False)
    return train, test


def extract_features(X_3d, pids, kp_index, frame_idx):
    """Extract compact feature vector per shot at a specific frame.

    X_3d: (N, 240, n_kp, 3) array
    Returns: (N, n_features) array, feature_names list
    """
    N = X_3d.shape[0]
    features = []
    feat_names = []

    # Hip center at frame
    rh_idx = kp_index["right_hip"]
    lh_idx = kp_index["left_hip"]
    hip_center = (X_3d[:, frame_idx, rh_idx, :] + X_3d[:, frame_idx, lh_idx, :]) / 2  # (N, 3)

    for j in KEY_JOINTS:
        ji = kp_index[j]
        pos = X_3d[:, frame_idx, ji, :]  # (N, 3)
        rel = pos - hip_center
        hoop_rel = pos - HOOP_POS[np.newaxis, :]

        for ci, cn in enumerate(["x", "y", "z"]):
            features.append(rel[:, ci])
            feat_names.append(f"{j}_r{cn}")
            features.append(hoop_rel[:, ci])
            feat_names.append(f"{j}_h{cn}")

    # Velocity (central diff) for shooting arm
    dt = 1.0 / 60.0
    for j in KEY_JOINTS[:6]:
        ji = kp_index[j]
        if frame_idx > 0 and frame_idx < 239:
            vel = (X_3d[:, frame_idx+1, ji, :] - X_3d[:, frame_idx-1, ji, :]) / (2*dt)
        elif frame_idx > 0:
            vel = (X_3d[:, frame_idx, ji, :] - X_3d[:, frame_idx-1, ji, :]) / dt
        else:
            vel = np.zeros((N, 3))
        for ci, cn in enumerate(["x", "y", "z"]):
            features.append(vel[:, ci])
            feat_names.append(f"{j}_v{cn}")

    # Acceleration for key joints
    for j in ["right_wrist", "right_elbow"]:
        ji = kp_index[j]
        if frame_idx > 0 and frame_idx < 239:
            acc = (X_3d[:, frame_idx+1, ji, :] - 2*X_3d[:, frame_idx, ji, :] + X_3d[:, frame_idx-1, ji, :]) / (dt**2)
        else:
            acc = np.zeros((N, 3))
        for ci, cn in enumerate(["x", "y", "z"]):
            features.append(acc[:, ci])
            feat_names.append(f"{j}_a{cn}")

    # Derived geometric features
    rs = X_3d[:, frame_idx, kp_index["right_shoulder"], :]
    re = X_3d[:, frame_idx, kp_index["right_elbow"], :]
    rw = X_3d[:, frame_idx, kp_index["right_wrist"], :]
    rf = X_3d[:, frame_idx, kp_index["right_first_finger_mcp"], :]
    ls = X_3d[:, frame_idx, kp_index["left_shoulder"], :]

    # Elbow angle
    v1 = rs - re
    v2 = rw - re
    cos_e = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8)
    features.append(np.arccos(np.clip(cos_e, -1, 1)))
    feat_names.append("elbow_angle")

    # Wrist angle
    v1 = re - rw
    v2 = rf - rw
    cos_w = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8)
    features.append(np.arccos(np.clip(cos_w, -1, 1)))
    feat_names.append("wrist_angle")

    # Shoulder elevation
    arm = re - rs
    features.append(np.arctan2(arm[:, 2], np.sqrt(arm[:, 0]**2 + arm[:, 1]**2)))
    feat_names.append("shoulder_elev")

    # Shoulder rotation
    shoulder_line = rs - ls
    features.append(np.arctan2(shoulder_line[:, 0], -shoulder_line[:, 1]))
    feat_names.append("shoulder_rot")

    # Hoop distances
    d_hoop = np.linalg.norm(rw - HOOP_POS[np.newaxis, :], axis=1)
    d_hoop_xy = np.linalg.norm((rw - HOOP_POS[np.newaxis, :])[:, :2], axis=1)
    h_diff = rw[:, 2] - HOOP_POS[2]
    features.extend([d_hoop, d_hoop_xy, h_diff])
    feat_names.extend(["dist_hoop", "dist_hoop_xy", "height_diff"])

    # Wrist velocity-hoop alignment
    wrist_idx = kp_index["right_wrist"]
    if frame_idx > 0 and frame_idx < 239:
        wrist_vel = (X_3d[:, frame_idx+1, wrist_idx, :] - X_3d[:, frame_idx-1, wrist_idx, :]) / (2*dt)
    elif frame_idx > 0:
        wrist_vel = (X_3d[:, frame_idx, wrist_idx, :] - X_3d[:, frame_idx-1, wrist_idx, :]) / dt
    else:
        wrist_vel = np.zeros((N, 3))

    hoop_dir = HOOP_POS[np.newaxis, :] - rw
    vel_norms = np.linalg.norm(wrist_vel, axis=1, keepdims=True)
    hoop_norms = np.linalg.norm(hoop_dir, axis=1, keepdims=True)
    align = np.sum(wrist_vel * hoop_dir, axis=1) / (vel_norms.flatten() * hoop_norms.flatten() + 1e-8)
    features.append(align)
    feat_names.append("vel_hoop_align")

    # Wrist speed
    features.append(np.linalg.norm(wrist_vel, axis=1))
    feat_names.append("wrist_speed")

    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feat_names


def locally_weighted_loo(X, y, pids, bw_q=0.3, alpha=10.0):
    """Simple LOO with locally weighted Ridge - no PLS to avoid leakage."""
    n = len(y)
    preds = np.zeros(n)
    unique_pids = np.unique(pids)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        # Per-player
        pmask = pids == pids[i]
        tmask = mask & pmask
        if tmask.sum() < 5:
            tmask = mask

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tmask])
        X_te = scaler.transform(X[i:i+1])

        dists = cdist(X_te, X_tr, metric='euclidean').flatten()
        bw = np.quantile(dists, bw_q)
        if bw < 1e-8:
            bw = 1.0
        w = np.exp(-0.5 * (dists / bw) ** 2)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr, y[tmask], sample_weight=w)
        preds[i] = np.clip(ridge.predict(X_te)[0], 0, 1)

    return preds


def predict_test(X_train, y_train, pids_train, X_test, pids_test, bw_q=0.3, alpha=10.0):
    """Predict test points using locally-weighted Ridge."""
    preds = np.zeros(len(X_test))
    for i in range(len(X_test)):
        pmask = pids_train == pids_test[i]
        if pmask.sum() < 5:
            pmask = np.ones(len(y_train), dtype=bool)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[pmask])
        X_te = scaler.transform(X_test[i:i+1])

        dists = cdist(X_te, X_tr, metric='euclidean').flatten()
        bw = np.quantile(dists, bw_q)
        if bw < 1e-8:
            bw = 1.0
        w = np.exp(-0.5 * (dists / bw) ** 2)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr, y_train[pmask], sample_weight=w)
        preds[i] = np.clip(ridge.predict(X_te)[0], 0, 1)
    return preds


def get_next_submission_number():
    SUBMISSION_DIR.mkdir(exist_ok=True)
    lock_path = SUBMISSION_DIR / ".submission_lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
        if existing:
            nums = []
            for p in existing:
                try:
                    nums.append(int(p.stem.split("_")[1]))
                except (ValueError, IndexError):
                    pass
            next_num = max(nums) + 1 if nums else 1
        else:
            next_num = 1
        placeholder = SUBMISSION_DIR / f"submission_{next_num}.csv"
        placeholder.touch()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return next_num


def save_submission(test_preds_df, desc):
    num = get_next_submission_number()
    path = SUBMISSION_DIR / f"submission_{num}.csv"
    test_preds_df.to_csv(path, index=False)
    print(f"  Saved submission_{num}: {desc}", flush=True)
    return num


def main():
    from gplearn.genetic import SymbolicTransformer, SymbolicRegressor

    print("=" * 70, flush=True)
    print("gplearn Symbolic Regression Feature Discovery", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()

    # Load data
    train, test = load_data()
    X_3d_train = train["X_3d"]
    X_3d_test = test["X_3d"]
    pids_train = train["pids"]
    pids_test = test["pids"]
    y_all = train["y"]  # (N, 3)
    kp_index = train["kp_index"]

    print(f"Train: {len(pids_train)} shots, Test: {len(pids_test)} shots", flush=True)

    # Scale targets to [0,1] using pre-fitted scalers
    scalers = {}
    for tname in TARGETS:
        scalers[tname] = joblib.load(DATA_DIR / f"scaler_{tname}.pkl")

    y_scaled = np.zeros_like(y_all)
    for tidx, tname in enumerate(TARGETS):
        y_scaled[:, tidx] = scalers[tname].transform(
            y_all[:, tidx].reshape(-1, 1)).ravel()

    print(f"Scaled target ranges:", flush=True)
    for tidx, tname in enumerate(TARGETS):
        print(f"  {tname}: [{y_scaled[:, tidx].min():.4f}, {y_scaled[:, tidx].max():.4f}]", flush=True)

    all_results = {}
    test_preds_all = {}

    for tidx, tname in enumerate(TARGETS):
        print(f"\n{'='*60}", flush=True)
        print(f"TARGET: {tname} (frame {TARGET_FRAMES[tname]})", flush=True)
        print(f"{'='*60}", flush=True)

        frame = TARGET_FRAMES[tname]
        y = y_scaled[:, tidx]

        # Extract features
        print("Extracting features...", flush=True)
        X_train, feat_names = extract_features(X_3d_train, pids_train, kp_index, frame)
        X_test, _ = extract_features(X_3d_test, pids_test, kp_index, frame)
        print(f"  {X_train.shape[1]} features extracted", flush=True)

        # Scale
        scaler_gp = StandardScaler()
        X_train_s = scaler_gp.fit_transform(X_train)
        X_test_s = scaler_gp.transform(X_test)

        # --- SymbolicTransformer: discover new features ---
        print("\n--- SymbolicTransformer ---", flush=True)

        st = SymbolicTransformer(
            population_size=2000,
            generations=20,
            tournament_size=20,
            hall_of_fame=100,
            n_components=10,
            function_set=["add", "sub", "mul", "div", "sqrt", "abs", "sin", "cos"],
            parsimony_coefficient=0.001,
            max_samples=1.0,
            feature_names=feat_names,
            random_state=42,
            verbose=1,
            n_jobs=1,
            metric="pearson",
        )

        t0 = time.time()
        st.fit(X_train_s, y)
        print(f"  Fitted in {time.time()-t0:.1f}s", flush=True)

        X_sym_train = st.transform(X_train_s)
        X_sym_test = st.transform(X_test_s)
        X_sym_train = np.nan_to_num(X_sym_train, nan=0, posinf=0, neginf=0)
        X_sym_test = np.nan_to_num(X_sym_test, nan=0, posinf=0, neginf=0)

        # Print discovered formulas
        for i, prog in enumerate(st._best_programs):
            corr = np.corrcoef(X_sym_train[:, i], y)[0, 1] if np.std(X_sym_train[:, i]) > 1e-8 else 0
            print(f"  F{i}: r={corr:.3f} | {prog}", flush=True)

        # --- SymbolicRegressor: direct prediction ---
        print("\n--- SymbolicRegressor ---", flush=True)

        sr = SymbolicRegressor(
            population_size=5000,
            generations=30,
            tournament_size=20,
            function_set=["add", "sub", "mul", "div", "sqrt", "abs", "sin", "cos"],
            parsimony_coefficient=0.0005,
            max_samples=1.0,
            feature_names=feat_names,
            random_state=42,
            verbose=1,
            n_jobs=1,
            metric="mse",
            stopping_criteria=0.001,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
        )

        t0 = time.time()
        sr.fit(X_train_s, y)
        print(f"  Fitted in {time.time()-t0:.1f}s", flush=True)
        print(f"  Best formula: {sr._program}", flush=True)

        sr_pred = sr.predict(X_train_s)
        sr_mse = np.mean((sr_pred - y) ** 2)
        sr_corr = np.corrcoef(sr_pred, y)[0, 1] if np.std(sr_pred) > 1e-8 else 0
        print(f"  Train MSE={sr_mse:.6f}, r={sr_corr:.4f}", flush=True)

        # --- LOO evaluation ---
        print("\n--- LOO Evaluation ---", flush=True)

        # Baseline
        preds_base = locally_weighted_loo(X_train, y, pids_train)
        mse_base = np.mean((preds_base - y) ** 2)
        print(f"  Base LOO: {mse_base:.6f}", flush=True)

        # Augmented (original + symbolic)
        X_aug_train = np.hstack([X_train, X_sym_train])
        preds_aug = locally_weighted_loo(X_aug_train, y, pids_train)
        mse_aug = np.mean((preds_aug - y) ** 2)
        pct_aug = (mse_aug - mse_base) / mse_base * 100
        print(f"  +Symbolic LOO: {mse_aug:.6f} ({pct_aug:+.2f}%)", flush=True)

        # Symbolic-only
        preds_sym = locally_weighted_loo(X_sym_train, y, pids_train)
        mse_sym = np.mean((preds_sym - y) ** 2)
        pct_sym = (mse_sym - mse_base) / mse_base * 100
        print(f"  Symbolic-only LOO: {mse_sym:.6f} ({pct_sym:+.2f}%)", flush=True)

        # --- Generate test predictions (augmented) ---
        X_aug_test = np.hstack([X_test, X_sym_test])
        test_preds = predict_test(X_aug_train, y, pids_train, X_aug_test, pids_test)
        test_preds_all[tname] = test_preds

        all_results[tname] = {
            "mse_base": float(mse_base),
            "mse_augmented": float(mse_aug),
            "mse_symbolic_only": float(mse_sym),
            "pct_augmented": float(pct_aug),
            "pct_symbolic_only": float(pct_sym),
            "sr_train_mse": float(sr_mse),
            "sr_train_corr": float(sr_corr),
            "best_formula": str(sr._program),
        }

    # --- Summary ---
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    for tname in TARGETS:
        r = all_results[tname]
        print(f"  {tname}:", flush=True)
        print(f"    Base: {r['mse_base']:.6f}", flush=True)
        print(f"    +Sym: {r['mse_augmented']:.6f} ({r['pct_augmented']:+.2f}%)", flush=True)
        print(f"    Formula: {r['best_formula']}", flush=True)

    mean_base = np.mean([all_results[t]["mse_base"] for t in TARGETS])
    mean_aug = np.mean([all_results[t]["mse_augmented"] for t in TARGETS])
    print(f"\n  Mean base: {mean_base:.6f}", flush=True)
    print(f"  Mean +sym: {mean_aug:.6f} ({(mean_aug-mean_base)/mean_base*100:+.2f}%)", flush=True)

    # --- Submissions ---
    print(f"\n--- Generating Submissions ---", flush=True)

    sub_df = pd.DataFrame({
        "shot_id": test["shot_ids"],
        "scaled_angle": test_preds_all["angle"],
        "scaled_depth": test_preds_all["depth"],
        "scaled_left_right": test_preds_all["left_right"],
    })

    SUB_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]

    # Standalone
    save_submission(sub_df, f"Standalone gplearn (mean LOO={mean_aug:.6f})")

    # Blends
    try:
        sub3507 = pd.read_csv(SUBMISSION_DIR / "submission_3507.csv")
        for w in [0.03, 0.05, 0.08]:
            blend = sub_df.copy()
            for tc in SUB_COLS:
                blend[tc] = w * sub_df[tc] + (1 - w) * sub3507[tc]
            save_submission(blend, f"{int(w*100)}% gplearn + {int((1-w)*100)}% Sub3507")
    except Exception as e:
        print(f"  Blend error: {e}", flush=True)

    # Diversity analysis
    print(f"\n--- Diversity vs Sub3507 ---", flush=True)
    try:
        sub3507 = pd.read_csv(SUBMISSION_DIR / "submission_3507.csv")
        for tc in SUB_COLS:
            r, _ = pearsonr(sub_df[tc], sub3507[tc])
            print(f"  {tc}: r={r:.4f}", flush=True)
    except Exception as e:
        print(f"  {e}", flush=True)

    total_time = time.time() - t_start
    print(f"\nTotal: {total_time:.1f}s ({total_time/60:.1f}min)", flush=True)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"gplearn_symbolic_run_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to output/gplearn_symbolic_run_{ts}.json", flush=True)


if __name__ == "__main__":
    main()

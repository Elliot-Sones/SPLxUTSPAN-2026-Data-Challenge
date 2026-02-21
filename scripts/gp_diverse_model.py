"""
Gaussian Process Regression for ensemble diversity.

GP is fundamentally different from all current models:
- Non-parametric (no fixed number of parameters)
- Kernel-based similarity (like our Ridge but non-linear)
- Naturally produces uncertainty estimates
- Completely different model family from Ridge, trees, k-NN, CNN

Strategy: Use GP with RBF kernel on compact features.
Per-player per-target training like our other models.
Multi-seed via different kernel hyperparameters.

Also: Kernel Ridge Regression with RBF kernel (faster than GP, still non-linear)
"""

import json
import time
import fcntl
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])


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


def load_and_extract_features():
    """Load data and extract compact features for GP/KRR."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])

    # Key joints
    key_joints = [
        "right_wrist", "right_elbow", "right_shoulder",
        "left_wrist", "left_elbow", "left_shoulder",
        "right_hip", "left_hip", "mid_hip",
        "right_knee", "left_knee", "neck", "nose",
        "right_ankle", "left_ankle",
    ]

    # Target frames (from best pipeline)
    target_frames = {"angle": 153, "depth": 150, "left_right": 170}

    def extract(df, is_train=True):
        n = len(df)
        features_list = []
        ids, pids, targets = [], [], []

        for idx in range(n):
            row = df.iloc[idx]

            # Parse all keypoint timeseries
            kp_data = {}
            for col in keypoint_cols:
                arr = parse_array_string(row[col])
                kp_data[col] = arr

            # Extract features for each target frame
            feat_row = []

            for frame_key, frame_num in target_frames.items():
                for jname in key_joints:
                    jx = kp_data.get(f"{jname}_x", np.zeros(240))
                    jy = kp_data.get(f"{jname}_y", np.zeros(240))
                    jz = kp_data.get(f"{jname}_z", np.zeros(240))

                    # Position at frame
                    feat_row.extend([
                        jx[frame_num] if frame_num < 240 else 0,
                        jy[frame_num] if frame_num < 240 else 0,
                        jz[frame_num] if frame_num < 240 else 0,
                    ])

                    # Velocity at frame (finite diff)
                    if frame_num > 0 and frame_num < 240:
                        feat_row.extend([
                            (jx[frame_num] - jx[frame_num-1]) * 60,
                            (jy[frame_num] - jy[frame_num-1]) * 60,
                            (jz[frame_num] - jz[frame_num-1]) * 60,
                        ])
                    else:
                        feat_row.extend([0, 0, 0])

            # Hoop-relative distance and angle
            mid_hip_x = kp_data.get("mid_hip_x", np.zeros(240))
            mid_hip_y = kp_data.get("mid_hip_y", np.zeros(240))
            mid_hip_z = kp_data.get("mid_hip_z", np.zeros(240))
            pos_120 = np.array([mid_hip_x[120], mid_hip_y[120], mid_hip_z[120]])
            hoop_dist = np.linalg.norm(HOOP_POS - pos_120)
            hoop_angle = np.arctan2(HOOP_POS[1] - pos_120[1], HOOP_POS[0] - pos_120[0])
            feat_row.extend([hoop_dist, hoop_angle])

            features_list.append(feat_row)
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

        X = np.array(features_list, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        result = {'X': X, 'pids': np.array(pids), 'ids': np.array(ids)}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    return extract(train_df, True), extract(test_df, False)


def run_model(train_data, test_data, model_type="krr"):
    """Run GP or Kernel Ridge per-player per-target."""
    X_train = train_data['X']
    y_raw = train_data['y']
    pids_train = train_data['pids']
    X_test = test_data['X']
    pids_test = test_data['pids']

    # Scale targets to [0, 1]
    target_scalers = {}
    y_scaled = np.zeros_like(y_raw)
    for t in range(3):
        scaler = MinMaxScaler()
        y_scaled[:, t] = scaler.fit_transform(y_raw[:, t:t+1]).ravel()
        target_scalers[t] = scaler

    unique_pids = sorted(set(pids_train.tolist()))
    n_test = len(X_test)
    test_preds = np.zeros((n_test, 3))
    loo_preds = np.zeros((len(X_train), 3))

    print(f"\nRunning {model_type.upper()} per-player per-target...")
    print(f"  Features: {X_train.shape[1]}")

    for pid in unique_pids:
        train_mask = pids_train == pid
        test_mask = pids_test == pid
        X_p_train = X_train[train_mask]
        y_p_scaled = y_scaled[train_mask]
        n_p = X_p_train.shape[0]

        # Scale features per player
        scaler = StandardScaler()
        X_p_scaled = scaler.fit_transform(X_p_train)

        if test_mask.any():
            X_p_test = scaler.transform(X_test[test_mask])

        for t in range(3):
            target_name = TARGETS[t]

            if model_type == "krr":
                # Kernel Ridge with RBF - multiple kernel widths averaged
                preds_list = []
                for gamma in [0.001, 0.005, 0.01, 0.05, 0.1]:
                    for alpha in [0.1, 1.0, 10.0]:
                        model = KernelRidge(kernel='rbf', gamma=gamma, alpha=alpha)
                        model.fit(X_p_scaled, y_p_scaled[:, t])
                        if test_mask.any():
                            preds_list.append(model.predict(X_p_test))

                if test_mask.any() and preds_list:
                    test_preds[test_mask, t] = np.mean(preds_list, axis=0)

                # LOO for KRR (faster than GP)
                train_idx = np.where(train_mask)[0]
                for i, idx in enumerate(train_idx):
                    mask_loo = np.ones(n_p, dtype=bool)
                    mask_loo[i] = False
                    loo_list = []
                    for gamma in [0.001, 0.01, 0.1]:
                        for alpha in [0.1, 1.0]:
                            m = KernelRidge(kernel='rbf', gamma=gamma, alpha=alpha)
                            m.fit(X_p_scaled[mask_loo], y_p_scaled[mask_loo, t])
                            loo_list.append(m.predict(X_p_scaled[i:i+1])[0])
                    loo_preds[idx, t] = np.mean(loo_list)

            elif model_type == "gp":
                # GP with RBF kernel
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                                  normalize_y=True, alpha=1e-3)
                model.fit(X_p_scaled, y_p_scaled[:, t])

                if test_mask.any():
                    pred, std = model.predict(X_p_test, return_std=True)
                    test_preds[test_mask, t] = pred

                # LOO
                train_idx = np.where(train_mask)[0]
                for i, idx in enumerate(train_idx):
                    mask_loo = np.ones(n_p, dtype=bool)
                    mask_loo[i] = False
                    m = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                                  normalize_y=True, alpha=1e-3)
                    m.fit(X_p_scaled[mask_loo], y_p_scaled[mask_loo, t])
                    loo_preds[idx, t] = m.predict(X_p_scaled[i:i+1])[0]

        print(f"  Player {pid}: done ({n_p} train, {test_mask.sum()} test)")

    # Clip predictions
    test_preds = np.clip(test_preds, 0, 1)
    loo_preds = np.clip(loo_preds, 0, 1)

    # LOO metrics
    print(f"\n=== {model_type.upper()} LOO Results ===")
    for t in range(3):
        mse = np.mean((loo_preds[:, t] - y_scaled[:, t]) ** 2)
        print(f"  {TARGETS[t]}: MSE = {mse:.6f}")
    mean_mse = np.mean([(np.mean((loo_preds[:, t] - y_scaled[:, t]) ** 2)) for t in range(3)])
    print(f"  Mean MSE: {mean_mse:.6f}")

    return test_preds, loo_preds, mean_mse


def main():
    train_data, test_data = load_and_extract_features()
    print(f"Train: {train_data['X'].shape}, Test: {test_data['X'].shape}")

    # Run Kernel Ridge (faster and usually comparable to GP)
    test_preds_krr, loo_preds_krr, loo_mse_krr = run_model(train_data, test_data, "krr")

    # Diversity check
    anchor_path = SUBMISSION_DIR / "submission_2716.csv"
    if anchor_path.exists():
        anchor = pd.read_csv(anchor_path)
        print("\nDiversity vs Sub 2716:")
        for t, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
            r = np.corrcoef(test_preds_krr[:, t], anchor[col].values)[0, 1]
            print(f"  {TARGETS[t]}: r = {r:.3f}")

    # Kill criteria
    if loo_mse_krr > 0.015:
        print("\nKILL: LOO too weak (> 0.015)")
        return

    # Save standalone submission
    sub_num = get_next_submission_number()
    sub_df = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': test_preds_krr[:, 0],
        'scaled_depth': test_preds_krr[:, 1],
        'scaled_left_right': test_preds_krr[:, 2],
    })
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"\nKRR standalone: {sub_path}")

    # Generate blend candidates
    if anchor_path.exists():
        anchor = pd.read_csv(anchor_path)
        for w in [0.02, 0.05, 0.10]:
            blend_num = get_next_submission_number()
            blend_df = anchor.copy()
            for t, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
                blend_df[col] = (1 - w) * anchor[col].values + w * test_preds_krr[:, t]
                blend_df[col] = blend_df[col].clip(0, 1)
            blend_path = SUBMISSION_DIR / f"submission_{blend_num}.csv"
            blend_df.to_csv(blend_path, index=False)
            print(f"  Sub {blend_num}: {int(w*100)}% KRR + {int((1-w)*100)}% Sub 2716")

        # Per-target: use KRR only for most diverse target
        for t, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
            r = np.corrcoef(test_preds_krr[:, t], anchor[col].values)[0, 1]
            if r < 0.85:  # Only if diverse enough
                for w in [0.03, 0.05, 0.08]:
                    blend_num = get_next_submission_number()
                    blend_df = anchor.copy()
                    blend_df[col] = (1 - w) * anchor[col].values + w * test_preds_krr[:, t]
                    blend_df[col] = blend_df[col].clip(0, 1)
                    blend_path = SUBMISSION_DIR / f"submission_{blend_num}.csv"
                    blend_df.to_csv(blend_path, index=False)
                    print(f"  Sub {blend_num}: {int(w*100)}% KRR {TARGETS[t]}-only + {int((1-w)*100)}% Sub 2716 (r={r:.3f})")

    print("\nDone!")


if __name__ == "__main__":
    main()

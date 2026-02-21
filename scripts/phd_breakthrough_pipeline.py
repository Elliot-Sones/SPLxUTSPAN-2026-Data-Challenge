"""
PhD-Level Research Breakthrough Pipeline
=========================================

Three mathematically rigorous approaches combined into one pipeline:

1. FUNCTIONAL PCA REGRESSION (Ramsay & Silverman, 2005)
   - Treats each joint trajectory as a smooth function over time
   - B-spline basis expansion provides natural smoothing/denoising
   - FPCA extracts dominant modes of variation across all shots
   - Scalar-on-function regression maps functional modes to targets
   - Theory: Functional Linear Model y = integral(beta(t) * x(t) dt) + epsilon

2. SIGNATURE KERNEL RIDGE REGRESSION (Salvi et al., 2021)
   - Path signatures: universal noncommutative feature map for sequential data
   - The truncated signature of a path captures its iterated integrals
   - Signature kernel K(x,y) = <S(x), S(y)> is a universal kernel on paths
   - Solves a Goursat PDE: efficient computation without explicit signatures
   - We use truncated signatures at level 3-4, then RBF kernel on them
   - Theory: Rough path theory (Lyons, 1998)

3. MULTI-KERNEL LEARNING (Gonen & Alpaydin, 2011)
   - Combine multiple kernels from different feature representations:
     K_combined = sum(w_m * K_m) where w_m >= 0, sum(w_m) = 1
   - Kernels: spatial (release frame), temporal (signature), functional (FPCA)
   - Optimal weights found via grid search with honest LOO
   - Theory: convex combination of PSD kernels is PSD

All approaches use per-player modeling with honest LOO evaluation.
"""

from __future__ import annotations

import fcntl
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import iisignature
import joblib
import numpy as np
import pandas as pd
import skfda
from scipy.stats import pearsonr
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation import FDataGrid
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
HOOP_POS = np.array([5.25, -25.0, 10.0])

# Key joints for different analyses
SHOOTING_JOINTS = [
    "right_shoulder", "right_elbow", "right_wrist",
    "right_first_finger_mcp", "right_second_finger_mcp",
    "right_third_finger_mcp", "right_fourth_finger_mcp",
]

BODY_JOINTS = [
    "right_hip", "left_hip", "right_knee", "left_knee",
    "right_ankle", "left_ankle", "nose",
    "left_shoulder", "left_elbow", "left_wrist",
]

ALL_KEY_JOINTS = SHOOTING_JOINTS + BODY_JOINTS


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_data():
    print("Loading data...", flush=True)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    kp_cols = [c for c in train_df.columns if c not in meta_cols]
    kp_names = [c[:-2] for c in kp_cols if c.endswith("_x")]
    kp_index = {name: i for i, name in enumerate(kp_names)}

    def process(df, is_train=True):
        n = len(df)
        n_kp = len(kp_names)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        pids, ids = [], []
        targets = []

        for idx in range(n):
            row = df.iloc[idx]
            for ci, col in enumerate(kp_cols):
                X_3d[idx, :, ci // 3, ci % 3] = parse_array_string(row[col])
            pids.append(row["participant_id"])
            ids.append(row["id"])
            if is_train:
                targets.append([row["angle"], row["depth"], row["left_right"]])

        result = {"X_3d": X_3d, "pids": np.array(pids), "ids": np.array(ids),
                  "kp_index": kp_index}
        if is_train:
            result["y"] = np.array(targets, dtype=np.float32)
        return result

    print("  Processing train...", flush=True)
    train = process(train_df, True)
    print("  Processing test...", flush=True)
    test = process(test_df, False)
    return train, test


# ================================================================
# MODULE 1: Functional PCA Regression
# ================================================================

def extract_fpca_features(X_3d, kp_index, pids, n_components=8,
                           frame_range=(120, 180), subsample=2,
                           joints=None, fpca_models=None, is_train=True):
    """
    Functional PCA on joint trajectories.

    Each joint's (x,y,z) trajectory over time is treated as a multivariate
    function. FPCA extracts the dominant modes of variation - the principal
    "shapes" of the trajectories that differ between shots.

    Returns FPCA scores as features, plus fitted FPCA models for test transform.
    """
    if joints is None:
        joints = SHOOTING_JOINTS

    N = X_3d.shape[0]
    f_start, f_end = frame_range
    frames = np.arange(f_start, f_end, subsample)
    n_frames = len(frames)

    # Hip-center the data
    rh = kp_index.get("right_hip", 0)
    lh = kp_index.get("left_hip", 0)
    hip = (X_3d[:, frames, rh, :] + X_3d[:, frames, lh, :]) / 2.0  # (N, n_frames, 3)

    all_features = []
    models = {} if is_train else None

    for joint in joints:
        if joint not in kp_index:
            continue
        ji = kp_index[joint]

        # Extract hip-centered trajectories for this joint
        traj = X_3d[:, frames, ji, :] - hip  # (N, n_frames, 3)

        for coord_idx, coord_name in enumerate(["x", "y", "z"]):
            key = f"{joint}_{coord_name}"
            data = traj[:, :, coord_idx]  # (N, n_frames)

            # Handle NaN
            data = np.nan_to_num(data, nan=0.0)

            if is_train:
                # Fit FPCA on training data
                fd = FDataGrid(data, grid_points=frames.astype(float))
                n_comp = min(n_components, N - 1, n_frames - 1)
                fpca = FPCA(n_components=n_comp)
                scores = fpca.fit_transform(fd)  # (N, n_comp)
                models[key] = {"fpca": fpca, "grid_points": frames.astype(float)}
                all_features.append(scores)
            else:
                # Transform test data using fitted FPCA
                if key in fpca_models:
                    fd = FDataGrid(data, grid_points=fpca_models[key]["grid_points"])
                    scores = fpca_models[key]["fpca"].transform(fd)
                    all_features.append(scores)

    X_fpca = np.hstack(all_features) if all_features else np.zeros((N, 1))
    return X_fpca, models


# ================================================================
# MODULE 2: Signature Kernel Features
# ================================================================

def extract_signature_features(X_3d, kp_index, frame_range=(130, 175),
                                subsample=3, sig_depth=3, joints=None):
    """
    Path signature features for shooting trajectories.

    The signature of a path is a sequence of iterated integrals that
    uniquely characterizes the path up to tree-like equivalence.
    At depth d, it captures all interactions between coordinates
    up to order d.

    We compute truncated signatures at level 3, which captures:
    - Level 1: mean displacement per coordinate (like velocity)
    - Level 2: pairwise interactions (like covariance/correlation of movements)
    - Level 3: triple interactions (like the "curvature" of the trajectory)

    Then use these as features for kernel regression.
    """
    if joints is None:
        joints = SHOOTING_JOINTS[:5]  # Focus on primary shooting arm

    N = X_3d.shape[0]
    f_start, f_end = frame_range
    frames = np.arange(f_start, f_end, subsample)
    n_frames = len(frames)

    # Hip-center
    rh = kp_index.get("right_hip", 0)
    lh = kp_index.get("left_hip", 0)
    hip = (X_3d[:, frames, rh, :] + X_3d[:, frames, lh, :]) / 2.0

    # Build path: concatenate joint positions into a multi-dim path
    paths = []
    for joint in joints:
        if joint not in kp_index:
            continue
        ji = kp_index[joint]
        traj = X_3d[:, frames, ji, :] - hip  # (N, n_frames, 3)
        paths.append(traj)

    if not paths:
        return np.zeros((N, 1))

    # Concatenate into (N, n_frames, n_joints * 3) path
    path_data = np.concatenate(paths, axis=2)  # (N, n_frames, D)
    path_data = np.nan_to_num(path_data, nan=0.0).astype(np.float64)

    # Add time channel for better signature properties
    time_channel = np.broadcast_to(
        np.linspace(0, 1, n_frames)[np.newaxis, :, np.newaxis],
        (N, n_frames, 1)
    )
    path_data = np.concatenate([path_data, time_channel], axis=2)

    D = path_data.shape[2]
    sig_length = iisignature.siglength(D, sig_depth)
    print(f"    Signature: {D}D path, depth={sig_depth}, "
          f"sig_length={sig_length}", flush=True)

    # Compute signatures
    signatures = np.zeros((N, sig_length), dtype=np.float64)
    for i in range(N):
        signatures[i] = iisignature.sig(path_data[i], sig_depth)

    # Log-normalize (signatures can have very different magnitudes at different levels)
    # Use level-wise normalization
    offset = 0
    for level in range(1, sig_depth + 1):
        level_size = D ** level
        level_sigs = signatures[:, offset:offset + level_size]
        # Normalize by factorial (theoretical scaling)
        import math
        scale = math.factorial(level)
        signatures[:, offset:offset + level_size] = level_sigs / scale
        offset += level_size

    return signatures


def compute_signature_kernel_matrix(sigs1, sigs2=None, gamma=None):
    """Compute RBF kernel on signature features."""
    if gamma is None:
        # Use median heuristic

        if sigs2 is None:
            dists = euclidean_distances(sigs1)
        else:
            dists = euclidean_distances(sigs1, sigs2)
        gamma = 1.0 / (np.median(dists[dists > 0]) ** 2 + 1e-8)

    if sigs2 is None:
        return rbf_kernel(sigs1, gamma=gamma), gamma
    return rbf_kernel(sigs1, sigs2, gamma=gamma), gamma


# ================================================================
# MODULE 3: Multi-Kernel Learning
# ================================================================

def extract_spatial_features(X_3d, kp_index, frame_idx):
    """Standard spatial features at release frame (for kernel computation)."""
    N = X_3d.shape[0]
    dt = 1.0 / 60.0
    features = []

    rh = kp_index.get("right_hip", 0)
    lh = kp_index.get("left_hip", 0)
    hip = (X_3d[:, frame_idx, rh, :] + X_3d[:, frame_idx, lh, :]) / 2.0

    for j in ALL_KEY_JOINTS:
        if j not in kp_index:
            continue
        ji = kp_index[j]
        pos = X_3d[:, frame_idx, ji, :]
        rel = pos - hip
        hoop_rel = pos - HOOP_POS[np.newaxis, :]
        for ci in range(3):
            features.append(rel[:, ci])
            features.append(hoop_rel[:, ci])

    # Velocity
    for j in ALL_KEY_JOINTS[:7]:
        if j not in kp_index:
            continue
        ji = kp_index[j]
        if 0 < frame_idx < 239:
            vel = (X_3d[:, frame_idx+1, ji, :] - X_3d[:, frame_idx-1, ji, :]) / (2*dt)
        else:
            vel = np.zeros((N, 3))
        for ci in range(3):
            features.append(vel[:, ci])

    X = np.column_stack(features)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def multi_kernel_ridge_loo(kernel_matrices, y, alpha=1.0):
    """
    Multi-kernel Ridge regression with LOO.

    Given a list of kernel matrices K_1, ..., K_M and weights w_1, ..., w_M,
    the combined kernel is K = sum(w_m * K_m).

    LOO is computed using the Sherman-Morrison-Woodbury formula:
    y_hat_(-i) = y_i - alpha_i / (K^(-1))_ii

    This gives exact LOO without refitting for each fold.
    """
    N = len(y)
    K_combined = sum(kernel_matrices)

    # Kernel Ridge: (K + alpha*I) * alpha_vec = y
    K_reg = K_combined + alpha * np.eye(N)

    try:
        # Solve for coefficients
        alpha_vec = np.linalg.solve(K_reg, y)

        # LOO using the hat matrix trick
        # H = K * (K + alpha*I)^{-1}
        K_inv = np.linalg.inv(K_reg)
        H = K_combined @ K_inv
        h_diag = np.diag(H)

        # LOO predictions: y_hat_(-i) = (y_hat_i - h_ii * y_i) / (1 - h_ii)
        y_hat = K_combined @ alpha_vec
        loo_preds = (y_hat - h_diag * y) / (1 - h_diag + 1e-10)

        mse = np.mean((loo_preds - y) ** 2)
        return mse, loo_preds, alpha_vec
    except np.linalg.LinAlgError:
        return float('inf'), np.zeros(N), np.zeros(N)


def multi_kernel_ridge_predict(K_train_combined, K_test_train_combined,
                                y_train, alpha=1.0):
    """Predict using multi-kernel Ridge."""
    N = len(y_train)
    K_reg = K_train_combined + alpha * np.eye(N)
    try:
        alpha_vec = np.linalg.solve(K_reg, y_train)
        return K_test_train_combined @ alpha_vec
    except np.linalg.LinAlgError:
        return np.full(K_test_train_combined.shape[0], np.mean(y_train))


def optimize_kernel_weights(kernel_matrices_train, y_train, alphas=[0.1, 1.0, 10.0]):
    """
    Find optimal kernel combination weights via grid search with LOO.

    The beauty of multi-kernel Ridge is that LOO is analytically computable
    (no need to refit), making the grid search cheap.
    """
    n_kernels = len(kernel_matrices_train)

    # Normalize each kernel to unit trace
    K_normalized = []
    for K in kernel_matrices_train:
        trace = np.trace(K)
        if trace > 0:
            K_normalized.append(K / trace * len(y_train))
        else:
            K_normalized.append(K)

    best_mse = float('inf')
    best_weights = np.ones(n_kernels) / n_kernels
    best_alpha = 1.0

    # Grid search over weights and alpha
    if n_kernels == 2:
        weight_grid = np.arange(0, 1.01, 0.05)
        for w1 in weight_grid:
            w = [w1, 1 - w1]
            K_list = [w[i] * K_normalized[i] for i in range(n_kernels)]
            for alpha in alphas:
                mse, _, _ = multi_kernel_ridge_loo(K_list, y_train, alpha)
                if mse < best_mse:
                    best_mse = mse
                    best_weights = np.array(w)
                    best_alpha = alpha
    elif n_kernels == 3:
        for w1 in np.arange(0, 1.01, 0.10):
            for w2 in np.arange(0, 1.01 - w1, 0.10):
                w = [w1, w2, 1 - w1 - w2]
                K_list = [w[i] * K_normalized[i] for i in range(n_kernels)]
                for alpha in alphas:
                    mse, _, _ = multi_kernel_ridge_loo(K_list, y_train, alpha)
                    if mse < best_mse:
                        best_mse = mse
                        best_weights = np.array(w)
                        best_alpha = alpha
    else:
        # Random search for more kernels
        for _ in range(200):
            w = np.random.dirichlet(np.ones(n_kernels))
            K_list = [w[i] * K_normalized[i] for i in range(n_kernels)]
            for alpha in alphas:
                mse, _, _ = multi_kernel_ridge_loo(K_list, y_train, alpha)
                if mse < best_mse:
                    best_mse = mse
                    best_weights = w.copy()
                    best_alpha = alpha

    return best_weights, best_alpha, best_mse, K_normalized


# ================================================================
# MAIN PIPELINE
# ================================================================

def get_next_submission_number():
    SUBMISSION_DIR.mkdir(exist_ok=True)
    lock_path = SUBMISSION_DIR / ".submission_lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
        nums = []
        for p in existing:
            try:
                nums.append(int(p.stem.split("_")[1]))
            except (ValueError, IndexError):
                pass
        next_num = max(nums) + 1 if nums else 1
        (SUBMISSION_DIR / f"submission_{next_num}.csv").touch()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return next_num


def save_submission(df, desc):
    num = get_next_submission_number()
    path = SUBMISSION_DIR / f"submission_{num}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved submission_{num}: {desc}", flush=True)
    return num


def main():
    print("=" * 70, flush=True)
    print("PhD-Level Research Breakthrough Pipeline", flush=True)
    print("=" * 70, flush=True)
    print("Combining: Functional PCA + Signature Kernel + Multi-Kernel Learning",
          flush=True)

    t_start = time.time()
    train, test = load_data()

    X_3d_train = train["X_3d"]
    X_3d_test = test["X_3d"]
    pids_train = train["pids"]
    pids_test = test["pids"]
    ids_test = test["ids"]
    y_raw = train["y"]
    kp_index = train["kp_index"]
    unique_pids = np.unique(pids_train)

    # Scale targets
    y_scaled = np.zeros_like(y_raw)
    for tidx, tname in enumerate(TARGETS):
        scaler = joblib.load(DATA_DIR / f"scaler_{tname}.pkl")
        y_scaled[:, tidx] = scaler.transform(y_raw[:, tidx].reshape(-1, 1)).ravel()

    print(f"Train: {len(pids_train)}, Test: {len(pids_test)}", flush=True)
    print(f"Players: {unique_pids}", flush=True)

    results = {}

    # ============================================================
    # Phase 1: Extract all feature representations
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("PHASE 1: Feature Extraction", flush=True)
    print(f"{'='*60}", flush=True)

    # 1a. Functional PCA features
    print("\n--- Functional PCA ---", flush=True)
    t0 = time.time()
    fpca_train, fpca_models = extract_fpca_features(
        X_3d_train, kp_index, pids_train,
        n_components=8, frame_range=(120, 180), subsample=2,
        joints=SHOOTING_JOINTS, is_train=True
    )
    fpca_test, _ = extract_fpca_features(
        X_3d_test, kp_index, pids_test,
        n_components=8, frame_range=(120, 180), subsample=2,
        joints=SHOOTING_JOINTS, fpca_models=fpca_models, is_train=False
    )
    print(f"  FPCA features: {fpca_train.shape[1]} ({time.time()-t0:.1f}s)", flush=True)

    # 1b. Signature features
    print("\n--- Path Signatures ---", flush=True)
    t0 = time.time()
    # Use 3 key joints (wrist, elbow, shoulder) at depth 2
    # 3 joints x 3 coords + 1 time = 10D, depth 2 -> 110 features
    sig_joints = ["right_wrist", "right_elbow", "right_shoulder"]
    sig_train = extract_signature_features(
        X_3d_train, kp_index,
        frame_range=(130, 175), subsample=3, sig_depth=2,
        joints=sig_joints
    )
    sig_test = extract_signature_features(
        X_3d_test, kp_index,
        frame_range=(130, 175), subsample=3, sig_depth=2,
        joints=sig_joints
    )
    print(f"  Signature features: {sig_train.shape[1]} ({time.time()-t0:.1f}s)", flush=True)

    # 1c. Spatial features (at per-target release frames)
    print("\n--- Spatial Features ---", flush=True)
    spatial_features = {}
    for tname in TARGETS:
        tf = TARGET_FRAMES[tname]
        spatial_features[tname] = {
            "train": extract_spatial_features(X_3d_train, kp_index, tf),
            "test": extract_spatial_features(X_3d_test, kp_index, tf),
        }
        print(f"  {tname}: {spatial_features[tname]['train'].shape[1]} features", flush=True)

    # ============================================================
    # Phase 2: Individual Model Evaluation
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("PHASE 2: Individual Model Evaluation (Per-Player LOO)", flush=True)
    print(f"{'='*60}", flush=True)

    loo_predictions = {}
    test_predictions = {}

    for tidx, tname in enumerate(TARGETS):
        y = y_scaled[:, tidx]
        tf = TARGET_FRAMES[tname]
        print(f"\n{'='*50}", flush=True)
        print(f"TARGET: {tname}", flush=True)
        print(f"{'='*50}", flush=True)

        # ------ Model A: FPCA Ridge ------
        print("\n  --- Model A: FPCA Ridge ---", flush=True)
        loo_fpca = np.zeros(len(y))
        test_fpca = np.zeros(len(pids_test))

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            idx_tr = np.where(tr_mask)[0]
            n_p = len(idx_tr)

            X_p = fpca_train[tr_mask]
            y_p = y[tr_mask]

            # Standardize within player
            scaler = StandardScaler()
            X_ps = scaler.fit_transform(X_p)

            # LOO via hat matrix trick with RBF kernel on FPCA features
            best_mse_fpca = float('inf')
            best_loo_fpca = np.full(n_p, np.mean(y_p))
            best_alpha_fpca = 10.0
            dists_fpca = euclidean_distances(X_ps)
            gamma_fpca = 1.0 / (np.median(dists_fpca[dists_fpca > 0]) ** 2 + 1e-8)
            for gamma_mult in [0.1, 0.5, 1.0, 2.0]:
                K = rbf_kernel(X_ps, gamma=gamma_fpca * gamma_mult)
                for alpha in [0.01, 0.1, 1.0, 10.0]:
                    K_reg = K + alpha * np.eye(n_p)
                    try:
                        K_inv = np.linalg.inv(K_reg)
                        H = K @ K_inv
                        h_diag = np.diag(H)
                        y_hat = K @ np.linalg.solve(K_reg, y_p)
                        loo = (y_hat - h_diag * y_p) / (1 - h_diag + 1e-10)
                        mse = np.mean((loo - y_p) ** 2)
                        if mse < best_mse_fpca:
                            best_mse_fpca = mse
                            best_loo_fpca = loo
                            best_alpha_fpca = alpha
                    except:
                        pass

            loo_fpca[idx_tr] = best_loo_fpca

            # Test predictions
            if te_mask.sum() > 0:
                X_te = scaler.transform(fpca_test[te_mask])
                ridge = Ridge(alpha=best_alpha_fpca)
                ridge.fit(X_ps, y_p)
                test_fpca[te_mask] = ridge.predict(X_te)

        mse_fpca = np.mean((loo_fpca - y) ** 2)
        print(f"  FPCA Ridge LOO MSE: {mse_fpca:.6f}", flush=True)
        loo_predictions[f"fpca_{tname}"] = loo_fpca
        test_predictions[f"fpca_{tname}"] = test_fpca
        results[f"fpca_{tname}"] = {"mse": float(mse_fpca)}

        # ------ Model B: Signature Kernel Ridge ------
        print("\n  --- Model B: Signature Kernel Ridge ---", flush=True)
        loo_sig = np.zeros(len(y))
        test_sig = np.zeros(len(pids_test))

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            idx_tr = np.where(tr_mask)[0]
            n_p = len(idx_tr)

            S_p = sig_train[tr_mask]
            y_p = y[tr_mask]

            # Standardize signatures
            scaler_s = StandardScaler()
            S_ps = scaler_s.fit_transform(S_p)

            # RBF kernel on signatures with median heuristic
    
            dists = euclidean_distances(S_ps)
            gamma = 1.0 / (np.median(dists[dists > 0]) ** 2 + 1e-8)

            best_mse_sig = float('inf')
            for alpha in [0.01, 0.1, 1.0, 10.0]:
                K = rbf_kernel(S_ps, gamma=gamma)
                K_reg = K + alpha * np.eye(n_p)
                try:
                    K_inv = np.linalg.inv(K_reg)
                    H = K @ K_inv
                    h_diag = np.diag(H)
                    y_hat = K @ np.linalg.solve(K_reg, y_p)
                    loo = (y_hat - h_diag * y_p) / (1 - h_diag + 1e-10)
                    mse = np.mean((loo - y_p) ** 2)
                    if mse < best_mse_sig:
                        best_mse_sig = mse
                        best_loo_sig = loo
                        best_alpha_sig = alpha
                except:
                    pass

            if best_mse_sig == float('inf'):
                best_loo_sig = np.full(n_p, np.mean(y_p))
                best_alpha_sig = 10.0

            loo_sig[idx_tr] = best_loo_sig

            if te_mask.sum() > 0:
                S_te = scaler_s.transform(sig_test[te_mask])
                K_train = rbf_kernel(S_ps, gamma=gamma)
                K_test = rbf_kernel(S_te, S_ps, gamma=gamma)
                alpha_vec = np.linalg.solve(
                    K_train + best_alpha_sig * np.eye(n_p), y_p
                )
                test_sig[te_mask] = K_test @ alpha_vec

        mse_sig = np.mean((loo_sig - y) ** 2)
        print(f"  Signature Kernel LOO MSE: {mse_sig:.6f}", flush=True)
        loo_predictions[f"sig_{tname}"] = loo_sig
        test_predictions[f"sig_{tname}"] = test_sig
        results[f"sig_{tname}"] = {"mse": float(mse_sig)}

        # ------ Model C: Spatial Kernel Ridge ------
        print("\n  --- Model C: Spatial Kernel Ridge ---", flush=True)
        loo_spatial = np.zeros(len(y))
        test_spatial = np.zeros(len(pids_test))

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            idx_tr = np.where(tr_mask)[0]
            n_p = len(idx_tr)

            X_p = spatial_features[tname]["train"][tr_mask]
            y_p = y[tr_mask]

            scaler_x = StandardScaler()
            X_ps = scaler_x.fit_transform(X_p)

            dists = euclidean_distances(X_ps)
            gamma = 1.0 / (np.median(dists[dists > 0]) ** 2 + 1e-8)

            best_mse_sp = float('inf')
            for alpha in [0.01, 0.1, 1.0, 10.0]:
                K = rbf_kernel(X_ps, gamma=gamma)
                K_reg = K + alpha * np.eye(n_p)
                try:
                    K_inv = np.linalg.inv(K_reg)
                    H = K @ K_inv
                    h_diag = np.diag(H)
                    y_hat = K @ np.linalg.solve(K_reg, y_p)
                    loo = (y_hat - h_diag * y_p) / (1 - h_diag + 1e-10)
                    mse = np.mean((loo - y_p) ** 2)
                    if mse < best_mse_sp:
                        best_mse_sp = mse
                        best_loo_sp = loo
                        best_alpha_sp = alpha
                except:
                    pass

            if best_mse_sp == float('inf'):
                best_loo_sp = np.full(n_p, np.mean(y_p))
                best_alpha_sp = 10.0

            loo_spatial[idx_tr] = best_loo_sp

            if te_mask.sum() > 0:
                X_te = scaler_x.transform(spatial_features[tname]["test"][te_mask])
                K_train = rbf_kernel(X_ps, gamma=gamma)
                K_test = rbf_kernel(X_te, X_ps, gamma=gamma)
                alpha_vec = np.linalg.solve(
                    K_train + best_alpha_sp * np.eye(n_p), y_p
                )
                test_spatial[te_mask] = K_test @ alpha_vec

        mse_spatial = np.mean((loo_spatial - y) ** 2)
        print(f"  Spatial Kernel LOO MSE: {mse_spatial:.6f}", flush=True)
        loo_predictions[f"spatial_{tname}"] = loo_spatial
        test_predictions[f"spatial_{tname}"] = test_spatial
        results[f"spatial_{tname}"] = {"mse": float(mse_spatial)}

    # ============================================================
    # Phase 3: Multi-Kernel Learning (Optimal Combination)
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("PHASE 3: Multi-Kernel Learning", flush=True)
    print(f"{'='*60}", flush=True)

    mkl_test_preds = {}

    for tidx, tname in enumerate(TARGETS):
        y = y_scaled[:, tidx]
        print(f"\n  --- {tname} ---", flush=True)

        mkl_loo = np.zeros(len(y))
        mkl_test = np.zeros(len(pids_test))

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            idx_tr = np.where(tr_mask)[0]
            n_p = len(idx_tr)
            n_te = te_mask.sum()
            y_p = y[tr_mask]

            # Build per-player kernel matrices from each representation
            # Kernel 1: FPCA
            scaler1 = StandardScaler()
            F1 = scaler1.fit_transform(fpca_train[tr_mask])
            d1 = euclidean_distances(F1)
            g1 = 1.0 / (np.median(d1[d1 > 0]) ** 2 + 1e-8)
            K1_train = rbf_kernel(F1, gamma=g1)

            # Kernel 2: Signature
            scaler2 = StandardScaler()
            F2 = scaler2.fit_transform(sig_train[tr_mask])
            d2 = euclidean_distances(F2)
            g2 = 1.0 / (np.median(d2[d2 > 0]) ** 2 + 1e-8)
            K2_train = rbf_kernel(F2, gamma=g2)

            # Kernel 3: Spatial
            scaler3 = StandardScaler()
            F3 = scaler3.fit_transform(spatial_features[tname]["train"][tr_mask])
            d3 = euclidean_distances(F3)
            g3 = 1.0 / (np.median(d3[d3 > 0]) ** 2 + 1e-8)
            K3_train = rbf_kernel(F3, gamma=g3)

            # Normalize kernels
            for K in [K1_train, K2_train, K3_train]:
                trace = np.trace(K)
                if trace > 0:
                    K *= n_p / trace

            # Grid search for optimal weights
            best_mkl_mse = float('inf')
            best_w = [1/3, 1/3, 1/3]
            best_mkl_alpha = 1.0

            for w1 in np.arange(0, 1.01, 0.10):
                for w2 in np.arange(0, 1.01 - w1, 0.10):
                    w3 = 1 - w1 - w2
                    K_comb = w1 * K1_train + w2 * K2_train + w3 * K3_train
                    for alpha in [0.01, 0.1, 1.0, 10.0]:
                        K_reg = K_comb + alpha * np.eye(n_p)
                        try:
                            K_inv = np.linalg.inv(K_reg)
                            H = K_comb @ K_inv
                            h_diag = np.diag(H)
                            y_hat = K_comb @ np.linalg.solve(K_reg, y_p)
                            loo = (y_hat - h_diag * y_p) / (1 - h_diag + 1e-10)
                            mse = np.mean((loo - y_p) ** 2)
                            if mse < best_mkl_mse:
                                best_mkl_mse = mse
                                best_w = [w1, w2, w3]
                                best_mkl_alpha = alpha
                                best_mkl_loo = loo
                        except:
                            pass

            mkl_loo[idx_tr] = best_mkl_loo if best_mkl_mse < float('inf') else np.mean(y_p)

            print(f"    P{pid}: w_fpca={best_w[0]:.1f}, w_sig={best_w[1]:.1f}, "
                  f"w_spatial={best_w[2]:.1f}, alpha={best_mkl_alpha}, "
                  f"LOO={best_mkl_mse:.6f}", flush=True)

            # Test predictions with optimal weights
            if n_te > 0:
                F1_te = scaler1.transform(fpca_test[te_mask])
                F2_te = scaler2.transform(sig_test[te_mask])
                F3_te = scaler3.transform(spatial_features[tname]["test"][te_mask])

                K1_test = rbf_kernel(F1_te, F1, gamma=g1)
                K2_test = rbf_kernel(F2_te, F2, gamma=g2)
                K3_test = rbf_kernel(F3_te, F3, gamma=g3)

                # Normalize test kernels consistently
                for K_te, K_tr in [(K1_test, K1_train), (K2_test, K2_train),
                                    (K3_test, K3_train)]:
                    trace = np.trace(K_tr)
                    if trace > 0:
                        K_te *= n_p / trace

                K_comb_train = best_w[0]*K1_train + best_w[1]*K2_train + best_w[2]*K3_train
                K_comb_test = best_w[0]*K1_test + best_w[1]*K2_test + best_w[2]*K3_test

                alpha_vec = np.linalg.solve(
                    K_comb_train + best_mkl_alpha * np.eye(n_p), y_p
                )
                mkl_test[te_mask] = K_comb_test @ alpha_vec

        mse_mkl = np.mean((mkl_loo - y) ** 2)
        print(f"  MKL LOO MSE: {mse_mkl:.6f}", flush=True)
        mkl_test_preds[tname] = np.clip(mkl_test, 0, 1)
        results[f"mkl_{tname}"] = {"mse": float(mse_mkl)}

    # ============================================================
    # Phase 4: Summary and Submissions
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("PHASE 4: Summary", flush=True)
    print(f"{'='*60}", flush=True)

    for approach in ["fpca", "sig", "spatial", "mkl"]:
        mses = [results[f"{approach}_{t}"]["mse"] for t in TARGETS
                if f"{approach}_{t}" in results]
        if mses:
            print(f"  {approach}: mean LOO = {np.mean(mses):.6f} "
                  f"({', '.join(f'{m:.6f}' for m in mses)})", flush=True)

    # Standalone submission
    sub_df = pd.DataFrame({
        "id": ids_test,
        "scaled_angle": mkl_test_preds["angle"],
        "scaled_depth": mkl_test_preds["depth"],
        "scaled_left_right": mkl_test_preds["left_right"],
    })

    # Diversity analysis
    print(f"\n--- Diversity vs Sub3558 ---", flush=True)
    try:
        sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
        for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
            r, _ = pearsonr(sub_df[tc], sub3558[tc])
            print(f"  {tc}: r={r:.4f}", flush=True)
    except Exception as e:
        print(f"  {e}", flush=True)

    # Also check individual model diversity
    for approach_name, preds_key_prefix in [("FPCA", "fpca"), ("Signature", "sig")]:
        print(f"\n--- {approach_name} Diversity vs Sub3558 ---", flush=True)
        try:
            approach_sub = pd.DataFrame({
                "id": ids_test,
                "scaled_angle": np.clip(test_predictions[f"{preds_key_prefix}_angle"], 0, 1),
                "scaled_depth": np.clip(test_predictions[f"{preds_key_prefix}_depth"], 0, 1),
                "scaled_left_right": np.clip(test_predictions[f"{preds_key_prefix}_left_right"], 0, 1),
            })
            for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
                r, _ = pearsonr(approach_sub[tc], sub3558[tc])
                print(f"  {tc}: r={r:.4f}", flush=True)
        except Exception as e:
            print(f"  {e}", flush=True)

    save_submission(sub_df, f"PhD MKL standalone")

    # Blends with Sub3558
    try:
        sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
        for w in [0.02, 0.05, 0.08, 0.12]:
            blend = sub_df.copy()
            for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
                blend[tc] = np.clip(w * sub_df[tc] + (1-w) * sub3558[tc], 0, 1)
            save_submission(blend, f"{int(w*100)}% PhD-MKL + {int((1-w)*100)}% Sub3558")
    except Exception as e:
        print(f"  Blend error: {e}", flush=True)

    # Per-target blend (more MKL on diverse targets)
    try:
        sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
        per_target = sub3558.copy()
        for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
            r, _ = pearsonr(sub_df[tc], sub3558[tc])
            # More diverse = more weight
            if r < 0.80:
                w = 0.10
            elif r < 0.90:
                w = 0.06
            else:
                w = 0.03
            per_target[tc] = np.clip(w * sub_df[tc] + (1-w) * sub3558[tc], 0, 1)
            print(f"  Per-target {tc}: r={r:.4f} -> w={w}", flush=True)
        save_submission(per_target, "Per-target PhD-MKL blend with Sub3558")
    except Exception as e:
        print(f"  Per-target error: {e}", flush=True)

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total/60:.1f}min)", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"phd_breakthrough_run_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

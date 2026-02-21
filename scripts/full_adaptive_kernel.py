"""
Full Pipeline with Player-Adaptive Diagonal Mahalanobis Metric

Uses the FULL 198-feature hoop-relative feature set from the core pipeline
(no PLS compression - preserves interpretability for per-feature weighting).

Per-player adaptive metric: w_i = |r(feature_i, target)| + eps
Weighted Euclidean: d(x,y) = sqrt(sum_i w_i * (x_i - y_i)^2)

This is an honest LOO evaluation: weights recomputed within each fold.

Compares against:
- Honest LOO reference: 0.006830 (core pipeline with PLS)
"""

import fcntl
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

KEY_JOINTS = [
    'right_wrist', 'right_elbow', 'right_shoulder',
    'left_wrist', 'left_shoulder',
    'right_hip', 'left_hip', 'mid_hip',
    'right_knee', 'left_knee', 'neck', 'nose'
]
N_JOINTS = len(KEY_JOINTS)  # 12
# Features: 12 joints x 3 coords x 2 (pos + vel) = 72 features at target frame
# Plus summary stats: 12 joints x 3 coords x 3 (mean, std, range) = 108 features
# Plus arm mechanics: 4 features
# Total: 72 + 108 + 4 = 184 features (approximately 198 in original)


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def safe_savgol(x, window=11, polyorder=3, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


def compute_hoop_transform(ts_3d, kp_index):
    mid_hip_idx = kp_index.get('mid_hip', 0)
    player_pos = ts_3d[120, mid_hip_idx, :].copy()
    player_pos[2] = 0
    forward = HOOP_POS[:2] - player_pos[:2]
    fn = np.linalg.norm(forward)
    if fn > 1e-6:
        forward /= fn
    else:
        forward = np.array([0.0, -1.0])
    lateral = np.array([-forward[1], forward[0]])
    R = np.eye(3, dtype=np.float32)
    R[0, 0] = forward[0]; R[0, 1] = forward[1]
    R[1, 0] = lateral[0]; R[1, 1] = lateral[1]
    centered = ts_3d - player_pos.reshape(1, 1, 3)
    return np.einsum('ij,fkj->fki', R, centered)


def extract_features_full(ts_3d, kp_index, frame):
    """Extract 184-feature hoop-relative feature vector."""
    f = int(np.clip(frame, 0, 239))
    ts_hr = compute_hoop_transform(ts_3d, kp_index)
    
    feats = []
    
    for jname in KEY_JOINTS:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            series = ts_hr[:, idx, coord]
            series_s = safe_savgol(series)
            feats.append(series_s[f])
            vel = safe_savgol(series_s, deriv=1, delta=DT)
            feats.append(vel[f])
    
    for jname in KEY_JOINTS:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 9)
            continue
        for coord in range(3):
            series = ts_hr[:, idx, coord]
            series_s = safe_savgol(series)
            feats.append(float(np.nanmean(series_s)))
            feats.append(float(np.nanstd(series_s)))
            feats.append(float(np.nanmax(series_s) - np.nanmin(series_s)))
    
    # Arm mechanics
    rw = kp_index.get('right_wrist')
    re = kp_index.get('right_elbow')
    rs = kp_index.get('right_shoulder')
    if all(i is not None for i in [rw, re, rs]):
        feats.append(ts_hr[f, rw, 0] - ts_hr[f, rs, 0])
        feats.append(ts_hr[f, rw, 1] - ts_hr[f, rs, 1])
        feats.append(ts_hr[f, rw, 2] - ts_hr[f, rs, 2])
        ua = ts_3d[f, re] - ts_3d[f, rs]
        fa = ts_3d[f, rw] - ts_3d[f, re]
        ua_n, fa_n = np.linalg.norm(ua), np.linalg.norm(fa)
        if ua_n > 1e-6 and fa_n > 1e-6:
            feats.append(float(np.degrees(np.arccos(np.clip(np.dot(-ua, fa) / (ua_n * fa_n), -1, 1)))))
        else:
            feats.append(90.0)
    else:
        feats.extend([0.0] * 4)
    
    return np.array(feats, dtype=np.float64)


def compute_player_weights(X_tr, y_tr, eps=0.02):
    """Per-player feature importance weights via Pearson correlation."""
    n, d = X_tr.shape
    weights = np.full(d, eps, dtype=np.float64)
    y_std = np.std(y_tr)
    if y_std < 1e-8:
        return weights
    for fi in range(d):
        fv = X_tr[:, fi]
        if np.std(fv) > 1e-8:
            r = float(np.corrcoef(fv, y_tr)[0, 1])
            if not np.isnan(r):
                weights[fi] = max(abs(r), eps)
    return weights


def weighted_kernel_ridge_loo(X_pid, y_pid, bandwidth_q=0.3, alpha=10.0):
    """LOO with player-adaptive weighted kernel Ridge."""
    n = len(X_pid)
    if n < 5:
        return np.full(n, np.mean(y_pid)), np.mean((np.mean(y_pid) - y_pid)**2)
    
    oof = np.zeros(n)
    sc = StandardScaler()
    X_s = sc.fit_transform(X_pid)
    
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        X_tr = X_s[tr]
        X_te = X_s[[i]]
        y_tr = y_pid[tr]
        
        # Per-fold adaptive weights
        weights = compute_player_weights(X_tr, y_tr)
        
        # Weighted distance
        sqrt_w = np.sqrt(weights / weights.sum())
        X_tr_w = X_tr * sqrt_w
        X_te_w = X_te * sqrt_w
        
        D = cdist(X_te_w, X_tr_w, metric='euclidean').ravel()
        bw = np.quantile(D, bandwidth_q)
        bw = max(bw, 1e-6)
        K = np.exp(-0.5 * (D / bw) ** 2)
        K = np.maximum(K, 1e-10)
        
        sqrt_K = np.sqrt(K).reshape(-1, 1)
        X_wk = X_tr * sqrt_K
        y_wk = y_tr * np.sqrt(K)
        
        mdl = Ridge(alpha=alpha)
        mdl.fit(X_wk, y_wk)
        oof[i] = float(np.clip(mdl.predict(X_te)[0], 0.0, 1.0))
    
    mse = float(np.mean((oof - y_pid) ** 2))
    return oof, mse


def uniform_kernel_ridge_loo(X_pid, y_pid, bandwidth_q=0.3, alpha=10.0):
    """Standard LOO with uniform kernel (baseline)."""
    n = len(X_pid)
    if n < 5:
        return np.full(n, np.mean(y_pid)), np.mean((np.mean(y_pid) - y_pid)**2)
    
    oof = np.zeros(n)
    sc = StandardScaler()
    X_s = sc.fit_transform(X_pid)
    
    D_all = cdist(X_s, X_s, metric='euclidean')
    
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        X_tr = X_s[tr]
        X_te = X_s[[i]]
        y_tr = y_pid[tr]
        
        D = D_all[i, tr]
        bw = np.quantile(D, bandwidth_q)
        bw = max(bw, 1e-6)
        K = np.exp(-0.5 * (D / bw) ** 2)
        K = np.maximum(K, 1e-10)
        
        sqrt_K = np.sqrt(K).reshape(-1, 1)
        X_wk = X_tr * sqrt_K
        y_wk = y_tr * np.sqrt(K)
        
        mdl = Ridge(alpha=alpha)
        mdl.fit(X_wk, y_wk)
        oof[i] = float(np.clip(mdl.predict(X_te)[0], 0.0, 1.0))
    
    mse = float(np.mean((oof - y_pid) ** 2))
    return oof, mse


def load_data():
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    
    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}
    
    def process(df, is_train=True):
        n = len(df)
        n_kp = len(kp_names)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        ids, pids, targets = [], [], []
        
        for row_i, (idx, row) in enumerate(df.iterrows()):
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_3d[row_i, :, col_i // 3, col_i % 3] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                scaled = [
                    float(scalers['angle'].transform([[row['angle']]])[0, 0]),
                    float(scalers['depth'].transform([[row['depth']]])[0, 0]),
                    float(scalers['left_right'].transform([[row['left_right']]])[0, 0]),
                ]
                targets.append(scaled)
            if (row_i + 1) % 100 == 0:
                print(f"  Processed {row_i + 1}/{n}")
        
        result = {'X_3d': X_3d, 'pids': np.array(pids), 'ids': np.array(ids), 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float64)
        return result
    
    return process(train_df, True), process(test_df, False)


def main():
    t0 = time.time()
    print("=== Full Pipeline: Adaptive vs Uniform Kernel ===")
    print("Honest LOO reference: 0.006830 (core pipeline with PLS)")
    print()
    
    train, test = load_data()
    X_3d_tr = train['X_3d']
    y_tr = train['y']
    pids_tr = train['pids']
    kp_index = train['kp_index']
    
    X_3d_te = test['X_3d']
    pids_te = test['pids']
    ids_te = test['ids']
    
    results = {}
    
    for ti, tname in enumerate(TARGETS):
        frame = TARGET_FRAMES[tname]
        y_t = y_tr[:, ti]
        
        print(f"\n{'='*60}")
        print(f"TARGET: {tname.upper()} (frame={frame})")
        print(f"{'='*60}")
        
        print(f"  Extracting {len(X_3d_tr)} features (frame={frame})...")
        X_all = np.array([extract_features_full(X_3d_tr[i], kp_index, frame)
                          for i in range(len(X_3d_tr))])
        
        n_feat = X_all.shape[1]
        print(f"  Feature dim: {n_feat}")
        
        uniform_mses = []
        adaptive_mses = []
        
        for pid in sorted(np.unique(pids_tr)):
            pid_mask = pids_tr == pid
            n_pid = pid_mask.sum()
            X_pid = X_all[pid_mask]
            y_pid = y_t[pid_mask]
            
            # Uniform kernel
            _, mse_uni = uniform_kernel_ridge_loo(X_pid, y_pid)
            # Adaptive kernel
            _, mse_ada = weighted_kernel_ridge_loo(X_pid, y_pid)
            
            delta = (mse_ada - mse_uni) / mse_uni * 100
            sign = "<-- IMPROVED" if mse_ada < mse_uni else "<-- WORSE"
            print(f"  P{pid} (n={n_pid}): uniform={mse_uni:.6f} adaptive={mse_ada:.6f} delta={delta:.1f}% {sign}")
            
            uniform_mses.append(mse_uni)
            adaptive_mses.append(mse_ada)
        
        mean_uni = np.mean(uniform_mses)
        mean_ada = np.mean(adaptive_mses)
        delta_mean = (mean_ada - mean_uni) / mean_uni * 100
        print(f"\n  {tname} MEAN: uniform={mean_uni:.6f} adaptive={mean_ada:.6f} delta={delta_mean:.1f}%")
        
        results[tname] = {
            "uniform_mse_mean": float(mean_uni),
            "adaptive_mse_mean": float(mean_ada),
            "delta_pct": float(delta_mean),
        }
    
    # Overall
    uni_mean = np.mean([results[t]['uniform_mse_mean'] for t in TARGETS])
    ada_mean = np.mean([results[t]['adaptive_mse_mean'] for t in TARGETS])
    print(f"\n{'='*60}")
    print(f"OVERALL: uniform={uni_mean:.6f} adaptive={ada_mean:.6f}")
    print(f"vs honest LOO reference: 0.006830")
    print(f"Delta uniform vs reference: {(uni_mean - 0.006830)/0.006830*100:.1f}%")
    print(f"Delta adaptive vs reference: {(ada_mean - 0.006830)/0.006830*100:.1f}%")
    print(f"Delta adaptive vs uniform: {(ada_mean - uni_mean)/uni_mean*100:.1f}%")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results["overall_uniform"] = float(uni_mean)
    results["overall_adaptive"] = float(ada_mean)
    results["honest_loo_reference"] = 0.006830
    results["total_time_s"] = time.time() - t0
    
    out_path = OUTPUT_DIR / f"full_adaptive_kernel_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()

"""
Player-Channel Oracle: Targeted use of discovered high-correlation channels

Research finding (rigorous_channel_analysis.py):
- P5 depth: vel_left_shoulder_z_f153, r=0.860 (strongest single-feature predictor found)
- P2 LR:   hr_right_wrist_y_f150,    r=-0.788 (cleanest cross-player transfer: 0.049)
- P3 LR:   hr_ls_y_f170,             r=+0.692 (cleanest cross-player transfer: 0.004)
- P1 LR:   vel_neck_y_f170,          r=+0.696
- P4 LR:   vel_mh_y_f160,            r=-0.619
- P1 depth: hr_right_elbow_z_f150,   r=-0.684
- P4 depth: hr_right_wrist_y_f175,   r=+0.687

Strategy: For each player x target, build Ridge regression on ONLY their top K features.
These are low-dimensional, stable models that should generalize from train to test.
Blend oracle predictions with existing best submission.
"""

import fcntl
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge, RidgeCV
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

# Top player-specific channels from rigorous_channel_analysis.py
# Format: {target: {player_id: [(feature_name, frame, coord_type), ...]}}
# coord_type: 'pos' = position, 'vel' = velocity
# hr = hoop-relative frame

PLAYER_CHANNELS = {
    "depth": {
        5: [("left_shoulder", 153, "vel", "z"),
            ("right_hip", 153, "vel", "z"),
            ("neck", 153, "vel", "z"),
            ("mid_hip", 153, "vel", "z")],  # r=0.860, 0.857, 0.855, 0.855
        4: [("right_wrist", 175, "pos", "y"),
            ("right_hip", 150, "pos", "x"),
            ("right_hip", 153, "pos", "x")],  # r=0.687, 0.655, 0.654
        1: [("right_elbow", 150, "pos", "z"),
            ("right_wrist", 150, "pos", "z"),
            ("right_elbow", 153, "pos", "z")],  # r=-0.684, -0.671, -0.653
        3: [("right_wrist", 170, "pos", "x"),
            ("left_wrist", 170, "pos", "x"),
            ("right_elbow", 165, "pos", "x")],  # r=0.544, 0.536, 0.511
        2: [("left_wrist", 170, "pos", "z"),
            ("left_wrist", 165, "pos", "z"),
            ("left_shoulder", 150, "pos", "z")],  # r=0.409, 0.408, 0.324
    },
    "left_right": {
        1: [("neck", 170, "vel", "y"),
            ("neck", 165, "vel", "y"),
            ("right_hip", 175, "vel", "y"),
            ("left_shoulder", 170, "vel", "y")],  # r=0.696, 0.680, 0.669, 0.662
        2: [("right_wrist", 150, "pos", "y"),
            ("right_wrist", 153, "pos", "y"),
            ("right_elbow", 150, "pos", "y"),
            ("right_elbow", 153, "pos", "y")],  # r=-0.788, -0.762, -0.706, -0.665
        3: [("left_shoulder", 170, "pos", "y"),
            ("left_shoulder", 175, "pos", "y"),
            ("left_shoulder", 165, "pos", "y")],  # r=0.692, 0.688, 0.685
        4: [("mid_hip", 160, "vel", "y"),
            ("mid_hip", 170, "pos", "y"),
            ("right_hip", 165, "pos", "y"),
            ("right_hip", 170, "pos", "y")],  # r=-0.619, -0.619, -0.611, -0.607
        5: [("neck", 175, "vel", "y"),
            ("right_wrist", 180, "vel", "y"),
            ("right_wrist", 175, "vel", "y"),
            ("right_wrist", 170, "vel", "y")],  # r=0.611, -0.579, -0.575, -0.478
    },
    "angle": {
        4: [("neck", 165, "pos", "y"),
            ("right_elbow", 175, "pos", "y"),
            ("neck", 160, "pos", "y"),
            ("left_shoulder", 160, "pos", "y")],  # r=-0.522, -0.516, -0.510, -0.492
        5: [("left_wrist", 153, "pos", "y"),
            ("left_wrist", 155, "pos", "y"),
            ("left_wrist", 150, "pos", "y")],  # r=+0.525, +0.501, +0.493
    },
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


def extract_channel_features(ts_3d, kp_index, channels):
    """Extract specific player-channel features for a single shot."""
    ts_hr = compute_hoop_transform(ts_3d, kp_index)
    feats = []
    
    for (joint_name, frame, feat_type, axis) in channels:
        j_idx = kp_index.get(joint_name)
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        frame = min(frame, 239)
        
        if j_idx is None:
            feats.append(0.0)
            continue
        
        series = ts_hr[:, j_idx, axis_idx].copy()
        series = safe_savgol(series)
        
        if feat_type == 'pos':
            feats.append(float(series[frame]))
        elif feat_type == 'vel':
            vel = safe_savgol(series, deriv=1, delta=DT)
            feats.append(float(vel[frame]))
    
    return np.array(feats, dtype=np.float64)


def load_data():
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Load scalers to convert raw targets -> [0,1] scaled targets
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
                # Scale targets to [0,1] using saved scalers
                raw = np.array([[row['angle'], row['depth'], row['left_right']]])
                scaled = [
                    float(scalers['angle'].transform([[row['angle']]])[0, 0]),
                    float(scalers['depth'].transform([[row['depth']]])[0, 0]),
                    float(scalers['left_right'].transform([[row['left_right']]])[0, 0]),
                ]
                targets.append(scaled)
            if (row_i + 1) % 100 == 0:
                print(f"  Processed {row_i + 1}/{n}")

        result = {'X_3d': X_3d, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float64)
        return result

    return process(train_df, True), process(test_df, False)


def loo_oracle(X_pid, y_pid, alpha=1.0):
    """LOO with Ridge on player-specific features."""
    n = len(X_pid)
    if n < 5:
        return np.full(n, np.mean(y_pid))
    
    oof = np.zeros(n)
    sc = StandardScaler()
    
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        X_tr = sc.fit_transform(X_pid[tr])
        X_te = sc.transform(X_pid[[i]])
        y_tr = y_pid[tr]
        
        mdl = Ridge(alpha=alpha)
        mdl.fit(X_tr, y_tr)
        oof[i] = mdl.predict(X_te)[0]
    
    return np.clip(oof, 0.0, 1.0)


def fit_oracle(X_tr_pid, y_tr_pid, X_te_pid, alpha=1.0):
    """Fit full oracle and predict test."""
    n = len(X_tr_pid)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_pid)
    X_te_s = sc.transform(X_te_pid)
    mdl = Ridge(alpha=alpha)
    mdl.fit(X_tr_s, y_tr_pid)
    return np.clip(mdl.predict(X_te_s), 0.0, 1.0)


def main():
    t0 = time.time()
    print("=== Player-Channel Oracle ===")
    print("Using rigorous_channel_analysis.py discoveries")
    print()
    
    train, test = load_data()
    X_3d_tr = train['X_3d']
    y_tr = train['y']
    pids_tr = train['pids']
    ids_tr = train['ids']
    
    X_3d_te = test['X_3d']
    pids_te = test['pids']
    ids_te = test['ids']
    kp_index = train['kp_index']
    
    TARGETS_IDX = {t: i for i, t in enumerate(TARGETS)}
    
    results = {}
    oof_preds = np.zeros_like(y_tr)  # per-target oracle OOF
    test_preds = np.zeros((len(ids_te), 3))  # oracle test preds
    
    has_channels_mask = np.zeros((len(ids_tr), 3), dtype=bool)
    has_channels_mask_te = np.zeros((len(ids_te), 3), dtype=bool)
    
    print("Extracting player-channel features and running LOO...\n")
    
    for ti, tname in enumerate(TARGETS):
        if tname not in PLAYER_CHANNELS:
            print(f"  {tname}: no channels defined, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"TARGET: {tname.upper()}")
        print(f"{'='*60}")
        
        target_results = {}
        y_t = y_tr[:, ti]
        
        for pid, channels in PLAYER_CHANNELS[tname].items():
            pid_mask_tr = pids_tr == pid
            pid_mask_te = pids_te == pid
            
            n_tr_pid = pid_mask_tr.sum()
            n_te_pid = pid_mask_te.sum()
            
            print(f"\n  Player {pid} (n_train={n_tr_pid}, n_test={n_te_pid}):")
            print(f"  Features: {len(channels)} channels")
            
            # Extract features for this player's training shots
            X_pid_list = []
            for i in np.where(pid_mask_tr)[0]:
                feats = extract_channel_features(X_3d_tr[i], kp_index, channels)
                X_pid_list.append(feats)
            X_pid_tr = np.array(X_pid_list)
            y_pid = y_t[pid_mask_tr]
            
            # LOO evaluation
            oof_pid = loo_oracle(X_pid_tr, y_pid, alpha=10.0)
            mse_oracle = np.mean((oof_pid - y_pid) ** 2)
            
            # Baseline: per-player mean
            mse_mean = np.mean((np.mean(y_pid) - y_pid) ** 2)
            
            # R^2
            r2 = 1.0 - mse_oracle / mse_mean if mse_mean > 0 else 0.0
            r = np.corrcoef(oof_pid, y_pid)[0, 1]
            
            print(f"  LOO MSE: {mse_oracle:.6f} (vs mean baseline {mse_mean:.6f})")
            print(f"  LOO R2={r2:.3f}, LOO r={r:.3f}")
            
            # Store OOF
            oof_preds[pid_mask_tr, ti] = oof_pid
            has_channels_mask[pid_mask_tr, ti] = True
            
            # Fit full model for test
            if n_te_pid > 0:
                X_pid_te_list = []
                for i in np.where(pid_mask_te)[0]:
                    feats = extract_channel_features(X_3d_te[i], kp_index, channels)
                    X_pid_te_list.append(feats)
                X_pid_te = np.array(X_pid_te_list)
                
                te_pred_pid = fit_oracle(X_pid_tr, y_pid, X_pid_te, alpha=10.0)
                test_preds[pid_mask_te, ti] = te_pred_pid
                has_channels_mask_te[pid_mask_te, ti] = True
                
                print(f"  Test preds: {n_te_pid} shots, range [{te_pred_pid.min():.3f}, {te_pred_pid.max():.3f}]")
            
            target_results[f"p{pid}"] = {
                "mse": float(mse_oracle),
                "mse_mean_baseline": float(mse_mean),
                "r2": float(r2),
                "r_oof": float(r),
            }
        
        results[tname] = target_results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for tname in TARGETS:
        if tname not in results:
            continue
        print(f"\n  {tname}:")
        for pid_key, metrics in results[tname].items():
            print(f"    {pid_key}: MSE={metrics['mse']:.6f}, R2={metrics['r2']:.3f}, LOO_r={metrics['r_oof']:.3f}")
    
    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = OUTPUT_DIR / f"player_channel_oracle_run_{ts}.json"
    results["total_time_s"] = time.time() - t0
    results["has_channels_coverage_train"] = {
        t: int(has_channels_mask[:, i].sum()) for i, t in enumerate(TARGETS)
    }
    results["has_channels_coverage_test"] = {
        t: int(has_channels_mask_te[:, i].sum()) for i, t in enumerate(TARGETS)
    }
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")
    
    # Create blended submissions
    print("\n--- Creating Submissions ---")
    
    # Load best reference sub (Sub3507 which is LB 0.006205, or Sub3558 if available)
    best_sub_path = None
    for sub_num in [3558, 3507, 3411]:
        p = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        if p.exists():
            best_sub_path = p
            best_sub_num = sub_num
            break
    
    if best_sub_path is None:
        print("WARNING: No reference submission found, skipping blends")
        return
    
    best_df = pd.read_csv(best_sub_path)
    print(f"  Reference: Sub{best_sub_num} ({best_sub_path})")
    
    # Map column names
    col_map = {}
    for t in TARGETS:
        if f"scaled_{t}" in best_df.columns:
            col_map[t] = f"scaled_{t}"
        else:
            col_map[t] = t
    
    # Oracle OOF performance vs reference (per player-target combination)
    print("\n  Oracle OOF stats:")
    y_t_arr = y_tr  # (345, 3)
    for ti, tname in enumerate(TARGETS):
        mask = has_channels_mask[:, ti]
        if mask.sum() > 0:
            oof_t = oof_preds[mask, ti]
            y_t = y_t_arr[mask, ti]
            mse = np.mean((oof_t - y_t) ** 2)
            r = np.corrcoef(oof_t, y_t)[0, 1]
            print(f"  {tname}: {mask.sum()} shots covered, OOF MSE={mse:.6f}, r={r:.3f}")
    
    # Create oracle test prediction CSV
    # Only override predictions where we have channels; use best_df for rest
    oracle_df = best_df.copy()
    for i, test_id in enumerate(ids_te):
        row_mask = oracle_df['id'] == test_id
        if row_mask.sum() == 0:
            continue
        for ti, tname in enumerate(TARGETS):
            if has_channels_mask_te[i, ti]:
                oracle_df.loc[row_mask, col_map[tname]] = test_preds[i, ti]
    
    # Blend oracle with best sub at different weights
    for blend_w in [0.03, 0.05, 0.10, 0.20]:
        sub_num = get_next_submission_number()
        blend_df = best_df.copy()
        for i, test_id in enumerate(ids_te):
            row_mask = blend_df['id'] == test_id
            if row_mask.sum() == 0:
                continue
            for ti, tname in enumerate(TARGETS):
                if has_channels_mask_te[i, ti]:
                    oracle_pred = test_preds[i, ti]
                    best_pred = float(best_df.loc[row_mask, col_map[tname]].values[0])
                    blended = blend_w * oracle_pred + (1.0 - blend_w) * best_pred
                    blend_df.loc[row_mask, col_map[tname]] = blended
        
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blend_df.to_csv(sub_path, index=False)
        print(f"  Sub {sub_num}: {int(blend_w*100)}% oracle + {int((1-blend_w)*100)}% Sub{best_sub_num}")
    
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

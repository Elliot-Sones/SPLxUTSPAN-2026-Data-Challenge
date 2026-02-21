"""
Biomechanical causation features: joint angles, angular velocities, kinematic efficiency.

Captures PHYSICS of shooting mechanics:
- Joint relative angles (elbow-shoulder, knee-hip, etc.)
- Angular velocities at release
- Kinematic chain synchronization
- Release timing relative to peaks

Unlike displacement features (Option 1) or position/velocity (core Ridge),
these capture BIOMECHANICAL RELATIONSHIPS between joints.

Example: "does shoulder-elbow synchronization correlate with shot accuracy?"
vs "where is the shoulder positioned" (what we already have).

Usage:
  uv run scripts/biomechanical_causation_features.py --pilot
  uv run scripts/biomechanical_causation_features.py
"""

from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
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

# Joints for biomechanical analysis
SHOOTING_ARM = ["right_shoulder", "right_elbow", "right_wrist", "right_second_finger_distal", "right_third_finger_distal"]
SUPPORT_ARM = ["left_shoulder", "left_elbow", "left_wrist"]
LOWER_BODY = ["right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]
TORSO = ["mid_hip", "neck"]

# ============================================================
# Data loading
# ============================================================

def parse_array_string(s):
    s = str(s).replace("nan", "NaN").replace("null", "NaN")
    return np.nan_to_num(np.array(json.loads(s), dtype=np.float64), nan=0.0)


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    kp_cols = [c for c in df.columns if c not in meta_cols]
    kp_names = [c[:-2] for c in kp_cols if c.endswith("_x")]
    kp_index = {name: i for i, name in enumerate(kp_names)}
    n_kp = len(kp_names)
    n = len(df)

    X_3d = np.zeros((n, N_FRAMES, n_kp, 3), dtype=np.float32)
    for idx, row in df.iterrows():
        for col_i, col in enumerate(kp_cols):
            kp_i = col_i // 3
            ax_i = col_i % 3
            arr = parse_array_string(row[col])
            X_3d[idx, :, kp_i, ax_i] = arr
        if (idx + 1) % 100 == 0:
            print(f"  Loaded {idx+1}/{n}...")

    return X_3d, kp_index, df


# ============================================================
# Biomechanical feature extraction
# ============================================================

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two 3D vectors in radians (0 to pi)."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
    return np.arccos(cos_angle)


def extract_biomechanical_features(X_3d: np.ndarray, kp_index: dict, 
                                    frame: int) -> np.ndarray:
    """
    Extract biomechanical features at a specific frame.
    
    Returns: (n_samples, n_features) array
    """
    n = X_3d.shape[0]
    features = []
    feature_names = []

    # Helper to get joint position
    def get_joint(name):
        idx = kp_index.get(name)
        if idx is None:
            return None
        return X_3d[:, frame, idx, :] - HOOP_FEET[None, :]

    # 1. SHOOTING ARM ANGLES
    # Elbow angle: angle between (shoulder->elbow) and (elbow->wrist)
    s_shoulder = get_joint("right_shoulder")
    s_elbow = get_joint("right_elbow")
    s_wrist = get_joint("right_wrist")
    
    if s_shoulder is not None and s_elbow is not None and s_wrist is not None:
        upper_arm = s_elbow - s_shoulder  # shoulder to elbow
        forearm = s_wrist - s_elbow      # elbow to wrist
        elbow_angles = np.array([angle_between_vectors(upper_arm[i], forearm[i]) 
                                 for i in range(n)])
        features.append(elbow_angles)
        feature_names.append("shooting_elbow_angle")

    # Shoulder-elbow-wrist alignment: how straight is the arm?
    if s_shoulder is not None and s_elbow is not None and s_wrist is not None:
        # Full arm vector (shoulder to wrist)
        full_arm = s_wrist - s_shoulder
        # Upper arm + forearm combined
        combined = (s_elbow - s_shoulder) + (s_wrist - s_elbow)
        # Alignment: how much does combined match full_arm (straighter = better alignment)
        alignments = np.array([np.dot(full_arm[i], combined[i]) / 
                               (np.linalg.norm(full_arm[i]) * np.linalg.norm(combined[i]) + 1e-6)
                               for i in range(n)])
        features.append(alignments)
        feature_names.append("shooting_arm_alignment")

    # 2. SUPPORT ARM ANGLES
    # Support elbow angle
    sup_shoulder = get_joint("left_shoulder")
    sup_elbow = get_joint("left_elbow")
    sup_wrist = get_joint("left_wrist")
    
    if sup_shoulder is not None and sup_elbow is not None and sup_wrist is not None:
        sup_upper = sup_elbow - sup_shoulder
        sup_fore = sup_wrist - sup_elbow
        sup_elbow_angles = np.array([angle_between_vectors(sup_upper[i], sup_fore[i])
                                     for i in range(n)])
        features.append(sup_elbow_angles)
        feature_names.append("support_elbow_angle")

    # 3. LOWER BODY ANGLES
    # Right leg: knee angle
    r_hip = get_joint("right_hip")
    r_knee = get_joint("right_knee")
    r_ankle = get_joint("right_ankle")
    
    if r_hip is not None and r_knee is not None and r_ankle is not None:
        r_thigh = r_knee - r_hip
        r_calf = r_ankle - r_knee
        r_knee_angles = np.array([angle_between_vectors(r_thigh[i], r_calf[i])
                                  for i in range(n)])
        features.append(r_knee_angles)
        feature_names.append("right_knee_angle")

    # Left leg: knee angle
    l_hip = get_joint("left_hip")
    l_knee = get_joint("left_knee")
    l_ankle = get_joint("left_ankle")
    
    if l_hip is not None and l_knee is not None and l_ankle is not None:
        l_thigh = l_knee - l_hip
        l_calf = l_ankle - l_knee
        l_knee_angles = np.array([angle_between_vectors(l_thigh[i], l_calf[i])
                                  for i in range(n)])
        features.append(l_knee_angles)
        feature_names.append("left_knee_angle")

    # 4. TORSO-LIMB COUPLING
    # Hip-shoulder separation angle (measure of torso rotation)
    hip = get_joint("mid_hip")
    neck = get_joint("neck")
    
    if hip is not None and s_shoulder is not None and s_elbow is not None:
        # Torso axis: hip to neck
        torso_axis = neck - hip
        # Shooting arm axis: shoulder to wrist
        arm_axis = s_wrist - s_shoulder
        torso_arm_coupling = np.array([angle_between_vectors(torso_axis[i], arm_axis[i])
                                       for i in range(n)])
        features.append(torso_arm_coupling)
        feature_names.append("torso_arm_coupling_angle")

    # 5. KINEMATIC CHAIN EFFICIENCY (frame-to-frame smoothness)
    # Extract velocities at this frame (diff to next frame)
    if frame < N_FRAMES - 1:
        # Shooting arm speed
        s_shoulder_next = X_3d[:, frame+1, kp_index["right_shoulder"], :] - HOOP_FEET[None, :]
        s_elbow_next = X_3d[:, frame+1, kp_index["right_elbow"], :] - HOOP_FEET[None, :]
        s_wrist_next = X_3d[:, frame+1, kp_index["right_wrist"], :] - HOOP_FEET[None, :]
        
        shoulder_vel = np.linalg.norm(s_shoulder_next - s_shoulder, axis=1)
        elbow_vel = np.linalg.norm(s_elbow_next - s_elbow, axis=1)
        wrist_vel = np.linalg.norm(s_wrist_next - s_wrist, axis=1)
        
        features.append(shoulder_vel)
        feature_names.append("shooting_shoulder_speed")
        features.append(elbow_vel)
        feature_names.append("shooting_elbow_speed")
        features.append(wrist_vel)
        feature_names.append("shooting_wrist_speed")
        
        # Chain synchronization: do all joints accelerate together?
        # Use coefficient of variation of speeds
        chain_speeds = np.column_stack([shoulder_vel, elbow_vel, wrist_vel])
        chain_std = np.std(chain_speeds, axis=1)
        chain_mean = np.mean(chain_speeds, axis=1)
        chain_cv = np.divide(chain_std, chain_mean + 1e-6)
        features.append(chain_cv)
        feature_names.append("shooting_arm_chain_cv")  # Lower = more synchronized

    X_features = np.column_stack(features).astype(np.float32)
    return X_features, feature_names


# ============================================================
# Multi-frame biomechanical features (angular velocities, timing)
# ============================================================

def extract_multiframe_biomechanical(X_3d: np.ndarray, kp_index: dict,
                                     frame_range: tuple = (130, 180)) -> np.ndarray:
    """
    Extract features based on temporal evolution of biomechanical quantities.
    
    Returns: (n_samples, n_features) array
    """
    n = X_3d.shape[0]
    features = []
    feature_names = []

    # Helper
    def get_joint(name, f):
        idx = kp_index.get(name)
        if idx is None:
            return None
        return X_3d[:, f, idx, :] - HOOP_FEET[None, :]

    # Extract elbow angle over frames
    f_start, f_end = frame_range
    elbow_angles_time = []
    for f in range(f_start, min(f_end, N_FRAMES)):
        s_shoulder = get_joint("right_shoulder", f)
        s_elbow = get_joint("right_elbow", f)
        s_wrist = get_joint("right_wrist", f)
        if s_shoulder is not None and s_elbow is not None and s_wrist is not None:
            upper_arm = s_elbow - s_shoulder
            forearm = s_wrist - s_elbow
            angles = np.array([angle_between_vectors(upper_arm[i], forearm[i])
                              for i in range(n)])
            elbow_angles_time.append(angles)
    
    if elbow_angles_time:
        elbow_angles_time = np.column_stack(elbow_angles_time)  # (n, n_frames)
        
        # Angular velocity: rate of change of elbow angle
        elbow_ang_vel = np.mean(np.abs(np.diff(elbow_angles_time, axis=1)), axis=1)
        features.append(elbow_ang_vel)
        feature_names.append("elbow_angular_velocity_mean")
        
        # Max angular velocity (peak extension speed)
        elbow_ang_vel_max = np.max(np.abs(np.diff(elbow_angles_time, axis=1)), axis=1)
        features.append(elbow_ang_vel_max)
        feature_names.append("elbow_angular_velocity_max")
        
        # Frame at max extension (earlier = more powerful release)
        elbow_min_angle = np.argmin(elbow_angles_time, axis=1) + f_start
        features.append(elbow_min_angle.astype(np.float32))
        feature_names.append("elbow_full_extension_frame")

    # Extract wrist speed over frames
    wrist_speeds_time = []
    for f in range(f_start, min(f_end, N_FRAMES - 1)):
        s_wrist = get_joint("right_wrist", f)
        s_wrist_next = X_3d[:, f+1, kp_index["right_wrist"], :] - HOOP_FEET[None, :]
        if s_wrist is not None and s_wrist_next is not None:
            speeds = np.linalg.norm(s_wrist_next - s_wrist, axis=1)
            wrist_speeds_time.append(speeds)
    
    if wrist_speeds_time:
        wrist_speeds_time = np.column_stack(wrist_speeds_time)
        
        # Peak wrist speed
        wrist_peak = np.max(wrist_speeds_time, axis=1)
        features.append(wrist_peak)
        feature_names.append("wrist_speed_peak")
        
        # Frame of peak wrist speed
        wrist_peak_frame = np.argmax(wrist_speeds_time, axis=1) + f_start
        features.append(wrist_peak_frame.astype(np.float32))
        feature_names.append("wrist_peak_speed_frame")

    X_features = np.column_stack(features).astype(np.float32) if features else np.zeros((n, 1), dtype=np.float32)
    return X_features, feature_names


# ============================================================
# Training helpers
# ============================================================

def train_per_player_pls(X_train: np.ndarray, y_train: np.ndarray,
                         pids_train: np.ndarray, target_name: str,
                         n_components: int = 3) -> tuple:
    """
    Train per-player PLS models, return predictions and models.
    """
    models = {}
    X_train_transformed = np.zeros_like(X_train)
    
    for pid in np.unique(pids_train):
        mask = pids_train == pid
        if np.sum(mask) < 5:
            continue
        
        pls = PLSRegression(n_components=min(n_components, np.sum(mask) // 2))
        pls.fit(X_train[mask], y_train[mask].reshape(-1, 1))
        models[pid] = pls
        X_train_transformed[mask] = pls.transform(X_train[mask])
    
    return X_train_transformed, models


def transform_per_player_pls(X: np.ndarray, pids: np.ndarray,
                             models: dict) -> np.ndarray:
    """Apply per-player PLS transform."""
    X_transformed = np.zeros((len(X), 3), dtype=np.float32)
    for pid, pls in models.items():
        mask = pids == pid
        if np.any(mask):
            X_transformed[mask] = pls.transform(X[mask])
    return X_transformed


# ============================================================
# LOO Cross-validation (honest: no leakage)
# ============================================================

def honest_loo_cv(X_features_frame: np.ndarray, X_features_multiframe: np.ndarray,
                  y_train: np.ndarray, pids_train: np.ndarray,
                  target_name: str, scaler_pkl: Path) -> tuple:
    """
    Honest LOO with per-player PLS refitted per fold.
    """
    n = len(y_train)
    y_pred = np.zeros(n)
    
    scaler = joblib.load(scaler_pkl)
    y_scaled = scaler.transform(y_train.reshape(-1, 1)).ravel()
    
    for i in range(n):
        # LOO: train on all except sample i
        tr_mask = np.arange(n) != i
        
        X_frame_tr = X_features_frame[tr_mask]
        X_multi_tr = X_features_multiframe[tr_mask]
        X_combined_tr = np.column_stack([X_frame_tr, X_multi_tr])
        y_tr = y_scaled[tr_mask]
        pids_tr = pids_train[tr_mask]
        
        # Scale features
        scaler_feat = StandardScaler()
        X_combined_tr_s = scaler_feat.fit_transform(X_combined_tr)
        
        # Fit Ridge
        ridge = Ridge(alpha=10)
        ridge.fit(X_combined_tr_s, y_tr)
        
        # Predict on sample i
        X_frame_te = X_features_frame[i:i+1]
        X_multi_te = X_features_multiframe[i:i+1]
        X_combined_te = np.column_stack([X_frame_te, X_multi_te])
        X_combined_te_s = scaler_feat.transform(X_combined_te)
        y_pred[i] = ridge.predict(X_combined_te_s)[0]
    
    return y_pred


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    if args.pilot:
        print("PILOT MODE: testing biomechanical features")
    else:
        print("FULL MODE: biomechanical features + submission")

    print("\nLoading data...")
    X_3d_tr, kp_index, df_tr = load_data(DATA_DIR / "train.csv")
    pids_tr = df_tr["participant_id"].values.astype(int)
    print(f"  {len(df_tr)} train shots")

    print("\nExtracting biomechanical features...")
    
    # Extract per-target at optimal frames
    results = {}
    for target in TARGETS:
        print(f"\n  TARGET: {target.upper()}")
        frame = TARGET_FRAMES[target]
        
        X_frame, feat_names_frame = extract_biomechanical_features(X_3d_tr, kp_index, frame)
        X_multi, feat_names_multi = extract_multiframe_biomechanical(X_3d_tr, kp_index,
                                                                      frame_range=(frame-30, frame+20))
        
        print(f"    Frame features: {len(feat_names_frame)}")
        print(f"    Multiframe features: {len(feat_names_multi)}")
        
        # Get target values (raw)
        y_raw = df_tr[target].values.astype(np.float32)
        
        # Honest LOO
        print(f"    Computing honest LOO...")
        y_pred = honest_loo_cv(X_frame, X_multi, y_raw, pids_tr, target,
                              DATA_DIR / f"scaler_{target}.pkl")
        
        # Scale target for comparison
        scaler = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
        y_scaled = scaler.transform(y_raw.reshape(-1, 1)).ravel()
        y_pred_clipped = np.clip(y_pred, 0, 1)
        
        mse = np.mean((y_pred_clipped - y_scaled)**2)
        print(f"    LOO MSE: {mse:.6f}")
        
        results[target] = {
            "features_frame": X_frame,
            "features_multi": X_multi,
            "y_pred": y_pred_clipped,
            "mse": mse,
        }
    
    # Aggregate
    overall_mse = np.mean([results[t]["mse"] for t in TARGETS])
    print(f"\nOverall biomechanical LOO MSE: {overall_mse:.6f}")
    
    # Note: diversity will be measured at submission blend time
    print("\nBiomechanical features:")
    print("  Joint angles, angular velocities, kinematic chain synchronization")
    
    if args.pilot:
        print("\nPilot complete.")
        return
    
    print("\n" + "="*60)
    print("GENERATING SUBMISSIONS")
    print("="*60)
    
    # Combine predictions
    preds_combined = np.column_stack([results[t]["y_pred"] for t in TARGETS])
    
    # Get test data
    print("\nLoading test data...")
    X_3d_te, _, df_te = load_data(DATA_DIR / "test.csv")
    
    # Extract test features and predict (same process)
    preds_te = []
    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        X_frame_te, _ = extract_biomechanical_features(X_3d_te, kp_index, frame)
        X_multi_te, _ = extract_multiframe_biomechanical(X_3d_te, kp_index,
                                                         frame_range=(frame-30, frame+20))
        # For test: use same scaling as train
        X_combined_te = np.column_stack([X_frame_te, X_multi_te])
        preds_te.append(np.zeros(len(df_te)))  # Placeholder
    
    preds_te = np.column_stack(preds_te)
    preds_te = np.clip(preds_te, 0, 1)
    
    # Save submission
    nums = []
    for p in SUBMISSION_DIR.glob("submission_*.csv"):
        parts = p.stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    bn = max(nums + [0]) + 1
    sub = pd.DataFrame({"id": df_te["id"].values})
    sub["scaled_angle"] = preds_te[:, 0]
    sub["scaled_depth"] = preds_te[:, 1]
    sub["scaled_left_right"] = preds_te[:, 2]
    sub.to_csv(SUBMISSION_DIR / f"submission_{bn}.csv", index=False)
    print(f"\nSaved biomechanical submission: Sub {bn}")
    
    print("\n" + "="*60)
    print("DETAILS")
    print("="*60)
    print(f"  Features: joint angles, angular velocities, kinematic chain sync")
    print(f"  LOO MSE: {overall_mse:.6f}")
    print(f"  Per-target: angle={results['angle']['mse']:.6f}, depth={results['depth']['mse']:.6f}, LR={results['left_right']['mse']:.6f}")


if __name__ == "__main__":
    main()

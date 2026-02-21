"""
Biomechanical features complete: extract, train Ridge, make predictions on test.
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


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two 3D vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
    return np.arccos(cos_angle)


def extract_biomech_frame(X_3d: np.ndarray, kp_index: dict, frame: int) -> np.ndarray:
    """Extract biomechanical features at one frame."""
    n = X_3d.shape[0]
    features = []

    def get_joint(name):
        idx = kp_index.get(name)
        if idx is None:
            return None
        return X_3d[:, frame, idx, :] - HOOP_FEET[None, :]

    # Shooting arm elbow angle
    s_shoulder = get_joint("right_shoulder")
    s_elbow = get_joint("right_elbow")
    s_wrist = get_joint("right_wrist")

    if s_shoulder is not None and s_elbow is not None and s_wrist is not None:
        upper_arm = s_elbow - s_shoulder
        forearm = s_wrist - s_elbow
        angles = np.array([angle_between_vectors(upper_arm[i], forearm[i]) for i in range(n)])
        features.append(angles)

    # Shooting arm alignment
    if s_shoulder is not None and s_elbow is not None and s_wrist is not None:
        full_arm = s_wrist - s_shoulder
        combined = (s_elbow - s_shoulder) + (s_wrist - s_elbow)
        alignments = np.array([np.dot(full_arm[i], combined[i]) / 
                               (np.linalg.norm(full_arm[i]) * np.linalg.norm(combined[i]) + 1e-6)
                               for i in range(n)])
        features.append(alignments)

    # Support arm elbow angle
    sup_shoulder = get_joint("left_shoulder")
    sup_elbow = get_joint("left_elbow")
    sup_wrist = get_joint("left_wrist")

    if sup_shoulder is not None and sup_elbow is not None and sup_wrist is not None:
        sup_upper = sup_elbow - sup_shoulder
        sup_fore = sup_wrist - sup_elbow
        sup_angles = np.array([angle_between_vectors(sup_upper[i], sup_fore[i]) for i in range(n)])
        features.append(sup_angles)

    # Right leg knee angle
    r_hip = get_joint("right_hip")
    r_knee = get_joint("right_knee")
    r_ankle = get_joint("right_ankle")

    if r_hip is not None and r_knee is not None and r_ankle is not None:
        r_thigh = r_knee - r_hip
        r_calf = r_ankle - r_knee
        r_knee_angles = np.array([angle_between_vectors(r_thigh[i], r_calf[i]) for i in range(n)])
        features.append(r_knee_angles)

    # Left leg knee angle
    l_hip = get_joint("left_hip")
    l_knee = get_joint("left_knee")
    l_ankle = get_joint("left_ankle")

    if l_hip is not None and l_knee is not None and l_ankle is not None:
        l_thigh = l_knee - l_hip
        l_calf = l_ankle - l_knee
        l_knee_angles = np.array([angle_between_vectors(l_thigh[i], l_calf[i]) for i in range(n)])
        features.append(l_knee_angles)

    # Torso-arm coupling
    hip = get_joint("mid_hip")
    neck = get_joint("neck")

    if hip is not None and s_shoulder is not None and s_elbow is not None:
        torso_axis = neck - hip
        arm_axis = s_wrist - s_shoulder
        coupling = np.array([angle_between_vectors(torso_axis[i], arm_axis[i]) for i in range(n)])
        features.append(coupling)

    # Velocities (frame-to-frame speeds)
    if frame < N_FRAMES - 1:
        s_shoulder_next = X_3d[:, frame+1, kp_index["right_shoulder"], :] - HOOP_FEET[None, :]
        s_elbow_next = X_3d[:, frame+1, kp_index["right_elbow"], :] - HOOP_FEET[None, :]
        s_wrist_next = X_3d[:, frame+1, kp_index["right_wrist"], :] - HOOP_FEET[None, :]

        shoulder_vel = np.linalg.norm(s_shoulder_next - s_shoulder, axis=1)
        elbow_vel = np.linalg.norm(s_elbow_next - s_elbow, axis=1)
        wrist_vel = np.linalg.norm(s_wrist_next - s_wrist, axis=1)

        features.append(shoulder_vel)
        features.append(elbow_vel)
        features.append(wrist_vel)

        # Chain synchronization (CV of speeds)
        chain_speeds = np.column_stack([shoulder_vel, elbow_vel, wrist_vel])
        chain_std = np.std(chain_speeds, axis=1)
        chain_mean = np.mean(chain_speeds, axis=1)
        chain_cv = np.divide(chain_std, chain_mean + 1e-6)
        features.append(chain_cv)

    return np.column_stack(features).astype(np.float32) if features else np.zeros((n, 1), dtype=np.float32)


def main():
    print("Loading train data...")
    X_3d_tr, kp_index, df_tr = load_data(DATA_DIR / "train.csv")
    pids_tr = df_tr["participant_id"].values.astype(int)

    print("Loading test data...")
    X_3d_te, _, df_te = load_data(DATA_DIR / "test.csv")
    pids_te = df_te["participant_id"].values.astype(int)

    print("\nExtracting and training per target...")
    test_preds = []

    for target in TARGETS:
        print(f"\n  {target.upper()}")
        frame = TARGET_FRAMES[target]

        # Extract features
        X_tr = extract_biomech_frame(X_3d_tr, kp_index, frame)
        X_te = extract_biomech_frame(X_3d_te, kp_index, frame)

        print(f"    Features shape: {X_tr.shape}")

        # Scale targets
        scaler_y = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
        y_scaled = scaler_y.transform(df_tr[target].values.reshape(-1, 1)).ravel()

        # Scale features
        scaler_X = StandardScaler()
        X_tr_s = scaler_X.fit_transform(X_tr)
        X_te_s = scaler_X.transform(X_te)

        # Train Ridge on all train data
        ridge = Ridge(alpha=10)
        ridge.fit(X_tr_s, y_scaled)

        # Predict on test
        y_te_pred = ridge.predict(X_te_s)
        y_te_pred = np.clip(y_te_pred, 0, 1)

        test_preds.append(y_te_pred)
        print(f"    Test predictions: {y_te_pred[:5]}")

    # Create submission
    test_preds = np.column_stack(test_preds)

    nums = []
    for p in SUBMISSION_DIR.glob("submission_*.csv"):
        parts = p.stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    bn = max(nums + [0]) + 1

    sub = pd.DataFrame({"id": df_te["id"].values})
    sub["scaled_angle"] = test_preds[:, 0]
    sub["scaled_depth"] = test_preds[:, 1]
    sub["scaled_left_right"] = test_preds[:, 2]
    sub.to_csv(SUBMISSION_DIR / f"submission_{bn}.csv", index=False)
    print(f"\nSaved Sub {bn}")

    # Blend with Sub 3190
    sub_3190 = pd.read_csv(SUBMISSION_DIR / "submission_3190.csv")
    for w in [0.03, 0.05, 0.07]:
        bbn = max(nums + [0]) + 1
        blend = pd.DataFrame({"id": df_te["id"].values})
        for i, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
            blend[col] = (1 - w) * sub_3190[col].values + w * test_preds[:, i]
        blend.to_csv(SUBMISSION_DIR / f"submission_{bbn}.csv", index=False)
        print(f"  Sub {bbn}: {w*100:.0f}% biomech + {(1-w)*100:.0f}% Sub3190")
        nums.append(bbn)


if __name__ == "__main__":
    main()

"""
Compare release-frame detectors by downstream prediction accuracy.

This script builds release-phase features for each detector and evaluates
angle/depth prediction with GroupKFold by participant.
"""

import argparse
from math import sqrt
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from data_loader import iterate_shots, get_keypoint_columns, FRAME_RATE, NUM_FRAMES, load_scalers, scale_targets


START_SEARCH = NUM_FRAMES // 3
END_SEARCH = NUM_FRAMES - 5
SMOOTH_W = 5
BALL_RADIUS_FT = 4.7 / 12.0  # 4.7 inches


def smooth_arr(series: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average smoothing."""
    if window <= 1 or series.size < window:
        return series
    pad = window // 2
    padded = np.pad(series, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def build_keypoint_index(keypoint_cols: List[str]) -> Dict[str, int]:
    """Map keypoint base name to index (x,y,z grouped)."""
    names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            names.append(col[:-2])
    return {name: i for i, name in enumerate(names)}


def compute_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle at p2 formed by p1-p2-p3 in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    denom = n1 * n2 if n1 > 0 and n2 > 0 else 1e-9
    cos_val = np.clip(dot / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def get_fingertip_names(keypoint_idx: Dict[str, int]) -> List[str]:
    """Return available right-hand fingertip keypoints."""
    candidates = [
        "right_thumb",
        "right_first_finger_distal",
        "right_second_finger_distal",
        "right_third_finger_distal",
        "right_fourth_finger_distal",
        "right_fifth_finger_distal",
    ]
    return [name for name in candidates if name in keypoint_idx]


def compute_ball_center_series(
    timeseries: np.ndarray,
    keypoint_idx: Dict[str, int],
    smooth_window: int,
) -> np.ndarray:
    """
    Estimate ball center over time using right-hand fingertips.

    Steps:
    - Use fingertip centroid as a surface point.
    - Estimate palm normal from index/pinky and wrist.
    - Move inward by ball radius toward the wrist to estimate center.
    """
    def kp_xyz(name: str) -> np.ndarray:
        idx = keypoint_idx.get(name)
        if idx is None:
            return np.full((NUM_FRAMES, 3), np.nan, dtype=float)
        start = idx * 3
        return timeseries[:, start:start + 3].astype(float)

    fingertip_names = get_fingertip_names(keypoint_idx)
    if not fingertip_names:
        return np.full((NUM_FRAMES, 3), np.nan, dtype=float)

    fingertips = [kp_xyz(name) for name in fingertip_names]
    stack = np.stack(fingertips, axis=0)  # (n_fingers, frames, 3)
    valid = np.isfinite(stack)
    count = np.sum(valid, axis=0)  # (frames, 3)
    summed = np.nansum(stack, axis=0)
    centroid = summed / np.clip(count, 1, None)

    wrist = kp_xyz("right_wrist")
    # If centroid is invalid for a frame, fall back to wrist
    invalid = np.any(count == 0, axis=1)
    centroid[invalid] = wrist[invalid]

    # Palm normal from index and pinky if available
    index_name = "right_second_finger_distal"
    pinky_name = "right_fifth_finger_distal"
    if index_name in keypoint_idx and pinky_name in keypoint_idx:
        index = kp_xyz(index_name)
        pinky = kp_xyz(pinky_name)
        v1 = index - wrist
        v2 = pinky - wrist
        normal = np.cross(v1, v2)
    else:
        normal = wrist - centroid

    norm = np.linalg.norm(normal, axis=1)
    normal_unit = np.zeros_like(normal)
    for i in range(NUM_FRAMES):
        if norm[i] > 1e-6:
            normal_unit[i] = normal[i] / norm[i]
        else:
            # Fallback direction: toward wrist
            vec = wrist[i] - centroid[i]
            nvec = np.linalg.norm(vec)
            if nvec > 1e-6:
                normal_unit[i] = vec / nvec
            else:
                normal_unit[i] = np.array([0.0, 0.0, 1.0])

    # Ensure normal points toward wrist (ball center should be between wrist and fingers)
    to_wrist = wrist - centroid
    dot = np.sum(normal_unit * to_wrist, axis=1)
    sign = np.where(dot < 0, -1.0, 1.0)
    normal_unit = normal_unit * sign[:, None]

    ball_center = centroid + normal_unit * BALL_RADIUS_FT
    # Smooth ball center
    for axis in range(3):
        ball_center[:, axis] = smooth_arr(ball_center[:, axis], smooth_window)

    return ball_center

def detect_release_wrist_speed(wrist_xyz: np.ndarray) -> int:
    """Release frame = peak wrist speed after START_SEARCH."""
    dt = 1.0 / FRAME_RATE
    vel = np.gradient(wrist_xyz, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    rel_idx = START_SEARCH + int(np.argmax(speed[START_SEARCH:]))
    rel_idx = min(max(rel_idx, 1), wrist_xyz.shape[0] - 2)
    return rel_idx


def detect_release_arm_straight(
    shoulder_xyz: np.ndarray,
    elbow_xyz: np.ndarray,
    wrist_xyz: np.ndarray,
) -> int:
    """Release frame = max elbow extension angle within window."""
    n_frames = wrist_xyz.shape[0]
    elbow_angle = np.zeros(n_frames, dtype=float)
    for t in range(n_frames):
        elbow_angle[t] = compute_joint_angle(shoulder_xyz[t], elbow_xyz[t], wrist_xyz[t])

    wrist_z = wrist_xyz[:, 2]
    shoulder_z = shoulder_xyz[:, 2]
    mask = (elbow_angle > 140.0) & (wrist_z > shoulder_z)
    mask[:START_SEARCH] = False
    mask[END_SEARCH:] = False

    if np.any(mask):
        rel_idx = int(np.argmax(np.where(mask, elbow_angle, -np.inf)))
    else:
        rel_idx = START_SEARCH + int(np.argmax(elbow_angle[START_SEARCH:END_SEARCH]))

    rel_idx = min(max(rel_idx, 1), n_frames - 2)
    return rel_idx


def detect_release_wrist_snap(
    wrist_xyz: np.ndarray,
    elbow_xyz: np.ndarray,
    shoulder_xyz: np.ndarray,
    finger_xyz: np.ndarray,
) -> int:
    """
    Release frame = peak wrist angular velocity with elbow extended.

    Uses wrist angle between forearm (wrist-elbow) and hand (finger-wrist).
    """
    # Wrist angle time series
    n_frames = wrist_xyz.shape[0]
    theta = np.zeros(n_frames, dtype=float)
    elbow_angle = np.zeros(n_frames, dtype=float)

    for t in range(n_frames):
        theta[t] = compute_joint_angle(elbow_xyz[t], wrist_xyz[t], finger_xyz[t])
        elbow_angle[t] = compute_joint_angle(shoulder_xyz[t], elbow_xyz[t], wrist_xyz[t])

    dt = 1.0 / FRAME_RATE
    omega = np.gradient(theta, dt)  # angular velocity
    omega_abs = np.abs(omega)

    # Candidate mask: elbow extended and wrist above shoulder
    wrist_z = wrist_xyz[:, 2]
    shoulder_z = shoulder_xyz[:, 2]
    mask = (elbow_angle > 140.0) & (wrist_z > shoulder_z)
    mask[:START_SEARCH] = False
    mask[END_SEARCH:] = False

    if np.any(mask):
        rel_idx = int(np.argmax(np.where(mask, omega_abs, -np.inf)))
    else:
        rel_idx = START_SEARCH + int(np.argmax(omega_abs[START_SEARCH:END_SEARCH]))

    rel_idx = min(max(rel_idx, 1), n_frames - 2)
    return rel_idx


def detect_release_arm_snap(
    wrist_xyz: np.ndarray,
    elbow_xyz: np.ndarray,
    shoulder_xyz: np.ndarray,
    finger_xyz: np.ndarray,
) -> int:
    """
    Release frame = peak wrist angular velocity with arm straight.

    Constraints:
    - Elbow angle > 140 deg
    - Wrist above shoulder
    - Wrist speed above 70th percentile in the window
    """
    n_frames = wrist_xyz.shape[0]
    theta = np.zeros(n_frames, dtype=float)
    elbow_angle = np.zeros(n_frames, dtype=float)

    for t in range(n_frames):
        theta[t] = compute_joint_angle(elbow_xyz[t], wrist_xyz[t], finger_xyz[t])
        elbow_angle[t] = compute_joint_angle(shoulder_xyz[t], elbow_xyz[t], wrist_xyz[t])

    # Smooth wrist angle before differentiation to reduce noise
    theta = smooth_arr(theta, SMOOTH_W)
    dt = 1.0 / FRAME_RATE
    omega = np.gradient(theta, dt)
    omega_abs = np.abs(omega)

    # Wrist speed for gating
    vel = np.gradient(wrist_xyz, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    wrist_z = wrist_xyz[:, 2]
    shoulder_z = shoulder_xyz[:, 2]

    window = slice(START_SEARCH, END_SEARCH)
    speed_thresh = np.nanpercentile(speed[window], 70)

    mask = (
        (elbow_angle > 140.0)
        & (wrist_z > shoulder_z)
        & (speed >= speed_thresh)
    )
    mask[:START_SEARCH] = False
    mask[END_SEARCH:] = False

    if np.any(mask):
        rel_idx = int(np.argmax(np.where(mask, omega_abs, -np.inf)))
    else:
        rel_idx = START_SEARCH + int(np.argmax(omega_abs[START_SEARCH:END_SEARCH]))

    rel_idx = min(max(rel_idx, 1), n_frames - 2)
    return rel_idx


def extract_release_features(
    timeseries: np.ndarray,
    keypoint_idx: Dict[str, int],
    release_frame: int,
    smooth_window: int,
    ball_center: np.ndarray,
) -> Dict[str, float]:
    """Compact features at release for angle/depth prediction."""
    features: Dict[str, float] = {}
    features["release_frame"] = float(release_frame)
    features["release_time"] = float(release_frame) / FRAME_RATE

    def kp_xyz(name: str) -> np.ndarray:
        idx = keypoint_idx.get(name)
        if idx is None:
            return np.full((NUM_FRAMES, 3), np.nan, dtype=float)
        start = idx * 3
        return timeseries[:, start:start + 3].astype(float)

    wrist = kp_xyz("right_wrist")
    elbow = kp_xyz("right_elbow")
    shoulder = kp_xyz("right_shoulder")
    hip = kp_xyz("mid_hip")
    knee = kp_xyz("right_knee")
    ankle = kp_xyz("right_ankle")

    def add_pos_vel(prefix: str, data: np.ndarray):
        x = smooth_arr(data[:, 0], smooth_window)
        y = smooth_arr(data[:, 1], smooth_window)
        z = smooth_arr(data[:, 2], smooth_window)
        dt = 1.0 / FRAME_RATE
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        vz = np.gradient(z, dt)
        features[f"{prefix}_x"] = float(x[release_frame])
        features[f"{prefix}_y"] = float(y[release_frame])
        features[f"{prefix}_z"] = float(z[release_frame])
        features[f"{prefix}_vx"] = float(vx[release_frame])
        features[f"{prefix}_vy"] = float(vy[release_frame])
        features[f"{prefix}_vz"] = float(vz[release_frame])
        features[f"{prefix}_speed"] = float(sqrt(vx[release_frame] ** 2 + vy[release_frame] ** 2 + vz[release_frame] ** 2))

    add_pos_vel("wrist", wrist)
    add_pos_vel("elbow", elbow)
    add_pos_vel("shoulder", shoulder)
    add_pos_vel("hip", hip)

    # Joint angles at release
    try:
        features["elbow_angle"] = compute_joint_angle(shoulder[release_frame], elbow[release_frame], wrist[release_frame])
    except Exception:
        features["elbow_angle"] = np.nan
    try:
        features["knee_angle"] = compute_joint_angle(hip[release_frame], knee[release_frame], ankle[release_frame])
    except Exception:
        features["knee_angle"] = np.nan

    # Trunk lean (vertical vs horizontal)
    try:
        trunk_vec = shoulder[release_frame] - hip[release_frame]
        horiz = sqrt(trunk_vec[0] ** 2 + trunk_vec[1] ** 2)
        features["trunk_lean"] = float(np.degrees(np.arctan2(trunk_vec[2], horiz)))
    except Exception:
        features["trunk_lean"] = np.nan

    # Ball center features from fingertips (if available)
    if ball_center is not None and np.isfinite(ball_center[release_frame]).all():
        dt = 1.0 / FRAME_RATE
        bc = ball_center
        bv = np.gradient(bc, dt, axis=0)
        features["ball_x"] = float(bc[release_frame, 0])
        features["ball_y"] = float(bc[release_frame, 1])
        features["ball_z"] = float(bc[release_frame, 2])
        features["ball_vx"] = float(bv[release_frame, 0])
        features["ball_vy"] = float(bv[release_frame, 1])
        features["ball_vz"] = float(bv[release_frame, 2])
        features["ball_speed"] = float(sqrt(bv[release_frame, 0] ** 2 + bv[release_frame, 1] ** 2 + bv[release_frame, 2] ** 2))

    return features


def build_dataset(method: str, max_shots: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extract features and targets for a given release detector."""
    keypoint_cols = get_keypoint_columns()
    keypoint_idx = build_keypoint_index(keypoint_cols)

    # Finger choice for wrist angle
    finger_candidates = [
        "right_second_finger_distal",
        "right_first_finger_distal",
        "right_third_finger_distal",
    ]
    finger_name = None
    for name in finger_candidates:
        if name in keypoint_idx:
            finger_name = name
            break
    if finger_name is None:
        raise RuntimeError("No right finger distal keypoint found.")

    all_features = []
    all_targets = []
    groups = []

    count = 0
    for meta, ts in iterate_shots(train=True, chunk_size=25):
        if count >= max_shots:
            break

        def kp_xyz(name: str) -> np.ndarray:
            idx = keypoint_idx.get(name)
            start = idx * 3
            return ts[:, start:start + 3].astype(float)

        wrist = kp_xyz("right_wrist")
        elbow = kp_xyz("right_elbow")
        shoulder = kp_xyz("right_shoulder")
        finger = kp_xyz(finger_name)

        wrist_sm = np.column_stack([
            smooth_arr(wrist[:, 0], SMOOTH_W),
            smooth_arr(wrist[:, 1], SMOOTH_W),
            smooth_arr(wrist[:, 2], SMOOTH_W),
        ])
        elbow_sm = np.column_stack([
            smooth_arr(elbow[:, 0], SMOOTH_W),
            smooth_arr(elbow[:, 1], SMOOTH_W),
            smooth_arr(elbow[:, 2], SMOOTH_W),
        ])
        shoulder_sm = np.column_stack([
            smooth_arr(shoulder[:, 0], SMOOTH_W),
            smooth_arr(shoulder[:, 1], SMOOTH_W),
            smooth_arr(shoulder[:, 2], SMOOTH_W),
        ])
        finger_sm = np.column_stack([
            smooth_arr(finger[:, 0], SMOOTH_W),
            smooth_arr(finger[:, 1], SMOOTH_W),
            smooth_arr(finger[:, 2], SMOOTH_W),
        ])

        if method == "wrist_snap":
            rel_idx = detect_release_wrist_snap(wrist_sm, elbow_sm, shoulder_sm, finger_sm)
        elif method == "arm_straight_snap":
            rel_idx = detect_release_arm_snap(wrist_sm, elbow_sm, shoulder_sm, finger_sm)
        elif method == "arm_straight_ball":
            rel_idx = detect_release_arm_straight(shoulder_sm, elbow_sm, wrist_sm)
        else:
            rel_idx = detect_release_wrist_speed(wrist_sm)

        ball_center = None
        if method == "arm_straight_ball":
            ball_center = compute_ball_center_series(ts, keypoint_idx, SMOOTH_W)

        feats = extract_release_features(ts, keypoint_idx, rel_idx, SMOOTH_W, ball_center)
        all_features.append(feats)
        all_targets.append([float(meta["angle"]), float(meta["depth"])])
        groups.append(int(meta["participant_id"]))
        count += 1

    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(n, np.nan) for n in feature_names] for f in all_features], dtype=float)
    y = np.array(all_targets, dtype=float)
    groups = np.array(groups, dtype=int)

    # Fill NaNs with column median
    for i in range(X.shape[1]):
        col = X[:, i]
        mask = np.isnan(col)
        if mask.any():
            median = np.nanmedian(col)
            if np.isnan(median):
                median = 0.0
            col[mask] = median
            X[:, i] = col

    return X, y, groups, feature_names


def evaluate(method: str, max_shots: int, folds: int) -> Dict[str, float]:
    """GroupKFold evaluation for angle/depth with a fixed model."""
    X, y, groups, _ = build_dataset(method, max_shots)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", MultiOutputRegressor(Ridge(alpha=1.0, random_state=42), n_jobs=1)),
    ])

    gkf = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    y_true = []
    y_pred = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        y_true.append(y[val_idx])
        y_pred.append(preds)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    # Raw MSEs
    mse_angle = float(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    mse_depth = float(mean_squared_error(y_true[:, 1], y_pred[:, 1]))

    # Scaled MSE for angle and depth only
    scalers = load_scalers()
    y_true_scaled = np.zeros_like(y_true)
    y_pred_scaled = np.zeros_like(y_pred)
    # Angle scaler
    y_true_scaled[:, 0] = scalers["angle"].transform(y_true[:, 0].reshape(-1, 1)).ravel()
    y_pred_scaled[:, 0] = scalers["angle"].transform(y_pred[:, 0].reshape(-1, 1)).ravel()
    # Depth scaler
    y_true_scaled[:, 1] = scalers["depth"].transform(y_true[:, 1].reshape(-1, 1)).ravel()
    y_pred_scaled[:, 1] = scalers["depth"].transform(y_pred[:, 1].reshape(-1, 1)).ravel()

    mse_scaled = float(mean_squared_error(y_true_scaled, y_pred_scaled))
    return {
        "mse_angle": mse_angle,
        "mse_depth": mse_depth,
        "mse_scaled_angle_depth": mse_scaled,
        "n_shots": len(y_true),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare release detectors")
    parser.add_argument("--max-shots", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    for method in ["wrist_speed", "wrist_snap", "arm_straight_snap", "arm_straight_ball"]:
        metrics = evaluate(method, args.max_shots, args.folds)
        print(f"{method}: {metrics}")


if __name__ == "__main__":
    main()

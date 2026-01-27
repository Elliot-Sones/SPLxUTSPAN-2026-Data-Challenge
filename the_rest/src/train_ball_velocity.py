"""
Train a model to predict ball release velocity (v0) from mocap inputs.

Pipeline:
1) Estimate release frame from right wrist speed.
2) Estimate release position r0 (wrist position at release).
3) Back-solve ball release velocity v0 using targets (angle/depth/left_right)
   by matching the rim-plane angle via a 1-D root search on flight time.
4) Train a model to predict v0 from release-phase features.
5) Evaluate by forward-simulating predicted v0 to angle/depth/left_right.
"""

import argparse
from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

from data_loader import iterate_shots, get_keypoint_columns, FRAME_RATE, NUM_FRAMES, load_scalers, scale_targets


# Court/rim constants (feet)
RIM_CENTER_X = 5.25
RIM_CENTER_Y = -25.0
RIM_PLANE_Z = 10.0
RIM_RADIUS = 0.75  # 9 inches

G_FTPS2 = 32.174


@dataclass
class ShotState:
    """Container for per-shot data used in training."""
    participant_id: int
    r0: np.ndarray  # (3,) release position
    v0: np.ndarray  # (3,) solved ball velocity
    targets: np.ndarray  # (3,) [angle, depth, left_right]
    features: Dict[str, float]


def smooth_arr(series: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average smoothing."""
    if window <= 1 or series.size < window:
        return series
    pad = window // 2
    padded = np.pad(series, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


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


def build_keypoint_index(keypoint_cols: List[str]) -> Dict[str, int]:
    """Map keypoint base name to index (x,y,z grouped)."""
    names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            names.append(col[:-2])
    return {name: i for i, name in enumerate(names)}


def detect_release_frame(
    wrist_xyz: np.ndarray,
    frame_rate: float,
    start_search: int,
) -> int:
    """Release frame = peak wrist speed after start_search."""
    dt = 1.0 / frame_rate
    vel = np.gradient(wrist_xyz, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    rel_idx = start_search + int(np.argmax(speed[start_search:]))
    rel_idx = min(max(rel_idx, 1), wrist_xyz.shape[0] - 2)
    return rel_idx


def solve_v0_from_targets(
    r0: np.ndarray,
    angle_deg: float,
    depth_in: float,
    left_right_in: float,
    t_min: float = 0.1,
    t_max: float = 2.0,
    t_steps: int = 200,
) -> Optional[Tuple[float, np.ndarray, float]]:
    """
    Solve for release velocity v0 given targets and release position.

    Returns:
        (t_star, v0, angle_pred_at_rim) or None if invalid.
    """
    x0, y0, z0 = r0

    # Direction of travel along y (toward the rim)
    direction = np.sign(RIM_CENTER_Y - y0)
    if direction == 0:
        direction = 1.0

    y_front = RIM_CENTER_Y - direction * RIM_RADIUS
    x_star = RIM_CENTER_X + (left_right_in / 12.0)
    y_star = y_front + (depth_in / 12.0)

    def angle_at_t(t: float) -> Tuple[float, np.ndarray]:
        vx0 = (x_star - x0) / t
        vy0 = (y_star - y0) / t
        vz0 = (RIM_PLANE_Z - z0 + 0.5 * G_FTPS2 * t * t) / t
        vxy = sqrt(vx0 * vx0 + vy0 * vy0)
        vz_rim = vz0 - G_FTPS2 * t
        angle = float(np.degrees(np.arctan2(vz_rim, vxy)))
        return angle, np.array([vx0, vy0, vz0], dtype=float)

    grid = np.linspace(t_min, t_max, t_steps)
    fvals = []
    for t in grid:
        ang, _ = angle_at_t(t)
        fvals.append(ang - angle_deg)

    # Find a sign change for bisection
    best_t = None
    for i in range(len(grid) - 1):
        f1 = fvals[i]
        f2 = fvals[i + 1]
        if not np.isfinite(f1) or not np.isfinite(f2):
            continue
        if f1 == 0:
            best_t = grid[i]
            break
        if f1 * f2 < 0:
            a, b = grid[i], grid[i + 1]
            fa, fb = f1, f2
            for _ in range(40):
                mid = 0.5 * (a + b)
                fmid = angle_at_t(mid)[0] - angle_deg
                if fa * fmid <= 0:
                    b, fb = mid, fmid
                else:
                    a, fa = mid, fmid
            best_t = 0.5 * (a + b)
            break

    if best_t is None:
        # Fall back to minimizing absolute error
        abs_vals = np.abs(np.array(fvals))
        idx = int(np.nanargmin(abs_vals))
        best_t = grid[idx]

    angle_pred, v0 = angle_at_t(best_t)
    if not np.all(np.isfinite(v0)):
        return None
    return best_t, v0, angle_pred


def extract_release_features(
    timeseries: np.ndarray,
    keypoint_idx: Dict[str, int],
    release_frame: int,
    smooth_window: int,
) -> Dict[str, float]:
    """Compute a compact set of features around release."""
    features: Dict[str, float] = {}
    features["release_frame"] = float(release_frame)
    features["release_time"] = float(release_frame) / FRAME_RATE

    def kp_xyz(name: str) -> np.ndarray:
        idx = keypoint_idx.get(name)
        if idx is None:
            return np.full((NUM_FRAMES, 3), np.nan, dtype=float)
        start = idx * 3
        return timeseries[:, start:start + 3].astype(float)

    # Key points
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
        elbow_angle = compute_joint_angle(shoulder[release_frame], elbow[release_frame], wrist[release_frame])
        features["elbow_angle"] = elbow_angle
    except Exception:
        features["elbow_angle"] = np.nan

    try:
        knee_angle = compute_joint_angle(hip[release_frame], knee[release_frame], ankle[release_frame])
        features["knee_angle"] = knee_angle
    except Exception:
        features["knee_angle"] = np.nan

    # Trunk lean (vertical vs horizontal)
    try:
        shoulder_mid = shoulder[release_frame]
        trunk_vec = shoulder_mid - hip[release_frame]
        horiz = sqrt(trunk_vec[0] ** 2 + trunk_vec[1] ** 2)
        features["trunk_lean"] = float(np.degrees(np.arctan2(trunk_vec[2], horiz)))
    except Exception:
        features["trunk_lean"] = np.nan

    return features


def targets_from_state(r0: np.ndarray, v0: np.ndarray) -> Optional[np.ndarray]:
    """Forward ballistic from release state to angle/depth/left_right at rim plane."""
    x0, y0, z0 = r0
    vx0, vy0, vz0 = v0

    # Solve z(t) = 10 ft
    A = 0.5 * G_FTPS2
    B = -vz0
    C = (RIM_PLANE_Z - z0)
    disc = B * B - 4 * A * C
    if disc <= 0:
        return None
    root = sqrt(disc)
    t1 = (-B - root) / (2 * A)
    t2 = (-B + root) / (2 * A)
    positives = [t for t in (t1, t2) if t > 0]
    if not positives:
        return None
    t_star = max(positives)

    x_star = x0 + vx0 * t_star
    y_star = y0 + vy0 * t_star

    # Entry angle at rim
    vz_rim = vz0 - G_FTPS2 * t_star
    vxy = sqrt(vx0 * vx0 + vy0 * vy0)
    angle = float(np.degrees(np.arctan2(vz_rim, vxy)))

    # Depth and left_right in inches
    direction = np.sign(vy0)
    if direction == 0:
        direction = 1.0
    y_front = RIM_CENTER_Y - direction * RIM_RADIUS
    depth_in = 12.0 * (y_star - y_front)
    left_right_in = 12.0 * (x_star - RIM_CENTER_X)

    return np.array([angle, depth_in, left_right_in], dtype=float)


def build_dataset(
    max_shots: Optional[int],
    smooth_window: int,
    t_min: float,
    t_max: float,
    t_steps: int,
) -> Tuple[List[ShotState], List[str]]:
    """Extract features and back-solved v0 labels."""
    keypoint_cols = get_keypoint_columns()
    keypoint_idx = build_keypoint_index(keypoint_cols)

    states: List[ShotState] = []
    feature_names: List[str] = []

    count = 0
    for meta, ts in iterate_shots(train=True, chunk_size=25):
        if max_shots is not None and count >= max_shots:
            break

        wrist_idx = keypoint_idx.get("right_wrist")
        if wrist_idx is None:
            continue
        wrist = ts[:, wrist_idx * 3:(wrist_idx + 1) * 3].astype(float)
        wrist_sm = np.column_stack([
            smooth_arr(wrist[:, 0], smooth_window),
            smooth_arr(wrist[:, 1], smooth_window),
            smooth_arr(wrist[:, 2], smooth_window),
        ])

        rel_idx = detect_release_frame(wrist_sm, FRAME_RATE, NUM_FRAMES // 3)
        r0 = wrist_sm[rel_idx]

        angle = float(meta["angle"])
        depth = float(meta["depth"])
        left_right = float(meta["left_right"])

        solved = solve_v0_from_targets(
            r0,
            angle,
            depth,
            left_right,
            t_min=t_min,
            t_max=t_max,
            t_steps=t_steps,
        )
        if solved is None:
            continue

        _, v0, _ = solved
        features = extract_release_features(ts, keypoint_idx, rel_idx, smooth_window)

        if not feature_names:
            feature_names = sorted(features.keys())

        states.append(ShotState(
            participant_id=int(meta["participant_id"]),
            r0=r0.astype(float),
            v0=v0.astype(float),
            targets=np.array([angle, depth, left_right], dtype=float),
            features=features,
        ))
        count += 1

    return states, feature_names


def build_feature_matrix(states: List[ShotState], feature_names: List[str]) -> np.ndarray:
    X = np.array([[s.features.get(n, np.nan) for n in feature_names] for s in states], dtype=float)
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
    return X


def get_model(model_type: str):
    """Pick a regression model for v0 prediction."""
    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
            base = lgb.LGBMRegressor(
                n_estimators=300,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            return MultiOutputRegressor(base, n_jobs=1)
        except Exception:
            pass
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        base = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        return MultiOutputRegressor(base, n_jobs=1)

    # Default fallback
    from sklearn.ensemble import GradientBoostingRegressor
    base = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def evaluate_predictions(
    states: List[ShotState],
    v0_preds: np.ndarray,
) -> Dict[str, float]:
    """Compute raw and scaled MSE for targets from predicted v0."""
    y_true = []
    y_pred = []

    for state, v0 in zip(states, v0_preds):
        pred = targets_from_state(state.r0, v0)
        if pred is None:
            continue
        y_true.append(state.targets)
        y_pred.append(pred)

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if len(y_true) == 0:
        return {"raw_mse": np.nan, "scaled_mse": np.nan}

    raw_mse = float(mean_squared_error(y_true, y_pred))

    scalers = load_scalers()
    y_true_scaled = scale_targets(y_true, scalers)
    y_pred_scaled = scale_targets(y_pred, scalers)
    scaled_mse = float(mean_squared_error(y_true_scaled, y_pred_scaled))

    return {"raw_mse": raw_mse, "scaled_mse": scaled_mse}


def main():
    parser = argparse.ArgumentParser(description="Train v0 predictor from mocap")
    parser.add_argument("--max-shots", type=int, default=200)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--t-min", type=float, default=0.1)
    parser.add_argument("--t-max", type=float, default=2.0)
    parser.add_argument("--t-steps", type=int, default=200)
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["random_forest", "lightgbm", "gbrt"])
    parser.add_argument("--per-participant", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    states, feature_names = build_dataset(
        max_shots=args.max_shots,
        smooth_window=args.smooth_window,
        t_min=args.t_min,
        t_max=args.t_max,
        t_steps=args.t_steps,
    )

    if len(states) == 0:
        print("No valid shots found for v0 solving.")
        return

    X = build_feature_matrix(states, feature_names)
    y_v0 = np.array([s.v0 for s in states], dtype=float)
    groups = np.array([s.participant_id for s in states], dtype=int)

    print(f"Shots used for training: {len(states)}")
    print(f"Features: {len(feature_names)}")

    if args.per_participant:
        # Per-participant CV
        all_true = []
        all_pred = []
        for pid in sorted(set(groups)):
            idx = np.where(groups == pid)[0]
            if len(idx) < 5:
                continue
            kf = KFold(n_splits=min(args.folds, len(idx)), shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(idx):
                tr = idx[train_idx]
                va = idx[val_idx]
                model = get_model(args.model)
                model.fit(X[tr], y_v0[tr])
                preds = model.predict(X[va])
                all_true.extend([states[i] for i in va])
                all_pred.append(preds)
        if not all_pred:
            print("No valid folds for per-participant CV.")
            return
        v0_preds = np.vstack(all_pred)
        metrics = evaluate_predictions(all_true, v0_preds)
        print("Per-participant CV metrics:", metrics)
        return

    # Shared model with GroupKFold by participant
    unique_groups = np.unique(groups)
    all_true = []
    all_pred = []

    if len(unique_groups) < 2:
        # Fallback when only one participant is present in the sample
        folds = min(args.folds, len(states))
        if folds < 2:
            print("Not enough shots for CV (need at least 2).")
            return
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            model = get_model(args.model)
            model.fit(X[train_idx], y_v0[train_idx])
            preds = model.predict(X[val_idx])
            all_true.extend([states[i] for i in val_idx])
            all_pred.append(preds)
        v0_preds = np.vstack(all_pred)
        metrics = evaluate_predictions(all_true, v0_preds)
        print("KFold CV metrics (single participant sample):", metrics)
        return

    gkf = GroupKFold(n_splits=min(args.folds, len(unique_groups)))
    for train_idx, val_idx in gkf.split(X, y_v0, groups):
        model = get_model(args.model)
        model.fit(X[train_idx], y_v0[train_idx])
        preds = model.predict(X[val_idx])
        all_true.extend([states[i] for i in val_idx])
        all_pred.append(preds)

    v0_preds = np.vstack(all_pred)
    metrics = evaluate_predictions(all_true, v0_preds)
    print("GroupKFold CV metrics:", metrics)


if __name__ == "__main__":
    main()

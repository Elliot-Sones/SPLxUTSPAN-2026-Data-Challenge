"""
Velocity-Based GBM for Diversity

Build a gradient boosted model on hand-crafted velocity/acceleration features
at validated frames (150, 153, 170). Uses per-player correlations from the
velocity diagnostic as feature engineering guidance.

Key velocity signals (from VELOCITY_DIAGNOSTIC_RESULTS):
- Depth: left_wrist v_up @ f150 (r=+0.463), right_wrist v_up @ f153 (r=+0.436)
- LR: neck v_lat @ f170 (r=+0.367), shoulders v_lat @ f170 (r=+0.306)
- Angle: weak velocity signal (r<0.18)

This creates a different model family from both the Ridge pipeline and the CNN.
"""

import argparse
import fcntl
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

TARGETS = ["angle", "depth", "left_right"]
# Raw target columns in train.csv (NOT scaled_)
RAW_TARGET_COLS = ["angle", "depth", "left_right"]
# Submission columns (scaled to [0,1])
SUB_TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]

# Key frames from velocity diagnostic
RELEASE_FRAMES = [140, 145, 150, 153, 155, 160]
FOLLOWTHROUGH_FRAMES = [165, 170, 175, 180]

# Key joints for velocity features
KEY_JOINTS = [
    "right_wrist", "left_wrist", "right_elbow", "left_elbow",
    "right_shoulder", "left_shoulder", "neck", "mid_hip",
    "right_knee", "left_knee", "right_ankle", "left_ankle",
    "right_hip", "left_hip", "nose",
]

# Hand joints for fine-grained features
HAND_JOINTS = [
    "right_thumb_tip", "right_index_tip", "right_middle_tip",
    "right_ring_tip", "right_pinky_tip",
    "left_thumb_tip", "left_index_tip", "left_middle_tip",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--base-submission", type=int, default=3190)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for fp in existing:
                parts = fp.stem.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_json_array(s):
    s = str(s).replace("nan", "null").replace("NaN", "null")
    arr = json.loads(s)
    out = np.zeros(240, dtype=float)
    for i in range(min(len(arr), 240)):
        v = arr[i]
        if v is not None:
            out[i] = float(v)
    return out


def load_keypoints(df, joint_names):
    """Load keypoint positions for specified joints. Returns dict: joint -> (N, 240, 3)"""
    n = len(df)
    positions = {}
    for joint in joint_names:
        coords = np.zeros((n, 240, 3), dtype=float)
        for c_idx, coord in enumerate(["x", "y", "z"]):
            col = f"{joint}_{coord}"
            if col in df.columns:
                for i in range(n):
                    coords[i, :, c_idx] = parse_json_array(df[col].iloc[i])
        positions[joint] = coords
    return positions


def compute_velocity(positions, smooth_window=5):
    """Compute smoothed velocity from positions. (N, 240, 3) -> (N, 239, 3)"""
    # Simple diff velocity
    vel = np.diff(positions, axis=1)  # (N, 239, 3)
    # Optional smoothing
    if smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        vel = uniform_filter1d(vel, smooth_window, axis=1)
    return vel


def compute_acceleration(velocity):
    """Compute acceleration from velocity. (N, 239, 3) -> (N, 238, 3)"""
    return np.diff(velocity, axis=1)


def extract_features(df, all_joints, is_train=True):
    """Extract hand-crafted velocity/acceleration features at key frames."""
    n = len(df)
    positions = load_keypoints(df, all_joints)

    features = {}

    for joint in all_joints:
        if joint not in positions:
            continue
        pos = positions[joint]  # (N, 240, 3)
        vel = compute_velocity(pos, smooth_window=5)  # (N, 239, 3)
        acc = compute_acceleration(vel)  # (N, 238, 3)

        # Speed at each frame
        speed = np.linalg.norm(vel, axis=2)  # (N, 239)

        # Extract features at key frames
        for f in RELEASE_FRAMES:
            if f < vel.shape[1]:
                # Velocity components
                features[f"{joint}_vx_{f}"] = vel[:, f, 0]
                features[f"{joint}_vy_{f}"] = vel[:, f, 1]
                features[f"{joint}_vz_{f}"] = vel[:, f, 2]
                features[f"{joint}_speed_{f}"] = speed[:, f]
            if f < acc.shape[1]:
                # Acceleration magnitude
                features[f"{joint}_acc_{f}"] = np.linalg.norm(acc[:, f], axis=1)

        for f in FOLLOWTHROUGH_FRAMES:
            if f < vel.shape[1]:
                features[f"{joint}_vx_{f}"] = vel[:, f, 0]
                features[f"{joint}_vy_{f}"] = vel[:, f, 1]
                features[f"{joint}_vz_{f}"] = vel[:, f, 2]
                features[f"{joint}_speed_{f}"] = speed[:, f]

        # Temporal summary features
        # Peak velocity in release window (140-160)
        release_speed = speed[:, 140:161]
        features[f"{joint}_peak_speed_release"] = np.max(release_speed, axis=1)
        features[f"{joint}_mean_speed_release"] = np.mean(release_speed, axis=1)

        # Speed at peak vs average (consistency)
        peak_frame = np.argmax(release_speed, axis=1) + 140
        features[f"{joint}_peak_frame"] = peak_frame.astype(float)

        # Position at key frames (relative to pelvis)
        if "mid_hip" in positions:
            hip = positions["mid_hip"]
            for f in [150, 153, 170]:
                rel_pos = pos[:, f] - hip[:, f]
                features[f"{joint}_relx_{f}"] = rel_pos[:, 0]
                features[f"{joint}_rely_{f}"] = rel_pos[:, 1]
                features[f"{joint}_relz_{f}"] = rel_pos[:, 2]

    # Inter-joint features
    if "right_wrist" in positions and "right_elbow" in positions:
        for f in [150, 153]:
            wrist_v = compute_velocity(positions["right_wrist"])
            elbow_v = compute_velocity(positions["right_elbow"])
            # Wrist-elbow velocity difference (whip effect)
            diff_v = wrist_v[:, f] - elbow_v[:, f]
            features[f"whip_effect_{f}"] = np.linalg.norm(diff_v, axis=1)

    if "right_wrist" in positions and "left_wrist" in positions:
        for f in [150, 153, 170]:
            rw = positions["right_wrist"][:, f]
            lw = positions["left_wrist"][:, f]
            features[f"hand_separation_{f}"] = np.linalg.norm(rw - lw, axis=1)

    # Convert to DataFrame
    feat_df = pd.DataFrame(features, index=range(n))

    # Handle NaN/Inf
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.fillna(0.0)

    return feat_df


def main():
    args = parse_args()
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Determine available joints
    all_cols = set(train_df.columns)
    available_joints = []
    for joint in KEY_JOINTS + HAND_JOINTS:
        if f"{joint}_x" in all_cols:
            available_joints.append(joint)

    if args.pilot:
        available_joints = available_joints[:8]  # Fewer joints for pilot

    print(f"Using {len(available_joints)} joints: {available_joints[:5]}...")

    print("Extracting features...")
    X_train = extract_features(train_df, available_joints)
    X_test = extract_features(test_df, available_joints)
    print(f"Feature matrix: {X_train.shape[1]} features")

    players_train = train_df["participant_id"].values
    players_test = test_df["participant_id"].values
    unique_players = sorted(np.unique(players_train))

    player_id_map = {p: i for i, p in enumerate(unique_players)}

    # Scale targets to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    target_scalers = {}
    scaled_targets = {}
    for target, raw_col in zip(TARGETS, RAW_TARGET_COLS):
        scaler = MinMaxScaler()
        raw_vals = train_df[raw_col].values.reshape(-1, 1)
        scaled_targets[target] = scaler.fit_transform(raw_vals).ravel()
        target_scalers[target] = scaler

    # Add player ID as feature
    X_train["player_id"] = [player_id_map[p] for p in players_train]
    X_test["player_id"] = [player_id_map.get(p, 0) for p in players_test]

    # Per-target training with per-player LOO
    results = {}
    test_predictions = {}

    for target_idx, (target, sub_col) in enumerate(zip(TARGETS, SUB_TARGET_COLS)):
        print(f"\n=== Target: {target} ===")
        y = scaled_targets[target]

        # LOO predictions
        loo_preds = np.zeros(len(train_df))

        for pid in unique_players:
            pid_mask = players_train == pid
            X_pid = X_train[pid_mask].values
            y_pid = y[pid_mask]
            n_pid = len(y_pid)

            # LOO within player
            for i in range(n_pid):
                mask = np.ones(n_pid, dtype=bool)
                mask[i] = False

                X_tr, y_tr = X_pid[mask], y_pid[mask]
                X_val = X_pid[i:i+1]

                # Multi-seed GBM
                preds_seeds = []
                n_seeds = 1 if args.pilot else 3
                for seed in range(n_seeds):
                    gbm = GradientBoostingRegressor(
                        n_estimators=50 if args.pilot else 200,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        min_samples_leaf=3,
                        random_state=args.seed + seed * 100,
                    )
                    gbm.fit(X_tr, y_tr)
                    pred = gbm.predict(X_val)[0]
                    preds_seeds.append(pred)

                global_idx = np.where(pid_mask)[0][i]
                loo_preds[global_idx] = np.mean(preds_seeds)

        loo_mse = np.mean((y - loo_preds) ** 2)
        print(f"  LOO MSE: {loo_mse:.6f}")
        results[target] = {"loo_mse": loo_mse, "loo_preds": loo_preds}

        # Train full model and predict test
        test_preds_seeds = []
        n_seeds = 1 if args.pilot else 5
        for seed in range(n_seeds):
            # Per-player models
            test_preds = np.zeros(len(test_df))
            for pid in unique_players:
                train_mask = players_train == pid
                test_mask = players_test == pid

                if np.sum(test_mask) == 0:
                    continue

                X_tr = X_train[train_mask].values
                y_tr = y[train_mask]
                X_te = X_test[test_mask].values

                gbm = GradientBoostingRegressor(
                    n_estimators=50 if args.pilot else 200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    min_samples_leaf=3,
                    random_state=args.seed + seed * 100,
                )
                gbm.fit(X_tr, y_tr)
                test_preds[test_mask] = gbm.predict(X_te)

            test_preds_seeds.append(test_preds)

        test_predictions[sub_col] = np.clip(np.mean(test_preds_seeds, axis=0), 0, 1)

    # Summary
    print("\n=== LOO Results ===")
    mean_loo = np.mean([r["loo_mse"] for r in results.values()])
    for target in TARGETS:
        print(f"  {target}: {results[target]['loo_mse']:.6f}")
    print(f"  Mean: {mean_loo:.6f}")

    # Diversity with anchor
    anchor_path = SUBMISSION_DIR / f"submission_{args.base_submission}.csv"
    if anchor_path.exists():
        anchor_df = pd.read_csv(anchor_path)
        print(f"\n=== Diversity vs Sub {args.base_submission} ===")
        for sub_col in SUB_TARGET_COLS:
            r = np.corrcoef(anchor_df[sub_col].values, test_predictions[sub_col])[0, 1]
            print(f"  {sub_col}: r = {r:.4f}")

    # Generate submissions
    standalone_sub = pd.DataFrame({
        "id": test_df["id"].values,
        "scaled_angle": test_predictions["scaled_angle"],
        "scaled_depth": test_predictions["scaled_depth"],
        "scaled_left_right": test_predictions["scaled_left_right"],
    })

    sub_num = get_next_submission_number()
    standalone_sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"\nSub {sub_num}: Velocity GBM standalone")

    # Blends with anchor
    if anchor_path.exists():
        anchor_df = pd.read_csv(anchor_path)
        for blend_pct in [2, 3, 5, 7, 10]:
            w = blend_pct / 100.0
            blend = anchor_df.copy()
            for col in SUB_TARGET_COLS:
                blend[col] = (1 - w) * anchor_df[col].values + w * test_predictions[col]

            blend_num = get_next_submission_number()
            blend.to_csv(SUBMISSION_DIR / f"submission_{blend_num}.csv", index=False)
            print(f"Sub {blend_num}: {blend_pct}% Velocity GBM + {100-blend_pct}% Sub {args.base_submission}")

        # Per-target blends (depth-heavy since velocity has strongest depth signal)
        for depth_w, lr_w in [(0.05, 0.03), (0.07, 0.05), (0.03, 0.05)]:
            blend = anchor_df.copy()
            blend["scaled_depth"] = (1 - depth_w) * anchor_df["scaled_depth"] + depth_w * test_predictions["scaled_depth"]
            blend["scaled_left_right"] = (1 - lr_w) * anchor_df["scaled_left_right"] + lr_w * test_predictions["scaled_left_right"]

            blend_num = get_next_submission_number()
            blend.to_csv(SUBMISSION_DIR / f"submission_{blend_num}.csv", index=False)
            print(f"Sub {blend_num}: depth {int(depth_w*100)}% + LR {int(lr_w*100)}% Velocity GBM on Sub {args.base_submission}")

    print("\nDone!")


if __name__ == "__main__":
    main()

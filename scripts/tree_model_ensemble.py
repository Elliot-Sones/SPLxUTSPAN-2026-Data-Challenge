"""
Tree Model Ensemble (Phase 3 - Diverse Model Development)

XGBoost/LightGBM per-player per-target models for ensemble diversity.
Goal: r < 0.70 correlation with Sub 2503/2622 on at least one target.

Key differences from Ridge pipeline:
- Uses ALL 240 frames via summary statistics (not just 3 frames)
- Non-linear splits capture interactions Ridge misses
- Different feature importance landscape
- Completely different model family

Kill criteria:
- KILL if honest LOO > 0.012 (too weak)
- KILL if correlation with Sub 2503 > 0.85 on ALL targets (no diversity)
- PROCEED if honest LOO < 0.012 AND correlation < 0.80 on at least one target
"""

import json
import time
import fcntl
import argparse
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0

# Key joints to extract features from
KEY_JOINTS = [
    "right_wrist", "right_elbow", "right_shoulder",
    "left_wrist", "left_elbow", "left_shoulder",
    "right_hip", "left_hip", "mid_hip",
    "right_knee", "left_knee",
    "neck", "nose",
    "right_ankle", "left_ankle",
]

# Temporal windows
WINDOWS = {
    "early": (60, 120),      # windup
    "pre_release": (120, 150),  # pre-release
    "release": (145, 170),    # release window
    "follow": (170, 210),     # follow-through
    "full": (80, 200),        # full shot
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


def safe_savgol(x, window, polyorder, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


def load_data(csv_path):
    """Load and parse keypoint data into 3D arrays."""
    df = pd.read_csv(csv_path)
    n = len(df)
    kp_cols = [c for c in df.columns if c.startswith(("right_", "left_", "mid_", "neck", "nose"))
               and c not in ("left_right",)]
    n_kp = len(kp_cols)

    raw = np.zeros((n, n_kp, 240), dtype=np.float32)
    for j, col in enumerate(kp_cols):
        for i in range(n):
            raw[i, j, :] = parse_array_string(df[col].iloc[i])

    # Reshape to (n, 240, n_keypoints_3d) - groups of 3 coords per keypoint
    n_joints = n_kp // 3
    kp3d = np.zeros((n, 240, n_joints, 3), dtype=np.float32)
    joint_names = []
    for j in range(n_joints):
        base = kp_cols[j * 3].rsplit("_", 1)[0]
        joint_names.append(base)
        for c in range(3):
            kp3d[:, :, j, c] = raw[:, j * 3 + c, :]

    # NaN cleanup
    nan_mask = np.isnan(kp3d) | np.isinf(kp3d)
    kp3d[nan_mask] = 0.0

    targets = {}
    target_scalers = {}
    for t in TARGETS:
        # Try both scaled_ and unscaled column names
        for col in [f"scaled_{t}", t]:
            if col in df.columns:
                raw_vals = df[col].values.astype(np.float32)
                if col.startswith("scaled_"):
                    # Already in [0,1]
                    targets[t] = raw_vals
                else:
                    # Need to scale to [0,1]
                    scaler = MinMaxScaler()
                    targets[t] = scaler.fit_transform(raw_vals.reshape(-1, 1)).ravel().astype(np.float32)
                    target_scalers[t] = scaler
                break

    players = df["participant_id"].values if "participant_id" in df.columns else np.ones(n, dtype=int)
    ids = df["id"].values if "id" in df.columns else np.arange(n)

    return kp3d, joint_names, targets, players, ids, target_scalers


def compute_hoop_relative(kp3d, joint_names):
    """Transform keypoints to hoop-relative coordinates."""
    n, T, n_joints, _ = kp3d.shape
    hoop = HOOP_POS.astype(np.float32)

    # Find key joint indices
    joint_idx = {name: i for i, name in enumerate(joint_names)}

    # Hoop-relative: subtract hoop position
    kp_hoop = kp3d.copy()
    for i in range(n):
        for t in range(T):
            kp_hoop[i, t, :, :] -= hoop

    return kp_hoop, joint_idx


def extract_summary_features(kp3d, kp_hoop, joint_names, joint_idx):
    """Extract summary statistics across temporal windows for each keypoint."""
    n = kp3d.shape[0]
    features = []
    feature_names = []

    key_joint_indices = []
    for jname in KEY_JOINTS:
        if jname in joint_idx:
            key_joint_indices.append(joint_idx[jname])

    for ji, jname in zip(key_joint_indices, KEY_JOINTS):
        for wname, (t_start, t_end) in WINDOWS.items():
            for coord in range(3):
                coord_name = ["x", "y", "z"][coord]
                prefix = f"{jname}_{coord_name}_{wname}"

                # Position stats
                vals = kp_hoop[:, t_start:t_end, ji, coord]  # (n, window_len)
                features.append(np.nanmean(vals, axis=1))
                feature_names.append(f"{prefix}_mean")
                features.append(np.nanstd(vals, axis=1))
                feature_names.append(f"{prefix}_std")
                features.append(np.nanmin(vals, axis=1))
                feature_names.append(f"{prefix}_min")
                features.append(np.nanmax(vals, axis=1))
                feature_names.append(f"{prefix}_max")

                # Range
                features.append(np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1))
                feature_names.append(f"{prefix}_range")

                # Velocity stats
                vel = np.diff(vals, axis=1) / DT
                if vel.shape[1] > 0:
                    features.append(np.nanmean(vel, axis=1))
                    feature_names.append(f"{prefix}_vel_mean")
                    features.append(np.nanstd(vel, axis=1))
                    feature_names.append(f"{prefix}_vel_std")
                    features.append(np.nanmax(np.abs(vel), axis=1))
                    feature_names.append(f"{prefix}_vel_peak")

    # Cross-joint features: distances and angles at key frames
    for frame_offset, frame_name in [(150, "release"), (170, "post")]:
        for j1_name, j2_name in [
            ("right_wrist", "right_elbow"),
            ("right_elbow", "right_shoulder"),
            ("right_shoulder", "right_hip"),
            ("left_shoulder", "right_shoulder"),
            ("mid_hip", "neck"),
        ]:
            if j1_name in joint_idx and j2_name in joint_idx:
                j1 = joint_idx[j1_name]
                j2 = joint_idx[j2_name]
                dist = np.linalg.norm(
                    kp_hoop[:, frame_offset, j1, :] - kp_hoop[:, frame_offset, j2, :],
                    axis=1
                )
                features.append(dist)
                feature_names.append(f"dist_{j1_name}_{j2_name}_{frame_name}")

    # Hoop distance at key frames
    for frame_offset, frame_name in [(140, "pre"), (150, "mid"), (160, "post"), (170, "late")]:
        if "right_wrist" in joint_idx:
            ji = joint_idx["right_wrist"]
            # Distance from wrist to hoop (already hoop-relative, so distance to origin)
            dist = np.linalg.norm(kp_hoop[:, frame_offset, ji, :], axis=1)
            features.append(dist)
            feature_names.append(f"wrist_hoop_dist_{frame_name}")

    X = np.column_stack(features)
    # Clean up NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, feature_names


def train_xgb_loo(X, y, players, target_name, use_lgbm=False):
    """Train XGBoost/LightGBM with leave-one-out CV, per-player."""
    n = len(y)
    loo_preds = np.zeros(n)

    unique_players = np.unique(players)

    if use_lgbm:
        import lightgbm as lgb
    else:
        from xgboost import XGBRegressor

    for i in range(n):
        # LOO: train on all except i
        mask_train = np.ones(n, dtype=bool)
        mask_train[i] = False

        X_train, y_train = X[mask_train], y[mask_train]
        X_test = X[i:i+1]

        player_i = players[i]

        # Per-player: only train on same player
        player_mask = players[mask_train] == player_i
        if np.sum(player_mask) < 10:
            # Fallback: use all players if too few same-player samples
            X_tr, y_tr = X_train, y_train
        else:
            X_tr, y_tr = X_train[player_mask], y_train[player_mask]

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_test)

        if use_lgbm:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                verbose=-1,
                random_state=42,
            )
        else:
            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                verbosity=0,
                random_state=42,
            )

        model.fit(X_tr_s, y_tr)
        loo_preds[i] = model.predict(X_te_s)[0]

    loo_mse = np.mean((loo_preds - y) ** 2)
    return loo_preds, loo_mse


def train_xgb_test(X_train, y_train, X_test, players_train, players_test,
                    target_name, use_lgbm=False, n_seeds=5):
    """Train on full train set, predict test, with multi-seed averaging."""
    n_test = X_test.shape[0]
    test_preds_all = np.zeros((n_seeds, n_test))

    unique_players = np.unique(players_test)

    for seed_idx in range(n_seeds):
        seed = [42, 7, 99, 133, 2026][seed_idx % 5]
        test_preds = np.zeros(n_test)

        for p in unique_players:
            train_mask = players_train == p
            test_mask = players_test == p

            if np.sum(train_mask) < 10:
                # Fallback: use all players
                X_tr, y_tr = X_train, y_train
            else:
                X_tr = X_train[train_mask]
                y_tr = y_train[train_mask]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_test[test_mask])

            if use_lgbm:
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=5,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    verbose=-1,
                    random_state=seed,
                )
            else:
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    verbosity=0,
                    random_state=seed,
                )

            model.fit(X_tr_s, y_tr)
            test_preds[test_mask] = model.predict(X_te_s)

        test_preds_all[seed_idx] = test_preds

    # Average across seeds
    return np.mean(test_preds_all, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Quick pilot run (XGB only)")
    parser.add_argument("--model", choices=["xgb", "lgbm", "both"], default="xgb")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--anchor", type=int, default=2622,
                       help="Anchor submission for blending")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("TREE MODEL ENSEMBLE - Phase 3 Diverse Model")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    kp3d_train, joint_names, targets_train, players_train, ids_train, target_scalers = load_data(
        DATA_DIR / "train.csv"
    )
    kp3d_test, _, _, players_test, ids_test, _ = load_data(
        DATA_DIR / "test.csv"
    )
    n_train = kp3d_train.shape[0]
    n_test = kp3d_test.shape[0]
    print(f"  Train: {n_train}, Test: {n_test}")
    print(f"  Joints: {len(joint_names)}, Frames: {kp3d_train.shape[1]}")

    # Compute hoop-relative coordinates
    print("\nComputing hoop-relative coordinates...")
    kp_hoop_train, joint_idx = compute_hoop_relative(kp3d_train, joint_names)
    kp_hoop_test, _ = compute_hoop_relative(kp3d_test, joint_names)

    # Extract features
    print("\nExtracting summary features...")
    X_train, feat_names = extract_summary_features(kp3d_train, kp_hoop_train, joint_names, joint_idx)
    X_test, _ = extract_summary_features(kp3d_test, kp_hoop_test, joint_names, joint_idx)
    print(f"  Features: {X_train.shape[1]} ({len(feat_names)} named)")

    # Load anchor for diversity check
    anchor_path = SUBMISSION_DIR / f"submission_{args.anchor}.csv"
    anchor_df = pd.read_csv(anchor_path)

    models_to_run = []
    if args.model in ("xgb", "both"):
        models_to_run.append(("XGB", False))
    if args.model in ("lgbm", "both"):
        models_to_run.append(("LGBM", True))

    all_results = {}

    for model_name, use_lgbm in models_to_run:
        print(f"\n{'='*40}")
        print(f"Training {model_name} models")
        print(f"{'='*40}")

        loo_preds = {}
        test_preds = {}
        loo_mse = {}

        for target in TARGETS:
            print(f"\n  Target: {target}")
            y = targets_train[target]

            # LOO evaluation
            print(f"    Running LOO...")
            loo_p, mse = train_xgb_loo(X_train, y, players_train, target, use_lgbm)
            loo_preds[target] = loo_p
            loo_mse[target] = mse
            print(f"    LOO MSE: {mse:.6f}")

            # Test predictions
            print(f"    Training final model ({args.n_seeds} seeds)...")
            test_p = train_xgb_test(
                X_train, y, X_test, players_train, players_test,
                target, use_lgbm, args.n_seeds
            )
            test_preds[target] = np.clip(test_p, 0, 1)
            print(f"    Test pred range: [{test_p.min():.4f}, {test_p.max():.4f}]")

        # Overall LOO
        mean_loo = np.mean([loo_mse[t] for t in TARGETS])
        print(f"\n  {model_name} Mean LOO MSE: {mean_loo:.6f}")

        # Diversity check
        print(f"\n  Diversity vs Sub {args.anchor}:")
        for target in TARGETS:
            col = f"scaled_{target}"
            corr = np.corrcoef(test_preds[target], anchor_df[col].values)[0, 1]
            print(f"    {target}: r = {corr:.4f}")

        # Kill criteria check
        print(f"\n  KILL CRITERIA CHECK:")
        if mean_loo > 0.012:
            print(f"    KILL: LOO {mean_loo:.6f} > 0.012 threshold")
        else:
            print(f"    PASS: LOO {mean_loo:.6f} < 0.012")

        max_corr = max(
            np.corrcoef(test_preds[t], anchor_df[f"scaled_{t}"].values)[0, 1]
            for t in TARGETS
        )
        min_corr = min(
            np.corrcoef(test_preds[t], anchor_df[f"scaled_{t}"].values)[0, 1]
            for t in TARGETS
        )
        if min_corr > 0.85:
            print(f"    KILL: All correlations > 0.85 (min={min_corr:.4f})")
        else:
            print(f"    PASS: Min correlation {min_corr:.4f} < 0.85")

        all_results[model_name] = {
            "loo_mse": loo_mse,
            "mean_loo": mean_loo,
            "test_preds": test_preds,
            "loo_preds": loo_preds,
        }

        # Save standalone submission
        sub_df = pd.DataFrame({
            "id": ids_test,
            "scaled_angle": test_preds["angle"],
            "scaled_depth": test_preds["depth"],
            "scaled_left_right": test_preds["left_right"],
        })
        sub_num = get_next_submission_number()
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"\n  Standalone: Sub {sub_num} -> {sub_path}")

        # Blend with anchor
        anchor_vals = {t: anchor_df[f"scaled_{t}"].values for t in TARGETS}
        for w in [0.02, 0.05, 0.10, 0.15, 0.20]:
            blend_df = pd.DataFrame({"id": ids_test})
            for t in TARGETS:
                blend_df[f"scaled_{t}"] = (1 - w) * anchor_vals[t] + w * test_preds[t]
            blend_num = get_next_submission_number()
            blend_path = SUBMISSION_DIR / f"submission_{blend_num}.csv"
            blend_df.to_csv(blend_path, index=False)
            print(f"  Blend {w*100:.0f}%: Sub {blend_num}")

        # Per-target blends (depth and LR only - where we need most diversity)
        for t_target in ["depth", "left_right"]:
            for w in [0.05, 0.10, 0.15]:
                blend_df = pd.DataFrame({"id": ids_test})
                for t in TARGETS:
                    if t == t_target:
                        blend_df[f"scaled_{t}"] = (1 - w) * anchor_vals[t] + w * test_preds[t]
                    else:
                        blend_df[f"scaled_{t}"] = anchor_vals[t]
                blend_num = get_next_submission_number()
                blend_path = SUBMISSION_DIR / f"submission_{blend_num}.csv"
                blend_df.to_csv(blend_path, index=False)
                print(f"  {t_target}-only {w*100:.0f}%: Sub {blend_num}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print(f"{'='*60}")

    # Summary
    for model_name, results in all_results.items():
        print(f"\n{model_name} Summary:")
        print(f"  LOO: angle={results['loo_mse']['angle']:.6f} "
              f"depth={results['loo_mse']['depth']:.6f} "
              f"LR={results['loo_mse']['left_right']:.6f} "
              f"mean={results['mean_loo']:.6f}")


if __name__ == "__main__":
    main()

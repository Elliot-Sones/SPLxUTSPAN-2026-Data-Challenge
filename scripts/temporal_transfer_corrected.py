"""
Temporal Transfer Learning - CORRECTED VERSION

Now that we know competition data has 240 temporal frames, implement proper transfer learning.

Strategy:
1. Pre-train temporal autoencoder on SPL (125 shots) + competition (345 shots)
2. Extract latent motion embeddings
3. Compute velocity/acceleration features
4. Train ensemble models with temporal features
5. Generate submissions
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Paths
DATA_DIR = Path("data")
EXTERNAL_DIR = Path("external_data/SPL-Open-Data/basketball/freethrow")
SUBMISSION_DIR = Path("submission")
OUTPUT_DIR = Path("output/temporal_transfer_corrected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
OPTIMAL_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}


def parse_temporal_array(s):
    """Parse string-serialized array"""
    s_clean = s.replace('nan', 'null')
    arr = json.loads(s_clean)
    arr = [np.nan if x is None else x for x in arr]
    return np.array(arr, dtype=np.float32)


def load_spl_temporal_data():
    """Load SPL temporal sequences"""
    print("Loading SPL temporal data...")
    data_dir = EXTERNAL_DIR / "data/P0001"

    spl_sequences = []
    for trial_file in sorted(data_dir.glob("BB_FT_P0001_T*.json")):
        with open(trial_file) as f:
            trial = json.load(f)

        # Extract wrist trajectory (key for shooting)
        frames_data = []
        for frame_data in trial["tracking"]:
            player = frame_data["data"]["player"]
            # Use wrist coordinates
            frame_vec = []
            for kp in ["R_WRIST", "L_WRIST", "R_ELBOW", "L_ELBOW", "R_SHOULDER", "L_SHOULDER"]:
                if kp in player:
                    frame_vec.extend(player[kp])
                else:
                    frame_vec.extend([np.nan, np.nan, np.nan])
            frames_data.append(frame_vec)

        frames_data = np.array(frames_data, dtype=np.float32)
        # Pad/trim to 240
        if len(frames_data) < 240:
            pad = np.full((240 - len(frames_data), frames_data.shape[1]), np.nan, dtype=np.float32)
            frames_data = np.vstack([frames_data, pad])
        else:
            frames_data = frames_data[:240]

        spl_sequences.append(frames_data)

    print(f"Loaded {len(spl_sequences)} SPL sequences")
    return spl_sequences


def load_competition_temporal_data(df):
    """Load competition temporal sequences"""
    print("Loading competition temporal data...")

    sequences = []
    for idx, row in df.iterrows():
        # Extract wrist/elbow/shoulder trajectories
        wrist_r_x = parse_temporal_array(row["right_wrist_x"])
        wrist_r_y = parse_temporal_array(row["right_wrist_y"])
        wrist_r_z = parse_temporal_array(row["right_wrist_z"])
        wrist_l_x = parse_temporal_array(row["left_wrist_x"])
        wrist_l_y = parse_temporal_array(row["left_wrist_y"])
        wrist_l_z = parse_temporal_array(row["left_wrist_z"])

        elbow_r_x = parse_temporal_array(row["right_elbow_x"])
        elbow_r_y = parse_temporal_array(row["right_elbow_y"])
        elbow_r_z = parse_temporal_array(row["right_elbow_z"])
        elbow_l_x = parse_temporal_array(row["left_elbow_x"])
        elbow_l_y = parse_temporal_array(row["left_elbow_y"])
        elbow_l_z = parse_temporal_array(row["left_elbow_z"])

        shoulder_r_x = parse_temporal_array(row["right_shoulder_x"])
        shoulder_r_y = parse_temporal_array(row["right_shoulder_y"])
        shoulder_r_z = parse_temporal_array(row["right_shoulder_z"])
        shoulder_l_x = parse_temporal_array(row["left_shoulder_x"])
        shoulder_l_y = parse_temporal_array(row["left_shoulder_y"])
        shoulder_l_z = parse_temporal_array(row["left_shoulder_z"])

        # Stack: (240, 18)
        sequence = np.column_stack([
            wrist_r_x, wrist_r_y, wrist_r_z,
            wrist_l_x, wrist_l_y, wrist_l_z,
            elbow_r_x, elbow_r_y, elbow_r_z,
            elbow_l_x, elbow_l_y, elbow_l_z,
            shoulder_r_x, shoulder_r_y, shoulder_r_z,
            shoulder_l_x, shoulder_l_y, shoulder_l_z,
        ])

        sequences.append(sequence)

    print(f"Loaded {len(sequences)} competition sequences")
    return sequences


def extract_velocity_features(sequence):
    """Extract velocity and acceleration features from temporal sequence"""
    # Compute velocity (first derivative)
    velocity = np.diff(sequence, axis=0)
    # Compute acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0)

    # Summary statistics
    features = []

    # Velocity stats
    vel_mag = np.linalg.norm(velocity, axis=1)
    vel_mag_clean = vel_mag[~np.isnan(vel_mag)]
    if len(vel_mag_clean) > 0:
        features.extend([
            np.max(vel_mag_clean),  # max velocity
            np.mean(vel_mag_clean),  # mean velocity
            np.std(vel_mag_clean),  # velocity variation
            np.argmax(vel_mag_clean),  # peak velocity frame
        ])
    else:
        features.extend([0, 0, 0, 0])

    # Acceleration stats
    acc_mag = np.linalg.norm(acceleration, axis=1)
    acc_mag_clean = acc_mag[~np.isnan(acc_mag)]
    if len(acc_mag_clean) > 0:
        features.extend([
            np.max(acc_mag_clean),  # max acceleration
            np.mean(acc_mag_clean),  # mean acceleration
            np.argmax(acc_mag_clean),  # peak acceleration frame
        ])
    else:
        features.extend([0, 0, 0])

    # Trajectory arc characteristics
    wrist_z = sequence[:, 2]  # right wrist height
    wrist_z_clean = wrist_z[~np.isnan(wrist_z)]
    if len(wrist_z_clean) > 0:
        features.extend([
            np.max(wrist_z_clean) - np.min(wrist_z_clean),  # height range
            np.argmax(wrist_z_clean),  # peak height frame
        ])
    else:
        features.extend([0, 0])

    return np.array(features)


def extract_features_with_temporal(df, target_frame=153):
    """Extract features: static keypoints at frame + velocity features"""
    print(f"Extracting features at frame {target_frame}...")

    all_features = []
    for idx, row in df.iterrows():
        # Get sequence
        sequence = load_competition_temporal_data(df.iloc[idx:idx+1])[0]

        # Static features at optimal frame
        static_features = sequence[target_frame, :]

        # Velocity features
        velocity_features = extract_velocity_features(sequence)

        # Combine
        combined_features = np.concatenate([static_features, velocity_features])
        all_features.append(combined_features)

    return np.array(all_features)


def train_with_temporal_features(train_df, test_df):
    """Train models with temporal velocity/acceleration features"""
    print("\n" + "="*80)
    print("Training with temporal features")
    print("="*80)

    results = {}

    for target_name in ["angle", "depth", "left_right"]:
        print(f"\n--- Target: {target_name} ---")
        target_frame = OPTIMAL_FRAMES[target_name]

        # Extract features
        print("Extracting train features...")
        X_train = extract_features_with_temporal(train_df, target_frame)
        print("Extracting test features...")
        X_test = extract_features_with_temporal(test_df, target_frame)
        y_train = train_df[target_name].values

        # Standardize
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Train LightGBM
        lgb_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1
        }
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_scores = cross_val_score(lgb_model, X_train_scaled, y_train, cv=cv, scoring="neg_mean_squared_error")
        lgb_model.fit(X_train_scaled, y_train)
        print(f"LGB CV MSE: {-lgb_scores.mean():.6f}")

        # Predict
        test_pred = lgb_model.predict(X_test_scaled)

        results[target_name] = {
            "predictions": test_pred,
            "cv_mse": -lgb_scores.mean(),
        }

    return results


def scale_targets(values, target_name):
    """Scale raw targets to [0, 1]"""
    ranges = {
        "angle": (23.78, 56.47),
        "depth": (-10.17, 24.97),
        "left_right": (-12.98, 10.15)
    }
    min_val, max_val = ranges[target_name]
    scaled = (values - min_val) / (max_val - min_val)
    return np.clip(scaled, 0, 1)


def generate_submissions(results, test_df):
    """Generate submissions"""
    print("\n" + "="*80)
    print("Generating submissions")
    print("="*80)

    # Get next submission number
    existing_subs = list(SUBMISSION_DIR.glob("submission_*.csv"))
    max_num = max([int(s.stem.split("_")[1]) for s in existing_subs])

    max_num += 1
    sub_num = max_num

    # Scale predictions
    angle_scaled = scale_targets(results["angle"]["predictions"], "angle")
    depth_scaled = scale_targets(results["depth"]["predictions"], "depth")
    lr_scaled = scale_targets(results["left_right"]["predictions"], "left_right")

    # Create submission
    submission_rows = []
    for idx, row in test_df.iterrows():
        shot_id = row["shot_id"]
        submission_rows.append({
            "id": shot_id,
            "scaled_angle": angle_scaled[idx],
            "scaled_depth": depth_scaled[idx],
            "scaled_left_right": lr_scaled[idx]
        })

    sub_df = pd.DataFrame(submission_rows)
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df.to_csv(sub_path, index=False)

    print(f"Generated {sub_path.name}")
    print(f"CV scores: angle={results['angle']['cv_mse']:.6f}, depth={results['depth']['cv_mse']:.6f}, LR={results['left_right']['cv_mse']:.6f}")

    return sub_num


def main():
    print("="*80)
    print("Temporal Transfer Learning (Corrected)")
    print("="*80)

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Train with temporal features
    results = train_with_temporal_features(train_df, test_df)

    # Generate submission
    sub_num = generate_submissions(results, test_df)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Submission {sub_num} generated with temporal velocity/acceleration features")
    print("Test on leaderboard!")


if __name__ == "__main__":
    main()

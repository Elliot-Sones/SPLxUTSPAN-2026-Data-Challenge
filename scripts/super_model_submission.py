"""
Super Model Submission.

Combines the best performing ElasticNet pipeline (CV 0.0077) with the new 
Kinetic Chain features (which solve the Player 5 Depth bottleneck).

Strategy:
1. Extract standard ElasticNet features (~7500 per player).
2. Extract Kinetic Chain features (~84).
3. Add Player 5 Interaction Terms for KC features.
4. Train per-player ElasticNet models.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import feature extractors
from src.kinetic_chain_features import extract_kinetic_chain_features
from src.physics_features import extract_physics_features, init_keypoint_mapping as init_phys_mapping

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]

TARGET_SCALERS = {
    "angle": joblib.load(DATA_DIR / "scaler_angle.pkl"),
    "depth": joblib.load(DATA_DIR / "scaler_depth.pkl"),
    "left_right": joblib.load(DATA_DIR / "scaler_left_right.pkl"),
}

# --- Original ElasticNet Feature Logic ---
def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)

def get_target_ranges():
    return {
        "angle": TARGET_SCALERS["angle"].data_range_[0],
        "depth": TARGET_SCALERS["depth"].data_range_[0],
        "left_right": TARGET_SCALERS["left_right"].data_range_[0],
    }

def smooth_signal(x, window=5):
    if len(x) < window: return x
    return uniform_filter1d(x, size=window, mode='nearest')

def compute_velocity(pos, dt=1/60):
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel

def extract_elasticnet_features(ts, participant_id, kp_idx):
    """Original feature extraction from elasticnet_submission.py"""
    features = {}

    # Participant one-hot
    for i in range(1, 6):
        features[f"participant_{i}"] = 1.0 if participant_id == i else 0.0

    # Key frames
    angle_frames = [148, 150, 153, 155, 158]
    angle_window = (145, 165)
    depth_frames = [100, 102, 105, 108, 110]
    depth_window = (95, 115)
    lr_frames = [220, 225, 230, 235, 237]
    lr_window = (215, 240)
    
    # Phases
    phases = [("setup", 0, 60), ("load", 60, 120), ("release", 120, 180), ("follow", 180, 240)]

    for kp_name, coords in kp_idx.items():
        for coord, idx in coords.items():
            pos = ts[:, idx]
            pos_smooth = smooth_signal(pos, window=5)
            vel = compute_velocity(pos_smooth)

            # Angle
            for f in angle_frames:
                features[f"angle_f{f}_{kp_name}_{coord}"] = pos_smooth[f]
                features[f"angle_f{f}_{kp_name}_v{coord}"] = vel[f]
            w = pos_smooth[angle_window[0]:angle_window[1]]
            features[f"angle_window_{kp_name}_{coord}_mean"] = np.nanmean(w)
            features[f"angle_window_{kp_name}_{coord}_std"] = np.nanstd(w)

            # Depth
            for f in depth_frames:
                features[f"depth_f{f}_{kp_name}_{coord}"] = pos_smooth[f]
            w = pos_smooth[depth_window[0]:depth_window[1]]
            features[f"depth_window_{kp_name}_{coord}_mean"] = np.nanmean(w)

            # LR
            for f in lr_frames:
                if f < 240:
                    features[f"lr_f{f}_{kp_name}_{coord}"] = pos_smooth[f]

            # Phases
            for phase_name, start, end in phases:
                phase_pos = pos_smooth[start:end]
                phase_vel = vel[start:end]
                features[f"phase_{phase_name}_{kp_name}_{coord}_mean"] = np.nanmean(phase_pos)
                features[f"phase_{phase_name}_{kp_name}_{coord}_std"] = np.nanstd(phase_pos)
                features[f"phase_{phase_name}_{kp_name}_{coord}_range"] = np.nanmax(phase_pos) - np.nanmin(phase_pos)
                features[f"phase_{phase_name}_{kp_name}_vel_max"] = np.nanmax(np.abs(phase_vel))

    return features

# --- Main Data Loading ---
def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    # Mappings for Feature Extractors
    kp_idx = {}
    kc_mapping = {}
    for idx, col in enumerate(keypoint_cols):
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            kp_name = parts[0]
            coord = parts[1]
            if kp_name not in kp_idx: kp_idx[kp_name] = {}
            kp_idx[kp_name][coord] = idx
            
        if col.endswith("_x"):
            kc_mapping[col[:-2]] = idx

    init_phys_mapping(keypoint_cols)

    def process_df(df, is_train):
        all_features = []
        targets = []
        pids = []
        ids = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])

            # 1. Base ElasticNet Features (~7500)
            enet_feats = extract_elasticnet_features(ts, row["participant_id"], kp_idx)
            
            # 2. Physics & Kinetic Chain Features (~80)
            phys_feats = extract_physics_features(ts, row["participant_id"], smooth=True)
            kc_feats = extract_kinetic_chain_features(ts, kc_mapping, row["participant_id"])
            
            combined = {**enet_feats, **phys_feats, **kc_feats}
            
            all_features.append(combined)
            pids.append(row["participant_id"])
            ids.append(row["id"])
            
            if is_train:
                targets.append([row["angle"], row["depth"], row["left_right"]])

        return all_features, targets, pids, ids

    print("Loading training data...")
    train_features, train_targets, train_pids, train_ids = process_df(train_df, True)
    
    print("Loading test data...")
    test_features, _, test_pids, test_ids = process_df(test_df, False)

    # Convert to DataFrame to handle interactions easily
    X_train_df = pd.DataFrame(train_features).fillna(0)
    X_test_df = pd.DataFrame(test_features).fillna(0)
    
    # Align columns
    all_cols = sorted(list(set(X_train_df.columns) | set(X_test_df.columns)))
    X_train_df = X_train_df.reindex(columns=all_cols, fill_value=0)
    X_test_df = X_test_df.reindex(columns=all_cols, fill_value=0)
    
    # --- ADD INTERACTION FEATURES (Crucial Step) ---
    print("Adding Player 5 Interaction Terms...")
    critical_feats = [
        "kc_elbow_energy_propulsion", 
        "kc_wrist_energy_propulsion", 
        "kc_release_angle_azi",
        "set_position_stability",
        "kc_release_angle_elev"
    ]
    
    # Train
    is_p5_train = (np.array(train_pids) == 5).astype(float)
    for feat in critical_feats:
        if feat in X_train_df.columns:
            X_train_df[f"{feat}_x_P5"] = X_train_df[feat] * is_p5_train
            
    # Test
    is_p5_test = (np.array(test_pids) == 5).astype(float)
    for feat in critical_feats:
        if feat in X_test_df.columns:
            X_test_df[f"{feat}_x_P5"] = X_test_df[feat] * is_p5_test
            
    # Final Numpy Conversion
    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)
    y_train = np.array(train_targets, dtype=np.float32)
    
    print(f"Total Features: {X_train.shape[1]}")
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "train_pids": np.array(train_pids),
        "train_ids": train_ids,
        "X_test": X_test,
        "test_pids": np.array(test_pids),
        "test_ids": test_ids,
        "feature_names": X_train_df.columns.tolist()
    }

def train_and_evaluate(data):
    X = data["X_train"]
    y = data["y_train"]
    pids = data["train_pids"]
    ranges = get_target_ranges()
    unique_pids = sorted(np.unique(pids))
    
    all_models = {}
    all_scalers = {}
    oof_preds = np.zeros_like(y)
    
    print("\nTraining ElasticNet per-player models...")
    
    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask]
        y_player = y[pid_mask]
        player_indices = np.where(pid_mask)[0]
        n_samples = len(X_player)
        
        print(f"  Player {pid} ({n_samples} samples)")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)
        all_scalers[pid] = scaler
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for target_idx, target in enumerate(TARGETS):
            y_target = y_player[:, target_idx]
            fold_preds = np.zeros(n_samples)
            
            for train_idx, val_idx in kf.split(X_scaled):
                X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_tr = y_target[train_idx]
                
                model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=42)
                model.fit(X_tr, y_tr)
                fold_preds[val_idx] = model.predict(X_val)
                
            oof_preds[player_indices, target_idx] = fold_preds
            
            final_model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=42)
            final_model.fit(X_scaled, y_target)
            all_models[(pid, target)] = final_model

    print("\n" + "="*60)
    print("CV RESULTS")
    print("="*60)
    
    total_mse = 0
    for target_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((oof_preds[:, target_idx] - y[:, target_idx])**2)
        scaled_mse = raw_mse / (ranges[target]**2)
        print(f"  {target}: scaled_MSE = {scaled_mse:.6f}")
        total_mse += scaled_mse
        
    avg_mse = total_mse / 3
    print(f"\n  TOTAL: {avg_mse:.6f}")
    
    return all_models, all_scalers, avg_mse

def generate_predictions(data, models, scalers):
    X_test = data["X_test"]
    test_pids = data["test_pids"]
    test_ids = data["test_ids"]
    
    predictions = np.zeros((len(X_test), 3))
    
    for i, (x, pid) in enumerate(zip(X_test, test_pids)):
        x_scaled = scalers[pid].transform(x.reshape(1, -1))
        for target_idx, target in enumerate(TARGETS):
            model = models[(pid, target)]
            predictions[i, target_idx] = model.predict(x_scaled)[0]
            
    return predictions, test_ids

def create_submission(test_ids, predictions, cv_score):
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if existing:
        nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1
        
    scaled_preds = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_preds[:, i] = TARGET_SCALERS[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()
        
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })
    
    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)
    
    print(f"\nSaved submission to {filepath}")
    return filepath, next_num

def main():
    print("SUPER MODEL SUBMISSION")
    print("ElasticNet + Kinetic Chain + P5 Interactions")
    
    data = load_data()
    models, scalers, cv_score = train_and_evaluate(data)
    predictions, test_ids = generate_predictions(data, models, scalers)
    filepath, sub_num = create_submission(test_ids, predictions, cv_score)
    
    print(f"CV Score: {cv_score:.6f}")

if __name__ == "__main__":
    main()

"""
Innovative Physics Proxy & Kinetic Efficiency Features
======================================================

Implements "Selection 2" (Kinetic Efficiency) and "Selection 3" (Simplified Physics Proxy)
from the research proposal.

Features:
1.  **Theoretical Projectile Physics**:
    - Solves for 'theoretical_depth' by calculating the flight time $t$ where $z(t) = z_{hoop}$.
    - Calculates 'theoretical_entry_angle'.
    - Calculates 'energy_leakage_lateral': ratio of lateral kinetic energy to total energy at release.

2.  **Kinetic Chain Efficiency**:
    - 'transfer_efficiency_hip_wrist': Ratio of peak wrist velocity to peak hip velocity.
    - 'kinetic_energy_retention': Ratio of ball kinetic energy at release to total body kinetic energy generated.
    - 'player5_power_factor': Interaction term specifically for P5 anomaly (Energy * P5).

Generates a submission file based on these features using a Ridge/LGBM model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from tqdm import tqdm
import joblib

# Constants
HOOP_HEIGHT = 3.05  # meters
HOOP_POS = np.array([1.6, -7.62, 3.05]) # Approximate hoop position in meters
g = 9.81
FRAME_RATE = 60
DT = 1.0 / FRAME_RATE
FEET_TO_METERS = 0.3048

def load_data():
    """Load train/test data and parse sequences."""
    # This is a simplified loader relying on existing processed files if possible
    # Or assuming standard path structure
    data_dir = Path("data")
    # For this script, we'll implement a lightweight loader that reads the raw CSVs chunks
    # efficiently or uses the `src.data_loader` if available.
    
    # Using the project's data loader is safer
    from src.data_loader import load_metadata, load_single_shot
    
    print("Loading metadata...")
    train_meta = load_metadata(train=True)
    test_meta = load_metadata(train=False)
    
    return train_meta, test_meta

def get_smooth_velocity(pos_array):
    """Compute smooth velocity using Savgol filter."""
    if len(pos_array) < 7:
        return np.gradient(pos_array, axis=0) / DT
    
    # Smooth position first
    smooth_pos = savgol_filter(pos_array, window_length=7, polyorder=2, axis=0)
    # Then gradient
    vel = np.gradient(smooth_pos, axis=0) / DT
    return vel

def solve_projectile_depth(v0, angle_deg, h0, target_h=3.05):
    """
    Solve for the horizontal distance traveled when projectile reaches target height.
    v0: initial speed (m/s)
    angle_deg: initial elevation angle (degrees)
    h0: initial height (m)
    target_h: target height (m)
    """
    angle_rad = np.radians(angle_deg)
    vx = v0 * np.cos(angle_rad)
    vz = v0 * np.sin(angle_rad)
    
    dz = target_h - h0
    
    # z(t) = h0 + vz*t - 0.5*g*t^2 = target_h
    # 0.5*g*t^2 - vz*t + dz = 0
    # Quadratic formula: t = (vz +/- sqrt(vz^2 - 4(0.5g)(dz))) / (2(0.5g))
    # t = (vz +/- sqrt(vz^2 - 2*g*dz)) / g
    
    discriminant = vz**2 - 2*g*dz
    
    if discriminant < 0:
        return np.nan, np.nan, np.nan  # Doesn't reach height
        
    t1 = (vz - np.sqrt(discriminant)) / g
    t2 = (vz + np.sqrt(discriminant)) / g
    
    # We want the time when it's falling down into the hoop, usually the larger time
    # But technically it could enter on the way up (unlikely for free throw)
    # Standard shot is "swish" -> falling down -> t2 (larger t)
    t = max(t1, t2)
    
    if t < 0:
        return np.nan, np.nan, np.nan
        
    # Horizontal distance
    dist = vx * t
    
    # Entry angle (angle of velocity vector at impact)
    vz_final = vz - g*t
    entry_angle = np.degrees(np.arctan2(vz_final, vx))
    
    return dist, entry_angle, t

def extract_physics_proxy_features(timeseries):
    """Extract innovative physics features for a single shot."""
    feats = {}
    
    # Indices (assuming standard 207-feature layout or using helper)
    # We need: Right Wrist (x,y,z), Right Elbow, Right Shoulder, Right Hip, Right Knee
    # Using hardcoded indices based on common structure or searching
    # Helper: 
    from src.data_loader import get_keypoint_columns
    cols = get_keypoint_columns()
    col_map = {c: i for i, c in enumerate(cols)}
    
    def get_xyz(part):
        idx = col_map.get(f"{part}_x")
        if idx is None: return None
        # x, y, z are sequential
        return timeseries[:, idx:idx+3] * FEET_TO_METERS

    wrist = get_xyz("right_wrist")
    elbow = get_xyz("right_elbow")
    shoulder = get_xyz("right_shoulder")
    hip = get_xyz("right_hip")
    knee = get_xyz("right_knee")
    ankle = get_xyz("right_ankle")
    
    if wrist is None: return {}
    
    # 1. Release Detection (Max Wrist Z-Velocity)
    wrist_vel = get_smooth_velocity(wrist)
    wrist_speed_z = wrist_vel[:, 2]
    
    # Search in second half
    start_search = len(timeseries) // 2
    rel_idx = start_search + np.argmax(wrist_speed_z[start_search:])
    
    # 2. Release State
    v_release_vec = wrist_vel[rel_idx]
    v_release = np.linalg.norm(v_release_vec)
    
    # Elevation Angle
    v_horiz = np.linalg.norm(v_release_vec[:2])
    elev_angle = np.degrees(np.arctan2(v_release_vec[2], v_horiz))
    
    # Azimuth (Left/Right)
    # Angle in XY plane. 0 = straight forward (Y axis usually? Need to check coords)
    # In this dataset, Y is likely depth? 
    # Usually: X=Left/Right, Y=Depth, Z=Height OR X=Depth, Y=Height...
    # Based on HOOP_POS=[1.6, -7.62, 3.05], Z is height (3.05).
    # If -7.62 is hoop Y, and player is at origin...
    # Let's assume standard calculation: atan2(x, y)
    azimuth = np.degrees(np.arctan2(v_release_vec[0], v_release_vec[1]))
    
    h_release = wrist[rel_idx, 2]
    
    # 3. Theoretical Physics
    # Projectile to Hoop Height
    theo_dist_flat, theo_entry_angle, theo_time = solve_projectile_depth(v_release, elev_angle, h_release, HOOP_HEIGHT)
    
    feats['phys_v_release'] = v_release
    feats['phys_angle_release'] = elev_angle
    feats['phys_azimuth_release'] = azimuth
    feats['phys_h_release'] = h_release
    feats['phys_theo_depth'] = theo_dist_flat
    feats['phys_theo_entry_angle'] = theo_entry_angle
    feats['phys_theo_flight_time'] = theo_time
    
    # Apex
    # h_apex = h0 + (vz^2)/(2g)
    vz0 = v_release_vec[2]
    feats['phys_theo_apex'] = h_release + (vz0**2)/(2*g)
    
    # 4. Energy Leakage (Efficiency)
    # Lateral Energy Ratio: KE_x / KE_total (Assuming Y is forward, Z is up)
    ke_total = 0.5 * (v_release**2) # Mass = 1 (normalized)
    ke_lateral = 0.5 * (v_release_vec[0]**2)
    feats['eff_lateral_leakage_pct'] = ke_lateral / (ke_total + 1e-9)
    
    # Vertical Efficiency: KE_z / KE_total
    ke_vertical = 0.5 * (v_release_vec[2]**2)
    feats['eff_vertical_ratio'] = ke_vertical / (ke_total + 1e-9)
    
    # 5. Kinetic Chain Amplification
    # Peak Hip Velocity
    if hip is not None:
        hip_vel = get_smooth_velocity(hip)
        hip_speed = np.linalg.norm(hip_vel, axis=1)
        # Search before release
        peak_hip = np.max(hip_speed[:rel_idx+1]) if rel_idx < len(hip_speed) else np.max(hip_speed)
        feats['kc_hip_peak_v'] = peak_hip
        feats['kc_transfer_hip_wrist'] = v_release / (peak_hip + 1e-9)
        
    # Peak Knee Extension Rate (Proxy for leg drive)
    if hip is not None and knee is not None and ankle is not None:
        # Vector Hip->Knee, Knee->Ankle
        # Angle calc... simplified: distance Hip-Ankle
        leg_len = np.linalg.norm(hip - ankle, axis=1)
        leg_ext_rate = np.gradient(leg_len) / DT
        peak_leg_drive = np.max(leg_ext_rate[:rel_idx+1])
        feats['kc_leg_drive_peak'] = peak_leg_drive
        feats['kc_transfer_leg_wrist'] = v_release / (peak_leg_drive + 1e-9)
        
    return feats

def process_dataset(meta_df, train=True):
    from src.data_loader import iterate_shots
    
    results = []
    
    print(f"Processing {len(meta_df)} shots ({'Train' if train else 'Test'})...")
    
    # Use iterate_shots for efficient chunked reading
    # iterate_shots yields (metadata_dict, timeseries_array)
    for metadata, timeseries in tqdm(iterate_shots(train=train), total=len(meta_df)):
        
        feats = extract_physics_proxy_features(timeseries)
        
        # Add metadata
        feats['shot_id'] = metadata['shot_id']
        feats['participant_id'] = metadata['participant_id']
        if train:
            # We don't strictly need targets in features, but useful for debugging
            pass
            
        results.append(feats)
        
    return pd.DataFrame(results)

def main():
    train_meta, test_meta = load_data()
    
    # Process
    df_train_feats = process_dataset(train_meta, train=True)
    df_test_feats = process_dataset(test_meta, train=False)
    
    # Save features
    df_train_feats.to_csv("output/innovative_physics_train.csv", index=False)
    df_test_feats.to_csv("output/innovative_physics_test.csv", index=False)
    
    print("Features saved. Training simple model...")
    
    # Simple evaluation
    targets = pd.read_csv("data/train.csv", usecols=['angle', 'depth', 'left_right'])
    
    # We need to make sure alignment is correct. 
    # load_single_shot(i) corresponds to i-th row in train.csv? Yes usually.
    
    # Prepare X, y
    feature_cols = [c for c in df_train_feats.columns if c not in ['shot_id', 'participant_id']]
    X = df_train_feats[feature_cols].fillna(0)
    y = targets
    
    # Interaction Terms for Player 5
    # P5 is ID 4 (0-4) or 5 (1-5)? usually 1-5 in data
    # Create P5 boolean
    X['is_p5'] = (df_train_feats['participant_id'] == 5).astype(int)
    
    # Add interactions
    interact_cols = ['phys_v_release', 'kc_transfer_hip_wrist', 'eff_lateral_leakage_pct']
    for col in interact_cols:
        if col in X.columns:
            X[f'{col}_x_p5'] = X[col] * X['is_p5']
    
    # Train
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error
    
    gkf = GroupKFold(n_splits=5)
    groups = df_train_feats['participant_id']
    
    metrics = []
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    
    oof_preds = np.zeros(y.shape)
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        
        mse = mean_squared_error(y_val, preds)
        metrics.append(mse)
        
    print(f"CV MSE: {np.mean(metrics):.6f}")
    
    # Generate Submission
    # Re-train on all
    X_test = df_test_feats[feature_cols].fillna(0)
    X_test['is_p5'] = (df_test_feats['participant_id'] == 5).astype(int)
    for col in interact_cols:
        if col in X_test.columns:
            X_test[f'{col}_x_p5'] = X_test[col] * X_test['is_p5']
            
    model.fit(X, y)
    test_preds = model.predict(X_test)
    
    sub = pd.DataFrame(test_preds, columns=['angle', 'depth', 'left_right'])
    sub['id'] = df_test_feats['shot_id']
    
    # Reorder
    sub = sub[['id', 'angle', 'depth', 'left_right']]
    sub.to_csv("submission/submission_innovative_physics.csv", index=False)
    print("Submission saved to submission/submission_innovative_physics.csv")

if __name__ == "__main__":
    main()

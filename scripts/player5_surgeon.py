"""
Player 5 Surgeon: Targeted Repair of Depth and Angle using Kinetic Chain Features.

Base: Submission 3336 (Best current LB).
Target: Player 5 (ID 5).
Method:
1. Extract Kinetic Chain features (Energy, Sequencing, Release Angles).
2. Train specialized Random Forest for P5 Depth and Angle.
3. Surgical replacement of P5 predictions in Sub 3336.
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Add src to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR / "src"))

from kinetic_chain_features import extract_kinetic_chain_features

DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
BASE_SUB_PATH = SUBMISSION_DIR / "submission_3336.csv"
PLAYER5_ID = 5

def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)

def load_and_process_data():
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Filter for Player 5 immediately to save time, but we might need global context?
    # Actually, let's keep all for now to see column names, then filter.
    
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    
    # Create mapping
    mapping = {}
    for i, col in enumerate(keypoint_cols):
        if col.endswith("_x"):
            mapping[col[:-2]] = i // 3
            
    print(f"  Mapping created for {len(mapping)} joints.")

    def extract_features(df):
        features_list = []
        ids = []
        targets = []
        
        # Filter for Player 5
        df_p5 = df[df['participant_id'] == PLAYER5_ID].copy()
        print(f"  Extracting features for {len(df_p5)} Player 5 samples...")
        
        for idx, row in df_p5.iterrows():
            # Reconstruct timeseries (240, n_kp*3)
            # This is slow, but we only have ~74 train and ~100 test for P5
            n_kp = len(keypoint_cols)
            ts = np.zeros((240, n_kp), dtype=np.float32)
            
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])
            
            # Reshape for kinetic chain: needs (240, n_kp) but extract_kinetic_chain_features
            # expects mapping to point to (idx*3) start.
            # Wait, extract_kinetic_chain_features implementation:
            # return timeseries[:, idx*3:(idx+1)*3]
            # So it expects (240, total_channels).
            
            feats = extract_kinetic_chain_features(ts, mapping, PLAYER5_ID)
            features_list.append(feats)
            ids.append(row['id'])
            if 'angle' in row:
                targets.append([row['angle'], row['depth'], row['left_right']])
                
        return pd.DataFrame(features_list), np.array(ids), np.array(targets)

    X_tr, ids_tr, y_tr = extract_features(train_df)
    X_te, ids_te, _ = extract_features(test_df)
    
    return X_tr, y_tr, X_te, ids_te

def load_scaler(target):
    import joblib
    return joblib.load(DATA_DIR / f"scaler_{target}.pkl")

def main():
    if not BASE_SUB_PATH.exists():
        print(f"Error: Base submission {BASE_SUB_PATH} not found.")
        return

    print("="*60)
    print("PLAYER 5 SURGEON")
    print("="*60)
    
    X_tr, y_tr, X_te, ids_te = load_and_process_data()
    
    # Impute NaNs (some kinetic features might be NaN if joints missing)
    X_tr = X_tr.fillna(0.0)
    X_te = X_te.fillna(0.0)
    
    print(f"\nTraining Data: {X_tr.shape}")
    print(f"Test Data: {X_te.shape}")
    
    # Targets
    targets = ['angle', 'depth', 'left_right']
    
    # Load Base Submission
    sub = pd.read_csv(BASE_SUB_PATH)
    print(f"\nBase Submission Loaded: {len(sub)} rows")
    
    # Models
    models = {}
    predictions = {}
    
    # CV Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, target in enumerate(targets):
        print(f"\n--- Processing {target} ---")
        y = y_tr[:, i]
        
        # Load scaler to evaluate in scaled metric (approx)
        # Note: We train on RAW targets, but should check scaled error for context
        try:
            scaler = load_scaler(target)
            y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
        except:
            print("  Warning: Could not load scaler. Using raw error.")
            scaler = None
            y_scaled = y

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)
        
        # CV
        oof_preds = np.zeros_like(y)
        for tr_idx, val_idx in kf.split(X_tr):
            rf.fit(X_tr.iloc[tr_idx], y[tr_idx])
            oof_preds[val_idx] = rf.predict(X_tr.iloc[val_idx])
            
        mse = mean_squared_error(y, oof_preds)
        print(f"  CV MSE (Raw): {mse:.4f}")
        
        if scaler:
            oof_scaled = scaler.transform(oof_preds.reshape(-1, 1)).flatten()
            scaled_mse = mean_squared_error(y_scaled, oof_scaled)
            print(f"  CV MSE (Scaled): {scaled_mse:.4f}")
            
            # Compare with estimated baseline error for P5
            # Baseline P5 Scaled MSE ~ 0.06 (Depth), 0.09 (Angle)
            if target == 'depth' and scaled_mse < 0.06:
                print("  >> IMPROVEMENT DETECTED for Depth!")
            elif target == 'angle' and scaled_mse < 0.09:
                print("  >> IMPROVEMENT DETECTED for Angle!")

        # Train Full
        rf.fit(X_tr, y)
        models[target] = rf
        pred_raw = rf.predict(X_te)
        
        if scaler:
            predictions[target] = scaler.transform(pred_raw.reshape(-1, 1)).flatten()
        else:
            # Should not happen if data structure is correct
            predictions[target] = pred_raw

    # Surgery
    print("\nPerforming Surgery on Base Submission...")
    sub_new = sub.copy()
    
    p5_mask = sub_new['id'].isin(ids_te)
    print(f"  Identified {p5_mask.sum()} rows for Player 5 in submission.")
    
    # We need to map ids_te to submission rows correctly
    # Create a dict mapping id -> prediction
    
    for target in targets:
        # Get column name
        col = f"scaled_{target}"
        
        # Create map
        pred_map = dict(zip(ids_te, predictions[target]))
        
        # Update only P5 rows
        # We blend 50/50 with the base to be safe, or 100% if confident?
        # Given "Surgeon" and specific P5 features, let's go 60% New, 40% Base
        # to pull it towards the physics-informed model.
        
        alpha = 0.6
        if target == 'depth':
            alpha = 0.7 # Higher confidence in Kinetic Chain for depth
        
        original_values = sub_new.loc[p5_mask, col].values
        new_values = sub_new.loc[p5_mask, 'id'].map(pred_map).values
        
        # Check for NaNs (if id mismatch)
        if np.isnan(new_values).any():
             print(f"Warning: NaN in new predictions for {target}. Filling with original.")
             mask_nan = np.isnan(new_values)
             new_values[mask_nan] = original_values[mask_nan]

        blended = (1 - alpha) * original_values + alpha * new_values
        
        print(f"  {target}: Blending {alpha*100}% Surgeon + {(1-alpha)*100}% Base")
        print(f"    Old Mean: {original_values.mean():.4f}, New Mean: {new_values.mean():.4f}, Blended: {blended.mean():.4f}")
        
        sub_new.loc[p5_mask, col] = blended

    # Save
    out_file = SUBMISSION_DIR / "submission_p5_surgeon.csv"
    sub_new.to_csv(out_file, index=False)
    print(f"\nSaved to {out_file}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

def main():
    print("Loading data...")
    
    # Load Data
    rigorous_df = pd.read_csv('physics_engine/output/rigorous_features_all.csv')
    targets_df = pd.read_csv('the_rest/output/ground_truth_velocities.csv')
    submission_ids = pd.read_csv('data/submission.csv')
    
    # Train Set: Shots in targets
    # Inner merge to keep only shots where we have Ground Truth Physics
    train_merged = pd.merge(rigorous_df, targets_df[['shot_id', 'gt_vx', 'gt_vy', 'gt_vz', 
                                                     'angle', 'depth', 'left_right', 
                                                     'release_x', 'release_y', 'release_z']], on='shot_id', how='inner')
    
    # Test Set: Shots NOT in targets (or using submission IDs)
    # We use submission_ids to define the test set order
    # Merge rigorous info onto submission IDs
    # Note: rigorous_df might not have 'id' column populated correctly or consistent with submission.
    # But usually 'id' matches.
    if 'id' in rigorous_df.columns:
        test_merged = pd.merge(submission_ids[['id']], rigorous_df, on='id', how='left')
    else:
        # If no ID, we are stuck unless we have another mapping. 
        # But rigorous_features_all.csv usually has 'id'.
        # Let's assume it does.
        raise ValueError("rigorous_features_all.csv missing 'id' column")

    print(f"Training samples (pre-clean): {len(train_merged)}")
    print(f"Test samples: {len(test_merged)}")
    
    # Drop rows where Targets are NaN
    train_merged = train_merged.dropna(subset=['gt_vx', 'gt_vy', 'gt_vz', 'angle', 'depth', 'left_right'])
    
    # Select Features (Numeric only)
    exclude_cols = ['shot_id', 'participant_id', 'id', 'player_id', 
                    'release_x', 'release_y', 'release_z', 
                    'gt_vx', 'gt_vy', 'gt_vz', 
                    'angle', 'depth', 'left_right', 
                    'target_angle', 'target_depth', 'target_left_right', 'is_train', 'physics_valid', 'error_msg']
    
    feature_cols = [c for c in train_merged.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_merged[c])]
    
    # Drop rows where Features are NaN - REMOVED
    # train_merged = train_merged.dropna(subset=feature_cols)
    print(f"Training samples (post-clean): {len(train_merged)}")
    
    if len(train_merged) == 0:
        raise ValueError("No training data left after cleaning!")

    X1 = train_merged[feature_cols].fillna(0)
    Y1 = train_merged[['gt_vx', 'gt_vy', 'gt_vz']]
    
    # Train M1
    print("Training Model 1...")
    model1 = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, max_depth=3, n_jobs=-1))
    model1.fit(X1, Y1)
    
    V_hat_train = model1.predict(X1)
    
    # Test Prediction M1
    X1_test = test_merged[feature_cols].fillna(0)
    V_hat_test = model1.predict(X1_test)
    
    # Calibration
    diff_x = (train_merged['release_x'] - train_merged['release_pos_x_ft']).mean()
    diff_y = (train_merged['release_y'] - train_merged['release_pos_y_ft']).mean()
    diff_z = (train_merged['release_z'] - train_merged['release_pos_z_ft']).mean()
    
    # Train M2
    print("Training Model 2...")
    X2 = train_merged[['gt_vx', 'gt_vy', 'gt_vz', 'release_x', 'release_y', 'release_z']]
    Y2 = train_merged[['angle', 'depth', 'left_right']]
    
    model2 = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, n_jobs=-1))
    model2.fit(X2, Y2)
    
    # Validation
    P_hat_x = (train_merged['release_pos_x_ft'] + diff_x).values
    P_hat_y = (train_merged['release_pos_y_ft'] + diff_y).values
    P_hat_z = (train_merged['release_pos_z_ft'] + diff_z).values
    
    X2_val = pd.DataFrame({
        'gt_vx': V_hat_train[:, 0],
        'gt_vy': V_hat_train[:, 1],
        'gt_vz': V_hat_train[:, 2],
        'release_x': P_hat_x,
        'release_y': P_hat_y,
        'release_z': P_hat_z
    })
    
    Y2_pred = model2.predict(X2_val)
    print(f"Train Correlation (Angle): {np.corrcoef(Y2_pred[:,0], Y2['angle'].values)[0,1]:.4f}")
    
    # Test Prediction M2
    P_test_x = (test_merged['release_pos_x_ft'] + diff_x).values
    P_test_y = (test_merged['release_pos_y_ft'] + diff_y).values
    P_test_z = (test_merged['release_pos_z_ft'] + diff_z).values
    
    X2_test = pd.DataFrame({
        'gt_vx': V_hat_test[:, 0],
        'gt_vy': V_hat_test[:, 1],
        'gt_vz': V_hat_test[:, 2],
        'release_x': P_test_x,
        'release_y': P_test_y,
        'release_z': P_test_z
    })
    
    Y_test_phys = model2.predict(X2_test)
    
    # Scaling
    scaler_angle = StandardScaler()
    scaler_depth = StandardScaler()
    scaler_lr = StandardScaler()
    
    scaler_angle.fit(Y2['angle'].values.reshape(-1, 1))
    scaler_depth.fit(Y2['depth'].values.reshape(-1, 1))
    scaler_lr.fit(Y2['left_right'].values.reshape(-1, 1))
    
    s_angle = scaler_angle.transform(Y_test_phys[:, 0].reshape(-1, 1)).flatten()
    s_depth = scaler_depth.transform(Y_test_phys[:, 1].reshape(-1, 1)).flatten()
    s_lr = scaler_lr.transform(Y_test_phys[:, 2].reshape(-1, 1)).flatten()
    
    sub_df = pd.DataFrame({
        'id': test_merged['id'],
        'scaled_angle': s_angle,
        'scaled_depth': s_depth,
        'scaled_left_right': s_lr
    })
    
    sub_df.to_csv('submission/submission_ballistic_proxy_final.csv', index=False)
    print("Done.")

if __name__ == "__main__":
    main()

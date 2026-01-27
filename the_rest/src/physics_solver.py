
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_single_shot, get_keypoint_columns
from src.physics_features import init_keypoint_mapping, extract_physics_features
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Physics Constants
G = 9.81  # m/s^2
HOOP_HEIGHT = 3.05  # meters
HOOP_DISTANCE_GUESS = 4.19  # Approx horizontal distance for free throw (4.572m is standard, but release is forward)
# Note: Free throw line is 4.572m (15ft) from backboard. Hoop center is 0.381m (15in) in front of backboard.
# So horizontal distance to hoop center is 4.572 - 0.381 = 4.191m.
HOOP_DIST_STD = 4.191

def solve_range_quadratic(v0, angle_deg, release_height):
    """
    Solve for Range R given initial velocity magnitude v0, entry angle beta, and release height. 
    
    Equation: A*R^2 + B*R + C = 0
    A = g * (1 + tan^2(beta))
    B = 2 * tan(beta) * (v0^2 - 2*g*H)
    C = 2 * H * (2*g*H - v0^2)
    where H = HOOP_HEIGHT - release_height
    """
    H = HOOP_HEIGHT - release_height
    # If release is above hoop (dunk?), H is negative. Usually H > 0.
    
    beta_rad = np.radians(angle_deg)
    tan_beta = np.tan(beta_rad)
    
    # Coefficients
    A = G * (1 + tan_beta**2)
    term = (v0**2 - 2*G*H)
    B = 2 * tan_beta * term
    C = -2 * H * term # Corrected algebraic manipulation: C = 2H(2gH - v0^2) = -2H(v0^2 - 2gH)
    
    # Discriminant
    delta = B**2 - 4*A*C
    
    if delta < 0:
        return np.nan # No real solution
        
    sqrt_delta = np.sqrt(delta)
    
    # Two solutions
    R1 = (-B + sqrt_delta) / (2*A)
    R2 = (-B - sqrt_delta) / (2*A)
    
    # We expect positive range. Usually the larger one corresponds to the shot?
    # Or maybe the smaller one if it's a lob?
    # For a free throw, the arc is usually minimizing energy or optimizing angle.
    # Let's check which one is closer to 4.19m
    
    candidates = [r for r in [R1, R2] if r > 0]
    if not candidates:
        return np.nan
        
    # Return the one closest to standard free throw distance
    return min(candidates, key=lambda x: abs(x - HOOP_DIST_STD))

def run_physics_solver_test(n_shots=100):
    print(f"Testing Physics Solver on {n_shots} shots...")
    
    # Init
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)
    
    results = []
    
    for i in range(n_shots):
        try:
            meta, ts = load_single_shot(i, train=True)
        except StopIteration:
            break
            
        # Get ground truth
        # Note: Targets in train.csv are SCALED? No, loader returns raw values?
        # Let's check data_loader.py again. It returns row["angle"] etc.
        # We need to know if train.csv has raw or scaled values.
        # SPEC.md says "Scale targets using provided scalers before submission".
        # This implies train.csv has RAW values.
        
        gt_angle = meta["angle"] # Entry angle
        gt_depth = meta["depth"] # Depth error
        
        # Get physics inputs
        feats = extract_physics_features(ts, smooth=True)
        v0 = feats.get("velocity_magnitude_release")
        z0 = feats.get("wrist_z_release")
        
        if v0 is None or z0 is None or np.isnan(v0):
            continue
            
        # Solve for Range
        # Note: gt_angle is the ENTRY angle at the hoop.
        # Physics solver gives Total Range R.
        # Predicted Depth = R - HOOP_DIST_STD
        
        # Is gt_angle positive or negative? 
        # Entry angle is usually measured downwards? Or geometric angle?
        # Physics derivation assumed tan(beta) where beta is angle with horizontal.
        # If ball enters from above, beta is negative? 
        # Let's try both signs if unsure, but typically 'entry angle' is reported as positive magnitude in sports analytics
        # or negative angle in physics.
        # However, the formula derivation: tan(beta) = vy/vx. vy is negative at hoop. vx positive.
        # So tan(beta) should be negative.
        # Let's check the range of gt_angle values in data first?
        
        # Assumption: input angle is positive degrees (e.g. 45 deg). We use -angle for physics.
        R_calc = solve_range_quadratic(v0, -abs(gt_angle), z0)
        
        if np.isnan(R_calc):
            continue
            
        # We don't know the exact hoop distance for *this* specific shot (player might stand slightly differently),
        # but we assume it's relative to the standard line.
        # But wait, 'depth' is the target.
        # If we assume depth = R_calc - Mean_R, we can check correlation.
        
        results.append({
            "id": meta["id"],
            "v0": v0,
            "z0": z0,
            "angle_in": gt_angle,
            "depth_gt": gt_depth,
            "R_calc": R_calc
        })
        
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No valid results.")
        return

    # Analyze relationship between R_calc and depth_gt
    # If precise, R_calc should be linearly related to depth_gt.
    # depth_gt = R_calc - D_actual
    
    # Let's fit a simple linear offset
    # depth_gt = R_calc + Intercept
    intercept = (df["depth_gt"] - df["R_calc"]).mean()
    df["depth_pred"] = df["R_calc"] + intercept
    
    mse = mean_squared_error(df["depth_gt"], df["depth_pred"])
    mae = mean_absolute_error(df["depth_gt"], df["depth_pred"])
    corr = df["depth_gt"].corr(df["R_calc"])
    
    print(f"\nResults on {len(df)} shots:")
    print(f"Correlation (Calc Range vs GT Depth): {corr:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Inferred Hoop Distance (Offset): {-intercept:.4f} m")
    
    print("\nSample predictions:")
    print(df[["angle_in", "v0", "R_calc", "depth_gt", "depth_pred"]].head(10))

if __name__ == "__main__":
    run_physics_solver_test()

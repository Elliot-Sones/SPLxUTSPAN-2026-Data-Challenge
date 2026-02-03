"""
Deep analysis of physics simulation results.

Investigates:
1. Why 27.5% of simulations still fail
2. Per-player differences
3. Correlation between physics outputs and actual targets
4. How to improve the target mapping
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "physics_engine"))

from data_loader import iterate_shots, load_scalers, get_keypoint_columns
from core import (
    BasketballSimulator,
    calibrate_scale_factor,
    get_keypoint_indices,
    extract_all_release_params,
)


def analyze_simulation_results():
    """
    Comprehensive analysis of simulation outputs.
    """
    print("=" * 80)
    print("PHYSICS SIMULATION DEEP ANALYSIS")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_idx = get_keypoint_indices(keypoint_cols)
    simulator = BasketballSimulator()
    scalers = load_scalers()

    # Collect all results
    results = []

    for metadata, timeseries in iterate_shots(train=True):
        player_id = metadata['participant_id']
        scale_factor = calibrate_scale_factor(timeseries, keypoint_idx)
        params = extract_all_release_params(timeseries, keypoint_idx, player_id, scale_factor)

        landing, entry_angle, trajectory = simulator.simulate_shot(
            params['position'],
            params['velocity'],
            params['backspin']
        )

        # Scaled targets
        target_angle = scalers['angle'].transform([[metadata['angle']]])[0, 0]
        target_depth = scalers['depth'].transform([[metadata['depth']]])[0, 0]
        target_lr = scalers['left_right'].transform([[metadata['left_right']]])[0, 0]

        results.append({
            'id': metadata['id'],
            'player_id': player_id,
            'success': landing is not None,
            'release_frame': params['release_frame'],
            'pos_x': params['position'][0],
            'pos_y': params['position'][1],
            'pos_z': params['position'][2],
            'vel_x': params['velocity'][0],
            'vel_y': params['velocity'][1],
            'vel_z': params['velocity'][2],
            'speed': np.linalg.norm(params['velocity']),
            'release_angle_deg': np.degrees(np.arctan2(params['velocity'][2], params['velocity'][0])),
            'landing_y': landing[0] if landing is not None else np.nan,
            'landing_z': landing[1] if landing is not None else np.nan,
            'entry_angle': entry_angle if entry_angle is not None else np.nan,
            'target_angle': target_angle,
            'target_depth': target_depth,
            'target_lr': target_lr,
            # Raw targets
            'raw_angle': metadata['angle'],
            'raw_depth': metadata['depth'],
            'raw_lr': metadata['left_right'],
        })

    df = pd.DataFrame(results)

    # --- ANALYSIS 1: Success rate by player ---
    print("\n1. SUCCESS RATE BY PLAYER")
    print("-" * 60)
    for pid in sorted(df['player_id'].unique()):
        player_df = df[df['player_id'] == pid]
        success_rate = player_df['success'].mean() * 100
        n_shots = len(player_df)
        print(f"  Player {pid}: {success_rate:.1f}% ({player_df['success'].sum()}/{n_shots})")

    # --- ANALYSIS 2: Why do simulations fail? ---
    print("\n2. FAILED SIMULATION ANALYSIS")
    print("-" * 60)
    failed = df[~df['success']]
    success = df[df['success']]

    print(f"  Total failed: {len(failed)}/{len(df)}")
    print(f"\n  Comparing failed vs successful shots:")
    for col in ['pos_z', 'vel_x', 'vel_z', 'release_angle_deg']:
        f_mean = failed[col].mean()
        s_mean = success[col].mean()
        print(f"    {col}: failed={f_mean:.2f}, success={s_mean:.2f}, diff={f_mean-s_mean:.2f}")

    # --- ANALYSIS 3: Physics output correlations ---
    print("\n3. PHYSICS OUTPUT vs TARGET CORRELATIONS")
    print("-" * 60)
    valid = df[df['success']]

    # Compute deviations from hoop center
    valid = valid.copy()
    valid['y_deviation'] = valid['landing_y'] - 0.15  # Hoop center at Y=0.15
    valid['z_deviation'] = valid['landing_z'] - 3.05  # Hoop height at Z=3.05

    print(f"  Using {len(valid)} successful shots")
    print(f"\n  Correlations with target_angle:")
    for col in ['landing_y', 'landing_z', 'entry_angle', 'y_deviation', 'z_deviation', 'vel_x', 'vel_z']:
        if col in valid.columns:
            corr, pval = pearsonr(valid[col], valid['target_angle'])
            sig = "*" if pval < 0.05 else ""
            print(f"    {col}: r={corr:.4f} {sig}")

    print(f"\n  Correlations with target_depth:")
    for col in ['landing_y', 'landing_z', 'entry_angle', 'y_deviation', 'z_deviation', 'vel_x', 'vel_z']:
        if col in valid.columns:
            corr, pval = pearsonr(valid[col], valid['target_depth'])
            sig = "*" if pval < 0.05 else ""
            print(f"    {col}: r={corr:.4f} {sig}")

    print(f"\n  Correlations with target_lr:")
    for col in ['landing_y', 'landing_z', 'entry_angle', 'y_deviation', 'z_deviation', 'vel_x', 'vel_z']:
        if col in valid.columns:
            corr, pval = pearsonr(valid[col], valid['target_lr'])
            sig = "*" if pval < 0.05 else ""
            print(f"    {col}: r={corr:.4f} {sig}")

    # --- ANALYSIS 4: Per-player mapping quality ---
    print("\n4. PER-PLAYER MAPPING R-SQUARED")
    print("-" * 60)
    for pid in sorted(df['player_id'].unique()):
        player_valid = valid[valid['player_id'] == pid]
        if len(player_valid) < 10:
            print(f"  Player {pid}: Not enough data ({len(player_valid)} shots)")
            continue

        # Build features
        X = player_valid[['y_deviation', 'z_deviation', 'entry_angle']].values
        y_angle = player_valid['target_angle'].values
        y_depth = player_valid['target_depth'].values
        y_lr = player_valid['target_lr'].values

        # Fit and score
        model = LinearRegression()

        model.fit(X, y_angle)
        r2_angle = model.score(X, y_angle)

        model.fit(X, y_depth)
        r2_depth = model.score(X, y_depth)

        model.fit(X, y_lr)
        r2_lr = model.score(X, y_lr)

        print(f"  Player {pid}: angle R2={r2_angle:.3f}, depth R2={r2_depth:.3f}, lr R2={r2_lr:.3f}")

    # --- ANALYSIS 5: What if we use raw release params instead of physics? ---
    print("\n5. RELEASE PARAMS vs PHYSICS OUTPUT COMPARISON")
    print("-" * 60)

    # Compare: can release params predict targets directly?
    release_cols = ['vel_x', 'vel_y', 'vel_z', 'pos_z', 'release_angle_deg']
    physics_cols = ['y_deviation', 'z_deviation', 'entry_angle']

    print("  Predicting target_angle:")
    X_release = valid[release_cols].values
    X_physics = valid[physics_cols].values
    y = valid['target_angle'].values

    model = LinearRegression()
    model.fit(X_release, y)
    r2_release = model.score(X_release, y)
    model.fit(X_physics, y)
    r2_physics = model.score(X_physics, y)
    print(f"    From release params: R2={r2_release:.3f}")
    print(f"    From physics output: R2={r2_physics:.3f}")

    print("  Predicting target_depth:")
    y = valid['target_depth'].values
    model.fit(X_release, y)
    r2_release = model.score(X_release, y)
    model.fit(X_physics, y)
    r2_physics = model.score(X_physics, y)
    print(f"    From release params: R2={r2_release:.3f}")
    print(f"    From physics output: R2={r2_physics:.3f}")

    # --- ANALYSIS 6: Distribution of physics outputs ---
    print("\n6. PHYSICS OUTPUT DISTRIBUTIONS")
    print("-" * 60)
    print("  Landing position (at hoop plane):")
    print(f"    Y: mean={valid['landing_y'].mean():.3f}, std={valid['landing_y'].std():.3f}")
    print(f"    Z: mean={valid['landing_z'].mean():.3f}, std={valid['landing_z'].std():.3f}")
    print(f"    Entry angle: mean={valid['entry_angle'].mean():.1f}, std={valid['entry_angle'].std():.1f}")
    print(f"\n  Hoop reference: Y=0.15, Z=3.05")
    print(f"  Y deviation: mean={valid['y_deviation'].mean():.3f}, std={valid['y_deviation'].std():.3f}")
    print(f"  Z deviation: mean={valid['z_deviation'].mean():.3f}, std={valid['z_deviation'].std():.3f}")

    # --- ANALYSIS 7: Target distributions ---
    print("\n7. TARGET DISTRIBUTIONS")
    print("-" * 60)
    print(f"  target_angle: mean={df['target_angle'].mean():.4f}, std={df['target_angle'].std():.4f}")
    print(f"  target_depth: mean={df['target_depth'].mean():.4f}, std={df['target_depth'].std():.4f}")
    print(f"  target_lr: mean={df['target_lr'].mean():.4f}, std={df['target_lr'].std():.4f}")

    return df


if __name__ == "__main__":
    df = analyze_simulation_results()

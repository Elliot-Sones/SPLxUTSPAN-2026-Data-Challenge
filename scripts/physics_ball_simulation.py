"""
Physics Ball Simulation Pipeline

Full pipeline that:
1. Loads skeleton data
2. Calibrates scale per shot
3. Detects release frame
4. Extracts release parameters (position, velocity, spin)
5. Simulates ball trajectory with MuJoCo
6. Maps physics outputs to target predictions
7. Generates submission file

Target: < 0.007 MSE (current best: 0.008305)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
import sys

# Add paths
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "physics_engine"))

from data_loader import iterate_shots, load_scalers, get_keypoint_columns
from core import (
    BasketballSimulator,
    calibrate_scale_factor,
    get_keypoint_indices,
    extract_all_release_params,
    TargetMapper,
    calibrate_mean_correction,
    apply_corrections
)

SUBMISSION_DIR = PROJECT_DIR / "submission"


def run_physics_pipeline(verbose: bool = True):
    """
    Run the full physics simulation pipeline.
    """
    print("=" * 80)
    print("PHYSICS BALL SIMULATION PIPELINE")
    print("=" * 80)

    # Initialize simulator
    if verbose:
        print("\n1. Initializing MuJoCo simulator...")
    simulator = BasketballSimulator()

    # Validate physics
    if simulator.validate_physics():
        print("   Physics validation: PASSED")
    else:
        print("   Physics validation: FAILED (continuing anyway)")

    # Get keypoint indices
    keypoint_cols = get_keypoint_columns()
    keypoint_idx = get_keypoint_indices(keypoint_cols)

    if verbose:
        print(f"   Keypoints loaded: {len(keypoint_idx)}")

    # Load scalers
    scalers = load_scalers()

    # ==================== TRAINING PHASE ====================
    if verbose:
        print("\n2. Processing training data...")

    training_data = []
    physics_outputs = []
    actual_targets = []
    player_ids = []

    for metadata, timeseries in iterate_shots(train=True):
        player_id = metadata['participant_id']

        # Scale calibration
        scale_factor = calibrate_scale_factor(timeseries, keypoint_idx)

        # Extract release parameters
        release_params = extract_all_release_params(
            timeseries, keypoint_idx, player_id, scale_factor
        )

        # Simulate shot
        landing, entry_angle, trajectory = simulator.simulate_shot(
            release_params['position'],
            release_params['velocity'],
            release_params['backspin']
        )

        # Store results
        physics_outputs.append({
            'landing_y': landing[0] if landing is not None else None,
            'landing_z': landing[1] if landing is not None else None,
            'entry_angle': entry_angle
        })

        # Scale targets
        actual_targets.append({
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        })

        player_ids.append(player_id)

        training_data.append({
            'id': metadata['id'],
            'player_id': player_id,
            'release_params': release_params,
            'physics_output': physics_outputs[-1],
            'actual_target': actual_targets[-1]
        })

    # Count successful simulations
    n_success = sum(1 for p in physics_outputs if p['landing_y'] is not None)
    if verbose:
        print(f"   Training shots: {len(training_data)}")
        print(f"   Successful simulations: {n_success} ({100*n_success/len(training_data):.1f}%)")

    # ==================== FIT TARGET MAPPER ====================
    if verbose:
        print("\n3. Fitting target mapping models...")

    mapper = TargetMapper()
    mapper.fit(physics_outputs, actual_targets, player_ids)

    # Generate training predictions
    train_predictions = []
    for phys, pid in zip(physics_outputs, player_ids):
        pred = mapper.predict(
            phys['landing_y'],
            phys['landing_z'],
            phys['entry_angle'],
            pid
        )
        train_predictions.append(pred)

    # Calibrate mean corrections
    corrections = calibrate_mean_correction(train_predictions, actual_targets)
    if verbose:
        print(f"   Mean corrections: {corrections}")

    # Apply corrections and compute training MSE
    corrected_preds = [apply_corrections(p, corrections) for p in train_predictions]

    train_mse = 0
    for pred, actual in zip(corrected_preds, actual_targets):
        train_mse += (pred['angle'] - actual['angle'])**2
        train_mse += (pred['depth'] - actual['depth'])**2
        train_mse += (pred['left_right'] - actual['left_right'])**2
    train_mse /= (3 * len(actual_targets))

    if verbose:
        print(f"   Training MSE: {train_mse:.6f}")

    # ==================== CROSS-VALIDATION ====================
    if verbose:
        print("\n4. Cross-validation...")

    gkf = GroupKFold(n_splits=5)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(training_data, groups=player_ids)):
        # Fit mapper on train fold
        fold_mapper = TargetMapper()
        fold_physics = [physics_outputs[i] for i in train_idx]
        fold_targets = [actual_targets[i] for i in train_idx]
        fold_pids = [player_ids[i] for i in train_idx]
        fold_mapper.fit(fold_physics, fold_targets, fold_pids)

        # Predict on validation fold
        fold_preds = []
        for i in val_idx:
            pred = fold_mapper.predict(
                physics_outputs[i]['landing_y'],
                physics_outputs[i]['landing_z'],
                physics_outputs[i]['entry_angle'],
                player_ids[i]
            )
            fold_preds.append(pred)

        # Compute fold MSE
        fold_mse = 0
        for pred, i in zip(fold_preds, val_idx):
            fold_mse += (pred['angle'] - actual_targets[i]['angle'])**2
            fold_mse += (pred['depth'] - actual_targets[i]['depth'])**2
            fold_mse += (pred['left_right'] - actual_targets[i]['left_right'])**2
        fold_mse /= (3 * len(val_idx))
        cv_scores.append(fold_mse)

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    if verbose:
        print(f"   CV MSE: {cv_mean:.6f} +/- {cv_std:.6f}")
        print(f"   Per-fold: {[f'{s:.4f}' for s in cv_scores]}")

    # ==================== TEST PREDICTIONS ====================
    if verbose:
        print("\n5. Generating test predictions...")

    test_predictions = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        player_id = metadata['participant_id']
        test_ids.append(metadata['id'])

        # Scale calibration
        scale_factor = calibrate_scale_factor(timeseries, keypoint_idx)

        # Extract release parameters
        release_params = extract_all_release_params(
            timeseries, keypoint_idx, player_id, scale_factor
        )

        # Simulate shot
        landing, entry_angle, _ = simulator.simulate_shot(
            release_params['position'],
            release_params['velocity'],
            release_params['backspin']
        )

        # Map to predictions
        pred = mapper.predict(
            landing[0] if landing is not None else None,
            landing[1] if landing is not None else None,
            entry_angle,
            player_id
        )

        # Apply corrections
        pred = apply_corrections(pred, corrections)
        test_predictions.append(pred)

    # Convert to arrays
    predictions = np.array([
        [p['angle'], p['depth'], p['left_right']]
        for p in test_predictions
    ])

    if verbose:
        print(f"   Test predictions: {len(predictions)}")
        print(f"   angle_std: {np.std(predictions[:, 0]):.4f}")
        print(f"   depth_mean: {np.mean(predictions[:, 1]):.4f}")
        print(f"   left_right_std: {np.std(predictions[:, 2]):.4f}")

    # ==================== SAVE SUBMISSION ====================
    if verbose:
        print("\n6. Saving submission...")

    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions[:, 0],
        'scaled_depth': predictions[:, 1],
        'scaled_left_right': predictions[:, 2]
    })

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)

    print(f"\n   Saved: {output_file}")
    print(f"   Submission #{next_num}")

    # ==================== COMPARISON WITH SUB 219 ====================
    if verbose:
        print("\n7. Comparison with Sub 219...")

    sub219_path = SUBMISSION_DIR / "submission_219.csv"
    if sub219_path.exists():
        sub219 = pd.read_csv(sub219_path)
        corr_angle = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
        corr_depth = np.corrcoef(predictions[:, 1], sub219['scaled_depth'].values)[0, 1]
        corr_lr = np.corrcoef(predictions[:, 2], sub219['scaled_left_right'].values)[0, 1]

        print(f"   Correlation with Sub 219:")
        print(f"     angle: {corr_angle:.4f}")
        print(f"     depth: {corr_depth:.4f}")
        print(f"     left_right: {corr_lr:.4f}")

        # Create blend submissions
        print("\n8. Creating blend submissions...")

        for blend_weight in [0.1, 0.2, 0.3]:
            blend = submission.copy()
            blend['scaled_angle'] = blend_weight * predictions[:, 0] + (1 - blend_weight) * sub219['scaled_angle'].values
            blend['scaled_depth'] = blend_weight * predictions[:, 1] + (1 - blend_weight) * sub219['scaled_depth'].values
            blend['scaled_left_right'] = blend_weight * predictions[:, 2] + (1 - blend_weight) * sub219['scaled_left_right'].values

            # Calibrate depth mean
            blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055

            next_num += 1
            blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
            blend.to_csv(blend_file, index=False)
            print(f"   {int(blend_weight*100)}% physics + {int((1-blend_weight)*100)}% Sub219: {blend_file}")

    # ==================== SUMMARY ====================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Training MSE: {train_mse:.6f}")
    print(f"CV MSE: {cv_mean:.6f} +/- {cv_std:.6f}")
    print(f"Successful simulations: {n_success}/{len(training_data)}")
    print(f"Target: < 0.007 MSE")
    print("=" * 80)

    return {
        'train_mse': train_mse,
        'cv_mse': cv_mean,
        'cv_std': cv_std,
        'n_success': n_success,
        'submission_file': output_file
    }


if __name__ == "__main__":
    results = run_physics_pipeline(verbose=True)

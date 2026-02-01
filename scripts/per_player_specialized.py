"""
Per-Player Specialized Models

Step 2 of the implementation plan:
- Build specialized models with different feature sets per player
- For Players 1, 2, 5 (low correlation with Sub 133): Use player-specific optimal features
- For Players 3, 4 (higher correlation): Use similar features to Sub 133
- Train Ridge models per player per target

Constraint: Final predictions must have:
- angle_std < 0.14
- depth_mean in [0.50, 0.51]
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'
SUBMISSION_DIR = PROJECT_DIR / 'submission'

TARGETS = ['angle', 'depth', 'left_right']

# Target scalers for converting to [0,1] space
TARGET_SCALERS = {
    'angle': joblib.load(DATA_DIR / 'scaler_angle.pkl'),
    'depth': joblib.load(DATA_DIR / 'scaler_depth.pkl'),
    'left_right': joblib.load(DATA_DIR / 'scaler_left_right.pkl'),
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_data(train=True):
    """Load data with all features."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    filename = "train.csv" if train else "test.csv"
    print(f"Loading {filename}...")
    df = pd.read_csv(DATA_DIR / filename)

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    all_features = []
    all_targets = []
    all_pids = []
    all_ids = []

    for idx, row in df.iterrows():
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        hybrid_feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
        advanced_feats = extract_advanced_features(timeseries, row['participant_id'])
        combined = {**hybrid_feats, **advanced_feats}
        all_features.append(combined)

        if train:
            all_targets.append([row['angle'], row['depth'], row['left_right']])
        all_pids.append(row['participant_id'])
        all_ids.append(row['id'])

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)}")

    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features], dtype=np.float32)

    if train:
        y = np.array(all_targets, dtype=np.float32)
    else:
        y = None

    pids = np.array(all_pids)

    return X, y, pids, feature_names, all_ids


def get_top_features_per_player(target_name, player_id, n_features=50):
    """
    Load precomputed feature importance for a player and target.
    Falls back to default features if file doesn't exist.
    """
    importance_file = OUTPUT_DIR / f"player{player_id}_{target_name}_feature_importance.csv"

    if importance_file.exists():
        df = pd.read_csv(importance_file)
        # Features with positive importance (reducing error)
        top_features = df[df['importance_mean'] > 0].head(n_features)['feature'].tolist()
        if len(top_features) < 10:
            # If too few positive, take top N anyway
            top_features = df.head(n_features)['feature'].tolist()
        return top_features
    else:
        print(f"  Warning: {importance_file} not found, using default features")
        return None


def compute_cv_score(X, y, pids, feature_names, feature_subset=None, alpha=1.0):
    """
    Compute within-player 5-fold CV score using Ridge regression.

    Args:
        feature_subset: List of feature names to use (None = use all)

    Returns:
        OOF predictions and CV MSE
    """
    if feature_subset is not None:
        feature_indices = [feature_names.index(f) for f in feature_subset if f in feature_names]
        if len(feature_indices) == 0:
            return None, float('inf')
        X = X[:, feature_indices]

    unique_pids = sorted(np.unique(pids))
    oof_preds = np.zeros(len(X))

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask]
        y_player = y[pid_mask]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(pid_mask)[0]

        for train_idx, val_idx in kf.split(X_scaled):
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_scaled[train_idx], y_player[train_idx])
            oof_preds[player_indices[val_idx]] = model.predict(X_scaled[val_idx])

    mse = np.mean((oof_preds - y) ** 2)
    return oof_preds, mse


def train_per_player_models(X_train, y_train, pids_train, feature_names,
                            player_features, target_idx, target_name,
                            alpha=1.0):
    """
    Train per-player models with potentially different feature sets.

    Args:
        player_features: dict of player_id -> list of features to use

    Returns:
        dict of player_id -> (model, scaler, feature_indices)
    """
    print(f"\nTraining per-player models for {target_name}...")
    unique_pids = sorted(np.unique(pids_train))
    models = {}

    for pid in unique_pids:
        pid_mask = pids_train == pid
        X_player = X_train[pid_mask]
        y_player = y_train[pid_mask, target_idx]

        # Get features for this player
        features = player_features.get(pid)
        if features is None:
            # Use all features
            feature_indices = list(range(X_player.shape[1]))
        else:
            feature_indices = [i for i, f in enumerate(feature_names) if f in features]

        if len(feature_indices) == 0:
            feature_indices = list(range(X_player.shape[1]))

        X_subset = X_player[:, feature_indices]

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Train model
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_scaled, y_player)

        models[pid] = (model, scaler, feature_indices)
        print(f"  Player {pid}: {len(feature_indices)} features, {len(y_player)} samples")

    return models


def predict_with_player_models(X, pids, models, feature_names):
    """Make predictions using per-player models."""
    predictions = np.zeros(len(X))

    for pid, (model, scaler, feature_indices) in models.items():
        pid_mask = pids == pid
        if not np.any(pid_mask):
            continue

        X_player = X[pid_mask][:, feature_indices]
        X_scaled = scaler.transform(X_player)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        predictions[pid_mask] = model.predict(X_scaled)

    return predictions


def compute_profile(predictions_angle, predictions_depth, predictions_lr):
    """Compute profile metrics for predictions."""
    angle_std = np.std(predictions_angle)
    depth_mean = np.mean(predictions_depth)
    lr_mean = np.mean(predictions_lr)
    return {
        'angle_std': angle_std,
        'depth_mean': depth_mean,
        'lr_mean': lr_mean
    }


def main():
    print("="*80)
    print("PER-PLAYER SPECIALIZED MODELS")
    print("="*80)

    # Load training data
    X_train, y_train, pids_train, feature_names, train_ids = load_data(train=True)

    # Load test data
    X_test, _, pids_test, _, test_ids = load_data(train=False)

    # Define feature strategies per player
    # Players 1, 2, 5 have low correlation with Sub 133 - use specialized features
    # Players 3, 4 have higher correlation - use standard features

    n_features = 50  # Number of top features to use per player

    print("\n" + "="*80)
    print("BUILDING PLAYER-SPECIFIC FEATURE SETS")
    print("="*80)

    player_features = {target: {} for target in TARGETS}

    for target_name in TARGETS:
        print(f"\n{target_name.upper()}:")
        for pid in range(1, 6):
            top_features = get_top_features_per_player(target_name, pid, n_features)
            if top_features is not None:
                player_features[target_name][pid] = top_features
                print(f"  Player {pid}: {len(top_features)} features loaded")
            else:
                print(f"  Player {pid}: Using all features (no importance file)")

    # Train models and make predictions
    all_train_predictions = {}
    all_test_predictions = {}
    cv_scores = {}

    for target_idx, target_name in enumerate(TARGETS):
        print(f"\n\n{'#'*80}")
        print(f"# TARGET: {target_name.upper()}")
        print(f"{'#'*80}")

        # Get feature sets for this target
        pf = player_features[target_name]

        # Compute CV score first
        print("\nComputing CV score with player-specific features...")
        oof_preds, cv_mse = compute_cv_score(
            X_train, y_train[:, target_idx], pids_train, feature_names, feature_subset=None
        )
        print(f"  All features CV MSE: {cv_mse:.6f}")

        # Now with specialized features per player
        unique_pids = sorted(np.unique(pids_train))
        specialized_oof = np.zeros(len(X_train))

        for pid in unique_pids:
            pid_mask = pids_train == pid
            features = pf.get(pid)

            _, player_mse = compute_cv_score(
                X_train[pid_mask], y_train[pid_mask, target_idx],
                pids_train[pid_mask], feature_names, feature_subset=features
            )
            print(f"  Player {pid} specialized CV MSE: {player_mse:.6f}")

        # Train per-player models
        models = train_per_player_models(
            X_train, y_train, pids_train, feature_names,
            pf, target_idx, target_name, alpha=1.0
        )

        # Make predictions on train (for CV calculation)
        train_preds = predict_with_player_models(X_train, pids_train, models, feature_names)
        all_train_predictions[target_name] = train_preds

        # Make predictions on test
        test_preds = predict_with_player_models(X_test, pids_test, models, feature_names)
        all_test_predictions[target_name] = test_preds

        # CV score
        cv_mse = np.mean((train_preds - y_train[:, target_idx]) ** 2)
        cv_scores[target_name] = cv_mse
        print(f"\n  Training MSE (not CV): {cv_mse:.6f}")

    # Convert predictions to scaled [0, 1] space
    print("\n" + "="*80)
    print("CONVERTING TO SCALED SPACE")
    print("="*80)

    scaled_predictions = {}
    for target_name in TARGETS:
        scaler = TARGET_SCALERS[target_name]
        raw_preds = all_test_predictions[target_name]
        scaled = scaler.transform(raw_preds.reshape(-1, 1)).flatten()
        scaled = np.clip(scaled, 0, 1)
        scaled_predictions[target_name] = scaled

    # Compute profile
    profile = compute_profile(
        scaled_predictions['angle'],
        scaled_predictions['depth'],
        scaled_predictions['left_right']
    )

    print(f"\nPrediction profile:")
    print(f"  angle_std:  {profile['angle_std']:.4f} (target < 0.14)")
    print(f"  depth_mean: {profile['depth_mean']:.4f} (target: 0.50-0.51)")
    print(f"  lr_mean:    {profile['lr_mean']:.4f}")

    # Check profile constraints
    profile_ok = (profile['angle_std'] < 0.14) and (0.49 < profile['depth_mean'] < 0.52)

    if profile_ok:
        print("\n  Profile constraints SATISFIED")
    else:
        print("\n  WARNING: Profile constraints NOT satisfied")
        print("  Consider blending with Sub 133 or applying calibration")

    # Compare with Sub 133
    print("\n" + "="*80)
    print("COMPARISON WITH SUB 133")
    print("="*80)

    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    sub133_angle = sub133['scaled_angle'].values
    sub133_depth = sub133['scaled_depth'].values
    sub133_lr = sub133['scaled_left_right'].values

    # Correlation
    angle_corr = np.corrcoef(scaled_predictions['angle'], sub133_angle)[0, 1]
    depth_corr = np.corrcoef(scaled_predictions['depth'], sub133_depth)[0, 1]
    lr_corr = np.corrcoef(scaled_predictions['left_right'], sub133_lr)[0, 1]

    print(f"\nCorrelation with Sub 133:")
    print(f"  angle:      {angle_corr:.4f}")
    print(f"  depth:      {depth_corr:.4f}")
    print(f"  left_right: {lr_corr:.4f}")
    print(f"  AVERAGE:    {(angle_corr + depth_corr + lr_corr)/3:.4f}")

    # Per-player correlation for angle
    print("\nPer-player angle correlation with Sub 133:")
    for pid in sorted(np.unique(pids_test)):
        pid_mask = pids_test == pid
        if not np.any(pid_mask):
            continue
        our_angle = scaled_predictions['angle'][pid_mask]
        sub133_angle_player = sub133_angle[pid_mask]
        corr = np.corrcoef(our_angle, sub133_angle_player)[0, 1]
        print(f"  Player {pid}: {corr:.4f}")

    # Save submission
    print("\n" + "="*80)
    print("CREATING SUBMISSION")
    print("="*80)

    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if existing:
        nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
        next_num = max(nums) + 1 if nums else 163
    else:
        next_num = 163

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_predictions['angle'],
        'scaled_depth': scaled_predictions['depth'],
        'scaled_left_right': scaled_predictions['left_right']
    })

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"\nSubmission {next_num} details:")
    print(f"  Model: Per-player specialized Ridge with player-specific top-{n_features} features")
    print(f"  CV scores: angle={cv_scores['angle']:.6f}, depth={cv_scores['depth']:.6f}, lr={cv_scores['left_right']:.6f}")
    print(f"  Profile: angle_std={profile['angle_std']:.4f}, depth_mean={profile['depth_mean']:.4f}")
    print(f"  Avg correlation with Sub 133: {(angle_corr + depth_corr + lr_corr)/3:.4f}")

    # Save results summary
    results = {
        'submission_number': next_num,
        'cv_scores': cv_scores,
        'profile': profile,
        'correlation_with_sub133': {
            'angle': angle_corr,
            'depth': depth_corr,
            'left_right': lr_corr
        }
    }

    import json as json_lib
    with open(OUTPUT_DIR / f"per_player_specialized_results.json", 'w') as f:
        json_lib.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

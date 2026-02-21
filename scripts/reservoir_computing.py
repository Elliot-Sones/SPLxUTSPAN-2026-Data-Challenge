"""
Reservoir Computing (Echo State Networks) for Small Data

ESNs are ideal for small datasets because:
1. Only the output layer is trained (linear regression)
2. Captures complex temporal dynamics without overfitting
3. No gradient-based training of recurrent connections

Also implements MiniRocket for fast feature extraction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from scipy.signal import butter, filtfilt
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
SUBMISSION_DIR = PROJECT_DIR / "submission"

from data_loader import iterate_shots, get_keypoint_columns

# Try to import optional dependencies
try:
    from sktime.transformations.panel.rocket import MiniRocket
    MINIROCKET_AVAILABLE = True
except ImportError:
    MINIROCKET_AVAILABLE = False
    print("MiniRocket not available. Install: pip install sktime")


class EchoStateNetwork:
    """
    Echo State Network (Reservoir Computing)

    Only the output weights are trained. The reservoir provides
    a rich nonlinear expansion of the input.
    """

    def __init__(self, n_reservoir=200, spectral_radius=0.9,
                 sparsity=0.1, noise=0.001, random_state=42):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.random_state = random_state

        self.W_in = None  # Input weights
        self.W = None     # Reservoir weights
        self.W_out = None # Output weights (trained)

    def _init_reservoir(self, n_inputs):
        """Initialize reservoir weights."""
        np.random.seed(self.random_state)

        # Input weights: dense, small
        self.W_in = np.random.uniform(-0.5, 0.5, (self.n_reservoir, n_inputs))

        # Reservoir weights: sparse
        W = np.random.uniform(-1, 1, (self.n_reservoir, self.n_reservoir))

        # Apply sparsity
        mask = np.random.random(W.shape) < self.sparsity
        W = W * mask

        # Scale to spectral radius
        eigenvalues = np.linalg.eigvals(W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            W = W * (self.spectral_radius / max_eigenvalue)

        self.W = W

    def _compute_reservoir_states(self, X):
        """
        Compute reservoir states for input sequence X.
        X: (n_timesteps, n_features)
        Returns: (n_timesteps, n_reservoir)
        """
        n_timesteps = X.shape[0]
        states = np.zeros((n_timesteps, self.n_reservoir))

        state = np.zeros(self.n_reservoir)

        for t in range(n_timesteps):
            # Add noise for regularization
            noise = self.noise * np.random.randn(self.n_reservoir)

            # Update state: tanh(W_in * x + W * state + noise)
            pre_activation = np.dot(self.W_in, X[t]) + np.dot(self.W, state) + noise
            state = np.tanh(pre_activation)
            states[t] = state

        return states

    def fit(self, X_sequences, y, alpha=1.0):
        """
        Fit ESN on sequences.
        X_sequences: list of (n_timesteps, n_features) arrays
        y: (n_samples,) target values
        """
        # Initialize reservoir if needed
        n_inputs = X_sequences[0].shape[1]
        if self.W_in is None:
            self._init_reservoir(n_inputs)

        # Compute features for all sequences
        features = []
        for seq in X_sequences:
            states = self._compute_reservoir_states(seq)
            # Use mean and final state as features
            feat = np.concatenate([
                states.mean(axis=0),
                states[-1],
                states.std(axis=0)
            ])
            features.append(feat)

        features = np.array(features)

        # Train output weights with Ridge regression
        self.ridge = Ridge(alpha=alpha)
        self.ridge.fit(features, y)

        self.W_out = self.ridge.coef_

    def predict(self, X_sequences):
        """Predict for new sequences."""
        features = []
        for seq in X_sequences:
            states = self._compute_reservoir_states(seq)
            feat = np.concatenate([
                states.mean(axis=0),
                states[-1],
                states.std(axis=0)
            ])
            features.append(feat)

        features = np.array(features)
        return self.ridge.predict(features)


def butterworth_filter(data, cutoff=8, fs=60, order=4):
    """Zero-lag Butterworth lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    result = np.zeros_like(data)
    for i in range(data.shape[1]):
        col = data[:, i]
        if np.any(np.isnan(col)):
            mask = ~np.isnan(col)
            if mask.sum() < 10:
                result[:, i] = col
                continue
            col = np.interp(np.arange(len(col)), np.where(mask)[0], col[mask])
        try:
            result[:, i] = filtfilt(b, a, col)
        except:
            result[:, i] = col

    return result


def get_keypoint_indices(keypoint_cols):
    keypoints = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoints.append(col[:-2])
    return {kp: i for i, kp in enumerate(keypoints)}


def extract_sequence(timeseries, kp_idx, start=80, end=160):
    """
    Extract relevant joint trajectories as sequence.
    Returns: (n_frames, n_features)
    """
    key_joints = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'mid_hip', 'right_knee', 'right_ankle'
    ]

    features = []
    for joint in key_joints:
        if joint in kp_idx:
            i = kp_idx[joint]
            # Get z coordinate and its velocity
            z = timeseries[start:end, i*3 + 2]
            vz = np.gradient(z) * 60
            features.append(z)
            features.append(vz)

    if len(features) == 0:
        return None

    sequence = np.column_stack(features)

    # Apply Butterworth filter
    sequence = butterworth_filter(sequence, cutoff=8)

    # Handle NaN
    sequence = np.nan_to_num(sequence, nan=0.0)

    return sequence


def get_next_submission_number():
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split('_')[1])
            numbers.append(num)
        except:
            pass
    return max(numbers) + 1 if numbers else 1


def main():
    print("=" * 80)
    print("RESERVOIR COMPUTING (ECHO STATE NETWORKS)")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    kp_idx = get_keypoint_indices(keypoint_cols)

    # Extract sequences for training
    print("\nExtracting training sequences...")
    train_sequences = []
    train_targets = {'angle': [], 'depth': [], 'left_right': []}
    train_players = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        seq = extract_sequence(timeseries, kp_idx)
        if seq is None:
            continue

        train_sequences.append(seq)
        train_players.append(metadata['participant_id'])

        for target in train_targets:
            train_targets[target].append(metadata[target])

        if i % 50 == 0:
            print(f"  Processed {i+1} shots...")

    print(f"  Training sequences: {len(train_sequences)}")
    print(f"  Sequence shape: {train_sequences[0].shape}")

    # Scale targets to 0-1
    target_stats = {}
    y_train = {}
    for target in train_targets:
        vals = np.array(train_targets[target])
        target_stats[target] = {'min': vals.min(), 'max': vals.max()}
        y_train[target] = (vals - vals.min()) / (vals.max() - vals.min())

    groups = np.array(train_players)

    # Extract test sequences
    print("\nExtracting test sequences...")
    test_sequences = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        seq = extract_sequence(timeseries, kp_idx)
        if seq is None:
            # Use zeros as fallback
            seq = np.zeros((80, train_sequences[0].shape[1]))

        test_sequences.append(seq)
        test_ids.append(metadata['id'])

    print(f"  Test sequences: {len(test_sequences)}")

    # =========================================================================
    # EXPERIMENT: Echo State Network
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING ECHO STATE NETWORKS")
    print("=" * 80)

    gkf = GroupKFold(n_splits=5)
    targets = ['angle', 'depth', 'left_right']

    esn_results = {}
    esn_models = {}

    for target in targets:
        y = y_train[target]
        mses = []

        print(f"\n{target.upper()}:")

        for fold, (train_idx, val_idx) in enumerate(gkf.split(train_sequences, y, groups)):
            X_tr = [train_sequences[i] for i in train_idx]
            X_val = [train_sequences[i] for i in val_idx]
            y_tr = y[train_idx]
            y_val = y[val_idx]

            esn = EchoStateNetwork(
                n_reservoir=300,
                spectral_radius=0.95,
                sparsity=0.1,
                noise=0.001,
                random_state=42 + fold
            )

            esn.fit(X_tr, y_tr, alpha=10.0)
            pred = esn.predict(X_val)
            pred = np.clip(pred, 0, 1)

            mse = mean_squared_error(y_val, pred)
            mses.append(mse)

        esn_results[target] = np.mean(mses)
        print(f"  CV MSE: {np.mean(mses):.6f} +/- {np.std(mses):.6f}")

        # Train final model
        esn_final = EchoStateNetwork(
            n_reservoir=300,
            spectral_radius=0.95,
            sparsity=0.1,
            noise=0.001,
            random_state=42
        )
        esn_final.fit(train_sequences, y, alpha=10.0)
        esn_models[target] = esn_final

    print(f"\nTotal ESN CV MSE: {sum(esn_results.values())/3:.6f}")

    # Make predictions
    print("\nMaking ESN predictions...")
    predictions = {}

    for target in targets:
        pred = esn_models[target].predict(test_sequences)
        pred = np.clip(pred, 0, 1)
        predictions[target] = pred

    # Create ESN submission
    sub_num = get_next_submission_number()

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions['angle'],
        'scaled_depth': predictions['depth'],
        'scaled_left_right': predictions['left_right']
    })

    # Calibrate depth
    submission['scaled_depth'] = submission['scaled_depth'] - submission['scaled_depth'].mean() + 0.5055
    submission['scaled_depth'] = np.clip(submission['scaled_depth'], 0, 1)

    print(f"\nESN Profile:")
    print(f"  angle_std: {submission['scaled_angle'].std():.6f}")
    print(f"  depth_mean: {submission['scaled_depth'].mean():.6f}")

    esn_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    submission.to_csv(esn_path, index=False)
    print(f"\nESN submission saved: {esn_path}")
    sub_num += 1

    # =========================================================================
    # EXPERIMENT: ESN + XGBoost Ensemble (on engineered features)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ESN + XGBOOST ENSEMBLE")
    print("=" * 80)

    # Load Sub 219 as XGBoost proxy
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    sub219_indexed = sub219.set_index('id')

    # Blend ESN with Sub 219
    for weight in [0.1, 0.2, 0.3]:
        blended = sub219.copy()

        for col, target in [('scaled_angle', 'angle'),
                            ('scaled_depth', 'depth'),
                            ('scaled_left_right', 'left_right')]:
            esn_pred = predictions[target]
            sub219_pred = sub219_indexed.loc[test_ids, col].values
            blended_pred = weight * esn_pred + (1 - weight) * sub219_pred
            blended[col] = blended_pred

        # Calibrate
        blended['scaled_depth'] = blended['scaled_depth'] - blended['scaled_depth'].mean() + 0.5055
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            blended[col] = blended[col].clip(0, 1)

        blend_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blended.to_csv(blend_path, index=False)

        print(f"  Sub {sub_num}: {int(weight*100)}% ESN + {int((1-weight)*100)}% Sub219, "
              f"angle_std={blended['scaled_angle'].std():.6f}")
        sub_num += 1

    # =========================================================================
    # MINIROCKET (if available)
    # =========================================================================
    if MINIROCKET_AVAILABLE:
        print("\n" + "=" * 80)
        print("MINIROCKET FEATURES")
        print("=" * 80)

        # Prepare data for MiniRocket (needs 3D array)
        X_train_3d = np.array(train_sequences)  # (n_samples, n_timesteps, n_features)

        # MiniRocket transform
        minirocket = MiniRocket(random_state=42)
        X_train_rocket = minirocket.fit_transform(X_train_3d)

        print(f"  MiniRocket features: {X_train_rocket.shape[1]}")

        # CV with MiniRocket features
        for target in targets:
            y = y_train[target]
            mses = []

            for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_rocket, y, groups)):
                X_tr, X_val = X_train_rocket[train_idx], X_train_rocket[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model = Ridge(alpha=10.0)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                mse = mean_squared_error(y_val, pred)
                mses.append(mse)

            print(f"  {target}: CV MSE = {np.mean(mses):.6f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Echo State Networks provide temporal feature extraction")
    print("without the overfitting risk of trained RNNs.")


if __name__ == "__main__":
    main()

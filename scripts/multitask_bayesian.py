"""
Two Approaches:
1. Multi-task Learning - Share information across players aggressively
2. Bayesian Models with Strong Physics Priors
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoints = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoints:
            keypoints.append(name)
    return {name: i for i, name in enumerate(keypoints)}


def extract_features(timeseries, keypoint_idx):
    """Extract features including physics-based ones."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    key_frames = [100, 120, 140, 150, 153, 156, 160]

    for f in key_frames:
        # Basic positions
        for joint in ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip']:
            pos = get_joint(joint, f)
            features[f'{joint}_x_f{f}'] = pos[0]
            features[f'{joint}_y_f{f}'] = pos[1]
            features[f'{joint}_z_f{f}'] = pos[2]

        # === PHYSICS-BASED FEATURES ===

        # Release height (higher = better angle typically)
        wrist = get_joint('right_wrist', f)
        features[f'release_height_f{f}'] = wrist[2]

        # Shoulder-wrist vertical alignment
        shoulder = get_joint('right_shoulder', f)
        features[f'vertical_alignment_f{f}'] = wrist[2] - shoulder[2]

        # Elbow angle (optimal ~90-100 degrees at release)
        elbow = get_joint('right_elbow', f)
        if np.any(wrist) and np.any(elbow) and np.any(shoulder):
            v1 = shoulder - elbow
            v2 = wrist - elbow
            cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            features[f'elbow_angle_f{f}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))

        # Body twist (alignment toward basket)
        ls = get_joint('left_shoulder', f)
        rs = get_joint('right_shoulder', f)
        lh = get_joint('left_hip', f)
        rh = get_joint('right_hip', f)

        if np.any(ls) and np.any(rs) and np.any(lh) and np.any(rh):
            shoulder_vec = rs - ls
            hip_vec = rh - lh
            cos_ang = np.dot(shoulder_vec, hip_vec) / (np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec) + 1e-8)
            features[f'body_twist_f{f}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))

            # Left-right alignment (shoulder midpoint)
            shoulder_mid_x = (ls[0] + rs[0]) / 2
            features[f'shoulder_center_x_f{f}'] = shoulder_mid_x

    # === VELOCITY FEATURES (physics: momentum transfer) ===
    for f_start, f_end in [(148, 153), (150, 155), (153, 158)]:
        wrist_s = get_joint('right_wrist', f_start)
        wrist_e = get_joint('right_wrist', f_end)
        if np.any(wrist_s) and np.any(wrist_e):
            vel = wrist_e - wrist_s
            speed = np.linalg.norm(vel)
            features[f'wrist_speed_{f_start}_{f_end}'] = speed
            features[f'wrist_vz_{f_start}_{f_end}'] = vel[2]  # Upward velocity
            features[f'wrist_vx_{f_start}_{f_end}'] = vel[0]  # Left-right velocity

            # Launch angle approximation
            if speed > 0:
                launch_angle = np.degrees(np.arcsin(vel[2] / (speed + 1e-8)))
                features[f'launch_angle_{f_start}_{f_end}'] = launch_angle

    # === FOLLOW-THROUGH FEATURES ===
    wrist_release = get_joint('right_wrist', 153)
    wrist_follow = get_joint('right_wrist', 170)
    if np.any(wrist_release) and np.any(wrist_follow):
        follow_through = wrist_follow - wrist_release
        features['follow_through_z'] = follow_through[2]
        features['follow_through_mag'] = np.linalg.norm(follow_through)

    return features


# =============================================================================
# APPROACH 1: MULTI-TASK LEARNING
# =============================================================================

class MultiTaskNet(nn.Module):
    """
    Multi-task network that shares a backbone across all players,
    with player-specific heads for fine-tuning.
    """
    def __init__(self, input_dim, n_players=5, hidden_dim=64):
        super().__init__()

        # Shared backbone - learns universal shooting physics
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Player-specific adaptation layers
        self.player_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 3)  # 3 targets: angle, depth, left_right
            ) for _ in range(n_players)
        ])

        # Global head (for players not in training)
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x, player_ids=None):
        # Shared representation
        shared_repr = self.shared(x)

        if player_ids is None:
            return self.global_head(shared_repr)

        # Player-specific predictions
        outputs = torch.zeros(x.size(0), 3).to(x.device)

        for i in range(x.size(0)):
            pid = int(player_ids[i].item()) - 1  # Convert to 0-indexed
            if 0 <= pid < len(self.player_adapters):
                outputs[i] = self.player_adapters[pid](shared_repr[i:i+1]).squeeze()
            else:
                outputs[i] = self.global_head(shared_repr[i:i+1]).squeeze()

        return outputs


def train_multitask(X_train, y_train, player_ids_train, epochs=150):
    """Train multi-task model with shared backbone."""

    dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.LongTensor(player_ids_train)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MultiTaskNet(X_train.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_state = model.state_dict().copy()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for bx, by, bp in loader:
            bx, by, bp = bx.to(DEVICE), by.to(DEVICE), bp.to(DEVICE)

            pred = model(bx, bp)
            loss = criterion(pred, by)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict().copy()

        if epoch % 30 == 0:
            print(f"    Epoch {epoch}: loss={avg_loss:.4f}")

    model.load_state_dict(best_state)
    return model


# =============================================================================
# APPROACH 2: BAYESIAN WITH PHYSICS PRIORS
# =============================================================================

def create_physics_prior_features(X, feature_names):
    """
    Create features that encode physics priors.

    Physics beliefs:
    1. Higher release height -> better arc -> better angle
    2. More vertical velocity -> higher arc
    3. Body alignment -> left_right accuracy
    4. Follow-through -> consistency
    5. Elbow angle near 90-100 degrees is optimal
    """
    n_samples = X.shape[0]
    physics_features = np.zeros((n_samples, 10))

    # Find relevant feature indices
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    for i in range(n_samples):
        # 1. Release height prior (higher is generally better for angle)
        height_feats = [name_to_idx.get(f'release_height_f{f}', -1) for f in [153, 156]]
        height_feats = [idx for idx in height_feats if idx >= 0]
        if height_feats:
            physics_features[i, 0] = np.mean([X[i, idx] for idx in height_feats])

        # 2. Vertical velocity prior (more upward = better arc)
        vz_feats = [name_to_idx.get(f'wrist_vz_{s}_{e}', -1) for s, e in [(148, 153), (150, 155)]]
        vz_feats = [idx for idx in vz_feats if idx >= 0]
        if vz_feats:
            physics_features[i, 1] = np.mean([X[i, idx] for idx in vz_feats])

        # 3. Launch angle prior
        launch_feats = [name_to_idx.get(f'launch_angle_{s}_{e}', -1) for s, e in [(148, 153), (150, 155)]]
        launch_feats = [idx for idx in launch_feats if idx >= 0]
        if launch_feats:
            physics_features[i, 2] = np.mean([X[i, idx] for idx in launch_feats])

        # 4. Body alignment prior (twist affects left-right)
        twist_feats = [name_to_idx.get(f'body_twist_f{f}', -1) for f in [150, 153]]
        twist_feats = [idx for idx in twist_feats if idx >= 0]
        if twist_feats:
            physics_features[i, 3] = np.mean([X[i, idx] for idx in twist_feats])

        # 5. Shoulder center prior (affects left-right)
        center_feats = [name_to_idx.get(f'shoulder_center_x_f{f}', -1) for f in [153, 156]]
        center_feats = [idx for idx in center_feats if idx >= 0]
        if center_feats:
            physics_features[i, 4] = np.mean([X[i, idx] for idx in center_feats])

        # 6. Elbow angle optimality (deviation from optimal ~95 degrees)
        elbow_feats = [name_to_idx.get(f'elbow_angle_f{f}', -1) for f in [153, 156]]
        elbow_feats = [idx for idx in elbow_feats if idx >= 0]
        if elbow_feats:
            elbow_angles = [X[i, idx] for idx in elbow_feats]
            # Optimal is around 95 degrees, penalize deviation
            physics_features[i, 5] = -np.mean([abs(a - 95) for a in elbow_angles])

        # 7. Follow-through magnitude (more = better control)
        if 'follow_through_mag' in name_to_idx:
            physics_features[i, 6] = X[i, name_to_idx['follow_through_mag']]

        # 8. Follow-through vertical (upward = good form)
        if 'follow_through_z' in name_to_idx:
            physics_features[i, 7] = X[i, name_to_idx['follow_through_z']]

        # 9. Speed consistency (release speed)
        speed_feats = [name_to_idx.get(f'wrist_speed_{s}_{e}', -1) for s, e in [(148, 153), (150, 155)]]
        speed_feats = [idx for idx in speed_feats if idx >= 0]
        if speed_feats:
            physics_features[i, 8] = np.mean([X[i, idx] for idx in speed_feats])

        # 10. Vertical alignment (wrist over shoulder)
        align_feats = [name_to_idx.get(f'vertical_alignment_f{f}', -1) for f in [153, 156]]
        align_feats = [idx for idx in align_feats if idx >= 0]
        if align_feats:
            physics_features[i, 9] = np.mean([X[i, idx] for idx in align_feats])

    return np.nan_to_num(physics_features, nan=0.0)


def train_bayesian_with_priors(X_train, y_train, X_test, feature_names, target_name):
    """
    Train Bayesian Ridge with physics-informed priors.

    The physics features get higher prior weight.
    """
    # Create physics prior features
    physics_train = create_physics_prior_features(X_train, feature_names)
    physics_test = create_physics_prior_features(X_test, feature_names)

    # Combine original features with physics features
    # Give physics features 3x weight by repeating
    X_train_aug = np.hstack([X_train, physics_train, physics_train, physics_train])
    X_test_aug = np.hstack([X_test, physics_test, physics_test, physics_test])

    # Bayesian Ridge with informative priors
    # alpha_1, alpha_2 control the prior on weights (higher = stronger regularization)
    # lambda_1, lambda_2 control the prior on noise

    if target_name == 'angle':
        # For angle: physics (height, launch angle) should matter most
        model = BayesianRidge(
            alpha_1=1e-6, alpha_2=1e-6,  # Weak prior on weights
            lambda_1=1e-6, lambda_2=1e-6,
            fit_intercept=True,
            compute_score=True
        )
    elif target_name == 'depth':
        # For depth: release speed, follow-through matter
        model = BayesianRidge(
            alpha_1=1e-5, alpha_2=1e-5,
            lambda_1=1e-6, lambda_2=1e-6,
            fit_intercept=True
        )
    else:  # left_right
        # For left_right: body alignment, shoulder center matter
        model = BayesianRidge(
            alpha_1=1e-5, alpha_2=1e-5,
            lambda_1=1e-6, lambda_2=1e-6,
            fit_intercept=True
        )

    model.fit(X_train_aug, y_train)

    # Predict with uncertainty
    pred_mean, pred_std = model.predict(X_test_aug, return_std=True)

    return pred_mean, pred_std


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("MULTI-TASK LEARNING & BAYESIAN WITH PHYSICS PRIORS")
    print("="*70)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Load data
    print("\nLoading data...")
    train_data = []
    test_data = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_features(timeseries, keypoint_idx)
        train_data.append({
            'features': features,
            'player': metadata['participant_id'],
            'targets': {
                'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
                'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
                'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
            }
        })

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_features(timeseries, keypoint_idx)
        test_data.append({
            'features': features,
            'id': metadata['id'],
            'player': metadata['participant_id']
        })

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Create feature matrices
    X_train = pd.DataFrame([d['features'] for d in train_data]).fillna(0)
    X_test = pd.DataFrame([d['features'] for d in test_data]).fillna(0)

    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    feature_names = list(common_cols)

    print(f"Features: {len(feature_names)}")

    y_train = np.array([[d['targets']['angle'], d['targets']['depth'], d['targets']['left_right']]
                        for d in train_data])
    player_ids_train = np.array([d['player'] for d in train_data])
    player_ids_test = np.array([d['player'] for d in test_data])
    test_ids = [d['id'] for d in test_data]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")

    # =========================================================================
    # APPROACH 1: MULTI-TASK LEARNING
    # =========================================================================
    print("\n" + "="*70)
    print("APPROACH 1: MULTI-TASK LEARNING")
    print("="*70)
    print("Training shared backbone with player-specific heads...")

    mt_model = train_multitask(X_train_scaled, y_train, player_ids_train, epochs=150)

    # Predict
    mt_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
        player_tensor = torch.LongTensor(player_ids_test).to(DEVICE)
        mt_preds = mt_model(X_test_tensor, player_tensor).cpu().numpy()

    # Calibrate
    mt_preds[:, 1] = mt_preds[:, 1] - np.mean(mt_preds[:, 1]) + 0.5055
    mt_preds = np.clip(mt_preds, 0, 1)

    mt_angle_std = np.std(mt_preds[:, 0])
    mt_corr = np.corrcoef(mt_preds[:, 0], sub219['scaled_angle'].values)[0, 1]

    print(f"\nMulti-task results:")
    print(f"  angle_std: {mt_angle_std:.6f}")
    print(f"  Correlation with Sub 219: {mt_corr:.4f}")

    # =========================================================================
    # APPROACH 2: BAYESIAN WITH PHYSICS PRIORS
    # =========================================================================
    print("\n" + "="*70)
    print("APPROACH 2: BAYESIAN WITH PHYSICS PRIORS")
    print("="*70)

    targets = ['angle', 'depth', 'left_right']
    bayes_preds = {}
    bayes_uncertainty = {}

    for i, target in enumerate(targets):
        print(f"\nTraining Bayesian model for {target}...")
        pred, uncertainty = train_bayesian_with_priors(
            X_train_scaled, y_train[:, i], X_test_scaled, feature_names, target
        )
        bayes_preds[target] = pred
        bayes_uncertainty[target] = uncertainty
        print(f"  Mean uncertainty: {np.mean(uncertainty):.4f}")

    # Calibrate
    bayes_preds['depth'] = bayes_preds['depth'] - np.mean(bayes_preds['depth']) + 0.5055
    for t in targets:
        bayes_preds[t] = np.clip(bayes_preds[t], 0, 1)

    bayes_angle_std = np.std(bayes_preds['angle'])
    bayes_corr = np.corrcoef(bayes_preds['angle'], sub219['scaled_angle'].values)[0, 1]

    print(f"\nBayesian with physics priors results:")
    print(f"  angle_std: {bayes_angle_std:.6f}")
    print(f"  Correlation with Sub 219: {bayes_corr:.4f}")

    # =========================================================================
    # APPROACH 3: UNCERTAINTY-WEIGHTED ENSEMBLE
    # =========================================================================
    print("\n" + "="*70)
    print("APPROACH 3: UNCERTAINTY-WEIGHTED ENSEMBLE")
    print("="*70)

    # Where Bayesian is uncertain, trust Sub 219 more
    ensemble_preds = {}
    for t in targets:
        # Normalize uncertainty to [0, 1]
        unc = bayes_uncertainty[t]
        unc_norm = (unc - unc.min()) / (unc.max() - unc.min() + 1e-8)

        # High uncertainty -> use Sub 219, low uncertainty -> use Bayesian
        confidence = 1 - unc_norm
        ensemble_preds[t] = confidence * bayes_preds[t] + (1 - confidence) * sub219[f'scaled_{t}'].values

    ensemble_preds['depth'] = ensemble_preds['depth'] - np.mean(ensemble_preds['depth']) + 0.5055
    for t in targets:
        ensemble_preds[t] = np.clip(ensemble_preds[t], 0, 1)

    ens_angle_std = np.std(ensemble_preds['angle'])
    ens_corr = np.corrcoef(ensemble_preds['angle'], sub219['scaled_angle'].values)[0, 1]

    print(f"Uncertainty-weighted ensemble results:")
    print(f"  angle_std: {ens_angle_std:.6f}")
    print(f"  Correlation with Sub 219: {ens_corr:.4f}")

    # =========================================================================
    # SAVE SUBMISSIONS
    # =========================================================================
    print("\n" + "="*70)
    print("SAVING SUBMISSIONS")
    print("="*70)

    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    results = [
        ('multitask', {'angle': mt_preds[:, 0], 'depth': mt_preds[:, 1], 'left_right': mt_preds[:, 2]},
         mt_angle_std, mt_corr),
        ('bayesian_physics', bayes_preds, bayes_angle_std, bayes_corr),
        ('uncertainty_ensemble', ensemble_preds, ens_angle_std, ens_corr),
    ]

    for name, preds, a_std, corr in results:
        sub = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': preds['angle'],
            'scaled_depth': preds['depth'],
            'scaled_left_right': preds['left_right']
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{next_num}.csv", index=False)
        print(f"Saved: submission_{next_num}.csv ({name}, angle_std={a_std:.6f}, corr={corr:.4f})")
        next_num += 1

    # Blends with Sub 219
    print("\nBlends with Sub 219:")
    for name, preds, _, _ in results:
        for w in [0.15, 0.25, 0.35]:
            blend = pd.DataFrame({
                'id': test_ids,
                'scaled_angle': w * preds['angle'] + (1-w) * sub219['scaled_angle'].values,
                'scaled_depth': w * preds['depth'] + (1-w) * sub219['scaled_depth'].values,
                'scaled_left_right': w * preds['left_right'] + (1-w) * sub219['scaled_left_right'].values
            })
            blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055
            blend.to_csv(SUBMISSION_DIR / f"submission_{next_num}.csv", index=False)
            print(f"Saved: submission_{next_num}.csv ({w:.0%} {name} + {1-w:.0%} Sub219, angle_std={blend['scaled_angle'].std():.6f})")
            next_num += 1

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Approach':<25} {'angle_std':<12} {'Corr w/Sub219':<15}")
    print("-"*55)
    for name, _, a_std, corr in results:
        print(f"{name:<25} {a_std:<12.6f} {corr:<15.4f}")


if __name__ == "__main__":
    main()

"""
Physics-Constrained Neural Network (Plan 3.1)

Network predicts release velocity. Loss includes physics consistency.
Physics equations for ball trajectory are known but not enforced.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, NUM_FRAMES
from data_loader import get_keypoint_columns
from feature_engineering import init_keypoint_mapping

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

# Physics constants
G = 32.174  # ft/s^2 (gravity)
RIM_HEIGHT = 10.0  # ft
RIM_DISTANCE = 23.75  # ft (NBA 3-point line)


def build_raw_data(train: bool = True):
    """Load raw timeseries and extract release features."""
    meta_df = load_metadata(train)
    n_shots = len(meta_df)

    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    features = []
    targets = []
    participant_ids = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train)):
        if i % 50 == 0:
            print(f"  Processing shot {i+1}/{n_shots}...")

        # Extract release-time features
        release_feats = extract_release_physics(timeseries)
        features.append(release_feats)
        participant_ids.append(metadata["participant_id"])

        if train:
            targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])

    X = pd.DataFrame(features).fillna(0).values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids


def extract_release_physics(timeseries: np.ndarray) -> dict:
    """Extract physics-relevant features at release."""
    from feature_engineering import KEYPOINT_INDEX

    features = {}

    if KEYPOINT_INDEX is None:
        return features

    # Find release frame (max wrist velocity)
    wrist_idx = KEYPOINT_INDEX.get("right_wrist", 10)
    wrist_data = timeseries[:, wrist_idx*3:(wrist_idx+1)*3]

    vel = np.gradient(wrist_data, 1/60, axis=0)
    vel_mag = np.linalg.norm(vel, axis=1)

    search_start = NUM_FRAMES // 3
    release_frame = search_start + np.argmax(vel_mag[search_start:])

    # Release position
    release_pos = wrist_data[release_frame]
    features["release_x"] = release_pos[0]
    features["release_y"] = release_pos[1]
    features["release_z"] = release_pos[2]

    # Release velocity
    release_vel = vel[release_frame]
    features["vel_x"] = release_vel[0]
    features["vel_y"] = release_vel[1]
    features["vel_z"] = release_vel[2]
    features["vel_mag"] = np.linalg.norm(release_vel)

    # Velocity direction angles
    horiz_vel = np.sqrt(release_vel[0]**2 + release_vel[1]**2)
    features["launch_angle"] = np.arctan2(release_vel[2], horiz_vel) * 180 / np.pi
    features["azimuth"] = np.arctan2(release_vel[0], release_vel[1]) * 180 / np.pi

    # Acceleration at release
    acc = np.gradient(vel, 1/60, axis=0)
    features["acc_mag"] = np.linalg.norm(acc[release_frame])

    # Height relative to rim
    features["height_above_rim"] = release_pos[2] - RIM_HEIGHT

    # Elbow and shoulder features
    elbow_idx = KEYPOINT_INDEX.get("right_elbow", 8)
    shoulder_idx = KEYPOINT_INDEX.get("right_shoulder", 6)

    if elbow_idx and shoulder_idx:
        elbow_data = timeseries[release_frame, elbow_idx*3:(elbow_idx+1)*3]
        shoulder_data = timeseries[release_frame, shoulder_idx*3:(shoulder_idx+1)*3]

        arm_vector = wrist_data[release_frame] - shoulder_data
        features["arm_extension"] = np.linalg.norm(arm_vector)
        features["arm_angle_z"] = np.arctan2(arm_vector[2], np.sqrt(arm_vector[0]**2 + arm_vector[1]**2)) * 180 / np.pi

    return features


def compute_trajectory_error(v0, pos0, target_angle_deg, target_depth):
    """Compute physics error between predicted velocity and expected trajectory."""
    # Convert launch angle to radians
    launch_rad = target_angle_deg * np.pi / 180

    # Decompose velocity
    v_horiz = v0 * np.cos(launch_rad)
    v_vert = v0 * np.sin(launch_rad)

    # Time to reach rim distance
    horizontal_dist = RIM_DISTANCE  # Approximate
    if v_horiz > 0:
        t_flight = horizontal_dist / v_horiz
    else:
        t_flight = 2.0  # Default

    # Height at rim using kinematic equation
    # z = z0 + v_vert * t - 0.5 * g * t^2
    predicted_z = pos0[2] + v_vert * t_flight - 0.5 * G * t_flight**2

    # Expected rim height (with depth adjustment)
    expected_z = RIM_HEIGHT + target_depth * 0.1  # Rough conversion

    # Physics error
    height_error = (predicted_z - expected_z)**2

    return height_error


try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True

    class PhysicsConstrainedNN(nn.Module):
        """NN that predicts targets with physics consistency loss."""

        def __init__(self, input_dim, hidden_dim=32):
            super().__init__()
            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )

            # Separate heads for each target
            self.angle_head = nn.Linear(hidden_dim, 1)
            self.depth_head = nn.Linear(hidden_dim, 1)
            self.lr_head = nn.Linear(hidden_dim, 1)

            # Auxiliary: predict velocity magnitude (for physics constraint)
            self.vel_head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            h = self.feature_net(x)
            angle = self.angle_head(h)
            depth = self.depth_head(h)
            lr = self.lr_head(h)
            vel = self.vel_head(h)
            return torch.cat([angle, depth, lr], dim=1), vel

    def physics_loss(angle_pred, depth_pred, vel_pred, release_pos, physics_weight=0.1):
        """Compute physics consistency loss."""
        # This is a simplified version - actual ballistics would be more complex
        # The idea is that angle and velocity should be consistent with trajectory

        # Expected relationship: higher angle requires higher velocity for same depth
        # Lower release height also requires higher velocity

        batch_size = angle_pred.shape[0]

        # Simple physics constraint: v^2 * sin(2*angle) should be proportional to range
        angle_rad = angle_pred * np.pi / 180
        expected_factor = vel_pred**2 * torch.sin(2 * angle_rad + 0.01)

        # Depth should correlate with this factor
        # (higher factor = longer shot = larger depth)
        physics_error = torch.var(expected_factor - depth_pred)

        return physics_weight * physics_error

    def train_physics_nn(X_train, y_train, X_val, y_val, epochs=100, physics_weight=0.1):
        """Train physics-constrained NN."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = PhysicsConstrainedNN(X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        mse_loss = nn.MSELoss()

        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)

        best_val_loss = float("inf")
        patience = 15
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            pred, vel = model(X_train_t)

            # MSE loss on targets
            target_loss = mse_loss(pred, y_train_t)

            # Physics consistency loss
            phys_loss = physics_loss(pred[:, 0:1], pred[:, 1:2], vel, X_train_t[:, :3], physics_weight)

            total_loss = target_loss + phys_loss
            total_loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred, _ = model(X_val_t)
                val_loss = mse_loss(val_pred, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)

        return model, best_val_loss

except ImportError:
    TORCH_AVAILABLE = False


def evaluate_lopo(X, y, pids, physics_weight=0.1):
    """LOPO CV with physics-constrained NN."""
    if not TORCH_AVAILABLE:
        return float("inf"), {}

    scalers_dict = load_scalers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        print(f"  Training for participant {pid}...")

        train_mask = pids != pid
        test_mask = pids == pid

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        # Scale
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # Scale targets
        y_train_scaled = np.zeros_like(y_train)
        for i, target in enumerate(["angle", "depth", "left_right"]):
            y_train_scaled[:, i] = scalers_dict[target].transform(y_train[:, i].reshape(-1, 1)).ravel()

        # Split for validation
        val_size = max(1, int(len(X_train_scaled) * 0.2))
        X_train_inner = X_train_scaled[:-val_size]
        y_train_inner = y_train_scaled[:-val_size]
        X_val = X_train_scaled[-val_size:]
        y_val = y_train_scaled[-val_size:]

        # Train model
        model, _ = train_physics_nn(X_train_inner, y_train_inner, X_val, y_val,
                                    epochs=100, physics_weight=physics_weight)

        # Predict
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_scaled).to(device)
            pred_scaled, _ = model(X_test_t)
            pred_scaled = pred_scaled.cpu().numpy()

        # Inverse scale
        for i, target in enumerate(["angle", "depth", "left_right"]):
            all_preds[test_mask, i] = scalers_dict[target].inverse_transform(
                pred_scaled[:, i].reshape(-1, 1)
            ).ravel()

    # Calculate MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def main():
    print("=" * 60)
    print("PHYSICS-CONSTRAINED NEURAL NETWORK (Plan 3.1)")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch required")
        return

    print("\nBuilding physics feature matrix...")
    X, y, pids = build_raw_data(train=True)
    print(f"X shape: {X.shape}")

    results = []

    for physics_weight in [0.0, 0.01, 0.05, 0.1, 0.2]:
        print(f"\n--- physics_weight={physics_weight} ---")

        lopo_mse, per_target = evaluate_lopo(X, y, pids, physics_weight=physics_weight)

        results.append({
            "physics_weight": physics_weight,
            "lopo_mse": lopo_mse,
            "angle_mse": per_target.get("angle", np.nan),
            "depth_mse": per_target.get("depth", np.nan),
            "lr_mse": per_target.get("left_right", np.nan),
        })

        print(f"  LOPO MSE: {lopo_mse:.6f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "physics_nn_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'physics_nn_results.csv'}")

    # Best config
    best_idx = results_df["lopo_mse"].idxmin()
    best = results_df.iloc[best_idx]
    print(f"\n=== BEST CONFIG ===")
    print(f"Physics weight: {best['physics_weight']}")
    print(f"LOPO MSE: {best['lopo_mse']:.6f}")


if __name__ == "__main__":
    main()

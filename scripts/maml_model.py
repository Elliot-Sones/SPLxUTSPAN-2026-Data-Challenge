"""
Meta-Learning MAML for Player Adaptation (Plan 3.2)

MAML finds initialization that adapts quickly to new player with few samples.
Captures shared structure across player mechanics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers
from data_loader import get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"


def build_feature_matrix(train: bool = True):
    """Build feature matrix."""
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    meta_df = load_metadata(train)
    n_shots = len(meta_df)

    all_features = []
    targets = []
    participant_ids = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train)):
        if i % 50 == 0:
            print(f"  Processing shot {i+1}/{n_shots}...")

        features = extract_hybrid_features(timeseries, metadata["participant_id"])
        all_features.append(features)
        participant_ids.append(metadata["participant_id"])

        if train:
            targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])

    feature_df = pd.DataFrame(all_features)
    feature_df = feature_df.fillna(feature_df.median())
    feature_df = feature_df.loc[:, feature_df.nunique() > 1]

    X = feature_df.values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids, feature_df.columns.tolist()


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True

    class MAMLPredictor(nn.Module):
        """Simple feedforward network for MAML."""

        def __init__(self, input_dim, hidden_dim=32, output_dim=3):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, weights=None):
            if weights is None:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
            else:
                x = F.relu(F.linear(x, weights["fc1.weight"], weights["fc1.bias"]))
                x = F.relu(F.linear(x, weights["fc2.weight"], weights["fc2.bias"]))
                x = F.linear(x, weights["fc3.weight"], weights["fc3.bias"])
            return x

    def maml_inner_update(model, X_support, y_support, inner_lr=0.01, num_steps=5):
        """Perform inner loop adaptation."""
        # Clone weights
        weights = {name: param.clone() for name, param in model.named_parameters()}

        for _ in range(num_steps):
            pred = model(X_support, weights)
            loss = F.mse_loss(pred, y_support)

            # Compute gradients manually
            grads = torch.autograd.grad(loss, weights.values(), create_graph=True)

            # Update weights
            weights = {name: w - inner_lr * g for (name, w), g in zip(weights.items(), grads)}

        return weights

    def maml_train_step(model, meta_optimizer, tasks, inner_lr=0.01, num_inner_steps=5):
        """One MAML meta-training step."""
        meta_loss = 0

        for X_support, y_support, X_query, y_query in tasks:
            # Inner loop: adapt to support set
            adapted_weights = maml_inner_update(model, X_support, y_support, inner_lr, num_inner_steps)

            # Outer loop: evaluate on query set with adapted weights
            pred = model(X_query, adapted_weights)
            loss = F.mse_loss(pred, y_query)
            meta_loss += loss

        meta_loss /= len(tasks)

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        return meta_loss.item()

    def create_tasks(X, y, pids, n_support=10, n_query=5):
        """Create meta-learning tasks (one task per player)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tasks = []

        for pid in np.unique(pids):
            mask = pids == pid
            X_pid = X[mask]
            y_pid = y[mask]

            if len(X_pid) < n_support + n_query:
                continue

            # Random split
            indices = np.random.permutation(len(X_pid))
            support_idx = indices[:n_support]
            query_idx = indices[n_support:n_support + n_query]

            X_support = torch.FloatTensor(X_pid[support_idx]).to(device)
            y_support = torch.FloatTensor(y_pid[support_idx]).to(device)
            X_query = torch.FloatTensor(X_pid[query_idx]).to(device)
            y_query = torch.FloatTensor(y_pid[query_idx]).to(device)

            tasks.append((X_support, y_support, X_query, y_query))

        return tasks

except ImportError:
    TORCH_AVAILABLE = False


def evaluate_maml_transfer(X, y, pids, meta_epochs=100, inner_lr=0.01, num_inner_steps=5):
    """Evaluate MAML transfer to held-out player."""
    if not TORCH_AVAILABLE:
        return float("inf"), {}

    scalers_dict = load_scalers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for test_pid in unique_pids:
        print(f"\n  Testing transfer to player {test_pid}...")

        train_mask = pids != test_pid
        test_mask = pids == test_pid

        X_train = X[train_mask]
        y_train = y[train_mask]
        pids_train = pids[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        # Scale
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # Scale targets
        y_train_scaled = np.zeros_like(y_train)
        for i, target in enumerate(["angle", "depth", "left_right"]):
            y_train_scaled[:, i] = scalers_dict[target].transform(y_train[:, i].reshape(-1, 1)).ravel()

        # Meta-train on 4 players
        model = MAMLPredictor(X_train_scaled.shape[1], hidden_dim=32, output_dim=3).to(device)
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(meta_epochs):
            tasks = create_tasks(X_train_scaled, y_train_scaled, pids_train, n_support=10, n_query=5)
            if len(tasks) == 0:
                break

            loss = maml_train_step(model, meta_optimizer, tasks, inner_lr, num_inner_steps)

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}: meta_loss = {loss:.4f}")

        # Adapt to test player with few samples
        n_adapt = min(10, len(X_test_scaled))
        X_adapt = torch.FloatTensor(X_test_scaled[:n_adapt]).to(device)

        # Scale test targets for adaptation
        y_test_scaled = np.zeros_like(y_test)
        for i, target in enumerate(["angle", "depth", "left_right"]):
            y_test_scaled[:, i] = scalers_dict[target].transform(y_test[:, i].reshape(-1, 1)).ravel()
        y_adapt = torch.FloatTensor(y_test_scaled[:n_adapt]).to(device)

        adapted_weights = maml_inner_update(model, X_adapt, y_adapt, inner_lr, num_inner_steps)

        # Predict on all test samples
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_scaled).to(device)
            pred_scaled = model(X_test_t, adapted_weights).cpu().numpy()

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


def compare_with_baseline(X, y, pids):
    """Compare MAML with simple Ridge baseline."""
    scalers_dict = load_scalers()

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        train_mask = pids != pid
        test_mask = pids == pid

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for target_idx in range(3):
            model = Ridge(alpha=100)
            model.fit(X_train_scaled, y_train[:, target_idx])
            all_preds[test_mask, target_idx] = model.predict(X_test_scaled)

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
    print("META-LEARNING MAML (Plan 3.2)")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch required")
        return

    print("\nBuilding feature matrix...")
    X, y, pids, feature_names = build_feature_matrix(train=True)
    print(f"X shape: {X.shape}")

    # Baseline comparison
    print("\n--- Baseline (Ridge LOPO) ---")
    baseline_mse, baseline_per_target = compare_with_baseline(X, y, pids)
    print(f"Baseline LOPO MSE: {baseline_mse:.6f}")

    results = []

    # Test MAML with different configs
    for meta_epochs in [50, 100]:
        for inner_lr in [0.01, 0.05]:
            for num_inner_steps in [3, 5]:
                print(f"\n--- MAML epochs={meta_epochs}, inner_lr={inner_lr}, steps={num_inner_steps} ---")

                maml_mse, maml_per_target = evaluate_maml_transfer(
                    X, y, pids,
                    meta_epochs=meta_epochs,
                    inner_lr=inner_lr,
                    num_inner_steps=num_inner_steps
                )

                improvement = (baseline_mse - maml_mse) / baseline_mse * 100

                results.append({
                    "meta_epochs": meta_epochs,
                    "inner_lr": inner_lr,
                    "num_inner_steps": num_inner_steps,
                    "maml_mse": maml_mse,
                    "baseline_mse": baseline_mse,
                    "improvement_pct": improvement,
                    **{f"maml_{k}": v for k, v in maml_per_target.items()},
                })

                print(f"  MAML MSE: {maml_mse:.6f}, Improvement: {improvement:.2f}%")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "maml_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'maml_results.csv'}")

    # Best config
    if len(results_df) > 0:
        best_idx = results_df["maml_mse"].idxmin()
        best = results_df.iloc[best_idx]
        print(f"\n=== BEST MAML CONFIG ===")
        print(f"Meta epochs: {best['meta_epochs']}")
        print(f"Inner LR: {best['inner_lr']}")
        print(f"Inner steps: {best['num_inner_steps']}")
        print(f"MAML MSE: {best['maml_mse']:.6f}")
        print(f"Improvement over baseline: {best['improvement_pct']:.2f}%")


if __name__ == "__main__":
    main()

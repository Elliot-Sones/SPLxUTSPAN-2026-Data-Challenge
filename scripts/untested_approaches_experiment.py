"""
Comprehensive experiment testing untested approaches from research document.

Tests:
1. Signal processing: Savitzky-Golay denoising, wavelet decomposition
2. TCN (Temporal Convolutional Network) with heavy regularization
3. Simple Transformer with heavy regularization
4. ST-GCN (Spatial-Temporal Graph Convolutional Network)
5. Bayesian ensemble with MC Dropout

All evaluated using GroupKFold CV by participant.
"""

import json
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.signal import savgol_filter
import pywt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

TARGETS = ["angle", "depth", "left_right"]
NUM_FRAMES = 240
NUM_KEYPOINTS = 69
NUM_COORDS = 3
FRAME_RATE = 60

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_array_string(s):
    """Parse string array to numpy array."""
    if pd.isna(s):
        return np.full(NUM_FRAMES, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_raw_data(train: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw timeseries data.

    Returns:
        X: (n_samples, 240, 207) raw timeseries
        y: (n_samples, 3) targets (None for test)
        pids: (n_samples,) participant IDs
    """
    filename = "train.csv" if train else "test.csv"
    df = pd.read_csv(DATA_DIR / filename)

    meta_cols = ["id", "shot_id", "participant_id"]
    if train:
        meta_cols.extend(TARGETS)
    keypoint_cols = [c for c in df.columns if c not in meta_cols]

    n_samples = len(df)
    X = np.zeros((n_samples, NUM_FRAMES, len(keypoint_cols)), dtype=np.float32)

    for idx, row in df.iterrows():
        for i, col in enumerate(keypoint_cols):
            X[idx, :, i] = parse_array_string(row[col])

    pids = df["participant_id"].values

    if train:
        y = df[TARGETS].values.astype(np.float32)
    else:
        y = None

    return X, y, pids


# =============================================================================
# 1. SIGNAL PROCESSING FEATURES
# =============================================================================

def apply_savgol_denoising(X: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to denoise timeseries.
    Preserves kinematic peaks better than simple smoothing.
    """
    X_denoised = np.zeros_like(X)

    for i in range(X.shape[0]):  # samples
        for j in range(X.shape[2]):  # features
            series = X[i, :, j]
            if np.isnan(series).all():
                X_denoised[i, :, j] = series
            else:
                # Handle NaN by interpolating first
                valid_mask = ~np.isnan(series)
                if valid_mask.sum() < window_length:
                    X_denoised[i, :, j] = series
                else:
                    # Interpolate NaN values
                    series_clean = np.interp(
                        np.arange(len(series)),
                        np.where(valid_mask)[0],
                        series[valid_mask]
                    )
                    X_denoised[i, :, j] = savgol_filter(series_clean, window_length, polyorder)

    return X_denoised


def extract_wavelet_features(X: np.ndarray, wavelet: str = 'db4', level: int = 4) -> np.ndarray:
    """
    Extract wavelet decomposition features.
    Returns coefficients at different scales capturing multi-resolution patterns.
    """
    n_samples = X.shape[0]
    n_features = X.shape[2]

    # Calculate output size: we'll extract statistics from each decomposition level
    # For each feature: approx coeffs + detail coeffs at each level (mean, std, energy per level)
    stats_per_level = 3  # mean, std, energy
    n_wavelet_features = n_features * (level + 1) * stats_per_level

    wavelet_features = np.zeros((n_samples, n_wavelet_features), dtype=np.float32)

    for i in range(n_samples):
        feat_idx = 0
        for j in range(n_features):
            series = X[i, :, j]

            if np.isnan(series).all():
                feat_idx += (level + 1) * stats_per_level
                continue

            # Interpolate NaN
            valid_mask = ~np.isnan(series)
            if valid_mask.sum() < 10:
                feat_idx += (level + 1) * stats_per_level
                continue

            series_clean = np.interp(
                np.arange(len(series)),
                np.where(valid_mask)[0],
                series[valid_mask]
            )

            # Wavelet decomposition
            try:
                coeffs = pywt.wavedec(series_clean, wavelet, level=level)

                for c in coeffs:
                    wavelet_features[i, feat_idx] = np.mean(c)
                    wavelet_features[i, feat_idx + 1] = np.std(c)
                    wavelet_features[i, feat_idx + 2] = np.sum(c**2)
                    feat_idx += stats_per_level

            except Exception:
                feat_idx += (level + 1) * stats_per_level

    return wavelet_features


def extract_frequency_features(X: np.ndarray, n_freq_bins: int = 10) -> np.ndarray:
    """
    Extract frequency domain features using FFT.
    """
    n_samples = X.shape[0]
    n_features = X.shape[2]

    # For each feature: top n_freq_bins magnitudes + spectral centroid + spectral entropy
    n_freq_features = n_features * (n_freq_bins + 2)
    freq_features = np.zeros((n_samples, n_freq_features), dtype=np.float32)

    for i in range(n_samples):
        feat_idx = 0
        for j in range(n_features):
            series = X[i, :, j]

            if np.isnan(series).all():
                feat_idx += n_freq_bins + 2
                continue

            # Interpolate NaN
            valid_mask = ~np.isnan(series)
            if valid_mask.sum() < 10:
                feat_idx += n_freq_bins + 2
                continue

            series_clean = np.interp(
                np.arange(len(series)),
                np.where(valid_mask)[0],
                series[valid_mask]
            )

            # FFT
            fft_vals = np.abs(np.fft.rfft(series_clean))
            freqs = np.fft.rfftfreq(len(series_clean), 1/FRAME_RATE)

            # Top magnitude bins
            sorted_indices = np.argsort(fft_vals)[::-1]
            for k in range(min(n_freq_bins, len(fft_vals))):
                freq_features[i, feat_idx + k] = fft_vals[sorted_indices[k]]
            feat_idx += n_freq_bins

            # Spectral centroid
            if fft_vals.sum() > 0:
                spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
            else:
                spectral_centroid = 0
            freq_features[i, feat_idx] = spectral_centroid
            feat_idx += 1

            # Spectral entropy
            fft_norm = fft_vals / (fft_vals.sum() + 1e-10)
            spectral_entropy = -np.sum(fft_norm * np.log(fft_norm + 1e-10))
            freq_features[i, feat_idx] = spectral_entropy
            feat_idx += 1

    return freq_features


# =============================================================================
# 2. TCN (TEMPORAL CONVOLUTIONAL NETWORK)
# =============================================================================

class TemporalBlock(nn.Module):
    """Single temporal block with dilated causal convolution."""

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.5):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, channels, time)
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # Remove padding for causal
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)

        # Match temporal dimension
        if out.size(2) != res.size(2):
            min_len = min(out.size(2), res.size(2))
            out = out[:, :, -min_len:]
            res = res[:, :, -min_len:]

        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for sequence regression.
    Heavy regularization for small data regime.
    """

    def __init__(self, input_channels=207, num_outputs=3,
                 num_channels=[32, 32, 32], kernel_size=3, dropout=0.5):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)

        # Global pooling + regression head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, time, features) -> (batch, features, time)
        x = x.permute(0, 2, 1)

        out = self.network(x)
        out = self.global_pool(out).squeeze(-1)
        out = self.dropout(out)
        out = self.fc(out)

        return out


# =============================================================================
# 3. SIMPLE TRANSFORMER
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=300, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    """
    Lightweight Transformer for sequence regression.
    Uses extreme regularization for small data regime.
    """

    def __init__(self, input_dim=207, d_model=64, nhead=4, num_layers=2,
                 num_outputs=3, dropout=0.5):
        super().__init__()

        # Input projection with dimensionality reduction
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Small feedforward
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_outputs)
        )

    def forward(self, x):
        # x: (batch, time, features)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Global average pooling over time
        x = x.mean(dim=1)

        return self.fc(x)


# =============================================================================
# 4. ST-GCN (SPATIAL-TEMPORAL GRAPH CONVOLUTIONAL NETWORK)
# =============================================================================

def get_skeleton_adjacency():
    """
    Define skeleton graph adjacency matrix.
    Based on human body structure with 69 keypoints.
    """
    # Key connections (simplified - major joints)
    # This is a subset focusing on the main skeletal connections
    connections = [
        # Spine
        ('nose', 'neck'), ('neck', 'mid_hip'),
        # Right arm
        ('neck', 'right_shoulder'), ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        # Left arm
        ('neck', 'left_shoulder'), ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        # Right leg
        ('mid_hip', 'right_hip'), ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
        # Left leg
        ('mid_hip', 'left_hip'), ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
    ]

    # Map keypoint names to indices (assuming standard ordering)
    keypoint_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        # Finger keypoints (17-56)
        'left_thumb_cmc', 'left_thumb_mcp', 'left_thumb_ip', 'left_thumb_tip',
        'left_first_finger_mcp', 'left_first_finger_pip', 'left_first_finger_dip', 'left_first_finger_tip',
        'left_second_finger_mcp', 'left_second_finger_pip', 'left_second_finger_dip', 'left_second_finger_tip',
        'left_third_finger_mcp', 'left_third_finger_pip', 'left_third_finger_dip', 'left_third_finger_tip',
        'left_fourth_finger_mcp', 'left_fourth_finger_pip', 'left_fourth_finger_dip', 'left_fourth_finger_tip',
        'right_thumb_cmc', 'right_thumb_mcp', 'right_thumb_ip', 'right_thumb_tip',
        'right_first_finger_mcp', 'right_first_finger_pip', 'right_first_finger_dip', 'right_first_finger_tip',
        'right_second_finger_mcp', 'right_second_finger_pip', 'right_second_finger_dip', 'right_second_finger_tip',
        'right_third_finger_mcp', 'right_third_finger_pip', 'right_third_finger_dip', 'right_third_finger_tip',
        'right_fourth_finger_mcp', 'right_fourth_finger_pip', 'right_fourth_finger_dip', 'right_fourth_finger_tip',
        # Lower body (57-68)
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
        # Additional
        'neck', 'mid_hip'
    ]

    # Build adjacency matrix (simplified for major joints only)
    n_joints = 15  # Use simplified skeleton
    adj = np.eye(n_joints, dtype=np.float32)  # Self-connections

    # Add neighbor connections
    simplified_connections = [
        (0, 1), (1, 2),  # spine
        (1, 3), (3, 4), (4, 5),  # right arm
        (1, 6), (6, 7), (7, 8),  # left arm
        (2, 9), (9, 10), (10, 11),  # right leg
        (2, 12), (12, 13), (13, 14),  # left leg
    ]

    for i, j in simplified_connections:
        adj[i, j] = 1
        adj[j, i] = 1

    # Normalize
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt

    return torch.FloatTensor(adj_normalized)


class GraphConvolution(nn.Module):
    """Simple graph convolution layer."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (batch, nodes, features)
        # adj: (nodes, nodes)
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support) + self.bias
        return output


class STGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network.
    Combines graph convolution (spatial) with temporal convolution.
    """

    def __init__(self, in_channels=3, num_joints=15, num_frames=240,
                 hidden_channels=32, num_outputs=3, dropout=0.5):
        super().__init__()

        # Spatial graph convolutions
        self.gc1 = GraphConvolution(in_channels, hidden_channels)
        self.gc2 = GraphConvolution(hidden_channels, hidden_channels)

        # Temporal convolution
        self.temporal_conv = nn.Conv1d(
            num_joints * hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1
        )

        # Output
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_outputs)
        )

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, adj):
        # x: (batch, frames, joints, coords)
        batch_size, num_frames, num_joints, _ = x.shape

        # Process each frame with GCN
        gcn_outputs = []
        for t in range(num_frames):
            frame = x[:, t, :, :]  # (batch, joints, coords)
            h = F.relu(self.gc1(frame, adj))
            h = self.dropout(h)
            h = F.relu(self.gc2(h, adj))
            gcn_outputs.append(h)

        # Stack: (batch, frames, joints, hidden)
        gcn_out = torch.stack(gcn_outputs, dim=1)

        # Reshape for temporal conv: (batch, joints*hidden, frames)
        gcn_out = gcn_out.permute(0, 2, 3, 1).reshape(
            batch_size, num_joints * gcn_out.shape[-1], num_frames
        )

        # Temporal convolution
        temporal_out = self.temporal_conv(gcn_out)
        temporal_out = self.bn(temporal_out)
        temporal_out = F.relu(temporal_out)

        # Global pooling
        out = temporal_out.mean(dim=-1)  # (batch, hidden)

        return self.fc(out)


# =============================================================================
# 5. BAYESIAN ENSEMBLE WITH MC DROPOUT
# =============================================================================

class MCDropoutMLP(nn.Module):
    """MLP with dropout that remains active during inference for MC sampling."""

    def __init__(self, input_dim, hidden_dims=[128, 64], num_outputs=3, dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_with_uncertainty(self, x, n_samples=50):
        """
        MC Dropout prediction with uncertainty estimation.

        Returns:
            mean: Mean prediction
            std: Standard deviation (uncertainty)
        """
        self.train()  # Keep dropout active

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_pytorch_model(model, train_loader, val_loader,
                        epochs=100, lr=0.001, weight_decay=1e-3,
                        patience=10, device=DEVICE):
    """Train a PyTorch model with early stopping."""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            if len(batch) == 2:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
            else:
                X_batch, adj, y_batch = batch
                X_batch, adj, y_batch = X_batch.to(device), adj.to(device), y_batch.to(device)
                pred = model(X_batch, adj)

            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    X_batch, y_batch = batch
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    pred = model(X_batch)
                else:
                    X_batch, adj, y_batch = batch
                    X_batch, adj, y_batch = X_batch.to(device), adj.to(device), y_batch.to(device)
                    pred = model(X_batch, adj)
                val_loss += criterion(pred, y_batch).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_groupkfold(X, y, pids, model_fn, n_splits=5, model_name="Model"):
    """
    Evaluate model using GroupKFold CV by participant.
    Trains separate model per target for non-multioutput regressors.

    Args:
        X: Features or raw data
        y: Targets (n_samples, 3)
        pids: Participant IDs
        model_fn: Function that takes (X_train, y_train_1d) and returns trained model with predict method
        n_splits: Number of folds
        model_name: Name for logging

    Returns:
        Dict with per-target and total MSE
    """
    gkf = GroupKFold(n_splits=n_splits)

    all_preds = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, pids)):
        if isinstance(X, np.ndarray):
            X_train, X_val = X[train_idx], X[val_idx]
        else:
            X_train, X_val = [X[i] for i in train_idx], [X[i] for i in val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train separate model per target
        fold_preds = np.zeros((len(val_idx), 3))
        for t_idx in range(3):
            model = model_fn(X_train, y_train[:, t_idx])
            if hasattr(model, 'predict'):
                fold_preds[:, t_idx] = model.predict(X_val)
            else:
                fold_preds[:, t_idx] = model(X_val)

        all_preds[val_idx] = fold_preds

        fold_mse = mean_squared_error(y_val, fold_preds)
        print(f"  Fold {fold+1}: MSE = {fold_mse:.6f}")

    # Calculate per-target MSE
    results = {}
    for i, target in enumerate(TARGETS):
        mse = mean_squared_error(y[:, i], all_preds[:, i])
        results[target] = mse

    results['total'] = mean_squared_error(y, all_preds)

    print(f"\n{model_name} GroupKFold Results:")
    for target in TARGETS:
        print(f"  {target}: MSE = {results[target]:.6f}")
    print(f"  TOTAL: MSE = {results['total']:.6f}")

    return results, all_preds


# =============================================================================
# MAIN EXPERIMENTS
# =============================================================================

def run_signal_processing_experiment():
    """Test signal processing features: Savitzky-Golay + wavelets."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: SIGNAL PROCESSING FEATURES")
    print("="*70)

    # Load raw data
    print("\nLoading raw data...")
    X_raw, y, pids = load_raw_data(train=True)
    print(f"Raw data shape: {X_raw.shape}")

    # 1a. Savitzky-Golay denoising
    print("\n--- 1a. Savitzky-Golay Denoising ---")
    X_denoised = apply_savgol_denoising(X_raw, window_length=11, polyorder=3)

    # Extract simple features from denoised data
    X_denoised_flat = X_denoised.reshape(X_denoised.shape[0], -1)

    # Use mean/std per feature as simple features
    X_denoised_stats = np.concatenate([
        X_denoised.mean(axis=1),
        X_denoised.std(axis=1),
    ], axis=1)
    print(f"Denoised stats features shape: {X_denoised_stats.shape}")

    # Train LightGBM on denoised features
    def lgb_model_fn(X_train, y_train):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(np.nan_to_num(X_train))

        model = lgb.LGBMRegressor(
            n_estimators=100, num_leaves=10, learning_rate=0.05,
            reg_alpha=0.5, reg_lambda=0.5, verbose=-1, n_jobs=-1
        )

        class ModelWrapper:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler

            def predict(self, X):
                X_scaled = self.scaler.transform(np.nan_to_num(X))
                return self.model.predict(X_scaled)

        model.fit(X_train_scaled, y_train)
        return ModelWrapper(model, scaler)

    results_savgol, _ = evaluate_groupkfold(
        X_denoised_stats, y, pids, lgb_model_fn,
        model_name="Savitzky-Golay Denoised"
    )

    # 1b. Wavelet features
    print("\n--- 1b. Wavelet Features ---")
    X_wavelet = extract_wavelet_features(X_raw, wavelet='db4', level=4)
    print(f"Wavelet features shape: {X_wavelet.shape}")

    results_wavelet, _ = evaluate_groupkfold(
        X_wavelet, y, pids, lgb_model_fn,
        model_name="Wavelet Features"
    )

    # 1c. Frequency features
    print("\n--- 1c. Frequency Features ---")
    X_freq = extract_frequency_features(X_raw, n_freq_bins=10)
    print(f"Frequency features shape: {X_freq.shape}")

    results_freq, _ = evaluate_groupkfold(
        X_freq, y, pids, lgb_model_fn,
        model_name="Frequency Features"
    )

    # 1d. Combined signal processing features
    print("\n--- 1d. Combined Signal Processing ---")
    X_combined = np.concatenate([X_denoised_stats, X_wavelet, X_freq], axis=1)
    print(f"Combined features shape: {X_combined.shape}")

    results_combined, _ = evaluate_groupkfold(
        X_combined, y, pids, lgb_model_fn,
        model_name="Combined Signal Processing"
    )

    return {
        'savgol': results_savgol,
        'wavelet': results_wavelet,
        'frequency': results_freq,
        'combined': results_combined
    }


def run_tcn_experiment():
    """Test TCN (Temporal Convolutional Network)."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: TCN (TEMPORAL CONVOLUTIONAL NETWORK)")
    print("="*70)

    # Load raw data
    print("\nLoading raw data...")
    X_raw, y, pids = load_raw_data(train=True)

    # Normalize per sample
    X_norm = np.zeros_like(X_raw)
    for i in range(len(X_raw)):
        mean = np.nanmean(X_raw[i])
        std = np.nanstd(X_raw[i]) + 1e-8
        X_norm[i] = (X_raw[i] - mean) / std
    X_norm = np.nan_to_num(X_norm, nan=0.0)

    gkf = GroupKFold(n_splits=5)
    all_preds = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_norm, y, pids)):
        print(f"\nFold {fold+1}/5")

        X_train = torch.FloatTensor(X_norm[train_idx])
        y_train = torch.FloatTensor(y[train_idx])
        X_val = torch.FloatTensor(X_norm[val_idx])
        y_val = torch.FloatTensor(y[val_idx])

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        # Create model with heavy regularization
        model = TCN(
            input_channels=207,
            num_outputs=3,
            num_channels=[32, 32],  # Small network
            kernel_size=3,
            dropout=0.5
        )

        model, val_loss = train_pytorch_model(
            model, train_loader, val_loader,
            epochs=100, lr=0.001, weight_decay=1e-2,
            patience=15
        )

        # Predict
        model.eval()
        with torch.no_grad():
            preds = model(X_val.to(DEVICE)).cpu().numpy()

        all_preds[val_idx] = preds
        fold_mse = mean_squared_error(y[val_idx], preds)
        print(f"  Fold {fold+1}: MSE = {fold_mse:.6f}")

    # Results
    results = {}
    for i, target in enumerate(TARGETS):
        results[target] = mean_squared_error(y[:, i], all_preds[:, i])
    results['total'] = mean_squared_error(y, all_preds)

    print(f"\nTCN GroupKFold Results:")
    for target in TARGETS:
        print(f"  {target}: MSE = {results[target]:.6f}")
    print(f"  TOTAL: MSE = {results['total']:.6f}")

    return results


def run_transformer_experiment():
    """Test simple Transformer with heavy regularization."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: SIMPLE TRANSFORMER")
    print("="*70)

    # Load raw data
    print("\nLoading raw data...")
    X_raw, y, pids = load_raw_data(train=True)

    # Normalize
    X_norm = np.zeros_like(X_raw)
    for i in range(len(X_raw)):
        mean = np.nanmean(X_raw[i])
        std = np.nanstd(X_raw[i]) + 1e-8
        X_norm[i] = (X_raw[i] - mean) / std
    X_norm = np.nan_to_num(X_norm, nan=0.0)

    # Subsample frames for efficiency (every 4th frame)
    X_subsampled = X_norm[:, ::4, :]  # (n, 60, 207)
    print(f"Subsampled shape: {X_subsampled.shape}")

    gkf = GroupKFold(n_splits=5)
    all_preds = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_subsampled, y, pids)):
        print(f"\nFold {fold+1}/5")

        X_train = torch.FloatTensor(X_subsampled[train_idx])
        y_train = torch.FloatTensor(y[train_idx])
        X_val = torch.FloatTensor(X_subsampled[val_idx])
        y_val = torch.FloatTensor(y[val_idx])

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        # Create model
        model = SimpleTransformer(
            input_dim=207,
            d_model=32,  # Very small
            nhead=4,
            num_layers=2,
            num_outputs=3,
            dropout=0.5
        )

        model, val_loss = train_pytorch_model(
            model, train_loader, val_loader,
            epochs=100, lr=0.0005, weight_decay=1e-2,
            patience=15
        )

        # Predict
        model.eval()
        with torch.no_grad():
            preds = model(X_val.to(DEVICE)).cpu().numpy()

        all_preds[val_idx] = preds
        fold_mse = mean_squared_error(y[val_idx], preds)
        print(f"  Fold {fold+1}: MSE = {fold_mse:.6f}")

    # Results
    results = {}
    for i, target in enumerate(TARGETS):
        results[target] = mean_squared_error(y[:, i], all_preds[:, i])
    results['total'] = mean_squared_error(y, all_preds)

    print(f"\nTransformer GroupKFold Results:")
    for target in TARGETS:
        print(f"  {target}: MSE = {results[target]:.6f}")
    print(f"  TOTAL: MSE = {results['total']:.6f}")

    return results


def run_stgcn_experiment():
    """Test ST-GCN (Spatial-Temporal Graph Convolutional Network)."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: ST-GCN")
    print("="*70)

    # Load raw data
    print("\nLoading raw data...")
    X_raw, y, pids = load_raw_data(train=True)

    # Reshape to (samples, frames, joints, coords)
    # Use simplified 15-joint skeleton
    # Select major joints from 69 keypoints
    joint_indices = [0, 66, 67,  # nose, neck, mid_hip
                    11, 13, 15,  # left shoulder, elbow, wrist
                    12, 14, 16,  # right shoulder, elbow, wrist
                    57, 59, 61,  # left hip, knee, ankle
                    58, 60, 62]  # right hip, knee, ankle

    n_joints = len(joint_indices)
    X_joints = np.zeros((len(X_raw), NUM_FRAMES, n_joints, 3), dtype=np.float32)

    for i, joint_idx in enumerate(joint_indices):
        X_joints[:, :, i, :] = X_raw[:, :, joint_idx*3:(joint_idx+1)*3]

    # Normalize
    X_norm = np.zeros_like(X_joints)
    for i in range(len(X_joints)):
        mean = np.nanmean(X_joints[i])
        std = np.nanstd(X_joints[i]) + 1e-8
        X_norm[i] = (X_joints[i] - mean) / std
    X_norm = np.nan_to_num(X_norm, nan=0.0)

    # Subsample frames
    X_subsampled = X_norm[:, ::4, :, :]  # (n, 60, 15, 3)
    print(f"Subsampled shape: {X_subsampled.shape}")

    # Get adjacency matrix
    adj = get_skeleton_adjacency().to(DEVICE)

    gkf = GroupKFold(n_splits=5)
    all_preds = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_subsampled, y, pids)):
        print(f"\nFold {fold+1}/5")

        X_train = torch.FloatTensor(X_subsampled[train_idx])
        y_train = torch.FloatTensor(y[train_idx])
        X_val = torch.FloatTensor(X_subsampled[val_idx])
        y_val = torch.FloatTensor(y[val_idx])

        # Custom dataset that includes adjacency
        class STGCNDataset(torch.utils.data.Dataset):
            def __init__(self, X, y, adj):
                self.X = X
                self.y = y
                self.adj = adj

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.adj, self.y[idx]

        train_dataset = STGCNDataset(X_train, y_train, adj.cpu())
        val_dataset = STGCNDataset(X_val, y_val, adj.cpu())

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        # Create model
        model = STGCN(
            in_channels=3,
            num_joints=n_joints,
            num_frames=60,
            hidden_channels=32,
            num_outputs=3,
            dropout=0.5
        )

        model, val_loss = train_pytorch_model(
            model, train_loader, val_loader,
            epochs=100, lr=0.001, weight_decay=1e-2,
            patience=15
        )

        # Predict
        model.eval()
        with torch.no_grad():
            preds = model(X_val.to(DEVICE), adj).cpu().numpy()

        all_preds[val_idx] = preds
        fold_mse = mean_squared_error(y[val_idx], preds)
        print(f"  Fold {fold+1}: MSE = {fold_mse:.6f}")

    # Results
    results = {}
    for i, target in enumerate(TARGETS):
        results[target] = mean_squared_error(y[:, i], all_preds[:, i])
    results['total'] = mean_squared_error(y, all_preds)

    print(f"\nST-GCN GroupKFold Results:")
    for target in TARGETS:
        print(f"  {target}: MSE = {results[target]:.6f}")
    print(f"  TOTAL: MSE = {results['total']:.6f}")

    return results


def run_mc_dropout_experiment():
    """Test Bayesian ensemble with MC Dropout."""
    print("\n" + "="*70)
    print("EXPERIMENT 5: MC DROPOUT BAYESIAN ENSEMBLE")
    print("="*70)

    # Load raw data and extract simple features
    print("\nLoading raw data...")
    X_raw, y, pids = load_raw_data(train=True)

    # Simple features: mean, std per feature
    X_features = np.concatenate([
        X_raw.mean(axis=1),
        X_raw.std(axis=1),
        X_raw.max(axis=1) - X_raw.min(axis=1),  # range
    ], axis=1)
    X_features = np.nan_to_num(X_features, nan=0.0)
    print(f"Feature shape: {X_features.shape}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    gkf = GroupKFold(n_splits=5)
    all_preds = np.zeros_like(y)
    all_stds = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, pids)):
        print(f"\nFold {fold+1}/5")

        X_train = torch.FloatTensor(X_scaled[train_idx])
        y_train = torch.FloatTensor(y[train_idx])
        X_val = torch.FloatTensor(X_scaled[val_idx])
        y_val = torch.FloatTensor(y[val_idx])

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Create model
        model = MCDropoutMLP(
            input_dim=X_scaled.shape[1],
            hidden_dims=[128, 64],
            num_outputs=3,
            dropout=0.3
        )

        model, val_loss = train_pytorch_model(
            model, train_loader, val_loader,
            epochs=200, lr=0.001, weight_decay=1e-3,
            patience=20
        )

        # MC Dropout prediction
        model = model.to(DEVICE)
        mean_pred, std_pred = model.predict_with_uncertainty(
            X_val.to(DEVICE), n_samples=50
        )

        all_preds[val_idx] = mean_pred.cpu().numpy()
        all_stds[val_idx] = std_pred.cpu().numpy()

        fold_mse = mean_squared_error(y[val_idx], mean_pred.cpu().numpy())
        print(f"  Fold {fold+1}: MSE = {fold_mse:.6f}, Mean Uncertainty = {std_pred.mean():.6f}")

    # Results
    results = {}
    for i, target in enumerate(TARGETS):
        results[target] = mean_squared_error(y[:, i], all_preds[:, i])
        results[f'{target}_uncertainty'] = all_stds[:, i].mean()
    results['total'] = mean_squared_error(y, all_preds)

    print(f"\nMC Dropout GroupKFold Results:")
    for target in TARGETS:
        print(f"  {target}: MSE = {results[target]:.6f}, Uncertainty = {results[f'{target}_uncertainty']:.6f}")
    print(f"  TOTAL: MSE = {results['total']:.6f}")

    # Analyze uncertainty-error correlation
    errors = np.abs(y - all_preds)
    for i, target in enumerate(TARGETS):
        corr = np.corrcoef(errors[:, i], all_stds[:, i])[0, 1]
        print(f"  {target} uncertainty-error correlation: {corr:.4f}")

    return results, all_preds, all_stds


def main():
    """Run all experiments."""
    print("="*70)
    print("UNTESTED APPROACHES EXPERIMENT")
    print("Testing: Signal Processing, TCN, Transformer, ST-GCN, MC Dropout")
    print("="*70)

    results = {}

    # 1. Signal Processing
    try:
        results['signal_processing'] = run_signal_processing_experiment()
    except Exception as e:
        print(f"Signal processing experiment failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. TCN
    try:
        results['tcn'] = run_tcn_experiment()
    except Exception as e:
        print(f"TCN experiment failed: {e}")
        import traceback
        traceback.print_exc()

    # 3. Transformer
    try:
        results['transformer'] = run_transformer_experiment()
    except Exception as e:
        print(f"Transformer experiment failed: {e}")
        import traceback
        traceback.print_exc()

    # 4. ST-GCN
    try:
        results['stgcn'] = run_stgcn_experiment()
    except Exception as e:
        print(f"ST-GCN experiment failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. MC Dropout
    try:
        mc_results, _, _ = run_mc_dropout_experiment()
        results['mc_dropout'] = mc_results
    except Exception as e:
        print(f"MC Dropout experiment failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    print("\nBaseline: Best LB = 0.007682 (Sub 219)")
    print("\nGroupKFold CV Results:")

    for exp_name, exp_results in results.items():
        if isinstance(exp_results, dict):
            if 'total' in exp_results:
                print(f"\n{exp_name}:")
                for target in TARGETS:
                    if target in exp_results:
                        print(f"  {target}: {exp_results[target]:.6f}")
                print(f"  TOTAL: {exp_results['total']:.6f}")
            elif 'combined' in exp_results:
                print(f"\n{exp_name}:")
                for sub_exp, sub_results in exp_results.items():
                    print(f"  {sub_exp}: {sub_results['total']:.6f}")

    return results


if __name__ == "__main__":
    main()

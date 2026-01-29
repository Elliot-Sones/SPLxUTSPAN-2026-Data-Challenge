"""
Creative Deep Learning models for the overnight experiment system.

Implements:
1. FrameAttentionModel: Learns which frames matter for each target
2. TargetSpecificModel: Uses different frame windows per target
3. SequenceAutoencoder: For hybrid DL+GBDT approach
4. MaskedFramePredictor: Self-supervised pre-training

All models are designed for:
- Small dataset (345 samples)
- Mac M2 MPS acceleration
- 240 frames x 207 features input
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Device selection for Mac M2
def get_device() -> torch.device:
    """Get best available device (MPS for Mac M2, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# 1. Frame Attention Model
# =============================================================================

class TemporalAttention(nn.Module):
    """Attention mechanism to learn which frames matter."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, features)

        Returns:
            weighted_sum: (batch, features)
            attention_weights: (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attention(x)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum
        weighted = (x * weights).sum(dim=1)  # (batch, features)

        return weighted, weights.squeeze(-1)


class FrameAttentionModel(nn.Module):
    """
    Model that learns which frames are important for prediction.

    Uses temporal attention to weight frames, then MLP for prediction.
    Attention weights can be visualized to understand frame importance.
    """

    def __init__(
        self,
        n_features: int = 207,
        hidden_dim: int = 64,
        attention_dim: int = 32,
        n_targets: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Temporal attention
        self.attention = TemporalAttention(hidden_dim, attention_dim)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_targets),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, n_features) raw sequence
            return_attention: Whether to return attention weights

        Returns:
            predictions: (batch, n_targets)
            attention_weights: (batch, seq_len) if return_attention
        """
        # Project features
        projected = self.feature_proj(x)  # (batch, seq_len, hidden)

        # Apply attention
        weighted, weights = self.attention(projected)

        # Predict
        predictions = self.head(weighted)

        if return_attention:
            return predictions, weights
        return predictions, None


# =============================================================================
# 2. Target-Specific Model
# =============================================================================

class TargetSpecificModel(nn.Module):
    """
    Model that uses different frame windows for each target.

    Based on R^2 analysis:
    - depth: frames 50-150 (peaks at 102)
    - angle: frames 100-175 (peaks at 153)
    - left_right: frames 175-240 (peaks at 237)
    """

    def __init__(
        self,
        n_features: int = 207,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Target-specific windows
        self.windows = {
            "depth": (50, 150),
            "angle": (100, 175),
            "left_right": (175, 240),
        }

        # Per-target encoders
        self.encoders = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        for target, (start, end) in self.windows.items():
            window_len = end - start

            # Simple 1D CNN encoder for each target
            self.encoders[target] = nn.Sequential(
                nn.Conv1d(n_features, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

            self.heads[target] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features) raw sequence

        Returns:
            Dict mapping target name to (batch, 1) predictions
        """
        predictions = {}

        for target, (start, end) in self.windows.items():
            # Extract window
            window = x[:, start:end, :]  # (batch, window_len, features)

            # CNN expects (batch, channels, seq_len)
            window = window.transpose(1, 2)

            # Encode
            encoded = self.encoders[target](window)  # (batch, hidden, 1)
            encoded = encoded.squeeze(-1)  # (batch, hidden)

            # Predict
            predictions[target] = self.heads[target](encoded)

        return predictions


# =============================================================================
# 3. Sequence Autoencoder (for hybrid DL+GBDT)
# =============================================================================

class SequenceAutoencoder(nn.Module):
    """
    Autoencoder for learning sequence representations.

    Can be trained on ALL data (train + test) since it's unsupervised.
    Bottleneck features can then be used with GBDT.
    """

    def __init__(
        self,
        n_features: int = 207,
        seq_len: int = 240,
        bottleneck_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.bottleneck_dim = bottleneck_dim

        # Encoder: sequence -> bottleneck
        self.encoder = nn.Sequential(
            # Flatten temporal dimension
            nn.Flatten(),  # (batch, seq_len * n_features)
            nn.Linear(seq_len * n_features, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        # Decoder: bottleneck -> sequence
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, seq_len * n_features),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to bottleneck representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck to sequence."""
        decoded = self.decoder(z)
        return decoded.view(-1, self.seq_len, self.n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            reconstructed: (batch, seq_len, n_features)
            bottleneck: (batch, bottleneck_dim)
        """
        bottleneck = self.encode(x)
        reconstructed = self.decode(bottleneck)
        return reconstructed, bottleneck


class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder - better for temporal patterns.

    Learns hierarchical temporal features through convolution.
    """

    def __init__(
        self,
        n_features: int = 207,
        seq_len: int = 240,
        bottleneck_dim: int = 32,
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.bottleneck_dim = bottleneck_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            # (batch, features, seq_len) -> (batch, 64, 120)
            nn.Conv1d(n_features, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # -> (batch, 32, 60)
            nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            # -> (batch, 16, 30)
            nn.Conv1d(32, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )

        # Calculate flattened size: 16 channels * 30 time steps
        self.flatten_size = 16 * 30

        self.encoder_fc = nn.Linear(self.flatten_size, bottleneck_dim)

        # Decoder
        self.decoder_fc = nn.Linear(bottleneck_dim, self.flatten_size)

        self.decoder_conv = nn.Sequential(
            # (batch, 16, 30) -> (batch, 32, 60)
            nn.ConvTranspose1d(16, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            # -> (batch, 64, 120)
            nn.ConvTranspose1d(32, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # -> (batch, features, 240)
            nn.ConvTranspose1d(64, n_features, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to bottleneck."""
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        conv_out = self.encoder_conv(x)  # (batch, 16, 30)
        flat = conv_out.flatten(1)  # (batch, 480)
        return self.encoder_fc(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck to sequence."""
        x = self.decoder_fc(z)  # (batch, 480)
        x = x.view(-1, 16, 30)  # (batch, 16, 30)
        x = self.decoder_conv(x)  # (batch, features, seq_len)
        return x.transpose(1, 2)  # (batch, seq_len, features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bottleneck = self.encode(x)
        reconstructed = self.decode(bottleneck)
        return reconstructed, bottleneck


# =============================================================================
# 4. Masked Frame Predictor (Self-Supervised)
# =============================================================================

class MaskedFramePredictor(nn.Module):
    """
    Self-supervised model that predicts masked frames.

    Pre-training task: Mask 20% of frames, predict them.
    Can use unlabeled test.csv data for pre-training.
    """

    def __init__(
        self,
        n_features: int = 207,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        mask_ratio: float = 0.2,
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, n_features))

        # Frame encoder
        self.frame_encoder = nn.Linear(n_features, hidden_dim)

        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Frame decoder
        self.frame_decoder = nn.Linear(hidden_dim, n_features)

    def create_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create random mask for frames."""
        mask = torch.rand(batch_size, seq_len, device=device) < self.mask_ratio
        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features)
            mask: Optional (batch, seq_len) boolean mask

        Returns:
            reconstructed: (batch, seq_len, n_features)
            mask: (batch, seq_len) the mask used
        """
        batch_size, seq_len, _ = x.shape

        # Create mask if not provided
        if mask is None:
            mask = self.create_mask(batch_size, seq_len, x.device)

        # Replace masked frames with mask token
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
        masked_input = torch.where(mask_expanded, mask_tokens, x)

        # Encode frames
        encoded = self.frame_encoder(masked_input)

        # Transform
        transformed = self.transformer(encoded)

        # Decode frames
        reconstructed = self.frame_decoder(transformed)

        return reconstructed, mask

    def pretrain_loss(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss only on masked frames."""
        reconstructed, mask = self.forward(x, mask)

        # Loss only on masked positions
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        masked_reconstruction = reconstructed[mask_expanded]
        masked_target = x[mask_expanded]

        return F.mse_loss(masked_reconstruction, masked_target)


# =============================================================================
# 5. Simple MLP Baseline
# =============================================================================

class SimpleMLP(nn.Module):
    """
    Simple MLP baseline that flattens the sequence.

    For comparison with attention-based models.
    """

    def __init__(
        self,
        n_features: int = 207,
        seq_len: int = 240,
        hidden_dims: List[int] = [256, 128, 64],
        n_targets: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        input_dim = n_features * seq_len
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_targets))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            predictions: (batch, n_targets)
        """
        flat = x.flatten(1)
        return self.network(flat)


# =============================================================================
# Training Utilities
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 20,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train a model with early stopping.

    Returns:
        History dict with train_loss and val_loss lists
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            X, y = batch
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            # Handle different model outputs
            output = model(X)
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, dict):
                # TargetSpecificModel returns dict
                loss = sum(F.mse_loss(v, y[:, i:i+1])
                          for i, (k, v) in enumerate(sorted(output.items())))
            else:
                loss = F.mse_loss(output, y)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    X, y = batch
                    X, y = X.to(device), y.to(device)

                    output = model(X)
                    if isinstance(output, tuple):
                        output = output[0]
                    if isinstance(output, dict):
                        loss = sum(F.mse_loss(v, y[:, i:i+1])
                                  for i, (k, v) in enumerate(sorted(output.items())))
                    else:
                        loss = F.mse_loss(output, y)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            history["val_loss"].append(avg_val_loss)
            scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: train={avg_train_loss:.6f}, val={avg_val_loss:.6f}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: train={avg_train_loss:.6f}")

    return history


def pretrain_autoencoder(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> List[float]:
    """
    Pre-train an autoencoder (unsupervised).

    Can use ALL data including test set.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                X = batch[0]
            else:
                X = batch
            X = X.to(device)

            optimizer.zero_grad()

            reconstructed, _ = model(X)
            loss = F.mse_loss(reconstructed, X)

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: reconstruction_loss={avg_loss:.6f}")

    return losses


def extract_bottleneck_features(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Extract bottleneck features from a trained autoencoder.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    all_features = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                X = batch[0]
            else:
                X = batch
            X = X.to(device)

            bottleneck = model.encode(X)
            all_features.append(bottleneck.cpu().numpy())

    return np.concatenate(all_features, axis=0)


if __name__ == "__main__":
    print("Testing creative DL models...")
    device = get_device()
    print(f"Device: {device}")

    # Test data
    batch_size = 16
    seq_len = 240
    n_features = 207

    X = torch.randn(batch_size, seq_len, n_features)
    y = torch.randn(batch_size, 3)

    # Test FrameAttentionModel
    print("\n1. FrameAttentionModel:")
    model = FrameAttentionModel()
    out, attn = model(X, return_attention=True)
    print(f"   Output shape: {out.shape}")
    print(f"   Attention shape: {attn.shape}")

    # Test TargetSpecificModel
    print("\n2. TargetSpecificModel:")
    model = TargetSpecificModel()
    out = model(X)
    for k, v in out.items():
        print(f"   {k}: {v.shape}")

    # Test SequenceAutoencoder
    print("\n3. SequenceAutoencoder:")
    model = SequenceAutoencoder()
    recon, bottleneck = model(X)
    print(f"   Reconstructed shape: {recon.shape}")
    print(f"   Bottleneck shape: {bottleneck.shape}")

    # Test ConvAutoencoder
    print("\n4. ConvAutoencoder:")
    model = ConvAutoencoder()
    recon, bottleneck = model(X)
    print(f"   Reconstructed shape: {recon.shape}")
    print(f"   Bottleneck shape: {bottleneck.shape}")

    # Test MaskedFramePredictor
    print("\n5. MaskedFramePredictor:")
    model = MaskedFramePredictor()
    recon, mask = model(X)
    print(f"   Reconstructed shape: {recon.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Masked frames: {mask.sum().item()} / {mask.numel()}")

    # Test SimpleMLP
    print("\n6. SimpleMLP:")
    model = SimpleMLP()
    out = model(X)
    print(f"   Output shape: {out.shape}")

    print("\nAll model tests passed!")

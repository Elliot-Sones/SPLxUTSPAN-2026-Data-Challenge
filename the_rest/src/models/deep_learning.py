"""
Deep Learning models for SPLxUTSPAN 2026 Data Challenge.

Provides CNN-LSTM and Transformer architectures for time series regression.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle


class ShotDataset(Dataset):
    """PyTorch Dataset for shot time series data."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Args:
            X: (n_samples, n_timesteps, n_features) time series data
            y: (n_samples, 3) targets [angle, depth, left_right]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class CNNLSTM(nn.Module):
    """CNN-LSTM hybrid for time series regression."""

    def __init__(
        self,
        n_features: int = 207,
        n_timesteps: int = 240,
        cnn_channels: List[int] = [64, 128, 256],
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_timesteps = n_timesteps

        # CNN layers for local pattern extraction
        cnn_layers = []
        in_channels = n_features
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate CNN output size
        cnn_out_time = n_timesteps // (2 ** len(cnn_channels))

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # angle, depth, left_right
        )

    def forward(self, x):
        # x: (batch, timesteps, features)
        # Conv1d expects (batch, features, timesteps)
        x = x.permute(0, 2, 1)

        # CNN feature extraction
        x = self.cnn(x)

        # Back to (batch, timesteps, features) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM temporal modeling
        x, _ = self.lstm(x)

        # Take last timestep output
        x = x[:, -1, :]

        # Output
        return self.fc(x)


class TransformerModel(nn.Module):
    """Transformer encoder for time series regression."""

    def __init__(
        self,
        n_features: int = 207,
        n_timesteps: int = 240,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_timesteps = n_timesteps

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, n_timesteps, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        # x: (batch, timesteps, features)

        # Project to d_model dimensions
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Output
        return self.fc(x)


class DeepLearningTrainer:
    """Trainer for deep learning models."""

    def __init__(
        self,
        model_type: str = "cnn_lstm",
        device: str = "auto",
        **model_kwargs
    ):
        self.model_type = model_type

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_kwargs = model_kwargs
        self.model = None
        self.history = {"train_loss": [], "val_loss": []}

    def _create_model(self, n_features: int, n_timesteps: int) -> nn.Module:
        if self.model_type == "cnn_lstm":
            return CNNLSTM(n_features=n_features, n_timesteps=n_timesteps, **self.model_kwargs)
        elif self.model_type == "transformer":
            return TransformerModel(n_features=n_features, n_timesteps=n_timesteps, **self.model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
    ):
        """Train the model."""
        n_timesteps, n_features = X_train.shape[1], X_train.shape[2]

        # Create model
        self.model = self._create_model(n_features, n_timesteps).to(self.device)

        # Data loaders
        train_dataset = ShotDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            val_dataset = ShotDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer and loss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_dataset)
            self.history["train_loss"].append(train_loss)

            # Validation
            if X_val is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        pred = self.model(X_batch)
                        val_loss += criterion(pred, y_batch).item() * len(X_batch)

                val_loss /= len(val_dataset)
                self.history["val_loss"].append(val_loss)
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets."""
        self.model.eval()
        dataset = ShotDataset(X)
        loader = DataLoader(dataset, batch_size=64)

        predictions = []
        with torch.no_grad():
            for X_batch in loader:
                if isinstance(X_batch, tuple):
                    X_batch = X_batch[0]
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                predictions.append(pred.cpu().numpy())

        return np.vstack(predictions)

    def save(self, filepath: Path):
        """Save model."""
        torch.save({
            "model_type": self.model_type,
            "model_kwargs": self.model_kwargs,
            "state_dict": self.model.state_dict(),
            "history": self.history,
        }, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "DeepLearningTrainer":
        """Load model."""
        checkpoint = torch.load(filepath, map_location="cpu")
        trainer = cls(model_type=checkpoint["model_type"], **checkpoint["model_kwargs"])
        trainer.history = checkpoint["history"]
        # Model will be created when predict is called with actual data dimensions
        trainer._saved_state = checkpoint["state_dict"]
        return trainer

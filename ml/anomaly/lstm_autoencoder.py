"""
MaintenAI— PyTorch LSTM Autoencoder Anomaly Detection Model

Deep learning reconstruction-based anomaly detector for multivariate time-series.
Reconstructs sliding sequence windows and scores anomalies via reconstruction error MSE.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # return last layer hidden state: (batch_size, hidden_dim)
        return h_n[-1]


class _Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (batch_size, hidden_dim)
        # repeat across seq_len
        repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(repeated)
        reconstruction = self.fc(out)  # (batch_size, seq_len, output_dim)
        return reconstruction


class _LSTMAutoencoderNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        seq_len: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = _Encoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = _Decoder(hidden_dim, input_dim, seq_len, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class LSTMAutoencoderDetector:
    """
    LSTM Autoencoder Anomaly Detector for sequential sensor windows.

    Attributes:
        sequence_length: Sliding window sequence length (timesteps).
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between LSTM layers.
        learning_rate: Optimizer learning rate.
        epochs: Max training epochs.
        batch_size: Batch size for training DataLoader.
        patience: Early stopping patience (epochs without validation loss improvement).
        reconstruction_percentile: Percentile threshold on training reconstruction error (e.g. 95).
    """

    def __init__(
        self,
        sequence_length: int = 50,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        reconstruction_percentile: float = 95.0,
        device: str | None = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.reconstruction_percentile = reconstruction_percentile

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model: _LSTMAutoencoderNet | None = None
        self.input_dim: int | None = None
        self.feature_names: list[str] = []
        self.threshold_: float = 0.5
        self.loss_min_: float = 0.0
        self.loss_max_: float = 1.0

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sliding window sequences of shape (num_sequences, seq_len, num_features)."""
        if len(data) < self.sequence_length:
            # Pad with first row if shorter than seq_len
            pad = np.repeat(data[:1], self.sequence_length - len(data), axis=0)
            data = np.vstack([pad, data])

        num_seq = len(data) - self.sequence_length + 1
        seqs = np.zeros((num_seq, self.sequence_length, data.shape[1]), dtype=np.float32)
        for i in range(num_seq):
            seqs[i] = data[i : i + self.sequence_length]
        return seqs

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> LSTMAutoencoderDetector:
        """Fit LSTM Autoencoder on training sequences."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            train_mat = X_train.values
        else:
            self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]
            train_mat = X_train

        train_mat = np.nan_to_num(train_mat, nan=0.0, posinf=0.0, neginf=0.0)
        self.input_dim = train_mat.shape[1]

        seqs_train = self._create_sequences(train_mat)
        tensor_train = torch.tensor(seqs_train, dtype=torch.float32)
        dataset_train = TensorDataset(tensor_train, tensor_train)
        loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)

        loader_val = None
        if X_val is not None:
            val_mat = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            val_mat = np.nan_to_num(val_mat, nan=0.0, posinf=0.0, neginf=0.0)
            seqs_val = self._create_sequences(val_mat)
            tensor_val = torch.tensor(seqs_val, dtype=torch.float32)
            dataset_val = TensorDataset(tensor_val, tensor_val)
            loader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False)

        self.model = _LSTMAutoencoderNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            seq_len=self.sequence_length,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, _ in loader_train:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            epoch_loss /= len(dataset_train)

            # Validation
            val_loss = epoch_loss
            if loader_val is not None:
                self.model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for batch_v, _ in loader_val:
                        batch_v = batch_v.to(self.device)
                        recon_v = self.model(batch_v)
                        loss_v = criterion(recon_v, batch_v)
                        val_loss_sum += loss_v.item() * len(batch_v)
                val_loss = val_loss_sum / len(dataset_val)
                self.model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Calibrate reconstruction errors
        self.model.eval()
        train_errors = self._compute_reconstruction_errors(seqs_train)
        self.loss_min_ = float(np.percentile(train_errors, 1))
        self.loss_max_ = float(np.percentile(train_errors, 99))
        if self.loss_max_ <= self.loss_min_:
            self.loss_max_ = self.loss_min_ + 1.0

        self.threshold_ = float(
            np.percentile(self._normalize(train_errors), self.reconstruction_percentile)
        )
        logger.info(
            "LSTM Autoencoder trained. Min error: %.4f, Max error: %.4f, Threshold: %.4f",
            self.loss_min_,
            self.loss_max_,
            self.threshold_,
        )
        return self

    def _compute_reconstruction_errors(self, seqs: np.ndarray) -> np.ndarray:
        """Compute MSE reconstruction error per sequence."""
        self.model.eval()
        errors = []
        with torch.no_grad():
            tensor = torch.tensor(seqs, dtype=torch.float32)
            loader = DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=False)
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                recon = self.model(batch_x)
                # MSE per sample across sequence & features
                mse = torch.mean((recon - batch_x) ** 2, dim=(1, 2)).cpu().numpy()
                errors.extend(mse.tolist())
        return np.array(errors)

    def _normalize(self, raw_errors: np.ndarray) -> np.ndarray:
        norm = (raw_errors - self.loss_min_) / (self.loss_max_ - self.loss_min_ + 1e-8)
        return np.clip(norm, 0.0, 1.0)

    def predict_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Compute anomaly score for each row in X using trailing sequence reconstruction.
        """
        if self.model is None or self.input_dim is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        mat = X.values if isinstance(X, pd.DataFrame) else X
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

        seqs = self._create_sequences(mat)
        errors = self._compute_reconstruction_errors(seqs)
        scores = self._normalize(errors)

        # Pad beginning to match input length
        if len(scores) < len(mat):
            pad = np.repeat(scores[0:1], len(mat) - len(scores), axis=0)
            scores = np.concatenate([pad, scores])
        return scores

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Predict binary anomaly flags (0 = Normal, 1 = Anomaly)."""
        scores = self.predict_score(X)
        th = threshold if threshold is not None else self.threshold_
        return (scores >= th).astype(int)

    def save(self, path: Path | str) -> None:
        """Save model checkpoint and metadata."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict() if self.model else None,
            "input_dim": self.input_dim,
            "feature_names": self.feature_names,
            "loss_min": self.loss_min_,
            "loss_max": self.loss_max_,
            "threshold": self.threshold_,
            "params": {
                "sequence_length": self.sequence_length,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "patience": self.patience,
                "reconstruction_percentile": self.reconstruction_percentile,
            },
        }
        joblib.dump(payload, target_path)
        logger.info("Saved LSTM Autoencoder model to %s", target_path)

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> LSTMAutoencoderDetector:
        """Load model from disk."""
        target_path = Path(path)
        payload = joblib.load(target_path)
        detector = cls(**payload.get("params", {}), device=device)
        detector.input_dim = payload["input_dim"]
        detector.feature_names = payload.get("feature_names", [])
        detector.loss_min_ = payload.get("loss_min", 0.0)
        detector.loss_max_ = payload.get("loss_max", 1.0)
        detector.threshold_ = payload.get("threshold", 0.5)

        if payload.get("state_dict") is not None and detector.input_dim is not None:
            detector.model = _LSTMAutoencoderNet(
                input_dim=detector.input_dim,
                hidden_dim=detector.hidden_dim,
                seq_len=detector.sequence_length,
                num_layers=detector.num_layers,
                dropout=detector.dropout,
            ).to(detector.device)
            detector.model.load_state_dict(payload["state_dict"])
            detector.model.eval()

        return detector

"""
MaintenAI— PyTorch Transformer RUL Prediction Model

Sequence-to-scalar Transformer encoder for predicting Remaining Useful Life (RUL)
from multivariate sensor degradation trajectories.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class _TransformerRULNet(nn.Module):
    """Transformer Encoder for RUL sequence regression."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        projected = self.input_projection(x)
        encoded = self.pos_encoder(projected)
        out = self.transformer_encoder(encoded)
        # Use mean pooling across time dimension
        pooled = torch.mean(out, dim=1)
        rul = self.head(pooled)
        return rul.squeeze(-1)


class TransformerRULRegressor:
    """
    Transformer RUL Regressor training and inference wrapper.
    """

    def __init__(
        self,
        sequence_length: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 0.0005,
        epochs: int = 100,
        batch_size: int = 128,
        patience: int = 15,
        scheduler: str = "cosine",
        device: str | None = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.scheduler = scheduler

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model: _TransformerRULNet | None = None
        self.input_dim: int | None = None
        self.feature_names: list[str] = []

    def _create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str | None = "rul",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Create sliding sequences grouped by machine_id to prevent boundary mixing."""
        X_seqs = []
        y_vals = []

        machine_col = "machine_id" if "machine_id" in df.columns else None
        groups = df.groupby(machine_col) if machine_col else [(0, df)]

        for _, group in groups:
            mat = group[feature_cols].values
            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            targets = (
                group[target_col].values if target_col and target_col in group.columns else None
            )

            if len(mat) < self.sequence_length:
                pad = np.repeat(mat[:1], self.sequence_length - len(mat), axis=0)
                mat = np.vstack([pad, mat])
                if targets is not None:
                    targets = np.pad(targets, (self.sequence_length - len(targets), 0), mode="edge")

            num_seq = len(mat) - self.sequence_length + 1
            for i in range(num_seq):
                X_seqs.append(mat[i : i + self.sequence_length])
                if targets is not None:
                    y_vals.append(targets[i + self.sequence_length - 1])

        X_arr = np.array(X_seqs, dtype=np.float32)
        y_arr = np.array(y_vals, dtype=np.float32) if y_vals else None
        return X_arr, y_arr

    def fit(
        self,
        df_train: pd.DataFrame,
        feature_cols: list[str],
        df_val: pd.DataFrame | None = None,
        target_col: str = "rul",
    ) -> TransformerRULRegressor:
        """Fit Transformer model on time-series sequences."""
        self.feature_names = list(feature_cols)
        self.input_dim = len(self.feature_names)

        X_tr, y_tr = self._create_sequences(df_train, feature_cols, target_col)
        dataset_train = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)

        loader_val = None
        if df_val is not None:
            X_v, y_v = self._create_sequences(df_val, feature_cols, target_col)
            dataset_val = TensorDataset(torch.tensor(X_v), torch.tensor(y_v))
            loader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False)

        self.model = _TransformerRULNet(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.HuberLoss(delta=10.0)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        if self.scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=1e-6
            )
        else:
            lr_scheduler = None

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader_train:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            epoch_loss /= len(dataset_train)
            if lr_scheduler:
                lr_scheduler.step()

            # Validation
            val_loss = epoch_loss
            if loader_val is not None:
                self.model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for batch_vx, batch_vy in loader_val:
                        batch_vx, batch_vy = batch_vx.to(self.device), batch_vy.to(self.device)
                        vpred = self.model(batch_vx)
                        vloss = criterion(vpred, batch_vy)
                        val_loss_sum += vloss.item() * len(batch_vx)
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

        logger.info("Transformer RUL training complete. Best Val Loss: %.4f", best_val_loss)
        return self

    def predict(self, df_or_X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict Remaining Useful Life."""
        if self.model is None or self.input_dim is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

        self.model.eval()
        if isinstance(df_or_X, pd.DataFrame):
            X_seqs, _ = self._create_sequences(df_or_X, self.feature_names, target_col=None)
        else:
            # Array shape: (num_seq, seq_len, num_features) or (num_samples, num_features)
            if df_or_X.ndim == 2:
                # Wrap as 2D sequence window
                if len(df_or_X) < self.sequence_length:
                    pad = np.repeat(df_or_X[:1], self.sequence_length - len(df_or_X), axis=0)
                    mat = np.vstack([pad, df_or_X])
                else:
                    mat = df_or_X
                num_seq = len(mat) - self.sequence_length + 1
                X_seqs = np.array(
                    [mat[i : i + self.sequence_length] for i in range(num_seq)], dtype=np.float32
                )
            else:
                X_seqs = df_or_X.astype(np.float32)

        predictions = []
        with torch.no_grad():
            tensor = torch.tensor(X_seqs, dtype=torch.float32)
            loader = DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=False)
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x)
                predictions.extend(preds.cpu().numpy().tolist())

        preds_arr = np.maximum(np.array(predictions), 0.0)

        # If evaluating row-by-row on dataframe, align length
        if isinstance(df_or_X, pd.DataFrame) and len(preds_arr) < len(df_or_X):
            pad = np.repeat(preds_arr[0:1], len(df_or_X) - len(preds_arr), axis=0)
            preds_arr = np.concatenate([pad, preds_arr])

        return preds_arr

    def save(self, path: Path | str) -> None:
        """Save model weights and hyperparameters."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict() if self.model else None,
            "input_dim": self.input_dim,
            "feature_names": self.feature_names,
            "params": {
                "sequence_length": self.sequence_length,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_encoder_layers": self.num_encoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "patience": self.patience,
                "scheduler": self.scheduler,
            },
        }
        joblib.dump(payload, target_path)
        logger.info("Saved Transformer RUL model to %s", target_path)

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> TransformerRULRegressor:
        """Load model from disk."""
        target_path = Path(path)
        payload = joblib.load(target_path)
        regressor = cls(**payload.get("params", {}), device=device)
        regressor.input_dim = payload["input_dim"]
        regressor.feature_names = payload.get("feature_names", [])

        if payload.get("state_dict") is not None and regressor.input_dim is not None:
            regressor.model = _TransformerRULNet(
                input_dim=regressor.input_dim,
                d_model=regressor.d_model,
                nhead=regressor.nhead,
                num_encoder_layers=regressor.num_encoder_layers,
                dim_feedforward=regressor.dim_feedforward,
                dropout=regressor.dropout,
            ).to(regressor.device)
            regressor.model.load_state_dict(payload["state_dict"])
            regressor.model.eval()

        return regressor

"""LSTM-AD detector (Malhotra et al., 2015): forecast-error Mahalanobis scoring."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import BaseTimeSeriesAnomalyDetector, log_epoch, rolling_windows_nd, subsample_windows


class LSTMADNet(nn.Module):
    """Stacked LSTM that predicts the next `horizon` steps from a history window."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, horizon: int, dropout: float) -> None:
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        prediction = self.head(hidden[-1])
        return prediction.view(x.shape[0], self.horizon, self.input_dim)


class LSTMAD(BaseTimeSeriesAnomalyDetector):
    """Classic LSTM-AD detector (Malhotra et al., 2015): a multi-step LSTM forecaster
    whose per-timestep prediction-error vectors are scored against a Gaussian fitted
    on the training errors, using the squared Mahalanobis distance as the anomaly score.
    """

    def __init__(
        self,
        window_size: int = 100,
        horizon: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        batch_size: int = 64,
        num_epochs: int = 40,
        learning_rate: float = 1e-3,
        patience: int = 6,
        ridge: float = 1e-6,
        max_windows: int = 4096,
        device: str | None = None,
        seed: int = 13,
    ) -> None:
        super().__init__(window_size=window_size, device=device, seed=seed)
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.ridge = ridge
        self.max_windows = max_windows
        self.net: LSTMADNet | None = None
        self.error_mean_: np.ndarray | None = None
        self.error_cov_inv_: np.ndarray | None = None

    def _resolve_training_stride(self, num_raw_windows: int) -> int:
        """Mirror CARLABase's window-count cap so training cost stops scaling with series length."""
        return max(1, -(-num_raw_windows // self.max_windows))

    def _fit_normalized(self, train_values: np.ndarray) -> None:
        """Train the LSTM forecaster, then fit a Gaussian on its prediction errors."""
        min_length = self.window_size + 2 * self.horizon - 1
        if train_values.shape[0] < min_length:
            raise ValueError(
                f"Series length {train_values.shape[0]} is too short for window_size={self.window_size} "
                f"and horizon={self.horizon}; need at least {min_length} points"
            )
        input_dim = train_values.shape[1]
        num_raw_windows = train_values.shape[0] - (self.window_size + self.horizon) + 1
        training_stride = self._resolve_training_stride(num_raw_windows)
        combined = rolling_windows_nd(train_values, self.window_size + self.horizon, stride=training_stride)
        combined = subsample_windows(combined, self.max_windows, self.seed)
        windows = combined[:, : self.window_size]
        targets = combined[:, self.window_size :]

        net = LSTMADNet(input_dim, self.hidden_size, self.num_layers, self.horizon, self.dropout).to(self.device)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(windows), torch.from_numpy(targets)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        best_loss = float("inf")
        best_state = None
        bad_epochs = 0
        epoch = 0
        for epoch in range(self.num_epochs):
            net.train()
            epoch_losses = []
            for batch_windows, batch_targets in loader:
                batch_windows = batch_windows.to(self.device)
                batch_targets = batch_targets.to(self.device)
                optimizer.zero_grad()
                predictions = net(batch_windows)
                loss = F.mse_loss(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            average_loss = float(np.mean(epoch_losses))
            log_epoch(self.__class__.__name__, epoch, self.num_epochs, average_loss)
            if average_loss < best_loss:
                best_loss = average_loss
                best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= self.patience:
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net.eval()
        self.training_summary_ = {"loss": float(best_loss), "epochs_trained": epoch + 1}

        train_errors, train_valid = self._error_vectors(train_values)
        valid_errors = train_errors[train_valid].astype(np.float64)
        self.error_mean_ = valid_errors.mean(axis=0)
        centered = valid_errors - self.error_mean_
        covariance = (centered.T @ centered) / max(1, len(valid_errors) - 1)
        covariance = covariance + self.ridge * np.eye(covariance.shape[0], dtype=np.float64)
        self.error_cov_inv_ = np.linalg.inv(covariance)

    def _score_normalized(self, values: np.ndarray) -> np.ndarray:
        """Score each timestep by the squared Mahalanobis distance of its error vector."""
        if self.net is None or self.error_mean_ is None or self.error_cov_inv_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        errors, valid = self._error_vectors(values)
        centered = errors.astype(np.float64) - self.error_mean_
        distances = np.einsum("ij,jk,ik->i", centered, self.error_cov_inv_, centered)
        return np.where(valid, distances, 0.0).astype(np.float32)

    def _predict_windows(self, windows: np.ndarray) -> np.ndarray:
        """Run the trained LSTM over history windows in eval mode, batched."""
        if self.net is None:
            raise RuntimeError("Model must be fitted before scoring")
        loader = DataLoader(torch.from_numpy(windows), batch_size=self.batch_size, shuffle=False)
        outputs = []
        self.net.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                outputs.append(self.net(batch).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _error_vectors(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-timestep stacked multi-horizon errors and their validity mask."""
        series_length, input_dim = values.shape
        combined = rolling_windows_nd(values, self.window_size + self.horizon)
        windows = combined[:, : self.window_size]
        predictions = self._predict_windows(windows)
        num_starts = windows.shape[0]

        errors = np.full((series_length, self.horizon * input_dim), np.nan, dtype=np.float64)
        for lookahead in range(self.horizon):
            target_indices = np.arange(num_starts) + self.window_size + lookahead
            column_slice = slice(lookahead * input_dim, (lookahead + 1) * input_dim)
            errors[target_indices, column_slice] = values[target_indices] - predictions[:, lookahead]
        valid = ~np.isnan(errors).any(axis=1)
        return np.nan_to_num(errors, nan=0.0), valid

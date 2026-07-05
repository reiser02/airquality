"""Shared infrastructure for the deep windowed detectors.

Provides the :class:`BaseTimeSeriesAnomalyDetector` fit/score contract
(standardize → ``_fit_normalized``/``_score_normalized``), windowing and
score-aggregation helpers, RNG seeding, progress logging, and the
:class:`GenIASWindowGenerator` used by the ``*GenIAS`` detector variants to
synthesize anomalous training windows.
"""

from __future__ import annotations

import logging
from multiprocessing import parent_process
import random
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .genias import GenIAS, GeniasLossConfig, compute_losses, patch_anomalies


logger = logging.getLogger("airquality.anomaly")

_PROGRESS_DISABLE: bool | None = None


def set_progress_settings(*, disable: bool | None = None) -> None:
    """Force training-progress logging on/off (``None`` = auto: off in subprocesses)."""
    global _PROGRESS_DISABLE
    _PROGRESS_DISABLE = disable


def progress_enabled() -> bool:
    """Whether training-progress logs are active (auto-off in worker processes)."""
    if _PROGRESS_DISABLE is not None:
        return not _PROGRESS_DISABLE
    return parent_process() is None


def log_epoch(desc: str, epoch: int, total: int, loss: float, *, every: int = 10) -> None:
    """Log a training-progress line every ``every`` epochs (plus first and last)."""
    if not progress_enabled():
        return
    if epoch == 0 or (epoch + 1) % every == 0 or (epoch + 1) == total:
        logger.info("    %s  epoch %d/%d  loss=%.4f", desc, epoch + 1, total, loss)


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible training runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_2d(values: np.ndarray) -> np.ndarray:
    """Convert univariate arrays to shape ``[time, features]`` and validate rank."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}")


def rolling_windows_nd(values: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """Slice a multivariate series into overlapping fixed-length windows."""

    if values.shape[0] < window_size:
        raise ValueError(f"Series length {values.shape[0]} is shorter than window size {window_size}")
    starts = range(0, values.shape[0] - window_size + 1, stride)
    windows = [values[start:start + window_size] for start in starts]
    return np.stack(windows).astype(np.float32, copy=False)


def subsample_windows(windows: np.ndarray, max_windows: int, seed: int) -> np.ndarray:
    """Cap training-window count so per-epoch cost doesn't scale with series length."""

    if windows.shape[0] <= max_windows:
        return windows
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(windows.shape[0], size=max_windows, replace=False))
    return windows[indices]


def aggregate_strided_scores(window_scores: np.ndarray, series_length: int, window_size: int, stride: int) -> np.ndarray:
    """Average strided window scores back onto the original timeline."""

    totals = np.zeros(series_length, dtype=np.float64)
    counts = np.zeros(series_length, dtype=np.float64)
    for index, score in enumerate(window_scores):
        start = index * stride
        end = min(start + window_size, series_length)
        totals[start:end] += float(score)
        counts[start:end] += 1.0
    counts[counts == 0.0] = 1.0
    return (totals / counts).astype(np.float32)


def aggregate_tail_scores(window_scores: np.ndarray, series_length: int, window_size: int, tail_length: int) -> np.ndarray:
    """Average window scores onto the trailing portion of each overlapping window."""

    totals = np.zeros(series_length, dtype=np.float64)
    counts = np.zeros(series_length, dtype=np.float64)
    clamped_tail_length = max(1, min(tail_length, window_size))
    for start, score in enumerate(window_scores):
        end = min(start + window_size, series_length)
        tail_start = max(start, end - clamped_tail_length)
        totals[tail_start:end] += float(score)
        counts[tail_start:end] += 1.0
    counts[counts == 0.0] = 1.0
    return (totals / counts).astype(np.float32)


def fit_standardizer_nd(train_values: np.ndarray) -> StandardScaler:
    """Fit a per-feature scaler using sklearn's StandardScaler."""

    scaler = StandardScaler()
    scaler.fit(np.asarray(train_values, dtype=np.float32))
    return scaler


def transform_standardize_nd(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply featurewise scaling with shape-preserving support for window arrays."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 2:
        return scaler.transform(array).astype(np.float32)
    if array.ndim == 3:
        original_shape = array.shape
        flattened = array.reshape(-1, original_shape[-1])
        transformed = scaler.transform(flattened)
        return transformed.reshape(original_shape).astype(np.float32)
    raise ValueError(f"Expected a 2D or 3D array, got shape {array.shape}")


class BaseTimeSeriesAnomalyDetector:
    """Shared fit/score/predict API with normalization and thresholding."""

    def __init__(
        self,
        window_size: int = 200,
        stride: int = 1,
        device: str | None = None,
        seed: int = 13,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.scaler_: StandardScaler | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.training_summary_: dict[str, Any] = {}

    def fit(self, train_values: np.ndarray) -> "BaseTimeSeriesAnomalyDetector":
        """Seed RNGs, fit the standardizer, and train on the normalized series."""
        set_random_seed(self.seed)
        train_array = ensure_2d(train_values)
        self.scaler_ = fit_standardizer_nd(train_array)
        self.mean_ = np.asarray(self.scaler_.mean_, dtype=np.float32)
        self.std_ = np.asarray(self.scaler_.scale_, dtype=np.float32)
        normalized_train = transform_standardize_nd(train_array, self.scaler_)
        self._fit_normalized(normalized_train)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        """Standardize with the fitted scaler and return per-timestep scores."""
        normalized = self._normalize(values)
        return self._score_normalized(normalized)

    def predict(self, values: np.ndarray, threshold: float) -> dict[str, np.ndarray | float]:
        """Score ``values`` and binarize with ``score >= threshold``."""
        scores = self.score(values)
        labels = (scores >= threshold).astype(np.int64)
        return {"scores": scores, "labels": labels, "threshold": float(threshold)}

    def get_params(self) -> dict[str, Any]:
        """Return a shallow copy of the detector's attributes."""
        return dict(self.__dict__)

    def set_params(self, **kwargs: Any) -> "BaseTimeSeriesAnomalyDetector":
        """Set attributes from keyword arguments and return ``self``."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Apply the fitted standardizer; raises if ``fit`` was never called."""
        if self.scaler_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        array = ensure_2d(values)
        return transform_standardize_nd(array, self.scaler_)

    def _fit_normalized(self, train_values: np.ndarray) -> None:
        """Train on the standardized series; implemented by subclasses."""
        raise NotImplementedError

    def _score_normalized(self, values: np.ndarray) -> np.ndarray:
        """Score the standardized series; implemented by subclasses."""
        raise NotImplementedError


class GenIASWindowGenerator:
    """Trains GenIAS on normal windows and uses it to generate patched anomalies."""

    def __init__(
        self,
        window_size: int,
        latent_dim: int = 50,
        learning_rate: float = 1e-4,
        batch_size: int = 100,
        max_epochs: int = 1000,
        patience: int = 12,
        tau: float = 0.2,
        anomaly_scale: float = 1.0,
        device: str = "cpu",
        seed: int = 13,
    ) -> None:
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.tau = tau
        self.anomaly_scale = anomaly_scale
        self.device = device
        self.seed = seed
        self.loss_config = GeniasLossConfig(tau=tau)
        self.model: GenIAS | None = None
        self.scaler_: StandardScaler | None = None
        self.history_: dict[str, float] = {}
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, train_windows: np.ndarray) -> None:
        """Train the GenIAS VAE on normal windows (early stopping on train loss)."""
        if train_windows.shape[-1] != 1:
            raise ValueError("GenIAS-backed variants currently support only univariate windows")
        set_random_seed(self.seed)
        flattened = train_windows.reshape(-1, train_windows.shape[-1])
        self.scaler_ = fit_standardizer_nd(flattened)
        self.mean_ = np.asarray(self.scaler_.mean_, dtype=np.float32)
        self.std_ = np.asarray(self.scaler_.scale_, dtype=np.float32)
        normalized_windows = transform_standardize_nd(train_windows, self.scaler_)
        tensor = torch.from_numpy(np.transpose(normalized_windows, (0, 2, 1)))
        loader = DataLoader(tensor, batch_size=self.batch_size, shuffle=True, drop_last=False)
        model = GenIAS(window_size=self.window_size, latent_dim=self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
        best_loss = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(self.max_epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch)
                losses = compute_losses(outputs, batch, self.loss_config)
                losses["loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(losses["loss"].detach().cpu())
                batch_count += 1
            average_loss = epoch_loss / max(batch_count, 1)
            scheduler.step(average_loss)
            log_epoch("GenIAS", epoch, self.max_epochs, average_loss)
            if average_loss < best_loss:
                best_loss = average_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        self.model.eval()
        self.history_ = {"best_loss": best_loss, "epochs_trained": epoch + 1}

    def generate_batch(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(anomalous_windows, anomaly_mask)`` generated from ``batch``."""
        if self.model is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("GenIAS generator must be fitted before use")
        input_device = batch.device
        self.model.eval()
        torch.manual_seed(self.seed)
        mean = torch.as_tensor(self.mean_, dtype=batch.dtype, device=self.device).view(1, 1, -1)
        std = torch.as_tensor(self.std_, dtype=batch.dtype, device=self.device).view(1, 1, -1)
        with torch.no_grad():
            normalized_batch = (batch.to(self.device) - mean) / std
            batch_channels_first = normalized_batch.transpose(1, 2)
            outputs = self.model(batch_channels_first)
            patched, mask = patch_anomalies(batch_channels_first, outputs["x_tilde"], tau=self.tau)
        scaled = batch.to(self.device) + (self.anomaly_scale * (((patched.transpose(1, 2) * std) + mean) - batch.to(self.device)))
        return scaled.to(input_device), mask.transpose(1, 2).to(input_device)

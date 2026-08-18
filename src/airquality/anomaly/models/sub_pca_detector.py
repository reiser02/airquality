"""TSB-AD-style Sub_PCA detector over sliding time-series subsequences.

The sliding-window selection, PCA scoring, normalization, and padding logic are
adapted from TSB-AD:
https://github.com/TheDatumOrg/TSB-AD

TSB-AD's PCA implementation acknowledges PyOD as an upstream source:
https://github.com/yzhao062/pyod

The segment-aware API and several edge-case behaviors are local modifications.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import argrelextrema
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf

from .common import (
    BaseTimeSeriesAnomalyDetector,
    ensure_2d,
    pooled_windows_nd,
    rolling_windows_nd,
)


def find_length_rank(
    values: np.ndarray,
    *,
    rank: int = 1,
    fallback_window: int = 125,
    max_period: int = 300,
) -> int:
    """Return the window at the requested strongest ACF local maximum."""

    if rank < 0:
        raise ValueError("rank must be non-negative")
    if fallback_window < 1 or max_period < 1:
        raise ValueError("fallback_window and max_period must be positive")
    if rank == 0:
        return 1

    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)][: min(20_000, data.size)]
    if data.size < 5 or np.isclose(np.var(data), 0.0):
        return int(fallback_window)

    try:
        autocorrelation = acf(
            data,
            nlags=min(400, data.size - 1),
            fft=True,
        )[3:]
        local_maxima = argrelextrema(autocorrelation, np.greater)[0]
        if local_maxima.size == 0:
            return int(fallback_window)
        order = np.argsort(autocorrelation[local_maxima])[::-1]
        selected = int(order[min(rank - 1, len(order) - 1)])
        period = int(local_maxima[selected] + 3)
        return int(fallback_window if period > max_period else period)
    except (ValueError, FloatingPointError):
        return int(fallback_window)


def _pad_window_scores(
    window_scores: np.ndarray,
    series_length: int,
    window_size: int,
) -> np.ndarray:
    """Pad scores to the original length as TSB-AD's PCA wrapper does."""

    scores = np.asarray(window_scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return np.full(series_length, np.nan, dtype=np.float32)
    if scores.size >= series_length:
        return scores[:series_length]
    left = (window_size - 1 + 1) // 2
    right = (window_size - 1) // 2
    padded = np.pad(scores, (left, right), mode="edge")
    return padded[:series_length].astype(np.float32, copy=False)


def _zscore_columns(windows: np.ndarray) -> np.ndarray:
    """Z-normalize each lag column, matching TSB-AD's univariate path."""

    mean = windows.mean(axis=0)
    std = windows.std(axis=0, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = (windows - mean) / std
    return np.nan_to_num(normalized).astype(np.float32, copy=False)


class SubPCADetector(BaseTimeSeriesAnomalyDetector):
    """Sub_PCA matching TSB-AD's PyOD-derived implementation.

    Sliding windows are normalized as a matrix, constant columns are pruned,
    PCA is fitted on the resulting subsequences, and scores are the weighted
    distances to the selected low-variance component vectors.
    """

    def __init__(
        self,
        *args: Any,
        window_size: int | None = None,
        periodicity: int = 1,
        fallback_window: int = 125,
        n_components: int | float | str | None = None,
        n_selected_components: int | None = None,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: str = "auto",
        tol: float = 0.0,
        iterated_power: int | str = "auto",
        weighted: bool = True,
        standardization: bool = True,
        zero_pruning: bool = True,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        initial_window = 1 if window_size is None else int(window_size)
        if initial_window < 1:
            raise ValueError("window_size must be positive")
        if periodicity < 0:
            raise ValueError("periodicity must be non-negative")
        if fallback_window < 1:
            raise ValueError("fallback_window must be positive")
        super().__init__(*args, window_size=initial_window, **kwargs)
        self.window_size = initial_window
        self.minimum_series_length = initial_window + 1
        self.explicit_window_size = window_size is not None
        self.periodicity = int(periodicity)
        self.fallback_window = int(fallback_window)
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.weighted = weighted
        self.standardization = standardization
        self.zero_pruning = zero_pruning
        self.normalize = normalize
        self.window_scaler_: StandardScaler | None = None
        self.model_: SklearnPCA | None = None
        self.selected_components_: np.ndarray | None = None
        self.selected_weights_: np.ndarray | None = None
        self.active_columns_: np.ndarray | None = None
        self.constant_series_ = False

    def _resolve_window_size(self, train_segments: list[np.ndarray]) -> int:
        """Resolve one station-level window without crossing segment gaps."""

        longest = max(train_segments, key=len)
        if longest.shape[1] != 1:
            raise ValueError("SubPCADetector currently supports only univariate series")
        if self.explicit_window_size:
            window = self.window_size
        else:
            window = find_length_rank(
                longest[:, 0],
                rank=self.periodicity,
                fallback_window=self.fallback_window,
            )
            # Keep short air-quality blocks usable while preserving two windows.
            window = min(window, max(1, len(longest) - 1))
        if window < 1:
            raise ValueError("Sub_PCA could not resolve a positive window")
        return int(window)

    def fit_segments(self, train_segments: list[np.ndarray]) -> "SubPCADetector":
        """Fit one pooled Sub_PCA model without creating cross-gap windows."""

        arrays = [ensure_2d(values) for values in train_segments if len(values) > 0]
        if not arrays:
            raise ValueError("At least one non-empty training segment is required")
        if any(array.shape[1] != 1 for array in arrays):
            raise ValueError("SubPCADetector currently supports only univariate series")
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("SubPCADetector requires finite training values")

        self.window_size = self._resolve_window_size(arrays)
        self.minimum_series_length = self.window_size + 1
        windows = pooled_windows_nd(arrays, self.window_size, stride=1)[:, :, 0]
        if len(windows) < 2:
            raise ValueError("Sub_PCA requires at least two training windows")
        self._fit_windows(windows)
        return self

    def _fit_windows(self, windows: np.ndarray) -> None:
        """Fit the TSB-AD PCA stage on a window matrix."""

        prepared = _zscore_columns(windows) if self.normalize else windows
        if self.standardization:
            self.window_scaler_ = StandardScaler().fit(prepared)
            prepared = self.window_scaler_.transform(prepared)
        else:
            self.window_scaler_ = None

        if self.zero_pruning:
            active_columns = np.any(prepared != 0.0, axis=0)
        else:
            active_columns = np.ones(prepared.shape[1], dtype=bool)
        self.active_columns_ = active_columns
        if not active_columns.any():
            self.constant_series_ = True
            self.training_summary_ = {
                "window_size": self.window_size,
                "periodicity": self.periodicity,
                "constant_series": True,
                "n_training_windows": int(len(windows)),
                "n_active_columns": 0,
                "weighted": bool(self.weighted),
            }
            return

        self.constant_series_ = False
        self.model_ = SklearnPCA(
            n_components=self.n_components,
            copy=self.copy,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            random_state=self.seed,
        )
        self.model_.fit(prepared[:, active_columns])

        n_components = self.model_.n_components_
        if self.n_selected_components is None:
            n_selected = n_components
        else:
            n_selected = int(self.n_selected_components)
            if not 1 <= n_selected <= n_components:
                raise ValueError(
                    "n_selected_components must be between 1 and the fitted "
                    "number of PCA components"
                )
        weights = (
            self.model_.explained_variance_ratio_
            if self.weighted
            else np.ones(n_components, dtype=np.float64)
        )
        self.selected_components_ = self.model_.components_[-n_selected:, :]
        self.selected_weights_ = np.maximum(
            weights[-n_selected:], np.finfo(np.float64).eps
        )
        self.training_summary_ = {
            "window_size": self.window_size,
            "periodicity": self.periodicity,
            "n_components": int(n_components),
            "n_selected_components": int(n_selected),
            "n_training_windows": int(len(windows)),
            "n_active_columns": int(active_columns.sum()),
            "constant_series": False,
            "weighted": bool(self.weighted),
        }

    def score_segments(
        self, segments: list[np.ndarray]
    ) -> list[np.ndarray | None]:
        """Score each segment independently, abstaining below fit support."""

        scores: list[np.ndarray | None] = []
        for segment in segments:
            if len(segment) < self.minimum_series_length:
                scores.append(None)
                continue
            try:
                scores.append(self.score(segment))
            except Exception:
                scores.append(None)
        return scores

    def score(self, values: np.ndarray) -> np.ndarray:
        """Score a series using TSB-AD's window normalization and padding."""

        array = ensure_2d(values)
        if array.shape[1] != 1:
            raise ValueError("SubPCADetector currently supports only univariate series")
        if not np.isfinite(array).all():
            raise ValueError("SubPCADetector requires finite scoring values")
        if array.shape[0] < self.minimum_series_length:
            raise ValueError(
                f"Series length {array.shape[0]} is shorter than the fitted "
                f"window support {self.minimum_series_length}"
            )
        if self.active_columns_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        if self.constant_series_:
            return np.zeros(array.shape[0], dtype=np.float32)
        if self.model_ is None or self.selected_components_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        windows = rolling_windows_nd(array, self.window_size, stride=1)[:, :, 0]
        prepared = _zscore_columns(windows) if self.normalize else windows
        if self.window_scaler_ is not None:
            prepared = self.window_scaler_.transform(prepared)
        window_scores = np.sum(
            cdist(prepared[:, self.active_columns_], self.selected_components_)
            / self.selected_weights_,
            axis=1,
        )
        return _pad_window_scores(
            window_scores.astype(np.float32), array.shape[0], self.window_size
        )

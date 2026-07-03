"""Classic statistical baseline detectors (z-score, IQR, forests, LOF, Hampel).

These detectors score each timestep directly from global or rolling statistics
of the series — no windows, no training loops. They share the
``fit``/``score``/``predict`` contract of :class:`BaseGlobalAnomalyDetector`
so the benchmark can treat them like the deep detectors.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .common import ensure_2d, set_random_seed


class BaseGlobalAnomalyDetector:
    """Global detectors that score each timestep directly."""

    def __init__(
        self,
        window_size: int = 1,
        device: str | None = None,
        seed: int = 13,
    ) -> None:
        self.window_size = window_size
        self.device = device
        self.seed = seed
        self.training_summary_: dict[str, Any] = {}

    def fit(self, train_values: np.ndarray) -> "BaseGlobalAnomalyDetector":
        """Seed RNGs, coerce input to 2D, and fit the concrete detector."""
        set_random_seed(self.seed)
        train_array = ensure_2d(train_values)
        self._fit_array(train_array)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        """Return one anomaly score per timestep (higher = more anomalous)."""
        return self._score_array(ensure_2d(values))

    def predict(self, values: np.ndarray, threshold: float) -> dict[str, np.ndarray | float]:
        """Score ``values`` and binarize with ``score >= threshold``."""
        scores = self.score(values)
        labels = (scores >= threshold).astype(np.int64)
        return {"scores": scores, "labels": labels, "threshold": float(threshold)}

    def get_params(self) -> dict[str, Any]:
        """Return a shallow copy of the detector's attributes."""
        return dict(self.__dict__)

    def set_params(self, **kwargs: Any) -> "BaseGlobalAnomalyDetector":
        """Set attributes from keyword arguments and return ``self``."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def _fit_array(self, train_values: np.ndarray) -> None:
        """Fit on a 2D ``(timesteps, features)`` array; implemented by subclasses."""
        raise NotImplementedError

    def _score_array(self, values: np.ndarray) -> np.ndarray:
        """Score a 2D ``(timesteps, features)`` array; implemented by subclasses."""
        raise NotImplementedError


class ModifiedZScoreDetector(BaseGlobalAnomalyDetector):
    """Modified z-score (0.6745 · |x − median| / MAD) fitted globally per feature."""

    def __init__(self, *args: Any, mad_epsilon: float = 1e-6, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mad_epsilon = mad_epsilon
        self.median_: np.ndarray | None = None
        self.mad_: np.ndarray | None = None

    def _fit_array(self, train_values: np.ndarray) -> None:
        """Store the per-feature median and (epsilon-clipped) MAD."""
        self.median_ = np.median(train_values, axis=0).astype(np.float32)
        mad = np.median(np.abs(train_values - self.median_), axis=0)
        self.mad_ = np.maximum(mad, self.mad_epsilon).astype(np.float32)
        self.training_summary_ = {"mad_epsilon": self.mad_epsilon}

    def _score_array(self, values: np.ndarray) -> np.ndarray:
        """Return the max modified z-score across features per timestep."""
        if self.median_ is None or self.mad_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        feature_scores = 0.6745 * np.abs(values - self.median_[None, :]) / self.mad_[None, :]
        return feature_scores.max(axis=1).astype(np.float32)


class IQRDetector(BaseGlobalAnomalyDetector):
    """Tukey-fence detector: scores how far a point falls outside Q1/Q3 ± k·IQR."""

    def __init__(self, *args: Any, fence_scale: float = 1.5, iqr_epsilon: float = 1e-6, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fence_scale = fence_scale
        self.iqr_epsilon = iqr_epsilon
        self.q1_: np.ndarray | None = None
        self.q3_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None

    def _fit_array(self, train_values: np.ndarray) -> None:
        """Store per-feature Q1/Q3 and the (epsilon-clipped) IQR."""
        self.q1_ = np.percentile(train_values, 25.0, axis=0).astype(np.float32)
        self.q3_ = np.percentile(train_values, 75.0, axis=0).astype(np.float32)
        self.iqr_ = np.maximum(self.q3_ - self.q1_, self.iqr_epsilon).astype(np.float32)
        self.training_summary_ = {"fence_scale": self.fence_scale}

    def _score_array(self, values: np.ndarray) -> np.ndarray:
        """Return the max IQR-scaled excess over the Tukey fences per timestep."""
        if self.q1_ is None or self.q3_ is None or self.iqr_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        lower = self.q1_[None, :] - (self.fence_scale * self.iqr_[None, :])
        upper = self.q3_[None, :] + (self.fence_scale * self.iqr_[None, :])
        lower_excess = np.maximum(lower - values, 0.0) / self.iqr_[None, :]
        upper_excess = np.maximum(values - upper, 0.0) / self.iqr_[None, :]
        return np.maximum(lower_excess, upper_excess).max(axis=1).astype(np.float32)


class IsolationForestDetector(BaseGlobalAnomalyDetector):
    """scikit-learn ``IsolationForest`` wrapper; score = negated ``score_samples``."""

    def __init__(
        self,
        *args: Any,
        n_estimators: int = 100,
        n_jobs: int = -1,
        contamination: str | float = "auto",
        max_samples: str | int | float = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.contamination = contamination
        self.max_samples = max_samples
        self.model_: IsolationForest | None = None

    def _fit_array(self, train_values: np.ndarray) -> None:
        """Fit an ``IsolationForest`` seeded with the detector's ``seed``."""
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.seed,
        )
        self.model_.fit(train_values)
        self.training_summary_ = {"n_estimators": self.n_estimators}

    def _score_array(self, values: np.ndarray) -> np.ndarray:
        """Return the negated isolation score (higher = more anomalous)."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        return (-self.model_.score_samples(values)).astype(np.float32)


class LOFDetector(BaseGlobalAnomalyDetector):
    """Local Outlier Factor in non-novelty mode: can only score the fitted series."""

    def __init__(
        self,
        *args: Any,
        n_neighbors: int = 50,
        n_jobs: int = -1,
        contamination: str | float = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.contamination = contamination
        self.model_: LocalOutlierFactor | None = None
        self.fit_values_: np.ndarray | None = None
        self.fit_scores_: np.ndarray | None = None

    def _fit_array(self, train_values: np.ndarray) -> None:
        """Fit LOF and cache the training array plus its outlier factors."""
        neighbor_count = max(1, min(self.n_neighbors, len(train_values) - 1))
        self.model_ = LocalOutlierFactor(
            n_neighbors=neighbor_count,
            n_jobs=self.n_jobs,
            contamination=self.contamination,
            novelty=False,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Duplicate values are leading to incorrect results.*",
                category=UserWarning,
                module="sklearn.neighbors._lof",
            )
            self.model_.fit(train_values)
        self.fit_values_ = train_values.copy()
        self.fit_scores_ = (-self.model_.negative_outlier_factor_).astype(np.float32)
        self.training_summary_ = {"n_neighbors": neighbor_count}

    def _score_array(self, values: np.ndarray) -> np.ndarray:
        """Return the cached fit scores; LOF (non-novelty) cannot score new data."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        if self.fit_values_ is not None and self.fit_scores_ is not None and np.array_equal(values, self.fit_values_):
            return self.fit_scores_
        raise RuntimeError("LOF can only score the fitted series")


class HampelDetector(BaseGlobalAnomalyDetector):
    """Rolling-window Hampel filter: scores each point by its deviation from a
    centered rolling median, in units of the rolling MAD-derived robust scale.

    Unlike the other baselines here, the rolling statistics are recomputed directly
    on whatever series is passed to `score`, so `fit` only records bookkeeping.
    """

    def __init__(self, *args: Any, window_size: int = 24, mad_epsilon: float = 1e-6, **kwargs: Any) -> None:
        super().__init__(*args, window_size=window_size, **kwargs)
        self.mad_epsilon = mad_epsilon

    def _fit_array(self, train_values: np.ndarray) -> None:
        """No-op: Hampel statistics are recomputed on the scored series."""
        self.training_summary_ = {"window_size": self.window_size}

    def _score_array(self, values: np.ndarray) -> np.ndarray:
        """Return the max rolling Hampel score across features per timestep."""
        feature_scores = [self._rolling_score(values[:, column]) for column in range(values.shape[1])]
        return np.max(np.stack(feature_scores, axis=1), axis=1).astype(np.float32)

    def _rolling_score(self, series: np.ndarray) -> np.ndarray:
        """Score one feature: |x − rolling median| / (1.4826 · rolling MAD)."""
        rolling = pd.Series(series).rolling(window=self.window_size, center=True, min_periods=1)
        median = rolling.median().to_numpy()
        mad = rolling.apply(lambda window: np.median(np.abs(window - np.median(window))), raw=True).to_numpy()
        scale = np.maximum(1.4826 * mad, self.mad_epsilon)
        return (np.abs(series - median) / scale).astype(np.float32)


class Hampel6Detector(HampelDetector):
    """Hampel variant with a short 6-hour window (vs. the 24-hour default)."""

    def __init__(self, *args: Any, window_size: int = 6, **kwargs: Any) -> None:
        super().__init__(*args, window_size=window_size, **kwargs)

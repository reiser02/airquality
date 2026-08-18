"""Prophet-based anomaly detector (forecast confidence-band excess)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

from .baselines import BaseGlobalAnomalyDetector
from .common import ensure_2d, set_random_seed

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def _build_frame(
    series: np.ndarray,
    freq: str,
    index: pd.DatetimeIndex | Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Wrap values and timestamps in the ``ds``/``y`` Prophet format."""

    if index is None:
        dates = pd.date_range("2000-01-01", periods=len(series), freq=freq)
    else:
        dates = pd.DatetimeIndex(index)
        if len(dates) != len(series):
            raise ValueError(
                f"Prophet timestamps ({len(dates)}) must match values ({len(series)})"
            )
        if not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError("Prophet timestamps must be sorted and unique")
    return pd.DataFrame(
        {
            "ds": dates,
            "y": series,
        }
    )


class ProphetDetector(BaseGlobalAnomalyDetector):
    """Anomaly detector based on Facebook Prophet's forecast confidence interval.

    Scores each point by how far it falls outside Prophet's `interval_width`
    prediction band, normalized by the band's own half-width (0 when inside).
    """

    def __init__(
        self,
        *args: Any,
        interval_width: float = 0.95,
        freq: str = "h",
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.minimum_series_length = 2
        self.interval_width = interval_width
        self.freq = freq
        self.epsilon = epsilon
        self.model_: Prophet | None = None
        self.models_: list[Prophet] = []
        self.segment_indices_: list[pd.DatetimeIndex | None] = []

    @staticmethod
    def _normalize_indices(
        segment_indices: Sequence[object] | None,
        lengths: Sequence[int],
    ) -> list[pd.DatetimeIndex | None]:
        """Validate optional real timestamps without changing other detectors."""

        if segment_indices is None:
            return [None] * len(lengths)
        if len(segment_indices) != len(lengths):
            raise ValueError("Prophet timestamps must match the fitted segment layout")
        normalized: list[pd.DatetimeIndex | None] = []
        for index, length in zip(segment_indices, lengths, strict=True):
            dates = pd.DatetimeIndex(index)
            if len(dates) != length:
                raise ValueError(
                    f"Prophet timestamps ({len(dates)}) must match values ({length})"
                )
            if not dates.is_monotonic_increasing or dates.has_duplicates:
                raise ValueError("Prophet timestamps must be sorted and unique")
            normalized.append(dates)
        return normalized

    def fit_segments(
        self,
        train_segments: list[np.ndarray],
        *,
        segment_indices: Sequence[object] | None = None,
    ) -> "ProphetDetector":
        """Fit one Prophet timeline per segment without crossing gaps."""
        set_random_seed(self.seed)
        arrays = [ensure_2d(values) for values in train_segments if len(values) > 0]
        if not arrays:
            raise ValueError("At least one non-empty training segment is required")
        if any(array.shape[1] != 1 for array in arrays):
            raise ValueError("ProphetDetector currently supports only univariate series")
        if any(np.isfinite(array[:, 0]).sum() < self.minimum_series_length for array in arrays):
            raise ValueError("Prophet requires at least two finite observations per segment")
        indices = self._normalize_indices(segment_indices, [len(array) for array in arrays])
        self.models_ = [
            self._fit_model(array, index)
            for array, index in zip(arrays, indices, strict=True)
        ]
        self.model_ = self.models_[0]
        self.segment_indices_ = indices
        self.training_summary_ = {
            "interval_width": self.interval_width,
            "freq": self.freq,
            "segment_lengths": [len(array) for array in arrays],
            "real_timestamps": any(index is not None for index in indices),
        }
        return self

    def score_segments(
        self,
        segments: list[np.ndarray],
        *,
        segment_indices: Sequence[object] | None = None,
    ) -> list[np.ndarray | None]:
        """Score each segment against the Prophet model fitted to that timeline."""
        if len(segments) != len(self.models_):
            raise ValueError("Prophet segments must match the fitted segment layout")
        if segment_indices is None:
            indices = self.segment_indices_
        else:
            indices = self._normalize_indices(segment_indices, [len(segment) for segment in segments])
        if len(indices) != len(segments):
            raise ValueError("Prophet timestamps must match the fitted segment layout")
        scores: list[np.ndarray | None] = []
        for model, values, index in zip(self.models_, segments, indices, strict=True):
            try:
                scores.append(self._score_with_model(model, ensure_2d(values), index))
            except Exception:
                scores.append(None)
        return scores

    def _fit_model(
        self,
        train_values: np.ndarray,
        index: pd.DatetimeIndex | None = None,
    ) -> Prophet:
        if train_values.shape[1] != 1:
            raise ValueError("ProphetDetector currently supports only univariate series")
        frame = _build_frame(
            train_values[:, 0].astype(np.float64), self.freq, index=index
        )
        model = Prophet(interval_width=self.interval_width)
        model.fit(frame, seed=self.seed)
        return model

    def _fit_array(self, train_values: np.ndarray) -> None:
        """Fit one Prophet model for the legacy array-only API."""
        self.model_ = self._fit_model(train_values)
        self.models_ = [self.model_]
        self.segment_indices_ = [None]
        self.training_summary_ = {
            "interval_width": self.interval_width,
            "freq": self.freq,
            "train_length": int(train_values.shape[0]),
            "real_timestamps": False,
        }

    def _score_array(self, values: np.ndarray) -> np.ndarray:
        """Score by band excess: distance outside [yhat_lower, yhat_upper] / half-width."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        index = self.segment_indices_[0] if self.segment_indices_ else None
        return self._score_with_model(self.model_, values, index=index)

    def _score_with_model(
        self,
        model: Prophet,
        values: np.ndarray,
        index: pd.DatetimeIndex | None = None,
    ) -> np.ndarray:
        """Score one segment against its fitted interval timeline."""
        series = values[:, 0].astype(np.float64)
        # Prophet's interval bounds come from Monte Carlo trend sampling in predict()
        # that draws from the global numpy RNG, independent of fit()'s `seed` kwarg.
        np.random.seed(self.seed)
        if index is None:
            history_dates = getattr(model, "history_dates", None)
            if history_dates is not None and len(history_dates) == len(series):
                index = pd.DatetimeIndex(history_dates)
        forecast = model.predict(_build_frame(series, self.freq, index=index)[["ds"]])
        yhat_lower = forecast["yhat_lower"].to_numpy()
        yhat_upper = forecast["yhat_upper"].to_numpy()
        half_width = np.maximum((yhat_upper - yhat_lower) / 2.0, self.epsilon)
        excess = np.maximum(np.maximum(series - yhat_upper, yhat_lower - series), 0.0)
        return (excess / half_width).astype(np.float32)

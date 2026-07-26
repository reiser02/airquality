"""Prophet-based anomaly detector (forecast confidence-band excess)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

from .baselines import BaseGlobalAnomalyDetector
from .common import ensure_2d, set_random_seed

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def _build_frame(series: np.ndarray, freq: str) -> pd.DataFrame:
    """Wrap a value array in the ``ds``/``y`` dataframe format Prophet expects."""
    return pd.DataFrame(
        {
            "ds": pd.date_range("2000-01-01", periods=len(series), freq=freq),
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
        freq: str = "s",
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.interval_width = interval_width
        self.freq = freq
        self.epsilon = epsilon
        self.model_: Prophet | None = None
        self.models_: list[Prophet] = []

    def fit_segments(
        self, train_segments: list[np.ndarray]
    ) -> "ProphetDetector":
        """Fit one timeline per segment so synthetic timestamps never cross gaps."""
        set_random_seed(self.seed)
        arrays = [ensure_2d(values) for values in train_segments if len(values) > 0]
        if not arrays:
            raise ValueError("At least one non-empty training segment is required")
        self.models_ = [self._fit_model(array) for array in arrays]
        self.model_ = self.models_[0]
        self.training_summary_ = {
            "interval_width": self.interval_width,
            "segment_lengths": [len(array) for array in arrays],
        }
        return self

    def score_segments(
        self, segments: list[np.ndarray]
    ) -> list[np.ndarray | None]:
        """Score each segment against the Prophet model fitted to that timeline."""
        if len(segments) != len(self.models_):
            raise ValueError("Prophet segments must match the fitted segment layout")
        scores: list[np.ndarray | None] = []
        for model, values in zip(self.models_, segments, strict=True):
            try:
                scores.append(self._score_with_model(model, ensure_2d(values)))
            except Exception:
                scores.append(None)
        return scores

    def _fit_model(self, train_values: np.ndarray) -> Prophet:
        if train_values.shape[1] != 1:
            raise ValueError("ProphetDetector currently supports only univariate series")
        frame = _build_frame(train_values[:, 0].astype(np.float64), self.freq)
        model = Prophet(interval_width=self.interval_width)
        model.fit(frame, seed=self.seed)
        return model

    def _fit_array(self, train_values: np.ndarray) -> None:
        """Fit one Prophet model on the (univariate) series with a synthetic index."""
        self.model_ = self._fit_model(train_values)
        self.models_ = [self.model_]
        self.training_summary_ = {"interval_width": self.interval_width, "train_length": int(train_values.shape[0])}

    def _score_array(self, values: np.ndarray) -> np.ndarray:
        """Score by band excess: distance outside [yhat_lower, yhat_upper] / half-width."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        return self._score_with_model(self.model_, values)

    def _score_with_model(self, model: Prophet, values: np.ndarray) -> np.ndarray:
        """Score one segment with its own synthetic Prophet timeline."""
        series = values[:, 0].astype(np.float64)
        # Prophet's interval bounds come from Monte Carlo trend sampling in predict()
        # that draws from the global numpy RNG, independent of fit()'s `seed` kwarg.
        np.random.seed(self.seed)
        forecast = model.predict(_build_frame(series, self.freq)[["ds"]])
        yhat_lower = forecast["yhat_lower"].to_numpy()
        yhat_upper = forecast["yhat_upper"].to_numpy()
        half_width = np.maximum((yhat_upper - yhat_lower) / 2.0, self.epsilon)
        excess = np.maximum(np.maximum(series - yhat_upper, yhat_lower - series), 0.0)
        return (excess / half_width).astype(np.float32)

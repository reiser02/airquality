"""Prophet-based anomaly detector (forecast confidence-band excess)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

from .baselines import BaseGlobalAnomalyDetector

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

    def _fit_array(self, train_values: np.ndarray) -> None:
        """Fit one Prophet model on the (univariate) series with a synthetic index."""
        if train_values.shape[1] != 1:
            raise ValueError("ProphetDetector currently supports only univariate series")
        frame = _build_frame(train_values[:, 0].astype(np.float64), self.freq)
        model = Prophet(interval_width=self.interval_width)
        model.fit(frame, seed=self.seed)
        self.model_ = model
        self.training_summary_ = {"interval_width": self.interval_width, "train_length": int(train_values.shape[0])}

    def _score_array(self, values: np.ndarray) -> np.ndarray:
        """Score by band excess: distance outside [yhat_lower, yhat_upper] / half-width."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        series = values[:, 0].astype(np.float64)
        # Prophet's interval bounds come from Monte Carlo trend sampling in predict()
        # that draws from the global numpy RNG, independent of fit()'s `seed` kwarg.
        np.random.seed(self.seed)
        forecast = self.model_.predict(_build_frame(series, self.freq)[["ds"]])
        yhat_lower = forecast["yhat_lower"].to_numpy()
        yhat_upper = forecast["yhat_upper"].to_numpy()
        half_width = np.maximum((yhat_upper - yhat_lower) / 2.0, self.epsilon)
        excess = np.maximum(np.maximum(series - yhat_upper, yhat_lower - series), 0.0)
        return (excess / half_width).astype(np.float32)

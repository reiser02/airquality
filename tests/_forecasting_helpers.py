"""Shared helpers for forecasting tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


BASELINE_DETECTORS = ["ModifiedZScore", "IQR", "Hampel_w24"]


def _seasonal_series(n: int = 900, name: str = "ST", seed: int = 0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    vals = (
        30.0
        + 8.0 * np.sin(np.arange(n) * 2 * np.pi / 24)
        + 4.0 * np.sin(np.arange(n) * 2 * np.pi / 168)
        + rng.normal(0, 1, n)
    )
    return pd.Series(vals, index=idx, name=name)

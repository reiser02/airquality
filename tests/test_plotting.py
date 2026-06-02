from __future__ import annotations

import pandas as pd
import pytest

from airquality.visualization.plotting import (
    get_prediction_time_window,
    plot_predictions_by_gap,
)


def test_get_prediction_time_window_returns_global_bounds() -> None:
    idx1 = pd.date_range("2024-01-01", periods=3, freq="h")
    idx2 = pd.date_range("2024-01-02", periods=2, freq="h")
    preds = {
        "m1": pd.Series([1.0, None, 2.0], index=idx1),
        "m2": pd.Series([3.0, 4.0], index=idx2),
    }
    window = get_prediction_time_window(preds)
    assert window is not None
    assert window[0] == idx1[0]
    assert window[1] == idx2[-1]


def test_get_prediction_time_window_none_when_all_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="h")
    assert get_prediction_time_window({"m": pd.Series([None, None], index=idx)}) is None


def test_plot_predictions_by_gap_validates_inputs() -> None:
    with pytest.raises(ValueError):
        plot_predictions_by_gap({}, grid_n_cols=0)
    with pytest.raises(ValueError):
        plot_predictions_by_gap({}, render_style="bad")

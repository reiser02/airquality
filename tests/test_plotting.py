"""Tests plotting helpers for benchmark windows, legends, and grid rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from airquality.visualization import plotting
from airquality.visualization.plotting import (
    get_prediction_time_window,
    plot_predictions_by_gap,
    plot_predictions_by_method_grid,
)


def _series(values: list[float | None], start: str = "2024-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="h")
    return pd.Series(values, index=idx, dtype=float)


def _gap_store() -> dict[int, dict[str, object]]:
    actual = _series([10.0, 11.0, 12.0])
    pred_a = pd.Series([10.2, 11.2], index=actual.index[:2], dtype=float)
    pred_b = pd.Series([9.8, 10.8], index=actual.index[:2], dtype=float)
    naive = pd.Series([1.0, 1.1], index=actual.index[:2], dtype=float)
    return {
        2: {
            "series": {
                "station": {
                    "actual": actual,
                    "preds": {"Worse": pred_a, "Better": pred_b},
                    "naive_mase": naive,
                }
            }
        }
    }


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


def test_plot_predictions_by_gap_sorts_legend_and_adds_textbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")

    results_df = pd.DataFrame(
        {
            "Serie": ["station", "station"],
            "Gap_Size": [2, 2],
            "Modelo": ["Worse", "Better"],
            "MASE": [2.0, 1.0],
        }
    )

    plot_predictions_by_gap(
        _gap_store(),
        results_df=results_df,
        error_display="both",
        legend_sort="mase_asc",
        grid_by_gap=True,
    )

    fig = plt.gcf()
    ax = fig.axes[0]
    labels = ax.get_legend_handles_labels()[1]
    assert labels[0] == "Real"
    assert labels[1].startswith("Better (MASE=1.000)")
    assert labels[2].startswith("Worse (MASE=2.000)")
    assert any("MASE por modelo" in text.get_text() for text in ax.texts)


def test_plot_predictions_by_gap_mixed_style_uses_connectors_and_naive_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")
    calls: list[tuple[pd.Series, pd.Series]] = []

    def _capture(ax, *, gap_real, pred, color):
        del ax, color
        calls.append((gap_real.copy(), pred.copy()))

    monkeypatch.setattr(plotting, "_plot_vertical_error_connectors", _capture)

    plot_predictions_by_gap(
        _gap_store(),
        results_df=pd.DataFrame(
            {
                "Serie": ["station", "station"],
                "Gap_Size": [2, 2],
                "Modelo": ["Worse", "Better"],
                "MASE": [2.0, 1.0],
            }
        ),
        error_display="none",
        render_style="mixed",
        show_naive_mase_plot=True,
        grid_by_gap=True,
    )

    ax = plt.gcf().axes[0]
    labels = ax.get_legend_handles_labels()[1]
    assert "Hueco real" in labels
    assert "Naive MASE y(t-m)" in labels
    assert len(calls) == 2


def test_plot_predictions_by_gap_grid_turns_off_axis_for_missing_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")
    store = {
        1: {"series": {"station": {"actual": _series([1.0, 2.0]), "preds": {"M": _series([1.0, 2.0])}}}},
        2: {"series": {"other": {"actual": _series([3.0, 4.0]), "preds": {"M": _series([3.0, 4.0])}}}},
    }

    plot_predictions_by_gap(store, series_name="station", grid_by_gap=True, grid_n_cols=2)

    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert fig.axes[1].axison is False


def test_plot_predictions_by_method_grid_supports_gap_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")

    plot_predictions_by_method_grid(_gap_store(), methods=["Better"], gaps=[2], render_style="mixed")

    fig = plt.gcf()
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Metodo: Better | Gap=2"
    assert fig.axes[0].get_title().startswith("Serie: station | Gap=2 | MAE=")


def test_plot_predictions_by_method_grid_supports_method_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")
    store = {
        "M1": {
            "station": {
                "actual": _series([5.0, 6.0, 7.0]),
                "prediction": _series([5.5, 6.5], start="2024-01-01"),
            }
        }
    }

    plot_predictions_by_method_grid(store, methods=["M1"], series_name="station")

    fig = plt.gcf()
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Metodo: M1"
    assert fig.axes[0].get_title().startswith("Serie: station | MAE=")

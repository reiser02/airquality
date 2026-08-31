"""Tests for the imputation-benchmark timing figure builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from airquality.visualizations.imputation import (
    load_darts_train_seconds,
    render_timing_figures,
    save_time_by_model_plot,
    save_time_vs_error_plot,
    summarize_timing,
)

MODELS = ("interp", "TiDE", "Prophet", "TSPulse")


def _results_df(n_series: int = 3, gap_sizes: tuple[int, ...] = (1, 5)) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    # Per-row (per-series) mean train time: 0 for interpolation, NaN for the
    # pretrained Darts model (its train time lives in the training CSV), a
    # per-hole fit mean for Prophet, and the one-time load for TSPulse.
    train_by_model = {"interp": 0.0, "TiDE": np.nan, "Prophet": 1.8, "TSPulse": 12.0}
    rows = []
    for s in range(n_series):
        for gap in gap_sizes:
            for model in MODELS:
                rows.append(
                    {
                        "Modelo": model,
                        "Serie": f"ST{s}",
                        "Gap_Size": gap,
                        "Train_Seconds": train_by_model[model],
                        "Impute_Seconds": 0.05 + rng.uniform(0, 0.4),
                        "MAE": 1.0 + rng.uniform(0, 0.5),
                        "RMSE": 1.2 + rng.uniform(0, 0.6),
                        "MASE": 0.8 + rng.uniform(0, 0.6),
                        "RMSSE": 0.9 + rng.uniform(0, 0.7),
                        "R2": 0.2 + rng.uniform(0, 0.7),
                    }
                )
    return pd.DataFrame(rows)


def test_summarize_timing_one_row_per_model_in_order():
    summary = summarize_timing(_results_df())
    assert list(summary["Modelo"]) == list(MODELS)
    # Constant per-model values survive the mean; Darts stays NaN without the CSV.
    by_model = summary.set_index("Modelo")
    assert by_model.loc["TSPulse", "Train_Seconds"] == 12.0
    assert by_model.loc["interp", "Train_Seconds"] == 0.0
    assert np.isnan(by_model.loc["TiDE", "Train_Seconds"])


def test_summarize_timing_merges_darts_train_seconds():
    summary = summarize_timing(_results_df(), darts_train_seconds={"TiDE": 42.0})
    by_model = summary.set_index("Modelo")
    # Darts train comes from the external CSV; the others keep their own values.
    assert by_model.loc["TiDE", "Train_Seconds"] == 42.0
    assert by_model.loc["Prophet", "Train_Seconds"] == 1.8


def test_summarize_timing_empty_or_no_columns():
    assert summarize_timing(pd.DataFrame()).empty
    assert summarize_timing(pd.DataFrame({"Modelo": ["a"], "Serie": ["x"]})).empty


def test_load_darts_train_seconds_reads_curves_csv(tmp_path):
    # Mimic train_global_methods' CSV: training_time_seconds repeated per epoch.
    path = tmp_path / "training_curves_and_times.csv"
    pd.DataFrame(
        {
            "model_name": ["TiDE", "TiDE", "NLinear"],
            "epoch": [0, 1, 0],
            "training_time_seconds": [30.0, 30.0, 12.5],
        }
    ).to_csv(path, index=False)

    mapping = load_darts_train_seconds(path)

    assert mapping == {"TiDE": 30.0, "NLinear": 12.5}
    assert load_darts_train_seconds(tmp_path / "missing.csv") == {}


def test_save_timing_figures_smoke(tmp_path):
    results_df = _results_df()
    by_model = tmp_path / "by_model.png"
    scatter = tmp_path / "scatter.png"
    assert save_time_by_model_plot(by_model, results_df, {"TiDE": 42.0})
    assert save_time_vs_error_plot(scatter, results_df, "MASE")
    for path in (by_model, scatter):
        assert path.exists() and path.stat().st_size > 5_000


def test_save_timing_figures_empty_inputs(tmp_path):
    empty = pd.DataFrame(columns=["Modelo", "Serie", "Gap_Size"])
    assert not save_time_by_model_plot(tmp_path / "a.png", empty)
    assert not save_time_vs_error_plot(tmp_path / "b.png", empty, "MASE")
    assert not list(tmp_path.iterdir())


def test_render_timing_figures_names(tmp_path):
    saved = render_timing_figures(_results_df(), tmp_path, {"TiDE": 42.0})
    names = {path.name for path in saved}
    assert names == {
        "imputation_time_by_model.png",
        "imputation_time_vs_mase.png",
        "imputation_time_vs_rmsse.png",
        "imputation_time_vs_rmse.png",
        "imputation_time_vs_mae.png",
        "imputation_time_vs_r2.png",
    }

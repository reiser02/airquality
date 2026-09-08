"""Focused checks for the data-block pre-study."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

import airquality.data.block_analysis as block_analysis
from airquality.data.block_analysis import (
    _worst_case_requirements,
    _pollutant_run_label,
    analyze_raw_blocks,
    classify_blocks,
    observed_blocks,
    run_analysis,
    summarize_blocks,
)
from airquality.data.segments import contiguous_observed_segments
from airquality.forecasting.backtest import get_strict_forecast_requirements
from airquality.forecasting.registry import resolve_forecasting_model_configs
from airquality.visualizations.block_analysis import (
    _pollutant_color,
    _summary_rows_for_plot,
)


def test_pollutant_run_label_is_explicit_and_ordered() -> None:
    assert _pollutant_run_label(("o3",)) == "O3"
    assert _pollutant_run_label(("no2", "O3")) == "NO2_O3"
    assert _pollutant_run_label(()) == "UNKNOWN"


def test_main_includes_pollutants_in_default_output_dir(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    def fake_create_run_dir(_base_dir, name):
        calls["name"] = name
        return tmp_path

    def fake_run_analysis(**kwargs):
        calls["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(block_analysis, "create_run_dir", fake_create_run_dir)
    monkeypatch.setattr(block_analysis, "run_analysis", fake_run_analysis)
    monkeypatch.setattr(sys, "argv", ["block_analysis", "--pollutants", "O3"])

    block_analysis.main()

    assert str(calls["name"]).startswith("O3_")
    assert calls["kwargs"]["pollutants"] == ("O3",)


def test_summary_plot_rows_keep_pollutants_and_optional_total() -> None:
    summary = pd.DataFrame(
        [
            {"pollutant": "NO2", "used_blocks": 1014},
            {"pollutant": "O3", "used_blocks": 1041},
            {"pollutant": "TOTAL", "used_blocks": 2055},
        ]
    )

    rows = _summary_rows_for_plot(summary)

    assert rows["pollutant"].tolist() == ["NO2", "O3", "TOTAL"]
    assert rows["used_blocks"].tolist() == [1014, 1041, 2055]

    individual = _summary_rows_for_plot(
        pd.DataFrame(
            [
                {"pollutant": "O3", "used_blocks": 1041},
                {"pollutant": "TOTAL", "used_blocks": 1041},
            ]
        )
    )
    assert individual["pollutant"].tolist() == ["O3"]


def test_block_length_plot_uses_distinct_stable_pollutant_colors() -> None:
    assert _pollutant_color("NO2", 0) != _pollutant_color("O3", 1)
    assert _pollutant_color("NO2", 0) == _pollutant_color("NO2", 1)


def test_observed_blocks_and_validation_chronology() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="h")
    series = pd.Series([1.0, 2.0, None, 3.0, 4.0], index=index)
    assert observed_blocks(series)["hours"].tolist() == [2, 2]

    blocks = pd.DataFrame(
        {
            "start": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "end": pd.to_datetime(["2024-01-10 03:00", "2024-02-06 09:00"]),
            "hours": [220, 130],
        }
    )
    classified = classify_blocks(
        blocks,
        minimum_hours=80,
        validation_hours=48,
        host_minimum_hours=128,
    )

    assert classified["used"].tolist() == [True, True]
    assert classified["training_hours"].tolist() == [220, 82]

    classified.insert(0, "station", "S1")
    classified.insert(0, "pollutant", "NO2")
    series_summary = pd.DataFrame(
        {
            "pollutant": ["NO2"],
            "forecast_models": ["NLinear, TiDE"],
            "minimum_hours": [80],
            "horizon_hours": [12],
            "stride_hours": [6],
            "limiting_models": ["NLinear/TiDE"],
            "requested_validation_hours": [48],
            "validation_reserve_hours": [48],
            "validation_forecasts": [7],
            "host_minimum_hours": [128],
            "validation_hours": [48],
        }
    )
    total = summarize_blocks(classified, series_summary).iloc[-1]
    assert total["used_blocks"] == 2
    assert total["validation_forecasts"] == 7
    assert total["unused_eligible_blocks"] == 0
    note = block_analysis._requirement_note(series_summary)
    assert note == (
        "Peor caso entre 2 modelos configurados: 128 h (NLinear/TiDE)."
    )


def test_contiguous_observed_segments_splits_missing_hour() -> None:
    index = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00", "2024-01-01 04:00"]
    )
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)

    segments = contiguous_observed_segments(series, min_len=2)

    assert [segment.tolist() for segment in segments] == [[1.0, 2.0], [3.0, 4.0]]
    assert observed_blocks(series)["hours"].tolist() == [2, 2]


def test_worst_case_requirements_use_native_model_geometry() -> None:
    requirements = _worst_case_requirements(
        ("TCN", "RNN"),
        context=72,
        horizon=12,
        stride=6,
        validation_hours=48,
        seasonality_m=24,
    )

    assert requirements == {
        "minimum_hours": 84,
        "prediction_context_hours": 72,
        "validation_hours": 108,
        "host_minimum_hours": 192,
        "validation_forecasts": 7,
        "limiting_models": "TCN",
    }


def test_strict_requirements_exclude_foundations_from_training_arm_comparison() -> None:
    configs = resolve_forecasting_model_configs(
        ["AutoARIMA", "Chronos2"], seasonality_m=24, context_length=72
    )

    requirements = get_strict_forecast_requirements(
        configs,
        size_k=8,
        validation_len=48,
        validation_stride=6,
        seasonality_m=24,
        context_len=72,
        training_arms_only=True,
    )

    assert requirements["minimum_hours"] == 48
    assert requirements["minimum_models"] == "AutoARIMA"
    assert requirements["prediction_context_hours"] == 72
    assert requirements["context_models"] == "configured_context"


def test_analysis_retains_same_run_prefix_before_fixed_holdout(
    tmp_path, monkeypatch
) -> None:
    index = pd.date_range("2024-01-01", periods=500, freq="h")
    hourly = pd.DataFrame({"NO2": range(500)}, index=index, dtype=float)
    monkeypatch.setattr(
        block_analysis,
        "load_raw_5m",
        lambda *_args, **_kwargs: [("ST0", hourly)],
    )
    monkeypatch.setattr(
        block_analysis,
        "preprocess",
        lambda *_args, **_kwargs: ([hourly], {}),
    )

    blocks, series, excluded = analyze_raw_blocks(
        tmp_path,
        ("NO2",),
        context=72,
        horizon=12,
        stride=6,
        validation_len=48,
        holdout=96,
        min_run=1,
        min_useful=1,
        forecast_models=("NLinear", "TiDE"),
        seasonality_m=24,
    )

    assert excluded.empty
    assert series.iloc[0]["test_target_hours"] == 96
    assert series.iloc[0]["prior_observed_hours"] == 404
    assert blocks["hours"].sum() == 404
    assert series.iloc[0]["source_run_start"] == index[0]


def test_analysis_rejects_validation_shorter_than_horizon(tmp_path) -> None:
    with pytest.raises(ValueError, match="deben ser validos"):
        run_analysis(
            base_dir=tmp_path,
            output_dir=tmp_path / "out",
            validation_len=4,
        )

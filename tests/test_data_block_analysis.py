"""Focused checks for the data-block pre-study."""

from __future__ import annotations

import pandas as pd
import pytest

from airquality.data.block_analysis import (
    _worst_case_requirements,
    classify_blocks,
    observed_blocks,
    run_analysis,
    summarize_blocks,
)
from airquality.data.segments import contiguous_observed_segments


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
        {"short": 80, "long": 120},
        validation_hours={"short": 48, "long": 96},
        host_minimum_hours={"short": 128, "long": 216},
    )

    # Short can validate on the newest block; long must validate on the older
    # 220-hour block and therefore cannot train on the posterior 130-hour block.
    assert classified["short_used"].tolist() == [True, True]
    assert classified["short_training_hours"].tolist() == [220, 82]
    assert classified["long_used"].tolist() == [True, False]
    assert classified["long_training_hours"].tolist() == [124, 0]

    classified.insert(0, "station", "S1")
    classified.insert(0, "pollutant", "NO2")
    series_summary = pd.DataFrame(
        {
            "pollutant": ["NO2"],
            "forecast_models": ["NLinear, TiDE"],
            "short_minimum_hours": [80],
            "short_horizon_hours": [8],
            "short_stride_hours": [4],
            "short_limiting_models": ["NLinear/TiDE"],
            "short_requested_validation_hours": [48],
            "short_validation_reserve_hours": [48],
            "short_validation_forecasts": [11],
            "short_host_minimum_hours": [128],
            "short_validation_hours": [48],
            "long_minimum_hours": [120],
            "long_horizon_hours": [48],
            "long_stride_hours": [24],
            "long_limiting_models": ["NLinear/TiDE"],
            "long_requested_validation_hours": [96],
            "long_validation_reserve_hours": [96],
            "long_validation_forecasts": [3],
            "long_host_minimum_hours": [216],
            "long_validation_hours": [96],
        }
    )
    total = summarize_blocks(classified, series_summary).iloc[-1]
    assert total["short_used_blocks"] == 2
    assert total["short_validation_forecasts"] == 11
    assert total["long_used_blocks"] == 1
    assert total["long_validation_forecasts"] == 3
    assert total["long_unused_eligible_blocks"] == 1


def test_contiguous_observed_segments_splits_missing_hour() -> None:
    index = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00", "2024-01-01 04:00"]
    )
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)

    segments = contiguous_observed_segments(series, min_len=2)

    assert [segment.tolist() for segment in segments] == [[1.0, 2.0], [3.0, 4.0]]


def test_worst_case_requirements_use_native_model_geometry() -> None:
    requirements = _worst_case_requirements(
        ("TCN", "RNN"),
        context=72,
        horizons={"short": 8, "long": 48},
        strides={"short": 4, "long": 24},
        validation_hours={"short": 48, "long": 96},
        seasonality_m=24,
    )

    assert requirements["short"] == {
        "minimum_hours": 80,
        "prediction_context_hours": 72,
        "host_minimum_hours": 152,
        "validation_hours": 72,
        "validation_forecasts": 1,
        "limiting_models": "TCN",
    }
    assert requirements["long"]["host_minimum_hours"] == 216
    assert requirements["long"]["limiting_models"] == "TCN"


def test_analysis_rejects_validation_shorter_than_horizon(tmp_path) -> None:
    with pytest.raises(ValueError, match="deben ser validos"):
        run_analysis(
            base_dir=tmp_path,
            output_dir=tmp_path / "out",
            short_validation_len=4,
        )

"""Focused checks for the data-block pre-study."""

from __future__ import annotations

import pandas as pd
import pytest

from airquality.data.block_analysis import (
    _training_window,
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
            "short_minimum_hours": [80],
            "short_horizon_hours": [8],
            "short_stride_hours": [4],
            "short_validation_forecasts": [11],
            "short_host_minimum_hours": [128],
            "short_validation_hours": [48],
            "long_minimum_hours": [120],
            "long_horizon_hours": [48],
            "long_stride_hours": [24],
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


def test_training_window_reserves_complete_test_block() -> None:
    index = pd.date_range("2024-01-01", periods=500, freq="h")
    series = pd.Series(range(500), index=index, dtype=float)
    series.iloc[168:180] = None

    train, details = _training_window(
        series,
        holdout=192,
        context=72,
        host_minimum=168,
        test_alignment=48,
    )

    assert train is not None and train.index[-1] == index[179]
    assert train.last_valid_index() == index[167]
    assert details["test_block_start"] == index[180]
    assert details["test_hours"] == 240


def test_analysis_rejects_validation_shorter_than_horizon(tmp_path) -> None:
    with pytest.raises(ValueError, match="deben ser validos"):
        run_analysis(
            base_dir=tmp_path,
            output_dir=tmp_path / "out",
            short_validation_len=4,
        )

import numpy as np
import pandas as pd

from airquality.data.detected_block_analysis import _station_stats, _summarize


def test_station_stats_keeps_unscored_points_in_blocks() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="h")
    series = pd.Series(np.arange(10.0), index=index)
    mask = pd.Series(False, index=index)
    mask.iloc[8] = True
    scored = pd.Series(False, index=index)
    scored.iloc[5:] = True

    row, blocks = _station_stats("test", "station", series, mask, scored)

    assert row["detection_scored_points"] == 5
    assert row["detection_unscored_points"] == 5
    assert row["flagged_points"] == 1
    assert list(blocks["hours"]) == [8, 1]


def test_summarize_handles_scenario_without_remaining_blocks() -> None:
    series_summary = pd.DataFrame(
        [
            {
                "scenario": "Raw",
                "station": "S",
                "input_observed_points": 1,
                "detection_scored_points": 1,
                "detection_unscored_points": 0,
                "flagged_points": 0,
                "remaining_points": 1,
            },
            {
                "scenario": "Detector",
                "station": "S",
                "input_observed_points": 1,
                "detection_scored_points": 1,
                "detection_unscored_points": 0,
                "flagged_points": 1,
                "remaining_points": 0,
            },
        ]
    )
    blocks = pd.DataFrame(
        [{"scenario": "Raw", "station": "S", "hours": 1}]
    )

    summary = _summarize(series_summary, blocks).set_index("scenario")

    assert summary.loc["Detector", "total_blocks"] == 0
    assert summary.loc["Detector", "max_block_hours"] == 0

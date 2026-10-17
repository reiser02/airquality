from __future__ import annotations

import pandas as pd

from airquality.visualizations.station_coverage import (
    _combined_station_name,
    _coverage_summary,
    _representative_station_subset,
    save_combined_station_coverage,
    save_comparison_station_coverage,
    save_station_coverage,
)


def test_combined_station_name_merges_upct_pollutants() -> None:
    assert _combined_station_name("UPCT_NO2") == "UPCT"
    assert _combined_station_name("UPCT_O3") == "UPCT"
    assert _combined_station_name("AQ1 BusCT") == "AQ1 BusCT"


def test_representative_station_subset_handles_three_stations() -> None:
    assert _representative_station_subset(["A", "B", "C"]) == ["B"]


def test_station_coverage_counts_blocks_and_renders(tmp_path) -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h")
    frame = pd.DataFrame({"NO2": [1.0, 2.0, None, 3.0, None, None]}, index=index)

    blocks, missing_pct = _coverage_summary(frame["NO2"])
    output = save_station_coverage(tmp_path / "coverage.png", [("ST1", frame)])
    combined = save_combined_station_coverage(
        tmp_path / "combined.png",
        {
            "NO2": [("ST1", frame)],
            "CO": [("ST1", frame.rename(columns={"NO2": "CO"}))],
            "O3": [("ST1", frame.rename(columns={"NO2": "O3"}))],
        },
    )

    assert blocks["hours"].tolist() == [2, 1]
    assert missing_pct == 50.0
    assert output.exists()
    assert combined.exists()


def test_comparison_station_coverage_renders_suppressed_and_unsuppressed_sources(
    tmp_path,
) -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h")
    suppressed = pd.DataFrame(
        {"NO2": [1.0, 2.0, None, 3.0, None, None]}, index=index
    )
    unsuppressed = pd.DataFrame(
        {"NO2": [1.0, None, 2.0, 3.0, None, 4.0]}, index=index
    )

    output = save_comparison_station_coverage(
        tmp_path / "comparison.png",
        {
            "NO2": [("ST1", suppressed)],
            "CO": [("ST1", suppressed.rename(columns={"NO2": "CO"}))],
            "O3": [("ST1", suppressed.rename(columns={"NO2": "O3"}))],
        },
        {
            "NO2": [("ST1", unsuppressed)],
            "CO": [("ST1", unsuppressed.rename(columns={"NO2": "CO"}))],
            "O3": [("ST1", unsuppressed.rename(columns={"NO2": "O3"}))],
        },
    )

    assert output.exists()

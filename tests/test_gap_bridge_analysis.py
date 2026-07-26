import numpy as np
import pandas as pd

from airquality.data.gap_bridge_analysis import bridge_components, summarize_components


def test_bridge_components_counts_new_and_extended_support() -> None:
    index = pd.date_range("2024-01-01", periods=200, freq="h")
    series = pd.Series(np.nan, index=index, name="NO2")
    series.iloc[:40] = 1.0
    series.iloc[42:80] = 2.0  # 78 real + 2 imputed -> new 80-point component
    series.iloc[90:170] = 3.0
    series.iloc[173:180] = 4.0  # extends the existing 80-point block over a gap of 3

    components = bridge_components(series, "station", minimum=80, max_gap=5)
    summary = summarize_components(components).iloc[0]

    assert list(components["recovery_type"]) == ["new", "extension"]
    assert list(components["span_points"]) == [80, 90]
    assert list(components["imputed_points"]) == [2, 3]
    assert summary["new_eligible_components"] == 1
    assert summary["extended_eligible_components"] == 1
    assert summary["short_blocks_recovered"] == 3
    assert summary["baseline_real_points"] == 80
    assert summary["recovered_real_points"] == 85
    assert summary["imputed_points"] == 5
    assert summary["usable_points_after"] == 170


def test_short_component_is_not_reported_as_recovered() -> None:
    index = pd.date_range("2024-01-01", periods=20, freq="h")
    series = pd.Series(1.0, index=index)

    components = bridge_components(series, "station", minimum=80, max_gap=5)

    assert components.iloc[0]["recovery_type"] == "none"
    assert components.iloc[0]["recovered_real_points"] == 0


def test_gain_percentage_is_undefined_without_baseline_support() -> None:
    index = pd.date_range("2024-01-01", periods=80, freq="h")
    series = pd.Series(1.0, index=index)
    series.iloc[40:42] = np.nan

    summary = summarize_components(
        bridge_components(series, "station", minimum=80, max_gap=5)
    ).iloc[0]

    assert summary["usable_points_after"] == 80
    assert np.isnan(summary["usable_gain_pct"])

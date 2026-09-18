from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from airquality.imputation.pollutant_statistics import (
    analyze_global_scores,
    build_comparison_panel,
    build_global_station_scores,
    write_global_comparison,
)


def _results(*, pollutant: str, station_names: list[str], scale: float = 1.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for station_index, station in enumerate(station_names):
        for model_index, model in enumerate(("A", "B")):
            for gap_size in (1, 5):
                for run in range(1, 4):
                    base = (1.0 + station_index / 10.0) * (1.0 + model_index / 10.0)
                    value = scale * base * (1.0 + gap_size / 100.0 + run / 1000.0)
                    same_protocol = station in {"S0", "S1"}
                    plan_suffix = "shared" if same_protocol else pollutant
                    holdout = "2026-01-01 00:00:00" if same_protocol else f"2026-02-{station_index + 1:02d} 00:00:00"
                    rows.append(
                        {
                            "Modelo": model,
                            "Serie": station,
                            "Gap_Size": gap_size,
                            "N_Gaps_Target": 30 // gap_size,
                            "N_Gaps_Scored": 30 // gap_size,
                            "N_Target_Points": 30,
                            "N_Scored_Points": 30,
                            "Support_Fraction": 1.0,
                            "Gap_Plan_ID": f"{station}-{gap_size}-{run}-{plan_suffix}",
                            "Test_Start": holdout,
                            "Test_End": "2026-01-08 23:00:00"
                            if same_protocol
                            else "2026-03-08 23:00:00",
                            "Test_Block_Points": 192,
                            "MASE": value,
                            "RMSSE": value * 1.1,
                            "Seed": 41 + run,
                            "MonteCarlo_Run": run,
                        }
                    )
    return pd.DataFrame(rows)


def test_upct_alias_pairs_all_physical_stations_and_keeps_protocol_diagnostic() -> None:
    left = _results(
        pollutant="NO2",
        station_names=["S0", "S1", "S2", "UPCT_NO2"],
    )
    right = _results(
        pollutant="O3",
        station_names=["S0", "S1", "S2", "UPCT_O3"],
    )

    panel = build_comparison_panel(left, right)

    assert len(panel) == 4 * 2 * 2 * 3
    assert set(panel["station"]) == {"S0", "S1", "S2", "UPCT"}
    assert panel["same_protocol"].sum() == 2 * 2 * 2 * 3
    assert set(panel.loc[panel["same_protocol"], "station"]) == {"S0", "S1"}


def test_global_score_averages_seeds_before_model_and_gap_catalog() -> None:
    left = _results(
        pollutant="NO2",
        station_names=["S0", "S1", "S2", "UPCT_NO2"],
    )
    right = _results(
        pollutant="O3",
        station_names=["S0", "S1", "S2", "UPCT_O3"],
        scale=2.0,
    )
    panel = build_comparison_panel(left, right)

    scores = build_global_station_scores(panel)

    assert len(scores) == 4
    assert (scores["n_model_gap_cells"] == 4).all()
    assert (scores["n_runs_min"] == 3).all()
    assert (scores["n_runs_max"] == 3).all()
    assert np.allclose(scores["MASE_O3"] / scores["MASE_NO2"], 2.0)


def test_global_analysis_reports_paired_tests_and_equivalence() -> None:
    left = _results(
        pollutant="NO2",
        station_names=[f"S{index}" for index in range(12)],
    )
    right = _results(
        pollutant="O3",
        station_names=[f"S{index}" for index in range(12)],
        scale=1.0,
    )
    panel = build_comparison_panel(left, right, station_aliases={})
    scores = build_global_station_scores(panel)
    summary = analyze_global_scores(
        scores,
        seed=7,
    )

    assert list(summary["metric"]) == ["MASE", "RMSSE"]
    assert (summary["n_stations"] == 12).all()
    assert summary["primary"].tolist() == [True, False]
    assert summary["p_wilcoxon"].between(0.0, 1.0).all()
    assert summary["p_wilcoxon_holm"].between(0.0, 1.0).all()


def test_writer_persists_both_scopes_and_metadata(tmp_path) -> None:
    left = _results(
        pollutant="NO2",
        station_names=["S0", "S1", "S2", "UPCT_NO2"],
    )
    right = _results(
        pollutant="O3",
        station_names=["S0", "S1", "S2", "UPCT_O3"],
    )
    left_dir = tmp_path / "no2"
    right_dir = tmp_path / "o3"
    left_dir.mkdir()
    right_dir.mkdir()
    left.to_csv(left_dir / "results_mc.csv", index=False)
    right.to_csv(right_dir / "results_mc.csv", index=False)
    output_dir = tmp_path / "comparison"

    artifacts = write_global_comparison(
        left_dir,
        right_dir,
        output_dir=output_dir,
        seed=5,
    )

    assert set(artifacts["paths"]) == {
        "station_scores",
        "summary",
        "protocol",
        "metadata",
        "report",
    }
    assert all(path.exists() for path in artifacts["paths"].values())
    station_scores = pd.read_csv(artifacts["paths"]["station_scores"])
    assert set(station_scores["scope"]) == {"all_stations", "same_protocol"}
    metadata = json.loads(artifacts["paths"]["metadata"].read_text(encoding="utf-8"))
    assert metadata["n_stations"] == 4
    assert len(metadata["same_protocol_stations"]) == 2


def test_comparison_rejects_incomplete_panels() -> None:
    left = _results(pollutant="NO2", station_names=["S0", "S1"])
    right = _results(pollutant="O3", station_names=["S0", "S1"])
    right = right.iloc[:-1]

    with pytest.raises(ValueError, match="mismo panel"):
        build_comparison_panel(left, right)

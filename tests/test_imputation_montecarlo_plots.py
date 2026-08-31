from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airquality.visualizations.montecarlo import (
    _aggregate_metrics_by_gap,
    _error_correlation_table,
    _gap_degradation_table,
    _global_performance_profile_table,
    _global_rank_frequency_tables,
    _global_station_error_table,
    _model_performance_tables,
    _pairwise_win_rate_table,
    _tail_risk_table,
    render_run_figures,
)


def test_aggregate_metrics_by_gap_weights_stations_equally() -> None:
    frame = pd.DataFrame(
        {
            "Modelo": ["M"] * 5,
            "Serie": ["A", "A", "A", "A", "B"],
            "Gap_Size": [1] * 5,
            "Seed": [1, 2, 3, 4, 1],
            "MAE": [1.0, 1.0, 1.0, 1.0, 3.0],
        }
    )

    result = _aggregate_metrics_by_gap(frame, ["MAE"])

    assert result.loc[0, "MAE_Mean"] == pytest.approx(2.0)
    assert result.loc[0, "MAE_N_Stations"] == 2


def test_overall_performance_weights_gaps_and_stations_equally() -> None:
    frame = pd.DataFrame(
        {
            "Modelo": ["M"] * 6,
            "Serie": ["A", "A", "A", "A", "B", "B"],
            "Gap_Size": [1, 1, 1, 10, 1, 10],
            "MASE": [1.0, 1.0, 1.0, 3.0, 5.0, 5.0],
        }
    )

    by_gap, overall = _model_performance_tables(frame, ["MASE"])

    # Station A: mean(1, 3)=2 despite having three repeats at gap 1. Station B=5.
    assert overall.loc[0, "Mean"] == pytest.approx(3.5)
    assert overall.loc[0, "N_Stations"] == 2
    assert overall.loc[0, "N_Gaps"] == 2
    assert set(by_gap["Gap_Size"]) == {1, 10}


def test_performance_tables_average_seeds_before_station_comparisons() -> None:
    frame = pd.DataFrame(
        [
            {
                "Modelo": model,
                "Serie": station,
                "Gap_Size": 1,
                "Seed": seed,
                "MAE": value,
                "Test_Block_Points": test_points,
            }
            for model, station, seed, value, test_points in (
                ("M", "A", 1, 1.0, 100),
                ("M", "A", 2, 3.0, 100),
                ("M", "B", 1, 10.0, 80),
                ("LinearInterp", "A", 1, 4.0, 100),
                ("LinearInterp", "A", 2, 4.0, 100),
                ("LinearInterp", "B", 1, 8.0, 80),
            )
        ]
    )

    by_gap, overall = _model_performance_tables(frame, ["MAE"])
    model_summary = overall.set_index("Modelo").loc["M"]

    assert model_summary["Mean"] == pytest.approx(6.0)
    assert model_summary["Station_SD"] == pytest.approx(np.sqrt(32.0))
    assert model_summary["N_Stations"] == 2
    assert model_summary["Mean_Rank"] == pytest.approx(1.5)
    assert model_summary["Winner_Percent"] == pytest.approx(50.0)
    gap_model = by_gap.set_index(["Gap_Size", "Modelo"]).loc[(1, "M")]
    assert gap_model["Mean"] == pytest.approx(6.0)
    assert gap_model["Mean_Rank"] == pytest.approx(1.5)


def test_global_summaries_derive_support_from_all_26_stations() -> None:
    frame = pd.DataFrame(
        [
            {
                "Modelo": model,
                "Serie": f"Station {station:02d}",
                "Gap_Size": gap,
                "Seed": seed,
                "MAE": value,
                "RMSE": value + 0.5,
                "MASE": value,
            }
            for station in range(26)
            for gap in (1, 10)
            for seed in (10, 20)
            for model, value in (("Best", 1.0), ("Other", 2.0))
        ]
    )

    rank_summary, frequencies = _global_rank_frequency_tables(
        frame, ["MAE", "RMSE", "MASE"]
    )
    best_mae = rank_summary.set_index(["Metric", "Modelo"]).loc[("MAE", "Best")]

    assert best_mae["Mean_Rank"] == pytest.approx(1.0)
    assert best_mae["Winner_Percent"] == pytest.approx(100.0)
    assert best_mae["N_Stations"] == 26
    assert best_mae["N_Gaps"] == 2
    assert best_mae["N_Contexts"] == 52
    frequency_totals = frequencies.groupby(["Metric", "Modelo"])[
        "Context_Percent"
    ].sum()
    assert np.allclose(frequency_totals, 100.0)

    profiles = _global_performance_profile_table(frame, ["MAE", "RMSE", "MASE"])
    winners = profiles[profiles["Threshold_Ratio"] == 1.0].set_index(
        ["Metric", "Modelo"]
    )
    assert winners.loc[("MASE", "Best"), "Context_Percent"] == pytest.approx(100.0)
    assert winners.loc[("MASE", "Other"), "Context_Percent"] == pytest.approx(0.0)
    assert winners.loc[("MASE", "Best"), "N_Stations"] == 26

    station_errors = _global_station_error_table(frame, ["MAE"])
    assert len(station_errors) == 52
    assert station_errors["N_Gaps"].eq(2).all()
    best_station_errors = station_errors[station_errors["Modelo"] == "Best"]
    assert best_station_errors["Error"].eq(1.0).all()

    pairwise = _pairwise_win_rate_table(frame, ["MAE"])
    pairwise = pairwise.set_index(["Modelo", "Opponent"])
    assert pairwise.loc[("Best", "Other"), "Win_Rate_Percent"] == pytest.approx(
        100.0
    )
    assert pairwise.loc[("Other", "Best"), "Win_Rate_Percent"] == pytest.approx(
        0.0
    )
    assert pairwise.loc[("Best", "Best"), "Win_Rate_Percent"] == pytest.approx(
        50.0
    )
    assert pairwise.loc[("Best", "Other"), "N_Stations"] == 26

    degradation = _gap_degradation_table(frame, ["MAE"])
    assert degradation["Degradation_Percent"].eq(0.0).all()
    assert set(degradation["Gap_Size"]) == {1, 10}

    tail_risk = _tail_risk_table(frame, ["MAE"]).set_index("Modelo")
    assert tail_risk.loc["Best", "Mean_Error"] == pytest.approx(1.0)
    assert tail_risk.loc["Best", "Tail_Mean_Error"] == pytest.approx(1.0)

    station_codes = pd.factorize(frame["Serie"])[0]
    varying_frame = frame.assign(MAE=frame["MAE"] + station_codes * 0.01)
    correlations = _error_correlation_table(varying_frame, ["MAE"]).set_index(
        ["Modelo", "Other_Model"]
    )
    assert correlations.loc[("Best", "Other"), "Mean_Spearman"] == pytest.approx(
        1.0
    )


def test_r2_rankings_and_pairwise_wins_maximize_metric() -> None:
    frame = pd.DataFrame(
        [
            {
                "Modelo": model,
                "Serie": station,
                "Gap_Size": gap,
                "R2": value,
            }
            for station in ("A", "B")
            for gap in (1, 2)
            for model, value in (("High", 0.8), ("Low", 0.2))
        ]
    )

    rank_summary, _ = _global_rank_frequency_tables(frame, ["R2"])
    ranks = rank_summary.set_index("Modelo")
    _, overall = _model_performance_tables(frame, ["R2"])
    performance = overall.set_index("Modelo")
    pairwise = _pairwise_win_rate_table(frame, ["R2"]).set_index(
        ["Modelo", "Opponent"]
    )

    assert ranks.loc["High", "Mean_Rank"] == pytest.approx(1.0)
    assert ranks.loc["High", "Winner_Percent"] == pytest.approx(100.0)
    assert performance.loc["High", "Mean_Rank"] == pytest.approx(1.0)
    assert pairwise.loc[("High", "Low"), "Win_Rate_Percent"] == pytest.approx(
        100.0
    )


def test_render_run_figures_adds_compact_global_summaries(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for gap in (1, 2, 6, 24):
        for station_index, station in enumerate(("A", "B", "C"), start=1):
            for seed in (10, 20):
                for model, offset in (("LinearInterp", 0.5), ("ModelX", 0.0)):
                    base = float(gap + station_index + seed / 100.0 + offset)
                    rows.append(
                        {
                            "Modelo": model,
                            "Serie": station,
                            "Gap_Size": gap,
                            "Seed": seed,
                            "MonteCarlo_Run": seed // 10,
                            "MAE": base,
                            "RMSE": base + 0.4,
                            "MASE": base / 5.0,
                            "RMSSE": base / 6.0,
                            "R2": 1.0 - base / 100.0,
                            "Test_Block_Points": 100 if station != "C" else 80,
                            "Test_Hours": 200 if station != "C" else 160,
                        }
                    )
    frame = pd.DataFrame(rows)
    obsolete_plots = (
        "metric_forest_mae.png",
        "station_distribution_mase.png",
        "paired_mean_rank_rmse.png",
        "paired_delta_vs_LinearInterp_mae.png",
        "station_vs_sampling_variability_mae.png",
        "metrics_mean_station_sd_heatmaps.png",
        "metrics_across_stations.png",
        "global_rank_frequency_mae.png",
        "global_performance_profile_mae.png",
    )
    for filename in obsolete_plots:
        (tmp_path / filename).write_bytes(b"obsolete")

    artifacts = render_run_figures(
        tmp_path,
        results_mc_df=frame,
        plot_store={},
    )

    assert artifacts["metric_gap_plot_path"] == tmp_path / "metrics_by_gap.png"
    for key in (
        "model_performance_by_gap_plot_paths",
        "overall_model_performance_plot_paths",
        "global_station_error_plot_paths",
        "pairwise_win_rate_plot_paths",
        "error_correlation_plot_paths",
    ):
        assert set(artifacts[key]) == {"MAE", "RMSE", "MASE", "RMSSE", "R2"}
        assert all(path.exists() for path in artifacts[key].values())
    for key in ("gap_degradation_plot_paths", "tail_risk_plot_paths"):
        assert set(artifacts[key]) == {"MAE", "RMSE", "MASE", "RMSSE"}
        assert all(path.exists() for path in artifacts[key].values())

    expected_tables = (
        "global_rank_summary.csv",
        "global_rank_frequencies.csv",
        "global_performance_profiles.csv",
        "global_station_errors.csv",
        "pairwise_win_rates.csv",
        "gap_degradation.csv",
        "tail_risk.csv",
        "error_correlations.csv",
        "model_performance_by_gap.csv",
        "overall_model_performance.csv",
    )
    assert all((tmp_path / filename).exists() for filename in expected_tables)
    assert set(artifacts["diagnostic_table_paths"].values()) == {
        tmp_path / filename for filename in expected_tables
    }
    gap_table = pd.read_csv(tmp_path / "model_performance_by_gap.csv")
    assert set(gap_table["Metric"]) == {"MAE", "RMSE", "MASE", "RMSSE", "R2"}
    assert gap_table["N_Stations"].eq(3).all()
    assert all(not (tmp_path / filename).exists() for filename in obsolete_plots)

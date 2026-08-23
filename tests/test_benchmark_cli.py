from __future__ import annotations

from pathlib import Path

import pandas as pd

from airquality.benchmark import main, run_benchmark_from_config
from airquality.visualizations.montecarlo import (
    load_plot_store,
    main as plot_main,
)


def test_run_benchmark_from_config_uses_parallel_runners_and_saves_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    results = pd.DataFrame(
        [
            {"Modelo": "TiDE", "Gap_Size": 1, "MAE": 1.0},
            {"Modelo": "TiDE", "Gap_Size": 2, "MAE": 1.5},
        ]
    )
    summary = pd.DataFrame([{"Modelo": "TiDE", "MAE_Mean": 1.0}])
    ranking_by_seed = pd.DataFrame([{"Modelo": "TiDE", "Seed": 42, "MAE": 1.0}])
    plot_store = {
        1: {
            "series": {
                "Series A": {
                    "actual": pd.Series(
                        [1.0, 2.0],
                        index=pd.date_range("2024-01-01", periods=2, freq="h"),
                    ),
                    "preds": {
                        "TiDE": pd.Series(
                            [1.1],
                            index=pd.date_range("2024-01-01", periods=1, freq="h"),
                        )
                    },
                }
            }
        }
    }

    monkeypatch.setattr(
        "airquality.benchmark.run_imputation_benchmark_parallel_montecarlo",
        lambda: (results, summary, ranking_by_seed, plot_store),
    )
    monkeypatch.setattr("airquality.benchmark._build_output_dir", lambda: tmp_path)

    artifacts = run_benchmark_from_config()

    assert artifacts["results_mc_df"] is results
    assert artifacts["summary_mc_df"] is summary
    assert artifacts["ranking_by_seed_df"] is ranking_by_seed
    assert (tmp_path / "results_mc.csv").exists()
    assert (tmp_path / "summary_mc.csv").exists()
    assert (tmp_path / "ranking_by_seed.csv").exists()
    assert artifacts["plot_store_path"] == tmp_path / "plot_store.csv.gz"
    assert (tmp_path / "plot_store.csv.gz").exists()
    assert (tmp_path / "plot_images.csv").exists()
    assert (tmp_path / "plots" / "gap_1" / "Series_A.png").exists()
    # Aggregated metric-by-gap artifacts (previously never generated).
    assert artifacts["metric_gap_plot_path"] == tmp_path / "metrics_by_gap.png"
    assert (tmp_path / "metrics_by_gap.csv").exists()
    assert (tmp_path / "metrics_by_gap.png").exists()
    assert set(artifacts["model_performance_by_gap_plot_paths"]) == {"MAE"}
    assert set(artifacts["overall_model_performance_plot_paths"]) == {"MAE"}
    assert set(artifacts["global_station_error_plot_paths"]) == {"MAE"}
    assert set(artifacts["pairwise_win_rate_plot_paths"]) == {"MAE"}
    assert set(artifacts["gap_degradation_plot_paths"]) == {"MAE"}
    assert set(artifacts["tail_risk_plot_paths"]) == {"MAE"}
    assert set(artifacts["error_correlation_plot_paths"]) == {"MAE"}
    assert (tmp_path / "model_performance_by_gap_mae.png").exists()
    assert (tmp_path / "overall_model_performance_mae.png").exists()
    assert (tmp_path / "global_station_error_mae.png").exists()
    assert (tmp_path / "pairwise_win_rate_mae.png").exists()
    assert (tmp_path / "gap_degradation_mae.png").exists()
    assert (tmp_path / "tail_risk_mae.png").exists()
    assert (tmp_path / "error_correlation_mae.png").exists()
    assert (tmp_path / "model_performance_by_gap.csv").exists()
    assert (tmp_path / "overall_model_performance.csv").exists()
    assert (tmp_path / "global_rank_summary.csv").exists()
    assert (tmp_path / "global_rank_frequencies.csv").exists()
    assert (tmp_path / "global_performance_profiles.csv").exists()
    assert (tmp_path / "global_station_errors.csv").exists()
    assert (tmp_path / "pairwise_win_rates.csv").exists()
    assert (tmp_path / "gap_degradation.csv").exists()
    assert (tmp_path / "tail_risk.csv").exists()
    assert (tmp_path / "error_correlations.csv").exists()

    metrics_by_gap_df = pd.read_csv(tmp_path / "metrics_by_gap.csv")
    assert list(metrics_by_gap_df["Gap_Size"]) == [1, 2]
    assert metrics_by_gap_df["MAE_Mean"].tolist() == [1.0, 1.5]

    plot_manifest_df = pd.read_csv(tmp_path / "plot_images.csv")
    assert list(plot_manifest_df["image_path"]) == ["plots/gap_1/Series_A.png"]

    restored = load_plot_store(tmp_path / "plot_store.csv.gz")
    restored_payload = restored[1]["series"]["Series A"]
    pd.testing.assert_series_equal(
        restored_payload["actual"], plot_store[1]["series"]["Series A"]["actual"],
        check_freq=False, check_names=False,
    )
    image_path = tmp_path / "plots" / "gap_1" / "Series_A.png"
    image_path.unlink()
    plot_main([str(tmp_path)])
    assert image_path.exists()


def test_main_prints_ranking_summary(monkeypatch, capsys) -> None:
    summary = pd.DataFrame(
        [
            {"Modelo": "TiDE", "MAE_Mean": 1.0},
            {"Modelo": "TCN", "MAE_Mean": 2.0},
        ]
    )

    monkeypatch.setattr(
        "airquality.benchmark.run_benchmark_from_config",
        lambda: {
            "output_dir": Path("/tmp/bench"),
            "results_mc_df": pd.DataFrame(),
            "summary_mc_df": summary,
            "ranking_by_seed_df": pd.DataFrame(),
            "plot_manifest_df": pd.DataFrame(),
            "plot_store_path": Path("/tmp/bench/plot_store.csv.gz"),
        },
    )

    main()

    out = capsys.readouterr().out
    assert "Monte Carlo benchmark summary" in out
    assert "TiDE" in out
    assert "TCN" in out
    assert "Saved benchmark artifacts under /tmp/bench" in out

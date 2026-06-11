from __future__ import annotations

from pathlib import Path

import pandas as pd

from airquality.benchmark import main, run_benchmark_from_config


def test_run_benchmark_from_config_uses_parallel_runners_and_saves_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    results = pd.DataFrame([{"Modelo": "TiDE", "MAE": 1.0}])
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
        lambda: (results, summary, ranking_by_seed),
    )
    monkeypatch.setattr(
        "airquality.benchmark.run_imputation_benchmark_parallel",
        lambda **kwargs: (pd.DataFrame(), pd.DataFrame(), plot_store),
    )
    monkeypatch.setattr("airquality.benchmark.cfg_get_int", lambda *args, **kwargs: 42)
    monkeypatch.setattr("airquality.benchmark._build_output_dir", lambda: tmp_path)

    artifacts = run_benchmark_from_config()

    assert artifacts["results_mc_df"] is results
    assert artifacts["summary_mc_df"] is summary
    assert artifacts["ranking_by_seed_df"] is ranking_by_seed
    assert (tmp_path / "results_mc.csv").exists()
    assert (tmp_path / "summary_mc.csv").exists()
    assert (tmp_path / "ranking_by_seed.csv").exists()
    assert (tmp_path / "plot_images.csv").exists()
    assert (tmp_path / "plots" / "gap_1" / "Series_A.png").exists()

    plot_manifest_df = pd.read_csv(tmp_path / "plot_images.csv")
    assert list(plot_manifest_df["image_path"]) == ["plots/gap_1/Series_A.png"]


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
        },
    )

    main()

    out = capsys.readouterr().out
    assert "Monte Carlo benchmark summary" in out
    assert "TiDE" in out
    assert "TCN" in out
    assert "Saved benchmark artifacts under /tmp/bench" in out

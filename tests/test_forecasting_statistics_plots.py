from __future__ import annotations

import json

import pandas as pd

from airquality.forecasting.statistics import write_statistical_analysis
from airquality.visualizations.forecasting_statistics import render_statistical_report


def _results() -> pd.DataFrame:
    arms = {
        "raw": 1.0,
        "raw+frozen": 1.1,
        "clean-a": 0.7,
        "clean-b": 1.2,
    }
    models = {"M1": 0.8, "M2": 1.0, "M3": 1.25}
    rows = []
    for station_index in range(12):
        station_factor = 1.0 + station_index / 50.0
        for model, model_factor in models.items():
            for arm, arm_factor in arms.items():
                mase = station_factor * model_factor * arm_factor
                rows.append(
                    {
                        "series": f"S{station_index:02d}",
                        "arm": arm,
                        "model": model,
                        "model_mode": "trained",
                        "mase": mase,
                        "rmsse": 1.2 * mase,
                        "relmae": arm_factor,
                        "relrmse": arm_factor,
                        "horizon": 12,
                        "forecast_stride": 6,
                        "validation_len": 48,
                        "validation_stride": 6,
                        "test_context_start": "2026-01-01 00:00:00",
                        "test_target_start": "2026-01-04 00:00:00",
                        "test_target_end": "2026-01-07 23:00:00",
                        "test_target_hours": 96,
                        "n_test_predictions": 180,
                        "n_forecasts": 15,
                        "n_expected_forecasts": 15,
                        "n_unique_targets": 96,
                        "status": "ok",
                        "failure_reason": "",
                    }
                )
    return pd.DataFrame(rows)


def _write_manifest(path, results: pd.DataFrame) -> None:
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            "holdout": 96,
            "horizon": 12,
            "forecast_stride": 6,
            "validation_len": 48,
            "validation_stride": 6,
            "context_len": 72,
        },
        "config": {"seed": 13},
        "models": [
            {"name": model, "mode": "trained", "uses_training_arms": True}
            for model in results["model"].drop_duplicates()
        ],
        "arms": [{"name": arm} for arm in results["arm"].drop_duplicates()],
        "selected_stations": results["series"].drop_duplicates().tolist(),
        "expected_result_rows": len(results),
        "result_rows": len(results),
        "failures": [],
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_report_renders_branch_scores_pairwise_plots_and_tables(tmp_path) -> None:
    results = _results()
    results.to_csv(tmp_path / "results.csv", index=False)
    _write_manifest(tmp_path, results)
    write_statistical_analysis(
        tmp_path,
        n_permutations=199,
        n_bootstrap=199,
        seed=7,
    )

    outputs = render_statistical_report(tmp_path)

    output_names = {path.name for path in outputs}
    assert "statistical_branch_scores.png" in output_names
    assert "statistical_branch_pairwise_mase.png" in output_names
    assert "statistical_branch_pairwise_rmsse.png" in output_names
    assert "statistical_branch_scores_summary.csv" in output_names
    assert "statistical_branch_pairwise.csv" in output_names
    assert "statistical_branch_tables.md" in output_names
    summary = pd.read_csv(tmp_path / "statistical_branch_scores_summary.csv")
    assert set(summary["metric"]) == {"mase", "rmsse"}
    assert set(summary["branch"]) == {"raw", "clean-a", "clean-b"}
    report = (tmp_path / "statistical_report.md").read_text(encoding="utf-8")
    assert "Ramas: scores usados en los contrastes" in report
    assert "Ramas: comparaciones post-hoc MASE" in report
    assert "Ramas mejores que raw con respaldo confirmatorio" in report
    assert "| clean-a" in report

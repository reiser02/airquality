from __future__ import annotations

from pathlib import Path

import pandas as pd

from airquality.imputation.latex_tables import (
    build_gap_summary,
    build_overall_summary,
    build_profile_summary,
    export_latex_tables,
)


def _overall_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": metric,
                "Modelo": model,
                "Mean": mean,
                "Median": mean - 0.1,
                "Station_SD": 0.2,
                "Mean_Rank": rank,
                "Winner_Percent": 50.0,
                "Top_3_Percent": 100.0,
                "N_Stations": 2,
                "N_Gaps": 2,
                "N_Contexts": 4,
            }
            for metric in ("MAE", "RMSE", "MASE", "RMSSE")
            for model, mean, rank in (("A_Model", 1.0, 1.0), ("B_Model", 2.0, 2.0))
        ]
    )


def test_summary_tables_select_and_sort_models() -> None:
    overall = _overall_frame()
    summary = build_overall_summary(overall, top_k=1)

    assert summary["Metric"].tolist() == ["MAE", "RMSE", "MASE", "RMSSE"]
    assert summary["Modelo"].tolist() == ["A_Model"] * 4
    assert summary["Rank_Mean"].tolist() == [1] * 4

    by_gap = overall.assign(Gap_Size=1).rename(columns={"Mean_Rank": "Mean_Rank"})
    gap_summary = build_gap_summary(by_gap, top_k=1)
    assert len(gap_summary) == 4
    assert gap_summary["Modelo"].tolist() == ["A_Model"] * 4


def test_profile_summary_keeps_relative_tolerances() -> None:
    overall = _overall_frame()
    rank_summary = overall[["Metric", "Modelo", "Mean_Rank"]]
    profiles = pd.DataFrame(
        [
            {
                "Metric": metric,
                "Modelo": model,
                "Threshold_Ratio": 1.0 + tolerance / 100.0,
                "Tolerance_Percent": tolerance,
                "Context_Percent": 100.0 if model == "A_Model" else 50.0,
            }
            for metric in ("MAE", "RMSE", "MASE", "RMSSE")
            for model in ("A_Model", "B_Model")
            for tolerance in (0.0, 5.0, 10.0, 25.0, 50.0)
        ]
    )

    result = build_profile_summary(profiles, rank_summary, overall)

    assert list(result.columns[:4]) == ["Metric", "Modelo", "Within_Best", "Within_5pct"]
    assert result.loc[result["Modelo"] == "A_Model", "Within_50pct"].eq(100.0).all()


def test_export_latex_tables_reads_run_csvs(tmp_path: Path) -> None:
    overall = _overall_frame()
    by_gap = overall.assign(Gap_Size=1)
    rank_summary = overall[["Metric", "Modelo", "Mean_Rank", "Winner_Percent", "Top_3_Percent"]]
    profiles = pd.DataFrame(
        [
            {
                "Metric": metric,
                "Modelo": model,
                "Threshold_Ratio": 1.0 + tolerance / 100.0,
                "Tolerance_Percent": tolerance,
                "Context_Percent": 100.0,
            }
            for metric in ("MAE", "RMSE", "MASE", "RMSSE")
            for model in ("A_Model", "B_Model")
            for tolerance in (0.0, 5.0, 10.0, 25.0, 50.0)
        ]
    )
    overall.to_csv(tmp_path / "overall_model_performance.csv", index=False)
    by_gap.to_csv(tmp_path / "model_performance_by_gap.csv", index=False)
    rank_summary.to_csv(tmp_path / "global_rank_summary.csv", index=False)
    profiles.to_csv(tmp_path / "global_performance_profiles.csv", index=False)

    outputs = export_latex_tables(tmp_path, top_k=1)

    assert len(outputs) == 6
    assert (tmp_path / "latex_tables/overall_model_summary.tex").exists()
    assert "A\\_Model" in (
        tmp_path / "latex_tables/overall_model_summary.tex"
    ).read_text(encoding="utf-8")
    assert "RMSSE" in (
        tmp_path / "latex_tables/overall_model_summary.tex"
    ).read_text(encoding="utf-8")

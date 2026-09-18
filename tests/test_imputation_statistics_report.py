from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from airquality.visualizations.imputation_statistics import (
    build_model_summary,
    build_top_comparisons,
    render_imputation_statistical_report,
)


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = ("B_Model", "A_Model", "C_Model")
    factors = {"A_Model": 0.7, "B_Model": 1.0, "C_Model": 1.4}
    score_rows = []
    for metric in ("MASE", "RMSSE"):
        for gap_size in (1, 5):
            for station_index in range(8):
                values = {
                    model: factors[model]
                    * (1.0 + station_index / 20.0)
                    * gap_size
                    * (1.2 if metric == "RMSSE" else 1.0)
                    for model in models
                }
                ordered = sorted(values, key=values.get)
                for model in models:
                    score_rows.append(
                        {
                            "metric": metric,
                            "gap_size": gap_size,
                            "series": "NA" if station_index == 0 else f"S{station_index}",
                            "model": model,
                            "score": values[model],
                            "rank": ordered.index(model) + 1,
                        }
                    )
    scores = pd.DataFrame(score_rows)
    omnibus = pd.DataFrame(
        [
            {
                "metric": metric,
                "gap_size": gap_size,
                "test": "friedman",
                "statistic": 16.0,
                "p_value": 0.0003,
                "alpha": 0.05,
                "significant": True,
                "confirmatory": metric == "MASE",
                "n_stations": 8,
                "n_models": 3,
            }
            for metric in ("MASE", "RMSSE")
            for gap_size in (1, 5)
        ]
    )
    pair_rows = []
    for metric in ("MASE", "RMSSE"):
        for gap_size in (1, 5):
            means = {
                model: scores.loc[
                    (scores["metric"] == metric)
                    & (scores["gap_size"] == gap_size)
                    & (scores["model"] == model),
                    "score",
                ].mean()
                for model in models
            }
            for left_index, left in enumerate(models):
                for right in models[left_index + 1 :]:
                    difference = means[left] - means[right]
                    left_better = difference < 0.0
                    pair_rows.append(
                        {
                            "metric": metric,
                            "gap_size": gap_size,
                            "left": left,
                            "right": right,
                            "left_mean": means[left],
                            "right_mean": means[right],
                            "mean_difference": difference,
                            "improvement_pct": 100 * (1 - means[left] / means[right]),
                            "rank_biserial": -1.0 if left_better else 1.0,
                            "wins": 8 if left_better else 0,
                            "ties": 0,
                            "losses": 0 if left_better else 8,
                            "p_holm": 0.01,
                            "alpha": 0.05,
                            "statistically_significant": True,
                        }
                    )
    return scores, omnibus, pd.DataFrame(pair_rows)


def test_model_summary_uses_station_scores_and_mean_ranks() -> None:
    scores, _, _ = _tables()

    summary = build_model_summary(scores, metric="MASE")

    assert summary.iloc[0]["model"] == "A_Model"
    assert set(summary["n_stations"]) == {8}
    assert summary.query("model == 'A_Model'")["mean_rank"].eq(1.0).all()


def test_top_comparisons_reorients_pair_when_reference_is_right() -> None:
    scores, _, pairwise = _tables()

    comparisons = build_top_comparisons(scores, pairwise, metric="MASE", top_k=3)

    row = comparisons.query(
        "gap_size == 1 and reference == 'A_Model' and comparison == 'B_Model'"
    ).iloc[0]
    assert row["mean_difference"] < 0.0
    assert row["mean_difference"] < 0.0
    assert row["rank_biserial"] < 0.0
    assert row["wins"] == 8
    assert row["losses"] == 0
    assert row["improvement_pct"] > 0.0
    assert row["p_holm"] == pytest.approx(0.01)


def test_statistical_report_writes_png_pdf_csv_and_latex(tmp_path: Path) -> None:
    scores, omnibus, pairwise = _tables()
    scores.to_csv(tmp_path / "statistical_station_scores.csv", index=False)
    omnibus.to_csv(tmp_path / "statistical_omnibus.csv", index=False)
    pairwise.to_csv(tmp_path / "statistical_pairwise.csv", index=False)

    outputs = render_imputation_statistical_report(tmp_path)
    names = {path.name for path in outputs}

    assert {
        "station_scores_mase.png",
        "station_scores_mase.pdf",
        "station_scores_rmsse.png",
        "omnibus_by_gap.pdf",
        "pairwise_matrix_mase_gap_1.png",
        "model_summary_mase.csv",
        "model_summary_mase.tex",
        "model_summary_rmsse.tex",
        "omnibus.tex",
        "top_comparisons_mase.tex",
        "README.md",
    }.issubset(names)
    assert all(path.stat().st_size > 100 for path in outputs)
    latex = (tmp_path / "statistical_report/tables/model_summary_mase.tex").read_text(
        encoding="utf-8"
    )
    assert "A\\_Model" in latex
    assert "\\textbf" in latex
    assert "\\usepackage{booktabs}" in (
        tmp_path / "statistical_report/README.md"
    ).read_text(encoding="utf-8")
    comparisons = pd.read_csv(
        tmp_path / "statistical_report/tables/top_comparisons_mase.csv"
    )
    assert set(comparisons["reference"]) == {"A_Model"}


def test_report_validates_input_and_top_k(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="statistical_station_scores.csv"):
        render_imputation_statistical_report(tmp_path)

    scores, omnibus, pairwise = _tables()
    scores.to_csv(tmp_path / "statistical_station_scores.csv", index=False)
    omnibus.to_csv(tmp_path / "statistical_omnibus.csv", index=False)
    pairwise.to_csv(tmp_path / "statistical_pairwise.csv", index=False)
    with pytest.raises(ValueError, match="top_k"):
        render_imputation_statistical_report(tmp_path, top_k=1)


def test_report_explains_when_posthoc_was_not_run(tmp_path: Path) -> None:
    scores, omnibus, pairwise = _tables()
    omnibus[["p_value", "significant"]] = [1.0, False]
    pairwise = pairwise.iloc[0:0]
    scores.to_csv(tmp_path / "statistical_station_scores.csv", index=False)
    omnibus.to_csv(tmp_path / "statistical_omnibus.csv", index=False)
    pairwise.to_csv(tmp_path / "statistical_pairwise.csv", index=False)

    outputs = render_imputation_statistical_report(tmp_path)

    names = {path.name for path in outputs}
    assert "pairwise_matrix_mase_gap_1.pdf" in names
    assert "pairwise_matrix_mase_gap_5.png" in names
    assert pd.read_csv(
        tmp_path / "statistical_report/tables/top_comparisons_mase.csv"
    ).empty

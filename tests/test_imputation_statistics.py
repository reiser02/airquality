from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality.imputation.statistics import (
    analyze_imputation_results,
    build_station_scores,
    write_statistical_analysis,
)


MODELS = ("A", "B", "C")
GAPS = (1, 5)


def _results(n_stations: int = 12, n_runs: int = 4) -> pd.DataFrame:
    factors = {"A": 0.7, "B": 1.0, "C": 1.4}
    rows = []
    for station_index in range(n_stations):
        station = f"S{station_index:02d}"
        station_factor = 1.0 + station_index / 20.0
        for run in range(1, n_runs + 1):
            seed = 40 + run
            run_factor = 1.0 + run / 100.0
            for gap_size in GAPS:
                n_gaps = 30 // gap_size
                for model in MODELS:
                    mase = station_factor * run_factor * factors[model] * gap_size
                    rows.append(
                        {
                            "Modelo": model,
                            "Serie": station,
                            "Gap_Size": gap_size,
                            "N_Gaps_Target": n_gaps,
                            "N_Gaps_Scored": n_gaps,
                            "N_Target_Points": 30,
                            "N_Scored_Points": 30,
                            "Support_Fraction": 1.0,
                            "Test_Start": "2026-01-01 00:00:00",
                            "Test_End": "2026-01-08 23:00:00",
                            "Test_Block_Points": 192,
                            "Train_Points_Before": 500,
                            "Train_Points_After": 100,
                            "MASE": mase,
                            "RMSSE": 1.2 * mase,
                            "Seed": seed,
                            "MonteCarlo_Run": run,
                        }
                    )
    return pd.DataFrame(rows)


def test_station_scores_average_runs_without_mixing_gap_sizes() -> None:
    results = _results()

    scores, exclusions = build_station_scores(results, metric="MASE", gap_size=1)

    assert exclusions.empty
    score = scores.query("series == 'S00' and model == 'A'").iloc[0]
    assert score["score"] == pytest.approx(0.7 * np.mean([1.01, 1.02, 1.03, 1.04]))
    assert score["n_runs"] == 4
    assert score["rank"] == 1.0
    assert set(scores["gap_size"]) == {1}


def test_missing_run_excludes_only_affected_station_and_gap() -> None:
    results = _results()
    missing = (
        (results["Serie"] == "S00")
        & (results["Gap_Size"] == 1)
        & (results["Modelo"] == "B")
        & (results["MonteCarlo_Run"] == 2)
    )
    results = results.loc[~missing]

    gap_one, excluded = build_station_scores(results, metric="MASE", gap_size=1)
    gap_five, gap_five_excluded = build_station_scores(
        results, metric="MASE", gap_size=5
    )

    assert "S00" not in set(gap_one["series"])
    assert excluded.query("series == 'S00'").iloc[0]["reason"] == (
        "incomplete_model_run_panel"
    )
    assert "S00" in set(gap_five["series"])
    assert gap_five_excluded.empty


def test_incomplete_support_and_nonfinite_metrics_exclude_station() -> None:
    support = _results()
    support.loc[
        (support["Serie"] == "S00") & (support["Gap_Size"] == 1),
        "Support_Fraction",
    ] = 0.9
    _, excluded = build_station_scores(support, metric="MASE", gap_size=1)
    assert excluded.query("series == 'S00'").iloc[0]["reason"] == "incomplete_support"

    invalid = _results()
    invalid.loc[
        (invalid["Serie"] == "S01")
        & (invalid["Gap_Size"] == 5)
        & (invalid["Modelo"] == "C"),
        "MASE",
    ] = np.nan
    _, excluded = build_station_scores(invalid, metric="MASE", gap_size=5)
    assert excluded.query("series == 'S01'").iloc[0]["reason"] == (
        "nonfinite_or_invalid_metric"
    )

    invalid_text = _results()
    invalid_text["MASE"] = invalid_text["MASE"].astype(object)
    invalid_text.loc[
        (invalid_text["Serie"] == "S01")
        & (invalid_text["Gap_Size"] == 5)
        & (invalid_text["Modelo"] == "C"),
        "MASE",
    ] = "valor-invalido"
    scores, excluded = build_station_scores(invalid_text, metric="MASE", gap_size=5)
    assert "S00" in set(scores["series"])
    assert "S01" not in set(scores["series"])
    assert excluded.query("series == 'S01'").iloc[0]["reason"] == (
        "nonfinite_or_invalid_metric"
    )


def test_duplicate_keys_and_inconsistent_holdout_are_rejected() -> None:
    results = _results()
    duplicate = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicadas"):
        analyze_imputation_results(duplicate, n_permutations=99)

    mismatch = results.copy()
    mismatch.loc[0, "Test_Start"] = "2026-02-01 00:00:00"
    with pytest.raises(ValueError, match="holdout"):
        analyze_imputation_results(mismatch, n_permutations=99)


def test_known_differences_run_friedman_then_holm_posthoc() -> None:
    artifacts = analyze_imputation_results(_results(), n_permutations=499, seed=7)

    omnibus = artifacts["omnibus"]
    assert set(omnibus["gap_size"]) == set(GAPS)
    assert set(omnibus["metric"]) == {"MASE", "RMSSE"}
    assert set(omnibus["test"]) == {"friedman_permutation"}
    assert omnibus["significant"].all()
    assert omnibus.query("metric == 'MASE'")["confirmatory"].all()
    assert not omnibus.query("metric == 'RMSSE'")["confirmatory"].any()

    pairs = artifacts["pairwise"]
    expected_pairs = len(MODELS) * (len(MODELS) - 1) // 2
    assert len(pairs) == len(GAPS) * 2 * expected_pairs
    pair = pairs.query("metric == 'MASE' and gap_size == 1 and left == 'A' and right == 'C'").iloc[0]
    assert pair["p_holm"] < 0.05
    assert pair["statistically_significant"]
    assert pair["improvement_pct"] > 0.0
    assert pair["rank_biserial"] < 0.0
    assert pair["wins"] == 12


def test_equal_models_skip_posthoc() -> None:
    results = _results()
    results[["MASE", "RMSSE"]] = 1.0

    artifacts = analyze_imputation_results(results, n_permutations=99)

    assert (artifacts["omnibus"]["p_value"] == 1.0).all()
    assert not artifacts["omnibus"]["significant"].any()
    assert artifacts["pairwise"].empty


def test_posthoc_gate_is_independent_for_each_metric_and_gap() -> None:
    results = _results()
    results.loc[results["Gap_Size"] == 5, "MASE"] = 1.0

    artifacts = analyze_imputation_results(results, n_permutations=99)

    pairs = artifacts["pairwise"]
    assert not pairs.query("metric == 'MASE' and gap_size == 1").empty
    assert pairs.query("metric == 'MASE' and gap_size == 5").empty
    assert not pairs.query("metric == 'RMSSE' and gap_size == 5").empty


def test_gap_plan_id_validates_exact_pairing_when_available() -> None:
    results = _results()
    results["Gap_Plan_ID"] = (
        results["Serie"].astype(str)
        + "-"
        + results["Gap_Size"].astype(str)
        + "-"
        + results["Seed"].astype(str)
    )
    results.loc[
        (results["Serie"] == "S00")
        & (results["Gap_Size"] == 1)
        & (results["Modelo"] == "B")
        & (results["MonteCarlo_Run"] == 2),
        "Gap_Plan_ID",
    ] = "different-plan"

    scores, excluded = build_station_scores(results, metric="MASE", gap_size=1)

    assert "S00" not in set(scores["series"])
    assert excluded.query("series == 'S00'").iloc[0]["reason"] == (
        "inconsistent_paired_protocol"
    )
    assert scores["gap_plan_verified"].all()


def test_two_models_use_paired_wilcoxon_as_omnibus() -> None:
    artifacts = analyze_imputation_results(
        _results().query("Modelo in ['A', 'B']"), n_permutations=99
    )

    assert set(artifacts["omnibus"]["test"]) == {"wilcoxon_signed_rank"}


def test_writer_persists_fixed_schemas_and_preserves_na_station(tmp_path) -> None:
    results = _results()
    results.loc[results["Serie"] == "S00", "Serie"] = "NA"
    results.to_csv(tmp_path / "results_mc.csv", index=False)

    paths = write_statistical_analysis(tmp_path, n_permutations=99, seed=5)

    assert set(paths) == {"station_scores", "omnibus", "pairwise", "exclusions"}
    assert all(path.exists() for path in paths.values())
    scores = pd.read_csv(paths["station_scores"], keep_default_na=False)
    assert "NA" in set(scores["series"])
    assert list(pd.read_csv(paths["exclusions"]).columns) == [
        "metric",
        "gap_size",
        "series",
        "reason",
        "analysis_seed",
    ]


def test_analysis_arguments_are_validated() -> None:
    with pytest.raises(ValueError, match="alpha"):
        analyze_imputation_results(_results(), alpha=1.0, n_permutations=9)
    with pytest.raises(ValueError, match="entero positivo"):
        analyze_imputation_results(_results(), n_permutations=True)

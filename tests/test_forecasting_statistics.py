from __future__ import annotations

import json
from types import SimpleNamespace

import airquality.forecasting.statistics as statistics
import numpy as np
import pandas as pd
import pytest

from airquality.forecasting.pipeline import RESULT_COLUMNS
from airquality.forecasting.statistics import (
    _bootstrap_intervals,
    _friedman_statistic,
    _pairwise,
    analyze_forecasting_results,
    build_branch_station_scores,
    build_model_station_scores,
    write_statistical_analysis,
)
from scipy.stats import friedmanchisquare, rankdata


ARMS = ("raw", "raw+frozen", "clean-a", "clean-b")
MODELS = ("M1", "M2", "M3")


def _results(n_stations: int = 12) -> pd.DataFrame:
    arm_factor = {"raw": 1.0, "raw+frozen": 1.1, "clean-a": 0.75, "clean-b": 1.05}
    model_factor = {"M1": 0.8, "M2": 1.0, "M3": 1.25, "Foundation": 0.9}
    rows = []
    for station_index in range(n_stations):
        station = f"S{station_index:02d}"
        station_factor = 1.0 + station_index / 50.0
        for model in (*MODELS, "Foundation"):
            arms = ("raw", "raw+frozen") if model == "Foundation" else ARMS
            for arm in arms:
                mase = station_factor * model_factor[model] * arm_factor[arm]
                rows.append(
                    {
                        "series": station,
                        "arm": arm,
                        "model": model,
                        "model_mode": "foundation" if model == "Foundation" else "trained",
                        "mase": mase,
                        "rmsse": 1.2 * mase,
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


def _pair(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    row = frame[
        frame.apply(
            lambda item: {item["left"], item["right"]} == {left, right}, axis=1
        )
    ]
    assert len(row) == 1
    return row.iloc[0]


def _write_manifest(path, results: pd.DataFrame) -> None:
    models = results[["model", "model_mode"]].drop_duplicates()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            "horizon": 12,
            "forecast_stride": 6,
            "validation_len": 48,
            "validation_stride": 6,
        },
        "models": [
            {
                "name": row.model,
                "mode": row.model_mode,
                "uses_training_arms": row.model_mode != "foundation",
            }
            for row in models.itertuples(index=False)
        ],
        "arms": [{"name": arm} for arm in results["arm"].drop_duplicates()],
        "selected_stations": results["series"].drop_duplicates().tolist(),
        "expected_result_rows": len(results),
        "result_rows": len(results),
        "failures": [],
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_station_scores_use_complete_balanced_panels() -> None:
    results = _results()

    model_scores, model_exclusions = build_model_station_scores(
        results, metric="mase", scope="all_branches"
    )
    common_scores, common_exclusions = build_model_station_scores(
        results, metric="mase", scope="common_source_arms"
    )
    branch_scores, branch_exclusions = build_branch_station_scores(results, metric="mase")

    assert model_exclusions.empty and common_exclusions.empty and branch_exclusions.empty
    assert set(model_scores["treatment"]) == set(MODELS)
    assert set(common_scores["treatment"]) == {*MODELS, "Foundation"}
    assert set(branch_scores["treatment"]) == {"raw", "clean-a", "clean-b"}
    assert "raw+frozen" not in set(branch_scores["treatment"])

    m1 = model_scores.query("series == 'S00' and treatment == 'M1'").iloc[0]
    expected = np.mean([0.8, 0.88, 0.6, 0.84])
    assert m1["score"] == pytest.approx(expected)
    assert m1["mean_error"] == pytest.approx(expected)


def test_invalid_cell_excludes_the_whole_station() -> None:
    results = _results()
    results.loc[
        (results["series"] == "S00")
        & (results["model"] == "M2")
        & (results["arm"] == "clean-a"),
        "mase",
    ] = np.nan

    scores, exclusions = build_branch_station_scores(results, metric="mase")

    assert "S00" not in set(scores["series"])
    exclusion = exclusions.query("series == 'S00'").iloc[0]
    assert exclusion["reason"] == "incomplete_or_nonfinite_panel"


def test_zero_error_is_valid_and_not_replaced_with_epsilon() -> None:
    results = _results()
    results.loc[
        (results["series"] == "S00")
        & (results["model"] == "M2")
        & (results["arm"] == "clean-a"),
        "mase",
    ] = 0.0

    scores, exclusions = build_branch_station_scores(results, metric="mase")

    assert exclusions.empty
    assert "S00" in set(scores["series"])
    clean = scores.query("series == 'S00' and treatment == 'clean-a'").iloc[0]
    assert clean["score"] == pytest.approx((0.6 + 0.0 + 0.9375) / 3.0)


def test_results_must_be_unique_and_share_the_temporal_protocol() -> None:
    results = _results()
    duplicate = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicad"):
        analyze_forecasting_results(duplicate, n_permutations=99, n_bootstrap=99)

    mismatched = results.copy()
    mismatched.loc[0, "test_target_start"] = "2026-02-01 00:00:00"
    with pytest.raises(ValueError, match="temporal"):
        analyze_forecasting_results(mismatched, n_permutations=99, n_bootstrap=99)

    mismatched = results.copy()
    station = mismatched["series"] == "S00"
    mismatched.loc[station, "test_target_end"] = "2026-01-08 05:00:00"
    mismatched.loc[station, "test_target_hours"] = 102
    mismatched.loc[station, ["n_forecasts", "n_expected_forecasts"]] = 16
    mismatched.loc[station, "n_test_predictions"] = 192
    mismatched.loc[station, "n_unique_targets"] = 102
    with pytest.raises(ValueError, match="protocolo temporal"):
        analyze_forecasting_results(mismatched, n_permutations=99, n_bootstrap=99)

    invalid_context = results.copy()
    invalid_context.loc[station, "test_context_start"] = "fecha-invalida"
    with pytest.raises(ValueError, match="contexto temporal"):
        analyze_forecasting_results(invalid_context, n_permutations=99, n_bootstrap=99)


def test_results_must_match_the_pipeline_model_arm_design() -> None:
    assert set(_results().columns).issubset(RESULT_COLUMNS)

    invalid_mode = _results()
    invalid_mode.loc[invalid_mode["model"] == "M3", "model_mode"] = "fundation"
    with pytest.raises(ValueError, match="model_mode"):
        analyze_forecasting_results(invalid_mode, n_permutations=99, n_bootstrap=99)

    missing_source = _results().query("arm != 'raw+frozen'")
    with pytest.raises(ValueError, match="raw"):
        analyze_forecasting_results(missing_source, n_permutations=99, n_bootstrap=99)

    foundation_only_sources = _results().query(
        "model == 'Foundation' or arm not in ['raw', 'raw+frozen']"
    )
    with pytest.raises(ValueError, match="no-foundation"):
        analyze_forecasting_results(
            foundation_only_sources, n_permutations=99, n_bootstrap=99
        )

    inconsistent_arms = _results().query(
        "not (model == 'M3' and arm == 'clean-b')"
    )
    with pytest.raises(ValueError, match="brazos"):
        analyze_forecasting_results(inconsistent_arms, n_permutations=99, n_bootstrap=99)

    failed = _results()
    failed.loc[0, ["status", "failure_reason"]] = ["failed", "forecast_error"]
    with pytest.raises(ValueError, match="reintentarse"):
        analyze_forecasting_results(failed, n_permutations=99, n_bootstrap=99)


def test_inconsistent_backtest_counts_exclude_the_station() -> None:
    results = _results()
    results.loc[
        (results["series"] == "S00")
        & (results["model"] == "M2")
        & (results["arm"] == "clean-a"),
        "n_test_predictions",
    ] = 1

    scores, exclusions = build_branch_station_scores(results, metric="mase")

    assert "S00" not in set(scores["series"])
    assert exclusions.query("series == 'S00'").iloc[0]["reason"] == (
        "incomplete_or_nonfinite_panel"
    )

    fractional = _results()
    fractional["n_forecasts"] = fractional["n_forecasts"].astype(float)
    fractional.loc[fractional["series"] == "S00", "n_forecasts"] = 14.5
    scores, exclusions = build_branch_station_scores(fractional, metric="mase")
    assert "S00" not in set(scores["series"])
    assert "S00" in set(exclusions["series"])

    missing_scope = _results().query(
        "not (series == 'S00' and model != 'Foundation')"
    )
    scores, exclusions = build_branch_station_scores(missing_scope, metric="mase")
    assert "S00" not in set(scores["series"])
    assert "S00" in set(exclusions["series"])


def test_friedman_statistic_matches_scipy_with_ties() -> None:
    values = np.array(
        [[1.0, 1.0, 3.0], [2.0, 1.0, 3.0], [1.0, 3.0, 2.0], [2.0, 2.0, 1.0]]
    )
    ranks = np.apply_along_axis(rankdata, 1, values, method="average")

    expected = friedmanchisquare(*values.T).statistic

    assert _friedman_statistic(ranks) == pytest.approx(expected)


def test_rank_biserial_uses_the_same_pratt_ranks_as_wilcoxon() -> None:
    differences = np.array([0.0] * 10 + [-1.0, -2.0, -3.0, 4.0, 5.0])
    wide = pd.DataFrame({"left": differences, "right": np.zeros(len(differences))})

    pair = _pairwise(
        wide,
        [("left", "right")],
        analysis="model",
        scope="test",
        metric="mase",
        family="all_pairs",
        confirmatory=True,
        alpha=0.05,
        omnibus_significant=True,
        n_bootstrap=99,
        rng=np.random.default_rng(1),
    ).iloc[0]

    assert pair["rank_biserial"] == pytest.approx(-7 / 65)


def test_partial_zero_denominator_keeps_a_ratio_interval() -> None:
    low, high, _, _ = _bootstrap_intervals(
        np.ones(4),
        np.array([0.0, 2.0, 2.0, 2.0]),
        alpha=0.05,
        n_bootstrap=999,
        rng=np.random.default_rng(2),
    )

    assert np.isfinite(low)
    assert np.isfinite(high)

    low, high, _, _ = _bootstrap_intervals(
        np.array([0.0, 1.0]),
        np.array([0.0, 2.0]),
        alpha=0.05,
        n_bootstrap=99,
        rng=np.random.default_rng(2),
    )
    assert np.isnan(low)
    assert np.isnan(high)


def test_joint_p_values_are_formed_before_holm(monkeypatch) -> None:
    wilcoxon_results = iter(((1.0, 0.03), (1.0, 0.015)))
    sign_results = iter((0.01, 0.04))
    monkeypatch.setattr(statistics, "_wilcoxon", lambda _: next(wilcoxon_results))
    monkeypatch.setattr(
        statistics,
        "binomtest",
        lambda *_args, **_kwargs: SimpleNamespace(pvalue=next(sign_results)),
    )
    wide = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 3.0]}
    )

    pairs = _pairwise(
        wide,
        [("A", "B"), ("A", "C")],
        analysis="model",
        scope="test",
        metric="mase",
        family="all_pairs",
        confirmatory=True,
        alpha=0.05,
        omnibus_significant=True,
        n_bootstrap=9,
        rng=np.random.default_rng(3),
    )

    assert pairs["p_joint"].tolist() == pytest.approx([0.06, 0.06])


def test_analysis_finds_known_model_and_branch_improvements() -> None:
    artifacts = analyze_forecasting_results(
        _results(), n_permutations=1999, n_bootstrap=999, seed=7
    )

    assert set(artifacts["model_omnibus"]["metric"]) == {"mase", "rmsse"}
    assert artifacts["model_omnibus"].query(
        "scope == 'all_branches' and metric == 'mase'"
    ).iloc[0]["p_value"] < 0.05
    model_pair = _pair(
        artifacts["model_pairwise"].query(
            "scope == 'all_branches' and metric == 'mase'"
        ),
        "M1",
        "M3",
    )
    assert model_pair["p_holm"] < 0.05
    assert model_pair["p_joint"] < 0.05
    assert model_pair["typical_station_outcome"] == "left_lower_typical_error"
    assert model_pair["error_ratio"] < 1.0
    assert model_pair["wins"] == 12

    raw_pair = _pair(
        artifacts["branch_pairwise"].query(
            "family == 'vs_raw' and metric == 'mase'"
        ),
        "clean-a",
        "raw",
    )
    assert raw_pair["p_holm"] < 0.05
    assert raw_pair["typical_station_outcome"] == "left_lower_typical_error"
    assert raw_pair["improvement_pct"] > 0.0


def test_wilcoxon_outcome_follows_rank_location_not_the_mean() -> None:
    results = _results().query("model in ['M1', 'M2']").copy()
    for station in results["series"].unique()[:-1]:
        results.loc[(results["series"] == station) & (results["model"] == "M1"), ["mase", "rmsse"]] = 1.0
        results.loc[(results["series"] == station) & (results["model"] == "M2"), ["mase", "rmsse"]] = 2.0
    last = results["series"].unique()[-1]
    results.loc[(results["series"] == last) & (results["model"] == "M1"), ["mase", "rmsse"]] = 25.0
    results.loc[(results["series"] == last) & (results["model"] == "M2"), ["mase", "rmsse"]] = 1.0

    artifacts = analyze_forecasting_results(
        results, n_permutations=99, n_bootstrap=999, seed=11
    )

    pair = _pair(
        artifacts["model_pairwise"].query(
            "scope == 'all_branches' and metric == 'mase'"
        ),
        "M1",
        "M2",
    )
    assert pair["p_holm"] < 0.05
    assert pair["mean_difference"] > 0.0
    assert pair["median_difference"] < 0.0
    assert pair["rank_biserial"] < 0.0
    assert pair["typical_station_outcome"] == "left_lower_typical_error"


def test_perfect_right_treatment_can_be_declared_better() -> None:
    results = _results().query("model in ['M1', 'M2']").copy()
    results.loc[results["model"] == "M1", ["mase", "rmsse"]] = 1.0
    results.loc[results["model"] == "M2", ["mase", "rmsse"]] = 0.0

    artifacts = analyze_forecasting_results(
        results, n_permutations=99, n_bootstrap=999, seed=17
    )

    pair = _pair(
        artifacts["model_pairwise"].query(
            "scope == 'all_branches' and metric == 'mase'"
        ),
        "M1",
        "M2",
    )
    assert pair["error_ratio"] == float("inf")
    assert pair["ci_ratio_low"] == float("inf")
    assert pair["ci_difference_low"] > 0.0
    assert pair["typical_station_outcome"] == "right_lower_typical_error"


def test_no_difference_skips_posthoc_comparisons() -> None:
    results = _results()
    results[["mase", "rmsse"]] = 1.0

    artifacts = analyze_forecasting_results(
        results, n_permutations=99, n_bootstrap=99, seed=3
    )

    assert (artifacts["model_omnibus"]["p_value"] == 1.0).all()
    assert (artifacts["branch_omnibus"]["p_value"] == 1.0).all()
    assert artifacts["model_pairwise"].empty
    assert artifacts["branch_pairwise"].empty


def test_write_statistical_analysis_persists_the_seven_tables(tmp_path) -> None:
    results = _results()
    results.loc[results["series"] == "S00", "series"] = "NA"
    results.to_csv(tmp_path / "results.csv", index=False)
    _write_manifest(tmp_path, results)

    paths = write_statistical_analysis(
        tmp_path, n_permutations=99, n_bootstrap=99, seed=5
    )

    assert set(paths) == {
        "model_station_scores",
        "model_omnibus",
        "model_pairwise",
        "branch_station_scores",
        "branch_omnibus",
        "branch_pairwise",
        "statistical_exclusions",
    }
    assert all(path.exists() for path in paths.values())
    assert all(isinstance(pd.read_csv(path), pd.DataFrame) for path in paths.values())
    scores = pd.read_csv(paths["model_station_scores"], keep_default_na=False)
    assert "NA" in set(scores["series"])


def test_foundation_only_results_keep_empty_artifact_schemas(tmp_path) -> None:
    results = _results().query("model == 'Foundation'")
    results.to_csv(tmp_path / "results.csv", index=False)
    _write_manifest(tmp_path, results)

    paths = write_statistical_analysis(
        tmp_path, n_permutations=99, n_bootstrap=99, seed=5
    )

    assert pd.read_csv(paths["branch_station_scores"]).empty
    assert list(pd.read_csv(paths["statistical_exclusions"]).columns) == [
        "analysis",
        "scope",
        "metric",
        "series",
        "reason",
        "seed",
    ]


def test_statistical_writer_requires_a_complete_matching_manifest(tmp_path) -> None:
    results = _results()
    results.to_csv(tmp_path / "results.csv", index=False)
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        write_statistical_analysis(tmp_path, n_permutations=9, n_bootstrap=9)

    _write_manifest(tmp_path, results)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "incomplete"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incompleto"):
        write_statistical_analysis(tmp_path, n_permutations=9, n_bootstrap=9)

    manifest["status"] = "complete"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    results.iloc[:-1].to_csv(tmp_path / "results.csv", index=False)
    with pytest.raises(ValueError, match="filas esperadas"):
        write_statistical_analysis(tmp_path, n_permutations=9, n_bootstrap=9)


def test_one_station_reports_insufficient_evidence() -> None:
    artifacts = analyze_forecasting_results(
        _results(n_stations=1), n_permutations=99, n_bootstrap=99
    )

    assert set(artifacts["model_omnibus"]["test"]) == {"insufficient_stations"}
    assert set(artifacts["branch_omnibus"]["test"]) == {"insufficient_stations"}
    assert artifacts["model_pairwise"].empty
    assert artifacts["branch_pairwise"].empty


def test_zero_valid_stations_are_reported_and_repetitions_must_be_integers() -> None:
    results = _results()
    results[["mase", "rmsse"]] = np.nan

    artifacts = analyze_forecasting_results(
        results, n_permutations=99, n_bootstrap=99
    )

    assert set(artifacts["model_omnibus"]["test"]) == {"insufficient_stations"}
    assert set(artifacts["branch_omnibus"]["test"]) == {"insufficient_stations"}
    assert (artifacts["model_omnibus"]["n_stations"] == 0).all()
    with pytest.raises(ValueError, match="enteros"):
        analyze_forecasting_results(results, n_permutations=1.5, n_bootstrap=99)


def test_empty_pipeline_run_reports_insufficient_stations() -> None:
    artifacts = analyze_forecasting_results(
        _results().iloc[:0], n_permutations=99, n_bootstrap=99
    )

    assert len(artifacts["model_omnibus"]) == 4
    assert len(artifacts["branch_omnibus"]) == 2
    assert set(artifacts["model_omnibus"]["test"]) == {"insufficient_stations"}
    assert artifacts["model_station_scores"].empty

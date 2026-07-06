"""Unit + smoke tests for the two-mode air-quality anomaly-detection benchmark."""

from __future__ import annotations

import json

import numpy as np
import pytest

from airquality.anomaly import benchmark as benchmark_module
from airquality.anomaly.anomalies import ANOMALY_TYPES, apply_anomaly_segment, inject_synthetic_anomalies
from airquality.anomaly.benchmark import (
    INJECTION_VARIANT,
    SYNTHETIC_METRIC_KEYS,
    UNLABELED_METRIC_KEYS,
    AnomalyBenchmarkConfig,
    AnomalyCase,
    _filter_model_kwargs,
    _summarize,
    macro_detection_rate,
    recompute_ensemble,
    run_benchmark,
    split_by_detection_rate,
)
from airquality.anomaly.ensemble import consensus, rank_top_k
from airquality.anomaly.metrics import (
    MAD_SCALE,
    compute_metrics,
    detect_mask,
    detection_rate,
    mad_threshold,
    normalize_scores,
)
from airquality.anomaly.plot_benchmark_results import save_benchmark_plots
from airquality.anomaly.registry import MODEL_REGISTRY, resolve_model_class, resolve_model_names


def _base_series(length: int = 800) -> np.ndarray:
    t = np.linspace(0, 8 * np.pi, length)
    return (10.0 + 3.0 * np.sin(t)).astype(np.float32)


def _spiky_series(length: int = 800, spike_positions: tuple[int, ...] = (120, 400, 650)) -> np.ndarray:
    rng = np.random.default_rng(7)
    values = _base_series(length) + rng.normal(0.0, 0.4, length).astype(np.float32)
    for position in spike_positions:
        values[position] += 25.0
    return values


# --- synthetic anomaly injection (no STL base) ------------------------------


@pytest.mark.parametrize("variant", [*ANOMALY_TYPES, INJECTION_VARIANT])
def test_inject_produces_labels(variant: str):
    values = _base_series()
    injected, labels = inject_synthetic_anomalies(values, variant, seed=7)
    assert injected.shape == values.shape == labels.shape
    assert labels.sum() >= 1
    assert set(np.unique(labels)).issubset({0, 1})


def test_injection_variant_is_raw_combined():
    assert INJECTION_VARIANT == "combined"


def test_inject_preserves_series_outside_labels():
    # No STL base: outside the injected segments the REAL series is untouched.
    values = _base_series(400)
    injected, labels = inject_synthetic_anomalies(values, INJECTION_VARIANT, seed=3)
    untouched = labels == 0
    assert np.array_equal(injected[untouched], values[untouched])
    assert not np.array_equal(injected, values)


def test_inject_rejects_stl_variant():
    with pytest.raises(ValueError):
        inject_synthetic_anomalies(_base_series(200), "STL-combined", seed=1)


def test_inject_short_series_returns_zero_labels():
    values = np.arange(5, dtype=np.float32)
    injected, labels = inject_synthetic_anomalies(values, INJECTION_VARIANT, seed=1)
    assert labels.sum() == 0
    assert injected.shape == values.shape


def test_apply_anomaly_segment_unknown_type_raises():
    arr = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError):
        apply_anomaly_segment(arr, arr, 0, 4, "bogus", np.random.default_rng(0), 1.0)


def test_inject_is_deterministic_for_seed():
    values = _base_series(400)
    a_values, a_labels = inject_synthetic_anomalies(values, INJECTION_VARIANT, seed=5)
    b_values, b_labels = inject_synthetic_anomalies(values, INJECTION_VARIANT, seed=5)
    assert np.array_equal(a_values, b_values)
    assert np.array_equal(a_labels, b_labels)


# --- supervised metrics (synthetic mode) ------------------------------------


def test_compute_metrics_perfect_score():
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    scores = labels.astype(np.float64)
    metrics = compute_metrics(labels, scores, window_size=2)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["vus_pr"] > 0.5


def test_compute_metrics_no_anomalies_returns_zero_dict():
    metrics = compute_metrics(np.zeros(8, dtype=np.int64), np.linspace(0, 1, 8), window_size=2)
    assert set(metrics) == set(SYNTHETIC_METRIC_KEYS)
    assert all(value == 0.0 for value in metrics.values())


def test_compute_metrics_empty_returns_zero_dict():
    metrics = compute_metrics(np.array([], dtype=np.int64), np.array([]), window_size=2)
    assert all(value == 0.0 for value in metrics.values())


def test_vus_sliding_window_is_median_segment_length():
    from airquality.anomaly.metrics import vus_sliding_window

    labels = np.zeros(60, dtype=np.int64)
    labels[2:4] = 1  # length 2
    labels[10:15] = 1  # length 5
    labels[30:40] = 1  # length 10
    assert vus_sliding_window(labels) == 5


def test_vus_sliding_window_handles_edges_and_spikes():
    from airquality.anomaly.metrics import vus_sliding_window

    labels = np.zeros(20, dtype=np.int64)
    labels[0] = 1  # spike at the start
    labels[19] = 1  # spike at the end
    assert vus_sliding_window(labels) == 1

    all_anomalous = np.ones(7, dtype=np.int64)
    assert vus_sliding_window(all_anomalous) == 7


def test_vus_sliding_window_defaults_to_one_without_anomalies():
    from airquality.anomaly.metrics import vus_sliding_window

    assert vus_sliding_window(np.zeros(10, dtype=np.int64)) == 1
    assert vus_sliding_window(np.array([], dtype=np.int64)) == 1


# --- score binarization (median + k*MAD) -----------------------------------


def test_mad_threshold_matches_manual_computation():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    median = 2.0
    mad = 1.0  # |scores - 2| -> [2, 1, 0, 1, 98], median = 1
    assert mad_threshold(scores, k=3.5) == pytest.approx(median + 3.5 * MAD_SCALE * mad)


def test_mad_threshold_no_finite_scores_returns_inf():
    assert mad_threshold(np.array([np.nan, np.inf])) == float("inf")


def test_detect_mask_flags_only_extreme_scores():
    rng = np.random.default_rng(0)
    scores = rng.normal(0.0, 1.0, 500)
    scores[[10, 200]] = 50.0
    mask = detect_mask(scores, k=3.5)
    assert mask[10] and mask[200]
    assert mask.sum() < 25  # the bulk of the gaussian stays below the threshold


def test_detect_mask_constant_scores_flags_nothing():
    assert not detect_mask(np.full(50, 3.3)).any()


def test_detect_mask_mad_zero_majority_value():
    # Hampel-style scores: mostly zero, outliers positive -> MAD = 0 and the
    # threshold degenerates to the median, flagging exactly the non-zero scores.
    scores = np.zeros(100)
    scores[[5, 50]] = 1.0
    mask = detect_mask(scores)
    assert mask.sum() == 2 and mask[5] and mask[50]


def test_detect_mask_never_flags_nan():
    scores = np.array([0.0, 0.1, np.nan, 99.0])
    mask = detect_mask(scores)
    assert not mask[2] and mask[3]


def test_detection_rate_simple_and_empty():
    assert detection_rate(np.array([True, False, False, False])) == pytest.approx(0.25)
    assert detection_rate(np.array([], dtype=bool)) == 0.0


def test_normalize_scores_constant_array_returns_zeros():
    out = normalize_scores(np.array([5.0, 5.0, 5.0]))
    assert np.all(out == 0.0)


def test_normalize_scores_scales_to_unit_range():
    out = normalize_scores(np.array([0.0, 5.0, 10.0]))
    assert out.tolist() == pytest.approx([0.0, 0.5, 1.0])


# --- ensemble ------------------------------------------------------------


def test_rank_top_k_picks_highest():
    metrics = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.7}
    assert rank_top_k(metrics, k=3) == ["b", "d", "c"]


def test_rank_top_k_breaks_ties_by_name():
    assert rank_top_k({"b": 0.5, "a": 0.5, "c": 0.1}, k=2) == ["a", "b"]


def test_rank_top_k_k_larger_than_input_returns_all():
    assert rank_top_k({"a": 0.1, "b": 0.2}, k=5) == ["b", "a"]


@pytest.mark.parametrize("method", ["AVG", "MAX", "AOM"])
def test_consensus_shape_and_range(method: str):
    scores = [np.array([0.0, 5.0, 1.0, 9.0]), np.array([2.0, 2.0, 8.0, 1.0])]
    fused = consensus(scores, method=method, seed=1)
    assert fused.shape == (4,)
    assert fused.min() >= 0.0 and fused.max() <= 1.0 + 1e-6


def test_consensus_weighted_avg_biases_toward_higher_weight():
    score_a = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    score_b = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    fused = consensus([score_a, score_b], method="AVG", weights=[0.9, 0.1])
    assert fused[:2].mean() < fused[2:].mean()


def test_consensus_rejects_unknown_method():
    with pytest.raises(ValueError):
        consensus([np.array([1.0, 2.0])], method="median")


def test_consensus_empty_list_raises():
    with pytest.raises(ValueError):
        consensus([])


def test_consensus_accepts_lowercase_method():
    fused = consensus([np.array([0.0, 1.0, 2.0])], method="avg")
    assert fused.shape == (3,)


# --- detection-rate filter -------------------------------------------------


def _fake_results(rates_by_model: dict[str, list[float]]) -> dict[str, dict[str, object]]:
    return {
        name: {"per_case": [{"metrics": {"detection_rate": rate}} for rate in rates]}
        for name, rates in rates_by_model.items()
    }


def test_split_by_detection_rate_discards_over_budget():
    results = _fake_results({"ok": [0.01, 0.03], "noisy": [0.20, 0.30], "silent": [0.0, 0.0]})
    kept, discarded = split_by_detection_rate(results, max_detection_rate=0.07)
    assert kept == ["ok", "silent"]
    assert discarded == ["noisy"]


def test_split_by_detection_rate_budget_is_inclusive():
    results = _fake_results({"at_budget": [0.07, 0.07]})
    kept, discarded = split_by_detection_rate(results, max_detection_rate=0.07)
    assert kept == ["at_budget"] and discarded == []


def test_macro_detection_rate_averages_cases():
    results = _fake_results({"m": [0.1, 0.3]})
    assert macro_detection_rate(results["m"]) == pytest.approx(0.2)


# --- registry ------------------------------------------------------------


def test_resolve_model_names_none_returns_full_registry():
    assert set(resolve_model_names(None)) == set(MODEL_REGISTRY)


def test_resolve_model_names_all_keyword_case_insensitive():
    assert set(resolve_model_names(["ALL"])) == set(MODEL_REGISTRY)


def test_resolve_model_names_unknown_raises():
    with pytest.raises(ValueError):
        resolve_model_names(["NoSuchModel"])


def test_resolve_model_class_returns_type():
    assert resolve_model_class("IQR").__name__ == "IQRDetector"


# --- benchmark glue/helpers ----------------------------------------------


def test_filter_model_kwargs_drops_unaccepted():
    class NoDevice:
        def __init__(self, seed=13):
            ...

    class WithDevice:
        def __init__(self, seed=13, device=None):
            ...

    class WithKwargs:
        def __init__(self, seed=13, **kwargs):
            ...

    assert _filter_model_kwargs(NoDevice, {"device": "cpu"}) == {}
    assert _filter_model_kwargs(WithDevice, {"device": "cpu"}) == {"device": "cpu"}
    assert _filter_model_kwargs(WithKwargs, {"device": "cpu"}) == {"device": "cpu"}


def test_summarize_aggregates_and_drops_scores():
    series_results = [
        {
            "metrics": {key: 1.0 for key in UNLABELED_METRIC_KEYS},
            "timing": {"fit_seconds": 1.0, "inference_seconds": 2.0},
            "scores": np.zeros(3),
        },
        {
            "metrics": {key: 0.0 for key in UNLABELED_METRIC_KEYS},
            "timing": {"fit_seconds": 3.0, "inference_seconds": 4.0},
            "scores": np.zeros(3),
        },
    ]

    out = _summarize(series_results)

    assert out["macro_metrics"]["detection_rate"] == 0.5
    assert out["timing"]["mean_fit_seconds"] == 2.0
    assert all("scores" not in entry for entry in out["series_results"])


def test_summarize_derives_metric_keys_from_entries():
    series_results = [
        {
            "metrics": {key: 0.5 for key in SYNTHETIC_METRIC_KEYS},
            "timing": {"fit_seconds": 1.0, "inference_seconds": 1.0},
        }
    ]
    out = _summarize(series_results)
    assert set(out["macro_metrics"]) == set(SYNTHETIC_METRIC_KEYS)


def _hourly_5m_frame(n_hours: int, start: str = "2024-01-01"):
    """Build a varied 5-minute frame that survives preprocessing to ~n_hours points."""
    import pandas as pd

    n = n_hours * 12
    idx = pd.date_range(start, periods=n, freq="5min")
    t = np.arange(n)
    values = 20.0 + 5.0 * np.sin(t / 3.0) + (t % 7)  # all >> threshold, never frozen
    return pd.DataFrame({"NO2": values}, index=idx)


def test_build_cases_skips_short_series(monkeypatch):
    stations = [("Good", _hourly_5m_frame(10)), ("Tiny", _hourly_5m_frame(2, start="2024-02-01"))]
    monkeypatch.setattr(benchmark_module, "load_raw_5m", lambda pollutant, base_dir: stations)

    config = AnomalyBenchmarkConfig(min_series_points=6)
    cases = benchmark_module.build_cases(config)

    assert {case.name for case in cases} == {"Good"}  # Tiny is skipped (too few points)
    # Unlabeled cases carry the REAL series untouched (no injection, no labels).
    assert all(case.labels is None and case.values_select is None for case in cases)


def test_build_cases_synthetic_injects_two_independent_seeds(monkeypatch):
    stations = [("Good", _hourly_5m_frame(30))]
    monkeypatch.setattr(benchmark_module, "load_raw_5m", lambda pollutant, base_dir: stations)

    config = AnomalyBenchmarkConfig(mode="synthetic", min_series_points=6)
    cases = benchmark_module.build_cases(config)

    (case,) = cases
    assert case.labels.sum() >= 1 and case.labels_select.sum() >= 1
    # Selection and evaluation injections are independent (different seeds).
    assert not np.array_equal(case.values, case.values_select)


def _real_cases() -> list[AnomalyCase]:
    return [
        AnomalyCase(name="StationA", values=_spiky_series(700)),
        AnomalyCase(name="StationB", values=_spiky_series(700, spike_positions=(80, 300))),
    ]


def test_run_benchmark_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: _real_cases())
    config = AnomalyBenchmarkConfig(
        models=["ModifiedZScore", "IQR", "IsolationForest"],
        device="cpu",
        output_dir=str(tmp_path),
    )
    summary = run_benchmark(config)

    assert set(summary["kept_models"]) | set(summary["discarded_models"]) == {
        "ModifiedZScore", "IQR", "IsolationForest",
    }
    # Spikes are 3/700 points; the baseline detectors stay within the 7% budget.
    assert summary["kept_models"], "expected at least one surviving detector"
    assert summary["model_names"][-1] == "Ensemble"
    for name in summary["model_names"]:
        rate = summary["models"][name]["macro_metrics"]["detection_rate"]
        assert 0.0 <= rate <= 1.0
    for name in summary["discarded_models"]:
        assert summary["models"][name]["discarded"]

    results_path = tmp_path / "results.json"
    assert results_path.exists()
    payload = json.loads(results_path.read_text())
    assert "Ensemble" in payload["models"]
    assert payload["models"]["IQR"]["series_results"][0]["metrics"]["detection_rate"] >= 0.0
    assert (tmp_path / "scores.npz").exists()
    # The benchmark itself does NOT render plots (that is a separate script).
    assert not (tmp_path / "detection_rate_distribution.png").exists()


def test_run_benchmark_all_discarded_builds_no_ensemble(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: _real_cases()[:1])
    config = AnomalyBenchmarkConfig(
        models=["ModifiedZScore", "IQR"],
        device="cpu",
        max_detection_rate=-1.0,  # every rate is > -1 -> everything discarded
        output_dir=str(tmp_path),
    )
    summary = run_benchmark(config)

    assert summary["kept_models"] == []
    assert set(summary["discarded_models"]) == {"ModifiedZScore", "IQR"}
    assert "Ensemble" not in summary["model_names"]
    assert "Ensemble" not in summary["models"]


def test_save_benchmark_plots_from_results(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: _real_cases())
    config = AnomalyBenchmarkConfig(models=["ModifiedZScore", "IQR"], device="cpu", output_dir=str(tmp_path))
    run_benchmark(config)

    plot_paths = save_benchmark_plots(tmp_path / "results.json")

    for key in ("metrics_plot", "scatter_plot", "training_plot"):
        assert plot_paths[key].exists()


def test_recompute_ensemble_matches_saved_run(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: _real_cases())
    config = AnomalyBenchmarkConfig(
        models=["ModifiedZScore", "IQR", "IsolationForest"],
        device="cpu",
        output_dir=str(tmp_path),
    )
    run_benchmark(config)

    out = recompute_ensemble(tmp_path, method="AVG", threshold_k=3.5)

    assert "Ensemble(method=AVG,k=3.5)" in out
    for name in ("ModifiedZScore", "IQR", "IsolationForest"):
        assert name in out
    assert all(np.isfinite(value) for value in out.values())


# --- synthetic mode end-to-end ---------------------------------------------


def _synthetic_cases() -> list[AnomalyCase]:
    cases = []
    for station in ("StationA", "StationB"):
        values = _base_series(700)
        sel_v, sel_l = inject_synthetic_anomalies(values, INJECTION_VARIANT, seed=3)
        eval_v, eval_l = inject_synthetic_anomalies(values, INJECTION_VARIANT, seed=101)
        cases.append(
            AnomalyCase(
                name=station,
                values=eval_v,
                labels=eval_l,
                values_select=sel_v,
                labels_select=sel_l,
            )
        )
    return cases


def test_run_benchmark_synthetic_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: _synthetic_cases())
    config = AnomalyBenchmarkConfig(
        mode="synthetic",
        models=["ModifiedZScore", "IQR", "IsolationForest"],
        device="cpu",
        output_dir=str(tmp_path),
    )
    summary = run_benchmark(config)

    assert summary["mode"] == "synthetic"
    assert summary["model_names"][-1] == "Ensemble"
    for name in summary["model_names"]:
        assert "vus_pr" in summary["models"][name]["macro_metrics"]

    results_path = tmp_path / "results.json"
    assert results_path.exists()
    payload = json.loads(results_path.read_text())
    assert "Ensemble" in payload["models"]
    assert payload["variants"] == [INJECTION_VARIANT]
    # Detectors carry a selection-injection VUS-PR distinct from the reported (eval) one.
    assert "vus_pr_select" in payload["models"]["IQR"]["series_results"][0]
    # Labels are persisted so the ensemble can be recomputed without retraining.
    scores_npz = np.load(tmp_path / "scores.npz")
    assert "__labels__case0" in scores_npz


def test_run_benchmark_rejects_unknown_mode():
    with pytest.raises(ValueError):
        run_benchmark(AnomalyBenchmarkConfig(mode="bogus"))


def test_save_benchmark_plots_synthetic_run(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: _synthetic_cases())
    config = AnomalyBenchmarkConfig(
        mode="synthetic", models=["ModifiedZScore", "IQR"], device="cpu", output_dir=str(tmp_path)
    )
    run_benchmark(config)

    plot_paths = save_benchmark_plots(tmp_path / "results.json")

    assert plot_paths["metrics_plot"].name == "vus_pr_distribution.png"
    for key in ("metrics_plot", "scatter_plot", "training_plot"):
        assert plot_paths[key].exists()


def test_recompute_ensemble_synthetic_run(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: _synthetic_cases())
    config = AnomalyBenchmarkConfig(
        mode="synthetic",
        models=["ModifiedZScore", "IQR", "IsolationForest"],
        device="cpu",
        output_dir=str(tmp_path),
    )
    run_benchmark(config)

    out = recompute_ensemble(tmp_path, method="AVG", top_k=3)

    assert "Ensemble(method=AVG,top_k=3)" in out
    for name in ("ModifiedZScore", "IQR", "IsolationForest"):
        assert name in out
    assert all(np.isfinite(value) for value in out.values())

"""Unit + smoke tests for the air-quality anomaly-detection benchmark."""

from __future__ import annotations

import json

import numpy as np
import pytest

from airquality.anomaly import benchmark as benchmark_module
from airquality.anomaly.anomalies import (
    STL_ANOMALY_TYPES,
    apply_stl_anomaly_segment,
    inject_synthetic_anomalies,
    synthetic_base,
)
from airquality.anomaly.benchmark import (
    INJECTION_VARIANT,
    METRIC_KEYS,
    AnomalyBenchmarkConfig,
    AnomalyCase,
    _filter_model_kwargs,
    _summarize,
    recompute_ensemble,
    run_benchmark,
)
from airquality.anomaly.ensemble import consensus, rank_top_k
from airquality.anomaly.metrics import compute_metrics, normalize_scores
from airquality.anomaly.plot_benchmark_results import save_benchmark_plots
from airquality.anomaly.registry import MODEL_REGISTRY, resolve_model_class, resolve_model_names


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _base_series(length: int = 800) -> np.ndarray:
    t = np.linspace(0, 8 * np.pi, length)
    return (10.0 + 3.0 * np.sin(t)).astype(np.float32)


# --- anomaly injection ---------------------------------------------------


@pytest.mark.parametrize("variant", [f"STL-{t}" for t in STL_ANOMALY_TYPES] + [INJECTION_VARIANT])
def test_inject_produces_labels(variant: str):
    values = _base_series()
    injected, labels = inject_synthetic_anomalies(values, variant, seed=7)
    assert injected.shape == values.shape == labels.shape
    assert labels.sum() >= 1
    assert set(np.unique(labels)).issubset({0, 1})


def test_injection_variant_is_stl_combined():
    assert INJECTION_VARIANT == "STL-combined"


def test_synthetic_base_real_stl_preserves_length_and_is_finite():
    values = _base_series(400)
    base = synthetic_base(values, seed=1)
    assert base.shape == values.shape
    assert np.isfinite(base).all()
    # The STL base is a synthetic look-alike, not the original series.
    assert not np.array_equal(base, values)


def test_synthetic_base_short_series_falls_back():
    values = np.arange(20, dtype=np.float32)  # < 2*period + 1
    base = synthetic_base(values, seed=1)
    assert base.shape == values.shape and np.isfinite(base).all()


# --- ensemble ------------------------------------------------------------


def test_rank_top_k_picks_highest():
    metrics = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.7}
    assert rank_top_k(metrics, k=3) == ["b", "d", "c"]


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


def test_rank_top_k_breaks_ties_by_name():
    assert rank_top_k({"b": 0.5, "a": 0.5, "c": 0.1}, k=2) == ["a", "b"]


def test_rank_top_k_k_larger_than_input_returns_all():
    assert rank_top_k({"a": 0.1, "b": 0.2}, k=5) == ["b", "a"]


# --- metrics -------------------------------------------------------------


def test_compute_metrics_perfect_score():
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    scores = labels.astype(np.float64)
    metrics = compute_metrics(labels, scores, window_size=2)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["vus_pr"] > 0.5


def test_normalize_scores_constant_array_returns_zeros():
    out = normalize_scores(np.array([5.0, 5.0, 5.0]))
    assert np.all(out == 0.0)


def test_normalize_scores_scales_to_unit_range():
    out = normalize_scores(np.array([0.0, 5.0, 10.0]))
    assert out.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_compute_metrics_no_anomalies_returns_zero_dict():
    metrics = compute_metrics(np.zeros(8, dtype=np.int64), np.linspace(0, 1, 8), window_size=2)
    assert set(metrics) == {"auroc", "aupr", "vus_pr", "vus_roc", "affiliation_f1"}
    assert all(value == 0.0 for value in metrics.values())


def test_compute_metrics_empty_returns_zero_dict():
    metrics = compute_metrics(np.array([], dtype=np.int64), np.array([]), window_size=2)
    assert all(value == 0.0 for value in metrics.values())


# --- anomalies edge cases ------------------------------------------------


def test_inject_short_series_returns_zero_labels():
    values = np.arange(5, dtype=np.float32)
    injected, labels = inject_synthetic_anomalies(values, "STL-combined", seed=1)
    assert labels.sum() == 0
    assert injected.shape == values.shape


def test_apply_stl_anomaly_segment_unknown_type_raises():
    arr = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError):
        apply_stl_anomaly_segment(arr, arr, 0, 4, "bogus", np.random.default_rng(0), 1.0)


def test_inject_is_deterministic_for_seed():
    values = _base_series(400)
    a_values, a_labels = inject_synthetic_anomalies(values, "STL-combined", seed=5)
    b_values, b_labels = inject_synthetic_anomalies(values, "STL-combined", seed=5)
    assert np.array_equal(a_values, b_values)
    assert np.array_equal(a_labels, b_labels)


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
            "metrics": {key: 1.0 for key in METRIC_KEYS},
            "timing": {"fit_seconds": 1.0, "inference_seconds": 2.0},
            "scores": np.zeros(3),
        },
        {
            "metrics": {key: 0.0 for key in METRIC_KEYS},
            "timing": {"fit_seconds": 3.0, "inference_seconds": 4.0},
            "scores": np.zeros(3),
        },
    ]

    out = _summarize(series_results)

    assert out["macro_metrics"]["vus_pr"] == 0.5
    assert out["timing"]["mean_fit_seconds"] == 2.0
    assert all("scores" not in entry for entry in out["series_results"])


def _hourly_5m_frame(n_hours: int, start: str = "2024-01-01"):
    """Build a varied 5-minute frame that survives preprocessing to ~n_hours points."""
    import pandas as pd

    n = n_hours * 12
    idx = pd.date_range(start, periods=n, freq="5min")
    t = np.arange(n)
    values = 20.0 + 5.0 * np.sin(t / 3.0) + (t % 7)  # all >> threshold, never frozen
    return pd.DataFrame({"NO2": values}, index=idx)


def test_build_cases_skips_short_series_and_injects_stl_combined(monkeypatch):
    stations = [("Good", _hourly_5m_frame(10)), ("Tiny", _hourly_5m_frame(2, start="2024-02-01"))]
    monkeypatch.setattr(benchmark_module, "load_raw_5m", lambda pollutant, base_dir: stations)

    config = AnomalyBenchmarkConfig(min_series_points=6)
    cases = benchmark_module.build_cases(config)

    assert {case.name for case in cases} == {"Good"}  # Tiny is skipped (too few points)
    assert {case.variant for case in cases} == {"STL-combined"}


def _synthetic_cases() -> list[AnomalyCase]:
    cases = []
    for station in ("StationA", "StationB"):
        values = _base_series(700)
        sel_v, sel_l = inject_synthetic_anomalies(values, INJECTION_VARIANT, seed=3)
        eval_v, eval_l = inject_synthetic_anomalies(values, INJECTION_VARIANT, seed=101)
        cases.append(
            AnomalyCase(
                name=station,
                variant=INJECTION_VARIANT,
                values=eval_v,
                labels=eval_l,
                values_select=sel_v,
                labels_select=sel_l,
            )
        )
    return cases


def test_run_benchmark_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: _synthetic_cases())
    config = AnomalyBenchmarkConfig(
        models=["ModifiedZScore", "IQR", "IsolationForest"],
        device="cpu",
        output_dir=str(tmp_path),
    )
    summary = run_benchmark(config)

    assert summary["model_names"][-1] == "Ensemble"
    for name in summary["model_names"]:
        assert "vus_pr" in summary["models"][name]["macro_metrics"]

    results_path = tmp_path / "results.json"
    assert results_path.exists()
    payload = json.loads(results_path.read_text())
    assert "Ensemble" in payload["models"]
    assert payload["variants"] == ["STL-combined"]
    assert payload["models"]["IQR"]["series_results"][0]["variant"] == "STL-combined"
    # Detectors carry a selection-injection VUS-PR distinct from the reported (eval) one.
    assert "vus_pr_select" in payload["models"]["IQR"]["series_results"][0]
    # The benchmark itself does NOT render plots (that is a separate script).
    assert not (tmp_path / "vus_pr_distribution.png").exists()


def test_save_benchmark_plots_from_results(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: _synthetic_cases())
    config = AnomalyBenchmarkConfig(models=["ModifiedZScore", "IQR"], device="cpu", output_dir=str(tmp_path))
    run_benchmark(config)

    plot_paths = save_benchmark_plots(tmp_path / "results.json")

    for key in ("metrics_plot", "scatter_plot", "training_plot"):
        assert plot_paths[key].exists()


def test_recompute_ensemble_matches_saved_run(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: _synthetic_cases())
    config = AnomalyBenchmarkConfig(
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

"""Unit + smoke tests for the two-mode air-quality anomaly-detection benchmark."""

from __future__ import annotations

from collections import Counter
import json

import numpy as np
import pandas as pd
import pytest

from airquality.anomaly import benchmark as benchmark_module
from airquality.anomaly import metrics as metrics_module
from airquality.anomaly.anomalies import (
    ANOMALY_TYPES,
    INJECTION_REFERENCE_WINDOW,
    _combined_group,
    _plan_synthetic_anomalies,
    apply_anomaly_segment,
    inject_synthetic_anomaly_segments,
    inject_synthetic_anomalies,
)
from airquality.anomaly.benchmark import (
    INJECTION_VARIANT,
    SYNTHETIC_METRIC_KEYS,
    UNLABELED_METRIC_KEYS,
    AnomalyBenchmarkConfig,
    AnomalyCase,
    _expand_model_names,
    _requested_model_kwargs,
    _filter_model_kwargs,
    _summarize,
    recompute_ensemble,
    run_benchmark,
)
from airquality.anomaly.ensemble import (
    rank_top_k,
    ranked_pointwise_minmax_mean,
    ranked_pointwise_vote,
)
from airquality.anomaly.metrics import (
    MAD_SCALE,
    compute_metrics,
    detect_mask,
    mad_threshold,
    normalize_scores,
)
from airquality.anomaly.registry import MODEL_REGISTRY, resolve_model_class, resolve_model_names
from airquality.visualizations import anomaly as presentation_module
from airquality.visualizations import anomaly_benchmark as plot_module
from airquality.visualizations.anomaly_benchmark import save_benchmark_plots


def _base_series(length: int = 800) -> np.ndarray:
    t = np.linspace(0, 8 * np.pi, length)
    return (10.0 + 3.0 * np.sin(t)).astype(np.float32)


def _segment_indices(lengths: tuple[int, ...]) -> tuple[pd.DatetimeIndex, ...]:
    """Build explicit timestamps for array-only AnomalyCase fixtures."""
    next_start = pd.Timestamp("2024-01-01")
    indices = []
    for length in lengths:
        index = pd.date_range(next_start, periods=length, freq="h")
        indices.append(index)
        next_start = index[-1] + pd.Timedelta(hours=2)
    return tuple(indices)


def test_anomaly_case_requires_segment_timestamps() -> None:
    with pytest.raises(TypeError):
        AnomalyCase("Station", np.zeros(10, dtype=np.float32))


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


@pytest.mark.parametrize("variant", ["combined", *ANOMALY_TYPES])
def test_injection_variant_is_normalized(variant: str):
    config = benchmark_module.AnomalyBenchmarkConfig(
        mode="synthetic", injection_variant=variant.upper()
    )

    assert config.injection_variant == variant


def test_injection_variant_rejects_unknown_profile():
    with pytest.raises(ValueError, match="Unknown injection variant"):
        benchmark_module.AnomalyBenchmarkConfig(
            mode="synthetic", injection_variant="unknown"
        )


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
        apply_anomaly_segment(arr, 0, 4, "bogus", np.random.default_rng(0), 1.0)


def test_inject_is_deterministic_for_seed():
    values = _base_series(400)
    a_values, a_labels = inject_synthetic_anomalies(values, INJECTION_VARIANT, seed=5)
    b_values, b_labels = inject_synthetic_anomalies(values, INJECTION_VARIANT, seed=5)
    assert np.array_equal(a_values, b_values)
    assert np.array_equal(a_labels, b_labels)


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (40, ({"spikes"}, {"scale"}, {"noise"})),
        (80, ({"spikes", "scale"}, {"spikes", "noise"})),
        (240, ({"spikes", "scale", "noise", "drift"},)),
    ],
)
def test_combined_injection_uses_four_w80_levels(length, expected):
    group = set(_combined_group(length, np.random.default_rng(7)))

    assert INJECTION_REFERENCE_WINDOW == 80
    assert group in expected


def test_combined_level_three_selects_one_of_two_groups():
    groups = {
        tuple(sorted(_combined_group(200, np.random.default_rng(seed))))
        for seed in range(100)
    }

    assert groups == {("drift",), ("noise", "scale", "spikes")}


def test_combined_level_three_is_balanced_50_50():
    drift = sum(
        _combined_group(200, np.random.default_rng(seed)) == ["drift"]
        for seed in range(1000)
    )

    assert 450 <= drift <= 550


def test_combined_budget_is_station_wide_not_per_segment():
    plans = _plan_synthetic_anomalies([40] * 10, "combined", seed=7)

    assert len(plans) == 4
    assert len({plan[0] for plan in plans}) == 4


def test_combined_budget_reaches_every_segment_before_second_events():
    plans = _plan_synthetic_anomalies([100] * 10, "combined", seed=7)

    assert len(plans) == 12  # The single drift quota cannot fit in these segments.
    assert len({plan[0] for plan in plans[:10]}) == 10
    assert {plan[0] for plan in plans} == set(range(10))


@pytest.mark.parametrize("seed", range(10))
def test_combined_respects_every_type_quota_across_segments(seed):
    plans = _plan_synthetic_anomalies([400] * 5, "combined", seed=seed)
    counts = {
        anomaly_type: sum(plan[3] == anomaly_type for plan in plans)
        for anomaly_type in ANOMALY_TYPES
    }

    assert counts == {"spikes": 13, "scale": 6, "noise": 6, "drift": 2}
    assert {plan[0] for plan in plans} == set(range(5))


@pytest.mark.parametrize("seed", range(10))
def test_combined_guarantees_every_type_when_station_can_fit_them(seed):
    plans = _plan_synthetic_anomalies([200] * 10, "combined", seed=seed)
    counts = Counter(plan[3] for plan in plans)

    assert counts == {"spikes": 13, "scale": 6, "noise": 6, "drift": 2}


def test_combined_spends_compatible_quotas_across_mixed_segment_levels():
    plans = _plan_synthetic_anomalies([200] + [100] * 18, "combined", seed=7)

    assert Counter(plan[3] for plan in plans) == {
        "spikes": 13,
        "scale": 6,
        "noise": 6,
        "drift": 2,
    }


def test_combined_reservations_preserve_first_round_segment_coverage():
    plans = _plan_synthetic_anomalies([160, *([8] * 100)], "combined", seed=7)

    assert len({plan[0] for plan in plans}) == len(plans)


def test_combined_does_not_force_drift_without_a_large_enough_segment():
    plans = _plan_synthetic_anomalies([100] * 20, "combined", seed=7)

    assert "drift" not in {plan[3] for plan in plans}


def test_combined_does_not_force_types_before_their_station_quota():
    plans = _plan_synthetic_anomalies([899], "combined", seed=7)

    assert "drift" not in {plan[3] for plan in plans}


def test_combined_level_four_cannot_exceed_station_budget():
    plans = _plan_synthetic_anomalies([240], "combined", seed=7)

    assert len(plans) == 1


@pytest.mark.parametrize(
    "lengths",
    ([80], [160], [240], [400], [40] * 10, [100, 200, 300]),
)
def test_combined_never_exceeds_global_event_budget(lengths):
    total = sum(lengths)
    budget = max(
        1,
        sum(total // points for points in (150, 300, 300, 900)),
    )

    assert len(_plan_synthetic_anomalies(lengths, "combined", seed=11)) <= budget


def test_combined_events_never_overlap():
    plans = _plan_synthetic_anomalies([900], "combined", seed=7)
    ordered = sorted(plans, key=lambda plan: plan[1])

    for left, right in zip(
        ordered[:-1],
        ordered[1:],
        strict=True,
    ):
        assert left[2] < right[1]


def test_combined_plans_remain_distinct_binary_events():
    lengths = [8, 1000]
    plans = _plan_synthetic_anomalies(lengths, "combined", seed=0)
    generated = inject_synthetic_anomaly_segments(
        [np.arange(length, dtype=np.float32) for length in lengths],
        "combined",
        seed=0,
    )
    event_count = sum(
        np.count_nonzero(np.diff(np.r_[0, labels, 0]) == 1)
        for _, labels in generated
    )

    assert event_count == len(plans)


def test_large_combined_segment_uses_station_wide_type_rates():
    plans = _plan_synthetic_anomalies([900], "combined", seed=7)
    counts = {
        anomaly_type: sum(plan[3] == anomaly_type for plan in plans)
        for anomaly_type in ANOMALY_TYPES
    }

    assert counts == {"spikes": 6, "scale": 3, "noise": 3, "drift": 1}


@pytest.mark.parametrize("seed", range(20))
def test_single_drift_variant_uses_a_segment_where_it_fits(seed):
    plans = _plan_synthetic_anomalies([8, 1000], "drift", seed=seed)

    assert plans
    assert {plan[0] for plan in plans} == {1}


# --- supervised metrics (synthetic mode) ------------------------------------


def test_compute_metrics_perfect_score():
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    scores = labels.astype(np.float64)
    metrics = compute_metrics(labels, scores, window_size=2)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["vus_pr"] > 0.5


def test_compute_metrics_no_anomalies_returns_undefined_metrics():
    metrics = compute_metrics(np.zeros(8, dtype=np.int64), np.linspace(0, 1, 8), window_size=2)
    assert set(metrics) == set(SYNTHETIC_METRIC_KEYS)
    assert all(np.isnan(value) for value in metrics.values())


def test_segmented_metrics_all_anomalous_keeps_affiliation_defined():
    labels = np.ones(8, dtype=np.int64)
    metrics = metrics_module.compute_segmented_metrics(
        [labels], [labels.astype(float)], window_size=2,
        prediction_masks_by_segment=[np.ones(8, dtype=bool)],
    )["metrics"]
    assert all(np.isnan(metrics[key]) for key in ("auroc", "aupr", "vus_pr", "vus_roc"))
    assert metrics["affiliation_f1"] == pytest.approx(1.0)


def test_compute_metrics_empty_returns_undefined_metrics():
    metrics = compute_metrics(np.array([], dtype=np.int64), np.array([]), window_size=2)
    assert all(np.isnan(value) for value in metrics.values())


def test_segmented_metrics_pool_opposite_single_class_segments():
    labels = [np.zeros(4, dtype=np.int64), np.ones(4, dtype=np.int64)]
    scores = [np.linspace(0.0, 0.3, 4), np.linspace(0.7, 1.0, 4)]

    result = metrics_module.compute_segmented_metrics(labels, scores, window_size=1)

    assert result["metrics"]["auroc"] == pytest.approx(1.0)
    assert result["metrics"]["aupr"] == pytest.approx(1.0)
    assert result["metrics"]["vus_pr"] > 0.9


def test_segmented_vus_scores_station_wide_generated_labels():
    generated = inject_synthetic_anomaly_segments(
        [_base_series(40) for _ in range(10)], "combined", seed=7
    )
    labels = [segment_labels for _, segment_labels in generated]
    result = metrics_module.compute_segmented_metrics(
        labels,
        [segment_labels.astype(float) for segment_labels in labels],
        metrics_module.vus_sliding_window_segments(labels),
    )

    assert sum(np.any(segment_labels) for segment_labels in labels) == 4
    assert sum(not np.any(segment_labels) for segment_labels in labels) == 6
    assert result["metrics"]["vus_pr"] > 0.9


def test_segmented_vus_matches_single_series_implementation():
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.4, 0.0])

    expected = metrics_module.vus_roc_pr(labels, scores, 2, thre=17)
    actual = metrics_module.vus_roc_pr_segments([labels], [scores], 2, thre=17)

    assert actual == expected


def test_segmented_vus_does_not_join_events_across_boundaries():
    labels = [np.array([0, 0, 1]), np.array([1, 0, 0])]

    assert metrics_module.vus_sliding_window_segments(labels) == 1
    assert metrics_module.vus_sliding_window(np.concatenate(labels)) == 2


def test_segmented_vus_tolerance_does_not_cross_boundaries():
    labels = [np.array([0, 0, 1]), np.array([0, 0, 0])]
    scores = [np.zeros(3), np.array([1.0, 0.0, 0.0])]
    normalized = metrics_module.normalize_scores(np.concatenate(scores))
    segmented = metrics_module.vus_roc_pr_segments(
        labels, list(np.split(normalized, [3])), 2, thre=7
    )[1]
    concatenated = metrics_module.vus_roc_pr(
        np.concatenate(labels), normalized, 2, thre=7
    )[1]

    assert segmented < concatenated


def test_segmented_vus_is_invariant_to_segment_order():
    labels = [np.array([0, 1, 1, 0]), np.array([0, 0, 1, 0])]
    scores = [np.array([0.1, 0.9, 0.8, 0.2]), np.array([0.3, 0.2, 0.7, 0.1])]

    forward = metrics_module.vus_roc_pr_segments(labels, scores, 2, thre=13)
    reverse = metrics_module.vus_roc_pr_segments(
        list(reversed(labels)), list(reversed(scores)), 2, thre=13
    )

    assert reverse == pytest.approx(forward)


def test_segmented_affiliation_reports_orphan_predictions():
    labels = [np.array([0, 0, 1, 1]), np.zeros(4, dtype=np.int64)]
    scores = [np.array([0.0, 0.0, 1.0, 1.0]), np.ones(4)]
    masks = [np.array([0, 0, 1, 1], dtype=bool), np.ones(4, dtype=bool)]

    result = metrics_module.compute_segmented_metrics(
        labels, scores, window_size=1, prediction_masks_by_segment=masks
    )

    assert np.isnan(result["metrics"]["affiliation_f1"])
    assert result["affiliation_diagnostics"]["status"] == "orphan_predictions"
    assert result["affiliation_diagnostics"]["orphan_prediction_events"] == 1
    assert result["affiliation_diagnostics"]["orphan_prediction_points"] == 4


def test_segmented_affiliation_aggregates_events_before_f1():
    labels = [np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0])]
    scores = [labels[0].astype(float), np.zeros(4)]
    masks = [labels[0].astype(bool), np.zeros(4, dtype=bool)]

    result = metrics_module.compute_segmented_metrics(
        labels, scores, window_size=1, prediction_masks_by_segment=masks
    )

    assert result["metrics"]["affiliation_f1"] == pytest.approx(2 / 3)
    assert result["affiliation_diagnostics"]["ground_truth_events"] == 2


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


def test_normalize_scores_constant_array_returns_zeros():
    out = normalize_scores(np.array([5.0, 5.0, 5.0]))
    assert np.all(out == 0.0)


def test_normalize_scores_scales_to_unit_range():
    out = normalize_scores(np.array([0.0, 5.0, 10.0]))
    assert out.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_scores_treats_nonfinite_values_as_no_detection():
    out = normalize_scores(np.array([np.nan, 2.0, 4.0, np.inf]))
    assert out.tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0])


# --- ensemble ------------------------------------------------------------


def test_rank_top_k_picks_highest():
    metrics = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.7}
    assert rank_top_k(metrics, k=3) == ["b", "d", "c"]


def test_rank_top_k_breaks_ties_by_name():
    assert rank_top_k({"b": 0.5, "a": 0.5, "c": 0.1}, k=2) == ["a", "b"]


def test_rank_top_k_k_larger_than_input_returns_all():
    assert rank_top_k({"a": 0.1, "b": 0.2}, k=5) == ["b", "a"]


def test_rank_top_k_excludes_models_without_finite_selection_metric():
    assert rank_top_k(
        {"valid": 0.4, "nan": np.nan, "inf": np.inf, "missing": None}, k=4
    ) == ["valid"]


def test_ranked_pointwise_vote_backfills_and_requires_two_models():
    scores = {
        "a": np.array([10.0, 10.0, 0.0, 10.0, 0.0, 0.0]),
        "b": np.array([np.nan, 10.0, np.nan, 0.0, 0.0, 0.0]),
        "c": np.array([10.0, 0.0, np.nan, np.nan, 0.0, 0.0]),
        "d": np.array([0.0, 0.0, np.nan, np.nan, 0.0, 0.0]),
    }

    fused, supported, used = ranked_pointwise_vote(
        scores, ["a", "b", "c", "d"], top_k=3, threshold_k=0.0
    )

    assert supported.tolist() == [True, True, False, True, True, True]
    assert fused.astype(bool).tolist() == [True, True, False, False, False, False]
    assert used == ["a", "b", "c", "d"]


def test_ranked_pointwise_minmax_mean_normalizes_per_model_and_station():
    scores = {
        "a": [np.array([0.0, 5.0]), np.array([10.0, np.nan])],
        "b": [np.array([100.0, 150.0]), np.array([200.0, 250.0])],
        "c": [np.full(2, 7.0), np.full(2, 7.0)],
        "d": [np.zeros(2), np.array([0.0, 1.0])],
    }

    fused, supported, used = ranked_pointwise_minmax_mean(
        scores,
        [{name: 1.0 - index / 10 for index, name in enumerate(scores)}] * 2,
        (2, 2),
    )

    assert [part.tolist() for part in supported] == [[True, True], [True, True]]
    assert fused[0].tolist() == pytest.approx([0.0, (0.5 + 1 / 3) / 3])
    assert fused[1].tolist() == pytest.approx([(1.0 + 2 / 3) / 3, 2 / 3])
    assert used == [["a", "b", "c"], ["a", "b", "c", "d"]]


def test_ranked_pointwise_minmax_mean_backfills_and_requires_two_scores():
    scores = {
        "a": [np.array([0.0, np.nan, np.nan])],
        "b": [np.array([np.nan, 0.0, np.nan])],
        "c": [np.array([1.0, np.nan, np.nan])],
        "d": [np.array([0.5, 1.0, np.nan])],
    }

    fused, supported, used = ranked_pointwise_minmax_mean(
        scores,
        [{"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6}],
        (3,),
    )

    assert supported[0].tolist() == [True, True, False]
    assert fused[0][:2].tolist() == pytest.approx([0.0, 0.5])
    assert np.isnan(fused[0][2])
    assert used == [["a", "b", "c", "d"]]


@pytest.mark.parametrize("top_k", [1, 4])
def test_synthetic_top_k_must_preserve_two_of_three_protocol(top_k: int):
    with pytest.raises(ValueError, match="ensemble_top_k"):
        AnomalyBenchmarkConfig(mode="synthetic", ensemble_top_k=top_k)


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


def test_sub_pca_benchmark_kwargs_select_components_and_weighting():
    config = AnomalyBenchmarkConfig(sub_pca_components=4)

    assert _requested_model_kwargs("Sub_PCA", config, "cpu") == {
        "device": "cpu",
        "weighted": True,
        "n_selected_components": 4,
    }
    assert _requested_model_kwargs(
        "Sub_PCA", AnomalyBenchmarkConfig(), "cpu"
    )["n_selected_components"] is None
    multi_config = AnomalyBenchmarkConfig(sub_pca_components=(None, 4, 8))
    assert _expand_model_names(
        ["IQR", "Sub_PCA"], multi_config.sub_pca_components
    ) == ["IQR", "Sub_PCA_all", "Sub_PCA_k4", "Sub_PCA_k8"]
    assert _requested_model_kwargs("Sub_PCA_k8", multi_config, "cpu")[
        "n_selected_components"
    ] == 8


def test_sub_pca_component_count_must_be_positive_or_all():
    with pytest.raises(ValueError, match="positive"):
        AnomalyBenchmarkConfig(sub_pca_components=0)


def test_summarize_aggregates_and_drops_scores():
    series_results = [
        {
            "series_length": 3,
            "segment_lengths": [3],
            "scored_points": 2,
            "scored_segments": [True],
            "metrics": {key: 1.0 for key in UNLABELED_METRIC_KEYS},
            "timing": {"fit_seconds": 1.0, "inference_seconds": 2.0},
            "scores": np.zeros(3),
        },
        {
            "series_length": 3,
            "segment_lengths": [1, 2],
            "scored_points": 1,
            "scored_segments": [False, True],
            "metrics": {key: 0.0 for key in UNLABELED_METRIC_KEYS},
            "timing": {"fit_seconds": 3.0, "inference_seconds": 4.0},
            "scores": np.zeros(3),
        },
    ]

    out = _summarize(series_results)

    assert out["macro_metrics"]["detection_rate"] == 0.5
    assert out["timing"]["mean_fit_seconds"] == 2.0
    assert all("scores" not in entry for entry in out["series_results"])
    assert out["series_results"][0]["counts"] == {
        "total_points": 3,
        "evaluated_points": 2,
        "total_segments": 1,
        "evaluated_segments": 1,
    }
    assert out["series_results"][1]["counts"] == {
        "total_points": 3,
        "evaluated_points": 1,
        "total_segments": 2,
        "evaluated_segments": 1,
    }


def test_summarize_derives_metric_keys_from_entries():
    series_results = [
        {
            "metrics": {key: 0.5 for key in SYNTHETIC_METRIC_KEYS},
            "timing": {"fit_seconds": 1.0, "inference_seconds": 1.0},
        }
    ]
    out = _summarize(series_results)
    assert set(out["macro_metrics"]) == set(SYNTHETIC_METRIC_KEYS)


def test_synthetic_metrics_include_single_class_segments_in_coverage():
    valid_labels = np.array([0, 0, 1, 1, 0, 0, 1, 0], dtype=np.int64)
    invalid_labels = np.ones(4, dtype=np.int64)
    scores = np.array([0.0, 0.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan])

    result = benchmark_module._synthetic_score_summary(
        [valid_labels, invalid_labels],
        [scores, np.arange(4, dtype=float)],
    )

    assert result["eligible_points"] == 12
    assert result["scored_points"] == 8
    assert result["coverage_rate"] == pytest.approx(2 / 3)
    assert result["positive_coverage_rate"] == pytest.approx(6 / 7)
    assert result["negative_coverage_rate"] == pytest.approx(2 / 5)
    for metric in SYNTHETIC_METRIC_KEYS:
        if np.isfinite(result["raw_metrics"][metric]):
            assert result["metrics"][metric] == pytest.approx(
                result["raw_metrics"][metric] * 2 / 3
            )


def test_synthetic_metrics_keep_globally_single_class_station_for_coverage():
    labels = np.ones(4, dtype=np.int64)

    result = benchmark_module._synthetic_score_summary(
        [labels], [np.arange(4, dtype=float)]
    )

    assert result["eligible_points"] == 4
    assert result["scored_points"] == 4
    assert result["coverage_rate"] == 1.0
    assert np.isnan(result["raw_metrics"]["vus_pr"])
    assert np.isnan(result["metrics"]["vus_pr"])


def test_synthetic_metrics_score_zero_when_detector_covers_no_eligible_points():
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    result = benchmark_module._synthetic_score_summary(
        [labels], [np.full(4, np.nan)]
    )

    assert result["eligible_points"] == 4
    assert result["scored_points"] == 0
    assert result["coverage_rate"] == 0.0
    assert all(np.isnan(value) for value in result["raw_metrics"].values())
    assert all(value == 0.0 for value in result["metrics"].values())


def test_selection_vus_is_missing_when_detector_has_no_finite_support():
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    assert np.isnan(
        benchmark_module._selection_vus_pr(labels, np.full(4, np.nan))
    )


def test_selection_vus_is_missing_when_finite_support_contains_one_class():
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.0, 1.0, np.nan, np.nan])

    assert np.isnan(benchmark_module._selection_vus_pr(labels, scores))


def test_selection_vus_uses_raw_metric_without_coverage_multiplier():
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 0], dtype=np.int64)
    scores = np.array([0.0, 0.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan])
    expected = compute_metrics(
        labels[:4], scores[:4], metrics_module.vus_sliding_window(labels)
    )["vus_pr"]

    assert benchmark_module._selection_vus_pr(labels, scores) == pytest.approx(expected)


def test_summarize_excludes_intrinsically_ineligible_station_but_keeps_detector_failure():
    timing = {"fit_seconds": 0.0, "inference_seconds": 0.0}
    series_results = [
        {
            "metrics": {key: np.nan for key in SYNTHETIC_METRIC_KEYS},
            "raw_metrics": {key: np.nan for key in SYNTHETIC_METRIC_KEYS},
            "eligible_points": 0,
            "scored_points": 0,
            "coverage_rate": np.nan,
            "timing": timing,
        },
        {
            "metrics": {key: 0.0 for key in SYNTHETIC_METRIC_KEYS},
            "raw_metrics": {key: np.nan for key in SYNTHETIC_METRIC_KEYS},
            "eligible_points": 10,
            "scored_points": 0,
            "coverage_rate": 0.0,
            "timing": timing,
        },
    ]

    summary = _summarize(series_results)

    assert summary["macro_metrics"]["vus_pr"] == 0.0
    assert np.isnan(summary["macro_raw_metrics"]["vus_pr"])
    assert summary["macro_coverage_rate"] == 0.0


def test_json_safe_serializes_nonfinite_metrics_as_null():
    safe = benchmark_module._json_safe(
        {"missing": np.nan, "infinite": np.inf, "valid": np.float64(0.5)}
    )

    assert safe == {"missing": None, "infinite": None, "valid": 0.5}


def test_plot_metric_values_omit_ineligible_null_but_keep_real_zero():
    summary = {
        "series_results": [
            {"metrics": {"vus_pr": None}},
            {"metrics": {"vus_pr": 0.0}},
            {"metrics": {"vus_pr": 0.4}},
        ]
    }

    assert presentation_module._series_metric_values(summary, "vus_pr") == [0.0, 0.4]


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
    stations = [("Good", _hourly_5m_frame(320))]
    monkeypatch.setattr(benchmark_module, "load_raw_5m", lambda pollutant, base_dir: stations)

    config = AnomalyBenchmarkConfig(mode="synthetic", min_series_points=6)
    cases = benchmark_module.build_cases(config)

    (case,) = cases
    assert case.labels.sum() >= 1 and case.labels_select.sum() >= 1
    # Selection and evaluation injections are independent (different seeds).
    assert not np.array_equal(case.values, case.values_select)


def test_build_cases_synthetic_uses_configured_injection_variant(monkeypatch):
    stations = [("Good", _hourly_5m_frame(320))]
    variants: list[str] = []

    def fake_inject(segments, variant, seed):
        variants.append(variant)
        return [
            (values.astype(np.float32, copy=True), np.ones(len(values), dtype=np.int64))
            for values in segments
        ]

    monkeypatch.setattr(benchmark_module, "load_raw_5m", lambda pollutant, base_dir: stations)
    monkeypatch.setattr(
        benchmark_module, "inject_synthetic_anomaly_segments", fake_inject
    )

    benchmark_module.build_cases(
        AnomalyBenchmarkConfig(mode="synthetic", injection_variant="drift")
    )

    assert variants == ["drift", "drift"]


def test_build_cases_synthetic_evaluates_short_segments_but_selects_long_ones(monkeypatch):
    frame = _hourly_5m_frame(340)
    gap_start = frame.index[320 * 12]
    gap_end = frame.index[330 * 12]
    frame = frame.loc[(frame.index < gap_start) | (frame.index >= gap_end)]
    monkeypatch.setattr(
        benchmark_module, "load_raw_5m", lambda pollutant, base_dir: [("Station", frame)]
    )

    cases = benchmark_module.build_cases(
        AnomalyBenchmarkConfig(mode="synthetic", min_series_points=8)
    )

    (case,) = cases
    assert case.segment_lengths == (320, 10)
    assert case.selection_segment_lengths == (320,)
    assert case.selection_segment_indices == (0,)
    assert len(case.values) == 330
    assert len(case.values_select) == 320


def test_build_cases_synthetic_uses_longest_selection_fallback(monkeypatch):
    frame = _hourly_5m_frame(120)
    gap_start = frame.index[100 * 12]
    gap_end = frame.index[110 * 12]
    frame = frame.loc[(frame.index < gap_start) | (frame.index >= gap_end)]
    monkeypatch.setattr(
        benchmark_module, "load_raw_5m", lambda pollutant, base_dir: [("Station", frame)]
    )

    (case,) = benchmark_module.build_cases(
        AnomalyBenchmarkConfig(mode="synthetic", min_series_points=8)
    )

    assert case.segment_lengths == (100, 10)
    assert case.selection_segment_lengths == (100,)
    assert case.selection_segment_indices == (0,)


def test_build_cases_synthetic_selects_a_fallback_for_each_context_regime(monkeypatch):
    frame = _hourly_5m_frame(1072)
    first_gap = frame.index[700 * 12 : 701 * 12]
    second_gap = frame.index[951 * 12 : 952 * 12]
    frame = frame.drop(first_gap.union(second_gap))
    monkeypatch.setattr(
        benchmark_module, "load_raw_5m", lambda pollutant, base_dir: [("Station", frame)]
    )
    injection_calls = []
    real_inject = benchmark_module.inject_synthetic_anomaly_segments

    def recording_inject(segments, variant, seed):
        injection_calls.append((list(map(len, segments)), seed))
        return real_inject(segments, variant, seed)

    monkeypatch.setattr(
        benchmark_module, "inject_synthetic_anomaly_segments", recording_inject
    )

    (case,) = benchmark_module.build_cases(
        AnomalyBenchmarkConfig(
            mode="synthetic", injection_variant="drift", min_series_points=8
        )
    )

    assert case.segment_lengths == (700, 250, 120)
    assert case.selection_segment_lengths == (700, 250)
    assert case.selection_segment_indices == (0, 1)
    assert injection_calls == [([700, 250], 13), ([700, 250, 120], 101)]
    assert case.labels_select.any()


def test_score_case_synthetic_inherits_station_selection_ranking(monkeypatch):
    lengths = (320, 10, 350)
    selection_lengths = (320, 350)
    first_labels = np.zeros(320, dtype=np.int64)
    first_labels[20:30] = 1
    short_labels = np.zeros(10, dtype=np.int64)
    short_labels[4:6] = 1
    last_labels = np.zeros(350, dtype=np.int64)
    last_labels[40:50] = 1
    case = AnomalyCase(
        name="Station",
        values=np.zeros(sum(lengths), dtype=np.float32),
        labels=np.concatenate([first_labels, short_labels, last_labels]),
        values_select=np.zeros(sum(selection_lengths), dtype=np.float32),
        labels_select=np.concatenate([first_labels, last_labels]),
        segment_lengths=lengths,
        segment_indices=_segment_indices(lengths),
        selection_segment_lengths=selection_lengths,
        selection_segment_indices=(0, 2),
    )
    call_count = 0

    def fake_fit_score(_cls, _kwargs, segments, _seed, _device, **_extra):
        nonlocal call_count
        call_count += 1
        model = type("Model", (), {"training_summary_": {}})()
        if call_count == 1:
            return model, [first_labels.astype(float), np.zeros(350)], 1.0, 2.0
        evaluation_scores = [np.zeros(len(segment)) for segment in segments]
        evaluation_scores[0][:5] = np.nan
        evaluation_scores[1] = None
        return model, evaluation_scores, 3.0, 4.0

    monkeypatch.setattr(benchmark_module, "_fit_score_timed", fake_fit_score)

    result = benchmark_module._score_case_synthetic(
        object, {}, case, AnomalyBenchmarkConfig(mode="synthetic"), "cpu"
    )

    rankings = result["vus_pr_select_by_segment"]
    assert rankings[0] != rankings[2]
    assert rankings[1] == pytest.approx((rankings[0] + rankings[2]) / 2)
    assert result["timing"]["selection_fit_seconds"] == 1.0
    assert result["timing"]["fit_seconds"] == 3.0
    assert result["timing"]["total_fit_seconds"] == 4.0
    assert result["scored_segments"] == [True, False, True]
    assert result["scored_points"] == 665


def test_score_case_synthetic_separates_native_and_padded_fallbacks(monkeypatch):
    lengths = (700, 420, 150)
    selected_lengths = (700, 420)
    labels = [np.zeros(length, dtype=np.int64) for length in lengths]
    selected_labels = [np.zeros(length, dtype=np.int64) for length in selected_lengths]
    selected_labels[0][10] = 1
    selected_labels[1][10] = 1
    case = AnomalyCase(
        name="Station",
        values=np.zeros(sum(lengths), dtype=np.float32),
        labels=np.concatenate(labels),
        values_select=np.zeros(sum(selected_lengths), dtype=np.float32),
        labels_select=np.concatenate(selected_labels),
        segment_lengths=lengths,
        segment_indices=_segment_indices(lengths),
        selection_segment_lengths=selected_lengths,
        selection_segment_indices=(0, 1),
    )
    call_count = 0

    def fake_fit_score(_cls, _kwargs, segments, _seed, _device, **_extra):
        nonlocal call_count
        call_count += 1
        model = type("Model", (), {"training_summary_": {}})()
        if call_count == 1:
            return model, [np.full(700, 0.9), np.full(420, 0.2)], 0.0, 0.0
        return model, [np.zeros(len(segment)) for segment in segments], 0.0, 0.0

    monkeypatch.setattr(benchmark_module, "_fit_score_timed", fake_fit_score)
    monkeypatch.setattr(
        benchmark_module,
        "_selection_vus_pr",
        lambda _labels, scores: float(scores[0]),
    )
    monkeypatch.setattr(
        benchmark_module, "_selection_vus_pr_segments", lambda *_args: 0.7
    )
    result = benchmark_module._score_case_synthetic(
        object, {}, case, AnomalyBenchmarkConfig(mode="synthetic"), "cpu"
    )

    assert result["vus_pr_select"] == pytest.approx(0.7)
    assert result["vus_pr_select_by_segment"] == pytest.approx([0.9, 0.2, 0.2])


def test_score_case_synthetic_fits_selection_sources_together(monkeypatch):
    lengths = (700, 250, 120)
    selected_lengths = (700, 250)
    labels = [np.zeros(length, dtype=np.int64) for length in lengths]
    selected_labels = [np.zeros(length, dtype=np.int64) for length in selected_lengths]
    selected_labels[0][10] = 1
    selected_labels[1][10] = 1
    case = AnomalyCase(
        name="Station",
        values=np.zeros(sum(lengths), dtype=np.float32),
        labels=np.concatenate(labels),
        values_select=np.zeros(sum(selected_lengths), dtype=np.float32),
        labels_select=np.concatenate(selected_labels),
        segment_lengths=lengths,
        segment_indices=_segment_indices(lengths),
        selection_segment_lengths=selected_lengths,
        selection_segment_indices=(0, 1),
    )
    calls = []

    def fake_fit_score(_cls, _kwargs, segments, _seed, _device, **_extra):
        calls.append(list(map(len, segments)))
        model = type("Model", (), {"training_summary_": {}})()
        return model, [np.zeros(len(segment)) for segment in segments], 0.0, 0.0

    monkeypatch.setattr(benchmark_module, "_fit_score_timed", fake_fit_score)

    benchmark_module._score_case_synthetic(
        object, {}, case, AnomalyBenchmarkConfig(mode="synthetic"), "cpu"
    )

    assert calls == [[700, 250], [700, 250, 120]]


def test_score_case_synthetic_rejects_missing_context_regime_selection():
    lengths = (700, 250)
    case = AnomalyCase(
        name="Station",
        values=np.zeros(sum(lengths), dtype=np.float32),
        labels=np.zeros(sum(lengths), dtype=np.int64),
        values_select=np.zeros(700, dtype=np.float32),
        labels_select=np.zeros(700, dtype=np.int64),
        segment_lengths=lengths,
        segment_indices=_segment_indices(lengths),
        selection_segment_lengths=(700,),
        selection_segment_indices=(0,),
    )

    with pytest.raises(ValueError, match="context regime"):
        benchmark_module._score_case_synthetic(
            object, {}, case, AnomalyBenchmarkConfig(mode="synthetic"), "cpu"
        )


def test_score_case_synthetic_rejects_misaligned_selection_lengths():
    lengths = (700, 420)
    case = AnomalyCase(
        name="Station",
        values=np.zeros(sum(lengths), dtype=np.float32),
        labels=np.zeros(sum(lengths), dtype=np.int64),
        values_select=np.zeros(sum(lengths), dtype=np.float32),
        labels_select=np.zeros(sum(lengths), dtype=np.int64),
        segment_lengths=lengths,
        segment_indices=_segment_indices(lengths),
        selection_segment_lengths=(420, 700),
        selection_segment_indices=(0, 1),
    )

    with pytest.raises(ValueError, match="lengths do not match"):
        benchmark_module._score_case_synthetic(
            object, {}, case, AnomalyBenchmarkConfig(mode="synthetic"), "cpu"
        )


def test_synthetic_ensemble_uses_local_and_inherited_rankings():
    first = np.zeros(320, dtype=np.float32)
    short = np.zeros(10, dtype=np.float32)
    last = np.zeros(350, dtype=np.float32)
    case = AnomalyCase(
        name="Station",
        values=np.concatenate([first, short, last]),
        labels=np.concatenate(
            [
                np.r_[np.zeros(300), np.ones(20)],
                np.r_[np.zeros(8), np.ones(2)],
                np.r_[np.zeros(330), np.ones(20)],
            ]
        ).astype(np.int64),
        values_select=np.concatenate([first, last]),
        labels_select=np.zeros(670, dtype=np.int64),
        segment_lengths=(320, 10, 350),
        segment_indices=_segment_indices((320, 10, 350)),
        selection_segment_lengths=(320, 350),
        selection_segment_indices=(0, 2),
    )
    scores_a = np.zeros(680)
    scores_a[:5] = np.nan
    scores_b = np.zeros(680)
    scores_b[320:330] = np.nan
    detector_results = {
        "a": {
            "per_case": [{
                "scores": scores_a,
                "vus_pr_select_by_segment": [0.9, 0.5, 0.1],
                "scored_segments": [True, True, True],
                "timing": {"fit_seconds": 1.0, "inference_seconds": 0.0},
            }]
        },
        "b": {
            "per_case": [{
                "scores": scores_b,
                "vus_pr_select_by_segment": [0.1, 0.5, 0.9],
                "scored_segments": [True, False, True],
                "timing": {"fit_seconds": 1.0, "inference_seconds": 0.0},
            }]
        },
    }

    (result,) = benchmark_module._build_synthetic_ensemble(
        AnomalyBenchmarkConfig(mode="synthetic", ensemble_top_k=2),
        [case],
        detector_results,
    )

    assert result["training_summary"]["selected_models_by_segment"] == [
        ["a", "b"],
        [],  # only one detector has support, so the ensemble abstains
        ["b", "a"],
    ]
    assert result["scored_segments"] == [True, False, True]
    assert result["scored_points"] == 665
    assert result["timing"]["fit_seconds"] == 2.0


def test_synthetic_ensemble_coverage_is_measured_after_pointwise_fallback():
    labels = np.array([0, 1, 1, 0, 0, 0], dtype=np.int64)
    case = AnomalyCase(
        name="Station",
        values=np.zeros(6, dtype=np.float32),
        labels=labels,
        values_select=np.zeros(6, dtype=np.float32),
        labels_select=labels,
        segment_lengths=(6,),
        segment_indices=_segment_indices((6,)),
        selection_segment_lengths=(6,),
        selection_segment_indices=(0,),
    )
    arrays = {
        "a": np.array([10.0, 10.0, 0.0, 10.0, 0.0, 0.0]),
        "b": np.array([np.nan, 10.0, np.nan, 0.0, 0.0, 0.0]),
        "c": np.array([10.0, 0.0, np.nan, np.nan, 0.0, 0.0]),
        "d": np.array([0.0, 0.0, np.nan, np.nan, 0.0, 0.0]),
    }
    detector_results = {
        name: {
            "per_case": [
                {
                    "scores": scores,
                    "vus_pr_select_by_segment": [1.0 - index / 10.0],
                    "scored_segments": [True],
                    "timing": {"fit_seconds": 1.0, "inference_seconds": 0.0},
                }
            ]
        }
        for index, (name, scores) in enumerate(arrays.items())
    }

    (result,) = benchmark_module._build_synthetic_ensemble(
        AnomalyBenchmarkConfig(mode="synthetic", ensemble_top_k=3, threshold_k=0.0),
        [case],
        detector_results,
    )

    assert result["eligible_points"] == 6
    assert result["scored_points"] == 5
    assert result["coverage_rate"] == pytest.approx(5 / 6)
    assert result["training_summary"]["selected_models_by_segment"] == [
        ["a", "b", "c", "d"]
    ]
    assert result["metrics"]["vus_pr"] == pytest.approx(
        result["raw_metrics"]["vus_pr"] * 5 / 6
    )
    assert result["timing"]["fit_seconds"] == 4.0
    assert result["training_summary"]["method"] == "MEAN_TOP_K_MINMAX"


def test_synthetic_ensemble_support_does_not_depend_on_evaluation_labels():
    scores = {
        "a": np.array([0.0, 0.0, 10.0, 0.0]),
        "b": np.array([0.0, 0.0, 9.0, 0.0]),
        "c": np.array([0.0, 0.0, 8.0, 0.0]),
    }
    detector_results = {
        name: {
            "per_case": [{
                "scores": values,
                "vus_pr_select_by_segment": [1.0 - index / 10.0],
                "scored_segments": [True],
                "timing": {"fit_seconds": 0.0, "inference_seconds": 0.0},
            }]
        }
        for index, (name, values) in enumerate(scores.items())
    }

    def build(labels: np.ndarray) -> dict[str, object]:
        case = AnomalyCase(
            name="Station",
            values=np.zeros(4, dtype=np.float32),
            labels=labels,
            values_select=np.zeros(4, dtype=np.float32),
            labels_select=np.array([0, 0, 1, 1]),
            segment_lengths=(4,),
            segment_indices=_segment_indices((4,)),
            selection_segment_lengths=(4,),
            selection_segment_indices=(0,),
        )
        return benchmark_module._build_synthetic_ensemble(
            AnomalyBenchmarkConfig(mode="synthetic"), [case], detector_results
        )[0]

    mixed = build(np.array([0, 0, 1, 1]))
    all_anomalous = build(np.ones(4, dtype=np.int64))

    assert mixed["scored_points"] == all_anomalous["scored_points"] == 4
    assert mixed["training_summary"]["selected_models_by_segment"] == (
        all_anomalous["training_summary"]["selected_models_by_segment"]
    )


def test_build_cases_keeps_all_observed_segments(monkeypatch):
    frame = _hourly_5m_frame(30)
    gap_start = frame.index.min() + np.timedelta64(10, "h")
    gap_end = gap_start + np.timedelta64(5, "h")
    frame = frame.loc[(frame.index < gap_start) | (frame.index >= gap_end)]
    monkeypatch.setattr(
        benchmark_module,
        "load_raw_5m",
        lambda pollutant, base_dir: [("Station", frame)],
    )

    (case,) = benchmark_module.build_cases(
        AnomalyBenchmarkConfig(min_series_points=8)
    )

    assert case.segment_lengths == (10, 15)
    assert len(case.values) == 25
    assert len(case.segment_indices) == 2


def test_unlabeled_scores_and_thresholds_each_segment_independently(monkeypatch):
    case = AnomalyCase(
        name="Station",
        values=np.zeros(30, dtype=np.float32),
        segment_lengths=(10, 20),
        segment_indices=_segment_indices((10, 20)),
    )

    fit_calls = 0

    def fake_fit_score(_cls, _kwargs, segments, _seed, _device, **_extra):
        nonlocal fit_calls
        fit_calls += 1
        scores = []
        for segment in segments:
            part = np.zeros(len(segment), dtype=float)
            part[-1] = 10.0
            scores.append(part)
        model = type("Model", (), {"training_summary_": {}})()
        return model, scores, 0.0, 0.0

    monkeypatch.setattr(benchmark_module, "_fit_score_timed", fake_fit_score)

    result = benchmark_module._score_case_unlabeled(
        object, {}, case, AnomalyBenchmarkConfig(), "cpu"
    )

    assert result["n_flagged"] == 2
    assert result["metrics"]["detection_rate"] == pytest.approx(2 / 30)
    assert result["scored_segments"] == [True, True]
    assert result["thresholds"] == [0.0, 0.0]
    assert fit_calls == 1


def test_unlabeled_counts_only_finite_score_coverage(monkeypatch):
    case = AnomalyCase(
        "Station",
        np.zeros(10, dtype=np.float32),
        segment_indices=_segment_indices((10,)),
    )

    def fake_fit_score(*_args, **_kwargs):
        model = type("Model", (), {"training_summary_": {}})()
        return model, [np.array([np.nan] * 5 + [0.0, 0.0, 0.0, 0.0, 10.0])], 0.0, 0.0

    monkeypatch.setattr(benchmark_module, "_fit_score_timed", fake_fit_score)

    result = benchmark_module._score_case_unlabeled(
        object, {}, case, AnomalyBenchmarkConfig(), "cpu"
    )

    assert result["scored_points"] == 5
    assert result["n_flagged"] == 1
    assert result["metrics"]["detection_rate"] == pytest.approx(0.2)


def test_unlabeled_ensemble_uses_only_points_with_finite_votes():
    case = AnomalyCase(
        "Station",
        np.zeros(6),
        segment_lengths=(2, 4),
        segment_indices=_segment_indices((2, 4)),
    )
    scores = np.array([np.nan, np.nan, 0.0, 0.0, 0.0, 10.0])
    detector_results = {
        "model": {
            "per_case": [
                {
                    "scores": scores,
                    "scored_segments": [True, True],
                    "timing": {"fit_seconds": 0.0, "inference_seconds": 0.0},
                }
            ]
        }
    }
    selection = {
        "Station": {
            "kept_models": ["model"],
            "discarded_models": [],
            "unavailable_models": [],
        }
    }

    (result,) = benchmark_module._build_unlabeled_ensemble(
        AnomalyBenchmarkConfig(), [case], detector_results, selection
    )

    assert result["voted_points"] == 4
    assert result["scored_segments"] == [False, True]
    assert result["n_flagged"] == 1
    assert result["metrics"]["detection_rate"] == pytest.approx(0.25)
    assert _summarize([result])["series_results"][0]["counts"] == {
        "total_points": 6,
        "evaluated_points": 4,
        "total_segments": 2,
        "evaluated_segments": 1,
    }


def test_synthetic_failed_scores_remain_nan(monkeypatch):
    values = np.linspace(10.0, 20.0, 40, dtype=np.float32)
    selected_values, selected_labels = inject_synthetic_anomalies(
        values, INJECTION_VARIANT, seed=3
    )
    evaluated_values, evaluated_labels = inject_synthetic_anomalies(
        values, INJECTION_VARIANT, seed=101
    )
    case = AnomalyCase(
        "Station",
        evaluated_values,
        labels=evaluated_labels,
        values_select=selected_values,
        labels_select=selected_labels,
        segment_lengths=(len(values),),
        segment_indices=_segment_indices((len(values),)),
    )

    monkeypatch.setattr(
        benchmark_module,
        "fit_model_segments",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        benchmark_module,
        "score_model_segments",
        lambda _model, _segments: [None],
    )
    monkeypatch.setattr(
        benchmark_module,
        "_fit_score_timed",
        lambda *args, **kwargs: (object(), [None], 0.0, 0.0),
    )

    result = benchmark_module._score_case_synthetic(
        object, {}, case, AnomalyBenchmarkConfig(mode="synthetic"), "cpu"
    )

    assert np.isnan(result["scores"]).all()


def test_unlabeled_selection_is_independent_per_series():
    cases = [
        AnomalyCase("A", np.zeros(10), segment_indices=_segment_indices((10,))),
        AnomalyCase("B", np.zeros(10), segment_indices=_segment_indices((10,))),
    ]
    detector_results = {
        "left": {
            "per_case": [
                {"scored_points": 10, "metrics": {"detection_rate": 0.01}},
                {"scored_points": 10, "metrics": {"detection_rate": 0.20}},
            ]
        },
        "right": {
            "per_case": [
                {"scored_points": 10, "metrics": {"detection_rate": 0.20}},
                {"scored_points": 10, "metrics": {"detection_rate": 0.01}},
            ]
        },
        "missing": {
            "per_case": [
                {"scored_points": 0, "metrics": {"detection_rate": 0.0}},
                {"scored_points": 0, "metrics": {"detection_rate": 0.0}},
            ]
        },
    }

    selection = benchmark_module._selection_by_series(cases, detector_results, 0.07)

    assert selection["A"] == {
        "kept_models": ["left"],
        "discarded_models": ["right"],
        "unavailable_models": ["missing"],
    }
    assert selection["B"] == {
        "kept_models": ["right"],
        "discarded_models": ["left"],
        "unavailable_models": ["missing"],
    }


def test_unlabeled_matches_forecasting_consensus_per_series():
    import pandas as pd

    from airquality.data.segments import contiguous_observed_segments
    from airquality.forecasting.detection import ConsensusDetection, SeriesDetectionContext

    index = pd.date_range("2024-01-01", periods=240, freq="h")
    values = 20.0 + np.sin(np.arange(240) / 6.0)
    series = pd.Series(values, index=index, name="Station")
    series.iloc[100:105] = np.nan
    series.iloc[50] = 80.0
    segments = contiguous_observed_segments(series, min_len=8)
    case = AnomalyCase(
        "Station",
        np.concatenate([segment.to_numpy(dtype=np.float32) for segment in segments]),
        segment_lengths=tuple(map(len, segments)),
        segment_indices=tuple(pd.DatetimeIndex(segment.index) for segment in segments),
    )
    names = ["ModifiedZScore", "IQR", "Hampel_w24"]
    config = AnomalyBenchmarkConfig(models=names)
    detector_results = {}
    for name in names:
        model_cls = resolve_model_class(name)
        detector_results[name] = {
            "per_case": [
                benchmark_module._score_case_unlabeled(
                    model_cls, {}, case, config, "cpu"
                )
            ]
        }
    selection = benchmark_module._selection_by_series(
        [case], detector_results, config.max_detection_rate
    )
    (benchmark_result,) = benchmark_module._build_unlabeled_ensemble(
        config, [case], detector_results, selection
    )

    context = SeriesDetectionContext(series, detectors=names, device="cpu")
    forecasting_result = ConsensusDetection().detect(context)
    forecasting_mask = np.concatenate(
        [forecasting_result.mask.loc[segment.index].to_numpy() for segment in segments]
    )

    assert selection["Station"]["kept_models"] == forecasting_result.detectors
    assert np.array_equal(benchmark_result["scores"].astype(bool), forecasting_mask)


def _real_cases() -> list[AnomalyCase]:
    return [
        AnomalyCase(
            name="StationA",
            values=_spiky_series(700),
            segment_indices=_segment_indices((700,)),
        ),
        AnomalyCase(
            name="StationB",
            values=_spiky_series(700, spike_positions=(80, 300)),
            segment_indices=_segment_indices((700,)),
        ),
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
    assert summary["selection_scope"] == "series"
    assert set(summary["selection_by_series"]) == {"StationA", "StationB"}
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
    for model in payload["models"].values():
        for entry in model["series_results"]:
            assert entry["counts"] == {
                "total_points": 700,
                "evaluated_points": 700,
                "total_segments": 1,
                "evaluated_segments": 1,
            }
    assert (tmp_path / "scores.npz").exists()
    log_text = (tmp_path / "benchmark.log").read_text(encoding="utf-8")
    assert "Run started: anomaly_benchmark" in log_text
    assert "Anomaly benchmark artifacts saved" in log_text
    assert "Run completed: anomaly_benchmark" in log_text
    # The benchmark itself does NOT render plots (that is a separate script).
    assert not (tmp_path / "detection_rate_distribution.png").exists()


def test_run_benchmark_preserves_log_when_case_building_fails(tmp_path, monkeypatch):
    def fail_cases(_config):
        raise RuntimeError("case loading failed")

    monkeypatch.setattr(benchmark_module, "build_cases", fail_cases)
    config = AnomalyBenchmarkConfig(output_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="case loading failed"):
        run_benchmark(config)

    log_text = (tmp_path / "benchmark.log").read_text(encoding="utf-8")
    assert "Run failed: anomaly_benchmark" in log_text
    assert "RuntimeError: case loading failed" in log_text


def test_run_benchmark_expands_sub_pca_component_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: _real_cases()[:1])
    config = AnomalyBenchmarkConfig(
        models=["Sub_PCA"],
        sub_pca_components=(None, 4, 8),
        device="cpu",
        output_dir=str(tmp_path),
    )

    summary = run_benchmark(config)

    assert summary["model_names"] == [
        "Sub_PCA_all",
        "Sub_PCA_k4",
        "Sub_PCA_k8",
        "Ensemble",
    ]
    assert summary["config"]["sub_pca_components"] == (None, 4, 8)
    assert all(
        summary["models"][name]["series_results"][0]["training_summary"]["segments"][0][
            "weighted"
        ]
        for name in ("Sub_PCA_all", "Sub_PCA_k4", "Sub_PCA_k8")
    )


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


@pytest.mark.parametrize("mode", ["unlabeled", "synthetic"])
def test_time_plots_exclude_ensemble(tmp_path, monkeypatch, mode):
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "config": {"max_detection_rate": 0.07},
                "models": {"Detector": {}, "Ensemble": {}},
            }
        )
    )
    received = {}
    monkeypatch.setattr(
        plot_module,
        "save_training_time_plot",
        lambda _path, summaries, *_: received.setdefault("training", set(summaries)),
    )
    if mode == "synthetic":
        monkeypatch.setattr(plot_module, "save_vus_pr_distribution_plot", lambda *_: None)
        monkeypatch.setattr(
            plot_module,
            "save_vus_pr_vs_inference_plot",
            lambda _path, summaries, **_: received.setdefault(
                "scatter", set(summaries)
            ),
        )
    else:
        monkeypatch.setattr(
            plot_module, "save_detection_rate_distribution_plot", lambda *_: None
        )
        monkeypatch.setattr(
            plot_module,
            "save_detection_rate_vs_inference_plot",
            lambda _path, summaries, _rate: received.setdefault(
                "scatter", set(summaries)
            ),
        )

    save_benchmark_plots(results_path)

    assert received == {"training": {"Detector"}, "scatter": {"Detector"}}


def test_recompute_ensemble_matches_saved_run(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: _real_cases())
    config = AnomalyBenchmarkConfig(
        models=["ModifiedZScore", "IQR", "IsolationForest"],
        device="cpu",
        output_dir=str(tmp_path),
    )
    run_benchmark(config)

    out = recompute_ensemble(tmp_path, threshold_k=3.5)

    assert "Ensemble(method=VOTE,k=3.5)" in out
    for name in ("ModifiedZScore", "IQR", "IsolationForest"):
        assert name in out
    assert all(np.isfinite(value) for value in out.values())


def test_recompute_ensemble_matches_multisegment_run(tmp_path, monkeypatch):
    first = _spiky_series(120, spike_positions=(30,))
    second = _spiky_series(180, spike_positions=(40, 120))
    case = AnomalyCase(
        name="Station",
        values=np.concatenate([first, second]),
        segment_lengths=(len(first), len(second)),
        segment_indices=_segment_indices((len(first), len(second))),
    )
    monkeypatch.setattr(benchmark_module, "build_cases", lambda config: [case])
    summary = run_benchmark(
        AnomalyBenchmarkConfig(
            models=["ModifiedZScore", "IQR"],
            device="cpu",
            output_dir=str(tmp_path),
        )
    )

    out = recompute_ensemble(tmp_path)

    assert out["Ensemble(method=VOTE,k=3.5)"] == pytest.approx(
        summary["models"]["Ensemble"]["macro_metrics"]["detection_rate"]
    )


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
                segment_indices=_segment_indices((len(eval_v),)),
            )
        )
    return cases


def _synthetic_multisegment_case() -> AnomalyCase:
    segments = [_base_series(length) for length in (320, 10, 350)]
    selected = inject_synthetic_anomaly_segments(
        [segments[index] for index in (0, 2)], INJECTION_VARIANT, seed=3
    )
    evaluated = inject_synthetic_anomaly_segments(
        segments, INJECTION_VARIANT, seed=101
    )
    return AnomalyCase(
        name="Station",
        values=np.concatenate([values for values, _ in evaluated]),
        labels=np.concatenate([labels for _, labels in evaluated]),
        values_select=np.concatenate([values for values, _ in selected]),
        labels_select=np.concatenate([labels for _, labels in selected]),
        segment_lengths=(320, 10, 350),
        segment_indices=_segment_indices((320, 10, 350)),
        selection_segment_lengths=(320, 350),
        selection_segment_indices=(0, 2),
    )


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
    assert "schema_version" not in summary
    assert summary["model_names"][-1] == "Ensemble"
    for name in summary["model_names"]:
        model_summary = summary["models"][name]
        assert "vus_pr" in model_summary["macro_metrics"]
        assert "vus_pr" in model_summary["macro_raw_metrics"]
        assert 0.0 <= model_summary["macro_coverage_rate"] <= 1.0
        assert 0.0 <= model_summary["macro_positive_coverage_rate"] <= 1.0
        assert 0.0 <= model_summary["macro_negative_coverage_rate"] <= 1.0
        entry = model_summary["series_results"][0]
        assert set(
            (
                "raw_metrics",
                "eligible_points",
                "coverage_rate",
                "positive_coverage_rate",
                "negative_coverage_rate",
                "affiliation_diagnostics",
            )
        ) <= set(entry)
        assert entry["counts"] == {
            "total_points": 700,
            "evaluated_points": 700,
            "total_segments": 1,
            "evaluated_segments": 1,
        }

    results_path = tmp_path / "results.json"
    assert results_path.exists()
    payload = json.loads(results_path.read_text())
    assert "schema_version" not in payload
    assert "Ensemble" in payload["models"]
    assert payload["variants"] == [INJECTION_VARIANT]
    assert "NaN" not in results_path.read_text()
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

    assert plot_paths["metrics_plot"].name == "vus_pr_adjusted_distribution.png"
    assert plot_paths["coverage_plot"].name == "vus_pr_raw_vs_coverage.png"
    for key in ("metrics_plot", "coverage_plot", "scatter_plot", "training_plot"):
        assert plot_paths[key].exists()


def test_recompute_ensemble_synthetic_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark_module, "build_cases", lambda config: [_synthetic_multisegment_case()]
    )
    config = AnomalyBenchmarkConfig(
        mode="synthetic",
        models=["ModifiedZScore", "IQR", "IsolationForest"],
        device="cpu",
        output_dir=str(tmp_path),
    )
    summary = run_benchmark(config)

    out = recompute_ensemble(tmp_path, top_k=3)

    assert "Ensemble(method=MEAN_MINMAX,top_k=3)" in out
    for name in ("ModifiedZScore", "IQR", "IsolationForest"):
        assert name in out
    assert all(np.isfinite(value) for value in out.values())
    assert out["Ensemble(method=MEAN_MINMAX,top_k=3)"] == pytest.approx(
        summary["models"]["Ensemble"]["macro_metrics"]["vus_pr"]
    )

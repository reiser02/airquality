"""Equivalence tests for the performance-optimized code paths.

Each test compares an optimized implementation against a straightforward
reference implementation (the pre-optimization code, inlined here), so any
future change that silently alters results is caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality.anomaly._vendor.vus_volume import vus_roc_pr
from airquality.anomaly.models.baselines import Hampel6Detector, HampelDetector
from airquality.anomaly.models.common import aggregate_tail_scores
from airquality.anomaly.windowing import aggregate_window_scores
from airquality.imputation.benchmark import _generate_hybrid_tspulse_gaps
from airquality.imputation.imputers import _derive_context_before_gap


# --------------------------------------------------------------------------- #
# Reference implementations (pre-optimization behavior)
# --------------------------------------------------------------------------- #
def _reference_hampel(series: np.ndarray, window_size: int, mad_epsilon: float = 1e-6) -> np.ndarray:
    rolling = pd.Series(series).rolling(window=window_size, center=True, min_periods=1)
    median = rolling.median().to_numpy()
    mad = rolling.apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True).to_numpy()
    scale = np.maximum(1.4826 * mad, mad_epsilon)
    return (np.abs(series - median) / scale).astype(np.float32)


def _reference_aggregate_window_scores(window_scores, series_length, window_size):
    totals = np.zeros(series_length, dtype=np.float64)
    counts = np.zeros(series_length, dtype=np.float64)
    for start in range(window_scores.shape[0]):
        end = start + window_size
        totals[start:end] += window_scores[start]
        counts[start:end] += 1.0
    counts[counts == 0.0] = 1.0
    return (totals / counts).astype(np.float32)


def _reference_aggregate_tail_scores(window_scores, series_length, window_size, tail_length):
    totals = np.zeros(series_length, dtype=np.float64)
    counts = np.zeros(series_length, dtype=np.float64)
    clamped = max(1, min(tail_length, window_size))
    for start, score in enumerate(window_scores):
        end = min(start + window_size, series_length)
        tail_start = max(start, end - clamped)
        totals[tail_start:end] += float(score)
        counts[tail_start:end] += 1.0
    counts[counts == 0.0] = 1.0
    return (totals / counts).astype(np.float32)


def _reference_vus(labels, score, window_size, thre=250):
    """Verbatim RangeAUC_volume_opt inner loops recomputing pred per (window, threshold)."""
    from airquality.anomaly._vendor.vus_volume import metricor

    m = metricor()
    P = np.sum(labels)
    seq = m.range_convers_new(labels)
    l = m.new_sequence(labels, seq, window_size)
    score_sorted = -np.sort(-score)
    auc_3d = np.zeros(window_size + 1)
    ap_3d = np.zeros(window_size + 1)
    tp = np.zeros(thre)
    N_pred = np.zeros(thre)
    for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
        N_pred[k] = np.sum(score >= score_sorted[i])
    for window in range(window_size + 1):
        labels_extended = m.sequencing(labels, seq, window)
        L = m.new_sequence(labels_extended, seq, window)
        TF_list = np.zeros((thre + 2, 2))
        Precision_list = np.ones(thre + 1)
        j = 0
        for i in np.linspace(0, len(score) - 1, thre).astype(int):
            threshold = score_sorted[i]
            pred = score >= threshold
            lbl = labels_extended.copy()
            existence = 0
            for seg in L:
                lbl[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                if (pred[seg[0]:(seg[1] + 1)] > 0).any():
                    existence += 1
            for seg in seq:
                lbl[seg[0]:seg[1] + 1] = 1
            TP = 0
            N_labels = 0
            for seg in l:
                TP += np.dot(lbl[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                N_labels += np.sum(lbl[seg[0]:seg[1] + 1])
            TP += tp[j]
            FP = N_pred[j] - TP
            existence_ratio = existence / len(L)
            P_new = (P + N_labels) / 2
            recall = min(TP / P_new, 1)
            TPR = recall * existence_ratio
            N_new = len(lbl) - P_new
            FPR = FP / N_new
            Precision = TP / N_pred[j]
            j += 1
            TF_list[j] = [TPR, FPR]
            Precision_list[j] = Precision
        TF_list[j + 1] = [1, 1]
        width = TF_list[1:, 1] - TF_list[:-1, 1]
        height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
        auc_3d[window] = np.dot(width, height)
        width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
        height_PR = Precision_list[1:]
        ap_3d[window] = np.dot(width_PR, height_PR)
    return sum(auc_3d) / len(auc_3d), sum(ap_3d) / len(ap_3d)


def _reference_hybrid_gaps(*, series, gap_size, num_gaps, rng, freq, random_fraction):
    """Pre-optimization hybrid gap sampler (Timestamp sets + list shuffle)."""
    from pandas.tseries.frequencies import to_offset

    from airquality.imputation.benchmark import _build_gap_index, _generate_block_gaps

    total_missing = max(1, int(gap_size * num_gaps))
    random_missing = int(total_missing * random_fraction)
    block_missing = max(0, total_missing - random_missing)
    block_count = max(1, block_missing // max(1, gap_size))
    block_windows = _generate_block_gaps(
        series=series, gap_size=gap_size, num_gaps=block_count, rng=rng, freq=freq, min_gap_points=1
    )
    used_points: set[pd.Timestamp] = set()
    for window in block_windows:
        used_points.update(window.tolist())
    offset = to_offset(freq)
    available_points = list(series.index)
    rng.shuffle(available_points)
    random_points: list[pd.Timestamp] = []
    for ts in available_points:
        point = pd.Timestamp(ts)
        if point in used_points:
            continue
        if (point - offset) in used_points or (point + offset) in used_points:
            continue
        random_points.append(point)
        used_points.add(point)
        if len(random_points) >= random_missing:
            break
    random_points = sorted(random_points)
    point_windows = [_build_gap_index(ts, 1, freq=freq) for ts in random_points]
    return sorted(block_windows + point_windows, key=lambda idx: pd.Timestamp(idx[0]))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("window_size", [5, 6, 24])
@pytest.mark.parametrize("n", [10, 400])
def test_hampel_matches_pandas_rolling_reference(window_size: int, n: int) -> None:
    rng = np.random.default_rng(7)
    values = np.cumsum(rng.normal(size=n)).astype(np.float32)
    values[n // 2] += 20.0

    detector = HampelDetector(seed=13, window_size=window_size)
    detector.fit(values)
    out = detector.score(values)
    ref = _reference_hampel(np.asarray(values, dtype=np.float64), window_size)
    np.testing.assert_array_equal(out, ref)


def test_hampel6_default_window_matches_reference() -> None:
    rng = np.random.default_rng(8)
    values = rng.normal(size=200).astype(np.float32)
    detector = Hampel6Detector(seed=13)
    detector.fit(values)
    np.testing.assert_array_equal(
        detector.score(values), _reference_hampel(np.asarray(values, dtype=np.float64), 6)
    )


@pytest.mark.parametrize("n,window", [(50, 50), (300, 24), (1000, 100), (40, 60)])
def test_aggregate_window_scores_matches_loop_reference(n: int, window: int) -> None:
    rng = np.random.default_rng(9)
    scores = rng.random(max(1, n - window + 1))
    out = aggregate_window_scores(scores, n, window)
    ref = _reference_aggregate_window_scores(scores, n, window)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("n,window,tail", [(300, 24, 6), (1000, 100, 25), (50, 50, 100), (30, 8, 1)])
def test_aggregate_tail_scores_matches_loop_reference(n: int, window: int, tail: int) -> None:
    rng = np.random.default_rng(10)
    scores = rng.random(max(1, n - window + 1))
    out = aggregate_tail_scores(scores, n, window, tail)
    ref = _reference_aggregate_tail_scores(scores, n, window, tail)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-10)


def test_vus_matches_unhoisted_reference() -> None:
    rng = np.random.default_rng(11)
    n = 600
    labels = np.zeros(n, dtype=np.int64)
    labels[100:104] = 1
    labels[400:410] = 1
    score = rng.random(n)
    score[labels == 1] += 0.5

    vus_roc, vus_pr = vus_roc_pr(labels, score, 20)
    ref_roc, ref_pr = _reference_vus(labels, score, 20)
    assert vus_roc == ref_roc
    assert vus_pr == ref_pr


@pytest.mark.parametrize("seed", [42, 99])
@pytest.mark.parametrize("gap_size", [1, 5])
def test_hybrid_gap_generation_matches_timestamp_reference(seed: int, gap_size: int) -> None:
    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    values = pd.Series(np.arange(500, dtype=float), index=idx, name="S")

    windows_new = _generate_hybrid_tspulse_gaps(
        series=values, gap_size=gap_size, num_gaps=3,
        rng=np.random.default_rng(seed), freq="h", random_fraction=0.75,
    )
    windows_ref = _reference_hybrid_gaps(
        series=values, gap_size=gap_size, num_gaps=3,
        rng=np.random.default_rng(seed), freq="h", random_fraction=0.75,
    )

    assert len(windows_new) == len(windows_ref)
    for new, ref in zip(windows_new, windows_ref):
        assert list(new) == list(ref)


@pytest.mark.parametrize("required", [24, 72, 10**9])
def test_context_trimming_matches_untrimmed_derivation(required: int) -> None:
    from airquality.imputation.imputers import _build_clean_left_context

    idx = pd.date_range("2024-01-01", periods=400, freq="h")
    rng = np.random.default_rng(12)
    series = pd.Series(rng.normal(size=400), index=idx, name="S")
    series.iloc[50:53] = np.nan
    series.iloc[300] = np.nan
    all_map = {"S": series}

    for gap_start in (idx[55], idx[200], idx[305], idx[399]):
        untrimmed, _ = _derive_context_before_gap("S", gap_start, all_map, None, "h")
        trimmed, _ = _derive_context_before_gap(
            "S", gap_start, all_map, None, "h", required_context=required
        )
        ctx_untrimmed = _build_clean_left_context(untrimmed, gap_start, required, "h")
        ctx_trimmed = _build_clean_left_context(trimmed, gap_start, required, "h")
        pd.testing.assert_series_equal(ctx_trimmed, ctx_untrimmed)


def test_backtest_mase_matches_mae_over_seasonal_naive_denominator() -> None:
    from darts import TimeSeries
    from darts.metrics import mae

    from airquality.forecasting.backtest import _compute_backtest_mase

    rng = np.random.default_rng(13)
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    insample = pd.Series(
        10 + 3 * np.sin(2 * np.pi * np.arange(n) / 24.0) + rng.normal(0, 0.5, n),
        index=idx,
        name="S",
    )
    insample.iloc[30:35] = np.nan

    hold_idx = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=48, freq="h")
    actual_vals = 10 + 3 * np.sin(2 * np.pi * np.arange(48) / 24.0) + rng.normal(0, 0.5, 48)
    actual = TimeSeries.from_series(pd.Series(actual_vals, index=hold_idx), freq="h")
    pred = TimeSeries.from_series(
        pd.Series(actual_vals + rng.normal(0, 0.7, 48), index=hold_idx), freq="h"
    )

    # Reference: the pre-migration convention, MAE over the in-sample
    # seasonal-naive MAE computed on the interpolated training history.
    filled = insample.interpolate(method="time", limit_direction="both").ffill().bfill()
    values = filled.to_numpy(dtype=float)
    denominator = float(np.mean(np.abs(values[24:] - values[:-24])))

    out = _compute_backtest_mase(actual, pred, insample, seasonality_m=24, freq="h")
    assert out == pytest.approx(float(mae(actual, pred)) / denominator, rel=1e-9)


def test_backtest_mase_returns_nan_when_history_is_too_short() -> None:
    from darts import TimeSeries

    from airquality.forecasting.backtest import _compute_backtest_mase

    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    insample = pd.Series(np.arange(10, dtype=float), index=idx, name="S")
    hold_idx = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=4, freq="h")
    ts = TimeSeries.from_series(pd.Series([1.0, 2.0, 3.0, 4.0], index=hold_idx), freq="h")

    out = _compute_backtest_mase(ts, ts, insample, seasonality_m=24, freq="h")
    assert np.isnan(out)

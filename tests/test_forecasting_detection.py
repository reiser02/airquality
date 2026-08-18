"""Tests for the forecasting-benchmark detection strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import airquality.forecasting.detection as detection_module
from airquality.forecasting.cache import BenchmarkCache
from airquality.forecasting.detection import (
    ConsensusDetection,
    DetectionResult,
    InjectionTopKDetection,
    SeriesDetectionContext,
    apply_mask_transforms,
    build_detection_strategy,
)

BASELINE_DETECTORS = ["ModifiedZScore", "IQR", "Hampel_w24"]


def _seasonal_series(n: int = 900, name: str = "ST", seed: int = 0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    vals = (
        30.0
        + 8.0 * np.sin(np.arange(n) * 2 * np.pi / 24)
        + 4.0 * np.sin(np.arange(n) * 2 * np.pi / 168)
        + rng.normal(0, 1, n)
    )
    return pd.Series(vals, index=idx, name=name)


def _spike_scores(n: int, positions: list[int]) -> np.ndarray:
    """Score array whose MAD mask flags exactly ``positions`` (10 over a 0 floor)."""
    scores = np.zeros(n, dtype=float)
    scores[positions] = 10.0
    return scores


class _StubContext:
    """Context stand-in with fixed scores/ranking to test the vote mechanics."""

    def __init__(
        self,
        n: int = 30,
        scores_by_model: dict[str, np.ndarray] | None = None,
        ranking: dict[str, float] | None = None,
    ) -> None:
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        self.series = pd.Series(np.zeros(n), index=idx, name="S")
        self.segments = [self.series]
        self.seed = 13
        self._scores = scores_by_model or {}
        self._ranking = ranking or {}

    def empty_mask(self) -> pd.Series:
        return pd.Series(False, index=self.series.index, name="S")

    def total_observed(self) -> int:
        return len(self.series)

    def real_scores(self, names):
        return {name: [self._scores[name]] for name in names if name in self._scores}

    def selection_ranking(self):
        return dict(self._ranking)


# --------------------------------------------------------------------------- #
# Vote mechanics (stubbed scores)
# --------------------------------------------------------------------------- #
def test_injection_vote_requires_min_votes():
    n = 30
    context = _StubContext(
        n=n,
        scores_by_model={
            "A": _spike_scores(n, [5]),
            "B": _spike_scores(n, [5, 20]),
            "C": _spike_scores(n, [12]),
        },
        ranking={"A": 0.9, "B": 0.8, "C": 0.7},
    )
    strategy = InjectionTopKDetection(name="inject-vote", top_k=3, min_votes=2)

    result = strategy.detect(context)

    # Only position 5 is flagged by >= 2 of the 3 masks (20 and 12 get 1 vote).
    assert list(np.flatnonzero(result.mask.to_numpy())) == [5]
    assert result.detectors == ["A", "B", "C"]
    assert result.n_flagged == 1


def test_injection_best_uses_top_ranked_only():
    n = 30
    context = _StubContext(
        n=n,
        scores_by_model={
            "A": _spike_scores(n, [5]),
            "B": _spike_scores(n, [20]),
        },
        ranking={"A": 0.9, "B": 0.8},
    )
    strategy = InjectionTopKDetection(name="inject-best", top_k=1, min_votes=1)

    result = strategy.detect(context)

    assert result.detectors == ["A"]
    assert "B" in result.discarded
    assert list(np.flatnonzero(result.mask.to_numpy())) == [5]
    assert result.ranking == {"A": 0.9, "B": 0.8}


def test_injection_vote_abstains_when_fewer_models_available():
    # B and C failed on the real series: one detector cannot make a 2-vote
    # decision, so the segment is explicitly unscored rather than normal.
    n = 30
    context = _StubContext(
        n=n,
        scores_by_model={"A": _spike_scores(n, [5])},
        ranking={"A": 0.9, "B": 0.8, "C": 0.7},
    )
    strategy = InjectionTopKDetection(name="inject-vote", top_k=3, min_votes=2)

    result = strategy.detect(context)

    assert result.detectors == ["A"]
    assert not result.mask.any()
    assert not result.scored_mask.any()
    assert result.selected_by_segment == [["A"]]
    assert result.n_unscored == n


def test_injection_vote_backfills_failed_top_ranked_detector():
    n = 30
    context = _StubContext(
        n=n,
        scores_by_model={
            "B": _spike_scores(n, [5]),
            "C": _spike_scores(n, [5]),
            "D": _spike_scores(n, [20]),
        },
        ranking={"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6},
    )

    result = InjectionTopKDetection(
        name="inject-vote", top_k=3, min_votes=2
    ).detect(context)

    assert result.selected_by_segment == [["B", "C", "D"]]
    assert result.detectors == ["B", "C", "D"]
    assert list(np.flatnonzero(result.mask.to_numpy())) == [5]
    assert result.scored_mask.all()


def test_injection_vote_abstains_pointwise_without_quorum():
    n = 30
    first = _spike_scores(n, [20])
    second = _spike_scores(n, [20])
    first[:10] = np.nan
    second[10:20] = np.nan
    context = _StubContext(
        n=n,
        scores_by_model={"A": first, "B": second},
        ranking={"A": 0.9, "B": 0.8},
    )

    result = InjectionTopKDetection(
        name="inject-vote", top_k=2, min_votes=2
    ).detect(context)

    assert not result.scored_mask.iloc[:20].any()
    assert result.scored_mask.iloc[20:].all()
    assert list(np.flatnonzero(result.mask.to_numpy())) == [20]


# --------------------------------------------------------------------------- #
# Strategies on a real context
# --------------------------------------------------------------------------- #
def test_injection_strategies_flag_spike_on_real_series():
    series = _seasonal_series(seed=1)
    series.iloc[300] = 140.0
    series.iloc[100:105] = np.nan  # pre-existing gap stays out of detection

    context = SeriesDetectionContext(series, detectors=BASELINE_DETECTORS, device="cpu")
    best = InjectionTopKDetection(name="inject-best", top_k=1, min_votes=1).detect(context)
    vote = InjectionTopKDetection(name="inject-vote", top_k=3, min_votes=2).detect(context)

    assert len(best.detectors) == 1
    assert set(best.ranking) <= set(BASELINE_DETECTORS) and best.ranking
    assert bool(best.mask.iloc[300])

    assert 1 <= len(vote.detectors) <= 3
    assert bool(vote.mask.iloc[300])
    # Gaps are never flagged (they are not observed points).
    assert not best.mask.iloc[100:105].any()
    assert not vote.mask.iloc[100:105].any()


def test_strategies_share_detector_fits_through_context():
    series = _seasonal_series(n=700, seed=2)
    context = SeriesDetectionContext(series, detectors=BASELINE_DETECTORS, device="cpu")

    ConsensusDetection().detect(context)
    cached = {name: scores for name, scores in context._real_scores.items()}
    InjectionTopKDetection(name="inject-vote", top_k=3, min_votes=2).detect(context)

    # The injection strategy reuses the consensus arm's real-series scores.
    for name, scores in cached.items():
        assert context._real_scores[name] is scores
    # And the selection ranking is memoized for the next injection strategy.
    assert context.selection_ranking() is context.selection_ranking()


def test_context_ignores_detector_scores_with_wrong_length(monkeypatch):
    series = _seasonal_series(n=30)
    context = SeriesDetectionContext(series, detectors=["IQR"])
    monkeypatch.setattr(
        detection_module,
        "fit_model_segments",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        detection_module,
        "score_model_segments",
        lambda _model, segments: [np.zeros(len(segment) + 1) for segment in segments],
    )

    assert context.real_scores(["IQR"]) == {}


def test_context_ignores_detector_scores_without_finite_values(monkeypatch):
    series = _seasonal_series(n=30)
    context = SeriesDetectionContext(series, detectors=["IQR"])
    monkeypatch.setattr(
        detection_module,
        "fit_model_segments",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        detection_module,
        "score_model_segments",
        lambda _model, segments: [
            np.full(len(segment), np.nan) for segment in segments
        ],
    )

    assert context.real_scores(["IQR"]) == {}


def test_consensus_tracks_partial_finite_coverage():
    n = 30
    scores = _spike_scores(n, [20])
    scores[:10] = np.nan
    context = _StubContext(n=n, scores_by_model={"A": scores})
    context.model_names = ["A"]

    result = ConsensusDetection(max_detection_rate=1.0).detect(context)

    assert not result.scored_mask.iloc[:10].any()
    assert result.scored_mask.iloc[10:].all()
    assert result.n_unscored == 10
    assert result.detection_rate == pytest.approx(1 / 20)


def test_real_scores_resume_per_detector(tmp_path, monkeypatch):
    series = _seasonal_series(n=500, seed=6)
    cache = BenchmarkCache(tmp_path)
    names = BASELINE_DETECTORS[:2]
    calls = []

    def fake_score_segments(model_names, segments, *, seed, device, freq):
        name = model_names[0]
        calls.append(name)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return {name: [np.arange(len(segment), dtype=float) for segment in segments]}

    monkeypatch.setattr(detection_module, "_score_segments", fake_score_segments)
    kwargs = {"detectors": names, "cache": cache, "cache_key": {"series": "ST"}}

    with pytest.raises(KeyboardInterrupt):
        SeriesDetectionContext(series, **kwargs).real_scores(names)

    scores = SeriesDetectionContext(series, **kwargs).real_scores(names)

    assert set(scores) == set(names)
    assert calls == [names[0], names[1], names[1]]


def test_selection_ranking_resumes_per_detector(tmp_path, monkeypatch):
    series = _seasonal_series(n=500, seed=7)
    cache = BenchmarkCache(tmp_path)
    names = BASELINE_DETECTORS[:2]
    fit_calls = 0

    def fake_fit_segments(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        if fit_calls == 2:
            raise KeyboardInterrupt
        return object()

    monkeypatch.setattr(detection_module, "fit_model_segments", fake_fit_segments)
    monkeypatch.setattr(
        detection_module,
        "score_model_segments",
        lambda _model, segments: [
            np.linspace(0.0, 1.0, len(segment)) for segment in segments
        ],
    )
    monkeypatch.setattr(detection_module, "_selection_vus_pr", lambda labels, scores: 0.5)
    kwargs = {"detectors": names, "cache": cache, "cache_key": {"series": "ST"}}

    with pytest.raises(KeyboardInterrupt):
        SeriesDetectionContext(series, **kwargs).selection_ranking()

    ranking = SeriesDetectionContext(series, **kwargs).selection_ranking()

    assert ranking == {name: 0.5 for name in names}
    assert fit_calls == 3


def test_strategies_return_empty_result_without_segments():
    idx = pd.date_range("2024-01-01", periods=20, freq="h")
    series = pd.Series(np.nan, index=idx, name="S")
    context = SeriesDetectionContext(series, detectors=BASELINE_DETECTORS)

    for strategy in (
        ConsensusDetection(),
        InjectionTopKDetection(name="inject-best", top_k=1, min_votes=1),
    ):
        result = strategy.detect(context)
        assert result.detectors == []
        assert result.n_flagged == 0
        assert not result.mask.any()


def test_selection_segments_filters_short_runs_with_longest_fallback():
    series = _seasonal_series(n=600, seed=3)
    series.iloc[100:110] = np.nan  # runs of 100 and 490 points

    context = SeriesDetectionContext(
        series, detectors=BASELINE_DETECTORS, min_selection_points=300
    )
    assert [len(seg) for seg in context.selection_segments()] == [490]

    fallback = SeriesDetectionContext(
        series, detectors=BASELINE_DETECTORS, min_selection_points=1000
    )
    assert [len(seg) for seg in fallback.selection_segments()] == [490]


def test_selection_ranking_scores_every_detector():
    series = _seasonal_series(n=500, seed=4)
    context = SeriesDetectionContext(series, detectors=BASELINE_DETECTORS)

    ranking = context.selection_ranking()

    assert set(ranking) == set(BASELINE_DETECTORS)
    assert all(0.0 <= value <= 1.0 for value in ranking.values())


def test_selection_ranking_scores_failed_segment_as_no_detections(monkeypatch):
    series = _seasonal_series(n=610, seed=5)
    series.iloc[300] = np.nan  # two eligible segments: 300 and 309 points
    context = SeriesDetectionContext(series, detectors=["IQR"], min_selection_points=300)
    fit_calls = 0
    metric_scores = []

    def fake_fit_segments(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return object()

    def fake_vus_pr(labels, scores):
        metric_scores.append(np.asarray(scores))
        return 0.8 if np.ptp(scores) else 0.2

    monkeypatch.setattr(detection_module, "fit_model_segments", fake_fit_segments)
    monkeypatch.setattr(
        detection_module,
        "score_model_segments",
        lambda _model, segments: [
            np.linspace(0.0, 1.0, len(segments[0])),
            None,
        ],
    )
    monkeypatch.setattr(detection_module, "_selection_vus_pr", fake_vus_pr)

    ranking = context.selection_ranking()

    assert ranking["IQR"] == pytest.approx(0.5)
    assert fit_calls == 1
    assert len(metric_scores) == 2
    assert np.all(metric_scores[1] == metric_scores[1][0])


def test_selection_rankings_are_local_with_station_fallback(monkeypatch):
    series = _seasonal_series(n=712, seed=8)
    series.iloc[[300, 611]] = np.nan  # blocks of 300, 310, and 100 points
    context = SeriesDetectionContext(
        series,
        detectors=["IQR", "Hampel_w24"],
        min_selection_points=300,
    )
    values = {
        "IQRDetector": {300: 0.9, 310: 0.2},
        "HampelDetector": {300: 0.1, 310: 0.8},
    }

    monkeypatch.setattr(
        detection_module,
        "fit_model_segments",
        lambda model_cls, *_args, **_kwargs: model_cls.__name__,
    )
    monkeypatch.setattr(
        detection_module,
        "score_model_segments",
        lambda model, segments: [
            np.full(len(segment), values[model][len(segment)]) for segment in segments
        ],
    )
    monkeypatch.setattr(
        detection_module,
        "_selection_vus_pr",
        lambda _labels, scores: float(scores[0]),
    )

    rankings = context.selection_rankings()

    assert rankings[0] == {"IQR": 0.9, "Hampel_w24": 0.1}
    assert rankings[1] == {"IQR": 0.2, "Hampel_w24": 0.8}
    assert rankings[2] == pytest.approx({"IQR": 0.55, "Hampel_w24": 0.45})
    assert context.selection_ranking() == pytest.approx(
        {"IQR": 0.55, "Hampel_w24": 0.45}
    )


# --------------------------------------------------------------------------- #
# Mask transforms + factory
# --------------------------------------------------------------------------- #
def test_apply_mask_transforms_recomputes_counts():
    series = _seasonal_series(n=50, seed=5)
    mask = pd.Series(False, index=series.index, name=series.name)
    mask.iloc[10] = True
    result = DetectionResult(
        strategy="unlabeled", detectors=["A"], discarded=[], rates={},
        threshold_k=3.5, mask=mask, n_flagged=1, detection_rate=1 / 50,
    )

    def dilate(_series: pd.Series, m: pd.Series) -> pd.Series:
        return m | m.shift(1, fill_value=False) | m.shift(-1, fill_value=False)

    out = apply_mask_transforms(series, result, [dilate])

    assert out.n_flagged == 3
    assert list(np.flatnonzero(out.mask.to_numpy())) == [9, 10, 11]
    assert out.detection_rate == pytest.approx(3 / 50)
    # No transforms -> the result passes through untouched.
    assert apply_mask_transforms(series, result, None) is result


def test_build_detection_strategy_resolves_specs():
    unlabeled = build_detection_strategy("unlabeled", max_detection_rate=0.05)
    assert isinstance(unlabeled, ConsensusDetection)
    assert unlabeled.max_detection_rate == 0.05

    best = build_detection_strategy("inject-best")
    assert isinstance(best, InjectionTopKDetection)
    assert (best.top_k, best.min_votes) == (1, 1)

    vote = build_detection_strategy("inject-vote", vote_top_k=5, vote_min_votes=3)
    assert (vote.name, vote.top_k, vote.min_votes) == ("inject-vote", 5, 3)

    with pytest.raises(ValueError, match="entre 2 y top_k"):
        build_detection_strategy("inject-vote", vote_min_votes=1)

    with pytest.raises(ValueError):
        build_detection_strategy("nope")

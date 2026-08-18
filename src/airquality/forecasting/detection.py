"""Detection strategies for the forecasting benchmark.

The benchmark compares forecasting error across *arms* that differ in how
anomalies are detected (and optionally imputed) before training. Every
strategy consumes a :class:`SeriesDetectionContext` — a per-series cache of
detector scores — and returns a :class:`DetectionResult` with a boolean
anomaly mask, so strategies share detector fits instead of refitting per arm:

- ``unlabeled`` (:class:`ConsensusDetection`): the production method — score
  every detector on the real contiguous segments, discard those over the
  detection-rate budget, binarize their scores with MAD and combine the masks
  by strict-majority vote (:func:`airquality.anomaly.ensemble.consensus`).
- ``inject-best`` (:class:`InjectionTopKDetection`, ``top_k=1``): inject
  synthetic anomalies (:func:`airquality.anomaly.anomalies.inject_synthetic_anomalies`)
  into a copy of the training segments, rank every detector by VUS-PR against
  the injection labels, and keep the single best detector's MAD-thresholded
  mask on the real series.
- ``inject-vote`` (``top_k=3, min_votes=2``): rank long blocks locally (short
  blocks inherit the station mean), backfill detectors that cannot score a
  block, and flag points where at least 2 selected masks agree.

The injected copies are used ONLY to select detectors; the final mask always
comes from scores on the real (uninjected) series.

:data:`MaskTransform` hooks (:func:`apply_mask_transforms`) let the pipeline
post-process a strategy's mask (e.g. dilate events, drop one-point flags)
without touching the strategies themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence

import numpy as np
import pandas as pd

from airquality.anomaly._vendor.vus_volume import vus_roc_pr
from airquality.anomaly.anomalies import inject_synthetic_anomalies
from airquality.anomaly.ensemble import rank_top_k
from airquality.anomaly.metrics import (
    DEFAULT_MAX_DETECTION_RATE,
    DEFAULT_THRESHOLD_K,
    detect_mask,
    normalize_scores,
    vus_sliding_window,
)
from airquality.anomaly.registry import (
    filter_model_kwargs as _filter_model_kwargs,
    fit_model_segments,
    resolve_model_class,
    resolve_model_names,
    score_model_segments,
)
from airquality.data.segments import contiguous_observed_segments
from airquality.data.series import ensure_datetime_series

if TYPE_CHECKING:
    from airquality.forecasting.cache import BenchmarkCache

DEFAULT_SEED = 13

#: Minimum contiguous observed points a segment needs to enter detection.
MIN_SEGMENT_POINTS = 8

#: Selection-injection defaults. The variant matches the anomaly benchmark's
#: ``synthetic`` mode; the seed is independent of the detector seed so the
#: injected shapes do not covary with the models' own randomness. Only
#: segments with at least ``DEFAULT_MIN_SELECTION_POINTS`` points are injected:
#: on short segments the fixed per-type rates of ``ANOMALY_PROFILE`` (notably
#: the 16..48-point drift span) would cover most of the segment and make the
#: VUS-PR ranking meaningless.
DEFAULT_INJECTION_VARIANT = "combined"
DEFAULT_INJECTION_SEED = 101
DEFAULT_MIN_SELECTION_POINTS = 300

DEFAULT_VOTE_TOP_K = 3
DEFAULT_VOTE_MIN_VOTES = 2

STRATEGY_UNLABELED = "unlabeled"
STRATEGY_INJECT_BEST = "inject-best"
STRATEGY_INJECT_VOTE = "inject-vote"
DETECTION_STRATEGIES = (STRATEGY_UNLABELED, STRATEGY_INJECT_BEST, STRATEGY_INJECT_VOTE)

#: Mask post-processing hook: ``(series, mask) -> mask`` over the same index.
MaskTransform = Callable[[pd.Series, pd.Series], pd.Series]


@dataclass
class DetectionResult:
    """Outcome of one detection strategy on one series."""

    strategy: str  # strategy spec name ("unlabeled", "inject-best", "inject-vote")
    detectors: list[str]  # detectors whose scores built the final mask
    discarded: list[str]  # scored but excluded (over the rate budget / below top-k)
    rates: dict[str, float]  # per-detector detection rate on this series
    threshold_k: float
    mask: pd.Series  # boolean, indexed like the input series (True = anomaly)
    ranking: dict[str, float] = field(default_factory=dict)  # selection VUS-PR per detector
    scored_mask: pd.Series | None = None  # True where the strategy had enough detectors
    selected_by_segment: list[list[str]] = field(default_factory=list)
    n_flagged: int = 0
    n_unscored: int = 0
    detection_rate: float = 0.0  # rate of the final mask over scored observed points


class DetectionStrategy(Protocol):
    """Anything that turns a per-series scoring context into a mask."""

    name: str

    def detect(self, context: "SeriesDetectionContext") -> DetectionResult: ...


def _score_segments(
    model_names: list[str],
    segments: list[pd.Series],
    *,
    seed: int,
    device: str,
    freq: str = "h",
    carla_stride: int = 1,
) -> dict[str, list[np.ndarray | None]]:
    """Score every detector on every segment (``None`` where a detector fails).

    Detectors that fail on every segment are dropped entirely; a per-segment
    failure (e.g. a windowed model on a segment shorter than its window) only
    removes that detector from that segment's consensus.
    """
    scores_by_model: dict[str, list[np.ndarray | None]] = {}
    values = [segment.to_numpy(dtype=np.float32) for segment in segments]
    segment_indices = [pd.DatetimeIndex(segment.index) for segment in segments]
    for name in model_names:
        model_cls = resolve_model_class(name)
        requested_kwargs = {"device": device}
        if name == "Prophet":
            requested_kwargs["freq"] = freq
        if name == "Sub_PCA":
            requested_kwargs["weighted"] = True
        if name in {"CARLABase", "CARLAGenIAS"}:
            requested_kwargs["stride"] = carla_stride
        kwargs = _filter_model_kwargs(model_cls, requested_kwargs)
        try:
            model = fit_model_segments(
                model_cls,
                values,
                seed=seed,
                model_kwargs=kwargs,
                segment_indices=segment_indices,
            )
            segment_scores = score_model_segments(model, values)
        except Exception as exc:  # pragma: no cover - detector-specific failures
            logging.warning("Detector %s fallo al entrenar la serie: %s", name, exc)
            continue
        for index, (segment, scores) in enumerate(zip(segments, segment_scores, strict=True)):
            if scores is not None:
                array = np.asarray(scores)
                if array.shape != (len(segment),) or not np.isfinite(array).any():
                    logging.warning(
                        "Detector %s no produjo scores finitos validos en segmento %d",
                        name,
                        index,
                    )
                    segment_scores[index] = None
        if all(scores is None for scores in segment_scores):
            logging.warning("Detector %s fallo en todos los segmentos; se omite.", name)
            continue
        scores_by_model[name] = segment_scores
    return scores_by_model


def _detector_rate(segment_scores: list[np.ndarray | None], threshold_k: float) -> float:
    """Detection rate of one detector over every segment it scored."""
    flagged = 0
    total = 0
    for scores in segment_scores:
        if scores is None:
            continue
        finite = np.isfinite(scores)
        if not finite.any():
            continue
        mask = detect_mask(scores, threshold_k)
        flagged += int(mask.sum())
        total += int(finite.sum())
    return flagged / total if total else 0.0


def _selection_vus_pr(labels: np.ndarray, scores: np.ndarray) -> float:
    """VUS-PR of ``scores`` against injection ``labels`` (0.0 with no positives).

    Same convention as :func:`airquality.anomaly.metrics.compute_metrics`:
    min-max normalized scores and the label-derived sliding window, so the
    values are comparable across detectors.
    """
    flat = np.asarray(labels, dtype=np.int64).ravel()
    if flat.size == 0 or int(flat.sum()) == 0:
        return 0.0
    normalized = normalize_scores(np.asarray(scores, dtype=np.float64))
    _, vus_pr = vus_roc_pr(flat, normalized, vus_sliding_window(flat))
    return float(vus_pr)


class SeriesDetectionContext:
    """Per-series cache of detector scores shared by every detection strategy.

    Detector fits are the expensive part of the benchmark, so the context
    memoizes the two score families and hands them to any strategy that asks:

    - :meth:`real_scores`: each detector fit once over all block-local windows,
      then scored separately per contiguous observed segment;
    - :meth:`selection_ranking`: every detector's mean VUS-PR against synthetic
      anomalies injected into the (sufficiently long) segments — the label-based
      selection signal the injection strategies use, with no real labels needed.
      A detector that cannot score an injected segment is evaluated as detecting
      no anomalies there, rather than having that segment omitted from its rank.
    """

    def __init__(
        self,
        series: pd.Series,
        *,
        detectors: list[str] | None = None,
        seed: int = DEFAULT_SEED,
        device: str = "cpu",
        freq: str = "h",
        carla_stride: int = 1,
        injection_variant: str = DEFAULT_INJECTION_VARIANT,
        injection_seed: int = DEFAULT_INJECTION_SEED,
        min_selection_points: int = DEFAULT_MIN_SELECTION_POINTS,
        cache: BenchmarkCache | None = None,
        cache_key: dict[str, Any] | None = None,
    ) -> None:
        self.series = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
        # Work per contiguous observed segment: `dropna()` would stitch stretches
        # that are hours or days apart into one "continuous" hourly signal.
        self.segments = contiguous_observed_segments(self.series, min_len=MIN_SEGMENT_POINTS)
        self.model_names = resolve_model_names(detectors if detectors else ["all"])
        self.seed = seed
        self.device = device
        self.freq = freq
        if carla_stride < 1:
            raise ValueError("carla_stride debe ser positivo")
        self.carla_stride = int(carla_stride)
        self.injection_variant = injection_variant
        self.injection_seed = injection_seed
        self.min_selection_points = min_selection_points
        self.cache = cache
        self.cache_key = cache_key
        self._real_scores: dict[str, list[np.ndarray | None]] = {}
        self._real_failed: set[str] = set()
        self._ranking: dict[str, float] | None = None
        self._segment_rankings: list[dict[str, float]] | None = None

    def _cached(self, namespace: str, key: dict[str, Any]) -> Any | None:
        if self.cache is None or self.cache_key is None:
            return None
        return self.cache.get(namespace, {**self.cache_key, **key})

    def _store(self, namespace: str, key: dict[str, Any], value: Any) -> None:
        if self.cache is not None and self.cache_key is not None:
            self.cache.put(namespace, {**self.cache_key, **key}, value)

    def empty_mask(self) -> pd.Series:
        """All-False mask on the series' regular grid."""
        return pd.Series(False, index=self.series.index, name=self.series.name)

    def total_observed(self) -> int:
        """All observed points, including short runs where detectors abstain."""
        return int(self.series.notna().sum())

    def real_scores(self, names: Sequence[str]) -> dict[str, list[np.ndarray | None]]:
        """Per-segment scores on the real series for ``names`` (memoized).

        Only detectors not yet scored are fit; detectors that failed on every
        segment are remembered and silently omitted from the returned map.
        """
        missing = [name for name in names if name not in self._real_scores and name not in self._real_failed]
        for name in missing:
            cached = self._cached("detector_scores", {"detector": name})
            if cached is not None:
                self._real_scores[name] = cached
                continue
            if not self.segments:
                continue
            scored = _score_segments(
                [name],
                self.segments,
                seed=self.seed,
                device=self.device,
                freq=self.freq,
                carla_stride=self.carla_stride,
            )
            if name in scored:
                self._real_scores[name] = scored[name]
                self._store("detector_scores", {"detector": name}, scored[name])
            else:
                self._real_failed.add(name)
        return {name: self._real_scores[name] for name in names if name in self._real_scores}

    def selection_segment_indices(self) -> list[int]:
        """Long segments ranked locally, or the longest segment as fallback."""
        eligible = [
            index
            for index, segment in enumerate(self.segments)
            if len(segment) >= self.min_selection_points
        ]
        if eligible:
            return eligible
        if not self.segments:
            return []
        return [max(range(len(self.segments)), key=lambda index: len(self.segments[index]))]

    def selection_segments(self) -> list[pd.Series]:
        """Segments used to establish local and station fallback rankings."""
        return [self.segments[index] for index in self.selection_segment_indices()]

    def selection_rankings(self) -> list[dict[str, float]]:
        """Ranking per real segment, with short segments inheriting the station mean."""
        if self._segment_rankings is not None:
            return self._segment_rankings

        selected_indices = self.selection_segment_indices()
        if not selected_indices:
            self._ranking = {}
            self._segment_rankings = []
            return self._segment_rankings

        injected_pairs = []
        for segment_index in selected_indices:
            segment = self.segments[segment_index]
            injected_values, labels = inject_synthetic_anomalies(
                segment.to_numpy(dtype=np.float32),
                self.injection_variant,
                self.injection_seed + segment_index,
            )
            injected_pairs.append(
                (injected_values, labels, pd.DatetimeIndex(segment.index))
            )

        local = {index: {} for index in selected_indices}
        for name in self.model_names:
            cached_values: dict[int, float] = {}
            for segment_index in selected_indices:
                cache_key = {
                    "detector": name,
                    "segment_index": segment_index,
                    "ranking_policy": "local-long-fallback-v1",
                    "injection_variant": self.injection_variant,
                    "injection_seed": self.injection_seed,
                    "min_selection_points": self.min_selection_points,
                }
                cached = self._cached("selection_scores", cache_key)
                if cached is not None:
                    cached_values[segment_index] = float(cached)
            if len(cached_values) == len(selected_indices):
                for segment_index, value in cached_values.items():
                    local[segment_index][name] = value
                continue

            model_cls = resolve_model_class(name)
            requested_kwargs = {"device": self.device}
            if name == "Prophet":
                requested_kwargs["freq"] = self.freq
            if name == "Sub_PCA":
                requested_kwargs["weighted"] = True
            if name in {"CARLABase", "CARLAGenIAS"}:
                requested_kwargs["stride"] = self.carla_stride
            kwargs = _filter_model_kwargs(model_cls, requested_kwargs)
            try:
                model = fit_model_segments(
                    model_cls,
                    [injected for injected, _, _ in injected_pairs],
                    seed=self.seed,
                    model_kwargs=kwargs,
                    segment_indices=[index for _, _, index in injected_pairs],
                )
                selected_scores = score_model_segments(
                    model, [injected for injected, _, _ in injected_pairs]
                )
            except Exception as exc:  # pragma: no cover - detector-specific failures
                logging.warning(
                    "Detector %s fallo al entrenar segmentos inyectados: %s", name, exc
                )
                selected_scores = [None] * len(injected_pairs)

            for segment_index, (_, labels, _), scores in zip(
                selected_indices, injected_pairs, selected_scores, strict=True
            ):
                if scores is None or np.asarray(scores).shape != labels.shape:
                    scores = np.zeros(labels.shape, dtype=np.float64)
                value = _selection_vus_pr(labels, scores)
                local[segment_index][name] = value
                self._store(
                    "selection_scores",
                    {
                        "detector": name,
                        "segment_index": segment_index,
                        "ranking_policy": "local-long-fallback-v1",
                        "injection_variant": self.injection_variant,
                        "injection_seed": self.injection_seed,
                        "min_selection_points": self.min_selection_points,
                    },
                    value,
                )

        self._ranking = {
            name: float(np.mean([local[index][name] for index in selected_indices]))
            for name in self.model_names
            if all(name in local[index] for index in selected_indices)
        }
        self._segment_rankings = [
            dict(local[index]) if index in local else dict(self._ranking)
            for index in range(len(self.segments))
        ]
        return self._segment_rankings

    def selection_ranking(self) -> dict[str, float]:
        """Station fallback ranking: mean of the locally ranked long segments."""
        self.selection_rankings()
        return self._ranking or {}


def _empty_result(
    strategy: str,
    context: SeriesDetectionContext,
    threshold_k: float,
    ranking: dict[str, float] | None = None,
) -> DetectionResult:
    """Result with an all-False mask (no segments / no usable detectors)."""
    return DetectionResult(
        strategy=strategy,
        detectors=[],
        discarded=[],
        rates={},
        threshold_k=threshold_k,
        mask=context.empty_mask(),
        ranking=dict(ranking or {}),
        scored_mask=context.empty_mask(),
        n_unscored=context.total_observed(),
    )


@dataclass(frozen=True)
class ConsensusDetection:
    """Label-free consensus of the detectors under the detection-rate budget.

    The production method (see :mod:`airquality.forecasting.cleaning`): the
    target anomalies are sensor faults (spikes, calibration drift, cutouts),
    which are rare — detectors flagging more than ``max_detection_rate`` of the
    observed points are marking normal variation and are discarded. The final
    mask requires a strict majority of the survivors' MAD-thresholded masks per
    segment.
    """

    name: str = STRATEGY_UNLABELED
    threshold_k: float = DEFAULT_THRESHOLD_K
    max_detection_rate: float = DEFAULT_MAX_DETECTION_RATE
    def detect(self, context: SeriesDetectionContext) -> DetectionResult:
        if not context.segments:
            return _empty_result(self.name, context, self.threshold_k)

        scores_by_model = context.real_scores(context.model_names)
        rates = {
            name: _detector_rate(segment_scores, self.threshold_k)
            for name, segment_scores in scores_by_model.items()
        }
        survivors = sorted(name for name, rate in rates.items() if rate <= self.max_detection_rate)
        discarded = sorted(name for name, rate in rates.items() if rate > self.max_detection_rate)

        mask = context.empty_mask()
        scored_mask = context.empty_mask()
        n_flagged = 0
        for segment_index, segment in enumerate(context.segments):
            score_arrays = [
                scores_by_model[name][segment_index]
                for name in survivors
                if scores_by_model[name][segment_index] is not None
            ]
            if not score_arrays:
                continue
            finite = np.stack([np.isfinite(scores) for scores in score_arrays])
            votes = np.stack(
                [detect_mask(scores, self.threshold_k) for scores in score_arrays]
            )
            available = finite.sum(axis=0)
            supported = available > 0
            flagged = supported & (votes.sum(axis=0) > available / 2.0)
            mask.loc[segment.index] = flagged
            scored_mask.loc[segment.index] = supported
            n_flagged += int(flagged.sum())

        total_observed = context.total_observed()
        n_scored = int(scored_mask.sum())
        return DetectionResult(
            strategy=self.name,
            detectors=survivors,
            discarded=discarded,
            rates=rates,
            threshold_k=self.threshold_k,
            mask=mask,
            scored_mask=scored_mask,
            n_flagged=n_flagged,
            n_unscored=total_observed - n_scored,
            detection_rate=n_flagged / n_scored if n_scored else 0.0,
        )


@dataclass(frozen=True)
class InjectionTopKDetection:
    """Detectors selected by injection VUS-PR; masks combined by vote.

    ``top_k=1`` keeps the single best detector's MAD-thresholded mask. With
    ``top_k > 1`` each selected detector's real scores are binarized
    independently (same MAD rule per segment) and a point is flagged when at
    least ``min_votes`` masks agree. Segments with fewer available detectors
    than ``min_votes`` are unscored rather than silently treated as normal.
    """

    name: str = STRATEGY_INJECT_BEST
    top_k: int = 1
    min_votes: int = 1
    threshold_k: float = DEFAULT_THRESHOLD_K

    def __post_init__(self) -> None:
        minimum = 2 if self.name == STRATEGY_INJECT_VOTE else 1
        if self.top_k < 1 or not minimum <= self.min_votes <= self.top_k:
            raise ValueError(
                f"{self.name}: min_votes debe estar entre {minimum} y top_k"
            )

    def detect(self, context: SeriesDetectionContext) -> DetectionResult:
        if not context.segments:
            return _empty_result(self.name, context, self.threshold_k)

        ranking = context.selection_ranking()
        if not ranking:
            return _empty_result(self.name, context, self.threshold_k)
        segment_rankings = (
            context.selection_rankings()
            if hasattr(context, "selection_rankings")
            else [ranking] * len(context.segments)
        )

        # The mask always comes from the REAL series: the injection only ranks.
        scores_by_model = context.real_scores(list(ranking))

        mask = context.empty_mask()
        scored_mask = context.empty_mask()
        n_flagged = 0
        selected_by_segment: list[list[str]] = []
        selected_set: set[str] = set()
        for segment_index, (segment, local_ranking) in enumerate(
            zip(context.segments, segment_rankings, strict=True)
        ):
            eligible = [
                name
                for name in rank_top_k(local_ranking, len(local_ranking))
                if name in scores_by_model
                and scores_by_model[name][segment_index] is not None
            ]
            selected = eligible[: self.top_k]
            selected_by_segment.append(selected)
            selected_set.update(selected)
            if len(selected) < self.min_votes:
                continue
            segment_masks = [
                detect_mask(scores_by_model[name][segment_index], self.threshold_k)
                for name in selected
            ]
            available = np.sum(
                [
                    np.isfinite(scores_by_model[name][segment_index])
                    for name in selected
                ],
                axis=0,
            )
            supported = available >= self.min_votes
            flagged = supported & (np.sum(segment_masks, axis=0) >= self.min_votes)
            mask.loc[segment.index] = flagged
            scored_mask.loc[segment.index] = supported
            n_flagged += int(flagged.sum())

        selected_all = [
            name for name in rank_top_k(ranking, len(ranking)) if name in selected_set
        ]
        rates = {
            name: _detector_rate(scores_by_model[name], self.threshold_k)
            for name in selected_all
        }
        total_observed = context.total_observed()
        n_scored = int(scored_mask.sum())
        return DetectionResult(
            strategy=self.name,
            detectors=selected_all,
            discarded=sorted(set(ranking) - selected_set),
            rates=rates,
            threshold_k=self.threshold_k,
            mask=mask,
            ranking=dict(ranking),
            scored_mask=scored_mask,
            selected_by_segment=selected_by_segment,
            n_flagged=n_flagged,
            n_unscored=total_observed - n_scored,
            detection_rate=n_flagged / n_scored if n_scored else 0.0,
        )


def build_detection_strategy(
    spec: str,
    *,
    threshold_k: float = DEFAULT_THRESHOLD_K,
    max_detection_rate: float = DEFAULT_MAX_DETECTION_RATE,
    vote_top_k: int = DEFAULT_VOTE_TOP_K,
    vote_min_votes: int = DEFAULT_VOTE_MIN_VOTES,
) -> DetectionStrategy:
    """Build the strategy registered under ``spec`` with the shared knobs."""
    normalized = str(spec).strip().lower()
    if normalized == STRATEGY_UNLABELED:
        return ConsensusDetection(
            threshold_k=threshold_k,
            max_detection_rate=max_detection_rate,
        )
    if normalized == STRATEGY_INJECT_BEST:
        return InjectionTopKDetection(
            name=STRATEGY_INJECT_BEST, top_k=1, min_votes=1, threshold_k=threshold_k
        )
    if normalized == STRATEGY_INJECT_VOTE:
        return InjectionTopKDetection(
            name=STRATEGY_INJECT_VOTE,
            top_k=vote_top_k,
            min_votes=vote_min_votes,
            threshold_k=threshold_k,
        )
    raise ValueError(
        f"Estrategia de deteccion desconocida: '{spec}'. Usa una de {DETECTION_STRATEGIES}"
    )


def apply_mask_transforms(
    series: pd.Series,
    result: DetectionResult,
    transforms: Sequence[MaskTransform] | None,
) -> DetectionResult:
    """Post-process ``result.mask`` with ``(series, mask) -> mask`` hooks, in order.

    This is where future preprocessing of the detection output (event dilation,
    minimum event length, domain filters, ...) plugs into the benchmark without
    touching the strategies. Flag counts and the detection rate are recomputed
    over the observed points of ``series``.
    """
    if not transforms:
        return result
    mask = result.mask
    for transform in transforms:
        mask = transform(series, mask).reindex(result.mask.index, fill_value=False).astype(bool)
    if result.scored_mask is not None:
        mask &= result.scored_mask.reindex(mask.index, fill_value=False).astype(bool)
    n_flagged = int(mask.sum())
    scored = result.scored_mask if result.scored_mask is not None else series.notna()
    n_scored = int((scored.reindex(series.index, fill_value=False) & series.notna()).sum())
    return replace(
        result,
        mask=mask,
        n_flagged=n_flagged,
        detection_rate=n_flagged / n_scored if n_scored else 0.0,
    )


def common_detection_support(
    series: pd.Series,
    detections: dict[str, DetectionResult],
) -> tuple[pd.Series, pd.Series]:
    """Mask anomalies and observed points where any strategy abstained."""
    common_mask = pd.Series(False, index=series.index, name=series.name)
    for detection in detections.values():
        common_mask |= detection.mask.reindex(series.index, fill_value=False).astype(bool)
        if detection.scored_mask is not None:
            scored = detection.scored_mask.reindex(series.index, fill_value=False).astype(bool)
            common_mask |= series.notna() & ~scored
    return series.mask(common_mask), common_mask


__all__ = [
    "DEFAULT_INJECTION_SEED",
    "DEFAULT_INJECTION_VARIANT",
    "DEFAULT_MIN_SELECTION_POINTS",
    "DEFAULT_SEED",
    "DEFAULT_VOTE_MIN_VOTES",
    "DEFAULT_VOTE_TOP_K",
    "DETECTION_STRATEGIES",
    "MIN_SEGMENT_POINTS",
    "STRATEGY_INJECT_BEST",
    "STRATEGY_INJECT_VOTE",
    "STRATEGY_UNLABELED",
    "ConsensusDetection",
    "DetectionResult",
    "DetectionStrategy",
    "InjectionTopKDetection",
    "MaskTransform",
    "SeriesDetectionContext",
    "apply_mask_transforms",
    "build_detection_strategy",
    "common_detection_support",
]

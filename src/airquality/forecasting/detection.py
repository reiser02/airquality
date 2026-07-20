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
- ``inject-vote`` (``top_k=3, min_votes=2``): same ranking, but the top-3
  detectors' real scores are binarized independently and a point is flagged
  when at least 2 of the 3 masks agree.

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
from airquality.anomaly.ensemble import consensus, rank_top_k
from airquality.anomaly.metrics import (
    DEFAULT_MAX_DETECTION_RATE,
    DEFAULT_THRESHOLD_K,
    detect_mask,
    normalize_scores,
    vus_sliding_window,
)
from airquality.anomaly.registry import (
    filter_model_kwargs as _filter_model_kwargs,
    resolve_model_class,
    resolve_model_names,
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
    n_flagged: int = 0
    detection_rate: float = 0.0  # rate of the final mask over the observed points


class DetectionStrategy(Protocol):
    """Anything that turns a per-series scoring context into a mask."""

    name: str

    def detect(self, context: "SeriesDetectionContext") -> DetectionResult: ...


def _fit_score(model_cls: type, values: np.ndarray, *, seed: int, device: str) -> np.ndarray:
    """Instantiate, fit and score one detector on ``values``."""
    kwargs = _filter_model_kwargs(model_cls, {"device": device})
    model = model_cls(seed=seed, **kwargs)
    model.fit(values)
    return np.asarray(model.score(values), dtype=float)


def _score_segments(
    model_names: list[str],
    segments: list[pd.Series],
    *,
    seed: int,
    device: str,
) -> dict[str, list[np.ndarray | None]]:
    """Score every detector on every segment (``None`` where a detector fails).

    Detectors that fail on every segment are dropped entirely; a per-segment
    failure (e.g. a windowed model on a segment shorter than its window) only
    removes that detector from that segment's consensus.
    """
    scores_by_model: dict[str, list[np.ndarray | None]] = {}
    for name in model_names:
        model_cls = resolve_model_class(name)
        segment_scores: list[np.ndarray | None] = []
        for segment in segments:
            try:
                segment_scores.append(
                    _fit_score(model_cls, segment.to_numpy(dtype=np.float32), seed=seed, device=device)
                )
            except Exception as exc:  # pragma: no cover - detector-specific failures
                logging.warning(
                    "Detector %s fallo en un segmento de %d puntos: %s", name, len(segment), exc
                )
                segment_scores.append(None)
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
        mask = detect_mask(scores, threshold_k)
        flagged += int(mask.sum())
        total += int(mask.size)
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

    - :meth:`real_scores`: each detector fit/scored per contiguous observed
      segment of the real series (same convention as the production cleaning);
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
        self.injection_variant = injection_variant
        self.injection_seed = injection_seed
        self.min_selection_points = min_selection_points
        self.cache = cache
        self.cache_key = cache_key
        self._real_scores: dict[str, list[np.ndarray | None]] = {}
        self._real_failed: set[str] = set()
        self._ranking: dict[str, float] | None = None

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
        """Observed points that enter detection (sum of segment lengths)."""
        return sum(len(segment) for segment in self.segments)

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
            scored = _score_segments([name], self.segments, seed=self.seed, device=self.device)
            if name in scored:
                self._real_scores[name] = scored[name]
                self._store("detector_scores", {"detector": name}, scored[name])
            else:
                self._real_failed.add(name)
        return {name: self._real_scores[name] for name in names if name in self._real_scores}

    def selection_segments(self) -> list[pd.Series]:
        """Segments long enough for injection-based selection (longest as fallback)."""
        eligible = [seg for seg in self.segments if len(seg) >= self.min_selection_points]
        if eligible:
            return eligible
        return [max(self.segments, key=len)] if self.segments else []

    def selection_ranking(self) -> dict[str, float]:
        """Mean VUS-PR per detector over the injected selection segments (memoized).

        Each eligible segment is injected once (seed = ``injection_seed`` +
        segment position, so shapes decorrelate across segments) and every
        requested detector is fit/scored on the injected copy. A detector's
        ranking value is the mean VUS-PR over every injected segment. A failed
        fit is represented by constant all-normal scores, so VUS-PR measures the
        missed injected anomalies instead of silently omitting that segment.
        """
        if self._ranking is not None:
            return self._ranking

        ranking: dict[str, float] = {}
        injected_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None
        for name in self.model_names:
            cache_key = {
                "detector": name,
                "injection_variant": self.injection_variant,
                "injection_seed": self.injection_seed,
                "min_selection_points": self.min_selection_points,
            }
            cached = self._cached("selection_scores", cache_key)
            if cached is not None:
                ranking[name] = float(cached)
                continue
            if injected_pairs is None:
                injected_pairs = []
                for position, segment in enumerate(self.selection_segments()):
                    injected, labels = inject_synthetic_anomalies(
                        segment.to_numpy(dtype=np.float32),
                        self.injection_variant,
                        self.injection_seed + position,
                    )
                    injected_pairs.append((injected, labels))
            model_cls = resolve_model_class(name)
            vus_values: list[float] = []
            for injected, labels in injected_pairs:
                try:
                    scores = _fit_score(model_cls, injected, seed=self.seed, device=self.device)
                except Exception as exc:  # pragma: no cover - detector-specific failures
                    logging.warning(
                        "Detector %s fallo en un segmento inyectado de %d puntos; "
                        "se evalua como ninguna anomalia detectada: %s",
                        name, len(injected), exc,
                    )
                    scores = np.zeros(labels.shape, dtype=np.float64)
                vus_values.append(_selection_vus_pr(labels, scores))
            if vus_values:
                ranking[name] = float(np.mean(vus_values))
                self._store("selection_scores", cache_key, ranking[name])
            else:
                logging.warning(
                    "Detector %s fallo en todos los segmentos inyectados; fuera de la seleccion.", name
                )
        self._ranking = ranking
        return ranking


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
        n_flagged = 0
        for segment_index, segment in enumerate(context.segments):
            score_arrays = [
                scores_by_model[name][segment_index]
                for name in survivors
                if scores_by_model[name][segment_index] is not None
            ]
            if not score_arrays:
                continue
            flagged = consensus(score_arrays, threshold_k=self.threshold_k).astype(bool)
            mask.loc[segment.index] = flagged
            n_flagged += int(flagged.sum())

        total_observed = context.total_observed()
        return DetectionResult(
            strategy=self.name,
            detectors=survivors,
            discarded=discarded,
            rates=rates,
            threshold_k=self.threshold_k,
            mask=mask,
            n_flagged=n_flagged,
            detection_rate=n_flagged / total_observed if total_observed else 0.0,
        )


@dataclass(frozen=True)
class InjectionTopKDetection:
    """Detectors selected by injection VUS-PR; masks combined by vote.

    ``top_k=1`` keeps the single best detector's MAD-thresholded mask. With
    ``top_k > 1`` each selected detector's real scores are binarized
    independently (same MAD rule per segment) and a point is flagged when at
    least ``min_votes`` masks agree. When fewer detectors than ``min_votes``
    could score a segment, the vote degrades to unanimity among the available
    ones (so a lone survivor still contributes its detections).
    """

    name: str = STRATEGY_INJECT_BEST
    top_k: int = 1
    min_votes: int = 1
    threshold_k: float = DEFAULT_THRESHOLD_K

    def detect(self, context: SeriesDetectionContext) -> DetectionResult:
        if not context.segments:
            return _empty_result(self.name, context, self.threshold_k)

        ranking = context.selection_ranking()
        if not ranking:
            return _empty_result(self.name, context, self.threshold_k)
        top_models = rank_top_k(ranking, self.top_k)

        # The mask always comes from the REAL series: the injection only ranks.
        scores_by_model = context.real_scores(top_models)
        selected = [name for name in top_models if name in scores_by_model]
        if not selected:
            return _empty_result(self.name, context, self.threshold_k, ranking)
        rates = {
            name: _detector_rate(scores_by_model[name], self.threshold_k) for name in selected
        }

        mask = context.empty_mask()
        n_flagged = 0
        for segment_index, segment in enumerate(context.segments):
            segment_masks = [
                detect_mask(scores_by_model[name][segment_index], self.threshold_k)
                for name in selected
                if scores_by_model[name][segment_index] is not None
            ]
            if not segment_masks:
                continue
            needed = min(self.min_votes, len(segment_masks))
            flagged = np.sum(segment_masks, axis=0) >= needed
            mask.loc[segment.index] = flagged
            n_flagged += int(flagged.sum())

        total_observed = context.total_observed()
        return DetectionResult(
            strategy=self.name,
            detectors=selected,
            discarded=sorted(set(ranking) - set(selected)),
            rates=rates,
            threshold_k=self.threshold_k,
            mask=mask,
            ranking=dict(ranking),
            n_flagged=n_flagged,
            detection_rate=n_flagged / total_observed if total_observed else 0.0,
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
    n_flagged = int(mask.sum())
    total_observed = int(series.notna().sum())
    return replace(
        result,
        mask=mask,
        n_flagged=n_flagged,
        detection_rate=n_flagged / total_observed if total_observed else 0.0,
    )


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
]

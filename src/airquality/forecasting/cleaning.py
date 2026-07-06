"""Detect and remove anomalies from a *real* (unlabeled) air-quality series.

This is the production-facing counterpart of the label-free anomaly benchmark
(:mod:`airquality.anomaly.benchmark`) and applies the exact same method — there
is no ground truth in real time, so no detector can be selected or calibrated
against labels (synthetic injection was deliberately dropped: its anomalies
were not representative of the sensor faults we target).

Strategy (per series):

1. Split the series into **contiguous observed segments** (no ``dropna()``
   gluing: stitching non-contiguous stretches would create artificial phase
   jumps in the daily cycle for windowed detectors).
2. Fit/score every requested detector on each segment and binarize its scores
   with the robust MAD threshold (:func:`airquality.anomaly.metrics.detect_mask`,
   median + k * scaled MAD).
3. **Discard** detectors whose detection rate over the observed points exceeds
   ``max_detection_rate`` (default 7%): the target anomalies are sensor faults
   (spikes, calibration drift, cutouts), which are rare — a higher rate means
   the detector flags normal variation.
4. Build the final mask from the **consensus** of the surviving detectors
   (:func:`airquality.anomaly.ensemble.consensus` over their normalized
   scores), re-thresholded with the same MAD rule per segment.

Flagged timestamps are then set to NaN by :func:`remove_anomalies` so the
imputation step can fill them.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from airquality.anomaly.ensemble import DEFAULT_ENSEMBLE_METHOD, consensus
from airquality.anomaly.metrics import (
    DEFAULT_MAX_DETECTION_RATE,
    DEFAULT_THRESHOLD_K,
    detect_mask,
)
from airquality.anomaly.registry import (
    filter_model_kwargs as _filter_model_kwargs,
    resolve_model_class,
    resolve_model_names,
)
from airquality.data.segments import contiguous_observed_segments
from airquality.data.series import ensure_datetime_series

DEFAULT_SEED = 13

#: Minimum contiguous observed points a segment needs to enter detection (the
#: former whole-series minimum, now applied per segment).
MIN_SEGMENT_POINTS = 8


@dataclass
class CleaningResult:
    """Outcome of label-free anomaly detection on one series."""

    detectors: list[str]  # survivors of the detection-rate filter (used in the consensus)
    discarded: list[str]  # detectors over the detection-rate budget
    rates: dict[str, float]  # per-detector detection rate on this series
    threshold_k: float
    mask: pd.Series  # boolean, indexed like the input series (True = anomaly)
    n_flagged: int = 0
    detection_rate: float = 0.0  # rate of the final consensus mask


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


def detect_anomaly_mask(
    series: pd.Series,
    *,
    detectors: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    freq: str = "h",
    threshold_k: float = DEFAULT_THRESHOLD_K,
    max_detection_rate: float = DEFAULT_MAX_DETECTION_RATE,
) -> CleaningResult:
    """Flag anomalies with the consensus of the detectors that pass the rate filter."""
    s = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
    empty_mask = pd.Series(False, index=s.index, name=s.name)

    # Work per contiguous observed segment: `dropna()` would stitch stretches
    # that are hours or days apart into one "continuous" hourly signal.
    segments = contiguous_observed_segments(s, min_len=MIN_SEGMENT_POINTS)
    if not segments:
        return CleaningResult(
            detectors=[], discarded=[], rates={}, threshold_k=threshold_k, mask=empty_mask
        )

    model_names = resolve_model_names(detectors if detectors else ["all"])
    scores_by_model = _score_segments(model_names, segments, seed=seed, device=device)

    # --- Detection-rate filter: sensor faults are rare, so a detector flagging
    # more than the budget is marking normal variation, not anomalies.
    rates = {
        name: _detector_rate(segment_scores, threshold_k)
        for name, segment_scores in scores_by_model.items()
    }
    survivors = sorted(name for name, rate in rates.items() if rate <= max_detection_rate)
    discarded = sorted(name for name, rate in rates.items() if rate > max_detection_rate)

    # --- Final mask: per segment, fuse the survivors' normalized scores and
    # re-threshold the consensus with the same MAD rule.
    mask = empty_mask.copy()
    n_flagged = 0
    for segment_index, segment in enumerate(segments):
        score_arrays = [
            scores_by_model[name][segment_index]
            for name in survivors
            if scores_by_model[name][segment_index] is not None
        ]
        if not score_arrays:
            continue
        fused = consensus(score_arrays, DEFAULT_ENSEMBLE_METHOD, seed)
        flagged = detect_mask(fused, threshold_k)
        mask.loc[segment.index] = flagged
        n_flagged += int(flagged.sum())

    total_observed = sum(len(segment) for segment in segments)
    return CleaningResult(
        detectors=survivors,
        discarded=discarded,
        rates=rates,
        threshold_k=threshold_k,
        mask=mask,
        n_flagged=n_flagged,
        detection_rate=n_flagged / total_observed if total_observed else 0.0,
    )


def remove_anomalies(series: pd.Series, result: CleaningResult) -> pd.Series:
    """Return a copy of ``series`` with flagged timestamps set to NaN."""
    cleaned = series.copy()
    flagged_index = result.mask.index[result.mask.to_numpy()]
    cleaned.loc[cleaned.index.intersection(flagged_index)] = np.nan
    return cleaned


__all__ = [
    "CleaningResult",
    "detect_anomaly_mask",
    "remove_anomalies",
]

"""Detect and remove anomalies from a *real* (unlabeled) air-quality series.

The benchmark pipeline (:mod:`.pipeline`) injects synthetic anomalies to *score*
detectors with VUS-PR. Here we reuse the same building blocks for a different
goal: cleaning one real series for which there is **no** ground truth.

Strategy (per series, agreed with the user):

1. Split the series into **contiguous observed segments** (no ``dropna()``
   gluing: stitching non-contiguous stretches would create artificial phase
   jumps in the daily cycle for STL and windowed detectors).
2. Inject synthetic anomalies into the longest contiguous segment
   (:func:`.anomalies.inject_synthetic_anomalies`) to obtain reference labels.
3. Fit every registered detector on the injected segment and, for each, pick the
   score threshold that **maximizes F1** against the injected labels.
4. **Select the detector** with the best F1 (one detector per series). The
   injected anomalies are the selection mechanism -- no VUS-PR is computed.
5. Transfer the calibrated threshold in **relative (quantile) terms**: the
   absolute score scale changes between the injected and the original series
   for many detectors, so the optimal threshold is mapped to its quantile in
   the injected-score distribution and that quantile is applied to the scores
   of each original segment.
6. Re-score the winning detector on every contiguous segment of the *original*
   (non-injected) series and flag with the transferred quantile threshold;
   segments shorter than :data:`MIN_SEGMENT_POINTS` get no detection.

Flagged timestamps are then set to NaN by :func:`remove_anomalies` so the
imputation step can fill them.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from airquality.anomaly.anomalies import inject_synthetic_anomalies
from airquality.anomaly.registry import resolve_model_class, resolve_model_names
from airquality.data.segments import contiguous_observed_segments
from airquality.data.series import ensure_datetime_series

DEFAULT_VARIANT = "STL-combined"
DEFAULT_SEED = 13

#: Minimum contiguous observed points a segment needs to enter detection (the
#: former whole-series minimum, now applied per segment).
MIN_SEGMENT_POINTS = 8


@dataclass
class CleaningResult:
    """Outcome of anomaly detection on one series."""

    detector: str
    threshold: float  # absolute F1-optimal threshold on the injected scores
    threshold_quantile: float  # rank of `threshold` in the injected-score distribution
    f1: float
    mask: pd.Series  # boolean, indexed like the input series (True = anomaly)
    n_flagged: int


def _filter_model_kwargs(model_cls: type, kwargs: dict[str, object]) -> dict[str, object]:
    """Keep only kwargs the detector's ``__init__`` accepts (pattern from pipeline.py)."""
    parameters = inspect.signature(model_cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(kwargs)
    valid = set(parameters) - {"self"}
    return {key: value for key, value in kwargs.items() if key in valid}


def best_threshold_f1(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    max_candidates: int = 100,
    min_quantile: float = 0.5,
) -> tuple[float, float]:
    """Return the ``(threshold, f1)`` over ``scores`` that best matches ``labels``.

    Candidate thresholds are a capped grid of score quantiles; a point is flagged
    when ``score >= threshold``. Used to calibrate a detector without real labels
    by reusing the synthetic-injection labels. The search starts at
    ``min_quantile`` (default the median) because anomalies are the high-score
    minority -- this prevents a degenerate threshold that flags the majority of
    the series when a weak detector barely separates the classes.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0 or labels.sum() == 0:
        return float("inf"), 0.0

    quantiles = np.linspace(min_quantile, 1.0, max_candidates)
    candidates = np.unique(np.quantile(finite, quantiles))

    best_threshold = float(candidates[-1])
    best_f1 = -1.0
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        score = f1_score(labels, predicted, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _fit_score(model_cls: type, values: np.ndarray, *, seed: int, device: str) -> np.ndarray:
    """Instantiate, fit and score one detector on ``values``."""
    kwargs = _filter_model_kwargs(model_cls, {"device": device})
    model = model_cls(seed=seed, **kwargs)
    model.fit(values)
    return np.asarray(model.score(values), dtype=float)


def detect_anomaly_mask(
    series: pd.Series,
    *,
    detectors: list[str] | None = None,
    variant: str = DEFAULT_VARIANT,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    freq: str = "h",
) -> CleaningResult:
    """Select the best detector via synthetic injection and flag real anomalies."""
    s = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
    empty_mask = pd.Series(False, index=s.index, name=s.name)

    # Work per contiguous observed segment: `dropna()` would stitch stretches
    # that are hours or days apart into one "continuous" hourly signal.
    segments = contiguous_observed_segments(s, min_len=MIN_SEGMENT_POINTS)
    if not segments:
        return CleaningResult(
            detector="", threshold=float("inf"), threshold_quantile=1.0,
            f1=0.0, mask=empty_mask, n_flagged=0,
        )

    # --- Selection: calibrate every detector on the longest contiguous segment.
    selection_values = max(segments, key=len).to_numpy(dtype=np.float32)
    injected, labels = inject_synthetic_anomalies(selection_values, variant, seed)
    model_names = resolve_model_names(detectors if detectors else ["all"])

    best_name = ""
    best_threshold = float("inf")
    best_f1 = -1.0
    best_scores: np.ndarray | None = None
    for name in model_names:
        try:
            model_cls = resolve_model_class(name)
            scores = _fit_score(model_cls, injected, seed=seed, device=device)
            threshold, f1 = best_threshold_f1(labels, scores)
        except Exception as exc:  # pragma: no cover - detector-specific failures
            logging.warning("Detector %s fallo durante la seleccion: %s", name, exc)
            continue
        if f1 > best_f1:
            best_f1, best_name, best_threshold = f1, name, threshold
            best_scores = scores

    if not best_name:
        return CleaningResult(
            detector="", threshold=float("inf"), threshold_quantile=1.0,
            f1=0.0, mask=empty_mask, n_flagged=0,
        )

    # --- Quantile transfer: the absolute score scale is not comparable between
    # the injected and the original series (z-scores, reconstruction errors...),
    # so the calibrated threshold travels as the rank it occupies among the
    # injected scores and is re-materialized on each original-score distribution.
    finite_selection = (
        best_scores[np.isfinite(best_scores)] if best_scores is not None else np.array([])
    )
    can_transfer = np.isfinite(best_threshold) and finite_selection.size > 0
    threshold_quantile = (
        float(np.mean(finite_selection < best_threshold)) if can_transfer else 1.0
    )

    # --- Application: re-score the winner on every ORIGINAL contiguous segment
    # and flag with the transferred quantile threshold.
    mask = empty_mask.copy()
    n_flagged = 0
    if can_transfer:
        best_cls = resolve_model_class(best_name)
        for segment in segments:
            segment_scores = _fit_score(
                best_cls, segment.to_numpy(dtype=np.float32), seed=seed, device=device
            )
            finite = segment_scores[np.isfinite(segment_scores)]
            if finite.size == 0:
                continue
            segment_threshold = float(np.quantile(finite, threshold_quantile))
            flagged = segment_scores >= segment_threshold
            mask.loc[segment.index] = flagged
            n_flagged += int(flagged.sum())

    return CleaningResult(
        detector=best_name,
        threshold=best_threshold,
        threshold_quantile=threshold_quantile,
        f1=best_f1,
        mask=mask,
        n_flagged=n_flagged,
    )


def remove_anomalies(series: pd.Series, result: CleaningResult) -> pd.Series:
    """Return a copy of ``series`` with flagged timestamps set to NaN."""
    cleaned = series.copy()
    flagged_index = result.mask.index[result.mask.to_numpy()]
    cleaned.loc[cleaned.index.intersection(flagged_index)] = np.nan
    return cleaned


__all__ = [
    "CleaningResult",
    "best_threshold_f1",
    "detect_anomaly_mask",
    "remove_anomalies",
]

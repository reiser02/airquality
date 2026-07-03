"""Label-free scoring utilities for the anomaly benchmark.

There is no ground truth for real air-quality series (the detectors will run
in real time), so no supervised metric (AUROC/VUS-PR/...) can be computed.
Instead, detector scores are binarized with a robust threshold on their own
distribution (median + k * scaled MAD, the Iglewicz-Hoaglin modified z-score
rule for k=3.5) and detectors are judged by their **detection rate**: the
fraction of points they flag. Detectors flagging more than a small budget
(default 7%) are discarded — sensor errors (spikes, calibration drift,
cutouts) are rare, so a high rate means the detector is flagging normal
variation, not faults.
"""

from __future__ import annotations

import numpy as np

#: 1.4826 * MAD estimates the standard deviation under normality, so
#: ``median + 3.5 * 1.4826 * MAD`` matches the Iglewicz-Hoaglin modified
#: z-score cutoff of 3.5.
MAD_SCALE = 1.4826
DEFAULT_THRESHOLD_K = 3.5

#: Maximum tolerated fraction of flagged points. Sensor faults are rare;
#: a detector above this budget is flagging normal variation.
DEFAULT_MAX_DETECTION_RATE = 0.07


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize ``scores`` to ``[0, 1]`` (all zeros when constant)."""
    minimum = float(np.min(scores))
    maximum = float(np.max(scores))
    if maximum <= minimum:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - minimum) / (maximum - minimum)


def mad_threshold(scores: np.ndarray, k: float = DEFAULT_THRESHOLD_K) -> float:
    """Robust score threshold: ``median + k * 1.4826 * MAD`` over finite scores.

    Degenerates gracefully: with MAD = 0 (more than half the scores identical,
    e.g. Hampel scoring 0 on every non-outlier) the threshold is the median, so
    only scores strictly above the majority value are flagged; an all-constant
    score array then flags nothing. Returns ``inf`` when no score is finite.
    """
    scores = np.asarray(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float("inf")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median + float(k) * MAD_SCALE * mad


def detect_mask(scores: np.ndarray, k: float = DEFAULT_THRESHOLD_K) -> np.ndarray:
    """Binarize ``scores`` with the MAD threshold (strict ``>``; NaN never flagged)."""
    scores = np.asarray(scores, dtype=np.float64)
    threshold = mad_threshold(scores, k)
    with np.errstate(invalid="ignore"):
        return np.isfinite(scores) & (scores > threshold)


def detection_rate(mask: np.ndarray) -> float:
    """Fraction of flagged points in a boolean ``mask`` (0.0 when empty)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0.0
    return float(mask.mean())

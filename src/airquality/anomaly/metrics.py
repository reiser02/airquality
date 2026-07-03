"""Scoring metrics for both anomaly-benchmark modes.

**Label-free** (``unlabeled`` mode, also used by the production cleaning):
there is no ground truth for real air-quality series, so detector scores are
binarized with a robust threshold on their own distribution (median + k *
scaled MAD, the Iglewicz-Hoaglin modified z-score rule for k=3.5) and
detectors are judged by their **detection rate**: the fraction of points they
flag. Detectors flagging more than a small budget (default 7%) are discarded —
sensor errors (spikes, calibration drift, cutouts) are rare, so a high rate
means the detector is flagging normal variation, not faults.

**Supervised** (``synthetic`` mode): :func:`compute_metrics` scores detectors
against injected-anomaly labels with the genias metric set — AUROC, AUPR,
VUS-PR/VUS-ROC (window-tolerant volumes, via the vendored ``vus_volume``) and
affiliation F1 (event-based, via the vendored ``affiliation`` package).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ._vendor.affiliation.generics import convert_vector_to_events
from ._vendor.affiliation.metrics import pr_from_events
from ._vendor.vus_volume import vus_roc_pr

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


def compute_metrics(labels: np.ndarray, scores: np.ndarray, window_size: int) -> dict[str, float]:
    """Compute the supervised metric set for one scored series (synthetic mode).

    ``window_size`` is the sliding-window tolerance used by VUS; scores are
    min-max normalized first. Returns all-zero metrics when there are no
    positive labels (metrics would be undefined).
    """
    if len(labels) == 0 or np.sum(labels) == 0:
        return {
            "auroc": 0.0,
            "aupr": 0.0,
            "vus_pr": 0.0,
            "vus_roc": 0.0,
            "affiliation_f1": 0.0,
        }

    labels = labels.astype(np.int64)
    score = normalize_scores(scores.astype(np.float64))
    sliding_window = max(1, int(window_size))

    auroc = float(roc_auc_score(labels, score))
    aupr = float(average_precision_score(labels, score))
    vus_roc, vus_pr = vus_roc_pr(labels, score, sliding_window)

    # Affiliation precision/recall over events thresholded at 0.5 (matches
    # vus.metrics.get_metrics).
    discrete_score = np.array(score > 0.5, dtype=np.float32)
    events_pred = convert_vector_to_events(discrete_score)
    events_gt = convert_vector_to_events(labels)
    affiliation = pr_from_events(events_pred, events_gt, (0, len(discrete_score)))
    affiliation_precision = float(affiliation["Affiliation_Precision"])
    affiliation_recall = float(affiliation["Affiliation_Recall"])

    affiliation_f1 = 0.0
    if affiliation_precision + affiliation_recall > 0.0:
        affiliation_f1 = (2.0 * affiliation_precision * affiliation_recall) / (affiliation_precision + affiliation_recall)

    return {
        "auroc": auroc,
        "aupr": aupr,
        "vus_pr": float(vus_pr),
        "vus_roc": float(vus_roc),
        "affiliation_f1": float(affiliation_f1),
    }

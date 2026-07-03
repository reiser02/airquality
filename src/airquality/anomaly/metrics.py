"""Scoring metrics for the anomaly benchmark (genias metric set).

Computes the five metrics reported per detector and case: AUROC, AUPR,
VUS-PR/VUS-ROC (window-tolerant volumes, via the vendored ``vus_volume``) and
affiliation F1 (event-based, via the vendored ``affiliation`` package).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ._vendor.affiliation.generics import convert_vector_to_events
from ._vendor.affiliation.metrics import pr_from_events
from ._vendor.vus_volume import vus_roc_pr


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize ``scores`` to ``[0, 1]`` (all zeros when constant)."""
    minimum = float(np.min(scores))
    maximum = float(np.max(scores))
    if maximum <= minimum:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - minimum) / (maximum - minimum)


def compute_metrics(labels: np.ndarray, scores: np.ndarray, window_size: int) -> dict[str, float]:
    """Compute the benchmark metric set for one scored series.

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

"""Scoring metrics for both anomaly-benchmark modes.

**Label-free** (``unlabeled`` mode, also used by the production cleaning):
there is no ground truth for real air-quality series, so detector scores are
binarized with a robust threshold on their own distribution (median + k *
scaled MAD, the Iglewicz-Hoaglin modified z-score rule for k=3.5) and
detectors are judged by their **detection rate**: the fraction of points they
flag. Detectors flagging more than a small budget (default 7%) are discarded —
sensor errors (spikes, calibration drift, cutouts) are rare, so a high rate
means the detector is flagging normal variation, not faults.

**Supervised** (``synthetic`` mode): :func:`compute_segmented_metrics` pools
disjoint finite-score components by station without crossing temporal gaps.
It reports AUROC, AUPR, VUS-PR/VUS-ROC (via the vendored ``vus_volume``) and
affiliation F1 (via the vendored ``affiliation`` package).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ._vendor.affiliation.generics import convert_vector_to_events
from ._vendor.affiliation.metrics import pr_from_events
from ._vendor.vus_volume import vus_roc_pr, vus_roc_pr_segments

#: 1.4826 * MAD estimates the standard deviation under normality, so
#: ``median + 3.5 * 1.4826 * MAD`` matches the Iglewicz-Hoaglin modified
#: z-score cutoff of 3.5.
MAD_SCALE = 1.4826
DEFAULT_THRESHOLD_K = 3.5

#: Maximum tolerated fraction of flagged points. Sensor faults are rare;
#: a detector above this budget is flagging normal variation.
DEFAULT_MAX_DETECTION_RATE = 0.07


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize finite ``scores``; non-finite values map to zero."""
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if not finite.any():
        return np.zeros_like(scores, dtype=np.float64)
    minimum = float(np.min(scores[finite]))
    maximum = float(np.max(scores[finite]))
    if maximum <= minimum:
        return np.zeros_like(scores, dtype=np.float64)
    normalized = np.zeros_like(scores, dtype=np.float64)
    normalized[finite] = (scores[finite] - minimum) / (maximum - minimum)
    return normalized


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


def vus_sliding_window(labels: np.ndarray) -> int:
    """VUS sliding-window tolerance: the median labeled anomaly length.

    Follows the original VUS convention (thedatumorg/VUS README:
    ``slidingWindow = int(np.median(get_list_anomaly(labels)))``): the window
    is a property of the labels, computed once and shared by every detector so
    their VUS values are comparable. Returns 1 when there are no anomalies.
    """
    flat = np.asarray(labels, dtype=np.int8).ravel()
    boundaries = np.diff(np.concatenate(([0], (flat != 0).astype(np.int8), [0])))
    lengths = np.flatnonzero(boundaries == -1) - np.flatnonzero(boundaries == 1)
    if lengths.size == 0:
        return 1
    return max(1, int(np.median(lengths)))


def vus_sliding_window_segments(labels_by_segment: list[np.ndarray]) -> int:
    """Return the median anomaly length without joining segment boundaries."""
    lengths: list[int] = []
    for labels in labels_by_segment:
        flat = np.asarray(labels, dtype=np.int8).ravel()
        boundaries = np.diff(
            np.concatenate(([0], (flat != 0).astype(np.int8), [0]))
        )
        lengths.extend(
            (
                np.flatnonzero(boundaries == -1)
                - np.flatnonzero(boundaries == 1)
            ).tolist()
        )
    return max(1, int(np.median(lengths))) if lengths else 1


def _finite_components(
    labels_by_segment: list[np.ndarray],
    scores_by_segment: list[np.ndarray],
    prediction_masks_by_segment: list[np.ndarray] | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray] | None]:
    """Split detector support into contiguous components without crossing gaps."""
    if len(labels_by_segment) != len(scores_by_segment):
        raise ValueError("Segment labels and scores must have matching lists")
    if prediction_masks_by_segment is not None and len(labels_by_segment) != len(
        prediction_masks_by_segment
    ):
        raise ValueError("Segment labels and prediction masks must have matching lists")

    component_labels: list[np.ndarray] = []
    component_scores: list[np.ndarray] = []
    component_masks: list[np.ndarray] | None = (
        [] if prediction_masks_by_segment is not None else None
    )
    masks = prediction_masks_by_segment or [None] * len(labels_by_segment)
    for labels, scores, prediction_mask in zip(
        labels_by_segment, scores_by_segment, masks, strict=True
    ):
        labels = np.asarray(labels, dtype=np.int64).ravel()
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if labels.shape != scores.shape:
            raise ValueError("Segment labels and scores must have matching shapes")
        if prediction_mask is not None:
            prediction_mask = np.asarray(prediction_mask, dtype=bool).ravel()
            if prediction_mask.shape != labels.shape:
                raise ValueError("Segment prediction masks must match labels")
        finite = np.isfinite(scores)
        boundaries = np.diff(
            np.concatenate(([0], finite.astype(np.int8), [0]))
        )
        for start, end in zip(
            np.flatnonzero(boundaries == 1),
            np.flatnonzero(boundaries == -1),
            strict=True,
        ):
            component_labels.append(labels[start:end])
            component_scores.append(scores[start:end])
            if component_masks is not None:
                component_masks.append(prediction_mask[start:end])
    return component_labels, component_scores, component_masks


def _affiliation_metrics(
    labels_by_component: list[np.ndarray],
    masks_by_component: list[np.ndarray],
    eligible_has_events: bool,
) -> tuple[float, dict[str, object]]:
    """Aggregate standard affiliation contributions without crossing components."""
    precision_values: list[float] = []
    recall_values: list[float] = []
    ground_truth_events = 0
    orphan_prediction_events = 0
    orphan_prediction_points = 0

    for labels, mask in zip(labels_by_component, masks_by_component, strict=True):
        events_gt = convert_vector_to_events(labels)
        events_pred = convert_vector_to_events(mask.astype(np.float32))
        if not events_gt:
            orphan_prediction_events += len(events_pred)
            orphan_prediction_points += int(mask.sum())
            continue
        ground_truth_events += len(events_gt)
        affiliation = pr_from_events(events_pred, events_gt, (0, len(labels)))
        precision_values.extend(
            float(value)
            for value in affiliation["individual_precision_probabilities"]
        )
        recall_values.extend(
            float(value) for value in affiliation["individual_recall_probabilities"]
        )

    if not eligible_has_events:
        status = "no_ground_truth"
        affiliation_f1 = float("nan")
    elif orphan_prediction_events:
        status = "orphan_predictions"
        affiliation_f1 = float("nan")
    elif not ground_truth_events:
        status = "no_supported_ground_truth"
        affiliation_f1 = float("nan")
    else:
        finite_precision = [value for value in precision_values if np.isfinite(value)]
        precision = (
            float(np.mean(finite_precision)) if finite_precision else float("nan")
        )
        recall = float(np.mean(recall_values)) if recall_values else float("nan")
        if np.isfinite(precision) and np.isfinite(recall):
            affiliation_f1 = (
                0.0
                if precision + recall == 0.0
                else 2.0 * precision * recall / (precision + recall)
            )
            status = "defined"
        else:
            affiliation_f1 = float("nan")
            status = "no_affiliable_predictions"

    return float(affiliation_f1), {
        "status": status,
        "ground_truth_events": ground_truth_events,
        "precision_contributions": int(
            sum(np.isfinite(value) for value in precision_values)
        ),
        "orphan_prediction_events": orphan_prediction_events,
        "orphan_prediction_points": orphan_prediction_points,
    }


def compute_segmented_metrics(
    labels_by_segment: list[np.ndarray],
    scores_by_segment: list[np.ndarray],
    window_size: int,
    prediction_masks_by_segment: list[np.ndarray] | None = None,
) -> dict[str, object]:
    """Compute station metrics over disjoint finite-score components."""
    labels = [np.asarray(values, dtype=np.int64).ravel() for values in labels_by_segment]
    eligible = np.concatenate(labels) if labels else np.array([], dtype=np.int64)
    eligible_has_both = bool(
        eligible.size and np.any(eligible == 0) and np.any(eligible != 0)
    )
    eligible_has_events = bool(np.any(eligible != 0))
    component_labels, component_scores, component_masks = _finite_components(
        labels, scores_by_segment, prediction_masks_by_segment
    )

    metrics = {
        "auroc": float("nan"),
        "aupr": float("nan"),
        "vus_pr": float("nan"),
        "vus_roc": float("nan"),
        "affiliation_f1": float("nan"),
    }
    if component_scores:
        lengths = [len(scores) for scores in component_scores]
        normalized = normalize_scores(np.concatenate(component_scores))
        normalized_components = list(
            np.split(normalized, np.cumsum(lengths)[:-1])
        )
        scored_labels = np.concatenate(component_labels)
        scored_has_both = bool(
            np.any(scored_labels == 0) and np.any(scored_labels != 0)
        )
        if eligible_has_both and scored_has_both:
            metrics["auroc"] = float(roc_auc_score(scored_labels, normalized))
            metrics["aupr"] = float(
                average_precision_score(scored_labels, normalized)
            )
            vus_roc, vus_pr = vus_roc_pr_segments(
                component_labels,
                normalized_components,
                max(1, int(window_size)),
            )
            metrics["vus_roc"] = float(vus_roc)
            metrics["vus_pr"] = float(vus_pr)
        if component_masks is None:
            component_masks = [scores > 0.5 for scores in normalized_components]
        metrics["affiliation_f1"], affiliation_diagnostics = _affiliation_metrics(
            component_labels, component_masks, eligible_has_events
        )
    else:
        affiliation_diagnostics = {
            "status": "no_supported_ground_truth" if eligible_has_events else "no_ground_truth",
            "ground_truth_events": 0,
            "precision_contributions": 0,
            "orphan_prediction_events": 0,
            "orphan_prediction_points": 0,
        }

    return {
        "metrics": metrics,
        "defined_by_labels": {
            "auroc": eligible_has_both,
            "aupr": eligible_has_both,
            "vus_pr": eligible_has_both,
            "vus_roc": eligible_has_both,
            "affiliation_f1": eligible_has_events,
        },
        "affiliation_diagnostics": affiliation_diagnostics,
    }


def compute_metrics(labels: np.ndarray, scores: np.ndarray, window_size: int) -> dict[str, float]:
    """Compute supervised metrics for one contiguous score series."""
    return compute_segmented_metrics([labels], [scores], window_size)["metrics"]

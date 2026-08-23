"""Detector selection and majority-vote ensemble for the anomaly pipeline.

In ``unlabeled`` mode there is no label-based metric to rank detectors with,
so the ensemble combines every detector that survives the detection-rate filter
(see :func:`airquality.anomaly.benchmark.split_by_detection_rate`). In
``synthetic`` mode detectors are ranked by selection-injection VUS-PR: long
segments use local rankings and short segments inherit the station ranking. At
each point, the first three ranked detectors with a finite score vote; later
detectors backfill missing scores and at least two votes are required.
"""

from __future__ import annotations

import numpy as np

from .metrics import DEFAULT_THRESHOLD_K, detect_mask

DEFAULT_TOP_K = 3


def rank_top_k(metric_by_model: dict[str, float | None], k: int = DEFAULT_TOP_K) -> list[str]:
    """Return up to ``k`` models with a finite metric (ties broken by name)."""
    finite_metrics = []
    for name, metric in metric_by_model.items():
        try:
            value = float(metric)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            finite_metrics.append((name, value))
    ordered = sorted(
        finite_metrics,
        key=lambda item: (-item[1], item[0]),
    )
    return [name for name, _ in ordered[:k]]


def ranked_pointwise_vote(
    scores_by_model: dict[str, np.ndarray],
    ranking: list[str],
    *,
    top_k: int = DEFAULT_TOP_K,
    threshold_k: float = DEFAULT_THRESHOLD_K,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Vote pointwise with ranked backfill; return mask, support, and used models."""
    if top_k not in (2, 3):
        raise ValueError("top_k must be 2 or 3")
    if not ranking:
        return np.array([], dtype=np.float32), np.array([], dtype=bool), []

    arrays = {name: np.asarray(scores_by_model[name], dtype=np.float64) for name in ranking}
    lengths = {scores.shape for scores in arrays.values()}
    if len(lengths) != 1 or any(scores.ndim != 1 for scores in arrays.values()):
        raise ValueError("ranked detector scores must be matching one-dimensional arrays")

    masks = {name: detect_mask(scores, threshold_k) for name, scores in arrays.items()}
    length = next(iter(arrays.values())).size
    fused = np.zeros(length, dtype=np.float32)
    supported = np.zeros(length, dtype=bool)
    used: set[str] = set()

    for point in range(length):
        selected = [
            name for name in ranking if np.isfinite(arrays[name][point])
        ][:top_k]
        if len(selected) < 2:
            continue
        used.update(selected)
        supported[point] = True
        fused[point] = float(sum(bool(masks[name][point]) for name in selected) >= 2)

    return fused, supported, [name for name in ranking if name in used]


def consensus(
    score_arrays: list[np.ndarray],
    threshold_k: float = DEFAULT_THRESHOLD_K,
) -> np.ndarray:
    """Binarize each score array with MAD and return the strict-majority mask."""
    if not score_arrays:
        raise ValueError("consensus requires at least one score array")
    masks = np.column_stack([detect_mask(scores, threshold_k) for scores in score_arrays])
    needed = masks.shape[1] // 2 + 1
    return (masks.sum(axis=1) >= needed).astype(np.float32)

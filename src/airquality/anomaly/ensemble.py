"""Detector selection and majority-vote ensemble for the anomaly pipeline.

In ``unlabeled`` mode there is no label-based metric to rank detectors with,
so the ensemble combines every detector that survives the detection-rate filter
(see :func:`airquality.anomaly.benchmark.split_by_detection_rate`). In
``synthetic`` mode detectors are ranked by selection-injection VUS-PR: long
segments use local rankings and short segments inherit the station mean. The
available top-k (default 3) vote per segment, with two required votes by
default, matching forecasting's ``inject-vote`` strategy.
"""

from __future__ import annotations

import numpy as np

from .metrics import DEFAULT_THRESHOLD_K, detect_mask

DEFAULT_TOP_K = 3


def rank_top_k(metric_by_model: dict[str, float], k: int = DEFAULT_TOP_K) -> list[str]:
    """Return the ``k`` model names with the highest metric (ties broken by name)."""
    ordered = sorted(metric_by_model.items(), key=lambda item: (-item[1], item[0]))
    return [name for name, _ in ordered[:k]]


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

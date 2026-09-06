"""Detector selection and majority-vote ensemble for the anomaly pipeline.

In ``unlabeled`` mode there is no label-based metric to rank detectors with,
so :class:`airquality.forecasting.detection.ConsensusDetection` combines every
detector that survives the detection-rate filter. In ``synthetic`` mode detectors
are ranked by selection-injection VUS-PR: long segments use local rankings and
short segments inherit the mean for their TSPulse context regime. At
each point, the first ``top_k`` ranked detectors with a finite score vote; later
detectors backfill missing scores and at least ``min_votes`` votes are required.
"""

from __future__ import annotations

import numpy as np

from .metrics import DEFAULT_THRESHOLD_K, detect_mask

DEFAULT_TOP_K = 3
NATIVE_CONTEXT_LENGTH = 512


def uses_native_context(length: int) -> bool:
    """Whether TSPulse can score the block without padding."""
    return length >= NATIVE_CONTEXT_LENGTH


def ranking_source_indices(
    segment_lengths: list[int] | tuple[int, ...], minimum_points: int
) -> list[int]:
    """Select local-ranking blocks independently for padded and native contexts."""
    selected: list[int] = []
    for native in (False, True):
        regime = [
            index
            for index, length in enumerate(segment_lengths)
            if uses_native_context(length) == native
        ]
        if not regime:
            continue
        eligible = [index for index in regime if segment_lengths[index] >= minimum_points]
        selected.extend(
            eligible
            or [max(regime, key=lambda index: segment_lengths[index])]
        )
    return sorted(selected)


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
    min_votes: int = 2,
    threshold_k: float = DEFAULT_THRESHOLD_K,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Vote pointwise with ranked backfill and a configurable quorum."""
    if top_k < 1 or min_votes < 1 or min_votes > top_k:
        raise ValueError("min_votes must be between 1 and top_k")
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
        if len(selected) < min_votes:
            continue
        used.update(selected)
        supported[point] = True
        fused[point] = float(
            sum(bool(masks[name][point]) for name in selected) >= min_votes
        )

    return fused, supported, [name for name in ranking if name in used]

"""Detector selection and score fusion for the anomaly pipeline.

In ``unlabeled`` mode there is no label-based metric to rank detectors with,
so :class:`airquality.forecasting.detection.ConsensusDetection` combines every
detector that survives the detection-rate filter. In ``synthetic`` mode detectors
are ranked by selection-injection VUS-PR: long segments use local rankings and
short segments inherit the mean for their TSPulse context regime. The synthetic
benchmark min-max normalizes each detector over the station, then averages the
first ``top_k`` finite scores pointwise. Forecasting retains the original hard
vote as a separate strategy. Both methods backfill unavailable ranked detectors.
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


def ranked_pointwise_minmax_mean(
    scores_by_model: dict[str, list[np.ndarray | None]],
    rankings_by_segment: list[dict[str, float | None]],
    segment_lengths: tuple[int, ...] | list[int],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_scores: int = 2,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]]]:
    """Average station-normalized ranked scores with pointwise backfill.

    Each detector is min-max normalized over all of its finite scores in the
    station. Temporal segments remain separate for local ranking and support;
    non-finite values remain unavailable rather than becoming zero scores.
    """
    lengths = tuple(int(length) for length in segment_lengths)
    if top_k < 1 or min_scores < 1 or min_scores > top_k:
        raise ValueError("min_scores must be between 1 and top_k")
    if len(rankings_by_segment) != len(lengths):
        raise ValueError("rankings and segment lengths must have matching lists")

    normalized_by_model: dict[str, list[np.ndarray | None]] = {}
    for name, score_parts in scores_by_model.items():
        if len(score_parts) != len(lengths):
            raise ValueError("detector scores and segment lengths must have matching lists")
        arrays: list[np.ndarray | None] = []
        finite_parts: list[np.ndarray] = []
        for scores, length in zip(score_parts, lengths, strict=True):
            if scores is None:
                arrays.append(None)
                continue
            array = np.asarray(scores, dtype=np.float64)
            if array.ndim != 1 or array.size != length:
                raise ValueError(
                    "ranked detector scores must match their one-dimensional segments"
                )
            arrays.append(array)
            finite_parts.append(array[np.isfinite(array)])

        available_parts = [part for part in finite_parts if part.size]
        finite_values = (
            np.concatenate(available_parts)
            if available_parts
            else np.array([], dtype=np.float64)
        )
        minimum = float(np.min(finite_values)) if finite_values.size else 0.0
        maximum = float(np.max(finite_values)) if finite_values.size else 0.0
        normalized_parts: list[np.ndarray | None] = []
        for array in arrays:
            if array is None:
                normalized_parts.append(None)
                continue
            finite = np.isfinite(array)
            normalized = np.full(array.shape, np.nan, dtype=np.float64)
            if maximum > minimum:
                normalized[finite] = (array[finite] - minimum) / (maximum - minimum)
            else:
                normalized[finite] = 0.0
            normalized_parts.append(normalized)
        normalized_by_model[name] = normalized_parts

    fused_by_segment: list[np.ndarray] = []
    support_by_segment: list[np.ndarray] = []
    used_by_segment: list[list[str]] = []
    for segment_index, (ranking, length) in enumerate(
        zip(rankings_by_segment, lengths, strict=True)
    ):
        ordered = rank_top_k(ranking, len(ranking))
        fused = np.full(length, np.nan, dtype=np.float64)
        supported = np.zeros(length, dtype=bool)
        used: set[str] = set()
        for point in range(length):
            selected = [
                name
                for name in ordered
                if name in normalized_by_model
                and normalized_by_model[name][segment_index] is not None
                and np.isfinite(normalized_by_model[name][segment_index][point])
            ][:top_k]
            if len(selected) < min_scores:
                continue
            used.update(selected)
            supported[point] = True
            fused[point] = float(
                np.mean(
                    [
                        normalized_by_model[name][segment_index][point]
                        for name in selected
                    ]
                )
            )
        fused_by_segment.append(fused)
        support_by_segment.append(supported)
        used_by_segment.append([name for name in ordered if name in used])

    return fused_by_segment, support_by_segment, used_by_segment

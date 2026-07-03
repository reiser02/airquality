"""Ensemble selection and score-consensus for the anomaly pipeline.

Per evaluation case we rank detectors by their (label-based) VUS-PR, keep the
top-k (default 3), and fuse their per-timestep scores into one consensus score.

Consensus methods (standard outlier-ensemble score combiners — Aggarwal & Sathe;
PyOD; LSCP):

- ``AVG``  — mean of min-max-normalized scores. Lowest variance, most robust;
  the default.
- ``MAX``  — per-timestep maximum of normalized scores. Reduces bias, favours
  recall (flags a point if any detector is confident).
- ``AOM``  — average-of-maximum: split detectors into random buckets, take the
  max per bucket, average the buckets. Balances bias/variance (with only 3
  detectors it degenerates towards ``MAX``).
"""

from __future__ import annotations

import numpy as np

from .metrics import normalize_scores

ENSEMBLE_METHODS = ("AVG", "MAX", "AOM")
DEFAULT_ENSEMBLE_METHOD = "AVG"
DEFAULT_TOP_K = 3


def rank_top_k(metric_by_model: dict[str, float], k: int = DEFAULT_TOP_K) -> list[str]:
    """Return the ``k`` model names with the highest metric (ties broken by name)."""
    ordered = sorted(metric_by_model.items(), key=lambda item: (-item[1], item[0]))
    return [name for name, _ in ordered[:k]]


def consensus(
    score_arrays: list[np.ndarray],
    method: str = DEFAULT_ENSEMBLE_METHOD,
    seed: int = 13,
    weights: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Fuse per-timestep score arrays into one consensus score in ``[0, 1]``-ish range.

    ``weights`` (e.g. per-model VUS-PR) are used by ``AVG`` to compute a
    weighted mean instead of a simple mean.  Ignored by ``MAX`` and ``AOM``.
    """
    if not score_arrays:
        raise ValueError("consensus requires at least one score array")
    method = method.upper()
    if method not in ENSEMBLE_METHODS:
        raise ValueError(f"Unknown ensemble method '{method}'. Use one of {ENSEMBLE_METHODS}.")

    normalized = np.column_stack([normalize_scores(np.asarray(scores, dtype=np.float64)) for scores in score_arrays])
    if method == "AVG":
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            w = w / w.sum()
            return (normalized * w).sum(axis=1).astype(np.float32)
        return normalized.mean(axis=1).astype(np.float32)
    if method == "MAX":
        return normalized.max(axis=1).astype(np.float32)

    # AOM: average over the max of random detector buckets.
    rng = np.random.default_rng(seed)
    n_detectors = normalized.shape[1]
    bucket_size = min(5, n_detectors)
    buckets = []
    for _ in range(20):
        indices = rng.choice(n_detectors, size=bucket_size, replace=False)
        buckets.append(normalized[:, indices].max(axis=1))
    return np.mean(buckets, axis=0).astype(np.float32)

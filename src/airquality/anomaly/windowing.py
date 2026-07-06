"""Windowing helper shared by the windowed detectors.

The windowed detectors (``PCADetector``, ``CARLA*``) score one value per sliding
window and then need to fold those per-window scores back onto the per-timestep
timeline. ``aggregate_window_scores`` does that averaging. It was the only helper
the kept detectors used from the original genias ``data.py`` (the rest was
TSB-AD-specific loading, dropped here). The tail-weighted variant lives in
``models/common.py`` (``aggregate_tail_scores``).
"""

from __future__ import annotations

import numpy as np


def aggregate_window_scores(window_scores: np.ndarray, series_length: int, window_size: int) -> np.ndarray:
    """Average overlapping per-window scores back onto the per-timestep timeline.

    Timestep ``i`` averages the scores of every window covering it, i.e.
    window starts ``[max(0, i - window_size + 1), min(i, n_windows - 1)]``.
    Computed with a float64 prefix sum (O(n) instead of O(n * window_size));
    positions covered by no window keep a score of 0 like the previous loop.
    """
    scores = np.asarray(window_scores, dtype=np.float64)
    n_windows = scores.shape[0]
    if n_windows == 0:
        return np.zeros(series_length, dtype=np.float32)

    prefix = np.concatenate(([0.0], np.cumsum(scores)))
    positions = np.arange(series_length)
    lo = np.maximum(positions - window_size + 1, 0)
    hi = np.minimum(positions, n_windows - 1)
    covered = hi >= lo
    totals = np.where(covered, prefix[np.maximum(hi, lo) + 1] - prefix[lo], 0.0)
    counts = np.where(covered, hi - lo + 1, 1.0)
    return (totals / counts).astype(np.float32)

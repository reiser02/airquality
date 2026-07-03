"""Windowing helper shared by the windowed detectors.

The windowed detectors (``PCADetector``, ``CARLA*``) score one value per sliding
window and then need to fold those per-window scores back onto the per-timestep
timeline. ``aggregate_window_scores`` does that averaging. It was the only helper
the kept detectors used from the original genias ``data.py`` (the rest was
TSB-AD-specific loading, dropped here). Stride-aware variants live in
``models/common.py`` (``aggregate_strided_scores``/``aggregate_tail_scores``).
"""

from __future__ import annotations

import numpy as np


def aggregate_window_scores(window_scores: np.ndarray, series_length: int, window_size: int) -> np.ndarray:
    """Average overlapping per-window scores back onto the per-timestep timeline."""
    totals = np.zeros(series_length, dtype=np.float64)
    counts = np.zeros(series_length, dtype=np.float64)
    for start in range(window_scores.shape[0]):
        end = start + window_size
        totals[start:end] += window_scores[start]
        counts[start:end] += 1.0
    counts[counts == 0.0] = 1.0
    return (totals / counts).astype(np.float32)

"""Synthetic anomaly injection for building labelled evaluation cases.

The benchmark uses a single variant, ``STL-combined``:

- ``base = STL``: a synthetic look-alike built from a **real STL decomposition**
  of the series (:func:`synthetic_base`) — trend + seasonal kept, residual
  replaced by resampled noise, so the daily (24h) seasonality is preserved.
- ``type = combined``: a random shape drawn per injected segment from
  :data:`STL_ANOMALY_TYPES` (``spikes``/``scale``/``noise``/``cutoff``/
  ``contextual``/``speedup``), so one series gets a mix of anomaly shapes.
"""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.seasonal import STL

# Hourly air-quality data -> daily seasonality for the STL decomposition.
STL_PERIOD = 24

STL_ANOMALY_TYPES = ["spikes", "scale", "noise", "cutoff", "contextual", "speedup"]


def synthetic_base(values: np.ndarray, seed: int, period: int = STL_PERIOD) -> np.ndarray:
    """Real STL-based synthetic look-alike: keep trend+seasonal, resample residual.

    Falls back to a moving-average trend + gaussian noise when the series is too
    short for an STL decomposition (``len < 2*period + 1``).
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 2 * period + 1:
        width = max(5, min(101, (n // 20) | 1))
        kernel = np.ones(width, dtype=np.float32) / width
        trend = np.convolve(values, kernel, mode="same")
        residual = values - trend
        noise = rng.normal(float(residual.mean()), float(residual.std() or 1.0), size=n)
        return (trend + noise).astype(np.float32)

    decomposition = STL(np.asarray(values, dtype=float), period=period, robust=True).fit()
    resid_std = float(np.std(decomposition.resid) or 1.0)
    synthetic_residual = rng.normal(0.0, resid_std, size=n)
    return (decomposition.trend + decomposition.seasonal + synthetic_residual).astype(np.float32)


def apply_stl_anomaly_segment(
    synthetic: np.ndarray,
    values: np.ndarray,
    start: int,
    end: int,
    anomaly_type: str,
    rng: np.random.Generator,
    scale: float,
) -> None:
    """Mutate ``synthetic[start:end]`` in place with one anomaly shape.

    ``anomaly_type`` is one of :data:`STL_ANOMALY_TYPES`; ``scale`` is the
    series' standard deviation, used to size the perturbation.
    """
    length = end - start
    if anomaly_type == "spikes":
        synthetic[start:end] += rng.choice([-1.0, 1.0]) * scale * 4.0
    elif anomaly_type == "scale":
        synthetic[start:end] *= rng.choice([0.25, 2.0])
    elif anomaly_type == "noise":
        synthetic[start:end] += rng.normal(0.0, scale * 2.0, size=length)
    elif anomaly_type == "cutoff":
        synthetic[start:end] = float(np.quantile(values, 0.75))
    elif anomaly_type == "contextual":
        synthetic[start:end] = synthetic[start:end][::-1] + rng.choice([-1.0, 1.0]) * scale
    elif anomaly_type == "speedup":
        segment = synthetic[start:end]
        compressed = segment[::2]
        if compressed.size:
            synthetic[start:end] = np.interp(
                np.linspace(0, compressed.size - 1, length), np.arange(compressed.size), compressed
            )
    else:
        raise ValueError(f"Unknown synthetic anomaly type: {anomaly_type}")


def inject_synthetic_anomalies(values: np.ndarray, variant: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Inject anomalies and return ``(series_with_anomalies, labels)``.

    Segment count scales with series length (``len // 300``, at least one); span
    length is type-aware (``spikes`` is a single point, the rest 2..32 points).
    """
    base_name, anomaly_type = variant.split("-", maxsplit=1) if "-" in variant else ("raw", variant)
    synthetic = synthetic_base(values, seed) if base_name == "STL" else values.astype(np.float32, copy=True)
    labels = np.zeros(len(values), dtype=np.int64)
    if len(values) < 8:
        return synthetic, labels

    rng = np.random.default_rng(seed)
    scale = float(np.std(values) or 1.0)
    count = max(1, len(values) // 300)
    max_len = max(4, min(32, len(values) // 20))
    for _ in range(count):
        segment_type = str(rng.choice(STL_ANOMALY_TYPES)) if anomaly_type == "combined" else anomaly_type
        length = 1 if segment_type == "spikes" else int(rng.integers(2, max_len + 1))
        start = int(rng.integers(0, max(1, len(values) - length)))
        end = start + length
        labels[start:end] = 1
        apply_stl_anomaly_segment(synthetic, values, start, end, segment_type, rng, scale)
    return synthetic.astype(np.float32), labels

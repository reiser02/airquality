"""Synthetic anomaly injection for the benchmark's ``synthetic`` mode.

Anomaly segments are injected **directly into the real series** (no synthetic
STL base: a 2026-07-03 study — ``docs/estudio_inyeccion_stl_2026-07-03.md`` —
showed the STL look-alike base distorts per-model metrics, so it was removed).
Outside the injected segments the series is untouched, which keeps every real
statistical quirk (autocorrelated residual, true extremes) in the evaluation.

The benchmark uses the ``combined`` variant: a random shape drawn per injected
segment from :data:`ANOMALY_TYPES` (``spikes``/``scale``/``noise``/``cutoff``/
``contextual``/``speedup``), so one series gets a mix of anomaly shapes. A
single type name is also accepted as variant (used by the anomaly-types plot).
"""

from __future__ import annotations

import numpy as np

ANOMALY_TYPES = ["spikes", "scale", "noise", "cutoff", "contextual", "speedup"]


def apply_anomaly_segment(
    injected: np.ndarray,
    values: np.ndarray,
    start: int,
    end: int,
    anomaly_type: str,
    rng: np.random.Generator,
    scale: float,
) -> None:
    """Mutate ``injected[start:end]`` in place with one anomaly shape.

    ``anomaly_type`` is one of :data:`ANOMALY_TYPES`; ``scale`` is the series'
    standard deviation, used to size the perturbation. ``values`` is the
    untouched original series (used by ``cutoff`` for its quantile).
    """
    length = end - start
    if anomaly_type == "spikes":
        injected[start:end] += rng.choice([-1.0, 1.0]) * scale * 4.0
    elif anomaly_type == "scale":
        injected[start:end] *= rng.choice([0.25, 2.0])
    elif anomaly_type == "noise":
        injected[start:end] += rng.normal(0.0, scale * 2.0, size=length)
    elif anomaly_type == "cutoff":
        injected[start:end] = float(np.quantile(values, 0.75))
    elif anomaly_type == "contextual":
        injected[start:end] = injected[start:end][::-1] + rng.choice([-1.0, 1.0]) * scale
    elif anomaly_type == "speedup":
        segment = injected[start:end]
        compressed = segment[::2]
        if compressed.size:
            injected[start:end] = np.interp(
                np.linspace(0, compressed.size - 1, length), np.arange(compressed.size), compressed
            )
    else:
        raise ValueError(f"Unknown synthetic anomaly type: {anomaly_type}")


def inject_synthetic_anomalies(values: np.ndarray, variant: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Inject anomalies into a copy of ``values``; return ``(injected, labels)``.

    ``variant`` is ``combined`` (mixed shapes) or one type from
    :data:`ANOMALY_TYPES`. Segment count scales with series length
    (``len // 300``, at least one); span length is type-aware (``spikes`` is a
    single point, the rest 2..32 points).
    """
    anomaly_type = variant.split("-", maxsplit=1)[1] if variant.startswith("raw-") else variant
    if anomaly_type != "combined" and anomaly_type not in ANOMALY_TYPES:
        raise ValueError(
            f"Unknown injection variant '{variant}'. Use 'combined' or one of {ANOMALY_TYPES} "
            "(the STL synthetic base was removed; anomalies are injected into the real series)."
        )
    injected = values.astype(np.float32, copy=True)
    labels = np.zeros(len(values), dtype=np.int64)
    if len(values) < 8:
        return injected, labels

    rng = np.random.default_rng(seed)
    scale = float(np.std(values) or 1.0)
    count = max(1, len(values) // 300)
    max_len = max(4, min(32, len(values) // 20))
    for _ in range(count):
        segment_type = str(rng.choice(ANOMALY_TYPES)) if anomaly_type == "combined" else anomaly_type
        length = 1 if segment_type == "spikes" else int(rng.integers(2, max_len + 1))
        start = int(rng.integers(0, max(1, len(values) - length)))
        end = start + length
        labels[start:end] = 1
        apply_anomaly_segment(injected, values, start, end, segment_type, rng, scale)
    return injected.astype(np.float32), labels

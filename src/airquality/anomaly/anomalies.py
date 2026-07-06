"""Synthetic anomaly injection for the benchmark's ``synthetic`` mode.

Anomaly segments are injected **directly into the real series** (no synthetic
STL base: a 2026-07-03 study — ``docs/estudio_inyeccion_stl_2026-07-03.md`` —
showed the STL look-alike base distorts per-model metrics, so it was removed).
Outside the injected segments the series is untouched, which keeps every real
statistical quirk (autocorrelated residual, true extremes) in the evaluation.

The benchmark uses the ``combined`` variant: every type in :data:`ANOMALY_TYPES`
(``spikes``/``scale``/``noise``/``drift``) is injected at its own rate and span
(see :data:`ANOMALY_PROFILE`), so one series gets a realistic mix — many short
spikes, some transient scale/noise bursts, and the odd slow drift. A single type
name is also accepted as variant (used by the anomaly-types plot).

``drift`` models a **sensor losing calibration** (deriva del sensor). Unlike the
transient shapes, drift is a slow, unidirectional degradation: as an
electrochemical / metal-oxide air-quality sensor ages, its error grows *and*
accelerates, so the readings deviate more — and more erratically — the longer
the sensor stays out of calibration. See :func:`apply_anomaly_segment` for the
three drift components (baseline, sensitivity and noise) and the references.
"""

from __future__ import annotations

import numpy as np

ANOMALY_TYPES = ["spikes", "scale", "noise", "drift"]

# Per-type injection profile (tunable). ``points_per_segment`` sets how often a
# segment of that type appears (one per that many series points); ``span`` is the
# (min, max) length in points of each segment. spikes are single points, so they
# can be the most frequent without covering much of the series; scale/noise are
# short transient bursts; drift is a rare, slow de-calibration episode with a
# longer (but bounded) span. Both ``inject_synthetic_anomalies`` and the
# anomaly-types plot read these, so the frequencies/spans stay consistent.
ANOMALY_PROFILE = {
    "spikes": {"points_per_segment": 150, "span": (1, 1)},
    "scale": {"points_per_segment": 300, "span": (2, 16)},
    "noise": {"points_per_segment": 300, "span": (2, 16)},
    "drift": {"points_per_segment": 900, "span": (16, 48)},
}


def _draw_span(anomaly_type: str, rng: np.random.Generator) -> int:
    """Sample a segment length (in points) for ``anomaly_type`` from its profile."""
    low, high = ANOMALY_PROFILE[anomaly_type]["span"]
    return low if low == high else int(rng.integers(low, high + 1))


def apply_anomaly_segment(
    injected: np.ndarray,
    start: int,
    end: int,
    anomaly_type: str,
    rng: np.random.Generator,
    scale: float,
) -> None:
    """Mutate ``injected[start:end]`` in place with one anomaly shape.

    ``anomaly_type`` is one of :data:`ANOMALY_TYPES`; ``scale`` is the series'
    standard deviation, used to size the perturbation.
    """
    length = end - start
    if anomaly_type == "spikes":
        injected[start:end] += rng.choice([-1.0, 1.0]) * scale * 4.0
    elif anomaly_type == "scale":
        injected[start:end] *= rng.choice([0.25, 2.0])
    elif anomaly_type == "noise":
        injected[start:end] += rng.normal(0.0, scale * 2.0, size=length)
    elif anomaly_type == "drift":
        # Sensor de-calibration (deriva): a slow, *unidirectional* loss of
        # calibration that worsens as the sensor ages. The literature on
        # low-cost electrochemical / metal-oxide air-quality sensors models
        # drift as two coupled terms — a baseline (zero/offset) drift and a
        # sensitivity (gain/span) drift — often with an accelerating
        # linear+exponential shape as the sensor degrades. Three components grow
        # together across the span, so the perturbation grows gradually and gets
        # a little larger toward the end (the user's "cada vez un poco más
        # grandes") — a smooth trend, not a burst of many spikes:
        #   * baseline drift  — an accelerating additive bias (∝ progress²),
        #   * sensitivity drift — fluctuations around the local mean amplified
        #     by a growing gain (1 + k·progress),
        #   * measurement noise — a mild variance rising with progress as the
        #     sensor becomes unstable.
        progress = np.linspace(0.0, 1.0, length, dtype=np.float32)
        segment = injected[start:end]
        local_mean = float(segment.mean())
        baseline = rng.choice([-1.0, 1.0]) * scale * 3.0 * progress**2
        gain = 1.0 + 1.0 * progress
        noise = rng.normal(0.0, 1.0, size=length).astype(np.float32) * scale * 0.3 * progress
        injected[start:end] = local_mean + (segment - local_mean) * gain + baseline + noise
    else:
        raise ValueError(f"Unknown synthetic anomaly type: {anomaly_type}")


def inject_synthetic_anomalies(values: np.ndarray, variant: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Inject anomalies into a copy of ``values``; return ``(injected, labels)``.

    ``variant`` is ``combined`` (every type, each at its own rate) or one type
    from :data:`ANOMALY_TYPES`. Both the per-type frequency (``points_per_segment``)
    and the span length are taken from :data:`ANOMALY_PROFILE`: ``spikes`` are
    frequent single points, ``scale``/``noise`` short transient bursts (2..16),
    and ``drift`` a rare, longer de-calibration window (16..48).
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
    length_n = len(values)
    # ``combined`` injects every type at its own rate (spikes densest, drift
    # rarest); a single-type variant injects just that type. Each type's count
    # and span come from ANOMALY_PROFILE, so the plot panels and the benchmark
    # agree on how often/how wide each shape appears.
    segment_types = ANOMALY_TYPES if anomaly_type == "combined" else [anomaly_type]
    for segment_type in segment_types:
        count = max(1, length_n // ANOMALY_PROFILE[segment_type]["points_per_segment"])
        for _ in range(count):
            length = min(_draw_span(segment_type, rng), length_n)  # clamp for short series
            start = int(rng.integers(0, max(1, length_n - length)))
            end = start + length
            labels[start:end] = 1
            apply_anomaly_segment(injected, start, end, segment_type, rng, scale)
    return injected.astype(np.float32), labels

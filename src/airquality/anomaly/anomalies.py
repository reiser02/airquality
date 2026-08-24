"""Synthetic anomaly injection for the benchmark's ``synthetic`` mode.

Anomaly segments are injected **directly into the real series** (no synthetic
STL base: a 2026-07-03 study — ``docs/estudio_inyeccion_stl_2026-07-03.md`` —
showed the STL look-alike base distorts per-model metrics, so it was removed).
Outside the injected segments the series is untouched, which keeps every real
statistical quirk (autocorrelated residual, true extremes) in the evaluation.

The default ``combined`` variant chooses anomaly groups from four segment-size
levels based on an 80-point reference window. Event frequencies are budgeted
over all segments of a station, not independently per segment, so splitting a
series at temporal gaps does not multiply its anomalies. A single type name is
also accepted as variant (used by the anomaly-types plot).

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
DEFAULT_INJECTION_VARIANT = "combined"
INJECTION_VARIANTS = (DEFAULT_INJECTION_VARIANT, *ANOMALY_TYPES)
INJECTION_REFERENCE_WINDOW = 80
INJECTION_POLICY_VERSION = "combined-series-w80-7to8pct-v3"

# Per-type injection profile (tunable). The historical ``points_per_segment``
# field is now applied to the station-wide observed point count; ``span`` is the
# (min, max) length in points of each event. spikes are single points, so they
# can be the most frequent without covering much of the series; scale/noise are
# short transient bursts; drift is a rare, slow de-calibration episode with a
# longer (but bounded) span. Both ``inject_synthetic_anomalies`` and the
# anomaly-types plot read these, so the frequencies/spans stay consistent.
ANOMALY_PROFILE = {
    "spikes": {"points_per_segment": 150, "span": (1, 1)},
    "scale": {"points_per_segment": 300, "span": (2, 11)},
    "noise": {"points_per_segment": 300, "span": (2, 11)},
    "drift": {"points_per_segment": 900, "span": (16, 36)},
}


def normalize_injection_variant(variant: str) -> str:
    """Normalize and validate a synthetic injection variant."""
    normalized = str(variant).strip().lower()
    if normalized.startswith("raw-"):
        normalized = normalized.removeprefix("raw-")
    if normalized not in INJECTION_VARIANTS:
        raise ValueError(
            f"Unknown injection variant '{variant}'. Use one of {INJECTION_VARIANTS} "
            "(the STL synthetic base was removed; anomalies are injected into the real series)."
        )
    return normalized


def _draw_span(anomaly_type: str, rng: np.random.Generator) -> int:
    """Sample a segment length (in points) for ``anomaly_type`` from its profile."""
    low, high = ANOMALY_PROFILE[anomaly_type]["span"]
    return low if low == high else int(rng.integers(low, high + 1))


def _combined_group(length: int, rng: np.random.Generator) -> list[str]:
    """Choose the four-level anomaly group for one selected segment."""
    window = INJECTION_REFERENCE_WINDOW
    if length < window:
        return [str(rng.choice(ANOMALY_TYPES[:3]))]
    if length < 2 * window:
        return ["spikes", str(rng.choice(("scale", "noise")))]
    if length < 3 * window:
        return list(ANOMALY_TYPES[:3]) if rng.random() < 0.5 else ["drift"]
    return list(ANOMALY_TYPES)


def _allowed_types(length: int) -> list[str]:
    """Types that can physically be placed in a segment of ``length`` points."""
    return (
        list(ANOMALY_TYPES[:3])
        if length < 2 * INJECTION_REFERENCE_WINDOW
        else list(ANOMALY_TYPES)
    )


def _package_alternatives(length: int, group: list[str]) -> list[str]:
    """Fallback types that preserve the selected level/package semantics."""
    if length < 2 * INJECTION_REFERENCE_WINDOW:
        return list(ANOMALY_TYPES[:3])
    if length < 3 * INJECTION_REFERENCE_WINDOW and group != ["drift"]:
        return list(ANOMALY_TYPES[:3])
    return _allowed_types(length)


def _choose_available_type(
    preferred: list[str],
    allowed: list[str],
    remaining: dict[str, int],
    rng: np.random.Generator,
) -> str | None:
    """Choose a preferred type with quota, falling back to any allowed quota."""
    candidates = [name for name in preferred if remaining[name] > 0]
    if not candidates:
        candidates = [name for name in allowed if remaining[name] > 0]
    if not candidates:
        return None
    weights = np.asarray([remaining[name] for name in candidates], dtype=float)
    return str(rng.choice(candidates, p=weights / weights.sum()))


def _place_event(
    segment_index: int,
    segment_length: int,
    anomaly_type: str,
    occupied: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int, int, str] | None:
    """Reserve one separated event interval, or return ``None`` if none fits."""
    drawn = _draw_span(anomaly_type, rng)
    maximum = segment_length - 1  # Always retain at least one normal point.
    minimum = 1
    if anomaly_type == "drift":
        minimum = ANOMALY_PROFILE[anomaly_type]["span"][0]
        maximum = min(maximum, segment_length // 4)
    if maximum < minimum:
        return None

    for length in range(min(drawn, maximum), minimum - 1, -1):
        starts = [
            start
            for start in range(segment_length - length + 1)
            if not occupied[
                max(0, start - 1) : min(segment_length, start + length + 1)
            ].any()
        ]
        if starts:
            start = int(rng.choice(starts))
            occupied[start : start + length] = True
            return segment_index, start, start + length, anomaly_type
    return None


def _plan_synthetic_anomalies(
    segment_lengths: list[int], variant: str, seed: int
) -> list[tuple[int, int, int, str]]:
    """Plan sparse station-wide events without crossing or overlapping segments."""
    anomaly_type = normalize_injection_variant(variant)
    eligible = [index for index, length in enumerate(segment_lengths) if length >= 8]
    if not eligible:
        return []

    rng = np.random.default_rng(seed)
    total_points = sum(segment_lengths[index] for index in eligible)
    occupied = [np.zeros(length, dtype=bool) for length in segment_lengths]
    weights = np.asarray([segment_lengths[index] for index in eligible], dtype=float)
    order = rng.choice(
        eligible, size=len(eligible), replace=False, p=weights / weights.sum()
    ).tolist()
    plans: list[tuple[int, int, int, str]] = []

    if anomaly_type == DEFAULT_INJECTION_VARIANT:
        remaining = {
            name: total_points // int(profile["points_per_segment"])
            for name, profile in ANOMALY_PROFILE.items()
        }
        if not any(remaining.values()):
            # Keep tiny synthetic cases usable without ever introducing drift.
            remaining[str(rng.choice(ANOMALY_TYPES[:3]))] = 1
        reserved: dict[int, set[str]] = {}
        if remaining["drift"] and any(
            segment_lengths[index] >= 2 * INJECTION_REFERENCE_WINDOW
            for index in eligible
        ):
            unused = order.copy()
            for required_type in ("drift", *ANOMALY_TYPES[:3]):
                candidates = [
                    index
                    for index in unused
                    if required_type != "drift"
                    or segment_lengths[index] >= 2 * INJECTION_REFERENCE_WINDOW
                ]
                if not candidates:
                    candidates = [
                        index
                        for index in order
                        if required_type != "drift"
                        or segment_lengths[index] >= 2 * INJECTION_REFERENCE_WINDOW
                    ]
                for segment_index in candidates:
                    plan = _place_event(
                        segment_index,
                        segment_lengths[segment_index],
                        required_type,
                        occupied[segment_index],
                        rng,
                    )
                    if plan is None:
                        continue
                    plans.append(plan)
                    remaining[required_type] -= 1
                    reserved.setdefault(segment_index, set()).add(required_type)
                    if segment_index in unused:
                        unused.remove(segment_index)
                    break
        target_events = sum(remaining.values())
        active = [index for index in order if index not in reserved][:target_events]
        if len(active) < target_events:
            active.extend(
                index
                for index in order
                if index in reserved
                and index not in active
                and len(active) < target_events
            )
        pending = {
            segment_index: _combined_group(segment_lengths[segment_index], rng)
            for segment_index in active
        }
        capacities = {
            segment_index: len(group) for segment_index, group in pending.items()
        }
        assigned = {
            segment_index: int(segment_index in reserved)
            for segment_index in active
        }
        used = {
            segment_index: reserved.get(segment_index, set()).copy()
            for segment_index in active
        }
        open_segments = {
            segment_index
            for segment_index in active
            if assigned[segment_index] < capacities[segment_index]
        }

        # Allocate in rounds: every active segment gets one opportunity before
        # any segment gets a second event, preventing early blocks from spending
        # the station-wide budget first.
        while any(remaining.values()) and open_segments:
            made_progress = False
            for segment_index in active:
                if not any(remaining.values()):
                    break
                if segment_index not in open_segments:
                    continue
                group = pending[segment_index]
                preferred = [
                    name
                    for name in group
                    if remaining[name] > 0 and name not in used[segment_index]
                ]
                alternatives = [
                    name
                    for name in _package_alternatives(
                        segment_lengths[segment_index], group
                    )
                    if name not in used[segment_index]
                ]
                event_type = _choose_available_type(
                    preferred,
                    alternatives,
                    remaining,
                    rng,
                )
                if event_type is None:
                    open_segments.discard(segment_index)
                    continue
                plan = _place_event(
                    segment_index,
                    segment_lengths[segment_index],
                    event_type,
                    occupied[segment_index],
                    rng,
                )
                if plan is not None:
                    plans.append(plan)
                    remaining[event_type] -= 1
                    assigned[segment_index] += 1
                    used[segment_index].add(event_type)
                    made_progress = True
                    if event_type in group:
                        group.remove(event_type)
                    if assigned[segment_index] >= capacities[segment_index]:
                        open_segments.discard(segment_index)
                else:
                    used[segment_index].add(event_type)
            if not made_progress:
                break

        compatible = {
            name: [
                index
                for index in order
                if name != "drift"
                or segment_lengths[index] >= 2 * INJECTION_REFERENCE_WINDOW
            ]
            for name in ANOMALY_TYPES
        }

        # Spend remaining station-wide quotas wherever each type can fit. This
        # keeps coverage stable when a station is split into many short blocks.
        extra_indices = dict.fromkeys(ANOMALY_TYPES, 0)
        failed_attempts = dict.fromkeys(ANOMALY_TYPES, 0)
        while any(remaining.values()):
            candidates = [
                name
                for name, quota in remaining.items()
                if quota and failed_attempts[name] < len(compatible[name])
            ]
            if not candidates:
                break
            weights = np.asarray([remaining[name] for name in candidates], dtype=float)
            event_type = str(rng.choice(candidates, p=weights / weights.sum()))
            segments = compatible[event_type]
            segment_index = segments[extra_indices[event_type] % len(segments)]
            extra_indices[event_type] += 1
            plan = _place_event(
                segment_index,
                segment_lengths[segment_index],
                event_type,
                occupied[segment_index],
                rng,
            )
            if plan is None:
                failed_attempts[event_type] += 1
                continue
            plans.append(plan)
            remaining[event_type] -= 1
            failed_attempts[event_type] = 0
    else:
        target_events = max(
            1,
            total_points // int(ANOMALY_PROFILE[anomaly_type]["points_per_segment"]),
        )
        placeable = [
            index
            for index in order
            if anomaly_type != "drift"
            or segment_lengths[index]
            >= 4 * int(ANOMALY_PROFILE["drift"]["span"][0])
        ]
        for event_number in range(target_events):
            if not placeable:
                break
            segment_index = placeable[event_number % len(placeable)]
            plan = _place_event(
                segment_index,
                segment_lengths[segment_index],
                anomaly_type,
                occupied[segment_index],
                rng,
            )
            if plan is not None:
                plans.append(plan)
    return plans


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


def inject_synthetic_anomaly_segments(
    segments: list[np.ndarray], variant: str, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inject a station's segments under one shared rare-event budget."""
    arrays = [np.asarray(values) for values in segments]
    injected = [values.astype(np.float32, copy=True) for values in arrays]
    labels = [np.zeros(len(values), dtype=np.int64) for values in arrays]
    rng = np.random.default_rng(seed)
    for segment_index, start, end, anomaly_type in _plan_synthetic_anomalies(
        [len(values) for values in arrays], variant, seed
    ):
        values = arrays[segment_index]
        scale = float(np.std(values) or 1.0)
        labels[segment_index][start:end] = 1
        apply_anomaly_segment(
            injected[segment_index], start, end, anomaly_type, rng, scale
        )
    return [
        (values.astype(np.float32), segment_labels)
        for values, segment_labels in zip(injected, labels, strict=True)
    ]


def inject_synthetic_anomalies(values: np.ndarray, variant: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Inject anomalies into a copy of ``values``; return ``(injected, labels)``.

    ``variant`` is the dynamic four-level ``combined`` policy or one type from
    :data:`ANOMALY_TYPES`. Frequencies are station-wide and spans come from
    :data:`ANOMALY_PROFILE`.
    """
    return inject_synthetic_anomaly_segments([values], variant, seed)[0]

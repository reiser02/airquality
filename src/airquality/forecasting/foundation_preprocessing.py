"""Paired synthetic contexts for the zero-shot foundation-model audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from airquality.anomaly.anomalies import ANOMALY_TYPES, inject_synthetic_anomalies
from airquality.data.series import ensure_datetime_series
from airquality.forecasting.detection import DetectionResult

CLEAN_REFERENCE = "clean_reference"
CORRUPTED = "corrupted"
REFERENCE_CONDITIONS = (CLEAN_REFERENCE, CORRUPTED)
FOUNDATION_METRICS = ("mase", "rmsse")
FOUNDATION_SUMMARY_COLUMNS = (
    "series",
    "regime",
    "horizon",
    "model",
    "case_id",
    "anomaly_type",
    "test_seed",
    "test_target_start",
    "strategy",
    "imputation_applied",
    "n_injected",
    "n_injected_detected",
    *(f"{metric}_{suffix}" for metric in FOUNDATION_METRICS for suffix in (
        "clean",
        "corrupted",
        "processed",
        "damage",
        "recovery",
        "residual",
        "recovery_pct",
    )),
)


@dataclass(frozen=True)
class SyntheticContextCase:
    """One clean forecast origin plus a synthetic corruption of its context."""

    case_id: str
    anomaly_type: str
    test_seed: int
    origin: pd.Timestamp
    clean_context: pd.Series
    corrupted_context: pd.Series
    target: pd.Series
    injected_mask: pd.Series


def build_synthetic_context_cases(
    test_series: pd.Series,
    *,
    test_target_start: pd.Timestamp,
    context_len: int,
    horizon: int,
    stride: int,
    repeats: int,
    test_seed: int,
    freq: str = "h",
) -> list[SyntheticContextCase]:
    """Inject one anomaly type per case into clean pre-target contexts.

    Origins are spread deterministically across the common test. The synthetic
    labels never touch the target horizon.
    """
    if min(context_len, horizon, stride, repeats) <= 0:
        raise ValueError("Contexto, horizonte, stride y repeticiones deben ser positivos")

    series = ensure_datetime_series(
        test_series, freq=freq, name=str(test_series.name or "series")
    )
    try:
        target_start_pos = int(series.index.get_loc(test_target_start))
    except KeyError as exc:
        raise ValueError("test_target_start debe pertenecer a test_series") from exc

    origins = list(range(target_start_pos, len(series) - horizon + 1, stride))
    if not origins:
        return []
    selected = np.linspace(0, len(origins) - 1, min(repeats, len(origins)), dtype=int)

    cases: list[SyntheticContextCase] = []
    for repeat_index, origin_index in enumerate(dict.fromkeys(selected.tolist())):
        origin_pos = origins[origin_index]
        if origin_pos < context_len:
            continue
        clean_context = series.iloc[origin_pos - context_len : origin_pos].copy()
        target = series.iloc[origin_pos : origin_pos + horizon].copy()
        if clean_context.isna().any() or target.isna().any():
            raise ValueError("El test comun debe aportar contexto y targets observados")

        for type_index, anomaly_type in enumerate(ANOMALY_TYPES):
            seed = test_seed + repeat_index * len(ANOMALY_TYPES) + type_index
            values, labels = inject_synthetic_anomalies(
                clean_context.to_numpy(dtype=np.float32), anomaly_type, seed
            )
            corrupted = pd.Series(
                values,
                index=clean_context.index,
                name=clean_context.name,
                dtype=float,
            )
            injected_mask = pd.Series(
                labels.astype(bool),
                index=clean_context.index,
                name=clean_context.name,
            )
            origin = pd.Timestamp(target.index[0])
            cases.append(
                SyntheticContextCase(
                    case_id=f"{origin.isoformat()}|{anomaly_type}|{seed}",
                    anomaly_type=anomaly_type,
                    test_seed=seed,
                    origin=origin,
                    clean_context=clean_context,
                    corrupted_context=corrupted,
                    target=target,
                    injected_mask=injected_mask,
                )
            )
    return cases


def build_preprocessing_contexts(
    case: SyntheticContextCase,
    detections: dict[str, DetectionResult],
) -> dict[str, pd.Series]:
    """Build clean, corrupted and strategy-cleaned contexts for one case."""
    contexts = {
        CLEAN_REFERENCE: case.clean_context,
        CORRUPTED: case.corrupted_context,
    }
    for strategy, detection in detections.items():
        mask = detection.mask.reindex(case.corrupted_context.index, fill_value=False)
        contexts[strategy] = case.corrupted_context.mask(mask.astype(bool))
    return contexts


def summarize_foundation_preprocessing(results: pd.DataFrame) -> pd.DataFrame:
    """Return long paired deltas against clean and corrupted references."""
    if results.empty:
        return pd.DataFrame(columns=FOUNDATION_SUMMARY_COLUMNS)

    case_keys = [
        "series",
        "regime",
        "horizon",
        "model",
        "case_id",
        "anomaly_type",
        "test_seed",
        "test_target_start",
    ]
    expected_conditions = set(results["condition"])
    rows: list[dict[str, Any]] = []
    for keys, case in results.groupby(case_keys, sort=False, dropna=False):
        by_condition = case.set_index("condition")
        complete = (
            set(by_condition.index) == expected_conditions
            and by_condition.index.is_unique
            and np.isfinite(
                by_condition[list(FOUNDATION_METRICS)].to_numpy(dtype=float)
            ).all()
            and (
                by_condition["n_test_predictions"].to_numpy(dtype=int)
                == int(case["horizon"].iloc[0])
            ).all()
        )
        if not complete:
            continue
        common = dict(zip(case_keys, keys, strict=True))
        for condition in (
            name for name in by_condition.index if name not in REFERENCE_CONDITIONS
        ):
            processed = by_condition.loc[condition]
            entry: dict[str, Any] = {
                **common,
                "strategy": condition,
                "imputation_applied": bool(processed["imputation_applied"]),
                "n_injected": int(processed["n_injected"]),
                "n_injected_detected": int(processed["n_injected_detected"]),
            }
            for metric in FOUNDATION_METRICS:
                clean = float(by_condition.loc[CLEAN_REFERENCE, metric])
                corrupted = float(by_condition.loc[CORRUPTED, metric])
                value = float(processed[metric])
                damage = corrupted - clean
                recovery = corrupted - value
                entry[f"{metric}_clean"] = clean
                entry[f"{metric}_corrupted"] = corrupted
                entry[f"{metric}_processed"] = value
                entry[f"{metric}_damage"] = damage
                entry[f"{metric}_recovery"] = recovery
                entry[f"{metric}_residual"] = value - clean
                entry[f"{metric}_recovery_pct"] = (
                    100.0 * recovery / damage if damage > 0.0 else float("nan")
                )
            rows.append(entry)
    return pd.DataFrame(rows, columns=FOUNDATION_SUMMARY_COLUMNS)


__all__ = [
    "CLEAN_REFERENCE",
    "CORRUPTED",
    "FOUNDATION_METRICS",
    "FOUNDATION_SUMMARY_COLUMNS",
    "SyntheticContextCase",
    "build_preprocessing_contexts",
    "build_synthetic_context_cases",
    "summarize_foundation_preprocessing",
]

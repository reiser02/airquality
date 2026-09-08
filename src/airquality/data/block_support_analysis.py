"""Audit raw, detected and imputed training support for forecasting.

The report follows the benchmark protocol through the common test selection,
but does not fit forecasting models. Run with::

    uv run python -m airquality.data.block_support_analysis
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from airquality.anomaly.registry import resolve_model_names
from airquality.config import (
    cfg_get_bool,
    cfg_get_csv_list,
    cfg_get_float,
    cfg_get_int,
    cfg_get_str,
)
from airquality.data.block_analysis import classify_blocks
from airquality.data.preprocessing import DETECTION_LIMITS
from airquality.data.segments import observed_blocks
from airquality.forecasting.backtest import (
    get_strict_forecast_requirements,
    select_holdout_window,
)
from airquality.forecasting.cache import (
    CACHE_VERSION,
    BenchmarkCache,
    series_fingerprint,
    transform_fingerprints,
)
from airquality.forecasting.cleaning import remove_anomalies
from airquality.forecasting.detection import (
    DEFAULT_INJECTION_VARIANT,
    DEFAULT_INJECTION_SEED,
    DEFAULT_MIN_SELECTION_POINTS,
    DEFAULT_VOTE_MIN_VOTES,
    DEFAULT_VOTE_TOP_K,
    INJECTION_POLICY_VERSION,
    MIN_SEGMENT_POINTS,
    DetectionResult,
    MaskTransform,
    build_detection_strategy,
    common_detection_support,
    normalize_injection_variant,
)
from airquality.forecasting.fill import (
    GapImputationOutcome,
    GapImputationPolicy,
    _repo_root,
    build_imputer,
    imputation_policy_cache_identity,
    impute_series_by_gap_result,
    nan_gap_windows,
    parse_imputation_gap_rules,
)
from airquality.forecasting.pipeline import (
    DEFAULT_STRATEGIES,
    _detect_for_strategies,
    _load_raw_hourly_series,
    resolve_forecasting_devices,
)
from airquality.forecasting.registry import resolve_forecasting_model_configs
from airquality.paths import create_run_dir
from airquality.run_logging import RunLogging

ANALYSIS_VERSION = 8
LOGGER = logging.getLogger(__name__)


def _normalize_pollutant(value: str) -> str:
    pollutant = str(value).strip().upper()
    if pollutant not in DETECTION_LIMITS:
        supported = ", ".join(sorted(DETECTION_LIMITS))
        raise ValueError(
            f"Contaminante no soportado: {value!r}. Valores validos: {supported}"
        )
    return pollutant


def _age_stats(
    index: pd.DatetimeIndex, test_target_start: pd.Timestamp
) -> tuple[float, float]:
    if index.empty:
        return float("nan"), float("nan")
    ages = (test_target_start - index) / pd.Timedelta(hours=1)
    return float(np.median(ages)), float(np.max(ages))


def _block_index(blocks: pd.DataFrame, selected: pd.Series) -> pd.DatetimeIndex:
    values: list[pd.Timestamp] = []
    for row in blocks.loc[selected].itertuples(index=False):
        values.extend(pd.date_range(row.start, row.end, freq="h"))
    return pd.DatetimeIndex(values)


def _effective_block_index(
    blocks: pd.DataFrame, validation_hours: int
) -> pd.DatetimeIndex:
    """Return the points actually available for training before the holdout."""
    values: list[pd.Timestamp] = []
    for row in blocks.loc[blocks["used"]].itertuples(index=False):
        block_index = pd.date_range(row.start, row.end, freq="h")
        if row.validation_host and validation_hours:
            block_index = block_index[:-validation_hours]
        values.extend(block_index)
    effective = pd.DatetimeIndex(values)
    if len(effective) != int(blocks["training_hours"].sum()):
        raise RuntimeError("El indice efectivo no coincide con las horas de entrenamiento")
    return effective


def _analyze_arm(
    series: pd.Series,
    *,
    raw_train: pd.Series,
    arm: str,
    stage: str,
    strategy: str,
    imputation_policy: str,
    detection: DetectionResult | None,
    imputed_mask: pd.Series,
    anomaly_imputed_mask: pd.Series,
    preexisting_imputed_mask: pd.Series,
    requirements: dict[str, object],
    test_target_start: pd.Timestamp,
    common: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    blocks = classify_blocks(
        observed_blocks(series),
        minimum_hours=int(requirements["minimum_hours"]),
        validation_hours=int(requirements["validation_hours"]),
        host_minimum_hours=int(requirements["host_minimum_hours"]),
    )
    blocks.insert(0, "strategy", strategy)
    blocks.insert(0, "stage", stage)
    blocks.insert(0, "arm", arm)
    blocks.insert(0, "series", str(series.name))
    blocks["observed_real_hours"] = 0
    blocks["imputed_hours"] = 0
    blocks["imputed_anomaly_hours"] = 0
    blocks["imputed_preexisting_gap_hours"] = 0
    for index, block in blocks.iterrows():
        block_index = series.loc[block["start"] : block["end"]].index
        blocks.loc[index, "imputed_hours"] = int(imputed_mask.loc[block_index].sum())
        blocks.loc[index, "imputed_anomaly_hours"] = int(
            anomaly_imputed_mask.loc[block_index].sum()
        )
        blocks.loc[index, "imputed_preexisting_gap_hours"] = int(
            preexisting_imputed_mask.loc[block_index].sum()
        )
        blocks.loc[index, "observed_real_hours"] = int(
            len(block_index) - blocks.loc[index, "imputed_hours"]
        )

    rows = []
    anomaly_mask = (
        detection.mask.reindex(raw_train.index, fill_value=False).astype(bool)
        if detection is not None
        else pd.Series(False, index=raw_train.index)
    )
    eligible = blocks["eligible"]
    valid_index = _block_index(blocks, eligible)
    valid_imputed = valid_index.intersection(imputed_mask.index[imputed_mask])
    valid_anomaly_imputed = valid_index.intersection(
        anomaly_imputed_mask.index[anomaly_imputed_mask]
    )
    valid_preexisting_imputed = valid_index.intersection(
        preexisting_imputed_mask.index[preexisting_imputed_mask]
    )
    effective_index = _effective_block_index(
        blocks, int(requirements["validation_hours"])
    )
    effective_imputed = effective_index.intersection(
        imputed_mask.index[imputed_mask]
    )
    effective_anomaly_imputed = effective_index.intersection(
        anomaly_imputed_mask.index[anomaly_imputed_mask]
    )
    effective_preexisting_imputed = effective_index.intersection(
        preexisting_imputed_mask.index[preexisting_imputed_mask]
    )
    valid_real = valid_index.difference(valid_imputed)
    imputed_age_median, imputed_age_max = _age_stats(
        valid_imputed, test_target_start
    )
    rows.append(
        {
            **common,
            "arm": arm,
            "stage": stage,
            "strategy": strategy,
            "imputed": stage == "imputed",
            "imputation_policy": imputation_policy,
            "detectors": (
                ",".join(detection.detectors) if detection is not None else ""
            ),
            "horizon_hours": int(requirements["horizon_hours"]),
            "stride_hours": int(requirements["stride_hours"]),
            "minimum_hours": int(requirements["minimum_hours"]),
            "minimum_models": str(requirements["minimum_models"]),
            "prediction_context_hours": int(requirements["prediction_context_hours"]),
            "host_minimum_hours": int(requirements["host_minimum_hours"]),
            "limiting_models": str(requirements["limiting_models"]),
            "validation_hours": int(requirements["validation_hours"]),
            "validation_forecasts": int(requirements["validation_forecasts"]),
            "observed_hours": int(series.notna().sum()),
            "real_observed_hours": int((series.notna() & ~imputed_mask).sum()),
            "imputed_hours": int(imputed_mask.sum()),
            "detected_anomalies": int((anomaly_mask & raw_train.notna()).sum()),
            "total_blocks": len(blocks),
            "valid_blocks": int(eligible.sum()),
            "valid_hours": len(valid_index),
            "valid_real_hours": len(valid_real),
            "valid_imputed_hours": len(valid_imputed),
            "valid_imputed_anomaly_hours": len(valid_anomaly_imputed),
            "valid_imputed_preexisting_gap_hours": len(valid_preexisting_imputed),
            "effective_imputed_hours": len(effective_imputed),
            "effective_imputed_anomaly_hours": len(effective_anomaly_imputed),
            "effective_imputed_preexisting_gap_hours": len(
                effective_preexisting_imputed
            ),
            "host_capable_blocks": int(blocks["host_capable"].sum()),
            "used_blocks": int(blocks["used"].sum()),
            "effective_training_hours": int(blocks["training_hours"].sum()),
            "trainable": bool(blocks["validation_host"].any()),
            "imputed_valid_age_hours_median": imputed_age_median,
            "imputed_valid_age_hours_max": imputed_age_max,
        }
    )
    return rows, blocks


def _gap_diagnostics(
    raw_train: pd.Series,
    cleaned: pd.Series,
    imputed: pd.Series,
    detection: DetectionResult,
    *,
    strategy: str,
    policy: GapImputationPolicy,
    outcomes: Sequence[GapImputationOutcome],
) -> list[dict[str, Any]]:
    mask = detection.mask.reindex(raw_train.index, fill_value=False).astype(bool)
    outcome_by_window = {
        (outcome.start, outcome.end): outcome for outcome in outcomes
    }
    rows = []
    for window in nan_gap_windows(cleaned):
        anomaly_hours = int((raw_train.loc[window].notna() & mask.loc[window]).sum())
        preexisting_hours = int(raw_train.loc[window].isna().sum())
        if anomaly_hours and preexisting_hours:
            origin = "mixed"
        elif anomaly_hours:
            origin = "anomaly"
        else:
            origin = "preexisting"
        n_filled = int(imputed.loc[window].notna().sum())
        configured_imputer = policy.model_for(len(window))
        outcome = outcome_by_window[(window[0], window[-1])]
        rows.append(
            {
                "series": str(raw_train.name),
                "strategy": strategy,
                "start": window[0],
                "end": window[-1],
                "hours": len(window),
                "origin": origin,
                "anomaly_hours": anomaly_hours,
                "preexisting_gap_hours": preexisting_hours,
                "eligible_for_imputation": configured_imputer is not None,
                "configured_imputer": configured_imputer or "none",
                "fallback_used": outcome.fallback_used,
                "effective_imputer": outcome.effective_imputer,
                "filled_hours": n_filled,
                "fully_filled": n_filled == len(window),
            }
        )
    return rows


def _add_comparisons(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    groups = out.groupby(keys, sort=False).groups if keys else {None: out.index}
    for indexes in groups.values():
        group = out.loc[indexes]
        raw = group.loc[group["arm"] == "raw"]
        if raw.empty:
            continue
        raw_valid = int(raw.iloc[0]["valid_hours"])
        raw_effective = int(raw.iloc[0]["effective_training_hours"])
        for index in indexes:
            valid = int(out.loc[index, "valid_hours"])
            effective = int(out.loc[index, "effective_training_hours"])
            out.loc[index, "valid_hours_delta_raw"] = valid - raw_valid
            out.loc[index, "valid_hours_pct_raw"] = (
                100.0 * valid / raw_valid if raw_valid else float("nan")
            )
            out.loc[index, "effective_hours_delta_raw"] = effective - raw_effective
            out.loc[index, "effective_hours_pct_raw"] = (
                100.0 * effective / raw_effective if raw_effective else float("nan")
            )
            if out.loc[index, "stage"] != "imputed":
                out.loc[index, "imputation_gain_valid_hours"] = 0
                out.loc[index, "imputation_gain_effective_hours"] = 0
                continue
            detected = group.loc[
                (group["strategy"] == out.loc[index, "strategy"])
                & (group["stage"] == "detected")
            ]
            if not detected.empty:
                out.loc[index, "imputation_gain_valid_hours"] = (
                    valid - int(detected.iloc[0]["valid_hours"])
                )
                out.loc[index, "imputation_gain_effective_hours"] = (
                    effective
                    - int(detected.iloc[0]["effective_training_hours"])
                )
    return out


def _summarize(series_summary: pd.DataFrame) -> pd.DataFrame:
    if series_summary.empty:
        return pd.DataFrame()
    totals = (
        series_summary.groupby(["arm", "stage", "strategy"], sort=False)
        .agg(
            series=("series", "nunique"),
            trainable_series=("trainable", "sum"),
            observed_hours=("observed_hours", "sum"),
            real_observed_hours=("real_observed_hours", "sum"),
            imputed_hours=("imputed_hours", "sum"),
            detected_anomalies=("detected_anomalies", "sum"),
            total_blocks=("total_blocks", "sum"),
            valid_blocks=("valid_blocks", "sum"),
            valid_hours=("valid_hours", "sum"),
            valid_real_hours=("valid_real_hours", "sum"),
            valid_imputed_hours=("valid_imputed_hours", "sum"),
            valid_imputed_anomaly_hours=("valid_imputed_anomaly_hours", "sum"),
            valid_imputed_preexisting_gap_hours=(
                "valid_imputed_preexisting_gap_hours",
                "sum",
            ),
            effective_imputed_hours=("effective_imputed_hours", "sum"),
            effective_imputed_anomaly_hours=(
                "effective_imputed_anomaly_hours",
                "sum",
            ),
            effective_imputed_preexisting_gap_hours=(
                "effective_imputed_preexisting_gap_hours",
                "sum",
            ),
            host_capable_blocks=("host_capable_blocks", "sum"),
            used_blocks=("used_blocks", "sum"),
            effective_training_hours=("effective_training_hours", "sum"),
            minimum_hours=("minimum_hours", "first"),
            host_minimum_hours=("host_minimum_hours", "first"),
            limiting_models=("limiting_models", "first"),
        )
        .reset_index()
    )
    return _add_comparisons(totals, [])


def _write_readme(path: Path, manifest: dict[str, Any]) -> None:
    requirements = manifest["comparative_requirements"]
    lines = [
        "# Soporte de bloques para forecasting",
        "",
        "El informe compara el mismo historial de train en tres estados: raw, ",
        "deteccion sin imputacion e imputacion real. No existe un brazo raw+impute.",
        "",
        "## Requisitos estrictos",
        "",
    ]
    lines.append(
        f"- Horizonte {requirements['horizon_hours']} h, stride "
        f"{requirements['stride_hours']} h: bloque valido >= "
        f"{requirements['minimum_hours']} h; anfitrion de validacion >= "
        f"{requirements['host_minimum_hours']} h ({requirements['limiting_models']})."
    )
    lines.extend(
        [
            "",
            "## Artefactos",
            "",
            "- `summary.csv`: comparacion agregada frente a raw.",
            "- `series_summary.csv`: soporte por serie y brazo.",
            "- `blocks.csv`: bloques resultantes y elegibilidad del protocolo.",
            "- `gaps.csv`: origen, longitud, imputador asignado y resultado real de cada gap.",
            "- `detection.csv`: cobertura y tasa por estrategia.",
            "- `excluded_series.csv`: series sin test comun viable.",
            "- `manifest.json`: configuracion efectiva.",
            "- `benchmark.log`: progreso y estado final de la ejecucion.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_analysis(
    *,
    output_dir: str | Path | None = None,
    pollutant: str | None = None,
    strategies: Sequence[str] | None = None,
    mask_transforms: Sequence[MaskTransform] | None = None,
) -> dict[str, Any]:
    """Run the config-driven support audit and persist tables."""
    freq = cfg_get_str("data", "freq", "h")
    requested_pollutant = (
        pollutant
        if pollutant is not None
        else cfg_get_str("forecasting", "pollutant", "NO2")
    )
    pollutant = _normalize_pollutant(requested_pollutant)
    raw_base_dir = cfg_get_str(
        "data", "raw_base_dir", "data/raw/datos_estaciones_5m"
    )
    holdout = cfg_get_int("forecasting", "holdout", 96)
    context_len = cfg_get_int("forecasting", "context_len", 72)
    seasonality_m = cfg_get_int("benchmark", "seasonality_m", 24)
    imputation_size_k = cfg_get_int("benchmark", "size_k", 5)
    horizon = cfg_get_int("forecasting", "horizon", 12)
    stride = cfg_get_int("forecasting", "stride", 6)
    validation_len = cfg_get_int("forecasting", "validation_len", 48)
    if (
        min(holdout, horizon, stride, validation_len) <= 0
        or stride > horizon
        or validation_len < horizon
        or (validation_len - horizon) % stride != 0
        or holdout < horizon
        or (holdout - horizon) % stride != 0
    ):
        raise ValueError("Holdout, horizonte, stride y validacion deben ser validos")

    model_names = list(
        cfg_get_csv_list("forecasting", "forecast_models", ("NLinear", "TiDE"))
    )
    model_configs = resolve_forecasting_model_configs(
        model_names,
        seasonality_m=seasonality_m,
        context_length=context_len,
    )
    all_requirements = {
        **get_strict_forecast_requirements(
            model_configs,
            size_k=horizon,
            validation_len=validation_len,
            validation_stride=stride,
            seasonality_m=seasonality_m,
            context_len=context_len,
        ),
        "horizon_hours": horizon,
        "stride_hours": stride,
    }
    comparative_requirements = {
        **get_strict_forecast_requirements(
            model_configs,
            size_k=horizon,
            validation_len=validation_len,
            validation_stride=stride,
            seasonality_m=seasonality_m,
            context_len=context_len,
            training_arms_only=True,
        ),
        "horizon_hours": horizon,
        "stride_hours": stride,
    }
    context_requirement = max(
        context_len,
        int(all_requirements["prediction_context_hours"]),
    )
    train_requirement = int(all_requirements["minimum_hours"])
    host_requirement = int(all_requirements["host_minimum_hours"])

    configured_strategies = (
        strategies
        if strategies is not None
        else cfg_get_csv_list("forecasting", "strategies", DEFAULT_STRATEGIES)
    )
    strategy_specs = [
        spec.strip().lower()
        for spec in configured_strategies
    ]
    threshold_k = cfg_get_float("forecasting", "threshold_k", 3.5)
    max_detection_rate = cfg_get_float("forecasting", "max_detection_rate", 0.07)
    vote_top_k = cfg_get_int("forecasting", "vote_top_k", DEFAULT_VOTE_TOP_K)
    vote_min_votes = cfg_get_int(
        "forecasting", "vote_min_votes", DEFAULT_VOTE_MIN_VOTES
    )
    strategies = [
        build_detection_strategy(
            spec,
            threshold_k=threshold_k,
            max_detection_rate=max_detection_rate,
            vote_top_k=vote_top_k,
            vote_min_votes=vote_min_votes,
        )
        for spec in dict.fromkeys(strategy_specs)
    ]
    if not strategies:
        raise ValueError("No hay estrategias de deteccion configuradas")

    seed = cfg_get_int("forecasting", "seed", 13)
    carla_stride = cfg_get_int("forecasting", "carla_stride", 1)
    if carla_stride < 1:
        raise ValueError("carla_stride debe ser positivo")
    device = resolve_forecasting_devices(
        cfg_get_str("forecasting", "device", "cpu")
    )[0]
    injection_seed = cfg_get_int(
        "forecasting", "injection_seed", DEFAULT_INJECTION_SEED
    )
    injection_variant = normalize_injection_variant(
        cfg_get_str("synthetic", "injection_variant", DEFAULT_INJECTION_VARIANT)
    )
    min_selection_points = cfg_get_int(
        "forecasting", "min_selection_points", DEFAULT_MIN_SELECTION_POINTS
    )
    if threshold_k < 0:
        raise ValueError("threshold_k no puede ser negativo")
    if not 0.0 <= max_detection_rate <= 1.0:
        raise ValueError("max_detection_rate debe estar entre 0 y 1")
    if min_selection_points < MIN_SEGMENT_POINTS:
        raise ValueError(
            f"min_selection_points debe ser al menos {MIN_SEGMENT_POINTS}"
        )
    detectors = resolve_model_names(
        list(cfg_get_csv_list("forecasting", "detectors", ("all",)))
    )
    imputation_policy = parse_imputation_gap_rules(
        cfg_get_str("forecasting", "imputation_gap_rules", "")
    )
    imputer_key = imputation_policy_cache_identity(
        imputation_policy,
        size_k=imputation_size_k,
    )
    use_cache = cfg_get_bool("forecasting", "use_cache", True)
    cache_dir = cfg_get_str("forecasting", "cache_dir", "reports/forecasting/cache")
    cache = BenchmarkCache((_repo_root() / cache_dir) if use_cache else None)
    imputer_ref: dict[str, Any] = {}

    def get_imputer(model_name: str) -> Any:
        if model_name not in imputer_ref:
            imputer_ref[model_name] = build_imputer(
                model_name, freq=freq, size_k=imputation_size_k
            )
        return imputer_ref[model_name]

    series_rows: list[dict[str, Any]] = []
    block_frames: list[pd.DataFrame] = []
    gap_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    series_dfs = _load_raw_hourly_series(
        pollutant=pollutant,
        raw_base_dir=raw_base_dir,
        freq=freq,
    )
    if not series_dfs:
        raise RuntimeError("No se cargaron series para el analisis")

    LOGGER.info(
        "Block support initialized: pollutant=%s series=%d strategies=%s detectors=%s device=%s cache=%s",
        pollutant,
        len(series_dfs),
        ",".join(strategy.name for strategy in strategies),
        ",".join(detectors),
        device,
        "enabled" if use_cache else "disabled",
    )
    transform_names = transform_fingerprints(mask_transforms)
    for series_index, frame in enumerate(series_dfs, start=1):
        series = frame.iloc[:, 0]
        name = str(series.name)
        LOGGER.info(
            "Series %d/%d started: %s (%d observed hours)",
            series_index,
            len(series_dfs),
            name,
            int(series.notna().sum()),
        )
        series_fp = series_fingerprint(series)
        base_key = {
            "version": CACHE_VERSION,
            "analysis": "forecasting-block-support",
            "analysis_version": ANALYSIS_VERSION,
            "series": name,
            "series_fp": series_fp,
            "freq": freq,
            "detectors": sorted(detectors),
            "seed": seed,
            "carla_stride": carla_stride,
            "injection_seed": injection_seed,
            "injection_variant": injection_variant,
            "injection_policy": INJECTION_POLICY_VERSION,
            "min_selection_points": min_selection_points,
            "transforms": transform_names,
            "holdout": holdout,
            "context_len": context_len,
            "horizon": horizon,
            "stride": stride,
            "validation_len": validation_len,
        }
        detections = _detect_for_strategies(
            series,
            strategies,
            mask_transforms,
            cache=cache,
            base_key=base_key,
            context_kwargs={
                "detectors": detectors,
                "seed": seed,
                "device": device,
                "freq": freq,
                "carla_stride": carla_stride,
                "injection_seed": injection_seed,
                "injection_variant": injection_variant,
                "min_selection_points": min_selection_points,
                "cache": cache,
                "cache_key": {
                    "version": CACHE_VERSION,
                    "analysis_version": ANALYSIS_VERSION,
                    "series": name,
                    "series_fp": series_fp,
                    "freq": freq,
                    "seed": seed,
                    "carla_stride": carla_stride,
                },
            },
        )
        support, _common_mask = common_detection_support(series, detections)
        window = select_holdout_window(
            support,
            holdout=holdout,
            context_len=context_requirement,
            train_min_len=train_requirement,
            validation_len=validation_len,
            freq=freq,
            host_min_len=host_requirement,
        )
        if window is None:
            excluded_rows.append(
                {
                    "series": name,
                    "series_start": series.index.min(),
                    "series_end": series.index.max(),
                    "observed_hours": int(series.notna().sum()),
                    "exclusion_reason": "no_common_fixed_test_and_training_host",
                }
            )
            LOGGER.warning(
                "Series %d/%d excluded: %s (no common fixed test and training host)",
                series_index,
                len(series_dfs),
                name,
            )
            continue

        train_raw = series.loc[window["train_index"]]
        common = {
            "series": name,
            "series_start": series.index.min(),
            "series_end": series.index.max(),
            "train_start": train_raw.index.min(),
            "train_end": train_raw.index.max(),
            "test_context_start": window["test_context_start"],
            "test_target_start": window["test_target_start"],
            "test_target_end": window["test_target_end"],
            "test_target_hours": window["test_target_hours"],
            "test_age_hours": int(
                (series.index.max() - window["test_target_end"])
                / pd.Timedelta(hours=1)
            ),
        }
        false_mask = pd.Series(False, index=train_raw.index)
        raw_rows, raw_blocks = _analyze_arm(
            train_raw,
            raw_train=train_raw,
            arm="raw",
            stage="raw",
            strategy="none",
            imputation_policy="none",
            detection=None,
            imputed_mask=false_mask,
            anomaly_imputed_mask=false_mask,
            preexisting_imputed_mask=false_mask,
            requirements=comparative_requirements,
            test_target_start=window["test_target_start"],
            common=common,
        )
        series_rows.extend(raw_rows)
        block_frames.append(raw_blocks)

        for strategy in strategies:
            detection = detections[strategy.name]
            scored = (
                detection.scored_mask.reindex(series.index, fill_value=False).astype(bool)
                if detection.scored_mask is not None
                else series.notna()
            )
            n_observed = int(series.notna().sum())
            n_scored = int((scored & series.notna()).sum())
            train_mask = detection.mask.reindex(train_raw.index, fill_value=False).astype(bool)
            detection_rows.append(
                {
                    "series": name,
                    "strategy": strategy.name,
                    "detectors": ",".join(detection.detectors),
                    "discarded": ",".join(detection.discarded),
                    "observed_hours": n_observed,
                    "scored_hours": n_scored,
                    "unscored_hours": n_observed - n_scored,
                    "coverage_pct": 100.0 * n_scored / n_observed if n_observed else 0.0,
                    "flagged_hours_full": int(
                        (detection.mask.reindex(series.index, fill_value=False) & series.notna()).sum()
                    ),
                    "flagged_hours_train": int((train_mask & train_raw.notna()).sum()),
                    "detection_rate_pct": 100.0 * detection.detection_rate,
                }
            )
            LOGGER.info(
                "Series %d/%d strategy=%s coverage=%.2f%% detection_rate=%.2f%% flagged_train=%d",
                series_index,
                len(series_dfs),
                strategy.name,
                100.0 * n_scored / n_observed if n_observed else 0.0,
                100.0 * detection.detection_rate,
                int((train_mask & train_raw.notna()).sum()),
            )
            cleaned = remove_anomalies(train_raw, detection)
            clean_rows, clean_blocks = _analyze_arm(
                cleaned,
                raw_train=train_raw,
                arm=f"{strategy.name}+noimpute",
                stage="detected",
                strategy=strategy.name,
                imputation_policy="none",
                detection=detection,
                imputed_mask=false_mask,
                anomaly_imputed_mask=false_mask,
                preexisting_imputed_mask=false_mask,
                requirements=comparative_requirements,
                test_target_start=window["test_target_start"],
                common=common,
            )
            series_rows.extend(clean_rows)
            block_frames.append(clean_blocks)

            imputation_key = {
                **base_key,
                "stage": "support_imputation",
                "train_fp": series_fingerprint(train_raw),
                "cleaned_fp": series_fingerprint(cleaned),
                "strategy": asdict(strategy),
                "imputer": imputer_key,
            }
            imputation_result = cache.get("imputation_support", imputation_key)
            if imputation_result is None:
                LOGGER.info(
                    "Series %d/%d strategy=%s imputation cache miss",
                    series_index,
                    len(series_dfs),
                    strategy.name,
                )
                imputation_result = impute_series_by_gap_result(
                    cleaned,
                    imputation_policy,
                    get_imputer,
                    freq=freq,
                )
                cache.put("imputation_support", imputation_key, imputation_result)
            else:
                LOGGER.info(
                    "Series %d/%d strategy=%s imputation cache hit",
                    series_index,
                    len(series_dfs),
                    strategy.name,
                )
            imputed = imputation_result.series
            imputed_mask = cleaned.isna() & imputed.notna()
            anomaly_imputed_mask = imputed_mask & train_raw.notna() & train_mask
            preexisting_imputed_mask = imputed_mask & train_raw.isna()
            imputed_rows, imputed_blocks = _analyze_arm(
                imputed,
                raw_train=train_raw,
                arm=f"{strategy.name}+impute",
                stage="imputed",
                strategy=strategy.name,
                imputation_policy=imputation_policy.spec,
                detection=detection,
                imputed_mask=imputed_mask,
                anomaly_imputed_mask=anomaly_imputed_mask,
                preexisting_imputed_mask=preexisting_imputed_mask,
                requirements=comparative_requirements,
                test_target_start=window["test_target_start"],
                common=common,
            )
            series_rows.extend(imputed_rows)
            block_frames.append(imputed_blocks)
            gap_rows.extend(
                _gap_diagnostics(
                    train_raw,
                    cleaned,
                    imputed,
                    detection,
                    strategy=strategy.name,
                    policy=imputation_policy,
                    outcomes=imputation_result.outcomes,
                )
            )
        LOGGER.info("Series %d/%d completed: %s", series_index, len(series_dfs), name)

    series_summary = _add_comparisons(pd.DataFrame(series_rows), ["series"])
    summary = _summarize(series_summary)
    blocks = pd.concat(block_frames, ignore_index=True) if block_frames else pd.DataFrame()
    gaps = pd.DataFrame(gap_rows)
    detection = pd.DataFrame(detection_rows)
    excluded = pd.DataFrame(excluded_rows)
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "pollutant": pollutant,
        "forecast_models": model_names,
        "strategies": [strategy.name for strategy in strategies],
        "detectors": detectors,
        "carla_stride": carla_stride,
        "injection_seed": injection_seed,
        "injection_variant": injection_variant,
        "injection_policy": INJECTION_POLICY_VERSION,
        "imputation_gap_rules": imputation_policy.spec,
        "holdout": holdout,
        "context_len": context_len,
        "horizon": horizon,
        "stride": stride,
        "validation_len": validation_len,
        "all_model_requirements": all_requirements,
        "comparative_requirements": comparative_requirements,
    }

    output = (
        Path(output_dir)
        if output_dir is not None
        else create_run_dir(
            _repo_root() / "reports" / "data_blocks",
            f"forecast_support_{pollutant}_{datetime.now():%Y%m%d_%H%M%S}",
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    series_summary.to_csv(output / "series_summary.csv", index=False)
    blocks.to_csv(output / "blocks.csv", index=False)
    gaps.to_csv(output / "gaps.csv", index=False)
    detection.to_csv(output / "detection.csv", index=False)
    excluded.to_csv(output / "excluded_series.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _write_readme(output / "README.md", manifest)
    LOGGER.info("Cache summary: %s", cache.stats())
    LOGGER.info("Block support artifacts saved under %s", output)
    return {
        "output_dir": output,
        "log_path": output / "benchmark.log",
        "summary_df": summary,
        "series_summary_df": series_summary,
        "blocks_df": blocks,
        "gaps_df": gaps,
        "detection_df": detection,
        "excluded_df": excluded,
    }


def run_analysis(
    *,
    output_dir: str | Path | None = None,
    pollutant: str | None = None,
    strategies: Sequence[str] | None = None,
    mask_transforms: Sequence[MaskTransform] | None = None,
) -> dict[str, Any]:
    """Run the support audit with a persistent log in its output directory."""
    requested_pollutant = (
        pollutant
        if pollutant is not None
        else cfg_get_str("forecasting", "pollutant", "NO2")
    )
    normalized_pollutant = _normalize_pollutant(requested_pollutant)
    output = (
        Path(output_dir)
        if output_dir is not None
        else create_run_dir(
            _repo_root() / "reports" / "data_blocks",
            f"forecast_support_{normalized_pollutant}_{datetime.now():%Y%m%d_%H%M%S}",
        )
    )
    with RunLogging(output, "block_support_analysis"):
        return _run_analysis(
            output_dir=output,
            pollutant=normalized_pollutant,
            strategies=strategies,
            mask_transforms=mask_transforms,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara soporte raw, detectado e imputado para forecasting"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--pollutant",
        default=None,
        help="Contaminante (CO, NO2 u O3); por defecto usa [forecasting] pollutant.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Estrategias de deteccion para este informe; por defecto usa [forecasting] strategies.",
    )
    args = parser.parse_args()
    run_analysis(
        output_dir=args.output_dir,
        pollutant=args.pollutant,
        strategies=args.strategies,
    )


if __name__ == "__main__":
    main()

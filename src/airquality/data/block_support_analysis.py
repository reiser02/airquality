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
    artifact_fingerprint,
    effective_config,
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
    DEFAULT_MAX_GAP_SIZE,
    _repo_root,
    _resolve_tspulse_model_path,
    build_imputer,
    impute_series,
    nan_gap_windows,
)
from airquality.forecasting.pipeline import (
    DEFAULT_STRATEGIES,
    ForecastRegime,
    _detect_for_strategies,
    _load_raw_hourly_series,
    resolve_forecasting_devices,
)
from airquality.forecasting.registry import resolve_forecasting_model_configs
from airquality.imputation.registry import DARTS_GLOBAL, TSPULSE, resolve_imputer_family
from airquality.paths import create_run_dir

ANALYSIS_VERSION = 5
REGIME_NAMES = ("short", "long")


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


def _imputer_identity(
    model_name: str, *, size_k: int, max_gap_size: int
) -> dict[str, Any]:
    family = resolve_imputer_family(model_name)
    config: dict[str, Any] = {
        "model": model_name,
        "family": family,
        "size_k": size_k,
        "max_gap_size": max_gap_size,
    }
    artifacts: dict[str, str | None] = {}
    if family == DARTS_GLOBAL:
        weights = _repo_root() / "models" / f"{model_name}_k{size_k}.pt"
        artifacts = {
            "model": artifact_fingerprint(weights),
            "checkpoint": artifact_fingerprint(Path(f"{weights}.ckpt")),
        }
    elif family == TSPULSE:
        model_path = _resolve_tspulse_model_path(model_name)
        model_id = cfg_get_str(
            "tspulse", "model_id", "ibm-granite/granite-timeseries-tspulse-r1"
        )
        config["tspulse"] = effective_config(
            {
                "model_path": model_path,
                "model_id": model_id,
                "revision": cfg_get_str(
                    "tspulse", "revision", "tspulse-hybrid-dualhead-512-p8-r1"
                ),
                "context_length": cfg_get_int("tspulse", "context_length", 512),
                "device": cfg_get_str("tspulse", "device", "cpu"),
            }
        )
        local_source = Path(model_path or model_id).expanduser()
        if local_source.exists():
            artifacts["model"] = artifact_fingerprint(local_source)
    return {"config": config, "artifacts": artifacts}


def _block_index(blocks: pd.DataFrame, selected: pd.Series) -> pd.DatetimeIndex:
    values: list[pd.Timestamp] = []
    for row in blocks.loc[selected].itertuples(index=False):
        values.extend(pd.date_range(row.start, row.end, freq="h"))
    return pd.DatetimeIndex(values)


def _analyze_arm(
    series: pd.Series,
    *,
    raw_train: pd.Series,
    arm: str,
    stage: str,
    strategy: str,
    imputation_model: str,
    detection: DetectionResult | None,
    imputed_mask: pd.Series,
    anomaly_imputed_mask: pd.Series,
    preexisting_imputed_mask: pd.Series,
    requirements: dict[str, dict[str, object]],
    test_target_start: pd.Timestamp,
    common: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    blocks = classify_blocks(
        observed_blocks(series),
        {name: int(req["minimum_hours"]) for name, req in requirements.items()},
        {name: int(req["validation_hours"]) for name, req in requirements.items()},
        {name: int(req["host_minimum_hours"]) for name, req in requirements.items()},
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
    for regime, req in requirements.items():
        eligible = blocks[f"{regime}_eligible"]
        valid_index = _block_index(blocks, eligible)
        valid_imputed = valid_index.intersection(imputed_mask.index[imputed_mask])
        valid_anomaly_imputed = valid_index.intersection(
            anomaly_imputed_mask.index[anomaly_imputed_mask]
        )
        valid_preexisting_imputed = valid_index.intersection(
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
                "imputation_model": imputation_model,
                "detectors": (
                    ",".join(detection.detectors) if detection is not None else ""
                ),
                "regime": regime,
                "horizon_hours": int(req["horizon_hours"]),
                "stride_hours": int(req["stride_hours"]),
                "minimum_hours": int(req["minimum_hours"]),
                "minimum_models": str(req["minimum_models"]),
                "prediction_context_hours": int(req["prediction_context_hours"]),
                "host_minimum_hours": int(req["host_minimum_hours"]),
                "limiting_models": str(req["limiting_models"]),
                "validation_hours": int(req["validation_hours"]),
                "validation_forecasts": int(req["validation_forecasts"]),
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
                "host_capable_blocks": int(blocks[f"{regime}_host_capable"].sum()),
                "used_blocks": int(blocks[f"{regime}_used"].sum()),
                "effective_training_hours": int(
                    blocks[f"{regime}_training_hours"].sum()
                ),
                "trainable": bool(blocks[f"{regime}_validation_host"].any()),
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
    max_gap_size: int,
) -> list[dict[str, Any]]:
    mask = detection.mask.reindex(raw_train.index, fill_value=False).astype(bool)
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
                "eligible_for_imputation": len(window) <= max_gap_size,
                "filled_hours": n_filled,
                "fully_filled": n_filled == len(window),
            }
        )
    return rows


def _add_comparisons(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    for _, indexes in out.groupby(keys, sort=False).groups.items():
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
                continue
            detected = group.loc[
                (group["strategy"] == out.loc[index, "strategy"])
                & (group["stage"] == "detected")
            ]
            if not detected.empty:
                out.loc[index, "imputation_gain_valid_hours"] = (
                    valid - int(detected.iloc[0]["valid_hours"])
                )
    return out


def _summarize(series_summary: pd.DataFrame) -> pd.DataFrame:
    if series_summary.empty:
        return pd.DataFrame()
    totals = (
        series_summary.groupby(["regime", "arm", "stage", "strategy"], sort=False)
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
            host_capable_blocks=("host_capable_blocks", "sum"),
            used_blocks=("used_blocks", "sum"),
            effective_training_hours=("effective_training_hours", "sum"),
            minimum_hours=("minimum_hours", "first"),
            host_minimum_hours=("host_minimum_hours", "first"),
            limiting_models=("limiting_models", "first"),
        )
        .reset_index()
    )
    return _add_comparisons(totals, ["regime"])


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
    for name in REGIME_NAMES:
        req = requirements[name]
        lines.append(
            f"- `{name}`: bloque valido >= {req['minimum_hours']} h; "
            f"anfitrion de validacion >= {req['host_minimum_hours']} h "
            f"({req['limiting_models']})."
        )
    lines.extend(
        [
            "",
            "## Artefactos",
            "",
            "- `summary.csv`: comparacion agregada frente a raw.",
            "- `series_summary.csv`: soporte por serie, brazo y regimen.",
            "- `blocks.csv`: bloques resultantes y elegibilidad short/long.",
            "- `gaps.csv`: origen, longitud y resultado real de cada gap.",
            "- `detection.csv`: cobertura y tasa por estrategia.",
            "- `excluded_series.csv`: series sin test comun viable.",
            "- `manifest.json`: configuracion efectiva.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    *,
    output_dir: str | Path | None = None,
    pollutant: str | None = None,
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
    holdout = cfg_get_int("forecasting", "holdout", 192)
    context_len = cfg_get_int("forecasting", "context_len", 72)
    seasonality_m = cfg_get_int("benchmark", "seasonality_m", 24)
    imputation_size_k = cfg_get_int("benchmark", "size_k", 5)
    regimes = (
        ForecastRegime(
            "short",
            cfg_get_int("forecasting", "short_horizon", 8),
            cfg_get_int("forecasting", "short_stride", 4),
            cfg_get_int("forecasting", "short_validation_len", 48),
        ),
        ForecastRegime(
            "long",
            cfg_get_int("forecasting", "long_horizon", 48),
            cfg_get_int("forecasting", "long_stride", 24),
            cfg_get_int("forecasting", "long_validation_len", 96),
        ),
    )
    if holdout <= 0 or any(
        min(regime.horizon, regime.stride, regime.validation_len) <= 0
        or regime.stride > regime.horizon
        or regime.validation_len < regime.horizon
        or (regime.validation_len - regime.horizon) % regime.stride != 0
        or holdout < regime.horizon
        or (holdout - regime.horizon) % regime.stride != 0
        for regime in regimes
    ):
        raise ValueError(
            "Holdout, horizonte, stride y validacion de cada regimen deben ser validos"
        )

    model_names = list(
        cfg_get_csv_list("forecasting", "forecast_models", ("NLinear", "TiDE"))
    )
    model_configs = resolve_forecasting_model_configs(
        model_names,
        seasonality_m=seasonality_m,
        context_length=context_len,
    )
    all_requirements = {
        regime.name: {
            **get_strict_forecast_requirements(
                model_configs,
                size_k=regime.horizon,
                validation_len=regime.validation_len,
                validation_stride=regime.stride,
                seasonality_m=seasonality_m,
                context_len=context_len,
            ),
            "horizon_hours": regime.horizon,
            "stride_hours": regime.stride,
        }
        for regime in regimes
    }
    comparative_requirements = {
        regime.name: {
            **get_strict_forecast_requirements(
                model_configs,
                size_k=regime.horizon,
                validation_len=regime.validation_len,
                validation_stride=regime.stride,
                seasonality_m=seasonality_m,
                context_len=context_len,
                training_arms_only=True,
            ),
            "horizon_hours": regime.horizon,
            "stride_hours": regime.stride,
        }
        for regime in regimes
    }
    context_requirement = max(
        context_len,
        *(int(req["prediction_context_hours"]) for req in all_requirements.values()),
    )
    train_requirement = max(
        int(req["minimum_hours"]) for req in all_requirements.values()
    )
    host_requirement = max(
        int(req["host_minimum_hours"]) for req in all_requirements.values()
    )

    strategy_specs = [
        spec.strip().lower()
        for spec in cfg_get_csv_list("forecasting", "strategies", DEFAULT_STRATEGIES)
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
    imputation_model = cfg_get_str("forecasting", "imputation_model", "TSPulse")
    max_imputation_gap = cfg_get_int(
        "forecasting", "max_imputation_gap", DEFAULT_MAX_GAP_SIZE
    )
    if max_imputation_gap < 1:
        raise ValueError("max_imputation_gap debe ser positivo")
    imputer_key = _imputer_identity(
        imputation_model,
        size_k=imputation_size_k,
        max_gap_size=max_imputation_gap,
    )
    use_scaler = imputation_model not in ("interp", "LinearInterp")
    use_cache = cfg_get_bool("forecasting", "use_cache", True)
    cache_dir = cfg_get_str("forecasting", "cache_dir", "reports/forecasting/cache")
    cache = BenchmarkCache((_repo_root() / cache_dir) if use_cache else None)
    imputer_ref: list[Any] = []

    def get_imputer() -> Any:
        if not imputer_ref:
            imputer_ref.append(
                build_imputer(imputation_model, freq=freq, size_k=imputation_size_k)
            )
        return imputer_ref[0]

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

    transform_names = transform_fingerprints(mask_transforms)
    for frame in series_dfs:
        series = frame.iloc[:, 0]
        name = str(series.name)
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
            validation_len=max(regime.validation_len for regime in regimes),
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
            imputation_model="none",
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
            cleaned = remove_anomalies(train_raw, detection)
            clean_rows, clean_blocks = _analyze_arm(
                cleaned,
                raw_train=train_raw,
                arm=f"{strategy.name}+noimpute",
                stage="detected",
                strategy=strategy.name,
                imputation_model="none",
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
            imputed = cache.get("imputation_support", imputation_key)
            if imputed is None:
                imputed = impute_series(
                    cleaned,
                    get_imputer(),
                    freq=freq,
                    use_scaler=use_scaler,
                    max_gap_size=max_imputation_gap,
                )
                cache.put("imputation_support", imputation_key, imputed)
            imputed_mask = cleaned.isna() & imputed.notna()
            anomaly_imputed_mask = imputed_mask & train_raw.notna() & train_mask
            preexisting_imputed_mask = imputed_mask & train_raw.isna()
            imputed_rows, imputed_blocks = _analyze_arm(
                imputed,
                raw_train=train_raw,
                arm=f"{strategy.name}+impute",
                stage="imputed",
                strategy=strategy.name,
                imputation_model=imputation_model,
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
                    max_gap_size=max_imputation_gap,
                )
            )

    series_summary = _add_comparisons(
        pd.DataFrame(series_rows), ["series", "regime"]
    )
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
        "imputation_model": imputation_model,
        "max_imputation_gap": max_imputation_gap,
        "holdout": holdout,
        "context_len": context_len,
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
    print(f"[cache] {cache.stats()}")
    print(f"[info] Diagnostico de soporte en {output}")
    return {
        "output_dir": output,
        "summary_df": summary,
        "series_summary_df": series_summary,
        "blocks_df": blocks,
        "gaps_df": gaps,
        "detection_df": detection,
        "excluded_df": excluded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara soporte raw, detectado e imputado para forecasting"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--pollutant",
        default=None,
        help="Contaminante (CO o NO2); por defecto usa [forecasting] pollutant.",
    )
    args = parser.parse_args()
    run_analysis(output_dir=args.output_dir, pollutant=args.pollutant)


if __name__ == "__main__":
    main()

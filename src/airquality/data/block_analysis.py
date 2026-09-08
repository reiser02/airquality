"""Analyze contiguous hourly blocks available for forecasting models.

Run with::

    uv run python -m airquality.data.block_analysis

The command preprocesses the raw 5-minute NO2/O3 files, reserves one complete
observed block for an exact 96-hour test, and audits the configured rolling
validation requirement. It writes CSV tables plus figures under
``reports/data_blocks/<pollutants>_<timestamp>/`` without running detectors or
models, so its raw-support coverage is an upper bound for the detector-aware
benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from airquality.config import cfg_get_csv_list, cfg_get_int
from airquality.data.loaders import load_raw_5m
from airquality.data.preprocessing import MIN_RUN, MIN_USEFUL, preprocess
from airquality.data.segments import observed_blocks
from airquality.data.series import ensure_datetime_series
from airquality.forecasting.backtest import (
    get_strict_forecast_requirements,
    select_holdout_window,
)
from airquality.forecasting.registry import resolve_forecasting_model_configs
from airquality.paths import create_run_dir


DEFAULT_POLLUTANTS = ("NO2", "O3")


def _pollutant_run_label(pollutants: tuple[str, ...]) -> str:
    """Return the underscore-separated pollutant label used in run directories."""
    labels = [
        str(pollutant).strip().upper()
        for pollutant in pollutants
        if str(pollutant).strip()
    ]
    return "_".join(labels) or "UNKNOWN"


def _requirement_note(table: pd.DataFrame) -> str:
    """Return the model-geometry note used by the report visualizations."""
    from airquality.visualizations.block_analysis import _requirement_note as build_note

    return build_note(table)


def _worst_case_requirements(
    model_names: tuple[str, ...],
    *,
    context: int,
    horizon: int,
    stride: int,
    validation_hours: int,
    seasonality_m: int,
) -> dict[str, object]:
    """Return conservative native geometry among configured forecast models."""
    if not model_names:
        raise ValueError("Debe configurarse al menos un modelo de forecasting")

    configs = resolve_forecasting_model_configs(
        list(model_names), seasonality_m=seasonality_m, context_length=context
    )
    strict = get_strict_forecast_requirements(
        configs,
        size_k=horizon,
        validation_len=validation_hours,
        validation_stride=stride,
        seasonality_m=seasonality_m,
        context_len=context,
    )
    return {
        key: strict[key]
        for key in (
            "minimum_hours",
            "prediction_context_hours",
            "host_minimum_hours",
            "validation_hours",
            "validation_forecasts",
            "limiting_models",
        )
    }


def classify_blocks(
    blocks: pd.DataFrame,
    *,
    minimum_hours: int,
    validation_hours: int,
    host_minimum_hours: int,
) -> pd.DataFrame:
    """Mark eligible, validation-host, and chronologically usable blocks."""
    out = blocks.copy()
    out["block_number"] = np.arange(1, len(out) + 1)

    eligible = out["hours"].ge(minimum_hours)
    host_capable = out["hours"].ge(host_minimum_hours)
    out["eligible"] = eligible
    out["host_capable"] = host_capable
    out["validation_host"] = False
    out["used"] = False
    out["training_hours"] = 0

    if host_capable.any():
        host = int(out.index[host_capable][-1])
        used = eligible & out.index.to_series().le(host).to_numpy()
        out.loc[used, "used"] = True
        out.loc[used, "training_hours"] = out.loc[used, "hours"]
        out.loc[host, "validation_host"] = True
        out.loc[host, "training_hours"] -= validation_hours

    return out


def analyze_raw_blocks(
    base_dir: str | Path,
    pollutants: tuple[str, ...],
    *,
    context: int,
    horizon: int,
    stride: int,
    validation_len: int,
    holdout: int,
    min_run: int,
    min_useful: int,
    forecast_models: tuple[str, ...],
    seasonality_m: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build per-block, per-series, and excluded-series audit tables."""
    if (
        min(holdout, horizon, stride, validation_len) <= 0
        or stride > horizon
        or validation_len < horizon
        or (validation_len - horizon) % stride != 0
        or holdout < horizon
        or (holdout - horizon) % stride != 0
    ):
        raise ValueError(
            "Holdout, horizonte, stride y validacion deben ser validos"
        )
    requirements = _worst_case_requirements(
        forecast_models,
        context=context,
        horizon=horizon,
        stride=stride,
        validation_hours=validation_len,
        seasonality_m=seasonality_m,
    )
    minimum_hours = int(requirements["minimum_hours"])
    host_minimum_hours = int(requirements["host_minimum_hours"])
    validation_reserve = int(requirements["validation_hours"])
    prediction_context = max(context, int(requirements["prediction_context_hours"]))
    block_frames: list[pd.DataFrame] = []
    series_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []

    for pollutant in pollutants:
        stations = load_raw_5m(pollutant, str(base_dir))
        if not stations:
            raise FileNotFoundError(f"No raw files found for {pollutant} under {base_dir}")

        for station, raw in stations:
            (hourly,), _ = preprocess(
                [raw], pollutant, min_run=min_run, min_useful=min_useful
            )
            series = ensure_datetime_series(
                hourly.iloc[:, 0], freq="h", name=f"{station}/{pollutant}"
            )
            full_blocks = observed_blocks(series)
            details: dict[str, object] = {
                "full_blocks": len(full_blocks),
                "full_observed_hours": int(series.notna().sum()),
                "max_full_block": (
                    int(full_blocks["hours"].max()) if not full_blocks.empty else 0
                ),
            }
            window = select_holdout_window(
                series,
                holdout=holdout,
                context_len=prediction_context,
                train_min_len=minimum_hours,
                validation_len=validation_len,
                host_min_len=host_minimum_hours,
            )
            if window is None:
                excluded_rows.append(
                    {"pollutant": pollutant, "station": station}
                    | details
                    | {"exclusion_reason": "no_fixed_test_or_training_host"}
                )
                continue

            train = series.loc[window["train_index"]]
            details |= {
                "test_context_start": window["test_context_start"],
                "source_run_start": window["source_run_start"],
                "test_target_start": window["test_target_start"],
                "test_target_end": window["test_target_end"],
                "test_target_hours": window["test_target_hours"],
                "prior_observed_hours": int(train.notna().sum()),
            }
            # The new split keeps every pre-target value, including the prefix
            # and context from the run that hosts the fixed holdout.
            training_blocks = observed_blocks(train)
            blocks = classify_blocks(
                training_blocks,
                minimum_hours=minimum_hours,
                validation_hours=validation_reserve,
                host_minimum_hours=host_minimum_hours,
            )
            blocks.insert(0, "station", station)
            blocks.insert(0, "pollutant", pollutant)
            block_frames.append(blocks)

            row: dict[str, object] = {
                "pollutant": pollutant,
                "station": station,
                "test_target_start": details["test_target_start"],
                "test_target_end": details["test_target_end"],
                "test_context_start": details["test_context_start"],
                "source_run_start": details["source_run_start"],
                "test_target_hours": details["test_target_hours"],
                "prior_observed_hours": details["prior_observed_hours"],
                "forecast_models": ", ".join(forecast_models),
                "observed_hours": int(blocks["hours"].sum()),
                "total_blocks": len(blocks),
                "max_block_hours": int(blocks["hours"].max()),
            }
            eligible = blocks["eligible"]
            used = blocks["used"]
            host = blocks["validation_host"]
            row |= {
                "minimum_hours": minimum_hours,
                "horizon_hours": horizon,
                "stride_hours": stride,
                "host_minimum_hours": host_minimum_hours,
                "limiting_models": requirements["limiting_models"],
                "requested_validation_hours": validation_len,
                "validation_reserve_hours": validation_reserve,
                "validation_forecasts": requirements["validation_forecasts"],
                "eligible_blocks": int(eligible.sum()),
                "host_candidates": int(blocks["host_capable"].sum()),
                "used_blocks": int(used.sum()),
                "unused_eligible_blocks": int((eligible & ~used).sum()),
                "eligible_hours": int(blocks.loc[eligible, "hours"].sum()),
                "effective_training_hours": int(blocks["training_hours"].sum()),
                "validation_hours": validation_reserve if host.any() else 0,
                "trainable": bool(host.any()),
            }
            row["retained_pct"] = (
                100.0 * row["effective_training_hours"] / row["observed_hours"]
                if row["observed_hours"]
                else 0.0
            )
            series_rows.append(row)

    blocks_df = pd.concat(block_frames, ignore_index=True) if block_frames else pd.DataFrame()
    return blocks_df, pd.DataFrame(series_rows), pd.DataFrame(excluded_rows)


def summarize_blocks(blocks: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the audit by pollutant and for all eligible series."""
    rows: list[dict[str, object]] = []
    groups = [(name, group) for name, group in blocks.groupby("pollutant", sort=False)]
    groups.append(("TOTAL", blocks))

    for pollutant, group in groups:
        selected_series = series if pollutant == "TOTAL" else series.loc[series["pollutant"] == pollutant]
        row: dict[str, object] = {
            "pollutant": pollutant,
            "forecast_models": selected_series["forecast_models"].iloc[0],
            "series": len(selected_series),
            "total_blocks": len(group),
            "observed_hours": int(group["hours"].sum()),
        }
        eligible = group["eligible"]
        used = group["used"]
        eligible_hours = int(group.loc[eligible, "hours"].sum())
        effective_hours = int(group["training_hours"].sum())
        validation = int(selected_series["validation_hours"].sum())
        row |= {
            "minimum_hours": int(selected_series["minimum_hours"].iloc[0]),
            "horizon_hours": int(selected_series["horizon_hours"].iloc[0]),
            "stride_hours": int(selected_series["stride_hours"].iloc[0]),
            "limiting_models": selected_series["limiting_models"].iloc[0],
            "requested_validation_hours": int(
                selected_series["requested_validation_hours"].iloc[0]
            ),
            "validation_reserve_hours": int(
                selected_series["validation_reserve_hours"].iloc[0]
            ),
            "validation_forecasts": int(
                selected_series["validation_forecasts"].iloc[0]
            ),
            "host_minimum_hours": int(selected_series["host_minimum_hours"].iloc[0]),
            "eligible_blocks": int(eligible.sum()),
            "used_blocks": int(used.sum()),
            "host_candidates": int(group["host_capable"].sum()),
            "unused_eligible_blocks": int((eligible & ~used).sum()),
            "too_short_blocks": int((~eligible).sum()),
            "eligible_hours": eligible_hours,
            "effective_training_hours": effective_hours,
            "validation_hours": validation,
            "unused_eligible_hours": eligible_hours - effective_hours - validation,
            "too_short_hours": int(group.loc[~eligible, "hours"].sum()),
            "retained_pct": (
                100.0 * effective_hours / row["observed_hours"]
                if row["observed_hours"]
                else 0.0
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(
    *,
    base_dir: str | Path,
    output_dir: str | Path,
    pollutants: tuple[str, ...] = DEFAULT_POLLUTANTS,
    context: int = 72,
    horizon: int = 12,
    stride: int = 6,
    validation_len: int = 48,
    holdout: int = 96,
    min_run: int = MIN_RUN,
    min_useful: int = MIN_USEFUL,
    forecast_models: tuple[str, ...] | None = None,
    seasonality_m: int | None = None,
) -> dict[str, Path]:
    """Run the pre-study and persist all tables and figures."""
    if forecast_models is None:
        forecast_models = cfg_get_csv_list(
            "forecasting", "forecast_models", ("NLinear", "TiDE")
        )
    if seasonality_m is None:
        seasonality_m = cfg_get_int("benchmark", "seasonality_m", 24)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output / "summary.csv",
        "series": output / "series_summary.csv",
        "blocks": output / "blocks.csv",
        "excluded": output / "excluded_series.csv",
    }
    blocks, series, excluded = analyze_raw_blocks(
        base_dir,
        pollutants,
        context=context,
        horizon=horizon,
        stride=stride,
        validation_len=validation_len,
        holdout=holdout,
        min_run=min_run,
        min_useful=min_useful,
        forecast_models=forecast_models,
        seasonality_m=seasonality_m,
    )
    if blocks.empty:
        pd.DataFrame(
            columns=["pollutant", "series", "total_blocks", "observed_hours"]
        ).to_csv(paths["summary"], index=False)
        series.reindex(columns=["pollutant", "station"]).to_csv(
            paths["series"], index=False
        )
        blocks.reindex(columns=["pollutant", "station", "start", "end", "hours"]).to_csv(
            paths["blocks"], index=False
        )
        excluded.to_csv(paths["excluded"], index=False)
        print(f"No hay series elegibles; diagnostico guardado en: {output}")
        return paths

    summary = summarize_blocks(blocks, series)
    summary.to_csv(paths["summary"], index=False)
    series.to_csv(paths["series"], index=False)
    blocks.to_csv(paths["blocks"], index=False)
    excluded.to_csv(paths["excluded"], index=False)
    from airquality.visualizations.block_analysis import save_figures

    for figure in save_figures(output, blocks, series, summary):
        paths[figure.stem] = figure

    block_table = summary[
        [
            "pollutant", "series", "total_blocks",
            "eligible_blocks", "used_blocks",
        ]
    ]
    hour_table = summary[
        [
            "pollutant", "observed_hours",
            "effective_training_hours", "retained_pct",
        ]
    ]
    print("\nBLOQUES\n" + block_table.to_markdown(index=False))
    print("\nHORAS\n" + hour_table.to_markdown(index=False, floatfmt=".2f"))
    if not excluded.empty:
        print("\nSERIES EXCLUIDAS\n" + excluded.to_markdown(index=False))
    print(f"\nInforme guardado en: {output}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze contiguous hourly blocks available for forecasting"
    )
    parser.add_argument("--base-dir", default="data/raw/datos_estaciones_5m")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pollutants", nargs="+", default=DEFAULT_POLLUTANTS)
    parser.add_argument(
        "--context", type=int, default=cfg_get_int("forecasting", "context_len", 72)
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=cfg_get_int("forecasting", "horizon", 12),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=cfg_get_int("forecasting", "stride", 6),
    )
    parser.add_argument(
        "--validation-len",
        type=int,
        default=cfg_get_int("forecasting", "validation_len", 48),
    )
    parser.add_argument(
        "--holdout", type=int, default=cfg_get_int("forecasting", "holdout", 96)
    )
    parser.add_argument("--min-run", type=int, default=MIN_RUN)
    parser.add_argument("--min-useful", type=int, default=MIN_USEFUL)
    parser.add_argument("--forecast-models", nargs="+", default=None)
    parser.add_argument("--seasonality-m", type=int, default=None)
    args = parser.parse_args()
    pollutants = tuple(args.pollutants)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else create_run_dir(
            Path("reports/data_blocks"),
            f"{_pollutant_run_label(pollutants)}_{datetime.now():%Y%m%d_%H%M%S}",
        )
    )
    run_analysis(
        base_dir=args.base_dir,
        output_dir=output_dir,
        pollutants=pollutants,
        context=args.context,
        horizon=args.horizon,
        stride=args.stride,
        validation_len=args.validation_len,
        holdout=args.holdout,
        min_run=args.min_run,
        min_useful=args.min_useful,
        forecast_models=(tuple(args.forecast_models) if args.forecast_models else None),
        seasonality_m=args.seasonality_m,
    )


if __name__ == "__main__":
    main()

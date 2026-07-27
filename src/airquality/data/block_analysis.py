"""Pre-study of contiguous hourly blocks available for forecasting models.

Run with::

    uv run python -m airquality.data.block_analysis

The command preprocesses the raw 5-minute NO2/CO files, reserves one complete
observed block for an exact 192-hour test, and audits rolling short/long
validation requirements. It writes CSV tables plus figures under
``reports/data_blocks/<timestamp>/`` without running detectors or models, so its
raw-support coverage is an upper bound for the detector-aware benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airquality.anomaly.presentation import (
    EDGE_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    TEXT_COLOR,
)
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

SHORT_COLOR = "#3d7ab5"
LONG_COLOR = "#cf6f1e"


def _worst_case_requirements(
    model_names: tuple[str, ...],
    *,
    context: int,
    horizons: dict[str, int],
    strides: dict[str, int],
    validation_hours: dict[str, int],
    seasonality_m: int,
) -> dict[str, dict[str, object]]:
    """Return conservative native geometry among configured forecast models."""
    if not model_names:
        raise ValueError("Debe configurarse al menos un modelo de forecasting")

    configs = resolve_forecasting_model_configs(
        list(model_names), seasonality_m=seasonality_m, context_length=context
    )
    out = {}
    for regime, horizon in horizons.items():
        strict = get_strict_forecast_requirements(
            configs,
            size_k=horizon,
            validation_len=validation_hours[regime],
            validation_stride=strides[regime],
            seasonality_m=seasonality_m,
            context_len=context,
        )
        out[regime] = {
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
    return out


def classify_blocks(
    blocks: pd.DataFrame,
    regimes: dict[str, int],
    validation_hours: dict[str, int],
    host_minimum_hours: dict[str, int],
) -> pd.DataFrame:
    """Mark eligible, validation-host, and chronologically usable blocks."""
    out = blocks.copy()
    out["block_number"] = np.arange(1, len(out) + 1)

    for regime, minimum in regimes.items():
        validation = validation_hours[regime]
        eligible = out["hours"].ge(minimum)
        host_capable = out["hours"].ge(host_minimum_hours[regime])
        out[f"{regime}_eligible"] = eligible
        out[f"{regime}_host_capable"] = host_capable
        out[f"{regime}_validation_host"] = False
        out[f"{regime}_used"] = False
        out[f"{regime}_training_hours"] = 0

        if not host_capable.any():
            continue

        host = int(out.index[host_capable][-1])
        used = eligible & out.index.to_series().le(host).to_numpy()
        out.loc[used, f"{regime}_used"] = True
        out.loc[used, f"{regime}_training_hours"] = out.loc[used, "hours"]
        out.loc[host, f"{regime}_validation_host"] = True
        out.loc[host, f"{regime}_training_hours"] -= validation

    return out


def analyze_raw_blocks(
    base_dir: str | Path,
    pollutants: tuple[str, ...],
    *,
    context: int,
    short_horizon: int,
    long_horizon: int,
    short_stride: int,
    long_stride: int,
    short_validation_len: int,
    long_validation_len: int,
    holdout: int,
    min_run: int,
    min_useful: int,
    forecast_models: tuple[str, ...],
    seasonality_m: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build per-block, per-series, and excluded-series audit tables."""
    regimes = {"short": context + short_horizon, "long": context + long_horizon}
    horizons = {"short": short_horizon, "long": long_horizon}
    strides = {"short": short_stride, "long": long_stride}
    validation_hours = {"short": short_validation_len, "long": long_validation_len}
    if holdout <= 0 or any(
        min(horizons[regime], strides[regime], validation_hours[regime]) <= 0
        or strides[regime] > horizons[regime]
        or validation_hours[regime] < horizons[regime]
        or (validation_hours[regime] - horizons[regime]) % strides[regime] != 0
        or holdout < horizons[regime]
        or (holdout - horizons[regime]) % strides[regime] != 0
        for regime in regimes
    ):
        raise ValueError(
            "Holdout, horizonte, stride y validacion de cada regimen deben ser validos"
        )
    requirements = _worst_case_requirements(
        forecast_models,
        context=context,
        horizons=horizons,
        strides=strides,
        validation_hours=validation_hours,
        seasonality_m=seasonality_m,
    )
    regimes = {
        regime: int(requirements[regime]["minimum_hours"]) for regime in requirements
    }
    host_minimum_hours = {
        regime: int(requirements[regime]["host_minimum_hours"])
        for regime in requirements
    }
    validation_reserves = {
        regime: int(requirements[regime]["validation_hours"])
        for regime in requirements
    }
    prediction_context = max(
        int(requirements[regime]["prediction_context_hours"])
        for regime in requirements
    )
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
                train_min_len=max(regimes.values()),
                validation_len=max(validation_hours.values()),
                host_min_len=max(host_minimum_hours.values()),
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
                regimes,
                validation_reserves,
                host_minimum_hours,
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
            for regime in regimes:
                eligible = blocks[f"{regime}_eligible"]
                used = blocks[f"{regime}_used"]
                host = blocks[f"{regime}_validation_host"]
                row |= {
                    f"{regime}_minimum_hours": regimes[regime],
                    f"{regime}_horizon_hours": horizons[regime],
                    f"{regime}_stride_hours": strides[regime],
                    f"{regime}_host_minimum_hours": host_minimum_hours[regime],
                    f"{regime}_limiting_models": requirements[regime]["limiting_models"],
                    f"{regime}_requested_validation_hours": validation_hours[regime],
                    f"{regime}_validation_reserve_hours": validation_reserves[regime],
                    f"{regime}_validation_forecasts": requirements[regime][
                        "validation_forecasts"
                    ],
                    f"{regime}_eligible_blocks": int(eligible.sum()),
                    f"{regime}_host_candidates": int(blocks[f"{regime}_host_capable"].sum()),
                    f"{regime}_used_blocks": int(used.sum()),
                    f"{regime}_unused_eligible_blocks": int((eligible & ~used).sum()),
                    f"{regime}_eligible_hours": int(blocks.loc[eligible, "hours"].sum()),
                    f"{regime}_effective_training_hours": int(
                        blocks[f"{regime}_training_hours"].sum()
                    ),
                    f"{regime}_validation_hours": (
                        validation_reserves[regime] if host.any() else 0
                    ),
                    f"{regime}_trainable": bool(host.any()),
                }
                row[f"{regime}_retained_pct"] = (
                    100.0 * row[f"{regime}_effective_training_hours"] / row["observed_hours"]
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
        for regime in ("short", "long"):
            eligible = group[f"{regime}_eligible"]
            used = group[f"{regime}_used"]
            eligible_hours = int(group.loc[eligible, "hours"].sum())
            effective_hours = int(group[f"{regime}_training_hours"].sum())
            validation = int(selected_series[f"{regime}_validation_hours"].sum())
            row |= {
                f"{regime}_minimum_hours": int(
                    selected_series[f"{regime}_minimum_hours"].iloc[0]
                ),
                f"{regime}_horizon_hours": int(
                    selected_series[f"{regime}_horizon_hours"].iloc[0]
                ),
                f"{regime}_stride_hours": int(
                    selected_series[f"{regime}_stride_hours"].iloc[0]
                ),
                f"{regime}_limiting_models": selected_series[
                    f"{regime}_limiting_models"
                ].iloc[0],
                f"{regime}_requested_validation_hours": int(
                    selected_series[f"{regime}_requested_validation_hours"].iloc[0]
                ),
                f"{regime}_validation_reserve_hours": int(
                    selected_series[f"{regime}_validation_reserve_hours"].iloc[0]
                ),
                f"{regime}_validation_forecasts": int(
                    selected_series[f"{regime}_validation_forecasts"].iloc[0]
                ),
                f"{regime}_host_minimum_hours": int(
                    selected_series[f"{regime}_host_minimum_hours"].iloc[0]
                ),
                f"{regime}_eligible_blocks": int(eligible.sum()),
                f"{regime}_used_blocks": int(used.sum()),
                f"{regime}_host_candidates": int(group[f"{regime}_host_capable"].sum()),
                f"{regime}_unused_eligible_blocks": int((eligible & ~used).sum()),
                f"{regime}_too_short_blocks": int((~eligible).sum()),
                f"{regime}_eligible_hours": eligible_hours,
                f"{regime}_effective_training_hours": effective_hours,
                f"{regime}_validation_hours": validation,
                f"{regime}_unused_eligible_hours": eligible_hours - effective_hours - validation,
                f"{regime}_too_short_hours": int(group.loc[~eligible, "hours"].sum()),
                f"{regime}_retained_pct": 100.0 * effective_hours / row["observed_hours"],
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _requirement_note(table: pd.DataFrame) -> str:
    row = (
        table.loc[table["pollutant"] == "TOTAL"].iloc[0]
        if "TOTAL" in table["pollutant"].values
        else table.iloc[0]
    )
    parts = [
        f"{regime} {int(row[f'{regime}_host_minimum_hours'])} h "
        f"({row[f'{regime}_limiting_models']})"
        for regime in ("short", "long")
    ]
    model_count = len(
        [name for name in str(row["forecast_models"]).split(",") if name.strip()]
    )
    return (
        f"Peor caso entre {model_count} modelos configurados: "
        + "; ".join(parts)
        + "."
    )


def _figure_header(figure: plt.Figure, title: str, subtitle: str) -> None:
    height = float(figure.get_size_inches()[1])
    figure.suptitle(
        title,
        x=0.06,
        y=1.0 - 0.10 / height,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.06, 1.0 - 0.48 / height, subtitle, ha="left", va="top", fontsize=9, color="#6d6258"
    )


def save_retention_overview(path: Path, summary: pd.DataFrame) -> None:
    """Contrast the small share of usable blocks with their large hour share."""
    total = summary.loc[summary["pollutant"] == "TOTAL"].iloc[0]
    regimes = ("short", "long")
    labels = []
    validation_labels = []
    for regime, display in (("short", "Short"), ("long", "Long")):
        minimum = int(total[f"{regime}_minimum_hours"])
        stride = int(total[f"{regime}_stride_hours"])
        host_minimum = int(total[f"{regime}_host_minimum_hours"])
        validation = int(total[f"{regime}_validation_reserve_hours"])
        forecasts = int(total[f"{regime}_validation_forecasts"])
        labels.append(
            f"{display}\ntrain mínimo nativo: {minimum} h\n"
            f"validación: +{validation} h, stride {stride} ({forecasts} ventanas; "
            f"anfitrión: {host_minimum} h)"
        )
        validation_labels.append(f"{display.lower()} {validation} h ({forecasts} ventanas)")
    block_counts = np.asarray([total[f"{regime}_used_blocks"] for regime in regimes], dtype=int)
    hour_counts = np.asarray(
        [total[f"{regime}_effective_training_hours"] for regime in regimes], dtype=int
    )
    block_pct = 100.0 * block_counts / float(total["total_blocks"])
    hour_pct = 100.0 * hour_counts / float(total["observed_hours"])

    figure, axis = plt.subplots(figsize=(10, 6.5), facecolor=FIGURE_FACE)
    x = np.arange(len(regimes))
    width = 0.32
    block_bars = axis.bar(
        x - width / 2,
        block_pct,
        width,
        color="#8c6d4b",
        edgecolor=EDGE_COLOR,
        label="Bloques usados",
    )
    hour_bars = axis.bar(
        x + width / 2,
        hour_pct,
        width,
        color="#5b8c5a",
        edgecolor=EDGE_COLOR,
        label="Horas retenidas",
    )
    for bars, percentages, counts, unit in (
        (block_bars, block_pct, block_counts, "bloques"),
        (hour_bars, hour_pct, hour_counts, "horas"),
    ):
        axis.bar_label(
            bars,
            labels=[
                f"{percentage:.1f}%\n{count:,.0f} {unit}".replace(",", ".")
                for percentage, count in zip(percentages, counts, strict=True)
            ],
            padding=5,
            fontsize=10,
            color=TEXT_COLOR,
        )

    axis.set_xticks(x, labels)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Porcentaje del total")
    axis.set_facecolor("#fffaf2")
    axis.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
    )
    _figure_header(
        figure,
        "Pocos bloques concentran la mayoría de las horas",
        "La proporción retenida ya descuenta validación por serie: "
        + " y ".join(validation_labels)
        + ".\n"
        + _requirement_note(summary),
    )
    figure.tight_layout(rect=(0.03, 0.14, 0.98, 0.89))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)


def save_block_length_distribution(path: Path, blocks: pd.DataFrame, series: pd.DataFrame) -> None:
    """Plot the highly skewed distribution of contiguous block lengths."""
    lengths = blocks["hours"].to_numpy(dtype=float)
    bins = np.geomspace(1, lengths.max() + 1, 45)
    short_min = int(series["short_minimum_hours"].iloc[0])
    long_min = int(series["long_minimum_hours"].iloc[0])
    short_host = int(series["short_host_minimum_hours"].iloc[0])
    long_host = int(series["long_host_minimum_hours"].iloc[0])

    figure, axis = plt.subplots(figsize=(11, 6), facecolor=FIGURE_FACE)
    axis.hist(lengths, bins=bins, color="#8c6d4b", edgecolor="#fffaf2", alpha=0.9)
    lines = (
        (short_min, SHORT_COLOR, f"Short train: {short_min} h"),
        (short_host, SHORT_COLOR, f"Short + validación: {short_host} h"),
        (long_min, LONG_COLOR, f"Long train: {long_min} h"),
        (long_host, LONG_COLOR, f"Long + validación: {long_host} h"),
    )
    for value, color, label in lines:
        axis.axvline(value, color=color, linestyle="--" if "validación" in label else "-", label=label)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Longitud del bloque continuo (horas, escala log)")
    axis.set_ylabel("Número de bloques (escala log)")
    axis.set_facecolor("#fffaf2")
    axis.grid(axis="both", color=GRID_COLOR, linestyle="--", alpha=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=2, loc="upper right")
    _figure_header(
        figure,
        "Distribución de longitudes",
        "Cada hueco rompe la serie; las líneas marcan los mínimos de entrenamiento y validación.\n"
        + _requirement_note(series),
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.98, 0.89))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)


def _save_series_comparison(
    path: Path,
    series: pd.DataFrame,
    *,
    short_column: str,
    long_column: str,
    title: str,
    subtitle: str,
    xlabel: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    pollutants = list(dict.fromkeys(series["pollutant"]))
    figure, axes = plt.subplots(
        1,
        len(pollutants),
        figsize=(16, max(9, 0.34 * series.groupby("pollutant").size().max())),
        facecolor=FIGURE_FACE,
        squeeze=False,
    )
    for axis, pollutant in zip(axes[0], pollutants, strict=True):
        subset = series.loc[series["pollutant"] == pollutant].sort_values(long_column)
        y = np.arange(len(subset))
        short = subset[short_column].to_numpy(dtype=float)
        long = subset[long_column].to_numpy(dtype=float)
        axis.hlines(y, long, short, color="#b8aa99", linewidth=1.2)
        axis.scatter(short, y, color=SHORT_COLOR, edgecolor=EDGE_COLOR, s=34, label="Short", zorder=3)
        axis.scatter(long, y, color=LONG_COLOR, edgecolor=EDGE_COLOR, s=34, label="Long", zorder=3)
        axis.set_yticks(y, subset["station"], fontsize=8)
        axis.set_title(pollutant, loc="left", fontweight="bold")
        axis.set_xlabel(xlabel)
        if xlim is not None:
            axis.set_xlim(*xlim)
        axis.set_facecolor("#fffaf2")
        axis.grid(axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
        axis.grid(axis="y", visible=False)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0, -1].legend(loc="lower right")
    _figure_header(figure, title, subtitle + "\n" + _requirement_note(series))
    figure.tight_layout(rect=(0.03, 0.02, 0.98, 0.91), w_pad=3)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)


def save_figures(output_dir: Path, blocks: pd.DataFrame, series: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    """Render the four figures in the block-analysis report."""
    paths = [
        output_dir / "retention_overview.png",
        output_dir / "block_length_distribution.png",
        output_dir / "retained_hours_by_series.png",
        output_dir / "usable_blocks_by_series.png",
    ]
    save_retention_overview(paths[0], summary)
    save_block_length_distribution(paths[1], blocks, series)
    _save_series_comparison(
        paths[2],
        series,
        short_column="short_retained_pct",
        long_column="long_retained_pct",
        title="Horas de entrenamiento retenidas por serie",
        subtitle="Porcentaje del historial observado anterior al holdout, después de reservar validación.",
        xlabel="Horas retenidas (%)",
        xlim=(0, 101),
    )
    _save_series_comparison(
        paths[3],
        series,
        short_column="short_used_blocks",
        long_column="long_used_blocks",
        title="Bloques utilizables por serie",
        subtitle="El prefijo del bloque anfitrión cuenta como train; los bloques posteriores se excluyen.",
        xlabel="Número de bloques efectivos",
    )
    return paths


def run_analysis(
    *,
    base_dir: str | Path,
    output_dir: str | Path,
    pollutants: tuple[str, ...] = ("NO2", "CO"),
    context: int = 72,
    short_horizon: int = 8,
    long_horizon: int = 48,
    short_stride: int = 4,
    long_stride: int = 24,
    short_validation_len: int = 48,
    long_validation_len: int = 96,
    holdout: int = 192,
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
        short_horizon=short_horizon,
        long_horizon=long_horizon,
        short_stride=short_stride,
        long_stride=long_stride,
        short_validation_len=short_validation_len,
        long_validation_len=long_validation_len,
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
    for figure in save_figures(output, blocks, series, summary):
        paths[figure.stem] = figure

    block_table = summary[
        [
            "pollutant", "series", "total_blocks",
            "short_eligible_blocks", "short_used_blocks",
            "long_eligible_blocks", "long_used_blocks",
        ]
    ]
    hour_table = summary[
        [
            "pollutant", "observed_hours",
            "short_effective_training_hours", "short_retained_pct",
            "long_effective_training_hours", "long_retained_pct",
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
        description="Analyze contiguous hourly blocks available for short/long forecasting"
    )
    parser.add_argument("--base-dir", default="data/raw/datos_estaciones_5m")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pollutants", nargs="+", default=["NO2", "CO"])
    parser.add_argument(
        "--context", type=int, default=cfg_get_int("forecasting", "context_len", 72)
    )
    parser.add_argument(
        "--short-horizon",
        type=int,
        default=cfg_get_int("forecasting", "short_horizon", 8),
    )
    parser.add_argument(
        "--long-horizon",
        type=int,
        default=cfg_get_int("forecasting", "long_horizon", 48),
    )
    parser.add_argument(
        "--short-stride",
        type=int,
        default=cfg_get_int("forecasting", "short_stride", 4),
    )
    parser.add_argument(
        "--long-stride",
        type=int,
        default=cfg_get_int("forecasting", "long_stride", 24),
    )
    parser.add_argument(
        "--short-validation-len",
        type=int,
        default=cfg_get_int("forecasting", "short_validation_len", 48),
    )
    parser.add_argument(
        "--long-validation-len",
        type=int,
        default=cfg_get_int("forecasting", "long_validation_len", 96),
    )
    parser.add_argument(
        "--holdout", type=int, default=cfg_get_int("forecasting", "holdout", 192)
    )
    parser.add_argument("--min-run", type=int, default=MIN_RUN)
    parser.add_argument("--min-useful", type=int, default=MIN_USEFUL)
    parser.add_argument("--forecast-models", nargs="+", default=None)
    parser.add_argument("--seasonality-m", type=int, default=None)
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else create_run_dir(
            Path("reports/data_blocks"), datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )
    run_analysis(
        base_dir=args.base_dir,
        output_dir=output_dir,
        pollutants=tuple(args.pollutants),
        context=args.context,
        short_horizon=args.short_horizon,
        long_horizon=args.long_horizon,
        short_stride=args.short_stride,
        long_stride=args.long_stride,
        short_validation_len=args.short_validation_len,
        long_validation_len=args.long_validation_len,
        holdout=args.holdout,
        min_run=args.min_run,
        min_useful=args.min_useful,
        forecast_models=(tuple(args.forecast_models) if args.forecast_models else None),
        seasonality_m=args.seasonality_m,
    )


if __name__ == "__main__":
    main()

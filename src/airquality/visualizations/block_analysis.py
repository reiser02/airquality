"""Render the figures produced by the contiguous-block data pre-study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airquality.visualizations.anomaly import (
    EDGE_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    TEXT_COLOR,
)


SHORT_COLOR = "#3d7ab5"
LONG_COLOR = "#cf6f1e"


def _requirement_note(table: pd.DataFrame) -> str:
    """Summarize the limiting model geometry shown in figure subtitles."""
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
    """Add the title and requirement summary above a block-analysis figure."""
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
        0.06,
        1.0 - 0.48 / height,
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color="#6d6258",
    )


def save_retention_overview(path: Path, summary: pd.DataFrame) -> None:
    """Contrast the share of usable blocks with their retained hour share."""
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
    block_counts = np.asarray(
        [total[f"{regime}_used_blocks"] for regime in regimes], dtype=int
    )
    hour_counts = np.asarray(
        [total[f"{regime}_effective_training_hours"] for regime in regimes],
        dtype=int,
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


def save_block_length_distribution(
    path: Path, blocks: pd.DataFrame, series: pd.DataFrame
) -> None:
    """Plot the distribution of contiguous block lengths and model minima."""
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
        axis.axvline(
            value,
            color=color,
            linestyle="--" if "validación" in label else "-",
            label=label,
        )

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
    """Compare short and long regime values for every pollutant series."""
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
        axis.scatter(
            short,
            y,
            color=SHORT_COLOR,
            edgecolor=EDGE_COLOR,
            s=34,
            label="Short",
            zorder=3,
        )
        axis.scatter(
            long,
            y,
            color=LONG_COLOR,
            edgecolor=EDGE_COLOR,
            s=34,
            label="Long",
            zorder=3,
        )
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


def save_figures(
    output_dir: Path,
    blocks: pd.DataFrame,
    series: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[Path]:
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

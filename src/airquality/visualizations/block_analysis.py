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


PROTOCOL_COLOR = "#3d7ab5"
HISTOGRAM_ALPHA = 0.55
POLLUTANT_COLORS = {
    "NO2": "#b46f44",
    "O3": "#4f8b8c",
}
POLLUTANT_FALLBACK_COLORS = ("#7b6aa6", "#c28e3f", "#6b8e62")


def _requirement_note(table: pd.DataFrame) -> str:
    """Summarize the limiting model geometry shown in figure subtitles."""
    row = (
        table.loc[table["pollutant"] == "TOTAL"].iloc[0]
        if "TOTAL" in table["pollutant"].values
        else table.iloc[0]
    )
    model_count = len(
        [name for name in str(row["forecast_models"]).split(",") if name.strip()]
    )
    return (
        f"Peor caso entre {model_count} modelos configurados: "
        f"{int(row['host_minimum_hours'])} h ({row['limiting_models']})"
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


def _summary_rows_for_plot(summary: pd.DataFrame) -> pd.DataFrame:
    """Return pollutant rows plus TOTAL only for multi-pollutant reports."""
    individual = summary.loc[summary["pollutant"] != "TOTAL"].copy()
    if individual.empty:
        return summary.loc[summary["pollutant"] == "TOTAL"].copy()
    if individual["pollutant"].nunique() > 1:
        total = summary.loc[summary["pollutant"] == "TOTAL"]
        if not total.empty:
            individual = pd.concat([individual, total], ignore_index=True)
    return individual.reset_index(drop=True)


def _pollutant_color(pollutant: str, index: int) -> str:
    """Return a stable histogram color for one pollutant panel."""
    if pollutant in POLLUTANT_COLORS:
        return POLLUTANT_COLORS[pollutant]
    return POLLUTANT_FALLBACK_COLORS[index % len(POLLUTANT_FALLBACK_COLORS)]


def save_retention_overview(path: Path, summary: pd.DataFrame) -> None:
    """Contrast usable blocks and retained hours separately per pollutant."""
    rows = _summary_rows_for_plot(summary)
    minimum = int(rows["minimum_hours"].iloc[0])
    stride = int(rows["stride_hours"].iloc[0])
    host_minimum = int(rows["host_minimum_hours"].iloc[0])
    validation = int(rows["validation_reserve_hours"].iloc[0])
    forecasts = int(rows["validation_forecasts"].iloc[0])
    labels = rows["pollutant"].astype(str).tolist()
    block_counts = rows["used_blocks"].to_numpy(dtype=int)
    hour_counts = rows["effective_training_hours"].to_numpy(dtype=int)
    block_totals = rows["total_blocks"].to_numpy(dtype=float)
    hour_totals = rows["observed_hours"].to_numpy(dtype=float)
    block_pct = np.divide(
        100.0 * block_counts,
        block_totals,
        out=np.zeros(len(rows), dtype=float),
        where=block_totals > 0,
    )
    hour_pct = np.divide(
        100.0 * hour_counts,
        hour_totals,
        out=np.zeros(len(rows), dtype=float),
        where=hour_totals > 0,
    )

    figure, axis = plt.subplots(
        figsize=(max(10, 2.8 * len(rows)), 6.5), facecolor=FIGURE_FACE
    )
    x = np.arange(len(rows))
    width = 0.34
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
    axis.set_ylabel("Porcentaje del total de cada contaminante")
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
        "Retención por contaminante",
        "Cada porcentaje se calcula dentro del contaminante mostrado; "
        f"validación: +{validation} h, stride {stride} ({forecasts} ventanas).\n"
        + _requirement_note(summary),
    )
    figure.tight_layout(rect=(0.03, 0.14, 0.98, 0.89))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)


def save_block_length_distribution(
    path: Path, blocks: pd.DataFrame, series: pd.DataFrame
) -> None:
    """Plot all pollutant block-length distributions on one colored axis."""
    pollutants = list(dict.fromkeys(blocks["pollutant"].astype(str)))
    lengths = blocks["hours"].to_numpy(dtype=float)
    bins = np.geomspace(1, lengths.max() + 1, 45)
    figure, axis = plt.subplots(figsize=(11, 6), facecolor=FIGURE_FACE)
    thresholds: set[tuple[int, int]] = set()
    for index, pollutant in enumerate(pollutants):
        pollutant_blocks = blocks.loc[blocks["pollutant"].eq(pollutant)]
        pollutant_series = series.loc[series["pollutant"].eq(pollutant)]
        minimum = int(pollutant_series["minimum_hours"].iloc[0])
        host_minimum = int(pollutant_series["host_minimum_hours"].iloc[0])
        thresholds.add((minimum, host_minimum))
        axis.hist(
            pollutant_blocks["hours"].to_numpy(dtype=float),
            bins=bins,
            color=_pollutant_color(pollutant, index),
            edgecolor=_pollutant_color(pollutant, index),
            linewidth=0.7,
            alpha=HISTOGRAM_ALPHA,
            label=pollutant,
        )
    for minimum, host_minimum in sorted(thresholds):
        axis.axvline(
            minimum,
            color=PROTOCOL_COLOR,
            linestyle="-",
            label=f"Train: {minimum} h",
        )
        axis.axvline(
            host_minimum,
            color=PROTOCOL_COLOR,
            linestyle="--",
            label=f"Train + validación: {host_minimum} h",
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Longitud del bloque (horas, escala log)")
    axis.set_ylabel("Número de bloques (escala log)")
    axis.set_facecolor("#fffaf2")
    axis.grid(axis="both", color=GRID_COLOR, linestyle="--", alpha=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=2, loc="upper right")
    _figure_header(
        figure,
        "Distribución de longitudes",
        "Los colores distinguen los contaminantes; cada hueco rompe la serie y las líneas "
        "marcan los mínimos de entrenamiento y validación.\n"
        + _requirement_note(series),
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.98, 0.89))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)


def _save_series_comparison(
    path: Path,
    series: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    subtitle: str,
    xlabel: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    """Compare one protocol value for every pollutant series."""
    pollutants = list(dict.fromkeys(series["pollutant"]))
    figure, axes = plt.subplots(
        1,
        len(pollutants),
        figsize=(16, max(9, 0.34 * series.groupby("pollutant").size().max())),
        facecolor=FIGURE_FACE,
        squeeze=False,
    )
    for axis, pollutant in zip(axes[0], pollutants, strict=True):
        subset = series.loc[series["pollutant"] == pollutant].sort_values(value_column)
        y = np.arange(len(subset))
        axis.scatter(
            subset[value_column].to_numpy(dtype=float),
            y,
            color=PROTOCOL_COLOR,
            edgecolor=EDGE_COLOR,
            s=34,
            label="Protocolo",
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
        value_column="retained_pct",
        title="Horas de entrenamiento retenidas por serie",
        subtitle="Porcentaje del historial observado anterior al holdout, después de reservar validación.",
        xlabel="Horas retenidas (%)",
        xlim=(0, 101),
    )
    _save_series_comparison(
        paths[3],
        series,
        value_column="used_blocks",
        title="Bloques utilizables por serie",
        subtitle="El prefijo del bloque anfitrión cuenta como train; los bloques posteriores se excluyen.",
        xlabel="Número de bloques efectivos",
    )
    return paths

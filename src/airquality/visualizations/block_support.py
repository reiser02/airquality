"""Render raw, detection, and imputation block-support reports from CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airquality.visualizations.anomaly import (
    EDGE_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    TEXT_COLOR,
    style_axis,
)


def _repo_root() -> Path:
    """Return the repository root used to discover persisted report runs."""
    return Path(__file__).resolve().parents[3]

RAW_COLOR = "#6d6258"
DETECTED_COLOR = "#3d7ab5"
IMPUTED_COLOR = "#cf6f1e"
RECOVERED_COLOR = "#5b8c5a"
PLOT_STRATEGY = "inject-vote"
AGE_BINS = (-1, 30, 90, 180, 365, 730, np.inf)
AGE_LABELS = (
    "0-30 d",
    "31-90 d",
    "91-180 d",
    "181-365 d",
    "1-2 años",
    ">2 años",
)
AGE_SUPPORT_TYPES = ("imputed", "recovered")
AGE_SUPPORT_LABELS = {
    "imputed": "Horas rellenadas por el imputador",
    "recovered": "Observaciones reales recuperadas",
}
AGE_SUPPORT_COLORS = {"imputed": IMPUTED_COLOR, "recovered": RECOVERED_COLOR}

SERIES_REQUIRED = {
    "series",
    "test_target_start",
    "arm",
    "stage",
    "strategy",
    "minimum_hours",
    "host_minimum_hours",
    "observed_hours",
    "total_blocks",
    "valid_blocks",
    "valid_hours",
    "valid_real_hours",
    "valid_imputed_hours",
    "valid_imputed_anomaly_hours",
    "valid_imputed_preexisting_gap_hours",
    "effective_imputed_hours",
    "imputation_gain_effective_hours",
    "valid_hours_pct_raw",
    "imputation_gain_valid_hours",
    "imputed_valid_age_hours_median",
    "used_blocks",
    "effective_training_hours",
}

PLOT_GUIDE = """## Figuras

- Todas las comparaciones de deteccion e imputacion usan exclusivamente la estrategia `inject-vote`; `raw` se conserva como referencia.
- `retention_overview.png`: resume el historial raw. Muestra el porcentaje y la cantidad absoluta de bloques usados y horas efectivas de entrenamiento; las horas ya descuentan la reserva de validación.
- `block_length_distribution.png`: distribución de longitudes de los bloques raw. Las líneas verticales marcan el mínimo de entrenamiento y el mínimo del bloque anfitrión que además debe alojar la validación.
- `retention_overview_{detected,imputed}.png`: aplica el mismo resumen de retencion al historial tras deteccion `inject-vote`, sin y con imputacion.
- `block_length_distribution_{detected,imputed}.png`: aplica la misma distribucion de longitudes a los bloques resultantes de `inject-vote`, sin y con imputacion.
- `retained_hours_by_series_{detected,imputed}.png`: porcentaje de horas efectivas retenidas por serie tras `inject-vote`, sin y con imputacion.
- `usable_blocks_by_series_{detected,imputed}.png`: bloques utilizados por serie tras `inject-vote`, sin y con imputacion.
- `detected_block_length_distribution.png`: muestra, por estrategia, dos paneles consecutivos: raw frente a detección y detección frente a imputación. El panel derecho incorpora el resumen numérico efectivo de `gap_recovery`.
- `support_overview.png`: horas observadas e imputadas dentro de bloques válidos para cada brazo. La etiqueta indica también cuántos bloques alcanzan el mínimo del protocolo.
- `support_by_series.png`: soporte válido de cada brazo respecto a raw para cada serie. Un valor de 100 conserva el mismo número de horas; menos de 100 pierde soporte y más de 100 lo amplía.
- `valid_blocks_by_series.png`: cantidad absoluta de bloques que alcanzan el mínimo del protocolo en cada serie y brazo. Más bloques no implica necesariamente más horas, porque sus longitudes difieren.
- `block_length_survival.png`: para cada longitud del eje X muestra cuántos bloques tienen al menos esa duración. Permite ver cómo detección e imputación fragmentan o conectan el historial.
- `gap_recovery.png`: parte de las horas efectivas antes de imputar y añade las horas imputadas y las horas observadas desbloqueadas dentro de los bloques usados; la etiqueta muestra el total efectivo final.
- `detection_strategy_summary.png`: muestra qué porcentaje del historial pudo puntuar `inject-vote` y qué porcentaje terminó retirando como anomalía. Una hora recibe score cuando hay suficientes salidas válidas de detectores para clasificarla.
- `imputation_age.png`: distribuye por tramos de antigüedad la ganancia efectiva de la imputación, separando las horas rellenadas de las observaciones reales que pasan a formar segmentos suficientemente largos.
"""


def _arm_order(table: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(table["arm"].astype(str)))


def _stage_rows(table: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Select raw or the configured plotting strategy at one processing stage."""
    if stage == "raw":
        return table.loc[table["arm"] == "raw"]
    return table.loc[
        (table["stage"] == stage) & (table["strategy"] == PLOT_STRATEGY)
    ]


def _stage_description(stage: str) -> str:
    if stage == "detected":
        return f"tras deteccion {PLOT_STRATEGY}"
    if stage == "imputed":
        return f"tras deteccion {PLOT_STRATEGY} e imputacion"
    return "raw"


def _stage_color(stage: str) -> str:
    return {
        "raw": RAW_COLOR,
        "detected": DETECTED_COLOR,
        "imputed": IMPUTED_COLOR,
    }[stage]


def _figure_header(
    figure: plt.Figure, title: str, subtitle: str, pollutant: str
) -> None:
    height = float(figure.get_size_inches()[1])
    figure.suptitle(
        f"{pollutant}: {title}",
        x=0.04,
        y=1.0 - 0.08 / height,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.04,
        1.0 - 0.45 / height,
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=RAW_COLOR,
    )


def _save_retention_overview(
    path: Path, table: pd.DataFrame, pollutant: str, *, stage: str = "raw"
) -> bool:
    selected = _stage_rows(table, stage)
    if selected.empty:
        return False
    total_blocks = int(selected["total_blocks"].sum())
    used_blocks = int(selected["used_blocks"].sum())
    observed_hours = int(selected["observed_hours"].sum())
    effective_hours = int(selected["effective_training_hours"].sum())
    minimum_hours = int(selected["minimum_hours"].iloc[0])
    host_minimum_hours = int(selected["host_minimum_hours"].iloc[0])
    block_pct = np.asarray([100.0 * used_blocks / total_blocks])
    hour_pct = np.asarray([100.0 * effective_hours / observed_hours])
    positions = np.arange(1)
    width = 0.32

    figure, axis = plt.subplots(figsize=(10, 6.5), facecolor=FIGURE_FACE)
    style_axis(axis)
    block_bars = axis.bar(
        positions - width / 2,
        block_pct,
        width,
        color="#8c6d4b",
        edgecolor=EDGE_COLOR,
        label="Bloques usados",
    )
    hour_bars = axis.bar(
        positions + width / 2,
        hour_pct,
        width,
        color=RECOVERED_COLOR if stage == "raw" else _stage_color(stage),
        edgecolor=EDGE_COLOR,
        label="Horas efectivas",
    )
    for bars, percentages, counts, unit in (
        (block_bars, block_pct, np.asarray([used_blocks]), "bloques"),
        (hour_bars, hour_pct, np.asarray([effective_hours]), "horas"),
    ):
        axis.bar_label(
            bars,
            labels=[
                f"{percentage:.1f}%\n{int(count):,} {unit}".replace(",", ".")
                for percentage, count in zip(percentages, counts, strict=True)
            ],
            padding=5,
            fontsize=10,
            color=TEXT_COLOR,
        )
    axis.set_xticks(
        positions,
        [
            f"{_stage_description(stage)}\ntrain >= {minimum_hours} h; "
            f"anfitrión >= {host_minimum_hours} h"
        ],
    )
    axis.set_ylim(0, 105)
    axis.set_ylabel(f"Porcentaje del historial {_stage_description(stage)}")
    axis.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.6)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    _figure_header(
        figure,
        f"Retención del historial {_stage_description(stage)} para entrenamiento",
        "Compara cuántos bloques se usan y cuántas horas efectivas conserva el protocolo; "
        "las horas descuentan la reserva de validación.",
        pollutant,
    )
    figure.tight_layout(rect=(0.03, 0.12, 0.98, 0.86))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_block_distribution(
    path: Path,
    blocks: pd.DataFrame,
    table: pd.DataFrame,
    pollutant: str,
    *,
    stage: str = "raw",
) -> bool:
    selected_blocks = _stage_rows(blocks, stage)
    selected_table = _stage_rows(table, stage)
    if selected_blocks.empty or selected_table.empty:
        return False
    lengths = selected_blocks["hours"].to_numpy(dtype=float)
    bins = np.geomspace(1, lengths.max() + 1, 45)
    minimum_hours = int(selected_table["minimum_hours"].iloc[0])
    host_minimum_hours = int(selected_table["host_minimum_hours"].iloc[0])

    figure, axis = plt.subplots(figsize=(11, 6), facecolor=FIGURE_FACE)
    style_axis(axis)
    axis.hist(
        lengths,
        bins=bins,
        color="#8c6d4b" if stage == "raw" else _stage_color(stage),
        edgecolor="#fffaf2",
        alpha=0.9,
    )
    axis.axvline(
        minimum_hours,
        color=DETECTED_COLOR,
        label=f"Train: {minimum_hours} h",
    )
    axis.axvline(
        host_minimum_hours,
        color=DETECTED_COLOR,
        linestyle="--",
        label=f"Train + validación: {host_minimum_hours} h",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(
        f"Longitud del bloque {_stage_description(stage)} (horas, escala log)"
    )
    axis.set_ylabel("Número de bloques (escala log)")
    axis.grid(True, which="major", color=GRID_COLOR, linestyle="--", alpha=0.55)
    axis.legend(ncols=2, loc="upper right", fontsize=8)
    _figure_header(
        figure,
        f"Distribución de bloques del historial {_stage_description(stage)}",
        "Cada hueco temporal o valor ausente rompe un bloque; las líneas marcan los mínimos "
        "de entrenamiento y de bloque anfitrión del protocolo único.",
        pollutant,
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.98, 0.86))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_series_comparison(
    path: Path,
    table: pd.DataFrame,
    pollutant: str,
    *,
    stage: str,
    value_column: str,
    title: str,
    subtitle: str,
    xlabel: str,
    xlim: tuple[float, float] | None = None,
) -> bool:
    """Compare one inject-vote support value across the retained series."""
    selected = _stage_rows(table, stage).copy()
    if selected.empty:
        return False
    if value_column == "retained_pct":
        observed = selected["observed_hours"].to_numpy(dtype=float)
        effective = selected["effective_training_hours"].to_numpy(dtype=float)
        selected[value_column] = np.divide(
            100.0 * effective,
            observed,
            out=np.full(len(selected), np.nan),
            where=observed > 0,
        )
    selected = selected.sort_values(value_column)
    positions = np.arange(len(selected))
    figure, axis = plt.subplots(
        figsize=(11, max(7, 2.8 + 0.34 * len(selected))), facecolor=FIGURE_FACE
    )
    style_axis(axis)
    axis.scatter(
        selected[value_column].to_numpy(dtype=float),
        positions,
        color=_stage_color(stage),
        edgecolor=EDGE_COLOR,
        s=34,
        label=selected["arm"].iloc[0],
        zorder=3,
    )
    axis.set_yticks(positions, selected["series"], fontsize=8)
    axis.set_xlabel(xlabel)
    if xlim is not None:
        axis.set_xlim(*xlim)
    axis.grid(axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(axis="y", visible=False)
    axis.legend(loc="lower right")
    _figure_header(figure, title, subtitle, pollutant)
    figure.tight_layout(rect=(0.03, 0.02, 0.98, 0.88))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_detected_block_distribution(
    path: Path, blocks: pd.DataFrame, table: pd.DataFrame, pollutant: str
) -> bool:
    raw = blocks.loc[blocks["arm"] == "raw", "hours"].to_numpy(dtype=float)
    detected = blocks.loc[blocks["stage"] == "detected"]
    strategies = list(dict.fromkeys(detected["strategy"].astype(str)))
    if not len(raw) or not strategies:
        return False
    maximum = max(float(blocks["hours"].max()), 2.0)
    bins = np.geomspace(1, maximum + 1, 45)
    minimum = int(table.loc[table["arm"] == "raw", "minimum_hours"].iloc[0])

    figure, axes = plt.subplots(
        2,
        2 * len(strategies),
        figsize=(10.4 * len(strategies), 7),
        facecolor=FIGURE_FACE,
        squeeze=False,
        sharex=True,
        sharey=True,
        gridspec_kw={"height_ratios": (1, 0.2), "hspace": 0.06},
    )
    for axis in axes[1]:
        axis.axis("off")
    for strategy_index, strategy in enumerate(strategies):
        stages = {
            "detected": detected.loc[
                detected["strategy"] == strategy, "hours"
            ].to_numpy(dtype=float),
            "imputed": blocks.loc[
                (blocks["stage"] == "imputed")
                & (blocks["strategy"] == strategy),
                "hours",
            ].to_numpy(dtype=float),
        }
        for stage_index, stage in enumerate(("detected", "imputed")):
            axis = axes[0, 2 * strategy_index + stage_index]
            style_axis(axis)
            lengths = stages[stage]
            color = DETECTED_COLOR if stage == "detected" else IMPUTED_COLOR
            stage_label = "Tras detección" if stage == "detected" else "Tras imputación"
            reference = raw if stage == "detected" else stages["detected"]
            reference_color = RAW_COLOR if stage == "detected" else DETECTED_COLOR
            reference_label = "Raw" if stage == "detected" else "Tras detección"
            if len(lengths):
                axis.hist(
                    lengths,
                    bins=bins,
                    color=color,
                    edgecolor="#fffaf2",
                    alpha=0.8,
                    label=stage_label,
                )
            axis.hist(
                reference,
                bins=bins,
                histtype="step",
                color=reference_color,
                linewidth=1.8,
                label=reference_label,
            )
            axis.axvline(
                minimum,
                color=RECOVERED_COLOR,
                linestyle="--",
                label=f"Mínimo: {minimum} h",
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("Longitud del bloque (h, escala log)")
            axis.set_title(f"{strategy} · {stage_label}", loc="left", fontweight="bold", color=TEXT_COLOR)
            axis.grid(True, which="major", color=GRID_COLOR, linestyle="--", alpha=0.55)
            if stage == "imputed":
                recovery = _gap_recovery_totals(table)
                arm = f"{strategy}+impute"
                if arm in recovery.index:
                    row = recovery.loc[arm]
                    summary = (
                        "Balance de soporte efectivo\n"
                        f"Base entrenable tras detección: {int(row['valid_before']):,} h\n"
                        f"Relleno hecho por el imputador: {int(row['imputed_valid']):,} h\n"
                        "Observaciones reales que pasan a bloques utilizables: "
                        f"{int(row['observed_unlocked']):,} h\n"
                        f"Total entrenable final: {int(row['effective_training_hours']):,} h"
                    ).replace(",", ".")
                    axes[1, 2 * strategy_index + stage_index].text(
                        0.78,
                        0.5,
                        summary,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=TEXT_COLOR,
                        bbox={
                            "boxstyle": "round,pad=0.35",
                            "facecolor": "#fffaf2",
                            "edgecolor": EDGE_COLOR,
                            "alpha": 0.9,
                        },
                    )
    axes[0, 0].set_ylabel("Número de bloques (escala log)")
    handles: list[object] = []
    labels: list[str] = []
    for axis in axes[0]:
        current_handles, current_labels = axis.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="lower center", ncols=4, fontsize=8)
    _figure_header(
        figure,
        "Evolución de bloques: detección e imputación",
        "Raw frente a detección a la izquierda; detección frente a imputación a la derecha. "
        "Ejes logarítmicos compartidos.",
        pollutant,
    )
    figure.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.12,
        top=0.8,
        wspace=0.06,
        hspace=0.35,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_support_overview(
    path: Path, table: pd.DataFrame, pollutant: str
) -> bool:
    if table.empty:
        return False
    totals = table.groupby("arm", sort=False).agg(
        valid_real_hours=("valid_real_hours", "sum"),
        valid_imputed_hours=("valid_imputed_hours", "sum"),
        valid_blocks=("valid_blocks", "sum"),
    )
    arms = _arm_order(table)
    figure, axes = plt.subplots(
        1, figsize=(7.2, 3.8 + 0.48 * len(arms)),
        facecolor=FIGURE_FACE, squeeze=False,
    )
    axis = axes[0, 0]
    style_axis(axis)
    selected = totals.reindex(arms)
    positions = np.arange(len(arms))
    real = selected["valid_real_hours"].fillna(0).to_numpy(dtype=float)
    imputed = selected["valid_imputed_hours"].fillna(0).to_numpy(dtype=float)
    axis.barh(positions, real, color=RAW_COLOR, edgecolor=EDGE_COLOR, label="Observado")
    axis.barh(
        positions, imputed, left=real, color=IMPUTED_COLOR,
        edgecolor=EDGE_COLOR, label="Imputado",
    )
    for position, total, blocks in zip(
        positions,
        real + imputed,
        selected["valid_blocks"].fillna(0).to_numpy(dtype=int),
        strict=True,
    ):
        axis.text(
            total, position, f"  {int(total):,} h | {blocks} bloques",
            va="center", fontsize=8, color=TEXT_COLOR,
        )
    minimum = int(table["minimum_hours"].iloc[0])
    axis.set_yticks(positions, arms, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Horas en bloques validos")
    axis.set_title(
        f"Protocolo (minimo {minimum} h)",
        loc="left", fontweight="bold", color=TEXT_COLOR,
    )
    axis.grid(axis="x", color=GRID_COLOR, linestyle="--", alpha=0.6)
    axis.margins(x=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=2)
    _figure_header(
        figure,
        "Soporte válido después de detección e imputación",
        "Horas observadas e imputadas que pertenecen a bloques suficientemente largos; "
        "cada etiqueta añade la cantidad de bloques válidos.",
        pollutant,
    )
    figure.tight_layout(rect=(0.02, 0.08, 1, 0.88))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_matrix(
    path: Path,
    table: pd.DataFrame,
    *,
    value: str,
    title: str,
    label: str,
    fmt: str,
    subtitle: str,
    pollutant: str,
    exclude_raw: bool = False,
) -> bool:
    selected = table.loc[table["arm"] != "raw"] if exclude_raw else table
    if selected.empty:
        return False
    arms = _arm_order(selected)
    series = list(dict.fromkeys(selected["series"]))
    finite = pd.to_numeric(selected[value], errors="coerce").dropna()
    color_min = float(finite.min()) if not finite.empty else 0.0
    color_max = float(finite.max()) if not finite.empty else 1.0
    if color_min == color_max:
        color_max = color_min + 1.0
    figure, axes = plt.subplots(
        1,
        figsize=(max(16, 2.0 * len(arms) + 7), 3.4 + 0.42 * len(series)),
        facecolor=FIGURE_FACE, squeeze=False,
    )
    axis = axes[0, 0]
    values = selected.pivot(index="series", columns="arm", values=value).reindex(
        index=series, columns=arms
    )
    array = values.to_numpy(dtype=float)
    image = axis.imshow(
        array,
        aspect="auto",
        cmap="YlGnBu",
        vmin=color_min,
        vmax=color_max,
    )
    for row, col in np.ndindex(array.shape):
        if np.isfinite(array[row, col]):
            axis.text(col, row, format(array[row, col], fmt), ha="center", va="center", fontsize=7)
    axis.set_xticks(np.arange(len(arms)), arms, rotation=35, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(series)), series, fontsize=8)
    axis.set_title("Protocolo", loc="left", fontweight="bold", color=TEXT_COLOR)
    axis.set_facecolor("#fffaf2")
    color_axis = figure.add_axes((0.925, 0.23, 0.015, 0.52))
    figure.colorbar(image, cax=color_axis, label=label)
    _figure_header(figure, title, subtitle, pollutant)
    figure.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.84, wspace=0.08)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_block_survival(
    path: Path, blocks: pd.DataFrame, table: pd.DataFrame, pollutant: str
) -> bool:
    if blocks.empty:
        return False
    arms = _arm_order(table)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(arms)))
    maximum = max(int(blocks["hours"].max()), 2)
    x = np.unique(np.geomspace(1, maximum, 120).astype(int))
    figure, axes = plt.subplots(
        1, figsize=(7, 5.5),
        facecolor=FIGURE_FACE, squeeze=False,
    )
    axis = axes[0, 0]
    style_axis(axis)
    for arm, color in zip(arms, colors, strict=True):
        lengths = blocks.loc[blocks["arm"] == arm, "hours"].to_numpy(dtype=int)
        axis.plot(x, [(lengths >= value).sum() for value in x], label=arm, color=color)
    minimum = int(table["minimum_hours"].iloc[0])
    axis.axvline(
        minimum,
        color="#bd3b37",
        linestyle="--",
        label="Mínimo del protocolo",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Longitud mínima del bloque (h)")
    axis.set_ylabel("Bloques con al menos esa longitud")
    axis.set_title("Protocolo", loc="left", fontweight="bold")
    axis.grid(True, which="major", color=GRID_COLOR, linestyle="--", alpha=0.55)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=min(4, len(labels)), fontsize=8)
    _figure_header(
        figure,
        "Cuántos bloques sobreviven a cada longitud mínima",
        "Cada curva cuenta bloques con al menos la duración del eje X; la línea roja marca "
        "el mínimo exigido por el protocolo.",
        pollutant,
    )
    figure.tight_layout(rect=(0.02, 0.12, 1, 0.86))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _gap_recovery_totals(table: pd.DataFrame) -> pd.DataFrame:
    """Reconcile effective training support before and after imputation by arm."""
    selected = table.loc[table["stage"] == "imputed"].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["valid_before"] = (
        selected["effective_training_hours"]
        - selected["imputation_gain_effective_hours"]
    )
    selected["imputed_valid"] = selected["effective_imputed_hours"]
    selected["observed_unlocked"] = (
        selected["imputation_gain_effective_hours"]
        - selected["effective_imputed_hours"]
    )
    components = ["valid_before", "imputed_valid", "observed_unlocked"]
    if (selected[components] < 0).any().any():
        raise ValueError("La recuperación de soporte contiene componentes negativos")
    accounted = selected[components].sum(axis=1)
    if not np.allclose(accounted, selected["effective_training_hours"]):
        raise ValueError(
            "La recuperación de soporte no cuadra con el total efectivo final"
        )
    return selected.groupby("arm", sort=False)[
        components + ["effective_training_hours"]
    ].sum()


def _save_gap_recovery(path: Path, table: pd.DataFrame, pollutant: str) -> bool:
    totals = _gap_recovery_totals(table)
    if totals.empty:
        return False
    figure, axes = plt.subplots(
        1, figsize=(7, 5.5),
        facecolor=FIGURE_FACE, squeeze=False,
    )
    axis = axes[0, 0]
    style_axis(axis)
    current = totals
    positions = np.arange(len(current))
    bottom = np.zeros(len(current))
    for column, label, color in (
        ("valid_before", "Base entrenable tras detección", DETECTED_COLOR),
        ("imputed_valid", "Puntos rellenados por el imputador", IMPUTED_COLOR),
        (
            "observed_unlocked",
            "Observaciones reales en bloques utilizables",
            RECOVERED_COLOR,
        ),
    ):
        values = current[column].to_numpy(dtype=float)
        bars = axis.bar(
            positions,
            values,
            bottom=bottom,
            color=color,
            edgecolor=EDGE_COLOR,
            label=label,
        )
        axis.bar_label(
            bars,
            labels=[
                f"{int(value):,} h".replace(",", ".") if value > 0 else ""
                for value in values
            ],
            label_type="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
        bottom += values
    for position, (_, row) in zip(positions, current.iterrows(), strict=True):
        valid_after = float(row["effective_training_hours"])
        if valid_after <= 0:
            continue
        axis.text(
            position,
            valid_after,
            f"{int(valid_after):,} h efectivas después".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT_COLOR,
        )
    axis.set_xticks(positions, current.index, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel("Puntos horarios efectivos de entrenamiento")
    axis.set_title("Protocolo", loc="left", fontweight="bold")
    axis.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.55)
    axis.margins(y=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=3, fontsize=8)
    _figure_header(
        figure,
        "Cómo la imputación convierte segmentos en soporte efectivo",
        "Solo cuenta los bloques usados antes del holdout y descuenta la reserva de "
        "validación; separa puntos imputados de observaciones desbloqueadas.",
        pollutant,
    )
    figure.tight_layout(rect=(0.02, 0.13, 1, 0.86))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_detection_summary(
    path: Path, detection: pd.DataFrame, pollutant: str
) -> bool:
    if detection.empty:
        return False
    totals = detection.groupby("strategy", sort=False).agg(
        observed=("observed_hours", "sum"),
        scored=("scored_hours", "sum"),
        flagged=("flagged_hours_full", "sum"),
    )
    totals["coverage_pct"] = 100.0 * totals["scored"] / totals["observed"]
    totals["detection_pct"] = 100.0 * totals["flagged"] / totals["observed"]
    positions = np.arange(len(totals))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=FIGURE_FACE)
    for axis in axes:
        style_axis(axis)
        axis.set_xticks(positions, totals.index, rotation=25, ha="right")
        axis.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.55)
    coverage = axes[0].bar(positions, totals["coverage_pct"], color=DETECTED_COLOR, edgecolor=EDGE_COLOR)
    axes[0].bar_label(coverage, fmt="%.1f%%", padding=3)
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Cobertura", loc="left", fontweight="bold")
    rates = axes[1].bar(positions, totals["detection_pct"], color="#bd3b37", edgecolor=EDGE_COLOR)
    axes[1].bar_label(rates, fmt="%.2f%%", padding=3)
    axes[1].set_title("Observaciones retiradas", loc="left", fontweight="bold")
    _figure_header(
        figure,
        f"Cobertura y agresividad de la estrategia {PLOT_STRATEGY}",
        "Una hora recibe score cuando hay suficientes salidas válidas de detectores para "
        "clasificarla; gaps, bloques demasiado cortos y fallos quedan sin score.",
        pollutant,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.84))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _effective_imputation_age_points(
    table: pd.DataFrame, blocks: pd.DataFrame, gaps: pd.DataFrame
) -> pd.DataFrame:
    """Locate every imputed hour that remains in effective training support."""
    selected = table.loc[
        (table["stage"] == "imputed")
        & (table["strategy"] == PLOT_STRATEGY)
        & table["imputed_valid_age_hours_median"].notna()
    ].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=["series", "timestamp", "age_days", "age_bin", "support_type"]
        )

    support_blocks = blocks.loc[
        (blocks["strategy"] == PLOT_STRATEGY) & blocks["used"].astype(bool)
    ].copy()
    selected_gaps = gaps.loc[
        (gaps["strategy"] == PLOT_STRATEGY)
        & gaps["fully_filled"].astype(bool)
    ].copy()
    selected["test_target_start"] = pd.to_datetime(selected["test_target_start"])
    for frame in (support_blocks, selected_gaps):
        frame["start"] = pd.to_datetime(frame["start"])
        frame["end"] = pd.to_datetime(frame["end"])

    rows: list[dict[str, object]] = []
    for summary in selected.itertuples(index=False):
        def effective_index(stage: str) -> pd.DatetimeIndex:
            effective: list[pd.Timestamp] = []
            station_blocks = support_blocks.loc[
                (support_blocks["series"] == summary.series)
                & (support_blocks["stage"] == stage)
            ]
            for block in station_blocks.itertuples(index=False):
                end = block.end
                if bool(block.validation_host):
                    end -= pd.Timedelta(hours=int(summary.validation_hours))
                if end >= block.start:
                    effective.extend(pd.date_range(block.start, end, freq="h"))
            return pd.DatetimeIndex(effective).unique()

        before = effective_index("detected")
        after = effective_index("imputed")
        station_gaps = selected_gaps.loc[selected_gaps["series"] == summary.series]
        filled_points: list[pd.Timestamp] = []
        for gap in station_gaps.itertuples(index=False):
            filled_points.extend(pd.date_range(gap.start, gap.end, freq="h"))
        imputed = after.intersection(pd.DatetimeIndex(filled_points))
        recovered = after.difference(before).difference(imputed)
        for support_type, index in (("imputed", imputed), ("recovered", recovered)):
            for timestamp in index:
                rows.append(
                    {
                        "series": summary.series,
                        "timestamp": timestamp,
                        "age_days": float(
                            (summary.test_target_start - timestamp)
                            / pd.Timedelta(days=1)
                        ),
                        "support_type": support_type,
                    }
                )

    points = pd.DataFrame(rows)
    expected_imputed = int(selected["effective_imputed_hours"].sum())
    expected_recovered = int(
        (
            selected["imputation_gain_effective_hours"]
            - selected["effective_imputed_hours"]
        ).sum()
    )
    actual = points["support_type"].value_counts() if not points.empty else pd.Series()
    if (
        int(actual.get("imputed", 0)) != expected_imputed
        or int(actual.get("recovered", 0)) != expected_recovered
    ):
        raise ValueError(
            "La ganancia por antigüedad no cuadra con el soporte efectivo "
            f"(imputadas {actual.get('imputed', 0)}/{expected_imputed}; "
            f"recuperadas {actual.get('recovered', 0)}/{expected_recovered})"
        )
    if points.empty:
        return points
    if points["age_days"].lt(0).any():
        raise ValueError("Hay horas imputadas posteriores al inicio del test")
    points["age_bin"] = pd.cut(
        points["age_days"], AGE_BINS, labels=AGE_LABELS, include_lowest=True
    )
    return points


def _save_imputation_age(
    path: Path,
    table: pd.DataFrame,
    blocks: pd.DataFrame,
    gaps: pd.DataFrame,
    pollutant: str,
) -> bool:
    points = _effective_imputation_age_points(table, blocks, gaps)
    if points.empty:
        return False
    by_age = (
        points.groupby(["age_bin", "support_type"], observed=False)
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [AGE_LABELS, AGE_SUPPORT_TYPES], names=["age_bin", "support_type"]
            ),
            fill_value=0,
        )
    )
    positions = np.arange(len(AGE_LABELS))
    figure, axis = plt.subplots(figsize=(10, 6.5), facecolor=FIGURE_FACE)
    style_axis(axis)
    axis.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.55)

    values_by_type = {
        support_type: by_age.xs(support_type, level="support_type").to_numpy(
            dtype=float
        )
        for support_type in AGE_SUPPORT_TYPES
    }
    width = 0.36
    for offset, support_type in zip(
        (-width / 2, width / 2), AGE_SUPPORT_TYPES, strict=True
    ):
        bars = axis.bar(
            positions + offset,
            values_by_type[support_type],
            width=width,
            color=AGE_SUPPORT_COLORS[support_type],
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            label=AGE_SUPPORT_LABELS[support_type],
        )
        axis.bar_label(
            bars,
            labels=[
                f"{int(value):,} h".replace(",", ".") if value else ""
                for value in values_by_type[support_type]
            ],
            padding=4,
            fontsize=8,
            color=TEXT_COLOR,
        )

    totals = values_by_type["imputed"] + values_by_type["recovered"]
    total = float(totals.sum())
    maximum = float(max(totals.max(), *(values.max() for values in values_by_type.values())))
    for position, total_value, imputed, recovered in zip(
        positions,
        totals,
        values_by_type["imputed"],
        values_by_type["recovered"],
        strict=True,
    ):
        if total_value:
            axis.text(
                position,
                max(imputed, recovered) + maximum * 0.055,
                f"Total {int(total_value):,} h\n{100 * total_value / total:.1f}%".replace(
                    ",", "."
                ),
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT_COLOR,
            )
    axis.set_xticks(positions, AGE_LABELS, rotation=25, ha="right")
    axis.set_ylabel("Horas efectivas de entrenamiento ganadas")
    axis.set_title("Ganancia de soporte por antigüedad", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)
    axis.margins(y=0.18)
    _figure_header(
        figure,
        "Cuándo aporta soporte la imputación",
        "Separa las horas rellenadas de las observaciones reales que pasan a un segmento "
        "suficientemente largo; excluye holdout y reserva de validación.",
        pollutant,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.84))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def render_plots(run_dir: Path) -> list[Path]:
    """Validate persisted tables and render all support figures."""
    series_path = run_dir / "series_summary.csv"
    blocks_path = run_dir / "blocks.csv"
    gaps_path = run_dir / "gaps.csv"
    detection_path = run_dir / "detection.csv"
    manifest_path = run_dir / "manifest.json"
    for path in (series_path, blocks_path, gaps_path, detection_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"No existe {path}")
    pollutant = str(json.loads(manifest_path.read_text(encoding="utf-8"))["pollutant"])
    series = pd.read_csv(series_path)
    if "regime" in series.columns:
        raise ValueError(
            "El reporte usa el esquema antiguo con regímenes; vuelve a ejecutar el análisis"
        )
    missing = sorted(SERIES_REQUIRED - set(series.columns))
    if missing:
        raise ValueError(f"Faltan columnas en series_summary.csv: {', '.join(missing)}")
    blocks = pd.read_csv(blocks_path)
    gaps = pd.read_csv(gaps_path)
    detection = pd.read_csv(detection_path)
    strategy_rows = series.loc[series["strategy"] == PLOT_STRATEGY]
    if strategy_rows.empty:
        raise ValueError(
            f"El reporte no contiene resultados para la estrategia {PLOT_STRATEGY}"
        )
    series = series.loc[
        (series["arm"] == "raw") | (series["strategy"] == PLOT_STRATEGY)
    ].copy()
    blocks = blocks.loc[
        (blocks["arm"] == "raw") | (blocks["strategy"] == PLOT_STRATEGY)
    ].copy()
    detection = detection.loc[detection["strategy"] == PLOT_STRATEGY].copy()
    builders = (
        (
            "retention_overview.png",
            lambda p: _save_retention_overview(p, series, pollutant),
        ),
        (
            "block_length_distribution.png",
            lambda p: _save_block_distribution(p, blocks, series, pollutant),
        ),
        (
            "retention_overview_detected.png",
            lambda p: _save_retention_overview(
                p, series, pollutant, stage="detected"
            ),
        ),
        (
            "retention_overview_imputed.png",
            lambda p: _save_retention_overview(
                p, series, pollutant, stage="imputed"
            ),
        ),
        (
            "block_length_distribution_detected.png",
            lambda p: _save_block_distribution(
                p, blocks, series, pollutant, stage="detected"
            ),
        ),
        (
            "block_length_distribution_imputed.png",
            lambda p: _save_block_distribution(
                p, blocks, series, pollutant, stage="imputed"
            ),
        ),
        (
            "retained_hours_by_series_detected.png",
            lambda p: _save_series_comparison(
                p,
                series,
                pollutant,
                stage="detected",
                value_column="retained_pct",
                title="Horas de entrenamiento retenidas tras deteccion",
                subtitle=(
                    "Porcentaje del historial observado tras retirar las anomalias de "
                    f"{PLOT_STRATEGY}, despues de reservar validacion."
                ),
                xlabel="Horas retenidas (%)",
                xlim=(0, 101),
            ),
        ),
        (
            "retained_hours_by_series_imputed.png",
            lambda p: _save_series_comparison(
                p,
                series,
                pollutant,
                stage="imputed",
                value_column="retained_pct",
                title="Horas de entrenamiento retenidas tras deteccion e imputacion",
                subtitle=(
                    f"Porcentaje del historial {PLOT_STRATEGY}+impute, despues de "
                    "reservar validacion."
                ),
                xlabel="Horas retenidas (%)",
                xlim=(0, 101),
            ),
        ),
        (
            "usable_blocks_by_series_detected.png",
            lambda p: _save_series_comparison(
                p,
                series,
                pollutant,
                stage="detected",
                value_column="used_blocks",
                title="Bloques utilizables tras deteccion",
                subtitle=(
                    f"Bloques efectivos por serie despues de aplicar {PLOT_STRATEGY}; "
                    "el prefijo anfitrion cuenta como train."
                ),
                xlabel="Numero de bloques efectivos",
            ),
        ),
        (
            "usable_blocks_by_series_imputed.png",
            lambda p: _save_series_comparison(
                p,
                series,
                pollutant,
                stage="imputed",
                value_column="used_blocks",
                title="Bloques utilizables tras deteccion e imputacion",
                subtitle=(
                    f"Bloques efectivos por serie de {PLOT_STRATEGY}+impute; "
                    "el prefijo anfitrion cuenta como train."
                ),
                xlabel="Numero de bloques efectivos",
            ),
        ),
        (
            "detected_block_length_distribution.png",
            lambda p: _save_detected_block_distribution(
                p, blocks, series, pollutant
            ),
        ),
        (
            "support_overview.png",
            lambda p: _save_support_overview(p, series, pollutant),
        ),
        (
            "support_by_series.png",
            lambda p: _save_matrix(
                p, series, value="valid_hours_pct_raw",
                title="Cambio del soporte válido respecto a raw, por serie",
                label="Porcentaje de raw", fmt=".0f", exclude_raw=True,
                subtitle="Cada celda expresa horas válidas como porcentaje de raw: 100 conserva "
                "el soporte, menos de 100 lo pierde y más de 100 lo amplía.",
                pollutant=pollutant,
            ),
        ),
        (
            "valid_blocks_by_series.png",
            lambda p: _save_matrix(
                p, series, value="valid_blocks",
                title="Bloques válidos disponibles por serie y brazo",
                label="Bloques", fmt=".0f",
                subtitle="Cantidad absoluta de bloques que alcanzan el mínimo del protocolo; "
                "el número de bloques no representa su longitud total.",
                pollutant=pollutant,
            ),
        ),
        (
            "block_length_survival.png",
            lambda p: _save_block_survival(p, blocks, series, pollutant),
        ),
        ("gap_recovery.png", lambda p: _save_gap_recovery(p, series, pollutant)),
        (
            "detection_strategy_summary.png",
            lambda p: _save_detection_summary(p, detection, pollutant),
        ),
        (
            "imputation_age.png",
            lambda p: _save_imputation_age(p, series, blocks, gaps, pollutant),
        ),
    )
    written = []
    for filename, builder in builders:
        path = run_dir / filename
        if builder(path):
            written.append(path)
    readme_path = run_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    base = existing.split("\n## Figuras\n", maxsplit=1)[0].rstrip()
    readme_path.write_text(f"{base}\n\n{PLOT_GUIDE}", encoding="utf-8")
    return written


def _latest_run() -> Path:
    root = _repo_root() / "reports" / "data_blocks"
    runs = [
        path
        for pattern in ("forecast_support_*", "forecasting_*")
        for path in root.glob(pattern)
        if (path / "series_summary.csv").is_file()
    ]
    if not runs:
        raise FileNotFoundError(f"No hay informes de soporte de forecasting bajo {root}")
    return max(runs, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Renderiza el diagnostico raw/deteccion/imputacion"
    )
    parser.add_argument("run_dir", nargs="?", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or _latest_run()
    paths = render_plots(run_dir)
    print(f"[info] {len(paths)} figuras escritas en {run_dir}")


if __name__ == "__main__":
    main()

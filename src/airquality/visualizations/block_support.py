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
PREEXISTING_COLOR = "#d99a4e"

SERIES_REQUIRED = {
    "series",
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
    "valid_hours_pct_raw",
    "imputation_gain_valid_hours",
    "imputed_valid_age_hours_median",
    "used_blocks",
    "effective_training_hours",
}

PLOT_GUIDE = """## Figuras

- `retention_overview.png`: resume el historial raw. Muestra el porcentaje y la cantidad absoluta de bloques usados y horas efectivas de entrenamiento; las horas ya descuentan la reserva de validación.
- `block_length_distribution.png`: distribución de longitudes de los bloques raw. Las líneas verticales marcan el mínimo de entrenamiento y el mínimo del bloque anfitrión que además debe alojar la validación.
- `detected_block_length_distribution.png`: compara la distribución raw con la resultante tras retirar las anomalías de cada estrategia. Un desplazamiento hacia bloques cortos indica fragmentación del historial.
- `support_overview.png`: horas observadas e imputadas dentro de bloques válidos para cada brazo. La etiqueta indica también cuántos bloques alcanzan el mínimo del protocolo.
- `support_by_series.png`: soporte válido de cada brazo respecto a raw para cada serie. Un valor de 100 conserva el mismo número de horas; menos de 100 pierde soporte y más de 100 lo amplía.
- `valid_blocks_by_series.png`: cantidad absoluta de bloques que alcanzan el mínimo del protocolo en cada serie y brazo. Más bloques no implica necesariamente más horas, porque sus longitudes difieren.
- `block_length_survival.png`: para cada longitud del eje X muestra cuántos bloques tienen al menos esa duración. Permite ver cómo detección e imputación fragmentan o conectan el historial.
- `gap_recovery.png`: descompone las horas válidas ganadas por imputación entre horas observadas desbloqueadas, anomalías imputadas y huecos previos imputados; las etiquetas muestran el total y el reparto porcentual.
- `detection_strategy_summary.png`: muestra qué porcentaje del historial pudo puntuar cada estrategia y qué porcentaje terminó retirando como anomalía. Una hora recibe score cuando hay suficientes salidas válidas de detectores para que la estrategia pueda clasificarla.
- `imputation_age.png`: distribución entre series de la antigüedad de las horas imputadas que acabaron dentro de bloques válidos, medida desde el inicio del test.
"""


def _arm_order(table: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(table["arm"].astype(str)))


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
    path: Path, table: pd.DataFrame, pollutant: str
) -> bool:
    raw = table.loc[table["arm"] == "raw"]
    if raw.empty:
        return False
    total_blocks = int(raw["total_blocks"].sum())
    used_blocks = int(raw["used_blocks"].sum())
    observed_hours = int(raw["observed_hours"].sum())
    effective_hours = int(raw["effective_training_hours"].sum())
    minimum_hours = int(raw["minimum_hours"].iloc[0])
    host_minimum_hours = int(raw["host_minimum_hours"].iloc[0])
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
        color=RECOVERED_COLOR,
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
            f"Protocolo\ntrain >= {minimum_hours} h; "
            f"anfitrión >= {host_minimum_hours} h"
        ],
    )
    axis.set_ylim(0, 105)
    axis.set_ylabel("Porcentaje del historial raw")
    axis.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.6)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    _figure_header(
        figure,
        "Retención del historial raw para entrenamiento",
        "Compara cuántos bloques se usan y cuántas horas efectivas conserva el protocolo; "
        "las horas descuentan la reserva de validación.",
        pollutant,
    )
    figure.tight_layout(rect=(0.03, 0.12, 0.98, 0.86))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_block_distribution(
    path: Path, blocks: pd.DataFrame, table: pd.DataFrame, pollutant: str
) -> bool:
    raw = blocks.loc[blocks["arm"] == "raw", "hours"]
    if raw.empty:
        return False
    lengths = raw.to_numpy(dtype=float)
    bins = np.geomspace(1, lengths.max() + 1, 45)
    raw_table = table.loc[table["arm"] == "raw"]
    minimum_hours = int(raw_table["minimum_hours"].iloc[0])
    host_minimum_hours = int(raw_table["host_minimum_hours"].iloc[0])

    figure, axis = plt.subplots(figsize=(11, 6), facecolor=FIGURE_FACE)
    style_axis(axis)
    axis.hist(lengths, bins=bins, color="#8c6d4b", edgecolor="#fffaf2", alpha=0.9)
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
    axis.set_xlabel("Longitud del bloque raw (horas, escala log)")
    axis.set_ylabel("Número de bloques (escala log)")
    axis.grid(True, which="major", color=GRID_COLOR, linestyle="--", alpha=0.55)
    axis.legend(ncols=2, loc="upper right", fontsize=8)
    _figure_header(
        figure,
        "Distribución de las longitudes de bloque del historial raw",
        "Cada hueco temporal o valor ausente rompe un bloque; las líneas marcan los mínimos "
        "de entrenamiento y de bloque anfitrión del protocolo único.",
        pollutant,
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.98, 0.86))
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
        1,
        len(strategies),
        figsize=(5.2 * len(strategies), 6),
        facecolor=FIGURE_FACE,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for axis, strategy in zip(axes[0], strategies, strict=True):
        style_axis(axis)
        lengths = detected.loc[detected["strategy"] == strategy, "hours"].to_numpy(
            dtype=float
        )
        axis.hist(
            lengths,
            bins=bins,
            color=DETECTED_COLOR,
            edgecolor="#fffaf2",
            alpha=0.8,
            label="Tras detección",
        )
        axis.hist(
            raw,
            bins=bins,
            histtype="step",
            color=RAW_COLOR,
            linewidth=1.8,
            label="Raw",
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
        axis.set_title(strategy, loc="left", fontweight="bold", color=TEXT_COLOR)
        axis.grid(True, which="major", color=GRID_COLOR, linestyle="--", alpha=0.55)
    axes[0, 0].set_ylabel("Número de bloques (escala log)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=4, fontsize=8)
    _figure_header(
        figure,
        "Distribución de bloques después de retirar anomalías",
        "Raw se muestra como contorno; más masa en longitudes cortas tras detectar indica "
        "que las observaciones retiradas fragmentaron el historial.",
        pollutant,
    )
    figure.tight_layout(rect=(0.02, 0.12, 1, 0.86))
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


def _save_gap_recovery(path: Path, table: pd.DataFrame, pollutant: str) -> bool:
    selected = table.loc[table["stage"] == "imputed"].copy()
    if selected.empty:
        return False
    selected["observed_unlocked"] = (
        selected["imputation_gain_valid_hours"] - selected["valid_imputed_hours"]
    ).clip(lower=0)
    totals = (
        selected.groupby("arm", sort=False)[
            [
                "observed_unlocked",
                "valid_imputed_anomaly_hours",
                "valid_imputed_preexisting_gap_hours",
            ]
        ]
        .sum()
    )
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
        ("observed_unlocked", "Observado desbloqueado", RECOVERED_COLOR),
        ("valid_imputed_anomaly_hours", "Anomalia imputada", IMPUTED_COLOR),
        ("valid_imputed_preexisting_gap_hours", "Gap previo imputado", PREEXISTING_COLOR),
    ):
        values = current[column].to_numpy(dtype=float)
        axis.bar(positions, values, bottom=bottom, color=color, edgecolor=EDGE_COLOR, label=label)
        bottom += values
    for position, (_, row) in zip(positions, current.iterrows(), strict=True):
        total = float(
            row["observed_unlocked"]
            + row["valid_imputed_anomaly_hours"]
            + row["valid_imputed_preexisting_gap_hours"]
        )
        if total <= 0:
            continue
        observed_pct = 100.0 * float(row["observed_unlocked"]) / total
        anomaly_pct = 100.0 * float(row["valid_imputed_anomaly_hours"]) / total
        gap_pct = 100.0 * float(row["valid_imputed_preexisting_gap_hours"]) / total
        axis.text(
            position,
            total,
            (
                f"{int(total):,} h\n"
                f"Obs. {observed_pct:.1f}%\n"
                f"Anom. {anomaly_pct:.1f}% | Gap {gap_pct:.1f}%"
            ).replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=7,
            color=TEXT_COLOR,
        )
    axis.set_xticks(positions, current.index, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel("Horas validas ganadas")
    axis.set_title("Protocolo", loc="left", fontweight="bold")
    axis.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.55)
    axis.margins(y=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=3, fontsize=8)
    _figure_header(
        figure,
        "Cómo la imputación recupera soporte de entrenamiento",
        "Separa las horas ganadas por conectar observaciones de las horas imputadas sobre "
        "anomalías detectadas y huecos que ya existían en raw.",
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
        "Cobertura y agresividad de las estrategias de detección",
        "Una hora recibe score cuando hay suficientes salidas válidas de detectores para "
        "clasificarla; gaps, bloques demasiado cortos y fallos quedan sin score.",
        pollutant,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.84))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_imputation_age(
    path: Path, table: pd.DataFrame, pollutant: str
) -> bool:
    selected = table.loc[
        (table["stage"] == "imputed")
        & table["imputed_valid_age_hours_median"].notna()
    ].copy()
    if selected.empty:
        return False
    selected["age_months"] = selected["imputed_valid_age_hours_median"] / (24.0 * 30.44)
    groups = list(selected.groupby("strategy", sort=False))
    figure, axis = plt.subplots(
        figsize=(9, 3 + 0.45 * len(groups)), facecolor=FIGURE_FACE
    )
    style_axis(axis)
    values = [group["age_months"].to_numpy(dtype=float) for _, group in groups]
    labels = [strategy for strategy, _ in groups]
    axis.boxplot(values, orientation="horizontal", tick_labels=labels, showfliers=False)
    axis.set_xlabel("Meses antes del inicio del test")
    _figure_header(
        figure,
        "Antigüedad temporal del soporte imputado",
        "Distribución entre series de los meses que separan las horas imputadas válidas "
        "del inicio del test; la línea central de cada caja es la mediana.",
        pollutant,
    )
    axis.grid(axis="x", color=GRID_COLOR, linestyle="--", alpha=0.55)
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.84))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def render_plots(run_dir: Path) -> list[Path]:
    """Validate persisted tables and render all support figures."""
    series_path = run_dir / "series_summary.csv"
    blocks_path = run_dir / "blocks.csv"
    detection_path = run_dir / "detection.csv"
    manifest_path = run_dir / "manifest.json"
    for path in (series_path, blocks_path, detection_path, manifest_path):
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
    detection = pd.read_csv(detection_path)
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
                title="Bloques válidos disponibles por serie y estrategia",
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
        ("imputation_age.png", lambda p: _save_imputation_age(p, series, pollutant)),
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

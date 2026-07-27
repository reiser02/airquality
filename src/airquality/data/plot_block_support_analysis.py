"""Render the raw/detection/imputation block-support report from persisted CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airquality.anomaly.presentation import (
    EDGE_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    TEXT_COLOR,
    style_axis,
)
from airquality.forecasting.fill import _repo_root

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
    "regime",
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

- `retention_overview.png`: resume el historial raw. Compara, para short y long, el porcentaje y la cantidad absoluta de bloques usados y horas efectivas de entrenamiento; las horas ya descuentan la reserva de validación.
- `block_length_distribution.png`: distribución de longitudes de los bloques raw. Las líneas verticales marcan el mínimo de entrenamiento y el mínimo del bloque anfitrión que además debe alojar la validación.
- `support_overview.png`: horas observadas e imputadas dentro de bloques válidos para cada brazo. La etiqueta indica también cuántos bloques alcanzan el mínimo del régimen.
- `support_by_series.png`: soporte válido de cada brazo respecto a raw para cada serie. Un valor de 100 conserva el mismo número de horas; menos de 100 pierde soporte y más de 100 lo amplía.
- `valid_blocks_by_series.png`: cantidad absoluta de bloques que alcanzan el mínimo short o long en cada serie y brazo. Más bloques no implica necesariamente más horas, porque sus longitudes difieren.
- `block_length_survival.png`: para cada longitud del eje X muestra cuántos bloques tienen al menos esa duración. Permite ver cómo detección e imputación fragmentan o conectan el historial.
- `gap_recovery.png`: descompone las horas válidas ganadas por imputación entre horas observadas desbloqueadas, anomalías imputadas y huecos previos imputados.
- `detection_strategy_summary.png`: muestra qué porcentaje del historial pudo puntuar cada estrategia y qué porcentaje terminó retirando como anomalía.
- `imputation_age.png`: distribución entre series de la antigüedad de las horas imputadas que acabaron dentro de bloques válidos, medida desde el inicio del test.
"""


def _arm_order(table: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(table["arm"].astype(str)))


def _figure_header(figure: plt.Figure, title: str, subtitle: str) -> None:
    height = float(figure.get_size_inches()[1])
    figure.suptitle(
        title,
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


def _save_retention_overview(path: Path, table: pd.DataFrame) -> bool:
    raw = table.loc[table["arm"] == "raw"]
    if raw.empty:
        return False
    totals = raw.groupby("regime", sort=False).agg(
        total_blocks=("total_blocks", "sum"),
        used_blocks=("used_blocks", "sum"),
        observed_hours=("observed_hours", "sum"),
        effective_hours=("effective_training_hours", "sum"),
        minimum_hours=("minimum_hours", "first"),
        host_minimum_hours=("host_minimum_hours", "first"),
    )
    regimes = list(totals.index)
    block_pct = 100.0 * totals["used_blocks"] / totals["total_blocks"]
    hour_pct = 100.0 * totals["effective_hours"] / totals["observed_hours"]
    positions = np.arange(len(regimes))
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
        (block_bars, block_pct, totals["used_blocks"], "bloques"),
        (hour_bars, hour_pct, totals["effective_hours"], "horas"),
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
            f"{regime.capitalize()}\ntrain >= {int(row.minimum_hours)} h; "
            f"anfitrión >= {int(row.host_minimum_hours)} h"
            for regime, row in totals.iterrows()
        ],
    )
    axis.set_ylim(0, 105)
    axis.set_ylabel("Porcentaje del historial raw")
    axis.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.6)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    _figure_header(
        figure,
        "Retención del historial raw para entrenamiento",
        "Compara cuántos bloques se usan y cuántas horas efectivas conservan short y long; "
        "las horas descuentan la reserva de validación.",
    )
    figure.tight_layout(rect=(0.03, 0.12, 0.98, 0.86))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_block_distribution(
    path: Path, blocks: pd.DataFrame, table: pd.DataFrame
) -> bool:
    raw = blocks.loc[blocks["arm"] == "raw", "hours"]
    if raw.empty:
        return False
    lengths = raw.to_numpy(dtype=float)
    bins = np.geomspace(1, lengths.max() + 1, 45)
    requirements = (
        table.loc[table["arm"] == "raw"]
        .groupby("regime", sort=False)
        .agg(
            minimum_hours=("minimum_hours", "first"),
            host_minimum_hours=("host_minimum_hours", "first"),
        )
    )

    figure, axis = plt.subplots(figsize=(11, 6), facecolor=FIGURE_FACE)
    style_axis(axis)
    axis.hist(lengths, bins=bins, color="#8c6d4b", edgecolor="#fffaf2", alpha=0.9)
    colors = {"short": DETECTED_COLOR, "long": IMPUTED_COLOR}
    for regime, row in requirements.iterrows():
        color = colors.get(regime, TEXT_COLOR)
        axis.axvline(
            row["minimum_hours"],
            color=color,
            label=f"{regime.capitalize()} train: {int(row['minimum_hours'])} h",
        )
        axis.axvline(
            row["host_minimum_hours"],
            color=color,
            linestyle="--",
            label=f"{regime.capitalize()} + validación: {int(row['host_minimum_hours'])} h",
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
        "de entrenamiento y de bloque anfitrión.",
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.98, 0.86))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_support_overview(path: Path, table: pd.DataFrame) -> bool:
    if table.empty:
        return False
    totals = (
        table.groupby(["regime", "arm"], sort=False)
        .agg(
            valid_real_hours=("valid_real_hours", "sum"),
            valid_imputed_hours=("valid_imputed_hours", "sum"),
            valid_blocks=("valid_blocks", "sum"),
        )
        .reset_index()
    )
    regimes = list(dict.fromkeys(totals["regime"]))
    arms = _arm_order(table)
    figure, axes = plt.subplots(
        1, len(regimes), figsize=(7.2 * len(regimes), 3.8 + 0.48 * len(arms)),
        facecolor=FIGURE_FACE, squeeze=False,
    )
    for axis, regime in zip(axes[0], regimes, strict=True):
        style_axis(axis)
        selected = totals.loc[totals["regime"] == regime].set_index("arm").reindex(arms)
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
        minimum = int(table.loc[table["regime"] == regime, "minimum_hours"].iloc[0])
        axis.set_yticks(positions, arms, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("Horas en bloques validos")
        axis.set_title(
            f"{regime.capitalize()} (minimo {minimum} h)",
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
    exclude_raw: bool = False,
) -> bool:
    selected = table.loc[table["arm"] != "raw"] if exclude_raw else table
    if selected.empty:
        return False
    regimes = list(dict.fromkeys(selected["regime"]))
    arms = _arm_order(selected)
    series = list(dict.fromkeys(selected["series"]))
    finite = pd.to_numeric(selected[value], errors="coerce").dropna()
    color_min = float(finite.min()) if not finite.empty else 0.0
    color_max = float(finite.max()) if not finite.empty else 1.0
    if color_min == color_max:
        color_max = color_min + 1.0
    figure, axes = plt.subplots(
        1,
        len(regimes),
        figsize=(max(16, 2.0 * len(arms) + 7), 3.4 + 0.42 * len(series)),
        facecolor=FIGURE_FACE, squeeze=False,
    )
    images = []
    for index, (axis, regime) in enumerate(zip(axes[0], regimes, strict=True)):
        values = (
            selected.loc[selected["regime"] == regime]
            .pivot(index="series", columns="arm", values=value)
            .reindex(index=series, columns=arms)
        )
        array = values.to_numpy(dtype=float)
        image = axis.imshow(
            array,
            aspect="auto",
            cmap="YlGnBu",
            vmin=color_min,
            vmax=color_max,
        )
        images.append(image)
        for row, col in np.ndindex(array.shape):
            if np.isfinite(array[row, col]):
                axis.text(col, row, format(array[row, col], fmt), ha="center", va="center", fontsize=7)
        axis.set_xticks(np.arange(len(arms)), arms, rotation=35, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(series)))
        if index == 0:
            axis.set_yticklabels(series, fontsize=8)
        else:
            axis.set_yticklabels([])
        axis.set_title(regime.capitalize(), loc="left", fontweight="bold", color=TEXT_COLOR)
        axis.set_facecolor("#fffaf2")
    color_axis = figure.add_axes((0.925, 0.23, 0.015, 0.52))
    figure.colorbar(images[-1], cax=color_axis, label=label)
    _figure_header(figure, title, subtitle)
    figure.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.84, wspace=0.08)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_block_survival(path: Path, blocks: pd.DataFrame, table: pd.DataFrame) -> bool:
    if blocks.empty:
        return False
    regimes = list(dict.fromkeys(table["regime"]))
    arms = _arm_order(table)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(arms)))
    maximum = max(int(blocks["hours"].max()), 2)
    x = np.unique(np.geomspace(1, maximum, 120).astype(int))
    figure, axes = plt.subplots(
        1, len(regimes), figsize=(7 * len(regimes), 5.5),
        facecolor=FIGURE_FACE, squeeze=False,
    )
    for axis, regime in zip(axes[0], regimes, strict=True):
        style_axis(axis)
        for arm, color in zip(arms, colors, strict=True):
            lengths = blocks.loc[blocks["arm"] == arm, "hours"].to_numpy(dtype=int)
            axis.plot(x, [(lengths >= value).sum() for value in x], label=arm, color=color)
        minimum = int(table.loc[table["regime"] == regime, "minimum_hours"].iloc[0])
        axis.axvline(
            minimum,
            color="#bd3b37",
            linestyle="--",
            label="Mínimo del régimen",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Longitud mínima del bloque (h)")
        axis.set_ylabel("Bloques con al menos esa longitud")
        axis.set_title(regime.capitalize(), loc="left", fontweight="bold")
        axis.grid(True, which="major", color=GRID_COLOR, linestyle="--", alpha=0.55)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=min(4, len(labels)), fontsize=8)
    _figure_header(
        figure,
        "Cuántos bloques sobreviven a cada longitud mínima",
        "Cada curva cuenta bloques con al menos la duración del eje X; la línea roja marca "
        "el mínimo exigido por el régimen.",
    )
    figure.tight_layout(rect=(0.02, 0.12, 1, 0.86))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_gap_recovery(path: Path, table: pd.DataFrame) -> bool:
    selected = table.loc[table["stage"] == "imputed"].copy()
    if selected.empty:
        return False
    selected["observed_unlocked"] = (
        selected["imputation_gain_valid_hours"] - selected["valid_imputed_hours"]
    ).clip(lower=0)
    totals = (
        selected.groupby(["regime", "arm"], sort=False)[
            [
                "observed_unlocked",
                "valid_imputed_anomaly_hours",
                "valid_imputed_preexisting_gap_hours",
            ]
        ]
        .sum()
        .reset_index()
    )
    regimes = list(dict.fromkeys(totals["regime"]))
    figure, axes = plt.subplots(
        1, len(regimes), figsize=(7 * len(regimes), 5.5),
        facecolor=FIGURE_FACE, squeeze=False,
    )
    for axis, regime in zip(axes[0], regimes, strict=True):
        style_axis(axis)
        current = totals.loc[totals["regime"] == regime]
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
        axis.set_xticks(positions, current["arm"], rotation=30, ha="right", fontsize=8)
        axis.set_ylabel("Horas validas ganadas")
        axis.set_title(regime.capitalize(), loc="left", fontweight="bold")
        axis.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.55)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=3, fontsize=8)
    _figure_header(
        figure,
        "Cómo la imputación recupera soporte de entrenamiento",
        "Separa las horas ganadas por conectar observaciones de las horas imputadas sobre "
        "anomalías detectadas y huecos que ya existían en raw.",
    )
    figure.tight_layout(rect=(0.02, 0.13, 1, 0.86))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_detection_summary(path: Path, detection: pd.DataFrame) -> bool:
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
        "Cobertura indica qué parte del historial recibió score; observaciones retiradas "
        "indica qué parte del historial raw se marcó como anomalía.",
    )
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.84))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_imputation_age(path: Path, table: pd.DataFrame) -> bool:
    selected = table.loc[
        (table["stage"] == "imputed")
        & table["imputed_valid_age_hours_median"].notna()
    ].copy()
    if selected.empty:
        return False
    selected["age_months"] = selected["imputed_valid_age_hours_median"] / (24.0 * 30.44)
    groups = list(selected.groupby(["regime", "strategy"], sort=False))
    figure, axis = plt.subplots(
        figsize=(9, 3 + 0.45 * len(groups)), facecolor=FIGURE_FACE
    )
    style_axis(axis)
    values = [group["age_months"].to_numpy(dtype=float) for _, group in groups]
    labels = [f"{regime} | {strategy}" for (regime, strategy), _ in groups]
    axis.boxplot(values, orientation="horizontal", tick_labels=labels, showfliers=False)
    axis.set_xlabel("Meses antes del inicio del test")
    _figure_header(
        figure,
        "Antigüedad temporal del soporte imputado",
        "Distribución entre series de los meses que separan las horas imputadas válidas "
        "del inicio del test; la línea central de cada caja es la mediana.",
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
    for path in (series_path, blocks_path, detection_path):
        if not path.is_file():
            raise FileNotFoundError(f"No existe {path}")
    series = pd.read_csv(series_path)
    missing = sorted(SERIES_REQUIRED - set(series.columns))
    if missing:
        raise ValueError(f"Faltan columnas en series_summary.csv: {', '.join(missing)}")
    blocks = pd.read_csv(blocks_path)
    detection = pd.read_csv(detection_path)
    builders = (
        ("retention_overview.png", lambda p: _save_retention_overview(p, series)),
        (
            "block_length_distribution.png",
            lambda p: _save_block_distribution(p, blocks, series),
        ),
        ("support_overview.png", lambda p: _save_support_overview(p, series)),
        (
            "support_by_series.png",
            lambda p: _save_matrix(
                p, series, value="valid_hours_pct_raw",
                title="Cambio del soporte válido respecto a raw, por serie",
                label="Porcentaje de raw", fmt=".0f", exclude_raw=True,
                subtitle="Cada celda expresa horas válidas como porcentaje de raw: 100 conserva "
                "el soporte, menos de 100 lo pierde y más de 100 lo amplía.",
            ),
        ),
        (
            "valid_blocks_by_series.png",
            lambda p: _save_matrix(
                p, series, value="valid_blocks",
                title="Bloques válidos disponibles por serie y estrategia",
                label="Bloques", fmt=".0f",
                subtitle="Cantidad absoluta de bloques que alcanzan el mínimo short o long; "
                "el número de bloques no representa su longitud total.",
            ),
        ),
        ("block_length_survival.png", lambda p: _save_block_survival(p, blocks, series)),
        ("gap_recovery.png", lambda p: _save_gap_recovery(p, series)),
        ("detection_strategy_summary.png", lambda p: _save_detection_summary(p, detection)),
        ("imputation_age.png", lambda p: _save_imputation_age(p, series)),
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
    runs = [path for path in root.glob("forecasting_*") if (path / "series_summary.csv").is_file()]
    if not runs:
        raise FileNotFoundError(f"No hay informes de forecasting bajo {root}")
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

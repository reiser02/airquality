"""Render hourly station coverage after the shared preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd

from airquality.visualizations.anomaly import (
    EDGE_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    SERIES_COLOR,
    SERIES_GHOST_COLOR,
    TEXT_COLOR,
    style_axis,
)
from airquality.config import cfg_get_str
from airquality.data.loaders import load_raw_5m
from airquality.data.preprocessing import preprocess
from airquality.data.segments import observed_blocks


DEFAULT_POLLUTANTS = ("NO2", "O3")
POLLUTANT_COLORS = {"CO": SERIES_COLOR, "NO2": "#f28c38", "O3": "#5b8c8a"}


def _coverage_summary(series: pd.Series) -> tuple[pd.DataFrame, float]:
    """Return observed blocks and the missing percentage over the station span."""
    blocks = observed_blocks(series)
    missing_pct = 100.0 * float(series.isna().mean()) if len(series) else 0.0
    return blocks, missing_pct


def save_station_coverage(
    output_path: str | Path,
    stations: list[tuple[str, pd.DataFrame]],
    *,
    pollutant: str = "NO2",
) -> Path:
    """Plot preprocessed hourly availability for every station."""
    if not stations:
        raise ValueError("No hay estaciones que representar")

    rows = []
    for station, frame in sorted(stations):
        series = frame.iloc[:, 0]
        if series.empty:
            continue
        blocks, missing_pct = _coverage_summary(series)
        rows.append((station, series, blocks, missing_pct))
    if not rows:
        raise ValueError("Las estaciones no contienen datos horarios")

    figure, axis = plt.subplots(
        figsize=(16.5, max(9.0, 2.8 + 0.46 * len(rows))),
        facecolor=FIGURE_FACE,
    )
    style_axis(axis)
    axis.grid(False, axis="y")
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.6)

    for position, (_station, series, blocks, _missing_pct) in enumerate(rows):
        start = mdates.date2num(series.index.min())
        active_hours = (series.index.max() - series.index.min()) / pd.Timedelta(hours=1) + 1
        axis.broken_barh(
            [(start, active_hours / 24.0)],
            (position - 0.34, 0.68),
            facecolors=SERIES_GHOST_COLOR,
            edgecolors="none",
        )
        axis.broken_barh(
            [
                (mdates.date2num(block.start), float(block.hours) / 24.0)
                for block in blocks.itertuples(index=False)
            ],
            (position - 0.34, 0.68),
            facecolors=SERIES_COLOR,
            edgecolors="none",
        )

    positions = range(len(rows))
    axis.set_yticks(list(positions), [row[0] for row in rows], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Fecha")
    axis.set_xlim(
        min(row[1].index.min() for row in rows),
        max(row[1].index.max() for row in rows) + pd.Timedelta(hours=1),
    )
    locator = mdates.MonthLocator(bymonth=(1, 5, 9))
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    summary_transform = axis.get_yaxis_transform()
    axis.text(
        1.015,
        1.015,
        "NaN     Bloques",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    for position, (_station, _series, blocks, missing_pct) in enumerate(rows):
        axis.text(
            1.015,
            position,
            f"{missing_pct:6.2f}%   {len(blocks):>5}",
            transform=summary_transform,
            ha="left",
            va="center",
            fontsize=8,
            color=TEXT_COLOR,
            clip_on=False,
        )

    axis.legend(
        handles=[
            Patch(facecolor=SERIES_COLOR, edgecolor=EDGE_COLOR, label="Hora observada"),
            Patch(
                facecolor=SERIES_GHOST_COLOR,
                edgecolor=EDGE_COLOR,
                label="Hora ausente dentro del rango de la estación",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncols=2,
    )
    height = float(figure.get_size_inches()[1])
    figure.text(
        0.06,
        1.0 - 0.08 / height,
        f"Disponibilidad horaria de {pollutant} por estación",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.06,
        1.0 - 0.4 / height,
        "Preprocesado: se descartan tramos congelados, se promedia por hora y las repeticiones "
        "horarias pasan a NaN.\n"
        "El porcentaje cuenta horas ausentes, no huecos; cada hueco separa bloques continuos.",
        ha="left",
        va="top",
        fontsize=9,
        color="#6d6258",
    )
    figure.tight_layout(rect=(0.02, 0.05, 0.88, 1.0 - 0.72 / height))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return path


def save_combined_station_coverage(
    output_path: str | Path,
    stations_by_pollutant: dict[str, list[tuple[str, pd.DataFrame]]],
) -> Path:
    """Plot pollutant availability as bands for each station."""
    pollutants = tuple(stations_by_pollutant)
    if not pollutants:
        raise ValueError("No hay contaminantes que representar")
    colors = {pollutant: POLLUTANT_COLORS[pollutant] for pollutant in pollutants}
    station_maps = {
        pollutant: {station: frame for station, frame in stations}
        for pollutant, stations in stations_by_pollutant.items()
    }
    station_names = sorted(
        {station for stations in stations_by_pollutant.values() for station, _ in stations}
    )
    if not station_names:
        raise ValueError("No hay estaciones que representar")

    figure, axis = plt.subplots(
        figsize=(16.5, max(9.0, 2.8 + 0.58 * len(station_names))),
        facecolor=FIGURE_FACE,
    )
    style_axis(axis)
    axis.grid(False, axis="y")
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.6)

    limits = []
    summaries: dict[tuple[str, str], float] = {}
    band_height = min(0.3, 0.7 / len(pollutants))
    offsets = {
        pollutant: (index - (len(pollutants) - 1) / 2) * 0.7 / len(pollutants)
        for index, pollutant in enumerate(pollutants)
    }
    for position, station in enumerate(station_names):
        for pollutant in pollutants:
            frame = station_maps.get(pollutant, {}).get(station)
            if frame is None or frame.empty:
                continue
            series = frame.iloc[:, 0]
            blocks, missing_pct = _coverage_summary(series)
            summaries[(station, pollutant)] = missing_pct
            limits.extend((series.index.min(), series.index.max()))
            start = mdates.date2num(series.index.min())
            active_hours = (series.index.max() - series.index.min()) / pd.Timedelta(hours=1) + 1
            band = (position + offsets[pollutant] - band_height / 2, band_height)
            axis.broken_barh(
                [(start, active_hours / 24.0)],
                band,
                facecolors=SERIES_GHOST_COLOR,
                edgecolors="none",
            )
            axis.broken_barh(
                [
                    (mdates.date2num(block.start), float(block.hours) / 24.0)
                    for block in blocks.itertuples(index=False)
                ],
                band,
                facecolors=colors[pollutant],
                edgecolors="none",
            )

    if not limits:
        plt.close(figure)
        raise ValueError("Las estaciones no contienen datos horarios")

    axis.set_yticks(list(range(len(station_names))), station_names, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Fecha")
    axis.set_xlim(min(limits), max(limits) + pd.Timedelta(hours=1))
    axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axis.text(
        1.015,
        1.015,
        "    ".join(f"{pollutant} NaN" for pollutant in pollutants),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    summary_transform = axis.get_yaxis_transform()
    for position, station in enumerate(station_names):
        values = [
            f"{summaries[(station, pollutant)]:6.2f}%"
            if (station, pollutant) in summaries
            else "    n/d"
            for pollutant in pollutants
        ]
        axis.text(
            1.015,
            position,
            "    ".join(values),
            transform=summary_transform,
            ha="left",
            va="center",
            fontsize=8,
            color=TEXT_COLOR,
            clip_on=False,
        )

    axis.legend(
        handles=[
            *[
                Patch(
                    facecolor=colors[pollutant],
                    edgecolor=EDGE_COLOR,
                    label=f"{pollutant} observado",
                )
                for pollutant in pollutants
            ],
            Patch(facecolor=SERIES_GHOST_COLOR, edgecolor=EDGE_COLOR, label="Hora ausente"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncols=min(3, len(pollutants) + 1),
    )
    height = float(figure.get_size_inches()[1])
    figure.text(
        0.06,
        1.0 - 0.08 / height,
        f"Disponibilidad horaria de {', '.join(pollutants)} por estación",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.06,
        1.0 - 0.4 / height,
        "Preprocesado: se descartan tramos congelados, se promedia por hora y las repeticiones "
        "horarias pasan a NaN.\n"
        "El porcentaje cuenta horas ausentes, no huecos; cada contaminante ocupa una banda.",
        ha="left",
        va="top",
        fontsize=9,
        color="#6d6258",
    )
    figure.tight_layout(rect=(0.02, 0.05, 0.86, 1.0 - 0.72 / height))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return path


def save_comparison_station_coverage(
    output_path: str | Path,
    suppressed_by_pollutant: dict[str, list[tuple[str, pd.DataFrame]]],
    unsuppressed_by_pollutant: dict[str, list[tuple[str, pd.DataFrame]]],
) -> Path:
    """Compare 5-minute coverage with and without frozen-value suppression.

    Each station gets two narrow bands per pollutant: one after the normal
    preprocessing suppression and one with both frozen-value filters disabled.
    Filled bars show observed hours; the gray backing shows the span covered by
    each variant, including its internal gaps.
    """
    pollutants = tuple(
        dict.fromkeys((*suppressed_by_pollutant, *unsuppressed_by_pollutant))
    )
    if not pollutants:
        raise ValueError("No hay contaminantes que comparar")
    colors = {pollutant: POLLUTANT_COLORS[pollutant] for pollutant in pollutants}
    sources = {
        "suppressed": suppressed_by_pollutant,
        "unsuppressed": unsuppressed_by_pollutant,
    }
    station_maps = {
        source: {
            pollutant: {station: frame for station, frame in stations}
            for pollutant, stations in source_data.items()
        }
        for source, source_data in sources.items()
    }
    station_names = sorted(
        {
            station
            for source_map in station_maps.values()
            for pollutant in pollutants
            for station in source_map.get(pollutant, {})
        }
    )
    if not station_names:
        raise ValueError("No hay estaciones que comparar")

    figure, axis = plt.subplots(
        figsize=(17.5, max(9.0, 2.8 + 0.72 * len(station_names))),
        facecolor=FIGURE_FACE,
    )
    style_axis(axis)
    axis.grid(False, axis="y")
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.6)

    band_height = min(0.16, 0.72 / (2 * len(pollutants)))
    offset_step = 0.8 / (2 * len(pollutants))
    band_specs = [
        (
            pollutant,
            source,
            (2 * index + source_index - (2 * len(pollutants) - 1) / 2)
            * offset_step,
            "//" if source == "unsuppressed" else None,
        )
        for index, pollutant in enumerate(pollutants)
        for source_index, source in enumerate(("suppressed", "unsuppressed"))
    ]
    limits: list[pd.Timestamp] = []
    summaries: dict[tuple[str, str, str], float] = {}

    for position, station in enumerate(station_names):
        for pollutant, source, offset, hatch in band_specs:
            frame = station_maps[source].get(pollutant, {}).get(station)
            if frame is None or frame.empty:
                continue
            series = frame.iloc[:, 0]
            if series.empty:
                continue
            blocks, missing_pct = _coverage_summary(series)
            summaries[(station, pollutant, source)] = missing_pct
            start = series.index.min()
            end = series.index.max()
            if pd.isna(start) or pd.isna(end):
                continue
            limits.extend((start, end))
            y = position + offset - band_height / 2
            active_hours = (end - start) / pd.Timedelta(hours=1) + 1
            axis.broken_barh(
                [(mdates.date2num(start), active_hours / 24.0)],
                (y, band_height),
                facecolors=SERIES_GHOST_COLOR,
                edgecolors="none",
            )
            axis.broken_barh(
                [
                    (mdates.date2num(block.start), float(block.hours) / 24.0)
                    for block in blocks.itertuples(index=False)
                ],
                (y, band_height),
                facecolors=colors[pollutant],
                edgecolors=EDGE_COLOR,
                linewidth=0.15,
                hatch=hatch,
            )

    if not limits:
        plt.close(figure)
        raise ValueError("Las estaciones no contienen datos horarios comparables")

    axis.set_yticks(list(range(len(station_names))), station_names, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Fecha")
    axis.set_xlim(min(limits), max(limits) + pd.Timedelta(hours=1))
    axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axis.text(
        1.015,
        1.015,
        "    ".join(f"{pollutant} sup./sin" for pollutant in pollutants),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    summary_transform = axis.get_yaxis_transform()
    for position, station in enumerate(station_names):
        values = []
        for pollutant in pollutants:
            for source in ("suppressed", "unsuppressed"):
                summary = summaries.get((station, pollutant, source))
                values.append(f"{summary:5.1f}%" if summary is not None else "  n/d")
        axis.text(
            1.015,
            position,
            "    ".join(
                f"{values[index]} / {values[index + 1]}"
                for index in range(0, len(values), 2)
            ),
            transform=summary_transform,
            ha="left",
            va="center",
            fontsize=7.5,
            color=TEXT_COLOR,
            clip_on=False,
        )

    axis.legend(
        handles=[
            *[
                Patch(
                    facecolor=colors[pollutant],
                    edgecolor=EDGE_COLOR,
                    label=f"{pollutant}: con supresión",
                )
                for pollutant in pollutants
            ],
            *[
                Patch(
                    facecolor=colors[pollutant],
                    edgecolor=EDGE_COLOR,
                    hatch="//",
                    label=f"{pollutant}: sin supresión",
                )
                for pollutant in pollutants
            ],
            Patch(
                facecolor=SERIES_GHOST_COLOR,
                edgecolor=EDGE_COLOR,
                label="Hora ausente dentro del rango de la fuente",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncols=min(3, 2 * len(pollutants) + 1),
    )
    height = float(figure.get_size_inches()[1])
    figure.text(
        0.06,
        1.0 - 0.08 / height,
        "Comparación de disponibilidad horaria en datos de 5 minutos",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.06,
        1.0 - 0.4 / height,
        "Con supresión: se excluyen tramos congelados de 5 minutos y repeticiones horarias. "
        "Sin supresión: se conservan ambos tipos de repetición; las bandas grises conservan "
        "el rango y sus huecos.",
        ha="left",
        va="top",
        fontsize=9,
        color="#6d6258",
    )
    figure.tight_layout(rect=(0.02, 0.05, 0.84, 1.0 - 0.72 / height))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return path


def main() -> None:
    """Preprocess raw pollutant stations and write their coverage figures."""
    root = Path(__file__).resolve().parents[3]
    raw_base_dir = Path(
        cfg_get_str("data", "raw_base_dir", "data/raw/datos_estaciones_5m")
    )
    if not raw_base_dir.is_absolute():
        raw_base_dir = root / raw_base_dir
    stations_by_pollutant = {}
    stations_without_freeze_filters = {}
    for pollutant in DEFAULT_POLLUTANTS:
        raw_stations = [
            item
            for item in load_raw_5m(pollutant, str(raw_base_dir))
            if "antiguo" not in item[0].casefold()
        ]
        if not raw_stations:
            raise FileNotFoundError(
                f"No se encontraron datos raw de {pollutant} en {raw_base_dir}"
            )

        stations = []
        stations_without_filters = []
        for station, raw in raw_stations:
            (hourly,), _ = preprocess([raw], pollutant)
            (hourly_without_filters,), _ = preprocess(
                [raw],
                pollutant,
                exclude_frozen=False,
                remove_repeated=False,
            )
            stations.append((station, hourly))
            stations_without_filters.append((station, hourly_without_filters))
        stations_by_pollutant[pollutant] = stations
        stations_without_freeze_filters[pollutant] = stations_without_filters

        path = save_station_coverage(
            root / "reports" / "figures" / f"Gantt-{pollutant}.png",
            stations,
            pollutant=pollutant,
        )
        print(f"Figura guardada en {path}")

    path = save_combined_station_coverage(
        root / "reports" / "figures" / "Gantt-combo.png",
        stations_by_pollutant,
    )
    print(f"Figura guardada en {path}")

    path = save_comparison_station_coverage(
        root / "reports" / "figures" / "Gantt-comparison.png",
        stations_by_pollutant,
        stations_without_freeze_filters,
    )
    print(f"Figura guardada en {path}")


if __name__ == "__main__":
    main()

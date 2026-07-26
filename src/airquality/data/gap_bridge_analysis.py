"""Audit the support gained by bridging short gaps in preprocessed series.

Run with::

    uv run python -m airquality.data.gap_bridge_analysis
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airquality.data.loaders import load_raw_5m
from airquality.data.preprocessing import MIN_RUN, MIN_USEFUL, preprocess
from airquality.data.segments import contiguous_observed_segments
from airquality.data.series import ensure_datetime_series
from airquality.paths import create_run_dir

AGE_BINS = (-1, 30, 90, 180, 365, 730, np.inf)
AGE_LABELS = ("0-30 d", "31-90 d", "91-180 d", "181-365 d", "1-2 anos", ">2 anos")
RECOVERY_ORDER = ("new", "extension")
RECOVERY_LABELS = {"new": "Bloque nuevo", "extension": "Extension"}
FIGURE_FACE = "#f6f1e8"
TEXT_COLOR = "#27313a"
EDGE_COLOR = "#3c4650"
GRID_COLOR = "#d8cabb"


def bridge_components(
    series: pd.Series,
    station: str,
    *,
    minimum: int = 80,
    max_gap: int = 5,
) -> pd.DataFrame:
    """Describe components formed by joining observed runs across short gaps."""
    segments = contiguous_observed_segments(series)
    if not segments:
        return pd.DataFrame()

    blocks = pd.DataFrame(
        {
            "start": [segment.index[0] for segment in segments],
            "end": [segment.index[-1] for segment in segments],
            "points": [len(segment) for segment in segments],
        }
    )
    blocks["gap_before"] = (
        (blocks["start"] - blocks["end"].shift()) / pd.Timedelta(hours=1) - 1
    ).astype("Int64")
    blocks["component_id"] = (
        blocks["gap_before"].isna() | blocks["gap_before"].gt(max_gap)
    ).cumsum()
    blocks["short_points"] = blocks["points"].where(blocks["points"].lt(minimum), 0)
    blocks["eligible_source_points"] = blocks["points"].where(
        blocks["points"].ge(minimum), 0
    )
    blocks["short_block"] = blocks["points"].lt(minimum).astype(int)
    blocks["eligible_source_block"] = blocks["points"].ge(minimum).astype(int)

    components = (
        blocks.groupby("component_id", sort=True)
        .agg(
            start=("start", "first"),
            end=("end", "last"),
            observed_points=("points", "sum"),
            source_blocks=("points", "size"),
            short_blocks=("short_block", "sum"),
            short_source_points=("short_points", "sum"),
            eligible_source_blocks=("eligible_source_block", "sum"),
            eligible_source_points=("eligible_source_points", "sum"),
        )
        .reset_index()
    )
    components["span_points"] = (
        (components["end"] - components["start"]) / pd.Timedelta(hours=1) + 1
    ).astype(int)
    components["imputed_points"] = (
        components["span_points"] - components["observed_points"]
    )
    components["eligible_after"] = components["span_points"].ge(minimum)
    components["recovery_type"] = np.select(
        [
            components["eligible_after"]
            & components["short_blocks"].gt(0)
            & components["eligible_source_blocks"].eq(0),
            components["eligible_after"]
            & components["short_blocks"].gt(0)
            & components["eligible_source_blocks"].gt(0),
        ],
        RECOVERY_ORDER,
        default="none",
    )
    components["recovered_real_points"] = components["short_source_points"].where(
        components["recovery_type"].ne("none"), 0
    )
    components["station"] = station
    components["age_days"] = (
        (series.index.max() - components["end"]) / pd.Timedelta(days=1)
    ).clip(lower=0)
    components["minimum"] = minimum
    components["max_gap"] = max_gap

    assert components["span_points"].eq(
        components["observed_points"] + components["imputed_points"]
    ).all()
    return components[
        [
            "station",
            "component_id",
            "start",
            "end",
            "age_days",
            "span_points",
            "observed_points",
            "imputed_points",
            "source_blocks",
            "short_blocks",
            "recovered_real_points",
            "eligible_source_blocks",
            "eligible_source_points",
            "eligible_after",
            "recovery_type",
            "minimum",
            "max_gap",
        ]
    ]


def summarize_components(components: pd.DataFrame) -> pd.DataFrame:
    """Return the global before/after support accounting."""
    eligible = components.loc[components["eligible_after"]]
    recovered = eligible.loc[eligible["recovery_type"].ne("none")]
    baseline_real = int(components["eligible_source_points"].sum())
    real_after = int(eligible["observed_points"].sum())
    imputed = int(eligible["imputed_points"].sum())
    usable_after = real_after + imputed
    return pd.DataFrame(
        [
            {
                "series": int(components["station"].nunique()),
                "source_blocks": int(components["source_blocks"].sum()),
                "source_eligible_blocks": int(
                    components["eligible_source_blocks"].sum()
                ),
                "eligible_components_after": len(eligible),
                "recovered_components": len(recovered),
                "new_eligible_components": int(recovered["recovery_type"].eq("new").sum()),
                "extended_eligible_components": int(
                    recovered["recovery_type"].eq("extension").sum()
                ),
                "short_blocks_recovered": int(recovered["short_blocks"].sum()),
                "baseline_real_points": baseline_real,
                "recovered_real_points": real_after - baseline_real,
                "real_points_after": real_after,
                "imputed_points": imputed,
                "usable_points_after": usable_after,
                "usable_gain_pct": (
                    100.0 * (usable_after - baseline_real) / baseline_real
                    if baseline_real
                    else float("nan")
                ),
                "real_share_after_pct": (
                    100.0 * real_after / usable_after if usable_after else 0.0
                ),
            }
        ]
    )


def _format_count(value: float | int) -> str:
    return f"{int(value):,}".replace(",", ".")


def save_overview(
    path: Path,
    components: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    pollutant: str,
    minimum: int,
    max_gap: int,
) -> None:
    """Save a four-panel overview of gain, cost, recovery type, and recency."""
    row = summary.iloc[0]
    recovered = components.loc[components["recovery_type"].ne("none")].copy()
    recovered["age_bin"] = pd.cut(
        recovered["age_days"], AGE_BINS, labels=AGE_LABELS, include_lowest=True
    )
    eligible = components.loc[components["eligible_after"]].copy()
    eligible["age_bin"] = pd.cut(
        eligible["age_days"], AGE_BINS, labels=AGE_LABELS, include_lowest=True
    )
    by_type = (
        recovered.groupby("recovery_type", observed=True)
        .agg(
            components=("station", "size"),
            short_blocks=("short_blocks", "sum"),
            real_points=("recovered_real_points", "sum"),
            imputed_points=("imputed_points", "sum"),
        )
        .reindex(RECOVERY_ORDER, fill_value=0)
    )
    by_age = (
        recovered.groupby(["age_bin", "recovery_type"], observed=False)
        .agg(
            short_blocks=("short_blocks", "sum"),
            real_points=("recovered_real_points", "sum"),
            imputed_points=("imputed_points", "sum"),
        )
        .reindex(
            pd.MultiIndex.from_product(
                [AGE_LABELS, RECOVERY_ORDER], names=["age_bin", "recovery_type"]
            ),
            fill_value=0,
        )
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=FIGURE_FACE)
    base_color, recovered_color, imputed_color = "#4778a8", "#4f9a70", "#e58b3a"

    axis = axes[0, 0]
    support = np.array(
        [row["baseline_real_points"], row["recovered_real_points"], row["imputed_points"]],
        dtype=float,
    )
    support_total = float(support.sum())
    left = 0.0
    for value, color, label in zip(
        support,
        (base_color, recovered_color, imputed_color),
        ("Real ya util", "Real recuperado", "Imputado"),
        strict=True,
    ):
        axis.barh([0], [value], left=left, color=color, edgecolor=EDGE_COLOR, label=label)
        axis.text(
            left + value / 2,
            0,
            f"{_format_count(value)}\n{100 * value / support_total if support_total else 0.0:.1f}%",
            ha="center",
            va="center",
            color="white" if value > support_total * 0.08 else TEXT_COLOR,
            fontsize=9,
            fontweight="bold",
        )
        left += value
    axis.set_yticks([])
    axis.set_xlabel(f"Puntos horarios en componentes >={minimum}")
    gain = (
        f"+{row['usable_gain_pct']:.1f}% sobre el inicial"
        if np.isfinite(row["usable_gain_pct"])
        else "sin soporte inicial comparable"
    )
    axis.set_title(
        f"A. Soporte utilizable: {gain}",
        loc="left",
        fontweight="bold",
    )
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=3, frameon=False)

    axis = axes[0, 1]
    x = np.arange(len(RECOVERY_ORDER))
    real_bars = axis.bar(
        x, by_type["real_points"], color=recovered_color, edgecolor=EDGE_COLOR, label="Real recuperado"
    )
    axis.bar(
        x,
        by_type["imputed_points"],
        bottom=by_type["real_points"],
        color=imputed_color,
        edgecolor=EDGE_COLOR,
        label="Imputado",
    )
    axis.bar_label(
        real_bars,
        labels=[
            f"{_format_count(by_type.loc[k, 'components'])} componentes\n"
            f"{_format_count(by_type.loc[k, 'short_blocks'])} bloques cortos"
            for k in RECOVERY_ORDER
        ],
        padding=12,
        color=TEXT_COLOR,
        fontsize=9,
    )
    axis.set_xticks(x, [RECOVERY_LABELS[k] for k in RECOVERY_ORDER])
    axis.set_ylabel("Puntos horarios")
    axis.set_title("B. Recuperacion nueva frente a extension", loc="left", fontweight="bold")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    x = np.arange(len(AGE_LABELS))
    bottom = np.zeros(len(AGE_LABELS))
    for recovery_type, color in zip(RECOVERY_ORDER, ("#71b08b", "#346b8c"), strict=True):
        values = by_age["short_blocks"].xs(recovery_type, level="recovery_type").to_numpy()
        axis.bar(
            x,
            values,
            bottom=bottom,
            color=color,
            edgecolor=EDGE_COLOR,
            label=RECOVERY_LABELS[recovery_type],
        )
        bottom += values
    axis.set_xticks(x, AGE_LABELS, rotation=25, ha="right")
    axis.set_ylabel("Bloques cortos recuperados")
    axis.set_title("C. Donde estan los bloques recuperados", loc="left", fontweight="bold")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    real_age = (
        by_age["real_points"]
        .groupby(level="age_bin", observed=False)
        .sum()
        .reindex(AGE_LABELS, fill_value=0)
    )
    imputed_age = (
        eligible.groupby("age_bin", observed=False)["imputed_points"]
        .sum()
        .reindex(AGE_LABELS, fill_value=0)
    )
    axis.bar(x, real_age, color=recovered_color, edgecolor=EDGE_COLOR, label="Real recuperado")
    axis.bar(
        x,
        imputed_age,
        bottom=real_age,
        color=imputed_color,
        edgecolor=EDGE_COLOR,
        label="Imputado",
    )
    axis.set_xticks(x, AGE_LABELS, rotation=25, ha="right")
    axis.set_ylabel("Puntos horarios")
    axis.set_title("D. Ganancia por antiguedad", loc="left", fontweight="bold")
    axis.legend(frameon=False)

    for axis in axes.flat:
        axis.set_facecolor("#fffaf2")
        axis.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        f"{pollutant}: valor de unir bloques a traves de gaps de hasta {max_gap} horas",
        x=0.04,
        y=0.99,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.04,
        0.95,
        f"Minimo final: {minimum} puntos. Antiguedad respecto al ultimo timestamp de cada estacion. "
        f"'Bloque nuevo' no contenia antes ninguna racha >={minimum}.",
        ha="left",
        fontsize=10,
        color="#6d6258",
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.99, 0.92), h_pad=3.2, w_pad=2.5)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)


def run_analysis(
    *,
    base_dir: str | Path,
    output_dir: str | Path,
    pollutant: str = "NO2",
    minimum: int = 80,
    max_gap: int = 5,
    min_run: int = MIN_RUN,
    min_useful: int = MIN_USEFUL,
) -> dict[str, Path]:
    """Preprocess raw series and persist component accounting plus its overview."""
    stations = load_raw_5m(pollutant, str(base_dir))
    if not stations:
        raise FileNotFoundError(f"No raw files found for {pollutant} under {base_dir}")

    frames = []
    for station, raw in stations:
        (hourly,), _ = preprocess(
            [raw], pollutant, min_run=min_run, min_useful=min_useful
        )
        series = ensure_datetime_series(hourly.iloc[:, 0], freq="h", name=station)
        frame = bridge_components(
            series, station, minimum=minimum, max_gap=max_gap
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No hay componentes observados despues del preprocesado")
    components = pd.concat(frames, ignore_index=True)
    components.insert(0, "pollutant", pollutant)
    summary = summarize_components(components)
    summary.insert(0, "pollutant", pollutant)
    summary.insert(1, "minimum", minimum)
    summary.insert(2, "max_gap", max_gap)
    assert components.isna().sum().sum() == 0

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "components": output / "components.csv",
        "summary": output / "summary.csv",
        "overview": output / "gap_bridge_overview.png",
    }
    components.to_csv(paths["components"], index=False)
    summary.to_csv(paths["summary"], index=False)
    save_overview(
        paths["overview"],
        components,
        summary,
        pollutant=pollutant,
        minimum=minimum,
        max_gap=max_gap,
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure support recovered by bridging short gaps in preprocessed data"
    )
    parser.add_argument("--base-dir", default="data/raw/datos_estaciones_5m")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pollutant", default="NO2")
    parser.add_argument("--minimum", type=int, default=80)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--min-run", type=int, default=MIN_RUN)
    parser.add_argument("--min-useful", type=int, default=MIN_USEFUL)
    args = parser.parse_args()
    if min(args.minimum, args.max_gap, args.min_run, args.min_useful) <= 0:
        parser.error("minimum, max-gap, min-run and min-useful must be positive")

    output = (
        Path(args.output_dir)
        if args.output_dir
        else create_run_dir(
            Path("reports/data_blocks"),
            f"gap_bridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    )
    paths = run_analysis(
        base_dir=args.base_dir,
        output_dir=output,
        pollutant=args.pollutant,
        minimum=args.minimum,
        max_gap=args.max_gap,
        min_run=args.min_run,
        min_useful=args.min_useful,
    )
    print(f"Informe guardado en: {output}")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

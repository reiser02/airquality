"""Persist and render figures for a completed Monte Carlo imputation run.

Run again without executing any model::

    uv run python -m airquality.imputation.plot_montecarlo_results \
        reports/benchmark/montecarlo_<stamp>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Base palette mirrored from `airquality.anomaly.presentation` (same "base color"
# the anomaly benchmark plots use). Copied verbatim on purpose: importing that
# module would drag in the heavy STL/anomaly stack just for a handful of colours.
FIGURE_FACE = "#f6f1e8"
AXIS_FACE = "#fffaf2"
TEXT_COLOR = "#27313a"
GRID_COLOR = "#d8cabb"
SPINE_COLOR = "#c2b3a3"
BAND_ALPHA = 0.15
PLOT_STORE_COLUMNS = (
    "gap_size",
    "series_name",
    "kind",
    "model_name",
    "timestamp",
    "value",
)

# Distinct line colours drawn from the same presentation.py hex family.
_LINE_PALETTE = (
    "#4b79a8",  # blue (SERIES_COLOR)
    "#f28c38",  # orange
    "#5b8c5a",  # green
    "#9b59b6",  # purple
    "#d6453c",  # red
    "#8c6d4b",  # brown
    "#cf6ba9",  # pink
    "#6d6258",  # taupe
    "#27313a",  # dark slate (TEXT_COLOR)
    "#2a9d8f",  # teal
    "#b59a00",  # mustard
    "#7b6ca8",  # muted violet
)
# Curated colours so key models stay visually stable across runs; the linear
# baseline gets the standout red so its gap=1 behaviour is easy to spot.
_MODEL_LINE_COLORS = {
    "DLinear": "#8c6d4b",
    "LinearInterp": "#d6453c",
    "LinearRegression": "#cf6ba9",
    "NLinear": "#b59a00",
    "TSPulse": "#9b59b6",
    "TSPulse_FineTuned": "#7b6ca8",
    "Prophet": "#5b8c5a",
    "TiDE": "#4b79a8",
    "NHiTS": "#f28c38",
    "TSMixer": "#6d6258",
    "RNN": "#27313a",
    "TCN": "#7b6ca8",
    "interp": "#2a9d8f",
}


def _model_color_map(models: list[str]) -> dict[str, str]:
    """Assign a stable, distinct colour to each model from the base palette."""
    color_map: dict[str, str] = {}
    used = {color for name, color in _MODEL_LINE_COLORS.items() if name in models}
    cycle = [color for color in _LINE_PALETTE if color not in used]
    cycle_pos = 0
    for model in sorted(models):
        if model in _MODEL_LINE_COLORS:
            color_map[model] = _MODEL_LINE_COLORS[model]
        else:
            color_map[model] = cycle[cycle_pos % len(cycle)] if cycle else "#4b79a8"
            cycle_pos += 1
    return color_map


def _style_metric_axis(ax: "plt.Axes") -> None:
    """Apply the shared cream/base styling to one metric subplot."""
    ax.set_facecolor(AXIS_FACE)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, linestyle="--", alpha=0.65)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)


def _aggregate_metrics_by_gap(
    results_mc_df: pd.DataFrame, metrics: list[str]
) -> pd.DataFrame:
    """Summarize metrics per (model, gap size) with mean and P05/P95 over runs."""
    rows: list[dict[str, Any]] = []
    for (model, gap), group in results_mc_df.groupby(["Modelo", "Gap_Size"], sort=True):
        row: dict[str, Any] = {"Modelo": str(model), "Gap_Size": int(gap)}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_Mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_P05"] = float(values.quantile(0.05)) if len(values) else float("nan")
            row[f"{metric}_P95"] = float(values.quantile(0.95)) if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _save_metric_gap_plot(
    results_mc_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> Path | None:
    """Plot each metric as one line per model across gap sizes (P05-P95 band).

    Persists both the aggregated table (`metrics_by_gap.csv`) and the figure
    (`metrics_by_gap.png`). Returns the image path, or None if there is nothing
    to plot.
    """
    if results_mc_df.empty:
        return None
    if not {"Modelo", "Gap_Size"}.issubset(results_mc_df.columns):
        return None

    metrics = [m for m in ("MAE", "RMSE", "MASE") if m in results_mc_df.columns]
    if not metrics:
        return None

    agg_df = _aggregate_metrics_by_gap(results_mc_df, metrics)
    if agg_df.empty:
        return None
    agg_df = agg_df.sort_values(["Modelo", "Gap_Size"]).reset_index(drop=True)
    agg_df.to_csv(output_dir / "metrics_by_gap.csv", index=False)

    gap_sizes = sorted(int(g) for g in agg_df["Gap_Size"].unique())
    models = sorted(str(m) for m in agg_df["Modelo"].unique())
    color_map = _model_color_map(models)
    x_positions = list(range(len(gap_sizes)))
    gap_to_x = {gap: pos for pos, gap in enumerate(gap_sizes)}

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(6.0 * len(metrics), 4.8),
        facecolor=FIGURE_FACE,
        squeeze=False,
    )
    axes_row = axes[0]

    for ax, metric in zip(axes_row, metrics):
        for model in models:
            model_df = agg_df[agg_df["Modelo"] == model].set_index("Gap_Size")
            xs, means, lows, highs = [], [], [], []
            for gap in gap_sizes:
                if gap not in model_df.index:
                    continue
                xs.append(gap_to_x[gap])
                means.append(model_df.loc[gap, f"{metric}_Mean"])
                lows.append(model_df.loc[gap, f"{metric}_P05"])
                highs.append(model_df.loc[gap, f"{metric}_P95"])
            if not xs:
                continue
            color = color_map[model]
            ax.plot(xs, means, marker="o", markersize=4.0, lw=1.6, color=color, label=model)
            ax.fill_between(xs, lows, highs, color=color, alpha=BAND_ALPHA, linewidth=0)

        _style_metric_axis(ax)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("Tamano de hueco")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(g) for g in gap_sizes])

    axes_row[0].set_ylabel("Valor de la metrica (media, banda P05-P95)")

    handles, labels = axes_row[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 6),
            frameon=True,
            framealpha=0.95,
            edgecolor="#d0d0d0",
            fontsize=9,
        )

    fig.suptitle(
        "Metricas de imputacion por tamano de hueco",
        x=0.5,
        y=0.99,
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    image_path = output_dir / "metrics_by_gap.png"
    fig.savefig(image_path, dpi=150, facecolor=FIGURE_FACE)
    plt.close(fig)
    return image_path


def _sanitize_filename(text: str) -> str:
    """Reduce free-form series names to a safe filesystem identifier."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return sanitized.strip("._") or "series"


def _to_pd_series(obj: Any) -> pd.Series:
    """Coerce a Series/TimeSeries-like plot payload into a sorted pandas Series."""
    if isinstance(obj, pd.Series):
        return obj.sort_index()
    if hasattr(obj, "to_series"):
        try:
            series = obj.to_series()
        except Exception:
            return pd.Series(dtype=float)
        if isinstance(series, pd.Series):
            return series.sort_index()
    return pd.Series(dtype=float)


def save_plot_store(
    plot_store: dict[int, dict[str, Any]], output_path: Path
) -> Path:
    """Persist actual values and predictions as a compressed long-form CSV."""
    frames: list[pd.DataFrame] = []
    for gap_size, gap_payload in sorted(plot_store.items()):
        by_series = gap_payload.get("series", {}) if isinstance(gap_payload, dict) else {}
        if not isinstance(by_series, dict):
            continue
        for series_name, payload in by_series.items():
            if not isinstance(payload, dict):
                continue
            series_items = [
                ("actual", "", payload.get("actual")),
                ("naive_mase", "", payload.get("naive_mase")),
            ]
            preds = payload.get("preds", {})
            if isinstance(preds, dict):
                series_items.extend(
                    ("prediction", str(model_name), prediction)
                    for model_name, prediction in preds.items()
                )
            for kind, model_name, value in series_items:
                series = _to_pd_series(value)
                if series.empty:
                    continue
                frames.append(
                    pd.DataFrame(
                        {
                            "gap_size": int(gap_size),
                            "series_name": str(series_name),
                            "kind": kind,
                            "model_name": model_name,
                            "timestamp": series.index,
                            "value": series.to_numpy(),
                        }
                    )
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=PLOT_STORE_COLUMNS)
    )
    frame.to_csv(output_path, index=False, compression="gzip")
    return output_path


def load_plot_store(path: Path) -> dict[int, dict[str, Any]]:
    """Load a plot store previously written by :func:`save_plot_store`."""
    frame = pd.read_csv(path)
    missing = set(PLOT_STORE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"CSV de plots invalido; faltan columnas: {sorted(missing)}")
    if frame.empty:
        return {}

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["gap_size"] = pd.to_numeric(frame["gap_size"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if frame[["timestamp", "gap_size"]].isna().any().any():
        raise ValueError("CSV de plots invalido; contiene timestamps o gaps no validos")

    def _restore(rows: pd.DataFrame) -> pd.Series:
        return pd.Series(
            rows["value"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(rows["timestamp"]),
            dtype=float,
        ).sort_index()

    plot_store: dict[int, dict[str, Any]] = {}
    for (gap_size, series_name), rows in frame.groupby(
        ["gap_size", "series_name"], sort=False
    ):
        payload: dict[str, Any] = {
            "actual": _restore(rows[rows["kind"] == "actual"]),
            "naive_mase": _restore(rows[rows["kind"] == "naive_mase"]),
            "preds": {},
        }
        prediction_rows = rows[rows["kind"] == "prediction"]
        for model_name, model_rows in prediction_rows.groupby("model_name", sort=False):
            payload["preds"][str(model_name)] = _restore(model_rows)
        plot_store.setdefault(int(gap_size), {"series": {}})["series"][str(series_name)] = payload
    return plot_store


def _save_plot_images(
    plot_store: dict[int, dict[str, Any]],
    *,
    output_dir: Path,
) -> pd.DataFrame:
    """Render one PNG per (gap size, series) with real values vs. model predictions.

    Writes the images under ``<output_dir>/plots/gap_<n>/`` plus a
    ``plot_images.csv`` manifest, which is also returned as a dataframe.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for gap_size, gap_payload in sorted(plot_store.items()):
        by_series = gap_payload.get("series", {}) if isinstance(gap_payload, dict) else {}
        if not isinstance(by_series, dict):
            continue

        gap_dir = plots_dir / f"gap_{int(gap_size)}"
        gap_dir.mkdir(parents=True, exist_ok=True)

        for series_name, series_payload in by_series.items():
            if not isinstance(series_payload, dict):
                continue

            actual = _to_pd_series(series_payload.get("actual", pd.Series(dtype=float)))
            preds_payload = series_payload.get("preds", {})
            preds_by_model = preds_payload if isinstance(preds_payload, dict) else {}

            mask_index = pd.DatetimeIndex([])
            naive_mase = _to_pd_series(series_payload.get("naive_mase", pd.Series(dtype=float)))
            if len(naive_mase) > 0:
                mask_index = pd.DatetimeIndex(naive_mase.index)

            clean_preds: dict[str, pd.Series] = {}
            for model_name, pred in preds_by_model.items():
                pred_series = _to_pd_series(pred).dropna()
                if len(pred_series) > 0:
                    clean_preds[str(model_name)] = pred_series
                    mask_index = mask_index.union(pd.DatetimeIndex(pred_series.index))

            gap_real = actual.reindex(mask_index).dropna().sort_index()
            model_scores: dict[str, float] = {}
            for model_name, pred_series in clean_preds.items():
                common = gap_real.index.intersection(pred_series.index)
                if len(common) > 0:
                    error = np.abs(
                        gap_real.reindex(common).to_numpy(dtype=float)
                        - pred_series.reindex(common).to_numpy(dtype=float)
                    )
                    finite = error[np.isfinite(error)]
                    if len(finite) > 0:
                        model_scores[model_name] = float(finite.mean())

            ranked_models = sorted(model_scores, key=lambda name: (model_scores[name], name))
            colors = _model_color_map(list(clean_preds))
            top_models = ranked_models[:5]

            fig = plt.figure(figsize=(13.2, 8.4), facecolor=FIGURE_FACE)
            grid = fig.add_gridspec(
                2, 2, height_ratios=(0.9, 1.35), width_ratios=(2.15, 1.0),
                hspace=0.42, wspace=0.28,
            )
            context_ax = fig.add_subplot(grid[0, :])
            comparison_ax = fig.add_subplot(grid[1, 0])
            ranking_ax = fig.add_subplot(grid[1, 1])

            for axis in (context_ax, comparison_ax, ranking_ax):
                _style_metric_axis(axis)

            if len(actual) > 0:
                context_ax.plot(
                    actual.index, actual.values, color="#8f8b84", alpha=0.72,
                    lw=0.9, label="Serie observada",
                )
            if len(gap_real) > 0:
                context_ax.scatter(
                    gap_real.index, gap_real.values, color="#d6453c", s=27,
                    edgecolor=AXIS_FACE, linewidth=0.7, zorder=3,
                    label="Valores ocultados",
                )
            context_ax.set_title("Contexto temporal", loc="left", fontsize=11, fontweight="bold")
            context_ax.set_ylabel("NO2")
            context_ax.legend(loc="upper left", ncol=2, fontsize=8)
            context_ax.grid(True, axis="both", color=GRID_COLOR, linestyle="--", alpha=0.45)
            context_ax.tick_params(axis="x", labelrotation=0)

            gap_index = pd.DatetimeIndex(gap_real.index)
            x = np.arange(len(gap_real))
            boundaries = [0]
            if len(gap_index) > 1:
                actual_steps = pd.DatetimeIndex(actual.index).to_series().diff().dropna()
                cadence = actual_steps.median() if len(actual_steps) else None
                if cadence is not None and cadence > pd.Timedelta(0):
                    boundaries.extend(
                        index for index, delta in enumerate(gap_index[1:] - gap_index[:-1], start=1)
                        if delta > cadence * 1.5
                    )
            boundaries.append(len(gap_real))

            for boundary in boundaries[1:-1]:
                comparison_ax.axvline(boundary - 0.5, color=GRID_COLOR, lw=0.9, linestyle=":")
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=False):
                comparison_ax.plot(
                    x[start:end], gap_real.iloc[start:end], color=TEXT_COLOR,
                    marker="o", markersize=4.5, lw=2.1,
                    label="Valor real" if start == 0 else None, zorder=4,
                )
                for model_name in top_models:
                    values = clean_preds[model_name].reindex(gap_index).to_numpy(dtype=float)
                    comparison_ax.plot(
                        x[start:end], values[start:end], color=colors[model_name],
                        marker="o", markersize=3.6, lw=1.25, alpha=0.9,
                        label=(
                            f"{model_name} | MAE {model_scores[model_name]:.2f}"
                            if start == 0 else None
                        ),
                    )

            if len(gap_real) > 0:
                tick_count = min(len(gap_real), 8)
                tick_positions = np.unique(np.linspace(0, len(gap_real) - 1, tick_count, dtype=int))
                comparison_ax.set_xticks(tick_positions)
                comparison_ax.set_xticklabels(
                    [gap_index[position].strftime("%d %b\n%H:%M") for position in tick_positions]
                )
            comparison_ax.set_title(
                f"Detalle de los {len(top_models)} mejores modelos",
                loc="left", fontsize=11, fontweight="bold",
            )
            comparison_ax.set_xlabel("Valores ocultados en orden temporal")
            comparison_ax.set_ylabel("NO2 imputado")
            comparison_ax.grid(True, axis="both", color=GRID_COLOR, linestyle="--", alpha=0.45)
            if top_models:
                comparison_ax.legend(loc="best", fontsize=8, ncol=2)

            if ranked_models:
                scores = np.asarray([model_scores[name] for name in ranked_models], dtype=float)
                y = np.arange(len(ranked_models))
                ranking_colors = [colors[name] for name in ranked_models]
                positive = scores[scores > 0]
                use_log = len(positive) > 0 and float(scores.max() / positive.min()) > 25.0
                if use_log:
                    floor = float(positive.min()) * 0.75
                    ranking_ax.hlines(y, floor, scores, color=ranking_colors, lw=2.0, alpha=0.65)
                    ranking_ax.scatter(scores, y, color=ranking_colors, s=34, zorder=3)
                    ranking_ax.set_xscale("log")
                else:
                    ranking_ax.barh(y, scores, color=ranking_colors, alpha=0.88, height=0.68)
                ranking_ax.set_yticks(y)
                ranking_ax.set_yticklabels(ranked_models, fontsize=8)
                ranking_ax.invert_yaxis()
                ranking_ax.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.55)
                ranking_ax.grid(False, axis="y")
                for position, score in zip(y, scores, strict=False):
                    ranking_ax.annotate(
                        f"{score:.2f}", (score, position), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7.5,
                        color=TEXT_COLOR,
                    )
                ranking_ax.margins(x=0.14)
            ranking_ax.set_title("Ranking de todos los modelos", loc="left", fontsize=11, fontweight="bold")
            ranking_ax.set_xlabel("MAE en estos valores" + (" (escala log)" if ranked_models and use_log else ""))

            fig.suptitle(
                f"Imputacion de {series_name} | Huecos de {int(gap_size)} h",
                x=0.07, y=0.975, ha="left", fontsize=15, fontweight="bold", color=TEXT_COLOR,
            )
            fig.text(
                0.07, 0.94,
                "Arriba: contexto de los huecos. Abajo: mejores predicciones y error de todos los modelos.",
                ha="left", fontsize=9, color="#6d6258",
            )
            fig.subplots_adjust(left=0.07, right=0.96, bottom=0.09, top=0.89)

            image_path = gap_dir / f"{_sanitize_filename(str(series_name))}.png"
            fig.savefig(image_path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
            plt.close(fig)

            rows.append(
                {
                    "gap_size": int(gap_size),
                    "series_name": str(series_name),
                    "model_count": len(preds_by_model),
                    "image_path": str(image_path.relative_to(output_dir)),
                }
            )

    manifest_df = pd.DataFrame(
        rows,
        columns=["gap_size", "series_name", "model_count", "image_path"],
    )
    manifest_df.to_csv(output_dir / "plot_images.csv", index=False)
    return manifest_df


def render_run_figures(
    output_dir: Path,
    *,
    results_mc_df: pd.DataFrame | None = None,
    plot_store: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render every figure from in-memory data or persisted run artifacts."""
    output_dir = Path(output_dir)
    if results_mc_df is None:
        results_mc_df = pd.read_csv(output_dir / "results_mc.csv")
    if plot_store is None:
        plot_store = load_plot_store(output_dir / "plot_store.csv.gz")
    return {
        "metric_gap_plot_path": _save_metric_gap_plot(
            results_mc_df, output_dir=output_dir
        ),
        "plot_manifest_df": _save_plot_images(plot_store, output_dir=output_dir),
    }


def main(argv: list[str] | None = None) -> None:
    """Regenerate figures from the CSV artifacts in an existing run directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Directorio reports/benchmark/montecarlo_*")
    args = parser.parse_args(argv)
    if not args.run_dir.is_dir():
        raise SystemExit(f"No existe el directorio: {args.run_dir}")

    artifacts = render_run_figures(args.run_dir)
    metric_path = artifacts["metric_gap_plot_path"]
    if metric_path is not None:
        print(f"[info] Figura guardada en {metric_path}")
    manifest = artifacts["plot_manifest_df"]
    print(f"[info] Regeneradas {len(manifest)} figuras en {args.run_dir / 'plots'}")


if __name__ == "__main__":
    main()

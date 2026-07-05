"""Plotting helpers for benchmark predictions, gaps, and model comparisons."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from airquality.data.io import to_pd_series


def _draw_series(
    ax: Any,
    x: pd.Index,
    y: Any,
    *,
    label: str,
    render_style: str,
    color: str | None = None,
    alpha: float = 1.0,
    lw: float = 1.5,
    linestyle: str = "-",
) -> str | None:
    """Draw one line or point series and return the rendered color."""
    if render_style == "points":
        line = ax.plot(
            x,
            y,
            label=label,
            color=color,
            alpha=alpha,
            linestyle="None",
            marker="o",
            markersize=3.0,
        )[0]
    else:
        line = ax.plot(
            x,
            y,
            label=label,
            color=color,
            alpha=alpha,
            lw=lw,
            linestyle=linestyle,
        )[0]
    return str(line.get_color()) if hasattr(line, "get_color") else color


def _plot_vertical_error_connectors(
    ax: Any,
    *,
    gap_real: pd.Series,
    pred: pd.Series,
    color: str | None,
) -> None:
    """Draw vertical connectors between real and predicted gap values."""
    common_idx = gap_real.index.intersection(pred.index)
    for ts in common_idx:
        y_true = float(gap_real.loc[ts])
        y_pred = float(pred.loc[ts])
        if np.isfinite(y_true) and np.isfinite(y_pred):
            ax.plot(
                [ts, ts],
                [y_true, y_pred],
                linestyle="--",
                lw=0.9,
                color=color or "gray",
                alpha=0.45,
            )


def get_prediction_time_window(
    preds_by_model: dict[str, pd.Series],
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Obtiene la ventana temporal que cubre todas las predicciones validas."""
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for pred_s in preds_by_model.values():
        ps = pred_s.dropna()
        if len(ps) == 0:
            continue
        starts.append(pd.Timestamp(ps.index.min()))
        ends.append(pd.Timestamp(ps.index.max()))

    if not starts:
        return None
    return min(starts), max(ends)


def plot_predictions_by_gap(
    plot_store: dict[int, dict[str, Any]],
    series_name: str = "all",
    show_naive_mase_plot: bool = False,
    grid_by_gap: bool = False,
    grid_n_cols: int = 2,
    grid_cell_width: float = 7.5,
    grid_cell_height: float = 4.8,
    results_df: pd.DataFrame | None = None,
    error_display: str = "both",
    legend_sort: str = "mase_asc",
    legend_outside_bottom: bool = False,
    legend_n_cols: int = 2,
    render_style: str = "line",
) -> None:
    """Grafica predicciones por gap con soporte de orden/overlay por MASE."""
    if grid_n_cols < 1:
        raise ValueError("grid_n_cols debe ser >= 1")
    if grid_cell_width <= 0 or grid_cell_height <= 0:
        raise ValueError("grid_cell_width y grid_cell_height deben ser > 0")
    if error_display not in {"none", "legend", "textbox", "both"}:
        raise ValueError("error_display debe ser one of: none, legend, textbox, both")
    if legend_sort not in {"none", "mase_asc", "mase_desc"}:
        raise ValueError("legend_sort debe ser one of: none, mase_asc, mase_desc")
    if legend_n_cols < 1:
        raise ValueError("legend_n_cols debe ser >= 1")
    if render_style not in {"line", "points", "mixed"}:
        raise ValueError("render_style debe ser one of: line, points, mixed")

    def _get_mase_map(serie: str, gap: int) -> dict[str, float]:
        """Look up per-model MASE values for one (series, gap size) pair."""
        if results_df is None or results_df.empty:
            return {}

        required_cols = {"Serie", "Gap_Size", "Modelo", "MASE"}
        if not required_cols.issubset(results_df.columns):
            return {}

        mask = (results_df["Serie"] == serie) & (results_df["Gap_Size"] == int(gap))
        subset = results_df.loc[mask, ["Modelo", "MASE"]]
        return {
            str(row["Modelo"]): float(row["MASE"])
            for _, row in subset.iterrows()
            if pd.notna(row["MASE"])
        }

    def _plot_on_axis(
        ax: Any, series_payload: dict[str, Any], gap: int, serie: str
    ) -> None:
        """Draw one series' actuals, per-model predictions, and naive reference."""
        mase_map = _get_mase_map(serie, gap)
        preds_payload = series_payload.get("preds", {})
        actual = series_payload.get("actual", pd.Series(dtype=float))
        window = get_prediction_time_window(preds_payload)
        if window is not None:
            start, end = window
            actual = actual.loc[start:end]

        if len(actual) > 0:
            _draw_series(
                ax,
                actual.index,
                actual.values,
                label="Real",
                render_style="line" if render_style == "mixed" else render_style,
                color="black",
                alpha=0.45,
                lw=1.3,
            )

        gap_real = pd.Series(dtype=float)
        if render_style == "mixed":
            mask_index = pd.DatetimeIndex(
                series_payload.get("naive_mase", pd.Series(dtype=float)).index
            )
            if len(mask_index) == 0:
                for pred_s in preds_payload.values():
                    if isinstance(pred_s, pd.Series):
                        mask_index = mask_index.union(pd.DatetimeIndex(pred_s.index))

            gap_real = actual.reindex(mask_index).dropna()
            if len(gap_real) > 0:
                ax.plot(
                    gap_real.index,
                    gap_real.values,
                    label="Hueco real",
                    color="red",
                    linestyle="None",
                    marker="o",
                    markersize=4.0,
                )

        model_items = list(preds_payload.items())
        if legend_sort != "none" and mase_map:
            reverse = legend_sort == "mase_desc"

            def _rank(item: tuple[str, pd.Series]) -> float:
                """Legend sort key: the model's MASE (missing values sort last)."""
                model_name = item[0]
                if model_name not in mase_map or not np.isfinite(mase_map[model_name]):
                    return np.inf if not reverse else -np.inf
                return mase_map[model_name]

            model_items = sorted(model_items, key=_rank, reverse=reverse)

        for model_name, pred_s in model_items:
            ps = pred_s.dropna()
            if len(ps) > 0:
                label = model_name
                if error_display in {"legend", "both"} and model_name in mase_map:
                    label = f"{model_name} (MASE={mase_map[model_name]:.3f})"
                pred_color = _draw_series(
                    ax,
                    ps.index,
                    ps.values,
                    lw=1.2,
                    label=label,
                    render_style="points",
                )
                if render_style == "mixed" and len(gap_real) > 0:
                    _plot_vertical_error_connectors(
                        ax,
                        gap_real=gap_real,
                        pred=ps,
                        color=pred_color,
                    )

        naive_mase = series_payload.get("naive_mase", pd.Series(dtype=float)).dropna()
        if window is not None:
            start, end = window
            naive_mase = naive_mase.loc[start:end]

        if show_naive_mase_plot and len(naive_mase) > 0:
            _draw_series(
                ax,
                naive_mase.index,
                naive_mase.values,
                label="Naive MASE y(t-m)",
                render_style="line" if render_style == "mixed" else render_style,
                lw=1.2,
                linestyle="--",
                color="orange",
            )

        if error_display in {"textbox", "both"} and mase_map:
            sorted_items = sorted(
                mase_map.items(),
                key=lambda item: item[1] if np.isfinite(item[1]) else np.inf,
            )
            lines = ["MASE por modelo"]
            lines.extend([f"{name}: {value:.3f}" for name, value in sorted_items])
            ax.text(
                0.01,
                0.99,
                "\n".join(lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

        ax.set_title(f"Serie: {serie} | Gap={gap}")
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("NO2")
        if legend_outside_bottom:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=legend_n_cols,
                frameon=True,
            )
        else:
            ax.legend(loc="upper right")

    gaps = sorted(plot_store.keys())
    if not gaps:
        return

    first_payload = plot_store[gaps[0]]
    if "series" not in first_payload or not isinstance(first_payload["series"], dict):
        raise ValueError("plot_store invalido: se esperaba la clave 'series' por gap.")

    all_series = list(first_payload["series"].keys())
    if not all_series:
        return
    target_series = all_series if series_name == "all" else [series_name]

    if grid_by_gap:
        n_rows = math.ceil(len(gaps) / grid_n_cols)
        for current_series in target_series:
            fig, axes = plt.subplots(
                n_rows,
                grid_n_cols,
                figsize=(grid_cell_width * grid_n_cols, grid_cell_height * n_rows),
                squeeze=False,
            )

            for i, gap in enumerate(gaps):
                ax = axes[i // grid_n_cols][i % grid_n_cols]
                payload = plot_store[gap]

                series_payload = payload["series"].get(current_series)
                if series_payload is None:
                    ax.axis("off")
                    continue

                _plot_on_axis(ax, series_payload, gap, current_series)

            for j in range(len(gaps), n_rows * grid_n_cols):
                axes[j // grid_n_cols][j % grid_n_cols].axis("off")

            if legend_outside_bottom:
                plt.tight_layout(rect=(0, 0.08, 1, 1))
            else:
                plt.tight_layout()
            plt.show()


def plot_predictions_by_method_grid(
    plot_store: dict[Any, Any],
    series_name: str = "all",
    methods: list[str] | tuple[str, ...] | None = None,
    gaps: list[int] | tuple[int, ...] | None = None,
    grid_n_cols: int = 2,
    grid_cell_width: float = 7.5,
    grid_cell_height: float = 4.8,
    render_style: str = "line",
) -> None:
    """
    Grafica un grid por metodo con una serie de test por subplot.

    Soporta dos formatos de `plot_store`:
    1) Formato por gap (benchmark de imputacion):
       {gap: {"series": {serie: {"actual":..., "preds": {metodo: pred}}}}}
    2) Formato por metodo (execute_complete_pipeline actual):
       {metodo: {serie: {"actual":..., "prediction":...}}}
    """
    if grid_n_cols < 1:
        raise ValueError("grid_n_cols debe ser >= 1")
    if grid_cell_width <= 0 or grid_cell_height <= 0:
        raise ValueError("grid_cell_width y grid_cell_height deben ser > 0")
    if render_style not in {"line", "points", "mixed"}:
        raise ValueError("render_style debe ser one of: line, points, mixed")
    if not plot_store:
        return

    def _to_pd_series(obj: Any) -> pd.Series:
        """Coerce a Series/TimeSeries payload to a sorted Series (empty otherwise)."""
        if isinstance(obj, pd.Series):
            return obj.sort_index()
        if isinstance(obj, TimeSeries):
            return to_pd_series(obj).sort_index()
        return pd.Series(dtype=float)

    def _plot_single_axis(
        ax: Any,
        actual: pd.Series,
        pred: pd.Series,
        serie: str,
        method: str,
        gap: int | None,
    ) -> None:
        """Draw one method's prediction vs. the actual series and title with MAE."""
        actual_clean = actual.dropna()
        pred_clean = pred.dropna()
        mae_value = float("nan")

        aligned_idx = actual_clean.index.intersection(pred_clean.index)
        if len(aligned_idx) > 0:
            y_true = actual_clean.reindex(aligned_idx).to_numpy(dtype=float)
            y_pred = pred_clean.reindex(aligned_idx).to_numpy(dtype=float)
            valid = np.isfinite(y_true) & np.isfinite(y_pred)
            if np.any(valid):
                mae_value = float(np.mean(np.abs(y_true[valid] - y_pred[valid])))

        if len(pred_clean) > 0:
            pred_start = pd.Timestamp(pred_clean.index.min())
            pred_end = pd.Timestamp(pred_clean.index.max())
            actual_plot = actual_clean.loc[pred_start:pred_end]
        else:
            actual_plot = actual_clean

        if len(actual_plot) > 0:
            _draw_series(
                ax,
                actual_plot.index,
                actual_plot.values,
                render_style="line" if render_style == "mixed" else render_style,
                color="black",
                alpha=0.5,
                lw=2.0,
                label="Serie original",
            )

        if render_style == "mixed" and len(pred_clean) > 0:
            gap_real = actual_clean.reindex(pred_clean.index).dropna()
            if len(gap_real) > 0:
                ax.plot(
                    gap_real.index,
                    gap_real.values,
                    color="red",
                    linestyle="None",
                    marker="o",
                    markersize=4.0,
                    label="Hueco real",
                )

        pred_color = "#0B3D91"
        if len(pred_clean) > 0:
            pred_color = _draw_series(
                ax,
                pred_clean.index,
                pred_clean.values,
                render_style="points" if render_style == "mixed" else render_style,
                color="#0B3D91",
                alpha=1.0,
                lw=2.4,
                label="Prediccion",
            )
            if render_style == "mixed":
                gap_real = actual_clean.reindex(pred_clean.index).dropna()
                _plot_vertical_error_connectors(
                    ax,
                    gap_real=gap_real,
                    pred=pred_clean,
                    color=pred_color,
                )
        elif len(actual_plot) == 0:
            ax.text(
                0.5,
                0.5,
                "Sin datos",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )

        if gap is None:
            title = f"Serie: {serie}"
        else:
            title = f"Serie: {serie} | Gap={gap}"

        if np.isfinite(mae_value):
            title = f"{title} | MAE={mae_value:.3f}"

        ax.set_title(title)
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("NO2")
        ax.legend(loc="best")

    def _plot_method_grid(
        method: str,
        by_series: dict[str, dict[str, Any]],
        gap: int | None,
    ) -> None:
        """Render one figure with a grid of per-series subplots for one method."""
        if series_name == "all":
            target_series = list(by_series.keys())
        else:
            target_series = [series_name] if series_name in by_series else []

        if not target_series:
            return

        n_rows = math.ceil(len(target_series) / grid_n_cols)
        fig, axes = plt.subplots(
            n_rows,
            grid_n_cols,
            figsize=(grid_cell_width * grid_n_cols, grid_cell_height * n_rows),
            squeeze=False,
        )

        for i, serie in enumerate(target_series):
            ax = axes[i // grid_n_cols][i % grid_n_cols]
            payload = by_series[serie]
            actual = _to_pd_series(payload.get("actual", pd.Series(dtype=float)))
            pred = _to_pd_series(payload.get("prediction", pd.Series(dtype=float)))
            _plot_single_axis(ax, actual, pred, serie, method, gap)

        for j in range(len(target_series), n_rows * grid_n_cols):
            axes[j // grid_n_cols][j % grid_n_cols].axis("off")

        title = f"Metodo: {method}" if gap is None else f"Metodo: {method} | Gap={gap}"
        fig.suptitle(title)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plt.show()

    first_key = next(iter(plot_store))
    first_payload = plot_store[first_key]
    is_gap_format = (
        isinstance(first_payload, dict)
        and "series" in first_payload
        and isinstance(first_payload["series"], dict)
    )

    if is_gap_format:
        available_gaps = sorted(plot_store.keys())
        target_gaps = (
            available_gaps if gaps is None else [g for g in gaps if g in plot_store]
        )

        for gap in target_gaps:
            payload_gap = plot_store[gap]
            by_series_gap = payload_gap.get("series", {})
            if not by_series_gap:
                continue

            method_set: set[str] = set()
            for series_payload in by_series_gap.values():
                preds_payload = series_payload.get("preds", {})
                if isinstance(preds_payload, dict):
                    method_set.update([str(m) for m in preds_payload.keys()])

            if not method_set:
                continue

            target_methods = (
                list(method_set)
                if methods is None
                else [m for m in methods if str(m) in method_set]
            )

            for method in target_methods:
                by_series_method: dict[str, dict[str, Any]] = {}
                for serie, series_payload in by_series_gap.items():
                    preds_payload = series_payload.get("preds", {})
                    pred = (
                        preds_payload.get(method, pd.Series(dtype=float))
                        if isinstance(preds_payload, dict)
                        else pd.Series(dtype=float)
                    )
                    by_series_method[str(serie)] = {
                        "actual": series_payload.get("actual", pd.Series(dtype=float)),
                        "prediction": pred,
                    }

                _plot_method_grid(str(method), by_series_method, int(gap))

        return

    available_methods = [str(m) for m in plot_store.keys()]
    target_methods = (
        available_methods
        if methods is None
        else [str(m) for m in methods if str(m) in plot_store]
    )

    for method in target_methods:
        by_series = plot_store[str(method)]
        if not isinstance(by_series, dict):
            continue
        _plot_method_grid(str(method), by_series, None)

    return

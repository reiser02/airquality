"""Render publication figures and LaTeX tables for imputation statistics.

Run with::

    uv run python -m airquality.visualizations.imputation_statistics RUN_DIR
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airquality.visualizations.style import (
    AXIS_FACE,
    EDGE_COLOR,
    FIGURE_DPI,
    FIGURE_FACE,
    GRID_COLOR,
    MUTED_TEXT,
    TEXT_COLOR,
    add_plot_header,
    style_axis,
)


STATISTICAL_FILES = {
    "scores": "statistical_station_scores.csv",
    "omnibus": "statistical_omnibus.csv",
    "pairwise": "statistical_pairwise.csv",
}
SCORE_REQUIRED = ("metric", "gap_size", "series", "model", "score", "rank")
OMNIBUS_REQUIRED = (
    "metric",
    "gap_size",
    "test",
    "statistic",
    "p_value",
    "alpha",
    "significant",
    "confirmatory",
    "n_stations",
    "n_models",
)
PAIRWISE_REQUIRED = (
    "metric",
    "gap_size",
    "left",
    "right",
    "left_mean",
    "right_mean",
    "mean_difference",
    "improvement_pct",
    "rank_biserial",
    "wins",
    "ties",
    "losses",
    "p_holm",
    "alpha",
    "statistically_significant",
)
MODEL_COLORS = (
    "#3d7ab5",
    "#cf6f1e",
    "#5b8c5a",
    "#9b59b6",
    "#2a7f76",
    "#c65353",
    "#8c6d4b",
    "#6677a8",
    "#b07d32",
    "#4c8b89",
    "#ad5c7d",
    "#708090",
    "#b28b4e",
)


def _read_csv(run_dir: Path, filename: str, required: tuple[str, ...]) -> pd.DataFrame:
    path = run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Falta {path}. Ejecuta primero airquality.imputation.statistics."
        )
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} no contiene las columnas requeridas: {missing}")
    return frame


def _normalize_boolean_column(frame: pd.DataFrame, column: str) -> None:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    normalized = frame[column].map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else mapping.get(str(value).strip().lower())
    )
    if normalized.isna().any():
        raise ValueError(f"{column} contiene valores booleanos desconocidos")
    frame[column] = normalized.astype(bool)


def load_statistical_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the three statistical tables required by this renderer."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio del run: {run_dir}")
    requirements = {
        "scores": SCORE_REQUIRED,
        "omnibus": OMNIBUS_REQUIRED,
        "pairwise": PAIRWISE_REQUIRED,
    }
    tables = {
        key: _read_csv(run_dir, filename, requirements[key])
        for key, filename in STATISTICAL_FILES.items()
    }
    for column in ("significant", "confirmatory"):
        _normalize_boolean_column(tables["omnibus"], column)
    _normalize_boolean_column(tables["pairwise"], "statistically_significant")
    return tables


def build_model_summary(scores: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    """Summarize station scores and ranks for every model and gap."""
    missing = sorted(set(SCORE_REQUIRED) - set(scores.columns))
    if missing:
        raise ValueError(f"Faltan columnas en los scores estadisticos: {missing}")
    selected = scores.loc[scores["metric"].astype(str) == metric].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "gap_size",
                "model",
                "mean_score",
                "median_score",
                "station_sd",
                "mean_rank",
                "n_stations",
            ]
        )
    selected["score"] = pd.to_numeric(selected["score"], errors="coerce")
    selected["rank"] = pd.to_numeric(selected["rank"], errors="coerce")
    selected["gap_size"] = pd.to_numeric(selected["gap_size"], errors="coerce")
    selected = selected.loc[
        np.isfinite(selected["score"])
        & np.isfinite(selected["rank"])
        & np.isfinite(selected["gap_size"])
    ]
    summary = (
        selected.groupby(["metric", "gap_size", "model"], sort=False)
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            station_sd=("score", "std"),
            mean_rank=("rank", "mean"),
            n_stations=("series", "nunique"),
        )
        .reset_index()
    )
    overall_rank = summary.groupby("model")["mean_rank"].mean().rename("overall_rank")
    return (
        summary.join(overall_rank, on="model")
        .sort_values(["overall_rank", "gap_size", "mean_rank", "model"], kind="stable")
        .drop(columns="overall_rank")
        .reset_index(drop=True)
    )


def _find_pair(pairwise: pd.DataFrame, left: str, right: str) -> pd.Series | None:
    rows = pairwise.loc[
        pairwise.apply(
            lambda row: {str(row["left"]), str(row["right"])} == {left, right},
            axis=1,
        )
    ]
    return rows.iloc[0] if len(rows) == 1 else None


def build_top_comparisons(
    scores: pd.DataFrame,
    pairwise: pd.DataFrame,
    *,
    metric: str = "MASE",
    top_k: int = 3,
) -> pd.DataFrame:
    """Compare the descriptive rank winner with the other top-k models per gap."""
    if top_k < 2:
        raise ValueError("top_k debe ser al menos 2")
    summary = build_model_summary(scores, metric=metric)
    selected_pairs = pairwise.loc[pairwise["metric"].astype(str) == metric].copy()
    rows: list[dict[str, object]] = []
    for gap_size, gap_summary in summary.groupby("gap_size", sort=True):
        ranked = gap_summary.sort_values(
            ["mean_rank", "mean_score", "model"], kind="stable"
        ).head(top_k)
        if len(ranked) < 2:
            continue
        reference = str(ranked.iloc[0]["model"])
        reference_mean = float(ranked.iloc[0]["mean_score"])
        gap_pairs = selected_pairs.loc[
            pd.to_numeric(selected_pairs["gap_size"], errors="coerce")
            == float(gap_size)
        ]
        for comparison_row in ranked.iloc[1:].itertuples(index=False):
            comparison = str(comparison_row.model)
            pair = _find_pair(gap_pairs, reference, comparison)
            if pair is None:
                continue
            reference_is_left = str(pair["left"]) == reference
            if reference_is_left:
                difference = float(pair["mean_difference"])
                effect = float(pair["rank_biserial"])
                wins = int(pair["wins"])
                losses = int(pair["losses"])
            else:
                difference = -float(pair["mean_difference"])
                effect = -float(pair["rank_biserial"])
                wins = int(pair["losses"])
                losses = int(pair["wins"])
            comparison_mean = float(comparison_row.mean_score)
            improvement = (
                100.0 * (1.0 - reference_mean / comparison_mean)
                if comparison_mean > 0.0
                else float("nan")
            )
            rows.append(
                {
                    "metric": metric,
                    "gap_size": int(gap_size),
                    "reference": reference,
                    "comparison": comparison,
                    "reference_mean": reference_mean,
                    "comparison_mean": comparison_mean,
                    "mean_difference": difference,
                    "improvement_pct": improvement,
                    "rank_biserial": effect,
                    "wins": wins,
                    "ties": int(pair["ties"]),
                    "losses": losses,
                    "p_holm": float(pair["p_holm"]),
                    "alpha": float(pair["alpha"]),
                    "confidence_level_pct": 100.0 * (1.0 - float(pair["alpha"])),
                    "statistically_significant": bool(
                        pair["statistically_significant"]
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "gap_size",
            "reference",
            "comparison",
            "reference_mean",
            "comparison_mean",
            "mean_difference",
            "improvement_pct",
            "rank_biserial",
            "wins",
            "ties",
            "losses",
            "p_holm",
            "alpha",
            "confidence_level_pct",
            "statistically_significant",
        ],
    )


def _model_order(scores: pd.DataFrame, metric: str) -> list[str]:
    summary = build_model_summary(scores, metric=metric)
    if summary.empty:
        return []
    return (
        summary.groupby("model", sort=False)["mean_rank"]
        .mean()
        .sort_values(kind="stable")
        .index.astype(str)
        .tolist()
    )


def _save_formats(figure: plt.Figure, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(
            path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            facecolor=FIGURE_FACE,
            edgecolor="none",
        )
    plt.close(figure)
    return paths


def save_station_score_plot(
    scores: pd.DataFrame,
    *,
    metric: str,
    output_stem: Path,
) -> list[Path]:
    """Plot station-score distributions in one panel per gap size."""
    selected = scores.loc[scores["metric"].astype(str) == metric].copy()
    models = _model_order(scores, metric)
    gaps = sorted(pd.to_numeric(selected["gap_size"], errors="coerce").dropna().unique())
    if selected.empty or not models or not gaps:
        return []
    ncols = 2
    nrows = math.ceil(len(gaps) / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, max(7.5, 6.2 * nrows)),
        squeeze=False,
    )
    colors = {model: MODEL_COLORS[index % len(MODEL_COLORS)] for index, model in enumerate(models)}
    for axis, gap_size in zip(axes.flat, gaps):
        gap = selected.loc[
            pd.to_numeric(selected["gap_size"], errors="coerce") == gap_size
        ]
        arrays = [
            pd.to_numeric(gap.loc[gap["model"].astype(str) == model, "score"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
            for model in models
        ]
        boxes = axis.boxplot(
            arrays,
            orientation="horizontal",
            tick_labels=models,
            patch_artist=True,
            widths=0.56,
            showfliers=False,
            medianprops={"color": EDGE_COLOR, "linewidth": 1.4},
            whiskerprops={"color": EDGE_COLOR},
            capprops={"color": EDGE_COLOR},
        )
        for patch, model in zip(boxes["boxes"], models):
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.34)
            patch.set_edgecolor(colors[model])
        for position, (model, values) in enumerate(zip(models, arrays), start=1):
            if not len(values):
                continue
            jitter = np.linspace(-0.16, 0.16, len(values))
            axis.scatter(
                values,
                position + jitter,
                s=15,
                color=colors[model],
                alpha=0.68,
                linewidths=0,
                zorder=3,
            )
            axis.scatter(
                [np.mean(values)],
                [position],
                marker="D",
                s=40,
                color=colors[model],
                edgecolor=FIGURE_FACE,
                linewidth=0.7,
                zorder=4,
            )
        style_axis(axis)
        axis.grid(False, axis="y")
        axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
        axis.set_title(f"Gap = {int(gap_size)} h", loc="left")
        axis.set_xlabel(metric)
        axis.invert_yaxis()
    for axis in axes.flat[len(gaps):]:
        axis.set_visible(False)
    add_plot_header(
        figure,
        f"Distribución de {metric} entre estaciones",
        "Cada punto es la media Monte Carlo de una estación; el diamante es la media entre estaciones.",
    )
    figure.subplots_adjust(top=0.89, hspace=0.28, wspace=0.34)
    return _save_formats(figure, output_stem)


def save_omnibus_plot(omnibus: pd.DataFrame, output_stem: Path) -> list[Path]:
    """Plot Friedman evidence by metric and gap size."""
    metrics = [metric for metric in ("MASE", "RMSSE") if metric in set(omnibus["metric"])]
    if not metrics:
        return []
    figure, axes = plt.subplots(1, len(metrics), figsize=(7.2 * len(metrics), 5.8), squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        frame = omnibus.loc[omnibus["metric"].astype(str) == metric].copy()
        frame["gap_size"] = pd.to_numeric(frame["gap_size"], errors="coerce")
        frame["p_value"] = pd.to_numeric(frame["p_value"], errors="coerce")
        frame = frame.sort_values("gap_size")
        finite = np.isfinite(frame["p_value"]) & (frame["p_value"] >= 0.0)
        plotted = frame.loc[finite]
        p_values = np.maximum(plotted["p_value"].to_numpy(dtype=float), np.finfo(float).tiny)
        evidence = -np.log10(p_values)
        axis.plot(
            plotted["gap_size"],
            evidence,
            color="#3d7ab5" if metric == "MASE" else "#cf6f1e",
            marker="o",
            linewidth=2,
            markersize=7,
        )
        missing = frame.loc[~finite]
        for x_value in missing["gap_size"]:
            axis.annotate(
                "Sin evidencia suficiente",
                (x_value, 0.0),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=MUTED_TEXT,
                rotation=25,
            )
        alpha_values = pd.to_numeric(frame["alpha"], errors="coerce").dropna()
        alpha = float(alpha_values.iloc[0]) if len(alpha_values) else 0.05
        axis.axhline(-math.log10(alpha), color="#a33f3f", linestyle="--", linewidth=1.2)
        for x_value, y_value, p_value in zip(plotted["gap_size"], evidence, p_values):
            label = f"p={p_value:.1e}" if p_value < 0.001 else f"p={p_value:.3f}"
            axis.annotate(label, (x_value, y_value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        style_axis(axis)
        axis.grid(True, axis="y", color=GRID_COLOR, linestyle="--", alpha=0.65)
        confirmatory = bool(frame["confirmatory"].iloc[0]) if len(frame) else False
        axis.set_title(
            f"{metric} ({'primario' if confirmatory else 'sensibilidad'})",
            loc="left",
        )
        axis.set_xlabel("Tamaño del gap (h)")
        axis.set_ylabel(r"Evidencia $-\log_{10}(p)$")
        axis.set_xticks(plotted["gap_size"])
    add_plot_header(
        figure,
        "Contrastes globales por tamaño de gap",
        "El contraste global compara los modelos sobre bloques pareados de estaciones; la línea discontinua marca alpha.",
    )
    figure.subplots_adjust(top=0.84, wspace=0.24)
    return _save_formats(figure, output_stem)


def save_pairwise_matrix(
    scores: pd.DataFrame,
    pairwise: pd.DataFrame,
    *,
    metric: str,
    gap_size: int,
    output_stem: Path,
) -> list[Path]:
    """Plot all Holm-adjusted pairwise rank-biserial effects for one gap."""
    models = _model_order(scores, metric)
    selected = pairwise.loc[
        (pairwise["metric"].astype(str) == metric)
        & (pd.to_numeric(pairwise["gap_size"], errors="coerce") == gap_size)
    ]
    if not models:
        return []
    if selected.empty:
        figure, axis = plt.subplots(figsize=(10.5, 4.0))
        axis.set_facecolor(AXIS_FACE)
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No se generaron comparaciones post-hoc para este gap.\n"
            "El contraste global no fue significativo o no hubo evidencia suficiente.",
            ha="center",
            va="center",
            color=MUTED_TEXT,
            fontsize=12,
            transform=axis.transAxes,
        )
        add_plot_header(
            figure,
            f"{metric} post-hoc · gap = {gap_size} h",
            "El post-hoc solo se ejecuta después de un contraste global significativo.",
        )
        figure.subplots_adjust(top=0.78)
        return _save_formats(figure, output_stem)
    n_models = len(models)
    matrix = np.full((n_models, n_models), np.nan)
    significant = np.zeros((n_models, n_models), dtype=bool)
    np.fill_diagonal(matrix, 0.0)
    index = {model: position for position, model in enumerate(models)}
    for row in selected.itertuples(index=False):
        left = str(row.left)
        right = str(row.right)
        if left not in index or right not in index:
            continue
        effect = float(row.rank_biserial)
        left_index = index[left]
        right_index = index[right]
        matrix[left_index, right_index] = effect
        matrix[right_index, left_index] = -effect
        is_significant = bool(row.statistically_significant)
        significant[left_index, right_index] = is_significant
        significant[right_index, left_index] = is_significant
    figure, axis = plt.subplots(figsize=(12.5, 10.5))
    figure.patch.set_facecolor(FIGURE_FACE)
    axis.set_facecolor(AXIS_FACE)
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    axis.set_xticks(range(n_models), labels=models, rotation=50, ha="right")
    axis.set_yticks(range(n_models), labels=models)
    axis.set_xticks(np.arange(-0.5, n_models, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, n_models, 1), minor=True)
    axis.grid(which="minor", color=FIGURE_FACE, linestyle="-", linewidth=1.2)
    axis.grid(False, which="major")
    for row_index in range(n_models):
        for column_index in range(n_models):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                continue
            suffix = "*" if significant[row_index, column_index] else ""
            color = "white" if abs(value) > 0.55 else TEXT_COLOR
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}{suffix}",
                ha="center",
                va="center",
                fontsize=6.7,
                color=color,
                fontweight="bold" if suffix else "normal",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlación biserial de rangos (fila - columna)")
    axis.set_title(f"{metric} post-hoc · gap = {gap_size} h", loc="left", color=TEXT_COLOR)
    add_plot_header(
        figure,
        "Magnitud y significación de las comparaciones pareadas",
        "Valores negativos (azul) favorecen al modelo de la fila; * indica Wilcoxon con p de Holm < alpha.",
    )
    figure.subplots_adjust(top=0.87)
    return _save_formats(figure, output_stem)


def _escape_latex(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return re.sub(r"[\\&%$#_{}~^]", lambda match: replacements[match.group(0)], str(value))


def _fmt_number(value: object, decimals: int = 3) -> str:
    number = float(value)
    return "--" if not np.isfinite(number) else f"{number:.{decimals}f}"


def _fmt_p(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "--"
    return "$<0.001$" if number < 0.001 else f"{number:.3f}"


def _latex_table(
    frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
    column_format: str,
) -> str:
    body = frame.to_latex(index=False, escape=False, column_format=column_format)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{_escape_latex(caption)}}}\n"
        f"\\label{{{_escape_latex(label)}}}\n"
        f"{body}"
        "\\end{table}\n"
    )


def _model_summary_latex(summary: pd.DataFrame, metric: str) -> str:
    gaps = sorted(pd.to_numeric(summary["gap_size"], errors="coerce").dropna().unique())
    model_order = (
        summary.groupby("model", sort=False)["mean_rank"]
        .mean()
        .sort_values(kind="stable")
        .index.astype(str)
        .tolist()
    )
    rows: list[dict[str, str]] = []
    for model in model_order:
        row: dict[str, str] = {"Modelo": _escape_latex(model)}
        model_summary = summary.loc[summary["model"].astype(str) == model]
        for gap_size in gaps:
            cell = model_summary.loc[
                pd.to_numeric(model_summary["gap_size"], errors="coerce") == gap_size
            ]
            if cell.empty:
                text = "--"
            else:
                item = cell.iloc[0]
                text = (
                    f"{_fmt_number(item['mean_score'])} $\\pm$ "
                    f"{_fmt_number(item['station_sd'])} [{_fmt_number(item['mean_rank'], 2)}]"
                )
                best = summary.loc[
                    pd.to_numeric(summary["gap_size"], errors="coerce") == gap_size,
                    "mean_score",
                ].min()
                if np.isclose(float(item["mean_score"]), float(best)):
                    text = f"\\textbf{{{text}}}"
            row[f"Gap {int(gap_size)} h"] = text
        row["Rango global"] = _fmt_number(
            model_summary["mean_rank"].mean(), 2
        )
        rows.append(row)
    display = pd.DataFrame(rows)
    return _latex_table(
        display,
        caption=(
            f"Rendimiento {metric} por tamaño de hueco: media y desviación "
            "estándar entre estaciones, con el rango medio entre corchetes; "
            "en negrita se muestra la menor media de cada gap."
        ),
        label=f"tab:imputation-statistical-{metric.lower()}",
        column_format="l" + "c" * (len(display.columns) - 1),
    )


def _omnibus_latex(omnibus: pd.DataFrame) -> str:
    rows = []
    for row in omnibus.sort_values(["metric", "gap_size"]).itertuples(index=False):
        p_value = _fmt_p(row.p_value)
        if bool(row.significant):
            p_value = f"\\textbf{{{p_value}}}"
        rows.append(
            {
                "Métrica": _escape_latex(row.metric),
                "Gap (h)": int(row.gap_size),
                "Rol": "Primario" if bool(row.confirmatory) else "Sensibilidad",
                "Test": _escape_latex(str(row.test).replace("_", " ").title()),
                "Estadístico": _fmt_number(row.statistic),
                "Estaciones": int(row.n_stations),
                "Modelos": int(row.n_models),
                "$p$": p_value,
            }
        )
    display = pd.DataFrame(rows)
    return _latex_table(
        display,
        caption="Contrastes globales independientes por métrica y tamaño de hueco.",
        label="tab:imputation-statistical-omnibus",
        column_format="lccccccc",
    )


def _top_comparisons_latex(comparisons: pd.DataFrame) -> str:
    rows = []
    for row in comparisons.itertuples(index=False):
        p_holm = _fmt_p(row.p_holm)
        if bool(row.statistically_significant):
            p_holm = f"\\textbf{{{p_holm}}}"
        rows.append(
            {
                "Gap (h)": int(row.gap_size),
                "Referencia": _escape_latex(row.reference),
                "Comparador": _escape_latex(row.comparison),
                "$\\Delta$": _fmt_number(row.mean_difference),
                "Mejora (\\%)": _fmt_number(row.improvement_pct, 1),
                "$r_{rb}$": _fmt_number(row.rank_biserial, 2),
                "V/E/D": f"{int(row.wins)}/{int(row.ties)}/{int(row.losses)}",
                "$p_{Holm}$": p_holm,
            }
        )
    display = pd.DataFrame(
        rows,
        columns=[
            "Gap (h)",
            "Referencia",
            "Comparador",
            "$\\Delta$",
            "Mejora (\\%)",
            "$r_{rb}$",
            "V/E/D",
            "$p_{Holm}$",
        ],
    )
    return _latex_table(
        display,
        caption=(
            "Comparaciones MASE del modelo con mejor rango descriptivo frente a "
            "los otros modelos del top-3. Delta es referencia menos comparador: "
            "Delta negativo, mejora positiva y rrb negativo favorecen a la referencia; "
            "V/E/D adopta su perspectiva. Holm se calculó sobre todas las parejas "
            "por separado dentro de cada gap."
        ),
        label="tab:imputation-statistical-top-pairwise-mase",
        column_format="cllccccc",
    )


def _report_readme(top_k: int) -> str:
    return f"""# Informe estadístico de imputación

Este directorio se genera exclusivamente desde los CSV estadísticos persistidos.
No vuelve a ejecutar modelos ni contrastes.

## Figuras

- `figures/station_scores_mase.pdf`: distribución primaria por estaciones y gap.
- `figures/station_scores_rmsse.pdf`: análisis de sensibilidad RMSSE.
- `figures/omnibus_by_gap.pdf`: evidencia de los contrastes globales.
- `figures/pairwise_matrix_mase_gap_*.pdf`: efecto biserial y significación de Holm.

Cada figura también está disponible como PNG a 300 dpi.

## Tablas

- `tables/model_summary_mase.tex`: resumen primario de todos los modelos.
- `tables/model_summary_rmsse.tex`: resumen de sensibilidad.
- `tables/omnibus.tex`: contrastes globales por métrica y gap.
- `tables/top_comparisons_mase.tex`: referencia descriptiva frente a los otros
  {top_k - 1} modelos del top-{top_k}; conserva el Holm de la familia completa.

Las tablas usan reglas de `booktabs`. Ejemplo mínimo de inclusión:

```latex
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}

\\input{{statistical_report/tables/model_summary_mase.tex}}

\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=\\textwidth]{{statistical_report/figures/station_scores_mase.pdf}}
  \\caption{{Distribución de MASE por estación y tamaño de hueco.}}
\\end{{figure}}
```

Los porcentajes de mejora son descriptivos y no representan un umbral de
relevancia práctica.
"""


def render_imputation_statistical_report(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    top_k: int = 3,
) -> list[Path]:
    """Create publication figures and tables without rerunning statistical tests."""
    if top_k < 2:
        raise ValueError("top_k debe ser al menos 2")
    run_dir = Path(run_dir)
    tables = load_statistical_tables(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "statistical_report"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for metric in ("MASE", "RMSSE"):
        outputs.extend(
            save_station_score_plot(
                tables["scores"],
                metric=metric,
                output_stem=figure_dir / f"station_scores_{metric.lower()}",
            )
        )
    outputs.extend(save_omnibus_plot(tables["omnibus"], figure_dir / "omnibus_by_gap"))
    mase_gaps = sorted(
        pd.to_numeric(
            tables["omnibus"].loc[
                tables["omnibus"]["metric"].astype(str) == "MASE", "gap_size"
            ],
            errors="coerce",
        ).dropna().astype(int).unique()
    )
    for gap_size in mase_gaps:
        outputs.extend(
            save_pairwise_matrix(
                tables["scores"],
                tables["pairwise"],
                metric="MASE",
                gap_size=int(gap_size),
                output_stem=figure_dir / f"pairwise_matrix_mase_gap_{gap_size}",
            )
        )

    for metric in ("MASE", "RMSSE"):
        summary = build_model_summary(tables["scores"], metric=metric)
        csv_path = table_dir / f"model_summary_{metric.lower()}.csv"
        tex_path = table_dir / f"model_summary_{metric.lower()}.tex"
        summary.to_csv(csv_path, index=False)
        tex_path.write_text(_model_summary_latex(summary, metric), encoding="utf-8")
        outputs.extend([csv_path, tex_path])

    omnibus_csv = table_dir / "omnibus.csv"
    omnibus_tex = table_dir / "omnibus.tex"
    tables["omnibus"].to_csv(omnibus_csv, index=False)
    omnibus_tex.write_text(_omnibus_latex(tables["omnibus"]), encoding="utf-8")
    outputs.extend([omnibus_csv, omnibus_tex])

    comparisons = build_top_comparisons(
        tables["scores"], tables["pairwise"], metric="MASE", top_k=top_k
    )
    comparisons_csv = table_dir / "top_comparisons_mase.csv"
    comparisons_tex = table_dir / "top_comparisons_mase.tex"
    comparisons.to_csv(comparisons_csv, index=False)
    comparisons_tex.write_text(
        _top_comparisons_latex(comparisons), encoding="utf-8"
    )
    outputs.extend([comparisons_csv, comparisons_tex])
    readme_path = output_dir / "README.md"
    readme_path.write_text(_report_readme(top_k), encoding="utf-8")
    outputs.append(readme_path)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Modelos del ranking descriptivo incluidos en la tabla post-hoc",
    )
    args = parser.parse_args(argv)
    outputs = render_imputation_statistical_report(
        args.run_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    for path in outputs:
        print(f"[info] {path}")


if __name__ == "__main__":
    main()

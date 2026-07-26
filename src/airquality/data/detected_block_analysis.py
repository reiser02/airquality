"""Detector-aware audit of contiguous NO2 blocks.

Run with::

    uv run python -m airquality.data.detected_block_analysis

The report compares raw data, every available registered detector, the
rate-filtered unlabeled consensus, and injection-ranked top-3 voting.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airquality.anomaly.ensemble import rank_top_k
from airquality.anomaly.metrics import detect_mask
from airquality.anomaly.presentation import EDGE_COLOR, FIGURE_FACE, GRID_COLOR, TEXT_COLOR
from airquality.anomaly.registry import (
    filter_model_kwargs,
    resolve_model_class,
    resolve_model_names,
)
from airquality.data.block_analysis import observed_blocks
from airquality.data.loaders import load_raw_5m
from airquality.data.preprocessing import MIN_RUN, MIN_USEFUL, preprocess
from airquality.data.series import ensure_datetime_series
from airquality.forecasting.cache import CACHE_VERSION, BenchmarkCache, series_fingerprint
from airquality.forecasting.detection import SeriesDetectionContext, build_detection_strategy
from airquality.paths import create_run_dir


DETECTORS = tuple(resolve_model_names(["all"]))
THRESHOLDS = (80, 120, 152, 216)
MAIN_SCENARIOS = ("Raw", "Consenso unlabeled", "Inject-vote")
LENGTH_BINS = (0, 7, 23, 79, 119, 151, 215, 511, np.inf)
LENGTH_LABELS = ("1-7", "8-23", "24-79", "80-119", "120-151", "152-215", "216-511", "512+")
MODEL_DESCRIPTIONS = {
    "ModifiedZScore": "z-score robusto global basado en mediana y MAD",
    "IQR": "distancia fuera de las vallas de Tukey Q1/Q3 +/- 1,5 IQR",
    "IsolationForest": "Isolation Forest de scikit-learn (100 arboles)",
    "LOF": "Local Outlier Factor de scikit-learn (hasta 50 vecinos)",
    "PCA": "PCA sobre ventanas deslizantes de 80 h",
    "COUTABase": "COUTA con generador sintético nativo y ventanas de 80 h",
    "COUTAGenIAS": "COUTA con generador GenIAS y ventanas de 80 h",
    "CARLABase": "CARLA contrastivo con generador nativo y ventanas de 80 h",
    "CARLAGenIAS": "CARLA contrastivo con generador GenIAS y ventanas de 80 h",
    "LSTMAD": "LSTM-AD por error de predicción, mínimo de 80 h",
    "Hampel_w24": "filtro Hampel con mediana y MAD moviles de 24 h",
    "Hampel_w6": "filtro Hampel con mediana y MAD moviles de 6 h",
    "Prophet": "exceso sobre el intervalo predictivo de Prophet",
    "TSPulse": "IBM Granite TSPulse zero-shot; mínimo de 80 h sobre contexto interno de 512 h",
}


def _model_minimum(detector: str) -> int:
    model_cls = resolve_model_class(detector)
    kwargs = filter_model_kwargs(model_cls, {"device": "cpu"})
    return int(model_cls(seed=13, **kwargs).minimum_series_length)


def _station_stats(
    scenario: str,
    station: str,
    series: pd.Series,
    mask: pd.Series,
    scored_mask: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame]:
    cleaned = series.mask(mask.reindex(series.index, fill_value=False))
    blocks = observed_blocks(cleaned)
    observed = series.notna()
    scored_points = int((scored_mask.reindex(series.index, fill_value=False) & observed).sum())
    input_points = int(observed.sum())
    row: dict[str, object] = {
        "scenario": scenario,
        "station": station,
        "input_observed_points": input_points,
        "detection_scored_points": scored_points,
        "detection_unscored_points": input_points - scored_points,
        "detection_coverage_pct": 100.0 * scored_points / input_points if input_points else 0.0,
        "flagged_points": int((mask & observed).sum()),
        "remaining_points": int(cleaned.notna().sum()),
        "total_blocks": len(blocks),
        "median_block_hours": float(blocks["hours"].median()) if len(blocks) else 0.0,
        "p90_block_hours": float(blocks["hours"].quantile(0.9)) if len(blocks) else 0.0,
        "max_block_hours": int(blocks["hours"].max()) if len(blocks) else 0,
    }
    for minimum in THRESHOLDS:
        eligible = blocks["hours"].ge(minimum)
        points = int(blocks.loc[eligible, "hours"].sum())
        row[f"blocks_ge_{minimum}"] = int(eligible.sum())
        row[f"points_ge_{minimum}"] = points
        row[f"points_outside_{minimum}"] = row["input_observed_points"] - points
    blocks.insert(0, "station", station)
    blocks.insert(0, "scenario", scenario)
    return row, blocks


def _summarize(series_summary: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in series_summary["scenario"].drop_duplicates():
        station_rows = series_summary.loc[series_summary["scenario"] == scenario]
        scenario_blocks = blocks.loc[blocks["scenario"] == scenario]
        input_points = int(station_rows["input_observed_points"].sum())
        flagged = int(station_rows["flagged_points"].sum())
        row: dict[str, object] = {
            "scenario": scenario,
            "series": int(station_rows["station"].nunique()),
            "input_observed_points": input_points,
            "detection_scored_points": int(station_rows["detection_scored_points"].sum()),
            "detection_unscored_points": int(station_rows["detection_unscored_points"].sum()),
            "flagged_points": flagged,
            "flagged_pct": 100.0 * flagged / input_points if input_points else 0.0,
            "remaining_points": int(station_rows["remaining_points"].sum()),
            "total_blocks": len(scenario_blocks),
            "median_block_hours": float(scenario_blocks["hours"].median()) if len(scenario_blocks) else 0.0,
            "p90_block_hours": float(scenario_blocks["hours"].quantile(0.9)) if len(scenario_blocks) else 0.0,
            "max_block_hours": int(scenario_blocks["hours"].max()) if len(scenario_blocks) else 0,
        }
        row["detection_coverage_pct"] = (
            100.0 * row["detection_scored_points"] / input_points if input_points else 0.0
        )
        for minimum in THRESHOLDS:
            eligible = scenario_blocks["hours"].ge(minimum)
            points = int(scenario_blocks.loc[eligible, "hours"].sum())
            row[f"blocks_ge_{minimum}"] = int(eligible.sum())
            row[f"points_ge_{minimum}"] = points
            row[f"points_outside_{minimum}"] = input_points - points
        rows.append(row)

    summary = pd.DataFrame(rows)
    raw = summary.loc[summary["scenario"] == "Raw"].iloc[0]
    for minimum in THRESHOLDS:
        summary[f"additional_points_outside_{minimum}_vs_raw"] = (
            summary[f"points_outside_{minimum}"] - raw[f"points_outside_{minimum}"]
        )
    summary["additional_blocks_vs_raw"] = summary["total_blocks"] - raw["total_blocks"]
    return summary


def _block_distribution(blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in blocks["scenario"].drop_duplicates():
        subset = blocks.loc[blocks["scenario"] == scenario].copy()
        subset["length_bin"] = pd.cut(
            subset["hours"], LENGTH_BINS, labels=LENGTH_LABELS, include_lowest=True
        )
        grouped = subset.groupby("length_bin", observed=False)["hours"].agg(["size", "sum"])
        for label in LENGTH_LABELS:
            rows.append(
                {
                    "scenario": scenario,
                    "length_bin": label,
                    "blocks": int(grouped.loc[label, "size"]),
                    "points": int(grouped.loc[label, "sum"]),
                }
            )
    return pd.DataFrame(rows)


def _format_count(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _save_retention(path: Path, summary: pd.DataFrame) -> None:
    selected = summary.set_index("scenario").loc[list(MAIN_SCENARIOS)]
    scenario_labels = [
        f"{scenario}\nAnomalías: {selected.loc[scenario, 'flagged_pct']:.2f}%"
        for scenario in selected.index
    ]
    raw_blocks = float(selected.loc["Raw", "total_blocks"])
    raw_hours = float(selected.loc["Raw", "input_observed_points"])
    figure, axes = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True, facecolor=FIGURE_FACE)
    colors = ("#8c6d4b", "#5b8c5a")
    for axis, minimum, title in zip(
        axes, (80, 120), ("Short: bloques de al menos 80 h", "Long: bloques de al menos 120 h"), strict=True
    ):
        x = np.arange(len(selected))
        counts = selected[f"blocks_ge_{minimum}"].to_numpy(dtype=int)
        hours = selected[f"points_ge_{minimum}"].to_numpy(dtype=int)
        block_pct = 100.0 * counts / raw_blocks
        hour_pct = 100.0 * hours / raw_hours
        bars_a = axis.bar(x - 0.18, block_pct, 0.36, color=colors[0], edgecolor=EDGE_COLOR, label="Bloques elegibles")
        bars_b = axis.bar(x + 0.18, hour_pct, 0.36, color=colors[1], edgecolor=EDGE_COLOR, label="Horas retenidas")
        for bars, percentages, values, unit in (
            (bars_a, block_pct, counts, "bloques"),
            (bars_b, hour_pct, hours, "horas"),
        ):
            axis.bar_label(
                bars,
                labels=[f"{pct:.1f}%\n{_format_count(int(value))} {unit}" for pct, value in zip(percentages, values, strict=True)],
                padding=4,
                fontsize=9,
                color=TEXT_COLOR,
            )
        axis.set_xticks(x, scenario_labels)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylim(0, max(105.0, float(np.max([*block_pct, *hour_pct])) * 1.15))
        axis.set_facecolor("#fffaf2")
        axis.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.65)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Porcentaje del universo raw")
    figure.suptitle(
        "Bloques elegibles y horas retenidas tras detectar anomalías",
        x=0.04, y=0.98, ha="left", fontsize=17, fontweight="bold", color=TEXT_COLOR,
    )
    figure.text(
        0.04, 0.91,
        f"Denominadores fijos: {_format_count(int(raw_blocks))} bloques raw y {_format_count(int(raw_hours))} horas raw. "
        "Se usa toda la serie preprocesada; no se descuenta holdout ni validación.",
        ha="left", fontsize=10, color="#6d6258",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=2, loc="lower center")
    figure.tight_layout(rect=(0.03, 0.10, 0.99, 0.86), w_pad=3)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)


def _save_block_distribution(path: Path, blocks: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True, facecolor=FIGURE_FACE)
    colors = ("#8c877e", "#3f8785", "#9b59b6")
    max_hours = int(blocks["hours"].max())
    bins = np.geomspace(1, max_hours + 1, 48)
    for axis, scenario, color in zip(axes, MAIN_SCENARIOS, colors, strict=True):
        lengths = blocks.loc[blocks["scenario"] == scenario, "hours"]
        axis.hist(lengths, bins=bins, color=color, alpha=0.9)
        axis.axvline(80, color="#397dcc", linestyle="--")
        axis.axvline(120, color="#df7216", linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(scenario)
        axis.set_xlabel("Longitud del bloque (h)")
        axis.grid(True, which="major", linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Número de bloques")
    figure.suptitle("Distribución de bloques NO2 tras retirar anomalías", fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_impact(path: Path, summary: pd.DataFrame) -> None:
    selected = summary.loc[summary["scenario"] != "Raw"]
    x = np.arange(len(selected))
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, facecolor=FIGURE_FACE)
    width = 0.2
    for position, minimum in enumerate(THRESHOLDS):
        axes[0].bar(
            x + (position - 1.5) * width,
            selected[f"additional_points_outside_{minimum}_vs_raw"],
            width,
            label=f">= {minimum} h",
        )
    axes[1].bar(x, selected["additional_blocks_vs_raw"], color="#8c6d4b")
    axes[0].set_ylabel("Puntos adicionales fuera vs raw")
    axes[1].set_ylabel("Bloques adicionales vs raw")
    axes[1].set_xticks(x, selected["scenario"], rotation=35, ha="right")
    axes[0].legend(ncol=4)
    for axis in axes:
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Impacto de la detección sobre bloques y soporte utilizable", fontsize=16)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_rates(path: Path, coverage: pd.DataFrame, max_rate: float) -> None:
    values = [
        coverage.loc[coverage["detector"] == detector, "detection_rate_pct"].to_numpy()
        for detector in DETECTORS
    ]
    figure, axis = plt.subplots(figsize=(12, 8.5), facecolor=FIGURE_FACE)
    axis.boxplot(values, vert=False, tick_labels=DETECTORS)
    axis.axvline(100.0 * max_rate, color="#bd3b37", linestyle="--", label=f"límite {100 * max_rate:g}%")
    axis.set_xlabel("Puntos marcados sobre puntos puntuados (%)")
    axis.set_title("Distribución de tasas de detección por estación")
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_readme(
    path: Path,
    summary: pd.DataFrame,
    *,
    threshold_k: float,
    max_rate: float,
    injection_seed: int,
    min_selection_points: int,
) -> None:
    table = summary.loc[summary["scenario"].isin(MAIN_SCENARIOS)]
    result_lines = [
        "| Escenario | Cobertura detección | Horas detectadas | % del total | Bloques | Puntos en bloques >=80 h | Puntos en bloques >=120 h |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for row in table.itertuples(index=False):
        result_lines.append(
            f"| {row.scenario} | {row.detection_coverage_pct:.2f}% | {_format_count(row.flagged_points)} | "
            f"{row.flagged_pct:.2f}% | {_format_count(row.total_blocks)} | "
            f"{_format_count(row.points_ge_80)} | {_format_count(row.points_ge_120)} |"
        )
    model_lines = [
        f"- `{name}` (mínimo {_model_minimum(name)} h): {MODEL_DESCRIPTIONS[name]}."
        for name in DETECTORS
    ]
    short_detectors = ", ".join(
        f"`{name}`" for name in DETECTORS if _model_minimum(name) < 80
    )
    path.write_text(
        "\n".join(
            [
                "# Detección completa NO2 y distribución de bloques",
                "",
                "## Protocolo",
                "",
                f"- Series preprocesadas: {int(table.iloc[0]['series'])}",
                f"- Puntos horarios observados raw: {_format_count(int(table.iloc[0]['input_observed_points']))}",
                f"- Umbral por racha: mediana + {threshold_k:g} x 1.4826 x MAD.",
                f"- `Consenso unlabeled`: conserva detectores con tasa <= {100 * max_rate:g}% por estación y exige mayoría estricta.",
                "- `Inject-vote`: ranking VUS-PR local sobre copias con anomalías sintéticas `combined`; elige los 3 primeros detectores aptos de cada bloque y exige 2 votos sobre la serie real.",
                f"- Inyección de selección: semilla {injection_seed}; las rachas de al menos {min_selection_points} h tienen ranking local y las menores heredan la media de la estación.",
                f"- Rachas de 8 a 79 h: solo participan los modelos aptos: {short_detectors}.",
                "- Rachas menores de 8 h: se conservan, pero quedan explícitamente sin puntuar.",
                "- No se concatenan rachas separadas por NaN o saltos temporales.",
                "- Cada detector se ajusta una vez por estación sobre todas sus rachas aptas y se puntúa por racha sin cruzar huecos.",
                "- Se usa la serie horaria completa de cada estación; este diagnóstico no reserva holdout ni validación.",
                "",
                "## Modelos usados",
                "",
                *model_lines,
                "",
                "## Resultado principal",
                "",
                *result_lines,
                "",
                "La inyección solo selecciona los detectores de `Inject-vote`; la máscara final siempre se calcula sobre la serie real sin inyectar.",
                "",
                "## Artefactos",
                "",
                "- `summary.csv`: comparación agregada de raw, detectores individuales y estrategias.",
                "- `series_summary.csv`: impacto por estación.",
                "- `blocks.csv`: todos los bloques resultantes.",
                "- `block_distribution.csv`: distribución por intervalos de longitud.",
                "- `detector_coverage.csv`: cobertura, tasa, filtro unlabeled y segmentos sin puntuar por detector/estación.",
                "- `selection.csv`: VUS-PR, disponibilidad y selección de `Inject-vote` por bloque.",
                "- `manifest.json`: parámetros efectivos de la ejecución.",
                "- `retention_overview.png`: bloques y horas retenidos con denominadores raw fijos.",
                "- `block_length_distribution.png`: distribución raw/consenso unlabeled/inject-vote.",
                "- `block_impact_by_detector.png`: pérdida de soporte y fragmentación frente a raw.",
                "- `detection_rate_distribution.png`: tasas individuales y límite del 7%.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_analysis(
    *,
    base_dir: Path,
    output_dir: Path,
    cache_dir: Path | None,
    threshold_k: float = 3.5,
    max_rate: float = 0.07,
    seed: int = 13,
    injection_seed: int = 101,
    min_selection_points: int = 300,
    min_run: int = MIN_RUN,
    min_useful: int = MIN_USEFUL,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = BenchmarkCache(cache_dir)
    series_rows: list[dict[str, object]] = []
    block_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    stations = load_raw_5m("NO2", str(base_dir))
    if not stations:
        raise FileNotFoundError(f"No se encontraron series NO2 bajo {base_dir}")

    for position, (station, raw) in enumerate(stations, start=1):
        (hourly,), _ = preprocess([raw], "NO2", min_run=min_run, min_useful=min_useful)
        series = ensure_datetime_series(hourly.iloc[:, 0], freq="h", name=station)
        context = SeriesDetectionContext(
            series,
            detectors=list(DETECTORS),
            seed=seed,
            device="cpu",
            injection_seed=injection_seed,
            min_selection_points=min_selection_points,
            cache=cache,
            cache_key={
                "analysis": "detected-blocks-v2",
                "cache_version": CACHE_VERSION,
                "series": series_fingerprint(series),
                "threshold_k": threshold_k,
                "detectors": list(DETECTORS),
                "seed": seed,
            },
        )
        detector_masks: dict[str, pd.Series] = {}
        detector_scored_masks: dict[str, pd.Series] = {}
        detector_details: dict[str, dict[str, object]] = {}
        for detector in DETECTORS:
            started = time.perf_counter()
            score_lists = context.real_scores([detector])
            elapsed = time.perf_counter() - started
            full_mask = context.empty_mask()
            full_scored_mask = context.empty_mask()
            scored_points = flagged_points = 0
            failed_segments = len(observed_blocks(series)) - len(context.segments)
            scores_for_detector = score_lists.get(detector, [None] * len(context.segments))
            for segment, scores in zip(context.segments, scores_for_detector, strict=True):
                if scores is None:
                    failed_segments += 1
                    continue
                segment_mask = detect_mask(scores, threshold_k).astype(bool)
                full_mask.loc[segment.index] = segment_mask
                full_scored_mask.loc[segment.index] = np.isfinite(scores)
                scored_points += int(np.isfinite(scores).sum())
                flagged_points += int(segment_mask.sum())
            detector_masks[detector] = full_mask
            detector_scored_masks[detector] = full_scored_mask
            detector_details[detector] = {
                "scored_points": scored_points,
                "flagged_points": flagged_points,
                "failed_segments": failed_segments,
                "real_score_seconds": elapsed,
            }

        unlabeled = build_detection_strategy(
            "unlabeled", threshold_k=threshold_k, max_detection_rate=max_rate
        ).detect(context)
        inject_vote = build_detection_strategy(
            "inject-vote", threshold_k=threshold_k, vote_top_k=3, vote_min_votes=2
        ).detect(context)

        observed_points = int(series.notna().sum())
        for detector in DETECTORS:
            details = detector_details[detector]
            scored = int(details["scored_points"])
            coverage_rows.append(
                {
                    "station": station,
                    "detector": detector,
                    "minimum_series_length": _model_minimum(detector),
                    "observed_points": observed_points,
                    "scored_points": scored,
                    "coverage_pct": 100.0 * scored / observed_points if observed_points else 0.0,
                    "flagged_points": details["flagged_points"],
                    "detection_rate_pct": 100.0 * int(details["flagged_points"]) / scored if scored else 0.0,
                    "kept_for_unlabeled": detector in unlabeled.detectors,
                    "unscored_segments": details["failed_segments"],
                    "real_score_seconds": details["real_score_seconds"],
                }
            )

        selection_indices = set(context.selection_segment_indices())
        for segment_index, (segment, ranking, selected) in enumerate(
            zip(
                context.segments,
                context.selection_rankings(),
                inject_vote.selected_by_segment,
                strict=True,
            )
        ):
            available = {
                detector
                for detector, scores in context.real_scores(list(ranking)).items()
                if scores[segment_index] is not None
            }
            for rank, detector in enumerate(
                rank_top_k(ranking, len(ranking)), start=1
            ):
                selection_rows.append(
                    {
                        "station": station,
                        "segment_index": segment_index,
                        "start": segment.index[0],
                        "end": segment.index[-1],
                        "hours": len(segment),
                        "ranking_source": "local" if segment_index in selection_indices else "station_mean",
                        "detector": detector,
                        "vus_pr": ranking[detector],
                        "rank": rank,
                        "score_available": detector in available,
                        "selected_for_vote": detector in selected,
                    }
                )

        scenarios = {
            "Raw": (context.empty_mask(), context.series.notna()),
            **{
                detector: (detector_masks[detector], detector_scored_masks[detector])
                for detector in DETECTORS
            },
            "Consenso unlabeled": (unlabeled.mask, unlabeled.scored_mask),
            "Inject-vote": (inject_vote.mask, inject_vote.scored_mask),
        }
        for scenario, (mask, scored_mask) in scenarios.items():
            row, scenario_blocks = _station_stats(
                scenario, station, context.series, mask, scored_mask
            )
            series_rows.append(row)
            block_frames.append(scenario_blocks)
        print(f"[{position}/{len(stations)}] {station}: unlabeled={unlabeled.n_flagged}, inject-vote={inject_vote.n_flagged}", flush=True)

    series_summary = pd.DataFrame(series_rows)
    blocks = pd.concat(block_frames, ignore_index=True)
    summary = _summarize(series_summary, blocks)
    coverage = pd.DataFrame(coverage_rows)
    selection = pd.DataFrame(selection_rows)
    distribution = _block_distribution(blocks)
    paths = {
        "summary": output_dir / "summary.csv",
        "series_summary": output_dir / "series_summary.csv",
        "blocks": output_dir / "blocks.csv",
        "block_distribution": output_dir / "block_distribution.csv",
        "detector_coverage": output_dir / "detector_coverage.csv",
        "selection": output_dir / "selection.csv",
        "manifest": output_dir / "manifest.json",
        "readme": output_dir / "README.md",
    }
    summary.to_csv(paths["summary"], index=False)
    series_summary.to_csv(paths["series_summary"], index=False)
    blocks.to_csv(paths["blocks"], index=False)
    distribution.to_csv(paths["block_distribution"], index=False)
    coverage.to_csv(paths["detector_coverage"], index=False)
    selection.to_csv(paths["selection"], index=False)
    paths["manifest"].write_text(
        json.dumps(
            {
                "pollutant": "NO2",
                "detectors": list(DETECTORS),
                "strategies": ["unlabeled", "inject-vote"],
                "threshold_k": threshold_k,
                "max_detection_rate": max_rate,
                "seed": seed,
                "injection_variant": "combined",
                "injection_seed": injection_seed,
                "min_selection_points": min_selection_points,
                "vote_top_k": 3,
                "vote_min_votes": 2,
                "min_run": min_run,
                "min_useful": min_useful,
                "minimum_series_length": {
                    detector: _model_minimum(detector) for detector in DETECTORS
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    _write_readme(
        paths["readme"], summary, threshold_k=threshold_k, max_rate=max_rate,
        injection_seed=injection_seed, min_selection_points=min_selection_points,
    )
    _save_retention(output_dir / "retention_overview.png", summary)
    _save_block_distribution(output_dir / "block_length_distribution.png", blocks)
    _save_impact(output_dir / "block_impact_by_detector.png", summary)
    _save_rates(output_dir / "detection_rate_distribution.png", coverage, max_rate)
    print(f"\n{summary.to_markdown(index=False)}")
    print(f"\n{cache.stats()}")
    print(f"Informe guardado en: {output_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita bloques NO2 tras unlabeled e inject-vote")
    parser.add_argument("--base-dir", type=Path, default=Path("data/raw/datos_estaciones_5m"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/data_blocks/detection_cache"))
    args = parser.parse_args()
    output = args.output_dir or create_run_dir(
        Path("reports/data_blocks"), f"detected_NO2_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_analysis(base_dir=args.base_dir, output_dir=output, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()

"""Config-driven entrypoint comparing forecasting on raw vs preprocessed series.

For every configured series the pipeline builds two arms and backtests the same
forecasting models on each, over the **same** observed holdout window:

- **raw**: the hourly-mean series as loaded (gaps + anomalies kept).
- **preprocessed**: anomalies detected and removed (:mod:`airquality.forecasting.cleaning`),
  then gaps imputed (:mod:`airquality.forecasting.fill`).

Detection and imputation touch only the training portion; the evaluation window
(context + holdout) stays the raw observed values for both arms, so the forecast
error difference reflects the effect of preprocessing on the trained model.

Run with::

    uv run python -m airquality.forecasting.pipeline
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from airquality.config import cfg_get_csv_list, cfg_get_float, cfg_get_int, cfg_get_str
from airquality.data.io import load_and_normalize_series
from airquality.forecasting.backtest import backtest_forecast, select_holdout_window
from airquality.forecasting.cleaning import detect_anomaly_mask, remove_anomalies
from airquality.forecasting.fill import build_imputer, impute_series


def _repo_root() -> Path:
    """Return the repository root (three levels above ``src/airquality/forecasting/``)."""
    return Path(__file__).resolve().parents[3]


def _build_output_dir() -> Path:
    """Create the timestamped output directory for one comparison run."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _repo_root() / "reports" / "comparison" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot raw vs preprocessed metrics per (series, model) and add deltas."""
    if results_df.empty:
        return pd.DataFrame()

    metric_cols = ["rmse", "mae", "mase"]
    wide = results_df.pivot_table(
        index=["series", "model"], columns="arm", values=metric_cols, aggfunc="first"
    )
    rows: list[dict[str, Any]] = []
    for (series, model), row in wide.iterrows():
        entry: dict[str, Any] = {"series": series, "model": model}
        for metric in metric_cols:
            raw_val = row.get((metric, "raw"))
            pre_val = row.get((metric, "preprocessed"))
            entry[f"{metric}_raw"] = raw_val
            entry[f"{metric}_pre"] = pre_val
            if pd.notna(raw_val) and pd.notna(pre_val):
                entry[f"{metric}_delta"] = float(pre_val) - float(raw_val)
                entry[f"{metric}_improve_pct"] = (
                    100.0 * (float(raw_val) - float(pre_val)) / float(raw_val)
                    if raw_val
                    else float("nan")
                )
        rows.append(entry)
    return pd.DataFrame(rows)


def run_comparison_from_config() -> dict[str, Any]:
    """Run the raw-vs-preprocessed forecasting comparison defined by the config."""
    freq = cfg_get_str("data", "freq", "h")
    size_k = cfg_get_int("benchmark", "size_k", 5)
    seasonality_m = cfg_get_int("benchmark", "seasonality_m", 24)

    holdout = cfg_get_int("forecasting", "holdout", 168)
    context_len = cfg_get_int("forecasting", "context_len", 72)
    min_series_points = cfg_get_int("forecasting", "min_series_points", 600)
    seed = cfg_get_int("forecasting", "seed", 13)
    device = cfg_get_str("forecasting", "device", "cpu")
    threshold_k = cfg_get_float("forecasting", "threshold_k", 3.5)
    max_detection_rate = cfg_get_float("forecasting", "max_detection_rate", 0.07)
    detectors = list(cfg_get_csv_list("forecasting", "detectors", ("all",)))
    imputation_model = cfg_get_str("forecasting", "imputation_model", "interp")
    forecast_models = list(
        cfg_get_csv_list("forecasting", "forecast_models", ("NLinear", "TiDE"))
    )

    series_dfs = load_and_normalize_series(freq=freq, name_from_path=True)
    if not series_dfs:
        raise RuntimeError("No se cargaron series para la comparacion.")

    imputer = build_imputer(imputation_model, freq=freq, size_k=size_k)

    rows: list[dict[str, Any]] = []
    for df in series_dfs:
        series = df.iloc[:, 0]
        name = str(series.name)
        if int(series.notna().sum()) < min_series_points:
            print(f"[skip] {name}: pocos puntos observados")
            continue

        window = select_holdout_window(series, holdout=holdout, context_len=context_len, freq=freq)
        if window is None:
            print(f"[skip] {name}: sin ventana observada contigua >= {context_len + holdout}")
            continue

        holdout_start = window["holdout_start"]
        eval_obs = series.loc[window["eval_index"]]
        train_raw = series.loc[:holdout_start].iloc[:-1]

        # Preprocessed training arm: detect + remove anomalies, then impute gaps.
        cleaning = detect_anomaly_mask(
            train_raw,
            detectors=detectors,
            seed=seed,
            device=device,
            freq=freq,
            threshold_k=threshold_k,
            max_detection_rate=max_detection_rate,
        )
        train_clean = remove_anomalies(train_raw, cleaning)
        train_pre = impute_series(
            train_clean, imputer, freq=freq, use_scaler=imputation_model not in ("interp", "LinearInterp")
        )
        print(
            f"[info] {name}: detectores={','.join(cleaning.detectors) or 'none'} "
            f"descartados={','.join(cleaning.discarded) or 'none'} "
            f"tasa={cleaning.detection_rate:.2%} anomalias={cleaning.n_flagged}"
        )

        for model_name in forecast_models:
            raw_res = backtest_forecast(
                train_raw, eval_obs, model_name,
                size_k=size_k, holdout_start=holdout_start,
                seasonality_m=seasonality_m, freq=freq,
            )
            pre_res = backtest_forecast(
                train_pre, eval_obs, model_name,
                size_k=size_k, holdout_start=holdout_start,
                seasonality_m=seasonality_m, freq=freq,
            )
            common = {
                "series": name,
                "model": model_name,
                "detectors": ",".join(cleaning.detectors),
                "n_anomalies": cleaning.n_flagged,
                "imputation_model": imputation_model,
            }
            rows.append({**common, "arm": "raw", **{k: raw_res[k] for k in ("rmse", "mae", "mase", "n_eval")}})
            rows.append({**common, "arm": "preprocessed", **{k: pre_res[k] for k in ("rmse", "mae", "mase", "n_eval")}})

    results_df = pd.DataFrame(rows)
    summary_df = _summarize(results_df)

    output_dir = _build_output_dir()
    results_df.to_csv(output_dir / "comparison.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    return {"output_dir": output_dir, "results_df": results_df, "summary_df": summary_df}


def main() -> None:
    """Execute the comparison and print a short summary plus artifact locations."""
    artifacts = run_comparison_from_config()
    summary_df = artifacts["summary_df"]
    output_dir = artifacts["output_dir"]

    if summary_df.empty:
        print("[info] Comparacion sin filas (revisa datos / holdout / min_series_points).")
    else:
        print("[info] Resumen raw vs preprocesado (mejora % positiva = preprocesado mejor)")
        print(summary_df.to_string(index=False))
    print(f"[info] Artefactos en {output_dir}")


if __name__ == "__main__":
    main()

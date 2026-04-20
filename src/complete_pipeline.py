from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae as darts_mae
from darts.metrics import mase
from darts.metrics import rmse as darts_rmse
from darts.metrics import smape as darts_smape

from plotting import get_prediction_time_window
from plotting import plot_predictions_by_gap
from plotting import plot_predictions_by_method_grid


DEFAULT_WORKERS = {
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
}


def suppress_torch_lightning_runtime_warnings() -> None:
    """Reduce noisy torch/lightning warnings and info logs in notebooks."""
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"torch\.random",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"lightning.*",
    )
    warnings.filterwarnings(
        "ignore",
        message="CUDA reports that you have .*fork_rng.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Trainer will use only 1 of .* interactive / notebook environment.*",
        category=UserWarning,
    )

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


def _ensure_datetime_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Valida que el indice sea temporal y devuelve el DataFrame ordenado."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name} debe tener DatetimeIndex")
    return df.sort_index()


def timeseries_to_pd_series(ts: TimeSeries) -> pd.Series:
    """Convierte un `TimeSeries` de Darts a `pd.Series`."""
    return ts.to_series()


def ts_to_pd(ts: TimeSeries) -> pd.Series:
    """Alias retrocompatible de `timeseries_to_pd_series`."""
    return timeseries_to_pd_series(ts)


def mean_ignore_invalid(values: list[float | None]) -> float:
    """Calcula la media ignorando `None`, `NaN` e infinitos."""
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def safe_mean(values: list[float | None]) -> float:
    """Alias retrocompatible de `mean_ignore_invalid`."""
    return mean_ignore_invalid(values)


def _normalize_metrics(
    metrics: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Normaliza, valida y deduplica el listado de metricas solicitadas."""
    allowed = {"rmse", "mae", "smape", "mase"}
    if metrics is None:
        return ["mase"]

    if len(metrics) == 0:
        raise ValueError("`metrics` no puede estar vacio")

    normalized: list[str] = []
    for metric in metrics:
        m = str(metric).strip().lower()
        if m not in allowed:
            raise ValueError(
                f"Metrica no soportada: '{metric}'. Permitidas: {sorted(allowed)}"
            )
        if m not in normalized:
            normalized.append(m)

    return normalized


def build_mase_naive_prediction(
    ref_col_series: pd.Series,
    target_index: pd.Index,
    seasonality_m: int,
) -> pd.Series:
    """Construye baseline naive estacional `y(t-m)` en el indice objetivo."""
    return ref_col_series.shift(seasonality_m).reindex(target_index)


def stitch_blocks(blocks: list[TimeSeries]) -> pd.Series:
    """Concatena bloques de `TimeSeries` en una sola serie sin duplicados."""
    parts = [timeseries_to_pd_series(b) for b in blocks if b is not None and len(b) > 0]
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def build_internal_split(
    df_full: pd.DataFrame,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa un bloque de test por rango temporal y deja el resto para train."""
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    mask = (df_full.index >= test_start) & (df_full.index <= test_end)
    df_test_block = df_full.loc[mask].copy()
    df_train_internal = df_full.loc[~mask].copy()

    if df_test_block.empty:
        raise ValueError("Bloque de test vacio.")
    if df_train_internal.empty:
        raise ValueError("Train interno vacio.")

    return df_train_internal, df_test_block


def build_mase_reference(
    df_full: pd.DataFrame,
    freq: str = "h",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regulariza e interpola la serie para usarla como referencia de MASE."""
    df_reg = df_full.sort_index().asfreq(freq)
    df_ref = df_reg.interpolate(method="time", limit_direction="both").ffill().bfill()
    return df_reg, df_ref


def build_insample_for_prediction(
    pred_block: TimeSeries,
    ref_col_series: pd.Series,
    m: int = 24,
    freq: str = "h",
) -> TimeSeries | None:
    """Construye insample regularizado para calcular MASE de un bloque predicho."""
    if ref_col_series.empty:
        return None

    pred_start = pd.Timestamp(pred_block.start_time())
    base_offset = to_offset(freq)
    cutoff = pred_start - base_offset

    ref_series = ref_col_series.sort_index()
    ref_series = ref_series[~ref_series.index.duplicated(keep="last")]

    if ref_series.index.min() >= pred_start:
        return None

    ins_pd = ref_series.loc[:cutoff]
    if len(ins_pd) == 0:
        return None

    try:
        base_delta = pd.Timedelta(base_offset)
    except Exception:
        return None

    if base_delta <= pd.Timedelta(0):
        return None

    span = cutoff - ins_pd.index.min()
    if span < pd.Timedelta(0):
        return None

    n_steps = int(span // base_delta) + 1
    if n_steps <= m:
        return None

    ins_index = pd.date_range(end=cutoff, periods=n_steps, freq=base_offset)
    ins_pd = ins_pd.reindex(ins_index)
    ins_pd = ins_pd.interpolate(method="time", limit_direction="both").ffill().bfill()
    if ins_pd.isna().any():
        return None

    ins_ts = TimeSeries.from_series(ins_pd)
    if not (ins_ts.start_time() < pred_block.start_time()):
        return None
    if not (ins_ts.end_time() >= cutoff):
        return None

    return ins_ts


def get_insample_for_pred(
    pred_block: TimeSeries,
    ref_col_series: pd.Series,
    m: int = 24,
    freq: str = "h",
) -> TimeSeries | None:
    """Alias retrocompatible de `build_insample_for_prediction`."""
    return build_insample_for_prediction(
        pred_block=pred_block,
        ref_col_series=ref_col_series,
        m=m,
        freq=freq,
    )


def safe_mase(
    actual_ts: TimeSeries,
    pred_ts: TimeSeries,
    insample_ts: TimeSeries | None,
    m: int = 24,
) -> float:
    """Calcula MASE devolviendo `NaN` cuando no se puede evaluar."""
    if insample_ts is None:
        return float("nan")
    try:
        return float(
            mase(
                actual_series=actual_ts,
                pred_series=pred_ts,
                insample=insample_ts,
                m=m,
                intersect=True,
            )
        )
    except Exception:
        return float("nan")


def compute_aligned_mase_from_reference(
    ts_test: TimeSeries,
    ts_pred: TimeSeries,
    ref_col_series: pd.Series,
    *,
    freq: str = "h",
    seasonality_m: int = 24,
) -> float:
    """Calcula MASE alineando indices de test/pred y reconstruyendo insample."""
    actual_pd = timeseries_to_pd_series(ts_test).sort_index()
    pred_pd = timeseries_to_pd_series(ts_pred).sort_index()
    common_idx = actual_pd.index.intersection(pred_pd.index)
    if len(common_idx) == 0:
        return float("nan")

    actual_i_pd = actual_pd.reindex(common_idx)
    pred_i_pd = pred_pd.reindex(common_idx)
    valid_mask = np.isfinite(actual_i_pd.to_numpy()) & np.isfinite(pred_i_pd.to_numpy())
    if not np.any(valid_mask):
        return float("nan")

    actual_i_pd = actual_i_pd.iloc[valid_mask]
    pred_i_pd = pred_i_pd.iloc[valid_mask]
    if len(actual_i_pd) == 0 or len(pred_i_pd) == 0:
        return float("nan")

    try:
        actual_i = TimeSeries.from_series(actual_i_pd)
        pred_i = TimeSeries.from_series(pred_i_pd)
    except Exception:
        return float("nan")

    insample_dyn = build_insample_for_prediction(
        pred_block=pred_i,
        ref_col_series=ref_col_series,
        m=seasonality_m,
        freq=freq,
    )
    return safe_mase(actual_i, pred_i, insample_dyn, m=seasonality_m)


def _align_actual_pred_timeseries(
    ts_test: TimeSeries,
    ts_pred: TimeSeries,
) -> tuple[TimeSeries | None, TimeSeries | None]:
    """Alinea test/pred por timestamps y elimina pares no finitos."""
    actual_pd = timeseries_to_pd_series(ts_test).sort_index()
    pred_pd = timeseries_to_pd_series(ts_pred).sort_index()
    common_idx = actual_pd.index.intersection(pred_pd.index)
    if len(common_idx) == 0:
        return None, None

    actual_i_pd = actual_pd.reindex(common_idx)
    pred_i_pd = pred_pd.reindex(common_idx)
    valid_mask = np.isfinite(actual_i_pd.to_numpy()) & np.isfinite(pred_i_pd.to_numpy())
    if not np.any(valid_mask):
        return None, None

    actual_i_pd = actual_i_pd.iloc[valid_mask]
    pred_i_pd = pred_i_pd.iloc[valid_mask]
    if len(actual_i_pd) == 0 or len(pred_i_pd) == 0:
        return None, None

    try:
        return TimeSeries.from_series(actual_i_pd), TimeSeries.from_series(pred_i_pd)
    except Exception:
        return None, None


def _is_prediction_within_test(ts_test: TimeSeries, ts_pred: TimeSeries) -> bool:
    """Verifica que todas las fechas predichas pertenezcan al bloque de test."""
    test_index = timeseries_to_pd_series(ts_test).index
    pred_index = timeseries_to_pd_series(ts_pred).index
    return pred_index.isin(test_index).all()


def _build_reference_from_full_series(
    full_series: pd.Series,
    freq: str,
) -> pd.Series:
    """Genera una referencia continua y sin huecos desde una serie completa."""
    ref_col = full_series.sort_index()
    ref_col = ref_col[~ref_col.index.duplicated(keep="last")]
    return (
        ref_col.asfreq(freq)
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )


def _build_full_series_map(
    all_series: list[pd.DataFrame | pd.Series],
) -> dict[str, pd.Series]:
    """Indexa series completas por nombre de columna con validaciones de formato."""
    by_col: dict[str, pd.Series] = {}

    for idx, series_like in enumerate(all_series):
        if isinstance(series_like, pd.DataFrame):
            if len(series_like.columns) != 1:
                raise ValueError(
                    "Cada DataFrame en `all_series` debe tener exactamente una columna."
                )
            series = series_like.iloc[:, 0].copy()
        elif isinstance(series_like, pd.Series):
            series = series_like.copy()
        else:
            raise TypeError(
                "Cada elemento en `all_series` debe ser pd.DataFrame o pd.Series; "
                f"recibido: {type(series_like)} en posicion {idx}."
            )

        col = str(series.name) if series.name is not None else ""
        if not col:
            raise ValueError(
                "Cada serie en `all_series` debe tener nombre en `Series.name`."
            )
        if col in by_col:
            raise ValueError(f"Nombre de serie duplicado en `all_series`: {col}")
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError(
                f"La serie '{col}' en `all_series` debe tener DatetimeIndex"
            )

        by_col[col] = series.sort_index()

    return by_col


PredictionFn = Callable[
    [str, Any, list[TimeSeries], int, dict[str, Any] | None],
    list[TimeSeries | pd.Series],
]


def default_darts_prediction_fn(
    model_name: str,
    model: Any,
    input_series: list[TimeSeries],
    forecast_horizon: int,
    config_workers: dict[str, Any] | None,
) -> list[TimeSeries | pd.Series]:
    """Genera predicciones usando `historical_forecasts` de Darts."""
    if not hasattr(model, "historical_forecasts"):
        raise TypeError(
            f"El modelo '{model_name}' no tiene historical_forecasts; "
            "pasa un prediction_fn personalizado."
        )

    predict_kwargs: dict[str, Any] = {}
    if config_workers:
        predict_kwargs["dataloader_kwargs"] = config_workers

    pred = model.historical_forecasts(
        series=input_series,
        forecast_horizon=forecast_horizon,
        stride=1,
        retrain=False,
        last_points_only=True,
        verbose=False,
        predict_kwargs=predict_kwargs,
    )

    if isinstance(pred, TimeSeries):
        return [pred]
    return list(pred)


def _coerce_predictions_to_timeseries(
    pred_raw: list[TimeSeries | pd.Series],
    col_names: list[str],
    freq: str,
) -> list[TimeSeries]:
    """Normaliza predicciones heterogeneas a lista de `TimeSeries`."""
    if len(pred_raw) != len(col_names):
        raise ValueError(
            "Cantidad de predicciones no coincide con el numero de series de entrada"
        )

    out: list[TimeSeries] = []
    for p in pred_raw:
        if isinstance(p, TimeSeries):
            out.append(p)
        elif isinstance(p, pd.Series):
            out.append(TimeSeries.from_series(p.sort_index(), freq=freq))
        else:
            raise TypeError(
                f"Cada prediccion debe ser TimeSeries o pd.Series; recibido: {type(p)}"
            )
    return out


def _build_plot_payload(
    model_name: str,
    col: str,
    ts_test: TimeSeries,
    ts_pred: TimeSeries,
    *,
    ref_col: pd.Series | None = None,
    mase_value: float | None = None,
    seasonality_m: int = 24,
) -> dict[str, Any]:
    """Arma el payload de plotting para un modelo/serie evaluados."""
    ts_pred_pd = timeseries_to_pd_series(ts_pred)
    payload: dict[str, Any] = {
        "actual": timeseries_to_pd_series(ts_test),
        "prediction": ts_pred_pd,
        "model_name": model_name,
        "series_name": col,
    }

    if mase_value is not None:
        payload["mase"] = mase_value
    if ref_col is not None:
        payload["naive_mase"] = build_mase_naive_prediction(
            ref_col_series=ref_col,
            target_index=ts_pred_pd.index,
            seasonality_m=seasonality_m,
        )

    return payload


def execute_complete_pipeline(
    model_dict: dict[str, Any],
    dataset_bundle: dict[str, Any],
    all_series: list[pd.DataFrame | pd.Series] | None = None,
    *,
    forecast_horizon: int = 1,
    config_workers: dict[str, Any] | None = None,
    freq: str = "h",
    seasonality_m: int = 24,
    metrics: list[str] | tuple[str, ...] | None = None,
    prediction_fn: PredictionFn | None = None,
    predict_on_scaled_series: bool = True,
    predictions_are_scaled: bool = True,
) -> tuple[pd.DataFrame, dict[str, Scaler], dict[str, dict[str, dict[str, Any]]]]:
    """
    Evalua modelos y calcula metricas por serie.

    - Usa `dataset_bundle` para prediccion/escalado del bloque de test.
    - Usa `all_series` solo si se solicita `mase` en `metrics`.
    - La generacion de predicciones se delega en `prediction_fn`.
    - El plotting queda fuera de esta funcion.
    """
    suppress_torch_lightning_runtime_warnings()

    if config_workers is None:
        config_workers = DEFAULT_WORKERS

    if not model_dict:
        raise ValueError("model_dict no puede estar vacio")

    if prediction_fn is None:
        prediction_fn = default_darts_prediction_fn

    metric_list = _normalize_metrics(metrics)
    metric_columns = [m.upper() for m in metric_list]
    need_mase = "mase" in metric_list

    valid_cols = list(dataset_bundle["valid_cols"])
    test_series_scaled = list(dataset_bundle["series_test"])
    bundle_scalers = dataset_bundle["dict_scalers"]

    if not valid_cols:
        raise ValueError("`dataset_bundle['valid_cols']` esta vacio")
    if len(valid_cols) != len(test_series_scaled):
        raise ValueError(
            "`dataset_bundle['valid_cols']` y `dataset_bundle['series_test']` "
            "no tienen la misma longitud."
        )

    missing_scalers = sorted([col for col in valid_cols if col not in bundle_scalers])
    if missing_scalers:
        raise ValueError(
            "Faltan escaladores en `dataset_bundle['dict_scalers']` para: "
            f"{missing_scalers}"
        )

    all_series_by_col: dict[str, pd.Series] = {}
    if need_mase:
        if all_series is None:
            raise ValueError(
                "Para calcular MASE debes pasar `all_series` con las series completas."
            )
        all_series_by_col = _build_full_series_map(all_series)
        missing_full_series = sorted(set(valid_cols).difference(set(all_series_by_col)))
        if missing_full_series:
            raise ValueError(
                "Faltan series completas en `all_series` para columnas de test: "
                f"{missing_full_series}"
            )

    results: list[dict[str, Any]] = []
    trained_scalers: dict[str, Scaler] = {
        col: bundle_scalers[col] for col in valid_cols
    }
    test_series_unscaled: list[TimeSeries] = []
    plot_store: dict[str, dict[str, dict[str, Any]]] = {}

    print(">>> Preparando series de test desde dataset_bundle...")
    for i, col in enumerate(valid_cols):
        ts_scaled = test_series_scaled[i]
        if not isinstance(ts_scaled, TimeSeries):
            raise TypeError(
                "Cada elemento de `dataset_bundle['series_test']` debe ser "
                f"TimeSeries; recibido: {type(ts_scaled)} en '{col}'."
            )

        sc = trained_scalers[col]
        test_series_unscaled.append(sc.inverse_transform(ts_scaled))

    input_series = (
        test_series_scaled if predict_on_scaled_series else test_series_unscaled
    )

    for model_name, model in model_dict.items():
        print(f"\nEvaluando modelo: {model_name}")

        pred_raw = prediction_fn(
            model_name,
            model,
            input_series,
            forecast_horizon,
            config_workers,
        )
        ts_preds = _coerce_predictions_to_timeseries(pred_raw, valid_cols, freq)

        plot_store[model_name] = {}
        per_model_results: list[dict[str, Any]] = []
        for i, col in enumerate(valid_cols):
            sc = trained_scalers[col]
            ts_test = test_series_unscaled[i]
            ts_pred = ts_preds[i]

            if predictions_are_scaled:
                ts_pred = sc.inverse_transform(ts_pred)

            if not _is_prediction_within_test(ts_test, ts_pred):
                raise ValueError(
                    "Se detectaron timestamps de prediccion fuera del bloque de test "
                    f"en la serie '{col}' del modelo '{model_name}'."
                )

            row: dict[str, Any] = {"Modelo": model_name, "Serie": col}

            actual_i, pred_i = _align_actual_pred_timeseries(ts_test, ts_pred)

            if "rmse" in metric_list:
                if actual_i is None or pred_i is None:
                    row["RMSE"] = float("nan")
                else:
                    row["RMSE"] = float(darts_rmse(actual_i, pred_i, intersect=True))

            if "mae" in metric_list:
                if actual_i is None or pred_i is None:
                    row["MAE"] = float("nan")
                else:
                    row["MAE"] = float(darts_mae(actual_i, pred_i, intersect=True))

            if "smape" in metric_list:
                if actual_i is None or pred_i is None:
                    row["SMAPE"] = float("nan")
                else:
                    row["SMAPE"] = float(darts_smape(actual_i, pred_i, intersect=True))

            ref_col: pd.Series | None = None
            mase_value: float | None = None
            if need_mase:
                ref_col = _build_reference_from_full_series(
                    all_series_by_col[col], freq
                )
                mase_value = compute_aligned_mase_from_reference(
                    ts_test=ts_test,
                    ts_pred=ts_pred,
                    ref_col_series=ref_col,
                    freq=freq,
                    seasonality_m=seasonality_m,
                )
                row["MASE"] = mase_value

            results.append(row)
            per_model_results.append(row)
            plot_store[model_name][col] = _build_plot_payload(
                model_name=model_name,
                col=col,
                ts_test=ts_test,
                ts_pred=ts_pred,
                ref_col=ref_col,
                mase_value=mase_value,
                seasonality_m=seasonality_m,
            )

        rank_col = metric_columns[0]
        per_model_results = sorted(
            per_model_results,
            key=lambda x: x[rank_col] if np.isfinite(x[rank_col]) else -np.inf,
            reverse=True,
        )

        print(f"Top series con mayor {rank_col}:")
        for r in per_model_results[:5]:
            value = r[rank_col]
            value_txt = f"{value:.4f}" if np.isfinite(value) else "nan"
            print(f"  - {r['Serie']}: {value_txt}")

    results_df = pd.DataFrame(results)
    return results_df[["Modelo", "Serie", *metric_columns]], trained_scalers, plot_store

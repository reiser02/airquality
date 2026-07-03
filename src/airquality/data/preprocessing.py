"""Preprocesamiento de series de aire: 5 min -> media horaria -> filtrado.

``datos_estaciones`` se genera con un AVG en SQL: media SIMPLE de todas las
lecturas de la hora (``date_trunc('hour', ...) + avg(...)`` con el contaminante
``>= 0 AND < 100000``), SIN minimo de lecturas. Luego se filtra por umbral y
congelados horarios.

Aqui se reconstruye desde los datos de 5 min priorizando la CALIDAD de la
media, no minimizar NaN. Frente al AVG del SQL se anaden tres reglas:

- la media usa SOLO lecturas por encima del umbral (no la basura cercana a 0),
- exige al menos ``MIN_USEFUL`` lecturas utiles en la hora (cobertura minima),
- los tramos congelados (sensor atascado) se ponen a 0 antes de promediar.

Como el SQL no hace nada de esto (media simple, sin minimo de lecturas, sin
tratamiento de congelados sub-horarios), estas reglas pueden dejar MAS NaN que
``datos_estaciones`` en horas pobres o con sensor atascado: es intencionado, se
descartan medias poco fiables. Tras la media, :func:`preprocess` solo elimina
los congelados horarios (``col == col.shift()``); el umbral no hace falta porque
la media inteligente nunca queda por debajo.
"""

from __future__ import annotations

import pandas as pd

from airquality.data.series import ensure_datetime_series

# Limites de deteccion por contaminante (mismas unidades que los datos crudos).
DETECTION_LIMITS = {
    "CO": 6.0,
    "NO2": 3.762,
}

RAW_FREQ = "5min"
HOURLY_FREQ = "h"
MIN_RUN = 6      # >= 6 lecturas de 5 min identicas (>=30 min) = sensor congelado
MIN_USEFUL = 3   # lecturas utiles minimas en la hora para calcular la media


def frozen_mask(series: pd.Series, min_run: int = MIN_RUN) -> pd.Series:
    """Marca (True) tramos de valores identicos consecutivos de longitud >= min_run.

    Los NaN no rompen ni alargan un tramo (se ignoran al comparar). Con datos de
    5 min, ``min_run=6`` equivale a 30 minutos de sensor atascado.
    """
    mask = pd.Series(False, index=series.index)
    valid = series.dropna()
    if valid.empty:
        return mask
    block = (valid != valid.shift()).cumsum()
    run_len = block.map(block.value_counts())
    mask.loc[valid.index[run_len >= min_run]] = True
    return mask


def hourly_mean(
    df: pd.DataFrame,
    pollutant: str,
    *,
    min_run: int = MIN_RUN,
    min_useful: int = MIN_USEFUL,
) -> pd.DataFrame:
    """Media horaria a partir de datos de 5 minutos.

    Se fija la frecuencia a 5 min (12 ranuras por hora; los huecos pasan a NaN).
    Los valores congelados se ponen a 0 para que caigan por debajo del umbral.
    Una lectura es "util" si supera el umbral. Si al menos ``min_useful`` lecturas
    de la hora son utiles, la media se calcula con ESAS; si no, la hora es NaN.
    """
    threshold = DETECTION_LIMITS[pollutant]
    col = df.columns[0]
    series = ensure_datetime_series(df[col], freq=RAW_FREQ, name=col)

    # Congelados -> 0: asi cuentan como por debajo del umbral.
    series = series.mask(frozen_mask(series, min_run=min_run), 0)

    hour = series.index.floor(HOURLY_FREQ)
    is_useful = series >= threshold

    n_useful = is_useful.groupby(hour).sum()
    mean_useful = series.where(is_useful).groupby(hour).mean()

    value = mean_useful.where(n_useful >= min_useful)
    value.index = pd.DatetimeIndex(value.index)
    return value.asfreq(HOURLY_FREQ).rename(col).to_frame()


def preprocess(
    dfs: list[pd.DataFrame],
    pollutant: str,
    *,
    min_run: int = MIN_RUN,
    min_useful: int = MIN_USEFUL,
):
    """Pipeline completo: 5 min -> media horaria -> eliminacion de congelados.

    Para cada estacion calcula la media horaria (:func:`hourly_mean`, que ya
    garantiza valor >= umbral o NaN) y elimina los congelados a nivel horario:
    toda fila igual a la anterior (repeticion consecutiva) se descarta y se
    conserva la primera de cada bloque. No hace falta filtrar por umbral (la
    media nunca queda por debajo) ni los NaN (no estorban al comparar ni aguas
    abajo, donde se vuelve a re-rejillar).

    Devuelve las series horarias limpias y el numero de filas congeladas
    eliminadas por estacion.
    """
    processed = []
    frozen_counts = []

    for df in dfs:
        hourly = hourly_mean(df, pollutant, min_run=min_run, min_useful=min_useful)
        col = hourly[hourly.columns[0]]

        # Congelado horario: igual a la fila anterior. Los NaN no se marcan
        # (NaN != NaN), asi que solo caen las repeticiones reales de valor.
        is_frozen = col == col.shift()
        frozen_counts.append(int(is_frozen.sum()))
        processed.append(hourly[~is_frozen])

    return processed, frozen_counts

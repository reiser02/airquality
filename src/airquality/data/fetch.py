"""Download pollutant measurements from the Cartagena data API."""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


DEFAULT_POLLUTANTS = ["CO", "NO2", "PM10", "O3"]
DEFAULT_START_DATE = datetime(2024, 1, 1)
POLLUTANT_ALIASES = {"PM2.5": "PM25", "PM2_5": "PM25"}


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _normalize_pollutants(values: list[str]) -> list[str]:
    pollutants: list[str] = []
    for value in values:
        for item in value.split(","):
            pollutant = POLLUTANT_ALIASES.get(item.strip().upper(), item.strip().upper())
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", pollutant):
                raise ValueError(f"Contaminante no valido para SQL: {item!r}")
            pollutants.append(pollutant)
    return pollutants


def _build_stmt(cont: str, query: str) -> str:
    if query == "hourly":
        return f"""
        SELECT date_trunc('hour', time_index) AS time,
        avg({cont}) AS "{cont}"
        FROM mtairquality.etairqualityobserved
        WHERE time_index >= floor(try_cast(? AS BIGINT) / 3600000) * 3600000
        AND time_index <= floor(try_cast(? AS BIGINT) / 3600000) * 3600000
        AND entity_id = ?
        AND {cont} >= 0 AND {cont} < 100000
        GROUP BY time
        ORDER BY time ASC
        """
    return f"""
    SELECT time_index AS time,
    {cont} AS "{cont}"
    FROM mtairquality.etairqualityobserved
    WHERE time_index >= ? AND time_index <= ?
    AND entity_id = ?
    AND {cont} >= 0 AND {cont} < 100000
    ORDER BY 1 ASC
    """


def ejecutar_scraper(
    contaminantes: list[str] | None = None,
    fecha_inicio: datetime | None = None,
    query: str = "5min",
) -> None:
    """Fetch all configured station pollutants and save them as CSV files."""
    estaciones = {
        "urn:ngsi:AirQualityObserved:HOP94e6867be9fa": "Aquatec - Calle Jorge Juan",
        "urn:ngsi:AirQualityObserved:HOPac67b2cd1cd6": "AQN4 - Alameda San Anton",
        "urn:ngsi:AirQualityObserved:HOP94e6867c1532": "Aquatec - Angel Bruna",
        # "urn:ngsi:AirQualityObserved:HOPac67b2cd222e": "AQN5 - Extremadura",
        "urn:ngsi:AirQualityObserved:HOP94e6867aa682": "Aquatec - Pintor Balaca",
        "urn:ngsi:AirQualityObserved:HOP94e6867c064e": "Aquatec - Parque Sauces",
        "urn:ngsi:AirQualityObserved:HOP94e6867c0966": "Aquatec - Juan Fernandez",
        "urn:ngsi:AirQualityObserved:HOP94e6867bf326": "Aquatec - Ramon y Cajal",
        # "urn:ngsi:AirQualityObserved:HOPac67b2d5cb4a": "AQN6 - Jimenez de la Espada 43",
        "urn:ngsi:AirQualityObserved:HOP94e6867c249e": "Aquatec - Jimenez de la Espada",
        "urn:ngsi:AirQualityObserved:HOPa842e389d9b6": "AQN3 - Paseo Alfonso XIII",
        "urn:ngsi:AirQualityObserved:HOP94e68679c5b2": "Aquatec - Juan de la Cosa",
        "urn:ngsi:AirQualityObserved:HOP94e6867c01f2": "Aquatec - San Juan",
        "urn:ngsi:AirQualityObserved:HOP94e6867c0222": "Aquatec - Capitanes Ripoll",
        "urn:ngsi:AirQualityObserved:HOP94e6867c2186": "Aquatec - Carlos III",
        # "urn:ngsi:AirQualityObserved:HOP94e6867c087e": "Aquatec - Plaza Juan XXIII",
        "urn:ngsi:AirQualityObserved:HOP94e6867c200a": "Aquatec - Plaza Lopez Pinto",
        "urn:ngsi:AirQualityObserved:HOP94e6867c2386": "Aquatec - Salitre",
        "urn:ngsi:AirQualityObserved:HOP94e6867c0a5a": "Aquatec - Calle Real",
        "urn:ngsi:AirQualityObserved:HOPe0e2e632fa1e": "Aquatec - Serreta",
        "urn:ngsi:AirQualityObserved:HOP94e6867c28e2": "Aquatec - San Diego",
        # "urn:ngsi:AirQualityObserved:HOP94e6867c1842": "Aquatec - Plaza Castellini",
        "urn:ngsi:AirQualityObserved:HOPac67b2cd1c9e": "AQN2 - San Francisco",
        "urn:ngsi:AirQualityObserved:HOPac67b2cd1d56": "AQN1 - Puerto",
        "urn:ngsi:AirQualityObserved:HOP94e6867c2712": "Aquatec - Paseo Alfonso XII",
        "urn:ngsi:AirQualityObserved:HOPa842e389da1e": "AQ1 BusCT",
    }

    contaminantes = contaminantes or DEFAULT_POLLUTANTS
    fecha_inicio = fecha_inicio or DEFAULT_START_DATE
    fecha_fin = datetime.now()

    url = "https://pma.ayto-cartagena.es/visualizador/api/datasources/proxy/1/_sql"
    base_dir = Path("datos_estaciones")

    print(f"Iniciando captura masiva {query} desde {fecha_inicio.date()}")


    session = requests.Session()
    session.verify = False

    for ent_id, nombre_corto in estaciones.items():
        ruta_sensor = base_dir / nombre_corto
        ruta_sensor.mkdir(parents=True, exist_ok=True)

        print(f"\n Procesando: {nombre_corto}")

        for cont in contaminantes:
            payload = {
                "stmt": _build_stmt(cont, query),
                "args": [
                    int(fecha_inicio.timestamp() * 1000),
                    int(fecha_fin.timestamp() * 1000),
                    ent_id,
                ],
            }

            try:
                response = session.post(url, json=payload, timeout=60)
                response.raise_for_status()
                time.sleep(1)
                data = response.json()

                if "rows" in data and data["rows"]:
                    df = pd.DataFrame(data["rows"], columns=["fecha", cont])
                    df["fecha"] = pd.to_datetime(df["fecha"], unit="ms")
                    df = df.sort_values("fecha")
                    freq = "h" if query == "hourly" else "5min"
                    df["fecha"] = df["fecha"].dt.floor(freq)
                    df = df.drop_duplicates(subset="fecha", keep="first")
                    df = df.set_index("fecha").asfreq(freq)

                    # Nombre del archivo: nombreSensor_CONTAMINANTE.csv
                    nombre_archivo = f"{nombre_corto}_{cont}.csv"
                    df.to_csv(ruta_sensor / nombre_archivo)
                    print(f"  {cont}: {len(df)} filas guardadas.")
                else:
                    print(f"  {cont}: No se encontraron datos.")

            except requests.exceptions.HTTPError as e:
                print(f"  Error HTTP en {cont}: {e}")
            except Exception as e:
                print(f"  Error en {cont}: {e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        choices=["5min", "hourly"],
        default="5min",
        help="Consulta cruda a 5 minutos o media horaria en SQL.",
    )
    parser.add_argument(
        "--pollutants",
        nargs="+",
        default=DEFAULT_POLLUTANTS,
        help="Contaminantes separados por espacios o comas. PM2.5 se normaliza a PM25.",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=DEFAULT_START_DATE,
        help="Fecha inicial ISO, por ejemplo 2024-01-01.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ejecutar_scraper(
        contaminantes=_normalize_pollutants(args.pollutants),
        fecha_inicio=args.start_date,
        query=args.query,
    )

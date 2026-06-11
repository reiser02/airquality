"""Download pollutant measurements from the Cartagena data API."""

import requests
import pandas as pd
import urllib3
import time
from datetime import datetime
from pathlib import Path

# Configuración de seguridad
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def ejecutar_scraper() -> None:
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

    contaminantes = ["CO", "NO2", "PM10", "O3"]
    fecha_inicio = datetime(2024, 1, 1)
    fecha_fin = datetime.now()

    url = "https://pma.ayto-cartagena.es/visualizador/api/datasources/proxy/1/_sql"
    base_dir = Path("datos_estaciones")

    print(f"Iniciando captura masiva desde {fecha_inicio.date()}")

    for ent_id, nombre_corto in estaciones.items():
        ruta_sensor = base_dir / nombre_corto
        ruta_sensor.mkdir(parents=True, exist_ok=True)

        print(f"\n Procesando: {nombre_corto}")

        for cont in contaminantes:
            # SQL Query
            stmt = f"""
            SELECT time_index AS time,
            {cont} AS "{cont}"
            FROM mtairquality.etairqualityobserved
            WHERE time_index >= ? AND time_index <= ?
            AND entity_id = ?
            AND {cont} >= 0 AND {cont} < 100000
            ORDER BY 1 ASC
            """

            payload = {
                "stmt": stmt,
                "args": [
                    int(fecha_inicio.timestamp() * 1000),
                    int(fecha_fin.timestamp() * 1000),
                    ent_id,
                ],
            }

            try:
                response = requests.post(url, json=payload, verify=False, timeout=60)
                response.raise_for_status()
                time.sleep(1)
                data = response.json()

                if "rows" in data and data["rows"]:
                    df = pd.DataFrame(data["rows"], columns=["fecha", cont])
                    df["fecha"] = pd.to_datetime(df["fecha"], unit="ms")
                    df = df.sort_values("fecha")
                    df["fecha"] = df["fecha"].dt.floor("5min")
                    df = df.drop_duplicates(subset="fecha", keep="first")
                    df = df.set_index("fecha").asfreq("5min")

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


if __name__ == "__main__":
    ejecutar_scraper()

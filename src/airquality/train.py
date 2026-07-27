"""Zero-argument terminal entrypoint for training the configured global models."""

from __future__ import annotations

from typing import Any, Sequence

from airquality.config import cfg_get_csv_list, cfg_get_int, cfg_get_str
from airquality.data.io import load_and_normalize_series
from airquality.data.segments import get_longest_segment
from airquality.modeling.training import (
    build_training_dataset_bundle,
    train_global_methods,
)
from airquality.modeling.training_config import build_model_configs


def _select_trainable_methods(method_names: Sequence[str]) -> list[str]:
    """Validate and deduplicate the Darts models selected for training."""
    requested = list(dict.fromkeys(name.strip() for name in method_names if name.strip()))
    available = build_model_configs()
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            "Modelos no entrenables en `[training] model_names`: "
            f"{', '.join(unknown)}. Disponibles: {', '.join(available)}"
        )
    if not requested:
        raise RuntimeError("No hay modelos Darts en `[training] model_names`.")
    return requested


def train_from_config() -> dict[str, Any]:
    """Load the configured dataset and train the configured forecasting models."""
    freq = cfg_get_str("data", "freq", "h")
    size_k = cfg_get_int("benchmark", "size_k", 5)
    val_size = cfg_get_int("benchmark", "val_size", 48)
    val_context_len = cfg_get_int("benchmark", "val_context_len", 72)
    min_train_len_base = cfg_get_int("benchmark", "min_train_len_base", 72)
    method_names = _select_trainable_methods(
        cfg_get_csv_list(
            "training",
            "model_names",
            (
                "TiDE",
                "NHiTS",
                "NLinear",
                "DLinear",
                "TCN",
                "TSMixer",
                "RNN",
                "LinearRegression",
            ),
        )
    )

    print("[info] Loading series selected by config")
    series_dfs = load_and_normalize_series(freq=freq)
    if not series_dfs:
        raise RuntimeError("No se pudieron construir series validas desde los archivos cargados.")

    print("[info] Selecting held-out segment")
    longest_segment = get_longest_segment(series_dfs, verbose=False)
    if longest_segment.empty:
        raise RuntimeError("get_longest_segment devolvio un DataFrame vacio.")

    print("[info] Building training dataset bundle")
    dataset_bundle = build_training_dataset_bundle(
        series_dfs=series_dfs,
        longest_segment=longest_segment,
        val_size=val_size,
        min_train_len=min_train_len_base + size_k,
        val_context_len=val_context_len,
    )

    print(f"[info] Training models: {', '.join(method_names)}")
    return train_global_methods(
        dataset_bundle=dataset_bundle,
        size_k=size_k,
        method_names=method_names,
    )


def main() -> None:
    """Run training using only values from the shared project config."""
    train_from_config()


if __name__ == "__main__":
    main()

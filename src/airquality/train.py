"""Zero-argument terminal entrypoint for training the configured global models."""

from __future__ import annotations

from typing import Any, Sequence

from airquality.config import cfg_get_csv_list, cfg_get_int, cfg_get_str
from airquality.data.io import load_and_normalize_series
from airquality.data.segments import get_longest_segment
from airquality.imputation.registry import DARTS_GLOBAL, resolve_imputer_family
from airquality.modeling.training import (
    build_training_dataset_bundle,
    train_global_methods,
)


def _select_trainable_methods(method_names: Sequence[str]) -> list[str]:
    """Keep only Darts-global forecasters; other benchmark imputers aren't trained.

    ``[benchmark] model_names`` is shared with the imputation benchmark and may
    include names like ``Prophet``, ``TSPulse`` or ``LinearInterp`` that are not
    trainable Darts artifacts. Those would make ``train_global_methods`` raise, so
    they are filtered out here (and reported) before training.
    """
    trainable: list[str] = []
    skipped: list[str] = []
    for name in method_names:
        if resolve_imputer_family(name) == DARTS_GLOBAL:
            trainable.append(name)
        else:
            skipped.append(name)

    if skipped:
        print(f"[info] Skipping non-trainable benchmark imputers: {', '.join(skipped)}")
    if not trainable:
        raise RuntimeError(
            "No hay modelos Darts entrenables en `[benchmark] model_names`."
        )
    return trainable


def train_from_config() -> dict[str, Any]:
    """Load the configured dataset and train the configured forecasting models."""
    freq = cfg_get_str("data", "freq", "h")
    size_k = cfg_get_int("benchmark", "size_k", 5)
    val_size = cfg_get_int("benchmark", "val_size", 48)
    val_context_len = cfg_get_int("benchmark", "val_context_len", 72)
    min_train_len_base = cfg_get_int("benchmark", "min_train_len_base", 72)
    method_names = _select_trainable_methods(
        cfg_get_csv_list(
            "benchmark",
            "model_names",
            ("TiDE", "NHiTS", "TCN", "TSMixer", "RNN", "NLinear", "DLinear"),
        )
    )

    print("[info] Loading series selected by config")
    series_dfs = load_and_normalize_series(freq=freq, name_from_path=True)
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

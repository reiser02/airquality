"""Zero-argument terminal entrypoint for training the configured global models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from airquality.config import cfg_get_csv_list, cfg_get_int, cfg_get_str
from airquality.data.io import load_and_normalize_series
from airquality.data.holdout import (
    build_holdout_manifest,
    select_retrospective_holdouts_with_exclusions,
    write_holdout_manifest,
)
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
    holdout_target_points = cfg_get_int("benchmark", "holdout_target_points", 192)
    holdout_context_points = cfg_get_int("benchmark", "holdout_context_points", 72)
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

    print("[info] Selecting retrospective holdout for each station")
    min_train_len = min_train_len_base + size_k
    min_train_points = max(min_train_len, val_context_len) + val_size
    holdouts, holdout_metadata, excluded_series = select_retrospective_holdouts_with_exclusions(
        series_dfs,
        target_points=holdout_target_points,
        context_points=holdout_context_points,
        min_train_points=min_train_points,
    )
    manifest = build_holdout_manifest(
        holdout_metadata,
        target_points=holdout_target_points,
        context_points=holdout_context_points,
        min_train_points=min_train_points,
        freq=freq,
        darts_models=method_names,
    )
    eligible_names = set(holdouts)
    eligible_series_dfs = [
        frame for frame in series_dfs if str(frame.columns[0]) in eligible_names
    ]
    if not excluded_series.empty:
        print(
            "[info] Excluding ineligible series from imputation training: "
            + ", ".join(excluded_series["Serie"].astype(str))
        )

    print("[info] Building training dataset bundle")
    dataset_bundle = build_training_dataset_bundle(
        series_dfs=eligible_series_dfs,
        holdouts_by_series=holdouts,
        val_size=val_size,
        min_train_len=min_train_len,
        val_context_len=val_context_len,
    )

    print(f"[info] Training models: {', '.join(method_names)}")
    trained = train_global_methods(
        dataset_bundle=dataset_bundle,
        size_k=size_k,
        method_names=method_names,
    )
    models_dir = Path(__file__).resolve().parents[2] / "models"
    write_holdout_manifest(manifest, models_dir / "imputation_holdout_manifest.json")
    holdout_metadata.to_csv(models_dir / "imputation_holdouts.csv", index=False)
    excluded_series.to_csv(models_dir / "imputation_excluded_series.csv", index=False)
    return trained


def main() -> None:
    """Run training using only values from the shared project config."""
    train_from_config()


if __name__ == "__main__":
    main()

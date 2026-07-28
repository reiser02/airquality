"""Checks for forecasting-only local and foundation model registration."""

import pytest
from darts import TimeSeries

from airquality.forecasting.backtest import (
    _fit_forecast_model,
    get_forecast_model_requirements,
)
from airquality.forecasting.registry import (
    ForecastModelConfig,
    resolve_forecasting_model_configs,
)
from airquality.imputation.registry import IMPUTER_FAMILIES
from airquality.modeling.training_config import build_model_configs


def test_forecast_only_models_do_not_enter_training_or_imputation_catalogs() -> None:
    names = ["AutoARIMA", "AutoETS", "Chronos2", "TimesFM2p5", "PatchTSTFM"]
    configs = resolve_forecasting_model_configs(names)

    assert [configs[name].mode for name in names] == [
        "local",
        "local",
        "foundation",
        "foundation",
        "foundation",
    ]
    assert set(names).isdisjoint(build_model_configs())
    assert set(names).isdisjoint(IMPUTER_FAMILIES)
    with pytest.raises(ValueError, match="AutoTBATS"):
        resolve_forecasting_model_configs(["AutoTBATS"])
    with pytest.raises(ValueError, match="TiRex"):
        resolve_forecasting_model_configs(["TiRex"])


def test_local_and_foundation_models_skip_validation_geometry() -> None:
    local = ForecastModelConfig(object, {}, "local")
    foundation = ForecastModelConfig(object, {}, "foundation")

    local_req = get_forecast_model_requirements(
        local, size_k=8, seasonality_m=24, context_len=72
    )
    foundation_req = get_forecast_model_requirements(
        foundation, size_k=48, seasonality_m=24, context_len=72
    )

    assert local_req.min_train_series_length == 48
    assert local_req.validation_target_offset is None
    assert foundation_req.min_train_series_length == 120
    assert foundation_req.prediction_context_length == 72
    assert foundation_req.validation_target_offset is None


def test_local_model_fits_only_latest_series() -> None:
    calls: dict[str, object] = {}

    class LocalModel:
        def __init__(self, **kwargs) -> None:
            calls["init"] = kwargs

        def fit(self, series, verbose=False) -> None:
            calls["fit"] = (series, verbose)

    older = TimeSeries.from_values([1.0, 2.0])
    latest = TimeSeries.from_values([3.0, 4.0])
    model = _fit_forecast_model(
        ForecastModelConfig(LocalModel, {"season_length": 24}, "local"),
        [older, latest],
        [],
        size_k=8,
    )

    assert isinstance(model, LocalModel)
    assert calls["init"] == {"season_length": 24}
    assert calls["fit"] == (latest, False)


def test_forecasting_registry_overrides_worker_gpu() -> None:
    configs = resolve_forecasting_model_configs(
        ["TiDE", "Chronos2"],
        accelerator="gpu",
        devices=[2],
    )

    for config in configs.values():
        trainer = config.kwargs["pl_trainer_kwargs"]
        assert trainer["accelerator"] == "gpu"
        assert trainer["devices"] == [2]
        assert trainer["precision"] == "16-mixed"

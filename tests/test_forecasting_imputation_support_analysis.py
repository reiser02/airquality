from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

import airquality.forecasting.imputation_support_analysis as analysis
from airquality.forecasting.detection import DetectionResult
from airquality.forecasting.imputation_support_analysis import (
    training_support_diagnostics,
)


def test_training_support_diagnostics_uses_one_worst_case_threshold() -> None:
    index = pd.date_range("2024-01-01", periods=12, freq="h")
    raw = pd.Series(1.0, index=index, name="ST")
    raw.iloc[3] = np.nan
    anomaly_mask = pd.Series(False, index=index)
    anomaly_mask.iloc[8] = True
    cleaned = raw.mask(anomaly_mask)
    imputed = cleaned.fillna(1.0)

    result = training_support_diagnostics(
        raw,
        cleaned,
        imputed,
        anomaly_mask,
        holdout_start=index[-1] + pd.Timedelta(hours=1),
        min_train_points=4,
    )

    assert result["n_imputed_before_holdout"] == 2
    assert result["n_imputed_anomalies_before_holdout"] == 1
    assert result["n_imputed_preexisting_gaps_before_holdout"] == 1
    assert result["n_eligible_blocks_before_imputation"] == 1
    assert result["n_eligible_blocks_after_imputation"] == 1
    assert result["n_eligible_points_before_imputation"] == 4
    assert result["n_eligible_points_after_imputation"] == 12
    assert result["n_eligible_points_added"] == 8
    assert result["n_imputed_in_eligible_blocks"] == 2
    assert result["n_observed_points_recovered"] == 6
    assert result["n_recovered_blocks"] == 1
    assert result["imputed_eligible_age_hours_median"] == 6.5
    assert result["imputed_eligible_age_hours_max"] == 9.0
    assert result["recovered_observed_age_hours_median"] == 6.5
    assert result["recovered_observed_age_hours_max"] == 12.0


def test_run_analysis_writes_one_row_per_imputed_arm(tmp_path, monkeypatch) -> None:
    index = pd.date_range("2024-01-01", periods=30, freq="h")
    series = pd.Series(1.0, index=index, name="ST")
    series.iloc[8] = np.nan
    mask = pd.Series(False, index=index)
    mask.iloc[3] = True
    detection = DetectionResult(
        "unlabeled",
        ["dummy"],
        [],
        {"dummy": 1 / 29},
        3.5,
        mask,
        scored_mask=series.notna(),
        n_flagged=1,
        detection_rate=1 / 29,
    )

    @dataclass(frozen=True)
    class FakeStrategy:
        name: str

        def detect(self, _context):
            return detection

    csv_values = {
        ("forecasting", "forecast_models"): ("Model",),
        ("forecasting", "strategies"): ("unlabeled",),
        ("forecasting", "detectors"): ("dummy",),
    }
    int_values = {
        ("forecasting", "holdout"): 4,
        ("forecasting", "context_len"): 2,
        ("forecasting", "short_horizon"): 2,
        ("forecasting", "short_stride"): 1,
        ("forecasting", "short_validation_len"): 2,
        ("forecasting", "long_horizon"): 4,
        ("forecasting", "long_stride"): 2,
        ("forecasting", "long_validation_len"): 4,
    }
    monkeypatch.setattr(
        analysis,
        "cfg_get_csv_list",
        lambda section, option, default, *, cfg=None: csv_values.get(
            (section, option), default
        ),
    )
    monkeypatch.setattr(
        analysis,
        "cfg_get_int",
        lambda section, option, default, cfg=None: int_values.get(
            (section, option), default
        ),
    )
    monkeypatch.setattr(
        analysis,
        "cfg_get_str",
        lambda section, option, default, cfg=None: "interp"
        if (section, option) == ("forecasting", "imputation_model")
        else default,
    )
    monkeypatch.setattr(analysis, "cfg_get_float", lambda *args, **kwargs: args[2])
    monkeypatch.setattr(analysis, "cfg_get_bool", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        analysis,
        "resolve_forecasting_model_configs",
        lambda *args, **kwargs: {"Model": SimpleNamespace(raw_only=False)},
    )
    monkeypatch.setattr(
        analysis,
        "get_forecast_model_requirements",
        lambda *args, **kwargs: SimpleNamespace(
            min_train_series_length=4,
            prediction_context_length=2,
            validation_target_offset=None,
            validation_target_length=0,
        ),
    )
    monkeypatch.setattr(analysis, "resolve_model_names", lambda names: names)
    monkeypatch.setattr(
        analysis, "build_detection_strategy", lambda spec, **kwargs: FakeStrategy(spec)
    )
    monkeypatch.setattr(analysis, "SeriesDetectionContext", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        analysis, "_load_raw_hourly_series", lambda **kwargs: [series.to_frame()]
    )
    monkeypatch.setattr(analysis, "_imputer_identity", lambda *args, **kwargs: {})
    monkeypatch.setattr(analysis, "build_imputer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        analysis,
        "impute_series",
        lambda values, *_args, **_kwargs: values.interpolate(
            method="time", limit_direction="both"
        ),
    )
    monkeypatch.setattr(analysis, "_repo_root", lambda: tmp_path)

    artifacts = analysis.run_analysis()
    results = artifacts["results_df"]

    assert len(results) == 1
    assert results.iloc[0]["arm"] == "unlabeled+impute"
    assert results.iloc[0]["worst_case_min_train_points"] == 4
    assert "model" not in results.columns
    assert "regime" not in results.columns
    assert (artifacts["output_dir"] / "training_support.csv").exists()
    assert not list(artifacts["output_dir"].glob("*.png"))

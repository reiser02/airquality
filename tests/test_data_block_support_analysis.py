import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import airquality.data.block_support_analysis as analysis
from airquality.forecasting.detection import DetectionResult
from airquality.data.block_support_analysis import (
    _gap_diagnostics,
    _normalize_pollutant,
)


def test_gap_diagnostics_keeps_adjacent_gap_over_limit_unfilled() -> None:
    index = pd.date_range("2024-01-01", periods=20, freq="h")
    raw = pd.Series(1.0, index=index, name="ST")
    raw.iloc[5:10] = np.nan
    mask = pd.Series(False, index=index)
    mask.iloc[4] = True
    cleaned = raw.mask(mask)
    detection = DetectionResult("test", [], [], {}, 3.5, mask)

    rows = _gap_diagnostics(
        raw, cleaned, cleaned, detection, strategy="test", max_gap_size=5
    )

    mixed = next(row for row in rows if row["origin"] == "mixed")
    assert mixed["hours"] == 6
    assert not mixed["eligible_for_imputation"]
    assert mixed["filled_hours"] == 0


@pytest.mark.parametrize("value, expected", [("co", "CO"), (" NO2 ", "NO2"), ("o3", "O3")])
def test_normalize_pollutant_accepts_supported_case_insensitive_values(
    value: str, expected: str
) -> None:
    assert _normalize_pollutant(value) == expected


@pytest.mark.parametrize("value", ["NO*", "../CO", "PM10", ""])
def test_normalize_pollutant_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="Contaminante no soportado"):
        _normalize_pollutant(value)


def test_run_analysis_writes_raw_detected_and_imputed_stages(tmp_path, monkeypatch) -> None:
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
        ("forecasting", "carla_stride"): 3,
        ("forecasting", "horizon"): 2,
        ("forecasting", "stride"): 1,
        ("forecasting", "validation_len"): 2,
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
        lambda *args, **kwargs: {"Model": SimpleNamespace(uses_training_arms=True)},
    )
    monkeypatch.setattr(
        analysis,
        "get_strict_forecast_requirements",
        lambda *args, **kwargs: {
            "minimum_hours": 4,
            "minimum_models": "Model",
            "prediction_context_hours": 2,
            "context_models": "Model",
            "validation_hours": 0,
            "host_minimum_hours": 4,
            "limiting_models": "Model",
            "validation_forecasts": 0,
        },
    )
    monkeypatch.setattr(analysis, "resolve_model_names", lambda names: names)
    monkeypatch.setattr(
        analysis, "build_detection_strategy", lambda spec, **kwargs: FakeStrategy(spec)
    )
    detect_call: dict[str, object] = {}

    def fake_detect(*_args, **kwargs):
        detect_call.update(kwargs)
        return {"unlabeled": detection}

    monkeypatch.setattr(analysis, "_detect_for_strategies", fake_detect)
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
    artifacts = analysis.run_analysis(output_dir=tmp_path, pollutant="co")
    results = artifacts["series_summary_df"]

    assert len(results) == 3
    assert set(results["arm"]) == {
        "raw",
        "unlabeled+noimpute",
        "unlabeled+impute",
    }
    assert set(results["stage"]) == {"raw", "detected", "imputed"}
    assert "regime" not in results.columns
    assert not results["arm"].eq("raw+impute").any()
    assert results.loc[results["stage"] == "imputed", "imputed"].all()
    assert set(results["minimum_hours"]) == {4}
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_version"] == 6
    assert manifest["pollutant"] == "CO"
    assert manifest["injection_variant"] == "combined"
    assert manifest["injection_seed"] == analysis.DEFAULT_INJECTION_SEED
    assert manifest["injection_policy"] == analysis.INJECTION_POLICY_VERSION
    assert manifest["carla_stride"] == 3
    assert detect_call["base_key"]["carla_stride"] == 3
    assert detect_call["context_kwargs"]["carla_stride"] == 3
    assert detect_call["context_kwargs"]["cache_key"]["carla_stride"] == 3
    for filename in (
        "summary.csv",
        "series_summary.csv",
        "blocks.csv",
        "gaps.csv",
        "detection.csv",
        "excluded_series.csv",
        "manifest.json",
        "README.md",
    ):
        assert (tmp_path / filename).exists()
    assert not list(artifacts["output_dir"].glob("*.png"))

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import airquality.data.block_support_analysis as analysis
from airquality.forecasting.detection import DetectionResult
from airquality.forecasting.fill import (
    GapImputationOutcome,
    GapImputationResult,
    nan_gap_windows,
    parse_imputation_gap_rules,
)
from airquality.data.block_support_analysis import (
    _effective_block_index,
    _gap_diagnostics,
    _normalize_pollutant,
)
from airquality.data.block_analysis import classify_blocks


def test_effective_block_index_excludes_validation_tail_and_later_blocks() -> None:
    starts = pd.date_range("2024-01-01", periods=3, freq="10h")
    blocks = pd.DataFrame(
        {
            "start": starts,
            "end": [start + pd.Timedelta(hours=hours - 1) for start, hours in zip(starts, (5, 8, 5))],
            "hours": (5, 8, 5),
        }
    )
    classified = classify_blocks(
        blocks,
        minimum_hours=4,
        validation_hours=2,
        host_minimum_hours=6,
    )

    effective = _effective_block_index(classified, validation_hours=2)

    assert len(effective) == 11
    assert effective[-1] == starts[1] + pd.Timedelta(hours=5)
    assert effective[-1] < starts[2]


def test_gap_diagnostics_keeps_adjacent_gap_over_limit_unfilled() -> None:
    index = pd.date_range("2024-01-01", periods=20, freq="h")
    raw = pd.Series(1.0, index=index, name="ST")
    raw.iloc[5:10] = np.nan
    mask = pd.Series(False, index=index)
    mask.iloc[4] = True
    cleaned = raw.mask(mask)
    detection = DetectionResult("test", [], [], {}, 3.5, mask)
    policy = parse_imputation_gap_rules(
        "1-5=interp;6-10=TSPulse"
    )

    rows = _gap_diagnostics(
        raw,
        cleaned,
        cleaned,
        detection,
        strategy="test",
        policy=policy,
        outcomes=(
            GapImputationOutcome(
                start=cleaned.index[4],
                end=cleaned.index[9],
                hours=6,
                configured_imputer="TSPulse",
                effective_imputer="none",
                fallback_used=False,
                filled_hours=0,
            ),
        ),
    )

    mixed = next(row for row in rows if row["origin"] == "mixed")
    assert mixed["hours"] == 6
    assert mixed["eligible_for_imputation"]
    assert mixed["configured_imputer"] == "TSPulse"
    assert not mixed["fallback_used"]
    assert mixed["effective_imputer"] == "none"
    assert mixed["filled_hours"] == 0


def test_gap_diagnostics_marks_unconfigured_sizes_ineligible() -> None:
    index = pd.date_range("2024-01-01", periods=20, freq="h")
    raw = pd.Series(1.0, index=index, name="ST")
    raw.iloc[5:16] = np.nan
    mask = pd.Series(False, index=index)
    detection = DetectionResult("test", [], [], {}, 3.5, mask)
    policy = parse_imputation_gap_rules("1-10=interp")

    (row,) = _gap_diagnostics(
        raw,
        raw,
        raw,
        detection,
        strategy="test",
        policy=policy,
        outcomes=(
            GapImputationOutcome(
                start=raw.index[5],
                end=raw.index[15],
                hours=11,
                configured_imputer="none",
                effective_imputer="none",
                fallback_used=False,
                filled_hours=0,
            ),
        ),
    )

    assert not row["eligible_for_imputation"]
    assert row["configured_imputer"] == "none"


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
        ("forecasting", "strategies"): ("unlabeled", "inject-best"),
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
        lambda section, option, default, cfg=None: "1-5=interp"
        if (section, option) == ("forecasting", "imputation_gap_rules")
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
        return {"unlabeled": detection, "inject-vote": detection}

    monkeypatch.setattr(analysis, "_detect_for_strategies", fake_detect)
    monkeypatch.setattr(
        analysis, "_load_raw_hourly_series", lambda **kwargs: [series.to_frame()]
    )
    monkeypatch.setattr(
        analysis, "imputation_policy_cache_identity", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(analysis, "build_imputer", lambda *args, **kwargs: object())

    def fake_impute(values, policy, *_args, **_kwargs):
        filled = values.interpolate(method="time", limit_direction="both")
        outcomes = tuple(
            GapImputationOutcome(
                start=window[0],
                end=window[-1],
                hours=len(window),
                configured_imputer=policy.model_for(len(window)) or "none",
                effective_imputer="interp",
                fallback_used=False,
                filled_hours=len(window),
            )
            for window in nan_gap_windows(values)
        )
        return GapImputationResult(filled, outcomes)

    monkeypatch.setattr(
        analysis,
        "impute_series_by_gap_result",
        fake_impute,
    )
    artifacts = analysis.run_analysis(
        output_dir=tmp_path,
        pollutant="co",
        strategies=("inject-vote",),
    )
    results = artifacts["series_summary_df"]

    assert len(results) == 3
    assert set(results["arm"]) == {
        "raw",
        "inject-vote+noimpute",
        "inject-vote+impute",
    }
    assert set(results["stage"]) == {"raw", "detected", "imputed"}
    assert "regime" not in results.columns
    assert not results["arm"].eq("raw+impute").any()
    assert results.loc[results["stage"] == "imputed", "imputed"].all()
    assert set(results["minimum_hours"]) == {4}
    assert {
        "effective_imputed_hours",
        "effective_imputed_anomaly_hours",
        "effective_imputed_preexisting_gap_hours",
        "imputation_gain_effective_hours",
    }.issubset(results.columns)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_version"] == 8
    assert manifest["pollutant"] == "CO"
    assert manifest["injection_variant"] == "combined"
    assert manifest["injection_seed"] == analysis.DEFAULT_INJECTION_SEED
    assert manifest["injection_policy"] == analysis.INJECTION_POLICY_VERSION
    assert manifest["carla_stride"] == 3
    assert manifest["imputation_gap_rules"] == "1-5=interp"
    assert set(artifacts["gaps_df"]["configured_imputer"]) == {"interp"}
    assert not artifacts["gaps_df"]["fallback_used"].any()
    assert set(artifacts["gaps_df"]["effective_imputer"]) == {"interp"}
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
        "benchmark.log",
    ):
        assert (tmp_path / filename).exists()
    log_text = (tmp_path / "benchmark.log").read_text(encoding="utf-8")
    assert "Run started: block_support_analysis" in log_text
    assert "Series 1/1 completed: ST" in log_text
    assert "Run completed: block_support_analysis" in log_text
    assert not list(artifacts["output_dir"].glob("*.png"))


def test_run_analysis_preserves_log_when_analysis_fails(tmp_path, monkeypatch) -> None:
    def fail_analysis(**_kwargs):
        raise RuntimeError("support failed")

    monkeypatch.setattr(analysis, "_run_analysis", fail_analysis)

    with pytest.raises(RuntimeError, match="support failed"):
        analysis.run_analysis(output_dir=tmp_path, pollutant="NO2")

    log_text = (tmp_path / "benchmark.log").read_text(encoding="utf-8")
    assert "Run failed: block_support_analysis" in log_text
    assert "RuntimeError: support failed" in log_text

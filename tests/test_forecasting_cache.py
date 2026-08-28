"""Tests for the forecasting-benchmark disk cache."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from _forecasting_helpers import BASELINE_DETECTORS, _seasonal_series
import airquality.forecasting.pipeline as cp
from airquality.forecasting.cache import (
    BenchmarkCache,
    artifact_fingerprint,
    effective_config,
    series_fingerprint,
    transform_fingerprints,
)

# --------------------------------------------------------------------------- #
# Fingerprints + store
# --------------------------------------------------------------------------- #
def test_series_fingerprint_tracks_content():
    series = _seasonal_series(n=100)
    same = series.copy()
    assert series_fingerprint(series) == series_fingerprint(same)

    changed = series.copy()
    changed.iloc[10] += 1.0
    assert series_fingerprint(series) != series_fingerprint(changed)

    with_nan = series.copy()
    with_nan.iloc[10] = np.nan
    assert series_fingerprint(series) != series_fingerprint(with_nan)


def test_artifact_fingerprint_tracks_local_content(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"first")
    first = artifact_fingerprint(artifact)
    artifact.write_bytes(b"second")
    assert artifact_fingerprint(artifact) != first


def test_cache_roundtrip_and_key_isolation(tmp_path):
    cache = BenchmarkCache(tmp_path)
    key = {"stage": "backtest", "model": "NLinear", "train_fp": "abc"}
    assert cache.get("backtest", key) is None  # miss before put
    cache.put("backtest", key, {"rmse": 1.5})
    assert cache.get("backtest", key) == {"rmse": 1.5}
    # A different key never reads another entry's value.
    assert cache.get("backtest", {**key, "model": "TiDE"}) is None
    assert cache.hits == 1 and cache.misses == 2
    assert cache.stats_by_namespace() == {"backtest": {"hits": 1, "misses": 2}}
    assert "backtest=1 aciertos/2 fallos" in cache.stats()


def test_cache_disabled_is_noop():
    cache = BenchmarkCache(None)
    cache.put("backtest", {"a": 1}, "value")
    assert cache.get("backtest", {"a": 1}) is None
    assert not cache.enabled


def test_cache_survives_corrupt_entry(tmp_path):
    cache = BenchmarkCache(tmp_path)
    key = {"a": 1}
    cache.put("detection", key, "value")
    path = next((tmp_path / "detection").glob("*.pkl"))
    path.write_bytes(b"corrupt")
    assert cache.get("detection", key) is None  # degrades to a miss


def test_cache_treats_malformed_payload_as_miss(tmp_path):
    cache = BenchmarkCache(tmp_path)
    key = {"a": 1}
    cache.put("detection", key, "value")
    path = next((tmp_path / "detection").glob("*.pkl"))

    for payload in ([], {"key": key}):
        path.write_bytes(pickle.dumps(payload))
        assert cache.get("detection", key) is None


def test_transform_fingerprints_names():
    def dilate(series, mask):
        return mask

    assert transform_fingerprints(None) == []
    assert transform_fingerprints([dilate])[0].startswith(
        f"{__name__}.test_transform_fingerprints_names.<locals>.dilate:"
    )


def test_transform_fingerprints_track_implementation_and_state():
    keep = lambda _series, mask: mask
    invert = lambda _series, mask: ~mask
    assert transform_fingerprints([keep]) != transform_fingerprints([invert])

    def make_transform(radius):
        def dilate(_series, mask):
            return mask.rolling(radius, min_periods=1).max().astype(bool)

        return dilate

    assert transform_fingerprints([make_transform(2)]) != transform_fingerprints(
        [make_transform(3)]
    )


def test_effective_config_and_key_exclude_runtime_device(tmp_path):
    cpu = effective_config(
        {
            "forecasting": {"seed": 13, "device": "cpu"},
            "training": {"n_epochs": "3", "accelerator": "cpu"},
        }
    )
    cuda = effective_config(
        {
            "forecasting": {"seed": 13, "device": "cuda"},
            "training": {"n_epochs": "3", "accelerator": "gpu"},
        }
    )
    assert cpu == cuda
    assert "device" not in cpu["forecasting"]

    cache = BenchmarkCache(tmp_path)
    cache.put("detection", {"config": cpu}, "cached")
    assert cache.get("detection", {"config": cuda}) == "cached"


# --------------------------------------------------------------------------- #
# Pipeline resume: second run recomputes nothing
# --------------------------------------------------------------------------- #
def test_run_benchmark_resumes_from_cache(tmp_path, monkeypatch):
    frozen_delta = {"value": 0.0}

    def fake_loader(**kwargs):
        s = _seasonal_series(n=900, name="ST0", seed=0)
        s.iloc[200] = 130.0
        s.iloc[300:330] = np.nan
        if kwargs.get("preserve_frozen"):
            s.iloc[100] += frozen_delta["value"]
        return [s.to_frame()]

    csv_map = {
        ("forecasting", "detectors"): tuple(BASELINE_DETECTORS),
        ("forecasting", "forecast_models"): ("LinearRegression",),
        ("forecasting", "strategies"): ("unlabeled",),
    }
    int_map = {
        ("forecasting", "holdout"): 96,
        ("forecasting", "context_len"): 72,
        ("forecasting", "min_series_points"): 300,
    }
    str_map = {
        ("forecasting", "imputation"): "impute",
        ("forecasting", "imputation_model"): "interp",
        ("forecasting", "cache_dir"): str(tmp_path / "cache"),
    }

    monkeypatch.setattr(cp, "_load_raw_hourly_series", fake_loader)
    monkeypatch.setattr(cp, "cfg_get_csv_list", lambda s, o, d, *, cfg=None: csv_map.get((s, o), d))
    monkeypatch.setattr(cp, "cfg_get_int", lambda s, o, d, cfg=None: int_map.get((s, o), d))
    monkeypatch.setattr(cp, "cfg_get_str", lambda s, o, d, cfg=None: str_map.get((s, o), d))
    monkeypatch.setattr(cp, "_build_output_dir", lambda: tmp_path / "run1")
    monkeypatch.setattr(cp, "resolve_forecasting_devices", lambda _request: ("cpu",))
    (tmp_path / "run1").mkdir()

    calls = {"backtest": 0, "detect": 0}
    real_backtest = cp.backtest_forecast
    real_detect = cp.SeriesDetectionContext.real_scores

    def counting_backtest(*args, **kwargs):
        calls["backtest"] += 1
        return real_backtest(*args, **kwargs)

    def counting_scores(self, names):
        calls["detect"] += 1
        return real_detect(self, names)

    monkeypatch.setattr(cp, "backtest_forecast", counting_backtest)
    monkeypatch.setattr(cp.SeriesDetectionContext, "real_scores", counting_scores)

    first = cp.run_benchmark_from_config()
    assert calls["backtest"] == 3  # three arms x one forecast protocol
    assert calls["detect"] > 0

    calls["backtest"] = 0
    calls["detect"] = 0
    monkeypatch.setattr(cp, "_build_output_dir", lambda: tmp_path / "run2")
    (tmp_path / "run2").mkdir()
    second = cp.run_benchmark_from_config()

    # Everything came from the cache: no detector fits, no model training.
    assert calls["backtest"] == 0
    assert calls["detect"] == 0
    pd.testing.assert_frame_equal(first["results_df"], second["results_df"])
    pd.testing.assert_frame_equal(first["detection_df"], second["detection_df"])
    pd.testing.assert_frame_equal(first["selection_df"], second["selection_df"])

    calls["backtest"] = 0
    frozen_delta["value"] = 1.0
    monkeypatch.setattr(cp, "_build_output_dir", lambda: tmp_path / "run3")
    (tmp_path / "run3").mkdir()
    cp.run_benchmark_from_config()

    assert calls["backtest"] == 1  # only raw+frozen, one forecast protocol
    assert calls["detect"] == 0

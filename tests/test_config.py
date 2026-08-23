"""Tests config path discovery and repository config precedence."""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path

from airquality import config


def test_candidate_config_paths_prioritize_repo_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = config._candidate_config_paths()

    assert paths == [
        repo_root / "config" / "pipeline.cfg",
        repo_root / "pipeline.cfg",
        repo_root / "src" / "airquality" / "pipeline.cfg",
    ]


def test_repository_config_separates_explicit_model_catalogs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigParser()
    cfg.read(repo_root / "config" / "pipeline.cfg")

    darts_models = [
        "TiDE",
        "NHiTS",
        "NLinear",
        "DLinear",
        "TCN",
        "TSMixer",
        "RNN",
        "LinearRegression",
    ]
    assert cfg.get("training", "model_names").split(",") == darts_models
    assert cfg.get("imputation", "model_names").split(",") == [
        *darts_models,
        "Prophet",
        "TSPulse",
        "TSPulse_FineTuned",
        "interp",
        "LinearInterp",
    ]
    assert cfg.getboolean("imputation", "strict_artifacts") is True
    assert cfg.getint("imputation", "max_workers") == 8
    assert cfg.get("data", "raw_base_dir") == "data/raw/datos_estaciones_5m"
    assert cfg.get("data", "data_root") == "data/raw/datos_estaciones_5m"
    assert cfg.get("synthetic", "injection_variant") == "combined"
    assert not cfg.has_option("data", "base_path_glob")
    assert cfg.getint("anomaly", "min_series_points") == 8
    assert not cfg.has_option("anomaly", "raw_base_dir")
    assert cfg.get("forecasting", "imputation_model") == "TSPulse"
    assert cfg.getint("forecasting", "max_imputation_gap") == 5
    assert cfg.get("forecasting", "pollutant") == "NO2"
    assert cfg.get("forecasting", "device") == "multi-gpu"
    assert not cfg.has_option("forecasting", "raw_base_dir")
    assert cfg.getboolean("forecasting", "foundation_preprocessing_test") is True
    assert cfg.getint("forecasting", "foundation_test_seed") == 1001
    assert cfg.getint("forecasting", "foundation_test_repeats") == 1
    assert not cfg.has_option("benchmark", "model_names")
    assert cfg.has_option("tspulse", "finetuned_model_path")

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
        "TSPulse",
    ]
    assert cfg.getboolean("imputation", "strict_artifacts") is True
    assert cfg.getint("imputation", "max_workers") == 1
    assert not cfg.has_option("benchmark", "model_names")
    assert cfg.has_option("tspulse", "finetuned_model_path")

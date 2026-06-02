from __future__ import annotations

from pathlib import Path

from airquality import config


def test_candidate_config_paths_prioritize_repo_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = config._candidate_config_paths()

    assert paths[0] == repo_root / "config" / "pipeline.cfg"

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path


def _candidate_config_paths() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return [
        repo_root / "config" / "pipeline.cfg",
        repo_root / "pipeline.cfg",
        here.parent / "pipeline.cfg",
    ]


def load_config() -> ConfigParser:
    cfg = ConfigParser()
    cfg.read([str(path) for path in _candidate_config_paths() if path.exists()])
    return cfg


CONFIG = load_config()


def cfg_get_str(section: str, option: str, default: str) -> str:
    return CONFIG.get(section, option, fallback=default)


def cfg_get_int(section: str, option: str, default: int) -> int:
    return CONFIG.getint(section, option, fallback=default)


def cfg_get_float(section: str, option: str, default: float) -> float:
    return CONFIG.getfloat(section, option, fallback=default)


def cfg_get_bool(section: str, option: str, default: bool) -> bool:
    return CONFIG.getboolean(section, option, fallback=default)


def cfg_get_csv_list(section: str, option: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = CONFIG.get(section, option, fallback=",".join(default))
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or default

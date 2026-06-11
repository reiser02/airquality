"""Configuration loading helpers for the airquality pipelines."""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path


def _candidate_config_paths() -> list[Path]:
    """Return config files in the order they should override each other."""
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return [
        repo_root / "config" / "pipeline.cfg",
        repo_root / "pipeline.cfg",
        here.parent / "pipeline.cfg",
    ]


def load_config() -> ConfigParser:
    """Load the first existing project config files into one parser."""
    cfg = ConfigParser()
    cfg.read([str(path) for path in _candidate_config_paths() if path.exists()])
    return cfg


CONFIG = load_config()


def get_config(cfg: ConfigParser | None = None) -> ConfigParser:
    """Return the provided parser or the module-level shared config."""
    return CONFIG if cfg is None else cfg


def cfg_get_str(section: str, option: str, default: str, cfg: ConfigParser | None = None) -> str:
    """Read one string option with a fallback default."""
    return get_config(cfg).get(section, option, fallback=default)


def cfg_get_int(section: str, option: str, default: int, cfg: ConfigParser | None = None) -> int:
    """Read one integer option with a fallback default."""
    return get_config(cfg).getint(section, option, fallback=default)


def cfg_get_float(section: str, option: str, default: float, cfg: ConfigParser | None = None) -> float:
    """Read one float option with a fallback default."""
    return get_config(cfg).getfloat(section, option, fallback=default)


def cfg_get_bool(section: str, option: str, default: bool, cfg: ConfigParser | None = None) -> bool:
    """Read one boolean option with a fallback default."""
    return get_config(cfg).getboolean(section, option, fallback=default)


def cfg_get_csv_list(
    section: str,
    option: str,
    default: tuple[str, ...],
    *,
    cfg: ConfigParser | None = None,
) -> tuple[str, ...]:
    """Read a comma-separated option and return trimmed non-empty values."""
    raw = get_config(cfg).get(section, option, fallback=",".join(default))
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or default

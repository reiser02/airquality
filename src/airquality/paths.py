"""Project path helpers."""

from pathlib import Path


def create_run_dir(base_dir: Path, name: str) -> Path:
    """Atomically create ``name``, adding a numeric suffix on collisions."""
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        candidate = base_dir / (name if suffix == 0 else f"{name}_{suffix}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1


__all__ = ["create_run_dir"]

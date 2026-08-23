"""Enforce the package boundary and module documentation for visualizations."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "src" / "airquality"
VISUALIZATIONS = PACKAGE / "visualizations"


def test_matplotlib_is_confined_to_visualizations() -> None:
    offenders = [
        path.relative_to(PACKAGE)
        for path in PACKAGE.rglob("*.py")
        if VISUALIZATIONS not in path.parents
        and "matplotlib" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_visualization_modules_start_with_a_purpose_docstring() -> None:
    undocumented = [
        path.name
        for path in VISUALIZATIONS.glob("*.py")
        if not ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    ]

    assert undocumented == []

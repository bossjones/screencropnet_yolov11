"""Pins for the package rename screencropnet_yolov11 -> screencropnet_yolo."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from typing import Any

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _load_pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text())


def test_package_importable_as_screencropnet_yolo() -> None:
    """The package imports under its new name and the old name is gone."""
    module = importlib.import_module("screencropnet_yolo")
    assert module.__name__ == "screencropnet_yolo"

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("screencropnet_yolov11")


def test_pyproject_project_name_is_screencropnet_yolo() -> None:
    """[project].name reflects the version-agnostic package name."""
    data = _load_pyproject()
    assert data["project"]["name"] == "screencropnet_yolo"


def test_pyproject_script_entrypoint() -> None:
    """The console script is declared against the renamed package."""
    scripts = _load_pyproject()["project"]["scripts"]
    assert scripts["screencropnet_yolo"] == "screencropnet_yolo:main"


def test_supervisor_console_scripts_resolve() -> None:
    """Both supervisor console scripts point at the same importable, callable entry point."""
    scripts = _load_pyproject()["project"]["scripts"]
    target = "screencropnet_yolo.client.supervisor:main"
    assert scripts["screencrop-supervisor-worker"] == target
    assert scripts["screencrop-supervisorctl"] == target

    module_name, attr = target.split(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attr))


def test_hatch_wheel_packages_point_to_new_src() -> None:
    """Hatch builds the wheel from the renamed source directory."""
    data = _load_pyproject()
    packages = data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert packages == ["src/screencropnet_yolo"]

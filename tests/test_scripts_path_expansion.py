"""Path-argument expansion behavior shared across the standalone ``scripts/``.

Every script that accepts a filesystem path on the command line must expand ``~``
and ``$VAR`` references so quoted/Makefile/env-var paths resolve the same way an
unquoted shell argument would. These tests pin that behavior per script and guard
that each script is actually wired to use the helper.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.add_images_to_labeling import expanded_path as add_images_expanded
from scripts.download_screennet_weights import expanded_path as download_expanded
from scripts.ls_yolo_export_to_dataset import expanded_path as ls_yolo_expanded
from scripts.pascal_csv_to_ls_tasks import expanded_path as pascal_expanded
from scripts.setup_ls_project import expanded_path as setup_expanded

Expander = Callable[[str], Path]

EXPANDERS = pytest.mark.parametrize(
    "expand",
    [
        add_images_expanded,
        download_expanded,
        ls_yolo_expanded,
        pascal_expanded,
        setup_expanded,
    ],
)


@EXPANDERS
def test_expands_bare_tilde(expand: Expander):
    assert expand("~") == Path.home()


@EXPANDERS
def test_expands_tilde_prefix(expand: Expander):
    assert expand("~/Downloads/x") == Path.home() / "Downloads" / "x"


@EXPANDERS
def test_expands_env_var(expand: Expander, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCN_TEST_DIR", "/tmp/scn-test")
    assert expand("$SCN_TEST_DIR/a") == Path("/tmp/scn-test/a")


@EXPANDERS
def test_combines_var_and_tilde(expand: Expander, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCN_SUB", "Downloads")
    assert expand("~/$SCN_SUB/img") == Path.home() / "Downloads" / "img"


@EXPANDERS
def test_absolute_path_unchanged(expand: Expander):
    assert expand("/var/data/x") == Path("/var/data/x")


@EXPANDERS
def test_unknown_var_left_literal(expand: Expander, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SCN_UNDEFINED", raising=False)
    assert expand("$SCN_UNDEFINED/x") == Path("$SCN_UNDEFINED/x")


@EXPANDERS
def test_returns_path(expand: Expander):
    assert isinstance(expand("~"), Path)


SCRIPTS = [
    "add_images_to_labeling.py",
    "download_screennet_weights.py",
    "ls_yolo_export_to_dataset.py",
    "pascal_csv_to_ls_tasks.py",
    "setup_ls_project.py",
]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_wires_expanded_path(name: str):
    src = (Path(__file__).parent.parent / "scripts" / name).read_text()
    assert "def expanded_path" in src
    assert "type=expanded_path" in src
    assert "type=Path" not in src  # all path args migrated

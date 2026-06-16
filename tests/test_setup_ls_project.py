"""Tests for the Label Studio project setup script.

These exercise the SDK-free helpers directly and the orchestrator via an
injected mock client, so they run inside the project venv without
``label-studio-sdk`` installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from scripts.setup_ls_project import (
    build_label_config,
    find_project_by_title,
    load_seed_tasks,
    resolve_api_key,
    setup_project,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_build_label_config_targets_image_with_tweet_region() -> None:
    config = build_label_config()
    assert "<RectangleLabels" in config
    assert 'value="tweet_region"' in config
    assert 'value="$image"' in config


def test_resolve_api_key_prefers_explicit_arg() -> None:
    assert resolve_api_key("abc", env={"LABEL_STUDIO_API_KEY": "from-env"}) == "abc"


def test_resolve_api_key_falls_back_to_env() -> None:
    assert resolve_api_key(None, env={"LABEL_STUDIO_API_KEY": "from-env"}) == "from-env"


def test_resolve_api_key_raises_when_missing() -> None:
    with pytest.raises(ValueError, match="LABEL_STUDIO_API_KEY"):
        resolve_api_key(None, env={})


def test_load_seed_tasks_reads_list(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    payload = [{"data": {"image": "/img/a.png"}}, {"data": {"image": "/img/b.png"}}]
    tasks_path.write_text(json.dumps(payload))

    assert load_seed_tasks(tasks_path) == payload


def test_load_seed_tasks_rejects_non_list(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps({"data": {}}))

    with pytest.raises(ValueError, match="list of tasks"):
        load_seed_tasks(tasks_path)


def test_load_seed_tasks_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_seed_tasks(tmp_path / "absent.json")


def test_find_project_by_title_returns_match(mocker: MockerFixture) -> None:
    client = mocker.MagicMock()
    other = mocker.MagicMock(title="something else")
    wanted = mocker.MagicMock(title="screencropnet")
    client.projects.list.return_value = [other, wanted]

    assert find_project_by_title(client, "screencropnet") is wanted


def test_find_project_by_title_returns_none_when_absent(mocker: MockerFixture) -> None:
    client = mocker.MagicMock()
    client.projects.list.return_value = [mocker.MagicMock(title="other")]

    assert find_project_by_title(client, "screencropnet") is None


def test_setup_project_creates_imports_and_connects(mocker: MockerFixture) -> None:
    client = mocker.MagicMock()
    client.projects.list.return_value = []
    client.projects.create.return_value = mocker.MagicMock(id=7)
    client.projects.import_tasks.return_value = mocker.MagicMock(task_count=2)
    client.ml.list.return_value = []
    client.ml.create.return_value = mocker.MagicMock(id=11)

    tasks = [{"data": {"image": "/img/a.png"}}, {"data": {"image": "/img/b.png"}}]
    summary = setup_project(
        client,
        title="screencropnet",
        label_config="<View/>",
        tasks=tasks,
        ml_backend_url="http://localhost:9090",
        ml_title="ScreenCropNet ML Backend",
        is_interactive=True,
        connect_ml_backend=True,
        connect_local_storage=False,
        local_storage_path=None,
        local_storage_title="Twitter screenshots (local)",
        reuse=True,
        force_import=False,
    )

    client.projects.create.assert_called_once_with(title="screencropnet", label_config="<View/>")
    client.projects.import_tasks.assert_called_once_with(id=7, request=tasks)
    client.ml.create.assert_called_once_with(
        project=7,
        url="http://localhost:9090",
        title="ScreenCropNet ML Backend",
        is_interactive=True,
    )
    assert summary["created"] is True
    assert summary["project_id"] == 7
    assert summary["imported_count"] == 2
    assert summary["ml_backend_id"] == 11


def test_setup_project_reuses_existing_and_skips_import(mocker: MockerFixture) -> None:
    client = mocker.MagicMock()
    existing = mocker.MagicMock(id=3, title="screencropnet")
    client.projects.list.return_value = [existing]
    client.ml.list.return_value = []

    summary = setup_project(
        client,
        title="screencropnet",
        label_config="<View/>",
        tasks=[{"data": {"image": "/img/a.png"}}],
        ml_backend_url="http://localhost:9090",
        ml_title="ScreenCropNet ML Backend",
        is_interactive=True,
        connect_ml_backend=True,
        connect_local_storage=False,
        local_storage_path=None,
        local_storage_title="Twitter screenshots (local)",
        reuse=True,
        force_import=False,
    )

    client.projects.create.assert_not_called()
    client.projects.import_tasks.assert_not_called()
    assert summary["created"] is False
    assert summary["project_id"] == 3
    assert summary["imported_count"] == 0


def test_setup_project_skips_duplicate_ml_backend(mocker: MockerFixture) -> None:
    client = mocker.MagicMock()
    existing = mocker.MagicMock(id=3, title="screencropnet")
    client.projects.list.return_value = [existing]
    client.ml.list.return_value = [mocker.MagicMock(id=99, url="http://localhost:9090")]

    summary = setup_project(
        client,
        title="screencropnet",
        label_config="<View/>",
        tasks=[],
        ml_backend_url="http://localhost:9090",
        ml_title="ScreenCropNet ML Backend",
        is_interactive=True,
        connect_ml_backend=True,
        connect_local_storage=False,
        local_storage_path=None,
        local_storage_title="Twitter screenshots (local)",
        reuse=True,
        force_import=False,
    )

    client.ml.create.assert_not_called()
    assert summary["ml_backend_id"] == 99


def test_setup_project_can_skip_ml_backend(mocker: MockerFixture) -> None:
    client = mocker.MagicMock()
    client.projects.list.return_value = []
    client.projects.create.return_value = mocker.MagicMock(id=7)
    client.projects.import_tasks.return_value = mocker.MagicMock(task_count=0)

    summary = setup_project(
        client,
        title="screencropnet",
        label_config="<View/>",
        tasks=[],
        ml_backend_url="http://localhost:9090",
        ml_title="ScreenCropNet ML Backend",
        is_interactive=True,
        connect_ml_backend=False,
        connect_local_storage=False,
        local_storage_path=None,
        local_storage_title="Twitter screenshots (local)",
        reuse=True,
        force_import=False,
    )

    client.ml.create.assert_not_called()
    client.projects.import_tasks.assert_not_called()
    assert summary["ml_backend_id"] is None


def test_setup_project_registers_local_storage_without_sync(mocker: MockerFixture) -> None:
    client = mocker.MagicMock()
    existing = mocker.MagicMock(id=3, title="screencropnet")
    client.projects.list.return_value = [existing]
    client.ml.list.return_value = []
    client.import_storage.local.list.return_value = []
    client.import_storage.local.create.return_value = mocker.MagicMock(id=42)

    summary = setup_project(
        client,
        title="screencropnet",
        label_config="<View/>",
        tasks=[],
        ml_backend_url="http://localhost:9090",
        ml_title="ScreenCropNet ML Backend",
        is_interactive=True,
        connect_ml_backend=False,
        connect_local_storage=True,
        local_storage_path="/data/twitter_screenshots_raw/train_images",
        local_storage_title="Twitter screenshots (local)",
        reuse=True,
        force_import=False,
    )

    client.import_storage.local.create.assert_called_once_with(
        project=3,
        path="/data/twitter_screenshots_raw/train_images",
        title="Twitter screenshots (local)",
        use_blob_urls=True,
    )
    # A sync would duplicate the imported seed tasks; registration alone suffices.
    client.import_storage.local.sync.assert_not_called()
    assert summary["local_storage_id"] == 42


def test_setup_project_skips_duplicate_local_storage(mocker: MockerFixture) -> None:
    client = mocker.MagicMock()
    existing = mocker.MagicMock(id=3, title="screencropnet")
    client.projects.list.return_value = [existing]
    client.ml.list.return_value = []
    path = "/data/twitter_screenshots_raw/train_images"
    client.import_storage.local.list.return_value = [mocker.MagicMock(id=88, path=path)]

    summary = setup_project(
        client,
        title="screencropnet",
        label_config="<View/>",
        tasks=[],
        ml_backend_url="http://localhost:9090",
        ml_title="ScreenCropNet ML Backend",
        is_interactive=True,
        connect_ml_backend=False,
        connect_local_storage=True,
        local_storage_path=path,
        local_storage_title="Twitter screenshots (local)",
        reuse=True,
        force_import=False,
    )

    client.import_storage.local.create.assert_not_called()
    assert summary["local_storage_id"] == 88


def test_setup_project_can_skip_local_storage(mocker: MockerFixture) -> None:
    client = mocker.MagicMock()
    existing = mocker.MagicMock(id=3, title="screencropnet")
    client.projects.list.return_value = [existing]
    client.ml.list.return_value = []

    summary = setup_project(
        client,
        title="screencropnet",
        label_config="<View/>",
        tasks=[],
        ml_backend_url="http://localhost:9090",
        ml_title="ScreenCropNet ML Backend",
        is_interactive=True,
        connect_ml_backend=False,
        connect_local_storage=False,
        local_storage_path="/data/twitter_screenshots_raw/train_images",
        local_storage_title="Twitter screenshots (local)",
        reuse=True,
        force_import=False,
    )

    client.import_storage.local.list.assert_not_called()
    client.import_storage.local.create.assert_not_called()
    assert summary["local_storage_id"] is None

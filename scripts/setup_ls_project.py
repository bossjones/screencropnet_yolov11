#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "label-studio-sdk>=1.0",
# ]
# ///
"""Create and configure the ``screencropnet`` Label Studio project via the SDK.

Reproduces the manual UI steps from ``docs/label-studio-annotation-guide.md``:
create the project with the ``tweet_region`` RectangleLabels interface, import the
pre-drawn ``tasks.json`` seed boxes, connect the local ML backend, and register a
Local Files import storage so the ``/data/local-files/?d=...`` URLs in the tasks
resolve. Re-runs are idempotent — an existing project of the same title is reused,
task re-import is skipped (unless ``--force-import``), and a duplicate ML backend
URL or local storage ``path`` is not added.

The local storage is registered but **never synced**: its existence alone
authorizes local-file serving, while syncing would create a second, duplicate set
of tasks from the directory and break the seed-prediction pairing. Creation is
validated server-side against the server's ``LOCAL_FILES_DOCUMENT_ROOT``, so the
server must already be launched with local-file serving (``make label-studio-local``)
for it to succeed.

Prerequisites: Label Studio running (``make label-studio-local``), the ML backend
running (``make ml-backend``), and an API key in ``LABEL_STUDIO_API_KEY`` (or
``--api-key``). The label-studio-sdk is pulled into uv's ephemeral env, so this
must be run via ``uv run`` (it deliberately is not a project dependency).

Run: ``uv run scripts/setup_ls_project.py --title screencropnet --tasks scratch/labeling/tasks.json``
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from label_studio_sdk import LabelStudio

API_KEY_ENV = "LABEL_STUDIO_API_KEY"
DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_TITLE = "screencropnet"
DEFAULT_TASKS = "scratch/labeling/tasks.json"
DEFAULT_ML_BACKEND_URL = "http://localhost:9090"
DEFAULT_ML_TITLE = "ScreenCropNet ML Backend"
DEFAULT_DOCUMENT_ROOT = "scratch/datasets/twitter_screenshots_raw"
DEFAULT_IMAGES_SUBDIR = "train_images"
DEFAULT_LOCAL_STORAGE_TITLE = "Twitter screenshots (local)"


def expanded_path(value: str) -> Path:
    """Resolve ``~`` and ``$VAR`` references in a user-supplied path argument."""
    return Path(os.path.expandvars(value)).expanduser()


def build_label_config() -> str:
    """Return the RectangleLabels labeling interface (mirrors tools/labeling/label_config.xml)."""
    return (
        "<View>\n"
        '  <Image name="image" value="$image"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        '    <Label value="tweet_region" background="#1da1f2"/>\n'
        "  </RectangleLabels>\n"
        "</View>"
    )


def resolve_api_key(api_key: str | None, env: Mapping[str, str]) -> str:
    """Return the explicit key, else ``LABEL_STUDIO_API_KEY`` from ``env``.

    Raises:
        ValueError: if neither source provides a key.
    """
    key = api_key or env.get(API_KEY_ENV)
    if not key:
        raise ValueError(
            f"no API key: pass --api-key or set {API_KEY_ENV} "
            "(create one under Label Studio → Account & Settings)"
        )
    return key


def load_seed_tasks(path: Path) -> list[dict[str, Any]]:
    """Read a Label Studio ``tasks.json`` and return its task list.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the JSON root is not a list of tasks.
    """
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of tasks")
    return payload


def find_project_by_title(client: LabelStudio, title: str) -> Any | None:
    """Return the first project whose title matches, or ``None``."""
    for project in client.projects.list():
        if getattr(project, "title", None) == title:
            return project
    return None


def _existing_ml_backend(client: LabelStudio, project_id: int, url: str) -> Any | None:
    """Return an already-connected ML backend with the same URL, or ``None``."""
    for backend in client.ml.list(project=project_id):
        if getattr(backend, "url", None) == url:
            return backend
    return None


def _existing_local_storage(client: LabelStudio, project_id: int, path: str) -> Any | None:
    """Return an already-registered local import storage with the same ``path``, or ``None``."""
    for storage in client.import_storage.local.list(project=project_id):
        if getattr(storage, "path", None) == path:
            return storage
    return None


def setup_project(
    client: LabelStudio,
    *,
    title: str,
    label_config: str,
    tasks: list[dict[str, Any]],
    ml_backend_url: str,
    ml_title: str,
    is_interactive: bool,
    connect_ml_backend: bool,
    connect_local_storage: bool,
    local_storage_path: str | None,
    local_storage_title: str,
    reuse: bool,
    force_import: bool,
) -> dict[str, Any]:
    """Create or reuse the project, import seed tasks, connect the ML backend, and
    register the local file storage.

    Idempotency: when ``reuse`` is set an existing same-titled project is reused
    instead of created; task import is skipped for a reused project unless
    ``force_import``; and neither an ML backend with the same URL nor a local
    storage with the same ``path`` is added twice.

    The local storage is created but never synced — its existence authorizes
    local-file serving, whereas a sync would duplicate the imported seed tasks.

    Returns a summary dict with ``project_id``, ``created``, ``imported_count``,
    ``ml_backend_id`` (``None`` when not connected), and ``local_storage_id``
    (``None`` when not registered).
    """
    existing = find_project_by_title(client, title) if reuse else None
    if existing is not None:
        project_id = int(existing.id)
        created = False
    else:
        project = client.projects.create(title=title, label_config=label_config)
        project_id = int(project.id)
        created = True

    imported_count = 0
    if tasks and (created or force_import):
        response = client.projects.import_tasks(id=project_id, request=tasks)
        imported_count = int(getattr(response, "task_count", 0) or 0)

    ml_backend_id: int | None = None
    if connect_ml_backend:
        backend = _existing_ml_backend(client, project_id, ml_backend_url)
        if backend is None:
            backend = client.ml.create(
                project=project_id,
                url=ml_backend_url,
                title=ml_title,
                is_interactive=is_interactive,
            )
        ml_backend_id = int(backend.id)

    local_storage_id: int | None = None
    if connect_local_storage and local_storage_path:
        storage = _existing_local_storage(client, project_id, local_storage_path)
        if storage is None:
            # Created, not synced: registering the path authorizes
            # /data/local-files/ serving; a sync would duplicate the seed tasks.
            storage = client.import_storage.local.create(
                project=project_id,
                path=local_storage_path,
                title=local_storage_title,
                use_blob_urls=True,
            )
        local_storage_id = int(storage.id)

    return {
        "project_id": project_id,
        "created": created,
        "imported_count": imported_count,
        "ml_backend_id": ml_backend_id,
        "local_storage_id": local_storage_id,
    }


def build_client(base_url: str, api_key: str) -> LabelStudio:
    """Construct a Label Studio SDK client (import is local so tests need no SDK)."""
    from label_studio_sdk import LabelStudio

    return LabelStudio(base_url=base_url, api_key=api_key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Label Studio base URL")
    parser.add_argument("--api-key", default=None, help=f"API key (defaults to ${API_KEY_ENV})")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="project title")
    parser.add_argument(
        "--tasks",
        type=expanded_path,
        default=Path(DEFAULT_TASKS),
        help="tasks.json to import; skipped if the file is absent",
    )
    parser.add_argument("--ml-backend-url", default=DEFAULT_ML_BACKEND_URL, help="ML backend URL")
    parser.add_argument("--ml-title", default=DEFAULT_ML_TITLE, help="ML backend title")
    parser.add_argument(
        "--no-ml-backend",
        dest="connect_ml_backend",
        action="store_false",
        help="do not connect the ML backend",
    )
    parser.add_argument(
        "--local-files-document-root",
        type=expanded_path,
        default=Path(DEFAULT_DOCUMENT_ROOT),
        help=(
            "local-file serving document root (must match the server's "
            "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT); the storage path is its "
            f"'{DEFAULT_IMAGES_SUBDIR}' subdir"
        ),
    )
    parser.add_argument(
        "--local-storage-title",
        default=DEFAULT_LOCAL_STORAGE_TITLE,
        help="title for the registered local file storage",
    )
    parser.add_argument(
        "--no-local-storage",
        dest="connect_local_storage",
        action="store_false",
        help="do not register a local file import storage",
    )
    parser.add_argument(
        "--no-reuse",
        dest="reuse",
        action="store_false",
        help="always create a new project even if the title already exists",
    )
    parser.add_argument(
        "--force-import",
        action="store_true",
        help="import tasks even when reusing an existing project",
    )
    parser.add_argument(
        "--no-interactive",
        dest="is_interactive",
        action="store_false",
        help="register the ML backend as non-interactive",
    )
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key, os.environ)
    client = build_client(args.base_url, api_key)

    tasks: list[dict[str, Any]] = []
    if args.tasks.is_file():
        tasks = load_seed_tasks(args.tasks)
    elif args.tasks != Path(DEFAULT_TASKS):
        raise FileNotFoundError(f"tasks file not found: {args.tasks}")

    # Absolute path under the document root; must be a prefix of the served files
    # (tasks reference /data/local-files/?d=<images_subdir>/<file>).
    local_storage_path = str(args.local_files_document_root.resolve() / DEFAULT_IMAGES_SUBDIR)

    summary = setup_project(
        client,
        title=args.title,
        label_config=build_label_config(),
        tasks=tasks,
        ml_backend_url=args.ml_backend_url,
        ml_title=args.ml_title,
        is_interactive=args.is_interactive,
        connect_ml_backend=args.connect_ml_backend,
        connect_local_storage=args.connect_local_storage,
        local_storage_path=local_storage_path,
        local_storage_title=args.local_storage_title,
        reuse=args.reuse,
        force_import=args.force_import,
    )

    verb = "created" if summary["created"] else "reused"
    ml = (
        f"ml-backend #{summary['ml_backend_id']}"
        if summary["ml_backend_id"] is not None
        else "no ml-backend"
    )
    storage = (
        f"local-storage #{summary['local_storage_id']}"
        if summary["local_storage_id"] is not None
        else "no local-storage"
    )
    print(
        f"✔︎ {verb} project '{args.title}' (#{summary['project_id']}); "
        f"imported {summary['imported_count']} tasks; {ml}; {storage}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

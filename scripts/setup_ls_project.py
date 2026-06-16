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
pre-drawn ``tasks.json`` seed boxes, and connect the local ML backend. Re-runs are
idempotent — an existing project of the same title is reused, task re-import is
skipped (unless ``--force-import``), and a duplicate ML backend URL is not added.

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


def build_label_config() -> str:
    """Return the RectangleLabels labeling interface (mirrors tools/labeling/label_config.xml)."""
    return (
        '<View>\n'
        '  <Image name="image" value="$image"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        '    <Label value="tweet_region" background="#1da1f2"/>\n'
        '  </RectangleLabels>\n'
        '</View>'
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
    reuse: bool,
    force_import: bool,
) -> dict[str, Any]:
    """Create or reuse the project, import seed tasks, and connect the ML backend.

    Idempotency: when ``reuse`` is set an existing same-titled project is reused
    instead of created; task import is skipped for a reused project unless
    ``force_import``; and an ML backend with the same URL is not added twice.

    Returns a summary dict with ``project_id``, ``created``, ``imported_count``
    and ``ml_backend_id`` (``None`` when not connected).
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

    return {
        "project_id": project_id,
        "created": created,
        "imported_count": imported_count,
        "ml_backend_id": ml_backend_id,
    }


def build_client(base_url: str, api_key: str) -> LabelStudio:
    """Construct a Label Studio SDK client (import is local so tests need no SDK)."""
    from label_studio_sdk import LabelStudio

    return LabelStudio(base_url=base_url, api_key=api_key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Label Studio base URL")
    parser.add_argument(
        "--api-key", default=None, help=f"API key (defaults to ${API_KEY_ENV})"
    )
    parser.add_argument("--title", default=DEFAULT_TITLE, help="project title")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path(DEFAULT_TASKS),
        help="tasks.json to import; skipped if the file is absent",
    )
    parser.add_argument(
        "--ml-backend-url", default=DEFAULT_ML_BACKEND_URL, help="ML backend URL"
    )
    parser.add_argument("--ml-title", default=DEFAULT_ML_TITLE, help="ML backend title")
    parser.add_argument(
        "--no-ml-backend",
        dest="connect_ml_backend",
        action="store_false",
        help="do not connect the ML backend",
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

    summary = setup_project(
        client,
        title=args.title,
        label_config=build_label_config(),
        tasks=tasks,
        ml_backend_url=args.ml_backend_url,
        ml_title=args.ml_title,
        is_interactive=args.is_interactive,
        connect_ml_backend=args.connect_ml_backend,
        reuse=args.reuse,
        force_import=args.force_import,
    )

    verb = "created" if summary["created"] else "reused"
    ml = (
        f"ml-backend #{summary['ml_backend_id']}"
        if summary["ml_backend_id"] is not None
        else "no ml-backend"
    )
    print(
        f"✔︎ {verb} project '{args.title}' (#{summary['project_id']}); "
        f"imported {summary['imported_count']} tasks; {ml}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Launch the API against a fuzzy-selected model.

The FastAPI app only enqueues; the *worker* loads the ScreenNet weights from
``settings.weights_path``. So ``serve`` picks a weights file (interactively via
fzf with ``--select``, otherwise the configured default), exports it as
``SCREENCROPNET_WEIGHTS_PATH`` so both the API and a co-launched worker read the
same model, and then boots uvicorn. ``get_settings`` is ``@lru_cache``d, so the
export is paired with a ``cache_clear`` to make the new path take effect.

The IO-free resolution helpers are unit-tested; the process launch is thin and
patched in tests.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from screencropnet_yolo.model_select import (
    SERVER_MODEL_EXTS,
    ModelSelector,
    discover_models,
    select_model,
)
from screencropnet_yolo.server.config import Settings, get_settings

WEIGHTS_ENV = "SCREENCROPNET_WEIGHTS_PATH"


def resolve_serve_weights(
    *, select: bool, settings: Settings, selector: ModelSelector | None = None
) -> Path:
    """Return the weights path to serve.

    With ``select``, fuzzy-pick a ``.pt``/``.onnx``/``.pth`` file across
    ``settings.model_search_roots``; otherwise return the configured
    ``settings.weights_path``. Raises when there is nothing to pick or the pick is
    cancelled (mirrors ``demo.resolve_model``).
    """
    if not select:
        return settings.weights_path

    seen: dict[Path, None] = {}
    for root in settings.model_search_roots:
        for path in discover_models(root, SERVER_MODEL_EXTS):
            seen[path] = None
    candidates = sorted(seen, key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        roots = ", ".join(str(r) for r in settings.model_search_roots)
        raise FileNotFoundError(f"No model files (.pt/.onnx/.pth) found under {roots}")

    chosen = select_model(candidates, selector=selector)
    if chosen is None:
        raise RuntimeError("Model selection cancelled")
    return chosen


def apply_weights_env(path: Path) -> None:
    """Export ``path`` as the weights env var and invalidate the settings cache."""
    os.environ[WEIGHTS_ENV] = str(path)
    get_settings.cache_clear()


def _launch_worker() -> subprocess.Popen[bytes]:
    """Spawn a detached worker in the current (weights-exported) environment."""
    return subprocess.Popen(["screencrop-worker"], env=os.environ.copy(), start_new_session=True)


def serve(
    *,
    select: bool = False,
    host: str | None = None,
    port: int | None = None,
    with_worker: bool = False,
    selector: ModelSelector | None = None,
    settings: Settings | None = None,
) -> None:
    """Resolve weights, export them, optionally launch a worker, then run uvicorn."""
    settings = settings or get_settings()
    weights = resolve_serve_weights(select=select, settings=settings, selector=selector)
    apply_weights_env(weights)

    if with_worker:
        _launch_worker()

    import uvicorn

    uvicorn.run(
        "screencropnet_yolo.server.api:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
    )

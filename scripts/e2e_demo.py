#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Live-stack end-to-end demo of the ingest/classify pipeline with the real model.

Orchestrates the whole thing on a developer machine: a Docker preflight, ``docker
compose up`` with a health-poll, Alembic migrations, background API + worker
(loading the real ScreenNet weights), then the CLI flow
(submit → status → twitter → export --dry-run), and a self-cleaning teardown.

Stdlib only so it runs under ``uv run``'s isolated PEP 723 environment; the actual
pipeline commands are invoked as ``uv run …`` subprocesses against the project env.

Lessons baked in from a prior Docker-down session: a Docker preflight with an
actionable message, a health-poll loop instead of a fixed sleep, weights/torch/port
checks before starting anything, and robust ``try/finally`` cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# Kept in sync with screencropnet_yolo.server.config.Settings defaults.
DEFAULT_WEIGHTS = Path("scratch/models/ScreenNetV1.pth")
RAW_DATASET_DIR = Path("scratch/datasets/twitter_screenshots_raw/train_images")
API_HOST = "127.0.0.1"
API_PORT = 8000
WORKER_METRICS_PORT = 8001
SERVICES = ("postgres", "rabbitmq")
DOCKER_DOWN_MSG = "✘ Docker daemon not running — start Docker Desktop (e.g. `open -a Docker`) and retry"


def _weights_path() -> Path:
    env = os.environ.get("SCREENCROPNET_WEIGHTS_PATH")
    return Path(env).expanduser() if env else DEFAULT_WEIGHTS


def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, text=True, **kwargs)  # type: ignore[call-overload]


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((API_HOST, port)) == 0


def preflight() -> None:
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        sys.exit(f"{DOCKER_DOWN_MSG}\n(exit 2)")
    if subprocess.run([sys.executable, "-c", "import torch"], capture_output=True).returncode != 0:
        # torch ships with ultralytics' main deps; flag a lean install early.
        if subprocess.run(["uv", "run", "python", "-c", "import torch"], capture_output=True).returncode != 0:
            sys.exit("✘ torch is not importable; install deps (`make install`) before the demo")
    weights = _weights_path()
    if not weights.exists():
        sys.exit(f"✘ weights not found at {weights}; run `make download-weights` first")
    busy = [p for p in (API_PORT, WORKER_METRICS_PORT) if _port_in_use(p)]
    if busy:
        sys.exit(f"✘ port(s) {busy} already in use; stop whatever is listening and retry")
    print("✔︎ preflight ok (docker, torch, weights, ports)")


def _container_id(service: str) -> str:
    out = subprocess.run(
        ["docker", "compose", "ps", "-q", service], capture_output=True, text=True
    )
    return out.stdout.strip()


def _health(container_id: str) -> str:
    out = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Health.Status}}", container_id],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def wait_for_healthy(timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        statuses = {svc: _health(_container_id(svc)) for svc in SERVICES}
        print(f"  health: {statuses}")
        if all(status == "healthy" for status in statuses.values()):
            print("✔︎ services healthy")
            return
        time.sleep(3.0)
    raise TimeoutError(f"services not healthy within {timeout:.0f}s: {SERVICES}")


def wait_for_healthz(timeout: float = 60.0) -> None:
    url = f"http://{API_HOST}:{API_PORT}/healthz"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    print("✔︎ API /healthz ok")
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"API /healthz not ready within {timeout:.0f}s")


def fetch_status(batch_id: str) -> dict[str, object]:
    url = f"http://{API_HOST}:{API_PORT}/status?batch_id={batch_id}"
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        return json.loads(resp.read())


def wait_for_batch(batch_id: str, expected: int, timeout: float = 180.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summary = fetch_status(batch_id)
        done = int(summary.get("done", 0))
        failed = int(summary.get("failed", 0))
        total = int(summary.get("total", 0))
        print(f"  batch {batch_id}: done={done} failed={failed} total={total}/{expected}")
        if total >= expected and done + failed >= total:
            return summary
        time.sleep(2.0)
    raise TimeoutError(f"batch {batch_id} did not finish within {timeout:.0f}s")


def stage_images(n: int) -> Path:
    if not RAW_DATASET_DIR.is_dir():
        sys.exit(f"✘ dataset dir not found: {RAW_DATASET_DIR}")
    images = sorted(RAW_DATASET_DIR.glob("*_twitter.PNG"))[:n]
    if not images:
        sys.exit(f"✘ no *_twitter.PNG screenshots under {RAW_DATASET_DIR}")
    staging = Path(tempfile.mkdtemp(prefix="screencrop_demo_"))
    for image in images:
        shutil.copy2(image, staging / image.name)
    print(f"✔︎ staged {len(images)} screenshots into {staging}")
    return staging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=8, help="number of screenshots to submit")
    parser.add_argument("--keep", action="store_true", help="leave docker services running on exit")
    args = parser.parse_args()

    preflight()
    staging = stage_images(args.images)
    batch_id = f"demo-{int(time.time())}"

    api_proc: subprocess.Popen[bytes] | None = None
    worker_proc: subprocess.Popen[bytes] | None = None
    try:
        _run(["docker", "compose", "up", "-d"], check=True)
        wait_for_healthy()
        _run(["uv", "run", "alembic", "upgrade", "head"], check=True)

        print("∆ starting API and worker (worker loads the real model)…")
        api_proc = subprocess.Popen(
            [
                "uv", "run", "uvicorn",
                "screencropnet_yolo.server.api:create_app",
                "--factory", "--host", API_HOST, "--port", str(API_PORT),
            ]
        )
        worker_proc = subprocess.Popen(["uv", "run", "screencrop-worker"])
        wait_for_healthz()

        _run(
            ["uv", "run", "screencrop-cli", "submit", str(staging), "--batch-id", batch_id],
            check=True,
        )
        summary = wait_for_batch(batch_id, expected=args.images)
        _run(["uv", "run", "screencrop-cli", "status", "--batch-id", batch_id], check=True)
        _run(["uv", "run", "screencrop-cli", "twitter", "--batch-id", batch_id], check=True)
        _run(
            ["uv", "run", "screencrop-cli", "export", "--batch-id", batch_id, "--dry-run"],
            check=True,
        )

        print(
            f"\n✔︎ demo complete: batch={batch_id} "
            f"done={summary.get('done')} twitter={summary.get('twitter_count')} "
            f"failed={summary.get('failed')}"
        )
        return 0
    finally:
        for name, proc in (("api", api_proc), ("worker", worker_proc)):
            if proc is not None and proc.poll() is None:
                print(f"∆ terminating {name} (pid {proc.pid})")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        shutil.rmtree(staging, ignore_errors=True)
        if args.keep:
            print("∆ --keep set; leaving docker services running")
        else:
            _run(["docker", "compose", "down"])


if __name__ == "__main__":
    raise SystemExit(main())

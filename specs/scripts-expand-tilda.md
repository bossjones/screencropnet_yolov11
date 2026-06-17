# Plan: Expand `~` and `$VARS` in script path arguments (TDD)

## Task Description

Standalone CLI scripts under `scripts/` accept filesystem paths via argparse but do not
normalize them. Passing a path such as
`~/Downloads/deduplicate_twitter_screenshots/Photos-3-001` only works because the
*shell* expands an unquoted `~` before the script runs. The moment the tilde is quoted
(`"~/Downloads/..."`), arrives from a Makefile/variable, or is written as `$HOME`, the
script receives a literal `~`/`$VAR` and treats it as a relative folder name — failing
or writing to the wrong place.

Make **every** CLI script expand both `~` (home) and environment variables (`$VAR`) on
user-supplied path arguments, **driven test-first** following the project's TDD
discipline (RED → GREEN → REFACTOR).

Task type: `chore` / robustness. Complexity: `medium` (same pattern across 5 scripts,
plus new tests).

## Objective

When complete, every path-typed argparse argument across `scripts/*.py` resolves `~`
and `$VAR` at parse time. Each behavior is locked by a test that was watched failing
first. The original failing invocation works even when the path is quoted or supplied
via an environment variable, with no behavior change for absolute or repo-relative paths.

## Problem Statement

`scripts/add_images_to_labeling.py` defines `--source-dir` / `--staging-dir` with
`type=Path` and never expands. The same gap exists in `ls_yolo_export_to_dataset.py`,
`pascal_csv_to_ls_tasks.py`, and `setup_ls_project.py` (which calls `.resolve()` — that
makes a path absolute but does **not** expand `~`). Only `download_screennet_weights.py`
expands (`.expanduser()` at lines 43, 103). Behavior is inconsistent and depends on
quoting/shell.

## Solution Approach

These are self-contained PEP 723 scripts run via `uv run`, so a shared import is
undesirable (it breaks portability). Add the same tiny, fully-typed helper to each
script and use it as a custom argparse `type`, replacing `type=Path`:

```python
def expanded_path(value: str) -> Path:
    """Resolve ``~`` and ``$VAR`` references in a user-supplied path argument."""
    return Path(os.path.expandvars(value)).expanduser()
```

`os.path.expandvars` + `Path.expanduser()` mirrors shell behavior and matches the
chosen scope (tilde **and** env vars). Applying it as `type=` covers `required=True`
and repeated args uniformly; `Path`-object defaults bypass `type=` and stay unchanged
(they are repo-relative and correct). Precedent already exists at
`scripts/download_screennet_weights.py:43,103` and
`src/screencropnet_yolo/server/config.py:46-49`.

### Why this is cleanly testable

Existing tests already import scripts as a package, e.g.
`from scripts.pascal_csv_to_ls_tasks import bbox_to_ls_value`
(`tests/test_pascal_to_ls_tasks.py:10`) and
`from scripts.ls_yolo_export_to_dataset import build_dataset`
(`tests/test_ls_yolo_export.py:11`). So `expanded_path` in each script is directly
importable and unit-testable with no new harness. All five scripts import only stdlib
(or already-installed deps like `yaml`/`httpx`) at module top level, so they import
cleanly in the pytest venv.

## Relevant Files

Source files to modify:

- `scripts/add_images_to_labeling.py` — needs `import os`; path args `--source-dir`
  (line 155), `--staging-dir` (line 161).
- `scripts/ls_yolo_export_to_dataset.py` — needs `import os`; `--export` (176), `--out` (177).
- `scripts/pascal_csv_to_ls_tasks.py` — needs `import os`; `--csv` (145),
  `--images-root` (148), `--out` (157).
- `scripts/setup_ls_project.py` — `import os` already present (37); `--tasks` (214),
  `--local-files-document-root` (228). Keep `.resolve()` at line 277 (runs *after*
  `expanded_path`).
- `scripts/download_screennet_weights.py` — `import os` already present; `--dest` (92).
  Switch to `type=expanded_path`, simplify line 103 `args.dest.expanduser()` → `args.dest`.
- `scripts/e2e_demo.py` — no path CLI args; no change (listed to confirm reviewed).

### New Files

- `tests/test_scripts_path_expansion.py` — the TDD test module. Two parametrized test
  groups covering all five scripts:
  1. **Behavior** — imports each script's `expanded_path` and asserts `~`/`$VAR`
     expansion, absolute pass-through, undefined-var pass-through, and `Path` return type.
  2. **Wiring guard** — reads each script's source text and asserts it defines
     `expanded_path`, references `type=expanded_path`, and contains no remaining
     `type=Path`.

Reference only (no edit): `src/screencropnet_yolo/server/config.py:46-49`,
`tests/test_pascal_to_ls_tasks.py`, `tests/test_ls_yolo_export.py` (import pattern).

## Implementation Phases

### Phase 1: Foundation — failing tests (RED)
Write `tests/test_scripts_path_expansion.py` with both parametrized groups and watch
every case fail (import error / `type=Path` still present) before touching the scripts.

### Phase 2: Core Implementation (GREEN)
Add `expanded_path` and swap `type=Path → type=expanded_path` in each script, one script
at a time, re-running its parametrized cases to green.

### Phase 3: Integration & Polish (REFACTOR + verify)
Simplify the now-redundant `.expanduser()` in `download_screennet_weights.py`, run the
full suite plus `make lint`, and do the real-path `--dry-run` smoke checks.

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom. Follow RED → GREEN → REFACTOR;
never write production code before a test for it has been watched failing.

### 1. RED — write the behavior tests
- Create `tests/test_scripts_path_expansion.py`.
- Parametrize over the five `expanded_path` callables, e.g.:
  ```python
  from __future__ import annotations

  from pathlib import Path

  import pytest

  from scripts.add_images_to_labeling import expanded_path as add_images_expanded
  from scripts.download_screennet_weights import expanded_path as download_expanded
  from scripts.ls_yolo_export_to_dataset import expanded_path as ls_yolo_expanded
  from scripts.pascal_csv_to_ls_tasks import expanded_path as pascal_expanded
  from scripts.setup_ls_project import expanded_path as setup_expanded

  EXPANDERS = pytest.mark.parametrize(
      "expand",
      [add_images_expanded, download_expanded, ls_yolo_expanded, pascal_expanded, setup_expanded],
  )

  @EXPANDERS
  def test_expands_bare_tilde(expand):
      assert expand("~") == Path.home()

  @EXPANDERS
  def test_expands_tilde_prefix(expand):
      assert expand("~/Downloads/x") == Path.home() / "Downloads" / "x"

  @EXPANDERS
  def test_expands_env_var(expand, monkeypatch):
      monkeypatch.setenv("SCN_TEST_DIR", "/tmp/scn-test")
      assert expand("$SCN_TEST_DIR/a") == Path("/tmp/scn-test/a")

  @EXPANDERS
  def test_combines_var_and_tilde(expand, monkeypatch):
      monkeypatch.setenv("SCN_SUB", "Downloads")
      assert expand("~/$SCN_SUB/img") == Path.home() / "Downloads" / "img"

  @EXPANDERS
  def test_absolute_path_unchanged(expand):
      assert expand("/var/data/x") == Path("/var/data/x")

  @EXPANDERS
  def test_unknown_var_left_literal(expand, monkeypatch):
      monkeypatch.delenv("SCN_UNDEFINED", raising=False)
      assert expand("$SCN_UNDEFINED/x") == Path("$SCN_UNDEFINED/x")

  @EXPANDERS
  def test_returns_path(expand):
      assert isinstance(expand("~"), Path)
  ```
- Run it; confirm it fails with `ImportError` (no `expanded_path` yet) — this is the
  expected RED. Do not proceed until you've seen it fail for that reason.

### 2. RED — write the wiring-guard test
- In the same file add a source-text guard parametrized over the five script paths:
  ```python
  SCRIPTS = [
      "add_images_to_labeling.py",
      "download_screennet_weights.py",
      "ls_yolo_export_to_dataset.py",
      "pascal_csv_to_ls_tasks.py",
      "setup_ls_project.py",
  ]

  @pytest.mark.parametrize("name", SCRIPTS)
  def test_script_wires_expanded_path(name):
      src = (Path(__file__).parent.parent / "scripts" / name).read_text()
      assert "def expanded_path" in src
      assert "type=expanded_path" in src
      assert "type=Path" not in src  # all path args migrated
  ```
- Run it; confirm it fails (scripts still use `type=Path`). Expected RED.

### 3. GREEN — `add_images_to_labeling.py`
- Add `import os`; define `expanded_path` near the top-level helpers (before
  `detect_next_index`).
- Replace `type=Path` → `type=expanded_path` on `--source-dir` and `--staging-dir`.
- Re-run the `add_images_expanded` cases and this script's guard case → green.

### 4. GREEN — `pascal_csv_to_ls_tasks.py`
- Add `import os`; define `expanded_path`.
- Replace `type=Path` → `type=expanded_path` on `--csv`, `--images-root`, `--out`. Re-run → green.

### 5. GREEN — `ls_yolo_export_to_dataset.py`
- Add `import os`; define `expanded_path`.
- Replace `type=Path` → `type=expanded_path` on `--export`, `--out`. Re-run → green.

### 6. GREEN — `setup_ls_project.py`
- `import os` already present; define `expanded_path`.
- Replace `type=Path` → `type=expanded_path` on `--tasks`, `--local-files-document-root`.
- Leave line 277 `.resolve()` unchanged. Re-run → green.

### 7. GREEN + REFACTOR — `download_screennet_weights.py`
- Define `expanded_path`; set `--dest` to `type=expanded_path`.
- Simplify line 103 `dest: Path = args.dest.expanduser()` → `dest: Path = args.dest`
  (expansion now happens at parse time). Optionally update `_default_dest()` to
  `Path(os.path.expandvars(env)).expanduser()` for parity. Re-run → green.

### 8. Validate everything
- Run the full validation command set below; confirm green tests, clean lint, and the
  real-path `--dry-run` smoke checks expand correctly.

## Testing Strategy

- **Unit (TDD core):** parametrized `expanded_path` behavior tests across all five
  scripts in `tests/test_scripts_path_expansion.py`. Each case is watched failing
  (RED) before its helper exists. Env vars set via the `monkeypatch` builtin fixture
  (idiomatic for env mutation; not `unittest.mock`, so it respects the project's
  mocking rule). Edge cases: bare `~`, `~/`-prefix, `$VAR`, combined `~`+`$VAR`,
  absolute pass-through, undefined-var literal pass-through, `Path` return type.
- **Wiring guard:** source-text assertions ensure each script actually adopts
  `type=expanded_path` and drops `type=Path` — catches a helper that exists but isn't
  wired, and prevents regressions when new path args are added later.
- **Smoke (manual e2e):** `--dry-run` runs with a quoted tilde and a `$VAR` path to
  confirm end-to-end argparse wiring on the flagship script.
- No production code is written before its corresponding test has failed.

## Acceptance Criteria

- `tests/test_scripts_path_expansion.py` exists; every behavior + guard case was seen
  RED before implementation and is now green.
- Every path-typed argparse argument in `scripts/*.py` uses `type=expanded_path`; no
  bare `type=Path` remains on a user path argument.
- A quoted `~`-path and a `$VAR`-path each resolve to the real location in a `--dry-run`
  of `add_images_to_labeling.py`.
- `download_screennet_weights.py` no longer double-expands and still resolves `~`/env paths.
- `make lint` and `make test` pass clean.

## Validation Commands

Execute these to validate the task is complete:

- `uv run pytest tests/test_scripts_path_expansion.py -v` — all behavior + guard cases green.
- `uv run pytest -q` — full suite still green (no regressions in existing script tests).
- `grep -rn "type=Path" scripts/` — returns no matches.
- `uv run scripts/add_images_to_labeling.py --source-dir '~/Downloads/deduplicate_twitter_screenshots/Photos-3-001' --dry-run --verbose` — quoted tilde resolves (no literal `~` in output).
- `HOME_TEST="$HOME/Downloads" uv run scripts/add_images_to_labeling.py --source-dir '$HOME_TEST/deduplicate_twitter_screenshots/Photos-3-001' --dry-run --verbose` — env var expands.
- `make lint` — codespell + ruff + basedpyright clean (typed `expanded_path` passes).
- `make test` — pytest suite green with coverage.

## Notes

- No new dependencies; `os` and `pathlib` are stdlib, so PEP 723 `dependencies` blocks
  are unaffected.
- The helper is intentionally duplicated per script: these are independent `uv run`
  scripts with isolated environments, and the parametrized tests verify each copy
  behaves identically — duplication without a shared import, but not without a test.
- A shared `src/screencropnet_yolo` helper was considered and rejected: PEP 723 scripts
  run in isolated envs and would not have the package importable without coupling them
  to the build, which `.claude/rules/python-scripts.md` discourages.
- `os.path.expandvars` leaves undefined variables untouched (`$NOPE` stays `$NOPE`);
  this is asserted explicitly so the behavior is intentional, not incidental.

"""Minimal pytest entrypoint for Scalene to profile.

Scalene 2.3's ``run`` subcommand profiles a program *file*, not a ``-m module``.
This file is that program: it forwards its argv to ``pytest.main`` so Scalene
profiles a real pytest run. Args come after Scalene's ``---`` separator.
"""

from __future__ import annotations

import sys

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))

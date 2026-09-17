"""Portable local output locations for Varo V2.

Every local artifact root is resolved at call time so the project keeps working
on another machine.  ``VARO_OUTPUT_ROOT`` overrides the default project-relative
``outputs/`` directory; no absolute user path is ever hard coded.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT_ENV = "VARO_OUTPUT_ROOT"


def output_root() -> Path:
    """Return the local output root used by every V2 artifact writer."""
    configured = os.environ.get(OUTPUT_ROOT_ENV)
    if configured and str(configured).strip():
        return Path(str(configured).strip()).expanduser()
    return PROJECT_ROOT / "outputs"


def dqn_output_dir() -> Path:
    return output_root() / "dqn"


def simulation_history_dir() -> Path:
    return output_root() / "simulation_history"

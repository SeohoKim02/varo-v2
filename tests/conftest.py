"""Keep test fixture training/history out of the user's local runtime artifacts.

Configure before test collection, because DQN storage paths bind at import time.
The T5 runner's explicit isolated output root is reused without changing it.
No test assertion, model parameter, or production behavior is changed.
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory

_scratch = None


def pytest_configure(config):
    global _scratch
    configured = os.environ.get("VARO_OUTPUT_ROOT")
    production = Path(__file__).resolve().parents[1] / "outputs"
    configured_path = Path(configured).resolve() if configured else None
    if not configured or configured_path == production.resolve() or configured_path == (production / "dqn").resolve():
        _scratch = TemporaryDirectory(prefix="varo-pytest-runtime-")
        os.environ["VARO_OUTPUT_ROOT"] = _scratch.name


def pytest_unconfigure(config):
    global _scratch
    if _scratch is not None:
        os.environ.pop("VARO_OUTPUT_ROOT", None)
        _scratch.cleanup()
        _scratch = None

"""Keep test fixture training/history out of the user's local runtime artifacts.

Configure before test collection, because DQN storage paths bind at import time.
The T5 runner's explicit isolated output root is reused without changing it.
No test assertion, model parameter, or production behavior is changed.
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory

_scratch = None
_previous_output_root = None


def pytest_configure(config):
    global _scratch, _previous_output_root
    configured = os.environ.get("VARO_OUTPUT_ROOT")
    production = Path(__file__).resolve().parents[1] / "outputs"
    configured_path = Path(configured.strip()).expanduser().resolve() if configured and configured.strip() else None
    if configured_path is None or configured_path.is_relative_to(production.resolve()):
        _previous_output_root = configured
        _scratch = TemporaryDirectory(prefix="varo-pytest-runtime-")
        os.environ["VARO_OUTPUT_ROOT"] = _scratch.name


def pytest_unconfigure(config):
    global _scratch, _previous_output_root
    if _scratch is not None:
        if _previous_output_root is None:
            os.environ.pop("VARO_OUTPUT_ROOT", None)
        else:
            os.environ["VARO_OUTPUT_ROOT"] = _previous_output_root
        _scratch.cleanup()
        _scratch = None
        _previous_output_root = None

"""Keep test fixture training/history out of the user's local runtime artifacts.

Configure before test collection, because DQN storage paths bind at import time.
The T5 runner's explicit isolated output root is reused without changing it.
No test assertion, model parameter, or production behavior is changed.
"""
import os
import sys
import time
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

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


def _serial_highs_call(original, *args):
    options = dict(args[-1])
    if options.get("threads", 0) not in (0, 1):
        raise RuntimeError("Explicit parallel HiGHS test requires a separate process")
    # Only native scheduling changes. Objective, bounds, constraints, gap and
    # time budget pass through unchanged, and the real solver still runs.
    options["threads"] = 1
    return original(*args[:-1], options)


def _finish_apptest_worker(runner, deadline):
    worker = runner._script_thread
    if worker is not None and worker.ident is not None:
        worker.join(max(0.0, deadline - time.monotonic()))
        if worker.is_alive():
            runner.request_stop()
            raise RuntimeError("AppTest worker teardown exceeded its original deadline")


@pytest.fixture(scope="session", autouse=True)
def _apptest_worker_lifecycle():
    """Bound serial AppTest teardown and avoid the Windows native-pool fault.

    AppTest creates a new script thread for every rerun, unlike a live Streamlit
    session. Windows / SciPy 1.17.1 / HiGHS 1.12 reproduced access violations
    and Thread.start waits with those short-lived callers. Serial native solves
    avoid the failing pool lifecycle; Python UI threads still run normally.
    This fixture never changes the application runtime or mocks a solver result.
    """
    from streamlit.testing.v1.local_script_runner import LocalScriptRunner

    original_run = LocalScriptRunner.run
    milp_module = None
    if sys.platform == "win32":
        import scipy.optimize._milp as milp_module
        original_highs = milp_module._highs_wrapper
        @wraps(original_highs)
        def serial_highs(*args):
            return _serial_highs_call(original_highs, *args)
        milp_module._highs_wrapper = serial_highs

    @wraps(original_run)
    def run(runner, widget_state=None, query_params=None, timeout=3, page_hash=""):
        deadline = time.monotonic() + timeout
        try:
            return original_run(runner, widget_state, query_params, timeout, page_hash)
        finally:
            # SHUTDOWN is emitted before actual thread/native teardown. Join
            # before AppTest resets shared Runtime state or starts the next run.
            # Keep the original deadline; no retry, sleep or assertion bypass.
            _finish_apptest_worker(runner, deadline)

    LocalScriptRunner.run = run
    try:
        yield
    finally:
        LocalScriptRunner.run = original_run
        if milp_module is not None:
            milp_module._highs_wrapper = original_highs

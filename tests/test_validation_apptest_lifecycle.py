"""Regression coverage for native teardown, without mocking solver results."""
import threading
from types import SimpleNamespace

import pytest
from streamlit.runtime import Runtime
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.local_script_runner import LocalScriptRunner

import conftest as lifecycle


def test_apptest_waits_for_actual_worker_exit_before_resetting_runtime(monkeypatch):
    original = LocalScriptRunner._run_script_thread
    shutdown, released = threading.Event(), threading.Event()
    observations, workers = [], []

    def delayed(runner):
        workers.append(threading.current_thread())
        original(runner)
        shutdown.set()
        assert released.wait(5)

    def release():
        assert shutdown.wait(5)
        observations.append(Runtime._instance is not None)
        released.set()

    monkeypatch.setattr(LocalScriptRunner, "_run_script_thread", delayed)
    timer = threading.Timer(0.15, release)
    timer.start()
    try:
        at = AppTest.from_string('import streamlit as st\nst.write("worker finished")').run()
        assert not at.exception
        assert at.markdown[0].value == "worker finished"
        assert observations == [True]
        assert not workers[0].is_alive()
        assert Runtime._instance is None
    finally:
        released.set()
        timer.join(5)
        for worker in workers:
            worker.join(5)


def test_repeated_real_milp_on_new_apptest_threads_keeps_exact_result():
    at = AppTest.from_string('''
import streamlit as st
from scipy.optimize import milp, Bounds, LinearConstraint
r = milp([-1., -2.], integrality=[1, 1], bounds=Bounds([0, 0], [3, 3]),
         constraints=LinearConstraint([[1., 1.]], 0, 4))
assert r.success and r.fun == -7
st.metric("검산 결과", int(-r.fun))
''', default_timeout=10)
    for _ in range(60):
        at.run()
        assert not at.exception
        assert at.metric[0].value == "7"


def test_cleanup_uses_existing_deadline_not_a_fresh_timeout(monkeypatch):
    waits = []
    worker = SimpleNamespace(ident=1, join=waits.append, is_alive=lambda: False)
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: 12.25)
    lifecycle._finish_apptest_worker(SimpleNamespace(_script_thread=worker), 13)
    assert waits == [0.75]


def test_overdue_worker_is_stopped_and_fails_without_retry(monkeypatch):
    waits, stops = [], []
    worker = SimpleNamespace(ident=1, join=waits.append, is_alive=lambda: True)
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: 14)
    runner = SimpleNamespace(_script_thread=worker, request_stop=lambda: stops.append(True))
    with pytest.raises(RuntimeError, match="original deadline"):
        lifecycle._finish_apptest_worker(runner, 13)
    assert waits == [0.0] and stops == [True]


def test_serial_solver_preserves_problem_options_and_real_result():
    problem = [object() for _ in range(9)]
    options = {"time_limit": 7, "mip_rel_gap": 0, "presolve": True}
    seen, result = [], object()
    def original(*args):
        seen.append(args)
        return result
    assert lifecycle._serial_highs_call(original, *problem, options) is result
    assert all(a is b for a, b in zip(seen[0][:-1], problem))
    assert seen[0][-1] == {**options, "threads": 1}
    assert options == {"time_limit": 7, "mip_rel_gap": 0, "presolve": True}


def test_solver_errors_are_not_swallowed():
    def original(*args):
        raise ValueError("solver failure")
    with pytest.raises(ValueError, match="solver failure"):
        lifecycle._serial_highs_call(original, {})


def test_explicit_parallel_test_is_not_silently_changed():
    calls = []
    with pytest.raises(RuntimeError, match="separate process"):
        lifecycle._serial_highs_call(lambda *a: calls.append(a), {"threads": 2})
    assert not calls


def test_app_exceptions_remain_visible_to_existing_assertions(caplog):
    at = AppTest.from_string('raise ValueError("visible test failure")').run()
    assert len(at.exception) == 1
    assert "error" in at.exception[0].message
    assert "ValueError: visible test failure" in caplog.text

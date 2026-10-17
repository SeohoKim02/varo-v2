from copy import deepcopy
from dataclasses import replace
import pickle
import threading
import pytest

from services.candidate_input import prepare_input, read_uploads
from services.candidate_input_examples import csv_test_files, mixed_test_metadata
from services.candidate_session import input_signature, request_key, cached, begin_request, commit_result
import services.candidate_session as session


def prepared():
    return prepare_input(read_uploads(csv_test_files()), mixed_test_metadata())


def test_file_content_and_all_metadata_invalidate_input():
    files = csv_test_files(); meta = mixed_test_metadata()
    baseline = input_signature(files, meta)
    changed = dict(files); name = next(iter(files)); changed[name] += b"\n"
    assert input_signature(changed, meta) != baseline
    for key, value in meta.items():
        assert input_signature(files, {**meta, key: str(value)+"changed"}) != baseline
    assert input_signature(dict(reversed(list(files.items()))), meta) == baseline


def test_cache_is_session_local_bounded_and_copy_safe():
    first, second = {}, {}; calls = []
    def calculate():
        calls.append(1); return {"quantity": [4]}
    a, cold = cached(first, "test", "same", calculate)
    a["quantity"].append(9)
    b, warm = cached(first, "test", "same", calculate)
    c, other = cached(second, "test", "same", calculate)
    assert b == c == {"quantity": [4]}
    assert not cold["cache_hit"] and warm["cache_hit"] and not other["cache_hit"]
    assert len(calls) == 2 and first["_candidate_cache_owner"] != second["_candidate_cache_owner"]
    for k in range(5): cached(first, "test", str(k), calculate, limit=2)
    assert len(first["_candidate_cache_test"]) == 2


def test_late_response_cannot_overwrite_newer_result():
    state = {}; old = begin_request(state, "old"); new = begin_request(state, "new")
    assert commit_result(state, new, {"value": 8}, "new_problem", input_key="new")
    assert not commit_result(state, old, {"value": 99}, "old_problem", input_key="old")
    assert state["_candidate_uploaded_result"] == {"value": 8}


def test_policy_observations_scope_and_budget_invalidate_plan():
    p = prepared()
    options = dict(policy="OPEN_LOOP", decision_at="2026-10-16T00:00:00+09:00", observations=[], risk="expected")
    key = request_key(p, **options)
    for field, value in [("policy", "ROLLING_HORIZON"), ("decision_at", "2026-10-17T00:00:00+09:00"),
                         ("observations", [{"event_id": "new", "quantity": 1}]), ("risk", "worst_case"), ("time_limit", 10)]:
        assert request_key(p, **{**options, field:value}) != key
    other = deepcopy(p); other.problem = replace(p.problem, lineage={**p.problem.lineage, "company": "different"})
    assert request_key(other, **options) != key


def test_optional_scenario_signals_reuse_existing_sales_table():
    bundle = read_uploads(csv_test_files())
    bundle.tables["economic_sales_policy"]["observation_period"] = "0"
    bundle.tables["economic_sales_policy"]["observation_signal"] = "보통 수요"
    p = prepare_input(bundle, mixed_test_metadata())
    assert p.ready, p.issues
    assert p.problem.scenarios[0].observations == {0:"보통 수요"}
    bundle.tables["economic_sales_policy"].loc[0, "observation_signal"] = "상충 수요"
    assert not prepare_input(bundle, mixed_test_metadata()).ready


def test_default_run_preserves_comparison_optional_fast_path(monkeypatch):
    import services.candidate_economics_policies as policies
    from services.candidate_input import run_prepared
    calls = []
    monkeypatch.setattr(policies, "fair_baselines", lambda p: calls.append(p) or {"status":"COMPLETED"})
    p = prepared()
    assert "alternative_comparison" not in run_prepared(p, include_comparison=False)
    assert not calls
    assert "alternative_comparison" in run_prepared(p)
    assert len(calls) == 1


def test_adaptive_entry_cannot_bypass_snapshot_pending_flow_gate():
    bundle = read_uploads(csv_test_files())
    bundle.tables["inventory_snapshot"]["reserved_qty"] = "2"
    p = prepare_input(bundle, mixed_test_metadata())
    assert not p.ready and p.problem is None
    assert any("확정 입출고 합계" in item["message_ko"] for item in p.issues)


def test_dependency_and_solver_versions_invalidate_prepared_input_and_plan(monkeypatch):
    files, meta, p = csv_test_files(), mixed_test_metadata(), prepared()
    options = dict(policy="OPEN_LOOP", decision_at="2026-10-16T00:00:00+09:00", observations=[], risk="expected")
    dependency = {"sources": {"policy": "before"}, "runtime_versions": {"scipy": "1"}}
    monkeypatch.setattr(session, "model_fingerprint", lambda: deepcopy(dependency))
    initial = input_signature(files, meta), request_key(p, **options)
    dependency["sources"]["policy"] = "after"
    assert input_signature(files, meta) != initial[0]
    assert request_key(p, **options) != initial[1]
    source_changed = input_signature(files, meta), request_key(p, **options)
    dependency["runtime_versions"]["scipy"] = "2"
    assert input_signature(files, meta) != source_changed[0]
    assert request_key(p, **options) != source_changed[1]


def test_fingerprint_covers_actual_processing_and_solver_dependencies():
    fingerprint = session.model_fingerprint()
    assert {"candidate_input.py", "candidate_adaptive.py", "candidate_economics.py",
            "candidate_economics_extended.py", "candidate_economics_policies.py",
            "operator_intake_candidate.py", "operating_snapshot.py", "canonical_schema.py",
            "target_need_contract.py", "operational_evidence.py", "seller_loss_engine.py",
            "shared_feasibility_selection.py", "optimality_gap_service.py"} <= fingerprint["sources"].keys()
    assert all(len(value) == 64 for value in fingerprint["sources"].values())
    assert {"python", "scipy", "numpy", "pandas", "openpyxl"} == fingerprint["runtime_versions"].keys()
    assert fingerprint["model_artifact"] == "NONE_ON_THIS_ECONOMIC_PATH"
    assert "HiGHS" in fingerprint["solver"]


def test_upload_name_multiplicity_invalidates_without_changing_order_semantics():
    files, meta = csv_test_files(), mixed_test_metadata()
    names = list(files)
    baseline = input_signature(files, meta, upload_names=names)
    assert input_signature(files, meta, upload_names=list(reversed(names))) == baseline
    assert input_signature(files, meta, upload_names=names + [names[0]]) != baseline


@pytest.mark.parametrize("field,value", [
    ("policy", "ROLLING_HORIZON"), ("input_signature", "changed-input"),
    ("problem_signature", "changed-problem"), ("frozen_open_loop", [{"quantity": 2}]),
    ("planned_actions", [{"quantity": 3}]), ("replan_count", 2), ("observation_update_count", 3),
])
def test_every_consumed_prior_context_field_invalidates_plan(field, value):
    p = prepared()
    options = dict(policy="ROLLING_HORIZON", decision_at="2026-10-16T00:00:00+09:00", observations=[], risk="expected")
    previous = dict(policy="OPEN_LOOP", input_signature="input", problem_signature="problem",
                    frozen_open_loop=[], planned_actions=[], replan_count=0, observation_update_count=0)
    before = request_key(p, previous=previous, **options)
    assert request_key(p, previous={**previous, field: value}, **options) != before
    assert request_key(p, previous={**previous, "display_only": "ignored"}, **options) == before


def test_origin_and_missing_versus_null_prior_plan_are_distinct():
    p = prepared()
    options = dict(policy="OPEN_LOOP", decision_at="2026-10-17T00:00:00+09:00", observations=[], risk="expected")
    previous = {"policy": "OPEN_LOOP", "planned_actions": []}
    first = request_key(p, previous=previous, origin_at="2026-10-16T00:00:00+09:00", **options)
    assert request_key(p, previous=previous, origin_at="2026-10-17T00:00:00+09:00", **options) != first
    assert request_key(p, previous={**previous, "frozen_open_loop": None},
                       origin_at="2026-10-16T00:00:00+09:00", **options) != first


def _thread_call(call, results, errors):
    def target():
        try:
            results.append(call())
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def _join_threads(threads, errors):
    for thread in threads:
        thread.join(3)
        assert not thread.is_alive(), "session operation deadlocked"
    assert not errors, errors


def test_new_request_cannot_interleave_with_partial_result_commit():
    old_write_started, release_old, new_attempted, new_done = (threading.Event() for _ in range(4))
    class PausingState(dict):
        def __setitem__(self, key, value):
            if key == "_candidate_uploaded_result" and value == {"value": "old"}:
                old_write_started.set()
                assert release_old.wait(3)
            super().__setitem__(key, value)
    state, results, errors = PausingState(), [], []
    old = begin_request(state, "old")
    old_thread = _thread_call(lambda: commit_result(state, old, {"value": "old"}, "old_problem", input_key="old"), results, errors)
    assert old_write_started.wait(3)
    def newer():
        new_attempted.set()
        token = begin_request(state, "new")
        outcome = commit_result(state, token, {"value": "new"}, "new_problem", input_key="new", envelope={"new": True})
        new_done.set()
        return outcome
    new_thread = _thread_call(newer, results, errors)
    try:
        assert new_attempted.wait(3)
        assert not new_done.wait(.05), "new request entered a partially committed result"
    finally:
        release_old.set()
    _join_threads([old_thread, new_thread], errors)
    assert results == [True, True]
    assert state["_candidate_active_request"] == state["_candidate_result_request"] == "new"
    assert state["_candidate_uploaded_result"] == {"value": "new"}
    assert state["_candidate_uploaded_problem"] == "new_problem"
    assert state["_candidate_result_input_key"] == "new"
    assert state["_candidate_adaptive_result"] == {"new": True}


def test_slow_compute_does_not_block_new_request_and_cannot_overwrite_it():
    entered, release = threading.Event(), threading.Event()
    state, results, errors = {}, [], []
    old = begin_request(state, "old")
    def slow():
        entered.set()
        assert release.wait(3)
        return {"value": "old"}
    def older():
        result, _ = cached(state, "plan", "old", slow)
        return commit_result(state, old, result, "old_problem", input_key="old")
    thread = _thread_call(older, results, errors)
    try:
        assert entered.wait(3)
        new = begin_request(state, "new")
        assert commit_result(state, new, {"value": "new"}, "new_problem", input_key="new")
    finally:
        release.set()
    _join_threads([thread], errors)
    assert results == [False]
    assert state["_candidate_uploaded_result"] == {"value": "new"}
    assert state["_candidate_result_request"] == "new"


def test_same_session_same_key_runs_once_and_returns_independent_copies(monkeypatch):
    entered, release, waiter_entered = (threading.Event() for _ in range(3))
    state, results, errors, calls = {}, [], [], []
    def compute():
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return {"items": [1]}
    first = _thread_call(lambda: cached(state, "plan", "same", compute), results, errors)
    assert entered.wait(3)
    # The producer remains in compute, so its control is live and pending is real.
    _, control = session._control(state)
    pending = next(iter(control.pending.values()))
    original_wait = pending.wait
    def wait(timeout=None):
        waiter_entered.set()
        return original_wait(timeout)
    monkeypatch.setattr(pending, "wait", wait)
    second = _thread_call(lambda: cached(state, "plan", "same", compute), results, errors)
    try:
        assert waiter_entered.wait(3)
    finally:
        release.set()
    _join_threads([first, second], errors)
    assert len(calls) == 1
    assert sorted(x[1]["cache_hit"] for x in results) == [False, True]
    assert any(x[1]["waited_for_same_request"] for x in results)
    results[0][0]["items"].append(2)
    assert results[1][0] == {"items": [1]}
    assert cached(state, "plan", "same", compute)[0] == {"items": [1]}


def test_failed_compute_is_propagated_and_does_not_cache_success():
    state = {}
    def fail():
        raise ValueError("expected failure")
    with pytest.raises(ValueError, match="expected failure"):
        cached(state, "plan", "same", fail)
    result, details = cached(state, "plan", "same", lambda: {"valid": True})
    assert result == {"valid": True} and not details["cache_hit"]
    assert not session._control(state)[1].pending


def test_failed_producer_releases_existing_waiter_to_retry(monkeypatch):
    entered, release, waiter_entered = (threading.Event() for _ in range(3))
    state, results, errors, calls = {}, [], [], []
    def compute():
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
            raise ValueError("producer failed")
        return {"valid": True}
    first = _thread_call(lambda: cached(state, "plan", "same", compute), results, errors)
    assert entered.wait(3)
    _, control = session._control(state)
    pending = next(iter(control.pending.values()))
    original_wait = pending.wait
    def wait(timeout=None):
        waiter_entered.set()
        return original_wait(timeout)
    monkeypatch.setattr(pending, "wait", wait)
    second = _thread_call(lambda: cached(state, "plan", "same", compute), results, errors)
    try:
        assert waiter_entered.wait(3)
    finally:
        release.set()
    for thread in (first, second):
        thread.join(3)
        assert not thread.is_alive(), "failed producer left its waiter blocked"
    assert len(errors) == 1 and isinstance(errors[0], ValueError)
    assert str(errors[0]) == "producer failed"
    assert len(calls) == 2 and len(results) == 1
    assert results[0][0] == {"valid": True}
    assert not results[0][1]["cache_hit"] and results[0][1]["waited_for_same_request"]


def test_lock_is_not_stored_in_session_state_and_other_users_do_not_wait():
    entered, release = threading.Event(), threading.Event()
    first, second, results, errors = {}, {}, [], []
    def slow():
        entered.set()
        assert release.wait(3)
        return {"owner": "first"}
    thread = _thread_call(lambda: cached(first, "plan", "same", slow), results, errors)
    try:
        assert entered.wait(3)
        other, detail = cached(second, "plan", "same", lambda: {"owner": "second"})
        assert other == {"owner": "second"} and not detail["cache_hit"]
        assert pickle.loads(pickle.dumps(second)) == second
    finally:
        release.set()
    _join_threads([thread], errors)
    assert first["_candidate_cache_owner"] != second["_candidate_cache_owner"]


@pytest.mark.parametrize("limit", [0, -1, 1.5])
def test_invalid_cache_bound_is_not_silently_accepted(limit):
    with pytest.raises(ValueError, match="positive"):
        cached({}, "plan", "key", lambda: {}, limit=limit)

"""Session-local, content-addressed candidate work. No shared data or model cache."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import threading
from time import perf_counter
from typing import Any, Callable, MutableMapping
import uuid
from weakref import WeakValueDictionary

VERSION = "candidate-session-0.1.3"

# Only synchronization objects live here. All input/result data remains in its
# own Streamlit session, and weak references avoid retaining disconnected users.
_CONTROLS = WeakValueDictionary()
_CONTROL_CREATION = threading.RLock()


class _SessionControl:
    def __init__(self):
        self.lock = threading.RLock()
        self.pending = {}


def _control(state):
    with _CONTROL_CREATION:
        owner = state.setdefault("_candidate_cache_owner", uuid.uuid4().hex)
        control = _CONTROLS.get(owner)
        if control is None:
            control = _SessionControl()
            _CONTROLS[owner] = control
        return owner, control


def model_fingerprint() -> dict:
    """Content/version identity of the actual input and finite economic path.

    No DQN checkpoint is used on this path. Model files are neither read nor
    admitted merely because they exist. Changed contents, not mtimes, invalidate.
    """
    names = ("candidate_input.py", "candidate_economics.py", "candidate_economics_extended.py",
             "candidate_economics_policies.py", "candidate_adaptive.py", "candidate_session.py",
             "operating_snapshot.py", "operator_intake_candidate.py", "canonical_schema.py",
             "target_need_contract.py", "operational_evidence.py", "seller_loss_engine.py",
             "shared_feasibility_selection.py", "optimality_gap_service.py")
    versions = {"python": platform.python_version()}
    for name in ("scipy", "numpy", "pandas", "openpyxl"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return {"sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names},
            "runtime_versions": versions, "solver": "scipy.optimize.milp / bundled HiGHS",
            "solver_threads": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "model_artifact": "NONE_ON_THIS_ECONOMIC_PATH"}


def signature(value: Any) -> str:
    def default(item):
        if is_dataclass(item):
            return asdict(item)
        if hasattr(item, "isoformat"):
            return item.isoformat()
        raise TypeError(f"Unsupported signature value: {type(item).__name__}")
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=default, allow_nan=False).encode()).hexdigest()


def input_signature(files: dict[str, bytes], metadata: dict, *, upload_names=None) -> str:
    from services.candidate_input import VERSION as input_version
    return signature({"files": {name: hashlib.sha256(value).hexdigest()
        for name, value in files.items()}, "metadata": metadata,
        "input_version": input_version, "session_version": VERSION,
        "upload_names": sorted(upload_names if upload_names is not None else files),
        "processing_dependency": model_fingerprint()})


def previous_context(previous):
    """Every prior-envelope field consumed by planning or change explanations."""
    if not previous:
        return None
    # Absence differs from an explicit None: the planner uses dict.get(...,
    # fallback) for a missing frozen plan, but rejects an explicitly null one.
    return {name: previous[name] for name in ("policy", "input_signature", "problem_signature",
            "frozen_open_loop", "planned_actions", "replan_count", "observation_update_count")
            if name in previous}


def request_key(prepared, *, policy, decision_at, observations, risk, time_limit=30, previous=None, origin_at=None):
    from services.candidate_economics import _canonical
    return signature({"files": prepared.file_hashes, "snapshot": prepared.snapshot,
        "problem": _canonical(prepared.problem), "data_class": prepared.data_class,
        "policy": policy, "decision_at": decision_at, "observations": observations,
        "risk": risk, "time_limit": time_limit, "session_version": VERSION, "origin_at": origin_at,
        "implementation": model_fingerprint(), "previous": previous_context(previous)})


def cached(state: MutableMapping, category: str, key: str, compute: Callable, *, limit=3):
    """Session-local LRU with single-flight computation and copy-safe results.

    The solver runs outside the session lock so a newer request can invalidate an
    older token immediately. Same-key callers wait for that one computation; no
    shared user data or non-serializable lock is stored in Streamlit state.
    """
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("cache limit must be positive")
    namespace, control = _control(state)
    cache_key = "_candidate_cache_" + category
    local_key = namespace + ":" + key
    pending_key = (category, local_key)
    start = perf_counter()
    waited = False
    while True:
        with control.lock:
            entries = state.setdefault(cache_key, {})
            if local_key in entries:
                value = entries.pop(local_key)
                entries[local_key] = value
                return deepcopy(value), {"cache_hit": True, "waited_for_same_request": waited,
                                          "elapsed_seconds": perf_counter() - start}
            pending = control.pending.get(pending_key)
            if pending is None:
                pending = threading.Event()
                control.pending[pending_key] = pending
                break
        waited = True
        pending.wait()
    try:
        value = compute()
        with control.lock:
            entries = state.setdefault(cache_key, {})
            entries[local_key] = deepcopy(value)
            while len(entries) > limit:
                entries.pop(next(iter(entries)))
            state[cache_key] = entries
            return deepcopy(entries[local_key]), {"cache_hit": False, "waited_for_same_request": waited,
                                                 "elapsed_seconds": perf_counter() - start}
    finally:
        with control.lock:
            control.pending.pop(pending_key, None)
            pending.set()


def begin_request(state: MutableMapping, key: str) -> tuple[int, str]:
    _, control = _control(state)
    with control.lock:
        generation = int(state.get("_candidate_request_generation", 0)) + 1
        state["_candidate_request_generation"] = generation
        state["_candidate_active_request"] = key
        return generation, key


def is_current_request(state: MutableMapping, token) -> bool:
    _, control = _control(state)
    with control.lock:
        return token == (state.get("_candidate_request_generation"), state.get("_candidate_active_request"))


def commit_result(state: MutableMapping, token, result, problem, *, input_key, envelope=None) -> bool:
    """A late completed request may not replace a newer request or its result."""
    _, control = _control(state)
    with control.lock:
        if not is_current_request(state, token):
            return False
        state["_candidate_uploaded_result"] = result
        state["_candidate_uploaded_problem"] = problem
        state["_candidate_result_request"] = token[1]
        state["_candidate_result_input_key"] = input_key
        state["_candidate_adaptive_result"] = envelope
        return True

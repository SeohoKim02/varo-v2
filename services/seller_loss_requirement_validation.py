"""Seller Loss input requirement planner validation on the real Suhyup 2026-07-31 production rows.

Stage A uses the real rows only, with no seller input, and gives the minimum-input analysis. Those counts do not depend
on any entered value. The staged E2E then has a test seller enter exactly what the planner asks for:
    A  no seller input -> plan
    B  only the first recommended input of every decision -> plan again (that blocking reason must disappear)
    C  every structural (tier 1) input that stage A asked for -> Seller Loss evaluation
    D  C + the robustness (tier 2) inputs the plan then asks for, repeated until nothing more is asked
    STEP  one recommended input per decision per step, until the planner stops asking (each step verified)
The entered numbers are TEST USER INPUT (labelled; not Suhyup data and not defaults). They only show that every request
unblocks what it claims. Any RECOMMENDABLE count reached with them is an illustration, not a property of Suhyup.

Run: python -m services.seller_loss_requirement_validation --data-root C:/VARO_V2_REAL_DATA
Writes (local only, never into git) <data-root>/_SELLER_LOSS_VALIDATION/:
    seller_loss_minimum_input_analysis.csv          scope x decision: minimum inputs (fields, seller entries, columns)
    seller_loss_suhyup_input_requirements.csv       scope x decision x field: state, provenance, request, reason (stage A)
    seller_loss_requirement_planner_validation.csv  scope x stage x decision: added inputs, unblocked reasons, consistency
    seller_loss_requirement_summary.json            engine trace, counts, buckets, staged results, checks, input hashes
Raw and processed files are read only; the TEST USER INPUT exists only in memory and in these result files.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from services import seller_loss_input_requirements as req
from services import seller_loss_inputs as sli
from services.real_data_adapters import DATA_ROOT
from services.seller_decision_validation import OUTPUT_FOLDER
from services.seller_loss_engine import ENGINE_VERSION, STRATEGIES, contract_document
from services.seller_loss_input_validation import PROCESSED, PROCESSED_FILES, _run, _sha256, load_suhyup_upload

LABEL = "SELLER_LOSS_INPUT_REQUIREMENT_PLANNER_VALIDATION"
TEST_INPUT_LABEL = "TEST USER INPUT (planner-requested field, explicit test value, not Suhyup data, not a default)"
OUTPUT_FILES = ("seller_loss_minimum_input_analysis.csv", "seller_loss_suhyup_input_requirements.csv",
                "seller_loss_requirement_planner_validation.csv", "seller_loss_requirement_summary.json")
STAGED_SCOPES = tuple(req.COMPARISON_SCOPES)
# Test values for the columns the planner may request (daily_demand depends on the store's role in the decisions).
TEST_VALUES: dict[str, Any] = {"normal_price": 30000, "remaining_shelf_life_days": 21, "discount_rate": 0.3,
                               "promotion_uplift": 0.5, "holding_cost_per_unit_day": 20, "disposal_cost_per_unit": 500,
                               "salvage_value_per_unit": 0, "transfer_cost": 100000, "transit_time_days": 0.5,
                               "quantity_unit": "BOX"}
TEST_DEMAND_BY_ROLE = {"source": 5, "target": 60, "both": 20}   # slow source, fast target
MAX_STEPS = 25


class TestSeller:
    """Answers planner requests with labelled TEST values; rows are keyed exactly as the planner suggests."""

    def __init__(self, decisions: Sequence[Mapping[str, Any]]):
        roles: dict[tuple[str, str], set[str]] = {}
        for d in decisions:
            roles.setdefault((d["product_id"], d["source_store_id"]), set()).add("source")
            if d.get("target_store_id"):
                roles.setdefault((d["product_id"], d["target_store_id"]), set()).add("target")
        self.roles = {key: ("both" if len(value) > 1 else next(iter(value))) for key, value in roles.items()}
        self.rows: dict[str, dict[str, Any]] = {}

    def value(self, entry: Mapping[str, Any]) -> Any:
        if entry["column"] == "daily_demand":
            keys = entry["keys"]
            return TEST_DEMAND_BY_ROLE[self.roles.get((keys["product_id"], keys["store_id"]), entry["store_role"])]
        return TEST_VALUES[entry["column"]]

    def answer(self, items: Sequence[Mapping[str, Any]]) -> list[str]:
        """Enter one row per requested seller entry (deduplicated); returns the engine fields answered."""
        for item in items:
            entry = item["seller_entry"]
            self.rows.setdefault(entry["entry_key"], {"scope": entry["suggested_scope"], **entry["keys"],
                                                      entry["column"]: self.value(entry), "note": TEST_INPUT_LABEL})
        return [item["field"] for item in items]

    def frame(self) -> pd.DataFrame | None:
        return pd.DataFrame(list(self.rows.values())) if self.rows else None

    def copy(self) -> "TestSeller":
        clone = TestSeller([])
        clone.roles, clone.rows = dict(self.roles), dict(self.rows)
        return clone


def _plans(analysis: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    return {d["decision_id"]: (d["input_requirements"], d) for d in analysis["decisions"]}


def _stage_rows(scope: str, stage: str, step: int | None, analysis: Mapping[str, Any],
                previous: Mapping[str, Any] | None, added: Mapping[str, list[str]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows, problems = [], []
    before = _plans(previous) if previous else {}
    for decision_id, (plan, decision) in _plans(analysis).items():
        consistency = req.check_plan_consistency(plan, decision)
        verify = None
        if decision_id in before and added.get(decision_id):
            verify = req.verify_unblock_step(before[decision_id][0], before[decision_id][1], plan, decision, added[decision_id])
            problems.extend(f"{scope}:{stage}:{decision_id}:{p}" for p in verify["problems"])
        problems.extend(f"{scope}:{stage}:{decision_id}:CONSISTENCY:{p}" for p in consistency)
        status = plan["comparison_status_if_run_now"]
        rows.append({
            "label": LABEL, "comparison_scope": scope, "stage": stage, "step": step, "decision_id": decision_id,
            "product_id": decision["product_id"], "source_store_id": decision["source_store_id"],
            "target_store_id": decision["target_store_id"], "added_inputs": "|".join(added.get(decision_id, [])),
            "planning_status": plan["planning_status"], "stop_asking": plan["stop_asking"],
            "required_user_inputs": "|".join(i["field"] for i in plan["required_user_inputs"]),
            "required_input_count": plan["required_input_count"], "structural_input_count": plan["structural_input_count"],
            "robustness_input_count": plan["robustness_input_count"],
            "first_recommended_input": plan["recommended_next_inputs"][0]["field"] if plan["recommended_next_inputs"] else None,
            "engine_comparison_status": decision["comparison_status"], "scope_status": status["scope_status"],
            "comparable_strategies": "|".join(decision["comparable_strategies"]),
            "recommendation_readiness": decision["recommendation_readiness"], "evidence_level": decision["evidence_level"],
            "recommended_strategy": decision["recommended_strategy"],
            "value_outcomes": "|".join(v["code"] for v in plan["value_outcomes"]),
            "removed_engine_reasons": "|".join(verify["removed_engine_reasons"]) if verify else "",
            "revealed_engine_reasons": "|".join(verify["revealed_engine_reasons"]) if verify else "",
            "unblock_check": ("PASS" if verify["ok"] else "FAIL") if verify else "",
            "unblock_problems": "|".join(verify["problems"]) if verify else "",
            "consistency_problems": "|".join(consistency),
            "production_action_applied": plan["production_action_applied"], "legacy_action": decision["legacy_action"],
            "legacy_action_changed": decision["legacy_action_changed"], "message_ko": plan["message_ko"],
        })
    return rows, problems


def staged_flow(run: Callable[[pd.DataFrame | None], Mapping[str, Any]], scope: str, *,
                max_steps: int = MAX_STEPS) -> dict[str, Any]:
    """A/B/C/D stages + the one-input-per-step loop for one comparison scope. ``run(seller_frame) -> state``."""
    def analysis_of(state: Mapping[str, Any]) -> Mapping[str, Any]:
        return state["pipeline_result"]["seller_loss_analysis"]

    states: dict[str, Mapping[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    states["A"] = run(None)
    a = analysis_of(states["A"])
    stage_rows, stage_problems = _stage_rows(scope, "A_NO_SELLER_INPUT", None, a, None, {})
    rows += stage_rows
    problems += stage_problems
    a_plans = _plans(a)

    seller_b = TestSeller(a["decisions"])
    added_b = {d: seller_b.answer([p["required_user_inputs"][0]]) for d, (p, _) in a_plans.items() if p["required_user_inputs"]}
    states["B"] = run(seller_b.frame())
    b = analysis_of(states["B"])
    stage_rows, stage_problems = _stage_rows(scope, "B_FIRST_RECOMMENDED_INPUT", None, b, a, added_b)
    rows += stage_rows
    problems += stage_problems

    seller_c = TestSeller(a["decisions"])
    added_c = {d: seller_c.answer([i for i in p["required_user_inputs"] if i["tier"] == 1]) for d, (p, _) in a_plans.items()}
    states["C"] = run(seller_c.frame())
    c = analysis_of(states["C"])
    stage_rows, stage_problems = _stage_rows(scope, "C_STRUCTURAL_MINIMUM", None, c, a, added_c)
    rows += stage_rows
    problems += stage_problems

    seller_d, previous, rounds = seller_c.copy(), c, 0
    robustness_added: dict[str, list[str]] = {}
    while rounds < 4:
        asks = {d: [i for i in p["required_user_inputs"] if i["tier"] == 2] for d, (p, _) in _plans(previous).items()}
        asks = {d: items for d, items in asks.items() if items}
        if not asks:
            break
        rounds += 1
        added = {d: seller_d.answer(items) for d, items in asks.items()}
        for d, fields in added.items():
            robustness_added.setdefault(d, []).extend(fields)
        states[f"D{rounds}"] = run(seller_d.frame())
        current = analysis_of(states[f"D{rounds}"])
        stage_rows, stage_problems = _stage_rows(scope, f"D{rounds}_ROBUSTNESS_INPUTS", None, current, previous, added)
        rows += stage_rows
        problems += stage_problems
        previous = current
    final_d = previous

    seller_s, previous, step = TestSeller(a["decisions"]), a, 0
    step_inputs: dict[str, list[str]] = {}
    while step < max_steps:
        asks = {d: p["required_user_inputs"][0] for d, (p, _) in _plans(previous).items() if p["required_user_inputs"]}
        if not asks:
            break
        step += 1
        added = {d: seller_s.answer([item]) for d, item in asks.items()}
        for d, fields in added.items():
            step_inputs.setdefault(d, []).extend(fields)
        current = analysis_of(run(seller_s.frame()))
        stage_rows, stage_problems = _stage_rows(scope, "STEP", step, current, previous, added)
        rows += stage_rows
        problems += stage_problems
        previous = current
    final_step = previous

    audit_problems = []
    for decision in final_step["decisions"]:
        for name in step_inputs.get(decision["decision_id"], []):
            if name == "quantity_unit":
                continue
            applied = [a_ for a_ in decision["seller_input_audit"] if a_["field"] == name and a_["outcome"] in req.APPLIED_OUTCOMES]
            if not applied or decision["input_sources"][name]["origin"] != "SELLER_LOSS_INPUTS":
                audit_problems.append(f"{scope}:{decision['decision_id']}:{name}:NOT_IN_AUDIT")
            elif decision["comparison_status"] != "COMPARISON_UNAVAILABLE" and name not in decision["used_input_fields"] \
                    and any(name in req.strategy_required_fields(s) for s in decision["comparable_strategies"]):
                audit_problems.append(f"{scope}:{decision['decision_id']}:{name}:NOT_USED_IN_COMPARISON")

    def counts(analysis: Mapping[str, Any], key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(d[key]) for d in analysis["decisions"]).items()))

    def plan_counts(analysis: Mapping[str, Any]) -> dict[str, int]:
        return dict(sorted(Counter(d["input_requirements"]["planning_status"] for d in analysis["decisions"]).items()))

    stage_summary = {}
    for name, analysis in (("A", a), ("B", b), ("C", c), ("D_final", final_d), ("STEP_final", final_step)):
        stage_summary[name] = {"comparison_status": counts(analysis, "comparison_status"),
                               "recommendation_readiness": counts(analysis, "recommendation_readiness"),
                               "planning_status": plan_counts(analysis),
                               "recommended_strategy": counts(analysis, "recommended_strategy")}
    return {
        "rows": rows, "problems": problems, "audit_problems": audit_problems, "states": states,
        "stage_summary": stage_summary, "steps": step, "robustness_rounds": rounds,
        "a": a, "b": b, "c": c, "final_d": final_d, "final_step": final_step,
        "inputs_per_decision_until_stop": {d: len(v) for d, v in sorted(step_inputs.items())},
        "robustness_inputs_after_structural": {d: v for d, v in sorted(robustness_added.items())},
        "test_rows": {"C": list(seller_c.rows.values()), "D": list(seller_d.rows.values()), "STEP": list(seller_s.rows.values())},
    }


def _minimum_rows(scope: str, analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for decision in analysis["decisions"]:
        plan = decision["input_requirements"]
        strategies = plan["strategy_requirements"]
        required = plan["required_user_inputs"]
        out.append({
            "label": LABEL, "comparison_scope": scope, "decision_id": decision["decision_id"],
            "product_id": decision["product_id"], "source_store_id": decision["source_store_id"],
            "target_store_id": decision["target_store_id"], "planning_status": plan["planning_status"],
            "required_input_count": plan["required_input_count"], "structural_input_count": plan["structural_input_count"],
            "required_entry_count": plan["required_entry_count"], "required_column_count": plan["required_column_count"],
            "bucket_by_entries": plan["input_count_bucket"], "bucket_by_columns": req.input_count_bucket(plan["required_column_count"]),
            "required_fields": "|".join(i["field"] for i in required),
            "required_labels": "|".join(i["label"] for i in required),
            "proxy_replacement_fields": "|".join(i["field"] for i in plan["proxy_replacement_inputs"]),
            "pending_conditional_fields": "|".join(i["field"] for i in plan["optional_user_inputs"] if i["request_kind"] == req.PENDING_EVALUATION),
            "recommendable_upper_bound": plan["required_input_count"] + plan["pending_conditional_count"],
            **{f"{s.lower()}_unmet_fields": "|".join(n for n in strategies[s]["unmet_fields"] if n != "decision_qty") for s in STRATEGIES},
            **{f"{s.lower()}_status": strategies[s]["status"] for s in STRATEGIES},
            "already_available_fields": "|".join(f"{x['field']}={x['provenance']}" for x in plan["already_available_fields"]),
            "auto_proxy_fields": "|".join(x["field"] for x in plan["auto_derived_fields"] if x["status"] == "AUTO_PROXY"),
            "first_recommended_input": plan["recommended_next_inputs"][0]["field"] if plan["recommended_next_inputs"] else None,
            "message_ko": plan["message_ko"],
        })
    return out


def _field_rows(scope: str, analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for decision in analysis["decisions"]:
        plan = decision["input_requirements"]
        base = {"label": LABEL, "comparison_scope": scope, "decision_id": decision["decision_id"],
                "product_id": decision["product_id"], "source_store_id": decision["source_store_id"],
                "target_store_id": decision["target_store_id"]}
        for item in plan["recommended_next_inputs"]:
            entry = item["seller_entry"]
            out.append({**base, "row_type": "REQUEST" if item["tier"] < 3 else "OPTIONAL", "field": item["field"],
                        "label_ko": item["label"], "requirement_class": item["requirement_class"],
                        "request_kind": item["request_kind"], "tier": item["tier"], "priority": item["priority"],
                        "current_state": item["current"]["state"], "current_provenance": item["current"]["provenance"],
                        "current_value": item["current"]["value"], "unblocks_strategies": "|".join(item["unblocks_strategies"]),
                        "unblock_count": item["unblock_count"], "seller_column": entry["column"],
                        "suggested_scope": entry["suggested_scope"],
                        "keys": "|".join(f"{k}={v}" for k, v in sorted(entry["keys"].items())),
                        "expected_currency": entry.get("expected_currency"), "bundle": item["bundle"],
                        "priority_basis": item["priority_basis"], "reason_ko": item["reason"]})
        for item in plan["already_available_fields"]:
            out.append({**base, "row_type": "AVAILABLE", "field": item["field"], "label_ko": item["label"],
                        "current_state": "AVAILABLE", "current_provenance": item["provenance"], "current_value": item["value"],
                        "reason_ko": f"이미 확보됨({item['satisfied_by']}): 다시 묻지 않음"})
        for item in plan["auto_derived_fields"]:
            if item["status"] == "AUTO_PROXY":
                out.append({**base, "row_type": "AUTO_PROXY", "field": item["field"], "label_ko": item["label"],
                            "current_state": "PROXY", "current_provenance": item["provenance"], "current_value": item["value"],
                            "reason_ko": item["status_label"]})
        for item in plan["not_needed_fields"]:
            out.append({**base, "row_type": "NOT_NEEDED", "field": item["field"], "label_ko": item["label"], "reason_ko": item["reason"]})
    return out


def _distribution(values: Sequence[int]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(Counter(values).items())}


def _scope_minimum(analysis: Mapping[str, Any]) -> dict[str, Any]:
    plans = [d["input_requirements"] for d in analysis["decisions"]]
    entries = {i["seller_entry"]["entry_key"] for p in plans for i in p["required_user_inputs"]}
    per_strategy = {}
    for s in STRATEGIES:
        unmet = [[n for n in p["strategy_requirements"][s]["unmet_fields"] if n != "decision_qty"] for p in plans]
        per_strategy[s] = {"unmet_count_per_decision": _distribution([len(u) for u in unmet]),
                           "fields": dict(Counter(n for u in unmet for n in u).most_common())}
    return {
        "decisions": len(plans),
        "fields_per_decision": _distribution([p["required_input_count"] for p in plans]),
        "seller_entries_per_decision": _distribution([p["required_entry_count"] for p in plans]),
        "distinct_columns_per_decision": _distribution([p["required_column_count"] for p in plans]),
        "bucket_by_entries": _distribution([p["input_count_bucket"] for p in plans]),
        "bucket_by_columns": _distribution([req.input_count_bucket(p["required_column_count"]) for p in plans]),
        "required_fields": dict(Counter(i["field"] for p in plans for i in p["required_user_inputs"]).most_common()),
        "proxy_replacements": dict(Counter(i["field"] for p in plans for i in p["proxy_replacement_inputs"]).most_common()),
        "pending_conditional_fields": dict(Counter(i["field"] for p in plans for i in p["optional_user_inputs"]
                                                   if i["request_kind"] == req.PENDING_EVALUATION).most_common()),
        "recommendable_upper_bound_per_decision": _distribution([p["required_input_count"] + p["pending_conditional_count"] for p in plans]),
        "workbook_distinct_seller_entries": len(entries),
        "planning_status": dict(Counter(p["planning_status"] for p in plans)),
        "per_strategy_minimum": per_strategy,
        "first_recommended_input": dict(Counter(p["recommended_next_inputs"][0]["field"] for p in plans if p["recommended_next_inputs"])),
    }


def _check(check_id: str, description: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"check_id": check_id, "description": description, "status": "PASS" if ok else "FAIL",
            "detail": json.dumps(detail, ensure_ascii=False, default=str) if detail is not None else ""}


def run_requirement_validation(data_root: Path) -> dict[str, Any]:
    data_root = Path(data_root)
    inputs = [data_root / PROCESSED / name for name in PROCESSED_FILES]
    hashes_before = {str(p.relative_to(data_root)).replace("\\", "/"): _sha256(p) for p in inputs}
    upload = load_suhyup_upload(data_root)

    def runner(scope: str) -> Callable[[pd.DataFrame | None], Mapping[str, Any]]:
        def run(frame: pd.DataFrame | None) -> Mapping[str, Any]:
            data = {**upload, req.COMPARISON_SCOPE_KEY: scope}
            if frame is not None:
                data[sli.SHEET_KEY] = frame
            return _run(data, data_root)
        return run

    stage_a = {scope: runner(scope)(None) for scope in req.COMPARISON_SCOPES}
    repeat_a = runner(req.ALL_THREE)(None)
    flows = {scope: staged_flow(runner(scope), scope) for scope in STAGED_SCOPES}

    minimum_rows = [row for scope, state in stage_a.items()
                    for row in _minimum_rows(scope, state["pipeline_result"]["seller_loss_analysis"])]
    field_rows = [row for scope, state in stage_a.items()
                  for row in _field_rows(scope, state["pipeline_result"]["seller_loss_analysis"])]
    planner_rows = [row for flow in flows.values() for row in flow["rows"]]
    analyses_a = {scope: state["pipeline_result"]["seller_loss_analysis"] for scope, state in stage_a.items()}
    minimum = {scope: _scope_minimum(analysis) for scope, analysis in analyses_a.items()}

    base_recs = stage_a[req.ALL_THREE]["recommendations"]
    base_summary = stage_a[req.ALL_THREE]["pipeline_result"]["summary"]
    all_states = list(stage_a.values()) + [s for flow in flows.values() for s in flow["states"].values()]
    a_plans = [d["input_requirements"] for a in analyses_a.values() for d in a["decisions"]]
    consistency = [p for a in analyses_a.values() for d in a["decisions"]
                   for p in req.check_plan_consistency(d["input_requirements"], d)]
    real_requested = [(i["field"], i["current"]["provenance"]) for p in a_plans for i in p["required_user_inputs"]
                      if i["current"]["provenance"] in ("DIRECT_REAL", "DERIVED_REAL") and i["current"]["state"] == "AVAILABLE"]
    discount_fields, transfer_fields = {"discount_rate", "promotion_uplift"}, {
        "target_daily_demand", "target_normal_price", "target_current_stock", "transfer_cost", "transit_time_days"}
    tn = analyses_a[req.TRANSFER_VS_NORMAL]["decisions"]
    nd = analyses_a[req.NORMAL_VS_DISCOUNT]["decisions"]
    b_rows = [r for r in planner_rows if r["stage"] == "B_FIRST_RECOMMENDED_INPUT" and r["added_inputs"]]
    step_rows = [r for r in planner_rows if r["unblock_check"]]
    stage_c = {scope: flow["c"] for scope, flow in flows.items()}
    missing_after_c = [(scope, d["decision_id"], code) for scope, c in stage_c.items() for d in c["decisions"]
                       for s in req.COMPARISON_SCOPES[scope] for code in d["strategies"][s]["unavailable_reasons"]
                       if code.partition(":")[0] in req.MISSING_CODE_FIELDS or code.startswith("PROXY_REJECTED")]
    final_plans = [d for flow in flows.values() for d in flow["final_step"]["decisions"]]
    stop_violations = [d["decision_id"] for flow in flows.values() for analysis in (flow["final_d"], flow["final_step"])
                       for d in analysis["decisions"]
                       if d["recommendation_readiness"] == sli.RECOMMENDABLE and d["input_requirements"]["scope_complete"]
                       and d["input_requirements"]["required_user_inputs"]]
    checks = [
        _check("R1", "Planner and engine/evidence agree for every decision in every scope and stage (0 contradictions)",
               not consistency and not [p for f in flows.values() for p in f["problems"] if ":CONSISTENCY:" in p],
               consistency[:20]),
        _check("R2", "Values already available (real stock, real transport cost, transit time) are never requested",
               not real_requested, real_requested[:20]),
        _check("R3", "Outbound-proxy demand is requested as a PROXY replacement at the source (and at the target when TRANSFER is in scope)",
               all(any(i["field"] == "source_daily_demand" and i["request_kind"] == req.REPLACE_PROXY for i in d["input_requirements"]["required_user_inputs"])
                   for a in analyses_a.values() for d in a["decisions"])
               and all(any(i["field"] == "target_daily_demand" and i["request_kind"] == req.REPLACE_PROXY for i in d["input_requirements"]["required_user_inputs"])
                       for scope in (req.ALL_THREE, req.TRANSFER_VS_NORMAL, req.TRANSFER_VS_DISCOUNT) for d in analyses_a[scope]["decisions"])),
        _check("R4", "Comparison scope filters requests: TRANSFER_VS_NORMAL never asks discount inputs, NORMAL_VS_DISCOUNT never asks transfer inputs",
               all(not discount_fields & {i["field"] for i in d["input_requirements"]["required_user_inputs"]} for d in tn)
               and all(not transfer_fields & {i["field"] for i in d["input_requirements"]["required_user_inputs"]} for d in nd)),
        _check("R5", "B: the first recommended input of every decision removed exactly its blocking reason",
               bool(b_rows) and all(r["unblock_check"] == "PASS" for r in b_rows),
               [r["unblock_problems"] for r in b_rows if r["unblock_check"] != "PASS"][:10]),
        _check("R6", "Every staged/stepwise input removed what it was requested for (no NO_EFFECT, no new structural request)",
               bool(step_rows) and all(r["unblock_check"] == "PASS" for r in step_rows),
               [p for f in flows.values() for p in f["problems"]][:20]),
        _check("R7", "C: after the structural minimum no missing-field or proxy reason remains for any in-scope strategy",
               not missing_after_c, missing_after_c[:20]),
        _check("R8", "Stop asking: a RECOMMENDABLE decision with a complete scope never has required inputs",
               not stop_violations, stop_violations[:20]),
        _check("R9", "Every planner-requested input the seller entered is applied in the audit trail and used by the comparison",
               not [p for f in flows.values() for p in f["audit_problems"]],
               [p for f in flows.values() for p in f["audit_problems"]][:20]),
        _check("R10", "Legacy outputs (recommendations, KPI summary) identical in every run; production action never applied",
               all(s["recommendations"] == base_recs and s["pipeline_result"]["summary"] == base_summary for s in all_states)
               and all(not d["legacy_action_changed"] and not d["input_requirements"]["production_action_applied"]
                       for s in all_states for d in s["pipeline_result"]["seller_loss_analysis"]["decisions"])),
        _check("R11", "Deterministic: re-running stage A reproduces every plan",
               [d["input_requirements"] for d in repeat_a["pipeline_result"]["seller_loss_analysis"]["decisions"]]
               == [d["input_requirements"] for d in analyses_a[req.ALL_THREE]["decisions"]]),
    ]
    hashes_after = {str(p.relative_to(data_root)).replace("\\", "/"): _sha256(p) for p in inputs}
    checks.append(_check("R12", "Processed real inputs unchanged (sha256 before == after); TEST USER INPUT never written to the data root",
                         hashes_before == hashes_after, hashes_after))
    staged = {scope: {"stage_summary": flow["stage_summary"], "steps_until_stop": flow["steps"],
                      "robustness_rounds": flow["robustness_rounds"],
                      "inputs_per_decision_until_stop_TEST_ILLUSTRATION": _distribution(list(flow["inputs_per_decision_until_stop"].values())),
                      "robustness_inputs_after_structural_TEST_ILLUSTRATION": dict(Counter(
                          n for v in flow["robustness_inputs_after_structural"].values() for n in v).most_common()),
                      "robustness_inputs_count_TEST_ILLUSTRATION": _distribution(
                          [len(flow["robustness_inputs_after_structural"].get(d["decision_id"], [])) for d in flow["c"]["decisions"]]),
                      "test_rows_structural": flow["test_rows"]["C"]} for scope, flow in flows.items()}
    return {"minimum_rows": minimum_rows, "field_rows": field_rows, "planner_rows": planner_rows, "minimum": minimum,
            "staged": staged, "checks": checks, "input_hashes": hashes_after,
            "decision_count": len(analyses_a[req.ALL_THREE]["decisions"]), "final_plans": len(final_plans)}


def run_validation(data_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    data_root = Path(data_root)
    output_dir = Path(output_dir) if output_dir else data_root / OUTPUT_FOLDER
    result = run_requirement_validation(data_root)
    trace = req.trace_engine_requirements()
    failed = [c["check_id"] for c in result["checks"] if c["status"] != "PASS"]
    summary = {
        "label": LABEL,
        "data_source": f"{PROCESSED} (Suhyup 2026-07-31 production rows) + real transport cost matrix",
        "answer_basis": "Minimum-input counts come from stage A (real rows, no seller input) and do not depend on any "
                        "entered value. Staged results use TEST USER INPUT and are wiring illustrations only.",
        "engine_version": ENGINE_VERSION, "engine_contract_signature": contract_document()["contract_signature"],
        "input_contract_signature": sli.input_contract_document()["contract_signature"],
        "planner_contract": req.requirement_contract_document(),
        "engine_trace": {name: {"field_class": f["field_class"], "label": f["label"], "seller_enterable": f["seller_enterable"],
                                "per_strategy": {s: [f["per_strategy"][s]["class"], f["per_strategy"][s]["condition"]] for s in STRATEGIES}}
                         for name, f in trace["fields"].items()},
        "engine_trace_probes": trace["references"],
        "decision_count": result["decision_count"],
        "minimum_inputs_by_scope": result["minimum"],
        "staged_e2e": result["staged"],
        "checks": result["checks"], "failed_checks": failed,
        "test_user_input_label": TEST_INPUT_LABEL, "test_values": TEST_VALUES, "test_demand_by_role": TEST_DEMAND_BY_ROLE,
        "input_hashes": result["input_hashes"],
        "production_action_replaced": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["minimum_rows"]).to_csv(output_dir / OUTPUT_FILES[0], index=False, encoding="utf-8-sig")
    pd.DataFrame(result["field_rows"]).to_csv(output_dir / OUTPUT_FILES[1], index=False, encoding="utf-8-sig")
    pd.DataFrame(result["planner_rows"]).to_csv(output_dir / OUTPUT_FILES[2], index=False, encoding="utf-8-sig")
    (output_dir / OUTPUT_FILES[3]).write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = run_validation(args.data_root, args.output_dir)
    print(json.dumps({"label": summary["label"], "decision_count": summary["decision_count"],
                      "failed_checks": summary["failed_checks"],
                      "minimum": {s: {k: v[k] for k in ("fields_per_decision", "seller_entries_per_decision",
                                                        "distinct_columns_per_decision", "bucket_by_entries")}
                                  for s, v in summary["minimum_inputs_by_scope"].items()},
                      "staged": {s: v["stage_summary"] for s, v in summary["staged_e2e"].items()}},
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

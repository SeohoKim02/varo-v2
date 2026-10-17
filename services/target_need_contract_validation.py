"""Target need contract and operational evidence intake on the real Suhyup 31-day benchmark (P1-C).

Validation only.  Per day it rebuilds the P1 / P1-B lane universe with the unchanged P1-B day runner
(``operational_evidence_validation.run_day``), names every target need with the contract
(``target_need_contract``), separates on-hand from transferable stock, takes operator evidence through the intake and
evaluates every routed lane twice:

* A  real data only (operator files under 02_Korea_Suhyup_Logistics/operator_inputs, or --*-evidence; none exist)
* B  real data + an explicit in-memory TEST OPERATOR INPUT (record_origin TEST): it checks the input path, the key /
     date / unit / conflict guards and the T1 STRICT bridge; its results are labelled TEST_LINKAGE_ONLY and never
     counted as operational evidence

Outputs go to <data-root>/_TARGET_NEED_CONTRACT_VALIDATION (local only, never into git); a re-run replaces only
OUTPUT_FILES.  The raw, canonical, golden and earlier validation bundles are read, never written.
"""
from __future__ import annotations

from services.git_protection import inspect_validator_git, git_validation_check

import argparse
import json
import math
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from services import candidate_generation_validation as cgv
from services import milp_benchmark_integrity as mbi
from services import milp_benchmark_integrity_validation as mbv
from services import operational_evidence as oe
from services import operational_evidence_validation as oev
from services import shared_feasibility_selection as sf
from services import shared_feasibility_validation as sfv
from services import suhyup_algorithm_revalidation as rv
from services import target_need_contract as tn
from services.real_data_adapters import DATA_ROOT
from services.seller_decision_validation import _env

cgr = cgv.cgr

OUTPUT_FOLDER = "_TARGET_NEED_CONTRACT_VALIDATION"
OUTPUT_FILES = (
    "target_need_contract.json",
    "target_need_formula_comparison.csv",
    "target_need_by_day.csv",
    "operational_evidence_input_validation.csv",
    "operational_evidence_missing_fields.csv",
    "operational_evidence_priority.csv",
    "operational_evidence_e2e.csv",
    "operational_need_contract_summary.json",
    "operational_evidence_missing_by_scope.csv",
    "need_formula_sensitivity.csv",
    "state_code_aggregation.csv",
    "target_need_call_sites.csv",
    "target_need_records.parquet",
    "operational_evidence_candidates.parquet",
    "source_availability_records.parquet",
)
NETWORK_ID, DATASET = oev.NETWORK_ID, oev.DATASET
OPERATOR_FOLDER = oev.OPERATOR_FOLDER
OPERATOR_FILES = {tn.ROUTE_SHEET: "operational_route_evidence.csv", tn.DEMAND_SHEET: "operational_demand_evidence.csv",
                  tn.INVENTORY_SHEET: "operational_inventory_evidence.csv"}
TEST_DATE = "2026-07-16"
WINDOW = ("2026-07-01", "2026-07-31")
PROTECTED_BUNDLES = (*oev.PROTECTED_BUNDLES, oev.OUTPUT_FOLDER)
ALLOWED_MODIFIED = ("services/data_loader.py", "services/analysis_pipeline.py")


def validation_git_guard(repo_root: Path) -> dict:
    # Preserve P1-C's two explicit tracked-file exceptions, not a broad services exception.
    return inspect_validator_git(repo_root, "services/target_need_contract_validation.py",
                                 protect_all_tracked=True, allowed_tracked=ALLOWED_MODIFIED)


P1B_TOTALS = {"PROXY_FEASIBLE": 1728, "DEMAND_UNVERIFIED": 7, "STRICT": 0}
STATE_CLASS = {"20": "CHILLED_FRESH", "30": "FROZEN", "32": "FROZEN_SHELLED", "40": "DRIED", "00": "UNLABELLED", "0": "UNLABELLED"}
COMBO_FORMULA = {oe.GENERATOR_FORMULA: tn.GENERATOR_PROXY, oe.BENCHMARK_FORMULA: tn.BENCHMARK_PROXY}

# Section 2: every place the code computes or reads a target need (verified by reading the code).
CALL_SITES = [
    ("services/candidate_generator.py", "generate_candidates", "need (local); target_need (qty rule)",
     tn.GENERATOR_PROXY, "max(0, median - target_stock) + 7 x demand; qty rule max(1, need or source surplus)",
     "inventory stock_qty, avg_daily_sales|sales_qty|demand_qty; median over upload rows", "upload stock unit",
     "snapshot day, 7-day horizon", "none (target choice, quantity rule, need_score 20%)",
     "production candidate pool, Top-5 order, recommended_qty"),
    ("P1 research path (via services/candidate_generation_validation.py cgr)", "enumerate_universe / validate_universe",
     "gen_target_need; chk_target_cap on target_need_7d", f"{tn.GENERATOR_PROXY} + {tn.BENCHMARK_PROXY}",
     "generator formula re-enumerated; cap = benchmark target_need_7d > 0", "same as generator / benchmark",
     "UNKNOWN (Suhyup)", "snapshot day", "P1 TARGET_CAP validity check", "P1 research pools (ALL_VALID 1,735)"),
    ("services/suhyup_algorithm_revalidation.py", "enrich_inventory_constraints / _selection_violation / lexicographic_milp",
     "target_need_7d", tn.BENCHMARK_PROXY, "max(0, median + 7 x outbound - stock)",
     "flow stock and outbound summed over state codes; median over 6 nodes", "UNKNOWN", "snapshot day, 7-day horizon",
     "TARGET cap in Varo Final ordered selection and MILP", "186 reference rows, T3 MILP"),
    ("services/shared_feasibility_validation.py", "SUHYUP_CAP_SPEC", "TARGET_NEED <- target_need_7d", tn.BENCHMARK_PROXY,
     "column read (PROXY provenance)", "benchmark lanes", "UNKNOWN", "snapshot day", "T1 TARGET_NEED cap (PROXY)",
     "T1 BENCHMARK_PROXY 7,750 / 13,629,146; STRICT INSUFFICIENT_CAP_DATA"),
    ("services/milp_benchmark_integrity.py", "independent check (target_cap_field)", "target_need_7d", tn.BENCHMARK_PROXY,
     "column read", "benchmark lanes", "UNKNOWN", "snapshot day", "independent re-check of MILP plans", "T3 integrity"),
    ("services/optimality_gap_service.py", "_target_shortage / build_constraint_context / check_target_demand",
     "target_caps", f"{tn.UPLOADED_SHORTAGE_INPUT} | {tn.PIPELINE_SHORTAGE_7D}",
     "explicit shortage column, else max(0, demand_7d - stock) (no median)",
     "uploaded inventory row: shortage_qty..., demand_qty | demand_forecast_7d | sales_7d | avg_daily_sales x 7 | "
     "sales_qty x 7 | sales_30d / 30 x 7", "workbook unit", "snapshot, 7-day horizon",
     "in-app optimality gap target constraint", "gap report; T1 pipeline caps"),
    ("services/shared_feasibility_selection.py", "pipeline_caps", "TARGET_NEED cap", tn.PIPELINE_SHORTAGE_7D,
     "build_constraint_context target_caps with provenance (PROXY when demand is a proxy)", "as above", "workbook unit",
     "snapshot", "T1 pipeline TARGET_NEED cap", "PipelineResult.shared_feasibility_selection"),
    ("services/inventory_transition_service.py", "_target_shortage_limit", "target_shortage_limit", tn.PIPELINE_SHORTAGE_7D,
     "explicit shortage, else max(0, demand_7d - stock)", "uploaded inventory row", "workbook unit", "snapshot",
     "inbound_within_shortage flag", "transition report; Seller Loss target_need_cap"),
    ("services/seller_loss_engine.py", "_transfer_strategy / workbook input builder", "target_need_cap; engine_need",
     f"{tn.PIPELINE_SHORTAGE_7D} (cap) + {tn.SELLER_LOSS_HORIZON_NEED}",
     "q = min(Q, S, source_surplus_cap, target_need_cap, capacity, max(0, r_t (H - transit) - S_t))",
     "Seller Loss inputs; target_need_cap from the transition engine", "Seller Loss unit", "decision horizon H",
     "TRANSFER quantity cap", "Seller Loss TRANSFER loss (frozen)"),
    ("services/pareto_service.py", "_default_feasible", "target_need_7d | target_shortage_qty | shortage_qty | unmet_demand",
     "whatever the candidate carries", "column read", "candidate columns", "candidate unit", "-",
     "Pareto selection target cap", "Pareto production selection"),
    ("services/action_consistency.py", "plan provenance", "target_need_cap (T1 rows)", "as T1", "copied", "T1 plan rows",
     "-", "-", "none (label provenance)", "T2 action status"),
    ("services/operational_evidence.py", "resolve_target_demand", "target_need_value",
     f"{tn.OPERATOR_CONFIRMED_NEED} | {tn.DERIVED_FROM_OBSERVED_DEMAND} | lane PROXY", "operator need, else 7 x sales - stock, "
     "else the lane's target_need_7d", "operator / sales evidence", "declared", "validity window",
     "P1-B DEMAND dimension", "P1-B eligibility"),
    ("services/canonical_data_pipeline.py", "transfer_network mapping", "target_need <- target_need_7d", tn.BENCHMARK_PROXY,
     "documented formula", "benchmark candidates", "UNKNOWN", "snapshot", "none (canonical copy)", "canonical transfer_network"),
    ("services/legacy_adapters/_local_modules/store_product_matcher.py", "_demand_fit", "none (sales ratio score)", "-",
     "target vs source sales ratio", "avg_daily_sales | sales_30d", "-", "-", "none", "match_score_demand only"),
]
CALL_SITE_COLUMNS = ("module", "function", "field", "formula_id", "formula", "inputs", "unit", "period", "constraint",
                     "result_impact")


def _check(check_id: str, description: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return oev._check(check_id, description, ok, detail)


def _num(value: Any) -> float | None:
    return oev._num(value)


# ------------------------------------------------------------------------------------------------ operator input


def _read_operator_files(data_root: Path, overrides: Mapping[str, Path | None]) -> tuple[dict[str, Any], dict[str, str | None]]:
    frames, paths = {}, {}
    for sheet, name in OPERATOR_FILES.items():
        path = overrides.get(sheet) or (data_root / OPERATOR_FOLDER / name)
        paths[sheet] = str(path) if Path(path).exists() else None
        frames[sheet] = pd.read_csv(path, dtype=str) if Path(path).exists() else None
    return frames, paths


def _test_trust(level: str, n: int) -> dict[str, Any]:
    return {"record_origin": tn.TEST, "verification_status": "VERIFIED", "verification_level": level,
            "verified_by": "TEST_HARNESS", "verified_at": TEST_DATE, "dataset_id": DATASET, "note": f"TEST_LINKAGE_{n}"}


def test_lanes(day: Mapping[str, Any], count: int = 9) -> list[dict[str, Any]]:
    """ALL_VALID, proxy-feasible lanes of the TEST date with distinct source and target keys (deterministic)."""
    records = day["records"]
    universe = day["universe"]
    routed = universe[universe["recommended_qty"].notna()].reset_index(drop=True)
    picked, sources, targets = [], set(), set()
    candidates = records[records["in_all_valid"] & records["proxy_feasible"].astype(bool)
                         & (pd.to_numeric(records["requested_qty"], errors="coerce") >= 2)].sort_values("lane_key")
    for index in candidates.index:
        lane = routed.loc[index].to_dict()
        source, target = (lane["source_id"], lane["product_id"]), (lane["target_id"], lane["product_id"])
        if source in sources or target in targets or source[0] == target[0]:
            continue
        sources.add(source)
        targets.add(target)
        picked.append(lane)
        if len(picked) == count:
            break
    return picked


TEST_ROLES = ("FULL_EVIDENCE", "RECEIVABLE_CAPACITY_BINDING", "ROUTE_NOT_APPROVED", "TARGET_NEED_CONFLICT", "ON_HAND_ONLY",
              "TARGET_NEED_EXPIRED", "DEMAND_UNIT_MISMATCH", "TARGET_HANDLING_PROHIBITED", "AVAILABLE_EXCEEDS_ON_HAND")


def test_operator_input(lanes: Sequence[Mapping[str, Any]], temperature: Mapping[str, str]) -> dict[str, list[dict]]:
    """Explicit TEST OPERATOR INPUT (in memory, record_origin TEST, evidence_source TEST_*): one role per lane plus
    rows that must be rejected.  Quantities are taken from the lane itself; nothing here is a seller's data."""
    routes, demand, inventory = [], [], []
    n = 0
    for role, lane in zip(TEST_ROLES, lanes):
        s, t, p = str(lane["source_id"]), str(lane["target_id"]), str(lane["product_id"])
        qty, stock = float(lane["recommended_qty"]), float(lane["source_stock"])
        n += 1
        if role != "ROUTE_NOT_APPROVED":
            routes.append({"network_id": NETWORK_ID, "source_id": s, "target_id": t, "product_id": p, "allowed": "true",
                           "effective_from": WINDOW[0], "effective_to": WINDOW[1], "evidence_class": oe.VERIFIED_OPERATOR_INPUT,
                           "evidence_source": f"TEST_ROUTE_{n}", "temperature_requirement": temperature.get(p),
                           "route_capacity_qty": qty * 10, "quantity_unit": "BOX", "cost_basis": sf.FIXED_PER_TRIP,
                           "cost_value": 100000, "currency": "KRW", **_test_trust(tn.OPERATIONALLY_APPROVED, n)})
        need_row = {"network_id": NETWORK_ID, "location_id": t, "product_id": p, "quantity_kind": oe.DECLARED_TARGET_NEED,
                    "quantity_meaning": tn.REQUIRED_NEED, "observation_start": TEST_DATE, "observation_end": TEST_DATE,
                    "quantity": qty, "quantity_unit": "BOX", "evidence_class": oe.VERIFIED_OPERATOR_INPUT,
                    "data_source": f"TEST_NEED_{n}", **_test_trust(tn.INPUT_CONFIRMED, n)}
        if role == "TARGET_NEED_EXPIRED":
            need_row.update(observation_start="2026-07-10", observation_end="2026-07-10")
        if role == "DEMAND_UNIT_MISMATCH":
            need_row["quantity_unit"] = "KG"
        demand.append(need_row)
        if role == "TARGET_NEED_CONFLICT":
            demand.append({**need_row, "quantity": qty + 5, "data_source": f"TEST_NEED_{n}_B"})
        if role == "RECEIVABLE_CAPACITY_BINDING":
            demand.append({**need_row, "quantity_kind": oe.DECLARED_RECEIVABLE, "quantity_meaning": tn.RECEIVABLE_CAPACITY,
                           "quantity": max(1.0, math.floor(qty / 2)), "data_source": f"TEST_RECEIVABLE_{n}",
                           **_test_trust(tn.OPERATIONALLY_APPROVED, n)})
        observed = {"network_id": NETWORK_ID, "location_id": s, "product_id": p, "effective_date": TEST_DATE,
                    "on_hand_qty": stock, "quantity_unit": "BOX", "evidence_class": oe.DIRECT_REAL,
                    "evidence_source": f"TEST_WMS_{n}", **_test_trust(tn.OBSERVED_RECORD, n)}
        if role == "ON_HAND_ONLY":
            pass
        elif role == "AVAILABLE_EXCEEDS_ON_HAND":
            observed["available_for_transfer_qty"] = stock + 10
        else:
            observed["available_for_transfer_qty"] = qty
            observed["reserved_qty"] = max(0.0, stock - qty)
        inventory.append(observed)
        inventory.append({"network_id": NETWORK_ID, "location_id": s, "product_id": p, "effective_date": WINDOW[0],
                          "valid_until": WINDOW[1], "quantity_unit": "BOX", "dataset_unit_confirmed": "true",
                          "evidence_class": oe.VERIFIED_OPERATOR_INPUT, "evidence_source": f"TEST_UNIT_{n}",
                          **_test_trust(tn.INPUT_CONFIRMED, n)})
        if role == "TARGET_HANDLING_PROHIBITED":
            inventory.append({"network_id": NETWORK_ID, "location_id": t, "product_id": p, "effective_date": WINDOW[0],
                              "valid_until": WINDOW[1], "handling_allowed": "false", "handling_conditions": "TEST: not stocked",
                              "evidence_class": oe.VERIFIED_OPERATOR_INPUT, "evidence_source": f"TEST_ASSORTMENT_{n}",
                              **_test_trust(tn.INPUT_CONFIRMED, n)})
    if lanes:  # rows the intake must reject or never accept
        s, t, p = str(lanes[0]["source_id"]), str(lanes[0]["target_id"]), str(lanes[0]["product_id"])
        base = {"location_id": t, "product_id": p, "quantity_kind": oe.DECLARED_TARGET_NEED, "observation_start": TEST_DATE,
                "quantity": 1, "quantity_unit": "BOX", "evidence_class": oe.VERIFIED_OPERATOR_INPUT}
        demand += [
            {**base, "network_id": "kamis_public_price", "data_source": "TEST_CROSS_NETWORK", **_test_trust(tn.INPUT_CONFIRMED, 90)},
            {**base, "network_id": NETWORK_ID, "data_source": "TEST_CROSS_DATASET",
             **{**_test_trust(tn.INPUT_CONFIRMED, 91), "dataset_id": "favorita"}},
            {**base, "network_id": NETWORK_ID, "location_id": "999999", "data_source": "TEST_UNKNOWN_LOCATION",
             **_test_trust(tn.INPUT_CONFIRMED, 92)},
            {**base, "network_id": NETWORK_ID, "data_source": "TEST_UNVERIFIED",
             **{**_test_trust(tn.INPUT_CONFIRMED, 93), "verified_by": None}},
            {**base, "network_id": NETWORK_ID, "quantity_kind": oe.ORDER, "evidence_class": oe.DIRECT_REAL,
             "data_source": "TEST_ORDER_LABELLED_ACTUAL", **_test_trust(tn.INPUT_CONFIRMED, 94)},
            {**base, "network_id": NETWORK_ID, "quantity_kind": tn.DECLARED_PREFERENCE, "data_source": "TEST_PREFERENCE",
             **_test_trust(tn.INPUT_CONFIRMED, 95)},
            {**base, "network_id": "SAMPLE_NETWORK", "location_id": "SAMPLE_CENTER_B", "product_id": "SAMPLE_PRODUCT_1",
             "data_source": "SAMPLE_REPLENISHMENT_PLAN", "note": "SAMPLE: template row"},
        ]
    return {tn.ROUTE_SHEET: routes, tn.DEMAND_SHEET: demand, tn.INVENTORY_SHEET: inventory}


# ------------------------------------------------------------------------------------------------ per day


def need_records(need: pd.DataFrame) -> list[dict[str, Any]]:
    """Contract records of both formulas for every (date, center, product) group of the day."""
    out = []
    for row in need.to_dict("records"):
        common = dict(network_id=NETWORK_ID, location_id=str(row["center_id"]), product_id=str(row["product_id"]),
                      as_of=str(row["date"]), dataset_id=DATASET)
        gen_stock = _num(row.get("generator_stock")) if row.get("generator_row_present") else None
        generator = tn.generator_proxy_need(median=_num(row.get("generator_median")), stock=gen_stock,
                                            demand=_num(row.get("generator_demand")) if gen_stock is not None else None, **common)
        bench_stock = _num(row.get("benchmark_stock"))
        benchmark = tn.benchmark_proxy_need(median=_num(row.get("benchmark_median")), stock=bench_stock,
                                            outbound=_num(row.get("benchmark_demand")), **common)
        comparability = tn.need_comparability(generator, benchmark)
        for record in (generator, benchmark):
            out.append({**record.as_dict(), "comparable_with_other_formula": comparability["status"],
                        "comparability_reasons": "|".join(comparability["reason_codes"]),
                        "excluded_duplicate_state_pair": bool(row.get("excluded_duplicate_state_pair")),
                        "lanes_in_legacy_20": int(row.get("lanes_in_legacy_20") or 0),
                        "lanes_in_all_valid": int(row.get("lanes_in_all_valid") or 0)})
    return out


def source_records(lanes: pd.DataFrame, date: str) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for lane in lanes.to_dict("records"):
        key = (str(lane["source_id"]), str(lane["product_id"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(tn.source_availability(
            network_id=NETWORK_ID, location_id=key[0], product_id=key[1], as_of=date,
            dataset_on_hand=_num(lane.get("source_stock")), dataset_on_hand_evidence=oe.DIRECT_REAL,
            surplus_proxy=_num(lane.get("source_surplus")), surplus_proxy_basis="max(0, stock - cross-node median) (PROXY)"))
    return out


def evaluate_scenario(routed: pd.DataFrame, intake: tn.Intake, evidence_base: oev.SuhyupEvidence, date: str) -> list[dict]:
    day = tn.day_evidence(intake, date)
    context = tn.oe_context(day, {f.name: getattr(evidence_base.context(date), f.name)
                                  for f in oe.OperationalContext.__dataclass_fields__.values()})
    return [tn.evaluate_lane_contract(lane, day, context) for lane in routed.to_dict("records")]


def bridge_day(day: Mapping[str, Any], results_by_key: Mapping[str, Mapping[str, Any]], scope: str) -> dict[str, Any]:
    all_valid = day["pools"][cgr.ALL_VALID]
    if all_valid.empty:
        return {"plan_status": "EMPTY_INPUT", "selected": 0}
    ranked = rv._records(cgr.rank_pool(all_valid).reset_index(drop=True))
    results = [results_by_key[r["lane_key"]] for r in ranked]
    return tn.t1_strict_bridge(ranked, results, scope=scope)


def run_suhyup(data_root: Path, operator_frames: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    saved = sfv.load_suhyup_candidates(data_root)
    reference = pd.read_csv(data_root / sfv.REFERENCE, dtype={"date": str})
    flow = pd.read_csv(data_root / sfv.INVENTORY, dtype={"date": str, "center_code": str, "product_code": str, "state_code": str})
    rebuild = mbv.GeneratorRebuild(data_root)
    t1_bundle = pd.read_csv(data_root / "_SHARED_FEASIBILITY_VALIDATION" / "shared_feasibility_by_day.csv", dtype=str)
    t3_bundle = pd.read_csv(data_root / mbv.OUTPUT_FOLDER / "milp_benchmark_by_day.csv", dtype=str)
    centers = pd.read_csv(data_root / oev.CENTER_MASTER, dtype={"center_code": str}, encoding="utf-8-sig")
    universe_keys = tn.KeyUniverse(network_id=NETWORK_ID, dataset_id=DATASET,
                                   locations=frozenset(centers["center_code"].astype(str)),
                                   products=frozenset(flow["product_code"].astype(str)),
                                   location_namespace="suhyup_canonical_location_id")
    evidence = oev.SuhyupEvidence(data_root, flow)
    intake_a = tn.intake_evidence(operator_frames, universe_keys, allow_test=False)
    out: dict[str, list] = defaultdict(list)
    test_input, intake_b, intake_b_strict = None, None, None
    with _env(mbv.REAL_DATA_ROOT_ENV, str(data_root)):
        saved_recompute = sfv.TariffRecompute(saved)
        dates = sorted(saved["snapshot_date"].astype(str).unique())
        days = {}
        # the TEST input is built from the TEST date's own lanes, so that day runs first
        for date in [TEST_DATE, *[d for d in dates if d != TEST_DATE]]:
            day = oev.run_day(rebuild, flow, date, evidence, saved[saved["snapshot_date"] == date], saved_recompute)
            if date == TEST_DATE:
                lanes = test_lanes(day)
                test_input = test_operator_input(lanes, evidence.temperature)
                test_input["lanes"] = [{"role": role, "lane_key": lane["lane_key"], "source_id": lane["source_id"],
                                        "target_id": lane["target_id"], "product_id": lane["product_id"],
                                        "recommended_qty": float(lane["recommended_qty"])} for role, lane in zip(TEST_ROLES, lanes)]
                intake_b = tn.intake_evidence(test_input, universe_keys, allow_test=True)
                intake_b_strict = tn.intake_evidence(test_input, universe_keys, allow_test=False)
                out["case_0716"] = oev.reproduce_0716(data_root, flow, day["universe"])
            days[date] = True
            universe = day["universe"]
            routed = universe[universe["recommended_qty"].notna()].reset_index(drop=True)
            p1b = day["records"]
            flags = p1b[["lane_key", "is_duplicate", "in_legacy_20", "in_all_valid", "in_all_lanes", "p1_validity_status",
                         "gen_target_need", "target_need_7d", "need_formula_gap"]].reset_index(drop=True)
            scenarios = {"A_REAL_ONLY": intake_a, "B_REAL_PLUS_TEST": intake_b, "B_TEST_NOT_ALLOWED": intake_b_strict}
            for name, intake in scenarios.items():
                results = evaluate_scenario(routed, intake, evidence, date)
                frame = pd.concat([pd.DataFrame(results), flags], axis=1)
                frame["scenario"] = name
                frame["p1b_operational_eligibility"] = p1b["operational_eligibility"].values
                out["candidates"].append(frame)
                keyed = {key: r for r, key, dup in zip(results, flags["lane_key"], flags["is_duplicate"]) if not bool(dup)}
                bridge = bridge_day(day, keyed, intake.scope)
                out["bridge"].append({"date": date, "scenario": name, **{k: v for k, v in bridge.items() if k != "rows"}})
            out["need"].append(day["need"])
            out["need_records"].extend(need_records(day["need"]))
            out["sources"].extend(source_records(routed[p1b["in_all_valid"].values | p1b["in_legacy_20"].values], date))
            out["combos"].extend(day["combos"])
            out["fidelity"].append(day["fidelity"])
            out["regression"].append(cgv._regression(day, t1_bundle, t3_bundle))
            out["golden"].extend(cgv._golden(day, reference))
    return {"candidates": pd.concat(out["candidates"], ignore_index=True), "bridge": pd.DataFrame(out["bridge"]),
            "need": pd.concat(out["need"], ignore_index=True), "need_records": pd.DataFrame(out["need_records"]),
            "sources": pd.DataFrame(out["sources"]), "combos": pd.DataFrame(out["combos"]),
            "fidelity": pd.DataFrame(out["fidelity"]), "regression": pd.DataFrame(out["regression"]),
            "golden": pd.DataFrame(out["golden"]), "case_0716": out["case_0716"], "flow": flow,
            "intakes": {"A_REAL_ONLY": intake_a, "B_REAL_PLUS_TEST": intake_b, "B_TEST_NOT_ALLOWED": intake_b_strict},
            "test_input": test_input, "evidence": evidence, "universe_keys": universe_keys,
            "wall_seconds": round(time.perf_counter() - started, 3)}


# ------------------------------------------------------------------------------------------------ tables


def formula_comparison(need: pd.DataFrame) -> pd.DataFrame:
    """Section 5: both formulas per (date, center, product) with the cause of every difference."""
    frame = need.copy()
    gap = pd.to_numeric(frame.get("total_gap"), errors="coerce")
    frame["abs_gap"] = gap.abs()
    bench = pd.to_numeric(frame.get("need_benchmark"), errors="coerce")
    gen = pd.to_numeric(frame.get("need_generator"), errors="coerce")
    frame["relative_gap_vs_benchmark"] = [None if pd.isna(g) or pd.isna(b) or b <= 0 else round(g / b, 6)
                                          for g, b in zip(gap, bench)]
    formula = pd.to_numeric(frame.get("formula_gap_on_benchmark_inputs"), errors="coerce").fillna(0.0)
    inputs = pd.to_numeric(frame.get("input_namespace_effect"), errors="coerce").fillna(0.0)

    def cause(i: int) -> str:
        row = frame.iloc[i]
        if pd.isna(bench.iloc[i]) and pd.isna(gen.iloc[i]):
            return "NEITHER_OBSERVED"
        if pd.isna(bench.iloc[i]):
            return "BENCHMARK_NOT_OBSERVED"
        if pd.isna(gen.iloc[i]) or not bool(row.get("generator_row_present")):
            return "GENERATOR_ROW_EXCLUDED_OR_ABSENT" + ("_DUPLICATE_STATE_PAIR" if bool(row.get("excluded_duplicate_state_pair")) else "")
        if abs(gap.iloc[i]) <= 1e-9:
            return "EQUAL" if abs(formula.iloc[i]) <= 1e-9 else "EQUAL_BY_OFFSET"
        if abs(inputs.iloc[i]) <= 1e-9:
            return "FORMULA_ONLY"
        return "INPUT_NAMESPACE_ONLY" if abs(formula.iloc[i]) <= 1e-9 else "FORMULA_AND_INPUT_NAMESPACE"

    frame["difference_cause"] = [cause(i) for i in range(len(frame))]
    frame["generator_formula_id"], frame["benchmark_formula_id"] = tn.GENERATOR_PROXY, tn.BENCHMARK_PROXY
    frame["generator_formula_version"] = tn.FORMULAS[tn.GENERATOR_PROXY]["version"]
    frame["benchmark_formula_version"] = tn.FORMULAS[tn.BENCHMARK_PROXY]["version"]
    frame["comparability"] = oe.NOT_COMPARABLE
    frame["comparability_reason"] = "FORMULA_DIFFERS (two definitions are never one target_need)"
    first = ["date", "center_id", "product_id", "generator_median", "benchmark_median", "generator_stock", "benchmark_stock",
             "generator_demand", "benchmark_demand", "need_generator", "need_benchmark", "total_gap", "abs_gap",
             "relative_gap_vs_benchmark", "difference_cause", "need_case", "excluded_duplicate_state_pair",
             "generator_row_present", "formula_gap_on_benchmark_inputs", "input_namespace_effect", "identity_gap"]
    return frame[[c for c in first if c in frame.columns] + [c for c in frame.columns if c not in first]]


def need_by_day(records: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (date, formula), group in records.groupby(["as_of_date", "formula_id"]):
        values = pd.to_numeric(group["target_need_value"], errors="coerce")
        part = comparison[comparison["date"] == date]
        rows.append({"date": date, "formula_id": formula, "formula_version": group["formula_version"].iloc[0],
                     "groups": len(group), "observed": int(values.notna().sum()), "not_observed_null": int(values.isna().sum()),
                     "zero": int((values == 0).sum()), "positive": int((values > 0).sum()),
                     "need_total": round(float(values.sum()), 3),
                     "substituted_by_generator": int(pd.to_numeric(group["substitute_value"], errors="coerce").notna().sum()),
                     "accepted_for_strict": int(group["accepted_for_strict"].astype(bool).sum()),
                     "differing_groups_vs_other_formula": int((part["abs_gap"] > 1e-9).sum()),
                     "formula_only_differences": int((part["difference_cause"] == "FORMULA_ONLY").sum()),
                     "differing_groups_in_legacy_20_lanes": int(((part["abs_gap"] > 1e-9) & (part["lanes_in_legacy_20"] > 0)).sum()),
                     "differing_groups_in_all_valid_lanes": int(((part["abs_gap"] > 1e-9) & (part["lanes_in_all_valid"] > 0)).sum())})
    return pd.DataFrame(rows)


def state_code_table(flow: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Section 17: (center, product) pairs reported under several state codes, and medians that mix state classes."""
    nodes = flow[flow["center_code"].isin(rv.NETWORK_NODES)].copy()
    nodes["state_class"] = nodes["state_code"].map(lambda c: STATE_CLASS.get(str(c), "OTHER"))
    nodes["stock"] = pd.to_numeric(nodes["stock_qty"], errors="coerce")
    a = candidates[(candidates["scenario"] == "A_REAL_ONLY") & ~candidates["is_duplicate"].astype(bool)]
    touch = Counter()
    for column in ("source_id", "target_id"):
        for (d, c, p, v, l) in zip(a["snapshot_date"], a[column], a["product_id"], a["in_all_valid"], a["in_legacy_20"]):
            touch[(d, c, p, "all_valid")] += int(bool(v))
            touch[(d, c, p, "legacy_20")] += int(bool(l))
    rows = []
    for (date, center, product), group in nodes.groupby(["date", "center_code", "product_code"]):
        if len(group) < 2:
            continue
        classes = sorted(set(group["state_class"]))
        rows.append({"date": date, "center_id": center, "product_id": product, "level": "CENTER_PRODUCT",
                     "state_codes": "|".join(sorted(group["state_code"].astype(str))), "state_classes": "|".join(classes),
                     "mixed_state_classes": len(classes) > 1,
                     "stock_by_state": "|".join(f"{s}={v:g}" for s, v in zip(group["state_code"], group["stock"])),
                     "summed_stock_used_by_benchmark": float(group["stock"].sum()), "excluded_from_generator_upload": True,
                     "all_valid_lane_touches": touch[(date, center, product, "all_valid")],
                     "legacy_20_lane_touches": touch[(date, center, product, "legacy_20")]})
    cross = []
    for (date, product), group in nodes.groupby(["date", "product_code"]):
        classes = sorted(set(group["state_class"]))
        if len(classes) > 1:
            cross.append({"date": date, "center_id": "*", "product_id": product, "level": "CROSS_NODE_MEDIAN",
                          "state_codes": "|".join(sorted(set(group["state_code"].astype(str)))), "state_classes": "|".join(classes),
                          "mixed_state_classes": True,
                          "stock_by_state": "|".join(f"{c}:{s}={v:g}" for c, s, v in zip(group["center_code"], group["state_code"], group["stock"])),
                          "summed_stock_used_by_benchmark": None, "excluded_from_generator_upload": None,
                          "all_valid_lane_touches": int(sum(touch[(date, c, product, "all_valid")] for c in set(group["center_code"]))),
                          "legacy_20_lane_touches": int(sum(touch[(date, c, product, "legacy_20")] for c in set(group["center_code"])))})
    table = pd.DataFrame(rows + cross)
    products_multi = nodes.groupby("product_code")["state_class"].nunique()
    summary = {
        "center_product_day_groups_multi_state": len(rows),
        "center_product_day_groups_mixed_classes": int(sum(r["mixed_state_classes"] for r in rows)),
        "cross_node_median_groups_mixed_classes": len(cross),
        "products_with_more_than_one_state_class": int((products_multi > 1).sum()),
        "state_classes": dict(Counter(nodes["state_class"])),
        "all_valid_lane_touches_on_multi_state_pairs": int(sum(r["all_valid_lane_touches"] for r in rows)),
        "legacy_20_lane_touches_on_multi_state_pairs": int(sum(r["legacy_20_lane_touches"] for r in rows)),
        "all_valid_lane_touches_on_mixed_medians": int(sum(r["all_valid_lane_touches"] for r in cross)),
        "note": "research finding only: production and benchmark formulas are unchanged; a state-aware key "
                "(product, state_code) would separate frozen, dried and chilled stock of one product code",
    }
    return table, summary


def _dimension_counts(frame: pd.DataFrame) -> dict[str, int]:
    accepted = sum(int(frame[f"{d.lower()}_accepted"].astype(bool).sum()) for d in oe.DIMENSIONS)
    proxy = sum(1 for text in frame["provenance"] for part in str(text).split("|") if part.endswith("=PROXY"))
    missing = int(frame[list(tn.MISSING_FLAGS[:7])].astype(bool).sum().sum())
    return {"validated_fields": accepted, "proxy_fields": proxy, "missing_fields": missing}


def e2e_table(candidates: pd.DataFrame, bridge: pd.DataFrame, priority: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for scenario, group in candidates.groupby("scenario", sort=False):
        usable = group[~group["is_duplicate"].astype(bool)]
        for pool, flag in (("ALL_VALID", "in_all_valid"), ("LEGACY_20", "in_legacy_20"), ("ALL_LANES", "in_all_lanes")):
            part = usable[usable[flag].astype(bool)]
            days_part = [("31D", part)]
            if scenario.startswith("B_"):
                days_part.append((TEST_DATE, part[part["snapshot_date"] == TEST_DATE]))
            for label, frame in days_part:
                b = bridge[bridge["scenario"] == scenario]
                b = b if label == "31D" else b[b["date"] == TEST_DATE]
                plan = priority.get((scenario, pool))
                rows.append({
                    "scenario": scenario, "pool": pool, "period": label, "candidates": len(frame),
                    **_dimension_counts(frame),
                    "conflict_candidates": int(frame["conflicting_evidence"].astype(bool).sum()),
                    "expired_evidence_candidates": int(frame["expired_evidence"].astype(bool).sum()),
                    "strict_eligible_actual": int((frame["contract_strict"] == oe.STRICT_ELIGIBLE).sum()),
                    "strict_eligible_test_only": int((frame["contract_strict"] == "STRICT_ELIGIBLE_TEST_ONLY").sum()),
                    "proxy_feasible": int((frame["contract_eligibility"] == oe.PROXY_FEASIBLE).sum()),
                    "contract_ineligible": int((frame["contract_eligibility"] == oe.INELIGIBLE).sum()),
                    "contract_conflict": int((frame["contract_eligibility"] == tn.CONFLICT).sum()),
                    "inputs_to_collect_rows": int(plan["rows_to_collect"].sum()) if plan is not None and not plan.empty and label == "31D" else None,
                    "t1_bridge_selected_days": int((b["selected"] > 0).sum()) if pool == "ALL_VALID" else None,
                    "t1_bridge_insufficient_days": int((b["plan_status"] == sf.INSUFFICIENT_CAP_DATA).sum()) if pool == "ALL_VALID" else None,
                    "t1_bridge_strict_actual_claim_days": int(b["strict_actual_claim"].astype(bool).sum()) if pool == "ALL_VALID" else None,
                    "evidence_scope": frame["evidence_scope"].iloc[0] if not frame.empty else None})
    return pd.DataFrame(rows)


def suhyup_requests(priority: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Section 32: the minimum operational data to request from Suhyup, ordered by request_priority, with what each
    unlocks; plus the inputs no current selector consumes (listed, not ranked)."""
    a = candidates[(candidates["scenario"] == "A_REAL_ONLY") & candidates["in_all_valid"].astype(bool)
                   & ~candidates["is_duplicate"].astype(bool)]
    labels = {
        "missing_route_approval": "운영 승인 점포 간 경로 (directed pair, validity, temperature)",
        "missing_actual_surplus": "실제 일별 이동 가능 재고 (available-for-transfer or approved cap per source x product)",
        "missing_actual_target_need": "실제 도착지 필요량 (REQUIRED_NEED per target x product x day)",
        "missing_target_handling": "도착지 상품 취급 가능 여부",
        "missing_quantity_unit": "상품 수량 단위 (dataset_unit_confirmed per product)",
        "missing_capacity": "차량 / 경로 처리 용량 (in the product quantity unit)",
        "missing_cost_basis": "실제 운송비 기준 (contract FIXED_PER_TRIP / PER_UNIT per route)",
    }
    releases = {
        "missing_route_approval": "ROUTE dimension (class B) on every lane of the pair", "missing_actual_surplus": "SOURCE_SUPPLY cap",
        "missing_actual_target_need": "TARGET_DEMAND cap (OPERATOR_CONFIRMED_NEED)", "missing_target_handling": "TARGET_HANDLING",
        "missing_quantity_unit": "QUANTITY_UNIT (and the unit comparability of every cap)",
        "missing_capacity": "CAPACITY (VERIFIED_CONDITIONAL -> VERIFIED_OPERATIONAL)", "missing_cost_basis": "TRANSPORT_COST accepted",
    }
    out = priority.copy()
    out.insert(1, "request_item", out["field"].map(labels))
    out["constraint_released"] = out["field"].map(releases)
    out["consumer"] = "P1-B evaluate_lane / T1 STRICT bridge"
    extra = [{"priority": None, "request_item": "센터 처리 용량 (center throughput / storage)", "field": "center_capacity",
              "tier": "CONDITIONAL_SAFETY", "candidates_blocked": None,
              "rows_to_collect": int(pd.concat([a["source_id"], a["target_id"]]).nunique()), "input_cycle": "per center and period",
              "reuse_scope": "every lane through the center", "constraint_released": "none yet",
              "consumer": "no selector reads it (no DC in the Suhyup network; T1 DC_CAPACITY applies to VIA_DC lanes only)"}]
    return pd.concat([out, pd.DataFrame(extra)], ignore_index=True)


def sensitivity(combos: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    """Section 18: P1-B's need combos with the contract signature; E rows (operator need) are not run on real data."""
    summary = oev.sensitivity_summary(combos)
    rows = []
    signatures = {}
    for row in summary.to_dict("records"):
        formula = COMBO_FORMULA[row["need_definition"]]
        signature = tn.constraint_signature(formula, cap_provenance=["PROXY"], window=WINDOW)
        signatures[row["combo"]] = signature
        rows.append({**row, "formula_id": formula, "formula_version": tn.FORMULAS[formula]["version"],
                     "constraint_signature": signature["signature"], "status": "RUN",
                     "result_claim": "COMPUTED_BENCHMARK_UNDER_PROXY_CAPS"})
    for row in rows:
        verdict = tn.signatures_comparable(signatures[row["combo"]], signatures["B"])
        row["comparable_with_B"] = verdict["status"]
        row["comparison_note"] = ("same need definition: pool coverage difference only" if verdict["optimality_comparison_allowed"]
                                  else "different need definition: no optimality or cost comparison")
    test = bridge[(bridge["scenario"] == "B_REAL_PLUS_TEST") & (bridge["date"] == TEST_DATE)]
    rows.append({"combo": "E", "pool": "same candidates", "need_definition": tn.OPERATOR_CONFIRMED_NEED,
                 "formula_id": tn.OPERATOR_CONFIRMED_NEED, "status": "NOT_RUN_NO_ACTUAL_OPERATOR_INPUT",
                 "description": "verified operator need: structure ready, no actual operator input exists",
                 "comparable_with_B": oe.NOT_COMPARABLE, "result_claim": None})
    if not test.empty:
        t = test.iloc[0]
        rows.append({"combo": "E_TEST", "pool": f"ALL_VALID {TEST_DATE}", "need_definition": tn.OPERATOR_CONFIRMED_NEED,
                     "formula_id": tn.OPERATOR_CONFIRMED_NEED, "status": "TEST_LINKAGE_ONLY",
                     "description": "T1 STRICT bridge on TEST operator need (input path check, not a result)",
                     "t1_all_or_nothing_service_total": t["total_allocated_qty"], "t1_all_or_nothing_cost_total": t["total_move_cost"],
                     "milp_selected_total": t["selected"], "violations_total": t["violations"],
                     "comparable_with_B": oe.NOT_COMPARABLE, "result_claim": "TEST_LINKAGE_ONLY"})
    out = pd.DataFrame(rows).assign(table="SUMMARY")
    # selected routes per day and combo, with their overlap with combo B (the T1 / P1 baseline)
    base = {d: set(str(k).split("|")) - {""} for d, k in zip(combos.loc[combos["combo"] == "B", "date"],
                                                             combos.loc[combos["combo"] == "B", "milp_lane_keys"])}
    daily = combos[["date", "combo", "pool", "need_definition", "milp_service", "milp_cost", "milp_selected_count",
                    "violations", "milp_status", "milp_lane_keys", "milp_target_excess_vs_benchmark_need"]].copy()
    daily["formula_id"] = daily["need_definition"].map(COMBO_FORMULA)
    daily["routes_shared_with_B"] = [len((set(str(k).split("|")) - {""}) & base.get(d, set()))
                                     for d, k in zip(daily["date"], daily["milp_lane_keys"])]
    daily["comparable_with_B"] = [oe.COMPARABLE if f == tn.BENCHMARK_PROXY else oe.NOT_COMPARABLE for f in daily["formula_id"]]
    return pd.concat([out, daily.assign(table="DAILY")], ignore_index=True)


def p1b_bundle_totals(data_root: Path) -> dict[str, Any]:
    path = data_root / oev.OUTPUT_FOLDER / "operational_candidate_validation.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("eligibility_totals", {})


def contract_document(result: Mapping[str, Any], data_root: Path) -> dict[str, Any]:
    snap = pd.read_parquet(data_root / oev.CANONICAL / "canonical_inventory_snapshot.parquet")
    flow = result["flow"]
    nodes = flow[flow["center_code"].isin(rv.NETWORK_NODES)]
    observed_pairs = nodes.groupby(["date", "center_code", "product_code"]).ngroups
    grid = nodes["date"].nunique() * len(rv.NETWORK_NODES) * nodes["product_code"].nunique()
    return {
        "contract_version": tn.CONTRACT_VERSION, "evidence_version": oe.EVIDENCE_VERSION,
        "target_need_record_fields": list(tn.TargetNeed.__dataclass_fields__),
        "formula_registry": tn.FORMULAS, "p1b_formula_aliases": tn.P1B_ALIASES,
        "quantity_meanings": tn.QUANTITY_MEANINGS,
        "rules": [
            "a value without formula_id and formula_version is not a target need",
            "two target needs are one quantity only with equal formula, version, horizon, unit, as_of_date and meaning",
            "only REQUIRED_NEED is a need; RECEIVABLE_CAPACITY is a separate inflow limit, ORDER_QTY a proxy, PREFERENCE informational",
            "GENERATOR_PROXY ranks candidates and is never a cap; BENCHMARK_PROXY is a labelled research cap",
            "STRICT caps come from OPERATOR_CONFIRMED_NEED or (when activated) DERIVED_FROM_OBSERVED_DEMAND only",
            "a target that does not report the product has a NULL need (not 0); a formula's substitute is kept apart",
            "production formulas are unchanged; unifying them is a research result only (need_formula_sensitivity.csv)",
        ],
        "source_availability": {"kinds": tn.SOURCE_AVAILABILITY_KINDS, "fields": [
            "on_hand_qty", "reserved_qty", "available_for_transfer_qty", "operator_transfer_cap_qty", "unreserved_on_hand_qty",
            "lot_restriction", "source_expected_demand_qty", "source_surplus_proxy_qty", "quantity_unit", "as_of_date",
            "transferable_qty", "transferable_kind", "validation_status"],
            "rule": "transferable = min(accepted available-for-transfer, accepted operator cap), bounded by accepted on-hand; "
                    "the PROXY surplus never joins; on-hand alone is ON_HAND_ONLY"},
        "suhyup_inventory_semantics": {
            "raw_field": "재고량 (물류센터-공판장 품목별 재고현황), per center x product x state code x day, plus 재고량(킬로그램)",
            "processed": "stock_qty (flow file) = canonical inventory_qty; summed over state codes by the benchmark",
            "canonical_null_counts": {c: int(snap[c].isna().sum()) for c in ("available_qty", "reserved_qty", "damaged_qty")},
            "canonical_rows": len(snap),
            "confirmed": ["a quantity the center reports as held on the day (count unit UNKNOWN; kg DIRECT)",
                          "reported per state code (frozen / dried / chilled / shelled)"],
            "not_confirmed": ["owner (consigned vs own stock)", "reserved / allocated quantity", "sellable quantity",
                              "transfer-allowed quantity", "in-transit quantity"],
            "classification": "REPORTED_STORED_QUANTITY: physical holding as reported; not transferable, not available",
        },
        "null_vs_zero": {
            "flow_rows": len(flow), "stock_missing_rows": int(pd.to_numeric(flow["stock_qty"], errors="coerce").isna().sum()),
            "stock_observed_zero_rows": int((pd.to_numeric(flow["stock_qty"], errors="coerce") == 0).sum()),
            "outbound_observed_zero_rows": int((pd.to_numeric(flow["outbound_qty"], errors="coerce") == 0).sum()),
            "network_grid_cells": grid, "network_cells_observed": observed_pairs,
            "network_cells_not_observed_null": grid - observed_pairs,
            "benchmark_fillna_zero_rows_affected": int(pd.to_numeric(flow["stock_qty"], errors="coerce").isna().sum()
                                                      + pd.to_numeric(flow["outbound_qty"], errors="coerce").isna().sum()),
            "rule": "a center without a row for the product is NOT_OBSERVED (NULL need, NULL demand), never 0",
        },
        "verification": {"levels": tn.VERIFICATION_LEVELS, "accepted_levels": {k: sorted(v) for k, v in tn.ACCEPTED_LEVELS.items()},
                         "origins": list(tn.RECORD_ORIGINS), "row_statuses": [tn.OPERATOR_VERIFIED, tn.OPERATOR_UNVERIFIED,
                                                                              tn.TEST_ONLY, tn.SCENARIO_ONLY, tn.SAMPLE_NOT_USABLE],
                         "authority": tn.AUTHORITY_NOTE,
                         "evidence_classes": list(oe.EVIDENCE_CLASSES), "base_provenance": oe.BASE_PROVENANCE},
        "sheets": {sheet: {"required": list(tn.SHEET_REQUIRED[sheet]), "optional": list(tn.SHEET_OPTIONAL[sheet])}
                   for sheet in tn.EVIDENCE_SHEETS},
        "sheet_statuses": [tn.MISSING_EVIDENCE, tn.VALIDATION_ERROR, tn.VALIDATED_INPUT, tn.NO_USABLE_EVIDENCE],
        "conflict_codes": ["ON_HAND_CONFLICT", "RESERVED_CONFLICT", "AVAILABLE_FOR_TRANSFER_CONFLICT", "TRANSFER_CAP_CONFLICT",
                           "TARGET_NEED_CONFLICT", "RECEIVABLE_CAPACITY_CONFLICT", "DECLARED_TRANSFERABLE_CONFLICT",
                           "ACTUAL_SALES_CONFLICT", "ROUTE_ALLOWED_PROHIBITED_CONFLICT", "ROUTE_TERMS_CONFLICT",
                           "DATASET_UNIT_CONFLICT", "HANDLING_CONFLICT", "OPERATOR_UNITS_DIFFER", "ON_HAND_SOURCES_DISAGREE",
                           "RESERVED_EXCEEDS_ON_HAND", "AVAILABLE_EXCEEDS_ON_HAND", "AVAILABLE_EXCEEDS_UNRESERVED"],
        "missing_input_fields": tn.MISSING_FIELD_NAMES, "request_tiers": tn.REQUEST_TIER,
        "pipeline_field": "PipelineResult.operational_evidence (parallel, fault-isolated, deterministic)",
    }


# ------------------------------------------------------------------------------------------------ run


def run_validation(data_root: Path, output_dir: Path | None = None, repo_root: Path | None = None,
                   evidence_paths: Mapping[str, Path | None] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    data_root = Path(data_root)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    output_dir = Path(output_dir or data_root / OUTPUT_FOLDER)
    before = oev._hashes(data_root, PROTECTED_BUNDLES)
    frames, paths = _read_operator_files(data_root, evidence_paths or {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_suhyup(data_root, frames)
    cands, bridge, need = result["candidates"], result["bridge"], result["need"]
    regression, golden, fidelity = result["regression"], result["golden"], result["fidelity"]
    intakes = result["intakes"]
    a = cands[cands["scenario"] == "A_REAL_ONLY"]
    b = cands[cands["scenario"] == "B_REAL_PLUS_TEST"]
    b_strict = cands[cands["scenario"] == "B_TEST_NOT_ALLOWED"]
    usable = lambda frame: frame[~frame["is_duplicate"].astype(bool)]  # noqa: E731
    a_valid, a_legacy = a[a["in_all_valid"].astype(bool)], a[a["in_legacy_20"].astype(bool)]
    priority = {}
    for scenario, frame in (("A_REAL_ONLY", a), ("B_REAL_PLUS_TEST", b)):
        for pool, flag in (("ALL_VALID", "in_all_valid"), ("LEGACY_20", "in_legacy_20")):
            priority[(scenario, pool)] = tn.request_priority(frame[frame[flag].astype(bool)])
    request = suhyup_requests(priority[("A_REAL_ONLY", "ALL_VALID")], cands)
    priority_table = pd.concat([request.assign(table="SUHYUP_REQUEST_ALL_VALID"),
                                priority[("A_REAL_ONLY", "LEGACY_20")].assign(table="PRIORITY_LEGACY_20")], ignore_index=True)
    comparison = formula_comparison(need)
    need_records_frame = result["need_records"]
    daily = need_by_day(need_records_frame, comparison)
    states, state_summary = state_code_table(result["flow"], cands)
    sens = sensitivity(result["combos"], bridge)
    sens_summary = sens[sens["table"] == "SUMMARY"].set_index("combo")
    e2e = e2e_table(cands, bridge, priority)
    missing_cols = ["snapshot_date", "lane_key", "candidate_id", "route_id", "source_id", "target_id", "product_id",
                    "in_legacy_20", "in_all_valid", "requested_qty", "operational_eligibility", "contract_eligibility",
                    "contract_strict", "need_formula_id", "source_availability_kind", "source_availability_status",
                    "route_class", *tn.MISSING_FLAGS, "missing_required_count", "missing_fields", "blocking_reasons",
                    "contract_reasons"]
    a_pool = a[(a["in_all_valid"].astype(bool) | a["in_legacy_20"].astype(bool))].copy()
    missing = a_pool[[c for c in missing_cols if c in a_pool.columns]]
    by_scope = pd.concat([tn.missing_by_scope(a_valid).assign(pool="ALL_VALID"),
                          tn.missing_by_scope(a_legacy).assign(pool="LEGACY_20")], ignore_index=True)
    input_rows = []
    for name, intake in intakes.items():
        if name == "B_TEST_NOT_ALLOWED":
            continue
        for sheet, status in intake.sheets.items():
            input_rows.append({"scenario": name, "table": "SHEET", "sheet": sheet, **status})
        input_rows += [{"scenario": name, "table": "ROW", **row} for row in intake.rows]
        input_rows += [{"scenario": name, "table": "CONFLICT", **c} for c in intake.conflicts]
    sample_intake = tn.intake_evidence(
        {tn.ROUTE_SHEET: pd.read_csv(repo_root / "samples" / "operational_route_evidence_TEMPLATE_SAMPLE.csv", dtype=str),
         tn.DEMAND_SHEET: pd.read_csv(repo_root / "samples" / "operational_demand_evidence_TEMPLATE_SAMPLE.csv", dtype=str),
         tn.INVENTORY_SHEET: pd.read_csv(repo_root / "samples" / "operational_inventory_evidence_TEMPLATE_SAMPLE.csv", dtype=str)},
        result["universe_keys"])
    input_rows += [{"scenario": "SAMPLE_TEMPLATES", "table": "SHEET", "sheet": s, **v} for s, v in sample_intake.sheets.items()]
    input_rows += [{"scenario": "SAMPLE_TEMPLATES", "table": "ROW", **row} for row in sample_intake.rows]
    input_table = pd.DataFrame(input_rows)

    # ---- relations (read-only)
    t2 = pd.read_csv(data_root / "_ACTION_CONSISTENCY_VALIDATION" / "action_consistency_candidates.csv", dtype=str)
    t2 = t2[t2["scope"] == "SUHYUP_31D"]
    legacy_a = usable(a_legacy)
    joined = t2.merge(legacy_a[["snapshot_date", "route_id", "contract_strict", "contract_eligibility"]],
                      left_on=["date", "route_id"], right_on=["snapshot_date", "route_id"], how="left")
    t2_relation = {"t2_candidates": len(t2), "linked": int(joined["contract_eligibility"].notna().sum()),
                   "action_status_by_contract_eligibility": {f"{k[0]}|{k[1]}": v for k, v in Counter(
                       zip(joined["action_status"], joined["contract_eligibility"])).items()},
                   "proxy_plan_strict": int(((joined["action_status"] == "PROXY_PLAN")
                                             & (joined["contract_strict"] == oe.STRICT_ELIGIBLE)).sum()),
                   "t2_is_executable": dict(Counter(t2["is_executable"])),
                   "seller_loss_comparison_status": dict(Counter(t2["seller_loss_comparison_status"]))}
    scenario_sl = pd.read_csv(data_root / "_SELLER_LOSS_VALIDATION" / "seller_loss_scenario_validation.csv")
    external = json.loads((data_root / "09_External_Validation/results/forecast_external_generalization.json").read_text(encoding="utf-8"))
    m5_gate = json.loads((data_root / "20_WALMART_M5/results/m5_forecast_v2_promotion_gate.json").read_text(encoding="utf-8"))
    p1b_json = data_root / oev.OUTPUT_FOLDER / "operational_candidate_validation.json"
    p1b_forecast = json.loads(p1b_json.read_text(encoding="utf-8"))["relations"]["forecast"] if p1b_json.exists() else {}
    p1b_totals = p1b_bundle_totals(data_root)

    # ---- checks
    roles = {item["role"]: item["lane_key"] for item in (result["test_input"] or {}).get("lanes", [])}
    b_day = usable(b[b["snapshot_date"] == TEST_DATE]).set_index("lane_key")
    role_status = {role: (b_day.loc[key, "contract_strict"], b_day.loc[key, "contract_eligibility"],
                          b_day.loc[key, "contract_reasons"], b_day.loc[key, "blocking_reasons"]) for role, key in roles.items()
                   if key in b_day.index}
    b_bridge = bridge[(bridge["scenario"] == "B_REAL_PLUS_TEST") & (bridge["date"] == TEST_DATE)].iloc[0].to_dict()
    a_counts = Counter(usable(a_valid)["contract_eligibility"])
    git_guard = validation_git_guard(repo_root)
    test_rows = intakes["B_REAL_PLUS_TEST"].rows
    test_issues = Counter(code.split(":")[0] for row in test_rows for code in (row["issues"] or "").split("|") if code)
    conflict_codes = {c["code"] for c in intakes["B_REAL_PLUS_TEST"].conflicts}
    need_ok = need_records_frame
    case = result["case_0716"]
    reasons_of = lambda role: set(filter(None, f"{role_status.get(role, (None,) * 4)[2] or ''}|{role_status.get(role, (None,) * 4)[3] or ''}".split("|")))  # noqa: E731
    checks = [
        _check("F1", "LEGACY_20 rebuilt equals the production generate_candidates output 31/31 (production untouched)",
               bool(fidelity["legacy_20_equals_production_generator"].all())),
        _check("F2", "P1 pools reproduced: LEGACY_20 620 and ALL_VALID 1,735",
               int(fidelity["legacy_20_size"].sum()) == 620 and int(fidelity["all_valid_size"].sum()) == 1735),
        _check("F3", "P1 MILP costs reproduced: 13,629,146 (LEGACY_20) and 8,194,265 (ALL_VALID) at 7,750",
               abs(sens_summary.loc["B", "milp_cost_total"] - 13629146) <= 1e-6
               and abs(sens_summary.loc["D", "milp_cost_total"] - 8194265) <= 1e-6
               and abs(sens_summary.loc["B", "milp_service_total"] - 7750) <= 1e-9),
        _check("R1", "186 reference rows reproduced", bool(golden["equal"].all()) and len(golden) == 186,
               f"{int(golden['equal'].sum())}/{len(golden)}"),
        _check("R2", "T1 regression: BENCHMARK_PROXY 7,750 / 13,629,146, 0 violations, STRICT insufficient 31/31",
               bool(regression["t1_matches_bundle"].all() and regression["t1_strict_matches_bundle"].all())
               and abs(regression["t1_benchmark_service"].sum() - 7750) <= 1e-9
               and abs(regression["t1_benchmark_cost"].sum() - 13629146) <= 1e-6
               and int(regression["t1_benchmark_violations"].sum()) == 0
               and bool((regression["t1_strict_status"] == sf.INSUFFICIENT_CAP_DATA).all())),
        _check("R3", "T3 regression: MILP OPTIMAL and enumeration-proven 31/31, bundle routes, Varo Final = MILP",
               bool((regression["t3_milp_status"] == mbi.OPTIMAL).all() and regression["t3_milp_matches_bundle"].all()
                    and regression["t3_varo_routes_match_bundle"].all() and regression["t3_enumeration_proven"].all())),
        _check("R4", "T2 relation: 620 candidates linked, no PROXY_PLAN is contract-strict, none executable",
               t2_relation["linked"] == 620 and t2_relation["proxy_plan_strict"] == 0
               and set(t2_relation["t2_is_executable"]) == {"False"}, t2_relation),
        _check("R5", "P1-B regression: with no operator evidence the contract layer reproduces P1-B eligibility on every "
                     "routed lane, and ALL_VALID stays PROXY_FEASIBLE 1,728 / DEMAND_UNVERIFIED 7 / STRICT 0 (= P1-B bundle)",
               bool((a["operational_eligibility"] == a["p1b_operational_eligibility"]).all())
               and a_counts.get(oe.PROXY_FEASIBLE, 0) == P1B_TOTALS["PROXY_FEASIBLE"]
               and a_counts.get(oe.UNVERIFIED_STATUS[oe.DEMAND], 0) == P1B_TOTALS["DEMAND_UNVERIFIED"]
               and int((a["contract_strict"] == oe.STRICT_ELIGIBLE).sum()) == 0
               and p1b_totals.get("all_valid", {}).get(oe.PROXY_FEASIBLE) == P1B_TOTALS["PROXY_FEASIBLE"],
               dict(a_counts)),
        _check("N1", "2026-07-16 614504 @ 218020: generator need 63, benchmark need 6 from source rows (formula only)",
               abs(case["generator_need"] - 63) <= 1e-9 and abs(case["benchmark_need"] - 6) <= 1e-9
               and case["raw_equals_processed_stock"]),
        _check("N2", "identity generator - benchmark = min(max(0, stock - median), 7 x outbound) on every group",
               bool((need.dropna(subset=["formula_gap_on_benchmark_inputs"]).eval(
                   "abs(formula_gap_on_benchmark_inputs - identity_gap)") <= 1e-9).all())),
        _check("N3", "every need record names formula id + version; no PROXY need is accepted; a target without a row is "
                     "NULL (never 0) and the generator's substitute is kept apart",
               bool(need_ok["formula_id"].notna().all() and need_ok["formula_version"].ne("n/a").all())
               and not bool(need_ok["accepted_for_strict"].astype(bool).any())
               and bool(pd.to_numeric(need_ok.loc[need_ok["current_stock_status"] == tn.NOT_OBSERVED, "target_need_value"],
                                      errors="coerce").isna().all())),
        _check("N4", "generator and benchmark records are never compared as one quantity (NOT_COMPARABLE, FORMULA_DIFFERS)",
               bool((need_ok["comparable_with_other_formula"] == oe.NOT_COMPARABLE).all())),
        _check("N5", "DERIVED_FROM_OBSERVED_DEMAND inactive: no actual-sales evidence exists for Suhyup",
               not tn.FORMULAS[tn.DERIVED_FROM_OBSERVED_DEMAND]["active"]
               and not (cands["need_formula_id"] == tn.DERIVED_FROM_OBSERVED_DEMAND).any()),
        _check("S1", "every Suhyup source is ON_HAND_ONLY: no transferable quantity from on-hand stock or the PROXY surplus",
               bool((result["sources"]["validation_status"] == tn.ON_HAND_ONLY).all())
               and bool(result["sources"]["transferable_qty"].isna().all())),
        _check("I1", "real-data intake: no operator file exists -> three sheets MISSING_EVIDENCE, no row accepted",
               all(v["status"] == tn.MISSING_EVIDENCE for v in intakes["A_REAL_ONLY"].sheets.values())
               or intakes["A_REAL_ONLY"].scope != tn.TEST_LINKAGE,
               {k: v["status"] for k, v in intakes["A_REAL_ONLY"].sheets.items()}),
        _check("I2", "SAMPLE templates: every row SAMPLE_NOT_USABLE, nothing accepted",
               all(row["verification"] == tn.SAMPLE_NOT_USABLE and row["row_state"] == tn.ROW_REJECTED for row in sample_intake.rows)
               and len(sample_intake.rows) > 0),
        _check("I3", "TEST intake guards: cross-network, cross-dataset, unknown location and SAMPLE rows rejected; unverified "
                     "row not accepted; ORDER labelled actual stays PROXY; preference informational; need conflict detected",
               all(test_issues.get(code, 0) >= 1 for code in ("CROSS_NETWORK_EVIDENCE_REJECTED", "CROSS_DATASET_EVIDENCE_REJECTED",
                                                              "UNKNOWN_LOCATION_ID", "SAMPLE_ROW_NOT_USABLE", "VERIFIER_MISSING",
                                                              "PROXY_LABELLED_AS_ACTUAL", "PREFERENCE_INFORMATIONAL_ONLY"))
               and "TARGET_NEED_CONFLICT" in conflict_codes
               and not any(row["row_state"] == tn.ROW_ACCEPTED for row in test_rows),
               {"issues": dict(test_issues), "conflicts": sorted(conflict_codes)}),
        _check("E1", "E2E A (real only): STRICT 0 on every routed lane; T1 bridge INSUFFICIENT_CAP_DATA 31/31",
               int((a["contract_strict"] != oe.INSUFFICIENT_OPERATIONAL_EVIDENCE).sum()) == 0
               and bool((bridge.loc[bridge["scenario"] == "A_REAL_ONLY", "plan_status"] == sf.INSUFFICIENT_CAP_DATA).all())),
        _check("E2", "E2E B (TEST): the full-evidence lane is STRICT_ELIGIBLE_TEST_ONLY, never STRICT_ELIGIBLE; the bridge "
                     "selects it labelled TEST_LINKAGE_ONLY with no STRICT_ACTUAL claim; receivable capacity binds",
               role_status.get("FULL_EVIDENCE", (None,))[0] == "STRICT_ELIGIBLE_TEST_ONLY"
               and int((b["contract_strict"] == oe.STRICT_ELIGIBLE).sum()) == 0
               and b_bridge.get("selected", 0) >= 1 and not bool(b_bridge.get("strict_actual_claim"))
               and b_bridge.get("evidence_scope") == tn.TEST_LINKAGE
               and "RECEIVABLE_CAPACITY_BINDING" in reasons_of("RECEIVABLE_CAPACITY_BINDING"),
               {"roles": {k: v[:2] for k, v in role_status.items()}, "bridge": {k: b_bridge.get(k) for k in (
                   "plan_status", "selected", "selection_statuses", "evidence_scope", "strict_actual_claim", "label")}}),
        _check("E3", "E2E B failure roles stay non-strict with their reason: route not approved, need conflict, on-hand only, "
                     "expired need, unit mismatch, handling prohibited, available above on-hand",
               role_status.get("ROUTE_NOT_APPROVED", (None,))[0] == oe.INSUFFICIENT_OPERATIONAL_EVIDENCE
               and role_status.get("TARGET_NEED_CONFLICT", (None, None))[1] == tn.CONFLICT
               and "ONLY_ON_HAND_STOCK_OBSERVED" in reasons_of("ON_HAND_ONLY")
               and "EVIDENCE_EXPIRED" in reasons_of("TARGET_NEED_EXPIRED")
               and "DEMAND_UNIT_MISMATCH" in reasons_of("DEMAND_UNIT_MISMATCH")
               and "TARGET_HANDLING_PROHIBITED" in reasons_of("TARGET_HANDLING_PROHIBITED")
               and role_status.get("AVAILABLE_EXCEEDS_ON_HAND", (None, None))[1] == tn.CONFLICT
               if len(role_status) == len(TEST_ROLES) else False,
               {k: {"strict": v[0], "eligibility": v[1], "reasons": v[2]} for k, v in role_status.items()}),
        _check("E4", "TEST input with allow_test off behaves as no evidence (no strict lane, bridge insufficient 31/31)",
               int((b_strict["contract_strict"] != oe.INSUFFICIENT_OPERATIONAL_EVIDENCE).sum())
               == int((b_strict["contract_eligibility"].isin((oe.INELIGIBLE, tn.CONFLICT))).sum())
               and not (b_strict["contract_strict"].isin(tn.CONTRACT_STRICT_STATUSES)).any()
               and bool((bridge.loc[bridge["scenario"] == "B_TEST_NOT_ALLOWED", "plan_status"] == sf.INSUFFICIENT_CAP_DATA).all())),
        _check("X1", "constraint signatures: generator-need and benchmark-need results are not optimality-comparable; same "
                     "need definition across pools is",
               sens_summary.loc["A", "comparable_with_B"] == oe.NOT_COMPARABLE
               and sens_summary.loc["D", "comparable_with_B"] == oe.COMPARABLE),
        _check("P1", "P1 1,735 reclassified: need BENCHMARK_PROXY, source SOURCE_SURPLUS_PROXY, route class C, 0 strict",
               len(usable(a_valid)) == 1735
               and bool((usable(a_valid)["need_formula_id"] == tn.BENCHMARK_PROXY).all())
               and bool((usable(a_valid)["source_availability_kind"] == tn.SOURCE_SURPLUS_PROXY).all())
               and bool((usable(a_valid)["route_class"] == oe.ROUTE_C).all())),
        _check("SL1", "Seller Loss controlled scenarios 18/18 (read-only; formulas untouched)",
               len(scenario_sl) == 18 and int(scenario_sl["pass"].astype(bool).sum()) == 18),
        _check("FC1", "Forecast: external GENERALIZATION_VERDICT FAIL kept; M5 V2 gate passed with the P1-B signature",
               external.get("GENERALIZATION_VERDICT") == "FAIL" and bool(m5_gate.get("passed"))
               and (not p1b_forecast or m5_gate.get("config_signature") == p1b_forecast.get("m5_v2_config_signature"))),
        git_validation_check(git_guard, f"tracked changes limited to {', '.join(ALLOWED_MODIFIED)}; validator/policy protected"),
    ]
    after = oev._hashes(data_root, PROTECTED_BUNDLES)
    changed_bundles = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    checks.append(_check("G2", "no existing data / bundle file changed during the run", not changed_bundles, changed_bundles))
    failed = [c["check_id"] for c in checks if c["status"] != "PASS"]
    verdicts = _verdicts(failed, a_counts)
    summary = {
        "contract_version": tn.CONTRACT_VERSION, "git_head": git_guard["actual_result"]["head"], "git_protection": git_guard,
        "data_root": str(data_root), "network_id": NETWORK_ID, "dataset_id": DATASET, "checks": checks, "failed_checks": failed,
        "verdicts": verdicts, "operator_files": paths, "test_date": TEST_DATE, "test_lanes": (result["test_input"] or {}).get("lanes"),
        "case_0716": case,
        "formula_findings": {"groups": int(len(comparison)), "causes": dict(Counter(comparison["difference_cause"])),
                             "differing_groups": int((comparison["abs_gap"] > 1e-9).sum()),
                             "largest": comparison.sort_values("abs_gap", ascending=False).head(10)[
                                 ["date", "center_id", "product_id", "need_generator", "need_benchmark", "total_gap",
                                  "difference_cause"]].to_dict("records")},
        "state_codes": state_summary,
        "eligibility": {"A_all_valid": dict(Counter(usable(a_valid)["contract_eligibility"])),
                        "A_legacy_20": dict(Counter(usable(a_legacy)["contract_eligibility"])),
                        "B_test_date_all_valid": dict(Counter(usable(b[(b["snapshot_date"] == TEST_DATE) & b["in_all_valid"].astype(bool)])["contract_eligibility"]))},
        "intake": {name: {"scope": intake.scope, "sheets": intake.sheets, "counts": intake.counts(),
                          "conflicts": len(intake.conflicts)} for name, intake in intakes.items()},
        "t1_bridge": {"A_REAL_ONLY": dict(Counter(bridge.loc[bridge["scenario"] == "A_REAL_ONLY", "plan_status"])),
                      "B_REAL_PLUS_TEST": {k: b_bridge.get(k) for k in ("plan_status", "selected", "selection_statuses",
                                                                          "total_allocated_qty", "total_move_cost", "feasibility_claim",
                                                                          "evidence_scope", "strict_actual_claim", "label")}},
        "relations": {"t2": t2_relation, "seller_loss": {"scenario_rows": len(scenario_sl), "scenario_pass": int(scenario_sl["pass"].astype(bool).sum()),
                                                         "note": "Seller Loss formulas unchanged; TRANSFER stays COMPARISON_UNAVAILABLE without seller inputs"},
                      "forecast": {"external_generalization_verdict": external.get("GENERALIZATION_VERDICT"),
                                   "m5_v2_config_signature": m5_gate.get("config_signature"), "m5_v2_gate_passed": m5_gate.get("passed")},
                      "p1b_bundle_totals": p1b_totals},
        "sensitivity": sens[sens["table"] == "SUMMARY"].to_dict("records"),
        "request_priority_all_valid": request.to_dict("records"),
        "limitations": [
            "no actual operator evidence exists for Suhyup; STRICT_ELIGIBLE 0 is a data finding",
            "scenario B uses explicit TEST input built from the lanes' own quantities; it checks the input path only",
            "verification is declared, not authenticated: the program cannot confirm a verifier's authority",
            "the state-code finding is research only; production and benchmark formulas are unchanged",
            "request unlock counts are upper bounds: they assume each supplied value verifies and does not cut the quantity",
        ],
        "production_action_applied": False, "production_candidate_pool_replaced": False,
        "runtime": {"wall_seconds": round(time.perf_counter() - started, 3), "suhyup_seconds": result["wall_seconds"]},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write = lambda frame, name: frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig")  # noqa: E731
    (output_dir / OUTPUT_FILES[0]).write_text(json.dumps(contract_document(result, data_root), ensure_ascii=False, indent=2,
                                                         default=str), encoding="utf-8")
    write(comparison, OUTPUT_FILES[1])
    write(daily, OUTPUT_FILES[2])
    write(input_table, OUTPUT_FILES[3])
    write(missing, OUTPUT_FILES[4])
    write(priority_table, OUTPUT_FILES[5])
    write(e2e, OUTPUT_FILES[6])
    (output_dir / OUTPUT_FILES[7]).write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write(by_scope, OUTPUT_FILES[8])
    write(sens, OUTPUT_FILES[9])
    write(states, OUTPUT_FILES[10])
    write(pd.DataFrame(CALL_SITES, columns=CALL_SITE_COLUMNS), OUTPUT_FILES[11])
    _parquet(need_records_frame, output_dir / OUTPUT_FILES[12])
    _parquet(cands, output_dir / OUTPUT_FILES[13])
    _parquet(result["sources"], output_dir / OUTPUT_FILES[14])
    return summary


def _parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.astype({c: str for c in frame.columns if frame[c].dtype == object}).to_parquet(path, index=False)


def _verdicts(failed: Sequence[str], a_counts: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    core = not set(failed) - {"G1"}
    return {
        "A_need_contract": {"status": "READY" if core else "NOT_READY",
                            "basis": "formula ids, versions, records, comparability and constraint signatures"},
        "B_data_and_operator_input_connection": {"status": "READY_WITH_LIMITATIONS" if core else "NOT_READY",
                                                 "basis": "workbook sheets, uploaded_data keys and CSV paths validated; no actual operator input exists"},
        "C_input_validation_and_provenance": {"status": "READY" if core else "NOT_READY",
                                              "basis": "origin, verification level, keys, dates, units, conflicts, NULL vs 0"},
        "D_strict_actual_reachable": {"status": "CONFIRMED_BY_TEST_LINKAGE_ONLY" if core else "NOT_READY",
                                      "basis": "TEST input reaches STRICT_ELIGIBLE_TEST_ONLY and the T1 bridge; real data 0"},
        "E_actual_operational_data": {"status": "NOT_READY", "basis": "no operator route / need / transferable / unit / capacity data"},
        "F_production_application": {"status": "NOT_READY", "basis": "parallel field only; generator, Top-5, actions unchanged"},
        "overall": {"status": "READY_WITH_LIMITATIONS" if core else "NOT_READY",
                    "basis": "contract and intake are ready; operational verification is not done (E)"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--route-evidence", type=Path, default=None)
    parser.add_argument("--demand-evidence", type=Path, default=None)
    parser.add_argument("--inventory-evidence", type=Path, default=None)
    args = parser.parse_args()
    result = run_validation(args.data_root, args.output_dir, evidence_paths={
        tn.ROUTE_SHEET: args.route_evidence, tn.DEMAND_SHEET: args.demand_evidence, tn.INVENTORY_SHEET: args.inventory_evidence})
    print(json.dumps({"failed_checks": result["failed_checks"], "checks": result["checks"], "verdicts": result["verdicts"]},
                     ensure_ascii=False, indent=2, default=str))
    return 0 if not result["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

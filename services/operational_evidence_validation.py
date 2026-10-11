"""Operational evidence and candidate eligibility on the real Suhyup 31-day benchmark (P1-B).

Validation only.  It reads the actual inventory flow, the processed 2026-07-31 upload, the saved 620 candidates, the
186-row reference bundle and the T1 / T2 / T3 / P1 / Seller Loss bundles, and never writes there.  Outputs go to
<data-root>/_OPERATIONAL_EVIDENCE_VALIDATION (local only, never into git); a re-run replaces only OUTPUT_FILES.

Per day it rebuilds the P1 lane universe with the unchanged research code, classifies every routed lane with
``operational_evidence.evaluate_lane`` (operator evidence: none exists for Suhyup unless supplied with --route-evidence
/ --demand-evidence), measures both target-need formulas on every (date, center, product), re-runs the pools under
each need definition (separate results; production untouched), and cross-checks the eligibility against T1.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import subprocess
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from services import candidate_generation_validation as cgv
from services import candidate_generator as cg
from services import milp_benchmark_integrity as mbi
from services import milp_benchmark_integrity_validation as mbv
from services import operational_evidence as oe
from services import shared_feasibility_selection as sf
from services import shared_feasibility_validation as sfv
from services import suhyup_algorithm_revalidation as rv
from services.real_data_adapters import DATA_ROOT
from services.seller_decision_validation import _env

# The P1 research functions (lane universe, pools, evaluate_pool) are reached through the P1 validation module, the one
# validation entry point allowed to use the research path; no production module imports either.
cgr = cgv.cgr

OUTPUT_FOLDER = "_OPERATIONAL_EVIDENCE_VALIDATION"
OUTPUT_FILES = (
    "operational_candidate_evidence_summary.csv",
    "operational_candidate_evidence_by_day.csv",
    "operational_candidate_eligibility.csv",
    "operational_route_evidence.csv",
    "operational_demand_semantics.csv",
    "target_need_formula_comparison.csv",
    "operational_missing_inputs.csv",
    "operational_candidate_validation.json",
    "operational_dataset_survey.csv",
    "need_formula_sensitivity.csv",
    "operational_candidate_eligibility.parquet",
)
NETWORK_ID = "suhyup_logistics_centers"
DATASET = mbv.DATASET
FLOW_FILE = "02_Korea_Suhyup_Logistics/processed/suhyup_logistics_inventory_flow_actual.csv"
RAW_STOCK = "02_Korea_Suhyup_Logistics/raw/해양수산부_수협물류센터공판장품목별재고현황_20260731.CSV"
RAW_FLOW = "02_Korea_Suhyup_Logistics/raw/해양수산부_수협물류센터공판장품목별입출고현황_20260731.CSV"
CANONICAL = "02_Korea_Suhyup_Logistics/processed"
OSRM = "10_Road_Distance_Time/processed/suhyup_6_actual_centers_road_distance_time_osrm.csv"
TARIFF = "13_Transport_Cost_Official/processed/suhyup_6_actual_center_route_vehicle_official_cost_matrix.csv"
VEHICLE_CLASSES = "12_Vehicle_Capacity/processed/varo_vehicle_capacity_classes.csv"
UNIT_WEIGHT = "14_Product_Unit_Weight/processed/suhyup_product_unit_weight_varo_final.csv"
CENTER_MASTER = "09_VARO_Mapping/suhyup_actual_center_location_master.csv"
TRANSFER_HISTORY = "14_Actual_Transfer_History"
OPERATOR_FOLDER = "02_Korea_Suhyup_Logistics/operator_inputs"
STATE_TEMPERATURE = {"20": "CHILLED", "30": "FROZEN", "32": "FROZEN", "40": "AMBIENT"}
TEMPERATURE_RANK = {"FROZEN": 0, "CHILLED": 1, "AMBIENT": 2}
P1_BASELINE_COST = {cgr.LEGACY_20: 13629146.0, cgr.ALL_VALID: 8194265.0}
P1_POOL_TOTAL = {cgr.LEGACY_20: 620, cgr.ALL_VALID: 1735}
SERVICE_TOTAL = 7750.0
FORMULA_G, FORMULA_B = oe.GENERATOR_FORMULA, oe.BENCHMARK_FORMULA
COMBOS = (
    # combo, pool, need definition used as the TARGET_NEED cap, description
    ("A", "LEGACY_20", FORMULA_G, "production pool, generator need as the target cap"),
    ("B", "LEGACY_20", FORMULA_B, "production pool, benchmark need (P1 / T1 baseline)"),
    ("C", "ALL_VALID_G", FORMULA_G, "every lane valid when the generator need is the cap"),
    ("D", "ALL_VALID", FORMULA_B, "every lane valid under the benchmark need (P1 ALL_VALID)"),
    ("C_FIXED_POOL", "ALL_VALID", FORMULA_G, "P1's 1,735-lane pool with the generator need as the cap"),
    ("E_REGEN_B", "LEGACY_20_REGEN_B", FORMULA_B,
     "generator re-run with the benchmark formula (generator inputs), benchmark need as the cap: one definition end to end"),
)
MAPPING_DOC = {" <-> ".join(pair): rule for pair, rule in oe.VERIFIED_KEY_MAPPINGS.items()}
PROTECTED_BUNDLES = ("_CANDIDATE_GENERATION_VALIDATION", "_SHARED_FEASIBILITY_VALIDATION", "_MILP_BENCHMARK_INTEGRITY",
                     "_ACTION_CONSISTENCY_VALIDATION", "_SELLER_LOSS_VALIDATION",
                     "21_SUHYUP_ALGORITHM_REVALIDATION_20260919", "16_VARO_E2E_20260731", "02_Korea_Suhyup_Logistics",
                     "09_External_Validation/results", "20_WALMART_M5/results")


# ------------------------------------------------------------------------------------------------ helpers


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True).stdout.rstrip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _check(check_id: str, description: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"check_id": check_id, "description": description, "status": "PASS" if ok else "FAIL", "detail": detail}


def _hashes(root: Path, folders: Sequence[str]) -> dict[str, str]:
    out = {}
    for folder in folders:
        base = root / folder
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            out[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ------------------------------------------------------------------------------------------------ Suhyup context


class SuhyupEvidence:
    """Real-data evidence of the Suhyup network that does not change per lane (temperature, unit weight, history)."""

    def __init__(self, data_root: Path, flow: pd.DataFrame, route_records: Sequence[Mapping[str, Any]] = (),
                 observed: Sequence[Mapping[str, Any]] = (), quantity_records: Sequence[Mapping[str, Any]] = ()) -> None:
        nodes = flow[flow["center_code"].isin(rv.NETWORK_NODES)]
        temperature: dict[str, str] = {}
        for product, states in nodes.groupby("product_code")["state_code"]:
            levels = sorted({STATE_TEMPERATURE.get(str(s), "AMBIENT") for s in states}, key=TEMPERATURE_RANK.get)
            temperature[str(product)] = levels[0]
        self.temperature = temperature
        weight = pd.read_csv(data_root / UNIT_WEIGHT, dtype={"product_code": str}, encoding="utf-8-sig")
        self.unit_weight_status = dict(zip(weight["product_code"], weight["varo_unit_weight_status"].astype(str)))
        seen: dict[tuple[str, str], list[str]] = defaultdict(list)
        for (center, product), dates in nodes.groupby(["center_code", "product_code"])["date"]:
            seen[(str(center), str(product))] = sorted(set(dates.astype(str)))
        self.seen = seen
        self.route_records, self.observed, self.quantity_records = list(route_records), list(observed), list(quantity_records)

    def history(self, date: str) -> dict[tuple[str, str], str]:
        out = {}
        for key, dates in self.seen.items():
            position = bisect.bisect_left(dates, date)
            if position:
                out[key] = dates[position - 1]
        return out

    def context(self, date: str, **overrides: Any) -> oe.OperationalContext:
        base = dict(network_id=NETWORK_ID, dataset_id=DATASET, route_records=self.route_records,
                    observed_transfers=self.observed, quantity_records=self.quantity_records,
                    product_temperature=self.temperature, handling_history=self.history(date),
                    unit_weight_status=self.unit_weight_status)
        base.update(overrides)
        return oe.OperationalContext(**base)


# ------------------------------------------------------------------------------------------------ need formulas


def need_groups(rebuild: mbv.GeneratorRebuild, flow: pd.DataFrame, date: str, universe: pd.DataFrame,
                pools: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Both need formulas on both input namespaces for every (date, center, product) of the day (section 9)."""
    upload = rebuild.upload(date)["inventory"]
    demand_column = next(c for c in ("avg_daily_sales", "sales_qty", "demand_qty") if c in upload.columns)
    gen = pd.DataFrame({"center": upload["store_id"].astype(str), "product": upload["product_id"].astype(str),
                        "stock": upload["stock_qty"].map(cg._num), "demand": upload[demand_column].map(cg._num)})
    gen_median = gen.groupby("product")["stock"].median().to_dict()
    gen_row = {(r.center, r.product): (r.stock, r.demand) for r in gen.itertuples()}
    lookup, medians = rv._inventory_lookup(flow[flow["date"] == date])
    products = set(rebuild.products["product_id"].astype(str))
    day_flow = flow[(flow["date"] == date) & flow["center_code"].isin(rv.NETWORK_NODES) & flow["product_code"].isin(products)]
    state_rows = day_flow.groupby(["center_code", "product_code"]).size().to_dict()
    lanes_by = universe[universe["recommended_qty"].notna()].groupby(["target_id", "product_id"]).size().to_dict()
    pool_by = {name: pool.groupby(["target_id", "product_id"]).size().to_dict() if not pool.empty else {}
               for name, pool in pools.items()}
    rows = []
    for center in rv.NETWORK_NODES:
        for product in sorted(products):
            bench = lookup.get((date, center, product))
            if bench is None and (center, product) not in gen_row:
                continue
            m_b = medians.get((date, product))
            m_g = gen_median.get(product)
            g_stock, g_demand = gen_row.get((center, product), (m_g, 0.0)) if m_g is not None else (None, None)
            row = {"date": date, "center_id": center, "product_id": product,
                   "flow_state_rows": int(state_rows.get((center, product), 0)),
                   "generator_row_present": (center, product) in gen_row,
                   "excluded_duplicate_state_pair": int(state_rows.get((center, product), 0)) > 1,
                   "generator_median": m_g, "benchmark_median": m_b,
                   "generator_stock": g_stock, "benchmark_stock": bench["stock"] if bench else None,
                   "generator_demand": g_demand, "benchmark_demand": bench["outbound"] if bench else None}
            if m_g is not None and g_stock is not None:
                row["need_generator"] = oe.generator_need(m_g, g_stock, g_demand)
                row["benchmark_formula_on_generator_inputs"] = oe.benchmark_need(m_g, g_stock, g_demand)
            if bench is not None and m_b is not None:
                row["need_benchmark"] = oe.benchmark_need(m_b, bench["stock"], bench["outbound"])
                row["generator_formula_on_benchmark_inputs"] = oe.generator_need(m_b, bench["stock"], bench["outbound"])
                row["formula_gap_on_benchmark_inputs"] = row["generator_formula_on_benchmark_inputs"] - row["need_benchmark"]
                row["identity_gap"] = oe.need_formula_gap(m_b, bench["stock"], bench["outbound"])
                row["need_case"] = oe.need_case(m_b, bench["stock"], bench["outbound"])
            if "need_generator" in row and "need_benchmark" in row:
                row["total_gap"] = row["need_generator"] - row["need_benchmark"]
                row["input_namespace_effect"] = row["need_generator"] - row["generator_formula_on_benchmark_inputs"]
                row["inputs_identical"] = (abs((m_g or 0) - (m_b or 0)) <= 1e-9 and abs((g_stock or 0) - bench["stock"]) <= 1e-9
                                           and abs((g_demand or 0) - bench["outbound"]) <= 1e-9 and row["generator_row_present"])
            row["lanes_targeting"] = int(lanes_by.get((center, product), 0))
            for name, counts in pool_by.items():
                row[f"lanes_in_{name.lower()}"] = int(counts.get((center, product), 0))
            rows.append(row)
    return pd.DataFrame(rows)


def reproduce_0716(data_root: Path, flow: pd.DataFrame, universe: pd.DataFrame) -> dict[str, Any]:
    """The 2026-07-16 V2C004 case from source rows: product 614504, source 156, target 218020 (section 9)."""
    date, product, target, source = "2026-07-16", "614504", "218020", "156"
    rows = flow[(flow["date"] == date) & (flow["product_code"] == product) & flow["center_code"].isin(rv.NETWORK_NODES)]
    raw = {}
    for name, path in (("raw_stock_file", RAW_STOCK), ("raw_flow_file", RAW_FLOW)):
        frame = pd.read_csv(data_root / path, encoding="cp949", dtype=str)
        frame.columns = [str(c).strip() for c in frame.columns]
        code, day, item = frame.columns[0], frame.columns[2], frame.columns[3]
        # raw center codes are zero-padded (000156); the verified mapping rule strips the zeros
        centers = frame[code].map(lambda value: oe.normalize_location_id(value, "suhyup_canonical_location_id"))
        hit = frame[(frame[day] == date) & (frame[item].str.strip() == product) & centers.isin(rv.NETWORK_NODES)]
        raw[name] = hit.to_dict("records")
    stocks = rows.groupby("center_code")["stock_qty"].sum().astype(float)
    stock_column = next(c for c in pd.DataFrame(raw["raw_stock_file"]).columns if c.strip() == "재고량") if raw["raw_stock_file"] else None
    raw_stocks = defaultdict(float)
    for row in raw["raw_stock_file"]:
        raw_stocks[oe.normalize_location_id(row[next(iter(row))], "suhyup_canonical_location_id")] += float(row[stock_column])
    median = float(stocks.median())
    target_row = rows[rows["center_code"] == target]
    stock, outbound = float(target_row["stock_qty"].sum()), float(target_row["outbound_qty"].sum())
    lane = universe[(universe["product_id"] == product) & (universe["source_id"] == source) & (universe["target_id"] == target)]
    lane = lane.iloc[0] if not lane.empty else None
    g, b = oe.generator_need(median, stock, outbound), oe.benchmark_need(median, stock, outbound)
    return {
        "date": date, "product_id": product, "source_id": source, "target_id": target,
        "processed_flow_rows": rows[["center_code", "state_code", "stock_qty", "inbound_qty", "outbound_qty"]].to_dict("records"),
        "raw_rows": raw,
        "node_stocks": {k: float(v) for k, v in stocks.items()}, "raw_node_stocks": dict(raw_stocks),
        "raw_equals_processed_stock": dict(raw_stocks) == {k: float(v) for k, v in stocks.items()}, "median_stock": median,
        "target_stock": stock, "target_outbound_qty": outbound,
        "generator_need_steps": f"max(0, {median:g} - {stock:g}) + 7 x {outbound:g} = {max(0.0, median - stock):g} + {7 * outbound:g} = {g:g}",
        "benchmark_need_steps": f"max(0, {median:g} + 7 x {outbound:g} - {stock:g}) = max(0, {median + 7 * outbound - stock:g}) = {b:g}",
        "generator_need": g, "benchmark_need": b, "gap": g - b,
        "identity_gap": oe.need_formula_gap(median, stock, outbound), "case": oe.need_case(median, stock, outbound),
        "lane_route_id": None if lane is None else lane["route_id"], "lane_recommended_qty": None if lane is None else float(lane["recommended_qty"]),
        "lane_gen_target_need": None if lane is None else float(lane["gen_target_need"]),
        "lane_target_need_7d": None if lane is None else float(lane["target_need_7d"]),
        "explanation": (f"target stock {stock:g} is {stock - median:g} above the median {median:g} and the 7-day outbound is "
                        f"{7 * outbound:g}: the generator adds the full {7 * outbound:g} because it never nets stock above the "
                        f"median; the benchmark nets it ({median:g} + {7 * outbound:g} - {stock:g} = {b:g}). The lane's own "
                        "generator and benchmark values equal these, so the gap is the formula, not the data."),
    }


# ------------------------------------------------------------------------------------------------ sensitivity pools


def _with_need(pool: pd.DataFrame, formula: str) -> pd.DataFrame:
    frame = pool.copy()
    if frame.empty:
        return frame
    frame["target_need_7d_benchmark"] = pd.to_numeric(frame["target_need_7d"], errors="coerce")
    if formula == FORMULA_G:
        frame["target_need_7d"] = pd.to_numeric(frame["gen_target_need"], errors="coerce")
    frame["target_need_formula"] = formula
    return frame


def all_valid_generator_pool(universe: pd.DataFrame) -> pd.DataFrame:
    """ALL_VALID when the generator need is the target cap: every check except the benchmark target cap."""
    frame = universe.copy()
    flags = frame["validity_flags"].map(lambda t: set(str(t).split("|")) if isinstance(t, str) and t else set())
    frame["validity_status"] = [cgr.VALID if not (f - {"TARGET_CAP_MISSING_OR_ZERO"}) else s
                                for f, s in zip(flags, frame["validity_status"])]
    frame["validity_flags"] = [None if not (f - {"TARGET_CAP_MISSING_OR_ZERO"}) else t
                               for f, t in zip(flags, frame["validity_flags"])]
    return cgr.policy_pool(frame, cgr.ALL_VALID)


def regenerated_benchmark_pool(universe: pd.DataFrame, store_ids: Sequence[str], product_ids: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(top-20 pool, all choices) of the generator re-run with the benchmark formula on its own inputs, re-priced."""
    frame = universe.copy()
    frame["bench_need_gen_inputs"] = [
        oe.benchmark_need(m, s, d) if None not in (_num(m), _num(s), _num(d)) else None
        for m, s, d in zip(frame["gen_median_stock"], frame["gen_target_stock"], frame["gen_target_demand"])]
    choices = oe.regenerated_choices(frame, "bench_need_gen_inputs")
    pool = choices.head(cg.MAX_CANDIDATES).copy()
    if pool.empty:
        return pool, choices
    pool["generator_rank"] = pool["regenerated_rank"]
    pool["route_id"] = [f"V2B{int(rank):03d}" for rank in pool["regenerated_rank"]]
    pool["recommendation_id"] = pool["route_id"]
    priced = cgr.price_universe(pool.drop(columns=[c for c in ("move_cost", "cost_provenance") if c in pool.columns]),
                                cgr.real_tariff_pricer)
    validated = cgr.validate_universe(priced.drop(columns=[c for c in priced.columns if c.startswith("chk_")]),
                                      store_ids=store_ids, product_ids=product_ids)
    validated["policy"], validated["policy_rank"] = "LEGACY_20_REGEN_B", range(1, len(validated) + 1)
    return validated.reset_index(drop=True), choices


def _plan_excess(rows: Sequence[Mapping[str, Any]], cap_by: Mapping[tuple[str, str], float]) -> float:
    used: defaultdict[tuple[str, str], float] = defaultdict(float)
    for row in rows:
        used[(str(row["target_id"]), str(row["product_id"]))] += float(row["recommended_qty"])
    return round(sum(max(0.0, qty - cap_by.get(key, math.inf)) for key, qty in used.items()), 6)


def run_combo(pool: pd.DataFrame, formula: str, date: str, gen_caps: Mapping, bench_caps: Mapping) -> dict[str, Any]:
    capped = _with_need(pool, formula)
    ev = cgr.evaluate_pool(capped, date=date, dataset=DATASET, cap_spec=sfv.SUHYUP_CAP_SPEC, milp_cap_spec=mbv.MILP_CAP_SPEC,
                           cost_recompute=sf.real_transport_cost_recompute, pool_total=len(capped))
    plans = ev["plans"]
    by_key = {r["lane_key"]: r for r in rv._records(capped)} if not capped.empty else {}
    milp_rows = [by_key[k] for k in plans["MILP"]["lane_keys"] if k in by_key]
    return {"pool_size": len(capped), "excluded": ev["excluded_count"],
            **{f"{p.lower()}_{m}": plans[p][m] for p in ("MILP", "VARO_FINAL", "T1_ALL_OR_NOTHING", "T1_PARTIAL")
               for m in ("service", "cost", "selected_count")},
            "violations": int(sum(plans[p]["check"]["violation_count"] for p in plans)),
            "t1_strict_status": plans["T1_STRICT_ACTUAL"]["plan_status"],
            "milp_status": ev["milp_evidence"]["solver_status"], "milp_enumeration_status": ev["enumeration"]["status"],
            "milp_enumeration_proven": ev["enumeration"]["proven"],
            "milp_lane_keys": "|".join(sorted(plans["MILP"]["lane_keys"])),
            "milp_target_excess_vs_benchmark_need": _plan_excess(milp_rows, bench_caps),
            "milp_target_excess_vs_generator_need": _plan_excess(milp_rows, gen_caps),
            "milp_signature": ev["milp_signature"]["signature"]}


# ------------------------------------------------------------------------------------------------ one day


def run_day(rebuild: mbv.GeneratorRebuild, flow: pd.DataFrame, date: str, evidence: SuhyupEvidence,
            saved_day: pd.DataFrame, saved_recompute: Any) -> dict[str, Any]:
    built = cgv.build_universe(rebuild, flow, date)
    universe, upload = built["lanes"], built["upload"]
    product_ids = upload["products"]["product_id"].astype(str).tolist()
    pools = {name: cgr.policy_pool(universe, name) for name in (cgr.LEGACY_20, cgr.ALL_VALID, cgr.ALL_LANES)}
    routed = universe[universe["recommended_qty"].notna()].reset_index(drop=True)
    context = evidence.context(date)
    records = oe.evaluate_lanes(routed, context)
    keys = {name: set(pool["lane_key"]) if not pool.empty else set() for name, pool in pools.items()}
    extra = routed[["lane_key", "inventory_row", "is_duplicate", "validity_status", "validity_flags", "generator_rank",
                    "gen_target_need", "target_need_7d", "gen_median_stock", "median_stock", "gen_target_stock",
                    "target_stock", "gen_target_demand", "target_daily_demand_proxy", "move_cost", "cost_provenance",
                    "distance_km", "distance_provenance", "route_status", "target_has_inventory_row"]].rename(
        columns={"validity_status": "p1_validity_status", "validity_flags": "p1_validity_flags"})
    records = pd.concat([records, extra.reset_index(drop=True)], axis=1)
    for name in (cgr.LEGACY_20, cgr.ALL_VALID, cgr.ALL_LANES):
        records[f"in_{name.lower()}"] = records["lane_key"].isin(keys[name]) & ~records["is_duplicate"].astype(bool)
    records["need_formula_gap"] = pd.to_numeric(records["gen_target_need"], errors="coerce") - pd.to_numeric(records["target_need_7d"], errors="coerce")

    # pools of the sensitivity matrix (section 29) and the regenerated generator
    regen_pool, regen_choices = regenerated_benchmark_pool(universe, built["store_ids"], product_ids)
    gen_choices = oe.regenerated_choices(universe, "gen_target_need")
    pool_frames = {"LEGACY_20": pools[cgr.LEGACY_20], "ALL_VALID": pools[cgr.ALL_VALID],
                   "ALL_VALID_G": all_valid_generator_pool(universe), "LEGACY_20_REGEN_B": regen_pool}
    usable = universe[universe["recommended_qty"].notna() & ~universe["is_duplicate"].astype(bool)]
    gen_caps = {(str(t), str(p)): float(v) for t, p, v in zip(usable["target_id"], usable["product_id"], usable["gen_target_need"]) if _num(v) is not None}
    bench_caps = {(str(t), str(p)): float(v) for t, p, v in zip(usable["target_id"], usable["product_id"], usable["target_need_7d"]) if _num(v) is not None}
    group_consistent = bool((usable.groupby(["target_id", "product_id"])["gen_target_need"].nunique(dropna=True) <= 1).all())
    combos = []
    for combo, pool_name, formula, description in COMBOS:
        result = run_combo(pool_frames[pool_name], formula, date, gen_caps, bench_caps)
        combos.append({"date": date, "combo": combo, "pool": pool_name, "need_definition": formula,
                       "description": description, **result})
    # generator change under the benchmark formula (section 9: candidate generation change)
    g_choice = gen_choices.set_index("inventory_row")
    b_choice = regen_choices.set_index("inventory_row")
    common = g_choice.index.intersection(b_choice.index)
    target_changed = int((g_choice.loc[common, "target_id"].astype(str) != b_choice.loc[common, "target_id"].astype(str)).sum())
    same_target = common[(g_choice.loc[common, "target_id"].astype(str) == b_choice.loc[common, "target_id"].astype(str)).values]
    qty_changed = int((g_choice.loc[same_target, "recommended_qty"].astype(float) != b_choice.loc[same_target, "recommended_qty"].astype(float)).sum())
    legacy_keys, regen_keys = keys[cgr.LEGACY_20], set(regen_pool["lane_key"]) if not regen_pool.empty else set()
    generation = {"date": date, "generator_rows_with_route": int(len(common)), "target_choice_changed_rows": target_changed,
                  "qty_changed_rows_same_target": qty_changed,
                  "top20_lanes_added": int(len(regen_keys - legacy_keys)), "top20_lanes_removed": int(len(legacy_keys - regen_keys)),
                  "regen_b_qty_total": float(pd.to_numeric(regen_pool.get("recommended_qty"), errors="coerce").sum()) if not regen_pool.empty else 0.0,
                  "legacy_qty_total": float(pd.to_numeric(pools[cgr.LEGACY_20]["recommended_qty"]).sum())}

    # fidelity and regressions
    production, _ = cg.generate_candidates(dict(upload))
    legacy_records = cgr.production_records(pools[cgr.LEGACY_20])
    regen_g = oe.regenerate_with_need(universe, "gen_target_need")
    fidelity = {"date": date,
                "legacy_20_equals_production_generator": production is not None and production.reset_index(drop=True).astype(str).equals(
                    legacy_records[list(production.columns)].astype(str)),
                "regenerate_generator_need_equals_legacy_20": list(regen_g["lane_key"]) == list(pools[cgr.LEGACY_20]["lane_key"])
                and [int(q) for q in regen_g["recommended_qty"]] == [int(q) for q in pools[cgr.LEGACY_20]["recommended_qty"]],
                "generator_need_single_value_per_target_product": group_consistent,
                "upload_avg_daily_sales_is_sales_qty_copy": bool((upload["inventory"]["avg_daily_sales"].map(cg._num)
                                                                  == upload["inventory"]["sales_qty"].map(cg._num)).all()),
                "upload_sales_30d_is_sales_qty_copy": bool((upload["inventory"]["sales_30d"].map(cg._num)
                                                            == upload["inventory"]["sales_qty"].map(cg._num)).all()),
                "legacy_20_size": len(pools[cgr.LEGACY_20]), "all_valid_size": len(pools[cgr.ALL_VALID]),
                "all_lanes_size": len(pools[cgr.ALL_LANES])}
    ranked_saved = sfv.ranked_day(saved_day, date)
    saved_eval = cgr.evaluate_pool(ranked_saved, date=date, dataset=DATASET, cap_spec=sfv.SUHYUP_CAP_SPEC,
                                   milp_cap_spec=mbv.MILP_CAP_SPEC, cost_recompute=saved_recompute,
                                   pool_total=len(ranked_saved), rank=False)

    # independent T1 cross-check on ALL_VALID (section 42)
    all_valid = pools[cgr.ALL_VALID]
    t1_rows = []
    if not all_valid.empty:
        ranked = cgr.rank_pool(all_valid)
        t1_records = rv._records(ranked.reset_index(drop=True))
        caps = sf.caps_from_columns(t1_records, sfv.SUHYUP_CAP_SPEC)
        strict = sf.select_shared_feasible(t1_records, caps, mode=sf.STRICT_ACTUAL, max_routes=None, partial_policy=sf.PARTIAL_NONE)
        bench = sf.select_shared_feasible(t1_records, caps, mode=sf.BENCHMARK_PROXY, max_routes=None, partial_policy=sf.PARTIAL_NONE)
        by_id = {sf.candidate_identity(r, i): r for i, r in enumerate(t1_records)}
        strict_by = {row["candidate_id"]: row for row in strict["rows"]}
        mine = records[~records["is_duplicate"].astype(bool)].set_index("lane_key")
        for row in bench["rows"]:
            lane = by_id[row["candidate_id"]]
            own = mine.loc[lane["lane_key"]]
            room = row.get("individual_cap_room")
            t1_rows.append({
                "date": date, "lane_key": lane["lane_key"], "t1_strict_status": strict_by[row["candidate_id"]]["selection_status"],
                "t1_individually_feasible": room is not None and float(room) >= float(lane["recommended_qty"]) - 1e-9,
                "oe_proxy_feasible": bool(own["proxy_feasible"]), "oe_strict": own["strict_eligibility"],
                "oe_caps_only_strict": oe.caps_only_strict(lane, context)})
    return {"date": date, "universe": universe, "records": records, "pools": pools, "pool_frames": pool_frames,
            "combos": combos, "generation": generation, "fidelity": fidelity, "saved_eval": saved_eval,
            "ranked_saved": ranked_saved, "t1_cross": t1_rows, "context": context,
            "need": need_groups(rebuild, flow, date, universe, {"LEGACY_20": pools[cgr.LEGACY_20], "ALL_VALID": pools[cgr.ALL_VALID]})}


# ------------------------------------------------------------------------------------------------ positive control


def positive_control(day: Mapping[str, Any], evidence: SuhyupEvidence) -> dict[str, Any]:
    """TEST evidence injected in memory for one real ALL_VALID lane (never written, never evidence): the validator must
    reach STRICT_ELIGIBLE when evidence exists, so a 0 on real data is a data finding, not a filter bug."""
    records = day["records"]
    lane_row = records[records["in_all_valid"] & records["proxy_feasible"]].iloc[0]
    universe = day["universe"]
    lane = universe[~universe["is_duplicate"].astype(bool)].set_index("lane_key").loc[lane_row["lane_key"]].to_dict()
    lane["lane_key"] = lane_row["lane_key"]
    source, target, product, date = lane["source_id"], lane["target_id"], lane["product_id"], day["date"]
    qty = float(lane["recommended_qty"])
    route_rows = [{"network_id": NETWORK_ID, "source_id": source, "target_id": target, "product_id": product,
                   "allowed": "true", "effective_from": date, "evidence_class": oe.VERIFIED_OPERATOR_INPUT,
                   "verification_status": "VERIFIED", "evidence_source": "TEST_POSITIVE_CONTROL",
                   "temperature_requirement": evidence.temperature.get(product), "route_capacity_qty": qty * 10,
                   "quantity_unit": "BOX", "cost_basis": sf.FIXED_PER_TRIP, "cost_value": 100000}]
    quantity_rows = [
        {"network_id": NETWORK_ID, "location_id": source, "product_id": product, "quantity_kind": oe.DECLARED_TRANSFERABLE,
         "observation_start": date, "quantity": qty, "quantity_unit": "BOX", "evidence_class": oe.VERIFIED_OPERATOR_INPUT,
         "verification_status": "VERIFIED", "data_source": "TEST_POSITIVE_CONTROL"},
        {"network_id": NETWORK_ID, "location_id": target, "product_id": product, "quantity_kind": oe.DECLARED_TARGET_NEED,
         "observation_start": date, "quantity": qty, "quantity_unit": "BOX", "evidence_class": oe.VERIFIED_OPERATOR_INPUT,
         "verification_status": "VERIFIED", "data_source": "TEST_POSITIVE_CONTROL"}]
    routes, _ = oe.normalize_route_evidence(route_rows, network_id=NETWORK_ID)
    quantities, _ = oe.normalize_demand_evidence(quantity_rows, network_id=NETWORK_ID)
    units = {product: ("BOX", oe.VERIFIED_OPERATOR_INPUT)}
    full = oe.evaluate_lane(lane, evidence.context(date, route_records=routes, quantity_records=quantities, product_units=units))
    no_capacity = oe.evaluate_lane(lane, evidence.context(date, route_records=[{**routes[0], "route_capacity_qty": None}],
                                                          quantity_records=quantities, product_units=units))
    stock_only = oe.evaluate_lane(lane, evidence.context(date, route_records=routes, quantity_records=quantities[1:], product_units=units))
    return {"date": date, "lane_key": lane["lane_key"], "label": "TEST_POSITIVE_CONTROL (in memory, not evidence)",
            "all_evidence": full["operational_eligibility"], "all_evidence_strict": full["strict_eligibility"],
            "without_route_capacity": no_capacity["operational_eligibility"],
            "without_declared_transferable": stock_only["operational_eligibility"],
            "without_declared_transferable_reasons": stock_only["blocking_reasons"]}


# ------------------------------------------------------------------------------------------------ static tables


def dataset_survey(data_root: Path) -> pd.DataFrame:
    """Section 2: which dataset can supply operational evidence for the Suhyup network (verified on the files)."""
    matrix = pd.read_csv(data_root / "_COLLECTION_STATUS/EXTERNAL_VALIDATION_MATRIX.csv", encoding="utf-8-sig").set_index("dataset")
    catalog = pd.read_csv(data_root / "_COLLECTION_STATUS/MASTER_DATASET_CATALOG.csv", encoding="utf-8-sig").set_index("dataset_id")
    history_files = [p for p in (data_root / TRANSFER_HISTORY).rglob("*") if p.is_file()] if (data_root / TRANSFER_HISTORY).exists() else []
    rows = []

    def add(dataset, catalog_id, network, scope, inventory, sales, flow, route_od, observed, capacity, cost, unit, joinable,
            contribution, reason, period=None):
        status = catalog.loc[catalog_id, "status"] if catalog_id in catalog.index else None
        kind = catalog.loc[catalog_id, "real_or_synthetic"] if catalog_id in catalog.index else None
        period = period or (matrix.loc[dataset, "date_range"] if dataset in matrix.index else None)
        rows.append({"dataset": dataset, "catalog_id": catalog_id, "catalog_status": status, "real_or_synthetic": kind,
                     "operational_network": network, "period": period, "scope": scope, "has_stock": inventory,
                     "has_actual_sales": sales, "has_flow": flow, "has_route_od": route_od, "has_observed_transfer": observed,
                     "has_capacity": capacity, "has_cost": cost, "quantity_unit_status": unit,
                     "joinable_with_suhyup_candidates": joinable, "operational_evidence_contribution": contribution,
                     "reason": reason})

    add("suhyup", "02_SUHYUP_LOGISTICS", NETWORK_ID, "9 centers (6 routable) x 90 products x 31 days, 2026-07", True, False, True,
        "BENCHMARK_ONLY (canonical transfer_network = the 620 generated candidates)", False, False, "DERIVED (official tariff)",
        "UNKNOWN (count); kg DIRECT", "SELF", "stock DIRECT_REAL; outbound PROXY demand; no surplus, need, route, capacity or unit",
        "README: inter-center transfer, distance, cost and vehicle capacity are not actual observations")
    add("suhyup_warehouse", "03_SUHYUP_WAREHOUSE", "suhyup_coop_warehouses", "51 coops x 59 warehouses, 2026-07", True, False, True,
        False, False, False, False, "UNKNOWN", "NOT_JOINABLE", "none for the 6-center network",
        "different facilities (cooperative warehouses, not the logistics centers); product codes of its two files are not one "
        "coding system; README: no warehouse-to-warehouse transfer history", period="2026-07 (31 days)")
    add("osrm_road_distance", "10_SUHYUP_WAREHOUSE_LOCATION_ROAD", NETWORK_ID, "30 directed center pairs", False, False, False,
        "C (road route between verified coordinates)", False, False, False, None, "KEYED (center_code)",
        "route class C and distance only", "OSRM computes a road path; it is not an operated or permitted lane", period="static")
    add("toll_cost", "11_TOLL_COST", "korea_expressway", "toll tables, 420 warehouse routes", False, False, False, False, False,
        False, "DIRECT (public tariff)", None, "NOT_USED", "none (the official all-in rate already covers tolls)",
        "public toll schedule; not a Suhyup contract")
    add("vehicle_capacity", "12_VEHICLE_CAPACITY", "seoul_metro_business_vehicles", "22 reference vehicles", False, False, False,
        False, False, "kg classes of reference vehicles", False, "kg", "NOT_JOINABLE",
        "kg class sizing for the tariff only", "another organisation's fleet: not Suhyup vehicles, not route capacity, no availability")
    add("transport_cost_official", "13_TRANSPORT_COST_OFFICIAL", "national_tariff", "2026 official safe-freight rates", False,
        False, False, False, False, False, "DIRECT (public rate)", "KRW per km by class", "KEYED (route x class)",
        "DERIVED_REAL computed cost (PROXY when an upper class stands in)", "a public rate, not an invoice or contract")
    add("transfer_history", None, NETWORK_ID, f"{len(history_files)} files in {TRANSFER_HISTORY}", False, False, False, False,
        bool(history_files), False, False, None, "SELF" if history_files else "EMPTY",
        "class A evidence if it existed", "folder exists and is empty: no observed transfer between centers")
    for name, cid, network, joinable, note in (
            ("logisall", "22_KADX_LOGISALL", "logisall_masked_zip_zones", "NOT_JOINABLE",
             "observed zone-to-zone distribution of another company; masked zones, not facilities; no cost"),
            ("jangbogo", "23_KADX_JANGBOGO", "jangbogo_warehouses", "NOT_JOINABLE",
             "another company's warehouse x category sales and purchase requests (requests are not demand)"),
            ("nfqs", "29_NFQS_MARINE_STOCK", "regional_cold_storage_statistics", "NOT_JOINABLE",
             "quarterly regional stock in tons from cooperating firms; statistics, not Suhyup centers"),
            ("aihub", "26_AIHUB_INDUSTRIAL_LOGISTICS", "unidentified_site", "NOT_JOINABLE", "one unidentified site; no location key"),
            ("kamp", "30_KAMP_REBAR_SUPPLY_CHAIN", "rebar_project", "NOT_JOINABLE", "rebar shipments of one construction project"),
            ("m5", "20_WALMART_M5", "walmart_us_stores", "NOT_JOINABLE", "actual unit sales of US stores (forecast validation only)"),
            ("favorita", "27_FAVORITA", "favorita_ecuador_stores", "NOT_JOINABLE", "actual sales of Ecuadorian stores"),
            ("freshretailnet", "05_FRESHRETAILNET", "dingdong_fresh_retail", "NOT_JOINABLE", "normalised sales with stockout flags")):
        row = matrix.loc[name]
        add(name, cid, network, f"{row['locations']} locations x {row['products']} products", bool(row["has_inventory"]),
            bool(row["has_sales"]), bool(row["has_flow"]), bool(row["has_source_target"]), name == "logisall",
            bool(row["has_capacity"]), bool(row["has_cost"]), row["quantity_unit_status"], joinable,
            "none for Suhyup (own-scope research only)", note)
    for name, cid, note in (("aT_public_data", "24_AT_PUBLIC_DATA", "public agricultural / fisheries statistics and prices"),
                            ("kamis", "25_KAMIS", "wholesale / retail price statistics (no quantity, location or stock of Suhyup)")):
        add(name, cid, "public_price_statistics", "national price series", False, False, False, False, False, False, False, None,
            "NOT_JOINABLE", "none (price and demand of different datasets are never joined)", note)
    return pd.DataFrame(rows)


def demand_semantics(data_root: Path, flow: pd.DataFrame, alias_days: Mapping[str, int] | None = None) -> pd.DataFrame:
    """Section 3 / 11: meaning, source, unit and coverage of each quantity field for Suhyup."""
    nodes = flow[flow["center_code"].isin(rv.NETWORK_NODES)]
    canonical_flow = pd.read_parquet(data_root / CANONICAL / "canonical_inventory_flow.parquet")
    canonical_snap = pd.read_parquet(data_root / CANONICAL / "canonical_inventory_snapshot.parquet")
    n = len(flow)

    def missing(column: str) -> int:
        return int(pd.to_numeric(flow[column], errors="coerce").isna().sum()) if column in flow.columns else n

    unit = "|".join(sorted(canonical_flow["unit_status"].astype(str).unique()))
    rows = [
        ("current_stock", "stock_qty", f"{RAW_STOCK} : 재고량", unit + " count unit; kg in 재고량(킬로그램) DIRECT",
         "daily, center x product x state code", "summed over state codes per (date, center, product) for caps; "
         "pairs with >1 state code excluded from the generator upload", oe.DIRECT_REAL, missing("stock_qty"),
         "on-hand stock; available, reserved, damaged and in-transit quantities are absent (canonical available_qty "
         f"non-null {int(canonical_snap['available_qty'].notna().sum())}/{len(canonical_snap)})", "SOURCE_STOCK cap (DIRECT_REAL)"),
        ("source_surplus", "source_surplus", "derived: suhyup_algorithm_revalidation.enrich_inventory_constraints",
         "same as stock (UNKNOWN)", "daily", "max(0, stock - same-day median over the 6 nodes)", oe.PROXY, None,
         "a balancing rule; it ignores the source's own demand, reservations and transit", "SOURCE_SURPLUS cap (PROXY)"),
        ("target_need", "target_need_7d / gen_target_need", "derived: two formulas (target_need_formula_comparison.csv)",
         "same as stock (UNKNOWN)", "daily, 7-day horizon", "benchmark max(0, m + 7d - s); generator max(0, m - s) + 7d",
         oe.PROXY, None, "outbound shipments as demand and a cross-node median as target level", "TARGET_NEED cap (PROXY) / generator"),
        ("inbound_qty", "inbound_qty", f"{RAW_FLOW} : 입고량", unit, "daily", "summed over state codes", oe.DIRECT_REAL,
         missing("inbound_qty"), "receipts into the center; not demand", "unit-weight derivation only"),
        ("outbound_qty", "outbound_qty", f"{RAW_FLOW} : 출고량", unit, "daily", "summed over state codes", oe.DIRECT_REAL,
         missing("outbound_qty"), "shipments out of the center: auction / wholesale release, transfers and returns are not "
         "separated; used as a demand PROXY (OUTBOUND_SHIPMENT), never as sales", "target need demand term (PROXY)"),
        ("sales_qty", "sales_qty", "canonical_inventory_flow.sales_qty",
         unit, "daily", "-", oe.MISSING, int(canonical_flow["sales_qty"].isna().sum()),
         f"no retail sales exist (canonical sales_qty non-null {int(canonical_flow['sales_qty'].notna().sum())}/{len(canonical_flow)}); "
         "the upload column sales_qty carries outbound_qty (sales_qty_semantics = demand_proxy_not_retail_sales)",
         "generator demand term (as PROXY)"),
        ("transfer_qty", "recommended_qty", "candidate_generator quantity rule", "same as stock (UNKNOWN)", "per candidate",
         "int(max(1, min(surplus rule, need rule, stock, move cap 50 / 20)))", oe.CONFIG, None,
         "a generated quantity; the 50 / 20 move caps are CONFIG constants with no documented basis", "MILP / T1 quantity"),
        ("capacity", "route_capacity / dc_capacity", "none", None, None, None, oe.MISSING, None,
         "no route, vehicle-availability, DC, storage, dispatch or cold-chain capacity of Suhyup exists", "unchecked in T1 / T3"),
        ("transport_cost", "move_cost", f"{TARIFF} x {UNIT_WEIGHT}", "KRW per trip (computed)", "per candidate quantity",
         "official 2026 rate per km x OSRM km x vehicle mix sized by derived kg", oe.DERIVED_REAL, None,
         "a computed public-tariff estimate (PROXY when an upper vehicle class stands in); no invoice or contract",
         "Varo Final key 2, MILP stage 2"),
        ("distance", "distance_km", OSRM, "km", "static", "OSRM road route between verified center coordinates",
         oe.DERIVED_REAL, None, "road connectivity and length only (route class C), not an operated lane", "tariff, candidate_score"),
        ("lead_time", "travel_time_min", OSRM, "minutes", "static", "OSRM drive time", oe.DERIVED_REAL, None,
         "drive time only; loading, dispatch schedule and handling time are absent; no selector reads it", "none"),
    ]
    alias_days = alias_days or {}
    rows += [
        ("avg_daily_sales (normalized upload)", "avg_daily_sales", "data_loader.normalize_loaded_data (_ensure_alias)", unit,
         "daily", "copy of sales_qty", oe.PROXY, None,
         f"an alias copy of one day's outbound_qty, not an average (identical on {alias_days.get('avg', 0)}/31 days); it is "
         "the column the generator reads first as demand", "generator demand term"),
        ("sales_30d (normalized upload)", "sales_30d", "data_loader.normalize_loaded_data (_ensure_alias)", unit, "daily",
         "copy of sales_qty", oe.PROXY, None,
         f"an alias copy of one day's outbound_qty, not a 30-day total (identical on {alias_days.get('s30', 0)}/31 days)",
         "consumers that read sales_30d"),
    ]
    out = pd.DataFrame(rows, columns=["field", "varo_column", "source", "unit", "period_grain", "aggregation", "evidence_class",
                                      "missing_rows", "meaning", "used_by"])
    out["source_rows"] = [n if r in ("current_stock", "inbound_qty", "outbound_qty") else None for r in out["field"]]
    out["network_rows"] = [len(nodes) if r in ("current_stock", "inbound_qty", "outbound_qty") else None for r in out["field"]]
    kinds = pd.DataFrame([{"field": f"quantity_kind:{kind}", "meaning": meaning,
                           "evidence_class": oe.PROXY if kind in (oe.OUTBOUND_SHIPMENT, oe.ORDER, oe.FORECAST, oe.REGIONAL_FLOW, oe.LOGISTICS_MOVEMENT) else "ACCEPTED_IF_VERIFIED",
                           "used_by": "operational_demand_evidence contract"} for kind, meaning in oe.DEMAND_SEMANTICS.items()])
    return pd.concat([out, kinds], ignore_index=True)


def route_evidence_table(data_root: Path, records: pd.DataFrame, context_records: Sequence[Mapping[str, Any]],
                         observed: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    routes = pd.read_csv(data_root / mbv.PROCESSED / "routes.csv", dtype={"source_id": str, "target_id": str}, encoding="utf-8-sig")
    osrm = pd.read_csv(data_root / OSRM, dtype={"source_center_code": str, "target_center_code": str}, encoding="utf-8-sig")
    tariff = pd.read_csv(data_root / TARIFF, dtype={"source_center_code": str, "target_center_code": str}, encoding="utf-8-sig")
    osrm_by = {(r.source_center_code, r.target_center_code): r for r in osrm.itertuples()}
    tariff_by = tariff.groupby(["source_center_code", "target_center_code"])["cost_mapping_method"].agg(lambda s: "|".join(sorted(set(s)))).to_dict()
    usage = {}
    for scope in ("in_legacy_20", "in_all_valid", "in_all_lanes"):
        usage[scope] = records[records[scope]].groupby(["source_id", "target_id"]).size().to_dict()
    rows = []
    for route in routes.itertuples():
        key = (route.source_id, route.target_id)
        o = osrm_by.get(key)
        rows.append({"source_id": route.source_id, "target_id": route.target_id, "distance_km": route.distance_km,
                     "travel_time_min": route.travel_time_min, "distance_provenance": route.distance_provenance,
                     "osrm_route_status": getattr(o, "route_status", None), "tariff_cost_mapping": tariff_by.get(key),
                     "observed_transfer_records": sum(1 for x in observed if (x["source_id"], x["target_id"]) == key),
                     "operator_rule_records": sum(1 for x in context_records if (x["source_id"], x["target_id"]) == key),
                     "class_a_observed": False, "class_b_operator_allowed": False,
                     "class_c_road_connectable": "osrm" in str(route.distance_provenance).lower(),
                     "class_d_distance_only": "osrm" not in str(route.distance_provenance).lower(),
                     "route_class": oe.ROUTE_C if "osrm" in str(route.distance_provenance).lower() else oe.ROUTE_D,
                     "operational_evidence": "NONE",
                     "lane_days_legacy_20": int(usage["in_legacy_20"].get(key, 0)),
                     "lane_days_all_valid": int(usage["in_all_valid"].get(key, 0)),
                     "lane_days_all_lanes": int(usage["in_all_lanes"].get(key, 0)),
                     "osrm_does_not_establish": "; ".join(oe.OSRM_DOES_NOT_ESTABLISH)})
    return pd.DataFrame(rows)


def evidence_gap_matrix(records: pd.DataFrame) -> pd.DataFrame:
    """Section 30: what STRICT_ACTUAL would need for Suhyup, from the code's requirements and the data's evidence."""
    valid = records[records["in_all_valid"]]
    def lacking(column: str) -> int:
        return int((~valid[column].astype(bool)).sum())
    rows = [
        ("actual_source_transferable", oe.SUPPLY, "REQUIRED", "source_surplus = max(0, stock - median): PROXY rule; stock is "
         "on-hand only (no available / reserved / in-transit)", "operator transferable stock per day, or on-hand plus own actual "
         "sales of the same unit", "WMS allocation / operator", lacking("source_supply_accepted")),
        ("actual_target_need", oe.DEMAND, "REQUIRED", "outbound_qty is shipments (PROXY); median is a rule; two formulas disagree",
         "operator-declared need, or actual retail sales of the same location / unit / window", "replenishment plan or POS",
         lacking("target_demand_accepted")),
        ("verified_allowed_routes", oe.ROUTE, "REQUIRED", "only OSRM road distance (class C); no observed transfer, no rule",
         "operated-lane list or contract per directed pair (allowed, validity, temperature)", "TMS / contract / operator",
         lacking("route_accepted")),
        ("quantity_unit", oe.UNIT, "REQUIRED", "count unit UNKNOWN for every product (kg DIRECT, kg-per-unit derived)",
         "declared counting unit per product (EA / BOX / KG)", "product master / operator", lacking("quantity_unit_accepted")),
        ("transport_cost_basis", oe.COST, "REQUIRED (accepted: DIRECT / DERIVED / verified)", "official tariff computed; some "
         "lanes priced with an upper-class proxy vehicle (PROXY)", "contract rate or verified vehicle class per route",
         "carrier contract / invoice", lacking("transport_cost_accepted")),
        ("target_receivable", oe.HANDLING, "REQUIRED", "the target reports the product on the day (DIRECT_REAL) for every "
         "ALL_VALID lane; lanes to targets without the product have no evidence", "operator assortment / receivable list",
         "operator", lacking("target_handling_accepted")),
        ("route_capacity", oe.CAPACITY, "CONDITIONAL (VERIFIED_OPERATIONAL)", "none; the kg vehicle classes are another "
         "organisation's reference fleet", "per-route capacity in the quantity unit, vehicle availability, cold-chain slots",
         "TMS / carrier", lacking("capacity_accepted")),
        ("dc_capacity", oe.CAPACITY, "CONDITIONAL for VIA_DC lanes", "no DC in the Suhyup network (DIRECT_NETWORK)",
         "throughput / storage capacity of the DC", "WMS", 0),
    ]
    return pd.DataFrame(rows, columns=["field", "dimension", "requirement", "current_suhyup_evidence", "evidence_needed",
                                       "likely_provider", "all_valid_candidates_lacking"]).assign(table="EVIDENCE_GAP_MATRIX")


# ------------------------------------------------------------------------------------------------ Suhyup 31 days


def run_suhyup(data_root: Path, route_frame: Any = None, demand_frame: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    saved = sfv.load_suhyup_candidates(data_root)
    reference = pd.read_csv(data_root / sfv.REFERENCE, dtype={"date": str})
    flow = pd.read_csv(data_root / sfv.INVENTORY, dtype={"date": str, "center_code": str, "product_code": str, "state_code": str})
    rebuild = mbv.GeneratorRebuild(data_root)
    t1_bundle = pd.read_csv(data_root / "_SHARED_FEASIBILITY_VALIDATION" / "shared_feasibility_by_day.csv", dtype=str)
    t3_bundle = pd.read_csv(data_root / mbv.OUTPUT_FOLDER / "milp_benchmark_by_day.csv", dtype=str)
    routes, route_issues = oe.normalize_route_evidence(route_frame, network_id=NETWORK_ID)
    quantities, quantity_issues = oe.normalize_demand_evidence(demand_frame, network_id=NETWORK_ID)
    evidence = SuhyupEvidence(data_root, flow, routes, (), quantities)
    out: dict[str, list] = defaultdict(list)
    controls = []
    with _env(mbv.REAL_DATA_ROOT_ENV, str(data_root)):
        saved_recompute = sfv.TariffRecompute(saved)
        for date in sorted(saved["snapshot_date"].astype(str).unique()):
            day = run_day(rebuild, flow, date, evidence, saved[saved["snapshot_date"] == date], saved_recompute)
            out["records"].append(day["records"])
            out["combos"].extend(day["combos"])
            out["generation"].append(day["generation"])
            out["fidelity"].append(day["fidelity"])
            out["need"].append(day["need"])
            out["t1_cross"].extend(day["t1_cross"])
            out["regression"].append(cgv._regression(day, t1_bundle, t3_bundle))
            out["golden"].extend(cgv._golden(day, reference))
            if date == "2026-07-16":
                out["case_0716"] = reproduce_0716(data_root, flow, day["universe"])
                controls.append(positive_control(day, evidence))
    return {"records": pd.concat(out["records"], ignore_index=True), "combos": pd.DataFrame(out["combos"]),
            "generation": pd.DataFrame(out["generation"]), "fidelity": pd.DataFrame(out["fidelity"]),
            "need": pd.concat(out["need"], ignore_index=True), "t1_cross": pd.DataFrame(out["t1_cross"]),
            "regression": pd.DataFrame(out["regression"]), "golden": pd.DataFrame(out["golden"]),
            "case_0716": out["case_0716"], "positive_control": controls, "flow": flow,
            "operator_inputs": {"route_rows_usable": len(routes), "route_issues": route_issues,
                                "demand_rows_usable": len(quantities), "demand_issues": quantity_issues},
            "route_records": routes, "wall_seconds": round(time.perf_counter() - started, 3)}


# ------------------------------------------------------------------------------------------------ aggregation


def by_day(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in records.groupby("snapshot_date"):
        row = {"date": date}
        for scope in ("in_legacy_20", "in_all_valid", "in_all_lanes"):
            part = group[group[scope]]
            label = scope[3:]
            row[f"{label}_candidates"] = len(part)
            row[f"{label}_strict_eligible"] = int((part["strict_eligibility"] == oe.STRICT_ELIGIBLE).sum())
            row[f"{label}_proxy_feasible"] = int(part["proxy_feasible"].sum())
            row[f"{label}_benchmark_cost_comparable"] = int((part["comparison_scope"] == oe.SCOPE_BENCHMARK).sum())
            for status, count in oe.summarize_statuses(part).items():
                row[f"{label}_status_{status}"] = count
            for status, count in oe.dimension_flag_counts(part).items():
                row[f"{label}_lacking_{status}"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def legacy_vs_all_valid(records: pd.DataFrame) -> dict[str, Any]:
    legacy = set(zip(records.loc[records["in_legacy_20"], "snapshot_date"], records.loc[records["in_legacy_20"], "lane_key"]))
    valid = set(zip(records.loc[records["in_all_valid"], "snapshot_date"], records.loc[records["in_all_valid"], "lane_key"]))
    indexed = records.drop_duplicates(["snapshot_date", "lane_key"]).set_index(["snapshot_date", "lane_key"])

    def profile(keys: set) -> dict[str, Any]:
        part = indexed.loc[sorted(keys)] if keys else indexed.iloc[0:0]
        return {"count": len(part), "strict_eligible": int((part["strict_eligibility"] == oe.STRICT_ELIGIBLE).sum()),
                "proxy_feasible": int(part["proxy_feasible"].sum()),
                "operational_evidence_lacking": int((part["strict_eligibility"] != oe.STRICT_ELIGIBLE).sum()),
                "ineligible": int((part["operational_eligibility"] == oe.INELIGIBLE).sum()),
                "statuses": oe.summarize_statuses(part.reset_index()),
                "p1_validity": dict(Counter(part["p1_validity_status"]))}
    return {"legacy_20": profile(legacy), "all_valid": profile(valid), "common": profile(legacy & valid),
            "added_by_all_valid": profile(valid - legacy), "legacy_20_not_in_all_valid": profile(legacy - valid)}


def sensitivity_summary(combos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for combo, group in combos.groupby("combo", sort=False):
        first = group.iloc[0]
        rows.append({"combo": combo, "pool": first["pool"], "need_definition": first["need_definition"],
                     "description": first["description"], "days": len(group), "pool_total": int(group["pool_size"].sum()),
                     **{f"{p}_{m}_total": float(group[f"{p}_{m}"].sum()) for p in ("milp", "varo_final", "t1_all_or_nothing")
                        for m in ("service", "cost")},
                     "milp_selected_total": int(group["milp_selected_count"].sum()),
                     "violations_total": int(group["violations"].sum()),
                     "milp_optimal_days": int((group["milp_status"] == mbi.OPTIMAL).sum()),
                     "milp_enumerated_days": int((group["milp_enumeration_status"] == "ENUMERATED").sum()),
                     "milp_enumeration_proven_days": int(group["milp_enumeration_proven"].astype(bool).sum()),
                     "milp_target_excess_vs_benchmark_need_total": float(group["milp_target_excess_vs_benchmark_need"].sum()),
                     "milp_days_exceeding_benchmark_need": int((group["milp_target_excess_vs_benchmark_need"] > 1e-9).sum()),
                     "milp_target_excess_vs_generator_need_total": float(group["milp_target_excess_vs_generator_need"].sum()),
                     "t1_strict_insufficient_days": int((group["t1_strict_status"] == sf.INSUFFICIENT_CAP_DATA).sum())})
    return pd.DataFrame(rows)


def comparisons(summary: pd.DataFrame) -> list[dict[str, Any]]:
    """Section 28: which totals may be compared directly."""
    s = summary.set_index("combo")

    def side(combo: str, scope: str = oe.SCOPE_BENCHMARK) -> dict[str, Any]:
        row = s.loc[combo]
        return {"service": row["milp_service_total"], "cost": row["milp_cost_total"], "eligibility_scope": scope,
                "constraint_definition": f"SOURCE=PROXY_SURPLUS|TARGET={row['need_definition']}", "cost_basis": "COMPUTED_TARIFF",
                "quantity_basis": "FIXED_BINARY", "eligible_count": int(row["pool_total"])}
    out = []
    for left, right, note in (("D", "B", "ALL_VALID vs LEGACY_20, both benchmark need"),
                              ("A", "B", "LEGACY_20 under the two need definitions"),
                              ("C", "D", "ALL_VALID under the two need definitions"),
                              ("E_REGEN_B", "B", "generator re-run with the benchmark formula vs production pool")):
        out.append({"left": left, "right": right, "note": note, **oe.fair_cost_comparison(side(left), side(right))})
    strict = {"service": 0.0, "cost": 0.0, "eligibility_scope": oe.SCOPE_STRICT, "constraint_definition": "STRICT_ACTUAL",
              "cost_basis": "ACTUAL_SPEND", "quantity_basis": "FIXED_BINARY", "eligible_count": 0}
    out.append({"left": "STRICT_ACTUAL(ALL_VALID)", "right": "STRICT_ACTUAL(LEGACY_20)", "note": "strict-eligible pools",
                **oe.fair_cost_comparison(strict, strict)})
    return out


def run_validation(data_root: Path, output_dir: Path | None = None, repo_root: Path | None = None,
                   route_evidence: Path | None = None, demand_evidence: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    data_root = Path(data_root)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    output_dir = Path(output_dir or data_root / OUTPUT_FOLDER)
    before = _hashes(data_root, PROTECTED_BUNDLES)
    operator_folder = data_root / OPERATOR_FOLDER
    route_path = route_evidence or (operator_folder / "operational_route_evidence.csv")
    demand_path = demand_evidence or (operator_folder / "operational_demand_evidence.csv")
    route_frame = pd.read_csv(route_path, dtype=str) if Path(route_path).exists() else None
    demand_frame = pd.read_csv(demand_path, dtype=str) if Path(demand_path).exists() else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_suhyup(data_root, route_frame, demand_frame)
    records, combos, need = result["records"], result["combos"], result["need"]
    fidelity, regression, golden, cross = result["fidelity"], result["regression"], result["golden"], result["t1_cross"]
    flow = result["flow"]
    sens = sensitivity_summary(combos)
    s = sens.set_index("combo")
    daily = by_day(records)
    all_valid = records[records["in_all_valid"]]
    legacy = records[records["in_legacy_20"]]
    all_lanes = records[records["in_all_lanes"]]
    plans = {"ALL_VALID": oe.plan_missing_inputs(all_valid).assign(table="PRIORITY_ALL_VALID"),
             "LEGACY_20": oe.plan_missing_inputs(legacy).assign(table="PRIORITY_LEGACY_20")}
    missing = pd.concat([evidence_gap_matrix(records), *plans.values()], ignore_index=True)
    routes_table = route_evidence_table(data_root, records, result["route_records"], [])
    survey = dataset_survey(data_root)
    semantics = demand_semantics(data_root, flow, {"avg": int(fidelity["upload_avg_daily_sales_is_sales_qty_copy"].sum()),
                                                   "s30": int(fidelity["upload_sales_30d_is_sales_qty_copy"].sum())})
    centers = pd.read_csv(data_root / CENTER_MASTER, dtype={"center_code": str}, encoding="utf-8-sig")
    unroutable = centers.loc[~centers["route_usable"].astype(bool), ["center_code", "center_name", "address_status"]].to_dict("records")
    case = result["case_0716"]
    pc = result["positive_control"][0] if result["positive_control"] else {}
    comparison = comparisons(sens)
    lva = legacy_vs_all_valid(records)

    # T2 / Seller Loss read-only relation
    t2 = pd.read_csv(data_root / "_ACTION_CONSISTENCY_VALIDATION" / "action_consistency_candidates.csv", dtype=str)
    t2 = t2[t2["scope"] == "SUHYUP_31D"]
    joined = t2.merge(legacy[["snapshot_date", "route_id", "strict_eligibility", "operational_eligibility"]],
                      left_on=["date", "route_id"], right_on=["snapshot_date", "route_id"], how="left")
    t2_relation = {"t2_candidates": len(t2), "linked": int(joined["strict_eligibility"].notna().sum()),
                   "t2_action_status": dict(Counter(t2["action_status"])), "t2_is_executable": dict(Counter(t2["is_executable"])),
                   "proxy_plan_rows_by_eligibility": dict(Counter(joined.loc[joined["action_status"] == "PROXY_PLAN", "operational_eligibility"])),
                   "proxy_plan_rows_strict_eligible": int(((joined["action_status"] == "PROXY_PLAN") & (joined["strict_eligibility"] == oe.STRICT_ELIGIBLE)).sum()),
                   "seller_loss_comparison_status": dict(Counter(t2["seller_loss_comparison_status"]))}
    outcomes = pd.read_csv(data_root / "_SELLER_LOSS_VALIDATION" / "seller_shadow_outcomes.csv")
    scenario = pd.read_csv(data_root / "_SELLER_LOSS_VALIDATION" / "seller_loss_scenario_validation.csv")
    external = json.loads((data_root / "09_External_Validation/results/forecast_external_generalization.json").read_text(encoding="utf-8"))
    m5_gate = json.loads((data_root / "20_WALMART_M5/results/m5_forecast_v2_promotion_gate.json").read_text(encoding="utf-8"))

    # key namespace check (section 25): canonical location ids vs flow center codes
    location_master = pd.read_parquet(data_root / CANONICAL / "canonical_location_master.parquet")
    names_by_code = flow.drop_duplicates("center_code").set_index("center_code")["center_name"].to_dict()
    mapped = {oe.normalize_location_id(r.location_id, "suhyup_canonical_location_id"): r.location_name for r in location_master.itertuples()}
    mapping_ok = all(names_by_code.get(code) == name for code, name in mapped.items())
    guard_cross = oe.join_guard(oe.DataScope("kamis", "public_price_statistics", "kamis_item", "kamis_market"),
                                oe.DataScope(DATASET, NETWORK_ID, "suhyup_product_code", "suhyup_center_code"))
    guard_same = oe.join_guard(oe.DataScope(DATASET, NETWORK_ID, "suhyup_product_code", "suhyup_center_code", date="2026-07-16"),
                               oe.DataScope("suhyup_canonical", NETWORK_ID, "suhyup_product_code", "suhyup_canonical_location_id", date="2026-07-16"))

    changed = (_git(repo_root, "status", "--porcelain") or "").splitlines()
    modified = [line for line in changed if not line.startswith("??")]
    need_rows = need.dropna(subset=["need_benchmark", "need_generator"])
    identity_ok = bool((need.dropna(subset=["formula_gap_on_benchmark_inputs"]).eval("abs(formula_gap_on_benchmark_inputs - identity_gap)") <= 1e-9).all())
    contract_fields = ("candidate_id", "source_id", "target_id", "product_id", "requested_qty", "validated_qty",
                       "source_surplus_basis", "target_need_basis", "route_evidence", "capacity_evidence",
                       "quantity_unit_evidence", "cost_evidence", "operational_eligibility", "blocking_reasons", "provenance",
                       "comparison_scope")
    checks = [
        _check("F1", "LEGACY_20 rebuilt equals the production generate_candidates output 31/31 (production untouched)",
               bool(fidelity["legacy_20_equals_production_generator"].all()), int(fidelity["legacy_20_equals_production_generator"].sum())),
        _check("F2", "P1 pools reproduced: LEGACY_20 620 and ALL_VALID 1,735 candidates",
               int(fidelity["legacy_20_size"].sum()) == P1_POOL_TOTAL[cgr.LEGACY_20] and int(fidelity["all_valid_size"].sum()) == P1_POOL_TOTAL[cgr.ALL_VALID],
               {"legacy_20": int(fidelity["legacy_20_size"].sum()), "all_valid": int(fidelity["all_valid_size"].sum()),
                "all_lanes": int(fidelity["all_lanes_size"].sum())}),
        _check("F3", "P1 MILP costs reproduced: combo B 13,629,146 (LEGACY_20) and combo D 8,194,265 (ALL_VALID) at 7,750",
               abs(s.loc["B", "milp_cost_total"] - P1_BASELINE_COST[cgr.LEGACY_20]) <= 1e-6
               and abs(s.loc["D", "milp_cost_total"] - P1_BASELINE_COST[cgr.ALL_VALID]) <= 1e-6
               and abs(s.loc["B", "milp_service_total"] - SERVICE_TOTAL) <= 1e-9 and abs(s.loc["D", "milp_service_total"] - SERVICE_TOTAL) <= 1e-9,
               {"B": s.loc["B", "milp_cost_total"], "D": s.loc["D", "milp_cost_total"]}),
        _check("F4", "generator re-run with the generator need reproduces LEGACY_20 (lanes, order, qty) 31/31",
               bool(fidelity["regenerate_generator_need_equals_legacy_20"].all())),
        _check("R1", "186 reference rows reproduced (incl. status)", bool(golden["equal"].all()) and len(golden) == 186,
               f"{int(golden['equal'].sum())}/{len(golden)}"),
        _check("R2", "T1 regression: BENCHMARK_PROXY 7,750 / 13,629,146, 0 violations, bundle routes 31/31; STRICT insufficient 31/31",
               bool(regression["t1_matches_bundle"].all() and regression["t1_partial_matches_bundle"].all()
                    and regression["t1_strict_matches_bundle"].all())
               and abs(regression["t1_benchmark_service"].sum() - SERVICE_TOTAL) <= 1e-9
               and abs(regression["t1_benchmark_cost"].sum() - 13629146) <= 1e-6
               and int(regression["t1_benchmark_violations"].sum()) == 0
               and bool((regression["t1_strict_status"] == sf.INSUFFICIENT_CAP_DATA).all())),
        _check("R3", "T3 regression: MILP OPTIMAL and enumeration-proven 31/31, bundle routes and totals, Varo Final = MILP",
               bool((regression["t3_milp_status"] == mbi.OPTIMAL).all() and regression["t3_milp_matches_bundle"].all()
                    and regression["t3_varo_routes_match_bundle"].all() and regression["t3_enumeration_proven"].all()
                    and (regression["t3_varo_vs_milp"] == mbi.DIRECTLY_COMPARABLE).all() and regression["t3_partial_not_comparable"].all())),
        _check("R4", "T2 relation: 620 Suhyup candidates linked; no PROXY_PLAN row is strict-eligible; none executable",
               t2_relation["linked"] == 620 and t2_relation["proxy_plan_rows_strict_eligible"] == 0
               and set(t2_relation["t2_is_executable"]) == {"False"}, t2_relation),
        _check("E1", "every routed lane has one eligibility status and the 16 contract fields",
               bool(records["operational_eligibility"].isin(oe.ELIGIBILITY_STATUSES).all())
               and all(field in records.columns for field in contract_fields)),
        _check("E2", "STRICT_ELIGIBLE = 0 on real data and every non-strict candidate names its missing evidence",
               int((records["strict_eligibility"] == oe.STRICT_ELIGIBLE).sum()) == 0
               and bool(records.loc[records["operational_eligibility"] != oe.INELIGIBLE, "blocking_reasons"].notna().all())),
        _check("E3", "positive control: in-memory TEST evidence makes a real lane STRICT_ELIGIBLE (VERIFIED_OPERATIONAL); "
                     "without route capacity VERIFIED_CONDITIONAL; on-hand stock alone does not verify the source",
               pc.get("all_evidence") == oe.VERIFIED_OPERATIONAL and pc.get("without_route_capacity") == oe.VERIFIED_CONDITIONAL
               and pc.get("without_declared_transferable") not in (oe.VERIFIED_OPERATIONAL, oe.VERIFIED_CONDITIONAL), pc),
        _check("E4", "no road distance is used as route verification (every class C/D/E lane has route_accepted False)",
               not bool(records.loc[records["route_class"].isin((oe.ROUTE_C, oe.ROUTE_D, oe.ROUTE_E)), "route_accepted"].any())),
        _check("E5", "outbound is never actual demand: no Suhyup target need is accepted",
               not bool(records["target_demand_accepted"].any())),
        _check("X1", "independent T1 cross-check: proxy feasibility equals T1 BENCHMARK individual feasibility on every ALL_VALID lane",
               bool((cross["t1_individually_feasible"] == cross["oe_proxy_feasible"]).all()),
               {"lanes": len(cross), "agree": int((cross["t1_individually_feasible"] == cross["oe_proxy_feasible"]).sum())}),
        _check("X2", "independent T1 cross-check: caps-only strict (this contract) and T1 STRICT_ACTUAL both reject every ALL_VALID lane",
               bool((cross["t1_strict_status"] == sf.INSUFFICIENT_CAP_DATA).all() and (~cross["oe_caps_only_strict"].astype(bool)).all())),
        _check("N1", "need identity generator - benchmark = min(max(0, stock - median), 7 x outbound) on every row (same inputs)", identity_ok),
        _check("N2", "2026-07-16 614504 @ 218020: generator need 63, benchmark need 6 reproduced from source rows (raw stock file = processed stock)",
               abs(case["generator_need"] - 63) <= 1e-9 and abs(case["benchmark_need"] - 6) <= 1e-9 and case["raw_equals_processed_stock"]
               and abs(case["lane_gen_target_need"] - 63) <= 1e-9 and abs(case["lane_target_need_7d"] - 6) <= 1e-9),
        _check("N3", "generator need has one value per (target, product) per day (a valid cap substitute)",
               bool(fidelity["generator_need_single_value_per_target_product"].all())),
        _check("S1", "every sensitivity combo: MILP OPTIMAL 31/31 and 0 independent-check violations; enumeration-proven on "
                     "every day it ran (pools above the 750,000-combination limit are NOT_ENUMERATED, as in P1)",
               bool((sens["milp_optimal_days"] == 31).all() and (sens["violations_total"] == 0).all()
                    and (sens["milp_enumeration_proven_days"] == sens["milp_enumerated_days"]).all()),
               sens.set_index("combo")[["milp_optimal_days", "milp_enumerated_days", "milp_enumeration_proven_days"]].to_dict("index")),
        _check("J1", "key namespaces: canonical location ids map to flow center codes by a verified rule; cross-dataset joins blocked",
               mapping_ok and not guard_cross["allowed"] and guard_same["allowed"],
               {"mapping_rule": MAPPING_DOC, "cross": guard_cross, "same_network": guard_same}),
        _check("G1", "no existing tracked file changed (new files only)", not modified, modified),
    ]
    after = _hashes(data_root, PROTECTED_BUNDLES)
    changed_bundles = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    checks.append(_check("G2", "no existing data / bundle file changed during the run", not changed_bundles, changed_bundles))
    failed = [c["check_id"] for c in checks if c["status"] == "FAIL"]

    totals = {"all_valid": oe.summarize_statuses(all_valid), "legacy_20": oe.summarize_statuses(legacy),
              "all_lanes": oe.summarize_statuses(all_lanes)}
    dims = {"all_valid": oe.dimension_flag_counts(all_valid), "legacy_20": oe.dimension_flag_counts(legacy),
            "all_lanes": oe.dimension_flag_counts(all_lanes)}
    generation = result["generation"]
    worst = need_rows.assign(abs_gap=need_rows["total_gap"].abs()).sort_values("abs_gap", ascending=False).head(15)
    formula_summary = {
        "groups": int(len(need_rows)), "differing_groups": int((need_rows["total_gap"].abs() > 1e-9).sum()),
        "formula_only_differing_groups": int((need_rows["formula_gap_on_benchmark_inputs"].abs() > 1e-9).sum()),
        "input_namespace_differing_groups": int((need_rows["input_namespace_effect"].abs() > 1e-9).sum()),
        "duplicate_state_pairs": int(need["excluded_duplicate_state_pair"].sum()),
        "gap_total_units": round(float(need_rows["total_gap"].sum()), 3),
        "gap_formula_units": round(float(need_rows["formula_gap_on_benchmark_inputs"].sum()), 3),
        "gap_input_units": round(float(need_rows["input_namespace_effect"].sum()), 3),
        "cases": dict(Counter(need_rows["need_case"])),
        "differing_groups_in_legacy_20_lanes": int(((need_rows["total_gap"].abs() > 1e-9) & (need_rows["lanes_in_legacy_20"] > 0)).sum()),
        "differing_groups_in_all_valid_lanes": int(((need_rows["total_gap"].abs() > 1e-9) & (need_rows["lanes_in_all_valid"] > 0)).sum()),
        "largest_gaps": worst[["date", "center_id", "product_id", "need_generator", "need_benchmark", "total_gap", "need_case",
                               "excluded_duplicate_state_pair"]].to_dict("records"),
        "lanes_generator_need_above_benchmark": int((pd.to_numeric(all_lanes["need_formula_gap"], errors="coerce") > 1e-9).sum()),
        "lanes_generator_need_below_benchmark": int((pd.to_numeric(all_lanes["need_formula_gap"], errors="coerce") < -1e-9).sum()),
        "generation_change_31d": {k: int(generation[k].sum()) for k in ("target_choice_changed_rows", "qty_changed_rows_same_target",
                                                                       "top20_lanes_added", "top20_lanes_removed")},
        "generation_change_qty": {"legacy_20": float(generation["legacy_qty_total"].sum()), "regen_b": float(generation["regen_b_qty_total"].sum())},
    }
    summary_rows = [("suhyup", "days", int(records["snapshot_date"].nunique())),
                    ("suhyup", "routed_lanes_evaluated", len(records)),
                    *[(scope, f"status_{k}", v) for scope, counts in totals.items() for k, v in counts.items()],
                    *[(scope, f"lacking_{k}", v) for scope, counts in dims.items() for k, v in counts.items()],
                    *[(scope, "strict_eligible", int((frame["strict_eligibility"] == oe.STRICT_ELIGIBLE).sum()))
                      for scope, frame in (("all_valid", all_valid), ("legacy_20", legacy), ("all_lanes", all_lanes))],
                    *[(scope, "proxy_feasible", int(frame["proxy_feasible"].sum()))
                      for scope, frame in (("all_valid", all_valid), ("legacy_20", legacy), ("all_lanes", all_lanes))],
                    *[(scope, "benchmark_cost_comparable", int((frame["comparison_scope"] == oe.SCOPE_BENCHMARK).sum()))
                      for scope, frame in (("all_valid", all_valid), ("legacy_20", legacy), ("all_lanes", all_lanes))],
                    *[(f"sensitivity_{row.combo}", metric, getattr(row, metric)) for row in sens.itertuples()
                      for metric in ("pool_total", "milp_service_total", "milp_cost_total", "t1_all_or_nothing_cost_total",
                                     "milp_target_excess_vs_benchmark_need_total", "milp_days_exceeding_benchmark_need")],
                    *[("need_formula", key, value) for key, value in formula_summary.items() if not isinstance(value, (dict, list))],
                    ("runtime", "wall_seconds", round(time.perf_counter() - started, 3))]
    validation = {
        "evidence_version": oe.EVIDENCE_VERSION, "git_head": _git(repo_root, "rev-parse", "HEAD"), "data_root": str(data_root),
        "network_id": NETWORK_ID, "checks": checks, "failed_checks": failed,
        "inputs": {"flow": sfv.INVENTORY, "processed_upload": mbv.PROCESSED, "candidates": sfv.CANDIDATES,
                   "reference": sfv.REFERENCE, "operator_route_evidence": str(route_path) if route_frame is not None else None,
                   "operator_demand_evidence": str(demand_path) if demand_frame is not None else None},
        "operator_inputs": result["operator_inputs"],
        "evidence_classes": {"classes": list(oe.EVIDENCE_CLASSES), "accepted": sorted(oe.ACCEPTED_EVIDENCE),
                             "base_provenance": oe.BASE_PROVENANCE,
                             "note": "metadata over the existing provenance contract; USER_INPUT without a verification flag is USER_ESTIMATE here"},
        "requirements": {"required": list(oe.REQUIRED_DIMENSIONS), "conditional": list(oe.CONDITIONAL_DIMENSIONS),
                         "status_order": list(oe.EVALUATION_ORDER), "statuses": list(oe.ELIGIBILITY_STATUSES)},
        "need_formulas": oe.NEED_FORMULAS, "need_cases": oe.NEED_CASES, "target_need_contract": oe.TARGET_NEED_CONTRACT,
        "need_formula_findings": formula_summary, "case_0716": case,
        "route_contract": {"required": list(oe.ROUTE_EVIDENCE_REQUIRED), "optional": list(oe.ROUTE_EVIDENCE_OPTIONAL),
                           "not_adopted": oe.ROUTE_EVIDENCE_NOT_ADOPTED, "classes": list(oe.ROUTE_CLASSES),
                           "operational_classes": [oe.ROUTE_A, oe.ROUTE_B], "osrm_does_not_establish": list(oe.OSRM_DOES_NOT_ESTABLISH),
                           "input_path": "uploaded_data['operational_route_evidence'] / CLI --route-evidence; no workbook sheet or UI in this step"},
        "demand_contract": {"required": list(oe.DEMAND_EVIDENCE_REQUIRED), "optional": list(oe.DEMAND_EVIDENCE_OPTIONAL),
                            "kinds": oe.DEMAND_SEMANTICS, "staleness_days": oe.NEED_HORIZON_DAYS},
        "eligibility_totals": totals, "dimension_lacking_counts": dims, "legacy_20_vs_all_valid": lva,
        "sensitivity": sens.to_dict("records"), "comparisons": comparison,
        "positive_control": result["positive_control"],
        "independent_validation": {
            "t1_cross_lanes": len(cross),
            "proxy_feasible_agree": int((cross["t1_individually_feasible"] == cross["oe_proxy_feasible"]).sum()),
            "differences_by_design": [
                "T1 STRICT_ACTUAL checks only the source and target caps (and route / DC capacity when present); this "
                "contract also requires an operated route, a declared unit, an accepted cost and a receivable target",
                "T1 accepts on-hand stock (DIRECT_REAL) as the source cap when no surplus column exists; this contract never "
                "treats on-hand stock as transferable surplus",
                "T1 accepts USER_INPUT caps; this contract accepts an operator value only when it is declared verified",
                "T1 keeps route / DC capacity optional; here a missing capacity leaves the lane VERIFIED_CONDITIONAL"]},
        "relations": {"t2": t2_relation, "seller_loss": {
            "scenario_rows": len(scenario), "scenario_pass": int(scenario["pass"].astype(bool).sum()),
            "shadow_outcome_rows": len(outcomes), "transfer_strategy_candidates_with_strict_evidence": 0,
            "note": "Seller Loss formulas untouched; only strict-eligible lanes with an actual-spend cost could feed its TRANSFER strategy"},
            "forecast": {"external_generalization_verdict": external.get("GENERALIZATION_VERDICT"),
                         "m5_v2_config_signature": m5_gate.get("config_signature"), "m5_v2_gate_passed": m5_gate.get("passed")}},
        "key_namespaces": {"canonical_location_to_center_code": mapping_ok, "rule": MAPPING_DOC},
        "centers_without_route_evidence": unroutable,
        "demand_alias_findings": {"avg_daily_sales_copy_days": int(fidelity["upload_avg_daily_sales_is_sales_qty_copy"].sum()),
                                  "sales_30d_copy_days": int(fidelity["upload_sales_30d_is_sales_qty_copy"].sum())},
        "verdicts": _verdicts(failed, totals),
        "limitations": [
            "no operator evidence exists for Suhyup: route, demand, unit and capacity cannot be verified from the data",
            "STRICT_ELIGIBLE 0 is a data finding (positive control passes), not a filter result",
            "PROXY_FEASIBLE means the lane passes the benchmark checks at its quantity; it is not executable",
            "the computed tariff is an estimate (official rate x OSRM km), never an actual expense",
            "the planner's unlock counts are upper bounds: they assume the supplied value verifies and does not cut the quantity",
            "sensitivity combos use different constraint definitions: their costs are not savings of one another",
        ],
        "production_action_applied": False, "production_candidate_pool_replaced": False,
        "runtime": {"wall_seconds": round(time.perf_counter() - started, 3), "suhyup_seconds": result["wall_seconds"]},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows, columns=["scope", "metric", "value"]).to_csv(output_dir / OUTPUT_FILES[0], index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / OUTPUT_FILES[1], index=False, encoding="utf-8-sig")
    records.to_csv(output_dir / OUTPUT_FILES[2], index=False, encoding="utf-8-sig")
    routes_table.to_csv(output_dir / OUTPUT_FILES[3], index=False, encoding="utf-8-sig")
    semantics.to_csv(output_dir / OUTPUT_FILES[4], index=False, encoding="utf-8-sig")
    need.to_csv(output_dir / OUTPUT_FILES[5], index=False, encoding="utf-8-sig")
    missing.to_csv(output_dir / OUTPUT_FILES[6], index=False, encoding="utf-8-sig")
    (output_dir / OUTPUT_FILES[7]).write_text(json.dumps(validation, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    survey.to_csv(output_dir / OUTPUT_FILES[8], index=False, encoding="utf-8-sig")
    combos.merge(result["generation"], on="date", how="left").to_csv(output_dir / OUTPUT_FILES[9], index=False, encoding="utf-8-sig")
    records.astype({c: str for c in records.columns if records[c].dtype == object}).to_parquet(output_dir / OUTPUT_FILES[10], index=False)
    return validation


def _verdicts(failed: Sequence[str], totals: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, Any]]:
    core = not set(failed) - {"G1"}
    strict = sum(totals["all_valid"].get(s, 0) for s in (oe.VERIFIED_OPERATIONAL, oe.VERIFIED_CONDITIONAL))
    return {
        "A_contract_and_code": {"status": "READY" if core else "NOT_READY",
                                "basis": "new files only; evidence contract, eligibility, need formulas, planner and checks"},
        "B_suhyup_31d_evidence_audit": {"status": "READY" if core else "NOT_READY",
                                        "basis": "every routed lane of 31 days classified; regressions held; positive control passes"},
        "C_strict_actual_operation": {"status": "NOT_READY" if strict == 0 else "READY_WITH_LIMITATIONS",
                                      "basis": f"{strict} ALL_VALID lanes are STRICT_ELIGIBLE: no operated-lane, actual-demand, "
                                               "unit or capacity evidence exists for Suhyup"},
        "D_cost_claims": {"status": "BENCHMARK_ONLY",
                          "basis": "13,629,146 -> 8,194,265 stays a computed tariff difference under PROXY caps and "
                                   "unverified routes; not an actual saving"},
        "E_production_replacement": {"status": "NOT_READY",
                                     "basis": "shadow only; the production generator, Top-5, final action, T2 and Seller Loss are unchanged"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--route-evidence", type=Path, default=None, help="operational_route_evidence CSV (operator input)")
    parser.add_argument("--demand-evidence", type=Path, default=None, help="operational_demand_evidence CSV (operator input)")
    args = parser.parse_args()
    result = run_validation(args.data_root, args.output_dir, route_evidence=args.route_evidence, demand_evidence=args.demand_evidence)
    print(json.dumps({"failed_checks": result["failed_checks"], "checks": result["checks"], "verdicts": result["verdicts"]},
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

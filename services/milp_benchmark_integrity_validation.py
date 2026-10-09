"""MILP benchmark integrity on the real Suhyup 31-day benchmark (T3).

Validation only. It reads the saved Suhyup candidates, the actual inventory flow, the 186-row reference bundle, the
saved 2026-07 operational MILP benchmark and the processed 2026-07-31 upload, and never writes there. Outputs go to
<data-root>/_MILP_BENCHMARK_INTEGRITY (local only, never into git); a re-run replaces only OUTPUT_FILES.

Per day it re-solves the lexicographic MILP (objective unchanged) with solver diagnostics, proves its optimum by
exhaustive enumeration inside the same 20-candidate pool, re-checks every plan without the solver, compares MILP /
offline Varo Final / T1 / the legacy Top-5 / the earlier saved MILP under comparison signatures, and keeps the solver MIP
gap apart from the decision gap. A separate cutoff experiment rebuilds the generator pool before its top-20 cut and
solves the same model on larger pools; it never replaces the benchmark.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import time
import tracemalloc
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest import mock

import pandas as pd
import scipy

from services import candidate_generator as cg
from services import milp_benchmark_integrity as mbi
from services import shared_feasibility_selection as sf
from services import shared_feasibility_validation as sfv
from services import suhyup_algorithm_revalidation as rv
from services.real_data_adapters import DATA_ROOT
from services.seller_decision_validation import _env

OUTPUT_FOLDER = "_MILP_BENCHMARK_INTEGRITY"
OUTPUT_FILES = (
    "milp_benchmark_integrity_summary.csv",
    "milp_benchmark_by_day.csv",
    "milp_candidate_cutoff_analysis.csv",
    "milp_constraint_comparison.csv",
    "milp_route_comparison.csv",
    "milp_optimality_gap_validation.csv",
    "milp_claims_audit.csv",
    "milp_benchmark_integrity.json",
)
SAVED_MILP_DAILY = "16_VARO_E2E_20260731/milp_benchmark/multi_snapshot/suhyup_202607_operational_milp_daily_summary.csv"
SAVED_MILP_ROUTES = "16_VARO_E2E_20260731/milp_benchmark/multi_snapshot/suhyup_202607_operational_milp_selected_routes.csv"
PROCESSED = "16_VARO_E2E_20260731/processed"
REAL_DATA_ROOT_ENV = "VARO_REAL_DATA_ROOT"
DATASET = "suhyup_202607_31d"
MAX_ROUTES = rv.MAX_DAILY_ROUTES
ID_COLUMNS = {"store_id": str, "product_id": str, "node_id": str, "source_id": str, "target_id": str}
CONSTRAINTS = ("SOURCE_CAP", "TARGET_NEED", "LANE_DUPLICATE", "SELECTION_LIMIT")
MILP_CAP_SPEC = {kind: sfv.SUHYUP_CAP_SPEC[kind] for kind in (sf.SOURCE_SURPLUS, sf.TARGET_NEED)}
COST_FIXED = "QUANTITY_SPECIFIC_TARIFF_AT_RECOMMENDED_QTY"
COST_PARTIAL = "QUANTITY_SPECIFIC_TARIFF_RECOMPUTED_FOR_PARTIAL"
POOLS = ("P20_SAVED", "P30_SCORE", "PALL_GENERATOR", "PLANES_ALL_TARGETS")
REFERENCE_STRATEGIES = ("Varo Final", "MILP", "Greedy", "VHS", "Pareto", "DQN")


# ------------------------------------------------------------------------------------------------ inputs


class GeneratorRebuild:
    """Per-day generator input rebuilt from the actual inventory flow, the way the saved 2026-07 snapshots were built.

    Rows: the six network nodes, the products of the processed upload; a (store, product) pair reported under more than
    one state code is excluded entirely (the snapshot summary's ``excluded_duplicate_rows``). Checked against the
    processed 07-31 inventory and against the saved 620 candidates.
    """

    def __init__(self, data_root: Path) -> None:
        folder = Path(data_root) / PROCESSED
        self.stores, self.products, self.routes, self.inventory_0731 = (
            pd.read_csv(folder / f"{name}.csv", dtype=ID_COLUMNS, encoding="utf-8-sig")
            for name in ("stores", "products", "routes", "inventory"))
        self.flow = pd.read_csv(Path(data_root) / sfv.INVENTORY, dtype={"date": str, "center_code": str, "product_code": str})

    def inventory(self, date: str) -> tuple[pd.DataFrame, int]:
        rows = self.flow[(self.flow["date"] == date) & self.flow["center_code"].isin(rv.NETWORK_NODES)
                         & self.flow["product_code"].isin(self.products["product_id"])]
        counts = rows.groupby(["center_code", "product_code"]).size()
        duplicated = set(counts[counts > 1].index)
        rows = rows[[(c, p) not in duplicated for c, p in zip(rows["center_code"], rows["product_code"])]]
        frame = pd.DataFrame({"store_id": rows["center_code"], "product_id": rows["product_code"],
                              "stock_qty": rows["stock_qty"], "sales_qty": rows["outbound_qty"], "snapshot_date": rows["date"]})
        return frame.reset_index(drop=True), len(duplicated)

    def upload(self, date: str) -> dict[str, pd.DataFrame]:
        from services.data_loader import normalize_loaded_data

        inventory, _ = self.inventory(date)
        return normalize_loaded_data({"stores": self.stores.copy(), "products": self.products.copy(),
                                      "inventory": inventory, "routes": self.routes.copy()})

    def reproduces_processed_0731(self) -> bool:
        rebuilt, _ = self.inventory("2026-07-31")
        left = rebuilt.set_index(["store_id", "product_id"])[["stock_qty", "sales_qty"]].sort_index()
        right = self.inventory_0731.set_index(["store_id", "product_id"])[["stock_qty", "sales_qty"]].sort_index()
        return left.astype(float).equals(right.astype(float))


def generator_pool(upload: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every candidate the generator scores before its MAX_CANDIDATES cut, in its own candidate_score order."""
    with mock.patch.object(cg, "MAX_CANDIDATES", 10 ** 9):
        frame, stats = cg.generate_candidates(dict(upload))
    frame = frame.copy() if frame is not None else pd.DataFrame()
    frame["generator_rank"] = range(1, len(frame) + 1)
    frame["candidate_score"] = [item["candidate_score"] for item in stats.get("candidates", [])][:len(frame)]
    return frame, stats


def lane_pool(upload: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Every reachable target per eligible (source, product), each with the generator's own quantity rule.

    The generator keeps one target per (source, product) and then the top MAX_CANDIDATES; this lifts both cuts and
    nothing else (same eligibility, need, quantity rule and 50 / 20 move caps as candidate_generator.generate_candidates).
    """
    stores, inventory = upload["stores"], upload["inventory"].copy()
    store_ids, dc_id = cg._store_ids_by_type(stores)
    direct = cg._route_lookup(upload["routes"])
    info = cg._product_info(upload["products"])
    store_col = next(c for c in ("store_id", "node_id") if c in inventory.columns)
    product_col = next(c for c in ("product_id", "item_id") if c in inventory.columns)
    stock_col = next(c for c in ("stock_qty", "current_stock", "quantity") if c in inventory.columns)
    expiry_col = next((c for c in ("days_to_expiry", "expiry_days") if c in inventory.columns), None)
    demand_col = next((c for c in ("avg_daily_sales", "sales_qty", "demand_qty") if c in inventory.columns), None)
    inv = inventory[inventory[store_col].astype(str).isin(store_ids)].copy()
    inv["store"], inv["product"] = inv[store_col].astype(str), inv[product_col].astype(str)
    inv["stock"] = inv[stock_col].map(cg._num)
    inv["demand"] = inv[demand_col].map(cg._num) if demand_col else 0.0
    inv["expiry"] = inv[expiry_col].map(cg.clean_numeric_value) if expiry_col else None
    stock_by = {(r.store, r.product): r.stock for r in inv.itertuples()}
    demand_by = {(r.store, r.product): r.demand for r in inv.itertuples()}
    median_stock = inv.groupby("product")["stock"].median().to_dict()
    rows = []
    for item in inv.itertuples():
        if item.product not in info or item.stock <= 0:
            continue
        expiry = None if item.expiry is None or pd.isna(item.expiry) else float(item.expiry)
        median = median_stock.get(item.product, item.stock)
        surplus = item.stock - median
        if not (surplus > 0 or (expiry is not None and expiry <= 7)):
            continue
        for target in store_ids:
            if target == item.store:
                continue
            resolved = cg._resolve_route(item.store, target, dc_id, direct)
            if resolved is None:
                continue
            need = max(0.0, median - stock_by.get((target, item.product), median)) + demand_by.get((target, item.product), 0.0) * 7
            source_surplus = max(1.0, surplus if surplus > 0 else item.stock * 0.3)
            target_need = max(1.0, need if need > 0 else source_surplus)
            cap = cg._SHORT_EXPIRY_CAP if (expiry is not None and expiry <= 3) else cg._MOVE_CAP
            rows.append({"product_id": item.product, "source_id": item.store, "target_id": target,
                         "route_type": resolved["route_type"], "dc_id": dc_id if resolved["route_type"] == "VIA_DC" else "",
                         "recommended_qty": float(int(max(1, min(source_surplus, target_need, item.stock, cap))))})
    frame = pd.DataFrame(rows)
    frame["route_id"] = [f"LANE{index:04d}" for index in range(1, len(frame) + 1)]
    return frame


def priced(frame: pd.DataFrame, date: str, inventory: pd.DataFrame) -> pd.DataFrame:
    """Official-tariff move cost (same engine as the pipeline) and the benchmark's median-based caps."""
    from services.real_transport_enrichment import enrich_real_transport

    data = frame.copy()
    data["snapshot_date"] = date
    data["dc_id"] = data.get("dc_id", pd.Series([""] * len(data))).fillna("")
    data = enrich_real_transport(data.reset_index(drop=True))
    data["move_cost"] = pd.to_numeric(data["move_cost"], errors="coerce")
    return rv.enrich_inventory_constraints(data, inventory)


def old_benchmark_caps(records: Sequence[Mapping[str, Any]], upload_inventory: pd.DataFrame) -> list[dict[str, Any]]:
    """Caps of the earlier saved 2026-07 MILP (generator conventions): median over the upload rows (duplicate pairs
    excluded) and need = max(0, median - target stock) + 7 x outbound. Checked against the 155 saved route rows."""
    stock = pd.to_numeric(upload_inventory["stock_qty"], errors="coerce").fillna(0.0)
    medians = stock.groupby(upload_inventory["product_id"].astype(str)).median().to_dict()
    rows = []
    for record in records:
        median = medians.get(str(record["product_id"]))
        if median is None:
            source_surplus = target_need = None
        else:
            source_surplus = max(0.0, float(record["source_stock"]) - median)
            target_need = max(0.0, median - float(record["target_stock"])) + 7.0 * float(record["target_daily_demand_proxy"])
        rows.append({**record, "median_stock": median, "source_surplus": source_surplus, "target_need_7d": target_need})
    return rows


# ------------------------------------------------------------------------------------------------ helpers


def effective_caps(caps: sf.CapTable) -> sf.CapTable:
    """Source caps reduced to the tightest per group, ties to the earlier kind in sf.SOURCE_KINDS (as sf.validate_plan
    applies them); missing caps dropped. Two models with the same effective caps have the same feasible set."""
    order = {kind: index for index, kind in enumerate(sf.CAP_KINDS)}
    result: sf.CapTable = {}
    for (kind, key), cap in sorted(caps.items(), key=lambda item: (order.get(item[0][0], 99), item[0][1])):
        if cap.value is None:
            continue
        kind_out = "SOURCE_CAP" if kind in sf.SOURCE_KINDS else kind
        current = result.get((kind_out, key))
        if current is None or float(cap.value) < float(current.value) - sf.EPS:
            result[(kind_out, key)] = sf.Cap(kind_out, key, float(cap.value), cap.provenance, cap.basis)
    return result


def _signature(date: str, records: Sequence[Mapping[str, Any]], caps: sf.CapTable, *, cost_basis: str = COST_FIXED,
               quantity_basis: str = mbi.FIXED_BINARY) -> dict[str, Any]:
    return mbi.comparison_signature(dataset=DATASET, date=date, candidates=records, constraints=CONSTRAINTS,
                                    caps=effective_caps(caps), cost_basis=cost_basis, quantity_basis=quantity_basis,
                                    selection_limit=MAX_ROUTES, objective=mbi.LEXICOGRAPHIC_OBJECTIVE)


def _plan(frame: pd.DataFrame | Sequence[Mapping[str, Any]], qty_field: str = "recommended_qty",
          cost_field: str = "move_cost") -> dict[str, Any]:
    data = pd.DataFrame(frame)
    if data.empty:
        return {"service": 0.0, "cost": 0.0, "route_ids": []}
    return {"service": float(pd.to_numeric(data[qty_field], errors="coerce").fillna(0).sum()),
            "cost": float(pd.to_numeric(data[cost_field], errors="coerce").fillna(0).sum()),
            "route_ids": data["route_id"].astype(str).tolist()}


def _t1_rows(plan: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(record["route_id"]): record for record in records}
    return [{**by_id[row["candidate_id"]], "allocated_qty": row["allocated_qty"], "allocated_move_cost": row["allocated_move_cost"]}
            for row in plan["rows"] if row["selection_status"] in (sf.SELECTED, sf.PARTIALLY_SELECTED)]


def _ids(values: Sequence[str]) -> str:
    return "|".join(sorted(map(str, values)))


def _measured(action: Any) -> tuple[Any, float, float | None]:
    """(result, elapsed ms, tracemalloc peak KB of this call)."""
    started_here = not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    tick = time.perf_counter()
    result = action()
    elapsed = (time.perf_counter() - tick) * 1000.0
    peak = tracemalloc.get_traced_memory()[1] - base
    if started_here:
        tracemalloc.stop()
    return result, elapsed, round(max(0, peak) / 1024, 1)


def process_peak_working_set_mb() -> float | None:
    """Peak working set of this process (Windows GetProcessMemoryInfo); None elsewhere."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in (
                            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # 64-bit pseudo handle, not a C int
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return round(counters.PeakWorkingSetSize / 1024 / 1024, 1)
    except (AttributeError, OSError):
        return None


def _relation(left: Sequence[str], right: Sequence[str]) -> str:
    a, b = set(map(str, left)), set(map(str, right))
    return "SAME_ROUTES" if a == b else f"DIFFERENT_ROUTES(common {len(a & b)}/{len(a | b)})"


# ------------------------------------------------------------------------------------------------ Suhyup 31 days


def _dqn_rows(day: pd.DataFrame, date: str, milp_service: float) -> dict[str, Any] | None:
    from services.action_consistency_validation import _dqn_model, _dqn_row

    model = _dqn_model()
    return None if model is None else _dqn_row(day, date, model, milp_service)


def run_suhyup(data_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    candidates = sfv.load_suhyup_candidates(data_root)
    reference = pd.read_csv(data_root / sfv.REFERENCE, dtype={"date": str})
    saved_daily = pd.read_csv(data_root / SAVED_MILP_DAILY, dtype={"snapshot_date": str})
    saved_routes = pd.read_csv(data_root / SAVED_MILP_ROUTES, dtype={"snapshot_date": str, "route_id": str, "product_id": str})
    rebuild = GeneratorRebuild(data_root)
    by_day, routes_out, gaps_out, regression = [], [], [], []
    old_cap_checks: list[bool] = []
    milp_ms, enum_ms = [], []
    with _env(REAL_DATA_ROOT_ENV, str(data_root)):
        recompute = sfv.TariffRecompute(candidates)
        transport = _transport_provenance(candidates)
        for date in sorted(candidates["snapshot_date"].astype(str).unique()):
            day = sfv.ranked_day(candidates, date)
            records = rv._records(day)
            caps = sf.caps_from_columns(records, sfv.SUHYUP_CAP_SPEC)
            milp_caps = sf.caps_from_columns(records, MILP_CAP_SPEC)
            upload = rebuild.upload(date)
            full, _ = generator_pool(upload)
            lanes = lane_pool(upload)

            tick = time.perf_counter()
            milp = rv.lexicographic_milp(day)
            milp_ms.append((time.perf_counter() - tick) * 1000.0)
            strict_tol = rv.lexicographic_milp(day, options={"mip_rel_gap": 0.0})
            evidence = mbi.lexicographic_solver_evidence(milp, [r["recommended_qty"] for r in records])
            tick = time.perf_counter()
            enumeration = mbi.enumerate_lexicographic_optimum(records, max_routes=MAX_ROUTES)
            enum_ms.append((time.perf_counter() - tick) * 1000.0)
            milp_plan = _plan(milp["selected"])
            enum_proven = (enumeration["status"] == "ENUMERATED"
                           and abs(enumeration["optimal_service"] - milp_plan["service"]) <= 1e-9
                           and abs(enumeration["optimal_cost"] - milp_plan["cost"]) <= 1e-6)
            scope = mbi.benchmark_scope(solver_status=evidence["solver_status"], pool_size=len(records), pool_total=len(full),
                                        cap_provenance=[cap.provenance for cap in milp_caps.values()])

            legacy = day.sort_values(["varo_final_rank", "route_id"], kind="mergesort").head(MAX_ROUTES)
            legacy_ids = legacy["route_id"].astype(str).tolist()
            plans = {
                "SF_BENCHMARK_ALL_OR_NOTHING": sf.select_shared_feasible(
                    records, caps, mode=sf.BENCHMARK_PROXY, max_routes=MAX_ROUTES, partial_policy=sf.PARTIAL_NONE,
                    legacy_ids=legacy_ids),
                "SF_BENCHMARK_PARTIAL": sf.select_shared_feasible(
                    records, caps, mode=sf.BENCHMARK_PROXY, max_routes=MAX_ROUTES, partial_policy=sf.PARTIAL_IF_SAFE,
                    cost_recompute=recompute, legacy_ids=legacy_ids),
                "SF_STRICT_ACTUAL": sf.select_shared_feasible(
                    records, caps, mode=sf.STRICT_ACTUAL, max_routes=MAX_ROUTES, partial_policy=sf.PARTIAL_IF_SAFE,
                    cost_recompute=recompute, legacy_ids=legacy_ids),
            }
            refs = sfv._reference_rows(day, date)
            dqn = _dqn_rows(day, date, float(refs["MILP"]["service_qty"]))
            if dqn is not None:
                refs["DQN"] = dqn
            for strategy, row in refs.items():
                saved = reference[(reference["date"] == date) & (reference["strategy"] == strategy)].iloc[0]
                regression.append({
                    "date": date, "strategy": strategy,
                    "equal": abs(float(row["service_qty"]) - float(saved["service_qty"])) <= 1e-6
                    and abs(float(row["total_cost"]) - float(saved["total_cost"])) <= 1e-6
                    and str(row["route_ids"]) == str(saved["route_ids"])
                    and int(row["feasibility_violations"]) == int(saved["feasibility_violations"])
                    and str(row["status"]) == str(saved["status"])})

            # earlier saved 2026-07 MILP (generator-convention caps)
            old_day = saved_daily[saved_daily["snapshot_date"] == date].iloc[0]
            old_ids = str(old_day["milp_route_ids"]).split("|")
            old_records = old_benchmark_caps(records, upload["inventory"])
            old_saved_rows = saved_routes[saved_routes["snapshot_date"] == date]
            old_by_id = {str(r["route_id"]): r for r in old_records}
            for row in old_saved_rows.itertuples():
                rebuilt = old_by_id[str(row.route_id)]
                old_cap_checks.append(abs(rebuilt["median_stock"] - row.median_stock) <= 1e-6
                                      and abs(rebuilt["source_surplus"] - row.source_surplus) <= 1e-6
                                      and abs(rebuilt["target_need_7d"] - row.target_need_7d) <= 1e-6)
            old_caps = sf.caps_from_columns(old_records, MILP_CAP_SPEC)
            old_plan = {"service": float(old_day["milp_served_qty"]), "cost": float(old_day["milp_transport_cost"]),
                        "route_ids": old_ids}
            new_sig = _signature(date, records, milp_caps)
            old_sig = _signature(date, records, old_caps)
            new_effective, old_effective = effective_caps(milp_caps), effective_caps(old_caps)
            differing_groups = sorted(
                f"{kind}:{'/'.join(key)}" for (kind, key), cap in new_effective.items()
                if (kind, key) not in old_effective or abs(float(cap.value) - float(old_effective[(kind, key)].value)) > 1e-6)
            stock_tighter = sum(
                1 for (kind, key), cap in caps.items() if kind == sf.SOURCE_STOCK and cap.value is not None
                and caps.get((sf.SOURCE_SURPLUS, key)) is not None and caps[(sf.SOURCE_SURPLUS, key)].value is not None
                and float(cap.value) < float(caps[(sf.SOURCE_SURPLUS, key)].value) - sf.EPS)

            def selected_by_ids(ids: Sequence[str]) -> list[dict[str, Any]]:
                wanted = set(map(str, ids))
                return [r for r in records if str(r["route_id"]) in wanted]

            old_under_new = mbi.independent_check(selected_by_ids(old_ids), caps, max_routes=MAX_ROUTES)

            # plans compared with the MILP
            compared: dict[str, dict[str, Any]] = {}
            for strategy in ("Varo Final", "Greedy", "VHS", "Pareto", "DQN"):
                if strategy not in refs:
                    continue
                ids = str(refs[strategy]["route_ids"]).split("|") if refs[strategy]["route_ids"] else []
                rows = selected_by_ids(ids)
                compared[f"OFFLINE_{strategy.upper().replace(' ', '_')}"] = {
                    "plan": _plan(rows), "rows": rows, "signature": _signature(date, records, milp_caps),
                    "check": mbi.independent_check(rows, caps, max_routes=MAX_ROUTES)}
            for name, plan in plans.items():
                rows = _t1_rows(plan, records)
                partial = name == "SF_BENCHMARK_PARTIAL"
                plan_caps = caps if plan["mode"] == sf.BENCHMARK_PROXY else {
                    slot: cap for slot, cap in caps.items() if cap.strict_accepted}
                compared[name] = {
                    "plan": _plan(rows, "allocated_qty", "allocated_move_cost") if rows else {"service": 0.0, "cost": 0.0, "route_ids": []},
                    "rows": rows, "plan_status": plan["plan_status"],
                    "signature": _signature(date, records, plan_caps, cost_basis=COST_PARTIAL if partial else COST_FIXED,
                                            quantity_basis=mbi.PARTIAL_ALLOWED if partial else mbi.FIXED_BINARY),
                    "check": mbi.independent_check(rows, caps, max_routes=MAX_ROUTES, allocated_field="allocated_qty",
                                                   quantity_basis=mbi.PARTIAL_ALLOWED if partial else mbi.FIXED_BINARY)}
            legacy_rows = rv._records(legacy)
            compared["LEGACY_TOP5_SLICE"] = {"plan": _plan(legacy_rows), "rows": legacy_rows,
                                             "signature": _signature(date, records, milp_caps),
                                             "check": mbi.independent_check(legacy_rows, caps, max_routes=MAX_ROUTES)}
            compared["SAVED_2026_07_MILP"] = {"plan": old_plan, "rows": selected_by_ids(old_ids), "signature": old_sig,
                                              "check": old_under_new}
            for name, item in compared.items():
                gap = mbi.decision_gap(item["plan"], milp_plan, plan_signature=item["signature"], reference_signature=new_sig,
                                       reference_status=evidence["solver_status"], reference_enumeration_proven=enum_proven,
                                       plan_feasible=item["check"]["passed"])
                if item.get("plan_status") == sf.INSUFFICIENT_CAP_DATA:
                    gap.update(comparison_status=mbi.INSUFFICIENT_DATA, decision_gap_status=mbi.GAP_NOT_COMPUTABLE,
                               reason_codes=mbi._unique([*gap["reason_codes"], "PROXY_CAPS", "MISSING_CAP"]))
                gaps_out.append({
                    "date": date, "plan": name, "reference": "MILP (lexicographic, saved 20-candidate pool)",
                    "plan_service": item["plan"]["service"], "plan_cost": item["plan"]["cost"],
                    "reference_service": milp_plan["service"], "reference_cost": milp_plan["cost"],
                    "plan_route_ids": _ids(item["plan"]["route_ids"]), "reference_route_ids": _ids(milp_plan["route_ids"]),
                    "route_relation": _relation(item["plan"]["route_ids"], milp_plan["route_ids"]),
                    "route_ids_in_some_alternative_optimum": sorted(item["plan"]["route_ids"]) in enumeration.get("optimal_selections", []),
                    "plan_independent_check_passed": item["check"]["passed"],
                    "plan_check_violations": item["check"]["violation_count"],
                    "plan_target_excess_qty": item["check"]["target_excess_qty"],
                    "plan_status": item.get("plan_status"),
                    "comparison_status": gap["comparison_status"], "signature_differences": "|".join(gap["signature_differences"]),
                    "decision_gap_status": gap["decision_gap_status"],
                    "decision_service_gap": gap["decision_service_gap"], "decision_service_gap_pct": gap["decision_service_gap_pct"],
                    "decision_cost_gap": gap["decision_cost_gap"], "decision_cost_gap_pct": gap["decision_cost_gap_pct"],
                    "zero_gap_flags": "|".join(gap["zero_gap_flags"]), "reason_codes": "|".join(gap["reason_codes"]),
                    "reference_solver_status": evidence["solver_status"],
                    "reference_stage1_mip_gap": evidence["stage1"]["solver_mip_gap"],
                    "reference_stage2_mip_gap": evidence["stage2"]["solver_mip_gap"],
                    "reference_enumeration_proven": enum_proven,
                    "raw_cost_difference_informational": round(item["plan"]["cost"] - milp_plan["cost"], 6),
                })
            gap_by_plan = {row["plan"]: row for row in gaps_out if row["date"] == date}

            # route-level comparison
            membership = {
                "MILP": milp_plan["route_ids"], "OFFLINE_VARO_FINAL": compared["OFFLINE_VARO_FINAL"]["plan"]["route_ids"],
                "T1_ALL_OR_NOTHING": compared["SF_BENCHMARK_ALL_OR_NOTHING"]["plan"]["route_ids"],
                "T1_PARTIAL": compared["SF_BENCHMARK_PARTIAL"]["plan"]["route_ids"], "LEGACY_TOP5": legacy_ids,
                "SAVED_2026_07_MILP": old_ids, "ENUMERATION_TIE_BREAK": enumeration.get("tie_break_route_ids", []),
            }
            partial_alloc = {row["route_id"]: row["allocated_qty"] for row in compared["SF_BENCHMARK_PARTIAL"]["rows"]}
            in_any_optimum = {route for selection in enumeration.get("optimal_selections", []) for route in selection}
            union = sorted(set().union(*map(set, membership.values())))
            for route in union:
                record = next(r for r in records if str(r["route_id"]) == route)
                old = old_by_id[route]
                routes_out.append({
                    "date": date, "route_id": route, "product_id": record["product_id"], "source_id": record["source_id"],
                    "target_id": record["target_id"], "recommended_qty": record["recommended_qty"], "move_cost": record["move_cost"],
                    **{f"in_{name.lower()}": route in set(map(str, ids)) for name, ids in membership.items()},
                    "t1_partial_allocated_qty": partial_alloc.get(route),
                    "in_any_alternative_optimum": route in in_any_optimum,
                    "varo_final_rank": record["varo_final_rank"], "vhs_rank": record["vhs_rank"],
                    "source_surplus_cap": record["source_surplus"], "target_need_cap": record["target_need_7d"],
                    "old_target_need_cap": old["target_need_7d"], "old_source_surplus_cap": old["source_surplus"],
                    "median_stock": record["median_stock"], "old_median_stock": old["median_stock"],
                    "cap_provenance": "SOURCE_SURPLUS=PROXY|TARGET_NEED=PROXY",
                    "cost_provenance": transport.get((date, route), {}).get("cost_provenance"),
                    "proxy_vehicle_count": transport.get((date, route), {}).get("proxy_vehicle_count"),
                })

            varo_gap = gap_by_plan["OFFLINE_VARO_FINAL"]
            t1_gap = gap_by_plan["SF_BENCHMARK_ALL_OR_NOTHING"]
            stage1, stage2 = evidence["stage1"], evidence["stage2"]
            proxy_routes = sum(int(transport.get((date, str(r["route_id"])), {}).get("proxy_vehicle_count") or 0) > 0 for r in records)
            by_day.append({
                "date": date, "generator_pool_before_cut": len(full), "lane_pool_all_targets": len(lanes),
                "milp_input_pool": len(records), "varo_final_pool": len(records), "t1_pool": len(records),
                "candidate_cutoff_applied": len(full) > len(records), "candidates_cut": len(full) - len(records),
                "cutoff_basis": "candidate_generator.MAX_CANDIDATES=20 by candidate_score (one target per source-product)",
                "solver": "SciPy milp / HiGHS", "solver_status": evidence["solver_status"],
                "termination_reason": evidence["termination_reason"],
                "stage1_status_code": stage1.get("solver_status_code"), "stage2_status_code": stage2.get("solver_status_code"),
                "stage1_service_objective": stage1["objective"], "stage1_bound": stage1["bound"],
                "stage1_mip_gap": stage1["solver_mip_gap"], "stage1_nodes": stage1.get("node_count"),
                "stage2_cost_objective_with_tiebreak": stage2["objective"], "stage2_bound": stage2["bound"],
                "stage2_mip_gap": stage2["solver_mip_gap"], "stage2_mip_gap_abs": stage2["solver_mip_gap_abs"],
                "stage2_tie_break": evidence["stage2_tie_break"],
                "service_optimum_exact_by_integrality": evidence["service_optimum_exact_by_integrality"],
                "solver_tolerance": json.dumps(evidence["solver_tolerance"]), "time_limit_s": evidence["time_limit_s"],
                "milp_ms_both_stages": round(milp_ms[-1], 3),
                "strict_tolerance_same_selection": set(strict_tol["selected"]["route_id"].astype(str)) == set(milp_plan["route_ids"]),
                "enumeration_status": enumeration["status"], "enumeration_combinations": enumeration["combinations"],
                "enumeration_nodes_visited": enumeration.get("nodes_visited"), "enumeration_ms": round(enum_ms[-1], 3),
                "enumeration_optimal_service": enumeration.get("optimal_service"),
                "enumeration_optimal_cost": enumeration.get("optimal_cost"),
                "alternative_optima": enumeration.get("alternative_optima"),
                "tie_break_unique": enumeration.get("tie_break_unique"),
                "milp_equals_enumeration_optimum": enum_proven,
                "milp_equals_enumeration_tie_break": sorted(milp_plan["route_ids"]) == enumeration.get("tie_break_route_ids"),
                "benchmark_scope": scope["benchmark_scope"], "scope_flags": "|".join(scope["scope_flags"]),
                "optimality_claim": scope["optimality_claim"],
                "milp_service_qty": milp_plan["service"], "milp_move_cost": milp_plan["cost"],
                "milp_route_ids": _ids(milp_plan["route_ids"]),
                "varo_final_route_ids": _ids(compared["OFFLINE_VARO_FINAL"]["plan"]["route_ids"]),
                "t1_all_or_nothing_route_ids": _ids(compared["SF_BENCHMARK_ALL_OR_NOTHING"]["plan"]["route_ids"]),
                "t1_partial_route_ids": _ids(compared["SF_BENCHMARK_PARTIAL"]["plan"]["route_ids"]),
                "t1_partial_count": plans["SF_BENCHMARK_PARTIAL"]["partial_count"],
                "t1_strict_plan_status": plans["SF_STRICT_ACTUAL"]["plan_status"],
                "legacy_top5_route_ids": _ids(legacy_ids), "saved_2026_07_milp_route_ids": _ids(old_ids),
                "varo_vs_milp_routes": _relation(compared["OFFLINE_VARO_FINAL"]["plan"]["route_ids"], milp_plan["route_ids"]),
                "milp_independent_violations": mbi.independent_check(rv._records(milp["selected"]), caps, max_routes=MAX_ROUTES)["violation_count"],
                "varo_final_independent_violations": compared["OFFLINE_VARO_FINAL"]["check"]["violation_count"],
                "t1_independent_violations": compared["SF_BENCHMARK_ALL_OR_NOTHING"]["check"]["violation_count"],
                "legacy_top5_independent_violations": compared["LEGACY_TOP5_SLICE"]["check"]["violation_count"],
                "strict_unverifiable": "|".join(compared["OFFLINE_VARO_FINAL"]["check"]["strict_unverifiable_constraints"]),
                "cap_provenance": "|".join(sorted({f"{k}={c.provenance}" for (k, _), c in caps.items()})),
                "cost_provenance": "DERIVED_REAL (official tariff x OSRM distance x derived unit weight)",
                "candidates_with_proxy_vehicle": proxy_routes,
                "milp_signature": new_sig["signature"],
                "varo_vs_milp_comparison": varo_gap["comparison_status"],
                "varo_decision_service_gap": varo_gap["decision_service_gap"], "varo_decision_cost_gap": varo_gap["decision_cost_gap"],
                "varo_zero_gap_flags": varo_gap["zero_gap_flags"],
                "t1_vs_milp_comparison": t1_gap["comparison_status"],
                "t1_decision_service_gap": t1_gap["decision_service_gap"], "t1_decision_cost_gap": t1_gap["decision_cost_gap"],
                "saved_2026_07_milp_service": old_plan["service"], "saved_2026_07_milp_cost": old_plan["cost"],
                "saved_vs_new_milp_comparison": gap_by_plan["SAVED_2026_07_MILP"]["comparison_status"],
                "saved_vs_new_cap_groups_differing": "|".join(differing_groups),
                "saved_milp_plan_violations_under_new_caps": old_under_new["violation_count"],
                "saved_milp_stage_status": f"{int(old_day['milp_stage1_status'])}/{int(old_day['milp_stage2_status'])}",
                "source_groups_stock_tighter_than_surplus": stock_tighter,
            })
    return {"by_day": pd.DataFrame(by_day), "routes": pd.DataFrame(routes_out), "gaps": pd.DataFrame(gaps_out),
            "regression": pd.DataFrame(regression), "old_cap_checks": old_cap_checks, "transport": transport,
            "runtime": {"wall_seconds": round(time.perf_counter() - started, 3),
                        "milp_ms_per_day": {"median": round(float(pd.Series(milp_ms).median()), 3), "max": round(max(milp_ms), 3),
                                            "total": round(sum(milp_ms), 3)},
                        "enumeration_ms_per_day": {"median": round(float(pd.Series(enum_ms).median()), 3),
                                                   "max": round(max(enum_ms), 3)}}}


def _transport_provenance(candidates: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    """Re-price the saved 620 rows with the official-tariff engine: reproduction, proxy vehicles, unit-weight status."""
    from services.real_transport_enrichment import enrich_real_transport

    frame = candidates[["snapshot_date", "route_id", "product_id", "source_id", "target_id", "recommended_qty"]].reset_index(drop=True)
    priced_rows = enrich_real_transport(frame)
    saved = candidates["move_cost"].reset_index(drop=True)
    result = {}
    for index, row in priced_rows.iterrows():
        result[(str(row["snapshot_date"]), str(row["route_id"]))] = {
            "reproduced": bool(row.get("real_transport_applied")) and abs(float(row["move_cost"]) - float(saved[index])) <= 1e-6,
            "proxy_vehicle_count": int(row.get("proxy_vehicle_count") or 0),
            "unit_weight_status": row.get("unit_weight_status"),
            "cost_provenance": row.get("transport_cost_provenance"),
            "vehicle_mix": row.get("vehicle_mix"),
            "lane": (str(row["source_id"]), str(row["target_id"])),
        }
    return result


# ------------------------------------------------------------------------------------------------ cutoff experiment


def run_cutoff(data_root: Path) -> dict[str, Any]:
    """Same objective, caps, limit and 0/1 granularity on larger pools. Separate experiment, never the benchmark."""
    started = time.perf_counter()
    saved = sfv.load_suhyup_candidates(data_root)
    inventory = pd.read_csv(data_root / sfv.INVENTORY, dtype={"date": str, "center_code": str, "product_code": str})
    rebuild = GeneratorRebuild(data_root)
    rows, generator_rows = [], []
    with _env(REAL_DATA_ROOT_ENV, str(data_root)):
        for date in sorted(saved["snapshot_date"].astype(str).unique()):
            upload = rebuild.upload(date)
            full, stats = generator_pool(upload)
            full = priced(full, date, inventory)
            lanes = priced(lane_pool(upload), date, inventory)
            day_saved = saved[saved["snapshot_date"] == date]
            key =lambda f: sorted(zip(f["route_id"].astype(str), f["product_id"].astype(str), f["source_id"].astype(str),
                                       f["target_id"].astype(str), f["recommended_qty"].astype(float),
                                       pd.to_numeric(f["move_cost"]).astype(float)))
            reproduced = key(full.head(len(day_saved))) == key(day_saved)
            generator_keys = set(zip(full["product_id"].astype(str), full["source_id"].astype(str), full["target_id"].astype(str),
                                     full["recommended_qty"].astype(float)))
            lane_keys = set(zip(lanes["product_id"].astype(str), lanes["source_id"].astype(str), lanes["target_id"].astype(str),
                                lanes["recommended_qty"].astype(float)))
            scores = list(full["candidate_score"])
            generator_rows.append({
                "date": date, "generator_pool": len(full), "lane_pool": len(lanes), "saved_pool": len(day_saved),
                "top20_reproduces_saved_candidates": reproduced, "generator_lanes_inside_lane_pool": generator_keys <= lane_keys,
                "score_rank20": scores[19] if len(scores) >= 20 else None, "score_rank21": scores[20] if len(scores) >= 21 else None,
                "lanes_without_cost": int(lanes["move_cost"].isna().sum()), "route_deferred": stats.get("route_deferred"),
            })
            ranks = dict(zip(full["route_id"].astype(str), full["generator_rank"]))
            pools = {"P20_SAVED": full.head(20), "P30_SCORE": full.head(30), "PALL_GENERATOR": full,
                     "PLANES_ALL_TARGETS": lanes[lanes["move_cost"].notna()]}
            base_cost = None
            for name, pool in pools.items():
                pool = pool.reset_index(drop=True)
                records = rv._records(pool)
                caps = sf.caps_from_columns(records, sfv.SUHYUP_CAP_SPEC)
                result, elapsed, peak_kb = _measured(lambda: rv.lexicographic_milp(pool))
                evidence = mbi.lexicographic_solver_evidence(result, [r["recommended_qty"] for r in records])
                plan = _plan(result["selected"])
                tick = time.perf_counter()
                enumeration = mbi.enumerate_lexicographic_optimum(records, max_routes=MAX_ROUTES)
                enum_elapsed = (time.perf_counter() - tick) * 1000.0
                varo_key = sf.select_shared_feasible(records, caps, mode=sf.BENCHMARK_PROXY, max_routes=MAX_ROUTES,
                                                     partial_policy=sf.PARTIAL_NONE)
                varo_plan = {"service": float(varo_key["total_allocated_qty"]), "cost": float(varo_key["total_move_cost"] or 0.0)}
                check = mbi.independent_check(rv._records(result["selected"]), caps, max_routes=MAX_ROUTES)
                truncated = len(pool) < len(full) if name != "PLANES_ALL_TARGETS" else False
                scope = mbi.benchmark_scope(solver_status=evidence["solver_status"], pool_size=len(pool),
                                            pool_total=len(full) if name != "PLANES_ALL_TARGETS" else len(pool),
                                            cap_provenance=[cap.provenance for cap in caps.values()])
                if name == "P20_SAVED":
                    base_cost = plan["cost"]
                rows.append({
                    "date": date, "pool": name, "pool_size": len(pool), "generator_pool_before_cut": len(full),
                    "pool_truncated_vs_generator": truncated,
                    "pool_rule": {"P20_SAVED": "generator top-20 by candidate_score (the saved benchmark pool)",
                                  "P30_SCORE": "generator top-30 by candidate_score",
                                  "PALL_GENERATOR": "every generator candidate (one target per source-product)",
                                  "PLANES_ALL_TARGETS": "every reachable target per eligible source-product"}[name],
                    "solver_status": evidence["solver_status"], "stage2_mip_gap": evidence["stage2"]["solver_mip_gap"],
                    "milp_service_qty": plan["service"], "milp_move_cost": plan["cost"],
                    "cost_delta_vs_p20": None if base_cost is None else round(plan["cost"] - base_cost, 6),
                    "cost_delta_pct_vs_p20": None if not base_cost else round(100.0 * (plan["cost"] - base_cost) / base_cost, 6),
                    "milp_route_ids": _ids(plan["route_ids"]),
                    "milp_routes_beyond_generator_rank20": sum(int(ranks.get(route, 0)) > 20 for route in plan["route_ids"])
                    if name != "PLANES_ALL_TARGETS" else None,
                    "enumeration_status": enumeration["status"], "enumeration_combinations": enumeration["combinations"],
                    "enumeration_matches_milp": None if enumeration["status"] != "ENUMERATED" else (
                        abs(enumeration["optimal_service"] - plan["service"]) <= 1e-9
                        and abs(enumeration["optimal_cost"] - plan["cost"]) <= 1e-6),
                    "alternative_optima": enumeration.get("alternative_optima"),
                    "varo_key_greedy_service": varo_plan["service"], "varo_key_greedy_cost": varo_plan["cost"],
                    "varo_key_greedy_equals_milp": abs(varo_plan["service"] - plan["service"]) <= 1e-9
                    and abs(varo_plan["cost"] - plan["cost"]) <= 1e-6,
                    "milp_independent_violations": check["violation_count"],
                    "benchmark_scope": scope["benchmark_scope"], "optimality_claim": scope["optimality_claim"],
                    "milp_ms": round(elapsed, 3), "milp_peak_kb": peak_kb, "enumeration_ms": round(enum_elapsed, 3),
                })
    return {"rows": pd.DataFrame(rows), "generator": pd.DataFrame(generator_rows),
            "runtime": {"wall_seconds": round(time.perf_counter() - started, 3)}}


# ------------------------------------------------------------------------------------------------ in-app gap path


def run_app_gap_e2e(data_root: Path) -> dict[str, Any]:
    """The Suhyup 2026-07-31 upload through build_v2_state, then the in-app optimality gap with default settings."""
    from services.optimality_gap_service import build_optimality_settings, clear_optimality_cache, run_optimality_gap
    from services.seller_loss_input_validation import _run, load_suhyup_upload

    upload = load_suhyup_upload(data_root)
    state = _run(upload, data_root)
    clear_optimality_cache()
    with _env(REAL_DATA_ROOT_ENV, str(data_root)):
        result = run_optimality_gap(state["recommendations"], upload, build_optimality_settings(), "suhyup-20260731-t3")
    contract = result["benchmark_integrity"]
    return {"recommendation_count": len(state["recommendations"]), "search_status": result["search"].get("status"),
            "search_method": result["search"].get("method"), "gap": {k: result["gap"].get(k) for k in ("available", "label", "reason")},
            "excluded_reason_counts": result["summary"]["exclusion_reason_counts"], "benchmark_integrity": contract}


# ------------------------------------------------------------------------------------------------ static tables


def constraint_comparison(by_day: pd.DataFrame, transport: Mapping[tuple, Mapping[str, Any]]) -> pd.DataFrame:
    """Which constraint each system actually applies (from the code paths read for T3), with measured evidence."""
    proxy = sum(1 for item in transport.values() if item["proxy_vehicle_count"] > 0)
    proxy_lanes = len({item["lane"] for item in transport.values() if item["proxy_vehicle_count"] > 0})
    stock_tighter = int(by_day["source_groups_stock_tighter_than_surplus"].sum())
    rows = [
        ("source surplus", "source_surplus = max(0, stock - median) PROXY", "same column via _selection_violation",
         "SOURCE_SURPLUS PROXY", "rejected: PROXY not accepted", "available_transfer_stock... else total stock (v1)",
         "generator convention: median without duplicate pairs", "none", True,
         "suhyup_algorithm_revalidation.enrich_inventory_constraints; shared_feasibility_validation.SUHYUP_CAP_SPEC"),
        ("source on-hand inventory", "implied (surplus <= stock)", "implied", "SOURCE_AVAILABLE_STOCK DIRECT_REAL (never tighter)",
         "DIRECT_REAL accepted", "total stock cap when no movable column", "implied", "generator qty <= stock only", True,
         f"effective cap = tightest source cap; {stock_tighter} source groups over 31 days where stock < surplus"),
        ("target need", "target_need_7d = max(0, median + 7 x outbound - stock) PROXY", "same column",
         "TARGET_NEED PROXY", "rejected: PROXY", "max(7-day demand - stock, 0) (v1)",
         "max(0, median - stock) + 7 x outbound (generator formula)", "none", True,
         "same name, three formulas: see target_need definitions in the JSON"),
        ("route / vehicle capacity", "not in model", "not checked", "ROUTE_CAPACITY MISSING (no unit-compatible field)",
         "MISSING", "applied only if route_capacity_qty/vehicle_capacity_qty/max_load_qty exists", "not in model", "none",
         True, "0/620 candidates carry a quantity-unit capacity; the tariff engine sizes vehicles per quantity"),
        ("DC capacity", "not in model", "not checked", "DC_CAPACITY (no DC in Suhyup)", "n/a",
         "applied only if a DC capacity column exists", "not in model", "none", True, "all 620 routes DIRECT"),
        ("duplicate allocation", "one move per (product, source, target, route_type, dc)", "same key",
         "one move per lane + candidate id", "same", "same duplicate key", "same", "generator dedupes", True,
         "lexicographic_milp duplicate_groups; sf validate_plan duplicate_lane_count"),
        ("product matching", "candidate carries product; no stock check", "no stock check",
         "source stock cap per (source, product)", "same", "inventory row lookup", "same as MILP", "none", False,
         "independent check: product held at source on every selected route"),
        ("Top-N / max_routes", "sum x <= 5 (MAX_DAILY_ROUTES)", "stop at 5", "max_routes=5", "max_routes=5",
         "DEFAULT_MAX_ROUTES=5 (user setting)", "5", "display slice of 5 (not a constraint)", True,
         "5 is CONFIG; no data basis"),
        ("quantity granularity", "fixed recommended_qty, x in {0,1}", "fixed qty", "ALL_OR_NOTHING: fixed; PARTIAL: split",
         "split if safe", "fixed qty, x in {0,1}", "fixed qty", "fixed qty", True,
         "only SF_BENCHMARK_ALL_OR_NOTHING shares MILP granularity"),
        ("integer quantity", "qty integers (generator int())", "same", "integers kept integer", "same", "as uploaded", "same",
         "same", True, "stage 1 exact by integrality"),
        ("cost model", "move_cost at recommended_qty (QUANTITY_SPECIFIC tariff)", "same", "same; PARTIAL recomputes tariff",
         "same", "not used (saving objective)", "same costs 155/155", "same", True,
         f"official tariff x OSRM distance; {proxy}/620 candidate rows ({proxy_lanes} lanes) use a proxy vehicle class"),
        ("objective", "lexicographic: max service, then min cost", "Varo Final order greedy (qty desc, cost asc, vhs)",
         "Varo Final key greedy with executable qty", "same", "max expected_saving", "lexicographic", "rank slice",
         False, "MILP optimizes; Varo Final/T1 order greedily; app gap optimizes saving"),
        ("candidate pool", "saved 20/day (generator top-20 by candidate_score)", "same 20", "same 20", "same 20",
         "top candidate_limit=20 by varo_final_rank", "same 20", "same 20", True,
         f"generator pool before the cut: {int(by_day['generator_pool_before_cut'].min())}-"
         f"{int(by_day['generator_pool_before_cut'].max())} per day"),
        ("cutline / time window flags", "ignored", "ignored", "explicit False only ('거리 초과' is not False)", "same",
         "explicit False only", "ignored", "VHS penalty only", True, "cutline_passed='거리 초과' on all 620 (T1 limitation)"),
    ]
    columns = ("constraint", "offline_milp", "offline_varo_final", "t1_sf_benchmark", "t1_sf_strict", "app_gap_milp",
               "saved_2026_07_milp", "production_top5", "same_in_milp_and_t1_all_or_nothing", "evidence")
    return pd.DataFrame(rows, columns=columns)


TARGET_NEED_DEFINITIONS = {
    "candidate_generator (quantity rule, saved 2026-07 MILP)": "max(0, median - target_stock) + 7 x target daily demand; "
        "target_stock defaults to the median when the target has no row; a need of 0 is replaced by the source surplus",
    "suhyup_algorithm_revalidation (21_ benchmark, T1 BENCHMARK_PROXY)": "max(0, median + 7 x outbound - target_stock)",
    "optimality_gap_service shared-feasibility-v1 (app gap, T1 pipeline caps)": "max(7-day demand - stock, 0), 7-day demand from "
        "demand_qty / demand_forecast_7d / sales_7d, else avg_daily_sales or sales_qty x 7, else sales_30d / 30 x 7",
    "median": "generator and saved 2026-07 MILP: median over upload rows with duplicated (store, product) state pairs "
              "excluded; revalidation: median over node totals with state codes summed",
}
SOURCE_SURPLUS_DEFINITIONS = {
    "candidate_generator": "max(1, stock - median) (stock x 0.3 when eligible only by expiry); quantity cap 50 (20 if expiry <= 3)",
    "suhyup_algorithm_revalidation": "max(0, stock - median), stock summed over state codes",
    "optimality_gap_service v1": "available_transfer_stock / transferable_stock / source_surplus / surplus_qty / excess_stock, "
                                 "else total stock",
}


# ------------------------------------------------------------------------------------------------ claims audit

CLAIM_RULES: tuple[tuple[str, str, str, str, str, str], ...] = (
    # claim_id, regex, file glob, verdict, reason, suggested wording
    ("M01", r"최적성 Gap", "README_V2.md|APP_SUMMARY.md|DEPLOY_CHECKLIST.md|pages/validation.py|pages/strategy_comparison.py",
     "NEEDS_QUALIFIER", "in-app gap: expected_saving objective over the top candidate_limit candidates by Varo rank, "
     "v1 caps; it is not the Suhyup service-then-cost benchmark and is not computable on Suhyup (saving 0)",
     "'최적성 Gap(절감액 기준, 상위 N개 후보 안)'"),
    ("M02", r"정확 최적해", "services/optimality_gap_service.py", "NEEDS_QUALIFIER",
     "proven within HiGHS mip_rel_gap 1e-4 and inside the candidate_limit pool cut by Varo rank",
     "'후보 상위 N개 안의 정확 최적해(허용오차 1e-4)'"),
    ("M03", r"\"최적 조합\"|'최적 조합|최적 조합 절감액", "services/optimality_gap_service.py|pages/validation.py",
     "NEEDS_QUALIFIER", "optimum of the restricted saving model only", "'후보 상위 N개 안 최적 조합'"),
    ("M04", r"최적해 인증", "pages/validation.py", "NEEDS_QUALIFIER", "certifies the restricted model, not the network",
     "'제한 후보 최적해 인증'"),
    ("M05", r"MILP: 서비스 최대화 후 비용 최소화", "README_V2.md", "SAFE",
     "matches lexicographic_milp (stage 1 service, stage 2 cost) and says offline benchmark", "-"),
    ("M06", r"정확 최적화로 표현하지 않습니다", "pages/validation.py", "SAFE", "explicit non-claim", "-"),
    ("M07", r"확정 최적성 Gap이 아닙니다|확정 최적성 Gap으로 해석할 수 없습니다", "pages/validation.py", "SAFE",
     "limited search is not presented as exact", "-"),
    ("M08", r"자동 가중치 최적화", "services/vhs_score_engine.py|services/analysis_pipeline.py", "NEEDS_QUALIFIER",
     "heuristic weight rule, no objective is optimized (audit C07)", "'VHS 자동 가중치(규칙 기반)'"),
    ("M09", r"return \"최적\"|\"최적\" if", "services/vhs_score_engine.py|services/analysis_pipeline.py", "NEEDS_QUALIFIER",
     "VHS score grade (>= 80), not an optimization result (audit C20)", "'최상 등급'"),
    ("M10", r"전역 최적|global optim", "README_V2.md|APP_SUMMARY.md|pages/*.py|components/*.py|services/*.py", "UNSUPPORTED",
     "no model includes every real route, vehicle, budget and DC constraint; global_optimum_claimable is always False",
     "do not use"),
    ("M11", r"실시간|real-time", "README_V2.md|APP_SUMMARY.md|pages/*.py|components/*.py", "UNSUPPORTED",
     "no live data feed; the benchmark is an offline 31-day replay", "do not use"),
    ("M12", r"폐기량 감소|폐기 감소", "README_V2.md|APP_SUMMARY.md|pages/*.py|components/*.py", "UNSUPPORTED",
     "no realised disposal outcome exists", "do not use"),
    ("M13", r"비용 절감", "README_V2.md|APP_SUMMARY.md|pages/*.py|components/*.py", "UNSUPPORTED",
     "no realised cost; MILP comparisons are estimated tariff costs inside a pool", "'추정 운송비 비교'"),
    ("M14", r"수학적으로 검증", "README_V2.md|APP_SUMMARY.md|pages/*.py|components/*.py", "NEEDS_QUALIFIER",
     "true only as 'same 20 candidates, MILP + enumeration'", "'후보 20개 안에서 MILP·전수 열거로 확인'"),
    ("M15", r"서비스 100", "README_V2.md|APP_SUMMARY.md|pages/*.py|components/*.py", "NEEDS_QUALIFIER",
     "service_level = plan qty / MILP planned qty in the pool, not a demand fill rate", "'MILP 기준 계획량 대비 100%'"),
)
PRESENTATION_CLAIMS = (
    ("P01", "Varo Final은 MILP와 31일 서비스·비용이 동일(7,750 / 13,629,146원)", "NEEDS_QUALIFIER",
     "true inside the same 20-candidate pool, PROXY caps, max 5 routes, fixed 0/1 quantities (signature identical, "
     "enumeration-proven); routes differ on some days (alternative optima); with the generator's full pool the same "
     "model reaches the same 7,750 at lower estimated cost",
     "'같은 후보 20개·같은 추정 제약에서 MILP 기준과 같은 총 이동 계획량·추정 운송비'"),
    ("P02", "MILP 31/31일 optimal", "NEEDS_QUALIFIER",
     "HiGHS status 0 on both stages, MIP gap 0, enumeration agrees; restricted pool and PROXY caps",
     "'후보 20개 안에서 최적성 증명(31/31일)'"),
    ("P03", "Gap 0% / 최적성 Gap 0", "NEEDS_QUALIFIER",
     "decision gap 0 inside the restricted pool; solver MIP gap 0 is a different quantity; the in-app gap is not "
     "computable on Suhyup", "'동일 조건 결정 Gap 0(제한 후보 안)'"),
    ("P04", "MILP 결과 = 전체 물류망 최적해", "UNSUPPORTED",
     "20 of 36-40 generator candidates per day, one target per source-product, no route/DC capacity; larger pools lower "
     "the optimal cost by 15.6% (generator pool) and 56.5% (all targets)", "do not use"),
    ("P05", "MILP와 동일", "NEEDS_QUALIFIER", "say which pool, caps, limit and objective", "'동일 후보·동일 추정 제약 기준 MILP와 동일'"),
    ("P06", "7,750개 판매/처리 완료", "UNSUPPORTED",
     "7,750 is the 31-day planned movement quantity in the benchmark; unit unknown; not sales or executed moves",
     "'31일 계획 이동량 7,750(원천 단위 미확인)'"),
    ("P07", "운송비 13,629,146원(실제 비용)", "NEEDS_QUALIFIER",
     "official-tariff estimate (one-way, no tolls/loading), proxy vehicle class on some candidates; no invoice",
     "'공식 요율 기반 추정 운송비'"),
    ("P08", "exact optimization", "NEEDS_QUALIFIER", "exact for the restricted model within solver tolerance only",
     "'제한 후보 모델의 정확해'"),
)


def _claim_hits(repo_root: Path, pattern: str, globs: str) -> list[tuple[str, int, str]]:
    regex = re.compile(pattern, re.IGNORECASE)
    hits = []
    for glob in globs.split("|"):
        for path in sorted(repo_root.glob(glob)):
            if path.name == "milp_benchmark_integrity_validation.py" or not path.is_file():
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if regex.search(line):
                    hits.append((str(path.relative_to(repo_root)).replace("\\", "/"), number, line.strip()[:160]))
    return hits


def claims_audit(repo_root: Path) -> pd.DataFrame:
    """Repo wording found by the rules (a line a SAFE rule matches, e.g. an explicit non-claim, is not re-classified)."""
    safe_lines = {(location, number) for _, pattern, globs, verdict, _, _ in CLAIM_RULES if verdict == "SAFE"
                  for location, number, _ in _claim_hits(repo_root, pattern, globs)}
    rows = []
    for claim_id, pattern, globs, verdict, reason, suggestion in CLAIM_RULES:
        hits = [hit for hit in _claim_hits(repo_root, pattern, globs)
                if verdict == "SAFE" or (hit[0], hit[1]) not in safe_lines]
        for location, number, text in hits or [("(not found in repo docs/UI)", None, "")]:
            rows.append({"claim_id": claim_id, "pattern": pattern, "location": location, "line": number, "text": text,
                         "verdict": verdict, "reason": reason, "suggested_wording": suggestion, "source": "repo scan"})
    for claim_id, claim, verdict, reason, suggestion in PRESENTATION_CLAIMS:
        rows.append({"claim_id": claim_id, "pattern": None, "location": "presentation / summary wording", "line": None,
                     "text": claim, "verdict": verdict, "reason": reason, "suggested_wording": suggestion,
                     "source": "T3 benchmark evidence"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ run


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True).stdout.rstrip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _check(check_id: str, description: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"check_id": check_id, "description": description, "status": "PASS" if ok else "FAIL", "detail": detail}


PROTECTED_PATHS = ("services/shared_feasibility_selection.py", "services/shared_feasibility_validation.py",
                   "services/action_consistency.py", "services/action_consistency_validation.py",
                   "services/seller_loss_engine.py", "services/seller_loss_inputs.py", "services/seller_loss_promotion_gate.py",
                   "services/seller_shadow_ledger.py", "services/vhs_score_engine.py", "services/pareto_service.py",
                   "services/dqn_service.py", "services/demand_forecast_router.py", "services/demand_forecast_v2.py",
                   "services/candidate_generator.py", "services/recommendation_adapter.py", "services/analysis_pipeline.py",
                   "services/legacy_adapters", "pages", "components", "app_v2.py")
PRESENTATION_STATEMENTS = {
    "safe_short": "수협 31일 데이터에서 날마다 같은 후보 20개와 같은 추정 재고 제약, 하루 최대 5경로 조건을 적용했을 때, "
                  "Varo의 이동 계획은 수학적 최적화(MILP) 기준과 같은 총 이동 계획량 7,750과 같은 추정 운송비 13,629,146원을 "
                  "기록했습니다.",
    "scope_note": "이 일치는 후보 20개 안에서 확인한 결과이며, 전체 물류망의 최적해라는 뜻은 아닙니다. 7,750은 판매량이 아니라 "
                  "31일 동안의 계획 이동량이고, 운송비는 공식 요율로 계산한 추정값입니다.",
    "next_step": "후보 생성 단계에서 잘린 후보까지 넣으면 같은 이동량을 더 낮은 추정 운송비로 계획할 수 있어, 후보 생성이 다음 개선 "
                 "대상입니다.",
    "forbidden": ["전역 최적", "전체 물류망 최적해", "실제 비용 절감", "실제 폐기량 감소", "7,750개 판매", "실시간 최적화"],
}


def run_validation(data_root: Path, output_dir: Path | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    data_root = Path(data_root)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    output_dir = Path(output_dir or data_root / OUTPUT_FOLDER)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suhyup = run_suhyup(data_root)
        cutoff = run_cutoff(data_root)
        app = run_app_gap_e2e(data_root)
    by_day, gaps, regression, routes = suhyup["by_day"], suhyup["gaps"], suhyup["regression"], suhyup["routes"]
    cut_rows, generator = cutoff["rows"], cutoff["generator"]
    claims = claims_audit(repo_root)
    constraints = constraint_comparison(by_day, suhyup["transport"])
    transport = suhyup["transport"]
    totals = {pool: {"service": float(group["milp_service_qty"].sum()), "cost": float(group["milp_move_cost"].sum()),
                     "candidates": int(group["pool_size"].sum()), "optimal_days": int((group["solver_status"] == mbi.OPTIMAL).sum()),
                     "varo_key_equals_milp_days": int(group["varo_key_greedy_equals_milp"].sum()),
                     "varo_key_cost": float(group["varo_key_greedy_cost"].sum()),
                     "milp_ms_total": round(float(group["milp_ms"].sum()), 3), "milp_ms_max": round(float(group["milp_ms"].max()), 3),
                     "milp_peak_kb_max": float(group["milp_peak_kb"].max())}
              for pool, group in cut_rows.groupby("pool", sort=False)}
    base = totals["P20_SAVED"]["cost"]
    for pool in totals:
        totals[pool]["cost_delta_pct_vs_p20"] = round(100.0 * (totals[pool]["cost"] - base) / base, 6)
    plan_gap = lambda plan: gaps[gaps["plan"] == plan]
    varo, t1, partial = plan_gap("OFFLINE_VARO_FINAL"), plan_gap("SF_BENCHMARK_ALL_OR_NOTHING"), plan_gap("SF_BENCHMARK_PARTIAL")
    saved_vs_new = plan_gap("SAVED_2026_07_MILP")
    changed = (_git(repo_root, "status", "--porcelain") or "").splitlines()
    protected_changed = [line for line in changed if any(line[3:].startswith(path) for path in PROTECTED_PATHS)]
    expected_rows = 186 if "DQN" in set(regression["strategy"]) else 155
    contract = app["benchmark_integrity"]
    checks = [
        _check("S1", "31 days, 620 candidates; the rebuilt generator pool's top-20 equals the saved candidates (ids, qty, cost)",
               len(by_day) == 31 and int(generator["top20_reproduces_saved_candidates"].sum()) == 31,
               f"{int(generator['top20_reproduces_saved_candidates'].sum())}/31"),
        _check("S2", "candidate cutoff confirmed: the generator scored 36-40 candidates per day and kept 20",
               bool(by_day["candidate_cutoff_applied"].all()),
               f"{int(by_day['generator_pool_before_cut'].min())}-{int(by_day['generator_pool_before_cut'].max())}"),
        _check("S3", "MILP OPTIMAL on both stages 31/31 with a computed solver MIP gap",
               bool((by_day["solver_status"] == mbi.OPTIMAL).all()) and by_day["stage2_mip_gap"].notna().all()),
        _check("S4", "exhaustive enumeration reproduces the MILP optimum (service, cost) and its tie-break selection 31/31",
               bool(by_day["milp_equals_enumeration_optimum"].all() and by_day["milp_equals_enumeration_tie_break"].all())),
        _check("S5", "re-solving with mip_rel_gap=0 keeps the same selection 31/31",
               bool(by_day["strict_tolerance_same_selection"].all())),
        _check("S6", "MILP, offline Varo Final and T1 ALL_OR_NOTHING pass the solver-independent check 31/31",
               int(by_day[["milp_independent_violations", "varo_final_independent_violations",
                           "t1_independent_violations"]].to_numpy().sum()) == 0),
        _check("S7", "Varo Final vs MILP: identical signature and 0 decision gap on 31/31 (SAME_RESTRICTED_OBJECTIVE)",
               bool((varo["comparison_status"] == mbi.DIRECTLY_COMPARABLE).all()
                    and varo["zero_gap_flags"].str.startswith(mbi.SAME_RESTRICTED_OBJECTIVE).all())),
        _check("S8", "T1 ALL_OR_NOTHING vs MILP directly comparable 31/31; T1 PARTIAL never directly comparable (granularity)",
               bool((t1["comparison_status"] == mbi.DIRECTLY_COMPARABLE).all()
                    and (partial["comparison_status"] == mbi.NOT_COMPARABLE).all())),
        _check("S9", f"reference strategies reproduced with current code ({expected_rows} rows incl. status)",
               bool(regression["equal"].all()) and len(regression) == expected_rows,
               f"{int(regression['equal'].sum())}/{len(regression)}"),
        _check("S10", "saved 2026-07 MILP caps rebuilt on its 155 route rows; it differs from the revalidated MILP only on 2026-07-16",
               bool(all(suhyup["old_cap_checks"]) and len(suhyup["old_cap_checks"]) == 155
                    and set(saved_vs_new.loc[(saved_vs_new["plan_cost"] - saved_vs_new["reference_cost"]).abs() > 1e-6, "date"]) == {"2026-07-16"}),
               f"{sum(suhyup['old_cap_checks'])}/{len(suhyup['old_cap_checks'])}"),
        _check("S11", "every alternative-optimum day is recorded (route identity is not unique)",
               bool((by_day["alternative_optima"] >= 1).all()),
               f"{int(by_day['alternative_optima'].min())}-{int(by_day['alternative_optima'].max())}"),
        _check("X1", "cutoff experiment: MILP OPTIMAL and independently feasible on every pool/day",
               bool((cut_rows["solver_status"] == mbi.OPTIMAL).all() and (cut_rows["milp_independent_violations"] == 0).all())),
        _check("X2", "cutoff experiment: enumeration agrees wherever it ran",
               bool(cut_rows.loc[cut_rows["enumeration_status"] == "ENUMERATED", "enumeration_matches_milp"].all())),
        _check("X3", "cutoff experiment: service unchanged (5 x 50 ceiling) and cost never above P20",
               all(abs(item["service"] - totals["P20_SAVED"]["service"]) <= 1e-9 for item in totals.values())
               and all(item["cost"] <= base + 1e-6 for item in totals.values())),
        _check("A1", "in-app gap on the Suhyup 07-31 upload: GAP_NOT_COMPUTABLE, no optimality claim",
               contract.get("decision_gap_status") == mbi.GAP_NOT_COMPUTABLE
               and contract.get("optimality_claim") == "NO_OPTIMALITY_CLAIM", contract.get("reason_codes")),
        _check("A2", "no contract claims a global optimum", contract.get("global_optimum_claimable") is False),
        _check("G1", "no protected path changed (UI, T1 selector/validation, T2, Seller Loss, VHS, Pareto, DQN, forecast, generator, pipeline)",
               not protected_changed, protected_changed),
    ]
    summary_rows = [
        ("suhyup", "days", len(by_day)), ("suhyup", "candidates", int(by_day["milp_input_pool"].sum())),
        ("suhyup", "generator_pool_before_cut_total", int(by_day["generator_pool_before_cut"].sum())),
        ("suhyup", "lane_pool_all_targets_total", int(by_day["lane_pool_all_targets"].sum())),
        ("suhyup", "milp_optimal_days", int((by_day["solver_status"] == mbi.OPTIMAL).sum())),
        ("suhyup", "milp_feasible_not_proven_days", int((by_day["solver_status"] == mbi.FEASIBLE_NOT_PROVEN_OPTIMAL).sum())),
        ("suhyup", "milp_time_limit_days", int((by_day["termination_reason"] == "TIME_LIMIT").sum())),
        ("suhyup", "milp_service_total", float(by_day["milp_service_qty"].sum())),
        ("suhyup", "milp_cost_total", float(by_day["milp_move_cost"].sum())),
        ("suhyup", "varo_final_directly_comparable_days", int((varo["comparison_status"] == mbi.DIRECTLY_COMPARABLE).sum())),
        ("suhyup", "varo_final_service_total", float(varo["plan_service"].sum())),
        ("suhyup", "varo_final_cost_total", float(varo["plan_cost"].sum())),
        ("suhyup", "varo_final_same_routes_as_milp_days", int((varo["route_relation"] == "SAME_ROUTES").sum())),
        ("suhyup", "t1_all_or_nothing_service_total", float(t1["plan_service"].sum())),
        ("suhyup", "t1_all_or_nothing_cost_total", float(t1["plan_cost"].sum())),
        ("suhyup", "alternative_optima_min_max", f"{int(by_day['alternative_optima'].min())}-{int(by_day['alternative_optima'].max())}"),
        ("suhyup", "saved_2026_07_milp_cost_total", float(by_day["saved_2026_07_milp_cost"].sum())),
        ("suhyup", "saved_vs_new_milp_not_comparable_days", int((saved_vs_new["comparison_status"] == mbi.NOT_COMPARABLE).sum())),
        ("suhyup", "candidates_with_proxy_vehicle", sum(1 for item in transport.values() if item["proxy_vehicle_count"] > 0)),
        ("suhyup", "candidates_cost_reproduced_by_tariff_engine", sum(1 for item in transport.values() if item["reproduced"])),
        *[("cutoff", f"{pool}_{metric}", value) for pool, item in totals.items() for metric, value in item.items()],
        ("app_gap_20260731", "decision_gap_status", contract.get("decision_gap_status")),
        ("app_gap_20260731", "benchmark_scope", contract.get("benchmark_scope")),
        ("runtime", "suhyup_wall_seconds", suhyup["runtime"]["wall_seconds"]),
        ("runtime", "cutoff_wall_seconds", cutoff["runtime"]["wall_seconds"]),
        ("runtime", "total_wall_seconds", round(time.perf_counter() - started, 3)),
        ("runtime", "process_peak_working_set_mb", process_peak_working_set_mb()),
        ("runtime", "environment", f"{platform.platform()} / Python {platform.python_version()} / scipy {scipy.__version__} "
                                   f"(HiGHS) / {platform.processor()}"),
    ]
    validation = {
        "integrity_version": mbi.BENCHMARK_INTEGRITY_VERSION, "git_head": _git(repo_root, "rev-parse", "HEAD"),
        "data_root": str(data_root),
        "inputs": {"candidates": sfv.CANDIDATES, "inventory": sfv.INVENTORY, "reference": sfv.REFERENCE,
                   "saved_milp_daily": SAVED_MILP_DAILY, "saved_milp_routes": SAVED_MILP_ROUTES, "processed_upload": PROCESSED},
        "checks": checks, "failed_checks": [item["check_id"] for item in checks if item["status"] == "FAIL"],
        "milp_role": {
            "A_full_operational_optimizer": False,
            "B_exact_within_restricted_pool": True,
            "C_proxy_benchmark": True,
            "D_limited_search_for_time": "offline: no time limit, no limited mode; in-app: limited BnB when > 24 candidates "
                                         "or > 750k combinations, time_limit 3 s default",
        },
        "objective": mbi.LEXICOGRAPHIC_OBJECTIVE, "app_objective": mbi.SAVING_OBJECTIVE,
        "solver_tolerance": mbi.HIGHS_DEFAULT_TOLERANCE,
        "target_need_definitions": TARGET_NEED_DEFINITIONS, "source_surplus_definitions": SOURCE_SURPLUS_DEFINITIONS,
        "cutoff_totals": totals, "app_gap_e2e_20260731": app,
        "suhyup_runtime": suhyup["runtime"], "cutoff_runtime": cutoff["runtime"],
        "presentation_statements": PRESENTATION_STATEMENTS,
        "kpi_7750": "31-day sum of planned movement quantity (recommended_qty of selected candidates) in the benchmark; "
                    "source unit unknown; not sales, not executed moves, not realised disposal reduction",
        "limitations": [
            "the benchmark pool is the generator's top-20 by candidate_score; larger pools reach the same service at lower cost",
            "caps are PROXY (median / outbound rules); STRICT_ACTUAL cannot form a plan on any day",
            "no unit-compatible route capacity, DC capacity, budget or time window is in any model",
            "OPTIMAL is HiGHS default tolerance; the Suhyup results are also enumeration-proven inside the pool",
            "route identity between strategies is not unique: several selections reach the same service and cost",
        ],
        "production_action_applied": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows, columns=["scope", "metric", "value"]).to_csv(output_dir / OUTPUT_FILES[0], index=False, encoding="utf-8-sig")
    by_day.to_csv(output_dir / OUTPUT_FILES[1], index=False, encoding="utf-8-sig")
    cut_rows.merge(generator, on="date", how="left").to_csv(output_dir / OUTPUT_FILES[2], index=False, encoding="utf-8-sig")
    constraints.to_csv(output_dir / OUTPUT_FILES[3], index=False, encoding="utf-8-sig")
    routes.to_csv(output_dir / OUTPUT_FILES[4], index=False, encoding="utf-8-sig")
    gaps.to_csv(output_dir / OUTPUT_FILES[5], index=False, encoding="utf-8-sig")
    claims.to_csv(output_dir / OUTPUT_FILES[6], index=False, encoding="utf-8-sig")
    (output_dir / OUTPUT_FILES[7]).write_text(json.dumps(validation, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_validation(args.data_root, args.output_dir)
    print(json.dumps({"failed_checks": result["failed_checks"], "checks": result["checks"],
                      "cutoff_totals": result["cutoff_totals"]}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

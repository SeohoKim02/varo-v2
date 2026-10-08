"""Shared-feasibility Top-N validation on the real Suhyup 31-day benchmark and the repository workbooks.

Validation only. It reads the saved Suhyup candidates, actual inventory and the 186-row reference bundle, and never
writes there. Outputs go to <data-root>/_SHARED_FEASIBILITY_VALIDATION (local only, never into git); a re-run replaces
only the files listed in OUTPUT_FILES.

Plans compared per day (same 20 saved candidates, Varo Final rank recomputed exactly as the offline revalidation does):
  LEGACY_TOP5            first five rows of varo_final_rank (what the production Top-5 slice does)
  OFFLINE_VARO_FINAL     suhyup_algorithm_revalidation.ordered_feasible_selection (186-row reference strategy)
  MILP                   suhyup_algorithm_revalidation.lexicographic_milp (unchanged objective)
  SF_BENCHMARK_ALL_OR_NOTHING   shared-feasibility selector, BENCHMARK_PROXY caps, no partial allocation
  SF_BENCHMARK_PARTIAL          same, partial allocation when safe (cost recomputed by the official-tariff engine)
  SF_STRICT_ACTUAL              shared-feasibility selector, validated caps only
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
import tracemalloc
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from services import shared_feasibility_selection as sf
from services import suhyup_algorithm_revalidation as rv
from services.pareto_service import _default_feasible as pareto_default_feasible
from services.real_data_adapters import DATA_ROOT
from services.seller_decision_validation import _env

OUTPUT_FOLDER = "_SHARED_FEASIBILITY_VALIDATION"
OUTPUT_FILES = (
    "shared_feasibility_summary.csv",
    "shared_feasibility_by_day.csv",
    "shared_feasibility_selected_routes.csv",
    "shared_feasibility_rejections.csv",
    "shared_feasibility_cap_provenance.csv",
    "shared_feasibility_validation.json",
    "shared_feasibility_problem_days.csv",
    "shared_feasibility_workbooks.csv",
    "shared_feasibility_workbook_diagnostics.csv",
)
CANDIDATES = "16_VARO_E2E_20260731/multi_snapshot_validation/suhyup_202607_multi_snapshot_recommendations.csv"
INVENTORY = "02_Korea_Suhyup_Logistics/processed/suhyup_logistics_inventory_flow_actual.csv"
REFERENCE = "21_SUHYUP_ALGORITHM_REVALIDATION_20260919/daily_strategy_summary.csv"
NUMERIC = (
    "recommended_qty", "move_cost", "estimated_cost", "expected_saving", "distance_km", "expected_time_min",
    "travel_time_min", "feasibility_score", "demand_fit_score", "disposal_risk_score", "inventory_balance_score",
    "promotion_score", "route_cost_score",
)
# Offline benchmark caps (suhyup_algorithm_revalidation.enrich_inventory_constraints) with their provenance.
SUHYUP_CAP_SPEC = {
    sf.SOURCE_STOCK: (("source_stock",), "DIRECT_REAL",
                      "actual daily center stock_qty (inventory_provenance=actual), summed over state codes"),
    sf.SOURCE_SURPLUS: (("source_surplus",), "PROXY",
                        "max(0, source stock - same-day cross-node product median); rule, not observed transferable stock"),
    sf.TARGET_NEED: (("target_need_7d",), "PROXY",
                     "max(0, median stock + 7 x actual outbound - target stock); outbound is a demand proxy, median a rule"),
    sf.ROUTE_CAPACITY: (sf.ROUTE_CAPACITY_FIELDS, "USER_INPUT",
                        "no unit-compatible route capacity in the Suhyup data; the tariff engine sizes vehicles per quantity"),
}
GENERATOR_MOVE_CAP = 50  # services.candidate_generator._MOVE_CAP (CONFIG)
PLAN_SF = ("SF_BENCHMARK_ALL_OR_NOTHING", "SF_BENCHMARK_PARTIAL", "SF_STRICT_ACTUAL")


def load_suhyup_candidates(data_root: Path) -> pd.DataFrame:
    candidates = pd.read_csv(data_root / CANDIDATES, dtype={
        "snapshot_date": str, "route_id": str, "product_id": str, "source_id": str, "target_id": str})
    inventory = pd.read_csv(data_root / INVENTORY, dtype={"date": str, "center_code": str, "product_code": str})
    return rv.enrich_inventory_constraints(rv._numeric(candidates, NUMERIC), inventory)


def ranked_day(candidates: pd.DataFrame, date: str) -> pd.DataFrame:
    """Same per-day recomputation as suhyup_algorithm_revalidation.run_validation."""
    from services.legacy_adapters._local_modules.heuristic_optimizer import add_heuristic_scores
    from services.vhs_score_engine import apply_auto_vhs

    day = candidates[candidates["snapshot_date"] == date].copy()
    day["final_recommendation"] = "재고 이동"
    return apply_auto_vhs(add_heuristic_scores(day)).frame


class TariffRecompute:
    """Partial-quantity cost from the official-tariff engine, only for rows whose saved cost it reproduces exactly."""

    def __init__(self, candidates: pd.DataFrame) -> None:
        from services.real_transport_enrichment import enrich_real_transport

        self._enrich = enrich_real_transport
        frame = candidates[["route_id", "product_id", "source_id", "target_id", "recommended_qty"]].copy()
        priced = enrich_real_transport(frame.reset_index(drop=True))
        saved = candidates["move_cost"].reset_index(drop=True)
        applied = priced["real_transport_applied"].fillna(False).astype(bool)
        same = applied & (pd.to_numeric(priced["move_cost"], errors="coerce") - saved).abs().le(1e-6)
        keys = list(zip(candidates["snapshot_date"], candidates["route_id"]))
        self.reproduced = {key for key, ok in zip(keys, same.tolist()) if ok}
        self.checked = len(keys)
        self.calls = 0

    def __call__(self, record: Mapping[str, Any], quantity: float) -> float | None:
        if (str(record.get("snapshot_date")), str(record.get("route_id"))) not in self.reproduced:
            return None
        self.calls += 1
        frame = pd.DataFrame([{**{name: record.get(name) for name in ("route_id", "product_id", "source_id", "target_id")},
                               "recommended_qty": quantity}])
        row = self._enrich(frame).iloc[0]
        return float(row["move_cost"]) if bool(row.get("real_transport_applied")) else None


def _cost(rows: pd.DataFrame | Sequence[Mapping[str, Any]], field: str = "move_cost") -> float:
    frame = pd.DataFrame(rows)
    return float(pd.to_numeric(frame.get(field, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())


def _qty(rows: pd.DataFrame | Sequence[Mapping[str, Any]], field: str = "recommended_qty") -> float:
    frame = pd.DataFrame(rows)
    return float(pd.to_numeric(frame.get(field, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())


def _ids(frame: pd.DataFrame) -> str:
    return "|".join(frame.get("route_id", pd.Series(dtype=str)).astype(str).tolist())


def _cap_types(caps: sf.CapTable) -> str:
    seen: dict[str, set[str]] = {}
    for (kind, _), cap in caps.items():
        seen.setdefault(kind, set()).add(cap.provenance if cap.value is not None else "MISSING")
    return "|".join(f"{kind}={'/'.join(sorted(values))}" for kind, values in sorted(seen.items()))


def _reference_rows(day: pd.DataFrame, date: str) -> dict[str, dict[str, Any]]:
    """Recompute the five non-DQN reference strategies exactly as run_validation does (for the 186-row check)."""
    milp = rv.lexicographic_milp(day)
    milp_service = float(pd.to_numeric(milp["selected"]["recommended_qty"], errors="coerce").sum())
    selections = {
        "VHS": rv.ordered_feasible_selection(day, ("vhs_rank", "route_id"), (True, True)),
        "Greedy": rv.ordered_feasible_selection(day, ("greedy_rank", "route_id"), (True, True)),
        "Varo Final": rv.ordered_feasible_selection(day, ("varo_final_rank", "route_id"), (True, True)),
        "MILP": milp["selected"],
    }
    selections["Pareto"], _ = rv.pareto_operational_selection(day)
    rows = {}
    for strategy, selected in selections.items():
        status = "optimal" if strategy == "MILP" and milp["optimal"] else "feasible"
        rows[strategy] = rv._strategy_row(date, strategy, day, selected, milp_service, status)
        rows[strategy]["_selected"] = selected
    rows["MILP"]["_optimal"] = bool(milp["optimal"])
    return rows


def _existing_validators(records: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> dict[str, Any]:
    """The plan re-checked by the two pre-existing shared validators, with recommended_qty := allocated_qty."""
    allocated = {row["candidate_id"]: row["allocated_qty"] for row in plan["rows"]
                 if row["selection_status"] in (sf.SELECTED, sf.PARTIALLY_SELECTED)}
    chosen = [{**record, "recommended_qty": allocated[str(record["route_id"])]}
              for record in records if str(record["route_id"]) in allocated]
    offline = rv._selection_violation(chosen)
    pareto_ok, pareto_reason = pareto_default_feasible(chosen)
    return {"offline_validator": offline or "OK", "pareto_validator": "OK" if pareto_ok else pareto_reason,
            "ok": offline is None and bool(pareto_ok)}


def _plan_signature(plan: Mapping[str, Any]) -> list[tuple]:
    return [(row["candidate_id"], row["selection_status"], row["allocated_qty"], row["allocated_move_cost"])
            for row in plan["rows"]]


def _problem_rows(date: str, day: pd.DataFrame, legacy: pd.DataFrame, caps: sf.CapTable,
                  plans: Mapping[str, Mapping[str, Any]], legacy_check: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    benchmark = plans["SF_BENCHMARK_PARTIAL"]
    status = {row["candidate_id"]: row for row in benchmark["rows"]}
    for violation in legacy_check["violations"]:
        target, product = violation["group"].split("/") if violation["constraint"] == sf.TARGET_NEED else ("", "")
        group = legacy[(legacy["target_id"].astype(str) == target) & (legacy["product_id"].astype(str) == product)]
        members = []
        for record in group.to_dict("records"):
            median = float(record["median_stock"])
            target_stock, outbound = float(record["target_stock"]), float(record["target_daily_demand_proxy"])
            # services.candidate_generator.generate_candidates, lines for need / source_surplus / target_need / moved
            source_stock = float(record["source_stock"])
            generator_need = max(0.0, median - target_stock) + 7.0 * outbound
            surplus = source_stock - median
            generator_surplus = max(1.0, surplus if surplus > 0 else source_stock * 0.3)
            generator_target_need = max(1.0, generator_need if generator_need > 0 else generator_surplus)
            generator_qty = int(max(1, min(generator_surplus, generator_target_need, source_stock, GENERATOR_MOVE_CAP)))
            members.append({
                "route_id": record["route_id"], "varo_final_rank": int(record["varo_final_rank"]),
                "source_id": record["source_id"], "recommended_qty": float(record["recommended_qty"]),
                "generator_need_reconstructed": round(generator_need, 6),
                "generator_qty_reconstructed": generator_qty,
                "reconstruction_matches": generator_qty == int(record["recommended_qty"]),
                "new_status": status[record["route_id"]]["selection_status"],
                "new_allocated_qty": status[record["route_id"]]["allocated_qty"],
            })
        cap = caps[(sf.TARGET_NEED, (target, product))] if target else None
        individual = [m for m in members if cap is not None and m["recommended_qty"] > float(cap.value) + 1e-9]
        first = group.iloc[0] if not group.empty else None
        cause = ("INDIVIDUAL_OVER_CAP: generator need max(0, median - stock) + 7 x outbound differs from benchmark "
                 "need max(0, median + 7 x outbound - stock) when target stock > median"
                 if individual else
                 "SHARED_OVERFLOW: generator keeps one target per (source, product), so two sources fill the same "
                 "target need; each move fits alone, together they exceed it")
        legacy_ids = legacy["route_id"].astype(str).tolist()
        rows.append({
            "date": date, "constraint": violation["constraint"], "group_target_product": violation["group"],
            "cap_value": violation["cap"], "cap_provenance": violation["provenance"],
            "cap_definition": SUHYUP_CAP_SPEC[sf.TARGET_NEED][2],
            "target_stock": None if first is None else float(first["target_stock"]),
            "median_stock": None if first is None else float(first["median_stock"]),
            "target_outbound_proxy": None if first is None else float(first["target_daily_demand_proxy"]),
            "legacy_used": violation["used"], "legacy_excess": violation["excess"],
            "same_target_product_candidates_in_day": int(((day["target_id"].astype(str) == target)
                                                          & (day["product_id"].astype(str) == product)).sum()),
            "legacy_members": json.dumps(members, ensure_ascii=False),
            "cause": cause,
            "legacy_top5_order": "|".join(legacy_ids),
            "new_selection_order": "|".join(benchmark["selected_ids"]),
            "dropped_from_legacy": "|".join(item for item in legacy_ids if item not in benchmark["selected_ids"]),
            "added_by_shared_feasibility": "|".join(item for item in benchmark["selected_ids"] if item not in legacy_ids),
            "fix_applied": "shared target-need ledger: the lower-ranked move gets only the remaining need "
                           "(or nothing), and the next executable candidate takes the slot",
        })
    return rows


def run_suhyup(data_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    candidates = load_suhyup_candidates(data_root)
    reference = pd.read_csv(data_root / REFERENCE, dtype={"date": str})
    with _env("VARO_REAL_DATA_ROOT", str(data_root)):
        recompute = TariffRecompute(candidates)
        by_day, selected_rows, rejection_rows, cap_rows, problem_rows, regression = [], [], [], [], [], []
        selection_ms: list[float] = []
        selection_peaks: list[int] = []
        tracemalloc.start()
        for date in sorted(candidates["snapshot_date"].astype(str).unique()):
            day = ranked_day(candidates, date)
            records = rv._records(day)
            caps = sf.caps_from_columns(records, SUHYUP_CAP_SPEC)
            legacy = day.sort_values(["varo_final_rank", "route_id"], kind="mergesort").head(rv.MAX_DAILY_ROUTES)
            legacy_ids = legacy["route_id"].astype(str).tolist()
            tracemalloc.reset_peak()
            tick = time.perf_counter()
            plans = {
                "SF_BENCHMARK_ALL_OR_NOTHING": sf.select_shared_feasible(
                    records, caps, mode=sf.BENCHMARK_PROXY, max_routes=rv.MAX_DAILY_ROUTES,
                    partial_policy=sf.PARTIAL_NONE, legacy_ids=legacy_ids),
                "SF_BENCHMARK_PARTIAL": sf.select_shared_feasible(
                    records, caps, mode=sf.BENCHMARK_PROXY, max_routes=rv.MAX_DAILY_ROUTES,
                    partial_policy=sf.PARTIAL_IF_SAFE, cost_recompute=recompute, legacy_ids=legacy_ids),
                "SF_STRICT_ACTUAL": sf.select_shared_feasible(
                    records, caps, mode=sf.STRICT_ACTUAL, max_routes=rv.MAX_DAILY_ROUTES,
                    partial_policy=sf.PARTIAL_IF_SAFE, cost_recompute=recompute, legacy_ids=legacy_ids),
            }
            selection_ms.append((time.perf_counter() - tick) * 1000.0)
            selection_peaks.append(tracemalloc.get_traced_memory()[1])
            shuffled = list(records)
            random.Random(date).shuffle(shuffled)
            order_invariant = all(
                _plan_signature(sf.select_shared_feasible(
                    shuffled, sf.caps_from_columns(shuffled, SUHYUP_CAP_SPEC), mode=plan["mode"],
                    max_routes=rv.MAX_DAILY_ROUTES, partial_policy=plan["partial_policy"],
                    cost_recompute=recompute, legacy_ids=legacy_ids)) == _plan_signature(plan)
                for plan in plans.values())
            cross = {name: _existing_validators(records, plan) for name, plan in plans.items()}
            legacy_bench = sf.validate_plan(rv._records(legacy), caps, mode=sf.BENCHMARK_PROXY, max_routes=rv.MAX_DAILY_ROUTES)
            legacy_strict = sf.validate_plan(rv._records(legacy), caps, mode=sf.STRICT_ACTUAL, max_routes=rv.MAX_DAILY_ROUTES)
            refs = _reference_rows(day, date)
            for strategy, row in refs.items():
                saved = reference[(reference["date"] == date) & (reference["strategy"] == strategy)].iloc[0]
                regression.append({
                    "date": date, "strategy": strategy,
                    "service_equal": abs(float(row["service_qty"]) - float(saved["service_qty"])) <= 1e-6,
                    "cost_equal": abs(float(row["total_cost"]) - float(saved["total_cost"])) <= 1e-6,
                    "route_ids_equal": str(row["route_ids"]) == str(saved["route_ids"]),
                    "violations_equal": int(row["feasibility_violations"]) == int(saved["feasibility_violations"]),
                })
            offline, milp = refs["Varo Final"]["_selected"], refs["MILP"]["_selected"]
            reference_checks = {strategy: sf.validate_plan(rv._records(refs[strategy]["_selected"]), caps,
                                                           mode=sf.BENCHMARK_PROXY, max_routes=rv.MAX_DAILY_ROUTES)
                                for strategy in ("Varo Final", "MILP", "Pareto", "VHS", "Greedy")}
            bench, partial, strict = (plans[name] for name in PLAN_SF)
            problem = legacy_bench["violation_count"] > 0
            by_day.append({
                "date": date, "candidate_count": len(day), "cap_types": _cap_types(caps),
                "legacy_route_ids": "|".join(legacy_ids), "legacy_service_qty": _qty(legacy), "legacy_cost": _cost(legacy),
                "legacy_source_excess_qty": legacy_bench["source_excess_qty"],
                "legacy_target_excess_qty": legacy_bench["target_excess_qty"],
                "legacy_duplicate_count": legacy_bench["duplicate_candidate_count"] + legacy_bench["duplicate_lane_count"],
                "legacy_violation_count": legacy_bench["violation_count"],
                "legacy_strict_certified": legacy_strict["certified"],
                "legacy_strict_unverifiable": "|".join(legacy_strict["unverifiable_constraints"]),
                "sf_benchmark_route_ids": "|".join(bench["selected_ids"]),
                "sf_benchmark_service_qty": bench["total_allocated_qty"], "sf_benchmark_cost": bench["total_move_cost"],
                "sf_benchmark_violation_count": bench["validation"]["violation_count"],
                "sf_benchmark_source_excess_qty": bench["validation"]["source_excess_qty"],
                "sf_benchmark_target_excess_qty": bench["validation"]["target_excess_qty"],
                "sf_benchmark_duplicate_count": bench["validation"]["duplicate_candidate_count"] + bench["validation"]["duplicate_lane_count"],
                "sf_partial_route_ids": "|".join(partial["selected_ids"]),
                "sf_partial_service_qty": partial["total_allocated_qty"], "sf_partial_cost": partial["total_move_cost"],
                "sf_partial_count": partial["partial_count"],
                "sf_partial_violation_count": partial["validation"]["violation_count"],
                "sf_partial_source_excess_qty": partial["validation"]["source_excess_qty"],
                "sf_partial_target_excess_qty": partial["validation"]["target_excess_qty"],
                "sf_partial_duplicate_count": partial["validation"]["duplicate_candidate_count"] + partial["validation"]["duplicate_lane_count"],
                "sf_strict_plan_status": strict["plan_status"], "sf_strict_selected_count": strict["selected_count"],
                "sf_strict_service_qty": strict["total_allocated_qty"], "sf_strict_cost": strict["total_move_cost"],
                "sf_strict_violation_count": strict["validation"]["violation_count"],
                "sf_strict_source_excess_qty": strict["validation"]["source_excess_qty"],
                "sf_strict_target_excess_qty": strict["validation"]["target_excess_qty"],
                "sf_strict_duplicate_count": strict["validation"]["duplicate_candidate_count"] + strict["validation"]["duplicate_lane_count"],
                "sf_strict_insufficient_count": strict["status_counts"].get(sf.INSUFFICIENT_CAP_DATA, 0),
                "offline_varo_route_ids": _ids(offline),
                "sf_benchmark_equals_offline_varo": "|".join(bench["selected_ids"]) == _ids(offline),
                "milp_route_ids": _ids(milp), "milp_service_qty": _qty(milp), "milp_cost": _cost(milp),
                "milp_optimal": refs["MILP"]["_optimal"],
                "sf_benchmark_equals_milp_service": abs(bench["total_allocated_qty"] - _qty(milp)) <= 1e-9,
                "sf_benchmark_equals_milp_cost": abs(float(bench["total_move_cost"] or 0) - _cost(milp)) <= 1e-6,
                "service_delta_vs_legacy": bench["total_allocated_qty"] - _qty(legacy),
                "cost_delta_vs_legacy": float(bench["total_move_cost"] or 0) - _cost(legacy),
                "legacy_problem_day": problem,
                "selection_ms_three_plans": round(selection_ms[-1], 3),
                "order_invariant_under_shuffle": order_invariant,
                "sf_benchmark_existing_validators": json.dumps(cross["SF_BENCHMARK_ALL_OR_NOTHING"]),
                "sf_partial_existing_validators": json.dumps(cross["SF_BENCHMARK_PARTIAL"]),
                "existing_validators_ok": all(item["ok"] for name, item in cross.items() if name != "SF_STRICT_ACTUAL"),
                "reference_strategies_violations_by_sf_validator": json.dumps(
                    {name: check["violation_count"] for name, check in reference_checks.items()}),
            })
            for name, plan in plans.items():
                for row in plan["rows"]:
                    target = selected_rows if row["selection_status"] in (sf.SELECTED, sf.PARTIALLY_SELECTED) else rejection_rows
                    target.append({"date": date, "plan": name, **row})
            for record in rv._records(legacy):
                selected_rows.append({
                    "date": date, "plan": "LEGACY_TOP5", "candidate_id": record["route_id"], "route_id": record["route_id"],
                    "product_id": record["product_id"], "source_id": record["source_id"], "target_id": record["target_id"],
                    "original_rank": int(record["varo_final_rank"]), "original_recommended_qty": record["recommended_qty"],
                    "allocated_qty": record["recommended_qty"], "original_move_cost": record["move_cost"],
                    "allocated_move_cost": record["move_cost"], "selection_status": "LEGACY_TOP5_SLICE",
                    "source_surplus_cap": record["source_surplus"], "source_stock_cap": record["source_stock"],
                    "target_need_cap": record["target_need_7d"], "feasibility_mode": "NONE (rank slice)",
                })
            cap_rows.extend({"date": date, **cap.as_row()} for _, cap in sorted(caps.items()))
            if problem:
                problem_rows.extend(_problem_rows(date, day, legacy, caps, plans, legacy_bench))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {
        "by_day": pd.DataFrame(by_day), "selected": pd.DataFrame(selected_rows), "rejections": pd.DataFrame(rejection_rows),
        "caps": pd.DataFrame(cap_rows), "problems": pd.DataFrame(problem_rows), "regression": pd.DataFrame(regression),
        "reference": reference,
        "runtime": {
            "days": len(by_day), "candidates_total": int(len(candidates)),
            "candidates_per_day": sorted(set(int(row["candidate_count"]) for row in by_day)),
            "selection_ms_per_day_three_plans": {"median": round(float(pd.Series(selection_ms).median()), 3),
                                                 "max": round(max(selection_ms), 3), "total": round(sum(selection_ms), 3)},
            "tracemalloc_peak_mb_including_vhs_recompute": round(peak / 1024 / 1024, 3),
            "tracemalloc_peak_mb_selection_only_max_day": round(max(selection_peaks) / 1024 / 1024, 3),
            "wall_seconds_total": round(time.perf_counter() - started, 3),
            "tariff_recompute": {"rows_checked": recompute.checked, "rows_reproduced": len(recompute.reproduced),
                                 "partial_cost_calls": recompute.calls},
        },
    }


def _summary(suhyup: Mapping[str, Any]) -> pd.DataFrame:
    by_day, selected = suhyup["by_day"], suhyup["selected"]
    rows = []
    legacy = selected[selected["plan"] == "LEGACY_TOP5"]
    rows.append({
        "plan": "LEGACY_TOP5", "mode": "NONE (rank slice)", "days": len(by_day),
        "selected_routes": int(len(legacy)), "total_service_qty": float(by_day["legacy_service_qty"].sum()),
        "total_cost": float(by_day["legacy_cost"].sum()), "violation_days": int((by_day["legacy_violation_count"] > 0).sum()),
        "violations": int(by_day["legacy_violation_count"].sum()),
        "source_excess_qty": float(by_day["legacy_source_excess_qty"].sum()),
        "target_excess_qty": float(by_day["legacy_target_excess_qty"].sum()),
        "duplicates": int(by_day["legacy_duplicate_count"].sum()), "partial_allocations": 0,
        "claim": "none; Top-5 slice is not checked against shared caps",
    })
    for name, prefix in (("SF_BENCHMARK_ALL_OR_NOTHING", "sf_benchmark"), ("SF_BENCHMARK_PARTIAL", "sf_partial")):
        plan_rows = selected[selected["plan"] == name]
        rows.append({
            "plan": name, "mode": sf.BENCHMARK_PROXY, "days": len(by_day), "selected_routes": int(len(plan_rows)),
            "total_service_qty": float(by_day[f"{prefix}_service_qty"].sum()),
            "total_cost": float(pd.to_numeric(by_day[f"{prefix}_cost"]).sum()),
            "violation_days": int((by_day[f"{prefix}_violation_count"] > 0).sum()),
            "violations": int(by_day[f"{prefix}_violation_count"].sum()),
            "source_excess_qty": float(by_day[f"{prefix}_source_excess_qty"].sum()),
            "target_excess_qty": float(by_day[f"{prefix}_target_excess_qty"].sum()),
            "duplicates": int(by_day[f"{prefix}_duplicate_count"].sum()),
            "partial_allocations": int((plan_rows["selection_status"] == sf.PARTIALLY_SELECTED).sum()),
            "claim": "PROXY caps satisfied (source_surplus / target_need_7d are PROXY); not an operational guarantee",
        })
    strict = selected[selected["plan"] == "SF_STRICT_ACTUAL"]
    rows.append({
        "plan": "SF_STRICT_ACTUAL", "mode": sf.STRICT_ACTUAL, "days": len(by_day), "selected_routes": int(len(strict)),
        "total_service_qty": float(by_day["sf_strict_service_qty"].sum()),
        "total_cost": float(pd.to_numeric(by_day["sf_strict_cost"]).sum()),
        "violation_days": int((by_day["sf_strict_violation_count"] > 0).sum()),
        "violations": int(by_day["sf_strict_violation_count"].sum()),
        "source_excess_qty": float(by_day["sf_strict_source_excess_qty"].sum()),
        "target_excess_qty": float(by_day["sf_strict_target_excess_qty"].sum()),
        "duplicates": int(by_day["sf_strict_duplicate_count"].sum()),
        "partial_allocations": int((strict["selection_status"] == sf.PARTIALLY_SELECTED).sum()) if not strict.empty else 0,
        "claim": "INSUFFICIENT_CAP_DATA on " + str(int((by_day["sf_strict_plan_status"] == sf.INSUFFICIENT_CAP_DATA).sum()))
                 + " days: target_need and source_surplus are PROXY",
    })
    reference = suhyup["reference"]
    for strategy, label in (("Varo Final", "REFERENCE_OFFLINE_VARO_FINAL"), ("MILP", "REFERENCE_MILP")):
        group = reference[reference["strategy"] == strategy]
        rows.append({
            "plan": label, "mode": "saved 21_SUHYUP_ALGORITHM_REVALIDATION_20260919", "days": int(group["date"].nunique()),
            "selected_routes": int(group["selected_count"].sum()), "total_service_qty": float(group["service_qty"].sum()),
            "total_cost": float(group["total_cost"].sum()), "violation_days": int((group["feasibility_violations"] > 0).sum()),
            "violations": int(group["feasibility_violations"].sum()), "claim": "read-only reference",
        })
    return pd.DataFrame(rows)


DIAGNOSTIC_COLUMNS = (
    "candidate_id", "original_rank", "in_legacy_top_n", "product_id", "source_id", "target_id", "route_type", "dc_id",
    "original_recommended_qty", "individual_cap_room", "source_stock_cap", "source_surplus_cap", "target_need_cap",
    "dc_capacity_cap", "route_capacity_cap", "cap_provenance", "selection_status", "rejection_category",
    "rejection_reason", "individual_over_caps", "binding_caps", "blocking_selected_ids", "partial_blocked_reason",
    "allocated_qty", "cost_basis", "cost_basis_declared", "quantity_unit", "unit_status",
)


def _v1_shortage_rows(data: Mapping[str, Any]) -> int:
    """Inventory rows with a positive shared-feasibility-v1 target need max(7-day demand - stock, 0)."""
    from services.optimality_gap_service import _target_shortage

    inventory = data.get("inventory")
    if not isinstance(inventory, pd.DataFrame):
        return 0
    return sum(1 for record in inventory.where(pd.notna(inventory), None).to_dict("records")
               if (_target_shortage(record)[0] or 0.0) > 0)


def run_workbooks(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Production pipeline field on every repository workbook (detail_level=core) plus per-candidate diagnostics."""
    from services.analysis_pipeline import build_v2_state, ensure_recommendations
    from services.data_loader import load_excel_data

    rows, diagnostics = [], []
    for folder in ("data", "samples", "Varo_DQN_training_samples_10pack"):
        for path in sorted((repo_root / folder).glob("*.xlsx")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = load_excel_data(path)
                _, source, _ = ensure_recommendations(dict(data))
                state = build_v2_state(data, detail_level="core")
            name = str(path.relative_to(repo_root)).replace("\\", "/")
            analysis = state["pipeline_result"].get("shared_feasibility_selection") or {}
            row = {"workbook": name, "status": analysis.get("status"), "recommendation_source": source,
                   "recommendation_count": len(state["recommendations"]),
                   "inventory_rows": int(len(data.get("inventory", []))),
                   "inventory_rows_with_v1_target_need": _v1_shortage_rows(data)}
            provenance = Counter(f"{item['cap_kind']}={item['provenance']}" for item in analysis.get("cap_provenance", []))
            row["cap_provenance_counts"] = json.dumps(dict(sorted(provenance.items())), ensure_ascii=False)
            for mode, plan in (analysis.get("modes") or {}).items():
                legacy = plan["legacy_comparison"]["legacy_validation"]
                view = plan["quantity_feasibility_view"]
                row.update({
                    f"{mode}_plan_status": plan["plan_status"], f"{mode}_selected": plan["selected_count"],
                    f"{mode}_qty": plan["total_allocated_qty"], f"{mode}_cost": plan["total_move_cost"],
                    f"{mode}_violations": plan["validation"]["violation_count"], f"{mode}_partial": plan["partial_count"],
                    f"{mode}_status_counts": json.dumps(plan["status_counts"], ensure_ascii=False),
                    f"{mode}_category_counts": json.dumps(plan["rejection_category_counts"], ensure_ascii=False),
                    f"{mode}_no_selection_cause": plan["no_selection_cause"],
                    f"{mode}_legacy_top5_violations": legacy["violation_count"],
                    f"{mode}_legacy_source_excess": legacy["source_excess_qty"],
                    f"{mode}_legacy_target_excess": legacy["target_excess_qty"],
                    f"{mode}_legacy_qty": plan["legacy_comparison"]["legacy_total_qty"],
                    f"{mode}_qty_delta": plan["legacy_comparison"]["qty_delta"],
                    f"{mode}_cost_delta": plan["legacy_comparison"]["cost_delta"],
                    f"{mode}_qtyview_status": view["plan_status"], f"{mode}_qtyview_selected": view["selected_count"],
                    f"{mode}_qtyview_qty": view["total_allocated_qty"], f"{mode}_qtyview_cost": view["total_move_cost"],
                    f"{mode}_qtyview_violations": view["validation"]["violation_count"],
                })
                for plan_name, current in (("MAIN", plan), ("QUANTITY_VIEW", view)):
                    for candidate in current["rows"]:
                        diagnostics.append({"workbook": name, "mode": mode, "plan": plan_name,
                                            **{column: candidate.get(column) for column in DIAGNOSTIC_COLUMNS}})
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def run_production_e2e(data_root: Path) -> dict[str, Any]:
    """The Suhyup 2026-07-31 production rows through build_v2_state (same path as the Seller Loss E2E)."""
    from services.seller_loss_input_validation import _run, load_suhyup_upload

    state = _run(load_suhyup_upload(data_root), data_root)
    analysis = state["pipeline_result"].get("shared_feasibility_selection") or {}
    result = {"recommendation_count": len(state["recommendations"]), "status": analysis.get("status"),
              "production_action_applied": analysis.get("production_action_applied"),
              "top5_route_ids": [item.get("route_id") for item in state["pipeline_result"]["top5"]],
              "cap_provenance_counts": dict(sorted(Counter(
                  f"{item['cap_kind']}={item['provenance']}" for item in analysis.get("cap_provenance", [])).items()))}
    for mode, plan in (analysis.get("modes") or {}).items():
        result[mode] = {
            "plan_status": plan["plan_status"], "feasibility_claim": plan["feasibility_claim"],
            "selected_ids": plan["selected_ids"], "total_allocated_qty": plan["total_allocated_qty"],
            "total_move_cost": plan["total_move_cost"], "partial_count": plan["partial_count"],
            "status_counts": plan["status_counts"], "violations": plan["validation"]["violation_count"],
            "legacy_top5_validation": {key: plan["legacy_comparison"]["legacy_validation"][key] for key in (
                "violation_count", "source_excess_qty", "target_excess_qty", "unverifiable_constraints", "certified")},
            "first_insufficient_reason": next((row["rejection_reason"] for row in plan["rows"]
                                               if row["selection_status"] == sf.INSUFFICIENT_CAP_DATA), None),
        }
    return result


def _git_head(repo_root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _check(check_id: str, description: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"check_id": check_id, "description": description, "status": "PASS" if ok else "FAIL", "detail": detail}


def run_validation(data_root: Path, output_dir: Path | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    data_root = Path(data_root)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    output_dir = Path(output_dir or data_root / OUTPUT_FOLDER)
    suhyup = run_suhyup(data_root)
    workbooks, diagnostics = run_workbooks(repo_root)
    e2e = run_production_e2e(data_root)
    by_day, regression, problems = suhyup["by_day"], suhyup["regression"], suhyup["problems"]
    summary = _summary(suhyup)
    sf_selected = suhyup["selected"][suhyup["selected"]["plan"].isin(PLAN_SF)]
    reg_ok = regression[["service_equal", "cost_equal", "route_ids_equal", "violations_equal"]].all(axis=1)
    checks = [
        _check("S1", "Suhyup 31 days processed", len(by_day) == 31, len(by_day)),
        _check("S2", "legacy Top-5 slice violates the benchmark target need on exactly the audited days",
               int((by_day["legacy_violation_count"] > 0).sum()) == 5,
               by_day.loc[by_day["legacy_violation_count"] > 0, "date"].tolist()),
        _check("S3", "SF_BENCHMARK (both policies) has 0 shared-constraint violations on 31/31 days",
               int(by_day["sf_benchmark_violation_count"].sum()) == 0 and int(by_day["sf_partial_violation_count"].sum()) == 0),
        _check("S4", "SF_BENCHMARK_ALL_OR_NOTHING equals the offline Varo Final selection on 31/31 days",
               bool(by_day["sf_benchmark_equals_offline_varo"].all()), int(by_day["sf_benchmark_equals_offline_varo"].sum())),
        _check("S5", "SF_BENCHMARK service equals MILP service on 31/31 days (same pool, caps, limit, 0/1 granularity)",
               bool(by_day["sf_benchmark_equals_milp_service"].all()), int(by_day["sf_benchmark_equals_milp_service"].sum())),
        _check("S6", "SF_STRICT_ACTUAL returns INSUFFICIENT_CAP_DATA (no plan) on every day: caps are PROXY",
               bool((by_day["sf_strict_plan_status"] == sf.INSUFFICIENT_CAP_DATA).all())),
        _check("S7", "every selected allocation <= recommended_qty, and allocated cost present",
               bool((sf_selected["allocated_qty"] <= sf_selected["original_recommended_qty"] + 1e-9).all())
               and bool(sf_selected[sf_selected["plan"] != "SF_STRICT_ACTUAL"]["allocated_move_cost"].notna().all())),
        _check("S8", "non-DQN 155 reference rows recomputed with current code are identical (service, cost, route_ids, violations)",
               bool(reg_ok.all()) and len(regression) == 155, f"{int(reg_ok.sum())}/{len(regression)}"),
        _check("S9", "MILP solver status optimal on 31/31 days", bool(by_day["milp_optimal"].all())),
        _check("S10", "problem-day generator quantities reconstructed from the inventory", bool(problems.empty) or all(
            member["reconstruction_matches"] for value in problems["legacy_members"] for member in json.loads(value))),
        _check("S11", "SF plans pass both pre-existing shared validators (offline _selection_violation, Pareto _default_feasible)",
               bool(by_day["existing_validators_ok"].all()), int(by_day["existing_validators_ok"].sum())),
        _check("S12", "SF plans identical after shuffling the input rows (31 days x 3 plans)",
               bool(by_day["order_invariant_under_shuffle"].all()), int(by_day["order_invariant_under_shuffle"].sum())),
        _check("S13", "offline reference strategies (Varo Final, MILP, Pareto, VHS, Greedy) have 0 violations under the SF validator",
               all(sum(json.loads(value).values()) == 0 for value in by_day["reference_strategies_violations_by_sf_validator"])),
        _check("W1", "all 16 repository workbooks carry the parallel field",
               len(workbooks) == 16 and bool((workbooks["status"] == "parallel_only").all())),
        _check("W2", "new plans have 0 violations on every workbook (both modes)",
               int(workbooks[[f"{mode}_violations" for mode in sf.MODES]].to_numpy().sum()) == 0),
        _check("W3", "a workbook reports NO_FEASIBLE_SELECTION only when every non-data candidate is infeasible on its own",
               all(set(json.loads(row[f"{mode}_category_counts"])) <= {"INDIVIDUALLY_OVER_CAP", "DATA_OR_INPUT", "DUPLICATE"}
                   for _, row in workbooks.iterrows() for mode in sf.MODES
                   if row[f"{mode}_plan_status"] == "NO_FEASIBLE_SELECTION")),
        _check("W4", "quantity-feasibility views have 0 violations on every workbook",
               int(workbooks[[f"{mode}_qtyview_violations" for mode in sf.MODES]].to_numpy().sum()) == 0),
        _check("E1", "Suhyup 2026-07-31 production E2E: STRICT_ACTUAL returns INSUFFICIENT_CAP_DATA (demand proxy)",
               e2e.get(sf.STRICT_ACTUAL, {}).get("plan_status") == sf.INSUFFICIENT_CAP_DATA,
               e2e.get(sf.STRICT_ACTUAL, {}).get("first_insufficient_reason")),
        _check("E2", "production action never applied", e2e.get("production_action_applied") is False),
    ]
    validation = {
        "selection_version": sf.SELECTION_VERSION, "git_head": _git_head(repo_root),
        "data_root": str(data_root), "inputs": {"candidates": CANDIDATES, "inventory": INVENTORY, "reference": REFERENCE},
        "checks": checks, "failed_checks": [item["check_id"] for item in checks if item["status"] == "FAIL"],
        "modes": {
            sf.STRICT_ACTUAL: f"caps with provenance in {sorted(sf.STRICT_ACCEPTED_PROVENANCE)} only; a candidate whose "
                              "source or target cap is missing or not accepted gets INSUFFICIENT_CAP_DATA",
            sf.BENCHMARK_PROXY: "every available cap including PROXY; results are labelled PROXY and are not an "
                                "operational guarantee",
        },
        "suhyup_cap_spec": {kind: {"columns": list(columns), "provenance": provenance, "basis": basis}
                            for kind, (columns, provenance, basis) in SUHYUP_CAP_SPEC.items()},
        "milp_comparison_conditions": {
            "candidate_pool": "same saved 20 candidates per day",
            "source_constraint": "MILP: source_surplus (PROXY); SF: source_surplus (PROXY) and source stock (DIRECT_REAL, never binding because surplus <= stock)",
            "target_constraint": "both: target_need_7d (PROXY)",
            "capacity": "none in either (no unit-compatible route capacity)",
            "top_n_limit": "both 5",
            "quantity_granularity": "MILP: fixed recommended_qty 0/1; SF_BENCHMARK_ALL_OR_NOTHING: same; SF_BENCHMARK_PARTIAL: may split -> not the same condition",
            "objective": "MILP: max service then min cost (exact within pool); SF: Varo Final greedy order",
            "same_condition_pairs": ["SF_BENCHMARK_ALL_OR_NOTHING vs MILP"],
            "different_condition_pairs": ["SF_BENCHMARK_PARTIAL vs MILP (granularity)", "SF_STRICT_ACTUAL vs MILP (caps)"],
        },
        "optimality_statement": "Equal to MILP only within the saved 20-candidate pool, generator cap 50, max 5 routes, "
                                "PROXY demand/surplus and the fixed objective; not a network-wide optimum.",
        "suhyup_runtime": suhyup["runtime"],
        "suhyup_totals": summary.to_dict("records"),
        "problem_days": problems.to_dict("records"),
        "production_e2e_20260731": e2e,
        "workbook_findings": [
            {"workbook": row["workbook"], "recommendation_source": row["recommendation_source"],
             "inventory_rows_with_v1_target_need": int(row["inventory_rows_with_v1_target_need"]),
             "legacy_top5_violations": int(row[f"{sf.BENCHMARK_PROXY}_legacy_top5_violations"]),
             **{f"{mode}_plan_status": row[f"{mode}_plan_status"] for mode in sf.MODES},
             "no_selection_cause": row[f"{sf.BENCHMARK_PROXY}_no_selection_cause"],
             "quantity_view": {"status": row[f"{sf.BENCHMARK_PROXY}_qtyview_status"],
                               "selected": int(row[f"{sf.BENCHMARK_PROXY}_qtyview_selected"]),
                               "qty": row[f"{sf.BENCHMARK_PROXY}_qtyview_qty"]}}
            for _, row in workbooks.iterrows()],
        "quantity_unit": "Suhyup stock/qty unit is UNKNOWN in the source; candidate qty and caps come from the same "
                         "stock_qty column (UNDECLARED_SAME_SOURCE). Totals across products add mixed units.",
        "limitations": [
            "cutline_passed='거리 초과' on all 620 Suhyup candidates is not treated as infeasible (same vocabulary as "
            "shared-feasibility-v1; distance_cutline_km=10 is a CONFIG default; T3 scope)",
            "route/vehicle capacity in quantity units does not exist in any dataset; ROUTE_CAPACITY is MISSING",
            "the selector is greedy in Varo Final order, not an optimizer; equality with MILP is structural here",
        ],
        "production_action_applied": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / OUTPUT_FILES[0], index=False, encoding="utf-8-sig")
    by_day.to_csv(output_dir / OUTPUT_FILES[1], index=False, encoding="utf-8-sig")
    suhyup["selected"].to_csv(output_dir / OUTPUT_FILES[2], index=False, encoding="utf-8-sig")
    suhyup["rejections"].to_csv(output_dir / OUTPUT_FILES[3], index=False, encoding="utf-8-sig")
    suhyup["caps"].to_csv(output_dir / OUTPUT_FILES[4], index=False, encoding="utf-8-sig")
    (output_dir / OUTPUT_FILES[5]).write_text(json.dumps(validation, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    problems.to_csv(output_dir / OUTPUT_FILES[6], index=False, encoding="utf-8-sig")
    workbooks.to_csv(output_dir / OUTPUT_FILES[7], index=False, encoding="utf-8-sig")
    diagnostics.to_csv(output_dir / OUTPUT_FILES[8], index=False, encoding="utf-8-sig")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_validation(args.data_root, args.output_dir)
    print(json.dumps({"failed_checks": result["failed_checks"], "checks": result["checks"],
                      "runtime": result["suhyup_runtime"]}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

"""Candidate generation, coverage and cost validation on the real Suhyup 31-day benchmark (P1).

Validation only.  It reads the actual inventory flow, the processed 2026-07-31 upload (stores, products, routes), the
saved 620 benchmark candidates, the 186-row reference bundle and the T1 / T3 bundles, and never writes there.  Outputs
go to <data-root>/_CANDIDATE_GENERATION_VALIDATION (local only, never into git); a re-run replaces only OUTPUT_FILES.

Per day it rebuilds the generator input exactly as the saved snapshots were built (T3 ``GeneratorRebuild``), enumerates
every lane the production generator looks at (``candidate_generation_research``), prices every lane with the official
tariff engine, attaches the offline PROXY caps, validates each lane, forms seven pools (LEGACY_20, TOP_30, TOP_40,
GENERATOR_ALL, MULTI_TARGET, ALL_VALID, ALL_LANES) and runs the unchanged Varo Final order, T1 selector and
lexicographic MILP on each.  LEGACY_20 is also checked against the production generator and the saved candidates, and
the saved candidates (pipeline VHS) are re-run to keep T1 / T3 / the 186 reference rows reproduced.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import scipy

from services import candidate_generation_research as cgr
from services import candidate_generator as cg
from services import milp_benchmark_integrity as mbi
from services import milp_benchmark_integrity_validation as mbv
from services import shared_feasibility_selection as sf
from services import shared_feasibility_validation as sfv
from services.real_data_adapters import DATA_ROOT
from services.seller_decision_validation import _env

OUTPUT_FOLDER = "_CANDIDATE_GENERATION_VALIDATION"
OUTPUT_FILES = (
    "candidate_coverage_summary.csv",
    "candidate_coverage_by_day.csv",
    "candidate_coverage_by_policy.csv",
    "candidate_cutoff_effect.csv",
    "candidate_newly_selected_routes.csv",
    "candidate_cost_comparison.csv",
    "candidate_feasibility_validation.csv",
    "candidate_runtime_analysis.csv",
    "candidate_generation_validation.json",
    "candidate_generation_trace.csv",
    "candidate_universe_rows.parquet",
    "candidate_policy_selections.parquet",
)
DATASET = mbv.DATASET
REAL_DATA_ROOT_ENV = mbv.REAL_DATA_ROOT_ENV
BASELINE = cgr.LEGACY_20
COMPARED_PLANS = ("VARO_FINAL", "T1_ALL_OR_NOTHING", "T1_PARTIAL", "MILP", "GREEDY")
NEW_ROUTE_PLANS = ("VARO_FINAL", "T1_ALL_OR_NOTHING", "T1_PARTIAL", "MILP")
CAP_DEFINITION = json.dumps({kind: [list(columns), provenance, basis] for kind, (columns, provenance, basis)
                             in sorted(sfv.SUHYUP_CAP_SPEC.items())}, ensure_ascii=False, sort_keys=True)
SHUFFLE_SEEDS = (11, 23)
PROTECTED_NOTE = "no existing tracked file may change: the research path is new files only"


# ------------------------------------------------------------------------------------------------ one day


def _keyed(frame: pd.DataFrame) -> list[tuple]:
    if frame.empty:
        return []
    return sorted(zip(frame["lane_key"].astype(str), pd.to_numeric(frame["recommended_qty"]).astype(float),
                      pd.to_numeric(frame["move_cost"]).round(6).astype(float)))


def build_universe(rebuild: mbv.GeneratorRebuild, flow: pd.DataFrame, date: str, *, shuffle_seed: int | None = None
                   ) -> dict[str, Any]:
    """Universe -> priced -> capped -> validated, with stage timings (one day)."""
    upload = rebuild.upload(date)
    if shuffle_seed is not None:
        upload = {**upload, "inventory": upload["inventory"].sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)}
    store_ids, _ = cg._store_ids_by_type(upload["stores"])
    product_ids = upload["products"]["product_id"].astype(str).tolist()
    (lanes, pairs, trace), enum_ms, enum_kb = cgr._measure(lambda: cgr.enumerate_universe(upload, snapshot_date=date))
    priced, price_ms, price_kb = cgr._measure(lambda: cgr.price_universe(lanes, cgr.real_tariff_pricer))
    capped, caps_ms, _ = cgr._measure(lambda: cgr.attach_benchmark_caps(priced, flow))
    universe, valid_ms, _ = cgr._measure(lambda: cgr.validate_universe(capped, store_ids=store_ids, product_ids=product_ids))
    return {"upload": upload, "lanes": universe, "pairs": pairs, "trace": trace, "store_ids": store_ids,
            "timing": {"enumerate_ms": enum_ms, "enumerate_peak_kb": enum_kb, "price_universe_ms": price_ms,
                       "price_universe_peak_kb": price_kb, "caps_ms": caps_ms, "validate_ms": valid_ms}}


def run_day(rebuild: mbv.GeneratorRebuild, flow: pd.DataFrame, date: str, saved_day: pd.DataFrame,
            saved_recompute: Any) -> dict[str, Any]:
    built = build_universe(rebuild, flow, date)
    universe, pairs, trace, upload = built["lanes"], built["pairs"], built["trace"], built["upload"]
    (production, production_stats), production_ms, production_kb = cgr._measure(lambda: cg.generate_candidates(dict(upload)))
    valid_total = int((universe["validity_status"] == cgr.VALID).sum())
    pools, timing = {}, {}
    for policy in cgr.POLICIES:
        pool, extract_ms, _ = cgr._measure(lambda: cgr.policy_pool(universe, policy))
        columns = ["route_id", "product_id", "source_id", "target_id", "dc_id", "route_type", "recommended_qty"]
        repriced, pool_price_ms, pool_price_kb = cgr._measure(lambda: cgr.real_tariff_pricer(pool[columns]) if not pool.empty else pool)
        consistent = pool.empty or bool((pd.to_numeric(repriced["move_cost"], errors="coerce").round(6).values
                                         == pool["move_cost"].round(6).values).all())
        pools[policy] = pool
        timing[policy] = {"extract_ms": extract_ms, "pool_pricing_ms": pool_price_ms, "pool_pricing_peak_kb": pool_price_kb,
                          "pool_cost_equals_universe_cost": consistent}
    evaluations = {}
    for policy, pool in pools.items():
        evaluations[policy] = cgr.evaluate_pool(
            pool, date=date, dataset=DATASET, cap_spec=sfv.SUHYUP_CAP_SPEC, milp_cap_spec=mbv.MILP_CAP_SPEC,
            cost_recompute=sf.real_transport_cost_recompute,
            pool_total=len(pool) if policy == cgr.ALL_LANES else max(valid_total, len(pool)))
    ranked_saved = sfv.ranked_day(saved_day, date)
    saved_eval = cgr.evaluate_pool(ranked_saved, date=date, dataset=DATASET, cap_spec=sfv.SUHYUP_CAP_SPEC,
                                   milp_cap_spec=mbv.MILP_CAP_SPEC, cost_recompute=saved_recompute,
                                   pool_total=max(valid_total, len(ranked_saved)), rank=False)
    membership = cgr.pool_membership(universe, pools)
    return {"date": date, "universe": universe, "pairs": pairs, "trace": trace, "pools": pools, "evals": evaluations,
            "saved_eval": saved_eval, "ranked_saved": ranked_saved, "membership": membership, "valid_total": valid_total,
            "production": production, "production_stats": production_stats,
            "timing": {**built["timing"], "production_generator_ms": production_ms, "production_generator_peak_kb": production_kb,
                       "policies": timing}}


# ------------------------------------------------------------------------------------------------ rows


def _ids(values: Sequence[str]) -> str:
    return "|".join(sorted(map(str, values)))


def _cost_mix(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return "|".join(f"{k}={v}" for k, v in sorted(Counter(frame["cost_provenance"].fillna(cgr.UNKNOWN)).items()))


def funnel_rows(day: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Section 2 (production step trace) and section 3 (A-G funnel) for one day."""
    date, trace, universe, pairs = day["date"], day["trace"], day["universe"], day["pairs"]
    lanes = universe[universe["route_status"].isin(cgr.ROUTE_OK + (cgr.ROUTE_UNVERIFIED, cgr.ROUTE_FORBIDDEN))]
    eligible_pairs = int((pairs["eligibility"] == "ELIGIBLE").sum())
    n_stores = trace["store_nodes"]
    rows = []
    steps = [
        ("1", "inventory rows (input)", trace["inventory_rows_input"], trace["inventory_rows_input"], "upload inventory sheet"),
        ("2", "rows at STORE nodes", trace["inventory_rows_input"], trace["inventory_rows_at_store_nodes"], "store_id in STORE nodes"),
        ("3", "product in product master", trace["inventory_rows_at_store_nodes"],
         trace["inventory_rows_at_store_nodes"] - trace["pairs_product_not_in_master"], "product_id in products"),
        ("4", "stock > 0", trace["inventory_rows_at_store_nodes"] - trace["pairs_product_not_in_master"],
         trace["inventory_rows_at_store_nodes"] - trace["pairs_product_not_in_master"] - trace["pairs_no_stock"], "stock_qty > 0"),
        ("5", "source eligibility", trace["inventory_rows_at_store_nodes"] - trace["pairs_product_not_in_master"] - trace["pairs_no_stock"],
         eligible_pairs, "stock > product median, or days_to_expiry <= 7"),
        ("6", "targets considered (other STORE nodes)", eligible_pairs, int(len(lanes)), f"x {n_stores - 1} other stores"),
        ("7", "route resolved (direct row or DC legs)", int(len(lanes)), trace["lanes_with_route"], "_resolve_route not None"),
        ("8", "one target per source-product row", trace["lanes_with_route"], trace["one_target_rule_kept"],
         "largest need; road distance (real mode) or route cost breaks ties; target id desc last"),
        ("9", "duplicate (product, source, target)", trace["one_target_rule_kept"],
         trace["one_target_rule_kept"] - trace["duplicate_removed"], "first inventory row kept"),
        ("10", "quantity >= 1", trace["one_target_rule_kept"] - trace["duplicate_removed"],
         trace["one_target_rule_kept"] - trace["duplicate_removed"], "int(max(1, ...)) never 0"),
        ("11", "positive expected saving (legacy mode only)", trace["one_target_rule_kept"] - trace["duplicate_removed"],
         trace["generator_pool_before_cut"], "skipped in real-transport mode (saving 0)"),
        ("12", "MAX_CANDIDATES cut by candidate_score", trace["generator_pool_before_cut"], trace["kept_by_max_candidates"],
         f"top {cg.MAX_CANDIDATES}; cost not known yet (priced after the cut)"),
        ("13", "pipeline tariff pricing, Varo Final rank (no exclusion)", trace["kept_by_max_candidates"],
         trace["kept_by_max_candidates"], "real_transport_enrichment; vhs_score_engine._rank_varo_operational"),
        ("14", "T1 shared-feasibility selection (BENCHMARK_PROXY)", trace["kept_by_max_candidates"],
         day["evals"][cgr.LEGACY_20]["plans"]["T1_ALL_OR_NOTHING"]["selected_count"], "max_routes 5, PROXY caps"),
    ]
    for order, step, before, after, condition in steps:
        rows.append({"date": date, "table": "PRODUCTION_TRACE", "stage": order, "description": step, "input_count": int(before),
                     "output_count": int(after), "excluded_count": int(before) - int(after) if order not in ("6",) else None,
                     "condition": condition})
    valid = universe["validity_status"] == cgr.VALID
    usable = universe["route_status"].isin(cgr.ROUTE_OK) & ~universe["is_duplicate"].astype(bool)
    a_lanes = trace["inventory_rows_at_store_nodes"] * (n_stores - 1)
    funnel = [
        ("A", "source-product rows at stores x other stores (potential lanes)", a_lanes),
        ("B", "lanes of eligible source rows", int(len(lanes))),
        ("C", "B + individual conditions: not self, unique, target holds the product, need evidenced, unit consistent",
         int((universe["chk_not_self"] & universe["chk_unique"] & universe["chk_target_holds_product"]
              & universe["chk_need_evidenced"] & (universe["chk_unit"] | universe["route_status"].eq(cgr.ROUTE_UNVERIFIED))).sum())),
        ("D", "C + route in the network table", int((universe["chk_not_self"] & universe["chk_unique"]
                                                     & universe["chk_target_holds_product"] & universe["chk_need_evidenced"]
                                                     & universe["chk_route"]).sum())),
        ("E", "D + quantity, cost and both caps computable (= VALID)", int(valid.sum())),
    ]
    for policy in cgr.POLICIES:
        funnel.append((f"F:{policy}", f"lanes in pool {policy}", int(len(day["pools"][policy]))))
        funnel.append((f"G:{policy}", f"selected by T1 BENCHMARK_PROXY in {policy}",
                       int(day["evals"][policy]["plans"]["T1_ALL_OR_NOTHING"]["selected_count"])))
    for stage, description, count in funnel:
        rows.append({"date": date, "table": "FUNNEL_A_G", "stage": stage, "description": description, "input_count": None,
                     "output_count": count, "excluded_count": None, "condition": None})
    rows.append({"date": date, "table": "FUNNEL_A_G", "stage": "usable", "description": "routed, non-duplicate lanes",
                 "input_count": None, "output_count": int(usable.sum()), "excluded_count": None, "condition": None})
    return rows


def day_rows(day: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    date, universe, pools, evals = day["date"], day["universe"], day["pools"], day["evals"]
    base_pool, base_eval = pools[BASELINE], evals[BASELINE]
    base_keys = set(base_pool["lane_key"])
    base_keyed = set(_keyed(base_pool))
    best_known = evals[cgr.ALL_VALID]["plans"]["MILP"]["lane_keys"]
    membership = day["membership"].drop_duplicates("lane_key").set_index("lane_key", drop=False)
    by_key = universe[~universe["is_duplicate"].astype(bool)].set_index("lane_key", drop=False)
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in ("by_day", "cutoff", "new", "cost", "feasibility",
                                                                   "runtime", "shortage", "selections")}
    counts = universe["validity_status"].value_counts().to_dict()
    for policy in cgr.POLICIES:
        pool, ev = pools[policy], evals[policy]
        plans, milp = ev["plans"], ev["plans"]["MILP"]
        coverage = cgr.coverage_metrics(pool, universe, reference_lane_keys=best_known)
        diversity = cgr.diversity_metrics(pool)
        shortage = cgr.shortage_diagnosis(day["pairs"], universe, pool, policy)
        short_counts = shortage["category"].value_counts().to_dict()
        superset = base_keyed <= set(_keyed(pool))
        selected_frame = pool[pool["lane_key"].isin(milp["lane_keys"])]
        row = {
            "date": date, "policy": policy, "family": cgr.POLICY_SPEC[policy]["family"],
            "universe_lanes": int(len(universe)), "universe_valid_lanes": day["valid_total"],
            "universe_invalid_lanes": int(len(universe) - day["valid_total"]),
            "pool_size": int(len(pool)), "pool_valid": int((pool["validity_status"] == cgr.VALID).sum()) if not pool.empty else 0,
            "pool_invalid": int((pool["validity_status"] != cgr.VALID).sum()) if not pool.empty else 0,
            "pool_invalid_statuses": json.dumps(pool["validity_status"][pool["validity_status"] != cgr.VALID].value_counts().to_dict()) if not pool.empty else "{}",
            "pool_missing_target_cap": int(pool["target_need_7d"].isna().sum()) if not pool.empty else 0,
            "pool_duplicates": int(pool.duplicated("lane_key").sum()) if not pool.empty else 0,
            "pool_at_move_cap": int((pd.to_numeric(pool["recommended_qty"]) >= cg._MOVE_CAP).sum()) if not pool.empty else 0,
            "pool_cost_provenance": _cost_mix(pool), "pool_cap_complete_share": ev["cap_complete_share"],
            "eval_excluded_data_invalid": ev["excluded_count"],
            "superset_of_legacy_20": superset,
            **{f"{plan.lower()}_selected": plans[plan]["selected_count"] for plan in plans},
            **{f"{plan.lower()}_service": plans[plan]["service"] for plan in plans},
            **{f"{plan.lower()}_cost": plans[plan]["cost"] for plan in plans},
            **{f"{plan.lower()}_violations": plans[plan]["check"]["violation_count"] for plan in plans},
            "t1_strict_plan_status": plans["T1_STRICT_ACTUAL"]["plan_status"],
            "t1_partial_count": plans["T1_PARTIAL"]["partial_count"],
            "t1_benchmark_unchecked": "|".join(plans["T1_ALL_OR_NOTHING"]["unchecked_constraints"] or []),
            "milp_solver_status": ev["milp_evidence"]["solver_status"], "milp_termination": ev["milp_evidence"]["termination_reason"],
            "milp_stage2_mip_gap": ev["milp_evidence"]["stage2_mip_gap"],
            "milp_enumeration_status": ev["enumeration"]["status"], "milp_enumeration_proven": ev["enumeration"]["proven"],
            "milp_alternative_optima": ev["enumeration"]["alternative_optima"],
            "milp_benchmark_scope": ev["scope"]["benchmark_scope"], "milp_optimality_claim": ev["scope"]["optimality_claim"],
            "milp_route_ids": _ids(milp["route_ids"]), "varo_final_route_ids": _ids(plans["VARO_FINAL"]["route_ids"]),
            "t1_route_ids": _ids(plans["T1_ALL_OR_NOTHING"]["route_ids"]),
            "milp_selected_cost_provenance": _cost_mix(selected_frame),
            "milp_selected_mean_distance_km": round(float(pd.to_numeric(selected_frame["distance_km"]).mean()), 3) if not selected_frame.empty else None,
            "varo_final_vs_milp": plans["VARO_FINAL"]["comparison_status"],
            "varo_final_decision_cost_gap": plans["VARO_FINAL"]["decision_cost_gap"],
            "t1_vs_milp": plans["T1_ALL_OR_NOTHING"]["comparison_status"],
            "t1_decision_cost_gap": plans["T1_ALL_OR_NOTHING"]["decision_cost_gap"],
            "varo_final_equals_t1_routes": set(plans["VARO_FINAL"]["lane_keys"]) == set(plans["T1_ALL_OR_NOTHING"]["lane_keys"]),
            "new_lanes_vs_legacy_20": int(len(set(pool["lane_key"]) - base_keys)) if not pool.empty else 0,
            "milp_new_lanes_selected": int(len(set(milp["lane_keys"]) - base_keys)),
            **{f"cov_{name}": (value["value"] if isinstance(value, dict) else value) for name, value in coverage.items()},
            **{f"cov_{name}_num_den": f"{value['numerator']}/{value['denominator']}" for name, value in coverage.items()
               if isinstance(value, dict)},
            **{f"div_{name}": (json.dumps(value) if isinstance(value, dict) else value) for name, value in diversity.items()},
            **{f"short_{category}": int(short_counts.get(category, 0)) for category in cgr.SHORTAGE_CATEGORIES},
            "generator_cut_boundary_tie": day["trace"]["cut_boundary_tie"],
        }
        out["by_day"].append(row)
        added = sorted(set(pool["lane_key"]) - base_keys) if not pool.empty else []
        removed = sorted(base_keys - set(pool["lane_key"])) if not pool.empty else sorted(base_keys)
        added_reasons = Counter(membership.loc[key, f"pool_{BASELINE}"] if key in membership.index else "UNKNOWN" for key in added)
        out["cutoff"].append({
            "date": date, "policy": policy, "pool_size": len(pool), "legacy_20_pool_size": len(base_pool),
            "added_vs_legacy_20": len(added), "removed_vs_legacy_20": len(removed),
            "added_by_legacy_exclusion_reason": json.dumps(dict(sorted(added_reasons.items()))),
            "added_lane_keys": "|".join(added), "removed_lane_keys": "|".join(removed),
            "generator_pool_before_cut": day["trace"]["generator_pool_before_cut"],
            "cut_by_max_candidates": day["trace"]["cut_by_max_candidates"],
            "one_target_rule_dropped": day["trace"]["one_target_rule_dropped"],
            "cut_boundary_tie": day["trace"]["cut_boundary_tie"],
            "score_rank20": _score_at(base_pool, pools[cgr.GENERATOR_ALL], 20),
            "score_rank21": _score_at(base_pool, pools[cgr.GENERATOR_ALL], 21),
            "milp_service": milp["service"], "milp_cost": milp["cost"],
            "milp_cost_delta_vs_legacy_20": round(milp["cost"] - base_eval["plans"]["MILP"]["cost"], 6)
            if abs(milp["service"] - base_eval["plans"]["MILP"]["service"]) <= 1e-9 else None,
            "best_known_plan_inclusion": coverage["best_known_plan_inclusion"]["value"],
            "milp_new_lanes_selected": int(len(set(milp["lane_keys"]) - base_keys)),
        })
        for plan in NEW_ROUTE_PLANS:
            for key in sorted(set(plans[plan]["lane_keys"]) - base_keys):
                lane = by_key.loc[key]
                choice = universe[(universe["inventory_row"] == lane["inventory_row"]) & universe["generator_choice"].astype(bool)]
                choice = choice.iloc[0] if not choice.empty else None
                out["new"].append({
                    "date": date, "policy": policy, "plan": plan, "lane_key": key, "route_id": lane["route_id"],
                    "product_id": lane["product_id"], "source_id": lane["source_id"], "target_id": lane["target_id"],
                    "recommended_qty": lane["recommended_qty"], "move_cost": lane["move_cost"],
                    "cost_per_unit": lane["cost_per_unit"], "distance_km": lane["distance_km"],
                    "cost_provenance": lane["cost_provenance"], "route_status": lane["route_status"],
                    "validity_status": lane["validity_status"], "source_surplus_cap": lane["source_surplus"],
                    "target_need_cap": lane["target_need_7d"], "gen_target_need": lane["gen_target_need"],
                    "generator_rank": lane["generator_rank"], "generator_target_rank": lane["generator_target_rank"],
                    "candidate_score": lane["candidate_score"],
                    "legacy_20_exclusion": membership.loc[key, f"pool_{BASELINE}"] if key in membership.index else None,
                    "generator_kept_target": None if choice is None else choice["target_id"],
                    "generator_kept_target_distance_km": None if choice is None else choice["distance_km"],
                    "generator_kept_target_cost": None if choice is None else choice["move_cost"],
                    "generator_kept_target_need": None if choice is None else choice["gen_target_need"],
                })
        for plan in COMPARED_PLANS:
            compared = cgr.compare_pools(ev, base_eval, plan, cap_definition=CAP_DEFINITION, superset=superset)
            item = plans[plan]
            out["cost"].append({
                "date": date, "policy": policy, "plan": plan, "service": item["service"], "cost": item["cost"],
                "selected_count": item["selected_count"], "legacy_20_service": base_eval["plans"][plan]["service"],
                "legacy_20_cost": base_eval["plans"][plan]["cost"], **compared,
                "reason_codes": "|".join(compared["reason_codes"]),
                "within_pool_vs_milp": item["comparison_status"], "within_pool_decision_cost_gap": item["decision_cost_gap"],
                "within_pool_decision_service_gap": item["decision_service_gap"],
                "milp_solver_status": ev["milp_evidence"]["solver_status"],
                "cost_label": "COMPUTED_TARIFF_ESTIMATE_NOT_ACTUAL_SPEND",
            })
        failed = Counter(flag for flags in pool["validity_flags"].dropna() for flag in str(flags).split("|")) if not pool.empty else Counter()
        out["feasibility"].append({
            "date": date, "scope": "POOL", "policy": policy, "rows": len(pool),
            **{f"status_{status}": int((pool["validity_status"] == status).sum()) if not pool.empty else 0
               for status in sorted(set(counts) | {cgr.VALID})},
            **{f"fails_{check}": int(failed.get(code or check, 0)) for check, code in cgr.VALIDITY_CHECKS if code},
            "duplicate_candidate_ids": int(pool["candidate_id"].duplicated().sum()) if not pool.empty else 0,
            "route_unverified": int((pool["route_status"] == cgr.ROUTE_UNVERIFIED).sum()) if not pool.empty else 0,
            "operational_evidence_observed": int((pool["route_operational_evidence"] != cgr.OPERATION_NOT_OBSERVED).sum()) if not pool.empty else 0,
            "unit_declared": int(pool["unit_status"].isin(("MATCHED",)).sum()) if not pool.empty else 0,
            "cost_provenance": _cost_mix(pool),
            "route_capacity_cap_present": 0, "dc_capacity_cap_present": 0,
            **{f"selection_{plan.lower()}_violations": plans[plan]["check"]["violation_count"] for plan in plans},
            "selection_strict_unverifiable": "|".join(plans["MILP"]["check"]["strict_unverifiable_constraints"]),
        })
        policy_timing = day["timing"]["policies"][policy]
        generation = (day["timing"]["production_generator_ms"] if policy == BASELINE
                      else day["timing"]["enumerate_ms"] + policy_timing["extract_ms"])
        t = ev["timing"]
        out["runtime"].append({
            "date": date, "policy": policy, "pool_size": len(pool), "generation_ms": round(generation, 3),
            "generation_basis": "production generate_candidates" if policy == BASELINE else "research enumeration + pool extraction",
            "pool_pricing_ms": policy_timing["pool_pricing_ms"], "rank_ms": t["rank_ms"], "varo_final_ms": t["varo_ms"],
            "t1_three_modes_ms": t["t1_ms"], "milp_ms": t["milp_ms"], "enumeration_ms": t["enumeration_ms"],
            "total_ms": round(generation + policy_timing["pool_pricing_ms"] + t["rank_ms"] + t["varo_ms"] + t["t1_ms"] + t["milp_ms"], 3),
            "rank_peak_kb": t["rank_peak_kb"], "t1_peak_kb": t["t1_peak_kb"], "milp_peak_kb": t["milp_peak_kb"],
            "pool_pricing_peak_kb": policy_timing["pool_pricing_peak_kb"],
            "milp_time_limit_s": ev["milp_evidence"]["time_limit_s"], "milp_solver_status": ev["milp_evidence"]["solver_status"],
            "pool_cost_equals_universe_cost": policy_timing["pool_cost_equals_universe_cost"],
        })
        shortage["date"] = date
        out["shortage"].extend(shortage.to_dict("records"))
        for plan in ("VARO_FINAL", "T1_ALL_OR_NOTHING", "T1_PARTIAL", "MILP", "GREEDY"):
            for key in plans[plan]["lane_keys"]:
                out["selections"].append({"date": date, "policy": policy, "plan": plan, "lane_key": key,
                                          "in_legacy_20_pool": key in base_keys})
    return out


def _score_at(base: pd.DataFrame, full: pd.DataFrame, rank: int) -> float | None:
    return float(full.iloc[rank - 1]["candidate_score"]) if len(full) >= rank else None


# ------------------------------------------------------------------------------------------------ Suhyup 31 days


def run_suhyup(data_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    saved = sfv.load_suhyup_candidates(data_root)
    reference = pd.read_csv(data_root / sfv.REFERENCE, dtype={"date": str})
    flow = pd.read_csv(data_root / sfv.INVENTORY, dtype={"date": str, "center_code": str, "product_code": str})
    rebuild = mbv.GeneratorRebuild(data_root)
    t1_bundle = pd.read_csv(data_root / "_SHARED_FEASIBILITY_VALIDATION" / "shared_feasibility_by_day.csv", dtype=str)
    t3_bundle = pd.read_csv(data_root / mbv.OUTPUT_FOLDER / "milp_benchmark_by_day.csv", dtype=str)
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ("by_day", "cutoff", "new", "cost", "feasibility",
                                                                    "runtime", "shortage", "selections")}
    trace_rows, universe_frames, fidelity, regression, golden, determinism, peaks = [], [], [], [], [], [], []
    with _env(REAL_DATA_ROOT_ENV, str(data_root)):
        saved_recompute = sfv.TariffRecompute(saved)
        for date in sorted(saved["snapshot_date"].astype(str).unique()):
            saved_day = saved[saved["snapshot_date"] == date]
            day = run_day(rebuild, flow, date, saved_day, saved_recompute)
            for name, items in day_rows(day).items():
                rows[name].extend(items)
            trace_rows.extend(funnel_rows(day))
            universe = day["universe"].copy()
            membership = day["membership"].drop(columns=["lane_key", "inventory_row"])
            universe = universe.merge(membership, on="route_id", how="left")
            universe_frames.append(universe)
            fidelity.append(_fidelity(day, saved_day))
            regression.append(_regression(day, t1_bundle, t3_bundle))
            golden.extend(_golden(day, reference))
            determinism.append(_determinism(day, rebuild, flow))
            peaks.append({"date": date, "process_peak_working_set_mb_after_day": mbv.process_peak_working_set_mb()})
    frames = {name: pd.DataFrame(items) for name, items in rows.items()}
    frames["runtime"] = frames["runtime"].merge(pd.DataFrame(peaks), on="date", how="left")
    return {**frames, "trace": pd.DataFrame(trace_rows), "universe": pd.concat(universe_frames, ignore_index=True),
            "fidelity": pd.DataFrame(fidelity), "regression": pd.DataFrame(regression), "golden": pd.DataFrame(golden),
            "determinism": pd.DataFrame(determinism), "wall_seconds": round(time.perf_counter() - started, 3)}


def _fidelity(day: Mapping[str, Any], saved_day: pd.DataFrame) -> dict[str, Any]:
    """LEGACY_20 vs the production generator, the saved 620 and T3's uncut pools."""
    pools, production, upload = day["pools"], day["production"], None
    legacy = cgr.production_records(pools[cgr.LEGACY_20])
    same_production = production is not None and production.reset_index(drop=True).astype(str).equals(
        legacy[list(production.columns)].astype(str))
    saved_key = sorted(zip(saved_day["route_id"].astype(str), saved_day["product_id"].astype(str), saved_day["source_id"].astype(str),
                           saved_day["target_id"].astype(str), saved_day["recommended_qty"].astype(float),
                           pd.to_numeric(saved_day["move_cost"]).astype(float)))
    pool = pools[cgr.LEGACY_20]
    mine_key = sorted(zip(pool["route_id"].astype(str), pool["product_id"].astype(str), pool["source_id"].astype(str),
                          pool["target_id"].astype(str), pool["recommended_qty"].astype(float), pool["move_cost"].astype(float)))
    full = cgr.production_records(pools[cgr.GENERATOR_ALL])
    return {"date": day["date"], "legacy_20_equals_production_generator": bool(same_production),
            "legacy_20_equals_saved_candidates": saved_key == mine_key,
            "generator_all_size": len(full), "production_stats_count": day["production_stats"].get("count"),
            "trace_generator_pool_before_cut": day["trace"]["generator_pool_before_cut"],
            "all_lanes_size": len(pools[cgr.ALL_LANES]), "valid_lanes": day["valid_total"],
            "pool_costs_consistent": all(item["pool_cost_equals_universe_cost"] for item in day["timing"]["policies"].values()),
            "research_legacy_selection_equals_saved_pipeline_vhs": {
                plan: set(day["evals"][cgr.LEGACY_20]["plans"][plan]["route_ids"]) == set(day["saved_eval"]["plans"][plan]["route_ids"])
                for plan in ("VARO_FINAL", "T1_ALL_OR_NOTHING", "MILP", "GREEDY")},
            "research_legacy_totals_equal_saved": {
                plan: abs(day["evals"][cgr.LEGACY_20]["plans"][plan]["cost"] - day["saved_eval"]["plans"][plan]["cost"]) <= 1e-6
                and abs(day["evals"][cgr.LEGACY_20]["plans"][plan]["service"] - day["saved_eval"]["plans"][plan]["service"]) <= 1e-9
                for plan in ("VARO_FINAL", "T1_ALL_OR_NOTHING", "MILP", "GREEDY")}}


def _regression(day: Mapping[str, Any], t1_bundle: pd.DataFrame, t3_bundle: pd.DataFrame) -> dict[str, Any]:
    """Saved 620 (pipeline VHS) re-run: T1 and T3 bundle values reproduced (read-only comparison)."""
    date, plans = day["date"], day["saved_eval"]["plans"]
    t1 = t1_bundle[t1_bundle["date"] == date].iloc[0]
    t3 = t3_bundle[t3_bundle["date"] == date].iloc[0]
    split = lambda text: set(str(text).split("|")) if isinstance(text, str) and text else set()
    return {
        "date": date,
        "t1_benchmark_service": plans["T1_ALL_OR_NOTHING"]["service"], "t1_benchmark_cost": plans["T1_ALL_OR_NOTHING"]["cost"],
        "t1_benchmark_violations": plans["T1_ALL_OR_NOTHING"]["check"]["violation_count"],
        "t1_matches_bundle": abs(plans["T1_ALL_OR_NOTHING"]["service"] - float(t1["sf_benchmark_service_qty"])) <= 1e-9
        and abs(plans["T1_ALL_OR_NOTHING"]["cost"] - float(t1["sf_benchmark_cost"])) <= 1e-6
        and set(plans["T1_ALL_OR_NOTHING"]["route_ids"]) == split(t1["sf_benchmark_route_ids"]),
        "t1_partial_matches_bundle": abs(plans["T1_PARTIAL"]["cost"] - float(t1["sf_partial_cost"])) <= 1e-6
        and set(plans["T1_PARTIAL"]["route_ids"]) == split(t1["sf_partial_route_ids"]),
        "t1_strict_status": plans["T1_STRICT_ACTUAL"]["plan_status"],
        "t1_strict_matches_bundle": plans["T1_STRICT_ACTUAL"]["plan_status"] == t1["sf_strict_plan_status"],
        "t3_milp_status": day["saved_eval"]["milp_evidence"]["solver_status"],
        "t3_milp_matches_bundle": abs(plans["MILP"]["service"] - float(t3["milp_service_qty"])) <= 1e-9
        and abs(plans["MILP"]["cost"] - float(t3["milp_move_cost"])) <= 1e-6
        and set(plans["MILP"]["route_ids"]) == split(t3["milp_route_ids"]),
        "t3_varo_routes_match_bundle": set(plans["VARO_FINAL"]["route_ids"]) == split(t3["varo_final_route_ids"]),
        "t3_enumeration_proven": day["saved_eval"]["enumeration"]["proven"],
        "t3_varo_vs_milp": plans["VARO_FINAL"]["comparison_status"],
        "t3_varo_zero_gap": mbi.SAME_RESTRICTED_OBJECTIVE in plans["VARO_FINAL"]["zero_gap_flags"],
        "t3_partial_not_comparable": plans["T1_PARTIAL"]["comparison_status"] == mbi.NOT_COMPARABLE,
    }


def _golden(day: Mapping[str, Any], reference: pd.DataFrame) -> list[dict[str, Any]]:
    """The 186 reference rows recomputed with current code (as T1/T3 do)."""
    date, ranked = day["date"], day["ranked_saved"]
    refs = sfv._reference_rows(ranked, date)
    dqn = mbv._dqn_rows(ranked, date, float(refs["MILP"]["service_qty"]))
    if dqn is not None:
        refs["DQN"] = dqn
    rows = []
    for strategy, row in refs.items():
        saved = reference[(reference["date"] == date) & (reference["strategy"] == strategy)].iloc[0]
        rows.append({"date": date, "strategy": strategy,
                     "equal": abs(float(row["service_qty"]) - float(saved["service_qty"])) <= 1e-6
                     and abs(float(row["total_cost"]) - float(saved["total_cost"])) <= 1e-6
                     and str(row["route_ids"]) == str(saved["route_ids"])
                     and int(row["feasibility_violations"]) == int(saved["feasibility_violations"])
                     and str(row["status"]) == str(saved["status"])})
    return rows


def _determinism(day: Mapping[str, Any], rebuild: mbv.GeneratorRebuild, flow: pd.DataFrame) -> dict[str, Any]:
    """Repeat run and shuffled inventory rows: uncut pools and ALL_VALID plans must not move; the generator family is
    reported as is (production keeps ties at the cut in inventory row order)."""
    base = {policy: _keyed(pool) for policy, pool in day["pools"].items()}
    base_plan = day["evals"][cgr.ALL_VALID]["plans"]
    result = {"date": day["date"]}
    for label, seed in (("repeat", None), *((f"shuffle_{seed}", seed) for seed in SHUFFLE_SEEDS)):
        built = build_universe(rebuild, flow, day["date"], shuffle_seed=seed)
        pools = {policy: cgr.policy_pool(built["lanes"], policy) for policy in cgr.POLICIES}
        ev = cgr.evaluate_pool(pools[cgr.ALL_VALID], date=day["date"], dataset=DATASET, cap_spec=sfv.SUHYUP_CAP_SPEC,
                               milp_cap_spec=mbv.MILP_CAP_SPEC, cost_recompute=sf.real_transport_cost_recompute)
        for policy in cgr.POLICIES:
            result[f"{label}_{policy}_same_pool"] = _keyed(pools[policy]) == base[policy]
        for plan in ("MILP", "T1_ALL_OR_NOTHING", "VARO_FINAL"):
            result[f"{label}_all_valid_{plan.lower()}_same_totals"] = (
                abs(ev["plans"][plan]["service"] - base_plan[plan]["service"]) <= 1e-9
                and abs(ev["plans"][plan]["cost"] - base_plan[plan]["cost"]) <= 1e-6)
            result[f"{label}_all_valid_{plan.lower()}_same_lanes"] = set(ev["plans"][plan]["lane_keys"]) == set(base_plan[plan]["lane_keys"])
    return result


# ------------------------------------------------------------------------------------------------ aggregation


def by_policy(by_day: pd.DataFrame, cost: pd.DataFrame, runtime: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_cost = by_day[by_day["policy"] == BASELINE].set_index("date")
    for policy, group in by_day.groupby("policy", sort=False):
        group = group.set_index("date")
        milp_cmp = cost[(cost["policy"] == policy) & (cost["plan"] == "MILP")]
        varo_cmp = cost[(cost["policy"] == policy) & (cost["plan"] == "VARO_FINAL")]
        rt = runtime[runtime["policy"] == policy]
        equal = milp_cmp["service_equal"].astype(bool)
        rows.append({
            "policy": policy, "family": cgr.POLICY_SPEC[policy]["family"], "rule": cgr.POLICY_SPEC[policy]["rule"],
            "days": len(group), "pool_total": int(group["pool_size"].sum()), "pool_min": int(group["pool_size"].min()),
            "pool_max": int(group["pool_size"].max()), "pool_invalid_total": int(group["pool_invalid"].sum()),
            "pool_missing_target_cap_total": int(group["pool_missing_target_cap"].sum()),
            "superset_of_legacy_20_days": int(group["superset_of_legacy_20"].sum()),
            "varo_final_service": float(group["varo_final_service"].sum()), "varo_final_cost": float(group["varo_final_cost"].sum()),
            "t1_service": float(group["t1_all_or_nothing_service"].sum()), "t1_cost": float(group["t1_all_or_nothing_cost"].sum()),
            "t1_partial_service": float(group["t1_partial_service"].sum()), "t1_partial_cost": float(group["t1_partial_cost"].sum()),
            "milp_service": float(group["milp_service"].sum()), "milp_cost": float(group["milp_cost"].sum()),
            "greedy_service": float(group["greedy_service"].sum()), "greedy_cost": float(group["greedy_cost"].sum()),
            "milp_cost_delta_vs_legacy_20": round(float(group["milp_cost"].sum() - base_cost["milp_cost"].sum()), 6),
            "milp_cost_delta_pct_vs_legacy_20": round(100.0 * float(group["milp_cost"].sum() - base_cost["milp_cost"].sum())
                                                      / float(base_cost["milp_cost"].sum()), 4),
            "milp_service_equal_days": int(equal.sum()),
            "milp_same_model_comparable_days": int((milp_cmp["pool_comparison_status"] == "SAME_MODEL_DIFFERENT_POOL").sum()),
            "milp_cost_lower_days": int((pd.to_numeric(milp_cmp["cost_delta"], errors="coerce") < -1e-6).sum()),
            "varo_final_cost_lower_days": int((pd.to_numeric(varo_cmp["cost_delta"], errors="coerce") < -1e-6).sum()),
            "superset_monotone_violations": int((milp_cmp["superset_monotone"] == False).sum()),  # noqa: E712
            "violations_total": int(group[[c for c in group.columns if c.endswith("_violations")]].to_numpy().sum()),
            "t1_strict_insufficient_days": int((group["t1_strict_plan_status"] == sf.INSUFFICIENT_CAP_DATA).sum()),
            "milp_optimal_days": int((group["milp_solver_status"] == mbi.OPTIMAL).sum()),
            "milp_enumeration_proven_days": int(group["milp_enumeration_proven"].astype(bool).sum()),
            "milp_scope": "|".join(sorted(set(group["milp_benchmark_scope"]))),
            "varo_final_vs_milp_direct_days": int((group["varo_final_vs_milp"] == mbi.DIRECTLY_COMPARABLE).sum()),
            "varo_final_zero_cost_gap_days": int((pd.to_numeric(group["varo_final_decision_cost_gap"], errors="coerce").abs() <= 1e-6).sum()),
            "t1_zero_cost_gap_days": int((pd.to_numeric(group["t1_decision_cost_gap"], errors="coerce").abs() <= 1e-6).sum()),
            "varo_final_equals_t1_days": int(group["varo_final_equals_t1_routes"].sum()),
            "milp_new_lanes_selected_total": int(group["milp_new_lanes_selected"].sum()),
            "milp_selected_mean_distance_km": round(float(pd.to_numeric(group["milp_selected_mean_distance_km"]).mean()), 3),
            "cov_source_product_mean": round(float(group["cov_source_product_coverage"].mean()), 6),
            "cov_target_product_mean": round(float(group["cov_target_product_coverage"].mean()), 6),
            "cov_valid_lane_mean": round(float(group["cov_valid_lane_coverage"].mean()), 6),
            "cov_cheapest_lane_mean": round(float(group["cov_cheapest_lane_inclusion"].mean()), 6),
            "cov_best_known_plan_mean": round(float(group["cov_best_known_plan_inclusion"].mean()), 6),
            "cov_valid_share_mean": round(float(group["cov_valid_share_benchmark"].mean()), 6),
            "alternative_targets_mean": round(float(pd.to_numeric(group["cov_alternative_targets_per_source_product_mean"]).mean()), 6),
            "generation_ms_total": round(float(rt["generation_ms"].sum()), 3), "pricing_ms_total": round(float(rt["pool_pricing_ms"].sum()), 3),
            "rank_ms_total": round(float(rt["rank_ms"].sum()), 3), "t1_ms_total": round(float(rt["t1_three_modes_ms"].sum()), 3),
            "milp_ms_total": round(float(rt["milp_ms"].sum()), 3), "milp_ms_max": round(float(rt["milp_ms"].max()), 3),
            "total_ms": round(float(rt["total_ms"].sum()), 3), "peak_kb_max": float(rt[["rank_peak_kb", "t1_peak_kb", "milp_peak_kb"]].to_numpy().max()),
        })
    return pd.DataFrame(rows)


def cause_analysis(by_policy_frame: pd.DataFrame, new_routes: pd.DataFrame, universe: pd.DataFrame,
                   by_day: pd.DataFrame) -> dict[str, Any]:
    """What lowers the computed cost, from measured pool differences only (section 26 / 35)."""
    cost = by_policy_frame.set_index("policy")["milp_cost"].to_dict()
    gen_all = universe[universe["generator_rank"].notna()]
    valid = universe[universe["validity_status"] == cgr.VALID]
    corr = []
    for _, group in gen_all.groupby("snapshot_date"):
        if len(group) > 2:
            corr.append(float(group["candidate_score"].astype(float).corr(group["cost_per_unit"].astype(float), method="spearman")))
    milp_new = new_routes[new_routes["plan"] == "MILP"]
    valid_new = milp_new[milp_new["policy"] == cgr.ALL_VALID]
    one_target = valid_new[valid_new["legacy_20_exclusion"] == "CUT_BY_ONE_TARGET_RULE"]
    limit_cut = valid_new[valid_new["legacy_20_exclusion"] == "CUT_BY_LIMIT"]
    lanes_new = milp_new[milp_new["policy"] == cgr.ALL_LANES]
    shorter = one_target.dropna(subset=["generator_kept_target_distance_km"])
    return {
        "milp_cost_by_policy": cost,
        "step_cut_by_score_limit": {"from": cgr.LEGACY_20, "to": cgr.GENERATOR_ALL,
                                    "delta": round(cost[cgr.GENERATOR_ALL] - cost[cgr.LEGACY_20], 6),
                                    "meaning": "lanes the generator keeps but cuts at rank > 20 by candidate_score"},
        "step_one_target_rule": {"from": cgr.GENERATOR_ALL, "to": cgr.ALL_VALID,
                                 "delta": round(cost[cgr.ALL_VALID] - cost[cgr.GENERATOR_ALL], 6),
                                 "meaning": "other targets of the same source-product row (the generator keeps only the "
                                            "largest-need target)"},
        "step_invalid_lanes": {"from": cgr.ALL_VALID, "to": cgr.ALL_LANES,
                               "delta": round(cost[cgr.ALL_LANES] - cost[cgr.ALL_VALID], 6),
                               "meaning": "lanes to a target that does not hold the product: no target cap, the MILP "
                                          "leaves them unconstrained; NOT an improvement"},
        "all_valid_milp_new_lanes": int(len(valid_new)),
        "all_valid_milp_new_lanes_by_legacy_exclusion": {"CUT_BY_ONE_TARGET_RULE": int(len(one_target)),
                                                         "CUT_BY_LIMIT": int(len(limit_cut))},
        "one_target_new_lanes_shorter_than_kept_target": int((shorter["distance_km"].astype(float)
                                                              < shorter["generator_kept_target_distance_km"].astype(float) - 1e-9).sum()),
        "one_target_new_lanes_compared": int(len(shorter)),
        "one_target_mean_distance_new_km": round(float(shorter["distance_km"].astype(float).mean()), 3) if len(shorter) else None,
        "one_target_mean_distance_kept_km": round(float(shorter["generator_kept_target_distance_km"].astype(float).mean()), 3) if len(shorter) else None,
        "one_target_new_lanes_with_lower_need_than_kept": int((shorter["gen_target_need"].astype(float)
                                                               < shorter["generator_kept_target_need"].astype(float) - 1e-9).sum()),
        "score_vs_cost_per_unit_spearman_median": round(float(pd.Series(corr).median()), 4) if corr else None,
        "score_vs_cost_per_unit_spearman_range": [round(min(corr), 4), round(max(corr), 4)] if corr else None,
        "score_has_cost_term_in_real_mode": False,
        "score_distance_term_zero_lanes_share": round(float((universe["distance_score"].fillna(0) <= 0).mean()), 6),
        "all_lanes_milp_new_lanes_target_without_product": int((lanes_new["validity_status"] == "TARGET_PRODUCT_NOT_HELD").sum()),
        "all_lanes_milp_new_lanes": int(len(lanes_new)),
        "route_provenance_difference": sorted(set(valid["route_status"])) + sorted(set(valid["distance_provenance"])),
        "cost_model_difference": "none: every lane priced by the same official-tariff engine at its own quantity",
        "selected_cost_provenance_by_policy": by_day.groupby("policy")["milp_selected_cost_provenance"].agg(
            lambda values: dict(Counter(item for text in values for item in str(text).split("|") if item))).to_dict(),
        "mean_selected_distance_km_by_policy": by_policy_frame.set_index("policy")["milp_selected_mean_distance_km"].to_dict(),
    }


def saturation_analysis(universe: pd.DataFrame, by_day: pd.DataFrame) -> dict[str, Any]:
    """Is 7,750 a ceiling of max_routes x move cap rather than demand coverage? (section 24)"""
    valid = universe[universe["validity_status"] == cgr.VALID]
    need = valid.drop_duplicates(["snapshot_date", "target_id", "product_id"]).groupby("snapshot_date")["target_need_7d"].sum()
    surplus = valid.drop_duplicates(["snapshot_date", "source_id", "product_id"]).groupby("snapshot_date")["source_surplus"].sum()
    milp = by_day[by_day["policy"] == cgr.ALL_VALID].set_index("date")["milp_service"]
    ceiling = cgr.MAX_ROUTES * cg._MOVE_CAP
    return {
        "daily_ceiling": ceiling, "ceiling_rule": f"max_routes {cgr.MAX_ROUTES} x generator move cap {cg._MOVE_CAP}",
        "days_at_ceiling_by_policy": {policy: int((group["milp_service"] >= ceiling - 1e-9).sum())
                                      for policy, group in by_day.groupby("policy")},
        "proxy_target_need_valid_groups_per_day": {"min": round(float(need.min()), 3), "median": round(float(need.median()), 3),
                                                   "max": round(float(need.max()), 3)},
        "proxy_source_surplus_valid_groups_per_day": {"min": round(float(surplus.min()), 3), "median": round(float(surplus.median()), 3)},
        "planned_share_of_proxy_need_median": round(float((milp / need.reindex(milp.index)).median()), 6),
        "valid_lanes_qty_at_move_cap_share": round(float((pd.to_numeric(valid["recommended_qty"]) >= cg._MOVE_CAP).mean()), 6),
        "valid_lanes_limited_by_move_cap_share": round(float(valid["qty_limited_by_move_cap"].astype(bool).mean()), 6),
        "move_cap_origin": "candidate_generator._MOVE_CAP = 50 (CONFIG constant, commit 7d1bf4f 2026-07-15 'Prepare Varo V2 "
                           "submission build', no documented basis); _SHORT_EXPIRY_CAP = 20 when days_to_expiry <= 3",
        "move_cap_is": {"contract_quantity": False, "operational_setting": False, "generator_constant": True,
                        "vehicle_or_pack_unit": False, "evidence": "Suhyup quantity unit is unknown for all 80 products; "
                        "the tariff engine sizes vehicles by kg, not by this cap"},
        "conclusion": "7,750 = 31 days x 250 is the max_routes x move-cap ceiling reached by every pool; it is planned "
                      "movement quantity, not demand coverage",
    }


def _check(check_id: str, description: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"check_id": check_id, "description": description, "status": "PASS" if ok else "FAIL", "detail": detail}


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True).stdout.rstrip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_validation(data_root: Path, output_dir: Path | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    data_root = Path(data_root)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    output_dir = Path(output_dir or data_root / OUTPUT_FOLDER)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_suhyup(data_root)
    peak_after_suhyup = mbv.process_peak_working_set_mb()
    by_day, cost, runtime, new_routes = result["by_day"], result["cost"], result["runtime"], result["new"]
    fidelity, regression, golden, determinism = result["fidelity"], result["regression"], result["golden"], result["determinism"]
    universe, feasibility = result["universe"], result["feasibility"]
    policies = by_policy(by_day, cost, runtime)
    causes = cause_analysis(policies, new_routes, universe, by_day)
    saturation = saturation_analysis(universe, by_day)
    totals = policies.set_index("policy")
    changed = (_git(repo_root, "status", "--porcelain") or "").splitlines()
    modified = [line for line in changed if not line.startswith("??")]
    shuffle_columns = [c for c in determinism.columns if c.startswith("shuffle_")]
    uncut_shuffle = [c for c in shuffle_columns if any(c.endswith(f"_{p}_same_pool") for p in (cgr.MULTI_TARGET, cgr.ALL_VALID, cgr.ALL_LANES))
                     or "_all_valid_" in c]
    legacy_shuffle_days = sorted(determinism.loc[~determinism[[c for c in shuffle_columns if c.endswith(f"_{cgr.LEGACY_20}_same_pool")]].all(axis=1), "date"])
    tie_days = sorted(by_day.loc[(by_day["policy"] == BASELINE) & by_day["generator_cut_boundary_tie"].astype(bool), "date"])
    expected_golden = 186 if "DQN" in set(golden["strategy"]) else 155
    checks = [
        _check("F1", "research LEGACY_20 equals the production generate_candidates output (all columns) 31/31",
               bool(fidelity["legacy_20_equals_production_generator"].all()), int(fidelity["legacy_20_equals_production_generator"].sum())),
        _check("F2", "research LEGACY_20 equals the saved 620 candidates (ids, lanes, qty, tariff cost) 31/31",
               bool(fidelity["legacy_20_equals_saved_candidates"].all()), int(fidelity["legacy_20_equals_saved_candidates"].sum())),
        _check("F3", "research GENERATOR_ALL size equals the production pool before the cut; pool costs equal universe costs",
               bool((fidelity["generator_all_size"] == fidelity["trace_generator_pool_before_cut"]).all()
                    and fidelity["pool_costs_consistent"].all())),
        _check("F4", "LEGACY_20 Varo Final / T1 / MILP totals with research VHS equal the saved pipeline-VHS run 31/31",
               all(all(item.values()) for item in fidelity["research_legacy_totals_equal_saved"])),
        _check("R1", f"reference strategies reproduced ({expected_golden} rows incl. status)",
               bool(golden["equal"].all()) and len(golden) == expected_golden, f"{int(golden['equal'].sum())}/{len(golden)}"),
        _check("R2", "T1 regression: BENCHMARK_PROXY 7,750 / 13,629,146, 0 violations, bundle routes 31/31; STRICT insufficient 31/31",
               bool(regression["t1_matches_bundle"].all() and regression["t1_partial_matches_bundle"].all()
                    and regression["t1_strict_matches_bundle"].all())
               and abs(regression["t1_benchmark_service"].sum() - 7750) <= 1e-9
               and abs(regression["t1_benchmark_cost"].sum() - 13629146) <= 1e-6
               and int(regression["t1_benchmark_violations"].sum()) == 0
               and bool((regression["t1_strict_status"] == sf.INSUFFICIENT_CAP_DATA).all())),
        _check("R3", "T3 regression: MILP OPTIMAL and enumeration-proven 31/31, bundle routes and totals, Varo Final = MILP "
                     "(directly comparable, 0 gap), T1 PARTIAL not comparable",
               bool((regression["t3_milp_status"] == mbi.OPTIMAL).all() and regression["t3_milp_matches_bundle"].all()
                    and regression["t3_varo_routes_match_bundle"].all() and regression["t3_enumeration_proven"].all()
                    and (regression["t3_varo_vs_milp"] == mbi.DIRECTLY_COMPARABLE).all()
                    and regression["t3_varo_zero_gap"].all() and regression["t3_partial_not_comparable"].all())),
        _check("S1", "every policy-day: MILP OPTIMAL (no time limit labelled optimal)",
               bool((by_day["milp_solver_status"] == mbi.OPTIMAL).all()), int((by_day["milp_solver_status"] == mbi.OPTIMAL).sum())),
        _check("S2", "every policy-day: Varo Final, Greedy, T1 (both BENCHMARK plans) and MILP pass the independent check",
               int(by_day[[c for c in by_day.columns if c.endswith("_violations")]].to_numpy().sum()) == 0),
        _check("S3", "STRICT_ACTUAL never selects on PROXY caps (INSUFFICIENT_CAP_DATA every policy-day)",
               bool((by_day["t1_strict_plan_status"] == sf.INSUFFICIENT_CAP_DATA).all())),
        _check("S4", "inside each pool Varo Final and T1 ALL_OR_NOTHING are directly comparable with the MILP (T3 signature)",
               bool((by_day["varo_final_vs_milp"] == mbi.DIRECTLY_COMPARABLE).all() and (by_day["t1_vs_milp"] == mbi.DIRECTLY_COMPARABLE).all())),
        _check("S5", "pools that contain LEGACY_20 never raise the MILP cost at equal service (monotone)",
               int(policies["superset_monotone_violations"].sum()) == 0),
        _check("S6", "service unchanged across policies (7,750; max_routes x move-cap ceiling)",
               bool((policies.set_index("policy")["milp_service"] == totals.loc[BASELINE, "milp_service"]).all())),
        _check("V1", "ALL_VALID pools hold only VALID lanes; ALL_LANES lanes without a target cap are counted and excluded from comparisons",
               int(totals.loc[cgr.ALL_VALID, "pool_invalid_total"]) == 0
               and int(totals.loc[cgr.ALL_LANES, "milp_same_model_comparable_days"]) == 0,
               {"all_lanes_missing_target_cap": int(totals.loc[cgr.ALL_LANES, "pool_missing_target_cap_total"])}),
        _check("V2", "no duplicate lane or candidate id in any pool",
               int(by_day["pool_duplicates"].sum()) == 0 and int(feasibility["duplicate_candidate_ids"].sum()) == 0),
        _check("V3", "every pool lane has a route in the network table (no ROUTE_UNVERIFIED lane in a pool)",
               int(feasibility["route_unverified"].sum()) == 0),
        _check("D1", "repeat run: every pool and the ALL_VALID plans identical 31/31",
               bool(determinism[[c for c in determinism.columns if c.startswith("repeat_")]].all().all())),
        _check("D2", "shuffled inventory rows: uncut pools (MULTI_TARGET, ALL_VALID, ALL_LANES) and ALL_VALID plan totals identical",
               bool(determinism[[c for c in uncut_shuffle if "same_lanes" not in c]].all().all()),
               {"legacy_20_pool_changed_under_shuffle_days": legacy_shuffle_days, "cut_boundary_tie_days": tie_days}),
        _check("G1", PROTECTED_NOTE, not modified, modified),
    ]
    failed = [item["check_id"] for item in checks if item["status"] == "FAIL"]
    base = totals.loc[BASELINE]
    verdicts = _verdicts(totals, failed, causes)
    summary_rows = [("suhyup", "days", int(by_day["date"].nunique())),
                    ("suhyup", "universe_lanes_total", int(by_day[by_day["policy"] == BASELINE]["universe_lanes"].sum())),
                    ("suhyup", "valid_lanes_total", int(by_day[by_day["policy"] == BASELINE]["universe_valid_lanes"].sum()))]
    for policy in cgr.POLICIES:
        row = totals.loc[policy]
        for metric in ("pool_total", "pool_invalid_total", "milp_service", "milp_cost", "milp_cost_delta_pct_vs_legacy_20",
                       "varo_final_cost", "t1_cost", "t1_partial_cost", "greedy_cost", "milp_cost_lower_days",
                       "milp_new_lanes_selected_total", "cov_best_known_plan_mean", "cov_cheapest_lane_mean",
                       "milp_ms_total", "total_ms"):
            summary_rows.append((policy, metric, row[metric]))
    summary_rows += [("runtime", "wall_seconds", round(time.perf_counter() - started, 3)),
                     ("runtime", "process_peak_working_set_mb", mbv.process_peak_working_set_mb()),
                     ("runtime", "environment", f"{platform.platform()} / Python {platform.python_version()} / scipy {scipy.__version__}"),
                     *[("verdict", key, value["status"]) for key, value in verdicts.items()]]
    validation = {
        "research_version": cgr.RESEARCH_VERSION, "git_head": _git(repo_root, "rev-parse", "HEAD"), "data_root": str(data_root),
        "inputs": {"inventory": sfv.INVENTORY, "candidates": sfv.CANDIDATES, "reference": sfv.REFERENCE,
                   "processed_upload": mbv.PROCESSED, "t1_bundle": "_SHARED_FEASIBILITY_VALIDATION/shared_feasibility_by_day.csv",
                   "t3_bundle": f"{mbv.OUTPUT_FOLDER}/milp_benchmark_by_day.csv"},
        "checks": checks, "failed_checks": failed, "verdicts": verdicts,
        "policies": cgr.POLICY_SPEC, "pool_modes_design": cgr.POOL_MODES,
        "internal_vs_displayed": {
            "internal_candidate_count": "candidate_generator.MAX_CANDIDATES (20) bounds the pool the pipeline ranks",
            "displayed_recommendation_count": "analysis_pipeline.top_recommendations(limit=5) / the T1 max_routes (5)",
            "separable": True, "note": "the two are already different numbers in code; a larger internal pool does not "
                                       "change the 5 displayed rows. No UI change was made."},
        "auto_threshold": "not introduced: ALL_VALID runs in milliseconds per day at Suhyup scale, so no size-driven "
                          "cut is needed here; a larger network should set an explicit, reported limit (CONFIG)",
        "generator_structure": {
            "call_path": "analysis_pipeline.build_v2_state -> ensure_recommendations -> candidate_generator.generate_candidates "
                         "(only when the upload has no recommendations sheet) -> run_analysis_pipeline (tariff pricing, "
                         "VHS, varo_final_rank, Top-5, T1 / T2 parallel fields)",
            "cut_1": "candidate_generator.py:250-251 targets.sort(reverse=True); targets[0]: one target per source-product row",
            "cut_2": "candidate_generator.py:311-312 raw.sort(candidate_score desc); raw[:MAX_CANDIDATES]",
            "cut_order": "eligibility -> route -> one-target -> dedupe -> qty -> (saving, legacy mode) -> score -> top-20 -> "
                         "pricing (pipeline). In real-transport mode both cuts run before the tariff cost exists.",
            "dedupe": "key (product, source, target) of the kept target, before the quantity and the cut; only duplicate "
                      "inventory rows of one (store, product) can collide",
            "max_candidates_origin": "MAX_CANDIDATES = 20 and _MOVE_CAP = 50 added in 7d1bf4f (2026-07-15, 'Prepare Varo V2 "
                                     "submission build'); no comment, test or document gives a basis",
            "other_limits_named_20": "optimality_gap_service.DEFAULT_CANDIDATE_LIMIT = 20 (in-app gap pool, cut by Varo rank) "
                                     "and integrated_validation_service candidate_limit 20 are separate settings",
        },
        "one_target_rule": {
            "location": "candidate_generator.generate_candidates targets loop",
            "keeps": "largest generator need = max(0, median - target_stock) + 7 x target demand",
            "tie_break": "real-transport mode: shortest road distance; legacy mode: lowest route estimated_cost; then target id desc",
            "considers_cost": "only as a tie-break; the quantity-specific tariff is not known at this stage",
            "considers_caps": False,
            "considers_proximity": "only as a tie-break",
        },
        "candidate_score": {
            "formula": "0.25 expiry + 0.20 surplus + 0.20 need + 0.20 saving + 0.10 route (DIRECT 100 / VIA_DC 70) + "
                       "0.05 distance (100 - 5 x km, clamped)",
            "real_transport_mode": "saving = 0 for every lane (no price), so the saving term is 0; distance score is 0 at "
                                   ">= 20 km (every Suhyup lane is >= 80 km); expiry is absent (30 -> 7.5 for all). The "
                                   "score is 17.5 + 0.2 surplus + 0.2 need: it holds no cost information",
            "selection_objective": "Varo Final: qty desc, cost asc, vhs_rank, route_id; MILP: max service, then min cost",
            "misalignment": "the cut keeps by surplus/need while the selection, at a saturated 250/day, decides on cost",
        },
        "definitions": {
            "stages_A_G": {"A": "inventory rows at stores x other stores", "B": "lanes of source rows passing the generator's "
                           "eligibility", "C": "B + not self, unique, target holds the product, need evidenced, unit consistent",
                           "D": "C + route in the network table", "E": "D + quantity, cost and both PROXY caps (VALID)",
                           "F": "lanes in the policy pool", "G": "lanes selected by T1 BENCHMARK_PROXY"},
            "coverage": {
                "source_product_coverage": "(source, product) with >= 1 VALID lane in the pool / (source, product) with >= 1 VALID lane in the universe",
                "target_product_coverage": "(target, product) reached by >= 1 VALID pool lane / (target, product) reachable by >= 1 VALID lane",
                "valid_lane_coverage": "VALID lanes in the pool / VALID lanes in the universe",
                "products/sources/targets_without_candidate": "items with >= 1 VALID lane in the universe and no lane in the pool",
                "alternative_targets_per_source_product": "distinct targets per (source, product) present in the pool",
                "cheapest_lane_inclusion": "(source, product) whose lowest cost-per-unit VALID lane is in the pool / (source, product) with VALID lanes",
                "best_known_plan_inclusion": "lanes of the ALL_VALID MILP plan in the pool / lanes of that plan",
                "valid_share_benchmark": "VALID lanes in the pool / lanes in the pool",
                "valid_share_strict_actual": "lanes with strict-accepted source and target caps / lanes in the pool (caps are PROXY: 0)",
                "duplicate_share": "duplicate lane rows in the pool / lanes in the pool"},
            "validity_checks": [{"check": check, "status_if_failed": code or "route_status / unit_status"} for check, code in cgr.VALIDITY_CHECKS],
            "cost_provenance": {"DERIVED_REAL": "official tariff x OSRM road distance x derived unit weight, direct-tariff vehicle classes only",
                                "PROXY": "same, with at least one vehicle class priced from the upper official class "
                                         "(seller_decision_validation rule)",
                                "USER_INPUT": "uploaded route estimated_cost (legacy mode)", "CONFIG": "distance x 100 default",
                                "UNKNOWN": "not computable (kept NULL, never 0)", "DIRECT_REAL": "invoice or contract cost (none exists)",
                                "note": "T3 called all tariff costs DERIVED_REAL and counted proxy vehicles separately; "
                                        "here the proxy vehicle class makes the lane PROXY"},
            "cost_basis": "QUANTITY_SPECIFIC (tariff recomputed at the lane's own quantity); compared plans share it",
            "route_status": {"ROUTE_NETWORK_TABLE": "directed row in the uploaded routes table (Suhyup: OSRM distance between "
                                                    "verified center coordinates)", "ROUTE_VIA_DC_DERIVED": "both DC legs in the table",
                             "ROUTE_UNVERIFIED": "no route row; never in a pool", "ROUTE_FORBIDDEN": "explicit false flag on the row",
                             "operation": "inter-center transfer history is not in the Suhyup data (README): every lane, old "
                                          "and new, is NOT_OBSERVED_IN_INPUT"},
        },
        "cause_analysis": causes, "saturation": saturation, "findings": _findings(by_day, policies, new_routes, result["trace"],
                                                                                   fidelity),
        "determinism": {"legacy_20_pool_changed_under_shuffle_days": legacy_shuffle_days, "cut_boundary_tie_days": tie_days},
        "runtime": {"wall_seconds": round(time.perf_counter() - started, 3), "suhyup_seconds": result["wall_seconds"],
                    "process_peak_working_set_mb_after_31_days": peak_after_suhyup,
                    "process_peak_working_set_mb": mbv.process_peak_working_set_mb(),
                    "note": "peak working set covers the whole validation process (torch for the DQN reference rows, "
                            "tracemalloc while measuring); per-stage peaks are tracemalloc peaks in KB"},
        "limitations": [
            "caps are PROXY (median and outbound rules); STRICT_ACTUAL forms no plan on any policy-day",
            "route operation is unobserved for every lane: the routes table gives OSRM distances, not an operated lane list",
            "quantity unit is unknown for all Suhyup products; no route, vehicle or DC capacity in quantity units exists",
            "costs are computed official-tariff estimates (one way, no tolls or handling), some with a proxy vehicle class; "
            "no invoice exists, so no cost here is an actual expense",
            "service is saturated at 250/day by max_routes x move cap in every pool: the comparison is cost at equal planned "
            "quantity, not more demand served",
            "VHS on research pools is recomputed by the same apply_auto_vhs from the columns the research pool carries; "
            "it is key 3 of the Varo Final order (a tie-break)",
            "the MILP optimum over ALL_VALID holds for that input pool only, not for lanes, vehicles or constraints "
            "outside it",
        ],
        "production_action_applied": False, "production_candidate_pool_replaced": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows, columns=["scope", "metric", "value"]).to_csv(output_dir / OUTPUT_FILES[0], index=False, encoding="utf-8-sig")
    by_day.to_csv(output_dir / OUTPUT_FILES[1], index=False, encoding="utf-8-sig")
    policies.to_csv(output_dir / OUTPUT_FILES[2], index=False, encoding="utf-8-sig")
    result["cutoff"].to_csv(output_dir / OUTPUT_FILES[3], index=False, encoding="utf-8-sig")
    new_routes.to_csv(output_dir / OUTPUT_FILES[4], index=False, encoding="utf-8-sig")
    cost.to_csv(output_dir / OUTPUT_FILES[5], index=False, encoding="utf-8-sig")
    universe_scope = universe.groupby("snapshot_date")["validity_status"].value_counts().unstack(fill_value=0).reset_index()
    universe_scope = universe_scope.rename(columns={"snapshot_date": "date"}).assign(scope="UNIVERSE", policy=None)
    pd.concat([feasibility, universe_scope], ignore_index=True).to_csv(output_dir / OUTPUT_FILES[6], index=False, encoding="utf-8-sig")
    runtime.to_csv(output_dir / OUTPUT_FILES[7], index=False, encoding="utf-8-sig")
    (output_dir / OUTPUT_FILES[8]).write_text(json.dumps(validation, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    result["trace"].to_csv(output_dir / OUTPUT_FILES[9], index=False, encoding="utf-8-sig")
    records = cgr.candidate_records(universe).join(universe[[c for c in universe.columns if c.startswith("pool_")]])
    records.astype({c: str for c in records.columns if records[c].dtype == object}).to_parquet(output_dir / OUTPUT_FILES[10], index=False)
    selections = result["selections"]
    selections.to_parquet(output_dir / OUTPUT_FILES[11], index=False)
    return validation


def _findings(by_day: pd.DataFrame, policies: pd.DataFrame, new_routes: pd.DataFrame, trace: pd.DataFrame,
              fidelity: pd.DataFrame) -> dict[str, Any]:
    """Measured findings reported next to the checks (they are results, not pass/fail gates)."""
    gaps = by_day[pd.to_numeric(by_day["varo_final_decision_cost_gap"], errors="coerce").abs() > 1e-6]
    funnel = trace[trace["table"] == "FUNNEL_A_G"].groupby("stage")["output_count"].sum().to_dict()
    steps = trace[trace["table"] == "PRODUCTION_TRACE"].groupby(["stage", "description"])[["input_count", "output_count"]].sum()
    greedy = by_day[by_day["greedy_service"] < by_day["milp_service"] - 1e-9]
    totals = policies.set_index("policy")
    base = float(totals.loc[BASELINE, "milp_cost"])
    return {
        "milp_cost_lower_than_legacy_20_days": totals["milp_cost_lower_days"].to_dict(),
        "milp_cost_delta_pct_vs_legacy_20": totals["milp_cost_delta_pct_vs_legacy_20"].to_dict(),
        "funnel_totals_31d": {str(key): int(value) for key, value in funnel.items()},
        "production_trace_totals_31d": [{"stage": stage, "description": description, "input": int(row["input_count"]),
                                         "output": int(row["output_count"])}
                                        for (stage, description), row in sorted(steps.iterrows(), key=lambda item: int(item[0][0]))],
        "newly_selected_lanes_not_in_legacy_20_pool": {f"{policy}/{plan}": int(count) for (policy, plan), count
                                                      in new_routes.groupby(["policy", "plan"]).size().items()},
        "newly_selected_distinct_lanes": {policy: int(group["lane_key"].nunique()) for policy, group in new_routes.groupby("policy")},
        "varo_final_vs_milp_nonzero_gap": [{"date": row["date"], "policy": row["policy"],
                                            "decision_cost_gap": row["varo_final_decision_cost_gap"],
                                            "varo_final_route_ids": row["varo_final_route_ids"], "milp_route_ids": row["milp_route_ids"]}
                                           for _, row in gaps.iterrows()],
        "varo_final_vs_milp_explanation": "with several targets per source-product, the greedy Varo Final order takes the "
                                          "cheapest lane first and can block a cheaper pair of lanes through a shared "
                                          "source surplus / target need; the MILP does not (07-01: product 614504, "
                                          "sources 156 and 218020, shared target 157)",
        "greedy_lower_service_days": {policy: int(count) for policy, count in greedy.groupby("policy").size().items()},
        "greedy_note": "Greedy plans less quantity than the MILP on those days; its lower cost is not compared (service first)",
        "legacy_20_invalid_candidates": int(totals.loc[BASELINE, "pool_invalid_total"]),
        "legacy_20_invalid_reason": "TARGET_CAP_MISSING_OR_ZERO: generator need > 0 but benchmark need 0 (the two need formulas)",
        "research_vhs_selection_identical_days": {plan: int(sum(bool(item[plan]) for item in fidelity["research_legacy_selection_equals_saved_pipeline_vhs"]))
                                                  for plan in ("VARO_FINAL", "T1_ALL_OR_NOTHING", "MILP", "GREEDY")},
        "all_valid_vs_legacy_20_cost_pct": round(100.0 * (float(totals.loc[cgr.ALL_VALID, "milp_cost"]) - base) / base, 4),
    }


def _verdicts(totals: pd.DataFrame, failed: Sequence[str], causes: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    base = float(totals.loc[BASELINE, "milp_cost"])
    valid = float(totals.loc[cgr.ALL_VALID, "milp_cost"])
    core_ok = not set(failed) & {"F1", "F2", "F3", "F4", "S1", "S2", "S3", "S4", "S5", "S6", "V1", "V2", "V3", "D1", "D2", "R1", "R2", "R3"}
    return {
        "A_code": {"status": "READY" if not {"F1", "F2", "F3", "G1"} & set(failed) else "NOT_READY",
                   "basis": "research path reproduces the production generator 31/31 and adds seven pool policies as new files"},
        "B_suhyup_31d_research": {"status": "READY" if core_ok else "NOT_READY",
                                  "basis": "31 days x 7 policies, OPTIMAL MILP, 0 violations, T1/T3/186 regressions held"},
        "C_meaningful_improvement": {
            "status": "READY_WITH_LIMITATIONS" if valid < base - 1e-6 and core_ok else "NOT_READY",
            "basis": f"ALL_VALID computed tariff cost {valid:,.0f} vs LEGACY_20 {base:,.0f} at the same 7,750 "
                     f"({100.0 * (valid - base) / base:.1f}%) under PROXY caps; not an actual-spend reduction"},
        "D_new_candidate_validity": {
            "status": "NOT_READY",
            "basis": "new lanes pass every check the current 20 pass (network-table route, target holds the product, "
                     "PROXY need > 0, tariff cost), but actual executability is unverified for old and new lanes alike: "
                     "no operated-lane record, PROXY caps (STRICT_ACTUAL insufficient), unknown quantity unit, no capacity"},
        "E_production_replacement": {
            "status": "NOT_READY",
            "basis": "D is not sufficient, so the production pool is not replaced regardless of the computed cost; the "
                     "research modes stay parallel until operated-lane and demand evidence exist and the user approves"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_validation(args.data_root, args.output_dir)
    print(json.dumps({"failed_checks": result["failed_checks"], "checks": result["checks"], "verdicts": result["verdicts"]},
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

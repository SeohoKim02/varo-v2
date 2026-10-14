"""Read-only Suhyup integration replay, non-overwriting local-only reports."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from unittest.mock import patch
import uuid

import pandas as pd

from services import final_decision_integration as fdi
from services import inventory_metric_integrity_validation as t4
from services import milp_benchmark_integrity_validation as mbv
from services import shared_feasibility_validation as sfv
from services.release_validation import protected_hashes, metadata_inventory, load, MANIFEST
from services.release_regression import json_safe
from services.seller_decision_validation import run_scenarios

OUTPUT_FILES = ("final_decision_integration_summary.json", "final_decision_by_product.csv", "final_decision_by_route.csv",
    "final_decision_input_requirements.csv", "final_decision_comparison_status.csv", "final_decision_explanations.csv",
    "final_decision_integration_validation.json")


def flat(rows):
    return [{k: json.dumps(json_safe(v), ensure_ascii=False, sort_keys=True, default=str) if isinstance(v, (dict, list, tuple)) else v
             for k, v in json_safe(row).items()} for row in rows]


def run_validation(root, output_dir=None):
    root = Path(root).resolve()
    output = Path(output_dir).resolve() if output_dir else root / ("_FINAL_DECISION_INTEGRATION_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    if output.parent != root or not output.name.startswith("_FINAL_DECISION_INTEGRATION_"):
        raise ValueError("new direct-child _FINAL_DECISION_INTEGRATION_* report directory required")
    output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    before, meta_before = protected_hashes(root, load(MANIFEST)), metadata_inventory(root, output)
    rebuild = mbv.GeneratorRebuild(root)
    saved = sfv.load_suhyup_candidates(root)
    products, routes, daily, input_rows, explanations, status_rows = [], [], [], [], [], []
    checks = []
    for day in sorted(saved.snapshot_date.unique()):
        print("Integration replay:", day, flush=True)
        upload = t4.declared_day_upload(rebuild, day)
        # Counterfactual disables only the new parallel layer. Same code/data/settings.
        with patch("services.analysis_pipeline.safe_build_final_decision_integration", return_value={}):
            baseline = t4._pipeline(upload, root)["pipeline_result"]
        state = t4._pipeline(upload, root)
        current = dict(state["pipeline_result"])
        integrated = current.pop("final_decision_integration")
        baseline.pop("final_decision_integration")
        old_equal = fdi.digest(baseline) == fdi.digest(current)
        catalog = fdi.product_catalog(upload)
        coverage = fdi.coverage_check(catalog, integrated["products"])
        n_recommended = sum(bool(p.get("recommended_strategy")) for p in integrated["products"])
        explanation_errors = []
        for p in integrated["products"]:
            if p["explanation_ko"] != fdi.explain(p): explanation_errors.append(p["decision_id"])
            if p["decision_mode"] == sfv.sf.BENCHMARK_PROXY and p["selected_transfer_routes"] and "추정 조건" not in p["explanation_ko"]:
                explanation_errors.append(p["decision_id"])
            if p.get("actual_saving") is not None or p.get("outcome") is not None: explanation_errors.append(p["decision_id"])
        day_rows = [{"date": day, **r} for r in integrated["products"]]
        route_data = [{"date": day, **r} for r in integrated["routes"]]
        products.extend(day_rows)
        routes.extend(route_data)
        for p in day_rows:
            for item in p["missing_required_inputs"]:
                input_rows.append({"date": day, "decision_id": p["decision_id"], "product_id": p["product_id"], "location_id": p["location_id"], **item})
            explanations.append({k: p[k] for k in ("date", "decision_id", "product_id", "location_id", "decision_mode", "decision_status", "explanation_ko")})
            status_rows.append({k: p[k] for k in ("date", "decision_id", "product_id", "location_id", "seller_loss_status", "promotion_status", "recommendation_readiness", "recommended_strategy")})
        checks.append({"date": day, "old_production_identical": old_equal, "coverage": coverage,
                       "integration_status": integrated["status"], "minimum_loss_recommendations": n_recommended,
                       "explanation_errors": len(set(explanation_errors)), "orphan_routes": len(integrated["orphan_routes"])})
        daily.append({"date": day, **integrated["summary"], "selected_allocated_qty_in_native_scale": sum(r["allocated_qty"] for r in integrated["routes"] if r["selected"]),
                      "quantity_unit_verified": all(bool(r["quantity_unit"]) for r in integrated["routes"] if r["selected"]),
                      "old_production_identical": old_equal, "explanation_errors": len(set(explanation_errors))})
    # Existing independent baselines stay separate from current production-plan quantities.
    golden = t4.golden_186(root)
    t1 = t4.t1_rerun(root)
    archived = t4.earlier_bundles(root)
    seller, _ = run_scenarios()
    after, meta_after = protected_hashes(root, load(MANIFEST)), metadata_inventory(root, output)
    protections = before == after and meta_before == meta_after
    results = {
        "suhyup_31_days": len(daily) == 31,
        "all_products_included": all(c["coverage"]["pass"] for c in checks),
        "old_production_unchanged": all(c["old_production_identical"] for c in checks),
        "integration_not_error": all(c["integration_status"] == "parallel_only" for c in checks),
        "no_unsupported_minimum_loss_claim": all(c["minimum_loss_recommendations"] == 0 for c in checks),
        "explanation_consistent": all(c["explanation_errors"] == 0 for c in checks),
        "candidate_links_complete": all(c["orphan_routes"] == 0 for c in checks),
        "golden_186": len(golden) == 186 and bool(golden.equal.all()),
        "t1_baseline": t1["benchmark_service"] == 7750 and t1["benchmark_cost"] == 13629146 and t1["benchmark_violations"] == 0,
        "seller_18": len(seller) == 18 and bool(seller["pass"].all()),
        "frozen_baselines": all(not archived[k]["failed_checks"] for k in ("T1", "T2", "T3", "P1", "P1B", "P1C")),
        "external_FAIL_preserved": archived["FORECAST"]["external_verdict"] == "FAIL",
        "protected_files_unchanged": protections,
    }
    summary = {"version": fdi.VERSION, "days": daily, "total_product_location_rows": len(products),
        "unique_products": len({p["product_id"] for p in products}), "total_candidates": len(routes),
        "selected_plans": sum(r["selected"] for r in routes), "proxy_plans": sum(r["selected"] and r["plan_mode"] == "BENCHMARK_PROXY" for r in routes),
        "strict_actual_plans": sum(r["selected"] and r["plan_mode"] == "STRICT_ACTUAL" for r in routes),
        "minimum_loss_recommendations": sum(bool(p["recommended_strategy"]) for p in products),
        "seller_comparable_product_locations": sum(bool(p["available_strategies"]) for p in products),
        "additional_input_products": sum(bool(p["missing_required_inputs"]) for p in products),
        "production_action_applied": False, "old_production_unchanged": results["old_production_unchanged"],
        "missing_product_rows": sum(len(c["coverage"]["missing"]) for c in checks),
        "duplicate_product_rows": sum(len(c["coverage"]["duplicates"]) for c in checks),
        "explanation_errors": sum(c["explanation_errors"] for c in checks), "golden_equal": int(golden.equal.sum()),
        "t1_independent_benchmark": t1, "archived_baselines": archived,
        "protected_content_files": len(before), "protected_metadata_files": len(meta_before),
        "seconds": round(time.perf_counter() - tick, 3), "peak_memory_mb": None,
        "readiness": {"contract": "READY_WITH_LIMITATIONS", "parallel_connection": "READY" if results["old_production_unchanged"] else "BLOCKED",
            "product_coverage": "READY" if results["all_products_included"] else "BLOCKED", "evidence_separation": "READY",
            "actual_seller_validation": "NOT_READY", "production_replacement": "BLOCKED", "UI_contract": "READY_WITH_LIMITATIONS"},
        "limitations": ["no actual retail sales/monetary outcomes", "no production replacement",
            "no-route products require explicit decision quantity/context; no quantity is invented",
            "pipeline need=max(7-day demand-stock,0); offline benchmark need=max(median stock+7*outbound-stock,0); different cap contracts, not one plan",
            "M5/external/P1 full research benchmarks not rerun here; frozen results only"]}
    validation = {"checks": {k: bool(v) for k, v in results.items()}, "failed_checks": [k for k,v in results.items() if not v], "by_day": checks}
    for value, name in ((summary, OUTPUT_FILES[0]), (validation, OUTPUT_FILES[-1])):
        (output / name).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    for rows, name in zip((products, routes, input_rows, status_rows, explanations), OUTPUT_FILES[1:6]):
        pd.DataFrame(flat(rows)).to_csv(output / name, index=False, encoding="utf-8-sig")
    return {"output_dir": str(output), "failed_checks": validation["failed_checks"], "summary": {k:v for k,v in summary.items() if k not in ("days", "archived_baselines")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\VARO_V2_REAL_DATA"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run_validation(args.data_root, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["failed_checks"] else 0)


if __name__ == "__main__": main()

"""Reproducible whole-plan benchmark and four-scale timing measurement.

Compares independent candidate totals, constrained Greedy, and the VHS-based
execution plan.  It also regenerates the existing anonymized workbook at the
four documented sizes and records candidate, evaluation, plan, validation, and
total analysis time.

    python tools/run_execution_plan_benchmark.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.analysis_pipeline import run_analysis_pipeline  # noqa: E402
from services.execution_plan import (  # noqa: E402
    PLAN_ALGORITHM_VERSION,
    build_execution_plan,
    compare_execution_plans,
)
from services.file_reader import read_uploaded_data  # noqa: E402
from services.partial_data import build_usable_data  # noqa: E402
from tools import generate_anonymized_operational_workbook as generator  # noqa: E402


REPORT_NAME = "varo_v2_execution_plan_benchmark.json"
SCALES = (
    (20, 20, 16),
    (40, 30, 24),
    (70, 45, 32),
    (100, 60, 40),
)


def _candidate(
    route_id: str, source: str, target: str, *, qty: int = 40,
    saving: float = 8_000, cost: float = 1_000, vhs: float = 80,
    route_type: str = "DIRECT", dc_id: str | None = None, rank: int = 1,
) -> dict[str, Any]:
    return {
        "route_id": route_id, "source_id": source, "source_name": source,
        "target_id": target, "target_name": target,
        "product_id": "P1", "product_name": "P1",
        "route_type": route_type, "dc_id": dc_id, "dc_name": dc_id,
        "recommended_qty": qty, "expected_saving": saving,
        "estimated_cost": cost, "net_benefit": saving - cost,
        "vhs_score": vhs, "vhs_rank": rank, "varo_final_rank": rank,
        "greedy_rank": rank, "robustness_status": "안정",
        "confidence_score": 80, "pareto_status": "비지배",
    }


def _workbook(
    recs: list[dict[str, Any]], source_stock: dict[str, int], target_gap: dict[str, int],
    safety: dict[str, int] | None = None,
) -> dict[str, pd.DataFrame]:
    safety = safety or {}
    inventory = [
        {"store_id": source, "product_id": "P1", "stock_qty": stock,
         "safety_stock": safety.get(source, 0), "demand_qty": 0}
        for source, stock in source_stock.items()
    ]
    inventory += [
        {"store_id": target, "product_id": "P1", "stock_qty": 0,
         "safety_stock": 0, "target_stock": gap}
        for target, gap in target_gap.items()
    ]
    dcs = {str(rec.get("dc_id")) for rec in recs if rec.get("dc_id")}
    node_ids = set(source_stock) | set(target_gap) | dcs
    stores = pd.DataFrame([
        {"store_id": node, "node_type": "DC" if node in dcs else "STORE"}
        for node in sorted(node_ids)
    ])
    routes = []
    seen = set()
    for rec in recs:
        key = (rec["source_id"], rec["target_id"], rec["route_type"], rec.get("dc_id"))
        if key in seen:
            continue
        seen.add(key)
        routes.append({
            "source_id": rec["source_id"], "target_id": rec["target_id"],
            "route_type": rec["route_type"], "dc_id": rec.get("dc_id"),
        })
    return {
        "stores": stores,
        "products": pd.DataFrame([{"product_id": "P1"}]),
        "inventory": pd.DataFrame(inventory),
        "routes": pd.DataFrame(routes),
    }


def _scenarios() -> list[tuple[str, str, list[dict[str, Any]], dict[str, pd.DataFrame]]]:
    one_to_many = [
        _candidate("R1", "S", "T1", qty=40, saving=10_000, rank=1),
        _candidate("R2", "S", "T2", qty=40, saving=8_000, rank=2),
        _candidate("R3", "S", "T3", qty=40, saving=6_000, rank=3),
    ]
    many_to_one = [
        _candidate("R1", "S1", "T", qty=40, saving=10_000, rank=1),
        _candidate("R2", "S2", "T", qty=40, saving=8_000, rank=2),
        _candidate("R3", "S3", "T", qty=40, saving=6_000, rank=3),
    ]
    source_value_tradeoff = [
        _candidate("R1", "S", "T1", qty=50, saving=4_000, cost=1_000, vhs=90, rank=1),
        _candidate("R2", "S", "T2", qty=50, saving=15_000, cost=1_000, vhs=70, rank=2),
    ]
    direct = _candidate("RD", "S", "T", saving=10_000, cost=1_000, route_type="DIRECT")
    via = _candidate("RV", "S", "T", saving=8_000, cost=2_000, route_type="VIA_DC", dc_id="DC01", rank=2)
    dc01 = _candidate("RDC1", "S", "T", saving=8_000, cost=2_000, route_type="VIA_DC", dc_id="DC01")
    dc02 = _candidate("RDC2", "S", "T", saving=10_000, cost=1_000, route_type="VIA_DC", dc_id="DC02", rank=2)
    expensive = [
        _candidate("R1", "S", "T1", saving=10_000, cost=9_000),
        _candidate("R2", "S", "T2", saving=8_000, cost=7_500, rank=2),
    ]
    return [
        ("supply_shortage", "공급 부족", one_to_many, _workbook(one_to_many, {"S": 60}, {"T1": 40, "T2": 40, "T3": 40})),
        ("supply_surplus", "공급 과다", one_to_many, _workbook(one_to_many, {"S": 300}, {"T1": 20, "T2": 20, "T3": 20})),
        ("source_concentration", "한 출발점 과잉 집중", source_value_tradeoff, _workbook(source_value_tradeoff, {"S": 50}, {"T1": 50, "T2": 50})),
        ("target_concentration", "한 도착점 부족 집중", many_to_one, _workbook(many_to_one, {"S1": 50, "S2": 50, "S3": 50}, {"T": 60})),
        ("multi_supply_single_demand", "다중 공급·단일 수요", many_to_one, _workbook(many_to_one, {"S1": 40, "S2": 40, "S3": 40}, {"T": 50})),
        ("single_supply_multi_demand", "단일 공급·다중 수요", one_to_many, _workbook(one_to_many, {"S": 50}, {"T1": 40, "T2": 40, "T3": 40})),
        ("direct_favourable", "DIRECT 우세", [direct, via], _workbook([direct, via], {"S": 100}, {"T": 40})),
        ("via_dc_favourable", "VIA_DC 우세", [direct, {**via, "expected_saving": 13_000, "estimated_cost": 1_000, "net_benefit": 12_000}], _workbook([direct, via], {"S": 100}, {"T": 40})),
        ("multi_dc", "다중 DC", [dc01, dc02], _workbook([dc01, dc02], {"S": 100}, {"T": 40})),
        ("high_transport_cost", "높은 운송비", expensive, _workbook(expensive, {"S": 60}, {"T1": 40, "T2": 40})),
        ("high_safety_stock", "안전재고 높음", one_to_many, _workbook(one_to_many, {"S": 100}, {"T1": 40, "T2": 40, "T3": 40}, {"S": 90})),
    ]


def run_plan_scenarios() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, label, recs, data in _scenarios():
        optimized = build_execution_plan(recs, data, data_signature=key)
        greedy = build_execution_plan(recs, data, data_signature=key, strategy="greedy")
        output[key] = {
            "label": label,
            "optimized_status": optimized["plan_status"],
            "optimized_valid": optimized["validation"]["valid"],
            "optimized_routes": [row["route_id"] for row in optimized["items"]],
            "comparison": compare_execution_plans(optimized, greedy, recs, data),
        }
    return output


def _usable_workbook(path: Path) -> dict[str, Any]:
    data, report = read_uploaded_data(path, path.name, return_report=True)
    metadata = {
        "filename": path.name, "source_type": "excel",
        "sheet_names": dict(report["raw_sheet_names"]),
    }
    return build_usable_data(data, report["raw_sheets"], metadata)["usable_data"]


def _measure_scale(store_count: int, product_count: int, products_per_store: int) -> dict[str, Any]:
    original = (
        generator.STORE_COUNT, generator.PRODUCT_COUNT,
        generator.PRODUCTS_PER_STORE, generator.MAX_VALID_RECOMMENDATIONS,
    )
    try:
        generator.STORE_COUNT = store_count
        generator.PRODUCT_COUNT = product_count
        generator.PRODUCTS_PER_STORE = products_per_store
        generator.MAX_VALID_RECOMMENDATIONS = 60
        with tempfile.TemporaryDirectory(prefix="varo_plan_scale_") as temp_name:
            workbook, manifest_path = generator.generate(Path(temp_name))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            data = _usable_workbook(workbook)
            events: dict[str, float] = {}
            started = time.perf_counter()

            def callback(event) -> None:
                events[event.stage] = time.perf_counter() - started

            result = run_analysis_pipeline(data, data_signature=f"scale-{store_count}", progress_callback=callback)
            total = time.perf_counter() - started
            plan = result.execution_plan
            candidate_start = events.get("candidates", 0.0)
            scoring_start = events.get("scoring", candidate_start)
            verification_start = events.get("verification", scoring_start)
            return {
                "stores": store_count,
                "products": product_count,
                "source_rows": manifest["scale"]["total_source_rows"],
                "applied_rows": manifest["expected"]["applied_rows"],
                "final_candidates": len(result.recommendations),
                "planned_moves": plan.get("selected_candidates"),
                "candidate_generation_seconds": round(scoring_start - candidate_start, 4),
                "candidate_evaluation_seconds": round(verification_start - scoring_start, 4),
                "plan_optimization_seconds": plan.get("optimization_seconds"),
                "plan_validation_seconds": plan.get("validation_seconds"),
                "total_analysis_seconds": round(total, 4),
                "plan_valid": (plan.get("validation") or {}).get("valid"),
                "fallback_used": plan.get("fallback_used"),
            }
    finally:
        (
            generator.STORE_COUNT, generator.PRODUCT_COUNT,
            generator.PRODUCTS_PER_STORE, generator.MAX_VALID_RECOMMENDATIONS,
        ) = original


def run(output_dir: Path) -> tuple[int, dict[str, Any]]:
    scenarios = run_plan_scenarios()
    scale_performance = [_measure_scale(*scale) for scale in SCALES]

    operational_path = output_dir / generator.WORKBOOK_NAME
    if not operational_path.is_file():
        generator.generate(output_dir)
    operational_data = _usable_workbook(operational_path)
    operational = run_analysis_pipeline(operational_data, data_signature="operational-plan-benchmark")

    report = {
        "algorithm_version": PLAN_ALGORITHM_VERSION,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "scale_performance": scale_performance,
        "operational_plan": {
            "plan_status": operational.execution_plan.get("plan_status"),
            "selected_candidates": operational.execution_plan.get("selected_candidates"),
            "total_transfer_qty": operational.execution_plan.get("total_transfer_qty"),
            "total_cost": operational.execution_plan.get("total_cost"),
            "total_expected_saving": operational.execution_plan.get("total_expected_saving"),
            "total_net_benefit": operational.execution_plan.get("total_net_benefit"),
            "validation": operational.execution_plan.get("validation"),
            "comparison": operational.plan_comparison,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    failed = any(not row["optimized_valid"] for row in scenarios.values())
    failed = failed or any(not row["plan_valid"] for row in scale_performance)
    failed = failed or not operational.execution_plan.get("validation", {}).get("valid")
    return (1 if failed else 0), report


def main() -> int:
    code, report = run(generator.OUTPUT_DIR)
    print(f"실행계획 시나리오: {report['scenario_count']}개")
    for key, row in report["scenarios"].items():
        comp = row["comparison"]
        independent = comp["independent_candidates"]
        optimized = comp["vhs_optimized_plan"]
        print(
            f"  {key:27s} 독립 충돌 "
            f"{independent['safety_stock_violations'] + independent['destination_overfill_violations']} "
            f"→ 계획 충돌 {optimized['safety_stock_violations'] + optimized['destination_overfill_violations']} "
            f"· 순효과 {optimized['total_net_benefit']:,.0f}"
        )
    print("규모별 분석:")
    for row in report["scale_performance"]:
        print(
            f"  {row['source_rows']:5d}행 · 계획 {row['plan_optimization_seconds']:.4f}s "
            f"· 검증 {row['plan_validation_seconds']:.4f}s · 전체 {row['total_analysis_seconds']:.2f}s"
        )
    print(f"보고서: {generator.OUTPUT_DIR / REPORT_NAME}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

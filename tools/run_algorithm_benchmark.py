"""Reproducible algorithm benchmark: scenarios, ablation and stress.

This is a measurement tool, not a product feature — nothing here is rendered in
the app. It answers the three questions a technical reviewer asks:

1. **Scenarios** — across operating conditions (volatile demand, expensive
   transport, sparse routes, DC-only paths, concentrated surplus, many shortage
   stores), how does the VHS ranking compare with a single-objective Greedy
   baseline on real quantities: net benefit, move cost, shortage covered, surplus
   relieved, Pareto status, robustness?
2. **Ablation** — if one VHS component is switched off, what changes? This is the
   evidence for "why is each component there".
3. **Stress** — do extreme or degenerate inputs break the scoring, produce NaN,
   or make the ranking depend on row order?

Every scenario is a *uniform, documented transformation* of the anonymized
operational workbook. Inputs are never hand-tuned to make a method look good, and
whatever comes out is written down.

    python tools/generate_anonymized_operational_workbook.py
    python tools/run_algorithm_benchmark.py

Writes ``validation_data/varo_v2_algorithm_benchmark.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.analysis_pipeline import run_analysis_pipeline, sort_recommendations  # noqa: E402
from services.decision_metrics import ALGORITHM_VERSION  # noqa: E402
from services.feasibility import annotate_feasibility  # noqa: E402
from services.file_reader import read_uploaded_data  # noqa: E402
from services.greedy_baseline import compare_to_vhs, greedy_ranking  # noqa: E402
from services.partial_data import build_usable_data  # noqa: E402
from services.vhs_score_engine import (  # noqa: E402
    COMPONENTS, COMPONENT_LABELS, _ranks_from_scores, _weighted_scores, apply_auto_vhs,
)
from tools.generate_anonymized_operational_workbook import (  # noqa: E402
    OUTPUT_DIR, WORKBOOK_NAME, generate,
)

REPORT_NAME = "varo_v2_algorithm_benchmark.json"
TOP_K = 5


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# --------------------------------------------------------------------------- #
# Scenarios — uniform transformations of the same anonymized workbook
# --------------------------------------------------------------------------- #
def _copy(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.copy() if isinstance(value, pd.DataFrame) else value for key, value in data.items()}


def _scale(frame: pd.DataFrame, column: str, factor: float) -> None:
    if column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce") * factor


def _prune_unsupported_recommendations(data: dict[str, Any]) -> dict[str, Any]:
    """Drop candidates whose route no longer exists after a route transformation.

    Without this a "DC 경유만 가능" scenario would still carry DIRECT candidates
    that have no edge behind them, and the scenario would not test anything.
    """
    routes, recommendations = data["routes"], data["recommendations"]
    edges = {
        (str(row["source_id"]).strip(), str(row["target_id"]).strip())
        for _, row in routes.iterrows()
    }
    keep: list[bool] = []
    for _, row in recommendations.iterrows():
        source, target = str(row["source_id"]).strip(), str(row["target_id"]).strip()
        dc_id = "" if pd.isna(row.get("dc_id")) else str(row.get("dc_id")).strip()
        if str(row.get("route_type")).upper() == "VIA_DC":
            keep.append(bool(dc_id) and (source, dc_id) in edges and (dc_id, target) in edges)
        else:
            keep.append((source, target) in edges)
    data["recommendations"] = recommendations[pd.Series(keep, index=recommendations.index)].reset_index(drop=True)
    return data


def _base(data: Mapping[str, Any]) -> dict[str, Any]:
    return _copy(data)


def _demand_volatile(data: Mapping[str, Any]) -> dict[str, Any]:
    """수요 변동이 큰 상황: 측정된 표준편차를 일괄 3배로 넓힌다."""
    result = _copy(data)
    _scale(result["inventory"], "demand_std", 3.0)
    return result


def _high_transport_cost(data: Mapping[str, Any]) -> dict[str, Any]:
    """이동 비용이 높은 상황: 모든 경로 비용을 일괄 6배로 올린다."""
    result = _copy(data)
    for column in ("estimated_cost", "transport_cost"):
        _scale(result["routes"], column, 6.0)
    for column in ("estimated_cost", "transport_cost"):
        _scale(result["recommendations"], column, 6.0)
    return result


def _sparse_routes(data: Mapping[str, Any]) -> dict[str, Any]:
    """경로 제한이 많은 상황: DIRECT 경로의 3분의 2를 일정 간격으로 제거한다."""
    result = _copy(data)
    routes = result["routes"].reset_index(drop=True)
    direct = routes["route_type"].astype(str).str.upper() == "DIRECT"
    drop = direct & (routes.index % 3 != 0)
    result["routes"] = routes[~drop].reset_index(drop=True)
    return _prune_unsupported_recommendations(result)


def _dc_favourable(data: Mapping[str, Any]) -> dict[str, Any]:
    """DC 경유가 유리한 상황: 점포 간 DIRECT 경로를 모두 제거한다."""
    result = _copy(data)
    routes = result["routes"]
    result["routes"] = routes[routes["route_type"].astype(str).str.upper() != "DIRECT"].reset_index(drop=True)
    return _prune_unsupported_recommendations(result)


def _direct_favourable(data: Mapping[str, Any]) -> dict[str, Any]:
    """DIRECT가 유리한 상황: DC 경유 구간을 모두 제거한다."""
    result = _copy(data)
    routes = result["routes"]
    result["routes"] = routes[routes["route_type"].astype(str).str.upper() == "DIRECT"].reset_index(drop=True)
    return _prune_unsupported_recommendations(result)


def _surplus_concentrated(data: Mapping[str, Any]) -> dict[str, Any]:
    """과잉 재고가 일부 점포에 몰린 상황."""
    result = _copy(data)
    inventory = result["inventory"].reset_index(drop=True)
    stock = pd.to_numeric(inventory["stock_qty"], errors="coerce")
    heavy = inventory.index % 4 == 0
    inventory["stock_qty"] = stock.where(~heavy, stock * 4.0).where(heavy, stock * 0.4)
    result["inventory"] = inventory
    return result


def _many_shortage_stores(data: Mapping[str, Any]) -> dict[str, Any]:
    """부족 점포가 다수인 상황: 공급 점포를 제외한 재고를 일괄로 낮춘다."""
    result = _copy(data)
    inventory = result["inventory"].reset_index(drop=True)
    stock = pd.to_numeric(inventory["stock_qty"], errors="coerce")
    supplier = inventory.index % 4 == 0
    inventory["stock_qty"] = stock.where(supplier, stock * 0.3)
    result["inventory"] = inventory
    return result


SCENARIOS: tuple[tuple[str, str, Callable[[Mapping[str, Any]], dict[str, Any]]], ...] = (
    ("base", "정상 수요 (익명 운영 데이터 원본)", _base),
    ("demand_volatile", "수요 변동 큼 (표준편차 ×3)", _demand_volatile),
    ("high_transport_cost", "이동 비용 높음 (경로 비용 ×6)", _high_transport_cost),
    ("sparse_routes", "경로 제한 많음 (DIRECT 경로 2/3 제거)", _sparse_routes),
    ("dc_favourable", "DC 경유만 가능", _dc_favourable),
    ("direct_favourable", "DIRECT만 가능", _direct_favourable),
    ("surplus_concentrated", "과잉 재고 집중", _surplus_concentrated),
    ("many_shortage_stores", "부족 점포 다수", _many_shortage_stores),
)


def _scenario_metrics(result: Any, seconds: float) -> dict[str, Any]:
    recommendations = list(result.recommendations)
    ledger = result.ledger_summary or {}
    feasibility = result.feasibility_summary or {}
    comparison = compare_to_vhs(recommendations, top_k=TOP_K)
    ordered = sort_recommendations(recommendations)
    top = ordered[0] if ordered else {}
    generated = int(ledger.get("generated") or 0)
    return {
        "status": result.status,
        "candidate_count": generated,
        "recommendable": len(recommendations),
        "feasible_ratio": round(len(recommendations) / generated, 4) if generated else None,
        "blocked": int(feasibility.get("blocked_count") or 0),
        "blocked_reasons": list(feasibility.get("blocked_reasons") or []),
        "route_type_mix": {
            kind: sum(1 for row in recommendations if str(row.get("route_type")) == kind)
            for kind in ("DIRECT", "VIA_DC")
        },
        "dc_mix": {
            dc: sum(1 for row in recommendations if str(row.get("dc_id")) == dc)
            for dc in sorted({str(row.get("dc_id")) for row in recommendations if row.get("dc_id")})
        },
        "top1": {
            "route_id": top.get("route_id"),
            "route_type": top.get("route_type"),
            "dc_id": top.get("dc_id"),
            "recommended_qty": top.get("recommended_qty"),
            "net_benefit": top.get("net_benefit"),
            "vhs_score": top.get("vhs_score"),
            "robustness": top.get("robustness_status"),
            "pareto_status": top.get("pareto_status"),
            "demand_scenario": top.get("demand_scenario_status"),
        },
        "pareto_front_ratio": (result.pareto_analysis or {}).get("front_ratio"),
        "stability": (result.stability_analysis_status or {}).get("status"),
        "confidence": (result.confidence_status or {}).get("status"),
        "comparison": comparison,
        "seconds": round(seconds, 2),
    }


def run_scenarios(usable: Mapping[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for key, label, transform in SCENARIOS:
        data = transform(usable)
        start = time.perf_counter()
        result = run_analysis_pipeline(data)
        metrics = _scenario_metrics(result, time.perf_counter() - start)
        metrics["label"] = label
        results[key] = metrics
    return results


# --------------------------------------------------------------------------- #
# Ablation — one component switched off, everything else identical
# --------------------------------------------------------------------------- #
def _selection_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def total(field: str) -> float | None:
        values = [_num(row.get(field)) for row in rows]
        usable = [value for value in values if value is not None]
        return round(sum(usable), 2) if usable else None

    covered = 0.0
    for row in rows:
        quantity, need = _num(row.get("recommended_qty")), _num(row.get("target_shortfall"))
        if quantity is not None and need is not None:
            covered += min(quantity, max(0.0, need))
    return {
        "top1_route_id": str(rows[0].get("route_id")) if rows else None,
        "net_benefit_total": total("net_benefit"),
        "move_cost_total": total("estimated_cost"),
        "shortage_covered": round(covered, 2),
        "top1_robustness": rows[0].get("robustness_status") if rows else None,
    }


def run_ablation(recommendations: list[Mapping[str, Any]], weights: Mapping[str, float]) -> dict[str, Any]:
    """Re-rank with each component's weight forced to zero (others renormalized).

    Reuses the component scores the candidates already carry, so nothing is
    recomputed and the only difference is the weighting.
    """
    recs = [dict(row) for row in recommendations]
    active = [component for component in COMPONENTS if float(weights.get(component, 0.0)) > 0]
    if len(recs) < 2 or not active:
        return {"comparable": False, "rows": []}

    base_ranks = _ranks_from_scores(_weighted_scores(recs, weights))
    base_order = [recs[i] for i in sorted(range(len(recs)), key=lambda i: base_ranks[i])]
    baseline = _selection_metrics(base_order[:TOP_K])

    rows: list[dict[str, Any]] = []
    for component in active:
        reduced = {key: (0.0 if key == component else value) for key, value in weights.items()}
        total = sum(reduced.values())
        if total <= 0:
            continue
        reduced = {key: value / total for key, value in reduced.items()}
        ranks = _ranks_from_scores(_weighted_scores(recs, reduced))
        order = [recs[i] for i in sorted(range(len(recs)), key=lambda i: ranks[i])]
        metrics = _selection_metrics(order[:TOP_K])
        moved = sum(1 for index in range(len(recs)) if ranks[index] != base_ranks[index])
        rows.append({
            "component": component,
            "label": COMPONENT_LABELS.get(component, component),
            "removed_weight": round(float(weights[component]), 4),
            "top1_changed": metrics["top1_route_id"] != baseline["top1_route_id"],
            "top1_route_id": metrics["top1_route_id"],
            "candidates_reordered": moved,
            "net_benefit_delta": (
                None if metrics["net_benefit_total"] is None or baseline["net_benefit_total"] is None
                else round(metrics["net_benefit_total"] - baseline["net_benefit_total"], 2)
            ),
            "move_cost_delta": (
                None if metrics["move_cost_total"] is None or baseline["move_cost_total"] is None
                else round(metrics["move_cost_total"] - baseline["move_cost_total"], 2)
            ),
            "shortage_covered_delta": round(metrics["shortage_covered"] - baseline["shortage_covered"], 2),
            "top1_robustness": metrics["top1_robustness"],
        })
    return {"comparable": True, "baseline": baseline, "rows": rows}


# --------------------------------------------------------------------------- #
# Stress — degenerate and extreme inputs
# --------------------------------------------------------------------------- #
def _candidate(route_id: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "route_id": route_id, "product_id": "P1", "product_name": "가상상품 01",
        "source_id": "S1", "source_name": "가상점포 01",
        "target_id": "S2", "target_name": "가상점포 02",
        "route_type": "DIRECT", "dc_id": None, "dc_name": None,
        "recommended_qty": 10, "transport_type": "일반 탑차",
        "estimated_cost": 1000, "expected_saving": 20000,
        "distance_km": 5.0, "travel_time_min": 15.0,
        "vhs_score": 60.0, "recommendation_grade": "보통",
        "confidence_score": 60.0, "reason": "-",
    }
    base.update(overrides)
    return base


def _stress_cases() -> list[tuple[str, list[dict[str, Any]]]]:
    huge, tiny = 10**9, 1e-6
    return [
        ("후보 0건", []),
        ("후보 1건", [_candidate("A")]),
        ("동일 점수 후보 다수", [_candidate(f"S{i}") for i in range(1, 9)]),
        ("매우 큰 재고·수량", [_candidate("A", recommended_qty=huge), _candidate("B")]),
        ("매우 작은 수량", [_candidate("A", recommended_qty=tiny), _candidate("B")]),
        ("이동 비용 0", [_candidate("A", estimated_cost=0), _candidate("B")]),
        ("이동 비용 매우 큼", [_candidate("A", estimated_cost=huge), _candidate("B")]),
        ("이동 시간 0", [_candidate("A", travel_time_min=0, distance_km=0), _candidate("B")]),
        ("이동 시간 매우 김", [_candidate("A", travel_time_min=huge), _candidate("B")]),
        ("모든 순효과 음수", [_candidate("A", expected_saving=1, estimated_cost=99999),
                              _candidate("B", expected_saving=2, estimated_cost=88888)]),
        ("절감액 계산 불가", [_candidate("A", expected_saving=None), _candidate("B")]),
        ("DIRECT만 존재", [_candidate("A"), _candidate("B")]),
        ("VIA_DC만 존재", [_candidate("A", route_type="VIA_DC", dc_id="DC01", dc_name="가상물류센터 1"),
                            _candidate("B", route_type="VIA_DC", dc_id="DC01", dc_name="가상물류센터 1")]),
        ("DC 여러 개", [_candidate("A", route_type="VIA_DC", dc_id="DC01", dc_name="가상물류센터 1"),
                        _candidate("B", route_type="VIA_DC", dc_id="DC02", dc_name="가상물류센터 2")]),
        ("후보 300건", [_candidate(f"R{i:04d}", expected_saving=1000 + i * 7,
                                   estimated_cost=100 + (i % 37)) for i in range(300)]),
    ]


def run_stress() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, candidates in _stress_cases():
        frame = pd.DataFrame(candidates)
        entry: dict[str, Any] = {"case": name, "candidates": len(candidates)}
        try:
            first = apply_auto_vhs(frame.copy())
            second = apply_auto_vhs(frame.copy().iloc[::-1].reset_index(drop=True))
            if first.frame.empty:
                entry.update({"ok": True, "note": "후보 없음 · 빈 결과", "nan_free": True, "deterministic": True})
            else:
                scores = pd.to_numeric(first.frame["vhs_score"], errors="coerce")
                nan_free = bool(scores.notna().all()) and bool(scores.between(0, 100).all())
                order_a = list(first.frame.sort_values("vhs_rank")["route_id"].astype(str))
                order_b = list(second.frame.sort_values("vhs_rank")["route_id"].astype(str))
                blocked = annotate_feasibility(candidates)["summary"]
                entry.update({
                    "ok": True,
                    "nan_free": nan_free,
                    # Reversing the input rows must not change the ranking.
                    "deterministic": order_a == order_b,
                    "top1": order_a[0],
                    "blocked": blocked["blocked_count"],
                    "feasible": blocked["feasible_count"],
                })
        except Exception as exc:  # noqa: BLE001 - the point of the check
            entry.update({"ok": False, "error": type(exc).__name__})
        rows.append(entry)
    return {
        "cases": len(rows),
        "failures": [row for row in rows if not row.get("ok")],
        "non_deterministic": [row["case"] for row in rows if row.get("deterministic") is False],
        "nan_producing": [row["case"] for row in rows if row.get("nan_free") is False],
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(output_dir: Path, regenerate: bool) -> tuple[int, dict[str, Any]]:
    workbook = output_dir / WORKBOOK_NAME
    if regenerate or not workbook.is_file():
        generate(output_dir)
    data, report = read_uploaded_data(workbook, workbook.name, return_report=True)
    metadata = {
        "filename": workbook.name, "source_type": "excel",
        "sheet_names": dict(report["raw_sheet_names"]),
    }
    usable = build_usable_data(data, report["raw_sheets"], metadata)["usable_data"]

    scenarios = run_scenarios(usable)
    base_result = run_analysis_pipeline(usable)
    ablation = run_ablation(base_result.recommendations, base_result.vhs_analysis.get("weights") or {})
    stress = run_stress()

    report_data = {
        "algorithm_version": ALGORITHM_VERSION,
        "workbook": str(workbook),
        "top_k": TOP_K,
        "weights": base_result.vhs_analysis.get("weights"),
        "scenarios": scenarios,
        "ablation": ablation,
        "stress": stress,
    }
    (output_dir / REPORT_NAME).write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    failed = bool(stress["failures"]) or bool(stress["non_deterministic"]) or bool(stress["nan_producing"])
    failed = failed or any(item["status"] not in ("success", "partial") for item in scenarios.values())
    return (1 if failed else 0), report_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    code, report = run(args.output_dir, args.regenerate)

    print(f"알고리즘 버전: {report['algorithm_version']}")
    print("\n[시나리오]")
    for key, item in report["scenarios"].items():
        comparison = item["comparison"]
        match = "일치" if comparison.get("top1_match") else "불일치"
        print(
            f"  {key:22s} 후보 {item['candidate_count']:3d} → 추천 {item['recommendable']:3d} "
            f"| 1순위 {item['top1']['route_type'] or '-':7s} {item['top1']['robustness'] or '-':7s} "
            f"| Greedy Top1 {match} · Top{report['top_k']} 교집합 {comparison.get('topk_overlap', 0)} "
            f"| {item['seconds']:.1f}s"
        )
    print("\n[Ablation · 구성요소 제거 시 변화]")
    for row in report["ablation"].get("rows", []):
        print(
            f"  {row['label']:10s} 가중치 {row['removed_weight']:.3f} 제거 → "
            f"1순위 {'변경' if row['top1_changed'] else '유지'} · "
            f"순효과 {row['net_benefit_delta']:+,.0f} · 부족 충족 {row['shortage_covered_delta']:+,.0f}"
        )
    stress = report["stress"]
    print(f"\n[Stress] {stress['cases']}개 케이스 · 실패 {len(stress['failures'])} · "
          f"비결정 {len(stress['non_deterministic'])} · NaN {len(stress['nan_producing'])}")
    print(f"\n보고서: {args.output_dir / REPORT_NAME}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

"""Revalidate Varo algorithms on the saved Suhyup July 2026 candidate universe.

This module is intentionally UI-free.  It reads the immutable source and
benchmark exports, reruns the current ranking/training code, and writes a new
validation bundle without replacing any existing artifact.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from services import dqn_service
from services.dqn_service import ACTION_LABELS, TRANSFER_ACTIONS
from services.legacy_adapters._local_modules.heuristic_optimizer import add_heuristic_scores
from services.pareto_service import ParetoSelectionResult, select_pareto_routes
from services.vhs_score_engine import (
    COMPONENTS,
    apply_auto_vhs,
    calculate_weighted_vhs_scores,
    rank_vhs_scores,
)

NETWORK_NODES = ("130030", "156", "157", "190010", "200330", "218020")
MAX_DAILY_ROUTES = 5
REQUIRED_OUTPUTS = (
    "daily_strategy_summary.csv",
    "aggregate_strategy_summary.csv",
    "dqn_seed_summary.csv",
    "dqn_action_distribution.csv",
    "vhs_sensitivity_summary.csv",
    "milp_comparison_summary.csv",
    "validation_notes.json",
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.where(pd.notna(frame), None).to_dict("records")


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _inventory_lookup(inventory: pd.DataFrame) -> tuple[dict[tuple[str, str, str], dict[str, float]], dict[tuple[str, str], float]]:
    data = inventory.copy()
    data["date"] = data["date"].astype(str)
    data["center_code"] = data["center_code"].astype(str)
    data["product_code"] = data["product_code"].astype(str)
    for column in ("stock_qty", "outbound_qty"):
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    data = data[data["center_code"].isin(NETWORK_NODES)]
    grouped = data.groupby(["date", "center_code", "product_code"], as_index=False).agg(
        stock_qty=("stock_qty", "sum"),
        outbound_qty=("outbound_qty", "sum"),
    )
    lookup = {
        (str(row.date), str(row.center_code), str(row.product_code)): {
            "stock": float(row.stock_qty), "outbound": float(row.outbound_qty),
        }
        for row in grouped.itertuples(index=False)
    }
    medians = (
        grouped.groupby(["date", "product_code"])["stock_qty"]
        .median()
        .to_dict()
    )
    return lookup, {(str(key[0]), str(key[1])): float(value) for key, value in medians.items()}


def enrich_inventory_constraints(candidates: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    lookup, medians = _inventory_lookup(inventory)
    rows: list[dict[str, Any]] = []
    for item in _records(candidates):
        date = str(item.get("snapshot_date") or "")
        product = str(item.get("product_id") or "")
        source = str(item.get("source_id") or "")
        target = str(item.get("target_id") or "")
        source_record = lookup.get((date, source, product))
        target_record = lookup.get((date, target, product))
        median_stock = medians.get((date, product))
        row = dict(item)
        row["inventory_source_matched"] = source_record is not None
        row["inventory_target_matched"] = target_record is not None
        row["source_stock"] = source_record["stock"] if source_record else None
        row["target_stock"] = target_record["stock"] if target_record else None
        row["target_daily_demand_proxy"] = target_record["outbound"] if target_record else None
        row["median_stock"] = median_stock
        row["source_surplus"] = (
            max(0.0, source_record["stock"] - median_stock)
            if source_record is not None and median_stock is not None else None
        )
        row["target_need_7d"] = (
            max(0.0, median_stock + 7.0 * target_record["outbound"] - target_record["stock"])
            if target_record is not None and median_stock is not None else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _selection_violation(rows: Sequence[Mapping[str, Any]]) -> str | None:
    if len(rows) > MAX_DAILY_ROUTES:
        return "max_routes"
    duplicate_keys: set[tuple[str, ...]] = set()
    source_usage: defaultdict[tuple[str, str], float] = defaultdict(float)
    target_usage: defaultdict[tuple[str, str], float] = defaultdict(float)
    source_caps: dict[tuple[str, str], float] = {}
    target_caps: dict[tuple[str, str], float] = {}
    for row in rows:
        qty = _number(row.get("recommended_qty"), float("nan"))
        cost = _number(row.get("move_cost") or row.get("estimated_cost"), float("nan"))
        if not math.isfinite(qty) or qty <= 0:
            return "invalid_quantity"
        if not math.isfinite(cost) or cost < 0:
            return "invalid_cost"
        key = (
            str(row.get("product_id") or ""), str(row.get("source_id") or ""),
            str(row.get("target_id") or ""), str(row.get("route_type") or "DIRECT"),
            str(row.get("dc_id") or ""),
        )
        if key in duplicate_keys:
            return "duplicate_allocation"
        duplicate_keys.add(key)
        source_key = (key[1], key[0])
        target_key = (key[2], key[0])
        source_cap = row.get("source_surplus")
        target_cap = row.get("target_need_7d")
        if source_cap is not None and math.isfinite(_number(source_cap, float("nan"))):
            source_caps[source_key] = max(0.0, _number(source_cap))
        if target_cap is not None and math.isfinite(_number(target_cap, float("nan"))):
            target_caps[target_key] = max(0.0, _number(target_cap))
        source_usage[source_key] += qty
        target_usage[target_key] += qty
    if any(value > source_caps[key] + 1e-8 for key, value in source_usage.items() if key in source_caps):
        return "source_inventory"
    if any(value > target_caps[key] + 1e-8 for key, value in target_usage.items() if key in target_caps):
        return "target_demand"
    return None


def ordered_feasible_selection(frame: pd.DataFrame, order_columns: Sequence[str], ascending: Sequence[bool]) -> pd.DataFrame:
    ordered = frame.sort_values(list(order_columns), ascending=list(ascending), kind="mergesort", na_position="last")
    selected: list[dict[str, Any]] = []
    for row in _records(ordered):
        if len(selected) >= MAX_DAILY_ROUTES:
            break
        if _selection_violation([*selected, row]) is None:
            selected.append(row)
    return pd.DataFrame(selected)


def lexicographic_milp(frame: pd.DataFrame) -> dict[str, Any]:
    rows = _records(frame.reset_index(drop=True))
    n = len(rows)
    if not n:
        return {"stage1_status": None, "stage2_status": None, "selected": pd.DataFrame(), "optimal": False}
    qty = np.asarray([_number(row.get("recommended_qty")) for row in rows], dtype=float)
    cost = np.asarray([_number(row.get("move_cost") or row.get("estimated_cost")) for row in rows], dtype=float)
    constraint_rows: list[list[float]] = [[1.0] * n]
    lower = [-np.inf]
    upper = [float(MAX_DAILY_ROUTES)]
    groups: dict[tuple[str, tuple[str, str]], list[int]] = defaultdict(list)
    duplicate_groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        product = str(row.get("product_id") or "")
        source = str(row.get("source_id") or "")
        target = str(row.get("target_id") or "")
        groups[("source", (source, product))].append(index)
        groups[("target", (target, product))].append(index)
        duplicate_groups[(product, source, target, str(row.get("route_type") or "DIRECT"), str(row.get("dc_id") or ""))].append(index)
    for (kind, _), indices in sorted(groups.items(), key=str):
        cap_name = "source_surplus" if kind == "source" else "target_need_7d"
        caps = [_number(rows[index].get(cap_name), float("nan")) for index in indices]
        finite = [value for value in caps if math.isfinite(value)]
        if not finite:
            continue
        vector = [0.0] * n
        for index in indices:
            vector[index] = qty[index]
        constraint_rows.append(vector)
        lower.append(-np.inf)
        upper.append(max(0.0, finite[0]))
    for indices in duplicate_groups.values():
        if len(indices) <= 1:
            continue
        vector = [0.0] * n
        for index in indices:
            vector[index] = 1.0
        constraint_rows.append(vector)
        lower.append(-np.inf)
        upper.append(1.0)
    matrix = np.asarray(constraint_rows, dtype=float)
    constraints = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    bounds = Bounds(np.zeros(n), np.ones(n))
    integrality = np.ones(n)
    stage1 = milp(c=-qty, integrality=integrality, bounds=bounds, constraints=constraints, options={"presolve": True})
    if stage1.status != 0 or stage1.x is None:
        return {"stage1_status": int(stage1.status), "stage2_status": None, "selected": pd.DataFrame(), "optimal": False}
    maximum_service = round(float(qty @ np.rint(stage1.x)), 8)
    service_row = qty.reshape(1, -1)
    stage2_constraints = [constraints, LinearConstraint(service_row, maximum_service - 1e-7, np.inf)]
    route_order = {value: rank for rank, value in enumerate(sorted(str(row.get("route_id") or "") for row in rows), start=1)}
    deterministic_cost = cost + np.asarray([route_order[str(row.get("route_id") or "")] * 1e-7 for row in rows])
    stage2 = milp(c=deterministic_cost, integrality=integrality, bounds=bounds, constraints=stage2_constraints, options={"presolve": True})
    selected_indices = [index for index, value in enumerate(stage2.x if stage2.x is not None else []) if value > 0.5]
    selected = frame.reset_index(drop=True).iloc[selected_indices].copy()
    return {
        "stage1_status": int(stage1.status), "stage2_status": int(stage2.status),
        "selected": selected, "optimal": bool(stage1.status == 0 and stage2.status == 0),
    }


def _strategy_row(
    date: str,
    strategy: str,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    milp_service: float,
    status: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    service = float(pd.to_numeric(selected.get("recommended_qty", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    cost_values = pd.to_numeric(selected.get("move_cost", pd.Series(dtype=float)), errors="coerce").fillna(0)
    saving_values = pd.to_numeric(selected.get("expected_saving", pd.Series(dtype=float)), errors="coerce").fillna(0)
    violation = _selection_violation(_records(selected)) if not selected.empty else None
    result = {
        "date": date,
        "strategy": strategy,
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
        "service_qty": round(service, 6),
        "service_level": round(service / milp_service, 6) if milp_service > 0 else None,
        "total_cost": round(float(cost_values.sum()), 6),
        "expected_saving": round(float(saving_values.sum()), 6),
        "feasibility_violations": 0 if violation is None else 1,
        "status": status if violation is None else f"{status};{violation}",
        "route_ids": "|".join(selected.get("route_id", pd.Series(dtype=str)).astype(str).tolist()),
    }
    result.update(dict(extra or {}))
    return result


def pareto_operational_selection(frame: pd.DataFrame) -> tuple[pd.DataFrame, ParetoSelectionResult]:
    """Select at most five routes with repeated Pareto-frontier recomputation."""
    records = _records(frame.reset_index(drop=True))

    def feasible(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, str | None]:
        violation = _selection_violation(rows)
        return violation is None, violation

    result = select_pareto_routes(
        records,
        max_routes=MAX_DAILY_ROUTES,
        feasibility_check=feasible,
    )
    selected = frame.reset_index(drop=True).iloc[result.selected_indices].copy()
    return selected, result


def _perturbed_weights(weights: Mapping[str, float], component: str, factor: float) -> dict[str, float]:
    changed = {key: max(0.0, float(value)) for key, value in weights.items()}
    changed[component] *= factor
    total = sum(changed.values()) or 1.0
    return {key: value / total for key, value in changed.items()}


def vhs_sensitivity_rows(frame: pd.DataFrame, base_weights: Mapping[str, float], date: str) -> list[dict[str, Any]]:
    base_top = frame.sort_values(["vhs_rank", "route_id"], kind="mergesort").head(MAX_DAILY_ROUTES)
    base_ids = set(base_top["route_id"].astype(str))
    base_rank = pd.to_numeric(frame.set_index("route_id")["vhs_rank"], errors="coerce")
    results: list[dict[str, Any]] = []
    active = [component for component in COMPONENTS if float(base_weights.get(component, 0.0)) > 0]
    for delta in (0.05, 0.10, 0.20):
        for direction in (-1, 1):
            for component in active:
                weights = _perturbed_weights(base_weights, component, 1.0 + direction * delta)
                scores = calculate_weighted_vhs_scores(frame, weights)
                ranks = rank_vhs_scores(scores)
                working = frame.assign(_sensitivity_rank=ranks)
                top = working.sort_values(["_sensitivity_rank", "route_id"], kind="mergesort").head(MAX_DAILY_ROUTES)
                ids = set(top["route_id"].astype(str))
                rank_series = pd.Series(ranks.values, index=frame["route_id"].astype(str))
                results.append({
                    "date": date, "delta_pct": int(delta * 100),
                    "direction": "increase" if direction > 0 else "decrease",
                    "component": component,
                    "top_k_overlap": len(base_ids & ids) / max(1, len(base_ids | ids)),
                    "service_qty": float(pd.to_numeric(top["recommended_qty"], errors="coerce").fillna(0).sum()),
                    "total_cost": float(pd.to_numeric(top["move_cost"], errors="coerce").fillna(0).sum()),
                    "rank_correlation": float(base_rank.corr(rank_series, method="spearman")),
                })
    return results


def _training_summary(result: dqn_service.DqnTrainingResult, stage: str) -> dict[str, Any]:
    losses = [value for value in result.loss_history if value is not None and math.isfinite(float(value))]
    diagnostics = dict(result.diagnostics or {})
    holdout = dict(diagnostics.get("holdout_metrics") or {})
    loaded = dqn_service.infer_dqn_actions(
        [], result.data_signature, model_path=result.model_path,
    ) if not result.model_path else None
    return {
        "stage": stage, "seed": result.seed, "episodes": result.episodes,
        "optimizer_steps": diagnostics.get("optimizer_steps"),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "train_mean_reward": result.reward_summary.get("avg"),
        "holdout_mean_reward": holdout.get("mean_reward"),
        "holdout_median_reward": holdout.get("median_reward"),
        "random_policy_reward": holdout.get("random_mean_reward"),
        "reward_optimal_mean": holdout.get("reward_optimal_mean"),
        "dominant_action": diagnostics.get("action_concentration", {}).get("dominant_action"),
        "dominant_action_ratio": diagnostics.get("action_concentration", {}).get("dominant_ratio"),
        "invalid_action_count": holdout.get("feasibility_violations"),
        "service_kpi": holdout.get("mean_service_score"),
        "cost_kpi": holdout.get("transport_cost"),
        "expected_saving": holdout.get("expected_saving"),
        "selected_quantity": holdout.get("throughput"),
        "parameters_changed": diagnostics.get("parameters_changed"),
        "model_path": result.model_path,
        "status": result.status,
        "time_order_column": diagnostics.get("time_order_column"),
        "train_candidate_count": diagnostics.get("train_candidate_count"),
        "holdout_candidate_count": diagnostics.get("holdout_candidate_count"),
        "save_load_equal": None if loaded is not None else False,
    }


def _train_stage(recommendations: Sequence[Mapping[str, Any]], episodes: int, seed: int, stage: str) -> tuple[dqn_service.DqnTrainingResult, dict[str, Any]]:
    result = dqn_service.train_dqn(
        recommendations, episodes=episodes, learning_rate=0.001, seed=seed,
        sample_id="suhyup_202607_31d", training_mode="original",
        store_count=len(NETWORK_NODES), dc_count=0,
    )
    summary = _training_summary(result, stage)
    inferred = dqn_service.infer_dqn_actions(
        recommendations, result.data_signature, model_path=result.model_path,
    )
    summary["save_load_equal"] = bool(
        inferred.model_status == "loaded"
        and inferred.dqn_action_by_route == result.dqn_action_by_route
        and inferred.q_value_summary_by_route == result.q_value_summary_by_route
    )
    summary["inference_source"] = inferred.diagnostics.get("inference_source")
    summary["finite_q_values"] = all(
        math.isfinite(float(value))
        for item in inferred.q_value_summary_by_route.values()
        for value in item.values() if value is not None
    )
    return result, summary


def _baseline_metrics(saved: pd.DataFrame) -> dict[str, Any]:
    service_loss = saved["varo_served_qty"] < saved["milp_served_qty"] - 1e-9
    equal_service = (saved["varo_served_qty"] - saved["milp_served_qty"]).abs() <= 1e-9
    cost_defeat = equal_service & (saved["varo_transport_cost"] > saved["milp_transport_cost"] + 1e-9)
    return {
        "dates": int(len(saved)),
        "varo_service": float(saved["varo_served_qty"].sum()),
        "greedy_service": float(saved["greedy_served_qty"].sum()),
        "milp_service": float(saved["milp_served_qty"].sum()),
        "varo_cost": float(saved["varo_transport_cost"].sum()),
        "greedy_cost": float(saved["greedy_transport_cost"].sum()),
        "milp_cost": float(saved["milp_transport_cost"].sum()),
        "service_loss_days": int(service_loss.sum()),
        "service_loss_units": float((saved.loc[service_loss, "milp_served_qty"] - saved.loc[service_loss, "varo_served_qty"]).sum()),
        "equal_service_cost_defeat_days": int(cost_defeat.sum()),
    }


def aggregate_strategy_summary(daily: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    """Build the common 31-day aggregate without changing strategy semantics."""
    aggregate_rows: list[dict[str, Any]] = []
    for strategy, group in daily.groupby("strategy", sort=True):
        strategy_comparison = comparison[comparison["strategy"] == strategy]
        aggregate_rows.append({
            "strategy": strategy,
            "total_service": float(group["service_qty"].sum()),
            "mean_service": float(group["service_qty"].mean()),
            "total_cost": float(group["total_cost"].sum()),
            "mean_cost": float(group["total_cost"].mean()),
            "total_saving": float(group["expected_saving"].sum()),
            "feasibility_violation_count": int(group["feasibility_violations"].sum()),
            "days_compared": int(group["date"].nunique()),
            "exact_comparable_days": int((strategy_comparison["comparison_label"] == "동일 서비스 비용 비교").sum()),
        })
    return pd.DataFrame(aggregate_rows)


def refresh_pareto_validation(data_root: Path, output_dir: Path) -> dict[str, Any]:
    """Refresh only Pareto rows in an existing validation bundle.

    DQN models/results and every non-Pareto daily row are read-only in this
    path, so the final Pareto closeout does not trigger model retraining.
    """
    source_root = data_root / "16_VARO_E2E_20260731"
    recommendation_path = source_root / "multi_snapshot_validation" / "suhyup_202607_multi_snapshot_recommendations.csv"
    inventory_path = data_root / "02_Korea_Suhyup_Logistics" / "processed" / "suhyup_logistics_inventory_flow_actual.csv"
    required = [
        recommendation_path,
        inventory_path,
        output_dir / REQUIRED_OUTPUTS[0],
        output_dir / REQUIRED_OUTPUTS[1],
        output_dir / REQUIRED_OUTPUTS[5],
        output_dir / REQUIRED_OUTPUTS[6],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Pareto refresh files: {missing}")

    candidates = pd.read_csv(
        recommendation_path,
        dtype={"snapshot_date": str, "route_id": str, "product_id": str, "source_id": str, "target_id": str},
    )
    inventory = pd.read_csv(
        inventory_path,
        dtype={"date": str, "center_code": str, "product_code": str},
    )
    candidates = _numeric(candidates, [
        "recommended_qty", "move_cost", "estimated_cost", "expected_saving", "distance_km",
        "expected_time_min", "travel_time_min", "feasibility_score", "demand_fit_score",
        "disposal_risk_score", "inventory_balance_score", "promotion_score", "route_cost_score",
    ])
    candidates = enrich_inventory_constraints(candidates, inventory)
    daily = pd.read_csv(output_dir / REQUIRED_OUTPUTS[0], dtype={"date": str})
    comparison = pd.read_csv(output_dir / REQUIRED_OUTPUTS[5], dtype={"date": str})
    protected_before = daily[daily["strategy"].isin(["Varo Final", "MILP"])].groupby("strategy")[["service_qty", "total_cost"]].sum()
    daily = daily[daily["strategy"] != "Pareto"].copy()
    comparison = comparison[comparison["strategy"] != "Pareto"].copy()

    pareto_daily_rows: list[dict[str, Any]] = []
    pareto_comparison_rows: list[dict[str, Any]] = []
    active_objectives: set[str] = set()
    inactive_objectives: set[str] = set()
    for date in sorted(candidates["snapshot_date"].astype(str).unique()):
        day = candidates[candidates["snapshot_date"] == date].copy()
        day["final_recommendation"] = "재고 이동"
        day = apply_auto_vhs(add_heuristic_scores(day)).frame
        selected, pareto = pareto_operational_selection(day)
        active_objectives.update(pareto.active_objectives)
        inactive_objectives.update(pareto.inactive_objectives)
        milp_daily = daily[(daily["date"] == date) & (daily["strategy"] == "MILP")]
        if milp_daily.empty:
            raise ValueError(f"Missing MILP daily row for {date}")
        milp_service = float(milp_daily.iloc[0]["service_qty"])
        milp_cost = float(milp_daily.iloc[0]["total_cost"])
        pareto_daily_rows.append(_strategy_row(
            date,
            "Pareto",
            day,
            selected,
            milp_service,
            "feasible",
            {
                "frontier_size": pareto.trace[0]["frontier_size"] if pareto.trace else 0,
                "frontier_recalculation_steps": len(pareto.trace),
                "active_objectives": "|".join(pareto.active_objectives),
                "inactive_objectives": "|".join(pareto.inactive_objectives),
            },
        ))
        service = float(pd.to_numeric(selected.get("recommended_qty", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        cost = float(pd.to_numeric(selected.get("move_cost", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        equal_service = abs(service - milp_service) <= 1e-8
        pareto_comparison_rows.append({
            "date": date,
            "strategy": "Pareto",
            "milp_solver_status": "optimal",
            "service_qty": service,
            "milp_service_qty": milp_service,
            "service_gap_units": milp_service - service,
            "cost": cost,
            "milp_cost": milp_cost,
            "comparison_label": "동일 서비스 비용 비교" if equal_service else "서비스 비동등 참고 Gap",
            "cost_gap_at_equal_service": cost - milp_cost if equal_service else None,
        })

    daily = pd.concat([daily, pd.DataFrame(pareto_daily_rows)], ignore_index=True).sort_values(["date", "strategy"], kind="mergesort")
    comparison = pd.concat([comparison, pd.DataFrame(pareto_comparison_rows)], ignore_index=True).sort_values(["date", "strategy"], kind="mergesort")
    protected_after = daily[daily["strategy"].isin(["Varo Final", "MILP"])].groupby("strategy")[["service_qty", "total_cost"]].sum()
    if not protected_before.equals(protected_after):
        raise AssertionError("Varo Final or MILP aggregate changed during Pareto-only refresh")
    aggregate = aggregate_strategy_summary(daily, comparison)

    notes_path = output_dir / REQUIRED_OUTPUTS[6]
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    notes.setdefault("selection_contract", {})["pareto_selection"] = (
        "feasible pool -> frontier -> normalized ideal-point distance -> select one -> "
        "update shared constraints -> recompute; max 5"
    )
    notes["pareto"] = {
        "status": "production_strategy",
        "active_objectives": sorted(active_objectives),
        "inactive_objectives": sorted(inactive_objectives),
        "frontier_recalculated_after_each_selection": True,
        "normalization": "per-frontier min-max, direction-aware",
        "compromise_rule": "minimum Euclidean distance to ideal point",
        "refresh_scope": "Pareto only; DQN was not retrained",
    }
    notes["limitations"] = [
        item for item in notes.get("limitations", [])
        if "Pareto has no production single-winner rule" not in str(item)
    ]
    optional_note = "expected_saving is disabled as a Pareto objective whenever it is missing or has zero variance"
    if optional_note not in notes["limitations"]:
        notes["limitations"].append(optional_note)

    daily.to_csv(output_dir / REQUIRED_OUTPUTS[0], index=False, encoding="utf-8-sig")
    aggregate.to_csv(output_dir / REQUIRED_OUTPUTS[1], index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / REQUIRED_OUTPUTS[5], index=False, encoding="utf-8-sig")
    notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    pareto_aggregate = aggregate[aggregate["strategy"] == "Pareto"].iloc[0].to_dict()
    return {
        "output_dir": str(output_dir),
        "pareto": pareto_aggregate,
        "varo_milp_unchanged": True,
        "active_objectives": sorted(active_objectives),
        "inactive_objectives": sorted(inactive_objectives),
    }


def run_validation(data_root: Path, output_dir: Path, production_episodes: int = 300) -> dict[str, Any]:
    source_root = data_root / "16_VARO_E2E_20260731"
    paths = {
        "recommendations": source_root / "multi_snapshot_validation" / "suhyup_202607_multi_snapshot_recommendations.csv",
        "snapshot_summary": source_root / "multi_snapshot_validation" / "suhyup_202607_multi_snapshot_summary.csv",
        "saved_milp": source_root / "milp_benchmark" / "multi_snapshot" / "suhyup_202607_operational_milp_daily_summary.csv",
        "saved_milp_routes": source_root / "milp_benchmark" / "multi_snapshot" / "suhyup_202607_operational_milp_selected_routes.csv",
        "actual_inventory": data_root / "02_Korea_Suhyup_Logistics" / "processed" / "suhyup_logistics_inventory_flow_actual.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required actual-data files: {missing}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    candidates = pd.read_csv(paths["recommendations"], dtype={"snapshot_date": str, "route_id": str, "product_id": str, "source_id": str, "target_id": str})
    inventory = pd.read_csv(paths["actual_inventory"], dtype={"date": str, "center_code": str, "product_code": str})
    snapshot_summary = pd.read_csv(paths["snapshot_summary"], dtype={"snapshot_date": str})
    saved_milp = _numeric(pd.read_csv(paths["saved_milp"], dtype={"snapshot_date": str}), [
        "milp_served_qty", "milp_transport_cost", "varo_served_qty", "varo_transport_cost",
        "greedy_served_qty", "greedy_transport_cost",
    ])
    candidates = _numeric(candidates, [
        "recommended_qty", "move_cost", "estimated_cost", "expected_saving", "distance_km",
        "expected_time_min", "travel_time_min", "feasibility_score", "demand_fit_score",
        "disposal_risk_score", "inventory_balance_score", "promotion_score", "route_cost_score",
    ])
    candidates = enrich_inventory_constraints(candidates, inventory)
    dates = sorted(candidates["snapshot_date"].astype(str).unique())

    quality = {
        "date_min": dates[0], "date_max": dates[-1], "date_count": len(dates),
        "missing_dates": sorted(set(pd.date_range(dates[0], dates[-1]).strftime("%Y-%m-%d")) - set(dates)),
        "duplicate_snapshot_route_rows": int(candidates.duplicated(["snapshot_date", "route_id"]).sum()),
        "candidate_count_min": int(candidates.groupby("snapshot_date").size().min()),
        "candidate_count_max": int(candidates.groupby("snapshot_date").size().max()),
        "negative_quantity_rows": int((candidates["recommended_qty"] < 0).sum()),
        "negative_cost_rows": int((candidates["move_cost"] < 0).sum()),
        "missing_cost_rows": int(candidates["move_cost"].isna().sum()),
        "zero_expected_saving_rows": int((candidates["expected_saving"].fillna(0) == 0).sum()),
        "source_inventory_match_rows": int(candidates["inventory_source_matched"].sum()),
        "target_inventory_match_rows": int(candidates["inventory_target_matched"].sum()),
        "strict_pipeline_success_days": int(snapshot_summary["strict_success"].astype(str).str.lower().eq("true").sum()),
    }

    daily_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    reranked_days: dict[str, pd.DataFrame] = {}
    direct_rows: list[dict[str, Any]] = []

    for date in dates:
        day = candidates[candidates["snapshot_date"] == date].copy()
        day["final_recommendation"] = "재고 이동"
        day = add_heuristic_scores(day)
        auto = apply_auto_vhs(day)
        day = auto.frame
        reranked_days[date] = day
        sensitivity.extend(vhs_sensitivity_rows(day, auto.analysis["weights"], date))

        milp_result = lexicographic_milp(day)
        milp_selected = milp_result["selected"]
        milp_service = float(pd.to_numeric(milp_selected["recommended_qty"], errors="coerce").sum())
        selections = {
            "VHS": ordered_feasible_selection(day, ("vhs_rank", "route_id"), (True, True)),
            "Greedy": ordered_feasible_selection(day, ("greedy_rank", "route_id"), (True, True)),
            "Varo Final": ordered_feasible_selection(day, ("varo_final_rank", "route_id"), (True, True)),
            "MILP": milp_selected,
        }
        pareto_selected, pareto_result = pareto_operational_selection(day)
        selections["Pareto"] = pareto_selected
        for strategy, selected in selections.items():
            status = "optimal" if strategy == "MILP" and milp_result["optimal"] else "feasible"
            extra = None
            if strategy == "Pareto":
                extra = {
                    "frontier_size": pareto_result.trace[0]["frontier_size"] if pareto_result.trace else 0,
                    "frontier_recalculation_steps": len(pareto_result.trace),
                    "active_objectives": "|".join(pareto_result.active_objectives),
                    "inactive_objectives": "|".join(pareto_result.inactive_objectives),
                }
            daily_rows.append(_strategy_row(date, strategy, day, selected, milp_service, status, extra))

        direct = day.sort_values(["varo_final_rank", "route_id"], kind="mergesort").head(MAX_DAILY_ROUTES)
        direct_service = float(pd.to_numeric(direct["recommended_qty"], errors="coerce").sum())
        direct_cost = float(pd.to_numeric(direct["move_cost"], errors="coerce").sum())
        saved = saved_milp[saved_milp["snapshot_date"] == date].iloc[0]
        direct_rows.append({
            "date": date, "service_qty": direct_service, "cost": direct_cost,
            "milp_service": float(saved["milp_served_qty"]), "milp_cost": float(saved["milp_transport_cost"]),
        })
        for strategy, selected in selections.items():
            service = float(pd.to_numeric(selected.get("recommended_qty", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            cost = float(pd.to_numeric(selected.get("move_cost", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            comparison_rows.append({
                "date": date, "strategy": strategy,
                "milp_solver_status": "optimal" if milp_result["optimal"] else "not_optimal",
                "service_qty": service, "milp_service_qty": milp_service,
                "service_gap_units": milp_service - service,
                "cost": cost, "milp_cost": float(pd.to_numeric(milp_selected["move_cost"], errors="coerce").sum()),
                "comparison_label": "동일 서비스 비용 비교" if abs(service - milp_service) <= 1e-8 else "서비스 비동등 참고 Gap",
                "cost_gap_at_equal_service": (
                    cost - float(pd.to_numeric(milp_selected["move_cost"], errors="coerce").sum())
                    if abs(service - milp_service) <= 1e-8 else None
                ),
            })

    combined = pd.concat([reranked_days[date] for date in dates], ignore_index=True)
    combined_records = _records(combined)
    smoke_result, smoke_summary = _train_stage(combined_records, 12, 17, "A_smoke")
    medium_result, medium_summary = _train_stage(combined_records, 90, 17, "B_medium")
    production: dict[int, dqn_service.DqnTrainingResult] = {}
    seed_summaries = [smoke_summary, medium_summary]
    action_rows: list[dict[str, Any]] = []
    for seed in (17, 29, 41):
        result, summary = _train_stage(combined_records, production_episodes, seed, "C_production")
        production[seed] = result
        seed_summaries.append(summary)
        for action, count in sorted(result.action_distribution.items()):
            action_rows.append({"stage": "C_production", "seed": seed, "action": action, "count": count, "ratio": count / max(1, result.candidate_count)})

    primary = production[17]
    for date in dates:
        day = reranked_days[date]
        inferred = dqn_service.infer_dqn_actions(
            _records(day), primary.data_signature, model_path=primary.model_path,
        )
        keyed = dqn_service._route_ids(_records(day))
        working = day.copy()
        working["dqn_action_eval"] = [inferred.dqn_action_by_route.get(key) for key in keyed]
        working["dqn_confidence_eval"] = [inferred.dqn_confidence_by_route.get(key) for key in keyed]
        transfer = working[working["dqn_action_eval"].isin(TRANSFER_ACTIONS)].copy()
        selected = ordered_feasible_selection(
            transfer, ("dqn_confidence_eval", "recommended_qty", "move_cost", "route_id"),
            (False, False, True, True),
        )
        milp_daily = next(row for row in daily_rows if row["date"] == date and row["strategy"] == "MILP")
        daily_rows.append(_strategy_row(date, "DQN", day, selected, float(milp_daily["service_qty"]), "saved_model_forward"))
        dqn_row = next(row for row in daily_rows if row["date"] == date and row["strategy"] == "DQN")
        comparison_rows.append({
            "date": date, "strategy": "DQN", "milp_solver_status": "optimal",
            "service_qty": dqn_row["service_qty"], "milp_service_qty": milp_daily["service_qty"],
            "service_gap_units": milp_daily["service_qty"] - dqn_row["service_qty"],
            "cost": dqn_row["total_cost"], "milp_cost": milp_daily["total_cost"],
            "comparison_label": "동일 서비스 비용 비교" if abs(dqn_row["service_qty"] - milp_daily["service_qty"]) <= 1e-8 else "서비스 비동등 참고 Gap",
            "cost_gap_at_equal_service": dqn_row["total_cost"] - milp_daily["total_cost"] if abs(dqn_row["service_qty"] - milp_daily["service_qty"]) <= 1e-8 else None,
        })

    daily = pd.DataFrame(daily_rows).sort_values(["date", "strategy"], kind="mergesort")
    aggregate = aggregate_strategy_summary(daily, pd.DataFrame(comparison_rows))

    direct_frame = pd.DataFrame(direct_rows)
    new_varo = {
        "service_loss_days": int((direct_frame["service_qty"] < direct_frame["milp_service"] - 1e-8).sum()),
        "service_loss_units": float((direct_frame["milp_service"] - direct_frame["service_qty"]).clip(lower=0).sum()),
        "equal_service_cost_defeat_days": int(((direct_frame["service_qty"] - direct_frame["milp_service"]).abs().le(1e-8) & (direct_frame["cost"] > direct_frame["milp_cost"] + 1e-8)).sum()),
    }

    legacy_models: list[dict[str, Any]] = []
    if dqn_service.OUTPUT_DIR.exists():
        for path in sorted(dqn_service.OUTPUT_DIR.glob("*.pt")):
            payload = dqn_service._load_model_payload(path)
            compatible, message = dqn_service.model_payload_is_compatible(payload, primary.data_signature, "original")
            legacy_models.append({"file": path.name, "compatible": compatible, "reason": message})

    notes = {
        "source_files": {name: str(path) for name, path in paths.items()},
        "date_range": [dates[0], dates[-1]], "data_quality": quality,
        "baseline": _baseline_metrics(saved_milp), "new_varo_direct": new_varo,
        "selection_contract": {
            "candidate_universe": "saved 20 daily production candidates",
            "max_routes": MAX_DAILY_ROUTES,
            "source_limit": "actual daily source stock minus cross-node product median",
            "target_limit": "median stock + 7x actual outbound proxy - target stock",
            "budget": "not applied; no daily budget field exists in the saved benchmark",
            "vehicle_capacity": "no route-level unit-compatible capacity field; same pre-capped candidate quantities used for all strategies",
            "pareto_selection": "feasible pool -> frontier -> normalized ideal-point distance -> select one -> update shared constraints -> recompute; max 5",
            "dqn_selection": "seed 17 saved-model transfer actions, confidence desc then shared constraints",
        },
        "dqn": {
            "chronological_train_dates": [dates[0], dates[-7]],
            "chronological_holdout_dates": [dates[-6], dates[-1]],
            "normalization": "fit on training dates only",
            "production_episodes": production_episodes,
            "legacy_models": legacy_models,
        },
        "limitations": [
            "expected_saving is zero in all 620 saved candidates, so saving comparisons remain zero/NA rather than fabricated",
            "route-level budget and unit-compatible vehicle capacity are absent from the saved 31-day benchmark",
            "expected_saving is disabled as a Pareto objective whenever it is missing or has zero variance",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    daily.to_csv(output_dir / REQUIRED_OUTPUTS[0], index=False, encoding="utf-8-sig")
    aggregate.to_csv(output_dir / REQUIRED_OUTPUTS[1], index=False, encoding="utf-8-sig")
    pd.DataFrame(seed_summaries).to_csv(output_dir / REQUIRED_OUTPUTS[2], index=False, encoding="utf-8-sig")
    pd.DataFrame(action_rows).to_csv(output_dir / REQUIRED_OUTPUTS[3], index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(output_dir / REQUIRED_OUTPUTS[4], index=False, encoding="utf-8-sig")
    pd.DataFrame(comparison_rows).sort_values(["date", "strategy"]).to_csv(output_dir / REQUIRED_OUTPUTS[5], index=False, encoding="utf-8-sig")
    (output_dir / REQUIRED_OUTPUTS[6]).write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_dir": str(output_dir), "notes": notes, "aggregate": aggregate.to_dict("records"), "dqn": seed_summaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--production-episodes", type=int, default=300)
    parser.add_argument("--refresh-pareto-only", action="store_true")
    args = parser.parse_args()
    result = (
        refresh_pareto_validation(args.data_root, args.output_dir)
        if args.refresh_pareto_only
        else run_validation(args.data_root, args.output_dir, args.production_episodes)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

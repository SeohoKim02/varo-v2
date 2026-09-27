"""Prove canonical Suhyup inputs preserve the existing 31-day benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from services import dqn_service
from services.real_data_adapters import DATA_ROOT, DATASETS, suhyup_to_existing, verified_identifier_crosswalk
from services.suhyup_algorithm_revalidation import (
    _records, _selection_violation, enrich_inventory_constraints, lexicographic_milp,
    ordered_feasible_selection, pareto_operational_selection,
)
from services.legacy_adapters._local_modules.heuristic_optimizer import add_heuristic_scores
from services.vhs_score_engine import apply_auto_vhs


def validate_reference(root=DATA_ROOT):
    root = Path(root)
    folder = root / DATASETS["suhyup"]
    snapshot = pd.read_parquet(folder / "processed/canonical_inventory_snapshot.parquet")
    flow = pd.read_parquet(folder / "processed/canonical_inventory_flow.parquet")
    bridge = suhyup_to_existing(snapshot, flow)
    previous = pd.read_csv(folder / "processed/suhyup_logistics_inventory_flow_actual.csv", dtype={"date": str, "center_code": str, "product_code": str, "state_code": str})
    crosswalks = {}
    for col in ["center_code", "product_code", "state_code"]:
        crosswalks[col] = verified_identifier_crosswalk(bridge[col], previous[col])
        bridge[col] = bridge[col].map(crosswalks[col])
    keys = ["date", "center_code", "product_code", "state_code"]
    quantities = ["stock_qty", "inbound_qty", "outbound_qty"]
    a = bridge.sort_values(keys).reset_index(drop=True)
    b = previous.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys].astype(str), b[keys].astype(str))
    pd.testing.assert_frame_equal(a[quantities].astype(float), b[quantities].astype(float), check_exact=True)

    candidates_path = root / "16_VARO_E2E_20260731/multi_snapshot_validation/suhyup_202607_multi_snapshot_recommendations.csv"
    raw = pd.read_csv(candidates_path, dtype={"snapshot_date": str, "route_id": str, "product_id": str, "source_id": str, "target_id": str})
    canonical_routes = pd.read_parquet(folder / "processed/canonical_transfer_network.parquet")
    route_keys = ["snapshot_date", "route_id"]
    reconstructed = canonical_routes.rename(columns={"date": "snapshot_date", "transport_cost": "move_cost"})
    for col in ["source_id", "target_id", "product_id"]:
        counterpart = pd.concat([raw.source_id, raw.target_id]) if col != "product_id" else raw.product_id
        crosswalk = verified_identifier_crosswalk(reconstructed[col], counterpart)
        reconstructed[col] = reconstructed[col].map(crosswalk)
    fields = ["product_id", "source_id", "target_id", "recommended_qty", "distance_km", "travel_time_min", "move_cost", "expected_saving"]
    left = raw.sort_values(route_keys).reset_index(drop=True)
    right = reconstructed.sort_values(route_keys).reset_index(drop=True)
    for col in fields:
        if col in {"product_id", "source_id", "target_id"}:
            pd.testing.assert_series_equal(left[col].astype(str), right[col].astype(str), check_names=False)
        else:
            pd.testing.assert_series_equal(left[col].astype(float), right[col].astype(float), check_names=False, check_exact=True)
    # Reconstruct actual algorithm input from the canonical columns while retaining
    # unchanged legacy scoring metadata not owned by this interchange schema.
    candidates = raw.drop(columns=fields).merge(reconstructed[[*route_keys, *fields]], on=route_keys, validate="one_to_one", how="left")
    for col in fields:
        if col not in {"product_id", "source_id", "target_id"}:
            candidates[col] = candidates[col].astype(float)
    candidates = enrich_inventory_constraints(candidates, bridge)
    old_constraints = enrich_inventory_constraints(raw, previous)
    for col in ["source_surplus", "target_need_7d"]:
        pd.testing.assert_series_equal(candidates[col].astype(float), old_constraints[col].astype(float), check_exact=True)

    model_result = None
    for path in sorted(dqn_service.OUTPUT_DIR.glob("dqn_result_suhyup_202607_31d_original_*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("seed") == 17 and data.get("episodes") == 300:
            model_result = data
    if not model_result:
        raise FileNotFoundError("Existing Suhyup seed 17/300 model result is required; no retraining or fallback permitted")
    model_path = dqn_service.OUTPUT_DIR / Path(model_result["model_path"]).name
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    rows = []
    for date, day in candidates.groupby("snapshot_date", sort=True):
        day = day.copy()
        day["final_recommendation"] = "재고 이동"
        day = apply_auto_vhs(add_heuristic_scores(day)).frame
        milp = lexicographic_milp(day)
        if not milp["optimal"]:
            raise AssertionError(f"MILP not optimal: {date}")
        selections = {
            "Varo Final": ordered_feasible_selection(day, ("varo_final_rank", "route_id"), (True, True)),
            "VHS": ordered_feasible_selection(day, ("vhs_rank", "route_id"), (True, True)),
            "Greedy": ordered_feasible_selection(day, ("greedy_rank", "route_id"), (True, True)),
            "Pareto": pareto_operational_selection(day)[0], "MILP": milp["selected"],
        }
        inferred = dqn_service.infer_dqn_actions(_records(day), model_result["data_signature"], model_path=str(model_path))
        if inferred.model_status != "loaded":
            raise AssertionError(f"DQN inference failed: {inferred.message}")
        keys_dqn = dqn_service._route_ids(_records(day))
        working = day.copy()
        working["dqn_action_eval"] = [inferred.dqn_action_by_route.get(k) for k in keys_dqn]
        working["dqn_confidence_eval"] = [inferred.dqn_confidence_by_route.get(k) for k in keys_dqn]
        selections["DQN"] = ordered_feasible_selection(working[working.dqn_action_eval.isin(dqn_service.TRANSFER_ACTIONS)], ("dqn_confidence_eval", "recommended_qty", "move_cost", "route_id"), (False, False, True, True))
        for strategy, selected in selections.items():
            violation = _selection_violation(_records(selected))
            rows.append({"date": date, "strategy": strategy, "service_qty": float(selected.recommended_qty.sum()), "total_cost": float(selected.move_cost.sum()), "feasibility_violations": int(violation is not None)})
    daily = pd.DataFrame(rows).sort_values(["date", "strategy"]).reset_index(drop=True)
    saved = pd.read_csv(root / "21_SUHYUP_ALGORITHM_REVALIDATION_20260919/daily_strategy_summary.csv").sort_values(["date", "strategy"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(daily, saved[daily.columns], check_dtype=False, check_exact=True)
    summary = daily.groupby("strategy")[["service_qty", "total_cost", "feasibility_violations"]].sum().to_dict("index")
    result = {"stock_flow_rows": len(bridge), "raw_to_existing_quantities_exact": True, "candidate_fields_exact": True,
              "proxy_constraints_exact": True, "daily_strategy_rows_exact": len(daily), "days": int(daily.date.nunique()),
              "aggregate": summary, "dqn": "existing seed17 saved-model forward only; no training or model writes",
              "identifier_crosswalks": crosswalks,
              "limitations": "Same existing proxy-constrained benchmark; does not promote proxies to actual unmet demand."}
    (folder / "results/canonical_reference_regression.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    print(json.dumps(validate_reference(parser.parse_args().data_root), ensure_ascii=False, indent=2))

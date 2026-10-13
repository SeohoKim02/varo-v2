"""T4-B read-only real-data replay, with a new non-overwriting local report bundle.

Run: python -m services.sales_input_normalization_validation --data-root C:\\VARO_V2_REAL_DATA
No raw, canonical, reference, earlier validation, model or UI file is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from services import inventory_metric_integrity as imi
from services import inventory_metric_integrity_validation as t4
from services import milp_benchmark_integrity_validation as mbv
from services import shared_feasibility_validation as sfv
from services.seller_decision_validation import run_scenarios


OUTPUT_FILES = (
    "sales_input_alias_audit.csv", "sales_semantics_legacy_vs_validated.csv",
    "turnover_legacy_vs_validated.csv", "abc_legacy_vs_validated.csv", "risk_legacy_vs_validated.csv",
    "sales_action_impact_by_day.csv", "sales_action_impact_candidates.csv",
    "sales_input_normalization_summary.json", "sales_semantics_migration_readiness.json",
)


def _protected_hashes(root):
    """Stream file hashes; do not load large canonical parquet files into memory."""
    folders = set(t4.PROTECTED_BUNDLES) | {t4.OUTPUT_FOLDER}
    paths = {p for folder in folders for p in (root / folder).rglob("*") if p.is_file()}
    out = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        out[str(path.relative_to(root))] = digest.hexdigest()
    return out


def _external(root):
    """Full streamed coverage plus fresh normalization of actual keyed sample windows."""
    specs = {
        "m5": dict(end="2016-05-22", unit="item", price=True),
        "favorita": dict(end="2017-08-15", unit=None, price=False),
        "freshretailnet": dict(end="2024-07-02", unit="normalized_sales_amount", price=False, stockout=True),
    }
    out = {}
    for name, spec in specs.items():
        print(f"External validation: {name}", flush=True)
        path = root / t4.CANONICAL[name]
        stream = t4.stream_demand(path, window_end=spec["end"], price=spec["price"], stockout=spec.get("stockout", False))
        series = stream["series"]
        picks = pd.concat([series[series["win_rows"] == 30].head(1),
                           series[(series["win_rows"] > 0) & (series["win_rows"] < 30)].head(1),
                           series[series["win_neg"] > 0].head(1)]).drop_duplicates(["location_id", "product_id"])
        samples = []
        for item in picks.itertuples(index=False):
            frame = pq.read_table(path, filters=[("location_id", "=", item.location_id),
                                                  ("product_id", "=", item.product_id),
                                                  ("date", ">=", stream["window"][0]),
                                                  ("date", "<=", spec["end"])]).to_pandas()
            history = frame.rename(columns={"sales_qty": "quantity"}).assign(
                dataset_id=name, signal_type=imi.OBSERVED_RETAIL_SALES, quantity_unit=spec["unit"])
            history.attrs["semantics_evidence"] = imi.SOURCE_DOCUMENTED
            if name == "favorita":
                history["negative_semantics"] = history["validation_flags"].fillna("").map(
                    lambda v: imi.RETURN_DOCUMENTED if "documented_return" in v else imi.NEGATIVE_UNKNOWN)
            row = dict(store_id=item.location_id, product_id=item.product_id, dataset_id=name,
                       quantity_unit=spec["unit"], snapshot_date=spec["end"])
            normalized = imi.normalize_sales_signals({"inventory": pd.DataFrame([row]),
                                                       "daily_sales_history": history}, semantics_mode="VALIDATED")["rows"][0]
            window = normalized["windows"].get("retail_30d", {})
            expected = float(item.win_sum)
            samples.append({"location_id": item.location_id, "product_id": item.product_id,
                            "observed_days": window.get("observed_days"), "streamed_sum": expected,
                            "contract_sum": window.get("quantity"), "validated_sales_30d": normalized["sales_30d"],
                            "avg_daily_sales": normalized["avg_daily_sales"], "unit": spec["unit"],
                            "daily_mean_contract": normalized["signals"].get("average_daily_sales"),
                            "stockout_zero_days": window.get("stockout_zero_days"),
                            "availability_basis": window.get("availability_basis"),
                            "negative_total": window.get("negative_total"), "status": window.get("gate_status"),
                            "reason_codes": window.get("reason_codes"),
                            "same_observations": window.get("observed_days") == int(item.win_rows),
                            "same_sum": window.get("quantity") is not None and abs(window["quantity"] - expected) < 1e-5})
        out[name] = {"totals": stream["totals"], "coverage": t4.series_summary(stream), "samples": samples,
                     "real_turnover": "TURNOVER_UNAVAILABLE (no inventory)",
                     "abc": t4.m5_abc(series) if name == "m5" else None}
    return out


def migration_readiness():
    return {
        "production_replacement": "NOT_READY", "default_mode": "LEGACY", "approval_received": False,
        "computed_shadow_changes": ["sales_30d NULL on Suhyup", "turnover UNKNOWN on Suhyup",
                                    "validated action unavailable; not substituted"],
        "possible_not_computed_changes": ["discount/hold counts", "forecast outputs", "VHS scores",
                                          "Varo ranks", "transfer plans", "service/cost/KPI"],
        "steps": [
            {"id": "A", "requirement": "stable input contract, trustworthy loader lineage", "status": "IMPLEMENTED_WITH_LIMITATIONS"},
            {"id": "B", "requirement": "actual workbook with retail sales AND inventory history", "status": "DATA_MISSING"},
            {"id": "C", "requirement": "review differences including all 84 alias-sensitive routes", "status": "REQUIRES_REVIEW"},
            {"id": "D", "requirement": "unavailable evidence must not generate a production action", "status": "SHADOW_ONLY"},
            {"id": "E", "requirement": "user labels consistent with unavailable/proxy metrics", "status": "UI_NOT_CHANGED"},
            {"id": "F", "requirement": "approve revised Golden/reference criteria", "status": "NOT_APPROVED"},
            {"id": "G", "requirement": "approve production action and ranking migration", "status": "NOT_APPROVED"}],
        "T5_dependency": "T5 may consume typed shadow evidence; never relabel outbound as retail or unavailable as zero",
    }


def run_validation(root: Path, output_dir: Path | None = None):
    root = root.resolve()
    output = (output_dir or root / ("_SALES_INPUT_NORMALIZATION_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if output.parent != root or not output.name.startswith("_SALES_INPUT_NORMALIZATION_"):
        raise ValueError("reports require a new direct child _SALES_INPUT_NORMALIZATION_* folder")
    if output.exists():
        raise FileExistsError("existing validation results are never overwritten")
    started = time.perf_counter()
    before = _protected_hashes(root)
    rebuild, saved = mbv.GeneratorRebuild(root), sfv.load_suhyup_candidates(root)
    all_rows, actions, daily, abc, risk, aliases = [], [], [], [], [], []
    fidelity = []
    for day in sorted(saved["snapshot_date"].unique()):
        print(f"Suhyup replay: {day}", flush=True)
        declared_upload, plain_upload = t4.declared_day_upload(rebuild, day), rebuild.upload(day)
        declared = t4._pipeline(declared_upload, root)
        plain = t4._pipeline(plain_upload, root)
        drec, prec = pd.DataFrame(declared["recommendations"]), pd.DataFrame(plain["recommendations"])
        expected = saved[saved["snapshot_date"] == day]
        expected_actions = dict(zip(expected["route_id"], expected["varo_action"]))
        current_actions = dict(zip(drec["route_id"], drec["varo_action"]))
        plain_actions = dict(zip(prec["route_id"], prec["varo_action"]))
        fidelity.append({"date": day, "same_actions": expected_actions == current_actions,
                         "same_routes": set(expected_actions) == set(current_actions)})
        field = declared["pipeline_result"]["inventory_metric_integrity"]
        plain_field = plain["pipeline_result"]["inventory_metric_integrity"]
        if field["status"] != "parallel_only" or plain_field["status"] != "parallel_only":
            raise RuntimeError(f"parallel contract failed on {day}: {field}")
        # Source metadata is attached only to a copy for evidence-aware shadow replay.
        shadow_upload = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k, v in declared_upload.items()}
        shadow_upload["inventory"]["dataset_id"] = "suhyup"
        shadow = imi.build_sales_normalization_analysis(shadow_upload)
        normalized = shadow["validated_sales_signals"]
        key = {(n["store_id"], n["product_id"]): (n, t, old, pold)
               for n, t, old, pold in zip(normalized, shadow["validated_turnover"], field["rows"], plain_field["rows"])}
        for n, t, old, pold in key.values():
            all_rows.append({"date": day, "dataset": "suhyup", "store_id": n["store_id"], "product_id": n["product_id"],
                             "original_signal_type": imi.OBSERVED_OUTBOUND_FLOW, "original_period_days": 1,
                             "original_quantity": n["signals"].get("sales_qty", {}).get("quantity"),
                             "legacy_sales_30d": n["legacy_values"]["sales_30d"], "undeclared_sales_30d": pold["legacy_sales_used"],
                             "validated_sales_30d": n["sales_30d"], "validated_avg_daily_sales": n["avg_daily_sales"],
                             "legacy_avg_daily_sales": n["legacy_values"]["avg_daily_sales"],
                             "validated_sales_status": "DATA_UNAVAILABLE", "validated_sales_period": "30 days",
                             "validated_sales_source": "no retail sales history (not outbound)",
                             "legacy_turnover": old["legacy_turnover_days"], "undeclared_turnover": pold["legacy_turnover_days"],
                             "validated_turnover": t["turnover_value"], "validated_turnover_status": t["turnover_status"],
                             "legacy_grade": old["legacy_turnover_grade"], "undeclared_grade": pold["legacy_turnover_grade"],
                             "validated_grade": t["turnover_classification"], "unit": n["quantity_unit"],
                             "source": "Suhyup actual outbound_qty", "retail_history_available": False,
                             "legacy_safety_stock_mean_source": old.get("safety_stock_mean_source"),
                             "validated_safety_stock_status": "INSUFFICIENT_SALES_EVIDENCE",
                             "legacy_matching_use": old.get("match_use"),
                             "validated_matching_status": "DESTINATION_DEMAND_UNAVAILABLE",
                             "legacy_forecast_codes": "|".join(old.get("forecast_codes", [])),
                             "validated_forecast_status": "OUTBOUND_PROXY_NOT_RETAIL_HISTORY",
                             "reason_codes": "|".join(n["reason_codes"] + t["reason_codes"])})
        dby = {r["route_id"]: r for r in drec.to_dict("records")}
        pby = {r["route_id"]: r for r in prec.to_dict("records")}
        for route, action in current_actions.items():
            cand, other = dby[route], pby.get(route, {})
            n, t, old, pold = key.get((str(cand["source_id"]), str(cand["product_id"])), ({}, {}, {}, {}))
            shifted = plain_actions.get(route) != action
            actions.append({"date": day, "route_id": route, "product_id": cand["product_id"],
                            "source_id": cand["source_id"], "target_id": cand["target_id"],
                            "route_type": cand.get("route_type"), "original_sales_signal": "OBSERVED_OUTBOUND_FLOW",
                            "original_period": "ONE_DAY", "legacy_action": action,
                            "undeclared_action": plain_actions.get(route), "alias_sensitive": shifted,
                            "legacy_sales_30d": old.get("legacy_sales_used"), "undeclared_sales_30d": pold.get("legacy_sales_used"),
                            "validated_sales_30d": n.get("sales_30d"),
                            "legacy_turnover": old.get("legacy_turnover_days"), "undeclared_turnover": pold.get("legacy_turnover_days"),
                            "validated_turnover": t.get("turnover_value"), "legacy_grade": old.get("legacy_turnover_grade"),
                            "undeclared_grade": pold.get("legacy_turnover_grade"), "validated_grade": t.get("turnover_classification"),
                            "legacy_disposal_score": cand.get("disposal_risk_score"), "undeclared_disposal_score": other.get("disposal_risk_score"),
                            "legacy_vhs": cand.get("vhs", cand.get("vhs_score")), "undeclared_vhs": other.get("vhs", other.get("vhs_score")),
                            "validated_vhs": None, "validated_action": None, "validated_canonical_action": None,
                            "status": "VALIDATED_ACTION_UNAVAILABLE", "retail_history_available": False,
                            "production_action_applied": False,
                            "reason_codes": "INSUFFICIENT_SALES_EVIDENCE|TURNOVER_UNAVAILABLE|DESTINATION_DEMAND_UNAVAILABLE",
                            "code_path": "data_loader aliases -> prepare_legacy_data daily*30 -> turnover/disposal -> VHS._recommend_action -> normalize_action -> varo_action"})
        day_actions = [a for a in actions if a["date"] == day]
        daily.append({"date": day, "candidates": len(day_actions), "alias_sensitive": sum(a["alias_sensitive"] for a in day_actions),
                      "comparison_unavailable": len(day_actions), "production_actions_changed": 0,
                      "decision_change_computed": False, "same_saved_actions": fidelity[-1]["same_actions"]})
        abc.append({"date": day, "legacy_basis": field["abc"]["legacy_basis"],
                    "validated_basis": shadow["validated_abc"]["basis"], "validated_status": shadow["validated_abc"]["gate_status"],
                    "reason_codes": "|".join(shadow["validated_abc"]["reason_codes"])})
        risk.extend({"date": day, "store_id": old["store_id"], "product_id": old["product_id"],
                     "legacy_score": old["disposal_legacy_score"], "validated_score": None,
                     "risk_basis": imi.DISPOSAL_MEANING, "risk_is_probability": False,
                     "status": "INSUFFICIENT_SALES_EVIDENCE", "actual_disposal_probability": None}
                    for old in field["rows"])
        aliases.extend({"date": day, **a} for a in shadow["legacy_input_audit"])
    external = _external(root)
    print("Golden and regression replay", flush=True)
    golden = t4.golden_186(root)
    t1 = t4.t1_rerun(root)
    bundles = t4.earlier_bundles(root)
    seller, _ = run_scenarios()
    old_scenarios = t4.controlled_scenarios()
    flow_proxy = t4.suhyup_flow_proxy(root)
    after = _protected_hashes(root)
    rows, action_frame = pd.DataFrame(all_rows), pd.DataFrame(actions)
    affected = action_frame[action_frame["alias_sensitive"]]
    checks = {
        "suhyup_620": len(actions) == 620, "suhyup_31_days": len(daily) == 31,
        "saved_production_actions_preserved": all(f["same_actions"] and f["same_routes"] for f in fidelity),
        "84_alias_sensitive_22_days": len(affected) == 84 and affected["date"].nunique() == 22,
        "validated_sales_not_outbound": rows["validated_sales_30d"].isna().all(),
        "no_fabricated_validated_actions": action_frame["validated_action"].isna().all(),
        "golden_186": len(golden) == 186 and golden["equal"].all(),
        "t1": t1["benchmark_service"] == 7750 and t1["benchmark_cost"] == 13629146 and t1["benchmark_violations"] == 0,
        "earlier_bundles": all(not bundles[k]["failed_checks"] for k in ("T1", "T2", "T3", "P1", "P1B", "P1C")),
        "seller_18": len(seller) == 18 and bool(seller["pass"].all()),
        "t4_controlled": all(s["pass"] for s in old_scenarios),
        "external_samples": all(s["same_observations"] and s["same_sum"] for item in external.values() for s in item["samples"]),
        "protected_files_unchanged": before == after,
        "flow_proxy_not_retail": bool((flow_proxy["rows"]["turnover_basis"] == imi.FLOW_BASED_TURNOVER_PROXY).all()),
    }
    summary = {"task": "T4-B", "checks": {k: bool(v) for k, v in checks.items()},
               "failed_checks": [k for k, v in checks.items() if not v], "suhyup_candidates": len(actions),
               "alias_sensitive_routes": len(affected), "affected_days": int(affected["date"].nunique()),
               "transitions": affected.groupby(["legacy_action", "undeclared_action"]).size().to_dict(),
               "validated_action_unavailable": len(actions), "production_action_applied": False,
               "golden": {"same": int(golden["equal"].sum()), "total": len(golden)}, "t1_rerun": t1,
               "earlier_bundles": bundles, "seller_scenarios": seller.to_dict("records"),
               "flow_proxy": {"series": len(flow_proxy["rows"]),
                              "status_counts": flow_proxy["rows"]["turnover_status"].value_counts().to_dict(),
                              "basis": imi.FLOW_BASED_TURNOVER_PROXY, "conservation": flow_proxy["conservation"]},
               "real_outcomes": len(pd.read_csv(root / "_SELLER_LOSS_VALIDATION/seller_shadow_outcomes.csv")),
               "external": external, "protected_file_count": len(before),
               "protected_changes": [k for k in set(before) | set(after) if before.get(k) != after.get(k)],
               "seconds": round(time.perf_counter() - started, 1),
               "limitations": ["validated action is unavailable until all downstream evidence is verified",
                               "risk/safety/matching/VHS formula replacements deliberately not applied",
                               "no real dataset combines keyed retail sales and sufficient stock history",
                               "historical serialized uploads lack original-column metadata; declarations still required"]}
    summary["transitions"] = {f"{a}->{b}": int(v) for (a, b), v in summary["transitions"].items()}
    output.mkdir()
    for frame, name in zip([pd.DataFrame(aliases), rows, rows[[c for c in rows if "turnover" in c or "grade" in c or c in ("date", "store_id", "product_id", "unit", "source", "reason_codes")]],
                            pd.DataFrame(abc), pd.DataFrame(risk), pd.DataFrame(daily), action_frame], OUTPUT_FILES[:7]):
        frame.to_csv(output / name, index=False, encoding="utf-8-sig")
    (output / OUTPUT_FILES[7]).write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=t4._json_default), encoding="utf-8")
    (output / OUTPUT_FILES[8]).write_text(json.dumps(migration_readiness(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_dir": str(output), "failed_checks": summary["failed_checks"],
            "checks": summary["checks"], "routes": len(actions), "alias_sensitive": len(affected),
            "days_affected": int(affected["date"].nunique()), "seconds": summary["seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\VARO_V2_REAL_DATA"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run_validation(args.data_root, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failed_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""REAL_DATA_PLUS_USER_INPUT_E2E: real Suhyup rows + explicit TEST USER INPUT through the production pipeline.

This is NOT a "FULL_REAL" validation. The Suhyup 2026-07-31 production rows (16_VARO_E2E_20260731/processed: actual
center stock, OSRM routes, official-tariff real transport) are loaded exactly like an upload; the business values
Suhyup does not have (price, holding/disposal cost, shelf life, retail demand, ...) are supplied as an explicit
`seller_loss_inputs` table labelled TEST USER INPUT. The numbers are test values, not Suhyup data and not defaults;
the results check the wiring (provenance, priority, conflicts, units, currency, modes), not a business recommendation.

Run: python -m services.seller_loss_input_validation --data-root C:/VARO_V2_REAL_DATA
Writes (local only, never into git) <data-root>/_SELLER_LOSS_VALIDATION/:
    seller_loss_input_contract.json               seller_loss_inputs contract (+ engine contract signature)
    seller_loss_input_provenance_validation.csv   variant x decision x field: value, provenance, origin, outcome
    seller_loss_real_plus_user_input_e2e.csv      variant x decision: status, evidence, readiness, losses, fields
    seller_loss_input_validation_summary.json     checks, variant counts, the TEST USER INPUT tables, input hashes
Raw and processed files are read only; the TEST USER INPUT exists only in memory and in these result files.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from services.real_data_adapters import DATA_ROOT
from services.seller_decision_validation import OUTPUT_FOLDER, _env
from services.seller_loss_engine import ENGINE_VERSION, STATUS_UNAVAILABLE, TRANSFER, contract_document
from services import seller_loss_inputs as sli

E2E_LABEL = "REAL_DATA_PLUS_USER_INPUT_E2E"
TEST_INPUT_LABEL = "TEST USER INPUT (explicit test value, not Suhyup data, not a default)"
PROCESSED = "16_VARO_E2E_20260731/processed"
PROCESSED_FILES = ("stores.csv", "products.csv", "inventory.csv", "routes.csv")
OUTPUT_FILES = ("seller_loss_input_contract.json", "seller_loss_input_provenance_validation.csv",
                "seller_loss_real_plus_user_input_e2e.csv", "seller_loss_input_validation_summary.json")
ID_COLUMNS = {"store_id": str, "product_id": str, "node_id": str, "source_id": str, "target_id": str}
VARIANTS = {
    "A_REAL_ONLY_NO_SELLER_INPUT": "Real Suhyup rows only (no seller_loss_inputs)",
    "B_MONETARY_TEST_INPUT_ONLY": "TEST USER INPUT for money fields only (price, unit cost, holding, disposal, salvage, discount rate)",
    "C_FULL_TEST_BUSINESS_INPUT": "B + declared unit, shelf life, retail daily demand, uplift for some products, one contradicting route cost",
    "D_UNIT_MISMATCH_CHECK": "C with prices declared per KG while the declared inventory unit is BOX",
    "E_CURRENCY_MISMATCH_CHECK": "C with every seller money value in USD while the real route cost is KRW",
    "F_SCENARIO_WHAT_IF": "C + SCENARIO rows (per-unit route cost, 50% markdown) in SCENARIO mode",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_suhyup_upload(data_root: Path) -> dict[str, pd.DataFrame]:
    """The real Suhyup 2026-07-31 production rows as an upload (same normalisation as a workbook)."""
    from services.data_loader import normalize_loaded_data

    folder = Path(data_root) / PROCESSED
    raw = {name.removesuffix(".csv"): pd.read_csv(folder / name, dtype=ID_COLUMNS, encoding="utf-8-sig")
           for name in PROCESSED_FILES}
    return normalize_loaded_data(raw)


def _roles(decisions: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    sources: dict[str, set[str]] = {}
    targets: dict[str, set[str]] = {}
    for d in decisions:
        sources.setdefault(d["product_id"], set()).add(d["source_store_id"])
        if d.get("target_store_id"):
            targets.setdefault(d["product_id"], set()).add(d["target_store_id"])
    roles = {}
    for product in sorted(set(sources) | set(targets)):
        for store in sorted(sources.get(product, set()) | targets.get(product, set())):
            in_source, in_target = store in sources.get(product, set()), store in targets.get(product, set())
            roles[(product, store)] = "both" if in_source and in_target else ("source" if in_source else "target")
    return roles


def test_user_inputs(decisions: Sequence[Mapping[str, Any]]) -> dict[str, pd.DataFrame]:
    """Explicit TEST USER INPUT tables per variant, keyed on the real decisions' product / store ids."""
    products = sorted({d["product_id"] for d in decisions})
    roles = _roles(decisions)
    demand = {"source": 5, "target": 60, "both": 20}   # test values: slow source, fast target
    note = TEST_INPUT_LABEL
    money_rows = [{"scope": "GLOBAL", "holding_cost_per_unit_day": 20, "disposal_cost_per_unit": 500,
                   "salvage_value_per_unit": 0, "discount_rate": 0.3, "currency": "KRW", "note": note}]
    money_rows += [{"scope": "PRODUCT", "product_id": p, "normal_price": 30000, "unit_cost": 20000, "currency": "KRW",
                    "note": note} for p in products]

    def full(price_unit: str, currency: str) -> list[dict[str, Any]]:
        rows = [{"scope": "GLOBAL", "quantity_unit": "BOX", "holding_cost_per_unit_day": 20, "disposal_cost_per_unit": 500,
                 "salvage_value_per_unit": 0, "discount_rate": 0.3, "currency": currency,
                 "note": note + "; quantity_unit declared by the test seller (Suhyup raw quantity unit is UNKNOWN)"}]
        for index, product in enumerate(products):
            row = {"scope": "PRODUCT", "product_id": product, "normal_price": 30000, "unit_cost": 20000,
                   "remaining_shelf_life_days": 21, "price_unit": price_unit, "note": note}
            if index % 2 == 0:
                row["promotion_uplift"] = 0.5   # only half the products get uplift evidence: the others stay PARTIAL
            rows.append(row)
        rows += [{"scope": "PRODUCT_STORE", "product_id": p, "store_id": s, "daily_demand": demand[role], "note": note}
                 for (p, s), role in sorted(roles.items())]
        first = decisions[0]
        rows.append({"scope": "ROUTE", "source_store_id": first["source_store_id"], "target_store_id": first["target_store_id"],
                     "product_id": first["product_id"], "transfer_cost": 1, "currency": currency,
                     "note": note + "; deliberately contradicts the real route cost (must be recorded, never used)"})
        return rows

    scenario_rows = [{"scope": "GLOBAL", "input_type": "SCENARIO", "transfer_cost_per_unit": 100, "price_unit": "KRW/BOX",
                      "note": "TEST SCENARIO INPUT (what-if, SCENARIO mode only)"},
                     {"scope": "GLOBAL", "input_type": "SCENARIO", "discount_rate": 0.5,
                      "note": "TEST SCENARIO INPUT (what-if, SCENARIO mode only)"}]
    return {
        "B_MONETARY_TEST_INPUT_ONLY": pd.DataFrame(money_rows),
        "C_FULL_TEST_BUSINESS_INPUT": pd.DataFrame(full("KRW/BOX", "KRW")),
        "D_UNIT_MISMATCH_CHECK": pd.DataFrame(full("KRW/KG", "KRW")),
        "E_CURRENCY_MISMATCH_CHECK": pd.DataFrame(full("USD/BOX", "USD")),
        "F_SCENARIO_WHAT_IF": pd.DataFrame(full("KRW/BOX", "KRW") + scenario_rows),
    }


def _run(data: Mapping[str, Any], data_root: Path) -> dict[str, Any]:
    from services.analysis_pipeline import build_v2_state

    with _env("VARO_REAL_DATA_ROOT", str(data_root)), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_v2_state(copy.deepcopy(dict(data)), detail_level="core")


def _join(values: Sequence[str] | None) -> str:
    return "|".join(values or [])


def _decision_row(variant: str, decision: Mapping[str, Any], date: str | None) -> dict[str, Any]:
    prov = decision["input_provenance"]
    return {
        "e2e_label": E2E_LABEL, "variant": variant, "decision_id": decision["decision_id"], "snapshot_date": date,
        "product_id": decision["product_id"], "source_store_id": decision["source_store_id"],
        "target_store_id": decision["target_store_id"], "decision_qty": decision["decision_qty"],
        "quantity_unit": decision["quantity_unit"], "currency": decision["currency"],
        "decision_mode": decision["decision_mode"], "evidence_level": decision["evidence_level"],
        "recommendation_readiness": decision["recommendation_readiness"],
        "readiness_reasons": _join(decision["readiness_reasons"]),
        "comparison_status": decision["comparison_status"], "recommendation_status": decision["recommendation_status"],
        "recommended_strategy": decision["recommended_strategy"],
        "expected_loss_transfer": decision["expected_loss_transfer"],
        "expected_loss_normal_sale": decision["expected_loss_normal_sale"],
        "expected_loss_discount_sale": decision["expected_loss_discount_sale"],
        "loss_difference_vs_second_best": decision["loss_difference_vs_second_best"],
        "strategy_readiness": "|".join(f"{k}={v}" for k, v in decision["strategy_readiness"].items()),
        "real_input_fields": _join(decision["real_input_fields"]),
        "seller_input_fields": _join(decision["seller_input_fields"]),
        "scenario_input_fields": _join(decision["scenario_input_fields"]),
        "proxy_input_fields": _join(decision["proxy_input_fields"]),
        "missing_required_fields": _join(decision["missing_required_fields"]),
        "unknown_optional_fields": _join(decision["unknown_optional_fields"]),
        "conflicting_fields": _join(decision["conflicting_fields"]),
        "source_current_stock": prov["source_current_stock"]["value"],
        "source_current_stock_provenance": prov["source_current_stock"]["provenance"],
        "target_current_stock": prov["target_current_stock"]["value"],
        "target_current_stock_provenance": prov["target_current_stock"]["provenance"],
        "transfer_cost": prov["transfer_cost"]["value"], "transfer_cost_provenance": prov["transfer_cost"]["provenance"],
        "transit_time_days": prov["transit_time_days"]["value"],
        "seller_inputs_changed_recommendation": decision["seller_input_influence"].get("seller_inputs_changed_recommendation"),
        "fields_changing_recommendation": _join(decision["seller_input_influence"].get("fields_changing_recommendation")),
        "legacy_action": decision["legacy_action"], "legacy_action_changed": decision["legacy_action_changed"],
        "reason_codes": _join(decision["reason_codes"]), "explanation": decision["explanation"],
    }


def _provenance_rows(variant: str, decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, source in decision["input_sources"].items():
        rows.append({"e2e_label": E2E_LABEL, "variant": variant, "decision_id": decision["decision_id"], "field": name,
                     "value": source["value"], "provenance": source["provenance"], "origin": source["origin"],
                     "scope": source["scope"], "outcome": source["outcome"], "used_in_comparison": source["used_in_comparison"],
                     "source": source["source"]})
    return rows


def _check(check_id: str, description: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"check_id": check_id, "description": description, "status": "PASS" if ok else "FAIL",
            "detail": json.dumps(detail, ensure_ascii=False, default=str) if detail is not None else ""}


def run_e2e(data_root: Path) -> dict[str, Any]:
    data_root = Path(data_root)
    inputs = [data_root / PROCESSED / name for name in PROCESSED_FILES]
    hashes_before = {str(p.relative_to(data_root)).replace("\\", "/"): _sha256(p) for p in inputs}
    upload = load_suhyup_upload(data_root)
    inventory = upload["inventory"]
    stock = {(str(r["store_id"]), str(r["product_id"])): float(r["stock_qty"]) for r in inventory.to_dict("records")}
    dates = {(str(r["store_id"]), str(r["product_id"])): str(r.get("snapshot_date")) for r in inventory.to_dict("records")}

    states = {"A_REAL_ONLY_NO_SELLER_INPUT": _run(upload, data_root)}
    baseline_decisions = states["A_REAL_ONLY_NO_SELLER_INPUT"]["pipeline_result"]["seller_loss_analysis"]["decisions"]
    tables = test_user_inputs(baseline_decisions)
    for variant, frame in tables.items():
        data = {**upload, sli.SHEET_KEY: frame}
        if variant == "F_SCENARIO_WHAT_IF":
            data[sli.DECISION_MODE_KEY] = sli.SCENARIO
        states[variant] = _run(data, data_root)
    repeat = _run({**upload, sli.SHEET_KEY: tables["C_FULL_TEST_BUSINESS_INPUT"]}, data_root)

    e2e_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    variant_summary: dict[str, Any] = {}
    analyses = {v: s["pipeline_result"]["seller_loss_analysis"] for v, s in states.items()}
    for variant, analysis in analyses.items():
        for decision in analysis["decisions"]:
            date = dates.get((decision["source_store_id"], decision["product_id"]))
            e2e_rows.append(_decision_row(variant, decision, date))
            provenance_rows.extend(_provenance_rows(variant, decision))
        variant_summary[variant] = {
            "description": VARIANTS[variant], "decision_mode": analysis["decision_mode"],
            "seller_loss_inputs_status": analysis["seller_loss_inputs"]["status"],
            "accepted_cells": analysis["seller_loss_inputs"]["accepted_cells"],
            "status_counts": analysis["status_counts"], "evidence_level_counts": analysis["evidence_level_counts"],
            "readiness_counts": analysis["readiness_counts"], "recommended_count": analysis["recommended_count"],
            "recommendable_count": analysis["recommendable_count"],
            "recommended_strategy_counts": dict(sorted(pd.Series([d["recommended_strategy"] or "NONE" for d in analysis["decisions"]]).value_counts().items())),
        }

    base_recs = states["A_REAL_ONLY_NO_SELLER_INPUT"]["recommendations"]
    move_cost = {str(r["route_id"]): float(r["move_cost"]) for r in base_recs}
    c_decisions = analyses["C_FULL_TEST_BUSINESS_INPUT"]["decisions"]
    actual_variants = [v for v in analyses if v != "F_SCENARIO_WHAT_IF"]
    first = c_decisions[0]
    checks = [
        _check("E1", "Legacy outputs (recommendations, KPI summary, top5) identical with and without seller inputs, in both modes",
               all(s["recommendations"] == base_recs and s["pipeline_result"]["summary"] == states["A_REAL_ONLY_NO_SELLER_INPUT"]["pipeline_result"]["summary"]
                   and s["pipeline_result"]["top5"] == states["A_REAL_ONLY_NO_SELLER_INPUT"]["pipeline_result"]["top5"]
                   for s in states.values())
               and all(not d["legacy_action_changed"] for a in analyses.values() for d in a["decisions"])),
        _check("E2", "Source/target stock equal the processed actual stock and stay DIRECT_REAL in every variant",
               all(d["input_provenance"]["source_current_stock"]["value"] == stock[(d["source_store_id"], d["product_id"])]
                   and d["input_provenance"]["source_current_stock"]["provenance"] == "DIRECT_REAL"
                   and d["input_provenance"]["target_current_stock"]["value"] == stock[(d["target_store_id"], d["product_id"])]
                   and d["input_provenance"]["target_current_stock"]["provenance"] == "DIRECT_REAL"
                   for a in analyses.values() for d in a["decisions"])),
        _check("E3", "Real transport cost (official tariff, quantity specific) preserved in every ACTUAL_OPERATION variant",
               all(d["input_provenance"]["transfer_cost"]["value"] == move_cost[d["decision_id"]]
                   and d["input_provenance"]["transfer_cost"]["provenance"] in ("DERIVED_REAL", "PROXY")
                   for v in actual_variants for d in analyses[v]["decisions"]),
               {"provenance": sorted({d["input_provenance"]["transfer_cost"]["provenance"] for d in baseline_decisions})}),
        _check("E4", "The contradicting TEST route cost is recorded as a conflict and never used",
               "transfer_cost" in first["conflicting_fields"]
               and any(c["kind"] == "REAL_VS_SELLER_INPUT" and c["field"] == "transfer_cost" for c in first["input_conflicts"])
               and first["input_provenance"]["transfer_cost"]["value"] == move_cost[first["decision_id"]]),
        _check("E5", "Every seller-supplied field carries USER_INPUT provenance and a seller_loss_inputs source",
               all(d["input_provenance"][f]["provenance"] == "USER_INPUT"
                   and d["input_provenance"][f]["source"].startswith("seller_loss_inputs[")
                   for d in c_decisions for f in d["seller_input_fields"]) and all(d["seller_input_fields"] for d in c_decisions)),
        _check("E6", "Without seller inputs, and with money-only test inputs, real gaps (shelf life, retail demand proxy) keep every decision INSUFFICIENT",
               all(d["evidence_level"] == sli.INSUFFICIENT for v in ("A_REAL_ONLY_NO_SELLER_INPUT", "B_MONETARY_TEST_INPUT_ONLY")
                   for d in analyses[v]["decisions"])
               and all({"SHELF_LIFE_MISSING", "PROXY_REJECTED:source_daily_demand"} <= set(d["reason_codes"])
                       for d in analyses["B_MONETARY_TEST_INPUT_ONLY"]["decisions"])),
        _check("E7", "Full test business input: comparable decisions are REAL_PLUS_USER_INPUT (never REAL_ONLY or FULL_REAL)",
               all(d["evidence_level"] in (sli.REAL_PLUS_USER_INPUT, sli.INSUFFICIENT) for d in c_decisions)
               and any(d["evidence_level"] == sli.REAL_PLUS_USER_INPUT for d in c_decisions),
               analyses["C_FULL_TEST_BUSINESS_INPUT"]["evidence_level_counts"]),
        _check("E8", "Price per KG with a declared BOX inventory unit blocks every comparison (UNIT_MISMATCH)",
               all(d["comparison_status"] == STATUS_UNAVAILABLE and "UNIT_MISMATCH" in d["reason_codes"]
                   for d in analyses["D_UNIT_MISMATCH_CHECK"]["decisions"])),
        _check("E9", "USD seller values with the KRW real route cost: TRANSFER unavailable (CURRENCY_MISMATCH), no exchange rate",
               all("CURRENCY_MISMATCH" in d["unavailable_strategies"].get(TRANSFER, [])
                   for d in analyses["E_CURRENCY_MISMATCH_CHECK"]["decisions"])),
        _check("E10", "SCENARIO run: every result SCENARIO_ONLY / SCENARIO evidence; the what-if route cost is SCENARIO_INPUT",
               all(d["decision_mode"] == sli.SCENARIO and d["recommendation_readiness"] in (sli.SCENARIO_ONLY, sli.UNAVAILABLE)
                   and d["input_provenance"]["transfer_cost"]["provenance"] == "SCENARIO_INPUT"
                   for d in analyses["F_SCENARIO_WHAT_IF"]["decisions"])),
        _check("E11", "Deterministic: re-running the full test input reproduces every decision",
               repeat["pipeline_result"]["seller_loss_analysis"]["decisions"] == c_decisions),
    ]
    hashes_after = {str(p.relative_to(data_root)).replace("\\", "/"): _sha256(p) for p in inputs}
    checks.append(_check("E12", "Processed real inputs unchanged (sha256 before == after); TEST USER INPUT never written to the data root",
                         hashes_before == hashes_after, hashes_after))
    return {"e2e_rows": e2e_rows, "provenance_rows": provenance_rows, "variants": variant_summary, "checks": checks,
            "test_user_inputs": {k: v.where(v.notna(), None).to_dict("records") for k, v in tables.items()},
            "input_hashes": hashes_after, "decision_count": len(baseline_decisions)}


def run_validation(data_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    data_root = Path(data_root)
    output_dir = Path(output_dir) if output_dir else data_root / OUTPUT_FOLDER
    result = run_e2e(data_root)
    contract = sli.input_contract_document()
    contract_bundle = {**contract, "engine_version": ENGINE_VERSION,
                       "engine_contract_signature": contract_document()["contract_signature"]}
    failed = [c["check_id"] for c in result["checks"] if c["status"] != "PASS"]
    summary = {
        "label": E2E_LABEL,
        "not_full_real": "Real Suhyup rows + explicit TEST USER INPUT. This is not a FULL_REAL validation and the results "
                         "are wiring checks, not business recommendations.",
        "data_source": f"{PROCESSED} (Suhyup 2026-07-31 production rows) + real transport ({DATA_ROOT.name} cost matrix)",
        "engine_version": ENGINE_VERSION,
        "input_contract_version": sli.INPUT_CONTRACT_VERSION,
        "input_contract_signature": contract["contract_signature"],
        "decision_count": result["decision_count"],
        "variants": result["variants"],
        "checks": result["checks"],
        "failed_checks": failed,
        "test_user_input_label": TEST_INPUT_LABEL,
        "test_user_inputs": result["test_user_inputs"],
        "input_hashes": result["input_hashes"],
        "production_action_replaced": False,
        "full_monetary_real_dataset_available": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / OUTPUT_FILES[0]).write_text(json.dumps(contract_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(result["provenance_rows"]).to_csv(output_dir / OUTPUT_FILES[1], index=False, encoding="utf-8-sig")
    pd.DataFrame(result["e2e_rows"]).to_csv(output_dir / OUTPUT_FILES[2], index=False, encoding="utf-8-sig")
    (output_dir / OUTPUT_FILES[3]).write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = run_validation(args.data_root, args.output_dir)
    print(json.dumps({k: summary[k] for k in ("label", "decision_count", "variants", "failed_checks")},
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

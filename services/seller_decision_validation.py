"""Seller Loss Decision validation: real-data monetary coverage, real component checks and controlled scenarios.

Run: python -m services.seller_decision_validation --data-root C:/VARO_V2_REAL_DATA
Writes (local only, never into git) <data-root>/_SELLER_LOSS_VALIDATION/:
    seller_loss_data_coverage.csv       dataset x decision-field matrix (verified against canonical parquet metadata)
    seller_loss_strategy_contract.json  engine contract (fields, provenance, formulas, statuses, reason codes)
    seller_loss_real_validation.csv     component checks and engine runs on real records (one dataset at a time)
    seller_loss_scenario_validation.csv CONTROLLED SCENARIOS: synthetic unit-test inputs, not real data
    seller_loss_validation_summary.json verdicts, field survey, legacy action structure

Raw and canonical files are read only. No dataset's price, stock or demand is combined with another dataset's.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from services.real_data_adapters import DATA_ROOT, DATASETS
from services.seller_loss_engine import (
    DISCOUNT_SALE, ENGINE_VERSION, NORMAL_SALE, STATUS_FULL, STATUS_PARTIAL, STATUS_UNAVAILABLE, TRANSFER,
    InputField, SellerDecisionInput, contract_document, evaluate_seller_decision, known, units_sold,
)

OUTPUT_FOLDER = "_SELLER_LOSS_VALIDATION"
OUTPUT_FILES = ("seller_loss_data_coverage.csv", "seller_loss_strategy_contract.json", "seller_loss_real_validation.csv",
                "seller_loss_scenario_validation.csv", "seller_loss_validation_summary.json")
DATASET_ORDER = ("suhyup", "jangbogo", "logisall", "nfqs", "aihub", "kamp", "m5", "favorita", "freshretailnet")
COVERAGE_FIELDS = ("inventory", "source_demand", "target_demand", "normal_selling_price", "unit_cost", "transfer_cost",
                   "holding_cost", "discount_rate", "observed_promotion_effect", "disposal_cost", "shelf_life_expiry",
                   "salvage_value")
CELL_STATUSES = ("DIRECT_REAL", "DERIVED_REAL", "PROXY", "UNUSABLE_GRAIN", "MISSING")
REAL_STATUSES = frozenset({"DIRECT_REAL", "DERIVED_REAL"})
MONETARY_FIELDS = ("normal_selling_price", "unit_cost", "transfer_cost", "holding_cost", "disposal_cost", "salvage_value")
# A 3-strategy monetary comparison needs these inside ONE dataset (holding/disposal/salvage are rates the engine can
# carry as bounded unknowns; unit_cost is sunk and never needed).
FULL_REQUIRED = ("inventory", "source_demand", "target_demand", "normal_selling_price", "transfer_cost", "discount_rate",
                 "observed_promotion_effect", "shelf_life_expiry")
VERDICTS = ("FULL_MONETARY", "PARTIAL_MONETARY", "NON_MONETARY_ONLY", "UNAVAILABLE")

M5_STORE, M5_TARGET_STORE = "CA_1", "CA_2"
M5_WINDOW = ("2016-04-25", "2016-05-22")   # the last 28 observed days (d_1914..d_1941)
SAMPLE_SIZE = 50


# ---------------------------------------------------------------- monetary field survey (code + schema + data)

MONETARY_FIELD_SURVEY: tuple[dict[str, str], ...] = (
    {"field": "promotion_analyzer.estimate_unit_cost", "location": "services/legacy_adapters/_local_modules/promotion_analyzer.py",
     "classification": "PROXY", "evidence": "Falls back to price/unit_price as a cost and to a hard-coded 1,000원 when no column exists."},
    {"field": "promotion_analyzer.estimate_daily_holding_cost", "location": "promotion_analyzer.py",
     "classification": "CONFIG", "evidence": "Hard-coded 20원/day default when no holding column exists (placeholder, not data)."},
    {"field": "promotion_discount_rate / promotion_sales_increase_rate", "location": "services/analysis_pipeline.py::_run_promotion",
     "classification": "CONFIG", "evidence": "Workbook config keys; in-code defaults 20% / 80% when absent (placeholder uplift, never observed)."},
    {"field": "promotion_effect / promotion_net_cost / promotion_transfer_cost", "location": "analysis_pipeline._enrich_promotion",
     "classification": "PROXY", "evidence": "Derived from the placeholders above; matched to candidates by display names."},
    {"field": "candidate_generator expected_saving", "location": "services/candidate_generator.py",
     "classification": "PROXY", "evidence": "moved x unit_price(default 1,000) x 0.5 disposal fraction - cost; disabled in real-transport mode (saving 0)."},
    {"field": "transfer_path_analyzer unit_cost / disposal_cost_per_unit / route cost", "location": "transfer_path_analyzer._prepare_*",
     "classification": "CONFIG", "evidence": "Defaults 1,000 / 300 / 1,500 + 900 x km when columns are absent (placeholders)."},
    {"field": "cutline_analyzer disposal_cost_per_unit", "location": "cutline_analyzer.py", "classification": "CONFIG",
     "evidence": "Default 300 when absent."},
    {"field": "eoq_analyzer order_cost / holding_cost_per_unit", "location": "eoq_analyzer.py", "classification": "CONFIG",
     "evidence": "order_cost default 5,000; holding = unit_cost x holding_rate."},
    {"field": "disposal_risk_score", "location": "disposal_risk_analyzer.py", "classification": "PROXY",
     "evidence": "0-100 heuristic (expiry, cover days, inbound age, category keywords); not a probability, never multiplied into money."},
    {"field": "VHS savings/disposal/promotion components", "location": "services/vhs_score_engine.py", "classification": "PROXY",
     "evidence": "Min-max normalised 0-100 scores of expected_saving, avoided_disposal_cost, recovered_margin; not money."},
    {"field": "real transport cost (estimated_cost/transport_cost/move_cost)", "location": "services/real_transport_enrichment.py",
     "classification": "DERIVED_REAL", "evidence": "Official 2026 per-km tariff x OSRM road distance x derived vehicle mix; reference estimate, not an invoice. Vehicle classes priced from the upper official class are PROXY."},
    {"field": "workbook unit_price / unit_cost / disposal_cost_per_unit / daily_holding_cost", "location": "uploaded workbook",
     "classification": "USER_INPUT", "evidence": "Seller-entered values; repository sample workbooks are synthetic (Varo class D/E), not real."},
    {"field": "workbook holding_cost", "location": "uploaded workbook", "classification": "MISSING",
     "evidence": "Period ambiguous (annual unit_price x holding_rate in samples, equal to daily_holding_cost in the network sample); not used."},
    {"field": "product shelf_life_days / expiry_days alias", "location": "column_aliases / data_adapter", "classification": "MISSING",
     "evidence": "Total shelf life aliased onto expiry_days; the engine reads only inventory days_to_expiry as remaining shelf life."},
    {"field": "canonical product_master unit_cost/sell_price/holding_cost/ordering_cost/disposal_cost/shelf_life", "location": "services/canonical_schema.py",
     "classification": "MISSING", "evidence": "Schema columns exist; no dataset populates them (verified from parquet metadata)."},
    {"field": "canonical transfer_network transport_cost / expected_saving", "location": "Suhyup canonical_transfer_network",
     "classification": "DERIVED_REAL", "evidence": "transport_cost KRW DERIVED (benchmark reference engine); expected_saving is 0 in all 620 rows (not a saving)."},
    {"field": "canonical demand_series price", "location": "M5 canonical_demand_series", "classification": "DIRECT_REAL",
     "evidence": "Weekly average sell_price in USD (SOURCE_METADATA) joined to its days; NULL when the item was not sold that week."},
    {"field": "canonical demand_series discount_rate", "location": "FreshRetailNet canonical_demand_series", "classification": "DIRECT_REAL",
     "evidence": "Source 'discount' is a price multiplier (1.0 = none, 0.9 = 10% off), not a rate; money impossible without a price."},
    {"field": "canonical demand_series promotion", "location": "Favorita / FreshRetailNet", "classification": "DIRECT_REAL",
     "evidence": "Boolean flags (onpromotion / activity_flag) without a discount depth."},
)

LEGACY_ACTION_STRUCTURE: tuple[dict[str, str], ...] = (
    {"action": "재고 이동", "producer": "varo_hybrid_score._recommend_action -> normalize_action('재배치 이동')", "kind": "RULE_BASED",
     "rule": "match_grade EXCELLENT/GOOD and (reorder CRITICAL/WARNING or demand INCREASING or vhs_raw >= 70)"},
    {"action": "할인", "producer": "varo_hybrid_score._recommend_action ('할인 판매')", "kind": "RULE_BASED",
     "rule": "disposal grade CRITICAL/HIGH or turnover SLOW/DEAD or disposal_risk_score >= 60"},
    {"action": "폐기", "producer": "varo_hybrid_score._recommend_action", "kind": "RULE_BASED",
     "rule": "disposal CRITICAL and turnover DEAD and ABC C and demand_risk_score < 40"},
    {"action": "보류", "producer": "varo_hybrid_score._recommend_action fallback", "kind": "RULE_BASED", "rule": "none of the above"},
    {"action": "긴급 할인 / 1+1", "producer": "dqn_service.ACTION_LABELS (button-trained DQN only) and alias tables", "kind": "PLACEHOLDER",
     "rule": "Not produced by the production pipeline; DQN reward is a heuristic action matrix; pipeline sets dqn_action='미연결'."},
    {"action": "정상 판매 유지", "producer": "none (closest legacy label: 보류)", "kind": "ABSENT", "rule": "No explicit keep-selling action exists."},
    {"action": "greedy_action", "producer": "uploaded varo_action/greedy_action or default '재고 이동'", "kind": "PLACEHOLDER",
     "rule": "Pass-through label; heuristic_optimizer ranks candidates but never chooses an action."},
    {"action": "promotion_recommended", "producer": "promotion_analyzer.analyze_promotion_vs_transfer", "kind": "RULE_BASED",
     "rule": "transfer_cost <= promotion net cost from placeholder unit cost/holding/discount/uplift -> '재배치 추천' else '프로모션 추천'; feeds VHS promotion_score only."},
    {"action": "varo_final_rank / varo_final_decision", "producer": "vhs_score_engine._rank_varo_operational", "kind": "CALCULATED",
     "rule": "Operational order: larger quantity, lower transport cost, better VHS rank; rank 1 = '최종 추천'. Ranks routes, not actions."},
    {"action": "expected_saving", "producer": "uploaded or candidate_generator", "kind": "PLACEHOLDER",
     "rule": "Uploaded values or default-price/0.5-fraction estimate; 0 in real-transport mode and in all 620 Suhyup candidates."},
)

ALGORITHM_ROLES = {
    "VHS": "candidate priority score (auto-weighted 0-100 of normalised components); vhs_rank",
    "Varo Final": "operational route order: quantity desc, transport cost asc, VHS rank (vhs_score_engine._rank_varo_operational)",
    "Greedy": "heuristic_optimizer score order (cost/quantity/strategy keywords) as a fast baseline ranking",
    "Pareto": "service-quantity / transport-cost / saving frontier, up to 5 routes chosen by ideal-point distance",
    "DQN": "button-trained learned comparison over 8 action labels; never feeds the VHS score",
    "MILP": "lexicographic benchmark (max service, then min cost) under shared constraints; offline validation only",
    "Seller Loss Engine": "per-decision monetary comparison of TRANSFER / NORMAL_SALE / DISCOUNT_SALE for the same quantity; parallel, never replaces the action",
}


# ---------------------------------------------------------------- dataset coverage (claims verified at run time)

def _cell(status: str, evidence: str, checks: Sequence[tuple[str, str, str]] = ()) -> dict[str, Any]:
    """checks: (canonical table, column, 'nonnull' | 'null') verified from parquet metadata."""
    if status not in CELL_STATUSES:
        raise ValueError(status)
    return {"status": status, "evidence": evidence, "checks": list(checks)}


_NO = "MISSING"
DATASET_COVERAGE: dict[str, dict[str, Any]] = {
    "suhyup": {
        "quantity_unit": "UNKNOWN (stock/flow headers carry no unit; parallel kg columns DIRECT)",
        "inventory": _cell("DIRECT_REAL", "Daily center x product stock 재고량.", [("inventory_snapshot", "inventory_qty", "nonnull")]),
        "source_demand": _cell("PROXY", "Daily logistics outbound 출고량, not retail demand.", [("inventory_flow", "outbound_qty", "nonnull")]),
        "target_demand": _cell("PROXY", "Same outbound proxy at the target center.", [("inventory_flow", "outbound_qty", "nonnull")]),
        "normal_selling_price": _cell(_NO, "No price field.", [("product_master", "sell_price", "null")]),
        "unit_cost": _cell(_NO, "No cost field.", [("product_master", "unit_cost", "null")]),
        "transfer_cost": _cell("DERIVED_REAL", "Benchmark move_cost = official tariff x OSRM distance x derived vehicle mix (KRW, DERIVED); per-route proxy share reported in the real validation.",
                               [("transfer_network", "transport_cost", "nonnull")]),
        "holding_cost": _cell(_NO, "None.", [("product_master", "holding_cost", "null")]),
        "discount_rate": _cell(_NO, "None."),
        "observed_promotion_effect": _cell(_NO, "None."),
        "disposal_cost": _cell(_NO, "None.", [("product_master", "disposal_cost", "null")]),
        "shelf_life_expiry": _cell(_NO, "No expiry/shelf-life field (frozen/processed seafood).", [("inventory_snapshot", "expiry_date", "null")]),
        "salvage_value": _cell(_NO, "None."),
    },
    "jangbogo": {
        "quantity_unit": "UNKNOWN",
        "inventory": _cell(_NO, "No stock table (purchase requests and occurrences only)."),
        "source_demand": _cell("DIRECT_REAL", "Warehouse x category monthly sales (category grain, unit UNKNOWN).", [("demand_series", "sales_qty", "nonnull")]),
        "target_demand": _cell("DIRECT_REAL", "Same grain at other warehouses.", [("demand_series", "sales_qty", "nonnull")]),
        "normal_selling_price": _cell(_NO, "None.", [("product_master", "sell_price", "null"), ("demand_series", "price", "null")]),
        "unit_cost": _cell(_NO, "None.", [("product_master", "unit_cost", "null")]),
        "transfer_cost": _cell(_NO, "None."), "holding_cost": _cell(_NO, "None."), "discount_rate": _cell(_NO, "None."),
        "observed_promotion_effect": _cell(_NO, "None."), "disposal_cost": _cell(_NO, "None."),
        "shelf_life_expiry": _cell(_NO, "None."), "salvage_value": _cell(_NO, "None."),
    },
    "logisall": {
        "quantity_unit": "UNKNOWN",
        "inventory": _cell("UNUSABLE_GRAIN", "Monthly national stock only, no location.", [("inventory_snapshot", "inventory_qty", "nonnull")]),
        "source_demand": _cell("DIRECT_REAL", "Zone-daily sales (7 ZIP zones, not stores).", [("demand_series", "sales_qty", "nonnull")]),
        "target_demand": _cell("DIRECT_REAL", "Zone-daily sales at other zones.", [("demand_series", "sales_qty", "nonnull")]),
        "normal_selling_price": _cell(_NO, "None.", [("demand_series", "price", "null")]),
        "unit_cost": _cell(_NO, "None."),
        "transfer_cost": _cell(_NO, "Zone-to-zone distribution volumes only, no cost.", [("transfer_network", "transport_cost", "null")]),
        "holding_cost": _cell(_NO, "None."), "discount_rate": _cell(_NO, "None."), "observed_promotion_effect": _cell(_NO, "None."),
        "disposal_cost": _cell(_NO, "None."), "shelf_life_expiry": _cell(_NO, "None."), "salvage_value": _cell(_NO, "None."),
    },
    "nfqs": {
        "quantity_unit": "ton (SOURCE_METADATA)",
        "inventory": _cell("DIRECT_REAL", "Quarterly region x product-group stock (aggregate of cooperating firms).", [("inventory_snapshot", "inventory_qty", "nonnull")]),
        "source_demand": _cell(_NO, "No sales/outbound."), "target_demand": _cell(_NO, "No sales/outbound."),
        "normal_selling_price": _cell(_NO, "None.", [("product_master", "sell_price", "null")]),
        "unit_cost": _cell(_NO, "None."), "transfer_cost": _cell(_NO, "None."), "holding_cost": _cell(_NO, "None."),
        "discount_rate": _cell(_NO, "None."), "observed_promotion_effect": _cell(_NO, "None."), "disposal_cost": _cell(_NO, "None."),
        "shelf_life_expiry": _cell(_NO, "None."), "salvage_value": _cell(_NO, "None."),
    },
    "aihub": {
        "quantity_unit": "item (SOURCE_METADATA)",
        "inventory": _cell("UNUSABLE_GRAIN", "One unidentified site aggregate per day, no product id.", [("inventory_snapshot", "inventory_qty", "nonnull")]),
        "source_demand": _cell("UNUSABLE_GRAIN", "Site-aggregate outbound items, no product id.", [("inventory_flow", "outbound_qty", "nonnull")]),
        "target_demand": _cell(_NO, "No second location."),
        "normal_selling_price": _cell(_NO, "None.", [("product_master", "sell_price", "null")]),
        "unit_cost": _cell(_NO, "None."), "transfer_cost": _cell(_NO, "None."), "holding_cost": _cell(_NO, "None."),
        "discount_rate": _cell(_NO, "None."), "observed_promotion_effect": _cell(_NO, "None."), "disposal_cost": _cell(_NO, "None."),
        "shelf_life_expiry": _cell(_NO, "None."), "salvage_value": _cell(_NO, "None."),
    },
    "kamp": {
        "quantity_unit": "UNKNOWN",
        "inventory": _cell(_NO, "No stock in the released workbook."),
        "source_demand": _cell("UNUSABLE_GRAIN", "Order-based shipment per rebar grade for one masked project; no shipping location."),
        "target_demand": _cell(_NO, "No location network."),
        "normal_selling_price": _cell(_NO, "None.", [("product_master", "sell_price", "null")]),
        "unit_cost": _cell(_NO, "None."), "transfer_cost": _cell(_NO, "None."), "holding_cost": _cell(_NO, "None."),
        "discount_rate": _cell(_NO, "None."), "observed_promotion_effect": _cell(_NO, "None."), "disposal_cost": _cell(_NO, "None."),
        "shelf_life_expiry": _cell(_NO, "None."), "salvage_value": _cell(_NO, "None."),
    },
    "m5": {
        "quantity_unit": "item (SOURCE_METADATA)",
        "inventory": _cell(_NO, "No on-hand stock released."),
        "source_demand": _cell("DIRECT_REAL", "Observed daily store x item unit sales (censored by unobserved stock-outs).", [("demand_series", "sales_qty", "nonnull")]),
        "target_demand": _cell("DIRECT_REAL", "Same item's unit sales at the other 9 stores.", [("demand_series", "sales_qty", "nonnull")]),
        "normal_selling_price": _cell("DIRECT_REAL", "Weekly average sell_price, USD (SOURCE_METADATA).", [("demand_series", "price", "nonnull")]),
        "unit_cost": _cell(_NO, "None.", [("product_master", "unit_cost", "null")]),
        "transfer_cost": _cell(_NO, "No store-to-store network or cost."),
        "holding_cost": _cell(_NO, "None.", [("product_master", "holding_cost", "null")]),
        "discount_rate": _cell(_NO, "No markdown flag; weekly price changes are unlabelled.", [("demand_series", "discount_rate", "null")]),
        "observed_promotion_effect": _cell(_NO, "No promotion field; SNAP/events are not item promotions.", [("demand_series", "promotion", "null")]),
        "disposal_cost": _cell(_NO, "None."), "shelf_life_expiry": _cell(_NO, "None.", [("product_master", "shelf_life", "null")]),
        "salvage_value": _cell(_NO, "None."),
    },
    "favorita": {
        "quantity_unit": "UNKNOWN (item-dependent count or kg)",
        "inventory": _cell(_NO, "No stock."),
        "source_demand": _cell("DIRECT_REAL", "Daily store x item unit_sales; zero-sales days absent; unit UNKNOWN.", [("demand_series", "sales_qty", "nonnull")]),
        "target_demand": _cell("DIRECT_REAL", "Same item at other stores.", [("demand_series", "sales_qty", "nonnull")]),
        "normal_selling_price": _cell(_NO, "No price released.", [("demand_series", "price", "null")]),
        "unit_cost": _cell(_NO, "None."), "transfer_cost": _cell(_NO, "None."), "holding_cost": _cell(_NO, "None."),
        "discount_rate": _cell(_NO, "onpromotion flag without depth.", [("demand_series", "discount_rate", "null")]),
        "observed_promotion_effect": _cell(_NO, "Promotion flag observed with sales, but no discount depth: no effect at a given rate.",
                                           [("demand_series", "promotion", "nonnull")]),
        "disposal_cost": _cell(_NO, "None."),
        "shelf_life_expiry": _cell(_NO, "Only a perishable flag, no shelf life days.", [("product_master", "perishable", "nonnull"), ("product_master", "shelf_life", "null")]),
        "salvage_value": _cell(_NO, "None."),
    },
    "freshretailnet": {
        "quantity_unit": "normalized_sales_amount (non-physical, coefficient undisclosed)",
        "inventory": _cell(_NO, "Stock level not released (only out-of-stock hours)."),
        "source_demand": _cell("DIRECT_REAL", "Daily normalised sales amount (non-physical unit).", [("demand_series", "sales_qty", "nonnull")]),
        "target_demand": _cell("DIRECT_REAL", "Same product at other stores (normalised).", [("demand_series", "sales_qty", "nonnull")]),
        "normal_selling_price": _cell(_NO, "None.", [("demand_series", "price", "null")]),
        "unit_cost": _cell(_NO, "None."), "transfer_cost": _cell(_NO, "None."), "holding_cost": _cell(_NO, "None."),
        "discount_rate": _cell("DIRECT_REAL", "'discount' price multiplier per day (1.0 = none, 0.9 = 10% off).", [("demand_series", "discount_rate", "nonnull")]),
        "observed_promotion_effect": _cell("DIRECT_REAL", "Discount depth and sales observed on the same rows (association only; normalised units).",
                                           [("demand_series", "discount_rate", "nonnull"), ("demand_series", "sales_qty", "nonnull")]),
        "disposal_cost": _cell(_NO, "None."), "shelf_life_expiry": _cell(_NO, "None."), "salvage_value": _cell(_NO, "None."),
    },
}


def coverage_verdict(cells: Mapping[str, Mapping[str, Any]], quantity_unit: str) -> tuple[str, str]:
    """Rule-based verdict from the cell statuses of ONE dataset (never a union across datasets)."""
    statuses = {name: cells[name]["status"] for name in COVERAGE_FIELDS}
    unit = str(quantity_unit).lower()
    physical_known = not unit.startswith("unknown") and "non-physical" not in unit
    if all(statuses[name] in REAL_STATUSES for name in FULL_REQUIRED) and physical_known:
        return "FULL_MONETARY", "All required fields are real in one dataset with a known physical unit."
    basis = any(statuses[name] in REAL_STATUSES | {"PROXY"} for name in ("inventory", "source_demand"))
    monetary = [name for name in MONETARY_FIELDS if statuses[name] in REAL_STATUSES | {"PROXY"}]
    missing = [name for name in FULL_REQUIRED if statuses[name] not in REAL_STATUSES]
    if not basis:
        return "UNAVAILABLE", "No product-level stock or demand at an identified location."
    if monetary:
        return "PARTIAL_MONETARY", f"Monetary fields {monetary}; missing for FULL: {missing}."
    return "NON_MONETARY_ONLY", f"No monetary field; missing for FULL: {missing}."


def _parquet_nonnull(path: Path, column: str) -> int | None:
    """Non-null count of a column from parquet row-group statistics (no data read); None if not determinable."""
    import pyarrow.parquet as pq

    if not path.exists():
        return None
    meta = pq.ParquetFile(path)
    names = meta.schema_arrow.names
    if column not in names:
        return 0
    if str(meta.schema_arrow.field(column).type) == "null":   # an untyped all-NULL column has no statistics
        return 0
    index = names.index(column)
    nulls = 0
    for group in range(meta.metadata.num_row_groups):
        stats = meta.metadata.row_group(group).column(index).statistics
        if stats is None or not stats.has_null_count:
            return None
        nulls += stats.null_count
    return meta.metadata.num_rows - nulls


def coverage_matrix(data_root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    verdicts: dict[str, dict[str, str]] = {}
    for dataset in DATASET_ORDER:
        spec = DATASET_COVERAGE[dataset]
        folder = data_root / DATASETS[dataset] / "processed"
        cells = {name: spec[name] for name in COVERAGE_FIELDS}
        for name in COVERAGE_FIELDS:
            cell = cells[name]
            outcomes = []
            for table, column, expectation in cell["checks"]:
                count = _parquet_nonnull(folder / f"canonical_{table}.parquet", column)
                ok = None if count is None else (count > 0 if expectation == "nonnull" else count == 0)
                outcomes.append({"table": table, "column": column, "expect": expectation, "nonnull_rows": count, "ok": ok})
            failed = [o for o in outcomes if o["ok"] is False]
            rows.append({
                "dataset": dataset, "field": name, "status": cell["status"], "evidence": cell["evidence"],
                "quantity_unit": spec["quantity_unit"],
                "verified_checks": json.dumps(outcomes, ensure_ascii=False),
                "check_result": "NOT_CHECKED" if not outcomes else ("FAIL" if failed else
                                 "PASS" if all(o["ok"] for o in outcomes) else "UNDETERMINED"),
            })
        verdict, why = coverage_verdict(cells, spec["quantity_unit"])
        verdicts[dataset] = {"verdict": verdict, "reason": why}
        rows.append({"dataset": dataset, "field": "__VERDICT__", "status": verdict, "evidence": why,
                     "quantity_unit": spec["quantity_unit"], "verified_checks": "", "check_result": ""})
    return pd.DataFrame(rows), verdicts


# ---------------------------------------------------------------- real-data checks


@contextmanager
def _env(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _row(dataset: str, check_id: str, component: str, description: str, records: int, metric: str, value: Any,
         expected: Any, status: str, note: str = "") -> dict[str, Any]:
    return {"dataset": dataset, "check_id": check_id, "component": component, "description": description,
            "records": records, "metric": metric, "value": value, "expected": expected, "status": status, "note": note}


def _status_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result["comparison_status"]] = counts.get(result["comparison_status"], 0) + 1
    return dict(sorted(counts.items()))


def _reason_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for code in result["reason_codes"]:
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def suhyup_validation(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from services.real_transport_enrichment import ROOT_ENV, enrich_real_transport
    from services.suhyup_algorithm_revalidation import _inventory_lookup, enrich_inventory_constraints

    candidates = pd.read_csv(data_root / "16_VARO_E2E_20260731/multi_snapshot_validation/suhyup_202607_multi_snapshot_recommendations.csv",
                             dtype={"snapshot_date": str, "route_id": str, "product_id": str, "source_id": str, "target_id": str})
    inventory = pd.read_csv(data_root / DATASETS["suhyup"] / "processed/suhyup_logistics_inventory_flow_actual.csv",
                            dtype={"date": str, "center_code": str, "product_code": str})
    for column in ("recommended_qty", "move_cost", "travel_time_min", "expected_saving"):
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    probe = candidates[["route_id", "snapshot_date", "source_id", "target_id", "product_id", "recommended_qty"]].copy()
    with _env(ROOT_ENV, str(data_root)):
        enriched = enrich_real_transport(probe)
    recomputed = pd.to_numeric(enriched["transport_cost"], errors="coerce")
    matches = int((recomputed - candidates["move_cost"]).abs().le(1e-6).sum())
    proxy_vehicles = pd.to_numeric(enriched["proxy_vehicle_count"], errors="coerce").fillna(0)
    applied = enriched["real_transport_status"].eq("applied")
    constrained = enrich_inventory_constraints(candidates, inventory)
    lookup, _ = _inventory_lookup(inventory)

    results = []
    executable_within_caps = 0
    for index, row in constrained.reset_index(drop=True).iterrows():
        date, product = str(row["snapshot_date"]), str(row["product_id"])
        source = lookup.get((date, str(row["source_id"]), product))
        target = lookup.get((date, str(row["target_id"]), product))
        caps = [row["recommended_qty"], row["source_stock"], row["source_surplus"], row["target_need_7d"]]
        if all(value is not None and not pd.isna(value) for value in caps) and row["recommended_qty"] <= min(caps[1:]) + 1e-9:
            executable_within_caps += 1
        cost_class = "PROXY" if proxy_vehicles.iloc[index] > 0 else "DERIVED_REAL"
        inp = SellerDecisionInput(
            decision_id=f"{date}:{row['route_id']}", product_id=product, source_store_id=str(row["source_id"]),
            target_store_id=str(row["target_id"]), legacy_action=str(row.get("varo_action") or ""),
            decision_qty=known(row["recommended_qty"], "DERIVED_REAL", "benchmark recommended_qty", dataset="suhyup"),
            source_current_stock=known(source["stock"] if source else None, "DIRECT_REAL", "재고량", dataset="suhyup"),
            source_daily_demand=known(source["outbound"] if source else None, "PROXY", "출고량 (logistics outbound, not retail demand)", dataset="suhyup"),
            remaining_shelf_life_days=InputField(source="Suhyup has no expiry/shelf-life field"),
            source_normal_price=InputField(source="Suhyup has no price field"),
            target_current_stock=known(target["stock"] if target else None, "DIRECT_REAL", "재고량", dataset="suhyup"),
            target_daily_demand=known(target["outbound"] if target else None, "PROXY", "출고량", dataset="suhyup"),
            transfer_cost=known(recomputed.iloc[index], cost_class, "official tariff x OSRM distance x vehicle mix", currency="KRW", dataset="suhyup"),
            transfer_cost_basis="QUANTITY_SPECIFIC",
            transfer_cost_qty=known(row["recommended_qty"], "DERIVED_REAL", "benchmark recommended_qty", dataset="suhyup"),
            transit_time_days=known(row["travel_time_min"] / 1440.0, "DERIVED_REAL", "OSRM travel_time_min / 1440"),
            source_surplus_cap=known(row["source_surplus"], "PROXY", "stock - cross-node median", dataset="suhyup"),
            target_need_cap=known(row["target_need_7d"], "PROXY", "median + 7 x outbound - stock", dataset="suhyup"),
        )
        results.append(evaluate_seller_decision(inp))

    rows = [
        _row("suhyup", "S1", "transfer_cost", "Official-tariff transfer cost recomputed from canonical cost matrix, unit weights and vehicle mix equals the benchmark move_cost",
             len(candidates), "exact_matches", matches, len(candidates), "PASS" if matches == len(candidates) else "FAIL",
             "Reference estimate (official tariff), not an invoice."),
        _row("suhyup", "S2", "transfer_cost", "Transfer-cost provenance by vehicle class", len(candidates), "routes_with_upper_class_proxy_vehicles",
             int((proxy_vehicles > 0).sum()), None, "INFO",
             f"applied={int(applied.sum())}; direct-tariff-only routes={int(((proxy_vehicles == 0) & applied).sum())}"),
        _row("suhyup", "S3", "quantity_constraints", "Benchmark recommended_qty within the existing shared caps (source stock, source_surplus, target_need_7d)",
             len(candidates), "within_caps", executable_within_caps, None, "INFO",
             "Caps are the benchmark proxies (median-based surplus, 7-day outbound need); same as the MILP constraints."),
        _row("suhyup", "S4", "engine_status", "Strict engine on Suhyup-only real fields", len(results), "status_counts",
             json.dumps(_status_counts(results)), json.dumps({STATUS_UNAVAILABLE: len(results)}),
             "PASS" if all(r["comparison_status"] == STATUS_UNAVAILABLE for r in results) else "FAIL",
             "No price and no shelf life; outbound demand is a proxy -> no monetary recommendation is fabricated."),
        _row("suhyup", "S5", "engine_reasons", "Reason codes of the strict engine runs", len(results), "reason_counts",
             json.dumps(_reason_counts(results), ensure_ascii=False), None, "INFO"),
        _row("suhyup", "S6", "expected_saving", "Benchmark expected_saving values", len(candidates), "nonzero_rows",
             int(candidates["expected_saving"].fillna(0).ne(0).sum()), 0, "PASS" if candidates["expected_saving"].fillna(0).eq(0).all() else "FAIL",
             "expected_saving is 0 everywhere: there is no saving to compare against."),
    ]
    legacy = {"legacy_action_values": {str(k): int(v) for k, v in candidates["varo_action"].fillna("").value_counts().sort_index().items()},
              "seller_loss_recommendations": sum(1 for r in results if r["recommended_strategy"])}
    return rows, {"status_counts": _status_counts(results), "transfer_cost_exact_matches": matches, "candidates": len(candidates), **legacy}


def m5_validation(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.dataset as ds

    path = data_root / DATASETS["m5"] / "processed/canonical_demand_series.parquet"
    dataset = ds.dataset(path)
    columns = ["date", "location_id", "product_id", "sales_qty", "price", "currency", "currency_status", "unit"]
    window = (ds.field("date") >= M5_WINDOW[0]) & (ds.field("date") <= M5_WINDOW[1])
    frame = dataset.to_table(columns=columns, filter=window & ds.field("location_id").isin([M5_STORE, M5_TARGET_STORE])).to_pandas()
    frame = frame.sort_values(["location_id", "product_id", "date"]).reset_index(drop=True)
    store = frame[frame["location_id"] == M5_STORE]
    priced = store["price"].notna()
    last_week = store[store["date"] >= "2016-05-16"]
    stable = last_week.groupby("product_id").agg(prices=("price", "nunique"), price=("price", "first"),
                                                sales=("sales_qty", "sum"), days=("date", "nunique"))
    stable = stable[(stable["prices"] == 1) & stable["price"].notna() & (stable["days"] == 7)]
    reproduced = 0
    for product, item in stable.iterrows():
        observed = float((last_week.loc[last_week["product_id"] == product, "price"] * last_week.loc[last_week["product_id"] == product, "sales_qty"]).sum())
        # Ex-post check of the NORMAL_SALE valuation: a lot equal to the units that sold, at the realised daily rate.
        sold = units_sold(float(item["sales"]), float(item["sales"]) / 7.0, 7.0)
        if abs(float(item["price"]) * sold - observed) <= 1e-6 * max(1.0, observed):
            reproduced += 1
    weekly = store.dropna(subset=["price"]).copy()
    weekly["week"] = pd.to_datetime(weekly["date"]).dt.to_period("W-FRI")
    weekly = weekly.groupby(["product_id", "week"])["price"].first().reset_index()
    weekly["previous"] = weekly.groupby("product_id")["price"].shift()
    drops = int((weekly["price"] < weekly["previous"] - 1e-9).sum())

    products = sorted(store["product_id"].unique())[:SAMPLE_SIZE]
    target = frame[frame["location_id"] == M5_TARGET_STORE]
    results = []
    for product in products:
        src = store[store["product_id"] == product]
        tgt = target[target["product_id"] == product]
        last_price = src["price"].dropna()
        inp = SellerDecisionInput(
            decision_id=f"M5:{M5_STORE}:{product}", product_id=product, source_store_id=M5_STORE, target_store_id=M5_TARGET_STORE,
            quantity_unit="item",
            decision_qty=InputField(source="M5 has no inventory"), source_current_stock=InputField(source="M5 has no inventory"),
            source_daily_demand=known(src["sales_qty"].mean(), "DERIVED_REAL", "mean observed daily unit sales (28 days)", unit="item", dataset="m5"),
            remaining_shelf_life_days=InputField(source="M5 has no shelf life"),
            source_normal_price=known(last_price.iloc[-1] if len(last_price) else None, "DIRECT_REAL", "weekly sell_price", unit="item", currency="USD", dataset="m5"),
            target_current_stock=InputField(source="M5 has no inventory"),
            target_daily_demand=known(tgt["sales_qty"].mean() if len(tgt) else None, "DERIVED_REAL", "mean observed daily unit sales (28 days)", unit="item", dataset="m5"),
            target_normal_price=known(tgt["price"].dropna().iloc[-1] if tgt["price"].notna().any() else None, "DIRECT_REAL", "weekly sell_price", unit="item", currency="USD", dataset="m5"),
        )
        results.append(evaluate_seller_decision(inp))
    rows = [
        _row("m5", "M1", "normal_selling_price", f"{M5_STORE} rows with an observed weekly price in the last 28 days", int(len(store)),
             "priced_share", round(float(priced.mean()), 4), None, "INFO", "USD, SOURCE_METADATA; NULL means not sold that week."),
        _row("m5", "M2", "opportunity_loss valuation", "Normal-price valuation of a lot equal to the realised 7-day sales reproduces observed dollar sales (stable-price items)",
             int(len(stable)), "reproduced", reproduced, int(len(stable)), "PASS" if reproduced == len(stable) else "FAIL",
             "Arithmetic consistency on real prices; not a forecast test."),
        _row("m5", "M3", "discount_rate", "Week-over-week price decreases (unlabelled)", int(len(weekly)), "price_drop_weeks", drops, None, "INFO",
             "Not usable as a markdown: no promotion label separates markdowns from regular price changes."),
        _row("m5", "M4", "engine_status", "Strict engine on M5-only fields (price + sales)", len(results), "status_counts",
             json.dumps(_status_counts(results)), json.dumps({STATUS_UNAVAILABLE: len(results)}),
             "PASS" if all(r["comparison_status"] == STATUS_UNAVAILABLE for r in results) else "FAIL",
             "No inventory, shelf life, cost or network -> unavailable."),
        _row("m5", "M5", "engine_reasons", "Reason codes", len(results), "reason_counts", json.dumps(_reason_counts(results)), None, "INFO"),
    ]
    return rows, {"status_counts": _status_counts(results), "stable_price_items": int(len(stable)), "reproduced": reproduced}


def freshretailnet_validation(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    path = data_root / DATASETS["freshretailnet"] / "processed/canonical_demand_series.parquet"
    frame = pq.read_table(path, columns=["location_id", "product_id", "date", "sales_qty", "discount_rate", "stockout_hours", "unit"]).to_pandas()
    multiplier = frame["discount_rate"]
    discounted = multiplier.lt(1.0) & multiplier.gt(0.0)
    invalid = int((multiplier.le(0.0) | multiplier.gt(1.0)).sum())
    clean = frame[frame["stockout_hours"].eq(0) & multiplier.gt(0.0) & multiplier.le(1.0)].copy()
    clean["discounted"] = clean["discount_rate"].lt(1.0)
    by_series = clean.groupby(["location_id", "product_id", "discounted"])["sales_qty"].mean().unstack()
    both = by_series.dropna()
    both = both[both[False] > 0]
    ratio = (both[True] / both[False]) if len(both) else pd.Series(dtype=float)

    sample = frame.drop_duplicates(["location_id", "product_id"]).head(SAMPLE_SIZE)
    results = []
    for _, row in sample.iterrows():
        series = frame[(frame["location_id"] == row["location_id"]) & (frame["product_id"] == row["product_id"])]
        observed_depth = 1.0 - float(series["discount_rate"].min())
        inp = SellerDecisionInput(
            decision_id=f"FRN:{row['location_id']}:{row['product_id']}", product_id=str(row["product_id"]),
            source_store_id=str(row["location_id"]), quantity_unit="normalized_sales_amount",
            source_daily_demand=known(series["sales_qty"].mean(), "DERIVED_REAL", "mean normalised daily sales", unit="normalized_sales_amount", dataset="freshretailnet"),
            discount_rate=known(observed_depth if 0 < observed_depth < 1 else None, "DIRECT_REAL", "1 - min(discount multiplier)", dataset="freshretailnet"),
            decision_qty=InputField(source="FRN releases no stock"), source_current_stock=InputField(source="FRN releases no stock"),
            remaining_shelf_life_days=InputField(source="no shelf life"), source_normal_price=InputField(source="no price"),
        )
        results.append(evaluate_seller_decision(inp))
    rows = [
        _row("freshretailnet", "F1", "discount_rate", "Share of rows with a discount multiplier in (0, 1) (engine discount_rate = 1 - multiplier)",
             int(len(frame)), "discounted_share", round(float(discounted.mean()), 4), None, "INFO",
             "Source field is a price multiplier (1.0 = no discount), not a rate."),
        _row("freshretailnet", "F2", "discount_rate", "Rows outside the valid multiplier range (0 or > 1) that the engine must reject",
             int(len(frame)), "invalid_rows", invalid, None, "INFO", "Flagged discount_rate_zero / discount_rate_above_one in canonical."),
        _row("freshretailnet", "F3", "observed_promotion_effect", "Median ratio of mean sales on discounted vs undiscounted in-stock days per series",
             int(len(both)), "median_sales_ratio", round(float(ratio.median()), 4) if len(ratio) else None, None, "INFO",
             "Association only (no causal identification, normalised units); never used as an engine uplift."),
        _row("freshretailnet", "F4", "engine_status", "Strict engine on FRN-only fields", len(results), "status_counts",
             json.dumps(_status_counts(results)), json.dumps({STATUS_UNAVAILABLE: len(results)}),
             "PASS" if all(r["comparison_status"] == STATUS_UNAVAILABLE and "NON_PHYSICAL_UNIT" in r["reason_codes"] for r in results) else "FAIL",
             "Normalised (non-physical) quantity and no price/stock -> unavailable."),
    ]
    return rows, {"status_counts": _status_counts(results), "series_with_both_regimes": int(len(both))}


def metadata_validation(data_root: Path, verdicts: Mapping[str, Mapping[str, str]]) -> list[dict[str, Any]]:
    """Datasets without any monetary field: confirm from metadata that no monetary column is populated."""
    rows = []
    money_columns = {"product_master": ("sell_price", "unit_cost", "holding_cost", "disposal_cost", "ordering_cost"),
                     "demand_series": ("price",), "transfer_network": ("transport_cost",)}
    for dataset in ("jangbogo", "logisall", "nfqs", "aihub", "kamp", "favorita"):
        folder = data_root / DATASETS[dataset] / "processed"
        populated = []
        for table, columns in money_columns.items():
            for column in columns:
                count = _parquet_nonnull(folder / f"canonical_{table}.parquet", column)
                if count:
                    populated.append(f"{table}.{column}={count}")
        rows.append(_row(dataset, "X1", "monetary_fields", "Populated monetary canonical columns", 0, "populated", "|".join(populated) or "none",
                         "none", "PASS" if not populated else "FAIL", f"verdict {verdicts[dataset]['verdict']}"))
    return rows


# ---------------------------------------------------------------- controlled scenarios (synthetic, unit-test inputs)

SCENARIO_DATASET = "CONTROLLED_SCENARIO"


def scenario_input(**overrides: Any) -> SellerDecisionInput:
    """Base controlled scenario (synthetic test values, KRW, unit 'ea'); overrides replace whole fields."""
    def q(value: float, source: str) -> InputField:
        return known(value, "USER_INPUT", source, dataset=SCENARIO_DATASET)

    def m(value: float, source: str) -> InputField:
        return known(value, "USER_INPUT", source, currency="KRW", dataset=SCENARIO_DATASET)

    base: dict[str, Any] = dict(
        decision_id="SCENARIO", product_id="P-TEST", source_store_id="S-SRC", target_store_id="S-TGT", quantity_unit="ea",
        legacy_action="보류",
        decision_qty=q(20, "scenario decision_qty"), source_current_stock=q(50, "scenario stock"),
        source_daily_demand=q(3, "scenario source demand"), remaining_shelf_life_days=q(10, "scenario shelf life"),
        source_normal_price=m(1000, "scenario price"), unit_cost=m(600, "scenario unit cost"),
        source_holding_cost_per_unit_day=m(5, "scenario holding"), source_disposal_cost_per_unit=m(200, "scenario disposal"),
        salvage_value_per_unit=m(0, "scenario salvage"),
        discount_rate=known(0.3, "CONFIG", "scenario discount 30%"), promotion_uplift=known(0.5, "CONFIG", "scenario uplift 50%"),
        target_current_stock=q(5, "scenario target stock"), target_daily_demand=q(4, "scenario target demand"),
        target_normal_price=m(1000, "scenario target price"), target_holding_cost_per_unit_day=m(5, "scenario target holding"),
        target_disposal_cost_per_unit=m(200, "scenario target disposal"),
        transfer_cost=known(1800, "USER_INPUT", "scenario trip cost", currency="KRW"), transfer_cost_basis="FIXED_PER_TRIP",
        transit_time_days=known(0.1, "USER_INPUT", "scenario transit"),
    )
    base.update(overrides)
    return SellerDecisionInput(**base)


def controlled_scenarios() -> list[dict[str, Any]]:
    m = lambda value, source="override": known(value, "USER_INPUT", source, currency="KRW", dataset=SCENARIO_DATASET)  # noqa: E731
    q = lambda value, source="override": known(value, "USER_INPUT", source, dataset=SCENARIO_DATASET)  # noqa: E731
    return [
        {"id": "A", "description": "Cheap transfer, large target demand", "input": scenario_input(),
         "expect": {"comparison_status": STATUS_FULL, "recommended_strategy": TRANSFER}},
        {"id": "B", "description": "Source demand covers the stock, transfer expensive",
         "input": scenario_input(source_daily_demand=q(6), transfer_cost=known(50000, "USER_INPUT", "expensive trip", currency="KRW")),
         "expect": {"comparison_status": STATUS_FULL, "recommended_strategy": NORMAL_SALE}},
        {"id": "C", "description": "Expiry in 2 days, observed uplift 150% at 30% off",
         "input": scenario_input(decision_qty=q(20), source_current_stock=q(20), remaining_shelf_life_days=q(2),
                                 promotion_uplift=known(1.5, "USER_INPUT", "seller-observed uplift")),
         "expect": {"comparison_status": STATUS_FULL, "recommended_strategy": DISCOUNT_SALE}},
        {"id": "D", "description": "Normal price missing", "input": scenario_input(source_normal_price=InputField(source="absent")),
         "expect": {"comparison_status": STATUS_UNAVAILABLE, "recommended_strategy": None}},
        {"id": "E", "description": "No promotion uplift evidence", "input": scenario_input(promotion_uplift=InputField(source="absent")),
         "expect": {"comparison_status": STATUS_PARTIAL, "recommended_strategy": TRANSFER, "unavailable": DISCOUNT_SALE}},
        {"id": "F", "description": "Target demand missing", "input": scenario_input(target_daily_demand=InputField(source="absent")),
         "expect": {"comparison_status": STATUS_PARTIAL, "recommended_strategy": DISCOUNT_SALE, "unavailable": TRANSFER}},
        {"id": "G", "description": "Source holding cost unknown but the ranking is robust",
         "input": scenario_input(source_holding_cost_per_unit_day=InputField(source="absent")),
         "expect": {"comparison_status": STATUS_FULL, "recommended_strategy": TRANSFER, "recommendation_status": "RECOMMENDED"}},
        {"id": "H", "description": "Holding cost unknown at both stores: ranking depends on it",
         "input": scenario_input(source_holding_cost_per_unit_day=InputField(source="absent"),
                                 target_holding_cost_per_unit_day=InputField(source="absent")),
         "expect": {"comparison_status": STATUS_FULL, "recommended_strategy": None, "recommendation_status": "NOT_ROBUST_TO_UNKNOWN_INPUTS"}},
        {"id": "I", "description": "Zero source demand and no target: keep and discount tie",
         "input": scenario_input(source_daily_demand=q(0), target_store_id=None),
         "expect": {"comparison_status": STATUS_PARTIAL, "recommended_strategy": NORMAL_SALE, "tie": True}},
        {"id": "J", "description": "Quantity unit mismatch (ea vs kg)", "input": scenario_input(source_current_stock=known(50, "USER_INPUT", "stock kg", unit="kg", dataset=SCENARIO_DATASET)),
         "expect": {"comparison_status": STATUS_UNAVAILABLE, "recommended_strategy": None, "reason": "UNIT_MISMATCH"}},
        {"id": "K", "description": "Currency mismatch (disposal cost in USD)",
         "input": scenario_input(source_disposal_cost_per_unit=known(0.15, "USER_INPUT", "usd", currency="USD", dataset=SCENARIO_DATASET)),
         "expect": {"comparison_status": STATUS_UNAVAILABLE, "recommended_strategy": None, "reason": "CURRENCY_MISMATCH"}},
        {"id": "L", "description": "Source demand is a proxy (logistics outbound)",
         "input": scenario_input(source_daily_demand=known(3, "PROXY", "outbound", dataset=SCENARIO_DATASET)),
         "expect": {"comparison_status": STATUS_UNAVAILABLE, "recommended_strategy": None, "reason": "PROXY_REJECTED:source_daily_demand"}},
        {"id": "M", "description": "Price from another dataset", "input": scenario_input(source_normal_price=known(1000, "DIRECT_REAL", "other", currency="KRW", dataset="OTHER_DATASET")),
         "expect": {"comparison_status": STATUS_UNAVAILABLE, "recommended_strategy": None, "reason": "CROSS_DATASET_INPUT"}},
        {"id": "N", "description": "Non-physical quantity unit", "input": scenario_input(quantity_unit="normalized_sales_amount"),
         "expect": {"comparison_status": STATUS_UNAVAILABLE, "recommended_strategy": None, "reason": "NON_PHYSICAL_UNIT"}},
        {"id": "O", "description": "Zero inventory", "input": scenario_input(decision_qty=q(0), source_current_stock=q(0)),
         "expect": {"comparison_status": STATUS_UNAVAILABLE, "recommended_strategy": None, "reason": "ZERO_INVENTORY"}},
        {"id": "P", "description": "Remaining shelf life missing", "input": scenario_input(remaining_shelf_life_days=InputField(source="absent")),
         "expect": {"comparison_status": STATUS_UNAVAILABLE, "recommended_strategy": None, "reason": "SHELF_LIFE_MISSING"}},
        {"id": "Q", "description": "Negative source demand (bad sales data)", "input": scenario_input(source_daily_demand=q(-2)),
         "expect": {"comparison_status": STATUS_UNAVAILABLE, "recommended_strategy": None, "reason": "NEGATIVE_INPUT:source_daily_demand"}},
        {"id": "R", "description": "Target capped below the decision quantity; quantity-specific cost only an upper bound",
         "input": scenario_input(target_current_stock=q(30), transfer_cost=known(1800, "USER_INPUT", "priced for 20", currency="KRW"),
                                 transfer_cost_basis="QUANTITY_SPECIFIC", transfer_cost_qty=q(20)),
         "expect": {"comparison_status": STATUS_FULL, "recommended_strategy": TRANSFER, "reason": "TRANSFER_COST_UPPER_BOUND_ONLY"}},
    ]


def run_scenarios() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows, results = [], []
    for scenario in controlled_scenarios():
        result = evaluate_seller_decision(scenario["input"])
        expect = scenario["expect"]
        checks = {key: result.get(key) == value for key, value in expect.items() if key not in ("reason", "unavailable")}
        if "reason" in expect:
            checks["reason"] = expect["reason"] in result["reason_codes"] or any(
                expect["reason"] in reasons for reasons in result["unavailable_strategies"].values())
        if "unavailable" in expect:
            checks["unavailable"] = expect["unavailable"] in result["unavailable_strategies"]
        results.append(result)
        rows.append({
            "data_kind": "CONTROLLED_SCENARIO (synthetic test input, not real data)",
            "scenario_id": scenario["id"], "description": scenario["description"],
            "expected": json.dumps(expect, ensure_ascii=False), "comparison_status": result["comparison_status"],
            "recommendation_status": result["recommendation_status"], "recommended_strategy": result["recommended_strategy"],
            "expected_loss_transfer": result["expected_loss_transfer"], "expected_loss_normal_sale": result["expected_loss_normal_sale"],
            "expected_loss_discount_sale": result["expected_loss_discount_sale"],
            "loss_difference_vs_second_best": result["loss_difference_vs_second_best"],
            "unknown_inputs": "|".join(result["unknown_inputs"]), "reason_codes": "|".join(result["reason_codes"]),
            "explanation": result["explanation"], "pass": all(checks.values()),
        })
    return pd.DataFrame(rows), results


# ---------------------------------------------------------------- bundle


def run_validation(data_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    data_root = Path(data_root)
    output_dir = Path(output_dir) if output_dir else data_root / OUTPUT_FOLDER
    coverage, verdicts = coverage_matrix(data_root)
    real_rows: list[dict[str, Any]] = []
    suhyup_rows, suhyup = suhyup_validation(data_root)
    m5_rows, m5 = m5_validation(data_root)
    frn_rows, frn = freshretailnet_validation(data_root)
    real_rows.extend(suhyup_rows + m5_rows + frn_rows + metadata_validation(data_root, verdicts))
    real = pd.DataFrame(real_rows)
    scenarios, _ = run_scenarios()
    contract = contract_document()
    full = [name for name, item in verdicts.items() if item["verdict"] == "FULL_MONETARY"]
    summary = {
        "engine_version": ENGINE_VERSION,
        "contract_signature": contract["contract_signature"],
        "full_monetary_datasets": full,
        "full_monetary_available": bool(full),
        "dataset_verdicts": verdicts,
        "coverage_check_failures": int((coverage["check_result"] == "FAIL").sum()),
        "real_validation": {"suhyup": suhyup, "m5": m5, "freshretailnet": frn,
                            "failed_checks": real.loc[real["status"] == "FAIL", "check_id"].tolist()},
        "scenario_validation": {"data_kind": "CONTROLLED_SCENARIO (synthetic test input, not real data)",
                                "scenarios": int(len(scenarios)), "passed": int(scenarios["pass"].sum())},
        "monetary_field_survey": list(MONETARY_FIELD_SURVEY),
        "legacy_action_structure": list(LEGACY_ACTION_STRUCTURE),
        "algorithm_roles": ALGORITHM_ROLES,
        "production_action_replaced": False,
        "note": "Cross-dataset fields are never combined; controlled scenarios are synthetic unit-test inputs.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / OUTPUT_FILES[0], index=False, encoding="utf-8-sig")
    (output_dir / OUTPUT_FILES[1]).write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    real.to_csv(output_dir / OUTPUT_FILES[2], index=False, encoding="utf-8-sig")
    scenarios.to_csv(output_dir / OUTPUT_FILES[3], index=False, encoding="utf-8-sig")
    (output_dir / OUTPUT_FILES[4]).write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = run_validation(args.data_root, args.output_dir)
    print(json.dumps({k: summary[k] for k in ("engine_version", "full_monetary_datasets", "dataset_verdicts",
                                              "coverage_check_failures", "scenario_validation")}, ensure_ascii=False, indent=2))
    print(json.dumps(summary["real_validation"], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

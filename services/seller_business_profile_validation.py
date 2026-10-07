"""Real Suhyup staged wiring validation with explicitly labelled TEST USER INPUT.

Writes only requested reports; no actual prices/demands are fabricated. Counts
describe this test seller's explicit sharing policy, not observed seller costs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from services import seller_business_profile as bp, seller_loss_inputs as sli, seller_loss_input_requirements as req
from services.seller_loss_input_validation import load_suhyup_upload, _run, _sha256, PROCESSED, PROCESSED_FILES
from services.seller_loss_requirement_validation import TestSeller
from services.seller_decision_validation import run_scenarios

LABEL = "TEST USER INPUT - reuse demonstration, not observed Suhyup business values"


def _analysis(state):
    return state["pipeline_result"]["seller_loss_analysis"]


def _items(state):
    return [i for d in _analysis(state)["decisions"] for i in d["input_requirements"]["required_user_inputs"]]


def _distribution(values):
    return dict(sorted(Counter(values).items()))


def run_validation(data_root: Path, output_dir: Path | None = None):
    root = Path(data_root)
    upload = load_suhyup_upload(root)
    paths = [root / PROCESSED / n for n in PROCESSED_FILES]
    hashes = {str(p): _sha256(p) for p in paths}
    date = str(pd.to_datetime(upload["inventory"].snapshot_date).max().date())
    reports, counts, audit_rows, summaries = [], [], [], {}
    for scope in (req.TRANSFER_VS_NORMAL, req.ALL_THREE):
        data = {**upload, req.COMPARISON_SCOPE_KEY: scope}
        a = _run(data, root)
        seller = TestSeller(_analysis(a)["decisions"])
        profiles = {}

        def add_policy(item):
            entry = item["seller_entry"]
            col, keys = entry["column"], entry["keys"]
            if col in {"daily_demand", "remaining_shelf_life_days", "promotion_uplift"}:
                return False
            if col in sli.ROUTE_FIELDS:
                target_scope, target_keys = "ROUTE", keys
            elif col in {"holding_cost_per_unit_day", "disposal_cost_per_unit"}:
                target_scope, target_keys = "STORE", {"store_id": keys["store_id"]}
            else:
                target_scope, target_keys = "PRODUCT", {"product_id": keys["product_id"]}
            key = (target_scope, tuple(sorted(target_keys.items())), col)
            profiles[key] = {"scope": target_scope, **target_keys, col: seller.value(entry),
                             "effective_date": date, "effective_until": date,
                             "profile_source": LABEL, "note": LABEL}
            return True

        def profile_frame():
            return pd.DataFrame([{"scope": "GLOBAL", "currency": "KRW", "note": LABEL}, *profiles.values()])

        def execute(with_decision):
            value = {**data, bp.SHEET_KEY: profile_frame()}
            if with_decision and seller.frame() is not None:
                frame = seller.frame().copy()
                frame["effective_date"] = date
                value[sli.SHEET_KEY] = frame
            return _run(value, root)

        for item in _items(a):
            add_policy(item)
        b = execute(False)
        c = b
        for _ in range(12):
            items = _items(c)
            if not items:
                break
            decision_items = [i for i in items if not add_policy(i)]
            seller.answer(decision_items)
            c = execute(True)
        if _items(c):
            raise AssertionError("Profile staged flow did not reach stop asking")
        repeat = execute(False)  # only common inputs remain; temporal values must be re-entered
        assert execute(True)["pipeline_result"]["seller_loss_analysis"] == _analysis(c)
        for stage, state in (("A_NO_PROFILE", a), ("B_INITIAL_PROFILE", b), ("C_DECISION_INPUTS", c), ("REPEAT_PROFILE_ONLY", repeat)):
            assert state["recommendations"] == a["recommendations"]
            assert state["pipeline_result"]["summary"] == a["pipeline_result"]["summary"]
            for d in _analysis(state)["decisions"]:
                assert not d["production_action_applied"] and not d["legacy_action_changed"]
                plan = d["input_requirements"]
                assert req.check_plan_consistency(plan, d) == []
                if stage == "C_DECISION_INPUTS":
                    assert d["recommendation_readiness"] == "RECOMMENDABLE"
                    assert not plan["required_user_inputs"]
                reports.append({"label": LABEL, "scope": scope, "stage": stage, "decision_id": d["decision_id"],
                    "required_fields": plan["required_input_count"], "required_entries": plan["required_entry_count"],
                    "requested": "|".join(i["field"] for i in plan["required_user_inputs"]),
                    "readiness": d["recommendation_readiness"], "comparison_status": d["comparison_status"],
                    "loss_transfer": d["expected_loss_transfer"], "loss_normal": d["expected_loss_normal_sale"],
                    "loss_discount": d["expected_loss_discount_sale"], "action_applied": d["production_action_applied"]})
                for field, source in d["input_sources"].items():
                    if source["origin"] == "SELLER_BUSINESS_PROFILE":
                        assert source["provenance"] == "USER_INPUT" and source["profile_source"]
                        audit_rows.append({"scope": scope, "stage": stage, "decision_id": d["decision_id"],
                                           "field": field, **source})
        setup_scopes = Counter(row["scope"] for row in profiles.values())
        setup_keys = {}
        for row in profiles.values():
            key = row.get("product_id") if row["scope"] == "PRODUCT" else row.get("store_id") if row["scope"] == "STORE" else str(row)
            setup_keys.setdefault(row["scope"], Counter())[key] += 1
        first_decisions = {d["decision_id"]: d for d in _analysis(a)["decisions"]}
        for d in _analysis(repeat)["decisions"]:
            counts.append({"label": LABEL, "scope": scope, "decision_id": d["decision_id"],
                "no_profile_fields": first_decisions[d["decision_id"]]["input_requirements"]["required_input_count"],
                "repeat_decision_fields": d["input_requirements"]["required_input_count"],
                "repeat_decision_entries": d["input_requirements"]["required_entry_count"],
                "setup_global": 1, "setup_total_shared_value_entries": 1 + len(profiles)})
        summaries[scope] = {"decisions": len(first_decisions), "setup_global_currency": 1,
            "setup_entries_by_scope": dict(setup_scopes),
            "setup_inputs_per_entity": {s: dict(c) for s,c in setup_keys.items()},
            "setup_shared_value_entries": 1 + len(profiles),
            "setup_key_and_validity_metadata_not_counted_as_value_entries": True,
            "decision_value_entries_deduplicated_across_candidates": len(seller.rows),
            "first_batch_total_value_entries": 1 + len(profiles) + len(seller.rows),
            "no_profile_fields": _distribution(d["input_requirements"]["required_input_count"] for d in _analysis(a)["decisions"]),
            "initial_profile_fields": _distribution(d["input_requirements"]["required_input_count"] for d in _analysis(b)["decisions"]),
            "repeat_decision_fields": _distribution(d["input_requirements"]["required_input_count"] for d in _analysis(repeat)["decisions"]),
            "recommendable_after_decision_inputs": _analysis(c)["recommendable_count"],
            "profile_rows_TEST_ONLY": profile_frame().fillna("").to_dict("records")}
    assert hashes == {str(p): _sha256(p) for p in paths}
    scenarios, _ = run_scenarios()
    summary = {"label": LABEL, "contract": bp.contract_document(), "scopes": summaries,
        "checks": {"all_staged_checks_passed": True, "source_hashes_unchanged": True,
                   "controlled_scenarios_passed": int(scenarios["pass"].sum()), "controlled_scenarios_total": len(scenarios)},
        "input_hashes": hashes, "limitations": "Same-day explicit TEST profile values; future dates require renewed policies/time windows. Counts are numeric/text value entries, excluding key/date/source metadata. Production action unchanged."}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit_rows).to_csv(output / "seller_business_profile_validation.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(counts).to_csv(output / "seller_loss_first_vs_repeat_input_counts.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(reports).to_csv(output / "seller_loss_profile_suhyup_e2e.csv", index=False, encoding="utf-8-sig")
        (output / "seller_loss_profile_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("C:/VARO_V2_REAL_DATA"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run_validation(args.data_root, args.output_dir or args.data_root / "_SELLER_LOSS_VALIDATION")
    print(json.dumps({"checks": result["checks"], "scopes": {k: {f:v for f,v in s.items() if f != "profile_rows_TEST_ONLY"} for k,s in result["scopes"].items()}}, ensure_ascii=False, indent=2))

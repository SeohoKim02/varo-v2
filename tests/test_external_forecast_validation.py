"""External generalisation check of the frozen forecast router (services.external_forecast_validation), on small panels."""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from services import demand_forecast_router as router
from services import external_forecast_validation as ev
from services.demand_history_bridge import canonical_to_daily_history
from services.real_data_adapters import DATASETS

FRN_TRAIN, FRN_EVAL = ev.PROTOCOLS["freshretailnet"]["train_file"], ev.PROTOCOLS["freshretailnet"]["eval_file"]


def canonical_frame(rows):
    frame = pd.DataFrame(rows)
    frame.insert(0, "source_row_id", [str(i + 1) for i in range(len(frame))])
    return frame


def frn_like(stores=3, products=8, seed=5):
    """FreshRetailNet-shaped canonical rows: 90 train + 7 eval days, a row every day (zeros recorded), stock-out hours."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-03-28", periods=97)
    rows = []
    for s in range(stores):
        for p in range(products):
            q = rng.poisson(rng.uniform(2, 9), len(days)).astype(float)
            if p == 0:
                q[:80] = 0.0                               # first sale 10 days before the final cutoff: cold start
            elif p == 1:
                q = np.where(np.arange(len(days)) % 15 == 0, 2.0, 0.0)      # mostly zero
            elif p == 2:
                q = np.where(rng.random(len(days)) < 0.3, q, 0.0)            # intermittent
            for d, value in zip(days, q):
                rows.append({"source_dataset": "freshretailnet", "source_file": FRN_TRAIN if d <= days[89] else FRN_EVAL,
                             "date_grain": "daily", "product_grain": "product", "date": d.strftime("%Y-%m-%d"), "location_id": str(s),
                             "product_id": str(100 + p), "category": "A" if p % 2 else "B", "sales_qty": value,
                             "unit": "normalized_sales_amount", "unit_status": "SOURCE_METADATA", "validation_flags": "",
                             "location_id_namespace": "FreshRetailNet:store_id", "product_id_namespace": "FreshRetailNet:product_id",
                             "stockout_hours": float(rng.integers(0, 3))})
    return canonical_frame(rows)


def favorita_like(stores=2, products=7, seed=9):
    """Favorita-shaped canonical rows: rows only on selling days (no zero rows), signed sales with documented returns."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(end="2017-08-15", periods=160)
    rows = []
    for s in range(stores):
        for p in range(products):
            q = rng.poisson(rng.uniform(4, 10), len(days)).astype(float)
            if p == 0:
                q[120] = -3.0                              # a documented return
            elif p == 1:
                q[:-20] = 0.0                              # new item
            elif p == 2:
                q = np.where(rng.random(len(days)) < 0.2, q, 0.0)
            for d, value in zip(days, q):
                if value == 0:
                    continue                               # Favorita records no zero-sale row
                rows.append({"source_dataset": "favorita", "source_file": "27_FAVORITA/raw/train.csv", "date_grain": "daily",
                             "product_grain": "product", "date": d.strftime("%Y-%m-%d"), "location_id": str(s + 1),
                             "product_id": str(9000 + p), "category": "GROCERY I" if p % 2 else "DAIRY", "sales_qty": value,
                             "unit": None, "unit_status": "UNKNOWN",
                             "validation_flags": "negative:sales_qty|documented_return" if value < 0 else "",
                             "location_id_namespace": "Favorita:store_nbr", "product_id_namespace": "Favorita:item_nbr",
                             "stockout_hours": np.nan})
    return canonical_frame(rows)


def write_dataset(root, dataset, frame, semantic_checks, row_group_size=None):
    folder = root / DATASETS[dataset]
    (folder / "processed").mkdir(parents=True, exist_ok=True)
    (folder / "results").mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / "processed" / "canonical_demand_series.parquet", index=False, row_group_size=row_group_size)
    (folder / "results" / "data_quality_report.json").write_text(json.dumps({"semantic_checks": semantic_checks}), encoding="utf-8")


FRN_QUALITY = {"provenance_and_realness": {"classification": "REAL_SALES: synthetic test rows"}, "stockout_semantics": {"inventory": "n/a"}}
FAVORITA_QUALITY = {"signed_sales_investigation": {"negative_rows": 2, "specification": "Negative values of unit_sales represent returns."},
                    "zero_sales_absence": {"zero_rows": 0}}


@pytest.fixture()
def frn_root(tmp_path):
    write_dataset(tmp_path, "freshretailnet", frn_like(), FRN_QUALITY)
    return tmp_path


def run_frn(root, out="frn", **kwargs):
    return ev.run_dataset("freshretailnet", root, root / out, root / "common", **kwargs)


# ---------------------------------------------------------------- frozen configuration and gate


def test_frozen_config_signature_unchanged():
    check = ev.frozen_config_check()
    assert check["FROZEN_CONFIG_UNCHANGED"] and check["differences"] == []
    record = check["record"]
    assert record["v2_config_signature"] == record["v2_config_signature_pinned_in_router"] == router.V2_CONFIG_SIGNATURE
    assert record["v2_config_signature"] == ev.FROZEN_REFERENCE["v2_config_signature"]
    assert record["v2_routes"] == ev.FROZEN_REFERENCE["v2_routes"] and record["v1_fingerprint_matches_baseline"]
    tampered = json.loads(json.dumps(record))
    tampered["router"]["min_sales_age_days"] = 14
    assert ev.frozen_config_check(tampered)["differences"] == ["router"]


def test_external_gate_is_frozen_before_any_series_is_scored(frn_root, monkeypatch):
    freeze_file = frn_root / "common" / ev.GATE_FREEZE_FILE
    original = ev.evaluate_origin
    seen = []

    def guarded(*args, **kwargs):
        seen.append(freeze_file.exists())
        return original(*args, **kwargs)

    monkeypatch.setattr(ev, "evaluate_origin", guarded)
    report = run_frn(frn_root)
    assert seen and all(seen)
    frozen = json.loads(freeze_file.read_text(encoding="utf-8"))
    assert frozen["gate_signature"] == ev.gate_signature() == report["gate"]["gate_signature"]
    assert frozen["gate"] == json.loads(json.dumps(ev.EXTERNAL_GATE))
    monkeypatch.setitem(ev.EXTERNAL_GATE["coverage"], "min_share", 0.5)          # a "result-driven" change of the gate
    with pytest.raises(ev.FreezeViolation):
        run_frn(frn_root, out="again")
    assert not (frn_root / "again").exists()


def test_gate_signature_covers_the_code_that_applies_it():
    names = {f.__name__ for f in (ev.classify_semantics, ev.evaluate_dataset_gate, ev.combined_verdict, ev.window_stats)}
    assert all(name in inspect.getsource(ev.gate_signature) for name in names)


# ---------------------------------------------------------------- semantics, gate and verdict rules


def gate_evidence(v1=0.40, routed=0.38, origins=((0.40, 0.38), (0.41, 0.39), (0.42, 0.43)), coverage=0.5, semantics="PARTIAL",
                  bias=(0.01, 0.015), integrity=True, high=(0.3, 0.29, 500, 0.6)):
    return {"integrity": {"frozen_config": integrity, "no_leakage": True, "fallback_policy": True, "train_eval_separation": None},
            "total_7d": {"v1": {"wape": v1, "bias_pct": bias[0]}, "router": {"wape": routed, "bias_pct": bias[1]}},
            "daily_h1_7": {"v1": {"wape": v1 + 0.1, "bias_pct": 0.0}, "router": {"wape": routed + 0.1, "bias_pct": 0.0}},
            "demand_type": {"high_frequency": dict(zip(("v1_wape", "router_wape", "scored", "actual_share"), high)),
                            "medium_frequency": {"v1_wape": 0.5, "router_wape": 0.6, "scored": 20, "actual_share": 0.2}},
            "origins": [{"origin": i, "v1_wape": a, "router_wape": b} for i, (a, b) in enumerate(origins)],
            "coverage_share": coverage, "semantics": semantics}


def test_dataset_verdict_rules():
    partial = ev.evaluate_dataset_gate(gate_evidence())
    assert partial["verdict"] == "PASS_WITH_LIMITATION" and partial["improved"] and partial["limitations"] == ["quantity semantics PARTIAL"]
    medium = next(c for c in partial["criteria"] if c["criterion"] == "demand_type:medium_frequency")
    assert medium["pass"] is None and "not gated" in medium["note"]                 # 20 series-weeks: reported, not gated
    assert ev.evaluate_dataset_gate(gate_evidence(semantics="FULL"))["verdict"] == "PASS"
    assert ev.evaluate_dataset_gate(gate_evidence(semantics="FULL", coverage=0.1))["verdict"] == "PASS_WITH_LIMITATION"
    worse = ev.evaluate_dataset_gate(gate_evidence(routed=0.405, origins=((0.40, 0.41),)))
    assert worse["verdict"] == "FAIL" and worse["failed"] == ["total_7d_wape", "daily_wape"]
    neutral = ev.evaluate_dataset_gate(gate_evidence(routed=0.398))
    assert neutral["verdict"] == "PASS_WITH_LIMITATION" and not neutral["improved"]
    assert ev.evaluate_dataset_gate(gate_evidence(bias=(0.01, -0.03)))["failed"] == ["bias"]
    assert ev.evaluate_dataset_gate(gate_evidence(bias=(0.05, 0.045)))["failed"] == []          # within |V1 bias|
    assert ev.evaluate_dataset_gate(gate_evidence(high=(0.30, 0.32, 500, 0.6)))["failed"] == ["demand_type:high_frequency"]
    assert ev.evaluate_dataset_gate(gate_evidence(integrity=False))["verdict"] == "FAIL"
    no_majority = ev.evaluate_dataset_gate(gate_evidence(origins=((0.40, 0.38), (0.40, 0.41), (0.40, 0.42))))
    assert not no_majority["improved"]
    with pytest.raises(ValueError):
        ev.evaluate_dataset_gate(gate_evidence(semantics="UNAVAILABLE"))


def test_combined_verdict_rules():
    ok, flat, bad = ({"verdict": "PASS_WITH_LIMITATION", "improved": True}, {"verdict": "PASS_WITH_LIMITATION", "improved": False},
                     {"verdict": "FAIL", "improved": True})
    assert ev.combined_verdict({"a": ok, "b": ok}) == "GENERALIZATION_PASS"
    assert ev.combined_verdict({"a": ok, "b": flat}) == "PARTIAL"
    assert ev.combined_verdict({"a": flat, "b": flat}) == "FAIL"
    assert ev.combined_verdict({"a": ok, "b": bad}) == "FAIL"


def test_quantity_semantics_are_measured_not_assumed():
    import pyarrow as pa
    for frame, observed, complete, expected in ((frn_like(1, 3), True, True, "PARTIAL"), (favorita_like(1, 3), True, False, "PARTIAL"),
                                                (frn_like(1, 3), False, True, "UNAVAILABLE")):
        facts = ev._new_facts()
        ev._accumulate(facts, pa.Table.from_pandas(frame, preserve_index=False))
        semantics = ev.measured_semantics(facts, (observed, "doc"), complete_grid=complete)
        assert ev.classify_semantics(semantics) == expected
    clean = frn_like(1, 3).assign(unit="item", stockout_hours=np.nan)
    facts = ev._new_facts()
    ev._accumulate(facts, pa.Table.from_pandas(clean, preserve_index=False))
    assert ev.classify_semantics(ev.measured_semantics(facts, (True, "doc"), complete_grid=True)) == "FULL"
    with pytest.raises(ValueError):
        ev.classify_semantics({"observed_sales": {"holds": True}})


# ---------------------------------------------------------------- dataset runs


def test_freshretail_train_eval_separation_and_protocol(frn_root):
    inputs = ev.load_freshretailnet(frn_root)
    final = np.datetime64("2024-06-25")
    assert [(o["phase"], o["cutoff"]) for o in inputs.origins] == [("development", final - 21), ("development", final - 14),
                                                                    ("development", final - 7), ("final", final)]
    assert inputs.split_check == {"origin": 4, "history_files": [FRN_TRAIN], "target_files": [FRN_EVAL]}
    report = ev.run_dataset("freshretailnet", frn_root, frn_root / "out", frn_root / "common", inputs=inputs)
    final_origin = report["per_origin"][-1]
    assert final_origin["checks"]["history_source_files"] == [FRN_TRAIN] and final_origin["checks"]["target_source_files"] == [FRN_EVAL]
    assert report["integrity"]["criteria"]["train_eval_separation"] is True
    leaked = frn_like()
    leaked.loc[leaked["date"] == "2024-06-26", "source_file"] = FRN_TRAIN          # an eval day inside the train file
    with pytest.raises(ValueError, match="eval dates"):
        ev.load_freshretailnet(frn_root, canonical=leaked)


def test_dataset_run_report_integrity_and_outputs(frn_root):
    report = run_frn(frn_root)
    assert report["FROZEN_CONFIG_UNCHANGED"] and report["verdict"] in ("PASS", "PASS_WITH_LIMITATION", "FAIL")
    criteria = report["integrity"]["criteria"]
    assert criteria["frozen_config"] and criteria["no_leakage"] and criteria["fallback_policy"]
    for origin in report["per_origin"]:
        checks = origin["checks"]
        assert checks["bridge_future_rows_excluded"] == checks["bridge_future_rows_expected"] > 0
        assert checks["router_future_rows_excluded"] == 0 and checks["router_self_exclusion"]["same_forecast"]
        assert checks["v2_rows_7d_equal_research"] and checks["v2_rows_daily_equal_research"] and checks["reasons_equal_router_assessment"]
    assert report["data"]["quantity_semantics"] == "PARTIAL"
    assert not report["data"]["semantics"]["physical_unit"]["holds"] and not report["data"]["semantics"]["unmodified_meaning"]["holds"]
    reasons = report["coverage"]["all_series"]["router_reason"]
    assert reasons[router.REASON_COLD_START]["series_origins"] == 3 and reasons[router.REASON_MOSTLY_ZERO]["series_origins"] == 3
    assert report["cold_start"]["fallback_as_intended"] and report["cold_start"]["routed_v1_share"] == 1.0
    assert report["production_bridge_e2e"]["pass"]
    for name in ev.RESULT_SUFFIXES:
        assert (frn_root / "frn" / f"freshretail{ev.RESULT_SUFFIXES[name]}").exists()
    by_type = pd.read_csv(frn_root / "frn" / "freshretail_forecast_by_demand_type.csv")
    assert {"v1_wape", "router_wape", "v2_wape", "router_v2_scored_volume_share", "actual_volume_share"} <= set(by_type.columns)
    summary = pd.read_csv(frn_root / "frn" / "freshretail_forecast_external_summary.csv")
    assert set(summary["method"]) == set(ev.METHODS) and set(summary["scope"]) == {"final", "development_pooled"}
    assert any("proxy" in c for c in summary.columns) and not any("cost" in c for c in summary.columns)


def test_favorita_documented_returns_are_not_modified(tmp_path, monkeypatch):
    frame = favorita_like()
    write_dataset(tmp_path, "favorita", frame, FAVORITA_QUALITY, row_group_size=300)
    parts = tmp_path / "partitions"
    monkeypatch.setattr(ev.tempfile, "mkdtemp", lambda prefix="": (parts.mkdir(), str(parts))[1])
    report = ev.run_dataset("favorita", tmp_path, tmp_path / "fav", tmp_path / "common")
    assert not parts.exists()                                                      # temporary per-store files removed
    returns = report["negative_returns"]["full_history"]
    assert returns["canonical_negative_rows"] == returns["bridged_negative_rows"] == 2 and returns["series_with_negative"] == 2
    assert returns["canonical_negative_quantity_sum"] == returns["bridged_negative_quantity_sum"] == -6.0
    assert returns["values_preserved_unchanged"]
    rows = pd.read_parquet(tmp_path / "fav" / "favorita_forecast_external_rows.parquet")
    negative = rows[rows["negative_in_history"]]
    assert len(negative) and (negative["reason"] == router.REASON_NEGATIVE).all() and (negative["version"] == "v1").all()
    assert report["negative_returns"]["per_origin"]["4"]["routed_negative_sales"] == 2
    assert report["data"]["semantics"]["zero_days_recorded"]["holds"] is False and report["data"]["quantity_semantics"] == "PARTIAL"
    assert [o["cutoff"] for o in report["protocol"]["origins"]] == ["2017-07-18", "2017-07-25", "2017-08-01", "2017-08-08"]
    assert report["data"]["profile"]["canonical"]["rows"] == len(frame) and report["bridge"]["history_rows"] == len(frame)


def test_production_bridge_e2e_on_canonical_rows():
    for frame, cutoff, present in ((frn_like(), "2024-06-25", {"cold_start", "mostly_zero", "zero_day", "v2_eligible"}),
                                   (favorita_like(), "2017-08-08", {"negative_history", "cold_start", "missing_day"})):
        result = ev.production_bridge_e2e(frame, cutoff)
        assert result["pass"], result["checks"]
        found = {c["case"] for c in result["cases"] if c["series"]}
        assert present | {"duplicate_date", "history_withheld"} <= found
        assert all(c["observed_reasons"] == {c["expected_reason"]: c["series"]} for c in result["cases"] if c["series"])


def test_future_rows_cannot_change_an_origin(frn_root):
    frame = frn_like()
    scrambled = frame.copy()
    after = scrambled["date"] > "2024-06-04"
    scrambled.loc[after, "sales_qty"] = scrambled.loc[after, "sales_qty"].to_numpy()[::-1] * 3
    cutoff = np.datetime64("2024-06-04")
    outputs = []
    for data in (frame, scrambled):
        panel = ev.build_panel(data, np.datetime64("2024-03-28"), np.datetime64("2024-07-02"))
        outputs.append(ev.evaluate_origin(panel, data, cutoff, router.v2_config()).rows)
    columns = ["reason", "demand_type", *[f"forecast_7d_{m}" for m in ev.METHODS]]
    pd.testing.assert_frame_equal(outputs[0][columns], outputs[1][columns])


def test_results_are_deterministic_and_independent_of_batching(frn_root):
    one = ev.run_dataset("freshretailnet", frn_root, frn_root / "a", frn_root / "common",
                         inputs=ev.load_freshretailnet(frn_root, locations_per_batch=1))
    many = ev.run_dataset("freshretailnet", frn_root, frn_root / "b", frn_root / "common",
                          inputs=ev.load_freshretailnet(frn_root, locations_per_batch=100))
    again = ev.run_dataset("freshretailnet", frn_root, frn_root / "c", frn_root / "common",
                           inputs=ev.load_freshretailnet(frn_root, locations_per_batch=100))
    for suffix in ("summary", "by_cutoff", "by_demand_type", "coverage", "bias", "metrics"):
        name = f"freshretail{ev.RESULT_SUFFIXES[suffix]}"
        assert (frn_root / "b" / name).read_bytes() == (frn_root / "c" / name).read_bytes()          # same input: byte-identical
        # other batching: the same numbers up to float summation order (CSV values are rounded to 6 decimals)
        pd.testing.assert_frame_equal(pd.read_csv(frn_root / "a" / name), pd.read_csv(frn_root / "b" / name), check_exact=False,
                                      rtol=0, atol=1.5e-6)
    assert many["gate"]["evaluation"] == again["gate"]["evaluation"] == one["gate"]["evaluation"]
    rows = [pd.read_parquet(frn_root / d / "freshretail_forecast_external_rows.parquet").sort_values(["origin", "store_id", "product_id"],
                                                                                                      ignore_index=True) for d in "ab"]
    pd.testing.assert_frame_equal(rows[0], rows[1])


# ---------------------------------------------------------------- generic code: no dataset-specific logic


def test_dataset_specific_logic_does_not_alter_the_generic_router():
    generic = [inspect.getsource(m) for m in (router, ev.core)] + [
        inspect.getsource(f) for f in (canonical_to_daily_history, ev.build_panel, ev.inventory_rows, ev.baseline_forecasts, ev.window_stats,
                                       ev.evaluate_origin, ev.metrics_frame, ev.coverage_table, ev.production_bridge_e2e)]
    for source in generic:
        assert not any(name in source.lower() for name in ("favorita", "freshretail", "dingdong", "store_nbr", "item_nbr"))
    # the same observations under another dataset's names and ids route and forecast identically
    frame = favorita_like(1, 5)
    renamed = frame.assign(source_dataset="other", location_id="X" + frame["location_id"], product_id="Y" + frame["product_id"],
                           location_id_namespace="Other:loc", product_id_namespace="Other:sku", category=frame["category"].str.lower())
    cutoff = np.datetime64("2017-08-08")
    results = []
    for data in (frame, renamed):
        panel = ev.build_panel(data, np.datetime64("2017-03-09"), np.datetime64("2017-08-15"))
        results.append(ev.evaluate_origin(panel, data, cutoff, router.v2_config()).rows)
    columns = ["reason", "version", "demand_type", *[f"forecast_7d_{m}" for m in ev.METHODS]]
    pd.testing.assert_frame_equal(results[0][columns], results[1][columns])


# ---------------------------------------------------------------- coverage and fallback aggregation


def coverage_rows():
    return pd.DataFrame({"origin": [1, 1, 1, 1, 2, 2], "store_id": ["S", "S", "S", "T", "S", "S"], "product_id": ["a", "b", "c", "a", "a", "b"],
                         "version": ["v2", "v1", "v1", "v2", "v2", "v2"],
                         "reason": [router.REASON_V2, router.REASON_NEGATIVE, router.REASON_INSUFFICIENT, router.REASON_V2, router.REASON_V2,
                                    router.REASON_V2],
                         "week_scored": [True, True, False, True, True, False], "actual_7d": [10.0, 30.0, np.nan, 60.0, 5.0, np.nan],
                         "history_volume_28d": [40.0, 100.0, 20.0, 240.0, 20.0, 10.0]})


def test_router_coverage_calculation():
    table = ev.coverage_table(coverage_rows(), {"one": [1], "both": [1, 2]}).set_index(["scope", "basis", "dimension", "segment"])
    v2 = table.loc[("one", "scored_series", "router_version", "v2")]
    assert v2["series_origins"] == 2 and v2["volume"] == 70.0 and v2["volume_share"] == pytest.approx(0.7)
    assert table.loc[("one", "all_series", "router_version", "v2"), "volume_share"] == pytest.approx(280 / 400)
    assert table.loc[("one", "all_series", "router_version", "v1"), "series_origin_share"] == pytest.approx(0.5)
    both = table.loc[("both", "all_series", "router_version", "v2")]
    assert both["series_origins"] == 4 and both["distinct_series"] == 3
    assert table.loc[("both", "scored_series", "router_version", "v2"), "volume_share"] == pytest.approx(75 / 105)


def test_fallback_reason_aggregation():
    table = ev.coverage_table(coverage_rows(), {"one": [1]})
    reasons = table[(table["basis"] == "all_series") & (table["dimension"] == "router_reason")].set_index("segment")
    assert list(reasons.index) == list(router.REASON_CODES)
    assert reasons["series_origins"].sum() == 4 and reasons["series_origin_share"].sum() == pytest.approx(1.0)
    assert reasons.loc[router.REASON_NEGATIVE, "volume_share"] == pytest.approx(0.25)
    merged = ev.merge_checks([{"ok": True, "n": 2, "files": ["a"], "nested": {"same_forecast": True, "rows": 1}, "cutoff": "x"},
                              {"ok": False, "n": 3, "files": ["b"], "nested": {"same_forecast": True, "rows": 4}, "cutoff": "x"}])
    assert merged == {"ok": False, "n": 5, "files": ["a", "b"], "nested": {"same_forecast": True, "rows": 5}, "cutoff": "x"}


def test_summary_writes_the_generalisation_verdict_and_support_matrix(tmp_path):
    write_dataset(tmp_path, "freshretailnet", frn_like(), FRN_QUALITY)
    write_dataset(tmp_path, "favorita", favorita_like(), FAVORITA_QUALITY)
    dirs = {d: tmp_path / d for d in ("freshretailnet", "favorita")}
    reports = {d: ev.run_dataset(d, tmp_path, dirs[d], tmp_path / "common") for d in dirs}
    summary = ev.write_summary(tmp_path, tmp_path / "common", dataset_dirs=dirs)
    expected = ev.combined_verdict({d: r["gate"]["evaluation"] for d, r in reports.items()})
    assert summary["GENERALIZATION_VERDICT"] == expected and summary["gate"]["same_gate_in_every_report"]
    assert summary["FROZEN_CONFIG_UNCHANGED"]
    matrix = pd.read_csv(tmp_path / "common" / ev.SUPPORT_MATRIX_FILE)
    assert list(matrix["dataset"]) == ["suhyup", "logisall", "jangbogo", "m5", "favorita", "freshretailnet"]
    assert set(matrix.columns) >= {"daily_history_support", "operational_use", "forecast_validation", "dataset_role"}
    assert set(matrix.loc[matrix["dataset_role"] == "external_benchmark", "operational_use"]) == {"no"}
    assert all(step["verified_in_code"] for step in summary["forecast_to_decision_path"])

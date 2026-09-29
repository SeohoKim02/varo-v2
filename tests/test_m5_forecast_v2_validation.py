"""M5 v2 development/final protocol: isolation, leakage, frozen selection and gate, on small in-memory panels and fixtures."""
import json
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from services import demand_forecast_v2 as core
from services import m5_forecast_v2_validation as v2
from services.m5_forecast_validation import (
    RAW_DIR, RAW_INPUTS, VARO_INPUT_COLUMNS, M5Panel, file_fingerprints, point_metrics, run_varo_forecast, varo_inputs)
from tests.test_m5_forecast_validation import write_fixture

SMALL = {"horizon": 7, "origin_count": 13, "origin_step": 7}     # 10 development + 3 final cutoffs, 7 days apart
DAYS = 160
VOLATILE = {"evaluation", "runtime", "frozen_at", "v1_baseline"}


def random_panel(seed=3, series=24, days=DAYS):
    rng = np.random.default_rng(seed)
    weekly = 1 + 0.3 * np.sin(np.arange(days) * 2 * np.pi / 7)
    sales = rng.poisson(rng.uniform(0.02, 6, size=(series, 1)) * weekly, size=(series, days)).astype(float)
    sales[5, :days - 20] = 0                                  # a late launch
    price = np.where(rng.random((series, days)) < 0.05, np.nan, rng.uniform(1, 5, (series, days))).astype(np.float32)
    frame = pd.DataFrame({"store_id": np.where(np.arange(series) % 2, "S1", "S2"), "item_id": [f"I{k:02d}" for k in range(series)],
                          "state_id": "CA", "category": np.where(np.arange(series) % 3, "A", "B"), "department": "D"})
    dates = pd.date_range("2015-01-01", periods=days).strftime("%Y-%m-%d").to_numpy()
    events = np.where(np.arange(days) % 17 == 0, "National", "none").astype(object)
    snap = np.tile((np.arange(days) % 3 == 0).astype(np.float32), (series, 1))
    return M5Panel(sales, price, dates, frame, events, snap, series * days)


def copy_panel(panel):
    return M5Panel(panel.sales.copy(), panel.price.copy(), panel.dates.copy(), panel.series.copy(), panel.event_label.copy(),
                   panel.snap.copy(), panel.canonical_rows)


def scramble_after(panel, day, seed=11, covariates=True):
    """Overwrite every value after ``day``: sales always, price/SNAP/events when ``covariates``."""
    out = copy_panel(panel)
    rng = np.random.default_rng(seed)
    out.sales[:, day + 1:] = rng.poisson(40, size=out.sales[:, day + 1:].shape)
    if covariates:
        out.price[:, day + 1:] = 0.37
        out.snap[:, day + 1:] = 1 - out.snap[:, day + 1:]
        out.event_label[day + 1:] = "Sporting"
    return out


def stable(selection):
    return {k: v for k, v in selection.items() if k not in VOLATILE}


@pytest.fixture(scope="module")
def developed(tmp_path_factory):
    out = tmp_path_factory.mktemp("v2dev")
    panel = random_panel()
    return panel, out, v2.run_development(None, out, panel=panel, **SMALL)


# ---------------------------------------------------------------- protocol


def test_development_final_cutoff_isolation():
    protocol = v2.split_protocol(1941)
    dev, final = protocol["development"], protocol["final"]
    assert [o["origin"] for o in dev] == list(range(1, 11)) and [o["origin"] for o in final] == [11, 12, 13]
    # Official holdout = origin 13: cutoff d_1913, targets d_1914..d_1941 (0-based index + 1 = d number).
    holdout = protocol["holdout"]
    assert holdout is final[-1] and (holdout["cutoff"] + 1, holdout["target_start"] + 1, holdout["target_end"] + 1) == (1913, 1914, 1941)
    last = protocol["development_last_day"]
    assert last + 1 == 1857 and max(o["target_end"] for o in dev) == last == final[0]["cutoff"]
    assert all(o["target_start"] > last for o in final)
    with pytest.raises(ValueError):
        v2.split_protocol(1941, dev_count=9)        # the final test is exactly the last 3 cutoffs
    panel = random_panel()
    cut = v2.truncate_panel(panel, 99)
    assert cut.sales.shape[1] == cut.price.shape[1] == cut.snap.shape[1] == len(cut.dates) == len(cut.event_label) == 100
    assert np.shares_memory(cut.sales, panel.sales)  # a view: nothing after day 99 is reachable from it
    record = v2.protocol_record(v2.split_protocol(DAYS, **SMALL), panel.dates)
    assert record["official_holdout_origin"]["origin"] == 13 and len(record["development"]) == 10 and len(record["final"]) == 3


# ---------------------------------------------------------------- leakage


def test_no_future_leakage():
    panel = random_panel()
    spec = v2.split_protocol(DAYS, **SMALL)["final"][0]
    cutoff = spec["cutoff"]
    groups = v2.pool_groups(panel.series)
    config = core.make_config({t: {"level": "sba_0.1" if t == "intermittent" else "ewma_w0.2_trend", "weekday": "shrunk_8w"}
                               for t in core.ROUTED_TYPES})

    def forecasts(p):
        od = v2.build_origin(p, spec, 7)
        events, _ = v2.event_matrix(v2._event_names_from_labels(p.event_label))
        result, exo, _ = v2.v2_forecasts(od, p, config, 7, groups, events)
        return od, v2.comparator_forecasts(od, 7), result, exo

    od, comparators, result, exo = forecasts(panel)
    # Every post-cutoff value changes: the Core and all comparators are blind to it.
    od_all, comparators_all, result_all, _ = forecasts(scramble_after(panel, cutoff))
    np.testing.assert_array_equal(result.daily, result_all.daily)
    for name in v2.COMPARATORS:
        np.testing.assert_array_equal(comparators[name][0], comparators_all[name][0])
        np.testing.assert_array_equal(comparators[name][1], comparators_all[name][1])
    assert not np.array_equal(od.actual, od_all.actual)
    # Only post-cutoff sales change: the exogenous variant (which may read known-in-advance covariates) is blind too.
    _, _, _, exo_sales = forecasts(scramble_after(panel, cutoff, covariates=False))
    np.testing.assert_array_equal(exo, exo_sales)


def test_final_data_cannot_affect_model_selection(developed, tmp_path):
    panel, out, selection = developed
    last = v2.split_protocol(DAYS, **SMALL)["development_last_day"]
    changed = v2.run_development(None, tmp_path, panel=scramble_after(panel, last), **SMALL)
    assert changed["selected"] == selection["selected"]
    assert changed["stage_1_level_selection"] == selection["stage_1_level_selection"]
    assert changed["stage_2_weekday_selection"] == selection["stage_2_weekday_selection"]
    assert changed["data_signature"]["development_sales_sha256"] == selection["data_signature"]["development_sales_sha256"]
    for name in ("development", "by_demand_type", "bias"):
        assert (tmp_path / v2.RESULT_FILES[name]).read_bytes() == (out / v2.RESULT_FILES[name]).read_bytes()
    # A change inside the development window does change the evidence (the selection is not a constant).
    touched = copy_panel(panel)
    touched.sales[:, 60:last + 1] *= 3
    moved = v2.run_development(None, tmp_path / "touched", panel=touched, **SMALL)
    assert moved["development_metrics"] != selection["development_metrics"]


def test_deterministic_model_selection(developed, tmp_path):
    panel, out, selection = developed
    again = v2.run_development(None, tmp_path, panel=copy_panel(panel), **SMALL)
    assert stable(again) == stable(selection) and again["random_seed"] is None
    for name in ("development", "by_demand_type", "by_horizon", "bias", "by_calendar"):
        assert (tmp_path / v2.RESULT_FILES[name]).read_bytes() == (out / v2.RESULT_FILES[name]).read_bytes()
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / v2.RESULT_FILES["development_sums"]),
                                  pd.read_parquet(out / v2.RESULT_FILES["development_sums"]))
    saved = json.loads((out / v2.RESULT_FILES["model_selection"]).read_text(encoding="utf-8"))
    assert saved["selected"]["config_signature"] == core.config_signature(saved["selected"]["config"])
    core.validate_config(saved["selected"]["config"])
    for key in ("protocol", "search_space", "selection_rule", "promotion_gate", "promotion_gate_signature", "frozen_at",
                "stage_1_level_selection", "stage_2_weekday_selection", "development_metrics", "leakage_controls", "runtime"):
        assert key in saved


# ---------------------------------------------------------------- frozen gate


def test_promotion_gate_frozen_before_final(developed, tmp_path, monkeypatch):
    panel, out, selection = developed
    with pytest.raises(v2.FreezeViolation):
        v2.run_final(None, tmp_path / "empty", panel=panel, **SMALL)          # nothing frozen yet
    work = tmp_path / "work"
    shutil.copytree(out, work)
    report = v2.run_final(None, work, panel=panel, **SMALL)
    assert report["decision"] in ("PASS", "FAIL") and report["production_changed"] is False
    assert report["frozen_at"] < report["final_evaluated_at"] and all(report["freeze_verification"].values())
    assert report["config_signature"] == selection["selected"]["config_signature"]
    assert report["promotion_gate_signature"] == selection["promotion_gate_signature"] == v2.gate_signature()
    assert {c["id"][:2] for c in report["criteria"]} == {"G1", "G2", "G3", "G4", "G5", "G6"}
    # Re-running the same frozen config reproduces the same verdict and tables.
    first = (work / v2.RESULT_FILES["final"]).read_bytes()
    assert v2.run_final(None, work, panel=panel, **SMALL)["criteria"] == report["criteria"]
    assert (work / v2.RESULT_FILES["final"]).read_bytes() == first

    # Editing the gate after the freeze is refused.
    with monkeypatch.context() as patch:
        patch.setitem(v2.PROMOTION_GATE["primary"], "min_relative_improvement", 0.0)
        with pytest.raises(v2.FreezeViolation, match="gate"):
            v2.run_final(None, work, panel=panel, **SMALL)
    # Editing the frozen configuration is refused (signature mismatch).
    tampered = tmp_path / "tampered"
    shutil.copytree(out, tampered)
    saved = json.loads((tampered / v2.RESULT_FILES["model_selection"]).read_text(encoding="utf-8"))
    saved["selected"]["config"]["routes"]["high_frequency"]["level"] = "ma7"
    (tampered / v2.RESULT_FILES["model_selection"]).write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(v2.FreezeViolation, match="config"):
        v2.run_final(None, tampered, panel=panel, **SMALL)
    # Once a configuration has been scored on the final origins, another one may not be.
    locked = tmp_path / "locked"
    shutil.copytree(work, locked)
    gate = json.loads((locked / v2.RESULT_FILES["promotion_gate"]).read_text(encoding="utf-8"))
    gate["config_signature"] = "0" * 64
    (locked / v2.RESULT_FILES["promotion_gate"]).write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(v2.FreezeViolation, match="already evaluated"):
        v2.run_final(None, locked, panel=panel, **SMALL)
    # Changing development data between the phases is refused.
    with pytest.raises(v2.FreezeViolation, match="development_data"):
        changed = copy_panel(panel)
        changed.sales[0, 50] += 1
        v2.run_final(None, work, panel=changed, **SMALL)


def _gate_table(changes=None):
    """A metrics table where v2 passes every criterion; ``changes`` overrides (method, scope, window, metric[, segment])."""
    rows = []
    scopes = ["final_rolling_pooled", "o11", "o12", "o13_holdout"]
    for scope in scopes:
        for window in ("total_7d", "daily_h1_7"):
            for dimension, segments in (("overall", ["all"]), ("frequency_class", ["high_frequency", "medium_frequency", "intermittent"])):
                for segment in segments:
                    rows.append({"scope": scope, "window": window, "dimension": dimension, "segment": segment, "method": v2.V1,
                                 "wape": 0.40, "mae": 1.0, "rmse": 2.0, "bias_pct": 0.025})
                    rows.append({"scope": scope, "window": window, "dimension": dimension, "segment": segment, "method": v2.V2,
                                 "wape": 0.38, "mae": 0.95, "rmse": 1.99, "bias_pct": 0.01})
    table = pd.DataFrame(rows)
    for (method, scope, window, metric, *segment), value in (changes or {}).items():
        mask = (table.method == method) & (table.scope == scope) & (table.window == window)
        mask &= (table.segment == segment[0]) if segment else (table.dimension == "overall")
        table.loc[mask, metric] = value
    return v2.evaluate_gate(table, "final_rolling_pooled", ["o11", "o12", "o13_holdout"], "o13_holdout")


def test_gate_decision_logic():
    assert _gate_table()["decision"] == "PASS"
    cases = {
        "G1_primary_7d_wape": {(v2.V2, "final_rolling_pooled", "total_7d", "wape"): 0.395},          # only 1.25% better
        "G2_consistency_origin_wins": {(v2.V2, "o11", "total_7d", "wape"): 0.41, (v2.V2, "o12", "total_7d", "wape"): 0.40},
        "G3_official_holdout_total_7d_wape": {(v2.V2, "o13_holdout", "total_7d", "wape"): 0.405},
        "G3_official_holdout_daily_h1_7_wape": {(v2.V2, "o13_holdout", "daily_h1_7", "wape"): 0.41},
        "G4_abs_bias": {(v2.V2, "final_rolling_pooled", "total_7d", "bias_pct"): -0.03},
        "G5_secondary_total_7d_rmse": {(v2.V2, "final_rolling_pooled", "total_7d", "rmse"): 2.05},
        "G5_secondary_daily_h1_7_wape": {(v2.V2, "final_rolling_pooled", "daily_h1_7", "wape"): 0.401},
        "G6_demand_type_intermittent": {(v2.V2, "final_rolling_pooled", "total_7d", "wape", "intermittent"): 0.413},
    }
    for criterion, change in cases.items():
        result = _gate_table(change)
        assert result["decision"] == "FAIL" and result["failed"] == [criterion], criterion
    # Inside the tolerances everything still passes.
    assert _gate_table({(v2.V2, "o13_holdout", "total_7d", "wape"): 0.404, (v2.V2, "final_rolling_pooled", "total_7d", "bias_pct"): -0.025,
                        (v2.V2, "final_rolling_pooled", "total_7d", "wape", "intermittent"): 0.411})["decision"] == "PASS"


def test_selection_rule_bias_guard_and_tie_break():
    rows = [("high_frequency", v2.V1, 0.33, 0.01), ("high_frequency", "option:ma7", 0.30, 0.05),
            ("high_frequency", "option:ma14", 0.31, 0.0), ("high_frequency", "option:ma28", 0.3102, 0.0),
            ("medium_frequency", v2.V1, 0.40, 0.0), ("medium_frequency", "option:ma7", 0.39, 0.06),
            ("mostly_zero", v2.V1, 0.90, 0.20), ("mostly_zero", "option:ma28", 0.85, 0.25), ("mostly_zero", "option:sba_0.1", 0.80, -0.15)]
    stage1 = pd.DataFrame([{"segment": s, "method": m, "wape": w, "bias_pct": b, "abs_err": w * 100, "actual": 100.0} for s, m, w, b in rows])
    routes, decisions = v2.select_levels(stage1)
    # ma7 wins on WAPE but its 5% bias exceeds max(|1%|, 3%); ma28 is within 0.1% of ma14 and comes first in the declared order.
    assert routes == {"high_frequency": "ma28", "medium_frequency": "ma28", "intermittent": "ma28", "mostly_zero": "sba_0.1"}
    reasons = {d["demand_type"]: d["reason"] for d in decisions}
    assert "tie-break" in reasons["high_frequency"] and reasons["medium_frequency"] == "no option inside the bias bound"
    assert reasons["intermittent"] == "no development series of this type"
    stage2 = pd.DataFrame([{"segment": "high_frequency", "method": f"weekday:{m}", "abs_err": e, "wape": e / 100, "rmse": 1.0}
                           for m, e in (("flat", 100.0), ("series_4w", 90.0), ("series_8w", 89.95), ("shrunk_8w", 95.0))])
    weekdays, _ = v2.select_weekdays(stage2)
    assert weekdays["high_frequency"] == "series_4w" and weekdays["intermittent"] == "flat"


# ---------------------------------------------------------------- metrics


def test_bias_and_inventory_proxy_metrics():
    f, a = np.array([2.0, 0.0, 3.0, 1.0, 4.0, 0.0]), np.array([1.0, 0.0, 5.0, 1.0, 1.0, 2.0])
    stats = v2.point_stats(f, a, np.ones(6, dtype=bool))
    sums = pd.DataFrame([{**{k: float(v.sum()) for k, v in stats.items()}, "scored": 1.0, "mase_n": 0.0, "mase_sum": 0.0, "rmsse_sum": 0.0,
                          "origin": 1, "method": v2.V2, "window": "total_7d", "dimension": "overall", "segment": "all"}])
    row = v2.pooled(sums, {"s": [1]}).iloc[0]
    metrics = point_metrics(f, a)
    assert row["wape"] == pytest.approx(metrics["wape"]) and row["bias_pct"] == pytest.approx((10 - 10) / 10)
    assert row["fill_rate_proxy"] == pytest.approx(1 - 4 / 10) and row["excess_units_ratio"] == pytest.approx(4 / 10)
    assert row["missed_demand_n"] == 1 and row["idle_forecast_units"] == 0.0   # F=0,A=2 is a certain stock-out
    assert row["over_units"] - row["under_units"] == pytest.approx(row["bias_units"])


def test_scored_sums_are_additive(developed):
    panel = developed[0]
    spec = v2.split_protocol(DAYS, **SMALL)["final"][0]
    od = v2.build_origin(panel, spec, 7)
    daily, total = v2.comparator_forecasts(od, 7)["moving_average_28"]
    frame = v2.pooled(pd.concat(v2.score(od, "moving_average_28", daily, total), ignore_index=True), {"s": [spec["origin"]]})
    overall = frame[(frame.dimension == "overall")].set_index("window")
    expected = point_metrics(daily[:, :7], od.actual[:, :7], od.valid[:, :7])
    assert overall.loc["daily_h1_7", "wape"] == pytest.approx(expected["wape"])
    assert overall.loc["daily_h1_7", "rmse"] == pytest.approx(expected["rmse"])
    week = point_metrics(total, od.week_actual, od.week_valid)
    assert overall.loc["total_7d", "mae"] == pytest.approx(week["mae"]) and overall.loc["total_7d", "bias_pct"] == pytest.approx(week["bias_pct"])
    by_type = frame[(frame.dimension == "frequency_class") & (frame.window == "total_7d")]
    assert by_type["abs_err"].sum() == pytest.approx(overall.loc["total_7d", "abs_err"])


# ---------------------------------------------------------------- end to end on the canonical fixture


def test_raw_untouched_and_outputs(tmp_path):
    dataset = write_fixture(tmp_path / "root", days=170)
    raw = dataset / RAW_DIR
    before = file_fingerprints(raw, RAW_INPUTS)
    processed = {p.name: p.stat().st_mtime_ns for p in (dataset / "processed").iterdir()}
    out = tmp_path / "out"
    selection = v2.run_development(dataset.parent, out, **SMALL)
    report = v2.run_final(dataset.parent, out, **SMALL)
    assert file_fingerprints(raw, RAW_INPUTS) == before
    assert {p.name: p.stat().st_mtime_ns for p in (dataset / "processed").iterdir()} == processed
    assert report["data_signature"]["raw_unchanged_during_run"] is True
    assert report["data_signature"]["raw_canonical_crosscheck"]["cells_equal_including_missing"] is True
    assert sorted(p.name for p in out.iterdir()) == sorted(v2.RESULT_FILES.values())
    assert selection["v1_baseline"]["current_fingerprint"]["source_sha256_lf"] == core.FORECAST_V1_BASELINE["source_sha256_lf"]
    # V1 in the final rows is the production function's output; the v2 daily vector sums to its 7-day scalar.
    rows = pq.read_table(out / v2.RESULT_FILES["final_rows"]).to_pandas()
    series = pd.read_parquet(out / v2.RESULT_FILES["final_series"])
    assert len(rows) == 3 * 4 * 7 and len(series) == 3 * 4
    panel = v2.load_inputs(dataset.parent).panel
    last = v2.split_protocol(170, **SMALL)["holdout"]
    expected = run_varo_forecast(varo_inputs(panel.sales[:, :last["cutoff"] + 1]), VARO_INPUT_COLUMNS["varo_production"])
    holdout_rows = rows[rows.origin == last["origin"]]
    np.testing.assert_allclose(holdout_rows.groupby(["store_id", "item_id"], sort=True, observed=True)["forecast_varo_v1_production"].first(),
                               expected["demand_forecast_daily"])
    weekly = rows.groupby(["origin", "store_id", "item_id"], observed=True)["forecast_varo_v2_core"].sum().to_numpy()
    assert np.abs(np.round(weekly, 1) - series.sort_values(["origin", "store_id", "item_id"])["forecast_7d_v2_core"].to_numpy()).max() <= 0.1 + 1e-9
    final = pd.read_csv(out / v2.RESULT_FILES["final"])
    assert set(final["scope"]) == {"final_rolling_pooled", "final_origin_11", "final_origin_12", "final_origin_13_official_holdout"}
    assert set(v2.REPORT_METHODS) == set(final["method"])
    for name in ("by_demand_type", "by_horizon", "bias", "by_calendar"):
        assert set(pd.read_csv(out / v2.RESULT_FILES[name])["phase"]) == {"development", "final"}

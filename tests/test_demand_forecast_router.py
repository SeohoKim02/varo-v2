"""Production demand-forecast router: V2 only on eligible daily histories, V1 (unchanged) everywhere else."""
import hashlib
import json
import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services import demand_forecast_router as router
from services import demand_forecast_v2 as core
from services.analysis_pipeline import PipelineResult, _run_inventory_analysis, _Runner, build_v2_state, run_analysis_pipeline
from services.data_loader import load_excel_data
from services.inventory_transition_service import run_inventory_scenario
from services.legacy_adapters.data_adapter import prepare_legacy_data
from services.legacy_adapters.loader import load_legacy_module
from services.m5_forecast_validation import AGE_BANDS, HISTORY_WINDOW, series_profile
from tests.fixtures import sample_workbook, workbook_excel_bytes

REPO = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp("2026-06-30")
V1_COLUMNS = ("demand_forecast_7d", "demand_forecast_daily", "demand_forecast_upper", "demand_forecast_lower", "demand_trend",
              "demand_stockout_days", "demand_risk_score", "demand_forecast_method", "demand_forecast_score")


def v1(frame):
    return load_legacy_module("demand_forecast_analyzer").analyze_demand_forecast(frame)


def inventory(products, **extra):
    base = {"store_id": "S1", "product_id": list(products), "product_name": "x", "category": "C", "sales_7d": 21.0,
            "sales_30d": 90.0, "avg_daily_sales": 3.0, "demand_std": 1.0, "stock_qty": 40.0, "lead_time_days": 2.0}
    return pd.DataFrame({**base, **extra})


def long_history(series, cutoff=CUTOFF, store="S1"):
    """{product: daily quantities ending at ``cutoff``} -> contract rows; NaN entries become missing days (no row)."""
    rows = []
    for product, values in series.items():
        days = pd.date_range(end=cutoff, periods=len(values), freq="D")
        rows += [{"store_id": store, "product_id": product, "date": d, "quantity": float(q)}
                 for d, q in zip(days, values) if not np.isnan(q)]
    return pd.DataFrame(rows)


def route(products, series, **kwargs):
    return router.route_demand_forecast(v1(inventory(products)), long_history(series), **kwargs)


def steady(days=200, level=4.0, seed=0):
    return np.random.default_rng(seed).poisson(level, days).astype(float)


# ---------------------------------------------------------------- contract A: no daily history -> V1 unchanged


def test_no_history_keeps_v1_exactly_and_adds_provenance():
    frame = inventory(["P1", "P2"], sales_7d=[21.0, 0.0])
    base = v1(frame)
    for history in (None, pd.DataFrame()):
        out, diagnostics = router.route_demand_forecast(base, history)
        pd.testing.assert_frame_equal(out[list(base.columns)], base, check_exact=True)
        assert out.columns.tolist() == list(base.columns) + list(router.ROUTER_COLUMNS)
        assert out["demand_forecast_version"].tolist() == ["v1", "v1"]
        assert out["demand_forecast_reason"].tolist() == [router.REASON_MISSING] * 2
        assert out["demand_forecast_fallback_reason"].tolist() == [router.REASON_MISSING] * 2
        np.testing.assert_array_equal(out[list(router.DAILY_COLUMNS)].to_numpy(), np.repeat(base[["demand_forecast_7d"]].to_numpy() / 7, 7, axis=1))
        assert diagnostics["history_supplied"] is False and diagnostics["counts_by_reason"] == {router.REASON_MISSING: 2}


def _scalar(value):
    if isinstance(value, dict):
        return {str(k): _scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scalar(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return "nan" if value != value else repr(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return "NA"
    return value


def _digest(value):
    if isinstance(value, pd.DataFrame):
        columns = [c for c in value.columns if c not in router.ROUTER_COLUMNS]
        value = {"columns": columns, "dtypes": [str(value[c].dtype) for c in columns],
                 "rows": [list(row) for row in value[columns].itertuples(index=False)]}
    text = json.dumps(_scalar(value), sort_keys=True, ensure_ascii=False, default=repr)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def golden_outputs(data):
    state = build_v2_state(data, detail_level="full")
    analyzed, _ = _run_inventory_analysis(_Runner(PipelineResult()), prepare_legacy_data(data)["inventory"])
    return {"analyzed_inventory": analyzed, "recommendations": state["recommendations"],
            "summary": state["pipeline_result"]["summary"], "scenario": run_inventory_scenario(data, state["recommendations"])}


# Digests of the production outputs captured with the code BEFORE the router existed (commit 8309c91), for every
# workbook in the repository: analyzed inventory (every pre-existing column: values, dtypes, rounding), standardized
# recommendations, KPI summary and the inventory transition scenario. Router columns are excluded (new fields).
GOLDEN_V1 = {
    "Varo_DQN_training_samples_10pack/Varo_DQN_sample_01_2stores_1dc_fresh_meal.xlsx": {"analyzed_inventory": "c299c631890712ea", "recommendations": "6dceead739371c2c",
        "summary": "9f20721ba997d50f", "scenario": "2f69ccaf55b9b47e"},
    "Varo_DQN_training_samples_10pack/Varo_DQN_sample_02_4stores_1dc_frozen.xlsx": {"analyzed_inventory": "ceb43c1ab5fac4c7", "recommendations": "053bcf2a83b098b8",
        "summary": "552d7d77fb1cb4ed", "scenario": "d46f7c643d6a4634"},
    "Varo_DQN_training_samples_10pack/Varo_DQN_sample_03_4stores_1dc_dairy_bakery.xlsx": {"analyzed_inventory": "542c8b93f4b2cdc3", "recommendations": "c822f57f0b875fa9",
        "summary": "2ecaa296e1ce6d17", "scenario": "6cc485f2082a55cd"},
    "Varo_DQN_training_samples_10pack/Varo_DQN_sample_04_5stores_1dc_produce.xlsx": {"analyzed_inventory": "7cf3f0fd80805616", "recommendations": "c8f13c10029fd8c2",
        "summary": "d2e692d274a18a6e", "scenario": "231516d0aa0558ef"},
    "Varo_DQN_training_samples_10pack/Varo_DQN_sample_05_5stores_1dc_bakery.xlsx": {"analyzed_inventory": "e812646f2b6e8a47", "recommendations": "7813c82189bb01d2",
        "summary": "45d56b148ed55e1d", "scenario": "f009788d1b9a69fa"},
    "Varo_DQN_training_samples_10pack/Varo_DQN_sample_06_6stores_1dc_beverage_dry.xlsx": {"analyzed_inventory": "c2e9db64a6d500fd", "recommendations": "faf208f149f152ca",
        "summary": "0b92d93d8d335504", "scenario": "47f9a59d2bde1243"},
    "Varo_DQN_training_samples_10pack/Varo_DQN_sample_07_6stores_1dc_meat_egg.xlsx": {"analyzed_inventory": "bc85e5f14fd0e544", "recommendations": "dddd832f3395292c",
        "summary": "65924b73f6c27152", "scenario": "aa90cc1f1f8bb780"},
    "Varo_DQN_training_samples_10pack/Varo_DQN_sample_08_6stores_1dc_seafood.xlsx": {"analyzed_inventory": "98a5504505fcd983", "recommendations": "50a7c8079a3f2914",
        "summary": "68ce0407043cf592", "scenario": "3f77517466acb86a"},
    "Varo_DQN_training_samples_10pack/Varo_DQN_sample_09_8stores_1dc_meal_kit.xlsx": {"analyzed_inventory": "0a26b73a246e85ee", "recommendations": "50e0d11575d38937",
        "summary": "8e4c4b7631009be1", "scenario": "8032cf4ffee9d938"},
    "Varo_DQN_training_samples_10pack/Varo_DQN_sample_10_10stores_2dc_mixed.xlsx": {"analyzed_inventory": "e1f351c8068a2a52", "recommendations": "321304def7a7d321",
        "summary": "67caba3d6e99f685", "scenario": "ca0c5ba8a56d4d10"},
    "data/Varo_V2_네트워크_샘플.xlsx": {"analyzed_inventory": "434c9d43095f6f23", "recommendations": "cba5b1f0bf73fea8",
        "summary": "f4cc7232de70579b", "scenario": "603366442243e3ad"},
    "samples/Varo_V2_sample_dual_dc_10stores_2dc.xlsx": {"analyzed_inventory": "661048852eac7845", "recommendations": "cf3f99d07d9e54a4",
        "summary": "14e12ed56653db11", "scenario": "7a5427d4a2df8ff9"},
    "samples/Varo_V2_sample_edge_3stores_1dc.xlsx": {"analyzed_inventory": "d163aee1473a92f9", "recommendations": "f7717531fb097f18",
        "summary": "2514fda53087badd", "scenario": "0d7299e8bd110865"},
    "samples/Varo_V2_sample_normal_6stores_1dc.xlsx": {"analyzed_inventory": "6e13787ec52fe82c", "recommendations": "939953a0de49e0c5",
        "summary": "a38ffbbdb5d3d31f", "scenario": "7143fed3fa013314"},
    "samples/Varo_V2_sample_small_4stores_1dc.xlsx": {"analyzed_inventory": "5ae6e20db896611a", "recommendations": "67fbc145cc352f1c",
        "summary": "71799c31c38b08b1", "scenario": "0e4e163585482d7e"},
    "samples/Varo_V2_sample_standard_8stores_1dc.xlsx": {"analyzed_inventory": "8473e14930c76faf", "recommendations": "ee8cee6980c97fe7",
        "summary": "2c49878eee2d5359", "scenario": "45decd378e6ec384"},
}


@pytest.mark.parametrize("workbook", sorted(GOLDEN_V1))
def test_existing_inputs_reproduce_pre_router_outputs(workbook):
    outputs = golden_outputs(load_excel_data(REPO / workbook))
    assert {name: _digest(value) for name, value in outputs.items()} == GOLDEN_V1[workbook]
    assert set(outputs["analyzed_inventory"]["demand_forecast_version"]) == {"v1"}
    assert set(outputs["analyzed_inventory"]["demand_forecast_reason"]) == {router.REASON_MISSING}


def test_golden_reference_covers_every_repository_workbook():
    workbooks = sorted(str(p.relative_to(REPO)).replace("\\", "/") for folder in ("data", "samples", "Varo_DQN_training_samples_10pack")
                       for p in (REPO / folder).glob("*.xlsx"))
    assert sorted(GOLDEN_V1) == workbooks and len(workbooks) == 16


# ---------------------------------------------------------------- routing and reason codes


def test_reason_codes_for_every_route():
    cold = np.r_[np.zeros(180), np.full(20, 3.0)]              # first sale 20 days before the cutoff
    sparse = np.zeros(200)
    sparse[::15] = 1.0                                          # sold on 1 day in 15 -> mostly_zero
    gap = steady(seed=4)
    gap[-3:] = np.nan                                           # last 3 days not observed
    negative = steady(seed=5)
    negative[100] = -2.0                                        # a return
    out, diagnostics = route(["OK", "COLD", "SPARSE", "GAP", "NEG", "NEVER", "NONE"],
                             {"OK": steady(), "COLD": cold, "SPARSE": sparse, "GAP": gap, "NEG": negative, "NEVER": np.zeros(200)})
    expected = [router.REASON_V2, router.REASON_COLD_START, router.REASON_MOSTLY_ZERO, router.REASON_INSUFFICIENT,
                router.REASON_NEGATIVE, router.REASON_NO_SALES, router.REASON_MISSING]
    assert out["demand_forecast_reason"].tolist() == expected
    assert out["demand_forecast_version"].tolist() == ["v2"] + ["v1"] * 6
    assert out["demand_forecast_fallback_reason"].isna().tolist() == [True] + [False] * 6
    assert out["demand_forecast_fallback_reason"].iloc[1:].tolist() == expected[1:]
    assert set(expected) | {router.REASON_INVALID, router.REASON_ERROR} == set(router.REASON_CODES)
    assert diagnostics["counts_by_version"] == {"v1": 6, "v2": 1}


def test_eligible_series_gets_the_frozen_v2_core_forecast():
    history = steady(level=5.0, seed=7)
    out, _ = route(["P1"], {"P1": history})
    result = core.forecast_v2(history[None, :], core.make_config(router.V2_ROUTES))
    row = out.iloc[0]
    assert row["demand_forecast_7d"] == np.round(result.aggregate_7d[0], 1)
    assert row["demand_forecast_daily"] == round(row["demand_forecast_7d"] / 7, 2)
    np.testing.assert_array_equal(row[list(router.DAILY_COLUMNS)].to_numpy(dtype=float), result.daily[0, :7])
    assert row["demand_forecast_method"] == f"V2:{result.level_method[0]}+{result.weekday_method[0]}" == "V2:ewma_w0.4+pooled_8w"


def test_v1_fallback_rows_are_untouched():
    base = v1(inventory(["OK", "COLD"]))
    out, _ = router.route_demand_forecast(base, long_history({"OK": steady(), "COLD": np.r_[np.zeros(190), np.ones(10)]}))
    fallback = out["demand_forecast_version"] == "v1"
    pd.testing.assert_frame_equal(out.loc[fallback, list(base.columns)], base.loc[fallback], check_exact=True)
    assert out.loc[~fallback, "demand_forecast_7d"].iloc[0] != base.loc[~fallback, "demand_forecast_7d"].iloc[0]
    # V1 keys that describe inputs (trend) are never replaced.
    assert out["demand_trend"].tolist() == base["demand_trend"].tolist()


def test_thresholds_are_the_m5_protocol_definitions():
    assert router.RECENT_OBSERVED_DAYS == HISTORY_WINDOW == 30
    assert router.MIN_SALES_AGE_DAYS == 30 and AGE_BANDS[1] == "<30d"
    history = np.zeros((2, 100))
    history[0, -30:] = 1.0                                      # first sale exactly 30 days before the cutoff: eligible age
    history[1, -29:] = 1.0                                      # 29 days: cold start (the M5 "<30d" band)
    assert series_profile(history)["age_band"].tolist() == [2, 1]
    assert router.assess_matrix(history).tolist() == [router.REASON_V2, router.REASON_COLD_START]
    assert core.config_signature(router.v2_config()) == router.V2_CONFIG_SIGNATURE


def test_mostly_zero_policy_uses_v1_not_the_under_forecasting_v2():
    sparse = np.zeros(364)
    sparse[::12] = 2.0                                           # 1 selling day in 12 (< 10%)
    base = v1(inventory(["P1"]))
    out, _ = router.route_demand_forecast(base, long_history({"P1": sparse}))
    assert router.MOSTLY_ZERO_POLICY == "v1"
    assert core.DEMAND_TYPES[core.demand_profile(sparse[None, :]).demand_type[0]] == "mostly_zero"
    assert out["demand_forecast_reason"].iloc[0] == router.REASON_MOSTLY_ZERO
    assert out["demand_forecast_7d"].iloc[0] == base["demand_forecast_7d"].iloc[0]


# ---------------------------------------------------------------- input contract


def test_invalid_history_nan_quantity_bad_date_non_numeric():
    rows = long_history({"NAN": steady(seed=1), "BADDATE": steady(seed=2), "TEXT": steady(seed=3), "OK": steady(seed=4)})
    rows["quantity"] = rows["quantity"].astype(object)
    rows.loc[(rows["product_id"] == "NAN") & (rows["date"] == CUTOFF - pd.Timedelta(days=40)), "quantity"] = np.nan
    rows.loc[(rows["product_id"] == "TEXT") & (rows["date"] == CUTOFF - pd.Timedelta(days=3)), "quantity"] = "abc"
    rows["date"] = rows["date"].astype(object)
    rows.loc[(rows["product_id"] == "BADDATE") & (rows["date"] == CUTOFF - pd.Timedelta(days=5)), "date"] = "not a date"
    out, diagnostics = router.route_demand_forecast(v1(inventory(["NAN", "BADDATE", "TEXT", "OK"])), rows)
    assert out["demand_forecast_reason"].tolist() == [router.REASON_INVALID] * 3 + [router.REASON_V2]
    assert diagnostics["non_finite_quantity_rows"] == 2 and diagnostics["unparseable_date_rows"] == 1
    inf = long_history({"P1": steady()})
    inf.loc[5, "quantity"] = np.inf
    assert route_rows(["P1"], inf)[0] == router.REASON_INVALID


def route_rows(products, rows, **kwargs):
    out, _ = router.route_demand_forecast(v1(inventory(products)), rows, **kwargs)
    return out["demand_forecast_reason"].tolist()


def test_duplicate_date_is_neither_summed_nor_picked():
    rows = long_history({"DUP": steady(seed=1), "OK": steady(seed=2)})
    repeated = rows[(rows["product_id"] == "DUP") & (rows["date"] == CUTOFF - pd.Timedelta(days=9))]
    for extra in (repeated, repeated.assign(quantity=repeated["quantity"] + 5)):   # identical or conflicting repeat
        out, diagnostics = router.route_demand_forecast(v1(inventory(["DUP", "OK"])), pd.concat([rows, extra], ignore_index=True))
        assert out["demand_forecast_reason"].tolist() == [router.REASON_INVALID, router.REASON_V2]
        assert diagnostics["duplicate_date_series"] == 1
    # A repeat after the cutoff is future data: never read, so it cannot invalidate the series.
    future = rows[rows["product_id"] == "DUP"].tail(1).assign(date=CUTOFF + pd.Timedelta(days=2))
    assert route_rows(["DUP", "OK"], pd.concat([rows, future, future], ignore_index=True), as_of=CUTOFF) == [router.REASON_V2] * 2


def test_missing_dates_are_not_zero_sales():
    history = steady(days=200, seed=8)
    gappy = history.copy()
    gappy[120:130] = np.nan                                      # 10 missing days, long before the recent window
    zeros = history.copy()
    zeros[120:130] = 0.0                                         # the same days as observed zero sales
    out, _ = route(["GAP", "ZERO"], {"GAP": gappy, "ZERO": zeros})
    assert out["demand_forecast_reason"].tolist() == [router.REASON_V2, router.REASON_V2]
    expected = core.forecast_v2(np.vstack([gappy, zeros]), router.v2_config(), groups=np.array([0, 0]))
    np.testing.assert_array_equal(out["demand_forecast_7d"].to_numpy(), np.round(expected.aggregate_7d, 1))
    assert out["demand_forecast_7d"].iloc[0] != out["demand_forecast_7d"].iloc[1]
    # A missing day inside the last 30 days is insufficient history; an observed zero there is not.
    recent_gap, recent_zero = history.copy(), history.copy()
    recent_gap[-10] = np.nan
    recent_zero[-10] = 0.0
    assert route(["A", "B"], {"A": recent_gap, "B": recent_zero})[0]["demand_forecast_reason"].tolist() == \
        [router.REASON_INSUFFICIENT, router.REASON_V2]


def test_zero_demand_rows_are_preserved_as_observed_zeros():
    intermittent = np.tile([3.0, 0.0, 0.0, 0.0], 60)             # 25% selling days, zeros are real observations
    out, _ = route(["P1"], {"P1": intermittent})
    profile = core.demand_profile(intermittent[None, :])
    assert profile.nonzero_ratio[0] == pytest.approx(0.25) and core.DEMAND_TYPES[profile.demand_type[0]] == "intermittent"
    result = core.forecast_v2(intermittent[None, :], router.v2_config())
    assert out["demand_forecast_reason"].iloc[0] == router.REASON_V2
    assert out["demand_forecast_7d"].iloc[0] == np.round(result.aggregate_7d[0], 1)
    assert 3.0 <= out["demand_forecast_7d"].iloc[0] <= 6.0         # every whole week holds 1 or 2 sales of 3 units
    only_sales = long_history({"P1": intermittent})
    only_sales = only_sales[only_sales["quantity"] > 0]          # dropping the zero rows turns them into missing days
    assert route_rows(["P1"], only_sales) == [router.REASON_INSUFFICIENT]


def test_future_rows_are_excluded_and_cannot_change_the_forecast():
    history = steady(days=240, seed=9)
    past = long_history({"P1": history[:200]})
    future = long_history({"P1": np.full(40, 500.0)}, cutoff=CUTOFF + pd.Timedelta(days=40))
    frame = v1(inventory(["P1"], snapshot_date=str(CUTOFF.date())))
    clean, _ = router.route_demand_forecast(frame, past)
    leaked, diagnostics = router.route_demand_forecast(frame, pd.concat([past, future], ignore_index=True))
    pd.testing.assert_frame_equal(clean, leaked, check_exact=True)
    assert diagnostics["future_rows_excluded"] == 40 and diagnostics["cutoffs"] == [str(CUTOFF.date())]
    explicit, _ = router.route_demand_forecast(v1(inventory(["P1"])), pd.concat([past, future]), as_of=CUTOFF)
    pd.testing.assert_frame_equal(explicit[clean.columns.drop("snapshot_date")], clean.drop(columns="snapshot_date"), check_exact=True)
    # Without snapshot_date or as_of, the cutoff is the latest history date (the future rows then ARE the history).
    _, latest = router.route_demand_forecast(v1(inventory(["P1"])), pd.concat([past, future]))
    assert latest["cutoffs"] == [str((CUTOFF + pd.Timedelta(days=40)).date())]
    with pytest.raises(ValueError):
        router.assess_history(inventory(["P1"]), past, as_of="not a date")


def test_each_snapshot_date_is_its_own_cutoff():
    history = steady(days=260, seed=10)
    early = CUTOFF - pd.Timedelta(days=30)
    rows = pd.concat([long_history({"A": history}), long_history({"B": history})], ignore_index=True)
    frame = v1(inventory(["A", "B"], snapshot_date=[str(early.date()), str(CUTOFF.date())]))
    out, diagnostics = router.route_demand_forecast(frame, rows)
    assert diagnostics["cutoffs"] == [str(early.date()), str(CUTOFF.date())] and diagnostics["future_rows_excluded"] == 30
    expected_a = core.forecast_v2(history[None, :-30], router.v2_config()).aggregate_7d[0]
    expected_b = core.forecast_v2(history[None, :], router.v2_config()).aggregate_7d[0]
    assert out["demand_forecast_7d"].tolist() == [np.round(expected_a, 1), np.round(expected_b, 1)]


def test_keys_are_matched_like_the_workbook_loader_and_ambiguity_falls_back():
    rows = long_history({"101": steady(seed=11)}, store="7")
    rows["store_id"], rows["product_id"] = 7.0, 101                 # numeric ids from a spreadsheet
    frame = v1(inventory(["101"]).assign(store_id=" 7 "))
    assert route_rows_frame(frame, rows) == [router.REASON_V2]
    canonical = long_history({"P1": steady(seed=12)}).rename(columns={"store_id": "location_id", "quantity": "sales_qty"})
    assert route_rows(["P1"], canonical) == [router.REASON_V2]
    shared = v1(pd.concat([inventory(["P1"]), inventory(["P1"])], ignore_index=True))   # two inventory rows, one series
    out, diagnostics = router.route_demand_forecast(shared, long_history({"P1": steady(seed=13)}))
    assert out["demand_forecast_reason"].tolist() == [router.REASON_INVALID] * 2 and diagnostics["ambiguous_inventory_rows"] == 2
    unmatched = pd.concat([long_history({"P1": steady(seed=14)}), long_history({"OTHER": steady(seed=15)})])
    _, diagnostics = router.route_demand_forecast(v1(inventory(["P1"])), unmatched)
    assert diagnostics["rows_without_inventory_match"] == 200
    broken = long_history({"P1": steady()}).drop(columns="quantity")
    out, diagnostics = router.route_demand_forecast(v1(inventory(["P1"])), broken)
    assert out["demand_forecast_reason"].tolist() == [router.REASON_INVALID] and "quantity" in diagnostics["contract_error"]


def route_rows_frame(frame, rows):
    out, _ = router.route_demand_forecast(frame, rows)
    return out["demand_forecast_reason"].tolist()


# ---------------------------------------------------------------- failure handling


def test_v2_exception_falls_back_to_v1_and_is_reported(monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic v2 failure")

    base = v1(inventory(["OK", "COLD"]))
    history = long_history({"OK": steady(), "COLD": np.r_[np.zeros(190), np.ones(10)]})
    monkeypatch.setattr(router.core, "forecast_v2", boom)
    with caplog.at_level(logging.WARNING, logger="services.demand_forecast_router"):
        out, diagnostics = router.route_demand_forecast(base, history)
    assert out["demand_forecast_reason"].tolist() == [router.REASON_ERROR, router.REASON_COLD_START]
    pd.testing.assert_frame_equal(out[list(base.columns)], base, check_exact=True)
    assert diagnostics["errors"][0]["error_type"] == "RuntimeError" and "synthetic v2 failure" in diagnostics["errors"][0]["message"]
    assert any("RuntimeError" in record.getMessage() and record.exc_info for record in caplog.records)
    # Inside the pipeline: the inventory analysis still completes, with a user-facing warning and no technical error.
    runner = _Runner(PipelineResult())
    analyzed, summaries = _run_inventory_analysis(runner, prepare_legacy_data({"inventory": inventory(["OK"])})["inventory"],
                                                  daily_sales_history=history)
    assert analyzed["demand_forecast_version"].tolist() == ["v1"] and not runner.technical_errors
    assert summaries["demand_forecast"]["forecast_router"]["errors"] and any("V2" in w for w in runner.result.warnings)
    assert "services.demand_forecast_router.route_demand_forecast" not in runner.result.connected_algorithms


def test_assessment_exception_falls_back_for_every_row(monkeypatch):
    monkeypatch.setattr(router, "assess_history", lambda *a, **k: (_ for _ in ()).throw(KeyError("broken")))
    base = v1(inventory(["P1", "P2"]))
    out, diagnostics = router.route_demand_forecast(base, long_history({"P1": steady()}))
    assert out["demand_forecast_reason"].tolist() == [router.REASON_ERROR] * 2
    pd.testing.assert_frame_equal(out[list(base.columns)], base, check_exact=True)
    assert diagnostics["errors"][0]["stage"] == "history_assessment"


# ---------------------------------------------------------------- output contract


def test_daily_vector_scalar_contract_rounding_and_dtypes():
    series = {f"P{k}": steady(level=0.5 + k, seed=k) for k in range(8)}
    series["COLD"] = np.r_[np.zeros(190), np.full(10, 2.0)]
    base = v1(inventory(list(series)))
    out, _ = router.route_demand_forecast(base, long_history(series))
    is_v2 = out["demand_forecast_version"] == "v2"
    assert is_v2.sum() == 8 and (~is_v2).sum() == 1
    daily = out[list(router.DAILY_COLUMNS)].to_numpy(dtype=float)
    total = out["demand_forecast_7d"].to_numpy()
    assert np.abs(daily[is_v2].sum(axis=1) - total[is_v2]).max() <= 0.05 + 1e-12      # 7-day total = rounded vector sum
    np.testing.assert_allclose(daily[~is_v2].sum(axis=1), total[~is_v2], rtol=0, atol=1e-9)
    np.testing.assert_array_equal(np.round(total * 10), total * 10)                     # 0.1 rounding like V1
    np.testing.assert_array_equal(out["demand_forecast_daily"], np.round(total / 7, 2))
    for column in ("demand_forecast_upper", "demand_forecast_lower"):
        np.testing.assert_array_equal(out[column], np.round(out[column], 2))
    np.testing.assert_array_equal(out["demand_risk_score"], np.round(out["demand_risk_score"], 1))
    assert out["demand_forecast_score"].tolist() == out["demand_risk_score"].tolist()
    assert {c: str(out[c].dtype) for c in V1_COLUMNS} == {c: str(base[c].dtype) for c in V1_COLUMNS}
    # The 7-day unit is unchanged: a series selling a steady 4/day gets ~28 over 7 days, 4/day as the daily rate.
    flat, _ = route(["P1"], {"P1": np.full(200, 4.0)})
    assert flat["demand_forecast_7d"].iloc[0] == 28.0 and flat["demand_forecast_daily"].iloc[0] == 4.0


def test_v2_rows_reuse_v1_formulas_for_dependent_fields():
    frame = inventory([f"P{k}" for k in range(6)], sales_7d=[14.0, 30.0, 7.0, 50.0, 3.0, 21.0],
                      demand_std=[0.5, 0.0, 2.0, 1.0, 0.0, 3.0], stock_qty=[5.0, 60.0, 0.0, 200.0, 12.0, 30.0],
                      lead_time_days=[1.0, 3.0, 7.0, 2.0, 0.0, 5.0])
    base = v1(frame)
    assert set(base["demand_forecast_method"]) == {"WMA"}
    fields = router.v1_dependent_fields(base, base["demand_forecast_7d"].to_numpy() / 7)
    for column, values in fields.items():
        np.testing.assert_array_equal(values, base[column].to_numpy())


def test_deterministic_output():
    series = {f"P{k}": steady(level=1 + k, seed=20 + k) for k in range(5)}
    first = router.route_demand_forecast(v1(inventory(list(series))), long_history(series))
    second = router.route_demand_forecast(v1(inventory(list(series))), long_history(series).sample(frac=1.0, random_state=3))
    pd.testing.assert_frame_equal(first[0], second[0], check_exact=True)
    assert first[1] == second[1]


# ---------------------------------------------------------------- production V2 == research V2


def test_production_router_equals_research_core_on_a_panel():
    rng = np.random.default_rng(31)
    n, t = 40, 420
    weekly = 1 + 0.4 * np.sin(np.arange(t) * 2 * np.pi / 7)
    sales = rng.poisson(rng.uniform(0.05, 6, (n, 1)) * weekly, (n, t)).astype(float)
    sales[3, :-20] = 0                                            # cold start
    sales[4] = (rng.random(t) < 0.05) * 1.0                       # mostly zero
    stores = np.where(np.arange(n) % 2, "S1", "S2")
    categories = np.where(np.arange(n) % 3, "A", "B")
    frame = pd.DataFrame({"store_id": stores, "product_id": [f"I{k:02d}" for k in range(n)], "category": categories,
                          "product_name": "x", "sales_7d": sales[:, -7:].sum(1), "sales_30d": sales[:, -30:].sum(1),
                          "avg_daily_sales": sales[:, -30:].sum(1) / 30, "demand_std": sales[:, -30:].std(1, ddof=1)})
    days = pd.date_range(end=CUTOFF, periods=t)
    rows = pd.DataFrame({"store_id": np.repeat(stores, t), "product_id": np.repeat(frame["product_id"], t),
                         "date": np.tile(days, n), "quantity": sales.ravel()})
    out, _ = router.route_demand_forecast(v1(frame), rows)
    groups = pd.factorize(pd.Series(stores) + "|" + categories, sort=True)[0]
    research = core.forecast_v2(sales, core.make_config(router.V2_ROUTES), groups=groups)
    is_v2 = out["demand_forecast_version"].to_numpy() == "v2"
    assert is_v2.sum() == n - 2 and out["demand_forecast_reason"].iloc[[3, 4]].tolist() == [router.REASON_COLD_START,
                                                                                               router.REASON_MOSTLY_ZERO]
    np.testing.assert_array_equal(out["demand_forecast_7d"].to_numpy()[is_v2], np.round(research.aggregate_7d, 1)[is_v2])
    np.testing.assert_array_equal(out[list(router.DAILY_COLUMNS)].to_numpy(dtype=float)[is_v2], research.daily[is_v2, :7])
    assert (research.weekday_applied[is_v2]).all()


# ---------------------------------------------------------------- pipeline and workbook wiring


def fixture_history(days=150, seed=40):
    rng = np.random.default_rng(seed)
    data = sample_workbook()
    rows = [{"store_id": r.store_id, "product_id": r.product_id, "date": d, "quantity": float(q)}
            for r in data["inventory"].itertuples() for d, q in zip(pd.date_range(end=CUTOFF, periods=days), rng.poisson(8, days))]
    return pd.DataFrame(rows)


def test_pipeline_uses_v2_with_history_and_is_unchanged_without():
    data = sample_workbook()
    without = run_analysis_pipeline(data)
    with_history = run_analysis_pipeline({**data, "daily_sales_history": fixture_history()})
    router_info = with_history.demand_analysis["demand_forecast"]["forecast_router"]
    assert router_info["counts_by_version"] == {"v2": len(data["inventory"])} and router_info["history_supplied"]
    assert without.demand_analysis["demand_forecast"]["forecast_router"]["counts_by_reason"] == {router.REASON_MISSING: len(data["inventory"])}
    assert with_history.demand_analysis["demand_forecast"]["function"] == "demand_forecast_analyzer.analyze_demand_forecast"
    assert "services.demand_forecast_router.route_demand_forecast" in with_history.connected_algorithms
    assert "services.demand_forecast_router.route_demand_forecast" not in without.connected_algorithms
    assert with_history.status == without.status == "success"


def test_workbook_daily_sales_history_sheet_reaches_the_router():
    data = sample_workbook()
    history = fixture_history()
    names = data["stores"].set_index("node_id")["node_name"]
    history.loc[history["store_id"] == "S002", "store_id"] = names["S002"]      # a store written by name, as inventory allows
    loaded = load_excel_data(workbook_excel_bytes({**data, "일별판매이력": history}))
    assert "daily_sales_history" in loaded and set(loaded["daily_sales_history"]["store_id"]) == set(data["inventory"]["store_id"])
    analyzed, summaries = _run_inventory_analysis(_Runner(PipelineResult()), prepare_legacy_data(loaded)["inventory"],
                                                  daily_sales_history=loaded["daily_sales_history"])
    assert set(analyzed["demand_forecast_version"]) == {"v2"}
    assert "daily_sales_history" not in load_excel_data(workbook_excel_bytes(data))

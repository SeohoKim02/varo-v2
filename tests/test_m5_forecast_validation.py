"""M5 external forecast validation: tiny explicit fixtures run through the real canonical pipeline, no production data."""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from services.analysis_pipeline import PipelineResult, _run_inventory_analysis, _Runner
from services.canonical_data_pipeline import Exporter
from services.external_canonical_pipeline import run_m5
from services.legacy_adapters.data_adapter import prepare_legacy_data
from services.m5_forecast_validation import (
    FREQUENCY_CLASSES, METHOD_ORDER, ORIGIN_ATTRIBUTES, OUTPUT_FILES, RAW_DIR, RAW_INPUTS, SBC_CLASSES, M5Panel,
    attach_origin_attributes, elementwise_stats, evaluate_origin, file_fingerprints, finalize_metrics, load_panel, naive_scales,
    point_metrics, raw_crosscheck, rolling_origins, run_validation, run_varo_forecast, scaled_errors, seasonal_naive_7,
    series_profile, varo_inputs)
from services.real_data_adapters import DATASETS
from tests.test_canonical_integrity import write_csv

DAYS = 70
SMALL = {"horizon": 7, "origin_count": 3, "origin_step": 7}   # cutoffs d_49, d_56, d_63
SERIES = [("FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA"), ("FOODS_1_001", "FOODS_1", "FOODS", "TX_1", "TX"),
          ("HOBBIES_1_002", "HOBBIES_1", "HOBBIES", "CA_1", "CA"), ("HOUSEHOLD_2_003", "HOUSEHOLD_2", "HOUSEHOLD", "TX_1", "TX")]
# Missing cells: d_26 sits inside the first cutoff's 30-day input window; d_65 is a holdout target day.
MISSING = {(1, 25), (1, 64)}


def sales_matrix(days=DAYS):
    i = np.arange(days)
    return np.vstack([np.array([2, 2, 2, 3, 4, 6, 5])[i % 7],        # weekly seasonal, sells daily
                      (i % 3 == 0).astype(int),                      # intermittent
                      np.isin(i, [20, 45, 66]).astype(int) * 2,      # mostly zero
                      np.where(i < 40, 0, 1 + i % 2)]).astype(float)  # launched on day 41


def write_fixture(root, days=DAYS):
    folder = root / DATASETS["m5"] / "raw/extracted/m5-forecasting-accuracy"
    dates = pd.date_range("2011-01-29", periods=days)
    calendar = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "wm_yr_wk": [str(11101 + i // 7) for i in range(days)],
                             "weekday": dates.day_name(), "wday": [str(i % 7 + 1) for i in range(days)], "month": dates.month.astype(str),
                             "year": dates.year.astype(str), "d": [f"d_{i + 1}" for i in range(days)],
                             "event_name_1": None, "event_type_1": None, "event_name_2": None, "event_type_2": None,
                             "snap_CA": [str(int(d.day <= 10)) for d in dates], "snap_TX": [str(int(d.day % 2)) for d in dates],
                             "snap_WI": "0"})
    calendar.loc[63, ["event_name_1", "event_type_1"]] = ["SuperBowl", "Sporting"]
    values = sales_matrix(days).astype(int).astype(str).astype(object)
    for row, day in MISSING:
        values[row, day] = None
    meta = pd.DataFrame(SERIES, columns=["item_id", "dept_id", "cat_id", "store_id", "state_id"])
    meta.insert(0, "id", meta.item_id + "_" + meta.store_id + "_evaluation")
    sales = pd.concat([meta, pd.DataFrame(values, columns=[f"d_{i + 1}" for i in range(days)])], axis=1)
    prices = []
    for (item, _, _, store, _), row in zip(SERIES, sales_matrix(days)):
        for week in range(days // 7):
            if item == "HOUSEHOLD_2_003" and row[week * 7:week * 7 + 7].sum() == 0:
                continue   # not yet listed: no weekly price (M5 guide)
            price = {"FOODS_1_001": 3.0 if week != 9 else 2.5, "HOBBIES_1_002": 9.5, "HOUSEHOLD_2_003": 4.0}[item]
            prices.append({"store_id": store, "item_id": item, "wm_yr_wk": str(11101 + week), "sell_price": str(price)})
    write_csv(folder / "calendar.csv", calendar)
    write_csv(folder / "sales_train_evaluation.csv", sales)
    # The prefix release stops before the first missing cell (the canonical release check compares cells, not NaN).
    write_csv(folder / "sales_train_validation.csv", sales.iloc[:, :6 + 20].assign(id=meta.item_id + "_" + meta.store_id + "_validation"))
    write_csv(folder / "sell_prices.csv", pd.DataFrame(prices))
    write_csv(folder / "sample_submission.csv", pd.DataFrame({"id": ["x_validation"], "F1": ["0"]}))
    exporter = Exporter(root, "m5")
    run_m5(exporter)
    exporter.finish()
    return root / DATASETS["m5"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp("m5root"))


def random_panel(seed=3, series=6, days=120):
    rng = np.random.default_rng(seed)
    sales = rng.poisson(rng.uniform(0.05, 4, size=(series, 1)), size=(series, days)).astype(float)
    price = np.where(rng.random((series, days)) < 0.1, np.nan, rng.uniform(1, 5, (series, days))).astype(np.float32)
    frame = pd.DataFrame({"store_id": "S", "item_id": [f"I{k}" for k in range(series)], "state_id": "CA", "category": "C", "department": "D"})
    dates = pd.date_range("2015-01-01", periods=days).strftime("%Y-%m-%d").to_numpy()
    events = np.where(np.arange(days) % 17 == 0, "National", "none").astype(object)
    snap = np.tile((np.arange(days) % 3 == 0).astype(np.float32), (series, 1))
    return M5Panel(sales, price, dates, frame, events, snap, series * days)


# ---------------------------------------------------------------- split

def test_chronological_split_and_rolling_cutoff_ordering():
    origins = rolling_origins(1941)
    cutoffs = [o["cutoff"] for o in origins]
    assert len(origins) == 13 and cutoffs == sorted(cutoffs) and len(set(cutoffs)) == 13
    # Target windows tile the final year: contiguous, non-overlapping, ending on the last day; each starts right after its cutoff.
    assert all(o["target_start"] == o["cutoff"] + 1 and o["target_end"] - o["target_start"] == 27 for o in origins)
    assert all(a["target_end"] + 1 == b["target_start"] for a, b in zip(origins, origins[1:]))
    assert origins[-1]["target_end"] == 1940 and origins[0]["target_start"] == 1940 - 13 * 28 + 1
    # Holdout = M5 d_1914..d_1941, validation = d_1886..d_1913 (0-based day index + 1 = d number).
    assert (origins[-1]["role"], origins[-1]["cutoff"] + 1) == ("holdout", 1913)
    assert (origins[-2]["role"], origins[-2]["cutoff"] + 1) == ("validation", 1885)
    assert {o["role"] for o in origins[:-2]} == {"rolling_backtest"}
    with pytest.raises(ValueError):
        rolling_origins(60, horizon=28, count=3, step=28)   # first cutoff would have fewer than 30 history days
    with pytest.raises(ValueError):
        rolling_origins(1941, horizon=28, step=14)          # overlapping targets are refused


def test_split_ranges_in_report(dataset, tmp_path):
    report = run_validation(dataset.parent, tmp_path, write_rows=False, **SMALL)
    train, validation, holdout = report["train_range"], report["validation_range"], report["holdout_range"]
    day = pd.Timedelta(days=1)
    assert pd.Timestamp(train[1]) + day == pd.Timestamp(validation[0]) and pd.Timestamp(validation[1]) + day == pd.Timestamp(holdout[0])
    assert holdout[1] == report["date_range"][1] == "2011-04-08" and report["holdout_history_range"][1] == validation[1]
    assert [c["cutoff_day"] for c in report["cutoffs"]] == ["d_49", "d_56", "d_63"]


# ---------------------------------------------------------------- leakage

def test_no_future_leakage():
    panel = random_panel()
    spec = rolling_origins(panel.sales.shape[1], horizon=28, count=3, step=28)[1]
    cutoff = spec["cutoff"]
    rng = np.random.default_rng(11)
    future = M5Panel(panel.sales.copy(), panel.price.copy(), panel.dates, panel.series, panel.event_label, panel.snap, panel.canonical_rows)
    future.sales[:, cutoff + 1:] = rng.poisson(50, size=future.sales[:, cutoff + 1:].shape)
    future.price[:, cutoff + 1:] = 0.01
    base, changed = evaluate_origin(panel, spec, 28), evaluate_origin(future, spec, 28)
    for method in METHOD_ORDER:
        np.testing.assert_array_equal(base["daily"][method], changed["daily"][method])
    history_only = [c for c in base["profile"].columns if not c.startswith("actual_")]
    pd.testing.assert_frame_equal(base["profile"][history_only], changed["profile"][history_only])
    # The targets did change, so the scores must differ: the forecasts are simply blind to them.
    assert not np.array_equal(base["actual"], changed["actual"])
    assert not base["series_stats"]["abs_err"].equals(changed["series_stats"]["abs_err"])


# ---------------------------------------------------------------- zero vs missing

def test_zero_sales_preservation(dataset):
    panel = load_panel(dataset)
    expected = sales_matrix()
    observed = ~np.isnan(panel.sales)
    # Rows are (store, item) sorted: CA_1 FOODS, CA_1 HOBBIES, TX_1 FOODS, TX_1 HOUSEHOLD.
    order = [0, 2, 1, 3]
    np.testing.assert_array_equal(panel.sales[observed], expected[order][observed])
    assert int((panel.sales == 0).sum()) == int((expected[order] == 0).sum() - sum(expected[r, d] == 0 for r, d in MISSING))
    # A zero actual is a scored point with error = forecast, not a dropped one.
    metrics = point_metrics(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    assert metrics["n"] == 2 and metrics["mae"] == 0.5 and np.isnan(metrics["wape"])


def test_missing_is_not_zero(dataset, tmp_path):
    panel = load_panel(dataset)
    tx_foods = 2
    assert np.isnan(panel.sales[tx_foods, 25]) and np.isnan(panel.sales[tx_foods, 64])
    assert raw_crosscheck(dataset, panel)["cells_equal_including_missing"]
    # Missing actuals are excluded from the point count; zero-filling them would add two scored points.
    assert point_metrics(np.array([1.0, 1.0, 1.0]), np.array([1.0, np.nan, 0.0]))["n"] == 2
    assert point_metrics(np.array([1.0, 1.0, 1.0]), np.array([1.0, 0.0, 0.0]))["n"] == 3
    report = run_validation(dataset.parent, tmp_path, write_rows=False, **SMALL)
    runs = {r["origin"]: r for r in report["per_origin_run"]}
    # d_26 is inside origin 1's 30-day input window only: that series is excluded there instead of reading NaN as 0.
    assert [runs[k]["excluded_series_missing_history"] for k in (1, 2, 3)] == [1, 0, 0]
    assert [runs[k]["missing_target_points"] for k in (1, 2, 3)] == [0, 0, 1]
    by_cutoff = pd.read_csv(tmp_path / OUTPUT_FILES["by_cutoff"])
    points = by_cutoff[(by_cutoff.window == "daily_h1_7") & (by_cutoff.method == "varo_production")].set_index("origin")["points"]
    assert points.to_dict() == {1: 21, 2: 28, 3: 27}
    assert report["data_profile"]["sales_missing_cells"] == 2


# ---------------------------------------------------------------- forecasts

def test_seasonal_naive_correctness():
    history = np.array([[9, 9, 1, 2, 3, 4, 5, 6, 7]], dtype=float)
    # Last week is days 3..9 = [1..7]; horizon day h takes the same weekday, repeating weekly.
    np.testing.assert_array_equal(seasonal_naive_7(history, 10), [[1, 2, 3, 4, 5, 6, 7, 1, 2, 3]])


def test_production_forecast_formula_and_fallback():
    history = np.zeros((3, 30))
    history[0, -7:] = 2          # sales_7d = 14, sales_30d = 30 -> WMA 0.6*2 + 0.4*1 = 1.6/day, 11.2 per week
    history[0, 0] = 16
    history[1, :10] = 3          # sales_7d = 0, sales_30d = 30 -> per-row NAIVE fallback = 1.0/day, not 0.4
    out = run_varo_forecast(varo_inputs(history), ("sales_7d", "avg_daily_sales", "sales_30d", "demand_std"))
    assert out["demand_forecast_7d"].tolist() == [11.2, 7.0, 0.0]
    assert out["demand_forecast_daily"].tolist() == [1.6, 1.0, 0.0]
    assert out["demand_forecast_method"].tolist() == ["WMA", "NAIVE", "NAIVE"]
    # Without sales_7d the same function takes its NAIVE branch; with only sales_7d, its SMA branch.
    naive = run_varo_forecast(varo_inputs(history), ("avg_daily_sales", "sales_30d", "demand_std"))
    sma = run_varo_forecast(varo_inputs(history), ("sales_7d",))
    assert naive["demand_forecast_daily"].tolist() == [1.0, 1.0, 0.0] and set(naive["demand_forecast_method"]) == {"NAIVE"}
    assert sma["demand_forecast_daily"].tolist() == [2.0, 0.0, 0.0] and set(sma["demand_forecast_method"]) == {"SMA"}


def test_evaluator_matches_production_pipeline_step():
    panel = random_panel(series=12, days=60)
    inputs = varo_inputs(panel.sales).assign(store_id="S1", product_id=[f"P{k}" for k in range(12)], product_name="x")
    direct = run_varo_forecast(inputs, ("sales_7d", "avg_daily_sales", "sales_30d", "demand_std"))
    # The analysis pipeline's own inventory step (abc -> turnover -> disposal -> demand forecast -> ...).
    analyzed, summaries = _run_inventory_analysis(_Runner(PipelineResult()), prepare_legacy_data({"inventory": inputs})["inventory"])
    assert summaries["demand_forecast"]["status"] == "연결"
    analyzed = analyzed.set_index("product_id").loc[inputs["product_id"]]
    for column in ("demand_forecast_7d", "demand_forecast_daily", "demand_forecast_method", "demand_trend"):
        assert analyzed[column].tolist() == direct[column].tolist()


def test_demand_type_classification():
    days = 400
    history = np.zeros((5, days))
    history[0] = 1                                   # sells every day
    history[1, ::3] = 1                              # 1 day in 3
    history[2, ::20] = 1                             # 1 day in 20
    history[3, -25:] = 2                             # launched 25 days ago, sells daily
    profile = series_profile(history)                # row 4 never sold
    classes = [FREQUENCY_CLASSES[c] for c in profile["frequency_class"]]
    assert classes == ["high_frequency", "intermittent", "mostly_zero", "high_frequency", "no_sales_history"]
    assert profile["days_since_first_sale"].tolist() == [400, 400, 400, 25, -1]
    assert [SBC_CLASSES[c] for c in profile["sbc_class"]] == ["smooth", "intermittent", "intermittent", "smooth", "undefined"]
    # The window starts at the first sale: pre-launch zeros do not make the new product look intermittent.
    assert profile["selling_day_share"][3] == 1.0


def test_origin_attribute_join_at_full_m5_scale():
    # 13 origins x 30,490 series: an int16 origin key would wrap and attach another series' class.
    n = 30490
    profile = pd.DataFrame({"origin": np.repeat(np.arange(1, 14), n), "series": np.tile(np.arange(n), 13)})
    for column in ORIGIN_ATTRIBUTES:
        profile[column] = np.arange(len(profile)) % 127
    stats = pd.DataFrame({"origin": np.array([1, 2, 13, 13], dtype=np.int16), "series": np.array([5, 7, 0, n - 1], dtype=np.int32)})
    attach_origin_attributes(stats, profile, n)
    expected = ((stats["origin"].astype(int) - 1) * n + stats["series"]) % 127
    for column in ORIGIN_ATTRIBUTES:
        assert stats[column].tolist() == expected.tolist()


def test_segment_tables_match_series_profile(dataset, tmp_path):
    run_validation(dataset.parent, tmp_path, write_rows=False, **SMALL)
    profile = pd.read_parquet(tmp_path / OUTPUT_FILES["series_profile"])
    table = pd.read_csv(tmp_path / OUTPUT_FILES["by_demand_type"])
    for origin, scope in ((3, "holdout"), (2, "validation")):
        rows = table[(table.scope == scope) & (table.window == "total_7d") & (table.method == "varo_production")]
        eligible = profile[(profile.origin == origin) & profile.eligible & profile.actual_7d.notna()]
        for dimension in ("frequency_class", "age_band", "varo_method"):
            counts = rows[rows.dimension == dimension].set_index("segment")["series_origins"].to_dict()
            assert counts == eligible[dimension].value_counts().to_dict()


# ---------------------------------------------------------------- metrics

def test_metric_correctness():
    f, a = np.array([2.0, 0.0, 3.0, 1.0]), np.array([1.0, 0.0, 5.0, 1.0])
    m = point_metrics(f, a)
    assert m["mae"] == pytest.approx(0.75) and m["rmse"] == pytest.approx(np.sqrt(5 / 4)) and m["wape"] == pytest.approx(3 / 7)
    assert np.isnan(point_metrics(np.array([1.0]), np.array([0.0]))["wape"])
    # Pooled metrics from summed statistics equal metrics on the concatenated points (additivity used by every group-by).
    parts = [elementwise_stats(f[:2], a[:2], np.ones(2, bool)), elementwise_stats(f[2:], a[2:], np.ones(2, bool))]
    summed = pd.DataFrame([{k: float(p[k].sum()) for k in p} for p in parts]).sum().to_frame().T
    row = finalize_metrics(summed).iloc[0]
    assert (row["mae"], row["wape"]) == (pytest.approx(m["mae"]), pytest.approx(m["wape"]))
    # MASE / RMSSE: scale from one-step naive differences after the first sale; leading zeros excluded.
    history = np.array([[0, 0, 2, 4, 2, 4]], dtype=float)
    mae_scale, mse_scale = naive_scales(history, np.array([2]))
    assert (mae_scale[0], mse_scale[0]) == (2.0, 4.0)
    mase, rmsse = scaled_errors(np.array([3.0]), np.array([5.0]), np.array([2]), mae_scale, mse_scale)
    assert (mase[0], rmsse[0]) == (pytest.approx(0.75), pytest.approx(np.sqrt(5 / 2 / 4)))


def test_forecast_bias_correctness():
    f, a = np.array([2.0, 0.0, 3.0, 1.0, 4.0]), np.array([1.0, 0.0, 5.0, 1.0, 1.0])
    m = point_metrics(f, a)
    assert m["mean_error"] == pytest.approx((1 + 0 - 2 + 0 + 3) / 5)
    assert m["bias_units"] == pytest.approx(f.sum() - a.sum()) and m["bias_pct"] == pytest.approx((10 - 8) / 8)
    assert (m["over_forecast_rate"], m["under_forecast_rate"], m["exact_rate"]) == (pytest.approx(0.4), pytest.approx(0.2), pytest.approx(0.4))
    assert (m["over_units"], m["under_units"]) == (4.0, 2.0)
    assert m["over_units"] - m["under_units"] == pytest.approx(m["bias_units"])


# ---------------------------------------------------------------- end to end

def _outputs(folder):
    return {name: (folder / name).read_bytes() for name in OUTPUT_FILES.values() if name.endswith(".csv")}


def test_deterministic_evaluation(dataset, tmp_path):
    first = run_validation(dataset.parent, tmp_path / "a", **SMALL)
    second = run_validation(dataset.parent, tmp_path / "b", **SMALL)
    assert _outputs(tmp_path / "a") == _outputs(tmp_path / "b")
    for name in ("rows", "series_profile"):
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "a" / OUTPUT_FILES[name]), pd.read_parquet(tmp_path / "b" / OUTPUT_FILES[name]))
    strip = lambda r: {k: v for k, v in r.items() if k not in {"runtime", "evaluation", "per_origin_run"}}
    assert strip(first) == strip(second) and first["random_seed"] is None
    saved = json.loads((tmp_path / "a" / OUTPUT_FILES["json"]).read_text(encoding="utf-8"))
    for key in ("data_signature", "date_range", "train_range", "validation_range", "holdout_range", "forecast_horizon", "cutoffs",
                "algorithm_version", "parameters", "metrics", "baseline_definitions", "row_count", "product_count", "store_count", "runtime"):
        assert key in saved
    rows = pq.read_table(tmp_path / "a" / OUTPUT_FILES["rows"]).to_pandas()
    assert len(rows) == 3 * 4 * 7 and set(f"forecast_{m}" for m in METHOD_ORDER) <= set(rows.columns)


def test_raw_untouched(dataset, tmp_path):
    raw = dataset / RAW_DIR
    before = file_fingerprints(raw, RAW_INPUTS)
    listing = sorted(p.name for p in raw.iterdir())
    processed = {p.name: p.stat().st_mtime_ns for p in (dataset / "processed").iterdir()}
    report = run_validation(dataset.parent, tmp_path, **SMALL)
    assert file_fingerprints(raw, RAW_INPUTS) == before and sorted(p.name for p in raw.iterdir()) == listing
    assert {p.name: p.stat().st_mtime_ns for p in (dataset / "processed").iterdir()} == processed
    assert report["data_signature"]["raw_unchanged_during_run"] is True
    assert report["data_signature"]["raw_canonical_crosscheck"]["cells_equal_including_missing"] is True
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(OUTPUT_FILES.values())

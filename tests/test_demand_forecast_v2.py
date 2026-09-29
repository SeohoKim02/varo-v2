"""Varo Demand Forecast v2 Core: hand-computed forecasts on tiny explicit histories (no production data)."""
import numpy as np
import pandas as pd
import pytest

from services import demand_forecast_v2 as core
from services.analysis_pipeline import PipelineResult, _run_inventory_analysis, _Runner
from services.legacy_adapters.data_adapter import prepare_legacy_data
from services.m5_forecast_validation import moving_average, series_profile

ALL_MA28 = {t: {"level": "ma28", "weekday": "flat"} for t in core.ROUTED_TYPES}


def block_of(history):
    history = np.asarray(history, dtype=float)
    profile = core.demand_profile(history)
    return core.recent_block(history, profile.first_sale)


# ---------------------------------------------------------------- V1 baseline preserved


def test_v1_baseline_is_frozen_and_still_the_production_forecast():
    current = core.v1_baseline_fingerprint()
    assert current["source_sha256_lf"] == core.FORECAST_V1_BASELINE["source_sha256_lf"]
    assert current["constants"] == core.FORECAST_V1_BASELINE["constants"]
    inventory = pd.DataFrame({"store_id": "S1", "product_id": ["P1", "P2", "P3"], "product_name": "x",
                              "sales_7d": [14.0, 0.0, 7.0], "sales_30d": [30.0, 30.0, 0.0], "avg_daily_sales": [1.0, 1.0, 0.0],
                              "demand_std": [0.5, 0.5, 0.0]})
    out = core.run_v1_baseline(inventory)
    # WMA 0.6*2 + 0.4*1 = 1.6/day; sales_7d = 0 -> NAIVE fallback avg_daily_sales; flat 7-day scalar.
    assert out["demand_forecast_7d"].tolist() == [11.2, 7.0, 4.2]
    assert out["demand_forecast_method"].tolist() == ["WMA", "NAIVE", "WMA"]
    analyzed, summaries = _run_inventory_analysis(_Runner(PipelineResult()), prepare_legacy_data({"inventory": inventory})["inventory"])
    assert summaries["demand_forecast"]["function"] == "demand_forecast_analyzer.analyze_demand_forecast"
    assert analyzed.set_index("product_id").loc[["P1", "P2", "P3"], "demand_forecast_7d"].tolist() == [11.2, 7.0, 4.2]


# ---------------------------------------------------------------- levels


def test_moving_average_correctness():
    t = 35
    history = np.zeros((4, t))
    history[0] = np.arange(t)                        # first sale at column 1
    history[1, -5:] = [2, 4, 6, 8, 10]               # launched 5 days ago
    history[2] = 1.0
    history[2, -1] = np.nan                          # missing cutoff day
    block, start = block_of(history)                 # row 3 never sold
    np.testing.assert_allclose(core.window_mean(block, start, 7), [31.0, 6.0, 1.0, 0.0])
    np.testing.assert_allclose(core.window_mean(block, start, 14), [27.5, 6.0, 1.0, 0.0])
    np.testing.assert_allclose(core.window_mean(block, start, 28), [20.5, 6.0, 1.0, 0.0])   # cold start: mean since first sale
    # The plain M5 baselines keep pre-launch zeros in the window (and are not NaN-aware).
    plain = np.nan_to_num(history)
    np.testing.assert_allclose(moving_average(plain, 3, 28)[:2, 0], [20.5, 30 / 28])
    np.testing.assert_allclose(moving_average(plain, 3, 7)[:, 1], [31.0, 30 / 7, 6 / 7, 0.0])
    np.testing.assert_allclose(moving_average(plain, 3, 14)[0], [27.5] * 3)


def test_weekly_ewma_weights_whole_weeks_only():
    history = np.repeat([[1.0, 2.0, 3.0]], 7, axis=1).reshape(1, 21)  # weekly totals 7, 14, 21 (newest last)
    block, start = block_of(history)
    weights = 0.2 * 0.8 ** np.array([2, 1, 0])
    expected = (weights * [7, 14, 21]).sum() / (7 * weights.sum())
    assert core.weekly_ewma(block, start, 0.2, 26)[0] == pytest.approx(expected)
    young = np.zeros((1, 21))
    young[0, -10:] = 1.0                              # first sale 10 days ago: the week holding it is partial -> excluded
    young[0, -7:] = 3.0
    block, start = block_of(young)
    assert core.weekly_ewma(block, start, 0.2, 26)[0] == pytest.approx(3.0)
    newborn = np.zeros((1, 21))
    newborn[0, -3:] = [0.0, 2.0, 4.0]                 # younger than a week: mean since the first sale
    block, start = block_of(newborn)
    assert core.weekly_ewma(block, start, 0.4, 26)[0] == pytest.approx(3.0)


def test_trend_is_damped_and_capped():
    history = np.ones((4, 28))
    history[0, -7:] = 3.0                             # MA7 = 3, MA28 = 1.5: ratio 2 -> sqrt(2)
    history[1, -7:] = 30.0                            # ratio 30 / 7.75 > cap 2 -> sqrt(2)
    history[2, -7:] = 0.0                             # ratio 0 -> floor 1/2 -> sqrt(1/2)
    history[3] = 0.0
    history[3, 0] = 1.0                               # sold once 27 days ago: MA7 = 0, MA28 > 0 -> floor
    block, start = block_of(history)
    np.testing.assert_allclose(core.trend_multiplier(block, start), [np.sqrt(2), np.sqrt(2), np.sqrt(0.5), np.sqrt(0.5)])
    base = core.level_forecast(block, start, "ma28")
    np.testing.assert_allclose(core.level_forecast(block, start, "ma28_trend"), base * core.trend_multiplier(block, start))


# ---------------------------------------------------------------- intermittent demand


def test_croston_sba_tsb_hand_computed():
    history = np.array([[0, 3, 0, 0, 2, 0, 1]], dtype=float)   # first sale at column 1: 6 active days, demands 3, 2, 1
    block, start = block_of(history)
    # z0 = 2, p0 = 6/3 = 2; alpha 0.5: first demand updates z only (2.5); then (q=3) z 2.25, p 2.5; then (q=2) z 1.625, p 2.25.
    assert core.croston(block, start, 0.5, "croston")[0] == pytest.approx(1.625 / 2.25)
    assert core.croston(block, start, 0.5, "sba")[0] == pytest.approx(1.625 / 2.25 * 0.75)
    # TSB: prob0 = 0.5, z0 = 2, alpha = beta = 0.5 -> prob 0.75, .375, .1875, .59375, .296875, .6484375; z ends 1.625.
    assert core.tsb(block, start, 0.5, 0.5)[0] == pytest.approx(0.6484375 * 1.625)
    # A zero is a demand-free period (the interval grows); a missing day is skipped (no update at all).
    missing = history.copy()
    missing[0, 2] = np.nan
    block, start = block_of(missing)
    # z0 = 2, p0 = 5/3; at column 4 the interval is 2 observed days: p = 5/3 + 0.5 (2 - 5/3); then q=2 again.
    p = 5 / 3 + 0.5 * (2 - 5 / 3)
    p = p + 0.5 * (2 - p)
    assert core.croston(block, start, 0.5, "croston")[0] == pytest.approx(1.625 / p)
    assert core.croston(block, start, 0.5, "croston")[0] != pytest.approx(1.625 / 2.25)
    never = np.zeros((1, 10))
    block, start = block_of(never)
    assert core.croston(block, start, 0.1, "sba")[0] == 0 and core.tsb(block, start, 0.1, 0.1)[0] == 0


def test_tsb_decays_after_obsolescence_but_croston_does_not():
    history = np.zeros((1, 200))
    history[0, :100:2] = 2.0                          # sold every other day, then 100 days of zeros
    block, start = block_of(history)
    assert core.tsb(block, start, 0.1, 0.1)[0] < 1e-3
    assert core.croston(block, start, 0.1, "croston")[0] > 0.5


def test_negative_demand_disables_croston_family():
    history = np.tile([[0.0, 2.0, 0.0, 0.0]], (2, 20))
    history[1, 10] = -1.0                             # a return
    profile = core.demand_profile(history)
    assert profile.has_negative.tolist() == [False, True]
    config = core.make_config({t: {"level": "sba_0.1", "weekday": "flat"} for t in core.ROUTED_TYPES})
    result = core.forecast_v2(history, config)
    block, start = core.recent_block(history, profile.first_sale)
    assert result.level_method.tolist() == ["sba_0.1", "ma28"] and result.negative_fallback.tolist() == [False, True]
    assert result.level[1] == pytest.approx(core.window_mean(block, start, 28)[1])
    assert result.level[0] == pytest.approx(core.croston(block, start, 0.1, "sba")[0])


# ---------------------------------------------------------------- weekday profile, routing, cold start


def test_weekday_profile_reproduces_a_periodic_series():
    pattern = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
    t = 70
    history = pattern[np.arange(t) % 7][None, :]
    for method in ("series_4w", "series_8w", "shrunk_8w", "pooled_8w"):
        config = core.make_config({tt: {"level": "ma28", "weekday": method} for tt in core.ROUTED_TYPES})
        result = core.forecast_v2(history, config, horizon=28)
        # Horizon day h is column t-1+h: the forecast continues the weekly pattern exactly.
        np.testing.assert_allclose(result.daily[0], pattern[(t - 1 + np.arange(1, 29)) % 7])
        assert result.aggregate_7d[0] == pytest.approx(28.0) and result.daily[0, :7].sum() == pytest.approx(28.0)
        assert result.weekday_applied[0]
    flat = core.forecast_v2(history, core.make_config(ALL_MA28), horizon=7)
    np.testing.assert_allclose(flat.daily[0], [4.0] * 7)


def test_weekday_credibility_shrinkage_and_cold_start():
    t = 56
    own = np.tile([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 8)[None, :t]         # all 16 units on one weekday
    other = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 70.0], 8)[None, :t]      # a heavy series in the same group
    history = np.vstack([own, other])
    block, start = block_of(history)
    groups = np.array([0, 0])
    series_index, _ = core.weekday_index(block, start, "series_8w", groups)
    pooled_index, _ = core.weekday_index(block, start, "pooled_8w", groups)
    shrunk_index, applied = core.weekday_index(block, start, "shrunk_8w", groups)
    z = 16 / (16 + core.WEEKDAY_SHRINK_UNITS)
    expected = z * series_index[0] + (1 - z) * pooled_index[0]
    np.testing.assert_allclose(shrunk_index[0], expected)
    # 'other' first sold on day 7: 50 days < the 56-day minimum, so it stays flat while 'own' (56 days) gets the mix.
    assert applied.tolist() == [True, False] and shrunk_index[0].mean() == pytest.approx(1.0) and (shrunk_index[1] == 1).all()
    young = np.zeros((1, 56))
    young[0, -20:] = np.tile([1.0, 5.0], 10)                                 # 20 days of history: < 28 -> flat
    block, start = block_of(young)
    for method in ("series_4w", "series_8w", "shrunk_8w", "pooled_8w"):
        index, applied = core.weekday_index(block, start, method)
        assert not applied[0] and (index[0] == 1).all()


def test_demand_type_routing():
    t = 400
    history = np.zeros((5, t))
    history[0] = 3.0                                  # high
    history[1, ::3] = 1.0
    history[1, 1::3] = 1.0                            # 2 of 3 days: medium
    history[2, ::4] = 2.0                             # intermittent
    history[3, ::25] = 1.0                            # mostly zero
    profile = core.demand_profile(history)            # row 4 never sold
    assert [core.DEMAND_TYPES[c] for c in profile.demand_type] == list(core.DEMAND_TYPES)
    routes = {"high_frequency": {"level": "ma7", "weekday": "flat"}, "medium_frequency": {"level": "ewma_w0.4", "weekday": "flat"},
              "intermittent": {"level": "sba_0.1", "weekday": "flat"}, "mostly_zero": {"level": "tsb_0.1_0.05", "weekday": "flat"}}
    result = core.forecast_v2(history, core.make_config(routes))
    block, start = core.recent_block(history, profile.first_sale)
    for row, route in enumerate(routes.values()):
        assert result.level_method[row] == route["level"]
        assert result.level[row] == pytest.approx(core.level_forecast(block, start, route["level"])[row])
    assert result.level_method[4] == "zero_no_sales_history" and (result.daily[4] == 0).all()
    changed = core.forecast_v2(history, core.make_config({**routes, "intermittent": {"level": "ma91", "weekday": "flat"}}))
    moved = ~np.isclose(changed.daily, result.daily).all(axis=1)
    assert moved.tolist() == [False, False, True, False, False]


def test_cold_start_fallback():
    history = np.zeros((2, 60))
    history[0, -3:] = [2.0, 4.0, 6.0]                 # first sale 3 days ago
    result = core.forecast_v2(history, core.make_config({t: {"level": "ma28", "weekday": "series_4w"} for t in core.ROUTED_TYPES}))
    assert result.profile.age.tolist() == [3, 0]
    assert result.level[0] == pytest.approx(4.0)                          # mean since the first sale, not 12/28
    assert not result.weekday_applied[0] and np.allclose(result.daily[0], 4.0)
    assert (result.daily[1] == 0).all() and core.DEMAND_TYPES[result.demand_type[1]] == "no_sales_history"
    np.testing.assert_allclose(core.forecast_v2(history, core.make_config(
        {t: {"level": "ewma_w0.2", "weekday": "flat"} for t in core.ROUTED_TYPES})).level, [4.0, 0.0])


def test_daily_vector_and_7d_aggregate_consistency():
    rng = np.random.default_rng(5)
    history = rng.poisson(rng.uniform(0.05, 8, (60, 1)), (60, 200)).astype(float)
    config = core.make_config({t: {"level": "ewma_w0.2_trend", "weekday": "shrunk_8w"} for t in core.ROUTED_TYPES})
    result = core.forecast_v2(history, config, horizon=28, groups=np.arange(60) % 3)
    np.testing.assert_allclose(result.daily[:, :7].sum(axis=1), result.aggregate_7d, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.daily[:, 7:14], result.daily[:, :7])  # flat level beyond week 1, same weekly shape
    out = core.compatible_output(result)
    np.testing.assert_allclose(out[[f"demand_forecast_d{h}" for h in range(1, 8)]].sum(axis=1), result.aggregate_7d)
    assert (out["demand_forecast_7d"] == np.round(result.aggregate_7d, 1)).all()
    assert (out["demand_forecast_daily"] == (out["demand_forecast_7d"] / 7).round(2)).all()
    assert (out["demand_forecast_7d"] - out[[f"demand_forecast_d{h}" for h in range(1, 8)]].sum(axis=1)).abs().max() <= 0.05 + 1e-12
    assert {"demand_type", "demand_forecast_route", "demand_forecast_version"} <= set(out.columns)


def test_zero_is_demand_and_missing_is_skipped():
    zeros = np.array([[1.0, 0.0, 1.0, 0.0] * 10])
    missing = zeros.copy()
    missing[0, 1::2] = np.nan
    a, b = core.demand_profile(zeros), core.demand_profile(missing)
    assert a.nonzero_ratio[0] == pytest.approx(0.5) and b.nonzero_ratio[0] == pytest.approx(1.0)
    assert core.DEMAND_TYPES[a.demand_type[0]] == "medium_frequency" and core.DEMAND_TYPES[b.demand_type[0]] == "high_frequency"
    ra, rb = core.forecast_v2(zeros, core.make_config(ALL_MA28)), core.forecast_v2(missing, core.make_config(ALL_MA28))
    assert ra.level[0] == pytest.approx(0.5) and rb.level[0] == pytest.approx(1.0)


def test_classification_matches_v1_validation_profile():
    rng = np.random.default_rng(9)
    history = rng.poisson(rng.uniform(0.01, 3, (80, 1)), (80, 500)).astype(float)
    history[:10, :450] = 0
    history[10:12] = 0
    profile, legacy = core.demand_profile(history), series_profile(history)
    np.testing.assert_array_equal(profile.demand_type, legacy["frequency_class"])
    np.testing.assert_allclose(profile.nonzero_ratio, legacy["selling_day_share"])
    np.testing.assert_array_equal(profile.age, np.where(legacy["days_since_first_sale"] < 0, 0, legacy["days_since_first_sale"]))


# ---------------------------------------------------------------- configuration


def test_config_signature_and_tamper_detection():
    config = core.make_config(ALL_MA28)
    assert core.config_signature(config) == core.config_signature(core.make_config(ALL_MA28))
    other = core.make_config({**ALL_MA28, "intermittent": {"level": "sba_0.1", "weekday": "flat"}})
    assert core.config_signature(other) != core.config_signature(config)
    tampered = core.make_config(ALL_MA28)
    tampered["level_methods"]["ma28"] = {**tampered["level_methods"]["ma28"], "window": 30}
    with pytest.raises(ValueError):
        core.validate_config(tampered)
    with pytest.raises(ValueError):
        core.make_config({**ALL_MA28, "high_frequency": {"level": "prophet", "weekday": "flat"}})
    with pytest.raises(ValueError):
        core.make_config({t: r for t, r in ALL_MA28.items() if t != "mostly_zero"})


# ---------------------------------------------------------------- exogenous research variant


def test_exogenous_factors_event_flag_price():
    t, horizon = 400, 7
    days = t + horizon
    history = np.full((2, t), 2.0)
    events = np.full((days, 1), -1)
    for day in (60, 130, 200, 270):                   # past occurrences: sales double on the event day
        events[day] = 0
        history[:, day] = 4.0
    events[t + 2] = 0                                 # event on horizon day 3
    events[150] = 1                                   # event 1 occurred once only -> no effect
    events[t + 4] = 1
    inputs = core.ExogenousInputs(events=events)
    factor = core.exogenous_factors(history, horizon, inputs)
    np.testing.assert_allclose(factor["event"][:, 2], 2.0)
    np.testing.assert_allclose(np.delete(factor["event"], 2, axis=1), 1.0)
    flags = np.zeros((2, days))
    flags[:, ::4] = 1
    flagged = np.full((2, t), 2.0)
    flagged[:, flags[0, :t] == 1] *= 1.5
    factor = core.exogenous_factors(flagged, horizon, core.ExogenousInputs(flags={"snap": flags}))
    multiplier = np.where(flags[0] == 1, 1.5, 1.0)
    np.testing.assert_allclose(factor["flag"][0], multiplier[t:t + horizon] / multiplier[t - 28:t].mean())
    price = np.full((2, days), 4.0)
    price[1, t + 3:] = np.nan
    factor = core.exogenous_factors(history, horizon, core.ExogenousInputs(price=price, missing_price_means_not_listed=True))
    assert factor["price"][1, 3:].sum() == 0 and (factor["price"][0] == 1).all() and (factor["price"][1, :3] == 1).all()
    unknown = core.exogenous_factors(history, horizon, core.ExogenousInputs(price=price))
    assert (unknown["price"] == 1).all()
    with pytest.raises(ValueError):
        core.exogenous_factors(history, horizon, core.ExogenousInputs(price=price[:, :t]))


def test_price_elasticity_is_estimated_from_history_only():
    rng = np.random.default_rng(2)
    t, horizon = 364, 7
    weekly_price = rng.choice([2.0, 3.0, 4.0], size=t // 7 + 2)
    price = np.repeat(weekly_price, 7)[None, :t + horizon].repeat(3, axis=0)
    units = (np.exp(4.0 - 1.5 * np.log(price[:, :t])) - 1) / 7  # log(1 + weekly units) = 4 - 1.5 log p exactly
    price[:, t:] = price[:, [t - 1]] * 0.5                      # planned 50% price cut
    factor = core.exogenous_factors(units, horizon, core.ExogenousInputs(price=price))
    reference = price[0, t - 28:t].mean()
    np.testing.assert_allclose(factor["price"][0], np.clip((price[0, t] / reference) ** -1.5, 0.5, 2.0), rtol=1e-6)
    core_result = core.forecast_v2(units, core.make_config(ALL_MA28))
    exo, _ = core.forecast_v2_exogenous(units, core.make_config(ALL_MA28), core.ExogenousInputs(price=price), core=core_result)
    np.testing.assert_allclose(exo, core_result.daily * factor["total"])

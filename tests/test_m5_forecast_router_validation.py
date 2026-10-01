"""M5 router checks: mostly_zero policy selection (development only) and the production-path simulation, on small panels."""
import json

import numpy as np
import pandas as pd
import pytest

from services import demand_forecast_router as router
from services import m5_forecast_router_validation as rv
from services import m5_forecast_v2_validation as v2
from services.m5_forecast_validation import M5Panel
from tests.test_m5_forecast_v2_validation import copy_panel, scramble_after

SMALL = {"horizon": 7, "origin_count": 13, "origin_step": 7}
DAYS = 420
SERIES = 64


def router_panel(seed=3, series=SERIES, days=DAYS):
    rng = np.random.default_rng(seed)
    weekly = 1 + 0.3 * np.sin(np.arange(days) * 2 * np.pi / 7)
    sales = rng.poisson(rng.uniform(0.5, 6, size=(series, 1)) * weekly, size=(series, days)).astype(float)
    sales[5, :days - 20] = 0                                   # cold start at the late origins
    sales[6:14] = (rng.random((8, days)) < 0.06) * rng.integers(1, 4, (8, days))   # mostly zero
    frame = pd.DataFrame({"store_id": np.where(np.arange(series) < series // 2, "S1", "S2"),
                          "item_id": [f"I{k:02d}" for k in range(series)], "state_id": "CA",
                          "category": np.where(np.arange(series) % 3, "A", "B"), "department": "D"})
    dates = pd.date_range("2015-01-01", periods=days).strftime("%Y-%m-%d").to_numpy()
    return M5Panel(sales, np.full((series, days), 2.0, dtype=np.float32), dates, frame, np.array(["none"] * days, dtype=object),
                   np.zeros((series, days), dtype=np.float32), series * days)


def metrics_table(rows):
    return pd.DataFrame([{"method": m, "wape": w, "bias_pct": b} for m, w, b in rows])


def test_rule_constants_are_the_frozen_v2_constants():
    assert rv.MOSTLY_ZERO_RULE["bias_floor"] == v2.SELECTION_RULE["stage_1_level"]["bias_floor"] == 0.03
    assert rv.MOSTLY_ZERO_RULE["min_relative_improvement"] == v2.PROMOTION_GATE["primary"]["min_relative_improvement"] == 0.02
    assert list(rv.MOSTLY_ZERO_CANDIDATES)[0] == rv.MOSTLY_ZERO_RULE["default"] == "v1"


def test_mostly_zero_rule_keeps_v1_unless_an_admissible_candidate_is_clearly_better():
    # V2-like: better WAPE but biased beyond max(|V1 bias|, 3%) -> inadmissible; floor: admissible but only 0.5% better.
    decision = rv.select_mostly_zero(metrics_table([("v1", 0.80, -0.06), ("v2_floor_v1", 0.796, -0.02), ("blend_v1_v2", 0.70, -0.15),
                                                    ("sparse_mean_364", 1.0, -0.5), ("v2", 0.78, -0.24)]))
    assert decision["chosen"] == "v1" and decision["bias_bound"] == pytest.approx(0.06)
    assert [o["admissible"] for o in decision["options"]] == [True, True, False, False, False]
    switch = rv.select_mostly_zero(metrics_table([("v1", 0.80, -0.06), ("v2_floor_v1", 0.77, -0.02), ("blend_v1_v2", 0.7695, 0.01),
                                                  ("sparse_mean_364", 1.0, -0.5), ("v2", 0.70, -0.24)]))
    assert switch["chosen"] == "v2_floor_v1"                      # within the 0.1% tie tolerance of the blend: declared order
    small_bias = rv.select_mostly_zero(metrics_table([("v1", 0.80, -0.01), ("v2", 0.70, -0.029)]))
    assert small_bias["bias_bound"] == 0.03 and small_bias["chosen"] == "v2"


def test_candidate_definitions():
    panel = router_panel()
    spec = v2.split_protocol(DAYS, **{"horizon": 7, "origin_count": 13, "origin_step": 7})["development"][-1]
    od = v2.build_origin(panel, spec, 7)
    v1_daily, v1_total = rv.v1_forecast(od.history, 7)
    result = router.core.forecast_v2(od.history, router.v2_config(), 7)
    candidates = rv.mostly_zero_candidates(od, v1_daily, v1_total, result, 7)
    floor_total, v2_total = candidates["v2_floor_v1"][1], candidates["v2"][1]
    np.testing.assert_array_equal(floor_total, np.maximum(v1_total, v2_total))
    np.testing.assert_allclose(candidates["blend_v1_v2"][0], 0.5 * (v1_daily + result.daily))
    assert list(candidates) == list(rv.MOSTLY_ZERO_CANDIDATES)


def test_mostly_zero_selection_never_reads_final_origins(tmp_path):
    panel = router_panel()
    first = rv.run_mostly_zero_selection(None, tmp_path / "a", panel=panel, **SMALL)
    last_dev = v2.split_protocol(DAYS, **SMALL)["development_last_day"]
    second = rv.run_mostly_zero_selection(None, tmp_path / "b", panel=scramble_after(copy_panel(panel), last_dev), **SMALL)
    assert first["decision"] == second["decision"] and first["development_total_7d"] == second["development_total_7d"]
    assert first["rule_signature"] == rv.rule_signature() and first["population_series_weeks_per_origin"]
    assert json.loads((tmp_path / "a" / rv.RESULT_FILES["mostly_zero_selection"]).read_text(encoding="utf-8"))["decision"] == first["decision"]


def test_production_path_routes_like_research_and_excludes_future_rows(tmp_path):
    report = rv.run_production_path(None, tmp_path, panel=router_panel(), wiring_series=SERIES, **SMALL)
    assert report["all_checks_pass"] and report["pipeline_wiring_check"]["pass"]
    for origin in report["per_origin"]:
        checks = origin["checks"]
        assert checks["v2_rows_7d_equal_research"] and checks["v2_rows_daily_vector_equal_research"]
        assert checks["future_rows_excluded"] == checks["future_rows_expected"] == SERIES * 7
        assert router.REASON_MOSTLY_ZERO in origin["counts_by_reason"]
    assert report["per_origin"][-1]["counts_by_reason"].get(router.REASON_COLD_START) == 1
    wiring = report["pipeline_wiring_check"]["counts_by_reason"]
    assert wiring[router.REASON_MISSING] == 20 and wiring[router.REASON_INVALID] == 10 and wiring[router.REASON_NEGATIVE] == 5
    assert (tmp_path / rv.RESULT_FILES["production_metrics"]).exists()

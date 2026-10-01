"""Canonical demand_series -> production daily_sales_history bridge (services.demand_history_bridge)."""
import numpy as np
import pandas as pd
import pytest

from services import analysis_pipeline as pipeline
from services import demand_forecast_router as router
from services.analysis_pipeline import run_analysis_pipeline
from services.demand_history_bridge import (
    HISTORY_COLUMNS, LINEAGE_COLUMNS, QUARANTINE_REASONS, attach_daily_sales_history, canonical_to_daily_history,
)
from services.legacy_adapters.loader import load_legacy_module
from tests.fixtures import sample_workbook

CUTOFF = pd.Timestamp("2026-06-30")


def canonical(series, end=CUTOFF, *, unit=None, dataset="synthetic", flags="", category="C"):
    """{(location, product): values ending at ``end``} -> canonical demand_series rows.

    np.nan = no row (an absent day); None = a row whose source quantity is NULL.
    """
    rows = []
    for (location, product), values in series.items():
        for day, q in zip(pd.date_range(end=end, periods=len(values)), values):
            if q is not None and isinstance(q, float) and np.isnan(q):
                continue
            rows.append({"source_dataset": dataset, "source_file": f"{dataset}/raw/sales.csv", "source_row_id": str(len(rows) + 1),
                         "date_grain": "daily", "product_grain": "product", "date": day.strftime("%Y-%m-%d"), "location_id": location,
                         "product_id": product, "category": category, "sales_qty": q, "unit": unit,
                         "unit_status": "UNKNOWN" if unit is None else "SOURCE_METADATA", "validation_flags": flags,
                         "location_id_namespace": f"{dataset}:store", "product_id_namespace": f"{dataset}:item"})
    return pd.DataFrame(rows)


def steady(days, level=6.0, seed=0):
    return list(np.random.default_rng(seed).poisson(level, days).astype(float))


def test_conversion_maps_canonical_rows_to_the_router_contract():
    frame = canonical({("S1", "P1"): [3.0, 0.0, 5.0], ("S1", "P2"): [1.0, 2.0, 4.0]})
    result = canonical_to_daily_history(frame)
    assert list(result.history.columns[:4]) == list(HISTORY_COLUMNS)
    assert result.history["quantity"].tolist() == [3.0, 0.0, 5.0, 1.0, 2.0, 4.0]
    assert pd.api.types.is_datetime64_any_dtype(result.history["date"])
    assert result.report["history_rows"] == 6 and result.report["series"] == 2 and result.quarantine.empty
    # the router reads it as is (store_id/product_id/date/quantity are its contract columns)
    assessment = router.assess_history(pd.DataFrame({"store_id": ["S1", "S1"], "product_id": ["P1", "P2"]}), result.history)
    assert assessment.diagnostics["history_rows"] == 6 and assessment.diagnostics["rows_without_inventory_match"] == 0


def test_dates_are_sorted_ascending_per_series():
    frame = canonical({("S2", "P1"): [1.0, 2.0, 3.0], ("S1", "P1"): [4.0, 5.0, 6.0]}).sample(frac=1.0, random_state=3)
    history = canonical_to_daily_history(frame).history
    assert history[["store_id", "product_id"]].drop_duplicates().values.tolist() == [["S1", "P1"], ["S2", "P1"]]
    for _, part in history.groupby(["store_id", "product_id"]):
        assert part["date"].is_monotonic_increasing


def test_duplicate_dates_are_passed_on_and_the_router_marks_the_series_invalid():
    frame = canonical({("S1", "P1"): steady(60)})
    frame = pd.concat([frame, frame.iloc[[50]].assign(sales_qty=99.0, source_row_id="dup")], ignore_index=True)
    result = canonical_to_daily_history(frame)
    assert result.report["duplicate_date_rows"] == 2 and result.report["duplicate_date_series"] == 1
    day = result.history["date"] == pd.Timestamp(frame["date"].iloc[50])
    assert sorted(result.history.loc[day, "quantity"]) == sorted([frame["sales_qty"].iloc[50], 99.0])   # neither summed nor picked
    reason = router.assess_history(pd.DataFrame({"store_id": ["S1"], "product_id": ["P1"]}), result.history).reason
    assert reason.tolist() == [router.REASON_INVALID]


def test_missing_days_and_null_quantities_never_become_zero():
    values = steady(60)
    values[40] = np.nan                     # absent day: no canonical row
    values[45] = None                       # NULL source quantity: a missing observation
    result = canonical_to_daily_history(canonical({("S1", "P1"): values}))
    assert result.history["quantity"].tolist() == [v for i, v in enumerate(values) if i not in (40, 45)]
    days = set(result.history["date"].dt.strftime("%Y-%m-%d"))
    all_days = pd.date_range(end=CUTOFF, periods=60).strftime("%Y-%m-%d")
    assert all_days[40] not in days and all_days[45] not in days
    assert result.report["missing_quantity_rows_not_emitted"] == 1 and result.quarantine["reason"].tolist() == ["missing_quantity"]
    reason = router.assess_history(pd.DataFrame({"store_id": ["S1"], "product_id": ["P1"]}), result.history).reason
    assert reason.tolist() == [router.REASON_INSUFFICIENT]      # gaps in the last 30 days: V1, never zero-filled


def test_zero_sales_are_preserved_as_observed_zeros():
    values = [0.0 if d % 5 == 0 else 3.0 for d in range(60)]
    result = canonical_to_daily_history(canonical({("S1", "P1"): values}))
    assert result.report["zero_rows"] == 12 and (result.history["quantity"] == 0).sum() == 12
    assert result.series["zero_rows"].tolist() == [12]


def test_negative_quantities_are_preserved_and_route_the_series_to_v1():
    values = steady(80)
    values[60], values[70] = -2.0, -0.5
    frame = canonical({("S1", "P1"): values, ("S1", "P2"): steady(80, seed=1)}, flags="")
    frame.loc[frame["sales_qty"] < 0, "validation_flags"] = "negative:sales_qty|documented_return"
    result = canonical_to_daily_history(frame)
    negative = result.history[result.history["quantity"] < 0]
    assert negative["quantity"].tolist() == [-2.0, -0.5]                      # never abs(), clipped or zeroed
    assert result.report["negative_rows"] == 2 and result.report["negative_series"] == 1
    assert result.report["negative_quantity_sum"] == -2.5
    reason = router.assess_history(pd.DataFrame({"store_id": ["S1", "S1"], "product_id": ["P1", "P2"]}), result.history).reason
    assert reason.tolist() == [router.REASON_NEGATIVE, router.REASON_V2]


def test_rows_after_as_of_are_future_data_and_are_blocked():
    frame = canonical({("S1", "P1"): steady(70)})
    as_of = CUTOFF - pd.Timedelta(days=10)
    result = canonical_to_daily_history(frame, as_of=as_of)
    assert result.history["date"].max() == as_of
    assert result.report["future_rows_excluded"] == 10 == result.report["quarantined_rows"]["after_as_of"]
    assert (result.quarantine["reason"] == "after_as_of").sum() == 10
    with pytest.raises(ValueError, match="calendar date"):
        canonical_to_daily_history(frame, as_of="2026-06-30 12:00")


def test_unit_is_preserved_and_a_series_mixing_units_is_quarantined():
    frame = pd.concat([canonical({("S1", "P1"): steady(40)}, unit="kg"), canonical({("S1", "P2"): steady(40)}, unit="item")], ignore_index=True)
    result = canonical_to_daily_history(frame)
    assert dict(zip(result.series["product_id"], result.series["unit"])) == {"P1": "kg", "P2": "item"}
    assert result.history["quantity"].sum() == frame["sales_qty"].sum()      # never converted
    mixed = frame.copy()
    mixed.loc[mixed.index[-1], "unit"] = "kg"                               # P2 now carries item and kg
    out = canonical_to_daily_history(mixed)
    assert set(out.history["product_id"]) == {"P1"} and out.report["mixed_unit_series_quarantined"] == 1
    assert (out.quarantine["reason"] == "mixed_unit_series").sum() == 40
    # unit consistency is judged on the rows up to as_of only
    early = canonical_to_daily_history(mixed, as_of=CUTOFF - pd.Timedelta(days=1))
    assert set(early.history["product_id"]) == {"P1", "P2"}


def test_lineage_is_preserved_on_every_row_and_per_series():
    frame = canonical({("S1", "P1"): steady(30), ("S2", "P1"): steady(30, seed=2)}).sample(frac=1.0, random_state=1)
    result = canonical_to_daily_history(frame)
    history = result.history
    assert set(LINEAGE_COLUMNS) <= set(history.columns) and "input_position" in history
    joined = frame.reset_index(drop=True).iloc[history["input_position"]]
    assert history["source_row_id"].tolist() == joined["source_row_id"].tolist()
    assert history["quantity"].tolist() == joined["sales_qty"].tolist()
    meta = result.series.set_index("store_id")
    assert meta.loc["S1", "source_files"] == "synthetic/raw/sales.csv" and meta.loc["S1", "location_namespace"] == "synthetic:store"
    assert meta.loc["S1", "rows"] == 30 and str(meta.loc["S1", "first_date"])[:10] == "2026-06-01"


def test_invalid_keys_dates_and_grains_are_quarantined_with_a_reason():
    frame = canonical({("S1", "P1"): steady(5)})
    bad = pd.concat([frame, frame.iloc[[0]].assign(location_id="  "), frame.iloc[[0]].assign(product_id=None),
                     frame.iloc[[0]].assign(date="2026-13-40"), frame.iloc[[0]].assign(date_grain="monthly"),
                     frame.iloc[[0]].assign(product_grain="category")], ignore_index=True)
    result = canonical_to_daily_history(bad)
    assert len(result.history) == 5
    assert sorted(result.quarantine["reason"]) == sorted(["missing_location_id", "missing_product_id", "invalid_date", "non_daily_grain",
                                                          "non_product_grain"])
    assert set(result.quarantine["reason"]) <= set(QUARANTINE_REASONS)


def test_raw_ids_of_several_namespaces_are_refused():
    frame = pd.concat([canonical({("1", "1"): steady(5)}, dataset="a"), canonical({("1", "1"): steady(5)}, dataset="b")], ignore_index=True)
    with pytest.raises(ValueError, match="namespaces"):
        canonical_to_daily_history(frame)


def test_unparseable_quantities_become_nan_and_the_router_marks_invalid():
    frame = canonical({("S1", "P1"): steady(40)})
    frame.loc[5, "validation_flags"] = "invalid_numeric:sales_qty"
    frame.loc[5, "sales_qty"] = None
    result = canonical_to_daily_history(frame)
    assert result.report["invalid_quantity_rows_emitted_as_nan"] == 1 and len(result.history) == 40
    reason = router.assess_history(pd.DataFrame({"store_id": ["S1"], "product_id": ["P1"]}), result.history).reason
    assert reason.tolist() == [router.REASON_INVALID]


def test_attach_never_overwrites_an_existing_history_silently():
    frame = canonical({("S1", "P1"): steady(5)})
    data, result = attach_daily_sales_history({"inventory": pd.DataFrame()}, frame)
    assert data[router.DAILY_SALES_HISTORY_KEY] is result.history
    with pytest.raises(ValueError, match="already holds"):
        attach_daily_sales_history(data, frame)
    replaced, _ = attach_daily_sales_history(data, frame, replace=True)
    assert len(replaced[router.DAILY_SALES_HISTORY_KEY]) == 5


# ---------------------------------------------------------------- production E2E: canonical -> bridge -> run_analysis_pipeline


def workbook_canonical():
    days = 150
    rng = np.random.default_rng(11)
    zeros = rng.poisson(5, days).astype(float)
    zeros[-12] = 0.0
    negative = rng.poisson(5, days).astype(float)
    negative[-40] = -1.0
    cold = [np.nan] * (days - 12) + list(rng.poisson(5, 12).astype(float))
    gap = list(rng.poisson(5, days).astype(float))
    gap[-6] = np.nan
    sparse = [2.0 if d % 15 == 0 else 0.0 for d in range(days)]
    series = {("S001", "P001"): list(rng.poisson(8, days).astype(float)), ("S001", "P002"): list(zeros),
              ("S002", "P001"): list(negative), ("S002", "P002"): cold, ("S003", "P001"): gap, ("S003", "P002"): sparse,
              ("DC01", "P001"): list(rng.poisson(6, days).astype(float))}
    future = canonical({key: [3.0] * 5 for key in series}, end=CUTOFF + pd.Timedelta(days=5))     # rows after the cutoff
    frame = pd.concat([canonical(series), future], ignore_index=True)
    duplicate = frame[(frame["location_id"] == "DC01") & (frame["date"] == "2026-06-20")]
    return pd.concat([frame, duplicate], ignore_index=True)                                      # DC01/P002 has no row at all


EXPECTED = {("S001", "P001"): router.REASON_V2, ("S001", "P002"): router.REASON_V2, ("S002", "P001"): router.REASON_NEGATIVE,
            ("S002", "P002"): router.REASON_COLD_START, ("S003", "P001"): router.REASON_INSUFFICIENT,
            ("S003", "P002"): router.REASON_MOSTLY_ZERO, ("DC01", "P001"): router.REASON_INVALID, ("DC01", "P002"): router.REASON_MISSING}


def run_pipeline(monkeypatch, data):
    """run_analysis_pipeline, keeping the analysed inventory its inventory step returns."""
    captured = {}
    original, original_vhs = pipeline._run_inventory_analysis, pipeline.apply_auto_vhs

    def capture(*args, **kwargs):
        captured["analyzed"], captured["summaries"] = original(*args, **kwargs)
        return captured["analyzed"], captured["summaries"]

    def capture_vhs(candidates, *args, **kwargs):
        captured.setdefault("vhs_candidates", candidates.copy())
        return original_vhs(candidates, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_run_inventory_analysis", capture)
    monkeypatch.setattr(pipeline, "apply_auto_vhs", capture_vhs)
    result = run_analysis_pipeline(data)
    rows = {(r.store_id, r.product_id): r for r in captured["analyzed"].itertuples()}
    result.vhs_candidates = captured["vhs_candidates"]
    return result, rows


def test_production_bridge_end_to_end_through_the_analysis_pipeline(monkeypatch):
    data = sample_workbook()
    with_history, bridged = attach_daily_sales_history(data, workbook_canonical(), as_of=CUTOFF)
    assert bridged.report["future_rows_excluded"] == 35 and bridged.report["duplicate_date_series"] == 1
    assert bridged.report["zero_rows"] >= 1 and bridged.report["negative_rows"] == 1
    result, rows = run_pipeline(monkeypatch, with_history)
    plain, plain_rows = run_pipeline(monkeypatch, data)
    assert result.status == plain.status == "success"
    info = result.demand_analysis["demand_forecast"]["forecast_router"]
    assert info["future_rows_excluded"] == 0 and info["history_supplied"]
    assert {key: rows[key].demand_forecast_reason for key in EXPECTED} == EXPECTED
    for key, reason in EXPECTED.items():
        if reason == router.REASON_V2:
            assert rows[key].demand_forecast_version == "v2" and rows[key].demand_forecast_method.startswith("V2:")
        else:      # every fallback row keeps the V1 output of the run without a history, field for field
            for column in ("demand_forecast_7d", "demand_forecast_daily", "demand_risk_score", "demand_forecast_score", "demand_trend"):
                assert getattr(rows[key], column) == getattr(plain_rows[key], column)
    assert {r.demand_forecast_reason for r in plain_rows.values()} == {router.REASON_MISSING}
    assert "services.demand_forecast_router.route_demand_forecast" in result.connected_algorithms
    assert "services.demand_forecast_router.route_demand_forecast" not in plain.connected_algorithms
    # the routed forecast of the source store is what its transfer candidates carry into the auto VHS (demand_fit_score)
    candidates = result.vhs_candidates
    assert len(candidates) and {"demand_forecast_7d", "demand_forecast_version"} <= set(candidates.columns)
    for c in candidates.itertuples():
        source = rows[(str(c.source_id), str(c.product_id))]
        assert c.demand_forecast_7d == source.demand_forecast_7d and c.demand_forecast_version == source.demand_forecast_version
    assert "v2" in set(candidates["demand_forecast_version"])


def test_without_history_the_v1_contract_is_unchanged(monkeypatch):
    from services.legacy_adapters.data_adapter import prepare_legacy_data
    data = sample_workbook()
    v1 = load_legacy_module("demand_forecast_analyzer").analyze_demand_forecast(prepare_legacy_data(data)["inventory"])
    _, rows = run_pipeline(monkeypatch, data)
    for r in v1.itertuples():
        routed = rows[(r.store_id, r.product_id)]
        assert routed.demand_forecast_version == "v1" and routed.demand_forecast_reason == router.REASON_MISSING
        for column in ("demand_forecast_7d", "demand_forecast_daily", "demand_risk_score", "demand_stockout_days", "demand_trend"):
            assert getattr(routed, column) == getattr(r, column)

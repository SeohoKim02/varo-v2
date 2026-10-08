"""Operational Top-N shared-feasibility selection: controlled scenarios A-R, cap invariants, cross-checks, isolation."""
from __future__ import annotations

import copy
import random
import warnings
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from services import analysis_pipeline
from services import shared_feasibility_selection as sf
from services.analysis_pipeline import build_v2_state
from services.data_loader import load_excel_data
from services.pareto_service import _default_feasible as pareto_default_feasible
from services.suhyup_algorithm_revalidation import _selection_violation, ordered_feasible_selection
from services.vhs_score_engine import _rank_varo_operational

REPO = Path(__file__).resolve().parents[1]
NORMAL_SAMPLE = REPO / "samples" / "Varo_V2_sample_normal_6stores_1dc.xlsx"
NETWORK_SAMPLE = REPO / "data" / "Varo_V2_네트워크_샘플.xlsx"


def rec(cid, qty, cost=10.0, *, product="P1", source="S1", target="T1", rank=None, vhs=None, **extra):
    row = {"route_id": cid, "product_id": product, "source_id": source, "target_id": target, "route_type": "DIRECT",
           "recommended_qty": qty, "move_cost": cost, "varo_final_rank": rank, "vhs_rank": vhs}
    row.update(extra)
    return row


def caps_of(*caps):
    table = {}
    for cap in caps:
        sf.add_cap(table, cap)
    return table


def stock(source, product, value, provenance="USER_INPUT", unit=None):
    return sf.Cap(sf.SOURCE_STOCK, (source, product), value, provenance, "test stock", unit)


def surplus(source, product, value, provenance="USER_INPUT"):
    return sf.Cap(sf.SOURCE_SURPLUS, (source, product), value, provenance, "test surplus")


def need(target, product, value, provenance="USER_INPUT", unit=None):
    return sf.Cap(sf.TARGET_NEED, (target, product), value, provenance, "test need", unit)


def select(records, caps, **kwargs):
    kwargs.setdefault("mode", sf.BENCHMARK_PROXY)
    kwargs.setdefault("max_routes", 5)
    return sf.select_shared_feasible(records, caps, **kwargs)


def by_id(plan):
    return {row["candidate_id"]: row for row in plan["rows"]}


def signature(plan):
    return [(row["candidate_id"], row["selection_status"], row["allocated_qty"], row["allocated_move_cost"])
            for row in plan["rows"]]


# ------------------------------------------------------------------------------------------- A-R controlled scenarios


def test_a_same_source_two_targets_shares_the_source_cap():
    records = [rec("R1", 30, 30, target="T1", rank=1, cost_basis="PER_UNIT"),
               rec("R2", 30, 30, target="T2", rank=2, cost_basis="PER_UNIT")]
    caps = caps_of(stock("S1", "P1", 50), need("T1", "P1", 100), need("T2", "P1", 100))
    rows = by_id(select(records, caps))
    assert (rows["R1"]["selection_status"], rows["R1"]["allocated_qty"], rows["R1"]["source_remaining_qty"]) == (sf.SELECTED, 30, 20)
    assert (rows["R2"]["selection_status"], rows["R2"]["allocated_qty"], rows["R2"]["source_remaining_qty"]) == (sf.PARTIALLY_SELECTED, 20, 0)
    assert rows["R2"]["binding_caps"] == sf.SOURCE_STOCK and rows["R1"]["binding_caps"] is None
    assert rows["R2"]["allocated_move_cost"] == pytest.approx(20.0)  # PER_UNIT: 30 KRW / 30 units x 20
    blocked = by_id(select(records, caps, partial_policy=sf.PARTIAL_NONE))["R2"]
    assert (blocked["selection_status"], blocked["allocated_qty"], blocked["executable_qty_at_end"]) == (sf.REJECTED_SOURCE_CAP, 0.0, 20.0)
    assert blocked["blocking_selected_ids"] == "R1" and blocked["source_remaining_qty"] == 20
    assert "partial allocation disabled" in blocked["rejection_reason"]


def test_b_same_target_two_sources_shares_the_target_need():
    records = [rec("R1", 30, 15, source="S1", rank=1, cost_basis="FIXED_PER_TRIP"),
               rec("R2", 30, 15, source="S2", rank=2, cost_basis="FIXED_PER_TRIP")]
    caps = caps_of(stock("S1", "P1", 100), stock("S2", "P1", 100), need("T1", "P1", 50))
    rows = by_id(select(records, caps))
    assert (rows["R1"]["allocated_qty"], rows["R1"]["target_remaining_need"]) == (30, 20)
    assert (rows["R2"]["selection_status"], rows["R2"]["allocated_qty"], rows["R2"]["target_remaining_need"]) == (sf.PARTIALLY_SELECTED, 20, 0)
    assert rows["R2"]["allocated_move_cost"] == 15  # the trip costs the same for 20 units
    assert rows["R2"]["cost_status"] == "FIXED_PER_TRIP_UNCHANGED"


def test_c_source_and_target_caps_together_and_dynamic_executable_order():
    records = [rec("R1", 30, 1, source="S1", target="T1", rank=1, cost_basis="PER_UNIT"),
               rec("R2", 30, 1, source="S1", target="T2", rank=2, cost_basis="PER_UNIT"),
               rec("R3", 30, 1, source="S2", target="T1", rank=3, cost_basis="PER_UNIT")]
    caps = caps_of(stock("S1", "P1", 40), stock("S2", "P1", 100), need("T1", "P1", 45), need("T2", "P1", 100))
    plan = select(records, caps)
    rows = by_id(plan)
    # after R1: S1 has 10 left, T1 has 15 left -> R3 (15 executable) outranks R2 (10 executable)
    assert plan["selected_ids"] == ["R1", "R3", "R2"]
    assert [rows[i]["allocated_qty"] for i in ("R1", "R3", "R2")] == [30, 15, 10]
    assert rows["R2"]["source_remaining_qty"] == 0 and rows["R3"]["target_remaining_need"] == 0
    assert plan["validation"]["violation_count"] == 0 and plan["total_allocated_qty"] == 55


def test_d_partial_allocation_keeps_integer_units():
    records = [rec("R1", 20, 20, rank=1, cost_basis="PER_UNIT")]
    rows = by_id(select(records, caps_of(stock("S1", "P1", 100), need("T1", "P1", 12.5))))
    assert (rows["R1"]["selection_status"], rows["R1"]["allocated_qty"]) == (sf.PARTIALLY_SELECTED, 12)
    assert rows["R1"]["target_remaining_need"] == 0.5 and rows["R1"]["allocated_move_cost"] == pytest.approx(12.0)


def test_d_partial_allocation_respects_declared_lot_size():
    records = [rec("R1", 40, 40, rank=1, cost_basis="PER_UNIT", lot_size_qty=10)]
    rows = by_id(select(records, caps_of(stock("S1", "P1", 100), need("T1", "P1", 25))))
    assert (rows["R1"]["selection_status"], rows["R1"]["allocated_qty"], rows["R1"]["target_remaining_need"]) == (sf.PARTIALLY_SELECTED, 20, 5)


@pytest.mark.parametrize("extra, cap, expected_reason", [
    ({"allow_partial": False}, 25, "forbids partial"),
    ({"lot_size_qty": 10}, 5, "no whole lot"),
    ({"min_transfer_qty": 20}, 15, "below the declared minimum"),
])
def test_e_partial_allocation_not_possible(extra, cap, expected_reason):
    records = [rec("R1", 40, 40, rank=1, cost_basis="PER_UNIT", **extra)]
    plan = select(records, caps_of(stock("S1", "P1", 100), need("T1", "P1", cap)))
    row = by_id(plan)["R1"]
    assert (row["selection_status"], row["allocated_qty"], row["target_remaining_need"]) == (sf.REJECTED_TARGET_CAP, 0.0, cap)
    assert expected_reason in row["partial_blocked_reason"] and row["rejection_category"] == "PARTIAL_ONLY"
    assert plan["selected_count"] == 0 and plan["plan_status"] == "NO_FEASIBLE_SELECTION"


def test_f_duplicate_route_ids_and_duplicate_lanes_keep_lineage():
    records = [rec("R1", 10, 1, rank=1), rec("R1", 10, 1, rank=2),
               rec("R2", 10, 1, rank=3), rec("R3", 10, 1, rank=4, route_type="VIA_DC", dc_id="DC1")]
    plan = select(records, caps_of(stock("S1", "P1", 100), need("T1", "P1", 100)))
    statuses = [(row["candidate_id"], row["selection_status"], row["duplicate_of"]) for row in plan["rows"]]
    assert len(plan["rows"]) == 4  # nothing is silently dropped
    assert ("R1", sf.REJECTED_DUPLICATE, "R1") in statuses and ("R2", sf.REJECTED_DUPLICATE, "R1") in statuses
    assert plan["selected_ids"] == ["R1", "R3"]  # VIA_DC is a different lane, same caps
    assert plan["total_allocated_qty"] == 20 and plan["validation"]["duplicate_lane_count"] == 0


def test_g_same_product_many_stores_only_shares_matching_groups():
    records = [rec("R1", 30, 1, source="S1", target="T1", rank=1), rec("R2", 30, 1, source="S2", target="T2", rank=2),
               rec("R3", 30, 1, source="S3", target="T1", rank=3)]
    caps = caps_of(*(stock(s, "P1", 100) for s in ("S1", "S2", "S3")), need("T1", "P1", 40), need("T2", "P1", 30))
    rows = by_id(select(records, caps, partial_policy=sf.PARTIAL_NONE))
    assert [rows[i]["allocated_qty"] for i in ("R1", "R2", "R3")] == [30, 30, 0]
    assert rows["R3"]["selection_status"] == sf.REJECTED_TARGET_CAP and rows["R3"]["target_remaining_need"] == 10
    assert rows["R2"]["target_remaining_need"] == 0


def test_h_different_products_do_not_share_caps():
    records = [rec("R1", 30, 1, product="P1", rank=1), rec("R2", 30, 1, product="P2", rank=2)]
    caps = caps_of(stock("S1", "P1", 30), stock("S1", "P2", 30), need("T1", "P1", 30), need("T1", "P2", 30))
    rows = by_id(select(records, caps))
    assert [(rows[i]["allocated_qty"], rows[i]["source_remaining_qty"]) for i in ("R1", "R2")] == [(30, 0), (30, 0)]


def test_i_zero_cap_is_a_cap_not_missing():
    plan = select([rec("R1", 30, 1, rank=1, cost_basis="PER_UNIT")], caps_of(stock("S1", "P1", 100), need("T1", "P1", 0)))
    row = by_id(plan)["R1"]
    assert (row["selection_status"], row["allocated_qty"], row["individual_cap_room"]) == (sf.REJECTED_TARGET_CAP, 0.0, 0.0)
    assert row["rejection_category"] == "INDIVIDUALLY_OVER_CAP" and row["target_remaining_need"] == 0
    assert plan["plan_status"] == "NO_FEASIBLE_SELECTION" and "cap room 0" in plan["no_selection_cause"]


def test_j_missing_cap_strict_vs_benchmark():
    records = [rec("R1", 30, 1, rank=1)]
    caps = caps_of(stock("S1", "P1", 100), sf.Cap(sf.TARGET_NEED, ("T1", "P1"), None, "MISSING", "absent"))
    strict = select(records, caps, mode=sf.STRICT_ACTUAL)
    assert strict["plan_status"] == sf.INSUFFICIENT_CAP_DATA and strict["selected_count"] == 0
    assert "no validated cap for TARGET_NEED" in by_id(strict)["R1"]["rejection_reason"]
    bench = select(records, caps)
    assert bench["selected_ids"] == ["R1"] and by_id(bench)["R1"]["allocated_qty"] == 30
    assert by_id(bench)["R1"]["unchecked_constraints"] == "TARGET_NEED" and "UNCHECKED:TARGET_NEED" in bench["feasibility_claim"]


def test_k_proxy_cap_strict_vs_benchmark():
    records = [rec("R1", 30, 30, rank=1, cost_basis="PER_UNIT")]
    caps = caps_of(stock("S1", "P1", 100, "DIRECT_REAL"), need("T1", "P1", 20, "PROXY"))
    strict = select(records, caps, mode=sf.STRICT_ACTUAL)
    assert strict["plan_status"] == sf.INSUFFICIENT_CAP_DATA
    assert "TARGET_NEED=PROXY" in by_id(strict)["R1"]["rejection_reason"]
    bench = select(records, caps)
    row = by_id(bench)["R1"]
    assert (row["allocated_qty"], row["target_remaining_need"]) == (20, 0) and bench["proxy_caps_used"] is True
    assert bench["feasibility_claim"].startswith("PROXY_CAPS_SATISFIED_NOT_AN_OPERATIONAL_GUARANTEE")
    assert "TARGET_NEED=PROXY" in row["cap_provenance"]


def test_l_route_capacity_is_enforced():
    records = [rec("R1", 30, 30, rank=1, cost_basis="PER_UNIT", route_capacity_qty=25),
               rec("R2", 30, 30, rank=2, target="T2", allow_partial=False, vehicle_capacity_qty=20)]
    spec = {sf.SOURCE_STOCK: (("stock",), "USER_INPUT", "t"), sf.TARGET_NEED: (("need",), "USER_INPUT", "t"),
            sf.ROUTE_CAPACITY: (sf.ROUTE_CAPACITY_FIELDS, "USER_INPUT", "t")}
    for row in records:
        row.update(stock=100, need=100)
    rows = by_id(select(records, sf.caps_from_columns(records, spec)))
    assert (rows["R1"]["selection_status"], rows["R1"]["allocated_qty"], rows["R1"]["binding_caps"]) == (sf.PARTIALLY_SELECTED, 25, sf.ROUTE_CAPACITY)
    assert rows["R1"]["route_capacity_cap"] == 25
    assert (rows["R2"]["selection_status"], rows["R2"]["allocated_qty"]) == (sf.REJECTED_ROUTE_CAP, 0.0)


@pytest.mark.parametrize("candidate_unit, cap_unit, status, unit_status", [
    ("BOX", "EA", sf.UNIT_MISMATCH, sf.UNIT_MISMATCH),
    ("EA", None, sf.UNKNOWN_UNIT, sf.UNKNOWN_UNIT),
    (None, "KG", sf.UNKNOWN_UNIT, sf.UNKNOWN_UNIT),
    ("개", "EA", sf.SELECTED, "MATCHED"),
    (None, None, sf.SELECTED, "UNDECLARED_SAME_SOURCE"),
])
def test_m_units_are_never_mixed(candidate_unit, cap_unit, status, unit_status):
    record = rec("R1", 10, 1, rank=1, **({"quantity_unit": candidate_unit} if candidate_unit else {}))
    plan = select([record], caps_of(stock("S1", "P1", 100, unit=cap_unit), need("T1", "P1", 100, unit=cap_unit)))
    row = by_id(plan)["R1"]
    assert (row["selection_status"], row["unit_status"]) == (status, unit_status)
    assert row["allocated_qty"] == (10 if status == sf.SELECTED else 0.0)


def test_n_cost_basis_controls_partial_cost():
    caps = caps_of(stock("S1", "P1", 100), need("T1", "P1", 12))
    assumed = by_id(select([rec("R1", 20, 500, rank=1)], caps))["R1"]
    assert (assumed["selection_status"], assumed["allocated_qty"], assumed["executable_qty_at_end"]) == (sf.COST_NOT_COMPARABLE, 0.0, 12.0)
    assert "assumed" in assumed["rejection_reason"] and assumed["cost_basis_declared"] is False
    unknown = by_id(select([rec("R1", 20, 500, rank=1, cost_basis="PER_KM")], caps))["R1"]
    assert unknown["selection_status"] == sf.COST_NOT_COMPARABLE
    recomputed = by_id(select([rec("R1", 20, 500, rank=1)], caps, cost_recompute=lambda record, qty: 777.0))["R1"]
    assert (recomputed["selection_status"], recomputed["allocated_qty"], recomputed["allocated_move_cost"]) == (sf.PARTIALLY_SELECTED, 12, 777.0)
    view = select([rec("R1", 20, 500, rank=1)], caps, partial_policy=sf.PARTIAL_COST_UNKNOWN)
    row = by_id(view)["R1"]
    assert (row["selection_status"], row["allocated_qty"], row["allocated_move_cost"], row["cost_status"]) == (
        sf.PARTIALLY_SELECTED, 12, None, sf.COST_NOT_COMPARABLE)  # never 500 x 12/20
    assert view["total_move_cost"] is None and "QUANTITY_FEASIBILITY_ONLY" in view["feasibility_claim"]


def test_n_mixed_currencies_make_the_plan_cost_not_comparable():
    records = [rec("R1", 10, 1, rank=1, currency="KRW"), rec("R2", 10, 1, rank=2, target="T2", currency="USD")]
    plan = select(records, caps_of(stock("S1", "P1", 100), need("T1", "P1", 100), need("T2", "P1", 100)))
    assert plan["selected_count"] == 2 and plan["total_move_cost"] is None and plan["cost_currencies"] == ["KRW", "USD"]


def test_o_top_n_limit():
    records = [rec(f"R{i}", 10, 1, product=f"P{i}", rank=i) for i in range(1, 8)]
    caps = caps_of(*(stock("S1", f"P{i}", 100) for i in range(1, 8)), *(need("T1", f"P{i}", 100) for i in range(1, 8)))
    plan = select(records, caps)
    assert plan["selected_ids"] == ["R1", "R2", "R3", "R4", "R5"]
    for cid in ("R6", "R7"):
        row = by_id(plan)[cid]
        assert (row["selection_status"], row["executable_qty_at_end"], row["allocated_qty"]) == (sf.REJECTED_SELECTION_LIMIT, 10, 0.0)
    assert select(records, caps, max_routes=None)["selected_count"] == 7


def _random_instance(seed, count=24):
    rng = random.Random(seed)
    records, caps = [], {}
    for index in range(count):
        product, source, target = f"P{rng.randint(1, 3)}", f"S{rng.randint(1, 4)}", f"T{rng.randint(1, 4)}"
        records.append(rec(f"R{index:02d}", rng.choice([5, 10, 20, 30, 50]), rng.randint(1, 9) * 100.0, product=product,
                           source=source, target=target, rank=index + 1, vhs=rng.randint(1, count),
                           cost_basis=rng.choice(["PER_UNIT", "FIXED_PER_TRIP", "QUANTITY_SPECIFIC"])))
        sf.add_cap(caps, stock(source, product, rng.choice([0, 20, 40, 80])))
        sf.add_cap(caps, need(target, product, rng.choice([0, 15, 35, 70])))
    # varo_final_rank exactly as production assigns it (services.vhs_score_engine._rank_varo_operational)
    for row, rank in zip(records, _rank_varo_operational(pd.DataFrame(records)).tolist()):
        row["varo_final_rank"] = rank
    return records, caps


def test_p_input_row_order_does_not_change_the_plan():
    records, caps = _random_instance(7)
    shuffled = list(records)
    random.Random(1).shuffle(shuffled)
    assert signature(select(records, caps)) == signature(select(shuffled, caps))
    unranked = [{**row, "varo_final_rank": None} for row in records]  # rank recomputed with the Varo Final key
    unranked_shuffled = [{**row, "varo_final_rank": None} for row in shuffled]
    assert signature(select(unranked, caps)) == signature(select(unranked_shuffled, caps))


def test_q_exact_quantity_and_cost_tie_is_broken_by_route_id():
    records = [rec("R-B", 30, 10, source="S2", vhs=1), rec("R-A", 30, 10, source="S1", vhs=1)]
    caps = caps_of(stock("S1", "P1", 100), stock("S2", "P1", 100), need("T1", "P1", 30))
    for ordering in (records, list(reversed(records))):
        plan = select(ordering, caps, partial_policy=sf.PARTIAL_NONE)
        assert plan["selected_ids"] == ["R-A"] and by_id(plan)["R-B"]["blocking_selected_ids"] == "R-A"


def test_r_on_hand_stock_bounds_an_explicit_surplus():
    records = [rec("R1", 60, 60, rank=1, cost_basis="PER_UNIT")]
    caps = caps_of(surplus("S1", "P1", 100), stock("S1", "P1", 40), need("T1", "P1", 100))
    row = by_id(select(records, caps))["R1"]
    assert (row["allocated_qty"], row["binding_caps"], row["source_remaining_qty"]) == (40, sf.SOURCE_STOCK, 0)
    data = {"inventory": pd.DataFrame([
        {"store_id": "S1", "product_id": "P1", "stock_qty": 40, "available_transfer_stock": 100, "sales_7d": 0},
        {"store_id": "T1", "product_id": "P1", "stock_qty": 0, "sales_7d": 100}])}
    pipeline = sf.pipeline_caps(records, data)
    assert pipeline[(sf.SOURCE_SURPLUS, ("S1", "P1"))].value == 100 and pipeline[(sf.SOURCE_STOCK, ("S1", "P1"))].value == 40
    assert by_id(select(records, pipeline))["R1"]["allocated_qty"] == 40


# ------------------------------------------------------------------------------------------- invariants and cross-checks


@pytest.mark.parametrize("seed", range(40))
def test_shared_caps_are_never_exceeded(seed):
    records, caps = _random_instance(seed)
    for policy in sf.PARTIAL_POLICIES:
        for limit in (3, 5, None):
            plan = select(records, caps, partial_policy=policy, max_routes=limit, cost_recompute=lambda r, q: 1.0)
            chosen = [row for row in plan["rows"] if row["selection_status"] in (sf.SELECTED, sf.PARTIALLY_SELECTED)]
            assert plan["validation"]["violation_count"] == 0
            assert limit is None or len(chosen) <= limit
            source, target = {}, {}
            for row in chosen:
                assert 0 < row["allocated_qty"] <= row["original_recommended_qty"]
                source[(row["source_id"], row["product_id"])] = source.get((row["source_id"], row["product_id"]), 0) + row["allocated_qty"]
                target[(row["target_id"], row["product_id"])] = target.get((row["target_id"], row["product_id"]), 0) + row["allocated_qty"]
            assert all(used <= caps[(sf.SOURCE_STOCK, key)].value + 1e-9 for key, used in source.items())
            assert all(used <= caps[(sf.TARGET_NEED, key)].value + 1e-9 for key, used in target.items())
            lanes = [(row["product_id"], row["source_id"], row["target_id"], row["route_type"]) for row in chosen]
            assert len(lanes) == len(set(lanes))
            assert len(plan["rows"]) == len(records)


@pytest.mark.parametrize("seed", range(40))
def test_all_or_nothing_equals_the_offline_ordered_feasible_selection(seed):
    records, caps = _random_instance(seed)
    for row in records:
        row["source_surplus"] = caps[(sf.SOURCE_STOCK, (row["source_id"], row["product_id"]))].value
        row["target_need_7d"] = caps[(sf.TARGET_NEED, (row["target_id"], row["product_id"]))].value
    spec = {sf.SOURCE_SURPLUS: (("source_surplus",), "PROXY", "t"), sf.TARGET_NEED: (("target_need_7d",), "PROXY", "t")}
    plan = select(records, sf.caps_from_columns(records, spec), partial_policy=sf.PARTIAL_NONE)
    offline = ordered_feasible_selection(pd.DataFrame(records), ("varo_final_rank", "route_id"), (True, True))
    assert plan["selected_ids"] == offline.get("route_id", pd.Series(dtype=str)).tolist()
    chosen = [row for row in records if row["route_id"] in plan["selected_ids"]]
    assert _selection_violation(chosen) is None and pareto_default_feasible(chosen) == (True, None)


def test_no_binding_cap_reproduces_the_varo_final_order():
    records = [rec(f"R{i}", 10 * (7 - i), 100 - i, product=f"P{i}", rank=i) for i in range(1, 7)]
    caps = caps_of(*(stock("S1", f"P{i}", 1000) for i in range(1, 7)), *(need("T1", f"P{i}", 1000) for i in range(1, 7)))
    assert select(records, caps)["selected_ids"] == ["R1", "R2", "R3", "R4", "R5"]


def test_varo_final_key_is_applied_and_a_supplied_rank_only_breaks_ties():
    # A rank that contradicts the Varo Final key (qty desc first) does not override it; production ranks never do.
    records = [rec("SMALL", 10, 1, product="P1", rank=1), rec("LARGE", 40, 1, product="P2", rank=2)]
    caps = caps_of(stock("S1", "P1", 100), stock("S1", "P2", 100), need("T1", "P1", 100), need("T1", "P2", 100))
    assert select(records, caps)["selected_ids"] == ["LARGE", "SMALL"]
    tied = [rec("B", 10, 1, product="P1", rank=1), rec("A", 10, 1, product="P2", rank=2)]
    tied_caps = caps_of(stock("S1", "P1", 100), stock("S1", "P2", 100), need("T1", "P1", 100), need("T1", "P2", 100))
    assert select(tied, tied_caps, max_routes=1)["selected_ids"] == ["A"]  # route_id precedes the supplied rank


def test_validate_plan_reports_legacy_overflow_like_suhyup_2026_07_05():
    # two sources each send 50 of 614504 to 157 whose benchmark need is 70 (shape of the audited problem days)
    records = [rec("V2C003", 50, 33905, product="614504", source="156", target="157", rank=1),
               rec("V2C008", 50, 51245, product="614504", source="218020", target="157", rank=2),
               rec("V2C012", 50, 54821, product="619601", source="130030", target="218020", rank=3)]
    caps = caps_of(surplus("156", "614504", 93, "PROXY"), surplus("218020", "614504", 72, "PROXY"),
                   surplus("130030", "619601", 16919, "PROXY"), need("157", "614504", 70, "PROXY"),
                   need("218020", "619601", 12103, "PROXY"))
    legacy = sf.validate_plan(records, caps, mode=sf.BENCHMARK_PROXY, max_routes=5)
    assert legacy["violation_count"] == 1 and legacy["target_excess_qty"] == 30
    plan = select(records, caps, partial_policy=sf.PARTIAL_NONE)
    assert plan["selected_ids"] == ["V2C003", "V2C012"] and plan["validation"]["violation_count"] == 0
    assert by_id(plan)["V2C008"]["target_remaining_need"] == 20
    strict = sf.validate_plan(records, caps, mode=sf.STRICT_ACTUAL, max_routes=5)
    assert strict["certified"] is False
    assert strict["unverifiable_constraints"] == ["SOURCE_SURPLUS=PROXY", "TARGET_NEED=PROXY"]  # PROXY, not MISSING
    missing = sf.validate_plan([rec("X", 5, 1, source="S9", target="T9")], {}, mode=sf.STRICT_ACTUAL, max_routes=5)
    assert missing["unverifiable_constraints"] == ["SOURCE=MISSING", "TARGET_NEED=MISSING"]


def test_inputs_are_not_mutated():
    records, caps = _random_instance(3)
    before = copy.deepcopy(records)
    select(records, caps)
    assert records == before


def test_real_transport_recompute_only_for_priced_rows():
    assert sf.real_transport_cost_recompute({"real_transport_applied": False, "source_id": "a"}, 5) is None
    assert sf.real_transport_cost_recompute({"source_id": "a"}, 5) is None


# ------------------------------------------------------------------------------------------- pipeline


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_excel_data(path)


def test_pipeline_field_is_parallel_and_fault_isolated():
    data = _load(NETWORK_SAMPLE)
    with_field = build_v2_state(copy.deepcopy(data), detail_level="full")
    with mock.patch.object(analysis_pipeline, "build_shared_feasibility_analysis", side_effect=RuntimeError("boom")):
        broken = build_v2_state(copy.deepcopy(data), detail_level="full")
    for key in ("top5", "summary", "connected_algorithms", "status", "seller_loss_analysis"):
        assert with_field["pipeline_result"][key] == broken["pipeline_result"][key]
    assert with_field["recommendations"] == broken["recommendations"]
    assert broken["pipeline_result"]["shared_feasibility_selection"]["status"] == "error"
    analysis = with_field["pipeline_result"]["shared_feasibility_selection"]
    again = build_v2_state(copy.deepcopy(data), detail_level="full")["pipeline_result"]["shared_feasibility_selection"]
    assert again == analysis  # deterministic: no timings in the pipeline field
    assert analysis["status"] == "parallel_only" and analysis["production_action_applied"] is False
    assert analysis["legacy_top_n_ids"] == [item["route_id"] for item in with_field["pipeline_result"]["top5"]]
    assert set(analysis["modes"]) == set(sf.MODES)


def test_normal_sample_has_no_executable_move_under_shared_feasibility_v1():
    from services import optimality_gap_service as og

    data = _load(NORMAL_SAMPLE)
    state = build_v2_state(copy.deepcopy(data), detail_level="core")
    for mode, plan in state["pipeline_result"]["shared_feasibility_selection"]["modes"].items():
        assert plan["plan_status"] == "NO_FEASIBLE_SELECTION" and plan["selected_count"] == 0
        assert plan["rejection_category_counts"] == {"INDIVIDUALLY_OVER_CAP": 6}
        assert all(row["target_need_cap"] == 0 and row["individual_cap_room"] == 0 for row in plan["rows"])
        assert plan["legacy_comparison"]["legacy_validation"]["violation_count"] == 5
    # the pre-existing in-app validator (same cap definition) also selects nothing
    settings = og.build_optimality_settings()
    problem = og.prepare_optimality_problem(state["recommendations"], data, settings)
    context = og.build_constraint_context(problem["candidates"], data, settings)
    assert og.build_ordered_feasible_combination(problem["candidates"], context, "varo")["route_ids"] == []


def test_network_sample_separates_executability_from_cost_comparability():
    state = build_v2_state(_load(NETWORK_SAMPLE), detail_level="core")
    plan = state["pipeline_result"]["shared_feasibility_selection"]["modes"][sf.STRICT_ACTUAL]
    assert plan["plan_status"] == sf.NO_COST_COMPARABLE_SELECTION and plan["selected_count"] == 0
    assert plan["status_counts"] == {sf.COST_NOT_COMPARABLE: 7, sf.REJECTED_INFEASIBLE_ROUTE: 1}
    view = plan["quantity_feasibility_view"]
    assert view["selected_count"] == 5 and view["total_move_cost"] is None and view["validation"]["violation_count"] == 0
    for row in view["rows"]:
        if row["selection_status"] == sf.PARTIALLY_SELECTED:
            assert row["allocated_qty"] <= row["target_need_cap"] and row["allocated_move_cost"] is None


def test_strict_mode_rejects_a_need_computed_from_a_demand_proxy():
    records = [rec("R1", 10, 1, rank=1)]
    data = {"inventory": pd.DataFrame([
        {"store_id": "S1", "product_id": "P1", "stock_qty": 500, "stock_qty_provenance": "actual", "sales_qty": 1,
         "sales_qty_provenance": "derived_proxy_from_actual_outbound_qty", "sales_qty_semantics": "demand_proxy_not_retail_sales"},
        {"store_id": "T1", "product_id": "P1", "stock_qty": 0, "stock_qty_provenance": "actual", "sales_qty": 5,
         "sales_qty_provenance": "derived_proxy_from_actual_outbound_qty", "sales_qty_semantics": "demand_proxy_not_retail_sales"}])}
    caps = sf.pipeline_caps(records, data)
    assert caps[(sf.SOURCE_STOCK, ("S1", "P1"))].provenance == "DIRECT_REAL"
    assert caps[(sf.TARGET_NEED, ("T1", "P1"))].provenance == "PROXY" and caps[(sf.TARGET_NEED, ("T1", "P1"))].value == 35
    assert select(records, caps, mode=sf.STRICT_ACTUAL)["plan_status"] == sf.INSUFFICIENT_CAP_DATA
    bench = select(records, caps)
    assert bench["selected_ids"] == ["R1"] and bench["proxy_caps_used"] is True

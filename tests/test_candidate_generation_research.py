"""P1 candidate generation research: controlled scenarios A-Z plus fidelity and diagnostics checks.

Synthetic uploads use the six Suhyup node ids (the offline benchmark caps read only those nodes) and run in the legacy
cost mode (uploaded route cost) unless a scenario injects a tariff-like pricer.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from services import candidate_generation_research as cgr
from services import candidate_generation_validation as cgv
from services import candidate_generator as cg
from services import milp_benchmark_integrity as mbi
from services import milp_benchmark_integrity_validation as mbv
from services import shared_feasibility_selection as sf
from services import shared_feasibility_validation as sfv
from services import suhyup_algorithm_revalidation as rv

N = rv.NETWORK_NODES
A, B, C, D, E, F = N
DATE = "2026-07-01"


@pytest.fixture(autouse=True)
def legacy_mode(monkeypatch):
    monkeypatch.delenv("VARO_REAL_DATA_ROOT", raising=False)


def _upload(inventory, *, costs=None, absent=(), prices=None, extra_routes=(), dc=None, nodes=N, distance=30.0,
            route_fields=None):
    """inventory: (store, product, stock, sales[, {extra columns}]); costs: {(s, t): route cost}, default 5000."""
    stores = [{"node_id": n, "node_name": f"n{n}", "node_type": "STORE"} for n in nodes]
    if dc:
        stores.append({"node_id": dc, "node_name": "dc", "node_type": "DC"})
    products = sorted({row[1] for row in inventory})
    rows = []
    for item in inventory:
        store, product, stock, sales, *extra = item
        rows.append({"store_id": store, "product_id": product, "stock_qty": stock, "sales_qty": sales, **(extra[0] if extra else {})})
    routes = []
    for s in nodes:
        for t in nodes:
            if s != t and (s, t) not in absent:
                routes.append({"source_id": s, "target_id": t, "distance_km": distance,
                               "estimated_cost": (costs or {}).get((s, t), 5000.0), "travel_time_min": 30.0,
                               **(route_fields or {}).get((s, t), {})})
    routes.extend(extra_routes)
    return {"stores": pd.DataFrame(stores),
            "products": pd.DataFrame({"product_id": products, "product_name": products,
                                      "unit_price": [(prices or {}).get(p, 1000.0) for p in products]}),
            "inventory": pd.DataFrame(rows), "routes": pd.DataFrame(routes)}


def _universe(upload, *, pricer=None, transport_mode=False):
    lanes, pairs, trace = cgr.enumerate_universe(upload, transport_mode=transport_mode, snapshot_date=DATE)
    priced = cgr.price_universe(lanes, pricer)
    capped = cgr.attach_benchmark_caps(priced, cgr.upload_inventory_flow(upload["inventory"], DATE))
    store_ids, _ = cg._store_ids_by_type(upload["stores"])
    universe = cgr.validate_universe(capped, store_ids=store_ids, product_ids=upload["products"]["product_id"].astype(str))
    return universe, pairs, trace


def _eval(pool, *, cap_spec=sfv.SUHYUP_CAP_SPEC, **kwargs):
    return cgr.evaluate_pool(pool, date=DATE, dataset="test", cap_spec=cap_spec, milp_cap_spec=mbv.MILP_CAP_SPEC, **kwargs)


def _run(upload, *policies, **kwargs):
    universe, pairs, trace = _universe(upload, **{k: kwargs.pop(k) for k in ("pricer", "transport_mode") if k in kwargs})
    pools = {policy: cgr.policy_pool(universe, policy) for policy in (policies or cgr.POLICIES)}
    return SimpleNamespace(universe=universe, pairs=pairs, trace=trace, pools=pools,
                           evals={policy: _eval(pool, **kwargs) for policy, pool in pools.items()})


def _lanes(plan):
    return sorted(plan["lane_keys"])


def _key(product, source, target):
    return cgr.lane_key(product, source, target)


def _one_source(source_stock, demands, costs, **kwargs):
    """P1 at A (surplus) and at B, C, D (stock 10, given daily demand); E and F do not hold P1."""
    inventory = [(A, "P1", source_stock, 0)] + [(node, "P1", 10, demands.get(node, 0)) for node in (B, C, D)]
    return _upload(inventory, costs={(A, node): cost for node, cost in costs.items()}, **kwargs)


def _ranked21(*, tie=False):
    """21 products, each on its own lane (source stock 300, target stock 10 / sales 10 -> qty 50, need 215).

    Products 1-20 have price 2000 and lane cost 1000 + 100 i; product 21 has the cheapest lane (100) but price 600, so
    the lowest saving and candidate_score: it is generator rank 21.  ``tie`` gives P21 exactly P20's score instead.
    """
    lanes = [(s, t) for s in N for t in N if s != t][:21]
    inventory, costs, prices = [], {}, {}
    for index, (s, t) in enumerate(lanes, start=1):
        product = f"P{index:02d}"
        inventory += [(s, product, 300, 0), (t, product, 10, 10)]
        costs[(s, t)] = 1000.0 + 100 * index if index <= 20 else (3000.0 if tie else 100.0)
        prices[product] = 2000.0 if index <= 20 or tie else 600.0
    return _upload(inventory, costs=costs, prices=prices), lanes


# --------------------------------------------------------------------------------------------- A-Z scenarios


def test_a_optimum_already_inside_the_top_20():
    upload = _upload([(A, "P1", 120, 0), (B, "P1", 10, 8), (C, "P1", 10, 0),
                      (C, "P2", 120, 0), (D, "P2", 10, 8), (E, "P2", 10, 0)], costs={(A, B): 1000.0, (C, D): 2000.0})
    run = _run(upload, cgr.LEGACY_20, cgr.TOP_30, cgr.GENERATOR_ALL, cgr.ALL_VALID)
    legacy = run.evals[cgr.LEGACY_20]
    assert len(run.pools[cgr.LEGACY_20]) == 2 and run.trace["cut_by_max_candidates"] == 0
    assert legacy["plans"]["MILP"]["service"] == 100 and legacy["plans"]["MILP"]["cost"] == 3000.0
    for policy in (cgr.TOP_30, cgr.GENERATOR_ALL, cgr.ALL_VALID):
        assert _lanes(run.evals[policy]["plans"]["MILP"]) == _lanes(legacy["plans"]["MILP"])
        compared = cgr.compare_pools(run.evals[policy], legacy, "MILP", cap_definition="x", superset=True)
        assert compared["cost_delta"] == 0.0 and compared["superset_monotone"] is True
    assert legacy["enumeration"]["proven"] and legacy["milp_evidence"]["solver_status"] == mbi.OPTIMAL
    assert legacy["plans"]["VARO_FINAL"]["comparison_status"] == mbi.DIRECTLY_COMPARABLE


def test_b_optimal_lane_at_generator_rank_21_is_cut_by_the_limit_and_recovered_by_top_30():
    upload, lanes = _ranked21()
    run = _run(upload, cgr.LEGACY_20, cgr.TOP_30)
    cheap = _key("P21", *lanes[20])
    full = run.universe[run.universe["generator_rank"].notna()]
    assert int(full.loc[full["lane_key"] == cheap, "generator_rank"].iloc[0]) == 21
    assert cheap not in set(run.pools[cgr.LEGACY_20]["lane_key"]) and cheap in set(run.pools[cgr.TOP_30]["lane_key"])
    legacy, top30 = run.evals[cgr.LEGACY_20]["plans"]["MILP"], run.evals[cgr.TOP_30]["plans"]["MILP"]
    assert legacy["service"] == top30["service"] == 250
    assert legacy["cost"] == 1100 + 1200 + 1300 + 1400 + 1500 and top30["cost"] == 100 + 1100 + 1200 + 1300 + 1400
    assert cheap in top30["lane_keys"]
    membership = cgr.pool_membership(run.universe, run.pools).set_index("lane_key")
    assert membership.loc[cheap, f"pool_{cgr.LEGACY_20}"] == "CUT_BY_LIMIT"
    assert membership.loc[cheap, f"pool_{cgr.TOP_30}"] == "IN_POOL"


def test_c_one_target_per_source_product_loses_a_cheaper_equal_service_lane():
    upload = _one_source(60, {B: 8, C: 1, D: 10}, {B: 1000.0, C: 2000.0, D: 9000.0})
    run = _run(upload, cgr.LEGACY_20, cgr.ALL_VALID)
    assert list(run.pools[cgr.LEGACY_20]["lane_key"]) == [_key("P1", A, D)]          # largest need (70) kept
    assert sorted(run.pools[cgr.ALL_VALID]["lane_key"]) == sorted(_key("P1", A, t) for t in (B, C, D))
    legacy, valid = run.evals[cgr.LEGACY_20]["plans"]["MILP"], run.evals[cgr.ALL_VALID]["plans"]["MILP"]
    assert (legacy["service"], legacy["cost"]) == (50, 9000.0)
    assert (valid["service"], valid["cost"]) == (50, 1000.0) and valid["lane_keys"] == [_key("P1", A, B)]
    compared = cgr.compare_pools(run.evals[cgr.ALL_VALID], run.evals[cgr.LEGACY_20], "MILP", cap_definition="x", superset=True)
    assert compared["pool_comparison_status"] == "SAME_MODEL_DIFFERENT_POOL" and compared["cost_delta"] == -8000.0
    membership = cgr.pool_membership(run.universe, run.pools).set_index("lane_key")
    assert membership.loc[_key("P1", A, B), f"pool_{cgr.LEGACY_20}"] == "CUT_BY_ONE_TARGET_RULE"


def test_d_several_destinations_raise_service_and_the_cost_is_then_not_compared():
    upload = _one_source(300, {B: 4, C: 5, D: 6}, {B: 1000.0, C: 2000.0, D: 3000.0})
    run = _run(upload, cgr.LEGACY_20, cgr.ALL_VALID)
    legacy, valid = run.evals[cgr.LEGACY_20]["plans"]["MILP"], run.evals[cgr.ALL_VALID]["plans"]["MILP"]
    assert legacy["service"] == 42 and valid["service"] == 28 + 35 + 42 and valid["cost"] == 6000.0
    compared = cgr.compare_pools(run.evals[cgr.ALL_VALID], run.evals[cgr.LEGACY_20], "MILP", cap_definition="x")
    assert compared["cost_delta"] is None and "SERVICE_NOT_EQUAL" in compared["reason_codes"]
    coverage = cgr.coverage_metrics(run.pools[cgr.ALL_VALID], run.universe)
    assert coverage["alternative_targets_per_source_product_max"] == 3
    assert run.evals[cgr.ALL_VALID]["plans"]["T1_ALL_OR_NOTHING"]["check"]["violation_count"] == 0


def test_e_duplicate_inventory_rows_are_identified_with_lineage_not_deleted():
    upload = _upload([(A, "P1", 120, 0), (A, "P1", 120, 0), (B, "P1", 10, 8), (C, "P1", 10, 0), (D, "P1", 10, 0)])
    run = _run(upload, cgr.LEGACY_20, cgr.ALL_VALID, cgr.ALL_LANES)
    produced, stats = cg.generate_candidates(dict(upload))
    assert stats["duplicate_removed"] == 1 and run.trace["duplicate_removed"] == 1
    second = run.universe[run.universe["inventory_row"] == 1]
    assert len(second) == 5 and second["is_duplicate"].all() and (second["first_inventory_row"] == 0).all()
    assert (second["validity_status"] == "DUPLICATE").all() and second["route_id"].str.contains("#DUP1").all()
    for pool in run.pools.values():
        assert not pool["lane_key"].duplicated().any() and not pool["is_duplicate"].any()
    assert cgr.production_records(run.pools[cgr.LEGACY_20]).astype(str).equals(produced.astype(str))
    records = cgr.candidate_records(run.universe)
    assert len(records) == len(run.universe) and records["lineage"].str.contains("first_row=0").sum() == 10


def test_f_shared_source_surplus_is_never_overdrawn():
    upload = _one_source(80, {B: 8, D: 10}, {B: 1000.0, D: 2000.0})        # surplus 70: one 50-move fits, not two
    run = _run(upload, cgr.ALL_VALID)
    plans = run.evals[cgr.ALL_VALID]["plans"]
    for name in ("VARO_FINAL", "T1_ALL_OR_NOTHING", "T1_PARTIAL", "MILP"):
        assert plans[name]["service"] == 50 and plans[name]["check"]["source_excess_qty"] == 0
    assert plans["MILP"]["lane_keys"] == [_key("P1", A, B)]
    assert plans["T1_PARTIAL"]["partial_count"] == 0                  # undeclared cost basis: no split without a cost


def test_g_shared_target_need_is_never_overfilled():
    upload = _upload([(A, "P1", 120, 0), (B, "P1", 120, 0), (C, "P1", 10, 8), (D, "P1", 10, 0), (E, "P1", 10, 0)],
                     costs={(A, C): 1000.0, (B, C): 1500.0})
    run = _run(upload, cgr.LEGACY_20)
    pool = run.pools[cgr.LEGACY_20]
    assert sorted(pool["lane_key"]) == sorted([_key("P1", A, C), _key("P1", B, C)])     # both rows keep target C
    legacy_slice = cgr.rank_pool(pool).sort_values("varo_final_rank").head(5)
    caps = sf.caps_from_columns(rv._records(legacy_slice), sfv.SUHYUP_CAP_SPEC)
    assert sf.validate_plan(rv._records(legacy_slice), caps, mode=sf.BENCHMARK_PROXY, max_routes=5)["target_excess_qty"] == 44
    plans = run.evals[cgr.LEGACY_20]["plans"]
    for name in ("VARO_FINAL", "T1_ALL_OR_NOTHING", "MILP"):
        assert plans[name]["service"] == 50 and plans[name]["check"]["target_excess_qty"] == 0
    assert plans["MILP"]["cost"] == 1000.0


def test_h_different_products_on_one_lane_are_separate_moves():
    upload = _upload([(A, "P1", 120, 0), (B, "P1", 10, 8), (C, "P1", 10, 0),
                      (A, "P2", 120, 0), (B, "P2", 10, 8), (C, "P2", 10, 0)], costs={(A, B): 1000.0})
    run = _run(upload, cgr.ALL_VALID)
    plan = run.evals[cgr.ALL_VALID]["plans"]["MILP"]
    assert _lanes(plan) == sorted([_key("P1", A, B), _key("P2", A, B)]) and plan["service"] == 100
    assert plan["check"]["duplicate_count"] == 0


def test_i_missing_or_forbidden_routes_never_become_selectable_candidates():
    upload = _one_source(120, {B: 8, C: 9, D: 10}, {B: 1000.0, D: 500.0}, absent=((A, C),),
                         route_fields={(A, D): {"feasible": False}})
    run = _run(upload, cgr.LEGACY_20, cgr.ALL_VALID, cgr.ALL_LANES)
    status = run.universe.set_index("lane_key")["route_status"]
    assert status[_key("P1", A, C)] == cgr.ROUTE_UNVERIFIED and status[_key("P1", A, D)] == cgr.ROUTE_FORBIDDEN
    assert all(_key("P1", A, C) not in set(pool["lane_key"]) for pool in run.pools.values())
    assert list(run.pools[cgr.LEGACY_20]["lane_key"]) == [_key("P1", A, D)]          # production ignores the flag
    legacy = run.evals[cgr.LEGACY_20]
    assert legacy["excluded_count"] == 1 and legacy["plans"]["MILP"]["selected_count"] == 0
    assert _key("P1", A, D) not in set(run.pools[cgr.ALL_VALID]["lane_key"])
    assert _key("P1", A, D) not in run.evals[cgr.ALL_LANES]["plans"]["MILP"]["lane_keys"]
    compared = cgr.compare_pools(run.evals[cgr.ALL_VALID], legacy, "MILP", cap_definition="x")
    assert compared["pool_comparison_status"] == mbi.NOT_COMPARABLE and "DATA_INVALID_LANES_EXCLUDED" in compared["reason_codes"]


def _stub_pricer(costs, *, unapplied=(), proxy=(), negative=()):
    def pricer(frame):
        out = frame.copy()
        lanes = list(zip(out["source_id"].astype(str), out["target_id"].astype(str)))
        applied = [lane not in unapplied for lane in lanes]
        out["real_transport_applied"] = applied
        out["real_transport_status"] = ["applied" if ok else "product_weight_insufficient" for ok in applied]
        out["move_cost"] = [(-5.0 if lane in negative else costs.get(lane, 4000.0)) if ok else 0.0
                            for lane, ok in zip(lanes, applied)]
        out["proxy_vehicle_count"] = [1 if lane in proxy else 0 for lane in lanes]
        return out
    return pricer


def test_j_unknown_cost_stays_null_and_never_reaches_a_selector_as_zero():
    upload = _one_source(120, {B: 10, C: 8}, {})
    run = _run(upload, cgr.LEGACY_20, cgr.ALL_VALID, transport_mode=True,
               pricer=_stub_pricer({(A, B): 3000.0, (A, C): 2000.0}, unapplied={(A, B)}))
    lane = run.universe.set_index("lane_key").loc[_key("P1", A, B)]
    assert pd.isna(lane["move_cost"]) and pd.isna(lane["estimated_cost"]) and lane["cost_provenance"] == cgr.UNKNOWN
    assert lane["validity_status"] == "COST_UNKNOWN"
    assert cgr.production_records(run.pools[cgr.LEGACY_20])["estimated_cost"].tolist() == [0.0]  # production's deferred 0
    legacy = run.evals[cgr.LEGACY_20]
    assert legacy["excluded_count"] == 1 and legacy["plans"]["MILP"]["selected_count"] == 0
    assert run.evals[cgr.ALL_VALID]["plans"]["MILP"]["lane_keys"] == [_key("P1", A, C)]


def test_k_unit_mismatch_or_unknown_unit_is_excluded_and_matched_units_reach_t1():
    upload = _upload([(A, "P1", 120, 0, {"quantity_unit": "EA"}), (B, "P1", 10, 10, {"quantity_unit": "BOX"}),
                      (C, "P1", 10, 8, {"quantity_unit": "EA"}), (D, "P1", 10, 9, {"quantity_unit": None})],
                     costs={(A, B): 500.0, (A, C): 2000.0, (A, D): 600.0})
    run = _run(upload, cgr.LEGACY_20, cgr.ALL_VALID)
    status = run.universe.set_index("lane_key")["validity_status"]
    assert status[_key("P1", A, B)] == "UNIT_MISMATCH" and status[_key("P1", A, D)] == "UNIT_UNKNOWN"
    assert status[_key("P1", A, C)] == cgr.VALID
    assert run.evals[cgr.LEGACY_20]["excluded_count"] == 1                 # generator kept B (largest need)
    plans = run.evals[cgr.ALL_VALID]["plans"]
    assert plans["MILP"]["lane_keys"] == [_key("P1", A, C)] and plans["T1_ALL_OR_NOTHING"]["lane_keys"] == [_key("P1", A, C)]


def test_l_proxy_route_and_proxy_cost_are_labelled_not_promoted():
    upload = _one_source(120, {B: 8}, {B: 0.0}, route_fields={(A, B): {"distance_provenance": "haversine_proxy"}})
    universe, _, _ = _universe(upload)
    lane = universe.set_index("lane_key").loc[_key("P1", A, B)]
    assert lane["cost_provenance"] == cgr.CONFIG and lane["cost_source"] == "distance_km_x_100_default"
    assert lane["move_cost"] == 30.0 * 100 and lane["distance_provenance"] == "haversine_proxy"
    assert lane["route_operational_evidence"] == cgr.OPERATION_NOT_OBSERVED
    real, _, _ = _universe(_one_source(120, {B: 8}, {}), transport_mode=True,
                           pricer=_stub_pricer({(A, B): 900.0}, proxy={(A, B)}))
    assert real.set_index("lane_key").loc[_key("P1", A, B), "cost_provenance"] == cgr.PROXY
    assert real.set_index("lane_key").loc[_key("P1", A, B), "cost_basis"] == sf.QUANTITY_SPECIFIC


def test_m_route_capacity_binds_t1_but_not_the_milp_so_they_are_not_comparable():
    run = _run(_one_source(120, {B: 8}, {B: 1000.0}), cgr.ALL_VALID)
    pool = run.pools[cgr.ALL_VALID].assign(route_capacity_qty=30.0)
    evaluated = _eval(pool)
    plans = evaluated["plans"]
    assert plans["T1_ALL_OR_NOTHING"]["selected_count"] == 0
    assert plans["MILP"]["selected_count"] == 1 and plans["MILP"]["check"]["passed"] is False
    assert plans["T1_ALL_OR_NOTHING"]["comparison_status"] == mbi.NOT_COMPARABLE


def test_n_dc_capacity_binds_t1_and_flags_the_milp():
    upload = _upload([(A, "P1", 120, 0), (B, "P1", 10, 8), (A, "P2", 120, 0), (C, "P2", 10, 8)], nodes=(A, B, C), dc="DC1",
                     absent={(s, t) for s in (A, B, C) for t in (A, B, C)},
                     extra_routes=[{"source_id": s, "target_id": t, "distance_km": 10.0, "estimated_cost": 500.0,
                                    "travel_time_min": 10.0} for s, t in ((A, "DC1"), ("DC1", B), ("DC1", C))])
    run = _run(upload, cgr.ALL_VALID)
    pool = run.pools[cgr.ALL_VALID]
    assert set(pool["route_status"]) == {cgr.ROUTE_VIA_DC_DERIVED} and set(pool["dc_id"]) == {"DC1"} and len(pool) == 2
    spec = {**sfv.SUHYUP_CAP_SPEC, sf.DC_CAPACITY: (("dc_capacity_qty",), "USER_INPUT", "test DC throughput")}
    evaluated = _eval(pool.assign(dc_capacity_qty=60.0), cap_spec=spec)
    assert evaluated["plans"]["T1_ALL_OR_NOTHING"]["service"] == 50
    assert evaluated["plans"]["MILP"]["service"] == 100 and evaluated["plans"]["MILP"]["check"]["passed"] is False


def test_o_p_q_candidate_limits_equal_the_production_generator(monkeypatch):
    upload, _ = _ranked21()
    universe, _, trace = _universe(upload)
    produced, _ = cg.generate_candidates(dict(upload))
    legacy = cgr.policy_pool(universe, cgr.LEGACY_20)
    assert len(legacy) == 20 and trace["generator_pool_before_cut"] == 21
    assert cgr.production_records(legacy).astype(str).equals(produced.astype(str))
    for policy, limit in ((cgr.TOP_30, 30), (cgr.GENERATOR_ALL, 10 ** 9)):
        monkeypatch.setattr(cg, "MAX_CANDIDATES", limit)
        produced, _ = cg.generate_candidates(dict(upload))
        assert cgr.production_records(cgr.policy_pool(universe, policy)).astype(str).equals(produced.astype(str))
    assert len(cgr.policy_pool(universe, cgr.TOP_30)) == 21


def test_r_equal_service_lower_cost_is_a_pool_sensitivity_not_a_decision_gap():
    upload, _ = _ranked21()
    run = _run(upload, cgr.LEGACY_20, cgr.TOP_30)
    compared = cgr.compare_pools(run.evals[cgr.TOP_30], run.evals[cgr.LEGACY_20], "MILP", cap_definition="x", superset=True)
    assert compared["pool_comparison_status"] == "SAME_MODEL_DIFFERENT_POOL" and compared["service_equal"]
    assert compared["cost_delta"] == -1400.0 and compared["superset_monotone"] is True
    gap = run.evals[cgr.TOP_30]["plans"]["MILP"]["signature"] != run.evals[cgr.LEGACY_20]["plans"]["MILP"]["signature"]
    assert gap                                                          # different pools never share a T3 signature


def test_s_a_cheaper_lower_service_plan_is_not_preferred():
    upload = _one_source(60, {B: 8, C: 1}, {B: 9000.0, C: 100.0})
    run = _run(upload, cgr.ALL_VALID)
    plan = run.evals[cgr.ALL_VALID]["plans"]["MILP"]
    assert plan["lane_keys"] == [_key("P1", A, B)] and (plan["service"], plan["cost"]) == (50, 9000.0)
    cheap = _eval(run.pools[cgr.ALL_VALID][run.pools[cgr.ALL_VALID]["target_id"] == C])
    compared = cgr.compare_pools(cheap, run.evals[cgr.ALL_VALID], "MILP", cap_definition="x")
    assert cheap["plans"]["MILP"]["cost"] < plan["cost"] and compared["cost_delta"] is None
    assert "SERVICE_NOT_EQUAL" in compared["reason_codes"]


def test_t_larger_pools_are_timed_and_lose_enumeration_but_keep_solver_proof():
    upload, _ = _ranked21()
    run = _run(upload, cgr.LEGACY_20, cgr.ALL_LANES)
    small, large = run.evals[cgr.LEGACY_20], run.evals[cgr.ALL_LANES]
    assert large["pool_size"] > 5 * small["pool_size"]
    for evaluated in (small, large):
        assert all(evaluated["timing"][key] >= 0 for key in ("rank_ms", "varo_ms", "t1_ms", "milp_ms", "enumeration_ms"))
    assert small["enumeration"]["status"] == "ENUMERATED" and large["enumeration"]["status"] == "NOT_ENUMERATED"
    assert large["milp_evidence"]["solver_status"] == mbi.OPTIMAL


def test_u_v_ordering_is_deterministic_and_repeat_runs_are_identical():
    upload, _ = _ranked21()
    first, second = _run(upload, cgr.ALL_VALID, cgr.ALL_LANES), _run(upload, cgr.ALL_VALID, cgr.ALL_LANES)
    pool = first.pools[cgr.ALL_LANES]
    ranked = pool[pool["generator_rank"].notna()]
    assert list(ranked["generator_rank"].astype(int)) == sorted(ranked["generator_rank"].astype(int))
    assert list(pool["route_id"]) == list(second.pools[cgr.ALL_LANES]["route_id"]) and pool["route_id"].is_unique
    columns = ["lane_key", "recommended_qty", "move_cost", "validity_status", "candidate_score"]
    assert cgr.frame_digest(first.universe, columns) == cgr.frame_digest(second.universe, columns)
    for policy in (cgr.ALL_VALID, cgr.ALL_LANES):
        for plan in ("VARO_FINAL", "T1_ALL_OR_NOTHING", "MILP"):
            assert first.evals[policy]["plans"][plan]["lane_keys"] == second.evals[policy]["plans"][plan]["lane_keys"]


def test_w_shuffled_rows_keep_uncut_pools_while_a_tie_at_the_cut_follows_row_order():
    upload, lanes = _ranked21()
    shuffled = {**upload, "inventory": upload["inventory"].sample(frac=1.0, random_state=7).reset_index(drop=True)}
    left, right = _run(upload, cgr.MULTI_TARGET, cgr.ALL_VALID, cgr.ALL_LANES), _run(shuffled, cgr.MULTI_TARGET, cgr.ALL_VALID, cgr.ALL_LANES)
    for policy in (cgr.MULTI_TARGET, cgr.ALL_VALID, cgr.ALL_LANES):
        assert cgv._keyed(left.pools[policy]) == cgv._keyed(right.pools[policy])
        assert left.evals[policy]["plans"]["MILP"]["cost"] == right.evals[policy]["plans"]["MILP"]["cost"]
    tied, _ = _ranked21(tie=True)
    reversed_rows = {**tied, "inventory": tied["inventory"].iloc[::-1].reset_index(drop=True)}
    forward, _, trace = _universe(tied)
    backward, _, _ = _universe(reversed_rows)
    assert trace["cut_boundary_tie"] is True
    keep = lambda universe: set(cgr.policy_pool(universe, cgr.LEGACY_20)["lane_key"])
    assert keep(forward) != keep(backward)                             # production keeps the earlier row at a tie


def test_x_strict_actual_never_plans_on_proxy_caps():
    run = _run(_one_source(120, {B: 8}, {B: 1000.0}), cgr.ALL_VALID)
    plans = run.evals[cgr.ALL_VALID]["plans"]
    assert plans["T1_STRICT_ACTUAL"]["plan_status"] == sf.INSUFFICIENT_CAP_DATA and plans["T1_STRICT_ACTUAL"]["service"] == 0
    assert plans["T1_STRICT_ACTUAL"]["comparison_status"] == mbi.INSUFFICIENT_DATA
    assert plans["T1_ALL_OR_NOTHING"]["service"] == 50
    assert "PROXY_CAPS_SATISFIED_NOT_AN_OPERATIONAL_GUARANTEE" in plans["T1_ALL_OR_NOTHING"]["feasibility_claim"]


def test_y_wrong_cost_provenance_is_rejected():
    upload = _one_source(120, {B: 10, C: 8}, {})
    universe, _, _ = _universe(upload, transport_mode=True,
                               pricer=_stub_pricer({(A, C): 2000.0}, unapplied={(A, B)}, negative={(A, C)}))
    status = universe.set_index("lane_key")
    assert status.loc[_key("P1", A, B), "cost_provenance"] == cgr.UNKNOWN and pd.isna(status.loc[_key("P1", A, B), "move_cost"])
    assert status.loc[_key("P1", A, C), "validity_status"] == "COST_UNKNOWN"   # a DERIVED_REAL label on a negative cost
    kept, excluded = cgr.selectable_split(cgr.policy_pool(universe, cgr.ALL_LANES))
    assert {_key("P1", A, B), _key("P1", A, C)} <= set(excluded["lane_key"])


def _fake_result(status, x=None, fun=None, bound=None, gap=None, message=""):
    return SimpleNamespace(status=status, message=message, x=None if x is None else np.asarray(x, dtype=float),
                           fun=fun, mip_dual_bound=bound, mip_gap=gap, mip_node_count=0, success=status == 0)


def test_z_milp_time_limit_is_never_reported_optimal(monkeypatch):
    run = _run(_one_source(120, {B: 8}, {B: 1000.0}), cgr.ALL_VALID)
    calls = iter([_fake_result(0, [1], fun=-50.0, bound=-50.0, gap=0.0), _fake_result(1, None, message="Time limit reached.")])
    monkeypatch.setattr(rv, "milp", lambda *a, **k: next(calls))
    evaluated = _eval(run.pools[cgr.ALL_VALID], milp_options={"time_limit": 0.01})
    assert evaluated["milp_evidence"]["solver_status"] == mbi.TIME_LIMIT and evaluated["milp_evidence"]["time_limit_s"] == 0.01
    assert evaluated["scope"]["optimality_claim"] == "NO_OPTIMALITY_CLAIM"
    assert evaluated["plans"]["VARO_FINAL"]["comparison_status"] != mbi.DIRECTLY_COMPARABLE


# --------------------------------------------------------------------------------------------- fidelity and diagnostics


def test_research_universe_reproduces_the_generator_with_a_dc_and_matches_the_t3_lane_pool():
    upload = _upload([(A, "P1", 120, 0, {"days_to_expiry": 2}), (B, "P1", 10, 8), (C, "P1", 10, 3)], nodes=(A, B, C), dc="DC1",
                     costs={(A, B): 9000.0}, extra_routes=[{"source_id": s, "target_id": t, "distance_km": 5.0,
                                                            "estimated_cost": 100.0, "travel_time_min": 5.0}
                                                           for s, t in ((A, "DC1"), ("DC1", B), ("DC1", C))])
    universe, _, trace = _universe(upload)
    produced, _ = cg.generate_candidates(dict(upload))
    legacy = cgr.policy_pool(universe, cgr.LEGACY_20)
    assert cgr.production_records(legacy).astype(str).equals(produced.astype(str))
    assert legacy["route_type"].tolist() == ["VIA_DC"] and legacy["recommended_qty"].tolist() == [cg._SHORT_EXPIRY_CAP]
    t3 = mbv.lane_pool(upload)
    key = lambda f: sorted(zip(f["product_id"].astype(str), f["source_id"].astype(str), f["target_id"].astype(str),
                               f["recommended_qty"].astype(float)))
    assert key(t3) == key(cgr.policy_pool(universe, cgr.ALL_LANES))


def test_all_lanes_without_a_target_cap_is_not_comparable_and_explains_a_lower_cost():
    upload = _one_source(60, {B: 8, D: 10}, {B: 1000.0, D: 9000.0, E: 100.0})     # E does not hold P1
    run = _run(upload, cgr.LEGACY_20, cgr.ALL_VALID, cgr.ALL_LANES)
    lanes = run.evals[cgr.ALL_LANES]
    assert lanes["plans"]["MILP"]["lane_keys"] == [_key("P1", A, E)] and lanes["plans"]["MILP"]["cost"] == 100.0
    assert lanes["missing_caps"] is True
    lane = run.universe.set_index("lane_key").loc[_key("P1", A, E)]
    assert lane["validity_status"] == "TARGET_PRODUCT_NOT_HELD" and lane["need_fabricated"] and pd.isna(lane["target_need_7d"])
    compared = cgr.compare_pools(lanes, run.evals[cgr.LEGACY_20], "MILP", cap_definition="x")
    assert compared["pool_comparison_status"] == mbi.NOT_COMPARABLE and "MISSING_CAP" in compared["reason_codes"]
    assert run.evals[cgr.ALL_VALID]["plans"]["MILP"]["lane_keys"] == [_key("P1", A, B)]


def test_coverage_metrics_report_numerators_and_denominators():
    run = _run(_one_source(60, {B: 8, C: 1, D: 10}, {B: 1000.0, C: 2000.0, D: 9000.0}), cgr.LEGACY_20, cgr.ALL_VALID)
    best = run.evals[cgr.ALL_VALID]["plans"]["MILP"]["lane_keys"]
    coverage = cgr.coverage_metrics(run.pools[cgr.LEGACY_20], run.universe, reference_lane_keys=best)
    assert coverage["valid_lane_coverage"] == {"numerator": 1, "denominator": 3, "value": round(1 / 3, 6)}
    assert coverage["source_product_coverage"]["value"] == 1.0
    assert coverage["target_product_coverage"] == {"numerator": 1, "denominator": 3, "value": round(1 / 3, 6)}
    assert coverage["cheapest_lane_inclusion"]["value"] == 0.0 and coverage["best_known_plan_inclusion"]["value"] == 0.0
    assert coverage["duplicate_share"]["value"] == 0.0
    empty = cgr.coverage_metrics(run.pools[cgr.LEGACY_20].iloc[0:0], run.universe.iloc[0:0])
    assert empty["valid_lane_coverage"]["value"] is None                    # no denominator -> NULL, not 0


def test_shortage_diagnosis_separates_cuts_from_infeasibility():
    upload, lanes = _ranked21()
    extra = pd.DataFrame([
        {"store_id": A, "product_id": "PX", "stock_qty": 50, "sales_qty": 0},        # product not in master
        {"store_id": F, "product_id": "P01", "stock_qty": 0, "sales_qty": 0},        # no stock
    ])
    upload = {**upload, "inventory": pd.concat([upload["inventory"], extra], ignore_index=True)}
    universe, pairs, _ = _universe(upload)
    legacy = cgr.policy_pool(universe, cgr.LEGACY_20)
    diagnosis = cgr.shortage_diagnosis(pairs, universe, legacy, cgr.LEGACY_20)

    def category(source, product):
        found = diagnosis[(diagnosis["source_id"] == source) & (diagnosis["product_id"] == product)]["category"]
        assert len(found) == 1
        return found.iloc[0]

    assert category(lanes[20][0], "P21") == "CUT_BY_LIMIT"
    assert category(A, "PX") == "PRODUCT_NOT_IN_MASTER"
    assert category(F, "P01") == "SOURCE_NO_STOCK"
    assert category(lanes[0][1], "P01") == "SOURCE_NOT_SURPLUS"
    assert category(lanes[0][0], "P01") == "IN_POOL"
    no_route = _upload([(A, "P1", 120, 0), (B, "P1", 10, 8)], nodes=(A, B), absent={(A, B)})
    universe, pairs, _ = _universe(no_route)
    assert cgr.shortage_diagnosis(pairs, universe, universe.iloc[0:0], cgr.ALL_VALID)["category"].tolist()[0] == "NO_ROUTE"
    no_need = _upload([(A, "P1", 120, 0), (B, "P1", 10, 0), (C, "P1", 10, 0)], nodes=(A, B, C))   # targets: no demand
    universe, pairs, _ = _universe(no_need)
    diagnosis = cgr.shortage_diagnosis(pairs, universe, cgr.policy_pool(universe, cgr.ALL_VALID), cgr.ALL_VALID)
    assert diagnosis.loc[diagnosis["source_id"] == A, "category"].tolist() == ["NO_TARGET_NEED"]
    legacy = cgr.policy_pool(universe, cgr.LEGACY_20)          # the generator still keeps one, with a fabricated need
    assert len(legacy) == 1 and bool(legacy["need_fabricated"].iloc[0]) and legacy["validity_status"].iloc[0] == "NEED_NOT_EVIDENCED"


def test_candidate_records_keep_section_7_fields_with_nulls():
    universe, _, _ = _universe(_one_source(120, {B: 8}, {B: 1000.0}, absent=((A, C),)))
    records = cgr.candidate_records(universe)
    for column in ("candidate_id", "source_id", "target_id", "product_id", "recommended_qty", "source_stock", "source_surplus",
                   "target_need", "distance_km", "transport_cost", "candidate_score", "candidate_rank", "lineage", "provenance"):
        assert column in records.columns
    unrouted = records[records["candidate_id"] == _key("P1", A, C)].iloc[0]
    assert pd.isna(unrouted["transport_cost"]) and pd.isna(unrouted["recommended_qty"])


def test_unknown_policy_and_unvalidated_all_valid_are_errors():
    universe, _, _ = _universe(_one_source(120, {B: 8}, {B: 1000.0}))
    with pytest.raises(ValueError):
        cgr.policy_pool(universe, "TOP_99")
    with pytest.raises(ValueError):
        cgr.policy_pool(universe.drop(columns="validity_status"), cgr.ALL_VALID)


def test_bad_upload_gives_an_empty_universe_and_a_reason():
    lanes, pairs, trace = cgr.enumerate_universe({"stores": pd.DataFrame()})
    assert lanes.empty and pairs.empty and trace["generated"] is False and trace["reason"]


def test_research_path_is_not_imported_by_production_modules():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    production = [*root.glob("services/*.py"), *root.glob("pages/*.py"), *root.glob("components/*.py"), root / "app_v2.py"]
    users = [path.name for path in production if path.is_file()
             and "candidate_generation_research" in path.read_text(encoding="utf-8", errors="replace")
             and path.name not in ("candidate_generation_research.py", "candidate_generation_validation.py")]
    assert users == []
    assert cg.MAX_CANDIDATES == 20 and cg._MOVE_CAP == 50                # production constants untouched


def test_validation_bundle_names_the_required_files():
    for name in ("candidate_coverage_summary.csv", "candidate_coverage_by_day.csv", "candidate_coverage_by_policy.csv",
                 "candidate_cutoff_effect.csv", "candidate_newly_selected_routes.csv", "candidate_cost_comparison.csv",
                 "candidate_feasibility_validation.csv", "candidate_runtime_analysis.csv", "candidate_generation_validation.json"):
        assert name in cgv.OUTPUT_FILES
    assert cgv.OUTPUT_FOLDER == "_CANDIDATE_GENERATION_VALIDATION"

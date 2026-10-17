from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from services import forecast_candidate as f, candidate_dqn as d
from services.learning_policy_audit import inspect_policy,structural_signature,rolling_allocation_bridge
from services.candidate_vhs_normalization import normalized_vhs_candidate
from services.candidate_selection import Candidate,AllocationOption,SelectionProblem

def problem():
    cs=tuple(Candidate(k,"p","S",k,"FRESH","L","EA",{"S":1,k:1},k,
        (AllocationOption(1,2),AllocationOption(2,3))) for k in ("A","B"))
    return SelectionProblem("test",{"S":4,"A":2,"B":2},{"A":2,"B":2},cs,2,"SCENARIO")

@pytest.mark.parametrize("method",f.SECOND_METHODS)
def test_second_forecasts_preserve_constant_and_nonnegative(method):
    assert f.forecast(np.ones(70)*3,method).tolist()==pytest.approx([3]*7)
    assert np.all(f.forecast(np.zeros(70),method)==0)

def test_second_selector_and_predictions_ignore_outer_future():
    data=pd.DataFrame({"series":"a","date":pd.date_range("2026-01-01",periods=84),
                       "sales_qty":np.resize([0,0,2,0,1,3,4],84),"stockout_hours":0})
    a,da=f.evaluate_second_rolling(data,[70])
    data.loc[70:,"sales_qty"]=9999
    data.loc[70:,"stockout_hours"]=24
    b,db=f.evaluate_second_rolling(data,[70])
    assert da==db
    assert a.forecast.tolist()==b.forecast.tolist()
    assert b.target_uncensored.eq(False).all()
    assert a.groupby("method").size().nunique()==1

def test_second_unknown_availability_blocks_common_origin():
    data=pd.DataFrame({"series":"a","date":pd.date_range("2026-01-01",periods=77),"sales_qty":1,"stockout_hours":np.nan})
    rows,decisions=f.evaluate_second_rolling(data,[70])
    assert rows.empty and not decisions[0]["common_eligible"]
    assert decisions[0]["exclusion_reason"]=="NO_COMMON_UNCENSORED_INNER_LABELS"

@pytest.mark.parametrize("policy",["UNTRAINED_DQN","RANDOM_VALID","GREEDY_IMMEDIATE","MILP","VARO","PARTIAL"])
def test_audited_policies_share_independent_feasibility(policy):
    p=problem();r=inspect_policy(p,policy=policy)
    assert not r["evaluation"]["violations"]
    assert all(step["action"] in step["valid_actions"] for step in r["trace"])

def test_structural_overlap_not_just_changed_id():
    p=problem();q=replace(p,problem_id="new",capacities={"S":3,"A":2,"B":2})
    assert structural_signature(p,include_values=False)==structural_signature(q,include_values=False)
    assert structural_signature(p)!=structural_signature(q)

def test_rolling_bridge_is_nonanticipating_and_stock_conserving():
    p=problem()
    a=rolling_allocation_bridge(None,p,[{"A":1,"B":1},{"A":2,"B":2}],policy="MILP")
    b=rolling_allocation_bridge(None,p,[{"A":1,"B":1},{"A":999,"B":999}],policy="MILP")
    assert a["records"][0]==b["records"][0]
    assert a["records"][1]["plan"]==b["records"][1]["plan"]
    assert a["total_sales_proxy"]<=4
    assert all(v>=0 for r in a["records"] for v in r["source_stock"].values())

def test_new_training_records_td_loss_without_global_rng_mutation():
    import torch
    state=torch.random.get_rng_state().clone()
    model=d.train([problem()],seed=2,episodes=20)
    assert torch.equal(state,torch.random.get_rng_state())
    diagnostics=model.metadata["training_diagnostics"]
    assert len(diagnostics)==20
    assert any(x["mean_td_loss"] is not None for x in diagnostics)
    assert all(x["invalid_action_attempts"]==0 for x in diagnostics)

def vhs_frame():
    return pd.DataFrame({"route_id":["b","a","c"],"recommended_qty":[2,2,1],"move_cost":[10,10,20],
                         "expected_saving":[0,np.nan,10]})

def test_vhs_candidate_unit_scaling_and_permutation_invariant():
    frame=vhs_frame();a,ma=normalized_vhs_candidate(frame)
    frame.move_cost*=1000
    frame.recommended_qty*=.1
    b,mb=normalized_vhs_candidate(frame.sample(frac=1,random_state=3))
    assert a.set_index("route_id").candidate_vhs_score.to_dict()==b.set_index("route_id").candidate_vhs_score.to_dict()
    assert a.set_index("route_id").candidate_vhs_rank.to_dict()==b.set_index("route_id").candidate_vhs_rank.to_dict()
    assert "savings_score" not in ma["active_weights"]
    assert ma["actual_risk_calibrated"] is False

def test_vhs_all_missing_no_fake_neutral_score():
    result,audit=normalized_vhs_candidate(pd.DataFrame({"route_id":["a","b"]}))
    assert result.candidate_vhs_score.isna().all() and audit["status"]=="INPUT_REQUIRED"

def test_vhs_hand_calculated_and_deterministic_tie():
    result,audit=normalized_vhs_candidate(vhs_frame())
    assert result.candidate_vhs_score.tolist()==pytest.approx([100,100,0])
    assert result.candidate_vhs_rank.tolist()==[2,1,3]

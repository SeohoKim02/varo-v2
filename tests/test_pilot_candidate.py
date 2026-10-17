import hashlib
import json
import pytest
from services.operator_intake_candidate import IntakeReport
from services import pilot_candidate as p

def report():
    return IntakeReport(records={"inventory_snapshot":[{"source_dataset":"shop"}]},file_hashes={"inventory.csv":"a"*64})
def meta():
    return {"owner_authorized":True,"allowed_use":"RESEARCH_SHADOW","publication_scope":"NOT_ALLOWED",
            "storage_until":"2026-12-31","consent_reference":"operator-signed-local-record","network_id":"shop","dataset_id":"shop"}
def start(tmp_path):
    receipt=p.start_pilot(report(),meta(),root=tmp_path,pilot_id="p1")
    folder=tmp_path/"p1"
    decision=p.record_shadow_decision(folder,decision_id="d1",input_signature=receipt["input_signature"],
        actions=[{"action_id":"a1","network_id":"shop","action":"TRANSFER","evidence_class":"VERIFIED_INPUT"}],expected_net=4,currency="KRW")
    return folder,receipt,decision

def outcome():
    return IntakeReport(records={"action_outcome":[{"action_id":"a1","network_id":"shop","outcome_kind":"OBSERVED_ACTUAL",
        "evidence_class":"DIRECT_REAL","verification_status":"VERIFIED","operator_approved":True,
        "actual_revenue":20,"actual_transport_spend":5,"actual_handling_spend":1,"actual_storage_spend":2,
        "actual_disposal_spend":1,"currency":"KRW"}]},file_hashes={"outcome.csv":"b"*64})

def test_shadow_review_and_observed_outcome_are_separate(tmp_path):
    folder,receipt,decision=start(tmp_path)
    review=p.record_operator_review(folder,review_id="r1",decision_id="d1",reviewer_reference="operator-1",
        decision_hash=decision["event_sha256"],accepted=True,reason="현장 대조 필요")
    observed=p.record_observed_outcomes(folder,event_id="o1",decision_id="d1",outcome_report=outcome())
    assert decision["executed"] is False and review["execution_authorized"] is False
    assert observed["observed_net_components"]==11 and observed["actual_saving"] is None
    assert observed["causal_effect"]=="NOT_IDENTIFIED_NO_COUNTERFACTUAL"
    with pytest.raises(FileExistsError):p.start_pilot(report(),meta(),root=tmp_path,pilot_id="p1")

def test_partial_cash_is_unknown_not_zero(tmp_path):
    folder,_,_=start(tmp_path)
    r=outcome();r.records["action_outcome"][0]["actual_storage_spend"]=None
    result=p.record_observed_outcomes(folder,event_id="partial",decision_id="d1",outcome_report=r)
    assert result["observed_net_components"] is None

@pytest.mark.parametrize("change",[{"network_id":"other"},{"action_id":"missing"},{"outcome_kind":"SIMULATION"},{"operator_approved":False}])
def test_incomparable_or_simulated_outcome_blocked(tmp_path,change):
    folder,_,_=start(tmp_path)
    r=outcome();r.records["action_outcome"][0].update(change)
    with pytest.raises(ValueError):p.record_observed_outcomes(folder,event_id="bad",decision_id="d1",outcome_report=r)

def test_input_signature_and_review_hash_protected(tmp_path):
    folder,_,decision=start(tmp_path)
    with pytest.raises(ValueError):p.record_shadow_decision(folder,decision_id="bad",input_signature="wrong",actions=[])
    with pytest.raises(ValueError):p.record_operator_review(folder,review_id="bad",decision_id="d1",reviewer_reference="x",decision_hash="0"*64,accepted=True,reason="x")

@pytest.mark.parametrize("change",[{"owner_authorized":False},{"storage_until":None},{"consent_reference":None}])
def test_missing_operator_consent_blocks_receipt(tmp_path,change):
    with pytest.raises(ValueError):p.start_pilot(report(),{**meta(),**change},root=tmp_path,pilot_id="p1")

def test_cannot_write_protected_folder(tmp_path):
    with pytest.raises(ValueError):p.start_pilot(report(),meta(),root=tmp_path/"raw",pilot_id="p1")

"""Isolated research-pilot receipt, shadow decision, review and observed outcome.

No execution API, production database, model or canonical-data writes exist in
this module. Each event is append-only and tied to immutable input hashes.
"""
from __future__ import annotations
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping, Any
import hashlib
import json
import math
import re
from services.operator_intake_candidate import IntakeReport
from services.operating_snapshot import signature

VERSION="research-shadow-pilot-v0.1"
ID=re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


def _json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)


def _append(root: Path, event_id: str, event: Mapping[str, Any]):
    if not ID.fullmatch(event_id):
        raise ValueError("기록 식별자는 영문·숫자·밑줄·하이픈으로 입력해 주세요.")
    payload={"version":VERSION,"recorded_at":datetime.now(timezone.utc).isoformat(),**event}
    serialized=_json(payload)
    path=root/f"{event_id}.json"
    with path.open("x",encoding="utf-8") as handle:
        handle.write(serialized)
    return {"event_path":str(path),"event_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),**payload}


def start_pilot(report: IntakeReport, metadata: Mapping[str,Any], *, root: str|Path, pilot_id: str):
    """Create one new isolated folder only after supplied receipt passed checks."""
    if report.status=="REJECTED" or not report.file_hashes:
        raise ValueError("입력 검증 또는 원본 파일 확인이 끝나지 않았습니다.")
    if metadata.get("owner_authorized") is not True or metadata.get("allowed_use")!="RESEARCH_SHADOW":
        raise ValueError("자료 제공자의 연구 검토 허락이 필요합니다.")
    if metadata.get("publication_scope") not in {"NOT_ALLOWED","AGGREGATES_ONLY"}:
        raise ValueError("공개 범위를 확인해 주세요.")
    if not metadata.get("storage_until") or not metadata.get("consent_reference"):
        raise ValueError("보관 기한과 제공 동의 근거가 필요합니다.")
    try:
        date.fromisoformat(str(metadata["storage_until"]))
    except ValueError as exc:
        raise ValueError("보관 기한은 연-월-일 형식으로 입력해 주세요.") from exc
    if not ID.fullmatch(pilot_id):raise ValueError("검토 식별자를 확인해 주세요.")
    base=Path(root).resolve()
    # The caller explicitly chooses a research output root, never raw/canonical.
    if any(x.lower() in {"raw","canonical","golden","dqn","latest"} for x in base.parts):
        raise ValueError("원본·기준 데이터·모델 폴더에는 검토 기록을 저장할 수 없습니다.")
    base.mkdir(parents=True,exist_ok=True)
    target=base/pilot_id
    target.mkdir(exist_ok=False)
    return _append(target,"receipt",{"kind":"RECEIVED_VALIDATED_INPUT","pilot_id":pilot_id,
        "network_id":metadata["network_id"],"dataset_id":metadata["dataset_id"],
        "input_signature":signature({"hashes":report.file_hashes,"records":report.records}),
        "source_hashes":report.file_hashes,"validation":report.as_dict(),
        "consent":{"reference":metadata["consent_reference"],"storage_until":metadata["storage_until"],"publication_scope":metadata["publication_scope"]},
        "execution_authorized":False,"actual_outcome_observed":False})


def _receipt(folder):
    root=Path(folder).resolve(strict=True)
    record=json.loads((root/"receipt.json").read_text(encoding="utf-8"))
    if record.get("kind")!="RECEIVED_VALIDATED_INPUT":raise ValueError("확인된 입력 접수 기록이 없습니다.")
    return root,record


def record_shadow_decision(folder, *, decision_id, input_signature, actions, expected_net=None, currency=None):
    root,receipt=_receipt(folder)
    if input_signature!=receipt["input_signature"]:raise ValueError("계산에 사용한 입력 자료가 접수 기록과 다릅니다.")
    if not actions or len({a.get("action_id") for a in actions})!=len(actions):
        raise ValueError("서로 구분되는 계획 항목이 필요합니다.")
    for action in actions:
        if action.get("network_id")!=receipt["network_id"]:raise ValueError("다른 사업장의 계획은 연결할 수 없습니다.")
        if not action.get("action_id") or action.get("action") not in {"TRANSFER","NORMAL_SALE","DISCOUNT_SALE","MIXED"}:
            raise ValueError("계획 항목과 판매·이동 방식을 확인해 주세요.")
        if action.get("evidence_class") not in {"SCENARIO","PROXY","VERIFIED_INPUT"}:
            raise ValueError("계획의 자료 근거를 구분해 주세요.")
    if expected_net is not None and (isinstance(expected_net,bool) or not math.isfinite(float(expected_net)) or not currency):
        raise ValueError("예상 금액의 값과 통화를 확인해 주세요.")
    return _append(root,"decision_"+decision_id,{"kind":"SHADOW_DECISION","decision_id":decision_id,
        "input_signature":input_signature,"actions":actions,"expected_net":expected_net,"currency":currency,
        "executed":False,"actual_saving":None,"operational_approval":False})


def record_operator_review(folder, *, review_id, decision_id, reviewer_reference, decision_hash, accepted, reason):
    root,_=_receipt(folder)
    if not ID.fullmatch(decision_id):raise ValueError("계획 식별자를 확인해 주세요.")
    path=root/f"decision_{decision_id}.json"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=decision_hash:raise ValueError("검토한 계획과 현재 기록이 다릅니다.")
    if not reviewer_reference or not reason or not isinstance(accepted,bool):raise ValueError("검토자 근거·검토 의견·수용 여부를 확인해 주세요.")
    return _append(root,"review_"+review_id,{"kind":"OPERATOR_REVIEW","decision_id":decision_id,
        "reviewer_reference":reviewer_reference,"decision_sha256":decision_hash,"accepted_for_review":accepted,
        "reason":reason,"execution_authorized":False})


def record_observed_outcomes(folder, *, event_id, decision_id, outcome_report: IntakeReport):
    root,receipt=_receipt(folder)
    if not ID.fullmatch(decision_id):raise ValueError("계획 식별자를 확인해 주세요.")
    if outcome_report.status=="REJECTED" or not outcome_report.file_hashes:
        raise ValueError("실제 사후 기록의 입력 검증과 원본 확인이 필요합니다.")
    decision=json.loads((root/f"decision_{decision_id}.json").read_text(encoding="utf-8"))
    known={a["action_id"]:a for a in decision["actions"]}
    rows=outcome_report.records.get("action_outcome",[])
    if not rows:raise ValueError("관측된 사후 기록이 없습니다. 미관측 상태를 유지합니다.")
    for row in rows:
        if row.get("network_id")!=receipt["network_id"] or row.get("action_id") not in known:
            raise ValueError("사후 기록이 같은 사업장·계획 항목에 연결되지 않습니다.")
        if known[row["action_id"]]["evidence_class"]=="SCENARIO":
            raise ValueError("가상 계획을 실제 실행 실적으로 연결할 수 없습니다.")
        from services import operational_evidence as oe
        if row.get("outcome_kind")!="OBSERVED_ACTUAL" or oe.evidence_class(row.get("evidence_class"),row.get("verification_status")) not in {oe.DIRECT_REAL,oe.VERIFIED_OPERATOR_INPUT}:
            raise ValueError("확인된 실제 관측 기록만 사후 실적으로 받을 수 있습니다.")
        if row.get("operator_approved") not in (True,"true"):
            raise ValueError("실제 실행에 대한 운영자 승인 근거가 필요합니다.")
    required=("actual_revenue","actual_transport_spend","actual_handling_spend","actual_storage_spend","actual_disposal_spend")
    complete=all(all(row.get(k) is not None and str(row[k]).strip()!="" for k in required) for row in rows)
    currencies={row.get("currency") for row in rows}
    complete=complete and len(currencies)==1 and None not in currencies
    observed_net=sum(float(r[required[0]])-sum(float(r[k]) for k in required[1:]) for r in rows) if complete else None
    return _append(root,"outcome_"+event_id,{"kind":"OBSERVED_OUTCOME_RECEIPT","decision_id":decision_id,
        "rows":rows,"source_hashes":outcome_report.file_hashes,"observed_net_components":observed_net,
        "cash_scope":"recorded revenue less recorded listed operating spend, excludes unprovided COGS/taxes",
        "currency":next(iter(currencies)) if len(currencies)==1 else None,
        "actual_saving":None,"causal_effect":"NOT_IDENTIFIED_NO_COUNTERFACTUAL","production_approval":False})

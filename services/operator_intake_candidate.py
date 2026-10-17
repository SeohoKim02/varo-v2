"""Read-only operator intake for the CANDIDATE research path.

Reuse canonical fields and operational-evidence parsers.  No row is dropped,
imputed, merged or written to production.  Successful import is not permission
to execute a plan, nor evidence of an economic treatment effect.
"""
from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from services import canonical_schema as cs
from services import operational_evidence as oe

VERSION = "operator-intake-candidate-v0.1"
CANONICAL_TABLES = ("inventory_snapshot", "inventory_flow", "demand_series", "product_master", "location_master")
EVIDENCE_PARSERS = {
    "operational_route_evidence": oe.normalize_route_evidence,
    "operational_demand_evidence": oe.normalize_demand_evidence,
    "observed_transfers": oe.normalize_observed_transfers,
}
OUTCOME_FIELDS = (
    "action_id", "network_id", "location_id", "product_id", "product_state", "lot_id",
    "action", "action_date", "observation_start", "observation_end", "quantity_unit",
    "executed_qty", "sales_qty", "waste_qty", "closing_stock_qty", "actual_revenue",
    "actual_transport_spend", "actual_handling_spend", "actual_storage_spend",
    "actual_disposal_spend", "currency", "evidence_class", "verification_status",
    "evidence_source", "outcome_kind", "operator_approved", *cs.LINEAGE,
)
OUTCOME_NUMERIC = (
    "executed_qty", "sales_qty", "waste_qty", "closing_stock_qty", "actual_revenue",
    "actual_transport_spend", "actual_handling_spend", "actual_storage_spend", "actual_disposal_spend",
)
ALLOWED_TABLES = (*CANONICAL_TABLES, *EVIDENCE_PARSERS, "action_outcome")


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and str(value).strip() != ""


def _date(value: Any):
    if not _present(value):
        return None
    try:
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.date()
    except (ValueError, TypeError, OverflowError):
        return None


@dataclass
class IntakeReport:
    records: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    file_hashes: dict[str, str] = field(default_factory=dict)
    version: str = VERSION
    production_approval: bool = False
    effect_claim_allowed: bool = False

    @property
    def status(self) -> str:
        if any(i["severity"] == "ERROR" for i in self.issues):
            return "REJECTED"
        return "REVIEW_REQUIRED" if self.issues else "VALID_FOR_SHADOW_REVIEW"

    def as_dict(self) -> dict[str, Any]:
        return {"version": self.version, "status": self.status, "issues": self.issues,
                "table_row_counts": {k: len(v) for k, v in self.records.items()},
                "file_hashes": self.file_hashes, "production_approval": False,
                "effect_claim_allowed": False}


def _issue(report, table, row, code, message, fields=(), severity="ERROR"):
    report.issues.append({"table": table, "row": row, "code": code, "severity": severity,
                          "fields": list(fields), "message_ko": message})


def _required(report, table, rownum, row, fields):
    missing = [f for f in fields if not _present(row.get(f))]
    if missing:
        _issue(report, table, rownum, "MISSING_REQUIRED", "필수 정보가 비어 있습니다. 원본 자료에서 확인해 주세요.", missing)


def validate_operator_intake(tables: Mapping[str, pd.DataFrame], metadata: Mapping[str, Any]) -> IntakeReport:
    """Validate canonical/operator tables without changing or automatically applying them.

    metadata: dataset_id, network_id, as_of_date, owner_authorized (bool),
    allowed_use='RESEARCH_SHADOW', publication_scope='AGGREGATES_ONLY' or
    'NOT_ALLOWED', source_hashes={source_file:sha256}.  Hash declarations here
    are not physical verification; use import_operator_csv_bundle for that.
    """
    report = IntakeReport()
    _required(report, "metadata", None, metadata,
              ("dataset_id", "network_id", "as_of_date", "allowed_use", "publication_scope"))
    if metadata.get("owner_authorized") is not True:
        _issue(report, "metadata", None, "OWNER_APPROVAL_REQUIRED", "자료 제공자의 연구 이용 허락을 먼저 확인해 주세요.")
    if metadata.get("allowed_use") != "RESEARCH_SHADOW":
        _issue(report, "metadata", None, "USE_SCOPE_INVALID", "이 입력 경로는 실행 없이 비교하는 연구 검토용입니다.")
    if metadata.get("publication_scope") not in {"AGGREGATES_ONLY", "NOT_ALLOWED"}:
        _issue(report, "metadata", None, "PUBLICATION_SCOPE_REQUIRED", "공개 가능 범위를 집계 자료 또는 비공개로 확인해 주세요.")
    as_of = _date(metadata.get("as_of_date"))
    if as_of is None:
        _issue(report, "metadata", None, "INVALID_AS_OF_DATE", "자료 확인 기준일을 올바른 날짜로 입력해 주세요.")
    source_hashes = metadata.get("source_hashes", {})
    if not isinstance(source_hashes, Mapping):
        source_hashes = {}
    if not tables or all(isinstance(f, pd.DataFrame) and len(f) == 0 for f in tables.values()):
        _issue(report, "metadata", None, "NO_OBSERVATIONS", "읽을 기록이 없습니다. 빈 양식은 관측 자료로 사용할 수 없습니다.")
    for table, frame in tables.items():
        if table not in ALLOWED_TABLES:
            _issue(report, table, None, "UNKNOWN_TABLE", "지원하지 않는 자료 유형입니다. 지정된 입력 양식을 확인해 주세요.")
            continue
        if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
            _issue(report, table, None, "INVALID_TABLE", "표 형식 또는 중복된 열 이름을 확인해 주세요.")
            continue
        rows = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
        report.records[table] = rows
        for index, row in enumerate(rows):
            _required(report, table, index, row, cs.LINEAGE)
            if row.get("source_dataset") != metadata.get("dataset_id"):
                _issue(report, table, index, "DATASET_MISMATCH", "자료 출처가 입력 묶음과 다릅니다. 서로 다른 사업장 자료를 섞지 마세요.")
            source = row.get("source_file")
            digest = source_hashes.get(source, "")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
                _issue(report, table, index, "SOURCE_HASH_REQUIRED", "원본 파일 확인 정보가 없습니다. 원본 파일과 출처 목록을 함께 제공해 주세요.")
            if oe.is_sample_row(row):
                _issue(report, table, index, "SAMPLE_NOT_OBSERVATION", "예시 행은 실제 기록으로 사용할 수 없습니다.")
            if table in CANONICAL_TABLES:
                _required(report, table, index, row, (*cs.REQUIRED[table], "date_grain", "scope"))
                for raw_id, namespace, key in cs.IDENTITY_KEYS:
                    if _present(row.get(raw_id)):
                        expected_namespace = f"{metadata.get('network_id')}:{raw_id}"
                        if row.get(namespace) != expected_namespace or row.get(key) != f"{expected_namespace}:{row[raw_id]}":
                            _issue(report, table, index, "IDENTITY_SCOPE_MISMATCH", "점포·상품 식별 범위가 맞지 않습니다. 같은 사업장의 원본 식별자를 확인해 주세요.", (raw_id, namespace, key))
                for quantity in cs.NUMERIC.intersection(row):
                    _numeric(report, table, index, quantity, row[quantity], quantity in cs.NONNEGATIVE)
                if table not in cs.NO_QUANTITY_UNIT and (not _present(row.get("unit")) or row.get("unit_status") not in {"DIRECT", "SOURCE_METADATA", "DERIVED"}):
                    _issue(report, table, index, "QUANTITY_UNIT_UNKNOWN", "수량 단위가 확인되지 않았습니다. 개·상자·킬로그램 등을 원본 근거와 함께 확인해 주세요.")
                when = _date(row.get("snapshot_date") if table == "inventory_snapshot" else row.get("date"))
                if table in {"inventory_snapshot", "inventory_flow", "demand_series"}:
                    if when is None or (as_of is not None and when > as_of):
                        _issue(report, table, index, "INVALID_OBSERVATION_DATE", "기록 날짜가 없거나 확인 기준일보다 미래입니다.")
                if _present(row.get("validation_flags")) and any(cs.is_blocking(f) for f in str(row["validation_flags"]).split("|")):
                    _issue(report, table, index, "SOURCE_QUALITY_BLOCKED", "기존 자료 검증에서 확인된 문제가 있습니다. 원본의 날짜·수량·중복 여부를 해결해 주세요.")
            elif table in EVIDENCE_PARSERS:
                _, issues = EVIDENCE_PARSERS[table]([row], network_id=str(metadata.get("network_id", "")))
                for issue in issues:
                    _issue(report, table, index, issue["code"], "운영 근거의 범위·기간·단위 또는 확인 상태를 확인해 주세요.")
                for name in ("quantity", "route_capacity_qty", "cost_value"):
                    _numeric(report, table, index, name, row.get(name))
                if table == "operational_demand_evidence" and row.get("quantity_kind") == oe.ACTUAL_SALES:
                    end = _date(row.get("observation_end") or row.get("observation_start"))
                    if as_of is not None and end is not None and end > as_of:
                        _issue(report, table, index, "FUTURE_SALES", "아직 지나지 않은 기간을 실제 판매 기록으로 사용할 수 없습니다.")
                if "evidence_class" in row and oe.evidence_class(row.get("evidence_class"), row.get("verification_status")) not in oe.ACCEPTED_EVIDENCE:
                    _issue(report, table, index, "UNVERIFIED_EVIDENCE", "추정값은 보존되지만 운영자가 확인한 실제 자료와 구분해야 합니다.", severity="WARNING")
            else:
                _outcome(report, index, row, metadata, as_of)
        keys = cs.KEYS.get(table)
        if table == "operational_route_evidence":
            keys = ("network_id", "source_id", "target_id", "product_id", "product_group", "effective_from", "effective_to")
        elif table == "operational_demand_evidence":
            keys = ("network_id", "location_id", "product_id", "quantity_kind", "observation_start", "observation_end", "quantity_unit")
        elif table == "observed_transfers":
            keys = ("network_id", "source_id", "target_id", "product_id", "transfer_date", "evidence_source")
        if table == "action_outcome":
            keys = ("network_id", "action_id", "location_id", "product_id", "product_state", "lot_id", "observation_start", "observation_end")
        if keys:
            # Keep storage state and lot distinct, including for legacy schemas without lot.
            keys = tuple(dict.fromkeys((*keys, "lot_id")))
            seen = {}
            for index, row in enumerate(rows):
                key = tuple(row.get(k) for k in keys)
                if key in seen:
                    _issue(report, table, index, "DUPLICATE_OR_CONFLICT", "같은 식별 범위의 기록이 중복됩니다. 합산하지 않았으니 원본을 확인해 주세요.")
                seen[key] = index
    return report


def _numeric(report, table, index, name, value, nonnegative=True):
    if not _present(value):
        return
    try:
        number = float(value)
        valid = not isinstance(value, bool) and math.isfinite(number) and (not nonnegative or number >= 0)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        _issue(report, table, index, "INVALID_NUMBER", "수량·금액은 유효한 숫자로 입력해 주세요. 음수 조정은 별도 근거가 필요합니다.", (name,))


def _outcome(report, index, row, metadata, as_of):
    _required(report, "action_outcome", index, row,
              ("action_id", "network_id", "location_id", "product_id", "action", "action_date",
               "observation_start", "observation_end", "quantity_unit", "evidence_source", "outcome_kind"))
    if row.get("network_id") != metadata.get("network_id"):
        _issue(report, "action_outcome", index, "CROSS_NETWORK_OUTCOME", "다른 사업장의 실행 결과를 연결할 수 없습니다.")
    if row.get("outcome_kind") != "OBSERVED_ACTUAL" or oe.evidence_class(row.get("evidence_class"), row.get("verification_status")) not in {oe.DIRECT_REAL, oe.VERIFIED_OPERATOR_INPUT}:
        _issue(report, "action_outcome", index, "NOT_OBSERVED_OUTCOME", "예상·시뮬레이션 결과는 실제 실행 결과 입력란에 넣을 수 없습니다.")
    if row.get("action") not in {"TRANSFER", "NORMAL_SALE", "DISCOUNT_SALE", "MIXED"}:
        _issue(report, "action_outcome", index, "UNKNOWN_ACTION", "실제로 수행한 행동을 확인해 주세요.")
    if row.get("operator_approved") is not True and row.get("operator_approved") != "true":
        _issue(report, "action_outcome", index, "ACTION_APPROVAL_REQUIRED", "해당 실행에 대한 운영자 승인 기록이 필요합니다.")
    action, start, end = (_date(row.get(f)) for f in ("action_date", "observation_start", "observation_end"))
    if None in (action, start, end) or (action and start and end and not action <= start <= end) or (end and as_of and end > as_of):
        _issue(report, "action_outcome", index, "INVALID_OUTCOME_WINDOW", "실행일과 관측 기간을 확인해 주세요. 미관측 기간은 실제 결과로 표시하지 않습니다.")
    for name in OUTCOME_NUMERIC:
        _numeric(report, "action_outcome", index, name, row.get(name))
    money = [f for f in OUTCOME_NUMERIC if f.startswith("actual_")]
    if any(_present(row.get(f)) for f in money) and not _present(row.get("currency")):
        _issue(report, "action_outcome", index, "CURRENCY_REQUIRED", "실제 금액의 통화를 확인해 주세요.")
    if not any(_present(row.get(f)) for f in OUTCOME_NUMERIC):
        _issue(report, "action_outcome", index, "OUTCOME_NOT_MEASURED", "관측된 수량·금액이 없습니다. 사후 기록이 도착할 때까지 성과 미관측으로 유지합니다.")
    if any(not _present(row.get(f)) for f in OUTCOME_NUMERIC):
        _issue(report, "action_outcome", index, "PARTIAL_OUTCOME", "일부 사후 기록이 없습니다. 비어 있는 항목은 0원·0개로 계산하지 않습니다.", severity="WARNING")


def import_operator_csv_bundle(table_paths: Mapping[str, str | Path], metadata: Mapping[str, Any], *,
                               root: str | Path, max_file_bytes: int = 16 * 1024 * 1024) -> IntakeReport:
    """Bounded UTF-8 CSV import.  Read only; original-byte hashes must match approval manifest."""
    base = Path(root).resolve(strict=True)
    frames, hashes = {}, {}
    for table, value in table_paths.items():
        path = (base / value).resolve(strict=True)
        if not path.is_relative_to(base) or not path.is_file() or path.suffix.lower() != ".csv":
            raise ValueError("입력 폴더 안의 CSV 파일만 읽을 수 있습니다.")
        if path.stat().st_size > max_file_bytes:
            raise ValueError("입력 파일이 크기 제한을 넘습니다. 기간을 나누어 제공해 주세요.")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        source = path.relative_to(base).as_posix()
        if metadata.get("source_hashes", {}).get(source, "").lower() != digest:
            raise ValueError("원본 확인 정보와 파일 내용이 다릅니다. 변경된 자료의 승인을 다시 확인해 주세요.")
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("문자 인코딩을 읽을 수 없습니다. UTF-8 CSV로 저장해 다시 제공해 주세요.") from exc
        # Check duplicate headers before pandas automatically renames them.
        import csv
        header = next(csv.reader(io.StringIO(text)), [])
        if not header or len(header) != len(set(header)):
            raise ValueError("열 이름이 비어 있거나 중복되어 있습니다.")
        try:
            frame = pd.read_csv(io.StringIO(text), dtype=object, keep_default_na=False, na_filter=False)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError("표의 행과 열 구성이 맞지 않습니다. CSV 구분자와 따옴표를 확인해 주세요.") from exc
        if "source_file" not in frame or not frame["source_file"].eq(source).all():
            raise ValueError("각 행의 원본 파일명이 실제 입력 파일과 다릅니다.")
        frames[table] = frame
        hashes[source] = digest
    report = validate_operator_intake(frames, metadata)
    report.file_hashes = hashes
    return report

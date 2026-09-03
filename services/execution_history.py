"""Persistent execution-history service for shadow operations.

The UI never issues SQL directly.  This module snapshots a validated execution
plan, records what operators actually did, and exports the joined expected vs
actual facts for later offline calibration.  It does not feed any value back to
VHS, DQN, or the execution-plan optimizer.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.execution_history_config import (
    DEFAULT_HISTORY_DB,
    HISTORY_DB_PATH_ENV,
    HistoryConfigurationError,
    load_execution_history_config,
)
from services.execution_history_store import (
    SCHEMA_VERSION,
    DuplicatePlanError,
    HistoryItemNotFoundError,
    HistoryStoreError,
    build_execution_history_store,
)


STATUS_LABELS = {
    "unconfirmed": "미확인",
    "executed": "실행",
    "partial": "일부 실행",
    "not_executed": "미실행",
    "cancelled": "취소",
}
STATUS_CODES_BY_LABEL = {label: code for code, label in STATUS_LABELS.items()}

REASON_LABELS = {
    "inventory_shortage": "재고 부족",
    "field_decision": "현장 판단",
    "transport_unavailable": "운송 불가",
    "time_shortage": "시간 부족",
    "store_request": "점포 요청",
    "other": "기타",
}
REASON_CODES_BY_LABEL = {label: code for code, label in REASON_LABELS.items()}

_QUANTITY_OUTCOME_FIELDS = {
    "post_source_stock",
    "post_destination_stock",
    "actual_sales_qty",
    "actual_waste_qty",
    "actual_stockout_qty",
}
_MONEY_OUTCOME_FIELDS = {"actual_transport_cost", "actual_saving"}
_OUTCOME_FIELDS = _QUANTITY_OUTCOME_FIELDS | _MONEY_OUTCOME_FIELDS
_FEATURE_FIELDS = (
    "net_benefit_score", "disposal_risk_score", "demand_fit_score",
    "inventory_balance_score", "route_cost_score", "feasibility_score",
    "demand_risk_score", "post_move_risk_score", "pareto_rank",
    "pareto_status", "robustness_status", "confidence_score",
)
_MAX_ACTUAL_QTY = 1_000_000_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def history_db_path(db_path: str | Path | None = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    configured = str(os.environ.get(HISTORY_DB_PATH_ENV) or "").strip()
    return Path(configured) if configured else DEFAULT_HISTORY_DB


def execution_history_backend_info(db_path: str | Path | None = None) -> dict[str, Any]:
    """Return only a user-safe backend label; never connection details."""
    try:
        config = load_execution_history_config(db_path)
        return _result(True, "configured", "실행 이력 저장소가 설정되었습니다.", backend=config.backend, label=config.user_label)
    except HistoryConfigurationError:
        return _result(False, "invalid_config", "실행 기록 저장 설정을 확인해주세요.", backend=None, label="설정 확인 필요")


def _result(ok: bool, code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": ok, "code": code, "message": message, **extra}


def _number(value: Any, *, integer: bool = False, allow_none: bool = True) -> int | float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_none:
            return None
        raise ValueError("값이 필요합니다.")
    if isinstance(value, bool):
        raise ValueError("숫자 형식이 아닙니다.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("숫자 형식이 아닙니다.") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError("0 이상의 유한한 값만 입력할 수 있습니다.")
    if integer:
        if not number.is_integer():
            raise ValueError("수량은 정수로 입력해주세요.")
        return int(number)
    return number


def _text(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned[:limit] if cleaned else None


def _optional_bool(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "있음"}:
        return 1
    if normalized in {"0", "false", "no", "n", "없음"}:
        return 0
    raise ValueError("품절 여부 값을 확인해주세요.")


def initialize_history_store(db_path: str | Path | None = None) -> dict[str, Any]:
    try:
        build_execution_history_store(db_path).initialize()
        return _result(True, "ready", "실행 기록 저장소가 준비되었습니다.", schema_version=SCHEMA_VERSION)
    except (HistoryConfigurationError, HistoryStoreError, OSError):
        return _result(False, "storage_error", "실행 기록 저장소를 준비하지 못했습니다.")


def _plan_snapshot(plan: Mapping[str, Any], recorded_at: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_id = _text(plan.get("plan_id"), 160)
    data_signature = _text(plan.get("data_signature"), 256)
    algorithm_version = _text(plan.get("algorithm_version"), 120)
    created_at = _text(plan.get("created_at"), 80)
    raw_items = list(plan.get("items") or [])
    if not plan_id or not data_signature or not algorithm_version or not created_at or not raw_items:
        raise ValueError("기록할 수 있는 실행계획이 없습니다.")
    if not bool((plan.get("validation") or {}).get("valid")):
        raise ValueError("검증을 통과한 실행계획만 기록할 수 있습니다.")

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            raise ValueError("실행계획 항목 형식을 확인해주세요.")
        candidate_id = _text(raw.get("candidate_id"), 160)
        source_id = _text(raw.get("source_id"), 160)
        target_id = _text(raw.get("target_id"), 160)
        product_id = _text(raw.get("product_id"), 160)
        route_type = _text(raw.get("route_type"), 40)
        planned_qty = _number(raw.get("planned_qty"), integer=True, allow_none=False)
        if not all((candidate_id, source_id, target_id, product_id, route_type)) or planned_qty is None or planned_qty <= 0:
            raise ValueError("실행계획 항목의 필수 값이 없습니다.")
        if candidate_id in seen:
            raise ValueError("중복 실행계획 항목이 있습니다.")
        seen.add(candidate_id)
        features = {key: raw.get(key) for key in _FEATURE_FIELDS if raw.get(key) is not None}
        items.append({
            "plan_id": plan_id,
            "candidate_id": candidate_id,
            "candidate_algorithm_version": _text(raw.get("algorithm_version"), 120),
            "source_store_id": source_id,
            "source_store_name": _text(raw.get("source_name"), 200),
            "destination_store_id": target_id,
            "destination_store_name": _text(raw.get("target_name"), 200),
            "product_id": product_id,
            "product_name": _text(raw.get("product_name"), 200),
            "route_type": route_type,
            "dc_id": _text(raw.get("dc_id"), 160),
            "planned_qty": planned_qty,
            "expected_cost": _number(raw.get("planned_cost")),
            "expected_saving": _number(raw.get("planned_expected_saving")),
            "expected_net_benefit": _number(raw.get("planned_net_benefit")),
            "vhs_score": _number(raw.get("vhs_score")),
            "stability": _text(raw.get("robustness_status") or raw.get("stability"), 80),
            "confidence": _number(raw.get("confidence_score") or raw.get("confidence")),
            "feature_snapshot_json": json.dumps(
                features, ensure_ascii=False, sort_keys=True,
                default=lambda value: value.item() if hasattr(value, "item") else str(value),
            ),
            "updated_at": recorded_at,
        })

    candidate_version = next(
        (item["candidate_algorithm_version"] for item in items if item["candidate_algorithm_version"]),
        None,
    )
    snapshot = {
        "plan_id": plan_id,
        "algorithm_version": algorithm_version,
        "candidate_algorithm_version": candidate_version,
        "data_signature": data_signature,
        "created_at": created_at,
        "recorded_at": recorded_at,
        "updated_at": recorded_at,
        "plan_status": _text(plan.get("plan_status"), 80) or "계산 불가",
        "total_actions": len(items),
        "total_planned_qty": sum(int(item["planned_qty"]) for item in items),
        "expected_total_cost": _number(plan.get("total_cost")),
        "expected_total_saving": _number(plan.get("total_expected_saving")),
        "expected_total_net_benefit": _number(plan.get("total_net_benefit")),
    }
    return snapshot, items


def record_execution_plan(
    plan: Mapping[str, Any], db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically store a plan and its items once."""
    recorded_at = _utc_now()
    try:
        snapshot, items = _plan_snapshot(plan, recorded_at)
    except (ValueError, TypeError, AttributeError) as exc:
        return _result(False, "invalid_plan", str(exc))

    try:
        build_execution_history_store(db_path).save_plan(snapshot, items)
        return _result(
            True, "recorded", "실행계획을 기록했습니다.",
            plan_id=snapshot["plan_id"], created=True,
        )
    except DuplicatePlanError:
        return _result(
            True, "duplicate", "이미 기록된 실행계획입니다.",
            plan_id=snapshot["plan_id"], created=False,
        )
    except (HistoryConfigurationError, HistoryStoreError, OSError):
        return _result(False, "storage_error", "실행 기록을 저장하지 못했습니다.")


def list_recorded_plans(
    db_path: str | Path | None = None, *, limit: int = 50, offset: int = 0,
) -> dict[str, Any]:
    try:
        safe_limit = max(1, min(int(limit), 100_000))
        safe_offset = max(0, int(offset))
        rows = build_execution_history_store(db_path).list_plans(limit=safe_limit, offset=safe_offset)
        message = "실행 기록을 불러왔습니다." if rows else "기록된 실행계획이 없습니다."
        return _result(True, "loaded", message, plans=rows)
    except (HistoryConfigurationError, HistoryStoreError, OSError, ValueError, TypeError):
        return _result(False, "storage_error", "실행 기록을 불러오지 못했습니다.", plans=[])


def get_recorded_plan(plan_id: str, db_path: str | Path | None = None) -> dict[str, Any]:
    try:
        plan, raw_items = build_execution_history_store(db_path).get_plan(str(plan_id))
        if plan is None:
            return _result(False, "not_found", "기록된 실행계획을 찾지 못했습니다.", plan=None, items=[])
        items = [_item_with_comparison(item) for item in raw_items]
        return _result(True, "loaded", "실행계획을 불러왔습니다.", plan=plan, items=items)
    except (HistoryConfigurationError, HistoryStoreError, OSError):
        return _result(False, "storage_error", "실행 기록을 불러오지 못했습니다.", plan=None, items=[])


def _item_with_comparison(item: dict[str, Any]) -> dict[str, Any]:
    item["execution_status_label"] = STATUS_LABELS.get(item.get("execution_status"), "확인 필요")
    item["nonexecution_reason_label"] = REASON_LABELS.get(item.get("nonexecution_reason"), "-")
    item["quantity_difference"] = (
        None if item.get("actual_qty") is None else item["actual_qty"] - item["planned_qty"]
    )
    item["cost_difference"] = (
        None if item.get("actual_transport_cost") is None or item.get("expected_cost") is None
        else item["actual_transport_cost"] - item["expected_cost"]
    )
    item["saving_difference"] = (
        None if item.get("actual_saving") is None or item.get("expected_saving") is None
        else item["actual_saving"] - item["expected_saving"]
    )
    item["net_benefit_difference"] = (
        None if item.get("actual_net_benefit") is None or item.get("expected_net_benefit") is None
        else item["actual_net_benefit"] - item["expected_net_benefit"]
    )
    return item


def update_execution_item(
    plan_id: str,
    candidate_id: str,
    execution_status: str,
    actual_qty: Any = None,
    *,
    nonexecution_reason: str | None = None,
    operator_note: str | None = None,
    outcomes: Mapping[str, Any] | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Update one recorded action and append a minimal audit event."""
    status = STATUS_CODES_BY_LABEL.get(execution_status, execution_status)
    if status not in STATUS_LABELS:
        return _result(False, "invalid_status", "실행 상태를 확인해주세요.")
    reason = REASON_CODES_BY_LABEL.get(nonexecution_reason or "", nonexecution_reason)
    if reason and reason not in REASON_LABELS:
        return _result(False, "invalid_reason", "미실행 사유를 확인해주세요.")
    if reason not in REASON_LABELS:
        reason = None
    outcomes_provided = outcomes is not None
    if outcomes is not None and not isinstance(outcomes, Mapping):
        return _result(False, "invalid_value", "사후 결과 형식을 확인해주세요.")
    try:
        quantity = _number(actual_qty, integer=True)
        if quantity is not None and quantity > _MAX_ACTUAL_QTY:
            raise ValueError("실제 수량 값이 너무 큽니다.")
        if status in {"executed", "partial"} and (quantity is None or quantity <= 0):
            raise ValueError("실행한 수량을 1개 이상의 정수로 입력해주세요.")
        if status in {"not_executed", "cancelled"}:
            if quantity not in (None, 0):
                raise ValueError("미실행 또는 취소 상태의 실제 수량은 0이어야 합니다.")
            quantity = 0
        if status == "unconfirmed":
            quantity = None
        clean_outcomes = {
            key: _number((outcomes or {}).get(key))
            for key in _OUTCOME_FIELDS
        }
        stockout_occurred = _optional_bool((outcomes or {}).get("actual_stockout_occurred"))
    except ValueError as exc:
        return _result(False, "invalid_value", str(exc))

    now = _utc_now()
    try:
        planned_qty = build_execution_history_store(db_path).update_execution_result(
            str(plan_id),
            str(candidate_id),
            status=status,
            quantity=quantity,
            reason=reason,
            operator_note=_text(operator_note, 300),
            outcomes=clean_outcomes,
            outcomes_provided=outcomes_provided,
            stockout_occurred=stockout_occurred,
            now=now,
        )
        warning = None
        if quantity is not None and quantity > planned_qty:
            warning = f"계획보다 {quantity - planned_qty}개 많이 실행되었습니다."
        return _result(True, "updated", "실행 기록을 저장했습니다.", warning=warning)
    except HistoryItemNotFoundError:
        return _result(False, "not_found", "기록된 이동을 찾지 못했습니다.")
    except (HistoryConfigurationError, HistoryStoreError, OSError):
        return _result(False, "storage_error", "실행 기록을 저장하지 못했습니다.")


def execution_history_metrics(db_path: str | Path | None = None) -> dict[str, Any]:
    try:
        summary = build_execution_history_store(db_path).metrics_summary()
    except (HistoryConfigurationError, HistoryStoreError, OSError):
        return _result(False, "storage_error", "실행 기록 요약을 계산하지 못했습니다.")

    def rate(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else round(numerator / denominator * 100.0, 2)

    def normalized_error(name: str) -> dict[str, Any]:
        raw = summary[name]
        count = int(raw.get("sample_count") or 0)
        return {
            "sample_count": count,
            "mean_error": None if not count else round(float(raw["mean_error"]), 6),
            "mean_absolute_error": None if not count else round(float(raw["mean_absolute_error"]), 6),
        }

    confirmed = int(summary.get("confirmed_items") or 0)
    planned_total = int(summary.get("planned_qty_sample_total") or 0)
    actual_total = int(summary.get("actual_qty_total") or 0)
    return _result(
        True, "calculated", "실행 기록 요약을 계산했습니다.",
        total_items=int(summary.get("total_items") or 0), confirmed_items=confirmed,
        execution_rate=rate(int(summary.get("executed_items") or 0), confirmed),
        partial_rate=rate(int(summary.get("partial_items") or 0), confirmed),
        not_executed_rate=rate(int(summary.get("not_executed_items") or 0), confirmed),
        quantity_adherence_rate=None if not planned_total else round(actual_total / planned_total * 100.0, 2),
        quantity_sample_count=int(summary.get("quantity_sample_count") or 0),
        cost_error=normalized_error("cost_error"),
        saving_error=normalized_error("saving_error"),
        net_benefit_error=normalized_error("net_benefit_error"),
    )


def export_execution_history_csv(db_path: str | Path | None = None) -> dict[str, Any]:
    """Return a user-triggered UTF-8 BOM calibration export; never includes DB paths."""
    try:
        stored_rows = build_execution_history_store(db_path).export_rows()
    except (HistoryConfigurationError, HistoryStoreError, OSError):
        return _result(False, "storage_error", "실행 기록을 내보내지 못했습니다.", data=b"")
    rows: list[dict[str, Any]] = []
    for item in stored_rows:
        try:
            features = json.loads(item.get("feature_snapshot_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            features = {}
        rows.append({
            "plan_id": item["plan_id"], "candidate_id": item["candidate_id"],
            "plan_algorithm_version": item["plan_algorithm_version"],
            "candidate_algorithm_version": item.get("candidate_algorithm_version") or item.get("plan_candidate_algorithm_version"),
            "data_signature": item["data_signature"], "plan_created_at": item["plan_created_at"],
            "recorded_at": item["recorded_at"], "source_store": item.get("source_store_name") or item["source_store_id"],
            "destination_store": item.get("destination_store_name") or item["destination_store_id"],
            "product_id": item["product_id"], "product": item.get("product_name"),
            "route_type": item["route_type"], "dc_id": item.get("dc_id"),
            "planned_qty": item["planned_qty"], "actual_qty": item.get("actual_qty"),
            "execution_status": STATUS_LABELS.get(item.get("execution_status"), "확인 필요"),
            "nonexecution_reason": REASON_LABELS.get(item.get("nonexecution_reason"), "-"),
            "expected_cost": item.get("expected_cost"), "actual_transport_cost": item.get("actual_transport_cost"),
            "expected_saving": item.get("expected_saving"), "actual_saving": item.get("actual_saving"),
            "expected_net_benefit": item.get("expected_net_benefit"), "actual_net_benefit": item.get("actual_net_benefit"),
            "post_source_stock": item.get("post_source_stock"), "post_destination_stock": item.get("post_destination_stock"),
            "actual_sales_qty": item.get("actual_sales_qty"), "actual_waste_qty": item.get("actual_waste_qty"),
            "actual_stockout_occurred": (
                "" if item.get("actual_stockout_occurred") is None
                else "있음" if item.get("actual_stockout_occurred") else "없음"
            ),
            "actual_stockout_qty": item.get("actual_stockout_qty"), "vhs_score": item.get("vhs_score"),
            "stability": item.get("stability"), "confidence": item.get("confidence"),
            **{key: features.get(key) for key in _FEATURE_FIELDS},
        })
    if not rows:
        return _result(True, "empty", "내보낼 실행 기록이 없습니다.", data=b"", row_count=0)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return _result(
        True, "exported", "실행 기록을 내보냈습니다.",
        data=stream.getvalue().encode("utf-8-sig"), row_count=len(rows),
    )


def list_item_events(
    plan_id: str, candidate_id: str, db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Small audit hook used by validation/tests; not exposed in the basic UI."""
    try:
        return build_execution_history_store(db_path).list_events(str(plan_id), str(candidate_id))
    except (HistoryConfigurationError, HistoryStoreError, OSError):
        return []

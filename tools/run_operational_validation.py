"""End-to-end operational validation of Varo V2 on the anonymized workbook.

This script does not re-implement anything the app does. It drives the *real*
services in the same order the UI does — read file → inspect into pending →
partial exclusion → user apply → user-triggered recommendation run → page
view-models — and compares what comes out against the manifest written by
``tools/generate_anonymized_operational_workbook.py``.

    python tools/generate_anonymized_operational_workbook.py
    python tools/run_operational_validation.py

Exit code is 0 only when every check passes. A JSON report with all measured
numbers and timings is written next to the workbook.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.analysis_pipeline import find_recommendation, sort_recommendations  # noqa: E402
from services.app_state import has_applied_data  # noqa: E402
from services.data_application import (  # noqa: E402
    cancel_pending_data, commit_pending_data, prepare_pending_data, run_applied_analysis,
)
from services.data_management_view import build_data_management_view  # noqa: E402
from services.file_reader import read_uploaded_data  # noqa: E402
from services.home_state import build_home_state  # noqa: E402
from tools.generate_anonymized_operational_workbook import (  # noqa: E402
    MANIFEST_NAME, OUTPUT_DIR, WORKBOOK_NAME, generate,
)

REPORT_NAME = "varo_v2_anonymized_operational_report.json"
SOURCE_TYPE = "업로드 데이터"
# Every human-readable label in the workbook must be a synthetic placeholder.
ANONYMOUS_NAME = re.compile(r"^(가상점포|가상물류센터|가상상품|가상권역)\s\S+$")
CONTACT_LIKE = re.compile(r"\d{2,4}-\d{3,4}-\d{4}|@|https?://")


def _names(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return [str(value).strip() for value in frame[column].dropna() if str(value).strip()]


class Checks:
    """Collects pass/fail results without ever aborting the run early."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def check(self, section: str, name: str, ok: bool, detail: Any = "") -> bool:
        self.results.append({
            "section": section, "check": name, "ok": bool(ok), "detail": _plain(detail),
        })
        return bool(ok)

    def equal(self, section: str, name: str, actual: Any, expected: Any) -> bool:
        return self.check(
            section, name, actual == expected, {"actual": actual, "expected": expected},
        )

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [item for item in self.results if not item["ok"]]


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in sorted(value, key=str)] if isinstance(value, set) else [
            _plain(item) for item in value
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _timed(function: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def _text_set(frame: Any, column: str) -> set[str]:
    if not isinstance(frame, pd.DataFrame) or column not in frame.columns:
        return set()
    return {str(value).strip() for value in frame[column].dropna() if str(value).strip()}


def _refs(items: Any) -> set[tuple[str, int]]:
    return {
        (str(item.get("source_sheet")), int(item.get("source_row_number")))
        for item in items or []
    }


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def check_generation(checks: Checks, output_dir: Path, regenerate: bool) -> dict[str, Any]:
    workbook = output_dir / WORKBOOK_NAME
    manifest_path = output_dir / MANIFEST_NAME
    if regenerate or not workbook.exists() or not manifest_path.exists():
        (_paths, seconds) = _timed(lambda: generate(output_dir))
    else:
        seconds = 0.0
    checks.check("생성", "워크북 파일 생성", workbook.is_file(), str(workbook))
    checks.check("생성", "기대값 manifest 생성", manifest_path.is_file(), str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks.check(
        "생성", "익명화 선언 (개인정보·실제 업체 정보 없음)",
        not manifest["anonymization"]["contains_personal_data"]
        and not manifest["anonymization"]["contains_real_company_data"],
    )
    return {"manifest": manifest, "workbook": workbook, "generate_seconds": seconds}


def check_reproducible(checks: Checks, output_dir: Path, temp_dir: Path) -> None:
    """Same seed must produce the same sheets and the same manifest."""
    generate(temp_dir)
    original = pd.read_excel(output_dir / WORKBOOK_NAME, sheet_name=None)
    repeat = pd.read_excel(temp_dir / WORKBOOK_NAME, sheet_name=None)
    checks.equal("생성", "재생성 시 시트 구성 동일", sorted(repeat), sorted(original))
    identical = all(
        original[sheet].equals(repeat[sheet]) for sheet in original if sheet in repeat
    )
    checks.check("생성", "재생성 시 모든 시트 내용 동일", identical)
    checks.check(
        "생성", "재생성 시 manifest 동일",
        (output_dir / MANIFEST_NAME).read_text(encoding="utf-8")
        == (temp_dir / MANIFEST_NAME).read_text(encoding="utf-8"),
    )


def check_schema(checks: Checks, workbook: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    data, report = read_uploaded_data(workbook, workbook.name, return_report=True)
    raw = report["raw_sheets"]
    checks.check(
        "스키마", "필수 시트 인식", {"stores", "products", "inventory", "routes"}.issubset(raw),
        sorted(raw),
    )
    checks.check("스키마", "추천 입력 시트 인식", "recommendations" in raw)
    node_type = data["stores"]["node_type"].astype(str).str.upper()
    scale = manifest["scale"]
    checks.equal("스키마", "점포 수", int((node_type == "STORE").sum()), scale["store_count"])
    checks.equal("스키마", "DC 수", int((node_type == "DC").sum()), scale["dc_count"])
    checks.equal("스키마", "상품 수", int(len(data["products"])), scale["product_count"])
    route_type = data["routes"]["route_type"].astype(str).str.upper()
    checks.equal("스키마", "DIRECT 경로 행", int((route_type == "DIRECT").sum()), manifest["routes"]["direct_rows"])
    checks.equal("스키마", "VIA_DC 경로 행", int((route_type == "VIA_DC").sum()), manifest["routes"]["via_dc_rows"])
    for dc_id, key in (("DC01", "dc01_rows"), ("DC02", "dc02_rows")):
        related = int((
            (data["routes"]["source_id"].astype(str) == dc_id)
            | (data["routes"]["target_id"].astype(str) == dc_id)
        ).sum())
        checks.equal("스키마", f"{dc_id} 관련 경로 행", related, manifest["routes"][key])
    labels = (
        _names(data["stores"], "node_name") + _names(data["products"], "product_name")
        + _names(data["stores"], "region")
    )
    unexpected = [name for name in labels if not ANONYMOUS_NAME.match(name)]
    checks.check("스키마", "모든 이름이 가상 명칭 규칙을 따름", not unexpected, unexpected[:5])
    contactish = [name for name in labels if CONTACT_LIKE.search(name)]
    checks.check("스키마", "이름에 연락처·이메일 형태 없음", not contactish, contactish[:5])
    return {"data": data, "report": report}


def check_intake(
    checks: Checks, workbook: Path, manifest: dict[str, Any],
) -> dict[str, Any]:
    """Inspect-only phase: nothing may reach the applied workspace."""
    expected = manifest["expected"]
    state: dict[str, Any] = {}
    status, seconds = _timed(
        lambda: prepare_pending_data(state, str(workbook), workbook.name, SOURCE_TYPE)
    )
    checks.check("검사", "파일 검사 성공", status not in ("오류",), status)
    checks.check("검사", "업로드만으로 자동 적용되지 않음", not has_applied_data(state.get("varo_data")))
    checks.equal("검사", "검사 후 현재 추천 없음", list(state.get("varo_recommendations") or []), [])
    checks.check("검사", "검사 결과가 pending에만 저장됨", bool(state.get("pending_usable_data")))

    quality = state["pending_quality_summary"]
    checks.equal("검사", "원본 전체 행 수", quality["total_rows"], expected["total_rows"])
    checks.equal("검사", "실제 적용 행 수", quality["applied_rows"], expected["applied_rows"])
    checks.equal("검사", "제외 예정 행 수", quality["excluded_rows"], expected["excluded_rows"])
    checks.equal("검사", "경고 행 수", quality["warning_rows"], expected["warning_rows"])
    checks.equal("검사", "유지 경고 행 수", quality["warning_included_rows"], expected["warning_included_rows"])
    checks.equal("검사", "테이블별 제외 행 수", dict(quality["excluded_by_table"]), expected["excluded_by_table"])
    checks.equal("검사", "원본 테이블별 행 수", dict(quality["source_table_rows"]), manifest["scale"]["source_rows_by_table"])
    checks.equal("검사", "파일 차단 오류 수", len(quality["blockers"]), manifest["issues"]["file_blocking_count"])
    checks.check("검사", "부분 적용 허용", bool(state["pending_apply_allowed"]))

    actual_refs = _refs(state["pending_excluded_row_refs"])
    expected_refs = _refs(expected["excluded_row_refs"])
    checks.check(
        "검사", "제외 원본 행이 기대값과 정확히 일치", actual_refs == expected_refs,
        {"missing": sorted(expected_refs - actual_refs), "unexpected": sorted(actual_refs - expected_refs)},
    )
    retained = _refs(expected["retained_warning_row_refs"])
    checks.check(
        "검사", "유지 경고 행이 제외되지 않음", not (retained & actual_refs),
        sorted(retained & actual_refs),
    )

    usable = state["pending_usable_data"]
    usable_rows = {name: int(len(usable[name])) for name in expected["usable_rows_by_table"]}
    checks.equal("검사", "제외 후 테이블별 행 수", usable_rows, expected["usable_rows_by_table"])
    excluded_inventory = {row for sheet, row in actual_refs if sheet == "inventory"}
    kept_warning_rows = {row for sheet, row in retained if sheet == "inventory"}
    checks.check(
        "검사", "유지 경고 행이 적용 데이터에 존재",
        bool(kept_warning_rows) and not (kept_warning_rows & excluded_inventory),
    )
    checks.check("검사", "제외 대상 행이 적용 데이터에 없음", _no_excluded_rows_remain(usable, state))
    return {"state": state, "status": status, "prepare_seconds": seconds}


def _no_excluded_rows_remain(usable: dict[str, Any], state: dict[str, Any]) -> bool:
    """The concrete bad values injected by the generator must be gone."""
    inventory = usable["inventory"]
    stock = pd.to_numeric(inventory["stock_qty"], errors="coerce")
    if bool((stock < 0).any()) or bool(stock.isna().any()):
        return False
    if bool(inventory["store_id"].astype(str).str.strip().eq("").any()):
        return False
    if bool(inventory.duplicated(["store_id", "product_id"]).any()):
        return False
    routes = usable["routes"]
    if bool((pd.to_numeric(routes["distance_km"], errors="coerce") < 0).any()):
        return False
    if bool(pd.to_numeric(routes["estimated_cost"], errors="coerce").isna().any()):
        return False
    recommendations = usable["recommendations"]
    if bool((pd.to_numeric(recommendations["recommended_qty"], errors="coerce") <= 0).any()):
        return False
    route_type = recommendations["route_type"].astype(str).str.upper()
    if not set(route_type) <= {"DIRECT", "VIA_DC"}:
        return False
    via = route_type == "VIA_DC"
    if bool(recommendations.loc[via, "dc_id"].isna().any()):
        return False
    same = recommendations["source_id"].astype(str) == recommendations["target_id"].astype(str)
    return not bool(same.any())


def check_references(checks: Checks, usable: dict[str, Any], manifest: dict[str, Any]) -> None:
    stores, products = usable["stores"], usable["products"]
    inventory, routes = usable["inventory"], usable["routes"]
    recommendations = usable["recommendations"]
    node_ids = _text_set(stores, "node_id")
    node_type = stores["node_type"].astype(str).str.upper()
    dc_ids = {str(value).strip() for value in stores.loc[node_type == "DC", "node_id"]}
    product_ids = _text_set(products, "product_id")

    checks.check("참조", "재고→점포 고아 참조 없음", _text_set(inventory, "store_id") <= node_ids,
                 sorted(_text_set(inventory, "store_id") - node_ids))
    checks.check("참조", "재고→상품 고아 참조 없음", _text_set(inventory, "product_id") <= product_ids,
                 sorted(_text_set(inventory, "product_id") - product_ids))
    for column in ("source_id", "target_id"):
        checks.check("참조", f"경로 {column}→점포 고아 참조 없음", _text_set(routes, column) <= node_ids,
                     sorted(_text_set(routes, column) - node_ids))
        checks.check("참조", f"추천 {column}→점포 고아 참조 없음",
                     _text_set(recommendations, column) <= node_ids,
                     sorted(_text_set(recommendations, column) - node_ids))
    checks.check("참조", "추천→상품 고아 참조 없음",
                 _text_set(recommendations, "product_id") <= product_ids)

    edges = {
        (str(row["source_id"]).strip(), str(row["target_id"]).strip())
        for _, row in routes.iterrows()
    }
    missing_direct, missing_via, wrong_dc = [], [], []
    for _, row in recommendations.iterrows():
        route_type = str(row["route_type"]).upper()
        source, target = str(row["source_id"]).strip(), str(row["target_id"]).strip()
        dc_id = "" if pd.isna(row.get("dc_id")) else str(row.get("dc_id")).strip()
        if route_type == "DIRECT":
            if (source, target) not in edges:
                missing_direct.append(str(row["route_id"]))
        else:
            if dc_id not in dc_ids:
                wrong_dc.append(str(row["route_id"]))
            elif (source, dc_id) not in edges or (dc_id, target) not in edges:
                missing_via.append(str(row["route_id"]))
    checks.check("참조", "DIRECT 추천에 실제 경로 존재", not missing_direct, missing_direct)
    checks.check("참조", "VIA_DC 추천의 두 구간 경로 존재", not missing_via, missing_via)
    checks.check("참조", "VIA_DC 추천의 DC가 실제 DC", not wrong_dc, wrong_dc)

    excluded_store = manifest["expected"]["excluded_store_id"]
    checks.check("참조", "기준정보 오류 점포가 제외됨", excluded_store not in node_ids)
    expected_stores = manifest["scale"]["store_count"] + manifest["scale"]["dc_count"] - 1
    checks.equal("참조", "오류 없는 점포·DC는 모두 유지", len(node_ids), expected_stores)
    checks.equal("참조", "오류 없는 상품은 모두 유지", len(product_ids), manifest["scale"]["product_count"])


def check_multi_dc(checks: Checks, usable: dict[str, Any], manifest: dict[str, Any]) -> None:
    stores, routes = usable["stores"], usable["routes"]
    recommendations = usable["recommendations"]
    node_type = stores["node_type"].astype(str).str.upper()
    dc_ids = sorted(str(value) for value in stores.loc[node_type == "DC", "node_id"])
    checks.equal("다중 DC", "DC 기준정보 유지", dc_ids, ["DC01", "DC02"])
    names = {
        str(row["node_id"]): str(row["node_name"])
        for _, row in stores.loc[node_type == "DC"].iterrows()
    }
    checks.check("다중 DC", "DC 이름이 서로 다름", names["DC01"] != names["DC02"], names)

    for dc_id in ("DC01", "DC02"):
        legs = routes[
            (routes["source_id"].astype(str) == dc_id) | (routes["target_id"].astype(str) == dc_id)
        ]
        checks.check("다중 DC", f"{dc_id} 경유 경로 존재", len(legs) > 0, int(len(legs)))
        via = recommendations[recommendations["dc_id"].astype(str) == dc_id]
        checks.check("다중 DC", f"{dc_id} VIA_DC 추천 후보 존재", len(via) > 0, int(len(via)))

    # DC01 lost one of its stores (the broken master row). DC02's own legs must be
    # untouched: expected_dc02_routes = manifest value, nothing removed.
    dc02_routes = int((
        (routes["source_id"].astype(str) == "DC02") | (routes["target_id"].astype(str) == "DC02")
    ).sum())
    checks.equal(
        "다중 DC", "DC01 쪽 오류가 DC02 경로를 제거하지 않음",
        dc02_routes, manifest["routes"]["dc02_rows"],
    )
    direct_rows = int((routes["route_type"].astype(str).str.upper() == "DIRECT").sum())
    excluded_direct = manifest["expected"]["excluded_by_table"]["routes"]
    checks.check(
        "다중 DC", "DIRECT 경로가 DC 오류로 불필요하게 제거되지 않음",
        direct_rows >= manifest["routes"]["direct_rows"] - excluded_direct,
        {"남은 DIRECT": direct_rows},
    )
    direct_recs = recommendations[recommendations["route_type"].astype(str).str.upper() == "DIRECT"]
    checks.check(
        "다중 DC", "DIRECT 추천에는 경유 DC가 붙지 않음",
        bool(direct_recs["dc_id"].isna().all()),
    )


def check_dc_isolation(checks: Checks, workbook: Path, temp_dir: Path) -> None:
    """Break one DC's master row and confirm the other DC keeps working."""
    data, report = read_uploaded_data(workbook, workbook.name, return_report=True)
    raw = report["raw_sheets"]
    for broken, survivor in (("DC01", "DC02"), ("DC02", "DC01")):
        sheets = {key: frame.copy() for key, frame in raw.items()}
        stores = sheets["stores"]
        position = stores.index[stores["node_id"].astype(str) == broken][0]
        stores.loc[position, ["node_name", "store_name"]] = ""
        path = temp_dir / f"broken_{broken}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for key, frame in sheets.items():
                sheet_name = "v2_recommendations" if key == "recommendations" else key
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
        state: dict[str, Any] = {}
        prepare_pending_data(state, str(path), path.name, SOURCE_TYPE)
        usable = state.get("pending_usable_data") or {}
        recommendations = usable.get("recommendations")
        surviving = (
            set(recommendations["dc_id"].dropna().astype(str)) if isinstance(recommendations, pd.DataFrame)
            else set()
        )
        checks.check(
            "다중 DC", f"{broken} 오류가 {survivor} 추천을 제거하지 않음",
            survivor in surviving, sorted(surviving),
        )
        checks.check(
            "다중 DC", f"{broken} 오류 시 {broken} 경유 추천은 제외됨",
            broken not in surviving, sorted(surviving),
        )
        direct_left = (
            int((recommendations["route_type"].astype(str).str.upper() == "DIRECT").sum())
            if isinstance(recommendations, pd.DataFrame) else 0
        )
        checks.check("다중 DC", f"{broken} 오류에도 DIRECT 추천 유지", direct_left > 0, direct_left)
    del data


def check_apply(checks: Checks, state: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    before_signature = state.get("data_signature")
    checks.check("적용", "적용 버튼 전 현재 데이터 미변경", before_signature is None)
    applied, seconds = _timed(lambda: commit_pending_data(state))
    checks.check("적용", "데이터 적용 성공", applied, state.get("data_apply_error"))
    checks.check("적용", "적용 후 현재 데이터 교체됨", has_applied_data(state.get("varo_data")))
    checks.check(
        "적용", "적용 후 pending 정리됨",
        not any(key.startswith("pending_") and state.get(key) for key in list(state)),
        [key for key in state if key.startswith("pending_") and state.get(key)],
    )
    checks.equal("적용", "적용 후 이전 추천 결과 초기화", list(state["varo_recommendations"]), [])
    checks.equal("적용", "적용 후 파이프라인 결과 초기화", dict(state["varo_pipeline_result"]), {})
    checks.check("적용", "분석 자동 실행 없음 (실행 필요 상태)", bool(state["analysis_run_required"]))
    quality = state["data_quality_summary"]
    expected = manifest["expected"]
    checks.equal("적용", "적용 데이터 행 수", quality["applied_rows"], expected["applied_rows"])
    checks.equal("적용", "적용 제외 행 수", quality["excluded_rows"], expected["excluded_rows"])
    checks.equal(
        "적용", "원본 스냅샷 보존 (제외 전 행 수)",
        int(len(state["raw_data"]["inventory"])),
        manifest["scale"]["source_rows_by_table"]["inventory"],
    )
    return {"commit_seconds": seconds}


def check_analysis(checks: Checks, state: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    checks.equal("추천", "실행 버튼 전 추천 0건", len(state["varo_recommendations"]), 0)
    ok, seconds = _timed(lambda: run_applied_analysis(state))
    checks.check("추천", "추천 실행 성공", ok, state.get("analysis_run_error"))
    pipeline = state["varo_pipeline_result"]
    recommendations = state["varo_recommendations"]
    checks.check("추천", "추천 후보 1건 이상", len(recommendations) >= manifest["expected"]["minimum_recommendation_count"], len(recommendations))
    checks.check("추천", "파이프라인 상태 정상", pipeline["status"] in ("success", "partial"), pipeline["status"])
    checks.equal("추천", "알고리즘 실행 오류 없음", list(pipeline["diagnostics"]["algorithm_errors"]), [])
    checks.check("추천", "분석 실행 필요 상태 해제", not state["analysis_run_required"])

    usable_recommendations = int(len(state["varo_data"]["recommendations"]))
    ledger = pipeline["ledger_summary"]
    checks.equal("추천", "후보 생성 수 = 적용된 추천 입력 행 수", ledger["generated"], usable_recommendations)
    status_total = sum(ledger["status_counts"].values())
    checks.equal("추천", "후보 상태별 합계 = 전체 후보 수", status_total, ledger["generated"])
    feasibility = pipeline["feasibility_summary"]
    checks.equal("추천", "추천 + 제외 = 전체 후보", feasibility["feasible_count"] + feasibility["blocked_count"], feasibility["total"])
    blocked_ids = {str(item.get("route_id")) for item in pipeline["excluded_candidates"]}
    expected_blocked = set(manifest["recommendations"]["feasibility_blocked_route_ids"])
    checks.check(
        "추천", "재고 초과 요청이 추천 전에 차단됨", expected_blocked <= blocked_ids,
        {"blocked": sorted(blocked_ids), "expected_subset": sorted(expected_blocked)},
    )
    final_ids = {str(item.get("route_id")) for item in recommendations}
    checks.check("추천", "차단된 후보가 최종 추천에 없음", not (final_ids & expected_blocked))
    cascade_id = manifest["recommendations"]["cascade_excluded_route_id"]
    checks.check("추천", "제외된 원본 행이 후보 생성에 포함되지 않음", cascade_id not in final_ids, cascade_id)

    sources = pipeline["validation_report"]["calculation_sources"]
    checks.check("역할", "VHS가 최종 점수 기준", "vhs_score_engine" in str(sources.get("vhs")), sources.get("vhs"))
    checks.check("역할", "Greedy는 기준선", "heuristic_optimizer" in str(sources.get("greedy")), sources.get("greedy"))
    checks.check("역할", "Pareto 보조 검증 결과 존재", bool(pipeline["pareto_analysis"]))
    checks.check("역할", "DQN 자동 실행 없음", pipeline["diagnostics"]["dqn_artifacts_read"] is False)
    checks.check(
        "역할", "모든 후보의 DQN 상태가 미연결",
        all(str(item.get("dqn_action")) == "미연결" for item in recommendations),
    )
    checks.check("역할", "DQN 학습 결과 미보유", state.get("dqn_training_result") is None)

    zero_disguise = [
        str(item.get("route_id")) for item in recommendations
        if item.get("expected_saving") == 0 and item.get("estimated_cost") == 0
    ]
    checks.check("추천", "계산 불가 값을 0으로 위장하지 않음", not zero_disguise, zero_disguise)
    quantities = [float(item.get("recommended_qty") or 0) for item in recommendations]
    checks.check("추천", "권장 이동 수량이 모두 양수", all(value > 0 for value in quantities))
    stock_lookup = {
        (str(row["store_id"]), str(row["product_id"])): float(row["stock_qty"])
        for _, row in state["varo_data"]["inventory"].iterrows()
    }
    over_stock = [
        str(item["route_id"]) for item in recommendations
        if stock_lookup.get((str(item["source_id"]), str(item["product_id"]))) is not None
        and float(item["recommended_qty"]) > stock_lookup[(str(item["source_id"]), str(item["product_id"]))]
    ]
    checks.check("추천", "이동 수량이 출발 재고를 넘지 않음", not over_stock, over_stock)

    route_types = {str(item.get("route_type")) for item in recommendations}
    checks.check("추천", "DIRECT·VIA_DC가 모두 최종 추천에 존재", {"DIRECT", "VIA_DC"} <= route_types, sorted(route_types))
    final_dcs = {str(item.get("dc_id")) for item in recommendations if item.get("dc_id")}
    checks.check("추천", "DC01·DC02가 모두 최종 추천에 존재", {"DC01", "DC02"} <= final_dcs, sorted(final_dcs))
    return {"analysis_seconds": seconds}


def check_consistency(checks: Checks, state: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Home / 추천 실행 / 경로 상세 / 분석 및 검증 / 데이터 관리 must agree."""
    home = build_home_state(state)
    view = build_data_management_view(state)
    current = view["current"]
    recommendations = state["varo_recommendations"]
    ordered = sort_recommendations(recommendations)
    detail = find_recommendation(recommendations, state.get("selected_route_id"))
    pipeline = state["varo_pipeline_result"]
    ledger = pipeline["ledger_summary"]
    quality = state["data_quality_summary"]

    checks.equal("화면 일치", "홈·추천 실행 후보 수", home["recommendation_count"], len(recommendations))
    checks.equal("화면 일치", "데이터 관리·홈 후보 수", current["recommendation_count"], home["recommendation_count"])
    checks.equal("화면 일치", "검증 페이지 추천 가능 후보 수", ledger["recommendable_total"], len(recommendations))
    checks.equal("화면 일치", "홈·검증 이동 불가 후보 수", home["blocked_count"], ledger["excluded_total"])

    checks.equal("화면 일치", "데이터 관리·검사 전체 행 수", current["total_rows"], quality["total_rows"])
    checks.equal("화면 일치", "데이터 관리·적용 행 수", current["usable_rows"], quality["applied_rows"])
    checks.equal("화면 일치", "데이터 관리·제외 행 수", current["excluded_rows"], quality["excluded_rows"])
    checks.equal("화면 일치", "데이터 관리·경고 행 수", current["warning_rows"], quality["warning_rows"])
    checks.equal("화면 일치", "데이터 관리 적용 행 수 = manifest", current["usable_rows"], manifest["expected"]["applied_rows"])

    top = home["top_recommendation"]
    checks.check("화면 일치", "홈 최우선 추천 존재", bool(top))
    checks.equal("화면 일치", "홈·추천표 최우선 후보 동일", str(top["route_id"]), str(ordered[0]["route_id"]))
    checks.equal("화면 일치", "홈·경로 상세 최우선 후보 동일", str(top["route_id"]), str(detail["route_id"]))
    for field in (
        "recommended_qty", "route_type", "dc_id", "estimated_cost",
        "expected_saving", "vhs_score", "confidence", "reason",
    ):
        checks.equal(
            "화면 일치", f"홈·경로 상세 {field} 동일",
            _plain(top.get(field)), _plain(detail.get(field)),
        )
    checks.equal(
        "화면 일치", "선택된 후보가 추천 목록에 존재",
        str(state["selected_route_id"]) in {str(item["route_id"]) for item in recommendations}, True,
    )
    checks.check("화면 일치", "선택 후보 유효 표시", bool(home["selected_candidate_valid"]))

    record = next(
        (item for item in pipeline["candidate_ledger"] if str(item.get("route_id")) == str(top["route_id"])),
        None,
    )
    checks.check("화면 일치", "최우선 후보의 원본 계보 기록 존재", bool(record))
    if record:
        checks.equal(
            "화면 일치", "계보의 이동 수량이 추천표와 동일",
            _plain(record.get("recommended_qty")), _plain(top.get("recommended_qty")),
        )
        checks.check(
            "화면 일치", "최우선 후보의 원본 행 위치 기록",
            int(record.get("traceable_row_count") or 0) > 0,
            record.get("traceable_row_count"),
        )

    # 추천 실행 / 경로 상세 both render (ledger reasons or analysis sentences)[:3].
    detail_reasons = pipeline["reason_analysis"]["reasons"].get(str(top["route_id"])) or {}
    shown = ((record or {}).get("recommendation_reasons") or detail_reasons.get("sentences") or [])[:3]
    checks.check("화면 일치", "화면에 표시되는 추천 이유 1~3문장", 1 <= len(shown) <= 3, shown)
    checks.check(
        "화면 일치", "추천 이유 문장이 짧게 유지됨",
        all(len(str(line)) <= 120 for line in shown), [len(str(line)) for line in shown],
    )

    confidence = pipeline["confidence_status"]
    checks.check("화면 일치", "신뢰도 상태가 계산 가능 여부와 함께 표시", bool(confidence.get("status")), confidence.get("status"))
    stability = pipeline["stability_analysis_status"]
    checks.check("화면 일치", "안정성 상태 표시", bool(stability.get("status")), stability.get("status"))


def check_ui(checks: Checks, state: dict[str, Any]) -> dict[str, Any]:
    """Render every page headlessly and look for leaked internals / errors."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as exc:  # pragma: no cover - streamlit missing
        checks.check("UI", "AppTest 사용 가능", False, str(exc))
        return {}
    from services.app_state import CANONICAL_DATA_KEYS

    app_path = str(PROJECT_ROOT / "app_v2.py")
    app = AppTest.from_file(app_path, default_timeout=180)
    app.run()
    for key in CANONICAL_DATA_KEYS:
        app.session_state[key] = state.get(key)
    signature = str(state.get("data_signature") or "")
    pages = ["운영 현황", "추천 실행", "경로 상세", "분석 및 검증", "데이터 관리"]
    rendered: dict[str, int] = {}
    for page in pages:
        app.session_state["current_menu"] = page
        app.run()
        checks.check("UI", f"{page} 페이지 렌더 오류 없음", not app.exception, [str(e) for e in app.exception])
        blob = " ".join(element.value for element in app.markdown)
        rendered[page] = len(blob)
        checks.check("UI", f"{page} 내부 signature 미노출", signature[:16] not in blob if signature else True)
        checks.check("UI", f"{page} traceback 미노출", "Traceback" not in blob)
        for token in ("candidate_id", "usable_signature", "pending_usable_data", "C:\\\\Projects"):
            checks.check("UI", f"{page} 내부 식별자 미노출 ({token})", token not in blob)
    return {"rendered_markdown_chars": rendered}


def measure_performance(checks: Checks, workbook: Path, repeats: int) -> dict[str, Any]:
    stages: dict[str, list[float]] = {"read": [], "prepare": [], "commit": [], "analysis": [], "total": []}
    for _ in range(repeats):
        start = time.perf_counter()
        _data, read_seconds = _timed(lambda: read_uploaded_data(workbook, workbook.name, return_report=True))
        state: dict[str, Any] = {}
        _status, prepare_seconds = _timed(
            lambda: prepare_pending_data(state, str(workbook), workbook.name, SOURCE_TYPE)
        )
        _ok, commit_seconds = _timed(lambda: commit_pending_data(state))
        _ran, analysis_seconds = _timed(lambda: run_applied_analysis(state))
        stages["read"].append(read_seconds)
        stages["prepare"].append(prepare_seconds)
        stages["commit"].append(commit_seconds)
        stages["analysis"].append(analysis_seconds)
        stages["total"].append(time.perf_counter() - start)
    summary = {
        stage: {
            "first": round(values[0], 3),
            "median": round(statistics.median(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "runs": len(values),
        }
        for stage, values in stages.items()
    }
    checks.check(
        "성능", "전체 흐름이 합리적인 시간 안에 완료", summary["total"]["max"] < 120.0,
        summary["total"],
    )
    return summary


def measure_stage_breakdown(workbook: Path, repeats: int) -> dict[str, Any]:
    """Per-stage timings, measured by calling the same services the app calls."""
    from services.data_issues import collect_data_issues, exclusion_row_refs
    from services.data_loader import normalize_loaded_data
    from services.data_validator import validate_workbook_data
    from services.partial_data import build_usable_data, usable_data_signature

    stages: dict[str, list[float]] = {}

    def record(name: str, seconds: float) -> None:
        stages.setdefault(name, []).append(seconds)

    for _ in range(repeats):
        (data, report), seconds = _timed(
            lambda: read_uploaded_data(workbook, workbook.name, return_report=True)
        )
        record("파일 읽기+정규화", seconds)
        raw = report["raw_sheets"]
        metadata = {
            "filename": workbook.name, "source_type": "excel",
            "sheet_names": dict(report["raw_sheet_names"]),
        }
        _normalized, seconds = _timed(
            lambda: normalize_loaded_data({k: v.copy() for k, v in raw.items()})
        )
        record("정규화", seconds)
        issue_result, seconds = _timed(lambda: collect_data_issues(data, raw, metadata))
        record("검증·문제 분류", seconds)
        _refs, seconds = _timed(lambda: exclusion_row_refs(issue_result["issues"]))
        record("제외 대상 계산", seconds)
        partial, seconds = _timed(lambda: build_usable_data(data, raw, metadata))
        record("참조 정리+usable data 생성(재검증 포함)", seconds)
        _revalidated, seconds = _timed(lambda: validate_workbook_data(partial["usable_data"]))
        record("제외 후 재검증", seconds)
        _signature, seconds = _timed(lambda: usable_data_signature(partial["usable_data"]))
        record("적용 데이터 서명", seconds)
        state: dict[str, Any] = {}
        _status, seconds = _timed(
            lambda: prepare_pending_data(state, str(workbook), workbook.name, SOURCE_TYPE)
        )
        record("pending 저장(전체 검사)", seconds)
    return {
        name: {
            "first": round(values[0], 3),
            "median": round(statistics.median(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }
        for name, values in stages.items()
    }


def check_memory_behaviour(checks: Checks, workbook: Path) -> None:
    """Pending intake must be released and lineage must stay reference-sized."""
    state: dict[str, Any] = {}
    prepare_pending_data(state, str(workbook), workbook.name, SOURCE_TYPE)
    commit_pending_data(state)
    leftovers = [key for key in state if key.startswith("pending_") and state.get(key)]
    checks.check("메모리", "적용 후 pending 대용량 객체 정리", not leftovers, leftovers)
    cancel_state: dict[str, Any] = {}
    prepare_pending_data(cancel_state, str(workbook), workbook.name, SOURCE_TYPE)
    cancel_pending_data(cancel_state)
    checks.check(
        "메모리", "검사 취소 시 pending 해제",
        not [key for key in cancel_state if key.startswith("pending_") and cancel_state.get(key)],
    )
    run_applied_analysis(state)
    ledger = state["varo_pipeline_result"]["candidate_ledger"]
    inventory_rows = int(len(state["varo_data"]["inventory"]))
    oversized = [
        str(record.get("route_id")) for record in ledger
        if len(str(record)) > 20000
    ]
    checks.check("메모리", "후보별 원본 전체 복제 없음", not oversized, oversized)
    checks.check(
        "메모리", "계보 기록이 참조 정보 중심",
        all(int(record.get("traceable_row_count") or 0) < inventory_rows for record in ledger),
        max(int(record.get("traceable_row_count") or 0) for record in ledger) if ledger else 0,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(output_dir: Path, repeats: int, regenerate: bool, skip_ui: bool) -> tuple[int, dict[str, Any]]:
    import tempfile

    checks = Checks()
    timings: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as temp_name:
        temp_dir = Path(temp_name)
        generation = check_generation(checks, output_dir, regenerate)
        manifest, workbook = generation["manifest"], generation["workbook"]
        check_reproducible(checks, output_dir, temp_dir)
        check_schema(checks, workbook, manifest)

        intake = check_intake(checks, workbook, manifest)
        state = intake["state"]
        timings["prepare_seconds"] = round(intake["prepare_seconds"], 3)
        check_references(checks, state["pending_usable_data"], manifest)
        check_multi_dc(checks, state["pending_usable_data"], manifest)
        check_dc_isolation(checks, workbook, temp_dir)

        timings.update({k: round(v, 3) for k, v in check_apply(checks, state, manifest).items()})
        timings.update({k: round(v, 3) for k, v in check_analysis(checks, state, manifest).items()})
        check_consistency(checks, state, manifest)
        check_memory_behaviour(checks, workbook)
        ui = {} if skip_ui else check_ui(checks, state)
        performance = measure_performance(checks, workbook, repeats)
        stage_breakdown = measure_stage_breakdown(workbook, repeats)

    report = {
        "workbook": str(workbook),
        "manifest": str(output_dir / MANIFEST_NAME),
        "scale": manifest["scale"],
        "single_run_seconds": timings,
        "performance": performance,
        "stage_breakdown": stage_breakdown,
        "ui": ui,
        "recommendation_summary": {
            "final_recommendations": len(state["varo_recommendations"]),
            "ledger": state["varo_pipeline_result"]["ledger_summary"],
            "feasibility": state["varo_pipeline_result"]["feasibility_summary"],
        },
        "checks_total": len(checks.results),
        "checks_failed": len(checks.failures),
        "failures": checks.failures,
        "checks": checks.results,
    }
    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return (1 if checks.failures else 0), report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=3, help="성능 측정 반복 횟수 (기본 3)")
    parser.add_argument("--regenerate", action="store_true", help="워크북을 다시 생성한 뒤 검증")
    parser.add_argument("--skip-ui", action="store_true", help="AppTest 페이지 렌더 검증 생략")
    args = parser.parse_args()

    code, report = run(args.output_dir, args.repeats, args.regenerate, args.skip_ui)
    by_section: dict[str, list[int]] = {}
    for item in report["checks"]:
        counts = by_section.setdefault(item["section"], [0, 0])
        counts[0] += 1
        counts[1] += 0 if item["ok"] else 1
    print(f"검증 파일: {report['workbook']}")
    for section, (total, failed) in by_section.items():
        mark = "OK " if not failed else "FAIL"
        print(f"  [{mark}] {section}: {total - failed}/{total}")
    performance = report["performance"]
    print("성능(초, 중앙값): " + " · ".join(
        f"{stage}={values['median']}" for stage, values in performance.items()
    ))
    print("단계별(초, 중앙값):")
    for stage, values in report["stage_breakdown"].items():
        print(f"  - {stage}: 최초 {values['first']} / 중앙값 {values['median']} "
              f"(범위 {values['min']}~{values['max']})")
    if report["failures"]:
        print(f"\n실패 {len(report['failures'])}건:")
        for item in report["failures"]:
            print(f"  - [{item['section']}] {item['check']}: {item['detail']}")
    else:
        print(f"\n전체 {report['checks_total']}개 검증 항목 통과")
    print(f"보고서: {report['manifest'].rsplit('_manifest', 1)[0]}_report.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

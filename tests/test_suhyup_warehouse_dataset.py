"""수협 조합창고 재고 + 입출고 실데이터 안전 결합 테스트.

논리 검증은 전부 합성 데이터로 한다. 실데이터는 프로젝트 밖에 있고 항상 있다고 보장할 수
없으므로, 실데이터 회귀 테스트는 원본이 실제로 있을 때만 실행한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from services.suhyup_warehouse_dataset import (
    DEFAULT_OUTPUT_NAME, MATCH_AMBIGUOUS, MATCH_MATCHED, MATCH_STOCK_ONLY,
    MATCH_UNMATCHED_DATE, MATCH_UNMATCHED_LOCATION, MATCH_UNMATCHED_PRODUCT,
    PROVENANCE_ACTUAL, PROVENANCE_ASSUMED, PROVENANCE_NOT_AVAILABLE,
    STATUS_UNUSABLE, SUPERSEDED_OUTPUTS, WarehouseSources, add_analysis_fields,
    add_balance_diagnostic, aggregate_flow, build_warehouse_dataset, classify_key_cardinality,
    detect_encoding, file_digest, join_stock_flow, key_uniqueness, locate_warehouse_sources,
    normalize_product_name, prepare_flow, prepare_stock, read_raw_csv, run_from_sources,
    species_code_from_standard_code, split_state_suffix, write_outputs,
)

# --------------------------------------------------------------------------- #
# 합성 원본 만들기
# --------------------------------------------------------------------------- #

FLOW_HEADER = ["표준코드", "조합코드", "창고코드", "입출고구분", "기준일자",
               "조합명", "창고명", "표준코드명", "입출고구분명", "수량"]
STOCK_HEADER = ["조합코드", "창고코드", "표준어종코드", "기준일자",
                "조합명", "창고명", "표준어종명", "수량"]


def flow_row(code="61010030", coop="358", warehouse="3580002", direction="1",
             date="2026-07-01", name="가오리류(냉동)", qty="10"):
    label = "입고" if direction == "1" else "출고"
    return [code, coop, warehouse, direction, date, "울산수협", "울산창고", name, label, qty]


def stock_row(coop="358", warehouse="3580002", species="610100", date="2026-07-01",
              name="가오리류", qty="100"):
    return [coop, warehouse, species, date, "울산수협", "울산창고", name, qty]


def make_raw(rows, header):
    frame = pd.DataFrame(rows, columns=header)
    frame["source_row"] = frame.index + 2
    return frame


def build(flow_rows, stock_rows, tmp_path: Path):
    sources = WarehouseSources(
        flow_path=tmp_path / "입출고.CSV",
        stock_path=tmp_path / "재고.CSV",
        processed_dir=tmp_path / "processed",
    )
    return build_warehouse_dataset(
        make_raw(flow_rows, FLOW_HEADER), make_raw(stock_rows, STOCK_HEADER), sources,
    )


# --------------------------------------------------------------------------- #
# A. 상품명 정규화
# --------------------------------------------------------------------------- #

def test_normalization_collapses_whitespace_only():
    assert normalize_product_name("  가오리류(냉동)  ") == "가오리류(냉동)"
    assert normalize_product_name("가오리류\t \n(냉동)") == "가오리류 (냉동)"


def test_normalization_applies_unicode_nfkc():
    assert normalize_product_name("ＡＢ１２３") == "AB123"
    assert normalize_product_name("가오리류") == "가오리류"


def test_normalization_removes_control_and_zero_width_characters():
    assert normalize_product_name("가오리​류﻿") == "가오리류"
    assert normalize_product_name("가오리\x00류") == "가오리류"


def test_normalization_never_merges_distinct_real_products():
    """숫자·용량·규격·등급·상태를 지우지 않는다. 서로 다른 상품은 계속 다르다."""
    distinct = ["갈치류(냉동)", "갈치류(냉장/신선)", "갈치류(활)", "기타갈치류(냉동)",
                "명태 1kg", "명태 2kg", "명태 특", "명태 상", "국산 명태", "수입 명태"]
    assert len({normalize_product_name(name) for name in distinct}) == len(distinct)


def test_normalization_is_deterministic_and_preserves_original():
    raw = "  가오리류 ​ (냉동) "
    first = normalize_product_name(raw)
    assert first == normalize_product_name(raw)
    assert raw == "  가오리류 ​ (냉동) "  # 원본 문자열은 그대로


def test_normalization_handles_missing_values():
    assert normalize_product_name(None) is None
    assert normalize_product_name("") is None
    assert normalize_product_name("   ") is None
    assert normalize_product_name(float("nan")) is None


def test_split_state_suffix_only_strips_observed_labels():
    assert split_state_suffix("가오리류(냉동)") == ("가오리류", "냉동")
    assert split_state_suffix("굴(참굴)(냉동)") == ("굴(참굴)", "냉동")
    # 모르는 표기는 떼지 않는다 → 계층 검증이 안전하게 실패한다.
    assert split_state_suffix("가오리류(초저온)") == ("가오리류(초저온)", None)
    assert split_state_suffix(None) == (None, None)
    assert split_state_suffix(float("nan")) == (None, None)


def test_species_code_is_a_deterministic_prefix():
    assert species_code_from_standard_code("61010030") == "610100"
    assert species_code_from_standard_code("61010020") == "610100"
    assert species_code_from_standard_code("12345") is None
    assert species_code_from_standard_code(None) is None


# --------------------------------------------------------------------------- #
# B. 코드 체계 검증 — 실패하면 결합하지 않는다
# --------------------------------------------------------------------------- #

def test_product_codes_are_never_joined_directly(tmp_path):
    """8자리 표준코드와 6자리 어종코드는 완전 일치가 0이어도 결합은 성공해야 한다."""
    report = build([flow_row()], [stock_row()], tmp_path)
    relation = report.manifest["product_code_relation"]
    assert relation["exact_full_code_overlap"] == 0
    assert relation["verified"] is True
    assert report.join.metrics["flow_matched_keys"] == 1


def test_join_is_blocked_when_names_contradict_the_code_hierarchy(tmp_path):
    """같은 어종코드인데 이름이 다르면 코드 계층 가설이 깨진 것이므로 차단한다."""
    report = build(
        [flow_row(code="61010030", name="가오리류(냉동)")],
        [stock_row(species="610100", name="고등어")],
        tmp_path,
    )
    assert report.manifest["product_code_relation"]["name_mismatches"] == 1
    assert report.status == STATUS_UNUSABLE
    assert report.join.blockers


def test_join_key_is_species_code_not_product_name(tmp_path):
    """이름이 같아도 어종코드가 다르면 다른 상품으로 유지한다(자동 병합 금지)."""
    report = build(
        [flow_row(code="61910100", name="붕장어(냉동)", qty="5"),
         flow_row(code="93210600", name="붕장어(건)", qty="7")],
        [stock_row(species="619101", name="붕장어", qty="50"),
         stock_row(species="932106", name="붕장어", qty="70")],
        tmp_path,
    )
    matched = report.output[report.output["match_status"] == MATCH_MATCHED]
    assert len(matched) == 2
    assert set(matched["species_code"]) == {"619101", "932106"}
    assert set(matched["inbound_qty_actual"]) == {5.0, 7.0}


# --------------------------------------------------------------------------- #
# C. 키 유일성과 1:1 / 1:N / N:1 / N:M
# --------------------------------------------------------------------------- #

def test_key_uniqueness_counts_duplicates():
    frame = pd.DataFrame({"a": ["x", "x", "y"], "b": ["1", "1", "2"]})
    stats = key_uniqueness(frame, ["a", "b"])
    assert stats == {"rows": 3, "unique_keys": 2, "duplicate_keys": 1,
                     "duplicate_rows": 2, "max_multiplicity": 2}


def test_cardinality_classification_covers_every_case():
    counts = classify_key_cardinality(
        {"a": 1, "b": 1, "c": 2, "d": 2, "e": 1},
        {"a": 1, "b": 3, "c": 1, "d": 2},
    )
    assert counts == {"one_to_one": 1, "one_to_many": 1, "many_to_one": 1, "many_to_many": 1}


def test_real_shape_is_one_to_one_after_aggregation(tmp_path):
    report = build(
        [flow_row(direction="1", qty="10"), flow_row(direction="2", qty="4")],
        [stock_row(qty="100")],
        tmp_path,
    )
    cardinality = report.join.metrics["cardinality"]
    assert cardinality == {"one_to_one": 1, "one_to_many": 0, "many_to_one": 0, "many_to_many": 0}


def test_no_uncontrolled_row_explosion(tmp_path):
    """flow 여러 event가 stock 한 행에 붙어도 출력 행이 늘어나지 않는다."""
    flow_rows = [
        flow_row(code="61010030", direction="1", qty="10"),
        flow_row(code="61010030", direction="2", qty="3"),
        flow_row(code="61010020", direction="1", name="가오리류(냉장/신선)", qty="5"),
        flow_row(code="61010020", direction="2", name="가오리류(냉장/신선)", qty="2"),
    ]
    report = build(flow_rows, [stock_row(qty="100")], tmp_path)
    assert report.join.metrics["row_expansion_factor"] == 1.0
    assert len(report.output) == 1
    assert not report.join.blockers


# --------------------------------------------------------------------------- #
# D. 집계 — 방향을 섞지 않고, 없는 값을 만들지 않는다
# --------------------------------------------------------------------------- #

def test_same_day_multiple_events_are_summed_per_direction(tmp_path):
    flow_rows = [
        flow_row(code="61010030", direction="1", qty="10"),
        flow_row(code="61010020", direction="1", name="가오리류(냉장/신선)", qty="15"),
        flow_row(code="61010030", direction="2", qty="4"),
        flow_row(code="61010020", direction="2", name="가오리류(냉장/신선)", qty="6"),
    ]
    report = build(flow_rows, [stock_row()], tmp_path)
    row = report.output.iloc[0]
    assert row["inbound_qty_actual"] == 25.0
    assert row["outbound_qty_actual"] == 10.0
    assert row["inbound_record_count"] == 2
    assert row["outbound_record_count"] == 2


def test_inbound_and_outbound_are_never_mixed(tmp_path):
    report = build([flow_row(direction="2", qty="9")], [stock_row()], tmp_path)
    row = report.output.iloc[0]
    assert row["outbound_qty_actual"] == 9.0
    assert pd.isna(row["inbound_qty_actual"])          # 실측 컬럼은 결측 유지
    assert row["inbound_qty_for_analysis"] == 0.0      # 분석용 컬럼에서만 0
    assert bool(row["inbound_qty_zero_filled"]) is True
    assert bool(row["outbound_qty_zero_filled"]) is False


def test_aggregation_keeps_every_contributing_source_code(tmp_path):
    flow_rows = [
        flow_row(code="61010030", direction="1", qty="10"),
        flow_row(code="61010020", direction="1", name="가오리류(냉장/신선)", qty="5"),
    ]
    report = build(flow_rows, [stock_row()], tmp_path)
    row = report.output.iloc[0]
    assert row["flow_standard_codes"] == "61010020|61010030"
    assert row["flow_standard_code_count"] == 2
    assert row["flow_product_names"] == "가오리류(냉동)|가오리류(냉장/신선)"


def test_aggregate_flow_skips_rows_without_species_code():
    flow = prepare_flow(make_raw([flow_row(code="123")], FLOW_HEADER))
    assert aggregate_flow(flow).empty


# --------------------------------------------------------------------------- #
# E. 모호한 키는 자동 선택하지 않는다
# --------------------------------------------------------------------------- #

def test_duplicate_stock_key_is_left_ambiguous_not_auto_picked(tmp_path):
    report = build(
        [flow_row(direction="1", qty="10")],
        [stock_row(qty="100"), stock_row(qty="123")],
        tmp_path,
    )
    statuses = list(report.output["match_status"])
    # 재고 후보 2행 + 붙일 곳을 정할 수 없는 입출고 1행이 모두 ambiguous로 남는다.
    assert statuses.count(MATCH_AMBIGUOUS) == 3
    assert MATCH_MATCHED not in statuses
    # 두 후보 값이 모두 남아 있고 어느 쪽도 선택되지 않았다.
    assert sorted(report.output["stock_qty_actual"].dropna()) == [100.0, 123.0]
    assert report.manifest["validation_status"] != STATUS_UNUSABLE
    assert not report.join.blockers


# --------------------------------------------------------------------------- #
# F. 미결합 행 보존과 사유 구분
# --------------------------------------------------------------------------- #

def test_unmatched_product_is_preserved_with_reason(tmp_path):
    report = build([flow_row(code="99999900", name="미등록어종(냉동)")], [stock_row()], tmp_path)
    unmatched = report.output[report.output["match_status"] == MATCH_UNMATCHED_PRODUCT]
    assert len(unmatched) == 1
    assert unmatched.iloc[0]["species_code"] == "999999"
    assert unmatched.iloc[0]["flow_standard_codes"] == "99999900"


def test_unmatched_location_and_date_are_distinguished(tmp_path):
    flow_rows = [
        flow_row(warehouse="9999999", coop="999"),                 # 재고에 없는 창고
        flow_row(date="2026-07-09"),                               # 창고는 있으나 날짜가 없음
    ]
    report = build(flow_rows, [stock_row()], tmp_path)
    statuses = set(report.output["match_status"])
    assert MATCH_UNMATCHED_LOCATION in statuses
    assert MATCH_UNMATCHED_DATE in statuses


def test_stock_without_flow_is_kept_and_not_called_unmatched_product(tmp_path):
    report = build([], [stock_row(), stock_row(date="2026-07-02")], tmp_path)
    assert list(report.output["match_status"]) == [MATCH_STOCK_ONLY] * 2
    assert report.join.metrics["stock_unmatched_rows"] == 2


def test_no_row_is_dropped_to_raise_the_join_rate(tmp_path):
    flow_rows = [flow_row(), flow_row(code="99999900", name="미등록어종(냉동)")]
    stock_rows = [stock_row(), stock_row(species="610200", name="가자미류")]
    report = build(flow_rows, stock_rows, tmp_path)
    metrics = report.join.metrics
    assert metrics["output_rows"] == len(stock_rows) + 1  # 미결합 flow 키 1건이 추가로 보존됨
    assert metrics["flow_unmatched_keys"] == 1


# --------------------------------------------------------------------------- #
# G. 계보
# --------------------------------------------------------------------------- #

def test_output_traces_back_to_original_rows_codes_and_names(tmp_path):
    report = build([flow_row(direction="1", qty="10")], [stock_row()], tmp_path)
    row = report.output.iloc[0]
    assert row["source_dataset"] == "03_Korea_Suhyup_Warehouse"
    assert row["stock_source_file"] == "재고.CSV"
    assert row["flow_source_file"] == "입출고.CSV"
    assert int(row["stock_source_row"]) == 2
    assert row["flow_source_rows"] == "2"
    assert row["flow_product_names"] == "가오리류(냉동)"          # 원본 상품명
    assert row["flow_standard_codes"] == "61010030"              # 원본 상품코드
    assert row["species_name"] == "가오리류"                     # 원본 어종명
    assert row["species_name_normalized"] == "가오리류"          # 정규화 상품명
    assert row["species_name_consistency"] == "verified_identical"
    assert row["species_code_source"] == "stock_standard_species_code"


def test_flow_only_row_marks_species_code_as_derived(tmp_path):
    report = build([flow_row(code="99999900", name="미등록어종(냉동)")], [stock_row()], tmp_path)
    unmatched = report.output[report.output["match_status"] == MATCH_UNMATCHED_PRODUCT].iloc[0]
    assert unmatched["species_code_source"] == "derived_from_flow_standard_code_prefix"
    assert unmatched["stock_source_file"] == ""
    assert unmatched["coop_name"] == "울산수협"   # 위치 이름은 flow 원본에서 보존


# --------------------------------------------------------------------------- #
# H. Provenance
# --------------------------------------------------------------------------- #

def test_observed_values_stay_actual_and_fills_stay_assumed(tmp_path):
    report = build([flow_row(direction="2", qty="9")], [stock_row()], tmp_path)
    row = report.output.iloc[0]
    assert row["inventory_provenance"] == PROVENANCE_ACTUAL
    assert row["outbound_provenance"] == PROVENANCE_ACTUAL
    assert row["inbound_provenance"] == PROVENANCE_NOT_AVAILABLE
    assert row["flow_zero_fill_provenance"] == PROVENANCE_ASSUMED


def test_absent_logistics_facts_are_not_available_everywhere(tmp_path):
    report = build([flow_row()], [stock_row()], tmp_path)
    for column in ("weight_kg_provenance", "interwarehouse_transfer_provenance",
                   "distance_provenance", "transfer_cost_provenance", "vehicle_capacity_provenance"):
        assert set(report.output[column]) == {PROVENANCE_NOT_AVAILABLE}
    provenance = report.manifest["provenance"]
    assert provenance["stock_qty"] == PROVENANCE_ACTUAL
    assert provenance["interwarehouse_transfer_history"] == PROVENANCE_NOT_AVAILABLE
    assert provenance["distance"] == PROVENANCE_NOT_AVAILABLE
    assert provenance["vehicle_capacity"] == PROVENANCE_NOT_AVAILABLE


def test_normalization_does_not_downgrade_actual_provenance(tmp_path):
    """이름 정규화를 거쳐도 실측 수량의 provenance는 actual로 남는다."""
    report = build([flow_row(name="  가오리류​(냉동) ", direction="1")], [stock_row()], tmp_path)
    row = report.output.iloc[0]
    assert row["species_name_consistency"] == "verified_identical"
    assert row["inbound_provenance"] == PROVENANCE_ACTUAL
    assert row["inventory_provenance"] == PROVENANCE_ACTUAL


# --------------------------------------------------------------------------- #
# I. 이상치 — 원본은 고치지 않고 flag만 남긴다
# --------------------------------------------------------------------------- #

def test_anomalies_are_flagged_not_corrected(tmp_path):
    flow_rows = [flow_row(code="61020030", name="가자미류(냉동)", direction="2", qty="-5")]
    stock_rows = [
        stock_row(qty="0"),
        stock_row(species="610200", name="가자미류", qty="-12"),
        stock_row(species="610300", name="갈치류", date="2026/07/01", qty="7"),
    ]
    report = build(flow_rows, stock_rows, tmp_path)
    anomalies = report.manifest["anomalies"]
    assert anomalies["stock_qty_zero"] == 1
    assert anomalies["stock_qty_negative"] == 1
    assert anomalies["outbound_qty_negative"] == 1
    assert anomalies["date_invalid"] == 1
    # 원본 값이 남아 있고 임의 수정되지 않았다.
    negative = report.output[report.output["species_code"] == "610200"].iloc[0]
    assert negative["stock_qty_actual"] == -12.0
    assert negative["stock_qty_raw"] == "-12"
    assert report.manifest["validation_status"] == "검토 필요"


def test_missing_product_name_is_flagged_but_row_survives(tmp_path):
    report = build([flow_row(code="93000000", name="")], [stock_row()], tmp_path)
    assert report.manifest["anomalies"]["product_name_missing"] == 1
    assert MATCH_UNMATCHED_PRODUCT in set(report.output["match_status"])


# --------------------------------------------------------------------------- #
# J. 재고 수지 진단
# --------------------------------------------------------------------------- #

def test_balance_diagnostic_matches_inbound_minus_outbound(tmp_path):
    flow_rows = [flow_row(date="2026-07-02", direction="2", qty="20")]
    stock_rows = [stock_row(date="2026-07-01", qty="100"), stock_row(date="2026-07-02", qty="80")]
    report = build(flow_rows, stock_rows, tmp_path)
    second = report.output[report.output["date"] == "2026-07-02"].iloc[0]
    assert second["stock_delta"] == -20.0
    assert second["net_flow"] == -20.0
    assert second["balance_gap"] == 0.0
    assert second["balance_status"] == "consistent"


def test_balance_diagnostic_reports_a_gap_without_editing_values(tmp_path):
    flow_rows = [flow_row(date="2026-07-02", direction="2", qty="20")]
    stock_rows = [stock_row(date="2026-07-01", qty="100"), stock_row(date="2026-07-02", qty="95")]
    report = build(flow_rows, stock_rows, tmp_path)
    second = report.output[report.output["date"] == "2026-07-02"].iloc[0]
    assert second["balance_status"] == "gap"
    assert second["balance_gap"] == 15.0
    assert second["stock_qty_actual"] == 95.0          # 값은 그대로
    assert len(report.balance_gaps) == 1


def test_balance_diagnostic_is_not_a_hard_validation(tmp_path):
    """수지가 안 맞아도 결합 자체는 실패로 만들지 않는다(참고 진단)."""
    flow_rows = [flow_row(date="2026-07-02", direction="2", qty="20")]
    stock_rows = [stock_row(date="2026-07-01", qty="100"), stock_row(date="2026-07-02", qty="95")]
    report = build(flow_rows, stock_rows, tmp_path)
    assert not report.join.blockers
    assert report.manifest["validation_status"] != STATUS_UNUSABLE


def test_balance_is_not_computed_without_a_previous_day(tmp_path):
    report = build([], [stock_row(date="2026-07-01"), stock_row(date="2026-07-05")], tmp_path)
    assert set(report.output["balance_status"]) == {"no_prior_day"}
    assert report.manifest["stock_flow_balance_diagnostic"]["comparable_day_pairs"] == 0


def test_balance_tolerates_float_representation_error(tmp_path):
    stock_rows = [stock_row(date="2026-07-01", qty="411"), stock_row(date="2026-07-02", qty="408.2")]
    flow_rows = [flow_row(date="2026-07-02", direction="2", qty="2.8")]
    report = build(flow_rows, stock_rows, tmp_path)
    second = report.output[report.output["date"] == "2026-07-02"].iloc[0]
    assert second["balance_status"] == "consistent"
    assert abs(second["balance_gap"]) > 0 or second["balance_gap"] == 0  # 부동소수 잔차 허용


# --------------------------------------------------------------------------- #
# K. 재현성
# --------------------------------------------------------------------------- #

def test_same_input_produces_the_same_output(tmp_path):
    flow_rows = [flow_row(direction="1"), flow_row(direction="2", qty="4")]
    stock_rows = [stock_row(), stock_row(date="2026-07-02", qty="90")]
    first = build(flow_rows, stock_rows, tmp_path).output
    second = build(flow_rows, stock_rows, tmp_path).output
    pd.testing.assert_frame_equal(first, second)


def test_input_row_order_does_not_change_the_logical_result(tmp_path):
    flow_rows = [flow_row(direction="1", qty="10"), flow_row(direction="2", qty="4"),
                 flow_row(date="2026-07-02", direction="1", qty="7")]
    stock_rows = [stock_row(), stock_row(date="2026-07-02", qty="90")]
    normal = build(flow_rows, stock_rows, tmp_path).output
    reversed_order = build(flow_rows[::-1], stock_rows[::-1], tmp_path).output
    compared = [column for column in normal.columns
                if column not in ("stock_source_row", "flow_source_rows")]
    pd.testing.assert_frame_equal(normal[compared], reversed_order[compared])


def test_manifest_reports_the_versions_and_the_real_key(tmp_path):
    report = build([flow_row()], [stock_row()], tmp_path)
    manifest = report.manifest
    assert manifest["join_key"] == ["coop_code", "warehouse_code", "date", "species_code"]
    assert manifest["normalization_version"].startswith("suhyup-warehouse-name-normalize/")
    assert manifest["join_version"].startswith("suhyup-warehouse-species-code-join/")
    assert manifest["output"]["encoding"] == "utf-8-sig"
    assert any("이동 이력" in item for item in manifest["limitations"])


def test_name_key_comparison_explains_the_difference(tmp_path):
    """이름 키를 쓰면 서로 다른 어종코드가 한 키로 합쳐진다는 사실을 수치로 남긴다."""
    report = build(
        [flow_row(code="61910100", name="붕장어(냉동)"), flow_row(code="93210600", name="붕장어(건)")],
        [stock_row(species="619101", name="붕장어"), stock_row(species="932106", name="붕장어")],
        tmp_path,
    )
    comparison = report.manifest["name_key_comparison"]
    assert comparison["flow_species_code_keys"] == 2
    assert comparison["flow_normalized_name_keys"] == 1
    assert comparison["keys_collapsed_by_name_collision"] == 1
    assert comparison["stock_name_key_duplicate_keys"] == 1


# --------------------------------------------------------------------------- #
# L. 파일 입출력 · raw 안전성
# --------------------------------------------------------------------------- #

def test_encoding_detection_uses_the_whole_payload():
    assert detect_encoding("가오리류".encode("cp949")) == "cp949"
    assert detect_encoding("가오리류".encode("utf-8")) == "utf-8"
    assert detect_encoding("가오리류".encode("utf-8-sig")) == "utf-8-sig"


def test_reading_raw_csv_does_not_modify_the_file(tmp_path):
    path = tmp_path / "입출고.CSV"
    payload = ",".join(FLOW_HEADER) + "\n" + ",".join(flow_row()) + "\n"
    path.write_bytes(payload.encode("cp949"))
    before = file_digest(path)
    frame, encoding = read_raw_csv(path)
    assert encoding == "cp949"
    assert list(frame["source_row"]) == [2]
    assert file_digest(path) == before
    assert path.read_bytes().decode("cp949") == payload


def test_end_to_end_run_verifies_raw_files_are_unchanged(tmp_path):
    flow_path = tmp_path / "해양수산부_수협조합창고품목별창고입출고현황.CSV"
    stock_path = tmp_path / "해양수산부_수협조합창고품목별창고재고현황.CSV"
    flow_path.write_bytes(
        (",".join(FLOW_HEADER) + "\n" + ",".join(flow_row()) + "\n").encode("cp949"))
    stock_path.write_bytes(
        (",".join(STOCK_HEADER) + "\n" + ",".join(stock_row()) + "\n").encode("cp949"))
    sources = WarehouseSources(flow_path, stock_path, tmp_path / "processed")
    digest_before = {"flow": file_digest(flow_path), "stock": file_digest(stock_path)}

    report, digests = run_from_sources(sources)
    assert report.manifest["raw_unchanged"] is True
    assert digests == digest_before
    assert {"flow": file_digest(flow_path), "stock": file_digest(stock_path)} == digest_before


def test_write_outputs_refuses_to_overwrite_by_default(tmp_path):
    report = build([flow_row()], [stock_row()], tmp_path)
    processed = tmp_path / "processed"
    written = write_outputs(report, processed)
    assert written["dataset"].name == DEFAULT_OUTPUT_NAME
    assert written["dataset"].read_bytes().startswith(b"\xef\xbb\xbf")  # UTF-8 BOM
    with pytest.raises(FileExistsError):
        write_outputs(report, processed)
    write_outputs(report, processed, overwrite=True)


def test_status_file_marks_the_old_zero_percent_join_as_invalid(tmp_path):
    report = build([flow_row()], [stock_row()], tmp_path)
    processed = tmp_path / "processed"
    written = write_outputs(report, processed)
    status = json.loads(written["status"].read_text(encoding="utf-8"))
    assert status["valid_outputs"] == [DEFAULT_OUTPUT_NAME]
    superseded = status["superseded_outputs"]
    assert superseded["suhyup_warehouse_inventory_flow_actual.csv"]["status"] == "invalid_join_key"
    assert superseded["suhyup_warehouse_inventory_flow_actual_final.csv"]["status"] == (
        "superseded_ambiguous_name_key")
    assert set(superseded) == set(SUPERSEDED_OUTPUTS)


def test_locate_sources_returns_none_without_real_data(tmp_path, monkeypatch):
    monkeypatch.setenv("VARO_REAL_DATA_DIR", str(tmp_path))
    assert locate_warehouse_sources() is None


def test_locate_sources_finds_both_files(tmp_path, monkeypatch):
    raw = tmp_path / "03_Korea_Suhyup_Warehouse" / "raw"
    raw.mkdir(parents=True)
    (raw / "해양수산부_수협조합창고품목별창고입출고현황_20260731.CSV").write_text("x", encoding="utf-8")
    (raw / "해양수산부_수협조합창고품목별창고재고현황_20260731.CSV").write_text("x", encoding="utf-8")
    monkeypatch.setenv("VARO_REAL_DATA_DIR", str(tmp_path))
    sources = locate_warehouse_sources()
    assert sources is not None
    assert "입출고" in sources.flow_path.name
    assert "재고" in sources.stock_path.name


# --------------------------------------------------------------------------- #
# M. 실데이터 회귀 (원본이 있을 때만)
# --------------------------------------------------------------------------- #

def _real_sources():
    return locate_warehouse_sources()


real_data = pytest.mark.skipif(
    _real_sources() is None,
    reason="수협 조합창고 실데이터가 이 환경에 없습니다 (VARO_REAL_DATA_DIR).",
)


@real_data
def test_real_warehouse_join_matches_the_verified_numbers():
    """실제 원본에서 나오는 숫자가 검증 시점과 같은지 확인한다."""
    report, _digests = run_from_sources(_real_sources())
    metrics = report.join.metrics
    relation = report.manifest["product_code_relation"]

    assert metrics["flow_source_rows"] == 8439
    assert metrics["stock_source_rows"] == 42237
    assert relation["exact_full_code_overlap"] == 0          # 코드 직접 join 불가
    assert relation["name_mismatches"] == 0                  # 계층은 이름으로 검증됨
    assert relation["shared_species_codes"] == 223

    assert metrics["stock_key_uniqueness"]["duplicate_keys"] == 0
    assert metrics["flow_event_key_uniqueness"]["unique_keys"] == 8439
    assert metrics["flow_aggregated_keys"] == 7059
    assert metrics["flow_matched_keys"] == 7057
    assert metrics["flow_unmatched_keys"] == 2
    assert metrics["ambiguous_rows"] == 0
    assert metrics["cardinality"] == {
        "one_to_one": 7057, "one_to_many": 0, "many_to_one": 0, "many_to_many": 0}
    assert metrics["row_expansion_factor"] < 1.001
    assert metrics["output_rows"] == 42239
    assert report.manifest["validation_status"] == "사용 가능"


@real_data
def test_real_data_name_key_difference_is_explained():
    """과거 진단의 공통 흐름키 7,053과 어종코드 키 7,059의 차이를 수치로 설명한다."""
    report, _digests = run_from_sources(_real_sources())
    comparison = report.manifest["name_key_comparison"]
    assert comparison["flow_normalized_name_keys"] == 7053
    assert comparison["keys_dropped_for_missing_product_name"] == 2
    assert comparison["keys_collapsed_by_name_collision"] == 4
    assert (comparison["flow_normalized_name_keys"]
            + comparison["keys_dropped_for_missing_product_name"]
            + comparison["keys_collapsed_by_name_collision"]) == comparison["flow_species_code_keys"]
    assert comparison["stock_name_key_duplicate_keys"] == 66


@real_data
def test_real_data_balance_diagnostic_supports_end_of_day_snapshot():
    report, _digests = run_from_sources(_real_sources())
    balance = report.manifest["stock_flow_balance_diagnostic"]
    assert balance["comparable_day_pairs"] == 40842
    assert balance["consistent_rate"] > 0.999
    # 입출고 기록이 없는 날은 재고가 전혀 변하지 않는다 → 0 채움이 관측과 일치한다.
    assert balance["pairs_without_flow_record"] == balance["pairs_without_flow_record_consistent"]


@real_data
def test_real_data_has_no_transfer_history_or_logistics_facts():
    report, _digests = run_from_sources(_real_sources())
    provenance = report.manifest["provenance"]
    for key in ("interwarehouse_transfer_history", "distance", "travel_time",
                "transfer_cost", "vehicle_capacity", "weight_kg"):
        assert provenance[key] == PROVENANCE_NOT_AVAILABLE


@real_data
def test_real_data_run_is_reproducible():
    first, _ = run_from_sources(_real_sources())
    second, _ = run_from_sources(_real_sources())
    pd.testing.assert_frame_equal(first.output, second.output)

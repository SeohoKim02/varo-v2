"""수협 조합창고 재고 + 입출고 실데이터 결합 검증 도구.

원본 두 CSV(입출고/재고)를 읽기 전용으로 열어 결합 키가 안전한지 검증하고, 안전한 경우에만
결합본과 manifest를 processed 폴더에 만든다.

    python tools/validate_suhyup_warehouse_join.py                # 검증만
    python tools/validate_suhyup_warehouse_join.py --write        # 산출물 생성
    python tools/validate_suhyup_warehouse_join.py --write --overwrite

원본 위치는 ``VARO_REAL_DATA_DIR`` 또는 ``--data-root``로 지정한다. 출력에는 원본 전량이나
개인 식별 정보를 찍지 않고 집계 수치만 보여준다. 종료코드는 결합이 안전할 때만 0이다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.suhyup_warehouse_dataset import (  # noqa: E402
    STATUS_UNUSABLE, SUPERSEDED_OUTPUTS, locate_warehouse_sources, run_from_sources, write_outputs,
)


def _print_section(title: str) -> None:
    print(f"\n[{title}]")


def report(manifest: dict, blockers: list[str]) -> None:
    metrics = manifest["key_metrics"]
    relation = manifest["product_code_relation"]

    _print_section("원본")
    for source in manifest["raw_sources"]:
        print(f"  {source['role']}: {source['file_name']} "
              f"({source['encoding']}, {source['rows']}행)")
    print(f"  raw 변경 없음: {manifest.get('raw_unchanged')}")

    _print_section("상품코드 관계")
    print(f"  표준코드 전체 일치 공통: {relation['exact_full_code_overlap']}")
    print(f"  flow 어종코드 {relation['flow_species_codes']} / "
          f"stock 어종코드 {relation['stock_species_codes']} / "
          f"공통 {relation['shared_species_codes']}")
    print(f"  이름으로 계층 검증: 일치 {relation['name_agreements']} / "
          f"불일치 {relation['name_mismatches']} (검증 {relation['verified']})")

    _print_section("키 유일성")
    for label, key in (("flow event 키", "flow_event_key_uniqueness"),
                       ("flow 어종 키", "flow_species_key_uniqueness"),
                       ("stock 키", "stock_key_uniqueness")):
        stats = metrics[key]
        print(f"  {label}: 행 {stats['rows']} / 유일키 {stats['unique_keys']} / "
              f"중복키 {stats['duplicate_keys']} / 최대 다중도 {stats['max_multiplicity']}")

    _print_section("결합")
    cardinality = metrics["cardinality"]
    print(f"  flow 집계 키 {metrics['flow_aggregated_keys']} / stock 키 {metrics['stock_keys']} / "
          f"공통 {metrics['common_keys']}")
    print(f"  1:1 {cardinality['one_to_one']} · 1:N {cardinality['one_to_many']} · "
          f"N:1 {cardinality['many_to_one']} · N:M {cardinality['many_to_many']}")
    print(f"  matched {metrics['flow_matched_keys']} / flow 미결합 {metrics['flow_unmatched_keys']} / "
          f"stock 단독 {metrics['stock_unmatched_rows']} / 모호 {metrics['ambiguous_rows']}")
    print(f"  결합률(flow 기준) {metrics['join_rate_flow_side'] * 100:.4f} % · "
          f"(stock 기준) {metrics['join_rate_stock_side'] * 100:.4f} %")
    print(f"  출력 행 {metrics['output_rows']} / 행 확장 계수 {metrics['row_expansion_factor']}")
    for status, count in sorted(metrics["match_status_counts"].items()):
        print(f"    - {status}: {count}")

    _print_section("과거 상품명 키와의 차이")
    comparison = manifest["name_key_comparison"]
    print(f"  어종코드 키 {comparison['flow_species_code_keys']} = "
          f"상품명 키 {comparison['flow_normalized_name_keys']} "
          f"+ 상품명 없는 키 {comparison['keys_dropped_for_missing_product_name']} "
          f"+ 이름 충돌로 합쳐진 키 {comparison['keys_collapsed_by_name_collision']}")
    print(f"  상품명 키를 쓰면 stock 쪽 중복키 {comparison['stock_name_key_duplicate_keys']}건이 생긴다.")

    _print_section("재고 수지 참고 진단")
    balance = manifest["stock_flow_balance_diagnostic"]
    print(f"  비교 가능한 연속일 쌍 {balance['comparable_day_pairs']} / "
          f"일치 {balance['consistent']} ({balance['consistent_rate'] * 100:.4f} %)")
    print(f"  입출고 기록 있는 쌍 {balance['pairs_with_flow_record']} → "
          f"일치 {balance['pairs_with_flow_record_consistent']}")
    print(f"  입출고 기록 없는 쌍 {balance['pairs_without_flow_record']} → "
          f"일치 {balance['pairs_without_flow_record_consistent']}")

    _print_section("이상치")
    for flag, count in manifest["anomalies"].items():
        print(f"  {flag}: {count}")

    _print_section("범위")
    scope = manifest["scope"]
    print(f"  조합 {scope['coops']} · 창고 {scope['warehouses']} · 어종 {scope['species']} · "
          f"날짜 {scope['dates']} ({scope['date_min']} ~ {scope['date_max']})")

    _print_section("판정")
    print(f"  상태: {manifest['validation_status']}")
    for note in manifest["validation_notes"]:
        print(f"  - {note}")
    for blocker in blockers:
        print(f"  차단: {blocker}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="VARO_V2_REAL_DATA 폴더 경로 (미지정 시 자동 탐색)")
    parser.add_argument("--write", action="store_true", help="processed 산출물과 manifest 생성")
    parser.add_argument("--overwrite", action="store_true", help="같은 이름의 산출물을 다시 만든다")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="산출물 폴더 (기본: 데이터셋의 processed 폴더)")
    args = parser.parse_args()

    sources = locate_warehouse_sources(args.data_root)
    if sources is None:
        print("수협 조합창고 원본을 찾지 못했습니다. "
              "VARO_REAL_DATA_DIR 환경변수나 --data-root로 실데이터 폴더를 지정하세요.")
        return 2

    print(f"프로젝트: {PROJECT_ROOT}")
    print(f"원본 폴더: {sources.flow_path.parent}")
    result, _digests = run_from_sources(sources)
    report(result.manifest, result.join.blockers)

    _print_section("과거 산출물 상태")
    for name, info in SUPERSEDED_OUTPUTS.items():
        print(f"  {name}: {info['status']}")

    if args.write:
        if result.manifest["validation_status"] == STATUS_UNUSABLE:
            print("\n결합이 안전하지 않아 산출물을 만들지 않았습니다.")
            return 1
        target_dir = args.output_dir or sources.processed_dir
        try:
            written = write_outputs(result, target_dir, overwrite=args.overwrite)
        except FileExistsError as error:
            print(f"\n{error}\n--overwrite 를 붙이면 같은 이름으로 다시 만듭니다.")
            return 1
        _print_section("생성 파일")
        for role, path in written.items():
            print(f"  {role}: {path}")

    return 1 if result.join.blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())

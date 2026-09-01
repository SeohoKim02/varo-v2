# Varo V2 검증 문서

## 테스트 구조

- 프레임워크: `unittest` 기반, `pytest`로 실행. 테스트는 `tests/`에 있습니다.
- 자체 완결: 외부 원본 폴더/백업 없이 in-package 픽스처(`tests/fixtures.py`)로 워크북을 코드로 생성.
- 실행:

```bash
python -m pytest -q
python -m compileall -q app_v2.py router.py styles.py components services pages simulation
```

## 파일 검증 계층

업로드는 세 계층으로 안전하게 처리됩니다.

1. **파일 읽기 안전 계층** (`services/file_reader.py`): 확장자 허용(.xlsx/.xls/.csv), 빈/손상/암호화
   파일 차단, CSV 인코딩 자동 인식(UTF-8/BOM/EUC-KR·CP949). 라이브러리 오류를 사용자용 짧은 메시지로 변환.
2. **시트·컬럼 검증** (`services/data_validator.py`): 필수 시트/컬럼, node_type(DC/STORE),
   route_type(DIRECT/VIA_DC), VIA_DC의 DC 존재, 숫자 음수/비수치, ID 존재 등. 오류/경고/정상 판정.
3. **행 단위 점검** (`services/data_issues.py`): 원본 파일의 행·컬럼·값 기준으로 구조화.
   화면에는 오류/경고/사용 가능/제외 행 수와 상위 5개만 보여주고, 전체는 접힌 표 + CSV로 제공.

## 원본 snapshot과 정규화 데이터 분리

- **원본 snapshot**: 업로드 시 각 시트를 정규화 전 상태 그대로 보존합니다
  (`data_loader.load_excel_data` → `report["raw_sheets"]`, 원본 컬럼명·원본 값·원본 행 순서).
  세션에는 `raw_data`(적용) / `pending_raw_data`(대기)와 출처 정보 `source_metadata`
  (`filename`·`source_type`(excel/csv/sample)·`sheet_names`)로 보관됩니다.
- **정규화 데이터**: 분석·검증에만 사용하는 표준 컬럼/숫자 변환 결과(`varo_data`).
- **분석 파이프라인은 정규화 데이터만** 받습니다. 원본 snapshot은 오직 오류 위치·값 추적과
  수정 안내에 쓰이며, 분석 계산에 직접 사용되지 않습니다.

## 원본 행 번호 기준

- 스프레드시트 1-based 기준: 헤더가 1행이므로 첫 데이터 행은 **2행**입니다.
- 원본 snapshot은 읽은 순서를 그대로 보존하므로 위치 i는 파일 i+2행에 대응합니다.
- **완전히 빈 행은 문제로 보고하지 않되, 아래 행 번호를 밀지 않습니다**(빈 행 다음 행이 계속 원래 번호 유지).
- 화면에는 pandas index가 아니라 파일 기준 행 번호를 표시합니다.

## 행 단위 검증

각 문제는 `issue_code`, `severity`(오류/경고), `source_type`, `source_sheet`,
`source_row_number`, `source_column_name`(원본 컬럼), `canonical_column_name`(표준 컬럼),
`original_value`(전체 원본 값), `normalized_value`, `blocks_analysis`, `related_rows`를 갖습니다.

- 필수 식별자 공백(`missing_id`), 비수치(`non_numeric`), 음수(`negative`), 이동 수량 0(`zero_quantity`),
  출발지=도착지(`same_source_target`) → **오류 · 분석 차단**.
- 안전재고·최소재고·재주문점·목표재고가 있으면 숫자 변환 가능·유한값·0 이상인지 같은 체계로 확인합니다.
  문제 목록에는 원본 행·원본 컬럼·입력값과 “0 이상의 숫자로 수정하세요.”를 표시합니다.
- 값 충돌 중복(`conflict_duplicate`)과 완전 동일 중복(`exact_duplicate`)은 행 제외 대상으로 기록합니다.
- 컬럼 별칭 충돌(`alias_conflict`)은 어떤 원본 컬럼이 맞는지 결정할 수 없어 **파일 구조 차단 오류**입니다.
- 식별자 숫자화(`id_numeric`, 앞자리 0 손실 가능)는 행을 유지하는 경고입니다.
- **원본 값 보존**: 숫자 변환 실패 시 원본 문자열(예: "십오")을 그대로 보여주고 정규화 값은 "변환 실패"로
  표시합니다. 실패 값을 임의로 0으로 만들지 않습니다. 빈 값은 `nan`/`None`이 아니라 "빈 값"으로 표시합니다.
- **컬럼 충돌**: 서로 다른 원본 컬럼이 같은 표준 컬럼(표준 컬럼 자체가 없을 때)으로 인식되면 원본 컬럼명을
  모두 표시하고, 임의로 하나를 고르지 않습니다.
- **중복**: 관련된 모든 원본 행 번호(`related_rows`)를 기록합니다. 값 충돌 중복은 관련 행을 모두 제외하고,
  완전히 같은 중복은 첫 행만 유지합니다. 날짜/시점 컬럼으로 구분되는 정상 시계열 다중 행은 유지합니다.
- 같은 점포×상품의 재고가 같아도 등록 안전재고/최소재고/재주문점/목표재고가 다르면 값 충돌 중복으로
  처리합니다. 어느 운영 정책도 임의로 선택하지 않습니다.

## 심각도와 처리 정책 분리

- `ISSUE_POLICY` 한 곳에서 `severity`, `blocks_analysis`, `scope`, `row_excludable`,
  `retain_after_warning`, 사용자 문제/수정 문구를 관리합니다. severity만으로 제거 여부를 결정하지 않습니다.
- 처리는 `file_blocking`, `row_excludable`, `row_warning`, `informational`로 나뉩니다. 알 수 없는 코드는
  안전하게 파일 차단으로 처리합니다.

## 문제 행 제외 정책

- 제외 집합은 pandas 위치가 아니라 `(source_sheet, source_row_number)`로 만들며 한 행의 여러 문제를 한 번만
  셉니다. 기준정보 제거로 고아가 되는 재고·경로·추천도 같은 1회 제외 집합에 추가합니다.
- 값 충돌 중복은 관련 행 전체, 완전 동일 중복은 첫 행 이외를 제외합니다.
- 제외 후 `validate_workbook_data`를 한 번 다시 실행합니다. 오류가 남으면 반복 삭제로 통과시키지 않습니다.
- `HEAVY_EXCLUSION_RATIO=0.5`는 전체 분석 행과 각 필수 테이블에 모두 적용합니다. 50%와 그 이상은 차단합니다.
  별도로 STORE 1개·DC 1개·상품·재고·경로가 남아야 합니다. 후보 0건 자체는 적용 차단 사유가 아닙니다.

## 분석 실행 게이트

- **UI**: 구조 오류 또는 제외 후 재검증 오류가 있으면 사용 불가 pending으로만 보관됩니다. 행 단위 오류는
  정책에 따라 정제본에서 제외하며, 재검증을 통과한 정제본만 적용할 수 있습니다.
- **파이프라인**: `run_analysis_pipeline`은 `validate_workbook_data`가 오류를 반환하면 상태
  `validation_error`로 즉시 중단하고 추천을 만들지 않습니다. **직접 함수 호출에서도** 게이트가 작동합니다.
- 실행 불가능한 개별 후보는 실행 가능성 게이트(`services/feasibility.py`)가 최종 추천에서 제외합니다.

## 데이터 signature와 상태 분리

- signature는 파일/후보 내용 해시. 데이터가 바뀌면 `apply_state_payload`가 이전 추천·선택·경로 상세·
  시뮬레이션·Greedy/Pareto/민감도/신뢰도·**DQN 반영 상태**를 초기화합니다.
- signature가 다른 DQN 결과는 현재 추천에 반영하지 않습니다(모델 파일은 삭제하지 않음).
- 샘플과 업로드 데이터는 `data_source_type`으로 구분되어 섞이지 않습니다.

## 후보 판단 기록과 원본 계보

- 구현: `services/candidate_ledger.py` + `services/candidate_lineage.py`.
  파이프라인은 실행 가능성 게이트의 **전체 주석 결과**(feasible + blocked)로 후보 기록을 만들어
  `PipelineResult.candidate_ledger`/`excluded_candidates`/`ledger_summary`에 담습니다. 제외 후보가
  사라지지 않고 접힌 영역에서 확인됩니다.
- **데이터 오류 ↔ 후보 제외 계보 일치.** 후보의 원본 행 참조는 데이터 관리 문제 목록과 **같은 함수**
  (`data_issues._resolve_column`/`_row_number`)로 계산합니다. 따라서 "재고현황 12행 값 오류"(데이터 관리)와
  "출발 재고를 계산할 수 없어 제외"(추천)가 같은 원본 행을 가리킵니다. `tests/test_candidate_ledger.py`의
  계보 테스트가 행 번호(예: S1/P1 → 파일 2행)를 고정 검증하고, `tests/test_page_render.py`의
  end-to-end 테스트가 실제 업로드 흐름에서 같은 행을 확인합니다.
- **원본 행 계보 구조.** 후보당 참조: 출발 재고·도착 수요·상품 정보·경로 정보(VIA_DC 2행)·DC 정보.
  각 참조는 `traceable` 여부를 표시하고, 찾지 못하면 임의 행을 만들지 않고 `추적 불가`로 남깁니다.
  계보 메타데이터는 VHS·수치형 피처에 섞이지 않습니다.
- **화면 간 상태·건수 일치.** `ledger_summary`의 버킷은 가산적이고(`추천 후보+확인 필요+이동 불가+데이터
  부족+계산 불가 == 전체 생성 후보 수`), `이동 불가+데이터 부족+계산 불가`는 실행 가능성 게이트의
  `blocked_count`와, `추천 후보`는 `ok_count`와, `확인 필요`는 `check_count`와 일치합니다.
  같은 후보는 홈·추천 실행·경로 상세·검증에서 같은 `candidate_id`와 같은 상태를 사용합니다.
- **데이터 변경 시 계보 제거.** 후보 기록은 파이프라인 결과 안에 있고, `apply_state_payload`가
  파이프라인 결과 전체를 교체하므로 데이터가 바뀌면 이전 계보가 사라집니다. `candidate_id`는
  data signature를 포함해 다른 데이터의 후보와 절대 섞이지 않습니다.
- **후보 판단 기록 구조.** 세션 상태가 커지지 않도록 원본 프레임 사본이 아니라 **참조 정보**(행 번호,
  소수의 값)만 저장합니다. 핵심 필드: `candidate_id`, `status`, `status_code`, `blocks_recommendation`,
  `quantity_basis`, `recommendation_reasons`, `exclusion_reasons`, `source_references`, `confidence`,
  `stability`, `calculated_at`.
- **후보 검토 CSV.** `review_candidates_csv_bytes`가 제외·확인 필요 후보만 UTF-8 BOM CSV로 내보냅니다.
  내부 ID·경로·traceback·session_state 키·모델 경로·signature 원문을 포함하지 않습니다(검토용, 제출 보고서 아님).

## 홈 상태 판정과 결과 일관성

- 구현: `services/home_state.build_home_state`. 홈 화면은 여러 UI 조건문으로 상태를 추정하지 않고
  **하나의 순수 함수**가 실제 세션 상태를 읽어 단일 상태를 반환합니다(never raises — 입력이 부족하면
  가짜 정상 상태 대신 `데이터 없음/확인 필요`로 안전하게 처리).
- **판정 입력(실제 상태만).** pending 데이터(`pending_load_error`/`pending_varo_validation`), 적용 데이터
  (`varo_data`의 stores), 검증 결과(`varo_validation`), 적용 서명(`data_signature`), 파이프라인 결과
  (`candidate_ledger`·`ledger_summary`·`diagnostics`·`confidence_status`), 선택 후보(`selected_route_id`).
  샘플 수치나 화면용 가짜 상태는 만들지 않습니다.
- **우선순위(2단계 반영).** 적용 데이터 없음 + 사용 불가 pending → 사용 불가; 적용 데이터 없음 +
  검사만 된 pending → "검사한 데이터를 적용하세요"; 적용 데이터 없음 → 데이터 없음 → 서명 불일치(stale)
  → 분석 실패 → 추천 있음/후보 0건. **적용된 정상 데이터가 있으면** 새 pending(정상/경고/사용 불가)은
  workspace 상태를 바꾸지 않고 `pending_notice`로만 알립니다(사용 불가 pending이 현재 결과를 숨기지 않음).
- **업로드 적용과 분석 실행 분리.** `commit_pending_data`는 최종 usable data만 적용하고 과거 결과를
  초기화한 뒤 `analysis_pending` 상태로 둡니다. 사용자가 추천 실행 페이지의 버튼을 눌러야
  `run_applied_analysis`가 usable data로 파이프라인을 실행합니다. 빠른 샘플 경로만 기존 일괄 실행을 유지합니다.
- **서명 불일치 시 과거 결과 숨김.** 적용 서명과 후보 기록(`candidate_ledger`)의 서명이 다르면 `stale`로
  판정해 과거 KPI·최우선 추천을 감춥니다. 정상 흐름에서는 `apply_state_payload`가 결과와 서명을 원자적으로
  교체하므로 항상 일치하며, 이 검사는 안전망입니다.
- **후보 0건 원인.** `ledger_summary.top_exclusion_reasons`로 상위 1~2개 원인만 사용자 문장으로 요약하고,
  내부 reason_code는 홈에 노출하지 않습니다. 상세는 추천 실행 페이지의 제외 후보 영역에서 확인합니다.
- **홈 KPI와 ledger 일치.** 결과 KPI는 추천 결과가 있을 때만 표시하며 최우선 추천은 공유 랭킹
  (`top_recommendations`)의 rank-1을 그대로 사용합니다(홈에서 재정렬·재계산 없음). `tests/test_home_state.py`가
  홈 top과 공유 랭킹 top의 일치, 상태별 KPI 노출 여부, 우선순위, never-raise를 검증하고,
  `tests/test_page_render.py`가 각 상태의 제목·단일 행동 버튼·과거 결과 숨김·내부 정보 미노출을 확인합니다.

## 분석·검증 페이지 상태 공유

- **홈과 같은 단일 기준.** `pages/validation.render_validation_page`는 진입 시 `build_home_state`를
  호출해 홈과 동일한 상태를 얻습니다. `varo_recommendations` 유무 같은 개별 플래그로 다시 판정하지 않으며,
  후보 수도 DataFrame 길이가 아니라 `ledger_summary`에서 가져옵니다.
- **상태 = READY일 때만 6개 알고리즘 탭.** 데이터 없음·사용 불가·stale·실패에서는 상태 카드 하나와
  다음 행동 하나만 렌더하고 탭을 만들지 않으므로, 분석 전/실패/stale에 빈 표·빈 차트·0으로 채운 검증 KPI가
  나타나지 않습니다. 공유 상태 카드는 `components/state_banner.render_state_action_card`(홈·검증 공용)로 렌더합니다.
- **후보 0건은 데이터 없음이 아니다.** 데이터가 적용됐지만 추천 가능한 이동이 0건이면(`state_code == no_candidates`)
  "데이터 없음"이 아니라 `_render_candidate_status`(전체 생성 후보 + 추천/확인 필요/이동 불가/데이터 부족/계산 불가
  카드 + 주요 제외 이유 + 제외 후보 목록)를 표시합니다. 상태 카드 합계는 `ledger_summary.generated`와 일치합니다.
- **stale·failed·unusable 처리.** 서명 불일치는 과거 검증 결과를 숨기고 재적용을 안내하며(stale), 기술 오류로
  후보가 0건이면 후보 0건과 구분해 "분석을 완료하지 못했습니다"(failed), 사용 불가 업로드(pending)는 이전 정상
  결과보다 우선해 검증 결과를 숨깁니다(unusable).
- **DQN은 선택형.** DQN 미실행은 전체 검증 실패로 판정하지 않습니다. 정상 결과에서 안정성·신뢰도는 DQN 없이
  계산되고, DQN 결과 서명이 현재 데이터와 다르면 반영하지 않고 그 사실만 표시합니다.
- **페이지 간 일관성.** 홈·추천 실행·검증이 같은 `candidate_ledger`/`ledger_summary`/`home_state`를 사용하므로
  전체 후보 수·상태별 수·후보 0건 원인이 일치합니다. `tests/test_home_state.py`의 교차 페이지 일관성 테스트와
  `tests/test_page_render.py`의 검증 상태 렌더 테스트가 이를 검증합니다.

## 검사와 적용 분리 (2단계 intake)

- **inspect vs apply.** UI 업로드 경로는 `data_application.prepare_pending_data`로 파일을 *검사만* 합니다:
  읽기 → 원본 snapshot 보존 → 전체 정규화 → 공통 정책 분류 → 원본 행 제외 집합 → 참조 행 정리 →
  최종 usable data → 1회 재검증 → source/usable signature 생성 → `pending_*` 저장. **여기서는 `varo_data`·`data_signature`·
  추천·candidate ledger를 절대 건드리지 않습니다.** 반환값은 상태 코드(사용 가능/확인 필요/사용 불가/현재
  데이터와 동일). 실제 적용은 사용자가 버튼을 눌러 `commit_pending_data`를 호출할 때만 일어납니다.
  (샘플/빠른 시작과 기존 프로그램적 호출은 한 번에 적용하는 `load_and_apply`를 계속 사용합니다.)
- **pending intake payload.** `pending_varo_data`(전체 정규화)·`pending_usable_data`(적용 가능 정제본)·
  `pending_raw_data`(원본)·`pending_varo_validation`·`pending_data_issues`·`pending_excluded_row_refs`·
  `pending_quality_summary`·`pending_source_signature`·`pending_usable_signature`·
  `pending_source_metadata`·`pending_data_signature`·`pending_uploaded_filename`·`pending_data_source_type`·
  `pending_recommendation_source`·`pending_apply_allowed`·`pending_usable_rows`·`pending_excluded_rows`·
  `pending_status`·`pending_created_at`·`pending_upload_report`. 과거 추천·ledger·선택·경로·시뮬레이션은
  pending에 포함되지 않으며, pending 데이터는 적용 전까지 분석 파이프라인에 전달되지 않습니다.
- **적용 전 최종 재검증.** `commit_pending_data`는 usable data·검증·source/usable signature·제외 집합과
  집계의 일치, `pending_apply_allowed`, 분석 사용 행, stores 존재를 재확인하고 usable signature를 다시 계산합니다.
  누락·불일치면 "검사 결과가 만료됐습니다"로 중단하고 적용하지 않습니다(내부 키/예외명 미노출).
- **원자적 적용.** 재검증 통과 시 usable data와 원본 감사 계보·품질 요약을 payload로 완성한 뒤
  `apply_state_payload` **한 번**으로 반영합니다. 성공 후에만 이전 추천·VHS·Greedy·Pareto·
  민감도·신뢰도·candidate ledger·선택 후보·경로 상세·시뮬레이션·현재 데이터와 다른 DQN 반영이 초기화됩니다.
- **적용 실패 롤백/보존.** signature 불일치·재검증 오류·적용 예외가 나면 현재 적용 데이터·추천·pending을 모두
  보존하고 `data_apply_error`에 사용자용 짧은 메시지만 둡니다(traceback은 `logging.exception`으로만 기록).
  일부만 바뀐 혼합 상태를 만들지 않습니다.
- **signature 동일·변경.** 내용 해시가 현재 적용 데이터와 같으면 pending 상태를 "현재 데이터와 동일"로 두고,
  commit해도 결과를 다시 지우지 않습니다. 해시가 다르면 새 pending으로 처리하고 적용 시에만 과거 분석을
  초기화합니다. 파일명이 아니라 내용으로 판단하므로 같은 이름·다른 내용은 새 데이터로 인식합니다.
- **workspace 상태 vs pending intake 상태.** `home_state`는 검사 중 데이터를 *workspace* 상태로 승격하지
  않습니다. 적용된 정상 데이터가 있으면 새 pending(정상/경고/사용 불가)이 있어도 홈·추천 실행·검증은 현재
  결과를 유지하고, `pending_notice`로 "검사 완료된 새 데이터가 있습니다"만 알립니다. 사용 불가 pending이
  workspace 전체를 사용 불가로 만드는 것은 **적용된 데이터가 없을 때뿐**입니다. 검증/구현은
  `tests/test_two_phase_apply.py`와 `tests/test_page_render.py`(apply/cancel element tree)가 담당합니다.

## 데이터 관리 페이지 상태 공유

- **home_state와 같은 단일 기준.** `pages/data_management`는 `services/data_management_view.build_data_management_view`
  로 상태를 얻고, 이 함수는 내부에서 `build_home_state`를 호출합니다. 그래서 데이터 관리의 상태 제목·짧은
  설명·상태 등급이 홈·추천 실행·검증과 항상 일치합니다(페이지별 개별 boolean 재판정 없음). 데이터 관리에만
  필요한 상세(현재/검사 중 구분, 행 수 점검, 파일·샘플 선택)만 view가 추가로 계산합니다.
- **현재 적용 데이터와 검사 중 데이터 분리.** view는 `varo_data`의 stores 유무로 현재 적용 데이터를,
  `pending_*`로 검사 중 데이터를 **독립적으로** 판정합니다. pending은 사용 가능·확인 필요·사용 불가를 모두
  가질 수 있고, 버튼을 누르기 전에는 기존 정상 적용 데이터와 추천을 변경하지 않습니다.
- **적용의 원자성.** 적용은 `apply_state_payload` 한 번으로 정규화 데이터·원본 계보·source metadata·
  signature를 넣고, 이전 추천·candidate ledger·선택 후보·경로 상세·시뮬레이션·민감도/신뢰도·DQN 반영
  상태를 함께 초기화합니다(일부만 적용되고 나머지가 남는 상태가 없음).
- **적용 실패 시 기존 데이터 보존.** `load_and_apply`는 로드/검증 실패를 사용자용 짧은 메시지로 바꾸고
  `pending_*`에만 저장하며 기존 적용 상태를 건드리지 않습니다. 예외의 클래스명·traceback·내부 경로는
  화면에 노출하지 않고 로그로만 남깁니다(`logging`).
- **현재 데이터 초기화.** `services/app_state.clear_applied_data`는 적용 데이터와 파생 결과(추천·ledger·
  선택·경로 상세·시뮬레이션·분석)와 검사 중 데이터만 비웁니다. 사용자의 원본 파일은 삭제하지 않고,
  초기화 후 워크스페이스는 "데이터 없음"으로 읽힙니다. 검증은 `tests/test_data_management_view.py`가 담당합니다.
- **네 페이지 일관성.** 홈·추천 실행·검증·데이터 관리가 같은 `home_state`/`candidate_ledger`/`ledger_summary`를
  사용하므로 현재 데이터 존재·출처·pending·사용 불가·signature·분석 실행·후보 수·stale·다음 행동이
  일치합니다. `tests/test_data_management_view.py`(교차 페이지 상태 일치)와 `tests/test_page_render.py`
  (데이터 관리 상태별 element tree)가 이를 검증합니다.

## 알고리즘 검증 방식

- **실행 가능성 게이트** (`tests/test_feasibility.py`): 출발지=도착지, 이동 수량 ≤0/NaN/inf,
  경로 없음, VIA_DC DC 없음, 재고 초과, 운영 재고 하한 침범, 중복 등 하드 블록과 소프트 플래그,
  정상 후보를 각각 검증.
- **운영 재고 정책** (`tests/test_inventory_policy.py`): 영문/한글 alias, 명시값 우선과 추정 fallback,
  명시 0/미입력 구분, 음수·inf·문자열·충돌 제외, 점포×상품 연결, 경계 수량, 목표재고 역할,
  provenance와 사용자 문구 비노출 계약을 검증합니다.
- **후보 판단 기록** (`tests/test_candidate_ledger.py`): 후보 식별자 안정성/분리, 원본 행 계보,
  제외 이유, 이동 수량 근거, 추천 이유, 화면 일관성(버킷 합·feasibility 일치), 상태 예외, 검토 CSV 안전성.
- **홈 상태 모델** (`tests/test_home_state.py`): 상태 판정(데이터 없음/사용 불가/후보 0건/실패/stale/정상),
  우선순위, 결과 KPI 위장 방지, 홈 top과 공유 랭킹 일치, garbage 입력 무예외.
- **VHS 점수** (`tests/test_vhs_*`, `test_algorithm_contracts.py`): 정규화 범위(0–100), 가중치 합 1.0,
  결측 중립값, 동일 입력 재현성, 구성요소 보존.
- **민감도** (`tests/test_pareto_sensitivity.py`, `weight_sensitivity`): 가중치 ±30% 섭동 시 Top1 유지율·취약 요소.
- **신뢰도/안정성** (`tests/test_decision_support.py`): 안정/검토 필요/불안정/계산 불가,
  DQN 없이 신뢰도 계산, DQN 부재가 신뢰도를 낮추지 않음.
- **파이프라인** (`tests/test_analysis_pipeline.py`): 샘플 기반 전체 흐름과 결과 구조.
- **페이지 렌더** (`tests/test_page_render.py`): 5개 페이지가 예외 없이 렌더, 탭/버튼/컬럼 계약,
  데이터 관리 상태별(없음/적용/사용 불가 pending+적용 동시) element tree와 현재 데이터 초기화.
- **데이터 관리 view** (`tests/test_data_management_view.py`): 현재/검사 중 구분, 홈과 상태 일치,
  행 수 vs 문제 항목 수, 적용 실패 시 기존 데이터 보존, 현재 데이터 초기화(원본 파일 미삭제), 내부 정보 미노출.

## DQN 품질 판정

- `services/dqn_quality.py`가 학습 **데이터** 품질(라벨 분포)을 진단: 정상/검토 필요/불안정/학습 부족.
- `services/dqn_service.evaluate_dqn_stability`가 학습 **결과**의 안정성(loss 유한성, 행동 쏠림, reward 분산)을 판정.
- 정상 + 데이터 signature 일치 + 안정성 통과일 때만 낮은 비중으로 참고 반영.

## original과 balanced 분리

- 원본 샘플과 균형형 파생 샘플의 학습 결과·모델·출력 파일을 섞지 않습니다.
- 파일명에 `original`/`balanced` 마커와 sample_id·store/dc·episodes·lr·timestamp를 포함.
- 균형형은 핵심 수치를 그대로 두고 target_action 라벨만 재분배(`services/dqn_balanced.py`), 원본은 무수정.

## 데이터 signature

- `dqn_service.data_signature_from_recommendations`가 현재 후보 집합의 해시를 만듭니다.
- 학습 결과의 signature가 현재 데이터와 다르면 "과거 결과"로 표시하고 **최종 반영을 차단**합니다.

## action 매핑 검증

- 공통 어휘 `dqn_service.ACTION_LABELS` 하나로 통일(학습 one-hot index = 예측 index = 표시명).
- `tests/test_dqn_service.py`: 숫자/한글/영문/별칭 정규화, 인덱스 왕복, 알 수 없는 값은 "확인 필요".

## 알려진 제한사항

- DQN 학습 샘플 10-pack은 외부 데이터로, 이 저장소에 없으면 통합 테스트
  (`tests/test_dqn_samples.py::RealDqnPackTests`)는 **명확한 사유로 skip**됩니다. 구조 검증은
  항상 실행되는 내부 픽스처(`DualDcStructureTests`, DC01/DC02 구분)로 대체됩니다.
- PyTorch가 없으면 DQN 학습은 "실행 환경 필요"로 표시되며 관련 테스트는 안전하게 우회됩니다.

## 재현 가능한 테스트 명령

```bash
# 전체
python -m pytest -q
# 핵심 알고리즘/의사결정
python -m pytest tests/test_feasibility.py tests/test_decision_support.py \
  tests/test_dqn_service.py tests/test_kpi_formatting.py -q
# 페이지 렌더
python -m pytest tests/test_page_render.py -q
# 운영 형식 익명화 데이터 end-to-end
python -m pytest tests/test_operational_validation.py tests/test_operational_ui_flow.py -q
```

## 운영 형식 익명화 데이터 검증

실제 운영 업로드와 같은 형태의 익명화 워크북(다중 점포·다중 상품·2개 DC·DIRECT/VIA_DC,
정상·유지 경고·제외 대상 행 혼합)으로 파일 업로드부터 추천 결과·화면 상태까지 실행한 기록은
[`docs/OPERATIONAL_VALIDATION.md`](OPERATIONAL_VALIDATION.md)에 있습니다. 데이터 구성, 기대값과
실제 결과, 성능 측정값, 발견·수정한 결함, 남은 한계, 재현 명령이 모두 그 문서에 있습니다.

```bash
python tools/generate_anonymized_operational_workbook.py   # 고정 seed로 워크북 + 기대값 manifest
python tools/run_operational_validation.py --repeats 3     # 실제 서비스로 end-to-end 검증
python tools/run_algorithm_benchmark.py                    # 명시 재고 하한 vs 추정 하한 비교 포함
```

기대값은 앱 코드나 화면이 아니라 `validation_data/*_manifest.json`으로만 관리합니다.

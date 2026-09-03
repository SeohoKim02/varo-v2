# Varo V2 데이터 형식

업로드 엑셀은 여러 시트로 구성됩니다. 컬럼명은 한글/영문 별칭으로 표준 컬럼에 자동 매핑됩니다
(`services/column_aliases.py`). 아래 필수/선택 컬럼은 `services/data_validator.py` 기준입니다.

## 지원 파일 형식

- **`.xlsx` · `.xls`**: 여러 시트를 담는 워크북. 권장 형식.
- **`.csv`**: 안전하게 읽되(UTF-8 / UTF-8 BOM / EUC-KR·CP949 자동 인식), 표 하나만 담을 수 있어
  4개 필수 시트(stores/products/inventory/routes)를 구성할 수 없으므로 Excel 워크북으로 안내합니다.
- 그 외 확장자·빈 파일·손상/암호화 파일은 읽기를 시도하지 않고 짧은 안내로 차단합니다
  (`services/file_reader.py`). 내부 예외는 로그로만 보관하고 화면에는 traceback을 노출하지 않습니다.

## 시트 선택

Varo 워크북은 시트를 **이름으로** 인식합니다(stores/products/inventory/routes/v2_recommendations 등).
"첫 시트를 임의 확정"하지 않으며, 필수 시트 이름이 없으면 어떤 시트가 없는지 안내합니다. 따라서
다중 시트에서 임의 선택 UI는 필요하지 않습니다(이름 기반 매칭).

## 시트 개요

| 시트 | 역할 | 필수 여부 |
|---|---|---|
| `stores` | 점포·DC 노드 | 필수 |
| `products` | 상품 | 필수 |
| `inventory` | 점포별 재고 | 필수 |
| `routes` | 이동 경로 | 필수 |
| `recommendations`(=`v2_recommendations`) | 사전 추천 | 선택(없으면 자동 생성) |
| `config`, `quality_check`, `readme` | 부가 정보 | 선택 |

## stores (점포/DC)

- **필수:** `node_id`, `node_name`, `node_type`
- `node_type`은 **`DC` 또는 `STORE`만** 허용. DC 수/점포 수는 이 컬럼으로 셉니다.
- **점포와 DC 구분:** DC01, DC02처럼 DC가 여러 개면 각각 `node_type=DC` 행으로 둡니다. DC끼리 섞이지 않습니다.

## products (상품)

- **필수:** `product_id`, `product_name`
- `product_id` 중복 불가. `unit_price` 등은 선택.

## inventory (재고)

- **필수:** `store_id`, `product_id`, `stock_qty`
- **권장(없으면 경고/제한):** `sales_qty`, 유통기한 계열(`expiry_days`/`expiry_date`/`shelf_life_days`/`days_to_expiry`),
  `demand_qty` 또는 `avg_daily_sales`, `dead_stock_qty`, `demand_std`, `lead_time_days`
- **운영 정책(모두 선택):** `safety_stock`, `min_stock`, `reorder_point`, `target_stock`.
  이 컬럼이 없어도 기존 파일은 그대로 동작합니다.
- `stock_qty`는 숫자·**음수 불가**.
- 실행 가능성 판정에 `stock_qty`(이동 후 음수 방지), `safety_stock`/`min_stock`(출발 재고 하한),
  `demand_qty`/`avg_daily_sales`(도착 수요), `demand_std`(명시 하한이 없을 때만 추정)를 사용합니다.

| 의미 | 표준 컬럼 | 허용 별칭 예 | 이동 하한 사용 |
|---|---|---|---|
| 등록 안전재고 | `safety_stock` | `safety_inventory`, `안전재고` | 예 |
| 최소 보유재고 | `min_stock` | `minimum_stock`, `minimum_inventory`, `min_inventory`, `stock_floor`, `safety_floor`, `최소재고`, `최소 보유량`, `최소 보유재고` | 예 |
| 재주문 트리거 | `reorder_point` | `reorder_level`, `재주문점` | 아니요(참고값) |
| 도착 목표 수준 | `target_stock` | `target_inventory`, `목표재고` | 아니요(도착 부족량에 사용) |

`safety_stock`과 `min_stock`이 모두 있으면 둘을 같은 필드로 덮어쓰지 않습니다. 두 등록 하한을 모두
지키기 위해 더 큰 값을 출발 하한으로 적용하고 출처는 결합 정책으로 기록합니다. `reorder_point`는 발주
트리거일 수 있고 `target_stock`은 도착 후 원하는 수준이므로 안전재고로 치환하지 않습니다.

## routes (경로)

- **필수:** `source_id`, `target_id`, `distance_km`, `estimated_cost`, `travel_time_min`
- `source_id`/`target_id`는 `stores.node_id`에 존재해야 합니다.
- 숫자 컬럼은 음수 불가. 동일 source/target 중복은 경고.
- **DIRECT vs VIA_DC:** DIRECT는 출발→도착 경로가 있어야 하고, VIA_DC는 출발→DC·DC→도착 경로가 모두 있어야 생성됩니다.

## recommendations (사전 추천, 선택)

시트 이름은 `v2_recommendations`. 없으면 재고·경로로 자동 생성합니다.

- **필수:** `route_id`, `product_id`, `product_name`, `source_id`, `source_name`, `target_id`, `target_name`,
  `route_type`, `recommended_qty`, `transport_type`, `estimated_cost`, `expected_saving`, `vhs_score`,
  `recommendation_grade`, `confidence_score`, `reason`
- `route_id`: 비어 있거나 중복 불가.
- `route_type`: **`DIRECT` 또는 `VIA_DC`만** 허용. VIA_DC 행에는 `dc_id`·`dc_name` 필요.
- `recommended_qty`: **0보다 커야 함**. `source_id`/`target_id`는 `stores.node_id`에 존재해야 함.

## 원본 컬럼 ↔ 표준 컬럼

- 컬럼명 앞뒤 공백·한글/영문 별칭은 표준 컬럼으로 자동 매핑됩니다(`column_aliases.py`).
  원본 컬럼은 삭제하지 않고 표준 컬럼을 복사해 추가합니다.
- 오류 표시는 **원본 컬럼명**(예: "현재고", "재고 수량 ")을 기준으로 하고, 표준 컬럼명(`stock_qty`)은
  상세/CSV에만 둡니다. 사용자가 파일에서 바로 찾을 수 있게 하기 위함입니다.
- **컬럼 별칭 충돌**: 표준 컬럼이 없는데 서로 다른 원본 컬럼 2개 이상이 같은 표준으로 인식되면
  임의로 하나를 고르지 않고 원본 컬럼명을 모두 표시(확인 필요). 표준 컬럼 자체가 있으면 그 컬럼이 우선하며
  충돌로 보지 않습니다.

## 식별자 보존 규칙

- 점포/상품/경로 ID는 숫자가 아니라 **문자열 식별자**로 다룹니다. `NUMERIC_COLUMNS`에 포함하지 않아
  `001`·`000123` 등 앞자리 0을 정규화 과정에서 제거하지 않습니다.
- 다만 **Excel이 파일 저장 시점에** 이미 ID를 숫자로 바꿔 앞자리 0을 잃은 경우는 라이브러리 단계 한계라
  완벽 복원할 수 없습니다. 이 경우 원본 컬럼이 숫자형이면 `id_numeric` **경고**로 알리고(복원한 척하지 않음),
  사용자가 원본을 텍스트 서식으로 저장하도록 안내합니다.

## 숫자 변환 규칙

- `"1,500"`·`"3.5km"`·`"15분"`·`"20개"`처럼 숫자를 뽑을 수 있으면 변환하고 정상 처리합니다.
- `"십오"`·`"열개"` 등 뽑을 수 없으면 **원본 값을 보존**하고 정규화 값을 "변환 실패"로 표시하며,
  핵심 수치이면 분석을 차단합니다. **실패 값을 0으로 만들지 않습니다.**
- 빈 값·실제 NaN은 `nan`/`None`이 아니라 "빈 값"으로 표시합니다.

## 행 번호 기준

- 스프레드시트 1-based(헤더 = 1행). 첫 데이터 행은 2행입니다.
- 완전히 빈 행은 문제로 보고하지 않되 이후 행 번호를 밀지 않습니다.
- 여러 시트가 있으면 문제 표에 원본 시트명(`source_sheet_name`)을 함께 표시합니다.

## 중복 키와 충돌 처리

- **중복 키**: 재고는 `store_id + product_id`, 경로는 `source_id + target_id (+ route_type/dc)`,
  추천은 `route_id`.
- **완전히 동일한 중복 행**(`exact_duplicate`): 첫 원본 행만 유지하고 이후 관련 행을 제외합니다.
- **값이 충돌하는 중복**(`conflict_duplicate`, 같은 키·재고/운영 정책 값이 다름): 임의로 합치거나 마지막 값을 쓰지 않고
  관련 원본 행 번호를 모두 남기고 어느 값도 임의 선택하지 않으며 관련 행을 모두 제외합니다.
- 날짜/시점 컬럼(예: `snapshot_date`)으로 구분되는 **정상 시계열 다중 행**은 중복으로 보지 않습니다.

## 기준정보와 부분 적용 참조 관계

- `stores.node_id`는 `inventory.store_id`, `routes.source_id/target_id`, 추천의 출발·도착·DC가 참조합니다.
- `products.product_id`는 재고와 추천이 참조합니다. 기준정보 행이 제외되면 해당 ID를 참조하는 행도 같은
  정제 단계에서 제외되며 고아 참조는 적용 데이터에 남지 않습니다.
- DIRECT 추천은 유효한 점포 간 경로가, VIA_DC 추천은 유효한 DC와 두 구간 경로 또는 동일 의미의 VIA_DC
  경로 행이 있어야 합니다. DC01 오류는 독립적인 DC02 경로를 제거하지 않습니다.
- 제외 후 전체/테이블별 50% 이상 손실, 필수 테이블 소실, STORE/DC/상품/재고/경로 최소 조건 미달이면
  파일 전체를 사용하지 않습니다. 추천 후보가 0건인 것과 분석 입력 자체가 무효인 것은 구분합니다.

## 허용값·결측값 처리

- `node_type` ∈ {DC, STORE}, `route_type` ∈ {DIRECT, VIA_DC} 외 값은 오류.
- 숫자 컬럼의 문자열/NaN/inf/음수는 오류 또는 정리 대상으로 처리하고, 화면은 `-`/`데이터 없음`으로 표시.
- 안전재고·최소재고·재주문점·목표재고는 0 이상의 숫자만 사용합니다. 빈 선택값은 미입력으로 처리하고,
  잘못된 값이 있는 재고 행은 기존 문제행 제외 정책으로 분리합니다.
- 선택 컬럼이 없으면 해당 VHS 구성요소에 **중립값 50**을 적용(좋은 값 위장 아님)하고, 분석이 제한될 수 있음을 안내.
- 컬럼명 앞뒤 공백·별칭은 자동 정규화(`column_aliases.py`).

## 샘플 파일

`samples/` 폴더(시뮬레이션 검토용):

- `Varo_V2_sample_small_4stores_1dc.xlsx` (4점포/1DC)
- `Varo_V2_sample_normal_6stores_1dc.xlsx` (6점포/1DC)
- `Varo_V2_sample_standard_8stores_1dc.xlsx` (8점포/1DC)
- `Varo_V2_sample_dual_dc_10stores_2dc.xlsx` (10점포/**2DC**, DC01·DC02 구분)
- `Varo_V2_sample_edge_3stores_1dc.xlsx` (3점포/1DC, 극단 케이스)

DQN 학습 샘플 팩(`Varo_DQN_training_samples_10pack`)은 이 저장소에 포함되지 않는 외부 데이터입니다.

`validation_data/` 폴더(운영 형식 검증용, 앱 기본 경로 아님):

- `varo_v2_anonymized_operational.xlsx` — 40점포/2DC/30상품, 분석 대상 2,742행.
  위 스키마만 사용하며 **의도적인 오류 행·유지 경고 행이 섞여 있습니다.**
  이름·좌표는 모두 가상 값이고 개인정보나 실제 업체 정보는 없습니다.
- `varo_v2_anonymized_operational_manifest.json` — 기대 결과(행 수·제외 행·DC 구성 등).

`python tools/generate_anonymized_operational_workbook.py`로 고정 seed에서 언제든 다시 만들 수
있으며, 앱의 기본 샘플을 대체하지 않습니다. 검증 내용은 `docs/OPERATIONAL_VALIDATION.md` 참고.

## 잘못된 데이터 예시

- `node_type`에 "물류센터" 같은 자유 텍스트 → 오류(DC/STORE만 허용).
- `recommended_qty`에 `0`, 음수, 문자열 → 오류/이동 불가.
- VIA_DC인데 `dc_id` 비어 있음 → 오류/이동 불가.
- `routes.source_id`가 `stores.node_id`에 없음 → 오류(존재하지 않는 점포/DC).
- 출발지=도착지 추천 → 실행 불가능(제외).
- 이동 수량 > 출발 점포 재고 → 이동 후 음수, 실행 불가능(제외).
- 이동 후 출발 재고 < 적용 재고 하한 → 실행 불가능(제외). 정확히 하한과 같으면 허용.

## 실행 이력 지속 저장 구조

실행 이력은 업로드 원본·추천 알고리즘과 분리된 공통 service/store 계약으로 저장합니다. 설정이 없으면
기존과 동일한 SQLite, `VARO_HISTORY_DATABASE_URL`이 있으면 PostgreSQL adapter를 선택합니다. 선택 우선순위는
테스트·이관에서 넘긴 명시적 SQLite 경로 → PostgreSQL URL → `VARO_HISTORY_DB_PATH` → 기본 SQLite 경로입니다.
PostgreSQL을 명시하고 연결에 실패한 경우 로컬 파일로 자동 전환하지 않습니다.

- SQLite 기본 위치: `runtime_data/varo_execution_history.sqlite3`
- SQLite schema version: `PRAGMA user_version=1`
- PostgreSQL schema version: `execution_history_schema_meta`의 단일 version 행
- 공통 저장 시각: timezone이 포함된 UTC ISO 8601 문자열
- 금액/점수: 현재 Python 계산 의미를 유지하는 SQLite `REAL` / PostgreSQL `DOUBLE PRECISION`
- 미입력 사후 결과: 두 backend 모두 `NULL`

현재 version 1의 업무 테이블은 다음과 같습니다.

- `execution_plans`: plan ID, 실행계획 알고리즘 버전, 후보 평가 알고리즘 버전, 데이터 signature,
  생성·기록·수정 시각, 상태, 계획 건수·수량, 예상 비용·절감·순효과.
- `execution_items`: plan/candidate 연결, 출발·도착·상품·경로·DC, 계획 수량과 예상값, VHS·안정성·신뢰도,
  실제 실행 상태·수량·사유·메모, 선택적 사후 재고·판매·폐기·품절·운송비·절감액.
- `execution_item_events`: 상태나 실제 수량을 수정할 때 이전/새 상태와 수량을 남기는 최소 변경 기록.

계획과 전체 항목은 하나의 transaction으로 저장됩니다. 같은 `plan_id`는 다시 생성하지 않지만 해당 계획의
실행 상태와 실제값은 수정할 수 있습니다. 삭제 API는 제공하지 않으며 취소 상태 또는 수정 기록을 사용합니다.
실제값이 없는 컬럼은 `NULL`이고 예상값을 복사하거나 0으로 대체하지 않습니다. 실제 순효과는 실제 운송비와
실제 절감액이 모두 입력된 경우에만 계산합니다.

두 backend 모두 `execution_plans.plan_id`와 `execution_items(plan_id, candidate_id)`를 primary key로 보호하고,
항목→계획 및 감사→항목 foreign key를 둡니다. 계획+항목 저장, 실행결과+감사 저장, SQLite→PostgreSQL 이관은
각각 같은 transaction에서 완료되거나 전부 rollback됩니다. PostgreSQL 항목 수정은 대상 행을 잠가 동시에
수정하는 요청을 순서대로 처리합니다. 연결은 작업마다 열고 예외 여부와 무관하게 닫으며, 자체 대형 pool은
구현하지 않습니다.

CSV export는 plan/candidate 연결에 필요한 ID와 알고리즘 lineage, 계획 당시의 제한된 VHS feature snapshot,
실제 실행·사후 결과만 포함합니다. 로컬 DB 경로와 자유 메모는 포함하지 않으며 UTF-8 BOM으로 생성합니다.
backend별 SQL 필드나 연결정보는 포함하지 않습니다.

### SQLite에서 PostgreSQL로 이관

`tools/migrate_execution_history.py`는 기존 SQLite를 읽기 전용으로 열고 plan/item/audit 관계와 집계 건수를
검증합니다. `--dry-run`은 대상 DB에 쓰지 않고 전체·중복·무효·신규 건수만 출력합니다. `--apply`는 이미 있는
plan을 건너뛰고 신규 plan과 연결 항목·감사를 하나의 transaction으로 삽입한 뒤 행 수를 다시 확인합니다.
원본 파일의 내용은 변경하지 않으며, 출력에는 DB URL·비밀번호·plan/candidate 상세를 표시하지 않습니다.

향후 조직/사용자 소유권이 필요할 때는 schema version을 올려 `organization_id`/`workspace_id` 같은 실제 소유
키를 추가합니다. 현재 알 수 없는 소유자를 가짜 ID로 생성하지 않습니다.

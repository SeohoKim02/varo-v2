# Varo V2 실데이터 검증 기반

Varo V2가 "실제 데이터 기반 결과"라고 주장하려면, 결합 방식 자체가 검증 가능해야 한다.
이 문서는 실데이터 원본의 위치·결합 규칙·한계를 기록한다. 업로드 워크북 스키마는
[`DATA_SCHEMA.md`](DATA_SCHEMA.md), 익명화 운영 형식 검증은
[`OPERATIONAL_VALIDATION.md`](OPERATIONAL_VALIDATION.md)에 있다.

## 원본 위치

실데이터는 **이 저장소 밖**에 있다. 원본은 공공데이터 파일 그대로이고, 저장소에 복사하지
않는다. 탐색 순서는 `VARO_REAL_DATA_DIR` 환경변수 → 프로젝트 인접 폴더 →
Desktop / OneDrive Desktop 이다(Windows 알려진 폴더 이동으로 Desktop이 OneDrive로 옮겨갈 수
있어 둘 다 본다).

```
VARO_V2_REAL_DATA/
  01_Retail_Stocks/            raw · processed   실제 판매 + On-hand 재고
  02_Korea_Suhyup_Logistics/   raw · processed   수협 물류센터·공판장 재고/입출고
  03_Korea_Suhyup_Warehouse/   raw · processed   수협 조합창고 재고/입출고  ← 이 문서
  ...
  91_Data_Reviews/                               진단 기록
```

원본(`raw/`)은 **읽기 전용**이다. 정규화·파생값·flag는 `processed/`에만 만든다. 결합 도구는
실행 전후로 raw 파일의 SHA-256을 비교해 변경이 없었음을 manifest에 남긴다.

## 03 수협 조합창고 — 재고 + 입출고 결합

### 원본

| 역할 | 파일 | 인코딩 | 행 |
|---|---|---|---|
| flow | `해양수산부_수협조합창고품목별창고입출고현황_20260731.CSV` | CP949 | 8,439 |
| stock | `해양수산부_수협조합창고품목별창고재고현황_20260731.CSV` | CP949 | 42,237 |

flow 컬럼: `표준코드 · 조합코드 · 창고코드 · 입출고구분 · 기준일자 · 조합명 · 창고명 ·
표준코드명 · 입출고구분명 · 수량`
stock 컬럼: `조합코드 · 창고코드 · 표준어종코드 · 기준일자 · 조합명 · 창고명 · 표준어종명 · 수량`

**이 원본에는 중량(kg) 컬럼이 없다.** 수량 단위만 있다. kg는 02 물류센터 데이터셋에만 있다.

### 상품코드를 직접 join하면 안 되는 이유

두 파일의 상품코드는 **같은 체계의 서로 다른 입도**다.

- flow `표준코드` — 8자리. 어종 6자리 + 상태 2자리. 예: `61010020`, `61010030`
- stock `표준어종코드` — 6자리. 어종만 있고 상태 구분이 없다. 예: `610100`

따라서 **전체 코드 일치는 0건**이고, 코드를 그대로 연결하면 결합률이 0%가 된다. 과거
`suhyup_warehouse_inventory_flow_actual.csv`가 이 방식으로 만들어졌다.

### 실제 사용하는 결합 키

```
조합코드 + 창고코드 + 기준일자 + 어종코드
```

어종코드는 stock 쪽은 `표준어종코드` 원본 값이고, flow 쪽은 `표준코드` 앞 6자리에서
유도한다. 이 유도는 **가정이 아니라 데이터로 검증한다**: 같은 어종코드에 대해 flow
`표준코드명`에서 상태 표기(`(냉동)` `(냉장/신선)` `(활)` `(건)` `(염장)`)를 뺀 부분이 stock
`표준어종명`과 글자 그대로 같아야 한다.

- 비교 가능한 (어종코드, 표준코드, 상품명) 조합 **472건 중 472건 일치, 불일치 0**
- 하나라도 어긋나면 결합을 실행하지 않고 상태를 `사용 불가`로 내린다.
- 상태 표기 목록에 없는 새 표기가 들어오면 stem이 그대로 남아 검증이 실패한다. 즉 모르는
  값을 조용히 통과시키지 않는다.

### 상품명을 결합 키로 쓰지 않는 이유

같은 창고·날짜에서 **서로 다른 어종코드가 같은 이름**을 갖는 경우가 실제로 있다
(`붕장어` = `619101`·`932106`, `장어류` = `619100`·`711800`). 이름 키를 쓰면 stock 쪽에
중복키 66건이 생기고 서로 다른 실제 어종이 한 키로 합쳐진다. 정규화된 상품명은 계보와
검증에만 쓴다.

### 상품명 정규화 규칙

`normalize_product_name()`은 표기 차이만 걷어내는 결정적 함수다.

1. 유니코드 NFKC 정규화(전각/반각 통일)
2. 제어문자·zero-width·BOM 계열 제거
3. 연속 공백을 하나로, 앞뒤 공백 제거

**하지 않는 것**: 숫자·용량·규격·등급·원산지·품종·상태 표기 제거, fuzzy 병합, AI 추측 병합.
서로 다른 실제 상품은 계속 다르게 남는다. 원본 상품명은 삭제하지 않고 함께 보존한다.

버전은 `NORMALIZATION_VERSION`으로 manifest에 기록한다. 규칙이 바뀌면 버전을 올린다.

### 키 유일성과 관계 유형

| 대상 | 행 | 유일키 | 중복키 | 최대 다중도 |
|---|---|---|---|---|
| flow `(조합·창고·일자·표준코드·입출고구분)` | 8,439 | 8,439 | 0 | 1 |
| flow `(조합·창고·일자·어종코드)` | 8,439 | 7,059 | 1,380 | 2 |
| stock `(조합·창고·일자·어종코드)` | 42,237 | 42,237 | 0 | 1 |

flow의 어종 단위 중복 1,380건은 **같은 날 같은 어종의 입고 1건 + 출고 1건**이다(최대 다중도
2, 방향까지 포함하면 중복 0). 데이터 중복이 아니므로 `drop_duplicates()`로 지우지 않고,
방향별로 각각 합산한다.

집계 후 결합 관계는 **1:1 7,057 / 1:N 0 / N:1 0 / N:M 0**. 통제되지 않은 N:M 확장은 없고
행 확장 계수는 1.00005다(미결합 flow 키 2건이 추가된 것이 전부).

### flow 집계

같은 조합·창고·일자·어종에 여러 event가 있을 수 있으므로 **방향별로 따로 합산**한다. 입고와
출고를 섞지 않는다. 기록이 없는 방향은 실측 컬럼(`inbound_qty_actual`)에서 **결측으로
유지**하고, 분석용 컬럼(`inbound_qty_for_analysis`)에서만 0으로 채운다. 채웠다는 사실은
`inbound_qty_zero_filled` / `flow_zero_fill_provenance=assumed`로 남는다.

### 결합 결과

| 지표 | 값 |
|---|---|
| flow 집계 키 | 7,059 |
| stock 키 | 42,237 |
| 공통 키 | 7,057 |
| matched | 7,057 |
| flow 미결합 | 2 |
| stock 단독(그날 입출고 없음) | 35,180 |
| ambiguous | 0 |
| 결합률 (flow 기준) | 99.9717 % |
| 결합률 (stock 기준) | 16.7081 % |
| 출력 행 | 42,239 |
| 범위 | 조합 51 · 창고 59 · 어종 266 · 31일 (2026-07-01 ~ 07-31) |

미결합 2건은 stock 어종 목록에 없는 코드다: `93000000`(상품명 결측), `79000020`(상품명이
`(냉장/신선)`뿐이라 어종명이 없음). 삭제하지 않고 `unmatched_product`로 보존한다.

### 과거 진단의 "공통 흐름키 7,053"과의 차이

```
어종코드 키 7,059 = 상품명 키 7,053 + 상품명 없는 키 2 + 이름 충돌로 합쳐진 키 4
```

- **+2**: 상품명이 없어 이름 키에서는 아예 빠졌던 행. 코드 키에서는 미결합으로 보존된다.
- **+4**: 서로 다른 어종코드(`붕장어`·`장어류`)가 같은 이름으로 한 키에 합쳐졌던 경우.

숫자를 억지로 맞추지 않았고, 차이는 전부 위 두 원인으로 설명된다.

### 재고 수지 참고 진단

`재고 변화 ≈ 입고 − 출고`를 **진단 지표로만** 쓴다(hard validation 아님). 전날 재고가 없는
계열 첫 행은 판정하지 않는다. 소수점 수량(예: 408.2)에서 생기는 1e-14 수준의 부동소수
잔차는 `BALANCE_EPSILON`으로 걸러낸다.

| 항목 | 값 |
|---|---|
| 비교 가능한 연속일 쌍 | 40,842 |
| 일치 | 40,838 (99.9902 %) |
| 입출고 기록 있는 쌍 | 6,717 → 일치 6,713 |
| 입출고 기록 없는 쌍 | 34,125 → 일치 **34,125 (100 %)** |

이 결과가 뜻하는 것:

1. 어종 단위 결합이 실제로 맞다. 잘못 연결했다면 수지가 이렇게 닫히지 않는다.
2. 재고 `기준일자`는 **그날 입출고가 반영된 일 마감 시점 snapshot**으로 관측과 일치한다.
   원본 설명에 시점이 명시되어 있지 않아 데이터 근거에 의한 해석이며, 계약된 정의가 아니다.
   시점 정의가 필요한 분석에서는 **확인 필요**로 다룬다.
3. 입출고 기록이 없는 날 재고가 100% 그대로다 → 기록 없음은 결측이 아니라 **이동 없음**이다.
   분석용 0 채움이 관측과 어긋나지 않는다.

불일치 4건은 `suhyup_warehouse_stock_flow_balance_gaps_review.csv`에 그대로 남긴다.

### 이상치

| flag | 건수 | 처리 |
|---|---|---|
| `stock_qty_zero` | 1,310 | 원본 값 그대로, flag만 |
| `stock_qty_missing` | 2 | 미결합 flow 키 |
| `product_name_missing` | 1 | 원본 결측, 보존 |
| `stock_qty_negative` · `inbound_qty_negative` · `outbound_qty_negative` | 0 | — |
| `date_invalid` · `location_missing` | 0 | — |

원본 이상값은 수정하지 않는다. flag만 남긴다.

### Provenance

| 값 | provenance |
|---|---|
| 재고 수량 | `actual` |
| 입고 수량 · 출고 수량 | `actual` |
| 입출고 0 채움(분석용 컬럼) | `assumed` |
| 중량(kg) | `not_available` |
| 창고 간 실제 이동 이력 | `not_available` |
| 거리 · 이동시간 · 운송비 · 차량용량 | `not_available` |

상품명 정규화는 **값의 provenance를 바꾸지 않는다.** 실측 수량은 정규화를 거쳐도 `actual`로
남는다. 매핑 방법은 `species_code_source` / `species_name_consistency` / manifest의
`product_code_relation`에 별도로 기록한다. 반대로 계산·채움 값을 `actual`이라고 부르지 않는다.

### 계보

출력 각 행은 `source_dataset` · `stock_source_file` · `stock_source_row` · `flow_source_file` ·
`flow_source_rows`(스프레드시트 1-based 행번호) · `flow_standard_codes`(원본 상품코드) ·
`flow_product_names`(원본 상품명) · `species_name`(원본 어종명) ·
`species_name_normalized` · `match_status`로 원본까지 되짚을 수 있다. 이 계보 컬럼은 내부
검증용이며 사용자 기본 UI에 노출하지 않는다.

### 산출물

`03_Korea_Suhyup_Warehouse/processed/` (모두 UTF-8 BOM CSV / JSON)

| 파일 | 내용 |
|---|---|
| `suhyup_warehouse_stock_flow_species_matched_actual.csv` | 결합본 42,239행 |
| `..._manifest.json` | 원본 해시·키·수치·provenance·판정 |
| `suhyup_warehouse_stock_flow_species_unmatched_review.csv` | 미결합·모호 행 |
| `suhyup_warehouse_stock_flow_balance_gaps_review.csv` | 수지 불일치 행 |
| `suhyup_warehouse_processed_status.json` | 과거 산출물의 유효/무효 표시 |

과거 산출물은 삭제하지 않되 **분석 입력으로 쓰지 않는다.**

| 파일 | 상태 |
|---|---|
| `suhyup_warehouse_inventory_flow_actual.csv` | `invalid_join_key` (상품코드 직접 연결, 결합률 0%) |
| `suhyup_warehouse_inventory_flow_actual_final.csv` | `superseded_ambiguous_name_key` (상품명 키, 모호 결합 잔존) |

### 재현

```
python tools/validate_suhyup_warehouse_join.py              # 검증만 (파일을 만들지 않음)
python tools/validate_suhyup_warehouse_join.py --write      # 산출물 생성
python tools/validate_suhyup_warehouse_join.py --write --overwrite
python tools/validate_suhyup_warehouse_join.py --data-root <VARO_V2_REAL_DATA 경로>
```

출력은 집계 수치만 보여주고 원본 전량이나 개인 식별 정보를 찍지 않는다. 종료코드는 결합이
안전할 때만 0이다. 결합이 안전하지 않으면 `--write`를 줘도 산출물을 만들지 않는다.

테스트: `python -m pytest tests/test_suhyup_warehouse_dataset.py -q`
실데이터 회귀 테스트는 원본이 있을 때만 실행되고, 없으면 skip된다.

## 이 데이터로 지금 검증할 수 있는 것

- 창고 단위 실제 재고 수준과 일별 변화
- 실제 입고·출고 수량과 재고의 수지 정합성
- 어종·창고·조합·날짜별 실제 재고 불균형
- 결측·이상치·모호성을 포함한 실데이터 처리 경로

## 아직 실제 데이터가 없는 것

이 데이터의 입고/출고는 실제 관측값이지만 **출발지-도착지가 명시되어 있지 않다.**
따라서 `창고 A → 창고 B` 형태의 거점간 transfer history로 해석하지 않는다.

```
actual transfer history = not_available
```

거리·이동시간·차량용량·운송비도 이 원본에 없다. 가짜 거리·가짜 운송비·가짜 차량용량을
만들어 넣지 않는다. 이 값들이 필요한 주장은 해당 실데이터를 확보한 뒤에 한다.

## Varo 본체 연결 상태

이 작업은 **실데이터 결합 검증까지**다. 결합본은 아직 Varo 추천 파이프라인
(VHS/Greedy/DQN/MILP/실행계획)에 연결되어 있지 않고, 기존 알고리즘·실행계획·실행 이력은
이 작업으로 바뀌지 않았다. 향후 actual validation dataset으로 쓰려면 adapter를 따로 두어야
하며, 그때도 위의 `not_available` 항목을 만들어내지 않는 것이 조건이다.

# Varo V2

## 제품 개요

Varo V2는 실제 재고 데이터를 분석해 폐기 위험과 재고 불균형을 찾고, 점포 간 이동 경로와 수량을 운영자가 검토할 수 있게 하는 Streamlit 기반 의사결정 지원 시스템이다. 데이터 적용, 추천 비교, 경로 확인, 재고 이동 시뮬레이션, 품질 검증과 운영용 내보내기를 하나의 흐름으로 제공한다.

## 해결하는 운영 문제

- 점포별 초과·부족 재고를 상품 단위로 함께 확인한다.
- 이동 전후 재고와 수요를 비교해 실행 가능한 수량을 검증한다.
- 직접 이동과 물류센터 경유 방식의 비용·거리·시간을 구분한다.
- 추천 순위가 데이터 편향, 조건 변화와 제약에 얼마나 민감한지 확인한다.
- 분석 결과를 CSV, Excel, JSON으로 내보내 후속 운영에 연결한다.

## 주요 기능

- 엑셀 업로드 및 기본·DQN 샘플 로드
- 악성재고 판단과 점포 간 이동 후보 생성
- Varo Hybrid Score(VHS) 기반 우선순위와 Greedy 비교
- DQN 원본·균형형 데이터 품질 및 명시적 학습 비교
- Pareto 보조 검증, 상세 민감도와 최적성 Gap 계산
- DIRECT/VIA_DC 경로 상세와 DC01/DC02 구분
- 실제 재고·수요 기반 이동 전후 상태 및 차량 시뮬레이션
- 전체 샘플 품질 검증과 CSV·Excel·JSON 내보내기

## 알고리즘 구조

각 알고리즘의 실제 역할은 다음과 같다.

- VHS: 정규화한 절감액·폐기 위험·수요·재고 균형·경로 비용·실행 가능성·프로모션 점수의 자동 가중합(0~100)으로 후보 우선순위(`vhs_rank`)를 만든다.
- Varo Final: 실행 수량이 큰 순, 운송비가 낮은 순, VHS 순위 순으로 경로 실행 순서(`varo_final_rank`)를 정한다. 1순위가 `최종 추천`이다.
- Greedy: 비용·수량·전략 키워드 휴리스틱 점수 순서로 빠른 기준 순위를 제공한다.
- Pareto: 서비스 수량·운송비·절감액 frontier에서 ideal-point 거리로 최대 5개 경로를 고른다.
- DQN: 사용자가 버튼으로 학습할 때만 8개 action label 위의 학습 기반 비교를 제공하며, VHS 점수 가중치는 0이다.
- MILP: 서비스 최대화 후 비용 최소화를 푸는 오프라인 benchmark다.
- 처리 action(`varo_action`)은 VHS 규칙(폐기·재배치·할인·보류)으로 정해지는 규칙 기반 label이다.

## 판매자 손실 비교 (Seller Loss Decision)

`services/seller_loss_engine.py`는 같은 결정 수량 Q에 대해 재고 이동(TRANSFER), 정상 판매 유지(NORMAL_SALE), 할인 판매(DISCOUNT_SALE) 세 가지만 금액으로 비교한다. 분석 파이프라인은 이를 `seller_loss_analysis`에 기존 결과와 나란히 기록하며, 기존 action·순위·KPI는 바꾸지 않는다.

- 기준은 예상 회피 가능 손실 최소화 하나다: `L(s) = Q × 정상가 − V(s)`, `V(s)`는 해당 전략으로 Q개에서 회수되는 현금(매출 − 운송·보관·폐기 처리비 + 잔존가치)이다.
- 이미 매입한 원가는 세 전략에서 같으므로 비교에 넣지 않는다. 미판매 재고의 가치는 정상가 기준으로 한 번만 세고, 폐기에는 처리비만 더한다.
- 이동의 출발 점포 판매기회 손실은 정상 판매 대비 차이(bridge)로 보고하며 손실 항목에 다시 더하지 않는다.
- 가격·수요·잔여 유통기한이 없거나, 대리지표(proxy)이거나, 단위·통화·데이터셋이 섞이면 금액 비교를 하지 않는다. 할인 판매는 관측값이나 판매자가 입력한 판매 증가율이 있을 때만 계산한다. 기존 프로모션 분석의 기본값(20%, 80%)과 config의 프로모션 설정은 실제 운영 판단에 쓰지 않는다.
- 보관비·폐기 처리비·잔존가치가 없으면 0으로 두지 않고 미상으로 둔다. 허용 범위의 어떤 값에서도 순위가 같을 때만 추천한다.
- 실제 데이터 검증 산출물은 `C:\VARO_V2_REAL_DATA\_SELLER_LOSS_VALIDATION`에만 저장한다(`python -m services.seller_decision_validation`, `python -m services.seller_loss_input_validation`).

### 실제 데이터 + 판매자 입력 (seller_loss_inputs)

Varo Seller Loss Engine은 실제 데이터에 있는 값은 그대로 쓴다. 데이터에 없는 사업장 값은 판매자가 직접 입력할 수 있다. 입력이 부족하면 Varo는 값을 추정해 추천하지 않고 `PARTIAL` 또는 `COMPARISON_UNAVAILABLE`을 반환한다.

- 판매자 입력은 `uploaded_data["seller_loss_inputs"]`나 업로드 workbook의 선택 시트 `seller_loss_inputs`로 받는다(`services/seller_loss_inputs.py`). 시트가 없으면 기존처럼 업로드 데이터만 쓴다. 시트가 잘못되어도 분석은 멈추지 않고, 오류는 `seller_loss_analysis.seller_loss_inputs`에 행·열 단위로 기록된다.
- 점포 값(`normal_price`, `holding_cost_per_unit_day`, `disposal_cost_per_unit`, `daily_demand`, `remaining_shelf_life_days`, `discount_rate`, `promotion_uplift`, `salvage_value_per_unit`, `unit_cost`)은 `store_id`로 입력한다. 같은 값이 그 점포가 출발지일 때와 도착지일 때 모두 쓰인다. 경로 값(`transfer_cost` 1회 운송비, `transfer_cost_per_unit`, `transit_time_days`)은 `source_store_id`·`target_store_id` 또는 `route_id`로 입력한다.
- `scope`(ROUTE > PRODUCT_STORE > PRODUCT > STORE > GLOBAL)는 반드시 명시한다. 가장 구체적인 입력이 쓰인다. 같은 구체성에서 값이 다르면 CONFLICT가 되어 그 값을 쓰지 않으며, 첫 행이나 마지막 행을 고르지 않는다.
- 우선순위: 실제값(DIRECT_REAL, DERIVED_REAL)과 업로드 데이터 행의 값이 먼저다. 판매자 입력은 비어 있는 값(MISSING)을 채우거나 대리지표(PROXY)를 대신할 때만 쓴다. 데이터 값과 다르면 데이터 값을 유지하고 충돌로 기록한다. 실제 운송비가 있으면 판매자 운송비로 덮어쓰지 않는다.
- 모드: `ACTUAL_OPERATION`(기본)은 실제 데이터와 판매자의 실제 사업장 값만 쓴다. `SCENARIO`(`seller_loss_decision_mode`)에서는 `input_type=SCENARIO` 행과 config 프로모션 설정을 가정값으로 쓴다. 이 결과는 항상 `SCENARIO_ONLY`이며 실제 추천과 섞이지 않는다.
- 통화는 환산하지 않는다. 통화가 다르면 해당 전략이나 비교 전체가 불가가 된다. `price_unit`(예: `KRW/EA`)을 쓰면 재고 수량 단위와 맞아야 하며, 재고 단위를 모르면 `quantity_unit`도 함께 명시해야 한다.
- 결과마다 다음을 기록한다.
  - `decision_mode`
  - `evidence_level`: REAL_ONLY / REAL_PLUS_USER_INPUT / USER_INPUT_ONLY / SCENARIO / INSUFFICIENT
  - `recommendation_readiness`: RECOMMENDABLE / NOT_ROBUST / SCENARIO_ONLY / UNAVAILABLE
  - 전략별 준비 상태
  - 실제·판매자·누락·충돌 필드
  - 판매자 입력 감사 기록: 행·scope·적용 결과, 그리고 그 입력을 빼면 추천이 바뀌는지
- 실제 action 승격 후보는 RECOMMENDABLE뿐이다. 이번 단계에서는 승격하지 않는다(legacy action 유지).
- 입력 예시는 `samples/seller_loss_inputs_TEMPLATE_SAMPLE.csv`에 있다. 모든 값은 형식을 보여 주는 SAMPLE이며 기본값이 아니다.

재고 시뮬레이션은 추천 알고리즘을 다시 실행하지 않는다. 기존 추천 순서를 그대로 받아 실제 재고·수요 한도 안에서 실행 가능한 이동량을 복사본에 순차 적용한다. 따라서 시뮬레이션 결과가 원본 추천이나 업로드 데이터를 덮어쓰지 않는다.

알고리즘 비교 어댑터는 VHS, Greedy, DQN, Pareto와 최적성 계산 결과를 다음 공통 필드로 투영할 수 있다.

`algorithm_name`, `recommendation_id`, `route_id`, `product`, `source`, `target`, `route_type`, `quantity`, `score`, `rank`, `expected_savings`, `feasibility`, `confidence`, `explanation`, `data_signature`

## 데이터 형식

필수 시트는 `stores`, `products`, `inventory`, `routes`다. `recommendations` 또는 `v2_recommendations`는 선택 시트이며, 없으면 현재 입력으로 후보를 생성한다.

- 별도 `dcs` 시트는 로더에서 `stores`에 `node_type=DC`로 병합한다.
- `from_store_id`, `to_store_id`, `source`, `target` 계열 별칭은 표준 컬럼으로 변환한다.
- 비어 있거나 중복된 경로 식별자는 메모리 안에서 고유하게 정리한다.
- 원본 엑셀은 읽기 전용이며 정규화 결과를 원본 파일에 쓰지 않는다.
- 재고 상태는 명시 수요를 우선한다. 없으면 일평균 판매량, 잔여 유통일, 실제 최소 진열재고를 사용하고, 가능한 경우 기존 7일 수요 규칙을 적용한다.
- 필수 수량 열이 없으면 값을 추정하지 않고 `데이터 부족`으로 표시한다.

## 샘플 위치와 보호 원칙

- 기본 샘플: `data/Varo_V2_네트워크_샘플.xlsx`
- 시뮬레이션 검수 샘플: `samples/Varo_V2_sample_*.xlsx`
- DQN 원본 탐색 순서: `../Varo_DQN_training_samples_10pack`, `dqn_samples`, `data/dqn_samples`, `Varo_DQN_training_samples_10pack`

DQN 원본 샘플 10개와 내부 검수 샘플은 읽기 전용으로 사용한다. 원본 폴더가 완전하지 않으면 내부 검수 샘플을 원본/균형형 흐름에 연결하되 원본 수치 필드는 변경하지 않는다.

## 재고 이동 시뮬레이션

홈은 1순위 단일 경로를 기본으로 표시한다. 운영자는 1~3순위, 단일 경로/상위 3개, 이동 전/이동 후/전후 비교, 속도와 전체 경로 표시를 바꿀 수 있다.

- DIRECT: 출발 점포에서 도착 점포까지 한 구간으로 이동한다.
- VIA_DC: 출발 점포에서 선택된 DC를 거쳐 도착 점포까지 두 구간으로 이동하며 DC 경유 구간을 표시한다.
- 점포 카드: 상품, 현재/이동 후 재고, 수요, 상태 변화와 재고 수준 막대를 표시한다.
- DC 카드: DC 이름·ID, 경유 여부, 처리 추천 수와 이동량을 표시한다. 실제 용량 열이 있을 때만 용량을 표시한다.
- 상위 3개: 현재 추천 순서대로 복사본에 적용하고, 다음 경로마다 남은 이동 가능 재고와 도착 한도를 다시 확인한다.

속도는 느림 24초, 보통 14초, 빠름 8초다. 애니메이션은 브라우저에서 실행되므로 Python 반복 루프나 연속 재실행을 만들지 않는다. 데이터 로더, 분석 파이프라인, VHS와 DQN은 시뮬레이션 조작으로 호출되지 않는다.

## 분석 및 검증 기능

분석 및 검증 화면은 VHS 분석, Greedy 비교, DQN 학습·비교, Pareto 검증, 최적성 Gap, 민감도/신뢰도와 전체 샘플 품질 검증을 제공한다.

전체 샘플 품질 검증은 등록된 샘플을 사용자가 선택한 범위에서 순차 분석한다. 빠른·표준·전체 검증을 구분하고 추천 생성, 데이터 편향, 민감도, 제약조건, 최적성 차이와 오류·제외 내역을 비교한다. 페이지 진입만으로 샘플 분석이나 DQN 학습을 시작하지 않는다.

## DQN 처리 방식

DQN은 사용자가 버튼을 눌렀을 때만 실행한다.

- `DQN 학습 실행`: 현재 데이터 단건 학습
- `DQN 샘플 10개 일괄 검증`: 샘플 01~10 진단·생성·순차 학습
- `DQN 원본 vs 균형형 비교`: 같은 수치 특성의 라벨 분포와 학습 안정성 비교

원본 수량·비용·거리·절감액은 유지하고, 균형형 파생본은 학습용 action label만 재분배한다. 모델과 파생 결과는 각각 `outputs/dqn/`, `outputs/dqn_balanced_samples/` 아래에 생성되며 Git에서 제외한다.

Streamlit Cloud는 루트 `requirements.txt`를 설치한다. `requirements-dqn.txt`는 기존 설치 흐름과 호환되도록 루트 파일을 참조한다. PyTorch import가 실패해도 VHS·Greedy·Pareto·업로드·시뮬레이션은 계속 동작하고 DQN 학습만 제한된다.

## 실행 방법

```powershell
cd C:\Users\user\OneDrive\Desktop\Projects\Varo\varo_v2
py -m compileall .
py -W default -m unittest discover -s tests
py -m streamlit run app_v2.py --server.headless true --server.port 8539
```

검증 주소는 `http://localhost:8539`이며 상태 주소는 `http://localhost:8539/_stcore/health`다. 일반 실행은 `py -m streamlit run app_v2.py`다.

## 배포 방법

Streamlit Community Cloud에서 `app_v2.py`를 진입점으로 지정하고 루트 `requirements.txt`를 사용한다. 로컬 검증 환경과 동일한 Python 3.11을 권장한다. 로컬 학습 산출물, 캐시, 로그와 secret 파일은 배포 저장소에 포함하지 않는다. 자세한 확인 항목은 [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md)를 따른다.

## 제한사항

- 결과는 입력 데이터 품질과 제공된 제약 열의 범위에 의존한다.
- 이동 후 폐기 위험 점수의 공식이 입력에 없으면 위험 수치를 새로 만들지 않는다.
- 부분 이동 시 기존 예상 절감액을 비례 추정하지 않는다.
- 제한 탐색 기반 Gap은 동일 후보와 적용 가능한 제약 안의 비교값이다.
- Cloud 로컬 파일은 영구 저장소로 간주하지 않는다.

## 최종 의사결정 병렬 계약 v0.1

`PipelineResult.final_decision_integration`은 기존 추천/순위/Top-5/KPI와 독립적인 결과다.
UI/production action을 교체하지 않으며, DB 기록이나 outcome를 자동 생성하지 않는다.
원본 inventory의 dataset/business/점포/상품 키를 모두 포함하고, inventory에 없는
상품 master도 점포 NULL 행으로 남긴다. 같은 상품의 다른 점포는 별도 결과다.
중복 재고 입력은 합산하지 않고 충돌로, 중복 route는 수량을 중복 합산하지 않고 검토 대상으로 표시한다.

`products`는 상품-점포 요약, `routes`는 경로별 상세, `coverage`는 누락/중복 검사다.
각 결과에 mode/status, legacy/canonical/operator 행동, 선택 경로, 계획량/계산 운송비,
Seller Loss 비교 범위/예상 손실/근거/승격 상태, 부족 입력, lineage와 한국어 설명이 있다.
`display`는 향후 UI용 상품·상황·추천/계획·예상 효과·확인사항·근거 상태를 제공한다.
정렬 키는 상태와 식별자의 사전식 순서이며 새로운 가중치나 Varo Final 점수가 아니다.

최종 추천은 기존 Seller Loss Engine과 Promotion Gate가 허용한 요청 범위 안에서만 가능하다.
이동 추천에는 실제 제약 계획과 운영 경로 근거가 추가로 필요하다. PROXY 계획, legacy 할인,
금액 기반 최소손실 전략은 서로 다르다. 보류(HOLD)를 정상판매(NORMAL_SALE)로 바꾸지 않는다.
SCENARIO/TEST 비교 결과는 실제 운영 추천과 분리되며 production_action_applied는 항상 False다.
모르는 손실/통화/단위/실제 성과는 NULL이다. 여러 경로의 손실을 상품 손실로 합산하지 않는다.

후보 없는 상품도 기존 엔진/입력 Planner/Profile로 비교 문맥을 확인한다.
결정 수량이나 도착점이 없으면 재고 전체를 임의로 결정 수량으로 삼지 않는다.
필요한 업로드 문맥과 판매자 입력을 구분해 안내하고, 이미 제공된 유효 입력은 다시 묻지 않는다.
Validated sales를 우선 확인하며 출고량을 실제 소매 판매로 간주하지 않는다.
예측/ABC/회전율/폐기위험은 근거와 제한을 함께 표시하는 참고 정보다. 폐기위험은 확률이 아니다.

```powershell
python -m services.final_decision_integration_validation
python -m services.release_validation --mode quick
python -m services.release_validation --mode full
```

통합 검증은 수협 31일에서 새 필드만 비활성화한 기존 결과와 모든 이전 필드를 비교한다.
새 `_FINAL_DECISION_INTEGRATION_<시각>_<고유값>` 폴더에 상품/경로/부족입력/비교상태/설명/
summary/validation 보고서를 저장하며, 기존 결과나 Golden을 덮어쓰지 않는다.
현재 pipeline은 재고와 7일 수요 부족량을, 별도 offline benchmark는 중앙 재고와 출고 proxy를
포함한 source surplus/target need를 사용한다. 서로 다른 cap 계약이므로 계획량을 동일하다고 가정하지 않는다.
T5 QUICK에도 통합 테스트를 추가했으며, 기존 G1 보호 검사는 유지한다.
G1은 기존 `PROTECTED_PATHS`에 등록된 파일의 변경만 검사한다. `analysis_pipeline.py`는
현재 그 목록에 없으므로 병렬 연결 변경만으로 G1이 BLOCKED가 되지는 않는다.
Git dirty 상태는 별도로 기록하며, G1 통과도 변경 승인이나 운영 승인을 의미하지 않는다.
검증 통과와 production 교체 승인은 별개다. 실제 판매·수요·가격·정책·운영 근거·outcome 확보와
사용자 승인 후에만 별도 UI/production 연결 작업을 진행한다.

## T5 통합 회귀 검증 (로컬 전용)

```powershell
python -m services.release_validation --mode quick
python -m services.release_validation --mode full
# 선택 사항: 저장된 M5 v2 개발 선택을 그대로 사용한 final 재실행
python -m services.release_validation --mode full --rerun-m5
```

기본 데이터 경로는 `C:\VARO_V2_REAL_DATA`이며 `--data-root`로 변경할 수 있다. 원본,
canonical, 이전 검증 보고서와 수협 seed 17 / 300 episodes 저장 모델이 필요하다.
모델은 `outputs/dqn`의 기존 결과 JSON이 가리키는 파일을 읽기만 한다. 재학습하지 않는다.
일부 기존 단위 테스트의 작은 학습 fixture는 별도의 임시 출력 폴더에서 실행된다.

QUICK은 핵심 계약·입력·행동·판매자 검증 테스트, 필수 reference/model 존재 여부,
고정 지문, Git 보호/공백 검사다. 수협 실제 benchmark는 **NOT_RUN**이며 FULL의 대체가 아니다.
FULL은 전체 pytest, canonical 수협 bridge, 31일 × 6전략 Golden, T1/T2/T3,
T4-B 실제 일별 재생, 판매자 입력·planner·profile·promotion 및 격리 SQLite 검증을 실행한다.
P1/P1-B/P1-C와 M5/외부 예측은 고정된 기존 보고서·CSV 지문 및 기존 unit test로 검사한다.
이 항목은 **REUSED_FROZEN_RESULT**이며 실제 benchmark 재실행으로 표시하지 않는다.
M5 추가 실행도 새 보고서 폴더의 frozen selection 복사본을 사용하며 기존 선택을 재학습/재선정하지 않는다.

실행 순서는 환경/보호 지문 → 테스트 → canonical → Golden → T1/T2/T3 →
P1 고정 결과 → T4-B → 판매자 → 예측 → 실행 후 보호 지문이다.
QUICK의 보호 해시 검사도 OneDrive 파일 내려받기 상태에 따라 시간이 걸린다.
FULL은 전체 테스트만 기존 환경에서 약 6분이며 데이터 재생·파일 검사 시간이 추가된다.
각 실행의 실제 시간은 `validation_runtime.csv`에 기록한다. 측정하지 않은 메모리는 NULL이다.

결과는 새 `_RELEASE_VALIDATION_<시각>_<고유값>` 폴더에만 저장된다.
summary/matrix/failures, 날짜·전략별 Golden 비교, runtime, reference_integrity,
release_readiness 및 검증기 목록/상세 로그를 생성한다. 기존 폴더 덮어쓰기는 거부한다.
실제 데이터와 보고서를 Git/CI에 업로드하지 않는다.

`services/release_reference_manifest.json`은 T5 도입 당시 기존 기준의 SHA-256,
Golden schema/날짜/전략, 저장 모델 지문을 등록한 읽기 전용 목록이다. 실행 중 갱신 기능은 없다.
기준 변경은 기존 보존 → 신규 결과 별도 생성 → 차이 전수 검토 → 근거 검증 →
사용자 승인 → 별도 버전 등록 순서로만 한다. hash만 등록하며 원시 데이터는 포함하지 않는다.
숫자는 계획 이동량/추정 운송비/예상 절감이며 실제 판매·지출·확정 절감이 아니다.
규칙 기반 폐기 위험 점수는 폐기 확률이 아니다.

FAIL은 계산/기준 변경, BLOCKED는 필수 입력·모델·보호 코드 승인 부족,
ENVIRONMENT_ERROR는 파일 잠금/읽기 실패로 구분한다. NOT_RUN/INSUFFICIENT_DATA는 PASS로 세지 않는다.
실데이터가 없으면 전체 회귀 성공을 주장하지 않는다. 실패해도 독립 검증은 계속하고,
Golden 실패는 최종 회귀 FAIL/출시 BLOCKED로 남긴다. 대표 실패 30건과 전체 실패 수를 기록한다.
T2 G1 Git 보호 검사는 계산 회귀와 별도 항목이며 기존 검증기의 기준은 완화하지 않는다.

OneDrive cloud-only/잠금 파일은 사용자가 동기화 상태를 확인해야 한다. 검증기가 캐시 폴더나
`.git`를 삭제/수정하지 않는다. 기존 모델·DB·canonical·archive는 전후 SHA-256으로 확인하고,
읽지 않는 raw 파일은 크기/수정 시각으로 보호한다(전체 raw 내용 재검증은 아니다).
운영 SQLite 대신 관리된 임시 SQLite를 쓴다.

코드 회귀 안전성, 실제 운영 근거, Seller 실제 적용, 시연 준비도를 별도로 판정한다.
STRICT_ACTUAL cap/소매판매 근거/실제 outcome가 부족하므로 테스트 통과는 운영 승인이 아니다.
production action은 이번 검증으로 교체되지 않는다. 최종 통합·추천 교체 승인·UI 최종 수정·
제출 전에 FULL을 실행하고, 변경 근거와 Golden 영향 검토 후 사용자 승인을 받아야 한다.

### 판매자 중심 결과 화면 (읽기 전용 v0.1)

상단 VARO 로고와 작은 `전체 기능` 팝업으로 시뮬레이션 / 전략 비교 / 학습 관리 / 결과 이력 / 설정에 접근한다.
홈에는 업로드·분석 실행·재고 흐름 시뮬레이션·핵심 결과만 표시한다.
전체 상품 검색·필터·정렬·상세·다운로드는 결과 이력의 `현재 상품 결과`에 모으고,
예시 A/B/C 비교는 전략 비교에 배치한다. 다른 화면의 `홈으로`와 `뒤로 가기`는 분석 결과와 화면 입력을 보존한다.
데이터를 불러오면 기존 파이프라인을 실행하고, `final_decision_integration`을 병렬로 표시한다.
화면 필터·상품 선택·다운로드는 분석 재실행이나 생산용 action 교체를 하지 않는다.
이동 후보가 없는 상품도 전체 상품·점포 결과에 포함한다. Top-5는 경로 선택 제한이지 상품 표시 제한이 아니다.

| 화면 정보 | 기존 결과 계약 |
| --- | --- |
| 상단 상품/계획/추천/추가 입력 수 | `summary.unique_products/selected_transfer_routes/minimum_loss_recommendations/additional_input_products` |
| 전체 상품·점포 행 및 상세 선택 | `products[]`, `decision_id`, 원자료 `source_lineage`의 기준일 |
| 쉬운 상태와 설명 | `decision_status`, `display.recommendation_or_plan`, `explanation_ko` |
| 계획 수량·계산 운송비 | `routes[].allocated_qty/calculated_transfer_cost/transfer_cost/plan_mode/plan_validation` |
| 정상판매/할인판매/이동 예상 손실 | `expected_loss_by_strategy`, `expected_loss_values`, `available_strategies`, `comparison_scope` |
| 추가 입력 안내 | Requirement Planner의 `missing_required_inputs` |
| 프로필과 근거 | 기존 `seller_business_profile`, `input_quality`, `seller_comparisons`, `promotion_status` |
| 참고 지표와 예측 | `reference_metrics`, `forecast` (점수는 확률이 아님) |
| 실제 행동·결과 | `operator_action`, `execution_status`, `outcome`, `actual_*` (기록이 없으면 미확인) |

상품명·코드·점포 검색, 상태/근거 필터, 입력 부족/이동 계획 필터, 정렬을 제공한다.
모든 CSV/JSON은 전체 결과의 원래 자료 모드와 근거를 보존하며, 모르는 금액은 0으로 만들지 않는다.
입력 양식의 SAMPLE 숫자는 예시이므로 실제 값으로 교체하거나 해당 행을 지워야 한다.
공통 프로필은 범위·유효기간·출처를 유지한다. 잔여 유통기한/현재 재고/결정 수량은 공통 프로필 값이 아니다.

시연 순서: 기본 샘플 → 전체 상품 검색/상세 → 입력 부족 확인 → 예시 A/B/C 계산 → 별도 실제 자료 업로드.
예시 계산은 기존 controlled scenario와 Seller Loss 엔진만 사용하며 SCENARIO로 격리한다.
실제 자료 기반의 추정 이동 계획과 예시 최소손실 전략은 같은 실증 결과가 아니다.
기존 재고 흐름 시뮬레이션은 복사본 전이이며, 상품 상세의 공유 제약 계획과 실행 기록을 대신하지 않는다.
DC 없는 시각화는 명시된 DIRECT_NETWORK에서 DIRECT 경로만 지원하며 가짜 DC를 만들지 않는다.
이력은 기존 SQLite/Shadow Ledger를 재사용한다. Shadow 조회는 읽기 전용이며 새 테이블을 만들지 않는다.
일반 상품별 통합 결과의 영구 이력 저장은 추가하지 않고 현재 결과 다운로드를 제공한다.

검증용 모델·이력이 사용자 로컬 `outputs`를 덮어쓰지 않도록 pytest 시작 전에 임시 저장소를 연결한다.
T5가 명시한 격리 저장소는 그대로 재사용한다. 작은 학습 fixture는 기존 테스트의 일부이며 운영 재학습이 아니다.
UI 수정 중 Git 보호 항목의 BLOCKED는 유지하고, 계산 회귀 통과와 별도로 보고한다.

### 독립 후보 기능 검토

기본 실행은 계속 `streamlit run app_v2.py`다. 기존 Varo Final, Seller Loss,
예측 경로와 저장 DQN 모델은 후보 구현으로 자동 교체하지 않는다.
후보 기능은 `streamlit run app_candidate.py`로 실행하고 명시적으로 검토를 시작한 뒤
설정의 `새 의사결정 결과 검토`에서 접근한다. 기본 앱의 다섯 메뉴는 유지한다.

후보 계산은 공통 제약 아래 부분 수량 배분, 기준 계획 보존, 경제성 비교,
알려진 관측만 사용하는 재계획을 별도로 검증한다. 예상 순현금가치와 계획 이동량은
실제 매출·지출·절감 실적이 아니다. 부족한 필수 입력은 임의 숫자로 대체하지 않는다.
후보 저장은 출력 루트의 `third_upgrade_20261017/candidate_history`에 격리하며
`VARO_OUTPUT_ROOT`를 존중한다. 기존 SQLite와 운영 모델을 덮어쓰지 않는다.

연구 검증용 코드와 테스트는 포함하되, 실험 산출물·모델·실제 데이터는 Git에 추가하지 않는다.
연구 복사본 전용 커밋을 요구하는 `selection_third_audit.py`는 통합하지 않았다.
대규모 재학습이나 운영 승격은 수행하지 않는다. latest 원본의 복원 근거가 미확인인
상태는 그대로 유지하며 테스트 통과를 운영 모델 승인으로 해석하지 않는다.

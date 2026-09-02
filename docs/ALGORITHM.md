# Varo V2 알고리즘 문서

이 문서는 화면에서 제거한 상세 알고리즘 설명을 코드 기준으로 정리한 것입니다. 서술은 실제 구현
(`services/`)과 일치해야 하며, 코드에서 확인되지 않는 내용은 적지 않습니다.

## 1. 전체 의사결정 흐름

`services/analysis_pipeline.py:run_analysis_pipeline` 기준.

1. **데이터 검증** — `data_validator.validate_workbook_data`. 오류가 있으면 상태 `validation_error`로
   즉시 중단(추천 계산 안 함).
2. **후보 확보** — 추천 시트가 있으면 사용, 없으면 `candidate_generator.generate_candidates`로
   재고·경로 기반 rule-based 후보 생성(`ensure_recommendations`).
3. **재고 분석 연결** — ABC/회전율/폐기위험/수요예측/안전재고/EOQ 등 승인된 레거시 분석을 결측 안전하게 연결.
4. **후보 프레임 구성 + Greedy 기준선** — `heuristic_optimizer.add_heuristic_scores`.
5. **VHS 재계산** — `vhs_score_engine.apply_auto_vhs` (2절).
6. **경로·컷라인·시간창·프로모션 보강** — DIRECT/VIA_DC 비용·시간, 컷라인 통과, 거래 가능 시간 등.
7. **Pareto 검증** — `pareto_analysis.compute_pareto` (5절).
8. **실행 가능성 게이트** — `feasibility.annotate_feasibility` (3절). **이동 불가 후보는 최종 추천에서 제외.**
9. **민감도·안정성·신뢰도 상태 요약** — `vhs_score_engine.weight_sensitivity`,
   `decision_support.recommendation_stability`, `decision_support.recommendation_confidence`.
10. **전체 실행계획 구성** — `execution_plan.build_execution_plan`. 후보 간 공유 출발 재고와 공유 도착
    부족량을 한 번에 배분하고, 경로 대안 중 하나와 정수 `planned_qty`를 결정.
11. **독립 계획 검증 + 사용자 행동 목록** — 재고 하한·목표량·경로/DC·상품/점포·중복·금액을 다시 검증한
    계획만 홈·추천 실행·경로 상세의 공통 행동 목록으로 사용.

DQN은 이 흐름에 자동으로 들어오지 않습니다(6절).

## 2. VHS (Varo Hybrid Score)

구현: `services/vhs_score_engine.py`. 결정적(deterministic): 같은 입력 → 같은 결과.

### 구성요소(9개)와 방향

각 구성요소는 서로 다른 의사결정 질문 하나에 대응합니다.

| 구성요소 | 라벨 | 무엇을 보는가 |
|---|---|---|
| `net_benefit_score` | 예상 순효과 | 절감액 − 이동 비용이 큰가 |
| `inventory_balance_score` | 재고 균형 | 출발지의 이동 가능 재고를 얼마나 걷어내는가 (상한 100%) |
| `disposal_risk_score` | 폐기 위험 | 유통기한 임박·악성재고를 얼마나 줄이는가 |
| `demand_fit_score` | 수요 적합도 | 도착지 부족량을 얼마나 채우는가 (상한 100%) |
| `route_cost_score` | 이동 부담 | 거리·시간·비용 부담이 작은가 |
| `feasibility_score` | 실행 여건 | 거리 기준·시간창을 통과하는가 |
| `demand_risk_score` | 수요 안정성 | 수요가 흔들려도 유효한가 |
| `post_move_risk_score` | 이동 후 위험 | 출발지 결품·도착지 과잉을 만들지 않는가 |
| `dqn_reference_score` | DQN 참고 | 선택형, 학습 결과가 정상일 때만 (기본 0) |

Greedy 기준선과 신뢰도는 **구성요소가 아닙니다.** 기준선이 점수 안에 있으면 비교가 순환하고,
신뢰도는 VHS 결과에서 파생되므로 다시 입력이 되면 순환합니다.

### 정규화

- 모든 구성요소는 **0–100** 범위(`_normalize_high` / `_normalize_low` / `_coverage_score` / `_bounded`).
- **Winsorized min–max**: 5~95 백분위로 clip한 뒤 스케일링합니다. 극단값 하나가 나머지 후보를
  좁은 구간에 몰아넣는 문제를 막으면서 순서는 그대로 보존합니다.
- **비율형 구성요소**(재고 균형·수요 적합도)는 후보 간 정규화를 쓰지 않고 `min(a/b, 1)`을 씁니다.
  이미 절대 0~1이라 이상치에 구조적으로 안전하고, **필요량보다 많이 보내도 점수가 오르지 않습니다**.
- `NaN`/`±inf`는 스케일 경계가 되기 전에 제거합니다.
- 결측/전부 동일/비수치 → **중립값 50** (좋은 점수로 위장하지 않음). `dqn_reference_score`는 DQN 미사용 시 0.

### 가중치

- 기본 가중치 `BASE_WEIGHTS`, 허용 범위 `WEIGHT_BOUNDS` — **단일 위치**(`vhs_score_engine.py`)에 정의.
- `optimize_weights`가 커버리지·분산 신호로 조정하고 `_project_to_bounds`로 **범위 내 + 합계 1.0**을 보장.
- DQN 미사용 시 `dqn_reference_score` 상한을 0으로 고정.

### 최종 점수·등급·동점 처리

- `auto_vhs_score = Σ(구성요소 × 가중치)`를 0–100으로 clip.
- 순위는 아래 tie-break 사슬로 정하며 **행 순서에 의존하지 않습니다**(`TIE_BREAK_KEYS`).

  1. VHS 점수 (높은 쪽)
  2. 예상 순효과 (큰 쪽)
  3. 이동 비용 (낮은 쪽)
  4. 이동 시간 (짧은 쪽)
  5. 경로 단순성 (DIRECT < VIA_DC)
  6. `route_id` (마지막 안정 키)

  모두 실제 계산값이므로 같은 입력은 항상 같은 순위를 만듭니다. 입력 행 순서를 뒤집어도
  순위가 같다는 것을 테스트로 고정했습니다.
- 등급: 80↑ 최적 · 65↑ 권장 · 50↑ 검토 · 그 외 보류.
- 구성요소별 점수는 각 후보 행에 보존되어 왜 높은지 추적 가능.

## 3. 실행 가능성 게이트 (feasibility)

구현: `services/feasibility.py`. 운영 재고 하한 위반은 VHS 정규화 전에 먼저 제거되어 다른 후보의 점수에도
영향을 주지 않습니다. 나머지 실행 조건은 최종 후보 확정 시 다시 확인하며, 실행 불가능한 이동은 낮은
점수로 남기지 않고 **제외**합니다. 3-상태:

- `추천 가능` — 모든 하드 조건 통과
- `데이터 확인 필요` — 유지하되 입력 부족/불확실 표시
- `이동 불가` — 하드 위반, 최종 추천에서 제외(사유 기록)

**하드 블록(이동 불가):** 출발지=도착지, 이동 수량 ≤0/NaN/inf, 경로 유형이 DIRECT/VIA_DC가 아님,
VIA_DC인데 DC 없음, 이동 수량 > 출발 점포 재고(이동 후 음수), 이동 후 출발 재고 < 적용 재고 하한,
음수 이동 비용, 순효과 ≤0, 동일 (상품·출발·도착·경로유형·DC) 경로의 정확한 중복. DIRECT/VIA_DC 또는
서로 다른 DC 대안은 실행계획 단계가 실제 값으로 하나를 고를 수 있도록 별도 후보로 유지합니다.

**소프트 플래그(데이터 확인 필요):** 이동 비용/절감액 계산 불가, 출발 점포 재고 데이터 없음,
재고 하한 계산 불가, 도착 점포 필요량 대비 과잉 공급(3배 초과).

화면에는 3-상태만 노출하고, 세부 `reason_code`는 진단/로그·이 문서에 둡니다.

### 3-1. 운영 재고 하한 결정

점포×상품 행마다 다음 우선순위를 한 번만 적용합니다.

1. 등록된 `min_stock` 또는 `safety_stock` (둘 다 있으면 두 하한을 모두 지키는 큰 값)
2. 등록 하한이 없고 실제 `demand_std`가 있으면 기존 `demand_std × 2` 추정
3. 둘 다 없으면 `unavailable`로 남기고 0으로 위장하지 않음

내부에는 `inventory_floor_value`, `inventory_floor_source`, `available_to_move`,
`quantity_limit_reason`을 남깁니다. 출처는 등록 최소재고/등록 안전재고/결합 등록값/추정/계산 불가로
구분합니다. `reorder_point`는 발주 트리거 참고값으로만 보존하고 하한으로 쓰지 않습니다.
`target_stock`은 도착점의 명시 목표로만 사용하며, 없으면 기존 수요 부족량 계산을 유지합니다.

## 3-2. 전체 재배치 실행계획

구현: `services/execution_plan.py`. VHS를 대체하지 않습니다. VHS와 강건성·신뢰도·Pareto가 후보의 가치를
평가한 뒤, 이 모듈은 여러 후보를 **동시에 실행할 수 있는 조합과 수량**으로 바꿉니다.

### 실제 제약

- 출발 점포×상품별 `Σ planned_qty ≤ current_stock − inventory_floor`.
- 도착 점포×상품별 `Σ planned_qty ≤ target_stock − current_stock`. 명시 목표가 없으면 후보 생성기가
  사용해 온 `상품 중앙재고와의 격차 + 7일 관측 수요`를 그대로 사용합니다.
- 같은 출발·도착·상품의 DIRECT/VIA_DC/DC01/DC02 대안은 최대 한 경로만 선택. 현재 데이터에는 경로별
  용량·구간 요금이 없어 동일 물량을 여러 경로로 나눌 근거가 없기 때문입니다.
- `0 < planned_qty ≤ recommended_qty`, 정수 1개 단위. pack size는 만들지 않습니다.
- 양의 순효과, 실제 존재하는 점포·상품·경로·DC만 선택. DC capacity가 없으면 가상 capacity를 두지 않습니다.
- 상품 키는 모든 공급·수요 제약에 포함되므로 다른 상품의 재고가 섞이지 않습니다.

### 목적과 수량별 금액

새로운 화폐 단위 가중치나 가상 패널티를 만들지 않고 단계별 우선순위를 사용합니다.

1. 실제 예상 순효과 합계 최대화
2. 같은 순효과 범위에서 부족·과잉 완화량(`planned_qty`) 최대화
3. 같은 결과에서 안정 후보 수량 우선
4. 마지막으로 VHS 점수가 높은 후보 우선

이동 비용은 1번 순효과에 이미 포함됩니다. 후보 데이터는 권장 수량 전체에 대한 비용·절감액 한 쌍만
제공하므로 수량을 줄일 때 두 금액을 같은 실제 후보의 단위당 비율로 함께 축소합니다. 차량 용량·고정비/
변동비 분리가 입력되기 전까지 별도 고정비나 pack size를 추정하지 않습니다.

### 최적화, fallback, 재현성

- 현재 환경에 이미 있는 `scipy.optimize.milp`를 선택형으로 사용합니다. 정수 수량과 경로 선택을 함께 풀며
  후보 조합 완전탐색은 하지 않습니다.
- SciPy가 없거나 실행 실패·시간 초과·수치 실패·검증 실패이면 VHS/Greedy 순서의 **결정론적 제약
  Greedy**로 다시 배분합니다. 이 경로도 같은 출발·도착·경로 제약을 사용합니다.
- 입력 후보를 점포·상품·도착·경로·DC·안정 후보 ID로 정렬하고 마지막 동률 키에 `candidate_id`를 사용해
  입력 행 순서를 뒤집어도 `plan_id`와 선택 수량이 같습니다.
- 선택되지 않은 후보는 나쁜 후보로 바꾸지 않고 `shared_source_inventory`, `destination_fulfilled`,
  `better_route_selected`, `lower_plan_value`, `infeasible`, `negative_net_benefit`로 이유를 보존합니다.

### 계획 검증

계획 계산 뒤 `validate_execution_plan`이 solver와 독립적으로 다음을 다시 계산합니다: 양의 정수 수량,
출발 재고 사용 합계와 이동 후 재고 하한, 도착 유입 합계와 목표/부족량, 양의 순효과, 실제 경로·DC,
점포·상품 ID, 경로 대안 중복, 후보 권장량 상한, 금액 비례 합계, NaN/무한대. 하나라도 실패하면 계산
결과는 행동 목록으로 노출하지 않고 안전한 fallback을 거친 뒤에도 실패하면 `계산 불가`로 반환합니다.

계획 결과는 `plan_id`, `algorithm_version`, `data_signature`, 후보/선택 수, 총 수량·비용·절감·순효과,
부족/과잉 완화량, `plan_status`, 선택 항목의 `recommended_qty`와 `planned_qty`, 제외 이유를 보존합니다.

## 4. Greedy의 역할

- 화면의 `greedy_rank`는 `heuristic_optimizer.add_heuristic_scores`(복합 점수)에서 옵니다.
- 검증용 기준선은 목적이 하나뿐인 **절감액 우선 Greedy**(`services/greedy_baseline.py`)입니다:
  예상 절감액이 큰 순서, 동점이면 이동 비용이 낮은 쪽, 그다음 `route_id`. 이동 비용 대비 효과,
  도착지 필요량, 이동 후 상태, 수요 불확실성을 보지 않는 **의도적으로 단순한 규칙**이라
  VHS와의 차이가 해석 가능합니다.
- 비교 결과(`validation_report.greedy_baseline`)는 기술 검증용이며 기본 화면에 넣지 않습니다.
  측정값은 [`docs/ALGORITHM_BENCHMARK.md`](ALGORITHM_BENCHMARK.md) 참고.
- 전체계획 비교에서는 기존 `greedy_rank` 순서로 같은 공유 제약을 적용한 `constrained Greedy`와 VHS 기반
  실행계획을 비교합니다. 따라서 차이는 제약 준수 여부가 아니라 제한된 재고를 어떤 후보에 먼저 배정했는지입니다.

## 5. Pareto의 역할

- 구현: `services/pareto_analysis.py`. **4개 축**(예상 순효과·폐기 위험 감소·이동 부담·수요 적합도)으로
  **비지배(front) 관계**를 계산해 후보별 `pareto_rank`/`pareto_status`를 부여.
- 축을 늘리면 거의 모든 후보가 front에 올라 변별력이 사라지므로 2~4개로 제한합니다.
- 후보 간 지배 관계를 확인하는 **보조 검증**이며 순위를 대체하지 않습니다.

## 6. DQN의 역할과 반영 조건

- 구현: `services/dqn_service.py` 등. **참고용 보조**이며 VHS를 대체하지 않습니다.
- **자동 실행 없음** — 사용자가 분석 및 검증 → DQN 학습 탭에서 버튼을 눌렀을 때만 학습/추론.
- 반영 조건: 상태 정상 + 현재 데이터 signature 일치 + 안정성 통과. 이때만 낮은 비중(`dqn_reference_score`)으로 참고.
- 검토 필요/불안정/데이터 불일치 결과는 비교표에만 표시하고 최종 추천에는 넣지 않습니다.
- 공통 행동 어휘는 `dqn_service.ACTION_LABELS` 하나로 통일(숫자/한글/영문/별칭 정규화, 알 수 없는 값은 "확인 필요").
- original/balanced 학습 결과·모델·출력 파일은 섞지 않습니다(`docs/VALIDATION.md`).

## 7. 민감도와 순위 안정성

- `vhs_score_engine.weight_sensitivity`: 활성 가중치를 ±30% 섭동·재정규화 후 **Top1 유지율**,
  평균 순위 변동(volatility), **취약 요소**(Top1을 뒤집는 요소)를 계산.
- `decision_support.recommendation_stability`가 이를 한 상태로 요약: `안정` / `검토 필요` / `불안정` / `계산 불가`.
  - 유지율 ≥ 0.999 & 변동 낮음 → 안정, ≥ 0.80 → 검토 필요, 그 외 → 불안정, 시나리오<2 → 계산 불가.
  - 변동 허용치는 후보 수에 비례합니다(`max(0.5, 5% × 후보 수)`). 후보가 많으면 하위권이 조금
    섞이는 것은 자연스러우며, 그것으로 추천 자체를 불안정으로 판정하지 않습니다.

### 후보별 강건성 (`candidate_robustness`)

- 같은 ±30% 섭동에서 후보마다 **Top-1 유지율 · Top-3 유지율 · 평균/최선/최악 순위 · 순위 변동 폭**을
  계산해 `안정 / 검토 필요 / 변동 가능성 큼` 상태를 부여합니다(`robustness_status`).
- **순위를 바꾸지 않습니다.** VHS가 순서를 정하고, 강건성은 그 순서를 얼마나 믿을 수 있는지만
  알려줍니다. 자동 교체는 하지 않습니다.

## 8. 신뢰도 계산 (DQN 비의존)

- `decision_support.recommendation_confidence`. **DQN 없이도 계산 가능한 요소**만 사용:
  데이터 완전성, 실행 가능성 통과율, Top1 점수 격차, 순위 안정성, Greedy 방향 일치, Pareto 최상위 포함.
- 활성 요소만 가중 평균(재정규화) → 0–100 점수 → `높음(≥75)` / `보통(≥55)` / `낮음` / `계산 불가`.
- DQN이 정상 상태이고 방향이 일치하면 **소폭 가산만** 가능하며, DQN 부재/불안정은 **신뢰도를 낮추지 않습니다**.
- 계산 불가 입력이면 상태 `계산 불가`, 점수 `None`.
- **신뢰도는 성공 확률이 아닙니다.** 통계적으로 calibration된 값이 아니라 상대적 의사결정
  신뢰 지표이므로, 화면은 `높음/보통/낮음` 등급을 먼저 보여주고 숫자는 상세에만 둡니다.

## 9. 추천 이유 생성

- `services/v2_summaries.recommendation_reason` + 실행 가능성 상태. 생성형 문장이 아니라 **실제 계산값 기반**.
- 화면 기본 노출은 **최대 3개**의 짧은 이유(예: 상위 VHS, 예상 절감액, DC 경유/직접 이동, Greedy 일치 등).
- 수식·가중치·정규화·Pareto 정의·DQN 학습 이론은 화면에 넣지 않고 이 문서에 둡니다.

## 9-1. 후보 판단 기록 (candidate ledger)

구현: `services/candidate_ledger.py` + `services/candidate_lineage.py`. 생성된 모든 후보
(추천·제외 포함)에 대해 **화면 전체가 공유하는 단일 판단 기록**을 만듭니다. 파이프라인은
실행 가능성 게이트의 **전체 주석 결과**(feasible + blocked)로 기록을 만들기 때문에 제외 후보가
목록에서 사라지지 않습니다.

각 기록의 핵심 필드: `candidate_id`, `route_id`, `data_signature`, `status`, `status_code`,
`blocks_recommendation`, `short_reason`, `recommendation_reasons`, `exclusion_reasons`,
`quantity_basis`, `source_references`, `score_components`, `confidence`, `stability`, `calculated_at`.

**후보 식별자(candidate_id).** `상품·출발·도착·경로유형·DC·데이터 signature`로 구성한 안정적
해시입니다. 화면 정렬 순서와 무관하며 DataFrame index를 쓰지 않고, DIRECT/VIA_DC·DC01/DC02·서로 다른
signature를 구분합니다. 내부 식별자이므로 기본 화면에는 노출하지 않습니다(`route_id`는 조회용).

**상태 체계.** 실행 가능성 3-상태를 확장하되 **새로운 판단을 만들지 않고** 재분류만 합니다:

- `추천` — 실행 가능(추천 가능) 후보 중 최종 1순위
- `추천 가능` — 실행 가능하지만 1순위가 아님
- `확인 필요` — 실행 가능성 `데이터 확인 필요`
- `이동 불가` — 물리적으로 불가능(출발=도착, 재고 초과, 중복 등)
- `데이터 부족` — 경로/DC 정보 부재(`no_route`, `via_dc_missing_dc`)
- `계산 불가` — 이동 수량 자체를 계산할 수 없음(수량 값 없음)

상태 버킷은 가산적입니다: `추천 후보(추천+추천 가능) + 확인 필요 + 이동 불가 + 데이터 부족 + 계산 불가
== 전체 생성 후보 수`. `이동 불가+데이터 부족+계산 불가`는 실행 가능성 게이트의 `blocked_count`와 항상 일치합니다.

**이동 수량 결정 근거(quantity_basis).** 출발 재고, 적용 재고 하한, 출발 이동 가능량(재고−하한),
도착 부족량을 확인해 **실제 제한 조건의 최솟값**을 근거로 표시합니다. 예: 출발 가능 20 · 도착 부족 12 →
"도착 점포 부족량을 기준으로 12개 이동을 권장합니다." 값을 계산할 수 없으면 0으로 위장하지 않고
"확인 필요"로 표시합니다. 수량이 0 이하이면 추천 후보로 남기지 않습니다.

**추천 이유 vs 제외 이유.** 추천/추천 가능 후보만 `recommendation_reasons`(최대 3개, 실제 계산값 기반),
제외/확인 필요 후보만 `exclusion_reasons`(기본 1~2개, 가장 먼저 고칠 원인 우선)를 가집니다. 두 목록은
섞이지 않으며, 제외 후보에는 거짓 추천 이유를 만들지 않습니다.

## 9-2. 원본 행 계보 (source lineage)

구현: `services/candidate_lineage.py`. 후보마다 판단에 사용된 **원본 파일 행**을 연결합니다:
출발 재고 행, 도착 수요 행, 상품 정보 행, 경로 정보 행(VIA_DC는 출발→DC·DC→도착 2행), DC 정보 행.
행/컬럼 해석은 데이터 관리 문제 목록과 **동일한 함수**(`data_issues._resolve_column`/`_row_number`)를
사용하므로 같은 원본 위치를 가리킵니다. 원본에서 행을 찾지 못하면 임의 행을 만들지 않고 `추적 불가`로
명시합니다. 계보 메타데이터는 VHS·수치형 피처에 섞이지 않습니다.

## 9-3. 결정 지표와 알고리즘 버전

- `services/decision_metrics.py`가 후보별 **순효과, 이동 가능량/부족량, 이동 후 위험, 수요 시나리오**를
  한 번만 계산하고, 실행 가능성 게이트와 VHS가 같은 숫자를 읽습니다. 두 계층이 같은 사실을
  서로 다르게 판단하는 일이 구조적으로 불가능합니다.
- 수요 시나리오(보수적/기준/공격적, ±1σ)는 파일에 `demand_std`가 **실제로 있을 때만** 만듭니다.
  없으면 `계산 불가`이며 표준편차를 지어내지 않습니다.
- `ALGORITHM_VERSION`(현재 `vhs-2.2`)은 구성요소·정규화·hard constraint·tie-break가 바뀌어
  같은 파일이 다른 순위를 낼 수 있게 되면 올립니다. 결과 객체와 검증 리포트에 기록되며
  (`analysis_timestamp`·`data_signature`와 함께) 사용자 기본 화면에는 표시하지 않습니다.

## 10. 알고리즘의 한계

- 후보 생성은 rule-based입니다. 그 후보를 동시에 배분하는 실행계획만 정수 최적화를 사용하며, 원래 후보
  생성이나 VHS 점수 계산을 정수 최적화로 대체하지 않습니다.
- 안전재고/최소재고가 입력되면 해당 등록값을 우선합니다. 없을 때만 `demand_std` 기반 추정을 사용합니다.
- 현재 지원 범위는 재고 시트의 점포×상품 수준입니다. 점포 전체·상품 전체 기본값, 기간별 정책 선택,
  단위 환산은 아직 지원하지 않습니다.
- VIA_DC 후보는 입력 `routes`에 출발→DC·DC→도착 경로가 모두 있을 때만 생성됩니다.
- 현재 비용·절감액은 후보 권장 수량에 비례해 축소합니다. 차량별 고정비/용량, 최소 출고 단위, 경로별
  처리 용량이 실제 컬럼으로 제공되면 그때 비용 곡선과 용량 제약을 추가해야 합니다.
- DQN은 학습 데이터 품질(라벨 분포)에 따라 정상에 도달하지 못할 수 있으며, 그 경우 참고에서 제외됩니다.

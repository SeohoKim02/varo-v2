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
10. **최종 추천/순위·KPI 산출** — VHS 순위 기준.

DQN은 이 흐름에 자동으로 들어오지 않습니다(6절).

## 2. VHS (Varo Hybrid Score)

구현: `services/vhs_score_engine.py`. 결정적(deterministic): 같은 입력 → 같은 결과.

### 구성요소(10개)와 방향

| 구성요소 | 라벨 | 방향 |
|---|---|---|
| `savings_score` | 절감액 | 클수록 좋음 |
| `disposal_risk_score` | 폐기 위험(회피 효과) | 클수록 좋음 |
| `demand_fit_score` | 수요 적합도 | 클수록 좋음 |
| `inventory_balance_score` | 재고 균형 | 클수록 좋음 |
| `route_cost_score` | 이동비용(낮을수록 점수↑) | 비용이 낮을수록 좋음 |
| `feasibility_score` | 실행 가능성 | 클수록 좋음 |
| `promotion_score` | 프로모션 대안 | 클수록 좋음 |
| `greedy_score` | Greedy 비교 | 클수록 좋음 |
| `confidence_score` | 신뢰도 | 클수록 좋음 |
| `dqn_reference_score` | DQN 참고 | 클수록 좋음(선택형, 기본 0) |

### 정규화

- 모든 구성요소는 **0–100** 범위로 정규화(`_normalize_high` / `_normalize_low` / `_bounded`).
- 값이 클수록 좋은 요소는 min–max 정규화, 작을수록 좋은 요소(비용·거리·시간)는 반전 정규화.
- 결측/전부 동일/비수치 → **중립값 50** (좋은 점수로 위장하지 않음). `dqn_reference_score`는 DQN 미사용 시 0.
- 한 요소의 원단위가 전체를 지배하지 않도록 정규화 후 가중합.

### 가중치

- 기본 가중치 `BASE_WEIGHTS`, 허용 범위 `WEIGHT_BOUNDS` — **단일 위치**(`vhs_score_engine.py`)에 정의.
- `optimize_weights`가 커버리지·분산 신호로 조정하고 `_project_to_bounds`로 **범위 내 + 합계 1.0**을 보장.
- DQN 미사용 시 `dqn_reference_score` 상한을 0으로 고정.

### 최종 점수·등급·동점 처리

- `auto_vhs_score = Σ(구성요소 × 가중치)`를 0–100으로 clip.
- 순위: `rank(method="first", ascending=False)` — **동점은 먼저 나온 행이 앞 순위**(결정적).
- 등급: 80↑ 최적 · 65↑ 권장 · 50↑ 검토 · 그 외 보류.
- 구성요소별 점수는 각 후보 행에 보존되어 왜 높은지 추적 가능.

## 3. 실행 가능성 게이트 (feasibility)

구현: `services/feasibility.py`. VHS **점수 전이 아니라 최종 후보 확정 시점**에 적용되어, 실행 불가능한
이동은 낮은 점수로 남기지 않고 **제외**합니다. 3-상태:

- `추천 가능` — 모든 하드 조건 통과
- `데이터 확인 필요` — 유지하되 입력 부족/불확실 표시
- `이동 불가` — 하드 위반, 최종 추천에서 제외(사유 기록)

**하드 블록(이동 불가):** 출발지=도착지, 이동 수량 ≤0/NaN/inf, 경로 유형이 DIRECT/VIA_DC가 아님,
VIA_DC인데 DC 없음, 이동 수량 > 출발 점포 재고(이동 후 음수), 동일 (상품·출발·도착) 중복.

**소프트 플래그(데이터 확인 필요):** 이동 비용/절감액 계산 불가, 출발 점포 재고 데이터 없음,
이동 후 안전재고 미만, 도착 점포 필요량 대비 과잉 공급(3배 초과).

화면에는 3-상태만 노출하고, 세부 `reason_code`는 진단/로그·이 문서에 둡니다.

## 4. Greedy의 역할

- `heuristic_optimizer.add_heuristic_scores`가 즉시 이익(절감−비용) 중심의 **단순 기준선**을 만듭니다.
- VHS 순위와 Greedy 순위/전략의 일치 여부(`strategy_match`)는 비교·검증 지표일 뿐, 최종 판단이 아닙니다.

## 5. Pareto의 역할

- 구현: `services/pareto_analysis.py`. 절감액·폐기 위험 감소·경로 비용·실행 가능성·수요 적합도 등
  다목적으로 **비지배(front) 관계**를 계산해 후보별 `pareto_rank`/`pareto_status`를 부여.
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

## 8. 신뢰도 계산 (DQN 비의존)

- `decision_support.recommendation_confidence`. **DQN 없이도 계산 가능한 요소**만 사용:
  데이터 완전성, 실행 가능성 통과율, Top1 점수 격차, 순위 안정성, Greedy 방향 일치, Pareto 최상위 포함.
- 활성 요소만 가중 평균(재정규화) → 0–100 점수 → `높음(≥75)` / `보통(≥55)` / `낮음` / `계산 불가`.
- DQN이 정상 상태이고 방향이 일치하면 **소폭 가산만** 가능하며, DQN 부재/불안정은 **신뢰도를 낮추지 않습니다**.
- 계산 불가 입력이면 상태 `계산 불가`, 점수 `None`.

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

**이동 수량 결정 근거(quantity_basis).** 출발 재고, 안전재고, 출발 이동 가능량(재고−안전재고),
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

## 10. 알고리즘의 한계

- 후보 생성은 rule-based이며 MILP/전역 최적화가 아닙니다.
- 안전재고는 `demand_std` 기반의 **보수적 추정치**이며 정책 상수가 아닙니다.
- VIA_DC 후보는 입력 `routes`에 출발→DC·DC→도착 경로가 모두 있을 때만 생성됩니다.
- DQN은 학습 데이터 품질(라벨 분포)에 따라 정상에 도달하지 못할 수 있으며, 그 경우 참고에서 제외됩니다.

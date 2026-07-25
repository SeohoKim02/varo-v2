# Varo V2

## 교수님 피드백 반영

기존 Varo는 졸업작품으로 제출된 버전이고, Varo V2는 SCC 프로젝트 기간 동안 기능 완성도와 데모 품질을 높이기 위해 개선 중인 버전입니다. 피드백은 두 갈래였습니다. (1) 시뮬레이션 모델은 성능 평가용으로 적합하지만 **데모용 UI가 부족**하다, (2) Greedy와 DQN만으로는 **학술적 확장성이 약하다**. 아래처럼 반영했습니다(목표는 **연구 확장 가능성·비교 검증 구조·데모 품질 개선**이며, 완성도를 과장하지 않습니다). 학술 설명은 홈이 아니라 분석 및 검증 페이지와 이 문서에만 둡니다.

**DQN 학습 실행 방식(정직한 상태 표기)**
- DQN은 자동 실행되지 않습니다. 분석 및 검증 → DQN 학습 탭에서 사용자가 버튼을 눌렀을 때만 (1) 현재 적용 데이터, (2) 선택한 DQN 샘플, (3) 10개 샘플 순차 방식으로 학습합니다.
- PyTorch가 없으면 "실행 환경 필요"로만 표시하고 가짜 결과를 만들지 않습니다. 미학습 상태에서는 비교 구조만 준비되어 있습니다.
- 학습 결과라도 안정성 검사를 통과해야 VHS에 반영됩니다. 예: 2점포 극단 샘플(후보 4개)이나 target 행동이 90% 이상 한쪽으로 쏠린 샘플은 규칙대로 "검토 필요"가 되어 비교표에만 표시되고 최종 점수에는 들어가지 않습니다.

**학습 데이터 품질 진단 + 균형형 샘플(원본 무수정)**
- 원본 DQN 샘플 10개는 target 행동이 대부분 "보류"로 쏠려 있어(진단 결과 대다수 불안정/검토 필요) DQN이 정상 상태에 도달하기 어렵습니다. 이는 코드 오류가 아니라 **학습 데이터 특성**입니다.
- 이를 검증 가능하게 만들기 위해 `services/dqn_quality.py`로 각 샘플의 action 분포를 진단(정상/검토 필요/불안정/학습 부족)하고, `services/dqn_balanced.py`로 **원본을 전혀 수정하지 않은 균형형 파생 데이터**를 만듭니다. 균형형은 수량·거리·비용·유통기한·수요·폐기 같은 핵심 수치는 그대로 두고, 각 후보의 특성 affinity + 쿼터 배분으로 target_action만 4종 이상으로 재라벨링합니다(폐기는 폐기 위험이 높을 때만, DC 경유는 VIA_DC일 때만 부여해 비현실적 추천을 만들지 않음).
- 파생본은 `outputs/dqn_balanced_samples/`에만 저장하며 `derived_from`·`original_sample_id`·`balance_policy`·`generated_at`을 기록합니다. 이 진단·균형형은 **DQN 모델 성능이 아니라 학습 데이터 품질을 다루는 것**임을 UI와 이 문서에 명시합니다.
- 실측(PyTorch 2.12 설치 환경): 원본 샘플 학습은 "검토 필요"(예측이 보류로 붕괴), 동일 샘플의 균형형 학습은 "정상"(예측이 6종으로 분산)으로 나뉘어, 상태 구분이 실제로 동작함을 확인했습니다.
- DQN 학습 탭의 `원본 vs 균형형 학습 비교` 버튼은 10개 샘플 각각을 원본→균형형으로 짧게(검증용 episodes) 실제 학습해, 후보 수·target/예측 행동 종류·loss 시작→끝·상태·VHS 반영 가능 여부를 한 표로 보여주고, 그 근거를 `outputs/dqn/comparison_report_{timestamp}.json`(sample_id·variant·action/prediction 분포·initial/final loss·reward 요약·stability_status·model_path 등)으로 저장합니다.
- 학습 결과 파일명에 `original`/`balanced` 마커와 sample_id·store/dc·episodes·lr·timestamp를 포함합니다.

**UI 밝은 테마 / 단순화 방향**
- 전체 화면을 밝은 실사용 앱 톤으로 통일했습니다. `.streamlit/config.toml`에 light 테마(배경 `#FAFBFC`, 보조 `#F3F5F7`, 텍스트 `#1F2937`, 포인트 `#2d6fa8`)를 명시해 OS 다크모드와 무관하게 검은 바탕이 나오지 않게 하고, `styles.py`에서 헤더·사이드바·버튼·업로더·입력·탭·표·확장·metric 등 Streamlit 네이티브 위젯을 흰색/연회색 배경 + 진한 글씨로 오버라이드했습니다.
- 선택/강조는 연한 파랑(`#EAF3FF`)으로 통일합니다. 사이드바 현재 페이지와 주요 액션 버튼은 연한 파랑 배경 + 진한 글씨, 비선택 버튼은 흰색 + 진한 글씨입니다.
- 상태 배지·재고 상태 pill(과잉/부족/정상/이동 대상)은 모두 아주 연한 배경 + 진한 글씨로 구분합니다. 강한 검정·네온·과한 그라데이션은 쓰지 않습니다.
- 홈은 실사용 대시보드(상태바·KPI·시뮬레이션·현재 실행 경로·Top5)로, 분석 및 검증은 상세 검증 페이지로 역할을 분리합니다. 긴 알고리즘 설명·함수명·검증표·원본표·다운로드는 홈에 노출하지 않고 데이터 관리/분석 및 검증/이 문서로 보냅니다.

**UI 개선 방향 (데모 가독성)**
- 홈 네트워크의 점포 노드에 재고 상태를 색상 pill로 직관화했습니다: `과잉`·`부족`·`정상`·`이동 대상`(추천 출발 점포). 범례에도 상태 색을 표시합니다.
- 트럭은 점이 아니라 캐빈+화물칸+바퀴 SVG로 그리고, Top1 경로를 가장 선명하게(굵기·불투명도) 표시합니다. 직접 이동은 실선, DC 경유는 점선으로 구분하고, 2DC 샘플은 DC01/DC02를 모두 구분해 배치합니다.
- 홈은 상태바·KPI·시뮬레이션·현재 실행 경로 Top3·추천 Top5(순위/상품/출발/도착/경로/수량/예상 절감액)만 노출합니다. 추천 실행 페이지는 기본 컬럼 표만 보이고 VHS/Greedy/DQN/Pareto/신뢰도는 접힌 상세 영역에서만 봅니다.
- 애니메이션은 SMIL `animateMotion` 기반(느림 16초·보통 10초·빠름 6초)으로, `time.sleep`·강제 rerun 루프를 쓰지 않습니다.

**알고리즘 개선 방향 (연구 확장용 비교 기준)**
- 단순 Greedy/DQN 비교를 넘어 **Varo Hybrid Score 다목적 자동 가중치**(10개 요소, 가중치 합 1.0, 결측·분산·후보 수 기반 재정규화)를 중심에 둡니다.
- **Pareto 후보 분석**: 절감액·폐기 위험 감소·경로 비용·실행 가능성·수요 적합도 5개 목적함수로 비지배 정렬(front)을 계산해 후보별 `pareto_rank`/`pareto_status`를 만듭니다(`services/pareto_analysis.py`).
- **가중치 민감도 분석**: VHS 가중치를 ±30% 섭동해 Top1 유지율, 평균 순위 변동, 취약 요소를 계산합니다(`services/vhs_score_engine.py:weight_sensitivity`).
- **최적성 검증**: 소규모 후보에서 제한 탐색 기반 best와 VHS 추천의 `optimality_gap`을 비교합니다.
- 분석 및 검증 페이지의 `Greedy와 DQN` 탭에 **VHS vs Greedy vs DQN vs Pareto** 비교표와 Pareto 요약을, `민감도 분석` 탭에 가중치 민감도를 표시합니다. 이 구조는 성능 완성 주장이 아니라 "연구 확장용 비교 기준"입니다.

**DQN 샘플 10개 활용 방식**
- 사용자가 만든 DQN 학습 샘플 10개를 읽기 전용으로 연결합니다. 데이터 관리 페이지에서 선택→`DQN 샘플 불러오기`로 기존 `load_and_apply` 흐름에 태워 적용합니다.
- DQN은 분석 및 검증의 DQN 학습 탭에서 버튼을 눌렀을 때만 현재 샘플 기준으로 학습하고, `outputs/dqn/`에 `data_signature`·`sample_id`·`store_count`·`dc_count`·`episodes`·`learning_rate`·`reward_history`·`loss_history`·`action_distribution`·후보별 action/confidence/reference를 저장합니다.
- signature가 현재 데이터와 다르면 과거 결과로 처리해 VHS에 반영하지 않고, 정상이며 일치할 때만 낮은 참고 비중으로 반영합니다. 결과를 가짜로 만들지 않습니다.

**PyTorch / 실행 로그 참고**
- DQN 학습 탭 상단 배지가 상태를 짧게 표시합니다: `PyTorch 설치 필요` / `CPU 학습 가능` / `GPU 학습 가능`. PyTorch가 없어도 앱은 죽지 않고, 진단·샘플 로드·VHS/Greedy/Pareto 비교는 그대로 동작하며 학습 버튼만 비활성화됩니다. 상태 판정은 `services/dqn_service.py:get_torch_runtime_status`(available/version/device/cuda_available/can_train/message)가 담당합니다. PyTorch 설치가 필요하면 사용 환경의 공식 안내를 따르세요(앱은 pip install을 실행하지 않습니다).
- **서버 종료 또는 브라우저 새로고침 중 나오는 `ConnectionResetError [WinError 10054]`(asyncio 소켓 종료) 로그는 앱 오류가 아닙니다.** Windows/Streamlit이 연결 정리 과정에서 남기는 무해한 메시지이며, `services/log_hygiene.py`가 이 한 종류의 종료 레코드만 좁게 걸러냅니다(다른 asyncio/앱/import 오류는 그대로 표시).

**남은 한계 / 실제 확인 필요**
- Pareto·민감도·최적성은 현재 후보 집합 기준의 비교 지표이며, 대규모(수백 후보) 최적성은 제한 탐색 근사입니다.
- 실제 DQN 학습 실행·모델 저장은 PyTorch가 설치된 환경에서 버튼 실행으로 최종 확인해야 합니다(자동 실행하지 않음).
- 원본 DQN 샘플은 개발 머신에서 OneDrive Desktop 리디렉션 이후 `OneDrive/Desktop/Projects/Varo/Varo_DQN_training_samples_10pack/`에 있습니다. 탐색은 환경변수 `VARO_DQN_SAMPLES_DIR` → 일반 Desktop → OneDrive Desktop → 프로젝트 상위 폴더 순서로 자동 수행되며, 폴더가 없으면 앱이 중단되지 않고 "DQN 샘플을 찾지 못했습니다" 상태만 표시합니다.

**실행 방법**은 아래 `## 실행` 섹션을 참고하세요(`compileall` → `unittest` → `streamlit run app_v2.py`).

## Kakao 지도와 DQN 연결

- Kakao 지도는 `경로 상세` 페이지에서만 로딩합니다. 홈, 추천 실행, 분석 및 검증 페이지에서는 SDK를 로딩하지 않습니다.
- 지도 키는 `st.secrets["KAKAO_JAVASCRIPT_KEY"]` 또는 환경변수 `KAKAO_JAVASCRIPT_KEY`에서 읽습니다. API key는 코드에 저장하지 않습니다.
- key가 없으면 경로 상세 지도 영역에 `Kakao 지도 키가 설정되면 실제 지도 표시가 가능합니다.` 안내만 표시하고 앱은 계속 동작합니다.
- DIRECT 경로는 출발/도착 마커와 단일 polyline을 표시합니다. VIA_DC 경로는 출발/DC/도착 마커와 출발->DC->도착 polyline을 표시합니다.
- 다중 DC에서는 추천 행의 `dc_id`를 우선 사용하고, 없으면 `dc_name`, 둘 다 없을 때만 좌표 기준으로 결정적인 fallback DC를 선택합니다.
- DQN은 `분석 및 검증 > DQN 학습` 탭에서 사용자가 버튼을 눌렀을 때만 실행합니다. 홈, 추천 실행, 경로 상세 진입만으로 학습/추론/모델 로딩을 하지 않습니다.
- PyTorch가 없으면 `DQN 실행 환경이 필요합니다.` 상태로 표시하고 앱은 계속 동작합니다. 의존성 설치는 자동으로 하지 않습니다.
- DQN 학습 결과는 `outputs/dqn/` 아래에 `data_signature`와 함께 저장합니다. 현재 데이터와 signature가 다르면 과거 결과로 보고 VHS에 반영하지 않습니다.
- DQN 상태가 `정상`이고 현재 데이터와 일치할 때만 `dqn_reference_score`가 VHS 자동 가중치의 낮은 비중 요소로 들어갑니다.
- DQN이 `학습 필요`, `검토 필요`, `학습 부족`, `과거 결과`, `실행 환경 필요` 상태면 최종 추천 점수에는 반영하지 않고 비교표에 상태만 표시합니다.
- Greedy/VHS/DQN 비교표는 분석 및 검증 페이지에서 확인합니다. 홈 Top5에는 VHS, Greedy, DQN, route_id를 표시하지 않습니다.

재고 재배치 추천을 검토·검증하는 Streamlit 앱입니다. **원본 폴더나 백업 ZIP 없이 독립 실행**되며,
승인된 비-DQN 알고리즘을 내부에서 직접 재계산합니다.

## 현재 동작 기준

- **재계산 기준**: 현재 Varo V2는 내부 비-DQN 알고리즘으로 직접 재계산한 결과(`실제 V2 내부 알고리즘 재계산 결과 기준`)로 동작합니다.
  - 연결된 알고리즘: VHS 재계산, Greedy, 경로 분석(직접/DC 경유·컷라인·시간창·최소비용·네트워크), 프로모션, 신뢰도, Optimality Gap, ABC·회전율·폐기위험·수요예측·안전재고·EOQ·점포상품 매칭·클러스터링.
  - 보류(설계상): 민감도 분석(`varo_sensitivity`)과 사유 생성(`vhs_reason`)은 추가 입력 그룹 정의가 필요해 보류이며, 핵심 재계산·추천 판단에는 영향이 없습니다.
- **VHS 점수 차이**: 업로드 파일의 사전 계산 VHS와 V2 재계산 VHS는 계산 기준·입력 컬럼 차이로 다를 수 있습니다.
  - 샘플 기준: 업로드 VHS 평균 ≈ 83.4, 재계산 VHS 평균 ≈ 42.7.
  - 두 값과 차이는 `분석 및 검증 → VHS 분석`과 `검증 리포트`, 그리고 다운로드한 `VHS비교` 시트에서 확인할 수 있습니다.
  - 입력 컬럼이 부족한 일부 VHS 구성요소는 원본 규칙대로 중립값(50)이 적용됩니다. 적용 현황(전체/계산/중립/제외 구성요소 수)은 `VHS 분석` 탭과 `VHS중립값` 시트에 표시됩니다.
- **V2 내부 요약 기능(연결)**: 원본 모듈을 가져오지 않고 V2 내부 결과로 계산하는 보조 요약을 제공합니다.
  - **V2 민감도 요약**: 비용·거리·수량·VHS·절감액 기준으로 후보별 순위 변동 위험을 낮음/보통/높음/제한적으로 표시(`분석 및 검증 → 민감도 분석`).
  - **V2 추천 사유 요약**: 재계산 VHS·절감액·경로 유형·프로모션·Greedy 일치·신뢰도 기준의 rule-based 추천 사유(`추천 실행`/`경로 상세` 상세 카드, 선택 route_id 기준).
  - 원본 `varo_sensitivity`/`vhs_reason`은 입력 그룹 정의가 필요해 **보류**이며, 위 V2 요약이 대체로 **연결**되어 있습니다(보류와 DQN '제외'는 다른 상태).
- **DQN**: 과거 학습 결과(reward 이상치 가능성)는 계속 제외합니다. DQN은 V2 현재 데이터로 사용자가 직접 학습 실행한 경우에만 상세 참고값으로 표시되며, 기본값은 추천 점수 미반영입니다.
- **카카오 지도**: 경로 상세 페이지에서만 SDK를 로딩합니다. API key는 secrets 또는 환경변수로만 읽고 코드에 저장하지 않습니다.

## 업로드 데이터 형식 (실제 엑셀 견고화)

- **필수 시트**: `stores`, `products`, `inventory`, `routes`. 이 4개가 있으면 로드됩니다.
- **선택 시트**: `v2_recommendations`(추천 결과), `transport_modes`, `config`, `Quality_Check`, `README`.
- **추천 결과 시트가 없으면**: 재고·상품·점포·경로 데이터로 **V2 생성 후보**(route_id `V2C001`…)를 자동 생성합니다(rule-based, MILP/DQN 아님).
  - 출발지: 상품별 중앙값 대비 초과재고 또는 잔여 유통기한이 짧은 점포.
  - 도착지: 같은 상품을 취급하고 재고가 낮거나 수요가 높은 점포.
  - 경로: 직접 경로가 있으면 **DIRECT**, 없고 DC 경유가 가능하면 **VIA_DC**, 둘 다 가능하면 비용이 낮은 경로 선택. 경로가 없으면 보류.
  - **candidate_score(0~100)**: 유통기한 긴급도·초과재고·도착지 수요/부족·예상 절감액·경로 가능성·거리 패널티로 구성한 후보 정렬 보조값(VHS와 별개, DQN 미반영).
  - **추천 수량**: 출발지 초과재고와 도착지 수요/부족 범위 안에서 산정하며 최소 1개·재고 초과 금지, 잔여 유통기한이 짧으면 과도한 이동을 제한합니다.
  - 자기 자신 전송·중복(같은 상품/출발/도착) 후보는 제외합니다.
- **컬럼 자동 매핑**: 한글/영문 컬럼명을 V2 표준 컬럼으로 자동 표준화합니다. 예) `상품명→product_name`, `출발점포→source_name`, `추천수량→transfer_qty(recommended_qty)`, `절감액→expected_saving`, `기존VHS→vhs_score`, `거리→distance_km`, `경로유형→route_type`.
- **숫자 정규화**: `"10,000원"→10000`, `"3.5km"→3.5`, `"15분"→15`. 빈 행은 자동 제거하고, 변환 불가 값은 앱을 멈추지 않고 검증 메시지로 표시합니다.
- **날짜 자동 환산**: 날짜 컬럼(`만료일`, `소비기한`, `expiry_date` 등)을 업로드 시점 기준 `days_to_expiry`(남은 일수)로 환산합니다(`2026-06-30`, `2026.07.05`, `2026/07/01`, Excel serial 지원). 기존 `days_to_expiry` 값이 있으면 우선하며, 해석 불가한 값은 보류로 처리하고 앱을 멈추지 않습니다.
- **검증 메시지**: 데이터 관리 → "업로드 품질 점검"에서 자동 매핑 컬럼 수·누락 필수 컬럼·숫자 변환 실패·빈 행 제거·추천 생성 방식·분석 가능 여부를 확인할 수 있습니다.
- **업로드 실패 방지**: 손상된 파일을 올려도 앱이 죽지 않고 오류 메시지를 표시하며, 이전 정상 데이터와 메뉴 이동이 유지됩니다.

## 화면 구성과 네비게이션

- **네비게이션은 좌측 사이드바**에 있습니다(앱은 `initial_sidebar_state="collapsed"`로 시작하며, 좌상단 `›`/`☰`로 펼칩니다). 가로 메뉴 막대는 없앴습니다.
  - 5개 페이지: `운영 현황`(홈) · `추천 실행` · `경로 상세` · `분석 및 검증` · `데이터 관리`. 현재 페이지는 사이드바에서 강조 표시되고, 선택한 `selected_route_id`는 페이지를 이동해도 유지됩니다.
- **상단 바**: 좌측 `VARO V2`, 우측에 상태 배지(데이터 적용/알고리즘 상태)와 파일명, 그리고 `데이터 교체` 토글만 둡니다. 중복되던 상단 `데이터 관리` 버튼은 제거했습니다.
- 페이지 이동·중복 안내·개발자용 문구는 화면에서 줄이고 설명은 이 문서로 옮겼습니다.

## 홈(운영 현황) 구성

홈은 "결과 확인" 화면입니다. 위에서 아래로 다음 순서로만 구성합니다.

1. **헤더**: 제목 `Varo 운영 결과` + 한 줄 설명, 우측에 작은 배지 4개(`데이터 적용`/`알고리즘 연결`/`DQN 제외`/`카카오 미연결`).
2. **KPI 5장**(숫자 위주, 캡션 없음): 처리 대상 재고 · 추천 경로 · 예상 절감액 · 평균 VHS · 상태.
3. **4단계 흐름 카드**: 엑셀 업로드 → 재고 분석 → 이동 추천 → 절감 확인.
4. **추천 경로 이동 현황**: 물류 네트워크 시뮬레이션 + 우측 `현재 실행 경로`(Top 3) 패널.
5. **추천 Top 5 표**: 순위 · 상품 · 출발 · 도착 · 경로 · 수량 · 예상 절감액 7개 컬럼만 표시합니다(route_id/VHS/Greedy/DQN/신뢰도/상태 등은 숨김).

홈에는 다운로드, 검증·알고리즘 표, 원본 데이터, 함수/모듈명, 페이지 이동 버튼을 두지 않습니다. 상세 표·다운로드는 각 전용 페이지에서 제공합니다.

## 홈 시뮬레이션(물류 네트워크)

- `stores`/`nodes`와 추천 결과에서 점포·DC를 동적으로 구성하며 특정 점포명이나 DC ID를 고정하지 않습니다.
- 위도·경도가 충분하면 SVG 캔버스에 정규화하고, 없으면 region과 node/store ID 순서로 결정적인 1~3개 링 배치를 사용합니다(랜덤 없음).
- DC 1·2·3개는 중앙 허브 형태, 4개 이상은 중앙 클러스터로 배치합니다. 점포는 카드, DC는 더 큰 물류센터 카드로 그리고 라벨이 겹치지 않도록 분리합니다.
- 차량은 점이 아니라 **탑차 SVG 아이콘**으로 표시하며, 냉동·냉장 운송은 파란 계열로 구분합니다. 경로는 **직접 이동=실선, DC 경유=점선**입니다.
- 애니메이션은 Python 루프(`time.sleep`/반복 `st.rerun`) 없이 **브라우저 SMIL `animateMotion`**으로 동작합니다.
  - `시작`을 누르면 재생, `일시정지`로 정지(차량이 출발 노드에 고정), `다시 시작`으로 재생을 다시 시작합니다.
  - 속도 `느림/보통/빠름`은 한 바퀴 소요 시간(16/10/6초)으로 반영됩니다.
- 기본 화면에는 추천 Top 3의 DIRECT/VIA_DC 경로와 차량만 강조합니다. `전체 경로 보기`는 **기본 꺼짐**이며, 켜면 최대 12개 고유 구간만 연하게 덧그립니다.
- 레이아웃·SVG는 `st.cache_data`로 캐시하며, 데이터를 다시 읽거나 후보를 매번 재생성하지 않습니다.
- 데이터 교체 시 노드, 경로, 선택 경로와 시뮬레이션 상태(재생/전체 경로 보기)가 새 데이터 기준으로 초기화됩니다.

## 시뮬레이션 검수 샘플

데이터 관리 → 시뮬레이션 검수 샘플에서 아래 파일을 선택해 바로 적용할 수 있습니다.

- samples/Varo_V2_sample_edge_3stores_1dc.xlsx: 점포 3개, DC 1개, 추천 3건
- samples/Varo_V2_sample_small_4stores_1dc.xlsx: 점포 4개, DC 1개, 추천 5건
- samples/Varo_V2_sample_normal_6stores_1dc.xlsx: 점포 6개, DC 1개, 추천 6건
- samples/Varo_V2_sample_standard_8stores_1dc.xlsx: 점포 8개, DC 1개, 추천 8건
- samples/Varo_V2_sample_dual_dc_10stores_2dc.xlsx: 점포 10개, DC 2개, 추천 10건

모든 샘플은 동일한 표준 시트를 사용하며 랜덤 값 없이 결정적으로 생성됩니다. 2DC 샘플의 VIA_DC 추천은 DC01과 DC02를 모두 사용합니다. 샘플 교체 시 선택 경로, 추천 필터, 시뮬레이션 스냅샷과 전체 경로 보기 상태가 새 데이터 기준으로 초기화됩니다. 2점포 및 30점포 이상 사용자용 샘플은 포함하지 않습니다.
## DQN 학습 샘플 10종 연결

사용자가 사전에 만든 DQN 학습용 엑셀 10개를 그대로 연결했습니다. **새 샘플을 만들지 않았고**, 원본은 수정·이동·이름변경·삭제하지 않았습니다.

- 원본 위치: `<사용자 홈>/OneDrive/Desktop/Projects/Varo/Varo_DQN_training_samples_10pack/` (읽기 전용, 절대경로 참조 · Windows Desktop이 OneDrive로 리디렉션되어도 자동 탐색)
- 탐색: `데이터 관리 → DQN 학습 샘플`에서 선택 후 `선택 샘플 불러오기`. `services/sample_catalog.py`의 `discover_dqn_samples`가 알려진 프로젝트 폴더(및 환경변수 `VARO_DQN_SAMPLES_DIR` 경로 오버라이드)를 pathlib으로 탐색합니다. `~$` 임시 파일과 Varo 구조가 아닌 파일은 제외합니다. 파일 수정시간·크기가 바뀌면 카탈로그 캐시가 갱신됩니다.
- **보정본을 만들지 않았습니다.** 컬럼 차이는 `data/normalized_samples`에 복사본을 만들지 않고 로딩 계층(`services/data_loader.py`, `services/column_aliases.py`)에서만 표준화합니다.

| # | 파일명 | 점포 | DC | 상품 | 재고행 | 경로 | 추천 | 검증 |
|---|--------|-----|----|-----|-------|-----|-----|------|
| 01 | Varo_DQN_sample_01_2stores_1dc_fresh_meal.xlsx | 2 | 1 | 6 | 12 | 4 | 4 | 주의 |
| 02 | Varo_DQN_sample_02_4stores_1dc_frozen.xlsx | 4 | 1 | 5 | 20 | 24 | 16 | 주의 |
| 03 | Varo_DQN_sample_03_4stores_1dc_dairy_bakery.xlsx | 4 | 1 | 5 | 20 | 24 | 17 | 주의 |
| 04 | Varo_DQN_sample_04_5stores_1dc_produce.xlsx | 5 | 1 | 6 | 30 | 40 | 30 | 주의 |
| 05 | Varo_DQN_sample_05_5stores_1dc_bakery.xlsx | 5 | 1 | 5 | 25 | 40 | 27 | 주의 |
| 06 | Varo_DQN_sample_06_6stores_1dc_beverage_dry.xlsx | 6 | 1 | 6 | 36 | 60 | 35 | 주의 |
| 07 | Varo_DQN_sample_07_6stores_1dc_meat_egg.xlsx | 6 | 1 | 5 | 30 | 60 | 30 | 주의 |
| 08 | Varo_DQN_sample_08_6stores_1dc_seafood.xlsx | 6 | 1 | 6 | 36 | 60 | 35 | 주의 |
| 09 | Varo_DQN_sample_09_8stores_1dc_meal_kit.xlsx | 8 | 1 | 6 | 48 | 112 | 35 | 주의 |
| 10 | Varo_DQN_sample_10_10stores_2dc_mixed.xlsx | 10 | 2 | 7 | 70 | 270 | 35 | 주의 |

- 검증 `주의`는 경고만 있고 오류가 없는 상태로, 10개 모두 정상 적용되고 파이프라인은 `success`로 연결됩니다. 2점포(01)는 극단 테스트용, 10점포 2DC(10)는 대규모·다중 DC 샘플입니다.
- 10번 샘플은 DC 2개(DC01·DC02)가 표시되고 VIA_DC 추천이 두 DC에 각각 연결됩니다(임의로 한 DC를 고르지 않음).

### 로딩 계층 표준화 (원본 미수정)

DQN 샘플은 기존 V2 샘플과 시트/컬럼 구조가 다릅니다. 화면 코드가 아니라 로더/검증 계층에서만 아래를 처리합니다.

- 별도 `dcs` 시트를 `stores`에 `node_type=DC` 행으로 병합하고, `store_type`(마감임박형 등 업무 분류)을 node_type으로 오인하지 않도록 점포 행은 `node_type=STORE`로 고정합니다.
- `recommendations` 시트명을 인식합니다(기존 `v2_recommendations`와 병행).
- `from_store_id/from_store_name/to_store_id/to_store_name`을 `source_id/source_name/target_id/target_name`으로 표준화합니다.
- 한 경로에 여러 상품 추천이 있어 `route_id`가 중복되면 고유한 `recommendation_id`를 `route_id`로 승격합니다.
- 원본에 없는 `transport_type`, `recommendation_grade`, `reason`은 로딩 시 파생 생성합니다.
- `inventory`에 없는 `dead_stock_qty`(악성재고), `demand_qty`(수요)는 재고·판매량에서 파생해 악성재고·최소비용 분석까지 연결합니다.
- 잘못된 ID/행은 앱을 중단시키지 않고 해당 행을 제외하거나 fallback 처리합니다.

### 샘플 교체 시 상태 초기화

새 샘플을 불러오면 `selected_route_id`, 추천 필터, 경로 상세 선택, 시뮬레이션 속도/재생/전체 경로 보기(`home_speed_select`/`home_show_all` 포함), DQN 학습 결과·반영 방식, Kakao 지도 상태, `raw_sheet_select`가 초기화됩니다. 현재 메뉴(`current_menu`)와 실제 적용된 데이터(`varo_data`/`varo_recommendations`/`varo_validation`/파이프라인 결과)는 유지됩니다.

### DQN 학습과 비교 (샘플 기준)

- DQN은 **분석 및 검증 → DQN 학습** 탭에서 버튼을 눌렀을 때만 현재 샘플 기준으로 학습합니다. 홈/추천 실행/경로 상세 진입만으로는 학습·모델 로딩·추론을 실행하지 않습니다(`pages/overview.py`는 `dqn_service`를 import하지 않음).
- DQN 학습 탭은 현재 샘플명, 점포 수, DC 수, 재고 행 수, 추천 후보 수, PyTorch 사용 가능 여부, `data_signature`, 저장된 최신 DQN 결과의 현재 데이터 일치 여부를 표시합니다.
- 학습 결과는 `outputs/dqn` 아래에 저장되며 파일명에 sample_id·store_count·dc_count·episodes·learning_rate·timestamp가 포함됩니다. `latest` 결과에도 현재 `data_signature`를 저장합니다.
- DQN 결과는 **현재 `data_signature`와 일치하고 정상 상태일 때만** 낮은 참고 비중으로 VHS 비교에 반영됩니다. 불일치는 `과거 결과`로 처리해 VHS에 반영하지 않고, `학습 필요`/`검토 필요`/`학습 부족`/`실행 환경 필요`/`미연결`도 최종 점수에 반영하지 않습니다.
- **Greedy는 항상 비교 기준으로 포함**됩니다. 분석 및 검증 페이지에 VHS vs Greedy vs DQN 비교표가 표시됩니다. 홈 Top5에는 route_id·VHS·Greedy·DQN 등 내부 값을 노출하지 않습니다.
- Kakao 지도는 홈에서 로딩하지 않고 경로 상세에서만 key(`st.secrets` → 환경변수 순서)가 있을 때 표시 준비합니다.

### 남은 확인 사항

- 원본 파일은 이 개발 머신에서 `C:\Users\user\OneDrive\Desktop\Projects\Varo\Varo_DQN_training_samples_10pack\`(Windows Desktop이 OneDrive로 리디렉션됨)에 있으며, 다른 머신/사용자에서는 홈 디렉터리 기준으로 자동 탐색하거나 `VARO_DQN_SAMPLES_DIR`로 지정합니다.
- 실제 DQN 학습 실행/저장 파일명·모델 저장은 PyTorch가 설치된 환경에서 버튼 실행으로 최종 확인이 필요합니다(자동 실행은 하지 않음).

## 다운로드 (4종)

`데이터 관리`/`추천 실행`/`분석 및 검증`에서 내려받을 수 있습니다.

- 추천 결과 CSV (UTF-8 BOM, 한글 헤더)
- 추천 결과 Excel
- 분석결과 전체 Excel (멀티시트: 추천결과·VHS분석·**VHS비교**·**VHS중립값**·**민감도요약**·**추천사유**·Greedy분석·신뢰도·최적성검증·알고리즘상태·검증요약·DQN제외, 업로드 시 **업로드품질·컬럼매핑·날짜환산** 추가, V2 생성 후보 시 **생성후보·후보점수·후보생성요약** 추가)
- 검증 리포트 Excel (검증개요·검증메시지·시트요약·알고리즘상태·최적성검증·DQN제외, 업로드 시 업로드품질·컬럼매핑·날짜환산, V2 생성 후보 시 생성후보·후보점수·후보생성요약 추가)

모든 다운로드는 DQN 과거 결과값(reward/loss/model/q-table/policy)을 포함하지 않습니다. "DQN 안내"는 값이 아닌 정적 안내문입니다.

## 실행

```
cd C:\Projects\Varo\varo_v2
py -m compileall .
py -m unittest discover -s tests -v
py -m streamlit run app_v2.py
```

## 카카오 지도 연결

- 지도는 **경로 상세** 페이지에서만 로딩합니다. 홈에서는 Kakao SDK를 로딩하지 않습니다.
- JavaScript key는 코드에 저장하지 않습니다.
- key 우선순위:
  1. `.streamlit/secrets.toml`의 `KAKAO_JAVASCRIPT_KEY`
  2. 환경변수 `KAKAO_JAVASCRIPT_KEY`
- key가 없으면 지도 영역에는 `Kakao 지도 키가 설정되면 실제 지도 표시가 가능합니다.`만 짧게 표시됩니다.
- 좌표가 없거나 유효하지 않으면 `지도 좌표를 확인할 수 없습니다.` 또는 `DC 좌표 없음` 상태로 표시하고 앱은 중단되지 않습니다.
- DIRECT는 출발/도착 marker와 단일 polyline, VIA_DC는 출발/DC/도착 marker와 DC 경유 polyline을 사용합니다.
- 여러 DC가 있을 때는 추천 row의 `dc_id`를 먼저 사용하고, 없으면 `dc_name`으로 찾습니다. 임의로 DC01을 선택하지 않습니다.

## DQN 연결 구조

- DQN은 **분석 및 검증 > DQN 학습** 탭에서 사용자가 `DQN 학습 실행`을 눌렀을 때만 학습합니다.
- 홈, 추천 실행, 경로 상세 진입만으로는 DQN 학습/모델 로딩/추론을 실행하지 않습니다.
- 과거 DQN reward, loss, model, Q-table, policy table, replay buffer, training history는 V2 추천 점수에 자동 반영하지 않습니다.
- DQN 기본 반영 방식은 `DQN 참고만`입니다. 옵션은 `DQN 참고만`, `DQN 약하게 반영` 두 가지이며 기본값은 항상 참고만입니다.
- DQN이 `검토 필요`, `비활성`, `학습 필요` 상태이면 VHS, 절감액, 최종 추천 점수에 반영하지 않습니다.

DQN 상태:

- `학습 필요`: 현재 데이터로 학습 전 상태입니다.
- `연결`: V2 현재 데이터로 학습 및 추론이 가능하고 안정성 검사를 통과한 상태입니다.
- `검토 필요`: action 쏠림, 후보 수 부족 등으로 상세 참고만 가능한 상태입니다.
- `비활성`: PyTorch 실행 환경이 없거나 loss/reward 안정성 문제가 있는 상태입니다.

DQN action 매핑:

- `transfer`, `direct_transfer`, `dc_transfer`, `store_transfer`, `relocation`, `multi_store_transfer` -> 재고 이동
- `discount`, `discount_sale` -> 할인
- `urgent_discount`, `emergency_discount` -> 긴급 할인
- `one_plus_one`, `plus_one` -> 1+1
- `dispose`, `discard`, `waste` -> 폐기
- `keep_inventory`, `hold`, `no_action`, `maintain` -> 보류

DQN feature:

- 추천 후보 row의 `recommended_qty`, `distance_km`, `expected_time_min`, `move_cost`, `estimated_cost`, `expected_saving`, `vhs_score`, `confidence_score`, `direct_distance_km`, `via_dc_distance_km`, `promotion_effect`를 사용합니다.
- 누락되거나 숫자로 변환할 수 없는 값은 중립값으로 처리합니다.
- 민감도/추천 사유 상세는 원본 상세 모듈을 무리하게 직접 연결하지 않고, V2 입력 구조 기준 요약으로 대체했습니다.

브라우저에서 `기본 샘플 불러오기` → 운영 현황 KPI → 추천 실행 → R002 선택 → 경로 상세 → 분석 및 검증 순으로 확인할 수 있습니다.

## 테스트

`tests/` 디렉터리에서 `py -m unittest discover -s tests -v`로 실행합니다. 정확한 테스트 수와 결과는 실행 출력으로 확인합니다.
주요 묶음:

- `test_data_loading` / `test_recommendation_adapter`: 데이터 로드·검증·표준화
- `test_analysis_pipeline`: 자기완결 재계산, DQN 차단, status/result_basis
- `test_app_state`: 페이지 간 상태·선택 유지(R002)
- `test_algorithm_contracts`: 알고리즘 입력 컬럼 계약과 graceful degradation
- `test_vhs_neutral_components` / `test_v2_sensitivity_summary` / `test_recommendation_reasons`: V2 내부 요약 기능
- `test_column_aliases` / `test_upload_normalization` / `test_upload_validation` / `test_candidate_generation`: 실제 업로드 견고화(한/영 매핑·숫자 정규화·검증·후보 생성)
- `test_export_service`: 다운로드 바이트·시트·DQN 누출 방지
- `test_page_render`: 5개 페이지 헤드리스 렌더 + 사이드바 네비게이션·선택 경로 유지 + 홈 결과 대시보드(7컬럼 Top5, 금지 문구·버튼 없음)
- `test_network_simulation`: 운영 현황 네트워크 시뮬레이션
- `test_dynamic_network`: 점포 수·DC 수(1·2·4)별 동적 배치, DIRECT/VIA_DC 세그먼트
- `test_simulation_samples`: 3/4/6/8/10점포 샘플 로드·검증·배치, 10점포 2DC 홈 렌더(DC 2·차량 3·점선/실선), 전체 경로 보기 기본 꺼짐·상한

## 알고리즘 위치

승인된 비-DQN 알고리즘은 `services/legacy_adapters/_local_modules/`에 있으며, `services/legacy_adapters/loader.py`가 이 내부 폴더만 사용합니다(외부 원본/백업 경로는 사용하지 않고, `bad_inventory_simulator`를 가리키는 `VARO_LEGACY_PATH`는 무시됩니다). DQN 모듈은 항상 차단됩니다.

## 백업 ZIP 참고 확인 (2026-06-21)

- 백업 ZIP은 수정·압축 해제 없이 메모리에서 비-DQN Python 알고리즘 파일 17개만 읽기 전용으로 확인했습니다.
- 확인 범위: VHS, Greedy, 직접/DC 경로, 컷라인, 거래 시간, 최소비용 네트워크, 프로모션, 수요예측, 폐기위험, 안전재고, EOQ, ABC, 회전율, 점포·상품 매칭, 데이터 검증.
- 위 핵심 기능은 현재 V2 내부 adapter/pipeline과 화면에 대응되어 있어 중복 구현하지 않았습니다.
- 상세 민감도와 추천 사유 확장 분석은 추가 입력 기준이 필요해 보류하고, 현재는 V2 요약을 제공합니다.
- DQN 산출물과 UI 원본은 열람하지 않았으며, ZIP 내용을 V2에 복사하거나 덮어쓰지 않았습니다.

## 현재 성능/역할 분리 기준

- 홈은 KPI, 흐름, 추천 경로 이동 현황, 추천 Top 5만 빠르게 렌더링합니다.
- 홈에서는 Kakao SDK, DQN 모델, 원본 데이터표, 검증 리포트, 다운로드 묶음을 자동으로 로딩하지 않습니다.
- 데이터 적용 시 파일 내용 기반 `data_signature`를 저장하고, 홈 네트워크 레이아웃과 SVG 마크업은 데이터와 표시 상태 기준으로 캐시합니다.
- 데이터 교체 시 선택 경로, 추천 필터, 시뮬레이션 재생 상태, 전체 경로 보기, DQN/Kakao 상태를 새 데이터 기준으로 초기화합니다.
- Kakao 지도는 경로 상세 페이지에서만 로딩하며, key가 없으면 짧은 안내만 표시합니다.
- DQN은 분석 및 검증의 DQN 학습 탭에서 사용자가 명시적으로 실행할 때만 동작합니다. PyTorch가 없으면 앱 전체가 아니라 해당 탭에서만 실행 환경 안내를 표시합니다.
## VHS 자동 가중치와 비교 구조

- Varo V2는 후보별 `savings_score`, `disposal_risk_score`, `demand_fit_score`, `inventory_balance_score`, `route_cost_score`, `feasibility_score`, `promotion_score`, `greedy_score`, `confidence_score`, `dqn_reference_score`를 0~100 범위로 계산합니다.
- 자동 가중치는 현재 데이터의 사용 가능 컬럼, 결측률, 분산 신호를 반영하고 요소별 min/max 제한을 적용한 뒤 합계 1.0으로 정규화합니다.
- 절감액과 실현 가능성은 항상 핵심 축으로 유지하고, 결측이 많거나 모든 값이 같은 요소는 낮은 비중으로 제한합니다.
- 최종 `vhs_score`는 구성 점수와 자동 가중치의 가중합이며, `vhs_rank` 1위가 Varo 최종 추천입니다.
- Greedy 결과는 항상 비교 기준으로 포함합니다. action 문자열은 재고 이동, 할인, 긴급 할인, 1+1, 폐기, 보류의 공통 전략명으로 정리합니다.
- DQN은 기본적으로 비교/참고용입니다. 현재 V2 데이터 기준 학습과 안정성 검사가 완료된 경우에만 낮은 비중의 `dqn_reference_score`로 반영할 수 있습니다.
- DQN이 미연결, 학습 필요, 검토 필요, 비활성 상태이면 최종 VHS에는 반영하지 않고 비교표에는 상태만 표시합니다.
- 홈에는 VHS 가중치, Greedy/DQN 비교표, fallback 상세를 노출하지 않습니다. 상세는 분석 및 검증 페이지의 VHS 비중, Greedy와 DQN, 검증 리포트에서 확인합니다.

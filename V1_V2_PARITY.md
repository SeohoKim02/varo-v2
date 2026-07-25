# Varo V1 ↔ V2 기능 대응표 (Parity)

비교 대상
- **V1(완성형 Varo)**: `C:\Projects\Varo\varo_v1\bad_inventory_simulator\` (app.py 1,850줄 + dashboard_pages.py 8,770줄). 읽기 전용 분석만 수행했으며 원본은 수정하지 않았습니다.
- **V2**: `C:\Projects\Varo\varo_v2\` (5개 페이지 · router · services · simulation · components).

분류 기준: **완료**(V2 구현·동작) · **개선됨**(V2가 더 쉬움) · **보완함**(이번 작업에서 추가) · **통합**(V1 중복 기능을 한 곳으로) · **의도적 이동/제외**(사유 명시).

| # | 기능 | V1 위치·방식 | V2 위치·방식 | 분류 |
|---|------|------|------|------|
| 1 | 데이터 업로드 | 사이드바 file_uploader | 상단 데이터 바 `render_quick_data_bar` (모든 페이지) | 개선됨 |
| 2 | 기본 샘플 | 데모 모드 selectbox | 기본 샘플 + 시뮬레이션 샘플 5종 + DQN 샘플 10종 | 개선됨 |
| 3 | 데이터 검증 | sample_validator | `data_validator` · 데이터 관리/검증 결과 탭 | 완료 |
| 4 | 악성재고 판단 | disposal_risk_analyzer | `disposal_risk_analyzer` (adapter) · 홈 재고 상태 pill(과잉/부족/이동 대상) | 완료 |
| 5 | Varo Hybrid Score | varo_hybrid_score | `services/vhs_score_engine.py` · 추천 점수 탭 | 완료 |
| 6 | 자동 가중치 | varo_weight_optimizer | `apply_auto_vhs` · 점수 구성 탭(그룹 요약) | 개선됨 |
| 7 | 추천 생성 | recommender / decision_analyzer | `candidate_generator` (rule-based, 시트 없으면 자동 생성) | 완료 |
| 8 | 추천 후보 필터 | 상품/등급 필터 | 추천 실행 필터 6종(상품·출발·도착·경로·등급·이동수단) | 완료 |
| 9 | 최종 추천 | 최종 추천 탭 | VHS 1위 = Varo 최종, 추천 실행 1순위 카드 | 개선됨 |
| 10 | Greedy 비교 | 추천 방식 비교 탭 | 비교 분석 탭(Varo/Greedy/DQN/Pareto 8열) | 완료 |
| 11 | DQN 학습 | torch_dqn_agent / train_rl_agent | `services/dqn_service.py` 실제 PyTorch 학습(버튼 실행) | 완료 |
| 12 | DQN 결과 저장·불러오기 | rl_data_logger / github_dqn_uploader | `save_dqn_result` / `load_latest_dqn_result` · 저장 결과 불러오기 버튼 | 완료 |
| 13 | DQN 안정성 | dqn_stability | `evaluate_dqn_stability` · 학습 결과 안정성 카드 | 완료 |
| 14 | DQN action 비교 | dqn_recommender | 행동 분포 표 · 원본 vs 균형형 비교 | 개선됨 |
| 15 | Pareto 분석 | optimization_analyzer | `services/pareto_analysis.py` · 비교 분석 탭 | 완료 |
| 16 | 민감도 분석 | varo_sensitivity | `weight_sensitivity` · 민감도 탭(3 KPI + 막대) | 완료 |
| 17 | 추천 신뢰도 | vhs_confidence | `confidence` · 검증 결과 탭 | 완료 |
| 18 | 최적성 검증 | varo_optimality_gap | `optimality_gap` → "최적해 차이" · 검증 결과 탭 | 개선됨 |
| 19 | 비용 비교 | min_cost_network / calculator | 경로 상세 핵심 수치 + 이동 방식 비교 | 완료 |
| 20 | 직접 이동 ↔ DC 경유 비교 | route_analyzer | 경로 상세 "이동 방식 비교"(판단 열·추천 행 강조) | 개선됨 |
| 21 | 지도 | kakao_map_viewer | 경로 상세 지도 영역 | 완료 |
| 22 | Kakao 지도 | 사이드바 JS key 입력 | `kakao_service` (st.secrets/환경변수, 경로 상세에서만 로딩) | 개선됨 |
| 23 | 이동 경로 | transfer_path_analyzer | 경로 상세 "이동 단계"(직접/DC 경유 카드) | 완료 |
| 24 | 시뮬레이션 | network_path_analyzer | 홈 DC 중심 방사형 네트워크 + SMIL 차량 애니메이션 | 개선됨 |
| 25 | 운영 KPI | varo_dashboard_kpi | 홈 5개 KPI(분석 재고·추천 경로·절감액·평균 점수·판단) | 완료 |
| 26 | 추천 Top5 | dashboard TOP5 | 홈 추천 Top5(7열) | 완료 |
| 27 | 현재 실행 경로 | candidate rail | 홈 "현재 이동 중" Top3 | 완료 |
| 28 | 검증 리포트 | 검증 요약 탭 | 검증 결과 탭 + "검증 결과 다운로드" | 완료 |
| 29 | Excel 다운로드 | 다수 download_button | 추천 CSV/Excel · 분석결과 Excel · 검증리포트 | 완료 |
| 30 | 분석 결과 다운로드 | filtered_result Excel | "분석 결과 전체 Excel" | 완료 |
| 31 | 학습 결과 다운로드 | rl_data_logger 로그 | **DQN 학습 탭 "학습 결과 다운로드"(JSON)** | 보완함 |
| 32 | 원본 데이터 확인 | 사이드바 미리보기 | 데이터 관리 "원본 데이터 보기" expander | 완료 |
| 33 | 페이지 이동 | icon nav / _go | 사이드바 5페이지 내비 + 현재 페이지 강조 | 개선됨 |
| 34 | 상태 초기화 | 세션 리셋 | `apply_state_payload` (새 데이터 적용 시에만) | 완료 |
| 35 | 새 데이터 적용 | 업로드 시 재계산 | `load_and_apply` (검증 통과 시 전체 페이지 반영) | 완료 |
| 36 | 오류·경고 표시 | show_friendly_error | 검증 메시지 · 조건부 warning(필수 컬럼/최소 정보 부족 등) | 완료 |

## 의도적 이동·통합 (사유)
- **개별 입력값 수동 계산기(V1 사이드바)**: 단일 점포·상품을 손으로 입력해 계산하던 방식은 V2의 데이터 파일 기반 일괄 분석으로 **통합**했습니다. 동일한 계산(폐기위험·이동비용·VHS)을 파일의 모든 후보에 적용하므로 기능 손실이 아니라 상위 집합입니다.
- **알고리즘 설명 탭(V1)**: 긴 알고리즘 설명은 화면 대신 `README_V2.md`와 각 탭의 "계산 기준 자세히 보기" expander로 **이동**했습니다(기본 화면 가독성 우선).
- **데모 모드**: V2의 기본 샘플 + 시뮬레이션 샘플 5종 + DQN 샘플 10종으로 **대체**했습니다.
- **GitHub DQN 업로더**: 외부 업로드는 배포 정책상 제외하고, 로컬 "학습 결과 다운로드"로 대체했습니다.

## V1에 없던 V2 신규 기능
- DC 중심 결정론적 방사형 시뮬레이션(노드 화면 밖 이탈 방지, 트럭 SVG, SMIL 애니메이션).
- 원본 vs 균형형 DQN 학습 데이터 진단·비교(라벨 쏠림 정량 진단).
- 통일 상태 배지(데이터/추천 계산/DQN/지도) 상단 1곳 표시.
- 기본 표 10행·8열 캡 + "전체 결과 보기" expander.
- data_signature 게이팅으로 현재 데이터와 일치할 때만 DQN 낮은 비중 반영.

## 이번 작업에서 보완한 항목
- **DQN 학습 결과 다운로드**(#31): V1의 학습 로그 다운로드에 대응하는 JSON 다운로드를 DQN 학습 탭에 추가.
- **추천 전략 표기 명확화**: 1순위 추천 카드에서 "이동 경로 방식 / Greedy 전략 / 최종 처리 전략 / 추천 등급"을 분리해, 경로 방식(직접 이동)과 최종 처리(보류)가 충돌해 보이지 않도록 정리(보류 시 사유 한 줄 표시).

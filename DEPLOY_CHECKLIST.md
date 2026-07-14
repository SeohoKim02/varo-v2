# Varo V2 배포·브라우저 확인 체크리스트

## 1. 실행

기본 수동 확인:

```powershell
cd C:\Users\user\OneDrive\Desktop\Projects\Varo\varo_v2
py -m streamlit run app_v2.py
```

- 접속 주소: `http://localhost:8501`
- 지정 포트 검증: `py -m streamlit run app_v2.py --server.headless true --server.port 8533`
- 지정 포트 주소: `http://localhost:8533`
- 상태 확인: `http://localhost:8533/_stcore/health`

## 2. GitHub·Streamlit Cloud 준비

- [ ] GitHub에 올리기 전에 `.gitignore`가 적용되는지 확인한다.
- [ ] 로컬 `.streamlit/secrets.toml`은 업로드하지 않는다.
- [ ] 기본 배포는 `requirements.txt`를 사용한다.
- [ ] DQN 학습 포함 배포는 `requirements-dqn.txt`를 참고하되 Streamlit Cloud의 용량과 시작 시간을 확인한다.
- [ ] 로컬 학습 산출물인 `outputs/dqn/`, `outputs/dqn_balanced_samples/`와 `outputs/qa/`는 배포에서 제외한다.
- [ ] 배포 후 공개 주소에서 홈, 추천 실행, 경로 상세, 분석 및 검증, 데이터 관리 페이지를 확인한다.

## 3. 홈 화면 확인

- [ ] 현재 데이터명, 점포/DC 수, 추천 후보 수, 예상 절감액, 평균 VHS가 표시된다.
- [ ] 빠른 이동 카드 4개와 범례가 포함된 상위 1~3개 네트워크 미리보기가 표시된다.
- [ ] 홈에 추천 Top5 표나 선택한 추천 요약이 중복 표시되지 않는다.
- [ ] 시뮬레이션 속도와 전체 경로 보기 ON/OFF가 동작한다.
- [ ] 모든 페이지가 밝은 배경이며 검은 카드·표·expander·버튼·사이드바가 없다.
- [ ] 화면에 traceback, 내부 함수명, 디버그 정보가 보이지 않는다.
- [ ] 브라우저 개발자도구 Console에 빨간 error가 없다.

## 4. 데이터와 샘플 확인

- [ ] 데이터 관리에서 DQN 샘플 01을 불러오면 2점포·1DC가 표시된다.
- [ ] 샘플 01의 추천 후보, Top5 규칙, 시뮬레이션이 정상이다.
- [ ] DQN 샘플 10을 불러오면 10점포·2DC가 표시된다.
- [ ] 샘플 10에서 DC01/DC02, 점포, 차량, 경로가 눈에 띄게 겹치지 않는다.
- [ ] 데이터 교체 후 이전 선택 경로와 일시 상태가 새 데이터 기준으로 초기화된다.

## 5. 페이지별 확인

- [ ] 홈, 추천 실행, 경로 상세, 분석 및 검증, 데이터 관리 5개 페이지가 모두 열린다.
- [ ] 페이지 이동 후 현재 데이터와 선택 경로가 유지된다.
- [ ] 추천 실행 표에 순위·상품·출발·도착·경로·수량·절감액·등급이 표시된다.
- [ ] 추천 후보 선택 UI와 선택한 추천 요약이 같은 추천을 가리킨다.
- [ ] 추천 실행의 상세 비교 영역은 접힌 상태이며 VHS/Greedy/DQN/Pareto 비교를 확인할 수 있다.
- [ ] 경로 상세에서 DIRECT 이동 단계가 자연스러운 문장으로 표시된다.
- [ ] VIA_DC 경로에서 선택된 DC와 이동 단계가 일치한다.
- [ ] 분석 및 검증에서 VHS 분석, Greedy 비교, DQN 학습·비교, Pareto 검증, 민감도/신뢰도 탭이 열린다.
- [ ] DQN 학습·샘플 10개 학습·원본 vs 균형형 비교 버튼이 보인다.
- [ ] 페이지 진입만으로 DQN 학습이 자동 실행되지 않는다.
- [ ] 데이터 관리에서 접힌 `DQN 샘플 10개 목록`을 열 수 있다.

## 6. 알고리즘 비교 확인

- [ ] VHS/Greedy/DQN/Pareto 비교표가 열리고 각 순위와 DQN 참고 여부가 구분된다.
- [ ] DQN 원본 10개 학습 결과에서 라벨 편향 진단과 검토 상태를 확인한다.
- [ ] DQN 균형형 10개 학습 결과에서 라벨 분포와 학습 안정성 결과를 확인한다.
- [ ] 정상·signature 일치 DQN만 최대 8% 참고되며, 그 외 결과는 최종 추천에 참고 제외된다.
- [ ] DQN 결과는 버튼을 누르기 전 자동으로 생성되지 않는다.

## 7. 경로 상세 확인

- [ ] DIRECT 경로는 출발 점포에서 도착 점포로 직접 이동하는 단계로 표시된다.
- [ ] VIA_DC 경로는 출발 점포, 선택된 DC, 도착 점포 순서로 표시된다.
- [ ] 10점포·2DC 샘플에서 DC01/DC02 식별자와 선택된 경유 DC가 일치한다.

## 8. 오류와 원본 보호

- [ ] 터미널에 warning, traceback, import error가 없다.
- [ ] 브라우저 개발자도구 Console에 빨간 error가 없다.
- [ ] 검은 배경이나 검은 위젯이 없다.
- [ ] DQN 원본 샘플 10개의 크기와 수정 시간이 바뀌지 않았다.
- [ ] 기존 Varo `app.py`, `dashboard_pages.py`, 백업 ZIP이 바뀌지 않았다.
- [ ] DQN 결과는 `outputs/dqn/`, 균형형 파생 샘플은 `outputs/dqn_balanced_samples/` 아래에만 생성된다.
- [ ] 개인 secret 값이 코드·requirements·README에 기록되지 않았다.

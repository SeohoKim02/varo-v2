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

VHS가 최종 추천 우선순위를 구성한다. Greedy는 단순 기준 비교선, Pareto는 여러 기준의 후보 우열을 확인하는 보조 검증으로 사용한다. DQN은 자동 실행하지 않으며, 사용자가 학습을 실행하고 현재 데이터와 결과 식별값이 일치하며 품질 조건을 통과한 경우에만 최대 8% 범위의 참고값으로 연결된다.

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

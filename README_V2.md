# Varo V2

## 프로젝트 개요

Varo V2는 기존 졸업작품 Varo를 바탕으로 악성재고 판단과 점포 간 재고 이동 추천을 더 쉽게 확인할 수 있도록 정리한 SCC 제출용 Streamlit 데모다. 심사자와 운영자가 업로드부터 추천 비교, 경로 확인, 검증까지 한 흐름으로 살펴볼 수 있게 구성했다.

최종 추천은 Varo Hybrid Score(VHS)를 중심으로 정렬한다. Greedy는 단순 기준 비교, Pareto는 후보 간 보조 검증, DQN은 현재 데이터와 일치하고 품질 조건을 통과한 결과만 최대 8% 범위에서 참고한다.

## 기존 Varo와 Varo V2의 차이

- 기존 Varo는 악성재고 판단과 점포 간 재고 이동 추천을 수행한다.
- Varo V2는 밝은 화면, 간결한 의사결정 표, 네트워크 시뮬레이션과 알고리즘 비교 검증을 강화했다.
- V2는 기존 `app.py`, `dashboard_pages.py`, `varo_v1`, 백업 ZIP, DQN 원본 엑셀을 수정하지 않는다.

## 교수님 피드백 반영 내용

교수님 피드백에 따라 UI 설명을 줄이고 결과 중심 흐름으로 정리했다. 전 화면을 밝은 테마로 통일하고, 네트워크 배치와 Top 경로 표현을 다듬었다. VHS 자동 가중치, Greedy 비교, DQN 원본·균형형 비교, Pareto 보조 검증, 민감도·신뢰도와 제한 탐색 기반 Gap을 한 화면에서 확인할 수 있게 연결했다.

## 주요 기능

- 엑셀 업로드: 필수 시트를 확인하고 직접 업로드, 기본 샘플, DQN 샘플을 같은 적용 흐름으로 읽는다.
- 악성재고 판단: 재고·수요·폐기 위험 등 현재 입력을 기준으로 이동 검토 대상을 만든다.
- Varo Hybrid Score: 사용 가능한 구성요소와 데이터 분포를 반영해 최종 추천 우선순위를 계산한다.
- Greedy 비교: 단순 정렬 전략과 VHS 결과의 차이를 비교한다.
- DQN 원본·균형형 학습 비교: 라벨 편향과 학습 안정성을 비교하되 자동 실행하지 않는다.
- Pareto 보조 검증: 절감액·비용 등 여러 기준에서 후보 간 우열을 보조 확인한다.
- 네트워크 시뮬레이션: DC, 점포, Top 경로와 차량 이동을 밝은 화면에 표시한다.
- 경로 상세: 선택한 경로의 DIRECT/VIA_DC 방식과 이동 단계를 표시하고, 경유 경로는 DC01/DC02를 구분한다.

## 엑셀 업로드 규칙

필수 시트는 `stores`, `products`, `inventory`, `routes`다. `recommendations` 또는 `v2_recommendations`는 선택 시트이며, 없으면 현재 입력으로 후보를 생성한다.

- 직접 업로드, 기본 샘플, DQN 선택 샘플은 모두 파일을 bytes로 고정한 뒤 같은 `load_and_apply` 흐름을 사용한다. 따라서 업로드 스트림 포인터 위치와 무관하게 동일한 내부 구조와 세션 초기화 규칙이 적용된다.
- 별도 `dcs` 시트는 로더 안에서 `stores`에 `node_type=DC`로 병합한다.
- DQN 샘플의 `store_type`은 상권 유형으로 보존하고, 네트워크 노드 구분은 `node_type` 또는 별도 `dcs` 시트에서만 안전하게 결정한다.
- `from_store_id`, `from_store_name`, `to_store_id`, `to_store_name`, `source`, `target` 계열 별칭을 표준 컬럼으로 변환한다.
- 비어 있거나 중복된 추천 ID는 메모리 안에서 고유한 `route_id`로 다시 만든다.
- 같은 출발·도착 조합의 중복은 경고로 처리하고, 실제 필수 데이터 누락만 적용을 막는다.
- 원본 엑셀은 읽기 전용이며 정규화 결과를 원본 파일에 쓰지 않는다.

## 샘플 위치

- 기본 샘플: `data/Varo_V2_네트워크_샘플.xlsx`
- 시뮬레이션 검수 샘플: `samples/Varo_V2_sample_*.xlsx`
- DQN 원본 탐색 순서: `../Varo_DQN_training_samples_10pack`, `dqn_samples`, `data/dqn_samples`, `Varo_DQN_training_samples_10pack`
- DQN 원본 폴더가 없거나 10개가 완전하지 않으면 V2 내부 검수 샘플 5개를 원본/균형형으로 연결해 01~10 선택 흐름을 유지한다.

DQN 원본 폴더와 V2 내부 샘플은 수정하지 않는다. 10번 fallback 샘플은 10점포·2DC 구성이다.

## 네트워크 시뮬레이션

DC 1개는 중앙 우측, DC 2개는 중앙부 좌우에 배치한다. 점포는 좌표 분포를 우선하되 간격을 보정하고, 좌표가 부족하면 중앙을 비운 완만한 사각 둘레에 배치한다. Top5 경로만 기본 표시하며 직접 이동은 실선, DC 경유는 점선이다. 차량은 SVG 트럭으로 표시하고 경로별 lane, 곡선, 시작 위상을 달리한다. 차량은 노드보다 아래 레이어에 그려 점포명과 DC명을 가리지 않는다.

속도는 느림 24초, 보통 14초, 빠름 8초의 한 경로 이동 시간으로 구분한다. 기본값은 보통이다. 애니메이션은 브라우저 SMIL을 사용하므로 Python 반복 루프나 연속 rerun을 만들지 않는다.

## DQN 처리 방식

DQN은 자동 실행하지 않고 사용자가 버튼을 누른 경우에만 실행한다.

- `DQN 학습 실행`: 원본/균형형과 20~500 에피소드를 선택한 현재 데이터 단건 학습
- `DQN 샘플 10개 일괄 검증`: 샘플 01~10 진단·생성·순차 학습
- `DQN 원본 vs 균형형 비교`: 같은 데이터의 두 label 분포 비교

원본 샘플의 라벨 편향을 먼저 진단한 뒤, 수량·비용·거리·절감액 등 원본 수치 특성을 바꾸지 않고 파생본의 target action label만 재분배한 균형형 샘플과 비교한다. 이 비교는 데이터 품질과 학습 안정성을 확인하기 위한 것으로 최종 추천 성능을 보장하는 장치가 아니다.

- 모델·결과: `outputs/dqn/`
- 균형형 파생 엑셀: `outputs/dqn_balanced_samples/`
- timestamp 결과와 `latest_dqn_result.json`, `latest_dqn_model.pt`를 함께 유지한다.
- timestamp 파일명에는 sample ID, original/balanced, 점포 수, DC 수, episodes, learning rate가 포함된다.

DQN 상태가 `정상`이고 저장된 `data_signature`가 현재 데이터와 일치할 때만 최대 8%의 낮은 비중 참고값을 사용할 수 있다. 편향이 크거나 검토가 필요한 결과, 미학습, 학습 부족, signature 불일치는 최종 추천에 넣지 않고 비교표에만 표시한다.

Streamlit Cloud의 루트 `requirements.txt`에 CPU 학습이 가능한 최소 PyTorch 패키지만 포함한다. torchvision·torchaudio·CUDA 전용 패키지는 사용하지 않으며, PyTorch import가 실패해도 VHS·Greedy·Pareto·업로드·시뮬레이션은 계속 실행되고 DQN 학습 버튼만 제한된다.

## 경로 상세 처리 방식

- DIRECT는 출발 점포에서 도착 점포로 직접 이동하는 한 단계로 표시한다.
- VIA_DC는 출발 점포에서 선택된 DC를 거쳐 도착 점포로 이동하는 세 단계로 표시한다.
- 2DC 데이터는 DC 이름과 DC01/DC02 식별자를 함께 보여 선택된 물류센터를 구분한다.

## 배포 방법

Streamlit Community Cloud는 루트 `requirements.txt`를 설치하므로 이 파일에 `torch>=2.2,<3`을 포함한다. `requirements-dqn.txt`는 기존 설치 명령과의 호환을 위해 루트 requirements를 그대로 참조한다. Cloud의 현재 기본값은 Python 3.12이며, 이 프로젝트의 로컬 검증 버전은 Python 3.11과 PyTorch 2.12.1이다. 배포 시 고급 설정에서 Python 3.11을 선택해 검증 환경과 맞춘다.

Cloud에서는 기본 CPU 학습을 사용하며 로컬 CUDA 사용 가능 환경은 기존 GPU 감지를 유지한다. Cloud 로컬 파일은 영구 저장소로 간주하지 않고, 학습 직후 결과를 session state에서 우선 표시한다.

앱 진입점은 `app_v2.py`이며 `.streamlit/config.toml`에 밝은 테마가 고정돼 있다.

Streamlit Community Cloud에서는 `app_v2.py`를 진입점으로 지정하고 루트 `requirements.txt`를 사용한다.

## 실행 방법

```powershell
cd C:\Users\user\OneDrive\Desktop\Projects\Varo\varo_v2
py -m compileall .
py -W default -m unittest discover -s tests
py -m streamlit run app_v2.py --server.headless true --server.port 8528
```

검증 주소는 `http://localhost:8528`이다. 일반 실행 `py -m streamlit run app_v2.py`의 기본 주소는 `http://localhost:8501`이다.

## 수동 검증 체크리스트

제출 전 5개 페이지, 샘플 01·10, DQN 비교 결과, DIRECT/VIA_DC 경로 단계와 브라우저 Console은 [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md)에서 확인한다.

## 남은 확인 사항

- 실제 제출 브라우저에서 5개 페이지의 육안 배치와 개발자도구 Console을 최종 확인한다.
- 경로 상세에서 DIRECT/VIA_DC 이동 단계와 2DC 데이터의 DC01/DC02 구분을 확인한다.

## 원본 보호 원칙

- `varo_v1`, 기존 Varo 원본, 백업 ZIP, DQN 원본 엑셀을 수정하거나 압축 해제하지 않는다.
- Git 설정과 pip 환경을 자동으로 변경하지 않는다.
- API key를 코드에 하드코딩하지 않는다.
- DQN 학습은 현재 데이터와 signature가 맞는지 확인한 뒤 해석한다.
- 브라우저 연결 종료로 발생하는 일시적인 Windows connection reset과 실제 Streamlit 서버 종료를 구분한다.

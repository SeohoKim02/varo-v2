# Known Issues

## 기존 `transfer_path_analyzer.py` 경로 조회 위험

- 문제 파일: 기존 Varo 원본 `transfer_path_analyzer.py`
- 문제 함수: `_build_route_lookup()`
- 예상 원인: `_distance_km`, `_transport_cost`, `_time_min`처럼 밑줄로 시작하는 임시 컬럼을 만든 뒤 `pandas.DataFrame.itertuples()`로 읽으면 pandas가 컬럼명을 자동 변경할 수 있다. 이 경우 의도한 속성명으로 값이 전달되지 않을 가능성이 있다.
- 영향: 다음 알고리즘 연결 단계에서 경로 거리, 비용, 시간 조회가 누락되거나 잘못 매핑될 수 있다.
- 필요 조치: 원본을 직접 수정하지 말고 wrapper 또는 연결 서비스에서 `iterrows()`, `to_dict("records")`, 밑줄 없는 명시적 컬럼명, 또는 `itertuples(name=None)` 위치 기반 접근을 검토한다.
- 현재 상태: `analysis_pipeline._run_routes`가 `transfer_module._build_route_lookup`을 `services/legacy_adapters/data_adapter.safe_transfer_route_lookup`으로 교체해 실행한다. 이식된 모듈은 그대로 두고 연결 서비스에서만 우회한다.

## openpyxl 서식 경고 (보류)

- 증상: 샘플 엑셀 로드 시 openpyxl이 `Unknown extension is not supported and will be removed`, `Conditional Formatting extension is not supported and will be removed` UserWarning을 출력한다.
- 원인: 샘플 워크북에 포함된 조건부 서식/확장 요소를 openpyxl 읽기 단계가 완전히 지원하지 않기 때문이다.
- 영향: 기능 오류 아님. 9개 시트(stores, products, inventory, routes, recommendations, transport_modes, config, quality_check, readme)가 모두 정상 로드된다.
- 조치: 원칙에 따라 보류한다. 서식 경고는 데이터 값과 무관하며 추천·검증·다운로드 결과에 영향을 주지 않는다.

## 자기완결(self-contained) 동작

- 현황: 승인된 비-DQN 알고리즘 21개가 `services/legacy_adapters/_local_modules/`에 이식되어, `loader.py`가 이 내부 폴더만 사용한다. 외부 원본 폴더(`bad_inventory_simulator`)와 백업 ZIP은 런타임에 접근하지 않으며, `VARO_LEGACY_PATH`가 `bad_inventory_simulator`를 가리키면 무시한다. DQN 모듈·아티팩트는 loader와 `dqn_guard`에서 항상 차단한다.
- 전체 테스트가 외부 접근 없이 통과한다.

## 보류(deferred) 알고리즘과 V2 대체 기능 — DQN '제외'와 다름

- 원본 `varo_sensitivity`, `vhs_reason`은 추가 입력 그룹(이력 보정 그룹) 정의가 필요해 **보류**다. 핵심 VHS 재계산·추천 판단에는 영향이 없다.
- 대신 V2 내부 결과 기준의 대체 요약을 **연결**해 제공한다:
  - **V2 민감도 요약**(`services/v2_summaries.sensitivity_summary`) — 비용·거리·수량·VHS·절감액 기준 순위 변동 위험(낮음/보통/높음/제한적).
  - **V2 추천 사유 요약**(`services/v2_summaries.recommendation_reasons`) — rule-based 추천 사유.
  - **V2 VHS 중립값 요약**(`services/v2_summaries.vhs_neutral_summary`) — 중립값 적용 구성요소 현황.
- 상태 구분: DQN은 '제외'(과거 학습 이상치 가능성), 원본 두 모듈은 '보류'(입력 조건 미비), V2 요약 3종은 '연결'. DQN 값은 어떤 요약에도 점수로 반영하지 않는다("DQN 안내"는 정적 안내문).

## `varo_score.py` 일부 구성요소 제외

- `varo_score.py`는 내부에 `rl_training` 참조가 있어 DQN 연계 가능성 때문에 `_local_modules`로 이식하지 않았다.
- 영향: `varo_hybrid_score`는 이 모듈에 하드 의존하지 않으므로 VHS 재계산은 정상 동작한다. 다만 varo_score가 제공하던 일부 VHS 구성요소는 입력이 없을 때 원본 규칙대로 중립값(50)이 적용될 수 있다(VHS 분석 탭의 "중립값 적용" 항목에 표기).

## 임시 파일 잔존 (삭제 금지 원칙)

- `_scratch_export_check.py`(inert stub), `_streamlit_boot.log`는 개발 중 생성된 임시 산출물이다. 삭제 명령 금지 원칙으로 제거하지 않았으며, 기능에 영향이 없고 `unittest discover`에 수집되지 않는다.

## 실제 엑셀 업로드 동작

- 컬럼명은 한글/영문 alias로 V2 표준 컬럼에 자동 매핑된다(`services/column_aliases.py`). 매핑·숫자 변환·빈 행 제거 현황은 데이터 관리 "업로드 품질 점검"과 검증 리포트/다운로드의 `업로드품질`·`컬럼매핑` 시트에서 확인한다.
- 일부 **선택 컬럼이 부족**하면 분석이 제한될 수 있고, 해당 VHS 구성요소에는 중립값이 적용된다(앱은 중단되지 않음).
- **추천 결과 시트가 없으면** 재고·경로 기반 **V2 생성 후보**로 대체한다(`services/candidate_generator.py`). DIRECT/VIA_DC를 모두 고려하고 candidate_score(0~100)로 정렬한다. 후보 생성이 불가능하면 추천 생성을 보류한다.
- **필수 컬럼/시트가 부족**하면 추천 계산은 보류되고 검증 메시지로 안내한다. 손상된 파일도 traceback 없이 오류 메시지로 처리하며 이전 정상 데이터는 유지된다.

## V2 생성 후보·날짜 환산 한계

- V2 생성 후보는 운영용 **rule-based** 후보이며 MILP/DQN 최적화가 아니다. candidate_score는 VHS와 별개의 정렬 보조값이며 DQN을 반영하지 않는다.
- **DC 경유(VIA_DC)** 후보는 입력 `routes`에 `출발→DC`와 `DC→도착` 경로가 모두 있을 때만 생성된다. 경로가 없으면 보류로 기록한다.
- **날짜 자동 환산**은 해석 가능한 날짜 형식(`YYYY-MM-DD`, `YYYY.MM.DD`, `YYYY/MM/DD`, Excel serial)에 한정한다. `유통기한`(일수)과 `만료일/소비기한`(날짜)을 구분하며, 기존 `days_to_expiry`가 있으면 우선한다. 해석 불가 값은 보류로 처리한다.

## 미연결 상태 (의도된 현재 상태)

- DQN: 미연결('제외'). `dqn_action`="미연결", `dqn_correction`=0. 결과·다운로드에 과거 DQN 값이 포함되지 않는다.
- 카카오 지도: 미연결. SDK/API key 미작성. 이후 연결 예정.
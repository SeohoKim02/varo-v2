# 로컬 알고리즘 드롭인 디렉터리 (선택)

이 디렉터리는 Varo V2가 **외부 원본 폴더나 백업 ZIP 없이 독립 실행**되도록 하는
자기완결(self-contained) 기본 경로입니다.

- 기본 상태(이 디렉터리가 비어 있음): 승인된 비-DQN 알고리즘 모듈이 없으므로
  `analysis_pipeline`은 모든 재계산 알고리즘을 **보류(deferred)** 처리하고,
  업로드된 사전 계산 추천 값을 그대로 사용해 `partial` 상태로 동작합니다.
  앱은 죽지 않으며 화면에는 "일부 알고리즘 보류 / 현재 V2 내부 정보 기준"으로 표시됩니다.

- 전체 재계산을 켜려면(선택): `services/legacy_adapters/loader.py`의
  `LEGACY_ALGORITHM_ALLOWLIST`에 정의된 **비-DQN** 알고리즘 `.py` 파일을 이 폴더에
  복사해 넣거나, 환경변수 `VARO_LEGACY_PATH`로 허용된 위치를 지정하면 됩니다.

## 금지 사항

- DQN 학습 산출물(reward/loss/model/q-table/policy/replay buffer/training history,
  `*.pt/.pth/.pkl/.joblib`, `rl_*`, `dqn_artifacts`)은 이 폴더에 두지 않습니다.
  `dqn_guard`와 `loader`가 DQN 모듈·아티팩트를 항상 차단합니다.
- `VARO_LEGACY_PATH`가 `bad_inventory_simulator`(원본/백업) 경로를 가리키면
  loader가 무시하고 이 내부 폴더로 폴백합니다.

# 실행 이력 저장소 운영 배포 가이드

이 문서는 **실행 이력(execution history) 저장소**를 상용 환경에 연결할 때 필요한 절차만 다룹니다.
추천 알고리즘(VHS·Greedy·Pareto·DQN·실행계획 최적화)은 이 문서의 어떤 설정에도 영향을 받지 않습니다.
데이터베이스가 없거나 죽어 있어도 추천 기능은 그대로 동작하며, 이력 저장 기능만 비활성 상태가 됩니다.

> 이 문서에는 실제 URL·호스트·계정·비밀번호를 절대 기록하지 않습니다. 모든 예시는 자리표시자입니다.

## 1. 저장 backend 선택

| 상황 | 설정 | 저장 위치 |
| --- | --- | --- |
| 로컬 개발 (기본) | 설정 없음 | `runtime_data/varo_execution_history.sqlite3` |
| 로컬 경로 지정 | `VARO_HISTORY_DB_PATH` | 지정한 SQLite 파일 |
| 상용 운영 | `VARO_HISTORY_DATABASE_URL` | PostgreSQL 서버 |

우선순위는 **테스트/이관에서 넘긴 명시적 SQLite 경로 → `VARO_HISTORY_DATABASE_URL` →
`VARO_HISTORY_DB_PATH` → 기본 SQLite 경로**입니다.

`VARO_HISTORY_DATABASE_URL`을 설정했는데 형식이 PostgreSQL이 아니거나 host/database가 빠져 있으면
**설정 오류로 즉시 실패**합니다. SQLite로 몰래 우회하지 않습니다. 연결·조회·쓰기가 실패하는 경우에도
로컬 SQLite 파일을 만들지 않습니다.

## 2. 환경변수

| 변수 | 필수 | 기본값 | 용도 |
| --- | --- | --- | --- |
| `VARO_HISTORY_DATABASE_URL` | 운영에서 필요 | 없음 | 배포용 PostgreSQL 연결 URL |
| `VARO_HISTORY_DB_PATH` | 아니오 | 기본 경로 | 로컬 SQLite 파일 경로 |
| `VARO_HISTORY_TEST_DATABASE_URL` | 아니오 | 없음 | **staging 전용** 검증 대상. 운영 URL과 같으면 거부 |
| `VARO_HISTORY_RESTORE_DATABASE_URL` | 아니오 | 없음 | **복원 전용** 대상. 운영 URL과 같으면 거부 |
| `VARO_HISTORY_DB_CONNECT_TIMEOUT` | 아니오 | `8`(초) | 연결 타임아웃. 2~60초로 제한 |
| `VARO_HISTORY_DB_CONNECT_RETRIES` | 아니오 | `1` | **연결 수립만** 재시도하는 횟수. 0~3 |

아무것도 설정하지 않아도 합리적인 기본값으로 동작합니다. 잘못된 숫자를 넣으면 앱을 멈추지 않고
기본값으로 되돌립니다.

### Streamlit Community Cloud

앱 설정의 **Secrets**에 최상위 키로 등록합니다. 최상위 secret은 환경변수로 전달되므로 저장 service가
Streamlit에 의존하지 않습니다.

```toml
# 예시 형식일 뿐이며 실제 값이 아닙니다.
VARO_HISTORY_DATABASE_URL = "postgresql://<user>:<password>@<host>:5432/<database>?sslmode=verify-full"
```

`.streamlit/secrets.toml`은 `.gitignore`에 있으며 저장소에 커밋하지 않습니다. 실제 값은 배포 환경의
비밀 설정에만 둡니다.

## 3. TLS

TLS 옵션은 URL에 그대로 실어 보내며 드라이버(libpq)에 손대지 않고 전달합니다. 코드가 인증서 검증을
끄거나 `sslmode`를 임의로 바꾸는 곳은 없습니다.

- 관리형 PostgreSQL 사용 시 **제공업체가 권장하는 `sslmode`를 그대로** URL에 넣으세요.
- 인증서 검증까지 하려면 `?sslmode=verify-full` (필요하면 `&sslrootcert=<경로>`)를 사용합니다.
- `sslmode=disable`은 운영 기본값으로 쓰지 마세요. 점검 도구가 경고로 표시합니다.
- 인증서/개인키 파일은 저장소에 커밋하지 않습니다(`*.pem`은 `.gitignore` 대상).

## 4. 배포 전 점검

```bash
python tools/check_execution_history_db.py
```

출력 예시(연결 정보는 절대 출력하지 않습니다):

```
Backend: PostgreSQL
Transport: 암호화 및 인증서 검증 (sslmode=verify-full)
Connect timeout: 8s (연결 재시도 1회)
Connection: OK
Latency: 12.4 ms
Schema version: 1
Schema: OK
```

종료 코드: `0` 정상 / `1` 스키마 불일치·미초기화 / `2` 설정 오류·연결 실패 / `3` 대상 미설정.

`--target staging`을 주면 `VARO_HISTORY_TEST_DATABASE_URL`을 점검합니다.

## 5. 스키마

현재 스키마 버전은 **1**입니다. 테이블은 `execution_history_schema_meta`, `execution_plans`,
`execution_items`, `execution_item_events`이며, 인덱스는
`idx_execution_plans_recorded`, `idx_execution_items_status`, `idx_execution_item_events_item`입니다.

- 스키마 생성은 `CREATE TABLE/INDEX IF NOT EXISTS` 기반이라 **몇 번을 실행해도 안전**하며 기존 데이터를
  지우지 않습니다.
- 앱은 시작 시 필요한 구조를 만들되, 이미 최신 버전이면 아무 것도 바꾸지 않습니다.
- 저장된 버전이 코드가 아는 버전보다 높으면 **쓰기를 거부**합니다(구버전 앱이 신버전 DB를 훼손하지 않도록).

## 6. SQLite → PostgreSQL 이관

```bash
python tools/migrate_execution_history.py --dry-run   # 쓰기 없음
python tools/migrate_execution_history.py --apply
```

- 원본 SQLite는 **읽기 전용**으로만 열며 이관 중에도 변경되지 않습니다.
- dry-run은 계획/항목/감사 건수, 대상 중복, 무결성 문제를 먼저 보고합니다.
- apply는 신규 `plan_id`만 하나의 transaction으로 넣고, 넣은 뒤 건수를 다시 세어 검증합니다.
  건수가 어긋나면 commit하지 않습니다.
- **중복 안전**: 같은 원본을 두 번 이관해도 신규 0건 / 중복 N건으로 처리되며 기존 데이터를 덮어쓰지 않습니다.
- 중간 실패 시 전체 rollback되어 계획만 남는 부분 이관이 생기지 않습니다.

## 7. 백업

> GitHub는 백업이 아닙니다. 실행 이력은 운영 데이터입니다.

**1순위: 관리형 PostgreSQL 제공업체의 백업/스냅샷 기능**을 켜 두세요. 보관 주기·복구 시점(PITR)은
코드와 무관한 운영 요구사항이며 제공업체 콘솔에서 설정합니다.

**2순위: 물리 백업(`pg_dump`)** — 운영자가 수동으로 실행합니다.

```bash
# 비밀번호는 명령줄이 아니라 ~/.pgpass 또는 환경의 비밀 설정으로 전달하세요.
pg_dump --format=custom --no-owner --file=varo_history.dump \
  --table=execution_history_schema_meta --table=execution_plans \
  --table=execution_items --table=execution_item_events "<연결 문자열>"
```

명령 이력(`history`)과 문서에 비밀번호가 남지 않게 하세요. `*.dump`, `*.pgdump`는 `.gitignore` 대상입니다.

**3순위: 논리 스냅샷(이 저장소의 도구)** — 이식 가능한 JSON으로 무결성까지 함께 검증합니다.

```bash
python tools/backup_execution_history.py --output backups/history-YYYYMMDD.json
```

계획/항목/감사 건수, 무결성 검증 결과, SHA-256을 출력하며 연결 정보는 출력하지 않습니다.
기존 파일이 있으면 `--overwrite` 없이는 덮어쓰지 않습니다.

## 8. 복원

복원은 **운영자 수동 절차**입니다. 앱 UI에는 복원 기능이 없고 앞으로도 넣지 않습니다.

```bash
# 대상은 복원 전용 변수로만 지정합니다. 운영 변수는 복원 대상이 될 수 없습니다.
export VARO_HISTORY_RESTORE_DATABASE_URL="postgresql://<user>:<password>@<host>:5432/<restore_db>"

python tools/restore_execution_history.py --input backups/history-YYYYMMDD.json \
  --confirm-database <restore_db> --dry-run
python tools/restore_execution_history.py --input backups/history-YYYYMMDD.json \
  --confirm-database <restore_db> --apply
```

안전장치:

- `VARO_HISTORY_DATABASE_URL`(운영)은 복원 대상으로 **절대 사용되지 않습니다.**
- 복원 URL이 운영 URL과 같으면 거부합니다.
- `--confirm-database` 값이 대상 DB 이름과 정확히 일치해야 실행합니다.
- `--dry-run` 또는 `--apply`를 명시해야 하며, 기본 동작은 없습니다.
- 기존 계획은 **덮어쓰거나 삭제하지 않고** 신규 `plan_id`만 넣습니다.
- 백업 파일의 형식·스키마 버전·무결성 검증을 통과하지 못하면 아무 것도 쓰지 않습니다.

`pg_dump`로 만든 물리 백업은 `pg_restore`로 복원하되, 반드시 **별도의 복원용 데이터베이스**를 만들어
먼저 복원하고 건수를 비교한 뒤 전환하세요. 운영 DB에 곧바로 복원하지 마세요.

## 9. 백업 검증

백업 파일을 만든 것만으로 성공이라고 보지 않습니다. 다음까지 확인해야 검증된 백업입니다.

1. 백업 생성 (건수·무결성·SHA-256 확인)
2. **별도 복원 대상**에 `--dry-run` → `--apply`
3. 원본과 복원 대상의 계획/항목/감사 건수 비교
4. 표본 계획 하나를 열어 상태·수량·사후 결과가 같은지 확인

```bash
python tools/check_execution_history_db.py --target staging   # 복원 대상 스키마 확인
```

## 10. 상태 점검(health check)

```bash
python tools/check_execution_history_db.py     # 연결 + 스키마 + 지연시간
```

- 운영/진단용 도구이며 일반 사용자 화면에는 노출하지 않습니다.
- 출력에는 URL·host·port·계정·비밀번호가 포함되지 않습니다.
- **DB health와 추천 health는 분리되어 있습니다.** DB가 죽어도 추천 계산은 계속 사용할 수 있습니다.

## 11. staging 통합 검증

배포 전 실제 PostgreSQL에 대해 전체 흐름을 검증하려면 **비운영 staging DB**를 준비한 뒤:

```bash
export VARO_HISTORY_TEST_DATABASE_URL="postgresql://<user>:<password>@<host>:5432/<staging_db>?sslmode=require"
python tools/validate_postgresql_history.py
python -m pytest -q tests/integration        # 같은 검증을 pytest로 실행
```

- 이 변수가 없으면 `PostgreSQL staging URL not configured.`를 출력하고 **미검증으로 종료**합니다.
  검증하지 못한 항목을 성공으로 표시하지 않습니다.
- staging URL이 운영 URL과 같으면 거부합니다.
- 검증이 만드는 기록은 `VARO-STAGING-CHECK-` 접두사를 쓰고 끝나면 지웁니다. 테이블을 DROP하거나
  TRUNCATE하지 않습니다.
- 검증 항목: 연결 / 스키마 생성 idempotency / PK·FK·인덱스·버전 / 쓰기 / 재조회 / 수정과 감사 /
  중복 안전 / transaction rollback / 동시 저장 / 동시 수정(lost update) / pagination / 재연결 / 내보내기.

`--read-only`를 주면 쓰기 없이 연결과 스키마만 확인합니다.

## 12. 장애 시 행동

| 증상 | 앱 동작 | 운영자 조치 |
| --- | --- | --- |
| DB 연결 불가 | 추천은 정상. 이력 저장/조회만 실패 메시지 | `check_execution_history_db.py`로 연결·스키마 확인 |
| 저장 중 실패 | 부분 저장 없이 전체 rollback. **성공으로 표시하지 않음** | 원인 해소 후 사용자가 다시 기록 |
| 스키마 불일치 | 저장 실패 메시지 | 배포 버전과 DB 스키마 버전 확인 |
| 설정 오류 | 저장 backend 설정 오류로 명확히 실패 | 환경변수 형식 확인. SQLite 우회 없음 |

원칙:

- 저장 실패를 성공으로 표시하지 않습니다.
- 이미 commit 되었는지 알 수 없는 쓰기는 **자동으로 재시도하지 않습니다**(중복 저장 위험).
  재시도는 부작용이 없는 **연결 수립**에만 적용합니다.
- 사용자 화면에는 DB URL·host·SQL·traceback을 노출하지 않습니다. 표시는 `실행 이력 저장: 로컬/서버`뿐입니다.

## 13. 성능과 규모

`python tools/run_history_storage_benchmark.py` 로 저장소 성능만 따로 측정합니다(추천 성능과 섞지 않음).

계획 1,000건 / 항목 30,000건 SQLite 기준 측정값:

| 작업 | 시간 |
| --- | --- |
| 계획 저장 (1건, 항목 30개) | 3.1 ms |
| 최근 이력 20건 조회 | 0.9 ms |
| 이력 20건 조회 (offset 500) | 1.4 ms |
| 항목 실행 결과 수정 1건 | 2.7 ms |
| calibration CSV 전체 내보내기 (30,000행) | 2.7 s, 피크 메모리 약 148 MB |
| 이관 dry-run / apply | 0.57 s / 0.90 s |

- 화면은 최근 20건만 불러오고, 사용자가 요청할 때만 과거 기록을 더 읽습니다.
- 조회는 `plan_id`(기본키), `recorded_at`(정렬), `execution_status`, 감사 조회용
  `(plan_id, candidate_id)` 인덱스로 처리합니다. 실제 쿼리가 쓰지 않는 컬럼에는 인덱스를 만들지 않습니다.
- 내보내기는 전체 이력을 한 번에 메모리에 올립니다. 항목 수만 건까지는 문제가 없지만
  **수십만 건 규모에서는 기간 필터나 DB 측 내보내기**로 바꾸는 것이 좋습니다.

## 14. 현재 한계

- 사용자 계정·로그인 기능은 없습니다. 이력은 배포 단위로 하나의 공용 기록입니다.
- 스키마에 `organization`/`workspace`/`user` scope 컬럼은 없습니다. 다만 계획 단위 기본키
  (`execution_plans.plan_id`)에 scope 컬럼을 추가하고 하위 테이블에 전파하는 방식이 막혀 있지는 않으며,
  그때는 스키마 버전을 올리고 기존 데이터 이관 계획을 함께 세워야 합니다. 지금은 의미 없는 tenant
  식별자를 미리 넣지 않습니다.
- 연결은 작업마다 열고 닫습니다. 현재 규모(요청당 1~3 ms 수준)에서 연결 비용이 병목이 아니어서
  connection pool을 두지 않았습니다. 동시 사용자가 늘어 측정상 병목이 확인되면 그때 도입합니다.

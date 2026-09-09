"""사용자 내보내기 버튼을 만드는 공용 헬퍼.

내보내기의 주인은 화면마다 하나다:

    재고 운영      실행계획 CSV / Excel        — 현장에서 실행할 계획 (실행 수량 기준)
    데이터 관리     데이터 오류 목록 CSV        — 입력 데이터를 고치기 위한 목록
    분석 및 검증    상세 분석 결과 · 검증 결과   — 연구 · 검증용 상세 결과
    실행 이력      실행 기록 CSV              — 계획 대비 실제 결과 누적 (calibration)

같은 데이터를 두 화면에서 다른 이름으로 다시 제공하지 않는다.

파일을 만들지 못하면 화면에는 한 줄 안내만 남는다. traceback · 파일 경로 · 내부
함수명 · 예외 이름은 사용자에게 보이지 않는다.
"""
from __future__ import annotations

from typing import Any, Callable

CSV_MIME = "text/csv"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
JSON_MIME = "application/json"

# 실패했다는 사실만 알린다. 원인 문자열은 사용자 화면에 올리지 않는다.
EXPORT_FAILED_MESSAGE = "파일을 만들지 못했습니다. 잠시 후 다시 시도해주세요."
# 내보낼 것이 없으면 빈 파일을 만들지 않고 이유를 말한다.
EMPTY_PLAN_MESSAGE = "내보낼 실행계획이 없습니다."


def render_download(
    container: Any,
    label: str,
    builder: Callable[[], bytes],
    file_name: str,
    mime: str,
    key: str,
    *,
    help: str | None = None,
    width: str | None = None,
) -> bool:
    """``builder()``가 만든 파일을 내려받는 버튼 하나. 실패하면 안내 한 줄.

    바이트는 버튼을 그릴 때 만들어지므로(Streamlit ``download_button``의 계약),
    빌더가 실패하면 페이지 전체가 예외로 죽는다. 그래서 여기서만 예외를 삼키고
    사용자에게는 :data:`EXPORT_FAILED_MESSAGE`만 보여준다.
    """
    try:
        payload = builder()
    except Exception:  # noqa: BLE001 - 내부 원인은 화면에 노출하지 않는다
        container.caption(EXPORT_FAILED_MESSAGE)
        return False
    kwargs: dict[str, Any] = {"data": payload, "file_name": file_name, "mime": mime, "key": key}
    if help:
        kwargs["help"] = help
    if width:
        kwargs["width"] = width
    container.download_button(label, **kwargs)
    return True

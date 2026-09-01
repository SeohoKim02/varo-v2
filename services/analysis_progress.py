"""Stage progress contract for the recommendation run.

The analysis pipeline is a long, blocking call. This module lets it report *which
real stage it is in* so a UI can show something other than a frozen screen. It
holds no Streamlit import and no rendering: the pipeline emits events, and the
page/component layer decides how to draw them.

Design rules this module encodes:

* **Real stages only.** Every stage below corresponds to work that actually
  happens in ``analysis_pipeline.run_analysis_pipeline``. Nothing is emitted on a
  timer and nothing advances while the pipeline is idle.
* **Stage granularity, not row granularity.** One event per major stage (8 total
  for a whole run), so a 14,000-row workbook costs the same number of events as a
  900-row one.
* **The callback can never change the result.** A callback that raises is
  swallowed and logged; the pipeline continues exactly as if no callback had been
  passed.

The ``progress`` values are "how far through the stages", not a time estimate.
They are spaced by measured stage cost (candidate clustering and the judgment-log
build dominate a real run) so the bar does not sit at 95% for half the wait, but
they still only move when a stage actually completes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgressEvent:
    """One stage boundary. ``stage`` is a stable code, ``message`` is for users."""

    stage: str
    message: str
    progress: float
    index: int
    total: int
    ok: bool = True


# (stage code, user-facing line, progress when the stage starts)
ANALYSIS_STAGES: tuple[tuple[str, str, float], ...] = (
    ("validation", "데이터를 확인하고 있습니다.", 0.03),
    ("inventory", "재고 상태를 확인하고 있습니다.", 0.08),
    ("candidates", "이동 가능한 후보를 찾고 있습니다.", 0.12),
    ("scoring", "추천 우선순위를 계산하고 있습니다.", 0.40),
    ("routes", "이동 경로와 비용을 비교하고 있습니다.", 0.45),
    ("verification", "추천 결과를 확인하고 있습니다.", 0.60),
    ("summary", "결과를 정리하고 있습니다.", 0.65),
    ("complete", "추천 계산이 완료됐습니다.", 1.0),
)

FAILED_STAGE = "failed"
FAILED_MESSAGE = "추천을 완료하지 못했습니다."
RUNNING_MESSAGE = "추천을 계산하고 있습니다."

_BY_CODE: Mapping[str, tuple[str, float, int]] = {
    code: (message, progress, index)
    for index, (code, message, progress) in enumerate(ANALYSIS_STAGES, start=1)
}

ProgressCallback = Callable[[ProgressEvent], None]


def stage_order() -> list[str]:
    """Canonical stage codes in the order the pipeline emits them."""
    return [code for code, _message, _progress in ANALYSIS_STAGES]


class ProgressReporter:
    """Emits stage events to an optional callback, never raising to the caller.

    Constructed once per pipeline run. When ``callback`` is ``None`` every call is
    a no-op, so a run without a UI attached behaves exactly as before.
    """

    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self._callback = callback
        self._progress = 0.0

    @property
    def enabled(self) -> bool:
        return self._callback is not None

    def __call__(self, stage: str) -> None:
        message, progress, index = _BY_CODE.get(stage, (stage, self._progress, 0))
        self._progress = progress
        self._emit(ProgressEvent(stage, message, progress, index, len(ANALYSIS_STAGES), True))

    def fail(self) -> None:
        """Report that the run stopped early; the bar stays where it was."""
        self._emit(ProgressEvent(
            FAILED_STAGE, FAILED_MESSAGE, self._progress, 0, len(ANALYSIS_STAGES), False,
        ))

    def _emit(self, event: ProgressEvent) -> None:
        if self._callback is None:
            return
        try:
            self._callback(event)
        except Exception:  # pragma: no cover - a broken UI must not break analysis
            logger.debug("progress callback failed at stage %s", event.stage, exc_info=True)

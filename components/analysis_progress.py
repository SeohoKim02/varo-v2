"""Streamlit rendering of the recommendation-run progress.

This is the only place that turns pipeline progress events into widgets. The
service layer stays free of Streamlit: it emits
``services.analysis_progress.ProgressEvent`` values and this view draws them.

The block is deliberately small — one status line, the current step, and a bar —
so it never competes with the result below it. It is created when the user clicks
추천 실행 and disappears on the rerun that follows, so nothing is left on screen
and no completion message accumulates across reruns.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from services.analysis_progress import FAILED_MESSAGE, RUNNING_MESSAGE, ProgressEvent

_COMPLETE_MESSAGE = "추천 계산이 완료됐습니다."


class AnalysisProgressView:
    """Owns one status container and updates it as real stages complete.

    Every update is driven by a pipeline event; nothing here advances on a timer
    and nothing sleeps.
    """

    def __init__(self, container: Any | None = None) -> None:
        target = container if container is not None else st
        self._status = target.status(RUNNING_MESSAGE, expanded=True)
        self._bar = self._status.progress(0.0, text="분석을 시작하고 있습니다.")
        self._progress = 0.0

    def callback(self, event: ProgressEvent) -> None:
        """Progress callback handed to the analysis run."""
        self._progress = max(0.0, min(1.0, float(event.progress)))
        if not event.ok:
            return
        self._bar.progress(self._progress, text=event.message)

    def finish(self, succeeded: bool, elapsed_seconds: float | None = None) -> None:
        """Collapse the block into a one-line outcome."""
        if succeeded:
            label = _COMPLETE_MESSAGE
            if elapsed_seconds is not None and float(elapsed_seconds) >= 1.0:
                label = f"{_COMPLETE_MESSAGE[:-1]} · {float(elapsed_seconds):.1f}초"
            self._bar.progress(1.0, text=_COMPLETE_MESSAGE)
            self._status.update(label=label, state="complete", expanded=False)
            return
        self._status.update(label=FAILED_MESSAGE, state="error", expanded=False)


def completion_note(elapsed_seconds: float | None) -> str:
    """One short line shown once after a successful run.

    The duration is the measured wall time of the run. It is context for the wait
    the user just sat through, not a performance figure, so it is only shown when
    the wait was long enough to notice.
    """
    if elapsed_seconds is not None and float(elapsed_seconds) >= 1.0:
        return f"추천 계산이 완료됐습니다 · {float(elapsed_seconds):.1f}초"
    return _COMPLETE_MESSAGE

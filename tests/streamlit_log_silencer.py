"""Test-only Streamlit log hygiene.

Streamlit's AppTest and cached decorators can emit bare-mode diagnostics when
unit tests import pages outside ``streamlit run``. These are expected test-harness
messages, not application warnings, so tests lower only Streamlit's log level
while leaving Python warnings and non-Streamlit errors visible.
"""
from __future__ import annotations

import logging
import warnings
from importlib import import_module


def quiet_streamlit_test_logs() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Implicitly cleaning up <TemporaryDirectory.*",
        category=ResourceWarning,
        module=r"tempfile",
    )
    for logger_name in (
        "streamlit",
        "streamlit.runtime.caching.cache_data_api",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    for module_name in (
        "streamlit.runtime.caching.cache_data_api",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
    ):
        try:
            logger = getattr(import_module(module_name), "_LOGGER", None)
        except Exception:
            continue
        if logger is not None:
            logger.setLevel(logging.ERROR)
            logger.disabled = True

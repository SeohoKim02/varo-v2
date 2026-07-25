"""Narrow runtime-log hygiene for the harmless Windows socket-shutdown message.

Streamlit on Windows logs a `ConnectionResetError [WinError 10054]` from asyncio's
ProactorEventLoop (`_ProactorBasePipeTransport._call_connection_lost`) whenever a
browser tab refreshes or the server shuts down and the peer drops the socket. It
is not an application error. This installs a logging filter that drops ONLY that
one shutdown record on the `asyncio` logger. Every other asyncio error — and all
application, import, and data errors — still logs normally.
"""
from __future__ import annotations

import logging

_CONNECTION_LOST_MARKER = "_ProactorBasePipeTransport._call_connection_lost"


class _SocketShutdownFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if _CONNECTION_LOST_MARKER not in message:
            return True  # not the shutdown noise → keep
        exc = record.exc_info[1] if record.exc_info else None
        # Only silence when it is genuinely the peer-reset shutdown case.
        if isinstance(exc, ConnectionResetError) or "10054" in message:
            return False
        return True


_installed = False


def quiet_windows_socket_shutdown_logs() -> None:
    """Install the narrow filter once (idempotent). Safe to call every rerun."""
    global _installed
    if _installed:
        return
    logging.getLogger("asyncio").addFilter(_SocketShutdownFilter())
    _installed = True

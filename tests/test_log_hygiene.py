"""The socket-shutdown log filter must drop ONLY the harmless peer-reset record."""
from __future__ import annotations

import logging
import unittest

from services.log_hygiene import _SocketShutdownFilter, quiet_windows_socket_shutdown_logs


def _record(msg: str, exc: BaseException | None = None) -> logging.LogRecord:
    exc_info = (type(exc), exc, None) if exc is not None else None
    return logging.LogRecord("asyncio", logging.ERROR, __file__, 1, msg, (), exc_info)


class SocketShutdownFilterTests(unittest.TestCase):
    def setUp(self):
        self.filter = _SocketShutdownFilter()

    def test_drops_connection_lost_reset(self):
        record = _record(
            "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)",
            ConnectionResetError(10054, "peer reset"),
        )
        self.assertFalse(self.filter.filter(record))

    def test_drops_when_winerror_10054_in_message(self):
        record = _record("_ProactorBasePipeTransport._call_connection_lost ... [WinError 10054]")
        self.assertFalse(self.filter.filter(record))

    def test_keeps_other_asyncio_errors(self):
        self.assertTrue(self.filter.filter(_record("Task exception was never retrieved")))
        self.assertTrue(self.filter.filter(_record("some other _call_connection_lost text without reset")))

    def test_keeps_real_app_errors(self):
        record = _record("Uncaught app execution", ValueError("boom"))
        self.assertTrue(self.filter.filter(record))

    def test_install_is_idempotent(self):
        logger = logging.getLogger("asyncio")
        before = len(logger.filters)
        quiet_windows_socket_shutdown_logs()
        quiet_windows_socket_shutdown_logs()
        after = len(logger.filters)
        self.assertLessEqual(after - before, 1)


if __name__ == "__main__":
    unittest.main()

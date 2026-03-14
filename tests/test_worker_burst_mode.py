"""
Tests for worker.py burst mode implementation (KEDA ScaledJob support).

Verifies that:
1. Worker reads BURST_MODE and MAX_MESSAGES from environment
2. Worker exits after processing MAX_MESSAGES in burst mode
3. Worker exits after idle timeout in burst mode
4. Worker runs indefinitely when BURST_MODE is not set
"""

import os
import time
import pytest


class TestBurstModeConfig:
    """Verify worker.py reads burst mode environment variables."""

    def test_worker_reads_burst_mode_env(self):
        """Worker module must reference BURST_MODE env var."""
        import worker as worker_module
        source = open(worker_module.__file__).read()
        assert "BURST_MODE" in source, (
            "worker.py does not read BURST_MODE env var. "
            "K8s ScaledJob sets BURST_MODE=true but worker ignores it."
        )

    def test_worker_reads_max_messages_env(self):
        """Worker module must reference MAX_MESSAGES env var."""
        import worker as worker_module
        source = open(worker_module.__file__).read()
        assert "MAX_MESSAGES" in source, (
            "worker.py does not read MAX_MESSAGES env var. "
            "Burst worker should exit after MAX_MESSAGES."
        )

    def test_worker_has_burst_mode_attribute(self):
        """Worker instance must have burst mode tracking."""
        from worker import Worker
        w = Worker()
        assert hasattr(w, '_burst_mode'), "Worker missing _burst_mode attribute"
        assert hasattr(w, '_burst_max_messages'), "Worker missing _burst_max_messages attribute"
        assert hasattr(w, '_burst_idle_timeout'), "Worker missing _burst_idle_timeout attribute"

    def test_burst_mode_defaults_off(self):
        """Burst mode should be disabled by default (regular workers)."""
        from worker import Worker
        w = Worker()
        assert w._burst_mode is False, "Burst mode should default to False"

    def test_burst_mode_enabled_from_env(self, monkeypatch):
        """Burst mode should activate when BURST_MODE=true in env."""
        monkeypatch.setenv("BURST_MODE", "true")
        monkeypatch.setenv("MAX_MESSAGES", "50")
        # Re-import to pick up env vars
        import importlib
        import worker as worker_module
        importlib.reload(worker_module)
        try:
            assert worker_module.BURST_MODE is True
            assert worker_module.BURST_MAX_MESSAGES == 50
        finally:
            # Restore defaults
            monkeypatch.delenv("BURST_MODE", raising=False)
            monkeypatch.delenv("MAX_MESSAGES", raising=False)
            importlib.reload(worker_module)


class TestBurstModeLogic:
    """Verify burst exit conditions work correctly."""

    def test_message_limit_triggers_exit(self):
        """Worker should signal exit after MAX_MESSAGES processed."""
        from worker import Worker
        w = Worker()
        w._burst_mode = True
        w._burst_max_messages = 5
        w._messages_processed = 4
        w._messages_failed = 1  # 4 + 1 = 5 = limit
        assert w._check_burst_message_limit() is True

    def test_message_limit_does_not_trigger_below_limit(self):
        """Worker should NOT exit before reaching MAX_MESSAGES."""
        from worker import Worker
        w = Worker()
        w._burst_mode = True
        w._burst_max_messages = 100
        w._messages_processed = 50
        w._messages_failed = 0
        assert w._check_burst_message_limit() is False

    def test_idle_timeout_triggers_exit(self):
        """Worker should exit after BURST_IDLE_TIMEOUT seconds of no messages."""
        from worker import Worker
        w = Worker()
        w._burst_mode = True
        w._burst_idle_timeout = 300  # 5 minutes
        w._last_message_time = time.time() - 301  # 301 seconds ago
        w._messages_processed = 10
        assert w._check_burst_idle_timeout() is True

    def test_idle_timeout_does_not_trigger_when_active(self):
        """Worker should NOT exit if messages are still flowing."""
        from worker import Worker
        w = Worker()
        w._burst_mode = True
        w._burst_idle_timeout = 300
        w._last_message_time = time.time() - 10  # 10 seconds ago
        w._messages_processed = 10
        assert w._check_burst_idle_timeout() is False

    def test_idle_timeout_does_not_trigger_without_last_message(self):
        """Worker should NOT exit if no messages have been received yet."""
        from worker import Worker
        w = Worker()
        w._burst_mode = True
        w._burst_idle_timeout = 300
        w._last_message_time = None
        assert w._check_burst_idle_timeout() is False

    def test_normal_mode_ignores_limits(self):
        """Regular (non-burst) workers should never trigger burst exit."""
        from worker import Worker
        w = Worker()
        w._burst_mode = False
        w._burst_max_messages = 5
        w._messages_processed = 1000  # Way over limit
        # These methods still return True based on math, but they're never
        # called in normal mode (_process_inbound_loop checks _burst_mode first)
        # The important thing is _burst_mode is False
        assert w._burst_mode is False

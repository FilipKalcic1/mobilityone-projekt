"""
Circuit Breaker Tests - Endpoint failure protection.

Tests state transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
and edge cases like concurrent access, manual reset, etc.
"""

import time
import pytest
from unittest.mock import AsyncMock, patch

from services.circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError


class TestCircuitBreakerStateTransitions:
    """Test the CLOSED -> OPEN -> HALF_OPEN -> CLOSED lifecycle."""

    @pytest.fixture
    def breaker(self):
        return CircuitBreaker()

    @pytest.mark.asyncio
    async def test_starts_closed(self, breaker):
        status = await breaker.get_status("test-endpoint")
        assert status["state"] == "closed"
        assert status["never_used"] is True

    @pytest.mark.asyncio
    async def test_stays_closed_on_success(self, breaker):
        func = AsyncMock(return_value="ok")
        result = await breaker.call("test-endpoint", func)
        assert result == "ok"

        status = await breaker.get_status("test-endpoint")
        assert status["state"] == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_3_consecutive_failures(self, breaker):
        func = AsyncMock(side_effect=Exception("API down"))

        for i in range(3):
            with pytest.raises(Exception, match="API down"):
                await breaker.call("test-endpoint", func)

        status = await breaker.get_status("test-endpoint")
        assert status["state"] == CircuitState.OPEN
        assert status["failure_count"] == 3

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_calls(self, breaker):
        # First, open the circuit
        func = AsyncMock(side_effect=Exception("API down"))
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call("test-endpoint", func)

        # Now calls should be blocked without even calling the function
        with pytest.raises(CircuitOpenError):
            await breaker.call("test-endpoint", AsyncMock())

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, breaker):
        # Open the circuit
        func = AsyncMock(side_effect=Exception("fail"))
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call("test-endpoint", func)

        # Simulate time passing (mock opened_at to be 61s ago)
        circuit = breaker.circuits["test-endpoint"]
        circuit.opened_at = time.time() - 61

        # Next call should go through (HALF_OPEN state allows one attempt)
        success_func = AsyncMock(return_value="recovered")
        result = await breaker.call("test-endpoint", success_func)
        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_closes_after_5_successes_in_half_open(self, breaker):
        # Open the circuit
        func = AsyncMock(side_effect=Exception("fail"))
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call("test-endpoint", func)

        # Move to HALF_OPEN
        circuit = breaker.circuits["test-endpoint"]
        circuit.opened_at = time.time() - 61

        # 5 successes should close the circuit
        success_func = AsyncMock(return_value="ok")
        for _ in range(5):
            # Transition to HALF_OPEN happens on first call
            if circuit.state == CircuitState.OPEN:
                circuit.opened_at = time.time() - 61
            await breaker.call("test-endpoint", success_func)

        status = await breaker.get_status("test-endpoint")
        assert status["state"] == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens(self, breaker):
        # Open the circuit
        func = AsyncMock(side_effect=Exception("fail"))
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call("test-endpoint", func)

        # Move to HALF_OPEN
        circuit = breaker.circuits["test-endpoint"]
        circuit.opened_at = time.time() - 61

        # Failure in HALF_OPEN should re-open
        with pytest.raises(Exception):
            await breaker.call("test-endpoint", AsyncMock(side_effect=Exception("still down")))

        status = await breaker.get_status("test-endpoint")
        assert status["state"] == CircuitState.OPEN


class TestCircuitBreakerEdgeCases:
    """Test edge cases and concurrent behavior."""

    @pytest.fixture
    def breaker(self):
        return CircuitBreaker()

    @pytest.mark.asyncio
    async def test_different_endpoints_independent(self, breaker):
        """Failures on one endpoint don't affect another."""
        fail_func = AsyncMock(side_effect=Exception("fail"))
        ok_func = AsyncMock(return_value="ok")

        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call("endpoint-A", fail_func)

        # Endpoint B should still work
        result = await breaker.call("endpoint-B", ok_func)
        assert result == "ok"

        status_a = await breaker.get_status("endpoint-A")
        status_b = await breaker.get_status("endpoint-B")
        assert status_a["state"] == CircuitState.OPEN
        assert status_b["state"] == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_success_resets_failure_counter(self, breaker):
        """A success between failures resets the failure count."""
        fail_func = AsyncMock(side_effect=Exception("fail"))
        ok_func = AsyncMock(return_value="ok")

        # 2 failures
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call("test-endpoint", fail_func)

        # 1 success resets counter
        await breaker.call("test-endpoint", ok_func)

        # 2 more failures should NOT open (counter reset)
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call("test-endpoint", fail_func)

        status = await breaker.get_status("test-endpoint")
        assert status["state"] == CircuitState.CLOSED  # Still closed (only 2 consecutive)

    @pytest.mark.asyncio
    async def test_manual_reset(self, breaker):
        """Manual reset should close an open circuit."""
        func = AsyncMock(side_effect=Exception("fail"))
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call("test-endpoint", func)

        await breaker.reset("test-endpoint")

        status = await breaker.get_status("test-endpoint")
        assert status["state"] == CircuitState.CLOSED
        assert status["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_reset_nonexistent_endpoint_noop(self, breaker):
        """Resetting an unknown endpoint should not raise."""
        await breaker.reset("never-used-endpoint")

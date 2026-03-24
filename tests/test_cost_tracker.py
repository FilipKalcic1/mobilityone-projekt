"""Tests for CostTracker - LLM usage cost tracking."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.cost_tracker import CostTracker, DailyStats


@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    pipe = AsyncMock()
    redis.pipeline.return_value = pipe
    pipe.hincrby = MagicMock()
    pipe.expire = MagicMock()
    pipe.execute = AsyncMock()
    return redis


@pytest.fixture
def tracker(mock_redis):
    return CostTracker(mock_redis)


class TestDailyStats:
    def test_defaults(self):
        s = DailyStats(date="2025-01-15")
        assert s.requests == 0
        assert s.cost_usd == 0.0


class TestCalculateCost:
    def test_basic_cost(self, tracker):
        cost = tracker._calculate_cost(1000, 500)
        assert cost > 0

    def test_zero_tokens(self, tracker):
        cost = tracker._calculate_cost(0, 0)
        assert cost == 0.0


class TestRecordUsage:
    async def test_updates_session(self, tracker):
        await tracker.record_usage(100, 50)
        assert tracker._session["prompt"] == 100
        assert tracker._session["completion"] == 50
        assert tracker._session["requests"] == 1

    async def test_returns_cost(self, tracker):
        cost = await tracker.record_usage(1000, 500)
        assert cost > 0

    async def test_redis_pipeline_called(self, tracker, mock_redis):
        await tracker.record_usage(100, 50)
        mock_redis.pipeline.assert_called_once()

    async def test_multiple_recordings(self, tracker):
        await tracker.record_usage(100, 50)
        await tracker.record_usage(200, 100)
        assert tracker._session["prompt"] == 300
        assert tracker._session["requests"] == 2

    async def test_redis_failure_doesnt_crash(self, tracker, mock_redis):
        pipe = mock_redis.pipeline.return_value
        pipe.execute.side_effect = Exception("Redis down")
        cost = await tracker.record_usage(100, 50)
        assert cost > 0


class TestGetDailyStats:
    async def test_returns_stats(self, tracker, mock_redis):
        mock_redis.hgetall.return_value = {
            "requests": "10",
            "prompt_tokens": "5000",
            "completion_tokens": "2000",
            "cost_microcents": "500000"
        }
        stats = await tracker.get_daily_stats("2025-01-15")
        assert stats.requests == 10
        assert stats.prompt_tokens == 5000

    async def test_empty_returns_defaults(self, tracker, mock_redis):
        mock_redis.hgetall.return_value = {}
        stats = await tracker.get_daily_stats("2025-01-15")
        assert stats.requests == 0

    async def test_redis_failure_returns_defaults(self, tracker, mock_redis):
        mock_redis.hgetall.side_effect = Exception("Redis down")
        stats = await tracker.get_daily_stats("2025-01-15")
        assert stats.requests == 0


class TestGetSessionStats:
    async def test_session_stats(self, tracker):
        await tracker.record_usage(100, 50)
        stats = await tracker.get_session_stats()
        assert stats["session_prompt_tokens"] == 100
        assert stats["session_completion_tokens"] == 50
        assert stats["session_requests"] == 1
        assert "session_duration_seconds" in stats


class TestGetTotalStats:
    """Test get_total_stats method - covers lines 108-116."""

    async def test_aggregates_daily_stats(self, tracker, mock_redis):
        """Test that total stats aggregates daily stats."""
        mock_redis.hgetall.return_value = {
            "requests": "10",
            "prompt_tokens": "1000",
            "completion_tokens": "500",
            "cost_microcents": "100000"
        }

        total = await tracker.get_total_stats()

        assert total["prompt_tokens"] >= 1000  # At least one day
        assert total["completion_tokens"] >= 500
        assert total["cost_usd"] >= 0
        assert "total_tokens" in total

    async def test_handles_empty_days(self, tracker, mock_redis):
        """Test handling of empty daily stats."""
        mock_redis.hgetall.return_value = {}

        total = await tracker.get_total_stats()

        assert total["prompt_tokens"] == 0
        assert total["completion_tokens"] == 0
        assert total["cost_usd"] == 0.0
        assert total["total_tokens"] == 0


class TestGetCostTrackerSingleton:
    """Test get_cost_tracker factory - covers lines 137-139."""

    async def test_creates_singleton(self):
        """Test singleton creation."""
        import services.cost_tracker as ct_module
        ct_module._cost_tracker = None  # Reset

        from services.cost_tracker import get_cost_tracker

        redis = AsyncMock()
        t1 = await get_cost_tracker(redis)
        t2 = await get_cost_tracker(redis)

        assert t1 is t2

    async def test_creates_instance_if_none(self):
        """Test creates new instance if none exists."""
        import services.cost_tracker as ct_module
        ct_module._cost_tracker = None  # Reset

        from services.cost_tracker import get_cost_tracker

        redis = AsyncMock()
        result = await get_cost_tracker(redis)

        assert result is not None
        assert isinstance(result, CostTracker)

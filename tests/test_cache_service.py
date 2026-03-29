"""Tests for CacheService - Redis caching with resilience."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from uuid import UUID
from services.cache_service import CacheService, SafeJSONEncoder


class TestSafeJSONEncoder:
    def test_datetime_serialization(self):
        dt = datetime(2025, 1, 15, 9, 0)
        result = json.dumps({"dt": dt}, cls=SafeJSONEncoder)
        assert "2025-01-15T09:00:00" in result

    def test_date_serialization(self):
        from datetime import date
        d = date(2025, 1, 15)
        result = json.dumps({"d": d}, cls=SafeJSONEncoder)
        assert "2025-01-15" in result

    def test_uuid_serialization(self):
        uid = UUID("12345678-1234-5678-1234-567812345678")
        result = json.dumps({"id": uid}, cls=SafeJSONEncoder)
        assert "12345678-1234-5678-1234-567812345678" in result

    def test_unknown_type_str_fallback(self):
        class Custom:
            def __str__(self):
                return "custom_value"
        result = json.dumps({"obj": Custom()}, cls=SafeJSONEncoder)
        assert "custom_value" in result


@pytest.fixture
def mock_redis():
    return AsyncMock()


@pytest.fixture
def cache(mock_redis):
    return CacheService(mock_redis)


class TestGet:
    pytestmark = pytest.mark.asyncio

    async def test_get_value(self, cache, mock_redis):
        mock_redis.get.return_value = "cached_value"
        result = await cache.get("key")
        assert result == "cached_value"

    async def test_get_miss(self, cache, mock_redis):
        mock_redis.get.return_value = None
        result = await cache.get("key")
        assert result is None

    async def test_get_failure_returns_none(self, cache, mock_redis):
        mock_redis.get.side_effect = Exception("fail")
        result = await cache.get("key")
        assert result is None


class TestGetJson:
    pytestmark = pytest.mark.asyncio

    async def test_get_json(self, cache, mock_redis):
        mock_redis.get.return_value = '{"a": 1}'
        result = await cache.get_json("key")
        assert result == {"a": 1}

    async def test_get_json_miss(self, cache, mock_redis):
        mock_redis.get.return_value = None
        result = await cache.get_json("key")
        assert result is None


class TestSet:
    pytestmark = pytest.mark.asyncio

    async def test_set_string(self, cache, mock_redis):
        result = await cache.set("key", "value", 300)
        assert result is True
        mock_redis.setex.assert_called_once()

    async def test_set_dict_serialized(self, cache, mock_redis):
        result = await cache.set("key", {"a": 1}, 300)
        assert result is True

    async def test_set_failure(self, cache, mock_redis):
        mock_redis.setex.side_effect = Exception("fail")
        result = await cache.set("key", "value")
        assert result is False


class TestSetJson:
    pytestmark = pytest.mark.asyncio

    async def test_set_json(self, cache, mock_redis):
        result = await cache.set_json("key", {"dt": "value"}, 300)
        assert result is True

    async def test_set_json_failure(self, cache, mock_redis):
        mock_redis.setex.side_effect = Exception("fail")
        result = await cache.set_json("key", {"a": 1})
        assert result is False


class TestDelete:
    pytestmark = pytest.mark.asyncio

    async def test_delete(self, cache, mock_redis):
        result = await cache.delete("key")
        assert result is True

    async def test_delete_failure(self, cache, mock_redis):
        mock_redis.delete.side_effect = Exception("fail")
        result = await cache.delete("key")
        assert result is False


class TestInvalidate:
    pytestmark = pytest.mark.asyncio

    async def test_invalidate_calls_delete(self, cache, mock_redis):
        result = await cache.invalidate("key")
        assert result is True
        mock_redis.delete.assert_called_once_with("key")


class TestInvalidatePattern:
    pytestmark = pytest.mark.asyncio

    async def test_invalidate_pattern(self, cache, mock_redis):
        async def mock_scan_iter(match=None, count=None):
            for k in ["key1", "key2"]:
                yield k
        mock_redis.scan_iter = mock_scan_iter
        mock_pipe = AsyncMock()
        mock_pipe.execute = AsyncMock(return_value=[1, 1])
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)
        count = await cache.invalidate_pattern("key*")
        assert count == 2

    async def test_invalidate_pattern_failure(self, cache, mock_redis):
        mock_redis.scan_iter = AsyncMock(side_effect=Exception("fail"))
        count = await cache.invalidate_pattern("key*")
        assert count == 0


class TestExists:
    pytestmark = pytest.mark.asyncio

    async def test_exists_true(self, cache, mock_redis):
        mock_redis.exists.return_value = 1
        assert await cache.exists("key") is True

    async def test_exists_false(self, cache, mock_redis):
        mock_redis.exists.return_value = 0
        assert await cache.exists("key") is False

    async def test_exists_failure(self, cache, mock_redis):
        mock_redis.exists.side_effect = Exception("fail")
        assert await cache.exists("key") is False


class TestGetOrCompute:
    pytestmark = pytest.mark.asyncio

    async def test_returns_cached(self, cache, mock_redis):
        mock_redis.get.return_value = '{"data": "cached"}'
        compute_fn = AsyncMock(return_value={"data": "computed"})
        result = await cache.get_or_compute("key", compute_fn)
        assert result == {"data": "cached"}
        compute_fn.assert_not_called()

    async def test_computes_on_miss(self, cache, mock_redis):
        mock_redis.get.return_value = None
        compute_fn = AsyncMock(return_value={"data": "computed"})
        result = await cache.get_or_compute("key", compute_fn)
        assert result == {"data": "computed"}
        compute_fn.assert_called_once()


class TestGetOrComputeStampede:
    """Tests for SETNX-based stampede protection in get_or_compute."""
    pytestmark = pytest.mark.asyncio

    async def test_acquires_lock_on_cache_miss(self, cache, mock_redis):
        """Lock should be acquired with unique token when cache is empty."""
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True  # Lock acquired
        mock_redis.eval = AsyncMock(return_value=1)
        compute_fn = AsyncMock(return_value={"data": "result"})

        result = await cache.get_or_compute("key", compute_fn, ttl=300)

        assert result == {"data": "result"}
        # Verify SETNX was called with lock key
        lock_call = [c for c in mock_redis.set.call_args_list
                     if c.args and str(c.args[0]).startswith("lock:")]
        assert len(lock_call) == 1
        assert lock_call[0].kwargs.get("nx") is True

    async def test_releases_lock_with_lua_after_compute(self, cache, mock_redis):
        """Lock should be released via Lua compare-and-delete after computing."""
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.eval = AsyncMock(return_value=1)
        compute_fn = AsyncMock(return_value={"data": "result"})

        await cache.get_or_compute("key", compute_fn, ttl=300)

        # Verify Lua eval was called for lock release
        mock_redis.eval.assert_called_once()
        lua_script = mock_redis.eval.call_args.args[0]
        assert "redis.call('get'" in lua_script
        assert "redis.call('del'" in lua_script

    async def test_waits_and_returns_cached_when_lock_held(self, cache, mock_redis):
        """When lock is held by another caller, should poll then return cached value."""
        call_count = 0

        async def mock_get(key):
            nonlocal call_count
            call_count += 1
            if key.startswith("lock:"):
                return None
            # First call: miss, subsequent calls: hit (other caller computed)
            if call_count <= 2:
                return None
            return '{"data": "from_other_caller"}'

        mock_redis.get = AsyncMock(side_effect=mock_get)
        mock_redis.set.return_value = False  # Lock NOT acquired
        compute_fn = AsyncMock(return_value={"data": "fallback"})

        import asyncio
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio, "sleep", AsyncMock())
            result = await cache.get_or_compute("key", compute_fn, ttl=300)

        assert result == {"data": "from_other_caller"}
        compute_fn.assert_not_called()

    async def test_lock_release_failure_does_not_crash(self, cache, mock_redis):
        """If Lua lock release fails, compute result should still be returned."""
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.eval = AsyncMock(side_effect=Exception("Redis gone"))
        compute_fn = AsyncMock(return_value={"data": "result"})

        result = await cache.get_or_compute("key", compute_fn, ttl=300)
        assert result == {"data": "result"}


class TestInvalidatePatternAccuracy:
    """Tests for batched pipeline delete count accuracy."""
    pytestmark = pytest.mark.asyncio

    async def test_counts_actual_deletions(self, cache, mock_redis):
        """Deleted count should reflect actual Redis delete results, not batch size."""
        mock_redis.scan_iter = self._make_scan_iter(["k1", "k2", "k3"])
        mock_pipe = AsyncMock()
        # Simulate: k1 deleted (1), k2 already gone (0), k3 deleted (1)
        mock_pipe.execute = AsyncMock(return_value=[1, 0, 1])
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)

        deleted = await cache.invalidate_pattern("test:*")
        assert deleted == 2  # Only 2 actually deleted, not 3

    @staticmethod
    def _make_scan_iter(keys):
        async def scan_iter(**kwargs):
            for k in keys:
                yield k
        return scan_iter


class TestIncrement:
    pytestmark = pytest.mark.asyncio

    async def test_increment(self, cache, mock_redis):
        mock_redis.incr.return_value = 1
        result = await cache.increment("counter")
        assert result == 1

    async def test_increment_with_ttl(self, cache, mock_redis):
        mock_redis.incr.return_value = 1
        result = await cache.increment("counter", ttl=3600)
        mock_redis.expire.assert_called_once()

    async def test_increment_failure(self, cache, mock_redis):
        mock_redis.incr.side_effect = Exception("fail")
        result = await cache.increment("counter")
        assert result == 0

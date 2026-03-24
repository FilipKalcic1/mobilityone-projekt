"""Tests for CacheService - Redis caching with resilience."""

import pytest
import json
from unittest.mock import AsyncMock
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

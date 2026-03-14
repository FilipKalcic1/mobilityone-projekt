"""
Tests for durable DLQ in webhook_simple.py.

Verifies that:
1. DLQ writes to Redis (primary path)
2. DLQ falls back to file when Redis is unreachable
3. DLQ entry format is consistent JSON
"""

import json
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_dlq_writes_to_redis_primary():
    """DLQ should LPUSH to Redis dlq:webhook as primary storage."""
    mock_redis = AsyncMock()
    mock_redis.lpush = AsyncMock()
    mock_redis.ltrim = AsyncMock()

    dlq_entry = json.dumps({
        "dlq": "webhook",
        "ts": "2026-03-11T10:00:00+00:00",
        "sender": "+385991234567",
        "text": "test message",
        "message_id": "msg_123",
        "error": "ConnectionError"
    })

    with patch("webhook_simple.get_redis", return_value=mock_redis):
        from webhook_simple import _write_dlq
        await _write_dlq(dlq_entry)

    mock_redis.lpush.assert_called_once_with("dlq:webhook", dlq_entry)
    mock_redis.ltrim.assert_called_once_with("dlq:webhook", 0, 9999)


@pytest.mark.asyncio
async def test_dlq_falls_back_to_file_when_redis_fails(tmp_path):
    """When Redis LPUSH fails, DLQ should write to local file."""
    mock_redis = AsyncMock()
    mock_redis.lpush = AsyncMock(side_effect=Exception("Redis gone"))

    dlq_path = str(tmp_path / "dlq.jsonl")
    dlq_entry = json.dumps({"dlq": "webhook", "sender": "+385991234567", "text": "test"})

    with patch("webhook_simple.get_redis", return_value=mock_redis), \
         patch("webhook_simple._DLQ_FILE_PATH", dlq_path):
        from webhook_simple import _write_dlq
        await _write_dlq(dlq_entry)

    # File should contain the entry
    assert os.path.exists(dlq_path)
    with open(dlq_path) as f:
        lines = f.readlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0].strip())
    assert parsed["sender"] == "+385991234567"


@pytest.mark.asyncio
async def test_dlq_file_appends_multiple_entries(tmp_path):
    """Multiple DLQ writes should append, not overwrite."""
    mock_redis = AsyncMock()
    mock_redis.lpush = AsyncMock(side_effect=Exception("Redis gone"))

    dlq_path = str(tmp_path / "dlq.jsonl")

    with patch("webhook_simple.get_redis", return_value=mock_redis), \
         patch("webhook_simple._DLQ_FILE_PATH", dlq_path):
        from webhook_simple import _write_dlq
        await _write_dlq(json.dumps({"dlq": "webhook", "message_id": "msg_1"}))
        await _write_dlq(json.dumps({"dlq": "webhook", "message_id": "msg_2"}))

    with open(dlq_path) as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["message_id"] == "msg_1"
    assert json.loads(lines[1])["message_id"] == "msg_2"


@pytest.mark.asyncio
async def test_dlq_does_not_use_only_stderr():
    """DLQ must NOT rely solely on stderr (the old bug)."""
    import webhook_simple
    source = open(webhook_simple.__file__).read()

    # The old pattern was: sys.stderr.write(f"DLQ_WEBHOOK: {dlq_entry}\n")
    # directly in the webhook handler. Now it should go through _write_dlq
    # which tries Redis first, then file, then stderr as last resort.
    assert "async def _write_dlq" in source, (
        "webhook_simple.py must have a _write_dlq function for durable DLQ storage"
    )
    assert "dlq:webhook" in source, (
        "DLQ must use Redis lpush to dlq:webhook as primary storage"
    )

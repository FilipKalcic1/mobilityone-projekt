"""
Tests for WhatsApp Service (services/whatsapp_service.py).

Covers the 3 highest-value paths:
1. Successful message dispatch (happy path)
2. Retry behavior on 503 server error
3. Phone number validation (UUID trap, invalid format, normalization)
"""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from services.whatsapp_service import WhatsAppService, SendResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def whatsapp_service():
    """Create a WhatsAppService with test config (no real Infobip calls)."""
    with patch("services.whatsapp_service.settings") as mock_settings:
        mock_settings.INFOBIP_API_KEY = "test-api-key-1234567890"
        mock_settings.INFOBIP_BASE_URL = "api.infobip.com"
        mock_settings.INFOBIP_SENDER_NUMBER = "385991234567"
        svc = WhatsAppService(
            api_key="test-api-key-1234567890",
            base_url="api.infobip.com",
            sender_number="385991234567",
        )
    return svc


def _mock_response(status_code: int, json_data: dict = None, headers: dict = None):
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = json.dumps(json_data or {})
    resp.headers = headers or {}
    return resp


# ===========================================================================
# TEST 1: Successful message dispatch
# ===========================================================================

class TestSuccessfulDispatch:
    """Verify the happy path: message sent, result contains message_id."""

    @pytest.mark.asyncio
    async def test_send_success_returns_message_id(self, whatsapp_service):
        """
        When Infobip returns 200 with a messageId,
        send() must return SendResult(success=True, message_id=...).
        """
        infobip_response = _mock_response(200, {
            "messages": [{"messageId": "abc-123-def"}]
        })

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = infobip_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await whatsapp_service.send("+385991234567", "Pozdrav!")

        assert result.success is True
        assert result.message_id == "abc-123-def"
        assert result.status_code == 200
        assert whatsapp_service._messages_sent == 1

    @pytest.mark.asyncio
    async def test_send_builds_correct_payload(self, whatsapp_service):
        """
        Verify the Infobip payload structure:
        {"from": "385...", "to": "385...", "content": {"text": "..."}}
        """
        infobip_response = _mock_response(200, {"messages": [{"messageId": "x"}]})

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = infobip_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await whatsapp_service.send("385991234567", "Test poruka")

        # Inspect the payload sent to Infobip
        call_args = mock_client.post.call_args
        sent_payload = call_args.kwargs.get("json") or call_args[1].get("json")

        assert sent_payload["from"] == "385991234567"
        assert sent_payload["to"] == "385991234567"
        assert sent_payload["content"]["text"] == "Test poruka"

    @pytest.mark.asyncio
    async def test_send_uses_correct_url_and_headers(self, whatsapp_service):
        """Verify the Infobip URL and authorization header format."""
        infobip_response = _mock_response(200, {"messages": [{"messageId": "x"}]})

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = infobip_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await whatsapp_service.send("385991234567", "Test")

        call_args = mock_client.post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")

        assert url == "https://api.infobip.com/whatsapp/1/message/text"
        assert headers["Authorization"] == "App test-api-key-1234567890"
        assert headers["Content-Type"] == "application/json"


# ===========================================================================
# TEST 2: Retry behavior on 503 server error
# ===========================================================================

class TestRetryBehavior:
    """Verify exponential backoff retry on 5xx errors."""

    @pytest.mark.asyncio
    async def test_retries_on_503_then_succeeds(self, whatsapp_service):
        """
        When Infobip returns 503 twice then 200,
        send() must retry and ultimately succeed.
        """
        fail_response = _mock_response(503, {"error": "Service Unavailable"})
        success_response = _mock_response(200, {
            "messages": [{"messageId": "retry-success-id"}]
        })

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return fail_response
            return success_response

        with patch("httpx.AsyncClient") as MockClient, \
             patch("asyncio.sleep", new_callable=AsyncMock):  # Skip real delays
            mock_client = AsyncMock()
            mock_client.post.side_effect = mock_post
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await whatsapp_service.send("385991234567", "Retry test")

        assert result.success is True
        assert result.message_id == "retry-success-id"
        assert call_count == 3  # 2 failures + 1 success
        assert whatsapp_service._total_retries == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_on_persistent_503(self, whatsapp_service):
        """
        When Infobip returns 503 for all MAX_RETRIES attempts,
        send() must return failure after exhausting retries.
        """
        fail_response = _mock_response(503, {"error": "Service Unavailable"})

        with patch("httpx.AsyncClient") as MockClient, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            mock_client = AsyncMock()
            mock_client.post.return_value = fail_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await whatsapp_service.send("385991234567", "Will fail")

        assert result.success is False
        assert result.error_code == "MAX_RETRIES_EXCEEDED"
        assert whatsapp_service._messages_failed == 1
        # Should have attempted MAX_RETRIES times
        assert mock_client.post.call_count == whatsapp_service.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_does_not_retry_on_400_client_error(self, whatsapp_service):
        """Client errors (4xx except 429) should NOT be retried."""
        error_response = _mock_response(400, {
            "requestError": {
                "serviceException": {
                    "text": "Invalid recipient"
                }
            }
        })

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = error_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await whatsapp_service.send("385991234567", "Bad request test")

        assert result.success is False
        assert result.error_code == "HTTP_400"
        assert "Invalid recipient" in result.error_message
        # Only 1 attempt - no retries for client errors
        assert mock_client.post.call_count == 1
        assert whatsapp_service._total_retries == 0


# ===========================================================================
# TEST 3: Phone number validation
# ===========================================================================

class TestPhoneValidation:
    """Verify phone validation catches bad numbers before they reach Infobip."""

    def test_valid_international_format(self, whatsapp_service):
        """Standard +385 format should normalize correctly."""
        valid, normalized, error = whatsapp_service.validate_phone_number("+385991234567")
        assert valid is True
        assert normalized == "385991234567"
        assert error is None

    def test_valid_with_double_zero_prefix(self, whatsapp_service):
        """00385 prefix should normalize to 385."""
        valid, normalized, error = whatsapp_service.validate_phone_number("00385991234567")
        assert valid is True
        assert normalized == "385991234567"
        assert error is None

    def test_valid_local_format(self, whatsapp_service):
        """Local format (0991234567) should normalize to international."""
        valid, normalized, error = whatsapp_service.validate_phone_number("0991234567")
        assert valid is True
        assert normalized == "385991234567"
        assert error is None

    def test_rejects_uuid_in_phone_field(self, whatsapp_service):
        """
        UUID TRAP: API sometimes returns person UUID instead of phone.
        Must be caught before sending to Infobip (would cause 400 error).
        """
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        valid, normalized, error = whatsapp_service.validate_phone_number(uuid)
        assert valid is False
        assert "UUID detected" in error

    def test_rejects_empty_phone(self, whatsapp_service):
        """Empty phone number should fail validation."""
        valid, normalized, error = whatsapp_service.validate_phone_number("")
        assert valid is False
        assert "empty" in error.lower()

    def test_rejects_too_short_number(self, whatsapp_service):
        """Phone numbers shorter than 10 digits should fail."""
        valid, normalized, error = whatsapp_service.validate_phone_number("38599")
        assert valid is False
        assert error is not None

    @pytest.mark.asyncio
    async def test_send_rejects_invalid_phone_without_http_call(self, whatsapp_service):
        """
        send() with invalid phone must return failure immediately
        WITHOUT making any HTTP request to Infobip.
        """
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await whatsapp_service.send(
                "550e8400-e29b-41d4-a716-446655440000",
                "This should never reach Infobip"
            )

        assert result.success is False
        assert result.error_code == "INVALID_PHONE"
        # No HTTP call was made
        mock_client.post.assert_not_called()


# ===========================================================================
# BONUS: Type guard and UTF-8 safety
# ===========================================================================

class TestTypeGuards:
    """Verify ensure_string handles AI output edge cases."""

    def test_dict_value_extracts_text_key(self, whatsapp_service):
        """AI sometimes returns {"text": "answer"} instead of string."""
        result, converted = whatsapp_service.ensure_string({"text": "Vozilo je Audi A4"})
        assert result == "Vozilo je Audi A4"
        assert converted is True

    def test_none_value_returns_empty_string(self, whatsapp_service):
        """None should return empty string, not crash."""
        result, converted = whatsapp_service.ensure_string(None)
        assert result == ""
        assert converted is True

    def test_string_value_passes_through(self, whatsapp_service):
        """Normal string should not be modified."""
        result, converted = whatsapp_service.ensure_string("Hello")
        assert result == "Hello"
        assert converted is False

    def test_utf8_safety_removes_control_characters(self, whatsapp_service):
        """Control characters should be stripped (except newline/tab)."""
        dirty = "Hello\x00World\nNew line\tTab"
        clean = whatsapp_service.ensure_utf8_safe(dirty)
        assert "\x00" not in clean
        assert "\n" in clean
        assert "\t" in clean
        assert "HelloWorld" in clean

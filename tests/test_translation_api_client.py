"""
Tests for Translation API Client.

Tests the DeepL/Google Translate integration with mocked API responses.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.registry.translation_api_client import (
    TranslationAPIClient,
    APITranslationResult,
    TranslationSource,
    translate_to_croatian,
)


class TestTranslationAPIClient:
    """Tests for TranslationAPIClient class."""

    def test_init_default(self):
        """Test default initialization."""
        client = TranslationAPIClient()
        assert client.deepl_api_key is None
        assert client.google_api_key is None
        assert client.enabled is True
        assert client.deepl_available is False
        assert client.google_available is False

    def test_init_with_keys(self):
        """Test initialization with API keys."""
        client = TranslationAPIClient(
            deepl_api_key="test-deepl-key:fx",
            google_api_key="test-google-key"
        )
        assert client.deepl_available is True
        assert client.google_available is True

    def test_deepl_url_selection_free(self):
        """Test free tier URL for DeepL."""
        client = TranslationAPIClient(deepl_api_key="test-key:fx")
        assert "api-free.deepl.com" in client.deepl_url

    def test_deepl_url_selection_pro(self):
        """Test pro tier URL for DeepL."""
        client = TranslationAPIClient(deepl_api_key="test-key-pro")
        assert "api.deepl.com" in client.deepl_url
        assert "api-free" not in client.deepl_url

    def test_cache_key_generation(self):
        """Test cache key generation."""
        client = TranslationAPIClient()
        key1 = client._cache_key("vehicle", "HR")
        key2 = client._cache_key("VEHICLE", "HR")  # Should be same (case insensitive)
        key3 = client._cache_key("vehicle", "DE")  # Different language

        assert key1 == key2  # Case insensitive
        assert key1 != key3  # Different language = different key
        assert key1.startswith("translation:")

    def test_disabled_returns_failed(self):
        """Test that disabled client returns failed result."""
        client = TranslationAPIClient(enabled=False)

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            client.translate("test")
        )

        assert result.source == TranslationSource.FAILED
        assert result.confidence == 0.3


class TestDeepLTranslation:
    """Tests for DeepL API integration."""

    @pytest.mark.asyncio
    async def test_deepl_success(self):
        """Test successful DeepL translation."""
        client = TranslationAPIClient(deepl_api_key="test-key:fx")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "translations": [{"text": "vozilo"}]
        }

        with patch.object(client, '_get_http_client') as mock_http:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_http.return_value = mock_client

            result = await client.translate_deepl("vehicle")

            assert result == "vozilo"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_deepl_quota_exceeded(self):
        """Test DeepL quota exceeded handling."""
        client = TranslationAPIClient(deepl_api_key="test-key:fx")
        assert client.deepl_available is True

        mock_response = MagicMock()
        mock_response.status_code = 456  # Quota exceeded

        with patch.object(client, '_get_http_client') as mock_http:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_http.return_value = mock_client

            result = await client.translate_deepl("vehicle")

            assert result is None
            assert client.deepl_available is False  # Disabled after quota error

    @pytest.mark.asyncio
    async def test_deepl_api_error(self):
        """Test DeepL API error handling."""
        client = TranslationAPIClient(deepl_api_key="test-key:fx")

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch.object(client, '_get_http_client') as mock_http:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_http.return_value = mock_client

            result = await client.translate_deepl("vehicle")

            assert result is None


class TestGoogleTranslation:
    """Tests for Google Translate API integration."""

    @pytest.mark.asyncio
    async def test_google_success(self):
        """Test successful Google translation."""
        client = TranslationAPIClient(google_api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "translations": [{"translatedText": "vozilo"}]
            }
        }

        with patch.object(client, '_get_http_client') as mock_http:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_http.return_value = mock_client

            result = await client.translate_google("vehicle")

            assert result == "vozilo"

    @pytest.mark.asyncio
    async def test_google_api_error(self):
        """Test Google API error handling."""
        client = TranslationAPIClient(google_api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch.object(client, '_get_http_client') as mock_http:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_http.return_value = mock_client

            result = await client.translate_google("vehicle")

            assert result is None


class TestCaching:
    """Tests for translation caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit returns cached value."""
        client = TranslationAPIClient(deepl_api_key="test-key:fx")

        with patch.object(client, '_check_cache', return_value="vozilo"):
            result = await client.translate("vehicle")

            assert result.translated == "vozilo"
            assert result.source == TranslationSource.CACHE
            assert result.cached is True
            assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_cache_miss_calls_api(self):
        """Test cache miss triggers API call."""
        client = TranslationAPIClient(deepl_api_key="test-key:fx")

        with patch.object(client, '_check_cache', return_value=None):
            with patch.object(client, 'translate_deepl', return_value="vozilo"):
                with patch.object(client, '_set_cache') as mock_set:
                    result = await client.translate("vehicle")

                    assert result.translated == "vozilo"
                    assert result.source == TranslationSource.DEEPL
                    assert result.cached is False
                    mock_set.assert_called_once()


class TestFallbackChain:
    """Tests for API fallback chain."""

    @pytest.mark.asyncio
    async def test_deepl_to_google_fallback(self):
        """Test fallback from DeepL to Google."""
        client = TranslationAPIClient(
            deepl_api_key="test-deepl:fx",
            google_api_key="test-google"
        )

        with patch.object(client, '_check_cache', return_value=None):
            with patch.object(client, 'translate_deepl', return_value=None):
                with patch.object(client, 'translate_google', return_value="vozilo"):
                    with patch.object(client, '_set_cache'):
                        result = await client.translate("vehicle")

                        assert result.translated == "vozilo"
                        assert result.source == TranslationSource.GOOGLE

    @pytest.mark.asyncio
    async def test_all_apis_fail(self):
        """Test behavior when all APIs fail."""
        client = TranslationAPIClient(
            deepl_api_key="test-deepl:fx",
            google_api_key="test-google"
        )

        with patch.object(client, '_check_cache', return_value=None):
            with patch.object(client, 'translate_deepl', return_value=None):
                with patch.object(client, 'translate_google', return_value=None):
                    result = await client.translate("vehicle")

                    assert result.translated == "vehicle"  # Original returned
                    assert result.source == TranslationSource.FAILED
                    assert result.confidence == 0.3


class TestBatchTranslation:
    """Tests for batch translation."""

    @pytest.mark.asyncio
    async def test_batch_all_cached(self):
        """Test batch with all items cached."""
        client = TranslationAPIClient(deepl_api_key="test:fx")

        async def mock_check_cache(text, lang):
            return f"{text}_hr"

        with patch.object(client, '_check_cache', side_effect=mock_check_cache):
            results = await client.translate_batch(["a", "b", "c"])

            assert len(results) == 3
            assert all(r.cached for r in results)
            assert all(r.source == TranslationSource.CACHE for r in results)

    @pytest.mark.asyncio
    async def test_batch_mixed(self):
        """Test batch with mixed cached and uncached."""
        client = TranslationAPIClient(deepl_api_key="test:fx")

        cache = {"a": "a_hr"}

        async def mock_check_cache(text, lang):
            return cache.get(text)

        async def mock_translate(text, lang):
            return APITranslationResult(
                original=text,
                translated=f"{text}_hr",
                source=TranslationSource.DEEPL,
                cached=False,
                confidence=1.0
            )

        with patch.object(client, '_check_cache', side_effect=mock_check_cache):
            with patch.object(client, 'translate', side_effect=mock_translate):
                results = await client.translate_batch(["a", "b"])

                assert len(results) == 2
                # First should be cached
                assert results[0].cached is True
                # Second should be from API
                assert results[1].cached is False


class TestAPITranslationResult:
    """Tests for APITranslationResult dataclass."""

    def test_result_structure(self):
        """Test result dataclass structure."""
        result = APITranslationResult(
            original="vehicle",
            translated="vozilo",
            source=TranslationSource.DEEPL,
            cached=False,
            confidence=1.0
        )

        assert result.original == "vehicle"
        assert result.translated == "vozilo"
        assert result.source == TranslationSource.DEEPL
        assert result.cached is False
        assert result.confidence == 1.0


class TestConvenienceFunction:
    """Tests for module-level convenience function."""

    @pytest.mark.asyncio
    async def test_translate_to_croatian(self):
        """Test convenience function."""
        with patch('services.registry.translation_api_client.TranslationAPIClient') as MockClient:
            mock_instance = MagicMock()
            mock_instance.translate = AsyncMock(return_value=APITranslationResult(
                original="vehicle",
                translated="vozilo",
                source=TranslationSource.DEEPL,
                cached=False,
                confidence=1.0
            ))
            MockClient.from_settings.return_value = mock_instance

            result = await translate_to_croatian("vehicle")
            assert result == "vozilo"


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_text(self):
        """Test empty text handling."""
        client = TranslationAPIClient(deepl_api_key="test:fx")

        result = await client.translate("")
        assert result.confidence == 0.0
        assert result.source == TranslationSource.FAILED

    @pytest.mark.asyncio
    async def test_whitespace_only(self):
        """Test whitespace-only text."""
        client = TranslationAPIClient(deepl_api_key="test:fx")

        result = await client.translate("   ")
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_single_character(self):
        """Test single character text."""
        client = TranslationAPIClient(deepl_api_key="test:fx")

        result = await client.translate("a")
        assert result.source == TranslationSource.FAILED  # Too short

    @pytest.mark.asyncio
    async def test_stats(self):
        """Test get_stats method."""
        client = TranslationAPIClient(
            deepl_api_key="test:fx",
            google_api_key="test-google",
            cache_ttl=86400 * 30
        )

        stats = await client.get_stats()

        assert stats["deepl_available"] is True
        assert stats["google_available"] is True
        assert stats["cache_ttl_days"] == 30
        assert stats["enabled"] is True

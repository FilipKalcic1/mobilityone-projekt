"""
Translation API Client - External translation service integration.
Version: 1.0

Provides high-quality Croatian translations via external APIs when
dictionary lookup fails. Uses DeepL (preferred) or Google Translate.

ARCHITECTURE:
    1. Check Redis cache first (30-day TTL)
    2. Try DeepL API (best Croatian quality)
    3. Fallback to Google Translate
    4. Return formatted English if all fail

USAGE:
    client = TranslationAPIClient()
    result = await client.translate("vehicle", target_lang="HR")
    # Returns: {"text": "vozilo", "source": "deepl", "cached": False}
"""

import asyncio
import hashlib
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import httpx

logger = logging.getLogger("translation_api")


class TranslationSource(Enum):
    """Source of the translation."""
    DEEPL = "deepl"
    GOOGLE = "google"
    CACHE = "cache"
    FAILED = "failed"


@dataclass
class APITranslationResult:
    """Result from translation API."""
    original: str
    translated: str
    source: TranslationSource
    cached: bool
    confidence: float  # 1.0 for API, 0.95 for cache, 0.3 for failed


class TranslationAPIClient:
    """
    External translation API client with caching.

    Priorities:
    1. Redis cache (instant, free)
    2. DeepL API (best quality for Croatian)
    3. Google Translate API (good fallback)
    4. Formatted English (last resort)
    """

    DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"
    DEEPL_PRO_URL = "https://api.deepl.com/v2/translate"
    GOOGLE_API_URL = "https://translation.googleapis.com/language/translate/v2"

    # Cache key prefix
    CACHE_PREFIX = "translation:"

    def __init__(
        self,
        deepl_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        cache_ttl: int = 86400 * 30,  # 30 days
        enabled: bool = True
    ):
        """
        Initialize the translation client.

        Args:
            deepl_api_key: DeepL API key (free or pro)
            google_api_key: Google Cloud Translation API key
            cache_ttl: Cache TTL in seconds
            enabled: Whether API calls are enabled
        """
        self.deepl_api_key = deepl_api_key
        self.google_api_key = google_api_key
        self.cache_ttl = cache_ttl
        self.enabled = enabled
        self._redis = None
        self._http_client: Optional[httpx.AsyncClient] = None

        # Track which APIs are available
        self.deepl_available = bool(deepl_api_key)
        self.google_available = bool(google_api_key)

        # Use DeepL Pro URL if key looks like pro key
        if deepl_api_key and not deepl_api_key.endswith(":fx"):
            self.deepl_url = self.DEEPL_PRO_URL
        else:
            self.deepl_url = self.DEEPL_API_URL

    @classmethod
    def from_settings(cls) -> "TranslationAPIClient":
        """Create client from application settings."""
        from config import get_settings
        settings = get_settings()

        return cls(
            deepl_api_key=settings.DEEPL_API_KEY,
            google_api_key=settings.GOOGLE_TRANSLATE_API_KEY,
            cache_ttl=settings.TRANSLATION_API_CACHE_TTL,
            enabled=settings.TRANSLATION_API_ENABLED
        )

    async def _get_redis(self):
        """Get Redis client lazily."""
        if self._redis is None:
            try:
                from services.cache_service import get_cache_service
                self._redis = await get_cache_service()
            except Exception as e:
                logger.warning(f"Redis unavailable for translation cache: {e}")
                self._redis = False  # Mark as unavailable
        return self._redis if self._redis else None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get HTTP client lazily."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=10.0)
        return self._http_client

    def _cache_key(self, text: str, target_lang: str) -> str:
        """Generate cache key for a translation."""
        # Use hash for long texts
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()[:12]
        return f"{self.CACHE_PREFIX}{target_lang}:{text_hash}"

    async def _check_cache(self, text: str, target_lang: str) -> Optional[str]:
        """Check if translation is cached."""
        redis = await self._get_redis()
        if not redis:
            return None

        try:
            key = self._cache_key(text, target_lang)
            cached = await redis.get(key)
            if cached:
                logger.debug(f"Cache hit for '{text[:20]}...' -> '{cached[:20]}...'")
                return cached
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")

        return None

    async def _set_cache(self, text: str, target_lang: str, translation: str) -> None:
        """Cache a translation."""
        redis = await self._get_redis()
        if not redis:
            return

        try:
            key = self._cache_key(text, target_lang)
            await redis.set(key, translation, ex=self.cache_ttl)
            logger.debug(f"Cached translation: '{text[:20]}...' -> '{translation[:20]}...'")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    async def translate_deepl(
        self,
        text: str,
        target_lang: str = "HR",
        source_lang: str = "EN"
    ) -> Optional[str]:
        """
        Translate using DeepL API.

        Args:
            text: Text to translate
            target_lang: Target language code (HR for Croatian)
            source_lang: Source language code

        Returns:
            Translated text or None if failed
        """
        if not self.deepl_api_key:
            return None

        try:
            client = await self._get_http_client()

            response = await client.post(
                self.deepl_url,
                data={
                    "auth_key": self.deepl_api_key,
                    "text": text,
                    "target_lang": target_lang,
                    "source_lang": source_lang,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if response.status_code == 200:
                data = response.json()
                translations = data.get("translations", [])
                if translations:
                    result = translations[0].get("text", "")
                    logger.info(f"DeepL translated: '{text}' -> '{result}'")
                    return result
            elif response.status_code == 456:
                logger.warning("DeepL quota exceeded")
                self.deepl_available = False
            else:
                logger.warning(f"DeepL API error: {response.status_code}")

        except httpx.TimeoutException:
            logger.warning("DeepL API timeout")
        except Exception as e:
            logger.error(f"DeepL API error: {e}")

        return None

    async def translate_google(
        self,
        text: str,
        target_lang: str = "hr",
        source_lang: str = "en"
    ) -> Optional[str]:
        """
        Translate using Google Cloud Translation API.

        Args:
            text: Text to translate
            target_lang: Target language code (hr for Croatian)
            source_lang: Source language code

        Returns:
            Translated text or None if failed
        """
        if not self.google_api_key:
            return None

        try:
            client = await self._get_http_client()

            response = await client.post(
                self.GOOGLE_API_URL,
                params={"key": self.google_api_key},
                json={
                    "q": text,
                    "target": target_lang,
                    "source": source_lang,
                    "format": "text"
                }
            )

            if response.status_code == 200:
                data = response.json()
                translations = data.get("data", {}).get("translations", [])
                if translations:
                    result = translations[0].get("translatedText", "")
                    logger.info(f"Google translated: '{text}' -> '{result}'")
                    return result
            else:
                logger.warning(f"Google API error: {response.status_code}")

        except httpx.TimeoutException:
            logger.warning("Google API timeout")
        except Exception as e:
            logger.error(f"Google API error: {e}")

        return None

    async def translate(
        self,
        text: str,
        target_lang: str = "HR"
    ) -> APITranslationResult:
        """
        Translate text to Croatian using available APIs.

        Priority:
        1. Redis cache
        2. DeepL API
        3. Google Translate API
        4. Return original (failed)

        Args:
            text: English text to translate
            target_lang: Target language (HR for Croatian)

        Returns:
            APITranslationResult with translation and metadata
        """
        text = text.strip()

        # Skip empty or very short text
        if not text or len(text) < 2:
            return APITranslationResult(
                original=text,
                translated=text,
                source=TranslationSource.FAILED,
                cached=False,
                confidence=0.0
            )

        # Check if API is enabled
        if not self.enabled:
            return APITranslationResult(
                original=text,
                translated=text,
                source=TranslationSource.FAILED,
                cached=False,
                confidence=0.3
            )

        # 1. Check cache
        cached = await self._check_cache(text, target_lang)
        if cached:
            return APITranslationResult(
                original=text,
                translated=cached,
                source=TranslationSource.CACHE,
                cached=True,
                confidence=0.95
            )

        # 2. Try DeepL
        if self.deepl_available:
            result = await self.translate_deepl(text, target_lang)
            if result:
                await self._set_cache(text, target_lang, result)
                return APITranslationResult(
                    original=text,
                    translated=result,
                    source=TranslationSource.DEEPL,
                    cached=False,
                    confidence=1.0
                )

        # 3. Try Google
        if self.google_available:
            # Google uses lowercase language codes
            google_target = target_lang.lower()
            result = await self.translate_google(text, google_target)
            if result:
                await self._set_cache(text, target_lang, result)
                return APITranslationResult(
                    original=text,
                    translated=result,
                    source=TranslationSource.GOOGLE,
                    cached=False,
                    confidence=0.95
                )

        # 4. Failed - return original
        logger.warning(f"All translation APIs failed for: '{text}'")
        return APITranslationResult(
            original=text,
            translated=text,
            source=TranslationSource.FAILED,
            cached=False,
            confidence=0.3
        )

    async def translate_batch(
        self,
        texts: List[str],
        target_lang: str = "HR"
    ) -> List[APITranslationResult]:
        """
        Translate multiple texts efficiently.

        Uses asyncio.gather for parallel translation of non-cached items.

        Args:
            texts: List of texts to translate
            target_lang: Target language

        Returns:
            List of APITranslationResult
        """
        if not texts:
            return []

        # First pass: check cache for all
        results: List[Optional[APITranslationResult]] = [None] * len(texts)
        uncached_indices: List[int] = []

        for i, text in enumerate(texts):
            cached = await self._check_cache(text, target_lang)
            if cached:
                results[i] = APITranslationResult(
                    original=text,
                    translated=cached,
                    source=TranslationSource.CACHE,
                    cached=True,
                    confidence=0.95
                )
            else:
                uncached_indices.append(i)

        # Translate uncached items in parallel
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            translations = await asyncio.gather(
                *[self.translate(text, target_lang) for text in uncached_texts]
            )
            for i, result in zip(uncached_indices, translations):
                results[i] = result

        return results  # type: ignore

    async def get_stats(self) -> Dict[str, Any]:
        """Get translation API statistics."""
        return {
            "deepl_available": self.deepl_available,
            "google_available": self.google_available,
            "cache_ttl_days": self.cache_ttl // 86400,
            "enabled": self.enabled
        }

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Convenience function for quick translation
async def translate_to_croatian(text: str) -> str:
    """
    Quick translation to Croatian.

    Args:
        text: English text to translate

    Returns:
        Croatian translation or original if failed
    """
    client = TranslationAPIClient.from_settings()
    result = await client.translate(text)
    return result.translated

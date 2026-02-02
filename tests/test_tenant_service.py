"""
Tenant Service Tests - Dynamic multi-tenant routing.

Tests the full tenant resolution chain:
1. UserMapping.tenant_id (DB) -> highest priority
2. Phone prefix rules -> phone-based routing
3. Default tenant (env) -> fallback
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.tenant_service import TenantService, TenantConfig, PHONE_PREFIX_RULES


class TestTenantResolutionFromPhone:
    """Test phone prefix -> tenant mapping."""

    @pytest.fixture
    def service(self):
        with patch("services.tenant_service.settings") as mock_settings:
            mock_settings.MOBILITY_TENANT_ID = "tenant-default"
            return TenantService()

    def test_croatian_phone_resolves_to_hr(self, service):
        assert service.resolve_tenant_from_phone("+385991234567") == "tenant-hr"

    def test_slovenian_phone_resolves_to_si(self, service):
        assert service.resolve_tenant_from_phone("+38641123456") == "tenant-si"

    def test_bosnian_phone_resolves_to_ba(self, service):
        assert service.resolve_tenant_from_phone("+38761123456") == "tenant-ba"

    def test_serbian_phone_resolves_to_rs(self, service):
        assert service.resolve_tenant_from_phone("+381641234567") == "tenant-rs"

    def test_austrian_phone_resolves_to_at(self, service):
        assert service.resolve_tenant_from_phone("+43664123456") == "tenant-at"

    def test_german_phone_resolves_to_de(self, service):
        assert service.resolve_tenant_from_phone("+491512345678") == "tenant-de"

    def test_phone_without_plus_prefix(self, service):
        """Phone numbers without + should still resolve."""
        assert service.resolve_tenant_from_phone("385991234567") == "tenant-hr"

    def test_unknown_country_code_falls_back_to_default(self, service):
        """US number should fall back to default tenant."""
        assert service.resolve_tenant_from_phone("+12025551234") == "tenant-default"

    def test_empty_phone_falls_back_to_default(self, service):
        assert service.resolve_tenant_from_phone("") == "tenant-default"

    def test_none_phone_falls_back_to_default(self, service):
        assert service.resolve_tenant_from_phone(None) == "tenant-default"

    def test_whitespace_phone_normalized(self, service):
        assert service.resolve_tenant_from_phone("  +385991234567  ") == "tenant-hr"


class TestTenantResolutionOrder:
    """Test the 3-level resolution: DB -> phone prefix -> default."""

    @pytest.fixture
    def service(self):
        with patch("services.tenant_service.settings") as mock_settings:
            mock_settings.MOBILITY_TENANT_ID = "tenant-default"
            svc = TenantService()
            return svc

    @pytest.mark.asyncio
    async def test_user_mapping_takes_priority_over_phone(self, service):
        """If user has tenant_id in DB, that overrides phone prefix."""
        user_mapping = MagicMock()
        user_mapping.tenant_id = "tenant-custom"

        result = await service.get_tenant_for_user("+385991234567", user_mapping)
        # Should return DB value, NOT "tenant-hr" from phone prefix
        assert result == "tenant-custom"

    @pytest.mark.asyncio
    async def test_phone_prefix_used_when_no_user_mapping(self, service):
        result = await service.get_tenant_for_user("+385991234567", None)
        assert result == "tenant-hr"

    @pytest.mark.asyncio
    async def test_phone_prefix_used_when_user_mapping_has_no_tenant(self, service):
        """If UserMapping exists but tenant_id is None, fall through to phone."""
        user_mapping = MagicMock()
        user_mapping.tenant_id = None

        result = await service.get_tenant_for_user("+385991234567", user_mapping)
        assert result == "tenant-hr"

    @pytest.mark.asyncio
    async def test_default_used_when_no_match(self, service):
        result = await service.get_tenant_for_user("+12025551234", None)
        assert result == "tenant-default"


class TestTenantCaching:
    """Test Redis-based tenant caching."""

    @pytest.fixture
    def service_with_redis(self):
        with patch("services.tenant_service.settings") as mock_settings:
            mock_settings.MOBILITY_TENANT_ID = "tenant-default"
            redis = AsyncMock()
            svc = TenantService(redis_client=redis)
            return svc, redis

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_value(self, service_with_redis):
        service, redis = service_with_redis
        redis.get = AsyncMock(return_value="tenant-cached")

        result = await service.get_tenant_for_user("+385991234567", None)
        assert result == "tenant-cached"

    @pytest.mark.asyncio
    async def test_cache_miss_resolves_and_stores(self, service_with_redis):
        service, redis = service_with_redis
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()

        result = await service.get_tenant_for_user("+385991234567", None)
        assert result == "tenant-hr"
        # Should have cached the result with 1h TTL
        redis.set.assert_called_once_with("tenant:+385991234567", "tenant-hr", ex=3600)

    @pytest.mark.asyncio
    async def test_redis_failure_does_not_break_resolution(self, service_with_redis):
        """If Redis is down, tenant resolution still works via phone prefix."""
        service, redis = service_with_redis
        redis.get = AsyncMock(side_effect=Exception("Redis connection lost"))
        redis.set = AsyncMock(side_effect=Exception("Redis connection lost"))

        result = await service.get_tenant_for_user("+385991234567", None)
        assert result == "tenant-hr"  # Falls through to phone prefix


class TestTenantValidation:
    """Test tenant ID format validation."""

    @pytest.fixture
    def service(self):
        with patch("services.tenant_service.settings") as mock_settings:
            mock_settings.MOBILITY_TENANT_ID = "tenant-default"
            return TenantService()

    def test_valid_tenant_id(self, service):
        assert service.validate_tenant("tenant-hr") is True
        assert service.validate_tenant("my-company-123") is True

    def test_empty_tenant_id_invalid(self, service):
        assert service.validate_tenant("") is False
        assert service.validate_tenant(None) is False

    def test_short_tenant_id_invalid(self, service):
        assert service.validate_tenant("ab") is False

    def test_special_characters_invalid(self, service):
        assert service.validate_tenant("tenant;DROP TABLE") is False
        assert service.validate_tenant("tenant<script>") is False
        assert service.validate_tenant("tenant/../etc") is False


class TestTenantConfig:
    """Test TenantConfig dataclass."""

    def test_defaults(self):
        config = TenantConfig(tenant_id="test", name="Test Tenant")
        assert config.rate_limit == 20
        assert config.is_active is True
        assert config.api_url is None

    def test_custom_values(self):
        config = TenantConfig(
            tenant_id="premium",
            name="Premium Tenant",
            rate_limit=100,
            api_url="https://custom-api.example.com"
        )
        assert config.rate_limit == 100
        assert config.api_url == "https://custom-api.example.com"


class TestPhonePrefixRules:
    """Verify the phone prefix rule configuration."""

    def test_all_expected_countries_have_rules(self):
        expected_prefixes = [r"^\+?385", r"^\+?386", r"^\+?387", r"^\+?381", r"^\+?43", r"^\+?49"]
        for prefix in expected_prefixes:
            assert prefix in PHONE_PREFIX_RULES, f"Missing rule for {prefix}"

    def test_rules_map_to_valid_tenant_ids(self):
        for pattern, tenant_id in PHONE_PREFIX_RULES.items():
            assert tenant_id.startswith("tenant-"), f"Tenant ID {tenant_id} doesn't follow naming convention"

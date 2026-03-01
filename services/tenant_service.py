"""
Tenant Service - Dynamic multi-tenant routing.

Manages tenant identification and routing for multi-tenant deployments.

ARCHITECTURE:
- Each user (phone number) is associated with a tenant
- Tenant determines which API instance/data partition to use
- Default tenant from ENV is used for new users
- Admin can reassign users to different tenants

TENANT RESOLUTION ORDER:
1. UserMapping.tenant_id (if user exists in DB)
2. Phone prefix rules (configurable)
3. Default tenant from settings (fallback)

KUBERNETES MULTI-TENANT:
- All tenants share same database (filtered by tenant_id)
- Each tenant can have different API endpoints (future)
- Cost tracking and rate limiting per tenant
"""

import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class TenantConfig:
    """Configuration for a single tenant."""
    tenant_id: str
    name: str
    api_url: Optional[str] = None  # Override API URL per tenant (future)
    rate_limit: int = 20  # Requests per minute
    is_active: bool = True


# Phone prefix to tenant mapping
# Format: regex pattern -> tenant_id
# This can be moved to database/config file for production
PHONE_PREFIX_RULES: Dict[str, str] = {
    r"^\+?385": "tenant-hr",      # Croatia
    r"^\+?386": "tenant-si",      # Slovenia
    r"^\+?387": "tenant-ba",      # Bosnia
    r"^\+?381": "tenant-rs",      # Serbia
    r"^\+?43": "tenant-at",       # Austria
    r"^\+?49": "tenant-de",       # Germany
}


class TenantService:
    """
    Manages tenant identification and routing.

    Features:
    - Phone-based tenant resolution
    - Configurable prefix rules
    - Default tenant fallback
    - Tenant validation
    """

    def __init__(self, db_session=None, redis_client=None):
        """
        Initialize TenantService.

        Args:
            db_session: Optional database session for UserMapping lookup
            redis_client: Optional Redis for caching tenant lookups
        """
        self.db = db_session
        self.redis = redis_client
        self.default_tenant = settings.MOBILITY_TENANT_ID

        # Compile regex patterns once
        self._compiled_rules = [
            (re.compile(pattern), tenant_id)
            for pattern, tenant_id in PHONE_PREFIX_RULES.items()
        ]

        logger.info(f"TenantService initialized: default={self.default_tenant}, rules={len(self._compiled_rules)}")

    def resolve_tenant_from_phone(self, phone: str) -> str:
        """
        Resolve tenant ID from phone number using prefix rules.

        Args:
            phone: Phone number (with or without +)

        Returns:
            Tenant ID or default tenant
        """
        if not phone:
            return self.default_tenant

        # Normalize phone number
        normalized = phone.strip()
        if not normalized.startswith("+"):
            normalized = "+" + normalized

        # Check prefix rules
        for pattern, tenant_id in self._compiled_rules:
            if pattern.match(normalized):
                logger.debug(f"Phone {phone[-4:]}... matched rule -> {tenant_id}")
                return tenant_id

        # No rule matched - use default
        logger.debug(f"Phone {phone[-4:]}... no rule match -> default {self.default_tenant}")
        return self.default_tenant

    async def get_tenant_for_user(
        self,
        phone: str,
        user_mapping: Any = None
    ) -> str:
        """
        Get tenant ID for a user.

        Resolution order:
        1. UserMapping.tenant_id (if provided and set)
        2. Phone prefix rules
        3. Default tenant

        Args:
            phone: User phone number
            user_mapping: Optional UserMapping model instance

        Returns:
            Tenant ID
        """
        # 1. Check UserMapping first (highest priority)
        if user_mapping and hasattr(user_mapping, 'tenant_id') and user_mapping.tenant_id:
            logger.debug(f"Using tenant from UserMapping: {user_mapping.tenant_id}")
            return user_mapping.tenant_id

        # 2. Check Redis cache
        if self.redis:
            cache_key = f"tenant:{phone}"
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    logger.debug(f"Using cached tenant for {phone[-4:]}...: {cached}")
                    return cached
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")

        # 3. Resolve from phone prefix
        tenant_id = self.resolve_tenant_from_phone(phone)

        # 4. Cache the result
        if self.redis:
            try:
                await self.redis.set(f"tenant:{phone}", tenant_id, ex=3600)  # 1 hour TTL
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")

        return tenant_id

    async def update_user_tenant(
        self,
        phone: str,
        new_tenant_id: str,
        admin_id: str = None
    ) -> bool:
        """
        Update tenant for a user (admin operation).

        Args:
            phone: User phone number
            new_tenant_id: New tenant ID
            admin_id: Admin performing the change (for audit)

        Returns:
            True if updated successfully
        """
        if not self.db:
            logger.error("Cannot update tenant: no database session")
            return False

        try:
            from sqlalchemy import update
            from models import UserMapping

            result = await self.db.execute(
                update(UserMapping)
                .where(UserMapping.phone_number == phone)
                .values(tenant_id=new_tenant_id)
            )
            await self.db.commit()

            # Invalidate cache
            if self.redis:
                await self.redis.delete(f"tenant:{phone}")

            logger.info(f"Updated tenant for {phone[-4:]}... to {new_tenant_id} (by {admin_id})")
            return result.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to update tenant: {e}")
            await self.db.rollback()
            return False

    def validate_tenant(self, tenant_id: str) -> bool:
        """
        Validate tenant ID format.

        Args:
            tenant_id: Tenant ID to validate

        Returns:
            True if valid format
        """
        if not tenant_id:
            return False

        # Basic validation - alphanumeric with hyphens, 3-50 chars
        if not re.match(r'^[a-zA-Z0-9\-]{3,50}$', tenant_id):
            return False

        return True

    def get_tenant_stats(self) -> Dict[str, Any]:
        """
        Get tenant service statistics.

        Returns:
            Stats dictionary
        """
        return {
            "default_tenant": self.default_tenant,
            "prefix_rules_count": len(self._compiled_rules),
            "rules": list(PHONE_PREFIX_RULES.keys())
        }


# Singleton instance
_tenant_service: Optional[TenantService] = None


def get_tenant_service(db_session=None, redis_client=None) -> TenantService:
    """
    Get or create TenantService singleton.

    Args:
        db_session: Optional database session
        redis_client: Optional Redis client

    Returns:
        TenantService instance
    """
    global _tenant_service

    if _tenant_service is None:
        _tenant_service = TenantService(db_session, redis_client)
    elif db_session and not _tenant_service.db:
        _tenant_service.db = db_session
    elif redis_client and not _tenant_service.redis:
        _tenant_service.redis = redis_client

    return _tenant_service

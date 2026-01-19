"""
Cache Service
Version: 11.0

Redis caching layer with resilience.
NO DEPENDENCIES on other services.

Phase 4 (v11.0):
- Custom JSON serializer for datetime/UUID
- set_json() with explicit JSON handling
- invalidate() method for cache busting
- Fail-Open design (returns None/False, never crashes)
"""

import json
import logging
from datetime import datetime, date
from typing import Any, Optional, Callable, Awaitable, List
from uuid import UUID

logger = logging.getLogger(__name__)


class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for datetime and UUID objects.

    Handles:
    - datetime -> ISO format string
    - date -> ISO format string
    - UUID -> string
    - Other objects -> str() fallback
    """

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        # Fallback for unknown types
        try:
            return str(obj)
        except Exception:
            return super().default(obj)


class CacheService:
    """Redis cache wrapper."""
    
    def __init__(self, redis_client):
        """
        Initialize cache service.
        
        Args:
            redis_client: Redis async client
        """
        self.redis = redis_client
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        try:
            return await self.redis.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value from cache."""
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Cache get_json failed: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """
        Set value with TTL.

        Args:
            key: Cache key
            value: Value to store (will be JSON encoded if not string)
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value, cls=SafeJSONEncoder)
            await self.redis.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False

    async def set_json(self, key: str, value: Any, ttl: int = 300) -> bool:
        """
        Set JSON value with TTL and safe serialization.

        Uses SafeJSONEncoder to handle datetime/UUID objects.

        Args:
            key: Cache key
            value: Value to store (will be JSON encoded)
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            serialized = json.dumps(value, cls=SafeJSONEncoder)
            await self.redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set_json failed: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed: {e}")
            return False

    async def invalidate(self, key: str) -> bool:
        """
        Invalidate cache key.

        Alias for delete() - semantic naming for cache busting.

        Args:
            key: Cache key to invalidate

        Returns:
            True if successful
        """
        return await self.delete(key)

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        CAUTION: Use sparingly - SCAN can be slow on large datasets.

        Args:
            pattern: Redis key pattern (e.g., "user:*", "context:*")

        Returns:
            Number of keys deleted
        """
        try:
            deleted = 0
            async for key in self.redis.scan_iter(match=pattern, count=100):
                await self.redis.delete(key)
                deleted += 1
            if deleted > 0:
                logger.info(f"Invalidated {deleted} keys matching '{pattern}'")
            return deleted
        except Exception as e:
            logger.warning(f"Cache invalidate_pattern failed: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return await self.redis.exists(key) > 0
        except Exception:
            return False
    
    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Awaitable[Any]],
        ttl: int = 300
    ) -> Any:
        """
        Get from cache or compute value.
        
        Args:
            key: Cache key
            compute_fn: Async function to compute value if not cached
            ttl: Time to live
            
        Returns:
            Cached or computed value
        """
        # Try cache first
        cached = await self.get_json(key)
        if cached is not None:
            return cached
        
        # Compute value
        result = await compute_fn()
        
        # Cache result
        if result is not None:
            await self.set(key, result, ttl)
        
        return result
    
    async def increment(self, key: str, ttl: int = None) -> int:
        """
        Increment counter.
        
        Args:
            key: Counter key
            ttl: Optional TTL for first increment
            
        Returns:
            New value
        """
        try:
            value = await self.redis.incr(key)
            if ttl and value == 1:
                await self.redis.expire(key, ttl)
            return value
        except Exception as e:
            logger.warning(f"Cache increment failed: {e}")
            return 0

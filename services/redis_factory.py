"""
Redis client factory.

Usage:
    from services.redis_factory import create_redis_client
"""

import redis.asyncio as aioredis


async def create_redis_client(settings) -> aioredis.Redis:
    """Create and verify an async Redis connection."""
    client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=10,  # API pod: healthcheck + registry + context — 10 is sufficient at 0.5 CPU
        socket_keepalive=True,
        health_check_interval=30,
    )
    await client.ping()
    return client

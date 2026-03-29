"""Shared retry and distributed lock utilities."""
import random

# Lua script for owner-safe lock release (compare-and-delete).
# Used by cache_service, rag_scheduler, and worker for distributed locks.
# Only deletes the key if the caller still holds it (value matches token).
ATOMIC_LOCK_RELEASE_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) else return 0 end"
)


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.5,
) -> float:
    """Calculate exponential backoff with jitter.

    Args:
        attempt: Zero-based attempt number
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        jitter: Maximum random jitter to add (0 to disable)
    """
    delay = min(max_delay, base_delay * (2 ** attempt))
    if jitter > 0:
        delay += random.uniform(0, jitter)
    return delay

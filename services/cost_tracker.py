"""
Cost Tracker Service
Version: 2.0 - Minimal & Clean

Tracks LLM token usage and costs in Redis.
Single model pricing (gpt-4o-mini) from config.
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Pricing from env or defaults (per 1K tokens)
INPUT_PRICE = float(os.getenv("LLM_INPUT_PRICE_PER_1K", "0.00015"))
OUTPUT_PRICE = float(os.getenv("LLM_OUTPUT_PRICE_PER_1K", "0.0006"))
DAILY_BUDGET = float(os.getenv("DAILY_COST_BUDGET_USD", "50.0"))


@dataclass
class DailyStats:
    """Daily statistics."""
    date: str
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class CostTracker:
    """Track LLM usage costs in Redis."""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.daily_budget = DAILY_BUDGET
        self._session = {"prompt": 0, "completion": 0, "cost": 0.0, "requests": 0}
        self._session_start = datetime.now(timezone.utc)

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD."""
        return (prompt_tokens * INPUT_PRICE + completion_tokens * OUTPUT_PRICE) / 1000

    async def record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        tenant_id: str = "default"
    ) -> float:
        """Record token usage. Returns cost in USD."""
        cost = self._calculate_cost(prompt_tokens, completion_tokens)

        # Update session
        self._session["prompt"] += prompt_tokens
        self._session["completion"] += completion_tokens
        self._session["cost"] += cost
        self._session["requests"] += 1

        # Persist to Redis
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"cost:daily:{today}"
        cost_microcents = int(cost * 100_000_000)

        try:
            pipe = self.redis.pipeline()
            pipe.hincrby(key, "requests", 1)
            pipe.hincrby(key, "prompt_tokens", prompt_tokens)
            pipe.hincrby(key, "completion_tokens", completion_tokens)
            pipe.hincrby(key, "cost_microcents", cost_microcents)
            pipe.hincrby(key, f"tenant:{tenant_id}", prompt_tokens + completion_tokens)
            pipe.expire(key, 90 * 86400)  # 90 days TTL
            await pipe.execute()
        except Exception as e:
            logger.warning(f"Failed to persist cost: {e}")

        return cost

    async def get_daily_stats(self, date: str = None) -> DailyStats:
        """Get daily statistics."""
        date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"cost:daily:{date}"

        try:
            data = await self.redis.hgetall(key)
            if not data:
                return DailyStats(date=date)

            def get_int(k):
                v = data.get(k) or data.get(k.encode()) or 0
                return int(v.decode() if isinstance(v, bytes) else v)

            return DailyStats(
                date=date,
                requests=get_int("requests"),
                prompt_tokens=get_int("prompt_tokens"),
                completion_tokens=get_int("completion_tokens"),
                cost_usd=get_int("cost_microcents") / 100_000_000
            )
        except Exception as e:
            logger.error(f"Failed to get daily stats: {e}")
            return DailyStats(date=date)

    async def get_total_stats(self) -> Dict:
        """Get all-time statistics from Redis."""
        total = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        for i in range(90):
            date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            stats = await self.get_daily_stats(date)
            total["prompt_tokens"] += stats.prompt_tokens
            total["completion_tokens"] += stats.completion_tokens
            total["cost_usd"] += stats.cost_usd
        total["total_tokens"] = total["prompt_tokens"] + total["completion_tokens"]
        return total

    async def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        return {
            "session_prompt_tokens": self._session["prompt"],
            "session_completion_tokens": self._session["completion"],
            "session_total_tokens": self._session["prompt"] + self._session["completion"],
            "session_cost_usd": self._session["cost"],
            "session_requests": self._session["requests"],
            "session_duration_seconds": (datetime.now(timezone.utc) - self._session_start).total_seconds()
        }


# Singleton
_cost_tracker: Optional[CostTracker] = None


async def get_cost_tracker(redis_client) -> CostTracker:
    """Get or create cost tracker singleton."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker(redis_client)
    return _cost_tracker

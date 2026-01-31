"""
Admin API - DATABASE-BACKED, Physically Isolated from Bot API
Version: 2.3 - Enterprise Security + Database + Drift Detection + Cost Tracking

SIGURNOSNA ARHITEKTURA:
- Ovaj API radi na ZASEBNOM PORTU (8080)
- Bot API radi na portu 8000
- Nikad ne miješati ova dva API-ja!
- Admin koristi admin_user PostgreSQL korisnika (puni pristup)
- Bot koristi bot_user (ograničen pristup - ne vidi admin tablice!)

DATABASE ARCHITECTURE:
- All data comes from PostgreSQL (hallucination_reports, audit_logs)
- No more file-based storage - enterprise-grade reliability
- Connection pooling configured in database.py

Produkcijska konfiguracija:
- Port: 8080 (ili admin.mobilityone.io subdomena)
- Network: Intranet/VPN only
- Auth: X-Admin-Token header

Pokretanje:
    uvicorn admin_api:app --port 8080 --host 0.0.0.0
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

# Load main .env (contains all config including admin tokens)
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# CRITICAL SECURITY: Set admin service type BEFORE importing database
# This ensures database.py uses admin connection string with full privileges
os.environ["SERVICE_TYPE"] = "admin"

from database import AsyncSessionLocal, engine
from services.admin_review import AdminReviewService, SecurityError
from services.model_drift_detector import get_drift_detector
from services.cost_tracker import CostTracker
from services.conflict_resolver import ConflictResolver

logger = logging.getLogger(__name__)

# Prometheus metrics for Admin API
ADMIN_REQUESTS = Counter(
    'admin_requests_total',
    'Total admin API requests',
    ['endpoint', 'status']
)
HALLUCINATIONS_PENDING = Gauge('hallucinations_pending', 'Pending hallucination reviews')

# =============================================================================
# LIFESPAN (Database initialization)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize database and Redis connections."""
    logger.info("=" * 60)
    logger.info("ADMIN API STARTING (v2.1 - Database + Drift Detection)")
    logger.info("Port: 8080 (ISOLATED from Bot API on 8000)")
    logger.info("Auth: X-Admin-Token header required")
    logger.info("Rate Limit: 30 requests/minute per admin")
    logger.info("=" * 60)

    # Verify database connection
    try:
        async with AsyncSessionLocal() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        logger.info("Database connection verified (admin_user)")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise RuntimeError("Cannot start without database")

    # Initialize Redis for drift detector
    redis_url = os.environ["REDIS_URL"]  # REQUIRED - no default
    try:
        app.state.redis = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        await app.state.redis.ping()
        logger.info("Redis connection established")

        # Initialize drift detector singleton with Redis
        app.state.drift_detector = get_drift_detector(
            redis_client=app.state.redis,
            db_session=None  # Admin API only reads, doesn't need DB session
        )
        logger.info("Drift detector initialized")

        # Initialize cost tracker
        app.state.cost_tracker = CostTracker(redis_client=app.state.redis)
        logger.info("Cost tracker initialized")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e} - drift detection will use in-memory only")
        app.state.redis = None
        app.state.drift_detector = get_drift_detector()
        app.state.cost_tracker = None

    yield

    # Cleanup
    logger.info("Admin API shutting down")
    if app.state.redis:
        await app.state.redis.aclose()
    await engine.dispose()


# =============================================================================
# APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="MobilityOne Admin API",
    description="Internal API for reviewing hallucinations and error patterns. NOT for bot use!",
    version="2.2.0",
    docs_url="/admin/docs",
    redoc_url="/admin/redoc",
    openapi_url="/admin/openapi.json",
    lifespan=lifespan
)

# CORS - Restrict to internal domains only
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ADMIN_CORS_ORIGINS", "https://admin.mobilityone.io").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-Admin-Token", "Content-Type"],
)

# =============================================================================
# SECURITY
# =============================================================================

admin_api_key = APIKeyHeader(name="X-Admin-Token", auto_error=True)

# Valid admin tokens - MUST be set via environment variables
# FIX v11.1: Removed hardcoded dev tokens - admin API won't work without real tokens
VALID_ADMIN_TOKENS = {}
for i in range(1, 5):  # Support up to 4 admin tokens
    token = os.environ.get(f"ADMIN_TOKEN_{i}")
    user = os.environ.get(f"ADMIN_TOKEN_{i}_USER")
    if token and user:
        VALID_ADMIN_TOKENS[token] = user

if not VALID_ADMIN_TOKENS:
    logger.warning("ADMIN API: No admin tokens configured! Set ADMIN_TOKEN_1 + ADMIN_TOKEN_1_USER env vars.")


async def verify_admin_token(token: str = Security(admin_api_key)) -> str:
    """Verify admin token and return admin_id."""
    admin_id = VALID_ADMIN_TOKENS.get(token)
    if not admin_id:
        logger.warning(f"Invalid admin token attempt: {token[:8]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired admin token"
        )
    return admin_id


# =============================================================================
# RATE LIMITING
# =============================================================================

class RedisRateLimiter:
    """Redis-based rate limiter for production."""

    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.redis: Optional[aioredis.Redis] = None
        self._fallback: Dict[str, List[float]] = defaultdict(list)
        self._init_redis()

    def _init_redis(self):
        redis_url = os.environ["REDIS_URL"]  # REQUIRED - no default
        try:
            self.redis = aioredis.from_url(redis_url, decode_responses=True)
            logging.info(f"Rate limiter connected to Redis: {redis_url}")
        except Exception as e:
            logging.warning(f"Redis unavailable for rate limiting, using in-memory: {e}")
            self.redis = None

    async def is_allowed(self, key: str) -> bool:
        if self.redis:
            return await self._redis_check(key)
        return self._memory_check(key)

    async def _redis_check(self, key: str) -> bool:
        try:
            now = time.time()
            window_start = now - 60
            redis_key = f"admin_rate_limit:{key}"

            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(redis_key, 0, window_start)
            pipe.zcard(redis_key)
            pipe.zadd(redis_key, {str(now): now})
            pipe.expire(redis_key, 120)

            results = await pipe.execute()
            current_count = results[1]

            return current_count < self.requests_per_minute
        except Exception as e:
            logging.warning(f"Redis rate limit error: {e}")
            return self._memory_check(key)

    def _memory_check(self, key: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        self._fallback[key] = [ts for ts in self._fallback[key] if ts > minute_ago]
        if len(self._fallback[key]) >= self.requests_per_minute:
            return False
        self._fallback[key].append(now)
        return True

    async def get_remaining(self, key: str) -> int:
        if self.redis:
            try:
                redis_key = f"admin_rate_limit:{key}"
                now = time.time()
                await self.redis.zremrangebyscore(redis_key, 0, now - 60)
                count = await self.redis.zcard(redis_key)
                return max(0, self.requests_per_minute - count)
            except Exception:
                pass
        now = time.time()
        count = sum(1 for ts in self._fallback[key] if ts > now - 60)
        return max(0, self.requests_per_minute - count)


rate_limiter = RedisRateLimiter(
    requests_per_minute=int(os.getenv("ADMIN_RATE_LIMIT_PER_MINUTE", "30"))
)


async def check_rate_limit(
    request: Request,
    admin_id: str = Depends(verify_admin_token)
) -> str:
    """Rate limit middleware."""
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"{admin_id}:{client_ip}"

    if not await rate_limiter.is_allowed(rate_key):
        remaining = await rate_limiter.get_remaining(rate_key)
        logger.warning(f"Rate limit exceeded for {admin_id} from {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 30 requests per minute.",
            headers={"X-RateLimit-Remaining": str(remaining)}
        )

    return admin_id


# =============================================================================
# DATABASE DEPENDENCY
# =============================================================================

async def get_db() -> AsyncSession:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_admin_service(
    db: AsyncSession = Depends(get_db)
) -> AdminReviewService:
    """Get AdminReviewService with database session."""
    return AdminReviewService(db)


# =============================================================================
# SCHEMAS
# =============================================================================

class ReviewUpdateSchema(BaseModel):
    """Schema for marking hallucination as reviewed."""
    correction: Optional[str] = Field(
        None,
        max_length=2000,
        description="Correct answer (what the bot should have said)"
    )
    category: Optional[str] = Field(
        None,
        description="Category of the error"
    )

    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        allowed = [
            "wrong_data", "outdated", "misunderstood",
            "api_error", "rag_failure", "hallucination", "user_error"
        ]
        if v and v not in allowed:
            raise ValueError(f"Category must be one of: {allowed}")
        return v

    @field_validator('correction')
    @classmethod
    def sanitize_correction(cls, v):
        if v:
            import re
            dangerous = ['<script', 'javascript:', 'onclick', 'onerror']
            for pattern in dangerous:
                if pattern.lower() in v.lower():
                    raise ValueError("Correction contains prohibited content")
        return v


class HallucinationResponse(BaseModel):
    """Response schema for hallucination report."""
    timestamp: str
    user_query: str
    bot_response: str
    user_feedback: str
    model: str
    reviewed: bool
    category: Optional[str]
    correction: Optional[str]


class StatisticsResponse(BaseModel):
    """Response schema for statistics."""
    total_errors: int
    corrected_errors: int
    hallucinations_reported: int
    hallucinations_pending_review: int
    false_positives_skipped: int
    category_breakdown: Dict[str, int]


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint (no auth required)."""
    # Verify database
    db_status = "disconnected"
    try:
        async with AsyncSessionLocal() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.error(f"Health check DB error: {e}")

    # Check Redis
    redis_status = "disconnected"
    if hasattr(request.app.state, 'redis') and request.app.state.redis:
        try:
            await request.app.state.redis.ping()
            redis_status = "connected"
        except Exception as e:
            logger.error(f"Health check Redis error: {e}")

    # Check drift detector
    drift_status = "not_initialized"
    if hasattr(request.app.state, 'drift_detector') and request.app.state.drift_detector:
        drift_status = "ready"

    all_healthy = db_status == "connected"  # Redis is optional

    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "service": "admin-api",
        "version": "2.1.0",
        "database": db_status,
        "redis": redis_status,
        "drift_detector": drift_status,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint (no auth required for scraping)."""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get(
    "/admin/hallucinations",
    response_model=List[Dict[str, Any]],
    summary="List hallucination reports",
    description="Get list of hallucinations reported by users ('krivo' feedback)"
)
async def list_hallucinations(
    limit: int = 50,
    unreviewed_only: bool = True,
    tenant_filter: Optional[str] = None,
    admin_id: str = Depends(check_rate_limit),
    service: AdminReviewService = Depends(get_admin_service)
):
    """Dohvaća halucinacije za admin pregled IZ BAZE PODATAKA."""
    try:
        reports = await service.get_hallucinations_for_review(
            admin_id=admin_id,
            limit=limit,
            unreviewed_only=unreviewed_only,
            tenant_filter=tenant_filter
        )
        return reports
    except Exception as e:
        logger.error(f"Error listing hallucinations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/admin/hallucinations/{report_id}",
    summary="Get hallucination detail",
    description="Get full details of a specific hallucination report"
)
async def get_hallucination_detail(
    report_id: str,
    admin_id: str = Depends(check_rate_limit),
    service: AdminReviewService = Depends(get_admin_service)
):
    """Dohvaća detalje pojedine halucinacije uključujući raw API response."""
    try:
        detail = await service.get_report_detail(
            admin_id=admin_id,
            report_id=report_id
        )
        if not detail:
            raise HTTPException(status_code=404, detail="Report not found")
        return detail
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting hallucination detail: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(
    "/admin/hallucinations/{report_id}/lock",
    summary="Acquire edit lock on hallucination",
    description="Lock a report for editing (prevents concurrent modifications)"
)
async def acquire_lock(
    report_id: str,
    request: Request,
    force: bool = False,
    admin_id: str = Depends(check_rate_limit),
    service: AdminReviewService = Depends(get_admin_service)
):
    """
    Dohvati edit lock na report.

    Optimistic locking:
    - Vraća version broj koji MORAŠ poslati pri review
    - Ako drugi admin editira, dobiješ conflict
    """
    try:
        # Get Redis for conflict resolver
        redis = request.app.state.redis
        if not redis:
            # Fallback: no locking, just return version 1
            return {
                "lock_acquired": True,
                "version": 1,
                "warning": "Redis not available, locking disabled"
            }

        async with AsyncSessionLocal() as session:
            resolver = ConflictResolver(db_session=session, redis_client=redis)
            new_lock, existing_lock = await resolver.acquire_edit_lock(
                record_id=report_id,
                admin_id=admin_id,
                force=force
            )

            if existing_lock and existing_lock.admin_id != admin_id:
                return {
                    "lock_acquired": False,
                    "locked_by": existing_lock.admin_id,
                    "locked_at": existing_lock.locked_at,
                    "expires_at": existing_lock.expires_at,
                    "version": existing_lock.version,
                    "message": f"Report is being edited by {existing_lock.admin_id}"
                }

            return {
                "lock_acquired": True,
                "version": new_lock.version,
                "expires_at": new_lock.expires_at,
                "admin_id": admin_id
            }

    except Exception as e:
        logger.error(f"Error acquiring lock: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(
    "/admin/hallucinations/{report_id}/review",
    summary="Mark hallucination as reviewed",
    description="Admin reviews and optionally corrects a hallucination report"
)
async def review_hallucination(
    report_id: str,
    update: ReviewUpdateSchema,
    request: Request,
    expected_version: int = None,
    admin_id: str = Depends(check_rate_limit),
    service: AdminReviewService = Depends(get_admin_service)
):
    """
    Admin ispravlja i zatvara report U BAZI PODATAKA.

    Conflict Detection:
    - Ako posaljes expected_version, provjerava se conflict
    - Ako je drugi admin izmijenio report, dobis 409 Conflict
    """
    client_ip = request.client.host if request.client else "unknown"
    redis = request.app.state.redis

    try:
        # Optional: conflict check if version provided and Redis available
        if expected_version is not None and redis:
            async with AsyncSessionLocal() as session:
                resolver = ConflictResolver(db_session=session, redis_client=redis)

                changes = {}
                if update.correction:
                    changes["correction"] = update.correction
                if update.category:
                    changes["category"] = update.category
                changes["reviewed"] = True

                save_result = await resolver.save_with_conflict_check(
                    record_id=report_id,
                    changes=changes,
                    expected_version=expected_version,
                    admin_id=admin_id,
                    ip_address=client_ip
                )

                if save_result.has_conflict:
                    return {
                        "success": False,
                        "conflict": True,
                        "conflict_type": save_result.conflict.conflict_type,
                        "their_admin_id": save_result.conflict.their_admin_id,
                        "their_changes": save_result.conflict.their_changes,
                        "suggested_resolution": save_result.conflict.suggested_resolution,
                        "can_auto_merge": save_result.conflict.can_auto_merge
                    }

                return {
                    "success": True,
                    "new_version": save_result.new_version,
                    "report_id": report_id
                }

        # Standard flow (no conflict checking)
        result = await service.mark_hallucination_reviewed(
            admin_id=admin_id,
            report_id=report_id,
            correction=update.correction,
            category=update.category,
            ip_address=client_ip
        )
        return result
    except SecurityError as e:
        logger.warning(f"Security violation by {admin_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error reviewing hallucination: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/admin/statistics",
    response_model=StatisticsResponse,
    summary="Get error learning statistics",
    description="Dashboard statistics for monitoring bot performance"
)
async def get_statistics(
    admin_id: str = Depends(check_rate_limit),
    service: AdminReviewService = Depends(get_admin_service)
):
    """Dohvaća statistike IZ BAZE PODATAKA."""
    try:
        stats = await service.get_statistics(admin_id=admin_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/admin/audit-log",
    summary="Get audit log",
    description="View admin actions audit trail"
)
async def get_audit_log(
    limit: int = 100,
    admin_id: str = Depends(check_rate_limit),
    service: AdminReviewService = Depends(get_admin_service)
):
    """Dohvaća audit log IZ BAZE PODATAKA."""
    try:
        audit = await service.get_audit_log(admin_id=admin_id, limit=limit)
        return {"entries": audit, "count": len(audit)}
    except Exception as e:
        logger.error(f"Error getting audit log: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/admin/export/training-data",
    summary="Export training data",
    description="Export reviewed hallucinations for model fine-tuning"
)
async def export_training_data(
    reviewed_only: bool = True,
    format: str = "openai_chat",
    admin_id: str = Depends(check_rate_limit),
    service: AdminReviewService = Depends(get_admin_service)
):
    """
    Exportaj podatke za fine-tuning modela IZ BAZE PODATAKA.

    Formats:
    - openai_chat: OpenAI Chat format (default) - za gpt-3.5-turbo, gpt-4 fine-tuning
    - openai_completion: OpenAI Completion format - za davinci (legacy)
    - raw: Raw format s svim podacima
    """
    try:
        # Validate format
        valid_formats = ["openai_chat", "openai_completion", "raw"]
        if format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Must be one of: {valid_formats}"
            )

        data = await service.export_for_training(
            admin_id=admin_id,
            reviewed_only=reviewed_only,
            format=format
        )
        return {
            "count": len(data),
            "format": format,
            "data": data,
            "exported_at": datetime.utcnow().isoformat(),
            "exported_by": admin_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting training data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/admin/export/training-data.jsonl",
    summary="Download training data as JSONL file",
    description="Download reviewed hallucinations as JSONL file ready for OpenAI fine-tuning upload"
)
async def download_training_data_jsonl(
    reviewed_only: bool = True,
    admin_id: str = Depends(check_rate_limit),
    service: AdminReviewService = Depends(get_admin_service)
):
    """
    Download training data as JSONL file.

    This file can be directly uploaded to OpenAI for fine-tuning:
    openai api fine_tunes.create -t training_data.jsonl -m gpt-3.5-turbo

    Or via Python SDK:
    openai.File.create(file=open("training_data.jsonl", "rb"), purpose="fine-tune")
    """
    from fastapi.responses import Response

    try:
        jsonl_content = await service.export_for_training_jsonl(
            admin_id=admin_id,
            reviewed_only=reviewed_only
        )

        # Return as downloadable file
        filename = f"training_data_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        return Response(
            content=jsonl_content,
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        logger.error(f"Error downloading training data JSONL: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/admin/cost-stats",
    summary="Get LLM cost statistics",
    description="View token usage and estimated costs"
)
async def get_cost_stats(
    request: Request,
    date: str = None,
    admin_id: str = Depends(check_rate_limit)
):
    """
    Dohvati statistiku troškova LLM-a.

    Returns:
        Daily stats uključujući:
        - Total tokens (prompt + completion)
        - Estimated cost in USD
        - Breakdown by model and tenant
    """
    try:
        tracker = request.app.state.cost_tracker
        if not tracker:
            raise HTTPException(
                status_code=503,
                detail="Cost tracker not initialized (Redis required)"
            )

        # Get daily stats
        daily = await tracker.get_daily_stats(date)
        total = await tracker.get_total_stats()
        session = await tracker.get_session_stats()

        return {
            "daily": {
                "date": daily.date,
                "requests": daily.requests,
                "prompt_tokens": daily.prompt_tokens,
                "completion_tokens": daily.completion_tokens,
                "total_tokens": daily.prompt_tokens + daily.completion_tokens,
                "cost_usd": round(daily.cost_usd, 4)
            },
            "total": total,
            "session": session,
            "budget": {
                "daily_limit_usd": tracker.daily_budget,
                "remaining_usd": round(tracker.daily_budget - daily.cost_usd, 2)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cost stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/admin/drift-status",
    summary="Check model drift status",
    description="Analyze model performance for drift detection"
)
async def get_drift_status(
    request: Request,
    force_refresh: bool = False,
    admin_id: str = Depends(check_rate_limit)
):
    """
    Provjeri status model drifta.

    Model drift = promjena u ponašanju AI modela tijekom vremena.
    Ovo uključuje:
    - Error rate spike
    - Latency degradation
    - Hallucination rate increase
    - Confidence score drop

    Returns:
        DriftReport s alertima i preporukama
    """
    try:
        detector = request.app.state.drift_detector
        if not detector:
            raise HTTPException(status_code=503, detail="Drift detector not initialized")

        report = await detector.check_drift()

        return {
            "has_drift": report.has_drift,
            "severity": report.severity,
            "sample_count": report.sample_count,
            "alerts": report.alerts,
            "current": {
                "error_rate": round(report.error_rate, 4),
                "latency_ms": round(report.latency_ms, 1),
                "hallucination_rate": round(report.hallucination_rate, 4)
            },
            "baseline": {
                "error_rate": round(report.baseline_error_rate, 4),
                "latency_ms": round(report.baseline_latency_ms, 1),
                "hallucination_rate": round(report.baseline_hallucination_rate, 4)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking drift status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [ADMIN-API] %(levelname)s: %(message)s"
    )

    logger.info("=" * 60)
    logger.info("MOBILITYONE ADMIN API v2.1")
    logger.info("=" * 60)
    logger.info("Port: 8080")
    logger.info("Docs: http://localhost:8080/admin/docs")
    logger.info("Database: PostgreSQL (admin_user)")
    logger.info("=" * 60)

    uvicorn.run(
        "admin_api:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )

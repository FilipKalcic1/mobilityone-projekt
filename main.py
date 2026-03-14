"""
MobilityOne WhatsApp Bot - FastAPI Application

Main entry point with automatic database initialization.
"""

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware

# Import config FIRST to get LOG_LEVEL
from config import get_settings

settings = get_settings()

# PII-Safe Logging Filter (shared module — single source of truth)
from services.pii_filter import PIIScrubFilter


# Configure logging with level from settings
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
_pii_filter = PIIScrubFilter()
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.addFilter(_pii_filter)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[_stdout_handler]
)

# Reduce noise from verbose libraries (CRITICAL for production readability)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info(f"Logging configured: level={settings.LOG_LEVEL}")

# --- Graceful Shutdown Flag ---
# Set to True on SIGTERM. Webhook checks this to stop accepting new messages
# during the K8s grace period, preventing message loss when Redis/worker are
# already draining.
APP_STOPPING = False

# --- Payload Size Guard (OOM prevention at 1GB RAM) ---
# JSON parsing a 10MB malicious payload at 20 concurrent requests = 200MB spike.
# Combined with baseline 280MB = OOM kill. Reject before parsing.
MAX_REQUEST_BODY_BYTES = 1 * 1024 * 1024  # 1MB

class PayloadSizeGuardMiddleware(BaseHTTPMiddleware):
    """Reject requests >1MB before JSON parsing to prevent OOM at 1GB RAM."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
            except (ValueError, OverflowError):
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header"}
                )
            if length > MAX_REQUEST_BODY_BYTES:
                logger.warning(
                    f"Payload rejected: {length} bytes > {MAX_REQUEST_BODY_BYTES} limit "
                    f"from {request.client.host if request.client else 'unknown'}"
                )
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large (max 1MB)"}
                )
        return await call_next(request)


# --- Request ID Middleware (for distributed tracing) ---
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for log correlation."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:12]
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

# --- HTTPS Redirect Middleware (production only) ---
class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """Redirect HTTP to HTTPS in production."""

    async def dispatch(self, request: Request, call_next):
        # Check X-Forwarded-Proto header (set by reverse proxy/load balancer)
        proto = request.headers.get("X-Forwarded-Proto", "https")
        if proto == "http" and settings.is_production:
            url = request.url.replace(scheme="https")
            return PlainTextResponse(
                content="Redirecting to HTTPS",
                status_code=301,
                headers={"Location": str(url)}
            )
        response = await call_next(request)
        # Add HSTS header in production
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)
TOOLS_LOADED = Gauge('tools_loaded_total', 'Number of tools loaded in registry')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')

async def wait_for_database(max_retries: int = 30, base_delay: int = 2) -> bool:
    """Wait for database to be available and create tables."""
    from database import engine, Base
    from sqlalchemy.exc import OperationalError, DatabaseError
    from models import AuditLog  # noqa

    logger.info("Waiting for database...")

    for attempt in range(max_retries):
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))

            logger.info("Database connection established")

            # Create tables
            logger.info("Creating database tables...")
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            logger.info("Database tables ready")
            return True

        except (OperationalError, DatabaseError, OSError, TimeoutError) as e:
            delay = min(base_delay * (2 ** min(attempt, 5)), 60)
            logger.warning(f"Database not ready (attempt {attempt + 1}/{max_retries}, retry in {delay}s): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
        except Exception as e:
            logger.error(f"Unexpected database error: {type(e).__name__}: {e}")
            return False

    logger.error("Could not connect to database after all retries")
    return False

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    logger.info("Starting MobilityOne Bot v11.0...")
    
    # 1. Wait for database and create tables
    db_ready = await wait_for_database()
    if not db_ready:
        logger.error("Cannot start without database")
        raise RuntimeError("Database not available")
    
    # 2. Initialize Redis with retry (supports Sentinel for HA)
    from services.redis_factory import create_redis_client
    redis_client = None
    for attempt in range(5):
        try:
            redis_client = await create_redis_client(settings)
            app.state.redis = redis_client
            break
        except (ConnectionError, OSError, TimeoutError) as e:
            delay = min(2 * (2 ** attempt), 30)
            logger.warning(f"Redis not ready (attempt {attempt + 1}/5, retry in {delay}s): {e}")
            if attempt < 4:
                await asyncio.sleep(delay)
            else:
                raise RuntimeError(f"Redis not available after 5 attempts: {e}")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise RuntimeError(f"Redis not available: {e}")
    
    # 3. Initialize services
    try:
        from services.api_gateway import APIGateway
        from services.tool_registry import ToolRegistry
        from services.queue_service import QueueService
        from services.cache_service import CacheService
        from services.context_service import ContextService
        
        # API Gateway
        app.state.gateway = APIGateway(redis_client=app.state.redis)
        logger.info("API Gateway initialized")
        
        # Tool Registry
        app.state.registry = ToolRegistry(redis_client=app.state.redis)

        # CRITICAL FIX: Initialize with ALL sources at once (not one by one!)
        # This enables proper caching and avoids 3x embedding generation
        success = await app.state.registry.initialize(settings.swagger_sources)

        if not success:
            logger.error("Tool Registry initialization failed")
            raise RuntimeError("Tool Registry initialization failed")

        logger.info(f"Tool Registry: {len(app.state.registry.tools)} tools")
        
        # Queue Service
        app.state.queue = QueueService(app.state.redis)
        await app.state.queue.create_consumer_group()
        logger.info("Queue Service initialized")
        
        # Cache Service
        app.state.cache = CacheService(app.state.redis)
        logger.info("Cache Service initialized")
        
        # Context Service
        app.state.context = ContextService(app.state.redis)
        logger.info("Context Service initialized")
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        raise

    # 4. Initialize ML models (train if missing)
    try:
        from services.intent_classifier import IntentClassifier, get_query_type_classifier_ml
        from pathlib import Path

        model_dir = Path("/app/models/intent")
        tfidf_model = model_dir / "tfidf_lr_model.pkl"

        if not tfidf_model.exists():
            logger.info("Intent classifier model not found, training...")
            model_dir.mkdir(parents=True, exist_ok=True)
            clf = IntentClassifier(algorithm="tfidf_lr")
            metrics = clf.train()
            logger.info(f"Intent classifier trained: {metrics.get('accuracy', 0):.1%} accuracy")
        else:
            logger.info("Intent classifier model found")

        # Also initialize query type classifier
        query_clf = get_query_type_classifier_ml()
        logger.info("Query type classifier ready")

    except Exception as e:
        logger.warning(f"ML model initialization failed (non-critical): {e}")

    logger.info("Application ready!")
    
    yield
    
    # Shutdown — set flag FIRST so webhook stops accepting new messages
    global APP_STOPPING
    APP_STOPPING = True
    logger.info("Shutting down... APP_STOPPING=True, webhook will reject new messages")

    # Shutdown OpenTelemetry tracing
    try:
        from services.tracing import shutdown_tracing
        await shutdown_tracing()
    except ImportError:
        logger.debug("Tracing module not available, skipping shutdown")
    except Exception as e:
        logger.warning(f"OpenTelemetry shutdown error (non-fatal): {type(e).__name__}: {e}")

    if hasattr(app.state, 'gateway') and app.state.gateway:
        await app.state.gateway.close()

    if hasattr(app.state, 'redis') and app.state.redis:
        await app.state.redis.aclose()

    logger.info("Goodbye!")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="WhatsApp Fleet Management Bot",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Payload size guard (outermost - runs first, rejects before JSON parsing)
app.add_middleware(PayloadSizeGuardMiddleware)

# Request ID
app.add_middleware(RequestIDMiddleware)

# HTTPS enforcement in production
if settings.is_production:
    app.add_middleware(HTTPSRedirectMiddleware)

# Security headers
from services.security_headers import SecurityHeadersMiddleware
app.add_middleware(SecurityHeadersMiddleware)

# CORS - restricted in production, permissive in development
if settings.DEBUG:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # Cannot use credentials with wildcard origin
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    _cors_origins = [o.strip() for o in settings.ADMIN_CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Hub-Signature-256"],
    )

# Include routers
# Simple webhook endpoint that pushes to Redis queue
from webhook_simple import router as webhook_router
app.include_router(webhook_router, prefix="/webhook", tags=["webhook"])

if settings.DEBUG:
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            logger.debug(f"Registered route: {route.path} {list(route.methods) if route.methods else []}")
        else:
            logger.debug(f"Registered non-HTTP route: {route.name if hasattr(route, 'name') else route}")

@app.get("/health")
async def health_check():
    """Health check endpoint.

    IMPORTANT: Only checks LOCAL resources (DB, Redis).
    Does NOT check external APIs (MobilityOne) - those timeouts
    would cause Docker health checks to fail and block worker startup.
    """
    from database import engine

    checks = {
        "status": "healthy",
        "version": settings.APP_VERSION,
    }

    # Only include detailed info in development
    if settings.DEBUG:
        checks["database"] = "disconnected"
        checks["redis"] = "disconnected"
        checks["tools"] = 0

    try:
        # Check database
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        if settings.DEBUG:
            checks["database"] = "connected"

        # Check redis
        if hasattr(app.state, 'redis') and app.state.redis:
            await app.state.redis.ping()
            if settings.DEBUG:
                checks["redis"] = "connected"

        # Check tools
        if hasattr(app.state, 'registry') and app.state.registry:
            if settings.DEBUG:
                checks["tools"] = len(app.state.registry.tools)

        # MobilityOne API status - non-blocking, just report cached token state
        if settings.DEBUG:
            if hasattr(app.state, 'gateway') and app.state.gateway:
                checks["mobility_api"] = "connected" if app.state.gateway.token_manager.is_valid else "no_token"
            else:
                checks["mobility_api"] = "not_initialized"

    except Exception as e:
        checks["status"] = "unhealthy"
        if settings.DEBUG:
            checks["error"] = str(e)

    return checks

@app.get("/ready")
async def readiness_check():
    """Readiness probe - returns 200 when local dependencies are available.

    Does NOT block on external API checks. MobilityOne being down
    should not prevent the bot from handling guest users.
    """
    # Fail readiness during shutdown so K8s stops routing traffic
    if APP_STOPPING:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "shutting down"})

    from database import engine

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "database unavailable"})

    try:
        if hasattr(app.state, 'redis') and app.state.redis:
            # Full write cycle: SET → GET → DEL
            # SET-only misses: read-only replica accepts SET silently in some configs.
            # GET verifies the write landed. DEL confirms delete works (disk-full
            # Redis may accept SET to memory but fail on AOF fsync — DEL catches this).
            # Pod-specific key avoids cross-pod collisions on the same key.
            _ready_key = f"readiness_check:{os.getpid()}"
            await app.state.redis.set(_ready_key, "ok", ex=5)
            val = await app.state.redis.get(_ready_key)
            if val != b"ok" and val != "ok":
                return JSONResponse(status_code=503, content={"ready": False, "reason": "redis write verification failed"})
            await app.state.redis.delete(_ready_key)
        else:
            return JSONResponse(status_code=503, content={"ready": False, "reason": "redis not initialized"})
    except Exception:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "redis unavailable"})

    if not hasattr(app.state, 'registry') or not app.state.registry or len(app.state.registry.tools) == 0:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "tool registry empty"})

    return {"ready": True}

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.get("/metrics")
async def metrics(request: Request):
    """Prometheus metrics endpoint. Protected in production."""
    # In production, only allow Prometheus scraper (by IP or token)
    if settings.is_production:
        token = request.query_params.get("token", "")
        admin_tokens = set()
        for i in range(1, 5):
            env_token = os.environ.get(f"ADMIN_TOKEN_{i}")
            if env_token:
                admin_tokens.add(env_token)
        if admin_tokens and token not in admin_tokens:
            return JSONResponse(status_code=403, content={"detail": "Forbidden"})

    # Read tools count from Redis (written by worker after registry init)
    if hasattr(app.state, 'redis') and app.state.redis:
        try:
            tools_count = await app.state.redis.get(settings.REDIS_STATS_KEY_TOOLS)
            if tools_count:
                TOOLS_LOADED.set(int(tools_count))
        except (ConnectionError, OSError) as e:
            logger.debug(f"Redis unavailable for metrics read: {e}")
        except (ValueError, TypeError) as e:
            logger.debug(f"Invalid tools_count value in Redis: {e}")

    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1
    )

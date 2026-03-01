"""
MobilityOne WhatsApp Bot - FastAPI Application

Main entry point with automatic database initialization.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text

# Import config FIRST to get LOG_LEVEL
from config import get_settings

settings = get_settings()

# Configure logging with level from settings
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Reduce noise from verbose libraries (CRITICAL for production readability)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info(f"Logging configured: level={settings.LOG_LEVEL}")

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


async def wait_for_database(max_retries: int = 30, delay: int = 2) -> bool:
    """Wait for database to be available and create tables."""
    from database import engine, Base
    from models import UserMapping, Conversation, Message, ToolExecution, AuditLog  # noqa
    
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
            
        except Exception as e:
            logger.warning(f"Database not ready (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
    
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
    
    # 2. Initialize Redis
    try:
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=settings.REDIS_MAX_CONNECTIONS
        )
        await redis_client.ping()
        app.state.redis = redis_client
        logger.info("Redis connected")
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
    
    # Shutdown
    logger.info("Shutting down...")
    
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
        "database": "disconnected",
        "redis": "disconnected",
        "tools": 0
    }

    try:
        # Check database
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "connected"

        # Check redis
        if hasattr(app.state, 'redis') and app.state.redis:
            await app.state.redis.ping()
            checks["redis"] = "connected"

        # Check tools
        if hasattr(app.state, 'registry') and app.state.registry:
            checks["tools"] = len(app.state.registry.tools)

        # MobilityOne API status - non-blocking, just report cached token state
        if hasattr(app.state, 'gateway') and app.state.gateway:
            checks["mobility_api"] = "connected" if app.state.gateway.token_manager.is_valid else "no_token"
        else:
            checks["mobility_api"] = "not_initialized"

    except Exception as e:
        checks["status"] = "unhealthy"
        checks["error"] = str(e)

    return checks


@app.get("/ready")
async def readiness_check():
    """Readiness probe - returns 200 when local dependencies are available.

    Does NOT block on external API checks. MobilityOne being down
    should not prevent the bot from handling guest users.
    """
    from database import engine

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "database unavailable"})

    try:
        if hasattr(app.state, 'redis') and app.state.redis:
            await app.state.redis.ping()
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
async def metrics():
    """Prometheus metrics endpoint."""
    # Read tools count from Redis (written by worker after registry init)
    # This fixes the "zero" bug - API doesn't load registry, worker does
    if hasattr(app.state, 'redis') and app.state.redis:
        try:
            tools_count = await app.state.redis.get(settings.REDIS_STATS_KEY_TOOLS)
            if tools_count:
                TOOLS_LOADED.set(int(tools_count))
        except Exception:
            pass  # Metrics should never crash the endpoint

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

"""
Configuration Module
Version: 11.0.1

Centralized configuration with validation.
NO HARDCODED SECRETS - all sensitive values must come from environment.
"""
import os
import logging
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("configuration")

class Settings(BaseSettings):
    
    VERIFY_WHATSAPP_SIGNATURE: bool = False
    # =========================================================================
    # APPLICATION
    # =========================================================================

    APP_ENV: str = Field(default="development")
    APP_NAME: str = Field(default="MobilityOne Bot")
    APP_VERSION: str = Field(default="11.0.0")
    
    # =========================================================================
    # DATABASE (Obavezno ako nema defaulta, ali ovdje često stavljamo default za local)
    # =========================================================================

    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://appuser:password@localhost:5432/mobility_db",
        description="PostgreSQL connection string"
    )
    DB_POOL_SIZE: int = Field(default=10)
    DB_MAX_OVERFLOW: int = Field(default=20)
    DB_POOL_RECYCLE: int = Field(default=3600)
    
    # =========================================================================
    # SERVICE ROLE (bot or admin - determines database permissions)
    # =========================================================================
    SERVICE_ROLE: str = Field(default="bot")

    # =========================================================================
    # REDIS
    # =========================================================================
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_MAX_CONNECTIONS: int = Field(default=50)
    REDIS_STATS_KEY_TOOLS: str = Field(default="stats:tools_loaded")
    
    # =========================================================================
    # INFOBIP (WhatsApp) - Ostavio sam Optional ako nije nužan za start
    # =========================================================================
    INFOBIP_API_KEY: Optional[str] = Field(default=None)
    INFOBIP_BASE_URL: str = Field(default="api.infobip.com")
    INFOBIP_SECRET_KEY: Optional[str] = Field(default=None)
    INFOBIP_SENDER_NUMBER: Optional[str] = Field(default=None)
    
    # =========================================================================
    # MOBILITYONE API - REQUIRED (NEMA DEFAULT VRIJEDNOSTI!)
    # Ovo forsira čitanje iz ENV. Ako fali, app se ruši.
    # =========================================================================
    MOBILITY_API_URL: str = Field(..., description="MobilityOne API base URL")
    MOBILITY_AUTH_URL: str = Field(..., description="MobilityOne OAuth2 token endpoint")
    MOBILITY_CLIENT_ID: str = Field(..., description="OAuth2 client ID")
    MOBILITY_CLIENT_SECRET: str = Field(..., description="OAuth2 client secret")
    MOBILITY_TENANT_ID: str = Field(..., description="Tenant ID for x-tenant header")
    MOBILITY_SCOPE: Optional[str] = Field(default=None, description="OAuth2 scope(s) for token request")
    
    # =========================================================================
    # AZURE OPENAI - REQUIRED (NEMA DEFAULT VRIJEDNOSTI!)
    # =========================================================================
    AZURE_OPENAI_ENDPOINT: str = Field(..., description="Azure OpenAI endpoint URL")
    AZURE_OPENAI_API_KEY: str = Field(..., description="Azure OpenAI API key")
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-08-01-preview")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = Field(default="gpt-4o-mini")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = Field(default="text-embedding-ada-002")
    
    # =========================================================================
    # AI SETTINGS
    # =========================================================================
    AI_MAX_ITERATIONS: int = Field(default=6)
    AI_TEMPERATURE: float = Field(default=0.2)
    AI_MAX_TOKENS: int = Field(default=1500)
    EMBEDDING_BATCH_SIZE: int = Field(default=5)
    SIMILARITY_THRESHOLD: float = Field(default=0.55)

    # ACTION-FIRST PROTOCOL: Force tool call when similarity >= this threshold
    # MASTER PROMPT v9.0: Bot MUST execute tool_call, NOT generic text response
    # v13.1: LOWERED from 0.85 to 0.75 - AI was ignoring our best matches!
    ACTION_THRESHOLD: float = Field(
        default=0.75,
        description="Similarity threshold for forced tool execution (no text fallback)"
    )
    
    # =========================================================================
    # RATE LIMITING
    # =========================================================================
    RATE_LIMIT_PER_MINUTE: int = Field(default=20)
    RATE_LIMIT_WINDOW: int = Field(default=60)
    
    # =========================================================================
    # CACHE TTL (seconds)
    # =========================================================================
    CACHE_TTL_TOKEN: int = Field(default=3500)
    CACHE_TTL_USER: int = Field(default=300)
    CACHE_TTL_CONTEXT: int = Field(default=86400)
    CACHE_TTL_TOOLS: int = Field(default=3600)
    CACHE_TTL_CONVERSATION: int = Field(default=1800)
    
    # =========================================================================
    # MONITORING & LOGGING
    # =========================================================================
    SENTRY_DSN: Optional[str] = Field(default=None)

    # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    # In production, set to WARNING or ERROR to reduce noise
    LOG_LEVEL: str = Field(default="INFO", description="Logging level (DEBUG/INFO/WARNING/ERROR)")

    # =========================================================================
    # CONFIGURATION (Pydantic V2 Style)
    # =========================================================================
    model_config = SettingsConfigDict(
        env_file=".env",            # Pokušaj učitati .env file
        env_file_encoding="utf-8",
        case_sensitive=True,        # Razlikuj velika/mala slova (MOBILITY_API_URL != mobility_api_url)
        extra="ignore"              # Ignoriraj viška varijable u ENV
    )

    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================
    @property
    def tenant_id(self) -> str:
        return self.MOBILITY_TENANT_ID
    
    @property
    def swagger_sources(self) -> List[str]:
        """
        Get Swagger sources from environment variable.

        Expected format: SWAGGER_SOURCES=https://api.example.com/service1/swagger.json,https://api.example.com/service2/swagger.json

        Fallback: Auto-discover from MOBILITY_API_URL if configured.
        """
        # Try environment variable first
        sources_env = os.getenv("SWAGGER_SOURCES", "")
        if sources_env:
            return [s.strip() for s in sources_env.split(",") if s.strip()]

        # Fallback: Auto-discover from API base (backward compatibility)
        if not self.MOBILITY_API_URL:
            return []

        base = self.MOBILITY_API_URL.rstrip("/")

        # Default services (can be overridden via env)
        default_services = {
            "automation": "v1.0.0",
            "tenantmgt": "v2.0.0-alpha",
            "vehiclemgt": "v2.0.0-alpha"
        }

        return [
            f"{base}/{service}/swagger/{version}/swagger.json"
            for service, version in default_services.items()
        ]
    
    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def DEBUG(self) -> bool:
        return self.APP_ENV == "development"

    # =========================================================================
    # VALIDATORS
    # =========================================================================
    @field_validator('MOBILITY_API_URL', 'MOBILITY_AUTH_URL', 'AZURE_OPENAI_ENDPOINT')
    @classmethod
    def validate_url(cls, v: str) -> str:
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError(f"URL must start with http or https: {v}")
        return v.rstrip('/') if v else v


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    This call will FAIL if required environment variables are missing.
    """
    try:
        # Pydantic ovdje automatski radi os.getenv za svako polje
        return Settings()
    except Exception as e:
        # Ovo će ti se ispisati u Docker logovima ako nešto fali
        logger.critical(f"FATAL CONFIG ERROR: Could not load settings. Missing env vars? Error: {e}")
        raise e
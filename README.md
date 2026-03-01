# MobilityOne WhatsApp Bot

AI-powered fleet management chatbot that connects WhatsApp users to the MobilityOne API platform via Azure OpenAI.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-2764%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen)
![Linting](https://img.shields.io/badge/linting-ruff-orange)
![Security](https://img.shields.io/badge/security-bandit-green)
![License](https://img.shields.io/badge/license-Proprietary-red)

## Architecture

```
WhatsApp (Infobip) --> FastAPI Webhook (8000) --> Redis Stream --> Worker
                                                                    |
                                                          AI Orchestrator
                                                          (Azure OpenAI)
                                                                    |
                                                          Tool Registry
                                                          (950+ API tools)
                                                                    |
                                                          MobilityOne APIs
                                                          (automation, vehiclemgt, tenantmgt)

Admin API (8080) --> PostgreSQL (audit_logs, hallucination_reports)
```

**Three processes:**
- `main.py` - FastAPI webhook receiver (port 8000). Accepts WhatsApp messages, pushes to Redis stream.
- `worker.py` - Redis stream consumer. Processes messages through the AI orchestrator, calls MobilityOne APIs, sends responses via WhatsApp.
- `admin_api.py` - Admin dashboard API (port 8080). Reviews hallucinations, manages users, views audit logs.

## Key Design Decisions

- **Multi-tenant**: Tenant resolved dynamically per user (DB lookup -> phone prefix -> env fallback). Each API call includes `x-tenant` header.
- **Dual-user PostgreSQL**: `bot_user` (limited access) for the bot, `admin_user` (full access) for the admin API.
- **950+ tools from Swagger**: Tools are auto-parsed from OpenAPI specs, indexed with FAISS for semantic search, and selected by the AI orchestrator.
- **Redis Streams**: Async message processing with consumer groups, automatic reconnection, and backpressure handling.
- **Pydantic Settings**: All configuration centralized in `config.py` with fail-fast validation. No hardcoded secrets.
- **Circuit Breaker**: Per-endpoint circuit breaker prevents cascading failures (CLOSED -> OPEN -> HALF_OPEN -> CLOSED).

## Project Structure

```
.
├── main.py                  # FastAPI webhook (port 8000)
├── worker.py                # Redis stream consumer
├── admin_api.py             # Admin API (port 8080)
├── config.py                # Centralized Pydantic settings
├── database.py              # SQLAlchemy async engine (dual-user)
├── models.py                # SQLAlchemy ORM models
├── base.py                  # Shared declarative base
├── webhook_simple.py        # WhatsApp webhook router
│
├── services/                # Core business logic (50+ modules)
│   ├── ai_orchestrator.py   # Main AI conversation loop
│   ├── api_gateway.py       # HTTP client for MobilityOne APIs
│   ├── tenant_service.py    # Dynamic multi-tenant routing
│   ├── user_service.py      # User context and session management
│   ├── tool_executor.py     # API tool execution with parameter injection
│   ├── cost_tracker.py      # Token usage and cost tracking
│   ├── gdpr_masking.py      # PII masking (phone, email, OIB)
│   ├── engine/              # Conversation flow engine
│   ├── registry/            # Tool registry (Swagger parsing, FAISS search)
│   ├── context/             # User context management
│   └── reasoning/           # Query planning
│
├── tests/                   # pytest test suite (2764 tests)
├── scripts/                 # Utility scripts
│   ├── benchmarks/          # Manual integration/accuracy benchmarks
│   ├── sync_tools.py        # Swagger tool synchronization
│   └── swagger_watcher.py   # Watch for Swagger spec changes
│
├── config/                  # JSON configuration files
├── models/                  # ML models (intent classifier, query type)
├── data/training/           # Training data for ML models
├── alembic/                 # Database migrations
├── k8s/                     # Kubernetes manifests
├── docker/                  # Docker support (Grafana, Prometheus)
├── docker-compose.yml       # Local development stack
├── Dockerfile               # Production container image
└── .github/workflows/       # CI pipeline (test + lint + security)
```

## Quick Start

```bash
# 1. Copy and fill environment variables
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # testing + linting tools

# 3. Run database migrations
alembic upgrade head

# 4. Start services
make run          # Webhook API (port 8000)
make run-worker   # Message processor
make run-admin    # Admin API (port 8080)
```

Or with Docker:

```bash
make docker-up    # Start full stack (PostgreSQL, Redis, API, Worker, Admin)
```

## Development

```bash
make test         # Run all 2764 tests
make coverage     # Run tests with coverage report
make lint         # Run ruff linter
make format       # Auto-format code
make check        # Run lint + tests
make help         # Show all available commands
```

## Environment Variables

See [.env.example](.env.example) for the full list. Required variables:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `MOBILITY_API_URL` | MobilityOne API base URL |
| `MOBILITY_AUTH_URL` | OAuth2 token endpoint |
| `MOBILITY_CLIENT_ID` | OAuth2 client ID |
| `MOBILITY_CLIENT_SECRET` | OAuth2 client secret |
| `MOBILITY_TENANT_ID` | Default tenant ID (fallback) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |

## CI/CD

GitHub Actions runs on every push to `main`:
- **test**: Runs the full pytest suite with coverage reporting
- **lint**: Ruff linting + ruff format check + entry point compilation
- **security**: Bandit static analysis + pip-audit dependency check

## Health Endpoints

| Endpoint | Service | Auth | Purpose |
|----------|---------|------|---------|
| `GET /health` | Webhook API | None | Liveness probe (DB + Redis + tools) |
| `GET /ready` | Webhook API | None | Readiness probe (all dependencies up) |
| `GET /health` | Admin API | None | Liveness probe (DB + Redis + drift) |
| `GET /ready` | Admin API | None | Readiness probe (all dependencies up) |

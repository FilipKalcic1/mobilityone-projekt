# MobilityOne WhatsApp Bot

AI-powered fleet management chatbot that connects WhatsApp users to the MobilityOne API platform via Azure OpenAI.

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
├── tests/                   # pytest test suite (194 tests)
├── scripts/                 # Utility scripts
│   ├── benchmarks/          # Manual integration/accuracy benchmarks
│   ├── sync_tools.py        # Swagger tool synchronization
│   └── swagger_watcher.py   # Watch for Swagger spec changes
│
├── config/                  # JSON configuration files
│   ├── context_param_schemas.json
│   ├── tool_categories.json
│   └── tool_documentation.json
│
├── models/                  # ML models (intent classifier, query type)
├── data/training/           # Training data for ML models
├── alembic/                 # Database migrations
├── k8s/                     # Kubernetes manifests
├── docker/                  # Docker support (Grafana, Prometheus)
├── docker-compose.yml       # Local development stack
├── Dockerfile               # Production container image
└── .github/workflows/       # CI pipeline (test + lint)
```

## Setup

```bash
# 1. Copy and fill environment variables
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run database migrations
alembic upgrade head

# 4. Start services
uvicorn main:app --port 8000          # Webhook API
python worker.py                       # Message processor
uvicorn admin_api:app --port 8080      # Admin API
```

## Testing

```bash
pytest                    # Run all 194 tests
pytest tests/ -v          # Verbose output
pytest tests/ -x          # Stop on first failure
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
- **test**: Runs the full pytest suite
- **lint**: Ruff linting + entry point compilation

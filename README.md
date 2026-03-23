# MobilityOne WhatsApp Bot

AI-powered fleet management chatbot that connects WhatsApp users to the MobilityOne platform (950+ API endpoints) via Azure OpenAI.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-3197%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen)
![Tools](https://img.shields.io/badge/API%20tools-950+-blueviolet)
![Top--1 Accuracy](https://img.shields.io/badge/routing%20accuracy-100%25-brightgreen)

## Pregled sustava

```
WhatsApp (Infobip)
    |
    v
FastAPI Webhook (port 8000)          Admin API (port 8080)
    |                                     |
    v                                     v
Redis Stream (inbound)               PostgreSQL
    |                                 (audit, hallucinations, GDPR)
    v
Worker (async processor)
    |
    +---> ML Intent Classifier (TF-IDF, <1ms)
    |         |
    |    [confidence >= 85%] ---> Fast Path (0 LLM tokena)
    |         |
    |    [confidence < 85%]  ---> FAISS Semantic Search (950 tools)
    |                                  |
    |                             LLM Routing (gpt-4o-mini)
    |
    +---> API Gateway ---> MobilityOne APIs
    |
    +---> Response Formatter
    |
    v
Redis List (outbound) ---> WhatsApp (Infobip)
```

### Tri procesa

| Proces | Port | Opis |
|--------|------|------|
| **main.py** | 8000 | FastAPI webhook - prima WhatsApp poruke, stavlja u Redis Stream |
| **worker.py** | — | Async procesor - ML routing, API pozivi, slanje odgovora |
| **admin_api.py** | 8080 | Admin panel - pregled halucinacija, GDPR operacije, audit log |

## Ključne brojke

| Metrika | Vrijednost |
|---------|-----------|
| API alata | 950+ (auto-parsed iz OpenAPI specifikacija) |
| Routing accuracy (Top-1) | **100%** na svih 950 alata |
| ML Fast Path | ~85% upita (0 LLM poziva) |
| Intent klasa | 45+ (TF-IDF + Logistic Regression) |
| Query Type klasa | 12 (_id, _documents, _metadata, _Agg, ...) |
| Prosječna latencija | 2-3 sekunde (uključuje Azure OpenAI) |
| FAISS search latencija | 1-5ms |
| Test suite | 3,197 testova, 88% coverage |

## Quick Start

### Docker Compose (preporučeno)

```bash
# 1. Kopiraj environment varijable
cp .env.example .env
# Popuni DATABASE_URL, REDIS_URL, AZURE_OPENAI_*, MOBILITY_* varijable

# 2. Pokreni cijeli stack
docker-compose up -d

# 3. Provjeri zdravlje sustava
curl http://localhost:8000/ready
curl http://localhost:8080/ready
```

### Lokalni development

```bash
# 1. Instaliraj dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 2. Pokreni migracije
alembic upgrade head

# 3. Pokreni servise
make run          # Webhook API (port 8000)
make run-worker   # Message processor
make run-admin    # Admin API (port 8080)
```

## Development

```bash
make test         # Pokreni svih 3,197 testova
make coverage     # Testovi + coverage report (minimum 85%)
make lint         # Ruff linter
make format       # Auto-format koda
make check        # lint + test u jednom koraku
```

## Struktura projekta

```
.
├── main.py                    # FastAPI webhook (port 8000)
├── worker.py                  # Redis Stream consumer + burst mode
├── admin_api.py               # Admin API (port 8080)
├── webhook_simple.py          # WhatsApp router + 3-tier DLQ
├── config.py                  # Pydantic Settings (80+ varijabli)
├── database.py                # SQLAlchemy async (dual-user security)
├── models.py                  # ORM modeli (6 tablica)
├── tool_routing.py            # Intent-to-tool mappings (200+ pravila)
│
├── services/                  # Core business logic (50+ modula)
│   ├── engine/                # Conversation engine (state machine)
│   ├── registry/              # Tool registry (Swagger parser, FAISS)
│   ├── context/               # User context management
│   ├── unified_router.py      # 3-tier routing (ML → CP → LLM)
│   ├── unified_search.py      # FAISS + BM25 + boost pipeline
│   ├── intent_classifier.py   # ML intent detection (45+ klasa)
│   ├── faiss_vector_store.py  # Vektorska pretraga (Ada-002, 1536d)
│   ├── dynamic_threshold.py   # Entropy-aware decision engine
│   ├── api_gateway.py         # HTTP klijent (circuit breaker)
│   ├── gdpr_masking.py        # PII masking + erasure + export
│   └── ...
│
├── config/                    # JSON konfiguracija
│   ├── tool_documentation.json  # 950 alata s primjerima (1.5MB)
│   ├── tool_categories.json     # Kategorizacija alata
│   └── processed_tool_registry.json  # Parsani OpenAPI (2.8MB)
│
├── models/                    # ML modeli (TF-IDF, LogReg)
├── data/training/             # Training podaci
├── tests/                     # pytest suite (3,197 testova)
├── scripts/                   # Utility skripte (30+)
├── k8s/                       # Kubernetes manifesti
├── docker/                    # Prometheus, Grafana konfiguracija
├── alembic/                   # Database migracije
└── docker-compose.yml         # Development stack
```

## Environment varijable

| Varijabla | Opis | Required |
|-----------|------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Da |
| `REDIS_URL` | Redis connection string | Da |
| `MOBILITY_API_URL` | MobilityOne API base URL | Da |
| `MOBILITY_AUTH_URL` | OAuth2 token endpoint | Da |
| `MOBILITY_CLIENT_ID` | OAuth2 client ID | Da |
| `MOBILITY_CLIENT_SECRET` | OAuth2 client secret | Da |
| `MOBILITY_TENANT_ID` | Default tenant ID (fallback) | Da |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Da |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Da |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | LLM deployment (gpt-4o-mini) | Da |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding deployment (ada-002) | Da |
| `INFOBIP_API_KEY` | Infobip API key | Ne* |
| `GDPR_HASH_SALT` | Salt za pseudonimizaciju (min 32 znaka) | Da |
| `ADMIN_AUTH_TOKEN` | Admin API autentifikacija | Da |

\* Potrebno za slanje WhatsApp poruka, ne za pokretanje sustava.

## Health Endpoints

| Endpoint | Servis | Opis |
|----------|--------|------|
| `GET /health` | API (8000) | Liveness — DB + Redis + registry |
| `GET /ready` | API (8000) | Readiness — sve ovisnosti + Redis write test |
| `GET /health` | Admin (8080) | Liveness — DB + Redis |
| `GET /ready` | Admin (8080) | Readiness — sve ovisnosti |
| `GET /metrics` | API (8000) | Prometheus metrike |

## Daljnja dokumentacija

| Dokument | Opis |
|----------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Detaljna tehnička arhitektura |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Docker Compose + Kubernetes deployment |
| [SECURITY.md](SECURITY.md) | Sigurnost, GDPR, EU AI Act compliance |
| [CHANGELOG.md](CHANGELOG.md) | Povijest verzija |

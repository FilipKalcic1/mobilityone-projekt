# CLAUDE.md — MobilityOne WhatsApp Bot

**Last Audit**: 2026-03-11 (Zero-Defect Audit — L8 Systems Engineer review)
**Status**: Construction Phase COMPLETE. Maintenance & Scale mode.

## Must-Not-Change (Invariants)

These patterns exist because simpler alternatives were tried and FAILED in production or under audit. Do NOT simplify, refactor, or "clean up" any of these without reading the full rationale below:

1. **3-Tier DLQ** (`webhook_simple.py`): Redis → File → stderr. All 3 tiers are required because the webhook returns 200 to Infobip regardless. Removing any tier = permanent message loss under the corresponding failure mode.
2. **Lua Atomic Lock Release** (`worker.py`): The Lua script conditionally deletes only if the caller still holds the lock. A plain `DELETE` causes double-processing under timeout races. The SHA is verified every 60s and auto-reloaded after `SCRIPT FLUSH`.
3. **Guaranteed QoS** (`k8s/deployment.yaml`): `requests == limits` on all pods. Do NOT set limits > requests — this changes QoS class from Guaranteed to Burstable, making pods eligible for eviction under node memory pressure.
4. **Startup Probes** (all deployments): 30 × 10s = 5-min window. At 0.5 CPU, FAISS/ML model loading takes 60-120s. Without startup probes, liveness kills the pod before it's ready.
5. **DLQ File Size Cap** (`webhook_simple.py`): 5MB limit prevents tmpfs (10-50Mi) exhaustion. Without this cap, a sustained Redis outage fills tmpfs → pod eviction → message loss cascade.
6. **Redis Write Healthcheck** (`main.py /ready`): Uses `SET` → `GET` → `DEL` (not `PING`). PING succeeds even when Redis is in read-only mode (replica promotion). Full write cycle catches disk-full, read-only, and silent-accept failures.
7. **ACK-After-Enqueue** (`worker.py`): `XACK` only fires after `_enqueue_outbound` RPUSH succeeds. If enqueue fails (Redis down mid-processing), the message stays pending in the stream and is reclaimed on restart. Do NOT move ACK back into `finally` unconditionally.
8. **APP_STOPPING Webhook Drain** (`main.py` + `webhook_simple.py`): On SIGTERM, `APP_STOPPING=True` is set before any cleanup. The webhook returns 503 (Infobip retries), and `/ready` returns 503 (K8s stops routing). Do NOT remove the flag check from the webhook handler.

## Tech Stack

- **Runtime**: Python 3.12, FastAPI, uvicorn
- **Database**: PostgreSQL 15 (async via asyncpg + SQLAlchemy 2.0)
- **Cache/Queue**: Redis 7 (Streams for inbound, Lists for outbound)
- **AI**: Azure OpenAI gpt-4o-mini, text-embedding-ada-002
- **ML**: scikit-learn TF-IDF + LogisticRegression, FAISS vector search
- **Infra**: Docker Compose (dev), Kubernetes + KEDA (prod)

## Commands

```bash
# Run tests (unit only, skips integration)
pytest tests/ -v --ignore=tests/test_booking_flow.py --ignore=tests/test_case_flow.py --ignore=tests/test_mileage_flow.py

# Run with coverage
pytest tests/ --cov=services --cov=config --cov-report=term-missing --cov-fail-under=85

# Lint
ruff check .
ruff format --check .

# Security scan
bandit -r services/ -f json
pip-audit

# Docker (dev)
docker-compose up -d                          # API + Worker + Redis + Postgres
docker-compose --profile admin up -d          # + Admin API
docker-compose --profile monitoring up -d     # + Prometheus + Grafana

# Docker (manual worker scale)
docker-compose up -d --scale worker=3

# K8s deploy (300s timeout: FAISS/ML loading takes 60-120s at 0.5 CPU)
kubectl apply -f k8s/deployment.yaml
kubectl wait --for=condition=complete job/mobility-db-migrate --timeout=120s
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/keda-autoscaler.yaml
kubectl rollout status deployment/mobility-api --timeout=300s
kubectl rollout status deployment/mobility-worker --timeout=300s

# Production readiness check (run inside container)
python scripts/verify_production_readiness.py

# Load test (requires locust)
locust -f scripts/locust_webhook_load.py --host http://localhost:8000

# Manual healthcheck
docker exec mobility_worker python -c "import os,redis; r=redis.from_url(os.environ['REDIS_URL']); r.ping()"
```

## Architecture — Critical Patterns

### 3-Tier DLQ (webhook_simple.py `_write_dlq`)

When a webhook message cannot be pushed to Redis after 3 retries:

1. **Primary**: `LPUSH dlq:webhook` in Redis (durable, monitorable via `LLEN`)
2. **Fallback**: Append to `/tmp/dlq.jsonl` (survives Redis outage within pod lifetime)
3. **Last resort**: `sys.stderr` (captured by log aggregators if configured)

**Why all 3 tiers**: The webhook returns 200 to Infobip regardless (line 389) to prevent retry storms. Once we return 200, Infobip considers the message delivered. If the DLQ is not durable, the user's message is permanently lost. Do NOT simplify this to stderr-only.

### Burst Mode (worker.py)

KEDA ScaledJob spawns short-lived worker pods for traffic spikes. These pods read `BURST_MODE=true` and `MAX_MESSAGES=100` from env vars and:

- Exit after processing `MAX_MESSAGES` (processed + failed combined)
- Exit after `BURST_IDLE_TIMEOUT` (default 5 min) of no messages
- Trigger `GracefulShutdown` on exit so in-flight messages complete

Regular workers (BURST_MODE not set) run indefinitely. Do NOT remove the burst exit logic — it's what makes `ScaledJob` pods actually terminate and report `Completed` to K8s.

### Atomic Lock Release (worker.py)

Distributed message dedup uses `SET NX EX` to acquire and a **Lua script** to release. The Lua script ensures only the lock holder can delete:

```lua
if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end
```

The SHA is registered at startup via `script_load()` and verified every 60s in `_health_reporter()`. If Redis flushes scripts (e.g., `SCRIPT FLUSH`), the worker auto-reloads. Do NOT replace this with a plain `DELETE` — it causes double-processing under timeout races.

### 3-Layer Routing (services/engine/__init__.py)

1. **QueryRouter** (ML, <1ms): TF-IDF classifier for known intents
2. **UnifiedRouter** (LLM): Azure OpenAI for ambiguous/complex queries
3. **ChainPlanner** (fallback): Multi-step operation planning

### Message Flow

```
Infobip → webhook_simple.py → Redis Stream → worker.py → MessageEngine → API Gateway → Redis List → worker outbound → WhatsApp
```

### Concurrency Model

- `MAX_CONCURRENT=10` via `asyncio.Semaphore` in worker (tuned for 0.5 CPU — KEDA scales pods instead of overloading one)
- Per-message distributed lock in Redis (`msg_lock:{sender}:{id}`) with Lua-based atomic release
- 90-second timeout on `_process_message()` via `asyncio.wait_for`
- Per-user conversation state in Redis (not shared singleton)
- `_lock_access_times` tracks lock age; stale locks cleaned every 100 messages

## K8s Production Config

### Resource Constraints (Guaranteed QoS)

All pods use `requests == limits` for Guaranteed QoS class (eviction-resistant):

| Component | CPU | Memory | Grace Period |
|-----------|-----|--------|-------------|
| API | 500m | 512Mi | 30s |
| Worker | 500m | 1Gi | 120s (> 90s msg timeout) |
| Admin | 100m | 256Mi | 30s |
| Burst Job | 500m | 512Mi | 30s |

### Security Context (all pods)

- `runAsNonRoot: true` / `runAsUser: 1000`
- `readOnlyRootFilesystem: true` (writable `/tmp` via tmpfs emptyDir)
- `allowPrivilegeEscalation: false`
- `capabilities: drop: ["ALL"]`

### Probes

All pods use **startup probes** (30 × 10s = 5-min window) to prevent slow-start kill loops at 0.5 CPU. Liveness/readiness probes only activate after startup succeeds.

### Monitoring Alerts (keda-autoscaler.yaml)

- **CPUThrottlingHigh**: >25% CFS periods throttled over 5min
- **MemoryHighWater**: >85% of 1Gi limit for 2min
- **WorkerQueueBacklog**: >100 pending messages for 5min
- **KedaNotScaling**: Queue >50 but KEDA inactive for 2min

### Memory Profiling

Set `TRACEMALLOC=true` env var to enable lightweight (1-frame) memory tracing. Top-5 allocators are logged automatically when memory exceeds 800MB.

### Memory Fragmentation Mitigation

Peak memory at 20 concurrent messages: **~280MB** (28% of 1GB). Burst workers (KEDA ScaledJob) naturally cycle every 100 messages or 5-min idle, fully clearing fragmentation. Long-lived regular workers have no `maxLifetime` — if memory drift is observed over weeks, restart the Deployment weekly via `kubectl rollout restart deployment/mobility-worker`.

## EU/Croatian Compliance

### EU AI Act Transparency

- Bot self-identifies as `"MobilityOne AI asistent"` in all system prompts AND all greeting responses (`flow_phrases.py`)
- Hallucination detection: `HallucinationReport` table with `reviewed` flag and admin review workflow
- Confidence threshold: 85%+ for ML fast path (99.24% model accuracy justifies lower threshold), lower confidence defers to LLM with human review capability
- Full audit trail: `Message → ToolExecution → HallucinationReport → AuditLog` tables

### Croatian Locale

- Diacritics: O(n) character map (`DIACRITIC_MAP` in intent_classifier.py) — no Unicode normalization overhead
- Date format: `DD.MM.YYYY` (Croatian standard), keywords: sutra/danas/prekosutra/ujutro/popodne
- All user-facing messages in Croatian
- All `open(..., 'w')` calls use `encoding='utf-8'` explicitly — prevents platform-dependent encoding fallback for č, ć, ž, š, đ

## GDPR Compliance

### PII Masking

Two layers of protection:
1. **Manual masking**: `phone[-4:]` in log strings. Verified files: `user_service.py`, `webhook_simple.py`, `worker.py`.
2. **PIIScrubFilter** (`main.py`, `worker.py`): `logging.Filter` that regex-replaces phone numbers (Croatian + international patterns) in ALL log messages, format args, and exception tracebacks before they reach stdout/stderr. This catches any PII that leaks through f-strings or traceback text.

Run `python scripts/verify_production_readiness.py` to scan for leaks.

### Right to Erasure

- **Database**: `gdpr_masking.py` `anonymize_user_data()` — anonymizes user_mappings, messages, conversations, hallucination_reports
- **Redis**: `gdpr_masking.py` `erase_redis_state()` — deletes `conv_state:`, `chat_history:`, `user_context:`, `tenant:` keys + scrubs DLQ entries
- **Admin endpoint**: `POST /admin/gdpr/erase/{phone}` — triggers both DB + Redis erasure

### Right to Data Portability

- **Database**: `gdpr_masking.py` `export_user_data()` — exports user profile, conversations, messages, hallucination reports as structured JSON
- **Redis**: Same function exports ephemeral state (`conv_state:`, `chat_history:`, `user_context:` keys) when `redis_client` is provided
- **Admin endpoint**: `GET /admin/gdpr/export/{phone}` — returns machine-readable JSON of all user data from Postgres + Redis (Article 20 compliant)

### DLQ Privacy

`dlq:webhook` stores full sender + message text. It has `LTRIM` capping at 10K entries and a **30-day TTL** (renewed on each write). For GDPR, the erasure endpoint scrubs matching entries. The fallback DLQ file `/tmp/dlq.jsonl` is capped at 5MB to prevent tmpfs exhaustion.

## Quality Standards

### Coverage Requirements

- **Global minimum**: 85% (`pyproject.toml` `fail_under = 85`)
- **whatsapp_service.py**: Must maintain 90%+ coverage (sole outbound exit point)
- **webhook_simple.py**: Must maintain 90%+ coverage (sole inbound entry point, contains DLQ logic)
- **worker.py**: Burst mode, lock cleanup, and DLQ paths must have dedicated tests

### Files Excluded from Coverage (intentional)

These are excluded because they depend on external services not mockable in unit tests:

- `faiss_vector_store.py` — FAISS C library bindings
- `llm_reranker.py` — Live LLM calls
- `unified_search.py` — Combines FAISS + LLM
- `queue_service.py` — Thin Redis wrapper (tested via integration)
- `message_engine.py` — Backward-compat shim (14 lines)
- `query_type_classifier.py` — ML model wrapper

Do NOT add more files to this list without adding a comment explaining why.

## Database Security

Dual-user model enforced in `docker/init-db.sh` and `alembic/versions/001_initial_schema.py`:

- **bot_user**: SELECT/INSERT/UPDATE/DELETE on operational tables. INSERT-only on `hallucination_reports`. NO access to `audit_logs`.
- **admin_user**: Full access to all tables. Used only by migrations and admin API.

## Release Checklist

Run these 6 checks before every release:

```markdown
- [ ] `pytest tests/ --cov --cov-fail-under=85` passes with no failures
- [ ] `ruff check . && ruff format --check .` clean
- [ ] `bandit -r services/ -q` no high-severity findings
- [ ] `python -c "import ast; [ast.parse(open(f).read()) for f in ['worker.py','webhook_simple.py','main.py','admin_api.py']]"` — all entry points parse
- [ ] `docker-compose build && docker-compose up -d && sleep 30 && curl -f http://localhost:8000/ready` — smoke test passes
- [ ] `python scripts/verify_production_readiness.py` — Lua cache, FAISS integrity, memory baseline, PII scan all pass
```

## Known Brittleness

- **intent_classifier.py** uses `print()` in its `__main__` block (lines 1078-1104) — acceptable for CLI training, but do not add print() to runtime paths
- **embedding_evaluator.py** uses `print()` in report generation — same: CLI-only, not runtime
- **worker.py `__main__`** does not catch `KeyboardInterrupt` explicitly — `asyncio.run()` handles it via `CancelledError`, which `main()` catches. This is correct Python 3.12 behavior.
- **Redis client reset in webhook_simple.py** is now protected by `_redis_lock` — do not remove the lock acquisition around `_redis_client = None`
- **admin_review.py** SQL injection regex uses bounded `.{0,100}?` — do NOT revert to greedy `.*` (ReDOS vulnerability)
- **DLQ file at `/tmp/dlq.jsonl`** is capped at 5MB (`_DLQ_FILE_MAX_BYTES`) to prevent tmpfs exhaustion; primary path is Redis `dlq:webhook`
- **`/ready` endpoint** verifies Redis write capability via `SET readiness_check` (not just `PING`) — do not downgrade to ping-only
- **FAISS vector store** contains only tool documentation embeddings — no user PII. NOT included in GDPR erasure (by design)

## Workflow Instructions

### Planning
- Start every non-trivial task with plan mode. Explore the codebase deeply before writing any code.
- Use subagents (Explore type) for parallel research across multiple files/systems.
- Exit plan mode only after the plan is approved.

### Implementation
- Follow the approved plan step-by-step. Mark each task as completed immediately.
- After each change, verify syntax: `python -c "import ast; ast.parse(open(f).read())"`
- Run tests after all changes: `pytest tests/ -v --ignore=tests/test_booking_flow.py --ignore=tests/test_case_flow.py --ignore=tests/test_mileage_flow.py`

### Self-Improvement Loop
- After fixing a bug, check if the same pattern exists elsewhere.
- If a fix changes routing or flow logic, trace the full message path end-to-end.
- Update CLAUDE.md if a new invariant or brittleness is discovered.

### Core Principles
- Never simplify patterns marked as "Must-Not-Change" without reading the full rationale.
- Prefer data-driven decisions: run evaluations, measure accuracy, then decide.
- Croatian locale: all user-facing messages in Croatian, use `encoding='utf-8'` for file I/O.
- Keep changes minimal — fix what's broken, don't refactor surroundings.

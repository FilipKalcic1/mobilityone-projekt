# Deployment Guide — MobilityOne WhatsApp Bot

> Verzija: 12.0 | Ažurirano: 2026-03-22

## Sadržaj

- [Pregled infrastrukture](#pregled-infrastrukture)
- [Docker Compose (Development)](#docker-compose-development)
- [Kubernetes (Production)](#kubernetes-production)
- [Environment varijable](#environment-varijable)
- [Cache i Embeddings](#cache-i-embeddings-kritično)
- [Database Setup](#database-setup)
- [Startup Sequence](#startup-sequence)
- [Health Checks](#health-checks)
- [Autoscaling (KEDA)](#autoscaling-keda)
- [Network i Security](#network-i-security)
- [Resource Limits](#resource-limits)
- [Backup Strategy](#backup-strategy)
- [Monitoring](#monitoring)
- [CI/CD Pipeline](#cicd-pipeline)
- [Troubleshooting](#troubleshooting)
- [Production Checklist](#production-checklist)

---

## Pregled infrastrukture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KUBERNETES CLUSTER                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │
│   │  API Pod (×2)   │  │ Worker Pod      │  │ Admin Pod (×1)  │       │
│   │  Port 8000      │  │  (×1-10, KEDA)  │  │  Port 8080      │       │
│   │                 │  │  (no port)      │  │  Internal only   │       │
│   │  • Webhook recv │  │  • ML Routing   │  │  • Audit log     │       │
│   │  • HMAC valid.  │  │  • CP Pipeline  │  │  • GDPR ops      │       │
│   │  • Redis XADD   │  │  • FAISS Search │  │  • Hallucination │       │
│   │  • Health/Ready │  │  • LLM calls    │  │    review        │       │
│   │  • Prometheus   │  │  • API Gateway  │  │                  │       │
│   └────────┬────────┘  └────────┬────────┘  └────────┬─────────┘       │
│            │                    │                     │                 │
│            └──────────┬─────────┘                     │                 │
│                       │                               │                 │
│            ┌──────────▼──────────┐        ┌───────────▼──────────┐     │
│            │   Redis 7           │        │   PostgreSQL 15      │     │
│            │   • Stream inbound  │        │   • user_mappings    │     │
│            │   • List outbound   │        │   • conversations    │     │
│            │   • Conv state      │        │   • messages         │     │
│            │   • Dist. locks     │        │   • tool_executions  │     │
│            │   • DLQ             │        │   • audit_logs       │     │
│            └─────────────────────┘        │   • hallucination_   │     │
│                                           │     reports          │     │
│            ┌─────────────────────┐        └──────────────────────┘     │
│            │   PVC: /app/.cache  │                                     │
│            │   • Embeddings 41MB │        ┌──────────────────────┐     │
│            │   • Tool metadata   │        │  Azure OpenAI        │     │
│            │   (ReadWriteMany)   │        │  (External)          │     │
│            └─────────────────────┘        └──────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Servisi i portovi

| Servis | Port | Replicas | Scaling | Opis |
|--------|------|----------|---------|------|
| **API** | 8000 | 2 (fixed) | Manual/HPA | WhatsApp webhook, Prometheus metrics |
| **Worker** | — | 1-10 | KEDA (Redis lag) | ML routing, FAISS search, LLM, API pozivi |
| **Admin API** | 8080 | 1 (fixed) | Nema | Audit, GDPR, hallucination review. **Interno!** |
| **PostgreSQL** | 5432 | 1 | StatefulSet | 6 tablica, dual-user security model |
| **Redis** | 6379 | 1 | StatefulSet | Stream, Lists, Hash, distributed locks |

### Container Images

```bash
# Svi koriste ISTI Dockerfile, razlika je u CMD:
docker build -t mobilityone/api:latest .

# CMD po servisu:
# api:       CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# worker:    CMD ["python", "worker.py"]
# admin-api: CMD ["uvicorn", "admin_api:app", "--host", "0.0.0.0", "--port", "8080"]
# migration: CMD ["alembic", "upgrade", "head"]
```

---

## Docker Compose (Development)

### Pokretanje

```bash
# Start osnovni stack (API + Worker + Redis + PostgreSQL)
docker-compose up -d

# Start s Admin API-jem
docker-compose --profile admin up -d

# Start s monitoringom (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Rebuild nakon promjena koda
docker-compose build api worker && docker-compose up -d

# Skaliranje workera
docker-compose up -d --scale worker=3

# Logovi
docker-compose logs -f api worker
```

### Development portovi

| Service | Port | URL |
|---------|------|-----|
| API | 8000 | http://localhost:8000 |
| Admin API | 8088 | http://localhost:8088 |
| PostgreSQL | 5432 | localhost:5432 |
| Redis | 6379 | localhost:6379 |
| Grafana | 3000 | http://localhost:3000 |
| Prometheus | 9090 | http://localhost:9090 |

### Docker Compose — Minimalna konfiguracija

```yaml
# .env datoteka (minimalno za lokalni razvoj):
DATABASE_URL=postgresql+asyncpg://bot_user:localpass@localhost:5432/mobility_db
REDIS_URL=redis://localhost:6379/0
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
MOBILITY_API_URL=https://your-instance.mobilityone.io/
MOBILITY_AUTH_URL=https://your-instance.mobilityone.io/sso/connect/token
MOBILITY_CLIENT_ID=your-client-id
MOBILITY_CLIENT_SECRET=your-client-secret
MOBILITY_TENANT_ID=your-tenant-uuid
GDPR_HASH_SALT=your-32-char-minimum-salt-value-here
```

---

## Kubernetes (Production)

### Deploy koraci

```bash
# Step 1: Storage + Config
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml

# Step 2: Secrets (generiraj najprije)
./k8s/create-sealed-secret.sh
kubectl apply -f k8s/sealed-secrets-generated.yaml

# Step 3: Redis (dev/staging — koristiti managed Redis u produkciji)
kubectl apply -f k8s/redis.yaml

# Step 4: Deploy + database migracija
kubectl apply -f k8s/deployment.yaml
kubectl wait --for=condition=complete job/mobility-db-migrate --timeout=120s

# Step 5: Services, Ingress, Network Policies
kubectl apply -f k8s/service.yaml

# Step 6: Autoscaling (zahtijeva KEDA operator)
kubectl apply -f k8s/keda-autoscaler.yaml

# Step 7: Verifikacija (600s timeout jer FAISS/ML loading traje 60-120s)
kubectl rollout status deployment/mobility-api --timeout=600s
kubectl rollout status deployment/mobility-worker --timeout=600s
```

### Kubernetes manifesti

| Datoteka | Opis |
|----------|------|
| `k8s/pvc.yaml` | PersistentVolumeClaim za `/app/.cache` (embeddings) |
| `k8s/configmap.yaml` | Non-secret konfiguracija (pool sizes, feature flags) |
| `k8s/sealed-secrets-generated.yaml` | Sealed Secrets (šifrirani u gitu) |
| `k8s/redis.yaml` | Redis StatefulSet (dev/staging) |
| `k8s/deployment.yaml` | API, Worker, Admin, Migration Job |
| `k8s/service.yaml` | Services, Ingress, NetworkPolicy |
| `k8s/keda-autoscaler.yaml` | KEDA ScaledObject + ScaledJob + Alerts |

---

## Environment varijable

### Obavezne (REQUIRED)

```bash
# === AZURE OPENAI (LLM + Embeddings) ===
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini        # LLM za routing i odgovore
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002  # Embedding za FAISS

# === DATABASE (dual-user security model) ===
DATABASE_URL=postgresql+asyncpg://bot_user:password@postgres:5432/mobility_db
BOT_DATABASE_URL=postgresql+asyncpg://bot_user:password@postgres:5432/mobility_db
ADMIN_DATABASE_URL=postgresql+asyncpg://admin_user:password@postgres:5432/mobility_db

# === REDIS ===
REDIS_URL=redis://redis:6379/0

# === INFOBIP (WhatsApp) ===
INFOBIP_BASE_URL=your-instance.api.infobip.com
INFOBIP_API_KEY=your-api-key
INFOBIP_SENDER_NUMBER=385xxxxxxxxx
INFOBIP_SECRET_KEY=webhook-signature-key   # Za HMAC-SHA256 validaciju

# === MOBILITY ONE BACKEND (OAuth2) ===
MOBILITY_API_URL=https://your-instance.mobilityone.io/
MOBILITY_AUTH_URL=https://your-instance.mobilityone.io/sso/connect/token
MOBILITY_CLIENT_ID=your-client-id
MOBILITY_CLIENT_SECRET=your-client-secret
MOBILITY_TENANT_ID=your-tenant-uuid

# === SWAGGER SOURCES ===
SWAGGER_URL=https://your-instance.mobilityone.io/automation/swagger/v1.0.0/swagger.json

# === GDPR ===
GDPR_HASH_SALT=your-32-char-minimum-salt-value-here  # Za pseudonimizaciju
```

### Opcionalne

```bash
# === MONITORING ===
SENTRY_DSN=https://xxx@sentry.io/xxx
GRAFANA_PASSWORD=admin

# === ADMIN API ===
ADMIN_TOKEN_1=64-char-hex-token
ADMIN_TOKEN_1_USER=admin.username
ADMIN_ALLOWED_IPS=10.0.0.0/8,192.168.0.0/16

# === PERFORMANCE ===
DB_POOL_SIZE=10                   # Bazne DB konekcije po podu
DB_MAX_OVERFLOW=20                # Extra konekcije za peak load
REDIS_MAX_CONNECTIONS=50
LOG_LEVEL=INFO                    # DEBUG za lokalni razvoj

# === WORKER ===
BURST_MODE=false                  # true za KEDA ScaledJob podove
MAX_MESSAGES=100                  # Burst: exit nakon N poruka
BURST_IDLE_TIMEOUT=300            # Burst: exit nakon 5min idle

# === REDIS SENTINEL (HA) ===
REDIS_SENTINEL_ENABLED=false
REDIS_SENTINEL_HOSTS=sentinel1:26379,sentinel2:26379
REDIS_SENTINEL_MASTER=mymaster

# === MEMORY PROFILING ===
TRACEMALLOC=false                 # true za memory debugging (1-frame tracing)
```

---

## Cache i Embeddings (KRITIČNO!)

### Zašto je ovo kritično?

```
┌────────────────────────────────────────────────────────────────────────┐
│  /app/.cache/tool_embeddings.json = 40.9 MB                            │
│                                                                        │
│  BEZ OVOG FILEA:                                                       │
│  - API startup: ~60-120 sekundi (generira embeddings via Azure API)    │
│  - Troši 950 Azure OpenAI embedding API poziva                         │
│  - Svi API pozivi za vrijeme startupa = dodatni troškovi               │
│                                                                        │
│  S OVIM FILEOM:                                                        │
│  - API startup: <5 sekundi                                             │
│  - 0 Azure API poziva za embeddings                                    │
│  - FAISS indeks gotov za pretragu odmah                                │
└────────────────────────────────────────────────────────────────────────┘
```

### PersistentVolumeClaim

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mobility-cache-pvc
spec:
  accessModes:
    - ReadWriteMany      # Svi podovi čitaju/pišu
  resources:
    requests:
      storage: 100Mi     # 50MB potrebno, 100Mi za headroom
  storageClassName: standard  # Prilagodite vašem clusteru
```

### Mount u svakom deploymentu

```yaml
spec:
  containers:
    - name: api
      volumeMounts:
        - name: cache-volume
          mountPath: /app/.cache
  volumes:
    - name: cache-volume
      persistentVolumeClaim:
        claimName: mobility-cache-pvc
```

### Cache datoteke

| Datoteka | Veličina | Opis | Kritična? |
|----------|----------|------|-----------|
| `tool_embeddings.json` | 40.9 MB | FAISS embedding vektori (1536 dim × 950 alata) | **DA** |
| `swagger_manifest.json` | ~400 B | Cache verzija (hash) | DA |
| `tool_metadata.json` | ~3 MB | Tool definicije i dokumentacija | DA |
| `error_learning.json` | ~100 KB | Naučene greške iz prethodnih poziva | NE |
| `api_capabilities.json` | ~10 KB | API capabilities cache | NE |

### ML modeli (na disku, u containeru)

```
models/
├── intent/
│   ├── tfidf_vectorizer.pkl        # TF-IDF vektorizator (45+ klasa)
│   ├── logistic_regression.pkl     # Logistic Regression model
│   ├── label_encoder.pkl           # Label encoder
│   └── cp_calibration.json         # Conformal Prediction q̂ prag
└── query_type/
    ├── tfidf_vectorizer.pkl        # Query type TF-IDF
    ├── logistic_regression.pkl     # Query type LR
    ├── label_encoder.pkl
    └── cp_calibration.json         # CP q̂ za query type
```

**Napomena:** ML modeli su ugrađeni u Docker image (`COPY models/ /app/models/`). Ne trebaju PVC — mijenjaju se samo pri retrainingu (rijetko).

---

## Database Setup

### Dual-User Security Model

```
┌─────────────────────────────────────────────────────────────────┐
│  bot_user (LIMITED)             │  admin_user (FULL)            │
├─────────────────────────────────┼───────────────────────────────┤
│  ✅ SELECT/INSERT/UPDATE/DELETE │  ✅ ALL PRIVILEGES             │
│  na operativne tablice          │  ✅ CREATE/ALTER tablice       │
│  (user_mappings, conversations, │  ✅ audit_logs (full access)   │
│   messages, tool_executions)    │  ✅ hallucination_reports (RW) │
│  ✅ INSERT-only na              │  ✅ Alembic migracije          │
│  hallucination_reports          │                               │
│  ❌ NE MOŽE pristupiti          │                               │
│  audit_logs tablici             │                               │
└─────────────────────────────────┴───────────────────────────────┘
```

### Inicijalizacija

```bash
# 1. PostgreSQL container → init-db.sh (automatski)
#    Kreira: mobility_db, bot_user, admin_user, GRANT-ove

# 2. Migration Job (koristi ADMIN_DATABASE_URL)
kubectl run migration --image=mobilityone/migration:latest --restart=Never
# ili:
kubectl wait --for=condition=complete job/mobility-db-migrate --timeout=120s

# 3. API/Worker startaju (koriste BOT_DATABASE_URL)
```

### Connection Pooling

```python
# database.py:
pool_size = 10           # Bazne konekcije (DB_POOL_SIZE)
max_overflow = 20        # Peak = 30 total po podu (DB_MAX_OVERFLOW)
pool_recycle = 3600      # Recycle svaki sat
pool_pre_ping = True     # Verify alive before use
pool_use_lifo = True     # Reuse recent connections first
```

**KRITIČNO:** 8 max podova × 30 = 240 burst konekcija. PostgreSQL default `max_connections=100` **NIJE DOVOLJNO**. Opcije:
1. Koristiti PgBouncer (preporučeno za produkciju)
2. Povećati `max_connections` u PostgreSQL konfiguraciji
3. Smanjiti `DB_POOL_SIZE` i `DB_MAX_OVERFLOW`

---

## Startup Sequence

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STARTUP ORDER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. PostgreSQL    ───────────────────────────────────────→  Ready       │
│     └── init-db.sh creates users & database                            │
│                                                                         │
│  2. Redis         ───────────────────────────────────────→  Ready       │
│                                                                         │
│  3. Migration Job ───────────────────────────────────────→  Complete    │
│     └── alembic upgrade head (creates/updates tables)                  │
│                                                                         │
│  4. API Server    ─────────┬─────────────────────────────→  Ready       │
│     ├── ML modeli:         │  (TF-IDF, LogReg, CP = <1s)              │
│     ├── FAISS indeks:      │                                           │
│     │   └── Cache postoji? │  DA → load (2-5s)                         │
│     │                      │  NE → generate embeddings (60-120s)       │
│     └── Tool registry:     │  Parse tool configs (~2s)                 │
│                            │                                           │
│  5. Worker        ─────────┴─────────────────────────────→  Ready       │
│     ├── Čeka API health check                                          │
│     ├── Učitava FAISS iz shared cache (ne regenerira!)                 │
│     └── Consumer group ready (XREADGROUP)                              │
│                                                                         │
│  6. Admin API     ───────────────────────────────────────→  Ready       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Startup probes** su konfigurirani za 60 × 10s = 10 minuta. Pri 50m CPU request-u (burst do 500m), FAISS/ML loading traje 60-120s. Bez startup probe-a, liveness probe ubija pod prije nego je spreman.

---

## Health Checks

### API Service (port 8000)

```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 10
  failureThreshold: 60    # 60 × 10s = 10 min za startup

livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

### Worker Service (no port)

```yaml
startupProbe:
  exec:
    command: ["pgrep", "-f", "python worker.py"]
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 60

livenessProbe:
  exec:
    command: ["pgrep", "-f", "python worker.py"]
  initialDelaySeconds: 30
  periodSeconds: 30
  failureThreshold: 3
```

### Admin API (port 8080)

```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 30

livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30
```

### Health vs Ready

| Endpoint | Što provjerava | Kad pada |
|----------|----------------|----------|
| `/health` | DB ping + Redis ping + registry status | DB ili Redis down |
| `/ready` | DB query + Redis **write** test (SET→GET→DEL) + registry loaded | DB/Redis degraded, embeddings not loaded |

**Važno:** `/ready` koristi Redis `SET`→`GET`→`DEL` cycle (ne samo `PING`) jer PING uspije i na read-only replica.

---

## Autoscaling (KEDA)

### Zašto KEDA umjesto HPA?

```
AI workload = I/O bound, NE CPU bound!

Čekanje Azure OpenAI odgovora:
  CPU: ~1% (izmjereno)
  Latency: 2-5 sekundi
  HPA NE BI SKALIRAO jer je CPU nizak!

KEDA skalira na Redis Stream LAG:
  Ako 10+ poruka čeka → dodaj worker
  Ako <5 poruka → smanji workere
```

### Instalacija

```bash
# Install KEDA operator
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda --namespace keda --create-namespace

# Apply ScaledObject (regular workers) + ScaledJob (burst workers)
kubectl apply -f k8s/keda-autoscaler.yaml
```

### ScaledObject (Regular Workers)

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: mobility-worker
spec:
  scaleTargetRef:
    name: mobility-worker
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
    - type: redis-streams
      metadata:
        address: redis:6379
        stream: whatsapp_stream_inbound
        consumerGroup: mobility_workers
        lagCount: "10"      # Scale up kad lag > 10
```

### ScaledJob (Burst Workers)

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledJob
metadata:
  name: mobility-burst-worker
spec:
  maxReplicaCount: 5
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  triggers:
    - type: redis-streams
      metadata:
        lagCount: "50"      # Burst kad lag > 50
  jobTargetRef:
    template:
      spec:
        containers:
          - name: burst-worker
            env:
              - name: BURST_MODE
                value: "true"
              - name: MAX_MESSAGES
                value: "100"
```

### Monitoring Alerts

| Alert | Uvjet | Opis |
|-------|-------|------|
| `CPUThrottlingHigh` | >25% CFS throttling, 5min | Pod nema dovoljno CPU |
| `MemoryHighWater` | >85% od 512Mi, 2min | Blizu OOM kill-a |
| `WorkerQueueBacklog` | >100 pending msg, 5min | Queue raste brže nego se procesira |
| `KedaNotScaling` | Queue >50, KEDA inactive, 2min | Autoscaler ne reagira |

---

## Network i Security

### Ingress pravila

```yaml
# API — Public (samo WhatsApp webhook)
- host: bot.yourdomain.com
  paths:
    - path: /webhook
      service: api-service
      port: 8000

# Admin API — Internal Only!
- host: admin.internal.yourdomain.com  # VPN/Internal DNS only!
  paths:
    - path: /
      service: admin-service
      port: 8080
```

### Network Policies

```yaml
# Admin API može biti pristupljen samo iz internog namespace-a
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: admin-api-internal-only
spec:
  podSelector:
    matchLabels:
      app: admin-api
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: internal
  policyTypes:
    - Ingress
```

### Security Context (svi podovi)

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  readOnlyRootFilesystem: true     # /tmp via tmpfs emptyDir
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
```

### Security Headers (main.py middleware)

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Cache-Control: no-store
```

---

## Resource Limits

### Izmjerene vrijednosti (2026-03-18)

| Component | CPU req/limit | Memory req/limit | **Measured RSS** | **Measured CPU** | Grace Period |
|-----------|--------------|-------------------|-----------------|-----------------|-------------|
| API (×2) | 50m / 500m | 400Mi / 512Mi | **323.5 MB** | 0.10% | 30s |
| Worker | 50m / 500m | 416Mi / 512Mi | **345.1 MB** | 0.07% | 120s |
| Admin | 25m / 200m | 96Mi / 192Mi | **65.7 MB** | 0.10% | 30s |
| Burst Job | 50m / 500m | 416Mi / 512Mi | **~345 MB** | — | 30s |
| Redis | 25m / 250m | 128Mi / 192Mi | **10.3 MB** | 0.34% | — |
| **TOTAL** | **200m** | **1440Mi** | — | — | — |

### QoS: Burstable

Podovi koriste **Burstable QoS** (`requests < limits`) jer:
- Bot je **I/O bound** (čeka Azure OpenAI API odgovor), ne CPU bound
- Izmjereni CPU: <1% pri normalnom opterećenju
- Burstable minimizira rezervirane resurse na shared clusteru
- Ako eviction postane problem, povisiti `requests` na razinu `limits` za **Guaranteed QoS**

### Memory Profiling

```bash
# Uključi memory tracing (lagani overhead):
TRACEMALLOC=true

# Worker automatski logira top-5 alokatora kad RSS > 800MB
# Peak memory pri 20 concurrent poruka: ~405MB (79% od 512Mi limita)
```

### Memory Fragmentation

Burst workeri (KEDA ScaledJob) prirodno recikliraju svako 100 poruka ili 5min idle — potpuno čiste fragmentaciju. Za long-lived regularne workere, ako se primijeti memory drift preko tjedana:

```bash
kubectl rollout restart deployment/mobility-worker
```

---

## Backup Strategy

### PostgreSQL

```bash
# Dnevni backup (CronJob):
kubectl create cronjob pg-backup \
  --image=postgres:15-alpine \
  --schedule="0 3 * * *" \
  -- pg_dump -h postgres -U admin_user mobility_db > /backup/$(date +%Y%m%d).sql

# Manual backup:
kubectl exec -it postgres-0 -- pg_dump -U admin_user mobility_db > backup.sql
```

### Cache (Embeddings)

```bash
# Opcionalno — embeddings se mogu regenerirati (traje 60-120s):
kubectl cp api-pod:/app/.cache/tool_embeddings.json ./backup/embeddings.json
```

### Redis

```bash
# Redis AOF + RDB persistence na StatefulSet volumenu
# Za manual backup:
kubectl exec -it redis-0 -- redis-cli BGSAVE
kubectl cp redis-0:/data/dump.rdb ./backup/redis-dump.rdb
```

---

## Monitoring

### Prometheus Metrics

```
GET /metrics  →  Prometheus format

Metrike:
  http_requests_total{method, endpoint, status}     # Counter
  http_request_duration_seconds{method, endpoint}    # Histogram
```

**Excludirane putanje:** `/health`, `/ready`, `/metrics` (probe/scraper noise).

### Grafana Dashboards

Ako se koristi `--profile monitoring` (Docker Compose):

| Dashboard | URL | Opis |
|-----------|-----|------|
| API Performance | http://localhost:3000 | Request rate, latency, errors |
| Worker Status | http://localhost:3000 | Queue lag, processing rate |
| Redis | http://localhost:3000 | Memory, connections, stream stats |

### Logovi

```bash
# API logovi:
kubectl logs -f deployment/mobility-api

# Worker logovi:
kubectl logs -f deployment/mobility-worker

# Admin logovi:
kubectl logs -f deployment/mobility-admin

# Filter po razini:
kubectl logs deployment/mobility-worker | grep ERROR
```

Logovi koriste **PIIScrubFilter** — svi telefonski brojevi, email adrese i OIB-ovi automatski maskirani.

---

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/ci.yml — pokreće se na svakom push/PR:

jobs:
  lint:
    - ruff check .
    - ruff format --check .
    - bandit -r services/ -q
    - mypy services/ (informational)

  test:
    - pytest tests/ --cov --cov-fail-under=85
    - python -c "import ast; ast.parse(open('main.py').read())"  # Syntax check

  build:
    - docker build -t mobilityone/api:latest .
    - trivy image mobilityone/api:latest  # Vulnerability scan
    - pip-audit  # Dependency CVE check

  deploy (manual trigger):
    - kubectl apply -f k8s/
    - kubectl rollout status deployment/mobility-api --timeout=600s
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml:
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
      - id: ruff-format
```

---

## Troubleshooting

### API Startup spor (>2 min)

```bash
# Provjeri ima li embeddings cache:
kubectl exec -it api-pod -- ls -la /app/.cache/

# Ako nema tool_embeddings.json → generira se (~60-120s)
# Rješenje: Osiguraj PersistentVolumeClaim
```

### Worker ne procesira poruke

```bash
# Provjeri Redis stream:
kubectl exec -it redis-pod -- redis-cli XINFO GROUPS whatsapp_stream_inbound

# Provjeri pending poruke:
kubectl exec -it redis-pod -- redis-cli XPENDING whatsapp_stream_inbound mobility_workers

# Provjeri worker logove:
kubectl logs -f deployment/mobility-worker
```

### Database Connection Errors

```bash
# Provjeri DATABASE_URL:
kubectl exec -it api-pod -- env | grep DATABASE

# Test konekcije:
kubectl exec -it api-pod -- python -c "
from database import engine
import asyncio
async def test():
    async with engine.connect() as conn:
        result = await conn.execute(text('SELECT 1'))
        print('DB OK:', result.scalar())
asyncio.run(test())
"

# Pool exhaustion — provjeri pool stats:
kubectl exec -it api-pod -- python -c "
from database import engine
print('Pool size:', engine.pool.size())
print('Checked out:', engine.pool.checkedout())
print('Overflow:', engine.pool.overflow())
"
```

### Health Check Fails

```bash
# Manual health check:
kubectl exec -it api-pod -- curl -f http://localhost:8000/health
kubectl exec -it api-pod -- curl -f http://localhost:8000/ready

# Provjeri logove:
kubectl logs api-pod --tail=100
```

### KEDA ne skalira

```bash
# Provjeri KEDA operator:
kubectl get scaledobject mobility-worker -o yaml
kubectl describe scaledobject mobility-worker

# Provjeri Redis lag:
kubectl exec -it redis-pod -- redis-cli XINFO GROUPS whatsapp_stream_inbound

# Provjeri KEDA logove:
kubectl logs -n keda deployment/keda-operator --tail=50
```

### OOM Kill

```bash
# Provjeri events:
kubectl get events --field-selector reason=OOMKilled

# Provjeri memory usage:
kubectl top pod

# Ako se ponavlja, povećaj memory limit u deployment.yaml
# ili uključi TRACEMALLOC=true za dijagnostiku
```

### Graceful Shutdown problemi

```bash
# Worker grace period mora biti > 90s (message timeout):
# deployment.yaml: terminationGracePeriodSeconds: 120

# Provjeri da APP_STOPPING flag radi:
kubectl exec -it api-pod -- python -c "from main import APP_STOPPING; print(APP_STOPPING)"
```

---

## Production Checklist

### Pre-Deploy

- [ ] PersistentVolumeClaim za `/app/.cache` kreiran i bound
- [ ] Sealed Secrets kreirani (DATABASE_URL, API keys, GDPR_HASH_SALT)
- [ ] PostgreSQL running s `max_connections` ≥ 240 (ili PgBouncer)
- [ ] Redis running s AOF persistence
- [ ] Migration Job completed (`alembic upgrade head`)
- [ ] GDPR_HASH_SALT ≥ 32 znakova

### Post-Deploy

- [ ] API health check prolazi: `curl -f http://bot.domain.com/health`
- [ ] API readiness prolazi: `curl -f http://bot.domain.com/ready`
- [ ] Worker procesira poruke (provjeri Redis stream lag)
- [ ] Embeddings loadani (startup <10s nakon prvog pokretanja)
- [ ] WhatsApp webhook registriran kod Infobip-a
- [ ] KEDA ScaledObject active

### Sigurnost

- [ ] Admin API **NIJE** izložen internetu (samo VPN/internal)
- [ ] bot_user ima ograničene DB permissions
- [ ] Secrets u K8s Secrets (ne ConfigMaps, ne env files)
- [ ] Network Policies primijenjene
- [ ] TLS uključen na Ingress-u
- [ ] Security headers middleware aktivan
- [ ] PIIScrubFilter aktivan na oba procesa (API + Worker)

### Monitoring

- [ ] Prometheus scrape aktivan (`/metrics` endpoint)
- [ ] KEDA alerts konfigurirani (CPU throttling, memory, queue backlog)
- [ ] Log aggregation konfiguriran (ELK/Loki/CloudWatch)
- [ ] `scripts/verify_production_readiness.py` prolazi sve provjere

---

## Kontakt i dokumentacija

| Dokument | Opis |
|----------|------|
| [README.md](README.md) | Pregled projekta |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Detaljna tehnička arhitektura |
| [SECURITY.md](SECURITY.md) | Sigurnost, GDPR, EU AI Act |
| [CHANGELOG.md](CHANGELOG.md) | Povijest verzija |

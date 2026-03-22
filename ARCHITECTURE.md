# Tehnička Arhitektura — MobilityOne WhatsApp Bot

> Verzija: 12.0 | Ažurirano: 2026-03-22

## Sadržaj

- [Pregled sustava](#pregled-sustava)
- [Procesi i komunikacija](#procesi-i-komunikacija)
- [Tok poruke (Message Flow)](#tok-poruke)
- [3-Tier Routing Pipeline](#3-tier-routing-pipeline)
- [ML Intent Classifier](#ml-intent-classifier)
- [Conformal Prediction (CP)](#conformal-prediction)
- [FAISS Semantic Search](#faiss-semantic-search)
- [Unified Search Pipeline](#unified-search-pipeline)
- [Decision Engine](#decision-engine)
- [LLM Reranker](#llm-reranker)
- [Conversation Engine](#conversation-engine)
- [API Gateway](#api-gateway)
- [Baza podataka](#baza-podataka)
- [Redis arhitektura](#redis-arhitektura)
- [Sigurnosna arhitektura](#sigurnosna-arhitektura)
- [Observability](#observability)

---

## Pregled sustava

```
                          ┌─────────────────┐
                          │  WhatsApp User   │
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │   Infobip API    │
                          │  (WhatsApp GW)   │
                          └────────┬─────────┘
                                   │ HTTPS webhook
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                        KUBERNETES CLUSTER                              │
│                                                                        │
│  ┌─────────────────┐    ┌───────────────────┐    ┌─────────────────┐  │
│  │  FastAPI (×2)   │    │   Worker (×1-10)  │    │  Admin API (×1) │  │
│  │  Port 8000      │    │   (no port)       │    │  Port 8080      │  │
│  │  main.py        │    │   worker.py       │    │  admin_api.py   │  │
│  │                 │    │                   │    │                 │  │
│  │  • Webhook      │    │  • ML Classifier  │    │  • Audit log    │  │
│  │  • Signature    │    │  • CP Routing     │    │  • GDPR ops     │  │
│  │    validation   │    │  • FAISS Search   │    │  • Hallucination│  │
│  │  • Redis XADD   │    │  • LLM Routing    │    │    review       │  │
│  │  • Health/Ready │    │  • API calls      │    │  • Monitoring   │  │
│  │  • Prometheus   │    │  • Response fmt   │    │                 │  │
│  └────────┬────────┘    └───────┬───────────┘    └────────┬────────┘  │
│           │                     │                         │           │
│           └─────────┬───────────┘                         │           │
│                     │                                     │           │
│           ┌─────────▼──────────┐              ┌───────────▼────────┐  │
│           │   Redis 7          │              │   PostgreSQL 15    │  │
│           │                    │              │                    │  │
│           │  • Stream (inbound)│              │  • user_mappings   │  │
│           │  • List (outbound) │              │  • conversations   │  │
│           │  • Conv state      │              │  • messages        │  │
│           │  • User context    │              │  • tool_executions │  │
│           │  • Dist. locks     │              │  • audit_logs      │  │
│           │  • DLQ             │              │  • hallucination_  │  │
│           └────────────────────┘              │    reports         │  │
│                                               └────────────────────┘  │
│           ┌────────────────────┐              ┌────────────────────┐   │
│           │  Shared Volume     │              │  Azure OpenAI      │   │
│           │  /app/.cache       │              │  (External)        │   │
│           │  • Embeddings 41MB │              │  • gpt-4o-mini     │   │
│           │  • Tool metadata   │              │  • ada-002 embed   │   │
│           └────────────────────┘              └────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Procesi i komunikacija

Sustav se sastoji od tri procesa koji komuniciraju isključivo putem Redis-a:

| Proces | Datoteka | Port | Uloga |
|--------|----------|------|-------|
| **API Server** | `main.py` | 8000 | FastAPI webhook — prima WhatsApp poruke od Infobip-a, validira HMAC potpis, stavlja u Redis Stream |
| **Worker** | `worker.py` | — | Async processor — čita iz Redis Stream-a, ML routing, API pozivi, šalje odgovore na Redis List |
| **Admin API** | `admin_api.py` | 8080 | Admin panel — pregled halucinacija, GDPR operacije, audit log. Interno, nije izložen internetu |

### Komunikacijski kanali

```
API Server ──XADD──→ Redis Stream (whatsapp_stream_inbound) ──XREADGROUP──→ Worker
Worker ──RPUSH──→ Redis List (outbound:{phone}) ──BLPOP──→ Worker (sender loop)
Worker ──HTTP──→ Infobip API ──→ WhatsApp
```

Svaki proces je statelss — svo stanje je u Redis-u ili PostgreSQL-u. To omogućava horizontalno skaliranje Worker podova putem KEDA autoscalera.

---

## Tok poruke

### Inbound (WhatsApp → Bot)

```
1. Infobip šalje POST /webhook/inbound na API Server
2. webhook_simple.py:
   a. Validira HMAC-SHA256 potpis (INFOBIP_SECRET_KEY)
   b. Parsira poruku (text, sender, message_id)
   c. XADD u Redis Stream s TTL=24h
   d. Return 200 OK (Infobip ne retryja ako je 200)
   e. Ako XADD fail → 3-Tier DLQ (Redis → File → stderr)

3. Worker (XREADGROUP consumer group):
   a. Acquire distributed lock (SET NX EX, Lua release)
   b. asyncio.Semaphore(10) za kontrolu concurrency-ja
   c. 90s timeout (asyncio.wait_for)
   d. MessageEngine.process_message()
   e. XACK tek nakon uspješnog RPUSH outbound
```

### Processing (MessageEngine)

```
4. UserHandler: Identifikacija korisnika
   a. Redis cache lookup (user_context:{phone})
   b. Fallback: MobilityOne API → UserMapping tablica
   c. GDPR consent gate (guest → blokiraj, valid → provjeri consent flag)

5. ConversationManager: State machine
   a. Redis hash (conv_state:{phone})
   b. States: idle → active → awaiting_confirmation → ...
   c. Multi-step flows: booking, mileage, case

6. Routing (UnifiedRouter.route()):
   a. Pattern checks (greetings, exits, show more, selections)
   b. ML Fast Path → QueryRouter (TF-IDF, <1ms)
   c. CP Mediation → LLM reranker (2-5 candidates)
   d. Full LLM Routing → UnifiedSearch (FAISS + boosts → LLM)

7. Tool Execution:
   a. ToolExecutor → API Gateway → MobilityOne API
   b. OAuth2 token management (auto-refresh)
   c. Circuit breaker (5 failures → 60s open)
   d. Retry with exponential backoff (3 attempts)

8. Response Formatting:
   a. ResponseExtractor: Parsira API odgovor
   b. ResponseFormatter: Formatira za WhatsApp (markdown → plain text)
   c. Pagination: Max 5 stavki, "pokaži još" za nastavak
```

### Outbound (Bot → WhatsApp)

```
9. RPUSH outbound:{phone} s formatiranom porukom
10. Worker outbound loop (BLPOP):
    a. Čita poruku iz outbound liste
    b. HTTP POST na Infobip Send Message API
    c. Rate limiting (max 30 msg/s po broju)
```

---

## 3-Tier Routing Pipeline

Routing pipeline odlučuje koji od 950+ API alata koristiti za korisnikov upit. Tri razine s postupno rastućom latencijom:

```
                    ┌─────────────────────────────┐
                    │        User Query            │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │     ML Intent Classifier     │
                    │  (TF-IDF + LogReg, <1ms)     │
                    │  45+ intent klasa             │
                    └──────────────┬───────────────┘
                                   │
                         ┌─────────┴─────────┐
                         │                   │
              ┌──────────▼──────────┐ ┌──────▼──────────────────┐
              │  CP set = 1         │ │  CP set = 2-5           │
              │  confidence ≥ 0.85  │ │  (Mediation Path)       │
              │                     │ │                         │
              │  ★ FAST PATH        │ │  LLM Reranker           │
              │  0 LLM poziva       │ │  (gpt-4o-mini)          │
              │  ~85% upita         │ │  bira iz CP skupa       │
              │  latencija <10ms    │ │  latencija ~200ms       │
              └─────────────────────┘ └─────────────────────────┘
                                              │
                                   CP set > 5 ili
                                   classifier unsure
                                              │
                              ┌────────────────▼───────────────┐
                              │       FULL LLM ROUTING         │
                              │                                │
                              │  1. UnifiedSearch (FAISS+BM25) │
                              │     → top 20 candidates        │
                              │  2. Ambiguity Detection        │
                              │  3. Single LLM call            │
                              │     (gpt-4o-mini, 20+ tools)   │
                              │  latencija 2-3s                │
                              └────────────────────────────────┘
```

### Tier 1: Fast Path (ML)

- **Modul:** `services/query_router.py` + `services/intent_classifier.py`
- **Algoritam:** TF-IDF vektorizacija + Logistic Regression
- **Klase:** 45+ intent klasa (svaka mapirana na točno jedan API alat)
- **Odluka:** `DecisionEngine.decide_with_cp()` s `ML_FAST_PATH` boundary
- **Kriterij:** CP prediction set = 1 alat I confidence ≥ 0.85
- **Latencija:** <1ms (nema LLM poziva, nema mrežnih poziva)
- **Pokriva:** ~85% svih upita

### Tier 2: Mediation Path (CP + LLM Reranker)

- **Modul:** `services/unified_router.py` (`_mediation_route()`)
- **Algoritam:** Conformal Prediction sužava na 2-5 kandidata, LLM reranker bira
- **Ulaz:** CP PredictionSet s labels i vjerojatnostima
- **Izlaz:** Jedan alat s LLM obrazloženjem
- **Latencija:** ~200ms (mali LLM prompt, 2-5 alata)
- **Garancija:** 98% coverage (pravi alat je u CP skupu s ≥98% vjerojatnosti)

### Tier 3: Full LLM Routing

- **Modul:** `services/unified_router.py` (`route()`)
- **Pipeline:** UnifiedSearch → Ambiguity Detection → Single LLM Call
- **Kandidati:** 20+ alata iz FAISS pretraga + PRIMARY_TOOLS
- **Disambiguation:** Ako FAISS vrati ambiguouzan rezultat (isti suffix, slični scores), LLM dobiva hint
- **Latencija:** 2-3 sekunde (uključuje Azure OpenAI API poziv)

---

## ML Intent Classifier

### Arhitektura

```
services/intent_classifier.py (~1310 linija)
├── IntentClassifier
│   ├── TF-IDF + LogReg pipeline (primary)
│   ├── SBERT + LogReg pipeline (secondary, optional)
│   └── Azure Embedding + LogReg pipeline (tertiary, optional)
│
├── QueryTypeClassifierML
│   └── TF-IDF + LogReg za query type detection (12 klasa)
│
├── detect_action_intent() → ActionIntent enum (GET/POST/PUT/DELETE)
│
└── predict_with_ensemble() → IntentPrediction
    └── Majority voting ili highest-confidence izbor
```

### Intent klase (45+)

Svaka intent klasa mapirana je na jedan API alat putem `tool_routing.py`:

```python
# Primjer iz tool_routing.py:
INTENT_CONFIG = {
    "vehicle_list": {
        "tool": "get_Vehicles",
        "extract_fields": ["registrationNumber", "vin", "status"],
        "response_template": "Pronašao sam {count} vozila...",
    },
    "expense_summary": {
        "tool": "get_Expenses_Agg",
        "extract_fields": ["totalAmount", "count"],
        "response_template": "Ukupno troškova: {totalAmount}...",
    },
    # ... 45+ intent mappings
}
```

### Query Type klase (12)

| Query Type | Suffix | Primjer upita |
|------------|--------|---------------|
| `by_id` | `_id` | "Pokaži vozilo ABC-123" |
| `documents` | `_documents` | "Dokumenti za vozilo" |
| `metadata` | `_metadata` | "Koji su statusi za vozila?" |
| `aggregation` | `_Agg` | "Ukupni troškovi po mjesecu" |
| `group_by` | `_GroupBy` | "Vozila grupirana po statusu" |
| `project_to` | `_ProjectTo` | "Samo imena i registracije" |
| `latest` | `Latest` | "Najnovije vožnje" |
| `default` | — | "Prikaži sva vozila" |
| `delete` | `_DeleteByCriteria` | "Obriši neaktivne korisnike" |
| `multipatch` | `_multipatch` | "Ažuriraj sve statusе" |
| `create` | (POST) | "Dodaj novo vozilo" |
| `update` | (PUT) | "Promijeni registraciju" |

### Training podaci

| Dataset | Lokacija | Primjeri | Namjena |
|---------|----------|----------|---------|
| Intent training | `data/training/intent_training.jsonl` | ~2500 | Training TF-IDF modela |
| Intent test | `data/training/intent_test.jsonl` | 397 | Holdout evaluacija + CP kalibracija |
| Query type | `data/training/query_type.jsonl` | 5830 | Training + 80/20 za CP |

### Modeli na disku

```
models/
├── intent/
│   ├── tfidf_vectorizer.pkl      # TF-IDF matrica
│   ├── logistic_regression.pkl   # LR model
│   ├── label_encoder.pkl         # Label encoder (45+ klasa)
│   └── cp_calibration.json       # CP q_hat za intent
└── query_type/
    ├── tfidf_vectorizer.pkl
    ├── logistic_regression.pkl
    ├── label_encoder.pkl
    └── cp_calibration.json       # CP q_hat za query_type
```

---

## Conformal Prediction

### Matematička osnova

Conformal Prediction (CP) koristi **Adaptive Prediction Sets (APS)** algoritam za konstrukciju skupova kandidata s garancijom pokrivenosti.

#### APS algoritam

Za zadani upit s vjerojatnostima `p₁ ≥ p₂ ≥ ... ≥ pₖ`:

1. Sortiraj klase po vjerojatnosti (opadajuće)
2. Akumuliraj dok kumulativna suma ≥ `1 - q̂`
3. Sve akumulirane klase čine prediction set

#### Kalibracija `q̂`

```
q̂ = quantile(APS_scores, ⌈(n+1)(1-α)/n⌉)

gdje:
- APS_score(x, y_true) = kumulativna vjerojatnost do y_true
- n = veličina kalibracijskog skupa
- α = 0.02 (željena greška = 2%)
```

**Garancija:** Za buduće podatke iz iste distribucije:
```
P(y_true ∈ PredictionSet) ≥ 1 - α = 0.98
```

### Implementacija

```python
# services/dynamic_threshold.py

@dataclass(frozen=True, slots=True)
class PredictionSet:
    labels: tuple           # Kandidati sortirani po vjerojatnosti
    probabilities: tuple    # Odgovarajuće vjerojatnosti
    coverage: float         # Ciljana pokrivenost (0.98)
    q_hat: float            # Kalibrirani prag

    @classmethod
    def from_probabilities(cls, probs, label_names, q_hat, coverage=0.98):
        # APS: sortiraj, akumuliraj do 1 - q_hat
        ...
```

### CP → Routing odluka

```python
# DecisionEngine.decide_with_cp():

if prediction_set.size == 1 and base.is_accept:
    return ACCEPT          # Fast Path — 1 kandidat, ML siguran
elif prediction_set.size <= 5:
    return BOOST           # Mediation — LLM bira iz malog skupa
else:
    return DEFER           # Full Search — previše kandidata
```

### Kalibracijska skripta

```bash
python scripts/calibrate_conformal.py

# Rezultat:
# - models/intent/cp_calibration.json (q_hat za intent, 397 calibration primjera)
# - models/query_type/cp_calibration.json (q_hat za query_type, ~1166 calibration primjera)
# - config/faiss_margin_calibration.json (optimalni FAISS margin threshold)
```

---

## FAISS Semantic Search

### Arhitektura

```
services/faiss_vector_store.py
├── FAISSVectorStore
│   ├── _index: faiss.IndexFlatIP   # Exact cosine similarity
│   ├── _tool_ids: List[str]        # FAISS index → tool_id mapping
│   ├── _tool_methods: Dict         # tool_id → HTTP method
│   └── _embeddings_cache: Dict     # Disk cache (41MB)
│
├── Embedding model: Azure text-embedding-ada-002 (1536 dimenzija)
├── Indeks: 950+ tool opisa embediranih iz tool_documentation.json
├── Pretraga: top-K cosine similarity, K=20 default
└── Latencija: 1-5ms (in-memory, no I/O)
```

### Embedding izvor

Tool opisi dolaze iz `config/tool_documentation.json` (~1.5MB, 950 alata). Svaki alat ima:

```json
{
  "get_Vehicles": {
    "purpose": "Dohvaća popis svih vozila u floti",
    "example_queries_hr": [
      "pokaži mi sva vozila",
      "lista vozila u floti",
      "koji automobili su dostupni"
    ],
    "parameters": { ... },
    "response_fields": [ ... ]
  }
}
```

### Cache strategija

```
Startup s cache-om (~5s):
  .cache/tool_embeddings.json (41MB) → numpy array → FAISS index

Startup bez cache-a (~60-120s):
  tool_documentation.json → Azure Ada-002 API (950 poziva) → embeddings → FAISS index → save cache
```

Cache se dijeli između svih podova putem PersistentVolumeClaim (`/app/.cache`).

---

## Unified Search Pipeline

`services/unified_search.py` — 6-koračni pipeline koji kombinira ML i vektorsku pretragu:

```
┌─────────────────────────────────────────────────────────┐
│                    UnifiedSearch.search()                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. TFI Override                                        │
│     detect_action_intent() → GET/POST/PUT/DELETE        │
│     Ako intent = POST/PUT/DELETE → filtrira alate       │
│                                                         │
│  2. FAISS Semantic Search                               │
│     query → Ada-002 embedding → FAISS top-20            │
│     Cosine similarity (1536 dim, <5ms)                  │
│                                                         │
│  3. BM25 Lexical Search (dopuna)                        │
│     Token matching za kratke, specifične upite           │
│                                                         │
│  4. Exact Match Index (O(1))                            │
│     Pre-computed dict: example_query → tool_id           │
│     Ako upit točno odgovara primjeru → boost +0.3       │
│                                                         │
│  5. Additive Boosts                                     │
│     ├── Category boost (tool_categories.json)           │
│     ├── Query type boost (_Agg, _documents, _metadata)  │
│     ├── Entity detection boost (HR keywords → entity)   │
│     ├── Possessive boost ("moji", "moja" → user tools)  │
│     └── Profile query boost ("tko sam ja")              │
│                                                         │
│  6. Sort by final score, return top-K                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Entity Detection

`services/entity_detector.py` detektira entitete iz hrvatskog teksta:

| Ključna riječ | Entitet | Primjer upita |
|---------------|---------|---------------|
| vozil, auto, flot | Vehicles | "prikaži vozila" |
| kompanij, tvrtk | Companies | "lista kompanija" |
| osob, korisnik | Persons | "svi zaposlenici" |
| trošk, račun | Expenses | "ukupni troškovi" |
| putovanj, vožnj | Trips | "moje vožnje" |
| slučaj, šteta, kvar | Cases | "otvori slučaj" |
| rezervacij, booking | VehicleCalendar | "rezerviraj vozilo" |

Normalizacija dijakritika (`č→c, ć→c, ž→z, š→s, đ→d`) omogućava fuzzy matching.

---

## Decision Engine

`services/dynamic_threshold.py` — matematički model za routing odluke:

### ClassificationSignal

Neizmjenjivi (frozen) sažetak distribucije vjerojatnosti:

```python
@dataclass(frozen=True, slots=True)
class ClassificationSignal:
    confidence: float      # Max vjerojatnost (p_max)
    margin: float          # p_max - p_second (margin of victory)
    entropy_norm: float    # H(p) / log(K), normalizirana entropija [0,1]
    n_classes: int         # Broj klasa (K) — MORA biti eksplicitno proslijeđen
```

### Effective Score

```
effective_score = confidence × (1 − α × entropy_norm)

α = 0.0 → čisti confidence (backward-kompatibilno)
α > 0   → entropija penalizira "razmazane" distribucije
```

### Decision Boundaries

| Boundary | Accept | Boost | Defer | Kontekst |
|----------|--------|-------|-------|----------|
| `ML_FAST_PATH` | ≥ 0.85 | 0.60-0.85 | < 0.60 | QueryRouter intent routing |
| `INTENT_FILTER` | ≥ 0.70 | 0.40-0.70 | < 0.40 | UnifiedSearch intent filter |
| `QUERY_TYPE` | ≥ 0.70 | 0.40-0.70 | < 0.40 | Query type classification |
| `POSSESSIVE` | ≥ 0.80 | 0.50-0.80 | < 0.50 | "Moji/moja" detection |
| `MUTATION` | ≥ 0.80 | 0.50-0.80 | < 0.50 | POST/PUT/DELETE operations |

### CP-Aware Decision (`decide_with_cp`)

```python
def decide_with_cp(self, signal, boundary, prediction_set=None):
    base = self.decide(signal, boundary)     # Existing 2-tier logic

    if prediction_set is None:
        return base                           # No CP → existing behavior

    if prediction_set.size == 1 and base.is_accept:
        return base                           # CP confirms: 1 candidate

    if prediction_set.size <= 5:
        return BOOST                          # Mediation: small set for LLM

    return DEFER                              # Too many: full search
```

---

## LLM Reranker

`services/llm_reranker.py` — Azure OpenAI gpt-4o-mini za reranking:

### Ulaz (Mediation Path)

```json
{
  "query": "ukupni troškovi po mjesecu",
  "candidates": [
    {"tool_id": "get_Expenses_Agg", "score": 0.87, "description": "Agregacija troškova"},
    {"tool_id": "get_Expenses_GroupBy", "score": 0.85, "description": "Grupiranje troškova"},
    {"tool_id": "get_Trips_Agg", "score": 0.82, "description": "Agregacija putovanja"}
  ]
}
```

### Izlaz

```json
{
  "tool_id": "get_Expenses_Agg",
  "confidence": 0.95,
  "reasoning": "Korisnik traži 'ukupne troškove' što je agregacija, ne grupiranje."
}
```

### Karakteristike

- **Max kandidata:** 10 (prompt triming)
- **Model:** gpt-4o-mini (isti kao full routing, ali manji prompt)
- **Latencija:** ~200ms (3-5 alata vs 25+ u full routing)
- **Prompt:** Hrvatski jezik, pravila za suffix razlikovanje (_Agg vs _GroupBy vs _ProjectTo)
- **Fallback:** Ako reranker fail → fallthrough na full LLM routing

---

## Conversation Engine

`services/engine/` — state machine za upravljanje konverzacijom:

```
services/engine/
├── __init__.py              # MessageEngine — koordinator
├── user_handler.py          # Identifikacija korisnika + GDPR consent
├── hallucination_handler.py # "Krivo" feedback detekcija
├── deterministic_executor.py # Fast path izvršenje (bez LLM-a)
├── flow_executors.py        # Multi-step flow handling
├── flow_handler.py          # Flow state management
└── tool_handler.py          # Tool execution + validacija
```

### Conversation States

```
idle → active → awaiting_confirmation → idle
  │                      ↕
  └→ multi_step_flow → collecting_params → executing → idle
```

### Multi-Step Flows

| Flow | Koraci | Primjer |
|------|--------|---------|
| **Booking** | Odaberi vozilo → Datum od/do → Potvrda | "Rezerviraj vozilo ABC-123 od sutra do petka" |
| **Mileage** | Odaberi vozilo → Kilometri → Datum → Potvrda | "Unesi kilometre za ABC-123" |
| **Case** | Tip → Opis → Vozilo → Potvrda | "Prijavi kvar na vozilu" |

Stanje flow-a čuva se u Redis hash-u (`conv_state:{phone}`), ne u memoriji procesa.

---

## API Gateway

`services/api_gateway.py` — enterprise HTTP klijent za MobilityOne API:

### Značajke

| Feature | Implementacija |
|---------|----------------|
| **OAuth2 Token** | `TokenManager` — auto-refresh, thread-safe cache |
| **Circuit Breaker** | 5 failures → 60s open → half-open probe |
| **Retry** | 3 pokušaja, exponential backoff (1s, 2s, 4s) |
| **Timeout** | 30s per request |
| **Rate Limiting** | Configurable, default 100 req/s |
| **Connection Pool** | httpx.AsyncClient s keepalive |

### Token Management

```
Token flow:
1. First request → POST /sso/connect/token (client_credentials grant)
2. Cache token u memoriji (TTL = expires_in - 60s buffer)
3. Auto-refresh 60s prije isteka
4. Thread-safe (asyncio.Lock)
```

### MobilityOne API Endpoints

Sustav koristi 950+ API endpointa parsiranih iz OpenAPI (Swagger) specifikacija:

```
config/processed_tool_registry.json (~2.8MB)
├── 950+ tool definicija
│   ├── tool_id (npr. "get_Vehicles")
│   ├── method (GET/POST/PUT/DELETE)
│   ├── path (npr. "/api/Vehicles")
│   ├── parameters (query, path, body)
│   └── response_schema
```

---

## Baza podataka

### Shema (6 tablica)

```sql
user_mappings          -- Phone → MobilityOne person mapping + GDPR consent
  ├── phone_number     (unique, indexed)
  ├── api_identity     (MobilityOne person ID)
  ├── tenant_id
  ├── gdpr_consent_given
  └── gdpr_anonymized_at

conversations          -- Conversation metadata
  ├── user_id          (FK → user_mappings)
  ├── status           (active/ended)
  └── flow_type        (booking/mileage/case/null)

messages               -- Individual messages
  ├── conversation_id  (FK → conversations)
  ├── role             (user/assistant/tool)
  ├── content
  ├── tool_name
  └── tool_result      (JSON)

tool_executions        -- API call audit trail
  ├── tool_name
  ├── parameters       (JSON)
  ├── result           (JSON)
  ├── success
  └── execution_time_ms

audit_logs             -- Admin action trail (admin_user ONLY)
  ├── action
  ├── entity_type
  ├── entity_id
  └── details          (JSON: admin_id, ip_address)

hallucination_reports  -- User "krivo" feedback
  ├── user_query
  ├── bot_response
  ├── user_feedback
  ├── reviewed         (bool, indexed)
  ├── correction
  └── category
```

### Dual-User Security Model

```
bot_user (API + Worker):
  ✅ SELECT/INSERT/UPDATE/DELETE na operativne tablice
  ✅ INSERT-only na hallucination_reports
  ❌ NEMA pristupa audit_logs tablici

admin_user (Admin API + Migracije):
  ✅ ALL PRIVILEGES na sve tablice
  ✅ CREATE/ALTER tablice (Alembic migracije)
```

### Connection Pooling

```python
# database.py
engine = create_async_engine(
    database_url,
    pool_size=10,           # Bazne konekcije
    max_overflow=20,        # Peak = 30 total
    pool_recycle=3600,      # Recycle svaki sat
    pool_pre_ping=True,     # Verify connection before use
    pool_use_lifo=True,     # Reuse recent connections first
)
```

**Upozorenje:** 8 max podova × (10 + 20) = 240 burst konekcija. PostgreSQL default `max_connections=100` nije dovoljno. Koristiti PgBouncer ili povećati `max_connections`.

---

## Redis arhitektura

### Ključevi i strukture

| Ključ | Tip | TTL | Namjena |
|-------|-----|-----|---------|
| `whatsapp_stream_inbound` | Stream | 24h | Inbound poruke od Infobip-a |
| `outbound:{phone}` | List | — | Outbound poruke za slanje |
| `conv_state:{phone}` | Hash | 1h | Conversation state machine |
| `chat_history:{phone}` | List | 1h | Chat history za LLM context |
| `user_context:{phone}` | Hash | 24h | User info cache (tenant, name) |
| `msg_lock:{sender}:{id}` | String | 5min | Distributed message dedup lock |
| `dlq:webhook` | List | 30d | Dead letter queue (primary) |
| `tenant:{phone}` | String | 24h | Tenant ID cache |
| `gdpr_consent:{phone}` | String | 5min | Negative consent cache |
| `stats:tools_loaded` | String | — | Loaded tools count |
| `readiness_check` | String | 10s | Health check write test |
| `delayed_outbound` | Sorted Set | — | Delayed messages (za burst mode) |

### Consumer Group

```
Stream: whatsapp_stream_inbound
Group: mobility_workers
Consumer: worker-{pod_id}

XREADGROUP GROUP mobility_workers worker-{pod_id}
  COUNT 1
  BLOCK 5000
  > (unread messages)
```

### Distributed Locking

```lua
-- Acquire: SET msg_lock:{sender}:{id} {worker_id} NX EX 300
-- Release (Lua, atomic):
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
```

---

## Sigurnosna arhitektura

### Webhook Security

```
Infobip → POST /webhook/inbound
  │
  ├── Body Size Guard: max 1MB (OOM prevention)
  ├── HMAC-SHA256 Signature Validation (INFOBIP_SECRET_KEY)
  ├── APP_STOPPING check (graceful shutdown → 503)
  └── Rate limiting (configurable)
```

### PII zaštita (2 sloja)

```
Layer 1: Manual masking
  phone[-4:] u log stringovima (user_service.py, webhook_simple.py, worker.py)

Layer 2: PIIScrubFilter (logging.Filter)
  Regex na SVIM log porukama — hvata phone, email, OIB, IBAN
  Primijenjen na main.py i worker.py logging handlere
```

### Container Security

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true    # /tmp via tmpfs emptyDir
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
```

Za kompletne sigurnosne detalje vidjeti [SECURITY.md](SECURITY.md).

---

## Observability

### Prometheus metrke (main.py)

| Metrika | Tip | Labels | Opis |
|---------|-----|--------|------|
| `http_requests_total` | Counter | method, endpoint, status | HTTP request count |
| `http_request_duration_seconds` | Histogram | method, endpoint | Latencija po endpointu |

Excludirane putanje: `/health`, `/ready`, `/metrics` (probe/scraper noise).

### Worker statistike

Worker objavljuje statistike u Redis (nema HTTP port):
- Obrađene poruke (success/fail)
- Queue lag (pending messages)
- Prosječna latencija obrade
- Lock contention count

API `/metrics` endpoint čita ove Redis ključeve i uključuje u Prometheus export.

### Health Endpoints

| Endpoint | Servis | Što provjerava |
|----------|--------|----------------|
| `GET /health` | API (8000) | DB ping, Redis ping, registry status |
| `GET /ready` | API (8000) | DB query, Redis SET→GET→DEL cycle, registry loaded |
| `GET /health` | Admin (8080) | DB ping, Redis ping |
| `GET /ready` | Admin (8080) | DB query, Redis ping |

**Važno:** `/ready` koristi Redis `SET`→`GET`→`DEL` (ne samo `PING`) jer PING uspije i na read-only replica.

### Alerting (KEDA Autoscaler)

| Alert | Uvjet | Opis |
|-------|-------|------|
| CPUThrottlingHigh | >25% CFS throttling, 5min | Pod nema dovoljno CPU |
| MemoryHighWater | >85% od 512Mi, 2min | Blizu OOM kill-a |
| WorkerQueueBacklog | >100 pending msg, 5min | Queue raste brže nego se procesira |
| KedaNotScaling | Queue >50, KEDA inactive, 2min | Autoscaler ne reagira |

---

## Dijagram ovisnosti modula

```
main.py
  └── webhook_simple.py → Redis Stream

worker.py
  └── services/engine/__init__.py (MessageEngine)
       ├── services/unified_router.py (UnifiedRouter)
       │    ├── services/query_router.py (QueryRouter)
       │    │    └── services/intent_classifier.py (IntentClassifier)
       │    │         └── services/dynamic_threshold.py (DecisionEngine)
       │    ├── services/unified_search.py (UnifiedSearch)
       │    │    ├── services/faiss_vector_store.py (FAISSVectorStore)
       │    │    ├── services/intent_classifier.py
       │    │    ├── services/entity_detector.py
       │    │    └── services/query_type_classifier.py
       │    ├── services/llm_reranker.py
       │    └── services/ambiguity_detector.py
       ├── services/api_gateway.py → MobilityOne API
       │    └── services/token_manager.py (OAuth2)
       ├── services/conversation_manager.py → Redis
       ├── services/context.py (UserContextManager) → Redis
       ├── services/response_formatter.py
       ├── services/response_extractor.py
       └── services/gdpr_masking.py

admin_api.py
  └── database.py → PostgreSQL (admin_user)
```

---

## Konfiguracija

### JSON konfiguracija (runtime)

| Datoteka | Veličina | Opis |
|----------|----------|------|
| `config/tool_documentation.json` | ~1.5MB | 950 alata s primjerima na hrvatskom |
| `config/tool_categories.json` | ~50KB | Kategorizacija alata (Vehicles, Expenses, ...) |
| `config/processed_tool_registry.json` | ~2.8MB | Parsani OpenAPI spec (parametri, paths) |
| `config/faiss_margin_calibration.json` | ~200B | Kalibrirani FAISS margin threshold |

### Environment varijable (80+)

Centralizirane u `config.py` putem Pydantic Settings. Vidjeti [DEPLOYMENT.md](DEPLOYMENT.md) za kompletnu listu.

---

## Ključne metrike

| Metrika | Vrijednost | Napomena |
|---------|------------|----------|
| API alata | 950+ | Auto-parsed iz OpenAPI specifikacija |
| Routing accuracy (Top-1) | 100% | Na 950 kuratiranih test upita |
| ML Fast Path coverage | ~85% upita | 0 LLM poziva, <10ms |
| Intent klasa | 45+ | TF-IDF + Logistic Regression |
| Query Type klasa | 12 | Suffix-based classification |
| FAISS search latencija | 1-5ms | In-memory, 1536 dimenzija |
| CP coverage garancija | 98% | Conformal Prediction (APS) |
| CP median set size | 2 | Većina upita → 1-2 kandidata |
| Full routing latencija | 2-3s | Uključuje Azure OpenAI |
| Test suite | 2,764 testova | 88% coverage |
| Measured API RSS | 323 MB | Pri normalnom opterećenju |
| Measured Worker RSS | 345 MB | Pri normalnom opterećenju |

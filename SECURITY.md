# Sigurnost i Usklađenost — MobilityOne WhatsApp Bot

> Verzija: 12.0 | Ažurirano: 2026-03-22

## Sadržaj

- [Pregled sigurnosne arhitekture](#pregled-sigurnosne-arhitekture)
- [GDPR usklađenost](#gdpr-usklađenost)
- [EU AI Act usklađenost](#eu-ai-act-usklađenost)
- [Autentifikacija i autorizacija](#autentifikacija-i-autorizacija)
- [Zaštita podataka u mirovanju](#zaštita-podataka-u-mirovanju)
- [Zaštita podataka u prijenosu](#zaštita-podataka-u-prijenosu)
- [PII zaštita (osobni podaci)](#pii-zaštita)
- [Container security](#container-security)
- [Mrežna sigurnost](#mrežna-sigurnost)
- [Sigurnost baze podataka](#sigurnost-baze-podataka)
- [Input validacija](#input-validacija)
- [Audit trail](#audit-trail)
- [Incident response](#incident-response)
- [Sigurnosno testiranje](#sigurnosno-testiranje)
- [Compliance matrica](#compliance-matrica)

---

## Pregled sigurnosne arhitekture

```
                    ┌──────────────────────────────────────┐
                    │         SIGURNOSNI SLOJEVI            │
                    ├──────────────────────────────────────┤
                    │                                      │
                    │  ┌──────────────────────────────┐    │
                    │  │  L1: Mrežna sigurnost        │    │
                    │  │  • TLS 1.2+ (Ingress)        │    │
                    │  │  • Network Policies           │    │
                    │  │  • Admin API interno only     │    │
                    │  └──────────────────────────────┘    │
                    │                                      │
                    │  ┌──────────────────────────────┐    │
                    │  │  L2: Aplikacijska sigurnost   │    │
                    │  │  • HMAC webhook validation    │    │
                    │  │  • OAuth2 token management    │    │
                    │  │  • Request size limiting      │    │
                    │  │  • Security headers           │    │
                    │  └──────────────────────────────┘    │
                    │                                      │
                    │  ┌──────────────────────────────┐    │
                    │  │  L3: Zaštita podataka         │    │
                    │  │  • PII masking (2 sloja)      │    │
                    │  │  • Dual-user DB model         │    │
                    │  │  • GDPR consent gate          │    │
                    │  │  • DLQ privacy                │    │
                    │  └──────────────────────────────┘    │
                    │                                      │
                    │  ┌──────────────────────────────┐    │
                    │  │  L4: Runtime sigurnost        │    │
                    │  │  • Non-root containers        │    │
                    │  │  • Read-only filesystem       │    │
                    │  │  • Drop ALL capabilities      │    │
                    │  │  • Resource limits             │    │
                    │  └──────────────────────────────┘    │
                    │                                      │
                    └──────────────────────────────────────┘
```

---

## GDPR usklađenost

### Pravna osnova

Sustav je usklađen sa sljedećim GDPR člancima:

| Članak | Zahtjev | Implementacija |
|--------|---------|----------------|
| Čl. 6 | Zakonitost obrade | Pristanak korisnika (consent gate) |
| Čl. 7 | Uvjeti za pristanak | Explicit opt-in, bilježenje vremena pristanka |
| Čl. 13 | Informiranje | Bot se identificira kao AI, objašnjava svrhu |
| Čl. 15 | Pravo pristupa | `GET /admin/gdpr/export/{phone}` |
| Čl. 17 | Pravo na brisanje | `POST /admin/gdpr/erase/{phone}` |
| Čl. 20 | Prenosivost podataka | JSON export svih korisnikovih podataka |
| Čl. 25 | Zaštita by design | PII masking, dual-user DB, minimizacija |
| Čl. 30 | Evidencija obrade | AuditLog tablica s punim trail-om |
| Čl. 32 | Sigurnost obrade | Enkripcija, pseudonimizacija, kontrola pristupa |
| Čl. 33 | Obavijest o povredi | Audit log omogućuje forenziku u 72h |

### GDPR Consent Gate

Svaki korisnik mora dati pristanak prije korištenja bota:

```
Novi korisnik → WhatsApp poruka
  │
  ├── UserHandler.identify_user()
  │   ├── Je li registriran u MobilityOne? → NE → Blokiraj ("Guest" user)
  │   └── DA → Provjeri gdpr_consent_given u user_mappings tablici
  │       ├── consent = true → Nastavi normalno
  │       └── consent = false/null → Prikaži GDPR consent poruku
  │           ├── Korisnik prihvati → SET gdpr_consent_given=true, gdpr_consent_at=now()
  │           └── Korisnik odbije → Blokiraj, poruka o odbijanju
  │
  └── Negative cache: gdpr_consent:{phone}=unknown, TTL=5min
      (sprječava ponavljanje API poziva za neregistrirane korisnike)
```

### Pravo na brisanje (Čl. 17)

**Endpoint:** `POST /admin/gdpr/erase/{phone}`

Proces brisanja u dva koraka:

```
1. PostgreSQL anonimizacija (gdpr_masking.py):
   ├── user_mappings: phone → SHA256 hash, display_name → "ANONIMIZIRANO"
   │   gdpr_anonymized_at = now()
   ├── messages: content → "***GDPR OBRISANO***"
   ├── conversations: metadata → {}
   └── hallucination_reports: user_query, bot_response → "***GDPR OBRISANO***"

2. Redis brisanje (gdpr_masking.py):
   ├── DEL conv_state:{phone}
   ├── DEL chat_history:{phone}
   ├── DEL user_context:{phone}
   ├── DEL tenant:{phone}
   ├── DEL gdpr_consent:{phone}
   └── DLQ scrub: ukloni sve zapise s phone brojem iz dlq:webhook
```

**Važno:** FAISS vektorski indeks NE sadrži korisničke podatke — sadrži samo opise alata. Ne zahtijeva GDPR brisanje.

### Pravo na pristup i prenosivost (Čl. 15, 20)

**Endpoint:** `GET /admin/gdpr/export/{phone}`

```json
{
  "user_profile": {
    "phone_hash": "sha256:...",
    "display_name": "Ivan Horvat",
    "tenant_id": "uuid",
    "consent_given": true,
    "consent_at": "2026-01-15T10:30:00Z",
    "created_at": "2026-01-15T10:29:00Z"
  },
  "conversations": [
    {
      "id": "uuid",
      "started_at": "2026-03-20T14:00:00Z",
      "messages": [
        {"role": "user", "content": "Prikaži moja vozila", "timestamp": "..."},
        {"role": "assistant", "content": "Pronašao sam 3 vozila...", "timestamp": "..."}
      ]
    }
  ],
  "tool_executions": [...],
  "hallucination_reports": [...],
  "ephemeral_state": {
    "conversation_state": {...},
    "chat_history": [...],
    "user_context": {...}
  }
}
```

### Retencija podataka

| Podatak | Lokacija | Retencija | Mehanizam |
|---------|----------|-----------|-----------|
| Poruke | PostgreSQL | 365 dana (configurable per user) | `gdpr_data_retention_days` |
| Chat history | Redis | 1 sat | TTL na ključu |
| Conv state | Redis | 1 sat | TTL na ključu |
| User context | Redis | 24 sata | TTL na ključu |
| DLQ | Redis | 30 dana | TTL + LTRIM 10K |
| DLQ fallback | `/tmp/dlq.jsonl` | Život poda | tmpfs, 5MB cap |
| Audit log | PostgreSQL | Neograničeno | Compliance zahtjev |

---

## EU AI Act usklađenost

### Klasifikacija rizika

MobilityOne WhatsApp Bot je klasificiran kao **LIMITED RISK** AI sustav (Članak 52) jer:
- Ne donosi autonomne odluke koje utječu na prava korisnika
- Služi kao informacijski asistent (read-mostly operacije)
- Mutacije (POST/PUT/DELETE) zahtijevaju eksplicitnu potvrdu korisnika

### Zahtjevi transparentnosti (Čl. 52)

| Zahtjev | Implementacija |
|---------|----------------|
| AI identifikacija | Bot se identificira kao "MobilityOne AI asistent" u system promptu I svim pozdravnim porukama |
| Jasna namjena | Poruke jasno komuniciraju da je ovo AI, ne čovjek |
| Feedback mehanizam | Korisnik može reći "krivo" za hallucination report |

### Implementacija transparentnosti

```python
# services/flow_phrases.py — svi pozdravi uključuju AI identifikaciju:
GREETING_RESPONSE = (
    "Pozdrav! 👋 Ja sam MobilityOne AI asistent.\n\n"
    "Mogu vam pomoći s informacijama o vozilima, troškovima, "
    "putovanjima i ostalim fleet management podacima.\n\n"
    "Kako vam mogu pomoći?"
)
```

### Hallucination Management

Sustav detektira i bilježi halucinacije putem korisničkog feedbacka:

```
Korisnik: "krivo" ili "to nije točno"
  │
  ├── HallucinationHandler detektira feedback
  ├── Sprema u hallucination_reports tablicu:
  │   ├── user_query (što je korisnik pitao)
  │   ├── bot_response (što je bot odgovorio)
  │   ├── user_feedback (korisnikov feedback)
  │   ├── model (koji LLM model je korišten)
  │   └── retrieved_chunks (koji alati/podaci su korišteni)
  │
  └── Admin review workflow:
      ├── GET /admin/hallucinations — lista nerecenziranih
      ├── POST /admin/hallucinations/{id}/review — označi recenziranim
      └── Kategorije: wrong_tool, wrong_data, wrong_format, other
```

### Confidence Thresholds

| Odluka | Prag | Obrazloženje |
|--------|------|-------------|
| ML Fast Path | ≥ 85% | Model accuracy 99.24% opravdava niži prag |
| Mutation (POST/PUT/DELETE) | ≥ 80% | Viši prag za destruktivne operacije |
| LLM routing | Nema praga | LLM odlučuje s punim kontekstom |
| CP coverage | ≥ 98% | Matematička garancija Conformal Prediction |

### Audit Trail

Kompletni lanac od upita do odgovora:

```
Message (user query)
  → ToolExecution (koji API alat, parametri, rezultat)
    → Message (bot response)
      → HallucinationReport (ako korisnik kaže "krivo")
        → AuditLog (admin review akcija)
```

---

## Autentifikacija i autorizacija

### Webhook autentifikacija (Infobip → Bot)

```python
# webhook_simple.py
def verify_signature(request_body: bytes, signature: str) -> bool:
    expected = hmac.new(
        INFOBIP_SECRET_KEY.encode(),
        request_body,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

- **Algoritam:** HMAC-SHA256
- **Ključ:** `INFOBIP_SECRET_KEY` (environment variable)
- **Timing-safe comparison:** `hmac.compare_digest()` sprječava timing attack

### Bot → MobilityOne API autentifikacija

```
OAuth2 Client Credentials Grant:
  POST /sso/connect/token
  Content-Type: application/x-www-form-urlencoded

  grant_type=client_credentials
  &client_id={MOBILITY_CLIENT_ID}
  &client_secret={MOBILITY_CLIENT_SECRET}
  &scope=openid

Odgovor:
  {"access_token": "...", "expires_in": 3600, "token_type": "Bearer"}
```

- **Token caching:** U memoriji, auto-refresh 60s prije isteka
- **Thread-safe:** `asyncio.Lock` za concurrent access
- **Per-tenant:** Token je vezan za specifičnog tenant-a

### Admin API autentifikacija

```python
# admin_api.py
# Bearer token validation:
Authorization: Bearer {ADMIN_TOKEN_1}

# IP whitelist (optional):
ADMIN_ALLOWED_IPS=10.0.0.0/8,192.168.0.0/16
```

- **Multi-token support:** Do 5 admin tokena (`ADMIN_TOKEN_1` do `ADMIN_TOKEN_5`)
- **Token → User mapping:** Svaki token mapiran na username (`ADMIN_TOKEN_1_USER`)
- **IP filtering:** Opcionalni IP whitelist za dodatnu zaštitu

---

## Zaštita podataka u mirovanju

### PostgreSQL

| Mjera | Status | Detalji |
|-------|--------|---------|
| Enkripcija diska | Infra odgovornost | AWS EBS / Azure Disk encryption |
| Column-level encryption | Ne | PII se pseudonimizira, ne kriptira |
| Pseudonimizacija | Da | SHA256 hash za phone_number pri GDPR brisanju |
| Backup enkripcija | Infra odgovornost | pg_dump → encrypted storage |

### Redis

| Mjera | Status | Detalji |
|-------|--------|---------|
| AUTH | Da | Redis requirepass konfiguracija |
| ACL | Preporučeno | Redis 7 ACL za fine-grained pristup |
| Persistence | AOF + RDB | Configurable u Redis konfiguraciji |
| TTL na PII | Da | Sve korisničke ključeve imaju TTL (1h-24h) |

### Secrets Management

```yaml
# K8s Sealed Secrets (šifrirani u git-u, dešifrirani u clusteru):
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
spec:
  encryptedData:
    AZURE_OPENAI_API_KEY: AgBy3...
    MOBILITY_CLIENT_SECRET: AgBy3...
    INFOBIP_API_KEY: AgBy3...
    GDPR_HASH_SALT: AgBy3...
```

**Nikada u kodu:** Svi secreti dolaze iz environment varijabli → `config.py` (Pydantic Settings, required fields).

---

## Zaštita podataka u prijenosu

| Kanal | Protokol | Certifikat |
|-------|----------|------------|
| Infobip → API | HTTPS (TLS 1.2+) | Ingress TLS termination |
| API → Redis | redis:// (internal) | K8s Network Policy isolacija |
| API → PostgreSQL | postgresql:// (internal) | K8s Network Policy isolacija |
| Worker → Azure OpenAI | HTTPS (TLS 1.3) | Azure managed certificate |
| Worker → MobilityOne API | HTTPS (TLS 1.2+) | MobilityOne managed |
| Admin browser → Admin API | HTTPS (TLS 1.2+) | Internal VPN + Ingress TLS |

**Napomena:** Interne komunikacije (Redis, PostgreSQL) koriste plain-text unutar Kubernetes clustera, zaštićene Network Policies-ima. Za enhanced security, moguće je uključiti mTLS putem service mesh-a (Istio/Linkerd).

---

## PII zaštita

### Definirani PII tipovi

`services/gdpr_masking.py` detektira i maskira sljedeće tipove osobnih podataka:

| PII Tip | Primjer | Maskirano kao | Validacija |
|---------|---------|---------------|------------|
| Telefon | +385 91 234 5678 | `[PHONE-MASKED]` | HR + international formati |
| Email | ivan@example.com | `[EMAIL-MASKED]` | RFC 5322 regex |
| OIB | 12345678901 | `[OIB-MASKED]` | 11 znamenki + checksum (Mod11) |
| IBAN | HR1210010051863000160 | `[IBAN-MASKED]` | HR prefix + 19 znakova |
| Kreditna kartica | 4111 1111 1111 1111 | `[CC-MASKED]` | Luhn algoritam |
| IP adresa | 192.168.1.1 | `[IP-MASKED]` | IPv4 + IPv6 |
| Ime | Ivan Horvat | `[NAME-MASKED]` | Heuristika (2+ capitalized riječi) |

### Dva sloja zaštite

#### Sloj 1: Ručno maskiranje u kodu

```python
# Svugdje gdje se logira phone number:
logger.info(f"Processing message for ...{phone[-4:]}")
# Umjesto:
logger.info(f"Processing message for {phone}")  # ZABRANJENO
```

Verificirane datoteke: `user_service.py`, `webhook_simple.py`, `worker.py`

#### Sloj 2: PIIScrubFilter (automatski)

```python
# services/pii_filter.py
class PIIScrubFilter(logging.Filter):
    """Regex zamjena PII u SVIM log porukama."""

    PHONE_PATTERN = re.compile(r'\+?385[\s.-]?\d{2}[\s.-]?\d{3,4}[\s.-]?\d{3,4}')
    INTL_PATTERN = re.compile(r'\+\d{10,15}')

    def filter(self, record):
        # Scrub message
        record.msg = self._scrub(str(record.msg))
        # Scrub args
        if record.args:
            record.args = tuple(self._scrub(str(a)) for a in record.args)
        # Scrub exception traceback
        if record.exc_text:
            record.exc_text = self._scrub(record.exc_text)
        return True
```

Primijenjen na **oba** procesa:
- `main.py`: `_stdout_handler.addFilter(_pii_filter)`
- `worker.py`: `_stderr_handler.addFilter(_pii_filter)`

### Pseudonimizacija

Za GDPR compliance, phone number se pseudonimizira putem HMAC-SHA256:

```python
def pseudonymize(phone: str, salt: str) -> str:
    return hmac.new(
        salt.encode('utf-8'),
        phone.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

# Salt zahtjevi:
# - Min 32 znaka (GDPR_HASH_SALT env var)
# - Rotacija invalidira sve pseudonimizacijske linkove
# - NIKADA u source code-u
```

---

## Container security

### Security Context (svi podovi)

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]

# Writable volume za privremene datoteke:
volumes:
  - name: tmp
    emptyDir:
      medium: Memory      # tmpfs — ne zapisuje na disk
      sizeLimit: 50Mi     # OOM zaštita
```

### Image Security

| Mjera | Detalji |
|-------|---------|
| Base image | `python:3.12-slim` (minimalni footprint) |
| Non-root user | `USER 1000` u Dockerfile-u |
| No secrets u image-u | Sve iz env vars / K8s Secrets |
| Vulnerability scanning | `trivy image` u CI/CD |
| Dependency audit | `pip-audit` u CI/CD, `bandit` za SAST |

### Resource Limits

| Component | CPU req/limit | Memory req/limit | Measured RSS |
|-----------|--------------|-------------------|-------------|
| API (×2) | 50m / 500m | 400Mi / 512Mi | 323 MB |
| Worker | 50m / 500m | 416Mi / 512Mi | 345 MB |
| Admin | 25m / 200m | 96Mi / 192Mi | 66 MB |
| Burst Job | 50m / 500m | 416Mi / 512Mi | ~345 MB |

**Ukupno rezervirano:** 200m CPU, 1440Mi memorije

---

## Mrežna sigurnost

### Kubernetes Network Policies

```yaml
# Admin API — samo interni pristup
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

### Ingress pravila

| Servis | Host | Pristup | Napomena |
|--------|------|---------|----------|
| API (webhook) | bot.domain.com | Public | Samo `/webhook` path |
| Admin API | admin.internal.domain.com | VPN only | Svi putovi |
| Redis | — | Cluster internal | Nema Ingress-a |
| PostgreSQL | — | Cluster internal | Nema Ingress-a |

### Security Headers

```python
# main.py — SecurityHeadersMiddleware
response.headers["X-Content-Type-Options"] = "nosniff"
response.headers["X-Frame-Options"] = "DENY"
response.headers["X-XSS-Protection"] = "1; mode=block"
response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
response.headers["Cache-Control"] = "no-store"
```

---

## Sigurnost baze podataka

### Dual-User Model

Implementiran u `docker/init-db.sh` i `alembic/versions/001_initial_schema.py`:

```sql
-- bot_user (API + Worker)
GRANT SELECT, INSERT, UPDATE, DELETE
  ON user_mappings, conversations, messages, tool_executions
  TO bot_user;

GRANT INSERT ON hallucination_reports TO bot_user;
-- NE MOŽE: SELECT hallucination_reports (samo admin)
-- NE MOŽE: pristup audit_logs tablici

-- admin_user (Admin API + Migracije)
GRANT ALL PRIVILEGES ON ALL TABLES TO admin_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES TO admin_user;
```

### SQL Injection zaštita

- **ORM:** SQLAlchemy 2.0 s parametriziranim upitima — nema raw SQL-a
- **Admin review:** Bounded regex `.{0,100}?` umjesto greedy `.*` (ReDOS zaštita)
- **Sanitization:** `services/scoring_and_filters.py` — SQL injection pattern detection u user input-u

### Connection Security

```python
# database.py
engine = create_async_engine(
    database_url,
    pool_pre_ping=True,     # Verify connection alive
    pool_recycle=3600,       # Recycle after 1 hour
    pool_timeout=30,         # Max wait for connection
)
```

---

## Input validacija

### Webhook Input

| Validacija | Lokacija | Detalj |
|------------|----------|--------|
| Body size | `main.py` | Max 1MB (`MAX_REQUEST_BODY_BYTES`) |
| HMAC signature | `webhook_simple.py` | SHA256 s `INFOBIP_SECRET_KEY` |
| Message format | `webhook_simple.py` | Infobip message schema validation |
| Phone format | `user_service.py` | E.164 format validation |

### Tool Parameter Sanitization

```python
# services/schema_sanitizer.py
# Sanitizes tool parameters before sending to MobilityOne API:
# - Removes unexpected fields
# - Validates field types against OpenAPI schema
# - Prevents injection through parameter values
```

### Croatian Date Parsing

```python
# services/flow_handler.py
# Sigurno parsiranje hrvatskih datuma:
# "sutra" → datetime.now() + timedelta(days=1)
# "15.03.2026" → datetime(2026, 3, 15)
# Nema eval() ili exec() — ručni parser
```

---

## Audit trail

### Što se bilježi

| Događaj | Tablica | Detalji |
|---------|---------|---------|
| Svaka poruka korisnika | messages | content, timestamp, role=user |
| Svaki bot odgovor | messages | content, timestamp, role=assistant |
| Svaki API poziv | tool_executions | tool_name, parameters, result, success, execution_time_ms |
| Hallucination feedback | hallucination_reports | query, response, feedback, model |
| Admin akcije | audit_logs | action, entity_type, admin_id, ip_address |
| GDPR operacije | audit_logs | action="gdpr_erase"/"gdpr_export", entity_id=phone_hash |

### Pristup audit logovima

```python
# Samo admin_user može pristupiti audit_logs tablici
# bot_user dobiva PostgreSQL ERROR ako pokuša SELECT na audit_logs
```

### Retention

- **Audit logovi:** Neograničena retencija (compliance zahtjev)
- **Poruke:** 365 dana (configurable per user via `gdpr_data_retention_days`)
- **Tool executions:** Prati retenciju poruka

---

## Incident response

### Forenzika

Sustav omogućuje potpunu rekonstrukciju interakcije:

```
1. Pronađi korisnika: SELECT * FROM user_mappings WHERE phone_number LIKE '%1234'
2. Pronađi razgovore: SELECT * FROM conversations WHERE user_id = '{user_id}'
3. Pronađi poruke: SELECT * FROM messages WHERE conversation_id = '{conv_id}' ORDER BY timestamp
4. Pronađi API pozive: SELECT * FROM tool_executions WHERE executed_at BETWEEN ... AND ...
5. Pronađi halucinacije: SELECT * FROM hallucination_reports WHERE conversation_id = '{conv_id}'
6. Pronađi admin akcije: SELECT * FROM audit_logs WHERE entity_id = '{user_id}'
```

### Breach Notification (Čl. 33 GDPR)

Audit log tablica + Redis DLQ + PII filter logovi omogućuju:
- **Identifikaciju zahvaćenih korisnika** (72h zahtjev)
- **Rekonstrukciju opsega** (koje su poruke bile izložene)
- **Timeline** (kada se incident dogodio)

### Graceful Shutdown

```
SIGTERM primi API pod
  │
  ├── APP_STOPPING = True
  ├── Webhook vraća 503 (Infobip retryja)
  ├── /ready vraća 503 (K8s stopira routing)
  ├── Worker čeka 120s grace period (> 90s msg timeout)
  └── Sve in-flight poruke završavaju processing
```

---

## Sigurnosno testiranje

### CI/CD Pipeline

```bash
# Statička analiza (SAST)
bandit -r services/ -f json          # Python security scanner
ruff check .                         # Linter s security pravilima

# Dependency audit
pip-audit                            # Provjera CVE-ova u dependency-ima

# Container scanning
trivy image mobilityone/api:latest   # Vulnerability scan

# PII leak scan
python scripts/verify_production_readiness.py
```

### Verifikacijska skripta

`scripts/verify_production_readiness.py` provjerava:

| Provjera | Opis |
|----------|------|
| Lua cache integritet | Redis Lua skripta SHA odgovara očekivanom |
| FAISS integritet | Vektorski indeks konzistentan s tool dokumentacijom |
| Memory baseline | RSS ispod konfiguriranih limita |
| PII scan | Nema hardcodiranih telefonskih brojeva ili API ključeva |
| DB connectivity | Oba korisnika (bot + admin) se mogu spojiti |
| Redis write test | SET → GET → DEL cycle uspijeva |

---

## Compliance matrica

### GDPR

| Zahtjev | Status | Implementacija |
|---------|--------|----------------|
| Pristanak (Čl. 6, 7) | ✅ | Consent gate, timestamp bilježenje |
| Informiranje (Čl. 13) | ✅ | AI identifikacija u svim porukama |
| Pravo pristupa (Čl. 15) | ✅ | `/admin/gdpr/export/{phone}` |
| Pravo na brisanje (Čl. 17) | ✅ | `/admin/gdpr/erase/{phone}` — DB + Redis |
| Prenosivost (Čl. 20) | ✅ | JSON export svih podataka |
| Zaštita by design (Čl. 25) | ✅ | PII masking, dual-user DB, minimizacija |
| Evidencija obrade (Čl. 30) | ✅ | AuditLog tablica |
| Sigurnost obrade (Čl. 32) | ✅ | Encryption, pseudonymization, access control |
| Notifikacija (Čl. 33) | ✅ | Audit trail za 72h forenziku |

### EU AI Act

| Zahtjev | Status | Implementacija |
|---------|--------|----------------|
| Klasifikacija rizika | ✅ | Limited risk (informacijski asistent) |
| Transparentnost (Čl. 52) | ✅ | AI self-identification u svim interakcijama |
| Human oversight | ✅ | Hallucination review workflow, admin panel |
| Tehnika documentation | ✅ | ARCHITECTURE.md, ovaj dokument |
| Logging | ✅ | Kompletni audit trail |

### OWASP Top 10

| Rizik | Mitigacija |
|-------|------------|
| A01 Broken Access Control | Dual-user DB, Admin IP whitelist, Network Policies |
| A02 Cryptographic Failures | TLS 1.2+, HMAC-SHA256, proper salt management |
| A03 Injection | SQLAlchemy ORM (parametrizirani upiti), input sanitization |
| A04 Insecure Design | Threat modeling, defense-in-depth |
| A05 Security Misconfiguration | Pydantic Settings (required fields), security headers |
| A06 Vulnerable Components | pip-audit u CI, trivy image scanning |
| A07 Auth Failures | OAuth2 + HMAC webhook + Bearer admin tokens |
| A08 Data Integrity Failures | HMAC webhook verification, Lua atomic locks |
| A09 Logging Failures | PII filter, comprehensive audit trail |
| A10 SSRF | Nema user-controlled URL-ova — svi API endpointi preddefinirani |

---

## Kontakt

Za sigurnosne incidente ili pitanja o usklađenosti:
- **Admin panel:** `https://admin.internal.domain.com`
- **Audit logovi:** `SELECT * FROM audit_logs ORDER BY created_at DESC`
- **GDPR zahtjevi:** Kroz Admin API endpoints

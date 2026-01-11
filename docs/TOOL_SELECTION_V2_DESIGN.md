# Tool Selection V2 - Comprehensive Design

## Problem Statement

Current system has **57.9% Top-1 accuracy** on 950 tools because:
1. LLM receives 50+ tools to choose from (information overload)
2. Embeddings don't understand Croatian well
3. No domain pre-filtering
4. Flat classification (no hierarchy)
5. No confidence thresholds or fallbacks

---

## Proposed Architecture: 4-Stage Hierarchical Selection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TOOL SELECTION V2 PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT: "Obriši trošak za gorivo"                                        │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 0: QUERY ROUTER (Fast Path - 0ms)                            │ │
│  │ ────────────────────────────────────────────────────────────────── │ │
│  │ • 100+ deterministic regex patterns                                 │ │
│  │ • If EXACT match → return immediately (skip all stages)            │ │
│  │ • Handles 60-70% of common queries                                  │ │
│  │ • Stays in CROATIAN (no translation needed)                         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│         ↓ (if no exact match)                                           │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 1: INTENT + SUPER-DOMAIN DETECTION (5ms)                     │ │
│  │ ────────────────────────────────────────────────────────────────── │ │
│  │ A) ACTION INTENT: DELETE (confidence: 0.95)                         │ │
│  │    Keywords: "obriši" → DELETE                                      │ │
│  │                                                                      │ │
│  │ B) SUPER-DOMAIN: FINANCIALS (confidence: 0.90)                      │ │
│  │    Keywords: "trošak", "gorivo" → FINANCIALS                        │ │
│  │                                                                      │ │
│  │ Filter: 950 → 82 tools (only FINANCIALS domain)                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│         ↓                                                                │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 2: SUB-DOMAIN + METHOD FILTER (2ms)                          │ │
│  │ ────────────────────────────────────────────────────────────────── │ │
│  │ A) SUB-DOMAIN: Expenses (not ExpenseTypes, not ExpenseGroups)       │ │
│  │    Keywords: "trošak" without "tip" or "grupa"                      │ │
│  │                                                                      │ │
│  │ B) METHOD FILTER: DELETE only                                       │ │
│  │    (from Stage 1 intent)                                             │ │
│  │                                                                      │ │
│  │ Filter: 82 → 8 tools (DELETE Expenses only)                         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│         ↓                                                                │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 3: SEMANTIC RANKING (10ms)                                   │ │
│  │ ────────────────────────────────────────────────────────────────── │ │
│  │ A) TRANSLATE query to English (optional, if confidence low)        │ │
│  │    "Obriši trošak za gorivo" → "Delete fuel expense"               │ │
│  │                                                                      │ │
│  │ B) FAISS search within 8 filtered tools                             │ │
│  │    Rank by semantic similarity                                       │ │
│  │                                                                      │ │
│  │ Result: Top 3 candidates with scores                                 │ │
│  │    1. delete_Expenses_id (0.92)                                      │ │
│  │    2. delete_Expenses (0.78)                                         │ │
│  │    3. delete_Expenses_DeleteByCriteria (0.71)                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│         ↓                                                                │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 4: LLM FINAL DECISION (100ms)                                │ │
│  │ ────────────────────────────────────────────────────────────────── │ │
│  │ LLM receives ONLY 3 tools with full details:                        │ │
│  │                                                                      │ │
│  │ Tool 1: delete_Expenses_id                                          │ │
│  │   - Purpose: Briše pojedinačni trošak po ID-u                       │ │
│  │   - Params: expenseId (required)                                     │ │
│  │   - When: Korisnik želi obrisati specifični trošak                  │ │
│  │                                                                      │ │
│  │ Tool 2: delete_Expenses                                              │ │
│  │   - Purpose: Bulk delete troškova                                    │ │
│  │   - Params: filter criteria                                          │ │
│  │   - When: Korisnik želi obrisati više troškova                      │ │
│  │                                                                      │ │
│  │ LLM Decision: delete_Expenses_id (needs expenseId)                  │ │
│  │ Confidence: 0.95                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│         ↓                                                                │
│                                                                          │
│  OUTPUT: { tool: "delete_Expenses_id", params: {expenseId: "?"} }       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. SUPER-DOMAIN HIERARCHY

### 1.1 Super-Domains (10 categories)

| Super-Domain | Sub-Domains | Tools | Keywords (HR) |
|--------------|-------------|-------|---------------|
| **VEHICLES** | Vehicles, VehicleTypes, VehicleContracts, VehicleCalendar | ~82 | vozilo, auto, registracija |
| **PERSONS** | Persons, PersonTypes, PersonOrgUnits, PersonActivities | ~80 | osoba, korisnik, zaposlenik |
| **FINANCIALS** | Expenses, ExpenseTypes, ExpenseGroups, CostCenters | ~82 | trošak, rashod, cijena, gorivo |
| **TRIPS** | Trips, TripTypes, MileageReports | ~61 | put, vožnja, kilometraža |
| **EQUIPMENT** | Equipment, EquipmentTypes, EquipmentCalendar | ~61 | oprema, alat |
| **CASES** | Cases, CaseTypes | ~40 | slučaj, šteta, kvar, prijava |
| **TEAMS** | Teams, TeamMembers | ~40 | tim, član, ekipa |
| **ORGANIZATION** | Companies, OrgUnits, Partners, Tenants | ~82 | tvrtka, organizacija, partner |
| **SCHEDULING** | PeriodicActivities, SchedulingModels, Pools | ~60 | raspored, aktivnost, servis |
| **SYSTEM** | Lookup, Settings, Dashboard, Documents, Tags | ~100 | postavke, oznaka, dokument |

### 1.2 Confidence Thresholds

```python
SUPER_DOMAIN_THRESHOLDS = {
    "min_confidence": 0.70,      # Below this → ask user
    "high_confidence": 0.90,     # Above this → proceed without LLM
    "multi_domain_gap": 0.15,    # If top 2 domains < this gap → ask user
}
```

---

## 2. TRANSLATION STRATEGY

### 2.1 When to Translate?

```python
TRANSLATION_RULES = {
    # Stage 0: QueryRouter - NEVER translate (patterns are in Croatian)
    "query_router": False,

    # Stage 1: Domain Detection - NEVER translate (keywords are in Croatian)
    "domain_detection": False,

    # Stage 3: Semantic Search - CONDITIONAL translate
    "semantic_search": {
        "translate_if": [
            "domain_confidence < 0.80",  # Uncertain domain
            "no_croatian_keywords_matched",  # Query has unknown words
            "semantic_score < 0.60",  # Low similarity scores
        ],
        "never_translate_if": [
            "query_contains_proper_nouns",  # "Škoda Octavia"
            "query_contains_ids",  # "ZG1234AB"
        ]
    },

    # Stage 4: LLM - NEVER translate (LLM understands Croatian)
    "llm_decision": False,
}
```

### 2.2 Translation Implementation

```python
async def translate_if_needed(query: str, context: dict) -> tuple[str, bool]:
    """
    Translate query to English only when beneficial.
    Returns (translated_query, was_translated)
    """
    # Check if translation would help
    if context.get("domain_confidence", 1.0) >= 0.80:
        return query, False

    if context.get("semantic_score", 1.0) >= 0.60:
        return query, False

    # Check for proper nouns / IDs that shouldn't be translated
    if contains_plate_number(query) or contains_vehicle_name(query):
        return query, False

    # Translate using Azure Translator or GPT
    translated = await translate_hr_to_en(query)

    return translated, True
```

---

## 3. ENRICHED TOOL METADATA

### 3.1 Current vs Enhanced Metadata

```python
# CURRENT (poor)
{
    "operation_id": "delete_Expenses_id",
    "method": "DELETE",
    "path": "/expenses/{id}",
    "summary": "Delete expense by id",  # Often empty or generic
}

# ENHANCED (rich)
{
    "operation_id": "delete_Expenses_id",
    "method": "DELETE",
    "path": "/expenses/{id}",

    # NEW: Hierarchical classification
    "super_domain": "FINANCIALS",
    "sub_domain": "Expenses",

    # NEW: Croatian keywords for matching
    "keywords_hr": ["trošak", "rashod", "izdatak", "brisanje"],
    "keywords_en": ["expense", "cost", "delete", "remove"],

    # NEW: Action verbs that trigger this tool
    "trigger_verbs_hr": ["obriši", "ukloni", "izbriši", "makni"],
    "trigger_verbs_en": ["delete", "remove", "erase"],

    # NEW: Negative keywords (when NOT to use this tool)
    "negative_keywords": ["tip", "vrsta", "grupa", "kategorija"],

    # NEW: Synonyms for better matching
    "synonyms": {
        "trošak": ["rashod", "izdatak", "račun"],
        "expense": ["cost", "spending", "expenditure"],
    },

    # NEW: Example queries (for few-shot + embedding)
    "example_queries_hr": [
        "Obriši ovaj trošak",
        "Ukloni rashod za gorivo",
        "Izbriši izdatak",
    ],
    "example_queries_en": [
        "Delete this expense",
        "Remove fuel cost",
    ],

    # NEW: Disambiguation hints
    "disambiguation": {
        "vs_delete_Expenses": "Use _id for single expense, bulk for multiple",
        "vs_delete_ExpenseTypes": "This deletes expense data, not expense categories",
    },

    # NEW: Required context
    "requires_context": ["expense_id"],
    "context_source": {
        "expense_id": "USER_INPUT or PREVIOUS_QUERY",
    },
}
```

### 3.2 Auto-Enrichment Script

```python
async def enrich_tool_metadata(tool: UnifiedToolDefinition) -> EnrichedTool:
    """
    Automatically enrich tool with keywords, synonyms, examples.
    Uses LLM for generation, then caches.
    """
    prompt = f"""
    Za ovaj API alat generiraj metadata na hrvatskom:

    Tool: {tool.operation_id}
    Method: {tool.method}
    Path: {tool.path}
    Current description: {tool.description}

    Generiraj:
    1. super_domain: jedna od [VEHICLES, PERSONS, FINANCIALS, TRIPS, EQUIPMENT, CASES, TEAMS, ORGANIZATION, SCHEDULING, SYSTEM]
    2. keywords_hr: 5-10 ključnih riječi na hrvatskom
    3. trigger_verbs_hr: glagoli koji pokreću ovaj alat
    4. negative_keywords: riječi koje znače da korisnik NE želi ovaj alat
    5. example_queries_hr: 3-5 primjera upita

    Odgovori u JSON formatu.
    """

    response = await llm.generate(prompt)
    return parse_enriched_metadata(response)
```

---

## 4. CONFIDENCE & FALLBACK SYSTEM

### 4.1 Confidence Propagation

```python
@dataclass
class SelectionConfidence:
    """Confidence at each stage."""
    stage1_domain: float      # 0.0 - 1.0
    stage2_subdomain: float   # 0.0 - 1.0
    stage3_semantic: float    # 0.0 - 1.0
    stage4_llm: float         # 0.0 - 1.0

    @property
    def overall(self) -> float:
        """Combined confidence score."""
        return (
            self.stage1_domain * 0.25 +
            self.stage2_subdomain * 0.25 +
            self.stage3_semantic * 0.25 +
            self.stage4_llm * 0.25
        )

    @property
    def should_confirm(self) -> bool:
        """Should we ask user to confirm?"""
        return (
            self.overall < 0.70 or
            self.stage1_domain < 0.60 or
            self.stage4_llm < 0.80
        )
```

### 4.2 Fallback Strategies

```python
FALLBACK_STRATEGIES = {
    # Stage 1: Domain unclear
    "domain_unclear": {
        "threshold": 0.60,
        "action": "ASK_USER_DOMAIN",
        "message": "Nisam siguran o čemu se radi. Želite li:\n1. Vozila\n2. Troškove\n3. Opremu\n...",
    },

    # Stage 2: Multiple sub-domains match
    "subdomain_ambiguous": {
        "threshold": 0.15,  # Gap between top 2
        "action": "ASK_USER_SUBDOMAIN",
        "message": "Mislite li na:\n1. Pojedinačni trošak\n2. Tip troška\n3. Grupu troškova",
    },

    # Stage 3: Low semantic scores
    "semantic_low": {
        "threshold": 0.50,
        "action": "TRANSLATE_AND_RETRY",
        "fallback": "ASK_USER_CLARIFY",
    },

    # Stage 4: LLM uncertain
    "llm_uncertain": {
        "threshold": 0.70,
        "action": "SHOW_TOP_3_OPTIONS",
        "message": "Pronašao sam ove opcije:\n1. {tool1}\n2. {tool2}\n3. {tool3}\nKoju želite?",
    },
}
```

### 4.3 User Clarification Flow

```
User: "trošak"  (ambiguous - could be GET, POST, DELETE, any subdomain)

Bot: "Što želite napraviti s troškovima?
      1. 📋 Pregledati troškove
      2. ➕ Dodati novi trošak
      3. ✏️ Urediti postojeći trošak
      4. 🗑️ Obrisati trošak"

User: "1"

Bot: [Now confident: GET + Expenses domain]
     "Evo vaših troškova za ovaj mjesec: ..."
```

---

## 5. EMBEDDING STRATEGY

### 5.1 Hybrid Embedding Approach

```python
class HybridEmbeddingEngine:
    """
    Combines multiple embedding strategies for better accuracy.
    """

    def __init__(self):
        # Primary: Multilingual model (understands Croatian)
        self.multilingual = load_model("intfloat/multilingual-e5-large")

        # Secondary: English-optimized (for translated queries)
        self.english = AzureOpenAIEmbeddings("text-embedding-ada-002")

        # Tertiary: Keyword-based (BM25 for exact matches)
        self.bm25 = BM25Index()

    async def search(self, query: str, tools: List[Tool], top_k: int = 5) -> List[SearchResult]:
        """
        Hybrid search combining semantic + keyword matching.
        """
        # 1. Keyword search (fast, exact)
        keyword_results = self.bm25.search(query, tools, top_k=top_k*2)

        # 2. Multilingual semantic search
        ml_results = await self.multilingual.search(query, tools, top_k=top_k*2)

        # 3. If low scores, try English
        if max(r.score for r in ml_results) < 0.60:
            translated = await translate_hr_to_en(query)
            en_results = await self.english.search(translated, tools, top_k=top_k*2)
            ml_results = merge_results(ml_results, en_results)

        # 4. Merge and rank
        final = self._reciprocal_rank_fusion(keyword_results, ml_results)

        return final[:top_k]
```

### 5.2 Embedding Text Construction

```python
def build_embedding_text(tool: EnrichedTool) -> str:
    """
    Build rich text for embedding that captures all semantics.
    """
    parts = [
        # Operation name (split camelCase)
        split_camel_case(tool.operation_id),

        # Croatian keywords (most important!)
        " ".join(tool.keywords_hr),

        # Trigger verbs
        " ".join(tool.trigger_verbs_hr),

        # Example queries (teaches embedding what queries look like)
        " | ".join(tool.example_queries_hr[:3]),

        # English equivalents (for translated queries)
        " ".join(tool.keywords_en),

        # Description
        tool.description or tool.summary,
    ]

    return " ".join(filter(None, parts))
```

---

## 6. ACCURACY GUARANTEES

### 6.1 Validation Layers

```python
class ToolSelectionValidator:
    """
    Validates tool selection before execution.
    """

    async def validate(self, selection: ToolSelection) -> ValidationResult:
        # 1. Check confidence thresholds
        if selection.confidence.overall < 0.60:
            return ValidationResult(
                valid=False,
                action="ASK_CLARIFICATION",
                reason="Low confidence"
            )

        # 2. Check required parameters
        missing = self._check_required_params(selection)
        if missing:
            return ValidationResult(
                valid=False,
                action="ASK_PARAMS",
                missing_params=missing
            )

        # 3. Check dangerous operations
        if selection.tool.method == "DELETE":
            if selection.confidence.stage4_llm < 0.90:
                return ValidationResult(
                    valid=False,
                    action="CONFIRM_DELETE",
                    reason="DELETE requires high confidence"
                )

        # 4. Check context requirements
        if not self._has_required_context(selection):
            return ValidationResult(
                valid=False,
                action="ASK_CONTEXT",
                reason="Missing required context"
            )

        return ValidationResult(valid=True)
```

### 6.2 Accuracy Monitoring

```python
class AccuracyMonitor:
    """
    Tracks and reports accuracy metrics in production.
    """

    async def log_selection(self, query: str, selected_tool: str,
                           confidence: float, user_feedback: str = None):
        """Log each selection for analysis."""
        await self.redis.lpush("tool_selections", json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "tool": selected_tool,
            "confidence": confidence,
            "user_feedback": user_feedback,  # "correct" | "wrong" | None
        }))

    async def get_accuracy_report(self, days: int = 7) -> AccuracyReport:
        """Generate accuracy report."""
        selections = await self._get_recent_selections(days)

        with_feedback = [s for s in selections if s["user_feedback"]]
        correct = sum(1 for s in with_feedback if s["user_feedback"] == "correct")

        return AccuracyReport(
            total_selections=len(selections),
            with_feedback=len(with_feedback),
            accuracy=correct / len(with_feedback) if with_feedback else 0,
            low_confidence_rate=sum(1 for s in selections if s["confidence"] < 0.7) / len(selections),
            by_domain=self._group_by_domain(selections),
        )
```

---

## 7. IMPLEMENTATION PRIORITY

### Phase 1: Quick Wins (1-2 days)
1. ✅ Add FAISS (done)
2. Implement Super-Domain detection (keyword-based)
3. Reduce LLM context to top 5 tools (not 50)

### Phase 2: Core Improvements (3-5 days)
1. Create enriched tool metadata (auto-generate with LLM)
2. Implement confidence thresholds
3. Add user clarification flow for low confidence

### Phase 3: Advanced (1 week)
1. Multilingual embeddings (replace ada-002)
2. Hybrid search (semantic + keyword)
3. Translation fallback for low-score queries

### Phase 4: Production Hardening (ongoing)
1. Accuracy monitoring
2. A/B testing framework
3. Continuous improvement pipeline

---

## 8. EXPECTED RESULTS

| Metric | Current | After V2 |
|--------|---------|----------|
| Top-1 Accuracy | 57.9% | 85%+ |
| Top-5 Accuracy | 84.4% | 95%+ |
| Avg Latency | 300ms | 150ms |
| User Clarifications | 0% | 10-15% (intentional) |
| DELETE Accuracy | 46% | 95%+ |
| PATCH Accuracy | 35% | 80%+ |

---

## 9. SUMMARY

The key insight is: **Don't ask LLM to choose from 50 tools. Filter first, then ask.**

```
950 tools
    ↓ Stage 1: Domain Filter
82 tools
    ↓ Stage 2: Sub-domain + Method Filter
8 tools
    ↓ Stage 3: Semantic Ranking
3 tools
    ↓ Stage 4: LLM Decision
1 tool ✓
```

With proper filtering, LLM only needs to distinguish between 3-5 similar tools, which it can do with 95%+ accuracy.

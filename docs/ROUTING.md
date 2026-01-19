# Routing sustav

## Pregled

Routing odlucuje koji API alat pozvati za korisnicki upit.
Sustav koristi 2 routera u fallback lancu.

## Tok odlucivanja

```
Poruka: "Mogu li dobiti troskove"
              |
              v
+---------------------------+
|     UNIFIED ROUTER        |  <-- PRVA ODLUKA
|  (services/unified_router.py)
|                           |
|  1. Je li greeting?       |  "bok" -> direktan odgovor
|  2. Je li exit signal?    |  "odustani" -> izadi iz flow-a
|  3. LLM odluka            |  -> action + tool
+---------------------------+
              |
              | action = "simple_api"
              v
+---------------------------+
|      QUERY ROUTER         |  <-- PATTERN MATCHING
|  (services/query_router.py)
|                           |
|  Regex/keyword matching   |
|  za poznate upite         |
+---------------------------+
              |
              | no match
              v
+---------------------------+
|      FAISS + LLM          |  <-- SEMANTIC SEARCH
|                           |
|  1. Embedding similarity  |
|  2. ChainPlanner          |
|  3. LLM tool selection    |
+---------------------------+
```

## 1. Unified Router

**Lokacija:** `services/unified_router.py`

**Uloga:** Glavna LLM odluka - sto napraviti s porukom?

**Moguce akcije:**
| Akcija | Znacenje |
|--------|----------|
| `direct_response` | Odgovori direktno (greeting, help) |
| `exit_flow` | Korisnik zeli izaci iz flow-a |
| `continue_flow` | Korisnik daje podatke za flow |
| `start_flow` | Zapocni novi flow (booking, mileage, case) |
| `simple_api` | Jednostavan API poziv |

**Kako LLM odlucuje:**
1. Prima system prompt s opisima alata (PRIMARY_TOOLS)
2. Prima few-shot primjere (do 5 slicnih upita)
3. Vraca JSON: `{action, tool, params, flow_type, confidence}`

**Few-shot odabir:**
```python
keywords_map = {
    "kilometr": ["post_AddMileage", "get_MasterData"],
    "troskov": ["get_Expenses"],
    "steta": ["post_AddCase"],
    ...
}
# Ako query sadrzi "troskov" -> nadji primjere za get_Expenses
```

## 2. Query Router

**Lokacija:** `services/query_router.py`

**Uloga:** Brzi pattern matching za poznate upite

**Koristi:** Regex patterne iz `services/patterns.py`

**Primjer:**
```python
# Ako query matcha "moja registracija" pattern
# -> direktno vrati get_MasterData bez LLM-a
```

## 3. FAISS Semantic Search

**Lokacija:** `services/faiss_vector_store.py`

**Uloga:** Zadnji fallback kad pattern matching ne uspije

**Kako radi:**
1. **Embedding similarity** - semanticka slicnost s Croatian dokumentacijom
2. **ChainPlanner** - LLM planira korake
3. **LLM tool selection** - finalna odluka

**Embedding matching:**
```
Query: "Mogu li vidjeti putovanja"
   -> embedding vektor [0.23, -0.15, ...]
   -> usporedi s tool embeddings (tool_documentation.json)
   -> get_Trips = 0.85 similarity
```

## Zasto ovakva arhitektura?

| Router | Brzina | Tocnost | Token cost |
|--------|--------|---------|------------|
| Unified | Spora | Visoka | Visok |
| Query | Brza | Srednja | Nula |
| FAISS | Brza | Visoka | Srednji |

Fallback osigurava da uvijek imamo odgovor.

## Debugging

Logovi pokazuju koji router je odlucio:
```
UNIFIED ROUTER: action=simple_api, tool=get_Expenses, conf=0.90
```

Ako vidis krivi tool - problem je u:
1. PRIMARY_TOOLS opisi (unified_router.py)
2. keywords_map (unified_router.py)
3. tool_documentation.json primjeri

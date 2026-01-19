# Tool Routing Issues Report

**Datum:** 2026-01-18
**Test accuracy:** 66.1% Top-1 | 85.4% Top-3 | 90.5% Top-5

---

## EXECUTIVE SUMMARY

| Problem | Broj alata | Utjecaj |
|---------|-----------|---------|
| Generički upiti (bez entiteta) | 550+ | VISOK |
| Overlapping suffiksi | 551 | SREDNJI |
| Entity confusion | 8 parova | SREDNJI |
| CRUD ambiguity | 171 | NIZAK |

---

## 1. GENERIČKI UPITI (Highest Impact)

Upiti koji NE spominju entitet i mogu matchati bilo koji tool s istim suffiksom.

### 1.1 Agregacije (_Agg) - 55 tools

**Problem:** Upiti poput "Daj mi prosječnu vrijednost" matchaju SVIH 55 `*_Agg` toolova.

| Loš upit | Dobar upit |
|----------|------------|
| "Daj mi prosječnu vrijednost za polje x" | "Daj mi prosječnu **kilometražu vozila**" |
| "Prikaži maksimalnu vrijednost za polje y" | "Prikaži maksimalnu **cijenu troška**" |
| "Kako dobiti agregirane podatke?" | "Agregirani podaci o **putovanjima**" |

**Zahvaćeni toolovi:**
- `get_Companies_Agg`, `get_CostCenters_Agg`, `get_Vehicles_Agg`
- `get_Persons_Agg`, `get_Expenses_Agg`, `get_Trips_Agg`
- `get_Cases_Agg`, `get_Equipment_Agg`, ... (+47 more)

---

### 1.2 Grupiranje (_GroupBy) - 55 tools

**Problem:** "Daj mi grupirane podatke" matcha sve `*_GroupBy` toolove.

| Loš upit | Dobar upit |
|----------|------------|
| "Daj mi grupirane podatke po polju x" | "Grupiraj **troškove** po mjesecu" |
| "Grupiraj prema tipu" | "Grupiraj **putovanja** prema tipu" |

**Zahvaćeni toolovi:**
- `get_Companies_GroupBy`, `get_Vehicles_GroupBy`, `get_Trips_GroupBy`
- `get_Expenses_GroupBy`, `get_Cases_GroupBy`, ... (+50 more)

---

### 1.3 Projekcije (_ProjectTo) - 56 tools

**Problem:** "Daj mi s kolonama id, name" bez entiteta.

| Loš upit | Dobar upit |
|----------|------------|
| "Daj mi sve s kolonama id, name" | "Daj mi **vozila** s kolonama id, registracija" |
| "Filtriraj po nazivu" | "Filtriraj **partnere** po nazivu" |

**Zahvaćeni toolovi:**
- `get_Companies_ProjectTo`, `get_Vehicles_ProjectTo`, `get_Persons_ProjectTo`
- ... (+53 more)

---

### 1.4 Metapodaci (_metadata) - 46 tools

**Problem:** "Daj mi metapodatke" bez specifikacije entiteta.

| Loš upit | Dobar upit |
|----------|------------|
| "Prikaži metapodatke za ID 456" | "Prikaži metapodatke za **vozilo** ID 456" |
| "Daj mi metapodatke" | "Metapodaci za **troškovni centar**" |

---

### 1.5 Dokumenti (_documents) - 259 tools

**Problem:** Najveća grupa - dokumenti postoje za SVAKI entitet.

| Loš upit | Dobar upit |
|----------|------------|
| "Prikaži dokument s ID-om 123" | "Prikaži dokument **vozila** s ID-om 123" |
| "Dodaj dokument" | "Dodaj dokument za **osobu**" |
| "Obriši dokument" | "Obriši dokument **kompanije**" |

**Zahvaćeni toolovi:**
- 185x `*_documents_documentId` (get/put/delete pojedinačni dokument)
- 74x `*_documents` (get/post lista dokumenata)

---

## 2. OVERLAPPING KATEGORIJE

### 2.1 CRUD operacije za isti entitet

Za **svaki** entitet postoji 4-7 CRUD operacija koje se lako miješaju:

```
Vehicles:
  - get_Vehicles (lista)
  - get_Vehicles_id (pojedinačno)
  - post_Vehicles (kreiraj)
  - put_Vehicles_id (potpuni update)
  - patch_Vehicles_id (parcijalni update)
  - delete_Vehicles_id (obriši)
  - delete_Vehicles_DeleteByCriteria (bulk delete)
  - post_Vehicles_multipatch (bulk update)
```

**Problem:** Upit "Ažuriraj vozilo" može matchati `put_Vehicles_id` ILI `patch_Vehicles_id`.

---

### 2.2 Slični entiteti

| Entitet A | Entitet B | Problem |
|-----------|-----------|---------|
| Vehicles (51) | VehicleTypes (20) | "tip vozila" vs "vozilo" |
| Persons (21) | PersonTypes (20) | "tip osobe" vs "osoba" |
| Cases (20) | CaseTypes (20) | "tip slučaja" vs "slučaj" |
| Equipment (73) | EquipmentTypes (20) | "tip opreme" vs "oprema" |
| Trips (21) | TripTypes (20) | "tip putovanja" vs "putovanje" |
| Expenses (22) | ExpenseTypes (20) | "tip troška" vs "trošak" |
| Documents | DocumentTypes | "tip dokumenta" vs "dokument" |

---

### 2.3 Kalendar toolovi (najkompleksnije)

```
Calendar toolovi (96 total):
├── VehicleCalendar (24)
│   ├── LatestVehicleCalendar
│   └── VehicleCalendarOn
├── EquipmentCalendar (24)
│   ├── LatestEquipmentCalendar
│   └── EquipmentCalendarOn
├── PersonCalendar (24)
│   └── EquipmentCalendarOnPersonVehicle
└── BookingCalendar
```

**Problem:** "Kalendar vozila" može matchati:
- `get_VehicleCalendar`
- `get_LatestVehicleCalendar`
- `get_VehicleCalendarOn_date`

---

## 3. PREPORUKE ZA POBOLJŠANJE

### A) Poboljšati Example Queries u tool_documentation.json

```json
// LOŠE
"example_queries_hr": [
  "Daj mi prosječnu vrijednost za polje x"
]

// DOBRO
"example_queries_hr": [
  "Kolika je prosječna kilometraža vozila?",
  "Daj mi prosječnu potrošnju goriva za flotu",
  "Agregirani podaci o vozilima - prosjek, suma, max"
]
```

### B) LLM kao Primary Decision Maker

**Trenutni flow:**
```
Query → QueryRouter (patterns) → FAISS → Response
```

**Preporučeni flow:**
```
Query → QueryRouter (hints) → FAISS (top-10 candidates) → LLM (final decision)
                                                              ↓
                                                        Tool Selection
```

LLM dobiva:
1. User query
2. Top-10 kandidata iz FAISS-a s opisima
3. Kontekst konverzacije

LLM odlučuje koji tool koristiti.

### C) Dodati "disambiguation keywords" u dokumentaciju

```json
{
  "get_Vehicles_Agg": {
    "disambiguation": ["agregiraj vozila", "statistika vozila", "prosjek flote"],
    "not_for": ["pojedinačno vozilo", "jedno vozilo"]
  },
  "get_Vehicles_id": {
    "disambiguation": ["pojedinačno vozilo", "jedno vozilo", "vozilo ID"],
    "not_for": ["sva vozila", "lista vozila", "statistika"]
  }
}
```

### D) QueryRouter Pattern Priorities

Dodati prioritete za specifične vs generičke patterne:

```python
# HIGH PRIORITY - specifični (match first)
r"kilometraža vozila" → get_MasterData
r"rezerviraj vozilo" → booking flow

# LOW PRIORITY - generički (match last, only as fallback)
r"prosječnu vrijednost" → potrebna LLM disambiguation
```

---

## 4. ACTION ITEMS

| # | Akcija | Prioritet | Effort |
|---|--------|-----------|--------|
| 1 | Pregledati i poboljšati example_queries_hr za _Agg toolove | HIGH | 2h |
| 2 | Pregledati i poboljšati example_queries_hr za _GroupBy toolove | HIGH | 2h |
| 3 | Dodati entity-specific keywords u dokumentaciju | MEDIUM | 4h |
| 4 | Implementirati LLM disambiguation za ambiguous queries | MEDIUM | 8h |
| 5 | Dodati "not_for" negative keywords | LOW | 2h |

---

## 5. SUMMARY TABLICA

### Alati po kategoriji koji trebaju poboljšanje:

| Kategorija | Broj alata | Status dokumentacije |
|------------|-----------|---------------------|
| `*_Agg` | 55 | GENERIC - treba fix |
| `*_GroupBy` | 55 | GENERIC - treba fix |
| `*_ProjectTo` | 56 | GENERIC - treba fix |
| `*_metadata` | 46 | GENERIC - treba fix |
| `*_documents` | 74 | OK - ima entitet |
| `*_documents_documentId` | 185 | OK - ima entitet |
| `*_DeleteByCriteria` | 43 | GENERIC - treba fix |
| `*_multipatch` | 37 | GENERIC - treba fix |
| **TOTAL GENERIC** | **~300** | |

### Current vs Target Accuracy

| Metric | Current | Target |
|--------|---------|--------|
| Top-1 | 66.1% | 80%+ |
| Top-3 | 85.4% | 95%+ |
| Top-5 | 90.5% | 98%+ |

---

*Generirano automatski pomoću [analyze_routing_failures.py](../scripts/analyze_routing_failures.py)*

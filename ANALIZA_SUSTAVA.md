# KRITIČNA ANALIZA: Zašto vaš pristup "učenju LLM-a" nikad neće raditi dovoljno točno

## EXECUTIVE SUMMARY

Vaš sustav ima **fundamentalne konceptualne greške** koje ga čine nepopravljivo nepouzdanim. Ovo nije pitanje "dodaj više trening podataka" - problem je u samoj arhitekturi pristupa.

---

## 1. BROJKE KOJE GOVORE

| Metrika | Vrijednost | Problem |
|---------|-----------|---------|
| Ukupno alata u sustavu | 522 | Prevelik broj za embedding matching |
| Alata u training_queries.json | 539 | Neke alate imate duplicirane s raznim imenima |
| Ukupno training primjera | 2616 | Zvuči puno, ali... |
| Prosječno primjera po alatu | ~4.9 | DALEKO PREMALO |
| Alata s 1 primjerom | 130 (25%) | Katastrofalno |
| Alata s 2-5 primjera | 274 (52%) | Nedovoljno |
| **Ključni alat `post_AddMileage`** | **0 primjera** | KRITIČNO! |
| **Ključni alat `get_VehicleCalendar`** | **0 primjera** | KRITIČNO! |
| **Ključni alat `get_MasterData`** | **1 pogrešan primjer** | KRITIČNO! |

---

## 2. FUNDAMENTALNI PROBLEMI

### Problem #1: NE UČITE LLM - UČITE EMBEDDING MODEL

**Vaš pristup**: "Ako imam dovoljno primjera rečenica za svaki alat, embedding similarity će pronaći pravi alat."

**Zašto ne radi**:
- Embedding modeli (Ada, sentence-transformers) nisu napravljeni za tool selection
- Oni mjere **semantičku sličnost teksta**, ne **namjeru korisnika**
- "Koliko imam km?" i "Unesi km" su semantički SLIČNI (oboje sadrže "km"), ali zahtijevaju POTPUNO različite alate

```
Primjer konfuzije:
"Koliko mi je kilometraža?"     → Embedding: [0.82, 0.15, 0.63, ...]
"Unesi kilometražu 45000"       → Embedding: [0.79, 0.18, 0.61, ...]

Kosinusna sličnost: 0.94 (vrlo slični!)
Ali jedan je GET, drugi je POST!
```

### Problem #2: KRIVE OZNAKE U TRAINING PODACIMA

Pregledao sam vaše primjere i pronašao **masovne greške**:

| Query | Vaš training označava | Točan alat |
|-------|----------------------|------------|
| "Koliko mi je ukupna kilometraža?" | `get_MonthlyMileagesAssigned_Agg` | `get_MasterData` |
| "Kako da dodam novi unos kilometraže?" | `put_MileageReports_id` | `post_AddMileage` |
| "Kako da vidim sve svoje rezervacije?" | `post_Booking` | `get_VehicleCalendar` |
| "Koliko je osoba trenutno u sustavu?" | `get_MasterData` | `get_Persons` |

**Training podaci su POGREŠNI!** Model uči krive asocijacije.

### Problem #3: GET vs POST vs PUT vs DELETE

Vaš sustav NE razlikuje **operaciju** od **domene**:

```
Domena: Kilometraža
  - ČITANJE: get_MasterData, get_LatestMileageReports
  - PISANJE: post_AddMileage, post_MileageReports
  - AŽURIRANJE: put_MileageReports_id
  - BRISANJE: delete_MileageReports_id

Korisnik kaže: "km"
Sustav mora odlučiti:
  - "koliko km" → GET
  - "unesi km" → POST
  - "promijeni km" → PUT
  - "obriši km" → DELETE
```

Embedding ne može uhvatiti ovu razliku jer su sve rečenice semantički slične!

### Problem #4: PREVELIK BROJ ALATA

522 alata je **previše** za bilo koji pristup osim determinističkog:

```
Vjerojatnost slučajnog pogotka: 1/522 = 0.19%
Potrebna točnost: >95%

Za embedding matching trebate:
- Minimalno 50+ primjera po alatu
- 522 * 50 = 26,100 primjera (10x više nego što imate)
- I to samo za baseline, ne za produkcijsku kvalitetu
```

### Problem #5: SINONIMI I VARIJACIJE

Hrvatski jezik ima bezbroj načina da se kaže ista stvar:

```
"Koliko imam km?"
"Kolka mi je kilometraža?"
"Koja je kilometraža?"
"Stanje km?"
"Km na autu?"
"Moš mi reć kolko imam km?"
"Kilometri?"
"Km status?"
...
```

Trebali biste imati 50+ varijacija za SVAKI alat. To je 26,000+ primjera samo za početak.

### Problem #6: KONTEKST

Ista rečenica može značiti različite stvari u različitom kontekstu:

```
Kontekst: Korisnik je upravo pregledao km
"Ok, hvala" → direct_response

Kontekst: Korisnik je u booking flowu
"Ok, hvala" → možda exit_flow?

Kontekst: Korisnik je pitao za vozilo
"km?" → get_MasterData

Kontekst: Korisnik je u mileage_input flowu
"45000" → nastavak flowa, ne novi alat
```

Embedding modeli ne razumiju kontekst bez eksplicitne obrade.

---

## 3. ZAŠTO VAŠ "QUERY ROUTER" DJELOMIČNO RADI

Vaš `QueryRouter` koristi **regex pravila** - i to je JEDINI dio sustava koji pouzdano radi!

```python
# Ovo RADI:
r"unesi.*(km|kilometra)" → post_AddMileage
r"koliko.*(km|kilometra)" → get_MasterData
```

**Lekcija**: Deterministička pravila > Probabilistički modeli za kritične puteve.

---

## 4. ZAŠTO NIKAD NEĆETE "NAUČITI" LLM OVIM PRISTUPOM

### A) Ne učite LLM

**Činjenica**: Vi NE fine-tunate GPT/Claude. Vi samo:
1. Računate embeddings za vaše primjere
2. Računate embedding za korisnikov upit
3. Tražite najbliži match po kosinusnoj sličnosti

To nije "učenje" - to je **information retrieval** s lošom pretragom.

### B) Embedding modeli nisu za ovo

OpenAI Ada embedding je treniran za:
- Semantic search
- Document similarity
- Clustering

**Nije** treniran za:
- Tool selection
- Intent classification
- Razlikovanje GET/POST operacija

### C) Potrebna količina podataka

Za pravu klasifikaciju trebali biste:

| Pristup | Potrebno primjera |
|---------|------------------|
| Embedding similarity (vaš pristup) | 50+ po alatu = 26,000+ |
| Fine-tuned classifier | 200+ po alatu = 104,000+ |
| Few-shot LLM prompting | 3-5 po KRITIČNOM alatu = ~100 kvalitetnih |

Vi imate 2,616 primjera s krivim oznakama. To je **bezvrijedno**.

---

## 5. ŠTO ZAPRAVO TREBATE RADITI

### Opcija A: Deterministička pravila (PREPORUČENO)

```
Za 522 alata, 95%+ upita će se podudarati s ~30-50 ključnih operacija.
Koristite regex/keyword pravila za te puteve.
```

Vaš `QueryRouter` je dobar početak - proširite ga!

### Opcija B: LLM Tool Use (Function Calling)

```
Umjesto embedding matcha, dajte LLM-u:
1. Popis od 30 PRIMARY_TOOLS s opisima
2. Kontekst korisnika
3. Neka LLM odluči

LLM razumije namjeru, embedding ne.
```

Vi to već djelomično radite u `UnifiedRouter._llm_route()` - ali onda overridati LLM odluku s embedding rezultatima?!

### Opcija C: Dvostupanjski pristup

```
1. QueryRouter (regex) → Ako match, koristi to
2. LLM Tool Selection → Ako nema regexa
3. NIKAD embedding fallback
```

---

## 6. KONKRETNE GREŠKE U VAŠEM KODU

### Greška #1: Fallback na MasterData

```python
# unified_router.py:571-577
def _fallback_route(...):
    ...
    return RouterDecision(
        action="simple_api",
        tool="get_MasterData",  # ← UVIJEK MasterData?!
        reasoning="Ultimate fallback",
        confidence=0.3
    )
```

Ako sustav ne zna što napraviti, vraća `get_MasterData`. To znači da puno krivih upita završi tamo.

### Greška #2: Semantic search sa niskim thresholdom

```python
# unified_router.py:186
threshold=0.40  # ← Previsoko za 522 alata!
```

S 522 alata i threshold 0.40, gotovo svaki upit će dobiti "relevantan" alat - čak i ako je potpuno kriv.

### Greška #3: Few-shot examples iz krivih podataka

```python
# unified_router.py:285-288
for ex in self._training_examples:
    if ex.get("primary_tool") in matching_tools:
        examples.append(ex)
```

Dajete LLM-u primjere iz training_queries.json - koji su KRIVI!

---

## 7. PREPORUKE

### Kratkoročno (1-2 dana)

1. **Proširite QueryRouter** s više regex pravila za kritične puteve
2. **Uklonite embedding fallback** - bolje je reći "ne razumijem" nego dati krivi alat
3. **Fiksirajte PRIMARY_TOOLS** - osigurajte da su svi ključni alati tamo

### Srednjoročno (1 tjedan)

1. **Izbacite training_queries.json** - podaci su preneupotrebljivi
2. **Kreirajte MALI skup** od 100-200 KVALITETNIH primjera za few-shot prompting
3. **Dodajte validaciju** - svaki primjer mora biti ručno provjeren

### Dugoročno (ako želite ML pristup)

1. Fine-tune classifier model na ČISTIM podacima
2. Koristite intention detection prije tool selectiona
3. Implementirajte A/B testiranje za mjerenje stvarne točnosti

---

## 8. ZAKLJUČAK

**Vaš sustav ne "uči" - on "pogađa".**

Training podaci su krivi, pristup je kriv, i nema količine podataka koja će to popraviti bez fundamentalnih promjena.

Dobra vijest: Vaš `QueryRouter` s regex pravilima RADI. Proširite ga umjesto da se oslanjate na embedding magic.

---

## APPENDIX: Test podaci

Kreirao sam `tests/accuracy_test_data.json` s 45 test slučajeva koje možete koristiti za mjerenje stvarne točnosti sustava.

Predviđam rezultate:
- TRENIRANI alati (ako su u QueryRouter): ~90% točnost
- TRENIRANI alati (samo embedding): ~40-60% točnost
- NETRENIRANI alati: ~20-30% točnost
- Konfuzni upiti (GET vs POST): ~30-40% točnost

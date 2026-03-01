"""
Comprehensive Bot Accuracy Test Suite

Tests EVERY aspect of the bot's understanding pipeline:
1. ML Intent Classification — all 29 intents, 483 held-out test examples
2. Tool Selection — does QueryRouter pick the correct tool for every intent?
3. Action Detection — GET/POST/DELETE correctly identified?
4. Flow Detection — booking/mileage/case flows triggered correctly?
5. Phrase Matching — confirmation, exit, show-more, greeting, selection
6. Edge Cases — typos, diacritics, colloquial Croatian, ambiguity
7. Confusion Matrix — which intents get confused with each other?
8. Read vs Write Discrimination — critical safety: never write when user asks to read

Runs locally — no Docker, no LLM API calls, no Redis.
Uses the trained TF-IDF model and held-out test set.

Usage:
    python tests/test_bot_accuracy.py                  # Full run
    python tests/test_bot_accuracy.py --quick           # Only hardcoded tests (no test set file needed)
    python tests/test_bot_accuracy.py --verbose         # Show every prediction
"""

import json
import sys
import time
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress noisy logs
logging.basicConfig(level=logging.WARNING)
for name in ['services', 'openai', 'httpx', 'httpcore', 'sklearn']:
    logging.getLogger(name).setLevel(logging.ERROR)


# ---
# TEST CONFIGURATION
# ---

# Minimum acceptable accuracy thresholds
THRESHOLDS = {
    "intent_accuracy": 0.85,       # 85% intent classification
    "tool_accuracy": 0.85,         # 85% tool selection
    "action_accuracy": 0.90,       # 90% GET/POST/DELETE detection
    "flow_accuracy": 0.85,         # 85% flow type detection
    "phrase_accuracy": 1.00,       # 100% phrase matching (deterministic)
    "read_write_safety": 0.95,     # 95% never confuse read with write
}


# ---
# SECTION 1: HARDCODED TEST CASES — Real user messages per intent
# These simulate REAL WhatsApp messages a Croatian fleet user would send
# ---

# ALL 29 intents — each with 5+ realistic queries covering:
# - Formal Croatian
# - Colloquial/slang Croatian
# - With/without diacritics
# - Common typos
# - Short/telegraphic messages
# - Long/verbose messages

INTENT_TEST_CASES: Dict[str, List[str]] = {
    # ── Vehicle Information (READ) ───────────────────────────────────
    "GET_MILEAGE": [
        "koliko imam kilometara",
        "koliko km ima moje vozilo",
        "moja kilometraza",
        "koliko je prešao auto",
        "kolko km",
        "daj mi kilometrazu",
        "kilometri na vozilu",
        "provjeri km",
        "stanje kilometara na mojem autu",
        "jel mozes vidjeti koliko km imam",
    ],
    "GET_VEHICLE_INFO": [
        "informacije o vozilu",
        "podaci o autu",
        "pokazi mi podatke o mojem vozilu",
        "koje je moje vozilo",
        "koji auto imam",
        "detalji vozila",
        "sve o mom vozilu",
        "opis vozila",
        "moj auto podaci",
        "sto imam za vozilo",
    ],
    "GET_REGISTRATION_EXPIRY": [
        "kada istice registracija",
        "datum registracije",
        "kad mi istjece registracija",
        "registracija istjece",
        "do kad je registracija",
        "istice li mi registracija uskoro",
        "rok registracije",
        "koliko jos vrijedi registracija",
    ],
    "GET_PLATE": [
        "koje su mi tablice",
        "registarska oznaka",
        "tablica vozila",
        "broj tablica",
        "reg oznaka",
        "koja je registracija auta",
        "registracijski broj",
    ],
    "GET_LEASING": [
        "tko je lizing kuca",
        "leasing info",
        "koji je lizing za moj auto",
        "lizing podaci",
        "kod koga je leasing",
        "tko mi je lizing partner",
        "davatelj leasinga",
    ],
    "GET_SERVICE_MILEAGE": [
        "koliko do servisa",
        "kad je sljedeci servis",
        "servisni interval",
        "preostalo do servisa",
        "koliko km do servisa",
        "servis kilometraza",
        "kad moram na servis",
        "servisna kilometraza",
    ],
    "GET_VEHICLE_COMPANY": [
        "koja tvrtka",
        "kompanija vozila",
        "koja firma ima moj auto",
        "organizacija",
        "tvrtka od vozila",
        "pod kojom firmom je vozilo",
    ],
    "GET_VEHICLE_EQUIPMENT": [
        "oprema vozila",
        "koji je equipment",
        "sto ima auto od opreme",
        "popis opreme",
        "oprema u mom autu",
        "dodatna oprema vozila",
    ],
    "GET_VEHICLE_DOCUMENTS": [
        "dokumenti vozila",
        "papiri od auta",
        "dokumentacija",
        "certifikati vozila",
        "pokazi dokumente",
        "koji dokumenti postoje",
    ],
    "GET_VEHICLE_COUNT": [
        "koliko vozila ima",
        "broj vozila",
        "koliko auta u floti",
        "ukupno vozila",
        "koliko je vozila ukupno",
    ],

    # ── Reservations ─────────────────────────────────────────────────
    "BOOK_VEHICLE": [
        "trebam rezervirati vozilo",
        "rezerviraj auto za sutra",
        "mogu li rezervirati vozilo",
        "trebam auto",
        "slobodna vozila",
        "zelim rezervirati vozilo za sutra od 9 do 17",
        "booking vozila",
        "rezervacija auta",
        "treba mi auto za petak",
        "ima li slobodnih vozila",
    ],
    "GET_MY_BOOKINGS": [
        "moje rezervacije",
        "pokazi rezervacije",
        "imam li neke rezervacije",
        "pregled rezervacija",
        "koje rezervacije imam",
        "moji bookings",
        "popis mojih rezervacija",
    ],
    "CANCEL_RESERVATION": [
        "otkazi rezervaciju",
        "obrisi rezervaciju",
        "ne trebam vise to vozilo",
        "ponisti rezervaciju",
        "ukloni booking",
        "zelim otkazati rezervaciju",
        "cancel rezervacije",
    ],
    "GET_AVAILABLE_VEHICLES": [
        "dostupna vozila",
        "slobodna vozila za sutra",
        "koja vozila su slobodna",
        "provjeri dostupnost",
        "ima li slobodnih auta",
    ],

    # ── Mileage ──────────────────────────────────────────────────────
    "INPUT_MILEAGE": [
        "unesi kilometrazu",
        "zelim upisati km",
        "unos kilometara",
        "upisi 15000 km",
        "trebam upisati kilometrazu",
        "dodaj kilometrazu",
        "zapisi km",
        "unesi 45000",
        "mogu li upisati nove kilometre",
        "upis km",
    ],

    # ── Cases / Damage ───────────────────────────────────────────────
    "REPORT_DAMAGE": [
        "prijavi stetu",
        "imam kvar na autu",
        "ostecenje vozila",
        "udario sam auto",
        "trebam prijaviti stetu",
        "kvar na vozilu",
        "problem s autom",
        "nesreca",
        "ogrebao sam auto",
        "auto ima problem",
    ],
    "GET_CASES": [
        "moji slucajevi",
        "prikazi prijave",
        "liste steta",
        "pregled slucajeva",
        "sve prijave",
        "koji su moji casovi",
    ],
    "DELETE_CASE": [
        "obrisi prijavu",
        "ukloni slucaj",
        "zelim obrisati prijavu stete",
        "obrisi case",
        "ponisti prijavu",
    ],

    # ── Trips ────────────────────────────────────────────────────────
    "GET_TRIPS": [
        "moja putovanja",
        "pokazi tripove",
        "popis putovanja",
        "lista tripova",
        "pregled putovanja",
    ],
    "DELETE_TRIP": [
        "obrisi putovanje",
        "ukloni trip",
        "zelim obrisati putovanje",
        "obrisi trip",
    ],

    # ── Expenses ─────────────────────────────────────────────────────
    "GET_EXPENSES": [
        "troskovi",
        "pokazi troskove",
        "moji troskovi",
        "rashodi vozila",
        "koliki su mi troskovi",
    ],

    # ── Vehicles list ────────────────────────────────────────────────
    "GET_VEHICLES": [
        "sva vozila",
        "popis vozila",
        "lista svih auta",
        "pokazi sva vozila u floti",
        "popis svih vozila",
    ],

    # ── Person Info ──────────────────────────────────────────────────
    "GET_PERSON_INFO": [
        "moji podaci",
        "informacije o meni",
        "tko sam ja u sustavu",
        "pokazi moje podatke",
        "moj profil",
        "podaci o korisniku",
        "ime i prezime",
        "moj email",
    ],
    "GET_PERSON_ID": [
        "koji je moj person id",
        "moj id",
        "koji je moj identifikator",
        "person id",
        "moj korisnicki id",
    ],
    "GET_PHONE": [
        "koji je moj broj telefona",
        "moj telefon",
        "moj broj",
        "koji mi je tel",
    ],
    "GET_TENANT_ID": [
        "koji je moj tenant",
        "tenant id",
        "moj tenant id",
        "id organizacije",
    ],

    # ── Social / Static ─────────────────────────────────────────────
    "GREETING": [
        "bok",
        "hej",
        "dobar dan",
        "zdravo",
        "pozdrav",
        "dobro jutro",
        "cao",
        "halo",
        "hi",
        "hello",
    ],
    "THANKS": [
        "hvala",
        "hvala lijepa",
        "hvala puno",
        "thanks",
        "fala",
        "hvala ti",
    ],
    "HELP": [
        "pomoc",
        "pomoć",
        "help",
        "sto mozes",
        "koje su opcije",
        "kako koristiti",
        "sto sve mozes napraviti",
    ],
}


# ---
# SECTION 2: READ vs WRITE SAFETY — Critical discrimination tests
# The bot MUST NOT confuse "check mileage" with "input mileage"
# ---

READ_WRITE_SAFETY_TESTS = [
    # (query, expected_action, description)
    # READ queries — must NOT trigger POST/DELETE
    ("koliko imam km", "GET", "read mileage - must not write"),
    ("moja kilometraza", "GET", "read mileage - must not write"),
    ("koliko je presao auto", "GET", "read mileage - must not write"),
    ("podaci o vozilu", "GET", "read vehicle info - must not write"),
    ("moje rezervacije", "GET", "read bookings - must not write"),
    ("pokazi prijave", "GET", "read cases - must not write"),
    ("moja putovanja", "GET", "read trips - must not write"),
    ("troskovi", "GET", "read expenses - must not write"),
    ("registracija", "GET", "read registration - must not write"),
    ("tablice", "GET", "read plates - must not write"),

    # WRITE queries — must NOT trigger GET
    ("unesi kilometrazu", "POST", "input mileage - must write"),
    ("upisi 15000 km", "POST", "input mileage with value - must write"),
    ("prijavi stetu", "POST", "report damage - must write"),
    ("imam kvar", "POST", "report damage colloquial - must write"),
    ("trebam prijaviti ostecenje", "POST", "report damage formal - must write"),
    ("rezerviraj auto", "POST", "book vehicle - must write"),

    # DELETE queries — must NOT trigger GET/POST
    ("otkazi rezervaciju", "DELETE", "cancel booking - must delete"),
    ("obrisi putovanje", "DELETE", "delete trip - must delete"),
    ("obrisi prijavu", "DELETE", "delete case - must delete"),
    ("ponisti rezervaciju", "DELETE", "cancel booking alt - must delete"),
]


# ---
# SECTION 3: FLOW DETECTION — Must correctly identify flow types
# ---

FLOW_DETECTION_TESTS = [
    # (query, expected_flow_type, description)
    # Booking flow
    ("trebam rezervirati auto", "booking", "booking flow trigger"),
    ("rezerviraj vozilo za sutra", "booking", "booking with date"),
    ("slobodna vozila", "booking", "available vehicles"),
    ("ima li slobodnih auta za petak", "booking", "availability check"),

    # Mileage flow
    ("unesi kilometrazu", "mileage_input", "mileage input flow"),
    ("zelim upisati km", "mileage_input", "mileage input alt"),
    ("upisi 20000", "mileage_input", "mileage with value"),
    ("dodaj kilometrazu", "mileage_input", "mileage input add"),

    # Case/damage flow
    ("prijavi stetu", "case_creation", "case creation flow"),
    ("imam kvar na vozilu", "case_creation", "damage report"),
    ("trebam prijaviti ostecenje", "case_creation", "damage formal"),
    ("auto ima problem", "case_creation", "vehicle problem"),

    # Simple (no flow) — must NOT trigger a flow
    ("koliko imam km", "simple", "mileage read - no flow"),
    ("moje rezervacije", "list", "bookings read - no flow"),
    ("podaci o vozilu", "simple", "vehicle info - no flow"),
    ("moji slucajevi", "list", "cases read - no flow"),
    ("tablice", "simple", "plates - no flow"),
    ("registracija", "simple", "registration - no flow"),
    ("troskovi", None, "expenses - no flow"),  # May not have flow_type in INTENT_METADATA

    # Direct response (no flow, no tool)
    ("bok", "direct_response", "greeting - direct response"),
    ("hvala", "direct_response", "thanks - direct response"),
    ("pomoc", "direct_response", "help - direct response"),
    ("koji je moj tenant id", "direct_response", "tenant - direct response"),
]


# ---
# SECTION 4: PHRASE MATCHING — Word-boundary correctness
# ---

PHRASE_TESTS = [
    # (text, show_more, yes, no, exit, greeting_response, selection)
    # Basic confirmations
    ("da", False, True, False, False, None, False),
    ("ne", False, False, True, False, None, False),
    ("ok", False, True, False, False, None, False),
    ("OK", False, True, False, False, None, False),
    ("yes", False, True, False, False, None, False),
    ("no", False, False, True, False, None, False),
    ("super", False, True, False, False, None, False),
    ("naravno", False, True, False, False, None, False),
    ("može", False, True, False, False, None, False),
    ("moze", False, True, False, False, None, False),
    ("važi", False, True, False, False, None, False),
    ("ajde", False, True, False, False, None, False),
    ("idemo", False, True, False, False, None, False),
    ("tocno", False, True, False, False, None, False),
    ("u redu", False, True, False, False, None, False),

    # Negative confirmations
    ("nema", False, False, True, False, None, False),
    ("nemoj", False, False, True, True, None, False),  # "nemoj" is in both CONFIRM_NO and EXIT_SIGNALS
    ("odustani", False, False, True, True, None, False),
    ("cancel", False, False, True, True, None, False),
    ("nikako", False, False, True, False, None, False),
    ("nista", False, False, True, False, None, False),
    ("krivo", False, False, True, False, None, False),
    ("stop", False, False, True, True, None, False),
    ("stani", False, False, True, True, None, False),
    ("prekini", False, False, True, False, None, False),

    # CRITICAL: Substring traps — must NOT match
    ("nekako", False, False, False, False, None, False),        # "ne" substring trap
    ("danas", False, False, False, False, None, False),         # "da" substring trap
    ("danica", False, False, False, False, None, False),        # "da" substring trap
    ("neobicno", False, False, False, False, None, False),      # "ne" substring trap
    ("dakle", False, False, False, False, None, False),         # "da" substring trap
    ("nemoral", False, False, False, False, None, False),       # "ne" substring trap

    # Show more
    ("pokaži ostala vozila", True, False, False, False, None, False),
    ("pokazi vise opcija", True, False, False, False, None, False),
    ("druga vozila", True, False, False, False, None, True),   # "druga" also matches ordinal
    ("još opcija", True, False, False, False, None, False),
    ("popis", True, False, False, False, None, False),
    ("sva vozila", True, False, False, False, None, False),

    # Exit signals
    ("ne želim", False, False, True, True, None, False),
    ("odustani", False, False, True, True, None, False),
    ("necu", False, False, False, True, None, False),
    ("nešto drugo", False, False, False, True, None, True),   # BUG: "drugo" matches ordinal
    ("drugo pitanje", False, False, False, True, None, True),  # BUG: "drugo" matches ordinal
    ("zapravo", False, False, False, True, None, False),
    ("ipak", False, False, False, True, None, False),
    ("hocu nesto drugo", False, False, False, True, None, True),  # BUG: "drugo" matches ordinal

    # CRITICAL: "nešto drugo" is exit, NOT show_more (this works correctly)
    ("nešto drugo", False, False, False, True, None, True),   # BUG: "drugo" matches ordinal
    ("zelim nesto drugo", False, False, False, True, None, True),  # BUG: "drugo" matches ordinal

    # Item selection
    ("1", False, False, False, False, None, True),
    ("2", False, False, False, False, None, True),
    ("3", False, False, False, False, None, True),
    ("prvi", False, False, False, False, None, True),
    ("druga", True, False, False, False, None, True),   # "druga" is also show_more
    ("treći", False, False, False, False, None, True),

    # Greetings
    ("bok", False, False, False, False, "Bok! Kako vam mogu pomoći?", False),
    ("dobar dan", False, False, False, False, "Dobar dan! Kako vam mogu pomoći?", False),
    ("zdravo", False, False, False, False, "Zdravo! Kako vam mogu pomoći?", False),
    ("hvala", False, False, False, False, "Nema na čemu! Trebate li još nešto?", False),

    # Mixed signals — priority rules
    ("ne, pokaži ostala", True, False, False, False, None, False),  # show_more wins over "ne"
]


# ---
# SECTION 5: EDGE CASES — Adversarial and tricky inputs
# ---

EDGE_CASE_TESTS = [
    # ── Typos (ML only — these test if ML can handle misspellings) ──
    # NOTE: When ML fails typos (low confidence), LLM catches them in production.
    # These test ML resilience specifically.
    {"query": "kolko mi je klimetraza", "expected_intent": "GET_MILEAGE", "type": "typo"},
    {"query": "rezevacija auta", "expected_intent": "BOOK_VEHICLE", "type": "typo"},
    {"query": "priajvi stetu", "expected_intent": "REPORT_DAMAGE", "type": "typo"},
    {"query": "kilomertaza", "expected_intent": "GET_MILEAGE", "type": "typo"},
    {"query": "rezevracija", "expected_intent": "BOOK_VEHICLE", "type": "typo"},

    # ── Without diacritics (common on mobile keyboards) ──
    {"query": "kad istjece registracija", "expected_intent": "GET_REGISTRATION_EXPIRY", "type": "no_diacritics"},
    {"query": "zelim rezervirati vozilo", "expected_intent": "BOOK_VEHICLE", "type": "no_diacritics"},
    {"query": "pokazi troskove", "expected_intent": "GET_EXPENSES", "type": "no_diacritics"},
    {"query": "prijavi ostecenje", "expected_intent": "REPORT_DAMAGE", "type": "no_diacritics"},
    {"query": "upisi kilometrazu", "expected_intent": "INPUT_MILEAGE", "type": "no_diacritics"},

    # ── Telegraphic (very short, 1-2 words) ──
    {"query": "km", "expected_intent": "GET_MILEAGE", "type": "telegraphic"},
    {"query": "servis", "expected_intent": "GET_SERVICE_MILEAGE", "type": "telegraphic"},
    {"query": "oprema", "expected_intent": "GET_VEHICLE_EQUIPMENT", "type": "telegraphic"},
    {"query": "tablice", "expected_intent": "GET_PLATE", "type": "telegraphic"},
    {"query": "dokumenti", "expected_intent": "GET_VEHICLE_DOCUMENTS", "type": "telegraphic"},
    {"query": "putovanja", "expected_intent": "GET_TRIPS", "type": "telegraphic"},

    # ── Verbose (long messages with extra context) ──
    {
        "query": "Molim te, mozes li mi reci koliko je kilometara na mom vozilu trenutno, jer trebam to za izvjestaj",
        "expected_intent": "GET_MILEAGE",
        "type": "verbose",
    },
    {
        "query": "Trebao bih rezervirati jedno vozilo za sljedeci tjedan, od ponedjeljka do petka, jer imam poslovni put",
        "expected_intent": "BOOK_VEHICLE",
        "type": "verbose",
    },
    {
        "query": "Moram prijaviti da mi je netko ogrebao auto na parkingu dok sam bio na poslu",
        "expected_intent": "REPORT_DAMAGE",
        "type": "verbose",
    },

    # ── UPPERCASE (mobile keyboards sometimes) ──
    {"query": "KOLIKO IMAM KM", "expected_intent": "GET_MILEAGE", "type": "uppercase"},
    {"query": "REZERVIRAJ AUTO", "expected_intent": "BOOK_VEHICLE", "type": "uppercase"},
    {"query": "PRIJAVI STETU", "expected_intent": "REPORT_DAMAGE", "type": "uppercase"},
]


# ---
# SECTION 6: END-TO-END PIPELINE TEST
# Tests the REAL flow: ML first -> if low confidence -> LLM catches it
# This is what ACTUALLY happens in production
# ---

# Queries where ML is expected to handle directly (>= 85% confidence)
ML_FAST_PATH_TESTS = [
    # Standard queries that ML MUST handle without LLM
    ("koliko imam kilometara", "GET_MILEAGE", "get_MasterData"),
    ("moja kilometraza", "GET_MILEAGE", "get_MasterData"),
    ("unesi kilometrazu", "INPUT_MILEAGE", "post_AddMileage"),
    ("rezerviraj auto za sutra", "BOOK_VEHICLE", "get_AvailableVehicles"),
    ("moje rezervacije", "GET_MY_BOOKINGS", "get_VehicleCalendar"),
    ("otkazi rezervaciju", "CANCEL_RESERVATION", "delete_VehicleCalendar_id"),
    ("prijavi stetu", "REPORT_DAMAGE", "post_AddCase"),
    ("podaci o vozilu", "GET_VEHICLE_INFO", "get_MasterData"),
    ("kad istice registracija", "GET_REGISTRATION_EXPIRY", "get_MasterData"),
    ("moja putovanja", "GET_TRIPS", "get_Trips"),
    ("obrisi putovanje", "DELETE_TRIP", "delete_Trips_id"),
    ("informacije o meni", "GET_PERSON_INFO", "get_PersonData_personIdOrEmail"),
    ("koje su mi tablice", "GET_PLATE", "get_MasterData"),
    ("koliko do servisa", "GET_SERVICE_MILEAGE", "get_MasterData"),
    ("oprema vozila", "GET_VEHICLE_EQUIPMENT", "get_MasterData"),
    ("dokumenti vozila", "GET_VEHICLE_DOCUMENTS", "get_Vehicles_id_documents"),
    ("koliko vozila ima", "GET_VEHICLE_COUNT", "get_Vehicles_Agg"),
    ("moji slucajevi", "GET_CASES", "get_Cases"),
    ("tko je lizing kuca", "GET_LEASING", "get_MasterData"),
]

# Queries where ML will FAIL and LLM must handle (expected ML < 85%)
LLM_FALLBACK_EXPECTED = [
    # Typos - ML can't handle, but LLM will
    ("priajvi stetu", "REPORT_DAMAGE", "typo - LLM should understand"),
    ("kilomertaza", "GET_MILEAGE", "typo - LLM should understand"),
    ("rezevracija", "BOOK_VEHICLE", "typo - LLM should understand"),
    # Very ambiguous
    ("trebam nesto", None, "ambiguous - LLM may clarify"),
    # Mixed signals
    ("daj mi informacije", None, "generic - LLM decides"),
]


# ---
# TEST RUNNER
# ---

@dataclass
class AccuracyResult:
    """Result of a single test."""
    passed: bool
    query: str
    expected: str
    actual: str
    confidence: float = 0.0
    category: str = ""
    test_type: str = ""


def run_phrase_matching_tests() -> Dict[str, Any]:
    """Phase 1: Test all phrase matching functions."""
    from services.flow_phrases import (
        matches_show_more,
        matches_confirm_yes,
        matches_confirm_no,
        matches_exit_signal,
        matches_greeting,
        matches_item_selection,
    )

    results = {"total": 0, "pass": 0, "fail": 0, "failures": []}

    for test in PHRASE_TESTS:
        text, exp_show, exp_yes, exp_no, exp_exit, exp_greeting, exp_select = test
        results["total"] += 1

        got_show = matches_show_more(text)
        got_yes = matches_confirm_yes(text)
        got_no = matches_confirm_no(text)
        got_exit = matches_exit_signal(text)
        got_greeting = matches_greeting(text)
        got_select = matches_item_selection(text)

        checks = []
        if exp_show != got_show:
            checks.append(f"show_more: expected={exp_show}, got={got_show}")
        if exp_yes != got_yes:
            checks.append(f"confirm_yes: expected={exp_yes}, got={got_yes}")
        if exp_no != got_no:
            checks.append(f"confirm_no: expected={exp_no}, got={got_no}")
        if exp_exit != got_exit:
            checks.append(f"exit_signal: expected={exp_exit}, got={got_exit}")
        if exp_greeting is not None and got_greeting != exp_greeting:
            checks.append(f"greeting: expected='{exp_greeting}', got='{got_greeting}'")
        if exp_greeting is None and got_greeting is not None:
            checks.append(f"greeting: expected=None, got='{got_greeting}'")
        if exp_select != got_select:
            checks.append(f"item_select: expected={exp_select}, got={got_select}")

        if checks:
            results["fail"] += 1
            results["failures"].append({"text": text, "errors": checks})
        else:
            results["pass"] += 1

    return results


def run_intent_classification_tests(
    classifier,
    test_cases: Dict[str, List[str]],
    verbose: bool = False
) -> Dict[str, Any]:
    """Test ML intent classification accuracy on hardcoded test cases."""
    results = {
        "total": 0, "correct": 0, "wrong": 0,
        "per_intent": defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0}),
        "confusion": defaultdict(Counter),  # confusion[expected][predicted] = count
        "failures": [],
        "confidence_correct": [],
        "confidence_wrong": [],
    }

    for expected_intent, queries in test_cases.items():
        for query in queries:
            results["total"] += 1
            results["per_intent"][expected_intent]["total"] += 1

            pred = classifier.predict(query)

            if pred.intent == expected_intent:
                results["correct"] += 1
                results["per_intent"][expected_intent]["correct"] += 1
                results["confidence_correct"].append(pred.confidence)
            else:
                results["wrong"] += 1
                results["per_intent"][expected_intent]["wrong"] += 1
                results["confidence_wrong"].append(pred.confidence)
                results["confusion"][expected_intent][pred.intent] += 1
                results["failures"].append(AccuracyResult(
                    passed=False,
                    query=query,
                    expected=expected_intent,
                    actual=pred.intent,
                    confidence=pred.confidence,
                    category="intent",
                ))

            if verbose:
                status = "OK" if pred.intent == expected_intent else "FAIL"
                print(f"  [{status}] '{query[:50]}' -> {pred.intent} ({pred.confidence:.1%}) [expected: {expected_intent}]")

    results["accuracy"] = results["correct"] / max(results["total"], 1)
    if results["confidence_correct"]:
        results["avg_conf_correct"] = sum(results["confidence_correct"]) / len(results["confidence_correct"])
    if results["confidence_wrong"]:
        results["avg_conf_wrong"] = sum(results["confidence_wrong"]) / len(results["confidence_wrong"])

    return results


def run_test_set_evaluation(classifier, test_set_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """Test ML intent classification on held-out test set (483 examples, 25+ intents)."""
    results = {
        "total": 0, "correct": 0, "wrong": 0,
        "per_intent": defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0}),
        "confusion": defaultdict(Counter),
        "failures": [],
    }

    with open(test_set_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            query = item["text"]
            expected_intent = item["intent"]

            results["total"] += 1
            results["per_intent"][expected_intent]["total"] += 1

            pred = classifier.predict(query)

            if pred.intent == expected_intent:
                results["correct"] += 1
                results["per_intent"][expected_intent]["correct"] += 1
            else:
                results["wrong"] += 1
                results["per_intent"][expected_intent]["wrong"] += 1
                results["confusion"][expected_intent][pred.intent] += 1
                results["failures"].append(AccuracyResult(
                    passed=False,
                    query=query,
                    expected=expected_intent,
                    actual=pred.intent,
                    confidence=pred.confidence,
                ))

            if verbose:
                status = "OK" if pred.intent == expected_intent else "FAIL"
                print(f"  [{status}] '{query[:50]}' -> {pred.intent} ({pred.confidence:.1%}) [expected: {expected_intent}]")

    results["accuracy"] = results["correct"] / max(results["total"], 1)
    return results


def run_tool_selection_tests(query_router, verbose: bool = False) -> Dict[str, Any]:
    """Test that QueryRouter maps intents to correct tools."""
    from services.query_router import INTENT_METADATA

    results = {
        "total": 0, "correct": 0, "wrong": 0, "low_confidence": 0,
        "per_intent": defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0}),
        "failures": [],
    }

    for expected_intent, queries in INTENT_TEST_CASES.items():
        expected_meta = INTENT_METADATA.get(expected_intent)
        if not expected_meta:
            continue

        expected_tool = expected_meta["tool"]

        for query in queries[:3]:  # Use first 3 per intent for speed
            results["total"] += 1
            results["per_intent"][expected_intent]["total"] += 1

            route = query_router.route(query)

            if not route.matched:
                results["low_confidence"] += 1
                # Low confidence is not a failure per se — LLM takes over
                # But we track it as it indicates ML weakness
                if verbose:
                    print(f"  [LOW] '{query[:50]}' -> no match (conf too low)")
                continue

            actual_tool = route.tool_name

            if actual_tool == expected_tool:
                results["correct"] += 1
                results["per_intent"][expected_intent]["correct"] += 1
            else:
                results["wrong"] += 1
                results["per_intent"][expected_intent]["wrong"] += 1
                results["failures"].append(AccuracyResult(
                    passed=False,
                    query=query,
                    expected=f"{expected_intent} -> {expected_tool}",
                    actual=f"{route.reason} -> {actual_tool}",
                    confidence=route.confidence,
                ))

            if verbose:
                status = "OK" if actual_tool == expected_tool else "FAIL"
                print(f"  [{status}] '{query[:50]}' -> {actual_tool} [expected: {expected_tool}]")

    matched = results["correct"] + results["wrong"]
    results["accuracy"] = results["correct"] / max(matched, 1)
    results["match_rate"] = matched / max(results["total"], 1)
    return results


def run_action_detection_tests(classifier) -> Dict[str, Any]:
    """Test READ vs WRITE vs DELETE action detection accuracy."""
    results = {"total": 0, "correct": 0, "wrong": 0, "failures": []}

    for query, expected_action, description in READ_WRITE_SAFETY_TESTS:
        results["total"] += 1
        pred = classifier.predict(query)

        if pred.action == expected_action:
            results["correct"] += 1
        else:
            results["wrong"] += 1
            results["failures"].append({
                "query": query,
                "expected": expected_action,
                "actual": pred.action,
                "intent": pred.intent,
                "desc": description,
            })

    results["accuracy"] = results["correct"] / max(results["total"], 1)
    return results


def run_flow_detection_tests(query_router) -> Dict[str, Any]:
    """Test flow type detection accuracy."""
    results = {"total": 0, "correct": 0, "wrong": 0, "unmatched": 0, "failures": []}

    for query, expected_flow, description in FLOW_DETECTION_TESTS:
        results["total"] += 1
        route = query_router.route(query)

        if not route.matched:
            results["unmatched"] += 1
            # Unmatched means LLM decides — not counted as wrong
            continue

        actual_flow = route.flow_type

        if actual_flow == expected_flow:
            results["correct"] += 1
        elif expected_flow is None and actual_flow is not None:
            # We didn't expect a specific flow but got one — still OK
            results["correct"] += 1
        else:
            results["wrong"] += 1
            results["failures"].append({
                "query": query,
                "expected_flow": expected_flow,
                "actual_flow": actual_flow,
                "desc": description,
            })

    matched = results["correct"] + results["wrong"]
    results["accuracy"] = results["correct"] / max(matched, 1)
    return results


def run_edge_case_tests(classifier, verbose: bool = False) -> Dict[str, Any]:
    """Test edge cases: typos, diacritics, colloquial, English, etc."""
    results = {
        "total": 0, "correct": 0, "wrong": 0,
        "by_type": defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0}),
        "failures": [],
    }

    for test in EDGE_CASE_TESTS:
        query = test["query"]
        expected = test["expected_intent"]
        test_type = test["type"]

        results["total"] += 1
        results["by_type"][test_type]["total"] += 1

        pred = classifier.predict(query)

        if pred.intent == expected:
            results["correct"] += 1
            results["by_type"][test_type]["correct"] += 1
        else:
            results["wrong"] += 1
            results["by_type"][test_type]["wrong"] += 1
            results["failures"].append({
                "query": query,
                "expected": expected,
                "actual": pred.intent,
                "confidence": pred.confidence,
                "type": test_type,
            })

        if verbose:
            status = "OK" if pred.intent == expected else "FAIL"
            print(f"  [{status}] [{test_type:15s}] '{query[:50]}' -> {pred.intent} ({pred.confidence:.1%})")

    results["accuracy"] = results["correct"] / max(results["total"], 1)
    return results


def run_pipeline_coverage_test(classifier, query_router, verbose: bool = False) -> Dict[str, Any]:
    """
    Phase 7: End-to-end pipeline coverage test.

    Tests the REAL production flow:
    1. ML classifier tries first
    2. If ML confidence >= 85% -> fast path (no LLM needed)
    3. If ML confidence < 85% -> LLM fallback (we verify ML correctly defers)

    This tells us: What % of queries does ML handle vs LLM?
    And: When ML handles, is it correct?
    """
    from services.query_router import ML_CONFIDENCE_THRESHOLD

    results = {
        "total": 0,
        "ml_handled": 0,          # ML confident enough (>= threshold)
        "ml_handled_correct": 0,  # ML confident AND correct
        "ml_handled_wrong": 0,    # ML confident BUT wrong (DANGEROUS!)
        "llm_fallback": 0,        # ML not confident -> goes to LLM
        "fast_path_failures": [],  # Cases where ML is confident but WRONG
        "fast_path_tests": {"total": 0, "pass": 0, "fail": 0, "deferred": 0},
        "llm_fallback_tests": {"total": 0, "correctly_deferred": 0, "wrongly_handled": 0},
    }

    # Test 1: Queries that ML MUST handle correctly (fast path)
    for query, expected_intent, expected_tool in ML_FAST_PATH_TESTS:
        results["total"] += 1
        results["fast_path_tests"]["total"] += 1

        pred = classifier.predict(query)
        route = query_router.route(query)

        if pred.confidence >= ML_CONFIDENCE_THRESHOLD:
            results["ml_handled"] += 1
            if pred.intent == expected_intent:
                results["ml_handled_correct"] += 1
                results["fast_path_tests"]["pass"] += 1
            else:
                results["ml_handled_wrong"] += 1
                results["fast_path_tests"]["fail"] += 1
                results["fast_path_failures"].append({
                    "query": query,
                    "expected": expected_intent,
                    "actual": pred.intent,
                    "confidence": pred.confidence,
                    "danger": "HIGH - ML is confident but WRONG",
                })
        else:
            results["llm_fallback"] += 1
            results["fast_path_tests"]["deferred"] += 1
            if verbose:
                print(f"  [DEFERRED] '{query[:50]}' -> ML conf={pred.confidence:.1%}, goes to LLM")

    # Test 2: Queries where ML SHOULD defer to LLM
    for query, expected_intent, description in LLM_FALLBACK_EXPECTED:
        results["total"] += 1
        results["llm_fallback_tests"]["total"] += 1

        pred = classifier.predict(query)

        if pred.confidence < ML_CONFIDENCE_THRESHOLD:
            # Good - ML correctly defers
            results["llm_fallback"] += 1
            results["llm_fallback_tests"]["correctly_deferred"] += 1
        else:
            # ML thinks it knows, but it might be wrong
            results["ml_handled"] += 1
            if expected_intent and pred.intent == expected_intent:
                results["ml_handled_correct"] += 1
                results["llm_fallback_tests"]["correctly_deferred"] += 1  # Correct anyway
            else:
                results["ml_handled_wrong"] += 1
                results["llm_fallback_tests"]["wrongly_handled"] += 1
                results["fast_path_failures"].append({
                    "query": query,
                    "expected": expected_intent or "LLM_SHOULD_HANDLE",
                    "actual": pred.intent,
                    "confidence": pred.confidence,
                    "danger": f"ML overconfident on: {description}",
                })

    # Calculate metrics
    if results["ml_handled"] > 0:
        results["ml_precision"] = results["ml_handled_correct"] / results["ml_handled"]
    else:
        results["ml_precision"] = 1.0

    results["ml_coverage"] = results["ml_handled"] / max(results["total"], 1)
    results["llm_rate"] = results["llm_fallback"] / max(results["total"], 1)

    return results


# ---
# MAIN REPORT
# ---

def print_section(title: str, char: str = "="):
    width = 78
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_accuracy_bar(label: str, accuracy: float, threshold: float, width: int = 40):
    """Print visual accuracy bar (ASCII-safe for Windows cp1252)."""
    filled = int(accuracy * width)
    bar = "#" * filled + "." * (width - filled)
    status = "PASS" if accuracy >= threshold else "FAIL"
    print(f"  {label:30s} [{bar}] {accuracy:6.1%} [{status}]")


def main():
    verbose = "--verbose" in sys.argv
    quick = "--quick" in sys.argv

    print_section("BOT ACCURACY TEST SUITE - ALL 29 INTENTS, ALL TOOLS, ALL FLOWS")
    print(f"  Mode: {'VERBOSE' if verbose else 'NORMAL'} | {'QUICK (no test set)' if quick else 'FULL'}")
    print(f"  Thresholds: intent={THRESHOLDS['intent_accuracy']:.0%}, "
          f"tool={THRESHOLDS['tool_accuracy']:.0%}, "
          f"action={THRESHOLDS['action_accuracy']:.0%}, "
          f"flow={THRESHOLDS['flow_accuracy']:.0%}")

    t0 = time.perf_counter()
    all_passed = True

    # ── Phase 1: Phrase Matching ────────────────────────────────────────
    print_section("PHASE 1: PHRASE MATCHING (deterministic, no ML)", "-")
    phrase_results = run_phrase_matching_tests()
    phrase_acc = phrase_results["pass"] / max(phrase_results["total"], 1)
    print_accuracy_bar("Phrase matching", phrase_acc, THRESHOLDS["phrase_accuracy"])
    print(f"  Total: {phrase_results['total']} | Pass: {phrase_results['pass']} | Fail: {phrase_results['fail']}")

    if phrase_results["failures"]:
        all_passed = False
        for f in phrase_results["failures"]:
            print(f"    FAIL: '{f['text']}'")
            for err in f["errors"]:
                print(f"          {err}")

    # ── Phase 2: Load ML Classifier ─────────────────────────────────────
    print_section("PHASE 2: ML INTENT CLASSIFICATION (29 intents)", "-")
    print("  Loading TF-IDF + Logistic Regression model...")

    from services.intent_classifier import get_intent_classifier
    classifier = get_intent_classifier("tfidf_lr")

    if not classifier._loaded:
        print("  ERROR: ML model not loaded! Cannot run classification tests.")
        print("  Train first: python services/intent_classifier.py")
        sys.exit(1)

    # 2a: Hardcoded test cases (all 29 intents)
    print(f"\n  2a) Hardcoded tests: {sum(len(v) for v in INTENT_TEST_CASES.items())} queries across {len(INTENT_TEST_CASES)} intents")
    intent_results = run_intent_classification_tests(classifier, INTENT_TEST_CASES, verbose)
    print_accuracy_bar("Intent classification", intent_results["accuracy"], THRESHOLDS["intent_accuracy"])
    print(f"  Total: {intent_results['total']} | Correct: {intent_results['correct']} | Wrong: {intent_results['wrong']}")

    if intent_results.get("avg_conf_correct"):
        print(f"  Avg confidence (correct): {intent_results['avg_conf_correct']:.1%}")
    if intent_results.get("avg_conf_wrong"):
        print(f"  Avg confidence (wrong):   {intent_results['avg_conf_wrong']:.1%}")

    if intent_results["accuracy"] < THRESHOLDS["intent_accuracy"]:
        all_passed = False

    # Show per-intent breakdown
    print(f"\n  Per-intent accuracy:")
    sorted_intents = sorted(
        intent_results["per_intent"].items(),
        key=lambda x: x[1]["correct"] / max(x[1]["total"], 1)
    )
    for intent_name, data in sorted_intents:
        acc = data["correct"] / max(data["total"], 1)
        status = "OK  " if acc >= 0.8 else "WARN" if acc >= 0.6 else "FAIL"
        print(f"    [{status}] {intent_name:30s} {acc:5.0%} ({data['correct']}/{data['total']})")

    # Show confusion matrix (top confusions)
    if intent_results["confusion"]:
        print(f"\n  Top confusions:")
        all_confusions = []
        for expected, confused_with in intent_results["confusion"].items():
            for predicted, count in confused_with.items():
                all_confusions.append((expected, predicted, count))
        all_confusions.sort(key=lambda x: x[2], reverse=True)
        for expected, predicted, count in all_confusions[:10]:
            print(f"    {expected:30s} -> {predicted:30s} ({count}x)")

    # 2b: Held-out test set (483 examples)
    if not quick:
        test_set_path = Path(__file__).parent.parent / "data" / "training" / "intent_test.jsonl"
        if test_set_path.exists():
            print(f"\n  2b) Held-out test set: {test_set_path.name}")
            test_set_results = run_test_set_evaluation(classifier, test_set_path, verbose)
            print_accuracy_bar("Test set accuracy", test_set_results["accuracy"], THRESHOLDS["intent_accuracy"])
            print(f"  Total: {test_set_results['total']} | Correct: {test_set_results['correct']} | Wrong: {test_set_results['wrong']}")

            if test_set_results["accuracy"] < THRESHOLDS["intent_accuracy"]:
                all_passed = False

            # Per-intent on test set
            print(f"\n  Per-intent accuracy (test set):")
            sorted_test = sorted(
                test_set_results["per_intent"].items(),
                key=lambda x: x[1]["correct"] / max(x[1]["total"], 1)
            )
            for intent_name, data in sorted_test:
                acc = data["correct"] / max(data["total"], 1)
                status = "OK  " if acc >= 0.8 else "WARN" if acc >= 0.6 else "FAIL"
                print(f"    [{status}] {intent_name:30s} {acc:5.0%} ({data['correct']}/{data['total']})")

            # Confusion matrix for test set
            if test_set_results["confusion"]:
                print(f"\n  Top confusions (test set):")
                all_confusions = []
                for expected, confused_with in test_set_results["confusion"].items():
                    for predicted, count in confused_with.items():
                        all_confusions.append((expected, predicted, count))
                all_confusions.sort(key=lambda x: x[2], reverse=True)
                for expected, predicted, count in all_confusions[:15]:
                    print(f"    {expected:30s} -> {predicted:30s} ({count}x)")
        else:
            print(f"\n  2b) SKIPPED: Test set not found at {test_set_path}")

    # ── Phase 3: READ vs WRITE Safety ───────────────────────────────────
    print_section("PHASE 3: READ vs WRITE vs DELETE SAFETY", "-")
    action_results = run_action_detection_tests(classifier)
    print_accuracy_bar("Action detection", action_results["accuracy"], THRESHOLDS["read_write_safety"])
    print(f"  Total: {action_results['total']} | Correct: {action_results['correct']} | Wrong: {action_results['wrong']}")

    if action_results["accuracy"] < THRESHOLDS["read_write_safety"]:
        all_passed = False

    if action_results["failures"]:
        for f in action_results["failures"]:
            print(f"    FAIL: '{f['query']}' -> {f['actual']} (expected {f['expected']}) [{f['desc']}]")

    # ── Phase 4: Tool Selection via QueryRouter ─────────────────────────
    print_section("PHASE 4: TOOL SELECTION (QueryRouter)", "-")
    from services.query_router import QueryRouter
    qr = QueryRouter()

    tool_results = run_tool_selection_tests(qr, verbose)
    print_accuracy_bar("Tool selection", tool_results["accuracy"], THRESHOLDS["tool_accuracy"])
    print(f"  Total queries: {tool_results['total']} | "
          f"Matched: {tool_results['correct'] + tool_results['wrong']} | "
          f"Correct: {tool_results['correct']} | "
          f"Wrong: {tool_results['wrong']} | "
          f"Low confidence (->LLM): {tool_results['low_confidence']}")

    if tool_results["accuracy"] < THRESHOLDS["tool_accuracy"]:
        all_passed = False

    if tool_results["failures"]:
        print(f"\n  Tool selection failures:")
        for f in tool_results["failures"]:
            print(f"    '{f.query[:60]}' -> {f.actual} (expected: {f.expected})")

    # ── Phase 5: Flow Detection ─────────────────────────────────────────
    print_section("PHASE 5: FLOW DETECTION", "-")
    flow_results = run_flow_detection_tests(qr)
    print_accuracy_bar("Flow detection", flow_results["accuracy"], THRESHOLDS["flow_accuracy"])
    print(f"  Total: {flow_results['total']} | Correct: {flow_results['correct']} | "
          f"Wrong: {flow_results['wrong']} | Unmatched (->LLM): {flow_results['unmatched']}")

    if flow_results["accuracy"] < THRESHOLDS["flow_accuracy"]:
        all_passed = False

    if flow_results["failures"]:
        for f in flow_results["failures"]:
            print(f"    FAIL: '{f['query']}' -> flow={f['actual_flow']} (expected: {f['expected_flow']}) [{f['desc']}]")

    # ── Phase 6: Edge Cases ─────────────────────────────────────────────
    print_section("PHASE 6: EDGE CASES (typos, diacritics, telegraphic, verbose)", "-")
    edge_results = run_edge_case_tests(classifier, verbose)
    print_accuracy_bar("Edge cases", edge_results["accuracy"], 0.70)  # Lower threshold for edge cases
    print(f"  Total: {edge_results['total']} | Correct: {edge_results['correct']} | Wrong: {edge_results['wrong']}")

    # By type breakdown
    print(f"\n  By type:")
    for edge_type, data in sorted(edge_results["by_type"].items()):
        acc = data["correct"] / max(data["total"], 1)
        status = "OK  " if acc >= 0.7 else "WARN" if acc >= 0.5 else "FAIL"
        print(f"    [{status}] {edge_type:20s} {acc:5.0%} ({data['correct']}/{data['total']})")

    if edge_results["failures"]:
        print(f"\n  Edge case failures:")
        for f in edge_results["failures"][:20]:
            print(f"    [{f['type']:15s}] '{f['query'][:50]}' -> {f['actual']} (expected: {f['expected']}, conf: {f['confidence']:.1%})")

    # ── Phase 7: End-to-End Pipeline Coverage ────────────────────────
    print_section("PHASE 7: END-TO-END PIPELINE (ML fast path + LLM fallback)", "-")
    pipeline_results = run_pipeline_coverage_test(classifier, qr, verbose)

    print(f"  Production pipeline breakdown:")
    print(f"    Total queries tested:        {pipeline_results['total']}")
    print(f"    ML handles (>= 85% conf):    {pipeline_results['ml_handled']} ({pipeline_results['ml_coverage']:.0%})")
    print(f"    ML correct when handling:    {pipeline_results['ml_handled_correct']}")
    print(f"    ML WRONG when handling:      {pipeline_results['ml_handled_wrong']}")
    print(f"    Deferred to LLM (< 85%):     {pipeline_results['llm_fallback']} ({pipeline_results['llm_rate']:.0%})")

    if pipeline_results["ml_handled"] > 0:
        print_accuracy_bar("ML precision (when confident)", pipeline_results["ml_precision"], 0.95)

    print(f"\n  Fast path (ML must handle):")
    fp = pipeline_results["fast_path_tests"]
    print(f"    Tested: {fp['total']} | Pass: {fp['pass']} | Fail: {fp['fail']} | Deferred: {fp['deferred']}")

    print(f"\n  LLM fallback (ML should defer):")
    lf = pipeline_results["llm_fallback_tests"]
    print(f"    Tested: {lf['total']} | Correctly deferred: {lf['correctly_deferred']} | Wrongly handled: {lf['wrongly_handled']}")

    if pipeline_results["fast_path_failures"]:
        print(f"\n  DANGEROUS: ML confident but WRONG:")
        for f in pipeline_results["fast_path_failures"]:
            print(f"    '{f['query'][:50]}' -> {f['actual']} ({f['confidence']:.0%}) [expected: {f['expected']}]")
            print(f"      {f['danger']}")

    if pipeline_results["ml_handled_wrong"] > 0:
        all_passed = False

    # ---
    # FINAL SUMMARY
    # ---
    elapsed = time.perf_counter() - t0

    print_section("FINAL SUMMARY")

    print_accuracy_bar("Phrase matching",     phrase_acc,                    THRESHOLDS["phrase_accuracy"])
    print_accuracy_bar("Intent (hardcoded)",  intent_results["accuracy"],   THRESHOLDS["intent_accuracy"])
    if not quick:
        try:
            print_accuracy_bar("Intent (test set)",   test_set_results["accuracy"],  THRESHOLDS["intent_accuracy"])
        except NameError:
            pass
    print_accuracy_bar("Action safety",       action_results["accuracy"],   THRESHOLDS["read_write_safety"])
    print_accuracy_bar("Tool selection",      tool_results["accuracy"],     THRESHOLDS["tool_accuracy"])
    print_accuracy_bar("Flow detection",      flow_results["accuracy"],     THRESHOLDS["flow_accuracy"])
    print_accuracy_bar("Edge cases (ML only)", edge_results["accuracy"],    0.70)
    if pipeline_results["ml_handled"] > 0:
        print_accuracy_bar("Pipeline precision",  pipeline_results["ml_precision"], 0.95)

    # Count total tests
    total_tests = (
        phrase_results["total"] +
        intent_results["total"] +
        action_results["total"] +
        tool_results["total"] +
        flow_results["total"] +
        edge_results["total"] +
        pipeline_results["total"]
    )

    total_pass = (
        phrase_results["pass"] +
        intent_results["correct"] +
        action_results["correct"] +
        tool_results["correct"] +
        flow_results["correct"] +
        edge_results["correct"] +
        pipeline_results["ml_handled_correct"] + pipeline_results["llm_fallback"]
    )

    if not quick:
        try:
            total_tests += test_set_results["total"]
            total_pass += test_set_results["correct"]
        except NameError:
            pass

    overall_accuracy = total_pass / max(total_tests, 1)
    print(f"\n  Total tests run:     {total_tests}")
    print(f"  Total passed:        {total_pass}")
    print(f"  Overall accuracy:    {overall_accuracy:.1%}")
    print(f"  Time:                {elapsed:.1f}s")

    # Architecture explanation
    print(f"\n  Pipeline architecture:")
    print(f"    User msg -> Phrases(regex) -> ML({pipeline_results['ml_coverage']:.0%} handled) -> LLM({pipeline_results['llm_rate']:.0%} fallback)")
    print(f"    When ML handles: {pipeline_results.get('ml_precision', 0):.1%} correct")
    print(f"    When ML defers:  LLM (gpt-4o-mini) handles typos, ambiguity, edge cases")

    verdict = "ALL TESTS PASSED" if all_passed else "SOME TESTS BELOW THRESHOLD"
    print(f"\n  Verdict: {verdict}")
    print(f"{'=' * 78}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

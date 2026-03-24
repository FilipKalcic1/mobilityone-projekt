"""
Search Pipeline Configuration — constants, boost values, entity keywords.

All tunable parameters for the 6-step unified search pipeline are centralized
here. Changing boost values or entity keywords only requires editing this file.

Design:
  - Additive boost model (v4.0): FAISS score stays dominant, boosts only nudge.
  - Cap enforced: total boost ∈ [MIN_TOTAL_BOOST, MAX_TOTAL_BOOST].
  - Entity keywords are stem-based (Croatian diacritics removed).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Final


# ---------------------------------------------------------------------------
# Additive boost values (v4.0)
# ---------------------------------------------------------------------------
# FAISS score (~0.3-0.9) stays dominant; boosts only nudge ranking.
# Max total boost: +0.55, min: -0.35. Predictable and debuggable.

BOOST_ENTITY_MATCH: Final[float] = 0.30        # Tool entity == detected entity
BOOST_ENTITY_MISMATCH: Final[float] = -0.20    # Tool entity ≠ detected entity
BOOST_QUERY_TYPE_MATCH: Final[float] = 0.25    # Suffix matches QueryType preference
BOOST_QUERY_TYPE_EXCLUDED: Final[float] = -0.15  # Suffix excluded for this QueryType
BOOST_PRIMARY_ENTITY: Final[float] = 0.20      # Primary entity tool (get_Vehicles)
BOOST_SECONDARY_ENTITY: Final[float] = 0.10    # Secondary entity (Types, Groups)
BOOST_BASE_LIST: Final[float] = 0.08           # Any base list tool
BOOST_PRIMARY_ACTION: Final[float] = 0.15      # PRIMARY_ACTION_TOOLS keyword match
BOOST_CATEGORY: Final[float] = 0.05            # Category match
BOOST_DOC: Final[float] = 0.05                 # Query words in tool docs
BOOST_HELPER_PENALTY: Final[float] = -0.15     # Lookup/helper/stats for list query
BOOST_COMPLEX_SUFFIX_PENALTY: Final[float] = -0.10  # Complex suffix for entity query
BOOST_LOOKUP_PENALTY: Final[float] = -0.10     # Lookup tool for single entity query
BOOST_GENERIC_CRUD_PENALTY: Final[float] = -0.20  # Hardcoded misrouting prevention
BOOST_FAMILY_MATCH: Final[float] = 0.35        # Tool Family Index direct match
BOOST_POSSESSIVE_ID: Final[float] = 0.15       # "moj auto" → get_Vehicles_id
BOOST_POSSESSIVE_PROFILE: Final[float] = 0.15  # "moj broj" → get_PersonData
BOOST_POSSESSIVE_LIST_PENALTY: Final[float] = -0.10  # Possessive penalizes list tools
BM25_WEIGHT: Final[float] = 0.15               # Additive BM25 boost weight

# Additive cap: prevent extreme swings
MAX_TOTAL_BOOST: Final[float] = 0.55
MIN_TOTAL_BOOST: Final[float] = -0.35


# ---------------------------------------------------------------------------
# Entity detection keywords for boost matching
# ---------------------------------------------------------------------------
# Keys = lowercase entity names matching tool_id parts (e.g., "get_Vehicles" → "vehicles")
# Values = stem prefixes for Croatian diacritic-normalized matching

ENTITY_KEYWORDS: Final[Dict[str, List[str]]] = {
    "companies": ["kompanij", "tvrtk", "firm", "poduzec"],
    "vehicles": ["vozil", "auto", "automobil", "flot"],
    "persons": ["osob", "korisnik", "zaposlenik", "radnik", "voditelj"],
    "expenses": ["trosk", "troska", "izdatak", "racun", "cijena"],
    "trips": ["putovanj", "trip", "voznj", "putni"],
    "cases": ["slucaj", "steta", "kvar", "incident"],
    "equipment": ["oprem", "uredaj", "stroj"],
    "partners": ["partner", "dobavljac", "klijent"],
    "teams": ["tim", "grupa", "ekip"],
    "orgunits": ["organizacij", "jedinic", "odjel", "sektor"],
    "costcenters": ["troskovn", "centar troska", "cost center"],
    "vehiclecalendar": ["rezervacij", "booking", "kalendar"],
    "documents": ["dokument", "prilog", "datoteka", "pdf"],
    "metadata": ["metapodac", "struktur", "shema", "polja"],
    # Extended entity coverage
    "vehicletypes": ["tip vozil", "vrsta vozil", "kategorija vozil"],
    "vehiclecontracts": ["ugovor vozil", "leasing", "najam vozil"],
    "equipmenttypes": ["tip oprem", "vrsta oprem"],
    "equipmentcalendar": ["kalendar oprem", "raspored oprem"],
    "periodicactivities": ["periodicn", "servis", "redovn aktiv"],
    "schedulingmodels": ["model raspored", "raspored model"],
    "mileagereports": ["kilometraz", "izvjest km", "km izvjest"],
    "roles": ["ulog", "rola", "dozvol", "permisij"],
    "tags": ["oznaka", "tag", "label"],
    "pools": ["pool", "bazen vozil"],
    "tenants": ["tenant", "najmoprimc"],
    "casetypes": ["tip slucaj", "vrsta stete", "kategorija prijav"],
    "expensetypes": ["tip trosk", "vrsta trosk", "kategorija trosk"],
    "expensegroups": ["grupa trosk", "skupine trosk"],
}


# ---------------------------------------------------------------------------
# Tool classification constants
# ---------------------------------------------------------------------------

# Primary entities — main business objects (highest entity boost)
PRIMARY_ENTITIES: Final[FrozenSet[str]] = frozenset([
    'companies', 'vehicles', 'persons', 'expenses', 'cases',
    'teams', 'trips', 'partners', 'tenants', 'roles', 'tags',
    'pools', 'orgunits', 'costcenters', 'equipment', 'booking',
])

# Secondary entities — types, groups, calendars (smaller boost)
SECONDARY_ENTITIES: Final[FrozenSet[str]] = frozenset([
    'vehicletypes', 'persontypes', 'expensetypes', 'casetypes',
    'equipmenttypes', 'triptypes', 'documenttypes', 'expensegroups',
    'vehiclecontracts', 'vehiclecalendar', 'equipmentcalendar',
    'periodicactivities', 'mileagereports', 'schedulingmodels',
])

# Complex suffixes that indicate nested/specialized tools
COMPLEX_SUFFIXES: Final[FrozenSet[str]] = frozenset([
    '_id', '_documents', '_metadata', '_thumb', '_agg', '_groupby',
    '_projectto', '_tree', '_deletebycriteria', 'lookup', 'helper',
    'input', 'stats', '_on', '_from', '_to',
])

# Penalty patterns for list queries (lookup/helper tools)
PENALTY_PATTERNS: Final[List[str]] = [
    'lookup', 'helper', 'input', 'available', 'latest', 'monthly',
    'dashboard', 'stats', '_agg', '_groupby', '_projectto',
    'historicalentries', 'assigned', 'fileids', 'distinctbrands',
]

# Generic CRUD keywords — prevent specific misrouting patterns
GENERIC_CRUD_KEYWORDS: Final[Dict[str, List[str]]] = {
    "post_cases": ["steta", "kvar", "udario", "ogrebao"],
    "post_vehicles": ["rezerv", "booking", "trebam"],
    "post_vehicleshistoricalentries": ["rezerv", "booking", "trebam"],
    "delete_triptypes_deletebycriteria": ["booking", "rezerv"],
    "get_monthlymileages_agg": ["koliko", "stanje", "imam"],
    "get_monthlymileagesassigned": ["koliko", "moja"],
}


# ---------------------------------------------------------------------------
# Verb-based method detection
# ---------------------------------------------------------------------------
# More reliable than ML for common Croatian verbs. Applied before ML intent.

VERB_METHOD_MAP: Final[Dict[str, str]] = {
    "obrisi": "delete", "izbrisi": "delete", "makni": "delete", "izbaci": "delete",
    "azuriraj": "put", "promijeni": "put", "izmijeni": "put", "izmjeni": "put",
    "dodaj": "post", "kreiraj": "post", "napravi": "post", "unesi": "post",
    "upisi": "post",
}


# ---------------------------------------------------------------------------
# Short query detection indicators
# ---------------------------------------------------------------------------

ID_INDICATORS: Final[List[str]] = [
    "id", "po id", "detalj", "konkret", "jednog", "jednu", "jedno",
]

CRITERIA_INDICATORS: Final[List[str]] = [
    "sve ", "svi ", "sva ", "prema", "kriterij", "filtr",
    "vise ", "više ", "za 20", "starij",
]


# ---------------------------------------------------------------------------
# Method verbs and suffix descriptions (for LLM entity descriptions)
# ---------------------------------------------------------------------------

METHOD_VERBS: Final[Dict[str, str]] = {
    "GET": "Dohvati", "POST": "Kreiraj", "PUT": "Ažuriraj",
    "PATCH": "Djelom. ažuriraj", "DELETE": "Obriši",
}

SUFFIX_DESCRIPTIONS: Final[Dict[str, str]] = {
    "_id": " po ID-u", "_deletebycriteria": " prema kriterijima",
    "_groupby": " grupirano", "_agg": " agregirano",
    "_documents": " dokumente", "_metadata": " metapodatke",
    "_multipatch": " bulk ažuriranje", "_projectto": " projekcija",
    "_setasdefault": " postavi zadano", "_filter": " filtrirano",
    "_thumb": " thumbnail",
}

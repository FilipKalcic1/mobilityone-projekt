"""
Consolidated entity detection module.

Single source of truth for detecting which entity (vehicles, expenses, etc.)
a user query refers to. Used by both unified_search.py and faiss_vector_store.py.

Uses diacritic-normalized substring matching to handle Croatian declensions.

Two-tier approach:
1. CURATED_STEMS: Hand-tuned stems for primary entities (highest priority, ordering matters)
2. AUTO_STEMS: Auto-generated from tool_documentation.json synonyms_hr (covers all 75 entities)
"""
import json
import logging
import os
import re
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# Diacritic normalization map
_DIACRITIC_MAP = str.maketrans({
    'č': 'c', 'ć': 'c', 'ž': 'z', 'š': 's', 'đ': 'd',
    'Č': 'C', 'Ć': 'C', 'Ž': 'Z', 'Š': 'S', 'Đ': 'D',
})


def _normalize(text: str) -> str:
    """Normalize text: lowercase + remove diacritics."""
    return text.lower().translate(_DIACRITIC_MAP)


# Hand-curated entity stems — highest quality, ordering matters.
# MORE SPECIFIC entities must come BEFORE generic ones.
CURATED_STEMS: Dict[str, List[str]] = {
    # Compound/specific entities FIRST
    "vehicletypes": ["tipovi vozil", "tip vozil", "tipov vozil", "vrsta vozil", "kategorija vozil"],
    "vehiclecontracts": ["ugovor vozil", "leasing", "najam vozil", "lizing"],
    "vehiclecalendar": ["rezervacij", "booking", "kalendar vozil"],
    "equipmenttypes": ["tipovi oprem", "tip oprem", "tipov oprem", "vrsta oprem", "kategorija oprem"],
    "equipmentcalendar": ["kalendar oprem", "raspored oprem"],
    "casetypes": ["tipovi slucaj", "tip slucaj", "tipov slucaj", "vrsta stete", "kategorija prijav", "tipovi slučaj", "tip slučaj", "tipov slučaj", "vrsta štete"],
    "expensetypes": ["tipovi trosk", "tip trosk", "tipov trosk", "vrsta trosk", "kategorija trosk", "tipovi trošk", "tip trošk", "tipov trošk", "vrsta trošk"],
    "expensegroups": ["grupa trosk", "skupine trosk", "grupa trošk", "skupine trošk"],
    "periodicactivities": ["periodicn", "periodičn", "servis", "redovn aktiv"],
    "schedulingmodels": ["model raspored", "modeli raspored", "raspored model", "model raspoređ", "modeli raspoređ"],
    "mileagereports": ["kilometraz", "kilometraž", "izvjest km", "izvješt km", "km izvjest", "km izvješt", "prijedeni put", "prijeđeni put"],
    "costcenters": ["troskovn", "troškovn", "centar troska", "centar troška", "cost center", "mjesto troska", "mjesto troška"],
    "personperiodicactivities": ["aktivnosti osobe", "aktivnosti zaposlenika"],
    "personorgunits": ["org jedinice osobe", "odjeli zaposlenika"],
    "tenantpermissions": ["dozvole tenanta", "korisnicke dozvole", "korisničke dozvole"],
    "teammembers": ["clanovi tima", "članovi tima", "zaposlenici u timu"],
    "vehicleassignments": ["dodjele vozila", "tko vozi"],
    "documenttypes": ["tipovi dokument", "tip dokument", "tipov dokument", "vrsta dokument", "kategorija dokument"],

    # Primary entities
    "companies": ["kompanij", "tvrtk", "firm", "poduzec", "poduzeć"],
    "vehicles": ["vozil", "auto", "automobil", "flot", "auti", "kola"],
    "persons": ["osob", "korisnik", "zaposlenik", "radnik", "voditelj", "djelatnik"],
    "expenses": ["trosk", "trosak", "troška", "trošak", "trošk", "izdatak", "racun", "račun", "cijena", "rashod"],
    "trips": ["putovanj", "trip", "voznj", "vožnj", "putni"],
    "cases": ["slucaj", "slučaj", "steta", "šteta", "kvar", "incident", "prijav"],
    "equipment": ["oprem", "uredaj", "uređaj", "inventar", "alat"],
    "partners": ["partner", "dobavljac", "dobavljač", "klijent"],
    "teams": ["tim ", "grupa", "ekip"],
    "orgunits": ["organizacij", "jedinic", "odjel", "sektor"],
    "roles": ["ulog", "rola", "dozvol", "permisij"],
    "tags": ["oznaka", "tag ", "label"],
    "pools": ["pool", "bazen vozil"],
    "tenants": ["tenant", "najmoprimc"],
    "documents": ["dokument", "prilog", "datoteka", "pdf"],
    "metadata": ["metapodac", "struktur", "shema", "polja"],
}

# Minimum stem length to avoid false positives
_MIN_STEM_LEN = 4

# Words that are too generic to be useful as stems (would match everything)
_STOP_STEMS = {
    "dohv", "prik", "poka", "obri", "dodaj", "krei", "azur",
    "novi", "nova", "novo", "po i", "spec", "konk", "deta",
    "info", "poda", "stav", "svih", "svi ", "sva ", "list",
    "popi", "preg",
}


def _generate_stems_from_synonym(synonym: str) -> List[str]:
    """
    Generate substring stems from a synonym.

    Strategy: take the first N characters (4-6) of each word as a stem.
    This handles Croatian declensions by matching word prefixes.
    """
    stems = []
    syn_norm = _normalize(synonym)

    # Single-word stem: first 4-5 chars
    words = syn_norm.split()
    for word in words:
        if len(word) >= _MIN_STEM_LEN:
            stem = word[:min(len(word), 5)]
            if stem not in _STOP_STEMS:
                stems.append(stem)

    # Multi-word stem: use the full normalized synonym if 2+ words and short enough
    if len(words) >= 2 and len(syn_norm) <= 25:
        stems.append(syn_norm)

    return stems


def build_auto_stems(tool_docs: dict) -> Dict[str, List[str]]:
    """
    Auto-generate entity→stems mapping from tool_documentation.json.

    Parses tool_id to extract entity, then collects stems from synonyms_hr.
    Only adds entities NOT already in CURATED_STEMS.
    """
    entity_stems: Dict[str, set] = {}

    for tool_id, doc in tool_docs.items():
        parts = tool_id.split("_")
        if len(parts) < 2:
            continue

        entity = parts[1].lower()

        # Skip entities already fully covered by curated stems
        if entity in CURATED_STEMS:
            continue

        synonyms = doc.get("synonyms_hr", [])
        for syn in synonyms:
            if len(syn) < 3:
                continue
            # Skip action verbs that appear in synonyms
            syn_stripped = syn.strip()
            for stem in _generate_stems_from_synonym(syn_stripped):
                entity_stems.setdefault(entity, set()).add(stem)

    # Convert to sorted lists and filter out overly short stems
    result = {}
    for entity, stems in entity_stems.items():
        filtered = [s for s in stems if len(s) >= _MIN_STEM_LEN]
        if filtered:
            result[entity] = sorted(filtered)

    return result


# Combined stems: curated (priority) + auto-generated (fallback)
# Initialized lazily when tool_documentation.json is available
ENTITY_STEMS: Dict[str, List[str]] = dict(CURATED_STEMS)
_auto_stems_loaded = False


def load_auto_stems(tool_docs: Optional[dict] = None):
    """
    Load auto-generated stems from tool_documentation.json.
    Called during application startup when tool docs are available.
    """
    global ENTITY_STEMS, _auto_stems_loaded

    if _auto_stems_loaded:
        return

    if tool_docs is None:
        # Try to load from file
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        doc_path = os.path.join(config_dir, 'tool_documentation.json')
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                tool_docs = json.load(f)

    if tool_docs:
        auto = build_auto_stems(tool_docs)
        # Merge: curated stems take priority (they're in ENTITY_STEMS already)
        # Auto stems fill gaps for uncovered entities
        for entity, stems in auto.items():
            if entity not in ENTITY_STEMS:
                ENTITY_STEMS[entity] = stems

        _auto_stems_loaded = True
        logger.info(
            f"EntityDetector: {len(CURATED_STEMS)} curated + "
            f"{len(auto)} auto-generated entities = {len(ENTITY_STEMS)} total"
        )


def detect_entity(query: str) -> Optional[str]:
    """
    Detect which entity a query refers to.

    Uses diacritic-normalized substring matching.
    Returns entity key (e.g., 'vehicles', 'expenses') or None.

    More specific entities are checked first (e.g., 'vehicletypes' before 'vehicles').
    """
    query_norm = _normalize(query)

    for entity, stems in ENTITY_STEMS.items():
        for stem in stems:
            stem_norm = _normalize(stem)
            if stem_norm in query_norm:
                return entity

    return None


def get_entity_keywords(entity: str) -> list:
    """Get the keyword stems for a given entity."""
    return ENTITY_STEMS.get(entity, [])


def get_all_entities() -> list:
    """Get all entity keys."""
    return list(ENTITY_STEMS.keys())


# --- Possessive signal detection ---

# All Croatian declension forms of "moj" (my) — F1 audit fix
# Nom: moj/moja/moje/moji, Gen: mojeg/mojega/mojih, Dat/Lok: mojem/mojemu/mojoj/mojim/mojima
# Akuz: moju, Inst: mojim/mojom/mojima. Contracted forms (mom/mome/momu) handled separately.
_MOJ = r'(?:moj[eai]?|mojeg|mojega|mojem|mojemu|mojoj|mojih|mojim|mojima|mojom|moju)'

_POSSESSIVE_PATTERNS = [
    # Vehicle possessives — \b prevents "nemoj vozilo" false positive (D2)
    (r'\b' + _MOJ + r'\s+vozil', 'vehicles'),
    (r'\b' + _MOJ + r'\s+aut', 'vehicles'),
    (r'\bmom(?:e|u|em)?\s+vozil', 'vehicles'),
    (r'\bmom(?:e|u|em)?\s+aut', 'vehicles'),
    (r'\bmoj\s+aut[oi]', 'vehicles'),
    # Expense possessives
    (r'\b' + _MOJ + r'\s+trosk', 'expenses'),
    (r'\b' + _MOJ + r'\s+trosak', 'expenses'),
    # Trip possessives
    (r'\b' + _MOJ + r'\s+putovanj', 'trips'),
    (r'\b' + _MOJ + r'\s+voznj', 'trips'),
    # Case possessives
    (r'\b' + _MOJ + r'\s+slucaj', 'cases'),
    (r'\b' + _MOJ + r'\s+stet', 'cases'),
    # Calendar/reservation possessives
    (r'\b' + _MOJ + r'\s+rezervacij', 'vehiclecalendar'),
    # Equipment possessives
    (r'\b' + _MOJ + r'\s+oprem', 'equipment'),
    # Generic "my" without entity — still possessive
    (r'\b' + _MOJ + r'\s+podac', None),  # "moji podaci" — possessive but no specific entity
    (r'\bpodaci\s+o\s+meni', None),
]


def detect_possessive(query: str) -> tuple:
    """
    Detect possessive patterns in query.

    Returns:
        (is_possessive: bool, entity_hint: Optional[str])
        entity_hint is the detected entity or None if possessive but generic.
    """
    q = _normalize(query)
    for pattern, entity in _POSSESSIVE_PATTERNS:
        if re.search(pattern, q):
            return True, entity
    return False, None


# --- User profile query detection ---

_USER_PROFILE_STEMS = [
    "telefon", "mobitel", "gsm", "broj telefon",
    "email", "e-mail", "mail adres",
    "person id", "moj id", "moj profil",
    "tenant id", "tko sam", "koji sam",
    "moj broj", "moj email", "moja adres",
]


def detect_user_profile_query(query: str) -> bool:
    """
    Detect queries about user's own profile data.

    These queries should route to get_PersonData or direct_response,
    not to entity list endpoints like get_Vehicles.
    """
    q = _normalize(query)
    return any(stem in q for stem in _USER_PROFILE_STEMS)

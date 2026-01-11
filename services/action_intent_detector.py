"""
ACTION INTENT DETECTOR - Faza 0 u tool selection pipeline.
Version: 1.0

KRITIČNO: Ovaj modul MORA doći PRIJE bilo kakvog semantic searcha!

Problem koji rješava:
- "unesi kilometražu" i "koliko imam kilometara" su semantički SLIČNI
- Ali prvi je POST (CREATE), drugi je GET (READ)
- Bez intent detekcije, embedding search ih može zamijeniti

Rješenje:
- Detektiraj ACTION INTENT (GET/POST/PUT/DELETE) IZ TEKSTA
- Filtriraj toolove po HTTP metodi PRIJE semantic searcha
- Tako embedding search uspoređuje samo toolove iste akcije
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Set

logger = logging.getLogger(__name__)


class ActionIntent(str, Enum):
    """HTTP action intent detected from user query."""
    READ = "GET"        # Dohvati podatke
    CREATE = "POST"     # Kreiraj novi zapis
    UPDATE = "PUT"      # Ažuriraj postojeći
    DELETE = "DELETE"   # Obriši
    UNKNOWN = "UNKNOWN" # Nejasan intent


@dataclass
class IntentDetectionResult:
    """Result of intent detection."""
    intent: ActionIntent
    confidence: float  # 0.0 - 1.0
    matched_pattern: Optional[str] = None
    reason: str = ""


# ═══════════════════════════════════════════════════════════════
# DELETE PATTERNS (highest priority - most specific)
# ═══════════════════════════════════════════════════════════════

DELETE_PATTERNS = [
    # Croatian - explicit delete commands (imperatives)
    (r'\bobri[sš]i\b', 0.95, "obriši"),
    (r'\bizbri[sš]i\b', 0.95, "izbriši"),
    (r'\bukloni\b', 0.90, "ukloni"),
    (r'\bmakni\b', 0.85, "makni"),
    (r'\botka[zž]i\b', 0.90, "otkaži"),
    (r'\bponi[sš]ti\b', 0.85, "poništi"),
    # Croatian - INFINITIVES (must catch "kako mogu obrisati", "želim obrisati")
    (r'\bobrisati\b', 0.92, "obrisati"),
    (r'\bizbrisati\b', 0.92, "izbrisati"),
    (r'\bukloniti\b', 0.88, "ukloniti"),
    (r'\bmaknuti\b', 0.85, "maknuti"),
    (r'\botkazati\b', 0.88, "otkazati"),
    (r'\bbrisanje\b', 0.88, "brisanje"),
    # Croatian - phrases
    (r'[zž]elim\s+obrisati', 0.95, "želim obrisati"),
    (r'ho[cć]u\s+obrisati', 0.95, "hoću obrisati"),
    (r'trebam\s+obrisati', 0.90, "trebam obrisati"),
    (r'mo[zž]e[sš]?\s+li\s+obrisati', 0.90, "možeš li obrisati"),
    (r'mogu\s+obrisati', 0.92, "mogu obrisati"),
    (r'mogu\s+izbrisati', 0.92, "mogu izbrisati"),
    # English
    (r'\bdelete\b', 0.95, "delete"),
    (r'\bremove\b', 0.90, "remove"),
    (r'\berase\b', 0.85, "erase"),
    (r'\bcancel\b', 0.85, "cancel"),
]

# ═══════════════════════════════════════════════════════════════
# CREATE PATTERNS (POST intent)
# ═══════════════════════════════════════════════════════════════

CREATE_PATTERNS = [
    # Croatian - explicit create commands (imperatives)
    (r'\bdodaj\b', 0.90, "dodaj"),
    (r'\bkreiraj\b', 0.95, "kreiraj"),
    (r'\bnapravi\b', 0.85, "napravi"),
    (r'\bstvori\b', 0.90, "stvori"),
    # Croatian - INFINITIVES (must catch "kako mogu dodati", "želim kreirati")
    (r'\bdodati\b', 0.92, "dodati"),
    (r'\bkreirati\b', 0.92, "kreirati"),
    (r'\bnapraviti\b', 0.88, "napraviti"),
    (r'\bstvoriti\b', 0.88, "stvoriti"),
    (r'\bdodavanje\b', 0.88, "dodavanje"),
    (r'\bkreiranje\b', 0.88, "kreiranje"),
    (r'mogu\s+dodati', 0.92, "mogu dodati"),
    (r'mogu\s+kreirati', 0.92, "mogu kreirati"),
    # Croatian - input/enter commands
    (r'\bunesi\b', 0.90, "unesi"),
    (r'\bupi[sš]i\b', 0.90, "upiši"),
    (r'\bunos\b', 0.85, "unos"),
    (r'\bzapi[sš]i\b', 0.85, "zapiši"),
    (r'\bunijeti\b', 0.88, "unijeti"),
    (r'\bupisati\b', 0.88, "upisati"),
    # Croatian - report commands (implies CREATE)
    (r'\bprijavi\b', 0.90, "prijavi"),
    (r'\bprijava\b', 0.85, "prijava"),
    (r'\bprijaviti\b', 0.88, "prijaviti"),
    # Croatian - booking commands
    (r'\brezerviraj\b', 0.95, "rezerviraj"),
    (r'\brezerviram\b', 0.90, "rezerviram"),
    (r'\brezervirati\b', 0.92, "rezervirati"),
    (r'[zž]elim\s+rezervirati', 0.95, "želim rezervirati"),
    (r'ho[cć]u\s+rezervirati', 0.95, "hoću rezervirati"),
    (r'trebam\s+rezervirati', 0.90, "trebam rezervirati"),
    (r'trebam\s+vozilo', 0.85, "trebam vozilo"),
    (r'trebam\s+auto', 0.85, "trebam auto"),
    # Croatian - damage reporting (implicit CREATE)
    (r'\budario\b', 0.85, "udario"),
    (r'\budarila\b', 0.85, "udarila"),
    (r'\bogrebao\b', 0.85, "ogrebao"),
    (r'\bogrebala\b', 0.85, "ogrebala"),
    (r'\bo[sš]tetio\b', 0.85, "oštetio"),
    (r'\bo[sš]tetila\b', 0.85, "oštetila"),
    (r'imam\s+[sš]tet', 0.85, "imam štetu"),
    (r'imam\s+kvar', 0.85, "imam kvar"),
    (r'dogodila.*nesre[cć]', 0.85, "nesreća"),
    # Croatian - phrases
    (r'[zž]elim\s+prijaviti', 0.90, "želim prijaviti"),
    (r'ho[cć]u\s+prijaviti', 0.90, "hoću prijaviti"),
    (r'trebam\s+prijaviti', 0.90, "trebam prijaviti"),
    (r'moram\s+unijeti', 0.90, "moram unijeti"),
    (r'moram\s+upisati', 0.90, "moram upisati"),
    # English
    (r'\badd\b', 0.90, "add"),
    (r'\bcreate\b', 0.95, "create"),
    (r'\bmake\b', 0.80, "make"),
    (r'\bnew\b', 0.80, "new"),
    (r'\bbook\b', 0.85, "book"),
    (r'\breport\b', 0.85, "report"),
    (r'\bsubmit\b', 0.85, "submit"),
    (r'\benter\b', 0.80, "enter"),
]

# ═══════════════════════════════════════════════════════════════
# UPDATE PATTERNS (PUT/PATCH intent)
# ═══════════════════════════════════════════════════════════════

UPDATE_PATTERNS = [
    # Croatian - explicit update commands (imperatives)
    (r'\ba[zž]uriraj\b', 0.95, "ažuriraj"),
    (r'\bpromijeni\b', 0.90, "promijeni"),
    (r'\bizmijeni\b', 0.90, "izmijeni"),
    (r'\buredi\b', 0.85, "uredi"),
    (r'\bispravi\b', 0.85, "ispravi"),
    # Croatian - INFINITIVES (must catch "kako mogu ažurirati", "želim promijeniti")
    (r'\ba[zž]urirati\b', 0.92, "ažurirati"),
    (r'\bpromijeniti\b', 0.92, "promijeniti"),
    (r'\bizmijeniti\b', 0.88, "izmijeniti"),
    (r'\burediti\b', 0.85, "urediti"),
    (r'\bispraviti\b', 0.85, "ispraviti"),
    (r'\ba[zž]uriranje\b', 0.88, "ažuriranje"),
    (r'\bizmjena\b', 0.85, "izmjena"),
    (r'mogu\s+a[zž]urirati', 0.92, "mogu ažurirati"),
    (r'mogu\s+promijeniti', 0.92, "mogu promijeniti"),
    # Croatian - phrases
    (r'[zž]elim\s+promijeniti', 0.90, "želim promijeniti"),
    (r'[zž]elim\s+a[zž]urirati', 0.95, "želim ažurirati"),
    (r'trebam\s+promijeniti', 0.90, "trebam promijeniti"),
    (r'trebam\s+a[zž]urirati', 0.95, "trebam ažurirati"),
    (r'trebam\s+ispraviti', 0.90, "trebam ispraviti"),
    # English
    (r'\bupdate\b', 0.95, "update"),
    (r'\bchange\b', 0.85, "change"),
    (r'\bmodify\b', 0.90, "modify"),
    (r'\bedit\b', 0.85, "edit"),
    (r'\bfix\b', 0.80, "fix"),
    (r'\bcorrect\b', 0.85, "correct"),
]

# ═══════════════════════════════════════════════════════════════
# READ PATTERNS (GET intent)
# ═══════════════════════════════════════════════════════════════

READ_PATTERNS = [
    # Croatian - questions
    (r'^koj[aei]?\s', 0.90, "koji/koja/koje"),
    (r'^koliko\b', 0.90, "koliko"),
    (r'koliko\s+(mi|imam)', 0.95, "koliko mi/imam"),
    (r'[sš]to\s+je\b', 0.85, "što je"),
    (r'kakv[aoi]\s+je\b', 0.85, "kakva je"),
    (r'gdje\s+je\b', 0.85, "gdje je"),
    (r'kada\s+(je|isti[cč]e)', 0.90, "kada je/ističe"),
    (r'ima\s+li\b', 0.85, "ima li"),
    # Croatian - show/display commands
    (r'\bprika[zž]i\b', 0.90, "prikaži"),
    (r'\bpoka[zž]i\b', 0.90, "pokaži"),
    (r'\bdaj\s+mi\b', 0.85, "daj mi"),
    (r'\breci\s+mi\b', 0.85, "reci mi"),
    (r'\bdohvati\b', 0.90, "dohvati"),
    (r'\bu[cč]itaj\b', 0.85, "učitaj"),
    # Croatian - search commands
    (r'\bprona[dđ]i\b', 0.85, "pronađi"),
    (r'\bna[dđ]i\b', 0.85, "nađi"),
    (r'\btra[zž]i\b', 0.85, "traži"),
    (r'\bpretra[zž]i\b', 0.85, "pretraži"),
    # Croatian - check commands
    (r'\bprovjeri\b', 0.85, "provjeri"),
    (r'\bpogledaj\b', 0.85, "pogledaj"),
    # Croatian - possessive (my data = READ)
    (r'\bmoje?\s+rezervacij', 0.90, "moje rezervacije"),
    (r'\bmoje?\s+vozil', 0.90, "moje vozilo"),
    (r'\bmoji\s+podaci', 0.90, "moji podaci"),
    # Croatian - availability
    (r'\bslobodn[aio]\b', 0.85, "slobodna"),
    (r'\bdostupn[aio]\b', 0.85, "dostupna"),
    # Croatian - status/info
    (r'\bstanje\b', 0.80, "stanje"),
    (r'\bstatus\b', 0.80, "status"),
    (r'\binformacij', 0.80, "informacije"),
    (r'\bpodaci\b', 0.80, "podaci"),
    # English
    (r'\bshow\b', 0.85, "show"),
    (r'\bdisplay\b', 0.85, "display"),
    (r'\bview\b', 0.85, "view"),
    (r'\bget\b', 0.80, "get"),
    (r'\bfetch\b', 0.85, "fetch"),
    (r'\blist\b', 0.85, "list"),
    (r'\bfind\b', 0.80, "find"),
    (r'\bsearch\b', 0.80, "search"),
    (r'\bcheck\b', 0.80, "check"),
    (r'what\s+is\b', 0.85, "what is"),
    (r'how\s+much\b', 0.85, "how much"),
    (r'how\s+many\b', 0.85, "how many"),
    # Question mark at end
    (r'\?$', 0.70, "question mark"),
]


# Pre-compile all patterns for performance
_DELETE_COMPILED = [(re.compile(p, re.IGNORECASE), c, m) for p, c, m in DELETE_PATTERNS]
_CREATE_COMPILED = [(re.compile(p, re.IGNORECASE), c, m) for p, c, m in CREATE_PATTERNS]
_UPDATE_COMPILED = [(re.compile(p, re.IGNORECASE), c, m) for p, c, m in UPDATE_PATTERNS]
_READ_COMPILED = [(re.compile(p, re.IGNORECASE), c, m) for p, c, m in READ_PATTERNS]


def detect_action_intent(query: str) -> IntentDetectionResult:
    """
    Detect the ACTION INTENT from user query.

    KRITIČNO: Ova funkcija MORA se pozvati PRIJE semantic searcha!

    Priority order:
    1. DELETE - most specific, highest risk
    2. UPDATE - specific modification
    3. CREATE - new data entry
    4. READ - default if nothing else matches

    Args:
        query: User query text

    Returns:
        IntentDetectionResult with detected intent and confidence
    """
    if not query or not query.strip():
        return IntentDetectionResult(
            intent=ActionIntent.UNKNOWN,
            confidence=0.0,
            reason="Empty query"
        )

    query_lower = query.lower().strip()

    # Check DELETE first (highest priority - destructive action)
    delete_match = _find_best_match(query_lower, _DELETE_COMPILED)
    if delete_match and delete_match[0] >= 0.85:
        return IntentDetectionResult(
            intent=ActionIntent.DELETE,
            confidence=delete_match[0],
            matched_pattern=delete_match[1],
            reason=f"DELETE pattern matched: {delete_match[1]}"
        )

    # Check UPDATE second
    update_match = _find_best_match(query_lower, _UPDATE_COMPILED)
    if update_match and update_match[0] >= 0.85:
        return IntentDetectionResult(
            intent=ActionIntent.UPDATE,
            confidence=update_match[0],
            matched_pattern=update_match[1],
            reason=f"UPDATE pattern matched: {update_match[1]}"
        )

    # Check CREATE third
    create_match = _find_best_match(query_lower, _CREATE_COMPILED)
    if create_match and create_match[0] >= 0.80:
        return IntentDetectionResult(
            intent=ActionIntent.CREATE,
            confidence=create_match[0],
            matched_pattern=create_match[1],
            reason=f"CREATE pattern matched: {create_match[1]}"
        )

    # Check READ fourth
    read_match = _find_best_match(query_lower, _READ_COMPILED)
    if read_match and read_match[0] >= 0.70:
        return IntentDetectionResult(
            intent=ActionIntent.READ,
            confidence=read_match[0],
            matched_pattern=read_match[1],
            reason=f"READ pattern matched: {read_match[1]}"
        )

    # Default to UNKNOWN
    return IntentDetectionResult(
        intent=ActionIntent.UNKNOWN,
        confidence=0.0,
        reason="No pattern matched"
    )


def _find_best_match(query: str, compiled_patterns) -> Optional[tuple]:
    """
    Find the best matching pattern with highest confidence.

    Returns:
        Tuple of (confidence, pattern_name) or None
    """
    best_confidence = 0.0
    best_pattern = None

    for pattern, confidence, name in compiled_patterns:
        if pattern.search(query):
            if confidence > best_confidence:
                best_confidence = confidence
                best_pattern = name

    if best_pattern:
        return (best_confidence, best_pattern)
    return None


def get_allowed_methods(intent: ActionIntent) -> Set[str]:
    """
    Get allowed HTTP methods for the detected intent.

    Args:
        intent: Detected action intent

    Returns:
        Set of allowed HTTP methods
    """
    if intent == ActionIntent.READ:
        return {"GET"}
    elif intent == ActionIntent.CREATE:
        return {"POST"}
    elif intent == ActionIntent.UPDATE:
        return {"PUT", "PATCH"}
    elif intent == ActionIntent.DELETE:
        return {"DELETE"}
    else:
        # UNKNOWN - allow all
        return {"GET", "POST", "PUT", "PATCH", "DELETE"}


def filter_tools_by_intent(
    tool_ids: Set[str],
    tool_methods: dict,
    intent: ActionIntent
) -> Set[str]:
    """
    Filter tools to only those matching the detected intent.

    Args:
        tool_ids: Set of tool IDs to filter
        tool_methods: Dict mapping tool_id -> HTTP method
        intent: Detected action intent

    Returns:
        Filtered set of tool IDs
    """
    if intent == ActionIntent.UNKNOWN:
        return tool_ids  # No filtering

    allowed_methods = get_allowed_methods(intent)
    filtered = set()

    for tool_id in tool_ids:
        method = tool_methods.get(tool_id, "GET")

        # Special case: POST methods that are actually searches
        if intent == ActionIntent.READ and method == "POST":
            tool_lower = tool_id.lower()
            if any(x in tool_lower for x in ["search", "query", "filter", "find", "list"]):
                filtered.add(tool_id)
                continue

        if method in allowed_methods:
            filtered.add(tool_id)

    # Log filtering result
    if filtered:
        logger.info(
            f"ACTION INTENT GATE: {intent.value} -> "
            f"{len(tool_ids)} -> {len(filtered)} tools"
        )
    else:
        # Fallback: if nothing matches, return all
        logger.warning(
            f"ACTION INTENT GATE: No tools match {intent.value}, "
            f"returning all {len(tool_ids)} tools"
        )
        return tool_ids

    return filtered

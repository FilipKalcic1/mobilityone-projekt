"""
Flow Phrases - Centralized phrase matching for conversation flow.

Replaces hardcoded string matching across engine/__init__.py,
unified_router.py, and other modules.

Uses word-boundary matching to prevent substring false positives
(e.g., "NEKAKO" matching "ne").
"""

import re
import logging
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


# ---
# PHRASE CATEGORIES
# ---

# "Show more" signals - user wants to see additional options within current flow
SHOW_MORE_PHRASES: List[str] = [
    "pokaži", "pokazi", "pokaz",
    "ostala", "ostale", "ostali",
    "druga", "druge", "drugi", "drugo",
    "više", "vise", "jos", "još",
    "sva vozila", "sve vozila",
    "lista", "popis", "prikazi", "prikaži",
]

# Confirmation "yes" phrases
CONFIRM_YES_PHRASES: List[str] = [
    "da", "potvrdi", "potvrditi", "potvrdeno", "potvrđeno",
    "ok", "okay", "yes", "oke",
    "moze", "može", "mozemo", "možemo",
    "super", "naravno", "svakako", "apsolutno",
    "slazem se", "slažem se", "vazi", "važi",
    "idem", "ajde", "ajmo", "idemo",
    "tocno", "točno", "ispravno", "u redu",
]

# Confirmation "no" phrases
CONFIRM_NO_PHRASES: List[str] = [
    "ne", "nema", "nemoj", "nemam",
    "odustani", "odustajem", "odustati",
    "cancel", "no", "nope",
    "ne zelim", "ne želim",
    "nikako", "ni slucajno", "ni slučajno",
    "nista", "ništa",
    "krivo", "pogresno", "pogrešno",
    "prekini", "stop", "stani",
]

# Exit signals - user explicitly wants to leave current flow
EXIT_SIGNALS: List[str] = [
    "ne želim", "ne zelim",
    "necu", "neću", "nećem",
    "odustani", "odustajem",
    "zapravo", "ipak",
    "ne treba", "nemoj",
    "stani", "stop",
    "nešto drugo", "nesto drugo",
    "drugo pitanje",
    "promijeni", "cancel",
    "hoću nešto drugo", "hocu nesto drugo",
    "želim nešto drugo", "zelim nesto drugo",
]

# Item selection phrases (ordinals)
ORDINAL_PHRASES: List[str] = [
    "prvi", "prva", "prvo",
    "drugi", "druga", "drugo",
    "treci", "treći", "treca", "treća",
    "cetvrti", "četvrti",
    "peti", "peta",
]

# Greeting phrases
GREETING_PHRASES = {
    "bok": "Bok! Kako vam mogu pomoći?",
    "hej": "Hej! Kako vam mogu pomoći?",
    "pozdrav": "Pozdrav! Kako vam mogu pomoći?",
    "zdravo": "Zdravo! Kako vam mogu pomoći?",
    "dobar dan": "Dobar dan! Kako vam mogu pomoći?",
    "dobro jutro": "Dobro jutro! Kako vam mogu pomoći?",
    "dobra večer": "Dobra večer! Kako vam mogu pomoći?",
    "dobra vecer": "Dobra večer! Kako vam mogu pomoći?",
    "hvala": "Nema na čemu! Trebate li još nešto?",
    "thanks": "You're welcome! Need anything else?",
    "help": "Mogu vam pomoći s:\n• Rezervacija vozila\n• Unos kilometraže\n• Prijava kvara\n• Informacije o vozilu",
    "pomoc": "Mogu vam pomoći s:\n• Rezervacija vozila\n• Unos kilometraže\n• Prijava kvara\n• Informacije o vozilu",
    "pomoć": "Mogu vam pomoći s:\n• Rezervacija vozila\n• Unos kilometraže\n• Prijava kvara\n• Informacije o vozilu",
}


# ---
# MATCHING FUNCTIONS (Word-boundary safe)
# ---

# Cache compiled regex patterns
_pattern_cache: dict = {}


def _get_word_pattern(phrase: str) -> re.Pattern:
    """Get compiled regex pattern for word-boundary matching."""
    if phrase not in _pattern_cache:
        # Escape regex special chars, add word boundaries
        escaped = re.escape(phrase)
        # Use \b for word boundary - prevents "nekako" matching "ne"
        _pattern_cache[phrase] = re.compile(
            r'(?:^|\b)' + escaped + r'(?:\b|$)',
            re.IGNORECASE
        )
    return _pattern_cache[phrase]


def matches_any(text: str, phrases: List[str]) -> bool:
    """
    Check if text matches any phrase using word-boundary matching.

    This prevents false positives like:
    - "nekako" matching "ne"
    - "pokaži" matching "pokaz" (this IS valid because pokaz is prefix)
    - "danas" matching "da"

    For single-word phrases (like "da", "ne"), uses strict word boundary.
    For multi-word phrases, uses substring match (they're specific enough).
    """
    text_lower = text.lower().strip()

    for phrase in phrases:
        if ' ' in phrase:
            # Multi-word phrase: substring match is safe (specific enough)
            if phrase in text_lower:
                return True
        else:
            # Single-word: use word boundary to prevent substring matches
            pattern = _get_word_pattern(phrase)
            if pattern.search(text_lower):
                return True

    return False


def matches_show_more(text: str) -> bool:
    """Check if text is a 'show more' request.

    Excludes exit signals that share words with show_more.
    E.g., "nešto drugo" contains "drugo" (show_more) but is an exit signal.
    """
    # Exit signals take priority - "nešto drugo" is NOT "show more"
    if matches_any(text, EXIT_SIGNALS):
        return False
    return matches_any(text, SHOW_MORE_PHRASES)


def matches_confirm_yes(text: str) -> bool:
    """Check if text is a confirmation 'yes'."""
    return matches_any(text, CONFIRM_YES_PHRASES)


def matches_confirm_no(text: str) -> bool:
    """Check if text is a confirmation 'no'."""
    # CRITICAL: Check "show more" first - it takes priority
    if matches_show_more(text):
        return False
    return matches_any(text, CONFIRM_NO_PHRASES)


def matches_exit_signal(text: str) -> bool:
    """Check if text is an exit/cancellation signal."""
    return matches_any(text, EXIT_SIGNALS)


def matches_item_selection(text: str) -> bool:
    """Check if text is a numeric or ordinal item selection."""
    text_stripped = text.strip()
    if text_stripped.isdigit():
        return True
    return matches_any(text, ORDINAL_PHRASES)


def matches_greeting(text: str) -> Optional[str]:
    """Check if text is a greeting and return response."""
    text_lower = text.lower().strip()

    for greeting, response in GREETING_PHRASES.items():
        if text_lower == greeting or text_lower.startswith(greeting + " "):
            return response

    return None

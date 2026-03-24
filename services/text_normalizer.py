"""
Text Normalization — Croatian diacritics, synonyms, and query preprocessing.

Extracted from intent_classifier.py. Single source of truth for all text
normalization in the routing pipeline.

Performance: O(n) for diacritics (character map), O(n) for synonyms (word split).

Usage:
    from services.text_normalizer import (
        normalize_query, normalize_diacritics, normalize_synonyms,
        DIACRITIC_MAP, SYNONYM_MAP,
    )

    normalized = normalize_query("Pokaži mi sva auta čija je registracija...")
    # → "pokazi mi sva vozila cija je registracija..."
"""

from __future__ import annotations

from typing import Dict, Final


# ---------------------------------------------------------------------------
# Croatian diacritic mapping (O(1) lookup per character)
# ---------------------------------------------------------------------------

DIACRITIC_MAP: Final[Dict[str, str]] = {
    'č': 'c', 'ć': 'c', 'đ': 'd', 'š': 's', 'ž': 'z',
    'Č': 'C', 'Ć': 'C', 'Đ': 'D', 'Š': 'S', 'Ž': 'Z',
}

# str.translate() table — 5x faster than character-by-character loop
_DIACRITIC_TABLE: Final = str.maketrans(DIACRITIC_MAP)


# ---------------------------------------------------------------------------
# Synonym map (canonical forms for common Croatian variations)
# ---------------------------------------------------------------------------

SYNONYM_MAP: Final[Dict[str, str]] = {
    # Vehicle synonyms → vozilo/vozila
    'auto': 'vozilo',
    'auta': 'vozila',
    'automobil': 'vozilo',
    'automobili': 'vozila',
    'automobila': 'vozila',
    'kola': 'vozilo',
    'kolima': 'vozilima',

    # Phone synonyms → telefon
    'mobitel': 'telefon',
    'mobitela': 'telefona',
    'gsm': 'telefon',
    'tel': 'telefon',

    # Mileage synonyms → kilometara
    'kilometraza': 'kilometara',
    'km': 'kilometara',

    # Common typos → corrected form
    'telfon': 'telefon',
    'telef': 'telefon',
    'rezevacija': 'rezervacija',
    'rezevirati': 'rezervirati',
    'osteio': 'ostetio',
    'ostetiti': 'ostetio',
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_diacritics(text: str) -> str:
    """Remove Croatian diacritics from text.

    Uses str.translate() for optimal performance (~5x faster than loop).

    >>> normalize_diacritics("čćđšž ČĆĐŠŽ")
    'ccdszz CCDSZ'
    """
    return text.translate(_DIACRITIC_TABLE)


def normalize_synonyms(text: str) -> str:
    """Replace common synonyms with canonical forms.

    >>> normalize_synonyms("prikaži mi sva auta")
    'prikaži mi sva vozila'
    """
    words = text.split()
    result = []
    for word in words:
        canonical = SYNONYM_MAP.get(word.lower())
        result.append(canonical if canonical is not None else word)
    return ' '.join(result)


def normalize_query(text: str) -> str:
    """Full normalization pipeline for ML classification input.

    Steps (in order):
    1. Lowercase
    2. Strip whitespace
    3. Remove diacritics (ž→z, č→c, etc.)
    4. Replace synonyms (auto→vozilo, etc.)

    >>> normalize_query("  Pokaži mi AUTA s Čistoćom  ")
    'pokazi mi vozila s cistocom'
    """
    text = text.lower().strip()
    text = normalize_diacritics(text)
    text = normalize_synonyms(text)
    return text

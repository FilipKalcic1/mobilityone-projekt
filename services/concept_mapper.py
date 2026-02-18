"""
Concept Mapper - Jargon to Standard Term Translation
Version: 1.0

Expands user queries by translating:
1. Croatian jargon -> standard Croatian terms
2. Colloquial expressions -> formal terms
3. Abbreviations -> full words

This improves FAISS search by ensuring the query embedding
captures related concepts even if the user uses informal language.

Examples:
- "daj mi auto" -> "daj mi auto vozilo vehicle prikaži dohvati get"
- "unesi km" -> "unesi km kilometraža mileage dodaj kreiraj post"
"""

import json
import logging
import os
import re
import threading
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)


def _load_concept_map_from_config() -> Dict[str, List[str]]:
    """Load concept map from config/croatian_mappings.json."""
    base_path = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base_path, "config", "croatian_mappings.json")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        raw = data.get("concept_map", {})
        return {k: v for k, v in raw.items() if k != "_comments"}
    except Exception as e:
        logger.error(f"Could not load concept_map from config: {e}")
        return {}


class ConceptMapper:
    """
    Maps informal/jargon terms to standard vocabulary for better semantic search.

    Philosophy:
    - EXPAND, don't replace - keep original terms plus add synonyms
    - Language-aware - handles Croatian characters (č, ć, ž, š, đ)
    - Domain-specific - fleet management vocabulary
    """

    # Loaded from config/croatian_mappings.json
    CONCEPT_MAP: Dict[str, List[str]] = {}

    # Phrase patterns that should trigger concept expansion
    # These are regex patterns that capture common query structures
    PHRASE_PATTERNS = [
        # "daj mi [X]" patterns
        (r'\bdaj\s+mi\b', ["prikaži", "dohvati", "get"]),
        # "trebam [X]" patterns
        (r'\btreba(m)?\s+mi?\b', ["dohvati", "prikaži", "get", "potrebno"]),
        # "pokaži mi [X]" patterns
        (r'\bpoka[zž]i\s+mi\b', ["prikaži", "dohvati", "get", "list"]),
        # "upiši/unesi [X]" patterns
        (r'\b(upi[sš]i|unesi)\b', ["dodaj", "kreiraj", "post", "add"]),
        # "koliko [X]" patterns
        (r'\bkoliko\b', ["broj", "count", "get", "količina"]),
        # "koji/koja/koje [X]" patterns
        (r'\bkoj[aei]\b', ["lista", "popis", "get", "which"]),
        # "ima li" patterns
        (r'\bima\s+li\b', ["dostupnost", "provjeri", "check", "available"]),
    ]

    def __init__(self, enabled: bool = True):
        """
        Initialize concept mapper.

        Args:
            enabled: If False, expand_query returns original query unchanged
        """
        self.enabled = enabled

        # Load concept map from config file
        loaded = _load_concept_map_from_config()
        if loaded:
            self.CONCEPT_MAP = loaded
        # else: keeps class-level empty dict as fallback

        self._build_normalized_map()
        logger.info(f"ConceptMapper initialized with {len(self.CONCEPT_MAP)} concept mappings")

    def _build_normalized_map(self) -> None:
        """Build a normalized version of the concept map for faster lookups."""
        self._normalized_map: Dict[str, List[str]] = {}

        for key, values in self.CONCEPT_MAP.items():
            # Normalize key (remove diacritics for matching)
            normalized_key = self._normalize(key)

            # Store both original and normalized keys
            if key not in self._normalized_map:
                self._normalized_map[key] = values
            if normalized_key not in self._normalized_map:
                self._normalized_map[normalized_key] = values

    def _normalize(self, text: str) -> str:
        """
        Normalize text by removing Croatian diacritics.

        This allows matching both "šef" and "sef", "čekaj" and "cekaj", etc.
        """
        replacements = {
            'č': 'c', 'ć': 'c',
            'ž': 'z',
            'š': 's',
            'đ': 'd',
            'Č': 'C', 'Ć': 'C',
            'Ž': 'Z',
            'Š': 'S',
            'Đ': 'D',
        }
        result = text
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result

    def expand_query(self, query: str) -> str:
        """
        Expand query with related concepts.

        IMPORTANT: This ADDS terms, it doesn't replace.
        The original query is always preserved.

        Args:
            query: Original user query

        Returns:
            Expanded query with original + related terms
        """
        if not self.enabled:
            return query

        if not query or not query.strip():
            return query

        query_lower = query.lower()
        query_normalized = self._normalize(query_lower)

        # Collect all expansions
        expansions: Set[str] = set()

        # 1. Check phrase patterns first (multi-word patterns)
        for pattern, terms in self.PHRASE_PATTERNS:
            if re.search(pattern, query_lower) or re.search(pattern, query_normalized):
                expansions.update(terms)

        # 2. Check individual terms
        words = query_lower.split()
        normalized_words = query_normalized.split()

        for word in words + normalized_words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)

            if clean_word in self._normalized_map:
                expansions.update(self._normalized_map[clean_word])

        # 3. Check two-word phrases
        for i in range(len(words) - 1):
            two_word = f"{words[i]} {words[i+1]}"
            normalized_two_word = f"{normalized_words[i]} {normalized_words[i+1]}"

            if two_word in self._normalized_map:
                expansions.update(self._normalized_map[two_word])
            elif normalized_two_word in self._normalized_map:
                expansions.update(self._normalized_map[normalized_two_word])

        # 4. Build expanded query
        if not expansions:
            logger.debug(f"No concept expansions for: '{query}'")
            return query

        # Remove any expansions that are already in the query
        new_terms = [term for term in expansions
                     if term.lower() not in query_lower
                     and self._normalize(term.lower()) not in query_normalized]

        if not new_terms:
            logger.debug(f"All expansions already in query: '{query}'")
            return query

        # PHASE 3 OPTIMIZATION: Limit expansion to max 5 terms
        # to prevent query explosion and maintain embedding quality
        MAX_EXPANSION_TERMS = 5
        if len(new_terms) > MAX_EXPANSION_TERMS:
            logger.debug(f"Limiting expansions from {len(new_terms)} to {MAX_EXPANSION_TERMS}")
            new_terms = new_terms[:MAX_EXPANSION_TERMS]

        expanded = f"{query} {' '.join(new_terms)}"

        logger.debug(f"Expanded query: '{query}' -> '{expanded}'")
        return expanded

    def get_expansions_only(self, query: str) -> List[str]:
        """
        Get just the expansion terms without the original query.

        Useful for debugging or displaying what terms were added.
        """
        original = query
        expanded = self.expand_query(query)

        if expanded == original:
            return []

        # Extract added terms
        added_part = expanded[len(original):].strip()
        return added_part.split() if added_part else []


# Singleton instance (thread-safe)
_concept_mapper: Optional[ConceptMapper] = None
_concept_mapper_lock = threading.Lock()


def get_concept_mapper(enabled: bool = True) -> ConceptMapper:
    """Get or create singleton ConceptMapper instance (thread-safe)."""
    global _concept_mapper
    if _concept_mapper is not None:
        return _concept_mapper
    with _concept_mapper_lock:
        if _concept_mapper is None:
            _concept_mapper = ConceptMapper(enabled=enabled)
    return _concept_mapper


def expand_query(query: str) -> str:
    """
    Convenience function to expand a query.

    Usage:
        from services.concept_mapper import expand_query
        expanded = expand_query("daj mi auto")
        # Returns: "daj mi auto vozilo vehicle prikaži dohvati get"
    """
    return get_concept_mapper().expand_query(query)

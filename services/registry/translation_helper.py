"""
Translation Helper - Runtime fallback for unmapped terms.
Version: 1.0

Provides fallback translation mechanisms when hardcoded dictionaries
don't have a mapping for a term.

STRATEGIES:
    1. Dictionary lookup (PATH_ENTITY_MAP, OUTPUT_KEY_MAP)
    2. Partial match (substring matching)
    3. Levenshtein distance (fuzzy matching for typos)
    4. English fallback (readable formatting)

USAGE:
    helper = TranslationHelper()

    # Translate a single term
    croatian = helper.translate("vehicle")  # "vozilo"

    # Translate with fallback
    croatian = helper.translate("unknownterm")  # "unknown term" (formatted)

    # Get suggestions for unmapped terms
    suggestions = helper.suggest_translations("vehiclstatus")  # ["status vozila"]
"""

import re
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class TranslationResult:
    """Result of a translation attempt."""
    original: str
    translated: str
    method: str  # "dictionary", "partial", "fuzzy", "fallback"
    confidence: float  # 0.0 to 1.0


class TranslationHelper:
    """
    Provides runtime translation with fallback mechanisms.

    Uses the dictionaries from EmbeddingEngine but adds:
    - Fuzzy matching for typos
    - Partial matching for compound words
    - Formatted English fallback
    """

    def __init__(self):
        """Initialize with dictionaries from EmbeddingEngine."""
        # Import here to avoid circular imports
        from services.registry.embedding_engine import EmbeddingEngine

        engine = EmbeddingEngine()
        self.path_entity_map = engine.PATH_ENTITY_MAP
        self.output_key_map = engine.OUTPUT_KEY_MAP
        self.synonyms = engine.CROATIAN_SYNONYMS

        # Build reverse lookup (Croatian -> English)
        self.croatian_to_english: Dict[str, str] = {}
        for eng, (cro_nom, cro_gen) in self.path_entity_map.items():
            self.croatian_to_english[cro_nom.lower()] = eng
            self.croatian_to_english[cro_gen.lower()] = eng

    def translate(self, term: str, prefer_genitive: bool = False) -> TranslationResult:
        """
        Translate a term to Croatian with fallback.

        Args:
            term: English term to translate
            prefer_genitive: If True, return genitive form

        Returns:
            TranslationResult with translated term and metadata
        """
        term_lower = term.lower().strip()

        # 1. Direct dictionary lookup
        if term_lower in self.path_entity_map:
            nom, gen = self.path_entity_map[term_lower]
            return TranslationResult(
                original=term,
                translated=gen if prefer_genitive else nom,
                method="dictionary",
                confidence=1.0
            )

        # 2. Output key map lookup
        if term_lower in self.output_key_map:
            return TranslationResult(
                original=term,
                translated=self.output_key_map[term_lower],
                method="dictionary",
                confidence=1.0
            )

        # 3. Partial match (for compound words like "vehiclestatus")
        partial = self._find_partial_match(term_lower)
        if partial:
            return TranslationResult(
                original=term,
                translated=partial[0],
                method="partial",
                confidence=partial[1]
            )

        # 4. Fuzzy match (for typos like "vehicel" -> "vehicle")
        fuzzy = self._find_fuzzy_match(term_lower)
        if fuzzy:
            return TranslationResult(
                original=term,
                translated=fuzzy[0],
                method="fuzzy",
                confidence=fuzzy[1]
            )

        # 5. Fallback: format English term as readable
        readable = self._make_readable(term)
        return TranslationResult(
            original=term,
            translated=readable,
            method="fallback",
            confidence=0.3
        )

    def _find_partial_match(self, term: str) -> Optional[Tuple[str, float]]:
        """Find partial matches in dictionary."""
        best_match = None
        best_coverage = 0.0

        for key, (nom, _) in self.path_entity_map.items():
            if key in term and len(key) >= 4:  # Minimum key length
                coverage = len(key) / len(term)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_match = nom

        if best_match and best_coverage >= 0.5:
            return (best_match, min(0.9, best_coverage))
        return None

    def _find_fuzzy_match(self, term: str) -> Optional[Tuple[str, float]]:
        """Find fuzzy matches using Levenshtein distance."""
        best_match = None
        best_distance = float('inf')
        max_distance = max(2, len(term) // 4)  # Allow up to 25% errors

        for key, (nom, _) in self.path_entity_map.items():
            dist = self._levenshtein(term, key)
            if dist < best_distance and dist <= max_distance:
                best_distance = dist
                best_match = nom

        if best_match:
            # Convert distance to confidence (0 distance = 1.0, max = 0.5)
            confidence = 1.0 - (best_distance / (max_distance * 2))
            return (best_match, max(0.5, confidence))
        return None

    def _levenshtein(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _make_readable(self, term: str) -> str:
        """Convert technical term to human-readable format."""
        # Insert space before uppercase letters (camelCase)
        readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', term)
        # Insert space between letters and numbers
        readable = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', readable)
        readable = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', readable)
        return readable.lower()

    def suggest_translations(self, term: str, max_suggestions: int = 5) -> List[TranslationResult]:
        """
        Suggest possible translations for a term.

        Useful for discovering mappings during development.

        Args:
            term: Term to find suggestions for
            max_suggestions: Maximum number of suggestions

        Returns:
            List of TranslationResult sorted by confidence
        """
        suggestions = []
        term_lower = term.lower()

        # Find all partial matches
        for key, (nom, gen) in self.path_entity_map.items():
            if key in term_lower or term_lower in key:
                overlap = min(len(key), len(term_lower)) / max(len(key), len(term_lower))
                suggestions.append(TranslationResult(
                    original=term,
                    translated=f"{nom} ({gen})",
                    method="partial",
                    confidence=overlap
                ))

        # Find fuzzy matches
        for key, (nom, gen) in self.path_entity_map.items():
            dist = self._levenshtein(term_lower, key)
            if dist <= 3 and dist > 0:  # Close but not exact
                confidence = 1.0 - (dist / max(len(term_lower), len(key)))
                suggestions.append(TranslationResult(
                    original=term,
                    translated=f"{nom} ({gen})",
                    method="fuzzy",
                    confidence=confidence
                ))

        # Sort by confidence and deduplicate
        seen = set()
        unique = []
        for s in sorted(suggestions, key=lambda x: x.confidence, reverse=True):
            if s.translated not in seen:
                seen.add(s.translated)
                unique.append(s)

        return unique[:max_suggestions]

    def get_coverage_stats(self) -> Dict[str, int]:
        """Get statistics about dictionary coverage."""
        return {
            "path_entity_map_entries": len(self.path_entity_map),
            "output_key_map_entries": len(self.output_key_map),
            "synonym_groups": len(self.synonyms),
            "total_synonyms": sum(len(v) for v in self.synonyms.values()),
            "croatian_to_english_entries": len(self.croatian_to_english),
        }

    def translate_croatian_to_english(self, croatian_term: str) -> Optional[str]:
        """
        Reverse translation: Croatian to English.

        Useful for understanding what entity a Croatian query refers to.
        """
        return self.croatian_to_english.get(croatian_term.lower())

    def expand_with_synonyms(self, term: str) -> List[str]:
        """
        Expand a term with all its synonyms.

        Useful for query expansion in search.
        """
        terms = [term]
        term_lower = term.lower()

        # Find matching synonym group
        for root, syn_list in self.synonyms.items():
            if root in term_lower:
                terms.extend(syn_list)
            elif any(syn.lower() in term_lower for syn in syn_list):
                # Term contains a synonym, add the root form
                for key, (nom, _) in self.path_entity_map.items():
                    if root in nom.lower():
                        terms.append(nom)
                        break
                terms.extend(syn_list)

        return list(set(terms))


def translate_term(term: str, prefer_genitive: bool = False) -> str:
    """
    Convenience function for quick translation.

    Args:
        term: English term to translate
        prefer_genitive: If True, return genitive form

    Returns:
        Croatian translation or formatted English
    """
    helper = TranslationHelper()
    result = helper.translate(term, prefer_genitive)
    return result.translated


def suggest_mapping(term: str) -> List[str]:
    """
    Convenience function to suggest mappings for unmapped terms.

    Args:
        term: Unmapped term

    Returns:
        List of suggested Croatian translations
    """
    helper = TranslationHelper()
    suggestions = helper.suggest_translations(term)
    return [s.translated for s in suggestions]

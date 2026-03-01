"""
Ambiguity Detector - Detects ambiguous tool selections and provides disambiguation hints.

When multiple tools have similar scores and the same suffix pattern (e.g., all _Agg tools),
the query is considered "ambiguous" and needs disambiguation.

This module:
1. Detects ambiguity in search results
2. Extracts entity hints from user context and query
3. Generates disambiguation hints for LLM
4. Suggests clarification questions for user when LLM can't decide
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from services.context import UserContextManager

logger = logging.getLogger(__name__)


# Suffix patterns that indicate generic operations
GENERIC_SUFFIX_PATTERNS = [
    "_Agg",           # Aggregations
    "_GroupBy",       # Grouping
    "_ProjectTo",     # Projections
    "_metadata",      # Metadata
    "_documents",     # Documents
    "_DeleteByCriteria",  # Bulk delete
    "_multipatch",    # Bulk update
]

# Entity extraction patterns (Croatian)
ENTITY_KEYWORDS = {
    "vozil": "Vehicles",
    "vozila": "Vehicles",
    "auto": "Vehicles",
    "flot": "Vehicles",
    "kompanij": "Companies",
    "tvrtk": "Companies",
    "firm": "Companies",
    "osob": "Persons",
    "korisnik": "Persons",
    "zaposlenik": "Persons",
    "trošk": "Expenses",
    "troska": "Expenses",
    "izdatak": "Expenses",
    "račun": "Expenses",
    "putovanj": "Trips",
    "trip": "Trips",
    "vožnj": "Trips",
    "slučaj": "Cases",
    "šteta": "Cases",
    "kvar": "Cases",
    "incident": "Cases",
    "oprem": "Equipment",
    "rezervacij": "VehicleCalendar",
    "booking": "VehicleCalendar",
    "kilometr": "MileageReports",
    "dokument": "Documents",
    "partner": "Partners",
    "tim": "Teams",
    "org": "OrgUnits",
    "centar troška": "CostCenters",
    "troškovn": "CostCenters",
}

# Clarification questions for ambiguous suffixes
CLARIFICATION_QUESTIONS = {
    "_Agg": "Za koju vrstu podataka želite izračunati statistiku? (npr. vozila, troškovi, putovanja)",
    "_GroupBy": "Koje podatke želite grupirati? (npr. troškove po mjesecu, vozila po tipu)",
    "_ProjectTo": "Iz koje tablice želite dohvatiti podatke? (npr. vozila, osobe, partneri)",
    "_metadata": "Za koji entitet želite vidjeti metapodatke?",
    "_documents": "Čije dokumente želite vidjeti? (npr. vozila, kompanije, osobe)",
    "_DeleteByCriteria": "Koje podatke želite obrisati?",
    "_multipatch": "Koje podatke želite masovno ažurirati?",
}


@dataclass
class AmbiguityResult:
    """Result of ambiguity detection."""
    is_ambiguous: bool = False
    ambiguous_suffix: Optional[str] = None
    similar_tools: List[str] = field(default_factory=list)
    score_variance: float = 0.0
    detected_entity: Optional[str] = None
    disambiguation_hint: str = ""
    clarification_question: Optional[str] = None


class AmbiguityDetector:
    """
    Detects and handles ambiguous tool selections.

    Used by UnifiedRouter to:
    1. Detect when search results are ambiguous
    2. Provide hints to LLM for better disambiguation
    3. Generate clarification questions for user
    """

    def __init__(self, tool_documentation: Optional[Dict] = None):
        """Initialize with optional tool documentation for entity extraction."""
        self._tool_documentation = tool_documentation or {}

    def detect_ambiguity(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.15,  # Tools within 15% of top score
        min_similar_count: int = 3      # Need at least 3 similar tools
    ) -> AmbiguityResult:
        """
        Detect if search results are ambiguous.

        Args:
            query: User query
            search_results: List of search results with tool_id and score
            user_context: User context for entity detection
            score_threshold: Consider tools within this % of top score as "similar"
            min_similar_count: Minimum number of similar tools to be ambiguous

        Returns:
            AmbiguityResult with detection details and hints
        """
        if not search_results or len(search_results) < min_similar_count:
            return AmbiguityResult()

        # Get top score
        top_score = search_results[0].get("score", 0)
        if top_score == 0:
            return AmbiguityResult()

        # Find tools within threshold of top score
        similar_tools = []
        for r in search_results[:20]:  # Check top 20
            score = r.get("score", 0)
            if score >= top_score * (1 - score_threshold):
                similar_tools.append(r.get("tool_id", ""))

        # Check if similar tools share a suffix pattern
        ambiguous_suffix = self._find_common_suffix(similar_tools)

        if not ambiguous_suffix or len(similar_tools) < min_similar_count:
            return AmbiguityResult()

        # Calculate score variance
        scores = [r.get("score", 0) for r in search_results[:len(similar_tools)]]
        score_variance = max(scores) - min(scores) if scores else 0

        # Detect entity from query and context
        detected_entity = self._detect_entity(query, user_context)

        # Build disambiguation hint
        hint = self._build_disambiguation_hint(
            query, ambiguous_suffix, similar_tools, detected_entity
        )

        # Get clarification question
        clarification = CLARIFICATION_QUESTIONS.get(ambiguous_suffix)

        logger.info(
            f"AMBIGUITY DETECTED: suffix={ambiguous_suffix}, "
            f"similar_tools={len(similar_tools)}, detected_entity={detected_entity}"
        )

        return AmbiguityResult(
            is_ambiguous=True,
            ambiguous_suffix=ambiguous_suffix,
            similar_tools=similar_tools[:5],  # Top 5
            score_variance=score_variance,
            detected_entity=detected_entity,
            disambiguation_hint=hint,
            clarification_question=clarification
        )

    def _find_common_suffix(self, tools: List[str]) -> Optional[str]:
        """Find common generic suffix among tools."""
        suffix_counts = {}

        for tool in tools:
            for suffix in GENERIC_SUFFIX_PATTERNS:
                if suffix in tool:
                    suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

        if not suffix_counts:
            return None

        # Return most common suffix if it's in majority of tools
        most_common = max(suffix_counts.items(), key=lambda x: x[1])
        if most_common[1] >= len(tools) * 0.5:  # At least 50% of tools
            return most_common[0]

        return None

    def _detect_entity(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Detect entity from query and user context.

        Priority:
        1. Explicit entity mention in query
        2. User's current vehicle (for vehicle-related queries)
        3. None if can't determine
        """
        query_lower = query.lower()

        # Check query for entity keywords
        for keyword, entity in ENTITY_KEYWORDS.items():
            if keyword in query_lower:
                return entity

        # Check user context for vehicle
        # Use UserContextManager for validated access
        if user_context:
            ctx = UserContextManager(user_context)
            if ctx.vehicle_id:
                # User has a vehicle - might be vehicle-related
                # Only infer if query doesn't mention other entities
                return None  # Don't assume, let LLM decide

        return None

    def _build_disambiguation_hint(
        self,
        query: str,
        suffix: str,
        similar_tools: List[str],
        detected_entity: Optional[str]
    ) -> str:
        """Build disambiguation hint for LLM."""
        hints = []

        # Suffix-specific hints
        suffix_hints = {
            "_Agg": "AGGREGACIJA - korisnik traži statistiku (prosječno, suma, max, min)",
            "_GroupBy": "GRUPIRANJE - korisnik traži podatke grupirane po nekom kriteriju",
            "_ProjectTo": "FILTRIRANJE - korisnik traži specifične kolone/polja",
            "_metadata": "METAPODACI - korisnik traži dodatne informacije o entitetu",
            "_documents": "DOKUMENTI - korisnik traži dokumente vezane uz entitet",
            "_DeleteByCriteria": "BRISANJE - korisnik želi obrisati podatke po kriteriju",
            "_multipatch": "MASOVNO AŽURIRANJE - korisnik želi ažurirati više zapisa",
        }

        hints.append(suffix_hints.get(suffix, ""))

        # Entity hint
        if detected_entity:
            hints.append(f"DETEKTIRANI ENTITET: {detected_entity}")

        # Similar tools hint
        entities_from_tools = set()
        for tool in similar_tools[:5]:
            # Extract entity from tool name (e.g., get_Vehicles_Agg -> Vehicles)
            parts = tool.split("_")
            if len(parts) >= 2:
                entities_from_tools.add(parts[1])

        if entities_from_tools:
            hints.append(f"MOGUĆI ENTITETI: {', '.join(sorted(entities_from_tools))}")

        return "\n".join(h for h in hints if h)

    def get_best_tool_for_entity(
        self,
        entity: str,
        suffix: str,
        similar_tools: List[str]
    ) -> Optional[str]:
        """
        Get the best tool for a detected entity and suffix.

        Used when entity is clear but multiple tools exist.
        """
        entity_lower = entity.lower()

        for tool in similar_tools:
            tool_lower = tool.lower()
            # Check if tool contains the entity name
            if entity_lower in tool_lower:
                return tool

        return None

    def needs_clarification(
        self,
        ambiguity_result: AmbiguityResult,
        llm_confidence: float
    ) -> bool:
        """
        Determine if we need to ask user for clarification.

        Args:
            ambiguity_result: Result from detect_ambiguity
            llm_confidence: LLM's confidence in its tool selection

        Returns:
            True if user clarification is needed
        """
        if not ambiguity_result.is_ambiguous:
            return False

        # If we detected an entity, no clarification needed
        if ambiguity_result.detected_entity:
            return False

        # If LLM is confident, no clarification needed
        if llm_confidence >= 0.7:
            return False

        # If many similar tools with low score variance, need clarification
        if (len(ambiguity_result.similar_tools) >= 5 and
            ambiguity_result.score_variance < 0.1):
            return True

        return False


# Singleton instance
_detector: Optional[AmbiguityDetector] = None


def get_ambiguity_detector(tool_documentation: Optional[Dict] = None) -> AmbiguityDetector:
    """Get singleton AmbiguityDetector instance."""
    global _detector
    if _detector is None:
        _detector = AmbiguityDetector(tool_documentation)
    return _detector

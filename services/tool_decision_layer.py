"""
Tool Decision Layer - Deterministic decision logic for tool selection.

This layer sits between FAISS search and LLM selection.
It implements hard gates and tie-breaker rules to improve Top-1 accuracy.

Key principles:
1. ACTION is a HARD GATE, not a suggestion
2. Confidence-based auto-accept (skip LLM if score gap is big)
3. Deterministic tie-breaker rules for similar tools
"""

import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Hard action types - mutually exclusive."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    UNKNOWN = "UNKNOWN"


@dataclass
class DecisionResult:
    """Result from the decision layer."""
    tool_id: str
    confidence: float
    decision_method: str  # "auto_accept", "tie_breaker", "llm_needed"
    reasoning: str


class ToolDecisionLayer:
    """
    Deterministic decision layer for tool selection.

    Improves Top-1 accuracy by:
    1. Hard action gating (not just filtering)
    2. Auto-accepting high-confidence results
    3. Applying deterministic tie-breaker rules
    """

    # ==========================================================================
    # ACTION DETECTION - HARD RULES
    # ==========================================================================

    # DELETE patterns (most specific) - with and without Croatian diacritics
    DELETE_PATTERNS = [
        r'\b(obriši|obrisat|obrisi|izbriši|izbrisat|izbrisi|ukloni|uklonit|makni|maknut)\b',
        r'\b(brisanje|uklanjanje)\b',
        r'\b(delete|remove)\b',
    ]

    # POST patterns (create/add AND bulk operations)
    POST_PATTERNS = [
        r'\b(dodaj|dodati|kreiraj|kreirat|napravi|napravit|unesi|unijet|stvori|stvorit)\b',
        r'\b(novi|nova|novo|novu|novih)\b.*\b(dodaj|kreiraj|unesi|napravi)\b',
        r'\b(dodaj|kreiraj|unesi|napravi)\b.*\b(novi|nova|novo)\b',
        r'\b(dodavanje|kreiranje|unos)\b',
        r'\b(create|add|insert|new)\b',
        # BULK/MULTIPATCH operations are POST!
        r'\b(bulk|grupno|više\s+stavki|više\s+zapisa|odjednom|s\s+id-evima)\b',
        r'\bviše\b.*(ažuriraj|azuriraj|promijeni)',
        r'(ažuriraj|azuriraj|promijeni).*\bviše\b',
    ]

    # PUT patterns (full update)
    PUT_PATTERNS = [
        r'\b(potpuno|kompletno|cijeli|sve)\b.*\b(zamijeni|ažuriraj|azuriraj)\b',
        r'\b(zamijeni|zamijenit)\b.*\b(sve|potpuno|kompletno)\b',
        r'\b(full|complete)\b.*\b(update|replace)\b',
    ]

    # PATCH patterns (partial update)
    PATCH_PATTERNS = [
        r'\b(djelomično|djelomicno|parcijalno|samo)\b.*\b(ažuriraj|azuriraj|promijeni)\b',
        r'\b(ažuriraj|azuriraj|promijeni)\b.*\b(samo|djelomično|djelomicno)\b',
        r'\b(partial|partially)\b.*\b(update|modify)\b',
    ]

    # Generic UPDATE patterns (could be PUT or PATCH)
    UPDATE_PATTERNS = [
        r'\b(ažuriraj|azuriraj|promijeni|izmijeni|modificiraj)\b',
        r'\b(ažuriranje|azuriranje|promjena|izmjena)\b',
        r'\b(update|modify|change|edit)\b',
    ]

    # GET patterns (read/retrieve) - with and without Croatian diacritics
    GET_PATTERNS = [
        r'\b(dohvati|dohvatit|prikaži|prikazi|prikazat|pogledaj|pogledat|pokazi|vrati|vratit)\b',
        r'\b(daj|dajte)\s+mi\b',
        r'\b(koji|koja|koje|što|sto|koliko|gdje|kada)\b',
        r'\b(prikaz|pregled|lista|popis)\b',
        r'\b(get|fetch|retrieve|show|list|find)\b',
    ]

    # ==========================================================================
    # SUFFIX PRIORITY RULES
    # ==========================================================================

    # Suffix priority (higher = more specific, wins in tie-breaker)
    SUFFIX_PRIORITY = {
        '_id_documents_documentId_thumb': 100,
        '_id_documents_documentId_SetAsDefault': 95,
        '_id_documents_documentId': 90,
        '_id_documents': 80,
        '_id_metadata': 70,
        '_DeleteByCriteria': 65,
        '_multipatch': 60,
        '_SetAsDefault': 55,
        '_GroupBy': 50,
        '_ProjectTo': 45,
        '_Agg': 40,
        '_tree': 35,
        '_id': 30,
        '': 10,  # Base endpoint
    }

    # ==========================================================================
    # QUERY CONTEXT PATTERNS
    # ==========================================================================

    # Patterns that indicate specific suffix needed
    SUFFIX_INDICATORS = {
        '_id_documents_documentId': [
            r'\bdokument\w*\s+(s\s+)?id',
            r'\bspecifičn\w*\s+dokument',
            r'\bodređen\w*\s+dokument',
            r'\bjedan\s+dokument',
        ],
        '_id_documents': [
            r'\bsvi\s+dokument',
            r'\bdokument\w*\s+(za|priložen)',
            r'\blista\s+dokumen',
            r'\bpopis\s+dokumen',
        ],
        '_id_metadata': [
            r'\bmetapodatk\w*',
            r'\bmetadata',
            r'\bstruktur\w*\s+podataka',
            r'\bshem\w*',
        ],
        '_multipatch': [
            r'\bviše\s+(stavki|zapisa|entiteta)',
            r'\bbulk',
            r'\bgrupn\w*\s+ažurir',
            r'\bodjednom',
            r'\bid-ev\w*\s+\d',
        ],
        '_DeleteByCriteria': [
            r'\bprema\s+kriterij',
            r'\bkoji\s+zadovoljavaju',
            r'\bfiltr\w*\s+briš',
            r'\buvjet\w*\s+briš',
        ],
        '_id': [
            r'\bs\s+id(-om)?\s+\d',
            r'\bpo\s+id(-u)?\b',
            r'\bjedna\s+stavka',
            r'\bpojedinačn',
            r'\bspecifičn\w*\s+(stavk|zapis)',
        ],
        '_Agg': [
            r'\bagregacij',
            r'\bprosje[čk]',
            r'\bsuma\b',
            r'\bmaksim',
            r'\bminim',
            r'\bstatistik',
        ],
        '_GroupBy': [
            r'\bgrupir',
            r'\bpo\s+(polju|kategorij|vrsti|tipu)',
            r'\bgroup\s*by',
        ],
        '_ProjectTo': [
            r'\bsamo\s+određen\w*\s+polj',
            r'\bprojekcij',
            r'\bodabran\w*\s+(polja|kolone)',
        ],
    }

    def __init__(self, auto_accept_threshold: float = 0.92,
                 score_gap_threshold: float = 0.08):
        """
        Initialize the decision layer.

        Args:
            auto_accept_threshold: Minimum score for auto-accept (higher = more conservative)
            score_gap_threshold: Minimum gap between top-1 and top-2 for auto-accept
        """
        self.auto_accept_threshold = auto_accept_threshold
        self.score_gap_threshold = score_gap_threshold

    def detect_action(self, query: str) -> ActionType:
        """
        Detect action type from query using HARD RULES.

        This is a HARD GATE - the result MUST be respected.
        """
        query_lower = query.lower()

        # Check DELETE first (most destructive, must be explicit)
        for pattern in self.DELETE_PATTERNS:
            if re.search(pattern, query_lower):
                return ActionType.DELETE

        # Check POST (create/add)
        for pattern in self.POST_PATTERNS:
            if re.search(pattern, query_lower):
                return ActionType.POST

        # Check PUT (full update)
        for pattern in self.PUT_PATTERNS:
            if re.search(pattern, query_lower):
                return ActionType.PUT

        # Check PATCH (partial update)
        for pattern in self.PATCH_PATTERNS:
            if re.search(pattern, query_lower):
                return ActionType.PATCH

        # Check generic UPDATE (default to PATCH as safer)
        for pattern in self.UPDATE_PATTERNS:
            if re.search(pattern, query_lower):
                return ActionType.PATCH  # Default to PATCH for safety

        # Check GET (read)
        for pattern in self.GET_PATTERNS:
            if re.search(pattern, query_lower):
                return ActionType.GET

        return ActionType.UNKNOWN

    def detect_required_suffix(self, query: str) -> Optional[str]:
        """
        Detect which suffix the query is asking for.

        Returns the suffix (e.g., '_id_documents') or None.
        """
        query_lower = query.lower()

        # Check each suffix's indicators (most specific first)
        for suffix in sorted(self.SUFFIX_INDICATORS.keys(),
                           key=lambda s: self.SUFFIX_PRIORITY.get(s, 0),
                           reverse=True):
            patterns = self.SUFFIX_INDICATORS[suffix]
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return suffix

        return None

    def get_suffix_priority(self, tool_id: str) -> int:
        """Get priority score for a tool based on its suffix."""
        tool_lower = tool_id.lower()
        for suffix, priority in sorted(self.SUFFIX_PRIORITY.items(),
                                       key=lambda x: len(x[0]),
                                       reverse=True):
            if suffix and tool_lower.endswith(suffix.lower()):
                return priority
        return self.SUFFIX_PRIORITY.get('', 10)

    def apply_tie_breaker(self, query: str, candidates: List[dict]) -> Optional[dict]:
        """
        Apply CONSERVATIVE tie-breaker rules to select best candidate.

        PHILOSOPHY: Trust FAISS embeddings. Only override when we have
        STRONG DETERMINISTIC evidence that a different tool is better.

        Args:
            query: User query
            candidates: List of {tool_id, score} dicts

        Returns:
            Best candidate or None if can't decide
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        query_lower = query.lower()
        faiss_top1 = candidates[0]

        # ================================================================
        # RULE 1: ONLY override FAISS if we have VERY STRONG evidence
        # ================================================================

        # 1a: If query explicitly mentions "NAJNOVIJI/LATEST" and FAISS top-1
        # doesn't have "latest" but another candidate does
        if re.search(r'\b(najnovij|latest|posljednj)\w*\b', query_lower):
            if 'latest' not in faiss_top1['tool_id'].lower():
                latest = [c for c in candidates if 'latest' in c['tool_id'].lower()]
                if latest:
                    # Only override if latest candidate has similar score
                    best_latest = max(latest, key=lambda x: x['score'])
                    if best_latest['score'] >= faiss_top1['score'] - 0.05:
                        return best_latest

        # 1b: If query mentions specific suffix pattern (metadata, documents, etc.)
        # and FAISS top-1 doesn't match but another candidate does
        required_suffix = self.detect_required_suffix(query)
        if required_suffix:
            if required_suffix.lower() not in faiss_top1['tool_id'].lower():
                matching = [c for c in candidates
                           if required_suffix.lower() in c['tool_id'].lower()]
                if matching:
                    best_match = max(matching, key=lambda x: x['score'])
                    if best_match['score'] >= faiss_top1['score'] - 0.05:
                        return best_match

        # ================================================================
        # RULE 2: DEFAULT - TRUST FAISS TOP-1
        # ================================================================
        # FAISS embeddings are good (93%+ Top-3). Don't second-guess them.
        return faiss_top1

    def decide(self, query: str, faiss_results: List[dict]) -> DecisionResult:
        """
        Make final tool selection decision.

        Args:
            query: User query
            faiss_results: List of {tool_id, score, method} from FAISS

        Returns:
            DecisionResult with selected tool and reasoning
        """
        if not faiss_results:
            return DecisionResult(
                tool_id="",
                confidence=0.0,
                decision_method="no_results",
                reasoning="No FAISS results"
            )

        # Step 1: Detect action (HARD GATE)
        detected_action = self.detect_action(query)

        # Step 2: Filter by action (HARD GATE, not suggestion)
        if detected_action != ActionType.UNKNOWN:
            action_methods = {
                ActionType.GET: ["GET"],
                ActionType.POST: ["POST"],
                ActionType.PUT: ["PUT", "PATCH"],  # PUT includes PATCH for updates
                ActionType.PATCH: ["PATCH", "PUT"],
                ActionType.DELETE: ["DELETE"],
            }
            allowed_methods = action_methods.get(detected_action, [])

            filtered = [r for r in faiss_results
                       if r.get('method', 'GET') in allowed_methods]

            if filtered:
                faiss_results = filtered
            else:
                logger.warning(f"Action filter too strict for {detected_action}, using all results")

        # Step 3: Auto-accept if high confidence with big gap
        if len(faiss_results) >= 2:
            top1_score = faiss_results[0]['score']
            top2_score = faiss_results[1]['score']
            score_gap = top1_score - top2_score

            if top1_score >= self.auto_accept_threshold and score_gap >= self.score_gap_threshold:
                return DecisionResult(
                    tool_id=faiss_results[0]['tool_id'],
                    confidence=top1_score,
                    decision_method="auto_accept",
                    reasoning=f"High confidence ({top1_score:.3f}) with gap ({score_gap:.3f})"
                )
        elif len(faiss_results) == 1:
            return DecisionResult(
                tool_id=faiss_results[0]['tool_id'],
                confidence=faiss_results[0]['score'],
                decision_method="auto_accept",
                reasoning="Single result"
            )

        # Step 4: Apply tie-breaker rules
        top_candidates = faiss_results[:5]  # Consider top 5
        best = self.apply_tie_breaker(query, top_candidates)

        if best:
            return DecisionResult(
                tool_id=best['tool_id'],
                confidence=best['score'],
                decision_method="tie_breaker",
                reasoning=f"Deterministic tie-breaker selected from {len(top_candidates)} candidates"
            )

        # Step 5: Fallback - need LLM
        return DecisionResult(
            tool_id=faiss_results[0]['tool_id'],
            confidence=faiss_results[0]['score'],
            decision_method="llm_needed",
            reasoning=f"Could not decide deterministically, LLM may help"
        )


# Singleton instance
_decision_layer = None

def get_decision_layer() -> ToolDecisionLayer:
    """Get singleton decision layer instance."""
    global _decision_layer
    if _decision_layer is None:
        _decision_layer = ToolDecisionLayer()
    return _decision_layer

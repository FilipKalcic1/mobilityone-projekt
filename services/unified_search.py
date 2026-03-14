"""
Unified Search - Single entry point for all tool discovery.

Consolidates all search paths into one consistent interface:
1. ACTION INTENT GATE (detect GET/POST/PUT/DELETE from query)
2. QUERY TYPE CLASSIFIER (detect suffix type: _id, _documents, _metadata, etc.)
3. FAISS semantic search with intent filter
4. Category boost (from tool_categories.json)
5. Documentation boost (from tool_documentation.json)
6. Query type boost (preferred suffixes get priority)

This replaces multiple inconsistent routing paths with a single,
well-tested search pipeline.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from tool_routing import PRIMARY_ACTION_TOOLS as _PRIMARY_ACTION_TOOLS

# ML-based intent detection
from services.intent_classifier import (
    detect_action_intent,
    ActionIntent,
    classify_query_type_ml,
)
from services.faiss_vector_store import get_faiss_store, SearchResult
from services.query_type_classifier import (
    get_query_type_classifier,
    QueryType,
    QueryTypeResult
)

if TYPE_CHECKING:
    from services.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class UnifiedSearchResult:
    """Result from unified search."""
    tool_id: str
    score: float  # Final score after all boosts
    method: str   # HTTP method
    description: str  # Tool description
    origin_guide: Dict[str, str] = field(default_factory=dict)  # Parameter origin guide
    boosts_applied: list = field(default_factory=list)  # Debug: (name, multiplier, score_after) tuples
    base_score: float = 0.0  # Debug: score before boosts


@dataclass
class UnifiedSearchResponse:
    """Complete response from unified search."""
    results: List[UnifiedSearchResult]
    intent: ActionIntent
    intent_confidence: float
    query_type: QueryType  # NEW: Detected query type
    query_type_confidence: float  # NEW: Query type confidence
    query: str
    total_candidates: int  # Before filtering


class UnifiedSearch:
    """
    Single interface for tool discovery.

    All routing paths should use this class for consistent results.

    V2.0 Architecture:
    1. ACTION INTENT GATE (detect GET/POST/PUT/DELETE from query)
    2. QUERY TYPE CLASSIFIER (detect suffix: _id, _documents, _metadata, etc.)
    3. FAISS semantic search with intent filter
    4. Category boosting (from tool_categories.json)
    5. Documentation boosting (from tool_documentation.json)
    6. Query type boosting (preferred suffixes get priority)

    REMOVED: training_queries.json boosting (unreliable)
    """

    # Boost multipliers - v3.1: Aggressive boosts for better differentiation
    CATEGORY_BOOST = 1.25       # 25% boost for matching category
    DOCUMENTATION_BOOST = 1.20  # 20% boost for documentation match
    QUERY_TYPE_BOOST = 2.00     # 100% boost for matching query type suffix (was 50%)
    QUERY_TYPE_PENALTY = 0.40   # 60% penalty for wrong query type suffix (was 40%)
    BASE_ENTITY_BOOST = 1.50    # 50% boost for base entity tools (no complex suffixes)

    # Entity detection keywords for boost matching (class constant, not rebuilt per call)
    ENTITY_KEYWORDS = {
        "companies": ["kompanij", "tvrtk", "firm", "poduzeć"],
        "vehicles": ["vozil", "auto", "automobil", "flot"],
        "persons": ["osob", "korisnik", "zaposlenik", "radnik", "voditelj"],
        "expenses": ["trošk", "troška", "izdatak", "račun", "cijena"],
        "trips": ["putovanj", "trip", "vožnj", "putni"],
        "cases": ["slučaj", "šteta", "steta", "kvar", "incident"],
        "equipment": ["oprem", "uređaj", "stroj"],
        "partners": ["partner", "dobavljač", "klijent"],
        "teams": ["tim", "grupa", "odjel"],
        "orgunits": ["organizacij", "jedinic", "odjel"],
        "costcenters": ["troškovn", "centar troška", "cost center"],
        "vehiclecalendar": ["rezervacij", "booking", "kalendar"],
        "documents": ["dokument", "prilog", "datoteka", "pdf"],
        "metadata": ["metapodac", "struktur", "shema", "polja"],
    }

    def __init__(self, registry: Optional["ToolRegistry"] = None):
        """Initialize unified search."""
        self._registry = registry
        self._tool_documentation: Optional[Dict] = None
        self._tool_categories: Optional[Dict] = None
        self._query_type_classifier = get_query_type_classifier()
        self._initialized = False

    def set_registry(self, registry: "ToolRegistry"):
        """Set tool registry (allows late binding)."""
        self._registry = registry

    async def initialize(self):
        """Load all configuration files."""
        if self._initialized:
            return

        config_dir = Path(__file__).parent.parent / "config"

        # Load tool documentation
        try:
            doc_path = config_dir / "tool_documentation.json"
            if doc_path.exists():
                with open(doc_path, 'r', encoding='utf-8') as f:
                    self._tool_documentation = json.load(f)
                logger.info(f"UnifiedSearch v2.0: Loaded {len(self._tool_documentation)} tool docs")
        except Exception as e:
            logger.warning(f"Failed to load tool documentation: {e}")
            self._tool_documentation = {}

        # Load tool categories
        try:
            cat_path = config_dir / "tool_categories.json"
            if cat_path.exists():
                with open(cat_path, 'r', encoding='utf-8') as f:
                    self._tool_categories = json.load(f)
                logger.info(f"UnifiedSearch v2.0: Loaded tool categories")
        except Exception as e:
            logger.warning(f"Failed to load tool categories: {e}")
            self._tool_categories = {}

        # NOTE: training_queries.json is NO LONGER USED
        # It was unreliable and caused confusion in tool selection

        self._initialized = True

    async def search(
        self,
        query: str,
        top_k: int = 20
    ) -> UnifiedSearchResponse:
        """
        Unified search entry point.

        All routing paths should call this method.

        Args:
            query: User query in Croatian
            top_k: Number of results to return

        Returns:
            UnifiedSearchResponse with ranked results
        """
        await self.initialize()

        # Step 1: ACTION INTENT DETECTION (v16.0: less aggressive filtering)
        intent_result = detect_action_intent(query)

        # Only filter by intent if confidence is very high (95%+)
        # Otherwise, show all tools and let LLM decide
        # This prevents filtering out valid tools when ML is uncertain
        INTENT_FILTER_THRESHOLD = 0.95
        action_filter = (
            intent_result.intent.value
            if intent_result.intent != ActionIntent.UNKNOWN
            and intent_result.confidence >= INTENT_FILTER_THRESHOLD
            else None  # Don't filter - show all tools, let LLM decide
        )

        # Step 2: QUERY TYPE CLASSIFICATION
        # Use ML-based classifier (replaces 91 regex patterns)
        ml_query_type = classify_query_type_ml(query)

        # Convert ML result to QueryTypeResult for backwards compatibility
        query_type_result = QueryTypeResult(
            query_type=QueryType[ml_query_type.query_type] if ml_query_type.query_type in QueryType.__members__ else QueryType.UNKNOWN,
            confidence=ml_query_type.confidence,
            matched_pattern="ML",  # No regex pattern - using ML
            preferred_suffixes=ml_query_type.preferred_suffixes,
            excluded_suffixes=ml_query_type.excluded_suffixes
        )

        logger.info(
            f"UnifiedSearch v16.0: Intent={intent_result.intent.value} "
            f"(conf={intent_result.confidence:.2f}), "
            f"QueryType={query_type_result.query_type.value} [ML] "
            f"(conf={query_type_result.confidence:.2f}), "
            f"ActionFilter={'YES: ' + action_filter if action_filter else 'NO - LLM decides'}"
        )

        # Step 2.5: EXACT MATCH DETECTION
        # If query exactly matches an example query, boost that tool significantly
        exact_match_tool = None
        query_normalized = query.lower().strip().rstrip('.')

        if self._tool_documentation:
            for tool_id, doc in self._tool_documentation.items():
                examples = doc.get('example_queries_hr', [])
                for example in examples:
                    example_normalized = example.lower().strip().rstrip('.')
                    # Check for exact or near-exact match
                    if query_normalized == example_normalized:
                        exact_match_tool = tool_id
                        logger.info(f"UnifiedSearch: EXACT MATCH found: {tool_id}")
                        break
                    # Also check if query contains example or vice versa (for partial matches)
                    elif len(query_normalized) > 10 and len(example_normalized) > 10:
                        if query_normalized in example_normalized or example_normalized in query_normalized:
                            if exact_match_tool is None:
                                exact_match_tool = tool_id
                if exact_match_tool:
                    break

        # Step 3: FAISS search with intent filter
        faiss_store = get_faiss_store()

        if not faiss_store.is_initialized():
            logger.warning("UnifiedSearch: FAISS not initialized, returning empty results")
            return UnifiedSearchResponse(
                results=[],
                intent=intent_result.intent,
                intent_confidence=intent_result.confidence,
                query_type=query_type_result.query_type,
                query_type_confidence=query_type_result.confidence,
                query=query,
                total_candidates=0
            )

        # Get MORE results to allow boosting to bring correct tools to top
        # FAISS embeddings may not have correct tool in top 10, but boost can fix it
        # V3.0: Enable auto_detect_entity for better entity-based filtering
        faiss_results = await faiss_store.search(
            query=query,
            top_k=max(top_k * 2, 40),  # Get enough for boosting without wasting computation
            action_filter=action_filter,
            auto_detect_entity=True  # V3.0: Enable entity detection for search space reduction
        )

        total_candidates = len(faiss_results)

        if not faiss_results:
            logger.info("UnifiedSearch: FAISS returned no results")
            return UnifiedSearchResponse(
                results=[],
                intent=intent_result.intent,
                intent_confidence=intent_result.confidence,
                query_type=query_type_result.query_type,
                query_type_confidence=query_type_result.confidence,
                query=query,
                total_candidates=0
            )

        # Step 4: Apply all boosts (including query type boost)
        boosted_results = self._apply_boosts(query, faiss_results, query_type_result)

        # Step 4.5: EXACT MATCH INJECTION
        # If we found an exact match, ensure it's #1 regardless of FAISS
        if exact_match_tool:
            # Check if tool is already in results
            found = False
            for result in boosted_results:
                if result.tool_id == exact_match_tool:
                    result.score = 100.0  # Maximum score to ensure #1
                    found = True
                    break

            # If not in results, inject it
            if not found:
                from services.faiss_vector_store import SearchResult
                boosted_results.insert(0, SearchResult(
                    tool_id=exact_match_tool,
                    score=100.0,
                    method=self._tool_methods.get(exact_match_tool, "GET") if hasattr(self, '_tool_methods') else "GET"
                ))

        # Step 5: Sort by final score (boost system already applied entity/suffix boosts)
        boosted_results.sort(key=lambda r: r.score, reverse=True)

        # Step 6: Convert to final format
        final_results = []
        for r in boosted_results[:top_k]:
            # Get description from registry or documentation
            description = ""
            origin_guide = {}

            if self._registry:
                tool = self._registry.get_tool(r.tool_id)
                if tool:
                    description = tool.summary or tool.description or ""

            if self._tool_documentation and r.tool_id in self._tool_documentation:
                doc = self._tool_documentation[r.tool_id]
                if not description:
                    description = doc.get("purpose", "")
                origin_guide = doc.get("parameter_origin_guide", {})

            final_results.append(UnifiedSearchResult(
                tool_id=r.tool_id,
                score=r.score,
                method=r.method,
                description=description[:250],  # Truncate for efficiency
                origin_guide=origin_guide,
                boosts_applied=getattr(r, 'boosts_applied', [])
            ))

        logger.info(
            f"UnifiedSearch v2.0: Query '{query[:30]}...' -> "
            f"{len(final_results)} results (top: {final_results[0].tool_id if final_results else 'none'})"
        )

        return UnifiedSearchResponse(
            results=final_results,
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            query_type=query_type_result.query_type,
            query_type_confidence=query_type_result.confidence,
            query=query,
            total_candidates=total_candidates
        )

    # Single source of truth: config/tool_routing.py
    PRIMARY_ACTION_TOOLS = _PRIMARY_ACTION_TOOLS

    def _apply_boosts(
        self,
        query: str,
        results: List[SearchResult],
        query_type_result: QueryTypeResult
    ) -> List[SearchResult]:
        """Apply all boosting factors to results."""
        from services.intent_classifier import normalize_diacritics
        query_lower = normalize_diacritics(query.lower())

        # Get matched categories
        matched_categories = self._match_categories(query_lower)

        # Get preferred/excluded suffixes from query type
        preferred_suffixes = query_type_result.preferred_suffixes
        excluded_suffixes = query_type_result.excluded_suffixes
        query_type_confidence = query_type_result.confidence

        # ENTITY DETECTION - pre-compute once outside loop (was N*M inside loop)
        detected_entity = None
        for entity, keywords in self.ENTITY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                detected_entity = entity
                break

        # Pre-compute query words for documentation matching (normalized for diacritics)
        query_words = [w for w in query_lower.split() if len(w) > 3]

        for result in results:
            boosts = []  # List of (name, multiplier, score_after)
            base_score = result.score  # Track original score for logging
            tool_lower = result.tool_id.lower()

            # V3.0: PRIMARY ACTION TOOL BOOST
            # Apply significant boost if query matches keywords for primary user-facing tools
            if tool_lower in self.PRIMARY_ACTION_TOOLS:
                tool_config = self.PRIMARY_ACTION_TOOLS[tool_lower]
                if any(kw in query_lower for kw in tool_config["keywords"]):
                    result.score *= tool_config["boost"]
                    boosts.append(("primary_action", tool_config["boost"], result.score))

            # ENTITY MATCH BOOST - AGGRESSIVE
            # If query mentions specific entity, strongly boost/penalize tools
            if detected_entity:
                # Extract entity from tool_id (e.g., get_Companies_id -> companies)
                tool_parts = result.tool_id.split("_")
                if len(tool_parts) >= 2:
                    tool_entity = tool_parts[1].lower()
                    if tool_entity == detected_entity or detected_entity in tool_entity:
                        result.score *= 2.50  # 150% boost for entity match (was 80%)
                        boosts.append(("entity_match", 2.50, result.score))
                    else:
                        # Penalize tools that don't match detected entity
                        result.score *= 0.50  # 50% penalty (was 30%)
                        boosts.append(("entity_mismatch", 0.50, result.score))

            # V3.0: GENERIC CRUD PENALTY
            # Penalize generic CRUD endpoints when specific action tools are more appropriate
            generic_crud_penalty = {
                "post_cases": {"penalty_if": ["šteta", "steta", "kvar", "udario", "ogrebao"], "factor": 0.3},
                "post_vehicles": {"penalty_if": ["rezerv", "booking", "trebam"], "factor": 0.4},
                "post_vehicleshistoricalentries": {"penalty_if": ["rezerv", "booking", "trebam"], "factor": 0.3},
                "delete_triptypes_deletebycriteria": {"penalty_if": ["booking", "rezerv"], "factor": 0.3},
                "get_monthlymileages_agg": {"penalty_if": ["koliko", "stanje", "imam"], "factor": 0.4},
                "get_monthlymileagesassigned": {"penalty_if": ["koliko", "moja"], "factor": 0.4},
            }
            if tool_lower in generic_crud_penalty:
                penalty_config = generic_crud_penalty[tool_lower]
                if any(kw in query_lower for kw in penalty_config["penalty_if"]):
                    result.score *= penalty_config["factor"]
                    boosts.append(("generic_crud_penalty", penalty_config["factor"], result.score))

            # Category boost
            if matched_categories:
                tool_categories = self._get_tool_categories(result.tool_id)
                if any(cat in tool_categories for cat in matched_categories):
                    result.score *= self.CATEGORY_BOOST
                    boosts.append(("category", self.CATEGORY_BOOST, result.score))

            # Documentation boost (if query words appear in tool documentation)
            if self._tool_documentation and result.tool_id in self._tool_documentation:
                doc = self._tool_documentation[result.tool_id]
                doc_text = " ".join([
                    doc.get("purpose", ""),
                    " ".join(doc.get("when_to_use", [])),
                    " ".join(doc.get("example_queries_hr", []))
                ]).lower()

                # Check if query words appear in documentation (uses pre-computed query_words)
                if any(word in doc_text for word in query_words):
                    result.score *= self.DOCUMENTATION_BOOST
                    boosts.append(("doc", self.DOCUMENTATION_BOOST, result.score))

            # Query Type boost - only apply when classifier is reasonably confident
            if query_type_confidence >= 0.65 and preferred_suffixes:
                # Boost if tool matches preferred suffix
                is_preferred = any(
                    tool_lower.endswith(suffix.lower())
                    for suffix in preferred_suffixes
                )
                if is_preferred:
                    result.score *= self.QUERY_TYPE_BOOST
                    boosts.append(("query_type", self.QUERY_TYPE_BOOST, result.score))

                # Penalize if tool matches excluded suffix
                is_excluded = any(
                    tool_lower.endswith(suffix.lower())
                    for suffix in excluded_suffixes
                )
                if is_excluded:
                    result.score *= self.QUERY_TYPE_PENALTY  # 40% penalty (v3.0: increased)
                    boosts.append(("excluded", self.QUERY_TYPE_PENALTY, result.score))

            # Base Entity Boost (NEW in v2.0)
            # For LIST queries: boost tools without complex suffixes (get_X, not get_X_id_documents)
            # For SINGLE_ENTITY queries: boost tools with simple _id suffix
            if query_type_result.query_type == QueryType.LIST:
                # Check if this is a base list endpoint (no _id, no _documents, etc.)
                is_base_list = self._is_base_list_tool(tool_lower)
                if is_base_list:
                    # HIGHEST boost for PRIMARY entities (get_Companies, get_Vehicles, get_Persons)
                    is_primary = self._is_pure_entity_tool(tool_lower)
                    if is_primary:
                        result.score *= 1.80  # 80% boost - highest priority
                        boosts.append(("primary_entity", 1.80, result.score))
                    # Medium boost for secondary entities (Types, Groups, etc.)
                    elif self._is_secondary_entity_tool(tool_lower):
                        result.score *= 1.40  # 40% boost
                        boosts.append(("secondary_entity", 1.40, result.score))
                    # Small boost for other base list tools
                    else:
                        result.score *= self.BASE_ENTITY_BOOST
                        boosts.append(("base_list", self.BASE_ENTITY_BOOST, result.score))
                # Penalize Lookup, Helper, Aggregate, and filter-type tools for generic list queries
                # V3.0: Extended penalty list to catch more false positives
                penalty_patterns = [
                    'lookup', 'helper', 'input', 'available', 'latest', 'monthly',
                    'dashboard', 'stats', '_agg', '_groupby', '_projectto',
                    'historicalentries', 'assigned', 'fileids', 'distinctbrands'
                ]
                if any(x in tool_lower for x in penalty_patterns):
                    result.score *= 0.4  # 60% penalty (increased from 50%)
                    boosts.append(("helper_penalty", 0.4, result.score))

            elif query_type_result.query_type == QueryType.SINGLE_ENTITY:
                # Check if this is a simple _id endpoint
                is_simple_id = self._is_simple_id_tool(tool_lower)
                if is_simple_id:
                    # Extra boost for PRIMARY entity _id endpoints
                    # e.g., get_Vehicles_id over get_VehicleCalendar_id
                    entity_name = tool_lower[4:].replace('_id', '')  # "vehicles"
                    primary_entities = [
                        'companies', 'vehicles', 'persons', 'expenses', 'cases',
                        'teams', 'trips', 'partners', 'tenants', 'roles', 'tags',
                        'pools', 'orgunits', 'costcenters', 'equipment',
                    ]
                    if entity_name in primary_entities:
                        result.score *= 1.80  # 80% boost for primary entities
                        boosts.append(("primary_id", 1.80, result.score))
                    else:
                        result.score *= self.BASE_ENTITY_BOOST
                        boosts.append(("simple_id", self.BASE_ENTITY_BOOST, result.score))
                # Penalize complex suffixes for single entity queries
                if any(s in tool_lower for s in ['_documents', '_metadata', '_thumb', '_agg', '_groupby']):
                    result.score *= 0.6  # 40% penalty
                    boosts.append(("complex_suffix_penalty", 0.6, result.score))
                # Penalize Lookup tools
                if 'lookup' in tool_lower:
                    result.score *= 0.5  # 50% penalty
                    boosts.append(("lookup_penalty", 0.5, result.score))

            # Cap score at 1.0
            result.score = min(result.score, 1.0)

            # Store boosts for debugging
            result.boosts_applied = boosts
            result.base_score = base_score

        # Log per-boost breakdown for top-3 results with intermediate scores
        top3 = sorted(results, key=lambda r: r.score, reverse=True)[:3]
        for r in top3:
            if r.boosts_applied:
                chain = " \u2192 ".join(
                    f"{name}(\u00d7{mult:.1f})={score:.3f}"
                    for name, mult, score in r.boosts_applied
                )
                logger.debug(
                    "Boost: %s base=%.3f \u2192 %s \u2192 cap=%.3f",
                    r.tool_id, r.base_score, chain, r.score
                )

        return results

    def _match_categories(self, query: str) -> List[str]:
        """Match query to categories based on keywords."""
        if not self._tool_categories:
            return []

        matched = []
        categories = self._tool_categories.get("categories", {})

        for cat_name, cat_data in categories.items():
            keywords = cat_data.get("keywords", [])
            if any(kw.lower() in query for kw in keywords):
                matched.append(cat_name)

        return matched

    def _get_tool_categories(self, tool_id: str) -> List[str]:
        """Get categories for a tool."""
        if not self._tool_categories:
            return []

        tool_map = self._tool_categories.get("tool_to_categories", {})
        return tool_map.get(tool_id, [])

    # NOTE: _match_example_queries() REMOVED in v2.0
    # training_queries.json was unreliable and caused confusion
    # Replaced by Query Type Classifier for better suffix handling

    def _is_base_list_tool(self, tool_lower: str) -> bool:
        """
        Check if tool is a base list endpoint (e.g., get_Companies, get_Vehicles).
        These are simple endpoints without complex suffixes.
        """
        # Must start with get_
        if not tool_lower.startswith('get_'):
            return False

        # Must NOT contain complex suffixes
        complex_suffixes = [
            '_id', '_documents', '_metadata', '_thumb', '_agg', '_groupby',
            '_projectto', '_tree', '_deletebycriteria', 'lookup', 'helper',
            'input', 'stats', '_on', '_from', '_to'
        ]

        for suffix in complex_suffixes:
            if suffix in tool_lower:
                return False

        return True

    def _is_pure_entity_tool(self, tool_lower: str) -> bool:
        """
        Check if tool is a PURE entity list endpoint (e.g., get_Companies, get_Vehicles).
        Not get_AvailableVehicles, get_LatestX, etc.
        """
        # Must start with get_
        if not tool_lower.startswith('get_'):
            return False

        # Extract entity name after get_
        entity_part = tool_lower[4:]  # Remove "get_"

        # Only PRIMARY entities - not Types, Groups, etc.
        # These are the main CRUD entities users typically want
        primary_entities = [
            'companies', 'vehicles', 'persons', 'expenses', 'cases',
            'teams', 'trips', 'partners', 'tenants', 'roles', 'tags',
            'pools', 'orgunits', 'costcenters', 'equipment', 'booking',
        ]

        return entity_part in primary_entities

    def _is_secondary_entity_tool(self, tool_lower: str) -> bool:
        """
        Check if tool is a secondary entity (Types, Groups, etc.).
        These get a smaller boost than primary entities.
        """
        if not tool_lower.startswith('get_'):
            return False

        entity_part = tool_lower[4:]

        secondary_entities = [
            'vehicletypes', 'persontypes', 'expensetypes', 'casetypes',
            'equipmenttypes', 'triptypes', 'documenttypes', 'expensegroups',
            'vehiclecontracts', 'vehiclecalendar', 'equipmentcalendar',
            'periodicactivities', 'mileagereports', 'schedulingmodels',
        ]

        return entity_part in secondary_entities

    def _is_simple_id_tool(self, tool_lower: str) -> bool:
        """
        Check if tool is a simple _id endpoint (e.g., get_Companies_id, get_Vehicles_id).
        These retrieve a single entity by ID without additional nesting.
        """
        # Must start with get_
        if not tool_lower.startswith('get_'):
            return False

        # Must end with _id (not _id_something)
        if not tool_lower.endswith('_id'):
            return False

        # Must NOT contain complex nested patterns
        if '_id_' in tool_lower:  # e.g., _id_documents, _id_metadata
            return False

        # Must NOT be a Lookup tool
        if 'lookup' in tool_lower:
            return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the search system."""
        faiss_store = get_faiss_store()

        return {
            "version": "2.0",
            "initialized": self._initialized,
            "faiss_initialized": faiss_store.is_initialized() if faiss_store else False,
            "faiss_stats": faiss_store.get_stats() if faiss_store else {},
            "tool_docs_loaded": len(self._tool_documentation) if self._tool_documentation else 0,
            "categories_loaded": bool(self._tool_categories),
            "query_type_classifier": "enabled",
            # NOTE: training_examples no longer used in v2.0
        }


# Singleton instance
_unified_search: Optional[UnifiedSearch] = None


def get_unified_search() -> UnifiedSearch:
    """Get singleton UnifiedSearch instance."""
    global _unified_search
    if _unified_search is None:
        _unified_search = UnifiedSearch()
    return _unified_search


async def initialize_unified_search(
    registry: Optional["ToolRegistry"] = None
) -> UnifiedSearch:
    """
    Initialize and return the unified search.

    Call this during application startup.
    """
    search = get_unified_search()
    if registry:
        search.set_registry(registry)
    await search.initialize()
    return search

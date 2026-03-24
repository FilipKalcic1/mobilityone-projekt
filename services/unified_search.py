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
    normalize_diacritics,
)
from services.faiss_vector_store import get_faiss_store, SearchResult
from services.entity_detector import detect_entity, detect_possessive, detect_user_profile_query
from services.query_type_classifier import (
    get_query_type_classifier,
    QueryType,
    QueryTypeResult
)
from services.dynamic_threshold import (
    get_engine as _get_engine, DecisionEngine, ClassificationSignal, NO_SIGNAL,
)

# Decomposed search modules (v12.0)
from services.search.config import (
    VERB_METHOD_MAP as _VERB_METHOD_MAP,
    ID_INDICATORS as _ID_INDICATORS,
    CRITERIA_INDICATORS as _CRITERIA_INDICATORS,
    METHOD_VERBS as _METHOD_VERBS,
    SUFFIX_DESCRIPTIONS as _SUFFIX_DESCRIPTIONS,
    BM25_WEIGHT as _BM25_WEIGHT,
)
from services.search.boost_engine import (
    BoostContext,
    apply_boosts as _apply_boosts_fn,
)
from services.tracing import get_tracer, trace_span
from services.errors import SearchError, ErrorCode

if TYPE_CHECKING:
    from services.registry import ToolRegistry

logger = logging.getLogger(__name__)
_tracer = get_tracer("unified_search")


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


# Specialized query types that need wider FAISS pool
_SPECIALIZED_QUERY_TYPES = frozenset({
    QueryType.AGGREGATION, QueryType.TREE, QueryType.BULK_UPDATE,
    QueryType.PROJECTION, QueryType.DOCUMENTS, QueryType.METADATA,
})


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

    v12.0: Boost constants and entity keywords extracted to services/search/config.py.
    Boost logic extracted to services/search/boost_engine.py.
    """

    def __init__(self, registry: Optional["ToolRegistry"] = None) -> None:
        """Initialize unified search."""
        self._registry = registry
        self._tool_documentation: Optional[Dict] = None
        self._tool_categories: Optional[Dict] = None
        self._query_type_classifier = get_query_type_classifier()
        self._initialized = False

    def set_registry(self, registry: "ToolRegistry") -> None:
        """Set tool registry (allows late binding)."""
        self._registry = registry

    async def initialize(self) -> None:
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
            err = SearchError(ErrorCode.TOOL_DOCS_NOT_LOADED, f"Failed to load tool documentation: {e}")
            logger.warning(str(err))
            self._tool_documentation = {}

        # Load tool categories
        try:
            cat_path = config_dir / "tool_categories.json"
            if cat_path.exists():
                with open(cat_path, 'r', encoding='utf-8') as f:
                    self._tool_categories = json.load(f)
                logger.info("UnifiedSearch v2.0: Loaded tool categories")
        except Exception as e:
            err = SearchError(ErrorCode.TOOL_DOCS_NOT_LOADED, f"Failed to load tool categories: {e}")
            logger.warning(str(err))
            self._tool_categories = {}

        # NOTE: training_queries.json is NO LONGER USED
        # It was unreliable and caused confusion in tool selection

        # Pre-compute duplicate purposes for entity-specific description generation
        self._duplicate_purposes: set = set()
        if self._tool_documentation:
            from collections import Counter
            purpose_counts = Counter(
                doc.get("purpose", "").strip()
                for doc in self._tool_documentation.values()
                if doc.get("purpose", "").strip()
            )
            self._duplicate_purposes = {p for p, c in purpose_counts.items() if c > 1}
            if self._duplicate_purposes:
                logger.info(f"Detected {len(self._duplicate_purposes)} duplicate purposes across tools")

        # Pre-compute exact match lookup: normalized_query → tool_id (O(1) per search)
        self._exact_match_index: Dict[str, str] = {}
        if self._tool_documentation:
            for tool_id, doc in self._tool_documentation.items():
                for example in doc.get("example_queries_hr", []):
                    key = example.lower().strip().rstrip('.')
                    # First tool wins for duplicate queries
                    if key not in self._exact_match_index:
                        self._exact_match_index[key] = tool_id
            if self._exact_match_index:
                logger.info(f"Built exact match index: {len(self._exact_match_index)} entries")

        # Load auto-generated entity stems from tool_documentation.json
        if self._tool_documentation:
            from services.entity_detector import load_auto_stems
            load_auto_stems(self._tool_documentation)

        # Build Tool Family Index from registry
        if self._registry:
            from services.tool_family_index import get_family_index
            family_index = get_family_index()
            # registry.tools is a dict keyed by tool_id
            tool_ids = list(self._registry.tools.keys()) if isinstance(self._registry.tools, dict) else [t.tool_id for t in self._registry.tools if hasattr(t, 'tool_id')]
            if tool_ids:
                family_index.build(tool_ids)

        # Build BM25 index from tool documentation
        if self._tool_documentation:
            from services.bm25_index import get_bm25_index
            self._bm25_index = get_bm25_index()
            self._bm25_index.build(self._tool_documentation)
        else:
            self._bm25_index = None

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

        with trace_span(_tracer, "search.unified", {
            "search.top_k": top_k,
            "query.preview": query[:80],
        }) as span:
            # Step 1: ACTION INTENT DETECTION (v16.0: less aggressive filtering)
            intent_result = detect_action_intent(query)

            # Only filter by intent if confidence is very high
            # Otherwise, show all tools and let LLM decide
            # This prevents filtering out valid tools when ML is uncertain
            _engine = _get_engine()
            _intent_signal = intent_result.signal  # Full distribution signal
            action_filter = (
                intent_result.intent.value
                if intent_result.intent != ActionIntent.UNKNOWN
                and _engine.decide(_intent_signal, DecisionEngine.INTENT_FILTER).is_accept
                else None  # Don't filter - show all tools, let LLM decide
            )

            # Step 2: QUERY TYPE CLASSIFICATION
            # Use ML-based classifier (replaces 91 regex patterns)
            ml_query_type = classify_query_type_ml(query)
            _qt_signal = ml_query_type.signal  # Full distribution signal

            # Convert ML result to QueryTypeResult for backwards compatibility
            query_type_result = QueryTypeResult(
                query_type=QueryType[ml_query_type.query_type] if ml_query_type.query_type in QueryType.__members__ else QueryType.UNKNOWN,
                confidence=ml_query_type.confidence,
                matched_pattern="ML",  # No regex pattern - using ML
                preferred_suffixes=ml_query_type.preferred_suffixes,
                excluded_suffixes=ml_query_type.excluded_suffixes,
                signal=ml_query_type.signal,
            )

            logger.info(
                f"UnifiedSearch v16.0: Intent={intent_result.intent.value} "
                f"(conf={intent_result.confidence:.2f}), "
                f"QueryType={query_type_result.query_type.value} [ML] "
                f"(conf={query_type_result.confidence:.2f}), "
                f"ActionFilter={'YES: ' + action_filter if action_filter else 'NO - LLM decides'}"
            )

            # Step 2.5: EXACT MATCH DETECTION (O(1) lookup via pre-computed index)
            query_normalized = query.lower().strip().rstrip('.')
            exact_match_tool = self._exact_match_index.get(query_normalized)
            if exact_match_tool:
                logger.info(f"UnifiedSearch: EXACT MATCH found: {exact_match_tool}")

            # Step 2.7: ENTITY DETECTION (early, for TFI override)
            detected_entity = detect_entity(query)

            # Step 2.7a: POSSESSIVE SIGNAL — "moj auto" vs "sva vozila"
            is_possessive, poss_entity = detect_possessive(query)
            if is_possessive and poss_entity and not detected_entity:
                detected_entity = poss_entity

            # Step 2.7b: USER PROFILE QUERY — "koji je moj broj telefona?"
            is_user_profile = detect_user_profile_query(query)

            # SHORT QUERY OVERRIDE + MUTATION CORRECTION
            query_lower = query.lower()
            query_lower_norm = normalize_diacritics(query_lower)
            effective_query_type = query_type_result.query_type

            # Verb-based method detection — more reliable than ML for common Croatian verbs
            detected_method = None
            for verb, method in _VERB_METHOD_MAP.items():
                if verb in query_lower_norm:
                    detected_method = method
                    break

            is_mutation = (
                (intent_result.intent in (ActionIntent.DELETE, ActionIntent.UPDATE, ActionIntent.PATCH, ActionIntent.CREATE)
                 and not _engine.decide(_intent_signal, DecisionEngine.MUTATION).is_defer)
                or detected_method in ("delete", "put", "patch", "post")
            )

            # Override 1: bare entity names (1-2 words) without ID indicators → LIST
            # Only for GET queries. "auto" → LIST, but "obriši trošak" stays SINGLE_ENTITY.
            # Exception: possessive queries ("moj auto") stay SINGLE_ENTITY.
            if (effective_query_type == QueryType.SINGLE_ENTITY
                    and len(query.split()) <= 2
                    and not _engine.decide(_qt_signal, DecisionEngine.INTENT_FILTER).is_accept
                    and not any(ind in query_lower_norm for ind in _ID_INDICATORS)
                    and not is_mutation
                    and not is_possessive):
                effective_query_type = QueryType.LIST

            # Override 0: Entity detected + UNKNOWN → LIST
            # "radnik" has entity=persons but queryType=UNKNOWN → default to LIST
            if detected_entity and effective_query_type == QueryType.UNKNOWN and not is_mutation:
                effective_query_type = QueryType.LIST

            # Override 1a: Possessive + LIST → SINGLE_ENTITY
            # "koji je moj auto?" should be SINGLE_ENTITY, not LIST
            # D5 fix: require minimum confidence to avoid flipping on random guesses
            if (is_possessive
                    and effective_query_type == QueryType.LIST
                    and not _engine.decide(_qt_signal, DecisionEngine.POSSESSIVE).is_defer):
                effective_query_type = QueryType.SINGLE_ENTITY

            # Override 2: DELETE_CRITERIA → SINGLE_ENTITY when no criteria words present
            # "obriši trošak" = single delete, "obriši sve troškove za 2023" = criteria delete
            if (effective_query_type == QueryType.DELETE_CRITERIA
                    and not any(ind in query_lower_norm for ind in _CRITERIA_INDICATORS)):
                effective_query_type = QueryType.SINGLE_ENTITY

            # Step 2.7c: USER PROFILE FLAG (defence-in-depth for ML fallback)
            # Profile tools are injected into results after boost pipeline (Step 4.7).

            # Step 2.8: TFI HARD OVERRIDE — deterministic path (~70% of queries)
            # When entity + queryType are both known, skip FAISS entirely.
            # Return the whole entity family so LLM has context for final pick.
            from services.tool_family_index import get_family_index
            family_index = get_family_index()
            tfi_override_result = None

            if detected_entity and effective_query_type != QueryType.UNKNOWN:
                # Prefer verb-detected method over ML intent (more reliable for Croatian)
                tfi_method = detected_method or (
                    intent_result.intent.value.lower()
                    if intent_result.intent not in (ActionIntent.UNKNOWN, ActionIntent.NONE)
                    and not _engine.decide(_intent_signal, DecisionEngine.MUTATION).is_defer
                    else None
                )
                # Aggregation sub-variant: "agregiraj" → agg, "grupiraj" → groupby
                tfi_variant = None
                if effective_query_type == QueryType.AGGREGATION:
                    if any(w in query_lower_norm for w in ["agregir", "agregiraj", "agregac"]):
                        tfi_variant = "agg"
                    elif any(w in query_lower_norm for w in ["grupir", "grupiraj"]):
                        tfi_variant = "groupby"

                tfi_tool = family_index.resolve(
                    detected_entity, effective_query_type,
                    method=tfi_method, variant_override=tfi_variant
                )
                family_tools = family_index.get_family_tools(detected_entity)

                if tfi_tool and family_tools:
                    # Build results from entire family, TFI match gets highest score
                    from services.faiss_vector_store import SearchResult
                    family_results = []
                    seen_tools = set()

                    for _variant, tool_id in family_tools.items():
                        score = 2.0 if tool_id.lower() == tfi_tool.lower() else 1.0
                        parts = tool_id.split("_")
                        method = parts[0].upper() if parts else "GET"
                        family_results.append(SearchResult(
                            tool_id=tool_id,
                            score=score,
                            method=method
                        ))
                        seen_tools.add(tool_id.lower())

                    # Inject TFI resolved tool if not in GET-preferred family
                    # (e.g., delete_Expenses_id won't be in simple family)
                    if tfi_tool.lower() not in seen_tools:
                        parts = tfi_tool.split("_")
                        family_results.append(SearchResult(
                            tool_id=tfi_tool,
                            score=2.0,
                            method=parts[0].upper() if parts else "GET"
                        ))

                    # Also inject exact match if found
                    if exact_match_tool:
                        found_exact = any(r.tool_id == exact_match_tool for r in family_results)
                        if not found_exact:
                            parts = exact_match_tool.split("_")
                            family_results.append(SearchResult(
                                tool_id=exact_match_tool,
                                score=15.0,
                                method=parts[0].upper() if parts else "GET"
                            ))
                        else:
                            for r in family_results:
                                if r.tool_id == exact_match_tool:
                                    r.score = 15.0

                    family_results.sort(key=lambda r: r.score, reverse=True)
                    tfi_override_result = family_results
                    logger.info(
                        f"UnifiedSearch TFI OVERRIDE: entity={detected_entity}, "
                        f"queryType={effective_query_type.value}, "
                        f"resolved={tfi_tool}, family_size={len(family_results)}"
                    )

            # Step 3: FAISS search (skipped if TFI override succeeded)
            if tfi_override_result is not None:
                faiss_results = tfi_override_result
                total_candidates = len(faiss_results)
                # Skip boost pipeline — TFI results are already scored
                boosted_results = faiss_results
            else:
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

                # Dynamic pool sizing (v4.0): specialized suffixes need wider pool
                pool_size = 120 if query_type_result.query_type in _SPECIALIZED_QUERY_TYPES else max(top_k * 3, 80)

                faiss_results = await faiss_store.search(
                    query=query,
                    top_k=pool_size,
                    action_filter=action_filter,
                    auto_detect_entity=True
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

            # Step 4: Apply boosts (only for FAISS path, TFI path already scored)
            if tfi_override_result is None:
                boosted_results = self._apply_boosts(
                    query, faiss_results, query_type_result,
                    detected_entity=detected_entity,
                    effective_query_type=effective_query_type,
                    is_possessive=is_possessive,
                    qt_signal=_qt_signal,
                )

                # Step 4.3: BM25 HYBRID BOOST
                # Add BM25 exact-term matching scores to complement FAISS semantic scores
                if self._bm25_index and self._bm25_index.is_built:
                    bm25_tool_ids = [r.tool_id for r in boosted_results]
                    bm25_scores = self._bm25_index.get_scores_batch(query, bm25_tool_ids)
                    if bm25_scores:
                        # Normalize BM25 scores to [0, 1] range
                        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
                        for r in boosted_results:
                            bm25_raw = bm25_scores.get(r.tool_id, 0.0)
                            if bm25_raw > 0:
                                bm25_normalized = bm25_raw / max_bm25
                                bm25_boost = _BM25_WEIGHT * bm25_normalized
                                r.score += bm25_boost

                # Step 4.5: EXACT MATCH INJECTION
                if exact_match_tool:
                    found = False
                    for result in boosted_results:
                        if result.tool_id == exact_match_tool:
                            result.score = 15.0
                            found = True
                            break
                    if not found:
                        from services.faiss_vector_store import SearchResult
                        boosted_results.insert(0, SearchResult(
                            tool_id=exact_match_tool,
                            score=15.0,
                            method=self._tool_methods.get(exact_match_tool, "GET") if hasattr(self, '_tool_methods') else "GET"
                        ))
            else:
                boosted_results = faiss_results

            # Step 4.7: USER PROFILE INJECTION (D1 fix)
            # Inject profile tools so LLM fallback has the right candidates.
            if is_user_profile and not is_mutation:
                from services.faiss_vector_store import SearchResult as _SR
                _profile_tools = [
                    ("get_PersonData_personIdOrEmail", 2.0),
                    ("get_Persons_id", 1.5),
                ]
                for pt_id, pt_score in _profile_tools:
                    if self._registry and self._registry.get_tool(pt_id):
                        if not any(r.tool_id == pt_id for r in boosted_results):
                            boosted_results.append(_SR(tool_id=pt_id, score=pt_score, method="GET"))

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

                # If purpose is shared by multiple tools, prepend entity-specific prefix
                # so the LLM can differentiate (e.g., "Obriši Expenses po ID-u" vs "Obriši VehicleContracts po ID-u")
                if description.strip() in self._duplicate_purposes:
                    entity_desc = self._build_entity_description(r.tool_id, r.method)
                    if entity_desc:
                        description = f"{entity_desc}. {description}"

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

            span.set_attribute("search.intent", intent_result.intent.value)
            span.set_attribute("search.result_count", len(final_results))

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
        query_type_result: QueryTypeResult,
        detected_entity: Optional[str] = None,
        effective_query_type: Optional[QueryType] = None,
        is_possessive: bool = False,
        qt_signal: ClassificationSignal = NO_SIGNAL,
    ) -> List[SearchResult]:
        """Apply ADDITIVE boosts to FAISS results (v4.0).

        Delegates to services.search.boost_engine.apply_boosts().
        """
        from services.tool_family_index import get_family_index

        # Resolve Tool Family Index match for boost engine
        family_match_tool = None
        if detected_entity and (effective_query_type or query_type_result.query_type) != QueryType.UNKNOWN:
            family_index = get_family_index()
            family_match_tool = family_index.resolve(
                detected_entity,
                effective_query_type or query_type_result.query_type,
            )

        ctx = BoostContext(
            tool_documentation=self._tool_documentation or {},
            tool_categories=self._tool_categories or {},
            primary_action_tools=self.PRIMARY_ACTION_TOOLS,
        )

        return _apply_boosts_fn(
            query, results, query_type_result, ctx,
            detected_entity=detected_entity,
            effective_query_type=effective_query_type,
            is_possessive=is_possessive,
            qt_signal=qt_signal,
            family_match_tool=family_match_tool,
        )

    def _build_entity_description(self, tool_id: str, method: str) -> str:
        """Build entity-specific description for tools with duplicate purposes."""
        verb = _METHOD_VERBS.get(method, "")
        parts = tool_id.split("_", 1)
        if len(parts) < 2:
            return ""
        entity = parts[1]
        suffix_desc = ""
        for suffix, desc in _SUFFIX_DESCRIPTIONS.items():
            if entity.lower().endswith(suffix.lstrip("_").lower()):
                suffix_desc = desc
                entity = entity[:len(entity) - len(suffix.lstrip("_"))]
                break
        if entity.endswith("_id"):
            suffix_desc = suffix_desc or " po ID-u"
            entity = entity[:-3]
        if entity.endswith("_"):
            entity = entity[:-1]
        return f"{verb} {entity}{suffix_desc}".strip()

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

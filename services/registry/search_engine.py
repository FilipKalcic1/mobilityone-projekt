"""
Search Engine - Semantic and keyword search for tool discovery.

Single responsibility: Find relevant tools using embeddings, categories, and scoring.
"""

import json
import logging
import os
from typing import Dict, List, Set, Tuple, Any, Optional

from config import get_settings
from services.tool_contracts import UnifiedToolDefinition, DependencyGraph
from services.scoring_utils import cosine_similarity
from services.tracing import get_tracer, trace_span
# ML-based intent detection (replaces regex patterns)
from services.intent_classifier import detect_action_intent, ActionIntent  # noqa: F401 - used by tests via patch
from services.registry.search_filters import (
    detect_intent as _detect_intent,
    detect_categories as _detect_categories,
    filter_by_method as _filter_by_method,
    filter_by_categories as _filter_by_categories,
    find_relevant_tools_filtered as _find_relevant_tools_filtered,
)
from services.registry.search_scorers import (
    DISAMBIGUATION_CONFIG,
    CATEGORY_CONFIG,
    apply_method_disambiguation,
    apply_user_specific_boosting,
    apply_learned_pattern_boosting,
    apply_evaluation_adjustment,
    apply_category_boosting,
    apply_documentation_boosting,
    apply_example_query_boosting,
)

logger = logging.getLogger(__name__)
_tracer = get_tracer("search_engine")
def _get_settings():
    return get_settings()


# Module-level cache for JSON files (loaded once, reused)
_json_file_cache: Dict[str, Optional[Dict]] = {}


def _load_json_file(filename: str) -> Optional[Dict]:
    """Load JSON file from config or data directory with caching."""
    # Return cached result if available
    if filename in _json_file_cache:
        return _json_file_cache[filename]

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    paths = [
        os.path.join(base_path, "config", filename),
        os.path.join(base_path, "data", filename),
    ]

    result = None
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    break
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    # Cache the result (even if None)
    _json_file_cache[filename] = result
    return result


class SearchEngine:
    """
    Handles tool discovery through semantic and keyword search.

    Responsibilities:
    - Generate query embeddings
    - Calculate similarity scores
    - Apply intent-based scoring adjustments
    - Handle fallback keyword search
    """

    # Configuration dicts imported from search_scorers module
    DISAMBIGUATION_CONFIG = DISAMBIGUATION_CONFIG
    CATEGORY_CONFIG = CATEGORY_CONFIG

    MAX_TOOLS_PER_RESPONSE = 20  # Increased from 12 to give LLM more options

    def __init__(self) -> None:
        """Initialize search engine with shared OpenAI client and category data."""
        from services.openai_client import get_embedding_client
        self.openai = get_embedding_client()

        # Load category data (categories are small, keep local cache)
        self._tool_categories = _load_json_file("tool_categories.json")
        # Tool documentation — shared cache (single source of truth)
        from services.schema_sanitizer import get_tool_documentation
        self._tool_documentation = get_tool_documentation()

        # Build reverse lookup: tool_id -> category
        self._tool_to_category: Dict[str, str] = {}
        self._category_keywords: Dict[str, Set[str]] = {}

        if self._tool_categories and "categories" in self._tool_categories:
            for cat_name, cat_data in self._tool_categories["categories"].items():
                # Map each tool to its category
                for tool_id in cat_data.get("tools", []):
                    self._tool_to_category[tool_id] = cat_name

                # Build category keyword sets
                keywords = set()
                keywords.update(cat_data.get("keywords_hr", []))
                keywords.update(cat_data.get("keywords_en", []))
                keywords.update(cat_data.get("typical_intents", []))
                self._category_keywords[cat_name] = {k.lower() for k in keywords}

            logger.info(f"Loaded {len(self._tool_to_category)} tool→category mappings")
        else:
            logger.warning("Tool categories not loaded - category boosting disabled")

        if self._tool_documentation:
            logger.info(f"Loaded documentation for {len(self._tool_documentation)} tools")

        logger.debug("SearchEngine initialized (v4.0 - documentation only, no training_queries)")

    async def find_relevant_tools_with_scores(
        self,
        query: str,
        tools: Dict[str, UnifiedToolDefinition],
        embeddings: Dict[str, List[float]],
        dependency_graph: Dict[str, DependencyGraph],
        retrieval_tools: Set[str],
        mutation_tools: Set[str],
        top_k: int = 5,
        threshold: float = 0.55,
        prefer_retrieval: bool = False,
        prefer_mutation: bool = False
    ) -> List[Dict[str, Any]]:
        """Find relevant tools with similarity scores. Returns list of dicts with name, score, schema."""
        with trace_span(_tracer, "search.find_tools", {
            "query.preview": query[:50],
            "top_k": top_k,
            "threshold": threshold,
            "tool_count": len(tools),
        }):
            return await self._find_relevant_inner(
                query, tools, embeddings, dependency_graph,
                retrieval_tools, mutation_tools, top_k, threshold,
                prefer_retrieval, prefer_mutation
            )

    async def _find_relevant_inner(
        self,
        query: str,
        tools: Dict[str, UnifiedToolDefinition],
        embeddings: Dict[str, List[float]],
        dependency_graph: Dict[str, DependencyGraph],
        retrieval_tools: Set[str],
        mutation_tools: Set[str],
        top_k: int,
        threshold: float,
        prefer_retrieval: bool,
        prefer_mutation: bool
    ) -> List[Dict[str, Any]]:
        """Inner implementation of find_relevant_tools_with_scores."""
        query_embedding = await self._get_query_embedding(query)

        if not query_embedding:
            fallback = self._fallback_keyword_search(query, tools, top_k)
            return [
                {"name": name, "score": 0.0, "schema": tools[name].to_openai_function()}
                for name in fallback
            ]

        # Determine search pool
        search_pool = set(tools.keys())
        if prefer_retrieval:
            search_pool = retrieval_tools
        elif prefer_mutation:
            search_pool = mutation_tools

        # Calculate similarity scores
        lenient_threshold = max(0.40, threshold - 0.20)
        scored = []

        for op_id in search_pool:
            if op_id not in embeddings:
                continue

            similarity = cosine_similarity(query_embedding, embeddings[op_id])

            if similarity >= lenient_threshold:
                scored.append((similarity, op_id))

        # Compute action intent once to avoid redundant ML inference calls
        action_intent = detect_action_intent(query)

        # Apply scoring adjustments (Rank Don't Filter architecture)
        scored = self._apply_method_disambiguation(query, scored, tools, action_intent=action_intent)
        scored = self._apply_user_specific_boosting(query, scored, tools)
        scored = self._apply_category_boosting(query, scored, tools)
        scored = self._apply_documentation_boosting(query, scored)
        scored = self._apply_example_query_boosting(query, scored)  # NEW: Match against example_queries_hr
        scored = self._apply_learned_pattern_boosting(query, scored)  # NEW: Feedback learning integration
        scored = self._apply_evaluation_adjustment(scored)

        scored.sort(key=lambda x: x[0], reverse=True)

        # INTENT-AWARE WILDCARD INJECTION
        # Ensure tools matching detected intent are always considered by LLM
        # This prevents training bias from excluding the correct HTTP method
        scored = self._inject_intent_matching_tools(query, scored, tools, search_pool, embeddings)

        # Expansion search if needed
        if len(scored) < top_k and len(scored) > 0:
            keyword_matches = self._description_keyword_search(query, search_pool, tools)
            for op_id, desc_score in keyword_matches:
                if op_id not in [s[1] for s in scored]:
                    scored.append((desc_score * 0.7, op_id))
            scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            fallback = self._fallback_keyword_search(query, tools, top_k)
            return [
                {"name": name, "score": 0.0, "schema": tools[name].to_openai_function()}
                for name in fallback
            ]

        # Apply dependency boosting
        base_tools = [op_id for _, op_id in scored[:top_k]]
        boosted_tools = self._apply_dependency_boosting(base_tools, dependency_graph)
        final_tools = boosted_tools[:self.MAX_TOOLS_PER_RESPONSE]

        # Build result with PILLAR 6: Origin Guide injection
        result = []
        scored_dict = {op_id: score for score, op_id in scored}

        for tool_id in final_tools:
            schema = tools[tool_id].to_openai_function()
            score = scored_dict.get(tool_id, 0.0)

            # PILLAR 6: Inject origin guide into result
            origin_guide = self._get_origin_guide(tool_id)

            result.append({
                "name": tool_id,
                "score": score,
                "schema": schema,
                "origin_guide": origin_guide  # NEW: Parameter origin info
            })

        logger.info(f"Returning {len(result)} tools with scores and origin guides")
        return result

    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get embedding for query text."""
        try:
            response = await self.openai.embeddings.create(
                input=[query[:8000]],
                model=_get_settings().AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Query embedding error: {e}")
            return None

    def detect_put_patch_ambiguity(
        self,
        scored: List[Tuple[float, str]],
        tools: Dict[str, UnifiedToolDefinition],
        score_threshold: float = 0.70,
        score_diff_threshold: float = 0.10
    ) -> Optional[Dict[str, Any]]:
        """Detect if both PUT and PATCH for the same resource are highly ranked.
        Returns dict with ambiguity info if detected, None otherwise."""
        # Get top tools that are PUT or PATCH
        put_tools = []
        patch_tools = []

        for score, op_id in scored[:10]:  # Check top 10
            if score < score_threshold:
                continue

            tool = tools.get(op_id)
            if not tool:
                continue

            if tool.method == "PUT" or "put_" in op_id.lower():
                put_tools.append((score, op_id))
            elif tool.method == "PATCH" or "patch_" in op_id.lower():
                patch_tools.append((score, op_id))

        if not put_tools or not patch_tools:
            return None

        # Check if they're for the same resource type
        # Extract resource name from tool ID (e.g., "put_Vehicles_id" -> "Vehicles")
        def extract_resource(op_id: str) -> str:
            parts = op_id.replace("put_", "").replace("patch_", "").split("_")
            return parts[0].lower() if parts else ""

        for put_score, put_id in put_tools:
            put_resource = extract_resource(put_id)
            for patch_score, patch_id in patch_tools:
                patch_resource = extract_resource(patch_id)

                # Same resource with similar scores = ambiguous
                if put_resource == patch_resource:
                    if abs(put_score - patch_score) <= score_diff_threshold:
                        logger.info(f"PUT/PATCH ambiguity detected: {put_id} vs {patch_id}")
                        return {
                            "ambiguous": True,
                            "resource": put_resource.title(),
                            "put_tool": put_id,
                            "put_score": put_score,
                            "patch_tool": patch_id,
                            "patch_score": patch_score,
                            "message": (
                                "Želite li potpuno ažuriranje (PUT - zamjenjuje sve podatke) "
                                "ili djelomično ažuriranje (PATCH - mijenja samo navedena polja)?"
                            )
                        }

        return None

    def _apply_method_disambiguation(
        self,
        query: str,
        scored: List[Tuple[float, str]],
        tools: Dict[str, UnifiedToolDefinition],
        action_intent=None
    ) -> List[Tuple[float, str]]:
        """Apply penalties/boosts based on detected intent using ML."""
        return apply_method_disambiguation(query, scored, tools, action_intent=action_intent)

    def _inject_intent_matching_tools(
        self,
        query: str,
        scored: List[Tuple[float, str]],
        tools: Dict[str, UnifiedToolDefinition],
        search_pool: set,
        embeddings: Dict[str, List[float]]
    ) -> List[Tuple[float, str]]:
        """INTENT-AWARE WILDCARD INJECTION: Ensures tools matching detected intent
        are ALWAYS in candidate list, even without training examples.
        The LLM (not embeddings) should make the final decision."""
        query_lower = query.lower().strip()
        scored_tools = {op_id for _, op_id in scored}

        # Detect DELETE intent
        delete_keywords = ["obriši", "obrisi", "delete", "ukloni", "makni", "izbriši", "izbrisi"]
        is_delete_intent = any(kw in query_lower for kw in delete_keywords)

        # Detect CREATE/POST intent
        create_keywords = ["dodaj", "kreiraj", "napravi", "novi", "nova", "prijavi", "unesi"]
        is_create_intent = any(kw in query_lower for kw in create_keywords)

        # Detect UPDATE intent
        update_keywords = ["ažuriraj", "azuriraj", "promijeni", "izmijeni", "update", "edit"]
        is_update_intent = any(kw in query_lower for kw in update_keywords)

        injected = list(scored)  # Copy
        injection_score = 0.50  # Minimum score to be considered
        max_injections = 5  # Don't flood with wildcards
        injections_count = 0

        for op_id in search_pool:
            if op_id in scored_tools:
                continue  # Already in results
            if injections_count >= max_injections:
                break

            tool = tools.get(op_id)
            if not tool:
                continue

            should_inject = False

            # DELETE intent → inject delete_ tools
            if is_delete_intent and (tool.method == "DELETE" or "delete" in op_id.lower()):
                should_inject = True

            # CREATE intent → inject post_ tools (but not search POSTs)
            elif is_create_intent and tool.method == "POST":
                is_search_post = any(x in op_id.lower() for x in ["search", "query", "filter", "find"])
                if not is_search_post:
                    should_inject = True

            # UPDATE intent → inject put_/patch_ tools
            elif is_update_intent and tool.method in ("PUT", "PATCH"):
                should_inject = True

            if should_inject:
                injected.append((injection_score, op_id))
                injections_count += 1
                logger.info(f"WILDCARD INJECT: {op_id} (intent match, no training)")

        # Re-sort after injection
        injected.sort(key=lambda x: x[0], reverse=True)
        return injected

    def _apply_user_specific_boosting(
        self,
        query: str,
        scored: List[Tuple[float, str]],
        tools: Dict[str, UnifiedToolDefinition]
    ) -> List[Tuple[float, str]]:
        """Boost tools that support user-specific filtering."""
        return apply_user_specific_boosting(query, scored, tools)

    def _apply_learned_pattern_boosting(
        self,
        query: str,
        scored: List[Tuple[float, str]]
    ) -> List[Tuple[float, str]]:
        """Apply learned boosts/penalties from feedback learning service."""
        return apply_learned_pattern_boosting(query, scored)

    def _apply_evaluation_adjustment(
        self,
        scored: List[Tuple[float, str]]
    ) -> List[Tuple[float, str]]:
        """Apply performance-based adjustments."""
        return apply_evaluation_adjustment(scored)

    def _apply_dependency_boosting(
        self,
        base_tools: List[str],
        dependency_graph: Dict[str, DependencyGraph]
    ) -> List[str]:
        """Add provider tools for dependency chaining."""
        result = list(base_tools)

        for tool_id in base_tools:
            dep_graph = dependency_graph.get(tool_id)
            if not dep_graph:
                continue

            for provider_id in dep_graph.provider_tools[:2]:
                if provider_id not in result:
                    result.append(provider_id)
                    logger.debug(f"Boosted {provider_id} for {tool_id}")

        return result

    def _description_keyword_search(
        self,
        query: str,
        search_pool: Set[str],
        tools: Dict[str, UnifiedToolDefinition],
        max_results: int = 5
    ) -> List[Tuple[str, float]]:
        """Search tool descriptions for keyword matches."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        matches = []

        for op_id in search_pool:
            tool = tools.get(op_id)
            if not tool:
                continue

            param_text = " ".join([
                p.name.lower() + " " + p.description.lower()
                for p in tool.parameters.values()
            ])
            output_text = " ".join(tool.output_keys).lower()

            searchable_text = " ".join([
                tool.description.lower(),
                tool.summary.lower(),
                tool.operation_id.lower(),
                " ".join(tool.tags).lower(),
                param_text,
                output_text
            ])

            score = 0.0
            searchable_words = set(searchable_text.split())
            word_overlap = len(query_words & searchable_words)
            score += word_overlap * 0.5

            for q_word in query_words:
                if len(q_word) >= 4 and q_word in searchable_text:
                    score += 0.3

            if score > 0:
                matches.append((op_id, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:max_results]

    def _fallback_keyword_search(
        self,
        query: str,
        tools: Dict[str, UnifiedToolDefinition],
        top_k: int
    ) -> List[str]:
        """Fallback keyword search when embeddings fail."""
        query_lower = query.lower()
        matches = []

        for op_id, tool in tools.items():
            text = f"{tool.description} {tool.path}".lower()
            score = sum(1 for word in query_lower.split() if word in text)

            if score > 0:
                matches.append((score, op_id))

        matches.sort(key=lambda x: x[0], reverse=True)
        return [op_id for _, op_id in matches[:top_k]]

    def _apply_category_boosting(
        self,
        query: str,
        scored: List[Tuple[float, str]],
        tools: Dict[str, UnifiedToolDefinition]
    ) -> List[Tuple[float, str]]:
        """Boost tools that belong to categories matching the query."""
        return apply_category_boosting(query, scored, tools, self._tool_to_category, self._category_keywords)

    def _apply_documentation_boosting(
        self,
        query: str,
        scored: List[Tuple[float, str]]
    ) -> List[Tuple[float, str]]:
        """Boost tools whose documentation matches the query."""
        return apply_documentation_boosting(query, scored, self._tool_documentation)

    def _apply_example_query_boosting(
        self,
        query: str,
        scored: List[Tuple[float, str]]
    ) -> List[Tuple[float, str]]:
        """Match user query against tool documentation example_queries_hr."""
        return apply_example_query_boosting(query, scored, self._tool_documentation)

    def get_tool_documentation(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get documentation for a specific tool."""
        if not self._tool_documentation:
            return None
        return self._tool_documentation.get(tool_id)

    def _get_origin_guide(self, tool_id: str) -> Dict[str, str]:
        """PILLAR 6: Get parameter origin guide for a tool.
        Returns dict mapping parameter names to their origins (CONTEXT/USER)."""
        if not self._tool_documentation:
            return {}

        doc = self._tool_documentation.get(tool_id, {})
        return doc.get("parameter_origin_guide", {})

    def get_tool_category(self, tool_id: str) -> Optional[str]:
        """Get category for a specific tool."""
        return self._tool_to_category.get(tool_id)

    def get_tools_in_category(self, category_name: str) -> List[str]:
        """Get all tools in a category."""
        if not self._tool_categories or "categories" not in self._tool_categories:
            return []
        cat_data = self._tool_categories["categories"].get(category_name, {})
        return cat_data.get("tools", [])

    # --- Filtering delegates (see search_filters.py for full docs) ---

    def detect_intent(self, query: str, action_intent=None) -> str:
        """Detect query intent: 'READ', 'WRITE', or 'UNKNOWN'."""
        return _detect_intent(query, action_intent=action_intent)

    def detect_categories(self, query: str) -> Set[str]:
        """Detect which categories the query is about."""
        return _detect_categories(query, self._category_keywords)

    def filter_by_method(self, tool_ids: Set[str], tools: Dict[str, UnifiedToolDefinition], intent: str) -> Set[str]:
        """Filter tools by HTTP method based on intent."""
        return _filter_by_method(tool_ids, tools, intent)

    def filter_by_categories(self, tool_ids: Set[str], categories: Set[str]) -> Set[str]:
        """Filter tools to only those in matching categories."""
        return _filter_by_categories(tool_ids, self._tool_to_category, categories)

    async def find_relevant_tools_filtered(
        self,
        query: str,
        tools: Dict[str, UnifiedToolDefinition],
        embeddings: Dict[str, List[float]],
        dependency_graph: Dict[str, DependencyGraph],
        retrieval_tools: Set[str],
        mutation_tools: Set[str],
        top_k: int = 5,
        threshold: float = 0.55
    ) -> List[Dict[str, Any]]:
        """
        Find relevant tools using FILTER-THEN-SEARCH approach.

        Delegates to search_filters.find_relevant_tools_filtered.
        """
        return await _find_relevant_tools_filtered(
            self, query, tools, embeddings, dependency_graph,
            retrieval_tools, mutation_tools, top_k, threshold
        )


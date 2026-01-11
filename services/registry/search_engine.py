"""
Search Engine - Semantic and keyword search for tool discovery.
Version: 2.0 (With Category-Based Search)

Single responsibility: Find relevant tools using embeddings, categories, and scoring.
"""

import json
import logging
import os
import re
from typing import Dict, List, Set, Tuple, Any, Optional

from openai import AsyncAzureOpenAI

from config import get_settings
from services.tool_contracts import UnifiedToolDefinition, DependencyGraph
from services.scoring_utils import cosine_similarity
from services.patterns import (
    READ_INTENT_PATTERNS,
    MUTATION_INTENT_PATTERNS,
    USER_SPECIFIC_PATTERNS,
    USER_FILTER_PARAMS
)

logger = logging.getLogger(__name__)
settings = get_settings()


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

    # Configuration for method disambiguation
    DISAMBIGUATION_CONFIG = {
        "read_intent_penalty_factor": 0.25,
        "unclear_intent_penalty_factor": 0.10,
        "delete_penalty_multiplier": 2.0,
        "update_penalty_multiplier": 1.0,
        "create_penalty_multiplier": 0.5,
        "get_boost_on_read_intent": 0.05,
    }

    MAX_TOOLS_PER_RESPONSE = 20  # Increased from 12 to give LLM more options

    # Category matching configuration
    # NOTE: Boost values adjusted for "Rank Don't Filter" architecture
    # All tools are scored, not filtered. Higher boosts help trained tools but
    # documentation matching allows untrained tools to be discovered too.
    CATEGORY_CONFIG = {
        "category_boost": 0.12,        # Boost for tools in matching category
        "keyword_match_boost": 0.08,   # Boost for keyword matches
        "documentation_boost": 0.20,   # INCREASED: Query matches tool documentation
        "training_match_boost": 0.20,  # INCREASED: Training examples matter
        "example_query_boost": 0.25,   # NEW: Match against example_queries_hr
    }

    def __init__(self):
        """Initialize search engine with OpenAI client and category data."""
        self.openai = AsyncAzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )

        # Load category and documentation data
        # v4.0: REMOVED training_queries.json (unreliable, only 55% coverage)
        # Now uses ONLY tool_documentation.json which has 100% tool coverage
        self._tool_categories = _load_json_file("tool_categories.json")
        self._tool_documentation = _load_json_file("tool_documentation.json")
        # DEPRECATED: self._training_queries - no longer used in v4.0

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

            logger.info(f"📂 Loaded {len(self._tool_to_category)} tool→category mappings")
        else:
            logger.warning("Tool categories not loaded - category boosting disabled")

        if self._tool_documentation:
            logger.info(f"📄 Loaded documentation for {len(self._tool_documentation)} tools")

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
        """
        Find relevant tools with similarity scores.

        Args:
            query: User query
            tools: Dict of all tools
            embeddings: Dict of embeddings by operation_id
            dependency_graph: Dependency graph for chaining
            retrieval_tools: Set of retrieval tool IDs
            mutation_tools: Set of mutation tool IDs
            top_k: Number of base tools to return
            threshold: Minimum similarity threshold
            prefer_retrieval: Only search retrieval tools
            prefer_mutation: Only search mutation tools

        Returns:
            List of dicts with name, score, and schema
        """
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

        # Apply scoring adjustments (Rank Don't Filter architecture)
        scored = self._apply_method_disambiguation(query, scored, tools)
        scored = self._apply_user_specific_boosting(query, scored, tools)
        scored = self._apply_category_boosting(query, scored, tools)
        scored = self._apply_documentation_boosting(query, scored)
        scored = self._apply_example_query_boosting(query, scored)  # NEW: Match against example_queries_hr
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

        logger.info(f"📦 Returning {len(result)} tools with scores and origin guides")
        return result

    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get embedding for query text."""
        try:
            response = await self.openai.embeddings.create(
                input=[query[:8000]],
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
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
        """
        Detect if both PUT and PATCH for the same resource are highly ranked.

        This helps the system ask users for clarification:
        - PUT = full update (replaces entire resource)
        - PATCH = partial update (modifies only specified fields)

        Args:
            scored: List of (score, tool_id) tuples
            tools: Tool definitions
            score_threshold: Minimum score to consider a tool "highly ranked"
            score_diff_threshold: Maximum score difference to consider ambiguous

        Returns:
            Dict with ambiguity info if detected, None otherwise
        """
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
                                f"Želite li potpuno ažuriranje (PUT - zamjenjuje sve podatke) "
                                f"ili djelomično ažuriranje (PATCH - mijenja samo navedena polja)?"
                            )
                        }

        return None

    def _apply_method_disambiguation(
        self,
        query: str,
        scored: List[Tuple[float, str]],
        tools: Dict[str, UnifiedToolDefinition]
    ) -> List[Tuple[float, str]]:
        """Apply penalties/boosts based on detected intent."""
        query_lower = query.lower().strip()

        # Use pre-compiled patterns for performance
        from services.patterns import is_read_intent as check_read, is_mutation_intent as check_mutation
        is_read_intent = check_read(query_lower)
        is_mutation_intent = check_mutation(query_lower)

        if is_mutation_intent:
            return scored

        config = self.DISAMBIGUATION_CONFIG
        penalty_factor = (
            config["read_intent_penalty_factor"] if is_read_intent
            else config["unclear_intent_penalty_factor"]
        )

        if is_read_intent:
            logger.info(f"🔍 Detected READ intent in: '{query}'")

        adjusted = []
        for score, op_id in scored:
            tool = tools.get(op_id)
            if not tool:
                adjusted.append((score, op_id))
                continue

            op_id_lower = op_id.lower()

            if tool.method == "DELETE" or "delete" in op_id_lower:
                penalty = penalty_factor * config["delete_penalty_multiplier"]
                new_score = max(0, score - penalty)
                adjusted.append((new_score, op_id))

            elif tool.method in ("PUT", "PATCH"):
                penalty = penalty_factor * config["update_penalty_multiplier"]
                new_score = max(0, score - penalty)
                adjusted.append((new_score, op_id))

            elif tool.method == "POST" and "get" not in op_id_lower:
                is_search_post = any(x in op_id_lower for x in ["search", "query", "find", "filter"])
                if is_search_post:
                    adjusted.append((score, op_id))
                else:
                    penalty = penalty_factor * config["create_penalty_multiplier"]
                    new_score = max(0, score - penalty)
                    adjusted.append((new_score, op_id))

            elif tool.method == "GET" and is_read_intent:
                boost = config["get_boost_on_read_intent"]
                adjusted.append((score + boost, op_id))

            else:
                adjusted.append((score, op_id))

        return adjusted

    def _inject_intent_matching_tools(
        self,
        query: str,
        scored: List[Tuple[float, str]],
        tools: Dict[str, UnifiedToolDefinition],
        search_pool: set,
        embeddings: Dict[str, List[float]]
    ) -> List[Tuple[float, str]]:
        """
        INTENT-AWARE WILDCARD INJECTION
        
        Ensures that tools matching detected intent are ALWAYS in candidate list,
        even if they don't have training examples. This prevents training bias
        from completely excluding correct tools.
        
        Example:
        - Query: "obriši vozilo" (DELETE intent)
        - Training only has examples for VehicleMgt_UpdateVehicle
        - Without this: delete_Vehicles_id might not reach LLM
        - With this: delete_Vehicles_id is injected with minimum score
        
        The LLM (not embeddings) should make the final decision!
        """
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
                logger.info(f"🎯 WILDCARD INJECT: {op_id} (intent match, no training)")
        
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
        query_lower = query.lower().strip()

        is_user_specific = any(
            re.search(pattern, query_lower)
            for pattern in USER_SPECIFIC_PATTERNS
        )

        if not is_user_specific:
            return scored

        logger.info(f"👤 Detected USER-SPECIFIC intent in: '{query}'")

        boost_value = 0.15
        penalty_value = 0.10
        masterdata_boost = 0.25
        calendar_penalty = 0.20

        adjusted = []
        for score, op_id in scored:
            tool = tools.get(op_id)
            if not tool:
                adjusted.append((score, op_id))
                continue

            tool_params_lower = {p.lower() for p in tool.parameters.keys()}
            has_user_filter = bool(tool_params_lower & USER_FILTER_PARAMS)
            op_id_lower = op_id.lower()

            if 'masterdata' in op_id_lower:
                new_score = score + masterdata_boost + boost_value
                adjusted.append((new_score, op_id))

            elif 'calendar' in op_id_lower or 'equipment' in op_id_lower:
                new_score = max(0, score - calendar_penalty)
                adjusted.append((new_score, op_id))

            elif has_user_filter:
                new_score = score + boost_value
                adjusted.append((new_score, op_id))

            elif tool.method == "GET" and 'vehicle' in op_id_lower:
                new_score = max(0, score - penalty_value)
                adjusted.append((new_score, op_id))

            else:
                adjusted.append((score, op_id))

        return adjusted

    def _apply_evaluation_adjustment(
        self,
        scored: List[Tuple[float, str]]
    ) -> List[Tuple[float, str]]:
        """Apply performance-based adjustments."""
        try:
            from services.tool_evaluator import get_tool_evaluator
            evaluator = get_tool_evaluator()

            adjusted = []
            for score, op_id in scored:
                new_score = evaluator.apply_evaluation_adjustment(op_id, score)
                adjusted.append((new_score, op_id))
            return adjusted
        except ImportError:
            return scored

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
                    logger.debug(f"🔗 Boosted {provider_id} for {tool_id}")

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
        if not self._tool_to_category or not self._category_keywords:
            return scored

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Find matching categories based on keywords
        matching_categories: Set[str] = set()

        for cat_name, keywords in self._category_keywords.items():
            # Check for word overlap
            if query_words & keywords:
                matching_categories.add(cat_name)
                continue

            # Check for substring matches
            for keyword in keywords:
                if len(keyword) >= 4 and keyword in query_lower:
                    matching_categories.add(cat_name)
                    break

        if matching_categories:
            logger.info(f"📂 Query matches categories: {matching_categories}")

        config = self.CATEGORY_CONFIG
        category_boost = config["category_boost"]
        keyword_boost = config["keyword_match_boost"]

        adjusted = []
        for score, op_id in scored:
            tool_category = self._tool_to_category.get(op_id)

            if tool_category and tool_category in matching_categories:
                new_score = score + category_boost
                adjusted.append((new_score, op_id))
            else:
                # Check for keyword matches in operation_id
                op_lower = op_id.lower()
                keyword_match = any(
                    word in op_lower
                    for word in query_words
                    if len(word) >= 4
                )
                if keyword_match:
                    adjusted.append((score + keyword_boost, op_id))
                else:
                    adjusted.append((score, op_id))

        return adjusted

    def _apply_documentation_boosting(
        self,
        query: str,
        scored: List[Tuple[float, str]]
    ) -> List[Tuple[float, str]]:
        """
        Boost tools whose documentation matches the query.

        v4.0: Uses ONLY tool_documentation.json (100% coverage).
        Removed training_queries.json dependency (was unreliable, 55% coverage).
        """
        if not self._tool_documentation:
            return scored

        query_lower = query.lower()
        query_words = set(query_lower.split())

        config = self.CATEGORY_CONFIG
        doc_boost = config["documentation_boost"]

        # v4.0: REMOVED training_queries matching - now uses only documentation
        # training_queries.json had only 55% tool coverage and inconsistent quality

        adjusted = []
        for score, op_id in scored:
            boost = 0.0

            # Check documentation matches (v4.0: primary method)
            doc = self._tool_documentation.get(op_id, {})

            # Match against example_queries
            example_queries = doc.get("example_queries", [])
            for example in example_queries:
                example_lower = example.lower()
                example_words = set(example_lower.split())
                if query_words & example_words:
                    boost += doc_boost * 0.5
                    break

            # Match against when_to_use
            when_to_use = doc.get("when_to_use", [])
            for use_case in when_to_use:
                use_case_lower = use_case.lower()
                if any(word in use_case_lower for word in query_words if len(word) >= 4):
                    boost += doc_boost * 0.3
                    break

            # Match against purpose
            purpose = doc.get("purpose", "").lower()
            if any(word in purpose for word in query_words if len(word) >= 4):
                boost += doc_boost * 0.2

            adjusted.append((score + boost, op_id))

        return adjusted

    def _apply_example_query_boosting(
        self,
        query: str,
        scored: List[Tuple[float, str]]
    ) -> List[Tuple[float, str]]:
        """
        Match user query against tool documentation example_queries_hr.

        This is critical for "Rank Don't Filter" architecture:
        - Allows untrained tools to be discovered through their documentation
        - Each tool has example_queries_hr in tool_documentation.json
        - Word overlap determines boost strength

        Args:
            query: User query
            scored: List of (score, tool_id) tuples

        Returns:
            Adjusted scored list with example query boosts applied
        """
        if not self._tool_documentation:
            return scored

        query_lower = query.lower()
        query_words = set(query_lower.split())

        config = self.CATEGORY_CONFIG
        example_boost = config.get("example_query_boost", 0.25)

        adjusted = []
        for score, op_id in scored:
            doc = self._tool_documentation.get(op_id, {})
            example_queries = doc.get("example_queries_hr", [])

            if not example_queries:
                adjusted.append((score, op_id))
                continue

            # Find best overlap with any example query
            best_overlap = 0
            for example in example_queries:
                example_words = set(example.lower().split())
                overlap = len(query_words & example_words)
                best_overlap = max(best_overlap, overlap)

            # Apply boost based on overlap strength
            if best_overlap >= 3:
                boost = example_boost  # Strong match
            elif best_overlap >= 2:
                boost = example_boost * 0.6  # Medium match
            elif best_overlap >= 1:
                boost = example_boost * 0.2  # Weak match
            else:
                boost = 0.0

            if boost > 0:
                logger.debug(f"📚 Example query boost +{boost:.2f} for {op_id} (overlap={best_overlap})")

            adjusted.append((score + boost, op_id))

        return adjusted

    def get_tool_documentation(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get documentation for a specific tool."""
        if not self._tool_documentation:
            return None
        return self._tool_documentation.get(tool_id)

    def _get_origin_guide(self, tool_id: str) -> Dict[str, str]:
        """
        PILLAR 6: Get parameter origin guide for a tool.

        Returns dict mapping parameter names to their origins:
        {
            "personId": "CONTEXT: Sustav automatski ubacuje",
            "vehicleId": "USER: Korisnik mora navesti",
            ...
        }
        """
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

    def _find_direct_training_matches(self, query: str) -> Set[str]:
        """
        DEPRECATED v4.0: No longer uses training_queries.json.

        This method used to match against training examples, but training_queries.json
        had only 55% tool coverage and inconsistent quality.

        Now uses tool_documentation.json via _apply_example_query_boosting() instead,
        which has 100% tool coverage and accurate example queries.

        Returns:
            Empty set (always)
        """
        # v4.0: DEPRECATED - training_queries.json removed
        # Use _apply_example_query_boosting() instead for example-based matching
        return set()

    # =========================================================================
    # NEW: FILTERING METHODS FOR OPTIMIZED SEARCH
    # =========================================================================

    def detect_intent(self, query: str) -> str:
        """
        Detect query intent: READ, WRITE, or UNKNOWN.

        Returns:
            'READ' - user wants information (GET)
            'WRITE' - user wants to create/update/delete (POST/PUT/DELETE)
            'UNKNOWN' - unclear intent
        """
        query_lower = query.lower().strip()

        # Check for mutation intent first (more specific)
        if any(re.search(pattern, query_lower) for pattern in MUTATION_INTENT_PATTERNS):
            return "WRITE"

        # Check for read intent
        if any(re.search(pattern, query_lower) for pattern in READ_INTENT_PATTERNS):
            return "READ"

        return "UNKNOWN"

    def detect_categories(self, query: str) -> Set[str]:
        """
        Detect which categories the query is about.

        Returns:
            Set of category names that match the query
        """
        if not self._category_keywords:
            return set()

        query_lower = query.lower()
        query_words = set(query_lower.split())
        matching_categories: Set[str] = set()

        for cat_name, keywords in self._category_keywords.items():
            # Check for word overlap
            if query_words & keywords:
                matching_categories.add(cat_name)
                continue

            # Check for substring matches (for longer keywords)
            for keyword in keywords:
                if len(keyword) >= 4 and keyword in query_lower:
                    matching_categories.add(cat_name)
                    break

        return matching_categories

    def filter_by_method(
        self,
        tool_ids: Set[str],
        tools: Dict[str, UnifiedToolDefinition],
        intent: str
    ) -> Set[str]:
        """
        Filter tools by HTTP method based on intent.

        Args:
            tool_ids: Set of tool IDs to filter
            tools: Dict of all tools
            intent: 'READ', 'WRITE', or 'UNKNOWN'

        Returns:
            Filtered set of tool IDs
        """
        if intent == "UNKNOWN":
            return tool_ids  # No filtering

        filtered = set()

        for tool_id in tool_ids:
            tool = tools.get(tool_id)
            if not tool:
                continue

            if intent == "READ":
                # For READ intent, prefer GET methods
                # But also include POST methods that are actually searches
                if tool.method == "GET":
                    filtered.add(tool_id)
                elif tool.method == "POST":
                    # Some POST methods are actually search/query operations
                    op_lower = tool_id.lower()
                    if any(x in op_lower for x in ["search", "query", "find", "filter", "list"]):
                        filtered.add(tool_id)

            elif intent == "WRITE":
                # For WRITE intent, prefer POST/PUT/PATCH/DELETE
                if tool.method in ("POST", "PUT", "PATCH", "DELETE"):
                    filtered.add(tool_id)

        logger.info(f"🔍 Method filter ({intent}): {len(tool_ids)} → {len(filtered)} tools")
        return filtered if filtered else tool_ids  # Fallback to all if nothing matches

    def filter_by_categories(
        self,
        tool_ids: Set[str],
        categories: Set[str]
    ) -> Set[str]:
        """
        Filter tools to only those in matching categories.

        Args:
            tool_ids: Set of tool IDs to filter
            categories: Set of category names to match

        Returns:
            Filtered set of tool IDs
        """
        if not categories or not self._tool_to_category:
            return tool_ids  # No filtering if no categories detected

        filtered = set()

        for tool_id in tool_ids:
            tool_category = self._tool_to_category.get(tool_id)
            if tool_category and tool_category in categories:
                filtered.add(tool_id)

        logger.info(f"📂 Category filter ({categories}): {len(tool_ids)} → {len(filtered)} tools")
        return filtered if filtered else tool_ids  # Fallback to all if nothing matches

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

        This method:
        1. Detects intent (READ vs WRITE)
        2. Filters by HTTP method
        3. Detects categories
        4. Filters by categories
        5. Runs semantic search on filtered set
        6. Applies boosts and returns results

        This dramatically reduces search space for better accuracy.
        """
        # Step 1: Detect intent
        intent = self.detect_intent(query)
        logger.info(f"🎯 Detected intent: {intent}")

        # Step 2: Start with all tools
        search_pool = set(tools.keys())
        original_size = len(search_pool)

        # Step 3: Filter by method based on intent
        if intent != "UNKNOWN":
            search_pool = self.filter_by_method(search_pool, tools, intent)

        # Step 4: Detect categories
        categories = self.detect_categories(query)
        if categories:
            logger.info(f"📂 Detected categories: {categories}")

        # Step 5: Filter by categories (only if we have matches)
        if categories:
            search_pool = self.filter_by_categories(search_pool, categories)

        logger.info(f"🔎 Search space: {original_size} → {len(search_pool)} tools")

        # Step 6: Run semantic search on filtered set
        query_embedding = await self._get_query_embedding(query)

        if not query_embedding:
            # Fallback to keyword search
            fallback = self._fallback_keyword_search(query, tools, top_k)
            return [
                {"name": name, "score": 0.0, "schema": tools[name].to_openai_function()}
                for name in fallback if name in search_pool
            ]

        # Calculate similarity scores on filtered pool
        lenient_threshold = max(0.40, threshold - 0.20)
        scored = []

        for op_id in search_pool:
            if op_id not in embeddings:
                continue

            similarity = cosine_similarity(query_embedding, embeddings[op_id])

            if similarity >= lenient_threshold:
                scored.append((similarity, op_id))

        # Apply scoring adjustments (Rank Don't Filter architecture)
        scored = self._apply_method_disambiguation(query, scored, tools)
        scored = self._apply_user_specific_boosting(query, scored, tools)
        scored = self._apply_category_boosting(query, scored, tools)
        scored = self._apply_documentation_boosting(query, scored)
        scored = self._apply_example_query_boosting(query, scored)  # NEW: Match against example_queries_hr
        scored = self._apply_evaluation_adjustment(scored)

        # CRITICAL: Find DIRECT training matches and boost them to TOP
        training_matched = self._find_direct_training_matches(query)
        if training_matched:
            logger.info(f"🎯 Direct training matches: {training_matched}")
            # Ensure training-matched tools WIN - use score 2.0 to beat any other tool
            scored_tools = {op_id for _, op_id in scored}
            for tool_id in training_matched:
                if tool_id in tools:
                    if tool_id in scored_tools:
                        # Set to highest score to ensure it wins
                        scored = [(2.0, t) if t == tool_id else (s, t) for s, t in scored]
                    else:
                        # Add with highest score if not present
                        scored.append((2.0, tool_id))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Expansion if needed
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
            if tool_id not in tools:
                continue
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

        logger.info(f"📦 Returning {len(result)} tools (filtered search with origin guides)")
        return result

    # =========================================================================
    # NEW: FAISS-BASED SEARCH WITH ACTION INTENT GATE
    # =========================================================================

    async def find_relevant_tools_faiss(
        self,
        query: str,
        tools: Dict[str, UnifiedToolDefinition],
        dependency_graph: Dict[str, DependencyGraph],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find relevant tools using FAISS + ACTION INTENT GATE.

        NEW ARCHITECTURE:
        1. ACTION INTENT GATE - Detect GET/POST/PUT/DELETE BEFORE search
        2. FAISS semantic search - Fast in-memory vector search
        3. Category boosting - Boost tools in matching categories
        4. Documentation boosting - Boost based on tool docs (NOT training_queries!)

        IMPORTANT: Does NOT use training_queries.json (UNRELIABLE).
        Uses tool_documentation.json (ACCURATE) for embeddings.

        Args:
            query: User query text
            tools: Dict of all tools
            dependency_graph: Dependency graph for chaining
            top_k: Number of tools to return

        Returns:
            List of dicts with name, score, schema, origin_guide
        """
        from services.faiss_vector_store import get_faiss_store
        from services.action_intent_detector import (
            detect_action_intent,
            filter_tools_by_intent,
            ActionIntent
        )

        logger.info(f"🔍 FAISS search for: '{query[:50]}...'")

        # =================================================================
        # STEP 0: ACTION INTENT GATE (CRITICAL - must come FIRST!)
        # =================================================================
        intent_result = detect_action_intent(query)
        logger.info(
            f"🎯 ACTION INTENT: {intent_result.intent.value} "
            f"(confidence={intent_result.confidence:.2f}, reason={intent_result.reason})"
        )

        # Build tool_methods mapping
        tool_methods = {
            tool_id: getattr(tool, 'method', 'GET')
            for tool_id, tool in tools.items()
        }

        # Filter tools by intent BEFORE search
        all_tool_ids = set(tools.keys())
        filtered_tool_ids = filter_tools_by_intent(
            all_tool_ids,
            tool_methods,
            intent_result.intent
        )

        logger.info(
            f"📊 Intent filter: {len(all_tool_ids)} → {len(filtered_tool_ids)} tools"
        )

        # =================================================================
        # STEP 1: FAISS SEMANTIC SEARCH
        # =================================================================
        faiss_store = get_faiss_store()

        if not faiss_store.is_initialized():
            logger.warning("FAISS store not initialized, falling back to legacy search")
            return await self.find_relevant_tools_filtered(
                query, tools, {}, dependency_graph, set(), set(), top_k
            )

        # Map action intent to FAISS filter
        action_filter = None
        if intent_result.intent == ActionIntent.READ:
            action_filter = "READ"
        elif intent_result.intent == ActionIntent.CREATE:
            action_filter = "CREATE"
        elif intent_result.intent == ActionIntent.UPDATE:
            action_filter = "UPDATE"
        elif intent_result.intent == ActionIntent.DELETE:
            action_filter = "DELETE"

        # FAISS search with action filter
        faiss_results = await faiss_store.search(
            query,
            top_k=top_k * 2,  # Get more results, then filter
            action_filter=action_filter
        )

        if not faiss_results:
            logger.warning("FAISS returned no results, falling back to keyword search")
            fallback = self._fallback_keyword_search(query, tools, top_k)
            return [
                {"name": name, "score": 0.0, "schema": tools[name].to_openai_function()}
                for name in fallback if name in filtered_tool_ids
            ]

        # =================================================================
        # STEP 2: APPLY BOOSTS (without training_queries!)
        # =================================================================
        scored = [(result.score, result.tool_id) for result in faiss_results]

        # Apply category boosting (from tool_categories.json)
        scored = self._apply_category_boosting(query, scored, tools)

        # Apply documentation boosting (from tool_documentation.json)
        # NOTE: This does NOT use training_queries.json!
        scored = self._apply_documentation_boosting_no_training(query, scored)

        # Apply example query boosting (from tool_documentation.json)
        scored = self._apply_example_query_boosting(query, scored)

        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # =================================================================
        # STEP 3: FILTER BY INTENT (second pass)
        # =================================================================
        # Ensure results match detected intent
        final_scored = []
        for score, tool_id in scored:
            if tool_id in filtered_tool_ids:
                final_scored.append((score, tool_id))

        # If nothing matches, use all scored results
        if not final_scored:
            final_scored = scored

        # =================================================================
        # STEP 4: APPLY DEPENDENCY BOOSTING
        # =================================================================
        base_tools = [op_id for _, op_id in final_scored[:top_k]]
        boosted_tools = self._apply_dependency_boosting(base_tools, dependency_graph)
        final_tools = boosted_tools[:self.MAX_TOOLS_PER_RESPONSE]

        # =================================================================
        # STEP 5: BUILD RESULT
        # =================================================================
        result = []
        scored_dict = {op_id: score for score, op_id in final_scored}

        for tool_id in final_tools:
            if tool_id not in tools:
                continue

            schema = tools[tool_id].to_openai_function()
            score = scored_dict.get(tool_id, 0.0)
            origin_guide = self._get_origin_guide(tool_id)

            result.append({
                "name": tool_id,
                "score": score,
                "schema": schema,
                "origin_guide": origin_guide
            })

        logger.info(
            f"📦 FAISS returned {len(result)} tools "
            f"(intent={intent_result.intent.value})"
        )
        return result

    def _apply_documentation_boosting_no_training(
        self,
        query: str,
        scored: List[Tuple[float, str]]
    ) -> List[Tuple[float, str]]:
        """
        Boost tools based on documentation matches.

        IMPORTANT: This version does NOT use training_queries.json!
        Uses only tool_documentation.json (ACCURATE source).
        """
        if not self._tool_documentation:
            return scored

        query_lower = query.lower()
        query_words = set(query_lower.split())

        config = self.CATEGORY_CONFIG
        doc_boost = config["documentation_boost"]

        adjusted = []
        for score, op_id in scored:
            boost = 0.0
            doc = self._tool_documentation.get(op_id, {})

            # Match against example_queries
            example_queries = doc.get("example_queries", [])
            for example in example_queries:
                example_lower = example.lower()
                example_words = set(example_lower.split())
                if query_words & example_words:
                    boost += doc_boost * 0.5
                    break

            # Match against when_to_use
            when_to_use = doc.get("when_to_use", [])
            for use_case in when_to_use:
                use_case_lower = use_case.lower()
                if any(word in use_case_lower for word in query_words if len(word) >= 4):
                    boost += doc_boost * 0.3
                    break

            # Match against purpose
            purpose = doc.get("purpose", "").lower()
            if any(word in purpose for word in query_words if len(word) >= 4):
                boost += doc_boost * 0.2

            adjusted.append((score + boost, op_id))

        return adjusted

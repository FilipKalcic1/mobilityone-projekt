"""
Search Filters - Standalone filtering functions for optimized search.

Extracted from search_engine.py. These functions filter tool sets by intent,
HTTP method, and categories.
"""

import logging
from typing import Dict, Set, List, Tuple, Any, Optional, TYPE_CHECKING

from services.tool_contracts import UnifiedToolDefinition, DependencyGraph
from services.scoring_utils import cosine_similarity
from services.intent_classifier import detect_action_intent, ActionIntent

if TYPE_CHECKING:
    from services.registry.search_engine import SearchEngine

logger = logging.getLogger(__name__)


def detect_intent(query: str, action_intent=None) -> str:
    """
    Detect query intent using ML classifier.

    Returns:
        'READ' - user wants information (GET)
        'WRITE' - user wants to create/update/delete (POST/PUT/DELETE)
        'UNKNOWN' - unclear intent
    """
    # ML-based intent detection (replaces 100+ regex patterns)
    if action_intent is None:
        intent_result = detect_action_intent(query)
    else:
        intent_result = action_intent

    if intent_result.intent == ActionIntent.READ:
        return "READ"
    elif intent_result.intent in (
        ActionIntent.CREATE, ActionIntent.UPDATE,
        ActionIntent.PATCH, ActionIntent.DELETE
    ):
        return "WRITE"

    return "UNKNOWN"


def detect_categories(query: str, category_keywords: Dict) -> Set[str]:
    """
    Detect which categories the query is about.

    Args:
        query: The search query
        category_keywords: Dict mapping category names to sets of keywords

    Returns:
        Set of category names that match the query
    """
    if not category_keywords:
        return set()

    query_lower = query.lower()
    query_words = set(query_lower.split())
    matching_categories: Set[str] = set()

    for cat_name, keywords in category_keywords.items():
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

    logger.info(f"Method filter ({intent}): {len(tool_ids)} → {len(filtered)} tools")
    return filtered if filtered else tool_ids  # Fallback to all if nothing matches


def filter_by_categories(
    tool_ids: Set[str],
    tool_to_category: Dict,
    categories: Set[str]
) -> Set[str]:
    """
    Filter tools to only those in matching categories.

    Args:
        tool_ids: Set of tool IDs to filter
        tool_to_category: Dict mapping tool IDs to their category
        categories: Set of category names to match

    Returns:
        Filtered set of tool IDs
    """
    if not categories or not tool_to_category:
        return tool_ids  # No filtering if no categories detected

    filtered = set()

    for tool_id in tool_ids:
        tool_category = tool_to_category.get(tool_id)
        if tool_category and tool_category in categories:
            filtered.add(tool_id)

    logger.info(f"Category filter ({categories}): {len(tool_ids)} → {len(filtered)} tools")
    return filtered if filtered else tool_ids  # Fallback to all if nothing matches


async def find_relevant_tools_filtered(
    engine: "SearchEngine",
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

    This function:
    1. Detects intent (READ vs WRITE)
    2. Filters by HTTP method
    3. Detects categories
    4. Filters by categories
    5. Runs semantic search on filtered set
    6. Applies boosts and returns results

    This dramatically reduces search space for better accuracy.

    Args:
        engine: SearchEngine instance (provides embedding, scoring, and lookup methods)
        query: User query
        tools: Dict of all tools
        embeddings: Dict of embeddings by operation_id
        dependency_graph: Dependency graph for chaining
        retrieval_tools: Set of retrieval tool IDs
        mutation_tools: Set of mutation tool IDs
        top_k: Number of base tools to return
        threshold: Minimum similarity threshold

    Returns:
        List of dicts with name, score, schema, and origin_guide
    """
    # Step 1: Detect intent (compute once, pass through to avoid redundant ML calls)
    action_intent = detect_action_intent(query)
    intent = detect_intent(query, action_intent=action_intent)
    logger.info(f"Detected intent: {intent}")

    # Step 2: Start with all tools
    search_pool = set(tools.keys())
    original_size = len(search_pool)

    # Step 3: Filter by method based on intent
    if intent != "UNKNOWN":
        search_pool = filter_by_method(search_pool, tools, intent)

    # Step 4: Detect categories
    categories = detect_categories(query, engine._category_keywords)
    if categories:
        logger.info(f"Detected categories: {categories}")

    # Step 5: Filter by categories (only if we have matches)
    if categories:
        search_pool = filter_by_categories(search_pool, engine._tool_to_category, categories)

    logger.info(f"Search space: {original_size} → {len(search_pool)} tools")

    # Step 6: Run semantic search on filtered set
    query_embedding = await engine._get_query_embedding(query)

    if not query_embedding:
        # Fallback to keyword search
        fallback = engine._fallback_keyword_search(query, tools, top_k)
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
    scored = engine._apply_method_disambiguation(query, scored, tools, action_intent=action_intent)
    scored = engine._apply_user_specific_boosting(query, scored, tools)
    scored = engine._apply_category_boosting(query, scored, tools)
    scored = engine._apply_documentation_boosting(query, scored)
    scored = engine._apply_example_query_boosting(query, scored)
    scored = engine._apply_learned_pattern_boosting(query, scored)  # Feedback learning integration
    scored = engine._apply_evaluation_adjustment(scored)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Expansion if needed
    if len(scored) < top_k and len(scored) > 0:
        keyword_matches = engine._description_keyword_search(query, search_pool, tools)
        scored_ids = {s[1] for s in scored}
        for op_id, desc_score in keyword_matches:
            if op_id not in scored_ids:
                scored.append((desc_score * 0.7, op_id))
                scored_ids.add(op_id)
        scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        fallback = engine._fallback_keyword_search(query, tools, top_k)
        return [
            {"name": name, "score": 0.0, "schema": tools[name].to_openai_function()}
            for name in fallback
        ]

    # Apply dependency boosting
    base_tools = [op_id for _, op_id in scored[:top_k]]
    boosted_tools = engine._apply_dependency_boosting(base_tools, dependency_graph)
    final_tools = boosted_tools[:engine.MAX_TOOLS_PER_RESPONSE]

    # Build result with PILLAR 6: Origin Guide injection
    result = []
    scored_dict = {op_id: score for score, op_id in scored}

    for tool_id in final_tools:
        if tool_id not in tools:
            continue
        schema = tools[tool_id].to_openai_function()
        score = scored_dict.get(tool_id, 0.0)

        # PILLAR 6: Inject origin guide into result
        origin_guide = engine._get_origin_guide(tool_id)

        result.append({
            "name": tool_id,
            "score": score,
            "schema": schema,
            "origin_guide": origin_guide  # NEW: Parameter origin info
        })

    logger.info(f"Returning {len(result)} tools (filtered search with origin guides)")
    return result

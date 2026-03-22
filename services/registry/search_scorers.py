"""
Search Scorers - Scoring and boosting functions for tool search.

Extracted from SearchEngine to keep search_engine.py focused on orchestration.
These are standalone functions that accept the data they need as parameters.
"""

import logging
import re
from typing import Dict, List, Set, Tuple, Any, Optional

from services.tool_contracts import UnifiedToolDefinition
from services.intent_classifier import detect_action_intent, ActionIntent
from services.patterns import USER_SPECIFIC_PATTERNS, USER_FILTER_PARAMS

logger = logging.getLogger(__name__)

# Cache for learned boosts file (avoid re-reading from disk on every search)
_learned_boosts_cache: Dict[str, Any] = {"mtime": 0.0, "data": None}


# Configuration for method disambiguation
DISAMBIGUATION_CONFIG = {
    "read_intent_penalty_factor": 0.25,
    "unclear_intent_penalty_factor": 0.10,
    "delete_penalty_multiplier": 2.0,
    "update_penalty_multiplier": 1.0,
    "create_penalty_multiplier": 0.5,
    "get_boost_on_read_intent": 0.05,
}

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


def apply_method_disambiguation(
    query: str,
    scored: List[Tuple[float, str]],
    tools: Dict[str, UnifiedToolDefinition],
    action_intent=None
) -> List[Tuple[float, str]]:
    """Apply penalties/boosts based on detected intent using ML."""
    # Use ML-based intent detection (replaces 100+ regex patterns)
    if action_intent is None:
        action_intent = detect_action_intent(query)
    intent_result = action_intent
    is_read_intent = intent_result.intent == ActionIntent.READ
    is_mutation_intent = intent_result.intent in (
        ActionIntent.CREATE, ActionIntent.UPDATE,
        ActionIntent.PATCH, ActionIntent.DELETE
    )

    if is_mutation_intent:
        return scored

    config = DISAMBIGUATION_CONFIG
    penalty_factor = (
        config["read_intent_penalty_factor"] if is_read_intent
        else config["unclear_intent_penalty_factor"]
    )

    if is_read_intent:
        logger.info(f"Detected READ intent in: '{query}'")

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


def apply_user_specific_boosting(
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

    logger.info(f"Detected USER-SPECIFIC intent in: '{query}'")

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


def apply_learned_pattern_boosting(
    query: str,
    scored: List[Tuple[float, str]]
) -> List[Tuple[float, str]]:
    """
    Apply learned boosts/penalties from feedback learning service.

    This method integrates the feedback loop by:
    1. Loading learned patterns from FeedbackLearningService
    2. Boosting tools that match positive patterns
    3. Penalizing tools that match negative patterns

    The boost values are confidence-weighted (0.10 to 0.28 typical).
    """
    try:
        from services.feedback_learning_service import LEARNED_BOOSTS_FILE
        import json

        # Load learned boosts from cache file (with mtime-based cache)
        if not LEARNED_BOOSTS_FILE.exists():
            return scored

        file_mtime = LEARNED_BOOSTS_FILE.stat().st_mtime
        if _learned_boosts_cache["mtime"] == file_mtime and _learned_boosts_cache["data"] is not None:
            boosts_data = _learned_boosts_cache["data"]
        else:
            with open(LEARNED_BOOSTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                boosts_data = data.get("boosts", [])
            _learned_boosts_cache["mtime"] = file_mtime
            _learned_boosts_cache["data"] = boosts_data

        if not boosts_data:
            return scored

        # Build lookup maps
        tool_boosts: Dict[str, Dict] = {}
        negative_tool_patterns: Dict[str, List[str]] = {}

        for boost in boosts_data:
            tool_id = boost.get("tool_id", "").lower()
            tool_boosts[tool_id] = boost

            # Build negative pattern map
            for neg_tool in boost.get("negative_patterns", []):
                neg_tool_lower = neg_tool.lower()
                if neg_tool_lower not in negative_tool_patterns:
                    negative_tool_patterns[neg_tool_lower] = []
                negative_tool_patterns[neg_tool_lower].extend(boost.get("patterns", []))

        query_lower = query.lower()
        adjusted = []

        for score, op_id in scored:
            op_id_lower = op_id.lower()
            adjustment = 0.0
            reason = ""

            # Check positive boost (tool was correct for these patterns)
            if op_id_lower in tool_boosts:
                boost = tool_boosts[op_id_lower]
                for pattern in boost.get("patterns", []):
                    if pattern in query_lower:
                        adjustment = boost.get("boost_value", 0.15)
                        reason = f"learned_boost:{pattern}"
                        logger.debug(f"Learned boost +{adjustment:.2f} for {op_id} ({reason})")
                        break

            # Check negative penalty (tool was wrong for these patterns)
            if op_id_lower in negative_tool_patterns and adjustment == 0.0:
                for pattern in negative_tool_patterns[op_id_lower]:
                    if pattern in query_lower:
                        adjustment = -0.15  # Penalty
                        reason = f"learned_penalty:{pattern}"
                        logger.debug(f"Learned penalty {adjustment:.2f} for {op_id} ({reason})")
                        break

            new_score = max(0.0, score + adjustment)
            adjusted.append((new_score, op_id))

        return adjusted

    except Exception as e:
        logger.warning(f"Could not apply learned pattern boosting: {e}")
        return scored


def apply_evaluation_adjustment(
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


def apply_category_boosting(
    query: str,
    scored: List[Tuple[float, str]],
    tools: Dict[str, UnifiedToolDefinition],
    tool_to_category: Dict[str, str],
    category_keywords: Dict[str, Set[str]]
) -> List[Tuple[float, str]]:
    """Boost tools that belong to categories matching the query."""
    if not tool_to_category or not category_keywords:
        return scored

    query_lower = query.lower()
    query_words = set(query_lower.split())

    # Find matching categories based on keywords
    matching_categories: Set[str] = set()

    for cat_name, keywords in category_keywords.items():
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
        logger.info(f"Query matches categories: {matching_categories}")

    config = CATEGORY_CONFIG
    category_boost = config["category_boost"]
    keyword_boost = config["keyword_match_boost"]

    adjusted = []
    for score, op_id in scored:
        tool_category = tool_to_category.get(op_id)

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


def apply_documentation_boosting(
    query: str,
    scored: List[Tuple[float, str]],
    tool_documentation: Optional[Dict]
) -> List[Tuple[float, str]]:
    """
    Boost tools whose documentation matches the query.

    Uses ONLY tool_documentation.json (100% coverage).
    Removed training_queries.json dependency (was unreliable, 55% coverage).
    """
    if not tool_documentation:
        return scored

    query_lower = query.lower()
    query_words = set(query_lower.split())

    config = CATEGORY_CONFIG
    doc_boost = config["documentation_boost"]

    adjusted = []
    for score, op_id in scored:
        boost = 0.0

        # Check documentation matches (v4.0: primary method)
        doc = tool_documentation.get(op_id, {})

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


def apply_example_query_boosting(
    query: str,
    scored: List[Tuple[float, str]],
    tool_documentation: Optional[Dict]
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
        tool_documentation: Tool documentation dict

    Returns:
        Adjusted scored list with example query boosts applied
    """
    if not tool_documentation:
        return scored

    query_lower = query.lower()
    query_words = set(query_lower.split())

    config = CATEGORY_CONFIG
    example_boost = config.get("example_query_boost", 0.25)

    adjusted = []
    for score, op_id in scored:
        doc = tool_documentation.get(op_id, {})
        example_queries = doc.get("example_queries_hr", [])

        if not example_queries:
            adjusted.append((score, op_id))
            continue

        # Find best overlap with any example query
        best_overlap = 0
        for example in example_queries:
            if not isinstance(example, str):
                continue
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
            logger.debug(f"Example query boost +{boost:.2f} for {op_id} (overlap={best_overlap})")

        adjusted.append((score + boost, op_id))

    return adjusted

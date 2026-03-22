"""
Action Intent Detection — HTTP method inference from Croatian user queries.

Extracted from intent_classifier.py. Determines whether a user query
implies GET (read), POST (create), PUT (update), or DELETE operations.

Uses ML classifier (TF-IDF ensemble) for intent detection with
ClassificationSignal for confidence assessment.

Usage:
    from services.action_intent_detector import (
        ActionIntent, detect_action_intent, get_allowed_methods,
        filter_tools_by_intent, IntentDetectionResult,
    )

    result = detect_action_intent("obriši to vozilo")
    # result.intent == ActionIntent.DELETE
    # result.confidence == 0.95
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from services.dynamic_threshold import (
    ClassificationSignal,
    NO_SIGNAL,
    PredictionSet,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ActionIntent enum
# ---------------------------------------------------------------------------

class ActionIntent(str, Enum):
    """HTTP action intent detected from user query."""
    READ = "GET"
    CREATE = "POST"
    UPDATE = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"  # For greetings, help, etc.


# ---------------------------------------------------------------------------
# Allowed methods per intent (frozen for safety)
# ---------------------------------------------------------------------------

_INTENT_TO_METHODS: Dict[ActionIntent, FrozenSet[str]] = {
    ActionIntent.READ: frozenset({"GET"}),
    ActionIntent.CREATE: frozenset({"POST"}),
    ActionIntent.UPDATE: frozenset({"PUT", "PATCH"}),
    ActionIntent.PATCH: frozenset({"PATCH"}),
    ActionIntent.DELETE: frozenset({"DELETE"}),
}

_ALL_METHODS: FrozenSet[str] = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})


def get_allowed_methods(intent: ActionIntent) -> Set[str]:
    """Get allowed HTTP methods for an action intent.

    Returns a mutable set for backward compatibility with callers
    that may modify the return value.
    """
    return set(_INTENT_TO_METHODS.get(intent, _ALL_METHODS))


# ---------------------------------------------------------------------------
# Intent detection result
# ---------------------------------------------------------------------------

@dataclass
class IntentDetectionResult:
    """Result of intent detection — compatible with old interface."""
    intent: ActionIntent
    confidence: float
    matched_pattern: Optional[str] = None
    reason: str = ""
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    signal: ClassificationSignal = None  # type: ignore[assignment]
    prediction_set: Optional[PredictionSet] = None

    def __post_init__(self) -> None:
        if self.signal is None:
            self.signal = ClassificationSignal.from_alternatives(
                self.confidence, self.alternatives, n_classes=7,
            )


# ---------------------------------------------------------------------------
# Action → intent mapping
# ---------------------------------------------------------------------------

_ACTION_TO_INTENT: Dict[str, ActionIntent] = {
    "GET": ActionIntent.READ,
    "POST": ActionIntent.CREATE,
    "PUT": ActionIntent.UPDATE,
    "PATCH": ActionIntent.PATCH,
    "DELETE": ActionIntent.DELETE,
    "NONE": ActionIntent.NONE,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_action_intent(
    query: str,
    use_ensemble: bool = True,
) -> IntentDetectionResult:
    """Detect action intent using ML classifier.

    Uses smart ensemble: TF-IDF first, semantic fallback if confidence < 85%.

    Args:
        query: User query text (Croatian or English).
        use_ensemble: If True, use TF-IDF → semantic cascade.

    Returns:
        IntentDetectionResult with intent, confidence, signal, and
        optional conformal prediction set.
    """
    # Lazy import to break circular dependency
    # (intent_classifier uses ActionIntent, but detect_action_intent
    #  needs predict_with_ensemble from intent_classifier)
    from services.intent_classifier import (
        get_intent_classifier,
        predict_with_ensemble,
    )

    if use_ensemble:
        prediction = predict_with_ensemble(query)
    else:
        classifier = get_intent_classifier()
        prediction = classifier.predict(query)

    action_intent = _ACTION_TO_INTENT.get(prediction.action, ActionIntent.UNKNOWN)

    return IntentDetectionResult(
        intent=action_intent,
        confidence=prediction.confidence,
        matched_pattern=f"ML:{prediction.intent}",
        reason=(
            f"ML classifier predicted {prediction.intent} "
            f"with {prediction.confidence:.2%} confidence"
        ),
        alternatives=prediction.alternatives,
        signal=prediction.signal,
        prediction_set=prediction.prediction_set,
    )


def filter_tools_by_intent(
    tools: List[Dict[str, Any]],
    intent: ActionIntent,
) -> List[Dict[str, Any]]:
    """Filter tools to only include those matching the detected intent.

    Special cases:
      - UNKNOWN/NONE: return all tools (no filtering).
      - READ intent: also allows POST tools named "search"/"query"/"filter"
        (data retrieval endpoints that use POST for complex query bodies).

    Falls back to returning all tools if filtering yields an empty list.
    """
    if intent in (ActionIntent.UNKNOWN, ActionIntent.NONE):
        return tools

    allowed_methods = _INTENT_TO_METHODS.get(intent, _ALL_METHODS)

    filtered: List[Dict[str, Any]] = []
    for tool in tools:
        method = tool.get("method", "GET").upper()
        if method in allowed_methods:
            filtered.append(tool)
        elif intent == ActionIntent.READ and method == "POST":
            tool_name = tool.get("name", "").lower()
            if "search" in tool_name or "query" in tool_name or "filter" in tool_name:
                filtered.append(tool)

    return filtered if filtered else tools

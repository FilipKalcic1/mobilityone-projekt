"""
Query Router - ML-based routing with response formatting.

Routes queries to tools and formats responses.
Uses ML model instead of regex patterns.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from services.intent_classifier import get_intent_classifier, predict_with_ensemble

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """Result of query routing."""
    matched: bool
    tool_name: Optional[str] = None
    extract_fields: List[str] = None
    response_template: Optional[str] = None
    flow_type: Optional[str] = None
    confidence: float = 1.0
    reason: str = ""

    def __post_init__(self):
        if self.extract_fields is None:
            self.extract_fields = []


# Single source of truth: tool_routing.py
# All intent-to-tool mappings defined there to avoid triple redundancy
from tool_routing import INTENT_CONFIG as INTENT_METADATA  # noqa: E402

# Confidence threshold for DETERMINISTIC routing (bypasses LLM)
# Balanced threshold: High-confidence ML predictions bypass LLM for speed
# Lower confidence queries still go to LLM for final decision
ML_CONFIDENCE_THRESHOLD = 0.85  # 85%+ confidence uses ML directly (faster, cheaper)


class QueryRouter:
    """
    Routes queries to tools using ML-based intent classification.

    Version 2.0: Uses trained ML model instead of regex patterns.
    - 99.25% accuracy vs ~67% with regex
    - Handles typos, variations, and Croatian diacritics
    - Single model instead of 51 regex rules
    """

    def __init__(self):
        """Initialize router with ML classifier."""
        self._classifier = None

    @property
    def classifier(self):
        """Lazy load classifier."""
        if self._classifier is None:
            self._classifier = get_intent_classifier()
        return self._classifier

    def route(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> RouteResult:
        """
        Route query to appropriate tool using ML.

        Args:
            query: User's query text
            user_context: Optional user context

        Returns:
            RouteResult with matched tool or not matched
        """
        # Get ML prediction (ensemble: TF-IDF first, semantic fallback if <75%)
        prediction = predict_with_ensemble(query)

        logger.info(
            f"ROUTER ML: '{query[:30]}...' -> {prediction.intent} "
            f"({prediction.confidence:.1%}) tool={prediction.tool}"
        )

        # Check confidence threshold
        if prediction.confidence < ML_CONFIDENCE_THRESHOLD:
            logger.info(f"ROUTER: Low confidence ({prediction.confidence:.1%}), using semantic search")
            return RouteResult(
                matched=False,
                confidence=prediction.confidence,
                reason=f"ML confidence {prediction.confidence:.1%} below threshold"
            )

        # Get metadata for this intent
        metadata = INTENT_METADATA.get(prediction.intent)

        if metadata is None:
            # Intent recognized but no metadata - use ML tool suggestion
            return RouteResult(
                matched=True,
                tool_name=prediction.tool,
                extract_fields=[],
                response_template=None,
                flow_type="simple",
                confidence=prediction.confidence,
                reason=f"ML: {prediction.intent}"
            )

        return RouteResult(
            matched=True,
            tool_name=metadata["tool"],
            extract_fields=metadata["extract_fields"],
            response_template=metadata["response_template"],
            flow_type=metadata["flow_type"],
            confidence=prediction.confidence,
            reason=f"ML: {prediction.intent}"
        )

    def format_response(
        self,
        route: RouteResult,
        api_response: Dict[str, Any],
        query: str
    ) -> Optional[str]:
        """
        Format response using template if available.

        Args:
            route: The route result with template
            api_response: Raw API response
            query: Original query

        Returns:
            Formatted response string or None if should use LLM
        """
        if not route.response_template:
            return None

        if not route.extract_fields:
            return route.response_template

        # Extract value from response
        value = self._extract_value(api_response, route.extract_fields)

        if value is None:
            return None  # Let LLM handle it

        # Format value
        formatted_value = self._format_value(value, route.extract_fields[0])
        return route.response_template.format(value=formatted_value)

    def _extract_value(self, data: Dict[str, Any], fields: List[str]) -> Optional[Any]:
        """Extract value from response using field list."""
        if not data:
            return None

        for field in fields:
            if field in data and data[field] is not None:
                return data[field]
            value = self._deep_get(data, field)
            if value is not None:
                return value

        return None

    def _deep_get(self, data: Any, key: str) -> Optional[Any]:
        """Recursively search for key in nested dict."""
        if isinstance(data, dict):
            if key in data:
                return data[key]
            for v in data.values():
                result = self._deep_get(v, key)
                if result is not None:
                    return result
        elif isinstance(data, list) and data:
            return self._deep_get(data[0], key)
        return None

    def _format_value(self, value: Any, field_name: str) -> str:
        """Format value based on field type."""
        if value is None:
            return "N/A"

        field_lower = field_name.lower()

        # Mileage - add thousand separators
        if "mileage" in field_lower:
            try:
                num = int(float(value))
                return f"{num:,}".replace(",", ".")
            except (ValueError, TypeError):
                return str(value)

        # Date - format as DD.MM.YYYY
        if "date" in field_lower or "expir" in field_lower:
            if isinstance(value, str) and "T" in value:
                try:
                    date_part = value.split("T")[0]
                    parts = date_part.split("-")
                    if len(parts) == 3:
                        return f"{parts[2]}.{parts[1]}.{parts[0]}"
                except (ValueError, AttributeError, IndexError):
                    pass
            return str(value)

        return str(value)


# Singleton
_router = None


def get_query_router() -> QueryRouter:
    """Get singleton instance."""
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router

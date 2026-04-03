"""
Query Router - ML-based routing with response formatting.

Routes queries to tools and formats responses.
Uses ML model instead of regex patterns.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import threading

from services.intent_classifier import predict_with_ensemble
from services.dynamic_threshold import (
    get_engine as _get_engine, DecisionEngine, DecisionAction,
    ClassificationSignal, PredictionSet,
)
from services.tracing import get_tracer, trace_span
from services.errors import RoutingError, ErrorCode

logger = logging.getLogger(__name__)
_tracer = get_tracer("query_router")


@dataclass
class RouteResult:
    """Result of query routing."""
    matched: bool
    tool_name: Optional[str] = None
    extract_fields: List[str] = None
    response_template: Optional[str] = None
    flow_type: Optional[str] = None
    confidence: float = 1.0
    signal: Optional[ClassificationSignal] = None
    prediction_set: Optional[PredictionSet] = None
    reason: str = ""

    def __post_init__(self):
        if self.extract_fields is None:
            self.extract_fields = []


# Single source of truth: tool_routing.py
# All intent-to-tool mappings defined there to avoid triple redundancy
from tool_routing import INTENT_CONFIG as INTENT_METADATA  # noqa: E402

# DecisionEngine replaces hardcoded ML_CONFIDENCE_THRESHOLD = 0.85
# At α=0.0: identical to old `confidence >= 0.85` check
ML_CONFIDENCE_THRESHOLD = 0.85  # backward-compat re-export for tests
_engine = _get_engine()


class QueryRouter:
    """
    Routes queries to tools using ML-based intent classification.

    Version 2.0: Uses trained ML model instead of regex patterns.
    - 99.25% accuracy vs ~67% with regex
    - Handles typos, variations, and Croatian diacritics
    - Single model instead of 51 regex rules
    """

    def route(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> RouteResult:
        """
        Route query to appropriate tool using ML.

        Args:
            query: User's query text
            user_context: Optional user context

        Returns:
            RouteResult with matched tool or not matched
        """
        with trace_span(_tracer, "qrouter.route", {
            "query.preview": query[:50],
            "has_context": user_context is not None,
        }) as span:
            # Get ML prediction (ensemble: TF-IDF first, semantic fallback if <75%)
            prediction = predict_with_ensemble(query)

            logger.info(
                f"ROUTER ML: '{query[:30]}...' -> {prediction.intent} "
                f"({prediction.confidence:.1%}) tool={prediction.tool}"
            )

            # CP-aware 3-zone decision: ACCEPT / BOOST / DEFER
            prediction_set = prediction.prediction_set
            decision = _engine.decide_with_cp(
                prediction.signal, DecisionEngine.ML_FAST_PATH, prediction_set
            )

            span.set_attribute("qrouter.intent", prediction.intent)
            span.set_attribute("qrouter.confidence", prediction.confidence)
            span.set_attribute("qrouter.decision", decision.action.value)

            if decision.is_defer:
                logger.info(f"ROUTER: DEFER ({prediction.confidence:.1%}), using semantic search")
                return RouteResult(
                    matched=False,
                    confidence=prediction.confidence,
                    signal=prediction.signal,
                    prediction_set=prediction_set,
                    reason=f"ML confidence {prediction.confidence:.1%} below threshold"
                )

            # Get metadata for this intent
            metadata = INTENT_METADATA.get(prediction.intent)

            # BOOST → mediation path (CP set 2-5, LLM reranks small candidate set)
            if decision.action is DecisionAction.BOOST:
                cp_size = prediction_set.size if prediction_set else 0
                logger.info(
                    f"ROUTER: BOOST ({prediction.confidence:.1%}), "
                    f"CP set size={cp_size}, mediation path"
                )
                return RouteResult(
                    matched=True,
                    tool_name=metadata["tool"] if metadata else prediction.tool,
                    extract_fields=metadata["extract_fields"] if metadata else [],
                    response_template=metadata["response_template"] if metadata else None,
                    flow_type="mediation",
                    confidence=prediction.confidence,
                    signal=prediction.signal,
                    prediction_set=prediction_set,
                    reason=f"ML+CP: {prediction.intent} (set={cp_size})"
                )

            # ACCEPT → fast path (high confidence, CP set=1 or no CP)
            if metadata is None:
                # Intent recognized but no metadata - use ML tool suggestion
                return RouteResult(
                    matched=True,
                    tool_name=prediction.tool,
                    extract_fields=[],
                    response_template=None,
                    flow_type="simple",
                    confidence=prediction.confidence,
                    signal=prediction.signal,
                    prediction_set=prediction_set,
                    reason=f"ML: {prediction.intent}"
                )

            return RouteResult(
                matched=True,
                tool_name=metadata["tool"],
                extract_fields=metadata["extract_fields"],
                response_template=metadata["response_template"],
                flow_type=metadata["flow_type"],
                confidence=prediction.confidence,
                signal=prediction.signal,
                prediction_set=prediction_set,
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

    def _deep_get(self, data: Any, key: str, _depth: int = 0) -> Optional[Any]:
        """Recursively search for key in nested dict/list."""
        if _depth > 10:
            return None
        if isinstance(data, dict):
            if key in data:
                return data[key]
            for v in data.values():
                result = self._deep_get(v, key, _depth + 1)
                if result is not None:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self._deep_get(item, key, _depth + 1)
                if result is not None:
                    return result
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
_singleton_lock = threading.Lock()


def get_query_router() -> QueryRouter:
    """Get singleton instance."""
    global _router
    if _router is None:
        with _singleton_lock:
            if _router is None:
                _router = QueryRouter()
    return _router

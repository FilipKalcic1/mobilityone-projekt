"""
Pydantic V2 Domain Models — MobilityOne WhatsApp Bot.

Replaces ad-hoc dataclasses with validated, frozen, JSON-serializable models.
All models use Pydantic V2 with:
  - model_config = ConfigDict(frozen=True) for immutability
  - Strict type hints with validation
  - Proper serialization via model_dump() / model_dump_json()

IMPORTANT: ClassificationSignal, PredictionSet, DecisionBoundary, and
ThresholdDecision remain as frozen dataclasses in dynamic_threshold.py.
They are performance-critical (called per-prediction) and their mathematical
invariants are proven correct — Pydantic overhead is not justified there.

This module covers the ROUTING and SEARCH domain models that benefit from
validation, serialization, and structured metadata.

Usage:
    from services.domain_models import (
        IntentPredictionResult, SearchResultItem, SearchResponse,
        RouterDecisionResult, RoutingTier, RoutingTrace,
        QueryAction, QueryTypeResult,
    )
"""

from __future__ import annotations

from enum import Enum, unique
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from services.dynamic_threshold import (
    ClassificationSignal,
    NO_SIGNAL,
    PredictionSet,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

@unique
class RoutingTier(str, Enum):
    """Which routing tier handled this query."""
    FAST_PATH = "fast_path"           # ML only, 0 LLM calls, <1ms
    MEDIATION = "mediation"           # CP set 2-5 → LLM reranker, ~200ms
    FULL_SEARCH = "full_search"       # FAISS + full LLM routing, 2-3s
    DETERMINISTIC = "deterministic"   # Pattern match (greeting, exit, flow signal)
    FALLBACK = "fallback"             # Circuit breaker / error recovery


@unique
class QueryAction(str, Enum):
    """Router decision action — what to do with the query."""
    CONTINUE_FLOW = "continue_flow"
    EXIT_FLOW = "exit_flow"
    START_FLOW = "start_flow"
    SIMPLE_API = "simple_api"
    DIRECT_RESPONSE = "direct_response"
    CLARIFY = "clarify"


@unique
class FlowType(str, Enum):
    """Multi-step flow types."""
    BOOKING = "booking"
    MILEAGE = "mileage"
    CASE = "case"
    GENERIC = "generic"


# ---------------------------------------------------------------------------
# Intent Prediction Models
# ---------------------------------------------------------------------------

class IntentPredictionResult(BaseModel):
    """Result of ML intent classification.

    Replaces the old IntentPrediction dataclass with Pydantic validation.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    intent: str = Field(
        ..., min_length=1,
        description="Predicted intent label (e.g. 'vehicle_list')",
    )
    action: str = Field(
        ...,
        description="HTTP method (GET/POST/PUT/PATCH/DELETE/NONE)",
    )
    tool: Optional[str] = Field(
        default=None,
        description="Mapped tool_id from INTENT_CONFIG",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model confidence (calibrated via Platt scaling)",
    )
    alternatives: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="Top-N alternative (label, probability) pairs",
    )
    signal: ClassificationSignal = Field(
        default=NO_SIGNAL,
        description="4-float distribution shape summary",
    )
    prediction_set: Optional[PredictionSet] = Field(
        default=None,
        description="Conformal prediction set (APS), if calibrated",
    )
    algorithm: str = Field(
        default="tfidf_lr",
        description="Which algorithm produced this prediction",
    )

    @property
    def has_cp(self) -> bool:
        """Whether conformal prediction data is available."""
        return self.prediction_set is not None and self.prediction_set.size > 0


# ---------------------------------------------------------------------------
# Query Type Models
# ---------------------------------------------------------------------------

class QueryTypeResult(BaseModel):
    """Result of query type classification (suffix detection)."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    query_type: str = Field(
        ...,
        description="Detected query type (DOCUMENTS, METADATA, AGGREGATION, ...)",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
    )
    preferred_suffixes: List[str] = Field(
        default_factory=list,
        description="Tool suffixes preferred for this query type",
    )
    excluded_suffixes: List[str] = Field(
        default_factory=list,
        description="Tool suffixes to penalize for this query type",
    )
    alternatives: List[Tuple[str, float]] = Field(default_factory=list)
    signal: ClassificationSignal = Field(default=NO_SIGNAL)
    prediction_set: Optional[PredictionSet] = Field(default=None)


# ---------------------------------------------------------------------------
# Search Models
# ---------------------------------------------------------------------------

class SearchResultItem(BaseModel):
    """Individual tool result from the search pipeline.

    Replaces UnifiedSearchResult dataclass.
    """
    model_config = ConfigDict(frozen=True)

    tool_id: str = Field(..., min_length=1)
    score: float = Field(
        ...,
        description="Final score after all boosts",
    )
    method: str = Field(
        ...,
        description="HTTP method (GET/POST/PUT/PATCH/DELETE)",
    )
    description: str = Field(
        default="",
        description="Tool description for LLM context",
    )
    origin_guide: Dict[str, str] = Field(
        default_factory=dict,
        description="Parameter origin guide (param → source hint)",
    )
    boosts_applied: List[Tuple[str, float, float]] = Field(
        default_factory=list,
        description="Debug: (boost_name, value, score_after) tuples",
    )
    base_score: float = Field(
        default=0.0,
        description="FAISS cosine similarity before boosts",
    )

    @property
    def total_boost(self) -> float:
        """Sum of all additive boosts applied."""
        return self.score - self.base_score


class SearchResponse(BaseModel):
    """Complete response from the search pipeline.

    Replaces UnifiedSearchResponse dataclass.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    results: List[SearchResultItem] = Field(default_factory=list)
    intent: str = Field(
        default="GET",
        description="ActionIntent value (GET/POST/PUT/DELETE/UNKNOWN/NONE)",
    )
    intent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    query_type: str = Field(
        default="DEFAULT_SET",
        description="Detected query type",
    )
    query_type_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    query: str = Field(default="")
    total_candidates: int = Field(
        default=0, ge=0,
        description="Candidates before filtering",
    )

    @property
    def top_result(self) -> Optional[SearchResultItem]:
        """Top-scoring result, if any."""
        return self.results[0] if self.results else None

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0


# ---------------------------------------------------------------------------
# Routing Models
# ---------------------------------------------------------------------------

class RouterDecisionResult(BaseModel):
    """Result of the unified router decision.

    Replaces RouterDecision dataclass. Adds routing_tier for observability.
    """
    model_config = ConfigDict(frozen=True)

    action: QueryAction = Field(
        ...,
        description="What to do with this query",
    )
    tool: Optional[str] = Field(
        default=None,
        description="Selected tool_id (if action requires tool execution)",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted parameters for tool execution",
    )
    flow_type: Optional[FlowType] = Field(
        default=None,
        description="Flow type for start_flow/continue_flow",
    )
    response: Optional[str] = Field(
        default=None,
        description="Direct text response (for direct_response action)",
    )
    clarification: Optional[str] = Field(
        default=None,
        description="Clarification question (for clarify action)",
    )
    reasoning: str = Field(
        default="",
        description="LLM or rule reasoning for decision",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
    )
    routing_tier: RoutingTier = Field(
        default=RoutingTier.FULL_SEARCH,
        description="Which routing tier handled this query",
    )
    ambiguity_detected: bool = Field(
        default=False,
        description="Whether ambiguity was detected in tool selection",
    )

    @property
    def needs_tool_execution(self) -> bool:
        """Whether this decision requires executing a tool."""
        return self.action in (QueryAction.SIMPLE_API, QueryAction.START_FLOW)


# ---------------------------------------------------------------------------
# Observability — Routing Trace
# ---------------------------------------------------------------------------

class RoutingTrace(BaseModel):
    """Structured trace of a routing decision for OpenTelemetry.

    Captures the full decision path through the 3-tier pipeline.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    query: str = Field(default="")
    normalized_query: str = Field(default="")
    tier: RoutingTier = Field(default=RoutingTier.FULL_SEARCH)

    # ML classification
    ml_intent: Optional[str] = Field(default=None)
    ml_confidence: float = Field(default=0.0)
    ml_algorithm: str = Field(default="tfidf_lr")

    # Conformal prediction
    cp_set_size: int = Field(default=0)
    cp_coverage: float = Field(default=0.0)
    cp_labels: List[str] = Field(default_factory=list)

    # Decision engine
    effective_score: float = Field(default=0.0)
    decision_action: str = Field(default="DEFER")
    boundary_name: str = Field(default="")

    # Search
    faiss_candidates: int = Field(default=0)
    top_faiss_score: float = Field(default=0.0)
    exact_match_hit: bool = Field(default=False)

    # Reranking
    rerank_winner: Optional[str] = Field(default=None)
    rerank_confidence: float = Field(default=0.0)

    # Ambiguity
    ambiguity_detected: bool = Field(default=False)
    ambiguity_suffix: Optional[str] = Field(default=None)

    # Final outcome
    selected_tool: Optional[str] = Field(default=None)
    final_confidence: float = Field(default=0.0)
    latency_ms: float = Field(default=0.0)

    def to_span_attributes(self) -> Dict[str, Any]:
        """Convert to flat dict for OpenTelemetry span attributes.

        OTel attributes must be str/int/float/bool — no nested objects.
        """
        attrs: Dict[str, Any] = {
            "routing.tier": self.tier.value,
            "routing.query_length": len(self.query),
            "routing.latency_ms": self.latency_ms,
        }
        if self.ml_intent:
            attrs["routing.ml.intent"] = self.ml_intent
            attrs["routing.ml.confidence"] = self.ml_confidence
            attrs["routing.ml.algorithm"] = self.ml_algorithm
        if self.cp_set_size > 0:
            attrs["routing.cp.set_size"] = self.cp_set_size
            attrs["routing.cp.coverage"] = self.cp_coverage
        attrs["routing.decision.action"] = self.decision_action
        attrs["routing.decision.effective_score"] = self.effective_score
        if self.boundary_name:
            attrs["routing.decision.boundary"] = self.boundary_name
        if self.faiss_candidates > 0:
            attrs["routing.search.candidates"] = self.faiss_candidates
            attrs["routing.search.top_score"] = self.top_faiss_score
            attrs["routing.search.exact_match"] = self.exact_match_hit
        if self.rerank_winner:
            attrs["routing.rerank.winner"] = self.rerank_winner
            attrs["routing.rerank.confidence"] = self.rerank_confidence
        attrs["routing.ambiguity.detected"] = self.ambiguity_detected
        if self.selected_tool:
            attrs["routing.result.tool"] = self.selected_tool
            attrs["routing.result.confidence"] = self.final_confidence
        return attrs

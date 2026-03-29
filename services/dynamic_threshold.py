"""
Defense-in-Depth Confidence Decision Engine.

Replaces magic-number thresholds with margin-aware, entropy-informed decisions.

Architecture:
  ClassificationSignal — immutable 4-float summary of a probability distribution,
  computed once at classification time, propagated through the pipeline.

  DecisionEngine — consumes signals, produces typed decisions. Zero string
  comparisons, zero unnamed constants.

Mathematical foundation:
  effective_score = confidence × (1 − α × entropy_norm)

  - α = 0.0: degrades to pure confidence (backward-compatible)
  - α > 0: entropy penalizes spread distributions, rewarding concentrated ones
  - entropy_norm ∈ [0, 1]: 0 = certain (one-hot), 1 = uniform (max disorder)

  Every named boundary has a documented derivation — either from legacy
  compatibility (Phase 0) or from class-count math (Phase 2+).

Usage:
    from services.dynamic_threshold import get_engine, ClassificationSignal

    engine = get_engine()
    signal = ClassificationSignal.from_probabilities(probs)
    decision = engine.decide(signal, engine.INTENT_FILTER)

    if decision.is_accept:
        ...  # proceed without LLM
    elif decision.is_defer:
        ...  # route to LLM
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence, Optional
import threading


# ---------------------------------------------------------------------------
# Decision action — type-safe enum, no string comparisons
# ---------------------------------------------------------------------------

class DecisionAction(Enum):
    """Routing decision — exhaustive, no default."""
    ACCEPT = auto()        # High certainty → proceed deterministically
    BOOST = auto()         # Medium certainty → apply ranking signals
    DEFER = auto()         # Low certainty → defer to LLM


# ---------------------------------------------------------------------------
# Classification signal — compact, immutable, computed once
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ClassificationSignal:
    """Immutable summary of a probability distribution's shape.

    Computed once at classification time (O(K)), propagated through the
    entire routing pipeline. Four orthogonal dimensions:

    confidence   : max(P) — how much mass is on the top class
    margin       : P₁ - P₂ — how far ahead the top class is
    entropy_norm : H / H_max ∈ [0, 1] — how spread the mass is
    n_classes    : K — number of classes (needed for interpretation)
    """
    confidence: float
    margin: float
    entropy_norm: float
    n_classes: int

    # --- Factory methods ---

    @classmethod
    def from_probabilities(cls, probs: Sequence[float]) -> ClassificationSignal:
        """Build signal from a full predict_proba() vector.  O(K)."""
        if not probs:
            return cls(confidence=0.0, margin=0.0, entropy_norm=1.0, n_classes=0)

        k = len(probs)
        sorted_p = sorted(probs, reverse=True)
        confidence = max(sorted_p[0], 0.0)
        margin = (sorted_p[0] - sorted_p[1]) if k > 1 else confidence

        # Shannon entropy, normalized to [0, 1]
        h = 0.0
        for p in sorted_p:
            if p > 1e-12:  # skip zeros for numerical stability
                h -= p * math.log(p)
        h_max = math.log(k) if k > 1 else 1.0
        entropy_norm = min(h / h_max, 1.0) if h_max > 0 else 0.0

        return cls(
            confidence=confidence,
            margin=max(margin, 0.0),
            entropy_norm=entropy_norm,
            n_classes=k,
        )

    @classmethod
    def from_alternatives(
        cls,
        confidence: float,
        alternatives: Sequence[tuple],
        n_classes: int,
    ) -> ClassificationSignal:
        """Build signal from confidence + top-N alternatives list.

        This is the bridge from IntentPrediction.alternatives to a full signal.
        Reconstructs an approximate probability vector from sparse data.
        """
        probs = [confidence]
        for _label, prob in alternatives:
            probs.append(prob)

        # Distribute remaining mass uniformly across unseen classes
        seen_mass = sum(probs)
        remaining = max(1.0 - seen_mass, 0.0)
        unseen = n_classes - len(probs)
        if unseen > 0 and remaining > 0:
            p_unseen = remaining / unseen
            probs.extend([p_unseen] * unseen)

        return cls.from_probabilities(probs)

    @classmethod
    def from_confidence_only(
        cls,
        confidence: float,
        n_classes: int,
    ) -> ClassificationSignal:
        """Backward-compatible: estimate signal from only max(P).

        Assumes worst-case (maximum entropy consistent with observed confidence):
        P = [confidence, (1-conf)/(K-1), ..., (1-conf)/(K-1)].
        """
        if n_classes < 2:
            return cls(confidence=confidence, margin=confidence,
                       entropy_norm=0.0, n_classes=max(n_classes, 1))

        remaining = max(1.0 - confidence, 0.0)
        p_rest = remaining / (n_classes - 1)

        # Reconstruct full vector and delegate
        probs = [confidence] + [p_rest] * (n_classes - 1)
        return cls.from_probabilities(probs)

    # --- Derived properties ---

    @property
    def dominance_ratio(self) -> float:
        """How dominant is the top prediction? ∈ [0, 1].

        0.0 = top two are identical (complete ambiguity)
        1.0 = P₂ = 0 (complete dominance)
        """
        return self.margin / self.confidence if self.confidence > 0 else 0.0


# Sentinel for "no signal available"
NO_SIGNAL = ClassificationSignal(
    confidence=0.0, margin=0.0, entropy_norm=1.0, n_classes=0,
)


# ---------------------------------------------------------------------------
# Prediction set — conformal prediction output
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PredictionSet:
    """Conformal prediction set with coverage guarantee.

    Immutable set of candidate labels produced by APS (Adaptive Prediction Sets).
    At calibrated coverage level (e.g. 98%), the true label is guaranteed to be
    in the set with at least that probability.
    """
    labels: tuple           # Candidate labels, sorted by probability (desc)
    probabilities: tuple    # Corresponding probabilities (desc)
    coverage: float         # Target coverage level (1 - alpha)
    q_hat: float            # Calibrated nonconformity threshold

    @property
    def size(self) -> int:
        return len(self.labels)

    @property
    def top_label(self) -> str:
        return self.labels[0] if self.labels else ""

    @property
    def top_probability(self) -> float:
        return self.probabilities[0] if self.probabilities else 0.0

    @classmethod
    def from_probabilities(
        cls,
        probs: Sequence[float],
        label_names: Sequence[str],
        q_hat: float,
        coverage: float = 0.98,
    ) -> PredictionSet:
        """APS (Adaptive Prediction Sets) algorithm.

        Sort probabilities descending.  Include label y in the set iff
        the cumulative probability up to (and including) y is <= q_hat.
        This guarantees coverage >= 1 - alpha by the conformal guarantee.

        O(K log K) from the sort.
        """
        # Sort (label, prob) pairs by probability descending
        pairs = sorted(zip(label_names, probs), key=lambda x: x[1], reverse=True)

        # APS: include labels while cumulative <= q_hat
        cumulative = 0.0
        selected_labels = []
        selected_probs = []

        for label, prob in pairs:
            cumulative += prob
            selected_labels.append(label)
            selected_probs.append(prob)
            if cumulative > q_hat:
                break  # Include this boundary label for coverage guarantee

        # Always include at least one label
        if not selected_labels and pairs:
            selected_labels.append(pairs[0][0])
            selected_probs.append(pairs[0][1])

        return cls(
            labels=tuple(selected_labels),
            probabilities=tuple(selected_probs),
            coverage=coverage,
            q_hat=q_hat,
        )


# Sentinel for "no calibration available"
NO_PREDICTION_SET = PredictionSet(
    labels=(), probabilities=(), coverage=0.0, q_hat=0.0,
)


# ---------------------------------------------------------------------------
# Decision boundary — named, documented, derivable
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DecisionBoundary:
    """Named decision boundary with accept/boost gates.

    Phase 0: gates match legacy fixed thresholds (from_legacy_threshold).
    Phase 2+: gates derived from class count + risk factor (derived).
    """
    name: str
    accept_gate: float   # effective_score >= this → ACCEPT
    boost_gate: float    # effective_score >= this → BOOST (else DEFER)

    @classmethod
    def from_legacy_threshold(cls, name: str, threshold: float) -> DecisionBoundary:
        """Create boundary that exactly replicates an old fixed threshold.

        Sets accept == boost → 2-zone model (ACCEPT or DEFER),
        matching old binary `conf >= threshold` checks.
        """
        return cls(name=name, accept_gate=threshold, boost_gate=threshold)

    @classmethod
    def derived(
        cls, name: str, n_classes: int, risk_factor: float,
    ) -> DecisionBoundary:
        """Derive gates from class count and risk tolerance.

        Derivation:
          max_useful = 1 - 1/K  (distance from random baseline to certainty)
          accept_gate = max_useful × (1 - risk_factor)
          boost_gate  = accept_gate / 2

        risk_factor ∈ [0, 1]:  0 = maximum caution,  1 = maximum tolerance.
        """
        if not 0.0 <= risk_factor <= 1.0:
            raise ValueError(f"risk_factor must be in [0, 1], got {risk_factor}")
        max_useful = 1.0 - 1.0 / max(n_classes, 2)
        accept = max_useful * (1.0 - risk_factor)
        return cls(name=name, accept_gate=accept, boost_gate=accept * 0.5)


# ---------------------------------------------------------------------------
# Threshold decision — immutable result with convenience properties
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    """Immutable result of a threshold decision."""
    action: DecisionAction
    signal: ClassificationSignal
    effective_score: float

    @property
    def is_accept(self) -> bool:
        return self.action is DecisionAction.ACCEPT

    @property
    def is_boost(self) -> bool:
        return self.action is DecisionAction.BOOST

    @property
    def is_defer(self) -> bool:
        return self.action is DecisionAction.DEFER

    @property
    def confidence(self) -> float:
        """Shortcut for downstream code that needs raw confidence."""
        return self.signal.confidence

    @property
    def margin(self) -> float:
        return self.signal.margin


# ---------------------------------------------------------------------------
# Decision engine — stateless logic, singleton lifecycle
# ---------------------------------------------------------------------------

class DecisionEngine:
    """Margin-aware threshold engine.

    With α=0.0 (default): every decision is identical to the old fixed
    thresholds — verified by backward-compatibility tests.

    With α>0: entropy penalizes ambiguous distributions.
    """
    __slots__ = ("_alpha",)

    # --- Pre-built boundaries (Phase 0: legacy-compatible) ---
    # Derivation: each gate matches the old hardcoded constant exactly.

    # Intent filter — old gate: conf >= 0.95
    # Rationale: only filter FAISS results when ML is near-certain about method
    INTENT_FILTER = DecisionBoundary.from_legacy_threshold("intent_filter", 0.95)

    # Query type — old gate: conf >= 0.65
    # Rationale: suffix boost is a soft signal, doesn't need high confidence
    QUERY_TYPE = DecisionBoundary.from_legacy_threshold("query_type", 0.65)

    # Possessive override — old gate: conf >= 0.50
    # Rationale: possessive regex is high-precision, low bar for ML confirmation
    POSSESSIVE = DecisionBoundary.from_legacy_threshold("possessive", 0.50)

    # ML fast path — old gate: conf >= 0.85
    # Rationale: skips LLM entirely, must be confident
    ML_FAST_PATH = DecisionBoundary.from_legacy_threshold("ml_fast_path", 0.85)

    # Mutation detection — old gate: conf >= 0.50
    # Rationale: verb detection already confirms, ML is secondary
    MUTATION = DecisionBoundary.from_legacy_threshold("mutation", 0.50)

    # General fallback
    GENERAL = DecisionBoundary.from_legacy_threshold("general", 0.70)

    def __init__(self, alpha: float = 0.0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def decide(
        self,
        signal: ClassificationSignal,
        boundary: DecisionBoundary,
    ) -> ThresholdDecision:
        """Make a routing decision from a classification signal.

        Effective score formula:
          α = 0: E = confidence              (backward-compatible)
          α > 0: E = confidence × (1 − α × entropy_norm)
        """
        if signal.n_classes == 0:
            return ThresholdDecision(DecisionAction.DEFER, signal, 0.0)

        # Core formula
        if self._alpha == 0.0:
            effective = signal.confidence
        else:
            effective = signal.confidence * (1.0 - self._alpha * signal.entropy_norm)

        # 3-zone decision
        if effective >= boundary.accept_gate:
            action = DecisionAction.ACCEPT
        elif effective >= boundary.boost_gate:
            action = DecisionAction.BOOST
        else:
            action = DecisionAction.DEFER

        return ThresholdDecision(action, signal, effective)

    def decide_with_cp(
        self,
        signal: ClassificationSignal,
        boundary: DecisionBoundary,
        prediction_set: Optional[PredictionSet] = None,
    ) -> ThresholdDecision:
        """CP-aware decision: prediction set size drives the zone.

        Layered on top of decide():
        - No CP data → falls through to base decide()
        - CP set size == 1 AND base ACCEPT → ACCEPT (CP confirms Top-1)
        - CP set size 2-5 → BOOST (mediation path — LLM reranks small set)
        - CP set size > 5 → DEFER (too many candidates — full search)
        """
        base = self.decide(signal, boundary)

        # No CP calibration → existing 2-tier behavior
        if prediction_set is None or prediction_set is NO_PREDICTION_SET:
            return base

        # CP set size == 1 and base accepts → high confidence single candidate
        if prediction_set.size == 1 and base.is_accept:
            return base

        # CP set size 2-5 → mediation path (LLM reranks small candidate set)
        if prediction_set.size <= 5:
            return ThresholdDecision(
                DecisionAction.BOOST, signal, base.effective_score
            )

        # CP set size > 5 → too ambiguous, defer to full search
        return ThresholdDecision(
            DecisionAction.DEFER, signal, base.effective_score
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: Optional[DecisionEngine] = None
_engine_lock = threading.Lock()


def get_engine(alpha: float = 0.0) -> DecisionEngine:
    """Get or create the DecisionEngine singleton."""
    global _engine
    if _engine is None or _engine.alpha != alpha:
        with _engine_lock:
            if _engine is None or _engine.alpha != alpha:
                _engine = DecisionEngine(alpha=alpha)
    return _engine

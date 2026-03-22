"""
Tests for services/dynamic_threshold.py

Covers:
- DecisionAction enum
- ClassificationSignal: from_probabilities, from_alternatives, from_confidence_only
- ClassificationSignal properties: dominance_ratio
- NO_SIGNAL sentinel
- PredictionSet: from_probabilities, properties, NO_PREDICTION_SET
- DecisionBoundary: from_legacy_threshold, derived
- ThresholdDecision properties
- DecisionEngine: decide (alpha=0 backward compat + alpha>0 entropy), decide_with_cp
- get_engine singleton
"""

import math
import pytest

from services.dynamic_threshold import (
    DecisionAction,
    ClassificationSignal,
    NO_SIGNAL,
    PredictionSet,
    NO_PREDICTION_SET,
    DecisionBoundary,
    ThresholdDecision,
    DecisionEngine,
    get_engine,
)


# ============================================================================
# DecisionAction enum
# ============================================================================

class TestDecisionAction:
    def test_three_actions(self):
        assert len(DecisionAction) == 3
        assert DecisionAction.ACCEPT
        assert DecisionAction.BOOST
        assert DecisionAction.DEFER


# ============================================================================
# ClassificationSignal
# ============================================================================

class TestClassificationSignalFromProbabilities:
    def test_uniform_distribution(self):
        """Uniform → max entropy, low confidence."""
        probs = [0.25, 0.25, 0.25, 0.25]
        sig = ClassificationSignal.from_probabilities(probs)
        assert sig.confidence == pytest.approx(0.25)
        assert sig.margin == pytest.approx(0.0)
        assert sig.entropy_norm == pytest.approx(1.0, abs=0.01)
        assert sig.n_classes == 4

    def test_one_hot_distribution(self):
        """One-hot → zero entropy, max confidence."""
        probs = [1.0, 0.0, 0.0]
        sig = ClassificationSignal.from_probabilities(probs)
        assert sig.confidence == pytest.approx(1.0)
        assert sig.margin == pytest.approx(1.0)
        assert sig.entropy_norm == pytest.approx(0.0)
        assert sig.n_classes == 3

    def test_two_class_split(self):
        """60/40 split."""
        probs = [0.6, 0.4]
        sig = ClassificationSignal.from_probabilities(probs)
        assert sig.confidence == pytest.approx(0.6)
        assert sig.margin == pytest.approx(0.2)
        assert sig.n_classes == 2
        assert 0.0 < sig.entropy_norm < 1.0

    def test_empty_probs(self):
        sig = ClassificationSignal.from_probabilities([])
        assert sig.confidence == 0.0
        assert sig.n_classes == 0
        assert sig.entropy_norm == 1.0

    def test_single_class(self):
        sig = ClassificationSignal.from_probabilities([1.0])
        assert sig.confidence == 1.0
        assert sig.margin == 1.0
        assert sig.n_classes == 1

    def test_frozen(self):
        sig = ClassificationSignal.from_probabilities([0.8, 0.2])
        with pytest.raises(AttributeError):
            sig.confidence = 0.5


class TestClassificationSignalFromAlternatives:
    def test_with_alternatives(self):
        """Reconstruct from confidence + alternatives list."""
        sig = ClassificationSignal.from_alternatives(
            confidence=0.7,
            alternatives=[("b", 0.2), ("c", 0.1)],
            n_classes=3,
        )
        assert sig.confidence == pytest.approx(0.7)
        assert sig.n_classes == 3

    def test_unseen_classes_get_mass(self):
        """Extra classes should receive remaining probability mass."""
        sig = ClassificationSignal.from_alternatives(
            confidence=0.7,
            alternatives=[("b", 0.1)],
            n_classes=5,
        )
        # 0.7 + 0.1 = 0.8 seen, 0.2 spread across 3 unseen
        assert sig.n_classes == 5
        assert sig.confidence == pytest.approx(0.7)


class TestClassificationSignalFromConfidenceOnly:
    def test_backward_compat(self):
        sig = ClassificationSignal.from_confidence_only(0.9, n_classes=10)
        assert sig.confidence == pytest.approx(0.9)
        assert sig.n_classes == 10
        # Worst-case entropy: remaining mass spread uniformly
        assert sig.entropy_norm > 0.0

    def test_single_class(self):
        sig = ClassificationSignal.from_confidence_only(0.95, n_classes=1)
        assert sig.confidence == pytest.approx(0.95)
        assert sig.entropy_norm == 0.0

    def test_perfect_confidence(self):
        sig = ClassificationSignal.from_confidence_only(1.0, n_classes=29)
        assert sig.confidence == pytest.approx(1.0)
        assert sig.margin == pytest.approx(1.0)


class TestClassificationSignalProperties:
    def test_dominance_ratio_perfect(self):
        sig = ClassificationSignal(confidence=1.0, margin=1.0,
                                   entropy_norm=0.0, n_classes=3)
        assert sig.dominance_ratio == pytest.approx(1.0)

    def test_dominance_ratio_zero_conf(self):
        sig = ClassificationSignal(confidence=0.0, margin=0.0,
                                   entropy_norm=1.0, n_classes=3)
        assert sig.dominance_ratio == 0.0

    def test_dominance_ratio_tie(self):
        """When top two are equal, margin=0 → dominance=0."""
        sig = ClassificationSignal.from_probabilities([0.5, 0.5])
        assert sig.dominance_ratio == pytest.approx(0.0)


class TestNoSignal:
    def test_sentinel_values(self):
        assert NO_SIGNAL.confidence == 0.0
        assert NO_SIGNAL.margin == 0.0
        assert NO_SIGNAL.entropy_norm == 1.0
        assert NO_SIGNAL.n_classes == 0


# ============================================================================
# PredictionSet
# ============================================================================

class TestPredictionSet:
    def test_from_probabilities_basic(self):
        ps = PredictionSet.from_probabilities(
            probs=[0.7, 0.2, 0.1],
            label_names=["a", "b", "c"],
            q_hat=0.85,
        )
        assert ps.size >= 1
        assert ps.top_label == "a"
        assert ps.top_probability == pytest.approx(0.7)
        assert ps.coverage == 0.98  # default

    def test_single_dominant(self):
        """When top prob > q_hat, set should have size 1."""
        ps = PredictionSet.from_probabilities(
            probs=[0.95, 0.03, 0.02],
            label_names=["a", "b", "c"],
            q_hat=0.90,
        )
        assert ps.size == 1
        assert ps.labels == ("a",)

    def test_spread_distribution(self):
        """When spread, more labels are needed."""
        ps = PredictionSet.from_probabilities(
            probs=[0.3, 0.3, 0.2, 0.2],
            label_names=["a", "b", "c", "d"],
            q_hat=0.90,
        )
        assert ps.size >= 3

    def test_custom_coverage(self):
        ps = PredictionSet.from_probabilities(
            probs=[0.5, 0.5],
            label_names=["a", "b"],
            q_hat=0.5,
            coverage=0.80,
        )
        assert ps.coverage == 0.80

    def test_frozen(self):
        ps = PredictionSet(labels=("a",), probabilities=(1.0,),
                           coverage=0.98, q_hat=0.9)
        with pytest.raises(AttributeError):
            ps.labels = ("b",)

    def test_empty_set_fallback(self):
        """Edge: very high q_hat should still return at least 1 label."""
        ps = PredictionSet.from_probabilities(
            probs=[0.01, 0.01, 0.98],
            label_names=["a", "b", "c"],
            q_hat=0.001,  # very low threshold
        )
        assert ps.size >= 1


class TestNoPredictionSet:
    def test_sentinel(self):
        assert NO_PREDICTION_SET.size == 0
        assert NO_PREDICTION_SET.labels == ()
        assert NO_PREDICTION_SET.top_label == ""
        assert NO_PREDICTION_SET.top_probability == 0.0


# ============================================================================
# DecisionBoundary
# ============================================================================

class TestDecisionBoundary:
    def test_from_legacy_threshold(self):
        b = DecisionBoundary.from_legacy_threshold("test", 0.85)
        assert b.name == "test"
        assert b.accept_gate == 0.85
        assert b.boost_gate == 0.85  # same for 2-zone

    def test_derived(self):
        b = DecisionBoundary.derived("test", n_classes=10, risk_factor=0.5)
        assert b.name == "test"
        max_useful = 1.0 - 1.0 / 10
        expected_accept = max_useful * 0.5
        assert b.accept_gate == pytest.approx(expected_accept)
        assert b.boost_gate == pytest.approx(expected_accept * 0.5)

    def test_derived_risk_factor_bounds(self):
        with pytest.raises(ValueError):
            DecisionBoundary.derived("test", n_classes=5, risk_factor=-0.1)
        with pytest.raises(ValueError):
            DecisionBoundary.derived("test", n_classes=5, risk_factor=1.1)

    def test_derived_zero_risk(self):
        """Zero risk = maximum caution = highest gates."""
        b = DecisionBoundary.derived("strict", n_classes=10, risk_factor=0.0)
        max_useful = 1.0 - 1.0 / 10
        assert b.accept_gate == pytest.approx(max_useful)

    def test_derived_max_risk(self):
        """Risk=1 = maximum tolerance = accept_gate=0."""
        b = DecisionBoundary.derived("lax", n_classes=10, risk_factor=1.0)
        assert b.accept_gate == pytest.approx(0.0)


# ============================================================================
# ThresholdDecision
# ============================================================================

class TestThresholdDecision:
    def test_properties(self):
        sig = ClassificationSignal(0.9, 0.5, 0.1, 3)
        td = ThresholdDecision(DecisionAction.ACCEPT, sig, 0.88)
        assert td.is_accept is True
        assert td.is_boost is False
        assert td.is_defer is False
        assert td.confidence == 0.9
        assert td.margin == 0.5
        assert td.effective_score == 0.88

    def test_frozen(self):
        sig = ClassificationSignal(0.5, 0.1, 0.5, 2)
        td = ThresholdDecision(DecisionAction.DEFER, sig, 0.4)
        with pytest.raises(AttributeError):
            td.action = DecisionAction.ACCEPT


# ============================================================================
# DecisionEngine.decide()
# ============================================================================

class TestDecisionEngineDecide:
    def test_alpha_zero_backward_compat(self):
        """alpha=0 → effective == confidence (legacy behavior)."""
        engine = DecisionEngine(alpha=0.0)
        sig = ClassificationSignal.from_confidence_only(0.90, n_classes=29)
        boundary = DecisionBoundary.from_legacy_threshold("test", 0.85)
        decision = engine.decide(sig, boundary)
        assert decision.is_accept
        assert decision.effective_score == pytest.approx(0.90)

    def test_alpha_zero_below_gate(self):
        engine = DecisionEngine(alpha=0.0)
        sig = ClassificationSignal.from_confidence_only(0.80, n_classes=29)
        boundary = DecisionBoundary.from_legacy_threshold("test", 0.85)
        decision = engine.decide(sig, boundary)
        assert decision.is_defer

    def test_alpha_positive_entropy_penalty(self):
        """alpha>0 → entropy penalizes effective score."""
        engine = DecisionEngine(alpha=0.5)
        # High confidence but spread distribution
        sig = ClassificationSignal(confidence=0.90, margin=0.1,
                                   entropy_norm=0.5, n_classes=10)
        boundary = DecisionBoundary.from_legacy_threshold("test", 0.85)
        decision = engine.decide(sig, boundary)
        # E = 0.9 * (1 - 0.5 * 0.5) = 0.9 * 0.75 = 0.675
        assert decision.effective_score == pytest.approx(0.675)
        assert decision.is_defer  # below 0.85

    def test_zero_classes_defers(self):
        engine = DecisionEngine()
        decision = engine.decide(NO_SIGNAL, DecisionEngine.GENERAL)
        assert decision.is_defer
        assert decision.effective_score == 0.0

    def test_three_zone_with_derived_boundary(self):
        """Derived boundary creates 3 distinct zones."""
        engine = DecisionEngine(alpha=0.0)
        boundary = DecisionBoundary.derived("3zone", n_classes=10, risk_factor=0.3)
        # accept_gate = 0.9 * 0.7 = 0.63, boost_gate = 0.315

        # ACCEPT
        sig_high = ClassificationSignal.from_confidence_only(0.70, n_classes=10)
        assert engine.decide(sig_high, boundary).is_accept

        # BOOST
        sig_mid = ClassificationSignal.from_confidence_only(0.40, n_classes=10)
        assert engine.decide(sig_mid, boundary).is_boost

        # DEFER
        sig_low = ClassificationSignal.from_confidence_only(0.20, n_classes=10)
        assert engine.decide(sig_low, boundary).is_defer

    def test_alpha_validation(self):
        with pytest.raises(ValueError):
            DecisionEngine(alpha=-0.1)
        with pytest.raises(ValueError):
            DecisionEngine(alpha=1.1)

    def test_prebuilt_boundaries(self):
        """All class-level boundaries exist and have expected gates."""
        assert DecisionEngine.ML_FAST_PATH.accept_gate == 0.85
        assert DecisionEngine.INTENT_FILTER.accept_gate == 0.95
        assert DecisionEngine.QUERY_TYPE.accept_gate == 0.65
        assert DecisionEngine.POSSESSIVE.accept_gate == 0.50
        assert DecisionEngine.MUTATION.accept_gate == 0.50
        assert DecisionEngine.GENERAL.accept_gate == 0.70


# ============================================================================
# DecisionEngine.decide_with_cp()
# ============================================================================

class TestDecisionEngineDecideWithCp:
    def test_no_cp_falls_through(self):
        """No prediction set → same as decide()."""
        engine = DecisionEngine()
        sig = ClassificationSignal.from_confidence_only(0.90, n_classes=29)
        base = engine.decide(sig, DecisionEngine.ML_FAST_PATH)
        cp_result = engine.decide_with_cp(sig, DecisionEngine.ML_FAST_PATH, None)
        assert cp_result.action == base.action

    def test_no_prediction_set_sentinel(self):
        engine = DecisionEngine()
        sig = ClassificationSignal.from_confidence_only(0.90, n_classes=29)
        result = engine.decide_with_cp(sig, DecisionEngine.ML_FAST_PATH,
                                       NO_PREDICTION_SET)
        assert result.is_accept  # falls through to base

    def test_cp_size_1_accept(self):
        """CP set size=1 + base ACCEPT → ACCEPT."""
        engine = DecisionEngine()
        sig = ClassificationSignal.from_confidence_only(0.90, n_classes=29)
        ps = PredictionSet(labels=("a",), probabilities=(0.9,),
                           coverage=0.98, q_hat=0.85)
        result = engine.decide_with_cp(sig, DecisionEngine.ML_FAST_PATH, ps)
        assert result.is_accept

    def test_cp_size_3_boost(self):
        """CP set size=3 → BOOST (mediation path)."""
        engine = DecisionEngine()
        sig = ClassificationSignal.from_confidence_only(0.90, n_classes=29)
        ps = PredictionSet(labels=("a", "b", "c"),
                           probabilities=(0.5, 0.3, 0.2),
                           coverage=0.98, q_hat=0.85)
        result = engine.decide_with_cp(sig, DecisionEngine.ML_FAST_PATH, ps)
        assert result.is_boost

    def test_cp_size_5_boost(self):
        """CP set size=5 → still BOOST."""
        engine = DecisionEngine()
        sig = ClassificationSignal.from_confidence_only(0.90, n_classes=29)
        ps = PredictionSet(labels=tuple(f"l{i}" for i in range(5)),
                           probabilities=(0.3, 0.2, 0.2, 0.2, 0.1),
                           coverage=0.98, q_hat=0.85)
        result = engine.decide_with_cp(sig, DecisionEngine.ML_FAST_PATH, ps)
        assert result.is_boost

    def test_cp_size_6_defer(self):
        """CP set size=6 → DEFER (too many candidates)."""
        engine = DecisionEngine()
        sig = ClassificationSignal.from_confidence_only(0.90, n_classes=29)
        ps = PredictionSet(labels=tuple(f"l{i}" for i in range(6)),
                           probabilities=(0.2, 0.2, 0.15, 0.15, 0.15, 0.15),
                           coverage=0.98, q_hat=0.85)
        result = engine.decide_with_cp(sig, DecisionEngine.ML_FAST_PATH, ps)
        assert result.is_defer


# ============================================================================
# get_engine singleton
# ============================================================================

class TestGetEngine:
    def test_returns_engine(self):
        engine = get_engine()
        assert isinstance(engine, DecisionEngine)
        assert engine.alpha == 0.0

    def test_singleton(self):
        a = get_engine()
        b = get_engine()
        assert a is b

    def test_alpha_change_creates_new(self):
        e1 = get_engine(alpha=0.0)
        e2 = get_engine(alpha=0.3)
        assert e2.alpha == 0.3
        assert e1 is not e2

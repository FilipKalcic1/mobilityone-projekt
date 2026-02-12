"""
Tests for AmbiguityDetector - ambiguous tool selection detection and disambiguation.
"""

import pytest
from services.ambiguity_detector import (
    AmbiguityDetector,
    AmbiguityResult,
    GENERIC_SUFFIX_PATTERNS,
    ENTITY_KEYWORDS,
    CLARIFICATION_QUESTIONS,
)


@pytest.fixture
def detector():
    return AmbiguityDetector()


class TestAmbiguityResult:
    def test_default_values(self):
        result = AmbiguityResult()
        assert result.is_ambiguous is False
        assert result.ambiguous_suffix is None
        assert result.similar_tools == []
        assert result.score_variance == 0.0
        assert result.detected_entity is None
        assert result.disambiguation_hint == ""
        assert result.clarification_question is None


class TestFindCommonSuffix:
    def test_agg_suffix(self, detector):
        tools = ["get_Vehicles_Agg", "get_Trips_Agg", "get_Expenses_Agg"]
        assert detector._find_common_suffix(tools) == "_Agg"

    def test_group_by_suffix(self, detector):
        tools = ["get_A_GroupBy", "get_B_GroupBy", "get_C_GroupBy"]
        assert detector._find_common_suffix(tools) == "_GroupBy"

    def test_no_common_suffix(self, detector):
        tools = ["get_Vehicles", "post_Booking", "delete_Case"]
        assert detector._find_common_suffix(tools) is None

    def test_mixed_suffixes_below_threshold(self, detector):
        tools = ["get_A_Agg", "get_B_GroupBy", "get_C_ProjectTo", "get_D"]
        assert detector._find_common_suffix(tools) is None

    def test_empty_list(self, detector):
        assert detector._find_common_suffix([]) is None


class TestDetectEntity:
    def test_vehicle_keyword(self, detector):
        assert detector._detect_entity("prikaži vozila", None) == "Vehicles"

    def test_auto_keyword(self, detector):
        assert detector._detect_entity("daj mi auto", None) == "Vehicles"

    def test_expense_keyword(self, detector):
        assert detector._detect_entity("prikaži troškove", None) == "Expenses"

    def test_booking_keyword(self, detector):
        assert detector._detect_entity("napravi rezervaciju", None) == "VehicleCalendar"

    def test_case_keyword(self, detector):
        # "šteta" matches but entity keyword is "slučaj", "kvar", etc.
        assert detector._detect_entity("prijavi kvar", None) == "Cases"

    def test_no_match(self, detector):
        assert detector._detect_entity("hello world", None) is None

    def test_with_context_vehicle(self, detector):
        context = {"vehicle_id": "v-123", "person_id": "p-1"}
        # Even with vehicle in context, returns None (doesn't assume)
        result = detector._detect_entity("daj mi podatke", context)
        assert result is None


class TestDetectAmbiguity:
    def test_not_ambiguous_few_results(self, detector):
        results = [{"tool_id": "get_A", "score": 0.9}]
        result = detector.detect_ambiguity("test", results)
        assert result.is_ambiguous is False

    def test_not_ambiguous_empty_results(self, detector):
        result = detector.detect_ambiguity("test", [])
        assert result.is_ambiguous is False

    def test_not_ambiguous_zero_score(self, detector):
        results = [
            {"tool_id": "get_A_Agg", "score": 0},
            {"tool_id": "get_B_Agg", "score": 0},
            {"tool_id": "get_C_Agg", "score": 0},
        ]
        result = detector.detect_ambiguity("test", results)
        assert result.is_ambiguous is False

    def test_ambiguous_agg_tools(self, detector):
        results = [
            {"tool_id": "get_Vehicles_Agg", "score": 0.85},
            {"tool_id": "get_Trips_Agg", "score": 0.83},
            {"tool_id": "get_Expenses_Agg", "score": 0.82},
        ]
        result = detector.detect_ambiguity("prikaži statistiku", results)
        assert result.is_ambiguous is True
        assert result.ambiguous_suffix == "_Agg"
        assert len(result.similar_tools) > 0
        assert result.clarification_question is not None

    def test_ambiguous_with_entity_detection(self, detector):
        results = [
            {"tool_id": "get_Vehicles_Agg", "score": 0.85},
            {"tool_id": "get_Trips_Agg", "score": 0.83},
            {"tool_id": "get_Expenses_Agg", "score": 0.82},
        ]
        result = detector.detect_ambiguity("statistika vozila", results)
        assert result.detected_entity == "Vehicles"

    def test_not_ambiguous_different_suffixes(self, detector):
        results = [
            {"tool_id": "get_Vehicles", "score": 0.9},
            {"tool_id": "post_Booking", "score": 0.85},
            {"tool_id": "get_Trips", "score": 0.83},
        ]
        result = detector.detect_ambiguity("test", results)
        assert result.is_ambiguous is False


class TestBuildDisambiguationHint:
    def test_agg_hint(self, detector):
        hint = detector._build_disambiguation_hint(
            "statistika", "_Agg",
            ["get_Vehicles_Agg", "get_Trips_Agg"],
            "Vehicles"
        )
        assert "AGGREGACIJA" in hint
        assert "DETEKTIRANI ENTITET" in hint
        assert "Vehicles" in hint

    def test_hint_without_entity(self, detector):
        hint = detector._build_disambiguation_hint(
            "test", "_GroupBy",
            ["get_A_GroupBy", "get_B_GroupBy"],
            None
        )
        assert "GRUPIRANJE" in hint
        assert "DETEKTIRANI ENTITET" not in hint

    def test_hint_extracts_entities_from_tools(self, detector):
        hint = detector._build_disambiguation_hint(
            "test", "_Agg",
            ["get_Vehicles_Agg", "get_Trips_Agg"],
            None
        )
        assert "MOGUĆI ENTITETI" in hint


class TestGetBestToolForEntity:
    def test_finds_matching_tool(self, detector):
        tools = ["get_Vehicles_Agg", "get_Trips_Agg", "get_Expenses_Agg"]
        result = detector.get_best_tool_for_entity("Vehicles", "_Agg", tools)
        assert result == "get_Vehicles_Agg"

    def test_no_match(self, detector):
        tools = ["get_Vehicles_Agg", "get_Trips_Agg"]
        result = detector.get_best_tool_for_entity("Cases", "_Agg", tools)
        assert result is None


class TestNeedsClarification:
    def test_not_ambiguous(self, detector):
        result = AmbiguityResult(is_ambiguous=False)
        assert detector.needs_clarification(result, 0.5) is False

    def test_entity_detected_no_clarification(self, detector):
        result = AmbiguityResult(
            is_ambiguous=True,
            detected_entity="Vehicles"
        )
        assert detector.needs_clarification(result, 0.3) is False

    def test_llm_confident_no_clarification(self, detector):
        result = AmbiguityResult(is_ambiguous=True)
        assert detector.needs_clarification(result, 0.8) is False

    def test_many_similar_low_variance_needs_clarification(self, detector):
        result = AmbiguityResult(
            is_ambiguous=True,
            similar_tools=["a", "b", "c", "d", "e"],
            score_variance=0.05
        )
        assert detector.needs_clarification(result, 0.3) is True

    def test_few_similar_no_clarification(self, detector):
        result = AmbiguityResult(
            is_ambiguous=True,
            similar_tools=["a", "b"],
            score_variance=0.05
        )
        assert detector.needs_clarification(result, 0.3) is False


class TestConstants:
    def test_generic_suffix_patterns_non_empty(self):
        assert len(GENERIC_SUFFIX_PATTERNS) > 0

    def test_entity_keywords_non_empty(self):
        assert len(ENTITY_KEYWORDS) > 0

    def test_clarification_questions_for_each_suffix(self):
        for suffix in GENERIC_SUFFIX_PATTERNS:
            assert suffix in CLARIFICATION_QUESTIONS


class TestSingleton:
    def test_get_ambiguity_detector_singleton(self):
        """get_ambiguity_detector should return singleton."""
        from services.ambiguity_detector import get_ambiguity_detector
        detector1 = get_ambiguity_detector()
        detector2 = get_ambiguity_detector()
        assert detector1 is detector2

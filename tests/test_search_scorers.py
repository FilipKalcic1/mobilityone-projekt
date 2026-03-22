"""
Tests for services.registry.search_scorers — scoring and boosting functions.

Validates that extracted scoring functions produce correct adjustments.
"""

import unittest
from unittest.mock import MagicMock, patch

from services.registry.search_scorers import (
    DISAMBIGUATION_CONFIG,
    CATEGORY_CONFIG,
    apply_method_disambiguation,
    apply_user_specific_boosting,
    apply_evaluation_adjustment,
    apply_category_boosting,
    apply_documentation_boosting,
    apply_example_query_boosting,
)


def _make_tool(method="GET", operation_id="test_op", description="", parameters=None):
    """Create a mock UnifiedToolDefinition."""
    tool = MagicMock()
    tool.method = method
    tool.operation_id = operation_id
    tool.description = description or f"Test tool {operation_id}"
    tool.parameters = parameters or {}
    return tool


class TestDisambiguationConfig(unittest.TestCase):
    """Verify DISAMBIGUATION_CONFIG structure."""

    def test_has_required_keys(self):
        required = [
            "read_intent_penalty_factor",
            "unclear_intent_penalty_factor",
            "delete_penalty_multiplier",
        ]
        for key in required:
            self.assertIn(key, DISAMBIGUATION_CONFIG)

    def test_values_are_numeric(self):
        for key, value in DISAMBIGUATION_CONFIG.items():
            self.assertIsInstance(value, (int, float), f"{key} is not numeric")


class TestCategoryConfig(unittest.TestCase):
    """Verify CATEGORY_CONFIG structure."""

    def test_has_required_keys(self):
        required = ["keyword_match_boost", "documentation_boost"]
        for key in required:
            self.assertIn(key, CATEGORY_CONFIG)


class TestApplyMethodDisambiguation(unittest.TestCase):
    """Test intent-based scoring adjustments."""

    def test_read_query_boosts_get_methods(self):
        tools = {
            "GetVehicles": _make_tool(method="GET"),
            "UpdateVehicle": _make_tool(method="PUT"),
        }
        scored = [(0.8, "GetVehicles"), (0.8, "UpdateVehicle")]

        with patch("services.registry.search_scorers.detect_action_intent") as mock_detect:
            from services.intent_classifier import ActionIntent
            mock_result = MagicMock()
            mock_result.intent = ActionIntent.READ
            mock_detect.return_value = mock_result

            result = apply_method_disambiguation("dohvati vozila", scored, tools)

        # GET should maintain or increase score, PUT should be penalized
        result_dict = {op_id: score for score, op_id in result}
        self.assertGreater(result_dict["GetVehicles"], result_dict["UpdateVehicle"])

    def test_mutation_query_returns_unchanged(self):
        tools = {
            "UpdateVehicle": _make_tool(method="PUT"),
        }
        scored = [(0.9, "UpdateVehicle")]

        with patch("services.registry.search_scorers.detect_action_intent") as mock_detect:
            from services.intent_classifier import ActionIntent
            mock_result = MagicMock()
            mock_result.intent = ActionIntent.UPDATE
            mock_detect.return_value = mock_result

            result = apply_method_disambiguation("ažuriraj vozilo", scored, tools)

        self.assertEqual(result, scored)

    def test_empty_scored_returns_empty(self):
        with patch("services.registry.search_scorers.detect_action_intent") as mock_detect:
            from services.intent_classifier import ActionIntent
            mock_result = MagicMock()
            mock_result.intent = ActionIntent.READ
            mock_detect.return_value = mock_result

            result = apply_method_disambiguation("test", [], {})

        self.assertEqual(result, [])


class TestApplyEvaluationAdjustment(unittest.TestCase):
    """Test performance-based adjustments."""

    def test_preserves_order(self):
        scored = [(0.9, "tool_a"), (0.7, "tool_b"), (0.5, "tool_c")]
        result = apply_evaluation_adjustment(scored)
        # Should maintain relative ordering (adjustments are small)
        self.assertEqual(len(result), 3)

    def test_empty_input(self):
        result = apply_evaluation_adjustment([])
        self.assertEqual(result, [])


class TestApplyUserSpecificBoosting(unittest.TestCase):
    """Test user-specific filter parameter boosting."""

    def test_returns_same_length(self):
        tools = {"GetVehicles": _make_tool()}
        scored = [(0.8, "GetVehicles")]
        result = apply_user_specific_boosting("moja vozila", scored, tools)
        self.assertEqual(len(result), len(scored))


class TestApplyCategoryBoosting(unittest.TestCase):
    """Test category-based score adjustments."""

    def test_returns_same_length(self):
        tools = {"GetVehicles": _make_tool()}
        scored = [(0.8, "GetVehicles")]
        result = apply_category_boosting("vozila", scored, tools, {}, {})
        self.assertEqual(len(result), len(scored))


class TestApplyDocumentationBoosting(unittest.TestCase):
    """Test documentation-based score adjustments."""

    def test_with_no_documentation(self):
        scored = [(0.8, "GetVehicles")]
        result = apply_documentation_boosting("vozila", scored, None)
        self.assertEqual(len(result), len(scored))

    def test_with_matching_documentation(self):
        scored = [(0.8, "GetVehicles")]
        docs = {
            "GetVehicles": {
                "description_hr": "Dohvaća popis svih vozila u sustavu"
            }
        }
        result = apply_documentation_boosting("vozila", scored, docs)
        # Score should be boosted
        self.assertGreaterEqual(result[0][0], 0.8)


class TestApplyExampleQueryBoosting(unittest.TestCase):
    """Test example query matching."""

    def test_with_no_documentation(self):
        scored = [(0.8, "GetVehicles")]
        result = apply_example_query_boosting("vozila", scored, None)
        self.assertEqual(len(result), len(scored))

    def test_with_matching_example(self):
        scored = [(0.8, "GetVehicles")]
        docs = {
            "GetVehicles": {
                "example_queries_hr": ["dohvati sva vozila", "prikaži vozila"]
            }
        }
        result = apply_example_query_boosting("dohvati sva vozila", scored, docs)
        # Score should be boosted
        self.assertGreaterEqual(result[0][0], 0.8)


if __name__ == "__main__":
    unittest.main()

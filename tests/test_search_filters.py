"""
Tests for services.registry.search_filters — filtering functions for search.

Validates intent detection, category detection, and method/category filtering.
"""

import unittest
from unittest.mock import MagicMock, patch

from services.registry.search_filters import (
    detect_intent,
    detect_categories,
    filter_by_method,
    filter_by_categories,
)


def _make_tool(method="GET"):
    """Create a mock UnifiedToolDefinition."""
    tool = MagicMock()
    tool.method = method
    return tool


class TestDetectIntent(unittest.TestCase):
    """Test ML-based intent detection wrapper."""

    def test_read_intent(self):
        with patch("services.registry.search_filters.detect_action_intent") as mock:
            from services.intent_classifier import ActionIntent
            mock_result = MagicMock()
            mock_result.intent = ActionIntent.READ
            mock.return_value = mock_result

            result = detect_intent("dohvati vozila")
            self.assertEqual(result, "READ")

    def test_write_intent_create(self):
        with patch("services.registry.search_filters.detect_action_intent") as mock:
            from services.intent_classifier import ActionIntent
            mock_result = MagicMock()
            mock_result.intent = ActionIntent.CREATE
            mock.return_value = mock_result

            result = detect_intent("kreiraj rezervaciju")
            self.assertEqual(result, "WRITE")

    def test_unknown_intent(self):
        with patch("services.registry.search_filters.detect_action_intent") as mock:
            from services.intent_classifier import ActionIntent
            mock_result = MagicMock()
            mock_result.intent = ActionIntent.UNKNOWN
            mock.return_value = mock_result

            result = detect_intent("nešto neodređeno")
            self.assertEqual(result, "UNKNOWN")


class TestDetectCategories(unittest.TestCase):
    """Test category detection from query keywords."""

    def test_empty_keywords_returns_empty(self):
        result = detect_categories("vozila", {})
        self.assertEqual(result, set())

    def test_exact_word_match(self):
        keywords = {"vehicles": {"vozila", "auto"}}
        result = detect_categories("prikaži vozila", keywords)
        self.assertIn("vehicles", result)

    def test_substring_match_for_long_keywords(self):
        keywords = {"vehicles": {"automobil"}}
        result = detect_categories("daj mi podatke o automobilu", keywords)
        self.assertIn("vehicles", result)

    def test_no_match(self):
        keywords = {"vehicles": {"vozila", "auto"}}
        result = detect_categories("daj mi račun", keywords)
        self.assertEqual(result, set())


class TestFilterByMethod(unittest.TestCase):
    """Test HTTP method filtering based on intent."""

    def test_unknown_intent_no_filtering(self):
        tool_ids = {"GetVehicles", "UpdateVehicle"}
        tools = {
            "GetVehicles": _make_tool("GET"),
            "UpdateVehicle": _make_tool("PUT"),
        }
        result = filter_by_method(tool_ids, tools, "UNKNOWN")
        self.assertEqual(result, tool_ids)

    def test_read_intent_keeps_get(self):
        tool_ids = {"GetVehicles", "UpdateVehicle"}
        tools = {
            "GetVehicles": _make_tool("GET"),
            "UpdateVehicle": _make_tool("PUT"),
        }
        result = filter_by_method(tool_ids, tools, "READ")
        self.assertIn("GetVehicles", result)
        self.assertNotIn("UpdateVehicle", result)

    def test_write_intent_keeps_mutations(self):
        tool_ids = {"GetVehicles", "CreateVehicle", "DeleteVehicle"}
        tools = {
            "GetVehicles": _make_tool("GET"),
            "CreateVehicle": _make_tool("POST"),
            "DeleteVehicle": _make_tool("DELETE"),
        }
        result = filter_by_method(tool_ids, tools, "WRITE")
        self.assertNotIn("GetVehicles", result)
        self.assertIn("CreateVehicle", result)
        self.assertIn("DeleteVehicle", result)

    def test_read_includes_search_post(self):
        tool_ids = {"SearchVehicles"}
        tools = {"SearchVehicles": _make_tool("POST")}
        result = filter_by_method(tool_ids, tools, "READ")
        self.assertIn("SearchVehicles", result)

    def test_empty_result_falls_back_to_all(self):
        tool_ids = {"GetVehicles"}
        tools = {"GetVehicles": _make_tool("GET")}
        result = filter_by_method(tool_ids, tools, "WRITE")
        # GET doesn't match WRITE, so falls back to all
        self.assertEqual(result, tool_ids)


class TestFilterByCategories(unittest.TestCase):
    """Test category-based filtering."""

    def test_no_categories_returns_all(self):
        tool_ids = {"GetVehicles", "GetBookings"}
        result = filter_by_categories(tool_ids, {}, set())
        self.assertEqual(result, tool_ids)

    def test_filters_to_matching_category(self):
        tool_ids = {"GetVehicles", "GetBookings"}
        tool_to_category = {
            "GetVehicles": "vehicles",
            "GetBookings": "bookings",
        }
        result = filter_by_categories(tool_ids, tool_to_category, {"vehicles"})
        self.assertEqual(result, {"GetVehicles"})

    def test_empty_result_falls_back_to_all(self):
        tool_ids = {"GetVehicles"}
        tool_to_category = {"GetVehicles": "vehicles"}
        result = filter_by_categories(tool_ids, tool_to_category, {"bookings"})
        # No match, falls back to all
        self.assertEqual(result, tool_ids)


if __name__ == "__main__":
    unittest.main()

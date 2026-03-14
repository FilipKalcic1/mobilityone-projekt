"""Tests for services/query_router.py – QueryRouter."""
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

from services.query_router import (
    QueryRouter,
    RouteResult,
    INTENT_METADATA,
    ML_CONFIDENCE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def router():
    """Router with mocked ensemble prediction."""
    return QueryRouter()


def _prediction(intent="GET_MILEAGE", confidence=0.99, tool="get_MasterData"):
    """Create a mock IntentPrediction."""
    p = MagicMock()
    p.intent = intent
    p.confidence = confidence
    p.tool = tool
    return p


# ===========================================================================
# RouteResult
# ===========================================================================

class TestRouteResult:
    def test_defaults(self):
        r = RouteResult(matched=False)
        assert r.extract_fields == []
        assert r.confidence == 1.0
        assert r.reason == ""

    def test_with_fields(self):
        r = RouteResult(matched=True, tool_name="t", extract_fields=["a", "b"])
        assert r.extract_fields == ["a", "b"]

    def test_none_extract_fields_becomes_list(self):
        r = RouteResult(matched=True, extract_fields=None)
        assert r.extract_fields == []


# ===========================================================================
# route
# ===========================================================================

class TestRoute:
    @patch("services.query_router.predict_with_ensemble")
    def test_high_confidence_known_intent(self, mock_predict, router):
        mock_predict.return_value = _prediction("GET_MILEAGE", 0.99, "get_MasterData")
        result = router.route("koliko km ima auto?")
        assert result.matched is True
        assert result.tool_name == "get_MasterData"
        assert "LastMileage" in result.extract_fields
        assert result.flow_type == "simple"

    @patch("services.query_router.predict_with_ensemble")
    def test_low_confidence_not_matched(self, mock_predict, router):
        mock_predict.return_value = _prediction("GET_MILEAGE", 0.5, "get_MasterData")
        result = router.route("nesto nejasno")
        assert result.matched is False
        assert result.confidence == 0.5

    @patch("services.query_router.predict_with_ensemble")
    def test_threshold_boundary(self, mock_predict, router):
        mock_predict.return_value = _prediction("GET_MILEAGE", ML_CONFIDENCE_THRESHOLD - 0.001)
        result = router.route("q")
        assert result.matched is False

    @patch("services.query_router.predict_with_ensemble")
    def test_above_threshold(self, mock_predict, router):
        mock_predict.return_value = _prediction("GET_MILEAGE", ML_CONFIDENCE_THRESHOLD)
        result = router.route("q")
        assert result.matched is True

    @patch("services.query_router.predict_with_ensemble")
    def test_unknown_intent_uses_ml_tool(self, mock_predict, router):
        mock_predict.return_value = _prediction("UNKNOWN_INTENT", 0.99, "some_tool")
        result = router.route("q")
        assert result.matched is True
        assert result.tool_name == "some_tool"
        assert result.flow_type == "simple"

    @patch("services.query_router.predict_with_ensemble")
    def test_greeting_direct_response(self, mock_predict, router):
        mock_predict.return_value = _prediction("GREETING", 0.99, None)
        result = router.route("bok")
        assert result.matched is True
        assert result.tool_name is None
        assert result.flow_type == "direct_response"

    @patch("services.query_router.predict_with_ensemble")
    def test_help_intent(self, mock_predict, router):
        mock_predict.return_value = _prediction("HELP", 0.99, None)
        result = router.route("pomoc")
        assert result.matched is True
        assert "Mogu vam pomoci" in result.response_template

    @patch("services.query_router.predict_with_ensemble")
    def test_booking_intent(self, mock_predict, router):
        mock_predict.return_value = _prediction("BOOK_VEHICLE", 0.99, "get_AvailableVehicles")
        result = router.route("rezerviraj auto")
        assert result.matched is True
        assert result.flow_type == "booking"

    def test_all_known_intents_have_metadata(self):
        for intent_name in INTENT_METADATA:
            meta = INTENT_METADATA[intent_name]
            assert "tool" in meta
            assert "extract_fields" in meta
            assert "flow_type" in meta


# ===========================================================================
# format_response
# ===========================================================================

class TestFormatResponse:
    def test_no_template_returns_none(self, router):
        route = RouteResult(matched=True, response_template=None, extract_fields=[])
        assert router.format_response(route, {"k": "v"}, "q") is None

    def test_template_no_fields(self, router):
        route = RouteResult(matched=True, response_template="Pozdrav!", extract_fields=[])
        assert router.format_response(route, {}, "q") == "Pozdrav!"

    def test_template_with_value(self, router):
        route = RouteResult(
            matched=True,
            response_template="**Kilometraza:** {value} km",
            extract_fields=["LastMileage"]
        )
        result = router.format_response(route, {"LastMileage": 50000}, "q")
        assert "50.000" in result

    def test_missing_value_returns_none(self, router):
        route = RouteResult(
            matched=True,
            response_template="**Km:** {value}",
            extract_fields=["Missing"]
        )
        assert router.format_response(route, {"Other": 1}, "q") is None


# ===========================================================================
# _extract_value
# ===========================================================================

class TestExtractValue:
    def test_direct_field(self, router):
        assert router._extract_value({"A": 1}, ["A"]) == 1

    def test_fallback_field(self, router):
        assert router._extract_value({"B": 2}, ["A", "B"]) == 2

    def test_nested_field(self, router):
        data = {"outer": {"inner": {"Mileage": 100}}}
        assert router._extract_value(data, ["Mileage"]) == 100

    def test_none_data(self, router):
        assert router._extract_value(None, ["A"]) is None

    def test_empty_data(self, router):
        assert router._extract_value({}, ["A"]) is None

    def test_none_value_skipped(self, router):
        assert router._extract_value({"A": None, "B": 2}, ["A", "B"]) == 2


# ===========================================================================
# _deep_get
# ===========================================================================

class TestDeepGet:
    def test_flat_dict(self, router):
        assert router._deep_get({"a": 1}, "a") == 1

    def test_nested_dict(self, router):
        assert router._deep_get({"x": {"y": {"z": 42}}}, "z") == 42

    def test_list_takes_first(self, router):
        assert router._deep_get([{"a": 1}, {"a": 2}], "a") == 1

    def test_not_found(self, router):
        assert router._deep_get({"a": 1}, "b") is None

    def test_empty_list(self, router):
        assert router._deep_get([], "a") is None

    def test_non_dict_non_list(self, router):
        assert router._deep_get("string", "a") is None


# ===========================================================================
# _format_value
# ===========================================================================

class TestFormatValue:
    def test_none_value(self, router):
        assert router._format_value(None, "anything") == "N/A"

    def test_mileage_formatting(self, router):
        assert router._format_value(50000, "LastMileage") == "50.000"
        assert router._format_value(1234.5, "Mileage") == "1.234"

    def test_mileage_non_numeric(self, router):
        assert router._format_value("abc", "Mileage") == "abc"

    def test_date_formatting(self, router):
        result = router._format_value("2024-12-25T00:00:00", "ExpirationDate")
        assert result == "25.12.2024"

    def test_date_no_T(self, router):
        result = router._format_value("2024-12-25", "ExpirationDate")
        assert result == "2024-12-25"

    def test_date_invalid(self, router):
        result = router._format_value("not-a-date", "SomeDate")
        assert result == "not-a-date"

    def test_regular_value(self, router):
        assert router._format_value("hello", "SomeField") == "hello"
        assert router._format_value(42, "Count") == "42"


# ===========================================================================
# Singleton
# ===========================================================================

class TestSingleton:
    def test_get_query_router(self):
        import services.query_router as qr
        qr._router = None
        r1 = qr.get_query_router()
        r2 = qr.get_query_router()
        assert r1 is r2
        qr._router = None  # cleanup

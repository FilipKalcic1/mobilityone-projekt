"""Tests for services/unified_router.py - UnifiedRouter.

Covers:
- RouterDecision dataclass
- PRIMARY_TOOLS / FLOW_TRIGGERS / EXIT_SIGNALS constants
- UnifiedRouter.__init__ / set_registry / initialize
- _check_exit_signal
- _check_greeting
- _get_relevant_tools / _get_relevant_tools_with_ambiguity
- route() - greeting, exit_signal, in-flow shortcuts, QueryRouter fast path, LLM path
- _llm_route - normal, clarify, exit_flow-without-flow fix, exception fallback
- _fallback_route
- _query_result_to_decision (direct_response, flows, simple_api)
- get_unified_router singleton
"""

import json
import pytest
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from services.unified_router import (
    RouterDecision,
    PRIMARY_TOOLS,
    FLOW_TRIGGERS,
    UnifiedRouter,
    get_unified_router,
    _router,
)
from services.flow_phrases import EXIT_SIGNALS
from services.query_router import RouteResult


# ==========================================================================
# Helpers
# ==========================================================================

def _mock_settings():
    """Create a mock Settings object with all required attributes."""
    s = MagicMock()
    s.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com"
    s.AZURE_OPENAI_API_KEY = "test-key"
    s.AZURE_OPENAI_API_VERSION = "2024-08-01-preview"
    s.AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-mini"
    s.MOBILITY_TENANT_ID = "00000000-0000-0000-0000-000000000001"
    s.tenant_id = "00000000-0000-0000-0000-000000000001"
    return s


def _user_context(with_vehicle=True):
    """Create a sample user_context dict."""
    ctx = {
        "person_id": "11111111-1111-1111-1111-111111111111",
        "phone": "+385991234567",
        "tenant_id": "00000000-0000-0000-0000-000000000001",
        "display_name": "Test User",
    }
    if with_vehicle:
        ctx["vehicle"] = {
            "Id": "22222222-2222-2222-2222-222222222222",
            "LicencePlate": "ZG-1234-AB",
            "FullVehicleName": "VW Passat 2020",
        }
    return ctx


def _route_result(matched=True, tool_name="get_MasterData", flow_type="simple",
                  confidence=1.0, reason="ML: GET_VEHICLE_INFO",
                  response_template=None):
    """Create a RouteResult helper."""
    return RouteResult(
        matched=matched,
        tool_name=tool_name,
        flow_type=flow_type,
        confidence=confidence,
        reason=reason,
        response_template=response_template,
    )


def _make_passthrough_cb():
    """Circuit breaker mock that passes calls through to the actual function."""
    cb = MagicMock()
    async def passthrough(endpoint_key, func, *args, **kwargs):
        return await func(*args, **kwargs)
    cb.call = AsyncMock(side_effect=passthrough)
    return cb


def _make_router(registry=None):
    """Create a UnifiedRouter with mocked external deps."""
    with patch("services.unified_router.get_settings", return_value=_mock_settings()):
        with patch("services.unified_router.get_openai_client", return_value=MagicMock()):
            with patch("services.unified_router.get_llm_circuit_breaker", return_value=_make_passthrough_cb()):
                with patch("services.unified_router.QueryRouter") as MockQR:
                    router = UnifiedRouter(registry=registry)
                    # Replace the query_router that was created in __init__
                    router.query_router = MockQR.return_value
    return router


def _llm_response(data: dict):
    """Build a mock OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = json.dumps(data)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ==========================================================================
# RouterDecision dataclass
# ==========================================================================

class TestRouterDecision:
    def test_default_values(self):
        d = RouterDecision(action="simple_api")
        assert d.action == "simple_api"
        assert d.tool is None
        assert d.params == {}
        assert d.flow_type is None
        assert d.response is None
        assert d.clarification is None
        assert d.reasoning == ""
        assert d.confidence == 0.0
        assert d.ambiguity_detected is False

    def test_all_fields(self):
        d = RouterDecision(
            action="clarify",
            tool="get_MasterData",
            params={"x": 1},
            flow_type="booking",
            response="hello",
            clarification="which?",
            reasoning="test",
            confidence=0.9,
            ambiguity_detected=True,
        )
        assert d.action == "clarify"
        assert d.tool == "get_MasterData"
        assert d.params == {"x": 1}
        assert d.flow_type == "booking"
        assert d.clarification == "which?"
        assert d.ambiguity_detected is True


# ==========================================================================
# Constants
# ==========================================================================

class TestConstants:
    def test_primary_tools_not_empty(self):
        assert len(PRIMARY_TOOLS) > 10

    def test_flow_triggers_keys(self):
        assert "post_VehicleCalendar" in FLOW_TRIGGERS
        assert FLOW_TRIGGERS["post_AddMileage"] == "mileage"
        assert FLOW_TRIGGERS["post_AddCase"] == "case"

    def test_exit_signals_contains_key_phrases(self):
        assert "odustani" in EXIT_SIGNALS
        assert "stop" in EXIT_SIGNALS
        assert "cancel" in EXIT_SIGNALS


# ==========================================================================
# UnifiedRouter init / set_registry / initialize
# ==========================================================================

class TestUnifiedRouterInit:
    def test_init_creates_router(self):
        router = _make_router()
        assert router._registry is None
        assert router._initialized is False

    def test_init_with_registry(self):
        reg = MagicMock()
        router = _make_router(registry=reg)
        assert router._registry is reg

    def test_set_registry(self):
        router = _make_router()
        reg = MagicMock()
        router.set_registry(reg)
        assert router._registry is reg

    @pytest.mark.asyncio
    async def test_initialize_sets_flag(self):
        router = _make_router()
        assert not router._initialized
        await router.initialize()
        assert router._initialized

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        router = _make_router()
        await router.initialize()
        await router.initialize()
        assert router._initialized


# ==========================================================================
# _check_exit_signal
# ==========================================================================

class TestCheckExitSignal:
    def test_basic_exit_signals(self):
        router = _make_router()
        assert router._check_exit_signal("odustani od rezervacije") is True
        assert router._check_exit_signal("STOP!") is True
        assert router._check_exit_signal("cancel please") is True

    def test_continue_signals_override_exit(self):
        """'pokazi ostala' should NOT be treated as exit."""
        router = _make_router()
        assert router._check_exit_signal("pokaži ostala vozila") is False
        assert router._check_exit_signal("sva vozila") is False
        assert router._check_exit_signal("popis vozila") is False

    def test_neutral_query_not_exit(self):
        router = _make_router()
        assert router._check_exit_signal("koliko imam kilometara") is False
        assert router._check_exit_signal("rezerviraj vozilo") is False


# ==========================================================================
# _check_greeting
# ==========================================================================

class TestCheckGreeting:
    def test_exact_greeting(self):
        router = _make_router()
        resp = router._check_greeting("bok")
        assert resp is not None
        assert "Bok" in resp

    def test_greeting_with_prefix(self):
        router = _make_router()
        resp = router._check_greeting("bok svima")
        assert resp is not None

    def test_greeting_dobar_dan(self):
        router = _make_router()
        resp = router._check_greeting("dobar dan")
        assert "Dobar dan" in resp

    def test_hvala_response(self):
        router = _make_router()
        resp = router._check_greeting("hvala")
        assert resp is not None
        assert "Nema na" in resp

    def test_help_response(self):
        router = _make_router()
        resp = router._check_greeting("help")
        assert "Mogu vam" in resp

    def test_not_a_greeting(self):
        router = _make_router()
        assert router._check_greeting("koliko imam km") is None
        assert router._check_greeting("") is None

    def test_case_insensitive(self):
        router = _make_router()
        resp = router._check_greeting("BOK")
        assert resp is not None


# ==========================================================================
# _get_relevant_tools_with_ambiguity
# ==========================================================================

class TestGetRelevantToolsWithAmbiguity:
    @pytest.mark.asyncio
    async def test_no_registry_returns_primary(self):
        router = _make_router()
        router._registry = None
        tools, amb = await router._get_relevant_tools_with_ambiguity("test query")
        assert tools == PRIMARY_TOOLS
        assert amb is None

    @pytest.mark.asyncio
    async def test_registry_not_ready_returns_primary(self):
        router = _make_router()
        reg = MagicMock()
        reg.is_ready = False
        router._registry = reg
        tools, amb = await router._get_relevant_tools_with_ambiguity("test query")
        assert tools == PRIMARY_TOOLS
        assert amb is None

    @pytest.mark.asyncio
    async def test_search_returns_results_merged_with_primary(self):
        router = _make_router()
        reg = MagicMock()
        reg.is_ready = True
        router._registry = reg

        # Mock UnifiedSearch
        mock_result = MagicMock()
        mock_result.tool_id = "get_CustomTool"
        mock_result.description = "Custom tool"
        mock_result.score = 0.9

        mock_response = MagicMock()
        mock_response.results = [mock_result]
        mock_response.intent = MagicMock()
        mock_response.intent.value = "info"

        mock_search = MagicMock()
        mock_search.set_registry = MagicMock()
        mock_search.search = AsyncMock(return_value=mock_response)

        mock_amb_result = MagicMock()
        mock_amb_result.is_ambiguous = False

        mock_detector = MagicMock()
        mock_detector.detect_ambiguity = MagicMock(return_value=mock_amb_result)

        with patch("services.unified_search.get_unified_search", return_value=mock_search):
            with patch("services.unified_router.get_ambiguity_detector", return_value=mock_detector):
                tools, amb = await router._get_relevant_tools_with_ambiguity("test")

        assert "get_CustomTool" in tools
        # PRIMARY_TOOLS should also be merged in
        for key in PRIMARY_TOOLS:
            assert key in tools

    @pytest.mark.asyncio
    async def test_search_exception_falls_back_to_primary(self):
        router = _make_router()
        reg = MagicMock()
        reg.is_ready = True
        router._registry = reg

        with patch("services.unified_search.get_unified_search", side_effect=Exception("boom")):
            tools, amb = await router._get_relevant_tools_with_ambiguity("test")

        assert tools == PRIMARY_TOOLS
        assert amb is None

    @pytest.mark.asyncio
    async def test_search_empty_results_falls_back(self):
        router = _make_router()
        reg = MagicMock()
        reg.is_ready = True
        router._registry = reg

        mock_response = MagicMock()
        mock_response.results = []

        mock_search = MagicMock()
        mock_search.set_registry = MagicMock()
        mock_search.search = AsyncMock(return_value=mock_response)

        with patch("services.unified_search.get_unified_search", return_value=mock_search):
            tools, amb = await router._get_relevant_tools_with_ambiguity("test")

        assert tools == PRIMARY_TOOLS
        assert amb is None


class TestGetRelevantTools:
    @pytest.mark.asyncio
    async def test_wrapper_returns_tools_only(self):
        router = _make_router()
        router._registry = None
        tools = await router._get_relevant_tools("test")
        assert tools == PRIMARY_TOOLS


# ==========================================================================
# route() - greeting path
# ==========================================================================

class TestRouteGreeting:
    @pytest.mark.asyncio
    async def test_greeting_returns_direct_response(self):
        router = _make_router()
        result = await router.route("bok", _user_context())
        assert result.action == "direct_response"
        assert result.confidence == 1.0
        assert "Bok" in result.response


# ==========================================================================
# route() - exit signal in flow
# ==========================================================================

class TestRouteExitSignal:
    @pytest.mark.asyncio
    async def test_exit_signal_in_flow(self):
        router = _make_router()
        state = {"flow": "booking", "state": "collecting"}
        result = await router.route("odustani", _user_context(), state)
        assert result.action == "exit_flow"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_exit_signal_not_in_flow_is_ignored(self):
        """Exit signal without active flow should not return exit_flow."""
        router = _make_router()
        # Mock query router to return no match, then mock LLM
        router.query_router.route.return_value = _route_result(matched=False, confidence=0.5)
        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "simple_api",
                "tool": "get_MasterData",
                "params": {},
                "reasoning": "test",
                "confidence": 0.8,
            })
        )
        # No conversation_state, so not in flow
        result = await router.route("odustani", _user_context(), None)
        # Should not be exit_flow since there's no active flow
        assert result.action != "exit_flow" or result.action == "direct_response"


# ==========================================================================
# route() - in-flow continue signals
# ==========================================================================

class TestRouteInFlowContinue:
    @pytest.mark.asyncio
    async def test_show_more_in_flow(self):
        router = _make_router()
        state = {"flow": "booking", "state": "selecting"}
        result = await router.route("pokaži ostala", _user_context(), state)
        assert result.action == "continue_flow"

    @pytest.mark.asyncio
    async def test_confirmation_yes_in_confirming(self):
        router = _make_router()
        state = {"flow": "booking", "state": "confirming"}
        result = await router.route("da", _user_context(), state)
        assert result.action == "continue_flow"

    @pytest.mark.asyncio
    async def test_confirmation_no_in_confirming(self):
        router = _make_router()
        state = {"flow": "booking", "state": "confirming"}
        result = await router.route("ne", _user_context(), state)
        assert result.action == "continue_flow"

    @pytest.mark.asyncio
    async def test_numeric_selection_in_selecting(self):
        router = _make_router()
        state = {"flow": "booking", "state": "selecting"}
        result = await router.route("2", _user_context(), state)
        assert result.action == "continue_flow"

    @pytest.mark.asyncio
    async def test_confirmation_ok_in_confirming(self):
        router = _make_router()
        state = {"flow": "booking", "state": "confirming"}
        result = await router.route("ok", _user_context(), state)
        assert result.action == "continue_flow"


# ==========================================================================
# route() - QueryRouter fast path
# ==========================================================================

class TestRouteQueryRouterFastPath:
    @pytest.mark.asyncio
    async def test_qr_high_confidence_match(self):
        router = _make_router()
        qr_result = _route_result(
            matched=True,
            tool_name="get_MasterData",
            flow_type="simple",
            confidence=1.0,
            reason="ML: GET_VEHICLE_INFO",
        )
        router.query_router.route.return_value = qr_result

        result = await router.route("podaci o vozilu", _user_context())
        assert result.action == "simple_api"
        assert result.tool == "get_MasterData"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_qr_low_confidence_goes_to_llm(self):
        """Confidence < 1.0 should NOT use fast path, goes to LLM."""
        router = _make_router()
        qr_result = _route_result(matched=True, confidence=0.8)
        router.query_router.route.return_value = qr_result

        # Mock LLM response
        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "simple_api",
                "tool": "get_MasterData",
                "params": {},
                "reasoning": "LLM decided",
                "confidence": 0.9,
            })
        )

        result = await router.route("nesto", _user_context())
        # Should have gone to LLM
        router.client.chat.completions.create.assert_awaited_once()


# ==========================================================================
# _llm_route
# ==========================================================================

class TestLlmRoute:
    @pytest.mark.asyncio
    async def test_llm_simple_api_response(self):
        router = _make_router()
        router._registry = None  # Force PRIMARY_TOOLS fallback

        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "simple_api",
                "tool": "get_Expenses",
                "params": {},
                "flow_type": None,
                "reasoning": "User wants expenses",
                "confidence": 0.85,
            })
        )

        result = await router._llm_route("troškovi", _user_context(), None)
        assert result.action == "simple_api"
        assert result.tool == "get_Expenses"
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_llm_clarify_action(self):
        router = _make_router()
        router._registry = None

        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "clarify",
                "clarification": "Koji entitet zelite?",
                "reasoning": "Query is ambiguous",
                "confidence": 0.3,
            })
        )

        result = await router._llm_route("prosjecna vrijednost", _user_context(), None)
        assert result.action == "clarify"
        assert result.clarification == "Koji entitet zelite?"
        assert result.ambiguity_detected is True

    @pytest.mark.asyncio
    async def test_llm_exit_flow_without_state_converted(self):
        """If LLM returns exit_flow but no conversation_state, convert to simple_api."""
        router = _make_router()
        router._registry = None

        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "exit_flow",
                "tool": None,
                "params": {},
                "reasoning": "wrong exit",
                "confidence": 0.6,
            })
        )

        result = await router._llm_route("zapravo", _user_context(), None)
        assert result.action == "simple_api"
        assert result.tool == "get_MasterData"  # Default fallback

    @pytest.mark.asyncio
    async def test_llm_exit_flow_with_state_kept(self):
        """If LLM returns exit_flow AND there IS a conversation_state, keep it."""
        router = _make_router()
        router._registry = None

        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "exit_flow",
                "tool": None,
                "params": {},
                "reasoning": "user wants out",
                "confidence": 0.9,
            })
        )

        state = {"flow": "booking", "state": "collecting"}
        result = await router._llm_route("zapravo ne", _user_context(), state)
        assert result.action == "exit_flow"

    @pytest.mark.asyncio
    async def test_llm_exception_triggers_fallback(self):
        """LLM failure should trigger _fallback_route."""
        router = _make_router()
        router._registry = None

        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(side_effect=Exception("LLM down"))

        # Setup fallback via query_router — no match
        router.query_router.route.return_value = _route_result(
            matched=False, confidence=0.3
        )

        result = await router._llm_route("nesto", _user_context(), None)
        # Fallback with no QR match returns direct_response asking user to clarify
        assert result.action == "direct_response"
        assert result.confidence == 0.1

    @pytest.mark.asyncio
    async def test_llm_with_conversation_state(self):
        """Verify conversation state is passed through to prompt building."""
        router = _make_router()
        router._registry = None

        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "continue_flow",
                "tool": "post_VehicleCalendar",
                "params": {"date": "2024-01-15"},
                "flow_type": "booking",
                "reasoning": "continuing booking",
                "confidence": 0.9,
            })
        )

        state = {
            "flow": "booking",
            "state": "collecting",
            "missing_params": ["FromTime"],
            "tool": "post_VehicleCalendar",
        }

        result = await router._llm_route("sutra u 9", _user_context(), state)
        assert result.action == "continue_flow"
        assert result.flow_type == "booking"

    @pytest.mark.asyncio
    async def test_llm_no_vehicle_context(self):
        """Verify it handles user context without vehicle."""
        router = _make_router()
        router._registry = None

        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "simple_api",
                "tool": "get_MasterData",
                "params": {},
                "reasoning": "test",
                "confidence": 0.7,
            })
        )

        result = await router._llm_route("podaci", _user_context(with_vehicle=False), None)
        assert result.action == "simple_api"

    @pytest.mark.asyncio
    async def test_llm_start_flow_response(self):
        router = _make_router()
        router._registry = None

        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "start_flow",
                "tool": "get_AvailableVehicles",
                "params": {},
                "flow_type": "booking",
                "reasoning": "User wants to book",
                "confidence": 0.95,
            })
        )

        result = await router._llm_route("trebam auto", _user_context(), None)
        assert result.action == "start_flow"
        assert result.flow_type == "booking"

    @pytest.mark.asyncio
    async def test_llm_direct_response(self):
        router = _make_router()
        router._registry = None

        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "direct_response",
                "tool": None,
                "params": {},
                "response": "Bok! Kako vam mogu pomoci?",
                "reasoning": "Greeting",
                "confidence": 1.0,
            })
        )

        result = await router._llm_route("zdravo", _user_context(), None)
        assert result.action == "direct_response"
        assert result.response == "Bok! Kako vam mogu pomoci?"


# ==========================================================================
# _fallback_route
# ==========================================================================

class TestFallbackRoute:
    def test_fallback_with_qr_match(self):
        router = _make_router()
        qr_result = _route_result(
            matched=True,
            tool_name="get_Expenses",
            flow_type="simple",
            confidence=0.8,
            reason="ML: partial match",
        )
        router.query_router.route.return_value = qr_result

        result = router._fallback_route("troskovi", _user_context())
        assert result.action == "simple_api"
        assert result.tool == "get_Expenses"
        # Fallback path reduces confidence by 0.8
        assert result.confidence == pytest.approx(0.64)

    def test_fallback_no_qr_match(self):
        router = _make_router()
        router.query_router.route.return_value = _route_result(matched=False, confidence=0.2)

        result = router._fallback_route("nesto neprepoznatljivo", _user_context())
        # No match → direct_response asking user to clarify
        assert result.action == "direct_response"
        assert result.confidence == 0.1


# ==========================================================================
# _query_result_to_decision
# ==========================================================================

class TestQueryResultToDecision:
    def test_direct_response_flow(self):
        router = _make_router()
        qr = _route_result(
            flow_type="direct_response",
            response_template="Pozdrav! Kako vam mogu pomoci?",
            tool_name=None,
        )
        result = router._query_result_to_decision(qr, _user_context())
        assert result.action == "direct_response"
        assert "Pozdrav" in result.response

    def test_direct_response_with_person_id(self):
        router = _make_router()
        with patch("services.unified_router.get_settings", return_value=_mock_settings()):
            qr = _route_result(
                flow_type="direct_response",
                response_template="**Person ID:** {person_id}",
                tool_name=None,
            )
            result = router._query_result_to_decision(qr, _user_context())
            assert result.action == "direct_response"
            assert "Person ID" in result.response

    def test_direct_response_with_phone(self):
        router = _make_router()
        with patch("services.unified_router.get_settings", return_value=_mock_settings()):
            qr = _route_result(
                flow_type="direct_response",
                response_template="**Telefon:** {phone}",
                tool_name=None,
            )
            result = router._query_result_to_decision(qr, _user_context())
            assert result.action == "direct_response"
            assert "Telefon" in result.response

    def test_direct_response_with_tenant_id(self):
        router = _make_router()
        with patch("services.unified_router.get_settings", return_value=_mock_settings()):
            qr = _route_result(
                flow_type="direct_response",
                response_template="**Tenant ID:** {tenant_id}",
                tool_name=None,
            )
            result = router._query_result_to_decision(qr, _user_context())
            assert result.action == "direct_response"
            assert "Tenant ID" in result.response

    def test_booking_flow(self):
        router = _make_router()
        qr = _route_result(
            flow_type="booking",
            tool_name="get_AvailableVehicles",
        )
        result = router._query_result_to_decision(qr, _user_context())
        assert result.action == "start_flow"
        assert result.flow_type == "booking"
        assert result.tool == "get_AvailableVehicles"

    def test_mileage_input_flow(self):
        router = _make_router()
        qr = _route_result(
            flow_type="mileage_input",
            tool_name="post_AddMileage",
        )
        result = router._query_result_to_decision(qr, _user_context())
        assert result.action == "start_flow"
        assert result.flow_type == "mileage"

    def test_case_creation_flow(self):
        router = _make_router()
        qr = _route_result(
            flow_type="case_creation",
            tool_name="post_AddCase",
        )
        result = router._query_result_to_decision(qr, _user_context())
        assert result.action == "start_flow"
        assert result.flow_type == "case"

    def test_simple_api(self):
        router = _make_router()
        qr = _route_result(
            flow_type="simple",
            tool_name="get_Expenses",
        )
        result = router._query_result_to_decision(qr, _user_context())
        assert result.action == "simple_api"
        assert result.tool == "get_Expenses"

    def test_list_flow_type(self):
        router = _make_router()
        qr = _route_result(
            flow_type="list",
            tool_name="get_VehicleCalendar",
        )
        result = router._query_result_to_decision(qr, _user_context())
        assert result.action == "simple_api"
        assert result.flow_type == "list"

    def test_fallback_flag_reduces_confidence(self):
        router = _make_router()
        qr = _route_result(confidence=1.0, flow_type="simple", tool_name="get_MasterData")
        result = router._query_result_to_decision(qr, _user_context(), is_fallback=True)
        assert result.confidence == pytest.approx(0.8)

    def test_non_fallback_keeps_confidence(self):
        router = _make_router()
        qr = _route_result(confidence=1.0, flow_type="simple", tool_name="get_MasterData")
        result = router._query_result_to_decision(qr, _user_context(), is_fallback=False)
        assert result.confidence == 1.0

    def test_direct_response_format_failure_returns_template(self):
        """If format() fails, response_template should be returned as-is."""
        router = _make_router()
        qr = _route_result(
            flow_type="direct_response",
            response_template="Hello {unknown_key}!",
            tool_name=None,
        )
        result = router._query_result_to_decision(qr, _user_context())
        assert result.action == "direct_response"
        # Should not crash, returns template as-is or formatted
        assert result.response is not None

    def test_direct_response_no_user_context(self):
        router = _make_router()
        qr = _route_result(
            flow_type="direct_response",
            response_template="Pozdrav!",
            tool_name=None,
        )
        result = router._query_result_to_decision(qr, None)
        assert result.action == "direct_response"
        assert result.response == "Pozdrav!"


# ==========================================================================
# get_unified_router singleton
# ==========================================================================

class TestGetUnifiedRouter:
    @pytest.mark.asyncio
    async def test_singleton_creation(self):
        import services.unified_router as mod
        mod._router = None  # Reset singleton

        with patch("services.unified_router.get_settings", return_value=_mock_settings()):
            with patch("services.unified_router.get_openai_client", return_value=MagicMock()):
                with patch("services.unified_router.get_llm_circuit_breaker", return_value=MagicMock()):
                    with patch("services.unified_router.QueryRouter"):
                        r = await get_unified_router()
                        assert r is not None
                        assert r._initialized is True

                        # Second call returns same instance
                        r2 = await get_unified_router()
                        assert r2 is r

        # Cleanup
        mod._router = None

    @pytest.mark.asyncio
    async def test_singleton_reset(self):
        import services.unified_router as mod
        mod._router = None

        with patch("services.unified_router.get_settings", return_value=_mock_settings()):
            with patch("services.unified_router.get_openai_client", return_value=MagicMock()):
                with patch("services.unified_router.get_llm_circuit_breaker", return_value=MagicMock()):
                    with patch("services.unified_router.QueryRouter"):
                        r1 = await get_unified_router()
                        mod._router = None
                        r2 = await get_unified_router()
                        assert r1 is not r2

        mod._router = None


# ==========================================================================
# route() integration with LLM path
# ==========================================================================

class TestRouteFullLLMPath:
    @pytest.mark.asyncio
    async def test_full_route_through_llm(self):
        """Non-greeting, non-exit, non-flow query with low QR confidence goes to LLM."""
        router = _make_router()
        router._registry = None

        # QR returns low confidence
        router.query_router.route.return_value = _route_result(
            matched=True, confidence=0.7
        )

        # LLM responds
        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "simple_api",
                "tool": "get_Trips",
                "params": {},
                "reasoning": "User wants trips",
                "confidence": 0.88,
            })
        )

        result = await router.route("moji putovanja", _user_context())
        assert result.action == "simple_api"
        assert result.tool == "get_Trips"

    @pytest.mark.asyncio
    async def test_route_llm_failure_uses_fallback(self):
        """If LLM fails, fallback through QR / ultimate fallback."""
        router = _make_router()
        router._registry = None

        # QR returns no match at step 4
        router.query_router.route.return_value = _route_result(
            matched=False, confidence=0.3
        )

        # LLM throws exception
        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(side_effect=Exception("timeout"))

        result = await router.route("nesto nepoznato", _user_context())
        # No QR match → ultimate fallback asks user to clarify
        assert result.action == "direct_response"
        assert result.confidence == 0.1


# ==========================================================================
# Edge cases
# ==========================================================================

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_query_not_greeting(self):
        router = _make_router()
        # Empty string should not match any greeting
        resp = router._check_greeting("")
        assert resp is None

    def test_exit_signal_mixed_case(self):
        router = _make_router()
        assert router._check_exit_signal("ODUSTANI") is True
        assert router._check_exit_signal("Cancel") is True

    @pytest.mark.asyncio
    async def test_ambiguity_detected_flag_set(self):
        """When ambiguity is detected, the flag should be set on result."""
        router = _make_router()
        reg = MagicMock()
        reg.is_ready = True
        router._registry = reg

        mock_result = MagicMock()
        mock_result.tool_id = "get_Vehicles_Agg"
        mock_result.description = "Aggregation"
        mock_result.score = 0.9

        mock_response = MagicMock()
        mock_response.results = [mock_result]
        mock_response.intent = MagicMock()
        mock_response.intent.value = "info"

        mock_search = MagicMock()
        mock_search.set_registry = MagicMock()
        mock_search.search = AsyncMock(return_value=mock_response)

        mock_amb_result = MagicMock()
        mock_amb_result.is_ambiguous = True
        mock_amb_result.similar_tools = ["get_Vehicles_Agg", "get_Expenses_Agg"]
        mock_amb_result.disambiguation_hint = "AGGREGACIJA"
        mock_amb_result.clarification_question = "Koji entitet?"

        mock_detector = MagicMock()
        mock_detector.detect_ambiguity = MagicMock(return_value=mock_amb_result)

        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "simple_api",
                "tool": "get_Vehicles_Agg",
                "params": {},
                "reasoning": "agg",
                "confidence": 0.7,
            })
        )

        with patch("services.unified_search.get_unified_search", return_value=mock_search):
            with patch("services.unified_router.get_ambiguity_detector", return_value=mock_detector):
                result = await router._llm_route("prosjecna", _user_context(), None)

        assert result.ambiguity_detected is True

    @pytest.mark.asyncio
    async def test_clarify_without_clarification_uses_detector(self):
        """If LLM returns clarify without text, use detector's question."""
        router = _make_router()
        reg = MagicMock()
        reg.is_ready = True
        router._registry = reg

        mock_result = MagicMock()
        mock_result.tool_id = "get_Vehicles_Agg"
        mock_result.description = "Agg"
        mock_result.score = 0.9

        mock_response = MagicMock()
        mock_response.results = [mock_result]
        mock_response.intent = MagicMock()
        mock_response.intent.value = "info"

        mock_search = MagicMock()
        mock_search.set_registry = MagicMock()
        mock_search.search = AsyncMock(return_value=mock_response)

        mock_amb_result = MagicMock()
        mock_amb_result.is_ambiguous = True
        mock_amb_result.similar_tools = ["a", "b", "c"]
        mock_amb_result.disambiguation_hint = "hint"
        mock_amb_result.clarification_question = "Fallback question from detector"

        mock_detector = MagicMock()
        mock_detector.detect_ambiguity = MagicMock(return_value=mock_amb_result)

        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "clarify",
                # No "clarification" key
                "reasoning": "ambiguous",
                "confidence": 0.3,
            })
        )

        with patch("services.unified_search.get_unified_search", return_value=mock_search):
            with patch("services.unified_router.get_ambiguity_detector", return_value=mock_detector):
                result = await router._llm_route("prosjecna", _user_context(), None)

        assert result.action == "clarify"
        assert result.clarification == "Fallback question from detector"

    @pytest.mark.asyncio
    async def test_llm_missing_fields_use_defaults(self):
        """LLM response missing optional fields should use defaults."""
        router = _make_router()
        router._registry = None

        router.client = MagicMock()
        router.client.chat = MagicMock()
        router.client.chat.completions = MagicMock()
        router.client.chat.completions.create = AsyncMock(
            return_value=_llm_response({
                "action": "simple_api",
                "tool": "get_MasterData",
                # No params, no reasoning, no confidence
            })
        )

        result = await router._llm_route("test", _user_context(), None)
        assert result.action == "simple_api"
        assert result.params == {}
        assert result.confidence == 0.5  # default
        assert result.reasoning == ""

    @pytest.mark.asyncio
    async def test_in_flow_visa_detected(self):
        """'vise opcija' in flow should be continue_flow."""
        router = _make_router()
        state = {"flow": "booking", "state": "selecting"}
        result = await router.route("vise opcija", _user_context(), state)
        assert result.action == "continue_flow"

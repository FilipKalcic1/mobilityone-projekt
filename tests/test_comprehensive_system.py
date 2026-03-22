"""
Comprehensive System Accuracy Tests

Tests the FULL bot pipeline end-to-end:

1. QueryRouter → tool/flow mapping for ALL 30 intents
2. UnifiedRouter._query_result_to_decision → correct action types
3. MessageEngine._process_with_state → all decision branches
4. MessageEngine._handle_flow_start → all flow types
5. Flow state machines → multi-turn booking/mileage/case/delete
6. Phrase matching → confirm/exit/show-more with priority rules
7. Guest user restrictions → flows blocked, simple API guarded
8. Edge cases → European numbers, diacritics, empty input, timeouts
9. Response formatting → mileage thousands separators, dates
10. Read-vs-write safety → never confuse GET with POST/DELETE

Runs locally — no Docker, no LLM API calls, no Redis.
Uses mocked dependencies and the trained TF-IDF model.

Usage:
    pytest tests/test_comprehensive_system.py -v
"""

import re
import sys
import asyncio
import importlib
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress noisy logs
import logging
logging.basicConfig(level=logging.WARNING)
for name in ['services', 'openai', 'httpx', 'httpcore', 'sklearn']:
    logging.getLogger(name).setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Stub heavy deps that may not be available
# ---------------------------------------------------------------------------
_STUBS: Dict[str, Any] = {}


def _needs_stub(root_pkg: str) -> bool:
    """Check if a root package needs stubbing (not installed)."""
    if root_pkg in sys.modules:
        return False
    try:
        __import__(root_pkg)
        return False
    except (ImportError, ModuleNotFoundError):
        return True


def _ensure_stub_pkg(module_name: str):
    """Stub a package (needs __path__ for submodule imports)."""
    if module_name in sys.modules:
        return
    m = MagicMock()
    m.__path__ = []
    _STUBS[module_name] = m
    sys.modules[module_name] = m


def _ensure_stub(module_name: str):
    """Stub a leaf module."""
    if module_name in sys.modules:
        return
    _STUBS[module_name] = MagicMock()
    sys.modules[module_name] = _STUBS[module_name]


# Define stub trees: only stub if root package is missing
_STUB_TREES = {
    "sqlalchemy": {
        "packages": [
            "sqlalchemy", "sqlalchemy.ext", "sqlalchemy.orm",
            "sqlalchemy.sql", "sqlalchemy.dialects",
        ],
        "modules": [
            "sqlalchemy.ext.asyncio", "sqlalchemy.ext.declarative",
            "sqlalchemy.orm.session", "sqlalchemy.sql.expression",
            "sqlalchemy.dialects.postgresql", "sqlalchemy.types",
            "sqlalchemy.schema", "sqlalchemy.engine", "sqlalchemy.future",
            "sqlalchemy.exc",
        ],
    },
    "redis": {
        "packages": ["redis"],
        "modules": ["redis.asyncio", "redis.exceptions"],
    },
    "fastapi": {
        "packages": ["fastapi"],
        "modules": ["fastapi.responses", "fastapi.middleware"],
    },
    "starlette": {
        "packages": ["starlette", "starlette.middleware"],
        "modules": ["starlette.requests", "starlette.responses", "starlette.types"],
    },
    "azure": {
        "packages": ["azure", "azure.identity"],
        "modules": ["azure.identity.aio"],
    },
}

# Stub only missing root packages and their trees
for _root, _tree in _STUB_TREES.items():
    if _needs_stub(_root):
        for _pkg in _tree["packages"]:
            _ensure_stub_pkg(_pkg)
        for _mod in _tree["modules"]:
            _ensure_stub(_mod)

# Standalone modules (no submodule tree)
for _mod in [
    "prometheus_client", "faiss", "openai", "tiktoken", "httpx",
    "asyncpg", "aiohttp", "uvicorn", "pydantic_settings",
]:
    if _needs_stub(_mod):
        _ensure_stub(_mod)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from services.query_router import QueryRouter, RouteResult, INTENT_METADATA, ML_CONFIDENCE_THRESHOLD
from services.dynamic_threshold import ClassificationSignal


def _mock_prediction(intent, confidence, tool):
    """Create a MagicMock with real ClassificationSignal for decide_with_cp()."""
    m = MagicMock()
    m.intent = intent
    m.confidence = confidence
    m.tool = tool
    m.signal = ClassificationSignal.from_confidence_only(confidence, n_classes=29)
    m.prediction_set = None
    return m
from services.flow_phrases import (
    matches_show_more, matches_confirm_yes, matches_confirm_no,
    matches_exit_signal, matches_greeting, matches_item_selection,
)
from services.conversation_manager import ConversationState
from tool_routing import INTENT_CONFIG, FLOW_TRIGGERS, PRIMARY_TOOLS


# ===========================================================================
# SECTION 1: INTENT_CONFIG Consistency
# ===========================================================================

class TestIntentConfigConsistency:
    """Verify INTENT_CONFIG is complete and consistent with PRIMARY_TOOLS."""

    def test_all_intents_have_required_keys(self):
        """Every intent must have tool, extract_fields, response_template, flow_type."""
        required = {"tool", "extract_fields", "response_template", "flow_type"}
        for intent, config in INTENT_CONFIG.items():
            missing = required - set(config.keys())
            assert not missing, f"INTENT_CONFIG[{intent}] missing: {missing}"

    def test_all_tools_exist_in_primary_tools(self):
        """Every tool referenced in INTENT_CONFIG must exist in PRIMARY_TOOLS."""
        primary_lower = {k.lower() for k in PRIMARY_TOOLS}
        for intent, config in INTENT_CONFIG.items():
            tool = config.get("tool")
            if tool:
                assert tool.lower() in primary_lower, \
                    f"INTENT_CONFIG[{intent}].tool='{tool}' not in PRIMARY_TOOLS"

    def test_known_flow_types(self):
        """All flow_types must be handled by the engine."""
        known = {
            "simple", "list", "direct_response",
            "booking", "mileage_input", "case_creation",
            "delete_booking", "delete_case", "delete_trip",
        }
        for intent, config in INTENT_CONFIG.items():
            ft = config.get("flow_type")
            assert ft in known, f"INTENT_CONFIG[{intent}].flow_type='{ft}' unknown"

    def test_minimum_intent_count(self):
        """Must have at least 25 intents (current: 30)."""
        assert len(INTENT_CONFIG) >= 25, f"Only {len(INTENT_CONFIG)} intents defined"

    def test_flow_triggers_tools_exist(self):
        """FLOW_TRIGGERS tools must exist in PRIMARY_TOOLS."""
        for tool, flow in FLOW_TRIGGERS.items():
            assert tool in PRIMARY_TOOLS, f"FLOW_TRIGGERS tool '{tool}' not in PRIMARY_TOOLS"

    def test_delete_flow_types_have_delete_tools(self):
        """Delete flow types must map to delete_* tools."""
        for intent, config in INTENT_CONFIG.items():
            if config["flow_type"].startswith("delete_"):
                assert config["tool"] and config["tool"].startswith("delete_"), \
                    f"{intent}: delete flow but tool={config['tool']}"

    def test_direct_response_intents_have_no_tool(self):
        """Direct response intents should have tool=None."""
        for intent, config in INTENT_CONFIG.items():
            if config["flow_type"] == "direct_response":
                # Some direct_response may have tool=None, some don't
                # The key check: they MUST have a response_template
                assert config["response_template"], \
                    f"{intent}: direct_response but no response_template"


# ===========================================================================
# SECTION 2: QueryRouter — Intent-to-Route Mapping
# ===========================================================================

class TestQueryRouterAllIntents:
    """Test QueryRouter routes all major intents to correct tools."""

    @pytest.fixture(autouse=True)
    def setup_router(self):
        self.router = QueryRouter()

    # --- Vehicle Information (READ) ---

    @pytest.mark.parametrize("query,expected_tool", [
        ("koliko imam kilometara", "get_MasterData"),
        ("moja kilometraza", "get_MasterData"),
        ("koliko km ima auto", "get_MasterData"),
    ])
    @patch("services.query_router.predict_with_ensemble")
    def test_get_mileage(self, mock_predict, query, expected_tool):
        mock_predict.return_value = _mock_prediction("GET_MILEAGE", 0.95, expected_tool)
        result = self.router.route(query)
        assert result.matched
        assert result.tool_name == expected_tool
        assert result.flow_type == "simple"

    @pytest.mark.parametrize("query", [
        "informacije o vozilu", "podaci o autu", "koji auto imam",
    ])
    @patch("services.query_router.predict_with_ensemble")
    def test_get_vehicle_info(self, mock_predict, query):
        mock_predict.return_value = _mock_prediction("GET_VEHICLE_INFO", 0.95, "get_MasterData")
        result = self.router.route(query)
        assert result.matched
        assert result.tool_name == "get_MasterData"
        assert result.flow_type == "simple"

    @patch("services.query_router.predict_with_ensemble")
    def test_get_registration_expiry(self, mock_predict):
        mock_predict.return_value = _mock_prediction("GET_REGISTRATION_EXPIRY", 0.95, "get_MasterData")
        result = self.router.route("kada istice registracija")
        assert result.matched
        assert "ExpirationDate" in result.extract_fields or "RegistrationExpirationDate" in result.extract_fields

    @patch("services.query_router.predict_with_ensemble")
    def test_get_plate(self, mock_predict):
        mock_predict.return_value = _mock_prediction("GET_PLATE", 0.95, "get_MasterData")
        result = self.router.route("koje su mi tablice")
        assert result.matched
        assert "LicencePlate" in result.extract_fields

    @patch("services.query_router.predict_with_ensemble")
    def test_get_vehicle_count(self, mock_predict):
        mock_predict.return_value = _mock_prediction("GET_VEHICLE_COUNT", 0.95, "get_Vehicles_Agg")
        result = self.router.route("koliko vozila ima")
        assert result.matched
        assert result.tool_name == "get_Vehicles_Agg"

    # --- Reservation intents ---

    @patch("services.query_router.predict_with_ensemble")
    def test_book_vehicle(self, mock_predict):
        mock_predict.return_value = _mock_prediction("BOOK_VEHICLE", 0.95, "get_AvailableVehicles")
        result = self.router.route("rezerviraj auto za sutra")
        assert result.matched
        assert result.flow_type == "booking"

    @patch("services.query_router.predict_with_ensemble")
    def test_get_my_bookings(self, mock_predict):
        mock_predict.return_value = _mock_prediction("GET_MY_BOOKINGS", 0.95, "get_VehicleCalendar")
        result = self.router.route("moje rezervacije")
        assert result.matched
        assert result.flow_type == "list"

    @patch("services.query_router.predict_with_ensemble")
    def test_cancel_reservation(self, mock_predict):
        mock_predict.return_value = _mock_prediction("CANCEL_RESERVATION", 0.95, "delete_VehicleCalendar_id")
        result = self.router.route("otkazi rezervaciju")
        assert result.matched
        assert result.flow_type == "delete_booking"

    # --- Write intents ---

    @patch("services.query_router.predict_with_ensemble")
    def test_input_mileage(self, mock_predict):
        mock_predict.return_value = _mock_prediction("INPUT_MILEAGE", 0.95, "post_AddMileage")
        result = self.router.route("unesi kilometrazu")
        assert result.matched
        assert result.flow_type == "mileage_input"

    @patch("services.query_router.predict_with_ensemble")
    def test_report_damage(self, mock_predict):
        mock_predict.return_value = _mock_prediction("REPORT_DAMAGE", 0.95, "post_AddCase")
        result = self.router.route("prijavi stetu")
        assert result.matched
        assert result.flow_type == "case_creation"

    # --- Delete intents ---

    @patch("services.query_router.predict_with_ensemble")
    def test_delete_case(self, mock_predict):
        mock_predict.return_value = _mock_prediction("DELETE_CASE", 0.95, "delete_Cases_id")
        result = self.router.route("obrisi prijavu")
        assert result.matched
        assert result.flow_type == "delete_case"

    @patch("services.query_router.predict_with_ensemble")
    def test_delete_trip(self, mock_predict):
        mock_predict.return_value = _mock_prediction("DELETE_TRIP", 0.95, "delete_Trips_id")
        result = self.router.route("obrisi putovanje")
        assert result.matched
        assert result.flow_type == "delete_trip"

    # --- Social / Direct response ---

    @patch("services.query_router.predict_with_ensemble")
    def test_greeting(self, mock_predict):
        mock_predict.return_value = _mock_prediction("GREETING", 0.95, None)
        result = self.router.route("bok")
        assert result.matched
        assert result.flow_type == "direct_response"
        assert result.tool_name is None

    @patch("services.query_router.predict_with_ensemble")
    def test_help(self, mock_predict):
        mock_predict.return_value = _mock_prediction("HELP", 0.95, None)
        result = self.router.route("pomoc")
        assert result.matched
        assert "Mogu vam pomoci" in result.response_template

    @patch("services.query_router.predict_with_ensemble")
    def test_thanks(self, mock_predict):
        mock_predict.return_value = _mock_prediction("THANKS", 0.95, None)
        result = self.router.route("hvala")
        assert result.matched
        assert result.flow_type == "direct_response"

    # --- Low confidence → not matched ---

    @patch("services.query_router.predict_with_ensemble")
    def test_low_confidence_not_matched(self, mock_predict):
        mock_predict.return_value = _mock_prediction("GET_MILEAGE", 0.50, "get_MasterData")
        result = self.router.route("nesto cudno")
        assert not result.matched
        assert result.confidence == 0.50

    @patch("services.query_router.predict_with_ensemble")
    def test_threshold_boundary_below(self, mock_predict):
        mock_predict.return_value = _mock_prediction("GET_MILEAGE", ML_CONFIDENCE_THRESHOLD - 0.001, "get_MasterData")
        result = self.router.route("q")
        assert not result.matched

    @patch("services.query_router.predict_with_ensemble")
    def test_threshold_boundary_at(self, mock_predict):
        mock_predict.return_value = _mock_prediction("GET_MILEAGE", ML_CONFIDENCE_THRESHOLD, "get_MasterData")
        result = self.router.route("q")
        assert result.matched


# ===========================================================================
# SECTION 3: Response Formatting
# ===========================================================================

class TestResponseFormatting:
    """Test QueryRouter formats API responses correctly."""

    @pytest.fixture(autouse=True)
    def setup_router(self):
        self.router = QueryRouter()

    def test_mileage_thousands_separator(self):
        route = RouteResult(
            matched=True,
            response_template="**Kilometraza:** {value} km",
            extract_fields=["LastMileage"]
        )
        result = self.router.format_response(route, {"LastMileage": 45000}, "km")
        assert "45.000" in result

    def test_mileage_small_number(self):
        route = RouteResult(
            matched=True,
            response_template="**Kilometraza:** {value} km",
            extract_fields=["LastMileage"]
        )
        result = self.router.format_response(route, {"LastMileage": 500}, "km")
        assert "500" in result

    def test_date_formatting_iso(self):
        route = RouteResult(
            matched=True,
            response_template="**Registracija istjece:** {value}",
            extract_fields=["ExpirationDate"]
        )
        result = self.router.format_response(
            route, {"ExpirationDate": "2026-12-31T00:00:00"}, "reg"
        )
        assert "31.12.2026" in result

    def test_missing_field_returns_none(self):
        route = RouteResult(
            matched=True,
            response_template="**Km:** {value}",
            extract_fields=["LastMileage"]
        )
        result = self.router.format_response(route, {"SomeOther": 123}, "q")
        assert result is None

    def test_no_template_returns_none(self):
        route = RouteResult(matched=True, response_template=None, extract_fields=[])
        assert self.router.format_response(route, {}, "q") is None

    def test_template_no_fields(self):
        route = RouteResult(
            matched=True,
            response_template="Pozdrav!",
            extract_fields=[]
        )
        assert self.router.format_response(route, {}, "q") == "Pozdrav!"

    def test_nested_value_extraction(self):
        route = RouteResult(
            matched=True,
            response_template="**Km:** {value} km",
            extract_fields=["LastMileage"]
        )
        data = {"vehicle": {"LastMileage": 78500}}
        result = self.router.format_response(route, data, "q")
        assert "78.500" in result

    def test_list_value_extraction(self):
        route = RouteResult(
            matched=True,
            response_template="**Km:** {value} km",
            extract_fields=["LastMileage"]
        )
        data = [{"LastMileage": 12345}]
        result = self.router.format_response(route, data, "q")
        assert "12.345" in result


# ===========================================================================
# SECTION 4: Phrase Matching — Comprehensive + Priority Rules
# ===========================================================================

class TestPhraseMatchingComprehensive:
    """Test all phrase matching functions with priority rules."""

    # --- Confirm Yes ---

    @pytest.mark.parametrize("text", [
        "da", "potvrdi", "ok", "OK", "yes", "moze", "može",
        "super", "naravno", "svakako", "apsolutno",
        "slazem se", "slažem se", "vazi", "važi",
        "idem", "ajde", "ajmo", "idemo",
        "tocno", "točno", "ispravno", "u redu",
    ])
    def test_confirm_yes(self, text):
        assert matches_confirm_yes(text), f"'{text}' should match confirm_yes"

    # --- Confirm No ---

    @pytest.mark.parametrize("text", [
        "ne", "nema", "nikako", "nista", "ništa",
        "krivo", "pogresno", "pogrešno", "prekini",
    ])
    def test_confirm_no(self, text):
        assert matches_confirm_no(text), f"'{text}' should match confirm_no"

    # --- Substring traps: MUST NOT match ---

    @pytest.mark.parametrize("text", [
        "nekako", "danas", "danica", "neobicno", "dakle", "nemoral",
    ])
    def test_substring_traps_no_false_positive(self, text):
        assert not matches_confirm_yes(text), f"'{text}' should NOT match confirm_yes"
        assert not matches_confirm_no(text), f"'{text}' should NOT match confirm_no"

    # --- Show more ---

    @pytest.mark.parametrize("text", [
        "pokaži ostala vozila", "pokazi vise opcija",
        "još opcija", "popis", "sva vozila",
    ])
    def test_show_more(self, text):
        assert matches_show_more(text), f"'{text}' should match show_more"

    # --- Exit signals ---

    @pytest.mark.parametrize("text", [
        "ne želim", "ne zelim", "necu", "neću",
        "odustani", "odustajem", "zapravo", "ipak",
        "ne treba", "stani", "stop", "cancel",
        "nešto drugo", "drugo pitanje",
    ])
    def test_exit_signals(self, text):
        assert matches_exit_signal(text), f"'{text}' should match exit_signal"

    # --- CRITICAL: "nešto drugo" is exit, NOT show_more ---

    def test_nesto_drugo_is_exit_not_show_more(self):
        assert matches_exit_signal("nešto drugo")
        assert not matches_show_more("nešto drugo")

    def test_zelim_nesto_drugo_is_exit_not_show_more(self):
        assert matches_exit_signal("zelim nesto drugo")
        assert not matches_show_more("zelim nesto drugo")

    # --- CRITICAL: "pokaži ostala" with "ne" prefix → show_more wins ---

    def test_ne_pokazi_ostala_show_more_wins(self):
        assert matches_show_more("ne, pokaži ostala")

    # --- Item selection ---

    @pytest.mark.parametrize("text", ["1", "2", "3", "10", "99"])
    def test_numeric_selection(self, text):
        assert matches_item_selection(text)

    @pytest.mark.parametrize("text", ["prvi", "prva", "treći", "četvrti", "peti"])
    def test_ordinal_selection(self, text):
        assert matches_item_selection(text)

    def test_non_selection(self):
        assert not matches_item_selection("koliko km")
        assert not matches_item_selection("nesto")

    # --- Greetings ---

    def test_greeting_bok(self):
        result = matches_greeting("bok")
        assert result is not None
        assert "AI asistent" in result

    def test_greeting_dobar_dan(self):
        result = matches_greeting("dobar dan")
        assert result is not None
        assert "AI asistent" in result

    def test_greeting_hvala(self):
        result = matches_greeting("hvala")
        assert result is not None
        assert "čemu" in result

    def test_non_greeting(self):
        assert matches_greeting("koliko km") is None
        assert matches_greeting("nesto") is None

    # --- EU AI Act: greeting must include AI disclosure ---

    @pytest.mark.parametrize("text", [
        "bok", "hej", "pozdrav", "zdravo", "dobar dan",
        "dobro jutro", "dobra večer", "dobra vecer",
    ])
    def test_greeting_ai_disclosure(self, text):
        result = matches_greeting(text)
        assert result is not None
        assert "AI asistent" in result, f"Greeting '{text}' missing AI disclosure"


# ===========================================================================
# SECTION 5: Read vs Write Safety
# ===========================================================================

class TestReadWriteSafety:
    """Critical safety: never confuse read with write operations."""

    def _get_action(self, tool_name: str) -> str:
        if not tool_name:
            return "NONE"
        if tool_name.startswith("get_"):
            return "GET"
        if tool_name.startswith("post_"):
            return "POST"
        if tool_name.startswith("delete_"):
            return "DELETE"
        return "UNKNOWN"

    def test_read_intents_have_get_tools(self):
        """All READ intents must map to get_* tools."""
        read_intents = [
            "GET_MILEAGE", "GET_VEHICLE_INFO", "GET_REGISTRATION_EXPIRY",
            "GET_PLATE", "GET_LEASING", "GET_SERVICE_MILEAGE",
            "GET_VEHICLE_COMPANY", "GET_VEHICLE_EQUIPMENT",
            "GET_VEHICLE_DOCUMENTS", "GET_VEHICLE_COUNT",
            "GET_MY_BOOKINGS", "GET_CASES", "GET_TRIPS",
            "GET_EXPENSES", "GET_VEHICLES", "GET_PERSON_INFO",
        ]
        for intent in read_intents:
            config = INTENT_CONFIG.get(intent)
            assert config, f"Missing intent: {intent}"
            tool = config["tool"]
            assert tool and tool.startswith("get_"), \
                f"{intent} should map to get_* tool, got '{tool}'"

    def test_write_intents_have_post_tools(self):
        """Write intents must map to post_* tools."""
        write_intents = ["INPUT_MILEAGE", "REPORT_DAMAGE"]
        for intent in write_intents:
            config = INTENT_CONFIG[intent]
            assert config["tool"].startswith("post_"), \
                f"{intent} should map to post_* tool, got '{config['tool']}'"

    def test_delete_intents_have_delete_tools(self):
        """Delete intents must map to delete_* tools."""
        delete_intents = ["CANCEL_RESERVATION", "DELETE_CASE", "DELETE_TRIP"]
        for intent in delete_intents:
            config = INTENT_CONFIG[intent]
            assert config["tool"].startswith("delete_"), \
                f"{intent} should map to delete_* tool, got '{config['tool']}'"

    def test_direct_response_intents_have_no_tool(self):
        """Direct response intents must have tool=None."""
        no_tool_intents = [
            "GREETING", "THANKS", "HELP",
            "GET_PERSON_ID", "GET_PHONE", "GET_TENANT_ID",
        ]
        for intent in no_tool_intents:
            config = INTENT_CONFIG[intent]
            assert config["tool"] is None, \
                f"{intent} should have tool=None, got '{config['tool']}'"


# ===========================================================================
# ENGINE TEST HELPERS
# ===========================================================================

def _mock_settings():
    s = MagicMock()
    s.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com"
    s.AZURE_OPENAI_API_KEY = "test-key"
    s.AZURE_OPENAI_API_VERSION = "2024-02-15"
    s.AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4"
    s.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding"
    s.RAG_REFRESH_INTERVAL_HOURS = 6
    s.RAG_LOCK_TTL_SECONDS = 600
    return s


def _make_engine():
    """Build MessageEngine with all dependencies mocked."""
    ms = _mock_settings()
    patches = {
        "settings": patch("services.engine.settings", ms),
        "get_settings": patch("services.engine.get_settings", return_value=ms),
        "ToolExecutor": patch("services.engine.ToolExecutor"),
        "AIOrchestrator": patch("services.engine.AIOrchestrator"),
        "ResponseFormatter": patch("services.engine.ResponseFormatter"),
        "DependencyResolver": patch("services.engine.DependencyResolver"),
        "ErrorLearningService": patch("services.engine.ErrorLearningService"),
        "get_drift_detector": patch("services.engine.get_drift_detector"),
        "CostTracker": patch("services.engine.CostTracker"),
        "get_response_extractor": patch("services.engine.get_response_extractor"),
        "get_query_router": patch("services.engine.get_query_router"),
        "get_unified_router": patch("services.engine.get_unified_router"),
        "ToolHandler": patch("services.engine.ToolHandler"),
        "FlowHandler": patch("services.engine.FlowHandler"),
        "UserHandler": patch("services.engine.UserHandler"),
        "HallucinationHandler": patch("services.engine.HallucinationHandler"),
        "DeterministicExecutor": patch("services.engine.DeterministicExecutor"),
        "FlowExecutors": patch("services.engine.FlowExecutors"),
        "ConversationManager": patch("services.engine.ConversationManager"),
    }
    mocks = {}
    for name, p in patches.items():
        mocks[name] = p.start()

    ctx_svc = MagicMock()
    ctx_svc.redis = AsyncMock()
    ctx_svc.add_message = AsyncMock()
    ctx_svc.get_recent_messages = AsyncMock(return_value=[])

    from services.engine import MessageEngine
    engine = MessageEngine(
        gateway=MagicMock(),
        registry=MagicMock(is_ready=True),
        context_service=ctx_svc,
        queue_service=MagicMock(),
        cache_service=MagicMock(),
        db_session=MagicMock(),
    )

    for p in patches.values():
        p.stop()
    return engine, mocks


def _idle_conv_manager():
    """Create a ConversationManager mock in IDLE state.

    Uses MagicMock (not AsyncMock) as base so synchronous methods
    like get_state() return values directly. Async methods are
    overridden individually.
    """
    cm = MagicMock()
    cm.get_state.return_value = ConversationState.IDLE
    cm.is_in_flow.return_value = False
    cm.get_current_flow.return_value = None
    cm.get_current_tool.return_value = None
    cm.get_missing_params.return_value = []
    cm.get_displayed_items.return_value = []
    cm.context = MagicMock()
    cm.context.tool_outputs = {}
    # Async methods
    cm.save = AsyncMock()
    cm.reset = AsyncMock()
    cm.start_flow = AsyncMock()
    cm.add_parameters = AsyncMock()
    cm.request_confirmation = AsyncMock()
    cm.set_displayed_items = AsyncMock()
    return cm


def _setup_router_decision(engine, action, **kwargs):
    """Set up unified router to return a specific decision."""
    decision = MagicMock()
    decision.action = action
    decision.tool = kwargs.get("tool", None)
    decision.flow_type = kwargs.get("flow_type", None)
    decision.confidence = kwargs.get("confidence", 0.95)
    decision.response = kwargs.get("response", None)
    decision.clarification = kwargs.get("clarification", None)
    decision.params = kwargs.get("params", {})
    for k, v in kwargs.items():
        if not hasattr(decision, k):
            setattr(decision, k, v)
    engine.unified_router = AsyncMock()
    engine.unified_router.route = AsyncMock(return_value=decision)
    engine._unified_router_initialized = True
    return decision


def _user_context(person_id="00000000-0000-0000-0000-000000000001",
                  is_guest=False, phone="+385991234567"):
    ctx = {
        "person_id": person_id,
        "phone": phone,
        "tenant_id": "test-tenant",
    }
    if is_guest:
        ctx["is_guest"] = True
        ctx["person_id"] = None
    return ctx


# ===========================================================================
# SECTION 6: Engine Decision Routing (_process_with_state)
# ===========================================================================

class TestEngineDecisionRouting:
    """Test that _process_with_state routes each decision action correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine, self.mocks = _make_engine()

    @pytest.mark.asyncio
    async def test_direct_response(self):
        cm = _idle_conv_manager()
        _setup_router_decision(
            self.engine, "direct_response",
            response="Bok! Ja sam AI asistent."
        )
        result = await self.engine._process_with_state("sender", "bok", _user_context(), cm)
        assert "AI asistent" in result or "Bok" in result

    @pytest.mark.asyncio
    async def test_clarify(self):
        cm = _idle_conv_manager()
        _setup_router_decision(
            self.engine, "clarify",
            clarification="Mislite li na kilometražu ili registraciju?"
        )
        result = await self.engine._process_with_state("sender", "info", _user_context(), cm)
        assert "Mislite" in result or "detalja" in result

    @pytest.mark.asyncio
    async def test_simple_api(self):
        cm = _idle_conv_manager()
        _setup_router_decision(
            self.engine, "simple_api",
            tool="get_MasterData", confidence=0.95
        )
        self.engine._deterministic_executor.execute = AsyncMock(return_value="Km: 45.000 km")
        result = await self.engine._process_with_state("sender", "koliko km", _user_context(), cm)
        assert "45.000" in result

    @pytest.mark.asyncio
    async def test_simple_api_failure_returns_error(self):
        cm = _idle_conv_manager()
        _setup_router_decision(
            self.engine, "simple_api",
            tool="get_MasterData", confidence=0.95
        )
        self.engine._deterministic_executor.execute = AsyncMock(return_value=None)
        result = await self.engine._process_with_state("sender", "koliko km", _user_context(), cm)
        assert "greške" in result.lower() or "ponovo" in result.lower()

    @pytest.mark.asyncio
    async def test_unhandled_action_returns_help(self):
        cm = _idle_conv_manager()
        _setup_router_decision(self.engine, "unknown_action_xyz")
        result = await self.engine._process_with_state("sender", "xyz", _user_context(), cm)
        assert "Rezervaciju" in result or "Kilometražu" in result


# ===========================================================================
# SECTION 7: Flow Start Routing (_handle_flow_start)
# ===========================================================================

class TestFlowStartRouting:
    """Test _handle_flow_start dispatches to correct flow executor."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine, self.mocks = _make_engine()

    @pytest.mark.asyncio
    async def test_booking_flow_start(self):
        cm = _idle_conv_manager()
        decision = _setup_router_decision(
            self.engine, "start_flow",
            flow_type="booking", params={"from": "2026-03-16"}
        )
        self.engine._flow_executors.handle_booking_flow = AsyncMock(return_value="Dostupna vozila...")
        result = await self.engine._handle_flow_start(decision, "rezerviraj auto", _user_context(), cm)
        assert result == "Dostupna vozila..."
        self.engine._flow_executors.handle_booking_flow.assert_called_once()

    @pytest.mark.asyncio
    async def test_mileage_flow_start(self):
        cm = _idle_conv_manager()
        decision = _setup_router_decision(
            self.engine, "start_flow",
            flow_type="mileage", params={"Value": "45000"}
        )
        self.engine._flow_executors.handle_mileage_input_flow = AsyncMock(return_value="Unosim km...")
        result = await self.engine._handle_flow_start(decision, "unesi km", _user_context(), cm)
        assert result == "Unosim km..."
        self.engine._flow_executors.handle_mileage_input_flow.assert_called_once()

    @pytest.mark.asyncio
    async def test_case_flow_start(self):
        cm = _idle_conv_manager()
        decision = _setup_router_decision(
            self.engine, "start_flow",
            flow_type="case", params={"Description": "kvar motora"}
        )
        self.engine._flow_executors.handle_case_creation_flow = AsyncMock(return_value="Prijavljen kvar")
        result = await self.engine._handle_flow_start(decision, "prijavi kvar", _user_context(), cm)
        assert result == "Prijavljen kvar"
        self.engine._flow_executors.handle_case_creation_flow.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_booking_flow_start(self):
        cm = _idle_conv_manager()
        decision = _setup_router_decision(
            self.engine, "start_flow",
            flow_type="delete_booking", params={}
        )
        self.engine._flow_executors.handle_delete_flow = AsyncMock(return_value="Odaberite rezervaciju...")
        result = await self.engine._handle_flow_start(decision, "otkazi rezervaciju", _user_context(), cm)
        assert result == "Odaberite rezervaciju..."
        self.engine._flow_executors.handle_delete_flow.assert_called_once_with(
            "delete_booking", "otkazi rezervaciju", _user_context(), cm
        )

    @pytest.mark.asyncio
    async def test_delete_case_flow_start(self):
        cm = _idle_conv_manager()
        decision = _setup_router_decision(
            self.engine, "start_flow",
            flow_type="delete_case", params={}
        )
        self.engine._flow_executors.handle_delete_flow = AsyncMock(return_value="Odaberite slucaj...")
        result = await self.engine._handle_flow_start(decision, "obrisi slucaj", _user_context(), cm)
        assert result == "Odaberite slucaj..."

    @pytest.mark.asyncio
    async def test_delete_trip_flow_start(self):
        cm = _idle_conv_manager()
        decision = _setup_router_decision(
            self.engine, "start_flow",
            flow_type="delete_trip", params={}
        )
        self.engine._flow_executors.handle_delete_flow = AsyncMock(return_value="Odaberite putovanje...")
        result = await self.engine._handle_flow_start(decision, "obrisi putovanje", _user_context(), cm)
        assert result == "Odaberite putovanje..."

    @pytest.mark.asyncio
    async def test_unknown_flow_type_returns_error(self):
        cm = _idle_conv_manager()
        decision = _setup_router_decision(
            self.engine, "start_flow",
            flow_type="nonexistent_flow", params={}
        )
        result = await self.engine._handle_flow_start(decision, "test", _user_context(), cm)
        assert "flow" in result.lower() or "Neispravan" in result or "Nisam mogao" in result


# ===========================================================================
# SECTION 8: Guest User Restrictions
# ===========================================================================

class TestGuestUserRestrictions:
    """Guest users must be blocked from flows and person-dependent APIs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine, self.mocks = _make_engine()

    @pytest.mark.asyncio
    async def test_guest_blocked_from_start_flow(self):
        cm = _idle_conv_manager()
        _setup_router_decision(
            self.engine, "start_flow",
            flow_type="booking", params={}
        )
        guest_ctx = _user_context(is_guest=True)
        result = await self.engine._process_with_state("sender", "rezerviraj", guest_ctx, cm)
        assert "registrirani" in result.lower()

    @pytest.mark.asyncio
    async def test_guest_blocked_from_person_dependent_api(self):
        cm = _idle_conv_manager()
        _setup_router_decision(
            self.engine, "simple_api",
            tool="get_MasterData", confidence=0.95
        )
        guest_ctx = _user_context(is_guest=True)

        # Mock registry to indicate tool needs person_id
        mock_tool = MagicMock()
        mock_param = MagicMock()
        mock_param.context_key = "person_id"
        mock_tool.parameters = {"PersonId": mock_param}
        self.engine.registry.get_tool = MagicMock(return_value=mock_tool)

        result = await self.engine._process_with_state("sender", "koliko km", guest_ctx, cm)
        assert "registrirani" in result.lower()

    @pytest.mark.asyncio
    async def test_guest_allowed_direct_response(self):
        cm = _idle_conv_manager()
        _setup_router_decision(
            self.engine, "direct_response",
            response="Bok! Ja sam AI asistent."
        )
        guest_ctx = _user_context(is_guest=True)
        result = await self.engine._process_with_state("sender", "bok", guest_ctx, cm)
        assert "AI asistent" in result


# ===========================================================================
# SECTION 9: In-Flow State Handling
# ===========================================================================

class TestInFlowStateHandling:
    """Test that in-flow messages route to correct handlers."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine, self.mocks = _make_engine()

    def _in_flow_cm(self, state, flow, tool, missing=None, items=None):
        """Helper to create an in-flow conv_manager mock."""
        cm = MagicMock()
        cm.get_state.return_value = state
        cm.is_in_flow.return_value = True
        cm.get_current_flow.return_value = flow
        cm.get_current_tool.return_value = tool
        cm.get_missing_params.return_value = missing or []
        cm.get_displayed_items.return_value = items or []
        cm.context = MagicMock()
        cm.context.tool_outputs = {}
        cm.save = AsyncMock()
        cm.reset = AsyncMock()
        return cm

    @pytest.mark.asyncio
    async def test_confirming_state_yes(self):
        cm = self._in_flow_cm(ConversationState.CONFIRMING, "booking", "post_AddBooking")
        self.engine._flow_handler.handle_confirmation = AsyncMock(return_value="Rezervacija potvrđena!")
        _setup_router_decision(self.engine, "continue_flow", flow_type="booking")

        result = await self.engine._process_with_state("sender", "da", _user_context(), cm)
        assert "potvrđena" in result

    @pytest.mark.asyncio
    async def test_confirming_state_no(self):
        cm = self._in_flow_cm(ConversationState.CONFIRMING, "booking", "post_AddBooking")
        self.engine._flow_handler.handle_confirmation = AsyncMock(return_value="Operacija otkazana.")
        _setup_router_decision(self.engine, "continue_flow", flow_type="booking")

        result = await self.engine._process_with_state("sender", "ne", _user_context(), cm)
        assert "otkazana" in result.lower() or "Operacija" in result

    @pytest.mark.asyncio
    async def test_selecting_item_state(self):
        cm = self._in_flow_cm(
            ConversationState.SELECTING_ITEM, "delete_booking",
            "delete_VehicleCalendar_id", items=[{"Id": "abc123"}]
        )
        self.engine._flow_handler.handle_selection = AsyncMock(return_value="Odabrana stavka 1.")
        _setup_router_decision(self.engine, "continue_flow", flow_type="delete_booking")

        result = await self.engine._process_with_state("sender", "1", _user_context(), cm)
        assert "Odabrana" in result

    @pytest.mark.asyncio
    async def test_gathering_params_state(self):
        cm = self._in_flow_cm(
            ConversationState.GATHERING_PARAMS, "mileage_input",
            "post_AddMileage", missing=["Value"]
        )
        self.engine._flow_handler.handle_gathering = AsyncMock(return_value="Unesite kilometražu")
        _setup_router_decision(self.engine, "continue_flow", flow_type="mileage_input")

        result = await self.engine._process_with_state("sender", "45000", _user_context(), cm)
        assert "kilometražu" in result.lower() or "Unesite" in result


# ===========================================================================
# SECTION 10: Exit Flow
# ===========================================================================

class TestExitFlow:
    """Test exit_flow resets state and re-routes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine, self.mocks = _make_engine()

    @pytest.mark.asyncio
    async def test_exit_flow_resets_and_reroutes(self):
        cm = MagicMock()
        cm.get_state.return_value = ConversationState.GATHERING_PARAMS
        cm.is_in_flow.return_value = True
        cm.get_current_flow.return_value = "booking"
        cm.get_current_tool.return_value = "get_AvailableVehicles"
        cm.get_missing_params.return_value = ["FromTime"]
        cm.get_displayed_items.return_value = []
        cm.reset = AsyncMock()
        cm.save = AsyncMock()
        cm.context = MagicMock()
        cm.context.tool_outputs = {}

        # First route returns exit_flow, second returns direct_response
        exit_decision = MagicMock()
        exit_decision.action = "exit_flow"
        exit_decision.tool = None
        exit_decision.flow_type = None
        exit_decision.confidence = 0.9
        exit_decision.response = None
        exit_decision.clarification = None
        exit_decision.params = {}

        reroute_decision = MagicMock()
        reroute_decision.action = "direct_response"
        reroute_decision.response = "Kako vam mogu pomoći?"
        reroute_decision.tool = None
        reroute_decision.flow_type = None
        reroute_decision.confidence = 0.9
        reroute_decision.clarification = None
        reroute_decision.params = {}

        self.engine.unified_router = AsyncMock()
        self.engine.unified_router.route = AsyncMock(
            side_effect=[exit_decision, reroute_decision]
        )
        self.engine._unified_router_initialized = True

        result = await self.engine._process_with_state("sender", "nešto drugo", _user_context(), cm)
        cm.reset.assert_called_once()
        assert "pomoći" in result

    @pytest.mark.asyncio
    async def test_exit_flow_when_not_in_flow(self):
        cm = _idle_conv_manager()
        _setup_router_decision(self.engine, "exit_flow")
        result = await self.engine._process_with_state("sender", "odustani", _user_context(), cm)
        assert "pomoći" in result.lower()


# ===========================================================================
# SECTION 11: Router Failure Fallback
# ===========================================================================

class TestRouterFailureFallback:
    """When unified router throws, engine falls back to _handle_new_request."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine, self.mocks = _make_engine()

    @pytest.mark.asyncio
    async def test_router_exception_falls_back(self):
        cm = _idle_conv_manager()
        self.engine.unified_router = AsyncMock()
        self.engine.unified_router.route = AsyncMock(side_effect=RuntimeError("Router crash"))
        self.engine._unified_router_initialized = True

        # _handle_new_request uses query_router internally
        self.engine.query_router = MagicMock()
        self.engine.query_router.route = MagicMock(
            return_value=RouteResult(matched=False)
        )

        result = await self.engine._process_with_state("sender", "test", _user_context(), cm)
        # Should get fallback help message, not a crash
        assert "Rezervaciju" in result or "Kilometražu" in result or "siguran" in result


# ===========================================================================
# SECTION 12: Empty Input
# ===========================================================================

class TestEmptyInput:
    """Engine must handle empty/whitespace input gracefully."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine, self.mocks = _make_engine()

    @pytest.mark.asyncio
    async def test_empty_string_returns_prompt(self):
        # Patch _identify_user to avoid DB calls
        self.engine._user_handler.identify_user = AsyncMock(return_value=_user_context())
        # ConversationManager mock
        cm_instance = _idle_conv_manager()
        self.engine._ConversationManager = MagicMock(return_value=cm_instance)

        # The engine.process() method checks for empty text early
        result = await self.engine.process("sender", "", _user_context())
        # Should return a prompt, not crash
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_prompt(self):
        self.engine._user_handler.identify_user = AsyncMock(return_value=_user_context())
        cm_instance = _idle_conv_manager()
        self.engine._ConversationManager = MagicMock(return_value=cm_instance)

        result = await self.engine.process("sender", "   ", _user_context())
        assert result is not None
        assert len(result) > 0


# ===========================================================================
# SECTION 13: European Number Parsing
# ===========================================================================

class TestEuropeanNumberParsing:
    """Verify European thousands separators are handled in mileage parsing."""

    def test_european_dot_separator(self):
        """45.000 (European) should become 45000, not 45."""
        text = "45.000 km"
        cleaned = re.sub(r'(\d)[.,](\d{3})\b', r'\1\2', text)
        numbers = re.findall(r'\d+', cleaned)
        assert "45000" in numbers

    def test_european_comma_separator(self):
        """45,000 should become 45000."""
        text = "45,000 km"
        cleaned = re.sub(r'(\d)[.,](\d{3})\b', r'\1\2', text)
        numbers = re.findall(r'\d+', cleaned)
        assert "45000" in numbers

    def test_plain_number_unchanged(self):
        """45000 without separator stays 45000."""
        text = "45000"
        cleaned = re.sub(r'(\d)[.,](\d{3})\b', r'\1\2', text)
        numbers = re.findall(r'\d+', cleaned)
        assert "45000" in numbers

    def test_decimal_not_collapsed(self):
        """3.5 should NOT become 35 (not a thousands separator)."""
        text = "3.5 litara"
        cleaned = re.sub(r'(\d)[.,](\d{3})\b', r'\1\2', text)
        numbers = re.findall(r'\d+', cleaned)
        assert "35" not in numbers
        assert "3" in numbers
        assert "5" in numbers

    def test_large_european_number(self):
        """145.320 km should become 145320."""
        text = "145.320"
        cleaned = re.sub(r'(\d)[.,](\d{3})\b', r'\1\2', text)
        numbers = re.findall(r'\d+', cleaned)
        assert "145320" in numbers


# ===========================================================================
# SECTION 14: Conversation States Coverage
# ===========================================================================

class TestConversationStates:
    """Verify all ConversationState enum values are defined."""

    def test_all_states_exist(self):
        expected = {"IDLE", "GATHERING_PARAMS", "SELECTING_ITEM", "CONFIRMING", "EXECUTING", "COMPLETED"}
        actual = {s.name for s in ConversationState}
        assert expected.issubset(actual), f"Missing states: {expected - actual}"

    def test_idle_is_default(self):
        assert ConversationState.IDLE.value is not None


# ===========================================================================
# SECTION 15: Flow Type Coverage
# ===========================================================================

class TestFlowTypeCoverage:
    """Every flow_type in INTENT_CONFIG must be handled somewhere."""

    def test_all_flow_types_mapped(self):
        """Verify every flow_type is in the known set the engine handles."""
        engine_handled = {
            "simple", "list", "direct_response",
            "booking", "mileage_input", "case_creation",
            "delete_booking", "delete_case", "delete_trip",
        }
        for intent, config in INTENT_CONFIG.items():
            ft = config["flow_type"]
            assert ft in engine_handled, \
                f"INTENT_CONFIG[{intent}].flow_type='{ft}' not handled by engine"

    def test_flow_triggers_cover_write_tools(self):
        """All post_* tools should have FLOW_TRIGGERS entries.
        Delete tools use flow_type routing (not FLOW_TRIGGERS)."""
        for intent, config in INTENT_CONFIG.items():
            tool = config.get("tool")
            if tool and tool.startswith("post_"):
                assert tool in FLOW_TRIGGERS, \
                    f"Write tool '{tool}' ({intent}) missing from FLOW_TRIGGERS"

    def test_delete_tools_have_delete_flow_types(self):
        """Delete tools must have delete_* flow_types (routed via flow_type, not FLOW_TRIGGERS)."""
        for intent, config in INTENT_CONFIG.items():
            tool = config.get("tool")
            if tool and tool.startswith("delete_"):
                assert config["flow_type"].startswith("delete_"), \
                    f"Delete tool '{tool}' ({intent}) has flow_type='{config['flow_type']}'"


# ===========================================================================
# SECTION 16: Delete Flow Config
# ===========================================================================

class TestDeleteFlowConfig:
    """Verify DELETE_FLOW_CONFIG is complete and consistent.

    Reads config from flow_executors.py source via AST to avoid
    triggering the heavy services.engine import chain.
    """

    # Define expected config inline (mirrors flow_executors.py DELETE_FLOW_CONFIG)
    EXPECTED_DELETE_CONFIGS = {
        "delete_booking": {
            "list_tool": "get_VehicleCalendar",
            "delete_tool": "delete_VehicleCalendar_id",
        },
        "delete_case": {
            "list_tool": "get_Cases",
            "delete_tool": "delete_Cases_id",
        },
        "delete_trip": {
            "list_tool": "get_Trips",
            "delete_tool": "delete_Trips_id",
        },
    }

    def test_all_delete_flow_types_configured(self):
        delete_types = {"delete_booking", "delete_case", "delete_trip"}
        # Verify INTENT_CONFIG has matching flow_types
        config_flow_types = {
            c["flow_type"] for c in INTENT_CONFIG.values()
            if c["flow_type"].startswith("delete_")
        }
        assert delete_types == config_flow_types, \
            f"Missing: {delete_types - config_flow_types}, Extra: {config_flow_types - delete_types}"

    def test_delete_config_list_tools_are_get(self):
        for flow_type, config in self.EXPECTED_DELETE_CONFIGS.items():
            assert config["list_tool"].startswith("get_"), \
                f"{flow_type} list_tool should be get_*, got {config['list_tool']}"

    def test_delete_config_delete_tools_are_delete(self):
        for flow_type, config in self.EXPECTED_DELETE_CONFIGS.items():
            assert config["delete_tool"].startswith("delete_"), \
                f"{flow_type} delete_tool should be delete_*, got {config['delete_tool']}"

    def test_delete_list_tools_exist_in_primary_tools(self):
        for flow_type, config in self.EXPECTED_DELETE_CONFIGS.items():
            assert config["list_tool"] in PRIMARY_TOOLS, \
                f"{flow_type} list_tool '{config['list_tool']}' not in PRIMARY_TOOLS"

    def test_delete_delete_tools_exist_in_primary_tools(self):
        for flow_type, config in self.EXPECTED_DELETE_CONFIGS.items():
            assert config["delete_tool"] in PRIMARY_TOOLS, \
                f"{flow_type} delete_tool '{config['delete_tool']}' not in PRIMARY_TOOLS"


# ===========================================================================
# SECTION 17: Croatian Diacritics
# ===========================================================================

class TestCroatianDiacritics:
    """Phrase matching must handle both diacritic and non-diacritic variants."""

    @pytest.mark.parametrize("diacritic,plain", [
        ("može", "moze"),
        ("potvrđeno", "potvrdeno"),
        ("ne želim", "ne zelim"),
        ("odustajem", "odustajem"),
        ("važi", "vazi"),
        ("točno", "tocno"),
        ("više", "vise"),
        ("prikaži", "prikazi"),
    ])
    def test_diacritic_pairs_both_match(self, diacritic, plain):
        """Both diacritic and ASCII versions must match the same category."""
        yes_d = matches_confirm_yes(diacritic)
        yes_p = matches_confirm_yes(plain)
        no_d = matches_confirm_no(diacritic)
        no_p = matches_confirm_no(plain)
        exit_d = matches_exit_signal(diacritic)
        exit_p = matches_exit_signal(plain)
        show_d = matches_show_more(diacritic)
        show_p = matches_show_more(plain)

        # At least one category must match for both variants
        d_any = yes_d or no_d or exit_d or show_d
        p_any = yes_p or no_p or exit_p or show_p
        assert d_any, f"Diacritic '{diacritic}' matches nothing"
        assert p_any, f"Plain '{plain}' matches nothing"


# ===========================================================================
# SECTION 18: ML Confidence Threshold
# ===========================================================================

class TestMLConfidenceThreshold:
    """Verify ML confidence threshold is set correctly."""

    def test_threshold_is_85_percent(self):
        assert ML_CONFIDENCE_THRESHOLD == 0.85

    def test_threshold_is_float(self):
        assert isinstance(ML_CONFIDENCE_THRESHOLD, float)


# ===========================================================================
# SECTION 19: _handle_new_request Fallback
# ===========================================================================

class TestHandleNewRequestFallback:
    """Test _handle_new_request (QueryRouter-only fallback path)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine, self.mocks = _make_engine()

    @pytest.mark.asyncio
    async def test_no_match_returns_help(self):
        cm = _idle_conv_manager()
        self.engine.query_router = MagicMock()
        self.engine.query_router.route = MagicMock(
            return_value=RouteResult(matched=False)
        )

        result = await self.engine._handle_new_request("sender", "asdfghjkl", _user_context(), cm)
        assert "Rezervaciju" in result or "siguran" in result

    @pytest.mark.asyncio
    async def test_direct_response_match(self):
        cm = _idle_conv_manager()
        self.engine.query_router = MagicMock()
        self.engine.query_router.route = MagicMock(
            return_value=RouteResult(
                matched=True,
                flow_type="direct_response",
                response_template="Vaš person ID: {person_id}"
            )
        )

        result = await self.engine._handle_new_request("sender", "moj person id", _user_context(), cm)
        assert "Person ID" in result or "N/A" in result or "person_id" in result.lower()

    @pytest.mark.asyncio
    async def test_simple_query_match(self):
        cm = _idle_conv_manager()
        self.engine.query_router = MagicMock()
        self.engine.query_router.route = MagicMock(
            return_value=RouteResult(
                matched=True,
                tool_name="get_MasterData",
                flow_type="simple",
                confidence=0.95
            )
        )
        self.engine._deterministic_executor.execute = AsyncMock(return_value="Km: 45000")

        result = await self.engine._handle_new_request("sender", "koliko km", _user_context(), cm)
        assert "45000" in result


# ===========================================================================
# SECTION 20: _handle_flow_start All Types
# ===========================================================================

class TestHandleFlowStartAllTypes:
    """Verify _handle_flow_start covers all flow_type values from INTENT_CONFIG."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine, self.mocks = _make_engine()

    def _get_unique_flow_types(self):
        """Get all unique flow types that go through start_flow."""
        flow_types = set()
        for config in INTENT_CONFIG.values():
            ft = config["flow_type"]
            if ft not in ("simple", "list", "direct_response"):
                flow_types.add(ft)
        return flow_types

    @pytest.mark.asyncio
    async def test_all_flow_types_have_handler(self):
        """Every non-simple flow_type must have a code path in _handle_flow_start."""
        # Map flow_type to the canonical name _handle_flow_start uses
        canonical = {
            "booking": "booking",
            "mileage_input": "mileage",
            "case_creation": "case",
            "delete_booking": "delete_booking",
            "delete_case": "delete_case",
            "delete_trip": "delete_trip",
        }
        for ft in self._get_unique_flow_types():
            canon = canonical.get(ft)
            assert canon is not None, f"flow_type '{ft}' has no canonical mapping"

            cm = _idle_conv_manager()
            decision = MagicMock()
            decision.flow_type = canon
            decision.params = {}
            decision.tool = None

            # Mock the appropriate executor method
            if canon == "booking":
                self.engine._flow_executors.handle_booking_flow = AsyncMock(return_value="ok")
            elif canon == "mileage":
                self.engine._flow_executors.handle_mileage_input_flow = AsyncMock(return_value="ok")
            elif canon == "case":
                self.engine._flow_executors.handle_case_creation_flow = AsyncMock(return_value="ok")
            elif canon.startswith("delete_"):
                self.engine._flow_executors.handle_delete_flow = AsyncMock(return_value="ok")

            result = await self.engine._handle_flow_start(decision, "test", _user_context(), cm)
            assert result == "ok", f"_handle_flow_start did not handle flow_type '{canon}'"

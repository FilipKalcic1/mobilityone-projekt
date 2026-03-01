"""Tests for services/engine/__init__.py – MessageEngine."""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _user_context(person_id="00000000-0000-0000-0000-000000000001",
                  vehicle=None, phone="+385991234567"):
    ctx = {
        "person_id": person_id,
        "phone": phone,
        "tenant_id": "test-tenant",
    }
    if vehicle:
        ctx["vehicle"] = vehicle
    return ctx


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
        "Planner": patch("services.engine.Planner"),
        "get_chain_planner": patch("services.engine.get_chain_planner"),
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

    gateway = MagicMock()
    registry = MagicMock()
    registry.is_ready = True
    context_service = MagicMock()
    context_service.redis = AsyncMock()
    context_service.add_message = AsyncMock()
    context_service.get_recent_messages = AsyncMock(return_value=[])
    queue_service = MagicMock()
    cache_service = MagicMock()
    db_session = MagicMock()

    from services.engine import MessageEngine
    engine = MessageEngine(
        gateway=gateway,
        registry=registry,
        context_service=context_service,
        queue_service=queue_service,
        cache_service=cache_service,
        db_session=db_session,
    )

    # Stop all patches
    for p in patches.values():
        p.stop()

    return engine, mocks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine_and_mocks():
    return _make_engine()


@pytest.fixture
def engine(engine_and_mocks):
    return engine_and_mocks[0]


# ===========================================================================
# __init__
# ===========================================================================

class TestInit:
    def test_engine_created(self, engine):
        assert engine is not None
        assert engine.MAX_ITERATIONS == 6

    def test_engine_has_handlers(self, engine):
        assert engine._tool_handler is not None
        assert engine._flow_handler is not None
        assert engine._user_handler is not None
        assert engine._hallucination_handler is not None
        assert engine._deterministic_executor is not None
        assert engine._flow_executors is not None

    def test_engine_no_redis(self):
        ms = _mock_settings()
        with patch("services.engine.settings", ms), \
             patch("services.engine.get_settings", return_value=ms), \
             patch("services.engine.ToolExecutor"), \
             patch("services.engine.AIOrchestrator"), \
             patch("services.engine.ResponseFormatter"), \
             patch("services.engine.DependencyResolver"), \
             patch("services.engine.ErrorLearningService"), \
             patch("services.engine.get_drift_detector"), \
             patch("services.engine.CostTracker"), \
             patch("services.engine.Planner"), \
             patch("services.engine.get_chain_planner"), \
             patch("services.engine.get_response_extractor"), \
             patch("services.engine.get_query_router"), \
             patch("services.engine.get_unified_router"), \
             patch("services.engine.ToolHandler"), \
             patch("services.engine.FlowHandler"), \
             patch("services.engine.UserHandler"), \
             patch("services.engine.HallucinationHandler"), \
             patch("services.engine.DeterministicExecutor"), \
             patch("services.engine.FlowExecutors"), \
             patch("services.engine.ConversationManager"):
            from services.engine import MessageEngine
            e = MessageEngine(
                gateway=MagicMock(),
                registry=MagicMock(),
                context_service=None,
                queue_service=None,
                cache_service=None,
                db_session=None,
            )
            assert e.redis is None
            assert e.cost_tracker is None


# ===========================================================================
# process
# ===========================================================================

class TestProcess:
    @pytest.mark.asyncio
    async def test_guest_user_gets_response(self, engine):
        """identify_user always returns a context (guest if not in MobilityOne)."""
        guest_ctx = {
            "person_id": None,
            "phone": "+385000000000",
            "tenant_id": "default",
            "display_name": "Korisnik",
            "vehicle": {},
            "is_new": False,
            "is_guest": True,
        }
        engine._user_handler.identify_user = AsyncMock(return_value=guest_ctx)

        conv = AsyncMock()
        conv.is_timed_out.return_value = False
        conv.get_state.return_value = MagicMock(value="idle")
        conv.is_in_flow.return_value = False
        conv.get_current_flow.return_value = None
        conv.get_current_tool.return_value = None
        conv.get_missing_params.return_value = []
        conv.get_displayed_items.return_value = []
        conv.reset = AsyncMock()
        conv.save = AsyncMock()

        with patch("services.engine.ConversationManager") as MockCM:
            MockCM.load_for_user = AsyncMock(return_value=conv)
            engine._hallucination_handler.check_hallucination_feedback = AsyncMock(return_value=None)

            with patch("services.engine.UserContextManager") as MockUCM:
                ucm_inst = MagicMock()
                ucm_inst.is_new = False
                ucm_inst.is_guest = True
                MockUCM.return_value = ucm_inst

                engine._process_with_state = AsyncMock(return_value="Guest response")
                result = await engine.process("sender123", "hello")
                assert result == "Guest response"

    @pytest.mark.asyncio
    async def test_user_found_normal_flow(self, engine):
        ctx = _user_context()
        engine._user_handler.identify_user = AsyncMock(return_value=ctx)

        conv = AsyncMock()
        conv.is_timed_out.return_value = False
        conv.get_state.return_value = MagicMock(value="idle")
        conv.is_in_flow.return_value = False
        conv.get_current_flow.return_value = None
        conv.get_current_tool.return_value = None
        conv.get_missing_params.return_value = []
        conv.get_displayed_items.return_value = []
        conv.reset = AsyncMock()
        conv.save = AsyncMock()

        with patch("services.engine.ConversationManager") as MockCM:
            MockCM.load_for_user = AsyncMock(return_value=conv)

            engine._hallucination_handler.check_hallucination_feedback = AsyncMock(return_value=None)

            with patch("services.engine.UserContextManager") as MockUCM:
                ucm_inst = MagicMock()
                ucm_inst.is_new = False
                MockUCM.return_value = ucm_inst

                engine._process_with_state = AsyncMock(return_value="Test response")

                result = await engine.process("sender123", "koliko km?")
                assert result == "Test response"

    @pytest.mark.asyncio
    async def test_hallucination_detected(self, engine):
        ctx = _user_context()
        engine._user_handler.identify_user = AsyncMock(return_value=ctx)

        conv = AsyncMock()
        conv.is_timed_out.return_value = False

        with patch("services.engine.ConversationManager") as MockCM:
            MockCM.load_for_user = AsyncMock(return_value=conv)
            engine._hallucination_handler.check_hallucination_feedback = AsyncMock(
                return_value="Hvala na povratnoj informaciji!"
            )

            result = await engine.process("sender123", "to nije tocno")
            assert "povratnoj" in result

    @pytest.mark.asyncio
    async def test_timed_out_resets(self, engine):
        ctx = _user_context()
        engine._user_handler.identify_user = AsyncMock(return_value=ctx)

        conv = AsyncMock()
        conv.is_timed_out.return_value = True
        conv.reset = AsyncMock()
        conv.get_state.return_value = MagicMock(value="idle")
        conv.is_in_flow.return_value = False
        conv.get_current_flow.return_value = None
        conv.get_current_tool.return_value = None
        conv.get_missing_params.return_value = []
        conv.get_displayed_items.return_value = []

        with patch("services.engine.ConversationManager") as MockCM:
            MockCM.load_for_user = AsyncMock(return_value=conv)
            engine._hallucination_handler.check_hallucination_feedback = AsyncMock(return_value=None)

            with patch("services.engine.UserContextManager") as MockUCM:
                ucm_inst = MagicMock()
                ucm_inst.is_new = False
                MockUCM.return_value = ucm_inst
                engine._process_with_state = AsyncMock(return_value="ok")

                await engine.process("sender123", "test")
                conv.reset.assert_called()

    @pytest.mark.asyncio
    async def test_new_user_greeting(self, engine):
        ctx = _user_context()
        engine._user_handler.identify_user = AsyncMock(return_value=ctx)
        engine._user_handler.build_greeting = MagicMock(return_value="Dobrodosli!")

        conv = AsyncMock()
        conv.is_timed_out.return_value = False
        conv.get_state.return_value = MagicMock(value="idle")
        conv.is_in_flow.return_value = False

        with patch("services.engine.ConversationManager") as MockCM:
            MockCM.load_for_user = AsyncMock(return_value=conv)
            engine._hallucination_handler.check_hallucination_feedback = AsyncMock(return_value=None)

            with patch("services.engine.UserContextManager") as MockUCM:
                ucm_inst = MagicMock()
                ucm_inst.is_new = True
                MockUCM.return_value = ucm_inst
                engine._process_with_state = AsyncMock(return_value="Data response")

                result = await engine.process("sender123", "bok")
                assert "Dobrodosli!" in result
                assert "Data response" in result

    @pytest.mark.asyncio
    async def test_missing_context_error(self, engine):
        from services.context import MissingContextError
        ctx = _user_context()
        engine._user_handler.identify_user = AsyncMock(return_value=ctx)

        conv = AsyncMock()
        conv.is_timed_out.return_value = False

        with patch("services.engine.ConversationManager") as MockCM:
            MockCM.load_for_user = AsyncMock(return_value=conv)
            engine._hallucination_handler.check_hallucination_feedback = AsyncMock(return_value=None)

            with patch("services.engine.UserContextManager") as MockUCM:
                ucm_inst = MagicMock()
                ucm_inst.is_new = False
                MockUCM.return_value = ucm_inst

                err = MissingContextError("VehicleId", "Molim odaberite vozilo.")
                engine._process_with_state = AsyncMock(side_effect=err)

                result = await engine.process("sender123", "test")
                assert "odaberite" in result.lower()

    @pytest.mark.asyncio
    async def test_vehicle_selection_required(self, engine):
        from services.context import VehicleSelectionRequired
        ctx = _user_context()
        engine._user_handler.identify_user = AsyncMock(return_value=ctx)

        conv = AsyncMock()
        conv.is_timed_out.return_value = False

        with patch("services.engine.ConversationManager") as MockCM:
            MockCM.load_for_user = AsyncMock(return_value=conv)
            engine._hallucination_handler.check_hallucination_feedback = AsyncMock(return_value=None)

            with patch("services.engine.UserContextManager") as MockUCM:
                ucm_inst = MagicMock()
                ucm_inst.is_new = False
                MockUCM.return_value = ucm_inst

                vehicles = [
                    {"LicencePlate": "ZG-1234-AB", "FullVehicleName": "Golf"},
                    {"LicencePlate": "ZG-5678-CD", "FullVehicleName": "Passat"},
                ]
                err = VehicleSelectionRequired(vehicles, "Odaberite vozilo:")
                engine._process_with_state = AsyncMock(side_effect=err)

                result = await engine.process("sender123", "test")
                assert "ZG-1234-AB" in result
                assert "Golf" in result

    @pytest.mark.asyncio
    async def test_general_exception(self, engine):
        ctx = _user_context()
        engine._user_handler.identify_user = AsyncMock(return_value=ctx)

        conv = AsyncMock()
        conv.is_timed_out.return_value = False

        with patch("services.engine.ConversationManager") as MockCM:
            MockCM.load_for_user = AsyncMock(return_value=conv)
            engine._hallucination_handler.check_hallucination_feedback = AsyncMock(return_value=None)

            with patch("services.engine.UserContextManager") as MockUCM:
                ucm_inst = MagicMock()
                ucm_inst.is_new = False
                MockUCM.return_value = ucm_inst
                engine._process_with_state = AsyncMock(side_effect=RuntimeError("boom"))

                result = await engine.process("sender123", "test")
                assert "greske" in result.lower()


# ===========================================================================
# _process_with_state
# ===========================================================================

class TestProcessWithState:
    def _make_conv(self, state_value="idle", in_flow=False, flow=None, tool=None,
                   missing=None, items=None):
        conv = MagicMock()
        state = MagicMock()
        state.value = state_value
        conv.get_state.return_value = state
        conv.is_in_flow.return_value = in_flow
        conv.get_current_flow.return_value = flow
        conv.get_current_tool.return_value = tool
        conv.get_missing_params.return_value = missing or []
        conv.get_displayed_items.return_value = items or []
        conv.reset = AsyncMock()
        conv.save = AsyncMock()
        conv.to_dict.return_value = {}
        # Async methods
        conv.start_flow = AsyncMock()
        conv.request_confirmation = AsyncMock()
        conv.request_selection = AsyncMock()
        conv.add_parameters = AsyncMock()
        conv.select_item = AsyncMock()
        conv.confirm = AsyncMock()
        conv.cancel = AsyncMock()
        conv.complete = AsyncMock()
        conv.context = MagicMock()
        conv.context.tool_outputs = {}
        return conv

    @pytest.mark.asyncio
    async def test_direct_response(self, engine):
        conv = self._make_conv()
        decision = MagicMock()
        decision.action = "direct_response"
        decision.response = "Pozdrav!"
        decision.confidence = 0.99

        engine.unified_router = AsyncMock()
        engine.unified_router.route = AsyncMock(return_value=decision)
        engine._unified_router_initialized = True

        result = await engine._process_with_state("sender", "bok", _user_context(), conv)
        assert result == "Pozdrav!"

    @pytest.mark.asyncio
    async def test_clarify_action(self, engine):
        conv = self._make_conv()
        decision = MagicMock()
        decision.action = "clarify"
        decision.clarification = "Sto tocno trebate?"
        decision.confidence = 0.5

        engine.unified_router = AsyncMock()
        engine.unified_router.route = AsyncMock(return_value=decision)
        engine._unified_router_initialized = True

        result = await engine._process_with_state("sender", "nesto", _user_context(), conv)
        assert "tocno" in result

    @pytest.mark.asyncio
    async def test_confirming_state_da(self, engine):
        from services.conversation_manager import ConversationState
        conv = self._make_conv(state_value="confirming", in_flow=True)
        state_mock = MagicMock()
        state_mock.value = "confirming"
        state_mock.__eq__ = lambda self, other: other == ConversationState.CONFIRMING
        conv.get_state.return_value = state_mock

        engine._flow_handler.handle_confirmation = AsyncMock(return_value="Potvrdjeno!")

        engine.unified_router = AsyncMock()
        engine._unified_router_initialized = True

        result = await engine._process_with_state("sender", "da", _user_context(), conv)
        assert result == "Potvrdjeno!"

    @pytest.mark.asyncio
    async def test_selecting_state_number(self, engine):
        from services.conversation_manager import ConversationState
        conv = self._make_conv(state_value="selecting_item", in_flow=True)
        state_mock = MagicMock()
        state_mock.value = "selecting_item"
        state_mock.__eq__ = lambda self, other: other == ConversationState.SELECTING_ITEM
        conv.get_state.return_value = state_mock

        engine._flow_handler.handle_selection = AsyncMock(return_value="Odabrano!")

        engine.unified_router = AsyncMock()
        engine._unified_router_initialized = True

        result = await engine._process_with_state("sender", "1", _user_context(), conv)
        assert result == "Odabrano!"

    @pytest.mark.asyncio
    async def test_exit_flow_in_flow(self, engine):
        conv = self._make_conv(in_flow=True)
        decision1 = MagicMock()
        decision1.action = "exit_flow"
        decision1.confidence = 0.9

        decision2 = MagicMock()
        decision2.action = "direct_response"
        decision2.response = "Izasli ste iz flowa."
        decision2.confidence = 0.9

        engine.unified_router = AsyncMock()
        engine.unified_router.route = AsyncMock(side_effect=[decision1, decision2])
        engine.unified_router.set_registry = MagicMock()
        engine._unified_router_initialized = True

        result = await engine._process_with_state("sender", "odustani", _user_context(), conv)
        conv.reset.assert_called()
        assert "Izasli" in result

    @pytest.mark.asyncio
    async def test_exit_flow_not_in_flow(self, engine):
        conv = self._make_conv(in_flow=False)
        decision = MagicMock()
        decision.action = "exit_flow"
        decision.confidence = 0.9

        engine.unified_router = AsyncMock()
        engine.unified_router.route = AsyncMock(return_value=decision)
        engine._unified_router_initialized = True

        # is_in_flow returns False on the "if conv_manager.is_in_flow():" check
        result = await engine._process_with_state("sender", "exit", _user_context(), conv)
        assert "pomoci" in result.lower()

    @pytest.mark.asyncio
    async def test_start_flow(self, engine):
        conv = self._make_conv()
        decision = MagicMock()
        decision.action = "start_flow"
        decision.flow_type = "booking"
        decision.params = {}
        decision.confidence = 0.9

        engine.unified_router = AsyncMock()
        engine.unified_router.route = AsyncMock(return_value=decision)
        engine._unified_router_initialized = True

        engine._handle_flow_start = AsyncMock(return_value="Flow started!")

        result = await engine._process_with_state("sender", "rezerviraj", _user_context(), conv)
        assert result == "Flow started!"

    @pytest.mark.asyncio
    async def test_simple_api_action(self, engine):
        conv = self._make_conv()
        decision = MagicMock()
        decision.action = "simple_api"
        decision.tool = "get_MasterData"
        decision.confidence = 0.9

        engine.unified_router = AsyncMock()
        engine.unified_router.route = AsyncMock(return_value=decision)
        engine._unified_router_initialized = True

        engine._deterministic_executor.execute = AsyncMock(return_value="Km: 50000")

        result = await engine._process_with_state("sender", "km?", _user_context(), conv)
        assert result == "Km: 50000"

    @pytest.mark.asyncio
    async def test_simple_api_falls_through_to_new_request(self, engine):
        conv = self._make_conv()
        decision = MagicMock()
        decision.action = "simple_api"
        decision.tool = "get_MasterData"
        decision.confidence = 0.9

        engine.unified_router = AsyncMock()
        engine.unified_router.route = AsyncMock(return_value=decision)
        engine._unified_router_initialized = True

        engine._deterministic_executor.execute = AsyncMock(return_value=None)
        engine._handle_new_request = AsyncMock(return_value="Fallback response")

        result = await engine._process_with_state("sender", "km?", _user_context(), conv)
        assert result == "Fallback response"

    @pytest.mark.asyncio
    async def test_continue_flow_gathering(self, engine):
        from services.conversation_manager import ConversationState
        conv = self._make_conv(state_value="gathering_params", in_flow=True, flow="booking")
        state_mock = MagicMock()
        state_mock.value = "gathering_params"
        state_mock.__eq__ = lambda self, other: other == ConversationState.GATHERING_PARAMS
        conv.get_state.return_value = state_mock

        decision = MagicMock()
        decision.action = "continue_flow"
        decision.confidence = 0.9

        engine.unified_router = AsyncMock()
        engine.unified_router.route = AsyncMock(return_value=decision)
        engine._unified_router_initialized = True
        engine._flow_handler.handle_gathering = AsyncMock(return_value="Gathering...")

        result = await engine._process_with_state("sender", "sutra", _user_context(), conv)
        assert result == "Gathering..."

    @pytest.mark.asyncio
    async def test_lazy_init_unified_router(self, engine):
        engine._unified_router_initialized = False
        engine.unified_router = None

        mock_router = AsyncMock()
        decision = MagicMock()
        decision.action = "direct_response"
        decision.response = "ok"
        decision.confidence = 0.9
        mock_router.route = AsyncMock(return_value=decision)
        mock_router.set_registry = MagicMock()

        conv = self._make_conv()

        with patch("services.engine.get_unified_router", new_callable=AsyncMock, return_value=mock_router):
            result = await engine._process_with_state("sender", "test", _user_context(), conv)
            assert engine._unified_router_initialized is True


# ===========================================================================
# _handle_flow_start
# ===========================================================================

class TestHandleFlowStart:
    @pytest.mark.asyncio
    async def test_booking_flow(self, engine):
        decision = MagicMock()
        decision.flow_type = "booking"
        decision.params = {"from": "sutra"}

        engine._flow_executors.handle_booking_flow = AsyncMock(return_value="Booking started")

        result = await engine._handle_flow_start(decision, "rezerviraj", _user_context(), MagicMock())
        assert result == "Booking started"

    @pytest.mark.asyncio
    async def test_mileage_flow(self, engine):
        decision = MagicMock()
        decision.flow_type = "mileage"
        decision.params = {}

        engine._flow_executors.handle_mileage_input_flow = AsyncMock(return_value="Mileage started")

        result = await engine._handle_flow_start(decision, "km", _user_context(), MagicMock())
        assert result == "Mileage started"

    @pytest.mark.asyncio
    async def test_case_flow(self, engine):
        decision = MagicMock()
        decision.flow_type = "case"
        decision.params = {}

        engine._flow_executors.handle_case_creation_flow = AsyncMock(return_value="Case started")

        result = await engine._handle_flow_start(decision, "prijavi", _user_context(), MagicMock())
        assert result == "Case started"

    @pytest.mark.asyncio
    async def test_unknown_flow_type(self, engine):
        decision = MagicMock()
        decision.flow_type = "unknown"
        decision.params = {}

        result = await engine._handle_flow_start(decision, "?", _user_context(), MagicMock())
        assert "Neispravan" in result


# ===========================================================================
# _handle_new_request
# ===========================================================================

class TestHandleNewRequest:
    @pytest.mark.asyncio
    async def test_deterministic_direct_response_template(self, engine):
        route = MagicMock()
        route.matched = True
        route.flow_type = "direct_response"
        route.response_template = "Pozdrav!"
        route.tool_name = None

        engine.query_router.route = MagicMock(return_value=route)

        conv = AsyncMock()
        result = await engine._handle_new_request("sender", "bok", _user_context(), conv)
        assert result == "Pozdrav!"

    @pytest.mark.asyncio
    async def test_deterministic_booking_flow(self, engine):
        route = MagicMock()
        route.matched = True
        route.flow_type = "booking"
        route.tool_name = "get_AvailableVehicles"

        engine.query_router.route = MagicMock(return_value=route)
        engine._flow_executors.handle_booking_flow = AsyncMock(return_value="Booking!")

        conv = AsyncMock()
        result = await engine._handle_new_request("sender", "rezerviraj", _user_context(), conv)
        assert result == "Booking!"

    @pytest.mark.asyncio
    async def test_deterministic_simple_tool(self, engine):
        route = MagicMock()
        route.matched = True
        route.flow_type = "simple"
        route.tool_name = "get_MasterData"

        engine.query_router.route = MagicMock(return_value=route)
        engine._deterministic_executor.execute = AsyncMock(return_value="Km: 50000")

        conv = AsyncMock()
        result = await engine._handle_new_request("sender", "km?", _user_context(), conv)
        assert result == "Km: 50000"

    @pytest.mark.asyncio
    async def test_deterministic_fails_falls_to_llm(self, engine):
        route = MagicMock()
        route.matched = True
        route.flow_type = "simple"
        route.tool_name = "get_MasterData"

        engine.query_router.route = MagicMock(return_value=route)
        engine._deterministic_executor.execute = AsyncMock(return_value=None)
        engine._deterministic_executor.pre_resolve_entity_references = AsyncMock(return_value=None)

        # Mock registry for tool search
        engine.registry.find_relevant_tools_with_scores = AsyncMock(return_value=[])

        # Mock chain planner
        plan = MagicMock()
        plan.understanding = "Test"
        plan.is_simple = True
        plan.direct_response = "No tools found answer"
        plan.missing_data = []
        plan.has_all_data = True
        plan.primary_path = []
        plan.fallback_paths = {}
        engine.chain_planner.create_plan = AsyncMock(return_value=plan)

        conv = AsyncMock()
        conv.is_in_flow.return_value = False
        result = await engine._handle_new_request("sender", "test?", _user_context(), conv)
        assert result == "No tools found answer"

    @pytest.mark.asyncio
    async def test_no_route_match_uses_chain_planner(self, engine):
        route = MagicMock()
        route.matched = False

        engine.query_router.route = MagicMock(return_value=route)
        engine._deterministic_executor.pre_resolve_entity_references = AsyncMock(return_value=None)

        engine.registry.find_relevant_tools_with_scores = AsyncMock(return_value=[
            {"name": "get_MasterData", "score": 0.8, "schema": {"name": "get_MasterData"}}
        ])

        plan = MagicMock()
        plan.understanding = "km"
        plan.is_simple = True
        plan.direct_response = None
        plan.missing_data = []
        plan.has_all_data = True
        plan.primary_path = [MagicMock(tool_name="get_MasterData")]
        plan.fallback_paths = {}

        engine.chain_planner.create_plan = AsyncMock(return_value=plan)

        # AI returns text response
        engine._instrumented_ai_call = AsyncMock(return_value={"type": "text", "content": "50000 km"})

        conv = AsyncMock()
        conv.is_in_flow.return_value = False
        result = await engine._handle_new_request("sender", "km?", _user_context(), conv)
        assert "50000" in result

    @pytest.mark.asyncio
    async def test_max_iterations_exhausted(self, engine):
        route = MagicMock()
        route.matched = False

        engine.query_router.route = MagicMock(return_value=route)
        engine._deterministic_executor.pre_resolve_entity_references = AsyncMock(return_value=None)
        engine.registry.find_relevant_tools_with_scores = AsyncMock(return_value=[])

        plan = MagicMock()
        plan.understanding = "test"
        plan.is_simple = True
        plan.direct_response = None
        plan.missing_data = []
        plan.has_all_data = True
        plan.primary_path = []
        plan.fallback_paths = {}

        engine.chain_planner.create_plan = AsyncMock(return_value=plan)

        # Always return tool_call, never finishing
        tool_result = {
            "type": "tool_call",
            "tool": "get_Something",
            "parameters": {},
            "tool_call_id": "tc1"
        }
        engine._instrumented_ai_call = AsyncMock(return_value=tool_result)

        tool_mock = MagicMock()
        tool_mock.method = "GET"
        engine.registry.get_tool = MagicMock(return_value=tool_mock)

        engine._tool_handler.execute_tool_call = AsyncMock(return_value={
            "success": True,
            "data": {"test": 1},
        })

        engine.response_extractor.extract = AsyncMock(return_value=None)

        conv = AsyncMock()
        conv.is_in_flow.return_value = False
        result = await engine._handle_new_request("sender", "test", _user_context(), conv)
        assert "uspio" in result.lower()

    @pytest.mark.asyncio
    async def test_plan_missing_data_starts_gathering(self, engine):
        route = MagicMock()
        route.matched = False

        engine.query_router.route = MagicMock(return_value=route)
        engine._deterministic_executor.pre_resolve_entity_references = AsyncMock(return_value=None)
        engine.registry.find_relevant_tools_with_scores = AsyncMock(return_value=[])

        plan = MagicMock()
        plan.understanding = "booking"
        plan.is_simple = False
        plan.direct_response = None
        plan.missing_data = ["FromTime", "ToTime"]
        plan.has_all_data = False
        plan.primary_path = [MagicMock(tool_name="get_AvailableVehicles")]
        plan.fallback_paths = {}

        engine.chain_planner.create_plan = AsyncMock(return_value=plan)

        with patch("services.engine.DeterministicExecutor") as MockDE:
            MockDE.build_missing_data_prompt = MagicMock(return_value="Navedite FromTime i ToTime")

            conv = AsyncMock()
            conv.is_in_flow.return_value = False
            conv.start_flow = AsyncMock()
            conv.save = AsyncMock()

            result = await engine._handle_new_request("sender", "rezerviraj", _user_context(), conv)
            assert "FromTime" in result or "Navedite" in result

    @pytest.mark.asyncio
    async def test_ai_error_response(self, engine):
        route = MagicMock()
        route.matched = False

        engine.query_router.route = MagicMock(return_value=route)
        engine._deterministic_executor.pre_resolve_entity_references = AsyncMock(return_value=None)
        engine.registry.find_relevant_tools_with_scores = AsyncMock(return_value=[])

        plan = MagicMock()
        plan.understanding = "test"
        plan.is_simple = True
        plan.direct_response = None
        plan.missing_data = []
        plan.has_all_data = True
        plan.primary_path = []
        plan.fallback_paths = {}

        engine.chain_planner.create_plan = AsyncMock(return_value=plan)
        engine._instrumented_ai_call = AsyncMock(return_value={
            "type": "error",
            "content": "AI error occurred"
        })

        conv = AsyncMock()
        conv.is_in_flow.return_value = False
        result = await engine._handle_new_request("sender", "test", _user_context(), conv)
        assert "error" in result.lower() or "AI" in result


# ===========================================================================
# _instrumented_ai_call
# ===========================================================================

class TestInstrumentedAICall:
    @pytest.mark.asyncio
    async def test_success_records_drift(self, engine):
        engine.ai.analyze = AsyncMock(return_value={
            "type": "text",
            "content": "hello",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        })
        engine.ai.model = "gpt-4"
        engine.drift_detector.record_interaction = AsyncMock()
        engine.cost_tracker = AsyncMock()
        engine.cost_tracker.record_usage = AsyncMock()

        result = await engine._instrumented_ai_call(
            messages=[{"role": "user", "content": "test"}]
        )
        assert result["type"] == "text"
        engine.drift_detector.record_interaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_call_records_tools(self, engine):
        engine.ai.analyze = AsyncMock(return_value={
            "type": "tool_call",
            "tool": "get_MasterData",
        })
        engine.ai.model = "gpt-4"
        engine.drift_detector.record_interaction = AsyncMock()
        engine.cost_tracker = None

        result = await engine._instrumented_ai_call(
            messages=[{"role": "user", "content": "test"}]
        )
        assert result["type"] == "tool_call"

    @pytest.mark.asyncio
    async def test_error_response_marks_failure(self, engine):
        engine.ai.analyze = AsyncMock(return_value={
            "type": "error",
            "content": "LLM failed",
        })
        engine.ai.model = "gpt-4"
        engine.drift_detector.record_interaction = AsyncMock()
        engine.cost_tracker = None

        result = await engine._instrumented_ai_call(
            messages=[{"role": "user", "content": "test"}]
        )
        assert result["type"] == "error"

    @pytest.mark.asyncio
    async def test_exception_in_ai_analyze(self, engine):
        engine.ai.analyze = AsyncMock(side_effect=RuntimeError("boom"))
        engine.ai.model = "gpt-4"
        engine.drift_detector.record_interaction = AsyncMock()
        engine.cost_tracker = None

        with pytest.raises(RuntimeError, match="boom"):
            await engine._instrumented_ai_call(
                messages=[{"role": "user", "content": "test"}]
            )
        # Drift should still be recorded in finally block
        engine.drift_detector.record_interaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_drift_failure_doesnt_crash(self, engine):
        engine.ai.analyze = AsyncMock(return_value={"type": "text", "content": "ok"})
        engine.ai.model = "gpt-4"
        engine.drift_detector.record_interaction = AsyncMock(side_effect=RuntimeError("drift fail"))
        engine.cost_tracker = None

        # Should not raise
        result = await engine._instrumented_ai_call(
            messages=[{"role": "user", "content": "test"}]
        )
        assert result["type"] == "text"

    @pytest.mark.asyncio
    async def test_cost_tracker_records_usage(self, engine):
        engine.ai.analyze = AsyncMock(return_value={
            "type": "text",
            "content": "ok",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        })
        engine.ai.model = "gpt-4"
        engine.drift_detector.record_interaction = AsyncMock()
        engine.cost_tracker = AsyncMock()
        engine.cost_tracker.record_usage = AsyncMock()

        await engine._instrumented_ai_call(
            messages=[{"role": "user", "content": "test"}],
            user_context=_user_context()
        )
        engine.cost_tracker.record_usage.assert_called_once()

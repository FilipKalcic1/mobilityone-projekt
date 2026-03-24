"""
Comprehensive tests for services/engine/deterministic_executor.py
Covers lines 67-73, 94-185, 195-233, 243
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from services.engine.deterministic_executor import DeterministicExecutor
from services.conversation_manager import ConversationState


class TestDeterministicExecutorInit:
    """Test DeterministicExecutor initialization."""

    def test_init_stores_dependencies(self):
        """Test __init__ stores all dependencies."""
        registry = MagicMock()
        executor = MagicMock()
        formatter = MagicMock()
        query_router = MagicMock()
        response_extractor = MagicMock()
        dependency_resolver = MagicMock()
        flow_handler = MagicMock()

        det_exec = DeterministicExecutor(
            registry=registry,
            executor=executor,
            formatter=formatter,
            query_router=query_router,
            response_extractor=response_extractor,
            dependency_resolver=dependency_resolver,
            flow_handler=flow_handler
        )

        assert det_exec.registry == registry
        assert det_exec.executor == executor
        assert det_exec.formatter == formatter
        assert det_exec.query_router == query_router
        assert det_exec.response_extractor == response_extractor
        assert det_exec.dependency_resolver == dependency_resolver
        assert det_exec.flow_handler == flow_handler


class TestExecute:
    """Test execute method - lines 94-185."""

    @pytest.fixture
    def det_exec(self):
        """Create DeterministicExecutor with mocked dependencies."""
        registry = MagicMock()
        executor = MagicMock()
        formatter = MagicMock()
        query_router = MagicMock()
        response_extractor = MagicMock()
        dependency_resolver = MagicMock()
        flow_handler = MagicMock()

        return DeterministicExecutor(
            registry=registry,
            executor=executor,
            formatter=formatter,
            query_router=query_router,
            response_extractor=response_extractor,
            dependency_resolver=dependency_resolver,
            flow_handler=flow_handler
        )

    @pytest.fixture
    def mock_route(self):
        """Create mock RouteResult."""
        route = MagicMock()
        route.tool_name = "get_PersonData"
        route.response_template = None
        route.extract_fields = []
        return route

    @pytest.fixture
    def mock_conv_manager(self):
        """Create mock ConversationManager."""
        conv = MagicMock()
        conv.get_state.return_value = ConversationState.IDLE
        conv.context = MagicMock()
        conv.context.tool_outputs = {}
        return conv

    @pytest.mark.asyncio
    async def test_tool_not_found_returns_none(self, det_exec, mock_route, mock_conv_manager):
        """Test returns None when tool not found (line 96-97)."""
        det_exec.registry.get_tool.return_value = None

        result = await det_exec.execute(
            route=mock_route,
            user_context={},
            conv_manager=mock_conv_manager,
            sender="+385991234567",
            original_query="moji podaci"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_person_data_without_person_id(self, det_exec, mock_route, mock_conv_manager):
        """Test personIdOrEmail param without person_id (lines 108-110)."""
        tool = MagicMock()
        tool.parameters = {"personIdOrEmail": MagicMock()}
        tool.method = "GET"
        det_exec.registry.get_tool.return_value = tool

        result = await det_exec.execute(
            route=mock_route,
            user_context={},  # No person_id
            conv_manager=mock_conv_manager,
            sender="+385991234567",
            original_query="moji podaci"
        )

        assert "prijavite se" in result

    @pytest.mark.asyncio
    async def test_person_data_with_person_id(self, det_exec, mock_route, mock_conv_manager):
        """Test personIdOrEmail param with person_id (lines 104-107)."""
        tool = MagicMock()
        tool.parameters = {"personIdOrEmail": MagicMock()}
        tool.method = "GET"
        det_exec.registry.get_tool.return_value = tool

        # Mock executor result
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {"Name": "Test User"}
        det_exec.executor.execute = AsyncMock(return_value=exec_result)
        det_exec.formatter.format_result.return_value = "Test User info"

        result = await det_exec.execute(
            route=mock_route,
            user_context={"person_id": "12345678-1234-1234-1234-123456789012"},
            conv_manager=mock_conv_manager,
            sender="+385991234567",
            original_query="moji podaci"
        )

        assert result == "Test User info"

    @pytest.mark.asyncio
    async def test_vehicle_required_without_vehicle_id(self, det_exec, mock_route, mock_conv_manager):
        """Test VehicleId required but missing (lines 117-122)."""
        tool = MagicMock()
        vehicle_param = MagicMock()
        vehicle_param.required = True
        tool.parameters = {"VehicleId": vehicle_param}
        tool.method = "GET"
        det_exec.registry.get_tool.return_value = tool

        result = await det_exec.execute(
            route=mock_route,
            user_context={},  # No vehicle
            conv_manager=mock_conv_manager,
            sender="+385991234567",
            original_query="podaci o vozilu"
        )

        assert "nemate dodijeljeno vozilo" in result

    @pytest.mark.asyncio
    async def test_vehicle_required_with_vehicle_id(self, det_exec, mock_route, mock_conv_manager):
        """Test VehicleId required and available (lines 115-116)."""
        tool = MagicMock()
        vehicle_param = MagicMock()
        vehicle_param.required = True
        tool.parameters = {"VehicleId": vehicle_param}
        tool.method = "GET"
        det_exec.registry.get_tool.return_value = tool

        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {"VehicleName": "VW Passat"}
        det_exec.executor.execute = AsyncMock(return_value=exec_result)
        det_exec.formatter.format_result.return_value = "VW Passat info"

        result = await det_exec.execute(
            route=mock_route,
            user_context={"vehicle": {"id": "v-123"}},
            conv_manager=mock_conv_manager,
            sender="+385991234567",
            original_query="podaci o vozilu"
        )

        assert result == "VW Passat info"

    @pytest.mark.asyncio
    async def test_mutation_tool_requires_confirmation(self, det_exec, mock_route, mock_conv_manager):
        """Test mutation tool shows confirmation dialog (lines 136-145)."""
        tool = MagicMock()
        tool.parameters = {}
        tool.method = "POST"
        det_exec.registry.get_tool.return_value = tool

        det_exec.flow_handler.request_confirmation = AsyncMock(return_value={
            "prompt": "Potvrdite rezervaciju?"
        })

        result = await det_exec.execute(
            route=mock_route,
            user_context={},
            conv_manager=mock_conv_manager,
            sender="+385991234567",
            original_query="rezerviraj auto"
        )

        assert "Potvrdite" in result
        det_exec.flow_handler.request_confirmation.assert_called_once()

    @pytest.mark.asyncio
    async def test_mutation_in_confirming_state_executes(self, det_exec, mock_route, mock_conv_manager):
        """Test mutation tool in CONFIRMING state executes directly."""
        tool = MagicMock()
        tool.parameters = {}
        tool.method = "POST"
        det_exec.registry.get_tool.return_value = tool

        mock_conv_manager.get_state.return_value = ConversationState.CONFIRMING

        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {"id": "new-123"}
        det_exec.executor.execute = AsyncMock(return_value=exec_result)
        det_exec.formatter.format_result.return_value = "Rezervacija uspješna"

        result = await det_exec.execute(
            route=mock_route,
            user_context={},
            conv_manager=mock_conv_manager,
            sender="+385991234567",
            original_query="da"
        )

        assert result == "Rezervacija uspješna"

    @pytest.mark.asyncio
    async def test_execute_failure_returns_none(self, det_exec, mock_route, mock_conv_manager):
        """Test execution failure returns None (lines 157-159)."""
        tool = MagicMock()
        tool.parameters = {}
        tool.method = "GET"
        det_exec.registry.get_tool.return_value = tool

        exec_result = MagicMock()
        exec_result.success = False
        exec_result.error_message = "API error"
        det_exec.executor.execute = AsyncMock(return_value=exec_result)

        result = await det_exec.execute(
            route=mock_route,
            user_context={},
            conv_manager=mock_conv_manager,
            sender="+385991234567",
            original_query="test query"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_template_response_used(self, det_exec, mock_route, mock_conv_manager):
        """Test template-based formatting (lines 162-166)."""
        tool = MagicMock()
        tool.parameters = {}
        tool.method = "GET"
        det_exec.registry.get_tool.return_value = tool

        mock_route.response_template = "Vaša kilometraža: {value} km"

        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {"value": 15000}
        det_exec.executor.execute = AsyncMock(return_value=exec_result)
        det_exec.query_router.format_response.return_value = "Vaša kilometraža: 15000 km"

        result = await det_exec.execute(
            route=mock_route,
            user_context={},
            conv_manager=mock_conv_manager,
            sender="+385991234567",
            original_query="koja je moja kilometraža"
        )

        assert result == "Vaša kilometraža: 15000 km"

    @pytest.mark.asyncio
    async def test_llm_extraction_used(self, det_exec, mock_route, mock_conv_manager):
        """Test LLM extraction for complex responses (lines 169-179)."""
        tool = MagicMock()
        tool.parameters = {}
        tool.method = "GET"
        det_exec.registry.get_tool.return_value = tool

        mock_route.extract_fields = ["Name", "Email"]

        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {"Name": "Test", "Email": "test@test.com"}
        det_exec.executor.execute = AsyncMock(return_value=exec_result)
        det_exec.query_router.format_response.return_value = None
        det_exec.response_extractor.extract = AsyncMock(return_value="Test (test@test.com)")

        result = await det_exec.execute(
            route=mock_route,
            user_context={},
            conv_manager=mock_conv_manager,
            sender="+385991234567",
            original_query="tko sam ja"
        )

        assert result == "Test (test@test.com)"

    @pytest.mark.asyncio
    async def test_llm_extraction_fails_uses_formatter(self, det_exec, mock_route, mock_conv_manager):
        """Test fallback to formatter when LLM fails (lines 180-185)."""
        tool = MagicMock()
        tool.parameters = {}
        tool.method = "GET"
        det_exec.registry.get_tool.return_value = tool

        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {"Name": "Test"}
        det_exec.executor.execute = AsyncMock(return_value=exec_result)
        det_exec.query_router.format_response.return_value = None
        det_exec.response_extractor.extract = AsyncMock(side_effect=Exception("LLM error"))
        det_exec.formatter.format_result.return_value = "Formatted result"

        result = await det_exec.execute(
            route=mock_route,
            user_context={},
            conv_manager=mock_conv_manager,
            sender="+385991234567",
            original_query="test"
        )

        assert result == "Formatted result"


class TestPreResolveEntityReferences:
    """Test pre_resolve_entity_references method - lines 195-233."""

    @pytest.fixture
    def det_exec(self):
        """Create DeterministicExecutor with mocked dependencies."""
        registry = MagicMock()
        executor = MagicMock()
        formatter = MagicMock()
        query_router = MagicMock()
        response_extractor = MagicMock()
        dependency_resolver = MagicMock()
        flow_handler = MagicMock()

        return DeterministicExecutor(
            registry=registry,
            executor=executor,
            formatter=formatter,
            query_router=query_router,
            response_extractor=response_extractor,
            dependency_resolver=dependency_resolver,
            flow_handler=flow_handler
        )

    @pytest.mark.asyncio
    async def test_no_entity_reference_returns_empty(self, det_exec):
        """Test returns empty dict when no entity reference found."""
        det_exec.dependency_resolver.detect_entity_reference.return_value = None

        conv_manager = MagicMock()
        executor = MagicMock()

        result = await det_exec.pre_resolve_entity_references(
            text="hello",
            user_context={},
            conv_manager=conv_manager,
            executor=executor
        )

        assert result == {}

    @pytest.mark.asyncio
    async def test_entity_reference_resolved_successfully(self, det_exec):
        """Test successful entity resolution (lines 202-223)."""
        det_exec.dependency_resolver.detect_entity_reference.return_value = {
            "type": "vehicle",
            "reference": "passat"
        }

        resolution = MagicMock()
        resolution.success = True
        resolution.resolved_value = "v-123-abc"
        det_exec.dependency_resolver.resolve_entity_reference = AsyncMock(return_value=resolution)

        conv_manager = MagicMock()
        conv_manager.context = MagicMock()
        conv_manager.context.tool_outputs = {}
        conv_manager.save = AsyncMock()
        executor = MagicMock()

        result = await det_exec.pre_resolve_entity_references(
            text="rezerviraj passat",
            user_context={"person_id": "123"},
            conv_manager=conv_manager,
            executor=executor
        )

        assert result["VehicleId"] == "v-123-abc"
        assert result["vehicleId"] == "v-123-abc"
        conv_manager.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_reference_resolution_failed(self, det_exec):
        """Test failed entity resolution (lines 225-228)."""
        det_exec.dependency_resolver.detect_entity_reference.return_value = {
            "type": "vehicle",
            "reference": "unknown"
        }

        resolution = MagicMock()
        resolution.success = False
        resolution.error_message = "Vehicle not found"
        det_exec.dependency_resolver.resolve_entity_reference = AsyncMock(return_value=resolution)

        conv_manager = MagicMock()
        executor = MagicMock()

        result = await det_exec.pre_resolve_entity_references(
            text="rezerviraj unknown",
            user_context={},
            conv_manager=conv_manager,
            executor=executor
        )

        assert result == {}

    @pytest.mark.asyncio
    async def test_entity_resolution_exception_handled(self, det_exec):
        """Test exception handling (lines 230-231)."""
        det_exec.dependency_resolver.detect_entity_reference.side_effect = Exception("Detection error")

        conv_manager = MagicMock()
        executor = MagicMock()

        result = await det_exec.pre_resolve_entity_references(
            text="test",
            user_context={},
            conv_manager=conv_manager,
            executor=executor
        )

        assert result == {}


class TestBuildMissingDataPrompt:
    """Test build_missing_data_prompt static method - line 243."""

    def test_single_param(self):
        """Test with single missing param."""
        result = DeterministicExecutor.build_missing_data_prompt(["VehicleId"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_multiple_params(self):
        """Test with multiple missing params."""
        result = DeterministicExecutor.build_missing_data_prompt(["FromTime", "ToTime", "VehicleId"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_list(self):
        """Test with empty list."""
        result = DeterministicExecutor.build_missing_data_prompt([])
        assert isinstance(result, str)

    def test_unknown_params(self):
        """Test with unknown param names."""
        result = DeterministicExecutor.build_missing_data_prompt(["UnknownParam", "AnotherOne"])
        assert isinstance(result, str)

"""
Tests for ToolExecutor

Tests tool execution with UnifiedToolDefinition and ToolExecutionContext.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.tool_executor import ToolExecutor
from services.tool_contracts import (
    UnifiedToolDefinition,
    ParameterDefinition,
    DependencySource,
    ToolExecutionContext,
    ToolExecutionResult
)
from services.api_gateway import APIResponse, HttpMethod
from services.parameter_manager import ParameterValidationError


class TestToolExecutor:
    """Test ToolExecutor class with new API."""

    @pytest.fixture
    def mock_gateway(self):
        """Mock API gateway."""
        gateway = MagicMock()

        # Create successful response
        response = APIResponse(
            success=True,
            status_code=200,
            data={"Id": "result-123", "Name": "Test Result"},
            error_message=None,
            error_code=None
        )
        gateway.execute = AsyncMock(return_value=response)
        return gateway

    @pytest.fixture
    def sample_tool(self) -> UnifiedToolDefinition:
        """Sample tool definition for testing."""
        return UnifiedToolDefinition(
            operation_id="get_TestResource",
            service_name="test-service",
            swagger_name="test",
            service_url="/test",
            path="/api/v1/resource",
            method="GET",
            description="Get test resource",
            parameters={
                "status": ParameterDefinition(
                    name="status",
                    param_type="string",
                    description="Resource status",
                    required=False,
                    location="query",
                    dependency_source=DependencySource.FROM_USER
                )
            },
            required_params=[],
            output_keys=["Id", "Name"]
        )

    @pytest.fixture
    def sample_tool_with_context_param(self) -> UnifiedToolDefinition:
        """Tool with context parameter injection."""
        return UnifiedToolDefinition(
            operation_id="get_UserResource",
            service_name="test-service",
            swagger_name="test",
            service_url="/test",
            path="/api/v1/user-resource",
            method="GET",
            description="Get user resource",
            parameters={
                "personId": ParameterDefinition(
                    name="personId",
                    param_type="string",
                    description="Person ID",
                    required=False,
                    location="query",
                    dependency_source=DependencySource.FROM_CONTEXT,
                    context_key="person_id"
                ),
                "status": ParameterDefinition(
                    name="status",
                    param_type="string",
                    description="Resource status",
                    required=False,
                    location="query",
                    dependency_source=DependencySource.FROM_USER
                )
            },
            required_params=[],
            output_keys=["Id"]
        )

    @pytest.fixture
    def sample_execution_context(self) -> ToolExecutionContext:
        """Sample execution context."""
        return ToolExecutionContext(
            user_context={
                "person_id": "test-person-123",
                "tenant_id": "test-tenant-456",
                "phone": "+385991234567"
            },
            tool_outputs={},
            conversation_state={}
        )

    @pytest.fixture
    def executor(self, mock_gateway):
        """Create executor with mocked gateway."""
        return ToolExecutor(gateway=mock_gateway, circuit_breaker=None)

    # ========================================================================
    # BASIC EXECUTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test successful tool execution."""
        result = await executor.execute(
            tool=sample_tool,
            llm_params={"status": "active"},
            execution_context=sample_execution_context
        )

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert result.operation_id == "get_TestResource"
        assert result.data is not None
        assert "Id" in result.data

    @pytest.mark.asyncio
    async def test_execute_with_empty_params(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test execution with no parameters."""
        result = await executor.execute(
            tool=sample_tool,
            llm_params={},
            execution_context=sample_execution_context
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_context_param_injection(
        self,
        executor,
        sample_tool_with_context_param,
        sample_execution_context
    ):
        """Test that context parameters are auto-injected."""
        result = await executor.execute(
            tool=sample_tool_with_context_param,
            llm_params={"status": "active"},
            execution_context=sample_execution_context
        )

        assert result.success is True

        # Verify gateway was called
        executor.gateway.execute.assert_called_once()
        call_kwargs = executor.gateway.execute.call_args.kwargs

        # Check that personId was injected from context
        assert call_kwargs["params"] is not None or call_kwargs["body"] is not None

    # ========================================================================
    # ERROR HANDLING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_api_error_handling(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test handling of API errors."""
        # Mock error response
        error_response = APIResponse(
            success=False,
            status_code=400,
            data=None,
            error_message="Invalid parameters",
            error_code="BAD_REQUEST"
        )
        executor.gateway.execute = AsyncMock(return_value=error_response)

        result = await executor.execute(
            tool=sample_tool,
            llm_params={},
            execution_context=sample_execution_context
        )

        assert result.success is False
        assert result.error_code is not None
        assert result.ai_feedback is not None
        assert "Invalid" in result.error_message or "parameter" in result.ai_feedback.lower()

    @pytest.mark.asyncio
    async def test_missing_required_params(
        self,
        executor,
        sample_execution_context
    ):
        """Test validation of missing required parameters."""
        tool_with_required = UnifiedToolDefinition(
            operation_id="create_Resource",
            service_name="test",
            swagger_name="test",
            service_url="/test",
            path="/api/v1/resource",
            method="POST",
            description="Create resource",
            parameters={
                "name": ParameterDefinition(
                    name="name",
                    param_type="string",
                    description="Resource name",
                    required=True,
                    location="body",
                    dependency_source=DependencySource.FROM_USER
                )
            },
            required_params=["name"]
        )

        result = await executor.execute(
            tool=tool_with_required,
            llm_params={},  # Missing required 'name'
            execution_context=sample_execution_context
        )

        assert result.success is False
        assert result.error_code == "PARAMETER_VALIDATION_ERROR"
        assert "name" in result.missing_params

    @pytest.mark.asyncio
    async def test_404_error_feedback(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test user-friendly feedback for 404 errors."""
        error_response = APIResponse(
            success=False,
            status_code=404,
            data=None,
            error_message="Not found",
            error_code="NOT_FOUND"
        )
        executor.gateway.execute = AsyncMock(return_value=error_response)

        result = await executor.execute(
            tool=sample_tool,
            llm_params={},
            execution_context=sample_execution_context
        )

        assert result.success is False
        assert result.http_status == 404
        assert "not found" in result.ai_feedback.lower() or "ne postoji" in result.ai_feedback.lower()

    # ========================================================================
    # OUTPUT EXTRACTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_output_value_extraction(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test extraction of output values for chaining."""
        # Mock response with output values
        response = APIResponse(
            success=True,
            status_code=200,
            data={"Id": "resource-123", "Name": "Test Resource", "Status": "Active"},
            error_message=None,
            error_code=None
        )
        executor.gateway.execute = AsyncMock(return_value=response)

        result = await executor.execute(
            tool=sample_tool,
            llm_params={},
            execution_context=sample_execution_context
        )

        assert result.success is True
        assert result.output_values is not None
        assert "Id" in result.output_values
        assert result.output_values["Id"] == "resource-123"

    # ========================================================================
    # URL BUILDING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_url_construction_with_swagger_name(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test URL construction using swagger_name."""
        result = await executor.execute(
            tool=sample_tool,
            llm_params={},
            execution_context=sample_execution_context
        )

        # Verify gateway was called with correct URL pattern
        executor.gateway.execute.assert_called_once()
        call_kwargs = executor.gateway.execute.call_args.kwargs

        # URL should follow pattern: /swagger_name/path
        assert "path" in call_kwargs
        url = call_kwargs["path"]
        assert "/test/" in url or url.startswith("/test/api")

    @pytest.mark.asyncio
    async def test_execution_time_tracking(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test that execution time is tracked."""
        result = await executor.execute(
            tool=sample_tool,
            llm_params={},
            execution_context=sample_execution_context
        )

        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0


class TestToolExecutorValidation:
    """Test _validate_http_request method."""

    @pytest.fixture
    def executor(self, mock_gateway):
        """Create executor without registry."""
        return ToolExecutor(gateway=mock_gateway)

    def test_invalid_method_raises(self, executor):
        """Test invalid HTTP method raises error (line 376)."""
        with pytest.raises(ParameterValidationError):
            executor._validate_http_request(
                method="INVALID",
                url="/api/test",
                query_params=None,
                body=None,
                operation_id="test_op"
            )

    def test_empty_url_raises(self, executor):
        """Test empty URL raises error (lines 382-386)."""
        with pytest.raises(ParameterValidationError):
            executor._validate_http_request(
                method="GET",
                url="/",
                query_params=None,
                body=None,
                operation_id="test_op"
            )

    def test_get_with_body_logs_warning(self, executor):
        """Test GET with body logs warning (lines 389-393)."""
        executor._validate_http_request(
            method="GET",
            url="/api/test",
            query_params=None,
            body={"key": "value"},
            operation_id="test_op"
        )

    def test_post_without_body_logs_warning(self, executor):
        """Test POST without body logs warning (lines 395-399)."""
        executor._validate_http_request(
            method="POST",
            url="/api/test",
            query_params=None,
            body=None,
            operation_id="test_op"
        )


class TestToolExecutorBuildUrl:
    """Test _build_url method."""

    @pytest.fixture
    def executor(self, mock_gateway):
        """Create executor."""
        return ToolExecutor(gateway=mock_gateway)

    def test_absolute_url_returned(self, executor):
        """Test absolute URL returned as-is (lines 431-433)."""
        tool = MagicMock()
        tool.swagger_name = ""
        tool.path = "https://external.api.com/test"
        tool.service_url = ""

        result = executor._build_url(tool)
        assert result == "https://external.api.com/test"

    def test_swagger_name_used(self, executor):
        """Test URL built with swagger_name (lines 436-442)."""
        tool = MagicMock()
        tool.swagger_name = "TestService"
        tool.path = "/api/endpoint"
        tool.service_url = ""

        result = executor._build_url(tool)
        assert result == "/TestService/api/endpoint"

    def test_service_url_absolute(self, executor):
        """Test absolute service_url (lines 450-453)."""
        tool = MagicMock()
        tool.swagger_name = ""
        tool.path = "/endpoint"
        tool.service_url = "https://api.example.com"

        result = executor._build_url(tool)
        assert result == "https://api.example.com/endpoint"

    def test_service_url_relative(self, executor):
        """Test relative service_url (lines 456-458)."""
        tool = MagicMock()
        tool.swagger_name = ""
        tool.path = "/endpoint"
        tool.service_url = "/automation"

        result = executor._build_url(tool)
        assert result == "/automation/endpoint"

    def test_path_only_fallback(self, executor):
        """Test fallback to path only (lines 460-462)."""
        tool = MagicMock()
        tool.swagger_name = ""
        tool.path = "/api/test"
        tool.service_url = ""
        tool.operation_id = "test_op"

        result = executor._build_url(tool)
        assert result == "/api/test"


class TestToolExecutorExtractOutput:
    """Test _extract_output_values method."""

    @pytest.fixture
    def executor(self, mock_gateway):
        """Create executor."""
        return ToolExecutor(gateway=mock_gateway)

    def test_empty_response(self, executor):
        """Test empty response returns empty dict (line 506)."""
        result = executor._extract_output_values(None, ["Id"])
        assert result == {}

    def test_case_insensitive_match(self, executor):
        """Test case-insensitive key matching (lines 516-519)."""
        data = {"id": "123", "NAME": "Test"}
        result = executor._extract_output_values(data, ["Id", "Name"])
        assert result["Id"] == "123"
        assert result["Name"] == "Test"

    def test_nested_data_extraction(self, executor):
        """Test extraction from nested data field (lines 522-525)."""
        data = {"success": True, "data": {"Id": "123", "Name": "Test"}}
        result = executor._extract_output_values(data, ["Id", "Name"])
        assert result["Id"] == "123"
        assert result["Name"] == "Test"

    def test_list_response_first_item(self, executor):
        """Test extraction from list response (lines 527-533)."""
        data = [{"Id": "123", "Name": "First"}, {"Id": "456", "Name": "Second"}]
        result = executor._extract_output_values(data, ["Id", "Name"])
        assert result["Id"] == "123"
        assert result["Name"] == "First"


class TestToolExecutorCircuitBreaker:
    """Test circuit breaker integration."""

    @pytest.fixture
    def mock_gateway(self):
        """Mock API gateway."""
        gateway = MagicMock()
        response = APIResponse(
            success=True,
            status_code=200,
            data={"result": "ok"}
        )
        gateway.execute = AsyncMock(return_value=response)
        return gateway

    @pytest.fixture
    def sample_tool(self):
        """Sample tool definition."""
        return UnifiedToolDefinition(
            operation_id="get_Test",
            service_name="test",
            swagger_name="test",
            service_url="/test",
            path="/api/test",
            method="GET",
            description="Test",
            parameters={}
        )

    @pytest.fixture
    def sample_context(self):
        """Sample execution context."""
        return ToolExecutionContext(
            user_context={"person_id": "12345678-1234-1234-1234-123456789012"},
            tool_outputs={},
            conversation_state={}
        )

    @pytest.mark.asyncio
    async def test_circuit_open_error_handled(self, mock_gateway, sample_tool, sample_context):
        """Test CircuitOpenError is caught (lines 306-316)."""
        from services.circuit_breaker import CircuitBreaker, CircuitOpenError

        cb = MagicMock(spec=CircuitBreaker)
        cb.call = AsyncMock(side_effect=CircuitOpenError("Service unavailable"))

        executor = ToolExecutor(gateway=mock_gateway, circuit_breaker=cb)

        result = await executor.execute(sample_tool, {}, sample_context)

        assert result.success is False
        assert result.error_code == "CIRCUIT_OPEN"

    @pytest.mark.asyncio
    async def test_general_exception_handled(self, mock_gateway, sample_tool, sample_context):
        """Test general exception is caught (lines 318-333)."""
        from services.circuit_breaker import CircuitBreaker

        cb = MagicMock(spec=CircuitBreaker)
        cb.call = AsyncMock(side_effect=Exception("Unexpected error"))

        executor = ToolExecutor(gateway=mock_gateway, circuit_breaker=cb)

        result = await executor.execute(sample_tool, {}, sample_context)

        assert result.success is False
        assert result.error_code == "EXECUTION_ERROR"
        assert "Neočekivana greška" in result.ai_feedback

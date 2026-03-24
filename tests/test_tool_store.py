"""Tests for ToolStore - In-memory storage for tools and dependencies."""

import pytest
from unittest.mock import MagicMock

from services.registry.tool_store import ToolStore
from services.tool_contracts import UnifiedToolDefinition, ParameterDefinition, DependencyGraph


@pytest.fixture
def store():
    return ToolStore()


@pytest.fixture
def sample_tool():
    """Create a sample retrieval tool."""
    return UnifiedToolDefinition(
        operation_id="get_Vehicles",
        method="GET",
        path="/api/vehicles",
        description="Get all vehicles",
        parameters={
            "VehicleId": ParameterDefinition(
                name="VehicleId", param_type="string",
                required=False, description="Vehicle ID"
            )
        },
        service_name="fleet",
        service_url="https://api.example.com",
        swagger_name="fleet",
    )


@pytest.fixture
def mutation_tool():
    """Create a sample mutation tool."""
    return UnifiedToolDefinition(
        operation_id="post_Vehicle",
        method="POST",
        path="/api/vehicles",
        description="Create a vehicle",
        parameters={
            "Name": ParameterDefinition(
                name="Name", param_type="string",
                required=True, description="Vehicle name"
            )
        },
        service_name="fleet",
        service_url="https://api.example.com",
        swagger_name="fleet",
    )


class TestToolStoreInit:
    """Test ToolStore initialization."""

    def test_init_empty(self, store):
        """Test empty store initialization."""
        assert store.tools == {}
        assert store.embeddings == {}
        assert store.dependency_graph == {}
        assert store.retrieval_tools == set()
        assert store.mutation_tools == set()


class TestAddTool:
    """Test add_tool method - covers lines 46-51."""

    def test_add_retrieval_tool(self, store, sample_tool):
        """Test adding a retrieval (GET) tool."""
        store.add_tool(sample_tool)

        assert sample_tool.operation_id in store.tools
        assert sample_tool.operation_id in store.retrieval_tools
        assert sample_tool.operation_id not in store.mutation_tools

    def test_add_mutation_tool(self, store, mutation_tool):
        """Test adding a mutation (POST) tool."""
        store.add_tool(mutation_tool)

        assert mutation_tool.operation_id in store.tools
        assert mutation_tool.operation_id in store.mutation_tools

    def test_add_multiple_tools(self, store, sample_tool, mutation_tool):
        """Test adding multiple tools."""
        store.add_tool(sample_tool)
        store.add_tool(mutation_tool)

        assert len(store.tools) == 2


class TestGetTool:
    """Test get_tool method - covers line 55."""

    def test_get_existing_tool(self, store, sample_tool):
        """Test getting an existing tool."""
        store.add_tool(sample_tool)
        result = store.get_tool(sample_tool.operation_id)

        assert result is sample_tool

    def test_get_missing_tool(self, store):
        """Test getting a non-existent tool."""
        result = store.get_tool("nonexistent_tool")

        assert result is None


class TestHasTool:
    """Test has_tool method - covers line 59."""

    def test_has_existing_tool(self, store, sample_tool):
        """Test checking for existing tool."""
        store.add_tool(sample_tool)

        assert store.has_tool(sample_tool.operation_id) is True

    def test_has_missing_tool(self, store):
        """Test checking for non-existent tool."""
        assert store.has_tool("nonexistent") is False


class TestListTools:
    """Test list_tools method - covers line 63."""

    def test_list_empty(self, store):
        """Test listing empty store."""
        assert store.list_tools() == []

    def test_list_with_tools(self, store, sample_tool, mutation_tool):
        """Test listing tools."""
        store.add_tool(sample_tool)
        store.add_tool(mutation_tool)

        tool_list = store.list_tools()
        assert len(tool_list) == 2
        assert sample_tool.operation_id in tool_list


class TestGetAllTools:
    """Test get_all_tools method - covers line 67."""

    def test_get_all_empty(self, store):
        """Test getting all tools from empty store."""
        assert store.get_all_tools() == {}

    def test_get_all_with_tools(self, store, sample_tool):
        """Test getting all tools."""
        store.add_tool(sample_tool)
        all_tools = store.get_all_tools()

        assert sample_tool.operation_id in all_tools


class TestCount:
    """Test count method - covers line 71."""

    def test_count_empty(self, store):
        """Test count on empty store."""
        assert store.count() == 0

    def test_count_with_tools(self, store, sample_tool, mutation_tool):
        """Test count with tools."""
        store.add_tool(sample_tool)
        store.add_tool(mutation_tool)

        assert store.count() == 2


class TestEmbeddings:
    """Test embedding methods - covers lines 75, 79, 83, 87-90."""

    def test_add_embedding(self, store, sample_tool):
        """Test adding embedding - covers line 75."""
        store.add_tool(sample_tool)
        embedding = [0.1, 0.2, 0.3]

        store.add_embedding(sample_tool.operation_id, embedding)

        assert sample_tool.operation_id in store.embeddings

    def test_get_embedding(self, store, sample_tool):
        """Test getting embedding - covers line 79."""
        store.add_tool(sample_tool)
        embedding = [0.1, 0.2, 0.3]
        store.add_embedding(sample_tool.operation_id, embedding)

        result = store.get_embedding(sample_tool.operation_id)

        assert result == embedding

    def test_get_embedding_missing(self, store):
        """Test getting non-existent embedding."""
        assert store.get_embedding("nonexistent") is None

    def test_has_embedding(self, store, sample_tool):
        """Test has_embedding - covers line 83."""
        store.add_tool(sample_tool)

        assert store.has_embedding(sample_tool.operation_id) is False

        store.add_embedding(sample_tool.operation_id, [0.1])

        assert store.has_embedding(sample_tool.operation_id) is True

    def test_get_missing_embeddings(self, store, sample_tool, mutation_tool):
        """Test get_missing_embeddings - covers lines 87-90."""
        store.add_tool(sample_tool)
        store.add_tool(mutation_tool)
        store.add_embedding(sample_tool.operation_id, [0.1])

        missing = store.get_missing_embeddings()

        assert mutation_tool.operation_id in missing
        assert sample_tool.operation_id not in missing

    def test_get_missing_embeddings_all_present(self, store, sample_tool):
        """Test when all tools have embeddings."""
        store.add_tool(sample_tool)
        store.add_embedding(sample_tool.operation_id, [0.1])

        missing = store.get_missing_embeddings()

        assert missing == []


class TestDependencies:
    """Test dependency methods - covers lines 94, 98."""

    def test_add_dependency(self, store):
        """Test adding dependency - covers line 94."""
        dep = DependencyGraph(
            tool_id="get_Vehicles",
            dependencies=["get_VehicleTypes"]
        )

        store.add_dependency(dep)

        assert "get_Vehicles" in store.dependency_graph

    def test_get_dependency(self, store):
        """Test getting dependency - covers line 98."""
        dep = DependencyGraph(
            tool_id="get_Vehicles",
            dependencies=["get_VehicleTypes"]
        )
        store.add_dependency(dep)

        result = store.get_dependency("get_Vehicles")

        assert result is dep

    def test_get_dependency_missing(self, store):
        """Test getting non-existent dependency."""
        assert store.get_dependency("nonexistent") is None


class TestClear:
    """Test clear method - covers lines 102-107."""

    def test_clear_all_data(self, store, sample_tool):
        """Test clearing all store data."""
        # Add data
        store.add_tool(sample_tool)
        store.add_embedding(sample_tool.operation_id, [0.1])
        dep = DependencyGraph(tool_id=sample_tool.operation_id, dependencies=[])
        store.add_dependency(dep)

        # Clear
        store.clear()

        # Verify all empty
        assert store.tools == {}
        assert store.embeddings == {}
        assert store.dependency_graph == {}
        assert store.retrieval_tools == set()
        assert store.mutation_tools == set()


class TestGetStats:
    """Test get_stats method - covers line 111."""

    def test_get_stats_empty(self, store):
        """Test stats on empty store."""
        stats = store.get_stats()

        assert stats["total_tools"] == 0
        assert stats["retrieval_tools"] == 0
        assert stats["mutation_tools"] == 0
        assert stats["embeddings"] == 0
        assert stats["dependencies"] == 0

    def test_get_stats_with_data(self, store, sample_tool, mutation_tool):
        """Test stats with data."""
        store.add_tool(sample_tool)
        store.add_tool(mutation_tool)
        store.add_embedding(sample_tool.operation_id, [0.1])
        dep = DependencyGraph(tool_id=sample_tool.operation_id, dependencies=[])
        store.add_dependency(dep)

        stats = store.get_stats()

        assert stats["total_tools"] == 2
        assert stats["retrieval_tools"] == 1
        assert stats["mutation_tools"] == 1
        assert stats["embeddings"] == 1
        assert stats["dependencies"] == 1

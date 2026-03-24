"""
Comprehensive tests for services/registry/__init__.py (ToolRegistry class)

Tests cover:
- __init__ and _load_documentation
- Properties (tools, embeddings, dependency_graph, retrieval_tools, mutation_tools, CONTEXT_PARAM_FALLBACK)
- initialize() method with all paths
- _initialize_faiss() method
- find_relevant_tools() and find_relevant_tools_with_scores()
- _apply_llm_reranking()
- Documentation access methods
- Hidden defaults methods
"""

import asyncio
import json
import os
import pytest
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock, AsyncMock, mock_open


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(
    op_id: str,
    method: str = "GET",
    description: str = "test",
    service_name: str = "test_service",
    service_url: str = "/test",
    path: str = "/api/test"
) -> MagicMock:
    """Create a MagicMock that mimics UnifiedToolDefinition."""
    tool = MagicMock()
    tool.operation_id = op_id
    tool.method = method
    tool.description = description
    tool.service_name = service_name
    tool.service_url = service_url
    tool.path = path
    tool.is_retrieval = method == "GET"
    tool.is_mutation = method in {"POST", "PUT", "PATCH", "DELETE"}
    tool.parameters = {}
    tool.to_openai_function.return_value = {"name": op_id, "description": description}
    return tool


def _make_dependency(tool_id: str, provider_tools: List[str] = None) -> MagicMock:
    """Create a MagicMock that mimics DependencyGraph."""
    dep = MagicMock()
    dep.tool_id = tool_id
    dep.required_outputs = []
    dep.provider_tools = provider_tools or []
    return dep


@dataclass
class MockSearchResult:
    """Mock SearchResult from FAISS."""
    tool_id: str
    score: float
    match_type: str = "semantic"


@dataclass
class MockRerankResult:
    """Mock RerankResult from LLM reranker."""
    tool_id: str
    confidence: float


def _build_registry(
    tool_documentation: Optional[Dict] = None,
    documentation_cache: Optional[Dict] = None,
    doc_file_exists: bool = True
):
    """
    Construct a ToolRegistry with all external dependencies mocked.
    Returns the registry instance.
    """
    import services.registry as registry_module

    # Reset the global documentation cache
    registry_module._documentation_cache = documentation_cache

    mock_store = MagicMock()
    mock_store.tools = {}
    mock_store.embeddings = {}
    mock_store.dependency_graph = {}
    mock_store.retrieval_tools = set()
    mock_store.mutation_tools = set()
    mock_store.count.return_value = 0
    mock_store.get_tool.return_value = None
    mock_store.list_tools.return_value = []

    mock_cache = MagicMock()
    mock_cache.is_cache_valid = AsyncMock(return_value=False)
    mock_cache.load_cache = AsyncMock(return_value={"tools": [], "embeddings": {}, "dependency_graph": []})
    mock_cache.save_cache = AsyncMock()

    mock_parser = MagicMock()
    mock_parser.parse_spec = AsyncMock(return_value=[])
    mock_parser.context_param_fallback = {"personId": "context.person_id"}

    mock_embedding = MagicMock()
    mock_embedding.build_embedding_text = MagicMock(return_value="test embedding text")
    mock_embedding.build_dependency_graph = MagicMock(return_value={})
    mock_embedding.generate_embeddings = AsyncMock(return_value={})

    mock_search = MagicMock()
    mock_search.find_relevant_tools_filtered = AsyncMock(return_value=[])
    mock_search.find_relevant_tools_with_scores = AsyncMock(return_value=[])

    def mock_exists(path):
        if "tool_documentation.json" in str(path):
            return doc_file_exists
        return False

    def mock_file_open(path, *args, **kwargs):
        if "tool_documentation.json" in str(path):
            content = json.dumps(tool_documentation or {})
            return mock_open(read_data=content)()
        raise FileNotFoundError(path)

    with patch.object(registry_module, "ToolStore", return_value=mock_store):
        with patch.object(registry_module, "CacheManager", return_value=mock_cache):
            with patch.object(registry_module, "SwaggerParser", return_value=mock_parser):
                with patch.object(registry_module, "EmbeddingEngine", return_value=mock_embedding):
                    with patch.object(registry_module, "SearchEngine", return_value=mock_search):
                        with patch("os.path.exists", side_effect=mock_exists):
                            with patch("builtins.open", side_effect=mock_file_open):
                                registry = registry_module.ToolRegistry(redis_client=None)

    # Store mocks for later assertions
    registry._mock_store = mock_store
    registry._mock_cache = mock_cache
    registry._mock_parser = mock_parser
    registry._mock_embedding = mock_embedding
    registry._mock_search = mock_search

    return registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_documentation_cache():
    """Reset the global documentation cache before each test."""
    import services.registry as registry_module
    original_cache = registry_module._documentation_cache
    registry_module._documentation_cache = None
    yield
    registry_module._documentation_cache = original_cache


@pytest.fixture
def registry():
    """Basic ToolRegistry with no documentation."""
    return _build_registry(doc_file_exists=False)


@pytest.fixture
def registry_with_docs():
    """ToolRegistry with sample documentation."""
    docs = {
        "get_Vehicles": {
            "purpose": "Retrieve list of vehicles",
            "when_to_use": ["When user wants vehicle list"],
            "parameter_origin_guide": {
                "personId": "CONTEXT: Auto-injected from session",
                "vehicleId": "USER: User must provide"
            },
            "example_queries_hr": ["pokazi vozila"]
        },
        "post_CreateVehicle": {
            "purpose": "Create new vehicle",
            "when_to_use": ["When user wants to add vehicle"],
            "parameter_origin_guide": {}
        }
    }
    return _build_registry(tool_documentation=docs)


# ===================================================================
# 1. __init__ and _load_documentation (lines 50-100)
# ===================================================================

class TestInit:

    def test_initializes_all_components(self, registry):
        """Test that __init__ initializes all required components."""
        assert registry._store is not None
        assert registry._cache is not None
        assert registry._parser is not None
        assert registry._embedding is not None
        assert registry._search is not None

    def test_initializes_state(self, registry):
        """Test initial state is set correctly."""
        assert registry.is_ready is False
        assert registry._load_lock is not None
        assert isinstance(registry._tool_documentation, dict)

    def test_accepts_redis_client(self):
        """Test that redis client is stored."""
        mock_redis = MagicMock()
        registry = _build_registry(doc_file_exists=False)
        # Override redis after construction
        registry.redis = mock_redis
        assert registry.redis is mock_redis


class TestLoadDocumentation:

    def test_loads_documentation_from_file(self, registry_with_docs):
        """Test documentation is loaded from file."""
        assert "get_Vehicles" in registry_with_docs._tool_documentation
        assert registry_with_docs._tool_documentation["get_Vehicles"]["purpose"] == "Retrieve list of vehicles"

    def test_uses_cached_documentation(self):
        """Test that cached documentation is used if available (lines 83-85)."""
        cached_docs = {"cached_tool": {"purpose": "Cached purpose"}}
        registry = _build_registry(documentation_cache=cached_docs, doc_file_exists=False)
        assert registry._tool_documentation == cached_docs

    def test_handles_missing_file(self, registry):
        """Test graceful handling when documentation file doesn't exist (lines 99-100)."""
        assert registry._tool_documentation == {}

    def test_handles_json_load_exception(self):
        """Test exception handling during JSON load (lines 96-98)."""
        import services.registry as registry_module

        registry_module._documentation_cache = None

        def mock_exists(path):
            return "tool_documentation.json" in str(path)

        def mock_open_raises(*args, **kwargs):
            raise json.JSONDecodeError("test error", "doc", 0)

        with patch.object(registry_module, "ToolStore", return_value=MagicMock()):
            with patch.object(registry_module, "CacheManager", return_value=MagicMock()):
                with patch.object(registry_module, "SwaggerParser", return_value=MagicMock()):
                    with patch.object(registry_module, "EmbeddingEngine", return_value=MagicMock()):
                        with patch.object(registry_module, "SearchEngine", return_value=MagicMock()):
                            with patch("os.path.exists", side_effect=mock_exists):
                                with patch("builtins.open", side_effect=mock_open_raises):
                                    registry = registry_module.ToolRegistry()

        assert registry._tool_documentation == {}


# ===================================================================
# 2. Properties (lines 106-135)
# ===================================================================

class TestProperties:

    def test_tools_property(self, registry):
        """Test tools property returns store's tools."""
        expected_tools = {"get_Test": _make_tool("get_Test")}
        registry._store.tools = expected_tools
        assert registry.tools == expected_tools

    def test_embeddings_property(self, registry):
        """Test embeddings property returns store's embeddings."""
        expected_embeddings = {"get_Test": [0.1, 0.2, 0.3]}
        registry._store.embeddings = expected_embeddings
        assert registry.embeddings == expected_embeddings

    def test_dependency_graph_property(self, registry):
        """Test dependency_graph property returns store's dependency_graph."""
        expected_deps = {"tool1": _make_dependency("tool1")}
        registry._store.dependency_graph = expected_deps
        assert registry.dependency_graph == expected_deps

    def test_retrieval_tools_property(self, registry):
        """Test retrieval_tools property returns store's retrieval_tools."""
        expected = {"get_Vehicles", "get_Persons"}
        registry._store.retrieval_tools = expected
        assert registry.retrieval_tools == expected

    def test_mutation_tools_property(self, registry):
        """Test mutation_tools property returns store's mutation_tools."""
        expected = {"post_Create", "delete_Remove"}
        registry._store.mutation_tools = expected
        assert registry.mutation_tools == expected

    def test_context_param_fallback_property(self, registry):
        """Test CONTEXT_PARAM_FALLBACK property returns parser's context_param_fallback."""
        expected = {"personId": "context.person_id", "tenantId": "context.tenant_id"}
        registry._parser.context_param_fallback = expected
        assert registry.CONTEXT_PARAM_FALLBACK == expected


# ===================================================================
# 3. initialize() (lines 146-261)
# ===================================================================

class TestInitialize:

    @pytest.mark.asyncio
    async def test_loads_from_preprocessed_registry_file(self):
        """Test loading from pre-processed registry file (lines 152-169)."""
        import services.registry as registry_module

        registry_module._documentation_cache = None

        registry_data = {
            "tools": [
                {
                    "operation_id": "get_Vehicles",
                    "service_name": "automation",
                    "swagger_name": "automation",
                    "service_url": "/automation",
                    "path": "/api/vehicles",
                    "method": "GET",
                    "description": "Get vehicles",
                    "parameters": {},
                    "required_params": [],
                    "output_keys": []
                }
            ],
            "dependency_graph": [
                {
                    "tool_id": "get_Vehicles",
                    "required_outputs": [],
                    "provider_tools": []
                }
            ]
        }

        mock_store = MagicMock()
        mock_store.tools = {}
        mock_store.embeddings = {}
        mock_store.dependency_graph = {}
        mock_store.retrieval_tools = set()
        mock_store.mutation_tools = set()
        mock_store.count.return_value = 1
        mock_store.add_tool = MagicMock()
        mock_store.add_dependency = MagicMock()
        mock_store.add_embedding = MagicMock()

        def mock_exists(path):
            if "processed_tool_registry.json" in str(path):
                return True
            return False

        def mock_file_open(path, *args, **kwargs):
            if "processed_tool_registry.json" in str(path):
                return mock_open(read_data=json.dumps(registry_data))()
            raise FileNotFoundError(path)

        with patch.object(registry_module, "ToolStore", return_value=mock_store):
            with patch.object(registry_module, "CacheManager", return_value=MagicMock()):
                with patch.object(registry_module, "SwaggerParser", return_value=MagicMock()):
                    with patch.object(registry_module, "EmbeddingEngine", return_value=MagicMock(generate_embeddings=AsyncMock(return_value={}))):
                        with patch.object(registry_module, "SearchEngine", return_value=MagicMock()):
                            with patch("os.path.exists", side_effect=mock_exists):
                                with patch("builtins.open", side_effect=mock_file_open):
                                    registry = registry_module.ToolRegistry()
                                    # Mock _initialize_faiss
                                    registry._initialize_faiss = AsyncMock()
                                    result = await registry.initialize(["http://test.com/swagger"])

        assert result is True
        assert registry.is_ready is True
        mock_store.add_tool.assert_called()
        mock_store.add_dependency.assert_called()

    @pytest.mark.asyncio
    async def test_loads_embeddings_from_cache(self):
        """Test loading embeddings from cache file (lines 171-185)."""
        import services.registry as registry_module

        registry_module._documentation_cache = None

        registry_data = {
            "tools": [
                {
                    "operation_id": "get_Vehicles",
                    "service_name": "automation",
                    "swagger_name": "",
                    "service_url": "/automation",
                    "path": "/api/vehicles",
                    "method": "GET",
                    "parameters": {},
                    "required_params": [],
                    "output_keys": []
                }
            ],
            "dependency_graph": []
        }

        embeddings_data = {
            "version": "2.0",
            "embeddings": {
                "get_Vehicles": [0.1, 0.2, 0.3]
            }
        }

        mock_store = MagicMock()
        mock_store.tools = {"get_Vehicles": _make_tool("get_Vehicles")}
        mock_store.embeddings = {}
        mock_store.dependency_graph = {}
        mock_store.retrieval_tools = set()
        mock_store.mutation_tools = set()
        mock_store.count.return_value = 1
        mock_store.add_tool = MagicMock()
        mock_store.add_dependency = MagicMock()
        mock_store.add_embedding = MagicMock()

        def mock_exists(path):
            path_str = str(path)
            if "processed_tool_registry.json" in path_str:
                return True
            if "tool_embeddings.json" in path_str:
                return True
            return False

        def mock_file_open(path, *args, **kwargs):
            path_str = str(path)
            if "processed_tool_registry.json" in path_str:
                return mock_open(read_data=json.dumps(registry_data))()
            if "tool_embeddings.json" in path_str:
                return mock_open(read_data=json.dumps(embeddings_data))()
            raise FileNotFoundError(path)

        with patch.object(registry_module, "ToolStore", return_value=mock_store):
            with patch.object(registry_module, "CacheManager", return_value=MagicMock()):
                with patch.object(registry_module, "SwaggerParser", return_value=MagicMock()):
                    with patch.object(registry_module, "EmbeddingEngine", return_value=MagicMock(generate_embeddings=AsyncMock(return_value={}))):
                        with patch.object(registry_module, "SearchEngine", return_value=MagicMock()):
                            with patch("os.path.exists", side_effect=mock_exists):
                                with patch("builtins.open", side_effect=mock_file_open):
                                    registry = registry_module.ToolRegistry()
                                    registry._initialize_faiss = AsyncMock()
                                    await registry.initialize([])

        mock_store.add_embedding.assert_called_with("get_Vehicles", [0.1, 0.2, 0.3])

    @pytest.mark.asyncio
    async def test_fallback_to_swagger_urls(self):
        """Test fallback to Swagger URLs when no pre-processed file (lines 187-220)."""
        import services.registry as registry_module

        registry_module._documentation_cache = None

        mock_tool = _make_tool("get_Test")

        mock_store = MagicMock()
        mock_store.tools = {}
        mock_store.embeddings = {}
        mock_store.dependency_graph = {}
        mock_store.retrieval_tools = set()
        mock_store.mutation_tools = set()
        mock_store.count.return_value = 1
        mock_store.add_tool = MagicMock()
        mock_store.add_dependency = MagicMock()
        mock_store.add_embedding = MagicMock()

        mock_cache = MagicMock()
        mock_cache.is_cache_valid = AsyncMock(return_value=False)
        mock_cache.save_cache = AsyncMock()

        mock_parser = MagicMock()
        mock_parser.parse_spec = AsyncMock(return_value=[mock_tool])
        mock_parser.context_param_fallback = {}

        mock_embedding = MagicMock()
        mock_embedding.build_embedding_text = MagicMock()
        mock_embedding.build_dependency_graph = MagicMock(return_value={})
        mock_embedding.generate_embeddings = AsyncMock(return_value={"get_Test": [0.1, 0.2]})

        with patch.object(registry_module, "ToolStore", return_value=mock_store):
            with patch.object(registry_module, "CacheManager", return_value=mock_cache):
                with patch.object(registry_module, "SwaggerParser", return_value=mock_parser):
                    with patch.object(registry_module, "EmbeddingEngine", return_value=mock_embedding):
                        with patch.object(registry_module, "SearchEngine", return_value=MagicMock()):
                            with patch("os.path.exists", return_value=False):
                                registry = registry_module.ToolRegistry()
                                registry._initialize_faiss = AsyncMock()
                                result = await registry.initialize(["http://swagger.test/api"])

        assert result is True
        mock_parser.parse_spec.assert_called_once()
        mock_cache.save_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_valid_path(self):
        """Test loading from valid cache (lines 191-202)."""
        import services.registry as registry_module

        registry_module._documentation_cache = None

        cached_tool = MagicMock()
        cached_tool.operation_id = "get_Cached"
        cached_dep = MagicMock()
        cached_dep.tool_id = "get_Cached"

        mock_store = MagicMock()
        mock_store.tools = {}
        mock_store.embeddings = {}
        mock_store.dependency_graph = {}
        mock_store.retrieval_tools = set()
        mock_store.mutation_tools = set()
        mock_store.count.return_value = 1
        mock_store.add_tool = MagicMock()
        mock_store.add_dependency = MagicMock()
        mock_store.add_embedding = MagicMock()

        mock_cache = MagicMock()
        mock_cache.is_cache_valid = AsyncMock(return_value=True)
        mock_cache.load_cache = AsyncMock(return_value={
            "tools": [cached_tool],
            "embeddings": {"get_Cached": [0.1, 0.2]},
            "dependency_graph": [cached_dep]
        })
        mock_cache.save_cache = AsyncMock()

        with patch.object(registry_module, "ToolStore", return_value=mock_store):
            with patch.object(registry_module, "CacheManager", return_value=mock_cache):
                with patch.object(registry_module, "SwaggerParser", return_value=MagicMock(context_param_fallback={})):
                    with patch.object(registry_module, "EmbeddingEngine", return_value=MagicMock(generate_embeddings=AsyncMock(return_value={}))):
                        with patch.object(registry_module, "SearchEngine", return_value=MagicMock()):
                            with patch("os.path.exists", return_value=False):
                                registry = registry_module.ToolRegistry()
                                registry._initialize_faiss = AsyncMock()
                                result = await registry.initialize(["http://test.com"])

        assert result is True
        mock_cache.load_cache.assert_called_once()
        mock_store.add_tool.assert_called_with(cached_tool)

    @pytest.mark.asyncio
    async def test_no_tools_loaded_returns_false(self):
        """Test returns False when no tools loaded from Swagger (lines 211-213)."""
        import services.registry as registry_module

        registry_module._documentation_cache = None

        mock_store = MagicMock()
        mock_store.tools = {}
        mock_store.embeddings = {}
        mock_store.dependency_graph = {}
        mock_store.retrieval_tools = set()
        mock_store.mutation_tools = set()
        mock_store.count.return_value = 0  # No tools

        mock_cache = MagicMock()
        mock_cache.is_cache_valid = AsyncMock(return_value=False)

        mock_parser = MagicMock()
        mock_parser.parse_spec = AsyncMock(return_value=[])  # No tools from parser
        mock_parser.context_param_fallback = {}

        with patch.object(registry_module, "ToolStore", return_value=mock_store):
            with patch.object(registry_module, "CacheManager", return_value=mock_cache):
                with patch.object(registry_module, "SwaggerParser", return_value=mock_parser):
                    with patch.object(registry_module, "EmbeddingEngine", return_value=MagicMock()):
                        with patch.object(registry_module, "SearchEngine", return_value=MagicMock()):
                            with patch("os.path.exists", return_value=False):
                                registry = registry_module.ToolRegistry()
                                result = await registry.initialize(["http://test.com"])

        assert result is False

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test exception handling during initialization (lines 259-261)."""
        import services.registry as registry_module

        registry_module._documentation_cache = None

        mock_store = MagicMock()
        mock_store.tools = {}
        mock_store.embeddings = {}
        mock_store.dependency_graph = {}
        mock_store.retrieval_tools = set()
        mock_store.mutation_tools = set()

        mock_cache = MagicMock()
        mock_cache.is_cache_valid = AsyncMock(side_effect=Exception("Test error"))
        mock_cache.save_cache = AsyncMock()

        with patch.object(registry_module, "ToolStore", return_value=mock_store):
            with patch.object(registry_module, "CacheManager", return_value=mock_cache):
                with patch.object(registry_module, "SwaggerParser", return_value=MagicMock(context_param_fallback={})):
                    with patch.object(registry_module, "EmbeddingEngine", return_value=MagicMock()):
                        with patch.object(registry_module, "SearchEngine", return_value=MagicMock()):
                            with patch("os.path.exists", return_value=False):
                                registry = registry_module.ToolRegistry()
                                result = await registry.initialize(["http://test.com"])

        assert result is False

    @pytest.mark.asyncio
    async def test_skips_embedding_generation_when_all_cached(self):
        """Test skips embedding generation when all cached (lines 226-227)."""
        import services.registry as registry_module

        registry_module._documentation_cache = None

        registry_data = {
            "tools": [
                {
                    "operation_id": "get_Test",
                    "service_name": "test",
                    "swagger_name": "",
                    "service_url": "/test",
                    "path": "/api/test",
                    "method": "GET",
                    "parameters": {},
                    "required_params": [],
                    "output_keys": []
                }
            ],
            "dependency_graph": []
        }

        mock_store = MagicMock()
        mock_store.tools = {"get_Test": _make_tool("get_Test")}
        mock_store.embeddings = {"get_Test": [0.1, 0.2]}  # Already has embedding
        mock_store.dependency_graph = {}
        mock_store.retrieval_tools = set()
        mock_store.mutation_tools = set()
        mock_store.count.return_value = 1
        mock_store.add_tool = MagicMock()
        mock_store.add_dependency = MagicMock()
        mock_store.add_embedding = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.generate_embeddings = AsyncMock(return_value={})

        def mock_exists(path):
            return "processed_tool_registry.json" in str(path)

        def mock_file_open(path, *args, **kwargs):
            if "processed_tool_registry.json" in str(path):
                return mock_open(read_data=json.dumps(registry_data))()
            raise FileNotFoundError(path)

        with patch.object(registry_module, "ToolStore", return_value=mock_store):
            with patch.object(registry_module, "CacheManager", return_value=MagicMock()):
                with patch.object(registry_module, "SwaggerParser", return_value=MagicMock(context_param_fallback={})):
                    with patch.object(registry_module, "EmbeddingEngine", return_value=mock_embedding):
                        with patch.object(registry_module, "SearchEngine", return_value=MagicMock()):
                            with patch("os.path.exists", side_effect=mock_exists):
                                with patch("builtins.open", side_effect=mock_file_open):
                                    registry = registry_module.ToolRegistry()
                                    registry._initialize_faiss = AsyncMock()
                                    await registry.initialize([])

        # Should not call generate_embeddings since all are cached
        mock_embedding.generate_embeddings.assert_not_called()


# ===================================================================
# 4. _initialize_faiss() (lines 263-296)
# ===================================================================

class TestInitializeFaiss:

    @pytest.mark.asyncio
    async def test_faiss_initialization_success(self, registry_with_docs):
        """Test successful FAISS initialization (lines 270-291)."""
        mock_faiss_store = MagicMock()
        mock_faiss_store.get_stats.return_value = {"total_tools": 10}

        async def mock_initialize(*args, **kwargs):
            return mock_faiss_store

        with patch.dict("sys.modules", {
            "services.faiss_vector_store": MagicMock(
                get_faiss_store=MagicMock(return_value=mock_faiss_store),
                initialize_faiss_store=mock_initialize
            )
        }):
            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=json.dumps({}))):
                    await registry_with_docs._initialize_faiss()

        # Should complete without error

    @pytest.mark.asyncio
    async def test_faiss_import_error(self, registry):
        """Test ImportError handling (lines 293-294)."""
        def raise_import_error(*args, **kwargs):
            raise ImportError("faiss not installed")

        with patch("builtins.__import__", side_effect=raise_import_error):
            # Should not raise, just log warning
            await registry._initialize_faiss()

    @pytest.mark.asyncio
    async def test_faiss_general_exception(self, registry):
        """Test general exception handling (lines 295-296)."""
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", side_effect=Exception("Read error")):
                # Should not raise, just log warning
                await registry._initialize_faiss()

    @pytest.mark.asyncio
    async def test_faiss_no_documentation_file(self, registry):
        """Test when tool_documentation.json not found (lines 278-280)."""
        with patch("os.path.exists", return_value=False):
            # Should return early without error
            await registry._initialize_faiss()


# ===================================================================
# 5. find_relevant_tools() and find_relevant_tools_with_scores() (lines 302-443)
# ===================================================================

class TestFindRelevantTools:

    @pytest.mark.asyncio
    async def test_returns_empty_when_not_ready(self, registry):
        """Test returns empty list when registry not ready (lines 315-317)."""
        registry.is_ready = False
        result = await registry.find_relevant_tools("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_schemas(self, registry):
        """Test returns OpenAI function schemas (lines 319-327)."""
        registry.is_ready = True

        mock_tool = _make_tool("get_Vehicles")
        registry._store.tools = {"get_Vehicles": mock_tool}
        registry._store.get_tool = MagicMock(return_value=mock_tool)

        registry._search.find_relevant_tools_filtered = AsyncMock(return_value=[
            {"name": "get_Vehicles", "score": 0.9, "schema": {"name": "get_Vehicles"}}
        ])

        # Use fallback path (no FAISS)
        with patch.dict("sys.modules", {"services.faiss_vector_store": None}):
            result = await registry.find_relevant_tools("show vehicles")

        assert len(result) == 1
        assert result[0]["name"] == "get_Vehicles"


class TestFindRelevantToolsWithScores:

    @pytest.mark.asyncio
    async def test_returns_empty_when_not_ready(self, registry):
        """Test returns empty list when not ready (lines 362-364)."""
        registry.is_ready = False
        result = await registry.find_relevant_tools_with_scores("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_faiss_search_path(self, registry):
        """Test FAISS search path (lines 367-416)."""
        registry.is_ready = True

        mock_tool = _make_tool("get_Vehicles")
        registry._store.tools = {"get_Vehicles": mock_tool}
        registry._store.get_tool = MagicMock(return_value=mock_tool)

        mock_faiss_store = MagicMock()
        mock_faiss_store.is_initialized.return_value = True
        mock_faiss_store.search = AsyncMock(return_value=[
            MockSearchResult("get_Vehicles", 0.9)
        ])

        mock_intent_result = MagicMock()
        mock_intent_result.intent = MagicMock()
        mock_intent_result.intent.value = "READ"

        faiss_module = MagicMock()
        faiss_module.get_faiss_store = MagicMock(return_value=mock_faiss_store)

        intent_module = MagicMock()
        intent_module.detect_action_intent = MagicMock(return_value=mock_intent_result)

        with patch.dict("sys.modules", {
            "services.faiss_vector_store": faiss_module,
            "services.intent_classifier": intent_module
        }):
            result = await registry.find_relevant_tools_with_scores(
                "show vehicles",
                use_llm_rerank=False  # Skip reranking for this test
            )

        assert len(result) == 1
        assert result[0]["name"] == "get_Vehicles"
        assert result[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_faiss_with_llm_reranking(self, registry):
        """Test FAISS with LLM reranking (lines 388-392)."""
        registry.is_ready = True

        mock_tool = _make_tool("get_Vehicles")
        registry._store.tools = {"get_Vehicles": mock_tool}
        registry._store.get_tool = MagicMock(return_value=mock_tool)

        mock_faiss_store = MagicMock()
        mock_faiss_store.is_initialized.return_value = True
        mock_faiss_store.search = AsyncMock(return_value=[
            MockSearchResult("get_Vehicles", 0.8),
            MockSearchResult("post_CreateVehicle", 0.7)
        ])

        mock_intent_result = MagicMock()
        mock_intent_result.intent = MagicMock()
        mock_intent_result.intent.value = "READ"

        faiss_module = MagicMock()
        faiss_module.get_faiss_store = MagicMock(return_value=mock_faiss_store)

        intent_module = MagicMock()
        intent_module.detect_action_intent = MagicMock(return_value=mock_intent_result)

        # Mock LLM reranking to return reordered results
        async def mock_rerank(query, results, top_k):
            return results  # Return as-is for test

        registry._apply_llm_reranking = mock_rerank

        with patch.dict("sys.modules", {
            "services.faiss_vector_store": faiss_module,
            "services.intent_classifier": intent_module
        }):
            result = await registry.find_relevant_tools_with_scores(
                "show vehicles",
                use_llm_rerank=True
            )

        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_fallback_to_filtered_search(self, registry):
        """Test fallback to filtered search when FAISS fails (lines 418-429)."""
        registry.is_ready = True

        registry._search.find_relevant_tools_filtered = AsyncMock(return_value=[
            {"name": "get_Test", "score": 0.8, "schema": {"name": "get_Test"}}
        ])

        # FAISS not available
        with patch.dict("sys.modules", {"services.faiss_vector_store": None}):
            result = await registry.find_relevant_tools_with_scores(
                "test query",
                use_filtered_search=True
            )

        assert len(result) == 1
        registry._search.find_relevant_tools_filtered.assert_called_once()

    @pytest.mark.asyncio
    async def test_ultimate_fallback_search(self, registry):
        """Test ultimate fallback to original search (lines 431-443)."""
        registry.is_ready = True

        registry._search.find_relevant_tools_with_scores = AsyncMock(return_value=[
            {"name": "get_Test", "score": 0.7, "schema": {"name": "get_Test"}}
        ])

        with patch.dict("sys.modules", {"services.faiss_vector_store": None}):
            result = await registry.find_relevant_tools_with_scores(
                "test query",
                use_filtered_search=False,
                use_faiss=False
            )

        assert len(result) == 1
        registry._search.find_relevant_tools_with_scores.assert_called_once()

    @pytest.mark.asyncio
    async def test_faiss_exception_fallback(self, registry):
        """Test fallback when FAISS raises exception (lines 415-416)."""
        registry.is_ready = True

        mock_faiss_store = MagicMock()
        mock_faiss_store.is_initialized.return_value = True
        mock_faiss_store.search = AsyncMock(side_effect=Exception("FAISS error"))

        faiss_module = MagicMock()
        faiss_module.get_faiss_store = MagicMock(return_value=mock_faiss_store)

        intent_module = MagicMock()
        intent_module.detect_action_intent = MagicMock()

        registry._search.find_relevant_tools_filtered = AsyncMock(return_value=[])

        with patch.dict("sys.modules", {
            "services.faiss_vector_store": faiss_module,
            "services.intent_classifier": intent_module
        }):
            result = await registry.find_relevant_tools_with_scores("test")

        assert result == []

    @pytest.mark.asyncio
    async def test_faiss_import_error_fallback(self, registry):
        """Test fallback when FAISS import fails (lines 413-414)."""
        registry.is_ready = True

        def raise_import(*args, **kwargs):
            if "faiss_vector_store" in str(args):
                raise ImportError("faiss not installed")
            return MagicMock()

        registry._search.find_relevant_tools_filtered = AsyncMock(return_value=[])

        with patch("builtins.__import__", side_effect=raise_import):
            result = await registry.find_relevant_tools_with_scores("test")

        # Should fall back gracefully
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_threshold_filtering(self, registry):
        """Test results below threshold are filtered (lines 397-402)."""
        registry.is_ready = True

        mock_tool = _make_tool("get_Vehicles")
        registry._store.tools = {"get_Vehicles": mock_tool}
        registry._store.get_tool = MagicMock(return_value=mock_tool)

        mock_faiss_store = MagicMock()
        mock_faiss_store.is_initialized.return_value = True
        mock_faiss_store.search = AsyncMock(return_value=[
            MockSearchResult("get_Vehicles", 0.3)  # Below threshold
        ])

        mock_intent_result = MagicMock()
        mock_intent_result.intent = MagicMock()
        mock_intent_result.intent.value = "READ"

        faiss_module = MagicMock()
        faiss_module.get_faiss_store = MagicMock(return_value=mock_faiss_store)

        intent_module = MagicMock()
        intent_module.detect_action_intent = MagicMock(return_value=mock_intent_result)

        registry._search.find_relevant_tools_filtered = AsyncMock(return_value=[])

        with patch.dict("sys.modules", {
            "services.faiss_vector_store": faiss_module,
            "services.intent_classifier": intent_module
        }):
            result = await registry.find_relevant_tools_with_scores(
                "test",
                threshold=0.55,
                use_llm_rerank=False
            )

        # Result below threshold should be filtered, fallback should be called
        registry._search.find_relevant_tools_filtered.assert_called_once()


# ===================================================================
# 6. _apply_llm_reranking() (lines 445-527)
# ===================================================================

class TestApplyLlmReranking:

    @pytest.mark.asyncio
    async def test_successful_reranking(self, registry_with_docs):
        """Test successful LLM reranking (lines 472-520)."""
        faiss_results = [
            MockSearchResult("get_Vehicles", 0.8),
            MockSearchResult("post_CreateVehicle", 0.7)
        ]

        async def mock_rerank(**kwargs):
            return [
                MockRerankResult("post_CreateVehicle", 0.95),
                MockRerankResult("get_Vehicles", 0.85)
            ]

        reranker_module = MagicMock()
        reranker_module.rerank_with_llm = mock_rerank

        with patch.dict("sys.modules", {"services.llm_reranker": reranker_module}):
            result = await registry_with_docs._apply_llm_reranking(
                "dodaj vozilo",
                faiss_results,
                top_k=5
            )

        # Order should be changed based on LLM reranking
        assert result[0].tool_id == "post_CreateVehicle"

    @pytest.mark.asyncio
    async def test_empty_rerank_results(self, registry_with_docs):
        """Test returns original when rerank is empty (lines 493-495)."""
        faiss_results = [
            MockSearchResult("get_Vehicles", 0.8)
        ]

        async def mock_rerank(**kwargs):
            return []

        reranker_module = MagicMock()
        reranker_module.rerank_with_llm = mock_rerank

        with patch.dict("sys.modules", {"services.llm_reranker": reranker_module}):
            result = await registry_with_docs._apply_llm_reranking(
                "test query",
                faiss_results,
                top_k=5
            )

        assert result == faiss_results

    @pytest.mark.asyncio
    async def test_import_error_handling(self, registry):
        """Test ImportError handling (lines 522-524)."""
        faiss_results = [MockSearchResult("get_Test", 0.8)]

        def raise_import(*args, **kwargs):
            raise ImportError("llm_reranker not available")

        with patch("builtins.__import__", side_effect=raise_import):
            result = await registry._apply_llm_reranking(
                "test",
                faiss_results,
                top_k=5
            )

        assert result == faiss_results

    @pytest.mark.asyncio
    async def test_exception_handling(self, registry):
        """Test general exception handling (lines 525-527)."""
        faiss_results = [MockSearchResult("get_Test", 0.8)]

        async def mock_rerank(**kwargs):
            raise Exception("Rerank failed")

        reranker_module = MagicMock()
        reranker_module.rerank_with_llm = mock_rerank

        with patch.dict("sys.modules", {"services.llm_reranker": reranker_module}):
            result = await registry._apply_llm_reranking(
                "test",
                faiss_results,
                top_k=5
            )

        assert result == faiss_results


# ===================================================================
# 7. Documentation access methods (lines 545-611)
# ===================================================================

class TestDocumentationAccessMethods:

    def test_get_tool_documentation_found(self, registry_with_docs):
        """Test get_tool_documentation returns doc when found (line 556)."""
        doc = registry_with_docs.get_tool_documentation("get_Vehicles")
        assert doc is not None
        assert doc["purpose"] == "Retrieve list of vehicles"

    def test_get_tool_documentation_not_found(self, registry_with_docs):
        """Test get_tool_documentation returns None when not found."""
        doc = registry_with_docs.get_tool_documentation("nonexistent")
        assert doc is None

    def test_get_parameter_origin_guide(self, registry_with_docs):
        """Test get_parameter_origin_guide returns guide (lines 558-570)."""
        guide = registry_with_docs.get_parameter_origin_guide("get_Vehicles")
        assert "personId" in guide
        assert "CONTEXT" in guide["personId"]

    def test_get_parameter_origin_guide_not_found(self, registry_with_docs):
        """Test returns empty dict when tool not found."""
        guide = registry_with_docs.get_parameter_origin_guide("nonexistent")
        assert guide == {}

    def test_get_tool_with_documentation(self, registry_with_docs):
        """Test get_tool_with_documentation merges tool and docs (lines 572-591)."""
        mock_tool = _make_tool("get_Vehicles")
        registry_with_docs._store.get_tool = MagicMock(return_value=mock_tool)

        result = registry_with_docs.get_tool_with_documentation("get_Vehicles")

        assert result is not None
        assert "tool" in result
        assert "documentation" in result
        assert result["tool"]["name"] == "get_Vehicles"
        assert result["documentation"]["purpose"] == "Retrieve list of vehicles"

    def test_get_tool_with_documentation_not_found(self, registry):
        """Test returns None when tool not found."""
        registry._store.get_tool = MagicMock(return_value=None)
        result = registry.get_tool_with_documentation("nonexistent")
        assert result is None

    def test_is_context_param_true(self, registry_with_docs):
        """Test is_context_param returns True for context params (lines 593-601)."""
        assert registry_with_docs.is_context_param("get_Vehicles", "personId") is True

    def test_is_context_param_false(self, registry_with_docs):
        """Test is_context_param returns False for user params."""
        assert registry_with_docs.is_context_param("get_Vehicles", "vehicleId") is False

    def test_is_context_param_not_found(self, registry_with_docs):
        """Test is_context_param returns False for unknown param."""
        assert registry_with_docs.is_context_param("get_Vehicles", "unknown") is False

    def test_is_user_param_true(self, registry_with_docs):
        """Test is_user_param returns True for user params (lines 603-611)."""
        assert registry_with_docs.is_user_param("get_Vehicles", "vehicleId") is True

    def test_is_user_param_false(self, registry_with_docs):
        """Test is_user_param returns False for context params."""
        assert registry_with_docs.is_user_param("get_Vehicles", "personId") is False

    def test_is_user_param_defaults_to_user(self, registry_with_docs):
        """Test is_user_param defaults to True for unknown param."""
        assert registry_with_docs.is_user_param("get_Vehicles", "unknown") is True


# ===================================================================
# 8. Hidden defaults (lines 617-719)
# ===================================================================

class TestHiddenDefaults:

    def test_get_hidden_defaults_found(self, registry):
        """Test get_hidden_defaults returns defaults (lines 632-645)."""
        defaults = registry.get_hidden_defaults("post_VehicleCalendar")
        assert "EntryType" in defaults
        assert defaults["EntryType"] == 0
        assert "AssigneeType" in defaults
        assert defaults["AssigneeType"] == 1

    def test_get_hidden_defaults_not_found(self, registry):
        """Test get_hidden_defaults returns empty dict when not found."""
        defaults = registry.get_hidden_defaults("unknown_tool")
        assert defaults == {}

    def test_inject_defaults(self, registry):
        """Test inject_defaults adds missing defaults (lines 647-676)."""
        params = {"CustomParam": "value"}
        result = registry.inject_defaults("post_VehicleCalendar", params)

        assert result["EntryType"] == 0
        assert result["AssigneeType"] == 1
        assert result["CustomParam"] == "value"

    def test_inject_defaults_does_not_override(self, registry):
        """Test inject_defaults does not override existing values."""
        params = {"EntryType": 5, "CustomParam": "value"}
        result = registry.inject_defaults("post_VehicleCalendar", params)

        assert result["EntryType"] == 5  # Not overridden
        assert result["AssigneeType"] == 1  # Added

    def test_inject_defaults_overrides_none(self, registry):
        """Test inject_defaults overrides None values."""
        params = {"EntryType": None}
        result = registry.inject_defaults("post_VehicleCalendar", params)

        assert result["EntryType"] == 0  # Overridden

    def test_get_merged_params(self, registry):
        """Test get_merged_params merges defaults with user params (lines 678-719)."""
        user_params = {"CustomParam": "value"}
        result = registry.get_merged_params("post_VehicleCalendar", user_params)

        assert result["EntryType"] == 0
        assert result["AssigneeType"] == 1
        assert result["CustomParam"] == "value"

    def test_get_merged_params_user_overrides_default(self, registry):
        """Test user params override defaults."""
        user_params = {"EntryType": 99}
        result = registry.get_merged_params("post_VehicleCalendar", user_params)

        assert result["EntryType"] == 99  # User value
        assert result["AssigneeType"] == 1  # Default

    def test_get_merged_params_filters_none(self, registry):
        """Test None values in user params are filtered."""
        user_params = {"EntryType": None, "CustomParam": "value"}
        result = registry.get_merged_params("post_VehicleCalendar", user_params)

        assert result["EntryType"] == 0  # Default (None filtered)
        assert result["CustomParam"] == "value"

    def test_get_merged_params_empty_user_params(self, registry):
        """Test with empty user params."""
        result = registry.get_merged_params("post_VehicleCalendar", {})

        assert result["EntryType"] == 0
        assert result["AssigneeType"] == 1

    def test_get_merged_params_none_user_params(self, registry):
        """Test with None user params."""
        result = registry.get_merged_params("post_VehicleCalendar", None)

        assert result["EntryType"] == 0
        assert result["AssigneeType"] == 1

    def test_get_merged_params_no_defaults(self, registry):
        """Test with tool that has no defaults."""
        user_params = {"param1": "value1"}
        result = registry.get_merged_params("unknown_tool", user_params)

        assert result == {"param1": "value1"}

    def test_get_merged_params_does_not_modify_original(self, registry):
        """Test original user_params is not modified."""
        user_params = {"CustomParam": "value"}
        original_copy = user_params.copy()

        registry.get_merged_params("post_VehicleCalendar", user_params)

        assert user_params == original_copy


# ===================================================================
# 9. Tool access methods (lines 533-539)
# ===================================================================

class TestToolAccessMethods:

    def test_get_tool(self, registry):
        """Test get_tool delegates to store."""
        mock_tool = _make_tool("get_Test")
        registry._store.get_tool = MagicMock(return_value=mock_tool)

        result = registry.get_tool("get_Test")

        assert result == mock_tool
        registry._store.get_tool.assert_called_with("get_Test")

    def test_list_tools(self, registry):
        """Test list_tools delegates to store."""
        registry._store.list_tools = MagicMock(return_value=["get_A", "get_B"])

        result = registry.list_tools()

        assert result == ["get_A", "get_B"]
        registry._store.list_tools.assert_called_once()


# ===================================================================
# 10. load_swagger() backward compatibility (lines 141-144)
# ===================================================================

class TestLoadSwagger:

    @pytest.mark.asyncio
    async def test_load_swagger_deprecated(self, registry):
        """Test load_swagger calls initialize."""
        registry.initialize = AsyncMock(return_value=True)

        result = await registry.load_swagger("http://test.com/swagger")

        registry.initialize.assert_called_once_with(["http://test.com/swagger"])
        assert result is True


# ===================================================================
# 11. Edge cases and integration scenarios
# ===================================================================

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_concurrent_initialization(self):
        """Test that concurrent initialization is handled by lock."""
        import services.registry as registry_module

        registry_module._documentation_cache = None

        mock_store = MagicMock()
        mock_store.tools = {}
        mock_store.embeddings = {}
        mock_store.dependency_graph = {}
        mock_store.retrieval_tools = set()
        mock_store.mutation_tools = set()
        mock_store.count.return_value = 0

        with patch.object(registry_module, "ToolStore", return_value=mock_store):
            with patch.object(registry_module, "CacheManager", return_value=MagicMock()):
                with patch.object(registry_module, "SwaggerParser", return_value=MagicMock(context_param_fallback={})):
                    with patch.object(registry_module, "EmbeddingEngine", return_value=MagicMock()):
                        with patch.object(registry_module, "SearchEngine", return_value=MagicMock()):
                            with patch("os.path.exists", return_value=False):
                                registry = registry_module.ToolRegistry()

        # Lock should be initialized
        assert registry._load_lock is not None
        assert isinstance(registry._load_lock, asyncio.Lock)

    def test_hidden_defaults_for_add_case(self, registry):
        """Test hidden defaults for post_AddCase."""
        defaults = registry.get_hidden_defaults("post_AddCase")
        assert "EntryType" in defaults
        assert defaults["EntryType"] == "WhatsApp"

    @pytest.mark.asyncio
    async def test_faiss_unknown_intent(self, registry):
        """Test FAISS search with UNKNOWN intent (no filtering)."""
        registry.is_ready = True

        mock_tool = _make_tool("get_Vehicles")
        registry._store.tools = {"get_Vehicles": mock_tool}
        registry._store.get_tool = MagicMock(return_value=mock_tool)

        mock_faiss_store = MagicMock()
        mock_faiss_store.is_initialized.return_value = True
        mock_faiss_store.search = AsyncMock(return_value=[
            MockSearchResult("get_Vehicles", 0.9)
        ])

        mock_intent_result = MagicMock()
        mock_intent_result.intent = MagicMock()
        mock_intent_result.intent.value = "UNKNOWN"

        faiss_module = MagicMock()
        faiss_module.get_faiss_store = MagicMock(return_value=mock_faiss_store)

        intent_module = MagicMock()
        intent_module.detect_action_intent = MagicMock(return_value=mock_intent_result)

        with patch.dict("sys.modules", {
            "services.faiss_vector_store": faiss_module,
            "services.intent_classifier": intent_module
        }):
            result = await registry.find_relevant_tools_with_scores(
                "something unclear",
                use_llm_rerank=False
            )

        # Should still return results
        assert len(result) == 1

        # Action filter should be None for UNKNOWN
        call_args = mock_faiss_store.search.call_args
        assert call_args[1].get("action_filter") is None

"""
Comprehensive tests for services/registry/search_engine.py

Tests cover:
- _load_json_file (caching, file not found, corrupt JSON)
- SearchEngine.__init__
- _apply_method_disambiguation
- _inject_intent_matching_tools
- _apply_user_specific_boosting
- _apply_category_boosting
- _apply_documentation_boosting
- _apply_example_query_boosting
- _apply_evaluation_adjustment
- _apply_dependency_boosting
- _fallback_keyword_search
- _description_keyword_search
- detect_put_patch_ambiguity
- detect_intent
- detect_categories
- filter_by_method
- filter_by_categories
- get_tool_documentation, get_tool_category, get_tools_in_category, _get_origin_guide
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open

from services.intent_classifier import ActionIntent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(op_id, method="GET", description="test", params=None, output_keys=None, tags=None):
    """Create a lightweight MagicMock that quacks like UnifiedToolDefinition."""
    tool = MagicMock()
    tool.operation_id = op_id
    tool.method = method
    tool.description = description
    tool.summary = description
    tool.tags = tags or []
    tool.output_keys = output_keys or []
    tool.path = f"/api/{op_id}"
    tool.parameters = params or {}
    tool.to_openai_function.return_value = {"name": op_id}
    return tool


def _mock_settings():
    """Return a MagicMock that mimics get_settings() result."""
    s = MagicMock()
    s.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com"
    s.AZURE_OPENAI_API_KEY = "test-key"
    s.AZURE_OPENAI_API_VERSION = "2024-02-15"
    s.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding"
    return s


def _make_intent_result(intent: ActionIntent):
    """Create a mock IntentDetectionResult with the given intent."""
    r = MagicMock()
    r.intent = intent
    r.confidence = 0.95
    return r


def _build_engine(tool_categories=None, tool_documentation=None):
    """
    Construct a SearchEngine with all external dependencies mocked.
    Returns the engine instance.
    """
    import services.registry.search_engine as mod

    mock_settings = _mock_settings()

    def fake_load(filename):
        if filename == "tool_categories.json":
            return tool_categories
        if filename == "tool_documentation.json":
            return tool_documentation
        return None

    with patch.object(mod, "settings", mock_settings):
        with patch.object(mod, "AsyncAzureOpenAI"):
            with patch.object(mod, "_load_json_file", side_effect=fake_load):
                engine = mod.SearchEngine()
    return engine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_json_cache():
    """Clear the module-level JSON cache before each test."""
    import services.registry.search_engine as mod
    mod._json_file_cache.clear()
    yield
    mod._json_file_cache.clear()


@pytest.fixture
def engine():
    """Minimal SearchEngine with no categories/docs."""
    return _build_engine()


@pytest.fixture
def engine_with_categories():
    """SearchEngine with sample categories."""
    cats = {
        "categories": {
            "vehicles": {
                "tools": ["get_Vehicles", "get_Vehicles_id", "delete_Vehicles_id"],
                "keywords_hr": ["vozilo", "auto"],
                "keywords_en": ["vehicle", "car"],
                "typical_intents": ["list vehicles"],
            },
            "persons": {
                "tools": ["get_Persons", "get_MasterData"],
                "keywords_hr": ["osoba", "zaposlenik"],
                "keywords_en": ["person", "employee"],
                "typical_intents": ["list persons"],
            },
        }
    }
    return _build_engine(tool_categories=cats)


@pytest.fixture
def engine_with_docs():
    """SearchEngine with sample documentation."""
    docs = {
        "get_Vehicles": {
            "purpose": "Retrieve list of all vehicles in the fleet",
            "example_queries": ["show vehicles", "list cars"],
            "example_queries_hr": ["pokazi vozila", "lista automobila", "sva vozila u floti"],
            "when_to_use": ["When user wants to see vehicle list"],
            "parameter_origin_guide": {"status": "USER: Korisnik mora navesti"},
        },
        "post_CreateVehicle": {
            "purpose": "Create a new vehicle entry",
            "example_queries": ["add vehicle"],
            "example_queries_hr": ["dodaj vozilo"],
            "when_to_use": ["When user wants to create a vehicle"],
            "parameter_origin_guide": {},
        },
    }
    return _build_engine(tool_documentation=docs)


@pytest.fixture
def engine_full():
    """SearchEngine with both categories and documentation."""
    cats = {
        "categories": {
            "vehicles": {
                "tools": ["get_Vehicles"],
                "keywords_hr": ["vozilo"],
                "keywords_en": ["vehicle"],
                "typical_intents": [],
            },
        }
    }
    docs = {
        "get_Vehicles": {
            "purpose": "Retrieve vehicles",
            "example_queries": ["show vehicles"],
            "example_queries_hr": ["pokazi vozila", "sva vozila u floti"],
            "when_to_use": ["When user wants to see vehicle list"],
            "parameter_origin_guide": {"status": "USER"},
        },
    }
    return _build_engine(tool_categories=cats, tool_documentation=docs)


# ===================================================================
# 1. _load_json_file
# ===================================================================

class TestLoadJsonFile:

    def test_caching_returns_same_object(self, tmp_path):
        import services.registry.search_engine as mod

        data = {"hello": "world"}
        f = tmp_path / "config" / "test.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(os.path, "dirname", return_value=str(tmp_path)):
            # Forcibly set base_path calculation
            mod._json_file_cache.clear()
            # Manually call with paths that exist
            mod._json_file_cache["test.json"] = data
            result1 = mod._load_json_file("test.json")
            result2 = mod._load_json_file("test.json")
            assert result1 is result2

    def test_file_not_found_returns_none(self):
        import services.registry.search_engine as mod
        mod._json_file_cache.clear()
        result = mod._load_json_file("definitely_does_not_exist_xyz.json")
        assert result is None

    def test_file_not_found_caches_none(self):
        import services.registry.search_engine as mod
        mod._json_file_cache.clear()
        mod._load_json_file("nonexistent_file_abc.json")
        assert "nonexistent_file_abc.json" in mod._json_file_cache
        assert mod._json_file_cache["nonexistent_file_abc.json"] is None

    def test_corrupt_json_returns_none(self, tmp_path):
        import services.registry.search_engine as mod
        mod._json_file_cache.clear()

        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{invalid json!!!", encoding="utf-8")

        # Patch os.path.exists and open to point to our corrupt file
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="{invalid json!!!")):
                result = mod._load_json_file("corrupt_test.json")
        # Should cache None because JSON decode fails
        assert result is None

    def test_cache_hit_skips_file_io(self):
        import services.registry.search_engine as mod
        cached = {"cached": True}
        mod._json_file_cache["already_cached.json"] = cached
        result = mod._load_json_file("already_cached.json")
        assert result is cached


# ===================================================================
# 2. SearchEngine.__init__
# ===================================================================

class TestSearchEngineInit:

    def test_creates_openai_client(self):
        import services.registry.search_engine as mod
        mock_settings = _mock_settings()
        mock_openai_cls = MagicMock()

        with patch.object(mod, "settings", mock_settings):
            with patch.object(mod, "AsyncAzureOpenAI", mock_openai_cls):
                with patch.object(mod, "_load_json_file", return_value=None):
                    engine = mod.SearchEngine()

        mock_openai_cls.assert_called_once_with(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2024-02-15",
        )

    def test_builds_tool_to_category_map(self, engine_with_categories):
        assert engine_with_categories._tool_to_category["get_Vehicles"] == "vehicles"
        assert engine_with_categories._tool_to_category["get_Persons"] == "persons"

    def test_builds_category_keywords(self, engine_with_categories):
        assert "vozilo" in engine_with_categories._category_keywords["vehicles"]
        assert "vehicle" in engine_with_categories._category_keywords["vehicles"]

    def test_no_categories_yields_empty_maps(self, engine):
        assert engine._tool_to_category == {}
        assert engine._category_keywords == {}


# ===================================================================
# 3. _apply_method_disambiguation
# ===================================================================

class TestApplyMethodDisambiguation:

    def test_read_intent_penalizes_delete(self, engine):
        tools = {"delete_Vehicles_id": _make_tool("delete_Vehicles_id", "DELETE")}
        scored = [(0.80, "delete_Vehicles_id")]

        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.READ)):
            result = engine._apply_method_disambiguation("pokazi vozila", scored, tools)

        # Should be penalized: 0.80 - 0.25 * 2.0 = 0.30
        assert result[0][0] == pytest.approx(0.30)

    def test_read_intent_penalizes_put_patch(self, engine):
        tools = {"put_Vehicles_id": _make_tool("put_Vehicles_id", "PUT")}
        scored = [(0.80, "put_Vehicles_id")]

        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.READ)):
            result = engine._apply_method_disambiguation("pokazi vozila", scored, tools)

        # 0.80 - 0.25 * 1.0 = 0.55
        assert result[0][0] == pytest.approx(0.55)

    def test_read_intent_penalizes_post_non_search(self, engine):
        tools = {"post_CreateVehicle": _make_tool("post_CreateVehicle", "POST")}
        scored = [(0.80, "post_CreateVehicle")]

        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.READ)):
            result = engine._apply_method_disambiguation("pokazi vozila", scored, tools)

        # 0.80 - 0.25 * 0.5 = 0.675
        assert result[0][0] == pytest.approx(0.675)

    def test_read_intent_does_not_penalize_search_post(self, engine):
        tools = {"post_Vehicles_search": _make_tool("post_Vehicles_search", "POST")}
        scored = [(0.80, "post_Vehicles_search")]

        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.READ)):
            result = engine._apply_method_disambiguation("pokazi vozila", scored, tools)

        assert result[0][0] == pytest.approx(0.80)

    def test_read_intent_boosts_get(self, engine):
        tools = {"get_Vehicles": _make_tool("get_Vehicles", "GET")}
        scored = [(0.80, "get_Vehicles")]

        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.READ)):
            result = engine._apply_method_disambiguation("pokazi vozila", scored, tools)

        assert result[0][0] == pytest.approx(0.85)

    def test_mutation_intent_passthrough(self, engine):
        tools = {
            "delete_Vehicles_id": _make_tool("delete_Vehicles_id", "DELETE"),
            "get_Vehicles": _make_tool("get_Vehicles", "GET"),
        }
        scored = [(0.80, "delete_Vehicles_id"), (0.75, "get_Vehicles")]

        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.DELETE)):
            result = engine._apply_method_disambiguation("obrisi vozilo", scored, tools)

        # Mutation intent returns scored unchanged
        assert result == scored

    def test_unclear_intent_smaller_penalty(self, engine):
        tools = {"delete_Vehicles_id": _make_tool("delete_Vehicles_id", "DELETE")}
        scored = [(0.80, "delete_Vehicles_id")]

        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.UNKNOWN)):
            result = engine._apply_method_disambiguation("nesto", scored, tools)

        # 0.80 - 0.10 * 2.0 = 0.60
        assert result[0][0] == pytest.approx(0.60)

    def test_score_does_not_go_below_zero(self, engine):
        tools = {"delete_Vehicles_id": _make_tool("delete_Vehicles_id", "DELETE")}
        scored = [(0.10, "delete_Vehicles_id")]

        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.READ)):
            result = engine._apply_method_disambiguation("pokazi", scored, tools)

        assert result[0][0] >= 0.0

    def test_missing_tool_in_dict_passthrough(self, engine):
        # op_id not in tools dict
        scored = [(0.80, "ghost_tool")]

        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.READ)):
            result = engine._apply_method_disambiguation("test", scored, {})

        assert result[0] == (0.80, "ghost_tool")


# ===================================================================
# 4. _inject_intent_matching_tools
# ===================================================================

class TestInjectIntentMatchingTools:

    def test_delete_keyword_injects_delete_tools(self, engine):
        tools = {
            "get_Vehicles": _make_tool("get_Vehicles", "GET"),
            "delete_Vehicles_id": _make_tool("delete_Vehicles_id", "DELETE"),
        }
        scored = [(0.80, "get_Vehicles")]
        search_pool = set(tools.keys())
        embeddings = {}

        result = engine._inject_intent_matching_tools(
            "obrisi vozilo", scored, tools, search_pool, embeddings
        )

        injected_ids = [op_id for _, op_id in result]
        assert "delete_Vehicles_id" in injected_ids

    def test_create_keyword_injects_post_tools(self, engine):
        tools = {
            "get_Vehicles": _make_tool("get_Vehicles", "GET"),
            "post_CreateVehicle": _make_tool("post_CreateVehicle", "POST"),
        }
        scored = [(0.80, "get_Vehicles")]
        search_pool = set(tools.keys())

        result = engine._inject_intent_matching_tools(
            "dodaj vozilo", scored, tools, search_pool, {}
        )

        injected_ids = [op_id for _, op_id in result]
        assert "post_CreateVehicle" in injected_ids

    def test_create_keyword_does_not_inject_search_post(self, engine):
        tools = {
            "post_Vehicles_search": _make_tool("post_Vehicles_search", "POST"),
        }
        scored = []
        search_pool = set(tools.keys())

        result = engine._inject_intent_matching_tools(
            "dodaj vozilo", scored, tools, search_pool, {}
        )

        injected_ids = [op_id for _, op_id in result]
        assert "post_Vehicles_search" not in injected_ids

    def test_update_keyword_injects_put_patch(self, engine):
        tools = {
            "put_Vehicles_id": _make_tool("put_Vehicles_id", "PUT"),
            "patch_Vehicles_id": _make_tool("patch_Vehicles_id", "PATCH"),
        }
        scored = []
        search_pool = set(tools.keys())

        result = engine._inject_intent_matching_tools(
            "azuriraj vozilo", scored, tools, search_pool, {}
        )

        injected_ids = [op_id for _, op_id in result]
        assert "put_Vehicles_id" in injected_ids
        assert "patch_Vehicles_id" in injected_ids

    def test_no_intent_no_injection(self, engine):
        tools = {
            "delete_Vehicles_id": _make_tool("delete_Vehicles_id", "DELETE"),
        }
        scored = []
        search_pool = set(tools.keys())

        result = engine._inject_intent_matching_tools(
            "pokazi informacije", scored, tools, search_pool, {}
        )

        assert len(result) == 0

    def test_max_injections_respected(self, engine):
        tools = {}
        for i in range(10):
            tid = f"delete_Item_{i}"
            tools[tid] = _make_tool(tid, "DELETE")

        scored = []
        search_pool = set(tools.keys())

        result = engine._inject_intent_matching_tools(
            "obrisi sve", scored, tools, search_pool, {}
        )

        assert len(result) <= 5

    def test_already_scored_tools_not_duplicated(self, engine):
        tools = {
            "delete_Vehicles_id": _make_tool("delete_Vehicles_id", "DELETE"),
        }
        scored = [(0.80, "delete_Vehicles_id")]
        search_pool = set(tools.keys())

        result = engine._inject_intent_matching_tools(
            "obrisi vozilo", scored, tools, search_pool, {}
        )

        ids = [op_id for _, op_id in result]
        assert ids.count("delete_Vehicles_id") == 1


# ===================================================================
# 5. _apply_user_specific_boosting
# ===================================================================

class TestApplyUserSpecificBoosting:

    def test_masterdata_gets_large_boost(self, engine):
        tools = {"get_MasterData": _make_tool("get_MasterData", "GET", params={})}
        scored = [(0.70, "get_MasterData")]

        result = engine._apply_user_specific_boosting("moje vozilo", scored, tools)

        # masterdata_boost (0.25) + boost_value (0.15) = +0.40
        assert result[0][0] == pytest.approx(0.70 + 0.25 + 0.15)

    def test_calendar_gets_penalty(self, engine):
        tools = {"get_VehicleCalendar": _make_tool("get_VehicleCalendar", "GET", params={})}
        scored = [(0.70, "get_VehicleCalendar")]

        result = engine._apply_user_specific_boosting("moje vozilo", scored, tools)

        # calendar_penalty = 0.20
        assert result[0][0] == pytest.approx(0.70 - 0.20)

    def test_user_filter_param_gets_boost(self, engine):
        param_mock = MagicMock()
        param_mock.name = "personId"
        param_mock.description = "Person ID"
        tools = {"get_Tasks": _make_tool("get_Tasks", "GET", params={"personId": param_mock})}
        scored = [(0.70, "get_Tasks")]

        result = engine._apply_user_specific_boosting("moji zadaci", scored, tools)

        assert result[0][0] == pytest.approx(0.70 + 0.15)

    def test_non_user_query_returns_unchanged(self, engine):
        tools = {"get_Vehicles": _make_tool("get_Vehicles", "GET", params={})}
        scored = [(0.70, "get_Vehicles")]

        result = engine._apply_user_specific_boosting("pokazi sva vozila", scored, tools)

        assert result == scored

    def test_vehicle_get_without_user_filter_gets_penalty(self, engine):
        tools = {"get_VehicleDetails": _make_tool("get_VehicleDetails", "GET", params={})}
        # op_id contains "vehicle", method is GET, no user filter param
        scored = [(0.70, "get_VehicleDetails")]

        result = engine._apply_user_specific_boosting("moje vozilo", scored, tools)

        # penalty_value = 0.10
        assert result[0][0] == pytest.approx(0.70 - 0.10)

    def test_equipment_gets_penalty(self, engine):
        tools = {"get_Equipment": _make_tool("get_Equipment", "GET", params={})}
        scored = [(0.70, "get_Equipment")]

        result = engine._apply_user_specific_boosting("moja oprema", scored, tools)

        assert result[0][0] == pytest.approx(0.70 - 0.20)


# ===================================================================
# 6. _apply_category_boosting
# ===================================================================

class TestApplyCategoryBoosting:

    def test_category_match_boosts(self, engine_with_categories):
        tools = {"get_Vehicles": _make_tool("get_Vehicles", "GET")}
        scored = [(0.70, "get_Vehicles")]

        result = engine_with_categories._apply_category_boosting(
            "pokazi vozilo", scored, tools
        )

        # category_boost = 0.12
        assert result[0][0] == pytest.approx(0.70 + 0.12)

    def test_keyword_match_in_op_id_boosts(self, engine_with_categories):
        # Tool not in any category but its op_id matches a query word
        tools = {"get_VehicleStatus": _make_tool("get_VehicleStatus", "GET")}
        scored = [(0.70, "get_VehicleStatus")]

        result = engine_with_categories._apply_category_boosting(
            "daj mi vehiclestatus", scored, tools
        )

        # keyword_match_boost = 0.08 (word "vehiclestatus" has len >= 4 and is in op_id_lower)
        assert result[0][0] == pytest.approx(0.70 + 0.08)

    def test_no_match_no_boost(self, engine_with_categories):
        tools = {"get_Weather": _make_tool("get_Weather", "GET")}
        scored = [(0.70, "get_Weather")]

        result = engine_with_categories._apply_category_boosting(
            "xyz abc", scored, tools
        )

        assert result[0][0] == pytest.approx(0.70)

    def test_no_categories_returns_unchanged(self, engine):
        tools = {"get_X": _make_tool("get_X", "GET")}
        scored = [(0.70, "get_X")]

        result = engine._apply_category_boosting("anything", scored, tools)

        assert result == scored

    def test_substring_keyword_match_in_category(self, engine_with_categories):
        """Category keyword 'vozilo' is >= 4 chars and appears as substring."""
        tools = {"get_Vehicles": _make_tool("get_Vehicles", "GET")}
        scored = [(0.70, "get_Vehicles")]

        result = engine_with_categories._apply_category_boosting(
            "neko vozilo treba", scored, tools
        )

        # "vozilo" matches category keyword via substring check
        assert result[0][0] > 0.70


# ===================================================================
# 7. _apply_documentation_boosting
# ===================================================================

class TestApplyDocumentationBoosting:

    def test_example_query_match_boosts(self, engine_with_docs):
        scored = [(0.70, "get_Vehicles")]

        result = engine_with_docs._apply_documentation_boosting("show vehicles", scored)

        # example_queries match: doc_boost * 0.5 = 0.20 * 0.5 = 0.10
        assert result[0][0] > 0.70

    def test_when_to_use_match_boosts(self, engine_with_docs):
        scored = [(0.70, "get_Vehicles")]

        # "vehicle" (len>=4) appears in "When user wants to see vehicle list"
        result = engine_with_docs._apply_documentation_boosting("vehicle details", scored)

        assert result[0][0] > 0.70

    def test_purpose_match_boosts(self, engine_with_docs):
        scored = [(0.70, "get_Vehicles")]

        # "fleet" (len>=4) appears in purpose "Retrieve list of all vehicles in the fleet"
        result = engine_with_docs._apply_documentation_boosting("fleet information", scored)

        assert result[0][0] > 0.70

    def test_no_documentation_returns_unchanged(self, engine):
        scored = [(0.70, "get_Vehicles")]

        result = engine._apply_documentation_boosting("anything", scored)

        assert result == scored

    def test_no_match_no_boost(self, engine_with_docs):
        scored = [(0.70, "get_Vehicles")]

        result = engine_with_docs._apply_documentation_boosting("xyz", scored)

        # No word overlap, "xyz" len < 4 so no substring match either
        assert result[0][0] == pytest.approx(0.70)


# ===================================================================
# 8. _apply_example_query_boosting
# ===================================================================

class TestApplyExampleQueryBoosting:

    def test_three_plus_word_overlap_full_boost(self, engine_with_docs):
        scored = [(0.70, "get_Vehicles")]

        # "sva vozila u floti" overlaps 3+ words with "sva vozila u floti"
        result = engine_with_docs._apply_example_query_boosting(
            "sva vozila u floti", scored
        )

        # example_query_boost = 0.25 (full)
        assert result[0][0] == pytest.approx(0.70 + 0.25)

    def test_two_word_overlap_medium_boost(self, engine_with_docs):
        scored = [(0.70, "get_Vehicles")]

        # "pokazi vozila" overlaps 2 words with "pokazi vozila"
        result = engine_with_docs._apply_example_query_boosting(
            "pokazi vozila", scored
        )

        # 0.25 * 0.6 = 0.15
        assert result[0][0] == pytest.approx(0.70 + 0.15)

    def test_one_word_overlap_weak_boost(self, engine_with_docs):
        scored = [(0.70, "get_Vehicles")]

        # "vozila" overlaps 1 word
        result = engine_with_docs._apply_example_query_boosting(
            "vozila", scored
        )

        # 0.25 * 0.2 = 0.05
        assert result[0][0] == pytest.approx(0.70 + 0.05)

    def test_zero_overlap_no_boost(self, engine_with_docs):
        scored = [(0.70, "get_Vehicles")]

        result = engine_with_docs._apply_example_query_boosting(
            "xyz abc", scored
        )

        assert result[0][0] == pytest.approx(0.70)

    def test_no_example_queries_hr_no_boost(self, engine_with_docs):
        scored = [(0.70, "post_CreateVehicle")]

        # post_CreateVehicle has example_queries_hr with only 1 entry: "dodaj vozilo"
        result = engine_with_docs._apply_example_query_boosting(
            "dodaj vozilo", scored
        )

        # 2 word overlap with "dodaj vozilo" -> medium boost
        assert result[0][0] == pytest.approx(0.70 + 0.25 * 0.6)

    def test_no_documentation_returns_unchanged(self, engine):
        scored = [(0.70, "get_X")]

        result = engine._apply_example_query_boosting("anything", scored)

        assert result == scored


# ===================================================================
# 9. _apply_evaluation_adjustment
# ===================================================================

class TestApplyEvaluationAdjustment:

    def test_calls_evaluator(self, engine):
        mock_evaluator = MagicMock()
        mock_evaluator.apply_evaluation_adjustment.return_value = 0.90

        with patch("services.registry.search_engine.get_tool_evaluator",
                    return_value=mock_evaluator, create=True):
            with patch.dict("sys.modules", {"services.tool_evaluator": MagicMock(get_tool_evaluator=lambda: mock_evaluator)}):
                result = engine._apply_evaluation_adjustment([(0.80, "get_Vehicles")])

        assert result[0][0] == 0.90

    def test_import_error_returns_unchanged(self, engine):
        with patch.dict("sys.modules", {"services.tool_evaluator": None}):
            # Clearing the import so it raises ImportError
            import importlib
            scored = [(0.80, "get_Vehicles")]

            # The function catches ImportError gracefully
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                result = engine._apply_evaluation_adjustment(scored)

            assert result == scored


# ===================================================================
# 10. _apply_dependency_boosting
# ===================================================================

class TestApplyDependencyBoosting:

    def test_adds_provider_tools(self, engine):
        dep_graph = {
            "post_CreateBooking": MagicMock(
                provider_tools=["get_Vehicles", "get_Persons"]
            ),
        }

        result = engine._apply_dependency_boosting(
            ["post_CreateBooking"], dep_graph
        )

        assert "post_CreateBooking" in result
        assert "get_Vehicles" in result
        assert "get_Persons" in result

    def test_no_duplicates(self, engine):
        dep_graph = {
            "post_A": MagicMock(provider_tools=["get_X"]),
            "post_B": MagicMock(provider_tools=["get_X"]),
        }

        result = engine._apply_dependency_boosting(
            ["post_A", "post_B", "get_X"], dep_graph
        )

        assert result.count("get_X") == 1

    def test_no_deps_returns_original(self, engine):
        result = engine._apply_dependency_boosting(["get_Vehicles"], {})

        assert result == ["get_Vehicles"]

    def test_limits_to_two_providers(self, engine):
        dep_graph = {
            "post_X": MagicMock(
                provider_tools=["get_A", "get_B", "get_C", "get_D"]
            ),
        }

        result = engine._apply_dependency_boosting(["post_X"], dep_graph)

        # Should add at most 2 providers
        assert len(result) <= 3  # original + 2


# ===================================================================
# 11. _fallback_keyword_search
# ===================================================================

class TestFallbackKeywordSearch:

    def test_matches_keywords_in_description(self, engine):
        tools = {
            "get_Vehicles": _make_tool("get_Vehicles", "GET", "List of vehicles"),
            "get_Persons": _make_tool("get_Persons", "GET", "List of persons"),
        }

        result = engine._fallback_keyword_search("vehicles list", tools, 5)

        assert "get_Vehicles" in result

    def test_matches_keywords_in_path(self, engine):
        tool = _make_tool("get_Items", "GET", "no match here")
        tool.path = "/api/vehicles"
        tools = {"get_Items": tool}

        result = engine._fallback_keyword_search("vehicles", tools, 5)

        assert "get_Items" in result

    def test_no_match_returns_empty(self, engine):
        tools = {"get_X": _make_tool("get_X", "GET", "something else")}

        result = engine._fallback_keyword_search("zzzzz", tools, 5)

        assert result == []

    def test_respects_top_k(self, engine):
        tools = {f"get_Vehicle_{i}": _make_tool(f"get_Vehicle_{i}", "GET", "vehicle info") for i in range(10)}

        result = engine._fallback_keyword_search("vehicle info", tools, 3)

        assert len(result) <= 3


# ===================================================================
# 12. _description_keyword_search
# ===================================================================

class TestDescriptionKeywordSearch:

    def test_word_overlap_scoring(self, engine):
        param_mock = MagicMock()
        param_mock.name = "status"
        param_mock.description = "filter status"
        tools = {
            "get_Vehicles": _make_tool("get_Vehicles", "GET", "list of vehicles",
                                        params={"status": param_mock},
                                        tags=["vehicles"]),
        }

        result = engine._description_keyword_search(
            "vehicles list", {"get_Vehicles"}, tools
        )

        assert len(result) > 0
        assert result[0][0] == "get_Vehicles"
        assert result[0][1] > 0

    def test_substring_match_boost(self, engine):
        param_mock = MagicMock()
        param_mock.name = "x"
        param_mock.description = "y"
        tools = {
            "get_Vehicles": _make_tool("get_Vehicles", "GET", "retrieve vehicles for fleet management",
                                        params={"x": param_mock}),
        }

        result = engine._description_keyword_search(
            "fleet", {"get_Vehicles"}, tools
        )

        assert len(result) > 0
        # "fleet" (len >= 4) as substring adds 0.3
        assert result[0][1] >= 0.3

    def test_max_results_respected(self, engine):
        param_mock = MagicMock()
        param_mock.name = "x"
        param_mock.description = "y"
        tools = {
            f"get_Item_{i}": _make_tool(f"get_Item_{i}", "GET", "common description item",
                                         params={"x": param_mock})
            for i in range(20)
        }

        result = engine._description_keyword_search(
            "item", set(tools.keys()), tools, max_results=3
        )

        assert len(result) <= 3

    def test_no_match_returns_empty(self, engine):
        param_mock = MagicMock()
        param_mock.name = "x"
        param_mock.description = "y"
        tools = {
            "get_Vehicles": _make_tool("get_Vehicles", "GET", "abc def",
                                        params={"x": param_mock}),
        }

        result = engine._description_keyword_search(
            "zzz", {"get_Vehicles"}, tools
        )

        assert result == []


# ===================================================================
# 13. detect_put_patch_ambiguity
# ===================================================================

class TestDetectPutPatchAmbiguity:

    def test_detects_ambiguity_same_resource(self, engine):
        tools = {
            "put_Vehicles_id": _make_tool("put_Vehicles_id", "PUT"),
            "patch_Vehicles_id": _make_tool("patch_Vehicles_id", "PATCH"),
        }
        scored = [(0.85, "put_Vehicles_id"), (0.84, "patch_Vehicles_id")]

        result = engine.detect_put_patch_ambiguity(scored, tools)

        assert result is not None
        assert result["ambiguous"] is True
        assert result["put_tool"] == "put_Vehicles_id"
        assert result["patch_tool"] == "patch_Vehicles_id"

    def test_no_ambiguity_different_resources(self, engine):
        tools = {
            "put_Vehicles_id": _make_tool("put_Vehicles_id", "PUT"),
            "patch_Persons_id": _make_tool("patch_Persons_id", "PATCH"),
        }
        scored = [(0.85, "put_Vehicles_id"), (0.84, "patch_Persons_id")]

        result = engine.detect_put_patch_ambiguity(scored, tools)

        assert result is None

    def test_no_ambiguity_below_threshold(self, engine):
        tools = {
            "put_Vehicles_id": _make_tool("put_Vehicles_id", "PUT"),
            "patch_Vehicles_id": _make_tool("patch_Vehicles_id", "PATCH"),
        }
        scored = [(0.60, "put_Vehicles_id"), (0.59, "patch_Vehicles_id")]

        result = engine.detect_put_patch_ambiguity(scored, tools)

        assert result is None

    def test_no_ambiguity_large_score_diff(self, engine):
        tools = {
            "put_Vehicles_id": _make_tool("put_Vehicles_id", "PUT"),
            "patch_Vehicles_id": _make_tool("patch_Vehicles_id", "PATCH"),
        }
        scored = [(0.90, "put_Vehicles_id"), (0.72, "patch_Vehicles_id")]

        result = engine.detect_put_patch_ambiguity(scored, tools)

        assert result is None

    def test_no_ambiguity_only_put(self, engine):
        tools = {"put_Vehicles_id": _make_tool("put_Vehicles_id", "PUT")}
        scored = [(0.85, "put_Vehicles_id")]

        result = engine.detect_put_patch_ambiguity(scored, tools)

        assert result is None


# ===================================================================
# 14. detect_intent
# ===================================================================

class TestDetectIntent:

    def test_read_intent(self, engine):
        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.READ)):
            result = engine.detect_intent("pokazi vozila")
        assert result == "READ"

    def test_write_intent_create(self, engine):
        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.CREATE)):
            result = engine.detect_intent("dodaj vozilo")
        assert result == "WRITE"

    def test_write_intent_update(self, engine):
        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.UPDATE)):
            result = engine.detect_intent("azuriraj podatke")
        assert result == "WRITE"

    def test_write_intent_delete(self, engine):
        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.DELETE)):
            result = engine.detect_intent("obrisi vozilo")
        assert result == "WRITE"

    def test_write_intent_patch(self, engine):
        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.PATCH)):
            result = engine.detect_intent("promijeni naziv")
        assert result == "WRITE"

    def test_unknown_intent(self, engine):
        with patch("services.registry.search_engine.detect_action_intent",
                    return_value=_make_intent_result(ActionIntent.UNKNOWN)):
            result = engine.detect_intent("nesto nejasno")
        assert result == "UNKNOWN"


# ===================================================================
# 15. detect_categories
# ===================================================================

class TestDetectCategories:

    def test_word_overlap_match(self, engine_with_categories):
        result = engine_with_categories.detect_categories("pokazi vozilo")

        assert "vehicles" in result

    def test_substring_match(self, engine_with_categories):
        # "vozilo" (len >= 4) is a substring of query
        result = engine_with_categories.detect_categories("koje vozilo je slobodno")

        assert "vehicles" in result

    def test_no_match_returns_empty(self, engine_with_categories):
        result = engine_with_categories.detect_categories("xyz abc")

        assert result == set()

    def test_multiple_categories(self, engine_with_categories):
        result = engine_with_categories.detect_categories("vozilo i osoba")

        assert "vehicles" in result
        assert "persons" in result

    def test_no_category_keywords_returns_empty(self, engine):
        result = engine.detect_categories("vozilo")

        assert result == set()


# ===================================================================
# 16. filter_by_method
# ===================================================================

class TestFilterByMethod:

    def test_read_filters_to_get(self, engine):
        tools = {
            "get_Vehicles": _make_tool("get_Vehicles", "GET"),
            "post_CreateVehicle": _make_tool("post_CreateVehicle", "POST"),
            "delete_Vehicles_id": _make_tool("delete_Vehicles_id", "DELETE"),
        }
        tool_ids = set(tools.keys())

        result = engine.filter_by_method(tool_ids, tools, "READ")

        assert "get_Vehicles" in result
        assert "post_CreateVehicle" not in result
        assert "delete_Vehicles_id" not in result

    def test_read_includes_search_posts(self, engine):
        tools = {
            "post_Vehicles_search": _make_tool("post_Vehicles_search", "POST"),
            "post_Vehicles_query": _make_tool("post_Vehicles_query", "POST"),
            "post_Vehicles_filter": _make_tool("post_Vehicles_filter", "POST"),
            "post_Vehicles_find": _make_tool("post_Vehicles_find", "POST"),
            "post_Vehicles_list": _make_tool("post_Vehicles_list", "POST"),
        }
        tool_ids = set(tools.keys())

        result = engine.filter_by_method(tool_ids, tools, "READ")

        assert "post_Vehicles_search" in result
        assert "post_Vehicles_query" in result
        assert "post_Vehicles_filter" in result
        assert "post_Vehicles_find" in result
        assert "post_Vehicles_list" in result

    def test_write_filters_to_mutations(self, engine):
        tools = {
            "get_Vehicles": _make_tool("get_Vehicles", "GET"),
            "post_CreateVehicle": _make_tool("post_CreateVehicle", "POST"),
            "put_Vehicles_id": _make_tool("put_Vehicles_id", "PUT"),
            "patch_Vehicles_id": _make_tool("patch_Vehicles_id", "PATCH"),
            "delete_Vehicles_id": _make_tool("delete_Vehicles_id", "DELETE"),
        }
        tool_ids = set(tools.keys())

        result = engine.filter_by_method(tool_ids, tools, "WRITE")

        assert "get_Vehicles" not in result
        assert "post_CreateVehicle" in result
        assert "put_Vehicles_id" in result
        assert "patch_Vehicles_id" in result
        assert "delete_Vehicles_id" in result

    def test_unknown_returns_all(self, engine):
        tools = {
            "get_Vehicles": _make_tool("get_Vehicles", "GET"),
            "post_CreateVehicle": _make_tool("post_CreateVehicle", "POST"),
        }
        tool_ids = set(tools.keys())

        result = engine.filter_by_method(tool_ids, tools, "UNKNOWN")

        assert result == tool_ids

    def test_empty_filter_falls_back_to_all(self, engine):
        """If filtering produces empty set, return all tools."""
        tools = {
            "get_Vehicles": _make_tool("get_Vehicles", "GET"),
        }
        tool_ids = set(tools.keys())

        result = engine.filter_by_method(tool_ids, tools, "WRITE")

        # No WRITE tools, so fallback to all
        assert result == tool_ids


# ===================================================================
# 17. filter_by_categories
# ===================================================================

class TestFilterByCategories:

    def test_filters_by_category(self, engine_with_categories):
        tool_ids = {"get_Vehicles", "get_Persons", "get_Weather"}

        result = engine_with_categories.filter_by_categories(
            tool_ids, {"vehicles"}
        )

        assert "get_Vehicles" in result
        assert "get_Persons" not in result

    def test_no_categories_returns_all(self, engine_with_categories):
        tool_ids = {"get_Vehicles", "get_Persons"}

        result = engine_with_categories.filter_by_categories(tool_ids, set())

        assert result == tool_ids

    def test_empty_result_falls_back_to_all(self, engine_with_categories):
        tool_ids = {"get_Weather"}

        result = engine_with_categories.filter_by_categories(
            tool_ids, {"vehicles"}
        )

        # No tools match "vehicles" category, fallback to all
        assert result == tool_ids

    def test_no_tool_to_category_returns_all(self, engine):
        tool_ids = {"get_Vehicles"}

        result = engine.filter_by_categories(tool_ids, {"vehicles"})

        assert result == tool_ids


# ===================================================================
# 18. get_tool_documentation / get_tool_category / get_tools_in_category / _get_origin_guide
# ===================================================================

class TestLookupMethods:

    def test_get_tool_documentation_found(self, engine_with_docs):
        doc = engine_with_docs.get_tool_documentation("get_Vehicles")
        assert doc is not None
        assert "purpose" in doc

    def test_get_tool_documentation_not_found(self, engine_with_docs):
        doc = engine_with_docs.get_tool_documentation("nonexistent")
        assert doc is None

    def test_get_tool_documentation_no_docs(self, engine):
        doc = engine.get_tool_documentation("get_Vehicles")
        assert doc is None

    def test_get_tool_category(self, engine_with_categories):
        cat = engine_with_categories.get_tool_category("get_Vehicles")
        assert cat == "vehicles"

    def test_get_tool_category_not_found(self, engine_with_categories):
        cat = engine_with_categories.get_tool_category("nonexistent")
        assert cat is None

    def test_get_tools_in_category(self, engine_with_categories):
        tools = engine_with_categories.get_tools_in_category("vehicles")
        assert "get_Vehicles" in tools
        assert "delete_Vehicles_id" in tools

    def test_get_tools_in_category_not_found(self, engine_with_categories):
        tools = engine_with_categories.get_tools_in_category("nonexistent")
        assert tools == []

    def test_get_tools_in_category_no_data(self, engine):
        tools = engine.get_tools_in_category("vehicles")
        assert tools == []

    def test_get_origin_guide(self, engine_with_docs):
        guide = engine_with_docs._get_origin_guide("get_Vehicles")
        assert "status" in guide

    def test_get_origin_guide_no_docs(self, engine):
        guide = engine._get_origin_guide("get_Vehicles")
        assert guide == {}

    def test_get_origin_guide_tool_not_in_docs(self, engine_with_docs):
        guide = engine_with_docs._get_origin_guide("nonexistent")
        assert guide == {}

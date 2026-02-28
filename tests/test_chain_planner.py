"""Tests for services/chain_planner.py – ChainPlanner."""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from services.chain_planner import (
    StepType,
    PlanStep,
    FallbackPath,
    ExecutionPlan,
    ChainPlanner,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_passthrough_cb():
    """Circuit breaker mock that passes calls through to the actual function."""
    cb = MagicMock()
    async def passthrough(endpoint_key, func, *args, **kwargs):
        return await func(*args, **kwargs)
    cb.call = AsyncMock(side_effect=passthrough)
    return cb


@pytest.fixture
def planner():
    mock_settings = MagicMock()
    mock_settings.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com"
    mock_settings.AZURE_OPENAI_API_KEY = "test-key"
    mock_settings.AZURE_OPENAI_API_VERSION = "2024-02-15"
    mock_settings.AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4"
    with patch("services.chain_planner.get_settings", return_value=mock_settings):
        with patch("services.chain_planner.settings", mock_settings):
            with patch("services.chain_planner.get_openai_client", return_value=MagicMock()):
                with patch("services.chain_planner.get_llm_circuit_breaker", return_value=_make_passthrough_cb()):
                    p = ChainPlanner()
    return p


def _tool_score(name="get_MasterData", score=0.9, schema=None):
    return {
        "name": name,
        "score": score,
        "schema": schema or {"description": "test", "parameters": {"properties": {}, "required": []}}
    }


# ===========================================================================
# DataClasses
# ===========================================================================

class TestDataClasses:
    def test_step_type_values(self):
        assert StepType.EXECUTE_TOOL.value == "execute_tool"
        assert StepType.ASK_USER.value == "ask_user"
        assert StepType.CONFIRM.value == "confirm"
        assert StepType.USER_SELECT.value == "user_select"
        assert StepType.EXTRACT_DATA.value == "extract_data"

    def test_plan_step_defaults(self):
        s = PlanStep(step_number=1, step_type=StepType.EXECUTE_TOOL)
        assert s.tool_name is None
        assert s.parameters == {}
        assert s.depends_on == []
        assert s.extract_fields == []

    def test_fallback_path_defaults(self):
        f = FallbackPath(trigger_error="403")
        assert f.steps == []
        assert f.reason == ""

    def test_execution_plan_defaults(self):
        p = ExecutionPlan(understanding="test", is_simple=True, has_all_data=True)
        assert p.missing_data == []
        assert p.primary_path == []
        assert p.fallback_paths == {}
        assert p.direct_response is None


# ===========================================================================
# _check_simple_cases
# ===========================================================================

class TestCheckSimpleCases:
    def test_greeting(self, planner):
        result = planner._check_simple_cases("bok", {}, [])
        assert result is not None
        assert result.is_simple is True
        assert "Pozdrav" in result.direct_response

    def test_greeting_variants(self, planner):
        for g in ["cao", "pozdrav", "hej", "zdravo", "hello", "hi"]:
            result = planner._check_simple_cases(g, {}, [])
            assert result is not None

    def test_thanks(self, planner):
        result = planner._check_simple_cases("hvala lijepa!", {}, [])
        assert result is not None
        assert "nema" in result.direct_response.lower()

    def test_help(self, planner):
        result = planner._check_simple_cases("pomoć", {}, [])
        assert result is not None
        assert "Kilometraža" in result.direct_response

    def test_high_score_tool(self, planner):
        tools = [_tool_score("get_MasterData", score=0.96)]
        result = planner._check_simple_cases("koliko km ima auto?", {}, tools)
        assert result is not None
        assert result.is_simple is True
        assert result.primary_path[0].tool_name == "get_MasterData"

    def test_low_score_no_match(self, planner):
        tools = [_tool_score("get_MasterData", score=0.6)]
        result = planner._check_simple_cases("nesto", {}, tools)
        assert result is None

    def test_regular_query_no_match(self, planner):
        result = planner._check_simple_cases("koliko km ima auto?", {}, [])
        assert result is None


# ===========================================================================
# _get_extraction_hint
# ===========================================================================

class TestGetExtractionHint:
    def test_kilometraza(self, planner):
        hint = planner._get_extraction_hint("koliko kilometraza?")
        assert "Mileage" in hint

    def test_registracija(self, planner):
        hint = planner._get_extraction_hint("kada istjece registracija?")
        assert "ExpirationDate" in hint

    def test_lizing(self, planner):
        hint = planner._get_extraction_hint("koji je lizing?")
        assert "Leasing" in hint

    def test_tablice(self, planner):
        hint = planner._get_extraction_hint("koje su tablice?")
        assert "LicencePlate" in hint

    def test_no_hint(self, planner):
        assert planner._get_extraction_hint("random question") is None


# ===========================================================================
# _summarize_context
# ===========================================================================

class TestSummarizeContext:
    def test_empty_context(self, planner):
        result = planner._summarize_context({})
        assert "Nema" in result

    def test_with_person_id(self, planner):
        pid = "00000000-0000-0000-0000-000000000123"
        result = planner._summarize_context({"person_id": pid})
        assert pid in result

    def test_with_vehicle(self, planner):
        result = planner._summarize_context({
            "person_id": "1",
            "vehicle": {"id": "42", "plate": "ZG-1234-AB", "name": "BMW 320"}
        })
        assert "42" in result or "BMW" in result or "ZG" in result


# ===========================================================================
# _summarize_tools
# ===========================================================================

class TestSummarizeTools:
    def test_basic(self, planner):
        tools = [_tool_score("get_MasterData", 0.9)]
        result = planner._summarize_tools(tools)
        assert "get_MasterData" in result
        assert "0.90" in result

    def test_empty(self, planner):
        assert planner._summarize_tools([]) == ""


# ===========================================================================
# _parse_plan_response
# ===========================================================================

class TestParsePlanResponse:
    def test_simple_plan(self, planner):
        response = {
            "understanding": "Dohvati km",
            "is_simple": True,
            "has_all_data": True,
            "primary_path": [
                {"step": 1, "type": "execute_tool", "tool": "get_MasterData",
                 "reason": "dohvati podatke", "extract_fields": ["Mileage"]}
            ],
            "fallback_paths": {}
        }
        plan = planner._parse_plan_response(response, [])
        assert plan.understanding == "Dohvati km"
        assert len(plan.primary_path) == 1
        assert plan.primary_path[0].tool_name == "get_MasterData"
        assert plan.primary_path[0].extract_fields == ["Mileage"]

    def test_multi_step_plan(self, planner):
        response = {
            "understanding": "Rezervacija",
            "is_simple": False,
            "has_all_data": False,
            "missing_data": ["FromTime"],
            "primary_path": [
                {"step": 1, "type": "ask_user", "question": "Kada?"},
                {"step": 2, "type": "execute_tool", "tool": "get_Vehicles", "depends_on": [1]}
            ],
            "fallback_paths": {
                "2": [{"trigger_error": "no_results",
                       "steps": [{"step": 1, "type": "ask_user", "question": "Drugi termin?"}],
                       "reason": "Nema vozila"}]
            }
        }
        plan = planner._parse_plan_response(response, [])
        assert len(plan.primary_path) == 2
        assert plan.primary_path[0].step_type == StepType.ASK_USER
        assert plan.primary_path[1].depends_on == [1]
        assert 2 in plan.fallback_paths
        assert len(plan.fallback_paths[2]) == 1

    def test_invalid_step_type_defaults(self, planner):
        response = {
            "understanding": "test",
            "is_simple": True,
            "has_all_data": True,
            "primary_path": [{"step": 1, "type": "INVALID_TYPE", "tool": "x"}],
            "fallback_paths": {}
        }
        plan = planner._parse_plan_response(response, [])
        assert plan.primary_path[0].step_type == StepType.EXECUTE_TOOL

    def test_invalid_fallback_step_num(self, planner):
        response = {
            "understanding": "test",
            "is_simple": True,
            "has_all_data": True,
            "primary_path": [],
            "fallback_paths": {"not_a_number": []}
        }
        plan = planner._parse_plan_response(response, [])
        assert plan.fallback_paths == {}

    def test_empty_response(self, planner):
        plan = planner._parse_plan_response({}, [])
        assert plan.understanding == ""
        assert plan.primary_path == []


# ===========================================================================
# _create_fallback_plan
# ===========================================================================

class TestCreateFallbackPlan:
    def test_no_tools(self, planner):
        plan = planner._create_fallback_plan("test", [])
        assert plan.direct_response is not None
        assert "pojasniti" in plan.direct_response.lower()

    def test_single_tool(self, planner):
        tools = [_tool_score("get_MasterData")]
        plan = planner._create_fallback_plan("km?", tools)
        assert len(plan.primary_path) == 1
        assert plan.primary_path[0].tool_name == "get_MasterData"
        assert plan.fallback_paths == {}

    def test_two_tools(self, planner):
        tools = [_tool_score("get_MasterData", 0.9), _tool_score("get_Vehicles", 0.7)]
        plan = planner._create_fallback_plan("km?", tools)
        assert len(plan.primary_path) == 1
        assert 1 in plan.fallback_paths
        assert plan.fallback_paths[1][0].steps[0].tool_name == "get_Vehicles"


# ===========================================================================
# _has_required_context
# ===========================================================================

class TestHasRequiredContext:
    def test_no_required_params(self, planner):
        tool = _tool_score()
        assert planner._has_required_context(tool, {}) is True

    def test_vehicle_required_missing(self, planner):
        tool = _tool_score(schema={"parameters": {"required": ["VehicleId"], "properties": {}}})
        assert planner._has_required_context(tool, {}) is False

    def test_vehicle_required_present(self, planner):
        tool = _tool_score(schema={"parameters": {"required": ["VehicleId"], "properties": {}}})
        assert planner._has_required_context(tool, {"vehicle": {"id": "42", "plate": "ZG-123"}}) is True

    def test_person_required_missing(self, planner):
        tool = _tool_score(schema={"parameters": {"required": ["PersonId"], "properties": {}}})
        assert planner._has_required_context(tool, {}) is False

    def test_person_required_present(self, planner):
        tool = _tool_score(schema={"parameters": {"required": ["PersonId"], "properties": {}}})
        assert planner._has_required_context(tool, {"person_id": "00000000-0000-0000-0000-000000000123"}) is True


# ===========================================================================
# create_plan (integration with mocked LLM)
# ===========================================================================

class TestCreatePlan:
    @pytest.mark.asyncio
    async def test_simple_greeting(self, planner):
        plan = await planner.create_plan("bok", {}, [], [])
        assert plan.is_simple is True
        assert plan.direct_response is not None

    @pytest.mark.asyncio
    async def test_high_score_tool(self, planner):
        tools = [_tool_score("get_MasterData", 0.96)]
        plan = await planner.create_plan("km auto?", {}, [], tools)
        assert plan.primary_path[0].tool_name == "get_MasterData"

    @pytest.mark.asyncio
    async def test_llm_returns_plan(self, planner):
        llm_response = {
            "understanding": "Dohvati km",
            "is_simple": True,
            "has_all_data": True,
            "primary_path": [{"step": 1, "type": "execute_tool", "tool": "get_MD"}],
            "fallback_paths": {}
        }
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(llm_response)
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        planner.openai.chat.completions.create = AsyncMock(return_value=mock_resp)

        tools = [_tool_score("get_MD", 0.7)]
        plan = await planner.create_plan("koliko km?", {"person_id": 1}, [], tools)
        assert plan.understanding == "Dohvati km"
        assert len(plan.primary_path) == 1

    @pytest.mark.asyncio
    async def test_llm_failure_creates_fallback(self, planner):
        planner.openai.chat.completions.create = AsyncMock(side_effect=Exception("LLM down"))
        tools = [_tool_score("get_MD", 0.7)]
        plan = await planner.create_plan("nesto", {}, [], tools)
        assert plan.primary_path[0].tool_name == "get_MD"

    @pytest.mark.asyncio
    async def test_llm_returns_none(self, planner):
        planner.openai.chat.completions.create = AsyncMock(side_effect=Exception("err"))
        plan = await planner.create_plan("test", {}, [], [])
        assert plan.direct_response is not None

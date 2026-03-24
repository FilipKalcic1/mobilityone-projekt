"""
Tests for services/dependency_resolver.py

Comprehensive tests covering:
- ResolutionResult and EntityReference dataclasses
- DependencyResolver initialization
- detect_value_type (licence plates, VIN, email, phone, empty/None)
- find_provider_tool (DependencyGraph, output_keys, name patterns)
- build_filter_query
- resolve_dependency (cache, provider not found, success/failure, ID extraction)
- _extract_id_from_result (dict, array, wrapped, nested, case-insensitive)
- clear_cache
- detect_entity_reference (ordinal, possessive, name, no match)
- resolve_entity_reference (possessive with/without vehicle, ordinal, name)
- _resolve_by_ordinal (no provider, success, index out of bounds)
- _resolve_by_name (fuzzy matching, ambiguity, single match, no match)
- _extract_vehicle_list (list, wrapped, single, empty)
- _fuzzy_match_vehicles (exact, partial, word-level, no match, empty)
- _fuzzy_match_vehicle (legacy wrapper)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.dependency_resolver import (
    DependencyResolver,
    ResolutionResult,
    EntityReference,
)


# ============================================================================
# HELPERS
# ============================================================================

def _make_registry(tools=None):
    """Create a mock registry with optional tools dict."""
    reg = MagicMock()
    reg.tools = tools or {}
    reg.get_tool = lambda tid: (tools or {}).get(tid)
    return reg


def _make_tool(op_id, method="GET", output_keys=None, params=None):
    """Create a mock tool with operation_id, method, output_keys, parameters."""
    tool = MagicMock()
    tool.operation_id = op_id
    tool.method = method
    tool.output_keys = output_keys or []
    tool.parameters = params or {}
    return tool


def _make_param_def(context_key=None):
    """Create a mock ParameterDefinition with optional context_key."""
    p = MagicMock()
    p.context_key = context_key
    return p


VALID_UUID = "00000000-0000-0000-0000-000000000123"
VALID_VEHICLE_UUID = "11111111-1111-1111-1111-111111111111"


def _user_ctx_with_vehicle(vehicle_id=VALID_VEHICLE_UUID, plate="ZG-123-AB", name="Golf"):
    """Build user_context dict with person_id and vehicle."""
    return {
        "person_id": VALID_UUID,
        "vehicle": {
            "Id": vehicle_id,
            "LicencePlate": plate,
            "FullVehicleName": name,
        },
    }


def _user_ctx_no_vehicle():
    """Build user_context with person_id but no vehicle."""
    return {"person_id": VALID_UUID}


def _user_ctx_empty():
    """Build empty user_context."""
    return {}


# ============================================================================
# 1. ResolutionResult dataclass
# ============================================================================

class TestResolutionResult:

    def test_defaults(self):
        r = ResolutionResult(success=True)
        assert r.success is True
        assert r.resolved_value is None
        assert r.provider_tool is None
        assert r.provider_params is None
        assert r.error_message is None
        assert r.feedback is None
        assert r.needs_user_selection is False

    def test_all_fields(self):
        r = ResolutionResult(
            success=False,
            resolved_value="uuid-x",
            provider_tool="get_Vehicles",
            provider_params={"Filter": "x"},
            error_message="err",
            feedback={"key": "val"},
            needs_user_selection=True,
        )
        assert r.success is False
        assert r.resolved_value == "uuid-x"
        assert r.provider_tool == "get_Vehicles"
        assert r.provider_params == {"Filter": "x"}
        assert r.error_message == "err"
        assert r.feedback == {"key": "val"}
        assert r.needs_user_selection is True


# ============================================================================
# 2. EntityReference dataclass
# ============================================================================

class TestEntityReference:

    def test_defaults(self):
        ref = EntityReference(entity_type="vehicle", reference_type="ordinal", value="Vozilo 1")
        assert ref.entity_type == "vehicle"
        assert ref.reference_type == "ordinal"
        assert ref.value == "Vozilo 1"
        assert ref.ordinal_index is None
        assert ref.is_possessive is False

    def test_all_fields(self):
        ref = EntityReference(
            entity_type="person",
            reference_type="possessive",
            value="moje vozilo",
            ordinal_index=2,
            is_possessive=True,
        )
        assert ref.entity_type == "person"
        assert ref.ordinal_index == 2
        assert ref.is_possessive is True


# ============================================================================
# 3. DependencyResolver.__init__
# ============================================================================

class TestInit:

    def test_registry_stored_and_cache_empty(self):
        reg = _make_registry()
        dr = DependencyResolver(reg)
        assert dr.registry is reg
        assert dr._resolution_cache == {}


# ============================================================================
# 4. detect_value_type
# ============================================================================

class TestDetectValueType:

    def setup_method(self):
        self.dr = DependencyResolver(_make_registry())

    def test_licence_plate_croatian(self):
        result = self.dr.detect_value_type("ZG-1234-AB")
        assert result is not None
        assert result[0] == "vehicleid"
        assert result[1] == "LicencePlate"

    def test_licence_plate_no_dashes(self):
        result = self.dr.detect_value_type("ZG1234AB")
        assert result is not None
        assert result[0] == "vehicleid"

    def test_licence_plate_spaces(self):
        result = self.dr.detect_value_type("ZG 1234 AB")
        assert result is not None
        assert result[0] == "vehicleid"

    def test_empty_string(self):
        assert self.dr.detect_value_type("") is None

    def test_none_input(self):
        assert self.dr.detect_value_type(None) is None

    def test_non_string_input(self):
        assert self.dr.detect_value_type(12345) is None

    def test_non_matching_value(self):
        assert self.dr.detect_value_type("hello world") is None

    def test_short_string(self):
        assert self.dr.detect_value_type("AB") is None

    def test_email_detected(self):
        result = self.dr.detect_value_type("user@example.com")
        assert result is not None
        assert result[0] == "personid"
        assert result[1] == "Email"


# ============================================================================
# 5. find_provider_tool
# ============================================================================

class TestFindProviderTool:

    def test_no_provider_config(self):
        dr = DependencyResolver(_make_registry())
        assert dr.find_provider_tool("UnknownParam") is None

    def test_strategy1_dependency_graph(self):
        """DependencyGraph lookup should be tried first."""
        dep_graph_entry = MagicMock()
        dep_graph_entry.provider_tools = {"VehicleId": "get_VehicleLookup"}

        reg = _make_registry()
        reg.dependency_graph = {"get_VehicleLookup": dep_graph_entry}

        dr = DependencyResolver(reg)
        result = dr.find_provider_tool("VehicleId")
        assert result == "get_VehicleLookup"

    def test_strategy2_output_keys(self):
        """Search tools whose output_keys match the expected keys."""
        tool = _make_tool("get_MasterData", method="GET", output_keys=["Id", "VehicleId"])
        reg = _make_registry(tools={"get_MasterData": tool})
        # no dependency_graph attr
        dr = DependencyResolver(reg)
        result = dr.find_provider_tool("VehicleId")
        assert result == "get_MasterData"

    def test_strategy3_name_pattern(self):
        """Fall back to name matching when output_keys do not match."""
        tool = _make_tool("get_Vehicles", method="GET", output_keys=[])
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)
        result = dr.find_provider_tool("VehicleId")
        assert result == "get_Vehicles"

    def test_strategy3_skips_delete_tools(self):
        """Name pattern match should skip delete/remove tools."""
        tool_del = _make_tool("delete_Vehicle", method="GET", output_keys=[])
        reg = _make_registry(tools={"delete_Vehicle": tool_del})
        dr = DependencyResolver(reg)
        assert dr.find_provider_tool("VehicleId") is None

    def test_strategy3_skips_wrong_method(self):
        """Name pattern match should prefer GET tools."""
        tool_post = _make_tool("post_Vehicles", method="POST", output_keys=[])
        reg = _make_registry(tools={"post_Vehicles": tool_post})
        dr = DependencyResolver(reg)
        assert dr.find_provider_tool("VehicleId") is None

    def test_person_id_lookup(self):
        tool = _make_tool("get_Persons", method="GET", output_keys=["PersonId"])
        reg = _make_registry(tools={"get_Persons": tool})
        dr = DependencyResolver(reg)
        assert dr.find_provider_tool("PersonId") == "get_Persons"


# ============================================================================
# 6. build_filter_query
# ============================================================================

class TestBuildFilterQuery:

    def test_basic(self):
        dr = DependencyResolver(_make_registry())
        result = dr.build_filter_query("LicencePlate", "ZG-1234-AB")
        assert result == {"Filter": "LicencePlate(=)ZG-1234-AB"}

    def test_strips_whitespace(self):
        dr = DependencyResolver(_make_registry())
        result = dr.build_filter_query("Name", "  Golf  ")
        assert result == {"Filter": "Name(=)Golf"}


# ============================================================================
# 7. resolve_dependency
# ============================================================================

class TestResolveDependency:

    @pytest.fixture
    def setup_dr(self):
        """Build a resolver with a GET vehicle tool that has a person_id param."""
        person_param = _make_param_def(context_key="person_id")
        filter_param = _make_param_def(context_key=None)
        tool = _make_tool(
            "get_Vehicles",
            method="GET",
            output_keys=["Id", "VehicleId"],
            params={"PersonId": person_param, "Filter": filter_param},
        )
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)
        return dr, tool

    @pytest.mark.asyncio
    async def test_cache_hit(self, setup_dr):
        dr, _ = setup_dr
        dr._resolution_cache["VehicleId:ZG-1234-AB"] = {
            "value": "cached-uuid",
            "tool": "get_Vehicles",
        }
        result = await dr.resolve_dependency(
            "VehicleId", "ZG-1234-AB", _user_ctx_no_vehicle(), AsyncMock()
        )
        assert result.success is True
        assert result.resolved_value == "cached-uuid"

    @pytest.mark.asyncio
    async def test_provider_not_found(self):
        dr = DependencyResolver(_make_registry())
        result = await dr.resolve_dependency(
            "VehicleId", "ZG-1234-AB", _user_ctx_no_vehicle(), AsyncMock()
        )
        assert result.success is False
        assert "pronaći" in result.error_message

    @pytest.mark.asyncio
    async def test_provider_tool_missing_from_registry(self):
        """find_provider_tool returns a tool_id, but get_tool returns None."""
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id"])
        reg = MagicMock()
        reg.tools = {"get_Vehicles": tool}
        reg.get_tool = MagicMock(return_value=None)

        dr = DependencyResolver(reg)
        result = await dr.resolve_dependency(
            "VehicleId", "ZG-1234-AB", _user_ctx_no_vehicle(), AsyncMock()
        )
        assert result.success is False
        assert "nije dostupan" in result.error_message

    @pytest.mark.asyncio
    async def test_provider_call_success(self, setup_dr):
        dr, tool = setup_dr

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = [{"Id": "uuid-resolved", "Name": "Golf"}]
        executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr.resolve_dependency(
                "VehicleId", "ZG-1234-AB", _user_ctx_no_vehicle(), executor
            )
        assert result.success is True
        assert result.resolved_value == "uuid-resolved"
        assert result.provider_tool == "get_Vehicles"
        # should be cached
        assert "VehicleId:ZG-1234-AB" in dr._resolution_cache

    @pytest.mark.asyncio
    async def test_provider_call_failure(self, setup_dr):
        dr, tool = setup_dr

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = False
        exec_result.error_message = "API error"
        executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr.resolve_dependency(
                "VehicleId", "ZG-1234-AB", _user_ctx_no_vehicle(), executor
            )
        assert result.success is False
        assert result.error_message == "API error"

    @pytest.mark.asyncio
    async def test_no_id_in_result(self, setup_dr):
        dr, tool = setup_dr

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = [{"SomeOtherField": "abc"}]
        executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr.resolve_dependency(
                "VehicleId", None, _user_ctx_no_vehicle(), executor
            )
        assert result.success is False
        assert "nije vratio" in result.error_message

    @pytest.mark.asyncio
    async def test_executor_raises_exception(self, setup_dr):
        dr, tool = setup_dr

        executor = AsyncMock()
        executor.execute = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr.resolve_dependency(
                "VehicleId", "ZG-1234-AB", _user_ctx_no_vehicle(), executor
            )
        assert result.success is False
        assert "boom" in result.error_message

    @pytest.mark.asyncio
    async def test_generic_filter_when_value_type_unknown(self, setup_dr):
        """If user_value doesn't match any pattern, use generic Name(~) filter."""
        dr, tool = setup_dr

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = [{"Id": "found-uuid"}]
        executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr.resolve_dependency(
                "VehicleId", "Golf", _user_ctx_no_vehicle(), executor
            )
        assert result.success is True
        # Verify the executor was called (we trust the params construction)
        executor.execute.assert_called_once()


# ============================================================================
# 8. _extract_id_from_result
# ============================================================================

class TestExtractIdFromResult:

    def setup_method(self):
        self.dr = DependencyResolver(_make_registry())

    def test_none_data(self):
        assert self.dr._extract_id_from_result(None, "VehicleId") is None

    def test_empty_dict(self):
        assert self.dr._extract_id_from_result({}, "VehicleId") is None

    def test_direct_dict_id(self):
        assert self.dr._extract_id_from_result({"Id": "abc"}, "VehicleId") == "abc"

    def test_direct_dict_vehicle_id(self):
        assert self.dr._extract_id_from_result({"VehicleId": "v1"}, "VehicleId") == "v1"

    def test_array_first_element(self):
        data = [{"Id": "first"}, {"Id": "second"}]
        assert self.dr._extract_id_from_result(data, "VehicleId") == "first"

    def test_empty_array(self):
        assert self.dr._extract_id_from_result([], "VehicleId") is None

    def test_wrapped_data(self):
        data = {"Data": [{"Id": "wrapped-id"}]}
        assert self.dr._extract_id_from_result(data, "VehicleId") == "wrapped-id"

    def test_wrapped_items(self):
        data = {"Items": [{"Id": "item-id"}]}
        assert self.dr._extract_id_from_result(data, "VehicleId") == "item-id"

    def test_wrapped_results(self):
        data = {"Results": [{"Id": "result-id"}]}
        assert self.dr._extract_id_from_result(data, "VehicleId") == "result-id"

    def test_wrapped_dict(self):
        data = {"Data": {"Id": "nested-dict-id"}}
        assert self.dr._extract_id_from_result(data, "VehicleId") == "nested-dict-id"

    def test_case_insensitive_search(self):
        data = {"vehicleid": "case-id"}
        assert self.dr._extract_id_from_result(data, "VehicleId") == "case-id"

    def test_case_insensitive_id(self):
        data = {"ID": "upper-id"}
        assert self.dr._extract_id_from_result(data, "SomethingElse") == "upper-id"

    def test_falsy_value_skipped(self):
        """If value is empty/None/0, should skip it."""
        data = {"Id": "", "VehicleId": "fallback"}
        assert self.dr._extract_id_from_result(data, "VehicleId") == "fallback"


# ============================================================================
# 9. clear_cache
# ============================================================================

class TestClearCache:

    def test_clears_cache(self):
        dr = DependencyResolver(_make_registry())
        dr._resolution_cache["key1"] = {"value": "v1", "tool": "t1"}
        dr._resolution_cache["key2"] = {"value": "v2", "tool": "t2"}
        dr.clear_cache()
        assert dr._resolution_cache == {}


# ============================================================================
# 10. detect_entity_reference
# ============================================================================

class TestDetectEntityReference:

    def setup_method(self):
        self.dr = DependencyResolver(_make_registry())

    # -- ordinal --
    def test_ordinal_vozilo_1(self):
        ref = self.dr.detect_entity_reference("Dodaj km na Vozilo 1")
        assert ref is not None
        assert ref.reference_type == "ordinal"
        assert ref.ordinal_index == 0  # 1-indexed input -> 0-indexed

    def test_ordinal_auto_2(self):
        ref = self.dr.detect_entity_reference("auto 2")
        assert ref is not None
        assert ref.reference_type == "ordinal"
        assert ref.ordinal_index == 1

    def test_ordinal_vehicle_3(self):
        ref = self.dr.detect_entity_reference("vehicle 3 details")
        assert ref is not None
        assert ref.ordinal_index == 2

    def test_ordinal_zero_skipped(self):
        """Ordinal < 1 should be skipped."""
        ref = self.dr.detect_entity_reference("vozilo 0")
        assert ref is None

    # -- possessive --
    def test_possessive_moje_vozilo(self):
        ref = self.dr.detect_entity_reference("Koja je km na moje vozilo")
        assert ref is not None
        assert ref.reference_type == "possessive"
        assert ref.is_possessive is True

    def test_possessive_moj_auto(self):
        ref = self.dr.detect_entity_reference("moj auto")
        assert ref is not None
        assert ref.reference_type == "possessive"

    def test_possessive_my_vehicle(self):
        ref = self.dr.detect_entity_reference("show my vehicle")
        assert ref is not None
        assert ref.reference_type == "possessive"
        assert ref.is_possessive is True

    def test_possessive_my_car(self):
        ref = self.dr.detect_entity_reference("check my car status")
        assert ref is not None
        assert ref.reference_type == "possessive"

    # -- empty / no match --
    def test_empty_text(self):
        assert self.dr.detect_entity_reference("") is None

    def test_none_text(self):
        assert self.dr.detect_entity_reference(None) is None

    def test_no_match(self):
        assert self.dr.detect_entity_reference("Koliko je sati?") is None

    # -- vehicle name patterns (empty list) --
    def test_vehicle_name_patterns_empty(self):
        """VEHICLE_NAME_PATTERNS is intentionally empty."""
        ref = self.dr.detect_entity_reference("Golf", entity_type="vehicle")
        assert ref is None  # no hardcoded name patterns

    # -- entity_type filtering --
    def test_different_entity_type_no_match(self):
        """Ordinal patterns only match entity_type='vehicle'."""
        ref = self.dr.detect_entity_reference("vozilo 1", entity_type="person")
        assert ref is None


# ============================================================================
# 11. resolve_entity_reference
# ============================================================================

class TestResolveEntityReference:

    @pytest.mark.asyncio
    async def test_possessive_with_vehicle_in_context(self):
        dr = DependencyResolver(_make_registry())
        ref = EntityReference(
            entity_type="vehicle",
            reference_type="possessive",
            value="moje vozilo",
            is_possessive=True,
        )
        result = await dr.resolve_entity_reference(
            ref, _user_ctx_with_vehicle(), AsyncMock()
        )
        assert result.success is True
        assert result.resolved_value == VALID_VEHICLE_UUID
        assert result.provider_tool == "user_context"
        assert result.feedback["entity_type"] == "vehicle"

    @pytest.mark.asyncio
    async def test_possessive_without_vehicle(self):
        dr = DependencyResolver(_make_registry())
        ref = EntityReference(
            entity_type="vehicle",
            reference_type="possessive",
            value="moje vozilo",
            is_possessive=True,
        )
        result = await dr.resolve_entity_reference(
            ref, _user_ctx_no_vehicle(), AsyncMock()
        )
        assert result.success is False
        assert result.needs_user_selection is True
        assert "nemate postavljeno" in result.error_message

    @pytest.mark.asyncio
    async def test_unknown_reference_type(self):
        dr = DependencyResolver(_make_registry())
        ref = EntityReference(
            entity_type="vehicle",
            reference_type="unknown_type",
            value="something",
            is_possessive=False,
        )
        result = await dr.resolve_entity_reference(
            ref, _user_ctx_no_vehicle(), AsyncMock()
        )
        assert result.success is False
        assert "Ne mogu" in result.error_message

    @pytest.mark.asyncio
    async def test_ordinal_delegates_to_resolve_by_ordinal(self):
        """Ordinal reference should call _resolve_by_ordinal."""
        person_param = _make_param_def(context_key="person_id")
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id", "VehicleId"],
                          params={"PersonId": person_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        ref = EntityReference(
            entity_type="vehicle",
            reference_type="ordinal",
            value="vozilo 1",
            ordinal_index=0,
        )

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = [
            {"Id": "v-001", "FullVehicleName": "Golf", "LicencePlate": "ZG-111-AA"},
        ]
        executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr.resolve_entity_reference(
                ref, _user_ctx_no_vehicle(), executor
            )
        assert result.success is True
        assert result.resolved_value == "v-001"

    @pytest.mark.asyncio
    async def test_name_delegates_to_resolve_by_name(self):
        """Name reference should call _resolve_by_name."""
        person_param = _make_param_def(context_key="person_id")
        filter_param = _make_param_def(context_key=None)
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id", "VehicleId"],
                          params={"PersonId": person_param, "Filter": filter_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        ref = EntityReference(
            entity_type="vehicle",
            reference_type="name",
            value="Golf",
        )

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = [
            {"Id": "v-golf", "FullVehicleName": "VW Golf", "LicencePlate": "ZG-111-AA"},
        ]
        executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr.resolve_entity_reference(
                ref, _user_ctx_no_vehicle(), executor
            )
        assert result.success is True
        assert result.resolved_value == "v-golf"


# ============================================================================
# 12. _resolve_by_ordinal
# ============================================================================

class TestResolveByOrdinal:

    def _ref(self, idx=0):
        return EntityReference(
            entity_type="vehicle",
            reference_type="ordinal",
            value=f"vozilo {idx + 1}",
            ordinal_index=idx,
        )

    @pytest.mark.asyncio
    async def test_no_provider_tool(self):
        dr = DependencyResolver(_make_registry())
        result = await dr._resolve_by_ordinal(self._ref(), _user_ctx_no_vehicle(), AsyncMock())
        assert result.success is False
        assert "pronaći alat" in result.error_message

    @pytest.mark.asyncio
    async def test_provider_tool_not_found_in_registry(self):
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id"])
        reg = MagicMock()
        reg.tools = {"get_Vehicles": tool}
        reg.get_tool = MagicMock(return_value=None)

        dr = DependencyResolver(reg)
        result = await dr._resolve_by_ordinal(self._ref(), _user_ctx_no_vehicle(), AsyncMock())
        assert result.success is False
        assert "nije dostupan" in result.error_message

    @pytest.mark.asyncio
    async def test_success_returns_vehicle_by_index(self):
        person_param = _make_param_def(context_key="person_id")
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id", "VehicleId"],
                          params={"PersonId": person_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = [
            {"Id": "v-001", "FullVehicleName": "Golf", "LicencePlate": "ZG-111-AA"},
            {"Id": "v-002", "FullVehicleName": "Passat", "LicencePlate": "ZG-222-BB"},
        ]
        executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr._resolve_by_ordinal(self._ref(1), _user_ctx_no_vehicle(), executor)
        assert result.success is True
        assert result.resolved_value == "v-002"
        assert result.feedback["resolved_to"] == "Passat"

    @pytest.mark.asyncio
    async def test_index_out_of_bounds(self):
        person_param = _make_param_def(context_key="person_id")
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id", "VehicleId"],
                          params={"PersonId": person_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = [{"Id": "v-001", "FullVehicleName": "Golf"}]
        executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr._resolve_by_ordinal(self._ref(5), _user_ctx_no_vehicle(), executor)
        assert result.success is False
        assert "ne postoji" in result.error_message

    @pytest.mark.asyncio
    async def test_empty_vehicle_list(self):
        person_param = _make_param_def(context_key="person_id")
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id", "VehicleId"],
                          params={"PersonId": person_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = []
        executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr._resolve_by_ordinal(self._ref(), _user_ctx_no_vehicle(), executor)
        assert result.success is False
        assert "Nema dostupnih" in result.error_message

    @pytest.mark.asyncio
    async def test_executor_exception(self):
        person_param = _make_param_def(context_key="person_id")
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id", "VehicleId"],
                          params={"PersonId": person_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        executor = AsyncMock()
        executor.execute = AsyncMock(side_effect=RuntimeError("connection lost"))

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr._resolve_by_ordinal(self._ref(), _user_ctx_no_vehicle(), executor)
        assert result.success is False
        assert "connection lost" in result.error_message

    @pytest.mark.asyncio
    async def test_executor_returns_failure(self):
        person_param = _make_param_def(context_key="person_id")
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id", "VehicleId"],
                          params={"PersonId": person_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = False
        exec_result.error_message = "forbidden"
        executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr._resolve_by_ordinal(self._ref(), _user_ctx_no_vehicle(), executor)
        assert result.success is False
        assert result.error_message == "forbidden"


# ============================================================================
# 13. _resolve_by_name
# ============================================================================

class TestResolveByName:

    def _ref(self, name="Golf"):
        return EntityReference(
            entity_type="vehicle",
            reference_type="name",
            value=name,
        )

    def _setup_dr_and_executor(self, data, second_data=None):
        person_param = _make_param_def(context_key="person_id")
        filter_param = _make_param_def(context_key=None)
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id", "VehicleId"],
                          params={"PersonId": person_param, "Filter": filter_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = data
        executor.execute = AsyncMock(return_value=exec_result)
        return dr, executor

    @pytest.mark.asyncio
    async def test_single_match(self):
        dr, executor = self._setup_dr_and_executor([
            {"Id": "v-golf", "FullVehicleName": "VW Golf", "LicencePlate": "ZG-111-AA"},
        ])

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr._resolve_by_name(self._ref("Golf"), _user_ctx_no_vehicle(), executor)
        assert result.success is True
        assert result.resolved_value == "v-golf"

    @pytest.mark.asyncio
    async def test_multiple_matches_ambiguity(self):
        dr, executor = self._setup_dr_and_executor([
            {"Id": "v-g7", "FullVehicleName": "VW Golf 7", "LicencePlate": "ZG-111-AA"},
            {"Id": "v-g8", "FullVehicleName": "VW Golf 8", "LicencePlate": "ZG-222-BB"},
        ])

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr._resolve_by_name(self._ref("Golf"), _user_ctx_no_vehicle(), executor)
        assert result.success is False
        assert result.needs_user_selection is True
        assert "2 vozila" in result.error_message

    @pytest.mark.asyncio
    async def test_no_match(self):
        dr, executor = self._setup_dr_and_executor([
            {"Id": "v-p", "FullVehicleName": "Passat", "LicencePlate": "ZG-333-CC"},
        ])

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr._resolve_by_name(self._ref("BMW"), _user_ctx_no_vehicle(), executor)
        assert result.success is False
        assert "nije pronađeno" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_no_provider_tool(self):
        dr = DependencyResolver(_make_registry())
        result = await dr._resolve_by_name(self._ref(), _user_ctx_no_vehicle(), AsyncMock())
        assert result.success is False

    @pytest.mark.asyncio
    async def test_executor_exception(self):
        person_param = _make_param_def(context_key="person_id")
        filter_param = _make_param_def(context_key=None)
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id", "VehicleId"],
                          params={"PersonId": person_param, "Filter": filter_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        executor = AsyncMock()
        executor.execute = AsyncMock(side_effect=RuntimeError("timeout"))

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr._resolve_by_name(self._ref(), _user_ctx_no_vehicle(), executor)
        assert result.success is False
        assert "timeout" in result.error_message

    @pytest.mark.asyncio
    async def test_fallback_retry_on_first_failure(self):
        """When first execute fails, it retries with empty params."""
        person_param = _make_param_def(context_key="person_id")
        filter_param = _make_param_def(context_key=None)
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id", "VehicleId"],
                          params={"PersonId": person_param, "Filter": filter_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        fail_result = MagicMock()
        fail_result.success = False
        fail_result.error_message = "filter error"

        success_result = MagicMock()
        success_result.success = True
        success_result.data = [{"Id": "v-retry", "FullVehicleName": "Golf"}]

        executor = AsyncMock()
        executor.execute = AsyncMock(side_effect=[fail_result, success_result])

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr._resolve_by_name(self._ref("Golf"), _user_ctx_no_vehicle(), executor)
        assert result.success is True
        assert result.resolved_value == "v-retry"
        assert executor.execute.call_count == 2


# ============================================================================
# 14. _extract_vehicle_list
# ============================================================================

class TestExtractVehicleList:

    def setup_method(self):
        self.dr = DependencyResolver(_make_registry())

    def test_none_data(self):
        assert self.dr._extract_vehicle_list(None) == []

    def test_empty_list(self):
        assert self.dr._extract_vehicle_list([]) == []

    def test_direct_list(self):
        data = [{"Id": "1"}, {"Id": "2"}]
        assert self.dr._extract_vehicle_list(data) == data

    def test_wrapped_data(self):
        inner = [{"Id": "d1"}]
        assert self.dr._extract_vehicle_list({"Data": inner}) == inner

    def test_wrapped_items(self):
        inner = [{"Id": "i1"}]
        assert self.dr._extract_vehicle_list({"Items": inner}) == inner

    def test_wrapped_results(self):
        inner = [{"Id": "r1"}]
        assert self.dr._extract_vehicle_list({"Results": inner}) == inner

    def test_wrapped_value(self):
        inner = [{"Id": "val1"}]
        assert self.dr._extract_vehicle_list({"value": inner}) == inner

    def test_single_vehicle_dict_with_id(self):
        data = {"Id": "single"}
        assert self.dr._extract_vehicle_list(data) == [data]

    def test_single_vehicle_dict_with_vehicle_id(self):
        data = {"VehicleId": "single-v"}
        assert self.dr._extract_vehicle_list(data) == [data]

    def test_dict_no_id_no_wrapper(self):
        data = {"SomeField": "value"}
        assert self.dr._extract_vehicle_list(data) == []

    def test_empty_dict(self):
        assert self.dr._extract_vehicle_list({}) == []


# ============================================================================
# 15. _fuzzy_match_vehicles
# ============================================================================

class TestFuzzyMatchVehicles:

    def setup_method(self):
        self.dr = DependencyResolver(_make_registry())
        self.vehicles = [
            {"Id": "1", "FullVehicleName": "VW Golf 7", "LicencePlate": "ZG-111-AA", "Description": "Compact car"},
            {"Id": "2", "FullVehicleName": "Skoda Octavia", "LicencePlate": "ZG-222-BB", "Description": "Sedan"},
            {"Id": "3", "FullVehicleName": "VW Passat", "LicencePlate": "ST-333-CC", "Description": "Estate"},
        ]

    def test_exact_match_in_name(self):
        result = self.dr._fuzzy_match_vehicles(self.vehicles, "Golf")
        assert len(result) == 1
        assert result[0]["Id"] == "1"

    def test_partial_match_in_description(self):
        """Second pass: search_lower in searchable (all fields)."""
        result = self.dr._fuzzy_match_vehicles(self.vehicles, "Compact")
        assert len(result) == 1
        assert result[0]["Id"] == "1"

    def test_partial_match_plate(self):
        result = self.dr._fuzzy_match_vehicles(self.vehicles, "ST-333")
        assert len(result) == 1
        assert result[0]["Id"] == "3"

    def test_word_level_match(self):
        """Third pass: word overlap."""
        result = self.dr._fuzzy_match_vehicles(self.vehicles, "VW")
        # Both "VW Golf 7" and "VW Passat" match by name
        assert len(result) == 2

    def test_no_match(self):
        result = self.dr._fuzzy_match_vehicles(self.vehicles, "BMW")
        assert result == []

    def test_empty_vehicle_list(self):
        result = self.dr._fuzzy_match_vehicles([], "Golf")
        assert result == []

    def test_empty_search_term(self):
        result = self.dr._fuzzy_match_vehicles(self.vehicles, "")
        assert result == []

    def test_multiple_exact_matches(self):
        """When two vehicles both contain 'VW' in name - first pass."""
        result = self.dr._fuzzy_match_vehicles(self.vehicles, "VW")
        ids = [v["Id"] for v in result]
        assert "1" in ids
        assert "3" in ids


# ============================================================================
# 16. _fuzzy_match_vehicle (legacy)
# ============================================================================

class TestFuzzyMatchVehicleLegacy:

    def setup_method(self):
        self.dr = DependencyResolver(_make_registry())

    def test_returns_first_match(self):
        vehicles = [
            {"Id": "1", "FullVehicleName": "VW Golf"},
            {"Id": "2", "FullVehicleName": "VW Passat"},
        ]
        result = self.dr._fuzzy_match_vehicle(vehicles, "VW")
        assert result is not None
        assert result["Id"] == "1"

    def test_no_match_returns_none(self):
        vehicles = [{"Id": "1", "FullVehicleName": "Passat"}]
        result = self.dr._fuzzy_match_vehicle(vehicles, "BMW")
        assert result is None

    def test_empty_list_returns_none(self):
        assert self.dr._fuzzy_match_vehicle([], "Golf") is None


# ============================================================================
# EXTRA EDGE CASES
# ============================================================================

class TestEdgeCases:

    def test_value_patterns_property(self):
        """VALUE_PATTERNS property returns list from PatternRegistry."""
        dr = DependencyResolver(_make_registry())
        patterns = dr.VALUE_PATTERNS
        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_param_providers_exist(self):
        """PARAM_PROVIDERS should have vehicleid, personid, locationid, bookingid."""
        dr = DependencyResolver(_make_registry())
        assert "vehicleid" in dr.PARAM_PROVIDERS
        assert "personid" in dr.PARAM_PROVIDERS
        assert "locationid" in dr.PARAM_PROVIDERS
        assert "bookingid" in dr.PARAM_PROVIDERS

    def test_ordinal_patterns_class_attr(self):
        assert len(DependencyResolver.ORDINAL_PATTERNS) > 0

    def test_possessive_patterns_class_attr(self):
        assert len(DependencyResolver.POSSESSIVE_PATTERNS) > 0

    def test_vehicle_name_patterns_empty(self):
        assert DependencyResolver.VEHICLE_NAME_PATTERNS == []

    @pytest.mark.asyncio
    async def test_resolve_dependency_no_user_value(self):
        """resolve_dependency with user_value=None should still work."""
        person_param = _make_param_def(context_key="person_id")
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id"],
                          params={"PersonId": person_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {"Id": "direct-id"}
        executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr.resolve_dependency(
                "VehicleId", None, _user_ctx_no_vehicle(), executor
            )
        assert result.success is True
        assert result.resolved_value == "direct-id"

    @pytest.mark.asyncio
    async def test_resolve_by_ordinal_caches_result(self):
        person_param = _make_param_def(context_key="person_id")
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id", "VehicleId"],
                          params={"PersonId": person_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = [{"Id": "v-cached", "FullVehicleName": "Cached Golf"}]
        executor.execute = AsyncMock(return_value=exec_result)

        ref = EntityReference(
            entity_type="vehicle", reference_type="ordinal",
            value="vozilo 1", ordinal_index=0,
        )

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr._resolve_by_ordinal(ref, _user_ctx_no_vehicle(), executor)
        assert result.success is True
        assert "ordinal:vozilo 1" in dr._resolution_cache

    @pytest.mark.asyncio
    async def test_resolve_dependency_with_person_id_filter_injection(self):
        """When tool has Filter param but no direct person_id param, PersonId is added to Filter."""
        filter_param = _make_param_def(context_key=None)
        tool = _make_tool("get_Vehicles", method="GET", output_keys=["Id"],
                          params={"Filter": filter_param})
        reg = _make_registry(tools={"get_Vehicles": tool})
        dr = DependencyResolver(reg)

        executor = AsyncMock()
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = [{"Id": "filter-id"}]
        executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext", MagicMock()):
            result = await dr.resolve_dependency(
                "VehicleId", "ZG-1234-AB", _user_ctx_no_vehicle(), executor
            )
        assert result.success is True

    def test_detect_value_type_vin(self):
        dr = DependencyResolver(_make_registry())
        # 17-char VIN (no I, O, Q)
        result = dr.detect_value_type("WVWZZZ3CZWE123456")
        assert result is not None
        assert result[0] == "vehicleid"
        assert result[1] == "VIN"

    def test_extract_id_from_result_lowercase_wrapper(self):
        dr = DependencyResolver(_make_registry())
        data = {"data": [{"Id": "lower-wrapped"}]}
        assert dr._extract_id_from_result(data, "VehicleId") == "lower-wrapped"

    def test_extract_id_from_result_items_lowercase(self):
        dr = DependencyResolver(_make_registry())
        data = {"items": [{"Id": "lower-items"}]}
        assert dr._extract_id_from_result(data, "VehicleId") == "lower-items"

    def test_extract_id_from_result_results_lowercase(self):
        dr = DependencyResolver(_make_registry())
        data = {"results": [{"Id": "lower-results"}]}
        assert dr._extract_id_from_result(data, "VehicleId") == "lower-results"

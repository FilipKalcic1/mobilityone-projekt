"""Tests for services/engine/flow_handler.py – FlowHandler."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler():
    registry = MagicMock()
    executor = AsyncMock()
    ai = MagicMock()
    ai.extract_parameters = AsyncMock(return_value={})
    formatter = MagicMock()
    formatter.format_vehicle_list = MagicMock(return_value="Vehicle list")
    formatter.format_result = MagicMock(return_value="Formatted")

    with patch("services.engine.flow_handler.get_confirmation_dialog") as mock_cd:
        dialog = MagicMock()
        dialog.format_parameters.return_value = {"param1": "val1"}
        dialog.generate_confirmation_message.return_value = "Potvrdite operaciju?"
        dialog.parse_modification.return_value = None
        dialog.DISPLAY_NAMES = {"Note": "Bilješka"}
        mock_cd.return_value = dialog

        from services.engine.flow_handler import FlowHandler
        handler = FlowHandler(registry, executor, ai, formatter)

    return handler


def _make_conv(state="idle", flow=None, tool=None, missing=None,
               items=None, selected=None, params=None):
    conv = MagicMock()
    conv.get_state.return_value = MagicMock(value=state)
    conv.get_current_flow.return_value = flow
    conv.get_current_tool.return_value = tool
    conv.get_missing_params.return_value = missing or []
    conv.get_displayed_items.return_value = items or []
    conv.get_selected_item.return_value = selected
    conv.get_parameters.return_value = params or {}
    conv.has_all_required_params.return_value = False
    conv.parse_item_selection.return_value = selected
    conv.parse_confirmation.return_value = None

    conv.start_flow = AsyncMock()
    conv.add_parameters = AsyncMock()
    conv.set_displayed_items = AsyncMock()
    conv.select_item = AsyncMock()
    conv.request_confirmation = AsyncMock()
    conv.request_selection = AsyncMock()
    conv.confirm = AsyncMock()
    conv.cancel = AsyncMock()
    conv.complete = AsyncMock()
    conv.reset = AsyncMock()
    conv.save = AsyncMock()
    conv.context = MagicMock()
    conv.context.tool_outputs = {}
    conv.context.current_tool = None
    return conv


def _user_context():
    return {
        "person_id": "00000000-0000-0000-0000-000000000001",
        "phone": "+385991234567",
        "tenant_id": "t1",
    }


@pytest.fixture
def handler():
    return _make_handler()


# ===========================================================================
# _extract_items
# ===========================================================================

class TestExtractItems:
    def test_dict_with_items(self, handler):
        data = {"items": [{"Id": "1"}, {"Id": "2"}]}
        assert len(handler._extract_items(data)) == 2

    def test_dict_with_data(self, handler):
        data = {"Data": [{"Id": "1"}]}
        assert len(handler._extract_items(data)) == 1

    def test_nested_data(self, handler):
        data = {"data": {"Data": [{"Id": "1"}]}}
        assert len(handler._extract_items(data)) == 1

    def test_nested_data_list(self, handler):
        data = {"data": [{"Id": "1"}, {"Id": "2"}]}
        assert len(handler._extract_items(data)) == 2

    def test_empty_dict(self, handler):
        assert handler._extract_items({}) == []

    def test_non_dict(self, handler):
        assert handler._extract_items("string") == []


# ===========================================================================
# _is_question
# ===========================================================================

class TestIsQuestion:
    def test_question_mark(self, handler):
        assert handler._is_question("koliko km?") is True

    def test_koliko_keyword(self, handler):
        assert handler._is_question("koliko ima km") is True

    def test_koji_keyword(self, handler):
        assert handler._is_question("koji auto imam") is True

    def test_sto_keyword(self, handler):
        assert handler._is_question("što je to") is True

    def test_ima_li(self, handler):
        assert handler._is_question("ima li slobodnih") is True

    def test_reci_mi(self, handler):
        assert handler._is_question("reci mi km") is True

    def test_not_question_da(self, handler):
        assert handler._is_question("da") is False

    def test_not_question_ne(self, handler):
        assert handler._is_question("ne") is False

    def test_not_question_regular(self, handler):
        assert handler._is_question("sutra od 8") is False


# ===========================================================================
# _extract_filter_text
# ===========================================================================

class TestExtractFilterText:
    def test_pokazi_passat(self, handler):
        assert handler._extract_filter_text("pokaži Passat") == "passat"

    def test_pokazi_samo_zg(self, handler):
        assert handler._extract_filter_text("pokaži samo ZG") == "zg"

    def test_filtriraj(self, handler):
        assert handler._extract_filter_text("filtriraj Golf") == "golf"

    def test_samo(self, handler):
        assert handler._extract_filter_text("samo Octavia") == "octavia"

    def test_no_filter(self, handler):
        assert handler._extract_filter_text("da") is None

    def test_confirmation_not_treated_as_filter(self, handler):
        assert handler._extract_filter_text("samo da") is None

    def test_trazi(self, handler):
        assert handler._extract_filter_text("traži VW") == "vw"


# ===========================================================================
# _build_param_prompt
# ===========================================================================

class TestBuildParamPrompt:
    def test_single_param(self, handler):
        with patch("services.engine.flow_handler.get_multiple_missing_prompts", return_value="Navedite FromTime:"):
            result = handler._build_param_prompt(["FromTime"])
            assert "FromTime" in result

    def test_multiple_params(self, handler):
        with patch("services.engine.flow_handler.get_multiple_missing_prompts", return_value="Navedite..."):
            result = handler._build_param_prompt(["FromTime", "ToTime"])
            assert "Navedite" in result


# ===========================================================================
# handle_availability
# ===========================================================================

class TestHandleAvailability:
    @pytest.mark.asyncio
    async def test_missing_time_params(self, handler):
        conv = _make_conv()
        result = await handler.handle_availability(
            "get_AvailableVehicles", {}, _user_context(), conv
        )
        assert result["needs_input"] is True
        assert "period" in result["prompt"].lower()
        conv.start_flow.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_time_partial(self, handler):
        conv = _make_conv()
        result = await handler.handle_availability(
            "get_AvailableVehicles", {"from": "sutra"}, _user_context(), conv
        )
        assert result["needs_input"] is True
        conv.add_parameters.assert_called()

    @pytest.mark.asyncio
    async def test_tool_not_found(self, handler):
        handler.registry.get_tool.return_value = None
        conv = _make_conv()
        result = await handler.handle_availability(
            "get_AvailableVehicles", {"from": "sutra", "to": "petak"}, _user_context(), conv
        )
        assert result["success"] is False
        assert "prona" in result["final_response"].lower()

    @pytest.mark.asyncio
    async def test_execution_fails(self, handler):
        tool = MagicMock()
        handler.registry.get_tool.return_value = tool

        exec_result = MagicMock()
        exec_result.success = False
        exec_result.error_message = "API error"
        exec_result.ai_feedback = None
        handler.executor.execute = AsyncMock(return_value=exec_result)

        conv = _make_conv()
        with patch("services.tool_contracts.ToolExecutionContext"):
            result = await handler.handle_availability(
                "get_AvailableVehicles", {"from": "sutra", "to": "petak"},
                _user_context(), conv
            )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_no_vehicles_found(self, handler):
        tool = MagicMock()
        handler.registry.get_tool.return_value = tool

        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {"items": []}
        handler.executor.execute = AsyncMock(return_value=exec_result)

        conv = _make_conv()
        with patch("services.tool_contracts.ToolExecutionContext"):
            result = await handler.handle_availability(
                "get_AvailableVehicles", {"from": "sutra", "to": "petak"},
                _user_context(), conv
            )
        assert "nema slobodnih" in result["final_response"].lower()

    @pytest.mark.asyncio
    async def test_vehicles_found(self, handler):
        tool = MagicMock()
        handler.registry.get_tool.return_value = tool

        items = [
            {"Id": "v1", "FullVehicleName": "Golf", "LicencePlate": "ZG-1234-AB"},
            {"Id": "v2", "FullVehicleName": "Passat", "LicencePlate": "ZG-5678-CD"},
        ]
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {"items": items}
        handler.executor.execute = AsyncMock(return_value=exec_result)

        conv = _make_conv()
        with patch("services.tool_contracts.ToolExecutionContext"):
            result = await handler.handle_availability(
                "get_AvailableVehicles", {"from": "sutra", "to": "petak"},
                _user_context(), conv
            )
        assert result["needs_input"] is True
        assert "Golf" in result["prompt"]
        assert "ZG-1234-AB" in result["prompt"]
        conv.set_displayed_items.assert_called_once()


# ===========================================================================
# request_confirmation
# ===========================================================================

class TestRequestConfirmation:
    @pytest.mark.asyncio
    async def test_basic(self, handler):
        tool = MagicMock()
        tool.description = "Test operation"
        handler.registry.get_tool.return_value = tool

        conv = _make_conv()
        result = await handler.request_confirmation(
            "post_VehicleCalendar", {"VehicleId": "v1"}, _user_context(), conv
        )
        assert result["needs_input"] is True
        assert "Potvrdite" in result["prompt"]
        conv.request_confirmation.assert_called_once()


# ===========================================================================
# handle_selection
# ===========================================================================

class TestHandleSelection:
    @pytest.mark.asyncio
    async def test_no_selection(self, handler):
        conv = _make_conv(selected=None)
        conv.parse_item_selection.return_value = None

        result = await handler.handle_selection(
            "sender", "abc", _user_context(), conv, AsyncMock()
        )
        assert "razumio" in result.lower()

    @pytest.mark.asyncio
    async def test_valid_selection(self, handler):
        selected = {"Id": "v1", "FullVehicleName": "Golf", "LicencePlate": "ZG-123"}
        conv = _make_conv(params={"from": "sutra", "to": "petak"})
        conv.parse_item_selection.return_value = selected

        result = await handler.handle_selection(
            "sender", "1", _user_context(), conv, AsyncMock()
        )
        assert "Golf" in result
        assert "ZG-123" in result
        conv.select_item.assert_called_once()
        conv.request_confirmation.assert_called_once()


# ===========================================================================
# handle_confirmation
# ===========================================================================

class TestHandleConfirmation:
    @pytest.mark.asyncio
    async def test_show_more_vehicles(self, handler):
        conv = _make_conv()
        conv.context.tool_outputs = {
            "all_available_vehicles": [
                {"Id": "v1", "DisplayName": "Golf"},
                {"Id": "v2", "DisplayName": "Passat"},
            ]
        }

        result = await handler.handle_confirmation(
            "sender", "pokaži ostala", _user_context(), conv
        )
        assert result == "Vehicle list"
        conv.request_selection.assert_called_once()

    @pytest.mark.asyncio
    async def test_show_more_no_vehicles(self, handler):
        conv = _make_conv()
        conv.context.tool_outputs = {"all_available_vehicles": [{"Id": "v1"}]}

        result = await handler.handle_confirmation(
            "sender", "pokaži ostala", _user_context(), conv
        )
        assert "nema drugih" in result.lower()

    @pytest.mark.asyncio
    async def test_filter_vehicles(self, handler):
        conv = _make_conv()
        conv.context.tool_outputs = {
            "all_available_vehicles": [{"Id": "v1"}, {"Id": "v2"}]
        }

        result = await handler.handle_confirmation(
            "sender", "pokaži Passat", _user_context(), conv
        )
        assert result == "Vehicle list"

    @pytest.mark.asyncio
    async def test_modification(self, handler):
        handler.confirmation_dialog.parse_modification.return_value = ("Note", "službeni put")

        conv = _make_conv(params={"Note": "test"})
        result = await handler.handle_confirmation(
            "sender", "Bilješka: službeni put", _user_context(), conv
        )
        assert "Ažurirano" in result

    @pytest.mark.asyncio
    async def test_cancel(self, handler):
        handler.confirmation_dialog.parse_modification.return_value = None
        conv = _make_conv()
        conv.parse_confirmation.return_value = False

        result = await handler.handle_confirmation(
            "sender", "ne", _user_context(), conv
        )
        assert "otkazana" in result.lower()
        conv.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_response_question(self, handler):
        handler.confirmation_dialog.parse_modification.return_value = None
        conv = _make_conv()
        conv.parse_confirmation.return_value = None

        result = await handler.handle_confirmation(
            "sender", "koliko km ima auto?", _user_context(), conv
        )
        assert isinstance(result, dict)
        assert result["mid_flow_question"] is True

    @pytest.mark.asyncio
    async def test_unknown_response_not_question(self, handler):
        handler.confirmation_dialog.parse_modification.return_value = None
        conv = _make_conv()
        conv.parse_confirmation.return_value = None

        result = await handler.handle_confirmation(
            "sender", "sutra", _user_context(), conv
        )
        assert "potvrdite" in result.lower()

    @pytest.mark.asyncio
    async def test_confirm_booking_success(self, handler):
        handler.confirmation_dialog.parse_modification.return_value = None

        selected = {"Id": "v1", "FullVehicleName": "Golf", "LicencePlate": "ZG-123"}
        conv = _make_conv(
            tool="post_VehicleCalendar",
            params={"from": "sutra 8:00", "to": "sutra 17:00"},
            selected=selected
        )
        conv.parse_confirmation.return_value = True

        tool = MagicMock()
        tool.description = "Book vehicle"
        handler.registry.get_tool.return_value = tool

        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {"Id": "booking-1"}
        handler.executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext"):
            result = await handler.handle_confirmation(
                "sender", "da", _user_context(), conv
            )
        assert "uspje" in result.lower()
        assert "Golf" in result
        conv.confirm.assert_called_once()
        conv.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_confirm_booking_no_vehicle(self, handler):
        handler.confirmation_dialog.parse_modification.return_value = None

        conv = _make_conv(
            tool="post_VehicleCalendar",
            params={"from": "sutra", "to": "petak"},
            selected=None,
        )
        conv.parse_confirmation.return_value = True
        conv.context.tool_outputs = {}

        result = await handler.handle_confirmation(
            "sender", "da", _user_context(), conv
        )
        assert "gre" in result.lower()
        conv.reset.assert_called()

    @pytest.mark.asyncio
    async def test_confirm_booking_no_time(self, handler):
        handler.confirmation_dialog.parse_modification.return_value = None

        selected = {"Id": "v1"}
        conv = _make_conv(
            tool="post_VehicleCalendar",
            params={},
            selected=selected,
        )
        conv.parse_confirmation.return_value = True

        result = await handler.handle_confirmation(
            "sender", "da", _user_context(), conv
        )
        assert "nedostaje" in result.lower() or "gre" in result.lower()

    @pytest.mark.asyncio
    async def test_confirm_booking_failure(self, handler):
        handler.confirmation_dialog.parse_modification.return_value = None

        selected = {"Id": "v1", "FullVehicleName": "Golf"}
        conv = _make_conv(
            tool="post_VehicleCalendar",
            params={"from": "sutra", "to": "petak"},
            selected=selected,
        )
        conv.parse_confirmation.return_value = True

        tool = MagicMock()
        tool.description = "Book"
        handler.registry.get_tool.return_value = tool

        exec_result = MagicMock()
        exec_result.success = False
        exec_result.error_message = "Conflict"
        handler.executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext"):
            with patch("services.engine.flow_handler.get_error_translator") as mock_et:
                mock_et.return_value.get_user_message.return_value = "Greška pri rezervaciji"
                result = await handler.handle_confirmation(
                    "sender", "da", _user_context(), conv
                )
        assert "gre" in result.lower()

    @pytest.mark.asyncio
    async def test_confirm_case_success(self, handler):
        handler.confirmation_dialog.parse_modification.return_value = None

        conv = _make_conv(
            tool="post_AddCase",
            params={"Subject": "Kvar motora", "from": "sutra", "to": "petak"},
            selected={"Id": "v1"},
        )
        conv.parse_confirmation.return_value = True

        tool = MagicMock()
        tool.description = "Create case"
        handler.registry.get_tool.return_value = tool

        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {"Id": "case-1"}
        handler.executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext"):
            result = await handler.handle_confirmation(
                "sender", "da", _user_context(), conv
            )
        assert "uspješno kreirana" in result.lower() or "uspjesno" in result.lower()

    @pytest.mark.asyncio
    async def test_confirm_mileage_success(self, handler):
        handler.confirmation_dialog.parse_modification.return_value = None

        conv = _make_conv(
            tool="post_AddMileage",
            params={"Value": 50000, "from": "x", "to": "y"},
            selected=None,
        )
        conv.parse_confirmation.return_value = True

        tool = MagicMock()
        tool.description = "Add mileage"
        handler.registry.get_tool.return_value = tool

        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {}
        handler.executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext"):
            result = await handler.handle_confirmation(
                "sender", "da", _user_context(), conv
            )
        assert "uspješno unesena" in result.lower() or "uspjesno" in result.lower()

    @pytest.mark.asyncio
    async def test_confirm_generic_success(self, handler):
        handler.confirmation_dialog.parse_modification.return_value = None

        conv = _make_conv(
            tool="post_Something",
            params={"from": "x", "to": "y"},
            selected=None,
        )
        conv.parse_confirmation.return_value = True

        tool = MagicMock()
        tool.description = "Do something"
        handler.registry.get_tool.return_value = tool

        exec_result = MagicMock()
        exec_result.success = True
        exec_result.data = {"Id": "new-1"}
        handler.executor.execute = AsyncMock(return_value=exec_result)

        with patch("services.tool_contracts.ToolExecutionContext"):
            result = await handler.handle_confirmation(
                "sender", "da", _user_context(), conv
            )
        assert "uspje" in result.lower() or "uspješna" in result.lower()

    @pytest.mark.asyncio
    async def test_confirm_tool_not_found(self, handler):
        handler.confirmation_dialog.parse_modification.return_value = None

        conv = _make_conv(tool="missing_tool", params={"from": "x", "to": "y"})
        conv.parse_confirmation.return_value = True
        handler.registry.get_tool.return_value = None

        result = await handler.handle_confirmation(
            "sender", "da", _user_context(), conv
        )
        assert "problem" in result.lower()
        conv.reset.assert_called()


# ===========================================================================
# handle_gathering
# ===========================================================================

class TestHandleGathering:
    @pytest.mark.asyncio
    async def test_extraction_fills_params(self, handler):
        conv = _make_conv(missing=["FromTime", "ToTime"])
        conv.has_all_required_params.return_value = False
        conv.get_missing_params.side_effect = [
            ["FromTime", "ToTime"],  # first call
            ["ToTime"],             # after extraction
        ]

        handler.ai.extract_parameters = AsyncMock(return_value={"FromTime": "sutra 8:00"})

        with patch("services.engine.flow_handler.get_multiple_missing_prompts", return_value="Navedite ToTime:"):
            result = await handler.handle_gathering(
                "sender", "sutra od 8", _user_context(), conv, AsyncMock()
            )
        assert "ToTime" in result

    @pytest.mark.asyncio
    async def test_all_params_collected_booking(self, handler):
        conv = _make_conv(
            missing=["ToTime"],
            flow="booking",
            tool="get_AvailableVehicles",
            params={"from": "sutra", "to": "petak"}
        )
        conv.has_all_required_params.return_value = True

        handler.ai.extract_parameters = AsyncMock(return_value={"ToTime": "petak"})
        handler.handle_availability = AsyncMock(return_value={
            "needs_input": True,
            "prompt": "Pronasao vozilo!"
        })

        result = await handler.handle_gathering(
            "sender", "do petka", _user_context(), conv, AsyncMock()
        )
        assert "Pronasao" in result

    @pytest.mark.asyncio
    async def test_all_params_collected_mileage(self, handler):
        conv = _make_conv(
            missing=["Value"],
            flow="mileage_input",
            tool="post_AddMileage",
            params={"Value": 50000}
        )
        conv.has_all_required_params.return_value = True

        handler.ai.extract_parameters = AsyncMock(return_value={"Value": 50000})
        handler._show_mileage_confirmation = AsyncMock(return_value="Potvrda km!")

        result = await handler.handle_gathering(
            "sender", "50000", _user_context(), conv, AsyncMock()
        )
        assert "Potvrda" in result

    @pytest.mark.asyncio
    async def test_all_params_collected_case(self, handler):
        conv = _make_conv(
            missing=["Description"],
            flow="case_creation",
            tool="post_AddCase",
            params={"Subject": "Kvar", "Description": "Motor ne radi"}
        )
        conv.has_all_required_params.return_value = True

        handler.ai.extract_parameters = AsyncMock(return_value={"Description": "Motor ne radi"})
        handler._show_case_confirmation = AsyncMock(return_value="Potvrda slučaja!")

        result = await handler.handle_gathering(
            "sender", "motor ne radi", _user_context(), conv, AsyncMock()
        )
        assert "Potvrda" in result

    @pytest.mark.asyncio
    async def test_fallback_single_param_time(self, handler):
        conv = _make_conv(missing=["ToTime"])
        conv.has_all_required_params.return_value = False
        conv.get_missing_params.side_effect = [
            ["ToTime"],
            ["ToTime"],  # still missing after fallback attempt
        ]

        handler.ai.extract_parameters = AsyncMock(return_value={})

        with patch("services.engine.flow_handler.get_multiple_missing_prompts", return_value="Navedite ToTime:"):
            result = await handler.handle_gathering(
                "sender", "petak 17h", _user_context(), conv, AsyncMock()
            )
        # The fallback should have tried to set ToTime to "petak 17h"
        conv.add_parameters.assert_called()

    @pytest.mark.asyncio
    async def test_fallback_single_param_value(self, handler):
        conv = _make_conv(missing=["Value"])
        conv.has_all_required_params.return_value = False
        conv.get_missing_params.side_effect = [["Value"], []]

        # Extraction returns nothing
        handler.ai.extract_parameters = AsyncMock(return_value={})
        # After add_parameters, all params are collected
        conv.has_all_required_params.return_value = True
        conv.get_current_flow.return_value = "other"
        conv.get_current_tool.return_value = "some_tool"

        handle_new = AsyncMock(return_value="Done!")
        result = await handler.handle_gathering(
            "sender", "50000", _user_context(), conv, handle_new
        )
        # Fallback should parse "50000" from text


# ===========================================================================
# _show_mileage_confirmation
# ===========================================================================

class TestShowMileageConfirmation:
    @pytest.mark.asyncio
    async def test_basic(self, handler):
        conv = _make_conv()
        result = await handler._show_mileage_confirmation(
            {"_vehicle_name": "Golf", "_vehicle_plate": "ZG-123", "Value": 50000},
            conv
        )
        assert "50000" in result
        assert "Golf" in result
        conv.request_confirmation.assert_called_once()


# ===========================================================================
# _show_case_confirmation
# ===========================================================================

class TestShowCaseConfirmation:
    @pytest.mark.asyncio
    async def test_with_vehicle(self, handler):
        ctx = _user_context()
        ctx["vehicle"] = {"id": "v1", "plate": "ZG-123", "name": "Golf"}
        conv = _make_conv()
        result = await handler._show_case_confirmation(
            {"Subject": "Kvar", "Description": "Motor ne radi"},
            ctx, conv
        )
        assert "Kvar" in result
        assert "Motor ne radi" in result
        conv.request_confirmation.assert_called_once()

    @pytest.mark.asyncio
    async def test_without_vehicle(self, handler):
        conv = _make_conv()
        result = await handler._show_case_confirmation(
            {"Subject": "Kvar"},
            _user_context(), conv
        )
        assert "Kvar" in result

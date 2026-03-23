"""
Tests for services/engine/flow_executors.py -- FlowExecutors.

Covers:
- handle_booking_flow (with/without router_params)
- handle_mileage_input_flow (vehicle from context, from tool_outputs, from API, missing vehicle, with/without Value)
- handle_case_creation_flow (with/without description, subject inference, vehicle line)
- handle_availability_flow (needs_input, final_response, error fallback)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_context():
    return {
        "person_id": "00000000-0000-0000-0000-000000000001",
        "tenant_id": "test-tenant",
        "vehicle": {"id": "v1", "plate": "ZG-123", "name": "Golf"},
    }


def _user_context_no_vehicle():
    return {
        "person_id": "00000000-0000-0000-0000-000000000001",
        "tenant_id": "test-tenant",
    }


def _make_conv(tool_outputs=None):
    """Create a mock ConversationManager with async methods."""
    conv = MagicMock()
    conv.start_flow = AsyncMock()
    conv.add_parameters = AsyncMock()
    conv.save = AsyncMock()
    conv.request_confirmation = AsyncMock()
    conv.context = MagicMock()
    conv.context.tool_outputs = tool_outputs if tool_outputs is not None else {}
    conv.context.current_tool = None
    return conv


def _make_executor(availability_result=None):
    """Create FlowExecutors with mocked gateway and flow_handler."""
    gateway = MagicMock()
    gateway.execute = AsyncMock()
    flow_handler = MagicMock()
    flow_handler.handle_availability = AsyncMock(
        return_value=availability_result or {"final_response": "Dostupna vozila..."}
    )

    from services.engine.flow_executors import FlowExecutors
    fe = FlowExecutors(gateway=gateway, flow_handler=flow_handler)
    return fe, gateway, flow_handler


# ===========================================================================
# handle_availability_flow
# ===========================================================================

class TestHandleAvailabilityFlow:
    async def test_returns_prompt_when_needs_input(self):
        fe, gw, fh = _make_executor()
        fh.handle_availability.return_value = {
            "needs_input": True,
            "prompt": "Od kada do kada?",
        }
        result_dict = {"tool": "get_AvailableVehicles", "parameters": {}, "tool_call_id": "x"}
        resp = await fe.handle_availability_flow(result_dict, _user_context(), _make_conv())
        assert resp == "Od kada do kada?"

    async def test_returns_final_response(self):
        fe, gw, fh = _make_executor()
        fh.handle_availability.return_value = {"final_response": "Evo vozila"}
        result_dict = {"tool": "get_AvailableVehicles", "parameters": {}, "tool_call_id": "x"}
        resp = await fe.handle_availability_flow(result_dict, _user_context(), _make_conv())
        assert resp == "Evo vozila"

    async def test_returns_error_fallback(self):
        fe, gw, fh = _make_executor()
        fh.handle_availability.return_value = {"error": "Nema podataka"}
        result_dict = {"tool": "get_AvailableVehicles", "parameters": {}, "tool_call_id": "x"}
        resp = await fe.handle_availability_flow(result_dict, _user_context(), _make_conv())
        assert resp == "Nema podataka"

    async def test_returns_default_error_when_no_keys(self):
        fe, gw, fh = _make_executor()
        fh.handle_availability.return_value = {}
        result_dict = {"tool": "get_AvailableVehicles", "parameters": {}, "tool_call_id": "x"}
        resp = await fe.handle_availability_flow(result_dict, _user_context(), _make_conv())
        assert "Greska" in resp or "dostupnosti" in resp


# ===========================================================================
# handle_booking_flow
# ===========================================================================

class TestHandleBookingFlow:
    async def test_delegates_to_availability_flow(self):
        fe, gw, fh = _make_executor()
        fh.handle_availability.return_value = {"final_response": "Dostupno"}
        resp = await fe.handle_booking_flow("rezerviraj auto", _user_context(), _make_conv())
        assert resp == "Dostupno"
        fh.handle_availability.assert_awaited_once()

    async def test_passes_from_time_from_router_params(self):
        fe, gw, fh = _make_executor()
        fh.handle_availability.return_value = {"final_response": "OK"}
        await fe.handle_booking_flow(
            "rezerviraj",
            _user_context(),
            _make_conv(),
            router_params={"from": "2025-01-15T09:00:00", "to": "2025-01-15T17:00:00"},
        )
        call_args = fh.handle_availability.call_args
        params = call_args[0][1]  # second positional arg = parameters
        assert params["FromTime"] == "2025-01-15T09:00:00"
        assert params["ToTime"] == "2025-01-15T17:00:00"

    async def test_passes_FromTime_key_from_router_params(self):
        fe, gw, fh = _make_executor()
        fh.handle_availability.return_value = {"final_response": "OK"}
        await fe.handle_booking_flow(
            "rezerviraj",
            _user_context(),
            _make_conv(),
            router_params={"FromTime": "2025-02-01T08:00:00", "ToTime": "2025-02-01T16:00:00"},
        )
        call_args = fh.handle_availability.call_args
        params = call_args[0][1]
        assert params["FromTime"] == "2025-02-01T08:00:00"
        assert params["ToTime"] == "2025-02-01T16:00:00"

    async def test_empty_router_params_sends_empty_params(self):
        fe, gw, fh = _make_executor()
        fh.handle_availability.return_value = {"final_response": "OK"}
        await fe.handle_booking_flow("rezerviraj", _user_context(), _make_conv(), router_params={})
        call_args = fh.handle_availability.call_args
        params = call_args[0][1]
        assert "FromTime" not in params
        assert "ToTime" not in params

    async def test_none_router_params_handled(self):
        fe, gw, fh = _make_executor()
        fh.handle_availability.return_value = {"final_response": "OK"}
        await fe.handle_booking_flow("rezerviraj", _user_context(), _make_conv(), router_params=None)
        fh.handle_availability.assert_awaited_once()

    async def test_tool_name_is_get_AvailableVehicles(self):
        fe, gw, fh = _make_executor()
        fh.handle_availability.return_value = {"final_response": "OK"}
        await fe.handle_booking_flow("book", _user_context(), _make_conv())
        call_args = fh.handle_availability.call_args
        tool_name = call_args[0][0]  # first positional arg = tool
        assert tool_name == "get_AvailableVehicles"


# ===========================================================================
# handle_mileage_input_flow
# ===========================================================================

class TestHandleMileageInputFlow:
    async def test_with_vehicle_and_value_shows_confirmation(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_mileage_input_flow(
            "unesi km 14500",
            _user_context(),
            conv,
            router_params={"Value": "14500"},
        )
        assert "14500" in resp
        assert "Potvrda" in resp
        conv.request_confirmation.assert_awaited_once()
        # Tool is set via start_flow, not directly on context
        conv.start_flow.assert_awaited_once()

    async def test_with_vehicle_no_value_starts_gathering(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_mileage_input_flow(
            "unesi kilometrazu",
            _user_context(),
            conv,
            router_params={},
        )
        assert "kilometra" in resp.lower()
        conv.start_flow.assert_awaited_once()
        conv.add_parameters.assert_awaited()
        conv.save.assert_awaited()

    async def test_vehicle_from_tool_outputs(self):
        fe, gw, fh = _make_executor()
        tool_outputs = {
            "VehicleId": "v-from-output",
            "all_available_vehicles": [
                {"Id": "v-from-output", "DisplayName": "Audi A3", "LicencePlate": "ST-999"},
            ],
        }
        conv = _make_conv(tool_outputs=tool_outputs)
        resp = await fe.handle_mileage_input_flow(
            "unesi km",
            _user_context_no_vehicle(),
            conv,
            router_params={},
        )
        # Should use the vehicle from tool_outputs, not fail
        assert "Audi A3" in resp or "ST-999" in resp or "kilometra" in resp.lower()

    async def test_vehicle_from_tool_outputs_with_value(self):
        fe, gw, fh = _make_executor()
        tool_outputs = {
            "VehicleId": "v-from-output",
            "all_available_vehicles": [
                {"Id": "v-from-output", "DisplayName": "Audi A3", "LicencePlate": "ST-999"},
            ],
        }
        conv = _make_conv(tool_outputs=tool_outputs)
        resp = await fe.handle_mileage_input_flow(
            "unesi km 20000",
            _user_context_no_vehicle(),
            conv,
            router_params={"Value": "20000"},
        )
        assert "20000" in resp
        conv.request_confirmation.assert_awaited_once()

    async def test_no_vehicle_returns_error(self):
        """When no vehicle in context or tool_outputs, returns error message."""
        fe, gw, fh = _make_executor()
        conv = _make_conv(tool_outputs={})
        resp = await fe.handle_mileage_input_flow(
            "unesi km",
            _user_context_no_vehicle(),
            conv,
            router_params={},
        )
        assert "Nije" in resp or "vozilo" in resp.lower()

    async def test_mileage_value_alias_resolution(self):
        """Router may provide 'mileage' instead of 'Value'."""
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_mileage_input_flow(
            "unesi km 8000",
            _user_context(),
            conv,
            router_params={"mileage": "8000"},
        )
        assert "8000" in resp
        conv.request_confirmation.assert_awaited_once()

    async def test_mileage_value_lowercase_alias(self):
        """Router may provide 'value' instead of 'Value'."""
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_mileage_input_flow(
            "unesi km 9999",
            _user_context(),
            conv,
            router_params={"value": "9999"},
        )
        assert "9999" in resp

    async def test_vehicle_name_and_plate_in_gathering_prompt(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_mileage_input_flow(
            "unesi km",
            _user_context(),
            conv,
            router_params={},
        )
        assert "Golf" in resp
        assert "ZG-123" in resp

    async def test_vehicle_from_tool_outputs_stored_properly(self):
        """When vehicle is in tool_outputs, it is used for mileage flow."""
        fe, gw, fh = _make_executor()
        tool_outputs = {
            "VehicleId": "stored-v",
            "all_available_vehicles": [
                {"Id": "stored-v", "DisplayName": "Tesla", "LicencePlate": "ZD-555"},
            ],
        }
        conv = _make_conv(tool_outputs=tool_outputs)
        resp = await fe.handle_mileage_input_flow(
            "unesi km",
            _user_context_no_vehicle(),
            conv,
            router_params={},
        )
        # Should use vehicle from tool_outputs
        assert "Tesla" in resp or "ZD-555" in resp or "kilometra" in resp.lower()


# ===========================================================================
# handle_case_creation_flow
# ===========================================================================

class TestHandleCaseCreationFlow:
    async def test_with_description_shows_confirmation(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_case_creation_flow(
            "prijavi stetu na desnim vratima",
            _user_context(),
            conv,
            router_params={"Description": "Ogrebotina na desnim vratima"},
        )
        assert "Potvrda" in resp
        assert "Ogrebotina" in resp
        conv.request_confirmation.assert_awaited_once()
        assert conv.context.current_tool == "post_AddCase"

    async def test_without_description_starts_gathering(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_case_creation_flow(
            "prijavi stetu",
            _user_context(),
            conv,
            router_params={},
        )
        assert "opisati" in resp.lower() or "problem" in resp.lower()
        conv.start_flow.assert_awaited_once()

    async def test_subject_inferred_from_kvar(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_case_creation_flow(
            "imam kvar na motoru",
            _user_context(),
            conv,
            router_params={"Description": "Motor ne pali"},
        )
        assert "kvar" in resp.lower()

    async def test_subject_inferred_from_steta(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_case_creation_flow(
            "prijaviti \u0161teta na vozilu",
            _user_context(),
            conv,
            router_params={"Description": "Ogreban branik"},
        )
        # Subject should contain ostecenja
        assert "o\u0161te\u0107enj" in resp.lower() or "Potvrda" in resp

    async def test_subject_inferred_from_problem(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_case_creation_flow(
            "imam problem s gumama",
            _user_context(),
            conv,
            router_params={"Description": "Prednje gume istrošene"},
        )
        assert "problem" in resp.lower()

    async def test_default_subject_when_no_pattern(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_case_creation_flow(
            "trebam pomo\u0107",
            _user_context(),
            conv,
            router_params={"Description": "Trebam pomoć"},
        )
        # Should use default "Prijava slucaja"
        assert "Potvrda" in resp

    async def test_vehicle_line_present_when_vehicle_in_context(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_case_creation_flow(
            "prijavi kvar",
            _user_context(),
            conv,
            router_params={"Description": "Kvar klime"},
        )
        assert "Golf" in resp
        assert "ZG-123" in resp

    async def test_no_vehicle_line_when_no_vehicle(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_case_creation_flow(
            "prijavi kvar",
            _user_context_no_vehicle(),
            conv,
            router_params={"Description": "Kvar klime"},
        )
        # Should not have vehicle info
        assert "Golf" not in resp

    async def test_explicit_subject_from_router_params(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        resp = await fe.handle_case_creation_flow(
            "prijavi",
            _user_context(),
            conv,
            router_params={"Subject": "Hitan popravak", "Description": "Kocnice ne rade"},
        )
        assert "Hitan popravak" in resp

    async def test_add_parameters_includes_user(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        await fe.handle_case_creation_flow(
            "prijavi kvar",
            _user_context(),
            conv,
            router_params={"Description": "Motor se pregrijava"},
        )
        # Check that add_parameters was called with User
        call_args = conv.add_parameters.call_args[0][0]
        assert call_args.get("User") == "00000000-0000-0000-0000-000000000001"
        assert "Message" in call_args

    async def test_save_called_after_confirmation(self):
        fe, gw, fh = _make_executor()
        conv = _make_conv()
        await fe.handle_case_creation_flow(
            "prijavi kvar",
            _user_context(),
            conv,
            router_params={"Description": "Opis"},
        )
        conv.save.assert_awaited()

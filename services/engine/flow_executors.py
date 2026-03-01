"""
Flow Executors - Deterministic flow handling.

Extracted from engine/__init__.py for better modularity.
Handles booking, mileage input, and case creation flows.
"""

import logging
from typing import Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timedelta

from services.context import UserContextManager

if TYPE_CHECKING:
    from services.conversation_manager import ConversationManager
    from services.api_gateway import APIGateway
    from .flow_handler import FlowHandler

logger = logging.getLogger(__name__)


class FlowExecutors:
    """
    Handles deterministic multi-step flows.

    Supported flows:
    - Booking: Vehicle availability and reservation
    - Mileage: Mileage input for vehicles
    - Case Creation: Damage/support case reporting

    SINGLE-PASS design: Uses params from UnifiedRouter instead of
    making additional LLM calls.
    """

    def __init__(
        self,
        gateway: 'APIGateway',
        flow_handler: 'FlowHandler'
    ):
        """
        Initialize FlowExecutors.

        Args:
            gateway: API gateway for external calls
            flow_handler: Flow handler for availability handling
        """
        self.gateway = gateway
        self.flow_handler = flow_handler

    async def handle_availability_flow(
        self,
        result: Dict[str, Any],
        user_context: Dict[str, Any],
        conv_manager: 'ConversationManager'
    ) -> str:
        """Handle availability check flow."""
        flow_result = await self.flow_handler.handle_availability(
            result["tool"],
            result["parameters"],
            user_context,
            conv_manager
        )

        if flow_result.get("needs_input"):
            return flow_result["prompt"]
        if flow_result.get("final_response"):
            return flow_result["final_response"]

        return flow_result.get("error", "Greska pri provjeri dostupnosti.")

    async def handle_booking_flow(
        self,
        text: str,
        user_context: Dict[str, Any],
        conv_manager: 'ConversationManager',
        router_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle booking flow deterministically.

        SINGLE-PASS FIX: Uses router_params from UnifiedRouter instead of
        calling extract_parameters (which was a second LLM call).
        """
        # Use params from router if available (single-pass)
        router_params = router_params or {}

        params = {}
        # Check router params first (already extracted by UnifiedRouter)
        if router_params.get("from") or router_params.get("FromTime"):
            from_time = router_params.get("from") or router_params.get("FromTime")
            params["FromTime"] = from_time
            params["from"] = from_time
        if router_params.get("to") or router_params.get("ToTime"):
            to_time = router_params.get("to") or router_params.get("ToTime")
            params["ToTime"] = to_time
            params["to"] = to_time

        # Start availability flow
        result = {
            "tool": "get_AvailableVehicles",
            "parameters": params,
            "tool_call_id": "booking_flow"
        }

        return await self.handle_availability_flow(result, user_context, conv_manager)

    async def handle_mileage_input_flow(
        self,
        text: str,
        user_context: Dict[str, Any],
        conv_manager: 'ConversationManager',
        router_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle mileage input flow deterministically.

        SINGLE-PASS FIX: Uses router_params from UnifiedRouter instead of
        calling extract_parameters (which was a second LLM call).
        """
        # Use params from router if available (single-pass)
        router_params = router_params or {}
        mileage_params = {
            "Value": router_params.get("Value") or router_params.get("value") or router_params.get("mileage")
        }

        # Try to get vehicle from multiple sources - use UserContextManager
        ctx = UserContextManager(user_context)
        vehicle = ctx.vehicle
        vehicle_id = vehicle.id if vehicle else None
        vehicle_name = vehicle.name or "" if vehicle else ""
        plate = vehicle.plate or "" if vehicle else ""

        # 1. Check if we have vehicle from recent booking/context
        if not vehicle_id and hasattr(conv_manager.context, 'tool_outputs'):
            vehicle_id = conv_manager.context.tool_outputs.get("VehicleId")
            if vehicle_id:
                # Try to get name from stored vehicles
                all_vehicles = conv_manager.context.tool_outputs.get("all_available_vehicles", [])
                for v in all_vehicles:
                    if v.get("Id") == vehicle_id:
                        vehicle_name = v.get("DisplayName") or v.get("FullVehicleName") or "Vozilo"
                        plate = v.get("LicencePlate") or ""
                        break

        # 2. If still no vehicle, fetch first available one
        if not vehicle_id:
            try:
                from services.api_gateway import HttpMethod

                tomorrow = datetime.now() + timedelta(days=1)
                result = await self.gateway.execute(
                    method=HttpMethod.GET,
                    path="/vehiclemgt/AvailableVehicles",
                    params={
                        "from": tomorrow.replace(hour=8, minute=0).isoformat(),
                        "to": tomorrow.replace(hour=17, minute=0).isoformat()
                    }
                )

                if result.success and result.data:
                    data = result.data.get("Data", result.data) if isinstance(result.data, dict) else result.data
                    vehicles = data if isinstance(data, list) else [data]

                    if vehicles:
                        v = vehicles[0]
                        vehicle_id = v.get("Id")
                        vehicle_name = v.get("DisplayName") or v.get("FullVehicleName") or "Vozilo"
                        plate = v.get("LicencePlate") or ""

                        # Store for later - ONLY minimal data to prevent serialization issues
                        if hasattr(conv_manager.context, 'tool_outputs'):
                            conv_manager.context.tool_outputs["VehicleId"] = vehicle_id
                            # Store only minimal vehicle data
                            minimal_vehicles = [{
                                "Id": v.get("Id"),
                                "DisplayName": v.get("DisplayName") or v.get("FullVehicleName") or "Vozilo",
                                "LicencePlate": v.get("LicencePlate") or v.get("Plate") or ""
                            } for v in vehicles]
                            conv_manager.context.tool_outputs["all_available_vehicles"] = minimal_vehicles
            except Exception as e:
                logger.warning(f"Failed to fetch vehicles for mileage: {e}")

        if not vehicle_id:
            return (
                "Nije prona\u0111eno vozilo za unos kilometra\u017ee.\n"
                "Poku\u0161ajte prvo rezervirati vozilo ili kontaktirajte podr\u0161ku."
            )

        if not mileage_params.get("Value"):
            # Start gathering flow - store vehicle info
            await conv_manager.start_flow(
                flow_name="mileage_input",
                tool="post_AddMileage",
                required_params=["Value"]
            )
            await conv_manager.add_parameters({
                "VehicleId": vehicle_id,
                "_vehicle_name": vehicle_name,
                "_vehicle_plate": plate
            })
            await conv_manager.save()

            return (
                f"Unosim kilometra\u017eu za **{vehicle_name}** ({plate}).\n\n"
                f"Kolika je trenutna kilometra\u017ea? _(npr. '14500')_"
            )

        # Have all params - ask for confirmation
        value = mileage_params["Value"]

        await conv_manager.add_parameters({
            "VehicleId": vehicle_id,
            "Value": value
        })

        message = (
            f"**Potvrda unosa kilometra\u017ee:**\n\n"
            f"Vozilo: {vehicle_name} ({plate})\n"
            f"Kilometra\u017ea: {value} km\n\n"
            f"_Potvrdite s 'Da' ili odustanite s 'Ne'._"
        )

        await conv_manager.request_confirmation(message)
        conv_manager.context.current_tool = "post_AddMileage"
        await conv_manager.save()

        return message

    async def handle_case_creation_flow(
        self,
        text: str,
        user_context: Dict[str, Any],
        conv_manager: 'ConversationManager',
        router_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle support case/damage report creation deterministically.

        SINGLE-PASS FIX: Uses router_params from UnifiedRouter instead of
        calling extract_parameters (which was a second LLM call).
        """
        # Use params from router if available (single-pass)
        router_params = router_params or {}
        case_params = {
            "Description": router_params.get("Description") or router_params.get("description"),
            "Subject": router_params.get("Subject") or router_params.get("subject")
        }

        ctx = UserContextManager(user_context)
        vehicle = ctx.vehicle
        vehicle_id = vehicle.id if vehicle else None
        vehicle_name = vehicle.name or "vozilo" if vehicle else "vozilo"
        plate = vehicle.plate or "" if vehicle else ""

        # Build subject from text if not extracted
        subject = case_params.get("Subject")
        if not subject:
            # Try to infer subject from common patterns
            text_lower = text.lower()
            if "kvar" in text_lower:
                subject = "Prijava kvara"
            elif "\u0161teta" in text_lower or "o\u0161te\u0107en" in text_lower:
                subject = "Prijava o\u0161te\u0107enja"
            elif "problem" in text_lower:
                subject = "Prijava problema"
            else:
                subject = "Prijava slu\u010daja"

        description = case_params.get("Description")

        if not description:
            # Need to gather description
            await conv_manager.start_flow(
                flow_name="case_creation",
                tool="post_AddCase",
                required_params=["Description"]
            )

            # Store what we have so far
            params = {"Subject": subject}
            if vehicle_id:
                params["VehicleId"] = vehicle_id
            await conv_manager.add_parameters(params)
            await conv_manager.save()

            return "Mo\u017eete li opisati problem ili kvar detaljnije?"

        # Have all data - request confirmation
        # API expects: User, Subject, Message
        params = {
            "User": ctx.person_id or "",  # Required by API
            "Subject": subject,
            "Message": description  # API uses "Message", not "Description"
        }

        await conv_manager.add_parameters(params)

        vehicle_line = f"Vozilo: {vehicle_name} ({plate})\n" if vehicle_id else ""

        message = (
            f"**Potvrda prijave slu\u010daja:**\n\n"
            f"Naslov: {subject}\n"
            f"{vehicle_line}"
            f"Opis: {description}\n\n"
            f"_Potvrdite s 'Da' ili odustanite s 'Ne'._"
        )

        await conv_manager.request_confirmation(message)
        conv_manager.context.current_tool = "post_AddCase"
        await conv_manager.save()

        return message

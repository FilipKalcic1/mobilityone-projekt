"""
Flow Handler - Multi-turn conversation flows.
Version: 2.0

Single responsibility: Handle selection, confirmation, gathering, and availability flows.

v2.0 Changes:
- Integrated ConfirmationDialog for human-readable parameter display
- Added parameter modification support ("Bilješka: tekst", "Od: 10:00")
- Better Croatian formatting for dates, vehicles, etc.
"""

import logging
import re
from typing import Dict, Any, List, Optional

from services.booking_contracts import AssigneeType, EntryType
from services.error_translator import get_error_translator
from services.confirmation_dialog import get_confirmation_dialog, ConfirmationDialog

logger = logging.getLogger(__name__)


class FlowHandler:
    """
    Handles multi-turn conversation flows.

    Responsibilities:
    - Handle item selection
    - Handle confirmations
    - Handle parameter gathering
    - Handle availability checks
    """

    def __init__(self, registry, executor, ai, formatter):
        """Initialize flow handler."""
        self.registry = registry
        self.executor = executor
        self.ai = ai
        self.formatter = formatter
        self.confirmation_dialog = get_confirmation_dialog()

    async def handle_availability(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_context: Dict[str, Any],
        conv_manager
    ) -> Dict[str, Any]:
        """
        Handle vehicle availability check.

        Returns vehicles found and sets up confirmation flow.
        """
        # Check if time parameters are provided
        has_from = parameters.get("from") or parameters.get("FromTime")
        has_to = parameters.get("to") or parameters.get("ToTime")

        if not has_from or not has_to:
            # Need to gather time parameters first
            await conv_manager.start_flow(
                flow_name="booking",
                tool="get_AvailableVehicles",
                required_params=["from", "to"]
            )

            # Store any partial params we have
            if has_from or has_to:
                await conv_manager.add_parameters(parameters)

            await conv_manager.save()

            return {
                "needs_input": True,
                "prompt": (
                    "Za rezervaciju vozila trebam znati period.\n\n"
                    "Molim navedite **od kada do kada** trebate vozilo?\n"
                    "_Npr. 'sutra od 8 do 17' ili 'od ponedjeljka do srijede'_"
                )
            }

        tool = self.registry.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool {tool_name} not found",
                "final_response": f"Tehnicki problem - alat '{tool_name}' nije pronaden."
            }

        from services.tool_contracts import ToolExecutionContext
        execution_context = ToolExecutionContext(
            user_context=user_context,
            tool_outputs={},
            conversation_state={}
        )

        result = await self.executor.execute(tool, parameters, execution_context)

        if not result.success:
            return {
                "success": False,
                "data": {"error": result.error_message},
                "ai_feedback": result.ai_feedback or "Availability check failed",
                "final_response": f"Greska: {result.error_message}"
            }

        # Extract items
        items = self._extract_items(result.data)

        if not items:
            return {
                "success": True,
                "data": result.data,
                "final_response": (
                    "Nazalost, nema slobodnih vozila za odabrani period.\n\n"
                    "Mozete li odabrati drugi termin? Na primjer:\n"
                    "* 'Sutra od 8 do 17'\n"
                    "* 'Sljedeci tjedan od ponedjeljka do srijede'"
                )
            }

        # Show first vehicle - use Swagger field names only
        first_vehicle = items[0]

        vehicle_name = (
            first_vehicle.get("FullVehicleName") or
            first_vehicle.get("DisplayName") or
            "Vozilo"
        )
        plate = first_vehicle.get("LicencePlate", "N/A")
        vehicle_id = first_vehicle.get("Id")

        # CRITICAL FIX v15.1: Store ALL items for later display
        await conv_manager.set_displayed_items(items)
        await conv_manager.add_parameters(parameters)
        await conv_manager.select_item(first_vehicle)

        # RACE CONDITION FIX v2.1: Batch all tool_outputs updates into single dict
        # to prevent partial state being saved if interrupted between updates
        if hasattr(conv_manager.context, 'tool_outputs'):
            # Build minimal vehicle data first (no async operations here)
            minimal_vehicles = []
            for v in items:
                minimal_vehicles.append({
                    "Id": v.get("Id") or v.get("VehicleId"),
                    "DisplayName": v.get("DisplayName") or v.get("FullVehicleName") or v.get("Name") or "Vozilo",
                    "LicencePlate": v.get("LicencePlate") or v.get("Plate") or "N/A"
                })

            # ATOMIC UPDATE: All tool_outputs changes in one dict.update() call
            conv_manager.context.tool_outputs.update({
                "VehicleId": vehicle_id,
                "vehicleId": vehicle_id,
                "all_available_vehicles": minimal_vehicles,
                "vehicle_count": len(items)
            })

            logger.info(f"Stored {len(minimal_vehicles)} vehicles for booking flow (atomic update)")

        conv_manager.context.current_tool = "post_VehicleCalendar"

        from_time = parameters.get("from") or parameters.get("FromTime")
        to_time = parameters.get("to") or parameters.get("ToTime")

        # CRITICAL FIX v15.1: Don't show dates if not provided by user
        message = f"**Pronasao sam slobodno vozilo:**\n\n**{vehicle_name}** ({plate})\n\n"

        if from_time and to_time:
            message += f"Period: {from_time} -> {to_time}\n\n"

        if len(items) > 1:
            message += f"_(Ima jos {len(items) - 1} slobodnih vozila. Recite 'pokaži ostala' za listu)_\n\n"

        # CRITICAL FIX v15.1: Only ask for confirmation if we have time params
        if from_time and to_time:
            message += "**Zelite li potvrditi rezervaciju?** (Da/Ne)"
        else:
            message += "**Za nastavak trebam period rezervacije.** (npr. 'od sutra 9h do 17h')"

        await conv_manager.request_confirmation(message)
        await conv_manager.save()

        return {
            "success": True,
            "data": result.data,
            "needs_input": True,
            "prompt": message
        }

    async def request_confirmation(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_context: Dict[str, Any],
        conv_manager
    ) -> Dict[str, Any]:
        """
        Request confirmation for critical operation.

        v2.0: Uses ConfirmationDialog for human-readable parameter display.
        - Vehicle IDs shown as names with plates
        - Dates shown in Croatian format
        - Users can modify params with "Bilješka: tekst" syntax
        """
        await conv_manager.add_parameters(parameters)

        tool = self.registry.get_tool(tool_name)

        # Build context data for better formatting
        context_data = {}
        if hasattr(conv_manager.context, 'tool_outputs'):
            # Get selected vehicle for display
            selected = conv_manager.get_selected_item()
            if selected:
                context_data['selected_vehicle'] = selected
            # Also check tool_outputs for vehicle info
            vehicles = conv_manager.context.tool_outputs.get("all_available_vehicles", [])
            if vehicles:
                context_data['vehicle'] = vehicles[0]

        # Format parameters using ConfirmationDialog
        param_displays = self.confirmation_dialog.format_parameters(
            tool_name=tool_name,
            parameters=parameters,
            tool_definition=tool,
            context_data=context_data
        )

        # Generate the confirmation message
        message = self.confirmation_dialog.generate_confirmation_message(
            tool_name=tool_name,
            parameters=param_displays,
            operation_description=tool.description[:100] if tool and tool.description else ""
        )

        await conv_manager.request_confirmation(message)
        conv_manager.context.current_tool = tool_name
        await conv_manager.save()

        return {
            "success": True,
            "data": {},
            "needs_input": True,
            "prompt": message
        }

    async def handle_selection(
        self,
        sender: str,
        text: str,
        user_context: Dict[str, Any],
        conv_manager,
        handle_new_request_fn
    ) -> str:
        """Handle item selection."""
        selected = conv_manager.parse_item_selection(text)

        if not selected:
            return "Nisam razumio odabir.\nMolimo navedite broj (npr. '1') ili naziv."

        await conv_manager.select_item(selected)

        params = conv_manager.get_parameters()

        vehicle_name = (
            selected.get("FullVehicleName") or
            selected.get("DisplayName") or
            selected.get("Name") or
            "Vozilo"
        )
        plate = selected.get("LicencePlate") or selected.get("Plate", "N/A")

        from_time = params.get("from") or params.get("FromTime", "N/A")
        to_time = params.get("to") or params.get("ToTime", "N/A")

        message = (
            f"**Potvrdite rezervaciju:**\n\n"
            f"Vozilo: {vehicle_name} ({plate})\n"
            f"Od: {from_time}\n"
            f"Do: {to_time}\n\n"
            f"_Potvrdite s 'Da' ili odustanite s 'Ne'._"
        )

        await conv_manager.request_confirmation(message)
        conv_manager.context.current_tool = "post_VehicleCalendar"
        await conv_manager.save()

        return message

    async def handle_confirmation(
        self,
        sender: str,
        text: str,
        user_context: Dict[str, Any],
        conv_manager
    ) -> str:
        """Handle confirmation response."""
        text_lower = text.lower()

        # NEW: Detect filter commands like "pokaži Passat" or "pokaži ZG"
        filter_text = self._extract_filter_text(text)
        if filter_text and hasattr(conv_manager.context, 'tool_outputs'):
            all_vehicles = conv_manager.context.tool_outputs.get("all_available_vehicles", [])
            if all_vehicles:
                message = self.formatter.format_vehicle_list(all_vehicles, filter_text=filter_text)
                # Stay in SELECTING_ITEM state to allow selection
                await conv_manager.request_selection(message)
                await conv_manager.save()
                return message

        # CRITICAL FIX v15.1: Check if user wants to see other vehicles
        if any(keyword in text_lower for keyword in ["pokaz", "ostala", "druga", "jos vozila", "vise", "sva"]):
            # User wants to see all available vehicles
            if hasattr(conv_manager.context, 'tool_outputs'):
                all_vehicles = conv_manager.context.tool_outputs.get("all_available_vehicles", [])

                if len(all_vehicles) > 1:
                    # Use formatter with filter instructions
                    message = self.formatter.format_vehicle_list(all_vehicles)

                    # Switch to SELECTING_ITEM state
                    await conv_manager.request_selection(message)
                    await conv_manager.save()

                    return message

            return "Trenutno nema drugih dostupnih vozila."

        # NEW v2.0: Check for parameter modifications
        # Examples: "Bilješka: službeni put", "Od: 10:00", "Note: xyz"
        modification = self.confirmation_dialog.parse_modification(text)
        if modification:
            param_name, new_value = modification
            params = conv_manager.get_parameters()
            old_value = params.get(param_name)

            # Update the parameter
            params[param_name] = new_value
            await conv_manager.add_parameters({param_name: new_value})
            await conv_manager.save()

            # Generate update message
            display_name = self.confirmation_dialog.DISPLAY_NAMES.get(param_name, param_name)
            message = (
                f"✏️ **Ažurirano!**\n"
                f"{display_name}: {new_value}\n\n"
                f"Potvrdite s **Da** ili nastavite s izmjenama."
            )
            return message

        confirmation = conv_manager.parse_confirmation(text)

        if confirmation is None:
            # P1 FIX: Detect if user is asking a question mid-flow
            # Allow them to get information without losing confirmation state
            if self._is_question(text):
                logger.info(f"Mid-flow question detected: '{text[:50]}'")
                # Return special marker to let message_engine handle the question
                # The flow state is preserved so user can still confirm after
                return {"mid_flow_question": True, "question": text}

            return "Molim potvrdite s 'Da' ili odustanite s 'Ne'.\n_Ili dodajte parametar, npr: 'Bilješka: tekst'_"

        if not confirmation:
            await conv_manager.cancel()
            return "Rezervacija otkazana. Kako vam jos mogu pomoci?"

        await conv_manager.confirm()

        tool_name = conv_manager.get_current_tool()
        params = conv_manager.get_parameters()
        selected = conv_manager.get_selected_item()

        # Build booking payload
        if tool_name == "post_VehicleCalendar":
            vehicle_id = None
            if selected:
                vehicle_id = selected.get("Id") or selected.get("VehicleId")
            if not vehicle_id and hasattr(conv_manager.context, 'tool_outputs'):
                vehicle_id = conv_manager.context.tool_outputs.get("VehicleId")

            if not vehicle_id:
                await conv_manager.reset()
                return "Greska: Nije odabrano vozilo. Pokusajte ponovno."

            from_time = params.get("from") or params.get("FromTime")
            to_time = params.get("to") or params.get("ToTime")

            if not from_time or not to_time:
                await conv_manager.reset()
                return "Greska: Nedostaje vrijeme rezervacije. Pokusajte ponovno."

            params = {
                "AssignedToId": user_context.get("person_id"),
                "VehicleId": vehicle_id,
                "FromTime": from_time,
                "ToTime": to_time,
                "AssigneeType": int(AssigneeType.PERSON),  # Explicit int conversion
                "EntryType": int(EntryType.BOOKING),  # Explicit int conversion
                "Description": params.get("Description") or params.get("description")
            }
        else:
            if selected:
                vehicle_id = selected.get("Id") or selected.get("VehicleId")
                if vehicle_id:
                    params["VehicleId"] = vehicle_id

            if "from" in params and "FromTime" not in params:
                params["FromTime"] = params.pop("from")
            if "to" in params and "ToTime" not in params:
                params["ToTime"] = params.pop("to")

        # Execute
        tool = self.registry.get_tool(tool_name)
        if not tool:
            await conv_manager.reset()
            return f"Tehnicki problem - alat '{tool_name}' nije pronaden."

        # FIX: Inject EntryType and AssigneeType into user_context for booking
        # These are required context params that need default values
        booking_context = user_context.copy()
        if tool_name == "post_VehicleCalendar":
            booking_context["entrytype"] = int(EntryType.BOOKING)  # 0
            booking_context["assigneetype"] = int(AssigneeType.PERSON)  # 1

        from services.tool_contracts import ToolExecutionContext
        execution_context = ToolExecutionContext(
            user_context=booking_context,
            tool_outputs=(
                conv_manager.context.tool_outputs
                if hasattr(conv_manager.context, 'tool_outputs')
                else {}
            ),
            conversation_state={}
        )

        result = await self.executor.execute(tool, params, execution_context)

        await conv_manager.complete()
        await conv_manager.reset()

        if result.success:
            if tool_name == "post_VehicleCalendar":
                vehicle_name = ""
                if selected:
                    vehicle_name = (
                        selected.get("FullVehicleName") or
                        selected.get("DisplayName") or
                        selected.get("Name") or
                        ""
                    )
                    plate = selected.get("LicencePlate") or selected.get("Plate") or ""
                    if plate:
                        vehicle_name = f"{vehicle_name} ({plate})"

                return (
                    f"**Rezervacija uspjesna!**\n\n"
                    f"Vozilo: {vehicle_name}\n"
                    f"Period: {params.get('FromTime')} -> {params.get('ToTime')}\n\n"
                    f"Sretno na putu!"
                )

            # Case creation success message
            if tool_name == "post_AddCase":
                case_id = ""
                if isinstance(result.data, dict):
                    case_id = result.data.get("Id", "") or result.data.get("CaseId", "")

                subject = params.get("Subject", "Slučaj")
                return (
                    f"**Prijava uspješno kreirana!**\n\n"
                    f"Naslov: {subject}\n"
                    f"{'Broj slučaja: ' + str(case_id) if case_id else ''}\n\n"
                    f"Naš tim će pregledati vašu prijavu i javiti vam se.\n"
                    f"Kako vam još mogu pomoći?"
                )

            # Mileage update success message
            if tool_name == "post_AddMileage":
                value = params.get("Value", "")
                return (
                    f"**Kilometraža uspješno unesena!**\n\n"
                    f"Nova kilometraža: {value} km\n\n"
                    f"Kako vam još mogu pomoći?"
                )

            created_id = ""
            if isinstance(result.data, dict):
                created_id = result.data.get("created_id", "") or result.data.get("Id", "")

            return (
                f"**Operacija uspjesna!**\n\n"
                f"{'ID: ' + str(created_id) if created_id else ''}\n\n"
                f"Kako vam jos mogu pomoci?"
            )
        else:
            error = result.error_message or "Nepoznata greska"
            translator = get_error_translator()
            return translator.get_user_message(error, tool_name)

    # Semantički opisi parametara za AI extraction
    PARAM_DESCRIPTIONS = {
        # Mileage parameters
        "Value": "Kilometraža vozila u km (npr. 14000, 50000)",
        "mileage": "Kilometraža vozila u km",
        "kilometraža": "Udaljenost u kilometrima",
        "Mileage": "Trenutna kilometraža vozila",
        # Vehicle parameters
        "VehicleId": "ID vozila (UUID format)",
        "vehicleId": "ID vozila",
        # Time parameters
        "from": "Početno vrijeme (datum i sat, npr. 'sutra u 9:00', 'prekosutra 10:00')",
        "to": "Završno vrijeme - do kada (npr. '17:00', 'prekosutra', 'petak')",
        "FromTime": "Početno vrijeme rezervacije (npr. 'sutra u 9:00')",
        "ToTime": "Završno vrijeme - DO KADA traje rezervacija (npr. '17:00', 'prekosutra 18:00', '28.12.2025.')",
        # Case/Support parameters
        "Description": "Opis problema, kvara ili situacije",
        "description": "Tekstualni opis",
        "Subject": "Naslov ili tema slučaja (npr. 'Prijava kvara', 'Problem s vozilom')",
        "subject": "Naslov slučaja",
    }

    async def handle_gathering(
        self,
        sender: str,
        text: str,
        user_context: Dict[str, Any],
        conv_manager,
        handle_new_request_fn
    ) -> str:
        """Handle parameter gathering with smart extraction."""
        missing = conv_manager.get_missing_params()

        logger.info(f"GATHERING: missing={missing}, user_input='{text[:50]}'")

        # Build context for better extraction
        context = None
        if len(missing) == 1:
            param = missing[0]
            context = f"Bot je pitao korisnika za '{param}'. Korisnikov odgovor je vjerojatno vrijednost za taj parametar."

        # Dodaj semantički kontekst za svaki parametar
        extracted = await self.ai.extract_parameters(
            text,
            [{"name": p, "type": "string", "description": self.PARAM_DESCRIPTIONS.get(p, p)} for p in missing],
            context=context
        )

        logger.info(f"GATHERING: extracted={extracted}")

        # FALLBACK: Ako extrakcija nije uspjela i tražimo samo jedan parametar,
        # koristi cijeli tekst kao vrijednost (smart assumption)
        if len(missing) == 1 and not extracted.get(missing[0]):
            param = missing[0]
            # Za vremenske parametre, pokušaj parsirati tekst
            if param.lower() in ['totime', 'fromtime', 'to', 'from']:
                # Korisnik je vjerovatno dao vrijeme/datum
                # Prihvati ga kao vrijednost
                if text.strip() and len(text.strip()) < 50:  # Razumna duljina za datum
                    extracted[param] = text.strip()
                    logger.info(f"GATHERING FALLBACK: Using raw text '{text.strip()}' as {param}")
            # Za Value (kilometraža), pokušaj parsirati broj
            elif param.lower() in ['value', 'mileage']:
                import re
                numbers = re.findall(r'\d+', text)
                if numbers:
                    extracted[param] = int(numbers[0])
                    logger.info(f"GATHERING FALLBACK: Extracted number '{numbers[0]}' as {param}")

        await conv_manager.add_parameters(extracted)

        if conv_manager.has_all_required_params():
            logger.info("GATHERING: All params collected, continuing flow")

            # CRITICAL: Handle flow completion based on flow type
            current_flow = conv_manager.get_current_flow()
            tool_name = conv_manager.get_current_tool()
            params = conv_manager.get_parameters()

            # MILEAGE INPUT: Show confirmation
            if current_flow == "mileage_input" and tool_name == "post_AddMileage":
                return await self._show_mileage_confirmation(params, conv_manager)

            # CASE CREATION: Show confirmation
            if current_flow == "case_creation" and tool_name == "post_AddCase":
                return await self._show_case_confirmation(params, user_context, conv_manager)

            # BOOKING FLOW: Execute availability check with collected params
            if current_flow == "booking" and tool_name == "get_AvailableVehicles":
                logger.info("GATHERING: Booking flow - executing availability check")
                result = await self.handle_availability(
                    tool_name=tool_name,
                    parameters=params,
                    user_context=user_context,
                    conv_manager=conv_manager
                )
                if result.get("needs_input"):
                    return result["prompt"]
                if result.get("final_response"):
                    return result["final_response"]
                return result.get("error", "Greška pri provjeri dostupnosti.")

            # Default: let new request handler continue
            return await handle_new_request_fn(sender, text, user_context, conv_manager)

        still_missing = conv_manager.get_missing_params()
        logger.info(f"GATHERING: Still missing {still_missing}")
        return self._build_param_prompt(still_missing)

    async def _show_mileage_confirmation(self, params: Dict[str, Any], conv_manager) -> str:
        """Show mileage input confirmation."""
        vehicle_name = params.get("_vehicle_name", "Vozilo")
        plate = params.get("_vehicle_plate", "")
        value = params.get("Value") or params.get("mileage") or params.get("Mileage")

        message = (
            f"**Potvrda unosa kilometraže:**\n\n"
            f"Vozilo: {vehicle_name} ({plate})\n"
            f"Kilometraža: {value} km\n\n"
            f"_Potvrdite s 'Da' ili odustanite s 'Ne'._"
        )

        await conv_manager.request_confirmation(message)
        await conv_manager.save()

        return message

    async def _show_case_confirmation(
        self, params: Dict[str, Any], user_context: Dict[str, Any], conv_manager
    ) -> str:
        """Show case creation confirmation."""
        subject = params.get("Subject", "Prijava slučaja")
        description = params.get("Description") or params.get("Message", "")

        vehicle = user_context.get("vehicle", {})
        vehicle_name = vehicle.get("name", "")
        plate = vehicle.get("plate", "")
        vehicle_line = f"Vozilo: {vehicle_name} ({plate})\n" if vehicle_name else ""

        message = (
            f"**Potvrda prijave slučaja:**\n\n"
            f"Naslov: {subject}\n"
            f"{vehicle_line}"
            f"Opis: {description}\n\n"
            f"_Potvrdite s 'Da' ili odustanite s 'Ne'._"
        )

        await conv_manager.request_confirmation(message)
        await conv_manager.save()

        return message

    def _extract_items(self, data: Any) -> List[Dict]:
        """Extract items from API response."""
        items = []
        if isinstance(data, dict):
            items = data.get("items", [])
            if not items:
                items = data.get("Data", [])
            if not items and "data" in data:
                nested = data["data"]
                if isinstance(nested, dict):
                    items = nested.get("Data", [])
                elif isinstance(nested, list):
                    items = nested
        return items

    def _build_param_prompt(self, missing: List[str]) -> str:
        """Build prompt for missing parameters."""
        prompts = {
            # Time parameters
            "from": "Od kada vam treba? (npr. 'sutra u 9:00')",
            "to": "Do kada? (npr. 'sutra u 17:00')",
            "FromTime": "Od kada vam treba?",
            "ToTime": "Do kada?",
            # Description & Subject (Case creation)
            "Description": "Možete li opisati problem ili situaciju?",
            "description": "Opišite situaciju",
            "Subject": "Koji je naslov prijave? (npr. 'Kvar klime')",
            "subject": "Naslov slučaja",
            # Vehicle
            "VehicleId": "Koje vozilo želite?",
            "vehicleId": "Koje vozilo?",
            # Mileage parameters
            "Value": "Kolika je kilometraža? (npr. '14000 km')",
            "mileage": "Kolika je trenutna kilometraža?",
            "Mileage": "Unesite kilometražu vozila",
            "kilometraža": "Koliko kilometara ima vozilo?",
        }

        if len(missing) == 1:
            return prompts.get(missing[0], f"Trebam jos: {missing[0]}")

        lines = ["Za nastavak trebam jos informacije:"]
        for param in missing[:3]:
            lines.append(f"* {prompts.get(param, param)}")

        return "\n".join(lines)

    def _is_question(self, text: str) -> bool:
        """
        Detect if user input is a question (P1 FIX: mid-flow escape).

        Allows users to ask questions during confirmation flow
        without losing their booking state.

        Examples:
        - "Kolika je kilometraža?" → True
        - "Koja vozila imam?" → True
        - "Da" → False
        - "Bilješka: tekst" → False
        """
        text_lower = text.lower().strip()

        # Question word patterns (Croatian)
        question_patterns = [
            r'\b(koliko|kolika|koliki)\b',   # How much/many
            r'\b(koji|koja|koje|kojih)\b',    # Which
            r'\b(što|sta|sto)\b',             # What
            r'\b(gdje|di)\b',                 # Where
            r'\b(kada|kad)\b',                # When
            r'\b(zašto|zasto)\b',             # Why
            r'\b(kako)\b',                    # How
            r'\b(tko|ko)\b',                  # Who
            r'\b(ima li|postoji li)\b',       # Is there
            r'\b(mogu li|može li|moze li)\b', # Can I/Can it
            r'\b(je li|jel)\b',               # Is it
        ]

        for pattern in question_patterns:
            if re.search(pattern, text_lower):
                return True

        # Check for question mark
        if text.strip().endswith('?'):
            return True

        # Check for common question phrases
        question_phrases = [
            "reci mi", "kaži mi", "kazi mi",
            "pokaži mi", "pokazi mi",
            "daj mi info", "trebam znati",
            "što je s", "sta je s", "sto je s",
        ]

        for phrase in question_phrases:
            if phrase in text_lower:
                return True

        return False

    def _extract_filter_text(self, text: str) -> Optional[str]:
        """
        Extract filter text from user input.

        Detects patterns like:
        - "pokaži Passat" → "Passat"
        - "pokaži samo ZG" → "ZG"
        - "filtriraj Golf" → "Golf"
        - "samo Octavia" → "Octavia"

        Returns filter text or None if no filter detected.
        """
        import re

        text_lower = text.lower().strip()

        # Filter patterns (Croatian)
        patterns = [
            r'pokaži\s+(?:samo\s+)?(.+)',        # "pokaži Passat", "pokaži samo ZG"
            r'pokazi\s+(?:samo\s+)?(.+)',        # "pokazi Passat" (without č)
            r'filtriraj\s+(.+)',                  # "filtriraj Golf"
            r'samo\s+(.+)',                       # "samo Octavia"
            r'traži\s+(.+)',                      # "traži VW"
            r'trazi\s+(.+)',                      # "trazi VW" (without ž)
            r'nađi\s+(.+)',                       # "nađi Škoda"
            r'nadji\s+(.+)',                      # "nadji Skoda" (without đ)
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                filter_val = match.group(1).strip()
                # Don't treat confirmation words as filters
                if filter_val not in ['da', 'ne', 'odustani', 'potvrdi', 'ostala', 'druga', 'vozila']:
                    return filter_val

        return None

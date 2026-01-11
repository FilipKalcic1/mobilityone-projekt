"""
Confirmation Dialog Engine
Version: 1.0

Handles the confirmation flow with parameter modification support.

Features:
1. Parse user input for parameter changes ("Note: xyz", "promijeni vrijeme na 10h")
2. Format parameters in human-readable way (not raw IDs/timestamps)
3. Allow modifications before final confirmation
4. Clear state transitions

Example flow:
    User: "Rezerviraj auto sutra od 9 do 17"
    Bot:  "**Potvrda rezervacije:**
           Vozilo: VW Passat (ZG-1234-AB)
           Od: utorak, 15.1.2024. u 9:00
           Do: utorak, 15.1.2024. u 17:00
           Bilješka: (prazno)

           Želite li:
           - Potvrditi s 'Da'
           - Otkazati s 'Ne'
           - Dodati bilješku: 'Bilješka: tekst'
           - Promijeniti vrijeme: 'Od: 10:00'"

    User: "Bilješka: službeni put Zagreb"
    Bot:  "**Ažurirano!**
           Bilješka: službeni put Zagreb

           Potvrdite s 'Da' ili nastavite s izmjenama."

    User: "Da"
    Bot:  "✅ Rezervacija uspješna!"
"""

import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParameterDisplay:
    """How to display a parameter to the user."""
    name: str              # Internal name (VehicleId)
    display_name: str      # Croatian name (Vozilo)
    value: Any             # Raw value
    display_value: str     # Formatted value for display
    is_required: bool
    is_editable: bool      # Can user modify this?
    description: str       # Help text


class ConfirmationDialog:
    """
    Handles confirmation dialogs with parameter modification.

    Responsibilities:
    1. Format parameters for human-readable display
    2. Parse user modifications (Note: xyz, Od: 10:00)
    3. Track modified parameters
    4. Generate appropriate prompts
    """

    # Parameter display names (internal -> Croatian)
    DISPLAY_NAMES = {
        # Vehicle
        "VehicleId": "Vozilo",
        "vehicleId": "Vozilo",

        # Time
        "FromTime": "Od",
        "ToTime": "Do",
        "from": "Od",
        "to": "Do",

        # Mileage
        "Value": "Kilometraža",
        "Mileage": "Kilometraža",
        "mileage": "Kilometraža",

        # Common
        "Note": "Bilješka",
        "Description": "Opis",
        "description": "Opis",
        "Subject": "Naslov",
        "subject": "Naslov",

        # Person
        "AssignedToId": "Dodijeljen",
        "PersonId": "Osoba",

        # Case
        "CaseTypeId": "Vrsta slučaja",
        "Priority": "Prioritet",
    }

    # Parameters that users can modify
    EDITABLE_PARAMS = {
        "Note", "Description", "description", "Subject", "subject",
        "FromTime", "ToTime", "from", "to",
        "Value", "Mileage", "mileage",
        "Priority",
    }

    # Parameters that should NOT be shown to users
    HIDDEN_PARAMS = {
        "tenantId", "tenant_id", "TenantId",
        "auth_token", "AuthToken",
        "AssigneeType", "EntryType",
    }

    # Patterns to detect parameter modifications in user input
    MODIFICATION_PATTERNS = [
        # "Bilješka: tekst" or "Note: tekst" (with and without Croatian chars)
        (r'^(bilješka|biljesku|biljeska|note|opis|description):\s*(.+)$', 'Note'),
        # "Od: 10:00" or "od 10h"
        (r'^od:\s*(.+)$', 'FromTime'),
        (r'^od\s+(\d{1,2}[:\.]?\d{0,2})\s*h?$', 'FromTime'),
        # "Do: 17:00" or "do 17h"
        (r'^do:\s*(.+)$', 'ToTime'),
        (r'^do\s+(\d{1,2}[:\.]?\d{0,2})\s*h?$', 'ToTime'),
        # "Km: 15000" or "kilometraža: 15000"
        (r'^(km|kilometraža|kilometraza|mileage):\s*(\d+)$', 'Value'),
        # "Naslov: tekst"
        (r'^(naslov|subject):\s*(.+)$', 'Subject'),
        # "Prioritet: visok"
        (r'^prioritet:\s*(.+)$', 'Priority'),
    ]

    def __init__(self):
        """Initialize the dialog handler."""
        self._vehicle_cache: Dict[str, Dict] = {}  # VehicleId -> vehicle info

    def format_parameters(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        tool_definition: Optional[Any] = None,
        context_data: Optional[Dict] = None
    ) -> List[ParameterDisplay]:
        """
        Format parameters for human-readable display.

        Args:
            tool_name: Name of the tool being confirmed
            parameters: Raw parameter dict
            tool_definition: UnifiedToolDefinition for metadata
            context_data: Additional context (vehicle info, person info)

        Returns:
            List of ParameterDisplay objects for rendering
        """
        displays = []
        context_data = context_data or {}

        for param_name, value in parameters.items():
            # Skip hidden params
            if param_name in self.HIDDEN_PARAMS:
                continue

            # Get display name
            display_name = self.DISPLAY_NAMES.get(param_name, param_name)

            # Format the value for display
            display_value = self._format_value(param_name, value, context_data)

            # Determine if required (from tool definition)
            is_required = False
            description = ""
            if tool_definition and hasattr(tool_definition, 'parameters'):
                param_def = tool_definition.parameters.get(param_name)
                if param_def:
                    is_required = getattr(param_def, 'required', False)
                    description = getattr(param_def, 'description', '')

            # Check if editable
            is_editable = param_name in self.EDITABLE_PARAMS

            displays.append(ParameterDisplay(
                name=param_name,
                display_name=display_name,
                value=value,
                display_value=display_value,
                is_required=is_required,
                is_editable=is_editable,
                description=description
            ))

        return displays

    def _format_value(
        self,
        param_name: str,
        value: Any,
        context_data: Dict
    ) -> str:
        """Format a parameter value for human-readable display."""
        if value is None:
            return "(prazno)"

        # Vehicle ID -> Vehicle Name
        if param_name in ("VehicleId", "vehicleId"):
            vehicle = context_data.get("selected_vehicle") or context_data.get("vehicle")
            if vehicle:
                name = vehicle.get("FullVehicleName") or vehicle.get("DisplayName") or "Vozilo"
                plate = vehicle.get("LicencePlate") or ""
                return f"{name} ({plate})" if plate else name
            return str(value)[:20] + "..."

        # Person ID -> Person Name
        if param_name in ("AssignedToId", "PersonId"):
            person = context_data.get("person")
            if person:
                return person.get("DisplayName") or person.get("Name") or str(value)
            return str(value)[:20] + "..."

        # DateTime -> Croatian format
        if param_name in ("FromTime", "ToTime", "from", "to"):
            return self._format_datetime(value)

        # Mileage -> with km suffix
        if param_name in ("Value", "Mileage", "mileage"):
            try:
                km = int(value)
                return f"{km:,} km".replace(",", ".")
            except (ValueError, TypeError):
                return str(value)

        # Default: just convert to string
        if isinstance(value, str) and len(value) > 50:
            return value[:47] + "..."

        return str(value)

    def _format_datetime(self, value: Any) -> str:
        """Format datetime for display in Croatian."""
        if not value:
            return "(nije postavljeno)"

        try:
            # Handle ISO format
            if isinstance(value, str):
                if "T" in value:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                else:
                    # Might be just date or time
                    return value
            elif isinstance(value, datetime):
                dt = value
            else:
                return str(value)

            # Croatian day names
            days = ["ponedjeljak", "utorak", "srijeda", "četvrtak",
                    "petak", "subota", "nedjelja"]
            day_name = days[dt.weekday()]

            # Format: "utorak, 15.1.2024. u 9:00"
            return f"{day_name}, {dt.day}.{dt.month}.{dt.year}. u {dt.hour}:{dt.minute:02d}"

        except Exception as e:
            logger.debug(f"DateTime format error: {e}")
            return str(value)

    def parse_modification(self, user_input: str) -> Optional[Tuple[str, Any]]:
        """
        Parse user input to detect parameter modifications.

        Args:
            user_input: What the user typed

        Returns:
            Tuple of (param_name, new_value) or None if not a modification
        """
        text = user_input.strip().lower()

        for pattern, param_name in self.MODIFICATION_PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                # Get the value (last capture group)
                groups = match.groups()
                new_value = groups[-1].strip()

                # Process the value based on param type
                processed = self._process_modification_value(param_name, new_value)

                logger.info(f"Parsed modification: {param_name} = {processed}")
                return (param_name, processed)

        return None

    def _process_modification_value(self, param_name: str, value: str) -> Any:
        """Process and validate a modified value."""
        # Time values: convert "10:00" or "10" to proper format
        if param_name in ("FromTime", "ToTime"):
            # Try to parse time
            time_match = re.match(r'(\d{1,2})[:\.]?(\d{0,2})', value)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                # Return as time string for now, will be combined with date later
                return f"{hour:02d}:{minute:02d}:00"
            return value

        # Mileage: ensure integer
        if param_name == "Value":
            try:
                return int(value.replace(".", "").replace(",", ""))
            except (ValueError, TypeError, AttributeError):
                return value

        # Text fields: return as-is
        return value

    def generate_confirmation_message(
        self,
        tool_name: str,
        parameters: List[ParameterDisplay],
        operation_description: str = ""
    ) -> str:
        """
        Generate the confirmation message to show the user.

        Args:
            tool_name: Name of the tool
            parameters: Formatted parameter list
            operation_description: Optional description of what will happen

        Returns:
            Formatted message string
        """
        lines = []

        # Header
        op_type = self._get_operation_type(tool_name)
        lines.append(f"**Potvrda {op_type}:**\n")

        # Show parameters
        required = [p for p in parameters if p.is_required]
        optional = [p for p in parameters if not p.is_required]

        for param in required:
            check = "✓" if param.value else "⚠️"
            lines.append(f"• {param.display_name}: **{param.display_value}** {check}")

        if optional:
            lines.append("")
            editable = [p for p in optional if p.is_editable and p.value]
            for param in editable:
                lines.append(f"• {param.display_name}: {param.display_value}")

            # Show hints for empty editable params
            empty_editable = [p for p in optional if p.is_editable and not p.value]
            if empty_editable:
                lines.append("")
                lines.append("_Možete dodati:_")
                for param in empty_editable[:3]:  # Max 3 hints
                    lines.append(f"  • '{param.display_name}: ...'")

        # Footer with actions
        lines.append("")
        lines.append("---")
        lines.append("**Da** - potvrdi | **Ne** - odustani")

        return "\n".join(lines)

    def generate_update_message(
        self,
        param_name: str,
        old_value: str,
        new_value: str
    ) -> str:
        """Generate message after a parameter update."""
        display_name = self.DISPLAY_NAMES.get(param_name, param_name)
        return (
            f"✏️ **Ažurirano!**\n"
            f"{display_name}: {new_value}\n\n"
            f"Potvrdite s **Da** ili nastavite s izmjenama."
        )

    def _get_operation_type(self, tool_name: str) -> str:
        """Get Croatian operation type from tool name."""
        tool_lower = tool_name.lower()

        if "calendar" in tool_lower or "booking" in tool_lower:
            return "rezervacije"
        if "mileage" in tool_lower:
            return "unosa kilometraže"
        if "case" in tool_lower:
            return "prijave slučaja"
        if "document" in tool_lower:
            return "dokumenta"
        if tool_lower.startswith("post_"):
            return "kreiranja"
        if tool_lower.startswith("put_") or tool_lower.startswith("patch_"):
            return "ažuriranja"
        if tool_lower.startswith("delete_"):
            return "brisanja"

        return "operacije"


# Singleton instance
_confirmation_dialog: Optional[ConfirmationDialog] = None


def get_confirmation_dialog() -> ConfirmationDialog:
    """Get or create singleton ConfirmationDialog instance."""
    global _confirmation_dialog
    if _confirmation_dialog is None:
        _confirmation_dialog = ConfirmationDialog()
    return _confirmation_dialog

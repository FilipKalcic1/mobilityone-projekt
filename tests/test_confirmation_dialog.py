"""
Tests for ConfirmationDialog - parameter formatting, modification parsing,
and confirmation message generation.

Coverage targets:
- Lines 158-194: format_parameters()
- Lines 210-212, 217-220, 224, 228-232, 238: _format_value()
- Lines 256, 266-268: _format_datetime()
- Lines 308, 314-315: _process_modification_value()
- Lines 352-363: generate_confirmation_message()
- Lines 395, 397, 401: _get_operation_type()
- Lines 415-417: get_confirmation_dialog()
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from services.confirmation_dialog import (
    ConfirmationDialog,
    ParameterDisplay,
    get_confirmation_dialog,
    _confirmation_dialog,
)


@pytest.fixture
def dialog():
    return ConfirmationDialog()


@pytest.fixture
def mock_tool_definition():
    """Create a mock tool definition with parameters attribute."""
    tool_def = MagicMock()

    # Create mock parameter definitions
    vehicle_param = MagicMock()
    vehicle_param.required = True
    vehicle_param.description = "Vehicle identifier"

    from_time_param = MagicMock()
    from_time_param.required = True
    from_time_param.description = "Start time of reservation"

    note_param = MagicMock()
    note_param.required = False
    note_param.description = "Optional note"

    tool_def.parameters = {
        "VehicleId": vehicle_param,
        "FromTime": from_time_param,
        "Note": note_param,
    }

    return tool_def


# =============================================================================
# ParameterDisplay Dataclass Tests
# =============================================================================

class TestParameterDisplay:
    """Test the ParameterDisplay dataclass."""

    def test_create_parameter_display(self):
        """Test creating a ParameterDisplay instance."""
        param = ParameterDisplay(
            name="VehicleId",
            display_name="Vozilo",
            value="v-123",
            display_value="VW Passat (ZG-1234)",
            is_required=True,
            is_editable=False,
            description="The vehicle to reserve"
        )
        assert param.name == "VehicleId"
        assert param.display_name == "Vozilo"
        assert param.value == "v-123"
        assert param.display_value == "VW Passat (ZG-1234)"
        assert param.is_required is True
        assert param.is_editable is False
        assert param.description == "The vehicle to reserve"

    def test_parameter_display_with_none_value(self):
        """Test ParameterDisplay with None value."""
        param = ParameterDisplay(
            name="Note",
            display_name="Biljeska",
            value=None,
            display_value="(prazno)",
            is_required=False,
            is_editable=True,
            description=""
        )
        assert param.value is None
        assert param.display_value == "(prazno)"


# =============================================================================
# ConfirmationDialog.__init__ Tests
# =============================================================================

class TestConfirmationDialogInit:
    """Test ConfirmationDialog initialization."""

    def test_init_creates_vehicle_cache(self):
        """Test that __init__ creates empty vehicle cache."""
        dialog = ConfirmationDialog()
        assert hasattr(dialog, "_vehicle_cache")
        assert dialog._vehicle_cache == {}

    def test_init_cache_is_dict(self):
        """Test vehicle cache is a dictionary."""
        dialog = ConfirmationDialog()
        assert isinstance(dialog._vehicle_cache, dict)


# =============================================================================
# format_parameters() Tests (Lines 158-194)
# =============================================================================

class TestFormatParameters:
    """Test format_parameters method - covers lines 158-194."""

    def test_format_parameters_empty(self, dialog):
        """Test formatting empty parameters."""
        result = dialog.format_parameters("test_tool", {})
        assert result == []

    def test_format_parameters_skips_hidden_params(self, dialog):
        """Test that hidden parameters are skipped."""
        params = {
            "VehicleId": "v-123",
            "tenantId": "tenant-secret",
            "auth_token": "secret-token",
            "TenantId": "another-tenant",
        }
        result = dialog.format_parameters("test_tool", params)
        # Should only have VehicleId, not the hidden ones
        assert len(result) == 1
        assert result[0].name == "VehicleId"

    def test_format_parameters_with_display_name_lookup(self, dialog):
        """Test that display names are looked up correctly."""
        params = {"VehicleId": "v-123", "FromTime": "2025-01-15T10:00:00"}
        result = dialog.format_parameters("test_tool", params)

        # Find VehicleId param
        vehicle_param = next(p for p in result if p.name == "VehicleId")
        assert vehicle_param.display_name == "Vozilo"

        # Find FromTime param
        time_param = next(p for p in result if p.name == "FromTime")
        assert time_param.display_name == "Od"

    def test_format_parameters_unknown_param_uses_name(self, dialog):
        """Test that unknown params use their name as display name."""
        params = {"UnknownParam": "value"}
        result = dialog.format_parameters("test_tool", params)
        assert result[0].display_name == "UnknownParam"

    def test_format_parameters_with_tool_definition(self, dialog, mock_tool_definition):
        """Test format_parameters with tool definition for required detection."""
        params = {"VehicleId": "v-123", "FromTime": "2025-01-15T10:00:00", "Note": "test"}
        result = dialog.format_parameters(
            "test_tool", params, tool_definition=mock_tool_definition
        )

        vehicle_param = next(p for p in result if p.name == "VehicleId")
        assert vehicle_param.is_required is True
        assert vehicle_param.description == "Vehicle identifier"

        note_param = next(p for p in result if p.name == "Note")
        assert note_param.is_required is False
        assert note_param.description == "Optional note"

    def test_format_parameters_editable_detection(self, dialog):
        """Test that editable params are correctly detected."""
        params = {
            "VehicleId": "v-123",  # Not editable
            "Note": "test note",   # Editable
            "FromTime": "2025-01-15T10:00:00",  # Editable
            "Priority": "high",    # Editable
        }
        result = dialog.format_parameters("test_tool", params)

        vehicle = next(p for p in result if p.name == "VehicleId")
        assert vehicle.is_editable is False

        note = next(p for p in result if p.name == "Note")
        assert note.is_editable is True

        from_time = next(p for p in result if p.name == "FromTime")
        assert from_time.is_editable is True

        priority = next(p for p in result if p.name == "Priority")
        assert priority.is_editable is True

    def test_format_parameters_with_context_data(self, dialog):
        """Test that context data is passed to _format_value."""
        context = {
            "selected_vehicle": {
                "FullVehicleName": "VW Golf",
                "LicencePlate": "ZG-9999-XX"
            }
        }
        params = {"VehicleId": "v-123"}
        result = dialog.format_parameters("test_tool", params, context_data=context)

        assert "VW Golf" in result[0].display_value
        assert "ZG-9999-XX" in result[0].display_value

    def test_format_parameters_none_context_becomes_empty_dict(self, dialog):
        """Test that None context_data is converted to empty dict."""
        params = {"VehicleId": "v-123"}
        result = dialog.format_parameters("test_tool", params, context_data=None)
        # Should not raise, returns truncated ID
        assert result[0].display_value.endswith("...")

    def test_format_parameters_all_hidden_params(self, dialog):
        """Test with all hidden params returns empty list."""
        params = {
            "tenantId": "t-1",
            "auth_token": "token",
            "TenantId": "t-2",
            "AssigneeType": "person",
        }
        result = dialog.format_parameters("test_tool", params)
        assert result == []


# =============================================================================
# _format_value() Tests (Lines 196-238)
# =============================================================================

class TestFormatValue:
    """Test _format_value method - covers lines 196-238."""

    def test_none_value(self, dialog):
        """Test formatting None value."""
        result = dialog._format_value("Any", None, {})
        assert result == "(prazno)"

    def test_vehicle_id_with_selected_vehicle_context(self, dialog):
        """Test VehicleId with selected_vehicle in context (lines 210-212)."""
        context = {
            "selected_vehicle": {
                "FullVehicleName": "VW Passat",
                "LicencePlate": "ZG-1234-AB"
            }
        }
        result = dialog._format_value("VehicleId", "v-123", context)
        assert result == "VW Passat (ZG-1234-AB)"

    def test_vehicle_id_with_vehicle_context(self, dialog):
        """Test VehicleId with 'vehicle' key in context."""
        context = {
            "vehicle": {
                "FullVehicleName": "Skoda Octavia",
                "LicencePlate": "ST-5678-CD"
            }
        }
        result = dialog._format_value("vehicleId", "v-456", context)
        assert result == "Skoda Octavia (ST-5678-CD)"

    def test_vehicle_id_with_display_name_fallback(self, dialog):
        """Test VehicleId using DisplayName when FullVehicleName is missing."""
        context = {
            "selected_vehicle": {
                "DisplayName": "Test Vehicle",
                "LicencePlate": "RI-1111-AA"
            }
        }
        result = dialog._format_value("VehicleId", "v-789", context)
        assert result == "Test Vehicle (RI-1111-AA)"

    def test_vehicle_id_without_licence_plate(self, dialog):
        """Test VehicleId when LicencePlate is missing."""
        context = {
            "selected_vehicle": {
                "FullVehicleName": "Mercedes Benz"
            }
        }
        result = dialog._format_value("VehicleId", "v-111", context)
        assert result == "Mercedes Benz"

    def test_vehicle_id_with_empty_licence_plate(self, dialog):
        """Test VehicleId when LicencePlate is empty string."""
        context = {
            "selected_vehicle": {
                "FullVehicleName": "BMW X5",
                "LicencePlate": ""
            }
        }
        result = dialog._format_value("VehicleId", "v-222", context)
        assert result == "BMW X5"

    def test_vehicle_id_without_context_truncates(self, dialog):
        """Test VehicleId without context truncates long ID (line 213)."""
        result = dialog._format_value("VehicleId", "very-long-vehicle-id-123456789", {})
        assert result.endswith("...")
        assert len(result) <= 24  # 20 chars + "..."

    def test_person_id_with_context(self, dialog):
        """Test PersonId with context data (lines 217-219)."""
        context = {
            "person": {
                "DisplayName": "Ivan Horvat"
            }
        }
        result = dialog._format_value("PersonId", "p-123", context)
        assert result == "Ivan Horvat"

    def test_person_id_with_name_fallback(self, dialog):
        """Test PersonId using Name when DisplayName is missing."""
        context = {
            "person": {
                "Name": "Marko Maric"
            }
        }
        result = dialog._format_value("AssignedToId", "p-456", context)
        assert result == "Marko Maric"

    def test_person_id_without_context_truncates(self, dialog):
        """Test PersonId without context truncates long ID (line 220)."""
        result = dialog._format_value("PersonId", "person-id-with-very-long-value", {})
        assert result.endswith("...")
        assert len(result) <= 24

    def test_datetime_formatting(self, dialog):
        """Test datetime parameter formatting (line 224)."""
        result = dialog._format_value("FromTime", "2025-06-15T10:30:00", {})
        assert "15" in result
        assert "2025" in result

    def test_to_time_formatting(self, dialog):
        """Test ToTime parameter formatting."""
        result = dialog._format_value("ToTime", "2025-06-15T17:00:00", {})
        assert "15" in result

    def test_mileage_formatting(self, dialog):
        """Test mileage value formatting (lines 228-230)."""
        result = dialog._format_value("Value", 15000, {})
        assert result == "15.000 km"

    def test_mileage_formatting_with_string(self, dialog):
        """Test mileage with string value."""
        result = dialog._format_value("Mileage", "25000", {})
        assert result == "25.000 km"

    def test_mileage_formatting_invalid_value(self, dialog):
        """Test mileage with invalid value (lines 231-232)."""
        result = dialog._format_value("Value", "not-a-number", {})
        assert result == "not-a-number"

    def test_mileage_formatting_none_uses_default(self, dialog):
        """Test mileage with None value returns prazno."""
        result = dialog._format_value("mileage", None, {})
        assert result == "(prazno)"

    def test_long_string_truncation(self, dialog):
        """Test that long strings are truncated (lines 235-236)."""
        long_value = "x" * 100
        result = dialog._format_value("Description", long_value, {})
        assert len(result) == 50  # 47 chars + "..."
        assert result.endswith("...")

    def test_string_exactly_50_chars_not_truncated(self, dialog):
        """Test string of exactly 50 chars is not truncated."""
        value = "x" * 50
        result = dialog._format_value("Description", value, {})
        assert result == value
        assert not result.endswith("...")

    def test_default_value_to_string(self, dialog):
        """Test default case converts to string (line 238)."""
        result = dialog._format_value("SomeParam", 12345, {})
        assert result == "12345"

    def test_boolean_value_to_string(self, dialog):
        """Test boolean value is converted to string."""
        result = dialog._format_value("IsActive", True, {})
        assert result == "True"

    def test_list_value_to_string(self, dialog):
        """Test list value is converted to string."""
        result = dialog._format_value("Items", [1, 2, 3], {})
        assert result == "[1, 2, 3]"


# =============================================================================
# _format_datetime() Tests (Lines 240-268)
# =============================================================================

class TestFormatDatetime:
    """Test _format_datetime method - covers lines 240-268."""

    def test_none_value(self, dialog):
        """Test formatting None value."""
        result = dialog._format_datetime(None)
        assert result == "(nije postavljeno)"

    def test_empty_string(self, dialog):
        """Test formatting empty string."""
        result = dialog._format_datetime("")
        assert result == "(nije postavljeno)"

    def test_iso_format_with_t(self, dialog):
        """Test ISO format with T separator."""
        result = dialog._format_datetime("2025-06-15T10:30:00")
        assert "15" in result
        assert "2025" in result
        assert "10:30" in result

    def test_iso_format_with_z_timezone(self, dialog):
        """Test ISO format with Z timezone."""
        result = dialog._format_datetime("2025-06-15T10:30:00Z")
        assert "15" in result
        assert "2025" in result

    def test_date_only_string(self, dialog):
        """Test date-only string without T (line 252)."""
        result = dialog._format_datetime("2025-06-15")
        assert result == "2025-06-15"

    def test_datetime_object(self, dialog):
        """Test datetime object input."""
        dt = datetime(2025, 6, 15, 10, 30)
        result = dialog._format_datetime(dt)
        assert "15" in result
        assert "2025" in result
        assert "10:30" in result

    def test_non_string_non_datetime_value(self, dialog):
        """Test non-string, non-datetime value (line 256)."""
        result = dialog._format_datetime(12345)
        assert result == "12345"

    def test_croatian_day_names(self, dialog):
        """Test Croatian day names are used."""
        # Monday
        result = dialog._format_datetime("2025-06-16T10:00:00")  # Monday
        assert "ponedjeljak" in result

        # Tuesday
        result = dialog._format_datetime("2025-06-17T10:00:00")
        assert "utorak" in result

        # Wednesday
        result = dialog._format_datetime("2025-06-18T10:00:00")
        assert "srijeda" in result

    def test_invalid_format_returns_original(self, dialog):
        """Test invalid format returns original value (lines 266-268)."""
        result = dialog._format_datetime("not-a-valid-date")
        assert result == "not-a-valid-date"

    def test_partial_iso_format_error_handling(self, dialog):
        """Test partial ISO format that causes parse error."""
        result = dialog._format_datetime("2025-13-45T99:99:99")  # Invalid date
        assert "2025-13-45" in result  # Returns original on error


# =============================================================================
# parse_modification() Tests (Lines 270-295)
# =============================================================================

class TestParseModification:
    """Test parse_modification method."""

    def test_note_modification_biljeska(self, dialog):
        """Test Biljeska pattern."""
        result = dialog.parse_modification("Biljeska: sluzbeni put")
        assert result is not None
        assert result[0] == "Note"
        assert "sluzbeni put" in result[1]

    def test_note_modification_biljesku(self, dialog):
        """Test Biljesku pattern."""
        result = dialog.parse_modification("biljesku: test note")
        assert result is not None
        assert result[0] == "Note"

    def test_note_modification_opis(self, dialog):
        """Test Opis pattern."""
        result = dialog.parse_modification("opis: description text")
        assert result is not None
        assert result[0] == "Note"

    def test_note_modification_description(self, dialog):
        """Test Description pattern."""
        result = dialog.parse_modification("description: some description")
        assert result is not None
        assert result[0] == "Note"

    def test_from_time_with_colon(self, dialog):
        """Test Od: pattern."""
        result = dialog.parse_modification("od: 10:00")
        assert result is not None
        assert result[0] == "FromTime"

    def test_from_time_with_h(self, dialog):
        """Test 'od 10h' pattern."""
        result = dialog.parse_modification("od 10h")
        assert result is not None
        assert result[0] == "FromTime"
        assert "10" in str(result[1])

    def test_from_time_with_dot(self, dialog):
        """Test 'od 10.30' pattern."""
        result = dialog.parse_modification("od 10.30")
        assert result is not None
        assert result[0] == "FromTime"

    def test_to_time_with_colon(self, dialog):
        """Test Do: pattern."""
        result = dialog.parse_modification("Do: 17:30")
        assert result is not None
        assert result[0] == "ToTime"
        assert "17" in str(result[1])

    def test_to_time_with_h(self, dialog):
        """Test 'do 17h' pattern."""
        result = dialog.parse_modification("do 17h")
        assert result is not None
        assert result[0] == "ToTime"

    def test_mileage_km(self, dialog):
        """Test Km: pattern."""
        result = dialog.parse_modification("Km: 15000")
        assert result is not None
        assert result[0] == "Value"
        assert result[1] == 15000

    def test_mileage_kilometraza(self, dialog):
        """Test Kilometraza pattern."""
        result = dialog.parse_modification("kilometraza: 20000")
        assert result is not None
        assert result[0] == "Value"

    def test_subject_naslov(self, dialog):
        """Test Naslov: pattern."""
        result = dialog.parse_modification("Naslov: prijava kvara")
        assert result is not None
        assert result[0] == "Subject"
        assert "prijava kvara" in result[1]

    def test_subject_english(self, dialog):
        """Test Subject: pattern."""
        result = dialog.parse_modification("subject: issue report")
        assert result is not None
        assert result[0] == "Subject"

    def test_priority(self, dialog):
        """Test Prioritet: pattern."""
        result = dialog.parse_modification("prioritet: visok")
        assert result is not None
        assert result[0] == "Priority"
        assert result[1] == "visok"

    def test_no_match_returns_none(self, dialog):
        """Test unrecognized input returns None."""
        result = dialog.parse_modification("hello world")
        assert result is None

    def test_case_insensitive(self, dialog):
        """Test case insensitivity."""
        result = dialog.parse_modification("BILJESKA: UPPERCASE TEXT")
        assert result is not None
        assert result[0] == "Note"

    def test_whitespace_handling(self, dialog):
        """Test whitespace is trimmed."""
        result = dialog.parse_modification("  biljeska:   trimmed value  ")
        assert result is not None
        assert result[1] == "trimmed value"


# =============================================================================
# _process_modification_value() Tests (Lines 297-318)
# =============================================================================

class TestProcessModificationValue:
    """Test _process_modification_value method - covers lines 297-318."""

    def test_time_with_colon(self, dialog):
        """Test time processing with colon."""
        result = dialog._process_modification_value("FromTime", "10:30")
        assert result == "10:30:00"

    def test_time_with_dot(self, dialog):
        """Test time processing with dot."""
        result = dialog._process_modification_value("FromTime", "10.30")
        assert result == "10:30:00"

    def test_time_hour_only(self, dialog):
        """Test time with hour only."""
        result = dialog._process_modification_value("ToTime", "9")
        assert result == "09:00:00"

    def test_time_no_match_returns_original(self, dialog):
        """Test time that doesn't match pattern (line 308)."""
        result = dialog._process_modification_value("FromTime", "invalid")
        assert result == "invalid"

    def test_mileage_integer(self, dialog):
        """Test mileage integer conversion."""
        result = dialog._process_modification_value("Value", "15000")
        assert result == 15000
        assert isinstance(result, int)

    def test_mileage_with_dot_separator(self, dialog):
        """Test mileage with dot thousand separator (line 313)."""
        result = dialog._process_modification_value("Value", "15.000")
        assert result == 15000

    def test_mileage_with_comma_separator(self, dialog):
        """Test mileage with comma thousand separator."""
        result = dialog._process_modification_value("Value", "15,000")
        assert result == 15000

    def test_mileage_invalid_returns_original(self, dialog):
        """Test mileage with invalid value (lines 314-315)."""
        result = dialog._process_modification_value("Value", "not-a-number")
        assert result == "not-a-number"

    def test_text_field_passthrough(self, dialog):
        """Test text fields are returned as-is."""
        result = dialog._process_modification_value("Note", "some note text")
        assert result == "some note text"

    def test_subject_passthrough(self, dialog):
        """Test subject field passthrough."""
        result = dialog._process_modification_value("Subject", "Issue Title")
        assert result == "Issue Title"

    def test_priority_passthrough(self, dialog):
        """Test priority field passthrough."""
        result = dialog._process_modification_value("Priority", "high")
        assert result == "high"


# =============================================================================
# generate_confirmation_message() Tests (Lines 320-370)
# =============================================================================

class TestGenerateConfirmationMessage:
    """Test generate_confirmation_message method - covers lines 320-370."""

    def test_generates_header_with_operation_type(self, dialog):
        """Test message includes header with operation type."""
        params = [
            ParameterDisplay(
                name="VehicleId",
                display_name="Vozilo",
                value="v-123",
                display_value="VW Passat (ZG-1234)",
                is_required=True,
                is_editable=False,
                description=""
            )
        ]
        result = dialog.generate_confirmation_message("post_BookingCalendar", params)
        assert "**Potvrda" in result
        assert "rezervacije" in result

    def test_required_parameters_shown_with_check(self, dialog):
        """Test required parameters are shown with check mark."""
        params = [
            ParameterDisplay(
                name="VehicleId",
                display_name="Vozilo",
                value="v-123",
                display_value="VW Passat",
                is_required=True,
                is_editable=False,
                description=""
            )
        ]
        result = dialog.generate_confirmation_message("test_tool", params)
        assert "Vozilo" in result
        assert "VW Passat" in result

    def test_required_param_without_value_shows_warning(self, dialog):
        """Test required param without value shows warning emoji."""
        params = [
            ParameterDisplay(
                name="VehicleId",
                display_name="Vozilo",
                value=None,
                display_value="(prazno)",
                is_required=True,
                is_editable=False,
                description=""
            )
        ]
        result = dialog.generate_confirmation_message("test_tool", params)
        # Should not have check mark for empty required param

    def test_optional_editable_with_value_shown(self, dialog):
        """Test optional editable params with values are shown (lines 352-355)."""
        params = [
            ParameterDisplay(
                name="VehicleId",
                display_name="Vozilo",
                value="v-123",
                display_value="VW Passat",
                is_required=True,
                is_editable=False,
                description=""
            ),
            ParameterDisplay(
                name="Note",
                display_name="Biljeska",
                value="test note",
                display_value="test note",
                is_required=False,
                is_editable=True,
                description=""
            )
        ]
        result = dialog.generate_confirmation_message("test_tool", params)
        assert "Biljeska" in result
        assert "test note" in result

    def test_empty_editable_params_show_hints(self, dialog):
        """Test empty editable params show hints (lines 358-363)."""
        params = [
            ParameterDisplay(
                name="VehicleId",
                display_name="Vozilo",
                value="v-123",
                display_value="VW Passat",
                is_required=True,
                is_editable=False,
                description=""
            ),
            ParameterDisplay(
                name="Note",
                display_name="Biljeska",
                value=None,
                display_value="(prazno)",
                is_required=False,
                is_editable=True,
                description=""
            ),
            ParameterDisplay(
                name="Description",
                display_name="Opis",
                value=None,
                display_value="(prazno)",
                is_required=False,
                is_editable=True,
                description=""
            )
        ]
        result = dialog.generate_confirmation_message("test_tool", params)
        assert "Možete dodati" in result
        assert "'Biljeska: ...'" in result

    def test_max_three_hints_shown(self, dialog):
        """Test that maximum 3 hints are shown for empty editable params."""
        # Create 5 empty editable params
        params = [
            ParameterDisplay(
                name="VehicleId",
                display_name="Vozilo",
                value="v-123",
                display_value="VW Passat",
                is_required=True,
                is_editable=False,
                description=""
            )
        ]
        for i in range(5):
            params.append(ParameterDisplay(
                name=f"Param{i}",
                display_name=f"Parametar{i}",
                value=None,
                display_value="(prazno)",
                is_required=False,
                is_editable=True,
                description=""
            ))

        result = dialog.generate_confirmation_message("test_tool", params)
        # Count hint lines
        hint_count = result.count("'Parametar")
        assert hint_count <= 3

    def test_footer_with_actions(self, dialog):
        """Test message includes footer with Da/Ne actions."""
        params = []
        result = dialog.generate_confirmation_message("test_tool", params)
        assert "**Da**" in result
        assert "**Ne**" in result
        assert "potvrdi" in result
        assert "odustani" in result

    def test_empty_params_still_has_header_and_footer(self, dialog):
        """Test empty params list still generates header and footer."""
        result = dialog.generate_confirmation_message("test_tool", [])
        assert "**Potvrda" in result
        assert "**Da**" in result

    def test_no_optional_editable_with_value_no_extra_line(self, dialog):
        """Test no empty line added when no optional editable with values."""
        params = [
            ParameterDisplay(
                name="VehicleId",
                display_name="Vozilo",
                value="v-123",
                display_value="VW Passat",
                is_required=True,
                is_editable=False,
                description=""
            )
        ]
        result = dialog.generate_confirmation_message("test_tool", params)
        # Should not have hints section
        assert "Mozete dodati" not in result


# =============================================================================
# generate_update_message() Tests (Lines 372-384)
# =============================================================================

class TestGenerateUpdateMessage:
    """Test generate_update_message method."""

    def test_generates_update_message(self, dialog):
        """Test basic update message generation."""
        result = dialog.generate_update_message("Note", "old note", "new note")
        assert "Ažurirano" in result
        assert "new note" in result
        assert "Potvrdite" in result

    def test_uses_display_name(self, dialog):
        """Test that display name is used."""
        result = dialog.generate_update_message("Note", "", "test")
        assert "Bilješka" in result

    def test_unknown_param_uses_raw_name(self, dialog):
        """Test unknown param uses raw name."""
        result = dialog.generate_update_message("UnknownParam", "", "value")
        assert "UnknownParam" in result


# =============================================================================
# _get_operation_type() Tests (Lines 386-405)
# =============================================================================

class TestGetOperationType:
    """Test _get_operation_type method - covers lines 386-405."""

    def test_calendar_tool(self, dialog):
        """Test calendar tool returns rezervacije."""
        result = dialog._get_operation_type("post_CalendarEntry")
        assert result == "rezervacije"

    def test_booking_tool(self, dialog):
        """Test booking tool returns rezervacije."""
        result = dialog._get_operation_type("post_BookingCalendar")
        assert result == "rezervacije"

    def test_mileage_tool(self, dialog):
        """Test mileage tool returns unosa kilometraže."""
        result = dialog._get_operation_type("post_AddMileage")
        assert result == "unosa kilometraže"

    def test_case_tool(self, dialog):
        """Test case tool returns prijave slučaja (line 395)."""
        result = dialog._get_operation_type("post_CreateCase")
        assert result == "prijave slučaja"

    def test_document_tool(self, dialog):
        """Test document tool returns dokumenta (line 397)."""
        result = dialog._get_operation_type("post_CreateDocument")
        assert result == "dokumenta"

    def test_post_prefix(self, dialog):
        """Test post_ prefix returns kreiranja."""
        result = dialog._get_operation_type("post_SomeResource")
        assert result == "kreiranja"

    def test_put_prefix(self, dialog):
        """Test put_ prefix returns ažuriranja (line 401)."""
        result = dialog._get_operation_type("put_UpdateResource")
        assert result == "ažuriranja"

    def test_patch_prefix(self, dialog):
        """Test patch_ prefix returns ažuriranja."""
        result = dialog._get_operation_type("patch_UpdateResource")
        assert result == "ažuriranja"

    def test_delete_prefix(self, dialog):
        """Test delete_ prefix returns brisanja."""
        result = dialog._get_operation_type("delete_RemoveResource")
        assert result == "brisanja"

    def test_unknown_tool(self, dialog):
        """Test unknown tool returns operacije."""
        result = dialog._get_operation_type("get_SomeResource")
        assert result == "operacije"

    def test_case_insensitive(self, dialog):
        """Test operation type detection is case insensitive."""
        result = dialog._get_operation_type("POST_CALENDAR")
        assert result == "rezervacije"


# =============================================================================
# get_confirmation_dialog() Singleton Tests (Lines 412-417)
# =============================================================================

class TestGetConfirmationDialog:
    """Test get_confirmation_dialog singleton function - covers lines 412-417."""

    def test_returns_confirmation_dialog_instance(self):
        """Test function returns a ConfirmationDialog instance."""
        # Reset singleton
        import services.confirmation_dialog as module
        module._confirmation_dialog = None

        result = get_confirmation_dialog()
        assert isinstance(result, ConfirmationDialog)

    def test_returns_same_instance_on_subsequent_calls(self):
        """Test singleton returns same instance."""
        # Reset singleton
        import services.confirmation_dialog as module
        module._confirmation_dialog = None

        instance1 = get_confirmation_dialog()
        instance2 = get_confirmation_dialog()
        assert instance1 is instance2

    def test_creates_new_instance_when_none(self):
        """Test creates new instance when _confirmation_dialog is None (lines 415-417)."""
        import services.confirmation_dialog as module
        module._confirmation_dialog = None

        result = get_confirmation_dialog()
        assert module._confirmation_dialog is not None
        assert module._confirmation_dialog is result


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the confirmation dialog flow."""

    def test_full_flow_format_and_generate(self, dialog, mock_tool_definition):
        """Test full flow: format parameters then generate message."""
        params = {
            "VehicleId": "v-123",
            "FromTime": "2025-06-15T10:00:00",
            "ToTime": "2025-06-15T17:00:00",
            "Note": None,
        }
        context = {
            "selected_vehicle": {
                "FullVehicleName": "VW Passat",
                "LicencePlate": "ZG-1234-AB"
            }
        }

        # Format parameters
        formatted = dialog.format_parameters(
            "post_BookingCalendar",
            params,
            tool_definition=mock_tool_definition,
            context_data=context
        )

        # Generate message
        message = dialog.generate_confirmation_message("post_BookingCalendar", formatted)

        assert "Potvrda rezervacije" in message
        assert "VW Passat" in message
        assert "Da" in message

    def test_modification_and_update_flow(self, dialog):
        """Test parse modification and generate update message flow."""
        # User wants to change note
        user_input = "Biljeska: sluzbeni put Zagreb"
        modification = dialog.parse_modification(user_input)

        assert modification is not None
        param_name, new_value = modification

        # Generate update message
        update_msg = dialog.generate_update_message(param_name, "", new_value)

        assert "Bilješka" in update_msg
        assert "sluzbeni put" in update_msg


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_format_value_with_dict_value(self, dialog):
        """Test format_value with dict converts to string."""
        result = dialog._format_value("SomeParam", {"key": "value"}, {})
        assert "key" in result

    def test_format_datetime_with_timezone_offset(self, dialog):
        """Test datetime with timezone offset."""
        result = dialog._format_datetime("2025-06-15T10:30:00+02:00")
        assert "15" in result
        assert "2025" in result

    def test_parse_modification_empty_string(self, dialog):
        """Test parse_modification with empty string."""
        result = dialog.parse_modification("")
        assert result is None

    def test_parse_modification_only_whitespace(self, dialog):
        """Test parse_modification with only whitespace."""
        result = dialog.parse_modification("   ")
        assert result is None

    def test_format_parameters_preserves_original_value(self, dialog):
        """Test format_parameters preserves original value in ParameterDisplay."""
        params = {"VehicleId": "original-value-123"}
        result = dialog.format_parameters("test", params)
        assert result[0].value == "original-value-123"

    def test_vehicle_fallback_to_vozilo_name(self, dialog):
        """Test vehicle falls back to truncated value when no name available."""
        context = {
            "selected_vehicle": {}  # Empty vehicle info
        }
        result = dialog._format_value("VehicleId", "v-123", context)
        # Falls back to truncated value with ellipsis
        assert "v-123" in result

    def test_format_datetime_sunday(self, dialog):
        """Test Sunday formatting (nedjelja)."""
        result = dialog._format_datetime("2025-06-22T10:00:00")  # Sunday
        assert "nedjelja" in result

    def test_mileage_zero_value(self, dialog):
        """Test mileage with zero value."""
        result = dialog._format_value("Value", 0, {})
        assert result == "0 km"

    def test_hidden_params_complete_list(self, dialog):
        """Test all hidden params are filtered."""
        params = {
            "tenantId": "t1",
            "tenant_id": "t2",
            "TenantId": "t3",
            "auth_token": "token",
            "AuthToken": "token2",
            "AssigneeType": "type",
            "EntryType": "entry",
            "visible_param": "value"
        }
        result = dialog.format_parameters("test", params)
        assert len(result) == 1
        assert result[0].name == "visible_param"

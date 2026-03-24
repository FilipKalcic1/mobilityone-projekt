"""
Error Translator + Booking Contracts Tests.

Tests pattern matching, context-aware error messages,
and booking contract constants.
"""

import pytest

from services.error_translator import ErrorPattern, ErrorTranslator
from services.booking_contracts import AssigneeType, EntryType, BOOKING_FIELD_MAPPING


class TestErrorPatternMatching:
    """Test ErrorPattern regex matching and context filtering."""

    def test_matches_simple_pattern(self):
        pattern = ErrorPattern(
            pattern=r"404|not found",
            user_message_template="Not found",
            ai_feedback_template="Not found",
            category="not_found"
        )
        assert pattern.matches("HTTP 404 Not Found") is True
        assert pattern.matches("Resource not found") is True
        assert pattern.matches("Everything is fine") is False

    def test_matches_with_context_keywords(self):
        pattern = ErrorPattern(
            pattern=r"permission|403",
            user_message_template="No permission",
            ai_feedback_template="No permission",
            category="permission",
            context_keywords=["booking", "calendar"]
        )
        # Matches both pattern AND context
        assert pattern.matches("403 Forbidden", "post_BookingCalendar") is True
        # Matches pattern but NOT context
        assert pattern.matches("403 Forbidden", "get_Vehicles") is False
        # No context keywords means matches any tool
        generic = ErrorPattern(
            pattern=r"403",
            user_message_template="Forbidden",
            ai_feedback_template="Forbidden",
            category="permission",
            context_keywords=[]
        )
        assert generic.matches("403", "anything") is True

    def test_case_insensitive_matching(self):
        pattern = ErrorPattern(
            pattern=r"timeout",
            user_message_template="Timeout",
            ai_feedback_template="Timeout",
            category="timeout"
        )
        assert pattern.matches("TIMEOUT ERROR") is True
        assert pattern.matches("Connection Timeout") is True

    def test_format_user_message_with_placeholders(self):
        pattern = ErrorPattern(
            pattern=r"error",
            user_message_template="Error in {tool_name}: {error}",
            ai_feedback_template="",
            category="generic"
        )
        msg = pattern.format_user_message("Something went wrong", "get_Vehicles")
        assert "get_Vehicles" in msg
        assert "Something went wrong" in msg

    def test_format_truncates_long_errors(self):
        pattern = ErrorPattern(
            pattern=r"error",
            user_message_template="{error}",
            ai_feedback_template="",
            category="generic"
        )
        long_error = "x" * 500
        msg = pattern.format_user_message(long_error, "tool")
        assert len(msg) <= 200


class TestErrorTranslator:
    """Test the ErrorTranslator service (translate returns str, not dict)."""

    @pytest.fixture
    def translator(self):
        return ErrorTranslator()

    def test_has_default_patterns(self, translator):
        assert len(translator.patterns) > 0

    def test_translates_403_error(self, translator):
        result = translator.translate("HTTP 403 Forbidden", "post_BookingCalendar")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should match permission pattern - Croatian user message
        assert "dozvolu" in result.lower() or "permission" in result.lower()

    def test_translates_404_error(self, translator):
        result = translator.translate("Resource not found (404)", "get_Vehicles")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should match not_found pattern
        assert "pronađen" in result.lower() or "not found" in result.lower()

    def test_translates_timeout(self, translator):
        result = translator.translate("Connection timed out after 30s", "get_VehicleCalendar")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should match timeout pattern
        assert "istekao" in result.lower() or "timeout" in result.lower()

    def test_unknown_error_returns_generic(self, translator):
        result = translator.translate("Some completely unknown error XYZ123", "unknown_tool")
        # Should still return a non-empty string (generic fallback)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_validation_error_detected(self, translator):
        result = translator.translate("Validation failed: FromTime is required", "post_BookingCalendar")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should match validation pattern
        assert "podaci" in result.lower() or "validation" in result.lower()

    def test_ai_feedback_mode(self, translator):
        """for_user=False should return English AI feedback."""
        result = translator.translate("HTTP 403 Forbidden", "post_BookingCalendar", for_user=False)
        assert isinstance(result, str)
        # AI feedback is in English
        assert "permission" in result.lower() or "denied" in result.lower()

    def test_translate_includes_tool_name_in_generic(self, translator):
        """Generic fallback should include the tool name."""
        result = translator.translate("ZZZZ totally unknown ZZZZ", "get_SpecialEndpoint")
        assert "get_SpecialEndpoint" in result


class TestBookingContracts:
    """Test booking contract constants match API spec."""

    def test_assignee_types(self):
        assert AssigneeType.PERSON == 1
        assert AssigneeType.TEAM == 2

    def test_entry_types(self):
        assert EntryType.BOOKING == 0
        assert EntryType.EVENT == 1
        assert EntryType.LEAVE == 2
        assert EntryType.MAINTENANCE == 3
        assert EntryType.UNAVAILABLE == 4

    def test_booking_field_mapping_completeness(self):
        """All common booking fields should be mapped."""
        assert "assigned_to_id" in BOOKING_FIELD_MAPPING
        assert "vehicle_id" in BOOKING_FIELD_MAPPING
        assert "from_time" in BOOKING_FIELD_MAPPING
        assert "to_time" in BOOKING_FIELD_MAPPING

    def test_booking_field_mapping_values_are_api_format(self):
        """Mapped values should be PascalCase (API format)."""
        for key, value in BOOKING_FIELD_MAPPING.items():
            assert value[0].isupper(), f"API field {value} should be PascalCase"


class TestErrorTranslatorAdvanced:
    """Advanced tests for ErrorTranslator - covers lines 252-270, 309, 313, 317, 333-343, 347-370, 380-382."""

    @pytest.fixture
    def translator(self):
        return ErrorTranslator()

    def test_get_ai_feedback(self, translator):
        """Test get_ai_feedback method (line 313)."""
        result = translator.get_ai_feedback("HTTP 404 Not Found", "get_Vehicles")
        assert isinstance(result, str)
        # AI feedback in English
        assert len(result) > 0

    def test_get_user_message(self, translator):
        """Test get_user_message method (line 317)."""
        result = translator.get_user_message("HTTP 404 Not Found", "get_Vehicles")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_for_ai_returns_english(self, translator):
        """Test translate with for_user=False (line 309)."""
        result = translator.translate("Unknown error XYZ", "get_Test", for_user=False)
        assert "Error in get_Test" in result

    def test_record_resolution_adds_hint(self, translator):
        """Test record_resolution adds new hint (lines 333-338)."""
        # First make the pattern match something
        error = "HTTP 404 Not Found"
        tool = "get_Vehicles"

        # Find matching pattern and check initial hints
        for p in translator.patterns:
            if p.matches(error, tool):
                initial_hints = len(p.resolution_hints)
                break
        else:
            initial_hints = 0

        # Record a resolution
        translator.record_resolution(error, tool, "Try using a different vehicle ID")

        # Check hint was added
        for p in translator.patterns:
            if p.matches(error, tool):
                assert len(p.resolution_hints) >= initial_hints + 1
                break

    def test_record_resolution_no_duplicate_hints(self, translator):
        """Test record_resolution doesn't add duplicate hints (line 335)."""
        error = "HTTP 404 Not Found"
        tool = "get_Vehicles"
        hint = "Unique hint for testing"

        # Add hint twice
        translator.record_resolution(error, tool, hint)
        translator.record_resolution(error, tool, hint)

        # Should only appear once
        for p in translator.patterns:
            if p.matches(error, tool):
                count = p.resolution_hints.count(hint)
                assert count == 1
                break


class TestGetErrorTranslatorSingleton:
    """Test get_error_translator factory function - lines 380-382."""

    def test_returns_singleton(self):
        """Test returns same instance on multiple calls."""
        import services.error_translator as et_module
        et_module._error_translator = None  # Reset

        from services.error_translator import get_error_translator

        t1 = get_error_translator()
        t2 = get_error_translator()

        assert t1 is t2

    def test_creates_instance_if_none(self):
        """Test creates new instance if none exists."""
        import services.error_translator as et_module
        et_module._error_translator = None  # Reset

        from services.error_translator import get_error_translator

        result = get_error_translator()

        assert result is not None
        assert isinstance(result, ErrorTranslator)

"""
Tests for ResponseFormatter - formatting API responses for WhatsApp display.
"""

import pytest
from services.response_formatter import ResponseFormatter


@pytest.fixture
def formatter():
    return ResponseFormatter()


class TestHumanizeField:
    def test_camel_case(self, formatter):
        assert "Full Vehicle Name" in formatter._humanize_field("FullVehicleName")

    def test_snake_case(self, formatter):
        result = formatter._humanize_field("licence_plate")
        assert "Licence" in result or "licence" in result

    def test_single_word(self, formatter):
        result = formatter._humanize_field("Status")
        assert result == "Status"

    def test_empty_string(self, formatter):
        result = formatter._humanize_field("")
        assert result == "Polje"  # Empty field name defaults to Croatian "Polje"


class TestIsNameField:
    def test_name_field(self, formatter):
        assert formatter._is_name_field("Name") is True
        assert formatter._is_name_field("FullName") is True
        assert formatter._is_name_field("DisplayName") is True

    def test_not_name_field(self, formatter):
        assert formatter._is_name_field("Id") is False
        assert formatter._is_name_field("VehicleId") is False


class TestShouldSkipField:
    def test_skip_internal_fields(self, formatter):
        assert formatter._should_skip_field("_internal") is True

    def test_skip_id_fields(self, formatter):
        assert formatter._should_skip_field("Id") is True
        assert formatter._should_skip_field("Guid") is True

    def test_keep_visible_fields(self, formatter):
        assert formatter._should_skip_field("Name") is False
        assert formatter._should_skip_field("Status") is False
        assert formatter._should_skip_field("LicencePlate") is False


class TestDateHandling:
    def test_looks_like_iso_date(self, formatter):
        assert formatter._looks_like_date("2025-05-15") is True
        assert formatter._looks_like_date("2025-05-15T10:30:00Z") is True

    def test_looks_like_eu_date(self, formatter):
        assert formatter._looks_like_date("15.05.2025") is True

    def test_not_a_date(self, formatter):
        assert formatter._looks_like_date("hello world") is False
        assert formatter._looks_like_date("12345") is False

    def test_format_iso_date(self, formatter):
        result = formatter._format_date("2025-05-15T10:30:00Z")
        assert "15" in result
        assert "05" in result or "5" in result
        assert "2025" in result

    def test_format_plain_date(self, formatter):
        result = formatter._format_date("2025-05-15")
        assert "15" in result
        assert "2025" in result


class TestExtractOperationName:
    def test_extracts_from_post(self, formatter):
        result = formatter._extract_operation_name("post_BookingCalendar")
        assert result is not None
        assert "Booking" in result or "booking" in result.lower()

    def test_extracts_from_get(self, formatter):
        result = formatter._extract_operation_name("get_Vehicles")
        assert result is not None
        assert "vehicle" in result.lower()

    def test_empty_returns_none(self, formatter):
        result = formatter._extract_operation_name("")
        assert result is None or result == ""


class TestTruncateMessage:
    def test_short_message_unchanged(self, formatter):
        msg = "Short message"
        assert formatter._truncate_message(msg) == msg

    def test_long_message_truncated(self, formatter):
        msg = "Line\n" * 2000
        result = formatter._truncate_message(msg)
        assert len(result) <= 3600  # MAX_MESSAGE_LENGTH + buffer


class TestGetDisplayName:
    def test_full_vehicle_name(self, formatter):
        item = {"FullVehicleName": "VW Passat", "Id": "123"}
        assert "VW Passat" in formatter._get_display_name(item)

    def test_display_name(self, formatter):
        item = {"DisplayName": "Ivan Horvat"}
        assert "Ivan Horvat" in formatter._get_display_name(item)

    def test_first_last_name(self, formatter):
        item = {"FirstName": "Ivan", "LastName": "Horvat"}
        result = formatter._get_display_name(item)
        # FirstName is in name_fields priority list, so "Ivan" is returned directly
        assert "Ivan" in result

    def test_no_name_field(self, formatter):
        item = {"Id": "abc-123", "Status": "Active"}
        result = formatter._get_display_name(item)
        assert isinstance(result, str)


class TestFormatResult:
    def test_error_response(self, formatter):
        result = formatter.format_result({
            "success": False,
            "error": "Not found",
            "status_code": 404
        })
        assert isinstance(result, str)
        assert len(result) > 0

    def test_success_single_item(self, formatter):
        result = formatter.format_result({
            "success": True,
            "data": {"Name": "VW Passat", "LicencePlate": "ZG-1234-AB"},
            "status_code": 200,
            "method": "GET"
        })
        assert isinstance(result, str)
        assert "VW Passat" in result or "Passat" in result

    def test_success_list(self, formatter):
        result = formatter.format_result({
            "success": True,
            "data": [
                {"Name": "Car 1"},
                {"Name": "Car 2"},
            ],
            "status_code": 200,
            "method": "GET"
        })
        assert isinstance(result, str)

    def test_success_post(self, formatter):
        result = formatter.format_result({
            "success": True,
            "data": {"Id": "new-id-123"},
            "status_code": 201,
            "method": "POST",
            "operation": "post_BookingCalendar"
        })
        assert isinstance(result, str)

    def test_success_delete(self, formatter):
        result = formatter.format_result({
            "success": True,
            "data": None,
            "status_code": 200,
            "method": "DELETE",
            "operation": "delete_Booking"
        })
        assert isinstance(result, str)

    def test_empty_data(self, formatter):
        result = formatter.format_result({
            "success": True,
            "data": [],
            "status_code": 200,
            "method": "GET"
        })
        assert isinstance(result, str)

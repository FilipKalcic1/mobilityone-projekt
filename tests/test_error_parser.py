"""
Tests for ErrorParser - API error parsing and Croatian feedback generation.
"""

import pytest
from services.error_parser import ErrorParser


class TestParseHttpError:
    def test_400_bad_request(self):
        result = ErrorParser.parse_http_error(400, {"message": "Invalid ID"}, "post_Booking")
        assert "Neispravni parametri" in result
        assert "post_Booking" in result
        assert "Invalid ID" in result

    def test_401_unauthorized(self):
        result = ErrorParser.parse_http_error(401, None, "get_Vehicles")
        assert "Autentifikacija" in result

    def test_403_booking(self):
        result = ErrorParser.parse_http_error(403, None, "post_BookingCalendar")
        assert "dozvolu" in result or "Pristup odbijen" in result
        assert "rezervacij" in result.lower() or "BookingCalendar" in result

    def test_403_delete(self):
        result = ErrorParser.parse_http_error(403, None, "delete_Vehicle")
        assert "brisanje" in result

    def test_403_generic(self):
        result = ErrorParser.parse_http_error(403, None, "get_Vehicles")
        assert "Pristup odbijen" in result

    def test_404_not_found(self):
        result = ErrorParser.parse_http_error(404, {"error": "Not found"}, "get_Vehicle")
        assert "nije pronađen" in result

    def test_405_method_not_allowed(self):
        result = ErrorParser.parse_http_error(405, None, "post_Endpoint")
        assert "metoda" in result.lower()

    def test_422_validation(self):
        result = ErrorParser.parse_http_error(422, {"detail": "Invalid date"}, "post_Booking")
        assert "Validacija" in result

    def test_429_rate_limit(self):
        result = ErrorParser.parse_http_error(429, None, "get_Data")
        assert "Previše zahtjeva" in result

    def test_500_server_error(self):
        result = ErrorParser.parse_http_error(500, "Internal Server Error", "get_Data")
        assert "Greška servera" in result

    def test_502_server_error(self):
        result = ErrorParser.parse_http_error(502, None, "get_Data")
        assert "Greška servera" in result

    def test_unknown_status(self):
        result = ErrorParser.parse_http_error(418, None, "get_Tea")
        assert "Nepoznata greška" in result
        assert "418" in result


class TestExtractErrorMessage:
    def test_none_body(self):
        result = ErrorParser._extract_error_message(None)
        assert "Nema dodatnih" in result

    def test_string_body(self):
        result = ErrorParser._extract_error_message("Error occurred")
        assert "Error occurred" in result

    def test_string_body_truncated(self):
        long_msg = "x" * 300
        result = ErrorParser._extract_error_message(long_msg)
        assert len(result) <= 200

    def test_dict_with_message(self):
        body = {"message": "Something went wrong"}
        result = ErrorParser._extract_error_message(body)
        assert "Something went wrong" in result

    def test_dict_with_error(self):
        body = {"error": "Bad request"}
        result = ErrorParser._extract_error_message(body)
        assert "Bad request" in result

    def test_dict_with_detail(self):
        body = {"detail": "Missing field"}
        result = ErrorParser._extract_error_message(body)
        assert "Missing field" in result

    def test_dict_with_errors_dict(self):
        body = {"errors": {"Name": ["Required"], "Email": ["Invalid format"]}}
        result = ErrorParser._extract_error_message(body)
        assert "Name" in result
        assert "Required" in result

    def test_dict_fallback(self):
        body = {"unknown_key": "value"}
        result = ErrorParser._extract_error_message(body)
        assert "unknown_key" in result

    def test_empty_body(self):
        result = ErrorParser._extract_error_message("")
        assert "Nema dodatnih" in result


class TestGenerateMissingParamFeedback:
    def test_with_suggested_tools(self):
        result = ErrorParser.generate_missing_param_feedback(
            ["VehicleId", "FromTime"],
            ["get_Vehicles"]
        )
        assert "VehicleId" in result
        assert "FromTime" in result
        assert "get_Vehicles" in result

    def test_without_suggested_tools(self):
        result = ErrorParser.generate_missing_param_feedback(["VehicleId"])
        assert "VehicleId" in result
        assert "Zatraži" in result


class TestGenerateTypeErrorFeedback:
    def test_type_mismatch(self):
        result = ErrorParser.generate_type_error_feedback("Count", "integer", "abc")
        assert "Count" in result
        assert "integer" in result
        assert "abc" in result


class TestGenerateHallucinationWarning:
    def test_warning(self):
        result = ErrorParser.generate_hallucination_warning(
            "VehicleId", ["get_Vehicles", "user_context"]
        )
        assert "VehicleId" in result
        assert "UPOZORENJE" in result
        assert "get_Vehicles" in result

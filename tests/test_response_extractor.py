"""
Tests for LLMResponseExtractor - data flattening, formatting, and validation.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def extractor():
    """Create extractor with mocked OpenAI client."""
    with patch("services.response_extractor.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            AZURE_OPENAI_ENDPOINT="https://fake.openai.azure.com",
            AZURE_OPENAI_API_KEY="fake-key",
            AZURE_OPENAI_API_VERSION="2024-02-01",
        )
        with patch("services.response_extractor.get_openai_client") as mock_client, \
             patch("services.response_extractor.get_llm_circuit_breaker") as mock_cb:
            mock_client.return_value = MagicMock()
            mock_cb.return_value = MagicMock()
            from services.response_extractor import LLMResponseExtractor
            return LLMResponseExtractor()


class TestFlattenResponse:
    def test_flat_dict(self, extractor):
        data = {"Name": "Golf", "Mileage": 14000}
        result = extractor._flatten_response(data)
        assert result["Name"] == "Golf"
        assert result["Mileage"] == 14000

    def test_nested_dict(self, extractor):
        data = {"vehicle": {"mileage": 14000}}
        result = extractor._flatten_response(data)
        assert result["vehicle.mileage"] == 14000

    def test_list_of_dicts(self, extractor):
        data = [{"Name": "Golf"}, {"Name": "Passat"}]
        result = extractor._flatten_response(data)
        assert result["item[0].Name"] == "Golf"
        assert result["items.count"] == 2

    def test_nested_list_of_dicts(self, extractor):
        data = {"vehicles": [{"Name": "Golf"}, {"Name": "Passat"}]}
        result = extractor._flatten_response(data)
        assert result["vehicles[0].Name"] == "Golf"
        assert result["vehicles.count"] == 2

    def test_simple_list(self, extractor):
        data = {"tags": ["a", "b", "c"]}
        result = extractor._flatten_response(data)
        assert result["tags"] == ["a", "b", "c"]

    def test_empty_dict(self, extractor):
        result = extractor._flatten_response({})
        assert result == {}

    def test_scalar_value(self, extractor):
        result = extractor._flatten_response("hello")
        assert result["value"] == "hello"

    def test_empty_list(self, extractor):
        data = {"items": []}
        result = extractor._flatten_response(data)
        assert result["items"] == []

    def test_deeply_nested(self, extractor):
        data = {"a": {"b": {"c": "deep"}}}
        result = extractor._flatten_response(data)
        assert result["a.b.c"] == "deep"


class TestHumanizeKey:
    def test_known_key(self, extractor):
        assert extractor._humanize_key("LastMileage") == "Kilometraža"
        assert extractor._humanize_key("LicencePlate") == "Registarska oznaka"
        assert extractor._humanize_key("FullVehicleName") == "Vozilo"

    def test_nested_key_strips_prefix(self, extractor):
        assert extractor._humanize_key("vehicle.LastMileage") == "Kilometraža"

    def test_key_with_array_index(self, extractor):
        assert extractor._humanize_key("items[0].Name") == "Naziv"

    def test_unknown_key_passthrough(self, extractor):
        assert extractor._humanize_key("SomeUnknownField") == "SomeUnknownField"

    def test_known_translations(self, extractor):
        assert extractor._humanize_key("ProviderName") == "Lizing kuća"
        assert extractor._humanize_key("FromTime") == "Od"
        assert extractor._humanize_key("ToTime") == "Do"
        assert extractor._humanize_key("Status") == "Status"


class TestFormatValue:
    def test_none_value(self, extractor):
        assert extractor._format_value("any", None) == "N/A"

    def test_mileage_formatting(self, extractor):
        result = extractor._format_value("LastMileage", 14328)
        assert "14.328" in result
        assert "km" in result

    def test_mileage_float(self, extractor):
        result = extractor._format_value("Mileage", 14328.5)
        assert "km" in result

    def test_mileage_invalid(self, extractor):
        result = extractor._format_value("Mileage", "not-a-number")
        assert result == "not-a-number"

    def test_date_formatting(self, extractor):
        result = extractor._format_value("ExpirationDate", "2025-05-15T00:00:00")
        assert "15.05.2025" in result

    def test_date_without_time(self, extractor):
        result = extractor._format_value("ExpirationDate", "2025-05-15")
        assert result == "2025-05-15"

    def test_boolean_true(self, extractor):
        assert extractor._format_value("Active", True) == "Da"

    def test_boolean_false(self, extractor):
        assert extractor._format_value("Active", False) == "Ne"

    def test_short_list(self, extractor):
        result = extractor._format_value("tags", ["a", "b", "c"])
        assert "a, b, c" in result

    def test_long_list(self, extractor):
        result = extractor._format_value("items", list(range(10)))
        assert "10 stavki" in result

    def test_string_passthrough(self, extractor):
        assert extractor._format_value("Name", "Golf") == "Golf"


class TestFormatSimpleResponse:
    def test_single_field(self, extractor):
        data = {"LastMileage": 14328}
        result = extractor._format_simple_response(data, "kilometraža")
        assert "14.328" in result
        assert "km" in result

    def test_multiple_fields(self, extractor):
        data = {"Name": "Golf", "Status": "Active"}
        result = extractor._format_simple_response(data, "podaci")
        assert "Golf" in result
        assert "Active" in result

    def test_none_values_skipped(self, extractor):
        data = {"Name": "Golf", "Extra": None}
        result = extractor._format_simple_response(data, "info")
        assert "N/A" not in result

    def test_empty_data(self, extractor):
        result = extractor._format_simple_response({}, "test")
        assert "Nema podataka" in result


class TestFormatFallback:
    def test_mileage_query(self, extractor):
        data = {"LastMileage": 14328, "Name": "Golf"}
        result = extractor._format_fallback(data, "kolika mi je kilometraža")
        assert "14.328" in result or "Kilometraža" in result

    def test_plate_query(self, extractor):
        data = {"LicencePlate": "ZG-1234-AB", "Name": "Golf"}
        result = extractor._format_fallback(data, "koje su mi tablice")
        assert "ZG-1234-AB" in result

    def test_generic_fallback_shows_fields(self, extractor):
        data = {"Field1": "val1", "Field2": "val2", "Field3": "val3"}
        result = extractor._format_fallback(data, "something unknown")
        assert "val1" in result

    def test_empty_data(self, extractor):
        result = extractor._format_fallback({}, "test")
        assert "Nema podataka" in result

    def test_max_5_fields(self, extractor):
        data = {f"Field{i}": f"val{i}" for i in range(10)}
        result = extractor._format_fallback(data, "show all")
        # Should show at most 5 fields
        assert result.count("**") <= 10  # 5 fields * 2 bold markers


class TestValidateExtraction:
    def test_no_warning_for_correct_count(self, extractor):
        """No warning when vehicle count matches."""
        data = {"Data": [{"Name": "Golf"}, {"Name": "Passat"}]}
        # Should not raise
        extractor._validate_extraction("Imate 2 vozila", data, "koliko imam vozila")

    def test_plate_not_in_data(self, extractor):
        """Logs warning for plates not found in original data."""
        data = {"Name": "Golf", "LicencePlate": "ZG-1234-AB"}
        # Should not raise, just logs
        extractor._validate_extraction(
            "Tablica: ST-9999-XY", data, "koje su mi tablice"
        )

    def test_plate_in_data_no_warning(self, extractor):
        """No warning when plate matches."""
        data = {"LicencePlate": "ZG-1234-AB"}
        extractor._validate_extraction(
            "Tablica: ZG-1234-AB", data, "tablice"
        )


class TestExtractAsync:
    @pytest.mark.asyncio
    async def test_empty_response(self, extractor):
        result = await extractor.extract("test query", {})
        assert "Nema podataka" in result

    @pytest.mark.asyncio
    async def test_error_response(self, extractor):
        result = await extractor.extract("test", {"error": "Not found"})
        assert "Not found" in result

    @pytest.mark.asyncio
    async def test_simple_response_no_llm(self, extractor):
        """Simple responses (<=3 fields) bypass LLM."""
        result = await extractor.extract(
            "kilometraža",
            {"LastMileage": 14328, "Name": "Golf"}
        )
        assert isinstance(result, str)
        assert len(result) > 0

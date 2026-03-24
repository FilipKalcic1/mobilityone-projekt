"""Tests for SchemaExtractor - API response field extraction."""

import pytest
from unittest.mock import MagicMock
from services.schema_extractor import SchemaExtractor, get_schema_extractor


@pytest.fixture
def extractor():
    return SchemaExtractor()


@pytest.fixture
def extractor_with_registry():
    registry = MagicMock()
    tool = MagicMock()
    tool.output_keys = ["Id", "Name", "LicencePlate"]
    registry.get_tool.return_value = tool
    return SchemaExtractor(registry=registry)


class TestInit:
    def test_no_registry(self, extractor):
        assert extractor._registry is None

    def test_set_registry(self, extractor):
        reg = MagicMock()
        extractor.set_registry(reg)
        assert extractor._registry is reg


class TestGetOutputKeys:
    def test_no_registry_returns_empty(self, extractor):
        assert extractor.get_output_keys("get_Vehicles") == []

    def test_with_registry(self, extractor_with_registry):
        keys = extractor_with_registry.get_output_keys("get_Vehicles")
        assert "Id" in keys
        assert "Name" in keys

    def test_tool_not_found(self):
        registry = MagicMock()
        registry.get_tool.return_value = None
        ext = SchemaExtractor(registry=registry)
        assert ext.get_output_keys("nonexistent") == []


class TestNormalizeResponse:
    def test_none(self, extractor):
        assert extractor._normalize_response(None) == {}

    def test_list_passthrough(self, extractor):
        data = [{"a": 1}]
        assert extractor._normalize_response(data) == [{"a": 1}]

    def test_unwrap_Data(self, extractor):
        data = {"Data": [{"a": 1}]}
        assert extractor._normalize_response(data) == [{"a": 1}]

    def test_unwrap_Items(self, extractor):
        data = {"Items": [{"a": 1}]}
        assert extractor._normalize_response(data) == [{"a": 1}]

    def test_unwrap_data_lowercase(self, extractor):
        data = {"data": {"a": 1}}
        assert extractor._normalize_response(data) == {"a": 1}

    def test_unwrap_items_lowercase(self, extractor):
        data = {"items": [{"a": 1}]}
        assert extractor._normalize_response(data) == [{"a": 1}]

    def test_dict_passthrough(self, extractor):
        data = {"a": 1, "b": 2}
        assert extractor._normalize_response(data) == {"a": 1, "b": 2}

    def test_non_dict_non_list(self, extractor):
        assert extractor._normalize_response("string") == {}


class TestExtractFields:
    def test_extract_non_null(self, extractor):
        item = {"a": 1, "b": None, "c": "test"}
        result = extractor._extract_fields(item, [])
        assert result == {"a": 1, "c": "test"}

    def test_non_dict_returns_empty(self, extractor):
        assert extractor._extract_fields("not a dict", []) == {}


class TestExtractAll:
    def test_dict_response(self, extractor):
        data = {"Id": "123", "Name": "Test"}
        result = extractor.extract_all(data, "op")
        assert result["Id"] == "123"

    def test_list_response_returns_first(self, extractor):
        data = [{"Id": "1"}, {"Id": "2"}]
        result = extractor.extract_all(data, "op")
        assert result["Id"] == "1"

    def test_empty_list(self, extractor):
        result = extractor.extract_all([], "op")
        assert result == {}

    def test_wrapped_response(self, extractor):
        data = {"Data": [{"Id": "123"}]}
        result = extractor.extract_all(data, "op")
        assert result["Id"] == "123"


class TestExtractList:
    def test_list_response(self, extractor):
        data = [{"Id": "1"}, {"Id": "2"}]
        result = extractor.extract_list(data, "op")
        assert len(result) == 2

    def test_dict_wrapped_to_list(self, extractor):
        data = {"Id": "1"}
        result = extractor.extract_list(data, "op")
        assert len(result) == 1

    def test_empty_response(self, extractor):
        result = extractor.extract_list(None, "op")
        assert result == []


class TestGetField:
    def test_get_existing_field(self, extractor):
        data = {"Id": "123", "Name": "Test"}
        assert extractor.get_field(data, "op", "Id") == "123"

    def test_get_missing_field_default(self, extractor):
        data = {"Id": "123"}
        assert extractor.get_field(data, "op", "Missing", default="N/A") == "N/A"

    def test_get_from_list(self, extractor):
        data = [{"Id": "123"}]
        assert extractor.get_field(data, "op", "Id") == "123"

    def test_get_from_empty_list(self, extractor):
        assert extractor.get_field([], "op", "Id", default="none") == "none"

    def test_get_from_none(self, extractor):
        assert extractor.get_field(None, "op", "Id", default="x") == "x"


class TestFieldExists:
    def test_field_exists(self, extractor_with_registry):
        assert extractor_with_registry.field_exists("get_Vehicles", "Id") is True

    def test_field_not_exists(self, extractor_with_registry):
        assert extractor_with_registry.field_exists("get_Vehicles", "Unknown") is False


class TestGetCount:
    def test_list_count(self, extractor):
        assert extractor.get_count([{"a": 1}, {"a": 2}]) == 2

    def test_dict_count(self, extractor):
        assert extractor.get_count({"a": 1}) == 1

    def test_empty(self, extractor):
        assert extractor.get_count(None) == 0

    def test_wrapped_list(self, extractor):
        assert extractor.get_count({"Data": [{"a": 1}, {"b": 2}]}) == 2


class TestIsAmbiguous:
    def test_single_not_ambiguous(self, extractor):
        assert extractor.is_ambiguous({"a": 1}) is False

    def test_multiple_ambiguous(self, extractor):
        assert extractor.is_ambiguous([{"a": 1}, {"a": 2}]) is True


class TestSingleton:
    def test_get_creates_singleton(self):
        import services.schema_extractor as mod
        mod._extractor = None
        ext = get_schema_extractor()
        assert ext is not None
        mod._extractor = None

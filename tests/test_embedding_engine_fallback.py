"""
Tests for EmbeddingEngine English fallback functionality.

Tests the graceful degradation when Croatian mappings are not available.
"""

import pytest
from services.registry.embedding_engine import EmbeddingEngine


@pytest.fixture
def engine():
    return EmbeddingEngine()


class TestMakeReadable:
    """Tests for _make_readable helper function."""

    def test_camel_case_split(self, engine):
        assert engine._make_readable("vehicleInfo") == "vehicle info"

    def test_compound_word(self, engine):
        assert engine._make_readable("fuelconsumption") == "fuelconsumption"

    def test_with_numbers(self, engine):
        assert engine._make_readable("v2api") == "v 2 api"

    def test_already_lowercase(self, engine):
        assert engine._make_readable("vehicle") == "vehicle"

    def test_multiple_camel_case(self, engine):
        assert engine._make_readable("getVehicleMileageHistory") == "get vehicle mileage history"


class TestExtractEntitiesFromPathFallback:
    """Tests for English fallback in path entity extraction."""

    def test_mapped_term_uses_croatian(self, engine):
        entities = engine._extract_entities_from_path("/api/vehicles/{id}/mileage")
        assert "vozilo" in entities
        assert "kilometraža" in entities

    def test_unmapped_term_uses_english_fallback(self, engine):
        entities = engine._extract_entities_from_path("/api/unknownentity/{id}")
        # "unknownentity" is not mapped, should appear as English fallback
        assert "unknownentity" in entities or "unknown entity" in entities

    def test_skips_api_prefixes(self, engine):
        entities = engine._extract_entities_from_path("/api/v1/v2/vehicles")
        assert "api" not in entities
        assert "v1" not in entities
        assert "v2" not in entities
        assert "vozilo" in entities

    def test_mixed_mapped_and_unmapped(self, engine):
        entities = engine._extract_entities_from_path("/api/vehicles/{id}/xyzwidget")
        assert "vozilo" in entities  # Mapped
        # "xyzwidget" not mapped, should be included as fallback
        assert any("xyz" in e or "widget" in e for e in entities)

    def test_short_segments_skipped(self, engine):
        entities = engine._extract_entities_from_path("/a/b/c/vehicles")
        assert "a" not in entities
        assert "b" not in entities

    def test_empty_path(self, engine):
        entities = engine._extract_entities_from_path("")
        assert entities == []

    def test_path_with_only_parameters(self, engine):
        entities = engine._extract_entities_from_path("/{id}/{subId}")
        assert entities == []


class TestParseOperationIdFallback:
    """Tests for English fallback in operation ID parsing."""

    def test_mapped_words_use_croatian(self, engine):
        entities, _ = engine._parse_operation_id("GetVehicleMileage")
        assert "vozilo" in entities
        assert "kilometraža" in entities

    def test_unmapped_words_use_english_fallback(self, engine):
        entities, _ = engine._parse_operation_id("GetCustomWidget")
        # "custom" and "widget" are not mapped
        assert "custom" in entities or "widget" in entities

    def test_action_verbs_skipped(self, engine):
        entities, _ = engine._parse_operation_id("GetCreateUpdateDelete")
        # All action verbs should be skipped
        assert "get" not in entities
        assert "create" not in entities
        assert "update" not in entities
        assert "delete" not in entities

    def test_short_words_skipped(self, engine):
        entities, _ = engine._parse_operation_id("GetByIdForAll")
        assert "by" not in entities
        assert "id" not in entities
        assert "for" not in entities

    def test_empty_operation_id(self, engine):
        entities, action_hint = engine._parse_operation_id("")
        assert entities == []
        assert action_hint == ""

    def test_action_hint_from_output_key_map(self, engine):
        # "mileage" is in OUTPUT_KEY_MAP, but operationId splits as "Mileage"
        # The engine converts to lowercase before checking, so it should find it
        entities, action_hint = engine._parse_operation_id("GetMileage")
        # Either action_hint has the translation OR mileage is in entities
        assert action_hint != "" or "mileage" in entities or "kilometraž" in str(entities).lower()

    def test_mixed_mapped_and_unmapped_words(self, engine):
        entities, _ = engine._parse_operation_id("GetVehicleCustomAttribute")
        assert "vozilo" in entities  # Mapped
        # At least one unmapped word should be present
        assert len(entities) >= 2


class TestSkipSegments:
    """Tests for SKIP_SEGMENTS constant."""

    def test_all_common_prefixes_skipped(self, engine):
        prefixes = ["api", "v1", "v2", "v3", "odata", "rest", "public", "private"]
        for prefix in prefixes:
            entities = engine._extract_entities_from_path(f"/{prefix}/vehicles")
            assert prefix not in entities
            assert "vozilo" in entities

    def test_management_prefixes_skipped(self, engine):
        entities = engine._extract_entities_from_path("/admin/management/vehicles")
        assert "admin" not in entities
        assert "management" not in entities
        assert "vozilo" in entities


class TestBuildEmbeddingTextWithFallback:
    """Tests for build_embedding_text with fallback terms."""

    def test_embedding_text_includes_unmapped_path(self, engine):
        text = engine.build_embedding_text(
            operation_id="GetCustomWidget",
            service_name="test",
            path="/api/custom/widget",
            method="GET",
            description="",
            parameters={},
            output_keys=[]
        )
        # Should include some representation of custom/widget
        assert "custom" in text.lower() or "widget" in text.lower()

    def test_embedding_text_with_mixed_terms(self, engine):
        text = engine.build_embedding_text(
            operation_id="GetVehicleCustomData",
            service_name="test",
            path="/api/vehicles/{id}/customdata",
            method="GET",
            description="",
            parameters={},
            output_keys=["mileage", "customField"]
        )
        # Should include Croatian for mapped terms
        assert "vozil" in text.lower()
        # Should include representation of unmapped terms
        text_lower = text.lower()
        assert "custom" in text_lower or "data" in text_lower


class TestEdgeCases:
    """Edge cases for fallback functionality."""

    def test_numeric_segments(self, engine):
        entities = engine._extract_entities_from_path("/api/v1/vehicles123")
        # Should handle segments with numbers
        assert any("vozil" in e.lower() for e in entities)

    def test_hyphenated_segments(self, engine):
        entities = engine._extract_entities_from_path("/api/vehicle-mileage")
        assert "vozilo" in entities or "kilometraža" in entities

    def test_underscore_segments(self, engine):
        entities = engine._extract_entities_from_path("/api/vehicle_mileage")
        assert "vozilo" in entities or "kilometraža" in entities

    def test_very_long_operation_id(self, engine):
        long_op = "GetVeryLongOperationIdWithManyWords" * 3
        entities, _ = engine._parse_operation_id(long_op)
        # Should not crash, should return limited entities
        assert len(entities) <= 3

    def test_special_characters_in_path(self, engine):
        # Should handle gracefully
        entities = engine._extract_entities_from_path("/api/vehicles/{id}?filter=test")
        assert "vozilo" in entities

"""
Tests for TranslationHelper - Runtime translation fallback.
"""

import pytest
from services.registry.translation_helper import (
    TranslationHelper,
    TranslationResult,
    translate_term,
    suggest_mapping,
)


@pytest.fixture
def helper():
    return TranslationHelper()


class TestTranslationResult:
    def test_result_structure(self):
        result = TranslationResult(
            original="vehicle",
            translated="vozilo",
            method="dictionary",
            confidence=1.0
        )
        assert result.original == "vehicle"
        assert result.translated == "vozilo"
        assert result.method == "dictionary"
        assert result.confidence == 1.0


class TestTranslationHelper:
    def test_direct_dictionary_lookup(self, helper):
        result = helper.translate("vehicle")
        assert result.translated == "vozilo"
        assert result.method == "dictionary"
        assert result.confidence == 1.0

    def test_direct_lookup_genitive(self, helper):
        result = helper.translate("vehicle", prefer_genitive=True)
        assert result.translated == "vozila"
        assert result.method == "dictionary"

    def test_output_key_lookup(self, helper):
        result = helper.translate("mileage")
        assert "kilometra" in result.translated.lower()
        assert result.method == "dictionary"

    def test_partial_match(self, helper):
        result = helper.translate("vehiclestatus")
        # Should find "vehicle" as partial match
        assert result.confidence >= 0.5

    def test_fuzzy_match_typo(self, helper):
        result = helper.translate("vehicel")  # Typo
        # Should find "vehicle" via fuzzy matching
        assert result.method in ("fuzzy", "partial", "dictionary")
        assert result.confidence >= 0.5

    def test_fallback_unknown_term(self, helper):
        result = helper.translate("xyzunkownterm123")
        assert result.method == "fallback"
        assert result.confidence < 0.5
        # Should be formatted readably
        assert "xyzunkownterm 123" in result.translated or "xyzunkownterm123" in result.translated

    def test_camelcase_fallback(self, helper):
        result = helper.translate("unknownCamelCase")
        assert result.method == "fallback"
        assert "unknown camel case" in result.translated.lower()


class TestMakeReadable:
    def test_camelcase_split(self, helper):
        assert helper._make_readable("vehicleInfo") == "vehicle info"

    def test_number_split(self, helper):
        assert helper._make_readable("v2api") == "v 2 api"

    def test_multiple_camelcase(self, helper):
        result = helper._make_readable("getVehicleMileageHistory")
        assert result == "get vehicle mileage history"


class TestLevenshtein:
    def test_same_string(self, helper):
        assert helper._levenshtein("vehicle", "vehicle") == 0

    def test_one_character_diff(self, helper):
        assert helper._levenshtein("vehicle", "vehicel") == 2  # swap

    def test_insertion(self, helper):
        assert helper._levenshtein("vehicle", "vehicles") == 1

    def test_deletion(self, helper):
        assert helper._levenshtein("vehicles", "vehicle") == 1

    def test_completely_different(self, helper):
        assert helper._levenshtein("abc", "xyz") == 3


class TestSuggestTranslations:
    def test_suggestions_for_partial_match(self, helper):
        suggestions = helper.suggest_translations("vehiclestat")
        assert len(suggestions) > 0
        # Should suggest vehicle-related terms
        assert any("vozil" in s.translated.lower() for s in suggestions)

    def test_suggestions_sorted_by_confidence(self, helper):
        suggestions = helper.suggest_translations("mileag")
        if len(suggestions) > 1:
            confidences = [s.confidence for s in suggestions]
            assert confidences == sorted(confidences, reverse=True)

    def test_max_suggestions_limit(self, helper):
        suggestions = helper.suggest_translations("vehicle", max_suggestions=3)
        assert len(suggestions) <= 3


class TestCoverageStats:
    def test_coverage_stats_structure(self, helper):
        stats = helper.get_coverage_stats()
        assert "path_entity_map_entries" in stats
        assert "output_key_map_entries" in stats
        assert "synonym_groups" in stats
        assert "total_synonyms" in stats

    def test_coverage_stats_values(self, helper):
        stats = helper.get_coverage_stats()
        assert stats["path_entity_map_entries"] >= 200  # We have 230+
        assert stats["output_key_map_entries"] >= 150  # We have 200+
        assert stats["synonym_groups"] >= 50  # We have 60+


class TestReverseTranslation:
    def test_croatian_to_english(self, helper):
        english = helper.translate_croatian_to_english("vozilo")
        assert english is not None
        assert "vehicle" in english.lower()

    def test_croatian_genitive_to_english(self, helper):
        english = helper.translate_croatian_to_english("vozila")
        assert english is not None
        assert "vehicle" in english.lower()

    def test_unknown_croatian(self, helper):
        english = helper.translate_croatian_to_english("nepoznatorijec")
        assert english is None


class TestSynonymExpansion:
    def test_expand_with_synonyms(self, helper):
        expanded = helper.expand_with_synonyms("vozilo")
        assert "vozilo" in expanded
        # Should include synonyms like auto, automobil
        assert len(expanded) > 1

    def test_expand_from_synonym(self, helper):
        expanded = helper.expand_with_synonyms("auto")
        # Should include the root form and other synonyms
        assert len(expanded) >= 2


class TestConvenienceFunctions:
    def test_translate_term_function(self):
        result = translate_term("vehicle")
        assert result == "vozilo"

    def test_translate_term_genitive(self):
        result = translate_term("vehicle", prefer_genitive=True)
        assert result == "vozila"

    def test_suggest_mapping_function(self):
        suggestions = suggest_mapping("vehic")
        assert len(suggestions) > 0


class TestEdgeCases:
    def test_empty_string(self, helper):
        result = helper.translate("")
        assert result.method == "fallback"
        assert result.translated == ""

    def test_whitespace_handling(self, helper):
        result = helper.translate("  vehicle  ")
        assert result.translated == "vozilo"

    def test_case_insensitive(self, helper):
        result1 = helper.translate("Vehicle")
        result2 = helper.translate("VEHICLE")
        result3 = helper.translate("vehicle")
        assert result1.translated == result2.translated == result3.translated

    def test_numbers_only(self, helper):
        result = helper.translate("12345")
        assert result.method == "fallback"

    def test_special_characters(self, helper):
        result = helper.translate("vehicle-status")
        # Should handle gracefully
        assert result.method in ("fallback", "partial", "fuzzy")

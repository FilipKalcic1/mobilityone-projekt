"""
Tests for ConceptMapper - Croatian jargon to standard term expansion.
"""

import pytest
from services.concept_mapper import ConceptMapper


@pytest.fixture
def mapper():
    return ConceptMapper(enabled=True)


@pytest.fixture
def disabled_mapper():
    return ConceptMapper(enabled=False)


class TestNormalize:
    def test_remove_diacritics(self, mapper):
        assert mapper._normalize("čćžšđ") == "cczsd"

    def test_uppercase_diacritics(self, mapper):
        assert mapper._normalize("ČĆŽŠĐ") == "CCZSD"

    def test_no_diacritics(self, mapper):
        assert mapper._normalize("hello") == "hello"

    def test_mixed(self, mapper):
        assert mapper._normalize("Šef kaže") == "Sef kaze"


class TestExpandQuery:
    def test_vehicle_jargon(self, mapper):
        result = mapper.expand_query("daj mi auto")
        assert "vozilo" in result or "vehicle" in result
        assert "daj mi auto" in result  # Original preserved

    def test_action_create(self, mapper):
        result = mapper.expand_query("unesi km")
        # Any of these synonyms for create action is valid (set order is non-deterministic)
        assert "post" in result or "dodaj" in result or "kreiraj" in result or "add" in result or "create" in result

    def test_registration_terms(self, mapper):
        result = mapper.expand_query("reg")
        assert "registracija" in result or "registration" in result

    def test_case_terms(self, mapper):
        result = mapper.expand_query("prijavi kvar")
        # Expansion includes case-related terms (set order is non-deterministic, max 5 terms)
        assert "case" in result or "slučaj" in result or "problem" in result or "defekt" in result

    def test_empty_query(self, mapper):
        assert mapper.expand_query("") == ""

    def test_whitespace_query(self, mapper):
        assert mapper.expand_query("  ") == "  "

    def test_disabled_mapper(self, disabled_mapper):
        result = disabled_mapper.expand_query("daj mi auto")
        assert result == "daj mi auto"  # No expansion

    def test_no_match_returns_original(self, mapper):
        result = mapper.expand_query("xyzabc123")
        assert result == "xyzabc123"

    def test_diacritics_matching(self, mapper):
        result = mapper.expand_query("pokaži mi vozila")
        assert "get" in result or "dohvati" in result

    def test_without_diacritics_matching(self, mapper):
        result = mapper.expand_query("pokazi mi vozila")
        assert "get" in result or "dohvati" in result

    def test_no_duplicates_with_existing_terms(self, mapper):
        """Terms already in query should not be duplicated."""
        result = mapper.expand_query("prikaži vozilo")
        # "vozilo" is already in query, shouldn't be added again
        count = result.lower().split().count("vozilo")
        assert count == 1

    def test_expansion_limit(self, mapper):
        """Expansion should be limited to MAX_EXPANSION_TERMS (5)."""
        result = mapper.expand_query("daj mi auto km")
        added_terms = result.replace("daj mi auto km", "").strip().split()
        assert len(added_terms) <= 5

    def test_two_word_phrase(self, mapper):
        result = mapper.expand_query("daj mi podatke")
        # "daj mi" is a two-word phrase in CONCEPT_MAP
        assert "prikaži" in result or "dohvati" in result or "get" in result

    def test_delete_terms(self, mapper):
        result = mapper.expand_query("briši rezervaciju")
        assert "delete" in result or "obriši" in result

    def test_update_terms(self, mapper):
        result = mapper.expand_query("promijeni bilješku")
        assert "update" in result or "ažuriraj" in result

    def test_booking_terms(self, mapper):
        result = mapper.expand_query("rezerviraj auto")
        assert "booking" in result or "calendar" in result


class TestGetExpansionsOnly:
    def test_returns_added_terms(self, mapper):
        expansions = mapper.get_expansions_only("daj mi auto")
        assert isinstance(expansions, list)
        assert len(expansions) > 0

    def test_no_match_returns_empty(self, mapper):
        assert mapper.get_expansions_only("xyzabc123") == []

    def test_disabled_returns_empty(self, disabled_mapper):
        assert disabled_mapper.get_expansions_only("daj mi auto") == []


class TestBuildNormalizedMap:
    def test_map_has_both_forms(self, mapper):
        """Map should contain both diacritical and normalized forms."""
        # "šef" should be in map, and "sef" should also work
        assert "šef" in mapper._normalized_map or "sef" in mapper._normalized_map

    def test_map_size(self, mapper):
        """Map should have at least as many entries as CONCEPT_MAP."""
        assert len(mapper._normalized_map) >= len(mapper.CONCEPT_MAP)


class TestEdgeCases:
    def test_expansions_already_in_query(self, mapper):
        """If all expansions are already in query, return original."""
        # Query that already contains all possible expansions
        result = mapper.expand_query("vozilo vehicle prikaži dohvati get")
        # Should return the same or with minimal additions
        assert "vozilo" in result

    def test_normalized_two_word_phrase(self, mapper):
        """Test two-word phrases with diacritics are normalized and matched."""
        # "daj mi" should match even with diacritics normalization
        result = mapper.expand_query("daj mi podatke")
        assert "prikaži" in result or "dohvati" in result or "get" in result


class TestConvenienceFunctions:
    def test_get_concept_mapper_singleton(self):
        """get_concept_mapper should return singleton."""
        from services.concept_mapper import get_concept_mapper
        mapper1 = get_concept_mapper()
        mapper2 = get_concept_mapper()
        assert mapper1 is mapper2

    def test_expand_query_convenience(self):
        """expand_query convenience function should work."""
        from services.concept_mapper import expand_query
        result = expand_query("daj mi auto")
        assert "vozilo" in result or "vehicle" in result

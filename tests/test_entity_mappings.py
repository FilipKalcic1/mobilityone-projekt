"""
Tests for services.registry.entity_mappings — dictionary integrity.

Validates that the Croatian translation dictionaries are well-formed
and contain expected entries for key domains.
"""

import unittest

from services.registry.entity_mappings import (
    PATH_ENTITY_MAP,
    OUTPUT_KEY_MAP,
    CROATIAN_SYNONYMS,
    SKIP_SEGMENTS,
)


class TestPathEntityMap(unittest.TestCase):
    """Validate PATH_ENTITY_MAP structure and content."""

    def test_is_dict(self):
        self.assertIsInstance(PATH_ENTITY_MAP, dict)

    def test_has_minimum_entries(self):
        self.assertGreaterEqual(len(PATH_ENTITY_MAP), 300)

    def test_values_are_tuples_of_two_strings(self):
        for key, value in PATH_ENTITY_MAP.items():
            self.assertIsInstance(key, str, f"Key {key!r} is not a string")
            self.assertIsInstance(value, tuple, f"Value for {key!r} is not a tuple")
            self.assertEqual(len(value), 2, f"Value for {key!r} has {len(value)} elements, expected 2")
            self.assertIsInstance(value[0], str, f"First element for {key!r} is not a string")
            self.assertIsInstance(value[1], str, f"Second element for {key!r} is not a string")

    def test_contains_core_entities(self):
        core = ["vehicle", "person", "booking", "driver", "location"]
        for entity in core:
            self.assertIn(entity, PATH_ENTITY_MAP, f"Missing core entity: {entity}")

    def test_keys_are_lowercase(self):
        for key in PATH_ENTITY_MAP:
            self.assertEqual(key, key.lower(), f"Key {key!r} is not lowercase")


class TestOutputKeyMap(unittest.TestCase):
    """Validate OUTPUT_KEY_MAP structure."""

    def test_is_dict(self):
        self.assertIsInstance(OUTPUT_KEY_MAP, dict)

    def test_has_minimum_entries(self):
        self.assertGreaterEqual(len(OUTPUT_KEY_MAP), 200)

    def test_values_are_strings(self):
        for key, value in OUTPUT_KEY_MAP.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, str, f"Value for {key!r} is not a string")

    def test_contains_common_fields(self):
        common = ["mileage", "name", "status", "date"]
        for field in common:
            self.assertIn(field, OUTPUT_KEY_MAP, f"Missing common field: {field}")


class TestCroatianSynonyms(unittest.TestCase):
    """Validate CROATIAN_SYNONYMS structure."""

    def test_is_dict(self):
        self.assertIsInstance(CROATIAN_SYNONYMS, dict)

    def test_has_minimum_entries(self):
        self.assertGreaterEqual(len(CROATIAN_SYNONYMS), 60)

    def test_values_are_lists_of_strings(self):
        for key, value in CROATIAN_SYNONYMS.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, list, f"Value for {key!r} is not a list")
            for item in value:
                self.assertIsInstance(item, str, f"Synonym item for {key!r} is not a string")

    def test_contains_core_roots(self):
        core = ["vozil", "osob", "rezervacij"]
        for root in core:
            self.assertIn(root, CROATIAN_SYNONYMS, f"Missing core synonym root: {root}")


class TestSkipSegments(unittest.TestCase):
    """Validate SKIP_SEGMENTS structure."""

    def test_is_set(self):
        self.assertIsInstance(SKIP_SEGMENTS, set)

    def test_contains_api_versions(self):
        for version in ["v1", "v2", "api"]:
            self.assertIn(version, SKIP_SEGMENTS, f"Missing skip segment: {version}")

    def test_all_lowercase(self):
        for segment in SKIP_SEGMENTS:
            self.assertEqual(segment, segment.lower())


class TestBackwardCompatibility(unittest.TestCase):
    """Verify that EmbeddingEngine can still access these as class attributes."""

    def test_embedding_engine_imports(self):
        from services.registry.embedding_engine import EmbeddingEngine
        engine = EmbeddingEngine()
        self.assertIs(engine.PATH_ENTITY_MAP, PATH_ENTITY_MAP)
        self.assertIs(engine.OUTPUT_KEY_MAP, OUTPUT_KEY_MAP)
        self.assertIs(engine.CROATIAN_SYNONYMS, CROATIAN_SYNONYMS)
        self.assertIs(engine.SKIP_SEGMENTS, SKIP_SEGMENTS)


if __name__ == "__main__":
    unittest.main()

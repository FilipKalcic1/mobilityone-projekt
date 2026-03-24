"""
Tests for scoring_utils.py and filter_builder.py - pure utility functions.
"""

import pytest
from services.scoring_utils import cosine_similarity
from services.filter_builder import FilterBuilder
from services.tool_contracts import UnifiedToolDefinition, ParameterDefinition


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=0.01)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=0.01)

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        result = cosine_similarity(a, b)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_single_element(self):
        assert cosine_similarity([3.0], [3.0]) == pytest.approx(1.0, abs=0.01)

    def test_high_dimensional(self):
        a = [1.0] * 100
        b = [1.0] * 100
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=0.01)

    def test_empty_vectors(self):
        """Empty vectors should return 0.0."""
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([], [1.0]) == 0.0
        assert cosine_similarity([1.0], []) == 0.0

    def test_different_length_vectors(self):
        """Vectors of different lengths should return 0.0."""
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


class TestFilterBuilderSanitize:
    def test_clean_value(self):
        assert FilterBuilder._sanitize_value("John") == "John"

    def test_sql_injection_stripped(self):
        result = FilterBuilder._sanitize_value("'; DROP TABLE users--")
        assert "DROP" not in result
        assert ";" not in result
        assert "--" not in result

    def test_or_injection(self):
        result = FilterBuilder._sanitize_value("1 OR 1=1")
        assert "OR" not in result.upper() or "1=1" not in result

    def test_parentheses_stripped(self):
        result = FilterBuilder._sanitize_value("test()")
        assert "(" not in result
        assert ")" not in result

    def test_long_value_truncated(self):
        long_value = "x" * 600
        result = FilterBuilder._sanitize_value(long_value)
        assert len(result) <= 500

    def test_empty_string(self):
        assert FilterBuilder._sanitize_value("") == ""

    def test_non_string_value_converted(self):
        """Non-string values should be converted to string."""
        assert FilterBuilder._sanitize_value(12345) == "12345"
        assert FilterBuilder._sanitize_value(3.14) == "3.14"
        assert FilterBuilder._sanitize_value(True) == "True"


class TestFilterBuilderBuild:
    def _make_tool(self, params):
        """Helper to create a tool definition with given parameters."""
        param_defs = {}
        for name, filterable in params.items():
            param_defs[name] = ParameterDefinition(
                name=name,
                param_type="string",
                required=False,
                description=f"Filter by {name}",
                is_filterable=filterable,
                preferred_operator="(=)",
            )
        return UnifiedToolDefinition(
            operation_id="get_TestEndpoint",
            method="GET",
            path="/api/test",
            description="Test endpoint",
            parameters=param_defs,
            service_name="test_service",
            service_url="https://api.example.com",
            swagger_name="test",
        )

    def test_builds_filter_for_filterable_params(self):
        tool = self._make_tool({"Phone": True, "Name": True})
        result = FilterBuilder.build_filter_string(tool, {"Phone": "123456", "Name": "John"})
        assert result is not None
        assert "Phone" in result
        assert "123456" in result

    def test_no_filter_for_non_filterable(self):
        tool = self._make_tool({"Phone": False})
        result = FilterBuilder.build_filter_string(tool, {"Phone": "123456"})
        assert result is None or result == ""

    def test_empty_params(self):
        tool = self._make_tool({"Phone": True})
        result = FilterBuilder.build_filter_string(tool, {})
        assert result is None or result == ""

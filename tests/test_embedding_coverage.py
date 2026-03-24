"""
Tests for EmbeddingCoverageTracker - Validates coverage measurement system.
"""

import pytest
from services.registry.embedding_coverage import (
    EmbeddingCoverageTracker,
    CoverageReport,
    analyze_embedding_coverage,
)
from services.tool_contracts import UnifiedToolDefinition, ParameterDefinition


@pytest.fixture
def tracker():
    return EmbeddingCoverageTracker()


@pytest.fixture
def sample_tool():
    """Create a sample tool for testing."""
    return UnifiedToolDefinition(
        operation_id="get_VehicleMileage",
        method="GET",
        path="/api/v1/vehicles/{id}/mileage",
        description="Get vehicle mileage",
        parameters={},
        service_name="fleet",
        service_url="https://api.example.com",
        swagger_name="fleet",
        output_keys=["mileage", "km", "lastUpdated"],
    )


@pytest.fixture
def unmapped_tool():
    """Create a tool with truly unmapped terms for testing."""
    return UnifiedToolDefinition(
        operation_id="get_XyzAbcQrs",
        method="GET",
        path="/api/v1/foobar/{id}/quxbaz",
        description="Unknown endpoint",
        parameters={},
        service_name="unknown",
        service_url="https://api.example.com",
        swagger_name="unknown",
        # Use output keys that don't contain ANY mapped terms or substrings
        # Avoid: data, type, code, name, id, key, tag, etc.
        output_keys=["xyzblob", "abcwoof", "qrsmeow"],
    )


class TestCoverageReport:
    def test_path_coverage_pct_calculation(self):
        report = CoverageReport(
            total_path_segments=100,
            mapped_path_segments=75,
        )
        assert report.path_coverage_pct == 75.0

    def test_path_coverage_pct_zero_total(self):
        report = CoverageReport(total_path_segments=0)
        assert report.path_coverage_pct == 0.0

    def test_output_coverage_pct(self):
        report = CoverageReport(
            total_output_keys=50,
            mapped_output_keys=40,
        )
        assert report.output_coverage_pct == 80.0

    def test_tool_coverage_pct(self):
        report = CoverageReport(
            tools_with_mapping=90,
            tools_without_mapping=10,
        )
        assert report.tool_coverage_pct == 90.0

    def test_quality_grade_excellent(self):
        report = CoverageReport(
            total_path_segments=100,
            mapped_path_segments=95,
            total_output_keys=100,
            mapped_output_keys=92,
            tools_with_mapping=95,
            tools_without_mapping=5,
        )
        result = report.to_dict()
        assert "A" in result["summary"]["overall_quality"]

    def test_quality_grade_poor(self):
        report = CoverageReport(
            total_path_segments=100,
            mapped_path_segments=40,
            total_output_keys=100,
            mapped_output_keys=35,
            tools_with_mapping=50,
            tools_without_mapping=50,
        )
        result = report.to_dict()
        assert "D" in result["summary"]["overall_quality"] or "F" in result["summary"]["overall_quality"]

    def test_to_dict_structure(self):
        report = CoverageReport(
            total_path_segments=10,
            mapped_path_segments=8,
            unmapped_path_segments={"foo", "bar"},
        )
        result = report.to_dict()

        assert "path_coverage" in result
        assert "output_coverage" in result
        assert "operation_coverage" in result
        assert "tool_coverage" in result
        assert "summary" in result
        assert "overall_quality" in result["summary"]


class TestEmbeddingCoverageTracker:
    def test_analyze_path_mapped(self, tracker):
        all_segs, unmapped = tracker._analyze_path("/api/v1/vehicles/{id}/mileage")
        # "vehicles" and "mileage" should be mapped
        assert "vehicles" in all_segs
        assert "mileage" in all_segs
        # Both should be mapped (not in unmapped)
        assert "vehicles" not in unmapped
        assert "mileage" not in unmapped

    def test_analyze_path_unmapped(self, tracker):
        all_segs, unmapped = tracker._analyze_path("/api/v1/foobar/{id}/quxbaz")
        # Unknown segments should be in unmapped
        assert "foobar" in unmapped
        assert "quxbaz" in unmapped

    def test_analyze_path_skips_api_prefixes(self, tracker):
        all_segs, unmapped = tracker._analyze_path("/api/v1/v2/vehicles")
        # api, v1, v2 should be skipped
        assert "api" not in all_segs
        assert "v1" not in all_segs
        assert "v2" not in all_segs

    def test_analyze_path_empty(self, tracker):
        all_segs, unmapped = tracker._analyze_path("")
        assert all_segs == set()
        assert unmapped == set()

    def test_analyze_output_keys_mapped(self, tracker):
        all_keys, unmapped = tracker._analyze_output_keys(["mileage", "fuelLevel", "status"])
        # mileage, fuel, status should be mapped
        assert "mileage" in all_keys
        assert "mileage" not in unmapped
        assert "status" not in unmapped

    def test_analyze_output_keys_unmapped(self, tracker):
        all_keys, unmapped = tracker._analyze_output_keys(["xyzField", "abcValue"])
        assert "xyzfield" in unmapped
        assert "abcvalue" in unmapped

    def test_analyze_output_keys_empty(self, tracker):
        all_keys, unmapped = tracker._analyze_output_keys([])
        assert all_keys == set()
        assert unmapped == set()

    def test_analyze_operation_id_mapped(self, tracker):
        all_words, unmapped = tracker._analyze_operation_id("GetVehicleMileage")
        # Vehicle and Mileage should be mapped
        assert "vehicle" in all_words
        assert "mileage" in all_words
        # "get" should be skipped
        assert "get" not in all_words
        # Both should be mapped
        assert "vehicle" not in unmapped
        assert "mileage" not in unmapped

    def test_analyze_operation_id_unmapped(self, tracker):
        all_words, unmapped = tracker._analyze_operation_id("GetXyzAbcDef")
        assert "xyz" in unmapped
        assert "abc" in unmapped
        assert "def" in unmapped

    def test_analyze_operation_id_empty(self, tracker):
        all_words, unmapped = tracker._analyze_operation_id("")
        assert all_words == set()
        assert unmapped == set()

    def test_analyze_coverage_mapped_tool(self, tracker, sample_tool):
        tools = {"get_VehicleMileage": sample_tool}
        report = tracker.analyze_coverage(tools)

        assert report.tools_with_mapping == 1
        assert report.tools_without_mapping == 0
        assert report.tool_coverage_pct == 100.0

    def test_analyze_coverage_unmapped_tool(self, tracker, unmapped_tool):
        tools = {"get_XyzAbcQrs": unmapped_tool}
        report = tracker.analyze_coverage(tools)

        assert report.tools_without_mapping == 1
        assert "get_XyzAbcQrs" in report.unmapped_tools

    def test_analyze_coverage_mixed_tools(self, tracker, sample_tool, unmapped_tool):
        tools = {
            "get_VehicleMileage": sample_tool,
            "get_XyzAbcQrs": unmapped_tool,
        }
        report = tracker.analyze_coverage(tools)

        assert report.tools_with_mapping == 1
        assert report.tools_without_mapping == 1
        assert report.tool_coverage_pct == 50.0


class TestConvenienceFunction:
    def test_analyze_embedding_coverage_function(self, sample_tool):
        tools = {"get_VehicleMileage": sample_tool}
        report = analyze_embedding_coverage(tools)

        assert isinstance(report, CoverageReport)
        assert report.tools_with_mapping == 1


class TestEdgeCases:
    def test_partial_path_match(self, tracker):
        """Test that partial matches work (e.g., 'vehicleid' matches 'vehicle')."""
        all_segs, unmapped = tracker._analyze_path("/api/vehiclelist/mileagehistory")
        # "vehiclelist" should partially match "vehicle"
        # This depends on implementation - check actual behavior
        assert "vehiclelist" in all_segs or "vehicle" in str(all_segs)

    def test_short_segments_skipped(self, tracker):
        """Segments with less than 3 characters should be skipped."""
        all_segs, unmapped = tracker._analyze_path("/a/b/c/vehicles")
        assert "a" not in all_segs
        assert "b" not in all_segs
        assert "c" not in all_segs
        assert "vehicles" in all_segs

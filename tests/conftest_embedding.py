"""
Pytest fixtures for embedding engine evaluation.

These fixtures provide reusable components for testing the embedding
engine, coverage tracking, and search quality.

Usage:
    def test_example(embedding_engine, coverage_tracker, evaluation_queries):
        # Use the fixtures in your tests
        pass
"""

import pytest
from typing import List, Dict, Any

from services.registry.embedding_engine import EmbeddingEngine
from services.registry.embedding_coverage import EmbeddingCoverageTracker, CoverageReport
from services.registry.embedding_evaluator import (
    EmbeddingEvaluator,
    QueryTestCase,
    EvaluationResult,
    EVALUATION_QUERIES,
)
from services.registry.translation_helper import TranslationHelper, TranslationResult
from services.tool_contracts import UnifiedToolDefinition, ParameterDefinition


# ---
# CORE FIXTURES
# ---


@pytest.fixture
def embedding_engine():
    """Provide a fresh EmbeddingEngine instance."""
    return EmbeddingEngine()


@pytest.fixture
def coverage_tracker():
    """Provide a fresh EmbeddingCoverageTracker instance."""
    return EmbeddingCoverageTracker()


@pytest.fixture
def evaluator():
    """Provide a fresh EmbeddingEvaluator instance."""
    return EmbeddingEvaluator()


@pytest.fixture
def translation_helper():
    """Provide a fresh TranslationHelper instance."""
    return TranslationHelper()


# ---
# TEST DATA FIXTURES
# ---


@pytest.fixture
def evaluation_queries() -> List[QueryTestCase]:
    """Provide the full set of evaluation queries."""
    return EVALUATION_QUERIES


@pytest.fixture
def sample_queries() -> List[QueryTestCase]:
    """Provide a small sample of evaluation queries for quick tests."""
    return [
        QueryTestCase(
            query="daj mi kilometražu vozila",
            expected_tool_patterns=["mileage", "odometer", "km"],
            category="vehicle_status",
            difficulty="easy"
        ),
        QueryTestCase(
            query="prikaži dostupna vozila",
            expected_tool_patterns=["available", "vehicle", "fleet"],
            category="availability",
            difficulty="easy"
        ),
        QueryTestCase(
            query="trebam rezervirati auto",
            expected_tool_patterns=["booking", "reservation", "create"],
            category="booking",
            difficulty="medium"
        ),
    ]


@pytest.fixture
def sample_tool() -> UnifiedToolDefinition:
    """Provide a sample tool with typical mappings."""
    return UnifiedToolDefinition(
        operation_id="get_VehicleMileage",
        method="GET",
        path="/api/v1/vehicles/{id}/mileage",
        description="Get vehicle mileage",
        parameters={
            "vehicleId": ParameterDefinition(
                name="vehicleId",
                param_type="string",
                required=True,
                description="Vehicle identifier",
            )
        },
        service_name="fleet",
        service_url="https://api.example.com",
        swagger_name="fleet",
        output_keys=["mileage", "km", "lastUpdated", "vehicleId"],
    )


@pytest.fixture
def unmapped_tool() -> UnifiedToolDefinition:
    """Provide a tool with unmapped terms for fallback testing."""
    return UnifiedToolDefinition(
        operation_id="get_XyzWidget",
        method="GET",
        path="/api/v1/foobar/{id}/quxbaz",
        description="Unknown endpoint",
        parameters={},
        service_name="unknown",
        service_url="https://api.example.com",
        swagger_name="unknown",
        output_keys=["xyzblob", "abcwoof", "qrsmeow"],
    )


@pytest.fixture
def mixed_tools(sample_tool, unmapped_tool) -> Dict[str, UnifiedToolDefinition]:
    """Provide a mix of mapped and unmapped tools."""
    return {
        sample_tool.operation_id: sample_tool,
        unmapped_tool.operation_id: unmapped_tool,
    }


# ---
# ASSERTION HELPERS
# ---


@pytest.fixture
def assert_coverage_grade():
    """Fixture that returns a helper to assert coverage grades."""
    def _assert_grade(report: CoverageReport, min_grade: str):
        """Assert that coverage report meets minimum grade."""
        grade_order = ["A+", "A", "B", "C", "D", "F"]
        result = report.to_dict()
        actual_grade = result["summary"]["overall_quality"].split()[0]

        actual_idx = next(
            (i for i, g in enumerate(grade_order) if g in actual_grade),
            len(grade_order)
        )
        min_idx = grade_order.index(min_grade)

        assert actual_idx <= min_idx, (
            f"Coverage grade {actual_grade} is below minimum {min_grade}"
        )

    return _assert_grade


@pytest.fixture
def assert_translation_confidence():
    """Fixture that returns a helper to assert translation confidence."""
    def _assert_confidence(result: TranslationResult, min_confidence: float):
        """Assert that translation meets minimum confidence."""
        assert result.confidence >= min_confidence, (
            f"Translation confidence {result.confidence} is below "
            f"minimum {min_confidence} for term '{result.original}'"
        )

    return _assert_confidence


@pytest.fixture
def assert_evaluation_metrics():
    """Fixture that returns a helper to assert evaluation metrics."""
    def _assert_metrics(result: EvaluationResult, **thresholds):
        """Assert that evaluation metrics meet thresholds."""
        if "min_mrr" in thresholds:
            assert result.mrr >= thresholds["min_mrr"], (
                f"MRR {result.mrr:.3f} below threshold {thresholds['min_mrr']}"
            )

        if "min_hit_at_1" in thresholds:
            assert result.hit_at_1 >= thresholds["min_hit_at_1"], (
                f"Hit@1 {result.hit_at_1:.3f} below threshold {thresholds['min_hit_at_1']}"
            )

        if "min_hit_at_5" in thresholds:
            assert result.hit_at_5 >= thresholds["min_hit_at_5"], (
                f"Hit@5 {result.hit_at_5:.3f} below threshold {thresholds['min_hit_at_5']}"
            )

        if "min_ndcg_5" in thresholds:
            assert result.ndcg_5 >= thresholds["min_ndcg_5"], (
                f"NDCG@5 {result.ndcg_5:.3f} below threshold {thresholds['min_ndcg_5']}"
            )

    return _assert_metrics


# ---
# MOCK SEARCH FUNCTION
# ---


@pytest.fixture
def mock_search_function():
    """Provide a mock search function for evaluation tests."""
    def _mock_search(query: str, top_k: int = 10) -> List[str]:
        """
        Mock search that returns predictable results based on keywords.

        This is useful for testing the evaluation framework itself
        without requiring actual embeddings.
        """
        results = []

        query_lower = query.lower()

        # Map keywords to tool patterns
        keyword_tools = {
            "kilometraž": ["get_VehicleMileage", "get_OdometerReading"],
            "vozil": ["get_VehicleDetails", "list_Vehicles", "get_VehicleStatus"],
            "goriva": ["get_FuelLevel", "get_FuelConsumption"],
            "rezervacij": ["create_Booking", "get_Reservation", "list_Bookings"],
            "dostupn": ["get_AvailableVehicles", "check_Availability"],
            "lokacij": ["get_VehicleLocation", "get_BranchLocations"],
            "servis": ["get_ServiceHistory", "schedule_Maintenance"],
        }

        for keyword, tools in keyword_tools.items():
            if keyword in query_lower:
                results.extend(tools)

        # Pad with generic results if needed
        while len(results) < top_k:
            results.append(f"generic_tool_{len(results)}")

        return results[:top_k]

    return _mock_search


# ---
# PERFORMANCE FIXTURES
# ---


@pytest.fixture
def performance_timer():
    """Fixture that returns a context manager for timing operations."""
    import time
    from contextlib import contextmanager

    @contextmanager
    def _timer(operation_name: str, max_seconds: float = None):
        """Time an operation and optionally assert max duration."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start

        print(f"{operation_name}: {elapsed:.3f}s")

        if max_seconds is not None:
            assert elapsed <= max_seconds, (
                f"{operation_name} took {elapsed:.3f}s, "
                f"exceeding max of {max_seconds}s"
            )

    return _timer

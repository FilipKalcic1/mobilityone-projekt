"""
Tests for EmbeddingEvaluator - Validates MRR/NDCG metrics calculation.
"""

import pytest
import tempfile
from pathlib import Path

from services.registry.embedding_evaluator import (
    EmbeddingEvaluator,
    QueryTestCase,
    EvaluationResult,
    create_evaluation_dataset,
)


@pytest.fixture
def evaluator():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield EmbeddingEvaluator(data_dir=Path(tmpdir))


@pytest.fixture
def sample_test_cases():
    return [
        QueryTestCase(
            query="prikaži mi vozilo",
            expected_tool_id="get_Vehicle",
            category="vehicle"
        ),
        QueryTestCase(
            query="rezerviraj auto",
            expected_tool_id="post_Booking",
            category="booking"
        ),
        QueryTestCase(
            query="kilometraža vozila",
            expected_tool_id="get_VehicleMileage",
            category="vehicle"
        ),
    ]


class TestQueryTestCase:
    def test_basic_creation(self):
        tc = QueryTestCase(
            query="test query",
            expected_tool_id="get_Test"
        )
        assert tc.query == "test query"
        assert tc.expected_tool_id == "get_Test"
        assert tc.category == "general"
        assert tc.relevance_scores == {}

    def test_with_relevance_scores(self):
        tc = QueryTestCase(
            query="test",
            expected_tool_id="get_Test",
            relevance_scores={"get_Test": 3, "get_Other": 1}
        )
        assert tc.relevance_scores["get_Test"] == 3


class TestEvaluationResult:
    def test_to_dict_structure(self):
        result = EvaluationResult(
            mrr=0.75,
            ndcg_at_5=0.8,
            total_queries=10,
        )
        data = result.to_dict()

        assert "metrics" in data
        assert "summary" in data
        assert "quality_grade" in data
        assert data["metrics"]["mrr"] == 0.75

    def test_quality_grade_excellent(self):
        result = EvaluationResult(mrr=0.95)
        assert "A+" in result.to_dict()["quality_grade"]

    def test_quality_grade_very_good(self):
        result = EvaluationResult(mrr=0.85)
        assert "A " in result.to_dict()["quality_grade"]

    def test_quality_grade_good(self):
        result = EvaluationResult(mrr=0.75)
        assert "B" in result.to_dict()["quality_grade"]

    def test_quality_grade_acceptable(self):
        result = EvaluationResult(mrr=0.65)
        assert "C" in result.to_dict()["quality_grade"]

    def test_quality_grade_poor(self):
        result = EvaluationResult(mrr=0.55)
        assert "D" in result.to_dict()["quality_grade"]

    def test_quality_grade_critical(self):
        result = EvaluationResult(mrr=0.3)
        assert "F" in result.to_dict()["quality_grade"]


class TestEmbeddingEvaluator:
    def test_create_initial_test_set(self, evaluator):
        test_cases = evaluator.create_initial_test_set()
        assert len(test_cases) > 0
        assert all(isinstance(tc, QueryTestCase) for tc in test_cases)

    def test_save_and_load_test_set(self, evaluator, sample_test_cases):
        # Save
        filepath = evaluator.save_test_set(sample_test_cases, "test_eval.json")
        assert filepath.exists()

        # Load
        loaded = evaluator.load_test_set("test_eval.json")
        assert len(loaded) == len(sample_test_cases)
        assert loaded[0].query == sample_test_cases[0].query
        assert loaded[0].expected_tool_id == sample_test_cases[0].expected_tool_id

    def test_load_nonexistent_returns_empty(self, evaluator):
        result = evaluator.load_test_set("nonexistent.json")
        assert result == []

    def test_evaluate_perfect_search(self, evaluator, sample_test_cases):
        """Test evaluation when search always returns correct tool first."""
        def perfect_search(query):
            # Return correct tool based on query
            if "vozilo" in query:
                return [("get_Vehicle", 1.0), ("get_Vehicles", 0.9)]
            elif "rezerviraj" in query:
                return [("post_Booking", 1.0), ("get_Booking", 0.9)]
            elif "kilometraža" in query:
                return [("get_VehicleMileage", 1.0), ("get_Vehicle", 0.8)]
            return []

        result = evaluator.evaluate(perfect_search, sample_test_cases)

        assert result.mrr == 1.0  # Perfect MRR
        assert result.hit_at_1 == 1.0  # 100% hit@1
        assert result.total_queries == 3
        assert result.successful_queries == 3
        assert len(result.failed_queries) == 0

    def test_evaluate_imperfect_search(self, evaluator):
        """Test evaluation when correct tool is at rank 2."""
        test_cases = [
            QueryTestCase(query="test", expected_tool_id="correct_tool")
        ]

        def imperfect_search(query):
            return [
                ("wrong_tool", 1.0),
                ("correct_tool", 0.9),  # Correct at rank 2
                ("another_tool", 0.8),
            ]

        result = evaluator.evaluate(imperfect_search, test_cases)

        assert result.mrr == 0.5  # 1/2 = 0.5
        assert result.hit_at_1 == 0.0
        assert result.hit_at_3 == 1.0
        assert result.successful_queries == 1

    def test_evaluate_failed_search(self, evaluator):
        """Test evaluation when correct tool is not in results."""
        test_cases = [
            QueryTestCase(query="test", expected_tool_id="missing_tool")
        ]

        def failed_search(query):
            return [
                ("wrong_tool_1", 1.0),
                ("wrong_tool_2", 0.9),
            ]

        result = evaluator.evaluate(failed_search, test_cases)

        assert result.mrr == 0.0
        assert result.hit_at_1 == 0.0
        assert result.hit_at_10 == 0.0
        assert result.successful_queries == 0
        assert len(result.failed_queries) == 1
        assert "test" in result.failed_queries[0]

    def test_evaluate_empty_test_cases(self, evaluator):
        """Test evaluation with no test cases."""
        result = evaluator.evaluate(lambda q: [], [])
        assert result.total_queries == 0
        assert result.mrr == 0.0

    def test_evaluate_category_breakdown(self, evaluator, sample_test_cases):
        """Test that per-category MRR is calculated."""
        def perfect_search(query):
            if "vozilo" in query:
                return [("get_Vehicle", 1.0)]
            elif "rezerviraj" in query:
                return [("post_Booking", 1.0)]
            elif "kilometraža" in query:
                return [("get_VehicleMileage", 1.0)]
            return []

        result = evaluator.evaluate(perfect_search, sample_test_cases)

        assert "vehicle" in result.category_mrr
        assert "booking" in result.category_mrr
        assert result.category_mrr["vehicle"] == 1.0
        assert result.category_mrr["booking"] == 1.0


class TestNDCGCalculation:
    def test_ndcg_perfect_ranking(self, evaluator):
        """NDCG should be 1.0 when correct tool is first."""
        test_case = QueryTestCase(query="test", expected_tool_id="correct")
        result_ids = ["correct", "wrong1", "wrong2"]

        ndcg = evaluator._calculate_ndcg(result_ids, test_case, k=3)
        assert ndcg == 1.0

    def test_ndcg_imperfect_ranking(self, evaluator):
        """NDCG should be less than 1.0 when correct tool is not first."""
        test_case = QueryTestCase(query="test", expected_tool_id="correct")
        result_ids = ["wrong1", "correct", "wrong2"]

        ndcg = evaluator._calculate_ndcg(result_ids, test_case, k=3)
        assert 0 < ndcg < 1.0

    def test_ndcg_not_found(self, evaluator):
        """NDCG should be 0.0 when correct tool is not in results."""
        test_case = QueryTestCase(query="test", expected_tool_id="correct")
        result_ids = ["wrong1", "wrong2", "wrong3"]

        ndcg = evaluator._calculate_ndcg(result_ids, test_case, k=3)
        assert ndcg == 0.0

    def test_ndcg_with_relevance_scores(self, evaluator):
        """Test NDCG with explicit relevance scores."""
        test_case = QueryTestCase(
            query="test",
            expected_tool_id="best",
            relevance_scores={"best": 3, "good": 2, "ok": 1}
        )
        result_ids = ["best", "good", "ok"]

        ndcg = evaluator._calculate_ndcg(result_ids, test_case, k=3)
        assert ndcg == 1.0  # Perfect ranking of graded relevance


class TestConvenienceFunction:
    def test_create_evaluation_dataset(self, monkeypatch, tmp_path):
        """Test that convenience function creates dataset."""
        # Monkeypatch the data_dir to use temp directory
        import services.registry.embedding_evaluator as module

        original_init = module.EmbeddingEvaluator.__init__

        def patched_init(self, data_dir=None):
            original_init(self, data_dir=tmp_path)

        monkeypatch.setattr(module.EmbeddingEvaluator, "__init__", patched_init)

        test_cases = create_evaluation_dataset()
        assert len(test_cases) > 0
        assert (tmp_path / "evaluation_queries.json").exists()


class TestEdgeCases:
    def test_evaluate_with_tuple_results(self, evaluator):
        """Test evaluation when search returns (id, score) tuples."""
        test_cases = [
            QueryTestCase(query="test", expected_tool_id="correct")
        ]

        def search_with_scores(query):
            return [("correct", 0.95), ("wrong", 0.8)]

        result = evaluator.evaluate(search_with_scores, test_cases)
        assert result.mrr == 1.0

    def test_evaluate_with_plain_results(self, evaluator):
        """Test evaluation when search returns plain IDs."""
        test_cases = [
            QueryTestCase(query="test", expected_tool_id="correct")
        ]

        def search_plain(query):
            return ["correct", "wrong"]

        result = evaluator.evaluate(search_plain, test_cases)
        assert result.mrr == 1.0

    def test_evaluate_handles_search_error(self, evaluator):
        """Test evaluation handles search function errors gracefully."""
        test_cases = [
            QueryTestCase(query="test", expected_tool_id="correct")
        ]

        def failing_search(query):
            raise ValueError("Search failed")

        result = evaluator.evaluate(failing_search, test_cases)
        assert result.mrr == 0.0
        assert len(result.failed_queries) == 1
        assert "error" in result.failed_queries[0]

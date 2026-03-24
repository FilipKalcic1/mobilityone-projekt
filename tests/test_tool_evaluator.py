"""Tests for services/tool_evaluator.py – ToolEvaluator."""
import pytest
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from services.tool_evaluator import ToolEvaluator, ToolMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluator():
    """ToolEvaluator with no cache file."""
    with patch("services.tool_evaluator.EVALUATION_CACHE_FILE") as mock_path:
        mock_path.exists.return_value = False
        ev = ToolEvaluator()
    return ev


# ===========================================================================
# ToolMetrics
# ===========================================================================

class TestToolMetrics:
    def test_defaults(self):
        m = ToolMetrics(operation_id="op1")
        assert m.total_calls == 0
        assert m.successful_calls == 0
        assert m.failed_calls == 0
        assert m.error_types == {}

    def test_success_rate_no_calls(self):
        m = ToolMetrics(operation_id="op1")
        assert m.success_rate == 0.5  # neutral

    def test_success_rate_with_calls(self):
        m = ToolMetrics(operation_id="op1", total_calls=10, successful_calls=8)
        assert m.success_rate == 0.8

    def test_user_satisfaction_no_feedback(self):
        m = ToolMetrics(operation_id="op1")
        assert m.user_satisfaction == 0.5

    def test_user_satisfaction_with_feedback(self):
        m = ToolMetrics(operation_id="op1", positive_feedback=7, negative_feedback=3)
        assert m.user_satisfaction == 0.7

    def test_overall_score_new_tool(self):
        m = ToolMetrics(operation_id="op1")
        # 0.5*0.6 + 0.5*0.3 + 0.10 (no errors) = 0.3 + 0.15 + 0.10 = 0.55
        assert abs(m.overall_score - 0.55) < 0.01

    def test_overall_score_perfect(self):
        m = ToolMetrics(operation_id="op1", total_calls=100, successful_calls=100,
                        positive_feedback=50, negative_feedback=0)
        # 1.0*0.6 + 1.0*0.3 + 0.10 = 1.0
        assert m.overall_score == pytest.approx(1.0, abs=0.001)

    def test_overall_score_recent_error(self):
        m = ToolMetrics(operation_id="op1", total_calls=10, successful_calls=5,
                        positive_feedback=5, negative_feedback=5)
        # recent error = 0 recency
        m.last_error_time = datetime.now(timezone.utc).isoformat()
        score = m.overall_score
        # 0.5*0.6 + 0.5*0.3 + 0.0 = 0.45
        assert abs(score - 0.45) < 0.01

    def test_overall_score_old_error(self):
        m = ToolMetrics(operation_id="op1", total_calls=10, successful_calls=10,
                        positive_feedback=10, negative_feedback=0)
        m.last_error_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        score = m.overall_score
        # 1.0*0.6 + 1.0*0.3 + 0.10 = 1.0
        assert score == pytest.approx(1.0, abs=0.001)

    def test_overall_score_error_within_day(self):
        m = ToolMetrics(operation_id="op1", total_calls=10, successful_calls=10)
        m.last_error_time = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        score = m.overall_score
        # 1.0*0.6 + 0.5*0.3 + 0.05 = 0.8
        assert abs(score - 0.80) < 0.01

    def test_overall_score_invalid_error_time(self):
        m = ToolMetrics(operation_id="op1", total_calls=10, successful_calls=10)
        m.last_error_time = "not-a-date"
        score = m.overall_score
        # 1.0*0.6 + 0.5*0.3 + 0.05 = 0.8
        assert abs(score - 0.80) < 0.01

    def test_to_dict(self):
        m = ToolMetrics(operation_id="op1", total_calls=5, successful_calls=3)
        d = m.to_dict()
        assert d["operation_id"] == "op1"
        assert d["total_calls"] == 5
        assert "success_rate" in d
        assert "user_satisfaction" in d
        assert "overall_score" in d

    def test_from_dict(self):
        d = {"operation_id": "op2", "total_calls": 10, "successful_calls": 8,
             "failed_calls": 2, "positive_feedback": 5, "negative_feedback": 1}
        m = ToolMetrics.from_dict(d)
        assert m.operation_id == "op2"
        assert m.total_calls == 10
        assert m.successful_calls == 8

    def test_from_dict_defaults(self):
        d = {"operation_id": "op3"}
        m = ToolMetrics.from_dict(d)
        assert m.total_calls == 0
        assert m.error_types == {}


# ===========================================================================
# ToolEvaluator init
# ===========================================================================

class TestInit:
    def test_empty_init(self, evaluator):
        assert evaluator.metrics == {}

    def test_load_from_cache(self, tmp_path):
        cache_file = tmp_path / "eval.json"
        data = {
            "version": "1.0",
            "metrics": [
                {"operation_id": "op1", "total_calls": 5, "successful_calls": 3}
            ]
        }
        cache_file.write_text(json.dumps(data), encoding="utf-8")

        with patch("services.tool_evaluator.EVALUATION_CACHE_FILE", cache_file):
            ev = ToolEvaluator()
        assert "op1" in ev.metrics
        assert ev.metrics["op1"].total_calls == 5

    def test_load_corrupt_cache(self, tmp_path):
        cache_file = tmp_path / "eval.json"
        cache_file.write_text("not json", encoding="utf-8")
        with patch("services.tool_evaluator.EVALUATION_CACHE_FILE", cache_file):
            ev = ToolEvaluator()
        assert ev.metrics == {}


# ===========================================================================
# record_success
# ===========================================================================

class TestRecordSuccess:
    def test_basic(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        evaluator.record_success("op1", response_time_ms=100.0)
        m = evaluator.metrics["op1"]
        assert m.total_calls == 1
        assert m.successful_calls == 1
        assert m.first_call is not None
        assert m.last_call is not None

    def test_response_time_tracked(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        evaluator.record_success("op1", response_time_ms=200.0)
        evaluator.record_success("op1", response_time_ms=400.0)
        m = evaluator.metrics["op1"]
        assert m.total_response_time_ms == 600.0
        assert m.avg_response_time_ms == 300.0

    def test_zero_response_time_ignored(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        evaluator.record_success("op1", response_time_ms=0)
        assert evaluator.metrics["op1"].total_response_time_ms == 0.0

    def test_saves_every_10_calls(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        for i in range(10):
            evaluator.record_success("op1")
        evaluator._save_to_cache.assert_called_once()

    def test_first_call_only_set_once(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        evaluator.record_success("op1")
        first = evaluator.metrics["op1"].first_call
        evaluator.record_success("op1")
        assert evaluator.metrics["op1"].first_call == first


# ===========================================================================
# record_failure
# ===========================================================================

class TestRecordFailure:
    def test_basic(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        evaluator.record_failure("op1", "Server Error", error_type="network")
        m = evaluator.metrics["op1"]
        assert m.total_calls == 1
        assert m.failed_calls == 1
        assert m.last_error == "Server Error"
        assert m.last_error_time is not None
        assert m.error_types["network"] == 1

    def test_error_message_truncated(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        evaluator.record_failure("op1", "x" * 500)
        assert len(evaluator.metrics["op1"].last_error) == 200

    def test_error_types_counted(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        evaluator.record_failure("op1", "err1", error_type="network")
        evaluator.record_failure("op1", "err2", error_type="network")
        evaluator.record_failure("op1", "err3", error_type="validation")
        m = evaluator.metrics["op1"]
        assert m.error_types["network"] == 2
        assert m.error_types["validation"] == 1

    def test_saves_on_every_failure(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        evaluator.record_failure("op1", "err")
        evaluator._save_to_cache.assert_called_once()


# ===========================================================================
# record_user_feedback
# ===========================================================================

class TestRecordUserFeedback:
    def test_positive(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        evaluator.record_user_feedback("op1", positive=True)
        assert evaluator.metrics["op1"].positive_feedback == 1

    def test_negative(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        evaluator.record_user_feedback("op1", positive=False, feedback_text="wrong")
        assert evaluator.metrics["op1"].negative_feedback == 1

    def test_saves_on_feedback(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        evaluator.record_user_feedback("op1", positive=True)
        evaluator._save_to_cache.assert_called_once()


# ===========================================================================
# get_score / get_penalty / get_boost
# ===========================================================================

class TestScoring:
    def test_unknown_tool_neutral(self, evaluator):
        assert evaluator.get_score("nonexistent") == 0.5

    def test_known_tool_score(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        for _ in range(10):
            evaluator.record_success("op1")
        score = evaluator.get_score("op1")
        assert score > 0.5

    def test_penalty_for_bad_tool(self, evaluator):
        evaluator.metrics["op1"] = ToolMetrics(
            operation_id="op1", total_calls=10, successful_calls=0, failed_calls=10,
            last_error_time=datetime.now(timezone.utc).isoformat()
        )
        penalty = evaluator.get_penalty("op1")
        assert penalty > 0.1

    def test_penalty_for_neutral(self, evaluator):
        assert evaluator.get_penalty("unknown") == pytest.approx(0.1, abs=0.01)

    def test_boost_for_good_tool(self, evaluator):
        evaluator.metrics["op1"] = ToolMetrics(
            operation_id="op1", total_calls=100, successful_calls=100,
            positive_feedback=50, negative_feedback=0
        )
        boost = evaluator.get_boost("op1")
        assert boost > 0

    def test_no_boost_below_threshold(self, evaluator):
        evaluator.metrics["op1"] = ToolMetrics(
            operation_id="op1", total_calls=10, successful_calls=5
        )
        assert evaluator.get_boost("op1") == 0.0


# ===========================================================================
# apply_evaluation_adjustment
# ===========================================================================

class TestApplyAdjustment:
    def test_neutral_tool(self, evaluator):
        result = evaluator.apply_evaluation_adjustment("unknown", 0.8)
        # penalty ~0.1, no boost -> ~0.7
        assert 0.6 <= result <= 0.8

    def test_clamp_to_bounds(self, evaluator):
        evaluator.metrics["bad"] = ToolMetrics(
            operation_id="bad", total_calls=100, successful_calls=0,
            last_error_time=datetime.now(timezone.utc).isoformat()
        )
        result = evaluator.apply_evaluation_adjustment("bad", 0.1)
        assert result >= 0.0

    def test_good_tool_gets_boosted(self, evaluator):
        evaluator.metrics["good"] = ToolMetrics(
            operation_id="good", total_calls=100, successful_calls=100,
            positive_feedback=50, negative_feedback=0
        )
        result = evaluator.apply_evaluation_adjustment("good", 0.8)
        assert result >= 0.8  # boosted above base


# ===========================================================================
# get_statistics
# ===========================================================================

class TestGetStatistics:
    def test_empty(self, evaluator):
        stats = evaluator.get_statistics()
        assert stats["message"] == "No tool evaluations yet"

    def test_with_data(self, evaluator):
        evaluator._save_to_cache = MagicMock()
        evaluator.record_success("op1")
        evaluator.record_success("op1")
        evaluator.record_failure("op2", "err")
        stats = evaluator.get_statistics()
        assert stats["total_tools_tracked"] == 2
        assert stats["total_calls"] == 3
        assert stats["total_failures"] == 1
        assert len(stats["top_performers"]) > 0


# ===========================================================================
# _save_to_cache
# ===========================================================================

class TestSaveToCache:
    def test_save(self, tmp_path):
        cache_file = tmp_path / "eval.json"
        with patch("services.tool_evaluator.EVALUATION_CACHE_FILE", cache_file):
            ev = ToolEvaluator()
            ev.metrics["op1"] = ToolMetrics(operation_id="op1", total_calls=5)
            ev._save_to_cache()
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert data["version"] == "1.0"
        assert len(data["metrics"]) == 1

    def test_save_error_handled(self, evaluator):
        with patch("services.tool_evaluator.EVALUATION_CACHE_FILE") as mock_path:
            mock_path.parent.mkdir.side_effect = PermissionError
            evaluator._save_to_cache()  # no exception


# ===========================================================================
# Singleton
# ===========================================================================

class TestSingleton:
    def test_get_tool_evaluator(self):
        import services.tool_evaluator as mod
        mod._tool_evaluator = None
        with patch("services.tool_evaluator.EVALUATION_CACHE_FILE") as mock_path:
            mock_path.exists.return_value = False
            e1 = mod.get_tool_evaluator()
            e2 = mod.get_tool_evaluator()
            assert e1 is e2
        mod._tool_evaluator = None

"""
Tests for QualityTracker service.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import tempfile
import os

from services.quality_tracker import (
    QualityTracker,
    QualitySnapshot,
    QualityTrend,
    get_quality_tracker
)


class TestQualitySnapshot:
    """Tests for QualitySnapshot dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        snapshot = QualitySnapshot(
            timestamp="2024-01-15T10:00:00Z",
            mrr=0.75,
            ndcg_at_5=0.70,
            ndcg_at_10=0.72,
            hit_at_1=0.50,
            hit_at_5=0.80,
            hit_at_10=0.90,
            total_queries=100,
            dictionary_version="v3.4",
            notes="Initial evaluation"
        )

        d = snapshot.to_dict()
        assert d["mrr"] == 0.75
        assert d["ndcg_at_5"] == 0.70
        assert d["dictionary_version"] == "v3.4"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "timestamp": "2024-01-15T10:00:00Z",
            "mrr": 0.75,
            "ndcg_at_5": 0.70,
            "ndcg_at_10": 0.72,
            "hit_at_1": 0.50,
            "hit_at_5": 0.80,
            "hit_at_10": 0.90,
            "total_queries": 100,
            "dictionary_version": "v3.4",
            "notes": ""
        }

        snapshot = QualitySnapshot.from_dict(data)
        assert snapshot.mrr == 0.75
        assert snapshot.total_queries == 100


class TestQualityTrend:
    """Tests for QualityTrend dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        current = QualitySnapshot(
            timestamp="2024-01-20T10:00:00Z",
            mrr=0.80, ndcg_at_5=0.75, ndcg_at_10=0.77,
            hit_at_1=0.55, hit_at_5=0.85, hit_at_10=0.92,
            total_queries=100
        )
        baseline = QualitySnapshot(
            timestamp="2024-01-15T10:00:00Z",
            mrr=0.70, ndcg_at_5=0.65, ndcg_at_10=0.68,
            hit_at_1=0.45, hit_at_5=0.75, hit_at_10=0.85,
            total_queries=100
        )

        trend = QualityTrend(
            current=current,
            baseline=baseline,
            mrr_change=0.10,
            ndcg_change=0.10,
            is_improving=True,
            is_degraded=False,
            recommendation="Quality improving!"
        )

        d = trend.to_dict()
        assert d["mrr_change"] == 0.10
        assert d["is_improving"] is True


class TestQualityTracker:
    """Tests for QualityTracker service."""

    @pytest.fixture
    def temp_history_file(self):
        """Create a temporary history file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"version": "1.0", "snapshots": []}')
            path = Path(f.name)
        yield path
        # Cleanup
        if path.exists():
            os.unlink(path)

    @pytest.fixture
    def tracker(self, temp_history_file):
        """Create tracker with temp file."""
        return QualityTracker(history_file=temp_history_file)

    def test_init_empty(self, tracker):
        """Test initialization with empty history."""
        assert len(tracker._history) == 0

    def test_init_with_existing_history(self, temp_history_file):
        """Test initialization with existing history."""
        # Write some history
        data = {
            "version": "1.0",
            "snapshots": [
                {
                    "timestamp": "2024-01-15T10:00:00Z",
                    "mrr": 0.75, "ndcg_at_5": 0.70, "ndcg_at_10": 0.72,
                    "hit_at_1": 0.50, "hit_at_5": 0.80, "hit_at_10": 0.90,
                    "total_queries": 100, "dictionary_version": "", "notes": ""
                }
            ]
        }
        with open(temp_history_file, 'w') as f:
            json.dump(data, f)

        tracker = QualityTracker(history_file=temp_history_file)
        assert len(tracker._history) == 1
        assert tracker._history[0].mrr == 0.75

    def test_record_from_object(self, tracker):
        """Test recording from EvaluationResult-like object."""
        mock_result = MagicMock()
        mock_result.mrr = 0.75
        mock_result.ndcg_at_5 = 0.70
        mock_result.ndcg_at_10 = 0.72
        mock_result.hit_at_1 = 0.50
        mock_result.hit_at_5 = 0.80
        mock_result.hit_at_10 = 0.90
        mock_result.total_queries = 100

        snapshot = tracker.record(mock_result, dictionary_version="v3.4")

        assert snapshot.mrr == 0.75
        assert snapshot.dictionary_version == "v3.4"
        assert len(tracker._history) == 1

    def test_record_from_dict(self, tracker):
        """Test recording from dictionary."""
        result_dict = {
            "metrics": {
                "mrr": 0.75,
                "ndcg@5": 0.70,
                "ndcg@10": 0.72,
                "hit@1": 0.50,
                "hit@5": 0.80,
                "hit@10": 0.90,
            },
            "total_queries": 100
        }

        snapshot = tracker.record(result_dict)

        assert snapshot.mrr == 0.75
        assert snapshot.ndcg_at_5 == 0.70

    def test_get_baseline(self, tracker):
        """Test getting baseline."""
        # Add two snapshots
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-15T10:00:00Z",
            mrr=0.70, ndcg_at_5=0.65, ndcg_at_10=0.68,
            hit_at_1=0.45, hit_at_5=0.75, hit_at_10=0.85,
            total_queries=100
        ))
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-20T10:00:00Z",
            mrr=0.80, ndcg_at_5=0.75, ndcg_at_10=0.77,
            hit_at_1=0.55, hit_at_5=0.85, hit_at_10=0.92,
            total_queries=100
        ))

        baseline = tracker.get_baseline()
        assert baseline.mrr == 0.70  # First one

    def test_get_latest(self, tracker):
        """Test getting latest snapshot."""
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-15T10:00:00Z",
            mrr=0.70, ndcg_at_5=0.65, ndcg_at_10=0.68,
            hit_at_1=0.45, hit_at_5=0.75, hit_at_10=0.85,
            total_queries=100
        ))
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-20T10:00:00Z",
            mrr=0.80, ndcg_at_5=0.75, ndcg_at_10=0.77,
            hit_at_1=0.55, hit_at_5=0.85, hit_at_10=0.92,
            total_queries=100
        ))

        latest = tracker.get_latest()
        assert latest.mrr == 0.80  # Last one

    def test_analyze_trend_no_data(self, tracker):
        """Test trend analysis with no data."""
        trend = tracker.analyze_trend()
        assert trend is None

    def test_analyze_trend_single_snapshot(self, tracker):
        """Test trend analysis with single snapshot."""
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-15T10:00:00Z",
            mrr=0.75, ndcg_at_5=0.70, ndcg_at_10=0.72,
            hit_at_1=0.50, hit_at_5=0.80, hit_at_10=0.90,
            total_queries=100
        ))

        trend = tracker.analyze_trend()
        assert trend is not None
        assert trend.baseline is None  # No baseline yet
        assert "First evaluation" in trend.recommendation

    def test_analyze_trend_improving(self, tracker):
        """Test trend analysis when quality is improving."""
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-15T10:00:00Z",
            mrr=0.70, ndcg_at_5=0.65, ndcg_at_10=0.68,
            hit_at_1=0.45, hit_at_5=0.75, hit_at_10=0.85,
            total_queries=100
        ))
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-20T10:00:00Z",
            mrr=0.80, ndcg_at_5=0.75, ndcg_at_10=0.77,
            hit_at_1=0.55, hit_at_5=0.85, hit_at_10=0.92,
            total_queries=100
        ))

        trend = tracker.analyze_trend()

        assert trend.is_improving is True
        assert trend.is_degraded is False
        assert abs(trend.mrr_change - 0.10) < 0.001  # Float comparison
        assert "improving" in trend.recommendation.lower()

    def test_analyze_trend_degraded(self, tracker):
        """Test trend analysis when quality is degrading."""
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-15T10:00:00Z",
            mrr=0.80, ndcg_at_5=0.75, ndcg_at_10=0.77,
            hit_at_1=0.55, hit_at_5=0.85, hit_at_10=0.92,
            total_queries=100
        ))
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-20T10:00:00Z",
            mrr=0.50, ndcg_at_5=0.45, ndcg_at_10=0.48,  # Significant drop
            hit_at_1=0.30, hit_at_5=0.60, hit_at_10=0.70,
            total_queries=100
        ))

        trend = tracker.analyze_trend()

        assert trend.is_degraded is True
        assert "ALERT" in trend.recommendation

    def test_analyze_trend_below_minimum(self, tracker):
        """Test trend analysis when below minimum thresholds."""
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-15T10:00:00Z",
            mrr=0.50, ndcg_at_5=0.45, ndcg_at_10=0.48,
            hit_at_1=0.30, hit_at_5=0.60, hit_at_10=0.70,
            total_queries=100
        ))
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-20T10:00:00Z",
            mrr=0.55, ndcg_at_5=0.50, ndcg_at_10=0.52,  # Still below minimum
            hit_at_1=0.35, hit_at_5=0.65, hit_at_10=0.75,
            total_queries=100
        ))

        trend = tracker.analyze_trend()

        # Even though improving, still below minimum
        assert trend.is_degraded is True  # Below MRR_MINIMUM

    def test_get_summary(self, tracker):
        """Test summary generation."""
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-15T10:00:00Z",
            mrr=0.75, ndcg_at_5=0.70, ndcg_at_10=0.72,
            hit_at_1=0.50, hit_at_5=0.80, hit_at_10=0.90,
            total_queries=100
        ))

        summary = tracker.get_summary()

        assert "current_metrics" in summary
        assert "quality_status" in summary
        assert summary["history_count"] == 1

    def test_get_quality_status(self, tracker):
        """Test quality status classification."""
        excellent = QualitySnapshot(
            timestamp="", mrr=0.85, ndcg_at_5=0.75, ndcg_at_10=0.77,
            hit_at_1=0.60, hit_at_5=0.90, hit_at_10=0.95, total_queries=100
        )
        assert tracker._get_quality_status(excellent) == "excellent"

        good = QualitySnapshot(
            timestamp="", mrr=0.70, ndcg_at_5=0.60, ndcg_at_10=0.62,
            hit_at_1=0.45, hit_at_5=0.75, hit_at_10=0.85, total_queries=100
        )
        assert tracker._get_quality_status(good) == "good"

        poor = QualitySnapshot(
            timestamp="", mrr=0.40, ndcg_at_5=0.35, ndcg_at_10=0.38,
            hit_at_1=0.20, hit_at_5=0.50, hit_at_10=0.60, total_queries=100
        )
        assert tracker._get_quality_status(poor) == "poor"

    def test_save_and_load(self, temp_history_file):
        """Test persistence of history."""
        # Create and record
        tracker1 = QualityTracker(history_file=temp_history_file)
        mock_result = MagicMock()
        mock_result.mrr = 0.75
        mock_result.ndcg_at_5 = 0.70
        mock_result.ndcg_at_10 = 0.72
        mock_result.hit_at_1 = 0.50
        mock_result.hit_at_5 = 0.80
        mock_result.hit_at_10 = 0.90
        mock_result.total_queries = 100

        tracker1.record(mock_result)

        # Load in new instance
        tracker2 = QualityTracker(history_file=temp_history_file)

        assert len(tracker2._history) == 1
        assert tracker2._history[0].mrr == 0.75

    def test_clear_history(self, tracker):
        """Test clearing history."""
        tracker._history.append(QualitySnapshot(
            timestamp="2024-01-15T10:00:00Z",
            mrr=0.75, ndcg_at_5=0.70, ndcg_at_10=0.72,
            hit_at_1=0.50, hit_at_5=0.80, hit_at_10=0.90,
            total_queries=100
        ))

        tracker.clear_history()

        assert len(tracker._history) == 0


class TestGetQualityTrackerFunction:
    """Tests for convenience function."""

    def test_get_quality_tracker(self):
        """Test getting default tracker."""
        tracker = get_quality_tracker()
        assert isinstance(tracker, QualityTracker)

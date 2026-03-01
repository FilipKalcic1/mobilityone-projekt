"""
Quality Tracker - Tracks search quality metrics over time.

Tracks MRR, NDCG, and Hit@K metrics from EmbeddingEvaluator runs.
Enables trend analysis and quality regression detection.

ARCHITECTURE:
    EmbeddingEvaluator → QualityTracker → Database/JSON
                              ↓
                        Trend Analysis
                              ↓
                        Quality Alerts

USE CASE:
    1. Run evaluation weekly/monthly
    2. Store results with timestamp
    3. Compare against baseline
    4. Alert on quality degradation
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Storage path
QUALITY_HISTORY_FILE = Path.cwd() / ".cache" / "quality_history.json"


@dataclass
class QualitySnapshot:
    """A single evaluation snapshot."""
    timestamp: str
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    total_queries: int
    dictionary_version: str = ""
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "QualitySnapshot":
        return cls(**data)


@dataclass
class QualityTrend:
    """Quality trend analysis."""
    current: QualitySnapshot
    baseline: Optional[QualitySnapshot]
    mrr_change: float = 0.0
    ndcg_change: float = 0.0
    is_improving: bool = True
    is_degraded: bool = False
    recommendation: str = ""

    def to_dict(self) -> Dict:
        return {
            "current": self.current.to_dict(),
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "mrr_change": round(self.mrr_change, 4),
            "ndcg_change": round(self.ndcg_change, 4),
            "is_improving": self.is_improving,
            "is_degraded": self.is_degraded,
            "recommendation": self.recommendation,
        }


class QualityTracker:
    """
    Tracks search quality over time for regression detection.

    Features:
    - Store evaluation snapshots
    - Compare against baseline
    - Detect quality degradation
    - Generate improvement recommendations

    Usage:
        tracker = QualityTracker()

        # After running evaluation
        from services.registry.embedding_evaluator import EmbeddingEvaluator
        evaluator = EmbeddingEvaluator()
        results = evaluator.evaluate(search_fn, test_cases)

        # Record and analyze
        snapshot = tracker.record(results)
        trend = tracker.analyze_trend()
    """

    # Quality thresholds
    MRR_MINIMUM = 0.60  # Minimum acceptable MRR
    NDCG_MINIMUM = 0.55  # Minimum acceptable NDCG@5
    DEGRADATION_THRESHOLD = 0.05  # Alert if drops more than 5%

    def __init__(self, history_file: Optional[Path] = None):
        """Initialize quality tracker."""
        self.history_file = history_file or QUALITY_HISTORY_FILE
        self._history: List[QualitySnapshot] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._history = [
                        QualitySnapshot.from_dict(s)
                        for s in data.get("snapshots", [])
                    ]
                logger.info(f"Loaded {len(self._history)} quality snapshots")
        except Exception as e:
            logger.warning(f"Could not load quality history: {e}")
            self._history = []

    def _save_history(self) -> None:
        """Save history to file."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump({
                    "version": "1.0",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "snapshots": [s.to_dict() for s in self._history]
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self._history)} quality snapshots")
        except Exception as e:
            logger.error(f"Failed to save quality history: {e}")

    def record(
        self,
        evaluation_result: Any,
        dictionary_version: str = "",
        notes: str = ""
    ) -> QualitySnapshot:
        """
        Record an evaluation result.

        Args:
            evaluation_result: Result from EmbeddingEvaluator.evaluate()
            dictionary_version: Version string for tracking
            notes: Optional notes about this evaluation

        Returns:
            The recorded QualitySnapshot
        """
        # Handle both EvaluationResult object and dict
        if hasattr(evaluation_result, "mrr"):
            snapshot = QualitySnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                mrr=evaluation_result.mrr,
                ndcg_at_5=evaluation_result.ndcg_at_5,
                ndcg_at_10=evaluation_result.ndcg_at_10,
                hit_at_1=evaluation_result.hit_at_1,
                hit_at_5=evaluation_result.hit_at_5,
                hit_at_10=evaluation_result.hit_at_10,
                total_queries=evaluation_result.total_queries,
                dictionary_version=dictionary_version,
                notes=notes
            )
        elif isinstance(evaluation_result, dict):
            metrics = evaluation_result.get("metrics", evaluation_result)
            snapshot = QualitySnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                mrr=metrics.get("mrr", 0),
                ndcg_at_5=metrics.get("ndcg@5", metrics.get("ndcg_at_5", 0)),
                ndcg_at_10=metrics.get("ndcg@10", metrics.get("ndcg_at_10", 0)),
                hit_at_1=metrics.get("hit@1", metrics.get("hit_at_1", 0)),
                hit_at_5=metrics.get("hit@5", metrics.get("hit_at_5", 0)),
                hit_at_10=metrics.get("hit@10", metrics.get("hit_at_10", 0)),
                total_queries=evaluation_result.get("total_queries", 0),
                dictionary_version=dictionary_version,
                notes=notes
            )
        else:
            raise ValueError(f"Unknown evaluation result type: {type(evaluation_result)}")

        self._history.append(snapshot)
        self._save_history()

        logger.info(
            f"Recorded quality snapshot: MRR={snapshot.mrr:.3f}, "
            f"NDCG@5={snapshot.ndcg_at_5:.3f}"
        )

        return snapshot

    def get_baseline(self) -> Optional[QualitySnapshot]:
        """Get the baseline snapshot (oldest or first good one)."""
        if not self._history:
            return None

        # Return the oldest snapshot as baseline
        return self._history[0]

    def get_latest(self) -> Optional[QualitySnapshot]:
        """Get the most recent snapshot."""
        if not self._history:
            return None
        return self._history[-1]

    def get_history(self, days: int = 30) -> List[QualitySnapshot]:
        """Get history for the last N days."""
        if not self._history:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        return [
            s for s in self._history
            if datetime.fromisoformat(s.timestamp.replace("Z", "+00:00")) > cutoff
        ]

    def analyze_trend(self) -> Optional[QualityTrend]:
        """
        Analyze quality trend by comparing latest to baseline.

        Returns:
            QualityTrend with analysis, or None if not enough data
        """
        current = self.get_latest()
        baseline = self.get_baseline()

        if not current:
            return None

        if not baseline or baseline == current:
            # First run - no comparison possible
            return QualityTrend(
                current=current,
                baseline=None,
                recommendation="First evaluation recorded. Run again to track trends."
            )

        # Calculate changes
        mrr_change = current.mrr - baseline.mrr
        ndcg_change = current.ndcg_at_5 - baseline.ndcg_at_5

        # Determine status
        is_improving = mrr_change > 0 and ndcg_change > 0
        is_degraded = (
            mrr_change < -self.DEGRADATION_THRESHOLD or
            ndcg_change < -self.DEGRADATION_THRESHOLD or
            current.mrr < self.MRR_MINIMUM or
            current.ndcg_at_5 < self.NDCG_MINIMUM
        )

        # Generate recommendation
        if is_degraded:
            recommendation = self._generate_degradation_recommendation(
                current, baseline, mrr_change, ndcg_change
            )
        elif is_improving:
            recommendation = (
                f"Quality improving! MRR +{mrr_change:.1%}, NDCG +{ndcg_change:.1%}. "
                "Continue current approach."
            )
        else:
            recommendation = "Quality stable. Consider running feedback analysis for improvements."

        return QualityTrend(
            current=current,
            baseline=baseline,
            mrr_change=mrr_change,
            ndcg_change=ndcg_change,
            is_improving=is_improving,
            is_degraded=is_degraded,
            recommendation=recommendation
        )

    def _generate_degradation_recommendation(
        self,
        current: QualitySnapshot,
        baseline: QualitySnapshot,
        mrr_change: float,
        ndcg_change: float
    ) -> str:
        """Generate actionable recommendation for quality degradation."""
        issues = []

        if current.mrr < self.MRR_MINIMUM:
            issues.append(f"MRR ({current.mrr:.2f}) below minimum ({self.MRR_MINIMUM})")

        if current.ndcg_at_5 < self.NDCG_MINIMUM:
            issues.append(f"NDCG@5 ({current.ndcg_at_5:.2f}) below minimum ({self.NDCG_MINIMUM})")

        if mrr_change < -self.DEGRADATION_THRESHOLD:
            issues.append(f"MRR dropped {abs(mrr_change):.1%} from baseline")

        actions = [
            "1. Run FeedbackAnalyzer to identify missing dictionary terms",
            "2. Review recent dictionary changes for regressions",
            "3. Check for new API tools that lack embeddings",
        ]

        return f"QUALITY ALERT: {'; '.join(issues)}. Actions: {' '.join(actions)}"

    def get_summary(self) -> Dict[str, Any]:
        """Get quality tracking summary."""
        latest = self.get_latest()
        baseline = self.get_baseline()
        trend = self.analyze_trend()

        return {
            "current_metrics": latest.to_dict() if latest else None,
            "baseline_metrics": baseline.to_dict() if baseline and baseline != latest else None,
            "trend": trend.to_dict() if trend else None,
            "history_count": len(self._history),
            "quality_status": self._get_quality_status(latest) if latest else "unknown",
        }

    def _get_quality_status(self, snapshot: QualitySnapshot) -> str:
        """Get quality status string."""
        if snapshot.mrr >= 0.8 and snapshot.ndcg_at_5 >= 0.7:
            return "excellent"
        elif snapshot.mrr >= self.MRR_MINIMUM and snapshot.ndcg_at_5 >= self.NDCG_MINIMUM:
            return "good"
        elif snapshot.mrr >= 0.5:
            return "acceptable"
        else:
            return "poor"

    def clear_history(self) -> None:
        """Clear all history (use with caution)."""
        self._history = []
        if self.history_file.exists():
            self.history_file.unlink()
        logger.warning("Quality history cleared")


# Convenience function
def get_quality_tracker() -> QualityTracker:
    """Get the default quality tracker instance."""
    return QualityTracker()

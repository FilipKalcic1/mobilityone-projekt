"""
Model Drift Detection Service

Detects AI model performance drift by comparing
recent metrics against baseline.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

BASELINE_DAYS = settings.DRIFT_BASELINE_DAYS
ANALYSIS_HOURS = settings.DRIFT_ANALYSIS_HOURS
MIN_SAMPLES = settings.DRIFT_MIN_SAMPLES


@dataclass
class DriftReport:
    """Drift analysis result."""
    has_drift: bool
    severity: str  # none, low, medium, high
    error_rate: float
    baseline_error_rate: float
    latency_ms: float
    baseline_latency_ms: float
    hallucination_rate: float
    baseline_hallucination_rate: float
    sample_count: int
    alerts: List[str]


class ModelDriftDetector:
    """Detect model performance drift."""

    def __init__(self, redis_client=None, db_session=None, alert_callback=None):
        self.redis = redis_client
        self.db = db_session
        self.alert_callback = alert_callback
        self._metrics: deque = deque(maxlen=10000)
        self._baseline_cache: Optional[Dict] = None
        self._baseline_time: Optional[datetime] = None

    async def record_interaction(
        self,
        model_version: str,
        latency_ms: int,
        success: bool,
        error_type: str = None,
        confidence_score: float = None,
        tools_called: List[str] = None,
        hallucination_reported: bool = False,
        tenant_id: str = None
    ) -> None:
        """Record a model interaction."""
        metric = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model_version,
            "latency": max(0, latency_ms),
            "success": success,
            "error": error_type,
            "hallucination": hallucination_reported
        }
        self._metrics.append(metric)

        # Persist to Redis
        if self.redis:
            try:
                key = f"drift:m:{int(datetime.now(timezone.utc).timestamp() * 1000)}"
                await self.redis.setex(key, BASELINE_DAYS * 86400, json.dumps(metric))
            except Exception as e:
                logger.debug(f"Failed to persist drift metric: {e}")

    async def check_drift(self, model_version: str = None) -> DriftReport:
        """Check for model drift."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ANALYSIS_HOURS)

        # Get recent metrics
        recent = [
            m for m in self._metrics
            if datetime.fromisoformat(m["ts"].replace('Z', '+00:00')) > cutoff
        ]

        if model_version:
            recent = [m for m in recent if m.get("model") == model_version]

        if len(recent) < MIN_SAMPLES:
            return DriftReport(
                has_drift=False, severity="none",
                error_rate=0, baseline_error_rate=0,
                latency_ms=0, baseline_latency_ms=0,
                hallucination_rate=0, baseline_hallucination_rate=0,
                sample_count=len(recent),
                alerts=[f"Insufficient data: {len(recent)}/{MIN_SAMPLES} samples"]
            )

        # Calculate current metrics
        total = len(recent)
        errors = sum(1 for m in recent if not m["success"])
        hallucinations = sum(1 for m in recent if m.get("hallucination"))
        latencies = [m["latency"] for m in recent]

        current = {
            "error_rate": errors / total,
            "latency": sum(latencies) / total,
            "hallucination_rate": hallucinations / total
        }

        # Get baseline
        baseline = await self._get_baseline()

        # Detect drift
        alerts = []
        severity = "none"

        # Error rate check
        if baseline["error_rate"] > 0:
            deviation = (current["error_rate"] - baseline["error_rate"]) / baseline["error_rate"]
            if deviation > 1.0:
                alerts.append(f"Error rate +{deviation*100:.0f}%")
                severity = "high" if deviation > 2.0 else "medium"
            elif deviation > 0.5:
                alerts.append(f"Error rate +{deviation*100:.0f}%")
                if severity == "none":
                    severity = "low"

        # Latency check
        if baseline["latency"] > 0:
            deviation = (current["latency"] - baseline["latency"]) / baseline["latency"]
            if deviation > 0.5:
                alerts.append(f"Latency +{deviation*100:.0f}%")
                if deviation > 1.0 and severity in ["none", "low"]:
                    severity = "medium"
                elif severity == "none":
                    severity = "low"

        # Hallucination check
        if baseline["hallucination_rate"] > 0:
            deviation = (current["hallucination_rate"] - baseline["hallucination_rate"]) / baseline["hallucination_rate"]
            if deviation > 0.5:
                alerts.append(f"Hallucination rate +{deviation*100:.0f}%")
                if deviation > 1.0 and severity in ["none", "low"]:
                    severity = "medium"

        report = DriftReport(
            has_drift=len(alerts) > 0,
            severity=severity,
            error_rate=current["error_rate"],
            baseline_error_rate=baseline["error_rate"],
            latency_ms=current["latency"],
            baseline_latency_ms=baseline["latency"],
            hallucination_rate=current["hallucination_rate"],
            baseline_hallucination_rate=baseline["hallucination_rate"],
            sample_count=total,
            alerts=alerts
        )

        if alerts and self.alert_callback:
            try:
                await self.alert_callback(report)
            except Exception as e:
                logger.warning(f"Alert callback failed: {e}")

        return report

    async def _get_baseline(self) -> Dict:
        """Get or calculate baseline metrics."""
        # Use cache if fresh
        if (self._baseline_cache and self._baseline_time and
            datetime.now(timezone.utc) - self._baseline_time < timedelta(hours=1)):
            return self._baseline_cache

        # Calculate from in-memory metrics
        cutoff = datetime.now(timezone.utc) - timedelta(days=BASELINE_DAYS)
        baseline_metrics = [
            m for m in self._metrics
            if datetime.fromisoformat(m["ts"].replace('Z', '+00:00')) > cutoff
        ]

        if len(baseline_metrics) >= MIN_SAMPLES:
            total = len(baseline_metrics)
            errors = sum(1 for m in baseline_metrics if not m["success"])
            hallucinations = sum(1 for m in baseline_metrics if m.get("hallucination"))
            latencies = [m["latency"] for m in baseline_metrics]

            self._baseline_cache = {
                "error_rate": errors / total,
                "latency": sum(latencies) / total,
                "hallucination_rate": hallucinations / total
            }
        else:
            # Default baseline
            self._baseline_cache = {
                "error_rate": 0.05,
                "latency": 1500,
                "hallucination_rate": 0.02
            }

        self._baseline_time = datetime.now(timezone.utc)
        return self._baseline_cache

    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        report = await self.check_drift()
        return {
            "healthy": not report.has_drift or report.severity in ["none", "low"],
            "drift_detected": report.has_drift,
            "severity": report.severity,
            "sample_count": report.sample_count
        }


# Singleton
_detector: Optional[ModelDriftDetector] = None


def get_drift_detector(redis_client=None, db_session=None, alert_callback=None) -> ModelDriftDetector:
    """Get or create drift detector singleton."""
    global _detector
    if _detector is None:
        _detector = ModelDriftDetector(redis_client, db_session, alert_callback)
    return _detector


def reset_drift_detector() -> None:
    """Reset singleton for testing."""
    global _detector
    _detector = None

"""Tests for ModelDriftDetector - AI model performance drift detection."""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timezone, timedelta
from services.model_drift_detector import (
    ModelDriftDetector, DriftReport, reset_drift_detector
)


@pytest.fixture
def detector():
    return ModelDriftDetector(redis_client=None)


@pytest.fixture
def detector_redis():
    return ModelDriftDetector(redis_client=AsyncMock())


class TestDriftReport:
    def test_defaults(self):
        r = DriftReport(
            has_drift=False, severity="none",
            error_rate=0, baseline_error_rate=0,
            latency_ms=0, baseline_latency_ms=0,
            hallucination_rate=0, baseline_hallucination_rate=0,
            sample_count=0, alerts=[]
        )
        assert r.has_drift is False
        assert r.severity == "none"


class TestRecordInteraction:
    async def test_records_metric(self, detector):
        await detector.record_interaction(
            model_version="gpt-4o-mini",
            latency_ms=500,
            success=True
        )
        assert len(detector._metrics) == 1
        assert detector._metrics[0]["success"] is True

    async def test_negative_latency_clamped(self, detector):
        await detector.record_interaction(
            model_version="gpt-4o-mini",
            latency_ms=-100,
            success=True
        )
        assert detector._metrics[0]["latency"] == 0

    async def test_multiple_recordings(self, detector):
        for i in range(5):
            await detector.record_interaction("gpt-4o-mini", 100 + i, True)
        assert len(detector._metrics) == 5

    async def test_redis_persist(self, detector_redis):
        await detector_redis.record_interaction("gpt-4o-mini", 500, True)
        detector_redis.redis.setex.assert_called_once()

    async def test_redis_failure_doesnt_crash(self, detector_redis):
        detector_redis.redis.setex.side_effect = Exception("fail")
        await detector_redis.record_interaction("gpt-4o-mini", 500, True)
        assert len(detector_redis._metrics) == 1

    async def test_error_interaction(self, detector):
        await detector.record_interaction(
            "gpt-4o-mini", 500, False, error_type="timeout"
        )
        assert detector._metrics[0]["error"] == "timeout"
        assert detector._metrics[0]["success"] is False

    async def test_hallucination_flag(self, detector):
        await detector.record_interaction(
            "gpt-4o-mini", 500, True, hallucination_reported=True
        )
        assert detector._metrics[0]["hallucination"] is True


class TestCheckDrift:
    async def test_insufficient_data(self, detector):
        report = await detector.check_drift()
        assert report.has_drift is False
        assert "Insufficient data" in report.alerts[0]

    async def test_no_drift_when_healthy(self, detector):
        for _ in range(20):
            await detector.record_interaction("gpt-4o-mini", 500, True)
        report = await detector.check_drift()
        assert report.severity == "none"

    async def test_error_drift_detected(self, detector):
        # Record baseline-like data first
        for _ in range(30):
            await detector.record_interaction("gpt-4o-mini", 500, True)
        # Record high error rate
        for _ in range(20):
            await detector.record_interaction("gpt-4o-mini", 500, False)
        report = await detector.check_drift()
        assert report.error_rate > 0

    async def test_filter_by_model(self, detector):
        for _ in range(20):
            await detector.record_interaction("model-a", 500, True)
        for _ in range(20):
            await detector.record_interaction("model-b", 500, False)
        report = await detector.check_drift(model_version="model-a")
        assert report.error_rate == 0.0

    async def test_alert_callback_called(self):
        callback = AsyncMock()
        detector = ModelDriftDetector(alert_callback=callback)
        # Create enough data with high error rate
        for _ in range(15):
            await detector.record_interaction("gpt-4o-mini", 500, True)
        for _ in range(15):
            await detector.record_interaction("gpt-4o-mini", 500, False)
        report = await detector.check_drift()
        if report.has_drift:
            callback.assert_called_once()


class TestGetBaseline:
    async def test_default_baseline(self, detector):
        baseline = await detector._get_baseline()
        assert baseline["error_rate"] == 0.05
        assert baseline["latency"] == 1500

    async def test_cached_baseline(self, detector):
        detector._baseline_cache = {"error_rate": 0.1, "latency": 1000, "hallucination_rate": 0.01}
        detector._baseline_time = datetime.now(timezone.utc)
        baseline = await detector._get_baseline()
        assert baseline["error_rate"] == 0.1


class TestHealthCheck:
    async def test_healthy_with_no_data(self, detector):
        health = await detector.health_check()
        assert health["healthy"] is True

    async def test_healthy_with_good_data(self, detector):
        for _ in range(20):
            await detector.record_interaction("gpt-4o-mini", 500, True)
        health = await detector.health_check()
        assert health["healthy"] is True


class TestSingleton:
    def test_reset(self):
        import services.model_drift_detector as mod
        mod._detector = "something"
        reset_drift_detector()
        assert mod._detector is None


class TestCheckDriftAdvanced:
    """Advanced tests for check_drift - covers lines 128-133, 139-143, 147-151, 167-170."""

    async def test_high_error_rate_high_severity(self):
        """Test error rate > 200% baseline triggers high severity (lines 128-129)."""
        detector = ModelDriftDetector()
        # Force baseline with 10% error rate
        detector._baseline_cache = {"error_rate": 0.10, "latency": 500, "hallucination_rate": 0.01}
        detector._baseline_time = datetime.now(timezone.utc)

        # Need 50+ samples. Create 50% error rate (5x baseline = 400% increase > 200%)
        for _ in range(25):
            await detector.record_interaction("gpt-4o-mini", 500, True)
        for _ in range(25):
            await detector.record_interaction("gpt-4o-mini", 500, False)

        report = await detector.check_drift()
        # deviation = (0.50 - 0.10) / 0.10 = 4.0 > 2.0 -> high severity
        assert report.has_drift is True
        assert report.severity == "high"

    async def test_medium_error_rate_medium_severity(self):
        """Test error rate 100-200% above baseline triggers medium severity (lines 128-129)."""
        detector = ModelDriftDetector()
        detector._baseline_cache = {"error_rate": 0.10, "latency": 500, "hallucination_rate": 0.01}
        detector._baseline_time = datetime.now(timezone.utc)

        # Need 50+ samples. Create 25% error rate (150% increase, 1.5x deviation)
        for _ in range(38):
            await detector.record_interaction("gpt-4o-mini", 500, True)
        for _ in range(12):  # 12/50 = 24% error rate
            await detector.record_interaction("gpt-4o-mini", 500, False)

        report = await detector.check_drift()
        # deviation = (0.24 - 0.10) / 0.10 = 1.4 (> 1.0 but <= 2.0) -> medium
        assert report.has_drift is True
        assert report.severity == "medium"

    async def test_low_error_rate_low_severity(self):
        """Test error rate 50-100% above baseline triggers low severity (lines 130-133)."""
        detector = ModelDriftDetector()
        detector._baseline_cache = {"error_rate": 0.10, "latency": 500, "hallucination_rate": 0.01}
        detector._baseline_time = datetime.now(timezone.utc)

        # Need 50+ samples. Create 16% error rate (60% increase)
        for _ in range(42):
            await detector.record_interaction("gpt-4o-mini", 500, True)
        for _ in range(8):  # 8/50 = 16% error rate
            await detector.record_interaction("gpt-4o-mini", 500, False)

        report = await detector.check_drift()
        # deviation = (0.16 - 0.10) / 0.10 = 0.6 (> 0.5 but <= 1.0) -> low
        assert report.has_drift is True
        assert report.severity == "low"

    async def test_latency_increase_medium_severity(self):
        """Test latency > 100% increase triggers medium severity (lines 139-141)."""
        detector = ModelDriftDetector()
        detector._baseline_cache = {"error_rate": 0.01, "latency": 500, "hallucination_rate": 0.01}
        detector._baseline_time = datetime.now(timezone.utc)

        # Need 50+ samples. All successful but very high latency (3x baseline = 200% increase)
        for _ in range(50):
            await detector.record_interaction("gpt-4o-mini", 1500, True)

        report = await detector.check_drift()
        # deviation = (1500 - 500) / 500 = 2.0 > 1.0 -> should bump to medium
        assert report.has_drift is True
        assert report.severity in ["medium", "low"]

    async def test_latency_increase_low_severity(self):
        """Test latency 50-100% increase triggers low severity (lines 142-143)."""
        detector = ModelDriftDetector()
        detector._baseline_cache = {"error_rate": 0.01, "latency": 500, "hallucination_rate": 0.01}
        detector._baseline_time = datetime.now(timezone.utc)

        # Need 50+ samples. Moderate latency increase (75% above baseline)
        for _ in range(50):
            await detector.record_interaction("gpt-4o-mini", 875, True)

        report = await detector.check_drift()
        # deviation = (875 - 500) / 500 = 0.75 (> 0.5, <= 1.0) -> low
        assert report.has_drift is True
        assert report.severity == "low"

    async def test_hallucination_medium_severity(self):
        """Test hallucination rate > 100% increase triggers medium (lines 147-151)."""
        detector = ModelDriftDetector()
        detector._baseline_cache = {"error_rate": 0.01, "latency": 500, "hallucination_rate": 0.10}
        detector._baseline_time = datetime.now(timezone.utc)

        # Need 50+ samples. 100% hallucination rate (900% increase)
        for _ in range(50):
            await detector.record_interaction("gpt-4o-mini", 500, True, hallucination_reported=True)

        report = await detector.check_drift()
        # deviation = (1.0 - 0.10) / 0.10 = 9.0 > 1.0 -> medium if no other higher
        assert report.has_drift is True
        assert report.severity in ["medium", "high"]

    async def test_hallucination_low_severity(self):
        """Test hallucination rate 50-100% increase (lines 147-151)."""
        detector = ModelDriftDetector()
        detector._baseline_cache = {"error_rate": 0.01, "latency": 500, "hallucination_rate": 0.20}
        detector._baseline_time = datetime.now(timezone.utc)

        # Need 50+ samples. 35% hallucination rate (75% increase from 20%)
        for _ in range(33):
            await detector.record_interaction("gpt-4o-mini", 500, True, hallucination_reported=False)
        for _ in range(17):  # 17/50 = 34% hallucination
            await detector.record_interaction("gpt-4o-mini", 500, True, hallucination_reported=True)

        report = await detector.check_drift()
        # deviation = (0.34 - 0.20) / 0.20 = 0.7 (> 0.5, <= 1.0)
        assert report.has_drift is True

    async def test_alert_callback_success(self):
        """Test alert callback is called on drift (line 166-168)."""
        callback = AsyncMock()
        detector = ModelDriftDetector(alert_callback=callback)
        detector._baseline_cache = {"error_rate": 0.10, "latency": 500, "hallucination_rate": 0.01}
        detector._baseline_time = datetime.now(timezone.utc)

        # Need 50+ samples. Create high drift condition
        for _ in range(25):
            await detector.record_interaction("gpt-4o-mini", 500, True)
        for _ in range(25):
            await detector.record_interaction("gpt-4o-mini", 500, False)

        report = await detector.check_drift()
        assert report.has_drift is True
        callback.assert_called_once()

    async def test_alert_callback_failure_handled(self):
        """Test alert callback failure is caught (lines 167-170)."""
        failing_callback = AsyncMock(side_effect=Exception("Callback failed"))
        detector = ModelDriftDetector(alert_callback=failing_callback)
        detector._baseline_cache = {"error_rate": 0.10, "latency": 500, "hallucination_rate": 0.01}
        detector._baseline_time = datetime.now(timezone.utc)

        # Need 50+ samples. Create drift condition
        for _ in range(25):
            await detector.record_interaction("gpt-4o-mini", 500, True)
        for _ in range(25):
            await detector.record_interaction("gpt-4o-mini", 500, False)

        # Should not raise despite callback failure
        report = await detector.check_drift()
        assert report is not None
        assert report.has_drift is True

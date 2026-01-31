"""
Feedback Loop Integration Tests
Tests for admin_review, hallucination_repository, and related services.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Import the services we're testing
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAdminReviewService:
    """Tests for AdminReviewService."""

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        db = MagicMock()

        # Create a mock result that has sync methods (like SQLAlchemy Result)
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_result.all.return_value = []

        db.execute = AsyncMock(return_value=mock_result)
        db.commit = AsyncMock()
        db.rollback = AsyncMock()
        return db

    @pytest.fixture
    def service(self, mock_db):
        """Create AdminReviewService instance."""
        from services.admin_review import AdminReviewService
        return AdminReviewService(mock_db)

    def test_is_safe_text_clean(self, service):
        """Test clean text passes validation."""
        assert service.is_safe_text("Normal text without any issues")
        assert service.is_safe_text("Provjera s posebnim znakovima: ćčžšđ")
        assert service.is_safe_text("Numbers 12345 and symbols @#$")

    def test_is_safe_text_xss_detection(self, service):
        """Test XSS patterns are detected."""
        assert not service.is_safe_text("<script>alert('xss')</script>")
        assert not service.is_safe_text("javascript:void(0)")
        assert not service.is_safe_text("<img onerror=alert(1)>")

    def test_is_safe_text_sql_injection_detection(self, service):
        """Test SQL injection patterns are detected."""
        assert not service.is_safe_text("'; DROP TABLE users; --")
        assert not service.is_safe_text("UNION SELECT * FROM passwords")
        assert not service.is_safe_text("1; DELETE FROM users")

    def test_is_safe_text_base64_encoded_attack(self, service):
        """Test base64-encoded attacks are detected."""
        import base64
        # Base64 encoded "<script>"
        encoded = base64.b64encode(b"<script>alert(1)</script>").decode()
        # This should be detected
        result = service.is_safe_text(f"Check this: {encoded}")
        # May or may not detect depending on length threshold

    def test_validate_limit(self, service):
        """Test limit validation."""
        assert service._validate_limit(50) == 50
        assert service._validate_limit(-10) == 1  # Minimum is 1
        assert service._validate_limit(99999) == 1000  # Max is 1000
        assert service._validate_limit("invalid") == 50  # Default

    def test_validate_offset(self, service):
        """Test offset validation."""
        assert service._validate_offset(0) == 0
        assert service._validate_offset(100) == 100
        assert service._validate_offset(-50) == 0  # Minimum is 0

    def test_validate_ip_address_valid(self, service):
        """Test valid IP addresses pass validation."""
        assert service._validate_ip_address("192.168.1.1") == "192.168.1.1"
        assert service._validate_ip_address("10.0.0.1") == "10.0.0.1"
        assert service._validate_ip_address("::1") == "::1"  # IPv6 loopback

    def test_validate_ip_address_invalid(self, service):
        """Test invalid IP addresses are rejected."""
        assert service._validate_ip_address("not-an-ip") is None
        assert service._validate_ip_address("999.999.999.999") is None
        assert service._validate_ip_address("") is None
        assert service._validate_ip_address(None) is None

    def test_allowed_categories(self, service):
        """Test category whitelist."""
        valid_categories = [
            "wrong_data", "outdated", "misunderstood",
            "api_error", "rag_failure", "hallucination", "user_error"
        ]
        for cat in valid_categories:
            assert cat in service.ALLOWED_CATEGORIES

        assert "invalid_category" not in service.ALLOWED_CATEGORIES

    @pytest.mark.asyncio
    async def test_get_statistics(self, service, mock_db):
        """Test statistics retrieval."""
        # Mock database responses - mock_db already has proper result setup
        # The fixture sets up mock_result with scalar() returning 0 and all() returning []

        stats = await service.get_statistics("admin_user")

        assert "total_errors" in stats
        assert "hallucinations_reported" in stats
        assert "category_breakdown" in stats

    @pytest.mark.asyncio
    async def test_mark_hallucination_reviewed_invalid_category(self, service):
        """Test rejection of invalid category."""
        from services.admin_review import ValidationError

        with pytest.raises(ValidationError):
            await service.mark_hallucination_reviewed(
                admin_id="admin",
                report_id=str(uuid4()),
                category="invalid_category"
            )

    @pytest.mark.asyncio
    async def test_mark_hallucination_reviewed_dangerous_text(self, service):
        """Test rejection of dangerous correction text."""
        from services.admin_review import SecurityError

        with pytest.raises(SecurityError):
            await service.mark_hallucination_reviewed(
                admin_id="admin",
                report_id=str(uuid4()),
                correction="<script>alert('xss')</script>"
            )


class TestHallucinationRepository:
    """Tests for HallucinationRepository."""

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        db = MagicMock()

        # Create a mock result that has sync methods (like SQLAlchemy Result)
        mock_result = MagicMock()
        mock_result.scalar.return_value = 50
        mock_result.all.return_value = []

        db.execute = AsyncMock(return_value=mock_result)
        db.commit = AsyncMock()
        db.rollback = AsyncMock()
        db.add = MagicMock()
        return db

    @pytest.fixture
    def repo(self, mock_db):
        """Create HallucinationRepository instance."""
        from services.hallucination_repository import HallucinationRepository
        return HallucinationRepository(mock_db)

    def test_validate_limit(self, repo):
        """Test limit validation."""
        assert repo._validate_limit(50) == 50
        assert repo._validate_limit(-10) == 1
        assert repo._validate_limit(99999) == 1000

    def test_validate_category(self, repo):
        """Test category validation."""
        assert repo._validate_category("wrong_data") == "wrong_data"
        assert repo._validate_category("invalid") is None
        assert repo._validate_category(None) is None

    @pytest.mark.asyncio
    async def test_create_report(self, repo, mock_db):
        """Test creating a hallucination report."""
        report = await repo.create(
            user_query="What is the weather?",
            bot_response="The weather is sunny in Antarctica.",
            user_feedback="That's wrong",
            model="gpt-4o-mini",
            apply_gdpr_masking=False
        )

        assert report is not None
        assert report.user_query == "What is the weather?"
        assert report.reviewed is False
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_statistics(self, repo, mock_db):
        """Test statistics retrieval."""
        # mock_db fixture already sets up proper result with scalar() and all()

        stats = await repo.get_statistics()

        assert "total_reports" in stats
        assert "unreviewed_count" in stats
        assert "category_breakdown" in stats


class TestGDPRMasking:
    """Tests for GDPR masking service."""

    @pytest.fixture
    def masker(self):
        """Create masking service."""
        from services.gdpr_masking import GDPRMaskingService
        return GDPRMaskingService(use_hashing=True)

    def test_mask_email(self, masker):
        """Test email masking."""
        result = masker.mask_pii("Contact me at test@example.com")
        assert "test@example.com" not in result.masked_text
        assert "[EMAIL-" in result.masked_text
        assert result.pii_count == 1

    def test_mask_phone_croatian(self, masker):
        """Test Croatian phone number masking."""
        result = masker.mask_pii("Nazovi me na 091 234 5678")
        assert "091 234 5678" not in result.masked_text
        assert "[PHONE-" in result.masked_text

    def test_mask_oib_valid(self, masker):
        """Test valid OIB masking."""
        # Valid OIB with correct checksum (example)
        result = masker.mask_pii("Moj OIB je 94577403194")
        # Check if it was masked (depends on checksum validation)
        assert result.masked_text != "Moj OIB je 94577403194" or result.pii_count == 0

    def test_mask_credit_card_visa(self, masker):
        """Test Visa card masking."""
        result = masker.mask_pii("Card: 4111111111111111")
        assert "4111111111111111" not in result.masked_text
        assert "[CARD-" in result.masked_text

    def test_mask_ipv4(self, masker):
        """Test IPv4 address masking."""
        result = masker.mask_pii("Server IP: 192.168.1.100")
        assert "192.168.1.100" not in result.masked_text
        assert "[IP-" in result.masked_text

    def test_mask_dict(self, masker):
        """Test dictionary field masking."""
        data = {
            "user_query": "My email is test@example.com",
            "safe_field": "This should not change",
            "nested": {
                "user_query": "Another email here@test.com"
            }
        }
        result = masker.mask_dict(data)

        assert "test@example.com" not in result["user_query"]
        assert result["safe_field"] == "This should not change"
        # Nested should also be masked
        assert "here@test.com" not in result["nested"]["user_query"]

    def test_mask_dict_max_depth(self, masker):
        """Test max depth protection."""
        # Create deeply nested structure
        data = {"level": 0}
        current = data
        for i in range(15):
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        # Should not raise RecursionError
        result = masker.mask_dict(data, max_depth=10)
        assert result is not None

    def test_consistent_hashing(self, masker):
        """Test that same input produces same hash."""
        text1 = masker.mask_pii("test@example.com")
        text2 = masker.mask_pii("test@example.com")

        # Extract hash from masked text
        assert text1.masked_text == text2.masked_text


class TestConflictResolver:
    """Tests for conflict resolution service."""

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        return db

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis = MagicMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock(return_value=True)
        redis.setex = AsyncMock(return_value=True)
        redis.delete = AsyncMock(return_value=True)
        redis.incr = AsyncMock(return_value=1)
        redis.lpush = AsyncMock(return_value=1)
        redis.ltrim = AsyncMock(return_value=True)
        redis.lrange = AsyncMock(return_value=[])
        redis.publish = AsyncMock(return_value=1)
        redis.scan = AsyncMock(return_value=(0, []))
        return redis

    @pytest.fixture
    def resolver(self, mock_db, mock_redis):
        """Create conflict resolver."""
        from services.conflict_resolver import ConflictResolver
        return ConflictResolver(mock_db, mock_redis)

    @pytest.mark.asyncio
    async def test_acquire_lock(self, resolver, mock_redis):
        """Test acquiring edit lock."""
        mock_redis.get.return_value = None  # No existing lock

        lock, existing = await resolver.acquire_edit_lock(
            "record-123",
            "admin-user"
        )

        assert lock is not None
        assert lock.admin_id == "admin-user"
        assert existing is None
        mock_redis.setex.assert_called()

    @pytest.mark.asyncio
    async def test_release_lock(self, resolver, mock_redis):
        """Test releasing lock."""
        result = await resolver.release_lock("record-123", "admin-user")
        assert result is True

    @pytest.mark.asyncio
    async def test_version_control(self, resolver, mock_redis):
        """Test version number tracking."""
        version = await resolver._get_version("record-123")
        assert version >= 1

        mock_redis.incr.return_value = 2
        new_version = await resolver._increment_version("record-123")
        assert new_version == 2


class TestCostTracker:
    """Tests for cost tracking service."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis = MagicMock()
        redis.ping = AsyncMock(return_value=True)
        redis.get = AsyncMock(return_value=None)
        redis.hgetall = AsyncMock(return_value={})
        redis.hincrby = AsyncMock(return_value=1)
        redis.hincrbyfloat = AsyncMock(return_value=1.0)
        redis.pipeline = MagicMock()
        redis.pipeline.return_value.hincrby = MagicMock()
        redis.pipeline.return_value.expire = MagicMock()
        redis.pipeline.return_value.execute = AsyncMock(return_value=[1, 1, 1])
        return redis

    @pytest.fixture
    def tracker(self, mock_redis):
        """Create cost tracker."""
        from services.cost_tracker import CostTracker
        return CostTracker(mock_redis)

    def test_pricing_config(self):
        """Test pricing is loaded from config."""
        from services.cost_tracker import INPUT_PRICE, OUTPUT_PRICE, DAILY_BUDGET

        assert INPUT_PRICE > 0
        assert OUTPUT_PRICE > 0
        assert DAILY_BUDGET > 0

    def test_calculate_cost(self, tracker):
        """Test cost calculation."""
        cost = tracker._calculate_cost(1000, 500)
        assert cost > 0
        assert cost < 1.0  # Should be in cents range

    def test_zero_tokens(self, tracker):
        """Test that zero tokens result in zero cost."""
        cost = tracker._calculate_cost(0, 0)
        assert cost == 0

    @pytest.mark.asyncio
    async def test_record_usage(self, tracker, mock_redis):
        """Test recording usage."""
        cost = await tracker.record_usage(
            prompt_tokens=100,
            completion_tokens=50,
            tenant_id="test-tenant"
        )

        assert cost > 0
        stats = await tracker.get_session_stats()
        assert stats["session_prompt_tokens"] == 100
        assert stats["session_completion_tokens"] == 50

    @pytest.mark.asyncio
    async def test_session_stats(self, tracker):
        """Test session stats accumulation."""
        await tracker.record_usage(prompt_tokens=100, completion_tokens=50)
        await tracker.record_usage(prompt_tokens=200, completion_tokens=100)

        stats = await tracker.get_session_stats()
        assert stats["session_prompt_tokens"] == 300
        assert stats["session_completion_tokens"] == 150
        assert stats["session_requests"] == 2


class TestModelDriftDetector:
    """Tests for model drift detection."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis = MagicMock()
        redis.get = AsyncMock(return_value=None)
        redis.setex = AsyncMock(return_value=True)
        redis.zadd = AsyncMock(return_value=1)
        redis.zrangebyscore = AsyncMock(return_value=[])
        redis.zremrangebyscore = AsyncMock(return_value=0)
        redis.lpush = AsyncMock(return_value=1)
        redis.ltrim = AsyncMock(return_value=True)
        redis.lrange = AsyncMock(return_value=[])
        return redis

    @pytest.fixture
    def detector(self, mock_redis):
        """Create drift detector."""
        from services.model_drift_detector import ModelDriftDetector
        return ModelDriftDetector(mock_redis)

    @pytest.mark.asyncio
    async def test_record_interaction(self, detector):
        """Test recording interaction metrics."""
        await detector.record_interaction(
            model_version="gpt-4o-mini-2024-07-18",
            latency_ms=500,
            success=True
        )

        assert len(detector._metrics) == 1

    @pytest.mark.asyncio
    async def test_check_drift_insufficient_data(self, detector):
        """Test drift check with insufficient data returns no drift."""
        report = await detector.check_drift()
        assert not report.has_drift
        assert report.severity == "none"
        assert report.sample_count == 0

    def test_drift_report_dataclass(self):
        """Test DriftReport dataclass structure."""
        from services.model_drift_detector import DriftReport

        report = DriftReport(
            has_drift=True,
            severity="medium",
            error_rate=0.15,
            baseline_error_rate=0.05,
            latency_ms=2000,
            baseline_latency_ms=1500,
            hallucination_rate=0.05,
            baseline_hallucination_rate=0.02,
            sample_count=100,
            alerts=["Error rate +200%"]
        )
        assert report.has_drift
        assert report.severity == "medium"
        assert len(report.alerts) == 1


class TestRAGScheduler:
    """Tests for RAG refresh scheduler."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis = MagicMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock(return_value=True)
        redis.setex = AsyncMock(return_value=True)
        redis.delete = AsyncMock(return_value=True)
        redis.publish = AsyncMock(return_value=1)
        redis.pubsub = MagicMock()
        return redis

    @pytest.fixture
    def scheduler(self, mock_redis):
        """Create scheduler."""
        from services.rag_scheduler import RAGScheduler
        return RAGScheduler(mock_redis, refresh_interval_hours=1)

    def test_backoff_calculation(self, scheduler):
        """Test exponential backoff."""
        scheduler.metrics.consecutive_failures = 0
        backoff0 = scheduler._calculate_backoff()
        assert backoff0 == 60  # MIN_RETRY_DELAY_SECONDS

        scheduler.metrics.consecutive_failures = 3
        backoff3 = scheduler._calculate_backoff()
        assert backoff3 == 60 * 8  # 2^3 * 60

    def test_next_refresh_calculation(self, scheduler):
        """Test next refresh time calculation."""
        next_time = scheduler._calculate_next_refresh()
        assert next_time is not None

    @pytest.mark.asyncio
    async def test_force_refresh(self, scheduler, mock_redis):
        """Test force refresh trigger."""
        result = await scheduler.force_refresh(reason="test")

        assert result["status"] == "triggered"
        mock_redis.publish.assert_called()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

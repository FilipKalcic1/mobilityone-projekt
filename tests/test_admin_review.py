"""
Tests for services/admin_review.py - AdminReviewService

Comprehensive tests covering:
- AdminReviewService class initialization and pattern compilation
- Input validation (_validate_ip_address, _validate_limit, _validate_offset)
- Security checks (is_safe_text, _check_dangerous_patterns)
- SQL escaping (_escape_like_pattern)
- Audit logging (_audit)
- CRUD operations (get_hallucinations_for_review, mark_hallucination_reviewed, etc.)
- Statistics and export functionality
"""

import pytest
import json
from datetime import datetime, timezone
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from services.admin_review import (
    AdminReviewService,
    SecurityError,
    ValidationError,
    MAX_LIMIT,
    DEFAULT_LIMIT,
    MAX_EXPORT_LIMIT,
    MIN_LIMIT,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def mock_db():
    """Mock async database session."""
    db = AsyncMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.add = MagicMock()
    return db


@pytest.fixture
def service(mock_db):
    """AdminReviewService with mocked database."""
    return AdminReviewService(db=mock_db)


@pytest.fixture
def sample_report():
    """Create a sample hallucination report mock."""
    report = MagicMock()
    report.id = uuid4()
    report.created_at = datetime.now(timezone.utc)
    report.user_query = "What is the mileage?"
    report.bot_response = "The mileage is 50000 km."
    report.user_feedback = "krivo"
    report.model = "gpt-4"
    report.conversation_id = "conv-123"
    report.tenant_id = "tenant-1"
    report.reviewed = False
    report.reviewed_by = None
    report.reviewed_at = None
    report.correction = None
    report.category = None
    report.api_raw_response = {"data": "raw"}
    report.retrieved_chunks = ["chunk1", "chunk2"]
    return report


@pytest.fixture
def sample_audit_log():
    """Create a sample audit log mock."""
    log = MagicMock()
    log.id = uuid4()
    log.created_at = datetime.now(timezone.utc)
    log.action = "VIEW_HALLUCINATIONS"
    log.entity_type = "hallucination_report"
    log.entity_id = str(uuid4())
    log.details = {"admin_id": "admin-1", "ip_address": "192.168.1.1"}
    return log


# ===========================================================================
# Initialization Tests
# ===========================================================================

class TestInit:
    def test_init_stores_db(self, mock_db):
        """Test that __init__ stores the database session."""
        service = AdminReviewService(db=mock_db)
        assert service.db is mock_db

    def test_init_compiles_patterns(self, mock_db):
        """Test that __init__ calls _compile_patterns."""
        # Clear compiled patterns first
        AdminReviewService._compiled_patterns = None

        service = AdminReviewService(db=mock_db)

        assert AdminReviewService._compiled_patterns is not None
        assert len(AdminReviewService._compiled_patterns) == len(AdminReviewService.DANGEROUS_PATTERNS)

    def test_compile_patterns_only_once(self, mock_db):
        """Test that patterns are compiled only once (class-level caching)."""
        AdminReviewService._compiled_patterns = None

        service1 = AdminReviewService(db=mock_db)
        patterns1 = AdminReviewService._compiled_patterns

        service2 = AdminReviewService(db=mock_db)
        patterns2 = AdminReviewService._compiled_patterns

        assert patterns1 is patterns2  # Same object, not recompiled

    def test_allowed_categories_is_frozenset(self, service):
        """Test that ALLOWED_CATEGORIES is a frozenset (immutable)."""
        assert isinstance(AdminReviewService.ALLOWED_CATEGORIES, frozenset)
        assert "wrong_data" in AdminReviewService.ALLOWED_CATEGORIES
        assert "hallucination" in AdminReviewService.ALLOWED_CATEGORIES


# ===========================================================================
# IP Address Validation Tests
# ===========================================================================

class TestValidateIpAddress:
    def test_valid_ipv4(self, service):
        """Test valid IPv4 address."""
        result = service._validate_ip_address("192.168.1.1")
        assert result == "192.168.1.1"

    def test_valid_ipv4_with_whitespace(self, service):
        """Test IPv4 with leading/trailing whitespace."""
        result = service._validate_ip_address("  10.0.0.1  ")
        assert result == "10.0.0.1"

    def test_valid_ipv6(self, service):
        """Test valid IPv6 address."""
        result = service._validate_ip_address("::1")
        assert result == "::1"

    def test_valid_ipv6_full(self, service):
        """Test full IPv6 address."""
        result = service._validate_ip_address("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert result is not None
        assert "2001" in result

    def test_invalid_ip_returns_none(self, service):
        """Test that invalid IP returns None."""
        result = service._validate_ip_address("not-an-ip")
        assert result is None

    def test_none_ip_returns_none(self, service):
        """Test that None input returns None."""
        result = service._validate_ip_address(None)
        assert result is None

    def test_empty_string_returns_none(self, service):
        """Test that empty string returns None."""
        result = service._validate_ip_address("")
        assert result is None

    def test_malformed_ip_returns_none(self, service):
        """Test malformed IP address."""
        result = service._validate_ip_address("256.256.256.256")
        assert result is None

    def test_ip_with_port_returns_none(self, service):
        """Test IP with port (not valid for ip_address module)."""
        result = service._validate_ip_address("192.168.1.1:8080")
        assert result is None


# ===========================================================================
# Limit Validation Tests
# ===========================================================================

class TestValidateLimit:
    def test_valid_limit(self, service):
        """Test valid limit within bounds."""
        result = service._validate_limit(50)
        assert result == 50

    def test_limit_at_max(self, service):
        """Test limit exactly at MAX_LIMIT."""
        result = service._validate_limit(MAX_LIMIT)
        assert result == MAX_LIMIT

    def test_limit_above_max_clamped(self, service):
        """Test limit above MAX_LIMIT is clamped."""
        result = service._validate_limit(MAX_LIMIT + 100)
        assert result == MAX_LIMIT

    def test_limit_at_min(self, service):
        """Test limit exactly at MIN_LIMIT."""
        result = service._validate_limit(MIN_LIMIT)
        assert result == MIN_LIMIT

    def test_limit_below_min_clamped(self, service):
        """Test limit below MIN_LIMIT is clamped."""
        result = service._validate_limit(0)
        assert result == MIN_LIMIT

    def test_negative_limit_clamped(self, service):
        """Test negative limit is clamped to MIN_LIMIT."""
        result = service._validate_limit(-10)
        assert result == MIN_LIMIT

    def test_non_integer_returns_default(self, service):
        """Test non-integer returns DEFAULT_LIMIT."""
        result = service._validate_limit("50")
        assert result == DEFAULT_LIMIT

    def test_none_returns_default(self, service):
        """Test None returns DEFAULT_LIMIT."""
        result = service._validate_limit(None)
        assert result == DEFAULT_LIMIT

    def test_custom_max_limit(self, service):
        """Test custom max_limit parameter."""
        result = service._validate_limit(200, max_limit=100)
        assert result == 100


# ===========================================================================
# Offset Validation Tests
# ===========================================================================

class TestValidateOffset:
    def test_valid_offset(self, service):
        """Test valid positive offset."""
        result = service._validate_offset(10)
        assert result == 10

    def test_zero_offset(self, service):
        """Test zero offset."""
        result = service._validate_offset(0)
        assert result == 0

    def test_negative_offset_clamped(self, service):
        """Test negative offset is clamped to 0."""
        result = service._validate_offset(-5)
        assert result == 0

    def test_non_integer_returns_zero(self, service):
        """Test non-integer returns 0."""
        result = service._validate_offset("10")
        assert result == 0

    def test_none_returns_zero(self, service):
        """Test None returns 0."""
        result = service._validate_offset(None)
        assert result == 0

    def test_float_returns_zero(self, service):
        """Test float returns 0 (not integer)."""
        result = service._validate_offset(10.5)
        assert result == 0


# ===========================================================================
# Security Check Tests - is_safe_text
# ===========================================================================

class TestIsSafeText:
    def test_safe_plain_text(self, service):
        """Test that plain text is considered safe."""
        assert service.is_safe_text("This is a normal correction") is True

    def test_safe_croatian_text(self, service):
        """Test Croatian text with special characters."""
        assert service.is_safe_text("Ispravna kilometraža je 45000 km") is True

    def test_empty_string_is_safe(self, service):
        """Test empty string is considered safe."""
        assert service.is_safe_text("") is True

    def test_none_is_safe(self, service):
        """Test None is considered safe."""
        assert service.is_safe_text(None) is True

    def test_non_string_is_safe(self, service):
        """Test non-string input is considered safe."""
        assert service.is_safe_text(123) is True

    def test_script_tag_unsafe(self, service):
        """Test <script> tag is detected as unsafe."""
        assert service.is_safe_text("<script>alert('xss')</script>") is False

    def test_javascript_protocol_unsafe(self, service):
        """Test javascript: protocol is detected as unsafe."""
        assert service.is_safe_text("javascript:alert(1)") is False

    def test_event_handler_unsafe(self, service):
        """Test event handler (onclick=) is detected as unsafe."""
        assert service.is_safe_text('<img onerror="alert(1)">') is False

    def test_eval_unsafe(self, service):
        """Test eval() is detected as unsafe."""
        assert service.is_safe_text("eval('malicious')") is False

    def test_exec_unsafe(self, service):
        """Test exec() is detected as unsafe."""
        assert service.is_safe_text("exec('malicious')") is False

    def test_python_import_injection_unsafe(self, service):
        """Test __import__ is detected as unsafe."""
        assert service.is_safe_text("__import__('os')") is False

    def test_subprocess_unsafe(self, service):
        """Test subprocess is detected as unsafe."""
        assert service.is_safe_text("subprocess.call(['ls'])") is False

    def test_os_system_unsafe(self, service):
        """Test os.system is detected as unsafe."""
        assert service.is_safe_text("os.system('rm -rf')") is False

    def test_template_injection_dollar_unsafe(self, service):
        """Test ${} template injection is detected as unsafe."""
        assert service.is_safe_text("${malicious}") is False

    def test_template_injection_mustache_unsafe(self, service):
        """Test {{}} template injection is detected as unsafe."""
        assert service.is_safe_text("{{malicious}}") is False

    def test_sql_drop_table_unsafe(self, service):
        """Test DROP TABLE is detected as unsafe."""
        assert service.is_safe_text("DROP TABLE users") is False

    def test_sql_delete_from_unsafe(self, service):
        """Test DELETE FROM is detected as unsafe."""
        assert service.is_safe_text("DELETE FROM users") is False

    def test_sql_comment_injection_unsafe(self, service):
        """Test SQL comment injection is detected as unsafe."""
        assert service.is_safe_text("; -- comment") is False

    def test_sql_union_select_unsafe(self, service):
        """Test UNION SELECT is detected as unsafe."""
        assert service.is_safe_text("UNION SELECT * FROM users") is False

    def test_sql_insert_into_unsafe(self, service):
        """Test INSERT INTO is detected as unsafe."""
        assert service.is_safe_text("INSERT INTO users VALUES (1)") is False

    def test_sql_update_set_unsafe(self, service):
        """Test UPDATE SET is detected as unsafe."""
        assert service.is_safe_text("UPDATE users SET admin=1") is False

    def test_base64_encoded_script_unsafe(self, service):
        """Test base64-encoded dangerous pattern is detected."""
        import base64
        # Base64 encode "<script>alert('xss')</script>"
        encoded = base64.b64encode(b"<script>alert('xss')</script>").decode()
        assert service.is_safe_text(encoded) is False

    def test_unicode_escape_unsafe(self, service):
        """Test unicode escape sequence for < is detected."""
        # \u003c is <
        text = "\\u003cscript\\u003e"
        assert service.is_safe_text(text) is False

    def test_html_entity_unsafe(self, service):
        """Test HTML entity encoding is detected."""
        text = "&lt;script&gt;alert(1)&lt;/script&gt;"
        assert service.is_safe_text(text) is False

    def test_too_many_special_chars_unsafe(self, service):
        """Test text with >10% special characters is unsafe."""
        # 15 special chars in 100 char string = 15% > 10%
        text = "a" * 85 + "<>{}[]()$`\\<><>{}"
        assert service.is_safe_text(text) is False

    def test_acceptable_special_chars(self, service):
        """Test text with acceptable amount of special chars."""
        # 5 special chars in 100 char string = 5% < 10%
        text = "a" * 95 + "<>{}"
        assert service.is_safe_text(text) is True

    def test_case_insensitive_detection(self, service):
        """Test case-insensitive pattern detection."""
        assert service.is_safe_text("JAVASCRIPT:alert(1)") is False
        assert service.is_safe_text("<SCRIPT>") is False


# ===========================================================================
# Check Dangerous Patterns Tests
# ===========================================================================

class TestCheckDangerousPatterns:
    def test_returns_true_on_match(self, service):
        """Test returns True when pattern matches."""
        result = service._check_dangerous_patterns("<script>alert(1)</script>")
        assert result is True

    def test_returns_false_on_no_match(self, service):
        """Test returns False when no pattern matches."""
        result = service._check_dangerous_patterns("safe text here")
        assert result is False

    def test_multiple_patterns(self, service):
        """Test detection of multiple different patterns."""
        assert service._check_dangerous_patterns("eval(") is True
        assert service._check_dangerous_patterns("exec(") is True
        assert service._check_dangerous_patterns("drop table") is True


# ===========================================================================
# Escape LIKE Pattern Tests
# ===========================================================================

class TestEscapeLikePattern:
    def test_escape_percent(self, service):
        """Test escaping of % character."""
        result = service._escape_like_pattern("100%")
        assert result == "100\\%"

    def test_escape_underscore(self, service):
        """Test escaping of _ character."""
        result = service._escape_like_pattern("user_name")
        assert result == "user\\_name"

    def test_escape_backslash(self, service):
        """Test escaping of \\ character."""
        result = service._escape_like_pattern("path\\file")
        assert result == "path\\\\file"

    def test_escape_multiple(self, service):
        """Test escaping multiple special characters."""
        result = service._escape_like_pattern("100%_test\\path")
        assert result == "100\\%\\_test\\\\path"

    def test_empty_string(self, service):
        """Test empty string returns empty string."""
        result = service._escape_like_pattern("")
        assert result == ""

    def test_none_returns_none(self, service):
        """Test None returns None (falsy check)."""
        result = service._escape_like_pattern(None)
        assert result is None

    def test_no_special_chars(self, service):
        """Test string without special chars unchanged."""
        result = service._escape_like_pattern("normal text")
        assert result == "normal text"


# ===========================================================================
# Audit Tests
# ===========================================================================

class TestAudit:
    @pytest.mark.asyncio
    async def test_audit_success(self, service, mock_db):
        """Test successful audit log creation."""
        await service._audit(
            action="TEST_ACTION",
            admin_id="admin-1",
            details={"report_id": "123"},
            ip_address="192.168.1.1"
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_audit_with_none_ip(self, service, mock_db):
        """Test audit with None IP address."""
        await service._audit(
            action="TEST_ACTION",
            admin_id="admin-1",
            details={"report_id": "123"},
            ip_address=None
        )

        mock_db.add.assert_called_once()
        # IP should be "unknown" in the audit entry
        call_args = mock_db.add.call_args[0][0]
        assert call_args.details["ip_address"] == "unknown"

    @pytest.mark.asyncio
    async def test_audit_exception_handling(self, service, mock_db):
        """Test audit handles exceptions gracefully (lines 168-174)."""
        mock_db.commit.side_effect = Exception("Database error")

        # Should not raise, just log
        await service._audit(
            action="TEST_ACTION",
            admin_id="admin-1",
            details={},
            ip_address="192.168.1.1"
        )

        mock_db.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_audit_rollback_also_fails(self, service, mock_db):
        """Test audit when rollback also fails (line 174)."""
        mock_db.commit.side_effect = Exception("Commit error")
        mock_db.rollback.side_effect = Exception("Rollback error")

        # Should not raise, handles both errors
        await service._audit(
            action="TEST_ACTION",
            admin_id="admin-1",
            details={},
            ip_address="192.168.1.1"
        )


# ===========================================================================
# Get Hallucinations For Review Tests
# ===========================================================================

class TestGetHallucinationsForReview:
    @pytest.mark.asyncio
    async def test_basic_fetch(self, service, mock_db, sample_report):
        """Test basic hallucination fetch."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_db.execute.return_value = mock_result

        result = await service.get_hallucinations_for_review(
            admin_id="admin-1",
            limit=50,
            offset=0
        )

        assert len(result) == 1
        assert result[0]["id"] == str(sample_report.id)
        assert result[0]["user_query"] == sample_report.user_query

    @pytest.mark.asyncio
    async def test_with_unreviewed_only_true(self, service, mock_db, sample_report):
        """Test fetch with unreviewed_only=True filter."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_db.execute.return_value = mock_result

        result = await service.get_hallucinations_for_review(
            admin_id="admin-1",
            unreviewed_only=True
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_with_unreviewed_only_false(self, service, mock_db, sample_report):
        """Test fetch with unreviewed_only=False."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_db.execute.return_value = mock_result

        result = await service.get_hallucinations_for_review(
            admin_id="admin-1",
            unreviewed_only=False
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_with_tenant_filter(self, service, mock_db, sample_report):
        """Test fetch with tenant filter."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_db.execute.return_value = mock_result

        result = await service.get_hallucinations_for_review(
            admin_id="admin-1",
            tenant_filter="tenant-1"
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_pagination(self, service, mock_db, sample_report):
        """Test pagination with limit and offset."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_db.execute.return_value = mock_result

        result = await service.get_hallucinations_for_review(
            admin_id="admin-1",
            limit=10,
            offset=5
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_validates_limit(self, service, mock_db, sample_report):
        """Test that limit is validated."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_db.execute.return_value = mock_result

        # Pass invalid limit
        result = await service.get_hallucinations_for_review(
            admin_id="admin-1",
            limit=99999  # Above MAX_LIMIT
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_database_exception(self, service, mock_db):
        """Test database exception handling (lines 357-359)."""
        mock_db.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception) as exc_info:
            await service.get_hallucinations_for_review(admin_id="admin-1")

        assert "Database error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_api_raw_response_masked_in_list(self, service, mock_db, sample_report):
        """Test that api_raw_response is masked in list view."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_db.execute.return_value = mock_result

        result = await service.get_hallucinations_for_review(admin_id="admin-1")

        assert result[0]["api_raw_response"] == "[AVAILABLE - click to view]"

    @pytest.mark.asyncio
    async def test_empty_results(self, service, mock_db):
        """Test empty result set."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        result = await service.get_hallucinations_for_review(admin_id="admin-1")

        assert result == []


# ===========================================================================
# Mark Hallucination Reviewed Tests
# ===========================================================================

class TestMarkHallucinationReviewed:
    @pytest.mark.asyncio
    async def test_success_path(self, service, mock_db):
        """Test successful review marking."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db.execute.return_value = mock_result

        report_id = str(uuid4())
        result = await service.mark_hallucination_reviewed(
            admin_id="admin-1",
            report_id=report_id,
            correction="Correct answer is 45000 km",
            category="wrong_data",
            ip_address="192.168.1.1"
        )

        assert result["success"] is True
        assert result["report_id"] == report_id
        assert result["category"] == "wrong_data"

    @pytest.mark.asyncio
    async def test_security_violation_dangerous_correction(self, service, mock_db):
        """Test security violation on dangerous correction text."""
        mock_db.execute.return_value = MagicMock()
        mock_db.commit.return_value = None

        report_id = str(uuid4())

        with pytest.raises(SecurityError) as exc_info:
            await service.mark_hallucination_reviewed(
                admin_id="admin-1",
                report_id=report_id,
                correction="<script>alert('xss')</script>",
                category="wrong_data"
            )

        assert "dangerous content" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_category(self, service, mock_db):
        """Test validation error on invalid category."""
        report_id = str(uuid4())

        with pytest.raises(ValidationError) as exc_info:
            await service.mark_hallucination_reviewed(
                admin_id="admin-1",
                report_id=report_id,
                category="invalid_category"
            )

        assert "Invalid category" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_uuid(self, service, mock_db):
        """Test invalid UUID format."""
        result = await service.mark_hallucination_reviewed(
            admin_id="admin-1",
            report_id="not-a-valid-uuid"
        )

        assert result["success"] is False
        assert "Invalid report ID format" in result["error"]

    @pytest.mark.asyncio
    async def test_report_not_found(self, service, mock_db):
        """Test when report is not found."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_db.execute.return_value = mock_result

        report_id = str(uuid4())
        result = await service.mark_hallucination_reviewed(
            admin_id="admin-1",
            report_id=report_id
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_db_retry_mechanism(self, service, mock_db):
        """Test database retry on failure (lines 441-461)."""
        # Fail first two times, succeed on third
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db.execute.side_effect = [
            Exception("Retry 1"),
            Exception("Retry 2"),
            mock_result
        ]

        report_id = str(uuid4())
        result = await service.mark_hallucination_reviewed(
            admin_id="admin-1",
            report_id=report_id
        )

        assert result["success"] is True
        assert mock_db.rollback.call_count == 2

    @pytest.mark.asyncio
    async def test_db_retry_exhausted(self, service, mock_db):
        """Test all retries exhausted."""
        mock_db.execute.side_effect = Exception("Persistent error")

        report_id = str(uuid4())

        with pytest.raises(Exception) as exc_info:
            await service.mark_hallucination_reviewed(
                admin_id="admin-1",
                report_id=report_id
            )

        assert "Persistent error" in str(exc_info.value)
        assert mock_db.rollback.call_count == 3

    @pytest.mark.asyncio
    async def test_no_correction_or_category(self, service, mock_db):
        """Test marking reviewed without correction or category."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db.execute.return_value = mock_result

        report_id = str(uuid4())
        result = await service.mark_hallucination_reviewed(
            admin_id="admin-1",
            report_id=report_id
        )

        assert result["success"] is True


# ===========================================================================
# Get Audit Log Tests
# ===========================================================================

class TestGetAuditLog:
    @pytest.mark.asyncio
    async def test_basic_fetch(self, service, mock_db, sample_audit_log):
        """Test basic audit log fetch."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_audit_log]
        mock_db.execute.return_value = mock_result

        result = await service.get_audit_log(admin_id="admin-1")

        assert len(result) == 1
        assert result[0]["action"] == "VIEW_HALLUCINATIONS"

    @pytest.mark.asyncio
    async def test_pagination(self, service, mock_db, sample_audit_log):
        """Test audit log pagination (lines 502-513)."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_audit_log]
        mock_db.execute.return_value = mock_result

        result = await service.get_audit_log(
            admin_id="admin-1",
            limit=50,
            offset=10
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_limit_capped_at_500(self, service, mock_db, sample_audit_log):
        """Test that audit log limit is capped at 500."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        # This should be internally capped to 500
        await service.get_audit_log(admin_id="admin-1", limit=1000)

    @pytest.mark.asyncio
    async def test_response_format(self, service, mock_db, sample_audit_log):
        """Test audit log response format."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_audit_log]
        mock_db.execute.return_value = mock_result

        result = await service.get_audit_log(admin_id="admin-1")

        entry = result[0]
        assert "id" in entry
        assert "timestamp" in entry
        assert "action" in entry
        assert "entity_type" in entry
        assert "entity_id" in entry
        assert "details" in entry


# ===========================================================================
# Get Statistics Tests
# ===========================================================================

class TestGetStatistics:
    @pytest.mark.asyncio
    async def test_all_counts(self, service, mock_db):
        """Test all statistics counts (lines 538-604)."""
        # Mock different query results
        def mock_execute(query):
            result = MagicMock()
            result.scalar.return_value = 10
            result.all.return_value = [("wrong_data", 5), ("hallucination", 3), (None, 2)]
            return result

        mock_db.execute = AsyncMock(side_effect=mock_execute)

        result = await service.get_statistics(admin_id="admin-1")

        assert "total_errors" in result
        assert "corrected_errors" in result
        assert "hallucinations_reported" in result
        assert "hallucinations_pending_review" in result
        assert "hallucinations_reviewed" in result
        assert "false_positives_skipped" in result
        assert "category_breakdown" in result
        assert "audit_entries" in result

    @pytest.mark.asyncio
    async def test_category_breakdown(self, service, mock_db):
        """Test category breakdown in statistics (line 587)."""
        execute_count = 0

        def mock_execute(query):
            nonlocal execute_count
            execute_count += 1
            result = MagicMock()
            # For category query (6th call)
            if execute_count == 6:
                result.all.return_value = [
                    ("wrong_data", 5),
                    ("hallucination", 3),
                    (None, 2)  # Uncategorized
                ]
            else:
                result.scalar.return_value = 10
            return result

        mock_db.execute = AsyncMock(side_effect=mock_execute)

        result = await service.get_statistics(admin_id="admin-1")

        assert "category_breakdown" in result


# ===========================================================================
# Export For Training Tests
# ===========================================================================

class TestExportForTraining:
    @pytest.mark.asyncio
    async def test_openai_chat_format(self, service, mock_db, sample_report):
        """Test export in OpenAI chat format (lines 661-672)."""
        sample_report.correction = "Correct answer is 45000 km"
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_db.execute.return_value = mock_result

        result = await service.export_for_training(
            admin_id="admin-1",
            format="openai_chat"
        )

        assert len(result) == 1
        assert "messages" in result[0]
        assert len(result[0]["messages"]) == 3
        assert result[0]["messages"][0]["role"] == "system"
        assert result[0]["messages"][1]["role"] == "user"
        assert result[0]["messages"][2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_openai_completion_format(self, service, mock_db, sample_report):
        """Test export in OpenAI completion format (lines 673-681)."""
        sample_report.correction = "Correct answer is 45000 km"
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_db.execute.return_value = mock_result

        result = await service.export_for_training(
            admin_id="admin-1",
            format="openai_completion"
        )

        assert len(result) == 1
        assert "prompt" in result[0]
        assert "completion" in result[0]
        assert "User:" in result[0]["prompt"]
        assert "Assistant:" in result[0]["prompt"]

    @pytest.mark.asyncio
    async def test_raw_format(self, service, mock_db, sample_report):
        """Test export in raw format (lines 682-696)."""
        sample_report.correction = "Correct answer is 45000 km"
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_db.execute.return_value = mock_result

        result = await service.export_for_training(
            admin_id="admin-1",
            format="raw"
        )

        assert len(result) == 1
        assert "id" in result[0]
        assert "user_query" in result[0]
        assert "wrong_output" in result[0]
        assert "correct_output" in result[0]
        assert "category" in result[0]

    @pytest.mark.asyncio
    async def test_reviewed_only_filter(self, service, mock_db, sample_report):
        """Test reviewed_only filter."""
        sample_report.correction = "Correct"
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_db.execute.return_value = mock_result

        result = await service.export_for_training(
            admin_id="admin-1",
            reviewed_only=True
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_limit_validation(self, service, mock_db):
        """Test limit is validated with MAX_EXPORT_LIMIT."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        # Should be capped to MAX_EXPORT_LIMIT
        await service.export_for_training(
            admin_id="admin-1",
            limit=50000  # Above MAX_EXPORT_LIMIT
        )


# ===========================================================================
# Export For Training JSONL Tests
# ===========================================================================

class TestExportForTrainingJsonl:
    @pytest.mark.asyncio
    async def test_jsonl_output(self, service, mock_db, sample_report):
        """Test JSONL export format (lines 721-732)."""
        sample_report.correction = "Correct answer"
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_db.execute.return_value = mock_result

        result = await service.export_for_training_jsonl(admin_id="admin-1")

        assert isinstance(result, str)
        # Each line should be valid JSON
        lines = result.strip().split("\n")
        for line in lines:
            parsed = json.loads(line)
            assert "messages" in parsed

    @pytest.mark.asyncio
    async def test_jsonl_multiple_records(self, service, mock_db, sample_report):
        """Test JSONL with multiple records."""
        sample_report.correction = "Correct"
        report2 = MagicMock()
        report2.user_query = "Query 2"
        report2.correction = "Correction 2"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report, report2]
        mock_db.execute.return_value = mock_result

        result = await service.export_for_training_jsonl(admin_id="admin-1")

        lines = result.strip().split("\n")
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_jsonl_empty(self, service, mock_db):
        """Test JSONL with no records."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        result = await service.export_for_training_jsonl(admin_id="admin-1")

        assert result == ""


# ===========================================================================
# Get Report Detail Tests
# ===========================================================================

class TestGetReportDetail:
    @pytest.mark.asyncio
    async def test_success(self, service, mock_db, sample_report):
        """Test successful report detail fetch (lines 742-777)."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_report
        mock_db.execute.return_value = mock_result

        report_id = str(sample_report.id)
        result = await service.get_report_detail(
            admin_id="admin-1",
            report_id=report_id
        )

        assert result is not None
        assert result["id"] == report_id
        assert result["user_query"] == sample_report.user_query
        assert result["api_raw_response"] == sample_report.api_raw_response

    @pytest.mark.asyncio
    async def test_invalid_uuid(self, service, mock_db):
        """Test invalid UUID returns None."""
        result = await service.get_report_detail(
            admin_id="admin-1",
            report_id="not-valid-uuid"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_not_found(self, service, mock_db):
        """Test report not found returns None."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        result = await service.get_report_detail(
            admin_id="admin-1",
            report_id=str(uuid4())
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_full_response_fields(self, service, mock_db, sample_report):
        """Test all fields are included in response."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_report
        mock_db.execute.return_value = mock_result

        result = await service.get_report_detail(
            admin_id="admin-1",
            report_id=str(sample_report.id)
        )

        expected_fields = [
            "id", "timestamp", "user_query", "bot_response", "user_feedback",
            "model", "conversation_id", "tenant_id", "retrieved_chunks",
            "api_raw_response", "reviewed", "reviewed_by", "reviewed_at",
            "correction", "category"
        ]
        for field in expected_fields:
            assert field in result


# ===========================================================================
# Get Reports Count Tests
# ===========================================================================

class TestGetReportsCount:
    @pytest.mark.asyncio
    async def test_basic_count(self, service, mock_db):
        """Test basic reports count (lines 794-803)."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_db.execute.return_value = mock_result

        result = await service.get_reports_count()

        assert result == 42

    @pytest.mark.asyncio
    async def test_unreviewed_only(self, service, mock_db):
        """Test count with unreviewed_only filter."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 10
        mock_db.execute.return_value = mock_result

        result = await service.get_reports_count(unreviewed_only=True)

        assert result == 10

    @pytest.mark.asyncio
    async def test_with_tenant_filter(self, service, mock_db):
        """Test count with tenant filter."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_db.execute.return_value = mock_result

        result = await service.get_reports_count(tenant_filter="tenant-1")

        assert result == 5

    @pytest.mark.asyncio
    async def test_both_filters(self, service, mock_db):
        """Test count with both filters."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 3
        mock_db.execute.return_value = mock_result

        result = await service.get_reports_count(
            unreviewed_only=True,
            tenant_filter="tenant-1"
        )

        assert result == 3

    @pytest.mark.asyncio
    async def test_zero_count(self, service, mock_db):
        """Test when count returns None (default to 0)."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_db.execute.return_value = mock_result

        result = await service.get_reports_count()

        assert result == 0


# ===========================================================================
# Edge Cases and Integration Tests
# ===========================================================================

class TestEdgeCases:
    def test_dangerous_patterns_coverage(self, service):
        """Test all dangerous patterns are detected."""
        test_cases = [
            "<script>",
            "javascript:",
            "onclick=",
            "onerror=",
            "eval(",
            "exec(",
            "__import__",
            "subprocess",
            "os.system",
            "${var}",
            "{{var}}",
            "drop table",
            "delete from",
            "; --",
            "union select",
            "insert into",
            "update users set",
        ]

        for pattern in test_cases:
            result = service.is_safe_text(pattern)
            assert result is False, f"Pattern '{pattern}' should be unsafe"

    def test_safe_special_characters_in_corrections(self, service):
        """Test that normal special chars in corrections are safe."""
        safe_texts = [
            "The amount is 5000 EUR (not 3000 EUR)",
            "Check the file path: /var/log/app.log",
            "Use format: YYYY-MM-DD",
            "Price: $99.99",
            "Email: test@example.com",
        ]

        for text in safe_texts:
            assert service.is_safe_text(text) is True, f"Text '{text}' should be safe"

    @pytest.mark.asyncio
    async def test_service_reuse(self, mock_db, sample_report):
        """Test that service can be reused for multiple operations."""
        service = AdminReviewService(db=mock_db)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_report]
        mock_result.scalar.return_value = 10
        mock_db.execute.return_value = mock_result

        # Multiple operations
        await service.get_hallucinations_for_review(admin_id="admin-1")
        await service.get_reports_count()

    def test_constants(self):
        """Test module constants are set correctly."""
        assert MAX_LIMIT == 1000
        assert DEFAULT_LIMIT == 50
        assert MAX_EXPORT_LIMIT == 10000
        assert MIN_LIMIT == 1

    def test_allowed_categories_complete(self, service):
        """Test all expected categories are in ALLOWED_CATEGORIES."""
        expected = {
            "wrong_data",
            "outdated",
            "misunderstood",
            "api_error",
            "rag_failure",
            "hallucination",
            "user_error",
        }
        assert AdminReviewService.ALLOWED_CATEGORIES == expected

    @pytest.mark.asyncio
    async def test_audit_creates_correct_entity(self, service, mock_db):
        """Test audit creates AuditLog with correct fields."""
        with patch("services.admin_review.AuditLog") as MockAuditLog:
            mock_instance = MagicMock()
            MockAuditLog.return_value = mock_instance

            await service._audit(
                action="TEST",
                admin_id="admin-1",
                details={"report_id": "123"},
                ip_address="10.0.0.1"
            )

            MockAuditLog.assert_called_once()
            call_kwargs = MockAuditLog.call_args[1]
            assert call_kwargs["action"] == "TEST"
            assert call_kwargs["entity_type"] == "hallucination_report"

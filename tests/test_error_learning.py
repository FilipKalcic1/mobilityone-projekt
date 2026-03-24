"""Tests for services/error_learning.py – ErrorLearningService."""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from services.error_learning import (
    ErrorLearningService,
    ErrorPattern,
    HallucinationReport,
    CorrectionRule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def svc():
    """Service with GDPR masking disabled (no external deps)."""
    return ErrorLearningService(redis_client=None, db_session=None, enable_gdpr_masking=False)


@pytest.fixture
def svc_redis():
    """Service with mocked Redis."""
    redis = AsyncMock()
    return ErrorLearningService(redis_client=redis, db_session=None, enable_gdpr_masking=False)


@pytest.fixture
def svc_db():
    """Service with mocked DB session."""
    db = AsyncMock()
    return ErrorLearningService(redis_client=None, db_session=db, enable_gdpr_masking=False)


# ===========================================================================
# DataClass Tests
# ===========================================================================

class TestErrorPattern:
    def test_defaults(self):
        ep = ErrorPattern(error_code="404", operation_id="op1",
                          error_message="not found", context={})
        assert ep.occurrence_count == 1
        assert ep.resolved is False
        assert ep.correction is None
        assert ep.last_seen  # ISO timestamp string

    def test_asdict_roundtrip(self):
        ep = ErrorPattern(error_code="500", operation_id="op2",
                          error_message="err", context={"a": 1})
        d = asdict(ep)
        assert d["error_code"] == "500"
        assert d["context"] == {"a": 1}


class TestHallucinationReport:
    def test_defaults(self):
        hr = HallucinationReport(
            timestamp="2024-01-01T00:00:00Z",
            user_query="q", bot_response="r",
            user_feedback="wrong", retrieved_chunks=["c1"],
            model="gpt-4"
        )
        assert hr.reviewed is False
        assert hr.correction is None
        assert hr.category is None

    def test_optional_fields(self):
        hr = HallucinationReport(
            timestamp="t", user_query="q", bot_response="r",
            user_feedback="bad", retrieved_chunks=[],
            model="m", conversation_id="conv1", tenant_id="t1"
        )
        assert hr.conversation_id == "conv1"


class TestCorrectionRule:
    def test_creation(self):
        cr = CorrectionRule(
            trigger_pattern="405", trigger_operation=None,
            correction_type="method",
            correction_action={"from": "POST", "to": "GET"},
            confidence=0.7
        )
        assert cr.success_count == 0
        assert cr.failure_count == 0


# ===========================================================================
# Init
# ===========================================================================

class TestInit:
    def test_default_init(self, svc):
        assert svc._total_errors == 0
        assert svc._corrected_errors == 0
        assert svc._hallucinations_reported == 0
        assert svc._false_positives_skipped == 0
        assert len(svc._correction_rules) == 3  # KNOWN_CORRECTIONS

    def test_redis_stored(self, svc_redis):
        assert svc_redis.redis is not None

    def test_gdpr_masking_disabled(self, svc):
        assert svc._gdpr_masker is None

    def test_set_drift_detector(self, svc):
        detector = MagicMock()
        svc.set_drift_detector(detector)
        assert svc._drift_detector is detector


# ===========================================================================
# _is_false_positive
# ===========================================================================

class TestIsFalsePositive:
    def test_none_status_not_fp(self, svc):
        assert svc._is_false_positive(None, None, "err") is False

    def test_non_2xx_not_fp(self, svc):
        assert svc._is_false_positive(500, None, "err") is False
        assert svc._is_false_positive(404, [], "err") is False

    def test_200_none_response_is_fp(self, svc):
        assert svc._is_false_positive(200, None, "err") is True

    def test_200_empty_list_is_fp(self, svc):
        assert svc._is_false_positive(200, [], "err") is True

    def test_200_empty_items_dict_is_fp(self, svc):
        assert svc._is_false_positive(200, {"items": []}, "err") is True
        assert svc._is_false_positive(200, {"data": []}, "err") is True

    def test_200_nonempty_not_fp(self, svc):
        assert svc._is_false_positive(200, [{"id": 1}], "err") is False

    def test_200_nonempty_dict_not_fp(self, svc):
        assert svc._is_false_positive(200, {"items": [1]}, "err") is False

    def test_204_none_is_fp(self, svc):
        # 204 No Content is 2xx
        assert svc._is_false_positive(204, None, "err") is True


# ===========================================================================
# _sanitize_context
# ===========================================================================

class TestSanitizeContext:
    def test_redacts_sensitive_keys(self, svc):
        ctx = {"token": "abc", "name": "John", "api_key": "secret", "auth_header": "xyz"}
        sanitized = svc._sanitize_context(ctx)
        assert sanitized["token"] == "[REDACTED]"
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["auth_header"] == "[REDACTED]"
        assert sanitized["name"] == "John"

    def test_empty_context(self, svc):
        assert svc._sanitize_context({}) == {}

    def test_password_redacted(self, svc):
        ctx = {"user_password": "p@ss"}
        assert svc._sanitize_context(ctx)["user_password"] == "[REDACTED]"


# ===========================================================================
# _rule_matches
# ===========================================================================

class TestRuleMatches:
    def test_matches_error_code(self, svc):
        rule = CorrectionRule(
            trigger_pattern="405", trigger_operation=None,
            correction_type="method", correction_action={}, confidence=0.7
        )
        assert svc._rule_matches(rule, "405", "any_op", "Method Not Allowed") is True

    def test_matches_error_message(self, svc):
        rule = CorrectionRule(
            trigger_pattern="required", trigger_operation=None,
            correction_type="param", correction_action={}, confidence=0.6
        )
        assert svc._rule_matches(rule, "400", "op", "field 'name' is required") is True

    def test_operation_filter_mismatch(self, svc):
        rule = CorrectionRule(
            trigger_pattern="405", trigger_operation="specific_op",
            correction_type="method", correction_action={}, confidence=0.7
        )
        assert svc._rule_matches(rule, "405", "other_op", "err") is False

    def test_operation_filter_match(self, svc):
        rule = CorrectionRule(
            trigger_pattern="405", trigger_operation="target_op",
            correction_type="method", correction_action={}, confidence=0.7
        )
        assert svc._rule_matches(rule, "405", "target_op", "err") is True

    def test_case_insensitive(self, svc):
        rule = CorrectionRule(
            trigger_pattern="HTML_RESPONSE", trigger_operation=None,
            correction_type="url", correction_action={}, confidence=0.8
        )
        assert svc._rule_matches(rule, "html_response", "op", "") is True


# ===========================================================================
# _mask_pii
# ===========================================================================

class TestMaskPII:
    def test_no_masker_returns_original(self, svc):
        assert svc._mask_pii("hello 091234567") == "hello 091234567"

    def test_empty_text_returns_empty(self, svc):
        assert svc._mask_pii("") == ""


# ===========================================================================
# record_error
# ===========================================================================

class TestRecordError:
    @pytest.mark.asyncio
    async def test_basic_record(self, svc):
        await svc.record_error("500", "get_users", "Server Error", {"param": "val"})
        assert svc._total_errors == 1
        assert "500:get_users" in svc._error_patterns
        pattern = svc._error_patterns["500:get_users"]
        assert pattern.occurrence_count == 1

    @pytest.mark.asyncio
    async def test_duplicate_increments_count(self, svc):
        await svc.record_error("500", "op1", "err", {})
        await svc.record_error("500", "op1", "err again", {})
        assert svc._error_patterns["500:op1"].occurrence_count == 2

    @pytest.mark.asyncio
    async def test_correction_marks_resolved(self, svc):
        await svc.record_error("405", "op1", "err", {},
                               was_corrected=True, correction="use GET")
        p = svc._error_patterns["405:op1"]
        assert p.resolved is True
        assert p.correction == "use GET"

    @pytest.mark.asyncio
    async def test_false_positive_skipped(self, svc):
        await svc.record_error("200", "op1", "empty", {},
                               http_status=200, response_data=[])
        assert svc._total_errors == 0
        assert svc._false_positives_skipped == 1
        assert "200:op1" not in svc._error_patterns

    @pytest.mark.asyncio
    async def test_redis_persist_called(self, svc_redis):
        await svc_redis.record_error("500", "op1", "err", {})
        svc_redis.redis.hset.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_failure_resilient(self, svc_redis):
        svc_redis.redis.hset.side_effect = Exception("conn lost")
        await svc_redis.record_error("500", "op1", "err", {})
        assert svc_redis._total_errors == 1  # still recorded in memory

    @pytest.mark.asyncio
    async def test_context_sanitized(self, svc):
        await svc.record_error("400", "op1", "err", {"token": "secret123", "user": "john"})
        p = svc._error_patterns["400:op1"]
        assert p.context["token"] == "[REDACTED]"
        assert p.context["user"] == "john"


# ===========================================================================
# suggest_correction
# ===========================================================================

class TestSuggestCorrection:
    @pytest.mark.asyncio
    async def test_405_suggests_method_change(self, svc):
        result = await svc.suggest_correction("405", "op1", "Method Not Allowed", {})
        assert result is not None
        assert result["type"] == "method"
        assert result["confidence"] >= 0.6

    @pytest.mark.asyncio
    async def test_required_suggests_param_injection(self, svc):
        result = await svc.suggest_correction("400", "op1", "field is required", {})
        assert result is not None
        assert result["type"] == "param"

    @pytest.mark.asyncio
    async def test_html_suggests_url_check(self, svc):
        result = await svc.suggest_correction("HTML_RESPONSE", "op1", "got HTML", {})
        assert result is not None
        assert result["type"] == "url"

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self, svc):
        result = await svc.suggest_correction("999", "op1", "unknown error", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_historical_pattern_match(self, svc):
        # Record a resolved error first
        svc._error_patterns["ERR:op1"] = ErrorPattern(
            error_code="ERR", operation_id="op1",
            error_message="some err", context={},
            correction="fix_X", resolved=True, occurrence_count=5
        )
        result = await svc.suggest_correction("ERR", "op1", "some err", {})
        assert result is not None
        assert result["type"] == "historical"
        assert result["confidence"] <= 0.9

    @pytest.mark.asyncio
    async def test_low_confidence_rule_skipped(self, svc):
        # Add a low-confidence rule
        svc._learned_rules.append(CorrectionRule(
            trigger_pattern="low_err", trigger_operation=None,
            correction_type="param", correction_action={},
            confidence=0.3  # below 0.6 threshold
        ))
        result = await svc.suggest_correction("low_err", "op1", "low_err msg", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_increments_pattern_matches(self, svc):
        await svc.suggest_correction("xxx", "op1", "err", {})
        assert svc._pattern_matches == 1


# ===========================================================================
# apply_correction
# ===========================================================================

class TestApplyCorrection:
    @pytest.mark.asyncio
    async def test_method_correction(self, svc):
        corr = {"type": "method", "action": {"from": "POST", "to": "GET"}}
        result = await svc.apply_correction(corr, None, {}, {})
        assert result["hint"] == "use_get_instead_of_post"
        assert result["suggested_method"] == "GET"

    @pytest.mark.asyncio
    async def test_param_inject_from_context(self, svc):
        corr = {"type": "param", "action": {"action": "inject_from_context"}}
        ctx = {"person_id": 123, "tenant_id": "T1"}
        result = await svc.apply_correction(corr, None, {}, ctx)
        assert result["params"]["person_id"] == 123
        assert result["params"]["tenant_id"] == "T1"

    @pytest.mark.asyncio
    async def test_param_no_overwrite_existing(self, svc):
        corr = {"type": "param", "action": {"action": "inject_from_context"}}
        ctx = {"person_id": 999}
        params = {"person_id": 1}
        result = await svc.apply_correction(corr, None, params, ctx)
        assert result["params"]["person_id"] == 1  # not overwritten

    @pytest.mark.asyncio
    async def test_url_correction(self, svc):
        corr = {"type": "url", "action": {"action": "verify_swagger_name"}}
        result = await svc.apply_correction(corr, None, {}, {})
        assert result["hint"] == "check_swagger_name"

    @pytest.mark.asyncio
    async def test_historical_correction(self, svc):
        corr = {"type": "historical", "action": {"correction": "use param X"}}
        result = await svc.apply_correction(corr, None, {}, {})
        assert result["hint"] == "use param X"

    @pytest.mark.asyncio
    async def test_unknown_type_returns_none(self, svc):
        corr = {"type": "unknown"}
        result = await svc.apply_correction(corr, None, {}, {})
        assert result is None


# ===========================================================================
# report_correction_result
# ===========================================================================

class TestReportCorrectionResult:
    @pytest.mark.asyncio
    async def test_success_increases_confidence(self, svc):
        original_conf = svc._correction_rules[0].confidence  # "405" rule
        await svc.report_correction_result({"rule_pattern": "405"}, success=True)
        assert svc._correction_rules[0].confidence == min(original_conf + 0.05, 0.95)
        assert svc._corrected_errors == 1
        assert svc._correction_rules[0].success_count == 1

    @pytest.mark.asyncio
    async def test_failure_decreases_confidence(self, svc):
        original_conf = svc._correction_rules[0].confidence
        await svc.report_correction_result({"rule_pattern": "405"}, success=False)
        assert svc._correction_rules[0].confidence == max(original_conf - 0.1, 0.1)
        assert svc._correction_rules[0].failure_count == 1

    @pytest.mark.asyncio
    async def test_confidence_max_cap(self, svc):
        svc._correction_rules[0].confidence = 0.93
        await svc.report_correction_result({"rule_pattern": "405"}, success=True)
        assert svc._correction_rules[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_confidence_min_cap(self, svc):
        svc._correction_rules[0].confidence = 0.15
        await svc.report_correction_result({"rule_pattern": "405"}, success=False)
        assert svc._correction_rules[0].confidence == 0.1

    @pytest.mark.asyncio
    async def test_unknown_pattern_no_crash(self, svc):
        await svc.report_correction_result({"rule_pattern": "nonexistent"}, success=True)
        # No exception raised


# ===========================================================================
# _analyze_patterns (learning)
# ===========================================================================

class TestAnalyzePatterns:
    @pytest.mark.asyncio
    async def test_learns_rule_from_resolved_patterns(self, svc):
        svc._error_patterns["E1:op1"] = ErrorPattern(
            error_code="E1", operation_id="op1",
            error_message="err", context={},
            correction="do_fix", resolved=True, occurrence_count=5
        )
        await svc._analyze_patterns()
        assert len(svc._learned_rules) == 1
        assert svc._learned_rules[0].trigger_pattern == "E1"
        assert svc._learned_rules[0].trigger_operation == "op1"

    @pytest.mark.asyncio
    async def test_no_learn_if_count_below_3(self, svc):
        svc._error_patterns["E1:op1"] = ErrorPattern(
            error_code="E1", operation_id="op1",
            error_message="err", context={},
            correction="fix", resolved=True, occurrence_count=2
        )
        await svc._analyze_patterns()
        assert len(svc._learned_rules) == 0

    @pytest.mark.asyncio
    async def test_no_duplicate_rules(self, svc):
        svc._error_patterns["E1:op1"] = ErrorPattern(
            error_code="E1", operation_id="op1",
            error_message="err", context={},
            correction="fix", resolved=True, occurrence_count=5
        )
        await svc._analyze_patterns()
        await svc._analyze_patterns()  # second time
        assert len(svc._learned_rules) == 1

    @pytest.mark.asyncio
    async def test_confidence_scales_with_count(self, svc):
        svc._error_patterns["E1:op1"] = ErrorPattern(
            error_code="E1", operation_id="op1",
            error_message="err", context={},
            correction="fix", resolved=True, occurrence_count=10
        )
        await svc._analyze_patterns()
        rule = svc._learned_rules[0]
        assert rule.confidence == 0.6 + min(10 * 0.05, 0.3)  # 0.9


# ===========================================================================
# record_hallucination
# ===========================================================================

class TestRecordHallucination:
    @pytest.mark.asyncio
    async def test_basic_record(self, svc):
        result = await svc.record_hallucination(
            user_query="What is limit?",
            bot_response="5000 EUR",
            user_feedback="krivo",
            retrieved_chunks=["chunk1"],
            model="gpt-4",
            conversation_id="conv1"
        )
        assert result["recorded"] is True
        assert "report_id" in result
        assert "follow_up_question" in result
        assert svc._hallucinations_reported == 1
        assert len(svc._hallucination_reports) == 1

    @pytest.mark.asyncio
    async def test_short_feedback_generates_question(self, svc):
        result = await svc.record_hallucination(
            user_query="q", bot_response="r",
            user_feedback="ne",
            retrieved_chunks=[], model="m"
        )
        assert "što je točno bilo pogrešno" in result["follow_up_question"]

    @pytest.mark.asyncio
    async def test_long_feedback_generates_thanks(self, svc):
        result = await svc.record_hallucination(
            user_query="q", bot_response="r",
            user_feedback="The amount shown was incorrect, it should be 3000 EUR not 5000.",
            retrieved_chunks=[], model="m"
        )
        assert "Hvala" in result["follow_up_question"]

    @pytest.mark.asyncio
    async def test_redis_persist_called(self, svc_redis):
        await svc_redis.record_hallucination(
            user_query="q", bot_response="r",
            user_feedback="wrong", retrieved_chunks=[], model="m",
            conversation_id="c1"
        )
        svc_redis.redis.hset.assert_called_once()

    @pytest.mark.asyncio
    async def test_drift_detector_notified(self, svc):
        detector = AsyncMock()
        svc.set_drift_detector(detector)
        await svc.record_hallucination(
            user_query="q", bot_response="r",
            user_feedback="wrong", retrieved_chunks=[], model="m",
            tenant_id="T1"
        )
        detector.record_interaction.assert_called_once_with(
            model_version="m", latency_ms=0, success=True,
            hallucination_reported=True, tenant_id="T1"
        )

    @pytest.mark.asyncio
    async def test_db_persist_called(self, svc_db):
        with patch("services.hallucination_repository.HallucinationRepository") as MockRepo:
            mock_repo_inst = AsyncMock()
            mock_report = MagicMock()
            mock_report.id = 42
            mock_repo_inst.create.return_value = mock_report
            MockRepo.return_value = mock_repo_inst

            result = await svc_db.record_hallucination(
                user_query="q", bot_response="r",
                user_feedback="wrong", retrieved_chunks=[], model="m",
                conversation_id="c1"
            )
            assert result["report_id"] == "42"

    @pytest.mark.asyncio
    async def test_db_error_resilient(self, svc_db):
        with patch("services.hallucination_repository.HallucinationRepository", side_effect=Exception("db err")):
            result = await svc_db.record_hallucination(
                user_query="q", bot_response="r",
                user_feedback="wrong", retrieved_chunks=[], model="m"
            )
            assert result["recorded"] is True  # still succeeds


# ===========================================================================
# _generate_hallucination_followup
# ===========================================================================

class TestGenerateFollowup:
    def test_short_feedback(self, svc):
        result = svc._generate_hallucination_followup("krivo", "some response")
        assert "što je točno" in result

    def test_long_feedback(self, svc):
        result = svc._generate_hallucination_followup(
            "Iznos od 5000 EUR je pogrešan, trebao je biti 3000 EUR", "response"
        )
        assert "Hvala" in result

    def test_exactly_20_chars_is_long(self, svc):
        feedback = "x" * 20  # 20 chars
        result = svc._generate_hallucination_followup(feedback, "r")
        assert "Hvala" in result


# ===========================================================================
# _persist_pattern / _persist_hallucination
# ===========================================================================

class TestPersistPattern:
    @pytest.mark.asyncio
    async def test_no_redis_noop(self, svc):
        await svc._persist_pattern("key", ErrorPattern(
            error_code="e", operation_id="o", error_message="m", context={}
        ))
        # No exception

    @pytest.mark.asyncio
    async def test_redis_hset_called(self, svc_redis):
        ep = ErrorPattern(error_code="e", operation_id="o",
                          error_message="m", context={})
        await svc_redis._persist_pattern("key", ep)
        svc_redis.redis.hset.assert_called_once()
        call_args = svc_redis.redis.hset.call_args
        assert call_args[0][0] == "error_learning:patterns"
        assert call_args[0][1] == "key"

    @pytest.mark.asyncio
    async def test_redis_error_handled(self, svc_redis):
        svc_redis.redis.hset.side_effect = Exception("fail")
        await svc_redis._persist_pattern("key", ErrorPattern(
            error_code="e", operation_id="o", error_message="m", context={}
        ))  # No exception


class TestPersistHallucination:
    @pytest.mark.asyncio
    async def test_no_redis_noop(self, svc):
        report = HallucinationReport(
            timestamp="t", user_query="q", bot_response="r",
            user_feedback="f", retrieved_chunks=[], model="m",
            conversation_id="c1"
        )
        await svc._persist_hallucination(report)  # No exception

    @pytest.mark.asyncio
    async def test_redis_error_handled(self, svc_redis):
        svc_redis.redis.hset.side_effect = Exception("fail")
        report = HallucinationReport(
            timestamp="t", user_query="q", bot_response="r",
            user_feedback="f", retrieved_chunks=[], model="m",
            conversation_id="c1"
        )
        await svc_redis._persist_hallucination(report)  # No exception


# ===========================================================================
# save_to_file
# ===========================================================================

class TestSaveToFile:
    def test_save_creates_json(self, svc, tmp_path):
        with patch("services.error_learning.ERROR_LEARNING_CACHE_FILE",
                    tmp_path / "error_learning.json"):
            svc._error_patterns["E1:op1"] = ErrorPattern(
                error_code="E1", operation_id="op1",
                error_message="err", context={}, occurrence_count=3
            )
            svc._hallucination_reports.append(HallucinationReport(
                timestamp="t", user_query="q", bot_response="r",
                user_feedback="f", retrieved_chunks=[], model="m"
            ))
            svc.save_to_file()

            data = json.loads((tmp_path / "error_learning.json").read_text(encoding="utf-8"))
            assert data["version"] == "2.0"
            assert len(data["patterns"]) == 1
            assert len(data["hallucinations"]) == 1
            assert data["statistics"]["total_errors"] == 0

    def test_save_error_resilient(self, svc):
        with patch("services.error_learning.ERROR_LEARNING_CACHE_FILE",
                    MagicMock(parent=MagicMock(mkdir=MagicMock(side_effect=PermissionError)))):
            svc.save_to_file()  # No exception raised

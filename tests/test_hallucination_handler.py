"""Tests for services/engine/hallucination_handler.py – HallucinationHandler."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.engine.hallucination_handler import HallucinationHandler, HALLUCINATION_PATTERNS


def _user_context():
    return {
        "person_id": "00000000-0000-0000-0000-000000000001",
        "tenant_id": "t1",
    }


def _conv():
    conv = MagicMock()
    conv.context = MagicMock()
    conv.context.current_tool = "get_MasterData"
    return conv


@pytest.fixture
def handler():
    ctx_service = MagicMock()
    ctx_service.get_history = AsyncMock(return_value=[])
    ctx_service.add_message = AsyncMock()
    error_learning = AsyncMock()
    error_learning.record_hallucination = AsyncMock(return_value={})
    error_learning.save_to_file = MagicMock()
    return HallucinationHandler(ctx_service, error_learning, "gpt-4")


class TestPatterns:
    def test_patterns_exist(self):
        assert len(HALLUCINATION_PATTERNS) > 5

    def test_krivo_in_patterns(self):
        assert "krivo" in HALLUCINATION_PATTERNS

    def test_nije_tocno_in_patterns(self):
        assert "nije točno" in HALLUCINATION_PATTERNS or "nije tocno" in HALLUCINATION_PATTERNS


class TestInit:
    def test_attributes(self, handler):
        assert handler.context is not None
        assert handler.error_learning is not None
        assert handler.ai_model == "gpt-4"


class TestCheckHallucinationFeedback:
    @pytest.mark.asyncio
    async def test_no_feedback_regular_text(self, handler):
        result = await handler.check_hallucination_feedback(
            "koliko km ima auto?", "sender", _user_context(), _conv()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_detects_krivo(self, handler):
        handler.context.get_history = AsyncMock(return_value=[
            {"role": "user", "content": "koliko km?"},
            {"role": "assistant", "content": "100000 km"},
        ])
        handler.error_learning.record_hallucination = AsyncMock(
            return_value={"follow_up_question": "Koja je ispravna kilometraža?"}
        )

        result = await handler.check_hallucination_feedback(
            "krivo", "sender", _user_context(), _conv()
        )
        assert result is not None
        assert "ispravna" in result

    @pytest.mark.asyncio
    async def test_detects_nije_tocno(self, handler):
        handler.context.get_history = AsyncMock(return_value=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "ans"},
        ])
        handler.error_learning.record_hallucination = AsyncMock(
            return_value={"follow_up_question": "Sto je tocno?"}
        )

        result = await handler.check_hallucination_feedback(
            "nije tocno", "sender", _user_context(), _conv()
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_long_message_ignored(self, handler):
        long_text = "krivo " + "x" * 200
        result = await handler.check_hallucination_feedback(
            long_text, "sender", _user_context(), _conv()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_history(self, handler):
        handler.context.get_history = AsyncMock(return_value=[])

        result = await handler.check_hallucination_feedback(
            "krivo", "sender", _user_context(), _conv()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_follow_up(self, handler):
        handler.context.get_history = AsyncMock(return_value=[
            {"role": "assistant", "content": "wrong answer"},
        ])
        handler.error_learning.record_hallucination = AsyncMock(
            return_value={"follow_up_question": ""}
        )

        result = await handler.check_hallucination_feedback(
            "greska", "sender", _user_context(), _conv()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_handled(self, handler):
        handler.context.get_history = AsyncMock(side_effect=RuntimeError("boom"))

        result = await handler.check_hallucination_feedback(
            "krivo", "sender", _user_context(), _conv()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_records_hallucination_with_tool_context(self, handler):
        handler.context.get_history = AsyncMock(return_value=[
            {"role": "user", "content": "km?"},
            {"role": "assistant", "content": "50000"},
        ])
        handler.error_learning.record_hallucination = AsyncMock(
            return_value={"follow_up_question": "Koliko je zapravo?"}
        )

        conv = _conv()
        conv.context.current_tool = "get_MasterData"

        result = await handler.check_hallucination_feedback(
            "pogrešno", "sender", _user_context(), conv
        )
        assert result is not None
        handler.error_learning.record_hallucination.assert_called_once()
        call_args = handler.error_learning.record_hallucination.call_args
        assert "get_MasterData" in call_args.kwargs.get("retrieved_chunks", [])

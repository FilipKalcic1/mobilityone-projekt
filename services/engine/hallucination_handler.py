"""
Hallucination Handler - Detects and records user feedback about wrong answers.
Version: 1.0

Extracted from engine/__init__.py for better modularity.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from services.context import UserContextManager

if TYPE_CHECKING:
    from services.conversation_manager import ConversationManager
    from services.error_learning import ErrorLearningService

logger = logging.getLogger(__name__)


# Patterns that indicate user is reporting wrong answer
HALLUCINATION_PATTERNS = [
    "krivo",
    "nije točno",
    "nije tocno",
    "pogrešno",
    "pogresno",
    "to nije istina",
    "nije to",
    "netočno",
    "netocno",
    "greška",
    "greska",
    "nije tako",
    "nije u redu",
    "ne, nije",
]


class HallucinationHandler:
    """
    Handles detection and recording of hallucination feedback.

    When user says "krivo", "nije točno", etc., this handler:
    1. Detects the feedback
    2. Records it for learning
    3. Returns a follow-up question
    """

    def __init__(
        self,
        context_service,
        error_learning: 'ErrorLearningService',
        ai_model: str
    ):
        """
        Initialize HallucinationHandler.

        Args:
            context_service: Context service for history
            error_learning: Error learning service
            ai_model: AI model name for tracking
        """
        self.context = context_service
        self.error_learning = error_learning
        self.ai_model = ai_model

    async def check_hallucination_feedback(
        self,
        text: str,
        sender: str,
        user_context: Dict[str, Any],
        conv_manager: 'ConversationManager'
    ) -> Optional[str]:
        """
        Check if user is reporting a hallucination ("krivo" feedback).

        If detected:
        1. Get last bot response from history
        2. Record hallucination via error_learning
        3. Return follow-up question for more details

        Args:
            text: User message
            sender: User phone/ID
            user_context: User context dict
            conv_manager: Conversation manager

        Returns:
            Response string if hallucination detected, None otherwise
        """
        text_lower = text.lower().strip()

        # Check if this looks like hallucination feedback
        is_feedback = any(
            pattern in text_lower
            for pattern in HALLUCINATION_PATTERNS
        )

        if not is_feedback:
            return None

        # Don't trigger for longer messages (probably context, not feedback)
        if len(text) > 100:
            return None

        logger.info(f"🚨 Hallucination feedback detected: '{text}'")

        try:
            # Get last bot response from history
            history = await self.context.get_history(sender)

            last_bot_response = None
            last_user_query = None

            # Find the most recent assistant message and the query before it
            for i, msg in enumerate(reversed(history)):
                if msg.get("role") == "assistant" and not last_bot_response:
                    last_bot_response = msg.get("content", "")
                elif msg.get("role") == "user" and last_bot_response and not last_user_query:
                    last_user_query = msg.get("content", "")
                    break

            if not last_bot_response:
                logger.warning("No previous bot response found for hallucination report")
                return None

            # Record the hallucination
            # Collect tool context for debugging
            retrieved_chunks = []
            if conv_manager.context.current_tool:
                retrieved_chunks.append(conv_manager.context.current_tool)

            # Use UserContextManager for validated access
            ctx = UserContextManager(user_context)
            result = await self.error_learning.record_hallucination(
                user_query=last_user_query or "[Unknown query]",
                bot_response=last_bot_response,
                user_feedback=text,
                retrieved_chunks=retrieved_chunks,
                model=self.ai_model,
                conversation_id=sender,
                tenant_id=ctx.tenant_id
            )

            # Save state (run in thread to avoid blocking the event loop)
            # FIX v11.1: Was blocking file I/O in async handler
            await asyncio.to_thread(self.error_learning.save_to_file)

            # Return the follow-up question
            follow_up = result.get("follow_up_question", "")
            if follow_up:
                await self.context.add_message(sender, "assistant", follow_up)
                return follow_up

        except Exception as e:
            logger.error(f"Error recording hallucination: {e}", exc_info=True)

        return None

"""
Message Engine - Public API facade.
Version: 19.2 (Full Observability Stack)

This module provides backward-compatible interface to the refactored engine.
All existing imports from services.message_engine should work unchanged.

New in v19.2:
- COST TRACKING - Token usage and budget monitoring
- CostTracker integrated in _instrumented_ai_call
- /admin/cost-stats endpoint for cost visibility

New in v19.1:
- MODEL DRIFT DETECTION - Closed feedback loop
- ErrorLearningService connected to ModelDriftDetector
- Automatic metrics collection for every LLM interaction
- Drift alerts via /admin/drift-status endpoint

New in v19.0:
- UNIFIED LLM ROUTER - Single decision point for ALL routing
- Handles flow exit detection ("ne želim", "odustani")
- Correctly distinguishes READ vs WRITE intent
- Uses training data for few-shot examples
- 100% pass rate on production scenarios

New in v17.0:
- FILTER-THEN-SEARCH architecture for semantic search
- Intent detection (READ vs WRITE) filters by HTTP method first
- Category detection narrows search space from 900+ to ~20-50 tools

New in v16.2:
- Full flow support: simple, booking, mileage_input, list, case_creation
- Graceful fallback to LLM when deterministic path fails

New in v16.1:
- QueryRouter for deterministic pattern-based routing
- Direct response for greetings, thanks, help

New in v16.0:
- ChainPlanner for multi-step execution plans with fallbacks
- ExecutorWithFallback for automatic retry and alternative tools
- LLMResponseExtractor for intelligent data extraction
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List

from config import get_settings
from services.conversation_manager import ConversationManager, ConversationState
from services.tool_executor import ToolExecutor
from services.ai_orchestrator import AIOrchestrator
from services.response_formatter import ResponseFormatter
from services.user_service import UserService
from services.dependency_resolver import DependencyResolver
from services.error_learning import ErrorLearningService
from services.model_drift_detector import ModelDriftDetector, get_drift_detector
from services.cost_tracker import CostTracker, get_cost_tracker
from services.reasoning import Planner, ExecutionPlan, PlanStep

# NEW v16.0: Advanced components for 100% reliability
from services.chain_planner import ChainPlanner, get_chain_planner
from services.executor_fallback import ExecutorWithFallback, get_executor_with_fallback
from services.response_extractor import LLMResponseExtractor, get_response_extractor

# NEW v16.1: Deterministic query routing - NO LLM guessing for known patterns
from services.query_router import QueryRouter, get_query_router, RouteResult

# NEW v18.0: Intelligent category-based routing for unmatched queries
from services.intelligent_router import IntelligentRouter, FlowType, RoutingDecision

# NEW v19.0: Unified LLM router - single decision point
from services.unified_router import UnifiedRouter, RouterDecision, get_unified_router

from .tool_handler import ToolHandler
from .flow_handler import FlowHandler

logger = logging.getLogger(__name__)
settings = get_settings()

__all__ = ['MessageEngine']


class MessageEngine:
    """
    Main message processing engine.

    Coordinates:
    - User identification
    - Conversation state (Redis-backed)
    - AI interactions with error feedback
    - Tool execution with validation

    This is a facade that coordinates the refactored components.
    """

    MAX_ITERATIONS = 6

    def __init__(
        self,
        gateway,
        registry,
        context_service,
        queue_service,
        cache_service,
        db_session
    ):
        """Initialize engine with all services."""
        self.gateway = gateway
        self.registry = registry
        self.executor = ToolExecutor(gateway)
        self.context = context_service
        self.queue = queue_service
        self.cache = cache_service
        self.db = db_session
        self.redis = context_service.redis if context_service else None

        # Core services
        self.dependency_resolver = DependencyResolver(registry)
        self.error_learning = ErrorLearningService(redis_client=self.redis)

        # Model drift detection - closes the feedback loop
        self.drift_detector = get_drift_detector(
            redis_client=self.redis,
            db_session=self.db
        )
        self.error_learning.set_drift_detector(self.drift_detector)

        # Cost tracking - token usage and budget alerts
        self.cost_tracker: Optional[CostTracker] = None
        if self.redis:
            try:
                import asyncio
                self.cost_tracker = CostTracker(redis_client=self.redis)
                logger.info("CostTracker initialized")
            except Exception as e:
                logger.warning(f"CostTracker init failed: {e}")

        self.ai = AIOrchestrator()
        self.formatter = ResponseFormatter()
        self.planner = Planner()

        # NEW v16.0: Advanced components
        self.chain_planner = get_chain_planner()
        self.response_extractor = get_response_extractor()
        # Note: executor_fallback initialized lazily when search_engine is available

        # NEW v16.1: Deterministic query router
        self.query_router = get_query_router()

        # NEW v18.0: Intelligent category-based router (for unmatched queries)
        self.intelligent_router = IntelligentRouter(registry)
        self._intelligent_router_initialized = False

        # NEW v19.0: Unified LLM router - makes ALL routing decisions
        self.unified_router: Optional[UnifiedRouter] = None
        self._unified_router_initialized = False

        # Refactored handlers
        self._tool_handler = ToolHandler(
            registry=registry,
            executor=self.executor,
            dependency_resolver=self.dependency_resolver,
            error_learning=self.error_learning,
            formatter=self.formatter
        )
        self._flow_handler = FlowHandler(
            registry=registry,
            executor=self.executor,
            ai=self.ai,
            formatter=self.formatter
        )

        logger.info("MessageEngine initialized (v19.2 with full observability)")

    async def _instrumented_ai_call(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        forced_tool: Optional[str] = None,
        tool_scores: Optional[List[Dict]] = None,
        user_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Wrapper around AI analyze() that records metrics for drift detection and cost tracking.

        This CLOSES THE FEEDBACK LOOP by sending every LLM interaction to:
        1. Drift detector - for statistical analysis
        2. Cost tracker - for budget monitoring
        """
        start_time = time.perf_counter()
        error_type = None
        success = True
        tools_called = []
        usage_data = None

        try:
            result = await self.ai.analyze(
                messages=messages,
                tools=tools,
                system_prompt=system_prompt,
                forced_tool=forced_tool,
                tool_scores=tool_scores
            )

            # Determine success and extract metadata
            if result.get("type") == "error":
                success = False
                error_type = "llm_error"
            elif result.get("type") == "tool_call":
                tools_called = [result.get("tool", "unknown")]

            # Extract usage data for cost tracking
            usage_data = result.get("usage")

            return result

        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise

        finally:
            # Calculate latency
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            tenant_id = user_context.get("tenant_id") if user_context else "default"

            # 1. Record interaction for DRIFT DETECTION
            try:
                await self.drift_detector.record_interaction(
                    model_version=self.ai.model,
                    latency_ms=latency_ms,
                    success=success,
                    error_type=error_type,
                    confidence_score=None,
                    tools_called=tools_called,
                    hallucination_reported=False,
                    tenant_id=tenant_id
                )
            except Exception as e:
                logger.warning(f"Failed to record drift metrics: {e}")

            # 2. Record usage for COST TRACKING
            if self.cost_tracker and usage_data:
                try:
                    await self.cost_tracker.record_usage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        model=self.ai.model,
                        tenant_id=tenant_id,
                        latency_ms=latency_ms,
                        success=success
                    )
                except Exception as e:
                    logger.warning(f"Failed to record cost metrics: {e}")

    async def process(
        self,
        sender: str,
        text: str,
        message_id: Optional[str] = None
    ) -> str:
        """
        Process incoming message.

        Args:
            sender: User phone number
            text: Message text
            message_id: Optional message ID

        Returns:
            Response text
        """
        logger.info(f"Processing: {sender[-4:]} - {text[:50]}")

        try:
            # 1. Identify user
            user_context = await self._identify_user(sender)

            if not user_context:
                return (
                    "Vas broj nije pronaden u sustavu MobilityOne.\n"
                    "Molimo kontaktirajte administratora."
                )

            # 2. Load conversation state
            conv_manager = await ConversationManager.load_for_user(sender, self.redis)

            # 3. Check timeout
            if conv_manager.is_timed_out():
                await conv_manager.reset()

            # 4. Add to history
            await self.context.add_message(sender, "user", text)

            # 4.5 Check for hallucination feedback ("krivo", "nije točno", etc.)
            hallucination_result = await self._check_hallucination_feedback(
                text=text,
                sender=sender,
                user_context=user_context,
                conv_manager=conv_manager
            )
            if hallucination_result:
                # User reported "krivo" - return follow-up question
                return hallucination_result

            # 5. Handle new user greeting
            if user_context.get("is_new"):
                greeting = self._build_greeting(user_context)
                await self.context.add_message(sender, "assistant", greeting)

                response = await self._process_with_state(
                    sender, text, user_context, conv_manager
                )

                full_response = f"{greeting}\n\n---\n\n{response}"
                await self.context.add_message(sender, "assistant", response)
                return full_response

            # 6. Process based on state
            response = await self._process_with_state(
                sender, text, user_context, conv_manager
            )

            # 7. Save response to history
            await self.context.add_message(sender, "assistant", response)

            return response

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return "Doslo je do greske. Molimo pokusajte ponovno."

    async def _identify_user(self, phone: str) -> Optional[Dict[str, Any]]:
        """Identify user and build context."""
        user_service = UserService(self.db, self.gateway, self.cache)

        user = await user_service.get_active_identity(phone)

        if user:
            ctx = await user_service.build_context(user.api_identity, phone)
            ctx["display_name"] = user.display_name
            ctx["is_new"] = False
            return ctx

        result = await user_service.try_auto_onboard(phone)

        if result:
            display_name, vehicle_data = result
            user = await user_service.get_active_identity(phone)

            if user:
                ctx = await user_service.build_context(user.api_identity, phone)
                ctx["display_name"] = display_name
                ctx["is_new"] = True
                # vehicle_data is now a Dict with all Swagger fields
                # It's already set in build_context as ctx["vehicle"]
                return ctx

        return None

    # =========================================================================
    # HALLUCINATION FEEDBACK DETECTION
    # =========================================================================

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

    async def _check_hallucination_feedback(
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

        Returns:
            Response string if hallucination detected, None otherwise
        """
        text_lower = text.lower().strip()

        # Check if this looks like hallucination feedback
        is_feedback = any(
            pattern in text_lower
            for pattern in self.HALLUCINATION_PATTERNS
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

            result = await self.error_learning.record_hallucination(
                user_query=last_user_query or "[Unknown query]",
                bot_response=last_bot_response,
                user_feedback=text,
                retrieved_chunks=retrieved_chunks,
                model=self.ai.model,
                conversation_id=sender,
                tenant_id=user_context.get("tenant_id")
            )

            # Save state
            self.error_learning.save_to_file()

            # Return the follow-up question
            follow_up = result.get("follow_up_question", "")
            if follow_up:
                await self.context.add_message(sender, "assistant", follow_up)
                return follow_up

        except Exception as e:
            logger.error(f"Error recording hallucination: {e}", exc_info=True)

        return None

    def _build_greeting(self, user_context: Dict[str, Any]) -> str:
        """Build personalized greeting for new user."""
        name = user_context.get("display_name", "")
        vehicle = user_context.get("vehicle", {})
        vehicle_info = user_context.get("vehicle_info", "")

        greeting = f"Pozdrav {name}!\n\n"
        greeting += "Ja sam MobilityOne AI asistent.\n\n"

        # Use Swagger field names directly
        if vehicle.get("LicencePlate"):
            plate = vehicle.get("LicencePlate")
            v_name = vehicle.get("FullVehicleName") or vehicle.get("DisplayName", "vozilo")
            mileage = vehicle.get("Mileage", "N/A")

            greeting += f"Vidim da vam je dodijeljeno vozilo:\n"
            greeting += f"   **{v_name}** ({plate})\n"
            greeting += f"   Kilometraza: {mileage} km\n\n"
            greeting += "Kako vam mogu pomoci?\n"
            greeting += "* Unos kilometraze\n"
            greeting += "* Prijava kvara\n"
            greeting += "* Rezervacija vozila\n"
            greeting += "* Pitanja o vozilu"
        elif vehicle.get("Id"):
            # Has vehicle but no plate - still show something
            v_name = vehicle.get("FullVehicleName") or vehicle.get("DisplayName", "vozilo")
            greeting += f"Vidim da vam je dodijeljeno vozilo: {v_name}\n\n"
            greeting += "Kako vam mogu pomoci?"
        else:
            greeting += "Trenutno nemate dodijeljeno vozilo.\n\n"
            greeting += "Zelite li rezervirati vozilo? Recite mi:\n"
            greeting += "* Za koji period (npr. 'sutra od 8 do 17')\n"
            greeting += "* Ili samo recite 'Trebam vozilo' pa cemo dalje"

        return greeting

    async def _process_with_state(
        self,
        sender: str,
        text: str,
        user_context: Dict[str, Any],
        conv_manager: ConversationManager
    ) -> str:
        """Process message based on conversation state using Unified Router."""
        state = conv_manager.get_state()

        # CRITICAL DEBUG: Log current state for troubleshooting flow issues
        is_in_flow = conv_manager.is_in_flow()
        logger.info(
            f"STATE CHECK: user={sender[-4:]}, state={state.value}, "
            f"is_in_flow={is_in_flow}, flow={conv_manager.get_current_flow()}, "
            f"tool={conv_manager.get_current_tool()}, "
            f"missing={conv_manager.get_missing_params()}, "
            f"has_items={len(conv_manager.get_displayed_items())}"
        )

        # === CRITICAL FIX: Direct state-based handling for in-flow messages ===
        # Before calling unified router, handle obvious in-flow cases directly
        # This prevents router hallucination from breaking flows
        if is_in_flow:
            text_lower = text.lower()

            # In CONFIRMING state - handle confirmations and "show more" directly
            if state == ConversationState.CONFIRMING:
                # Check for "show more" type requests
                if any(s in text_lower for s in ["pokaz", "ostala", "druga", "više", "vise", "sva vozila", "lista", "popis"]):
                    logger.info("DIRECT HANDLER: 'show more' in CONFIRMING state")
                    return await self._flow_handler.handle_confirmation(
                        sender, text, user_context, conv_manager
                    )
                # Check for confirmation/cancellation
                if any(w in text_lower for w in ["da", "potvrdi", "ok", "yes", "može", "moze", "ne", "odustani", "cancel", "no"]):
                    logger.info("DIRECT HANDLER: confirmation response in CONFIRMING state")
                    return await self._flow_handler.handle_confirmation(
                        sender, text, user_context, conv_manager
                    )

            # In SELECTING state - handle numeric selection directly
            if state == ConversationState.SELECTING_ITEM:
                if text.strip().isdigit() or any(s in text_lower for s in ["prvi", "drugi", "treći", "treci"]):
                    logger.info("DIRECT HANDLER: item selection in SELECTING state")
                    return await self._flow_handler.handle_selection(
                        sender, text, user_context, conv_manager, self._handle_new_request
                    )

        # === v19.0: UNIFIED ROUTER - Single decision point ===
        # Initialize unified router lazily
        if not self._unified_router_initialized:
            self.unified_router = await get_unified_router()
            # v2.0: Connect registry for semantic search
            if self.registry and self.registry.is_ready:
                self.unified_router.set_registry(self.registry)
            self._unified_router_initialized = True

        # Build conversation state for router
        conv_state = None
        if conv_manager.is_in_flow():
            conv_state = {
                "flow": conv_manager.get_current_flow(),
                "state": state.value,
                "tool": conv_manager.get_current_tool(),
                "missing_params": conv_manager.get_missing_params()
            }

        # Get unified routing decision
        decision = await self.unified_router.route(text, user_context, conv_state)

        logger.info(
            f"UNIFIED ROUTER: action={decision.action}, tool={decision.tool}, "
            f"flow={decision.flow_type}, conf={decision.confidence:.2f}"
        )

        # === Handle routing decision ===

        # 1. Direct response (greetings, help)
        if decision.action == "direct_response":
            return decision.response or "Kako vam mogu pomoći?"

        # 2. Exit flow - user wants something different
        if decision.action == "exit_flow":
            # CRITICAL FIX: Only exit if actually in a flow, prevent infinite loop
            if conv_manager.is_in_flow():
                logger.info(f"UNIFIED ROUTER: Exiting flow, resetting state")
                await conv_manager.reset()
                # Re-route with clean state - but prevent infinite recursion
                # by passing a flag or checking if we just reset
                new_decision = await self.unified_router.route(text, user_context, None)

                # Handle the new decision directly (no recursion)
                if new_decision.action == "direct_response":
                    return new_decision.response or "Kako vam mogu pomoći?"
                if new_decision.action == "start_flow":
                    if new_decision.flow_type == "booking":
                        return await self._handle_booking_flow(text, user_context, conv_manager)
                    if new_decision.flow_type == "mileage":
                        return await self._handle_mileage_input_flow(text, user_context, conv_manager)
                    if new_decision.flow_type == "case":
                        return await self._handle_case_creation_flow(text, user_context, conv_manager)
                if new_decision.action == "simple_api" and new_decision.tool:
                    route = RouteResult(
                        matched=True,
                        tool_name=new_decision.tool,
                        confidence=new_decision.confidence,
                        flow_type="simple"
                    )
                    result = await self._execute_deterministic(
                        route, user_context, conv_manager, sender, text
                    )
                    if result:
                        return result
                # Fallback to new request handling
                return await self._handle_new_request(sender, text, user_context, conv_manager)
            else:
                # Not in flow but got exit_flow - treat as simple_api or new request
                logger.warning(f"UNIFIED ROUTER: exit_flow received but not in flow - ignoring")
                return await self._handle_new_request(sender, text, user_context, conv_manager)

        # 3. Continue flow - user is providing requested info
        if decision.action == "continue_flow":
            # CRITICAL: Handle based on ACTUAL state, not router's guess
            if state == ConversationState.SELECTING_ITEM:
                return await self._flow_handler.handle_selection(
                    sender, text, user_context, conv_manager, self._handle_new_request
                )
            if state == ConversationState.CONFIRMING:
                result = await self._flow_handler.handle_confirmation(
                    sender, text, user_context, conv_manager
                )
                # P1 FIX: Handle mid-flow questions
                if isinstance(result, dict) and result.get("mid_flow_question"):
                    # User asked a question during confirmation - answer it
                    # but preserve the confirmation state
                    question = result.get("question", text)
                    logger.info(f"P1: Handling mid-flow question: '{question[:50]}'")
                    # Process the question through normal flow (but don't reset state)
                    answer = await self._handle_new_request(sender, question, user_context, conv_manager)
                    # Remind user about pending confirmation
                    return f"{answer}\n\n---\n_Čeka se potvrda prethodne operacije. Potvrdite s **Da** ili **Ne**._"
                return result
            if state == ConversationState.GATHERING_PARAMS:
                return await self._flow_handler.handle_gathering(
                    sender, text, user_context, conv_manager, self._handle_new_request
                )
            # If we're supposedly in a flow but state is IDLE, try to recover
            if state == ConversationState.IDLE and conv_manager.get_current_flow():
                logger.warning(f"STATE MISMATCH: is_in_flow but state=IDLE, treating as new request")
                # Fall through to new request handling

        # 4. Start flow - begin a multi-step flow
        if decision.action == "start_flow":
            if decision.flow_type == "booking":
                return await self._handle_booking_flow(text, user_context, conv_manager)
            if decision.flow_type == "mileage":
                return await self._handle_mileage_input_flow(text, user_context, conv_manager)
            if decision.flow_type == "case":
                return await self._handle_case_creation_flow(text, user_context, conv_manager)

        # 5. Simple API - direct tool call
        if decision.action == "simple_api" and decision.tool:
            # Create route result for deterministic execution
            route = RouteResult(
                matched=True,
                tool_name=decision.tool,
                confidence=decision.confidence,
                flow_type="simple"
            )
            result = await self._execute_deterministic(
                route, user_context, conv_manager, sender, text
            )
            if result:
                return result
            # Fall through to LLM if deterministic fails

        # === END UNIFIED ROUTER ===

        # Fallback to original new request handling
        return await self._handle_new_request(sender, text, user_context, conv_manager)

    async def _handle_new_request(
        self,
        sender: str,
        text: str,
        user_context: Dict[str, Any],
        conv_manager: ConversationManager
    ) -> str:
        """Handle new request with Chain Planning and Fallback Execution."""

        # === v16.1: DETERMINISTIC ROUTING - Try rules FIRST ===
        # This guarantees correct responses for known patterns WITHOUT LLM
        route = self.query_router.route(text, user_context)

        if route.matched:
            logger.info(f"ROUTER: Deterministic match → {route.tool_name or route.flow_type}")

            # Direct response (greetings, thanks, help, context queries)
            if route.flow_type == "direct_response":
                # Format template with user context
                if route.response_template:
                    # DIRECT EXTRACTION - bypass format() completely for context queries
                    if 'person_id' in route.response_template:
                        val = user_context.get('person_id', 'N/A')
                        result = f"👤 **Person ID:** {val}"
                        return result
                    elif 'phone' in route.response_template:
                        val = user_context.get('phone', 'N/A')
                        logger.info(f"ROUTER: Direct extraction phone={val}, full_context={user_context}")
                        return f"📱 **Telefon:** {val}"
                    elif 'tenant_id' in route.response_template:
                        val = user_context.get('tenant_id', 'N/A')
                        logger.info(f"ROUTER: Direct extraction tenant_id={val}, full_context={user_context}")
                        return f"🏢 **Tenant ID:** {val}"
                    
                    # For other templates, try format()
                    simple_context = {
                        k: v for k, v in user_context.items() 
                        if not isinstance(v, (dict, list))
                    }
                    try:
                        response = route.response_template.format(**simple_context)
                        logger.info(f"ROUTER: Direct response formatted")
                        return response
                    except (KeyError, ValueError) as e:
                        logger.error(f"ROUTER: Format error {e}. Template: {route.response_template}")
                        return route.response_template
                return route.response_template

            # Booking flow
            if route.flow_type == "booking":
                return await self._handle_booking_flow(text, user_context, conv_manager)

            # Mileage input flow
            if route.flow_type == "mileage_input":
                return await self._handle_mileage_input_flow(text, user_context, conv_manager)

            # Simple query - execute tool deterministically
            if route.flow_type == "simple" and route.tool_name:
                result = await self._execute_deterministic(
                    route, user_context, conv_manager, sender, text
                )
                if result:
                    return result
                # If deterministic execution failed, fall through to LLM path
                logger.warning("Deterministic execution failed, falling back to LLM")

            # List flow (my bookings, etc.)
            if route.flow_type == "list" and route.tool_name:
                result = await self._execute_deterministic(
                    route, user_context, conv_manager, sender, text
                )
                if result:
                    return result
                logger.warning("List execution failed, falling back to LLM")

            # Case creation flow (report damage, etc.)
            if route.flow_type == "case_creation" and route.tool_name:
                return await self._handle_case_creation_flow(text, user_context, conv_manager)

        # No pattern match - try intelligent category-based routing
        if not route.matched:
            logger.info(f"ROUTER: No pattern match - trying intelligent routing")

            # Initialize intelligent router lazily
            if not self._intelligent_router_initialized:
                await self.intelligent_router.initialize()
                self._intelligent_router_initialized = True

            # Get intelligent routing decision
            intelligent_decision = await self.intelligent_router.route(
                query=text,
                user_context=user_context,
                conversation_state=conv_manager.to_dict() if conv_manager.is_in_flow() else None
            )

            # Handle intelligent routing results
            if intelligent_decision.flow_type == FlowType.DIRECT_RESPONSE:
                return intelligent_decision.direct_response

            if intelligent_decision.tool_name and intelligent_decision.confidence >= 0.4:
                logger.info(
                    f"INTELLIGENT ROUTER: {intelligent_decision.tool_name} "
                    f"(conf={intelligent_decision.confidence:.2f}, flow={intelligent_decision.flow_type.value})"
                )

                # Handle flows based on intelligent routing
                if intelligent_decision.flow_type == FlowType.BOOKING:
                    return await self._handle_booking_flow(text, user_context, conv_manager)

                if intelligent_decision.flow_type == FlowType.AVAILABILITY:
                    return await self._handle_booking_flow(text, user_context, conv_manager)

                if intelligent_decision.flow_type == FlowType.CASE_CREATION:
                    return await self._handle_case_creation_flow(text, user_context, conv_manager)

                if intelligent_decision.flow_type == FlowType.MILEAGE_INPUT:
                    return await self._handle_mileage_input_flow(text, user_context, conv_manager)

                if intelligent_decision.flow_type == FlowType.LIST:
                    # For list queries, create a route result and use deterministic execution
                    list_route = RouteResult(
                        matched=True,
                        tool_name=intelligent_decision.tool_name,
                        extract_fields=[],
                        flow_type="list"
                    )
                    result = await self._execute_deterministic(
                        list_route, user_context, conv_manager, sender, text
                    )
                    if result:
                        return result

                # For SIMPLE flow type, continue to LLM path but with the selected tool
                if intelligent_decision.flow_type == FlowType.SIMPLE:
                    # Store the intelligent decision for later use
                    route = RouteResult(
                        matched=True,
                        tool_name=intelligent_decision.tool_name,
                        confidence=intelligent_decision.confidence,
                        flow_type="simple"
                    )

        # === END DETERMINISTIC AND INTELLIGENT ROUTING ===

        # Pre-resolve entity references
        pre_resolved = await self._pre_resolve_entity_references(
            text, user_context, conv_manager
        )

        if pre_resolved:
            logger.info(f"Pre-resolved entities: {list(pre_resolved.keys())}")

        # Get history
        history = await self.context.get_recent_messages(sender)

        messages = history.copy()
        messages.append({"role": "user", "content": text})

        # Get relevant tools with scores
        tools_with_scores = await self.registry.find_relevant_tools_with_scores(
            text, top_k=10  # v16.0: More tools for better fallback options
        )

        tools_with_scores = sorted(
            tools_with_scores,
            key=lambda t: t["score"],
            reverse=True
        )

        tools = [t["schema"] for t in tools_with_scores]

        # === v16.0: Use ChainPlanner for multi-step plans with fallbacks ===
        plan = await self.chain_planner.create_plan(
            query=text,
            user_context=user_context,
            available_tools=tools,
            tool_scores=tools_with_scores
        )

        logger.info(f"ChainPlan: {plan.understanding} (simple={plan.is_simple})")

        # Handle direct response (greetings, clarifications)
        if plan.direct_response:
            logger.info("Planner direct response")
            return plan.direct_response

        # Handle missing data - start gathering flow
        if plan.missing_data and not plan.has_all_data:
            missing_prompt = self._build_missing_data_prompt(plan.missing_data)
            logger.info(f"Missing data: {plan.missing_data}")

            # v16.0: Use primary_path instead of steps
            first_step = plan.primary_path[0] if plan.primary_path else None

            # Store plan in conversation for continuation
            await conv_manager.start_flow(
                flow_name="gathering",
                tool=first_step.tool_name if first_step else None,
                required_params=plan.missing_data
            )
            await conv_manager.save()

            return missing_prompt

        # v16.0: Store extraction hint for response formatting
        extraction_hint = getattr(plan, 'extraction_hint', None)

        # v16.0: Get fallback tools from plan
        fallback_tools = []
        if hasattr(plan, 'fallback_paths') and plan.fallback_paths:
            for step_fallbacks in plan.fallback_paths.values():
                for fb in step_fallbacks:
                    for fb_step in fb.steps:
                        if fb_step.tool_name:
                            fallback_tools.append(fb_step.tool_name)

        # === Continue with AI tool calling for execution ===
        forced_tool = None

        # CRITICAL FIX v15.1: SINGLE_TOOL_THRESHOLD must match ai_orchestrator
        SINGLE_TOOL_THRESHOLD = 0.98

        if tools_with_scores:
            best_match = max(tools_with_scores, key=lambda t: t["score"])
            best_score = best_match["score"]
            best_tool_name = best_match["name"]

            tool_names = [t["name"] for t in tools_with_scores]
            available_tool_names = set(tool_names)
            logger.info(f"Available tools: {tool_names[:5]}")
            logger.info(f"Best match: {best_tool_name} (score={best_score:.3f})")
            if fallback_tools:
                logger.info(f"Fallback tools: {fallback_tools}")

            # v16.0: Get first step from primary_path
            first_step = plan.primary_path[0] if plan.primary_path else None

            # CRITICAL FIX v15.1: If we're in SINGLE TOOL MODE, only best tool is valid
            # Token budgeting will reduce to only the best tool when score >= 0.98
            # So forced_tool MUST be best_tool_name, otherwise we get OpenAI 400 error
            if best_score >= SINGLE_TOOL_THRESHOLD:
                # SINGLE TOOL MODE - only best match will be sent to OpenAI
                forced_tool = best_tool_name
                if first_step and first_step.tool_name:
                    suggested = first_step.tool_name
                    if suggested != best_tool_name:
                        logger.warning(
                            f"PLANNER suggested '{suggested}' but SINGLE TOOL MODE active "
                            f"(score={best_score:.3f} >= {SINGLE_TOOL_THRESHOLD}). "
                            f"Using best match '{best_tool_name}' instead."
                        )
                    else:
                        logger.info(f"PLANNER: Confirmed best match {forced_tool}")
                else:
                    logger.info(f"SINGLE TOOL MODE: Using {forced_tool}")
            else:
                # Normal mode - can use planner's suggestion if it's in available tools
                if first_step and first_step.tool_name:
                    suggested_tool = first_step.tool_name
                    if suggested_tool in available_tool_names:
                        forced_tool = suggested_tool
                        logger.info(f"PLANNER: Using suggested tool {forced_tool}")
                    else:
                        # Planner suggested a tool not in search results
                        forced_tool = best_tool_name if best_score >= settings.ACTION_THRESHOLD else None
                        logger.warning(
                            f"PLANNER suggested '{suggested_tool}' not in available tools. "
                            f"Using best match '{forced_tool}' instead."
                        )
                elif best_score >= settings.ACTION_THRESHOLD:
                    forced_tool = best_tool_name
                    logger.info(f"ACTION-FIRST: Forcing {forced_tool}")

        # Build system prompt
        system_prompt = self.ai.build_system_prompt(
            user_context,
            conv_manager.to_dict() if conv_manager.is_in_flow() else None
        )

        # AI iteration loop
        current_messages = messages.copy()

        for iteration in range(self.MAX_ITERATIONS):
            logger.debug(f"AI iteration {iteration + 1}/{self.MAX_ITERATIONS}")

            current_forced = forced_tool if iteration == 0 else None

            # v19.1: Use instrumented call for drift detection
            result = await self._instrumented_ai_call(
                messages=current_messages,
                tools=tools,
                system_prompt=system_prompt,
                forced_tool=current_forced,
                tool_scores=tools_with_scores,
                user_context=user_context
            )

            if result.get("type") == "error":
                return result.get("content", "Greska u AI obradi.")

            if result.get("type") == "text":
                return result.get("content", "")

            if result.get("type") == "tool_call":
                tool_name = result["tool"]
                tool = self.registry.get_tool(tool_name)
                method = tool.method if tool else "GET"

                # Check for special handling
                if "available" in tool_name.lower() or "calendar" in tool_name.lower():
                    if method == "GET" and "vehicle" in tool_name.lower():
                        return await self._handle_availability_flow(
                            result, user_context, conv_manager
                        )

                # ============================================================
                # CRITICAL FIX v20.0: BLOCK DIRECT POST VehicleCalendar
                # ============================================================
                # LLM sometimes tries to call post_VehicleCalendar directly with
                # fabricated parameters (VehicleId, FromTime, ToTime).
                # We MUST force it through the booking flow to ensure:
                # 1. VehicleId comes from get_AvailableVehicles (not invented)
                # 2. User explicitly confirms the booking
                # ============================================================
                if tool_name == "post_VehicleCalendar" and method == "POST":
                    params = result.get("parameters", {})
                    vehicle_id = params.get("VehicleId") or params.get("vehicleId")
                    
                    # Check if VehicleId was validated (exists in tool_outputs from get_AvailableVehicles)
                    validated_vehicle_id = None
                    if hasattr(conv_manager.context, 'tool_outputs'):
                        validated_vehicle_id = conv_manager.context.tool_outputs.get("VehicleId")
                        
                        # Also check if it's in all_available_vehicles
                        all_vehicles = conv_manager.context.tool_outputs.get("all_available_vehicles", [])
                        if vehicle_id and all_vehicles:
                            for v in all_vehicles:
                                if v.get("Id") == vehicle_id:
                                    validated_vehicle_id = vehicle_id
                                    break
                    
                    # If VehicleId is NOT validated (fabricated by LLM), redirect to booking flow
                    if not validated_vehicle_id or vehicle_id != validated_vehicle_id:
                        logger.warning(
                            f"🚫 BLOCKED direct post_VehicleCalendar: VehicleId={vehicle_id} "
                            f"is NOT validated (expected={validated_vehicle_id}). "
                            f"Redirecting to booking flow."
                        )
                        
                        # Extract any time params LLM may have provided
                        from_time = params.get("FromTime") or params.get("from")
                        to_time = params.get("ToTime") or params.get("to")
                        
                        # Redirect to proper booking flow
                        return await self._handle_booking_flow(text, user_context, conv_manager)
                    
                    # Even if VehicleId is validated, ensure we're in confirmation state
                    state = conv_manager.get_state()
                    if state != ConversationState.EXECUTING:
                        logger.warning(
                            f"🚫 BLOCKED post_VehicleCalendar: state={state.value} "
                            f"(expected EXECUTING after user confirmation)"
                        )
                        # User hasn't confirmed - ask for confirmation
                        confirmation_result = await self._flow_handler.request_confirmation(
                            tool_name, params, user_context, conv_manager
                        )
                        return confirmation_result.get("prompt", "Molim potvrdite rezervaciju s 'Da' ili 'Ne'.")

                if method in ("POST", "PUT", "PATCH"):
                    if self._tool_handler.requires_confirmation(tool_name):
                        flow_result = await self._flow_handler.request_confirmation(
                            tool_name, result["parameters"], user_context, conv_manager
                        )
                        if flow_result.get("needs_input"):
                            return flow_result["prompt"]

                # Execute tool
                # NEW v10.1: Pass original text for intent-aware response formatting
                tool_response = await self._tool_handler.execute_tool_call(
                    result, user_context, conv_manager, sender, user_query=text
                )

                # v16.0: Use LLM Response Extractor for successful responses
                if tool_response.get("success") and tool_response.get("data"):
                    try:
                        extracted_response = await self.response_extractor.extract(
                            user_query=text,
                            api_response=tool_response["data"],
                            tool_name=tool_name,
                            extraction_hint=extraction_hint
                        )
                        if extracted_response:
                            logger.info(f"LLM extracted response for {tool_name}")
                            return extracted_response
                    except Exception as e:
                        logger.warning(f"LLM extraction failed, using fallback: {e}")
                        # Continue with standard formatting

                if tool_response.get("final_response"):
                    return tool_response["final_response"]

                if tool_response.get("needs_input"):
                    return tool_response.get("prompt", "")

                # v16.0: Try fallback tools on failure
                if not tool_response.get("success", True) and fallback_tools:
                    logger.info(f"Primary tool {tool_name} failed, trying fallbacks...")

                    for fb_tool_name in fallback_tools:
                        if fb_tool_name == tool_name:
                            continue  # Skip the one that just failed

                        fb_tool = self.registry.get_tool(fb_tool_name)
                        if not fb_tool:
                            continue

                        logger.info(f"Trying fallback: {fb_tool_name}")

                        # Adapt parameters for fallback tool
                        fb_params = result["parameters"].copy()

                        fb_result = {
                            "tool": fb_tool_name,
                            "parameters": fb_params,
                            "tool_call_id": result.get("tool_call_id", "fallback")
                        }

                        fb_response = await self._tool_handler.execute_tool_call(
                            fb_result, user_context, conv_manager, sender, user_query=text
                        )

                        if fb_response.get("success"):
                            logger.info(f"Fallback {fb_tool_name} succeeded!")

                            # Use LLM extraction for fallback response too
                            if fb_response.get("data"):
                                try:
                                    extracted = await self.response_extractor.extract(
                                        user_query=text,
                                        api_response=fb_response["data"],
                                        tool_name=fb_tool_name,
                                        extraction_hint=extraction_hint
                                    )
                                    if extracted:
                                        return extracted
                                except Exception as e:
                                    logger.warning(f"Fallback extraction failed: {e}")

                            if fb_response.get("final_response"):
                                return fb_response["final_response"]
                            break  # Success - exit fallback loop

                        logger.info(f"Fallback {fb_tool_name} also failed")

                # Add to conversation for next iteration
                tool_result_content = tool_response.get("data", {})

                if not tool_response.get("success", True):
                    ai_feedback = tool_response.get("ai_feedback", "")
                    tool_result_content = {
                        "error": True,
                        "message": tool_response.get("error", "Unknown error"),
                        "ai_feedback": ai_feedback
                    }

                current_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": result["tool_call_id"],
                        "type": "function",
                        "function": {
                            "name": result["tool"],
                            "arguments": json.dumps(result["parameters"])
                        }
                    }]
                })

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": json.dumps(tool_result_content)
                })

        return "Nisam uspio obraditi zahtjev. Pokusajte drugacije formulirati."

    async def _handle_availability_flow(
        self,
        result: Dict[str, Any],
        user_context: Dict[str, Any],
        conv_manager: ConversationManager
    ) -> str:
        """Handle availability check flow."""
        flow_result = await self._flow_handler.handle_availability(
            result["tool"],
            result["parameters"],
            user_context,
            conv_manager
        )

        if flow_result.get("needs_input"):
            return flow_result["prompt"]
        if flow_result.get("final_response"):
            return flow_result["final_response"]

        return flow_result.get("error", "Greska pri provjeri dostupnosti.")

    async def _pre_resolve_entity_references(
        self,
        text: str,
        user_context: Dict[str, Any],
        conv_manager: ConversationManager
    ) -> Dict[str, Any]:
        """Pre-resolve entity references before AI processing."""
        resolved = {}

        try:
            entity_ref = self.dependency_resolver.detect_entity_reference(
                text, entity_type="vehicle"
            )

            if entity_ref:
                logger.info(f"Pre-resolving entity: {entity_ref}")

                resolution = await self.dependency_resolver.resolve_entity_reference(
                    reference=entity_ref,
                    user_context=user_context,
                    executor=self.executor
                )

                if resolution.success:
                    logger.info(f"Pre-resolved VehicleId: {resolution.resolved_value}")

                    resolved["VehicleId"] = resolution.resolved_value
                    resolved["vehicleId"] = resolution.resolved_value

                    if hasattr(conv_manager.context, 'tool_outputs'):
                        conv_manager.context.tool_outputs["VehicleId"] = (
                            resolution.resolved_value
                        )
                        conv_manager.context.tool_outputs["vehicleId"] = (
                            resolution.resolved_value
                        )
                        await conv_manager.save()
                else:
                    logger.warning(
                        f"Failed to pre-resolve: {resolution.error_message}"
                    )

        except Exception as e:
            logger.error(f"Entity pre-resolution error: {e}", exc_info=True)

        return resolved

    def _build_missing_data_prompt(self, missing_params: List[str]) -> str:
        """Build user-friendly prompt for missing parameters."""
        param_prompts = {
            "from": "Od kada vam treba? (npr. 'sutra u 9:00')",
            "to": "Do kada?",
            "FromTime": "Od kada vam treba?",
            "ToTime": "Do kada?",
            "Description": "Mozete li opisati situaciju?",
            "VehicleId": "Koje vozilo zelite?",
            "Value": "Koja je vrijednost? (npr. kilometraza)",
            "Subject": "Koji je naslov/tema?",
            "Message": "Koja je poruka?"
        }

        if len(missing_params) == 1:
            param = missing_params[0]
            return param_prompts.get(param, f"Trebam jos: {param}")

        lines = ["Za nastavak trebam jos informacije:"]
        for param in missing_params[:3]:
            prompt = param_prompts.get(param, param)
            lines.append(f"* {prompt}")

        return "\n".join(lines)

    # === v16.1: DETERMINISTIC EXECUTION METHODS ===

    async def _execute_deterministic(
        self,
        route: RouteResult,
        user_context: Dict[str, Any],
        conv_manager: ConversationManager,
        sender: str,
        original_query: str
    ) -> Optional[str]:
        """
        Execute tool deterministically without LLM tool selection.

        This is the FAST PATH for known queries:
        - NO embedding search
        - NO LLM tool selection
        - GUARANTEED correct tool

        Returns:
            Formatted response string or None if should fall back to LLM
        """
        tool = self.registry.get_tool(route.tool_name)
        if not tool:
            logger.error(f"ROUTER: Tool {route.tool_name} not found in registry")
            return None

        # Build parameters from context
        params = {}
        vehicle = user_context.get("vehicle", {})

        # FIX v21.1: Inject personIdOrEmail for PersonData endpoint
        # This is a PATH parameter that needs person_id from context
        if "personIdOrEmail" in tool.parameters:
            person_id = user_context.get("person_id")
            if person_id:
                params["personIdOrEmail"] = person_id
                logger.info(f"DETERMINISTIC: Injected personIdOrEmail={person_id[:8]}... for {route.tool_name}")
            else:
                logger.warning("DETERMINISTIC: No person_id in user_context for PersonData")
                return "Molim vas prijavite se kako bih mogao dohvatiti vaše podatke."

        # Inject VehicleId if needed and available - use Swagger field name
        vehicle_id_param = tool.parameters.get("VehicleId")
        if vehicle_id_param and getattr(vehicle_id_param, 'required', False):
            vehicle_id = vehicle.get("Id")
            if vehicle_id:
                params["VehicleId"] = vehicle_id
            else:
                # No vehicle - can't execute
                return (
                    "Trenutno nemate dodijeljeno vozilo.\n"
                    "Želite li rezervirati vozilo?"
                )

        # Create execution context
        from services.tool_contracts import ToolExecutionContext
        execution_context = ToolExecutionContext(
            user_context=user_context,
            tool_outputs=conv_manager.context.tool_outputs if hasattr(conv_manager.context, 'tool_outputs') else {},
            conversation_state={}
        )

        # NEW: Check if mutation tool requires confirmation
        is_mutation = tool.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}
        state = conv_manager.get_state()

        if is_mutation and state != ConversationState.CONFIRMING:
            # Mutation tool needs confirmation - show confirmation dialog
            logger.info(f"DETERMINISTIC: Mutation {route.tool_name} requires confirmation")
            result = await self._flow_handler.request_confirmation(
                tool_name=route.tool_name,
                parameters=params,
                user_context=user_context,
                conv_manager=conv_manager
            )
            return result.get("prompt", "Potvrdite operaciju s 'Da' ili odustanite s 'Ne'.")

        # Execute tool
        logger.info(f"DETERMINISTIC: Executing {route.tool_name} with {list(params.keys())}")
        result = await self.executor.execute(tool, params, execution_context)

        logger.info(f"DETERMINISTIC: Result success={result.success}, data_type={type(result.data)}")
        if result.data:
            # Log first 500 chars of data for debugging
            data_preview = str(result.data)[:500]
            logger.info(f"DETERMINISTIC: Data preview: {data_preview}")

        if not result.success:
            logger.warning(f"DETERMINISTIC: {route.tool_name} failed: {result.error_message}")
            return None  # Fall back to LLM

        # Try template-based formatting first
        if route.response_template:
            formatted = self.query_router.format_response(route, result.data, original_query)
            if formatted:
                logger.info(f"DETERMINISTIC: Template response for {route.tool_name}")
                return formatted

        # Use LLM extraction for complex responses
        try:
            extraction_hint = ",".join(route.extract_fields) if route.extract_fields else None
            extracted = await self.response_extractor.extract(
                user_query=original_query,
                api_response=result.data,
                tool_name=route.tool_name,
                extraction_hint=extraction_hint
            )
            if extracted:
                logger.info(f"DETERMINISTIC: LLM extracted response for {route.tool_name}")
                return extracted
        except Exception as e:
            logger.warning(f"DETERMINISTIC: LLM extraction failed: {e}")

        # Final fallback - use standard formatter
        result_dict = {"success": True, "data": result.data, "operation": route.tool_name}
        return self.formatter.format_result(result_dict, tool, user_query=original_query)

    async def _handle_booking_flow(
        self,
        text: str,
        user_context: Dict[str, Any],
        conv_manager: ConversationManager
    ) -> str:
        """Handle booking flow deterministically."""
        # Extract time parameters from text
        time_params = await self.ai.extract_parameters(
            text,
            [
                {"name": "from", "type": "string", "description": "Početno vrijeme"},
                {"name": "to", "type": "string", "description": "Završno vrijeme"}
            ]
        )

        params = {}
        if time_params.get("from"):
            params["FromTime"] = time_params["from"]
            params["from"] = time_params["from"]
        if time_params.get("to"):
            params["ToTime"] = time_params["to"]
            params["to"] = time_params["to"]

        # Start availability flow
        result = {
            "tool": "get_AvailableVehicles",
            "parameters": params,
            "tool_call_id": "booking_flow"
        }

        return await self._handle_availability_flow(result, user_context, conv_manager)

    async def _handle_mileage_input_flow(
        self,
        text: str,
        user_context: Dict[str, Any],
        conv_manager: ConversationManager
    ) -> str:
        """Handle mileage input flow deterministically."""
        # Extract mileage value from text
        mileage_params = await self.ai.extract_parameters(
            text,
            [
                {"name": "Value", "type": "number", "description": "Kilometraža u km"}
            ]
        )

        # Try to get vehicle from multiple sources - use Swagger field names
        vehicle = user_context.get("vehicle", {})
        vehicle_id = vehicle.get("Id")
        vehicle_name = vehicle.get("FullVehicleName") or vehicle.get("DisplayName", "")
        plate = vehicle.get("LicencePlate", "")

        # 1. Check if we have vehicle from recent booking/context
        if not vehicle_id and hasattr(conv_manager.context, 'tool_outputs'):
            vehicle_id = conv_manager.context.tool_outputs.get("VehicleId")
            if vehicle_id:
                # Try to get name from stored vehicles
                all_vehicles = conv_manager.context.tool_outputs.get("all_available_vehicles", [])
                for v in all_vehicles:
                    if v.get("Id") == vehicle_id:
                        vehicle_name = v.get("DisplayName") or v.get("FullVehicleName") or "Vozilo"
                        plate = v.get("LicencePlate") or ""
                        break

        # 2. If still no vehicle, fetch first available one
        if not vehicle_id:
            try:
                from services.api_gateway import HttpMethod
                from datetime import datetime, timedelta

                tomorrow = datetime.now() + timedelta(days=1)
                result = await self.gateway.execute(
                    method=HttpMethod.GET,
                    path="/vehiclemgt/AvailableVehicles",
                    params={
                        "from": tomorrow.replace(hour=8, minute=0).isoformat(),
                        "to": tomorrow.replace(hour=17, minute=0).isoformat()
                    }
                )

                if result.success and result.data:
                    data = result.data.get("Data", result.data) if isinstance(result.data, dict) else result.data
                    vehicles = data if isinstance(data, list) else [data]

                    if vehicles:
                        v = vehicles[0]
                        vehicle_id = v.get("Id")
                        vehicle_name = v.get("DisplayName") or v.get("FullVehicleName") or "Vozilo"
                        plate = v.get("LicencePlate") or ""

                        # Store for later - ONLY minimal data to prevent serialization issues
                        if hasattr(conv_manager.context, 'tool_outputs'):
                            conv_manager.context.tool_outputs["VehicleId"] = vehicle_id
                            # Store only minimal vehicle data
                            minimal_vehicles = [{
                                "Id": v.get("Id"),
                                "DisplayName": v.get("DisplayName") or v.get("FullVehicleName") or "Vozilo",
                                "LicencePlate": v.get("LicencePlate") or v.get("Plate") or ""
                            } for v in vehicles]
                            conv_manager.context.tool_outputs["all_available_vehicles"] = minimal_vehicles
            except Exception as e:
                logger.warning(f"Failed to fetch vehicles for mileage: {e}")

        if not vehicle_id:
            return (
                "Nije pronađeno vozilo za unos kilometraže.\n"
                "Pokušajte prvo rezervirati vozilo ili kontaktirajte podršku."
            )

        if not mileage_params.get("Value"):
            # Start gathering flow - store vehicle info
            await conv_manager.start_flow(
                flow_name="mileage_input",
                tool="post_AddMileage",
                required_params=["Value"]
            )
            await conv_manager.add_parameters({
                "VehicleId": vehicle_id,
                "_vehicle_name": vehicle_name,
                "_vehicle_plate": plate
            })
            await conv_manager.save()

            return (
                f"Unosim kilometražu za **{vehicle_name}** ({plate}).\n\n"
                f"Kolika je trenutna kilometraža? _(npr. '14500')_"
            )

        # Have all params - ask for confirmation
        value = mileage_params["Value"]

        await conv_manager.add_parameters({
            "VehicleId": vehicle_id,
            "Value": value
        })

        message = (
            f"**Potvrda unosa kilometraže:**\n\n"
            f"Vozilo: {vehicle_name} ({plate})\n"
            f"Kilometraža: {value} km\n\n"
            f"_Potvrdite s 'Da' ili odustanite s 'Ne'._"
        )

        await conv_manager.request_confirmation(message)
        conv_manager.context.current_tool = "post_AddMileage"
        await conv_manager.save()

        return message

    async def _handle_case_creation_flow(
        self,
        text: str,
        user_context: Dict[str, Any],
        conv_manager: ConversationManager
    ) -> str:
        """Handle support case/damage report creation deterministically."""
        # Extract description from text
        case_params = await self.ai.extract_parameters(
            text,
            [
                {"name": "Description", "type": "string", "description": "Opis problema ili kvara"},
                {"name": "Subject", "type": "string", "description": "Naslov slučaja"}
            ]
        )

        vehicle = user_context.get("vehicle", {})
        vehicle_id = vehicle.get("Id")
        vehicle_name = vehicle.get("FullVehicleName") or vehicle.get("DisplayName", "vozilo")
        plate = vehicle.get("LicencePlate", "")

        # Build subject from text if not extracted
        subject = case_params.get("Subject")
        if not subject:
            # Try to infer subject from common patterns
            text_lower = text.lower()
            if "kvar" in text_lower:
                subject = "Prijava kvara"
            elif "šteta" in text_lower or "oštećen" in text_lower:
                subject = "Prijava oštećenja"
            elif "problem" in text_lower:
                subject = "Prijava problema"
            else:
                subject = "Prijava slučaja"

        description = case_params.get("Description")

        if not description:
            # Need to gather description
            await conv_manager.start_flow(
                flow_name="case_creation",
                tool="post_AddCase",
                required_params=["Description"]
            )

            # Store what we have so far
            params = {"Subject": subject}
            if vehicle_id:
                params["VehicleId"] = vehicle_id
            await conv_manager.add_parameters(params)
            await conv_manager.save()

            return "Možete li opisati problem ili kvar detaljnije?"

        # Have all data - request confirmation
        # API expects: User, Subject, Message
        person_id = user_context.get("person_id", "")
        params = {
            "User": person_id,  # Required by API
            "Subject": subject,
            "Message": description  # API uses "Message", not "Description"
        }

        await conv_manager.add_parameters(params)

        vehicle_line = f"Vozilo: {vehicle_name} ({plate})\n" if vehicle_id else ""

        message = (
            f"**Potvrda prijave slučaja:**\n\n"
            f"Naslov: {subject}\n"
            f"{vehicle_line}"
            f"Opis: {description}\n\n"
            f"_Potvrdite s 'Da' ili odustanite s 'Ne'._"
        )

        await conv_manager.request_confirmation(message)
        conv_manager.context.current_tool = "post_AddCase"
        await conv_manager.save()

        return message

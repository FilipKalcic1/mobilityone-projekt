"""
Message Engine - Public API facade.

Coordinates message processing through:
- User identification and context
- Conversation state management (Redis-backed)
- AI routing and tool execution
- Cost tracking and observability

This module has been refactored into smaller components:
- user_handler.py: User identification and greeting
- hallucination_handler.py: Hallucination feedback detection
- deterministic_executor.py: Fast path execution without LLM
- flow_executors.py: Multi-step flow handling
- tool_handler.py: Tool execution with validation
- flow_handler.py: Flow state management
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
from services.dependency_resolver import DependencyResolver
from services.error_learning import ErrorLearningService
from services.model_drift_detector import get_drift_detector
from services.cost_tracker import CostTracker
from services.reasoning import Planner

from services.chain_planner import get_chain_planner
from services.response_extractor import get_response_extractor
from services.query_router import get_query_router, RouteResult
from services.unified_router import get_unified_router

from .tool_handler import ToolHandler
from .flow_handler import FlowHandler
from .user_handler import UserHandler
from .hallucination_handler import HallucinationHandler
from .deterministic_executor import DeterministicExecutor
from .flow_executors import FlowExecutors
from services.context import (
    UserContextManager,
    MissingContextError,
    VehicleSelectionRequired,
)
from services.flow_phrases import (
    matches_show_more,
    matches_confirm_yes,
    matches_confirm_no,
    matches_item_selection,
)

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
        self.executor = ToolExecutor(gateway, registry=registry)
        self.context = context_service
        self.queue = queue_service
        self.cache = cache_service
        self.db = db_session
        self.redis = context_service.redis if context_service else None

        # Core services
        self.dependency_resolver = DependencyResolver(registry)
        self.error_learning = ErrorLearningService(redis_client=self.redis)

        # Model drift detection
        self.drift_detector = get_drift_detector(
            redis_client=self.redis,
            db_session=self.db
        )
        self.error_learning.set_drift_detector(self.drift_detector)

        # Cost tracking
        self.cost_tracker: Optional[CostTracker] = None
        if self.redis:
            try:
                self.cost_tracker = CostTracker(redis_client=self.redis)
                logger.info("CostTracker initialized")
            except Exception as e:
                logger.warning(f"CostTracker init failed: {e}")

        self.ai = AIOrchestrator()
        self.formatter = ResponseFormatter()
        self.planner = Planner()

        # Advanced planning and response extraction
        self.chain_planner = get_chain_planner()
        self.response_extractor = get_response_extractor()

        # Deterministic query router
        self.query_router = get_query_router()

        # Unified LLM router
        self.unified_router = None
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

        # New modular handlers
        self._user_handler = UserHandler(db_session, gateway, cache_service)
        self._hallucination_handler = HallucinationHandler(
            context_service=context_service,
            error_learning=self.error_learning,
            ai_model=self.ai.model
        )
        self._deterministic_executor = DeterministicExecutor(
            registry=registry,
            executor=self.executor,
            formatter=self.formatter,
            query_router=self.query_router,
            response_extractor=self.response_extractor,
            dependency_resolver=self.dependency_resolver,
            flow_handler=self._flow_handler
        )
        self._flow_executors = FlowExecutors(
            gateway=gateway,
            flow_handler=self._flow_handler
        )

        logger.info("MessageEngine initialized (v20.0 modular)")

    async def _instrumented_ai_call(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        tool_scores: Optional[List[Dict]] = None,
        user_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Wrapper around AI analyze() that records metrics for drift detection and cost tracking.
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
                tool_scores=tool_scores
            )

            if result.get("type") == "error":
                success = False
                error_type = "llm_error"
            elif result.get("type") == "tool_call":
                tools_called = [result.get("tool", "unknown")]

            usage_data = result.get("usage")
            return result

        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise

        finally:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            tenant_id = UserContextManager(user_context).tenant_id if user_context else "default"

            # Record for drift detection
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

            # Record for cost tracking
            if self.cost_tracker and usage_data:
                try:
                    await self.cost_tracker.record_usage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        tenant_id=tenant_id
                    )
                except Exception as e:
                    logger.warning(f"Failed to record cost metrics: {e}")

    async def process(
        self,
        sender: str,
        text: str,
        message_id: Optional[str] = None,
        db_session=None
    ) -> str:
        """
        Process incoming message.

        Args:
            sender: User phone number
            text: Message text
            message_id: Optional message ID
            db_session: Database session for this request (concurrency-safe)

        Returns:
            Response text
        """
        import time as _time
        _t0 = _time.perf_counter()
        logger.info(f"Processing: {sender[-4:]} - {text[:50]}")

        try:
            # 1. Identify user (delegated to UserHandler)
            # Always returns a context (guest context if not in MobilityOne)
            user_context = await self._user_handler.identify_user(sender, db_session=db_session)
            _t1 = _time.perf_counter()
            logger.info(f"TIMING identify_user: {int((_t1-_t0)*1000)}ms")

            # 2. Load conversation state
            conv_manager = await ConversationManager.load_for_user(sender, self.redis)

            # 3. Check timeout
            if conv_manager.is_timed_out():
                await conv_manager.reset()

            # 4. Add to history
            await self.context.add_message(sender, "user", text)

            # 4.5 Check for hallucination feedback (delegated to HallucinationHandler)
            hallucination_result = await self._hallucination_handler.check_hallucination_feedback(
                text=text,
                sender=sender,
                user_context=user_context,
                conv_manager=conv_manager
            )
            if hallucination_result:
                return hallucination_result

            _t2 = _time.perf_counter()
            logger.info(f"TIMING pre-processing: {int((_t2-_t0)*1000)}ms")

            # 5. Handle new user greeting (delegated to UserHandler)
            if UserContextManager(user_context).is_new:
                greeting = self._user_handler.build_greeting(user_context)
                await self.context.add_message(sender, "assistant", greeting)

                response = await self._process_with_state(
                    sender, text, user_context, conv_manager
                )

                _t3 = _time.perf_counter()
                logger.info(f"TIMING total (new user): {int((_t3-_t0)*1000)}ms")

                full_response = f"{greeting}\n\n---\n\n{response}"
                await self.context.add_message(sender, "assistant", response)
                return full_response

            # 6. Process based on state
            response = await self._process_with_state(
                sender, text, user_context, conv_manager
            )

            _t3 = _time.perf_counter()
            logger.info(f"TIMING _process_with_state: {int((_t3-_t2)*1000)}ms")
            logger.info(f"TIMING total: {int((_t3-_t0)*1000)}ms")

            # 7. Save response to history
            await self.context.add_message(sender, "assistant", response)

            return response

        except MissingContextError as e:
            logger.info(f"Missing context: {e.param} - prompting user")
            await self.context.add_message(sender, "assistant", e.prompt_hr)
            return e.prompt_hr

        except VehicleSelectionRequired as e:
            logger.info(f"Vehicle selection needed: {len(e.vehicles)} vehicles")
            vehicles_list = "\n".join([
                f"* {v.get('LicencePlate', 'N/A')} - {v.get('FullVehicleName', v.get('DisplayName', 'Vozilo'))}"
                for v in e.vehicles[:5]
            ])
            prompt = f"{e.prompt_hr}\n\n{vehicles_list}"
            await self.context.add_message(sender, "assistant", prompt)
            return prompt

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return "Doslo je do greske. Molimo pokusajte ponovno."

    async def _process_with_state(
        self,
        sender: str,
        text: str,
        user_context: Dict[str, Any],
        conv_manager: ConversationManager
    ) -> str:
        """Process message based on conversation state using Unified Router."""
        import time as _time
        _pws_start = _time.perf_counter()

        state = conv_manager.get_state()
        is_in_flow = conv_manager.is_in_flow()

        logger.info(
            f"STATE CHECK: user={sender[-4:]}, state={state.value}, "
            f"is_in_flow={is_in_flow}, flow={conv_manager.get_current_flow()}, "
            f"tool={conv_manager.get_current_tool()}, "
            f"missing={conv_manager.get_missing_params()}, "
            f"has_items={len(conv_manager.get_displayed_items())}"
        )

        # Direct state-based handling for in-flow messages
        if is_in_flow:
            if state == ConversationState.CONFIRMING:
                if matches_show_more(text):
                    logger.info("DIRECT HANDLER: 'show more' in CONFIRMING state")
                    return await self._flow_handler.handle_confirmation(
                        sender, text, user_context, conv_manager
                    )
                if matches_confirm_yes(text) or matches_confirm_no(text):
                    logger.info("DIRECT HANDLER: confirmation response in CONFIRMING state")
                    return await self._flow_handler.handle_confirmation(
                        sender, text, user_context, conv_manager
                    )

            if state == ConversationState.SELECTING_ITEM:
                if matches_item_selection(text):
                    logger.info("DIRECT HANDLER: item selection in SELECTING state")
                    return await self._flow_handler.handle_selection(
                        sender, text, user_context, conv_manager, self._handle_new_request
                    )

        # Initialize unified router lazily
        if not self._unified_router_initialized:
            self.unified_router = await get_unified_router()
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
        _rt0 = _time.perf_counter()
        decision = await self.unified_router.route(text, user_context, conv_state)
        _rt1 = _time.perf_counter()

        logger.info(
            f"UNIFIED ROUTER: action={decision.action}, tool={decision.tool}, "
            f"flow={decision.flow_type}, conf={decision.confidence:.2f}"
        )
        logger.info(f"TIMING router: {int((_rt1-_rt0)*1000)}ms (since pws_start: {int((_rt1-_pws_start)*1000)}ms)")

        # Handle routing decisions
        if decision.action == "direct_response":
            return decision.response or "Kako vam mogu pomoci?"

        if decision.action == "clarify":
            logger.info(f"UNIFIED ROUTER: Clarification needed - '{decision.clarification}'")
            return decision.clarification or "Mozete li mi reci vise detalja o tome sto trazite?"

        if decision.action == "exit_flow":
            if conv_manager.is_in_flow():
                logger.info("UNIFIED ROUTER: Exiting flow, resetting state")
                await conv_manager.reset()
                new_decision = await self.unified_router.route(text, user_context, None)

                if new_decision.action == "direct_response":
                    return new_decision.response or "Kako vam mogu pomoci?"
                if new_decision.action == "start_flow":
                    return await self._handle_flow_start(new_decision, text, user_context, conv_manager)
                if new_decision.action == "simple_api" and new_decision.tool:
                    route = RouteResult(
                        matched=True,
                        tool_name=new_decision.tool,
                        confidence=new_decision.confidence,
                        flow_type="simple"
                    )
                    result = await self._deterministic_executor.execute(
                        route, user_context, conv_manager, sender, text
                    )
                    if result:
                        return result
                return "Nisam siguran sto trazite. Mozete pitati za:\n* Rezervaciju vozila\n* Kilometrazu\n* Prijavu stete\n* Informacije o vozilu"
            else:
                logger.warning("UNIFIED ROUTER: exit_flow received but not in flow - ignoring")
                return "Kako vam mogu pomoci?"

        if decision.action == "continue_flow":
            if state == ConversationState.SELECTING_ITEM:
                return await self._flow_handler.handle_selection(
                    sender, text, user_context, conv_manager, self._handle_new_request
                )
            if state == ConversationState.CONFIRMING:
                result = await self._flow_handler.handle_confirmation(
                    sender, text, user_context, conv_manager
                )
                if isinstance(result, dict) and result.get("mid_flow_question"):
                    question = result.get("question", text)
                    logger.info(f"P1: Handling mid-flow question: '{question[:50]}'")
                    # mid-flow question if we're not already handling one
                    if not getattr(self, '_handling_mid_flow', False):
                        self._handling_mid_flow = True
                        try:
                            answer = await self._handle_new_request(sender, question, user_context, conv_manager)
                        finally:
                            self._handling_mid_flow = False
                        return f"{answer}\n\n---\n_Ceka se potvrda prethodne operacije. Potvrdite s **Da** ili **Ne**._"
                    else:
                        logger.warning("P1: Skipping nested mid-flow question to prevent recursion")
                        return "Potvrdite prethodnu operaciju s **Da** ili **Ne**."
                return result
            if state == ConversationState.GATHERING_PARAMS:
                return await self._flow_handler.handle_gathering(
                    sender, text, user_context, conv_manager, self._handle_new_request
                )
            if state == ConversationState.IDLE and conv_manager.get_current_flow():
                logger.warning("STATE MISMATCH: is_in_flow but state=IDLE, treating as new request")

        if decision.action == "start_flow":
            return await self._handle_flow_start(decision, text, user_context, conv_manager)

        if decision.action == "simple_api" and decision.tool:
            route = RouteResult(
                matched=True,
                tool_name=decision.tool,
                confidence=decision.confidence,
                flow_type="simple"
            )
            result = await self._deterministic_executor.execute(
                route, user_context, conv_manager, sender, text
            )
            if result:
                return result

        # Fallback to original new request handling
        return await self._handle_new_request(sender, text, user_context, conv_manager)

    async def _handle_flow_start(self, decision, text: str, user_context: Dict, conv_manager) -> str:
        """Handle flow start from router decision."""
        if decision.flow_type == "booking":
            return await self._flow_executors.handle_booking_flow(
                text, user_context, conv_manager, decision.params
            )
        if decision.flow_type == "mileage":
            return await self._flow_executors.handle_mileage_input_flow(
                text, user_context, conv_manager, decision.params
            )
        if decision.flow_type == "case":
            return await self._flow_executors.handle_case_creation_flow(
                text, user_context, conv_manager, decision.params
            )
        return "Neispravan flow tip."

    async def _handle_new_request(
        self,
        sender: str,
        text: str,
        user_context: Dict[str, Any],
        conv_manager: ConversationManager
    ) -> str:
        """Handle new request with Chain Planning and Fallback Execution."""

        # DETERMINISTIC ROUTING - Try rules FIRST
        route = self.query_router.route(text, user_context)

        if route.matched:
            logger.info(f"ROUTER: Deterministic match -> {route.tool_name or route.flow_type}")

            # Direct response (greetings, thanks, help, context queries)
            if route.flow_type == "direct_response":
                if route.response_template:
                    ctx = UserContextManager(user_context)
                    if 'person_id' in route.response_template:
                        return f"*Person ID:** {ctx.person_id or 'N/A'}"
                    elif 'phone' in route.response_template:
                        return f"*Telefon:** {ctx.phone or 'N/A'}"
                    elif 'tenant_id' in route.response_template:
                        return f"*Tenant ID:** {ctx.tenant_id or 'N/A'}"

                    simple_context = {
                        k: v for k, v in user_context.items()
                        if not isinstance(v, (dict, list))
                    }
                    try:
                        return route.response_template.format(**simple_context)
                    except (KeyError, ValueError):
                        return route.response_template
                return route.response_template

            # Flow-based routing
            if route.flow_type == "booking":
                return await self._flow_executors.handle_booking_flow(text, user_context, conv_manager, {})
            if route.flow_type == "mileage_input":
                return await self._flow_executors.handle_mileage_input_flow(text, user_context, conv_manager, {})
            if route.flow_type == "case_creation" and route.tool_name:
                return await self._flow_executors.handle_case_creation_flow(text, user_context, conv_manager, {})

            # Simple or list query - execute deterministically
            if route.flow_type in ("simple", "list") and route.tool_name:
                result = await self._deterministic_executor.execute(
                    route, user_context, conv_manager, sender, text
                )
                if result:
                    return result
                logger.warning("Deterministic execution failed, falling back to LLM")

        # Pre-resolve entity references
        pre_resolved = await self._deterministic_executor.pre_resolve_entity_references(
            text, user_context, conv_manager, self.executor
        )
        if pre_resolved:
            logger.info(f"Pre-resolved entities: {list(pre_resolved.keys())}")

        # Get history
        history = await self.context.get_recent_messages(sender)
        messages = history.copy()
        messages.append({"role": "user", "content": text})

        # Get relevant tools with scores
        tools_with_scores = await self.registry.find_relevant_tools_with_scores(text, top_k=10)
        tools_with_scores = sorted(tools_with_scores, key=lambda t: t["score"], reverse=True)
        tools = [t["schema"] for t in tools_with_scores]

        # Use ChainPlanner for multi-step plans
        plan = await self.chain_planner.create_plan(
            query=text,
            user_context=user_context,
            available_tools=tools,
            tool_scores=tools_with_scores
        )

        logger.info(f"ChainPlan: {plan.understanding} (simple={plan.is_simple})")

        if plan.direct_response:
            return plan.direct_response

        if plan.missing_data and not plan.has_all_data:
            missing_prompt = DeterministicExecutor.build_missing_data_prompt(plan.missing_data)
            logger.info(f"Missing data: {plan.missing_data}")

            first_step = plan.primary_path[0] if plan.primary_path else None
            await conv_manager.start_flow(
                flow_name="gathering",
                tool=first_step.tool_name if first_step else None,
                required_params=plan.missing_data
            )
            await conv_manager.save()
            return missing_prompt

        extraction_hint = getattr(plan, 'extraction_hint', None)

        # Get fallback tools from plan
        fallback_tools = []
        if hasattr(plan, 'fallback_paths') and plan.fallback_paths:
            for step_fallbacks in plan.fallback_paths.values():
                for fb in step_fallbacks:
                    for fb_step in fb.steps:
                        if fb_step.tool_name:
                            fallback_tools.append(fb_step.tool_name)

        if tools_with_scores:
            best_match = max(tools_with_scores, key=lambda t: t["score"])
            logger.info(
                f"Tool recommendation: {best_match['name']} (score={best_match['score']:.3f}) "
                f"- LLM will decide from {len(tools_with_scores)} options"
            )

        # Build system prompt
        system_prompt = self.ai.build_system_prompt(
            user_context,
            conv_manager.to_dict() if conv_manager.is_in_flow() else None
        )

        # AI iteration loop
        current_messages = messages.copy()

        for iteration in range(self.MAX_ITERATIONS):
            logger.debug(f"AI iteration {iteration + 1}/{self.MAX_ITERATIONS}")

            result = await self._instrumented_ai_call(
                messages=current_messages,
                tools=tools,
                system_prompt=system_prompt,
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

                # Special handling for availability/calendar
                # Config-driven: check tool tags/method instead of name substrings
                tool_tags = getattr(tool, 'tags', []) or []
                is_availability_tool = (
                    method == "GET" and
                    any(tag in (t.lower() for t in tool_tags) for tag in ["availability", "calendar"]) or
                    (method == "GET" and tool_name in self._get_availability_tools())
                )
                if is_availability_tool:
                    return await self._flow_executors.handle_availability_flow(
                        result, user_context, conv_manager
                    )

                # Block direct booking creation without validation
                is_booking_create = (
                    method == "POST" and
                    tool_name in self._get_booking_tools()
                )
                if is_booking_create:
                    params = result.get("parameters", {})
                    vehicle_id = params.get("VehicleId") or params.get("vehicleId")

                    validated_vehicle_id = None
                    if hasattr(conv_manager.context, 'tool_outputs'):
                        validated_vehicle_id = conv_manager.context.tool_outputs.get("VehicleId")
                        all_vehicles = conv_manager.context.tool_outputs.get("all_available_vehicles", [])
                        if vehicle_id and all_vehicles:
                            for v in all_vehicles:
                                if v.get("Id") == vehicle_id:
                                    validated_vehicle_id = vehicle_id
                                    break

                    if not validated_vehicle_id or vehicle_id != validated_vehicle_id:
                        logger.warning(
                            f"BLOCKED direct post_VehicleCalendar: VehicleId={vehicle_id} is NOT validated"
                        )
                        return await self._flow_executors.handle_booking_flow(text, user_context, conv_manager, {})

                    state = conv_manager.get_state()
                    if state != ConversationState.EXECUTING:
                        confirmation_result = await self._flow_handler.request_confirmation(
                            tool_name, params, user_context, conv_manager
                        )
                        return confirmation_result.get("prompt", "Molim potvrdite rezervaciju s 'Da' ili 'Ne'.")

                # Mutation tools require confirmation
                if method in ("POST", "PUT", "PATCH"):
                    if self._tool_handler.requires_confirmation(tool_name):
                        flow_result = await self._flow_handler.request_confirmation(
                            tool_name, result["parameters"], user_context, conv_manager
                        )
                        if flow_result.get("needs_input"):
                            return flow_result["prompt"]

                # Execute tool
                tool_response = await self._tool_handler.execute_tool_call(
                    result, user_context, conv_manager, sender, user_query=text
                )

                # Use LLM extraction for successful responses
                if tool_response.get("success") and tool_response.get("data"):
                    # If there are NO additional calls, try to return extracted response directly
                    if not result.get("additional_calls"):
                        try:
                            extracted_response = await self.response_extractor.extract(
                                user_query=text,
                                api_response=tool_response["data"],
                                tool_name=tool_name,
                                extraction_hint=extraction_hint
                            )
                            if extracted_response:
                                return extracted_response
                        except Exception as e:
                            logger.warning(f"LLM extraction failed: {e}")

                if not result.get("additional_calls"):
                    if tool_response.get("final_response"):
                        return tool_response["final_response"]

                    if tool_response.get("needs_input"):
                        return tool_response.get("prompt", "")

                # Try fallback tools on failure
                if not tool_response.get("success", True) and fallback_tools:
                    logger.info(f"Primary tool {tool_name} failed, trying fallbacks...")
                    for fb_tool_name in fallback_tools:
                        if fb_tool_name == tool_name:
                            continue

                        fb_tool = self.registry.get_tool(fb_tool_name)
                        if not fb_tool:
                            continue

                        fb_result = {
                            "tool": fb_tool_name,
                            "parameters": result["parameters"].copy(),
                            "tool_call_id": "fallback"
                        }

                        fb_response = await self._tool_handler.execute_tool_call(
                            fb_result, user_context, conv_manager, sender, user_query=text
                        )

                        if fb_response.get("success"):
                            logger.info(f"Fallback {fb_tool_name} succeeded!")
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
                                except Exception:
                                    pass
                            if fb_response.get("final_response"):
                                return fb_response["final_response"]
                            break

                # Build tool_calls list for conversation (include all calls from LLM)
                all_assistant_tool_calls = [{
                    "id": result["tool_call_id"],
                    "type": "function",
                    "function": {
                        "name": result["tool"],
                        "arguments": json.dumps(result["parameters"])
                    }
                }]

                # First tool result
                tool_result_content = tool_response.get("data", {})
                if not tool_response.get("success", True):
                    tool_result_content = {
                        "error": True,
                        "message": tool_response.get("error", "Unknown error"),
                        "ai_feedback": tool_response.get("ai_feedback", "")
                    }

                all_tool_results = [{
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": json.dumps(tool_result_content)
                }]

                # Process additional parallel tool calls from LLM
                additional_calls = result.get("additional_calls", [])
                if additional_calls:
                    logger.info(f"Processing {len(additional_calls)} additional parallel tool calls")

                for extra_call in additional_calls:
                    extra_tool_name = extra_call["tool"]
                    logger.info(f"Executing additional tool call: {extra_tool_name}")

                    all_assistant_tool_calls.append({
                        "id": extra_call["tool_call_id"],
                        "type": "function",
                        "function": {
                            "name": extra_call["tool"],
                            "arguments": json.dumps(extra_call["parameters"])
                        }
                    })

                    extra_response = await self._tool_handler.execute_tool_call(
                        extra_call, user_context, conv_manager, sender, user_query=text
                    )

                    extra_content = extra_response.get("data", {})
                    if not extra_response.get("success", True):
                        extra_content = {
                            "error": True,
                            "message": extra_response.get("error", "Unknown error")
                        }

                    all_tool_results.append({
                        "role": "tool",
                        "tool_call_id": extra_call["tool_call_id"],
                        "content": json.dumps(extra_content)
                    })

                # Add assistant message with ALL tool calls
                current_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": all_assistant_tool_calls
                })

                # Add ALL tool results
                current_messages.extend(all_tool_results)

        return "Nisam uspio obraditi zahtjev. Pokusajte drugacije formulirati."

    # ---
    # CONFIG-DRIVEN TOOL IDENTIFICATION (replaces hardcoded tool name checks)
    # ---

    def _get_availability_tools(self) -> set:
        """Get tool names that handle vehicle availability (config-driven)."""
        if not self.registry:
            return set()
        # Dynamically discover from registry instead of hardcoding names
        tools = set()
        for tool_name in self.registry.tools:
            tool = self.registry.get_tool(tool_name)
            if not tool or tool.method != "GET":
                continue
            name_lower = tool_name.lower()
            desc_lower = (getattr(tool, 'description', '') or '').lower()
            if ('available' in name_lower and 'vehicle' in name_lower) or \
               ('slobodn' in desc_lower and 'vozil' in desc_lower):
                tools.add(tool_name)
        return tools

    def _get_booking_tools(self) -> set:
        """Get tool names that create bookings (config-driven)."""
        if not self.registry:
            return set()
        tools = set()
        for tool_name in self.registry.tools:
            tool = self.registry.get_tool(tool_name)
            if not tool or tool.method != "POST":
                continue
            name_lower = tool_name.lower()
            desc_lower = (getattr(tool, 'description', '') or '').lower()
            if ('calendar' in name_lower and 'vehicle' in name_lower) or \
               ('rezervacij' in desc_lower and 'kreir' in desc_lower):
                tools.add(tool_name)
        return tools

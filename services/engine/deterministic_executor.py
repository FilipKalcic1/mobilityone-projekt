"""
Deterministic Executor - Fast path execution without LLM.

Extracted from engine/__init__.py for better modularity.
"""

import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from services.conversation_manager import ConversationState
from services.context import (
    get_multiple_missing_prompts,
    UserContextManager,
)

if TYPE_CHECKING:
    from services.conversation_manager import ConversationManager
    from services.registry import ToolRegistry
    from services.tool_executor import ToolExecutor
    from services.response_formatter import ResponseFormatter
    from services.dependency_resolver import DependencyResolver
    from services.query_router import RouteResult
    from services.response_extractor import LLMResponseExtractor
    from .flow_handler import FlowHandler

logger = logging.getLogger(__name__)


class DeterministicExecutor:
    """
    Handles deterministic execution without LLM tool selection.

    This is the FAST PATH for known queries:
    - NO embedding search
    - NO LLM tool selection
    - GUARANTEED correct tool

    Responsibilities:
    - Execute matched routes directly
    - Pre-resolve entity references
    - Build missing data prompts
    """

    def __init__(
        self,
        registry: 'ToolRegistry',
        executor: 'ToolExecutor',
        formatter: 'ResponseFormatter',
        query_router,
        response_extractor: 'LLMResponseExtractor',
        dependency_resolver: 'DependencyResolver',
        flow_handler: 'FlowHandler'
    ):
        """
        Initialize DeterministicExecutor.

        Args:
            registry: Tool registry
            executor: Tool executor
            formatter: Response formatter
            query_router: Query router for template formatting
            response_extractor: LLM response extractor
            dependency_resolver: Dependency resolver for entities
            flow_handler: Flow handler for confirmations
        """
        self.registry = registry
        self.executor = executor
        self.formatter = formatter
        self.query_router = query_router
        self.response_extractor = response_extractor
        self.dependency_resolver = dependency_resolver
        self.flow_handler = flow_handler

    async def execute(
        self,
        route: 'RouteResult',
        user_context: Dict[str, Any],
        conv_manager: 'ConversationManager',
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
        ctx = UserContextManager(user_context)
        params = {}

        # Inject personIdOrEmail for PersonData endpoint
        if "personIdOrEmail" in tool.parameters:
            if ctx.person_id:
                params["personIdOrEmail"] = ctx.person_id
                logger.info(f"DETERMINISTIC: Injected personIdOrEmail={ctx.person_id[:8]}... for {route.tool_name}")
            else:
                logger.warning("DETERMINISTIC: No person_id in user_context for PersonData")
                return "Molim vas prijavite se kako bih mogao dohvatiti va\u0161e podatke."

        # Inject VehicleId if needed and available - use VehicleContext
        vehicle_id_param = tool.parameters.get("VehicleId")
        if vehicle_id_param and getattr(vehicle_id_param, 'required', False):
            if ctx.vehicle_id:
                params["VehicleId"] = ctx.vehicle_id
            else:
                # No vehicle - can't execute
                return (
                    "Trenutno nemate dodijeljeno vozilo.\n"
                    "\u017delite li rezervirati vozilo?"
                )

        # Create execution context
        from services.tool_contracts import ToolExecutionContext
        execution_context = ToolExecutionContext(
            user_context=user_context,
            tool_outputs=conv_manager.context.tool_outputs if hasattr(conv_manager.context, 'tool_outputs') else {},
            conversation_state={}
        )

        # Check if mutation tool requires confirmation
        is_mutation = tool.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}
        state = conv_manager.get_state()

        if is_mutation and state != ConversationState.CONFIRMING:
            # Mutation tool needs confirmation - show confirmation dialog
            logger.info(f"DETERMINISTIC: Mutation {route.tool_name} requires confirmation")
            result = await self.flow_handler.request_confirmation(
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

    async def pre_resolve_entity_references(
        self,
        text: str,
        user_context: Dict[str, Any],
        conv_manager: 'ConversationManager',
        executor: 'ToolExecutor'
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
                    executor=executor
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

    @staticmethod
    def build_missing_data_prompt(missing_params: List[str]) -> str:
        """
        Build user-friendly prompt for missing parameters.

        Uses centralized param_prompts from services.context module.
        Supports 30+ parameter types with Croatian user-friendly messages.
        """
        return get_multiple_missing_prompts(missing_params)

"""
Context Management Module

Centralized user context management - SINGLE SOURCE OF TRUTH.

This module solves the problem of 354+ scattered user_context.get() calls
that can fail silently or return wrong data.

Instead of:
    vehicle = user_context.get("vehicle", {})
    vehicle_id = vehicle.get("Id")  # Can be None!

Use:
    ctx = UserContextManager(user_context)
    vehicle_id = ctx.require_vehicle_id()  # Raises if missing
"""

from .user_context_manager import (
    UserContextManager,
    VehicleContext,
    MissingContextError,
    VehicleSelectionRequired,
    InvalidContextError,
)
from .param_prompts import (
    PARAM_PROMPTS,
    get_missing_param_prompt,
    get_multiple_missing_prompts,
)

__all__ = [
    "UserContextManager",
    "VehicleContext",
    "MissingContextError",
    "VehicleSelectionRequired",
    "InvalidContextError",
    "PARAM_PROMPTS",
    "get_missing_param_prompt",
    "get_multiple_missing_prompts",
]

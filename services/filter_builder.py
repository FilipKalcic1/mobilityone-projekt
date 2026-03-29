import re
from typing import Optional

from services.tool_contracts import UnifiedToolDefinition


class FilterBuilder:
    """
    Builds filter strings for API queries with SANITIZATION.

    SECURITY: All values are sanitized to prevent injection attacks.
    """

    # Allowlist: only alphanumeric, spaces, hyphens, dots, @, plus, underscores
    # This is safer than a blocklist because it rejects anything unexpected.
    _ALLOWED_CHARS = re.compile(r"[^a-zA-Z0-9čćžšđČĆŽŠĐ\s\-_.@+]")

    # Blocklist for dangerous SQL keywords and comment sequences
    _DANGEROUS_PATTERNS = re.compile(
        r"(--|\b(exec|execute|insert|update|delete|drop|truncate|union|select|"
        r"from|where|alter|create|grant|revoke|xp_)\b)",
        re.IGNORECASE
    )

    @staticmethod
    def _sanitize_value(value: str) -> str:
        """
        Sanitize filter value to prevent injection attacks.

        Uses allowlist approach: only permits safe characters.
        Blocklist as secondary defense for SQL keywords.
        """
        if not isinstance(value, str):
            value = str(value)

        # Primary: strip non-allowed characters
        sanitized = FilterBuilder._ALLOWED_CHARS.sub('', value)

        # Secondary: remove dangerous SQL keywords
        sanitized = FilterBuilder._DANGEROUS_PATTERNS.sub('', sanitized)

        # Trim and limit length to prevent buffer attacks
        sanitized = sanitized.strip()[:500]

        return sanitized

    @staticmethod
    def build_filter_string(tool: UnifiedToolDefinition, resolved_params: dict) -> Optional[str]:
        """
        Builds a filter string for parameters that are marked as filterable.
        e.g., Phone(contains)123456 and Name(=)John

        Args:
            tool: The tool definition containing parameter metadata.
            resolved_params: The dictionary of resolved parameters and their values.

        Returns:
            A filter string if any filterable parameters are found, otherwise None.

        SECURITY: All values are sanitized to prevent injection attacks.
        """
        filters = []
        for name, value in resolved_params.items():
            param_def = tool.parameters.get(name)
            if param_def and param_def.is_filterable:
                # SECURITY FIX: Sanitize value before interpolation
                safe_value = FilterBuilder._sanitize_value(value)
                if safe_value:  # Only add if sanitized value is not empty
                    # Build string: e.g., Phone(contains)123456
                    filters.append(f"{name}{param_def.preferred_operator}{safe_value}")

        return " and ".join(filters) if filters else None

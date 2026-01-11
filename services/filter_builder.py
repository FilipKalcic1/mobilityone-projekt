import re
from typing import Optional

from services.tool_contracts import UnifiedToolDefinition


class FilterBuilder:
    """
    Builds filter strings for API queries with SANITIZATION.

    SECURITY: All values are sanitized to prevent injection attacks.
    """

    # Characters that could be used for injection attacks
    # Removes: SQL keywords, special operators, quotes, semicolons
    DANGEROUS_PATTERNS = re.compile(
        r"(--|;|'|\"|\\|/\*|\*/|xp_|exec|execute|insert|update|delete|drop|"
        r"truncate|union|select|from|where|and\s+\d|or\s+\d)",
        re.IGNORECASE
    )

    @staticmethod
    def _sanitize_value(value: str) -> str:
        """
        Sanitize filter value to prevent injection attacks.

        Removes dangerous SQL/injection patterns while preserving
        legitimate search characters.
        """
        if not isinstance(value, str):
            value = str(value)

        # Remove dangerous patterns
        sanitized = FilterBuilder.DANGEROUS_PATTERNS.sub('', value)

        # Escape parentheses (used in filter syntax)
        sanitized = sanitized.replace('(', '').replace(')', '')

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

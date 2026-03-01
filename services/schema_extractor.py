"""
Schema Extractor - Extract data using Swagger schema

Extracts fields from API responses using output_keys from ToolRegistry.
NO hardcoded field names - everything comes from Swagger spec.

DEPENDS ON: tool_registry.py (for output_keys from parsed swagger)
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SchemaExtractor:
    """
    Extracts data from API responses using schema from ToolRegistry.
    
    Instead of hardcoded:
        data.get("LicencePlate") or data.get("Plate")  # WRONG
        
    Use schema-driven:
        extractor.get_field(data, tool, "LicencePlate")  # RIGHT - validates against schema
    """
    
    def __init__(self, registry=None):
        """
        Initialize with optional registry reference.
        
        Args:
            registry: ToolRegistry instance (for output_keys lookup)
        """
        self._registry = registry
    
    def set_registry(self, registry) -> None:
        """Set registry reference (for late binding)."""
        self._registry = registry
    
    def get_output_keys(self, operation_id: str) -> List[str]:
        """
        Get output keys for an operation from registry.
        
        Args:
            operation_id: Tool operation ID (e.g., "get_MasterData")
            
        Returns:
            List of field names from Swagger response schema
        """
        if not self._registry:
            # No registry - return empty list, _extract_fields will return all fields
            return []
        
        tool = self._registry.get_tool(operation_id)
        if not tool:
            return []
        
        return list(tool.output_keys) if tool.output_keys else []
    
    def extract_all(
        self,
        data: Union[Dict, List],
        operation_id: str
    ) -> Dict[str, Any]:
        """
        Extract all fields from response according to schema.
        
        Args:
            data: Raw API response
            operation_id: Tool operation ID
            
        Returns:
            Dict with all schema-defined fields that exist in data
        """
        output_keys = self.get_output_keys(operation_id)
        normalized = self._normalize_response(data)
        
        if isinstance(normalized, list):
            # Return first item's fields for single extraction
            if normalized:
                return self._extract_fields(normalized[0], output_keys)
            return {}
        
        return self._extract_fields(normalized, output_keys)
    
    def extract_list(
        self,
        data: Union[Dict, List],
        operation_id: str
    ) -> List[Dict[str, Any]]:
        """
        Extract all items from list response.
        
        Args:
            data: Raw API response
            operation_id: Tool operation ID
            
        Returns:
            List of dicts with schema-defined fields
        """
        output_keys = self.get_output_keys(operation_id)
        normalized = self._normalize_response(data)
        
        if not isinstance(normalized, list):
            normalized = [normalized] if normalized else []
        
        return [self._extract_fields(item, output_keys) for item in normalized]
    
    def get_field(
        self,
        data: Union[Dict, List],
        operation_id: str,
        field_name: str,
        default: Any = None
    ) -> Any:
        """
        Get specific field value, validating it exists in schema.
        
        Args:
            data: Raw API response
            operation_id: Tool operation ID
            field_name: Field to extract
            default: Default value if not found
            
        Returns:
            Field value or default
        """
        output_keys = self.get_output_keys(operation_id)
        
        # Validate field is in schema
        if output_keys and field_name not in output_keys:
            logger.debug(
                f"Field '{field_name}' not in schema for {operation_id}. "
                f"Available: {output_keys[:10]}..."
            )
        
        normalized = self._normalize_response(data)
        
        if isinstance(normalized, list):
            if normalized:
                return normalized[0].get(field_name, default)
            return default
        
        return normalized.get(field_name, default) if normalized else default
    
    def field_exists(self, operation_id: str, field_name: str) -> bool:
        """Check if field exists in schema for operation."""
        output_keys = self.get_output_keys(operation_id)
        return field_name in output_keys
    
    def _normalize_response(self, data: Any) -> Union[Dict, List]:
        """
        Normalize API response - unwrap common wrappers.
        
        Handles:
        - {"Data": [...]}
        - {"Items": [...]}
        - {"data": {...}}
        - Direct list or dict
        """
        if data is None:
            return {}
        
        if isinstance(data, list):
            return data
        
        if isinstance(data, dict):
            # Unwrap common API response wrappers
            if "Data" in data:
                return data["Data"]
            if "Items" in data:
                return data["Items"]
            if "data" in data:
                return data["data"]
            if "items" in data:
                return data["items"]
            return data
        
        return {}
    
    def _extract_fields(
        self,
        item: Dict,
        output_keys: List[str]
    ) -> Dict[str, Any]:
        """
        Extract ALL fields from item.
        
        output_keys is just metadata about what Swagger says SHOULD be there,
        but we return ALL non-null fields from actual API response.
        This ensures we never lose data.
        """
        if not isinstance(item, dict):
            return {}
        
        # ALWAYS return all fields - output_keys is just documentation
        # The API might return more fields than documented in Swagger
        return {k: v for k, v in item.items() if v is not None}
    
    def get_count(self, data: Union[Dict, List]) -> int:
        """Get count of items in response."""
        normalized = self._normalize_response(data)
        
        if isinstance(normalized, list):
            return len(normalized)
        
        return 1 if normalized else 0
    
    def is_ambiguous(self, data: Union[Dict, List]) -> bool:
        """Check if response has multiple items (ambiguous result)."""
        return self.get_count(data) > 1


# Module-level singleton
_extractor: Optional[SchemaExtractor] = None


def get_schema_extractor(registry=None) -> SchemaExtractor:
    """Get or create schema extractor singleton."""
    global _extractor
    
    if _extractor is None:
        _extractor = SchemaExtractor(registry)
    elif registry is not None:
        _extractor.set_registry(registry)
    
    return _extractor

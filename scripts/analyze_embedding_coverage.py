#!/usr/bin/env python
"""
Analyze embedding coverage across all tools.

This script loads all tools from the registry and analyzes
what percentage of path segments, output keys, and operation IDs
are covered by the hardcoded dictionaries.

Usage:
    python scripts/analyze_embedding_coverage.py
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.registry.embedding_coverage import EmbeddingCoverageTracker, CoverageReport
from services.tool_contracts import UnifiedToolDefinition, ParameterDefinition


def load_tools_from_cache() -> dict:
    """Load tools from the file cache if available."""
    cache_path = project_root / "data" / "tool_registry_cache.json"

    if not cache_path.exists():
        print(f"Cache not found at {cache_path}")
        return {}

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tools = {}
    for tool_id, tool_data in data.get("tools", {}).items():
        # Convert to UnifiedToolDefinition
        params = {}
        for p_name, p_data in tool_data.get("parameters", {}).items():
            params[p_name] = ParameterDefinition(
                name=p_data.get("name", p_name),
                param_type=p_data.get("param_type", "string"),
                required=p_data.get("required", False),
                description=p_data.get("description", ""),
                is_filterable=p_data.get("is_filterable", False),
            )

        tools[tool_id] = UnifiedToolDefinition(
            operation_id=tool_data.get("operation_id", tool_id),
            method=tool_data.get("method", "GET"),
            path=tool_data.get("path", ""),
            description=tool_data.get("description", ""),
            parameters=params,
            service_name=tool_data.get("service_name", ""),
            service_url=tool_data.get("service_url", ""),
            swagger_name=tool_data.get("swagger_name", ""),
            output_keys=tool_data.get("output_keys", []),
        )

    return tools


def create_mock_tools() -> dict:
    """Create mock tools for testing when cache is not available."""
    # Extract unique paths/operations from swagger files
    swagger_dir = project_root / "data" / "swagger"

    if not swagger_dir.exists():
        print(f"Swagger directory not found: {swagger_dir}")
        return {}

    tools = {}
    tool_id = 0

    for swagger_file in swagger_dir.glob("*.json"):
        try:
            with open(swagger_file, "r", encoding="utf-8") as f:
                swagger = json.load(f)

            service_name = swagger_file.stem
            paths = swagger.get("paths", {})

            for path, methods in paths.items():
                for method, details in methods.items():
                    if method.lower() not in ("get", "post", "put", "patch", "delete"):
                        continue

                    op_id = details.get("operationId", f"{method}_{path.replace('/', '_')}")
                    description = details.get("summary", "") or details.get("description", "")

                    # Extract output keys from response schema
                    output_keys = []
                    responses = details.get("responses", {})
                    for code, resp in responses.items():
                        if code.startswith("2"):
                            schema = resp.get("schema", {})
                            if "properties" in schema:
                                output_keys.extend(schema["properties"].keys())
                            elif "items" in schema and "properties" in schema["items"]:
                                output_keys.extend(schema["items"]["properties"].keys())

                    tools[op_id] = UnifiedToolDefinition(
                        operation_id=op_id,
                        method=method.upper(),
                        path=path,
                        description=description,
                        parameters={},
                        service_name=service_name,
                        service_url="",
                        swagger_name=service_name,
                        output_keys=output_keys[:20],  # Limit
                    )
                    tool_id += 1

        except Exception as e:
            print(f"Error processing {swagger_file}: {e}")

    return tools


def main():
    print("=" * 70)
    print("EMBEDDING COVERAGE ANALYSIS")
    print("=" * 70)

    # Try to load from cache first
    tools = load_tools_from_cache()

    if not tools:
        print("\nLoading tools from swagger files...")
        tools = create_mock_tools()

    if not tools:
        print("ERROR: No tools found. Cannot analyze coverage.")
        return

    print(f"\nAnalyzing {len(tools)} tools...")

    # Run coverage analysis
    tracker = EmbeddingCoverageTracker()
    report = tracker.analyze_coverage(tools)

    # Print detailed report
    tracker.print_report(report)

    # Save report to JSON
    report_path = project_root / "data" / "coverage_report.json"
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"\nReport saved to: {report_path}")

    # Print recommendations for missing mappings
    if report.unmapped_path_segments:
        print("\n" + "=" * 70)
        print("RECOMMENDED PATH_ENTITY_MAP ADDITIONS:")
        print("=" * 70)
        for segment in sorted(report.unmapped_path_segments)[:30]:
            print(f'    "{segment}": ("???", "???"),  # TODO: Add Croatian translation')

    if report.unmapped_output_keys:
        print("\n" + "=" * 70)
        print("RECOMMENDED OUTPUT_KEY_MAP ADDITIONS:")
        print("=" * 70)
        for key in sorted(report.unmapped_output_keys)[:30]:
            print(f'    "{key}": "???",  # TODO: Add Croatian translation')

    return report


if __name__ == "__main__":
    main()

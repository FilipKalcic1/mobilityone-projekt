#!/usr/bin/env python
"""
CI Coverage Check Script - Verify embedding dictionary coverage.

This script is designed to run in CI/CD pipelines to:
1. Verify that embedding dictionaries meet minimum coverage thresholds
2. Report unmapped terms that need attention
3. Fail the build if coverage drops below acceptable levels

Usage:
    python scripts/check_embedding_coverage.py [--min-coverage 80]

Exit codes:
    0: Coverage meets thresholds
    1: Coverage below thresholds
    2: Error running analysis

Configuration (environment variables):
    EMBEDDING_MIN_PATH_COVERAGE: Minimum path coverage % (default: 70)
    EMBEDDING_MIN_OUTPUT_COVERAGE: Minimum output key coverage % (default: 60)
    EMBEDDING_MIN_TOOL_COVERAGE: Minimum tool coverage % (default: 75)
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_tools_from_cache() -> dict:
    """Load tools from the file cache if available."""
    cache_path = project_root / "data" / "tool_registry_cache.json"

    if not cache_path.exists():
        return {}

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from services.tool_contracts import UnifiedToolDefinition, ParameterDefinition

    tools = {}
    for tool_id, tool_data in data.get("tools", {}).items():
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


def load_tools_from_swagger() -> dict:
    """Load tools from swagger files as fallback."""
    swagger_dir = project_root / "data" / "swagger"

    if not swagger_dir.exists():
        return {}

    from services.tool_contracts import UnifiedToolDefinition

    tools = {}
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

                    # Extract output keys
                    output_keys = []
                    responses = details.get("responses", {})
                    for code, resp in responses.items():
                        if code.startswith("2"):
                            schema = resp.get("schema", {})
                            if "properties" in schema:
                                output_keys.extend(schema["properties"].keys())

                    tools[op_id] = UnifiedToolDefinition(
                        operation_id=op_id,
                        method=method.upper(),
                        path=path,
                        description=details.get("summary", ""),
                        parameters={},
                        service_name=service_name,
                        service_url="",
                        swagger_name=service_name,
                        output_keys=output_keys[:20],
                    )

        except Exception as e:
            print(f"Warning: Error processing {swagger_file}: {e}", file=sys.stderr)

    return tools


def check_dictionary_coverage():
    """Check coverage of embedding dictionaries."""
    from services.registry.embedding_engine import EmbeddingEngine
    from services.registry.translation_helper import TranslationHelper

    engine = EmbeddingEngine()
    helper = TranslationHelper()

    stats = helper.get_coverage_stats()

    print("=" * 60)
    print("EMBEDDING DICTIONARY COVERAGE REPORT")
    print("=" * 60)
    print()
    print("Dictionary Statistics:")
    print(f"  PATH_ENTITY_MAP entries:    {stats['path_entity_map_entries']}")
    print(f"  OUTPUT_KEY_MAP entries:     {stats['output_key_map_entries']}")
    print(f"  CROATIAN_SYNONYMS groups:   {stats['synonym_groups']}")
    print(f"  Total synonyms:             {stats['total_synonyms']}")
    print()

    return stats


def check_tool_coverage(tools: dict, thresholds: dict):
    """Check coverage against actual tools."""
    from services.registry.embedding_coverage import EmbeddingCoverageTracker

    if not tools:
        print("Warning: No tools found to analyze", file=sys.stderr)
        return True, {}

    tracker = EmbeddingCoverageTracker()
    report = tracker.analyze_coverage(tools)

    print("Tool Coverage Analysis:")
    print(f"  Total tools analyzed:       {len(tools)}")
    print(f"  Path coverage:              {report.path_coverage_pct:.1f}%")
    print(f"  Output key coverage:        {report.output_coverage_pct:.1f}%")
    print(f"  Tool coverage:              {report.tool_coverage_pct:.1f}%")
    print()

    # Check thresholds
    passed = True
    failures = []

    if report.path_coverage_pct < thresholds["path"]:
        passed = False
        failures.append(f"Path coverage {report.path_coverage_pct:.1f}% < {thresholds['path']}%")

    if report.output_coverage_pct < thresholds["output"]:
        passed = False
        failures.append(f"Output coverage {report.output_coverage_pct:.1f}% < {thresholds['output']}%")

    if report.tool_coverage_pct < thresholds["tool"]:
        passed = False
        failures.append(f"Tool coverage {report.tool_coverage_pct:.1f}% < {thresholds['tool']}%")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        print()

    # Report unmapped terms (top 10)
    if report.unmapped_path_segments:
        print("Top unmapped path segments:")
        for seg in sorted(report.unmapped_path_segments)[:10]:
            print(f"  - {seg}")
        print()

    if report.unmapped_output_keys:
        print("Top unmapped output keys:")
        for key in sorted(report.unmapped_output_keys)[:10]:
            print(f"  - {key}")
        print()

    return passed, {
        "path_coverage": report.path_coverage_pct,
        "output_coverage": report.output_coverage_pct,
        "tool_coverage": report.tool_coverage_pct,
        "unmapped_paths": len(report.unmapped_path_segments),
        "unmapped_outputs": len(report.unmapped_output_keys),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check embedding dictionary coverage for CI/CD"
    )
    parser.add_argument(
        "--min-path-coverage",
        type=float,
        default=float(os.environ.get("EMBEDDING_MIN_PATH_COVERAGE", 70)),
        help="Minimum path segment coverage percentage"
    )
    parser.add_argument(
        "--min-output-coverage",
        type=float,
        default=float(os.environ.get("EMBEDDING_MIN_OUTPUT_COVERAGE", 60)),
        help="Minimum output key coverage percentage"
    )
    parser.add_argument(
        "--min-tool-coverage",
        type=float,
        default=float(os.environ.get("EMBEDDING_MIN_TOOL_COVERAGE", 75)),
        help="Minimum tool coverage percentage"
    )
    parser.add_argument(
        "--json-output",
        type=str,
        help="Output results to JSON file"
    )
    parser.add_argument(
        "--skip-tool-analysis",
        action="store_true",
        help="Skip tool analysis (only check dictionary stats)"
    )

    args = parser.parse_args()

    thresholds = {
        "path": args.min_path_coverage,
        "output": args.min_output_coverage,
        "tool": args.min_tool_coverage,
    }

    print()
    print("Coverage Thresholds:")
    print(f"  Minimum path coverage:      {thresholds['path']}%")
    print(f"  Minimum output coverage:    {thresholds['output']}%")
    print(f"  Minimum tool coverage:      {thresholds['tool']}%")
    print()

    try:
        # Check dictionary coverage
        stats = check_dictionary_coverage()

        # Check tool coverage if tools available
        results = {"dictionary_stats": stats}
        passed = True

        if not args.skip_tool_analysis:
            # Try cache first, then swagger
            tools = load_tools_from_cache()
            if not tools:
                print("Cache not found, loading from swagger files...")
                tools = load_tools_from_swagger()

            if tools:
                passed, coverage_results = check_tool_coverage(tools, thresholds)
                results["tool_coverage"] = coverage_results
            else:
                print("No tools found for analysis")

        # Output JSON if requested
        if args.json_output:
            results["thresholds"] = thresholds
            results["passed"] = passed
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.json_output}")

        # Final result
        print("=" * 60)
        if passed:
            print("RESULT: PASSED - Coverage meets all thresholds")
            return 0
        else:
            print("RESULT: FAILED - Coverage below thresholds")
            return 1

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())

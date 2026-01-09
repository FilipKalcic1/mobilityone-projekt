"""
Regenerate missing tool documentation.
Reads missing tools from file and calls generate_documentation in delta mode.
"""
import asyncio
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_documentation import DocumentationGenerator
from services.registry.swagger_parser import SwaggerParser


async def main():
    print("=" * 60)
    print("REGENERATING MISSING TOOL DOCUMENTATION")
    print("=" * 60)

    # Find missing tools
    config_dir = Path(__file__).parent.parent / "config"

    registry = json.load(open(config_dir / "processed_tool_registry.json", encoding='utf-8'))
    docs = json.load(open(config_dir / "tool_documentation.json", encoding='utf-8'))

    tools_data = registry.get('tools', [])
    all_tools = set(t.get('operation_id', t.get('id', '')) for t in tools_data)
    documented_tools = set(docs.keys())

    missing = sorted(all_tools - documented_tools)

    print(f"Total tools: {len(all_tools)}")
    print(f"Already documented: {len(documented_tools)}")
    print(f"Missing: {len(missing)}")
    print()

    if not missing:
        print("All tools are already documented!")
        return

    print(f"Missing tools: {missing[:10]}... (and {len(missing)-10} more)")
    print()

    # Load ALL tool definitions from registry (not just missing)
    # This prevents _merge_documentation from deleting existing docs
    parser = SwaggerParser()
    tools_dict = {}
    # Import here to avoid circular imports
    from services.tool_contracts import UnifiedToolDefinition, ParameterDefinition, DependencySource

    for tool in tools_data:
        op_id = tool.get('operation_id', tool.get('id', ''))

        # Parameters is already a dict with ParameterDefinition-like structure
        params = {}
        for param_name, param_data in tool.get('parameters', {}).items():
            dep_source = param_data.get('dependency_source', 'user_input')
            if isinstance(dep_source, str):
                dep_source = DependencySource(dep_source)

            params[param_name] = ParameterDefinition(
                name=param_data.get('name', param_name),
                param_type=param_data.get('param_type', 'string'),
                required=param_data.get('required', False),
                description=param_data.get('description', ''),
                location=param_data.get('location', 'query'),
                dependency_source=dep_source
            )

        tools_dict[op_id] = UnifiedToolDefinition(
            operation_id=op_id,
            service_name=tool.get('service_name', 'unknown'),
            service_url=tool.get('service_url', ''),
            swagger_name=tool.get('swagger_name', ''),
            method=tool.get('method', 'GET'),
            path=tool.get('path', ''),
            description=tool.get('description', ''),
            summary=tool.get('summary', ''),
            parameters=params,
            required_params=tool.get('required_params', []),
            output_keys=tool.get('output_keys', []),
            tags=tool.get('tags', []),
            embedding_text=tool.get('embedding_text', ''),
            version_hash=tool.get('version_hash', '')
        )

    print(f"Loaded ALL {len(tools_dict)} tool definitions")
    print(f"Will regenerate documentation for {len(missing)} missing tools only")
    print()

    # Generate documentation
    generator = DocumentationGenerator()
    result = await generator.generate_all(
        tools_dict,
        delta_mode=True,
        changed_tools=missing
    )

    print()
    print("=" * 60)
    print("REGENERATION COMPLETE")
    print(f"Documentation entries: {len(result.get('documentation', {}))}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

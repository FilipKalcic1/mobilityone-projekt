import asyncio
import json
import logging
import os
import sys

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import get_settings
from services.registry.swagger_parser import SwaggerParser
from services.registry.embedding_engine import EmbeddingEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sync_tools")

async def main():
    """
    Parses Swagger specifications, enriches them with custom rules,
    and saves them to a single JSON file for fast startup.
    """
    logger.info("Starting tool registry sync...")
    
    try:
        settings = get_settings()
    except Exception as e:
        logger.error(f"Failed to load settings. Make sure .env file is configured. Error: {e}")
        return

    parser = SwaggerParser()
    embedding_engine = EmbeddingEngine()

    all_tools_map = {}
    for source in settings.swagger_sources:
        logger.info(f"Parsing {source}...")
        
        tools = await parser.parse_spec(source, embedding_engine.build_embedding_text)
        for tool in tools:
            if tool.operation_id not in all_tools_map:
                all_tools_map[tool.operation_id] = tool

    logger.info(f"Parsed {len(all_tools_map)} unique tools in total.")

    # Build dependency graph
    logger.info("Building dependency graph...")
    dep_graph = embedding_engine.build_dependency_graph(all_tools_map)
    
    # Prepare for serialization
    tools_as_dict = [tool.model_dump(mode='json') for tool in all_tools_map.values()]
    dep_graph_as_dict = [dep.model_dump(mode='json') for dep in dep_graph.values()]

    output_data = {
        "version": "1.0",
        "tools": tools_as_dict,
        "dependency_graph": dep_graph_as_dict
    }

    # Save to JSON file in config directory
    output_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'processed_tool_registry.json')
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved tool registry to {output_path}")
    except IOError as e:
        logger.error(f"Failed to write to {output_path}. Error: {e}")

if __name__ == "__main__":
    # This script requires environment variables to be set (e.g., from .env file)
    # to access MOBILITY_API_URL etc.
    asyncio.run(main())

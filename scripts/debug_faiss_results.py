"""
Debug FAISS Results - See exactly what FAISS returns for specific queries.
"""

import asyncio
import sys
import io
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.query_type_classifier import classify_query_type
from services.action_intent_detector import detect_action_intent


async def debug_query(query: str, expected_tool: str):
    """Debug FAISS results for a specific query."""

    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"EXPECTED: {expected_tool}")
    print(f"{'='*80}")

    # Classification
    intent_result = detect_action_intent(query)
    query_type_result = classify_query_type(query)

    print(f"\nIntent: {intent_result.intent.value} (conf={intent_result.confidence:.2f})")
    print(f"QueryType: {query_type_result.query_type.value} (conf={query_type_result.confidence:.2f})")
    print(f"Preferred suffixes: {query_type_result.preferred_suffixes}")
    print(f"Excluded suffixes: {query_type_result.excluded_suffixes}")

    # Load tool documentation
    doc_path = Path(__file__).parent.parent / "config" / "tool_documentation.json"
    with open(doc_path, 'r', encoding='utf-8') as f:
        tool_documentation = json.load(f)

    # Initialize FAISS
    from services.faiss_vector_store import initialize_faiss_store, get_faiss_store

    faiss_store = get_faiss_store()
    if not faiss_store.is_initialized():
        await initialize_faiss_store(tool_documentation)
        faiss_store = get_faiss_store()

    # Raw FAISS search (no filtering)
    print("\n--- RAW FAISS RESULTS (no filter) ---")
    raw_results = await faiss_store.search(query, top_k=10, action_filter=None)
    for i, r in enumerate(raw_results):
        marker = " <-- EXPECTED" if r.tool_id == expected_tool else ""
        print(f"  {i+1}. {r.tool_id} (score={r.score:.4f}){marker}")

    # FAISS with action filter
    action_filter = intent_result.intent.value if intent_result.intent.name != "UNKNOWN" else None
    if action_filter:
        print(f"\n--- FAISS WITH ACTION FILTER ({action_filter}) ---")
        filtered_results = await faiss_store.search(query, top_k=10, action_filter=action_filter)
        for i, r in enumerate(filtered_results):
            marker = " <-- EXPECTED" if r.tool_id == expected_tool else ""
            print(f"  {i+1}. {r.tool_id} (score={r.score:.4f}){marker}")

    # Check if expected tool exists in documentation
    if expected_tool in tool_documentation:
        doc = tool_documentation[expected_tool]
        print(f"\n--- EXPECTED TOOL DOCUMENTATION ---")
        print(f"  Purpose: {doc.get('purpose', 'N/A')[:100]}")
        print(f"  Example queries: {doc.get('example_queries_hr', [])[:3]}")


async def main():
    """Debug problematic queries."""

    problematic = [
        ("dohvati sve kompanije", "get_Companies"),
        ("dohvati kompaniju po ID-u", "get_Companies_id"),
        ("dohvati sva vozila", "get_Vehicles"),
        ("dohvati vozilo po ID-u", "get_Vehicles_id"),
        ("dohvati sve zaposlenike", "get_Persons"),
        ("kalendar vozila", "get_VehicleCalendar"),
        ("koliko km ima vozilo", "get_MasterData"),
    ]

    for query, expected in problematic:
        await debug_query(query, expected)


if __name__ == "__main__":
    asyncio.run(main())

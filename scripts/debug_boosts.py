"""
Debug boost application for problematic queries.
"""

import asyncio
import sys
import io
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent))


async def debug_query(query: str, expected_tool: str):
    """Debug boost application."""

    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"EXPECTED: {expected_tool}")
    print(f"{'='*80}")

    # Load and initialize
    doc_path = Path(__file__).parent.parent / "config" / "tool_documentation.json"
    with open(doc_path, 'r', encoding='utf-8') as f:
        tool_documentation = json.load(f)

    from services.faiss_vector_store import initialize_faiss_store, get_faiss_store
    from services.unified_search import initialize_unified_search
    from services.query_type_classifier import classify_query_type
    from services.action_intent_detector import detect_action_intent

    faiss_store = get_faiss_store()
    if not faiss_store.is_initialized():
        await initialize_faiss_store(tool_documentation)
        faiss_store = get_faiss_store()

    unified_search = await initialize_unified_search()

    # Classification
    intent = detect_action_intent(query)
    query_type = classify_query_type(query)

    print(f"\nIntent: {intent.intent.value} (conf={intent.confidence:.2f})")
    print(f"QueryType: {query_type.query_type.value} (conf={query_type.confidence:.2f})")

    # Get raw FAISS results
    action_filter = intent.intent.value if intent.intent.name != "UNKNOWN" else None
    raw_results = await faiss_store.search(query, top_k=15, action_filter=action_filter)

    print(f"\n--- RAW FAISS (top 15) ---")
    for i, r in enumerate(raw_results[:10]):
        marker = " <-- EXPECTED" if r.tool_id == expected_tool else ""
        print(f"  {i+1}. {r.tool_id[:50]:<50} score={r.score:.4f}{marker}")

    # Now run unified search to see boosted results
    response = await unified_search.search(query, top_k=15)

    print(f"\n--- BOOSTED RESULTS (top 15) ---")
    for i, r in enumerate(response.results[:10]):
        marker = " <-- EXPECTED" if r.tool_id == expected_tool else ""
        boosts = r.boosts_applied if r.boosts_applied else []
        print(f"  {i+1}. {r.tool_id[:50]:<50} score={r.score:.4f} boosts={boosts}{marker}")


async def main():
    """Debug problematic queries."""

    problematic = [
        ("dohvati sva vozila", "get_Vehicles"),
        ("dohvati vozilo po ID-u", "get_Vehicles_id"),
        ("dohvati sve zaposlenike", "get_Persons"),
        ("dohvati sve troskove", "get_Expenses"),
    ]

    for query, expected in problematic:
        await debug_query(query, expected)


if __name__ == "__main__":
    asyncio.run(main())

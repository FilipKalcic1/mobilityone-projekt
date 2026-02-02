"""
Test Full Integration - FAISS + ACTION INTENT GATE + Registry.

Verifies that:
1. FAISS initializes correctly from cache
2. ACTION INTENT GATE filters correctly
3. Integration with Registry works
4. FLOWS are not affected

Usage:
    python scripts/test_full_integration.py
"""

import asyncio
import sys
import io
import json
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()


async def test_faiss_initialization():
    """Test FAISS initializes from cache."""
    print("=" * 70)
    print("TEST 1: FAISS Initialization")
    print("=" * 70)

    from services.faiss_vector_store import FAISSVectorStore

    # Load tool documentation
    tool_doc_path = project_root / "config" / "tool_documentation.json"
    with open(tool_doc_path, 'r', encoding='utf-8') as f:
        tool_documentation = json.load(f)

    faiss_store = FAISSVectorStore()
    await faiss_store.initialize(tool_documentation)

    stats = faiss_store.get_stats()
    print(f"Total tools indexed: {stats['total_tools']}")
    print(f"Initialized: {faiss_store.is_initialized()}")

    assert faiss_store.is_initialized(), "FAISS should be initialized"
    assert stats['total_tools'] > 900, f"Expected 900+ tools, got {stats['total_tools']}"

    print("[PASS] FAISS initialization")
    return True


async def test_action_intent_gate():
    """Test ACTION INTENT correctly filters queries."""
    print()
    print("=" * 70)
    print("TEST 2: ACTION INTENT GATE")
    print("=" * 70)

    from services.action_intent_detector import detect_action_intent

    test_cases = [
        ("unesi kilometrazu", "POST"),
        ("koliko imam kilometara", "GET"),
        ("rezerviraj vozilo", "POST"),
        ("moje rezervacije", "GET"),
        ("obrisi rezervaciju", "DELETE"),
        ("promijeni registraciju", "PUT"),
    ]

    passed = 0
    for query, expected in test_cases:
        result = detect_action_intent(query)
        actual = result.intent.value
        is_pass = actual == expected
        status = "[PASS]" if is_pass else "[FAIL]"
        print(f"{status} '{query}' -> {actual} (expected {expected})")
        if is_pass:
            passed += 1

    assert passed == len(test_cases), f"Expected all {len(test_cases)} to pass, got {passed}"
    print(f"[PASS] ACTION INTENT GATE ({passed}/{len(test_cases)})")
    return True


async def test_faiss_search_with_intent():
    """Test FAISS search correctly uses intent filter."""
    print()
    print("=" * 70)
    print("TEST 3: FAISS Search with Intent Filter")
    print("=" * 70)

    from services.faiss_vector_store import FAISSVectorStore
    from services.action_intent_detector import detect_action_intent

    # Initialize FAISS
    tool_doc_path = project_root / "config" / "tool_documentation.json"
    with open(tool_doc_path, 'r', encoding='utf-8') as f:
        tool_documentation = json.load(f)

    faiss_store = FAISSVectorStore()
    await faiss_store.initialize(tool_documentation)

    # Test: "unesi km" should return POST tools
    query1 = "unesi kilometrazu"
    intent1 = detect_action_intent(query1)
    results1 = await faiss_store.search(query1, top_k=5, action_filter=intent1.intent.value)

    print(f"\nQuery: '{query1}'")
    print(f"Intent: {intent1.intent.value}")
    print(f"Top 3 results:")
    for i, r in enumerate(results1[:3], 1):
        print(f"  {i}. {r.tool_id} ({r.method})")

    # Verify POST tools returned
    has_post = any(r.method == "POST" for r in results1[:3])
    print(f"Has POST tool in top 3: {has_post}")
    assert has_post, "Should have POST tool for 'unesi km'"

    # Test: "koliko km" should return GET tools
    query2 = "koliko imam kilometara"
    intent2 = detect_action_intent(query2)
    results2 = await faiss_store.search(query2, top_k=5, action_filter=intent2.intent.value)

    print(f"\nQuery: '{query2}'")
    print(f"Intent: {intent2.intent.value}")
    print(f"Top 3 results:")
    for i, r in enumerate(results2[:3], 1):
        print(f"  {i}. {r.tool_id} ({r.method})")

    # Verify GET tools returned
    has_get = any(r.method == "GET" for r in results2[:3])
    print(f"Has GET tool in top 3: {has_get}")
    assert has_get, "Should have GET tool for 'koliko km'"

    # Verify disambiguation - results should be different
    tools1 = set(r.tool_id for r in results1[:3])
    tools2 = set(r.tool_id for r in results2[:3])
    overlap = tools1 & tools2

    print(f"\nOverlap between queries: {overlap}")
    assert len(overlap) < 2, f"Too much overlap: {overlap}"

    print("[PASS] FAISS search with intent filter")
    return True


async def test_flows_not_affected():
    """Test that FLOWS detection still works."""
    print()
    print("=" * 70)
    print("TEST 4: FLOWS Detection (not affected by FAISS)")
    print("=" * 70)

    from services.query_router import QueryRouter

    router = QueryRouter()

    # These should trigger flows
    flow_queries = [
        ("trebam vozilo za sutra", "booking"),
        ("rezerviraj auto", "booking"),
        ("unesi km 12500", "mileage_input"),
        ("prijavi kvar na vozilu", "case_creation"),
    ]

    passed = 0
    for query, expected_flow in flow_queries:
        result = router.route(query, {"vehicle": {"Id": "test"}})
        actual_flow = result.flow_type if result.matched else None
        is_pass = actual_flow == expected_flow
        status = "[PASS]" if is_pass else "[FAIL]"
        print(f"{status} '{query}' -> flow={actual_flow} (expected {expected_flow})")
        if is_pass:
            passed += 1

    assert passed >= 3, f"Expected at least 3/4 flows to work, got {passed}"
    print(f"[PASS] FLOWS detection ({passed}/{len(flow_queries)})")
    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("FULL INTEGRATION TEST")
    print("=" * 70 + "\n")

    try:
        await test_faiss_initialization()
        await test_action_intent_gate()
        await test_faiss_search_with_intent()
        await test_flows_not_affected()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
        return True

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

"""
Live Test of LLM Disambiguation
Version: 1.0

Tests the actual LLM routing with disambiguation for generic queries.
Requires Azure OpenAI credentials to be set.

This script:
1. Initializes UnifiedRouter with registry
2. Tests generic queries that should trigger disambiguation
3. Tests specific queries that should NOT trigger disambiguation
4. Verifies LLM returns appropriate responses
"""

import asyncio
import sys
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    print("=" * 70)
    print("LIVE LLM DISAMBIGUATION TEST")
    print("=" * 70)

    # Check environment
    from config import get_settings
    settings = get_settings()

    if not settings.AZURE_OPENAI_API_KEY:
        print("ERROR: AZURE_OPENAI_API_KEY not set")
        return 1

    print(f"Azure endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
    print(f"Model: {settings.AZURE_OPENAI_DEPLOYMENT_NAME}")

    # Initialize registry
    print("\nInitializing registry...")
    from services.registry import ToolRegistry
    registry = ToolRegistry(redis_client=None)
    await registry.initialize(settings.swagger_sources)
    print(f"Registry loaded: {len(registry.tools)} tools")

    # Initialize router
    print("\nInitializing UnifiedRouter v2.0...")
    from services.unified_router import UnifiedRouter
    router = UnifiedRouter(registry=registry)
    await router.initialize()
    print("Router initialized")

    # Test queries
    test_cases = [
        # Generic queries that SHOULD trigger disambiguation
        {
            "query": "Daj mi prosječnu vrijednost",
            "expect_clarify": True,
            "description": "Generic aggregation (no entity)"
        },
        {
            "query": "Grupiraj podatke",
            "expect_clarify": True,
            "description": "Generic grouping (no entity)"
        },

        # Specific queries that should NOT trigger disambiguation
        {
            "query": "Koliko kilometara ima moje vozilo?",
            "expect_clarify": False,
            "expected_tool": "get_MasterData",
            "description": "Specific vehicle query"
        },
        {
            "query": "Rezerviraj vozilo za sutra",
            "expect_clarify": False,
            "expected_action": "start_flow",
            "description": "Booking flow"
        },
        {
            "query": "Prijavi štetu",
            "expect_clarify": False,
            "expected_action": "start_flow",
            "description": "Case creation flow"
        },
        {
            "query": "Prikaži moje troškove",
            "expect_clarify": False,
            "expected_tool": "get_Expenses",
            "description": "Specific expenses query"
        },
    ]

    # Run tests
    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    user_context = {"vehicle": {}, "person_id": None}
    results = []

    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        print(f"\n[Test {i}] {test['description']}")
        print(f"  Query: '{query}'")

        try:
            decision = await router.route(query, user_context, None)

            print(f"  Action: {decision.action}")
            print(f"  Tool: {decision.tool}")
            print(f"  Confidence: {decision.confidence:.2f}")
            print(f"  Ambiguity detected: {decision.ambiguity_detected}")

            if decision.action == "clarify":
                print(f"  Clarification: '{decision.clarification}'")

            # Check expectations
            passed = True
            if test.get("expect_clarify"):
                if decision.action != "clarify":
                    print(f"  ✗ Expected 'clarify' action, got '{decision.action}'")
                    passed = False
                else:
                    print(f"  ✓ Correctly triggered clarification")

            elif test.get("expected_tool"):
                if decision.tool != test["expected_tool"]:
                    print(f"  ✗ Expected tool '{test['expected_tool']}', got '{decision.tool}'")
                    # Partial pass if tool is related
                    if decision.tool and test["expected_tool"].lower() in decision.tool.lower():
                        print(f"    (partial match)")
                        passed = True
                    else:
                        passed = False
                else:
                    print(f"  ✓ Correct tool selected")

            elif test.get("expected_action"):
                if decision.action != test["expected_action"]:
                    print(f"  ✗ Expected action '{test['expected_action']}', got '{decision.action}'")
                    passed = False
                else:
                    print(f"  ✓ Correct action")

            results.append((test["description"], passed))

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append((test["description"], False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, p in results if p)
    total = len(results)

    for name, passed in results:
        icon = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        print(f"  {icon} {name}: {status}")

    print(f"\nTotal: {passed_count}/{total} tests passed")
    print("=" * 70)

    return 0 if passed_count == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

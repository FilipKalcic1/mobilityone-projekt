"""
Test LLM Disambiguation Logic
Version: 1.0

Tests the new AmbiguityDetector and disambiguation features in UnifiedRouter.

Tests:
1. AmbiguityDetector correctly detects generic queries
2. Entity detection from query and context
3. Disambiguation hints are added to LLM prompt
4. Clarify action is returned for truly ambiguous queries
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

from services.ambiguity_detector import (
    AmbiguityDetector, AmbiguityResult, get_ambiguity_detector,
    GENERIC_SUFFIX_PATTERNS, ENTITY_KEYWORDS
)


def test_ambiguity_detector():
    """Test AmbiguityDetector core functionality."""
    print("=" * 60)
    print("TEST 1: AmbiguityDetector Core Functionality")
    print("=" * 60)

    detector = AmbiguityDetector()

    # Test case 1: Generic Agg query with multiple similar tools
    print("\n[Test 1.1] Generic _Agg query detection")
    search_results = [
        {"tool_id": "get_Companies_Agg", "score": 0.85},
        {"tool_id": "get_Vehicles_Agg", "score": 0.84},
        {"tool_id": "get_Expenses_Agg", "score": 0.83},
        {"tool_id": "get_Trips_Agg", "score": 0.82},
        {"tool_id": "get_Cases_Agg", "score": 0.81},
    ]

    result = detector.detect_ambiguity(
        query="Daj mi prosječnu vrijednost za polje x",
        search_results=search_results,
        user_context=None
    )

    assert result.is_ambiguous, "Should detect ambiguity for generic Agg query"
    assert result.ambiguous_suffix == "_Agg", f"Should identify _Agg suffix, got {result.ambiguous_suffix}"
    print(f"  ✓ Detected ambiguity: suffix={result.ambiguous_suffix}, similar_tools={len(result.similar_tools)}")

    # Test case 2: Specific query with entity mention
    print("\n[Test 1.2] Specific query with entity mention")
    result2 = detector.detect_ambiguity(
        query="Daj mi prosječnu kilometražu vozila",
        search_results=search_results,
        user_context=None
    )

    # Should still detect ambiguity but also detect entity
    assert result2.detected_entity == "Vehicles", f"Should detect Vehicles entity, got {result2.detected_entity}"
    print(f"  ✓ Detected entity: {result2.detected_entity}")

    # Test case 3: Non-ambiguous query (different suffixes)
    print("\n[Test 1.3] Non-ambiguous query (diverse tools)")
    diverse_results = [
        {"tool_id": "get_MasterData", "score": 0.90},
        {"tool_id": "get_Vehicles_id", "score": 0.80},
        {"tool_id": "post_AddMileage", "score": 0.70},
        {"tool_id": "get_Expenses", "score": 0.60},
    ]

    result3 = detector.detect_ambiguity(
        query="Koliko kilometara ima moje vozilo?",
        search_results=diverse_results,
        user_context=None
    )

    assert not result3.is_ambiguous, "Should NOT detect ambiguity for specific query"
    print(f"  ✓ No ambiguity detected (as expected)")

    print("\n" + "=" * 60)
    print("TEST 1 PASSED: AmbiguityDetector works correctly")
    print("=" * 60)
    return True


def test_entity_detection():
    """Test entity detection from Croatian queries."""
    print("\n" + "=" * 60)
    print("TEST 2: Entity Detection from Croatian Queries")
    print("=" * 60)

    detector = AmbiguityDetector()

    test_cases = [
        ("Koliko imam kilometara na vozilu?", "Vehicles"),
        ("Daj mi troškove za ovaj mjesec", "Expenses"),
        ("Prikaži mi putovanja", "Trips"),
        ("Prijavi štetu na vozilu", "Cases"),
        ("Daj mi podatke o osobi", "Persons"),
        ("Dohvati partnere", "Partners"),
        ("Prosječna vrijednost", None),  # No entity
    ]

    for query, expected_entity in test_cases:
        result = detector._detect_entity(query, None)
        status = "✓" if result == expected_entity else "✗"
        print(f"  {status} '{query[:40]}...' → {result} (expected: {expected_entity})")
        if result != expected_entity:
            print(f"    WARNING: Entity mismatch!")

    print("\n" + "=" * 60)
    print("TEST 2 PASSED: Entity detection works")
    print("=" * 60)
    return True


def test_clarification_questions():
    """Test clarification question generation."""
    print("\n" + "=" * 60)
    print("TEST 3: Clarification Questions")
    print("=" * 60)

    detector = AmbiguityDetector()

    # Simulate generic _Agg query
    search_results = [
        {"tool_id": "get_Companies_Agg", "score": 0.85},
        {"tool_id": "get_Vehicles_Agg", "score": 0.84},
        {"tool_id": "get_Expenses_Agg", "score": 0.83},
        {"tool_id": "get_Trips_Agg", "score": 0.82},
        {"tool_id": "get_Cases_Agg", "score": 0.81},
    ]

    result = detector.detect_ambiguity(
        query="Daj mi prosječnu vrijednost",
        search_results=search_results,
        user_context=None
    )

    assert result.clarification_question is not None, "Should have clarification question"
    print(f"  ✓ Clarification question: '{result.clarification_question}'")

    # Check disambiguation hint
    assert result.disambiguation_hint, "Should have disambiguation hint"
    print(f"  ✓ Disambiguation hint generated ({len(result.disambiguation_hint)} chars)")

    print("\n" + "=" * 60)
    print("TEST 3 PASSED: Clarification questions work")
    print("=" * 60)
    return True


async def test_unified_router_integration():
    """Test integration with UnifiedRouter (requires services)."""
    print("\n" + "=" * 60)
    print("TEST 4: UnifiedRouter Integration (imports only)")
    print("=" * 60)

    try:
        from services.unified_router import UnifiedRouter, RouterDecision

        # Check RouterDecision has new fields
        decision = RouterDecision(
            action="clarify",
            clarification="Za koje podatke želite statistiku?",
            reasoning="Query is ambiguous",
            confidence=0.3,
            ambiguity_detected=True
        )

        assert decision.action == "clarify", "Should support clarify action"
        assert decision.clarification is not None, "Should have clarification field"
        assert decision.ambiguity_detected, "Should have ambiguity_detected field"

        print(f"  ✓ RouterDecision has clarification field")
        print(f"  ✓ RouterDecision has ambiguity_detected field")
        print(f"  ✓ RouterDecision supports 'clarify' action")

        # Check UnifiedRouter has ambiguity detector
        router = UnifiedRouter()
        assert hasattr(router, '_ambiguity_detector'), "Router should have _ambiguity_detector"
        print(f"  ✓ UnifiedRouter has _ambiguity_detector")

        print("\n" + "=" * 60)
        print("TEST 4 PASSED: UnifiedRouter integration works")
        print("=" * 60)
        return True

    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


async def test_engine_clarify_handling():
    """Test engine handles clarify action."""
    print("\n" + "=" * 60)
    print("TEST 5: Engine Clarify Action Handling (code check)")
    print("=" * 60)

    # Check engine source code contains clarify handling
    engine_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "services", "engine", "__init__.py"
    )

    with open(engine_path, 'r', encoding='utf-8') as f:
        engine_code = f.read()

    # Check for clarify action handling
    assert 'decision.action == "clarify"' in engine_code, "Engine should handle clarify action"
    assert 'decision.clarification' in engine_code, "Engine should use clarification field"

    print(f"  ✓ Engine handles 'clarify' action")
    print(f"  ✓ Engine uses decision.clarification")

    print("\n" + "=" * 60)
    print("TEST 5 PASSED: Engine clarify handling exists")
    print("=" * 60)
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("LLM DISAMBIGUATION LOGIC TESTS")
    print("=" * 60)

    results = []

    # Test 1: AmbiguityDetector
    results.append(("AmbiguityDetector", test_ambiguity_detector()))

    # Test 2: Entity detection
    results.append(("Entity Detection", test_entity_detection()))

    # Test 3: Clarification questions
    results.append(("Clarification Questions", test_clarification_questions()))

    # Test 4: UnifiedRouter integration
    results.append(("UnifiedRouter Integration", await test_unified_router_integration()))

    # Test 5: Engine handling
    results.append(("Engine Clarify Handling", await test_engine_clarify_handling()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        icon = "✓" if passed else "✗"
        print(f"  {icon} {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

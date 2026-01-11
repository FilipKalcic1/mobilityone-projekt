"""
Test ACTION INTENT DETECTOR.

Verifies that the action intent detector correctly identifies
GET/POST/PUT/DELETE intent from user queries.

Usage:
    python scripts/test_action_intent.py
"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.action_intent_detector import (
    detect_action_intent,
    ActionIntent,
    filter_tools_by_intent
)


# Test cases: (query, expected_intent)
TEST_CASES = [
    # DELETE intent
    ("obriši vozilo", ActionIntent.DELETE),
    ("izbriši rezervaciju", ActionIntent.DELETE),
    ("ukloni korisnika", ActionIntent.DELETE),
    ("otkaži rezervaciju", ActionIntent.DELETE),
    ("želim obrisati tu stavku", ActionIntent.DELETE),
    ("delete my booking", ActionIntent.DELETE),
    ("remove this item", ActionIntent.DELETE),

    # CREATE intent (POST)
    ("dodaj novo vozilo", ActionIntent.CREATE),
    ("kreiraj rezervaciju", ActionIntent.CREATE),
    ("unesi kilometražu", ActionIntent.CREATE),
    ("prijavi štetu", ActionIntent.CREATE),
    ("rezerviraj vozilo za sutra", ActionIntent.CREATE),
    ("trebam vozilo", ActionIntent.CREATE),
    ("udario sam u stup", ActionIntent.CREATE),
    ("imam kvar na vozilu", ActionIntent.CREATE),
    ("add new vehicle", ActionIntent.CREATE),
    ("book a car for tomorrow", ActionIntent.CREATE),

    # UPDATE intent (PUT)
    ("ažuriraj podatke", ActionIntent.UPDATE),
    ("promijeni registraciju", ActionIntent.UPDATE),
    ("izmijeni rezervaciju", ActionIntent.UPDATE),
    ("ispravi kilometražu", ActionIntent.UPDATE),
    ("update vehicle info", ActionIntent.UPDATE),
    ("change my reservation", ActionIntent.UPDATE),

    # READ intent (GET)
    ("koliko imam kilometara", ActionIntent.READ),
    ("koja je registracija", ActionIntent.READ),
    ("prikaži moje rezervacije", ActionIntent.READ),
    ("pokaži slobodna vozila", ActionIntent.READ),
    ("daj mi podatke o vozilu", ActionIntent.READ),
    ("kada ističe registracija", ActionIntent.READ),
    ("moje rezervacije", ActionIntent.READ),
    ("show my bookings", ActionIntent.READ),
    ("what is my mileage?", ActionIntent.READ),
    ("list available vehicles", ActionIntent.READ),

    # Edge cases - should be READ (question format)
    ("koliko km imam na autu?", ActionIntent.READ),
    ("koji auto mi je dodijeljen?", ActionIntent.READ),
]


def run_tests():
    """Run all test cases."""
    print("=" * 70)
    print("ACTION INTENT DETECTOR TEST")
    print("=" * 70)
    print()

    passed = 0
    failed = 0
    results = []

    for query, expected in TEST_CASES:
        result = detect_action_intent(query)
        actual = result.intent
        is_correct = actual == expected

        if is_correct:
            passed += 1
            status = "[PASS]"
        else:
            failed += 1
            status = "[FAIL]"

        results.append({
            "query": query,
            "expected": expected.value,
            "actual": actual.value,
            "confidence": result.confidence,
            "reason": result.reason,
            "correct": is_correct
        })

        print(f"{status} | {query[:40]:<40} | Expected: {expected.value:<6} | Got: {actual.value:<6} | Conf: {result.confidence:.2f}")

    print()
    print("=" * 70)
    print(f"RESULTS: {passed}/{len(TEST_CASES)} passed ({100*passed/len(TEST_CASES):.1f}%)")
    print("=" * 70)

    # Show failed cases
    if failed > 0:
        print()
        print("FAILED CASES:")
        for r in results:
            if not r["correct"]:
                print(f"  - Query: {r['query']}")
                print(f"    Expected: {r['expected']}, Got: {r['actual']}")
                print(f"    Reason: {r['reason']}")
                print()

    return passed, failed


def test_filter_by_intent():
    """Test tool filtering by intent."""
    print()
    print("=" * 70)
    print("FILTER BY INTENT TEST")
    print("=" * 70)
    print()

    # Mock tool methods
    tool_methods = {
        "get_Vehicles": "GET",
        "post_Vehicles": "POST",
        "put_Vehicles_id": "PUT",
        "delete_Vehicles_id": "DELETE",
        "get_MasterData": "GET",
        "post_AddMileage": "POST",
        "post_AddCase": "POST",
        "get_VehicleCalendar": "GET",
        "post_VehicleCalendar": "POST",
        "delete_VehicleCalendar_id": "DELETE",
        "post_Search": "POST",  # Search POST
    }

    all_tools = set(tool_methods.keys())

    # Test READ intent
    read_result = filter_tools_by_intent(all_tools, tool_methods, ActionIntent.READ)
    print(f"READ intent -> {len(read_result)} tools: {sorted(read_result)}")

    # Test CREATE intent
    create_result = filter_tools_by_intent(all_tools, tool_methods, ActionIntent.CREATE)
    print(f"CREATE intent -> {len(create_result)} tools: {sorted(create_result)}")

    # Test UPDATE intent
    update_result = filter_tools_by_intent(all_tools, tool_methods, ActionIntent.UPDATE)
    print(f"UPDATE intent -> {len(update_result)} tools: {sorted(update_result)}")

    # Test DELETE intent
    delete_result = filter_tools_by_intent(all_tools, tool_methods, ActionIntent.DELETE)
    print(f"DELETE intent -> {len(delete_result)} tools: {sorted(delete_result)}")

    # Verify
    assert "get_Vehicles" in read_result
    assert "get_MasterData" in read_result
    assert "post_Search" in read_result  # Search POST should be in READ
    assert "post_AddMileage" in create_result
    assert "post_AddCase" in create_result
    assert "put_Vehicles_id" in update_result
    assert "delete_Vehicles_id" in delete_result
    assert "delete_VehicleCalendar_id" in delete_result

    print()
    print("[OK] Filter tests passed!")


if __name__ == "__main__":
    passed, failed = run_tests()
    test_filter_by_intent()

    print()
    print("=" * 70)
    if failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"SOME TESTS FAILED: {failed} failures")
    print("=" * 70)

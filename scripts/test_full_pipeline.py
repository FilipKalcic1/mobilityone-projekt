"""
Full Pipeline Accuracy Test - Tests QueryRouter + UnifiedSearch together.

This is the definitive test that validates the hybrid architecture:
1. QueryRouter (deterministic) handles known patterns with 100% accuracy
2. UnifiedSearch (FAISS) handles the long tail

Run with: python scripts/test_full_pipeline.py
"""

import asyncio
import json
import sys
import io
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.query_router import QueryRouter
from services.action_intent_detector import detect_action_intent
from services.unified_search import get_unified_search


@dataclass
class TestCase:
    query: str
    expected_tool: str
    expected_intent: str
    category: str


# Comprehensive test cases
TEST_CASES = [
    # ═══════════════════════════════════════════════════════════════
    # VEHICLE INFO / MILEAGE READ (GET) - QueryRouter should catch these
    # ═══════════════════════════════════════════════════════════════
    TestCase("koliko imam kilometara", "get_MasterData", "GET", "mileage_read"),
    TestCase("koja je moja kilometraža", "get_MasterData", "GET", "mileage_read"),
    TestCase("koliko km ima vozilo", "get_MasterData", "GET", "mileage_read"),
    TestCase("stanje kilometar sata", "get_MasterData", "GET", "mileage_read"),
    TestCase("podaci o vozilu", "get_MasterData", "GET", "vehicle_info"),
    TestCase("informacije o autu", "get_MasterData", "GET", "vehicle_info"),
    TestCase("koja je registracija", "get_MasterData", "GET", "vehicle_info"),
    TestCase("do kada vrijedi registracija", "get_MasterData", "GET", "vehicle_info"),
    TestCase("koja je lizing kuća", "get_MasterData", "GET", "vehicle_info"),

    # ═══════════════════════════════════════════════════════════════
    # MILEAGE INPUT (POST) - QueryRouter should catch these
    # ═══════════════════════════════════════════════════════════════
    TestCase("unesi kilometražu", "post_AddMileage", "POST", "mileage_input"),
    TestCase("upiši km", "post_AddMileage", "POST", "mileage_input"),
    TestCase("dodaj kilometre", "post_AddMileage", "POST", "mileage_input"),
    TestCase("prijavi kilometražu", "post_AddMileage", "POST", "mileage_input"),

    # ═══════════════════════════════════════════════════════════════
    # BOOKINGS READ (GET) - QueryRouter should catch these
    # ═══════════════════════════════════════════════════════════════
    TestCase("moje rezervacije", "get_VehicleCalendar", "GET", "bookings"),
    TestCase("pokaži moje bookinge", "get_VehicleCalendar", "GET", "bookings"),
    TestCase("kada imam auto", "get_VehicleCalendar", "GET", "bookings"),

    # ═══════════════════════════════════════════════════════════════
    # AVAILABILITY (GET) - QueryRouter should catch these
    # ═══════════════════════════════════════════════════════════════
    TestCase("slobodna vozila", "get_AvailableVehicles", "GET", "availability"),
    TestCase("koja vozila su dostupna", "get_AvailableVehicles", "GET", "availability"),
    TestCase("ima li slobodnih auta", "get_AvailableVehicles", "GET", "availability"),

    # ═══════════════════════════════════════════════════════════════
    # BOOKING CREATE (POST) - QueryRouter triggers booking flow
    # ═══════════════════════════════════════════════════════════════
    TestCase("rezerviraj vozilo", "get_AvailableVehicles", "POST", "booking_create"),
    TestCase("trebam auto", "get_AvailableVehicles", "POST", "booking_create"),
    TestCase("napravi rezervaciju", "get_AvailableVehicles", "POST", "booking_create"),

    # ═══════════════════════════════════════════════════════════════
    # BOOKING CANCEL (DELETE) - QueryRouter should catch these
    # ═══════════════════════════════════════════════════════════════
    TestCase("otkaži rezervaciju", "delete_VehicleCalendar_id", "DELETE", "booking_cancel"),
    TestCase("obriši booking", "delete_VehicleCalendar_id", "DELETE", "booking_cancel"),
    TestCase("ne trebam više auto", "delete_VehicleCalendar_id", "DELETE", "booking_cancel"),

    # ═══════════════════════════════════════════════════════════════
    # CASE/DAMAGE CREATE (POST) - QueryRouter should catch these
    # ═══════════════════════════════════════════════════════════════
    TestCase("prijavi štetu", "post_AddCase", "POST", "case_create"),
    TestCase("imam kvar na autu", "post_AddCase", "POST", "case_create"),
    TestCase("udario sam auto", "post_AddCase", "POST", "case_create"),
    TestCase("ogrebao sam vozilo", "post_AddCase", "POST", "case_create"),

    # ═══════════════════════════════════════════════════════════════
    # CASES LIST (GET) - UnifiedSearch handles
    # ═══════════════════════════════════════════════════════════════
    TestCase("prijavljene štete", "get_Cases", "GET", "cases_list"),
    TestCase("moje prijave", "get_Cases", "GET", "cases_list"),

    # ═══════════════════════════════════════════════════════════════
    # EXPENSES (GET) - UnifiedSearch handles
    # ═══════════════════════════════════════════════════════════════
    TestCase("troškovi", "get_Expenses", "GET", "expenses"),
    TestCase("pregled troškova", "get_Expenses", "GET", "expenses"),

    # ═══════════════════════════════════════════════════════════════
    # TRIPS (GET) - UnifiedSearch handles
    # ═══════════════════════════════════════════════════════════════
    TestCase("moja putovanja", "get_Trips", "GET", "trips"),
    TestCase("putni nalozi", "get_Trips", "GET", "trips"),

    # ═══════════════════════════════════════════════════════════════
    # PERSON INFO (GET) - QueryRouter should catch some, UnifiedSearch others
    # ═══════════════════════════════════════════════════════════════
    TestCase("moji podaci", "get_PersonData_personIdOrEmail", "GET", "person_info"),
    TestCase("tko sam ja", "get_PersonData_personIdOrEmail", "GET", "person_info"),

    # ═══════════════════════════════════════════════════════════════
    # ENTITY LISTS (GET) - UnifiedSearch handles
    # ═══════════════════════════════════════════════════════════════
    TestCase("sva vozila", "get_Vehicles", "GET", "vehicle_list"),
    TestCase("lista vozila", "get_Vehicles", "GET", "vehicle_list"),
    TestCase("sve kompanije", "get_Companies", "GET", "companies"),
    TestCase("svi timovi", "get_Teams", "GET", "teams"),
]


async def main():
    print("=" * 70)
    print("FULL PIPELINE ACCURACY TEST (QueryRouter + UnifiedSearch)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    # Initialize components
    query_router = QueryRouter()
    unified_search = get_unified_search()
    await unified_search.initialize()

    # Initialize FAISS
    config_dir = Path(__file__).parent.parent / "config"
    with open(config_dir / "tool_documentation.json", 'r', encoding='utf-8') as f:
        docs = json.load(f)

    from services.faiss_vector_store import get_faiss_store
    faiss_store = get_faiss_store()
    await faiss_store.initialize(docs)

    print(f"QueryRouter rules: {len(query_router.rules)}")
    print(f"FAISS vectors: {faiss_store.get_stats()['index_size']}")
    print()

    # Run tests
    results = {
        "total": 0,
        "qr_matched": 0,
        "qr_correct": 0,
        "faiss_correct": 0,
        "final_correct": 0,
        "categories": {}
    }

    print("Testing...")
    print("-" * 70)

    for test in TEST_CASES:
        results["total"] += 1

        if test.category not in results["categories"]:
            results["categories"][test.category] = {"total": 0, "qr": 0, "faiss": 0, "final": 0}
        results["categories"][test.category]["total"] += 1

        # Test QueryRouter first
        qr_result = query_router.route(test.query, {})
        qr_matched = qr_result.matched and qr_result.confidence >= 0.8
        qr_tool = qr_result.tool_name if qr_matched else None

        if qr_matched:
            results["qr_matched"] += 1
            if qr_tool and qr_tool.lower() == test.expected_tool.lower():
                results["qr_correct"] += 1
                results["categories"][test.category]["qr"] += 1

        # Test UnifiedSearch (FAISS)
        search_response = await unified_search.search(test.query, top_k=10)
        faiss_top1 = search_response.results[0].tool_id if search_response.results else None

        if faiss_top1 and faiss_top1.lower() == test.expected_tool.lower():
            results["faiss_correct"] += 1
            results["categories"][test.category]["faiss"] += 1

        # Final result (hybrid - use QR if matched, else FAISS)
        final_tool = qr_tool if qr_matched else faiss_top1
        final_correct = final_tool and final_tool.lower() == test.expected_tool.lower()

        if final_correct:
            results["final_correct"] += 1
            results["categories"][test.category]["final"] += 1

        # Print result
        source = "QR" if qr_matched else "FAISS"
        status = "OK" if final_correct else "FAIL"
        print(f"  {status:4} [{source:5}] \"{test.query[:30]}\" -> {final_tool or 'N/A'}")

    # Summary
    total = results["total"]
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"QueryRouter Coverage: {results['qr_matched']}/{total} ({100*results['qr_matched']/total:.1f}%)")
    print(f"QueryRouter Accuracy: {results['qr_correct']}/{results['qr_matched']} ({100*results['qr_correct']/max(1,results['qr_matched']):.1f}%)" if results['qr_matched'] > 0 else "QueryRouter Accuracy: N/A")
    print(f"FAISS-only Accuracy:  {results['faiss_correct']}/{total} ({100*results['faiss_correct']/total:.1f}%)")
    print(f"HYBRID Accuracy:      {results['final_correct']}/{total} ({100*results['final_correct']/total:.1f}%)")
    print()
    print("By Category:")
    for cat, cat_results in sorted(results["categories"].items()):
        t = cat_results["total"]
        qr = cat_results["qr"]
        faiss = cat_results["faiss"]
        final = cat_results["final"]
        print(f"  {cat:20s} → QR: {qr}/{t} | FAISS: {faiss}/{t} | HYBRID: {final}/{t}")

    print()
    print("=" * 70)
    print(f"FINAL HYBRID ACCURACY: {100*results['final_correct']/total:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

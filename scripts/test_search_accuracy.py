"""
Test Search Accuracy - Measures FAISS + Intent Gate accuracy.

This script tests the complete search pipeline:
1. ActionIntentDetector (GET/POST/PUT/DELETE)
2. FAISS semantic search
3. Boost pipeline (category, documentation, query type)

Run with: python scripts/test_search_accuracy.py
"""

import asyncio
import json
import sys
import io
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.action_intent_detector import detect_action_intent, ActionIntent
from services.faiss_vector_store import get_faiss_store, initialize_faiss_store
from services.unified_search import get_unified_search, initialize_unified_search


@dataclass
class TestCase:
    """A test case for search accuracy."""
    query: str
    expected_tool: str
    expected_intent: str  # GET, POST, PUT, DELETE
    category: str  # For grouping results


# Test cases covering various user intents
TEST_CASES = [
    # ═══════════════════════════════════════════════════════════════
    # VEHICLE INFO (GET)
    # ═══════════════════════════════════════════════════════════════
    TestCase("koliko imam kilometara", "get_MasterData", "GET", "vehicle_info"),
    TestCase("koja je moja kilometraža", "get_MasterData", "GET", "vehicle_info"),
    TestCase("koliko km ima moje vozilo", "get_MasterData", "GET", "vehicle_info"),
    TestCase("stanje kilometar sata", "get_MasterData", "GET", "vehicle_info"),
    TestCase("podaci o vozilu", "get_MasterData", "GET", "vehicle_info"),
    TestCase("informacije o autu", "get_MasterData", "GET", "vehicle_info"),
    TestCase("koja je registracija", "get_MasterData", "GET", "vehicle_info"),
    TestCase("do kada vrijedi registracija", "get_MasterData", "GET", "vehicle_info"),
    TestCase("koje su moje tablice", "get_MasterData", "GET", "vehicle_info"),
    TestCase("koja je lizing kuća", "get_MasterData", "GET", "vehicle_info"),

    # ═══════════════════════════════════════════════════════════════
    # MILEAGE INPUT (POST)
    # ═══════════════════════════════════════════════════════════════
    TestCase("unesi kilometražu", "post_AddMileage", "POST", "mileage_input"),
    TestCase("upiši km", "post_AddMileage", "POST", "mileage_input"),
    TestCase("dodaj kilometre", "post_AddMileage", "POST", "mileage_input"),
    TestCase("nova kilometraža", "post_AddMileage", "POST", "mileage_input"),
    TestCase("prijavi koliko sam prešao", "post_AddMileage", "POST", "mileage_input"),
    TestCase("unesi 15000 km", "post_AddMileage", "POST", "mileage_input"),
    TestCase("ažuriraj kilometražu", "post_AddMileage", "POST", "mileage_input"),

    # ═══════════════════════════════════════════════════════════════
    # BOOKINGS (GET)
    # ═══════════════════════════════════════════════════════════════
    TestCase("moje rezervacije", "get_VehicleCalendar", "GET", "bookings"),
    TestCase("pokaži moje bookinge", "get_VehicleCalendar", "GET", "bookings"),
    TestCase("kada imam auto", "get_VehicleCalendar", "GET", "bookings"),
    TestCase("kalendar vozila", "get_VehicleCalendar", "GET", "bookings"),

    # ═══════════════════════════════════════════════════════════════
    # AVAILABILITY (GET)
    # ═══════════════════════════════════════════════════════════════
    TestCase("slobodna vozila", "get_AvailableVehicles", "GET", "availability"),
    TestCase("koja vozila su dostupna", "get_AvailableVehicles", "GET", "availability"),
    TestCase("ima li slobodnih auta", "get_AvailableVehicles", "GET", "availability"),
    TestCase("raspoloživa vozila", "get_AvailableVehicles", "GET", "availability"),

    # ═══════════════════════════════════════════════════════════════
    # BOOKING CREATION (POST)
    # ═══════════════════════════════════════════════════════════════
    TestCase("rezerviraj vozilo", "post_VehicleCalendar", "POST", "booking_create"),
    TestCase("trebam auto", "post_VehicleCalendar", "POST", "booking_create"),
    TestCase("napravi rezervaciju", "post_VehicleCalendar", "POST", "booking_create"),
    TestCase("želim rezervirati vozilo", "post_VehicleCalendar", "POST", "booking_create"),
    TestCase("zauzmi auto za sutra", "post_VehicleCalendar", "POST", "booking_create"),

    # ═══════════════════════════════════════════════════════════════
    # BOOKING CANCELLATION (DELETE)
    # ═══════════════════════════════════════════════════════════════
    TestCase("otkaži rezervaciju", "delete_VehicleCalendar_id", "DELETE", "booking_cancel"),
    TestCase("obriši booking", "delete_VehicleCalendar_id", "DELETE", "booking_cancel"),
    TestCase("ne trebam više auto", "delete_VehicleCalendar_id", "DELETE", "booking_cancel"),

    # ═══════════════════════════════════════════════════════════════
    # CASE/DAMAGE (POST)
    # ═══════════════════════════════════════════════════════════════
    TestCase("prijavi štetu", "post_AddCase", "POST", "case_create"),
    TestCase("imam kvar na autu", "post_AddCase", "POST", "case_create"),
    TestCase("udario sam auto", "post_AddCase", "POST", "case_create"),
    TestCase("ogrebao sam vozilo", "post_AddCase", "POST", "case_create"),
    TestCase("nova prijava kvara", "post_AddCase", "POST", "case_create"),

    # ═══════════════════════════════════════════════════════════════
    # CASES LIST (GET)
    # ═══════════════════════════════════════════════════════════════
    TestCase("prijavljene štete", "get_Cases", "GET", "cases_list"),
    TestCase("moje prijave", "get_Cases", "GET", "cases_list"),
    TestCase("povijest kvarova", "get_Cases", "GET", "cases_list"),

    # ═══════════════════════════════════════════════════════════════
    # EXPENSES (GET)
    # ═══════════════════════════════════════════════════════════════
    TestCase("troškovi", "get_Expenses", "GET", "expenses"),
    TestCase("koliko sam potrošio", "get_Expenses", "GET", "expenses"),
    TestCase("pregled troškova", "get_Expenses", "GET", "expenses"),

    # ═══════════════════════════════════════════════════════════════
    # TRIPS (GET)
    # ═══════════════════════════════════════════════════════════════
    TestCase("moja putovanja", "get_Trips", "GET", "trips"),
    TestCase("povijest vožnji", "get_Trips", "GET", "trips"),
    TestCase("putni nalozi", "get_Trips", "GET", "trips"),

    # ═══════════════════════════════════════════════════════════════
    # PERSON INFO (GET)
    # ═══════════════════════════════════════════════════════════════
    TestCase("moji podaci", "get_PersonData_personIdOrEmail", "GET", "person_info"),
    TestCase("tko sam ja", "get_PersonData_personIdOrEmail", "GET", "person_info"),
    TestCase("moj profil", "get_PersonData_personIdOrEmail", "GET", "person_info"),

    # ═══════════════════════════════════════════════════════════════
    # VEHICLE LIST (GET)
    # ═══════════════════════════════════════════════════════════════
    TestCase("sva vozila", "get_Vehicles", "GET", "vehicle_list"),
    TestCase("lista vozila", "get_Vehicles", "GET", "vehicle_list"),
    TestCase("popis automobila", "get_Vehicles", "GET", "vehicle_list"),

    # ═══════════════════════════════════════════════════════════════
    # COMPANIES (GET)
    # ═══════════════════════════════════════════════════════════════
    TestCase("sve kompanije", "get_Companies", "GET", "companies"),
    TestCase("lista tvrtki", "get_Companies", "GET", "companies"),
    TestCase("dohvati firme", "get_Companies", "GET", "companies"),

    # ═══════════════════════════════════════════════════════════════
    # TEAMS (GET)
    # ═══════════════════════════════════════════════════════════════
    TestCase("svi timovi", "get_Teams", "GET", "teams"),
    TestCase("lista timova", "get_Teams", "GET", "teams"),
]


async def run_accuracy_test():
    """Run accuracy test on all test cases."""
    print("=" * 70)
    print("SEARCH ACCURACY TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    # Load tool documentation
    config_dir = Path(__file__).parent.parent / "config"
    doc_path = config_dir / "tool_documentation.json"

    with open(doc_path, 'r', encoding='utf-8') as f:
        tool_docs = json.load(f)

    print(f"Loaded {len(tool_docs)} tools from documentation")

    # Count tools with synonyms
    tools_with_synonyms = sum(1 for doc in tool_docs.values() if doc.get("synonyms_hr"))
    print(f"Tools with synonyms_hr: {tools_with_synonyms}")
    print()

    # Initialize FAISS store
    print("Initializing FAISS store...")
    faiss_store = get_faiss_store()
    await faiss_store.initialize(tool_docs)
    print(f"FAISS index: {faiss_store.get_stats()['index_size']} vectors")
    print()

    # Initialize UnifiedSearch
    print("Initializing UnifiedSearch...")
    unified_search = get_unified_search()
    await unified_search.initialize()
    print()

    # Run tests
    print("Running tests...")
    print("-" * 70)

    results = {
        "total": 0,
        "intent_correct": 0,
        "tool_in_top1": 0,
        "tool_in_top3": 0,
        "tool_in_top5": 0,
        "tool_in_top10": 0,
        "categories": {}
    }

    failed_cases = []

    for test in TEST_CASES:
        results["total"] += 1

        # Initialize category
        if test.category not in results["categories"]:
            results["categories"][test.category] = {
                "total": 0, "top1": 0, "top3": 0, "intent_ok": 0
            }
        results["categories"][test.category]["total"] += 1

        # Test Intent Detection
        intent_result = detect_action_intent(test.query)
        intent_correct = intent_result.intent.value == test.expected_intent

        if intent_correct:
            results["intent_correct"] += 1
            results["categories"][test.category]["intent_ok"] += 1

        # Test FAISS Search via UnifiedSearch
        search_response = await unified_search.search(test.query, top_k=10)

        # Check tool position in results
        tool_position = None
        for i, result in enumerate(search_response.results):
            if result.tool_id.lower() == test.expected_tool.lower():
                tool_position = i + 1
                break

        # Update metrics
        if tool_position == 1:
            results["tool_in_top1"] += 1
            results["tool_in_top3"] += 1
            results["tool_in_top5"] += 1
            results["tool_in_top10"] += 1
            results["categories"][test.category]["top1"] += 1
            results["categories"][test.category]["top3"] += 1
        elif tool_position and tool_position <= 3:
            results["tool_in_top3"] += 1
            results["tool_in_top5"] += 1
            results["tool_in_top10"] += 1
            results["categories"][test.category]["top3"] += 1
        elif tool_position and tool_position <= 5:
            results["tool_in_top5"] += 1
            results["tool_in_top10"] += 1
        elif tool_position and tool_position <= 10:
            results["tool_in_top10"] += 1
        else:
            # Failed - tool not in top 10
            failed_cases.append({
                "query": test.query,
                "expected": test.expected_tool,
                "got_top3": [r.tool_id for r in search_response.results[:3]],
                "intent_expected": test.expected_intent,
                "intent_got": intent_result.intent.value
            })

        # Print progress (using ASCII for Windows compatibility)
        status = "OK" if tool_position == 1 else ("~" if tool_position else "FAIL")
        intent_status = "OK" if intent_correct else "X"
        top1 = search_response.results[0].tool_id if search_response.results else "N/A"
        print(f"  {status:4} [{intent_status:2}] \"{test.query[:35]}\" -> pos={tool_position or 'N/A'} (top1: {top1[:30]})")

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    total = results["total"]
    print(f"\nOverall Accuracy:")
    print(f"  Intent Detection:  {results['intent_correct']}/{total} ({100*results['intent_correct']/total:.1f}%)")
    print(f"  Tool in Top-1:     {results['tool_in_top1']}/{total} ({100*results['tool_in_top1']/total:.1f}%)")
    print(f"  Tool in Top-3:     {results['tool_in_top3']}/{total} ({100*results['tool_in_top3']/total:.1f}%)")
    print(f"  Tool in Top-5:     {results['tool_in_top5']}/{total} ({100*results['tool_in_top5']/total:.1f}%)")
    print(f"  Tool in Top-10:    {results['tool_in_top10']}/{total} ({100*results['tool_in_top10']/total:.1f}%)")

    print(f"\nAccuracy by Category:")
    for cat, cat_results in sorted(results["categories"].items()):
        cat_total = cat_results["total"]
        top1_pct = 100 * cat_results["top1"] / cat_total if cat_total > 0 else 0
        top3_pct = 100 * cat_results["top3"] / cat_total if cat_total > 0 else 0
        intent_pct = 100 * cat_results["intent_ok"] / cat_total if cat_total > 0 else 0
        print(f"  {cat:20s} → Top1: {top1_pct:5.1f}% | Top3: {top3_pct:5.1f}% | Intent: {intent_pct:5.1f}%")

    if failed_cases:
        print(f"\nFailed Cases ({len(failed_cases)}):")
        for case in failed_cases[:10]:
            print(f"  Query: \"{case['query']}\"")
            print(f"    Expected: {case['expected']}")
            print(f"    Got Top3: {', '.join(case['got_top3'])}")
            print(f"    Intent: expected={case['intent_expected']}, got={case['intent_got']}")
            print()

    # Final score
    print("=" * 70)
    final_score = 100 * results['tool_in_top1'] / total
    print(f"FINAL SCORE (Top-1 Accuracy): {final_score:.1f}%")
    print("=" * 70)

    return results


if __name__ == "__main__":
    asyncio.run(run_accuracy_test())

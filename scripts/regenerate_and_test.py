"""
Regenerate FAISS embeddings with new synonyms and test accuracy.
"""

import asyncio
import json
import sys
import io
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.faiss_vector_store import get_faiss_store
from services.action_intent_detector import detect_action_intent
from services.unified_search import get_unified_search


# Test cases
TEST_CASES = [
    # VEHICLE INFO (GET)
    ("koliko imam kilometara", "get_MasterData", "GET"),
    ("koja je moja kilometraža", "get_MasterData", "GET"),
    ("podaci o vozilu", "get_MasterData", "GET"),
    ("informacije o autu", "get_MasterData", "GET"),
    ("koja je registracija", "get_MasterData", "GET"),
    ("do kada vrijedi registracija", "get_MasterData", "GET"),
    ("koja je lizing kuća", "get_MasterData", "GET"),

    # MILEAGE INPUT (POST)
    ("unesi kilometražu", "post_AddMileage", "POST"),
    ("upiši km", "post_AddMileage", "POST"),
    ("dodaj kilometre", "post_AddMileage", "POST"),
    ("prijavi koliko sam prešao", "post_AddMileage", "POST"),

    # BOOKINGS (GET)
    ("moje rezervacije", "get_VehicleCalendar", "GET"),
    ("pokaži moje bookinge", "get_VehicleCalendar", "GET"),
    ("kalendar vozila", "get_VehicleCalendar", "GET"),

    # AVAILABILITY (GET)
    ("slobodna vozila", "get_AvailableVehicles", "GET"),
    ("koja vozila su dostupna", "get_AvailableVehicles", "GET"),
    ("ima li slobodnih auta", "get_AvailableVehicles", "GET"),

    # BOOKING CREATION (POST)
    ("rezerviraj vozilo", "post_VehicleCalendar", "POST"),
    ("trebam auto", "post_VehicleCalendar", "POST"),
    ("napravi rezervaciju", "post_VehicleCalendar", "POST"),

    # CASE/DAMAGE (POST)
    ("prijavi štetu", "post_AddCase", "POST"),
    ("imam kvar na autu", "post_AddCase", "POST"),
    ("udario sam auto", "post_AddCase", "POST"),

    # CASES LIST (GET)
    ("prijavljene štete", "get_Cases", "GET"),
    ("moje prijave", "get_Cases", "GET"),

    # EXPENSES (GET)
    ("troškovi", "get_Expenses", "GET"),
    ("pregled troškova", "get_Expenses", "GET"),

    # TRIPS (GET)
    ("moja putovanja", "get_Trips", "GET"),
    ("putni nalozi", "get_Trips", "GET"),

    # PERSON INFO (GET)
    ("moji podaci", "get_PersonData_personIdOrEmail", "GET"),
    ("tko sam ja", "get_PersonData_personIdOrEmail", "GET"),

    # VEHICLE LIST (GET)
    ("sva vozila", "get_Vehicles", "GET"),
    ("lista vozila", "get_Vehicles", "GET"),

    # COMPANIES (GET)
    ("sve kompanije", "get_Companies", "GET"),
    ("lista tvrtki", "get_Companies", "GET"),

    # TEAMS (GET)
    ("svi timovi", "get_Teams", "GET"),
    ("lista timova", "get_Teams", "GET"),

    # BOOKING CANCELLATION (DELETE)
    ("otkaži rezervaciju", "delete_VehicleCalendar_id", "DELETE"),
    ("obriši booking", "delete_VehicleCalendar_id", "DELETE"),
]


async def main():
    print("=" * 70)
    print("FAISS EMBEDDING REGENERATION + ACCURACY TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load documentation
    config_dir = Path(__file__).parent.parent / "config"
    with open(config_dir / "tool_documentation.json", 'r', encoding='utf-8') as f:
        docs = json.load(f)

    # Count tools with synonyms
    tools_with_syn = sum(1 for d in docs.values() if d.get("synonyms_hr"))
    print(f"Total tools: {len(docs)}")
    print(f"Tools with synonyms_hr: {tools_with_syn}")

    # Sample synonyms
    for tool_id, doc in list(docs.items())[:5]:
        if doc.get("synonyms_hr"):
            print(f"  {tool_id}: {doc['synonyms_hr'][:3]}...")
            break
    print()

    # Check cache status
    cache_path = config_dir.parent / ".cache" / "tool_embeddings.json"
    print(f"Cache exists: {cache_path.exists()}")
    print()

    # Initialize FAISS store (will regenerate if cache is empty)
    print("Initializing FAISS store...")
    faiss_store = get_faiss_store()
    await faiss_store.initialize(docs)
    stats = faiss_store.get_stats()
    print(f"FAISS stats: {stats['total_tools']} tools, {stats['index_size']} vectors")
    print()

    # Initialize UnifiedSearch
    print("Initializing UnifiedSearch...")
    unified_search = get_unified_search()
    await unified_search.initialize()
    print()

    # Run tests
    print("Running accuracy tests...")
    print("-" * 70)

    results = {"total": 0, "top1": 0, "top3": 0, "top5": 0, "intent_ok": 0}

    for query, expected_tool, expected_intent in TEST_CASES:
        results["total"] += 1

        # Test intent
        intent = detect_action_intent(query)
        intent_ok = intent.intent.value == expected_intent
        if intent_ok:
            results["intent_ok"] += 1

        # Test search
        response = await unified_search.search(query, top_k=10)

        # Find position
        pos = None
        for i, r in enumerate(response.results):
            if r.tool_id.lower() == expected_tool.lower():
                pos = i + 1
                break

        # Update results
        if pos == 1:
            results["top1"] += 1
            results["top3"] += 1
            results["top5"] += 1
        elif pos and pos <= 3:
            results["top3"] += 1
            results["top5"] += 1
        elif pos and pos <= 5:
            results["top5"] += 1

        # Print result
        status = "OK" if pos == 1 else ("~" if pos else "FAIL")
        int_st = "OK" if intent_ok else "X"
        top1 = response.results[0].tool_id if response.results else "N/A"
        print(f"  {status:4} [{int_st:2}] \"{query[:30]}\" -> pos={pos or 'N/A'} (top1: {top1[:25]})")

    # Summary
    total = results["total"]
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Intent Detection:  {results['intent_ok']}/{total} ({100*results['intent_ok']/total:.1f}%)")
    print(f"Tool in Top-1:     {results['top1']}/{total} ({100*results['top1']/total:.1f}%)")
    print(f"Tool in Top-3:     {results['top3']}/{total} ({100*results['top3']/total:.1f}%)")
    print(f"Tool in Top-5:     {results['top5']}/{total} ({100*results['top5']/total:.1f}%)")
    print()
    print(f"FINAL SCORE: {100*results['top1']/total:.1f}% (Top-1 Accuracy)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

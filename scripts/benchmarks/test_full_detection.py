#!/usr/bin/env python3
"""
FULL Detection Test - Tests ALL 950 tools with proper registry initialization.

This test runs inside Docker with full stack:
- Redis connection
- Tool Registry with 950 tools
- Embeddings loaded
- FAISS vector search

Usage (inside Docker):
    python scripts/test_full_detection.py
"""

import asyncio
import sys
import os
import random
import time

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis.asyncio as aioredis


# 50 diverse test queries covering different domains and actions
# Format: (query, expected_tool_pattern, description)
TEST_QUERIES = [
    # === VEHICLES (GET) ===
    ("Pokaži mi vozila", "get_Vehicles", "List vehicles"),
    ("Detalji vozila ZG1234AB", "get_Vehicles", "Vehicle details"),
    ("Tipovi vozila", "get_VehicleTypes", "Vehicle types"),
    ("Koja je moja registracija?", "get_Master", "My registration"),
    ("Info o mom autu", "get_Master", "My vehicle info"),

    # === AVAILABILITY & BOOKING ===
    ("Slobodna vozila sutra", "get_AvailableVehicles", "Available vehicles"),
    ("Ima li slobodnih auta?", "get_AvailableVehicles", "Check availability"),
    ("Rezerviraj auto za sutra", "get_AvailableVehicles|post_VehicleCalendar|post_Booking", "Book vehicle"),
    ("Moje rezervacije", "get_VehicleCalendar", "My reservations"),
    ("Kalendar vozila", "get_VehicleCalendar", "Vehicle calendar"),

    # === MILEAGE ===
    ("Koliko imam kilometara?", "get_Mileage|get_Master|get_LatestMileage", "Check mileage"),
    ("Unesi 45000 km", "post_Mileage|post_AddMileage", "Enter mileage"),
    ("Kilometraža je 120000", "post_Mileage|post_AddMileage", "Report mileage"),
    ("Mjesečna kilometraža", "get_MonthlyMileage", "Monthly mileage"),

    # === CASES / DAMAGE ===
    ("Prijavi štetu", "post_Case|post_AddCase", "Report damage"),
    ("Udario sam u stup", "post_Case|post_AddCase", "Accident report"),
    ("Ogrebao sam branik", "post_Case|post_AddCase", "Scratch report"),
    ("Imam kvar na autu", "post_Case|post_AddCase", "Breakdown report"),
    ("Probušena guma", "post_Case|post_AddCase", "Flat tire"),
    ("Moji slučajevi", "get_Cases", "My cases"),
    ("Tipovi slučajeva", "get_CaseTypes", "Case types"),

    # === EXPENSES ===
    ("Troškovi", "get_Expenses", "Expenses list"),
    ("Koliko sam potrošio?", "get_Expenses", "My spending"),
    ("Troškovi za ovaj mjesec", "get_Expenses", "Monthly expenses"),
    ("Tipovi troškova", "get_ExpenseTypes", "Expense types"),
    ("Grupe troškova", "get_ExpenseGroups", "Expense groups"),

    # === TRIPS ===
    ("Moja putovanja", "get_Trips", "My trips"),
    ("Lista vožnji", "get_Trips", "Trip list"),
    ("Tipovi putovanja", "get_TripTypes", "Trip types"),
    ("Dodaj putovanje", "post_Trips", "Add trip"),

    # === PERSONS ===
    ("Moji podaci", "get_Person|get_Master", "My data"),
    ("Info o korisniku", "get_Person", "Person info"),
    ("Lista osoba", "get_Persons", "Persons list"),
    ("Tipovi osoba", "get_PersonTypes", "Person types"),

    # === EQUIPMENT ===
    ("Lista opreme", "get_Equipment", "Equipment list"),
    ("Tipovi opreme", "get_EquipmentTypes", "Equipment types"),
    ("Kalendar opreme", "get_EquipmentCalendar", "Equipment calendar"),

    # === PARTNERS ===
    ("Lista partnera", "get_Partners", "Partners list"),
    ("Dodaj partnera", "post_Partners", "Add partner"),
    ("Dobavljači", "get_Partners", "Suppliers"),

    # === TEAMS ===
    ("Timovi", "get_Teams", "Teams list"),
    ("Članovi tima", "get_TeamMembers", "Team members"),
    ("Moj tim", "get_TeamMembers", "My team"),

    # === COMPANIES & ORG ===
    ("Tvrtke", "get_Companies", "Companies"),
    ("Troškovna mjesta", "get_CostCenters", "Cost centers"),
    ("Organizacijske jedinice", "get_OrgUnits", "Org units"),

    # === DELETE OPERATIONS ===
    ("Otkaži rezervaciju", "delete_VehicleCalendar", "Cancel booking"),
    ("Obriši putovanje", "delete_Trips", "Delete trip"),
    ("Ukloni partnera", "delete_Partners", "Remove partner"),

    # === DOCUMENTS ===
    ("Tipovi dokumenata", "get_DocumentTypes", "Document types"),
    ("Priloži dokument", "post_documents", "Attach document"),

    # === STATISTICS ===
    ("Statistika flote", "get_Stats|get_Dashboard", "Fleet stats"),
    ("Dashboard", "get_Dashboard", "Dashboard"),
    ("Prosječni troškovi", "get_Stats|get_Expenses", "Average expenses"),
]


async def initialize_full_stack():
    """Initialize full stack with Redis and Tool Registry."""
    from config import get_settings
    settings = get_settings()

    print("Initializing full stack...")
    print(f"  Redis URL: {settings.REDIS_URL}")

    # Connect to Redis
    redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=50
    )
    await redis_client.ping()
    print("  Redis: connected")

    # Initialize Tool Registry
    from services.tool_registry import ToolRegistry
    registry = ToolRegistry(redis_client=redis_client)

    print("  Loading tools from swagger sources...")
    success = await registry.initialize(settings.swagger_sources)

    if not success:
        print("  ERROR: Tool Registry initialization failed!")
        return None, None

    print(f"  Tool Registry: {len(registry.tools)} tools loaded")
    print(f"  Embeddings: {'ready' if registry.embeddings else 'not loaded'}")

    # Initialize Unified Router with registry
    from services.unified_router import UnifiedRouter
    router = UnifiedRouter(registry=registry)
    await router.initialize()
    print("  Unified Router: initialized with full registry")

    return router, registry


def check_tool_match(actual_tool: str, expected_pattern: str) -> bool:
    """Check if actual tool matches expected pattern (supports | for alternatives)."""
    if not actual_tool:
        return False

    expected_options = expected_pattern.split("|")
    for option in expected_options:
        option = option.strip()
        # Partial match - actual tool should contain the pattern
        if option.lower() in actual_tool.lower():
            return True
    return False


async def run_full_test():
    """Run full detection test with 950 tools."""
    print("=" * 70)
    print("  FULL DETECTION TEST - 950 TOOLS")
    print("=" * 70)
    print()

    # Initialize
    start_time = time.time()
    router, registry = await initialize_full_stack()
    init_time = time.time() - start_time

    if not router or not registry:
        print("FATAL: Could not initialize full stack!")
        return None

    print(f"\nInitialization took: {init_time:.1f}s")
    print()

    # User context for testing
    user_context = {
        "person_id": "test-user-123",
        "tenant_id": "test-tenant",
        "vehicle": {
            "id": "vehicle-abc",
            "name": "Škoda Octavia",
            "plate": "ZG1234AB"
        }
    }

    # Run tests
    results = {
        "total": 0,
        "correct": 0,
        "close": 0,  # Right domain, wrong specific tool
        "wrong": 0,
        "no_match": 0,
        "details": []
    }

    print(f"Running {len(TEST_QUERIES)} tests...\n")
    print("-" * 70)

    for i, (query, expected_pattern, description) in enumerate(TEST_QUERIES, 1):
        results["total"] += 1

        try:
            start = time.time()
            decision = await router.route(query, user_context, None)
            latency = (time.time() - start) * 1000  # ms

            actual_tool = decision.tool
            action = decision.action

            # Check match
            is_match = check_tool_match(actual_tool, expected_pattern)

            # Determine status
            if is_match:
                results["correct"] += 1
                status = "OK"
                icon = "OK"
            elif not actual_tool:
                results["no_match"] += 1
                status = "NO_TOOL"
                icon = "?"
            else:
                results["wrong"] += 1
                status = "WRONG"
                icon = "X"

            # Print result
            q_display = query[:40] + "..." if len(query) > 40 else query
            print(f"[{i:2d}] [{icon:2s}] {q_display:<45} | {latency:4.0f}ms")
            print(f"      Expected: {expected_pattern}")
            print(f"      Actual:   {actual_tool or 'NO TOOL'} ({action})")
            if not is_match and decision.reasoning:
                print(f"      Reason:   {decision.reasoning[:60]}...")
            print()

            results["details"].append({
                "query": query,
                "description": description,
                "expected": expected_pattern,
                "actual": actual_tool,
                "action": action,
                "status": status,
                "latency_ms": latency,
                "reasoning": decision.reasoning
            })

        except Exception as e:
            results["wrong"] += 1
            print(f"[{i:2d}] [!!] ERROR: {query[:40]}...")
            print(f"      Error: {str(e)[:60]}")
            print()
            results["details"].append({
                "query": query,
                "expected": expected_pattern,
                "status": "ERROR",
                "error": str(e)
            })

    # Calculate statistics
    total = results["total"]
    correct = results["correct"]
    wrong = results["wrong"]
    no_match = results["no_match"]

    accuracy = (correct / total) * 100 if total > 0 else 0

    # Print summary
    print("\n" + "=" * 70)
    print("  REZULTATI PUNE EVALUACIJE")
    print("=" * 70)

    print(f"""
  Ukupno testova:      {total}
  Točni odgovori:      {correct} ({accuracy:.1f}%)
  Pogrešni odgovori:   {wrong} ({wrong/total*100:.1f}%)
  Bez odgovora:        {no_match} ({no_match/total*100:.1f}%)

  Registry:            {len(registry.tools)} alata
  Embeddings:          {'DA' if registry.embeddings else 'NE'}
""")

    # Evaluation
    print("=" * 70)
    print("  KONAČNA OCJENA")
    print("=" * 70)
    print()

    if accuracy >= 80:
        grade = "ODLIČAN"
        emoji = "A"
        assessment = "Sustav pouzdano prepoznaje upite."
    elif accuracy >= 60:
        grade = "DOBAR"
        emoji = "B"
        assessment = "Sustav uglavnom radi, ali ima prostora za poboljšanje."
    elif accuracy >= 40:
        grade = "SLAB"
        emoji = "C"
        assessment = "Sustav ima ozbiljnih problema s prepoznavanjem!"
    else:
        grade = "LOŠ"
        emoji = "D"
        assessment = "Sustav ne radi ispravno. HITNO potrebna revizija!"

    print(f"  OCJENA: {grade} ({emoji}) - {accuracy:.1f}%")
    print(f"  {assessment}")
    print()

    # Breakdown by category
    print("-" * 70)
    print("  BREAKDOWN PO KATEGORIJAMA:")
    print("-" * 70)

    categories = {}
    for d in results["details"]:
        desc = d.get("description", "unknown")
        cat = desc.split()[0] if desc else "unknown"  # First word
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if d["status"] == "OK":
            categories[cat]["correct"] += 1

    for cat, stats in sorted(categories.items()):
        pct = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {cat:20s} {bar} {stats['correct']:2d}/{stats['total']:2d} ({pct:.0f}%)")

    # Average latency
    latencies = [d.get("latency_ms", 0) for d in results["details"] if "latency_ms" in d]
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n  Prosječna latencija: {avg_latency:.0f}ms")

    print()
    print("=" * 70)

    return results


async def main():
    """Main entry point."""
    try:
        results = await run_full_test()

        if results:
            accuracy = (results["correct"] / results["total"]) * 100
            sys.exit(0 if accuracy >= 50 else 1)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

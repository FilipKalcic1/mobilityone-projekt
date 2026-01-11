#!/usr/bin/env python3
"""
Test 50 Random Diverse Tools - Detection Accuracy Test

Tests the system's ability to detect and route to correct tools
using random selections from the tool registry.

Usage:
    python scripts/test_random_50_tools.py
"""

import asyncio
import sys
import os
import random

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Query generators for different tool types
QUERY_TEMPLATES = {
    # GET queries (reading data)
    "get_Vehicles": ["Pokaži mi vozila", "Lista vozila", "Koja vozila imam?", "Daj mi popis vozila"],
    "get_Vehicles_id": ["Detalji vozila", "Info o vozilu", "Podaci o autu"],
    "get_VehicleCalendar": ["Kalendar vozila", "Rezervacije vozila", "Raspored vozila"],
    "get_VehicleTypes": ["Tipovi vozila", "Vrste auta", "Kategorije vozila"],
    "get_VehicleContracts": ["Ugovori za vozila", "Leasing ugovori", "Nabavni ugovori"],
    "get_Expenses": ["Troškovi", "Prikaži troškove", "Lista rashoda", "Koliko sam potrošio?"],
    "get_ExpenseTypes": ["Tipovi troškova", "Kategorije rashoda", "Vrste troškova"],
    "get_Trips": ["Putovanja", "Lista vožnji", "Prikaži trips", "Moje vožnje"],
    "get_TripTypes": ["Tipovi putovanja", "Vrste vožnji", "Kategorije tripova"],
    "get_Persons": ["Osobe", "Lista korisnika", "Zaposlenici", "Tko sve koristi sustav?"],
    "get_Persons_id": ["Detalji osobe", "Info o korisniku", "Podaci zaposlenika"],
    "get_PersonTypes": ["Tipovi osoba", "Vrste korisnika", "Kategorije zaposlenika"],
    "get_Companies": ["Tvrtke", "Lista kompanija", "Pokaži firme"],
    "get_CostCenters": ["Troškovna mjesta", "Cost centri", "Mjesta troška"],
    "get_Partners": ["Partneri", "Dobavljači", "Lista partnera"],
    "get_Cases": ["Slučajevi", "Prikaži slučajeve", "Lista prijava", "Ima li otvorenih slučajeva?"],
    "get_CaseTypes": ["Tipovi slučajeva", "Vrste prijava", "Kategorije slučajeva"],
    "get_Equipment": ["Oprema", "Lista opreme", "Alati"],
    "get_EquipmentTypes": ["Tipovi opreme", "Vrste alata", "Kategorije opreme"],
    "get_EquipmentCalendar": ["Kalendar opreme", "Raspored opreme", "Rezervacije alata"],
    "get_Teams": ["Timovi", "Lista timova", "Pokaži ekipe"],
    "get_TeamMembers": ["Članovi tima", "Tko je u timu?", "Lista članova"],
    "get_Tags": ["Oznake", "Tagovi", "Lista oznaka"],
    "get_Pools": ["Poolovi", "Skupovi vozila", "Grupe vozila"],
    "get_Roles": ["Uloge", "Role", "Prava pristupa"],
    "get_OrgUnits_id": ["Organizacijske jedinice", "Odjeli", "Sektori"],
    "get_AvailableVehicles": ["Slobodna vozila", "Dostupni auti", "Koja vozila su slobodna?", "Ima li slobodnih auta?"],
    "get_MileageReports": ["Izvještaji kilometraže", "Kilometri", "Koliko sam prešao?"],
    "get_LatestMileageReports": ["Zadnja kilometraža", "Trenutni km", "Koliko imam kilometara?"],
    "get_LatestVehicleCalendar": ["Zadnje rezervacije", "Aktualne rezervacije", "Aktivne rezervacije"],
    "get_MonthlyMileages": ["Mjesečna kilometraža", "Km po mjesecima", "Koliko mjesečno prelazim?"],
    "get_VehiclesMonthlyExpenses": ["Mjesečni troškovi vozila", "Rashodi po mjesecima"],
    "get_Stats_FleetAverageMonthlyMileage_year_month": ["Statistika flote", "Prosječna km flote"],
    "get_DocumentTypes": ["Tipovi dokumenata", "Vrste dokumenata"],
    "get_PeriodicActivities": ["Periodične aktivnosti", "Redovni servisi", "Održavanje"],
    "get_SchedulingModels": ["Modeli rasporeda", "Sheme raspoređivanja"],
    "get_Tenants": ["Tenanti", "Klijenti sustava"],
    "get_Master": ["Master podaci", "Glavni podaci", "Moji osnovni podaci"],

    # POST queries (creating/writing data)
    "post_Trips": ["Dodaj putovanje", "Kreiraj trip", "Nova vožnja", "Unesi put"],
    "post_Expenses_id_documents": ["Dodaj dokument troška", "Priloži račun"],
    "post_Cases": ["Prijavi slučaj", "Kreiraj slučaj", "Nova prijava", "Prijavi štetu", "Imam problem"],
    "post_MileageReports": ["Unesi kilometražu", "Dodaj km", "Upiši kilometre"],
    "post_Booking": ["Rezerviraj vozilo", "Book auto", "Trebam auto", "Želim rezervirati"],
    "post_VehicleCalendar": ["Dodaj u kalendar vozila", "Nova rezervacija", "Zakaži vozilo"],
    "post_Equipment_id_documents": ["Dodaj dokument opreme", "Priloži za opremu"],
    "post_Partners": ["Dodaj partnera", "Novi dobavljač", "Kreiraj partnera"],
    "post_Tags": ["Dodaj oznaku", "Novi tag", "Kreiraj tag"],
    "post_Teams": ["Dodaj tim", "Novi tim", "Kreiraj ekipu"],
    "post_PersonPeriodicActivities": ["Dodaj periodičnu aktivnost", "Nova aktivnost osobe"],
    "post_SendEmail": ["Pošalji email", "Send mail", "Šalji poruku"],

    # PUT queries (updating)
    "put_Vehicles_id": ["Ažuriraj vozilo", "Update auto", "Izmijeni vozilo"],
    "put_Cases_id": ["Ažuriraj slučaj", "Update prijavu", "Izmijeni slučaj"],
    "put_Expenses_id": ["Ažuriraj trošak", "Izmijeni rashod"],
    "put_Trips_id": ["Ažuriraj putovanje", "Izmijeni vožnju"],
    "put_Partners_id": ["Ažuriraj partnera", "Izmijeni dobavljača"],

    # DELETE queries
    "delete_Trips_id": ["Obriši putovanje", "Ukloni trip", "Izbriši vožnju"],
    "delete_Cases": ["Obriši slučaj", "Ukloni prijavu"],
    "delete_VehicleCalendar_id": ["Obriši rezervaciju", "Otkaži booking", "Ukloni iz kalendara"],
    "delete_Tags_id": ["Obriši oznaku", "Ukloni tag"],
    "delete_Partners_id": ["Obriši partnera", "Ukloni dobavljača"],

    # PATCH queries
    "patch_Vehicles_id": ["Djelomično ažuriraj vozilo", "Patch auto"],
    "patch_Cases_id": ["Djelomično ažuriraj slučaj"],
    "patch_Trips_id": ["Djelomično ažuriraj put"],
}

# Additional natural language queries mapped to expected tools
# NOTE: Using PRIMARY_TOOLS names from unified_router.py
NATURAL_QUERIES = [
    # Vehicles - MasterData returns vehicle info (registration, km, service)
    ("Koji auto vozim?", "get_MasterData"),
    ("Koja je moja registracija?", "get_MasterData"),
    ("Detalji mog vozila", "get_MasterData"),
    ("Pokaži mi moj auto", "get_MasterData"),

    # Availability
    ("Ima li slobodnih auta sutra?", "get_AvailableVehicles"),
    ("Koja vozila su dostupna ovaj tjedan?", "get_AvailableVehicles"),
    ("Trebam auto za ponedjeljak", "get_AvailableVehicles"),

    # Booking - starts with AvailableVehicles, then VehicleCalendar
    ("Rezerviraj mi Octaviu za sutra", "get_AvailableVehicles"),
    ("Želim bukirati auto od 10 do 14", "get_AvailableVehicles"),
    ("Book Passat za idući tjedan", "get_AvailableVehicles"),

    # Mileage - READ uses MasterData or LatestMileageReports
    ("Koliko kilometara imam?", "get_MasterData"),
    ("Unesi 45000 km", "post_AddMileage"),
    ("Kilometraža je 120000", "post_AddMileage"),
    ("Upiši kilometre 55000", "post_AddMileage"),

    # Cases/Damage - uses post_AddCase
    ("Udario sam u stup", "post_AddCase"),
    ("Ogrebao sam branik", "post_AddCase"),
    ("Imam kvar na autu", "post_AddCase"),
    ("Probušena guma", "post_AddCase"),
    ("Prijavi štetu", "post_AddCase"),

    # Expenses
    ("Koliko sam potrošio?", "get_Expenses"),
    ("Troškovi za ovaj mjesec", "get_Expenses"),
    ("Lista mojih rashoda", "get_Expenses"),

    # Trips
    ("Moje vožnje ovaj mjesec", "get_Trips"),
    ("Evidentiraj poslovni put", "get_Trips"),  # Note: post_Trips not in PRIMARY_TOOLS
    ("Dodaj novo putovanje", "get_Trips"),

    # Calendar - can be VehicleCalendar or AvailableVehicles
    ("Kad je vozilo zauzeto?", "get_VehicleCalendar"),
    ("Raspored rezervacija", "get_VehicleCalendar"),
    ("Tko koristi Passat sutra?", "get_VehicleCalendar"),

    # Cancellation
    ("Otkaži rezervaciju", "delete_VehicleCalendar_id"),
    ("Obrisi booking", "delete_VehicleCalendar_id"),
    ("Ne trebam više auto", "delete_VehicleCalendar_id"),

    # Stats - Dashboard or Expenses
    ("Statistika flote", "get_DashboardItems"),
    ("Koliko flota troši mjesečno?", "get_Expenses"),
    ("Prosječni troškovi", "get_Expenses"),

    # Equipment - NOT in PRIMARY_TOOLS (will likely fail)
    ("Lista opreme", "get_Equipment"),
    ("Dostupni alati", "get_Equipment"),

    # Teams - NOT in PRIMARY_TOOLS (will likely fail)
    ("Tko je u mom timu?", "get_PersonData_personIdOrEmail"),
    ("Lista timova", "get_Teams"),

    # Documents - NOT in PRIMARY_TOOLS (will likely fail)
    ("Tipovi dokumenata", "get_DocumentTypes"),
    ("Priloži račun za gorivo", "get_Expenses"),

    # Partners - NOT in PRIMARY_TOOLS (will likely fail)
    ("Lista dobavljača", "get_Partners"),
    ("Dodaj novog partnera", "post_Partners"),

    # Person - PersonData returns user info
    ("Moji podaci", "get_PersonData_personIdOrEmail"),
    ("Info o meni", "get_PersonData_personIdOrEmail"),
]


def generate_random_query(tool_name: str) -> str:
    """Generate a random query that should match the given tool."""
    if tool_name in QUERY_TEMPLATES:
        return random.choice(QUERY_TEMPLATES[tool_name])

    # Fallback: generate from tool name
    parts = tool_name.split("_")
    action = parts[0]  # get, post, put, delete, patch
    entity = "_".join(parts[1:]) if len(parts) > 1 else "data"

    if action == "get":
        return f"Pokaži mi {entity.replace('_', ' ')}"
    elif action == "post":
        return f"Dodaj {entity.replace('_', ' ')}"
    elif action == "put":
        return f"Ažuriraj {entity.replace('_', ' ')}"
    elif action == "delete":
        return f"Obriši {entity.replace('_', ' ')}"
    elif action == "patch":
        return f"Izmijeni {entity.replace('_', ' ')}"
    else:
        return f"Potraži {entity}"


async def test_random_tools(num_tests: int = 50):
    """Test random tool detection."""
    from services.unified_router import get_unified_router
    from services.tool_registry import ToolRegistry

    print("=" * 70)
    print("  TEST 50 RANDOM DIVERSE TOOLS - DETECTION ACCURACY")
    print("=" * 70)
    print()

    # Initialize router
    print("Initializing unified router...")
    router = await get_unified_router()

    # Check if registry is available
    if router._registry and hasattr(router._registry, 'tools'):
        print(f"Router initialized with {len(router._registry.tools)} tools")
    else:
        print("Router initialized (using PRIMARY_TOOLS fallback - no registry)")
    print()

    # Build test cases - mix of template-based and natural queries
    test_cases = []

    # Add all natural queries
    for query, expected_tool in NATURAL_QUERIES:
        test_cases.append((query, expected_tool))

    # Add template-based queries to reach 50
    template_tools = list(QUERY_TEMPLATES.keys())
    random.shuffle(template_tools)

    remaining = num_tests - len(test_cases)
    for tool in template_tools[:remaining]:
        query = generate_random_query(tool)
        test_cases.append((query, tool))

    # Shuffle all test cases
    random.shuffle(test_cases)
    test_cases = test_cases[:num_tests]

    # Stats
    results = {
        "total": 0,
        "correct": 0,
        "close_match": 0,  # Same domain but different tool
        "wrong": 0,
        "no_match": 0,
        "details": []
    }

    user_context = {
        "person_id": "test-user-123",
        "tenant_id": "test-tenant",
        "vehicle": {
            "id": "vehicle-abc",
            "name": "Škoda Octavia",
            "plate": "ZG1234AB"
        }
    }

    print(f"Running {len(test_cases)} test cases...\n")
    print("-" * 70)

    for i, (query, expected_tool) in enumerate(test_cases, 1):
        results["total"] += 1

        try:
            decision = await router.route(query, user_context, None)

            actual_tool = decision.tool
            action = decision.action

            # Check if exact match
            is_exact = actual_tool == expected_tool

            # Check if close match (same entity type)
            is_close = False
            if actual_tool and expected_tool:
                expected_entity = expected_tool.split("_")[1] if "_" in expected_tool else ""
                actual_entity = actual_tool.split("_")[1] if "_" in actual_tool else ""
                is_close = expected_entity.lower() == actual_entity.lower()

            # Determine status
            if is_exact:
                results["correct"] += 1
                status = "CORRECT"
                icon = "OK"
            elif is_close:
                results["close_match"] += 1
                status = "CLOSE"
                icon = "~"
            elif not actual_tool:
                results["no_match"] += 1
                status = "NO_MATCH"
                icon = "?"
            else:
                results["wrong"] += 1
                status = "WRONG"
                icon = "X"

            # Print result
            print(f"[{i:2d}] [{icon}] \"{query[:45]}...\"")
            print(f"     Expected: {expected_tool}")
            print(f"     Actual:   {actual_tool or 'NO TOOL'} (action={action})")
            if not is_exact and decision.reasoning:
                print(f"     Reason:   {decision.reasoning[:60]}...")
            print()

            results["details"].append({
                "query": query,
                "expected": expected_tool,
                "actual": actual_tool,
                "action": action,
                "status": status,
                "reasoning": decision.reasoning
            })

        except Exception as e:
            results["wrong"] += 1
            print(f"[{i:2d}] [!] ERROR: {query[:40]}...")
            print(f"     Error: {str(e)[:60]}")
            print()
            results["details"].append({
                "query": query,
                "expected": expected_tool,
                "actual": None,
                "status": "ERROR",
                "error": str(e)
            })

    # Print summary
    print("\n" + "=" * 70)
    print("  REZULTATI TESTA")
    print("=" * 70)

    total = results["total"]
    correct = results["correct"]
    close = results["close_match"]
    wrong = results["wrong"]
    no_match = results["no_match"]

    accuracy = (correct / total) * 100 if total > 0 else 0
    close_accuracy = ((correct + close) / total) * 100 if total > 0 else 0

    print(f"\n  Ukupno testova:     {total}")
    print(f"  Točni odgovori:     {correct} ({accuracy:.1f}%)")
    print(f"  Blizu (isti tip):   {close} ({close/total*100:.1f}%)")
    print(f"  Pogrešni:           {wrong} ({wrong/total*100:.1f}%)")
    print(f"  Bez odgovora:       {no_match} ({no_match/total*100:.1f}%)")
    print()
    print(f"  PRECIZNA TOČNOST:   {accuracy:.1f}%")
    print(f"  PRIBLIŽNA TOČNOST:  {close_accuracy:.1f}%")
    print()

    # Evaluation
    print("=" * 70)
    print("  EVALUACIJA SUSTAVA")
    print("=" * 70)
    print()

    if accuracy >= 80:
        print("  OCJENA: ODLIČAN (>80%)")
        print("  Sustav pouzdano prepoznaje namjeru korisnika.")
    elif accuracy >= 60:
        print("  OCJENA: DOBAR (60-80%)")
        print("  Sustav uglavnom radi, ali ima prostora za poboljšanje.")
    elif accuracy >= 40:
        print("  OCJENA: SLAB (40-60%)")
        print("  Sustav ima problema s prepoznavanjem. Treba poboljšanje!")
    else:
        print("  OCJENA: LOŠ (<40%)")
        print("  Sustav ne prepoznaje ispravno. HITNO potrebna revizija!")

    print()

    # Detailed breakdown by action type
    print("-" * 70)
    print("  DETALJNI BREAKDOWN PO AKCIJAMA:")
    print("-" * 70)

    action_stats = {}
    for d in results["details"]:
        action = d.get("action", "unknown")
        if action not in action_stats:
            action_stats[action] = {"total": 0, "correct": 0}
        action_stats[action]["total"] += 1
        if d["status"] in ["CORRECT", "CLOSE"]:
            action_stats[action]["correct"] += 1

    for action, stats in sorted(action_stats.items()):
        pct = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {action:20s}: {stats['correct']:3d}/{stats['total']:3d} ({pct:.0f}%)")

    print()
    print("=" * 70)

    return results


async def main():
    """Main entry point."""
    try:
        results = await test_random_tools(50)

        # Return exit code based on accuracy
        accuracy = (results["correct"] / results["total"]) * 100
        sys.exit(0 if accuracy >= 50 else 1)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

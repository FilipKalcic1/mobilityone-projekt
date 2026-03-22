"""
Fix generic/duplicate example_queries_hr in tool_documentation.json.

Problem: 40 tools have generic queries like "Daj mi prosječnu vrijednost za određeno polje."
that don't mention the entity, making them indistinguishable from each other.

Fix: Replace first example query with entity-specific Croatian query.
"""

import json
import sys
from pathlib import Path
from copy import deepcopy

PROJECT_DIR = Path(__file__).parent.parent
DOC_PATH = PROJECT_DIR / "config" / "tool_documentation.json"

# Entity -> Croatian name (genitive/accusative forms for natural queries)
ENTITY_HR = {
    "AvailabilityProjection": "projekcije dostupnosti",
    "CaseTypes": "tipova slučajeva",
    "Companies": "kompanija",
    "Equipment": "opreme",
    "EquipmentCalendar": "rasporeda opreme",
    "ExpenseGroups": "grupa troškova",
    "LatestEquipmentCalendar": "najnovijeg rasporeda opreme",
    "LatestPeriodicActivities": "najnovijih periodičnih aktivnosti",
    "LatestVehicleCalendar": "najnovijih rezervacija vozila",
    "LatestVehicleContracts": "najnovijih ugovora vozila",
    "Lookup": "šifrarnika",
    "Metadata": "metapodataka",
    "OrgUnits": "organizacijskih jedinica",
    "PeriodicActivities": "periodičnih aktivnosti",
    "PeriodicActivitiesSchedules": "rasporeda periodičnih aktivnosti",
    "PersonActivityTypes": "tipova aktivnosti osoba",
    "PersonPeriodicActivities": "periodičnih aktivnosti osoba",
    "Persons": "osoba",
    "SchedulingModels": "modela rasporeda",
    "Tags": "oznaka",
    "Tenants": "zakupaca",
    "TenantPermissions": "prava pristupa",
    "Trips": "putovanja",
    "Vehicles": "vozila",
    "VehiclesHistoricalEntries": "povijesnih zapisa vozila",
}

# Entity -> Croatian nominative (for "Agregiraj X" pattern)
ENTITY_HR_NOM = {
    "AvailabilityProjection": "projekcije dostupnosti",
    "CaseTypes": "tipove slučajeva",
    "Companies": "kompanije",
    "Equipment": "opremu",
    "EquipmentCalendar": "raspored opreme",
    "ExpenseGroups": "grupe troškova",
    "LatestEquipmentCalendar": "najnoviji raspored opreme",
    "LatestPeriodicActivities": "najnovije periodične aktivnosti",
    "LatestVehicleCalendar": "najnovije rezervacije vozila",
    "LatestVehicleContracts": "najnovije ugovore vozila",
    "Lookup": "šifrarnike",
    "Metadata": "metapodatke",
    "OrgUnits": "organizacijske jedinice",
    "PeriodicActivities": "periodične aktivnosti",
    "PeriodicActivitiesSchedules": "rasporede periodičnih aktivnosti",
    "PersonActivityTypes": "tipove aktivnosti osoba",
    "PersonPeriodicActivities": "periodične aktivnosti osoba",
    "Persons": "osobe",
    "SchedulingModels": "modele rasporeda",
    "Tags": "oznake",
    "Tenants": "zakupce",
    "TenantPermissions": "prava pristupa",
    "Trips": "putovanja",
    "Vehicles": "vozila",
    "VehiclesHistoricalEntries": "povijesne zapise vozila",
}

# Specific fixes for each problematic tool
FIXES = {
    # === _Agg tools: make query entity-specific ===
    "get_AvailabilityProjection_Agg": [
        "Agregiraj projekcije dostupnosti vozila.",
        "Izračunaj prosječnu dostupnost vozila.",
    ],
    "get_CaseTypes_Agg": [
        "Agregiraj tipove slučajeva.",
        "Koliko ima aktivnih tipova slučajeva?",
    ],
    "get_Companies_Agg": [
        "Agregiraj podatke o kompanijama.",
        "Koliko ima registriranih kompanija?",
    ],
    "get_Equipment_Agg": [
        "Agregiraj podatke o opremi.",
        "Koliko ukupno ima opreme?",
    ],
    "get_EquipmentCalendar_Agg": [
        "Agregiraj raspored opreme.",
        "Statistika korištenja opreme po mjesecu.",
    ],
    "get_LatestEquipmentCalendar_Agg": [
        "Agregiraj najnoviji raspored opreme.",
        "Prosječno korištenje opreme u zadnjem razdoblju.",
    ],
    "get_LatestPeriodicActivities_Agg": [
        "Agregiraj najnovije periodične aktivnosti.",
        "Koliko periodičnih aktivnosti je obavljeno?",
    ],
    "get_LatestVehicleCalendar_Agg": [
        "Agregiraj najnovije rezervacije vozila.",
        "Prosječan broj rezervacija vozila po mjesecu.",
    ],
    "get_LatestVehicleContracts_Agg": [
        "Agregiraj najnovije ugovore vozila.",
        "Prosječna cijena ugovora za vozila.",
    ],
    "get_Metadata_Agg": [
        "Agregiraj metapodatke entiteta.",
        "Statistika metapodataka po tipu.",
    ],
    "get_OrgUnits_Agg": [
        "Agregiraj organizacijske jedinice.",
        "Koliko ima organizacijskih jedinica?",
    ],
    "get_PersonPeriodicActivities_Agg": [
        "Agregiraj periodične aktivnosti osoba.",
        "Prosječan broj aktivnosti po osobi.",
    ],
    "get_Persons_Agg": [
        "Agregiraj podatke o osobama.",
        "Koliko je registriranih korisnika?",
    ],
    "get_SchedulingModels_Agg": [
        "Agregiraj modele rasporeda.",
        "Statistika modela rasporeda.",
    ],
    "get_Tags_Agg": [
        "Agregiraj oznake.",
        "Koliko ima različitih oznaka?",
    ],
    "get_Tenants_Agg": [
        "Agregiraj podatke o zakupcima.",
        "Koliko ima aktivnih zakupaca?",
    ],
    "get_TenantPermissions_Agg": [
        "Agregiraj prava pristupa zakupaca.",
        "Statistika dozvola po zakupcu.",
    ],
    "get_Trips_Agg": [
        "Agregiraj putovanja.",
        "Koliko je ukupno putovanja zabilježeno?",
    ],

    # === _ProjectTo tools: differentiate from base entity ===
    "get_AvailabilityProjection_ProjectTo": [
        "Prikaži specifična polja projekcije dostupnosti.",
        "Filtriraj polja projekcije dostupnosti.",
    ],
    "get_CaseTypes_ProjectTo": [
        "Prikaži specifična polja tipova slučajeva.",
        "Filtriraj kolone tipova slučajeva.",
    ],
    "get_Equipment_ProjectTo": [
        "Prikaži specifična polja opreme.",
        "Dohvati samo odabrane kolone za opremu.",
    ],
    "get_ExpenseGroups_ProjectTo": [
        "Prikaži specifična polja grupa troškova.",
        "Dohvati samo odabrane kolone grupa troškova.",
    ],

    # === _metadata tools: make entity clear ===
    "get_CaseTypes_id_metadata": [
        "Daj mi metapodatke za tip slučaja s ID-om 5.",
        "Metapodaci tipa slučaja.",
    ],
    "get_Equipment_id_metadata": [
        "Daj mi metapodatke za opremu s ID-om 42.",
        "Metapodaci opreme.",
    ],
    "get_ExpenseGroups_id_metadata": [
        "Daj mi metapodatke za grupu troškova s ID-om 10.",
        "Metapodaci grupe troškova.",
    ],
    "get_LatestVehicleContracts_id_metadata": [
        "Daj mi metapodatke za najnoviji ugovor vozila s ID-om 123.",
        "Metapodaci najnovijeg ugovora vozila.",
    ],
    "get_Vehicles_id_metadata": [
        "Daj mi metapodatke za vozilo s ID-om 123.",
        "Metapodaci specifičnog vozila.",
    ],
    "get_VehiclesHistoricalEntries_id_metadata": [
        "Daj mi metapodatke za povijesni zapis vozila s ID-om 123.",
        "Metapodaci povijesnog zapisa vozila.",
    ],

    # === _GroupBy tools: entity-specific grouping ===
    "get_Companies_GroupBy": [
        "Grupiraj kompanije po polju.",
        "Kompanije grupirane po gradu.",
    ],
    "get_PeriodicActivitiesSchedules_GroupBy": [
        "Grupiraj rasporede periodičnih aktivnosti prema tipu.",
        "Rasporedi aktivnosti grupirani po vozilu.",
    ],
    "get_PeriodicActivities_GroupBy": [
        "Grupiraj periodične aktivnosti prema tipu.",
        "Periodične aktivnosti grupirane po mjesecu.",
    ],

    # === Lookup tools: differentiate from base get_ ===
    "get_Lookup_CompanyId": [
        "Dohvati šifrarnik kompanija za odabir.",
        "Dropdown lista kompanija.",
    ],
    "get_Lookup_CostCenterId": [
        "Dohvati šifrarnik centara troškova za odabir.",
        "Dropdown lista centara troškova.",
    ],
    "get_Lookup_OrgUnitId": [
        "Dohvati šifrarnik organizacijskih jedinica za odabir.",
        "Dropdown lista organizacijskih jedinica.",
    ],

    # === delete tools with duplicate queries ===
    "delete_PeriodicActivities": [
        "Obriši više periodičnih aktivnosti odjednom.",
        "Masovno brisanje periodičnih aktivnosti.",
    ],
    "delete_PeriodicActivities_id": [
        "Obriši periodičnu aktivnost s ID-om 15.",
        "Ukloni specifičnu periodičnu aktivnost.",
    ],
    "delete_PersonActivityTypes_DeleteByCriteria": [
        "Obriši tipove aktivnosti osoba prema kriterijima.",
        "Masovno brisanje tipova aktivnosti po filteru.",
    ],
}


def main():
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    tools = data.get("tools", data) if isinstance(data, dict) else data
    is_dict = isinstance(tools, dict)

    if is_dict:
        tool_list = list(tools.values())
        tool_ids_map = tools
    else:
        tool_list = tools
        tool_ids_map = {t.get("operation_id", ""): t for t in tool_list}

    fixed = 0
    for tool_id, new_queries in FIXES.items():
        tool = tool_ids_map.get(tool_id)
        if tool is None:
            print(f"  WARNING: {tool_id} not found in documentation!")
            continue

        old_q = tool.get("example_queries_hr", [])
        old_first = old_q[0] if old_q else "(empty)"

        # Replace queries: new first + new second, keep rest
        remaining = old_q[2:] if len(old_q) > 2 else []
        tool["example_queries_hr"] = new_queries + remaining

        print(f"  FIXED: {tool_id}")
        print(f"    OLD: {old_first}")
        print(f"    NEW: {new_queries[0]}")
        fixed += 1

    # Write back
    with open(DOC_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nFixed {fixed} tools in {DOC_PATH}")
    print("IMPORTANT: FAISS embeddings must be regenerated!")


if __name__ == "__main__":
    main()

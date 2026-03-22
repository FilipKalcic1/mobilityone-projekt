#!/usr/bin/env python3
"""
Generate synonyms_hr, fix duplicate purposes, and fix API-style examples
for all 950 tools in tool_documentation.json.

This is a one-time enrichment script that improves FAISS search quality
by populating the synonyms_hr field (5x weight in embeddings) and
differentiating tools with identical purposes.

Usage:
    python scripts/generate_synonyms.py
    # Then delete .cache/tool_embeddings.json to force re-embedding
"""

import json
import re
import os
import sys

# ── Entity Translation Map ──
# Maps API entity names (from operation_id) to Croatian translations.
# Format: {EntityName: (nominative, genitive, [colloquial synonyms])}
ENTITY_MAP = {
    # Core fleet
    "Vehicles": ("vozilo", "vozila", ["auto", "automobil", "kola"]),
    "AvailableVehicles": ("dostupno vozilo", "dostupnih vozila", ["slobodan auto", "slobodno vozilo"]),
    "VehicleBoard": ("board vozila", "boarda vozila", ["defleet", "deregistracija", "stanje vozila"]),
    "VehicleCalendar": ("kalendar vozila", "kalendara vozila", ["rezervacija", "raspored vozila", "booking"]),
    "VehicleCalendarOn": ("kalendar vozila za datum", "kalendara vozila za datum", ["rezervacije na dan"]),
    "VehicleContracts": ("ugovor za vozilo", "ugovora za vozilo", ["leasing", "najam"]),
    "VehicleTypes": ("tip vozila", "tipova vozila", ["vrsta auta", "kategorija vozila"]),
    "VehicleAssignments": ("dodjela vozila", "dodjela vozila", ["dodijeljeno vozilo", "assignment"]),
    "VehiclesAssignmentsOverview": ("pregled dodjela", "pregleda dodjela", ["tko vozi što"]),
    "VehiclesHistoricalEntries": ("povijesni unosi", "povijesnih unosa", ["povijest vozila", "stari zapisi"]),
    "VehiclesMonthlyExpenses": ("mjesečni troškovi vozila", "mjesečnih troškova", ["troškovi po mjesecu"]),
    "VehicleInputHelper": ("pomoćnik za unos vozila", "pomoćnika za unos", ["input helper"]),
    # Mileage
    "MileageReports": ("izvještaj o kilometraži", "izvještaja o kilometraži", ["km izvještaj", "prijeđeni km"]),
    "MonthlyMileages": ("mjesečna kilometraža", "mjesečne kilometraže", ["km po mjesecu"]),
    "MonthlyMileagesAssigned": ("mjesečna kilometraža dodijeljenih", "mjesečne kilometraže", ["km dodijeljenih vozila"]),
    "LatestMileageReports": ("zadnji izvještaj km", "zadnjeg izvještaja km", ["posljednja kilometraža"]),
    "AddMileage": ("unos kilometraže", "unosa kilometraže", ["upiši km", "dodaj kilometražu"]),
    "AverageFuelExpensesAndMileages": ("prosječna potrošnja", "prosječne potrošnje", ["prosjek goriva i km"]),
    "MonthlyFuelExpensesAndMileages": ("mjesečna potrošnja goriva", "mjesečne potrošnje", ["gorivo po mjesecu"]),
    # People
    "Persons": ("osoba", "osoba", ["korisnik", "zaposlenik", "vozač"]),
    "PersonData": ("podaci o osobi", "podataka o osobi", ["moji podaci", "profil", "info o korisniku"]),
    "PersonTypes": ("tip osobe", "tipova osobe", ["vrsta korisnika"]),
    "PersonOrgUnits": ("org. jedinica osobe", "org. jedinica", ["odjel korisnika"]),
    "PersonPeriodicActivities": ("periodične aktivnosti osobe", "periodičnih aktivnosti", ["redovne aktivnosti"]),
    "PersonActivityTypes": ("tipovi aktivnosti osobe", "tipova aktivnosti", ["vrste aktivnosti"]),
    "LatestPersonPeriodicActivities": ("zadnje aktivnosti osobe", "zadnjih aktivnosti", ["posljednje aktivnosti"]),
    # Cases & Support
    "Cases": ("slučaj", "slučajeva", ["prijava", "kvar", "problem", "šteta", "ticket"]),
    "CaseTypes": ("tip slučaja", "tipova slučaja", ["vrsta prijave"]),
    "AddCase": ("prijava slučaja", "prijave slučaja", ["prijavi problem", "novi kvar", "nova šteta"]),
    # Expenses & Financial
    "Expenses": ("trošak", "troškova", ["rashod", "izdatak", "cijena"]),
    "ExpenseTypes": ("tip troška", "tipova troška", ["vrsta troška", "kategorija rashoda"]),
    "ExpenseGroups": ("grupa troškova", "grupa troškova", ["kategorija troškova"]),
    "ExpensesInputHelper": ("pomoćnik za troškove", "pomoćnika za troškove", ["input helper troškovi"]),
    # Trips
    "Trips": ("putovanje", "putovanja", ["vožnja", "trip", "ruta"]),
    "TripTypes": ("tip putovanja", "tipova putovanja", ["vrsta vožnje"]),
    # Equipment
    "Equipment": ("oprema", "opreme", ["dodatak", "uređaj", "accessory"]),
    "EquipmentTypes": ("tip opreme", "tipova opreme", ["vrsta opreme"]),
    "EquipmentCalendar": ("kalendar opreme", "kalendara opreme", ["raspored opreme"]),
    "EquipmentCalendarOn": ("kalendar opreme za datum", "kalendara opreme", ["oprema na dan"]),
    "EquipmentCalendarOnPersonVehicle": ("oprema osoba-vozilo", "opreme", ["oprema po vozilu i osobi"]),
    "LatestEquipmentCalendar": ("zadnji kalendar opreme", "zadnjeg kalendara", ["posljednja oprema"]),
    # Organization
    "Companies": ("tvrtka", "tvrtki", ["firma", "poduzeće", "kompanija"]),
    "OrgUnits": ("organizacijska jedinica", "org. jedinica", ["odjel", "sektor"]),
    "Teams": ("tim", "timova", ["ekipa", "grupa"]),
    "TeamMembers": ("član tima", "članova tima", ["članovi ekipe"]),
    "Tenants": ("tenant", "tenanata", ["zakupac", "organizacija"]),
    "TenantPermissions": ("dozvole tenanta", "dozvola tenanta", ["prava pristupa"]),
    "Partners": ("partner", "partnera", ["dobavljač", "suradnik"]),
    "Roles": ("uloga", "uloga", ["rola", "pozicija"]),
    # Periodic activities
    "PeriodicActivities": ("periodična aktivnost", "periodičnih aktivnosti", ["redovna aktivnost", "raspored"]),
    "PeriodicActivitiesSchedules": ("raspored aktivnosti", "rasporeda aktivnosti", ["plan aktivnosti"]),
    "PeriodicActivityTypes": ("tip periodične aktivnosti", "tipova", ["vrsta aktivnosti"]),
    "LatestPeriodicActivities": ("zadnje aktivnosti", "zadnjih aktivnosti", ["posljednje aktivnosti"]),
    # Documents & Settings
    "DocumentTypes": ("tip dokumenta", "tipova dokumenata", ["vrsta dokumenta"]),
    "DashboardItems": ("stavka dashboarda", "stavki dashboarda", ["widget", "nadzorna ploča"]),
    "Settings": ("postavke", "postavki", ["konfiguracija", "opcije"]),
    "Tags": ("oznaka", "oznaka", ["tag", "label"]),
    # Lookups
    "Lookup": ("lookup", "lookupa", ["šifrarnik", "referentni podaci"]),
    # Contracts
    "LatestVehicleContracts": ("zadnji ugovor vozila", "zadnjeg ugovora", ["posljednji ugovor"]),
    "LatestVehicleCalendar": ("zadnji kalendar vozila", "zadnjeg kalendara", ["posljednja rezervacija"]),
    # Scheduling
    "SchedulingModels": ("model raspoređivanja", "modela raspoređivanja", ["scheduling", "plan rasporeda"]),
    # Stats
    "Stats": ("statistika", "statistike", ["analitika", "brojke", "metrike"]),
    "AvailabilityProjection": ("projekcija dostupnosti", "projekcije dostupnosti", ["predviđanje slobodnih"]),
    # Other
    "CostCenters": ("mjesto troška", "mjesta troška", ["cost center", "profitni centar"]),
    "Pools": ("bazen vozila", "bazena vozila", ["pool", "grupa vozila"]),
    "MasterData": ("matični podaci", "matičnih podataka", ["master data", "osnovni podaci"]),
    "Master": ("matični podaci", "matičnih podataka", ["master", "osnovni"]),
    "Metadata": ("metapodaci", "metapodataka", ["meta", "informacije o strukturi"]),
    "SendEmail": ("slanje emaila", "slanja emaila", ["pošalji mail", "email"]),
    "Demo": ("demo", "demoa", ["demonstracija", "primjer"]),
    "Upsert": ("upsert", "upserta", ["kreiraj ili ažuriraj"]),
    "SyncExternalIdAndLicencePlate": ("sinkronizacija tablica", "sinkronizacije", ["sync registracija"]),
    "SyncExternalIdAndVIN": ("sinkronizacija VIN-a", "sinkronizacije", ["sync šasija"]),
    "WhatCanIDo": ("što mogu", "mogućnosti", ["pomoć", "mogućnosti", "dostupne akcije"]),
    "Booking": ("rezervacija", "rezervacije", ["booking", "zauzimanje"]),
}

# ── Method Translation Map ──
METHOD_MAP = {
    "get": ["dohvati", "prikaži", "pokaži", "pregledaj", "provjeri"],
    "post": ["dodaj", "kreiraj", "napravi", "unesi", "stvori"],
    "put": ["ažuriraj", "promijeni", "izmijeni", "update"],
    "patch": ["djelomično ažuriraj", "promijeni polje", "update polje"],
    "delete": ["obriši", "ukloni", "makni", "izbriši", "briši"],
}

# ── Suffix Translation Map ──
SUFFIX_MAP = {
    "id": ["po ID-u", "specifični", "jedan", "detalji"],
    "DeleteByCriteria": ["prema kriterijima", "bulk brisanje", "masovno brisanje", "po filteru"],
    "documents": ["dokumenti", "datoteke", "prilozi"],
    "metadata": ["metapodaci", "informacije o strukturi"],
    "multipatch": ["grupno ažuriranje", "batch update", "više odjednom"],
    "GroupBy": ["grupirano", "po grupama", "grupiranje"],
    "Agg": ["agregacija", "ukupno", "suma", "prosjek"],
    "ProjectTo": ["projekcija", "odabrana polja", "filtrirani prikaz"],
    "thumb": ["sličica", "thumbnail", "mali prikaz"],
    "SetAsDefault": ["postavi kao zadano", "default", "glavni dokument"],
    "Filter": ["filtrirano", "pretraga", "po kriterijima"],
}

# ── Generic Purpose Prefixes to Replace ──
GENERIC_PREFIXES = [
    "Ovaj endpoint omogućuje brisanje više stavki na temelju kriterija filtriranja.",
    "Dobivanje svih stavki na temelju parametara učitavanja (stranice, sortiranje i filtriranje)",
    "Vraća metapodatke entiteta s određenom primarnom ključnom vrijednošću",
    "Dohvaća sve stavke na temelju parametara učitavanja.",
    "Dohvaća sve stavke na temelju parametara učitavanja, ali projicirane",
    "Ovaj endpoint vraća metapodatke entiteta s određenim primarnim ključem",
    "Ovaj endpoint omogućuje dohvaćanje sličice specifičnog dokumenta",
    "Ovaj endpoint omogućuje postavljanje dokumenta kao zadani za entitet",
    "Ovaj endpoint vraća pojedinačnu stavku na temelju primarnog ključa",
    "Ažurira informacije o dokumentu.",
    "Postavlja dokument kao zadani za entitet.",
    "Briše više stavki na temelju kriterija filtriranja.",
    "Ovaj endpoint omogućuje dohvaćanje specifičnog dokumenta za entitet",
    "Ovaj endpoint omogućuje djelomično ažuriranje više stavki odjednom",
    "Dohvaća vrijednost specificiranu agregatnom funkcijom za stavke",
    "Dohvaća stavke grupirane prema navedenom polju ili poljima",
]

# ── Method Templates for Natural Language Examples ──
EXAMPLE_TEMPLATES = {
    "get": [
        "Pokaži {entity_acc}",
        "Prikaži sve {entity_acc}",
        "Dohvati {entity_acc}",
        "Koji su {entity_nom}?",
    ],
    "get_id": [
        "Pokaži detalje za {entity_acc}",
        "Dohvati specifični {entity_nom}",
    ],
    "post": [
        "Dodaj novi {entity_nom}",
        "Kreiraj {entity_acc}",
        "Unesi novi {entity_nom}",
    ],
    "put": [
        "Ažuriraj {entity_acc}",
        "Promijeni {entity_acc}",
        "Izmijeni {entity_acc}",
    ],
    "patch": [
        "Djelomično ažuriraj {entity_acc}",
        "Promijeni polje za {entity_acc}",
    ],
    "delete": [
        "Obriši {entity_acc}",
        "Ukloni {entity_acc}",
        "Izbriši {entity_acc}",
    ],
    "delete_DeleteByCriteria": [
        "Obriši {entity_acc} prema kriterijima",
        "Masovno brisanje {entity_gen}",
    ],
}


def parse_operation_id(op_id: str):
    """Parse operation_id into (method, entity, suffix)."""
    parts = op_id.split("_", 1)
    if len(parts) < 2:
        return op_id, "", ""

    method = parts[0]  # get, post, put, patch, delete
    rest = parts[1]

    # Known suffixes to strip
    suffix = ""
    for s in ["DeleteByCriteria", "GroupBy", "Agg", "ProjectTo",
              "multipatch", "SetAsDefault", "Filter", "thumb"]:
        if rest.endswith("_" + s):
            suffix = s
            rest = rest[: -(len(s) + 1)]
            break
        elif rest.endswith(s):
            suffix = s
            rest = rest[: -len(s)]
            break

    # Check for _id_documents_documentId pattern (before _id check)
    if "_id_documents" in rest:
        suffix = suffix or "documents"
        rest = rest.split("_id_documents")[0]
    elif "_documents" in rest:
        suffix = suffix or "documents"
        rest = rest.split("_documents")[0]

    # Check for _id_metadata pattern (before _id check)
    if "_id_metadata" in rest:
        suffix = suffix or "metadata"
        rest = rest.split("_id_metadata")[0]
    elif "_metadata" in rest or rest.endswith("_Metadata"):
        suffix = suffix or "metadata"
        rest = rest.split("_metadata")[0].split("_Metadata")[0]

    # Check for _id suffix
    if rest.endswith("_id"):
        suffix = suffix or "id"
        rest = rest[:-3]

    entity = rest.strip("_")
    return method, entity, suffix


def generate_synonyms(method: str, entity: str, suffix: str):
    """Generate Croatian synonyms for a tool."""
    synonyms = []

    # Get entity translations — try exact, then first part of compound
    entity_info = ENTITY_MAP.get(entity)
    if not entity_info and "_" in entity:
        # Try the last part (e.g., Lookup_Companies -> Companies)
        last_part = entity.rsplit("_", 1)[-1]
        entity_info = ENTITY_MAP.get(last_part)
        if not entity_info:
            # Try the first part (e.g., Stats_PeriodicActivities -> Stats)
            first_part = entity.split("_", 1)[0]
            entity_info = ENTITY_MAP.get(first_part)
    if not entity_info:
        return []

    entity_nom, entity_gen, entity_synonyms = entity_info

    # Add entity synonyms
    synonyms.extend(entity_synonyms)

    # Add method + entity combinations
    method_verbs = METHOD_MAP.get(method, [])
    for verb in method_verbs[:3]:
        synonyms.append(f"{verb} {entity_nom}")

    # Add suffix-specific phrases
    if suffix and suffix in SUFFIX_MAP:
        for phrase in SUFFIX_MAP[suffix][:2]:
            synonyms.append(f"{phrase} {entity_gen}")

    # Add colloquial combined phrases
    if method == "get":
        synonyms.append(f"koji su {entity_nom}")
        synonyms.append(f"koliko {entity_gen}")
    elif method == "delete":
        synonyms.append(f"želim obrisati {entity_nom}")
    elif method == "post":
        synonyms.append(f"želim dodati {entity_nom}")
        synonyms.append(f"novi {entity_nom}")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in synonyms:
        s_lower = s.lower()
        if s_lower not in seen:
            seen.add(s_lower)
            unique.append(s)

    return unique[:10]  # Max 10 synonyms


def fix_purpose(purpose: str, entity: str, method: str):
    """Replace generic purpose with entity-specific version."""
    entity_info = ENTITY_MAP.get(entity)
    if not entity_info:
        return purpose

    entity_nom, entity_gen, _ = entity_info

    for prefix in GENERIC_PREFIXES:
        if purpose.startswith(prefix[:40]):
            # Build entity-specific purpose
            actions = {
                "get": "Dohvaćanje",
                "post": "Kreiranje",
                "put": "Ažuriranje",
                "patch": "Djelomično ažuriranje",
                "delete": "Brisanje",
            }
            action = actions.get(method, "Obrada")

            if "brisanje više stavki" in purpose.lower() or "deletebycriterio" in purpose.lower():
                return f"{action} više {entity_gen} na temelju kriterija filtriranja."
            elif "metapodatke" in purpose.lower() or "metadata" in purpose.lower():
                return f"Vraća metapodatke za {entity_gen}."
            elif "sličice" in purpose.lower() or "thumb" in purpose.lower():
                return f"Dohvaćanje sličice dokumenta za {entity_gen}."
            elif "postavljanje dokumenta" in purpose.lower() or "zadani" in purpose.lower():
                return f"Postavljanje dokumenta kao zadanog za {entity_gen}."
            elif "djelomično ažuriranje više" in purpose.lower():
                return f"Djelomično ažuriranje više {entity_gen} odjednom."
            elif "agregatnom" in purpose.lower():
                return f"Dohvaća agregiranu vrijednost za {entity_gen}."
            elif "grupirane" in purpose.lower():
                return f"Dohvaća {entity_gen} grupirane po polju."
            elif "projicirane" in purpose.lower():
                return f"Dohvaća {entity_gen} s odabranim poljima (projekcija)."
            elif "pojedinačnu" in purpose.lower():
                return f"Dohvaća pojedinačni {entity_nom} po primarnom ključu."
            elif "svih stavki" in purpose.lower() or "parametara učitavanja" in purpose.lower():
                return f"Dohvaćanje svih {entity_gen} s podrškom za straničenje i sortiranje."
            elif "informacije o dokumentu" in purpose.lower():
                return f"Ažurira informacije o dokumentu za {entity_gen}."
            else:
                return f"{action} {entity_gen}."

    return purpose


def fix_examples(examples: list, method: str, entity: str, suffix: str):
    """Replace API-style examples with natural language ones."""
    # Check if ALL examples are API-style
    all_api = all(
        ex.startswith(("GET ", "POST ", "PUT ", "DELETE ", "PATCH "))
        for ex in examples
    )
    if not all_api:
        return examples

    entity_info = ENTITY_MAP.get(entity)
    if not entity_info:
        return examples

    entity_nom, entity_gen, _ = entity_info
    # Use nominative as accusative approximation
    entity_acc = entity_nom

    # Pick template based on method + suffix
    template_key = method
    if suffix == "id":
        template_key = "get_id" if method == "get" else method
    elif suffix == "DeleteByCriteria":
        template_key = "delete_DeleteByCriteria"

    templates = EXAMPLE_TEMPLATES.get(template_key, EXAMPLE_TEMPLATES.get(method, []))

    new_examples = []
    for tmpl in templates[:2]:
        new_examples.append(tmpl.format(
            entity_nom=entity_nom,
            entity_gen=entity_gen,
            entity_acc=entity_acc,
        ))

    return new_examples if new_examples else examples


def main():
    doc_path = os.path.join(os.path.dirname(__file__), "..", "config", "tool_documentation.json")
    doc_path = os.path.abspath(doc_path)

    print(f"Loading {doc_path}...")
    with open(doc_path, encoding="utf-8") as f:
        docs = json.load(f)

    total = len(docs)
    synonyms_added = 0
    purposes_fixed = 0
    examples_fixed = 0

    for op_id, tool_doc in docs.items():
        method, entity, suffix = parse_operation_id(op_id)

        # 1. Generate synonyms_hr
        synonyms = generate_synonyms(method, entity, suffix)
        if synonyms:
            tool_doc["synonyms_hr"] = synonyms
            synonyms_added += 1

        # 2. Fix duplicate/generic purposes
        old_purpose = tool_doc.get("purpose", "")
        new_purpose = fix_purpose(old_purpose, entity, method)
        if new_purpose != old_purpose:
            tool_doc["purpose"] = new_purpose
            purposes_fixed += 1

        # 3. Fix API-style examples
        old_examples = tool_doc.get("example_queries_hr", [])
        new_examples = fix_examples(old_examples, method, entity, suffix)
        if new_examples != old_examples:
            tool_doc["example_queries_hr"] = new_examples
            examples_fixed += 1

    # Write back
    print(f"Writing {doc_path}...")
    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Results:")
    print(f"  Total tools: {total}")
    print(f"  synonyms_hr added: {synonyms_added}")
    print(f"  purposes fixed: {purposes_fixed}")
    print(f"  examples fixed: {examples_fixed}")
    print(f"  Unchanged: {total - max(synonyms_added, purposes_fixed, examples_fixed)}")
    print(f"\nNext: delete .cache/tool_embeddings.json and restart Docker")


if __name__ == "__main__":
    main()

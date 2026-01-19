"""
Script to add Croatian synonyms to tool_documentation.json.
This improves FAISS semantic search accuracy by providing
multiple ways users might phrase their requests.
"""

import json
from pathlib import Path
from datetime import datetime

# Key tools and their Croatian synonyms
# These are the most frequently used tools by end users
TOOL_SYNONYMS = {
    # ═══════════════════════════════════════════════════════════════
    # VEHICLE INFO & MILEAGE (READ)
    # ═══════════════════════════════════════════════════════════════
    "get_MasterData": [
        "podaci o vozilu", "informacije o autu", "moje vozilo",
        "koliko imam kilometara", "kolika je kilometraža", "stanje km",
        "registracija vozila", "tablice auta", "koja je tablica",
        "do kada vrijedi registracija", "istek registracije",
        "lizing kuća", "leasing", "tko je vlasnik",
        "koji auto vozim", "koje vozilo imam", "detalji vozila",
        "stanje brojača", "prijeđeni kilometri", "ukupna km"
    ],
    "get_Vehicles": [
        "sva vozila", "lista vozila", "popis automobila",
        "dohvati vozila", "prikaži vozila", "pregledaj vozila",
        "koji auti postoje", "koja vozila imamo", "flota vozila"
    ],
    "get_Vehicles_id": [
        "detalji vozila", "informacije o vozilu", "podaci auta",
        "dohvati vozilo po id-u", "specifično vozilo", "jedno vozilo"
    ],
    "get_LatestMileageReports": [
        "zadnja kilometraža", "najnovija km", "posljednji unos km",
        "trenutna kilometraža", "aktualna km", "latest mileage"
    ],
    "get_MileageReports": [
        "povijest kilometraže", "izvještaji km", "svi unosi km",
        "pregled kilometraže", "mileage report", "km izvještaj"
    ],

    # ═══════════════════════════════════════════════════════════════
    # BOOKING & AVAILABILITY (READ)
    # ═══════════════════════════════════════════════════════════════
    "get_VehicleCalendar": [
        "moje rezervacije", "moji bookingi", "kalendar vozila",
        "kada imam auto", "kad imam vozilo", "raspored rezervacija",
        "prikaži bookinge", "pokaži rezervacije", "pregled rezervacija"
    ],
    "get_AvailableVehicles": [
        "slobodna vozila", "dostupni auti", "koja vozila su slobodna",
        "raspoloživa vozila", "mogu li rezervirati", "ima li slobodnih",
        "available vehicles", "free cars", "koji auti su dostupni"
    ],

    # ═══════════════════════════════════════════════════════════════
    # CASES & DAMAGES (READ)
    # ═══════════════════════════════════════════════════════════════
    "get_Cases": [
        "prijavljene štete", "svi kvarovi", "slučajevi",
        "moje prijave", "povijest šteta", "pregled kvarova",
        "lista slučajeva", "damage reports", "incidents"
    ],
    "get_Cases_id": [
        "detalji štete", "informacije o kvaru", "specifični slučaj",
        "jedna prijava", "podaci o šteti"
    ],

    # ═══════════════════════════════════════════════════════════════
    # EXPENSES & TRIPS (READ)
    # ═══════════════════════════════════════════════════════════════
    "get_Expenses": [
        "troškovi", "izdaci", "računi", "potrošnja",
        "koliko sam potrošio", "pregled troškova", "svi troškovi",
        "expense report", "costs"
    ],
    "get_Trips": [
        "putovanja", "vožnje", "tripovi", "putni nalozi",
        "moja putovanja", "povijest vožnji", "pregled tripova"
    ],

    # ═══════════════════════════════════════════════════════════════
    # PERSONS & TEAMS (READ)
    # ═══════════════════════════════════════════════════════════════
    "get_PersonData_personIdOrEmail": [
        "moji podaci", "moj profil", "tko sam ja",
        "moje ime", "moj email", "moj telefon",
        "osobni podaci", "informacije o meni", "user profile"
    ],
    "get_Persons": [
        "svi zaposlenici", "lista osoba", "popis radnika",
        "tko sve radi", "djelatnici", "korisnici sustava"
    ],
    "get_Teams": [
        "timovi", "grupe", "ekipe", "odjeli",
        "lista timova", "svi timovi", "team list"
    ],

    # ═══════════════════════════════════════════════════════════════
    # COMPANIES & PARTNERS (READ)
    # ═══════════════════════════════════════════════════════════════
    "get_Companies": [
        "kompanije", "tvrtke", "firme", "poduzeća",
        "lista kompanija", "sve tvrtke", "company list"
    ],
    "get_Partners": [
        "partneri", "dobavljači", "klijenti", "suradnici",
        "lista partnera", "svi dobavljači", "partner list"
    ],

    # ═══════════════════════════════════════════════════════════════
    # EQUIPMENT (READ)
    # ═══════════════════════════════════════════════════════════════
    "get_Equipment": [
        "oprema", "inventar", "alati", "sredstva",
        "lista opreme", "sva oprema", "equipment list"
    ],

    # ═══════════════════════════════════════════════════════════════
    # MILEAGE INPUT (POST/WRITE)
    # ═══════════════════════════════════════════════════════════════
    "post_AddMileage": [
        "unesi kilometražu", "upiši km", "dodaj kilometre",
        "nova kilometraža", "prijavi kilometre", "stanje brojača",
        "koliko sam prešao", "add mileage", "input km",
        "ažuriraj km", "spremi kilometražu", "zabilježi km",
        "unos kilometara", "upis stanja", "record mileage"
    ],

    # ═══════════════════════════════════════════════════════════════
    # CASE/DAMAGE CREATION (POST/WRITE)
    # ═══════════════════════════════════════════════════════════════
    "post_AddCase": [
        "prijavi štetu", "prijavi kvar", "nova šteta",
        "udario sam", "ogrebao sam", "oštetio sam auto",
        "imam problem s autom", "nešto ne radi", "auto je pokvaren",
        "report damage", "create case", "nova prijava",
        "incident", "nesreća", "oštećenje vozila"
    ],

    # ═══════════════════════════════════════════════════════════════
    # BOOKING CREATION (POST/WRITE)
    # ═══════════════════════════════════════════════════════════════
    "post_VehicleCalendar": [
        "rezerviraj vozilo", "nova rezervacija", "zauzmi auto",
        "trebam vozilo", "trebam auto", "book vehicle",
        "kreiraj booking", "dodaj rezervaciju", "napravi booking",
        "želim rezervirati", "hoću auto", "mogu li uzeti auto"
    ],
    "post_Booking": [
        "nova rezervacija", "kreiraj booking", "rezerviraj",
        "zauzmi", "book", "create reservation"
    ],

    # ═══════════════════════════════════════════════════════════════
    # BOOKING CANCELLATION (DELETE)
    # ═══════════════════════════════════════════════════════════════
    "delete_VehicleCalendar_id": [
        "otkaži rezervaciju", "poništi booking", "obriši rezervaciju",
        "cancel booking", "ne trebam više auto", "odustani od rezervacije",
        "storniraj booking", "ukloni rezervaciju"
    ],

    # ═══════════════════════════════════════════════════════════════
    # ORG UNITS & COST CENTERS (READ)
    # ═══════════════════════════════════════════════════════════════
    "get_OrgUnits": [
        "organizacijske jedinice", "odjeli", "sektori",
        "org struktura", "hijerarhija", "departments"
    ],
    "get_CostCenters": [
        "troškovni centri", "mjesta troška", "cost centers",
        "profitni centri", "centri troškova"
    ],

    # ═══════════════════════════════════════════════════════════════
    # PERIODIC ACTIVITIES / SERVICE (READ)
    # ═══════════════════════════════════════════════════════════════
    "get_PeriodicActivities": [
        "periodične aktivnosti", "servisi", "održavanje",
        "redovne aktivnosti", "planirani servisi", "maintenance"
    ],
    "get_LatestPeriodicActivities": [
        "zadnji servis", "posljednje održavanje", "najnovija aktivnost",
        "kada je bio servis", "latest service"
    ],

    # ═══════════════════════════════════════════════════════════════
    # VEHICLE CONTRACTS (READ)
    # ═══════════════════════════════════════════════════════════════
    "get_VehicleContracts": [
        "ugovori vozila", "lizing ugovori", "leasing contracts",
        "najam vozila", "rental agreements"
    ],
    "get_LatestVehicleContracts": [
        "zadnji ugovor", "aktualni ugovor", "trenutni najam",
        "latest contract"
    ],
}


def add_synonyms_to_documentation():
    """Add synonyms to tool_documentation.json."""
    config_dir = Path(__file__).parent.parent / "config"
    doc_path = config_dir / "tool_documentation.json"

    # Load existing documentation
    with open(doc_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} tools from tool_documentation.json")

    # Track changes
    updated_count = 0
    added_tools = []

    # Add synonyms to matching tools
    for tool_id, synonyms in TOOL_SYNONYMS.items():
        # Try exact match first
        if tool_id in docs:
            docs[tool_id]["synonyms_hr"] = synonyms
            updated_count += 1
            added_tools.append(tool_id)
        else:
            # Try case-insensitive match
            for doc_tool_id in docs.keys():
                if doc_tool_id.lower() == tool_id.lower():
                    docs[doc_tool_id]["synonyms_hr"] = synonyms
                    updated_count += 1
                    added_tools.append(doc_tool_id)
                    break
            else:
                print(f"  WARNING: Tool not found: {tool_id}")

    # Save updated documentation
    with open(doc_path, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"\nUpdated {updated_count} tools with synonyms_hr")
    print(f"Tools updated: {', '.join(added_tools[:10])}...")

    return updated_count


if __name__ == "__main__":
    add_synonyms_to_documentation()

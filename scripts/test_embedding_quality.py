#!/usr/bin/env python3
"""
Test Embedding Quality Improvement Script (Standalone)
Tests the enhanced _generate_purpose function logic.

This script verifies that:
1. Path-based entity extraction works correctly
2. operationId parsing works correctly
3. Output key mappings work correctly
4. Tools without descriptions now get meaningful embedding text
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


# Minimal dataclass to simulate ParameterDefinition
class DependencySource(Enum):
    FROM_USER = "from_user"
    FROM_CONTEXT = "from_context"
    FROM_TOOL_OUTPUT = "from_tool_output"

@dataclass
class ParameterDefinition:
    name: str
    param_type: str = "string"
    required: bool = False
    location: str = "query"
    dependency_source: DependencySource = DependencySource.FROM_USER
    format: Optional[str] = None
    description: str = ""
    default_value: Optional[any] = None


# Copy of the enhanced mapping and functions from embedding_engine.py
PATH_ENTITY_MAP = {
    # Vehicles & Fleet
    "vehicle": ("vozilo", "vozila"),
    "vehicles": ("vozilo", "vozila"),
    "car": ("automobil", "automobila"),
    "fleet": ("flota", "flote"),
    "fleets": ("flota", "flote"),
    "asset": ("imovina", "imovine"),
    "assets": ("imovina", "imovine"),
    # People
    "person": ("osoba", "osobe"),
    "persons": ("osoba", "osobe"),
    "people": ("osoba", "osobe"),
    "driver": ("vozac", "vozaca"),
    "drivers": ("vozac", "vozaca"),
    "user": ("korisnik", "korisnika"),
    "users": ("korisnik", "korisnika"),
    "employee": ("zaposlenik", "zaposlenika"),
    "employees": ("zaposlenik", "zaposlenika"),
    "customer": ("kupac", "kupca"),
    "customers": ("kupac", "kupaca"),
    "contact": ("kontakt", "kontakta"),
    "contacts": ("kontakt", "kontakata"),
    # Bookings & Reservations
    "booking": ("rezervacija", "rezervacije"),
    "bookings": ("rezervacija", "rezervacije"),
    "reservation": ("rezervacija", "rezervacije"),
    "reservations": ("rezervacija", "rezervacija"),
    "calendar": ("kalendar", "kalendara"),
    "appointment": ("termin", "termina"),
    # Locations
    "location": ("lokacija", "lokacije"),
    "locations": ("lokacija", "lokacija"),
    "address": ("adresa", "adrese"),
    "zone": ("zona", "zone"),
    "zones": ("zona", "zona"),
    "region": ("regija", "regije"),
    "branch": ("poslovnica", "poslovnice"),
    "branches": ("poslovnica", "poslovnica"),
    "station": ("stanica", "stanice"),
    "depot": ("depo", "depoa"),
    # Documents & Records
    "document": ("dokument", "dokumenta"),
    "documents": ("dokument", "dokumenata"),
    "invoice": ("racun", "racuna"),
    "invoices": ("racun", "racuna"),
    "contract": ("ugovor", "ugovora"),
    "contracts": ("ugovor", "ugovora"),
    "report": ("izvjestaj", "izvjestaja"),
    "reports": ("izvjestaj", "izvjestaja"),
    "log": ("zapis", "zapisa"),
    "logs": ("zapis", "zapisa"),
    "history": ("povijest", "povijesti"),
    "record": ("zapis", "zapisa"),
    # Maintenance & Service
    "maintenance": ("odrzavanje", "odrzavanja"),
    "service": ("servis", "servisa"),
    "repair": ("popravak", "popravka"),
    "inspection": ("inspekcija", "inspekcije"),
    "damage": ("steta", "stete"),
    "damages": ("steta", "steta"),
    "accident": ("nesreca", "nesrece"),
    "insurance": ("osiguranje", "osiguranja"),
    # Financial
    "payment": ("placanje", "placanja"),
    "payments": ("placanje", "placanja"),
    "cost": ("trosak", "troska"),
    "costs": ("trosak", "troskova"),
    "expense": ("trosak", "troska"),
    "expenses": ("trosak", "troskova"),
    "fee": ("naknada", "naknade"),
    "fees": ("naknada", "naknada"),
    "price": ("cijena", "cijene"),
    "pricing": ("cjenik", "cjenika"),
    "tariff": ("tarifa", "tarife"),
    "billing": ("naplata", "naplate"),
    "transaction": ("transakcija", "transakcije"),
    # Status & Metrics
    "status": ("status", "statusa"),
    "state": ("stanje", "stanja"),
    "mileage": ("kilometraza", "kilometraze"),
    "odometer": ("kilometraza", "kilometraze"),
    "fuel": ("gorivo", "goriva"),
    "battery": ("baterija", "baterije"),
    "tire": ("guma", "gume"),
    "tires": ("guma", "guma"),
    "oil": ("ulje", "ulja"),
    # Access & Permissions
    "permission": ("dozvola", "dozvole"),
    "permissions": ("dozvola", "dozvola"),
    "role": ("uloga", "uloge"),
    "roles": ("uloga", "uloga"),
    "access": ("pristup", "pristupa"),
    "group": ("grupa", "grupe"),
    "groups": ("grupa", "grupa"),
    "team": ("tim", "tima"),
    "department": ("odjel", "odjela"),
    # Equipment & Accessories
    "equipment": ("oprema", "opreme"),
    "accessory": ("dodatak", "dodatka"),
    "accessories": ("dodatak", "dodataka"),
    "device": ("uredaj", "uredaja"),
    "devices": ("uredaj", "uredaja"),
    "tracker": ("tracker", "trackera"),
    "gps": ("GPS", "GPS-a"),
    "telematics": ("telematika", "telematike"),
    # Categories & Types
    "category": ("kategorija", "kategorije"),
    "categories": ("kategorija", "kategorija"),
    "type": ("tip", "tipa"),
    "types": ("tip", "tipova"),
    "class": ("klasa", "klase"),
    "brand": ("marka", "marke"),
    "model": ("model", "modela"),
    # Time-related
    "period": ("period", "perioda"),
    "schedule": ("raspored", "rasporeda"),
    "shift": ("smjena", "smjene"),
    "availability": ("dostupnost", "dostupnosti"),
    "slot": ("termin", "termina"),
    # Misc
    "notification": ("obavijest", "obavijesti"),
    "notifications": ("obavijest", "obavijesti"),
    "alert": ("upozorenje", "upozorenja"),
    "alerts": ("upozorenje", "upozorenja"),
    "message": ("poruka", "poruke"),
    "note": ("biljeska", "biljeske"),
    "notes": ("biljeska", "biljeski"),
    "comment": ("komentar", "komentara"),
    "tag": ("oznaka", "oznake"),
    "tags": ("oznaka", "oznaka"),
    "image": ("slika", "slike"),
    "images": ("slika", "slika"),
    "photo": ("fotografija", "fotografije"),
    "file": ("datoteka", "datoteke"),
    "attachment": ("privitak", "privitka"),
    "tenant": ("najmodavac", "najmodavca"),
    "organization": ("organizacija", "organizacije"),
    "company": ("tvrtka", "tvrtke"),
    "license": ("licenca", "licence"),
    "registration": ("registracija", "registracije"),
    "certificate": ("certifikat", "certifikata"),
    "policy": ("polica", "police"),
    "claim": ("zahtjev", "zahtjeva"),
    "request": ("zahtjev", "zahtjeva"),
    "order": ("narudzba", "narudzbe"),
    "pool": ("bazen", "bazena"),
    "trip": ("putovanje", "putovanja"),
    "route": ("ruta", "rute"),
    "journey": ("voznja", "voznje"),
    "ride": ("voznja", "voznje"),
    "transfer": ("transfer", "transfera"),
    "pickup": ("preuzimanje", "preuzimanja"),
    "dropoff": ("vracanje", "vracanja"),
    "checkin": ("prijava", "prijave"),
    "checkout": ("odjava", "odjave"),
    "handover": ("primopredaja", "primopredaje"),
    "key": ("kljuc", "kljuca"),
    "keys": ("kljuc", "kljuceva"),
    "card": ("kartica", "kartice"),
    "fuelcard": ("kartica za gorivo", "kartice za gorivo"),
    "tollcard": ("ENC kartica", "ENC kartice"),
    "violation": ("prekrsaj", "prekrsaja"),
    "fine": ("kazna", "kazne"),
    "penalty": ("kazna", "kazne"),
}

OUTPUT_KEY_MAP = {
    "mileage": "kilometrazu",
    "km": "kilometre",
    "odometer": "stanje kilometara",
    "fuel": "razinu goriva",
    "fuellevel": "razinu goriva",
    "fuelconsumption": "potrosnju goriva",
    "battery": "stanje baterije",
    "batterylevel": "razinu baterije",
    "speed": "brzinu",
    "location": "lokaciju",
    "position": "poziciju",
    "coordinates": "koordinate",
    "status": "status",
    "state": "stanje",
    "available": "dostupnost",
    "availability": "dostupnost",
    "active": "aktivnost",
    "registration": "registraciju",
    "registrationnumber": "registarsku oznaku",
    "plate": "tablice",
    "licenseplate": "registarske tablice",
    "vin": "broj sasije",
    "expiry": "datum isteka",
    "expirydate": "datum isteka",
    "validuntil": "vrijedi do",
    "date": "datum",
    "time": "vrijeme",
    "datetime": "datum i vrijeme",
    "timestamp": "vremensku oznaku",
    "createdat": "datum kreiranja",
    "updatedat": "datum azuriranja",
    "duration": "trajanje",
    "price": "cijenu",
    "cost": "trosak",
    "amount": "iznos",
    "total": "ukupan iznos",
    "tax": "porez",
    "discount": "popust",
    "id": "identifikator",
    "name": "naziv",
    "title": "naslov",
    "description": "opis",
    "code": "sifru",
    "number": "broj",
    "email": "e-mail",
    "phone": "telefon",
    "address": "adresu",
    "city": "grad",
    "country": "drzavu",
    "firstname": "ime",
    "lastname": "prezime",
    "fullname": "puno ime",
    "count": "broj",
    "items": "stavke",
    "list": "popis",
    "results": "rezultate",
    "bookingid": "ID rezervacije",
    "bookingnumber": "broj rezervacije",
    "pickupdate": "datum preuzimanja",
    "returndate": "datum vracanja",
}


def extract_entities_from_path(path: str) -> List[str]:
    """Extract entities from API path segments."""
    if not path:
        return []

    entities = []
    clean_path = re.sub(r'\{[^}]+\}', '', path)
    segments = re.split(r'[/\-_]', clean_path.lower())

    for segment in segments:
        if not segment or len(segment) < 3:
            continue

        if segment in PATH_ENTITY_MAP:
            singular, _ = PATH_ENTITY_MAP[segment]
            if singular not in entities:
                entities.append(singular)
        else:
            for key, (singular, _) in PATH_ENTITY_MAP.items():
                if key in segment and singular not in entities:
                    entities.append(singular)
                    break

    return entities[:3]


def parse_operation_id(operation_id: str) -> tuple:
    """Parse operationId to extract action and entities."""
    if not operation_id:
        return [], ""

    words = re.findall(r'[A-Z][a-z]*|[a-z]+', operation_id)

    if not words:
        return [], ""

    entities = []
    action_hint = ""

    action_verbs = {"get", "create", "update", "delete", "post", "put",
                    "patch", "list", "find", "search", "add", "remove",
                    "set", "fetch", "retrieve", "check", "validate"}

    for word in words:
        word_lower = word.lower()

        if word_lower in action_verbs:
            continue

        if word_lower in PATH_ENTITY_MAP:
            singular, _ = PATH_ENTITY_MAP[word_lower]
            if singular not in entities:
                entities.append(singular)
        elif word_lower in OUTPUT_KEY_MAP:
            if not action_hint:
                action_hint = OUTPUT_KEY_MAP[word_lower]

    return entities[:2], action_hint


def generate_purpose(
    method: str,
    parameters: Dict[str, ParameterDefinition],
    output_keys: List[str],
    path: str = "",
    operation_id: str = ""
) -> str:
    """Auto-generate purpose from API structure (v3.0 - Enhanced)."""
    actions = {
        "GET": "Dohvaca",
        "POST": "Kreira",
        "PUT": "Azurira",
        "PATCH": "Azurira",
        "DELETE": "Brise"
    }
    action = actions.get(method.upper(), "Obraduje")

    path_entities = extract_entities_from_path(path)
    op_entities, op_action_hint = parse_operation_id(operation_id)

    param_context = []
    has_time = False

    if parameters:
        names = [p.name.lower() for p in parameters.values()]

        for name in names:
            for key, (singular, _) in PATH_ENTITY_MAP.items():
                if key in name and singular not in param_context:
                    param_context.append(singular)
                    break

        has_time = (
            any(x in n for n in names for x in ["from", "start", "begin"]) and
            any(x in n for n in names for x in ["to", "end", "until"])
        )

    result = []

    if output_keys:
        keys_lower = [k.lower() for k in output_keys]

        for key in keys_lower:
            for pattern, translation in OUTPUT_KEY_MAP.items():
                if pattern in key and translation not in result:
                    result.append(translation)
                    if len(result) >= 4:
                        break
            if len(result) >= 4:
                break

    all_entities = []
    seen = set()

    for entity in path_entities + op_entities + param_context:
        if entity.lower() not in seen:
            all_entities.append(entity)
            seen.add(entity.lower())

    purpose = action

    if result:
        purpose += " " + ", ".join(result[:3])
    elif op_action_hint:
        purpose += " " + op_action_hint
    elif method == "GET":
        purpose += " podatke"
    elif method == "POST":
        purpose += " novi zapis"
    elif method in ("PUT", "PATCH"):
        purpose += " postojece podatke"
    elif method == "DELETE":
        purpose += " zapis"

    if all_entities:
        entity_genitives = []
        for entity in all_entities[:2]:
            for key, (singular, genitive) in PATH_ENTITY_MAP.items():
                if singular == entity:
                    entity_genitives.append(genitive)
                    break
            else:
                entity_genitives.append(entity)

        purpose += " za " + ", ".join(entity_genitives)

    if has_time:
        purpose += " u zadanom periodu"

    return purpose


def test_path_entity_extraction():
    """Test that entities are correctly extracted from API paths."""
    print("\n" + "="*60)
    print("TEST: Path Entity Extraction")
    print("="*60)

    test_cases = [
        ("/api/v1/vehicles/{vehicleId}/mileage", ["vozilo"]),
        ("/api/v1/persons/{personId}/bookings", ["osoba", "rezervacija"]),
        ("/api/v1/fleet/{fleetId}/vehicles", ["flota", "vozilo"]),
        ("/api/v1/locations/{locationId}", ["lokacija"]),
        ("/api/v1/maintenance/schedule", ["odrzavanje", "raspored"]),
        ("/api/v1/invoices/{invoiceId}/payments", ["racun", "placanje"]),
        ("/api/v1/drivers/{driverId}/trips", ["vozac", "putovanje"]),
        ("/api/v1/fuel-cards/{cardId}", ["gorivo", "kartica"]),
        ("/api/v1/booking-calendar/availability", ["rezervacija"]),
        ("/api/v1/damage-reports/{reportId}", ["steta"]),
    ]

    passed = 0
    failed = 0

    for path, expected_entities in test_cases:
        result = extract_entities_from_path(path)

        found_any = any(e in result for e in expected_entities)

        if found_any:
            print(f"  [OK] {path}")
            print(f"       -> {result}")
            passed += 1
        else:
            print(f"  [FAIL] {path}")
            print(f"       Expected one of: {expected_entities}")
            print(f"       Got: {result}")
            failed += 1

    print(f"\nResult: {passed}/{passed+failed} passed")
    return passed, failed


def test_operation_id_parsing():
    """Test that operationId is correctly parsed."""
    print("\n" + "="*60)
    print("TEST: OperationId Parsing")
    print("="*60)

    test_cases = [
        ("GetVehicleMileage", ["vozilo"]),
        ("CreateBooking", ["rezervacija"]),
        ("UpdatePersonAddress", ["osoba", "adresa"]),
        ("DeleteInvoice", ["racun"]),
        ("GetFleetVehicles", ["flota", "vozilo"]),
        ("ListDriverTrips", ["vozac", "putovanje"]),
        ("GetMaintenanceSchedule", ["odrzavanje", "raspored"]),
        ("CheckAvailability", ["dostupnost"]),
        ("GetFuelConsumption", ["gorivo"]),
        ("GetRegistrationExpiry", ["registracija"]),
    ]

    passed = 0
    failed = 0

    for op_id, expected_entities in test_cases:
        entities, hint = parse_operation_id(op_id)

        found_entity = any(e in entities for e in expected_entities)

        if found_entity:
            print(f"  [OK] {op_id}")
            print(f"       -> entities: {entities}, hint: '{hint}'")
            passed += 1
        else:
            print(f"  [FAIL] {op_id}")
            print(f"       Expected one of: {expected_entities}")
            print(f"       Got entities: {entities}")
            failed += 1

    print(f"\nResult: {passed}/{passed+failed} passed")
    return passed, failed


def test_purpose_generation():
    """Test full purpose generation with various inputs."""
    print("\n" + "="*60)
    print("TEST: Purpose Generation (Full)")
    print("="*60)

    test_cases = [
        # (method, path, operation_id, params, output_keys, expected_keywords)
        (
            "GET",
            "/api/v1/vehicles/{vehicleId}/mileage",
            "GetVehicleMileage",
            {"vehicleId": ParameterDefinition(name="vehicleId", param_type="string", required=True, location="path")},
            ["Mileage", "LastUpdated"],
            ["vozil", "kilometr"]
        ),
        (
            "POST",
            "/api/v1/bookings",
            "CreateBooking",
            {},
            ["BookingId", "Status"],
            ["Kreira", "rezervacij"]
        ),
        (
            "PUT",
            "/api/v1/persons/{personId}",
            "UpdatePerson",
            {},
            ["PersonId", "UpdatedAt"],
            ["Azurira", "osob"]
        ),
        (
            "DELETE",
            "/api/v1/invoices/{invoiceId}",
            "DeleteInvoice",
            {},
            [],
            ["Brise", "racun"]
        ),
        (
            "GET",
            "/api/v1/drivers/{driverId}/trips",
            "GetDriverTrips",
            {
                "driverId": ParameterDefinition(name="driverId", param_type="string", required=True, location="path"),
                "fromDate": ParameterDefinition(name="fromDate", param_type="string", required=False, location="query"),
                "toDate": ParameterDefinition(name="toDate", param_type="string", required=False, location="query"),
            },
            ["TripId", "Duration", "Distance"],
            ["vozac", "putovanj", "period"]
        ),
        (
            "GET",
            "/api/v1/vehicles/{vehicleId}/fuel-consumption",
            "GetFuelConsumption",
            {},
            ["FuelLevel", "Consumption"],
            ["goriv", "vozil"]
        ),
        (
            "GET",
            "/api/v1/maintenance/schedule",
            "GetMaintenanceSchedule",
            {},
            ["ScheduleId", "NextServiceDate"],
            ["odrzavanj", "raspored"]
        ),
        (
            "GET",
            "/api/v1/fleet/{fleetId}/location-restrictions",
            "GetFleetLocationRestrictions",
            {},
            [],
            ["flot", "lokacij"]
        ),
    ]

    passed = 0
    failed = 0

    for method, path, op_id, params, output_keys, expected_keywords in test_cases:
        purpose = generate_purpose(method, params, output_keys, path, op_id)

        purpose_lower = purpose.lower()
        found_all = all(kw.lower() in purpose_lower for kw in expected_keywords)

        if found_all:
            print(f"  [OK] {op_id}")
            print(f"       -> \"{purpose}\"")
            passed += 1
        else:
            print(f"  [FAIL] {op_id}")
            print(f"       Generated: \"{purpose}\"")
            print(f"       Missing: {[k for k in expected_keywords if k.lower() not in purpose_lower]}")
            failed += 1

    print(f"\nResult: {passed}/{passed+failed} passed")
    return passed, failed


def test_comparison_old_vs_new():
    """Show comparison of what the old vs new generation would produce."""
    print("\n" + "="*60)
    print("COMPARISON: Old vs New Purpose Generation")
    print("="*60)

    def old_generate_purpose(method: str) -> str:
        actions = {
            "GET": "Dohvaca",
            "POST": "Kreira",
            "PUT": "Azurira",
            "DELETE": "Brise"
        }
        action = actions.get(method.upper(), "Obraduje")
        if method == "GET":
            return action + " podatke"
        elif method == "POST":
            return action + " novi zapis"
        elif method in ("PUT", "PATCH"):
            return action + " postojece podatke"
        elif method == "DELETE":
            return action + " zapis"
        return action

    test_cases = [
        ("GET", "/api/v1/vehicles/{id}/mileage", "GetVehicleMileage", {}, ["Mileage"]),
        ("GET", "/api/v1/fleet/{id}/location-restrictions", "GetFleetLocationRestrictions", {}, []),
        ("POST", "/api/v1/bookings", "CreateBooking", {}, ["BookingId"]),
        ("DELETE", "/api/v1/invoices/{id}", "DeleteInvoice", {}, []),
        ("GET", "/api/v1/drivers/{id}/fuel-consumption", "GetDriverFuelConsumption", {}, ["FuelLevel"]),
    ]

    print("\n{:<35} | {:<20} | {:<40}".format("Operation", "OLD (generic)", "NEW (enhanced)"))
    print("-" * 100)

    improvements = 0
    for method, path, op_id, params, output_keys in test_cases:
        old = old_generate_purpose(method)
        new = generate_purpose(method, params, output_keys, path, op_id)

        old_display = old[:18] + ".." if len(old) > 20 else old
        new_display = new[:38] + ".." if len(new) > 40 else new

        is_better = len(new) > len(old) and new != old
        if is_better:
            improvements += 1

        print(f"{op_id:<35} | {old_display:<20} | {new_display:<40}")

    print(f"\nImproved: {improvements}/{len(test_cases)} operations")


def test_worst_case_scenarios():
    """Test tools with NO description - the main problem we're solving."""
    print("\n" + "="*60)
    print("TEST: Worst Case - Tools With NO Description")
    print("="*60)

    # These are the tools that only have path and operationId, no description
    test_cases = [
        ("GET", "/api/v1/fleet/{fleetId}/location-restrictions", "GetFleetLocationRestrictions", {}, []),
        ("GET", "/api/v1/vehicles/{vehicleId}/damages", "GetVehicleDamageHistory", {}, []),
        ("POST", "/api/v1/maintenance/requests", "CreateMaintenanceRequest", {}, []),
        ("GET", "/api/v1/zones/{zoneId}/pricing", "GetZonePricing", {}, []),
        ("DELETE", "/api/v1/bookings/{bookingId}/extras", "RemoveBookingExtras", {}, []),
        ("PUT", "/api/v1/drivers/{driverId}/license", "UpdateDriverLicense", {}, []),
        ("GET", "/api/v1/telematics/{deviceId}/status", "GetTelematicsStatus", {}, []),
        ("GET", "/api/v1/insurance/{policyId}/claims", "GetInsuranceClaims", {}, []),
    ]

    passed = 0
    failed = 0

    for method, path, op_id, params, output_keys in test_cases:
        purpose = generate_purpose(method, params, output_keys, path, op_id)

        # Check if it's NOT just generic
        generic_responses = [
            "Dohvaca podatke",
            "Kreira novi zapis",
            "Azurira postojece podatke",
            "Brise zapis"
        ]

        is_generic = purpose in generic_responses
        has_context = " za " in purpose  # Has entity context

        if not is_generic and has_context:
            print(f"  [OK] {op_id}")
            print(f"       -> \"{purpose}\"")
            passed += 1
        else:
            print(f"  [FAIL] {op_id}")
            print(f"       Too generic: \"{purpose}\"")
            failed += 1

    print(f"\nResult: {passed}/{passed+failed} passed")
    print("  (Tools with NO description should still get meaningful text)")
    return passed, failed


def test_synonym_extraction():
    """Test that synonyms are correctly extracted for embedding text."""
    print("\n" + "="*60)
    print("TEST: Synonym Extraction")
    print("="*60)

    # Croatian synonyms map (matching the one in embedding_engine.py)
    # Uses root form to match both nominative and genitive (vozilo, vozila)
    CROATIAN_SYNONYMS = {
        "vozil": ["auto", "automobil", "kola", "car"],  # Matches vozilo, vozila
        "osob": ["covjek", "korisnik", "user"],  # Matches osoba, osobe
        "vozac": ["driver", "sofer"],  # Matches vozac, vozaca
        "rezervacij": ["booking", "najam", "iznajmljivanje", "rent"],  # Matches rezervacija, rezervacije
        "lokacij": ["mjesto", "adresa", "pozicija", "location"],
        "kilometraz": ["km", "kilometri", "prijedeno", "mileage"],  # Matches kilometraza, kilometraze
        "goriv": ["benzin", "nafta", "dizel", "fuel", "tank"],
        "racun": ["faktura", "invoice", "naplata"],
        "odrzavanj": ["servis", "service", "popravak", "maintenance"],
    }

    def get_synonyms_for_purpose(purpose: str) -> list:
        if not purpose:
            return []
        synonyms = []
        purpose_lower = purpose.lower()
        for entity_root, syn_list in CROATIAN_SYNONYMS.items():
            # Use root form for matching (works with both nominative and genitive)
            if entity_root.lower() in purpose_lower:
                for syn in syn_list:
                    if syn.lower() not in purpose_lower and syn not in synonyms:
                        synonyms.append(syn)
        return synonyms[:8]

    test_cases = [
        ("Dohvaca podatke za vozila", ["auto", "automobil", "kola", "car"]),
        ("Kreira rezervaciju", ["booking", "najam", "iznajmljivanje", "rent"]),
        ("Dohvaca kilometrazu za vozila", ["km", "kilometri"]),  # Should have both
        ("Dohvaca podatke za vozaca, putovanja", ["driver", "sofer"]),
        ("Azurira podatke za osobe", ["covjek", "korisnik", "user"]),
    ]

    passed = 0
    failed = 0

    for purpose, expected_synonyms in test_cases:
        synonyms = get_synonyms_for_purpose(purpose)

        # Check if at least some expected synonyms are found
        found_any = any(exp in synonyms for exp in expected_synonyms)

        if found_any:
            print(f"  [OK] Purpose: \"{purpose[:40]}...\"")
            print(f"       Synonyms: {synonyms[:5]}")
            passed += 1
        else:
            print(f"  [FAIL] Purpose: \"{purpose}\"")
            print(f"       Expected some of: {expected_synonyms}")
            print(f"       Got: {synonyms}")
            failed += 1

    print(f"\nResult: {passed}/{passed+failed} passed")
    return passed, failed


def main():
    print("="*60)
    print("EMBEDDING QUALITY IMPROVEMENT TEST")
    print("="*60)

    total_passed = 0
    total_failed = 0

    p, f = test_path_entity_extraction()
    total_passed += p
    total_failed += f

    p, f = test_operation_id_parsing()
    total_passed += p
    total_failed += f

    p, f = test_purpose_generation()
    total_passed += p
    total_failed += f

    p, f = test_worst_case_scenarios()
    total_passed += p
    total_failed += f

    p, f = test_synonym_extraction()
    total_passed += p
    total_failed += f

    test_comparison_old_vs_new()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total = total_passed + total_failed
    accuracy = (total_passed / total * 100) if total > 0 else 0

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Accuracy: {accuracy:.1f}%")

    if accuracy >= 90:
        print("\n>>> EXCELLENT: Enhancement is working well!")
        return 0
    elif accuracy >= 70:
        print("\n[WARN] GOOD: Some improvements needed")
        return 1
    else:
        print("\n[FAIL] POOR: Significant issues")
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())

"""
Generate Training Data for Intent Classification ML Model.

Extracts patterns from existing codebase and generates training examples.

Output format: JSONL with structure:
{
    "text": "user query",
    "intent": "INTENT_NAME",
    "action": "GET|POST|PUT|DELETE",
    "tool": "tool_name",
    "entities": ["VehicleId", "date"]
}
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any

# Training data patterns - extracted and expanded from existing code
# Each entry: (intent, action, tool, patterns, example_variations)

TRAINING_PATTERNS = [
    # === MILEAGE QUERIES (GET) ===
    {
        "intent": "GET_MILEAGE",
        "action": "GET",
        "tool": "get_MasterData",
        "examples": [
            "koliko imam kilometara",
            "koliko kilometara ima moje vozilo",
            "kolika je kilometraza",
            "kolika je kilometraza vozila",
            "koja je trenutna kilometraza",
            "stanje kilometar sata",
            "stanje kilometraze",
            "moja kilometraza",
            "koliko km ima auto",
            "koliko je presao auto",
            "koliko je preslo vozilo",
            "kolko je presao",
            "pokazi mi kilometrazu",
            "daj mi kilometrazu",
            "kilometraza vozila",
            "trenutna kilometraza",
        ],
    },
    # === MILEAGE INPUT (POST) ===
    {
        "intent": "INPUT_MILEAGE",
        "action": "POST",
        "tool": "post_AddMileage",
        "examples": [
            "unesi kilometrazu",
            "unesi km",
            "upisi kilometrazu",
            "upisi km",
            "unos kilometraze",
            "prijavi kilometrazu",
            "nova kilometraza",
            "azuriraj kilometrazu",
            "hocu unijeti kilometrazu",
            "zelim unijeti km",
            "trebam unijeti kilometrazu",
            "dodaj kilometrazu",
            "dodaj km",
            "stavi kilometrazu",
            "unesi 15000",
            "unesi 45000 km",
            "kilometraza je 12500",
        ],
    },
    # === BOOKING (POST) ===
    {
        "intent": "BOOK_VEHICLE",
        "action": "POST",
        "tool": "get_AvailableVehicles",
        "examples": [
            "rezerviraj vozilo",
            "zelim rezervirati auto",
            "hocu rezervirati vozilo",
            "trebam vozilo",
            "trebam auto",
            "trebam kola",
            "treba mi vozilo",
            "treba mi auto",
            "daj mi auto",
            "daj mi vozilo",
            "ima li slobodnih vozila",
            "ima li slobodnih auta",
            "slobodna vozila",
            "slobodna auta",
            "zauzmi vozilo",
            "zakupi auto",
            "zelim auto",
            "zelim vozilo",
            "rezervacija vozila",
            "rezerviram auto za sutra",
            "trebam auto za ponedjeljak",
            "trebam vozilo od 8 do 17",
            "rezerviraj za sutra",
        ],
    },
    # === MY BOOKINGS (GET) ===
    {
        "intent": "GET_MY_BOOKINGS",
        "action": "GET",
        "tool": "get_VehicleCalendar",
        "examples": [
            "moje rezervacije",
            "moje bookinge",
            "moji bookingi",
            "kada imam rezervaciju",
            "kad imam auto",
            "kad imam vozilo",
            "pokazi rezervacije",
            "pokazi moje bookinge",
            "prikazi rezervacije",
            "sve rezervacije",
            "imam li rezervaciju",
            "imam li booking",
            "pregled rezervacija",
            "lista mojih rezervacija",
        ],
    },
    # === CANCEL BOOKING (DELETE) ===
    {
        "intent": "CANCEL_RESERVATION",
        "action": "DELETE",
        "tool": "delete_VehicleCalendar_id",
        "examples": [
            "otkazi rezervaciju",
            "otkazi booking",
            "cancel rezervaciju",
            "ponisti rezervaciju",
            "obrisi rezervaciju",
            "ukloni rezervaciju",
            "storniraj rezervaciju",
            "obrisi booking",
            "ne trebam vise auto",
            "ne trebam vise vozilo",
            "odustani od rezervacije",
        ],
    },
    # === REPORT DAMAGE (POST) ===
    {
        "intent": "REPORT_DAMAGE",
        "action": "POST",
        "tool": "post_AddCase",
        "examples": [
            "prijavi kvar",
            "prijava kvara",
            "prijavi stetu",
            "prijava stete",
            "nova steta",
            "ostecenje",
            "nesto ne radi",
            "problem s vozilom",
            "problem s autom",
            "imam problem s autom",
            "imam kvar",
            "imam stetu",
            "dogodila se nesreca",
            "nesreca",
            "sudar",
            "udario sam",
            "udarila sam",
            "ogrebao sam auto",
            "ogrebala sam auto",
            "ostetio sam vozilo",
            "ostetila sam vozilo",
            "motor ne radi",
            "guma je pukla",
            "staklo je razbijeno",
        ],
    },
    # === GET CASES (GET) ===
    {
        "intent": "GET_CASES",
        "action": "GET",
        "tool": "get_Cases",
        "examples": [
            "prijavljene stete",
            "popis steta",
            "lista steta",
            "pregled steta",
            "povijest steta",
            "pokazi stete",
            "prikazi stete",
            "svi slucajevi",
            "lista slucajeva",
            "moji slucajevi",
            "otvoreni slucajevi",
        ],
    },
    # === VEHICLE INFO (GET) ===
    {
        "intent": "GET_VEHICLE_INFO",
        "action": "GET",
        "tool": "get_MasterData",
        "examples": [
            "podaci o vozilu",
            "informacije o vozilu",
            "moje vozilo",
            "moja vozila",
            "koje vozilo imam",
            "koji auto imam",
            "koje auto imam",
            "daj info o autu",
            "info o autu",
            "info o vozilu",
            "pokazi mi vozila",
            "pokazi mi auto",
            "detalji o vozilu",
            "sto jos znas o vozilu",
            "sto sve znas",
            "svi podaci",
            "sve o vozilu",
            "registracija auta",
            "registracija vozila",
        ],
    },
    # === REGISTRATION EXPIRY (GET) ===
    {
        "intent": "GET_REGISTRATION_EXPIRY",
        "action": "GET",
        "tool": "get_MasterData",
        "examples": [
            "registracija istice",
            "kada istice registracija",
            "istjece registracija",
            "istek registracije",
            "do kad vrijedi registracija",
            "vrijedi registracija",
            "kada je registracija",
            "datum registracije",
        ],
    },
    # === LICENCE PLATE (GET) ===
    {
        "intent": "GET_PLATE",
        "action": "GET",
        "tool": "get_MasterData",
        "examples": [
            "koje su tablice",
            "tablica vozila",
            "registarska oznaka",
            "registracijski broj",
            "koja je tablica",
            "koje tablice imam",
            "broj tablica",
        ],
    },
    # === SERVICE/MAINTENANCE (GET) ===
    {
        "intent": "GET_SERVICE_MILEAGE",
        "action": "GET",
        "tool": "get_MasterData",
        "examples": [
            "servis",
            "koliko do servisa",
            "kad je servis",
            "kada je servis",
            "sljedeci servis",
            "trebam na servis",
            "preostalo do servisa",
            "do servisa",
            "odrzavanje",
            "zadnji servis",
            "prosli servis",
            "povijest servisa",
            "koliko km do servisa",
        ],
    },
    # === TRIPS (GET) ===
    {
        "intent": "GET_TRIPS",
        "action": "GET",
        "tool": "get_Trips",
        "examples": [
            "putni nalog",
            "putni nalozi",
            "putovanje",
            "putovanja",
            "moja putovanja",
            "povijest voznji",
            "moje voznje",
            "tripovi",
            "moji tripovi",
            "popis putovanja",
        ],
    },
    # === DELETE TRIP (DELETE) ===
    {
        "intent": "DELETE_TRIP",
        "action": "DELETE",
        "tool": "delete_Trips_id",
        "examples": [
            "obrisi putovanje",
            "obrisi trip",
            "obrisi voznju",
            "ukloni putovanje",
            "izbrisi putovanje",
            "otkazi putovanje",
        ],
    },
    # === PERSONAL INFO (GET) ===
    {
        "intent": "GET_PERSON_INFO",
        "action": "GET",
        "tool": "get_PersonData_personIdOrEmail",
        "examples": [
            "kako se zovem",
            "moje ime",
            "tko sam ja",
            "moji podaci",
            "moj profil",
            "osobni podaci",
            "moj email",
            "moj telefon",
            "moja tvrtka",
            "u kojoj sam firmi",
            "moja firma",
            "koji je moj email",
        ],
    },
    # === LEASING INFO (GET) ===
    {
        "intent": "GET_LEASING",
        "action": "GET",
        "tool": "get_MasterData",
        "examples": [
            "lizing",
            "leasing",
            "koja je lizing kuca",
            "lizing provider",
            "lizing tvrtka",
            "leasing kompanija",
            "tko je lizing",
        ],
    },
    # === CONTEXT QUERIES (GET) ===
    {
        "intent": "GET_TENANT_ID",
        "action": "GET",
        "tool": None,
        "examples": [
            "koji je moj tenant",
            "sto je moj tenant",
            "koji je tenant id",
            "sto je tenant",
            "moj tenant",
            "tenant id",
        ],
    },
    {
        "intent": "GET_PERSON_ID",
        "action": "GET",
        "tool": None,
        "examples": [
            "koji je moj person id",
            "sto je moj person id",
            "moj person id",
            "koji je person",
            "person id",
        ],
    },
    {
        "intent": "GET_PHONE",
        "action": "GET",
        "tool": None,
        "examples": [
            "koji je moj broj telefona",
            "sto je moj broj",
            "moj telefon",
            "koji je telefon",
            "broj telefona",
        ],
    },
    # === GREETINGS ===
    {
        "intent": "GREETING",
        "action": "NONE",
        "tool": None,
        "examples": [
            "bok",
            "cao",
            "pozdrav",
            "zdravo",
            "hej",
            "hi",
            "hello",
            "dobro jutro",
            "dobar dan",
            "dobra vecer",
        ],
    },
    # === THANKS ===
    {
        "intent": "THANKS",
        "action": "NONE",
        "tool": None,
        "examples": [
            "hvala",
            "hvala puno",
            "hvala lijepa",
            "zahvaljujem",
            "thanks",
            "fala",
            "hvala ti",
            "hvala vam",
        ],
    },
    # === HELP ===
    {
        "intent": "HELP",
        "action": "NONE",
        "tool": None,
        "examples": [
            "pomoc",
            "help",
            "sto mozes",
            "kako koristiti",
            "sto znas",
            "kako ovo radi",
            "upute",
            "sto mogu pitati",
        ],
    },
    # === VEHICLE COUNT (GET) - AGGREGATIONS ===
    {
        "intent": "GET_VEHICLE_COUNT",
        "action": "GET",
        "tool": "get_Vehicles_Agg",
        "examples": [
            "koliko ima vozila",
            "koliko vozila imam",
            "broj vozila",
            "ukupno vozila",
            "koliko auta ima",
            "koliko auta",
        ],
    },
    # === COMPANY QUERY (common user question) ===
    {
        "intent": "GET_VEHICLE_COMPANY",
        "action": "GET",
        "tool": "get_MasterData",
        "examples": [
            "kojoj kompaniji pripada moje vozilo",
            "koja je kompanija",
            "koja je firma",
            "cije je vozilo",
            "u cijoj je firmi vozilo",
            "kompanija vozila",
            "vlasnik vozila",
            "koja tvrtka",
        ],
    },
    # === EQUIPMENT (common user question) ===
    {
        "intent": "GET_VEHICLE_EQUIPMENT",
        "action": "GET",
        "tool": "get_MasterData",
        "examples": [
            "koja oprema",
            "oprema vozila",
            "sto ima u autu",
            "sto je u vozilu",
            "dodatna oprema",
            "ima li klimu",
            "ima li navigaciju",
            "popis opreme",
        ],
    },
    # === DOCUMENTS (common user question) ===
    {
        "intent": "GET_VEHICLE_DOCUMENTS",
        "action": "GET",
        "tool": "get_Vehicles_id_documents",
        "examples": [
            "dokumenti vozila",
            "dokumentacija",
            "papiri od auta",
            "prometna dozvola",
            "pokazi dokumente",
            "koji dokumenti",
        ],
    },
]


def generate_variations(text: str) -> List[str]:
    """Generate variations of a text with common transformations."""
    variations = [text]

    # Add with question marks
    if not text.endswith("?"):
        variations.append(text + "?")

    # Add politeness prefix
    polite_prefixes = ["molim te ", "moze li ", "mozete li ", "jel mozes "]
    for prefix in polite_prefixes:
        if not text.startswith(prefix):
            variations.append(prefix + text)

    # Add urgency suffix
    urgency_suffixes = [" odmah", " hitno", " brzo"]
    for suffix in urgency_suffixes:
        if not text.endswith(suffix):
            variations.append(text + suffix)

    return variations


def augment_with_typos(text: str) -> List[str]:
    """Generate common typo variations (Croatian-specific)."""
    augmented = []

    # Common Croatian typo substitutions
    typo_map = {
        "š": "s",
        "č": "c",
        "ć": "c",
        "ž": "z",
        "đ": "dj",
    }

    # Generate version without diacritics
    no_diacritics = text
    for orig, repl in typo_map.items():
        no_diacritics = no_diacritics.replace(orig, repl)

    if no_diacritics != text:
        augmented.append(no_diacritics)

    return augmented


def generate_training_data() -> List[Dict[str, Any]]:
    """Generate full training dataset."""
    training_data = []

    for pattern_group in TRAINING_PATTERNS:
        intent = pattern_group["intent"]
        action = pattern_group["action"]
        tool = pattern_group["tool"]
        examples = pattern_group["examples"]

        for example in examples:
            # Base example
            training_data.append({
                "text": example,
                "intent": intent,
                "action": action,
                "tool": tool,
            })

            # Variations
            for variation in generate_variations(example):
                if variation != example:
                    training_data.append({
                        "text": variation,
                        "intent": intent,
                        "action": action,
                        "tool": tool,
                    })

            # Typo variations
            for typo_var in augment_with_typos(example):
                training_data.append({
                    "text": typo_var,
                    "intent": intent,
                    "action": action,
                    "tool": tool,
                })

    return training_data


def save_training_data(data: List[Dict], output_path: Path):
    """Save training data in JSONL format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def create_train_test_split(data: List[Dict], test_ratio: float = 0.2):
    """Split data into train and test sets."""
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]


def print_stats(data: List[Dict]):
    """Print dataset statistics."""
    intent_counts = {}
    action_counts = {}
    tool_counts = {}

    for item in data:
        intent = item["intent"]
        action = item["action"]
        tool = item["tool"] or "NONE"

        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print("\n=== Intent Distribution ===")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        print(f"  {intent}: {count}")

    print("\n=== Action Distribution ===")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count}")

    print(f"\n=== Total ===")
    print(f"  Total examples: {len(data)}")
    print(f"  Unique intents: {len(intent_counts)}")
    print(f"  Unique tools: {len(tool_counts)}")


if __name__ == "__main__":
    # Generate data
    print("Generating training data...")
    data = generate_training_data()

    # Print stats
    print_stats(data)

    # Split and save
    train_data, test_data = create_train_test_split(data)

    output_dir = Path(__file__).parent.parent / "data" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "intent_train.jsonl"
    test_path = output_dir / "intent_test.jsonl"
    full_path = output_dir / "intent_full.jsonl"

    save_training_data(train_data, train_path)
    save_training_data(test_data, test_path)
    save_training_data(data, full_path)

    print(f"\n=== Saved ===")
    print(f"  Train: {train_path} ({len(train_data)} examples)")
    print(f"  Test: {test_path} ({len(test_data)} examples)")
    print(f"  Full: {full_path} ({len(data)} examples)")

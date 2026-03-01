"""
Parameter Prompts

Centralized prompts for missing parameters.

When a required parameter is missing, these prompts tell the user
what information is needed in a friendly Croatian message.

Usage:
    from services.context import get_missing_param_prompt

    prompt = get_missing_param_prompt("VehicleId")
    # Returns: "Koje vozilo? (unesite tablice npr. ZG-1234-AB)"
"""

from typing import Optional, Dict

# ---
# PARAMETER PROMPTS - Croatian user-friendly messages
# ---

PARAM_PROMPTS: Dict[str, str] = {
    # === TIME/DATE PARAMETERS ===
    "from": "Od kada vam treba? (npr. 'sutra u 9:00' ili '15.01.2025 09:00')",
    "FromTime": "Od kada vam treba? (npr. 'sutra u 9:00')",
    "to": "Do kada? (npr. 'petak u 17:00')",
    "ToTime": "Do kada? (npr. 'petak u 17:00')",
    "date": "Za koji datum? (npr. 'sutra' ili '15.01.2025')",
    "Date": "Za koji datum?",
    "duration": "Koliko dugo? (npr. '3 dana' ili '1 tjedan')",
    "Duration": "Koliko dugo?",

    # === VEHICLE PARAMETERS ===
    "VehicleId": "Koje vozilo? (unesite tablice npr. ZG-1234-AB)",
    "vehicleId": "Koje vozilo? (unesite tablice npr. ZG-1234-AB)",
    "vehicle_id": "Koje vozilo? (unesite tablice npr. ZG-1234-AB)",
    "LicencePlate": "Koje su tablice vozila? (npr. ZG-1234-AB)",
    "RegistrationNumber": "Registracijski broj vozila?",
    "Plate": "Tablice vozila?",

    # === PERSON PARAMETERS ===
    "PersonId": "Za koju osobu? (ime i prezime ili email)",
    "personId": "Za koju osobu? (ime i prezime ili email)",
    "person_id": "Za koju osobu? (ime i prezime ili email)",
    "UserId": "Koji korisnik?",
    "DriverId": "Koji vozač?",
    "User": "Ime i prezime korisnika?",
    "Driver": "Ime vozača?",

    # === MILEAGE PARAMETERS ===
    "Mileage": "Koliko kilometara? (npr. '45000')",
    "Value": "Koliko kilometara? (broj bez jedinice)",
    "mileage": "Koliko kilometara?",
    "LastMileage": "Trenutna kilometraža?",

    # === CASE/DAMAGE PARAMETERS ===
    "subject": "Koji je predmet prijave?",
    "Subject": "Naslov prijave?",
    "Message": "Opišite problem ili poruku:",
    "Description": "Opis?",
    "CaseTypeId": "Koja vrsta prijave? (npr. 'šteta', 'kvar', 'servis')",

    # === LOCATION PARAMETERS ===
    "LocationId": "Koja lokacija?",
    "locationId": "Koja lokacija?",
    "location_id": "Koja lokacija?",
    "Address": "Koja adresa?",

    # === BOOKING PARAMETERS ===
    "BookingId": "Koja rezervacija? (ID rezervacije)",
    "bookingId": "Koja rezervacija?",
    "booking_id": "Koja rezervacija?",
    "ReservationId": "Koja rezervacija?",

    # === OTHER COMMON PARAMETERS ===
    "Id": "Koji ID?",
    "Filter": "Koji filter želite primijeniti?",
    "Rows": "Koliko rezultata želite?",
    "Comment": "Koji komentar?",
    "Note": "Koja napomena?",
    "Reason": "Koji razlog?",
}


# ---
# PARAMETER CATEGORIES
# ---

CATEGORY_PROMPTS: Dict[str, str] = {
    "time": "Trebam informacije o vremenu.",
    "vehicle": "Trebam informacije o vozilu.",
    "person": "Trebam informacije o osobi.",
    "location": "Trebam informacije o lokaciji.",
    "booking": "Trebam informacije o rezervaciji.",
    "case": "Trebam informacije o prijavi.",
}


def get_param_category(param_name: str) -> Optional[str]:
    """Determine parameter category from name."""
    param_lower = param_name.lower()

    if any(x in param_lower for x in ["time", "date", "from", "to", "duration"]):
        return "time"
    if any(x in param_lower for x in ["vehicle", "car", "plate", "mileage"]):
        return "vehicle"
    if any(x in param_lower for x in ["person", "user", "driver"]):
        return "person"
    if any(x in param_lower for x in ["location", "address", "site"]):
        return "location"
    if any(x in param_lower for x in ["booking", "reservation"]):
        return "booking"
    if any(x in param_lower for x in ["case", "subject", "message", "damage"]):
        return "case"

    return None


def get_missing_param_prompt(param_name: str) -> str:
    """
    Get user-friendly prompt for a missing parameter.

    Args:
        param_name: Name of the missing parameter

    Returns:
        Croatian prompt string asking user for the parameter
    """
    # Try exact match first
    if param_name in PARAM_PROMPTS:
        return PARAM_PROMPTS[param_name]

    # Try case-insensitive match
    param_lower = param_name.lower()
    for key, prompt in PARAM_PROMPTS.items():
        if key.lower() == param_lower:
            return prompt

    # Try partial match (e.g., "SomeVehicleId" should match "VehicleId")
    for key, prompt in PARAM_PROMPTS.items():
        if key.lower() in param_lower or param_lower in key.lower():
            return prompt

    # Fallback based on category
    category = get_param_category(param_name)
    if category:
        return CATEGORY_PROMPTS[category]

    # Generic fallback
    return f"Trebam još informaciju: {param_name}"


def get_multiple_missing_prompts(param_names: list) -> str:
    """
    Build combined prompt for multiple missing parameters.

    Args:
        param_names: List of missing parameter names

    Returns:
        Combined Croatian prompt string
    """
    if not param_names:
        return ""

    if len(param_names) == 1:
        return get_missing_param_prompt(param_names[0])

    lines = ["Za nastavak trebam još nekoliko informacija:"]
    for param in param_names[:5]:  # Limit to 5 to avoid overwhelming user
        prompt = get_missing_param_prompt(param)
        # Extract just the question part (remove examples)
        question = prompt.split("(")[0].strip()
        lines.append(f"• {question}")

    if len(param_names) > 5:
        lines.append(f"• ... i još {len(param_names) - 5} informacija")

    return "\n".join(lines)

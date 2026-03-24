"""
Unified Tool Routing Configuration.

SINGLE SOURCE OF TRUTH for all tool/intent mappings.
Used by: QueryRouter, UnifiedRouter, UnifiedSearch.

Adding a new tool? Add it here ONLY. All three routing layers
will automatically pick it up.
"""

import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# INTENT_CONFIG: Maps ML intents to tools, flow types, templates
# Used by: services/query_router.py
# ──────────────────────────────────────────────────────────────

INTENT_CONFIG = {
    # ── Vehicle Information (READ) ──────────────────────────────
    "GET_MILEAGE": {
        "tool": "get_MasterData",
        "extract_fields": ["LastMileage", "Mileage", "CurrentMileage"],
        "response_template": "**Kilometraza:** {value} km",
        "flow_type": "simple",
    },
    "GET_VEHICLE_INFO": {
        "tool": "get_MasterData",
        "extract_fields": ["FullVehicleName", "LicencePlate", "LastMileage", "Manufacturer", "Model"],
        "response_template": None,
        "flow_type": "simple",
    },
    "GET_REGISTRATION_EXPIRY": {
        "tool": "get_MasterData",
        "extract_fields": ["RegistrationExpirationDate", "ExpirationDate"],
        "response_template": "**Registracija istjece:** {value}",
        "flow_type": "simple",
    },
    "GET_PLATE": {
        "tool": "get_MasterData",
        "extract_fields": ["LicencePlate", "RegistrationNumber"],
        "response_template": "**Tablice:** {value}",
        "flow_type": "simple",
    },
    "GET_LEASING": {
        "tool": "get_MasterData",
        "extract_fields": ["ProviderName", "SupplierName"],
        "response_template": "**Lizing kuca:** {value}",
        "flow_type": "simple",
    },
    "GET_SERVICE_MILEAGE": {
        "tool": "get_MasterData",
        "extract_fields": ["ServiceMileage", "NextServiceMileage"],
        "response_template": "**Do servisa:** {value} km",
        "flow_type": "simple",
    },
    "GET_VEHICLE_COMPANY": {
        "tool": "get_MasterData",
        "extract_fields": ["Company", "CompanyName", "Organization"],
        "response_template": "**Tvrtka:** {value}",
        "flow_type": "simple",
    },
    "GET_VEHICLE_EQUIPMENT": {
        "tool": "get_MasterData",
        "extract_fields": ["Equipment", "Equipments"],
        "response_template": None,
        "flow_type": "simple",
    },
    "GET_VEHICLE_DOCUMENTS": {
        "tool": "get_Vehicles_id_documents",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "list",
    },
    "GET_VEHICLE_COUNT": {
        "tool": "get_Vehicles_Agg",
        "extract_fields": ["Count", "TotalCount"],
        "response_template": "**Broj vozila:** {value}",
        "flow_type": "simple",
    },

    # ── Reservations ────────────────────────────────────────────
    "BOOK_VEHICLE": {
        "tool": "get_AvailableVehicles",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "booking",
    },
    "GET_MY_BOOKINGS": {
        "tool": "get_VehicleCalendar",
        "extract_fields": ["FromTime", "ToTime", "VehicleName"],
        "response_template": None,
        "flow_type": "list",
    },
    "CANCEL_RESERVATION": {
        "tool": "delete_VehicleCalendar_id",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "delete_booking",
    },
    # GET_AVAILABLE_VEHICLES merged into BOOK_VEHICLE (same tool + flow_type, split caused 28% of ML errors)

    # ── Mileage ─────────────────────────────────────────────────
    "INPUT_MILEAGE": {
        "tool": "post_AddMileage",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "mileage_input",
    },

    # ── Cases / Damage ──────────────────────────────────────────
    "REPORT_DAMAGE": {
        "tool": "post_AddCase",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "case_creation",
    },
    "GET_CASES": {
        "tool": "get_Cases",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "list",
    },
    "DELETE_CASE": {
        "tool": "delete_Cases_id",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "delete_case",
    },

    # ── Trips ───────────────────────────────────────────────────
    "GET_TRIPS": {
        "tool": "get_Trips",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "list",
    },
    "DELETE_TRIP": {
        "tool": "delete_Trips_id",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "delete_trip",
    },

    # ── Expenses ────────────────────────────────────────────────
    "GET_EXPENSES": {
        "tool": "get_Expenses",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "list",
    },

    # ── Vehicles List ───────────────────────────────────────────
    "GET_VEHICLES": {
        "tool": "get_Vehicles",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "list",
    },

    # ── Person Info ─────────────────────────────────────────────
    "GET_PERSON_INFO": {
        "tool": "get_PersonData_personIdOrEmail",
        "extract_fields": ["FirstName", "LastName", "DisplayName", "Email"],
        "response_template": None,
        "flow_type": "simple",
    },
    "GET_PERSON_ID": {
        "tool": None,
        "extract_fields": [],
        "response_template": "**Person ID:** {person_id}",
        "flow_type": "direct_response",
    },
    "GET_PHONE": {
        "tool": None,
        "extract_fields": [],
        "response_template": "**Telefon:** {phone}",
        "flow_type": "direct_response",
    },
    "GET_TENANT_ID": {
        "tool": None,
        "extract_fields": [],
        "response_template": "**Tenant ID:** {tenant_id}",
        "flow_type": "direct_response",
    },

    # ── Social / Static ────────────────────────────────────────
    "GREETING": {
        "tool": None,
        "extract_fields": [],
        "response_template": "Pozdrav! Kako vam mogu pomoci?",
        "flow_type": "direct_response",
    },
    "THANKS": {
        "tool": None,
        "extract_fields": [],
        "response_template": "Nema na cemu! Slobodno pitajte ako trebate jos nesto.",
        "flow_type": "direct_response",
    },
    "HELP": {
        "tool": None,
        "extract_fields": [],
        "response_template": (
            "Mogu vam pomoci s:\n"
            "* **Kilometraza** - provjera ili unos km\n"
            "* **Rezervacije** - rezervacija vozila\n"
            "* **Podaci o vozilu** - registracija, lizing\n"
            "* **Prijava kvara** - kreiranje slucaja\n\n"
            "Sto vas zanima?"
        ),
        "flow_type": "direct_response",
    },
}


# ──────────────────────────────────────────────────────────────
# PRIMARY_TOOLS: Tool descriptions for LLM routing context
# Used by: services/unified_router.py
# ──────────────────────────────────────────────────────────────

PRIMARY_TOOLS = {
    # Vehicle Info (READ)
    "get_MasterData": "Dohvati podatke o vozilu (registracija, kilometraža, servis)",
    "get_Vehicles_id": "Dohvati detalje specifičnog vozila",

    # Person Info (READ)
    "get_PersonData_personIdOrEmail": "Dohvati podatke o korisniku (ime, prezime, email, telefon)",

    # Availability & Booking
    "get_AvailableVehicles": "Provjeri dostupna/slobodna vozila za period",
    "get_VehicleCalendar": "Dohvati moje rezervacije",
    "post_VehicleCalendar": "Kreiraj novu rezervaciju vozila",
    "delete_VehicleCalendar_id": "Obriši/otkaži rezervaciju",

    # Mileage
    "get_LatestMileageReports": "Dohvati zadnju kilometražu",
    "get_MileageReports": "Dohvati izvještaje o kilometraži",
    "post_AddMileage": "Unesi/upiši novu kilometražu",

    # Case/Damage
    "post_AddCase": "Prijavi štetu, kvar, problem, nesreću",
    "get_Cases": "Dohvati prijavljene slučajeve",

    # Expenses
    "get_Expenses": "Dohvati troškove",
    "get_ExpenseGroups": "Dohvati grupe troškova",

    # Trips
    "get_Trips": "Dohvati putovanja/tripove",

    # Dashboard
    "get_DashboardItems": "Dohvati dashboard podatke",

    # Lists
    "get_Companies": "Dohvati popis tvrtki/kompanija",
    "get_Persons": "Dohvati popis osoba/korisnika",
    "get_Vehicles": "Dohvati popis svih vozila u floti",
    "get_Vehicles_Agg": "Dohvati ukupan broj vozila",
    "get_Vehicles_id_documents": "Dohvati dokumente vozila",
    "delete_Cases_id": "Obriši/zatvori prijavljeni slučaj",
    "delete_Trips_id": "Obriši putovanje/trip",
}


# ──────────────────────────────────────────────────────────────
# PRIMARY_ACTION_TOOLS: Keyword-based boost for FAISS search
# Used by: services/unified_search.py
# Keys are LOWERCASE (FAISS tool IDs are lowercased)
# ──────────────────────────────────────────────────────────────

PRIMARY_ACTION_TOOLS = {
    "get_masterdata": {
        "keywords": [
            "kilometr", "koliko km", "koliko imam", "stanje km", "moja km",
            "registracij", "tablic", "lizing", "leasing",
            "podaci o vozil", "podaci vozil", "informacij", "moje vozilo",
            "koji auto", "servis", "do servisa",
        ],
        "boost": 2.0,
    },
    "post_addmileage": {
        "keywords": [
            "unesi km", "upiši km", "dodaj km", "dodaj kilometr",
            "unesi kilometr", "upiši kilometr", "nova km", "prijeđen",
            "prijavi km", "prijavi kilometr",
        ],
        "boost": 1.8,
    },
    "post_addcase": {
        "keywords": [
            "šteta", "steta", "prijavi kvar", "prijavi štetu",
            "udario", "ogrebao", "oštetio", "oštećenj",
            "incident", "nesreća", "ima kvar", "imam kvar",
            "nova šteta", "novi kvar",
        ],
        "boost": 2.0,
    },
    "post_vehiclecalendar": {
        "keywords": [
            "rezerviraj", "rezervacija", "nova rezervacija",
            "booking", "zauzmi", "trebam auto", "trebam vozilo",
            "želim rezerv", "hoću rezerv", "napravi rezerv",
        ],
        "boost": 1.8,
    },
    "get_vehiclecalendar": {
        "keywords": ["moje rezervacij", "moji booking", "kalendar vozil"],
        "boost": 1.5,
    },
    "get_availablevehicles": {
        "keywords": ["slobodn", "dostupn", "raspoloživ", "available", "ima li slobodn"],
        "boost": 1.5,
    },
    "get_trips": {
        "keywords": ["putovanj", "trip", "vožnj", "putni nalog"],
        "boost": 1.5,
    },
    "get_expenses": {
        "keywords": ["troškov", "troška", "izdatak", "račun", "potrošio"],
        "boost": 1.5,
    },
    "get_persondata_personidoremail": {
        "keywords": ["moji podaci", "moj profil", "moje ime", "tko sam"],
        "boost": 1.5,
    },
    "get_cases": {
        "keywords": ["slučaj", "slucaj", "štete", "stete", "prijave", "kvarovi", "moji slučajevi"],
        "boost": 1.5,
    },
    "delete_vehiclecalendar_id": {
        "keywords": ["otkaži", "otkazi", "cancel", "storniraj rezerv"],
        "boost": 1.5,
    },
    "get_vehicles_agg": {
        "keywords": ["koliko vozila", "ukupno vozila", "broj vozila", "broj auta"],
        "boost": 1.5,
    },
    "get_vehicles_id_documents": {
        "keywords": ["dokument", "papir", "certifikat", "prometna"],
        "boost": 1.5,
    },
    "delete_trips_id": {
        "keywords": ["obriši trip", "obrisi trip", "obriši putovanj", "ukloni trip"],
        "boost": 1.5,
    },
    "delete_cases_id": {
        "keywords": ["obriši prijav", "obrisi prijav", "obriši slučaj", "ukloni prijav"],
        "boost": 1.5,
    },
    "get_vehicles": {
        "keywords": ["sva vozila", "popis vozila", "lista vozila", "vozila u floti"],
        "boost": 1.5,
    },
    "get_companies": {
        "keywords": ["sve kompanije", "popis kompanija", "lista kompanija", "tvrtke"],
        "boost": 1.5,
    },
    "get_persons": {
        "keywords": ["sve osobe", "popis osoba", "lista osoba", "zaposlenici", "korisnici"],
        "boost": 1.5,
    },
    "get_vehicles_id": {
        "keywords": ["detalji vozila", "specifično vozilo", "to vozilo", "taj auto"],
        "boost": 1.5,
    },
    "get_expensegroups": {
        "keywords": ["grupe troškova", "kategorije troškova", "vrste troškova"],
        "boost": 1.5,
    },
    "get_dashboarditems": {
        "keywords": ["dashboard", "pregled", "sažetak", "nadzorna ploča"],
        "boost": 1.5,
    },
    "get_latestmileagereports": {
        "keywords": ["zadnja km", "zadnja kilometr", "posljednja km", "latest km"],
        "boost": 1.5,
    },
    "get_mileagereports": {
        "keywords": ["izvještaj km", "izvještaj kilometr", "povijest km", "km izvještaj"],
        "boost": 1.5,
    },
}


# ──────────────────────────────────────────────────────────────
# FLOW_TRIGGERS: Which tools trigger which conversation flows
# Used by: services/unified_router.py
# ──────────────────────────────────────────────────────────────

FLOW_TRIGGERS = {
    "post_VehicleCalendar": "booking",
    "get_AvailableVehicles": "booking",
    "post_AddMileage": "mileage",
    "post_AddCase": "case",
}


# ──────────────────────────────────────────────────────────────
# VALIDATION: Detect config drift between dicts at import time
# ──────────────────────────────────────────────────────────────

def validate_tool_routing():
    """Validate consistency across all routing config dicts at import time."""
    primary_tools_lower = {k.lower() for k in PRIMARY_TOOLS}
    action_tools = set(PRIMARY_ACTION_TOOLS.keys())

    missing_keywords = primary_tools_lower - action_tools
    missing_descriptions = action_tools - primary_tools_lower

    if missing_keywords:
        logger.warning(f"Tools in PRIMARY_TOOLS without keyword boosts: {missing_keywords}")
    if missing_descriptions:
        logger.warning(f"Tools in PRIMARY_ACTION_TOOLS without LLM descriptions: {missing_descriptions}")

    # Validate INTENT_CONFIG tools exist in PRIMARY_TOOLS
    known_tools = primary_tools_lower | {None}
    for intent, config in INTENT_CONFIG.items():
        tool = config.get("tool")
        if tool and tool.lower() not in known_tools:
            logger.warning(f"INTENT_CONFIG[{intent}].tool='{tool}' not in PRIMARY_TOOLS")

    # Validate all flow_types have handlers
    known_flow_types = {
        "simple", "list", "direct_response",
        "booking", "mileage_input", "case_creation",
        "delete_booking", "delete_case", "delete_trip",
    }
    for intent, config in INTENT_CONFIG.items():
        ft = config.get("flow_type")
        if ft and ft not in known_flow_types:
            logger.warning(f"INTENT_CONFIG[{intent}].flow_type='{ft}' has no handler")


validate_tool_routing()

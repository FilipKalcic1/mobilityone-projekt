"""
Query Router - Deterministic routing for known query patterns.
Version: 1.0

Single responsibility: Route queries to correct tools WITHOUT LLM guessing.
For known patterns, we use RULES, not probabilities.

This guarantees correct responses for common queries.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """Result of query routing."""
    matched: bool
    tool_name: Optional[str] = None
    extract_fields: List[str] = None
    response_template: Optional[str] = None
    flow_type: Optional[str] = None  # "simple", "booking", "mileage_input", etc.
    confidence: float = 1.0
    reason: str = ""

    def __post_init__(self):
        if self.extract_fields is None:
            self.extract_fields = []


class QueryRouter:
    """
    Routes queries to tools using deterministic rules.

    For known patterns:
    - NO embedding search needed
    - NO LLM tool selection needed
    - GUARANTEED correct tool

    This is the "fast path" for common queries.
    """

    def __init__(self):
        """Initialize router with rules."""
        self.rules = self._build_rules()

    def _build_rules(self) -> List[Dict[str, Any]]:
        """Build deterministic routing rules."""
        return [
            # === CONTEXT QUERIES (tenant_id, person_id, phone) ===
            {
                "patterns": [
                    r"koji.*moj.*tenant",
                    r"[sš]to.*moj.*tenant",
                    r"koji.*je.*tenant.*id",
                    r"[sš]to.*je.*tenant",
                    r"moj.*tenant",
                ],
                "intent": "GET_TENANT_ID",
                "tool": None,
                "extract_fields": [],
                "response_template": "🏢 **Tenant ID:** {tenant_id}",
                "flow_type": "direct_response",
            },
            {
                "patterns": [
                    r"koji.*moj.*person.*id",
                    r"[sš]to.*moj.*person.*id",
                    r"moj.*person.*id",
                    r"koji.*je.*person",
                ],
                "intent": "GET_PERSON_ID",
                "tool": None,
                "extract_fields": [],
                "response_template": "👤 **Person ID:** {person_id}",
                "flow_type": "direct_response",
            },
            {
                "patterns": [
                    r"koji.*moj.*broj.*telefon",
                    r"[sš]to.*moj.*broj",
                    r"moj.*telefon",
                    r"koji.*je.*telefon",
                ],
                "intent": "GET_PHONE",
                "tool": None,
                "extract_fields": [],
                "response_template": "📱 **Telefon:** {phone}",
                "flow_type": "direct_response",
            },
            # === MILEAGE INPUT (must be BEFORE GET_MILEAGE to catch "unesi" first) ===
            {
                "patterns": [
                    r"unesi.*(km|kilometra)",
                    r"upiši.*(km|kilometra)",
                    r"unos.*(km|kilometra)",
                    r"prijavi.*(km|kilometra)",
                    r"unesite.*(km|kilometra)",
                    r"nova.*kilometra",
                    r"ažuriraj.*(km|kilometra)",
                    r"ho[cć]u.*unijeti.*(km|kilometra)",
                    r"želim.*unijeti.*(km|kilometra)",
                    r"trebam.*unijeti.*(km|kilometra)",
                    r"unijeti.*(km|kilometra)",
                    r"unesi.*\d+",  # "unesi 15000" with number
                    r"dodaj.*(km|kilometra)",     # Phase 3: "dodaj kilometražu"
                    r"dodaj.*km",                 # Phase 3: "dodaj km"
                    r"stavi.*(km|kilometra)",     # Phase 3: "stavi kilometražu"
                ],
                "intent": "INPUT_MILEAGE",
                "tool": "post_AddMileage",
                "extract_fields": [],
                "response_template": None,
                "flow_type": "mileage_input",
            },
            # === MILEAGE QUERIES ===
            {
                "patterns": [
                    r"koliko.*(km|kilometra)",
                    r"kolika.*(km|kilometra)",
                    r"koliko.*imam.*km",          # V3.0: "koliko imam kilometara"
                    r"koliko.*imam.*kilometar",   # V3.0: "koliko imam kilometara"
                    r"moja.*kilometra[zž]",       # V3.0: "moja kilometraža"
                    r"stanje.*(km|kilometra)",
                    r"stanje.*kilometar.*sat",    # V3.0: "stanje kilometar sata"
                    r"\bkm\b.*vozil",
                    r"mileage",
                    r"koja.*kilometra[zž]",
                    r"trenutna.*kilometra",
                    r"kilometra[zž]a.*vozil",
                    r"kolko.*pre[sš]",            # Phase 3: "kolko je prešo"
                    r"koliko.*pre[sš]",           # Phase 3: "koliko je prešao"
                    r"pre[sš]ao.*auto",           # Phase 3: "prešao auto"
                    r"pre[sš]lo.*vozil",          # Phase 3: "prešlo vozilo"
                ],
                "intent": "GET_MILEAGE",
                "tool": "get_MasterData",
                "extract_fields": ["LastMileage", "Mileage", "CurrentMileage"],
                "response_template": "📏 **Kilometraža:** {value} km",
                "flow_type": "simple",
            },
            # === TRIPS / PUTOVANJA ===
            {
                "patterns": [
                    r"putni.*(nalog|naloz)",     # V3.0: putni nalog/nalozi (g→z u množini)
                    r"putovanje",                # putovanje/putovanja
                    r"moj[ae]?.*putovanj",       # moja putovanja
                    r"povijest.*vo[zž]nj",       # povijest vožnji
                    r"moj[ae]?.*vo[zž]nj",       # moje vožnje
                    r"trip",                     # trip/tripovi
                ],
                "intent": "GET_TRIPS",
                "tool": "get_Trips",
                "extract_fields": [],
                "response_template": None,
                "flow_type": "list",
            },
            # === REGISTRATION EXPIRY ===
            {
                "patterns": [
                    r"registracij.*isti[cč]e",
                    r"kada.*registracij",
                    r"istje[cč]e.*registracij",
                    r"istek.*registracij",
                    r"do.*kad.*registracij",
                    r"vrijedi.*registracij",
                ],
                "intent": "GET_REGISTRATION_EXPIRY",
                "tool": "get_MasterData",
                "extract_fields": ["RegistrationExpirationDate", "ExpirationDate"],
                "response_template": "📅 **Registracija istječe:** {value}",
                "flow_type": "simple",
            },
            # === VEHICLE INFO ===
            {
                "patterns": [
                    r"podaci.*vozil",
                    r"informacij.*vozil",
                    r"vozilo.*podaci",
                    r"moje.*vozilo",
                    r"moja.*vozila",             # Phase 3: "moja vozila"
                    r"koje.*vozilo",
                    r"koji.*auto.*imam",         # Phase 3: "koji auto imam"
                    r"koje.*auto.*imam",         # Phase 3: "koje auto imam"
                    r"daj.*info.*auto",          # Phase 3: "daj info o autu"
                    r"info.*o.*auto",            # Phase 3: "info o autu"
                    r"info.*o.*vozil",           # Phase 3: "info o vozilu"
                    r"poka[zž]i.*mi.*vozil",     # Phase 3: "pokaži mi vozila"
                    r"poka[zž]i.*mi.*auto",      # Phase 3: "pokaži mi auto"
                    r"detalji.*vozil",
                    r"[sš]to.*jo[sš].*zna[sš]",  # što još znaš (about vehicle)
                    r"[sš]to.*sve.*zna",         # što sve znaš
                    r"svi.*podaci",              # svi podaci
                    r"sve.*o.*vozil",            # sve o vozilu
                    r"registracij.*auto",        # Phase 3: "registracija auta"
                    r"registracij.*vozil",       # Phase 3: "registracija vozila"
                ],
                "intent": "GET_VEHICLE_INFO",
                "tool": "get_MasterData",
                "extract_fields": ["FullVehicleName", "LicencePlate", "LastMileage", "Manufacturer", "Model", "ProductionYear", "VIN", "Driver", "ProviderName", "MonthlyAmount", "GeneralStatusName"],
                "response_template": None,  # Use _format_masterdata for comprehensive response
                "flow_type": "simple",
            },
            # === LICENCE PLATE ===
            {
                "patterns": [
                    r"tablice?",
                    r"registarsk.*oznaka",
                    r"registracij.*broj",
                    r"koje.*tablice",
                ],
                "intent": "GET_PLATE",
                "tool": "get_MasterData",
                "extract_fields": ["LicencePlate", "RegistrationNumber", "Plate"],
                "response_template": "🔢 **Tablica:** {value}",
                "flow_type": "simple",
            },
            # === LEASING ===
            {
                "patterns": [
                    r"lizing",
                    r"leasing",
                    r"koja.*lizing.*ku[cć]a",
                    r"lizing.*provider",
                ],
                "intent": "GET_LEASING",
                "tool": "get_MasterData",
                "extract_fields": ["ProviderName", "SupplierName"],
                "response_template": "🏢 **Lizing kuća:** {value}",
                "flow_type": "simple",
            },
            # === PERSONAL INFO (PersonData) ===
            {
                "patterns": [
                    r"kako.*se.*zovem",            # kako se zovem
                    r"moje.*ime",                  # moje ime
                    r"tko.*sam.*ja",               # tko sam ja
                    r"moji.*podaci",               # moji podaci
                    r"moj.*profil",                # moj profil
                    r"osobni.*podaci",             # osobni podaci
                    r"moj.*email",                 # moj email
                    r"moj.*telefon",               # moj telefon
                    r"moja.*tvrtka",               # moja tvrtka/firma
                    r"u.*kojoj.*firmi",            # u kojoj sam firmi
                ],
                "intent": "GET_PERSON_INFO",
                "tool": "get_PersonData_personIdOrEmail",
                "extract_fields": ["FirstName", "LastName", "DisplayName", "Email", "Phone", "CompanyName"],
                "response_template": None,  # Use _format_person_details
                "flow_type": "simple",
            },
            # === SERVICE / MAINTENANCE ===
            {
                "patterns": [
                    r"servis",                          # servis
                    r"koliko.*do.*servis",              # koliko do servisa
                    r"kad.*servis",                     # kad je servis, kad trebam na servis
                    r"kada.*servis",                    # kada je servis
                    r"sljede[cć]i.*servis",             # sljedeći servis
                    r"trebam.*servis",                  # trebam na servis
                    r"preostalo.*servis",               # preostalo do servisa
                    r"do.*servisa",                     # do servisa
                    r"odr[zž]avanj",                    # održavanje/odrzavanje
                    r"zadnji.*servis",                  # zadnji servis
                    r"pro[sš]li.*servis",               # prošli/prosli servis
                    r"povijest.*servis",                # povijest servisa
                ],
                "intent": "GET_SERVICE_MILEAGE",
                "tool": "get_MasterData",
                "extract_fields": ["ServiceMileage", "NextServiceMileage", "LastServiceDate"],
                "response_template": "🔧 **Do servisa:** {value} km",
                "flow_type": "simple",
            },
            # === MY BOOKINGS (must be BEFORE booking to catch "moje rezervacije" first) ===
            {
                "patterns": [
                    r"moje.*rezervacij",
                    r"moje.*booking",
                    r"moji.*booking",          # Phase 3: "moji bookingi"
                    r"kada.*imam.*rezerv",
                    r"kad.*imam.*auto",        # Phase 3: "kad imam auto"
                    r"kad.*imam.*vozilo",      # Phase 3: "kad imam vozilo"
                    r"poka[zž]i.*rezervacij",  # pokaži/pokazi rezervacije
                    r"poka[zž]i.*moje.*booking", # Phase 3: "pokaži moje bookinge"
                    r"prika[zž]i.*rezervacij", # prikaži/prikazi rezervacije
                    r"sve.*rezervacij",        # sve rezervacije
                    r"ima[lm].*rezerv",        # imam/imali rezervaciju
                ],
                "intent": "GET_MY_BOOKINGS",
                "tool": "get_VehicleCalendar",
                "extract_fields": ["FromTime", "ToTime", "VehicleName"],
                "response_template": None,
                "flow_type": "list",
            },
            # === DELETE / CANCEL OPERATIONS (must be BEFORE booking!) ===
            {
                "patterns": [
                    # Cancel reservation
                    r"otka[zž]i.*rezerv",       # otkaži/otkazi rezervaciju
                    r"cancel.*rezerv",          # cancel rezervaciju
                    r"poni[sš]ti.*rezerv",      # poništi/ponisti rezervaciju
                    r"obri[sš]i.*rezerv",       # obriši/obrisi rezervaciju
                    r"ukloni.*rezerv",          # ukloni rezervaciju
                    r"storniraj.*rezerv",       # storniraj rezervaciju
                    # Cancel booking
                    r"otka[zž]i.*booking",
                    r"cancel.*booking",
                    r"obri[sš]i.*booking",
                    # General delete for calendar
                    r"ne.*trebam.*vi[sš]e.*auto",   # ne trebam više auto
                    r"ne.*trebam.*vi[sš]e.*vozilo", # ne trebam više vozilo
                ],
                "intent": "CANCEL_RESERVATION",
                "tool": "delete_VehicleCalendar_id",
                "extract_fields": [],
                "response_template": None,
                "flow_type": "delete_booking",
            },
            {
                "patterns": [
                    # Delete trips
                    r"obri[sš]i.*putovanj",     # obriši/obrisi putovanje
                    r"obri[sš]i.*trip",         # obriši/obrisi trip
                    r"obri[sš]i.*vo[zž]nj",     # obriši/obrisi vožnju
                    r"ukloni.*putovanj",        # ukloni putovanje
                    r"izbri[sš]i.*putovanj",    # izbriši/izbrisi putovanje
                ],
                "intent": "DELETE_TRIP",
                "tool": "delete_Trips_id",
                "extract_fields": [],
                "response_template": None,
                "flow_type": "delete_trip",
            },
            {
                "patterns": [
                    # Delete partner
                    r"obri[sš]i.*partner",      # obriši/obrisi partnera
                    r"ukloni.*partner",         # ukloni partnera
                    r"izbri[sš]i.*partner",     # izbriši/izbrisi partnera
                    r"obri[sš]i.*dobavlja[cč]", # obriši/obrisi dobavljača
                    r"ukloni.*dobavlja[cč]",    # ukloni dobavljača
                ],
                "intent": "DELETE_PARTNER",
                "tool": "delete_Partners_id",
                "extract_fields": [],
                "response_template": None,
                "flow_type": "delete_partner",
            },
            {
                "patterns": [
                    # Delete case
                    r"obri[sš]i.*slu[cč]aj",    # obriši/obrisi slučaj
                    r"obri[sš]i.*prijav",       # obriši/obrisi prijavu
                    r"ukloni.*slu[cč]aj",       # ukloni slučaj
                    r"zatvori.*slu[cč]aj",      # zatvori slučaj
                ],
                "intent": "DELETE_CASE",
                "tool": "delete_Cases",
                "extract_fields": [],
                "response_template": None,
                "flow_type": "delete_case",
            },
            {
                "patterns": [
                    # Delete expense
                    r"obri[sš]i.*tro[sš]ak",    # obriši/obrisi trošak
                    r"obri[sš]i.*rashod",       # obriši/obrisi rashod
                    r"ukloni.*tro[sš]ak",       # ukloni trošak
                ],
                "intent": "DELETE_EXPENSE",
                "tool": "delete_Expenses_id",
                "extract_fields": [],
                "response_template": None,
                "flow_type": "delete_expense",
            },
            # === BOOKING / RESERVATION ===
            {
                "patterns": [
                    r"rezervir",
                    r"rezervacij",  # Note: "moje rezervacije" caught by MY_BOOKINGS above
                    r"trebam.*vozilo",
                    r"trebam.*auto",          # Phase 3: Informal "trebam auto"
                    r"trebam.*kola",          # Phase 3: Informal "trebam kola"
                    r"treba.*mi.*vozilo",
                    r"treba.*mi.*auto",       # Phase 3: Informal
                    r"daj.*mi.*auto",         # Phase 3: Informal "daj mi auto"
                    r"daj.*mi.*vozilo",       # Phase 3: Informal
                    r"ima.*li.*slobodn",      # Phase 3: "ima li slobodnih auta"
                    r"slobodn.*vozil",        # Phase 3: "slobodna vozila"
                    r"slobodn.*auto",         # Phase 3: "slobodna auta"
                    r"book(?!ing)",           # book but not booking (for "moje booking")
                    r"zauzmi",
                    r"zakup",
                    r"ho[cć]u.*rezerv",       # hoću rezervirati
                    r"[zž]elim.*rezerv",      # želim rezervirati
                    r"[zž]elim.*auto",        # Phase 3: "želim auto"
                    r"[zž]elim.*vozilo",      # Phase 3: "želim vozilo"
                ],
                "intent": "BOOK_VEHICLE",
                "tool": "get_AvailableVehicles",
                "extract_fields": [],
                "response_template": None,
                "flow_type": "booking",
            },
            # === CASES LIST (must be BEFORE REPORT DAMAGE to catch "prijavljene štete") ===
            {
                "patterns": [
                    r"prijavljene.*[sš]tet",     # V3.0: prijavljene štete (existing cases)
                    r"popis.*[sš]tet",           # popis šteta
                    r"lista.*[sš]tet",           # lista šteta
                    r"pregled.*[sš]tet",         # pregled šteta
                    r"povijest.*[sš]tet",        # povijest šteta
                    r"prika[zž]i.*[sš]tet",      # prikaži štete
                    r"poka[zž]i.*[sš]tet",       # pokaži štete
                    r"svi.*slu[cč]ajev",         # svi slučajevi
                    r"lista.*slu[cč]aj",         # lista slučajeva
                ],
                "intent": "GET_CASES",
                "tool": "get_Cases",
                "extract_fields": [],
                "response_template": None,
                "flow_type": "list",
            },
            # === REPORT DAMAGE ===
            {
                "patterns": [
                    r"prijavi.*kvar",
                    r"prijava.*kvar",
                    r"prijavi.*[sš]tet",    # prijavi štetu, prijavi stetu (with/without diacritics)
                    r"prijava.*[sš]tet",    # prijava štete, prijava stete
                    r"nova.*[sš]tet",       # V3.0: nova šteta (create, not list)
                    r"o[sš]te[cć]enj",      # oštećenje, ostecenje
                    r"ne[sš]to.*ne.*radi",
                    r"problem.*vozil",
                    r"problem.*auto",       # Phase 3: "problem s autom"
                    r"imam.*problem.*auto", # Phase 3: "imam problem s autom"
                    r"imam.*kvar",          # imam kvar
                    r"imam.*[sš]tet",       # imam štetu/stetu
                    r"dogodila.*nesre[cć]", # dogodila se nesreća/nesreca
                    r"nesre[cć]",           # nesreća/nesreca
                    r"sudar",               # sudar
                    r"udari",               # udario/udarila
                    r"ogreba",              # V3.0: ogrebao/ogrebala
                ],
                "intent": "REPORT_DAMAGE",
                "tool": "post_AddCase",
                "extract_fields": [],
                "response_template": None,
                "flow_type": "case_creation",
            },
            # === GREETINGS ===
            {
                "patterns": [
                    r"^bok$",
                    r"^cao$",
                    r"^pozdrav$",
                    r"^zdravo$",
                    r"^hej$",
                    r"^hi$",
                    r"^hello$",
                ],
                "intent": "GREETING",
                "tool": None,
                "extract_fields": [],
                "response_template": "Pozdrav! Kako vam mogu pomoći?",
                "flow_type": "direct_response",
            },
            # === THANKS ===
            {
                "patterns": [
                    r"hvala",
                    r"zahvalju",
                    r"thanks",
                    r"fala",
                ],
                "intent": "THANKS",
                "tool": None,
                "extract_fields": [],
                "response_template": "Nema na čemu! Slobodno pitajte ako trebate još nešto.",
                "flow_type": "direct_response",
            },
            # === HELP ===
            {
                "patterns": [
                    r"^pomo[cć]$",
                    r"^help$",
                    r"što.*može[sš]",
                    r"kako.*koristiti",
                    r"što.*zna[sš]",
                ],
                "intent": "HELP",
                "tool": None,
                "extract_fields": [],
                "response_template": (
                    "Mogu vam pomoći s:\n"
                    "• **Kilometraža** - provjera ili unos km\n"
                    "• **Rezervacije** - rezervacija vozila\n"
                    "• **Podaci o vozilu** - registracija, lizing\n"
                    "• **Prijava kvara** - kreiranje slučaja\n\n"
                    "Što vas zanima?"
                ),
                "flow_type": "direct_response",
            },
        ]

    def route(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> RouteResult:
        """
        Route query to appropriate tool.

        Args:
            query: User's query text
            user_context: Optional user context

        Returns:
            RouteResult with matched tool or not matched
        """
        query_lower = query.lower().strip()

        for rule in self.rules:
            for pattern in rule["patterns"]:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    logger.info(
                        f"ROUTER: Matched '{query[:30]}...' to {rule['intent']} "
                        f"→ {rule['tool'] or 'direct_response'}"
                    )

                    return RouteResult(
                        matched=True,
                        tool_name=rule["tool"],
                        extract_fields=rule["extract_fields"],
                        response_template=rule["response_template"],
                        flow_type=rule["flow_type"],
                        confidence=1.0,
                        reason=f"Matched pattern: {pattern}"
                    )

        # No exact match - let semantic search handle it
        logger.info(f"ROUTER: No match for '{query[:30]}...' - using semantic search")
        return RouteResult(
            matched=False,
            confidence=0.0,
            reason="No pattern matched, no domain detected"
        )


    def format_response(
        self,
        route: RouteResult,
        api_response: Dict[str, Any],
        query: str
    ) -> Optional[str]:
        """
        Format response using template if available.

        Args:
            route: The route result with template
            api_response: Raw API response
            query: Original query

        Returns:
            Formatted response string or None if should use LLM
        """
        if not route.response_template:
            return None

        if not route.extract_fields:
            return route.response_template

        # Try to extract value
        value = self._extract_value(api_response, route.extract_fields)
        
        logger.info(f"FORMAT_RESPONSE: fields={route.extract_fields}, value={value}, data_keys={list(api_response.keys()) if isinstance(api_response, dict) else 'not_dict'}")

        if value is None:
            return None  # Let LLM handle it

        # Format value based on field type
        formatted_value = self._format_value(value, route.extract_fields[0])

        return route.response_template.format(value=formatted_value)

    def _extract_value(
        self,
        data: Dict[str, Any],
        fields: List[str]
    ) -> Optional[Any]:
        """Extract value from response using field list."""
        if not data:
            return None

        # Try each field
        for field in fields:
            # Direct match
            if field in data and data[field] is not None:
                return data[field]

            # Nested search
            value = self._deep_get(data, field)
            if value is not None:
                return value

        return None

    def _deep_get(self, data: Any, key: str) -> Optional[Any]:
        """Recursively search for key in nested dict."""
        if isinstance(data, dict):
            if key in data:
                return data[key]
            for v in data.values():
                result = self._deep_get(v, key)
                if result is not None:
                    return result
        elif isinstance(data, list) and data:
            return self._deep_get(data[0], key)
        return None

    def _format_value(self, value: Any, field_name: str) -> str:
        """Format value based on field type."""
        if value is None:
            return "N/A"

        field_lower = field_name.lower()

        # Mileage - add thousand separators
        if "mileage" in field_lower:
            try:
                num = int(float(value))
                return f"{num:,}".replace(",", ".")
            except (ValueError, TypeError):
                return str(value)

        # Date - format as DD.MM.YYYY
        if "date" in field_lower or "expir" in field_lower:
            if isinstance(value, str) and "T" in value:
                try:
                    date_part = value.split("T")[0]
                    parts = date_part.split("-")
                    if len(parts) == 3:
                        return f"{parts[2]}.{parts[1]}.{parts[0]}"
                except (ValueError, AttributeError, IndexError):
                    pass
            return str(value)

        return str(value)


# Singleton
_router = None


def get_query_router() -> QueryRouter:
    """Get singleton instance."""
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router

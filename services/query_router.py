"""
Query Router - ML-based routing with response formatting.
Version: 2.0 (ML-based, replaces regex patterns)

CHANGELOG v2.0:
- REMOVED: 51 regex rules (~500 lines of patterns)
- ADDED: ML-based routing via IntentClassifier
- KEPT: Response formatting utilities

Single responsibility: Route queries to tools and format responses.
Uses ML model instead of regex patterns for 99%+ accuracy.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from services.intent_classifier import get_intent_classifier, IntentPrediction

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """Result of query routing."""
    matched: bool
    tool_name: Optional[str] = None
    extract_fields: List[str] = None
    response_template: Optional[str] = None
    flow_type: Optional[str] = None
    confidence: float = 1.0
    reason: str = ""

    def __post_init__(self):
        if self.extract_fields is None:
            self.extract_fields = []


# Intent to metadata mapping - minimal config instead of 500 lines of regex
# This maps ML intents to the metadata needed for execution
INTENT_METADATA = {
    "GET_MILEAGE": {
        "tool": "get_MasterData",
        "extract_fields": ["LastMileage", "Mileage", "CurrentMileage"],
        "response_template": "**Kilometraza:** {value} km",
        "flow_type": "simple",
    },
    "INPUT_MILEAGE": {
        "tool": "post_AddMileage",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "mileage_input",
    },
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
    "GET_PERSON_INFO": {
        "tool": "get_PersonData_personIdOrEmail",
        "extract_fields": ["FirstName", "LastName", "DisplayName", "Email"],
        "response_template": None,
        "flow_type": "simple",
    },
    "GET_VEHICLE_COUNT": {
        "tool": "get_Vehicles_Agg",
        "extract_fields": ["Count", "TotalCount"],
        "response_template": "**Broj vozila:** {value}",
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
    "GET_TENANT_ID": {
        "tool": None,
        "extract_fields": [],
        "response_template": "**Tenant ID:** {tenant_id}",
        "flow_type": "direct_response",
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
    "GET_AVAILABLE_VEHICLES": {
        "tool": "get_AvailableVehicles",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "list",
    },
    "GET_EXPENSES": {
        "tool": "get_Expenses",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "list",
    },
    "GET_VEHICLES": {
        "tool": "get_Vehicles",
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
    "GET_PERSONS": {
        "tool": "get_Persons",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "list",
    },
    "GET_COMPANIES": {
        "tool": "get_Companies",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "list",
    },
    "UPDATE_VEHICLE": {
        "tool": "patch_Vehicles_id",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "update_vehicle",
    },
    "DELETE_VEHICLE": {
        "tool": "delete_Vehicles_id",
        "extract_fields": [],
        "response_template": None,
        "flow_type": "delete_vehicle",
    },
}

# v16.0: Confidence threshold for DETERMINISTIC routing (bypasses LLM)
# Balanced threshold: High-confidence ML predictions bypass LLM for speed
# Lower confidence queries still go to LLM for final decision
ML_CONFIDENCE_THRESHOLD = 0.85  # 85%+ confidence uses ML directly (faster, cheaper)


class QueryRouter:
    """
    Routes queries to tools using ML-based intent classification.

    Version 2.0: Uses trained ML model instead of regex patterns.
    - 99.25% accuracy vs ~67% with regex
    - Handles typos, variations, and Croatian diacritics
    - Single model instead of 51 regex rules
    """

    def __init__(self):
        """Initialize router with ML classifier."""
        self._classifier = None

    @property
    def classifier(self):
        """Lazy load classifier."""
        if self._classifier is None:
            self._classifier = get_intent_classifier()
        return self._classifier

    def route(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> RouteResult:
        """
        Route query to appropriate tool using ML.

        Args:
            query: User's query text
            user_context: Optional user context

        Returns:
            RouteResult with matched tool or not matched
        """
        # Get ML prediction
        prediction = self.classifier.predict(query)

        logger.info(
            f"ROUTER ML: '{query[:30]}...' -> {prediction.intent} "
            f"({prediction.confidence:.1%}) tool={prediction.tool}"
        )

        # Check confidence threshold
        if prediction.confidence < ML_CONFIDENCE_THRESHOLD:
            logger.info(f"ROUTER: Low confidence ({prediction.confidence:.1%}), using semantic search")
            return RouteResult(
                matched=False,
                confidence=prediction.confidence,
                reason=f"ML confidence {prediction.confidence:.1%} below threshold"
            )

        # Get metadata for this intent
        metadata = INTENT_METADATA.get(prediction.intent)

        if metadata is None:
            # Intent recognized but no metadata - use ML tool suggestion
            return RouteResult(
                matched=True,
                tool_name=prediction.tool,
                extract_fields=[],
                response_template=None,
                flow_type="simple",
                confidence=prediction.confidence,
                reason=f"ML: {prediction.intent}"
            )

        return RouteResult(
            matched=True,
            tool_name=metadata["tool"],
            extract_fields=metadata["extract_fields"],
            response_template=metadata["response_template"],
            flow_type=metadata["flow_type"],
            confidence=prediction.confidence,
            reason=f"ML: {prediction.intent}"
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

        # Extract value from response
        value = self._extract_value(api_response, route.extract_fields)

        if value is None:
            return None  # Let LLM handle it

        # Format value
        formatted_value = self._format_value(value, route.extract_fields[0])
        return route.response_template.format(value=formatted_value)

    def _extract_value(self, data: Dict[str, Any], fields: List[str]) -> Optional[Any]:
        """Extract value from response using field list."""
        if not data:
            return None

        for field in fields:
            if field in data and data[field] is not None:
                return data[field]
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


# Singleton (thread-safe)
_router = None
_router_lock = threading.Lock()


def get_query_router() -> QueryRouter:
    """Get singleton instance (thread-safe)."""
    global _router
    if _router is not None:
        return _router
    with _router_lock:
        if _router is None:
            _router = QueryRouter()
    return _router

"""
Dependency Resolver - Data Models

Contains dataclasses and pattern constants used by the dependency resolver.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from services.entity_detector import _MOJ


@dataclass
class ResolutionResult:
    """Result of dependency resolution."""
    success: bool
    resolved_value: Optional[Any] = None
    provider_tool: Optional[str] = None
    provider_params: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    # NEW: User-facing feedback about what was resolved
    # Used for confirmations like "Razumijem: Golf (ZG-123-AB)"
    feedback: Optional[Dict[str, Any]] = None
    # Flag to indicate user selection is needed (e.g., no default vehicle)
    needs_user_selection: bool = False


@dataclass
class EntityReference:
    """
    Detected entity reference from user text.

    Enables "Vozilo 1" -> UUID resolution.
    """
    entity_type: str  # "vehicle", "person", "location"
    reference_type: str  # "ordinal", "possessive", "name", "pattern"
    value: str  # Original text ("Vozilo 1", "moje vozilo", "Golf")
    ordinal_index: Optional[int] = None  # For "Vozilo 1" -> 0 (0-indexed)
    is_possessive: bool = False  # For "moje vozilo", "moj auto"


# Ordinal patterns: "Vozilo 1", "Vozilo 2", "Vehicle 1"
ORDINAL_PATTERNS: List[Tuple[str, str]] = [
    # Croatian
    (r'vozilo\s*(\d+)', 'vehicle'),
    (r'auto\s*(\d+)', 'vehicle'),
    (r'automobil\s*(\d+)', 'vehicle'),
    # English
    (r'vehicle\s*(\d+)', 'vehicle'),
    (r'car\s*(\d+)', 'vehicle'),
    # Generic numbered
    (r'#(\d+)\s*vozilo', 'vehicle'),
    (r'broj\s*(\d+)', 'vehicle'),
]

# Possessive patterns: "moje vozilo", "moj auto", "my car"
# D6: Keep in sync with entity_detector._POSSESSIVE_PATTERNS (broader routing patterns).
# These patterns are vehicle-specific for entity resolution; entity_detector covers all entities.
# F1: _MOJ covers all Croatian declension forms (nom/gen/dat/lok/akuz/inst)
POSSESSIVE_PATTERNS: List[Tuple[str, str]] = [
    # Croatian possessives — all declension forms via _MOJ
    # \b prevents "nemoj vozilo" false positive (D2)
    (r'\b' + _MOJ + r'\s+vozil', 'vehicle'),
    (r'\b' + _MOJ + r'\s+auto', 'vehicle'),
    (r'\b' + _MOJ + r'\s+automobil', 'vehicle'),
    (r'\b' + _MOJ + r'\s+registracij', 'vehicle'),
    # Croatian possessives (dative/locative contracted - "na mom vozilu", "mom autu")
    (r'\bmom(?:e|u|em)?\s+vozil[ua]', 'vehicle'),
    (r'\bmom(?:e|u|em)?\s+aut[ua]', 'vehicle'),
    # English possessives
    (r'\bmy\s+vehicle', 'vehicle'),
    (r'\bmy\s+car', 'vehicle'),
    # NOTE: Implicit possessive patterns (just "vozilo" or "auto") are
    # intentionally NOT included because they are too aggressive and
    # could match unintended queries like "Koje vozilo je dostupno?"
]

# Vehicle name patterns are intentionally empty.
# A hardcoded brand/model list is fragile (can't cover all brands, doesn't
# auto-update, causes false positives like "Golf" the sport).
# Instead we rely on: ordinal references ("Vozilo 1"), possessive references
# ("moje vozilo"), and _fuzzy_match_vehicle() which searches actual data.
VEHICLE_NAME_PATTERNS: List[str] = []  # Intentionally empty - use fuzzy match instead

# Mapping: parameter name -> provider tool patterns
# These are semantic mappings, not hardcoded tool names
PARAM_PROVIDERS: Dict[str, Dict[str, Any]] = {
    'vehicleid': {
        'search_terms': ['vehicle', 'vehicles', 'masterdata'],
        'output_keys': ['Id', 'VehicleId', 'vehicleId'],
        'preferred_method': 'GET',
    },
    'personid': {
        'search_terms': ['person', 'persons', 'driver', 'user'],
        'output_keys': ['Id', 'PersonId', 'personId', 'UserId'],
        'preferred_method': 'GET',
    },
    'locationid': {
        'search_terms': ['location', 'locations', 'site'],
        'output_keys': ['Id', 'LocationId', 'locationId'],
        'preferred_method': 'GET',
    },
    'bookingid': {
        'search_terms': ['booking', 'calendar', 'reservation'],
        'output_keys': ['Id', 'BookingId', 'bookingId'],
        'preferred_method': 'GET',
    },
}

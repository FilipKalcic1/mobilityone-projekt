"""
Domain Detector - Deterministic domain classification for tool selection.
Version: 1.0

KRITIČNO: Ovo MORA biti točno. FAISS radi samo ako je domena ispravno zaključana.

Pravila:
1. Koristi HRVATSKE ključne riječi (korisnik piše na hrvatskom)
2. Nema fallbacka - ako nismo sigurni, vraćamo UNKNOWN + pitamo korisnika
3. Confidence mora biti visok (>0.7) ili ne zaključavamo domenu
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Set

logger = logging.getLogger(__name__)


class SuperDomain(str, Enum):
    """10 super-domena koje pokrivaju svih 950 alata."""
    VEHICLES = "VEHICLES"           # Vozila, registracije, kalendar vozila
    PERSONS = "PERSONS"             # Osobe, korisnici, tipovi osoba
    FINANCIALS = "FINANCIALS"       # Troškovi, rashodi, grupe troškova
    TRIPS = "TRIPS"                 # Putovanja, vožnje, kilometraža
    EQUIPMENT = "EQUIPMENT"         # Oprema, alati, kalendar opreme
    CASES = "CASES"                 # Slučajevi, štete, kvarovi
    TEAMS = "TEAMS"                 # Timovi, članovi tima
    ORGANIZATION = "ORGANIZATION"   # Tvrtke, partneri, org jedinice
    SCHEDULING = "SCHEDULING"       # Periodične aktivnosti, rasporedi
    SYSTEM = "SYSTEM"               # Lookup, postavke, dokumenti, tagovi
    UNKNOWN = "UNKNOWN"             # Nije moguće odrediti


class ActionIntent(str, Enum):
    """HTTP akcija koju korisnik želi izvršiti."""
    READ = "GET"
    CREATE = "POST"
    UPDATE = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    UNKNOWN = "UNKNOWN"


@dataclass
class DomainResult:
    """Rezultat detekcije domene."""
    domain: SuperDomain
    confidence: float
    matched_keywords: List[str]
    sub_domains: List[str]  # Specifičnije pod-domene (npr. Expenses, ExpenseTypes)


@dataclass
class ActionResult:
    """Rezultat detekcije akcije."""
    action: ActionIntent
    confidence: float
    matched_patterns: List[str]


# =============================================================================
# DOMAIN KEYWORDS - HRVATSKI
# =============================================================================

DOMAIN_KEYWORDS = {
    SuperDomain.VEHICLES: {
        "primary": [
            "vozilo", "vozila", "auto", "auta", "automobil",
            "registracija", "registracij", "tablica", "tablice",
            "proizvođač", "marka", "model", "vin",
        ],
        "secondary": [
            "vožnja", "vozač", "vozači", "flota", "fleet",
            "leasing", "lizing", "ugovor",
        ],
        "sub_domains": {
            "VehicleCalendar": ["kalendar vozila", "rezervacija vozila", "booking", "zauzet"],
            "VehicleContracts": ["ugovor", "leasing", "lizing", "najam"],
            "VehicleTypes": ["tip vozila", "vrsta vozila", "kategorija vozila"],
        },
    },

    SuperDomain.PERSONS: {
        "primary": [
            "osoba", "osobe", "korisnik", "korisnici", "korisnika",
            "zaposlenik", "zaposlenici", "radnik", "radnici",
            "ime", "prezime", "email", "telefon",
        ],
        "secondary": [
            "profil", "moj", "moja", "moje", "meni",
        ],
        "sub_domains": {
            "PersonTypes": ["tip osobe", "vrsta osobe", "kategorija osobe"],
            "PersonOrgUnits": ["org jedinica osobe", "odjel osobe"],
            "PersonActivities": ["aktivnost osobe", "zadatak osobe"],
        },
    },

    SuperDomain.FINANCIALS: {
        "primary": [
            "trošak", "troškovi", "troška", "troškove",
            "rashod", "rashodi", "rashoda",
            "izdatak", "izdaci",
            "cijena", "cijene", "iznos",
            "gorivo", "benzin", "diesel", "nafta",
            "račun", "računi", "faktura",
        ],
        "secondary": [
            "novac", "kuna", "eura", "euro", "eur", "hrk",
            "plaćanje", "plati", "platiti",
        ],
        "sub_domains": {
            "Expenses": ["trošak", "rashod", "izdatak", "račun"],
            "ExpenseTypes": ["tip troška", "vrsta troška", "kategorija troška"],
            "ExpenseGroups": ["grupa troškova", "grupa rashoda"],
            "CostCenters": ["troškovno mjesto", "cost center", "mjesto troška"],
        },
    },

    SuperDomain.TRIPS: {
        "primary": [
            "putovanje", "putovanja", "put", "puta",
            "vožnja", "vožnje",
            "kilometar", "kilometri", "kilometraža", "km",
            "mile", "mileage",
        ],
        "secondary": [
            "destinacija", "odredište", "polazište",
            "ruta", "trasa",
        ],
        "sub_domains": {
            "Trips": ["putovanje", "vožnja", "put"],
            "TripTypes": ["tip putovanja", "vrsta vožnje"],
            "MileageReports": ["kilometraža", "km", "izvještaj km", "mileage"],
        },
    },

    SuperDomain.EQUIPMENT: {
        "primary": [
            "oprema", "opreme", "opremu",
            "alat", "alati", "alata",
            "uređaj", "uređaji",
            "stroj", "strojevi",
        ],
        "secondary": [
            "inventar", "imovina",
        ],
        "sub_domains": {
            "Equipment": ["oprema", "alat", "uređaj"],
            "EquipmentTypes": ["tip opreme", "vrsta opreme"],
            "EquipmentCalendar": ["kalendar opreme", "rezervacija opreme"],
        },
    },

    SuperDomain.CASES: {
        "primary": [
            "slučaj", "slučajevi", "slučaja",
            "šteta", "štete", "oštećenje",
            "kvar", "kvarovi", "pokvaren",
            "prijava", "prijave", "prijavi",
            "incident", "nesreća",
        ],
        "secondary": [
            "problem", "greška", "bug",
            "sudar", "udar", "ogrebotina",
        ],
        "sub_domains": {
            "Cases": ["slučaj", "šteta", "kvar", "prijava"],
            "CaseTypes": ["tip slučaja", "vrsta štete", "kategorija kvara"],
        },
    },

    SuperDomain.TEAMS: {
        "primary": [
            "tim", "timovi", "tima",
            "član", "članovi", "člana",
            "ekipa", "ekipe",
            "grupa", "grupe", "grupu",
        ],
        "secondary": [
            "kolega", "kolege",
        ],
        "sub_domains": {
            "Teams": ["tim", "ekipa", "grupa"],
            "TeamMembers": ["član tima", "članovi", "tko je u timu"],
        },
    },

    SuperDomain.ORGANIZATION: {
        "primary": [
            "tvrtka", "tvrtke", "firma", "firme",
            "kompanija", "kompanij",
            "partner", "partneri", "dobavljač", "dobavljači",
            "organizacija", "org",
            "tenant", "klijent",
        ],
        "secondary": [
            "sektor", "odjel", "jedinica",
        ],
        "sub_domains": {
            "Companies": ["tvrtka", "firma", "kompanija"],
            "Partners": ["partner", "dobavljač", "supplier"],
            "OrgUnits": ["org jedinica", "odjel", "sektor"],
            "Tenants": ["tenant", "klijent sustava"],
        },
    },

    SuperDomain.SCHEDULING: {
        "primary": [
            "raspored", "rasporedi",
            "aktivnost", "aktivnosti",
            "periodičn", "redovit", "ponavljajuć",
            "servis", "održavanje",
            "pool", "bazen",
        ],
        "secondary": [
            "planiranje", "zakazivanje",
            "interval", "ciklus",
        ],
        "sub_domains": {
            "PeriodicActivities": ["periodična aktivnost", "redovita aktivnost"],
            "SchedulingModels": ["model rasporeda", "raspored"],
            "Pools": ["pool", "bazen vozila", "grupa vozila"],
        },
    },

    SuperDomain.SYSTEM: {
        "primary": [
            "postavka", "postavke", "settings",
            "lookup", "šifrarnik",
            "dokument", "dokumenti", "dokumenta",
            "tag", "tagovi", "oznaka", "oznake",
            "dashboard", "nadzorna", "pregled",
            "metadata", "metapodaci",
        ],
        "secondary": [
            "sustav", "sistem", "admin",
            "konfiguracija", "config",
        ],
        "sub_domains": {
            "Lookup": ["lookup", "šifrarnik", "lista vrijednosti"],
            "Settings": ["postavka", "settings", "konfiguracija"],
            "Documents": ["dokument", "dokumenti", "prilog"],
            "Tags": ["tag", "oznaka", "label"],
            "Dashboard": ["dashboard", "pregled", "statistika"],
        },
    },
}


# =============================================================================
# ACTION PATTERNS - HRVATSKI
# =============================================================================

ACTION_PATTERNS = {
    ActionIntent.DELETE: {
        "patterns": [
            r"\bobri[sš]i\b",      # obriši/obrisi
            r"\bizbri[sš]i\b",     # izbriši/izbrisi
            r"\bukloni\b",         # ukloni
            r"\bmakni\b",          # makni
            r"\bdelete\b",         # delete
            r"\botka[zž]i\b",      # otkaži/otkazi
            r"\bcancel\b",         # cancel
            r"\bponi[sš]ti\b",     # poništi/ponisti
            r"\bstorniraj\b",      # storniraj
        ],
        "confidence": 0.95,
        "dangerous": True,
    },

    ActionIntent.CREATE: {
        "patterns": [
            r"\bdodaj\b",          # dodaj
            r"\bkreiraj\b",        # kreiraj
            r"\bstvori\b",         # stvori
            r"\bnapravi\b",        # napravi
            r"\bunesi\b",          # unesi
            r"\bupi[sš]i\b",       # upiši/upisi
            r"\bprijavi\b",        # prijavi
            r"\brezerviraj\b",     # rezerviraj
            r"\bbook\b",           # book
            r"\bnov[aeiou]\b",     # nova/novi/novo
        ],
        "confidence": 0.90,
        "dangerous": False,
    },

    ActionIntent.UPDATE: {
        "patterns": [
            r"\ba[zž]uriraj\b",    # ažuriraj/azuriraj
            r"\bizmijeni\b",       # izmijeni
            r"\bpromijeni\b",      # promijeni
            r"\buredi\b",          # uredi
            r"\bupdate\b",         # update
            r"\bpopuni\b",         # popuni
        ],
        "confidence": 0.85,
        "dangerous": False,
    },

    ActionIntent.READ: {
        "patterns": [
            r"\bpoka[zž]i\b",      # pokaži/pokazi
            r"\bprika[zž]i\b",     # prikaži/prikazi
            r"\bdohvati\b",        # dohvati
            r"\bdaj\b",            # daj (mi)
            r"\blista\b",          # lista
            r"\bkoje?\b",          # koji/koja/koje
            r"\bkoliko\b",         # koliko
            r"\b[sš]to\b",         # što/sto
            r"\bima\s*li\b",       # ima li
            r"\bpregledaj\b",      # pregledaj
            r"\bvidi\b",           # vidi
            r"\btrebam\b",         # trebam (info)
            r"\binfo\b",           # info
        ],
        "confidence": 0.80,
        "dangerous": False,
    },

    ActionIntent.PATCH: {
        "patterns": [
            r"\bdjelomi[cč]no\b",  # djelomično
            r"\bpatch\b",          # patch
            r"\bsamo\s+promijeni\b",  # samo promijeni
        ],
        "confidence": 0.80,
        "dangerous": False,
    },
}


class DomainDetector:
    """
    Deterministički detektor domene i akcije.

    KRITIČNO: Ovaj modul MORA biti točan jer FAISS ovisi o ispravnoj domeni.
    Ako nismo sigurni → vraćamo UNKNOWN i pitamo korisnika.
    """

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for speed."""
        self._action_patterns = {}
        for action, config in ACTION_PATTERNS.items():
            self._action_patterns[action] = {
                "compiled": [re.compile(p, re.IGNORECASE) for p in config["patterns"]],
                "confidence": config["confidence"],
                "dangerous": config["dangerous"],
            }

    def detect_domain(self, query: str) -> DomainResult:
        """
        Detektiraj domenu iz upita.

        Returns:
            DomainResult s domenom, confidence i matched keywords.
            Ako confidence < 0.7, vraća UNKNOWN.
        """
        query_lower = query.lower()
        scores = {}
        matched = {}
        sub_domains_found = {}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = 0.0
            domain_matched = []
            domain_subs = []

            # Check primary keywords (weight: 1.0)
            for kw in keywords["primary"]:
                if kw in query_lower:
                    score += 1.0
                    domain_matched.append(kw)

            # Check secondary keywords (weight: 0.5)
            for kw in keywords["secondary"]:
                if kw in query_lower:
                    score += 0.5
                    domain_matched.append(kw)

            # Check sub-domains (helps narrow down)
            for sub_name, sub_keywords in keywords.get("sub_domains", {}).items():
                for skw in sub_keywords:
                    if skw in query_lower:
                        domain_subs.append(sub_name)
                        score += 0.3  # Bonus for sub-domain match

            if score > 0:
                scores[domain] = score
                matched[domain] = domain_matched
                sub_domains_found[domain] = list(set(domain_subs))

        if not scores:
            return DomainResult(
                domain=SuperDomain.UNKNOWN,
                confidence=0.0,
                matched_keywords=[],
                sub_domains=[]
            )

        # Find best domain
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_domain, best_score = sorted_domains[0]

        # Calculate confidence based on score and gap to second best
        max_possible = len(DOMAIN_KEYWORDS[best_domain]["primary"]) + \
                       len(DOMAIN_KEYWORDS[best_domain]["secondary"]) * 0.5

        base_confidence = min(best_score / max(max_possible, 1), 1.0)

        # Boost confidence if there's a clear gap
        if len(sorted_domains) > 1:
            second_score = sorted_domains[1][1]
            gap = (best_score - second_score) / max(best_score, 1)
            confidence = base_confidence * (0.7 + 0.3 * gap)
        else:
            confidence = base_confidence

        # Normalize to 0-1
        confidence = min(max(confidence, 0.0), 1.0)

        # If confidence too low, return UNKNOWN
        if confidence < 0.5:
            return DomainResult(
                domain=SuperDomain.UNKNOWN,
                confidence=confidence,
                matched_keywords=matched.get(best_domain, []),
                sub_domains=sub_domains_found.get(best_domain, [])
            )

        return DomainResult(
            domain=best_domain,
            confidence=confidence,
            matched_keywords=matched[best_domain],
            sub_domains=sub_domains_found.get(best_domain, [])
        )

    def detect_action(self, query: str) -> ActionResult:
        """
        Detektiraj akciju (GET/POST/DELETE/PUT/PATCH) iz upita.

        KRITIČNO za DELETE: Mora imati visoki confidence (>0.9).
        """
        query_lower = query.lower()

        # Check each action type
        matches = {}

        for action, config in self._action_patterns.items():
            action_matches = []
            for pattern in config["compiled"]:
                if pattern.search(query_lower):
                    action_matches.append(pattern.pattern)

            if action_matches:
                # Higher confidence for more matches
                base_conf = config["confidence"]
                boost = min(len(action_matches) * 0.05, 0.1)
                matches[action] = {
                    "confidence": min(base_conf + boost, 1.0),
                    "patterns": action_matches,
                    "dangerous": config["dangerous"],
                }

        if not matches:
            # Default to READ for queries without clear action
            return ActionResult(
                action=ActionIntent.READ,
                confidence=0.6,  # Lower confidence for implicit READ
                matched_patterns=[]
            )

        # Find best action
        sorted_actions = sorted(
            matches.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True
        )
        best_action, best_match = sorted_actions[0]

        # For dangerous actions (DELETE), require higher confidence
        if best_match["dangerous"] and best_match["confidence"] < 0.85:
            # Demote to UNKNOWN if not confident enough for DELETE
            return ActionResult(
                action=ActionIntent.UNKNOWN,
                confidence=best_match["confidence"],
                matched_patterns=best_match["patterns"]
            )

        return ActionResult(
            action=best_action,
            confidence=best_match["confidence"],
            matched_patterns=best_match["patterns"]
        )

    def detect(self, query: str) -> Tuple[DomainResult, ActionResult]:
        """
        Detektiraj i domenu i akciju iz upita.

        Returns:
            Tuple of (DomainResult, ActionResult)
        """
        domain = self.detect_domain(query)
        action = self.detect_action(query)

        logger.info(
            f"DomainDetector: '{query[:40]}...' → "
            f"Domain={domain.domain.value} ({domain.confidence:.2f}), "
            f"Action={action.action.value} ({action.confidence:.2f})"
        )

        return domain, action

    def get_sub_domain_tools_prefix(self, domain: SuperDomain, sub_domains: List[str]) -> List[str]:
        """
        Get tool name prefixes for sub-domains.

        Returns list of prefixes like ["Expenses", "ExpenseTypes"].
        """
        if not sub_domains:
            # Return all sub-domains for the domain
            config = DOMAIN_KEYWORDS.get(domain, {})
            return list(config.get("sub_domains", {}).keys())

        return sub_domains


# Singleton
_detector: Optional[DomainDetector] = None


def get_domain_detector() -> DomainDetector:
    """Get singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = DomainDetector()
    return _detector

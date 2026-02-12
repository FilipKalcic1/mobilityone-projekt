"""
Embedding Engine - Generate and manage embeddings for tool discovery.
Version: 1.0

Single responsibility: Generate embeddings using Azure OpenAI.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional

from openai import AsyncAzureOpenAI

from config import get_settings
from services.tool_contracts import (
    UnifiedToolDefinition,
    ParameterDefinition,
    DependencySource,
    DependencyGraph
)

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingEngine:
    """
    Manages embedding generation for semantic search.

    Responsibilities:
    - Build embedding text from tool definitions
    - Generate embeddings via Azure OpenAI
    - Build dependency graph for chaining
    """

    def __init__(self):
        """Initialize embedding engine with OpenAI client."""
        self.openai = AsyncAzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        logger.debug("EmbeddingEngine initialized")

    def build_embedding_text(
        self,
        operation_id: str,
        service_name: str,
        path: str,
        method: str,
        description: str,
        parameters: Dict[str, ParameterDefinition],
        output_keys: List[str] = None
    ) -> str:
        """
        Build embedding text with auto-generated PURPOSE description.

        v3.0: Enhanced inference from path, operationId, params, and outputs.

        Strategy:
        1. Generate PURPOSE from: path + operationId + method + params + outputs
        2. Include original description from Swagger
        3. List output fields for semantic matching

        Example:
            GET /vehicles/{id}/mileage + GetVehicleMileage
            → "Dohvaća kilometražu vozila. Vraća podatke o prijeđenim kilometrima."
        """
        # 1. Auto-generate purpose from structure (enhanced v3.0)
        purpose = self._generate_purpose(method, parameters, output_keys, path, operation_id)

        # 2. Build embedding text
        # FIXED: Removed operation_id (English) from embedding text
        # to ensure pure Croatian embeddings that match user queries
        parts = [
            purpose,  # Croatian auto-generated purpose
            description if description else "",
            # Removed: f"{method} {path}" - English, not helpful for Croatian queries
        ]

        # 3. Add output fields (human-readable)
        if output_keys:
            readable = [
                re.sub(r'([a-z])([A-Z])', r'\1 \2', k)
                for k in output_keys[:10]
            ]
            parts.append(f"Returns: {', '.join(readable)}")

        # 4. Add synonyms for better query matching (v3.1)
        synonyms = self._get_synonyms_for_purpose(purpose)
        if synonyms:
            parts.append(f"Sinonimi: {', '.join(synonyms)}")

        text = ". ".join(p for p in parts if p)

        if len(text) > 1500:
            text = text[:1500]

        return text

    # Comprehensive entity mappings for path/operationId extraction
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
        "driver": ("vozač", "vozača"),
        "drivers": ("vozač", "vozača"),
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
        "invoice": ("račun", "računa"),
        "invoices": ("račun", "računa"),
        "contract": ("ugovor", "ugovora"),
        "contracts": ("ugovor", "ugovora"),
        "report": ("izvještaj", "izvještaja"),
        "reports": ("izvještaj", "izvještaja"),
        "log": ("zapis", "zapisa"),
        "logs": ("zapis", "zapisa"),
        "history": ("povijest", "povijesti"),
        "record": ("zapis", "zapisa"),
        # Maintenance & Service
        "maintenance": ("održavanje", "održavanja"),
        "service": ("servis", "servisa"),
        "repair": ("popravak", "popravka"),
        "inspection": ("inspekcija", "inspekcije"),
        "damage": ("šteta", "štete"),
        "damages": ("šteta", "šteta"),
        "accident": ("nesreća", "nesreće"),
        "insurance": ("osiguranje", "osiguranja"),
        # Financial
        "payment": ("plaćanje", "plaćanja"),
        "payments": ("plaćanje", "plaćanja"),
        "cost": ("trošak", "troška"),
        "costs": ("trošak", "troškova"),
        "expense": ("trošak", "troška"),
        "expenses": ("trošak", "troškova"),
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
        "mileage": ("kilometraža", "kilometraže"),
        "odometer": ("kilometraža", "kilometraže"),
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
        "device": ("uređaj", "uređaja"),
        "devices": ("uređaj", "uređaja"),
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
        "note": ("bilješka", "bilješke"),
        "notes": ("bilješka", "bilješki"),
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
        "order": ("narudžba", "narudžbe"),
        "pool": ("bazen", "bazena"),
        "trip": ("putovanje", "putovanja"),
        "route": ("ruta", "rute"),
        "journey": ("vožnja", "vožnje"),
        "ride": ("vožnja", "vožnje"),
        "transfer": ("transfer", "transfera"),
        "pickup": ("preuzimanje", "preuzimanja"),
        "dropoff": ("vraćanje", "vraćanja"),
        "checkin": ("prijava", "prijave"),
        "checkout": ("odjava", "odjave"),
        "handover": ("primopredaja", "primopredaje"),
        "key": ("ključ", "ključa"),
        "keys": ("ključ", "ključeva"),
        "card": ("kartica", "kartice"),
        "fuelcard": ("kartica za gorivo", "kartice za gorivo"),
        "tollcard": ("ENC kartica", "ENC kartice"),
        "violation": ("prekršaj", "prekršaja"),
        "fine": ("kazna", "kazne"),
        "penalty": ("kazna", "kazne"),
    }

    # Output key mappings for result description
    OUTPUT_KEY_MAP = {
        # Vehicle data
        "mileage": "kilometražu",
        "km": "kilometre",
        "odometer": "stanje kilometara",
        "fuel": "razinu goriva",
        "fuellevel": "razinu goriva",
        "fuelconsumption": "potrošnju goriva",
        "battery": "stanje baterije",
        "batterylevel": "razinu baterije",
        "speed": "brzinu",
        "location": "lokaciju",
        "position": "poziciju",
        "coordinates": "koordinate",
        "latitude": "geografsku širinu",
        "longitude": "geografsku dužinu",
        # Status
        "status": "status",
        "state": "stanje",
        "available": "dostupnost",
        "availability": "dostupnost",
        "active": "aktivnost",
        "enabled": "omogućenost",
        "locked": "zaključanost",
        "online": "online status",
        "connected": "povezanost",
        # Registration & Documents
        "registration": "registraciju",
        "registrationnumber": "registarsku oznaku",
        "plate": "tablice",
        "licenseplate": "registarske tablice",
        "vin": "broj šasije",
        "chassisnumber": "broj šasije",
        "expiry": "datum isteka",
        "expirydate": "datum isteka",
        "expiration": "datum isteka",
        "expirationdate": "datum isteka",
        "validuntil": "vrijedi do",
        "validfrom": "vrijedi od",
        # Time
        "date": "datum",
        "time": "vrijeme",
        "datetime": "datum i vrijeme",
        "timestamp": "vremensku oznaku",
        "createdat": "datum kreiranja",
        "updatedat": "datum ažuriranja",
        "startedat": "vrijeme početka",
        "endedat": "vrijeme završetka",
        "duration": "trajanje",
        # Financial
        "price": "cijenu",
        "cost": "trošak",
        "amount": "iznos",
        "total": "ukupan iznos",
        "subtotal": "međuzbroj",
        "tax": "porez",
        "vat": "PDV",
        "discount": "popust",
        "balance": "stanje računa",
        "deposit": "polog",
        # Identification
        "id": "identifikator",
        "name": "naziv",
        "title": "naslov",
        "description": "opis",
        "code": "šifru",
        "number": "broj",
        "reference": "referencu",
        # Contact
        "email": "e-mail",
        "phone": "telefon",
        "mobile": "mobitel",
        "address": "adresu",
        "city": "grad",
        "country": "državu",
        "postalcode": "poštanski broj",
        # Person data
        "firstname": "ime",
        "lastname": "prezime",
        "fullname": "puno ime",
        "birthdate": "datum rođenja",
        "age": "dob",
        "gender": "spol",
        "nationality": "nacionalnost",
        "driverlicense": "vozačku dozvolu",
        # Lists & counts
        "count": "broj",
        "items": "stavke",
        "list": "popis",
        "results": "rezultate",
        # Booking specific
        "bookingid": "ID rezervacije",
        "bookingnumber": "broj rezervacije",
        "pickupdate": "datum preuzimanja",
        "returndate": "datum vraćanja",
        "pickuplocation": "mjesto preuzimanja",
        "returnlocation": "mjesto vraćanja",
    }

    # Croatian synonyms for common user queries
    # Uses root forms to match both nominative and genitive (vozilo/vozila)
    # Maps entity ROOT to list of alternative words users might use
    CROATIAN_SYNONYMS = {
        "vozil": ["auto", "automobil", "kola", "car"],  # vozilo, vozila
        "osob": ["čovjek", "korisnik", "user"],  # osoba, osobe
        "vozač": ["driver", "šofer"],  # vozač, vozača
        "rezervacij": ["booking", "najam", "iznajmljivanje", "rent"],  # rezervacija, rezervacije
        "lokacij": ["mjesto", "adresa", "pozicija", "location"],  # lokacija, lokacije
        "kilometraž": ["km", "kilometri", "prijeđeno", "mileage"],  # kilometraža, kilometraže
        "goriv": ["benzin", "nafta", "dizel", "fuel", "tank"],  # gorivo, goriva
        "račun": ["faktura", "invoice", "naplata"],  # račun, računa
        "ugovor": ["contract", "dogovor", "sporazum"],  # ugovor, ugovora
        "održavanj": ["servis", "service", "popravak", "maintenance"],  # održavanje, održavanja
        "štet": ["oštećenje", "damage", "kvar"],  # šteta, štete
        "cijen": ["trošak", "cost", "price", "iznos"],  # cijena, cijene
        "dostupnost": ["slobodno", "available", "raspoloživo"],  # dostupnost, dostupnosti
        "registracij": ["tablice", "plates", "oznaka"],  # registracija, registracije
        "osiguranj": ["polica", "insurance"],  # osiguranje, osiguranja
        "plaćanj": ["uplata", "payment", "transakcija"],  # plaćanje, plaćanja
        "izvještaj": ["report", "pregled", "statistika"],  # izvještaj, izvještaja
        "raspored": ["schedule", "plan", "kalendar"],  # raspored, rasporeda
        "putovanj": ["trip", "vožnja", "ruta", "journey"],  # putovanje, putovanja
        "kartic": ["card", "ENC", "fuel card"],  # kartica, kartice
        "flot": ["fleet", "vozni park"],  # flota, flote
    }

    def _generate_purpose(
        self,
        method: str,
        parameters: Dict[str, ParameterDefinition],
        output_keys: List[str],
        path: str = "",
        operation_id: str = ""
    ) -> str:
        """
        Auto-generate purpose from API structure (v3.0 - Enhanced).

        Infers from:
        - HTTP method → action (Dohvaća/Kreira/Ažurira/Briše)
        - PATH → entity (iz /vehicles/ → vozilo)
        - operationId → action + entity (GetVehicleMileage → dohvaća kilometražu vozila)
        - Input params → context (za vozilo/korisnika/period)
        - Output keys → result (kilometražu/registraciju/status)
        """
        # 1. Action from method
        actions = {
            "GET": "Dohvaća",
            "POST": "Kreira",
            "PUT": "Ažurira",
            "PATCH": "Ažurira",
            "DELETE": "Briše"
        }
        action = actions.get(method.upper(), "Obrađuje")

        # 2. Extract entities from PATH (most reliable source)
        path_entities = self._extract_entities_from_path(path)

        # 3. Extract entities from operationId
        op_entities, op_action_hint = self._parse_operation_id(operation_id)

        # 4. Context from input parameters
        param_context = []
        has_time = False

        if parameters:
            names = [p.name.lower() for p in parameters.values()]

            # Check each parameter name against entity map
            for name in names:
                for key, (singular, _) in self.PATH_ENTITY_MAP.items():
                    if key in name and singular not in param_context:
                        param_context.append(singular)
                        break

            has_time = (
                any(x in n for n in names for x in ["from", "start", "begin"]) and
                any(x in n for n in names for x in ["to", "end", "until"])
            )

        # 5. Result from output keys
        result = []

        if output_keys:
            keys_lower = [k.lower() for k in output_keys]

            for key in keys_lower:
                # Check against output key map
                for pattern, translation in self.OUTPUT_KEY_MAP.items():
                    if pattern in key and translation not in result:
                        result.append(translation)
                        if len(result) >= 4:
                            break
                if len(result) >= 4:
                    break

        # 6. Combine all sources to build purpose
        # Priority: path_entities > op_entities > param_context
        all_entities = []
        seen = set()

        for entity in path_entities + op_entities + param_context:
            if entity.lower() not in seen:
                all_entities.append(entity)
                seen.add(entity.lower())

        # Build the sentence
        purpose = action

        # Add result/what we're getting
        if result:
            purpose += " " + ", ".join(result[:3])
        elif op_action_hint:
            purpose += " " + op_action_hint
        elif method == "GET":
            purpose += " podatke"
        elif method == "POST":
            purpose += " novi zapis"
        elif method in ("PUT", "PATCH"):
            purpose += " postojeće podatke"
        elif method == "DELETE":
            purpose += " zapis"

        # Add context (what entity)
        if all_entities:
            # Use genitive form for "za X"
            entity_genitives = []
            for entity in all_entities[:2]:
                # Try to find genitive form
                for key, (singular, genitive) in self.PATH_ENTITY_MAP.items():
                    if singular == entity:
                        entity_genitives.append(genitive)
                        break
                else:
                    entity_genitives.append(entity)

            purpose += " za " + ", ".join(entity_genitives)

        if has_time:
            purpose += " u zadanom periodu"

        return purpose

    def _extract_entities_from_path(self, path: str) -> List[str]:
        """Extract entities from API path segments."""
        if not path:
            return []

        entities = []
        # Remove path parameters like {vehicleId}
        clean_path = re.sub(r'\{[^}]+\}', '', path)
        # Split by / and -
        segments = re.split(r'[/\-_]', clean_path.lower())

        for segment in segments:
            if not segment or len(segment) < 3:
                continue

            # Check against entity map
            if segment in self.PATH_ENTITY_MAP:
                singular, _ = self.PATH_ENTITY_MAP[segment]
                if singular not in entities:
                    entities.append(singular)
            else:
                # Try partial match for compound words
                for key, (singular, _) in self.PATH_ENTITY_MAP.items():
                    if key in segment and singular not in entities:
                        entities.append(singular)
                        break

        return entities[:3]  # Limit to 3 entities

    def _parse_operation_id(self, operation_id: str) -> tuple:
        """Parse operationId to extract action and entities."""
        if not operation_id:
            return [], ""

        # Split CamelCase: GetVehicleMileage -> ['Get', 'Vehicle', 'Mileage']
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', operation_id)

        if not words:
            return [], ""

        entities = []
        action_hint = ""

        # Skip common action verbs
        action_verbs = {"get", "create", "update", "delete", "post", "put",
                        "patch", "list", "find", "search", "add", "remove",
                        "set", "fetch", "retrieve", "check", "validate"}

        for word in words:
            word_lower = word.lower()

            if word_lower in action_verbs:
                continue

            # Check if word maps to an entity
            if word_lower in self.PATH_ENTITY_MAP:
                singular, _ = self.PATH_ENTITY_MAP[word_lower]
                if singular not in entities:
                    entities.append(singular)
            # Check output key map for action hints
            elif word_lower in self.OUTPUT_KEY_MAP:
                if not action_hint:
                    action_hint = self.OUTPUT_KEY_MAP[word_lower]

        return entities[:2], action_hint

    def _get_synonyms_for_purpose(self, purpose: str) -> List[str]:
        """
        Extract synonyms for entities mentioned in the purpose.

        This helps RAG match user queries that use alternative words.
        E.g., user says "auto" but API uses "vozilo" - synonyms bridge this gap.
        """
        if not purpose:
            return []

        synonyms = []
        purpose_lower = purpose.lower()

        # Check each entity in CROATIAN_SYNONYMS
        for entity, syn_list in self.CROATIAN_SYNONYMS.items():
            # If entity appears in purpose, add its synonyms
            if entity.lower() in purpose_lower:
                for syn in syn_list:
                    if syn.lower() not in purpose_lower and syn not in synonyms:
                        synonyms.append(syn)

        return synonyms[:8]  # Limit to 8 synonyms

    async def generate_embeddings(
        self,
        tools: Dict[str, UnifiedToolDefinition],
        existing_embeddings: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """
        Generate embeddings for tools that don't have them.

        Args:
            tools: Dict of tools by operation_id
            existing_embeddings: Already generated embeddings

        Returns:
            Updated embeddings dict
        """
        embeddings = dict(existing_embeddings)

        missing = [
            op_id for op_id in tools
            if op_id not in embeddings
        ]

        if not missing:
            logger.info("All embeddings cached")
            return embeddings

        logger.info(f"Generating {len(missing)} embeddings...")

        for op_id in missing:
            tool = tools[op_id]
            text = tool.embedding_text

            embedding = await self._get_embedding(text)
            if embedding:
                embeddings[op_id] = embedding

            await asyncio.sleep(0.05)  # Rate limiting

        logger.info(f"Generated {len(missing)} embeddings")
        return embeddings

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text from Azure OpenAI."""
        try:
            response = await self.openai.embeddings.create(
                input=[text[:8000]],
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding error: {e}")
            return None

    def build_dependency_graph(
        self,
        tools: Dict[str, UnifiedToolDefinition]
    ) -> Dict[str, DependencyGraph]:
        """
        Build dependency graph for automatic tool chaining.

        Identifies which tools can provide outputs needed by other tools.

        Args:
            tools: Dict of all tools

        Returns:
            Dict of DependencyGraph by tool_id
        """
        logger.info("Building dependency graph...")
        graph = {}

        for tool_id, tool in tools.items():
            # Find parameters that need FROM_TOOL_OUTPUT
            output_params = tool.get_output_params()
            required_outputs = list(output_params.keys())

            # Find tools that provide these outputs
            provider_tools = []
            for req_output in required_outputs:
                providers = self._find_providers(req_output, tools)
                provider_tools.extend(providers)

            if required_outputs:
                graph[tool_id] = DependencyGraph(
                    tool_id=tool_id,
                    required_outputs=required_outputs,
                    provider_tools=list(set(provider_tools))
                )

        logger.info(f"Built dependency graph: {len(graph)} tools with dependencies")
        return graph

    def _find_providers(
        self,
        output_key: str,
        tools: Dict[str, UnifiedToolDefinition]
    ) -> List[str]:
        """Find tools that provide given output key."""
        providers = []

        for tool_id, tool in tools.items():
            if output_key in tool.output_keys:
                providers.append(tool_id)
            # Case-insensitive match
            elif any(
                ok.lower() == output_key.lower()
                for ok in tool.output_keys
            ):
                providers.append(tool_id)

        return providers

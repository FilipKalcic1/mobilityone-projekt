"""
FAISS Vector Store - In-memory semantic search for tool selection.
Version: 1.0

ZERO DATABASE IMPACT - All operations in memory.

Uses tool_documentation.json (ACCURATE) as the source.
Does NOT use training_queries.json (UNRELIABLE).

Performance:
- Search latency: ~1-5ms (vs ~50ms with O(n) cosine loop)
- Memory: ~50MB for 950 tools (1536 dims * 4 bytes * 950)
- Startup: ~2s if embeddings cached, ~5min if regenerating
"""

import json
import logging
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

import numpy as np
import faiss

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Cache directory for embeddings
CACHE_DIR = Path(__file__).parent.parent / ".cache"
EMBEDDINGS_FILE = CACHE_DIR / "tool_embeddings.json"
EMBEDDING_DIM = 1536  # Azure OpenAI text-embedding-ada-002


@dataclass
class SearchResult:
    """Result from FAISS search."""
    tool_id: str
    score: float  # Cosine similarity (0-1)
    method: str   # HTTP method (GET/POST/PUT/DELETE)


class FAISSVectorStore:
    """
    In-memory vector store using FAISS for fast similarity search.

    Key features:
    - Uses tool_documentation.json as source (ACCURATE)
    - Caches embeddings to disk for fast startup
    - FAISS IndexFlatIP for exact cosine similarity
    - Zero database impact

    IMPORTANT: This class does NOT use training_queries.json because
    it is unreliable (55% coverage, word overlap issues).
    """

    def __init__(self):
        """Initialize the vector store."""
        self._index: Optional[faiss.IndexFlatIP] = None
        self._tool_ids: List[str] = []  # Maps FAISS index to tool_id
        self._tool_methods: Dict[str, str] = {}  # tool_id -> HTTP method
        self._embeddings: Dict[str, List[float]] = {}
        self._initialized = False
        self._openai_client = None

        # Ensure cache directory exists
        CACHE_DIR.mkdir(exist_ok=True)

        logger.info("FAISSVectorStore created (not yet initialized)")

    async def initialize(
        self,
        tool_documentation: Dict,
        tool_registry_tools: Optional[Dict] = None
    ) -> None:
        """
        Initialize the vector store with tool documentation.

        Args:
            tool_documentation: Dict from tool_documentation.json
            tool_registry_tools: Optional dict of UnifiedToolDefinition for method info
        """
        if self._initialized:
            logger.info("FAISSVectorStore already initialized")
            return

        logger.info(f"Initializing FAISSVectorStore with {len(tool_documentation)} tools...")

        # Extract HTTP methods from registry if available
        if tool_registry_tools:
            for tool_id, tool in tool_registry_tools.items():
                self._tool_methods[tool_id] = getattr(tool, 'method', 'GET')
        else:
            # Fallback: Extract methods from tool_id prefix (get_, post_, put_, delete_)
            for tool_id in tool_documentation:
                tool_lower = tool_id.lower()
                if tool_lower.startswith("get_"):
                    self._tool_methods[tool_id] = "GET"
                elif tool_lower.startswith("post_"):
                    self._tool_methods[tool_id] = "POST"
                elif tool_lower.startswith("put_"):
                    self._tool_methods[tool_id] = "PUT"
                elif tool_lower.startswith("patch_"):
                    self._tool_methods[tool_id] = "PATCH"
                elif tool_lower.startswith("delete_"):
                    self._tool_methods[tool_id] = "DELETE"
                else:
                    self._tool_methods[tool_id] = "GET"  # Default
            logger.info(f"Extracted HTTP methods from tool_id prefixes: {len(self._tool_methods)} tools")

        # Try to load cached embeddings
        cached = self._load_cached_embeddings()

        # Determine which tools need embedding generation
        tools_to_embed = []
        for tool_id in tool_documentation:
            if tool_id not in cached:
                tools_to_embed.append(tool_id)

        logger.info(f"Cached: {len(cached)}, Need to generate: {len(tools_to_embed)}")

        # Generate missing embeddings
        if tools_to_embed:
            await self._generate_embeddings(tools_to_embed, tool_documentation)
        else:
            self._embeddings = cached

        # Build FAISS index
        self._build_index()

        self._initialized = True
        logger.info(f"FAISSVectorStore initialized: {len(self._tool_ids)} tools indexed")

    def _load_cached_embeddings(self) -> Dict[str, List[float]]:
        """Load embeddings from cache file."""
        if not EMBEDDINGS_FILE.exists():
            return {}

        try:
            with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both formats:
            # 1. New flat format: {tool_id: [embedding], ...}
            # 2. Old nested format: {version, timestamp, embeddings: {tool_id: [embedding]}}
            if "embeddings" in data and isinstance(data["embeddings"], dict):
                # Nested format - extract embeddings dict
                embeddings = data["embeddings"]
                logger.info(f"Loaded {len(embeddings)} cached embeddings (nested format)")
            else:
                # Flat format - filter out non-embedding keys
                embeddings = {
                    k: v for k, v in data.items()
                    if isinstance(v, list) and len(v) == EMBEDDING_DIM
                }
                logger.info(f"Loaded {len(embeddings)} cached embeddings (flat format)")

            return embeddings
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}")
            return {}

    def _save_embeddings_to_cache(self) -> None:
        """Save embeddings to cache file."""
        try:
            with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._embeddings, f)

            logger.info(f"Saved {len(self._embeddings)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embeddings to cache: {e}")

    async def _generate_embeddings(
        self,
        tool_ids: List[str],
        tool_documentation: Dict
    ) -> None:
        """
        Generate embeddings for tools using tool_documentation.json.

        IMPORTANT: Uses purpose, when_to_use, and example_queries_hr
        from tool_documentation.json (ACCURATE source).

        Does NOT use training_queries.json (UNRELIABLE).
        """
        from openai import AsyncAzureOpenAI

        if self._openai_client is None:
            self._openai_client = AsyncAzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION
            )

        # Load existing cached embeddings
        self._embeddings = self._load_cached_embeddings()

        logger.info(f"Generating embeddings for {len(tool_ids)} tools...")

        generated_count = 0
        for tool_id in tool_ids:
            doc = tool_documentation.get(tool_id, {})

            # Build embedding text from DOCUMENTATION (not training_queries!)
            text = self._build_embedding_text(tool_id, doc)

            if not text:
                logger.warning(f"No text for {tool_id}, skipping")
                continue

            try:
                response = await self._openai_client.embeddings.create(
                    input=[text[:8000]],
                    model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
                )
                embedding = response.data[0].embedding
                self._embeddings[tool_id] = embedding
                generated_count += 1

                # Rate limiting
                await asyncio.sleep(0.05)

                if generated_count % 100 == 0:
                    logger.info(f"Generated {generated_count}/{len(tool_ids)} embeddings")
                    # Save intermediate progress
                    self._save_embeddings_to_cache()

            except Exception as e:
                logger.warning(f"Failed to generate embedding for {tool_id}: {e}")

        # Save final embeddings
        self._save_embeddings_to_cache()
        logger.info(f"Generated {generated_count} new embeddings")

    def _build_embedding_text(self, tool_id: str, doc: Dict) -> str:
        """
        Build text for embedding from tool documentation.

        V3.0: Added synonyms_hr support for improved semantic matching.

        Priority order (highest weight first):
        1. SYNONYMS_HR (5x) - User-provided Croatian synonyms (HIGHEST WEIGHT)
        2. BASE TOOL KEYWORDS (3x) - explicit phrases for primary entities
        3. Structural info (method, entity, suffix) - AUTO-GENERATED CROATIAN
        4. purpose (what it does) - CROATIAN
        5. when_to_use (use cases) - CROATIAN
        6. example_queries_hr (Croatian example queries) - CROATIAN

        Does NOT use training_queries.json!
        """
        parts = []

        # V3.0: SYNONYMS_HR - highest priority (5x repetition)
        # These are curated Croatian phrases that users actually use
        synonyms = doc.get("synonyms_hr", [])
        if synonyms:
            synonym_text = " ".join(synonyms)
            # Repeat 5x to heavily weight user synonyms
            for _ in range(5):
                parts.append(synonym_text)

        # V2.0: Add explicit keywords for BASE CRUD tools
        # These ensure primary entities rank higher than helpers/lookups
        base_keywords = self._get_base_tool_keywords(tool_id)
        if base_keywords:
            # Repeat 3x to heavily weight these keywords
            parts.append(base_keywords)
            parts.append(base_keywords)
            parts.append(base_keywords)

        # Structural info (method, entity, suffix)
        structural_info = self._extract_structural_info(tool_id)
        if structural_info:
            parts.append(structural_info)

        # Purpose
        purpose = doc.get("purpose", "")
        if purpose:
            parts.append(purpose)

        # When to use
        when_to_use = doc.get("when_to_use", [])
        if when_to_use:
            parts.append(" ".join(when_to_use))

        # Example queries (Croatian)
        example_queries = doc.get("example_queries_hr", [])
        if example_queries:
            parts.append(" ".join(example_queries))

        return " ".join(parts)

    def _get_base_tool_keywords(self, tool_id: str) -> str:
        """
        Get explicit Croatian keywords for base CRUD tools.

        These keywords ensure that:
        - get_Vehicles ranks #1 for "dohvati sva vozila"
        - get_Persons ranks #1 for "dohvati sve zaposlenike"
        - etc.
        """
        tool_lower = tool_id.lower()

        # PRIMARY ENTITY LIST ENDPOINTS (get_X without suffixes)
        BASE_LIST_KEYWORDS = {
            "get_companies": "dohvati sve kompanije lista svih kompanija popis kompanija pregledaj kompanije tvrtke firme",
            "get_vehicles": "dohvati sva vozila lista svih vozila popis vozila pregledaj vozila automobili auti",
            "get_persons": "dohvati sve osobe lista svih osoba popis osoba zaposlenici radnici djelatnici ljudi",
            "get_expenses": "dohvati sve troškove lista svih troškova popis troškova pregledaj troškove izdaci računi",
            "get_cases": "dohvati sve slučajeve lista svih slučajeva popis slučajeva štete kvarovi prijave",
            "get_teams": "dohvati sve timove lista svih timova popis timova grupe ekipe",
            "get_trips": "dohvati sva putovanja lista svih putovanja popis putovanja putni nalozi",
            "get_partners": "dohvati sve partnere lista svih partnera popis partnera dobavljači klijenti",
            "get_equipment": "dohvati svu opremu lista sve opreme popis opreme inventar alati",
            "get_orgunits": "dohvati sve organizacijske jedinice lista org jedinica popis odjela sektori",
            "get_costcenters": "dohvati sve troškovne centre lista mjesta troška popis cost centara",
            "get_roles": "dohvati sve uloge lista svih uloga popis uloga dozvole permisije",
            "get_tags": "dohvati sve oznake lista svih oznaka popis tagova",
            "get_pools": "dohvati sve poolove lista svih poolova popis poolova",
            "get_tenants": "dohvati sve tenante lista svih tenanata popis najmova",
        }

        # PRIMARY ENTITY BY ID ENDPOINTS (get_X_id)
        BASE_ID_KEYWORDS = {
            "get_companies_id": "dohvati jednu kompaniju po ID-u detalji kompanije informacije o kompaniji",
            "get_vehicles_id": "dohvati jedno vozilo po ID-u detalji vozila informacije o vozilu podaci vozila",
            "get_persons_id": "dohvati jednu osobu po ID-u detalji osobe informacije o zaposleniku podaci osobe",
            "get_expenses_id": "dohvati jedan trošak po ID-u detalji troška informacije o trošku",
            "get_cases_id": "dohvati jedan slučaj po ID-u detalji slučaja informacije o šteti",
            "get_teams_id": "dohvati jedan tim po ID-u detalji tima informacije o timu",
            "get_trips_id": "dohvati jedno putovanje po ID-u detalji putovanja informacije o putovanju",
            "get_equipment_id": "dohvati jednu opremu po ID-u detalji opreme informacije o opremi",
            "get_orgunits_id": "dohvati jednu org jedinicu po ID-u detalji odjela informacije o sektoru",
        }

        # SPECIAL TOOLS
        SPECIAL_KEYWORDS = {
            "get_vehiclecalendar": "kalendar vozila raspored vozila moje rezervacije vozila booking vozila pregled rezervacija svi bookings",
            "get_availablevehicles": "dostupna vozila slobodna vozila koja vozila su slobodna raspoloživa vozila",
            "get_masterdata": "osnovni podaci vozila master data registracija tablica kilometraža koliko km ima vozilo ukupna kilometraža podaci o vozilu",
            "post_addcase": "prijavi štetu prijavi kvar nova šteta novi kvar udario sam ogrebao sam oštećenje imam kvar na autu slomio sam",
            "post_addmileage": "unesi kilometražu dodaj km upiši kilometre prijeđeni put nova kilometraža",
            "post_vehiclecalendar": "nova rezervacija vozila rezerviraj vozilo zauzmi vozilo booking dodaj rezervaciju",
            "post_booking": "nova rezervacija booking rezerviraj zauzmi dodaj booking",
            "get_orgunits_tree": "hijerarhija organizacijskih jedinica stablo odjela struktura odjela parent child",
        }

        # Check which keywords to return
        if tool_lower in BASE_LIST_KEYWORDS:
            return BASE_LIST_KEYWORDS[tool_lower]
        elif tool_lower in BASE_ID_KEYWORDS:
            return BASE_ID_KEYWORDS[tool_lower]
        elif tool_lower in SPECIAL_KEYWORDS:
            return SPECIAL_KEYWORDS[tool_lower]

        return ""

    def _extract_structural_info(self, tool_id: str) -> str:
        """
        Extract structural information from tool_id and convert to Croatian.
        This helps differentiate similar tools.
        """
        # Suffix meanings for structural differentiation (MORE DISTINCT!)
        SUFFIX_MEANINGS = {
            "_id_documents_documentId_thumb": "slicica thumbnail preview dokumenta datoteke priloga",
            "_id_documents_documentId_SetAsDefault": "postavljanje dokumenta kao zadanog default",
            "_id_documents_documentId": "dohvati jedan konkretni dokument datoteku prilog po njegovom ID-u",
            "_id_documents": "lista svih dokumenata datoteka priloga prilozen stavci",
            "_id_metadata": "metapodaci shema struktura polja kolone definicija",
            "_DeleteByCriteria": "brisanje prema kriterijima filtriranja uvjetno selektivno",
            "_multipatch": "bulk batch grupno visestruko azuriranje vise stavki odjednom",
            "_SetAsDefault": "postavljanje kao zadano default",
            "_GroupBy": "grupiranje podataka po kategoriji agregacija grupa",
            "_ProjectTo": "projekcija samo odredenih polja select fields",
            "_Agg": "agregacija suma prosjek minimum maksimum count statistika",
            "_tree": "hijerarhijska struktura stablo parent child",
            "_id": "dohvati informacije detalje o jednoj konkretnoj stavci entitetu po ID-u",
        }

        # Entity names in Croatian (ORDER MATTERS - longer/specific first!)
        ENTITY_NAMES = {
            # Person-specific entities (MUST come before "periodicactivities")
            "personperiodicactivities": "aktivnosti osobe zaposlenika osobne periodicne",
            "personorgunits": "organizacijske jedinice osobe zaposlenika",
            "latestpersonperiodicactivities": "najnovije aktivnosti osobe zaposlenika",
            # Latest entities
            "latestvehiclecalendar": "najnoviji kalendar vozila",
            "latestvehiclecontracts": "najnoviji ugovori vozila",
            "latestperiodicactivities": "najnovije periodicne aktivnosti opcenito",
            # Standard entities
            "companies": "kompanija tvrtka firma poduzece",
            "vehicles": "vozilo automobil auto",
            "vehicletypes": "tip vrste vozila kategorija",
            "vehiclecalendar": "kalendar vozila raspored rezervacija",
            "vehiclecontracts": "ugovori vozila",
            "persons": "osoba zaposlenik radnik",
            "teams": "tim grupa ekipa",
            "cases": "predmet slucaj steta kvar prijava",
            "expenses": "trosak izdatak racun",
            "expensetypes": "tip vrste troska kategorija",
            "mileage": "kilometraza km prijedeni put",
            "calendar": "kalendar raspored rezervacija",
            "documents": "dokument prilog datoteka",
            "equipment": "oprema inventar alat",
            "equipmenttypes": "tip vrste opreme kategorija",
            "equipmentcalendar": "kalendar opreme raspored",
            "partners": "partner dobavljac klijent",
            "tenants": "najam tenant",
            "tenantpermissions": "dozvole najma korisnicke dozvole",
            "orgunits": "organizacijska jedinica odjel sektor",
            "costcenters": "troskovni centar mjesto troska",
            "trips": "putovanje trip putni nalog",
            "triptypes": "tip vrste putovanja kategorija",
            "roles": "uloga dozvola rola",
            "periodicactivities": "periodicna aktivnost servis opcenito",
            "periodicactivitiesschedules": "raspored periodicnih aktivnosti",
            "schedulingmodels": "model rasporedivanja",
            "metadata": "metapodaci shema struktura",
            "settings": "postavke konfiguracija",
        }

        # HTTP method meanings (VERY DISTINCT between PUT and PATCH!)
        METHOD_MEANINGS = {
            "get": "dohvati prikazi pogledaj vrati citaj preuzmi",
            "post": "dodaj kreiraj napravi unesi stvori zapisi novi nova novo",
            "put": "potpuno zamijeni sve azuriraj cijeli objekt kompletno update full",
            "patch": "djelomicno azuriraj samo neka polja parcijalno partial modificiraj",
            "delete": "obrisi ukloni izbrisi makni trajno",
        }

        parts = []
        tool_lower = tool_id.lower()

        # =====================================================
        # ENTITY FIRST! (repeated 5x to dominate embedding)
        # This is the MOST IMPORTANT differentiator
        # =====================================================
        import re
        name = re.sub(r'^(get|post|put|patch|delete)_', '', tool_id, flags=re.IGNORECASE)
        for suffix in SUFFIX_MEANINGS.keys():
            if name.lower().endswith(suffix.lower()):
                name = name[:-len(suffix)]
                break
        name = re.sub(r'_id$', '', name, flags=re.IGNORECASE)

        # Find entity and ADD IT FIRST (5x repetition for weight!)
        entity_value = ""
        name_lower = name.lower()
        for key, value in ENTITY_NAMES.items():
            if key in name_lower:
                entity_value = value
                break

        if entity_value:
            # Repeat entity 5x at START to heavily weight it
            parts.append(entity_value)
            parts.append(entity_value)
            parts.append(entity_value)
            parts.append(entity_value)
            parts.append(entity_value)

        # Extract suffix meaning (check longest first)
        for suffix, meaning in sorted(SUFFIX_MEANINGS.items(), key=lambda x: len(x[0]), reverse=True):
            if tool_lower.endswith(suffix.lower()):
                parts.append(meaning)
                break

        # Extract HTTP method (LAST - least important for differentiation)
        method = None
        for m in ["get", "post", "put", "patch", "delete"]:
            if tool_lower.startswith(f"{m}_"):
                method = m
                break

        if method:
            parts.append(METHOD_MEANINGS.get(method, ""))

        return " ".join(parts)

    def _build_index(self) -> None:
        """Build FAISS index from embeddings."""
        if not self._embeddings:
            logger.warning("No embeddings to index")
            return

        # Convert embeddings to numpy array
        self._tool_ids = list(self._embeddings.keys())
        embeddings_matrix = np.array(
            [self._embeddings[tool_id] for tool_id in self._tool_ids],
            dtype=np.float32
        )

        # Normalize for cosine similarity (IndexFlatIP = inner product)
        faiss.normalize_L2(embeddings_matrix)

        # Create FAISS index
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._index.add(embeddings_matrix)

        logger.info(f"FAISS index built: {self._index.ntotal} vectors")

    def _detect_entity_from_query(self, query: str) -> Optional[str]:
        """
        Detect entity mentioned in the query for hierarchical filtering.
        Returns the entity key (e.g., 'companies', 'vehicles') or None.
        """
        query_lower = query.lower()

        # Entity detection patterns (Croatian keywords -> entity key)
        # Order matters: LONGER/MORE SPECIFIC patterns MUST come first!
        ENTITY_PATTERNS = [
            # LATEST entities MUST come before regular ones
            (['najnovija aktivnost osobe', 'najnovije aktivnosti osobe', 'latest aktivnost osobe'], 'latestpersonperiodicactivities'),
            (['najnovija periodična aktivnost', 'najnovije periodične aktivnosti', 'najnovija periodična'], 'latestperiodicactivities'),
            (['najnoviji kalendar vozila', 'latest kalendar vozila'], 'latestvehiclecalendar'),
            (['najnoviji ugovor vozila', 'najnoviji ugovori vozila'], 'latestvehiclecontracts'),
            (['najnoviji kalendar opreme', 'latest kalendar opreme'], 'latestequipmentcalendar'),
            (['najnovija kilometraža', 'najnoviji izvještaj kilometraže'], 'latestmileagereports'),

            # Type entities before base entities
            (['tip periodične aktivnosti', 'tipovi periodičnih aktivnosti'], 'periodicactivitytypes'),
            (['tip aktivnosti osobe', 'tipovi aktivnosti osobe'], 'personactivitytypes'),

            # Regular compound entities
            (['aktivnost osobe', 'aktivnosti osobe', 'osobne aktivnosti'], 'personperiodicactivities'),
            (['periodična aktivnost', 'periodične aktivnosti', 'periodicna aktivnost'], 'periodicactivities'),
            (['kalendar vozila', 'raspored vozila'], 'vehiclecalendar'),
            (['kalendar opreme', 'raspored opreme'], 'equipmentcalendar'),
            (['ugovor vozila', 'ugovori vozila'], 'vehiclecontracts'),
            (['organizacijska jedinica osobe', 'org jedinica osobe'], 'personorgunits'),
            (['organizacijska jedinica', 'org. jedinica', 'odjel'], 'orgunits'),
            (['troškovni centar', 'mjesto troška', 'cost center'], 'costcenters'),
            (['tip dokumenta', 'tipovi dokumenta', 'vrsta dokumenta'], 'documenttypes'),
            (['tip troška', 'tipovi troškova', 'vrsta troška'], 'expensetypes'),
            (['tip vozila', 'tipovi vozila', 'vrsta vozila'], 'vehicletypes'),
            (['tip opreme', 'tipovi opreme', 'vrsta opreme'], 'equipmenttypes'),
            (['tip putovanja', 'tipovi putovanja', 'vrsta putovanja'], 'triptypes'),
            (['model raspoređivanja', 'modeli raspoređivanja'], 'schedulingmodels'),
            (['raspored periodičnih', 'rasporedi periodičnih'], 'periodicactivitiesschedules'),
            (['dozvola najma', 'dozvole najma', 'korisničke dozvole'], 'tenantpermissions'),
            (['član tima', 'članovi tima'], 'teammembers'),
            (['povijesni zapis vozila', 'povijest vozila'], 'vehicleshistoricalentries'),
            (['mjesečni trošak vozila'], 'vehiclesmonthlyexpenses'),
            (['izvještaj o kilometraži', 'kilometraža'], 'mileage'),

            # Simple entities
            (['kompanij', 'tvrtk', 'firma', 'poduzeć'], 'companies'),
            (['vozil', 'automobil', 'auto '], 'vehicles'),
            (['osob', 'zaposlen', 'radnik'], 'persons'),
            (['tim', 'ekip', 'grup'], 'teams'),
            (['predmet', 'slučaj', 'šteta', 'kvar', 'prijav'], 'cases'),
            (['trošak', 'troška', 'izdatak', 'račun'], 'expenses'),
            (['putovanj', 'putni nalog'], 'trips'),
            (['oprem', 'inventar'], 'equipment'),
            (['partner', 'dobavljač', 'klijent'], 'partners'),
            (['najam', 'tenant'], 'tenants'),
            (['ulog', 'role', 'rola'], 'roles'),
            (['oznaka', 'tag'], 'tags'),
            (['pool'], 'pools'),
            (['metapodatak', 'metapodaci', 'shema', 'struktura'], 'metadata'),
            (['nadzorna ploča', 'dashboard'], 'dashboarditems'),
        ]

        for keywords, entity in ENTITY_PATTERNS:
            for kw in keywords:
                if kw in query_lower:
                    return entity

        return None

    async def search(
        self,
        query: str,
        top_k: int = 10,
        action_filter: Optional[str] = None,
        entity_filter: Optional[str] = None,
        auto_detect_entity: bool = False  # Disabled by default - causes accuracy drop
    ) -> List[SearchResult]:
        """
        Search for similar tools using FAISS with optional hierarchical filtering.

        Args:
            query: User query text
            top_k: Number of results to return
            action_filter: Optional filter by action (GET/POST/PUT/DELETE)
            entity_filter: Optional filter by entity (companies, vehicles, etc.)
            auto_detect_entity: If True, auto-detect entity from query text

        Returns:
            List of SearchResult sorted by similarity (highest first)
        """
        if not self._initialized or self._index is None:
            logger.warning("FAISSVectorStore not initialized, returning empty results")
            return []

        # Auto-detect entity from query if not provided
        detected_entity = None
        if auto_detect_entity and not entity_filter:
            detected_entity = self._detect_entity_from_query(query)
            if detected_entity:
                logger.debug(f"Auto-detected entity: {detected_entity}")

        effective_entity_filter = entity_filter or detected_entity

        # Expand query with concept mapper (jargon -> standard terms)
        from services.concept_mapper import get_concept_mapper
        concept_mapper = get_concept_mapper()
        expanded_query = concept_mapper.expand_query(query)

        if expanded_query != query:
            logger.debug(f"ConceptMapper expanded: '{query}' -> '{expanded_query}'")

        # Get query embedding (using expanded query for better matching)
        query_embedding = await self._get_query_embedding(expanded_query)
        if query_embedding is None:
            return []

        # Convert to numpy and normalize
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        # Search with larger k if filtering (need more candidates)
        has_filter = action_filter or effective_entity_filter
        search_k = top_k * 5 if has_filter else top_k

        # FAISS search
        distances, indices = self._index.search(query_vector, min(search_k, len(self._tool_ids)))

        # Build results with filtering
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue

            tool_id = self._tool_ids[idx]
            score = float(distance)  # Already cosine similarity due to normalization
            method = self._tool_methods.get(tool_id, "GET")
            tool_lower = tool_id.lower()

            # Apply entity filter if specified
            if effective_entity_filter:
                # Extract entity from tool_id (e.g., get_Companies_id -> companies)
                parts = tool_lower.split('_')
                tool_entity = parts[1] if len(parts) >= 2 else ''

                # Check if entity matches (with some flexibility)
                if effective_entity_filter not in tool_entity and tool_entity not in effective_entity_filter:
                    continue

            # Apply action filter if specified
            if action_filter:
                # GET filter: allow GET and search POSTs
                if action_filter == "GET" and method != "GET":
                    if method == "POST" and any(x in tool_lower for x in ["search", "query", "filter", "list"]):
                        pass  # Allow search POSTs
                    else:
                        continue
                # POST filter: only POST methods (excluding search POSTs)
                elif action_filter == "POST" and method != "POST":
                    continue
                # PUT filter: PUT or PATCH
                elif action_filter in ("PUT", "PATCH") and method not in ("PUT", "PATCH"):
                    continue
                # DELETE filter
                elif action_filter == "DELETE" and method != "DELETE":
                    continue

            results.append(SearchResult(
                tool_id=tool_id,
                score=score,
                method=method
            ))

            if len(results) >= top_k:
                break

        # If entity filter was too restrictive and we got no results, retry without it
        if not results and effective_entity_filter and detected_entity:
            logger.debug(f"Entity filter too restrictive, retrying without entity filter")
            return await self.search(
                query=query,
                top_k=top_k,
                action_filter=action_filter,
                entity_filter=None,
                auto_detect_entity=False
            )

        return results

    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get embedding for query text."""
        from openai import AsyncAzureOpenAI

        if self._openai_client is None:
            self._openai_client = AsyncAzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION
            )

        try:
            response = await self._openai_client.embeddings.create(
                input=[query[:8000]],
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Failed to get query embedding: {e}")
            return None

    def get_tool_method(self, tool_id: str) -> str:
        """Get HTTP method for a tool."""
        return self._tool_methods.get(tool_id, "GET")

    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        return self._initialized

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "initialized": self._initialized,
            "total_tools": len(self._tool_ids),
            "total_embeddings": len(self._embeddings),
            "index_size": self._index.ntotal if self._index else 0,
            "cache_file": str(EMBEDDINGS_FILE),
            "cache_exists": EMBEDDINGS_FILE.exists()
        }


# Singleton instance
_faiss_store: Optional[FAISSVectorStore] = None


def get_faiss_store() -> FAISSVectorStore:
    """Get singleton FAISSVectorStore instance."""
    global _faiss_store
    if _faiss_store is None:
        _faiss_store = FAISSVectorStore()
    return _faiss_store


async def initialize_faiss_store(
    tool_documentation: Dict,
    tool_registry_tools: Optional[Dict] = None
) -> FAISSVectorStore:
    """
    Initialize and return the FAISS vector store.

    Call this during application startup.
    """
    store = get_faiss_store()
    await store.initialize(tool_documentation, tool_registry_tools)
    return store

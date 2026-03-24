"""
Generate Tool Embeddings Script.

Generates embeddings for all tools from tool_documentation.json
and saves them to .cache/tool_embeddings.json.

Usage:
    python scripts/generate_tool_embeddings.py

IMPORTANT: Uses tool_documentation.json (ACCURATE) as source.
Does NOT use training_queries.json (UNRELIABLE).
"""

import asyncio
import json
import logging
import os
import re as regex_module
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncAzureOpenAI
from config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
CONFIG_DIR = project_root / "config"
CACHE_DIR = project_root / ".cache"
TOOL_DOC_FILE = CONFIG_DIR / "tool_documentation.json"
EMBEDDINGS_FILE = CACHE_DIR / "tool_embeddings.json"

# Embedding dimension
EMBEDDING_DIM = 1536


def load_tool_documentation() -> dict:
    """Load tool documentation from config."""
    if not TOOL_DOC_FILE.exists():
        raise FileNotFoundError(f"Tool documentation not found: {TOOL_DOC_FILE}")

    with open(TOOL_DOC_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} tools from tool_documentation.json")
    return data


def load_existing_embeddings() -> dict:
    """Load existing embeddings from cache."""
    if not EMBEDDINGS_FILE.exists():
        return {}

    try:
        with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} existing embeddings from cache")
        return data
    except Exception as e:
        logger.warning(f"Failed to load cached embeddings: {e}")
        return {}


def save_embeddings(embeddings: dict):
    """Save embeddings to cache."""
    CACHE_DIR.mkdir(exist_ok=True)

    with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f)

    logger.info(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_FILE}")


# Suffix meanings for structural differentiation (SYNCED with faiss_vector_store.py!)
SUFFIX_MEANINGS_HR = {
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

# Entity names in Croatian (SYNCED with faiss_vector_store.py - ORDER MATTERS!)
ENTITY_NAMES_HR = {
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
    "company": "kompanija tvrtka firma poduzece",
    "vehicles": "vozilo automobil auto",
    "vehicle": "vozilo automobil auto",
    "vehicletypes": "tip vrste vozila kategorija",
    "vehiclecalendar": "kalendar vozila raspored rezervacija",
    "vehiclecontracts": "ugovori vozila",
    "persons": "osoba zaposlenik radnik",
    "person": "osoba zaposlenik radnik",
    "teams": "tim grupa ekipa",
    "team": "tim grupa ekipa",
    "cases": "predmet slucaj steta kvar prijava",
    "case": "predmet slucaj steta kvar prijava",
    "expenses": "trosak izdatak racun",
    "expense": "trosak izdatak racun",
    "expensetypes": "tip vrste troska kategorija",
    "mileage": "kilometraza km prijedeni put",
    "calendar": "kalendar raspored rezervacija",
    "documents": "dokument prilog datoteka",
    "document": "dokument prilog datoteka",
    "equipment": "oprema inventar alat",
    "equipmenttypes": "tip vrste opreme kategorija",
    "equipmentcalendar": "kalendar opreme raspored",
    "partners": "partner dobavljac klijent",
    "partner": "partner dobavljac klijent",
    "tenants": "najam tenant",
    "tenant": "najam tenant",
    "tenantpermissions": "dozvole najma korisnicke dozvole",
    "orgunits": "organizacijska jedinica odjel sektor",
    "orgunit": "organizacijska jedinica odjel sektor",
    "costcenters": "troskovni centar mjesto troska",
    "costcenter": "troskovni centar mjesto troska",
    "trips": "putovanje trip putni nalog",
    "triptypes": "tip vrste putovanja kategorija",
    "roles": "uloga dozvola rola",
    "periodicactivities": "periodicna aktivnost servis opcenito",
    "periodicactivitiesschedules": "raspored periodicnih aktivnosti",
    "schedulingmodels": "model rasporedivanja",
    "metadata": "metapodaci shema struktura",
    "settings": "postavke konfiguracija",
}

# HTTP method meanings (SYNCED with faiss_vector_store.py - VERY DISTINCT!)
METHOD_MEANINGS_HR = {
    "get": "dohvati prikazi pogledaj vrati citaj preuzmi",
    "post": "dodaj kreiraj napravi unesi stvori zapisi novi nova novo",
    "put": "potpuno zamijeni sve azuriraj cijeli objekt kompletno update full",
    "patch": "djelomicno azuriraj samo neka polja parcijalno partial modificiraj",
    "delete": "obrisi ukloni izbrisi makni trajno",
}


def extract_structural_info(tool_id: str) -> str:
    """
    Extract structural information from tool_id and convert to Croatian.
    This helps differentiate similar tools.

    IMPORTANT: Entity is placed FIRST and repeated 5x to dominate the embedding!
    This ensures entity-specific queries match correctly.
    """
    parts = []
    tool_lower = tool_id.lower()

    # ---
    # ENTITY FIRST! (repeated 5x to dominate embedding)
    # This is the MOST IMPORTANT differentiator
    # ---

    # Extract entity name first
    name = regex_module.sub(r'^(get|post|put|patch|delete)_', '', tool_id, flags=regex_module.IGNORECASE)
    for suffix in SUFFIX_MEANINGS_HR.keys():
        if name.lower().endswith(suffix.lower()):
            name = name[:-len(suffix)]
            break
    name = regex_module.sub(r'_id$', '', name, flags=regex_module.IGNORECASE)

    # Find entity and ADD IT FIRST (5x repetition for weight!)
    entity_value = ""
    name_lower = name.lower()
    for key, value in ENTITY_NAMES_HR.items():
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
    for suffix, meaning in sorted(SUFFIX_MEANINGS_HR.items(), key=lambda x: len(x[0]), reverse=True):
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
        parts.append(METHOD_MEANINGS_HR.get(method, ""))

    return " ".join(parts)


def build_embedding_text(tool_id: str, doc: dict) -> str:
    """
    Build text for embedding from tool documentation.

    Uses:
    1. Structural info (method, entity, suffix) - AUTO-GENERATED CROATIAN
    2. purpose (what it does) - CROATIAN
    3. when_to_use (use cases) - CROATIAN
    4. example_queries_hr (Croatian example queries) - CROATIAN

    IMPORTANT: Does NOT include tool_id (English) to avoid
    language mismatch with Croatian user queries.

    Does NOT use training_queries.json (UNRELIABLE)!
    """
    parts = []

    # NEW: Add structural info first (most important for differentiation)
    structural_info = extract_structural_info(tool_id)
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

    text = " ".join(parts)

    # Truncate if too long
    if len(text) > 8000:
        text = text[:8000]

    return text


async def generate_embeddings(
    tool_documentation: dict,
    existing_embeddings: dict
) -> dict:
    """Generate embeddings for all tools."""
    settings = get_settings()

    client = AsyncAzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION
    )

    embeddings = dict(existing_embeddings)

    # Find tools that need embeddings
    tools_to_embed = []
    for tool_id in tool_documentation:
        if tool_id not in embeddings:
            tools_to_embed.append(tool_id)

    if not tools_to_embed:
        logger.info("All tools already have embeddings")
        return embeddings

    logger.info(f"Generating embeddings for {len(tools_to_embed)} tools...")

    generated = 0
    errors = 0

    for i, tool_id in enumerate(tools_to_embed):
        doc = tool_documentation[tool_id]
        text = build_embedding_text(tool_id, doc)

        if not text.strip():
            logger.warning(f"Skipping {tool_id}: no text to embed")
            continue

        try:
            response = await client.embeddings.create(
                input=[text],
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            )
            embedding = response.data[0].embedding
            embeddings[tool_id] = embedding
            generated += 1

            # Progress update every 50 tools
            if generated % 50 == 0:
                logger.info(f"Progress: {generated}/{len(tools_to_embed)} embeddings generated")
                # Save intermediate progress
                save_embeddings(embeddings)

            # Rate limiting
            await asyncio.sleep(0.05)

        except Exception as e:
            logger.error(f"Failed to generate embedding for {tool_id}: {e}")
            errors += 1
            await asyncio.sleep(1)  # Longer delay on error

    logger.info(f"Generated {generated} embeddings, {errors} errors")
    return embeddings


async def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("TOOL EMBEDDINGS GENERATOR")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Using tool_documentation.json as source (ACCURATE)")
    logger.info("NOT using training_queries.json (UNRELIABLE)")
    logger.info("")

    # Load tool documentation
    tool_documentation = load_tool_documentation()

    # Load existing embeddings
    existing_embeddings = load_existing_embeddings()

    # Generate missing embeddings
    embeddings = await generate_embeddings(tool_documentation, existing_embeddings)

    # Save final embeddings
    save_embeddings(embeddings)

    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE!")
    logger.info(f"Total tools: {len(tool_documentation)}")
    logger.info(f"Total embeddings: {len(embeddings)}")
    logger.info(f"Cache file: {EMBEDDINGS_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

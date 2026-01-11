"""
Documentation Generator - Automated tool documentation using LLM.
Version: 2.0

Generates:
- config/tool_categories.json - Tool categorization (15-20 categories)
- config/tool_documentation.json - Detailed docs for each tool (includes example_queries_hr)
- config/knowledge_graph.json - Entity relationships

NOTE: training_queries.json is DEPRECATED (v4.0) - examples are now in tool_documentation.json

Usage:
    python -m scripts.generate_documentation

Estimated time: ~40 minutes for 900+ tools
Estimated cost: ~$5-10 (GPT-4)
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncAzureOpenAI, RateLimitError
from config import get_settings
from services.tool_contracts import UnifiedToolDefinition

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
settings = get_settings()

# Output directories
CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent / "data"

# Batch sizes for LLM calls
CATEGORY_BATCH_SIZE = 50  # Tools per categorization request
DOC_BATCH_SIZE = 10       # Tools per documentation request
TRAINING_EXAMPLES_PER_CATEGORY = 25


class DocumentationGenerator:
    """
    Generates rich documentation for all tools using LLM.
    Runs ONCE to create config files, not on every request.
    """

    def __init__(self):
        """Initialize with Azure OpenAI client."""
        self.client = AsyncAzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            max_retries=3,
            timeout=120.0
        )
        self.model = settings.AZURE_OPENAI_DEPLOYMENT_NAME

        # Statistics
        self.stats = {
            "llm_calls": 0,
            "tokens_used": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }

    async def generate_all(
        self,
        tools: Dict[str, UnifiedToolDefinition],
        delta_mode: bool = False,
        changed_tools: List[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point - generates all documentation.

        PILLAR 9: Supports delta mode for incremental updates.

        Args:
            tools: Dictionary of operation_id -> UnifiedToolDefinition
            delta_mode: If True, only process changed tools
            changed_tools: List of operation IDs that changed (for delta mode)

        Returns:
            Dict with categories, documentation, training_data, knowledge_graph
        """
        self.stats["start_time"] = datetime.now()

        # PILLAR 9: Delta mode - only process changed tools
        if delta_mode and changed_tools:
            tools_to_process = {
                op_id: tool for op_id, tool in tools.items()
                if op_id in changed_tools
            }
            logger.info(f"DELTA MODE: Processing {len(tools_to_process)} changed tools (skipping {len(tools) - len(tools_to_process)} unchanged)")
        else:
            tools_to_process = tools
            logger.info(f"FULL MODE: Processing all {len(tools)} tools")

        # Ensure output directories exist
        CONFIG_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        # PILLAR 9: Load existing documentation for delta merge
        existing_categories = self._load_existing_json(CONFIG_DIR / "tool_categories.json")
        existing_documentation = self._load_existing_json(CONFIG_DIR / "tool_documentation.json")

        # Step 1: Categorize tools (only new/changed in delta mode)
        logger.info("=" * 60)
        logger.info("STEP 1: Categorizing tools...")
        if delta_mode and existing_categories:
            # Only categorize changed tools, merge with existing
            new_categories = await self._categorize_tools(tools_to_process)
            categories = self._merge_categories(existing_categories, new_categories)
            logger.info(f"✅ Merged categories (updated {len(tools_to_process)} tools)")
        else:
            categories = await self._categorize_tools(tools)
            logger.info(f"✅ Created {len(categories.get('categories', {}))} categories")
        self._save_json(CONFIG_DIR / "tool_categories.json", categories)

        # Step 2: Generate documentation for each tool
        logger.info("=" * 60)
        logger.info("STEP 2: Generating tool documentation...")
        if delta_mode and existing_documentation:
            # Only document changed tools, merge with existing
            new_documentation = await self._generate_documentation(tools_to_process, categories)
            documentation = self._merge_documentation(existing_documentation, new_documentation, tools)
            logger.info(f"✅ Merged documentation (updated {len(new_documentation)} tools)")
        else:
            documentation = await self._generate_documentation(tools, categories)
            logger.info(f"✅ Documented {len(documentation)} tools")
        self._save_json(CONFIG_DIR / "tool_documentation.json", documentation)

        # Step 3: DEPRECATED - training_queries.json no longer used (v4.0)
        # Training examples are now in tool_documentation.json as example_queries_hr
        logger.info("=" * 60)
        logger.info("STEP 3: SKIPPED - training_queries.json deprecated (using tool_documentation.json)")

        # Step 4: Build knowledge graph
        logger.info("=" * 60)
        logger.info("STEP 4: Building knowledge graph...")
        knowledge_graph = await self._build_knowledge_graph(tools, categories)
        self._save_json(CONFIG_DIR / "knowledge_graph.json", knowledge_graph)
        logger.info("✅ Knowledge graph created")

        self.stats["end_time"] = datetime.now()
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        logger.info("=" * 60)
        logger.info("GENERATION COMPLETE")
        logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"LLM calls: {self.stats['llm_calls']}")
        logger.info(f"Tokens used: {self.stats['tokens_used']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info("=" * 60)

        return {
            "categories": categories,
            "documentation": documentation,
            "training_data": training_data,
            "knowledge_graph": knowledge_graph,
            "stats": self.stats
        }

    async def _categorize_tools(self, tools: Dict[str, UnifiedToolDefinition]) -> Dict:
        """
        Use LLM to categorize all tools into 15-20 categories.

        Batches tools into chunks for efficient processing.
        """
        # Prepare tool summaries
        tool_summaries = []
        for op_id, tool in tools.items():
            summary = {
                "id": op_id,
                "method": tool.method,
                "path": tool.path,
                "description": tool.description[:200] if tool.description else "",
                "params": [p.name for p in tool.parameters.values()][:5],
                "outputs": tool.output_keys[:5] if tool.output_keys else []
            }
            tool_summaries.append(summary)

        # Batch categorization
        all_suggestions = []
        batches = [tool_summaries[i:i + CATEGORY_BATCH_SIZE]
                   for i in range(0, len(tool_summaries), CATEGORY_BATCH_SIZE)]

        for batch_num, batch in enumerate(batches):
            logger.info(f"Categorizing batch {batch_num + 1}/{len(batches)} ({len(batch)} tools)")

            prompt = f"""Analiziraj ove API endpointe i predloži kategorije za svaki.

ENDPOINTI:
{json.dumps(batch, ensure_ascii=False, indent=2)}

Za svaki endpoint predloži jednu kategoriju (engleski, snake_case).
Kategorije trebaju biti:
- Opisne i razumljive (npr. "vehicle_info", "booking_management", "mileage_tracking")
- Grupirane po funkcionalnosti, ne po tehničkim detaljima
- Maksimalno 20 različitih kategorija za cijeli sustav

Vrati JSON array:
[
  {{"id": "operationId", "category": "category_name"}},
  ...
]

Samo JSON, bez objašnjenja."""

            result = await self._call_llm(prompt)
            if result:
                try:
                    parsed = json.loads(result)
                    all_suggestions.extend(parsed)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse categorization batch: {e}")
                    self.stats["errors"] += 1

        # Aggregate categories
        category_tools = {}
        tool_to_category = {}

        for suggestion in all_suggestions:
            cat = suggestion.get("category", "uncategorized")
            tool_id = suggestion.get("id", "")

            if cat not in category_tools:
                category_tools[cat] = []
            category_tools[cat].append(tool_id)
            tool_to_category[tool_id] = cat

        # Generate category metadata
        categories_result = {
            "categories": {},
            "tool_to_category": tool_to_category,
            "generated_at": datetime.now().isoformat()
        }

        for cat_name, cat_tools in category_tools.items():
            # Get sample tools for description
            sample_tools = cat_tools[:5]
            sample_info = []
            for tid in sample_tools:
                if tid in tools:
                    t = tools[tid]
                    sample_info.append(f"{tid}: {t.description[:50] if t.description else 'No desc'}")

            prompt = f"""Generiraj opis za kategoriju API endpointa.

Kategorija: {cat_name}
Broj alata: {len(cat_tools)}
Primjeri:
{chr(10).join(sample_info)}

Vrati JSON:
{{
  "name": "{cat_name}",
  "description_hr": "Opis na hrvatskom (1-2 rečenice)",
  "description_en": "English description (1-2 sentences)",
  "keywords_hr": ["ključna", "riječ"],
  "keywords_en": ["key", "words"],
  "typical_intents": ["WHAT_USER_WANTS_1", "WHAT_USER_WANTS_2"]
}}

Samo JSON."""

            meta_result = await self._call_llm(prompt)
            if meta_result:
                try:
                    meta = json.loads(meta_result)
                    meta["tools"] = cat_tools
                    meta["tool_count"] = len(cat_tools)
                    categories_result["categories"][cat_name] = meta
                except json.JSONDecodeError:
                    # Fallback - basic category
                    categories_result["categories"][cat_name] = {
                        "name": cat_name,
                        "description_hr": cat_name.replace("_", " ").title(),
                        "description_en": cat_name.replace("_", " ").title(),
                        "keywords_hr": [],
                        "keywords_en": [],
                        "typical_intents": [],
                        "tools": cat_tools,
                        "tool_count": len(cat_tools)
                    }

        return categories_result

    async def _document_single_tool(
        self,
        op_id: str,
        tool: UnifiedToolDefinition,
        category: str
    ) -> Optional[Dict]:
        """
        Fallback: Document a single tool when batch fails.
        Simpler prompt, more reliable JSON.
        """
        parameters_with_origin = {}
        for name, p in tool.parameters.items():
            parameters_with_origin[name] = {
                "type": p.param_type,
                "required": p.required,
                "description": p.description or "",
                "origin": p.dependency_source.value,
                "context_key": p.context_key if p.context_key else None
            }

        info = {
            "id": op_id,
            "method": tool.method,
            "path": tool.path,
            "description": tool.description or "",
            "category": category,
            "parameters": parameters_with_origin
        }

        prompt = f"""Generiraj dokumentaciju za ovaj API endpoint. Vrati SAMO JSON objekt.

ENDPOINT:
{json.dumps(info, ensure_ascii=False, indent=2)}

Vrati TOČNO ovaj format (JSON objekt, NE array):
{{
  "operation_id": "{op_id}",
  "purpose": "Kratki opis svrhe (hrvatski)",
  "when_to_use": ["Scenarij 1", "Scenarij 2"],
  "when_not_to_use": ["Kad ne koristiti"],
  "prerequisites": ["Preduvjet 1"],
  "common_errors": {{"400": "Nevaljani podaci", "404": "Nije pronađeno"}},
  "example_queries_hr": ["Primjer pitanja 1", "Primjer pitanja 2"],
  "parameter_origin_guide": {{}}
}}

SAMO JSON objekt, bez teksta prije ili poslije."""

        for attempt in range(3):
            result = await self._call_llm(prompt, max_tokens=1500)
            if result:
                try:
                    # Try to extract JSON from response
                    result = result.strip()
                    if result.startswith("```"):
                        result = result.split("```")[1]
                        if result.startswith("json"):
                            result = result[4:]
                    doc = json.loads(result)
                    doc["operation_id"] = op_id  # Ensure correct ID
                    return doc
                except json.JSONDecodeError as e:
                    logger.debug(f"Single tool {op_id} attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(1)

        # Final fallback - return minimal doc
        logger.warning(f"Failed to document {op_id} after 3 attempts, using minimal doc")
        return {
            "operation_id": op_id,
            "purpose": tool.description or f"{tool.method} {tool.path}",
            "when_to_use": [f"Za {category.replace('_', ' ')}"],
            "when_not_to_use": [],
            "prerequisites": [],
            "common_errors": {},
            "example_queries_hr": [],
            "parameter_origin_guide": {},
            "_auto_generated": True
        }

    async def _generate_documentation(
        self,
        tools: Dict[str, UnifiedToolDefinition],
        categories: Dict
    ) -> Dict[str, Any]:
        """
        Generate detailed documentation for each tool.

        Batches tools for efficient processing.
        Falls back to single-tool documentation when batch fails.
        """
        documentation = {}
        failed_tools = []  # Track tools that need retry
        tool_list = list(tools.items())
        batches = [tool_list[i:i + DOC_BATCH_SIZE]
                   for i in range(0, len(tool_list), DOC_BATCH_SIZE)]

        for batch_num, batch in enumerate(batches):
            logger.info(f"Documenting batch {batch_num + 1}/{len(batches)} ({len(batch)} tools)")

            # Track which tools are in this batch
            batch_tool_ids = [op_id for op_id, _ in batch]

            # Prepare batch info
            batch_info = []
            for op_id, tool in batch:
                category = categories.get("tool_to_category", {}).get(op_id, "uncategorized")

                # PILLAR 2: Include dependency_source for each parameter
                parameters_with_origin = {}
                for name, p in tool.parameters.items():
                    parameters_with_origin[name] = {
                        "type": p.param_type,
                        "required": p.required,
                        "description": p.description or "",
                        "origin": p.dependency_source.value,
                        "context_key": p.context_key if p.context_key else None
                    }

                info = {
                    "id": op_id,
                    "method": tool.method,
                    "path": tool.path,
                    "description": tool.description or "",
                    "category": category,
                    "parameters": parameters_with_origin,
                    "output_fields": tool.output_keys[:10] if tool.output_keys else []
                }
                batch_info.append(info)

            prompt = f"""Generiraj detaljnu dokumentaciju za ove API endpointe.

ENDPOINTI:
{json.dumps(batch_info, ensure_ascii=False, indent=2)}

VAŽNO - ORIGIN POLJA:
Svaki parametar ima "origin" polje koje označava odakle dolazi vrijednost:
- "context" = Sustav automatski ubacuje (tenant_id, person_id) - NE TRAŽI OD KORISNIKA
- "user_input" = Korisnik mora dati vrijednost - PITAJ AKO NEDOSTAJE
- "output" = Dolazi iz prethodnog API poziva

Za SVAKI endpoint vrati:
{{
  "operation_id": "...",
  "purpose": "Zašto ovaj endpoint postoji (hrvatski, 1 rečenica)",
  "when_to_use": ["Kada koristiti 1", "Kada koristiti 2"],
  "when_not_to_use": ["Kada NE koristiti"],
  "prerequisites": ["Što mora biti ispunjeno"],
  "output_fields_explained": {{"FieldName": "Što znači ovo polje"}},
  "common_errors": {{"400": "Opis", "403": "Opis", "404": "Opis"}},
  "next_steps": ["Što napraviti nakon uspješnog poziva"],
  "related_tools": ["povezani_tool_1", "povezani_tool_2"],
  "example_queries_hr": ["primjer pitanja 1", "primjer pitanja 2", "primjer 3"],
  "parameter_origin_guide": {{
    "ParameterName": "CONTEXT: Sustav ubacuje iz sesije" ili "USER: Pitaj korisnika za ovo" ili "OUTPUT: Dohvati iz prethodnog poziva",
    ...za svaki parametar
  }}
}}

Vrati JSON array svih endpointa. Samo JSON."""

            result = await self._call_llm(prompt, max_tokens=4000)
            batch_success = False
            if result:
                try:
                    docs = json.loads(result)
                    for doc in docs:
                        op_id = doc.get("operation_id", "")
                        if op_id:
                            documentation[op_id] = doc
                    batch_success = True
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse documentation batch: {e}")
                    self.stats["errors"] += 1

            # Check which tools from this batch are still missing
            missing_from_batch = [
                (op_id, tool) for op_id, tool in batch
                if op_id not in documentation
            ]

            if missing_from_batch:
                if not batch_success:
                    logger.info(f"Batch failed, falling back to single-tool for {len(missing_from_batch)} tools")
                else:
                    logger.info(f"Batch partial success, {len(missing_from_batch)} tools missing")

                # Fallback: document missing tools one by one
                for op_id, tool in missing_from_batch:
                    category = categories.get("tool_to_category", {}).get(op_id, "uncategorized")
                    doc = await self._document_single_tool(op_id, tool, category)
                    if doc:
                        documentation[op_id] = doc
                        logger.info(f"  ✓ Documented {op_id} via fallback")
                    else:
                        failed_tools.append(op_id)
                        logger.warning(f"  ✗ Failed to document {op_id}")

        # Final validation
        all_tool_ids = set(tools.keys())
        documented_ids = set(documentation.keys())
        still_missing = all_tool_ids - documented_ids

        if still_missing:
            logger.warning(f"COMPLETENESS CHECK: {len(still_missing)} tools still missing after all retries")
            self.stats["missing_tools"] = list(still_missing)
        else:
            logger.info(f"✅ COMPLETENESS CHECK: All {len(tools)} tools documented!")

        return documentation

    async def _generate_training_examples(
        self,
        tools: Dict[str, UnifiedToolDefinition],
        categories: Dict
    ) -> Dict[str, Any]:
        """
        Generate query→tool training examples for each category.

        PILLAR 7: Now includes CLARIFICATION examples where user doesn't
        provide all required info and bot must ask questions.
        """
        training_data = {
            "examples": [],
            "clarification_examples": [],  # NEW: Examples where bot asks for info
            "generated_at": datetime.now().isoformat(),
            "version": "2.0"  # Updated version for new format
        }

        category_list = list(categories.get("categories", {}).items())

        for cat_num, (cat_name, cat_info) in enumerate(category_list):
            logger.info(f"Generating examples for category {cat_num + 1}/{len(category_list)}: {cat_name}")

            # Get tools in this category
            cat_tools = cat_info.get("tools", [])[:10]  # Limit to 10 for prompt size

            tool_details = []
            for tid in cat_tools:
                if tid in tools:
                    t = tools[tid]
                    tool_details.append({
                        "id": tid,
                        "method": t.method,
                        "description": t.description[:100] if t.description else "",
                        "outputs": t.output_keys[:5] if t.output_keys else []
                    })

            if not tool_details:
                continue

            prompt = f"""Generiraj {TRAINING_EXAMPLES_PER_CATEGORY} primjera pitanja korisnika za ovu kategoriju API alata.

KATEGORIJA: {cat_name}
OPIS: {cat_info.get('description_hr', '')}

DOSTUPNI ALATI:
{json.dumps(tool_details, ensure_ascii=False, indent=2)}

Za svaki primjer:
- query: Pitanje na HRVATSKOM (različite formulacije! formalno i neformalno)
- intent: Što korisnik želi (UPPERCASE, engleski)
- primary_tool: Najbolji tool za ovo pitanje
- alternative_tools: Backup opcije ako primarni ne radi
- extract_fields: Koja polja izvući iz response-a
- response_template: Kratki predložak odgovora

VAŽNO:
- Uključi RAZLIČITE formulacije (formalno, neformalno, skraćeno)
- Uključi greške u pisanju (npr "kolko" umjesto "koliko")
- Uključi sinonime i različite načine postavljanja istog pitanja

Vrati JSON array:
[
  {{
    "query": "kolika mi je kilometraza",
    "intent": "GET_MILEAGE",
    "primary_tool": "get_MasterData",
    "alternative_tools": ["get_Mileage"],
    "extract_fields": ["Mileage"],
    "response_template": "📏 Kilometraža: {{Mileage}} km",
    "category": "{cat_name}"
  }},
  ...
]

Samo JSON array."""

            result = await self._call_llm(prompt, max_tokens=3000)
            if result:
                try:
                    examples = json.loads(result)
                    for ex in examples:
                        ex["category"] = cat_name
                    training_data["examples"].extend(examples)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse training examples for {cat_name}: {e}")
                    self.stats["errors"] += 1

        # PILLAR 7: Generate clarification examples
        logger.info("Generating clarification examples (PILLAR 7)...")
        clarification_examples = await self._generate_clarification_examples(tools)
        training_data["clarification_examples"] = clarification_examples

        return training_data

    async def _generate_clarification_examples(
        self,
        tools: Dict[str, UnifiedToolDefinition]
    ) -> List[Dict[str, Any]]:
        """
        PILLAR 7: Generate examples where bot asks for missing information.

        These teach the bot to ask ONE question at a time when user
        doesn't provide all required parameters.
        """
        # Find tools with user-required parameters (mutation tools are best candidates)
        candidate_tools = []
        for op_id, tool in tools.items():
            user_params = tool.get_user_params()
            required_user_params = [
                p for p in user_params.values()
                if p.required and p.dependency_source.value == "user_input"
            ]
            if required_user_params and tool.is_mutation:
                candidate_tools.append({
                    "id": op_id,
                    "method": tool.method,
                    "description": tool.description[:100] if tool.description else "",
                    "required_user_params": [
                        {"name": p.name, "description": p.description[:50] if p.description else ""}
                        for p in required_user_params[:5]
                    ]
                })

        if not candidate_tools:
            logger.info("No mutation tools with user params found for clarification examples")
            return []

        # Generate clarification examples
        prompt = f"""Generiraj primjere razgovora gdje korisnik NE daje sve podatke i bot mora pitati.

ALATI S OBAVEZNIM PARAMETRIMA:
{json.dumps(candidate_tools[:20], ensure_ascii=False, indent=2)}

Za svaki primjer:
1. Korisnik šalje nepotpunu poruku (npr. "rezerviraj vozilo" bez datuma)
2. Bot prepoznaje namjeru, ali traži JEDAN nedostajući parametar
3. ONE-QUESTION-AT-A-TIME princip - nikad ne pitaj za više stvari odjednom!

Generiraj 15-20 primjera u formatu:
[
  {{
    "incomplete_query": "što korisnik kaže (nepotpuno)",
    "intent": "INTENT_NAME",
    "detected_tool": "operation_id",
    "missing_param": "ime_parametra",
    "bot_question": "Jasno pitanje na hrvatskom za TAJ JEDAN parametar",
    "example_follow_up": "Primjer korisnikova odgovora"
  }},
  ...
]

VAŽNO:
- Pitanja moraju biti JASNA i JEDNOSTAVNA
- Samo JEDNO pitanje po interakciji
- Koristi prijateljski ton (npr. "Kada želite rezervirati?" umjesto "Unesite datum")
- Uključi različite scenarije (rezervacija, prijava štete, unos km, itd.)

Samo JSON array."""

        result = await self._call_llm(prompt, max_tokens=3000)

        if result:
            try:
                examples = json.loads(result)
                logger.info(f"Generated {len(examples)} clarification examples")
                return examples
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse clarification examples: {e}")
                self.stats["errors"] += 1

        # Fallback: Generate hardcoded examples for common scenarios
        return self._get_fallback_clarification_examples()

    def _get_fallback_clarification_examples(self) -> List[Dict[str, Any]]:
        """Fallback clarification examples if LLM fails."""
        return [
            {
                "incomplete_query": "rezerviraj vozilo",
                "intent": "CREATE_BOOKING",
                "detected_tool": "post_VehicleCalendar",
                "missing_param": "FromTime",
                "bot_question": "Od kada trebate vozilo? (npr. 'sutra u 9:00')",
                "example_follow_up": "sutra u 9 ujutro"
            },
            {
                "incomplete_query": "unesi kilometražu",
                "intent": "ADD_MILEAGE",
                "detected_tool": "post_AddMileage",
                "missing_param": "Value",
                "bot_question": "Kolika je trenutna kilometraža?",
                "example_follow_up": "45230"
            },
            {
                "incomplete_query": "prijavi štetu",
                "intent": "REPORT_DAMAGE",
                "detected_tool": "post_AddCase",
                "missing_param": "Message",
                "bot_question": "Možete li opisati što se dogodilo?",
                "example_follow_up": "Ogrebao sam branik na parkingu"
            },
            {
                "incomplete_query": "treba mi slobodno vozilo",
                "intent": "GET_AVAILABLE_VEHICLES",
                "detected_tool": "get_AvailableVehicles",
                "missing_param": "FromTime",
                "bot_question": "Za koji datum vam treba vozilo?",
                "example_follow_up": "za sutra"
            },
            {
                "incomplete_query": "moram otkazati rezervaciju",
                "intent": "CANCEL_BOOKING",
                "detected_tool": "delete_VehicleCalendar_id",
                "missing_param": "id",
                "bot_question": "Koju rezervaciju želite otkazati? (Mogu vam pokazati vaše aktivne rezervacije)",
                "example_follow_up": "onu za sutra"
            }
        ]

    async def _build_knowledge_graph(
        self,
        tools: Dict[str, UnifiedToolDefinition],
        categories: Dict
    ) -> Dict[str, Any]:
        """
        Build knowledge graph of entity relationships.
        """
        # Extract entity types from tools
        entity_hints = set()
        for op_id, tool in tools.items():
            # From parameters
            for param_name in tool.parameters.keys():
                if param_name.endswith("Id"):
                    entity = param_name[:-2]
                    entity_hints.add(entity)
            # From output keys
            for key in (tool.output_keys or []):
                if key.endswith("Id"):
                    entity = key[:-2]
                    entity_hints.add(entity)

        prompt = f"""Na temelju ovih entiteta iz API-ja, izgradi knowledge graph odnosa.

DETEKTIRANI ENTITETI:
{sorted(entity_hints)}

Za svaki glavni entitet definiraj:
- properties: Glavna svojstva
- relationships: Odnosi s drugim entitetima (format: relation_name -> TargetEntity)
- constraints: Poslovna pravila

Fokusiraj se na glavne entitete: Person, Vehicle, Booking, Tenant, Case, Registration

Vrati JSON:
{{
  "entities": {{
    "Person": {{
      "properties": ["PersonId", "Name", "Phone", "Email"],
      "relationships": {{
        "drives": "Vehicle",
        "has_bookings": "Booking",
        "works_for": "Tenant"
      }},
      "description": "Korisnik sustava (vozač ili admin)"
    }},
    ...
  }},
  "constraints": [
    {{"name": "booking_no_overlap", "description": "Vozilo ne može imati dvije rezervacije u isto vrijeme"}},
    ...
  ],
  "entity_resolution": {{
    "my_vehicle": "Vehicle assigned to current Person",
    "moje vozilo": "Vehicle assigned to current Person",
    ...
  }}
}}

Samo JSON."""

        result = await self._call_llm(prompt, max_tokens=2000)

        if result:
            try:
                knowledge_graph = json.loads(result)
                knowledge_graph["generated_at"] = datetime.now().isoformat()
                knowledge_graph["detected_entities"] = sorted(entity_hints)
                return knowledge_graph
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse knowledge graph: {e}")
                self.stats["errors"] += 1

        # Fallback - basic structure
        return {
            "entities": {
                "Person": {
                    "properties": ["PersonId", "Name", "Phone", "Email"],
                    "relationships": {"drives": "Vehicle", "has_bookings": "Booking"},
                    "description": "Korisnik sustava"
                },
                "Vehicle": {
                    "properties": ["VehicleId", "LicencePlate", "Name", "Mileage"],
                    "relationships": {"assigned_to": "Person", "has_bookings": "Booking"},
                    "description": "Vozilo u floti"
                },
                "Booking": {
                    "properties": ["BookingId", "FromTime", "ToTime", "Status"],
                    "relationships": {"for_vehicle": "Vehicle", "booked_by": "Person"},
                    "description": "Rezervacija vozila"
                }
            },
            "constraints": [],
            "entity_resolution": {},
            "detected_entities": sorted(entity_hints),
            "generated_at": datetime.now().isoformat()
        }

    async def _call_llm(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
        """
        Make LLM call with retry logic.
        """
        self.stats["llm_calls"] += 1

        messages = [
            {"role": "system", "content": "Ti si stručnjak za API dokumentaciju. Odgovaraj SAMO s validnim JSON-om."},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,  # Lower for more consistent output
                    max_tokens=max_tokens
                )

                if response.usage:
                    self.stats["tokens_used"] += response.usage.total_tokens

                if response.choices:
                    content = response.choices[0].message.content
                    # Clean up common JSON issues
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    return content.strip()

            except RateLimitError:
                wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
                logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue

            except Exception as e:
                logger.error(f"LLM call error: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(2)
                continue

        return None

    def _save_json(self, path: Path, data: Any):
        """Save data to JSON file with pretty printing."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {path}")

    def _load_existing_json(self, path: Path) -> Optional[Dict]:
        """
        PILLAR 9: Load existing JSON file for delta merge.

        Returns None if file doesn't exist or can't be loaded.
        """
        if not path.exists():
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")
            return None

    def _merge_categories(
        self,
        existing: Dict,
        new: Dict
    ) -> Dict:
        """
        PILLAR 9: Merge new categories with existing.

        - Updates tool_to_category mappings
        - Keeps existing categories, adds new ones
        """
        result = existing.copy()

        # Merge tool_to_category
        if "tool_to_category" not in result:
            result["tool_to_category"] = {}
        result["tool_to_category"].update(new.get("tool_to_category", {}))

        # Merge categories
        if "categories" not in result:
            result["categories"] = {}

        for cat_name, cat_data in new.get("categories", {}).items():
            if cat_name in result["categories"]:
                # Update existing category's tool list
                existing_tools = set(result["categories"][cat_name].get("tools", []))
                new_tools = set(cat_data.get("tools", []))
                result["categories"][cat_name]["tools"] = list(existing_tools | new_tools)
                result["categories"][cat_name]["tool_count"] = len(result["categories"][cat_name]["tools"])
            else:
                # Add new category
                result["categories"][cat_name] = cat_data

        result["generated_at"] = datetime.now().isoformat()
        result["merge_mode"] = "delta"

        return result

    def _merge_documentation(
        self,
        existing: Dict,
        new: Dict,
        all_tools: Dict[str, UnifiedToolDefinition]
    ) -> Dict:
        """
        PILLAR 9: Merge new documentation with existing.

        - Updates documentation for changed tools
        - Removes documentation for deleted tools
        - Keeps existing documentation for unchanged tools
        """
        result = existing.copy()

        # Update with new documentation
        result.update(new)

        # Add hash to each doc for future delta detection
        for op_id, doc in result.items():
            if op_id in all_tools:
                doc["_hash"] = all_tools[op_id].version_hash
                doc["_updated_at"] = datetime.now().isoformat()

        # Remove documentation for deleted tools
        current_tool_ids = set(all_tools.keys())
        deleted_tools = [
            op_id for op_id in result.keys()
            if op_id not in current_tool_ids and not op_id.startswith("_")
        ]

        for op_id in deleted_tools:
            del result[op_id]
            logger.info(f"Removed documentation for deleted tool: {op_id}")

        return result


async def load_tools_from_registry() -> Dict[str, UnifiedToolDefinition]:
    """
    Load tools from the registry (requires full initialization).
    """
    from services.registry import ToolRegistry
    from services.context_service import ContextService

    logger.info("Initializing registry...")

    # Create minimal context service for Redis
    try:
        context = ContextService()
        redis = context.redis
    except Exception:
        redis = None
        logger.warning("Redis not available, continuing without cache")

    registry = ToolRegistry(redis_client=redis)

    # Get swagger sources from settings
    swagger_sources = settings.swagger_sources

    if not swagger_sources:
        # Fallback to main API
        swagger_sources = [
            f"{settings.MOBILITY_API_URL.rstrip('/')}/swagger/v1/swagger.json"
        ]

    success = await registry.initialize(swagger_sources)

    if not success:
        raise RuntimeError("Failed to initialize registry")

    logger.info(f"Loaded {len(registry.tools)} tools from registry")
    return registry.tools


async def main():
    """
    Main entry point.

    Usage:
        python -m scripts.generate_documentation              # Full regeneration
        python -m scripts.generate_documentation --dry-run    # Check config only
        python -m scripts.generate_documentation --delta      # Delta mode (via swagger_watcher)
    """
    import argparse

    parser = argparse.ArgumentParser(description="Documentation Generator")
    parser.add_argument("--dry-run", action="store_true", help="Check configuration only")
    parser.add_argument("--delta", action="store_true", help="Delta mode - only process changed tools")
    parser.add_argument("--tools", nargs="*", help="Specific tools to regenerate (for delta mode)")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DOCUMENTATION GENERATOR")
    logger.info(f"Mode: {'DELTA' if args.delta else 'FULL'}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("Dry run mode - checking configuration only")
        logger.info(f"Azure endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
        logger.info(f"Model: {settings.AZURE_OPENAI_DEPLOYMENT_NAME}")
        logger.info(f"Output dirs: {CONFIG_DIR}, {DATA_DIR}")
        return

    try:
        # Load tools
        tools = await load_tools_from_registry()

        # Generate documentation
        generator = DocumentationGenerator()

        if args.delta:
            # PILLAR 9: Delta mode
            changed_tools = args.tools if args.tools else None

            if not changed_tools:
                # Auto-detect changes using swagger_watcher
                logger.info("Auto-detecting changes...")
                from scripts.swagger_watcher import SwaggerWatcher
                watcher = SwaggerWatcher()
                swagger_sources = settings.swagger_sources or [
                    f"{settings.MOBILITY_API_URL.rstrip('/')}/swagger/v1/swagger.json"
                ]
                await watcher.check_for_changes(swagger_sources)
                changed_tools = watcher.changes["new"] + watcher.changes["modified"]

                if not changed_tools:
                    logger.info("No changes detected - nothing to regenerate")
                    return

                logger.info(f"Detected {len(changed_tools)} changed tools")

            result = await generator.generate_all(
                tools,
                delta_mode=True,
                changed_tools=changed_tools
            )
        else:
            # Full mode
            result = await generator.generate_all(tools)

        logger.info("Documentation generation completed successfully!")

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

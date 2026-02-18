"""
Embedding Engine - Generate and manage embeddings for tool discovery.
Version: 4.0

Single responsibility: Generate embeddings using Azure OpenAI.

ARCHITECTURE (v4.0):
    Croatian language dictionaries are loaded from config/croatian_mappings.json
    (single source of truth) instead of being hardcoded in this file.

    Primary embedding source: LLM-generated Croatian descriptions from
    tool_documentation.json (via scripts/generate_croatian_descriptions.py).
    Fallback: Dictionary-based generation from croatian_mappings.json.

    Three dictionaries (loaded from config):
    1. PATH_ENTITY_MAP - Maps English path segments to Croatian
    2. OUTPUT_KEY_MAP - Maps output field names to Croatian
    3. CROATIAN_SYNONYMS - Maps Croatian roots to user query alternatives
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

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


def _load_croatian_mappings() -> dict:
    """Load Croatian mappings from config/croatian_mappings.json."""
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(base_path, "config", "croatian_mappings.json")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded Croatian mappings from {config_path}")
        return data
    except Exception as e:
        logger.error(f"Could not load Croatian mappings: {e}. Using empty defaults.")
        return {}


# Module-level cache for Croatian mappings (loaded once)
_croatian_mappings_cache: Optional[dict] = None


def _get_croatian_mappings() -> dict:
    """Get cached Croatian mappings (loads on first access)."""
    global _croatian_mappings_cache
    if _croatian_mappings_cache is None:
        _croatian_mappings_cache = _load_croatian_mappings()
    return _croatian_mappings_cache


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

        # Load dictionaries from config/croatian_mappings.json
        mappings = _get_croatian_mappings()

        # Convert JSON arrays back to tuples for PATH_ENTITY_MAP
        raw_path_map = mappings.get("path_entity_map", {})
        self.PATH_ENTITY_MAP: Dict[str, Tuple[str, str]] = {}
        for key, value in raw_path_map.items():
            if key == "_comments":
                continue
            if isinstance(value, list) and len(value) == 2:
                self.PATH_ENTITY_MAP[key] = (value[0], value[1])

        # OUTPUT_KEY_MAP: string -> string (no conversion needed)
        raw_output_map = mappings.get("output_key_map", {})
        self.OUTPUT_KEY_MAP: Dict[str, str] = {
            k: v for k, v in raw_output_map.items() if k != "_comments"
        }

        # CROATIAN_SYNONYMS: string -> list of strings (no conversion needed)
        raw_synonyms = mappings.get("croatian_synonyms", {})
        self.CROATIAN_SYNONYMS: Dict[str, List[str]] = {
            k: v for k, v in raw_synonyms.items() if k != "_comments"
        }

        logger.info(
            f"EmbeddingEngine initialized: "
            f"{len(self.PATH_ENTITY_MAP)} path mappings, "
            f"{len(self.OUTPUT_KEY_MAP)} output mappings, "
            f"{len(self.CROATIAN_SYNONYMS)} synonym groups"
        )

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
        Build embedding text with Croatian description.

        v4.0: Uses LLM-generated Croatian descriptions as PRIMARY source
        (from tool_documentation.json). Falls back to dictionary-based
        generation only if no LLM description exists.

        Strategy:
        1. Check for LLM-generated Croatian description (best quality)
        2. Fall back to auto-generated purpose from dictionaries
        3. Add synonyms for query matching
        """
        # 1. Try LLM-generated Croatian description (Phase 4 output)
        croatian_desc = self._get_llm_croatian_description(operation_id)
        if croatian_desc:
            # Use LLM description as primary, still add synonyms
            parts = [croatian_desc]

            synonyms = self._get_synonyms_for_purpose(croatian_desc)
            if synonyms:
                parts.append(f"Sinonimi: {', '.join(synonyms)}")

            text = ". ".join(p for p in parts if p)
            if len(text) > 1500:
                text = text[:1500]
            return text

        # 2. Fallback: Auto-generate purpose from structure (dictionary-based)
        purpose = self._generate_purpose(method, parameters, output_keys, path, operation_id)

        parts = [
            purpose,
            description if description else "",
        ]

        if output_keys:
            readable = [
                re.sub(r'([a-z])([A-Z])', r'\1 \2', k)
                for k in output_keys[:10]
            ]
            parts.append(f"Returns: {', '.join(readable)}")

        synonyms = self._get_synonyms_for_purpose(purpose)
        if synonyms:
            parts.append(f"Sinonimi: {', '.join(synonyms)}")

        text = ". ".join(p for p in parts if p)

        if len(text) > 1500:
            text = text[:1500]

        return text

    def _get_llm_croatian_description(self, operation_id: str) -> Optional[str]:
        """
        Get LLM-generated Croatian description from tool_documentation.json.

        Returns the croatian_embedding_text or croatian_description field
        if available, None otherwise.
        """
        try:
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            doc_path = os.path.join(base_path, "config", "tool_documentation.json")

            if not hasattr(self, '_tool_docs_cache'):
                if os.path.exists(doc_path):
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        self._tool_docs_cache = json.load(f)
                else:
                    self._tool_docs_cache = {}

            doc = self._tool_docs_cache.get(operation_id, {})
            # Prefer embedding_text (optimized for search), fall back to description
            return doc.get("croatian_embedding_text") or doc.get("croatian_description")
        except Exception:
            return None

    # Dictionaries loaded from config/croatian_mappings.json in __init__
    # PATH_ENTITY_MAP: Dict[str, Tuple[str, str]] - English path → (nominative, genitive)
    # OUTPUT_KEY_MAP: Dict[str, str] - output key → Croatian description
    # CROATIAN_SYNONYMS: Dict[str, List[str]] - Croatian root → alternative terms

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

    # Common API prefixes to skip (not meaningful for embedding)
    SKIP_SEGMENTS = {
        "api", "v1", "v2", "v3", "v4", "odata", "rest", "public", "private",
        "internal", "external", "admin", "management", "system", "core",
    }

    def _extract_entities_from_path(self, path: str) -> List[str]:
        """
        Extract entities from API path segments.

        Uses Croatian mapping when available, falls back to English
        (with space-separated camelCase) for unmapped terms.
        This ensures ALL paths contribute to embedding quality.
        """
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

            # Skip common API prefixes
            if segment in self.SKIP_SEGMENTS:
                continue

            # Check against entity map (Croatian translation available)
            if segment in self.PATH_ENTITY_MAP:
                singular, _ = self.PATH_ENTITY_MAP[segment]
                if singular not in entities:
                    entities.append(singular)
            else:
                # Try partial match for compound words
                found = False
                for key, (singular, _) in self.PATH_ENTITY_MAP.items():
                    if key in segment and singular not in entities:
                        entities.append(singular)
                        found = True
                        break

                # FALLBACK: Use English term with readable formatting
                # This ensures unmapped terms still contribute to embedding
                if not found and segment not in entities:
                    # Convert camelCase/compound to readable: "vehicleinfo" -> "vehicle info"
                    readable = self._make_readable(segment)
                    if readable not in entities:
                        entities.append(readable)

        return entities[:4]  # Increased limit for fallback terms

    def _make_readable(self, term: str) -> str:
        """
        Convert technical term to human-readable format.

        Examples:
            vehicleinfo -> vehicle info
            fuelconsumption -> fuel consumption
            getbyid -> get by id
        """
        # Insert space before uppercase letters (camelCase)
        readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', term)
        # Insert space between letters and numbers
        readable = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', readable)
        readable = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', readable)
        return readable.lower()

    def _parse_operation_id(self, operation_id: str) -> tuple:
        """
        Parse operationId to extract action and entities.

        Uses Croatian mapping when available, falls back to English
        for unmapped terms to ensure all operation IDs contribute.
        """
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
                        "set", "fetch", "retrieve", "check", "validate",
                        "by", "for", "all", "id", "ids", "the", "and", "or"}

        for word in words:
            word_lower = word.lower()

            if word_lower in action_verbs or len(word_lower) < 3:
                continue

            # Check if word maps to an entity (Croatian translation)
            if word_lower in self.PATH_ENTITY_MAP:
                singular, _ = self.PATH_ENTITY_MAP[word_lower]
                if singular not in entities:
                    entities.append(singular)
            # Check output key map for action hints
            elif word_lower in self.OUTPUT_KEY_MAP:
                if not action_hint:
                    action_hint = self.OUTPUT_KEY_MAP[word_lower]
            # FALLBACK: Use English word as-is (readable format)
            # This ensures unmapped operation IDs still contribute
            elif word_lower not in entities:
                entities.append(word_lower)

        return entities[:3], action_hint  # Increased limit for fallback

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

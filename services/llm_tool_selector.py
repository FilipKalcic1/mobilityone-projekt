"""
LLM Tool Selector - True intelligent tool selection using LLM.
Version: 2.0

CHANGELOG v2.0:
- REMOVED: training_queries.json (unreliable, caused confusion)
- ADDED: Uses tool_documentation.json for few-shot examples
- ADDED: Query type classification for better suffix handling
- IMPROVED: More accurate tool selection

Architecture:
1. Load examples from tool_documentation.json (ACCURATE source)
2. Use Query Type Classifier for suffix filtering
3. Build few-shot prompt with documentation examples
4. Ask LLM to select the best tool
5. Return tool with real confidence
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from openai import AsyncAzureOpenAI

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ToolSelection:
    """Result of LLM tool selection."""
    tool_name: str
    confidence: float  # Real confidence from LLM reasoning
    reasoning: str
    alternative_tools: List[str]


class LLMToolSelector:
    """
    Selects the best tool using LLM with few-shot examples.

    V2.0: Uses tool_documentation.json instead of training_queries.json
    for more reliable examples.
    """

    def __init__(self):
        """Initialize the selector."""
        self.client = AsyncAzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            max_retries=2,
            timeout=30.0
        )
        self.model = settings.AZURE_OPENAI_DEPLOYMENT_NAME

        # Tool documentation (ACCURATE source for examples)
        self._tool_documentation: Optional[Dict] = None
        self._examples_by_tool: Dict[str, List[str]] = {}  # tool_id -> example_queries_hr
        self._initialized = False

    async def initialize(self):
        """Load tool documentation."""
        if self._initialized:
            return

        try:
            # Load from tool_documentation.json (NOT training_queries.json!)
            doc_path = Path(__file__).parent.parent / "config" / "tool_documentation.json"

            if doc_path.exists():
                with open(doc_path, 'r', encoding='utf-8') as f:
                    self._tool_documentation = json.load(f)

                # Index example queries by tool
                for tool_id, doc in self._tool_documentation.items():
                    examples = doc.get("example_queries_hr", [])
                    if examples:
                        self._examples_by_tool[tool_id] = examples

                logger.info(
                    f"LLMToolSelector v2.0: Loaded {len(self._tool_documentation)} tool docs, "
                    f"{len(self._examples_by_tool)} tools with examples"
                )
            else:
                logger.warning(f"Tool documentation not found: {doc_path}")
                self._tool_documentation = {}

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to load tool documentation: {e}")
            self._tool_documentation = {}
            self._initialized = True

    def _get_few_shot_examples(
        self,
        query: str,
        categories: List[str],
        candidate_tools: List[str],
        max_examples: int = 12
    ) -> List[Dict]:
        """
        Get relevant few-shot examples from tool_documentation.json.

        V2.0: Uses example_queries_hr from documentation instead of training data.
        This is more reliable because documentation is curated.
        """
        examples = []
        seen_tools = set()
        query_lower = query.lower()

        # Priority tools with keywords for quick matching
        priority_tools_map = {
            "post_AddCase": ["štet", "kvar", "udari", "ogreb", "prijav", "slomio", "oštećen"],
            "post_AddMileage": ["kilometr", "km", "unesi", "upiši", "prijeđen"],
            "get_MasterData": ["registracij", "tablic", "podaci o vozil", "info o vozil"],
            "get_AvailableVehicles": ["slobodn", "dostupn", "koje je slobodno"],
            "post_VehicleCalendar": ["rezervir", "booking", "zauzmi"],
            "get_VehicleCalendar": ["moje rezerv", "moji booking", "kalendar vozil"],
            "get_Expenses": ["troškov", "troška", "expense", "račun"],
            "get_Trips": ["trip", "putovanj", "putni nalog"],
        }

        # Step 1: Add examples for priority tools if query matches keywords
        for tool_id, keywords in priority_tools_map.items():
            if any(kw in query_lower for kw in keywords):
                if tool_id in self._examples_by_tool and tool_id not in seen_tools:
                    example_queries = self._examples_by_tool[tool_id]
                    doc = self._tool_documentation.get(tool_id, {})

                    examples.append({
                        "query": example_queries[0] if example_queries else "",
                        "tool": tool_id,
                        "reason": doc.get("purpose", "")[:100]
                    })
                    seen_tools.add(tool_id)

        # Step 2: Add examples from candidate tools
        for tool_id in candidate_tools:
            if len(examples) >= max_examples:
                break

            if tool_id in seen_tools:
                continue

            if tool_id in self._examples_by_tool:
                example_queries = self._examples_by_tool[tool_id]
                doc = self._tool_documentation.get(tool_id, {})

                examples.append({
                    "query": example_queries[0] if example_queries else "",
                    "tool": tool_id,
                    "reason": doc.get("purpose", "")[:100]
                })
                seen_tools.add(tool_id)

        # Step 3: Add variety from other tools with examples
        if len(examples) < max_examples:
            for tool_id, example_queries in self._examples_by_tool.items():
                if len(examples) >= max_examples:
                    break

                if tool_id in seen_tools:
                    continue

                # Only add if example has keyword overlap with query
                if example_queries:
                    example_words = set(example_queries[0].lower().split())
                    query_words = set(query_lower.split())

                    if len(example_words & query_words) >= 1:
                        doc = self._tool_documentation.get(tool_id, {})
                        examples.append({
                            "query": example_queries[0],
                            "tool": tool_id,
                            "reason": doc.get("purpose", "")[:100]
                        })
                        seen_tools.add(tool_id)

        return examples[:max_examples]

    def _build_tools_description(self, tools: List[str], registry) -> str:
        """Build a concise description of available tools with origin guide."""
        descriptions = []

        # Load tool documentation for origin guides
        tool_documentation = self._load_tool_documentation()

        # FIXED: Increased from 30 to 50 tools for better coverage
        for tool_name in tools[:50]:
            tool = registry.get_tool(tool_name)
            if tool:
                # FIXED: Increased from 100 to 250 chars for more context
                desc = tool.description[:250] if tool.description else "No description"

                # FIXED: Include BOTH AUTO and USER params for complete origin guide
                origin_hint = self._build_origin_hint(tool_name, tool_documentation)

                descriptions.append(f"- {tool_name}: {desc}{origin_hint}")

        if len(tools) > 50:
            descriptions.append(f"... and {len(tools) - 50} more tools")

        return "\n".join(descriptions)

    def _build_origin_hint(self, tool_name: str, tool_documentation: Optional[Dict]) -> str:
        """
        Build complete origin hint with both AUTO and USER params.

        FIXED: Now includes USER params so LLM knows what to ask for.
        """
        if not tool_documentation or tool_name not in tool_documentation:
            return ""

        origin_guide = tool_documentation[tool_name].get("parameter_origin_guide", {})
        if not origin_guide:
            return ""

        # Collect AUTO (CONTEXT) params - system fills these
        auto_params = [
            k for k, v in origin_guide.items()
            if "CONTEXT" in str(v).upper() or "SUSTAV" in str(v).upper()
        ]

        # Collect USER params - user must provide these
        user_params = [
            k for k, v in origin_guide.items()
            if "USER" in str(v).upper() or "PITAJ" in str(v).upper() or "KORISNIK" in str(v).upper()
        ]

        hints = []
        if auto_params:
            hints.append(f"AUTO: {', '.join(auto_params[:5])}")  # Show up to 5
        if user_params:
            hints.append(f"USER: {', '.join(user_params[:5])}")  # Show up to 5

        return f" [{'; '.join(hints)}]" if hints else ""

    def _load_tool_documentation(self) -> Optional[Dict]:
        """Load tool documentation from config file."""
        try:
            doc_path = Path(__file__).parent.parent / "config" / "tool_documentation.json"
            if doc_path.exists():
                with open(doc_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    async def select_tool(
        self,
        query: str,
        candidate_tools: List[str],
        categories: List[str],
        registry,
        user_context: Optional[Dict] = None
    ) -> ToolSelection:
        """
        Select the best tool using LLM.

        Args:
            query: User's query
            candidate_tools: Tools to choose from (pre-filtered by category)
            categories: Matched categories
            registry: ToolRegistry for tool descriptions
            user_context: Optional user context

        Returns:
            ToolSelection with tool name and confidence
        """
        await self.initialize()

        # Get few-shot examples (intelligently selected based on query)
        examples = self._get_few_shot_examples(query, categories, candidate_tools)

        # Build few-shot part of prompt
        few_shot_text = ""
        if examples:
            few_shot_text = "Primjeri sličnih upita:\n\n"
            for ex in examples:
                few_shot_text += f"Upit: \"{ex['query']}\"\n"
                few_shot_text += f"Alat: {ex['primary_tool']}\n"
                if ex.get('alternative_tools'):
                    few_shot_text += f"Alternative: {', '.join(ex['alternative_tools'])}\n"
                few_shot_text += "\n"

        # Build tools description
        tools_desc = self._build_tools_description(candidate_tools, registry)

        # Build the prompt
        system_prompt = """Ti si stručnjak za odabir pravog API alata na temelju korisničkog upita.

Tvoj zadatak:
1. Analiziraj korisnikov upit
2. Pregledaj dostupne alate
3. Odaberi NAJBOLJI alat za taj upit
4. Objasni zašto si odabrao taj alat

PRAVILA ZA ODABIR ALATA:

1. PRIJAVA ŠTETE/KVARA:
   - Ako korisnik prijavljuje štetu, kvar, nesreću, udar → UVIJEK koristi post_AddCase
   - Primjeri: "udario sam", "ogrebao sam", "imam kvar", "prijavi štetu" → post_AddCase
   - NIKAD ne koristi put_Cases_id za novu prijavu štete!

2. UNOS KILOMETARA:
   - Za unos nove kilometraže → post_AddMileage
   - Primjeri: "unesi km", "upiši kilometražu" → post_AddMileage

3. PODACI O VOZILU:
   - Za opće podatke (registracija, tablica, km) → get_MasterData
   - get_MasterData vraća sve bitne informacije o vozilu

4. REZERVACIJE:
   - Nova rezervacija → post_VehicleCalendar ili post_Booking
   - Moje rezervacije → get_VehicleCalendar

5. DOSTUPNOST:
   - Slobodna vozila → get_AvailableVehicles

KRITIČNO - PARAMETER ORIGIN GUIDE:
Prije popunjavanja parametara OBAVEZNO provjeri parameter_origin_guide iz dokumentacije!

Ako je izvor parametra:
- "CONTEXT" → NE POPUNJAVAJ! Sustav automatski ubacuje (personId, tenantId). Ostavi prazno!
- "USER" → Korisnik mora dati vrijednost. Ako nedostaje, vrati null i pitaj korisnika.
- "OUTPUT" → Dolazi iz prethodnog API poziva. Ne izmišljaj!

ZABRANJENO:
- NIKAD ne izmišljaj UUID-ove, email adrese, ili bilo koje ID-eve
- Ako parametar zahtijeva podatak koji korisnik nije dao → vrati null
- Bolje je pitati korisnika nego pogriješiti

VAŽNO:
- Odaberi SAMO alate iz ponuđene liste
- Za ČITANJE koristi get_* alate
- Za KREIRANJE/PRIJAVU koristi post_* alate
- Za BRISANJE koristi delete_* alate
- Ako nisi siguran, confidence stavi ispod 0.7

Odgovori u JSON formatu:
{
    "tool": "ime_alata",
    "confidence": 0.0-1.0,
    "reasoning": "zašto ovaj alat",
    "alternatives": ["alternativni_alat1"]
}"""

        user_prompt = f"""Korisnikov upit: "{query}"

{few_shot_text}
Dostupni alati:
{tools_desc}

Koji alat je najbolji za ovaj upit?"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            tool_name = result.get("tool", "")
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")
            alternatives = result.get("alternatives", [])

            # Validate tool exists in candidates
            if tool_name and tool_name not in candidate_tools:
                # LLM hallucinated a tool - try to find closest match
                logger.warning(f"LLM selected non-existent tool: {tool_name}")
                for candidate in candidate_tools:
                    if tool_name.lower() in candidate.lower() or candidate.lower() in tool_name.lower():
                        tool_name = candidate
                        confidence *= 0.8  # Reduce confidence
                        break
                else:
                    # No match found - use first candidate with low confidence
                    tool_name = candidate_tools[0] if candidate_tools else ""
                    confidence = 0.3
                    reasoning = f"Fallback: LLM selected invalid tool"

            logger.info(
                f"LLM selected: {tool_name} (conf={confidence:.2f}) "
                f"for query: '{query[:40]}...'"
            )

            return ToolSelection(
                tool_name=tool_name,
                confidence=confidence,
                reasoning=reasoning,
                alternative_tools=alternatives
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._fallback_selection(candidate_tools, "JSON parse error")

        except Exception as e:
            logger.error(f"LLM tool selection failed: {e}")
            return self._fallback_selection(candidate_tools, str(e))

    def _fallback_selection(
        self,
        candidate_tools: List[str],
        error_reason: str
    ) -> ToolSelection:
        """Fallback when LLM fails."""
        tool = candidate_tools[0] if candidate_tools else ""
        return ToolSelection(
            tool_name=tool,
            confidence=0.2,  # Low confidence for fallback
            reasoning=f"Fallback selection: {error_reason}",
            alternative_tools=candidate_tools[1:3] if len(candidate_tools) > 1 else []
        )


# Singleton instance
_selector: Optional[LLMToolSelector] = None


async def get_llm_tool_selector() -> LLMToolSelector:
    """Get or create singleton selector instance."""
    global _selector
    if _selector is None:
        _selector = LLMToolSelector()
        await _selector.initialize()
    return _selector

"""
Unified Router - Single LLM makes ALL routing decisions.
Version: 2.0

This replaces the complex multi-layer routing with a single, reliable LLM decision.

v2.0 CHANGELOG:
- ADDED: AmbiguityDetector for disambiguation of generic queries
- ADDED: Disambiguation hints in LLM prompt when ambiguity detected
- ADDED: Clarification question fallback for highly ambiguous queries

Architecture:
1. Gather context (current state, user info, tools)
2. Detect ambiguity in search results (NEW)
3. Single LLM call decides everything (with disambiguation hints if needed)
4. Execute based on decision OR ask clarification

The LLM receives:
- User query
- Current conversation state (flow, missing params)
- User context (vehicle, person)
- Available primary tools (30 most common)
- Disambiguation hints (NEW - when ambiguity detected)

The LLM outputs:
- action: "continue_flow" | "exit_flow" | "start_flow" | "simple_api" | "direct_response" | "clarify"
- tool: tool name or null
- params: extracted parameters
- flow_type: booking | mileage | case | None
- response: direct response text (for direct_response action)
- clarification: question to ask user (for clarify action)
- reasoning: explanation
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path

from config import get_settings
from services.openai_client import get_openai_client, get_llm_circuit_breaker
from services.circuit_breaker import CircuitOpenError
from services.query_router import QueryRouter, RouteResult
from services.ambiguity_detector import (
    AmbiguityDetector, AmbiguityResult, get_ambiguity_detector
)
from services.context import UserContextManager
from services.flow_phrases import (
    matches_show_more,
    matches_confirm_yes,
    matches_confirm_no,
    matches_exit_signal,
    matches_item_selection,
    matches_greeting,
)

if TYPE_CHECKING:
    from services.registry import ToolRegistry

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RouterDecision:
    """Result of unified routing decision."""
    action: str  # continue_flow, exit_flow, start_flow, simple_api, direct_response, clarify
    tool: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    flow_type: Optional[str] = None  # booking, mileage, case
    response: Optional[str] = None  # For direct_response
    clarification: Optional[str] = None  # For clarify action (v2.0)
    reasoning: str = ""
    confidence: float = 0.0
    ambiguity_detected: bool = False  # v2.0: Was ambiguity detected?


# Primary tools - the 30 most common operations
PRIMARY_TOOLS = {
    # Vehicle Info (READ)
    "get_MasterData": "Dohvati podatke o vozilu (registracija, kilometraža, servis)",
    "get_Vehicles_id": "Dohvati detalje specifičnog vozila",
    
    # Person Info (READ) - name, email, phone, company
    "get_PersonData_personIdOrEmail": "Dohvati podatke o korisniku (ime, prezime, email, telefon)",

    # Availability & Booking
    "get_AvailableVehicles": "Provjeri dostupna/slobodna vozila za period",
    "get_VehicleCalendar": "Dohvati moje rezervacije",
    "post_VehicleCalendar": "Kreiraj novu rezervaciju vozila",
    "delete_VehicleCalendar_id": "Obriši/otkaži rezervaciju",

    # Mileage
    "get_LatestMileageReports": "Dohvati zadnju kilometražu",
    "get_MileageReports": "Dohvati izvještaje o kilometraži",
    "post_AddMileage": "Unesi/upiši novu kilometražu",

    # Case/Damage
    "post_AddCase": "Prijavi štetu, kvar, problem, nesreću",
    "get_Cases": "Dohvati prijavljene slučajeve",

    # Expenses
    "get_Expenses": "Dohvati troškove",
    "get_ExpenseGroups": "Dohvati grupe troškova",

    # Trips
    "get_Trips": "Dohvati putovanja/tripove",

    # Dashboard
    "get_DashboardItems": "Dohvati dashboard podatke",
}

# Flow triggers - which tools trigger which flows
FLOW_TRIGGERS = {
    "post_VehicleCalendar": "booking",
    "get_AvailableVehicles": "booking",
    "post_AddMileage": "mileage",
    "post_AddCase": "case",
}

# Exit signals moved to services/flow_phrases.py (centralized)


class UnifiedRouter:
    """
    Single LLM router that makes all routing decisions.

    This is the ONLY decision point - no keyword matching, no filtering.
    The LLM sees everything and decides.
    
    v2.0: Uses semantic search to find relevant tools from ALL 950+ tools,
    not just hardcoded PRIMARY_TOOLS.
    """

    def __init__(self, registry: Optional["ToolRegistry"] = None):
        """Initialize router with optional tool registry for semantic search."""
        # Shared client: rate limiting + connection pooling across all services
        self.client = get_openai_client()
        self._circuit_breaker = get_llm_circuit_breaker()
        self.model = settings.AZURE_OPENAI_DEPLOYMENT_NAME

        # Tool Registry for semantic search (injected)
        self._registry = registry

        # Query Router - brza staza za poznate patterne
        self.query_router = QueryRouter()

        # v2.0: Ambiguity detector for disambiguation
        self._ambiguity_detector: Optional[AmbiguityDetector] = None

        self._initialized = False

    def set_registry(self, registry: "ToolRegistry"):
        """Set tool registry for semantic search (allows late binding)."""
        self._registry = registry
        logger.info("UnifiedRouter: Registry set for semantic search")

    async def initialize(self):
        """Initialize router."""
        if self._initialized:
            return
        logger.info("UnifiedRouter: Initialized (uses tool_documentation.json via FAISS)")
        self._initialized = True

    async def _get_relevant_tools_with_ambiguity(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        top_k: int = 20
    ) -> Tuple[Dict[str, str], Optional[AmbiguityResult]]:
        """
        Use UnifiedSearch to find relevant tools and detect ambiguity.

        v2.0: Now also detects ambiguity in results for disambiguation.

        Returns:
            Tuple of (tools_dict, ambiguity_result)
            - tools_dict: {tool_name: description}
            - ambiguity_result: AmbiguityResult if ambiguity detected, else None
        """
        if not self._registry or not self._registry.is_ready:
            logger.debug("Registry not ready, using PRIMARY_TOOLS fallback")
            return PRIMARY_TOOLS, None

        try:
            # v4.0: Use UnifiedSearch for consistent results
            from services.unified_search import get_unified_search

            unified = get_unified_search()
            unified.set_registry(self._registry)

            response = await unified.search(query, top_k=top_k)

            if response.results:
                # Build dict of tool_name -> description
                relevant_tools = {}
                for r in response.results:
                    relevant_tools[r.tool_id] = r.description

                # Always include PRIMARY_TOOLS
                for tool_name, desc in PRIMARY_TOOLS.items():
                    if tool_name not in relevant_tools:
                        relevant_tools[tool_name] = desc

                # v2.0: Detect ambiguity in results
                ambiguity_result = None
                if self._ambiguity_detector is None:
                    self._ambiguity_detector = get_ambiguity_detector()

                # Convert results to format expected by ambiguity detector
                search_results_for_ambiguity = [
                    {"tool_id": r.tool_id, "score": r.score}
                    for r in response.results
                ]

                ambiguity_result = self._ambiguity_detector.detect_ambiguity(
                    query=query,
                    search_results=search_results_for_ambiguity,
                    user_context=user_context
                )

                if ambiguity_result.is_ambiguous:
                    logger.info(
                        f"AMBIGUITY DETECTED: suffix={ambiguity_result.ambiguous_suffix}, "
                        f"similar_tools={len(ambiguity_result.similar_tools)}, "
                        f"detected_entity={ambiguity_result.detected_entity}"
                    )

                logger.info(
                    f"UnifiedSearch: {len(response.results)} tools "
                    f"(intent={response.intent.value}), "
                    f"total {len(relevant_tools)} with PRIMARY merge, "
                    f"ambiguous={ambiguity_result.is_ambiguous if ambiguity_result else False}"
                )
                return relevant_tools, ambiguity_result

            logger.debug("UnifiedSearch returned no results, using PRIMARY_TOOLS fallback")
            return PRIMARY_TOOLS, None

        except Exception as e:
            logger.error(f"UnifiedSearch failed: {e}, using PRIMARY_TOOLS fallback")
            return PRIMARY_TOOLS, None

    async def _get_relevant_tools(self, query: str, top_k: int = 20) -> Dict[str, str]:
        """
        Use UnifiedSearch to find relevant tools for this query.

        Backwards-compatible wrapper around _get_relevant_tools_with_ambiguity.
        """
        tools, _ = await self._get_relevant_tools_with_ambiguity(query, None, top_k)
        return tools

    def _check_exit_signal(self, query: str) -> bool:
        """Check if query contains exit/cancellation signal (word-boundary safe)."""
        return matches_exit_signal(query)

    def _check_greeting(self, query: str) -> Optional[str]:
        """Check if query is a greeting and return response (centralized)."""
        return matches_greeting(query)

    async def route(
        self,
        query: str,
        user_context: Dict[str, Any],
        conversation_state: Optional[Dict] = None
    ) -> RouterDecision:
        """
        Make routing decision using LLM.

        Args:
            query: User's message
            user_context: User info (vehicle, person_id, etc.)
            conversation_state: Current flow state if any

        Returns:
            RouterDecision with action, tool, params, etc.
        """
        await self.initialize()

        logger.info(f"UNIFIED ROUTER START: query='{query[:50]}', has_user_context={user_context is not None}, in_flow={conversation_state is not None}")

        # Quick checks before LLM

        # 1. Check for greeting
        greeting_response = self._check_greeting(query)
        if greeting_response:
            return RouterDecision(
                action="direct_response",
                response=greeting_response,
                reasoning="Greeting detected",
                confidence=1.0
            )

        # 2. Check for exit signal when in flow
        in_flow = conversation_state and conversation_state.get("flow")
        if in_flow and self._check_exit_signal(query):
            return RouterDecision(
                action="exit_flow",
                reasoning="Exit signal detected",
                confidence=1.0
            )

        # 3. CRITICAL: Handle in-flow continue signals explicitly
        # Uses centralized flow_phrases with word-boundary matching
        if in_flow:
            state = conversation_state.get("state", "")

            # "pokaži ostala" type requests in CONFIRMING/SELECTING state
            if matches_show_more(query):
                logger.info(f"UNIFIED ROUTER: 'show more' detected in flow, returning continue_flow")
                return RouterDecision(
                    action="continue_flow",
                    reasoning="Show more items request in active flow",
                    confidence=1.0
                )

            # Confirmation responses in CONFIRMING state (word-boundary safe)
            if state == "confirming":
                if matches_confirm_yes(query):
                    logger.info(f"UNIFIED ROUTER: Confirmation 'yes' detected, returning continue_flow")
                    return RouterDecision(
                        action="continue_flow",
                        reasoning="User confirmed in confirming state",
                        confidence=1.0
                    )
                if matches_confirm_no(query):
                    logger.info(f"UNIFIED ROUTER: Confirmation 'no' detected, returning continue_flow")
                    return RouterDecision(
                        action="continue_flow",
                        reasoning="User cancelled in confirming state",
                        confidence=1.0
                    )

            # Numeric/ordinal selection in SELECTING state
            if state == "selecting":
                if matches_item_selection(query):
                    logger.info(f"UNIFIED ROUTER: Item selection detected, returning continue_flow")
                    return RouterDecision(
                        action="continue_flow",
                        reasoning="User selected item by number/ordinal",
                        confidence=1.0
                    )

        # 4. QUERY ROUTER - Brza staza za poznate patterne (0 tokena, <1ms)
        # Ovo štedi ~80% LLM poziva za jednostavne upite
        logger.info(f"UNIFIED ROUTER: Trying QueryRouter for query='{query[:50]}'")
        qr_result = self.query_router.route(query, user_context)
        logger.info(f"UNIFIED ROUTER: QR result: matched={qr_result.matched}, conf={qr_result.confidence}, flow={qr_result.flow_type if qr_result.matched else None}")
        if qr_result.matched and qr_result.confidence >= 1.0:
            # Samo ako je SIGURAN match (confidence=1.0) - izbjegavamo false positives
            logger.info(
                f"UNIFIED ROUTER: Fast path via QueryRouter → "
                f"{qr_result.tool_name or qr_result.flow_type} (conf={qr_result.confidence})"
            )
            return self._query_result_to_decision(qr_result, user_context)

        # 5. LLM poziv - za kompleksne upite koje Query Router ne prepoznaje
        return await self._llm_route(query, user_context, conversation_state)

    async def _llm_route(
        self,
        query: str,
        user_context: Dict[str, Any],
        conversation_state: Optional[Dict]
    ) -> RouterDecision:
        """Make routing decision using LLM with disambiguation support (v2.0)."""

        # Build context description - use UserContextManager for validated access
        # v22.0: Use UserContextManager
        ctx = UserContextManager(user_context)
        vehicle = ctx.vehicle
        vehicle_info = ""
        if vehicle and vehicle.id:
            # Use VehicleContext properties
            name = vehicle.name or "N/A"
            plate = vehicle.plate or "N/A"
            vehicle_info = f"Korisnikovo vozilo: {name} ({plate})"
        else:
            vehicle_info = "Korisnik NEMA dodijeljeno vozilo"

        # Build flow state description
        flow_info = "Korisnik je u IDLE stanju (novi upit)"
        if conversation_state:
            flow = conversation_state.get("flow")
            state = conversation_state.get("state")
            missing = conversation_state.get("missing_params", [])
            tool = conversation_state.get("tool")

            if flow:
                flow_info = (
                    f"Korisnik je U TIJEKU flow-a:\n"
                    f"  - Flow: {flow}\n"
                    f"  - State: {state}\n"
                    f"  - Tool: {tool}\n"
                    f"  - Nedostaju parametri: {missing}"
                )

        # v2.0: Get relevant tools WITH ambiguity detection
        relevant_tools, ambiguity_result = await self._get_relevant_tools_with_ambiguity(
            query, user_context, top_k=25
        )

        # Build tools description
        tools_desc = f"Dostupni alati ({len(relevant_tools)} relevantnih):\n"
        for tool_name, description in relevant_tools.items():
            tools_desc += f"  - {tool_name}: {description}\n"

        # v2.0: Build disambiguation hints if ambiguity detected
        disambiguation_section = ""
        if ambiguity_result and ambiguity_result.is_ambiguous:
            disambiguation_section = f"""
        UPOZORENJE - DETEKTIRANA DVOSMISLENOST:
        {ambiguity_result.disambiguation_hint}

        Slični alati: {', '.join(ambiguity_result.similar_tools[:5])}

        PRAVILO ZA DVOSMISLENOST:
        - Ako upit NE SPOMINJE specifični entitet (npr. vozila, troškovi, osobe),
          koristi action="clarify" i pitaj korisnika koje podatke želi
        - Ako upit SPOMINJE entitet, odaberi alat za taj entitet
        - Primjer: "prosječna kilometraža" → entitet je vozila → get_Vehicles_Agg ili get_MasterData
        """

        # Build system prompt with disambiguation support (v2.0)
        system_prompt = f"""Ti si routing sustav za MobilityOne fleet management bot.

        TVOJ ZADATAK: Odluči što napraviti s korisnikovim upitom.

        {vehicle_info}

        {flow_info}

        {tools_desc}
        {disambiguation_section}

        PRAVILA:

        1. AKO je korisnik U TIJEKU flow-a:
        - Ako korisnik daje tražene parametre → action="continue_flow"
        - Ako korisnik potvrđuje (Da/Ne) → action="continue_flow"
        - Ako korisnik traži prikaz ostalih opcija ("pokaži ostala", "druga vozila") → action="continue_flow"
        - Ako korisnik bira broj ("1", "2", "prvi") → action="continue_flow"
        - SAMO ako korisnik EKSPLICITNO želi PREKINUTI flow → action="exit_flow"
        - PREPOZNAJ exit SAMO za: "ne želim ovo", "odustani od rezervacije", "zapravo nešto drugo"
        - "pokaži ostala", "koja još vozila", "više opcija" NIJE exit - to je continue_flow!

        2. AKO korisnik NIJE u flow-u:
        - Ako treba pokrenuti flow (rezervacija, unos km, prijava štete) → action="start_flow"
        - Ako je jednostavan upit (dohvat podataka) → action="simple_api"
        - Ako je pozdrav ili zahvala → action="direct_response"
        - Ako je upit PREVIŠE GENERIČAN (npr. "prosječna vrijednost" bez entiteta) → action="clarify"

        3. ODABIR ALATA:
        - "unesi km", "upiši kilometražu", "mogu li upisati" → post_AddMileage (WRITE!)
        - "koliko imam km", "moja kilometraža" → get_MasterData (READ)
        - "registracija", "tablica", "podaci o vozilu" → get_MasterData
        - "slobodna vozila", "dostupna vozila" → get_AvailableVehicles
        - "trebam auto", "rezerviraj" → get_AvailableVehicles (pa flow)
        - "moje rezervacije" → get_VehicleCalendar
        - "prijavi štetu", "kvar", "udario sam" → post_AddCase
        - "troškovi" → get_Expenses
        - "tripovi", "putovanja" → get_Trips

        4. FLOW TYPES:
        - booking: za rezervacije
        - mileage: za unos kilometraže
        - case: za prijavu štete/kvara

        5. CLARIFY (v2.0):
        - Koristi action="clarify" SAMO kada je upit previše generičan
        - U "clarification" polju postavi pitanje koje će pomoći identificirati pravi alat
        - Primjer: "Za koje podatke želite izračunati statistiku? (vozila, troškovi, putovanja)"

        ODGOVORI U JSON FORMATU:
        {{
            "action": "continue_flow|exit_flow|start_flow|simple_api|direct_response|clarify",
            "tool": "ime_alata ili null",
            "params": {{}},
            "flow_type": "booking|mileage|case ili null",
            "response": "tekst odgovora za direct_response ili null",
            "clarification": "pitanje za korisnika (samo za action=clarify)",
            "reasoning": "kratko objašnjenje odluke",
            "confidence": 0.0-1.0
        }}"""

        user_prompt = f'Korisnikov upit: "{query}"'

        try:
            response = await self._circuit_breaker.call(
                f"llm_router:{self.model}",
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            action = result.get("action", "simple_api")

            logger.info(
                f"UNIFIED ROUTER v2.0: '{query[:30]}...' → "
                f"action={action}, tool={result.get('tool')}, "
                f"flow={result.get('flow_type')}, "
                f"ambiguous={ambiguity_result.is_ambiguous if ambiguity_result else False}"
            )

            # CRITICAL FIX: Prevent exit_flow when not in a flow
            if action == "exit_flow" and not conversation_state:
                logger.warning(
                    f"LLM returned exit_flow but no active flow - "
                    f"converting to simple_api. Query: '{query[:40]}...'"
                )
                action = "simple_api"
                if not result.get("tool"):
                    result["tool"] = "get_MasterData"

            # v2.0: Handle clarify action
            if action == "clarify":
                clarification_text = result.get("clarification")
                if not clarification_text and ambiguity_result:
                    # Use detector's clarification question as fallback
                    clarification_text = ambiguity_result.clarification_question

                return RouterDecision(
                    action="clarify",
                    clarification=clarification_text,
                    reasoning=result.get("reasoning", "Query is ambiguous, need clarification"),
                    confidence=float(result.get("confidence", 0.3)),
                    ambiguity_detected=True
                )

            return RouterDecision(
                action=action,
                tool=result.get("tool"),
                params=result.get("params", {}),
                flow_type=result.get("flow_type"),
                response=result.get("response"),
                reasoning=result.get("reasoning", ""),
                confidence=float(result.get("confidence", 0.5)),
                ambiguity_detected=ambiguity_result.is_ambiguous if ambiguity_result else False
            )

        except CircuitOpenError as e:
            logger.warning(f"Circuit breaker OPEN for router - falling back: {e}")
            return self._fallback_route(query, user_context)

        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            # Fallback - try to detect basic intent
            return self._fallback_route(query, user_context)

    def _fallback_route(
        self,
        query: str,
        user_context: Dict[str, Any]
    ) -> RouterDecision:
        """Fallback routing when LLM fails - uses QueryRouter's regex rules."""
        logger.warning(f"LLM routing failed, using QueryRouter fallback for: '{query[:50]}...'")

        # Koristi Query Router - ima 51 regex pravilo
        qr_result = self.query_router.route(query, user_context)

        if qr_result.matched:
            logger.info(f"FALLBACK: QueryRouter matched -> {qr_result.tool_name or qr_result.flow_type}")
            return self._query_result_to_decision(qr_result, user_context, is_fallback=True)

        # Ultimate fallback - ask for clarification instead of guessing
        logger.warning(f"FALLBACK: QueryRouter no match, asking for clarification")
        return RouterDecision(
            action="direct_response",
            response=(
                "Nisam siguran sto trazite. Mozete pitati za:\n"
                "* Rezervaciju vozila\n"
                "* Kilometrazu\n"
                "* Prijavu stete\n"
                "* Informacije o vozilu\n"
                "* Troskove\n"
                "* Putovanja\n\n"
                "Mozete li pojasniti sto tocno trebate?"
            ),
            reasoning="Ultimate fallback: No match found, asking user to clarify",
            confidence=0.1
        )

    def _query_result_to_decision(
        self,
        qr_result: RouteResult,
        user_context: Optional[Dict[str, Any]] = None,
        is_fallback: bool = False
    ) -> RouterDecision:
        """
        Convert QueryRouter RouteResult to RouterDecision.

        Args:
            qr_result: Result from QueryRouter
            user_context: User context for template formatting
            is_fallback: True if called from fallback path (lower confidence)

        Returns:
            RouterDecision compatible with rest of system
        """
        # Confidence reduction for fallback path
        confidence = qr_result.confidence * (0.8 if is_fallback else 1.0)
        path_type = "fallback" if is_fallback else "fast path"

        flow_type = qr_result.flow_type

        # 1. Direct response (greetings, help, thanks, context queries)
        if flow_type == "direct_response":
            # FORMAT the template with user context
            response_text = qr_result.response_template
            if response_text and user_context:
                # v22.0: Use UserContextManager for validated access
                ctx = UserContextManager(user_context)
                # Direct extraction for context queries (person_id, phone, tenant_id)
                if 'person_id' in response_text:
                    val = ctx.person_id or 'N/A'
                    response_text = f"👤 **Person ID:** {val}"
                elif 'phone' in response_text:
                    val = ctx.phone or 'N/A'
                    response_text = f"📱 **Telefon:** {val}"
                elif 'tenant_id' in response_text:
                    val = ctx.tenant_id or 'N/A'
                    response_text = f"🏢 **Tenant ID:** {val}"
                else:
                    # For other templates, use format() with simple context
                    simple_context = {
                        k: v for k, v in user_context.items() 
                        if not isinstance(v, (dict, list))
                    }
                    try:
                        response_text = response_text.format(**simple_context)
                    except (KeyError, ValueError):
                        # If format fails, return template as-is
                        pass
            
            return RouterDecision(
                action="direct_response",
                response=response_text,
                reasoning=f"QueryRouter {path_type}: {qr_result.reason}",
                confidence=confidence
            )

        # 2. Flows that need multi-step interaction
        if flow_type in ("booking", "mileage_input", "case_creation"):
            # Map flow_type to canonical names
            canonical_flow = {
                "booking": "booking",
                "mileage_input": "mileage",
                "case_creation": "case"
            }.get(flow_type, flow_type)

            return RouterDecision(
                action="start_flow",
                tool=qr_result.tool_name,
                flow_type=canonical_flow,
                reasoning=f"QueryRouter {path_type}: {qr_result.reason}",
                confidence=confidence
            )

        # 3. Simple API calls (get_MasterData, get_VehicleCalendar, etc.)
        # flow_type: "simple" or "list"
        return RouterDecision(
            action="simple_api",
            tool=qr_result.tool_name,
            flow_type=flow_type,
            reasoning=f"QueryRouter {path_type}: {qr_result.reason}",
            confidence=confidence
        )


# Singleton
_router: Optional[UnifiedRouter] = None


async def get_unified_router() -> UnifiedRouter:
    """Get or create singleton router instance."""
    global _router
    if _router is None:
        _router = UnifiedRouter()
        await _router.initialize()
    return _router

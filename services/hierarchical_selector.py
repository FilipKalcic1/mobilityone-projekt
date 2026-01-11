"""
Hierarchical Tool Selector - Accurate tool selection through staged filtering.
Version: 1.0

PRINCIP: FAISS je RANKER, ne GATEKEEPER.
         Daj mu ≤10 već ispravnih kandidata, i on će ih rangirati točno.

FLOW:
    1. Domain Detection (deterministički) → 950 → ~80 alata
    2. Action + Sub-domain Filter → ~80 → ~8 alata
    3. FAISS Ranking (semantic) → ~8 → Top 3
    4. LLM Final Decision → Top 3 → 1 alat

NIKAD ne šaljemo 50+ alata FAISS-u ili LLM-u.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from services.domain_detector import (
    DomainDetector, get_domain_detector,
    DomainResult, ActionResult,
    SuperDomain, ActionIntent
)

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Rezultat odabira alata."""
    tool: Optional[str]
    confidence: float
    domain: SuperDomain
    action: ActionIntent
    candidates_count: int  # Koliko kandidata je FAISS vidio
    reasoning: str

    # Za debugging
    stage1_candidates: int  # Nakon domain filtera
    stage2_candidates: int  # Nakon action filtera
    stage3_candidates: int  # Nakon FAISS rankinga


# =============================================================================
# DOMAIN → TOOL PREFIX MAPPING
# =============================================================================

DOMAIN_TOOL_PREFIXES = {
    SuperDomain.VEHICLES: [
        "Vehicles", "VehicleTypes", "VehicleCalendar", "VehicleContracts",
        "VehiclesHistoricalEntries", "VehiclesMonthlyExpenses",
        "VehicleBoard", "VehicleInputHelper",
        "AvailableVehicles", "LatestVehicleCalendar", "LatestVehicleContracts",
    ],
    SuperDomain.PERSONS: [
        "Persons", "PersonTypes", "PersonOrgUnits",
        "PersonPeriodicActivities", "PersonActivityTypes",
        "LatestPersonPeriodicActivities",
    ],
    SuperDomain.FINANCIALS: [
        "Expenses", "ExpenseTypes", "ExpenseGroups",
        "CostCenters",
        "MonthlyFuelExpensesAndMileage", "AverageFuelExpensesAndMileage",
    ],
    SuperDomain.TRIPS: [
        "Trips", "TripTypes",
        "MileageReports", "LatestMileageReports",
        "MonthlyMileages", "MonthlyMileagesAssigned",
    ],
    SuperDomain.EQUIPMENT: [
        "Equipment", "EquipmentTypes", "EquipmentCalendar",
        "LatestEquipmentCalendar", "EquipmentCalendarOn",
    ],
    SuperDomain.CASES: [
        "Cases", "CaseTypes",
    ],
    SuperDomain.TEAMS: [
        "Teams", "TeamMembers",
    ],
    SuperDomain.ORGANIZATION: [
        "Companies", "Partners", "OrgUnits", "Tenants", "TenantPermissions",
    ],
    SuperDomain.SCHEDULING: [
        "PeriodicActivities", "PeriodicActivitiesSchedules", "PeriodicActivityTypes",
        "LatestPeriodicActivities",
        "SchedulingModels", "Pools",
    ],
    SuperDomain.SYSTEM: [
        "Lookup", "Settings", "DocumentTypes", "Documents",
        "Tags", "Metadata", "Dashboard", "DashboardItems",
        "Roles", "Master", "Demo", "SendEmail",
        "Stats",
    ],
}

# HTTP Method mapping
ACTION_TO_METHOD = {
    ActionIntent.READ: ["GET"],
    ActionIntent.CREATE: ["POST"],
    ActionIntent.UPDATE: ["PUT"],
    ActionIntent.DELETE: ["DELETE"],
    ActionIntent.PATCH: ["PATCH"],
    ActionIntent.UNKNOWN: ["GET", "POST", "PUT", "DELETE", "PATCH"],  # All
}


class HierarchicalToolSelector:
    """
    Hijerarhijski selektor alata.

    Koristi 4-stage filtriranje za točan odabir:
    1. Domain Detection → smanjuje prostor
    2. Action + Sub-domain Filter → dodatno sužava
    3. FAISS Ranking → rangira preostale (≤10)
    4. LLM Decision → bira iz top 3-5
    """

    def __init__(self, registry, faiss_store=None):
        """
        Args:
            registry: ToolRegistry s učitanim alatima
            faiss_store: FAISS vector store (optional, za rangiranje)
        """
        self.registry = registry
        self.faiss_store = faiss_store
        self.domain_detector = get_domain_detector()

        # Index tools by domain for fast filtering
        self._tools_by_domain: Dict[SuperDomain, List[str]] = {}
        self._build_domain_index()

    def _build_domain_index(self):
        """Pre-index tools by domain for O(1) lookup."""
        for domain, prefixes in DOMAIN_TOOL_PREFIXES.items():
            self._tools_by_domain[domain] = []

            for tool_id, tool in self.registry.tools.items():
                # Check if tool matches any prefix for this domain
                for prefix in prefixes:
                    # Tool format: get_Vehicles_id, post_Expenses, etc.
                    parts = tool_id.split("_")
                    if len(parts) >= 2 and parts[1].startswith(prefix):
                        self._tools_by_domain[domain].append(tool_id)
                        break

        # Log domain sizes
        for domain, tools in self._tools_by_domain.items():
            logger.info(f"Domain {domain.value}: {len(tools)} tools")

    def _filter_by_domain(self, domain: SuperDomain) -> List[str]:
        """Stage 1: Filter tools by super-domain."""
        if domain == SuperDomain.UNKNOWN:
            # Return all tools if domain unknown (bad case)
            return list(self.registry.tools.keys())

        return self._tools_by_domain.get(domain, [])

    def _filter_by_action(self, tool_ids: List[str], action: ActionIntent) -> List[str]:
        """Stage 2: Filter tools by HTTP method (action)."""
        allowed_methods = ACTION_TO_METHOD.get(action, ["GET"])

        filtered = []
        for tool_id in tool_ids:
            tool = self.registry.tools.get(tool_id)
            if tool and tool.method.upper() in allowed_methods:
                filtered.append(tool_id)

        return filtered

    def _filter_by_subdomain(self, tool_ids: List[str], sub_domains: List[str]) -> List[str]:
        """Stage 2b: Further filter by sub-domain if detected."""
        if not sub_domains:
            return tool_ids

        filtered = []
        for tool_id in tool_ids:
            parts = tool_id.split("_")
            if len(parts) >= 2:
                tool_entity = parts[1]  # e.g., "Expenses" from "delete_Expenses_id"
                for sub in sub_domains:
                    if tool_entity.startswith(sub) or sub.startswith(tool_entity):
                        filtered.append(tool_id)
                        break

        # If filtering was too aggressive, return original
        if len(filtered) < 2:
            return tool_ids

        return filtered

    async def _rank_with_faiss(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Stage 3: Use FAISS to rank candidates semantically.

        KRITIČNO: candidates MORA biti ≤10-15 za točan rad FAISS-a.
        """
        if not self.faiss_store or not self.faiss_store.is_initialized():
            # No FAISS - return candidates as-is with equal scores
            logger.warning("FAISS not available, returning unranked candidates")
            return [(c, 0.5) for c in candidates[:top_k]]

        if len(candidates) > 20:
            logger.warning(
                f"FAISS received {len(candidates)} candidates - TOO MANY! "
                f"Should be ≤10 for accurate ranking."
            )

        # Search only within candidates
        results = await self.faiss_store.search_within_candidates(
            query=query,
            candidates=candidates,
            top_k=min(top_k, len(candidates))
        )

        return [(r.tool_id, r.score) for r in results]

    async def select(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> SelectionResult:
        """
        Odaberi alat kroz hijerarhijsko filtriranje.

        Args:
            query: Korisnički upit na hrvatskom
            user_context: Kontekst (vehicle, person, etc.)
            top_k: Koliko kandidata vratiti za LLM

        Returns:
            SelectionResult s odabranim alatom i metapodacima
        """
        # Stage 1: Domain Detection
        domain_result, action_result = self.domain_detector.detect(query)

        stage1_tools = self._filter_by_domain(domain_result.domain)
        logger.info(
            f"Stage 1 (Domain={domain_result.domain.value}): "
            f"950 → {len(stage1_tools)} tools"
        )

        # Stage 2: Action + Sub-domain Filter
        stage2_tools = self._filter_by_action(stage1_tools, action_result.action)

        if domain_result.sub_domains:
            stage2_tools = self._filter_by_subdomain(
                stage2_tools, domain_result.sub_domains
            )

        logger.info(
            f"Stage 2 (Action={action_result.action.value}, "
            f"SubDomains={domain_result.sub_domains}): "
            f"{len(stage1_tools)} → {len(stage2_tools)} tools"
        )

        # Check if we have reasonable number of candidates
        if len(stage2_tools) == 0:
            logger.warning("No tools found after filtering!")
            return SelectionResult(
                tool=None,
                confidence=0.0,
                domain=domain_result.domain,
                action=action_result.action,
                candidates_count=0,
                reasoning="No tools found for this domain/action combination",
                stage1_candidates=len(stage1_tools),
                stage2_candidates=0,
                stage3_candidates=0
            )

        if len(stage2_tools) > 15:
            logger.warning(
                f"Still have {len(stage2_tools)} candidates after Stage 2 - "
                f"FAISS accuracy may be reduced"
            )

        # Stage 3: FAISS Ranking
        ranked = await self._rank_with_faiss(query, stage2_tools, top_k=top_k)

        if not ranked:
            # Fallback to first candidate if FAISS fails
            return SelectionResult(
                tool=stage2_tools[0] if stage2_tools else None,
                confidence=0.5,
                domain=domain_result.domain,
                action=action_result.action,
                candidates_count=len(stage2_tools),
                reasoning="FAISS ranking failed, using first filtered candidate",
                stage1_candidates=len(stage1_tools),
                stage2_candidates=len(stage2_tools),
                stage3_candidates=0
            )

        logger.info(
            f"Stage 3 (FAISS Ranking): {len(stage2_tools)} → {len(ranked)} tools"
        )

        # Top result
        best_tool, best_score = ranked[0]

        # Calculate overall confidence
        # Combine domain confidence, action confidence, and FAISS score
        overall_confidence = (
            domain_result.confidence * 0.3 +
            action_result.confidence * 0.3 +
            best_score * 0.4
        )

        return SelectionResult(
            tool=best_tool,
            confidence=overall_confidence,
            domain=domain_result.domain,
            action=action_result.action,
            candidates_count=len(stage2_tools),
            reasoning=f"Domain={domain_result.domain.value}, Action={action_result.action.value}, Score={best_score:.2f}",
            stage1_candidates=len(stage1_tools),
            stage2_candidates=len(stage2_tools),
            stage3_candidates=len(ranked)
        )

    async def select_with_candidates(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> Tuple[SelectionResult, List[Tuple[str, float]]]:
        """
        Odaberi alat i vrati top-k kandidata za LLM.

        Returns:
            (SelectionResult, List of (tool_id, score) candidates for LLM)
        """
        result = await self.select(query, user_context, top_k)

        if result.tool is None:
            return result, []

        # Get Stage 2 filtered tools again for candidates
        domain_result, action_result = self.domain_detector.detect(query)
        stage1 = self._filter_by_domain(domain_result.domain)
        stage2 = self._filter_by_action(stage1, action_result.action)

        if domain_result.sub_domains:
            stage2 = self._filter_by_subdomain(stage2, domain_result.sub_domains)

        # Re-rank for candidates
        ranked = await self._rank_with_faiss(query, stage2, top_k=top_k)

        return result, ranked


# =============================================================================
# FACTORY
# =============================================================================

_selector: Optional[HierarchicalToolSelector] = None


async def get_hierarchical_selector(registry=None, faiss_store=None) -> HierarchicalToolSelector:
    """Get or create singleton selector."""
    global _selector

    if _selector is None:
        if registry is None:
            from services.tool_registry import ToolRegistry
            from config import get_settings
            import redis.asyncio as aioredis

            settings = get_settings()
            redis = aioredis.from_url(settings.REDIS_URL)
            registry = ToolRegistry(redis_client=redis)
            await registry.initialize(settings.swagger_sources)

        if faiss_store is None:
            from services.faiss_vector_store import get_faiss_store
            faiss_store = get_faiss_store()

        _selector = HierarchicalToolSelector(registry, faiss_store)

    return _selector

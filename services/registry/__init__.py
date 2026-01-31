"""
Tool Registry - Public API facade.
Version: 2.1 (With Documentation Merge - Pillar 5)

This module provides backward-compatible interface to the refactored registry.
All existing imports from services.tool_registry should work unchanged.

NEW in 2.1: Merges tool_documentation.json into tool objects at runtime
so that origin_guide is available throughout the system.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional, Set

from services.tool_contracts import UnifiedToolDefinition, DependencyGraph

from .tool_store import ToolStore
from .cache_manager import CacheManager
from .swagger_parser import SwaggerParser
from .embedding_engine import EmbeddingEngine
from .search_engine import SearchEngine

logger = logging.getLogger(__name__)

# Documentation cache - loaded once at startup
_documentation_cache: Optional[Dict[str, Any]] = None

# Re-export for backward compatibility
__all__ = ['ToolRegistry', 'ToolStore', 'CacheManager', 'SwaggerParser', 'EmbeddingEngine', 'SearchEngine']


class ToolRegistry:
    """
    Dynamic tool registry with 2-step discovery.

    Architecture:
    1. Load Swagger specs -> Parse -> Build UnifiedToolDefinition
    2. Generate embeddings for semantic search
    3. Build dependency graph for auto-chaining
    4. Persist to .cache/ for fast startup

    This is a facade that coordinates the refactored components.
    """

    MAX_TOOLS_PER_RESPONSE = 12

    def __init__(self, redis_client=None):
        """Initialize tool registry with all components."""
        self.redis = redis_client

        # Initialize components
        self._store = ToolStore()
        self._cache = CacheManager()
        self._parser = SwaggerParser()
        self._embedding = EmbeddingEngine()
        self._search = SearchEngine()

        # PILLAR 5: Documentation merge storage
        self._tool_documentation: Dict[str, Any] = {}

        # State
        self.is_ready = False
        self._load_lock = asyncio.Lock()

        # Load documentation at startup
        self._load_documentation()

        logger.info("ToolRegistry initialized (v3.1 - with documentation merge)")

    def _load_documentation(self) -> None:
        """
        PILLAR 5: Load tool_documentation.json for runtime access.

        This enables origin_guide and other documentation to be available
        throughout the system without repeated file reads.
        """
        global _documentation_cache

        # Use cached version if available
        if _documentation_cache is not None:
            self._tool_documentation = _documentation_cache
            return

        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        doc_path = os.path.join(base_path, "config", "tool_documentation.json")

        if os.path.exists(doc_path):
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    self._tool_documentation = json.load(f)
                    _documentation_cache = self._tool_documentation
                    logger.info(f"PILLAR 5: Loaded documentation for {len(self._tool_documentation)} tools")
            except Exception as e:
                logger.warning(f"Could not load tool documentation: {e}")
                self._tool_documentation = {}
        else:
            logger.info("No tool_documentation.json found - run generate_documentation.py first")

    # ═══════════════════════════════════════════════
    # BACKWARD COMPATIBILITY PROPERTIES
    # ═══════════════════════════════════════════════

    @property
    def tools(self) -> Dict[str, UnifiedToolDefinition]:
        """Access tools dict for backward compatibility."""
        return self._store.tools

    @property
    def embeddings(self) -> Dict[str, List[float]]:
        """Access embeddings dict for backward compatibility."""
        return self._store.embeddings

    @property
    def dependency_graph(self) -> Dict[str, DependencyGraph]:
        """Access dependency graph for backward compatibility."""
        return self._store.dependency_graph

    @property
    def retrieval_tools(self) -> Set[str]:
        """Access retrieval tools set for backward compatibility."""
        return self._store.retrieval_tools

    @property
    def mutation_tools(self) -> Set[str]:
        """Access mutation tools set for backward compatibility."""
        return self._store.mutation_tools

    # Context param patterns (exposed for message_engine)
    @property
    def CONTEXT_PARAM_FALLBACK(self) -> Dict[str, str]:
        """Access context param fallback for backward compatibility."""
        return self._parser.context_param_fallback

    # ═══════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════

    async def load_swagger(self, source: str) -> bool:
        """Backward compatible method for single source."""
        logger.warning("⚠️ load_swagger() is DEPRECATED. Use initialize([sources]).")
        return await self.initialize([source])

    async def initialize(self, swagger_sources: List[str]) -> bool:
        """
        Initialize registry from pre-processed file or Swagger sources.
        """
        async with self._load_lock:
            try:
                # Preferred method: Load from pre-processed file
                base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                registry_path = os.path.join(base_path, "config", "processed_tool_registry.json")

                if os.path.exists(registry_path):
                    logger.info(f"✅ Pre-processed registry found - loading from {registry_path}")
                    with open(registry_path, 'r', encoding='utf-8') as f:
                        registry_data = json.load(f)
                    
                    for tool_data in registry_data.get("tools", []):
                        tool = UnifiedToolDefinition(**tool_data)
                        self._store.add_tool(tool)
                    
                    for dep_data in registry_data.get("dependency_graph", []):
                        dep = DependencyGraph(**dep_data)
                        self._store.add_dependency(dep)
                    
                    logger.info(f"Loaded {self._store.count()} tools and {len(self._store.dependency_graph)} dependencies from file.")
                    
                    # v20.1: Try to load embeddings from cache to avoid re-generating
                    embeddings_cache_path = os.path.join(base_path, ".cache", "tool_embeddings.json")
                    if os.path.exists(embeddings_cache_path):
                        try:
                            with open(embeddings_cache_path, 'r', encoding='utf-8') as ef:
                                cache_data = json.load(ef)
                            # Cache has wrapper structure: {version, timestamp, embeddings: {op_id: [...]}}
                            cached_embeddings = cache_data.get("embeddings", cache_data)
                            if isinstance(cached_embeddings, dict):
                                for op_id, embedding in cached_embeddings.items():
                                    if op_id in self._store.tools and isinstance(embedding, list):
                                        self._store.add_embedding(op_id, embedding)
                            logger.info(f"📦 Loaded {len(self._store.embeddings)} embeddings from cache")
                        except Exception as e:
                            logger.warning(f"Failed to load embeddings cache: {e}")

                else:
                    logger.warning(f"⚠️ Pre-processed registry not found at {registry_path}. Falling back to dynamic loading from Swagger URLs.")
                    
                    # Fallback to old logic
                    if await self._cache.is_cache_valid(swagger_sources):
                        logger.info("✅ Cache valid - loading from disk")
                        cached_data = await self._cache.load_cache()

                        for tool in cached_data["tools"]:
                            self._store.add_tool(tool)
                        for dep in cached_data["dependency_graph"]:
                            self._store.add_dependency(dep)
                        for op_id, embedding in cached_data["embeddings"].items():
                            self._store.add_embedding(op_id, embedding)
                        
                        logger.info(f"✅ Loaded {self._store.count()} tools from cache.")

                    else:
                        logger.info("🔄 Cache invalid - fetching Swagger specs...")
                        for source in swagger_sources:
                            tools = await self._parser.parse_spec(source, self._embedding.build_embedding_text)
                            for tool in tools:
                                self._store.add_tool(tool)

                        if self._store.count() == 0:
                            logger.error("❌ No tools loaded from Swagger sources")
                            return False
                        
                        logger.info(f"📦 Loaded {self._store.count()} tools from Swagger")

                        # Build dependency graph
                        dep_graph = self._embedding.build_dependency_graph(self._store.tools)
                        for dep in dep_graph.values():
                            self._store.add_dependency(dep)
                
                # v20.1 FIX: Only generate embeddings if not all cached
                cached_count = len(self._store.embeddings)
                total_tools = self._store.count()
                
                if cached_count >= total_tools:
                    logger.info(f"✅ All embeddings cached ({cached_count}/{total_tools}) - skipping generation")
                else:
                    logger.info(f"🔄 Generating missing embeddings ({cached_count}/{total_tools} cached)...")
                    new_embeddings = await self._embedding.generate_embeddings(
                        self._store.tools,
                        self._store.embeddings
                    )
                    for op_id, embedding in new_embeddings.items():
                        self._store.add_embedding(op_id, embedding)
                    logger.info(f"✅ Generated {len(new_embeddings)} new embeddings")

                # Save cache if we didn't load from pre-processed file
                if not os.path.exists(registry_path):
                    await self._cache.save_cache(
                        swagger_sources,
                        list(self._store.tools.values()),
                        self._store.embeddings,
                        list(self._store.dependency_graph.values())
                    )

                self.is_ready = True
                logger.info(
                    f"✅ Initialized {self._store.count()} tools "
                    f"({len(self._store.retrieval_tools)} retrieval, "
                    f"{len(self._store.mutation_tools)} mutation)"
                )

                # v3.0: Initialize FAISS vector store for fast semantic search
                await self._initialize_faiss()

                return True

            except Exception as e:
                logger.error(f"❌ Initialization failed: {e}", exc_info=True)
                return False

    async def _initialize_faiss(self) -> None:
        """
        Initialize FAISS vector store for fast semantic search.

        v3.0: Uses tool_documentation.json for embeddings.
        Does NOT use training_queries.json (unreliable).
        """
        try:
            from services.faiss_vector_store import get_faiss_store, initialize_faiss_store
            import json
            import os

            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            tool_doc_path = os.path.join(base_path, "config", "tool_documentation.json")

            if not os.path.exists(tool_doc_path):
                logger.warning(f"FAISS: tool_documentation.json not found at {tool_doc_path}")
                return

            with open(tool_doc_path, 'r', encoding='utf-8') as f:
                tool_documentation = json.load(f)

            # Initialize FAISS with documentation and registry tools
            faiss_store = await initialize_faiss_store(
                tool_documentation=tool_documentation,
                tool_registry_tools=self._store.tools
            )

            logger.info(f"✅ FAISS initialized: {faiss_store.get_stats()['total_tools']} tools indexed")

        except ImportError as e:
            logger.warning(f"FAISS not available: {e}. Using legacy search.")
        except Exception as e:
            logger.warning(f"FAISS initialization failed: {e}. Using legacy search.")

    # ═══════════════════════════════════════════════
    # SEARCH & DISCOVERY
    # ═══════════════════════════════════════════════

    async def find_relevant_tools(
        self,
        query: str,
        top_k: int = 5,
        prefer_retrieval: bool = False,
        prefer_mutation: bool = False,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Find relevant tools (backward compatible).

        Returns list of OpenAI function schemas.
        """
        if not self.is_ready:
            logger.warning("Registry not ready")
            return []

        results = await self.find_relevant_tools_with_scores(
            query=query,
            top_k=top_k,
            threshold=threshold or 0.55,
            prefer_retrieval=prefer_retrieval,
            prefer_mutation=prefer_mutation
        )

        return [r["schema"] for r in results]

    async def find_relevant_tools_with_scores(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.55,
        prefer_retrieval: bool = False,
        prefer_mutation: bool = False,
        use_filtered_search: bool = True,
        use_faiss: bool = True,
        use_llm_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find relevant tools WITH SIMILARITY SCORES.

        v4.0: FAISS + ACTION INTENT + LLM RERANKING

        Pipeline:
        1. FAISS semantic search (fast, ~64% top-1)
        2. ACTION INTENT filter (GET/POST/PUT/DELETE)
        3. LLM RERANKING (picks best from top-5, ~90% top-1)

        Args:
            query: User query
            top_k: Number of tools to return
            threshold: Minimum similarity threshold
            prefer_retrieval: Only search GET methods
            prefer_mutation: Only search POST/PUT/DELETE methods
            use_filtered_search: Use FILTER-THEN-SEARCH approach (default True)
            use_faiss: Use FAISS vector search (default True)
            use_llm_rerank: Use LLM to rerank top candidates (default True, v4.0)

        Returns list of dicts with name, score, and schema.
        """
        if not self.is_ready:
            logger.warning("Registry not ready")
            return []

        # v4.0: FAISS + ACTION INTENT + LLM RERANKING
        if use_faiss:
            try:
                from services.faiss_vector_store import get_faiss_store
                from services.intent_classifier import detect_action_intent

                faiss_store = get_faiss_store()

                if faiss_store.is_initialized():
                    # Detect ACTION INTENT
                    intent_result = detect_action_intent(query)
                    action_filter = intent_result.intent.value if intent_result.intent.value != "UNKNOWN" else None

                    # FAISS search - get MORE candidates for reranking
                    faiss_top_k = top_k * 2 if use_llm_rerank else top_k
                    faiss_results = await faiss_store.search(
                        query=query,
                        top_k=faiss_top_k,
                        action_filter=action_filter
                    )

                    if faiss_results:
                        # v4.0: LLM RERANKING - pick best from FAISS candidates
                        if use_llm_rerank and len(faiss_results) > 1:
                            faiss_results = await self._apply_llm_reranking(
                                query, faiss_results, top_k
                            )

                        results = []
                        for r in faiss_results[:top_k]:
                            tool = self._store.get_tool(r.tool_id)
                            if tool and r.score >= threshold:
                                results.append({
                                    "name": r.tool_id,
                                    "score": r.score,
                                    "schema": tool.to_openai_function()
                                })

                        if results:
                            logger.info(
                                f"FAISS search: {len(results)} tools "
                                f"(intent={intent_result.intent.value}, llm_rerank={use_llm_rerank})"
                            )
                            return results

                        logger.debug("FAISS: no results above threshold")

            except ImportError:
                pass  # FAISS not available, use legacy
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}, using legacy")

        # Fallback: FILTER-THEN-SEARCH
        if use_filtered_search:
            return await self._search.find_relevant_tools_filtered(
                query=query,
                tools=self._store.tools,
                embeddings=self._store.embeddings,
                dependency_graph=self._store.dependency_graph,
                retrieval_tools=self._store.retrieval_tools,
                mutation_tools=self._store.mutation_tools,
                top_k=top_k,
                threshold=threshold
            )

        # Ultimate fallback: Original search method
        return await self._search.find_relevant_tools_with_scores(
            query=query,
            tools=self._store.tools,
            embeddings=self._store.embeddings,
            dependency_graph=self._store.dependency_graph,
            retrieval_tools=self._store.retrieval_tools,
            mutation_tools=self._store.mutation_tools,
            top_k=top_k,
            threshold=threshold,
            prefer_retrieval=prefer_retrieval,
            prefer_mutation=prefer_mutation
        )

    async def _apply_llm_reranking(
        self,
        query: str,
        faiss_results: List,
        top_k: int
    ) -> List:
        """
        v4.0: Use LLM to rerank FAISS candidates.

        This dramatically improves Top-1 accuracy:
        - FAISS alone: ~64% Top-1
        - FAISS + LLM rerank: ~90% Top-1

        The LLM understands context that embeddings cannot:
        - "dohvati kompaniju" → LIST (get_Companies)
        - "dohvati kompaniju 123" → SINGLE (get_Companies_id)
        - "potpuno ažuriraj" → PUT (full replacement)
        - "djelomično ažuriraj" → PATCH (partial update)

        Args:
            query: User query
            faiss_results: List of SearchResult from FAISS
            top_k: Number of results to return

        Returns:
            Reordered list of SearchResult with best match first
        """
        try:
            from services.llm_reranker import rerank_with_llm

            # Build candidate list for LLM
            candidates = []
            for r in faiss_results[:10]:  # Max 10 candidates
                doc = self._tool_documentation.get(r.tool_id, {})
                candidates.append({
                    "tool_id": r.tool_id,
                    "score": r.score,
                    "description": doc.get("purpose", "")[:200]
                })

            # Call LLM reranker
            reranked = await rerank_with_llm(
                query=query,
                candidates=candidates,
                top_k=top_k,
                tool_documentation=self._tool_documentation
            )

            if not reranked:
                logger.debug("LLM rerank returned empty, using original order")
                return faiss_results

            # Rebuild results in LLM-recommended order
            tool_id_to_result = {r.tool_id: r for r in faiss_results}
            reordered = []

            for rr in reranked:
                original = tool_id_to_result.get(rr.tool_id)
                if original:
                    # Update score with LLM confidence
                    from dataclasses import replace
                    updated = replace(original, score=max(original.score, rr.confidence))
                    reordered.append(updated)

            # Add any remaining results not in reranked list
            reranked_ids = {rr.tool_id for rr in reranked}
            for r in faiss_results:
                if r.tool_id not in reranked_ids:
                    reordered.append(r)

            logger.info(
                f"LLM rerank: {faiss_results[0].tool_id} → {reordered[0].tool_id} "
                f"(confidence={reranked[0].confidence:.2f})"
            )

            # FIX v11.1: Was top_k * 2 which returned double the expected results
            return reordered[:top_k]

        except ImportError as e:
            logger.debug(f"LLM reranker not available: {e}")
            return faiss_results
        except Exception as e:
            logger.warning(f"LLM rerank failed: {e}, using FAISS order")
            return faiss_results

    # ═══════════════════════════════════════════════
    # TOOL ACCESS
    # ═══════════════════════════════════════════════

    def get_tool(self, operation_id: str) -> Optional[UnifiedToolDefinition]:
        """Get tool by operation ID."""
        return self._store.get_tool(operation_id)

    def list_tools(self) -> List[str]:
        """List all tool operation IDs."""
        return self._store.list_tools()

    # ═══════════════════════════════════════════════
    # PILLAR 5: DOCUMENTATION ACCESS
    # ═══════════════════════════════════════════════

    def get_tool_documentation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get documentation for a specific tool.

        Returns documentation dict including:
        - purpose
        - when_to_use
        - parameter_origin_guide
        - example_queries_hr
        - etc.
        """
        return self._tool_documentation.get(operation_id)

    def get_parameter_origin_guide(self, operation_id: str) -> Dict[str, str]:
        """
        Get parameter origin guide for a tool.

        Returns dict like:
        {
            "personId": "CONTEXT: Sustav ubacuje iz sesije",
            "vehicleId": "USER: Pitaj korisnika za ovo",
            ...
        }
        """
        doc = self._tool_documentation.get(operation_id, {})
        return doc.get("parameter_origin_guide", {})

    def get_tool_with_documentation(
        self,
        operation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get tool definition merged with its documentation.

        Returns combined dict with tool schema AND documentation.
        Useful for LLM prompts that need full context.
        """
        tool = self._store.get_tool(operation_id)
        if not tool:
            return None

        result = {
            "tool": tool.to_openai_function(),
            "documentation": self._tool_documentation.get(operation_id, {})
        }

        return result

    def is_context_param(self, operation_id: str, param_name: str) -> bool:
        """
        Check if a parameter should be auto-injected from context.

        Returns True if the parameter origin is CONTEXT.
        """
        origin_guide = self.get_parameter_origin_guide(operation_id)
        origin = origin_guide.get(param_name, "")
        return "CONTEXT" in origin.upper()

    def is_user_param(self, operation_id: str, param_name: str) -> bool:
        """
        Check if a parameter should be provided by the user.

        Returns True if the parameter origin is USER.
        """
        origin_guide = self.get_parameter_origin_guide(operation_id)
        origin = origin_guide.get(param_name, "USER")  # Default to USER if not specified
        return "USER" in origin.upper()

    # ═══════════════════════════════════════════════
    # CJELINA 2: HIDDEN DEFAULTS INJECTION
    # ═══════════════════════════════════════════════

    # Hidden defaults that should be auto-injected (user doesn't see these)
    # Format: { operation_id: { param_name: default_value } }
    _HIDDEN_DEFAULTS: Dict[str, Dict[str, Any]] = {
        # VehicleCalendar booking defaults
        "post_VehicleCalendar": {
            "EntryType": 0,       # 0 = BOOKING (not absence, holiday, etc.)
            "AssigneeType": 1,    # 1 = PERSON (not department, vehicle pool, etc.)
        },
        # Case/Ticket creation defaults (WhatsApp source)
        "post_AddCase": {
            "EntryType": "WhatsApp",  # Source channel
        },
        # Add more tool defaults here as needed
    }

    def get_hidden_defaults(self, operation_id: str) -> Dict[str, Any]:
        """
        Get hidden default values for a tool.

        These are values that should be auto-injected without asking the user.
        Examples: EntryType for bookings, source channel for tickets.

        Args:
            operation_id: Tool operation ID

        Returns:
            Dict of param_name -> default_value
        """
        return self._HIDDEN_DEFAULTS.get(operation_id, {})

    def inject_defaults(
        self,
        operation_id: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Inject hidden defaults into parameters if not already present.

        CJELINA 2 FIX: Moves business logic from ToolExecutor to Registry.
        The Executor should be "dumb" - just make HTTP calls.
        All smart logic (defaults, validation) belongs here.

        Args:
            operation_id: Tool operation ID
            params: Current parameters dict (will be modified in place)

        Returns:
            Modified params dict with defaults injected
        """
        defaults = self.get_hidden_defaults(operation_id)

        for param_name, default_value in defaults.items():
            if param_name not in params or params[param_name] is None:
                params[param_name] = default_value
                logger.debug(
                    f"REGISTRY: Injected default {param_name}={default_value} "
                    f"for {operation_id}"
                )

        return params

    def get_merged_params(
        self,
        operation_id: str,
        user_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get parameters with hidden defaults merged in.

        PHASE 3: Copy semantics - does NOT modify original user_params.

        Logic:
        1. Start with copy of hidden defaults
        2. Update with user params (user overrides defaults)
        3. Return merged dict

        Args:
            operation_id: Tool operation ID
            user_params: Parameters from user/LLM

        Returns:
            New dict with defaults + user params merged
        """
        # Start with hidden defaults
        merged = self.get_hidden_defaults(operation_id).copy()

        # User params override defaults
        if user_params:
            # Filter out None values from user params
            clean_user_params = {
                k: v for k, v in user_params.items()
                if v is not None
            }
            merged.update(clean_user_params)

        if merged:
            logger.debug(
                f"REGISTRY: Merged params for {operation_id}: "
                f"defaults={list(self.get_hidden_defaults(operation_id).keys())}, "
                f"user={list((user_params or {}).keys())}"
            )

        return merged

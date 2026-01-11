#!/usr/bin/env python3
"""
GLOBAL 950 TOOLS TEST - True Semantic Search Accuracy

This test evaluates the REAL accuracy of the system by:
1. Loading ALL 950 tools from the registry
2. For each tool, generating a natural language query based on tool metadata
3. Testing if semantic search correctly identifies the tool
4. Bypassing QueryRouter to test pure semantic understanding

This exposes the TRUE quality of FAISS + embeddings, not pattern matching shortcuts.

Usage (inside Docker):
    python scripts/test_global_950.py
"""

import asyncio
import sys
import os
import random
import time
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Fix encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis.asyncio as aioredis


# Croatian query templates based on HTTP method
QUERY_TEMPLATES = {
    "GET": [
        "Pokaži mi {entity}",
        "Prikaži {entity}",
        "Dohvati {entity}",
        "Lista {entity}",
        "Koje su {entity}",
        "Daj mi {entity}",
        "Trebam {entity}",
        "{entity}",  # Just the entity name
    ],
    "POST": [
        "Dodaj {entity}",
        "Kreiraj {entity}",
        "Unesi {entity}",
        "Nova {entity}",
        "Napravi {entity}",
        "Stvori {entity}",
    ],
    "PUT": [
        "Ažuriraj {entity}",
        "Izmijeni {entity}",
        "Promijeni {entity}",
        "Update {entity}",
    ],
    "DELETE": [
        "Obriši {entity}",
        "Ukloni {entity}",
        "Izbriši {entity}",
        "Makni {entity}",
    ],
    "PATCH": [
        "Djelomično ažuriraj {entity}",
        "Patch {entity}",
        "Izmijeni dio {entity}",
    ],
}

# Entity name translations (common ones)
ENTITY_TRANSLATIONS = {
    "vehicles": "vozila",
    "vehicle": "vozilo",
    "persons": "osobe",
    "person": "osobu",
    "trips": "putovanja",
    "trip": "putovanje",
    "expenses": "troškove",
    "expense": "trošak",
    "cases": "slučajeve",
    "case": "slučaj",
    "partners": "partnere",
    "partner": "partnera",
    "teams": "timove",
    "team": "tim",
    "equipment": "opremu",
    "documents": "dokumente",
    "document": "dokument",
    "calendar": "kalendar",
    "booking": "rezervaciju",
    "bookings": "rezervacije",
    "mileage": "kilometražu",
    "reports": "izvještaje",
    "report": "izvještaj",
    "types": "tipove",
    "type": "tip",
    "stats": "statistiku",
    "dashboard": "dashboard",
    "companies": "tvrtke",
    "company": "tvrtku",
    "pools": "poolove",
    "pool": "pool",
    "tags": "oznake",
    "tag": "oznaku",
    "roles": "uloge",
    "role": "ulogu",
    "members": "članove",
    "member": "člana",
    "contracts": "ugovore",
    "contract": "ugovor",
    "activities": "aktivnosti",
    "activity": "aktivnost",
    "notifications": "obavijesti",
    "notification": "obavijest",
    "settings": "postavke",
    "imports": "importi",
    "exports": "eksporti",
    "master": "master podatke",
    "available": "dostupna",
    "latest": "najnovije",
    "monthly": "mjesečne",
    "periodic": "periodične",
    "scheduling": "raspoređivanje",
    "org": "organizacijske",
    "units": "jedinice",
    "cost": "troškovna",
    "centers": "mjesta",
}


def extract_entity_from_tool(tool_name: str, tool_summary: str = None) -> str:
    """Extract entity name from tool name and translate to Croatian."""
    # Remove method prefix (get_, post_, etc.)
    entity = re.sub(r'^(get|post|put|delete|patch)_', '', tool_name, flags=re.IGNORECASE)

    # Remove _id suffix
    entity = re.sub(r'_id$', '', entity)
    entity = re.sub(r'_\{[^}]+\}$', '', entity)  # Remove {param} suffix

    # Split by underscore and translate each part
    parts = entity.split('_')
    translated_parts = []

    for part in parts:
        part_lower = part.lower()
        if part_lower in ENTITY_TRANSLATIONS:
            translated_parts.append(ENTITY_TRANSLATIONS[part_lower])
        else:
            # Keep original but make it more readable
            translated_parts.append(part.lower())

    return ' '.join(translated_parts)


def generate_query_for_tool(tool_name: str, method: str, summary: str = None) -> str:
    """Generate a natural Croatian query for a tool."""
    entity = extract_entity_from_tool(tool_name, summary)

    # Get templates for this method
    templates = QUERY_TEMPLATES.get(method.upper(), QUERY_TEMPLATES["GET"])

    # Pick a random template
    template = random.choice(templates)

    return template.format(entity=entity)


async def initialize_registry():
    """Initialize tool registry with all 950 tools."""
    from config import get_settings
    settings = get_settings()

    print("Initializing registry...")

    redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    await redis_client.ping()
    print("  Redis: connected")

    from services.tool_registry import ToolRegistry
    registry = ToolRegistry(redis_client=redis_client)

    success = await registry.initialize(settings.swagger_sources)
    if not success:
        raise RuntimeError("Failed to initialize registry")

    print(f"  Tools loaded: {len(registry.tools)}")
    print(f"  Embeddings: {'ready' if registry.embeddings else 'not loaded'}")

    return registry, redis_client


async def test_semantic_search_direct(registry, query: str, expected_tool: str, top_k: int = 10) -> Tuple[bool, str, float, List[str]]:
    """
    Test semantic search DIRECTLY, bypassing QueryRouter.

    Returns: (is_match, actual_tool, score, top_k_tools)
    """
    try:
        # Use FAISS if available
        from services.faiss_vector_store import get_faiss_store

        faiss_store = get_faiss_store()

        if faiss_store.is_initialized():
            # Direct FAISS search
            results = await faiss_store.search(
                query=query,
                top_k=top_k,
                action_filter=None  # No filtering to test pure semantic
            )

            if results:
                top_tools = [r.tool_id for r in results]
                top_tool = results[0].tool_id
                top_score = results[0].score

                # Check if expected tool is in top-k
                is_match = expected_tool in top_tools
                is_top1 = top_tool == expected_tool

                return is_top1, top_tool, top_score, top_tools

        # Fallback to legacy search
        if registry._search and registry.embeddings:
            results = await registry._search.find_relevant_tools_with_scores(
                query=query,
                tools=registry.tools,
                embeddings=registry.embeddings,
                dependency_graph=registry.dependency_graph,
                retrieval_tools=registry.retrieval_tools,
                mutation_tools=registry.mutation_tools,
                top_k=top_k,
                threshold=0.0  # No threshold to see all results
            )

            if results:
                top_tools = [r["name"] for r in results]
                top_tool = results[0]["name"]
                top_score = results[0].get("score", 0)

                is_match = expected_tool in top_tools
                is_top1 = top_tool == expected_tool

                return is_top1, top_tool, top_score, top_tools

        return False, None, 0.0, []

    except Exception as e:
        print(f"  Search error: {e}")
        return False, None, 0.0, []


async def run_global_test(sample_size: int = 100):
    """
    Run global test on random sample of tools.

    Args:
        sample_size: Number of tools to test (use -1 for all 950)
    """
    print("=" * 70)
    print("  GLOBAL 950 TOOLS TEST - TRUE SEMANTIC ACCURACY")
    print("  (Bypassing QueryRouter - Pure Semantic Search)")
    print("=" * 70)
    print()

    registry, redis = await initialize_registry()

    # Get all tools
    all_tools = list(registry.tools.values())
    print(f"\nTotal tools available: {len(all_tools)}")

    # Sample tools for testing
    if sample_size == -1 or sample_size >= len(all_tools):
        test_tools = all_tools
    else:
        test_tools = random.sample(all_tools, sample_size)

    print(f"Testing {len(test_tools)} tools...\n")
    print("-" * 70)

    # Results tracking
    results = {
        "total": 0,
        "top1_correct": 0,
        "top5_correct": 0,
        "top10_correct": 0,
        "failed": 0,
        "by_method": defaultdict(lambda: {"total": 0, "top1": 0, "top5": 0}),
        "by_domain": defaultdict(lambda: {"total": 0, "top1": 0, "top5": 0}),
        "failures": [],
    }

    start_time = time.time()

    for i, tool in enumerate(test_tools, 1):
        tool_name = tool.operation_id
        method = tool.method.upper()
        summary = tool.summary or tool.description or ""

        # Generate query
        query = generate_query_for_tool(tool_name, method, summary)

        # Test semantic search
        is_top1, actual_tool, score, top_k_tools = await test_semantic_search_direct(
            registry, query, tool_name, top_k=10
        )

        # Track results
        results["total"] += 1
        results["by_method"][method]["total"] += 1

        # Extract domain from tool name
        domain = tool_name.split("_")[1] if "_" in tool_name else "unknown"
        results["by_domain"][domain]["total"] += 1

        # Check accuracy at different levels
        in_top5 = tool_name in top_k_tools[:5]
        in_top10 = tool_name in top_k_tools[:10]

        if is_top1:
            results["top1_correct"] += 1
            results["by_method"][method]["top1"] += 1
            results["by_domain"][domain]["top1"] += 1
            status = "TOP1"
            icon = "1"
        elif in_top5:
            results["top5_correct"] += 1
            results["by_method"][method]["top5"] += 1
            results["by_domain"][domain]["top5"] += 1
            status = "TOP5"
            icon = "5"
        elif in_top10:
            results["top10_correct"] += 1
            status = "TOP10"
            icon = "X"
        else:
            results["failed"] += 1
            status = "FAIL"
            icon = "-"
            results["failures"].append({
                "tool": tool_name,
                "query": query,
                "actual": actual_tool,
                "top5": top_k_tools[:5]
            })

        # Print progress every 10 tools
        if i % 10 == 0 or i == len(test_tools):
            pct = (i / len(test_tools)) * 100
            top1_pct = (results["top1_correct"] / i) * 100
            print(f"[{i:3d}/{len(test_tools)}] {pct:5.1f}% | Top-1: {top1_pct:.1f}% | Last: [{icon}] {tool_name[:30]}")

    elapsed = time.time() - start_time

    # Calculate final stats
    total = results["total"]
    top1 = results["top1_correct"]
    top5 = results["top5_correct"] + top1  # Cumulative
    top10 = results["top10_correct"] + top5  # Cumulative

    top1_pct = (top1 / total) * 100
    top5_pct = (top5 / total) * 100
    top10_pct = (top10 / total) * 100

    # Print summary
    print("\n" + "=" * 70)
    print("  REZULTATI GLOBALNOG TESTA")
    print("=" * 70)

    print(f"""
  Testirano alata:     {total}
  Vrijeme:             {elapsed:.1f}s ({elapsed/total*1000:.0f}ms/tool)

  TOP-1 ACCURACY:      {top1:4d} / {total} ({top1_pct:.1f}%)
  TOP-5 ACCURACY:      {top5:4d} / {total} ({top5_pct:.1f}%)
  TOP-10 ACCURACY:     {top10:4d} / {total} ({top10_pct:.1f}%)
  FAILED:              {results['failed']:4d} / {total} ({results['failed']/total*100:.1f}%)
""")

    # Breakdown by HTTP method
    print("-" * 70)
    print("  BREAKDOWN BY HTTP METHOD:")
    print("-" * 70)

    for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
        stats = results["by_method"][method]
        if stats["total"] > 0:
            t1_pct = (stats["top1"] / stats["total"]) * 100
            t5_pct = ((stats["top1"] + stats["top5"]) / stats["total"]) * 100
            bar = "█" * int(t1_pct / 10) + "░" * (10 - int(t1_pct / 10))
            print(f"  {method:8s} {bar} {stats['top1']:3d}/{stats['total']:3d} ({t1_pct:5.1f}%) Top-1 | ({t5_pct:5.1f}%) Top-5")

    # Top 10 worst domains
    print("\n" + "-" * 70)
    print("  WORST PERFORMING DOMAINS (by Top-1):")
    print("-" * 70)

    domain_stats = []
    for domain, stats in results["by_domain"].items():
        if stats["total"] >= 3:  # Only domains with 3+ tools
            pct = (stats["top1"] / stats["total"]) * 100
            domain_stats.append((domain, pct, stats["total"], stats["top1"]))

    domain_stats.sort(key=lambda x: x[1])  # Sort by percentage

    for domain, pct, total, correct in domain_stats[:10]:
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {domain:25s} {bar} {correct:2d}/{total:2d} ({pct:5.1f}%)")

    # Sample failures
    if results["failures"]:
        print("\n" + "-" * 70)
        print("  SAMPLE FAILURES (first 10):")
        print("-" * 70)

        for fail in results["failures"][:10]:
            print(f"\n  Tool:   {fail['tool']}")
            print(f"  Query:  \"{fail['query']}\"")
            print(f"  Actual: {fail['actual']}")
            print(f"  Top-5:  {', '.join(fail['top5'][:3])}...")

    # Final grade
    print("\n" + "=" * 70)
    print("  KONAČNA OCJENA SEMANTIC SEARCH-a")
    print("=" * 70)

    if top1_pct >= 80:
        grade = "ODLIČAN (A)"
    elif top1_pct >= 60:
        grade = "DOBAR (B)"
    elif top1_pct >= 40:
        grade = "SLAB (C)"
    elif top1_pct >= 20:
        grade = "LOŠ (D)"
    else:
        grade = "KRITIČNO (F)"

    print(f"""
  TOP-1 ACCURACY: {top1_pct:.1f}%

  OCJENA: {grade}

  {'Sustav pouzdano pronalazi ispravne alate.' if top1_pct >= 60 else 'Sustav ima problema s pronalaženjem ispravnih alata!'}
""")

    print("=" * 70)

    await redis.aclose()

    return results


async def main():
    """Main entry point."""
    try:
        # Test with 100 random tools first, use -1 for all 950
        sample_size = 100

        if len(sys.argv) > 1:
            sample_size = int(sys.argv[1])

        results = await run_global_test(sample_size)

        top1_pct = (results["top1_correct"] / results["total"]) * 100
        sys.exit(0 if top1_pct >= 40 else 1)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

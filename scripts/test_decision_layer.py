"""
Test Decision Layer accuracy.

Compares:
1. FAISS-only Top-1
2. Decision Layer Top-1 (with deterministic rules)
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)

from services.faiss_vector_store import initialize_faiss_store
from services.tool_decision_layer import get_decision_layer


def load_tool_documentation() -> Dict:
    """Load tool documentation."""
    doc_path = project_root / "config" / "tool_documentation.json"
    with open(doc_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sanitize_for_print(text: str) -> str:
    """Remove special characters that can't be printed on Windows console."""
    replacements = {
        'č': 'c', 'ć': 'c', 'š': 's', 'ž': 'z', 'đ': 'd',
        'Č': 'C', 'Ć': 'C', 'Š': 'S', 'Ž': 'Z', 'Đ': 'D'
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text


async def run_test():
    print("=" * 70)
    print("DECISION LAYER ACCURACY TEST")
    print("Comparing FAISS-only vs Decision Layer")
    print("=" * 70)
    print()

    # Load tool documentation
    print("[1/3] Loading tool documentation...")
    tool_docs = load_tool_documentation()
    print(f"      Loaded {len(tool_docs)} tools")

    # Initialize FAISS
    print("[2/3] Initializing FAISS...")
    faiss_store = await initialize_faiss_store(tool_docs)
    print(f"      FAISS ready")

    # Get decision layer
    decision_layer = get_decision_layer()

    # Collect test cases
    test_cases = []
    for tool_id, doc in tool_docs.items():
        example_queries = doc.get("example_queries_hr", [])
        for query in example_queries:
            if query and len(query.strip()) > 3:
                test_cases.append((query.strip(), tool_id))

    print(f"[3/3] Running {len(test_cases)} tests...")
    print("-" * 70)

    # Metrics
    faiss_top1 = 0
    faiss_top3 = 0
    decision_top1 = 0
    decision_top3 = 0

    # Decision method stats
    method_stats = {
        "auto_accept": {"total": 0, "correct": 0},
        "tie_breaker": {"total": 0, "correct": 0},
        "llm_needed": {"total": 0, "correct": 0},
        "no_results": {"total": 0, "correct": 0},
    }

    # Track improvements and regressions
    improvements = []
    regressions = []

    for i, (query, expected_tool) in enumerate(test_cases):
        # Get expected method
        expected_method = "GET"
        if expected_tool.lower().startswith("post_"):
            expected_method = "POST"
        elif expected_tool.lower().startswith("put_"):
            expected_method = "PUT"
        elif expected_tool.lower().startswith("patch_"):
            expected_method = "PATCH"
        elif expected_tool.lower().startswith("delete_"):
            expected_method = "DELETE"

        # FAISS search (without decision layer)
        results = await faiss_store.search(
            query=query,
            top_k=10,
            action_filter=None  # No action filter for fair comparison
        )

        faiss_results = [
            {"tool_id": r.tool_id, "score": r.score, "method": r.method}
            for r in results
        ]

        # Check FAISS accuracy
        faiss_tools = [r['tool_id'] for r in faiss_results]
        faiss_got_top1 = faiss_tools[0] == expected_tool if faiss_tools else False
        faiss_got_top3 = expected_tool in faiss_tools[:3]

        if faiss_got_top1:
            faiss_top1 += 1
        if faiss_got_top3:
            faiss_top3 += 1

        # Decision layer
        decision = decision_layer.decide(query, faiss_results)
        decision_got_top1 = decision.tool_id == expected_tool

        if decision_got_top1:
            decision_top1 += 1
            # Check if in top 3 (for comparison)
            decision_top3 += 1
        elif expected_tool in faiss_tools[:3]:
            decision_top3 += 1

        # Track method stats
        method_stats[decision.decision_method]["total"] += 1
        if decision_got_top1:
            method_stats[decision.decision_method]["correct"] += 1

        # Track improvements and regressions
        if decision_got_top1 and not faiss_got_top1:
            improvements.append({
                "query": query[:50],
                "expected": expected_tool,
                "faiss_got": faiss_tools[0] if faiss_tools else "N/A",
                "decision_got": decision.tool_id,
                "method": decision.decision_method
            })
        elif faiss_got_top1 and not decision_got_top1:
            regressions.append({
                "query": query[:50],
                "expected": expected_tool,
                "faiss_got": faiss_tools[0] if faiss_tools else "N/A",
                "decision_got": decision.tool_id,
                "method": decision.decision_method
            })

        # Progress
        if (i + 1) % 500 == 0:
            print(f"      Processed {i + 1}/{len(test_cases)}...")

    total = len(test_cases)
    print(f"      Completed {total} tests")
    print()

    # Print results
    print("=" * 70)
    print("REZULTATI")
    print("=" * 70)
    print()

    print("OVERALL ACCURACY:")
    print(f"   FAISS-only Top-1:     {faiss_top1/total*100:6.2f}%  ({faiss_top1}/{total})")
    print(f"   Decision Layer Top-1: {decision_top1/total*100:6.2f}%  ({decision_top1}/{total})")
    print()
    improvement = (decision_top1 - faiss_top1) / total * 100
    print(f"   IMPROVEMENT: {improvement:+.2f}%")
    print()

    print("DECISION METHOD BREAKDOWN:")
    for method, stats in method_stats.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"   {method:15s}: {stats['correct']:4d}/{stats['total']:4d} ({acc:5.1f}%)")
    print()

    print(f"IMPROVEMENTS (Decision Layer fixed FAISS error): {len(improvements)}")
    for imp in improvements[:10]:
        q = sanitize_for_print(imp['query'])
        print(f"   Query: \"{q}...\"")
        print(f"   FAISS: {imp['faiss_got']} -> Decision: {imp['decision_got']} (via {imp['method']})")
        print()

    print(f"REGRESSIONS (Decision Layer broke FAISS correct): {len(regressions)}")
    for reg in regressions[:10]:
        q = sanitize_for_print(reg['query'])
        print(f"   Query: \"{q}...\"")
        print(f"   FAISS: {reg['faiss_got']} -> Decision: {reg['decision_got']} (via {reg['method']})")
        print()

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_test())

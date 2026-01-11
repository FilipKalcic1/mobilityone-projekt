"""
Test Search Accuracy V2 - Koristimo example_queries_hr iz tool_documentation.json

Logika:
1. Za SVAKI tool u tool_documentation.json
2. Uzmi njegove example_queries_hr (upiti koje smo generirali za taj tool)
3. Testiraj svaki upit - pronalazi li FAISS tocno taj tool?

Ovo je "ground truth" test jer koristimo upite koji su NAMIJENJENI za taj tool.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# Suppress logs
import logging
logging.basicConfig(level=logging.WARNING)

from services.faiss_vector_store import get_faiss_store, initialize_faiss_store
from services.action_intent_detector import detect_action_intent


def load_tool_documentation() -> Dict:
    """Load tool documentation."""
    doc_path = project_root / "config" / "tool_documentation.json"
    with open(doc_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sanitize_for_print(text: str) -> str:
    """Remove special characters that can't be printed on Windows console."""
    # Replace Croatian special chars with ASCII equivalents
    replacements = {
        'č': 'c', 'ć': 'c', 'š': 's', 'ž': 'z', 'đ': 'd',
        'Č': 'C', 'Ć': 'C', 'Š': 'S', 'Ž': 'Z', 'Đ': 'D'
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text


async def run_test():
    print("=" * 70)
    print("FAISS ACCURACY TEST V2")
    print("Koristimo example_queries_hr iz tool_documentation.json")
    print("=" * 70)
    print()

    # Load tool documentation
    print("[1/3] Loading tool documentation...")
    tool_docs = load_tool_documentation()
    print(f"      Loaded {len(tool_docs)} tools")

    # Initialize FAISS
    print("[2/3] Initializing FAISS...")
    faiss_store = await initialize_faiss_store(tool_docs)
    print(f"      FAISS ready: {faiss_store.get_stats()['total_tools']} tools indexed")
    print()

    # Collect all test cases: (query, expected_tool)
    test_cases = []
    tools_with_examples = 0
    tools_without_examples = 0

    for tool_id, doc in tool_docs.items():
        example_queries = doc.get("example_queries_hr", [])
        if example_queries:
            tools_with_examples += 1
            for query in example_queries:
                if query and len(query.strip()) > 3:  # Skip empty/short
                    test_cases.append((query.strip(), tool_id))
        else:
            tools_without_examples += 1

    print(f"[3/3] Test cases collected:")
    print(f"      Tools WITH example_queries_hr: {tools_with_examples}")
    print(f"      Tools WITHOUT examples: {tools_without_examples}")
    print(f"      Total test queries: {len(test_cases)}")
    print()

    # Run tests
    print("Running tests...")
    print("-" * 70)

    # Metrics
    total = 0
    top_1 = 0
    top_3 = 0
    top_5 = 0
    top_10 = 0
    not_found = 0

    # Track per-method accuracy
    method_stats = {
        "GET": {"total": 0, "top1": 0, "top3": 0},
        "POST": {"total": 0, "top1": 0, "top3": 0},
        "PUT": {"total": 0, "top1": 0, "top3": 0},
        "DELETE": {"total": 0, "top1": 0, "top3": 0},
    }

    # Sample of failures for analysis
    failures = []

    for i, (query, expected_tool) in enumerate(test_cases):
        # Detect intent for filtering
        intent_result = detect_action_intent(query)
        action_filter = intent_result.intent.value if intent_result.intent.value != "UNKNOWN" else None

        # Search
        results = await faiss_store.search(
            query=query,
            top_k=10,
            action_filter=action_filter
        )

        # Get result tool IDs
        result_tools = [r.tool_id for r in results]

        # Find rank
        rank = 0
        if expected_tool in result_tools:
            rank = result_tools.index(expected_tool) + 1

        total += 1

        # Update metrics
        if rank == 1:
            top_1 += 1
            top_3 += 1
            top_5 += 1
            top_10 += 1
        elif rank <= 3:
            top_3 += 1
            top_5 += 1
            top_10 += 1
        elif rank <= 5:
            top_5 += 1
            top_10 += 1
        elif rank <= 10:
            top_10 += 1
        else:
            not_found += 1
            if len(failures) < 50:  # Collect up to 50 failures
                failures.append({
                    "query": query,
                    "expected": expected_tool,
                    "got": result_tools[0] if result_tools else "N/A",
                    "rank": rank,
                    "intent": action_filter
                })

        # Update method stats
        method = "GET"
        if expected_tool.lower().startswith("post_"):
            method = "POST"
        elif expected_tool.lower().startswith("put_") or expected_tool.lower().startswith("patch_"):
            method = "PUT"
        elif expected_tool.lower().startswith("delete_"):
            method = "DELETE"

        method_stats[method]["total"] += 1
        if rank == 1:
            method_stats[method]["top1"] += 1
        if rank <= 3:
            method_stats[method]["top3"] += 1

        # Progress
        if (i + 1) % 500 == 0:
            print(f"      Processed {i + 1}/{len(test_cases)}...")

    print(f"      Completed {total} tests")
    print()

    # Print results
    print("=" * 70)
    print("REZULTATI")
    print("=" * 70)
    print()

    print("UKUPNA TOCNOST:")
    print(f"   Total test queries: {total}")
    print()
    print(f"   Top-1 Accuracy:  {top_1/total*100:6.2f}%  ({top_1}/{total})")
    print(f"   Top-3 Accuracy:  {top_3/total*100:6.2f}%  ({top_3}/{total})")
    print(f"   Top-5 Accuracy:  {top_5/total*100:6.2f}%  ({top_5}/{total})")
    print(f"   Top-10 Accuracy: {top_10/total*100:6.2f}%  ({top_10}/{total})")
    print(f"   NOT FOUND:       {not_found/total*100:6.2f}%  ({not_found}/{total})")
    print()

    print("PO HTTP METODI:")
    for method, stats in method_stats.items():
        if stats["total"] > 0:
            t1_acc = stats["top1"] / stats["total"] * 100
            t3_acc = stats["top3"] / stats["total"] * 100
            print(f"   {method:8s}: Top-1={t1_acc:5.1f}%  Top-3={t3_acc:5.1f}%  (n={stats['total']})")
    print()

    # Print sample failures
    if failures:
        print("PRIMJERI NEUSPJELIH UPITA (prvih 20):")
        print("-" * 70)
        for f in failures[:20]:
            query_safe = sanitize_for_print(f["query"][:60])
            print(f"   Query: \"{query_safe}\"")
            print(f"   Expected: {f['expected']}")
            print(f"   Got: {f['got']} (intent={f['intent']})")
            print()

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_test())

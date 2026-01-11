"""
Confusion Analysis - Koji alati se najcesce mijesaju?

Za svaki alat gdje Top-1 nije tocan, pokazuje:
- Koji alat je vratio umjesto pravog
- Koliko puta se to dogodilo
- Koje upite je pomijesao

Ovo pomaze identificirati alate koji trebaju bolju dokumentaciju.
"""

import asyncio
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

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


def sanitize(text: str) -> str:
    """Remove special characters for Windows console."""
    replacements = {
        'č': 'c', 'ć': 'c', 'š': 's', 'ž': 'z', 'đ': 'd',
        'Č': 'C', 'Ć': 'C', 'Š': 'S', 'Ž': 'Z', 'Đ': 'D'
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text


async def run_confusion_analysis():
    print("=" * 70)
    print("CONFUSION ANALYSIS")
    print("Koji alati se najcesce mijesaju?")
    print("=" * 70)
    print()

    # Load and initialize
    print("[1/2] Loading...")
    tool_docs = load_tool_documentation()
    faiss_store = await initialize_faiss_store(tool_docs)
    print(f"      {len(tool_docs)} tools loaded")
    print()

    # Collect test cases from example_queries_hr
    test_cases = []
    for tool_id, doc in tool_docs.items():
        for query in doc.get("example_queries_hr", []):
            if query and len(query.strip()) > 3:
                test_cases.append((query.strip(), tool_id))

    print(f"[2/2] Running {len(test_cases)} tests...")

    # Track confusions: {expected_tool: {got_tool: [(query, score), ...]}}
    confusions = defaultdict(lambda: defaultdict(list))

    # Track per-tool stats
    tool_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0})

    for i, (query, expected_tool) in enumerate(test_cases):
        # Search
        intent_result = detect_action_intent(query)
        action_filter = intent_result.intent.value if intent_result.intent.value != "UNKNOWN" else None

        results = await faiss_store.search(query=query, top_k=10, action_filter=action_filter)

        if not results:
            continue

        got_tool = results[0].tool_id
        got_score = results[0].score

        # Update stats
        tool_stats[expected_tool]["total"] += 1

        # Check ranks
        result_tools = [r.tool_id for r in results]
        if expected_tool in result_tools:
            rank = result_tools.index(expected_tool) + 1
            if rank == 1:
                tool_stats[expected_tool]["top1"] += 1
            if rank <= 3:
                tool_stats[expected_tool]["top3"] += 1

        # Track confusion if Top-1 is wrong
        if got_tool != expected_tool:
            confusions[expected_tool][got_tool].append((query, got_score))

        if (i + 1) % 500 == 0:
            print(f"      Processed {i + 1}/{len(test_cases)}...")

    print(f"      Done!")
    print()

    # Analyze confusions
    print("=" * 70)
    print("NAJCESCE KONFUZIJE (expected -> got)")
    print("=" * 70)
    print()

    # Flatten and sort confusions by frequency
    confusion_pairs = []
    for expected, got_dict in confusions.items():
        for got, queries in got_dict.items():
            confusion_pairs.append({
                "expected": expected,
                "got": got,
                "count": len(queries),
                "queries": queries
            })

    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Print top 30 confusions
    print("TOP 30 KONFUZIJA:")
    print("-" * 70)
    for i, cp in enumerate(confusion_pairs[:30], 1):
        print(f"{i:2}. {cp['expected']}")
        print(f"    POMIJESAN SA: {cp['got']} ({cp['count']}x)")
        # Show 2 example queries
        for q, s in cp["queries"][:2]:
            print(f"    - \"{sanitize(q[:50])}\" (score={s:.3f})")
        print()

    # Analyze which tools have the WORST Top-1
    print("=" * 70)
    print("ALATI S NAJGOROM TOP-1 TOCNOSCU")
    print("(ovi trebaju bolju dokumentaciju)")
    print("=" * 70)
    print()

    # Calculate per-tool accuracy
    tool_accuracy = []
    for tool_id, stats in tool_stats.items():
        if stats["total"] >= 1:  # At least 1 test query
            top1_acc = stats["top1"] / stats["total"] * 100
            top3_acc = stats["top3"] / stats["total"] * 100
            tool_accuracy.append({
                "tool": tool_id,
                "total": stats["total"],
                "top1": stats["top1"],
                "top1_acc": top1_acc,
                "top3_acc": top3_acc
            })

    # Sort by Top-1 accuracy (worst first)
    tool_accuracy.sort(key=lambda x: x["top1_acc"])

    print("ALATI S 0% TOP-1 TOCNOSCU:")
    print("-" * 70)
    zero_acc = [t for t in tool_accuracy if t["top1_acc"] == 0]
    for t in zero_acc[:30]:
        # Get what it was confused with
        confused_with = list(confusions[t["tool"]].keys())[:3]
        confused_str = ", ".join(confused_with) if confused_with else "N/A"
        print(f"   {t['tool']}")
        print(f"      Queries: {t['total']}, Top-3: {t['top3_acc']:.0f}%")
        print(f"      Pomijesan sa: {confused_str}")
        print()

    if len(zero_acc) > 30:
        print(f"   ... i jos {len(zero_acc) - 30} alata s 0% Top-1")
        print()

    # Show tools that are "stealing" the most
    print("=" * 70)
    print("ALATI KOJI 'KRADU' REZULTATE")
    print("(cesto se pojavljuju umjesto pravog alata)")
    print("=" * 70)
    print()

    # Count how many times each tool appears as "got" when it shouldn't
    stealer_count = defaultdict(int)
    for expected, got_dict in confusions.items():
        for got, queries in got_dict.items():
            stealer_count[got] += len(queries)

    stealers = sorted(stealer_count.items(), key=lambda x: x[1], reverse=True)

    print("TOP 20 'KRADLJIVACA':")
    print("-" * 70)
    for tool, count in stealers[:20]:
        # Find which tools it steals from
        stolen_from = []
        for expected, got_dict in confusions.items():
            if tool in got_dict:
                stolen_from.append((expected, len(got_dict[tool])))
        stolen_from.sort(key=lambda x: x[1], reverse=True)

        victims = ", ".join([f"{v[0]}({v[1]}x)" for v in stolen_from[:3]])
        print(f"   {tool}: {count}x")
        print(f"      Krade od: {victims}")
        print()

    # Summary stats
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_tools = len(tool_stats)
    zero_top1 = len([t for t in tool_accuracy if t["top1_acc"] == 0])
    low_top1 = len([t for t in tool_accuracy if 0 < t["top1_acc"] < 50])
    good_top1 = len([t for t in tool_accuracy if t["top1_acc"] >= 50])
    perfect_top1 = len([t for t in tool_accuracy if t["top1_acc"] == 100])

    print(f"   Total alata testiranih: {total_tools}")
    print(f"   Alati s 0% Top-1:       {zero_top1} ({zero_top1/total_tools*100:.1f}%)")
    print(f"   Alati s 1-49% Top-1:    {low_top1} ({low_top1/total_tools*100:.1f}%)")
    print(f"   Alati s 50%+ Top-1:     {good_top1} ({good_top1/total_tools*100:.1f}%)")
    print(f"   Alati s 100% Top-1:     {perfect_top1} ({perfect_top1/total_tools*100:.1f}%)")
    print()

    # Save detailed results to file
    output_file = project_root / "confusion_report.json"
    report = {
        "summary": {
            "total_tools": total_tools,
            "zero_top1": zero_top1,
            "low_top1": low_top1,
            "good_top1": good_top1,
            "perfect_top1": perfect_top1
        },
        "top_confusions": [
            {
                "expected": cp["expected"],
                "got": cp["got"],
                "count": cp["count"],
                "example_queries": [q for q, s in cp["queries"][:3]]
            }
            for cp in confusion_pairs[:50]
        ],
        "tools_needing_work": [
            {
                "tool": t["tool"],
                "queries": t["total"],
                "top1_acc": t["top1_acc"],
                "top3_acc": t["top3_acc"],
                "confused_with": list(confusions[t["tool"]].keys())[:5]
            }
            for t in tool_accuracy if t["top1_acc"] < 50
        ]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Detaljan report spremnjen u: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_confusion_analysis())

"""
Test LLM Re-ranking accuracy.

Tests how much LLM re-ranking improves over FAISS-only selection.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)

from services.faiss_vector_store import initialize_faiss_store
from services.llm_reranker import rerank_with_llm
from services.action_intent_detector import detect_action_intent


def load_tool_documentation() -> Dict:
    """Load tool documentation."""
    doc_path = project_root / "config" / "tool_documentation.json"
    with open(doc_path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def run_test(sample_size: int = 100):
    print("=" * 70)
    print("LLM RE-RANKING ACCURACY TEST")
    print(f"Testing {sample_size} random queries")
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

    # Collect test cases
    test_cases = []
    for tool_id, doc in tool_docs.items():
        example_queries = doc.get("example_queries_hr", [])
        for query in example_queries:
            if query and len(query.strip()) > 3:
                test_cases.append((query.strip(), tool_id))

    # Random sample
    random.seed(42)
    test_cases = random.sample(test_cases, min(sample_size, len(test_cases)))

    print(f"[3/3] Running {len(test_cases)} tests...")
    print()

    # Metrics
    faiss_top1 = 0
    llm_top1 = 0
    faiss_top3 = 0

    for i, (query, expected_tool) in enumerate(test_cases):
        # FAISS search
        intent_result = detect_action_intent(query)
        action_filter = intent_result.intent.value if intent_result.intent.value != "UNKNOWN" else None

        results = await faiss_store.search(
            query=query,
            top_k=5,
            action_filter=action_filter
        )

        faiss_results = [r.tool_id for r in results]

        # Check FAISS accuracy
        if faiss_results and faiss_results[0] == expected_tool:
            faiss_top1 += 1
            faiss_top3 += 1
        elif expected_tool in faiss_results[:3]:
            faiss_top3 += 1

        # LLM Re-ranking
        candidates = [
            {"tool_id": r.tool_id, "score": r.score}
            for r in results[:5]
        ]

        reranked = await rerank_with_llm(
            query=query,
            candidates=candidates,
            top_k=3,
            tool_documentation=tool_docs
        )

        if reranked and reranked[0].tool_id == expected_tool:
            llm_top1 += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"      Processed {i + 1}/{len(test_cases)}...")

    # Results
    total = len(test_cases)
    print()
    print("=" * 70)
    print("REZULTATI")
    print("=" * 70)
    print()
    print(f"FAISS-only Top-1:     {faiss_top1/total*100:6.2f}%  ({faiss_top1}/{total})")
    print(f"FAISS-only Top-3:     {faiss_top3/total*100:6.2f}%  ({faiss_top3}/{total})")
    print(f"LLM Re-ranked Top-1:  {llm_top1/total*100:6.2f}%  ({llm_top1}/{total})")
    print()
    print(f"Improvement: +{(llm_top1-faiss_top1)/total*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    asyncio.run(run_test(sample_size))

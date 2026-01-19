"""
Tool Routing Accuracy Test
Version: 2.0

Tests if the routing algorithm correctly identifies tools based on example queries.

For EACH documented tool:
1. Takes its example_queries_hr from tool_documentation.json
2. Runs query through FAISS semantic search
3. Checks if correct tool is in Top-1, Top-3, Top-5
4. Generates accuracy report

Usage:
    python scripts/test_tool_routing_accuracy.py [--limit N] [--verbose]

Output:
    - Per-tool results (PASS/FAIL)
    - Overall accuracy metrics
    - Failed tools list for debugging
"""

import asyncio
import json
import sys
import os
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TestResult:
    """Result for a single tool test."""
    operation_id: str
    query: str
    expected_tool: str
    top_1_match: bool = False
    top_3_match: bool = False
    top_5_match: bool = False
    actual_top_5: List[str] = field(default_factory=list)
    score: float = 0.0
    method_used: str = "unknown"
    error: Optional[str] = None


@dataclass
class AccuracyReport:
    """Overall accuracy report."""
    total_tools: int = 0
    total_queries: int = 0
    top_1_correct: int = 0
    top_3_correct: int = 0
    top_5_correct: int = 0
    failed_tools: List[TestResult] = field(default_factory=list)
    skipped_tools: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    method_stats: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def top_1_accuracy(self) -> float:
        return (self.top_1_correct / self.total_queries * 100) if self.total_queries > 0 else 0

    @property
    def top_3_accuracy(self) -> float:
        return (self.top_3_correct / self.total_queries * 100) if self.total_queries > 0 else 0

    @property
    def top_5_accuracy(self) -> float:
        return (self.top_5_correct / self.total_queries * 100) if self.total_queries > 0 else 0


class ToolRoutingTester:
    """Tests tool routing accuracy using documented example queries."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.tool_docs: Dict[str, Any] = {}
        self.registry = None
        self.faiss_store = None
        self.query_router = None

    async def initialize(self):
        """Initialize routing components."""
        # Load tool documentation
        with open('config/tool_documentation.json', 'r', encoding='utf-8') as f:
            self.tool_docs = json.load(f)

        print(f"Loaded {len(self.tool_docs)} tools from documentation")

        # Initialize registry
        from services.registry import ToolRegistry
        self.registry = ToolRegistry(redis_client=None)

        # Load swagger to populate registry
        from config import get_settings
        settings = get_settings()

        if settings.swagger_sources:
            print(f"Initializing registry with {len(settings.swagger_sources)} swagger sources...")
            await self.registry.initialize(settings.swagger_sources)
            print(f"Registry has {len(self.registry.tools)} tools")

        # Initialize FAISS store
        try:
            from services.faiss_vector_store import get_faiss_store
            self.faiss_store = get_faiss_store()
            if self.faiss_store.is_initialized():
                print("FAISS vector store: initialized")
            else:
                print("FAISS vector store: NOT initialized - building index...")
                # Try to build the index
                if hasattr(self.registry, 'embeddings') and self.registry.embeddings:
                    await self.faiss_store.build_index(
                        self.registry.tools,
                        self.registry.embeddings,
                        self.tool_docs
                    )
                    print("FAISS index built")
        except Exception as e:
            print(f"FAISS initialization error: {e}")
            self.faiss_store = None

        # Initialize QueryRouter
        try:
            from services.query_router import QueryRouter
            self.query_router = QueryRouter()
            print("QueryRouter initialized")
        except Exception as e:
            print(f"QueryRouter error: {e}")
            self.query_router = None

    def get_test_queries(self, operation_id: str) -> List[str]:
        """Get test queries for a tool from documentation."""
        doc = self.tool_docs.get(operation_id, {})
        queries = doc.get('example_queries_hr', [])

        # If no example queries, try to generate from purpose
        if not queries and doc.get('purpose'):
            purpose = doc['purpose']
            queries = [purpose.split('.')[0]]

        return queries[:3]  # Limit to 3 queries per tool

    async def search_faiss(self, query: str, top_k: int = 5) -> List[tuple]:
        """Search using FAISS vector store."""
        if not self.faiss_store or not self.faiss_store.is_initialized():
            return []

        try:
            results = await self.faiss_store.search(
                query=query,
                top_k=top_k,
                action_filter=None
            )
            return [(r.tool_id, r.score) for r in results]
        except Exception as e:
            if self.verbose:
                print(f"    FAISS error: {e}")
            return []

    def search_query_router(self, query: str) -> List[tuple]:
        """Search using QueryRouter patterns."""
        if not self.query_router:
            return []

        try:
            route = self.query_router.route(query)
            if route.matched and route.tools:
                return [(t.operation_id, route.confidence) for t in route.tools[:5]]
        except Exception as e:
            if self.verbose:
                print(f"    QueryRouter error: {e}")
        return []

    async def test_single_tool(self, operation_id: str) -> List[TestResult]:
        """Test routing for a single tool."""
        results = []
        queries = self.get_test_queries(operation_id)

        if not queries:
            return results

        for query in queries:
            result = TestResult(
                operation_id=operation_id,
                query=query,
                expected_tool=operation_id
            )

            try:
                top_tools = []

                # Method 1: Try QueryRouter first (fast pattern matching)
                qr_results = self.search_query_router(query)
                if qr_results:
                    top_tools = [t[0] for t in qr_results]
                    result.score = qr_results[0][1] if qr_results else 0
                    result.method_used = "QueryRouter"

                # Method 2: Try FAISS semantic search
                if not top_tools:
                    faiss_results = await self.search_faiss(query, top_k=5)
                    if faiss_results:
                        top_tools = [t[0] for t in faiss_results]
                        result.score = faiss_results[0][1] if faiss_results else 0
                        result.method_used = "FAISS"

                # Method 3: Fallback to keyword matching
                if not top_tools:
                    top_tools = self._keyword_match(query)
                    result.method_used = "Keyword"

                result.actual_top_5 = top_tools[:5]

                # Check matches
                if top_tools and top_tools[0] == operation_id:
                    result.top_1_match = True
                if operation_id in top_tools[:3]:
                    result.top_3_match = True
                if operation_id in top_tools[:5]:
                    result.top_5_match = True

            except Exception as e:
                result.error = str(e)

            results.append(result)

            if self.verbose:
                status = "PASS" if result.top_1_match else ("TOP3" if result.top_3_match else ("TOP5" if result.top_5_match else "FAIL"))
                print(f"  [{status}] [{result.method_used}] {operation_id}: '{query[:40]}...'")
                if not result.top_1_match:
                    print(f"       Got: {result.actual_top_5[:3]}")

        return results

    def _keyword_match(self, query: str) -> List[str]:
        """Simple keyword matching fallback."""
        query_lower = query.lower()
        scores = []

        for op_id, doc in self.tool_docs.items():
            score = 0

            # Match operation_id parts
            op_parts = op_id.lower().replace('_', ' ').split()
            for part in op_parts:
                if len(part) > 2 and part in query_lower:
                    score += 3

            # Match purpose keywords
            purpose = doc.get('purpose', '').lower()
            query_words = [w for w in query_lower.split() if len(w) > 3]
            for word in query_words:
                if word in purpose:
                    score += 2

            # Match synonyms
            synonyms = ' '.join(doc.get('synonyms_hr', [])).lower()
            for word in query_words:
                if word in synonyms:
                    score += 2

            if score > 0:
                scores.append((op_id, score))

        scores.sort(key=lambda x: -x[1])
        return [s[0] for s in scores[:5]]

    async def run_full_test(self, limit: Optional[int] = None) -> AccuracyReport:
        """Run accuracy test on all tools."""
        report = AccuracyReport()

        tools_to_test = list(self.tool_docs.keys())
        if limit:
            tools_to_test = tools_to_test[:limit]

        report.total_tools = len(tools_to_test)

        print(f"\nTesting {len(tools_to_test)} tools...")
        print("=" * 60)

        for i, operation_id in enumerate(tools_to_test):
            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{len(tools_to_test)} | Top-1: {report.top_1_accuracy:.1f}%")

            try:
                results = await self.test_single_tool(operation_id)

                if not results:
                    report.skipped_tools.append(operation_id)
                    continue

                for result in results:
                    report.total_queries += 1
                    report.method_stats[result.method_used] += 1

                    if result.top_1_match:
                        report.top_1_correct += 1
                    if result.top_3_match:
                        report.top_3_correct += 1
                    if result.top_5_match:
                        report.top_5_correct += 1

                    if not result.top_5_match:
                        report.failed_tools.append(result)

                    if result.error:
                        report.errors.append(f"{operation_id}: {result.error}")

            except Exception as e:
                report.errors.append(f"{operation_id}: {str(e)}")

        return report

    def print_report(self, report: AccuracyReport):
        """Print accuracy report."""
        print("\n" + "=" * 60)
        print("TOOL ROUTING ACCURACY REPORT")
        print("=" * 60)

        print(f"\nTOTAL TOOLS TESTED: {report.total_tools}")
        print(f"TOTAL QUERIES: {report.total_queries}")
        print(f"SKIPPED (no queries): {len(report.skipped_tools)}")

        print(f"\n{'='*60}")
        print("SEARCH METHOD USAGE")
        print("=" * 60)
        for method, count in sorted(report.method_stats.items(), key=lambda x: -x[1]):
            pct = (count / report.total_queries * 100) if report.total_queries > 0 else 0
            print(f"  {method}: {count} ({pct:.1f}%)")

        print(f"\n{'='*60}")
        print("ACCURACY METRICS")
        print("=" * 60)
        print(f"  Top-1 Accuracy: {report.top_1_accuracy:.1f}% ({report.top_1_correct}/{report.total_queries})")
        print(f"  Top-3 Accuracy: {report.top_3_accuracy:.1f}% ({report.top_3_correct}/{report.total_queries})")
        print(f"  Top-5 Accuracy: {report.top_5_accuracy:.1f}% ({report.top_5_correct}/{report.total_queries})")

        # Grade
        if report.top_3_accuracy >= 90:
            grade = "EXCELLENT"
        elif report.top_3_accuracy >= 80:
            grade = "GOOD"
        elif report.top_3_accuracy >= 70:
            grade = "ACCEPTABLE"
        elif report.top_3_accuracy >= 60:
            grade = "NEEDS IMPROVEMENT"
        else:
            grade = "POOR"

        print(f"\n  OVERALL GRADE: {grade}")

        # Failed tools analysis
        if report.failed_tools:
            print(f"\n{'='*60}")
            print(f"FAILED TOOLS (not in Top-5): {len(report.failed_tools)}")
            print("=" * 60)

            # Group by tool
            by_tool = defaultdict(list)
            for result in report.failed_tools:
                by_tool[result.operation_id].append(result)

            # Show first 15 failures
            for op_id, results in sorted(by_tool.items())[:15]:
                print(f"\n  {op_id}:")
                for r in results[:1]:
                    print(f"    Query: '{r.query[:50]}...'")
                    print(f"    Got: {r.actual_top_5[:3]}")
                    print(f"    Method: {r.method_used}")

            if len(by_tool) > 15:
                print(f"\n  ... and {len(by_tool) - 15} more failed tools")

        # Errors
        if report.errors:
            print(f"\n{'='*60}")
            print(f"ERRORS: {len(report.errors)}")
            print("=" * 60)
            for err in report.errors[:5]:
                print(f"  - {err[:80]}")
            if len(report.errors) > 5:
                print(f"  ... and {len(report.errors) - 5} more errors")

        print("\n" + "=" * 60)


async def main():
    parser = argparse.ArgumentParser(description='Test tool routing accuracy')
    parser.add_argument('--limit', type=int, help='Limit number of tools to test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print("=" * 60)
    print("TOOL ROUTING ACCURACY TEST v2.0")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tester = ToolRoutingTester(verbose=args.verbose)

    try:
        await tester.initialize()
        report = await tester.run_full_test(limit=args.limit)
        tester.print_report(report)

        # Return exit code based on accuracy
        if report.top_3_accuracy >= 70:
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

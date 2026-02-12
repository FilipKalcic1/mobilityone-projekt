"""
Embedding Coverage Tracker - Measures PATH_ENTITY_MAP and OUTPUT_KEY_MAP coverage.

This module identifies gaps in the hardcoded dictionaries and tracks
what percentage of API paths/output keys are actually mapped.

Created to address the 4/10 rating issue: "No coverage measurement" → "Know gaps"
"""

import re
import logging
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field

from services.tool_contracts import UnifiedToolDefinition
from services.registry.embedding_engine import EmbeddingEngine

logger = logging.getLogger(__name__)


@dataclass
class CoverageReport:
    """Coverage statistics for embedding mappings."""

    # Path entity coverage
    total_path_segments: int = 0
    mapped_path_segments: int = 0
    unmapped_path_segments: Set[str] = field(default_factory=set)

    # Output key coverage
    total_output_keys: int = 0
    mapped_output_keys: int = 0
    unmapped_output_keys: Set[str] = field(default_factory=set)

    # OperationId coverage
    total_operation_words: int = 0
    mapped_operation_words: int = 0
    unmapped_operation_words: Set[str] = field(default_factory=set)

    # Tools without any mapping
    tools_with_mapping: int = 0
    tools_without_mapping: int = 0
    unmapped_tools: List[str] = field(default_factory=list)

    @property
    def path_coverage_pct(self) -> float:
        """Percentage of path segments that are mapped."""
        if self.total_path_segments == 0:
            return 0.0
        return (self.mapped_path_segments / self.total_path_segments) * 100

    @property
    def output_coverage_pct(self) -> float:
        """Percentage of output keys that are mapped."""
        if self.total_output_keys == 0:
            return 0.0
        return (self.mapped_output_keys / self.total_output_keys) * 100

    @property
    def operation_coverage_pct(self) -> float:
        """Percentage of operationId words that are mapped."""
        if self.total_operation_words == 0:
            return 0.0
        return (self.mapped_operation_words / self.total_operation_words) * 100

    @property
    def tool_coverage_pct(self) -> float:
        """Percentage of tools with at least one mapping."""
        total = self.tools_with_mapping + self.tools_without_mapping
        if total == 0:
            return 0.0
        return (self.tools_with_mapping / total) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "path_coverage": {
                "total": self.total_path_segments,
                "mapped": self.mapped_path_segments,
                "coverage_pct": round(self.path_coverage_pct, 2),
                "unmapped_count": len(self.unmapped_path_segments),
                "unmapped_top_20": sorted(self.unmapped_path_segments)[:20],
            },
            "output_coverage": {
                "total": self.total_output_keys,
                "mapped": self.mapped_output_keys,
                "coverage_pct": round(self.output_coverage_pct, 2),
                "unmapped_count": len(self.unmapped_output_keys),
                "unmapped_top_20": sorted(self.unmapped_output_keys)[:20],
            },
            "operation_coverage": {
                "total": self.total_operation_words,
                "mapped": self.mapped_operation_words,
                "coverage_pct": round(self.operation_coverage_pct, 2),
                "unmapped_count": len(self.unmapped_operation_words),
                "unmapped_top_20": sorted(self.unmapped_operation_words)[:20],
            },
            "tool_coverage": {
                "with_mapping": self.tools_with_mapping,
                "without_mapping": self.tools_without_mapping,
                "coverage_pct": round(self.tool_coverage_pct, 2),
                "unmapped_tools_sample": self.unmapped_tools[:10],
            },
            "summary": {
                "overall_quality": self._calculate_quality_score(),
                "recommendation": self._get_recommendation(),
            }
        }

    def _calculate_quality_score(self) -> str:
        """Calculate overall quality grade based on coverage."""
        avg_coverage = (
            self.path_coverage_pct +
            self.output_coverage_pct +
            self.tool_coverage_pct
        ) / 3

        if avg_coverage >= 90:
            return "A (Excellent)"
        elif avg_coverage >= 80:
            return "B (Good)"
        elif avg_coverage >= 70:
            return "C (Acceptable)"
        elif avg_coverage >= 50:
            return "D (Poor)"
        else:
            return "F (Critical)"

    def _get_recommendation(self) -> str:
        """Get improvement recommendation based on gaps."""
        issues = []

        if self.path_coverage_pct < 80:
            issues.append(f"PATH_ENTITY_MAP needs {len(self.unmapped_path_segments)} more entries")
        if self.output_coverage_pct < 80:
            issues.append(f"OUTPUT_KEY_MAP needs {len(self.unmapped_output_keys)} more entries")
        if self.tool_coverage_pct < 90:
            issues.append(f"{self.tools_without_mapping} tools have no Croatian mapping")

        if not issues:
            return "Coverage is acceptable"

        return "; ".join(issues)


class EmbeddingCoverageTracker:
    """
    Analyzes coverage of hardcoded mappings in EmbeddingEngine.

    Purpose: Know what percentage of API elements are mapped to Croatian.
    This addresses the "No coverage measurement" problem identified in the
    critical analysis (rated 1/10 for coverage tracking).
    """

    def __init__(self):
        self.engine = EmbeddingEngine()
        self.path_entity_map = self.engine.PATH_ENTITY_MAP
        self.output_key_map = self.engine.OUTPUT_KEY_MAP

    def analyze_coverage(
        self,
        tools: Dict[str, UnifiedToolDefinition]
    ) -> CoverageReport:
        """
        Analyze coverage of all tools.

        Args:
            tools: Dictionary of tool definitions

        Returns:
            CoverageReport with detailed statistics
        """
        report = CoverageReport()

        # Track unique segments/keys across all tools
        all_path_segments: Set[str] = set()
        all_output_keys: Set[str] = set()
        all_operation_words: Set[str] = set()

        for tool_id, tool in tools.items():
            # Analyze this tool
            path_result = self._analyze_path(tool.path)
            output_result = self._analyze_output_keys(tool.output_keys)
            op_result = self._analyze_operation_id(tool.operation_id)

            # Aggregate
            all_path_segments.update(path_result[0])
            all_output_keys.update(output_result[0])
            all_operation_words.update(op_result[0])

            report.unmapped_path_segments.update(path_result[1])
            report.unmapped_output_keys.update(output_result[1])
            report.unmapped_operation_words.update(op_result[1])

            # Check if this tool has ANY mapping
            has_mapping = (
                len(path_result[0]) - len(path_result[1]) > 0 or
                len(output_result[0]) - len(output_result[1]) > 0 or
                len(op_result[0]) - len(op_result[1]) > 0
            )

            if has_mapping:
                report.tools_with_mapping += 1
            else:
                report.tools_without_mapping += 1
                report.unmapped_tools.append(tool_id)

        # Calculate totals
        report.total_path_segments = len(all_path_segments)
        report.mapped_path_segments = len(all_path_segments) - len(report.unmapped_path_segments)

        report.total_output_keys = len(all_output_keys)
        report.mapped_output_keys = len(all_output_keys) - len(report.unmapped_output_keys)

        report.total_operation_words = len(all_operation_words)
        report.mapped_operation_words = len(all_operation_words) - len(report.unmapped_operation_words)

        return report

    def _analyze_path(self, path: str) -> Tuple[Set[str], Set[str]]:
        """
        Analyze path segments and return (all_segments, unmapped_segments).
        """
        if not path:
            return set(), set()

        all_segments = set()
        unmapped = set()

        # Remove path parameters
        clean_path = re.sub(r'\{[^}]+\}', '', path)
        segments = re.split(r'[/\-_]', clean_path.lower())

        for segment in segments:
            if not segment or len(segment) < 3:
                continue

            # Skip common API prefixes
            if segment in ('api', 'v1', 'v2', 'v3', 'odata'):
                continue

            all_segments.add(segment)

            # Check if mapped
            is_mapped = False
            if segment in self.path_entity_map:
                is_mapped = True
            else:
                # Partial match
                for key in self.path_entity_map:
                    if key in segment:
                        is_mapped = True
                        break

            if not is_mapped:
                unmapped.add(segment)

        return all_segments, unmapped

    def _analyze_output_keys(self, output_keys: List[str]) -> Tuple[Set[str], Set[str]]:
        """
        Analyze output keys and return (all_keys, unmapped_keys).
        """
        if not output_keys:
            return set(), set()

        all_keys = set()
        unmapped = set()

        for key in output_keys:
            key_lower = key.lower()
            all_keys.add(key_lower)

            # Check if mapped
            is_mapped = False
            for pattern in self.output_key_map:
                if pattern in key_lower:
                    is_mapped = True
                    break

            if not is_mapped:
                unmapped.add(key_lower)

        return all_keys, unmapped

    def _analyze_operation_id(self, operation_id: str) -> Tuple[Set[str], Set[str]]:
        """
        Analyze operationId words and return (all_words, unmapped_words).
        """
        if not operation_id:
            return set(), set()

        all_words = set()
        unmapped = set()

        # Split CamelCase
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', operation_id)

        # Common verbs to skip
        skip_words = {
            'get', 'create', 'update', 'delete', 'post', 'put', 'patch',
            'list', 'find', 'search', 'add', 'remove', 'set', 'fetch',
            'retrieve', 'check', 'validate', 'by', 'for', 'all', 'id',
            'the', 'and', 'or', 'a', 'an'
        }

        for word in words:
            word_lower = word.lower()

            if word_lower in skip_words or len(word_lower) < 3:
                continue

            all_words.add(word_lower)

            # Check if mapped
            is_mapped = (
                word_lower in self.path_entity_map or
                word_lower in self.output_key_map
            )

            if not is_mapped:
                unmapped.add(word_lower)

        return all_words, unmapped

    def print_report(self, report: CoverageReport) -> None:
        """Print human-readable coverage report."""
        print("\n" + "=" * 60)
        print("EMBEDDING COVERAGE REPORT")
        print("=" * 60)

        print(f"\n📍 PATH ENTITY COVERAGE:")
        print(f"   Total unique segments: {report.total_path_segments}")
        print(f"   Mapped segments: {report.mapped_path_segments}")
        print(f"   Coverage: {report.path_coverage_pct:.1f}%")
        if report.unmapped_path_segments:
            print(f"   Unmapped (top 10): {sorted(report.unmapped_path_segments)[:10]}")

        print(f"\n📦 OUTPUT KEY COVERAGE:")
        print(f"   Total unique keys: {report.total_output_keys}")
        print(f"   Mapped keys: {report.mapped_output_keys}")
        print(f"   Coverage: {report.output_coverage_pct:.1f}%")
        if report.unmapped_output_keys:
            print(f"   Unmapped (top 10): {sorted(report.unmapped_output_keys)[:10]}")

        print(f"\n🔧 OPERATION ID COVERAGE:")
        print(f"   Total unique words: {report.total_operation_words}")
        print(f"   Mapped words: {report.mapped_operation_words}")
        print(f"   Coverage: {report.operation_coverage_pct:.1f}%")
        if report.unmapped_operation_words:
            print(f"   Unmapped (top 10): {sorted(report.unmapped_operation_words)[:10]}")

        print(f"\n🛠️ TOOL COVERAGE:")
        print(f"   Tools with mapping: {report.tools_with_mapping}")
        print(f"   Tools without mapping: {report.tools_without_mapping}")
        print(f"   Coverage: {report.tool_coverage_pct:.1f}%")

        summary = report.to_dict()["summary"]
        print(f"\n📊 QUALITY GRADE: {summary['overall_quality']}")
        print(f"   {summary['recommendation']}")
        print("=" * 60 + "\n")


# Convenience function
def analyze_embedding_coverage(tools: Dict[str, UnifiedToolDefinition]) -> CoverageReport:
    """Analyze embedding coverage for given tools."""
    tracker = EmbeddingCoverageTracker()
    return tracker.analyze_coverage(tools)

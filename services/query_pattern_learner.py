"""
Query Pattern Learner - Learns from user corrections to improve tool selection.

This service analyzes hallucination_reports with corrections to:
1. Extract query patterns that led to wrong tool selection
2. Learn which tools should have been used
3. Build query→tool mappings for future improvement
4. Generate training data for embedding refinement

ARCHITECTURE:
    HallucinationReport (with correction)
              ↓
       QueryPatternLearner
              ↓
    ┌─────────┴─────────┐
    │                   │
    QueryToolMapping    PatternInsight
    (query → tool)      (why it failed)

DIFFERENCE FROM FeedbackAnalyzer:
- FeedbackAnalyzer: Finds missing DICTIONARY terms (keyword matching)
- QueryPatternLearner: Learns QUERY→TOOL mappings (semantic learning)

EXAMPLE:
    User query: "daj mi broj šasije za golf"
    Bot response: "Kilometraža je 45000km" (WRONG - used get_VehicleMileage)
    Correction: "Trebao je koristiti get_VehicleVIN"

    Learning:
    - Pattern: "broj šasije" → get_VehicleVIN
    - Pattern: "šasija" → get_VehicleVIN
    - Negative: "broj šasije" should NOT → get_VehicleMileage
"""

import logging
import re
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import HallucinationReport

logger = logging.getLogger(__name__)

# Persist learned mappings
LEARNED_MAPPINGS_FILE = Path.cwd() / ".cache" / "learned_query_mappings.json"


@dataclass
class QueryToolMapping:
    """A learned mapping from query pattern to correct tool."""
    pattern: str                    # Query pattern (e.g., "broj šasije")
    correct_tool: str               # Tool that should be used
    wrong_tools: Set[str] = field(default_factory=set)  # Tools that were wrongly used
    confidence: float = 0.0         # Based on occurrence count
    occurrence_count: int = 1
    sample_queries: List[str] = field(default_factory=list)
    learned_from: List[str] = field(default_factory=list)  # Report IDs

    def to_dict(self) -> Dict:
        return {
            "pattern": self.pattern,
            "correct_tool": self.correct_tool,
            "wrong_tools": list(self.wrong_tools),
            "confidence": round(self.confidence, 3),
            "occurrence_count": self.occurrence_count,
            "sample_queries": self.sample_queries[:3],
        }


@dataclass
class FailurePattern:
    """Analysis of why a query failed."""
    category: str                   # misunderstood, wrong_data, etc.
    query_type: str                 # What user was asking for
    tool_used: Optional[str]        # What tool was selected
    tool_needed: Optional[str]      # What tool should have been used
    root_cause: str                 # Why it failed
    count: int = 1

    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "query_type": self.query_type,
            "tool_used": self.tool_used,
            "tool_needed": self.tool_needed,
            "root_cause": self.root_cause,
            "count": self.count,
        }


@dataclass
class LearningResult:
    """Complete result from learning analysis."""
    total_analyzed: int = 0
    with_corrections: int = 0
    patterns_learned: int = 0

    # Learned mappings
    query_tool_mappings: List[QueryToolMapping] = field(default_factory=list)

    # Failure analysis
    failure_patterns: List[FailurePattern] = field(default_factory=list)

    # Category breakdown
    category_breakdown: Dict[str, int] = field(default_factory=dict)

    # High-value insights
    top_confused_queries: List[Tuple[str, str, str]] = field(default_factory=list)
    # (query, wrong_tool, right_tool)

    analyzed_at: str = ""

    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_analyzed": self.total_analyzed,
                "with_corrections": self.with_corrections,
                "patterns_learned": self.patterns_learned,
            },
            "query_tool_mappings": [m.to_dict() for m in self.query_tool_mappings[:20]],
            "failure_patterns": [p.to_dict() for p in self.failure_patterns[:10]],
            "category_breakdown": self.category_breakdown,
            "top_confused_queries": [
                {"query": q, "wrong_tool": w, "right_tool": r}
                for q, w, r in self.top_confused_queries[:10]
            ],
            "analyzed_at": self.analyzed_at,
        }


class QueryPatternLearner:
    """
    Learns query→tool mappings from hallucination corrections.

    Unlike simple keyword matching, this service:
    1. Extracts PATTERNS from user queries
    2. Identifies which tool SHOULD have been used
    3. Builds mappings that can improve future search ranking
    4. Analyzes WHY queries failed

    Usage:
        learner = QueryPatternLearner(db)
        result = await learner.learn()
        mappings = result.query_tool_mappings
    """

    # Common Croatian query patterns to extract
    QUERY_PATTERNS = [
        # Vehicle identification
        (r"broj\s+šasije|šasij[au]|vin", "vehicle_vin"),
        (r"registracij[au]|tablice|reg\.?\s*oznaka", "vehicle_registration"),
        (r"kilometraž[au]|km|prijeđen", "vehicle_mileage"),
        (r"goriv[ou]|tank|benzin|nafta|dizel", "vehicle_fuel"),
        (r"lokacij[au]|gdje\s+je|pozicij", "vehicle_location"),

        # Booking patterns
        (r"rezervacij[au]|booking|rezervira", "booking"),
        (r"slobodn[ao]|dostupn[ao]|slobodnih", "availability"),
        (r"otkaž|cancel|otkazi", "cancellation"),

        # Person patterns
        (r"vozač[au]?|driver", "driver"),
        (r"korisnik|kupac|klijent", "customer"),

        # Document patterns
        (r"ugovor|contract", "contract"),
        (r"račun|faktur|invoice", "invoice"),
        (r"izvještaj|report", "report"),

        # Maintenance patterns
        (r"servis|održavanj|popravak", "maintenance"),
        (r"štet[au]|damage|nesreć", "damage"),
        (r"osiguran[je]|insurance", "insurance"),
    ]

    # Tool name patterns to extract from corrections/responses
    TOOL_PATTERNS = [
        r"get_(\w+)",
        r"post_(\w+)",
        r"put_(\w+)",
        r"delete_(\w+)",
        r"koristiti\s+(\w+)",
        r"trebao?\s+(\w+)",
        r"tool:\s*(\w+)",
    ]

    def __init__(self, db: AsyncSession):
        """
        Initialize query pattern learner.

        Args:
            db: Async SQLAlchemy session
        """
        self.db = db
        self._mappings: Dict[str, QueryToolMapping] = {}
        self._load_existing_mappings()

        # Compile patterns
        self._query_patterns = [
            (re.compile(pattern, re.IGNORECASE), label)
            for pattern, label in self.QUERY_PATTERNS
        ]
        self._tool_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.TOOL_PATTERNS
        ]

    def _load_existing_mappings(self) -> None:
        """Load previously learned mappings."""
        try:
            if LEARNED_MAPPINGS_FILE.exists():
                with open(LEARNED_MAPPINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for m in data.get("mappings", []):
                        mapping = QueryToolMapping(
                            pattern=m["pattern"],
                            correct_tool=m["correct_tool"],
                            wrong_tools=set(m.get("wrong_tools", [])),
                            confidence=m.get("confidence", 0.5),
                            occurrence_count=m.get("occurrence_count", 1),
                            sample_queries=m.get("sample_queries", []),
                        )
                        self._mappings[m["pattern"]] = mapping
                logger.info(f"Loaded {len(self._mappings)} existing query mappings")
        except Exception as e:
            logger.warning(f"Could not load existing mappings: {e}")

    def _save_mappings(self) -> None:
        """Save learned mappings to file."""
        try:
            LEARNED_MAPPINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": "1.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "mappings": [
                    {
                        "pattern": m.pattern,
                        "correct_tool": m.correct_tool,
                        "wrong_tools": list(m.wrong_tools),
                        "confidence": m.confidence,
                        "occurrence_count": m.occurrence_count,
                        "sample_queries": m.sample_queries[:5],
                    }
                    for m in self._mappings.values()
                ]
            }
            with open(LEARNED_MAPPINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self._mappings)} query mappings")
        except Exception as e:
            logger.error(f"Failed to save mappings: {e}")

    def _extract_query_type(self, query: str) -> Optional[str]:
        """Extract the type of query from user text."""
        query_lower = query.lower()
        for pattern, label in self._query_patterns:
            if pattern.search(query_lower):
                return label
        return None

    def _extract_tool_from_text(self, text: str) -> Optional[str]:
        """Extract tool name from correction or response text."""
        if not text:
            return None

        text_lower = text.lower()

        # Try each pattern
        for pattern in self._tool_patterns:
            match = pattern.search(text_lower)
            if match:
                tool = match.group(1)
                # Normalize tool name
                if not tool.startswith(("get_", "post_", "put_", "delete_")):
                    # Try to find full tool name
                    full_match = re.search(
                        r"(get|post|put|delete)_" + re.escape(tool),
                        text_lower
                    )
                    if full_match:
                        return full_match.group(0)
                return tool

        # Look for common tool references
        tool_refs = re.findall(r"(get_\w+|post_\w+|put_\w+|delete_\w+)", text_lower)
        if tool_refs:
            return tool_refs[0]

        return None

    def _extract_wrong_tool(self, report: HallucinationReport) -> Optional[str]:
        """Extract the tool that was wrongly used."""
        # Check retrieved_chunks first
        if report.retrieved_chunks:
            chunks = report.retrieved_chunks
            if isinstance(chunks, list) and chunks:
                for chunk in chunks:
                    if isinstance(chunk, str) and "_" in chunk:
                        return chunk
                    elif isinstance(chunk, dict):
                        return chunk.get("tool") or chunk.get("operation_id")

        # Try to extract from bot_response
        return self._extract_tool_from_text(report.bot_response)

    def _extract_patterns_from_query(self, query: str) -> List[str]:
        """Extract meaningful patterns from a query."""
        patterns = []
        query_lower = query.lower()

        # Remove common words
        stopwords = {
            "mi", "me", "ja", "ti", "on", "ona", "ono", "daj", "dajte",
            "molim", "te", "vas", "trebam", "treba", "hoću", "želim",
            "mogu", "možeš", "može", "li", "da", "ne", "i", "ili",
            "za", "od", "do", "na", "u", "s", "sa", "po", "iz",
        }

        # Split into words
        words = re.findall(r'\b\w+\b', query_lower)
        words = [w for w in words if w not in stopwords and len(w) > 2]

        # Create n-grams (1, 2, 3 words)
        for n in [2, 3, 1]:  # Prefer longer patterns
            for i in range(len(words) - n + 1):
                pattern = " ".join(words[i:i+n])
                if len(pattern) > 3:
                    patterns.append(pattern)

        return patterns[:5]  # Top 5 patterns

    def _analyze_failure_reason(
        self,
        report: HallucinationReport,
        wrong_tool: Optional[str],
        right_tool: Optional[str]
    ) -> FailurePattern:
        """Analyze why a query failed."""
        category = report.category or "unknown"
        query_type = self._extract_query_type(report.user_query) or "unknown"

        # Determine root cause
        if category == "misunderstood":
            root_cause = "Query intent not recognized"
        elif category == "wrong_data":
            root_cause = "Correct tool but wrong data returned"
        elif category == "hallucination":
            root_cause = "Bot fabricated information"
        elif wrong_tool and right_tool:
            root_cause = f"Wrong tool selected: {wrong_tool} instead of {right_tool}"
        else:
            root_cause = "Unknown failure reason"

        return FailurePattern(
            category=category,
            query_type=query_type,
            tool_used=wrong_tool,
            tool_needed=right_tool,
            root_cause=root_cause,
        )

    async def learn(
        self,
        min_occurrences: int = 1,
        include_uncorrected: bool = False,
        limit: int = 1000
    ) -> LearningResult:
        """
        Learn query→tool mappings from hallucination reports.

        Args:
            min_occurrences: Minimum pattern occurrences to include
            include_uncorrected: Include reports without corrections
            limit: Maximum reports to analyze

        Returns:
            LearningResult with mappings and insights
        """
        result = LearningResult(
            analyzed_at=datetime.now(timezone.utc).isoformat()
        )

        # Build query - prefer reports with corrections
        query = select(HallucinationReport).limit(limit)
        if not include_uncorrected:
            query = query.where(HallucinationReport.correction.isnot(None))

        # Execute query
        try:
            db_result = await self.db.execute(query)
            reports = db_result.scalars().all()
        except Exception as e:
            logger.error(f"Failed to fetch reports: {e}")
            return result

        result.total_analyzed = len(reports)

        # Track patterns and failures
        pattern_counts: Counter = Counter()
        pattern_tools: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        failure_counts: Dict[str, FailurePattern] = {}
        confused_queries: List[Tuple[str, str, str]] = []

        for report in reports:
            # Track category
            category = report.category or "uncategorized"
            result.category_breakdown[category] = \
                result.category_breakdown.get(category, 0) + 1

            if report.correction:
                result.with_corrections += 1

            # Extract tools
            wrong_tool = self._extract_wrong_tool(report)
            right_tool = self._extract_tool_from_text(report.correction) if report.correction else None

            # Analyze failure
            failure = self._analyze_failure_reason(report, wrong_tool, right_tool)
            failure_key = f"{failure.category}:{failure.query_type}"
            if failure_key in failure_counts:
                failure_counts[failure_key].count += 1
            else:
                failure_counts[failure_key] = failure

            # Extract patterns if we know the right tool
            if right_tool:
                patterns = self._extract_patterns_from_query(report.user_query)
                for pattern in patterns:
                    pattern_counts[pattern] += 1
                    pattern_tools[pattern][right_tool] += 1

                    # Track confused queries
                    if wrong_tool and wrong_tool != right_tool:
                        confused_queries.append((
                            report.user_query[:100],
                            wrong_tool,
                            right_tool
                        ))

        # Build mappings from patterns
        for pattern, count in pattern_counts.items():
            if count < min_occurrences:
                continue

            # Find most common tool for this pattern
            tool_counts = pattern_tools[pattern]
            if not tool_counts:
                continue

            correct_tool = max(tool_counts.items(), key=lambda x: x[1])[0]
            total_for_tool = sum(tool_counts.values())

            # Calculate confidence
            confidence = tool_counts[correct_tool] / total_for_tool if total_for_tool > 0 else 0

            # Create or update mapping
            if pattern in self._mappings:
                mapping = self._mappings[pattern]
                mapping.occurrence_count += count
                mapping.confidence = (mapping.confidence + confidence) / 2
            else:
                mapping = QueryToolMapping(
                    pattern=pattern,
                    correct_tool=correct_tool,
                    confidence=confidence,
                    occurrence_count=count,
                )
                self._mappings[pattern] = mapping

        # Sort mappings by confidence and count
        sorted_mappings = sorted(
            self._mappings.values(),
            key=lambda m: (m.confidence, m.occurrence_count),
            reverse=True
        )

        result.query_tool_mappings = sorted_mappings
        result.patterns_learned = len(sorted_mappings)
        result.failure_patterns = sorted(
            failure_counts.values(),
            key=lambda f: f.count,
            reverse=True
        )
        result.top_confused_queries = confused_queries[:20]

        # Save learned mappings
        self._save_mappings()

        logger.info(
            f"Learning complete: {result.total_analyzed} reports, "
            f"{result.with_corrections} with corrections, "
            f"{result.patterns_learned} patterns learned"
        )

        return result

    def get_tool_for_query(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Get recommended tool for a query based on learned patterns.

        Args:
            query: User query text

        Returns:
            Tuple of (tool_name, confidence) or None
        """
        query_lower = query.lower()
        best_match: Optional[Tuple[str, float]] = None

        for pattern, mapping in self._mappings.items():
            if pattern in query_lower:
                if best_match is None or mapping.confidence > best_match[1]:
                    best_match = (mapping.correct_tool, mapping.confidence)

        return best_match

    def get_negative_signals(self, query: str) -> List[str]:
        """
        Get tools that should NOT be used for this query.

        Based on learned wrong_tools associations.

        Args:
            query: User query text

        Returns:
            List of tool names to avoid
        """
        query_lower = query.lower()
        avoid_tools: Set[str] = set()

        for pattern, mapping in self._mappings.items():
            if pattern in query_lower:
                avoid_tools.update(mapping.wrong_tools)

        return list(avoid_tools)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_mappings": len(self._mappings),
            "high_confidence_mappings": sum(
                1 for m in self._mappings.values() if m.confidence > 0.7
            ),
            "total_occurrences": sum(
                m.occurrence_count for m in self._mappings.values()
            ),
            "top_patterns": [
                {"pattern": m.pattern, "tool": m.correct_tool, "confidence": m.confidence}
                for m in sorted(
                    self._mappings.values(),
                    key=lambda x: x.confidence,
                    reverse=True
                )[:10]
            ],
        }


# Convenience function
async def learn_from_feedback(db: AsyncSession) -> LearningResult:
    """Run learning and return results."""
    learner = QueryPatternLearner(db)
    return await learner.learn()

"""
Feedback Learning Service - Complete self-improving feedback loop.
Version: 1.0

This service closes the feedback loop by:
1. Learning from hallucination corrections (QueryPatternLearner)
2. Applying learned patterns to SearchEngine (real-time boosting)
3. Using embedding similarity for semantic pattern matching
4. Auto-generating tool documentation from high-confidence patterns
5. Tracking quality metrics and detecting degradation

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    FeedbackLearningService                       │
    │                                                                  │
    │  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
    │  │QueryPattern  │   │ Embedding    │   │ QualityTracker   │    │
    │  │Learner       │   │ Similarity   │   │                  │    │
    │  └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘    │
    │         │                  │                     │              │
    │         └──────────────────┴─────────────────────┘              │
    │                            │                                    │
    │                   ┌────────▼────────┐                          │
    │                   │ LearnedBoosts   │                          │
    │                   │ (confidence>0.7)│                          │
    │                   └────────┬────────┘                          │
    │                            │                                    │
    └────────────────────────────┼────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     SearchEngine        │
                    │ _apply_learned_boosting │
                    └─────────────────────────┘

USAGE:
    # Initialize service
    service = FeedbackLearningService(db)

    # Learn from feedback (call periodically or after review)
    result = await service.learn_and_apply()

    # Get boost for a query (called by SearchEngine)
    boost = await service.get_search_boost(query, tool_id)

    # Export learned patterns to tool documentation
    await service.export_to_documentation()
"""

import json
import logging
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from models import HallucinationReport
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Cache file for learned boosts
LEARNED_BOOSTS_FILE = Path.cwd() / ".cache" / "learned_search_boosts.json"

# Thresholds
CONFIDENCE_THRESHOLD_AUTO_APPLY = 0.70  # Auto-apply patterns with >70% confidence
CONFIDENCE_THRESHOLD_DOCUMENTATION = 0.85  # Add to docs with >85% confidence
MIN_OCCURRENCES_FOR_LEARNING = 2  # Need at least 2 examples
EMBEDDING_SIMILARITY_THRESHOLD = 0.75  # For semantic matching
MAX_BOOSTS_PER_TOOL = 10  # Limit patterns per tool


@dataclass
class LearnedBoost:
    """A learned score adjustment for a tool."""
    tool_id: str
    patterns: List[str]  # Query patterns that should boost this tool
    boost_value: float  # +0.1 to +0.3 typically
    negative_patterns: List[str] = field(default_factory=list)  # Patterns that should NOT use this tool
    penalty_value: float = 0.0  # Penalty for negative patterns
    confidence: float = 0.0
    occurrence_count: int = 0
    last_updated: str = ""
    source: str = "feedback"  # "feedback", "evaluation", "manual"

    def to_dict(self) -> Dict:
        return {
            "tool_id": self.tool_id,
            "patterns": self.patterns[:10],
            "boost_value": round(self.boost_value, 3),
            "negative_patterns": self.negative_patterns[:10],
            "penalty_value": round(self.penalty_value, 3),
            "confidence": round(self.confidence, 3),
            "occurrence_count": self.occurrence_count,
            "last_updated": self.last_updated,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LearnedBoost":
        return cls(
            tool_id=data["tool_id"],
            patterns=data.get("patterns", []),
            boost_value=data.get("boost_value", 0.0),
            negative_patterns=data.get("negative_patterns", []),
            penalty_value=data.get("penalty_value", 0.0),
            confidence=data.get("confidence", 0.0),
            occurrence_count=data.get("occurrence_count", 0),
            last_updated=data.get("last_updated", ""),
            source=data.get("source", "feedback"),
        )


@dataclass
class LearningResult:
    """Result of a learning cycle."""
    patterns_learned: int = 0
    boosts_applied: int = 0
    penalties_applied: int = 0
    documentation_updates: int = 0
    quality_status: str = ""
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "patterns_learned": self.patterns_learned,
            "boosts_applied": self.boosts_applied,
            "penalties_applied": self.penalties_applied,
            "documentation_updates": self.documentation_updates,
            "quality_status": self.quality_status,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


class FeedbackLearningService:
    """
    Complete feedback learning service that closes the loop.

    Features:
    - Learns query→tool patterns from corrections
    - Applies learned boosts to SearchEngine
    - Uses embedding similarity for semantic matching
    - Tracks quality metrics
    - Auto-generates tool documentation
    """

    def __init__(
        self,
        db: AsyncSession,
        embedding_client: Optional[Any] = None
    ):
        """
        Initialize the feedback learning service.

        Args:
            db: Async SQLAlchemy session
            embedding_client: Optional OpenAI client for embeddings
        """
        self.db = db
        self._embedding_client = embedding_client

        # Load existing learned boosts
        self._learned_boosts: Dict[str, LearnedBoost] = {}
        self._pattern_embeddings: Dict[str, List[float]] = {}
        self._load_learned_boosts()

        logger.info(
            f"FeedbackLearningService initialized with {len(self._learned_boosts)} "
            f"existing boosts"
        )

    def _load_learned_boosts(self) -> None:
        """Load previously learned boosts from cache."""
        try:
            if LEARNED_BOOSTS_FILE.exists():
                with open(LEARNED_BOOSTS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for boost_data in data.get("boosts", []):
                        boost = LearnedBoost.from_dict(boost_data)
                        self._learned_boosts[boost.tool_id] = boost
                    self._pattern_embeddings = data.get("pattern_embeddings", {})
                logger.info(f"Loaded {len(self._learned_boosts)} learned boosts")
        except Exception as e:
            logger.warning(f"Could not load learned boosts: {e}")

    def _save_learned_boosts(self) -> None:
        """Save learned boosts to cache."""
        try:
            LEARNED_BOOSTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": "1.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "boosts": [b.to_dict() for b in self._learned_boosts.values()],
                "pattern_embeddings": self._pattern_embeddings,
            }
            with open(LEARNED_BOOSTS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self._learned_boosts)} learned boosts")
        except Exception as e:
            logger.error(f"Failed to save learned boosts: {e}")

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using OpenAI."""
        if not self._embedding_client:
            try:
                from openai import AsyncAzureOpenAI
                self._embedding_client = AsyncAzureOpenAI(
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION
                )
            except Exception as e:
                logger.warning(f"Could not initialize embedding client: {e}")
                return None

        try:
            response = await self._embedding_client.embeddings.create(
                input=[text[:8000]],
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding error for '{text[:50]}...': {e}")
            return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def learn_and_apply(
        self,
        min_occurrences: int = MIN_OCCURRENCES_FOR_LEARNING,
        confidence_threshold: float = CONFIDENCE_THRESHOLD_AUTO_APPLY,
        limit: int = 1000
    ) -> LearningResult:
        """
        Complete learning cycle: analyze feedback and apply learnings.

        This method:
        1. Fetches unprocessed hallucination reports
        2. Extracts query→tool patterns from corrections
        3. Computes confidence scores
        4. Applies high-confidence patterns as search boosts
        5. Optionally updates tool documentation

        Args:
            min_occurrences: Minimum pattern occurrences to learn
            confidence_threshold: Minimum confidence to auto-apply
            limit: Maximum reports to analyze

        Returns:
            LearningResult with statistics
        """
        result = LearningResult(
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Step 1: Fetch reports with corrections
        try:
            query = (
                select(HallucinationReport)
                .where(HallucinationReport.correction.isnot(None))
                .order_by(HallucinationReport.created_at.desc())
                .limit(limit)
            )
            db_result = await self.db.execute(query)
            reports = list(db_result.scalars().all())
        except Exception as e:
            logger.error(f"Failed to fetch reports: {e}")
            result.recommendations.append(f"Database error: {e}")
            return result

        if not reports:
            result.quality_status = "no_data"
            result.recommendations.append("No corrected reports found for learning")
            return result

        # Step 2: Extract patterns from corrections
        pattern_tool_map: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"tool": None, "count": 0, "queries": [], "wrong_tools": set()}
        )

        for report in reports:
            correct_tool = self._extract_tool_from_correction(report.correction)
            if not correct_tool:
                continue

            # Extract patterns from query
            patterns = self._extract_patterns(report.user_query)
            wrong_tool = self._extract_tool_from_response(report.bot_response, report.retrieved_chunks)

            for pattern in patterns:
                data = pattern_tool_map[pattern]
                data["tool"] = correct_tool
                data["count"] += 1
                if len(data["queries"]) < 5:
                    data["queries"].append(report.user_query[:100])
                if wrong_tool:
                    data["wrong_tools"].add(wrong_tool)

        # Step 3: Build learned boosts
        tool_patterns: Dict[str, LearnedBoost] = defaultdict(
            lambda: LearnedBoost(tool_id="", patterns=[], boost_value=0.0)
        )

        for pattern, data in pattern_tool_map.items():
            if data["count"] < min_occurrences:
                continue

            tool_id = data["tool"]
            if not tool_id:
                continue

            # Calculate confidence
            confidence = min(0.5 + (data["count"] * 0.1), 0.95)

            # Get or create boost
            if tool_id not in tool_patterns:
                tool_patterns[tool_id] = LearnedBoost(
                    tool_id=tool_id,
                    patterns=[],
                    boost_value=0.0,
                    confidence=0.0,
                    occurrence_count=0,
                )

            boost = tool_patterns[tool_id]

            # Add pattern if not too many
            if len(boost.patterns) < MAX_BOOSTS_PER_TOOL and pattern not in boost.patterns:
                boost.patterns.append(pattern)
                boost.occurrence_count += data["count"]
                boost.confidence = max(boost.confidence, confidence)

            # Add wrong tools as negative patterns
            for wrong_tool in data["wrong_tools"]:
                if wrong_tool not in boost.negative_patterns:
                    boost.negative_patterns.append(wrong_tool)

            result.patterns_learned += 1

        # Step 4: Apply high-confidence boosts
        now = datetime.now(timezone.utc).isoformat()

        for tool_id, boost in tool_patterns.items():
            if boost.confidence < confidence_threshold:
                continue

            # Calculate boost value based on confidence
            boost.boost_value = 0.10 + (boost.confidence - 0.5) * 0.4  # 0.10 to 0.28
            boost.penalty_value = 0.15  # Penalty for wrong tool
            boost.last_updated = now
            boost.source = "feedback"

            # Merge with existing boost
            if tool_id in self._learned_boosts:
                existing = self._learned_boosts[tool_id]
                # Merge patterns
                for p in boost.patterns:
                    if p not in existing.patterns and len(existing.patterns) < MAX_BOOSTS_PER_TOOL:
                        existing.patterns.append(p)
                # Update confidence (weighted average)
                existing.confidence = (existing.confidence + boost.confidence) / 2
                existing.occurrence_count += boost.occurrence_count
                existing.boost_value = max(existing.boost_value, boost.boost_value)
                existing.last_updated = now
                result.boosts_applied += 1
            else:
                self._learned_boosts[tool_id] = boost
                result.boosts_applied += 1

        # Step 5: Save learned boosts
        if result.boosts_applied > 0:
            self._save_learned_boosts()

        # Step 6: Generate recommendations
        result.recommendations = self._generate_recommendations(tool_patterns, result)

        # Step 7: Update quality status
        result.quality_status = "learning_applied" if result.boosts_applied > 0 else "no_new_patterns"

        logger.info(
            f"Learning cycle complete: {result.patterns_learned} patterns, "
            f"{result.boosts_applied} boosts applied"
        )

        return result

    def _extract_tool_from_correction(self, correction: str) -> Optional[str]:
        """Extract tool name from correction text."""
        if not correction:
            return None

        import re
        # Look for tool patterns
        patterns = [
            r"(get_\w+)",
            r"(post_\w+)",
            r"(put_\w+)",
            r"(patch_\w+)",
            r"(delete_\w+)",
            r"koristiti\s+(\w+)",
            r"trebao?\s+(\w+)",
        ]

        correction_lower = correction.lower()
        for pattern in patterns:
            match = re.search(pattern, correction_lower)
            if match:
                tool = match.group(1)
                # Normalize
                if not tool.startswith(("get_", "post_", "put_", "patch_", "delete_")):
                    # Try to find full tool name
                    full_match = re.search(
                        r"(get|post|put|patch|delete)_\w*" + re.escape(tool),
                        correction_lower
                    )
                    if full_match:
                        return full_match.group(0)
                return tool

        return None

    def _extract_tool_from_response(
        self,
        response: Optional[str],
        chunks: Optional[Any]
    ) -> Optional[str]:
        """Extract the tool that was used (wrong tool)."""
        # Check chunks first
        if chunks:
            if isinstance(chunks, list) and chunks:
                for chunk in chunks:
                    if isinstance(chunk, str) and "_" in chunk:
                        return chunk.lower()
                    elif isinstance(chunk, dict):
                        tool = chunk.get("tool") or chunk.get("operation_id")
                        if tool:
                            return tool.lower()

        # Extract from response
        if response:
            return self._extract_tool_from_correction(response)

        return None

    def _extract_patterns(self, query: str) -> List[str]:
        """Extract meaningful patterns from a query."""
        if not query:
            return []

        # Croatian stopwords
        stopwords = {
            "mi", "me", "ja", "ti", "on", "ona", "ono", "daj", "dajte",
            "molim", "te", "vas", "trebam", "treba", "hoću", "želim",
            "mogu", "možeš", "može", "li", "da", "ne", "i", "ili",
            "za", "od", "do", "na", "u", "s", "sa", "po", "iz",
            "je", "su", "sam", "si", "smo", "ste", "biti", "bio",
            "koja", "koji", "koje", "što", "kako", "gdje", "kada",
            "prikaži", "pokaži", "reci", "kaži", "dohvati",
        }

        import re
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)
        words = [w for w in words if w not in stopwords and len(w) > 2]

        patterns = []
        # Create n-grams (prefer 2-3 words)
        for n in [2, 3, 1]:
            for i in range(len(words) - n + 1):
                pattern = " ".join(words[i:i+n])
                if len(pattern) > 4:
                    patterns.append(pattern)

        return patterns[:5]

    def _generate_recommendations(
        self,
        tool_patterns: Dict[str, LearnedBoost],
        result: LearningResult
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if result.patterns_learned == 0:
            recommendations.append(
                "No patterns learned. Ensure admins are adding corrections to reports."
            )

        # Find tools with most patterns
        sorted_tools = sorted(
            tool_patterns.values(),
            key=lambda x: x.occurrence_count,
            reverse=True
        )

        if sorted_tools:
            top_tool = sorted_tools[0]
            recommendations.append(
                f"Most confused tool: {top_tool.tool_id} "
                f"({top_tool.occurrence_count} corrections needed)"
            )

        # Check for low confidence patterns
        low_conf = [b for b in tool_patterns.values() if 0.5 <= b.confidence < 0.7]
        if low_conf:
            recommendations.append(
                f"{len(low_conf)} patterns have low confidence (<70%). "
                "More examples needed for auto-application."
            )

        return recommendations

    def get_search_boost(
        self,
        query: str,
        tool_id: str
    ) -> Tuple[float, str]:
        """
        Get the learned boost/penalty for a tool given a query.

        This method is called by SearchEngine during tool scoring.

        Args:
            query: User query
            tool_id: Tool being scored

        Returns:
            Tuple of (adjustment_value, reason)
            Positive = boost, Negative = penalty
        """
        query_lower = query.lower()

        # Check if tool has learned boosts
        boost = self._learned_boosts.get(tool_id.lower())
        if not boost:
            # Check if query matches negative patterns of other tools
            for other_boost in self._learned_boosts.values():
                if tool_id.lower() in [np.lower() for np in other_boost.negative_patterns]:
                    for pattern in other_boost.patterns:
                        if pattern in query_lower:
                            return (-other_boost.penalty_value, f"negative_learned:{pattern}")
            return (0.0, "no_learned_pattern")

        # Check positive patterns (boost this tool)
        for pattern in boost.patterns:
            if pattern in query_lower:
                return (boost.boost_value, f"learned_boost:{pattern}")

        return (0.0, "no_pattern_match")

    async def get_search_boost_semantic(
        self,
        query: str,
        tool_id: str,
        query_embedding: Optional[List[float]] = None
    ) -> Tuple[float, str]:
        """
        Get learned boost using semantic similarity.

        More accurate than string matching but requires embeddings.

        Args:
            query: User query
            tool_id: Tool being scored
            query_embedding: Pre-computed query embedding (optional)

        Returns:
            Tuple of (adjustment_value, reason)
        """
        # First try string matching (fast)
        string_boost = self.get_search_boost(query, tool_id)
        if string_boost[0] != 0.0:
            return string_boost

        # If no string match, try semantic similarity
        boost = self._learned_boosts.get(tool_id.lower())
        if not boost or not boost.patterns:
            return (0.0, "no_learned_pattern")

        # Get query embedding
        if query_embedding is None:
            query_embedding = await self._get_embedding(query)
        if not query_embedding:
            return (0.0, "no_embedding")

        # Check semantic similarity with learned patterns
        for pattern in boost.patterns:
            # Get or compute pattern embedding
            pattern_key = f"{tool_id}:{pattern}"
            if pattern_key not in self._pattern_embeddings:
                pattern_embedding = await self._get_embedding(pattern)
                if pattern_embedding:
                    self._pattern_embeddings[pattern_key] = pattern_embedding

            pattern_embedding = self._pattern_embeddings.get(pattern_key)
            if not pattern_embedding:
                continue

            similarity = self._cosine_similarity(query_embedding, pattern_embedding)
            if similarity >= EMBEDDING_SIMILARITY_THRESHOLD:
                return (boost.boost_value * similarity, f"semantic_boost:{pattern}:{similarity:.2f}")

        return (0.0, "no_semantic_match")

    async def export_to_documentation(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD_DOCUMENTATION,
        output_path: Optional[Path] = None
    ) -> int:
        """
        Export high-confidence patterns to tool_documentation.json.

        This enables automatic improvement of tool descriptions with
        real user query examples.

        Args:
            confidence_threshold: Minimum confidence to export
            output_path: Custom output path (default: config/tool_documentation.json)

        Returns:
            Number of tools updated
        """
        if output_path is None:
            output_path = Path.cwd() / "config" / "tool_documentation.json"

        # Load existing documentation
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load tool documentation: {e}")
            docs = {}

        updates = 0

        for tool_id, boost in self._learned_boosts.items():
            if boost.confidence < confidence_threshold:
                continue

            # Get or create tool doc
            if tool_id not in docs:
                docs[tool_id] = {}

            tool_doc = docs[tool_id]

            # Add example queries
            if "example_queries_hr" not in tool_doc:
                tool_doc["example_queries_hr"] = []

            existing_examples = set(tool_doc["example_queries_hr"])

            for pattern in boost.patterns[:3]:
                # Convert pattern to example query
                example = f"{pattern}"
                if example not in existing_examples:
                    tool_doc["example_queries_hr"].append(example)
                    updates += 1

            # Add auto-generated note
            tool_doc["_auto_learned"] = {
                "confidence": boost.confidence,
                "occurrence_count": boost.occurrence_count,
                "updated_at": boost.last_updated,
            }

        # Save updated documentation
        if updates > 0:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(docs, f, indent=2, ensure_ascii=False)
                logger.info(f"Exported {updates} learned patterns to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save documentation: {e}")

        return updates

    async def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        total_patterns = sum(len(b.patterns) for b in self._learned_boosts.values())
        high_confidence = sum(1 for b in self._learned_boosts.values() if b.confidence >= 0.8)
        total_occurrences = sum(b.occurrence_count for b in self._learned_boosts.values())

        # Get top tools by occurrence
        sorted_boosts = sorted(
            self._learned_boosts.values(),
            key=lambda x: x.occurrence_count,
            reverse=True
        )

        return {
            "total_learned_tools": len(self._learned_boosts),
            "total_patterns": total_patterns,
            "high_confidence_tools": high_confidence,
            "total_occurrences": total_occurrences,
            "pattern_embeddings_cached": len(self._pattern_embeddings),
            "top_learned_tools": [
                {
                    "tool_id": b.tool_id,
                    "patterns": len(b.patterns),
                    "confidence": b.confidence,
                    "occurrences": b.occurrence_count,
                }
                for b in sorted_boosts[:10]
            ],
            "cache_file": str(LEARNED_BOOSTS_FILE),
        }

    def clear_learned_boosts(self) -> None:
        """Clear all learned boosts (for testing/reset)."""
        self._learned_boosts.clear()
        self._pattern_embeddings.clear()
        if LEARNED_BOOSTS_FILE.exists():
            LEARNED_BOOSTS_FILE.unlink()
        logger.info("Cleared all learned boosts")


# Singleton instance
_feedback_learning_service: Optional[FeedbackLearningService] = None


def get_feedback_learning_service(db: AsyncSession) -> FeedbackLearningService:
    """Get or create feedback learning service instance.

    Updates the db session each call to avoid stale sessions.
    """
    global _feedback_learning_service
    if _feedback_learning_service is None:
        _feedback_learning_service = FeedbackLearningService(db)
    else:
        _feedback_learning_service.db = db
    return _feedback_learning_service


async def run_learning_cycle(db: AsyncSession) -> LearningResult:
    """Convenience function to run a learning cycle."""
    service = get_feedback_learning_service(db)
    return await service.learn_and_apply()

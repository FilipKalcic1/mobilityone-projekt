"""
Feedback Analyzer - Learns from user corrections to improve search quality.
Version: 1.0

This service closes the feedback loop:
1. Users report errors ("krivo", "pogrešno")
2. Admins review and add corrections
3. FeedbackAnalyzer identifies patterns
4. Suggests dictionary additions
5. System improves automatically

ARCHITECTURE:
    HallucinationReport (DB) → FeedbackAnalyzer → DictionarySuggestions
                                                       ↓
                                              Admin Approval
                                                       ↓
                                              Dictionary Update
                                                       ↓
                                              Re-index Embeddings

CRITICAL: This service only SUGGESTS changes, never auto-applies them.
Human review is required for dictionary updates.
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import json

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from models import HallucinationReport

logger = logging.getLogger(__name__)


@dataclass
class TermFrequency:
    """A term found in user queries with frequency data."""
    term: str
    count: int = 1
    queries: List[str] = field(default_factory=list)
    categories: Set[str] = field(default_factory=set)

    def add_occurrence(self, query: str, category: Optional[str] = None):
        """Record another occurrence of this term."""
        self.count += 1
        if len(self.queries) < 5:  # Keep only sample queries
            self.queries.append(query[:100])
        if category:
            self.categories.add(category)


@dataclass
class DictionarySuggestion:
    """A suggested addition to the embedding dictionaries."""
    term: str
    suggested_mapping: str
    dictionary_type: str  # "path_entity", "output_key", "synonym"
    confidence: float
    frequency: int
    sample_queries: List[str]
    reason: str

    def to_dict(self) -> Dict:
        return {
            "term": self.term,
            "suggested_mapping": self.suggested_mapping,
            "dictionary_type": self.dictionary_type,
            "confidence": round(self.confidence, 2),
            "frequency": self.frequency,
            "sample_queries": self.sample_queries[:3],
            "reason": self.reason,
        }


@dataclass
class FeedbackAnalysisResult:
    """Complete analysis of feedback data."""
    total_reports_analyzed: int = 0
    reports_with_corrections: int = 0

    # Missing terms found in queries
    missing_croatian_terms: Dict[str, TermFrequency] = field(default_factory=dict)

    # Patterns in failures
    common_failure_patterns: List[Tuple[str, int]] = field(default_factory=list)

    # Category breakdown
    category_distribution: Dict[str, int] = field(default_factory=dict)

    # Generated suggestions
    suggestions: List[DictionarySuggestion] = field(default_factory=list)

    # Analysis metadata
    analyzed_at: str = ""

    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_analyzed": self.total_reports_analyzed,
                "with_corrections": self.reports_with_corrections,
                "suggestions_generated": len(self.suggestions),
            },
            "category_distribution": self.category_distribution,
            "common_failure_patterns": self.common_failure_patterns[:10],
            "top_missing_terms": [
                {"term": t.term, "count": t.count}
                for t in sorted(
                    self.missing_croatian_terms.values(),
                    key=lambda x: x.count,
                    reverse=True
                )[:20]
            ],
            "suggestions": [s.to_dict() for s in self.suggestions[:30]],
            "analyzed_at": self.analyzed_at,
        }


class FeedbackAnalyzer:
    """
    Analyzes hallucination reports to identify improvement opportunities.

    Key capabilities:
    - Extract Croatian terms from user queries
    - Identify terms not covered by existing dictionaries
    - Generate dictionary addition suggestions
    - Track category-specific failure patterns

    Usage:
        analyzer = FeedbackAnalyzer(db_session)
        result = await analyzer.analyze(min_occurrences=2)
        suggestions = result.suggestions
    """

    # Croatian stopwords to ignore
    STOPWORDS = frozenset([
        "a", "ako", "ali", "bi", "bez", "bilo", "bih", "biti", "da", "do",
        "dok", "ga", "gdje", "i", "ili", "iz", "ja", "je", "jedan", "jedna",
        "jedno", "jer", "još", "kada", "kako", "kao", "koja", "koje", "koji",
        "koju", "li", "me", "mi", "može", "mogu", "na", "ne", "nego", "neka",
        "neki", "nešto", "ni", "nije", "no", "o", "od", "ona", "one", "oni",
        "ono", "pa", "po", "pod", "prema", "pri", "s", "sa", "sam", "samo",
        "se", "smo", "su", "sve", "svi", "ta", "te", "ti", "to", "toga",
        "tu", "u", "uz", "va", "već", "vi", "za", "što", "će", "ću", "tom",
        "mu", "ju", "ih", "njoj", "njemu", "njima", "nje", "njega", "svoj",
        "moj", "tvoj", "naš", "vaš", "njihov", "ovaj", "taj", "onaj",
        # Common bot interaction words
        "molim", "hvala", "hej", "bok", "daj", "pokaži", "prikaži", "reci",
        "trebam", "treba", "želim", "hoću", "mogu", "dohvati", "pronađi",
    ])

    # Word length constraints
    MIN_WORD_LENGTH = 3
    MAX_WORD_LENGTH = 30

    def __init__(
        self,
        db: AsyncSession,
        embedding_engine: Optional[Any] = None
    ):
        """
        Initialize feedback analyzer.

        Args:
            db: Async SQLAlchemy session
            embedding_engine: Optional EmbeddingEngine instance for dictionary access
        """
        self.db = db

        # Load existing dictionaries for comparison
        self._path_entity_croatian: Set[str] = set()
        self._output_key_croatian: Set[str] = set()
        self._synonym_words: Set[str] = set()

        self._load_existing_dictionaries(embedding_engine)

        logger.info(
            f"FeedbackAnalyzer initialized with {len(self._path_entity_croatian)} "
            f"path terms, {len(self._output_key_croatian)} output terms, "
            f"{len(self._synonym_words)} synonyms"
        )

    def _load_existing_dictionaries(self, embedding_engine: Optional[Any] = None):
        """Load Croatian terms from existing dictionaries."""
        try:
            if embedding_engine:
                engine = embedding_engine
            else:
                from services.registry.embedding_engine import EmbeddingEngine
                engine = EmbeddingEngine

            # Extract Croatian terms from PATH_ENTITY_MAP
            # Format: {"vehicle": ("vozilo", "vozila"), ...}
            for eng, (cro_nom, cro_gen) in engine.PATH_ENTITY_MAP.items():
                self._path_entity_croatian.add(cro_nom.lower())
                self._path_entity_croatian.add(cro_gen.lower())
                # Also add stem (first 4+ chars)
                if len(cro_nom) >= 4:
                    self._path_entity_croatian.add(cro_nom[:4].lower())

            # Extract Croatian terms from OUTPUT_KEY_MAP
            # Format: {"mileage": "kilometražu", ...}
            for eng, cro in engine.OUTPUT_KEY_MAP.items():
                self._output_key_croatian.add(cro.lower())
                # Add stem
                if len(cro) >= 4:
                    self._output_key_croatian.add(cro[:4].lower())

            # Extract all synonym words
            # Format: {"vozil": ["auto", "automobil", ...], ...}
            for root, synonyms in engine.CROATIAN_SYNONYMS.items():
                self._synonym_words.add(root.lower())
                for syn in synonyms:
                    self._synonym_words.add(syn.lower())

        except Exception as e:
            logger.warning(f"Could not load existing dictionaries: {e}")

    def _is_covered_term(self, word: str) -> bool:
        """Check if a word is already covered by our dictionaries."""
        word_lower = word.lower()

        # Check exact match
        if word_lower in self._path_entity_croatian:
            return True
        if word_lower in self._output_key_croatian:
            return True
        if word_lower in self._synonym_words:
            return True

        # Check stem match (first 4 chars)
        if len(word_lower) >= 4:
            stem = word_lower[:4]
            if stem in self._path_entity_croatian:
                return True
            if stem in self._output_key_croatian:
                return True
            if stem in self._synonym_words:
                return True

        return False

    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from Croatian text."""
        if not text:
            return []

        # Normalize: lowercase, remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)

        # Split and filter
        words = []
        for word in text.split():
            # Length check
            if len(word) < self.MIN_WORD_LENGTH:
                continue
            if len(word) > self.MAX_WORD_LENGTH:
                continue

            # Stopword check
            if word in self.STOPWORDS:
                continue

            # Must contain at least one letter
            if not re.search(r'[a-zčćžšđ]', word):
                continue

            words.append(word)

        return words

    def _generate_suggestions(
        self,
        missing_terms: Dict[str, TermFrequency],
        min_occurrences: int = 2
    ) -> List[DictionarySuggestion]:
        """Generate dictionary suggestions from missing terms."""
        suggestions = []

        for term, freq in missing_terms.items():
            if freq.count < min_occurrences:
                continue

            # Calculate confidence based on frequency
            confidence = min(0.3 + (freq.count * 0.1), 0.95)

            # Determine likely dictionary type based on term characteristics
            if self._looks_like_entity(term):
                dict_type = "path_entity"
                suggested = f'"{term}": ("{term}", "{term}a")'  # nominative, genitive
                reason = f"Entity term found {freq.count}x in failed queries"
            elif self._looks_like_output_field(term):
                dict_type = "output_key"
                suggested = f'"{term}": "{term}"'
                reason = f"Output field term found {freq.count}x"
            else:
                dict_type = "synonym"
                # Find potential root
                root = term[:4] if len(term) > 4 else term
                suggested = f'"{root}": ["{term}"]'
                reason = f"User synonym found {freq.count}x, add to synonym group"

            suggestions.append(DictionarySuggestion(
                term=term,
                suggested_mapping=suggested,
                dictionary_type=dict_type,
                confidence=confidence,
                frequency=freq.count,
                sample_queries=list(freq.queries)[:3],
                reason=reason
            ))

        # Sort by frequency (highest first)
        suggestions.sort(key=lambda x: x.frequency, reverse=True)

        return suggestions

    def _looks_like_entity(self, term: str) -> bool:
        """Heuristic: does this term look like an entity (noun)?"""
        # Croatian nouns often end in these
        noun_endings = ['a', 'e', 'i', 'o', 'u', 'k', 'n', 't', 'j']
        return len(term) >= 4 and term[-1] in noun_endings

    def _looks_like_output_field(self, term: str) -> bool:
        """Heuristic: does this term look like an output field?"""
        # Output fields are often descriptive
        output_patterns = ['stanje', 'broj', 'datum', 'vrijeme', 'status', 'tip']
        return any(p in term for p in output_patterns)

    async def analyze(
        self,
        min_occurrences: int = 2,
        include_reviewed_only: bool = False,
        tenant_id: Optional[str] = None,
        limit: int = 1000
    ) -> FeedbackAnalysisResult:
        """
        Analyze hallucination reports to find improvement opportunities.

        Args:
            min_occurrences: Minimum term occurrences to generate suggestion
            include_reviewed_only: Only analyze reviewed reports
            tenant_id: Filter by tenant
            limit: Maximum reports to analyze

        Returns:
            FeedbackAnalysisResult with suggestions
        """
        result = FeedbackAnalysisResult(analyzed_at=datetime.now(timezone.utc).isoformat())

        # Build query
        query = select(HallucinationReport).limit(limit)

        if include_reviewed_only:
            query = query.where(HallucinationReport.reviewed.is_(True))

        if tenant_id:
            query = query.where(HallucinationReport.tenant_id == tenant_id)

        # Execute query
        try:
            db_result = await self.db.execute(query)
            reports = db_result.scalars().all()
        except Exception as e:
            logger.error(f"Failed to fetch reports: {e}")
            return result

        result.total_reports_analyzed = len(reports)

        # Track missing terms and patterns
        missing_terms: Dict[str, TermFrequency] = {}
        query_patterns: Counter = Counter()

        for report in reports:
            # Track corrections
            if report.correction:
                result.reports_with_corrections += 1

            # Track categories
            category = report.category or "uncategorized"
            result.category_distribution[category] = \
                result.category_distribution.get(category, 0) + 1

            # Extract words from user query
            words = self._extract_words(report.user_query)

            for word in words:
                # Track common patterns
                query_patterns[word] += 1

                # Check if word is NOT covered by our dictionaries
                if not self._is_covered_term(word):
                    if word in missing_terms:
                        missing_terms[word].add_occurrence(
                            report.user_query,
                            report.category
                        )
                    else:
                        missing_terms[word] = TermFrequency(
                            term=word,
                            count=1,
                            queries=[report.user_query[:100]],
                            categories={report.category} if report.category else set()
                        )

        # Store results
        result.missing_croatian_terms = missing_terms
        result.common_failure_patterns = query_patterns.most_common(20)

        # Generate suggestions
        result.suggestions = self._generate_suggestions(missing_terms, min_occurrences)

        logger.info(
            f"Analysis complete: {result.total_reports_analyzed} reports, "
            f"{len(missing_terms)} missing terms, "
            f"{len(result.suggestions)} suggestions generated"
        )

        return result

    async def get_quick_stats(self) -> Dict[str, Any]:
        """Get quick statistics without full analysis."""
        try:
            # Total reports
            total_result = await self.db.execute(
                select(func.count(HallucinationReport.id))
            )
            total = total_result.scalar() or 0

            # Unreviewed
            unreviewed_result = await self.db.execute(
                select(func.count(HallucinationReport.id)).where(
                    HallucinationReport.reviewed.is_(False)
                )
            )
            unreviewed = unreviewed_result.scalar() or 0

            # With corrections
            corrected_result = await self.db.execute(
                select(func.count(HallucinationReport.id)).where(
                    HallucinationReport.correction.isnot(None)
                )
            )
            corrected = corrected_result.scalar() or 0

            return {
                "total_reports": total,
                "unreviewed": unreviewed,
                "with_corrections": corrected,
                "review_rate": round((total - unreviewed) / total * 100, 1) if total > 0 else 0,
                "correction_rate": round(corrected / total * 100, 1) if total > 0 else 0,
            }
        except Exception as e:
            logger.error(f"Failed to get quick stats: {e}")
            return {"error": str(e)}

    async def export_suggestions(
        self,
        min_occurrences: int = 3,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export suggestions to a JSON file for admin review.

        Args:
            min_occurrences: Minimum term frequency
            output_path: Where to save (default: .cache/feedback_suggestions.json)

        Returns:
            Path to the exported file
        """
        result = await self.analyze(min_occurrences=min_occurrences)

        output_path = output_path or Path.cwd() / ".cache" / "feedback_suggestions.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(result.suggestions)} suggestions to {output_path}")

        return output_path


# Convenience function for one-off analysis
async def analyze_feedback(db: AsyncSession) -> FeedbackAnalysisResult:
    """Run feedback analysis and return results."""
    analyzer = FeedbackAnalyzer(db)
    return await analyzer.analyze()

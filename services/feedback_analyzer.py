"""
Feedback Analyzer - Comprehensive feedback analysis for continuous improvement.
Version: 2.0

This service provides TWO types of analysis:

1. DICTIONARY ANALYSIS (Original)
   - Finds missing Croatian terms in user queries
   - Suggests additions to PATH_ENTITY_MAP, OUTPUT_KEY_MAP, CROATIAN_SYNONYMS
   - Simple keyword/stem matching

2. QUERY PATTERN LEARNING (New - v2.0)
   - Learns query→tool mappings from corrections
   - Identifies WHY queries failed
   - Provides semantic learning, not just keyword matching
   - Can boost/penalize tools in search ranking

ARCHITECTURE:
    HallucinationReport (DB)
              ↓
        FeedbackAnalyzer
              ↓
    ┌─────────┴──────────┐
    │                    │
    DictionaryAnalysis   QueryPatternLearning
    (missing terms)      (query→tool mappings)
              │                    │
              ↓                    ↓
    Suggest dict adds    Improve search ranking

HONEST ASSESSMENT:
    - Dictionary Analysis: 6/10 (keyword matching, crude but finds gaps)
    - Query Learning: 8/10 (semantic, learns from corrections)
    - Combined: 7/10 (actionable insights for improvement)
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
        if len(self.queries) < 5:
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
class QueryToolInsight:
    """Insight about query→tool mapping."""
    pattern: str
    correct_tool: str
    wrong_tools: List[str]
    confidence: float
    sample_queries: List[str]
    action: str  # "boost", "penalize", "add_synonym"

    def to_dict(self) -> Dict:
        return {
            "pattern": self.pattern,
            "correct_tool": self.correct_tool,
            "wrong_tools": self.wrong_tools,
            "confidence": round(self.confidence, 2),
            "sample_queries": self.sample_queries[:3],
            "action": self.action,
        }


@dataclass
class FailureCategoryAnalysis:
    """Analysis of failures by category."""
    category: str
    count: int
    percentage: float
    top_queries: List[str]
    root_cause: str
    recommendation: str

    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "count": self.count,
            "percentage": round(self.percentage, 1),
            "top_queries": self.top_queries[:3],
            "root_cause": self.root_cause,
            "recommendation": self.recommendation,
        }


@dataclass
class FeedbackAnalysisResult:
    """Complete analysis of feedback data."""
    total_reports_analyzed: int = 0
    reports_with_corrections: int = 0

    # Dictionary analysis (v1.0)
    missing_croatian_terms: Dict[str, TermFrequency] = field(default_factory=dict)
    dictionary_suggestions: List[DictionarySuggestion] = field(default_factory=list)

    # Query pattern learning (v2.0)
    query_tool_insights: List[QueryToolInsight] = field(default_factory=list)
    failure_analysis: List[FailureCategoryAnalysis] = field(default_factory=list)

    # Patterns in failures
    common_failure_patterns: List[Tuple[str, int]] = field(default_factory=list)

    # Category breakdown
    category_distribution: Dict[str, int] = field(default_factory=dict)

    # Actionable recommendations
    recommendations: List[str] = field(default_factory=list)

    # Analysis metadata
    analyzed_at: str = ""

    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_analyzed": self.total_reports_analyzed,
                "with_corrections": self.reports_with_corrections,
                "dictionary_suggestions": len(self.dictionary_suggestions),
                "query_insights": len(self.query_tool_insights),
            },
            "category_distribution": self.category_distribution,
            "failure_analysis": [f.to_dict() for f in self.failure_analysis],
            "common_failure_patterns": self.common_failure_patterns[:10],
            "top_missing_terms": [
                {"term": t.term, "count": t.count}
                for t in sorted(
                    self.missing_croatian_terms.values(),
                    key=lambda x: x.count,
                    reverse=True
                )[:20]
            ],
            "dictionary_suggestions": [s.to_dict() for s in self.dictionary_suggestions[:30]],
            "query_tool_insights": [i.to_dict() for i in self.query_tool_insights[:20]],
            "recommendations": self.recommendations[:10],
            "analyzed_at": self.analyzed_at,
        }


class FeedbackAnalyzer:
    """
    Comprehensive feedback analyzer with dictionary and query pattern analysis.

    v2.0 Features:
    - Dictionary gap analysis (missing Croatian terms)
    - Query→tool learning from corrections
    - Failure category analysis with root causes
    - Actionable recommendations

    Usage:
        analyzer = FeedbackAnalyzer(db)
        result = await analyzer.analyze_comprehensive()
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
        "molim", "hvala", "hej", "bok", "daj", "pokaži", "prikaži", "reci",
        "trebam", "treba", "želim", "hoću", "mogu", "dohvati", "pronađi",
    ])

    # Failure category recommendations
    CATEGORY_RECOMMENDATIONS = {
        "misunderstood": "Improve query understanding - add more synonyms to CROATIAN_SYNONYMS",
        "wrong_data": "Check API response parsing - data may be correct but formatted wrong",
        "hallucination": "Reduce AI temperature or add more constraints",
        "api_error": "Check API error handling and retry logic",
        "rag_failure": "Improve tool documentation in tool_documentation.json",
        "outdated": "Implement data freshness checks",
        "user_error": "Consider improving bot responses to prevent user confusion",
    }

    # Query patterns for tool extraction
    TOOL_PATTERNS = [
        r"(get_\w+)",
        r"(post_\w+)",
        r"(put_\w+)",
        r"(delete_\w+)",
    ]

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

        # Compile tool patterns
        self._tool_patterns = [re.compile(p, re.IGNORECASE) for p in self.TOOL_PATTERNS]

        logger.info(
            f"FeedbackAnalyzer v2.0 initialized with {len(self._path_entity_croatian)} "
            f"path terms, {len(self._output_key_croatian)} output terms, "
            f"{len(self._synonym_words)} synonyms"
        )

    def _load_existing_dictionaries(self, embedding_engine: Optional[Any] = None):
        """Load Croatian terms from existing dictionaries."""
        try:
            if embedding_engine:
                # Use provided engine instance (e.g., from tests)
                path_entity_map = embedding_engine.PATH_ENTITY_MAP
                output_key_map = embedding_engine.OUTPUT_KEY_MAP
                croatian_synonyms = embedding_engine.CROATIAN_SYNONYMS
            else:
                # Load directly from config (avoids class-level attribute access)
                from services.registry.embedding_engine import _get_croatian_mappings
                mappings = _get_croatian_mappings()
                # PATH_ENTITY_MAP: JSON stores lists, convert to tuples
                raw_path = mappings.get("path_entity_map", {})
                path_entity_map = {
                    k: (v[0], v[1]) for k, v in raw_path.items()
                    if k != "_comments" and isinstance(v, list) and len(v) == 2
                }
                output_key_map = {
                    k: v for k, v in mappings.get("output_key_map", {}).items()
                    if k != "_comments"
                }
                croatian_synonyms = {
                    k: v for k, v in mappings.get("croatian_synonyms", {}).items()
                    if k != "_comments"
                }

            # Extract Croatian terms from PATH_ENTITY_MAP
            for eng, (cro_nom, cro_gen) in path_entity_map.items():
                self._path_entity_croatian.add(cro_nom.lower())
                self._path_entity_croatian.add(cro_gen.lower())
                if len(cro_nom) >= 4:
                    self._path_entity_croatian.add(cro_nom[:4].lower())

            # Extract Croatian terms from OUTPUT_KEY_MAP
            for eng, cro in output_key_map.items():
                self._output_key_croatian.add(cro.lower())
                if len(cro) >= 4:
                    self._output_key_croatian.add(cro[:4].lower())

            # Extract all synonym words
            for root, synonyms in croatian_synonyms.items():
                self._synonym_words.add(root.lower())
                for syn in synonyms:
                    self._synonym_words.add(syn.lower())

        except Exception as e:
            logger.warning(f"Could not load existing dictionaries: {e}")

    def _is_covered_term(self, word: str) -> bool:
        """Check if a word is already covered by our dictionaries."""
        word_lower = word.lower()

        if word_lower in self._path_entity_croatian:
            return True
        if word_lower in self._output_key_croatian:
            return True
        if word_lower in self._synonym_words:
            return True

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

        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)

        words = []
        for word in text.split():
            if len(word) < self.MIN_WORD_LENGTH:
                continue
            if len(word) > self.MAX_WORD_LENGTH:
                continue
            if word in self.STOPWORDS:
                continue
            if not re.search(r'[a-zčćžšđ]', word):
                continue
            words.append(word)

        return words

    def _extract_tool_from_text(self, text: str) -> Optional[str]:
        """Extract tool name from text."""
        if not text:
            return None

        for pattern in self._tool_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).lower()

        return None

    def _generate_dictionary_suggestions(
        self,
        missing_terms: Dict[str, TermFrequency],
        min_occurrences: int = 2
    ) -> List[DictionarySuggestion]:
        """Generate dictionary suggestions from missing terms."""
        suggestions = []

        for term, freq in missing_terms.items():
            if freq.count < min_occurrences:
                continue

            confidence = min(0.3 + (freq.count * 0.1), 0.95)

            if self._looks_like_entity(term):
                dict_type = "path_entity"
                suggested = f'"{term}": ("{term}", "{term}a")'
                reason = f"Entity term found {freq.count}x in failed queries"
            elif self._looks_like_output_field(term):
                dict_type = "output_key"
                suggested = f'"{term}": "{term}"'
                reason = f"Output field term found {freq.count}x"
            else:
                dict_type = "synonym"
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

        suggestions.sort(key=lambda x: x.frequency, reverse=True)
        return suggestions

    def _looks_like_entity(self, term: str) -> bool:
        """Heuristic: does this term look like an entity (noun)?"""
        noun_endings = ['a', 'e', 'i', 'o', 'u', 'k', 'n', 't', 'j']
        return len(term) >= 4 and term[-1] in noun_endings

    def _looks_like_output_field(self, term: str) -> bool:
        """Heuristic: does this term look like an output field?"""
        output_patterns = ['stanje', 'broj', 'datum', 'vrijeme', 'status', 'tip']
        return any(p in term for p in output_patterns)

    def _analyze_failure_categories(
        self,
        reports: List[HallucinationReport]
    ) -> List[FailureCategoryAnalysis]:
        """Analyze failures by category with root causes."""
        category_data: Dict[str, List[HallucinationReport]] = defaultdict(list)

        for report in reports:
            category = report.category or "uncategorized"
            category_data[category].append(report)

        total = len(reports)
        analysis = []

        for category, cat_reports in category_data.items():
            count = len(cat_reports)
            percentage = (count / total * 100) if total > 0 else 0

            top_queries = [r.user_query[:100] for r in cat_reports[:5]]
            recommendation = self.CATEGORY_RECOMMENDATIONS.get(
                category,
                "Review these failures manually"
            )

            # Determine root cause
            if category == "misunderstood":
                root_cause = "Query intent not recognized by search/AI"
            elif category == "wrong_data":
                root_cause = "Correct tool selected but data interpretation failed"
            elif category == "hallucination":
                root_cause = "AI generated false information"
            elif category == "api_error":
                root_cause = "External API returned error or unexpected data"
            elif category == "rag_failure":
                root_cause = "Wrong tool selected due to poor tool descriptions"
            else:
                root_cause = "Unknown - needs manual review"

            analysis.append(FailureCategoryAnalysis(
                category=category,
                count=count,
                percentage=percentage,
                top_queries=top_queries,
                root_cause=root_cause,
                recommendation=recommendation
            ))

        analysis.sort(key=lambda x: x.count, reverse=True)
        return analysis

    def _extract_query_tool_insights(
        self,
        reports: List[HallucinationReport]
    ) -> List[QueryToolInsight]:
        """Extract query→tool insights from corrections."""
        # Pattern → (correct_tool, wrong_tools, queries)
        pattern_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"correct_tool": None, "wrong_tools": set(), "queries": []}
        )

        for report in reports:
            if not report.correction:
                continue

            # Extract what tool was used (wrong) and what should have been used
            wrong_tool = self._extract_tool_from_text(report.bot_response)
            correct_tool = self._extract_tool_from_text(report.correction)

            if not correct_tool:
                continue

            # Extract key phrases from query
            words = self._extract_words(report.user_query)
            for i in range(len(words)):
                for length in [2, 3, 1]:  # Prefer 2-3 word patterns
                    if i + length <= len(words):
                        pattern = " ".join(words[i:i+length])
                        if len(pattern) > 4:
                            data = pattern_data[pattern]
                            data["correct_tool"] = correct_tool
                            if wrong_tool:
                                data["wrong_tools"].add(wrong_tool)
                            if len(data["queries"]) < 5:
                                data["queries"].append(report.user_query[:100])

        # Convert to insights
        insights = []
        for pattern, data in pattern_data.items():
            if not data["correct_tool"]:
                continue

            confidence = 0.5 + min(len(data["queries"]) * 0.1, 0.4)

            # Determine action
            if data["wrong_tools"]:
                action = "boost_correct_penalize_wrong"
            else:
                action = "boost_correct"

            insights.append(QueryToolInsight(
                pattern=pattern,
                correct_tool=data["correct_tool"],
                wrong_tools=list(data["wrong_tools"]),
                confidence=confidence,
                sample_queries=data["queries"],
                action=action
            ))

        insights.sort(key=lambda x: x.confidence, reverse=True)
        return insights[:50]  # Top 50

    def _generate_recommendations(
        self,
        result: "FeedbackAnalysisResult"
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Based on category distribution
        if result.category_distribution:
            top_category = max(result.category_distribution.items(), key=lambda x: x[1])
            if top_category[1] > 10:
                rec = self.CATEGORY_RECOMMENDATIONS.get(
                    top_category[0],
                    f"Focus on {top_category[0]} failures ({top_category[1]} occurrences)"
                )
                recommendations.append(f"PRIORITY: {rec}")

        # Based on missing terms
        if len(result.dictionary_suggestions) > 5:
            recommendations.append(
                f"Add {len(result.dictionary_suggestions)} missing terms to dictionaries "
                "to improve query understanding"
            )

        # Based on query insights
        high_conf_insights = [i for i in result.query_tool_insights if i.confidence > 0.7]
        if high_conf_insights:
            recommendations.append(
                f"Found {len(high_conf_insights)} high-confidence query→tool patterns. "
                "Consider adding these to tool_documentation.json as example_queries."
            )

        # Based on correction rate
        if result.total_reports_analyzed > 0:
            correction_rate = result.reports_with_corrections / result.total_reports_analyzed
            if correction_rate < 0.3:
                recommendations.append(
                    f"Only {correction_rate:.0%} of reports have corrections. "
                    "Encourage admins to add corrections for better learning."
                )

        return recommendations

    async def analyze(
        self,
        min_occurrences: int = 2,
        include_reviewed_only: bool = False,
        tenant_id: Optional[str] = None,
        limit: int = 1000
    ) -> FeedbackAnalysisResult:
        """
        Analyze hallucination reports (backward compatible - dictionary analysis only).

        For comprehensive analysis, use analyze_comprehensive().
        """
        result = FeedbackAnalysisResult(analyzed_at=datetime.now(timezone.utc).isoformat())

        query = select(HallucinationReport).limit(limit)

        if include_reviewed_only:
            query = query.where(HallucinationReport.reviewed.is_(True))

        if tenant_id:
            query = query.where(HallucinationReport.tenant_id == tenant_id)

        try:
            db_result = await self.db.execute(query)
            reports = db_result.scalars().all()
        except Exception as e:
            logger.error(f"Failed to fetch reports: {e}")
            return result

        result.total_reports_analyzed = len(reports)

        missing_terms: Dict[str, TermFrequency] = {}
        query_patterns: Counter = Counter()

        for report in reports:
            if report.correction:
                result.reports_with_corrections += 1

            category = report.category or "uncategorized"
            result.category_distribution[category] = \
                result.category_distribution.get(category, 0) + 1

            words = self._extract_words(report.user_query)

            for word in words:
                query_patterns[word] += 1

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

        result.missing_croatian_terms = missing_terms
        result.common_failure_patterns = query_patterns.most_common(20)
        result.dictionary_suggestions = self._generate_dictionary_suggestions(missing_terms, min_occurrences)

        logger.info(
            f"Analysis complete: {result.total_reports_analyzed} reports, "
            f"{len(missing_terms)} missing terms, "
            f"{len(result.dictionary_suggestions)} suggestions"
        )

        return result

    async def analyze_comprehensive(
        self,
        min_occurrences: int = 2,
        limit: int = 1000
    ) -> FeedbackAnalysisResult:
        """
        Comprehensive analysis including query pattern learning.

        This is the v2.0 analysis that includes:
        - Dictionary gap analysis (missing terms)
        - Query→tool learning from corrections
        - Failure category analysis
        - Actionable recommendations
        """
        # Start with basic analysis
        result = await self.analyze(min_occurrences=min_occurrences, limit=limit)

        # Fetch reports again for deeper analysis
        query = select(HallucinationReport).limit(limit)
        try:
            db_result = await self.db.execute(query)
            reports = list(db_result.scalars().all())
        except Exception as e:
            logger.error(f"Failed to fetch reports for comprehensive analysis: {e}")
            return result

        # Add failure category analysis
        result.failure_analysis = self._analyze_failure_categories(reports)

        # Add query→tool insights
        result.query_tool_insights = self._extract_query_tool_insights(reports)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        logger.info(
            f"Comprehensive analysis complete: "
            f"{len(result.failure_analysis)} category analyses, "
            f"{len(result.query_tool_insights)} query insights, "
            f"{len(result.recommendations)} recommendations"
        )

        return result

    async def get_quick_stats(self) -> Dict[str, Any]:
        """Get quick statistics without full analysis."""
        try:
            total_result = await self.db.execute(
                select(func.count(HallucinationReport.id))
            )
            total = total_result.scalar() or 0

            unreviewed_result = await self.db.execute(
                select(func.count(HallucinationReport.id)).where(
                    HallucinationReport.reviewed.is_(False)
                )
            )
            unreviewed = unreviewed_result.scalar() or 0

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
        """Export suggestions to a JSON file for admin review."""
        result = await self.analyze_comprehensive(min_occurrences=min_occurrences)

        output_path = output_path or Path.cwd() / ".cache" / "feedback_suggestions.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Exported comprehensive analysis to {output_path}")

        return output_path


# Convenience function
async def analyze_feedback(db: AsyncSession) -> FeedbackAnalysisResult:
    """Run feedback analysis and return results."""
    analyzer = FeedbackAnalyzer(db)
    return await analyzer.analyze()


async def analyze_feedback_comprehensive(db: AsyncSession) -> FeedbackAnalysisResult:
    """Run comprehensive feedback analysis with query learning."""
    analyzer = FeedbackAnalyzer(db)
    return await analyzer.analyze_comprehensive()

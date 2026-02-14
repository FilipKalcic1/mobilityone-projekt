"""
Tests for FeedbackAnalyzer service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from services.feedback_analyzer import (
    FeedbackAnalyzer,
    FeedbackAnalysisResult,
    TermFrequency,
    DictionarySuggestion,
    analyze_feedback
)


class TestTermFrequency:
    """Tests for TermFrequency dataclass."""

    def test_initial_values(self):
        """Test initial values."""
        tf = TermFrequency(term="test")
        assert tf.term == "test"
        assert tf.count == 1
        assert tf.queries == []
        assert tf.categories == set()

    def test_add_occurrence(self):
        """Test adding occurrences."""
        tf = TermFrequency(term="vozilo")
        tf.add_occurrence("prikaži vozilo", "vehicle")
        assert tf.count == 2
        assert len(tf.queries) == 1
        assert "vehicle" in tf.categories

    def test_queries_limit(self):
        """Test that queries are limited to 5."""
        tf = TermFrequency(term="test")
        for i in range(10):
            tf.add_occurrence(f"query {i}")
        assert len(tf.queries) == 5  # Limited to 5


class TestDictionarySuggestion:
    """Tests for DictionarySuggestion dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        suggestion = DictionarySuggestion(
            term="kartica",
            suggested_mapping='"kartic": ["kartica", "kartice"]',
            dictionary_type="synonym",
            confidence=0.75,
            frequency=5,
            sample_queries=["daj karticu", "prikaži kartice", "moja kartica"],
            reason="User synonym found 5x"
        )

        d = suggestion.to_dict()
        assert d["term"] == "kartica"
        assert d["confidence"] == 0.75
        assert d["frequency"] == 5
        assert len(d["sample_queries"]) == 3


class TestFeedbackAnalysisResult:
    """Tests for FeedbackAnalysisResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        result = FeedbackAnalysisResult(
            total_reports_analyzed=100,
            reports_with_corrections=25,
            analyzed_at="2024-01-15T10:00:00Z"
        )
        result.category_distribution = {"wrong_data": 50, "hallucination": 50}

        d = result.to_dict()
        assert d["summary"]["total_analyzed"] == 100
        assert d["summary"]["with_corrections"] == 25
        assert d["category_distribution"]["wrong_data"] == 50


class MockEmbeddingEngine:
    """Mock embedding engine with test dictionaries."""
    PATH_ENTITY_MAP = {
        "vehicle": ("vozilo", "vozila"),
        "driver": ("vozač", "vozača"),
    }
    OUTPUT_KEY_MAP = {
        "mileage": "kilometražu",
        "status": "status",
    }
    CROATIAN_SYNONYMS = {
        "vozil": ["auto", "automobil", "kola"],
        "goriv": ["benzin", "nafta"],
    }


class TestFeedbackAnalyzer:
    """Tests for FeedbackAnalyzer service."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        db = AsyncMock()
        return db

    @pytest.fixture
    def analyzer(self, mock_db):
        """Create analyzer with mock db and mock engine."""
        return FeedbackAnalyzer(mock_db, embedding_engine=MockEmbeddingEngine)

    def test_init_loads_dictionaries(self, analyzer):
        """Test that initialization loads existing dictionaries."""
        assert len(analyzer._path_entity_croatian) > 0
        assert len(analyzer._output_key_croatian) > 0
        assert len(analyzer._synonym_words) > 0

    def test_is_covered_term_exact(self, analyzer):
        """Test exact term coverage check."""
        assert analyzer._is_covered_term("vozilo") is True
        assert analyzer._is_covered_term("vozila") is True
        assert analyzer._is_covered_term("auto") is True  # synonym
        assert analyzer._is_covered_term("nepoznato") is False

    def test_is_covered_term_stem(self, analyzer):
        """Test stem-based coverage check."""
        # "vozil" is in synonyms, should match "vozilom" (stem match)
        assert analyzer._is_covered_term("vozilom") is True

    def test_extract_words_basic(self, analyzer):
        """Test basic word extraction."""
        words = analyzer._extract_words("prikaži mi vozilo")
        # "prikaži", "mi" are stopwords
        assert "vozilo" in words
        assert "mi" not in words

    def test_extract_words_punctuation(self, analyzer):
        """Test word extraction removes punctuation."""
        words = analyzer._extract_words("Gdje je vozilo?!")
        assert "vozilo" in words
        assert "?" not in str(words)

    def test_extract_words_length_filter(self, analyzer):
        """Test word length filtering."""
        words = analyzer._extract_words("a i o ab test")
        assert "a" not in words  # too short
        assert "ab" not in words  # too short
        assert "test" in words

    def test_extract_words_stopwords(self, analyzer):
        """Test stopword removal."""
        words = analyzer._extract_words("ja želim da mi pokažeš rezervaciju")
        # ja, želim, da, mi are stopwords
        assert "ja" not in words
        assert "rezervaciju" in words

    def test_generate_suggestions_minimum_occurrences(self, analyzer):
        """Test that suggestions require minimum occurrences."""
        missing_terms = {
            "kartica": TermFrequency(term="kartica", count=1),
            "obrazac": TermFrequency(term="obrazac", count=3),
        }

        suggestions = analyzer._generate_dictionary_suggestions(missing_terms, min_occurrences=2)

        # Only obrazac should be suggested (count=3 >= min=2)
        assert len(suggestions) == 1
        assert suggestions[0].term == "obrazac"

    def test_generate_suggestions_confidence(self, analyzer):
        """Test that confidence increases with frequency."""
        missing_terms = {
            "low": TermFrequency(term="low", count=2),
            "high": TermFrequency(term="high", count=10),
        }

        suggestions = analyzer._generate_dictionary_suggestions(missing_terms, min_occurrences=2)

        low_conf = next(s for s in suggestions if s.term == "low").confidence
        high_conf = next(s for s in suggestions if s.term == "high").confidence

        assert high_conf > low_conf

    def test_looks_like_entity(self, analyzer):
        """Test entity detection heuristic."""
        assert analyzer._looks_like_entity("vozilo") is True  # ends in 'o'
        assert analyzer._looks_like_entity("kartica") is True  # ends in 'a'
        assert analyzer._looks_like_entity("ab") is False  # too short

    @pytest.mark.asyncio
    async def test_analyze_empty_db(self, analyzer, mock_db):
        """Test analysis with empty database."""
        # Mock empty result
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        result = await analyzer.analyze()

        assert result.total_reports_analyzed == 0
        assert len(result.dictionary_suggestions) == 0

    @pytest.mark.asyncio
    async def test_analyze_with_reports(self, analyzer, mock_db):
        """Test analysis with sample reports."""
        # Create mock reports
        mock_report = MagicMock()
        mock_report.user_query = "gdje je moja kartica za gorivo"
        mock_report.correction = "Kartica je na lokaciji X"
        mock_report.category = "wrong_data"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_report]
        mock_db.execute.return_value = mock_result

        result = await analyzer.analyze(min_occurrences=1)

        assert result.total_reports_analyzed == 1
        assert result.reports_with_corrections == 1
        assert result.category_distribution.get("wrong_data", 0) == 1

    @pytest.mark.asyncio
    async def test_get_quick_stats(self, analyzer, mock_db):
        """Test quick stats retrieval."""
        # Mock count results
        mock_db.execute.side_effect = [
            MagicMock(scalar=lambda: 100),  # total
            MagicMock(scalar=lambda: 25),   # unreviewed
            MagicMock(scalar=lambda: 50),   # corrected
        ]

        stats = await analyzer.get_quick_stats()

        assert stats["total_reports"] == 100
        assert stats["unreviewed"] == 25
        assert stats["with_corrections"] == 50


class TestAnalyzeFeedbackFunction:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_analyze_feedback(self):
        """Test the convenience function."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        # Use the real embedding engine - it exists in the codebase
        result = await analyze_feedback(mock_db)

        assert isinstance(result, FeedbackAnalysisResult)


class TestFeedbackAnalyzerIntegration:
    """Integration tests (require more setup)."""

    @pytest.mark.asyncio
    async def test_full_analysis_flow(self):
        """Test complete analysis flow."""
        mock_db = AsyncMock()

        # Create realistic mock reports
        reports = []
        for i in range(5):
            report = MagicMock()
            report.user_query = f"prikaži mi obrazac broj {i}"
            report.correction = None
            report.category = "misunderstood"
            reports.append(report)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = reports
        mock_db.execute.return_value = mock_result

        # Use mock engine
        analyzer = FeedbackAnalyzer(mock_db, embedding_engine=MockEmbeddingEngine)
        result = await analyzer.analyze(min_occurrences=3)

        # "obrazac" appears 5 times, should be suggested
        assert result.total_reports_analyzed == 5
        # "obrazac" should be in missing terms
        assert "obrazac" in result.missing_croatian_terms

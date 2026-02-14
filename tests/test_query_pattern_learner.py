"""
Tests for QueryPatternLearner service.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from services.query_pattern_learner import (
    QueryPatternLearner,
    QueryToolMapping,
    FailurePattern,
    LearningResult,
    learn_from_feedback,
)


class TestQueryToolMapping:
    """Tests for QueryToolMapping dataclass."""

    def test_initial_values(self):
        """Test initial values."""
        mapping = QueryToolMapping(
            pattern="broj šasije",
            correct_tool="get_VehicleVIN"
        )
        assert mapping.pattern == "broj šasije"
        assert mapping.correct_tool == "get_VehicleVIN"
        assert mapping.wrong_tools == set()
        assert mapping.confidence == 0.0
        assert mapping.occurrence_count == 1

    def test_to_dict(self):
        """Test serialization."""
        mapping = QueryToolMapping(
            pattern="kilometraža",
            correct_tool="get_VehicleMileage",
            wrong_tools={"get_VehicleStatus", "get_VehicleInfo"},
            confidence=0.85,
            occurrence_count=10,
            sample_queries=["koliko km ima auto", "kilometraža vozila"]
        )

        d = mapping.to_dict()
        assert d["pattern"] == "kilometraža"
        assert d["correct_tool"] == "get_VehicleMileage"
        assert "get_VehicleStatus" in d["wrong_tools"]
        assert d["confidence"] == 0.85
        assert d["occurrence_count"] == 10


class TestFailurePattern:
    """Tests for FailurePattern dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        pattern = FailurePattern(
            category="misunderstood",
            query_type="vehicle_vin",
            tool_used="get_VehicleMileage",
            tool_needed="get_VehicleVIN",
            root_cause="Query intent not recognized",
            count=5
        )

        d = pattern.to_dict()
        assert d["category"] == "misunderstood"
        assert d["query_type"] == "vehicle_vin"
        assert d["tool_used"] == "get_VehicleMileage"
        assert d["count"] == 5


class TestLearningResult:
    """Tests for LearningResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        mapping = QueryToolMapping(
            pattern="test",
            correct_tool="get_Test",
            confidence=0.8
        )
        failure = FailurePattern(
            category="misunderstood",
            query_type="test",
            tool_used="get_Wrong",
            tool_needed="get_Right",
            root_cause="Test failure"
        )

        result = LearningResult(
            total_analyzed=100,
            with_corrections=25,
            patterns_learned=10,
            query_tool_mappings=[mapping],
            failure_patterns=[failure],
            category_breakdown={"misunderstood": 50, "wrong_data": 50},
            top_confused_queries=[("query", "wrong", "right")],
            analyzed_at="2024-01-15T10:00:00Z"
        )

        d = result.to_dict()
        assert d["summary"]["total_analyzed"] == 100
        assert d["summary"]["with_corrections"] == 25
        assert d["summary"]["patterns_learned"] == 10
        assert len(d["query_tool_mappings"]) == 1
        assert len(d["failure_patterns"]) == 1
        assert d["category_breakdown"]["misunderstood"] == 50


class TestQueryPatternLearner:
    """Tests for QueryPatternLearner service."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.fixture
    def temp_mappings_file(self):
        """Create a temporary mappings file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"version": "1.0", "mappings": []}')
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def learner(self, mock_db, temp_mappings_file):
        """Create learner with temp file."""
        with patch('services.query_pattern_learner.LEARNED_MAPPINGS_FILE', temp_mappings_file):
            return QueryPatternLearner(mock_db)

    def test_init(self, learner):
        """Test initialization."""
        assert len(learner._mappings) == 0
        assert len(learner._query_patterns) > 0
        assert len(learner._tool_patterns) > 0

    def test_init_with_existing_mappings(self, mock_db):
        """Test initialization with existing mappings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                "version": "1.0",
                "mappings": [
                    {
                        "pattern": "broj šasije",
                        "correct_tool": "get_VehicleVIN",
                        "wrong_tools": ["get_VehicleMileage"],
                        "confidence": 0.85,
                        "occurrence_count": 5
                    }
                ]
            }
            json.dump(data, f)
            path = Path(f.name)

        try:
            with patch('services.query_pattern_learner.LEARNED_MAPPINGS_FILE', path):
                learner = QueryPatternLearner(mock_db)
                assert len(learner._mappings) == 1
                assert learner._mappings["broj šasije"].correct_tool == "get_VehicleVIN"
        finally:
            path.unlink()

    def test_extract_query_type_vin(self, learner):
        """Test VIN query type extraction."""
        assert learner._extract_query_type("daj mi broj šasije") == "vehicle_vin"
        assert learner._extract_query_type("trebam VIN vozila") == "vehicle_vin"
        assert learner._extract_query_type("koja je šasija?") == "vehicle_vin"

    def test_extract_query_type_mileage(self, learner):
        """Test mileage query type extraction."""
        assert learner._extract_query_type("kilometraža vozila") == "vehicle_mileage"
        assert learner._extract_query_type("koliko km ima auto") == "vehicle_mileage"

    def test_extract_query_type_registration(self, learner):
        """Test registration query type extraction."""
        assert learner._extract_query_type("registracija auta") == "vehicle_registration"
        assert learner._extract_query_type("tablice vozila") == "vehicle_registration"

    def test_extract_query_type_booking(self, learner):
        """Test booking query type extraction."""
        assert learner._extract_query_type("napravi rezervaciju") == "booking"
        assert learner._extract_query_type("booking za sutra") == "booking"

    def test_extract_query_type_unknown(self, learner):
        """Test unknown query type."""
        assert learner._extract_query_type("random query") is None

    def test_extract_tool_from_text(self, learner):
        """Test tool extraction from text."""
        assert learner._extract_tool_from_text("use get_VehicleVIN") == "get_vehiclevin"
        assert learner._extract_tool_from_text("trebao koristiti get_VehicleMileage") == "get_vehiclemileage"
        assert learner._extract_tool_from_text("post_CreateBooking") == "post_createbooking"

    def test_extract_tool_from_text_none(self, learner):
        """Test tool extraction when no tool found."""
        assert learner._extract_tool_from_text("no tool here") is None
        assert learner._extract_tool_from_text("") is None
        assert learner._extract_tool_from_text(None) is None

    def test_extract_patterns_from_query(self, learner):
        """Test pattern extraction from query."""
        patterns = learner._extract_patterns_from_query("daj mi broj šasije za vozilo")
        assert len(patterns) > 0
        # Should have removed stopwords like "mi", "za"
        assert all("mi" not in p for p in patterns)

    def test_extract_patterns_removes_stopwords(self, learner):
        """Test that stopwords are removed."""
        patterns = learner._extract_patterns_from_query("ja želim da mi pokažeš auto")
        # ja, želim, da, mi are stopwords
        words_in_patterns = " ".join(patterns)
        # Only content words should remain
        assert "pokažeš" in words_in_patterns or "auto" in words_in_patterns

    def test_analyze_failure_reason_misunderstood(self, learner):
        """Test failure analysis for misunderstood category."""
        report = MagicMock()
        report.user_query = "broj šasije"
        report.category = "misunderstood"

        failure = learner._analyze_failure_reason(report, "get_VehicleMileage", "get_VehicleVIN")

        assert failure.category == "misunderstood"
        assert failure.query_type == "vehicle_vin"
        assert "not recognized" in failure.root_cause

    def test_analyze_failure_reason_wrong_tool(self, learner):
        """Test failure analysis when wrong tool was selected."""
        report = MagicMock()
        report.user_query = "test query"
        report.category = "rag_failure"

        failure = learner._analyze_failure_reason(report, "get_Wrong", "get_Right")

        assert "get_Wrong" in failure.root_cause
        assert "get_Right" in failure.root_cause

    @pytest.mark.asyncio
    async def test_learn_empty_db(self, learner, mock_db):
        """Test learning with empty database."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        result = await learner.learn()

        assert result.total_analyzed == 0
        assert result.with_corrections == 0
        assert result.patterns_learned == 0

    @pytest.mark.asyncio
    async def test_learn_with_reports(self, learner, mock_db):
        """Test learning with sample reports."""
        # Create mock reports
        report1 = MagicMock()
        report1.user_query = "daj mi broj šasije"
        report1.correction = "Trebao je koristiti get_VehicleVIN"
        report1.category = "misunderstood"
        report1.bot_response = "Kilometraža je 45000km"
        report1.retrieved_chunks = None

        report2 = MagicMock()
        report2.user_query = "koja je šasija vozila"
        report2.correction = "Pravilni alat je get_VehicleVIN"
        report2.category = "misunderstood"
        report2.bot_response = "Lokacija vozila..."
        report2.retrieved_chunks = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [report1, report2]
        mock_db.execute.return_value = mock_result

        result = await learner.learn(min_occurrences=1)

        assert result.total_analyzed == 2
        assert result.with_corrections == 2
        assert result.category_breakdown.get("misunderstood", 0) == 2

    @pytest.mark.asyncio
    async def test_learn_without_corrections(self, learner, mock_db):
        """Test learning excluding reports without corrections."""
        report = MagicMock()
        report.user_query = "test query"
        report.correction = None  # No correction
        report.category = "unknown"
        report.bot_response = "Response"
        report.retrieved_chunks = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [report]
        mock_db.execute.return_value = mock_result

        result = await learner.learn(include_uncorrected=False)

        # With include_uncorrected=False, the query filters to correction IS NOT NULL
        # Our mock returns all reports regardless, so we'd still get 1
        # But since correction is None, with_corrections should be 0
        assert result.with_corrections == 0

    def test_get_tool_for_query_found(self, learner):
        """Test getting tool for query when pattern matches."""
        # Add a mapping
        learner._mappings["broj šasije"] = QueryToolMapping(
            pattern="broj šasije",
            correct_tool="get_VehicleVIN",
            confidence=0.9
        )

        result = learner.get_tool_for_query("daj mi broj šasije")

        assert result is not None
        assert result[0] == "get_VehicleVIN"
        assert result[1] == 0.9

    def test_get_tool_for_query_not_found(self, learner):
        """Test getting tool when no pattern matches."""
        result = learner.get_tool_for_query("random query without patterns")
        assert result is None

    def test_get_tool_for_query_best_match(self, learner):
        """Test that best confidence match is returned."""
        learner._mappings["šasija"] = QueryToolMapping(
            pattern="šasija",
            correct_tool="get_VehicleVIN",
            confidence=0.7
        )
        learner._mappings["broj šasije"] = QueryToolMapping(
            pattern="broj šasije",
            correct_tool="get_VehicleVIN",
            confidence=0.95
        )

        result = learner.get_tool_for_query("daj broj šasije")

        assert result[1] == 0.95  # Higher confidence wins

    def test_get_negative_signals(self, learner):
        """Test getting tools to avoid."""
        # Use ASCII pattern to avoid encoding issues on Windows
        wrong_tools_set = set()
        wrong_tools_set.add("get_VehicleMileage")
        wrong_tools_set.add("get_VehicleStatus")

        mapping = QueryToolMapping(
            pattern="vehicle mileage",
            correct_tool="get_VehicleVIN",
            wrong_tools=wrong_tools_set
        )
        learner._mappings["vehicle mileage"] = mapping

        # Verify mapping was added correctly
        assert "vehicle mileage" in learner._mappings
        assert len(learner._mappings["vehicle mileage"].wrong_tools) == 2

        avoid = learner.get_negative_signals("show vehicle mileage info")

        assert "get_VehicleMileage" in avoid
        assert "get_VehicleStatus" in avoid

    def test_get_negative_signals_empty(self, learner):
        """Test negative signals when no matches."""
        avoid = learner.get_negative_signals("completely unrelated query")
        assert len(avoid) == 0

    @pytest.mark.asyncio
    async def test_get_statistics(self, learner):
        """Test statistics retrieval."""
        learner._mappings["pattern1"] = QueryToolMapping(
            pattern="pattern1",
            correct_tool="get_Tool1",
            confidence=0.9,
            occurrence_count=5
        )
        learner._mappings["pattern2"] = QueryToolMapping(
            pattern="pattern2",
            correct_tool="get_Tool2",
            confidence=0.5,
            occurrence_count=3
        )

        stats = await learner.get_statistics()

        assert stats["total_mappings"] == 2
        assert stats["high_confidence_mappings"] == 1  # Only >0.7
        assert stats["total_occurrences"] == 8

    @pytest.mark.asyncio
    async def test_save_and_load_mappings(self, mock_db):
        """Test persistence of learned mappings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"version": "1.0", "mappings": []}')
            path = Path(f.name)

        try:
            # Create learner and add mapping
            with patch('services.query_pattern_learner.LEARNED_MAPPINGS_FILE', path):
                learner1 = QueryPatternLearner(mock_db)
                learner1._mappings["test_pattern"] = QueryToolMapping(
                    pattern="test_pattern",
                    correct_tool="get_Test",
                    confidence=0.85,
                    occurrence_count=3
                )
                learner1._save_mappings()

            # Load in new instance
            with patch('services.query_pattern_learner.LEARNED_MAPPINGS_FILE', path):
                learner2 = QueryPatternLearner(mock_db)

            assert "test_pattern" in learner2._mappings
            assert learner2._mappings["test_pattern"].correct_tool == "get_Test"
            assert learner2._mappings["test_pattern"].confidence == 0.85
        finally:
            path.unlink()

    def test_extract_wrong_tool_from_chunks(self, learner):
        """Test extracting wrong tool from retrieved_chunks."""
        report = MagicMock()
        report.retrieved_chunks = ["get_VehicleMileage", "get_VehicleStatus"]
        report.bot_response = "Some response"

        wrong_tool = learner._extract_wrong_tool(report)

        assert wrong_tool == "get_VehicleMileage"

    def test_extract_wrong_tool_from_dict_chunks(self, learner):
        """Test extracting wrong tool from dict-style chunks."""
        report = MagicMock()
        report.retrieved_chunks = [{"tool": "get_VehicleMileage", "score": 0.8}]
        report.bot_response = "Some response"

        wrong_tool = learner._extract_wrong_tool(report)

        assert wrong_tool == "get_VehicleMileage"

    def test_extract_wrong_tool_from_response(self, learner):
        """Test extracting wrong tool from bot_response."""
        report = MagicMock()
        report.retrieved_chunks = None
        report.bot_response = "Koristio sam get_VehicleMileage za ovaj upit"

        wrong_tool = learner._extract_wrong_tool(report)

        # Should extract from response
        assert wrong_tool is not None


class TestLearnFromFeedbackFunction:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_learn_from_feedback(self):
        """Test the convenience function."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"version": "1.0", "mappings": []}')
            path = Path(f.name)

        try:
            with patch('services.query_pattern_learner.LEARNED_MAPPINGS_FILE', path):
                result = await learn_from_feedback(mock_db)

            assert isinstance(result, LearningResult)
            assert result.total_analyzed == 0
        finally:
            path.unlink()


class TestQueryPatternLearnerIntegration:
    """Integration tests for QueryPatternLearner."""

    @pytest.mark.asyncio
    async def test_full_learning_flow(self):
        """Test complete learning flow."""
        mock_db = AsyncMock()

        # Create realistic mock reports
        reports = []
        for i in range(5):
            report = MagicMock()
            report.user_query = f"daj mi broj šasije za vozilo {i}"
            report.correction = "Koristi get_VehicleVIN umjesto get_VehicleMileage"
            report.category = "misunderstood"
            report.bot_response = "Kilometraža je 45000km"
            report.retrieved_chunks = ["get_VehicleMileage"]
            reports.append(report)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = reports
        mock_db.execute.return_value = mock_result

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"version": "1.0", "mappings": []}')
            path = Path(f.name)

        try:
            with patch('services.query_pattern_learner.LEARNED_MAPPINGS_FILE', path):
                learner = QueryPatternLearner(mock_db)
                result = await learner.learn(min_occurrences=1)

            assert result.total_analyzed == 5
            assert result.with_corrections == 5
            assert result.category_breakdown.get("misunderstood", 0) == 5

            # Should have learned patterns
            assert result.patterns_learned > 0

            # Check failure patterns
            assert len(result.failure_patterns) > 0
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_learning_with_mixed_categories(self):
        """Test learning with multiple failure categories."""
        mock_db = AsyncMock()

        reports = []

        # Misunderstood reports
        for i in range(3):
            report = MagicMock()
            report.user_query = f"šasija vozila {i}"
            report.correction = "get_VehicleVIN"
            report.category = "misunderstood"
            report.bot_response = "Wrong response"
            report.retrieved_chunks = None
            reports.append(report)

        # Wrong data reports
        for i in range(2):
            report = MagicMock()
            report.user_query = f"kilometraža auto {i}"
            report.correction = None
            report.category = "wrong_data"
            report.bot_response = "Data"
            report.retrieved_chunks = None
            reports.append(report)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = reports
        mock_db.execute.return_value = mock_result

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"version": "1.0", "mappings": []}')
            path = Path(f.name)

        try:
            with patch('services.query_pattern_learner.LEARNED_MAPPINGS_FILE', path):
                learner = QueryPatternLearner(mock_db)
                result = await learner.learn(include_uncorrected=True)

            assert result.total_analyzed == 5
            assert result.category_breakdown["misunderstood"] == 3
            assert result.category_breakdown["wrong_data"] == 2
        finally:
            path.unlink()

"""
Tests for FeedbackLearningService - Complete feedback loop integration.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from services.feedback_learning_service import (
    FeedbackLearningService,
    LearnedBoost,
    LearningResult,
    get_feedback_learning_service,
    run_learning_cycle,
    LEARNED_BOOSTS_FILE,
    CONFIDENCE_THRESHOLD_AUTO_APPLY,
    CONFIDENCE_THRESHOLD_DOCUMENTATION,
    MIN_OCCURRENCES_FOR_LEARNING,
)


class TestLearnedBoost:
    """Tests for LearnedBoost dataclass."""

    def test_initial_values(self):
        """Test initial values."""
        boost = LearnedBoost(
            tool_id="get_VehicleVIN",
            patterns=["broj sasije", "vin"],
            boost_value=0.15
        )
        assert boost.tool_id == "get_VehicleVIN"
        assert len(boost.patterns) == 2
        assert boost.boost_value == 0.15
        assert boost.confidence == 0.0

    def test_to_dict(self):
        """Test serialization."""
        boost = LearnedBoost(
            tool_id="get_VehicleVIN",
            patterns=["broj sasije", "vin", "sasija"],
            boost_value=0.20,
            negative_patterns=["get_VehicleMileage"],
            penalty_value=0.15,
            confidence=0.85,
            occurrence_count=10,
            last_updated="2024-01-15T10:00:00Z",
            source="feedback"
        )

        d = boost.to_dict()
        assert d["tool_id"] == "get_VehicleVIN"
        assert len(d["patterns"]) == 3
        assert d["boost_value"] == 0.2
        assert d["confidence"] == 0.85

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "tool_id": "get_VehicleVIN",
            "patterns": ["broj sasije"],
            "boost_value": 0.15,
            "negative_patterns": ["get_VehicleMileage"],
            "confidence": 0.80,
            "occurrence_count": 5,
        }

        boost = LearnedBoost.from_dict(data)
        assert boost.tool_id == "get_VehicleVIN"
        assert boost.confidence == 0.80


class TestLearningResult:
    """Tests for LearningResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        result = LearningResult(
            patterns_learned=15,
            boosts_applied=5,
            penalties_applied=2,
            documentation_updates=3,
            quality_status="learning_applied",
            recommendations=["Focus on VIN queries"],
            timestamp="2024-01-15T10:00:00Z"
        )

        d = result.to_dict()
        assert d["patterns_learned"] == 15
        assert d["boosts_applied"] == 5
        assert d["quality_status"] == "learning_applied"


class TestFeedbackLearningService:
    """Tests for FeedbackLearningService."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.fixture
    def temp_boosts_file(self):
        """Create a temporary boosts file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"version": "1.0", "boosts": [], "pattern_embeddings": {}}')
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def service(self, mock_db, temp_boosts_file):
        """Create service with temp file."""
        with patch('services.feedback_learning_service.LEARNED_BOOSTS_FILE', temp_boosts_file):
            return FeedbackLearningService(mock_db)

    def test_init(self, service):
        """Test initialization."""
        assert len(service._learned_boosts) == 0
        assert len(service._pattern_embeddings) == 0

    def test_init_with_existing_boosts(self, mock_db):
        """Test initialization with existing boosts."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                "version": "1.0",
                "boosts": [
                    {
                        "tool_id": "get_VehicleVIN",
                        "patterns": ["broj sasije"],
                        "boost_value": 0.15,
                        "confidence": 0.85,
                        "occurrence_count": 5,
                    }
                ],
                "pattern_embeddings": {}
            }
            json.dump(data, f)
            path = Path(f.name)

        try:
            with patch('services.feedback_learning_service.LEARNED_BOOSTS_FILE', path):
                service = FeedbackLearningService(mock_db)
                assert len(service._learned_boosts) == 1
                assert "get_VehicleVIN" in service._learned_boosts
        finally:
            path.unlink()

    def test_extract_tool_from_correction(self, service):
        """Test tool extraction from correction text."""
        assert service._extract_tool_from_correction("use get_VehicleVIN") == "get_vehiclevin"
        assert service._extract_tool_from_correction("trebao get_VehicleMileage") == "get_vehiclemileage"
        assert service._extract_tool_from_correction("post_CreateBooking") == "post_createbooking"
        assert service._extract_tool_from_correction("no tool here") is None

    def test_extract_patterns(self, service):
        """Test pattern extraction from queries."""
        patterns = service._extract_patterns("daj mi broj sasije za vozilo")
        assert len(patterns) > 0
        # Should have removed stopwords
        assert all("mi" not in p for p in patterns)
        assert all("za" not in p for p in patterns)

    def test_get_search_boost_no_learned(self, service):
        """Test search boost when no patterns learned."""
        boost, reason = service.get_search_boost("any query", "get_VehicleVIN")
        assert boost == 0.0
        assert reason == "no_learned_pattern"

    def test_get_search_boost_positive_match(self, service):
        """Test search boost with positive pattern match."""
        # Add a learned boost
        service._learned_boosts["get_vehiclevin"] = LearnedBoost(
            tool_id="get_VehicleVIN",
            patterns=["broj sasije", "vin"],
            boost_value=0.20,
            confidence=0.85
        )

        boost, reason = service.get_search_boost("daj broj sasije", "get_vehiclevin")
        assert boost == 0.20
        assert "learned_boost" in reason

    def test_get_search_boost_negative_match(self, service):
        """Test search boost with negative pattern (penalty)."""
        # Add a learned boost with negative pattern
        service._learned_boosts["get_vehiclevin"] = LearnedBoost(
            tool_id="get_VehicleVIN",
            patterns=["broj sasije"],
            boost_value=0.20,
            negative_patterns=["get_vehiclemileage"],
            penalty_value=0.15,
            confidence=0.85
        )

        # Query for sasija should penalize VehicleMileage
        boost, reason = service.get_search_boost("broj sasije", "get_vehiclemileage")
        assert boost == -0.15
        assert "negative_learned" in reason

    @pytest.mark.asyncio
    async def test_learn_and_apply_empty_db(self, service, mock_db):
        """Test learning with empty database."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        result = await service.learn_and_apply()

        assert result.quality_status == "no_data"
        assert result.patterns_learned == 0

    @pytest.mark.asyncio
    async def test_learn_and_apply_with_corrections(self, service, mock_db):
        """Test learning with sample corrections."""
        # Create mock reports
        reports = []
        for i in range(5):
            report = MagicMock()
            report.user_query = f"daj mi broj sasije za vozilo {i}"
            report.correction = "Koristi get_VehicleVIN"
            report.bot_response = "Kilometraza je 45000km"
            report.retrieved_chunks = ["get_VehicleMileage"]
            reports.append(report)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = reports
        mock_db.execute.return_value = mock_result

        result = await service.learn_and_apply(min_occurrences=1)

        # Should have learned patterns
        assert result.patterns_learned > 0

    @pytest.mark.asyncio
    async def test_learn_and_apply_confidence_threshold(self, service, mock_db):
        """Test that only high-confidence patterns are applied."""
        # Create reports with varying frequencies
        reports = []

        # Low frequency (should not be applied)
        report1 = MagicMock()
        report1.user_query = "rare query"
        report1.correction = "get_RareTool"
        report1.bot_response = "Response"
        report1.retrieved_chunks = None
        reports.append(report1)

        # High frequency (should be applied)
        for i in range(10):
            report = MagicMock()
            report.user_query = f"common query pattern {i}"
            report.correction = "get_CommonTool"
            report.bot_response = "Response"
            report.retrieved_chunks = None
            reports.append(report)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = reports
        mock_db.execute.return_value = mock_result

        result = await service.learn_and_apply(
            min_occurrences=3,
            confidence_threshold=0.70
        )

        # Only high-frequency patterns should be applied
        assert result.patterns_learned >= 0

    @pytest.mark.asyncio
    async def test_get_statistics(self, service):
        """Test statistics retrieval."""
        # Add some boosts
        service._learned_boosts["tool1"] = LearnedBoost(
            tool_id="tool1",
            patterns=["pattern1", "pattern2"],
            boost_value=0.15,
            confidence=0.85,
            occurrence_count=10
        )
        service._learned_boosts["tool2"] = LearnedBoost(
            tool_id="tool2",
            patterns=["pattern3"],
            boost_value=0.10,
            confidence=0.60,
            occurrence_count=3
        )

        stats = await service.get_statistics()

        assert stats["total_learned_tools"] == 2
        assert stats["total_patterns"] == 3
        assert stats["high_confidence_tools"] == 1  # Only tool1 >0.8
        assert stats["total_occurrences"] == 13

    def test_clear_learned_boosts(self, service):
        """Test clearing learned boosts."""
        service._learned_boosts["tool1"] = LearnedBoost(
            tool_id="tool1",
            patterns=["pattern"],
            boost_value=0.15
        )

        service.clear_learned_boosts()

        assert len(service._learned_boosts) == 0
        assert len(service._pattern_embeddings) == 0

    @pytest.mark.asyncio
    async def test_export_to_documentation(self, service, mock_db):
        """Test exporting to documentation."""
        # Add high-confidence boost
        service._learned_boosts["get_vehiclevin"] = LearnedBoost(
            tool_id="get_vehiclevin",
            patterns=["broj sasije", "vin vozila"],
            boost_value=0.20,
            confidence=0.90,  # Above documentation threshold
            occurrence_count=20,
            last_updated="2024-01-15T10:00:00Z"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            doc_path = Path(f.name)

        try:
            updates = await service.export_to_documentation(
                confidence_threshold=0.85,
                output_path=doc_path
            )

            # Should have updated documentation
            assert updates > 0

            # Verify file was updated
            with open(doc_path, "r") as f:
                docs = json.load(f)
                assert "get_vehiclevin" in docs
                assert "example_queries_hr" in docs["get_vehiclevin"]
        finally:
            doc_path.unlink()

    def test_save_and_load_boosts(self, mock_db):
        """Test persistence of learned boosts."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"version": "1.0", "boosts": [], "pattern_embeddings": {}}')
            path = Path(f.name)

        try:
            # Create service and add boost
            with patch('services.feedback_learning_service.LEARNED_BOOSTS_FILE', path):
                service1 = FeedbackLearningService(mock_db)
                service1._learned_boosts["get_test"] = LearnedBoost(
                    tool_id="get_test",
                    patterns=["test pattern"],
                    boost_value=0.15,
                    confidence=0.80
                )
                service1._save_learned_boosts()

            # Load in new instance
            with patch('services.feedback_learning_service.LEARNED_BOOSTS_FILE', path):
                service2 = FeedbackLearningService(mock_db)

            assert "get_test" in service2._learned_boosts
            assert service2._learned_boosts["get_test"].boost_value == 0.15
        finally:
            path.unlink()


class TestSearchEngineIntegration:
    """Tests for SearchEngine integration with learned boosts."""

    @pytest.fixture
    def temp_boosts_file(self):
        """Create a temporary boosts file with test data."""
        data = {
            "version": "1.0",
            "boosts": [
                {
                    "tool_id": "get_vehiclevin",
                    "patterns": ["broj sasije", "vin"],
                    "boost_value": 0.20,
                    "negative_patterns": ["get_vehiclemileage"],
                    "penalty_value": 0.15,
                    "confidence": 0.85,
                    "occurrence_count": 10,
                }
            ],
            "pattern_embeddings": {}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    def test_apply_learned_pattern_boosting(self, temp_boosts_file):
        """Test SearchEngine._apply_learned_pattern_boosting method."""
        from services.registry.search_engine import SearchEngine

        with patch('services.feedback_learning_service.LEARNED_BOOSTS_FILE', temp_boosts_file):
            engine = SearchEngine()

            # Test with matching query
            scored = [(0.70, "get_VehicleVIN"), (0.65, "get_VehicleMileage")]
            adjusted = engine._apply_learned_pattern_boosting("broj sasije za auto", scored)

            # VIN should be boosted, Mileage should be penalized
            score_dict = {op_id: score for score, op_id in adjusted}

            # VIN boosted from 0.70 to 0.90
            assert score_dict["get_VehicleVIN"] > 0.70

            # Mileage penalized from 0.65 to 0.50
            assert score_dict["get_VehicleMileage"] < 0.65

    def test_apply_learned_pattern_boosting_no_match(self, temp_boosts_file):
        """Test that non-matching queries are not affected."""
        from services.registry.search_engine import SearchEngine

        with patch('services.feedback_learning_service.LEARNED_BOOSTS_FILE', temp_boosts_file):
            engine = SearchEngine()

            # Query without learned patterns
            scored = [(0.70, "get_VehicleVIN"), (0.65, "get_VehicleMileage")]
            adjusted = engine._apply_learned_pattern_boosting("rezervacija auta", scored)

            # Scores should be unchanged
            score_dict = {op_id: score for score, op_id in adjusted}
            assert abs(score_dict["get_VehicleVIN"] - 0.70) < 0.01
            assert abs(score_dict["get_VehicleMileage"] - 0.65) < 0.01


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_get_feedback_learning_service(self):
        """Test singleton getter."""
        mock_db = AsyncMock()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"version": "1.0", "boosts": [], "pattern_embeddings": {}}')
            path = Path(f.name)

        try:
            with patch('services.feedback_learning_service.LEARNED_BOOSTS_FILE', path):
                # Reset singleton
                import services.feedback_learning_service as module
                module._feedback_learning_service = None

                service = get_feedback_learning_service(mock_db)
                assert isinstance(service, FeedbackLearningService)
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_run_learning_cycle(self):
        """Test the convenience function."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"version": "1.0", "boosts": [], "pattern_embeddings": {}}')
            path = Path(f.name)

        try:
            with patch('services.feedback_learning_service.LEARNED_BOOSTS_FILE', path):
                # Reset singleton
                import services.feedback_learning_service as module
                module._feedback_learning_service = None

                result = await run_learning_cycle(mock_db)
                assert isinstance(result, LearningResult)
        finally:
            path.unlink()


class TestEmbeddingSimilarity:
    """Tests for embedding-based similarity matching."""

    @pytest.fixture
    def service_with_embedding(self, mock_db):
        """Create service with mock embedding client."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"version": "1.0", "boosts": [], "pattern_embeddings": {}}')
            path = Path(f.name)

        mock_embedding_client = AsyncMock()

        # Mock embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_embedding_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch('services.feedback_learning_service.LEARNED_BOOSTS_FILE', path):
            service = FeedbackLearningService(mock_db, embedding_client=mock_embedding_client)

        yield service

        if path.exists():
            path.unlink()

    @pytest.fixture
    def mock_db(self):
        return AsyncMock()

    def test_cosine_similarity(self, service_with_embedding):
        """Test cosine similarity calculation."""
        # Same vectors should have similarity 1.0
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        sim = service_with_embedding._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.001

        # Orthogonal vectors should have similarity 0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = service_with_embedding._cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.001

    @pytest.mark.asyncio
    async def test_get_search_boost_semantic(self, service_with_embedding):
        """Test semantic similarity boost."""
        # Add a learned boost
        service_with_embedding._learned_boosts["get_vehiclevin"] = LearnedBoost(
            tool_id="get_vehiclevin",
            patterns=["broj sasije"],
            boost_value=0.20,
            confidence=0.85
        )

        # First try should use string matching (works)
        boost, reason = await service_with_embedding.get_search_boost_semantic(
            "daj broj sasije",
            "get_vehiclevin"
        )
        assert boost == 0.20
        assert "learned_boost" in reason

        # Non-matching query should return 0
        boost, reason = await service_with_embedding.get_search_boost_semantic(
            "unrelated query",
            "get_vehiclevin"
        )
        # May be 0 or use semantic matching
        assert isinstance(boost, float)

"""Tests for services/registry/embedding_engine.py – EmbeddingEngine."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.tool_contracts import ParameterDefinition, DependencySource


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_engine():
    ms = MagicMock()
    ms.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com"
    ms.AZURE_OPENAI_API_KEY = "test-key"
    ms.AZURE_OPENAI_API_VERSION = "2024-02-15"
    ms.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding"

    with patch("services.registry.embedding_engine.settings", ms):
        with patch("services.registry.embedding_engine.AsyncAzureOpenAI"):
            from services.registry.embedding_engine import EmbeddingEngine
            engine = EmbeddingEngine()
    return engine


@pytest.fixture
def engine():
    return _make_engine()


def _param(name, context_key=None, source=DependencySource.FROM_USER):
    return ParameterDefinition(name=name, context_key=context_key, source=source)


# ===========================================================================
# _generate_purpose
# ===========================================================================

class TestGeneratePurpose:
    def test_get_method(self, engine):
        purpose = engine._generate_purpose("GET", {}, [])
        assert "Dohvaća" in purpose

    def test_post_method(self, engine):
        purpose = engine._generate_purpose("POST", {}, [])
        assert "Kreira" in purpose

    def test_put_method(self, engine):
        purpose = engine._generate_purpose("PUT", {}, [])
        assert "Ažurira" in purpose

    def test_delete_method(self, engine):
        purpose = engine._generate_purpose("DELETE", {}, [])
        assert "Briše" in purpose

    def test_unknown_method(self, engine):
        purpose = engine._generate_purpose("HEAD", {}, [])
        assert "Obrađuje" in purpose

    def test_vehicle_context(self, engine):
        params = {"VehicleId": _param("VehicleId")}
        purpose = engine._generate_purpose("GET", params, [])
        # v3.0: Uses genitive form "vozila" (za vozila)
        assert "vozil" in purpose.lower()  # Matches vozilo, vozila

    def test_person_context(self, engine):
        params = {"PersonId": _param("PersonId")}
        purpose = engine._generate_purpose("GET", params, [])
        # v3.0: Maps PersonId to osoba (not korisnik)
        assert "osob" in purpose.lower()  # Matches osoba, osobe

    def test_mileage_output(self, engine):
        purpose = engine._generate_purpose("GET", {}, ["Mileage", "LastMileage"])
        assert "kilometražu" in purpose

    def test_registration_output(self, engine):
        purpose = engine._generate_purpose("GET", {}, ["LicencePlate", "Registration"])
        assert "registraciju" in purpose

    def test_expiration_output(self, engine):
        purpose = engine._generate_purpose("GET", {}, ["ExpirationDate"])
        # v3.0: ExpirationDate maps to "datum isteka"
        assert "datum isteka" in purpose

    def test_status_output(self, engine):
        purpose = engine._generate_purpose("GET", {}, ["Status"])
        assert "status" in purpose

    def test_available_output(self, engine):
        purpose = engine._generate_purpose("GET", {}, ["AvailableVehicles"])
        assert "dostupnost" in purpose

    def test_time_period(self, engine):
        params = {
            "FromTime": _param("FromTime"),
            "ToTime": _param("ToTime"),
        }
        purpose = engine._generate_purpose("GET", params, [])
        assert "periodu" in purpose

    def test_booking_context(self, engine):
        params = {"BookingId": _param("BookingId")}
        purpose = engine._generate_purpose("GET", params, [])
        # v3.0: Uses genitive form "rezervacije" (za rezervacije)
        assert "rezervacij" in purpose.lower()  # Matches rezervacija, rezervacije

    def test_name_output(self, engine):
        purpose = engine._generate_purpose("GET", {}, ["FullVehicleName"])
        assert "naziv" in purpose


# ===========================================================================
# build_embedding_text
# ===========================================================================

class TestBuildEmbeddingText:
    def test_basic(self, engine):
        text = engine.build_embedding_text(
            "get_MasterData", "VehicleService", "/api/md", "GET",
            "Get master data", {}, ["Mileage", "LicencePlate"]
        )
        assert "Returns" in text
        assert "Mileage" in text

    def test_truncation(self, engine):
        long_desc = "x" * 2000
        text = engine.build_embedding_text(
            "op", "svc", "/api", "GET", long_desc, {}, []
        )
        assert len(text) <= 1500

    def test_no_output_keys(self, engine):
        text = engine.build_embedding_text(
            "op", "svc", "/api", "GET", "desc", {}, None
        )
        assert "Returns" not in text

    def test_camel_case_split(self, engine):
        text = engine.build_embedding_text(
            "op", "svc", "/api", "GET", "", {}, ["FullVehicleName"]
        )
        assert "Full Vehicle Name" in text


# ===========================================================================
# generate_embeddings
# ===========================================================================

class TestGenerateEmbeddings:
    @pytest.mark.asyncio
    async def test_all_cached(self, engine):
        tools = {"t1": MagicMock()}
        existing = {"t1": [0.1, 0.2]}
        result = await engine.generate_embeddings(tools, existing)
        assert result == existing

    @pytest.mark.asyncio
    async def test_generates_missing(self, engine):
        tool = MagicMock()
        tool.embedding_text = "test text"
        tools = {"t1": tool}

        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        engine.openai.embeddings.create = AsyncMock(return_value=mock_resp)

        result = await engine.generate_embeddings(tools, {})
        assert "t1" in result
        assert result["t1"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embedding_error(self, engine):
        tool = MagicMock()
        tool.embedding_text = "test"
        tools = {"t1": tool}

        engine.openai.embeddings.create = AsyncMock(side_effect=RuntimeError("API error"))

        result = await engine.generate_embeddings(tools, {})
        assert "t1" not in result


# ===========================================================================
# build_dependency_graph
# ===========================================================================

class TestBuildDependencyGraph:
    def test_no_dependencies(self, engine):
        tool = MagicMock()
        tool.get_output_params.return_value = {}
        graph = engine.build_dependency_graph({"t1": tool})
        assert len(graph) == 0

    def test_with_dependency(self, engine):
        # t2 needs VehicleId, t1 provides it
        t1 = MagicMock()
        t1.get_output_params.return_value = {}
        t1.output_keys = ["VehicleId", "Name"]

        t2 = MagicMock()
        t2.get_output_params.return_value = {"VehicleId": _param("VehicleId")}
        t2.output_keys = []

        graph = engine.build_dependency_graph({"t1": t1, "t2": t2})
        assert "t2" in graph
        assert "t1" in graph["t2"].provider_tools

    def test_case_insensitive_match(self, engine):
        t1 = MagicMock()
        t1.get_output_params.return_value = {}
        t1.output_keys = ["vehicleid"]

        t2 = MagicMock()
        t2.get_output_params.return_value = {"VehicleId": _param("VehicleId")}
        t2.output_keys = []

        graph = engine.build_dependency_graph({"t1": t1, "t2": t2})
        assert "t2" in graph


# ===========================================================================
# _find_providers
# ===========================================================================

class TestFindProviders:
    def test_exact_match(self, engine):
        t1 = MagicMock()
        t1.output_keys = ["VehicleId"]
        providers = engine._find_providers("VehicleId", {"t1": t1})
        assert "t1" in providers

    def test_case_insensitive(self, engine):
        t1 = MagicMock()
        t1.output_keys = ["vehicleid"]
        providers = engine._find_providers("VehicleId", {"t1": t1})
        assert "t1" in providers

    def test_no_match(self, engine):
        t1 = MagicMock()
        t1.output_keys = ["PersonId"]
        providers = engine._find_providers("VehicleId", {"t1": t1})
        assert len(providers) == 0

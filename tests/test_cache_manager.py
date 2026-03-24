"""Tests for services/registry/cache_manager.py – CacheManager."""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from services.registry.cache_manager import CacheManager, CACHE_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path


@pytest.fixture
def manager(cache_dir):
    with patch("services.registry.cache_manager.CACHE_DIR", cache_dir), \
         patch("services.registry.cache_manager.MANIFEST_CACHE_FILE", cache_dir / "swagger_manifest.json"), \
         patch("services.registry.cache_manager.METADATA_CACHE_FILE", cache_dir / "tool_metadata.json"), \
         patch("services.registry.cache_manager.EMBEDDINGS_CACHE_FILE", cache_dir / "tool_embeddings.json"):
        m = CacheManager()
    return m


def _write_cache(cache_dir, manifest=None, metadata=None, embeddings=None):
    if manifest is not None:
        (cache_dir / "swagger_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if metadata is not None:
        (cache_dir / "tool_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    if embeddings is not None:
        (cache_dir / "tool_embeddings.json").write_text(json.dumps(embeddings), encoding="utf-8")


# ===========================================================================
# __init__
# ===========================================================================

class TestInit:
    def test_creates_dir(self, cache_dir):
        with patch("services.registry.cache_manager.CACHE_DIR", cache_dir / "new"):
            CacheManager()
            assert (cache_dir / "new").exists()


# ===========================================================================
# is_cache_valid
# ===========================================================================

class TestIsCacheValid:
    @pytest.mark.asyncio
    async def test_no_manifest(self, manager, cache_dir):
        with patch("services.registry.cache_manager.MANIFEST_CACHE_FILE", cache_dir / "swagger_manifest.json"):
            assert await manager.is_cache_valid(["url1"]) is False

    @pytest.mark.asyncio
    async def test_no_metadata(self, manager, cache_dir):
        _write_cache(cache_dir, manifest={"test": 1})
        with patch("services.registry.cache_manager.MANIFEST_CACHE_FILE", cache_dir / "swagger_manifest.json"), \
             patch("services.registry.cache_manager.METADATA_CACHE_FILE", cache_dir / "missing.json"):
            assert await manager.is_cache_valid(["url1"]) is False

    @pytest.mark.asyncio
    async def test_empty_manifest(self, manager, cache_dir):
        (cache_dir / "swagger_manifest.json").write_text("", encoding="utf-8")
        _write_cache(cache_dir, metadata={"tools": [{"test": 1}]}, embeddings={"embeddings": {}})
        with patch("services.registry.cache_manager.MANIFEST_CACHE_FILE", cache_dir / "swagger_manifest.json"), \
             patch("services.registry.cache_manager.METADATA_CACHE_FILE", cache_dir / "tool_metadata.json"), \
             patch("services.registry.cache_manager.EMBEDDINGS_CACHE_FILE", cache_dir / "tool_embeddings.json"):
            assert await manager.is_cache_valid(["url1"]) is False

    @pytest.mark.asyncio
    async def test_version_mismatch(self, manager, cache_dir):
        _write_cache(
            cache_dir,
            manifest={"swagger_sources": ["url1"], "cache_version": "old"},
            metadata={"tools": [{"op_id": "t1"}]},
            embeddings={"embeddings": {}}
        )
        with patch("services.registry.cache_manager.MANIFEST_CACHE_FILE", cache_dir / "swagger_manifest.json"), \
             patch("services.registry.cache_manager.METADATA_CACHE_FILE", cache_dir / "tool_metadata.json"), \
             patch("services.registry.cache_manager.EMBEDDINGS_CACHE_FILE", cache_dir / "tool_embeddings.json"):
            assert await manager.is_cache_valid(["url1"]) is False

    @pytest.mark.asyncio
    async def test_source_mismatch(self, manager, cache_dir):
        _write_cache(
            cache_dir,
            manifest={"swagger_sources": ["url1"], "cache_version": CACHE_VERSION},
            metadata={"tools": [{"op_id": "t1"}]},
            embeddings={"embeddings": {}}
        )
        with patch("services.registry.cache_manager.MANIFEST_CACHE_FILE", cache_dir / "swagger_manifest.json"), \
             patch("services.registry.cache_manager.METADATA_CACHE_FILE", cache_dir / "tool_metadata.json"), \
             patch("services.registry.cache_manager.EMBEDDINGS_CACHE_FILE", cache_dir / "tool_embeddings.json"):
            assert await manager.is_cache_valid(["url2"]) is False

    @pytest.mark.asyncio
    async def test_empty_tools(self, manager, cache_dir):
        _write_cache(
            cache_dir,
            manifest={"swagger_sources": ["url1"], "cache_version": CACHE_VERSION},
            metadata={"tools": []},
            embeddings={"embeddings": {}}
        )
        with patch("services.registry.cache_manager.MANIFEST_CACHE_FILE", cache_dir / "swagger_manifest.json"), \
             patch("services.registry.cache_manager.METADATA_CACHE_FILE", cache_dir / "tool_metadata.json"), \
             patch("services.registry.cache_manager.EMBEDDINGS_CACHE_FILE", cache_dir / "tool_embeddings.json"):
            assert await manager.is_cache_valid(["url1"]) is False

    @pytest.mark.asyncio
    async def test_valid_cache(self, manager, cache_dir):
        _write_cache(
            cache_dir,
            manifest={"swagger_sources": ["url1"], "cache_version": CACHE_VERSION},
            metadata={"tools": [{"op_id": "t1"}]},
            embeddings={"embeddings": {"t1": [0.1]}}
        )
        with patch("services.registry.cache_manager.MANIFEST_CACHE_FILE", cache_dir / "swagger_manifest.json"), \
             patch("services.registry.cache_manager.METADATA_CACHE_FILE", cache_dir / "tool_metadata.json"), \
             patch("services.registry.cache_manager.EMBEDDINGS_CACHE_FILE", cache_dir / "tool_embeddings.json"):
            assert await manager.is_cache_valid(["url1"]) is True

    @pytest.mark.asyncio
    async def test_corrupt_json(self, manager, cache_dir):
        (cache_dir / "swagger_manifest.json").write_text("not json", encoding="utf-8")
        (cache_dir / "tool_metadata.json").write_text("{}", encoding="utf-8")
        (cache_dir / "tool_embeddings.json").write_text("{}", encoding="utf-8")
        with patch("services.registry.cache_manager.MANIFEST_CACHE_FILE", cache_dir / "swagger_manifest.json"), \
             patch("services.registry.cache_manager.METADATA_CACHE_FILE", cache_dir / "tool_metadata.json"), \
             patch("services.registry.cache_manager.EMBEDDINGS_CACHE_FILE", cache_dir / "tool_embeddings.json"):
            assert await manager.is_cache_valid(["url1"]) is False


# ===========================================================================
# _read_json_sync / _write_json_sync
# ===========================================================================

class TestJsonIO:
    def test_read_write(self, manager, tmp_path):
        path = tmp_path / "test.json"
        manager._write_json_sync(path, {"key": "value"})
        data = manager._read_json_sync(path)
        assert data["key"] == "value"

    def test_unicode(self, manager, tmp_path):
        path = tmp_path / "test.json"
        manager._write_json_sync(path, {"key": "Čakovec"})
        data = manager._read_json_sync(path)
        assert data["key"] == "Čakovec"


# ===========================================================================
# save_cache / load_cache
# ===========================================================================

class TestSaveLoadCache:
    @pytest.mark.asyncio
    async def test_save_and_load(self, cache_dir):
        with patch("services.registry.cache_manager.CACHE_DIR", cache_dir), \
             patch("services.registry.cache_manager.MANIFEST_CACHE_FILE", cache_dir / "swagger_manifest.json"), \
             patch("services.registry.cache_manager.METADATA_CACHE_FILE", cache_dir / "tool_metadata.json"), \
             patch("services.registry.cache_manager.EMBEDDINGS_CACHE_FILE", cache_dir / "tool_embeddings.json"):
            m = CacheManager()

            # Create mock tool
            tool = MagicMock()
            tool.model_dump.return_value = {
                "operation_id": "get_Test",
                "service_name": "TestService",
                "service_url": "http://test",
                "path": "/api/test",
                "method": "GET",
            }

            dep = MagicMock()
            dep.model_dump.return_value = {
                "tool_id": "t1",
                "required_outputs": [],
                "provider_tools": [],
            }

            await m.save_cache(
                swagger_sources=["url1"],
                tools=[tool],
                embeddings={"get_Test": [0.1, 0.2]},
                dependency_graph=[dep]
            )

            # Verify files exist
            assert (cache_dir / "swagger_manifest.json").exists()
            assert (cache_dir / "tool_metadata.json").exists()
            assert (cache_dir / "tool_embeddings.json").exists()


# ===========================================================================
# _verify_cache_files
# ===========================================================================

class TestVerifyCacheFiles:
    @pytest.mark.asyncio
    async def test_missing_file(self, cache_dir):
        with patch("services.registry.cache_manager.CACHE_DIR", cache_dir), \
             patch("services.registry.cache_manager.MANIFEST_CACHE_FILE", cache_dir / "missing.json"), \
             patch("services.registry.cache_manager.METADATA_CACHE_FILE", cache_dir / "tool_metadata.json"), \
             patch("services.registry.cache_manager.EMBEDDINGS_CACHE_FILE", cache_dir / "tool_embeddings.json"):
            m = CacheManager()
            with pytest.raises(RuntimeError, match="not created"):
                await m._verify_cache_files()

    @pytest.mark.asyncio
    async def test_empty_file(self, cache_dir):
        (cache_dir / "swagger_manifest.json").write_text("", encoding="utf-8")
        (cache_dir / "tool_metadata.json").write_text("{}", encoding="utf-8")
        (cache_dir / "tool_embeddings.json").write_text("{}", encoding="utf-8")
        with patch("services.registry.cache_manager.MANIFEST_CACHE_FILE", cache_dir / "swagger_manifest.json"), \
             patch("services.registry.cache_manager.METADATA_CACHE_FILE", cache_dir / "tool_metadata.json"), \
             patch("services.registry.cache_manager.EMBEDDINGS_CACHE_FILE", cache_dir / "tool_embeddings.json"):
            m = CacheManager()
            with pytest.raises(RuntimeError, match="empty"):
                await m._verify_cache_files()

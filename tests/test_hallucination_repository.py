"""Tests for services/hallucination_repository.py – HallucinationRepository."""
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from services.hallucination_repository import (
    HallucinationRepository,
    ALLOWED_CATEGORIES,
    MAX_LIMIT,
    DEFAULT_LIMIT,
    MIN_LIMIT,
    MAX_EXPORT_LIMIT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_db():
    """Create a mocked AsyncSession."""
    db = AsyncMock()
    return db


def _mock_result(scalars=None, scalar=None, rowcount=0, all_rows=None):
    """Create a mock DB execution result."""
    result = MagicMock()
    if scalars is not None:
        result.scalars.return_value.all.return_value = scalars
    if scalar is not None:
        result.scalar.return_value = scalar
    result.scalar_one_or_none = MagicMock(return_value=scalar)
    result.rowcount = rowcount
    if all_rows is not None:
        result.all.return_value = all_rows
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    return _mock_db()


@pytest.fixture
def repo(db):
    return HallucinationRepository(db)


# ===========================================================================
# Validation helpers
# ===========================================================================

class TestValidateLimit:
    def test_normal_limit(self, repo):
        assert repo._validate_limit(50) == 50

    def test_above_max(self, repo):
        assert repo._validate_limit(2000) == MAX_LIMIT

    def test_below_min(self, repo):
        assert repo._validate_limit(0) == MIN_LIMIT
        assert repo._validate_limit(-5) == MIN_LIMIT

    def test_non_int(self, repo):
        assert repo._validate_limit("abc") == DEFAULT_LIMIT

    def test_custom_max(self, repo):
        assert repo._validate_limit(20000, max_limit=MAX_EXPORT_LIMIT) == MAX_EXPORT_LIMIT


class TestValidateOffset:
    def test_normal(self, repo):
        assert repo._validate_offset(10) == 10

    def test_negative(self, repo):
        assert repo._validate_offset(-1) == 0

    def test_non_int(self, repo):
        assert repo._validate_offset("abc") == 0


class TestValidateCategory:
    def test_valid_categories(self, repo):
        for cat in ALLOWED_CATEGORIES:
            assert repo._validate_category(cat) == cat

    def test_none(self, repo):
        assert repo._validate_category(None) is None

    def test_invalid(self, repo):
        assert repo._validate_category("invalid_cat") is None


# ===========================================================================
# GDPR masking
# ===========================================================================

class TestGDPRMasking:
    @pytest.mark.asyncio
    async def test_no_masking_service(self, repo):
        q, r, f = await repo._apply_gdpr_masking("query", "response", "feedback")
        assert q == "query"
        assert r == "response"
        assert f == "feedback"

    @pytest.mark.asyncio
    async def test_with_masking_service(self, db):
        masker = MagicMock()
        masker.mask_pii.side_effect = lambda t: MagicMock(masked_text=f"MASKED:{t}")
        repo = HallucinationRepository(db, gdpr_masking_service=masker)
        q, r, f = await repo._apply_gdpr_masking("q", "r", "f")
        assert q == "MASKED:q"
        assert r == "MASKED:r"
        assert f == "MASKED:f"

    @pytest.mark.asyncio
    async def test_masking_error_returns_original(self, db):
        masker = MagicMock()
        masker.mask_pii.side_effect = Exception("masking failed")
        repo = HallucinationRepository(db, gdpr_masking_service=masker)
        q, r, f = await repo._apply_gdpr_masking("q", "r", "f")
        assert q == "q"


# ===========================================================================
# create
# ===========================================================================

class TestCreate:
    @pytest.mark.asyncio
    async def test_basic_create(self, repo, db):
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        report = await repo.create(
            user_query="q", bot_response="r",
            user_feedback="wrong", model="gpt-4"
        )
        assert report.user_query == "q"
        assert report.reviewed is False
        db.add.assert_called_once()
        db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_with_all_fields(self, repo, db):
        db.commit = AsyncMock()
        report = await repo.create(
            user_query="q", bot_response="r", user_feedback="f",
            model="m", conversation_id="c1", tenant_id="t1",
            retrieved_chunks=["chunk1"], api_raw_response={"key": "val"}
        )
        assert report.conversation_id == "c1"
        assert report.tenant_id == "t1"
        assert report.retrieved_chunks == ["chunk1"]

    @pytest.mark.asyncio
    async def test_create_no_gdpr_masking(self, repo, db):
        db.commit = AsyncMock()
        report = await repo.create(
            user_query="q", bot_response="r",
            user_feedback="f", model="m",
            apply_gdpr_masking=False
        )
        assert report.user_query == "q"

    @pytest.mark.asyncio
    async def test_create_failure_rollback(self, repo, db):
        db.commit = AsyncMock(side_effect=Exception("db error"))
        db.rollback = AsyncMock()
        with pytest.raises(Exception, match="db error"):
            await repo.create(
                user_query="q", bot_response="r",
                user_feedback="f", model="m"
            )
        db.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_generates_uuid(self, repo, db):
        db.commit = AsyncMock()
        report = await repo.create(
            user_query="q", bot_response="r",
            user_feedback="f", model="m"
        )
        assert isinstance(report.id, uuid.UUID)


# ===========================================================================
# get_by_id
# ===========================================================================

class TestGetById:
    @pytest.mark.asyncio
    async def test_found(self, repo, db):
        mock_report = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_report
        db.execute = AsyncMock(return_value=result)
        report = await repo.get_by_id(uuid.uuid4())
        assert report is mock_report

    @pytest.mark.asyncio
    async def test_not_found(self, repo, db):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=result)
        assert await repo.get_by_id(uuid.uuid4()) is None

    @pytest.mark.asyncio
    async def test_db_error(self, repo, db):
        db.execute = AsyncMock(side_effect=Exception("fail"))
        assert await repo.get_by_id(uuid.uuid4()) is None


# ===========================================================================
# get_unreviewed
# ===========================================================================

class TestGetUnreviewed:
    @pytest.mark.asyncio
    async def test_returns_list(self, repo, db):
        reports = [MagicMock(), MagicMock()]
        result = MagicMock()
        result.scalars.return_value.all.return_value = reports
        db.execute = AsyncMock(return_value=result)
        got = await repo.get_unreviewed()
        assert len(got) == 2

    @pytest.mark.asyncio
    async def test_with_tenant_filter(self, repo, db):
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result)
        await repo.get_unreviewed(tenant_id="T1")
        db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_error_returns_empty(self, repo, db):
        db.execute = AsyncMock(side_effect=Exception("fail"))
        assert await repo.get_unreviewed() == []


# ===========================================================================
# get_all
# ===========================================================================

class TestGetAll:
    @pytest.mark.asyncio
    async def test_basic(self, repo, db):
        result = MagicMock()
        result.scalars.return_value.all.return_value = [MagicMock()]
        db.execute = AsyncMock(return_value=result)
        got = await repo.get_all()
        assert len(got) == 1

    @pytest.mark.asyncio
    async def test_with_filters(self, repo, db):
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result)
        await repo.get_all(tenant_id="T1", category="wrong_data", reviewed_only=True)
        db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_category_ignored(self, repo, db):
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result)
        await repo.get_all(category="INVALID")
        db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_error_returns_empty(self, repo, db):
        db.execute = AsyncMock(side_effect=Exception("fail"))
        assert await repo.get_all() == []


# ===========================================================================
# exists
# ===========================================================================

class TestExists:
    @pytest.mark.asyncio
    async def test_exists_true(self, repo, db):
        result = MagicMock()
        result.scalar.return_value = 1
        db.execute = AsyncMock(return_value=result)
        assert await repo.exists(uuid.uuid4()) is True

    @pytest.mark.asyncio
    async def test_exists_false(self, repo, db):
        result = MagicMock()
        result.scalar.return_value = 0
        db.execute = AsyncMock(return_value=result)
        assert await repo.exists(uuid.uuid4()) is False

    @pytest.mark.asyncio
    async def test_exists_none_scalar(self, repo, db):
        result = MagicMock()
        result.scalar.return_value = None
        db.execute = AsyncMock(return_value=result)
        assert await repo.exists(uuid.uuid4()) is False


# ===========================================================================
# mark_reviewed
# ===========================================================================

class TestMarkReviewed:
    @pytest.mark.asyncio
    async def test_success(self, repo, db):
        # exists check returns True
        exists_result = MagicMock()
        exists_result.scalar.return_value = 1
        # update returns rowcount=1
        update_result = MagicMock()
        update_result.rowcount = 1
        db.execute = AsyncMock(side_effect=[exists_result, update_result])
        db.commit = AsyncMock()

        result = await repo.mark_reviewed(uuid.uuid4(), "admin1", correction="fix", category="wrong_data")
        assert result is True
        db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_found(self, repo, db):
        exists_result = MagicMock()
        exists_result.scalar.return_value = 0
        db.execute = AsyncMock(return_value=exists_result)
        result = await repo.mark_reviewed(uuid.uuid4(), "admin1")
        assert result is False

    @pytest.mark.asyncio
    async def test_db_error_rollback(self, repo, db):
        exists_result = MagicMock()
        exists_result.scalar.return_value = 1
        db.execute = AsyncMock(side_effect=[exists_result, Exception("fail")])
        db.rollback = AsyncMock()
        result = await repo.mark_reviewed(uuid.uuid4(), "admin1")
        assert result is False

    @pytest.mark.asyncio
    async def test_invalid_category_validated(self, repo, db):
        exists_result = MagicMock()
        exists_result.scalar.return_value = 1
        update_result = MagicMock()
        update_result.rowcount = 1
        db.execute = AsyncMock(side_effect=[exists_result, update_result])
        db.commit = AsyncMock()
        result = await repo.mark_reviewed(uuid.uuid4(), "admin1", category="INVALID_CAT")
        assert result is True  # still works, just category ignored


# ===========================================================================
# get_statistics
# ===========================================================================

class TestGetStatistics:
    @pytest.mark.asyncio
    async def test_basic_stats(self, repo, db):
        total_result = MagicMock(); total_result.scalar.return_value = 100
        unrev_result = MagicMock(); unrev_result.scalar.return_value = 60
        rev_result = MagicMock(); rev_result.scalar.return_value = 40
        cat_result = MagicMock(); cat_result.all.return_value = [("wrong_data", 30), (None, 70)]
        db.execute = AsyncMock(side_effect=[total_result, unrev_result, rev_result, cat_result])

        stats = await repo.get_statistics()
        assert stats["total_reports"] == 100
        assert stats["unreviewed_count"] == 60
        assert stats["reviewed_count"] == 40
        assert stats["category_breakdown"]["wrong_data"] == 30
        assert stats["category_breakdown"][None] == 70

    @pytest.mark.asyncio
    async def test_with_tenant(self, repo, db):
        for _ in range(4):
            db.execute = AsyncMock(return_value=MagicMock(
                scalar=MagicMock(return_value=0),
                all=MagicMock(return_value=[])
            ))
        # Just verify it doesn't crash
        await repo.get_statistics(tenant_id="T1")

    @pytest.mark.asyncio
    async def test_db_error(self, repo, db):
        db.execute = AsyncMock(side_effect=Exception("fail"))
        stats = await repo.get_statistics()
        assert stats["total_reports"] == 0
        assert "error" in stats


# ===========================================================================
# export_for_training
# ===========================================================================

class TestExportForTraining:
    @pytest.mark.asyncio
    async def test_basic_export(self, repo, db):
        mock_report = MagicMock()
        mock_report.user_query = "q"
        mock_report.bot_response = "r"
        mock_report.correction = "c"
        mock_report.category = "wrong_data"
        mock_report.model = "gpt-4"
        result = MagicMock()
        result.scalars.return_value.all.return_value = [mock_report]
        db.execute = AsyncMock(return_value=result)

        data = await repo.export_for_training()
        assert len(data) == 1
        assert data[0]["instruction"] == "q"
        assert data[0]["correct_output"] == "c"

    @pytest.mark.asyncio
    async def test_db_error_returns_empty(self, repo, db):
        db.execute = AsyncMock(side_effect=Exception("fail"))
        assert await repo.export_for_training() == []


# ===========================================================================
# get_count
# ===========================================================================

class TestGetCount:
    @pytest.mark.asyncio
    async def test_total_count(self, repo, db):
        result = MagicMock(); result.scalar.return_value = 42
        db.execute = AsyncMock(return_value=result)
        assert await repo.get_count() == 42

    @pytest.mark.asyncio
    async def test_reviewed_only(self, repo, db):
        result = MagicMock(); result.scalar.return_value = 10
        db.execute = AsyncMock(return_value=result)
        assert await repo.get_count(reviewed_only=True) == 10

    @pytest.mark.asyncio
    async def test_unreviewed_only(self, repo, db):
        result = MagicMock(); result.scalar.return_value = 32
        db.execute = AsyncMock(return_value=result)
        assert await repo.get_count(unreviewed_only=True) == 32

    @pytest.mark.asyncio
    async def test_with_tenant(self, repo, db):
        result = MagicMock(); result.scalar.return_value = 5
        db.execute = AsyncMock(return_value=result)
        assert await repo.get_count(tenant_id="T1") == 5

    @pytest.mark.asyncio
    async def test_db_error(self, repo, db):
        db.execute = AsyncMock(side_effect=Exception("fail"))
        assert await repo.get_count() == 0


# ===========================================================================
# delete
# ===========================================================================

class TestDelete:
    @pytest.mark.asyncio
    async def test_success(self, repo, db):
        result = MagicMock(); result.rowcount = 1
        db.execute = AsyncMock(return_value=result)
        db.commit = AsyncMock()
        assert await repo.delete(uuid.uuid4()) is True

    @pytest.mark.asyncio
    async def test_not_found(self, repo, db):
        result = MagicMock(); result.rowcount = 0
        db.execute = AsyncMock(return_value=result)
        db.commit = AsyncMock()
        assert await repo.delete(uuid.uuid4()) is False

    @pytest.mark.asyncio
    async def test_db_error(self, repo, db):
        db.execute = AsyncMock(side_effect=Exception("fail"))
        db.rollback = AsyncMock()
        assert await repo.delete(uuid.uuid4()) is False
        db.rollback.assert_called_once()


# ===========================================================================
# bulk_mark_reviewed
# ===========================================================================

class TestBulkMarkReviewed:
    @pytest.mark.asyncio
    async def test_success(self, repo, db):
        result = MagicMock(); result.rowcount = 3
        db.execute = AsyncMock(return_value=result)
        db.commit = AsyncMock()
        ids = [uuid.uuid4() for _ in range(3)]
        assert await repo.bulk_mark_reviewed(ids, "admin1") == 3

    @pytest.mark.asyncio
    async def test_empty_list(self, repo, db):
        assert await repo.bulk_mark_reviewed([], "admin1") == 0

    @pytest.mark.asyncio
    async def test_with_category(self, repo, db):
        result = MagicMock(); result.rowcount = 2
        db.execute = AsyncMock(return_value=result)
        db.commit = AsyncMock()
        ids = [uuid.uuid4(), uuid.uuid4()]
        assert await repo.bulk_mark_reviewed(ids, "admin1", category="wrong_data") == 2

    @pytest.mark.asyncio
    async def test_db_error(self, repo, db):
        db.execute = AsyncMock(side_effect=Exception("fail"))
        db.rollback = AsyncMock()
        ids = [uuid.uuid4()]
        assert await repo.bulk_mark_reviewed(ids, "admin1") == 0
        db.rollback.assert_called_once()

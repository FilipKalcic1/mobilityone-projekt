"""
Tests for ConflictResolver service.

Covers:
- EditLock dataclass and expiration logic
- Pure logic: _can_auto_merge
- Dataclass construction: ConflictInfo, SaveResult, FieldChange, ChangeHistoryEntry
- Lock acquisition / release (with mocked Redis)
- Version control (_get_version, _increment_version)
- Conflict detection (save_with_conflict_check)
- Change history recording and retrieval
- Snapshot save / get
- Rollback to version
- Auto-merge logic
- Active editors
- Notification via pub/sub
- Expired lock cleanup
- Version history for admin UI
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4

from services.conflict_resolver import (
    ConflictResolver,
    ConflictType,
    LockStatus,
    EditLock,
    FieldChange,
    ConflictInfo,
    SaveResult,
    ChangeHistoryEntry,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_settings():
    """Patch get_settings so ConflictResolver.__init__ does not hit real config."""
    settings = MagicMock()
    settings.CONFLICT_LOCK_TTL_MINUTES = 30
    settings.CONFLICT_SNAPSHOT_TTL_DAYS = 90
    with patch("config.get_settings", return_value=settings):
        yield settings


@pytest.fixture
def mock_redis():
    """Full async-mock Redis client."""
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.setex = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=True)
    redis.incr = AsyncMock(return_value=2)
    redis.lpush = AsyncMock(return_value=1)
    redis.ltrim = AsyncMock(return_value=True)
    redis.lrange = AsyncMock(return_value=[])
    redis.scan = AsyncMock(return_value=(0, []))
    redis.publish = AsyncMock(return_value=1)
    return redis


@pytest.fixture
def mock_db():
    """Mock async database session."""
    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def resolver(mock_settings, mock_db, mock_redis):
    """ConflictResolver with both Redis and DB mocked."""
    return ConflictResolver(db_session=mock_db, redis_client=mock_redis)


@pytest.fixture
def resolver_no_redis(mock_settings, mock_db):
    """ConflictResolver WITHOUT Redis (fallback behaviour)."""
    return ConflictResolver(db_session=mock_db, redis_client=None)


@pytest.fixture
def record_id():
    return str(uuid4())


@pytest.fixture
def admin_id():
    return "admin-alice"


@pytest.fixture
def other_admin_id():
    return "admin-bob"


def _make_lock(record_id, admin_id, *, expired=False, status=LockStatus.ACTIVE.value):
    """Helper to create an EditLock with controllable expiry."""
    now = datetime.now(timezone.utc)
    if expired:
        locked_at = (now - timedelta(hours=2)).isoformat()
        expires_at = (now - timedelta(hours=1)).isoformat()
    else:
        locked_at = now.isoformat()
        expires_at = (now + timedelta(minutes=30)).isoformat()
    return EditLock(
        record_id=record_id,
        admin_id=admin_id,
        locked_at=locked_at,
        expires_at=expires_at,
        version=1,
        status=status,
    )


def _make_history_entry(version, admin_id, changes_dict):
    """Helper to create a serialised ChangeHistoryEntry."""
    now = datetime.now(timezone.utc).isoformat()
    entry = ChangeHistoryEntry(
        version=version,
        admin_id=admin_id,
        timestamp=now,
        changes=changes_dict,
    )
    return json.dumps(asdict(entry))


# ============================================================================
# 1. ENUM VALUES
# ============================================================================

class TestEnums:
    def test_conflict_type_values(self):
        assert ConflictType.VERSION_MISMATCH == "version_mismatch"
        assert ConflictType.CONCURRENT_EDIT == "concurrent_edit"
        assert ConflictType.ALREADY_REVIEWED == "already_reviewed"
        assert ConflictType.DELETED == "deleted"

    def test_lock_status_values(self):
        assert LockStatus.ACTIVE == "active"
        assert LockStatus.EXPIRED == "expired"
        assert LockStatus.RELEASED == "released"
        assert LockStatus.STOLEN == "stolen"


# ============================================================================
# 2. DATACLASS CONSTRUCTION
# ============================================================================

class TestDataclasses:
    def test_edit_lock_not_expired(self, record_id, admin_id):
        lock = _make_lock(record_id, admin_id, expired=False)
        assert lock.is_expired() is False
        assert lock.status == LockStatus.ACTIVE.value

    def test_edit_lock_expired(self, record_id, admin_id):
        lock = _make_lock(record_id, admin_id, expired=True)
        assert lock.is_expired() is True

    def test_field_change_creation(self):
        fc = FieldChange(
            field_name="correction",
            old_value="old text",
            new_value="new text",
            changed_by="admin-1",
            changed_at=datetime.now(timezone.utc).isoformat(),
        )
        assert fc.field_name == "correction"
        assert fc.old_value == "old text"
        assert fc.new_value == "new text"

    def test_conflict_info_defaults(self):
        ci = ConflictInfo(
            conflict_type=ConflictType.VERSION_MISMATCH.value,
            your_changes={"a": 1},
            their_changes={"b": 2},
            their_admin_id="admin-x",
            their_timestamp=datetime.now(timezone.utc).isoformat(),
        )
        assert ci.suggested_resolution is None
        assert ci.can_auto_merge is False

    def test_save_result_success(self, record_id):
        sr = SaveResult(success=True, record_id=record_id, new_version=5)
        assert sr.success is True
        assert sr.has_conflict is False
        assert sr.conflict is None
        assert sr.error is None

    def test_save_result_with_conflict(self, record_id):
        conflict = ConflictInfo(
            conflict_type=ConflictType.VERSION_MISMATCH.value,
            your_changes={"correction": "a"},
            their_changes={"correction": "b"},
            their_admin_id="admin-z",
            their_timestamp=datetime.now(timezone.utc).isoformat(),
        )
        sr = SaveResult(
            success=False,
            record_id=record_id,
            has_conflict=True,
            conflict=conflict,
            error="Record was modified",
        )
        assert sr.has_conflict is True
        assert sr.conflict.conflict_type == "version_mismatch"

    def test_change_history_entry_optional_ip(self):
        entry = ChangeHistoryEntry(
            version=3,
            admin_id="admin-1",
            timestamp=datetime.now(timezone.utc).isoformat(),
            changes={"field": {"old_value": "a", "new_value": "b"}},
        )
        assert entry.ip_address is None


# ============================================================================
# 3. PURE LOGIC: _can_auto_merge
# ============================================================================

class TestCanAutoMerge:
    def test_no_overlap_can_merge(self, resolver):
        assert resolver._can_auto_merge({"a": 1}, {"b": 2}) is True

    def test_overlap_cannot_merge(self, resolver):
        assert resolver._can_auto_merge({"a": 1}, {"a": 2}) is False

    def test_partial_overlap_cannot_merge(self, resolver):
        assert resolver._can_auto_merge({"a": 1, "b": 2}, {"b": 3, "c": 4}) is False

    def test_empty_changes_can_merge(self, resolver):
        assert resolver._can_auto_merge({}, {"a": 1}) is True
        assert resolver._can_auto_merge({"a": 1}, {}) is True
        assert resolver._can_auto_merge({}, {}) is True


# ============================================================================
# 4. INIT WITHOUT REDIS
# ============================================================================

class TestInitNoRedis:
    def test_no_redis_warning(self, resolver_no_redis):
        assert resolver_no_redis.redis is None

    @pytest.mark.asyncio
    async def test_get_version_no_redis(self, resolver_no_redis, record_id):
        version = await resolver_no_redis._get_version(record_id)
        assert version == 1

    @pytest.mark.asyncio
    async def test_increment_version_no_redis(self, resolver_no_redis, record_id):
        version = await resolver_no_redis._increment_version(record_id)
        assert version == 1

    @pytest.mark.asyncio
    async def test_release_lock_no_redis(self, resolver_no_redis, record_id, admin_id):
        result = await resolver_no_redis.release_lock(record_id, admin_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_cleanup_expired_locks_no_redis(self, resolver_no_redis):
        cleaned = await resolver_no_redis.cleanup_expired_locks()
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_get_change_history_no_redis(self, resolver_no_redis, record_id):
        history = await resolver_no_redis._get_change_history(record_id)
        assert history == []

    @pytest.mark.asyncio
    async def test_notify_other_editors_no_redis(self, resolver_no_redis, record_id, admin_id):
        result = await resolver_no_redis.notify_other_editors(record_id, admin_id, "msg")
        assert result is False


# ============================================================================
# 5. VERSION CONTROL (with Redis)
# ============================================================================

class TestVersionControl:
    @pytest.mark.asyncio
    async def test_get_version_existing(self, resolver, mock_redis, record_id):
        mock_redis.get = AsyncMock(return_value=b"5")
        version = await resolver._get_version(record_id)
        assert version == 5

    @pytest.mark.asyncio
    async def test_get_version_new_record(self, resolver, mock_redis, record_id):
        mock_redis.get = AsyncMock(return_value=None)
        version = await resolver._get_version(record_id)
        assert version == 1
        mock_redis.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_increment_version(self, resolver, mock_redis, record_id):
        mock_redis.incr = AsyncMock(return_value=3)
        version = await resolver._increment_version(record_id)
        assert version == 3


# ============================================================================
# 6. LOCK ACQUISITION
# ============================================================================

class TestLockAcquisition:
    @pytest.mark.asyncio
    async def test_acquire_new_lock(self, resolver, mock_redis, record_id, admin_id):
        """No existing lock -> new lock created."""
        mock_redis.get = AsyncMock(return_value=None)
        new_lock, existing = await resolver.acquire_edit_lock(record_id, admin_id)
        assert new_lock.admin_id == admin_id
        assert new_lock.record_id == record_id
        assert existing is None

    @pytest.mark.asyncio
    async def test_acquire_own_lock_refreshes(self, resolver, mock_redis, record_id, admin_id):
        """Same admin re-acquires -> refresh, no conflict."""
        existing = _make_lock(record_id, admin_id)
        mock_redis.get = AsyncMock(side_effect=[
            json.dumps(asdict(existing)).encode(),  # first call: _get_lock
            None,  # second call: _get_version inside _create_lock
        ])
        new_lock, old = await resolver.acquire_edit_lock(record_id, admin_id)
        assert new_lock.admin_id == admin_id
        assert old is None

    @pytest.mark.asyncio
    async def test_acquire_lock_held_by_other_no_force(self, resolver, mock_redis, record_id, admin_id, other_admin_id):
        """Another admin holds lock, no force -> return existing lock."""
        existing = _make_lock(record_id, other_admin_id)
        mock_redis.get = AsyncMock(return_value=json.dumps(asdict(existing)).encode())
        returned_lock, conflict_lock = await resolver.acquire_edit_lock(record_id, admin_id, force=False)
        assert returned_lock.admin_id == other_admin_id
        assert conflict_lock is not None
        assert conflict_lock.admin_id == other_admin_id

    @pytest.mark.asyncio
    async def test_acquire_lock_held_by_other_with_force(self, resolver, mock_redis, record_id, admin_id, other_admin_id):
        """Force acquire steals lock from other admin."""
        existing = _make_lock(record_id, other_admin_id)
        mock_redis.get = AsyncMock(side_effect=[
            json.dumps(asdict(existing)).encode(),  # _get_lock in acquire_edit_lock
            None,  # _get_version in _create_lock
        ])
        new_lock, old_lock = await resolver.acquire_edit_lock(record_id, admin_id, force=True)
        assert new_lock.admin_id == admin_id
        assert old_lock.status == LockStatus.STOLEN.value

    @pytest.mark.asyncio
    async def test_acquire_expired_lock(self, resolver, mock_redis, record_id, admin_id, other_admin_id):
        """Expired lock falls through to new lock creation, but existing is still returned."""
        expired = _make_lock(record_id, other_admin_id, expired=True)
        mock_redis.get = AsyncMock(side_effect=[
            json.dumps(asdict(expired)).encode(),  # _get_lock in acquire_edit_lock
            None,  # _get_version in _create_lock
        ])
        new_lock, old = await resolver.acquire_edit_lock(record_id, admin_id)
        assert new_lock.admin_id == admin_id
        # The expired lock is returned but not treated as a conflict (force was not needed)
        # Because code checks `existing_lock and force` after creating new lock
        assert old == expired


# ============================================================================
# 7. LOCK RELEASE
# ============================================================================

class TestLockRelease:
    @pytest.mark.asyncio
    async def test_release_own_lock(self, resolver, mock_redis, record_id, admin_id):
        existing = _make_lock(record_id, admin_id)
        mock_redis.get = AsyncMock(return_value=json.dumps(asdict(existing)).encode())
        result = await resolver.release_lock(record_id, admin_id)
        assert result is True
        mock_redis.delete.assert_awaited()

    @pytest.mark.asyncio
    async def test_release_other_admin_lock_denied(self, resolver, mock_redis, record_id, admin_id, other_admin_id):
        existing = _make_lock(record_id, other_admin_id)
        mock_redis.get = AsyncMock(return_value=json.dumps(asdict(existing)).encode())
        result = await resolver.release_lock(record_id, admin_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_release_expired_lock_by_anyone(self, resolver, mock_redis, record_id, admin_id, other_admin_id):
        expired = _make_lock(record_id, other_admin_id, expired=True)
        mock_redis.get = AsyncMock(return_value=json.dumps(asdict(expired)).encode())
        result = await resolver.release_lock(record_id, admin_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_release_nonexistent_lock(self, resolver, mock_redis, record_id, admin_id):
        mock_redis.get = AsyncMock(return_value=None)
        result = await resolver.release_lock(record_id, admin_id)
        assert result is True


# ============================================================================
# 8. SAVE WITH CONFLICT CHECK
# ============================================================================

class TestSaveWithConflictCheck:
    @pytest.mark.asyncio
    async def test_version_mismatch_returns_conflict(self, resolver, mock_redis, record_id, admin_id):
        """When current version != expected, a conflict is returned."""
        mock_redis.get = AsyncMock(return_value=b"3")
        mock_redis.lrange = AsyncMock(return_value=[])
        result = await resolver.save_with_conflict_check(
            record_id, {"correction": "new"}, expected_version=1, admin_id=admin_id,
        )
        assert result.success is False
        assert result.has_conflict is True
        assert result.conflict is not None
        assert result.conflict.conflict_type == ConflictType.VERSION_MISMATCH.value

    @pytest.mark.asyncio
    async def test_successful_save(self, resolver, mock_redis, mock_db, record_id, admin_id):
        """Happy path: version matches, save succeeds."""
        version_key = f"record_version:{record_id}"
        lock_key = f"edit_lock:{record_id}"

        async def redis_get_side_effect(key):
            if key == version_key:
                return b"1"
            if key == lock_key:
                return None  # release_lock -> _get_lock -> no lock found
            return None

        mock_redis.get = AsyncMock(side_effect=redis_get_side_effect)
        mock_redis.incr = AsyncMock(return_value=2)

        # Mock DB to return a record for _get_current_record_values and _apply_changes
        mock_record = MagicMock()
        mock_record.correction = "old"
        mock_record.category = "cat"
        mock_record.reviewed = False
        mock_record.reviewed_by = None
        mock_record.reviewed_at = None
        mock_scalar = MagicMock()
        mock_scalar.scalar_one_or_none = MagicMock(return_value=mock_record)
        mock_db.execute = AsyncMock(return_value=mock_scalar)

        with patch("services.conflict_resolver.HallucinationReport", create=True):
            with patch("services.conflict_resolver.select", create=True) as mock_select, \
                 patch("services.conflict_resolver.update", create=True):
                result = await resolver.save_with_conflict_check(
                    record_id, {"correction": "new text"}, expected_version=1,
                    admin_id=admin_id, ip_address="127.0.0.1",
                )

        assert result.success is True
        assert result.new_version == 2
        assert result.has_conflict is False

    @pytest.mark.asyncio
    async def test_save_exception_returns_error(self, resolver, mock_redis, mock_db, record_id, admin_id):
        """When DB raises, SaveResult.error is set."""
        mock_redis.get = AsyncMock(return_value=b"1")
        mock_db.execute = AsyncMock(side_effect=Exception("DB down"))

        with patch("services.conflict_resolver.HallucinationReport", create=True), \
             patch("services.conflict_resolver.select", create=True):
            result = await resolver.save_with_conflict_check(
                record_id, {"correction": "x"}, expected_version=1, admin_id=admin_id,
            )

        assert result.success is False
        assert result.error is not None
        assert "DB down" in result.error


# ============================================================================
# 9. CHANGE HISTORY
# ============================================================================

class TestChangeHistory:
    @pytest.mark.asyncio
    async def test_record_change_history(self, resolver, mock_redis, record_id, admin_id):
        await resolver._record_change_history(
            record_id, 2, {"correction": "new"}, admin_id, "10.0.0.1",
        )
        mock_redis.lpush.assert_awaited_once()
        mock_redis.ltrim.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_change_history(self, resolver, mock_redis, record_id, admin_id):
        entry_json = _make_history_entry(2, admin_id, {"correction": {"old_value": "a", "new_value": "b"}})
        mock_redis.lrange = AsyncMock(return_value=[entry_json.encode()])

        history = await resolver._get_change_history(record_id, limit=10)
        assert len(history) == 1
        assert history[0].version == 2
        assert history[0].admin_id == admin_id

    @pytest.mark.asyncio
    async def test_get_changes_since_version(self, resolver, mock_redis, record_id, admin_id):
        entries = [
            _make_history_entry(1, admin_id, {"field1": {"old_value": None, "new_value": "x"}}).encode(),
            _make_history_entry(3, admin_id, {"field2": {"old_value": None, "new_value": "y"}}).encode(),
        ]
        mock_redis.lrange = AsyncMock(return_value=entries)

        changes = await resolver._get_changes_since_version(record_id, since_version=2)
        assert "field2" in changes
        assert "field1" not in changes


# ============================================================================
# 10. SNAPSHOTS
# ============================================================================

class TestSnapshots:
    @pytest.mark.asyncio
    async def test_save_full_snapshot(self, resolver, mock_redis, record_id):
        await resolver._save_full_snapshot(record_id, 1, {"correction": "text"})
        mock_redis.setex.assert_awaited_once()
        call_args = mock_redis.setex.call_args
        assert f"snapshot:{record_id}:v1" == call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_snapshot_found(self, resolver, mock_redis, record_id):
        payload = json.dumps({"version": 1, "timestamp": "2025-01-01T00:00:00", "data": {"correction": "v1"}})
        mock_redis.get = AsyncMock(return_value=payload.encode())

        result = await resolver._get_snapshot(record_id, version=1)
        assert result is not None
        assert result["data"]["correction"] == "v1"

    @pytest.mark.asyncio
    async def test_get_snapshot_not_found(self, resolver, mock_redis, record_id):
        mock_redis.get = AsyncMock(return_value=None)
        result = await resolver._get_snapshot(record_id, version=99)
        assert result is None

    @pytest.mark.asyncio
    async def test_save_snapshot_serializes_datetime(self, resolver, mock_redis, record_id):
        dt = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        await resolver._save_full_snapshot(record_id, 2, {"reviewed_at": dt})
        call_args = mock_redis.setex.call_args
        stored = json.loads(call_args[0][2])
        assert stored["data"]["reviewed_at"] == dt.isoformat()


# ============================================================================
# 11. ROLLBACK
# ============================================================================

class TestRollback:
    @pytest.mark.asyncio
    async def test_rollback_invalid_version(self, resolver, record_id, admin_id):
        result = await resolver.rollback_to_version(record_id, 0, admin_id)
        assert result.success is False
        assert "Must be >= 1" in result.error

    @pytest.mark.asyncio
    async def test_rollback_no_snapshot(self, resolver, mock_redis, record_id, admin_id):
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.lrange = AsyncMock(return_value=[])

        result = await resolver.rollback_to_version(record_id, 5, admin_id)
        assert result.success is False
        assert "no snapshot or history available" in result.error

    @pytest.mark.asyncio
    async def test_rollback_with_snapshot(self, resolver, mock_redis, mock_db, record_id, admin_id):
        snapshot_data = json.dumps({
            "version": 2,
            "timestamp": "2025-01-01T00:00:00",
            "data": {"correction": "old_value"},
        })

        call_count = 0

        async def get_side_effect(key):
            nonlocal call_count
            call_count += 1
            if "snapshot:" in key:
                return snapshot_data.encode()
            # version key
            return b"5"

        mock_redis.get = AsyncMock(side_effect=get_side_effect)
        mock_redis.incr = AsyncMock(return_value=6)

        mock_record = MagicMock()
        mock_record.correction = "current_value"
        mock_scalar = MagicMock()
        mock_scalar.scalar_one_or_none = MagicMock(return_value=mock_record)
        mock_db.execute = AsyncMock(return_value=mock_scalar)

        with patch("services.conflict_resolver.HallucinationReport", create=True), \
             patch("services.conflict_resolver.select", create=True), \
             patch("services.conflict_resolver.update", create=True):
            result = await resolver.rollback_to_version(record_id, 2, admin_id, ip_address="10.0.0.1")

        assert result.success is True
        assert result.new_version == 6


# ============================================================================
# 12. RECONSTRUCT FROM HISTORY
# ============================================================================

class TestReconstructFromHistory:
    @pytest.mark.asyncio
    async def test_reconstruct_found(self, resolver, mock_redis, record_id, admin_id):
        changes = {
            "correction": {
                "field_name": "correction",
                "old_value": "original",
                "new_value": "modified",
                "changed_by": admin_id,
                "changed_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        entry = _make_history_entry(3, admin_id, changes)
        mock_redis.lrange = AsyncMock(return_value=[entry.encode()])

        result = await resolver._reconstruct_from_history(record_id, target_version=3)
        assert result is not None
        assert result["data"]["correction"] == "original"
        assert result["reconstructed"] is True

    @pytest.mark.asyncio
    async def test_reconstruct_version_not_found(self, resolver, mock_redis, record_id, admin_id):
        entry = _make_history_entry(5, admin_id, {"f": {"old_value": "x", "new_value": "y"}})
        mock_redis.lrange = AsyncMock(return_value=[entry.encode()])

        result = await resolver._reconstruct_from_history(record_id, target_version=2)
        assert result is None

    @pytest.mark.asyncio
    async def test_reconstruct_empty_history(self, resolver, mock_redis, record_id):
        mock_redis.lrange = AsyncMock(return_value=[])
        result = await resolver._reconstruct_from_history(record_id, target_version=1)
        assert result is None


# ============================================================================
# 13. AUTO MERGE
# ============================================================================

class TestAutoMerge:
    @pytest.mark.asyncio
    async def test_auto_merge_overlapping_fails(self, resolver, mock_redis, record_id, admin_id):
        mock_redis.get = AsyncMock(return_value=b"2")
        # Simulate history with overlapping field
        entry = _make_history_entry(2, "admin-bob", {
            "correction": {"old_value": None, "new_value": "theirs"}
        })
        mock_redis.lrange = AsyncMock(return_value=[entry.encode()])

        result = await resolver.auto_merge(
            record_id, {"correction": "mine"}, admin_id,
        )
        assert result.success is False
        assert "Cannot auto-merge" in result.error

    @pytest.mark.asyncio
    async def test_auto_merge_non_overlapping_succeeds(self, resolver, mock_redis, mock_db, record_id, admin_id):
        """Non-overlapping fields -> auto merge delegates to save_with_conflict_check."""
        version_key = f"record_version:{record_id}"
        lock_key = f"edit_lock:{record_id}"

        async def redis_get_side_effect(key):
            if key == version_key:
                return b"2"
            if key == lock_key:
                return None
            return None

        mock_redis.get = AsyncMock(side_effect=redis_get_side_effect)
        mock_redis.incr = AsyncMock(return_value=3)

        # History has category change, we're changing correction -> no overlap
        entry = _make_history_entry(2, "admin-bob", {
            "category": {"old_value": None, "new_value": "new_cat"}
        })
        mock_redis.lrange = AsyncMock(return_value=[entry.encode()])

        mock_record = MagicMock()
        mock_record.correction = "old"
        mock_record.category = "new_cat"
        mock_record.reviewed = False
        mock_record.reviewed_by = None
        mock_record.reviewed_at = None
        mock_scalar = MagicMock()
        mock_scalar.scalar_one_or_none = MagicMock(return_value=mock_record)
        mock_db.execute = AsyncMock(return_value=mock_scalar)

        with patch("services.conflict_resolver.HallucinationReport", create=True), \
             patch("services.conflict_resolver.select", create=True), \
             patch("services.conflict_resolver.update", create=True):
            result = await resolver.auto_merge(
                record_id, {"correction": "mine"}, admin_id,
            )

        assert result.success is True


# ============================================================================
# 14. ACTIVE EDITORS
# ============================================================================

class TestActiveEditors:
    @pytest.mark.asyncio
    async def test_active_editor_returned(self, resolver, mock_redis, record_id, admin_id):
        lock = _make_lock(record_id, admin_id)
        mock_redis.get = AsyncMock(return_value=json.dumps(asdict(lock)).encode())

        editors = await resolver.get_active_editors(record_id)
        assert len(editors) == 1
        assert editors[0]["admin_id"] == admin_id

    @pytest.mark.asyncio
    async def test_expired_lock_no_editors(self, resolver, mock_redis, record_id, admin_id):
        lock = _make_lock(record_id, admin_id, expired=True)
        mock_redis.get = AsyncMock(return_value=json.dumps(asdict(lock)).encode())

        editors = await resolver.get_active_editors(record_id)
        assert len(editors) == 0

    @pytest.mark.asyncio
    async def test_no_lock_no_editors(self, resolver, mock_redis, record_id):
        mock_redis.get = AsyncMock(return_value=None)
        editors = await resolver.get_active_editors(record_id)
        assert len(editors) == 0


# ============================================================================
# 15. NOTIFICATIONS
# ============================================================================

class TestNotifications:
    @pytest.mark.asyncio
    async def test_notify_publishes_to_redis(self, resolver, mock_redis, record_id, admin_id):
        mock_redis.publish = AsyncMock(return_value=2)
        result = await resolver.notify_other_editors(record_id, admin_id, "hello")
        assert result is True
        mock_redis.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_notify_no_subscribers(self, resolver, mock_redis, record_id, admin_id):
        mock_redis.publish = AsyncMock(return_value=0)
        result = await resolver.notify_other_editors(record_id, admin_id, "hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_notify_exception_returns_false(self, resolver, mock_redis, record_id, admin_id):
        mock_redis.publish = AsyncMock(side_effect=Exception("Redis error"))
        result = await resolver.notify_other_editors(record_id, admin_id, "hello")
        assert result is False


# ============================================================================
# 16. CLEANUP EXPIRED LOCKS
# ============================================================================

class TestCleanupExpiredLocks:
    @pytest.mark.asyncio
    async def test_cleanup_removes_expired(self, resolver, mock_redis, record_id, admin_id):
        expired = _make_lock(record_id, admin_id, expired=True)
        key = f"edit_lock:{record_id}"

        mock_redis.scan = AsyncMock(return_value=(0, [key.encode()]))
        mock_redis.get = AsyncMock(return_value=json.dumps(asdict(expired)).encode())

        cleaned = await resolver.cleanup_expired_locks()
        assert cleaned == 1
        mock_redis.delete.assert_awaited()

    @pytest.mark.asyncio
    async def test_cleanup_skips_active(self, resolver, mock_redis, record_id, admin_id):
        active = _make_lock(record_id, admin_id, expired=False)
        key = f"edit_lock:{record_id}"

        mock_redis.scan = AsyncMock(return_value=(0, [key.encode()]))
        mock_redis.get = AsyncMock(return_value=json.dumps(asdict(active)).encode())

        cleaned = await resolver.cleanup_expired_locks()
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_cleanup_no_keys(self, resolver, mock_redis):
        mock_redis.scan = AsyncMock(return_value=(0, []))
        cleaned = await resolver.cleanup_expired_locks()
        assert cleaned == 0


# ============================================================================
# 17. VERSION HISTORY (admin UI)
# ============================================================================

class TestVersionHistory:
    @pytest.mark.asyncio
    async def test_version_history_formatting(self, resolver, mock_redis, record_id, admin_id):
        changes = {
            "correction": {"old_value": "a", "new_value": "b"},
            "category": {"old_value": "c1", "new_value": "c2"},
        }
        entry = _make_history_entry(3, admin_id, changes)
        mock_redis.lrange = AsyncMock(return_value=[entry.encode()])

        history = await resolver.get_version_history(record_id, limit=20)
        assert len(history) == 1
        assert history[0]["version"] == 3
        assert history[0]["admin_id"] == admin_id
        assert history[0]["field_count"] == 2
        assert set(history[0]["fields_changed"]) == {"correction", "category"}

    @pytest.mark.asyncio
    async def test_version_history_empty(self, resolver, mock_redis, record_id):
        mock_redis.lrange = AsyncMock(return_value=[])
        history = await resolver.get_version_history(record_id)
        assert history == []


# ============================================================================
# 18. BUILD CONFLICT INFO
# ============================================================================

class TestBuildConflictInfo:
    @pytest.mark.asyncio
    async def test_build_conflict_auto_mergeable(self, resolver, mock_redis, record_id, admin_id):
        """Different fields -> can_auto_merge=True."""
        entry = _make_history_entry(2, "admin-bob", {
            "category": {"old_value": "a", "new_value": "b"}
        })
        mock_redis.lrange = AsyncMock(return_value=[entry.encode()])

        conflict = await resolver._build_conflict_info(
            record_id,
            your_changes={"correction": "new_val"},
            your_version=1,
            current_version=2,
            your_admin_id=admin_id,
        )
        assert conflict.can_auto_merge is True
        assert "auto-merged" in conflict.suggested_resolution

    @pytest.mark.asyncio
    async def test_build_conflict_overlapping_fields(self, resolver, mock_redis, record_id, admin_id):
        """Same field changed -> can_auto_merge=False, shows conflicting fields."""
        entry = _make_history_entry(2, "admin-bob", {
            "correction": {"old_value": "a", "new_value": "b"}
        })
        mock_redis.lrange = AsyncMock(return_value=[entry.encode()])

        conflict = await resolver._build_conflict_info(
            record_id,
            your_changes={"correction": "mine"},
            your_version=1,
            current_version=2,
            your_admin_id=admin_id,
        )
        assert conflict.can_auto_merge is False
        assert "correction" in conflict.suggested_resolution


# ============================================================================
# 19. EDGE CASES
# ============================================================================

class TestEdgeCases:
    def test_edit_lock_with_z_suffix_timestamp(self, record_id, admin_id):
        """EditLock.is_expired handles Z-suffix ISO timestamps."""
        now = datetime.now(timezone.utc)
        lock = EditLock(
            record_id=record_id,
            admin_id=admin_id,
            locked_at=now.isoformat(),
            expires_at=(now + timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
            version=1,
        )
        assert lock.is_expired() is False

    def test_edit_lock_with_naive_timestamp(self, record_id, admin_id):
        """EditLock.is_expired handles naive timestamps (no tzinfo)."""
        now = datetime.now(timezone.utc)
        naive_future = (now + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S")
        lock = EditLock(
            record_id=record_id,
            admin_id=admin_id,
            locked_at=now.isoformat(),
            expires_at=naive_future,
            version=1,
        )
        # Naive timestamp is treated as UTC in the code
        assert lock.is_expired() is False

    @pytest.mark.asyncio
    async def test_get_lock_bytes_decoded(self, resolver, mock_redis, record_id, admin_id):
        """_get_lock properly decodes bytes from Redis."""
        lock = _make_lock(record_id, admin_id)
        mock_redis.get = AsyncMock(return_value=json.dumps(asdict(lock)).encode("utf-8"))
        result = await resolver._get_lock(record_id)
        assert result is not None
        assert result.admin_id == admin_id

    @pytest.mark.asyncio
    async def test_get_lock_string_data(self, resolver, mock_redis, record_id, admin_id):
        """_get_lock handles string data (not bytes)."""
        lock = _make_lock(record_id, admin_id)
        mock_redis.get = AsyncMock(return_value=json.dumps(asdict(lock)))
        result = await resolver._get_lock(record_id)
        assert result is not None
        assert result.admin_id == admin_id

    @pytest.mark.asyncio
    async def test_get_version_bytes_decoded(self, resolver, mock_redis, record_id):
        """_get_version decodes bytes."""
        mock_redis.get = AsyncMock(return_value=b"42")
        version = await resolver._get_version(record_id)
        assert version == 42

    @pytest.mark.asyncio
    async def test_get_change_history_bytes_entries(self, resolver, mock_redis, record_id, admin_id):
        """_get_change_history decodes bytes entries from lrange."""
        entry = _make_history_entry(1, admin_id, {"f": {"old_value": None, "new_value": "v"}})
        mock_redis.lrange = AsyncMock(return_value=[entry.encode("utf-8")])
        history = await resolver._get_change_history(record_id)
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_snapshot_no_redis(self, resolver_no_redis, record_id):
        result = await resolver_no_redis._get_snapshot(record_id, 1)
        assert result is None

    @pytest.mark.asyncio
    async def test_record_change_history_no_redis(self, resolver_no_redis, record_id, admin_id):
        # Should simply return without error
        await resolver_no_redis._record_change_history(record_id, 1, {"x": 1}, admin_id)

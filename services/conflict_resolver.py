"""
Conflict Resolution Service
Version: 1.1 - Optimistic Locking for Admin Operations (FIXED)

PROBLEM SOLVED:
- Two admins edit same hallucination report simultaneously
- Last write wins = data loss
- No visibility into who changed what

SOLUTION:
- Optimistic locking with version numbers
- Conflict detection before save
- Merge suggestions for conflicts
- Full change history (audit trail)

PATTERN: Optimistic Concurrency Control (OCC)
1. Read record with version number
2. User makes changes
3. On save, check if version still matches
4. If mismatch -> CONFLICT -> show diff, ask user to resolve

USAGE:
    resolver = ConflictResolver(db_session)

    # Lock record for editing
    lock = await resolver.acquire_edit_lock(report_id, admin_id)

    # Save with conflict check
    result = await resolver.save_with_conflict_check(
        report_id,
        changes,
        expected_version=lock.version
    )

    if result.has_conflict:
        # Log conflict for debugging
        logger.warning(f"Conflict detected: {result.conflict_diff}")
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class ConflictType(str, Enum):
    """Types of conflicts."""
    VERSION_MISMATCH = "version_mismatch"
    CONCURRENT_EDIT = "concurrent_edit"
    ALREADY_REVIEWED = "already_reviewed"
    DELETED = "deleted"


class LockStatus(str, Enum):
    """Edit lock status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    RELEASED = "released"
    STOLEN = "stolen"


@dataclass
class EditLock:
    """Represents an edit lock on a record."""
    record_id: str
    admin_id: str
    locked_at: str
    expires_at: str
    version: int
    status: str = LockStatus.ACTIVE.value

    def is_expired(self) -> bool:
        # Use timezone-aware comparison
        expires = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > expires


@dataclass
class FieldChange:
    """Represents a change to a single field."""
    field_name: str
    old_value: Any
    new_value: Any
    changed_by: str
    changed_at: str


@dataclass
class ConflictInfo:
    """Information about a detected conflict."""
    conflict_type: str
    your_changes: Dict[str, Any]
    their_changes: Dict[str, Any]
    their_admin_id: str
    their_timestamp: str
    suggested_resolution: Optional[str] = None
    can_auto_merge: bool = False


@dataclass
class SaveResult:
    """Result of a save operation."""
    success: bool
    record_id: str
    new_version: int = 0
    has_conflict: bool = False
    conflict: Optional[ConflictInfo] = None
    error: Optional[str] = None


@dataclass
class ChangeHistoryEntry:
    """Entry in the change history."""
    version: int
    admin_id: str
    timestamp: str
    changes: Dict[str, Any]  # Dict of FieldChange as dicts
    ip_address: Optional[str] = None


class ConflictResolver:
    """
    Handles concurrent editing conflicts using optimistic locking.

    STRATEGY:
    1. Soft locks (advisory) - warn other admins someone is editing
    2. Version numbers - detect if record changed since read
    3. Change history - full audit trail for rollback
    4. Merge suggestions - help resolve conflicts automatically where possible
    """

    # Lock configuration - made configurable via environment
    LOCK_TTL_MINUTES = int(os.getenv("CONFLICT_LOCK_TTL_MINUTES", "30"))  # Increased from 15
    REDIS_LOCK_PREFIX = "edit_lock:"
    REDIS_VERSION_PREFIX = "record_version:"
    REDIS_HISTORY_PREFIX = "change_history:"

    # Snapshot configuration - 90 days for compliance
    SNAPSHOT_TTL_DAYS = int(os.getenv("CONFLICT_SNAPSHOT_TTL_DAYS", "90"))  # Increased from 30

    def __init__(
        self,
        db_session: AsyncSession,
        redis_client=None
    ):
        """
        Initialize conflict resolver.

        Args:
            db_session: Database session for persistence
            redis_client: Redis for distributed locking (required in production)
        """
        self.db = db_session
        self.redis = redis_client

        if not redis_client:
            logger.warning(
                "ConflictResolver initialized WITHOUT Redis - "
                "version control and locking will be limited!"
            )
        else:
            logger.info("ConflictResolver initialized with Redis")

    # =========================================================================
    # EDIT LOCKS
    # =========================================================================

    async def acquire_edit_lock(
        self,
        record_id: str,
        admin_id: str,
        force: bool = False
    ) -> Tuple[EditLock, Optional[EditLock]]:
        """
        Acquire an edit lock on a record.

        Args:
            record_id: ID of record to lock
            admin_id: Admin requesting the lock
            force: Force acquire even if locked by someone else

        Returns:
            Tuple of (new_lock, existing_lock_if_any)
        """
        # Check existing lock
        existing_lock = await self._get_lock(record_id)

        if existing_lock and not existing_lock.is_expired():
            if existing_lock.admin_id == admin_id:
                # Refresh own lock
                return await self._create_lock(record_id, admin_id), None
            elif not force:
                # Someone else has the lock
                logger.warning(
                    f"Lock conflict: {admin_id} tried to lock {record_id}, "
                    f"held by {existing_lock.admin_id}"
                )
                return existing_lock, existing_lock

        # Create new lock
        new_lock = await self._create_lock(record_id, admin_id)

        if existing_lock and force:
            # Notify original holder about stolen lock
            await self._notify_lock_stolen(record_id, existing_lock.admin_id, admin_id)
            logger.warning(
                f"Lock stolen: {admin_id} force-acquired lock on {record_id} "
                f"from {existing_lock.admin_id}"
            )
            existing_lock.status = LockStatus.STOLEN.value

        return new_lock, existing_lock

    async def _notify_lock_stolen(
        self,
        record_id: str,
        original_admin: str,
        new_admin: str
    ) -> None:
        """Notify original lock holder that their lock was stolen."""
        if not self.redis:
            return

        await self.notify_other_editors(
            record_id,
            new_admin,
            f"Lock taken by {new_admin}"
        )

    async def release_lock(self, record_id: str, admin_id: str) -> bool:
        """
        Release an edit lock.

        Args:
            record_id: Record to unlock
            admin_id: Admin releasing the lock (must match holder)

        Returns:
            True if released, False otherwise
        """
        if not self.redis:
            return True

        lock_key = f"{self.REDIS_LOCK_PREFIX}{record_id}"

        existing = await self._get_lock(record_id)
        if not existing:
            return True  # No lock to release

        # Check if lock is expired (can be released by anyone)
        if existing.is_expired():
            await self.redis.delete(lock_key)
            logger.info(f"Expired lock released: {record_id}")
            return True

        # Only owner can release non-expired lock
        if existing.admin_id == admin_id:
            await self.redis.delete(lock_key)
            logger.info(f"Lock released: {record_id} by {admin_id}")
            return True

        logger.warning(
            f"Cannot release lock: {admin_id} is not owner of {record_id} "
            f"(owned by {existing.admin_id})"
        )
        return False

    async def _get_lock(self, record_id: str) -> Optional[EditLock]:
        """Get existing lock if any."""
        if not self.redis:
            return None

        lock_key = f"{self.REDIS_LOCK_PREFIX}{record_id}"
        data = await self.redis.get(lock_key)

        if data:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            return EditLock(**json.loads(data))
        return None

    async def _create_lock(self, record_id: str, admin_id: str) -> EditLock:
        """Create a new edit lock."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=self.LOCK_TTL_MINUTES)

        # Get current version
        version = await self._get_version(record_id)

        lock = EditLock(
            record_id=record_id,
            admin_id=admin_id,
            locked_at=now.isoformat(),
            expires_at=expires.isoformat(),
            version=version,
            status=LockStatus.ACTIVE.value
        )

        if self.redis:
            lock_key = f"{self.REDIS_LOCK_PREFIX}{record_id}"
            await self.redis.setex(
                lock_key,
                self.LOCK_TTL_MINUTES * 60,
                json.dumps(asdict(lock))
            )

        return lock

    async def cleanup_expired_locks(self) -> int:
        """
        Clean up expired locks.

        Returns:
            Number of locks cleaned up
        """
        if not self.redis:
            return 0

        cleaned = 0
        # Note: In production, use SCAN instead of KEYS for large datasets
        pattern = f"{self.REDIS_LOCK_PREFIX}*"
        cursor = 0

        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                record_id = key.replace(self.REDIS_LOCK_PREFIX, "")
                lock = await self._get_lock(record_id)
                if lock and lock.is_expired():
                    await self.redis.delete(key)
                    cleaned += 1
                    logger.debug(f"Cleaned up expired lock: {key}")

            if cursor == 0:
                break

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired locks")

        return cleaned

    # =========================================================================
    # VERSION CONTROL
    # =========================================================================

    async def _get_version(self, record_id: str) -> int:
        """
        Get current version number for a record.

        Returns 1 as initial version when no Redis available or no version stored.
        """
        if not self.redis:
            # Without Redis, we can't track versions properly
            # Log warning and return 1
            logger.warning(
                f"Version control unavailable for {record_id}: no Redis connection"
            )
            return 1

        version_key = f"{self.REDIS_VERSION_PREFIX}{record_id}"
        version = await self.redis.get(version_key)

        if version:
            if isinstance(version, bytes):
                version = version.decode('utf-8')
            return int(version)

        # Initialize version for new records
        await self.redis.set(version_key, "1")
        return 1

    async def _increment_version(self, record_id: str) -> int:
        """
        Increment and return new version number.

        Returns new version, or 1 if Redis unavailable.
        """
        if not self.redis:
            logger.warning(
                f"Cannot increment version for {record_id}: no Redis connection"
            )
            return 1

        version_key = f"{self.REDIS_VERSION_PREFIX}{record_id}"
        new_version = await self.redis.incr(version_key)
        return int(new_version)

    # =========================================================================
    # CONFLICT DETECTION & RESOLUTION
    # =========================================================================

    async def save_with_conflict_check(
        self,
        record_id: str,
        changes: Dict[str, Any],
        expected_version: int,
        admin_id: str,
        ip_address: str = None
    ) -> SaveResult:
        """
        Save changes with optimistic locking conflict check.

        Args:
            record_id: ID of record to update
            changes: Dictionary of field changes
            expected_version: Version number when record was read
            admin_id: Admin making the change
            ip_address: Admin's IP for audit

        Returns:
            SaveResult with success/conflict info
        """
        current_version = await self._get_version(record_id)

        # Check for version mismatch (someone else saved)
        if current_version != expected_version:
            conflict = await self._build_conflict_info(
                record_id,
                changes,
                expected_version,
                current_version,
                admin_id
            )

            return SaveResult(
                success=False,
                record_id=record_id,
                has_conflict=True,
                conflict=conflict,
                error="Record was modified by another admin"
            )

        # No conflict - proceed with save
        try:
            # Fetch old values BEFORE applying changes (for audit trail)
            old_values = await self._get_current_record_values(
                record_id,
                list(changes.keys())
            )

            # Create snapshot of current state for rollback
            full_snapshot = await self._get_current_record_values(
                record_id,
                ["correction", "category", "reviewed", "reviewed_by", "reviewed_at"]
            )
            await self._save_full_snapshot(record_id, current_version, full_snapshot)

            # Save to database
            await self._apply_changes(record_id, changes, admin_id)

            # Increment version
            new_version = await self._increment_version(record_id)

            # Record history with old values
            await self._record_change_history(
                record_id,
                new_version,
                changes,
                admin_id,
                ip_address,
                old_values  # Pass old values for complete audit trail
            )

            # Release lock
            await self.release_lock(record_id, admin_id)

            logger.info(
                f"Save successful: {record_id} v{new_version} by {admin_id}"
            )

            return SaveResult(
                success=True,
                record_id=record_id,
                new_version=new_version
            )

        except Exception as e:
            logger.error(f"Save failed: {record_id} - {e}")
            return SaveResult(
                success=False,
                record_id=record_id,
                error=str(e)
            )

    async def _build_conflict_info(
        self,
        record_id: str,
        your_changes: Dict[str, Any],
        your_version: int,
        current_version: int,
        your_admin_id: str
    ) -> ConflictInfo:
        """Build detailed conflict information."""
        # Get the changes made by the other admin
        their_changes = await self._get_changes_since_version(
            record_id,
            your_version
        )

        # Find the admin who made conflicting changes
        their_admin_id = "unknown"
        their_timestamp = datetime.now(timezone.utc).isoformat()

        if their_changes:
            # Get from most recent change
            history = await self._get_change_history(record_id, limit=1)
            if history:
                their_admin_id = history[0].admin_id
                their_timestamp = history[0].timestamp

        # Check if changes can be auto-merged (non-overlapping fields)
        can_merge = self._can_auto_merge(your_changes, their_changes)

        # Generate suggestion
        suggestion = None
        if can_merge:
            suggestion = "Changes are to different fields - can be auto-merged"
        else:
            overlapping = set(your_changes.keys()) & set(their_changes.keys())
            suggestion = f"Conflicting fields: {', '.join(overlapping)}"

        return ConflictInfo(
            conflict_type=ConflictType.VERSION_MISMATCH.value,
            your_changes=your_changes,
            their_changes=their_changes,
            their_admin_id=their_admin_id,
            their_timestamp=their_timestamp,
            suggested_resolution=suggestion,
            can_auto_merge=can_merge
        )

    def _can_auto_merge(
        self,
        your_changes: Dict[str, Any],
        their_changes: Dict[str, Any]
    ) -> bool:
        """Check if changes can be automatically merged (no overlapping fields)."""
        your_fields = set(your_changes.keys())
        their_fields = set(their_changes.keys())
        return len(your_fields & their_fields) == 0

    async def auto_merge(
        self,
        record_id: str,
        your_changes: Dict[str, Any],
        admin_id: str,
        ip_address: str = None
    ) -> SaveResult:
        """
        Attempt automatic merge of non-conflicting changes.

        Only works if changes are to different fields.
        """
        # Get fresh version to check against
        current_version = await self._get_version(record_id)
        their_changes = await self._get_changes_since_version(record_id, 1)

        if not self._can_auto_merge(your_changes, their_changes):
            return SaveResult(
                success=False,
                record_id=record_id,
                error="Cannot auto-merge: overlapping field changes"
            )

        # Merge changes (their changes are already applied, so just apply yours)
        # Use current_version as expected - we're intentionally merging
        return await self.save_with_conflict_check(
            record_id,
            your_changes,
            current_version,  # Use current version, not their version
            admin_id,
            ip_address
        )

    # =========================================================================
    # CHANGE HISTORY
    # =========================================================================

    async def _record_change_history(
        self,
        record_id: str,
        version: int,
        changes: Dict[str, Any],
        admin_id: str,
        ip_address: str = None,
        old_values: Dict[str, Any] = None
    ) -> None:
        """Record change in history for audit trail with full old values."""
        if not self.redis:
            return

        history_key = f"{self.REDIS_HISTORY_PREFIX}{record_id}"
        old_values = old_values or {}
        now = datetime.now(timezone.utc).isoformat()

        entry = ChangeHistoryEntry(
            version=version,
            admin_id=admin_id,
            timestamp=now,
            changes={
                k: asdict(FieldChange(
                    field_name=k,
                    old_value=old_values.get(k),  # Store actual old value
                    new_value=v,
                    changed_by=admin_id,
                    changed_at=now
                ))
                for k, v in changes.items()
            },
            ip_address=ip_address
        )

        await self.redis.lpush(history_key, json.dumps(asdict(entry), default=str))
        await self.redis.ltrim(history_key, 0, 99)  # Keep last 100 changes

    async def _save_full_snapshot(
        self,
        record_id: str,
        version: int,
        snapshot: Dict[str, Any]
    ) -> None:
        """Save full record snapshot for rollback capability."""
        if not self.redis:
            return

        snapshot_key = f"snapshot:{record_id}:v{version}"
        # Serialize datetime objects
        serialized = {}
        for k, v in snapshot.items():
            if isinstance(v, datetime):
                serialized[k] = v.isoformat()
            else:
                serialized[k] = v

        await self.redis.setex(
            snapshot_key,
            86400 * self.SNAPSHOT_TTL_DAYS,  # Configurable TTL
            json.dumps({
                "version": version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": serialized
            })
        )

    async def _get_snapshot(
        self,
        record_id: str,
        version: int
    ) -> Optional[Dict[str, Any]]:
        """Get full snapshot for a specific version."""
        if not self.redis:
            return None

        snapshot_key = f"snapshot:{record_id}:v{version}"
        data = await self.redis.get(snapshot_key)
        if data:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            return json.loads(data)
        return None

    async def _get_current_record_values(
        self,
        record_id: str,
        fields: List[str]
    ) -> Dict[str, Any]:
        """Fetch current values from database for specified fields."""
        from models import HallucinationReport

        try:
            uuid_id = UUID(record_id)
        except ValueError:
            return {}

        result = await self.db.execute(
            select(HallucinationReport).where(HallucinationReport.id == uuid_id)
        )
        record = result.scalar_one_or_none()

        if not record:
            return {}

        values = {}
        for field_name in fields:
            if hasattr(record, field_name):
                value = getattr(record, field_name, None)
                # Serialize datetime for JSON compatibility
                if isinstance(value, datetime):
                    value = value.isoformat()
                values[field_name] = value

        return values

    async def _get_change_history(
        self,
        record_id: str,
        limit: int = 10
    ) -> List[ChangeHistoryEntry]:
        """Get change history for a record, sorted by timestamp (newest first)."""
        if not self.redis:
            return []

        history_key = f"{self.REDIS_HISTORY_PREFIX}{record_id}"
        entries = await self.redis.lrange(history_key, 0, limit - 1)

        result = []
        for e in entries:
            if isinstance(e, bytes):
                e = e.decode('utf-8')
            result.append(ChangeHistoryEntry(**json.loads(e)))

        # Already sorted by insertion order (newest first due to lpush)
        return result

    async def _get_changes_since_version(
        self,
        record_id: str,
        since_version: int
    ) -> Dict[str, Any]:
        """Get all field changes since a specific version, sorted by time."""
        history = await self._get_change_history(record_id, limit=50)

        # Sort history by version to ensure correct ordering
        sorted_history = sorted(history, key=lambda x: x.version)

        merged_changes = {}
        for entry in sorted_history:
            if entry.version > since_version:
                for field_name, change in entry.changes.items():
                    if isinstance(change, dict):
                        merged_changes[field_name] = change.get("new_value")
                    else:
                        merged_changes[field_name] = change

        return merged_changes

    async def rollback_to_version(
        self,
        record_id: str,
        target_version: int,
        admin_id: str,
        ip_address: str = None
    ) -> SaveResult:
        """
        Rollback a record to a previous version using stored snapshots.

        Args:
            record_id: Record to rollback
            target_version: Version to restore (must be >= 1)
            admin_id: Admin performing rollback
            ip_address: Admin's IP for audit

        Returns:
            SaveResult with restored data
        """
        # Validate target version
        if target_version < 1:
            return SaveResult(
                success=False,
                record_id=record_id,
                error=f"Invalid target version: {target_version}. Must be >= 1"
            )

        # Try to get snapshot for target version
        snapshot = await self._get_snapshot(record_id, target_version)

        if not snapshot:
            # Fallback: try to reconstruct from change history
            snapshot = await self._reconstruct_from_history(record_id, target_version)

        if not snapshot:
            return SaveResult(
                success=False,
                record_id=record_id,
                error=f"Cannot restore version {target_version}: no snapshot or history available"
            )

        try:
            # Get current values for audit
            current_values = await self._get_current_record_values(
                record_id,
                list(snapshot.get("data", {}).keys())
            )

            # Save current state as snapshot before rollback
            current_version = await self._get_version(record_id)
            await self._save_full_snapshot(record_id, current_version, current_values)

            # Apply rollback data
            restore_data = snapshot.get("data", {})
            await self._apply_changes(record_id, restore_data, admin_id)

            # Increment version
            new_version = await self._increment_version(record_id)

            # Record rollback in history
            await self._record_change_history(
                record_id,
                new_version,
                {"_rollback_to": target_version, **restore_data},
                admin_id,
                ip_address,
                current_values
            )

            logger.warning(
                f"ROLLBACK SUCCESS: {record_id} restored to v{target_version} "
                f"(now v{new_version}) by {admin_id}"
            )

            return SaveResult(
                success=True,
                record_id=record_id,
                new_version=new_version
            )

        except Exception as e:
            logger.error(f"Rollback failed: {record_id} - {e}")
            return SaveResult(
                success=False,
                record_id=record_id,
                error=f"Rollback failed: {str(e)}"
            )

    async def _reconstruct_from_history(
        self,
        record_id: str,
        target_version: int
    ) -> Optional[Dict[str, Any]]:
        """
        Reconstruct record state from change history if snapshot not available.

        Works backwards through history applying inverse changes.
        """
        history = await self._get_change_history(record_id, limit=100)

        if not history:
            return None

        # Find target entry in history
        target_entry = None
        for entry in history:
            if entry.version == target_version:
                target_entry = entry
                break

        if not target_entry:
            return None

        # Collect old values from the target version entry
        reconstructed = {}
        changes = target_entry.changes
        if isinstance(changes, dict):
            for field_name, change_data in changes.items():
                if isinstance(change_data, dict):
                    old_val = change_data.get("old_value")
                    if old_val is not None:
                        reconstructed[field_name] = old_val

        if reconstructed:
            return {
                "version": target_version,
                "timestamp": target_entry.timestamp,
                "data": reconstructed,
                "reconstructed": True
            }

        return None

    async def _apply_changes(
        self,
        record_id: str,
        changes: Dict[str, Any],
        admin_id: str
    ) -> None:
        """Apply changes to database."""
        from models import HallucinationReport

        try:
            uuid_id = UUID(record_id)
        except ValueError:
            raise ValueError(f"Invalid record ID: {record_id}")

        # Check if record exists
        check_result = await self.db.execute(
            select(HallucinationReport.id).where(HallucinationReport.id == uuid_id)
        )
        if not check_result.scalar_one_or_none():
            raise ValueError(f"Record not found: {record_id}")

        # Build update values - only set reviewed_by/reviewed_at if changes include review-related fields
        update_values = {}

        # Map allowed fields
        allowed_fields = {"correction", "category", "reviewed"}
        has_review_changes = False

        for field_name, value in changes.items():
            if field_name in allowed_fields:
                update_values[field_name] = value
                if field_name in ("correction", "category", "reviewed"):
                    has_review_changes = True

        # Only update reviewed_by and reviewed_at if we're making review-related changes
        if has_review_changes:
            update_values["reviewed_by"] = admin_id
            update_values["reviewed_at"] = datetime.now(timezone.utc)

        if update_values:
            await self.db.execute(
                update(HallucinationReport)
                .where(HallucinationReport.id == uuid_id)
                .values(**update_values)
            )
            await self.db.commit()

    # =========================================================================
    # ACTIVE EDITORS
    # =========================================================================

    async def get_active_editors(self, record_id: str) -> List[Dict]:
        """
        Get list of admins currently editing a record.

        Note: Currently only returns one editor (the lock holder).
        Future: Could track multiple viewers via Redis pub/sub.
        """
        lock = await self._get_lock(record_id)

        if lock and not lock.is_expired():
            return [{
                "admin_id": lock.admin_id,
                "since": lock.locked_at,
                "expires": lock.expires_at,
                "status": lock.status
            }]

        return []

    async def notify_other_editors(
        self,
        record_id: str,
        admin_id: str,
        message: str
    ) -> bool:
        """
        Send notification to other editors (via Redis pub/sub).

        Returns:
            True if message sent, False otherwise
        """
        if not self.redis:
            return False

        try:
            channel = f"editors:{record_id}"
            result = await self.redis.publish(channel, json.dumps({
                "from": admin_id,
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }))
            return result > 0  # Returns number of subscribers who received the message
        except Exception as e:
            logger.error(f"Failed to notify editors: {e}")
            return False

    async def get_version_history(
        self,
        record_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get version history for admin UI.

        Returns list of versions with timestamps and admins.
        """
        history = await self._get_change_history(record_id, limit=limit)

        return [
            {
                "version": entry.version,
                "admin_id": entry.admin_id,
                "timestamp": entry.timestamp,
                "field_count": len(entry.changes),
                "fields_changed": list(entry.changes.keys()),
                "ip_address": entry.ip_address
            }
            for entry in history
        ]

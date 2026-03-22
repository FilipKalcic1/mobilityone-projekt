"""
Conflict Resolution Models

Dataclasses and enums used by the conflict resolver service.
"""

from datetime import datetime, timezone
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


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
        try:
            expires = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) > expires
        except (ValueError, TypeError):
            return True  # Corrupted lock data → treat as expired for safety


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

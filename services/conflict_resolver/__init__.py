"""
Conflict Resolution Service Package

Re-exports all public names for backward compatibility.
Usage: from services.conflict_resolver import ConflictResolver
"""

from .models import (
    ConflictType,
    LockStatus,
    EditLock,
    FieldChange,
    ConflictInfo,
    SaveResult,
    ChangeHistoryEntry,
)
from .resolver import ConflictResolver

__all__ = [
    "ConflictType",
    "LockStatus",
    "EditLock",
    "FieldChange",
    "ConflictInfo",
    "SaveResult",
    "ChangeHistoryEntry",
    "ConflictResolver",
]

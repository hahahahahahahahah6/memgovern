"""Data models for memgovern."""

from dataclasses import dataclass, field
from typing import List, Optional


class MemoryStatus:
    ALIVE = "alive"            # live and readable
    PENDING = "pending"       # written but quarantined until a conflict is resolved
    SUPERSEDED = "superseded" # replaced by a newer version under the same key
    TOMBSTONED = "tombstoned" # deleted on purpose; kept for audit, never returned


class ConflictPolicy:
    OVERWRITE = "overwrite"   # newest write wins automatically; loser is superseded
    MANUAL = "manual"         # new write is quarantined as PENDING until resolve_conflict()
    KEEP_BOTH = "keep_both"   # both stay alive; conflict is recorded in the audit log


@dataclass
class Memory:
    id: int
    key: str
    text: str
    status: str
    created_at: float
    updated_at: float
    expires_at: Optional[float]
    source: str
    importance: float
    half_life: Optional[float]   # seconds; None -> store default
    tags: List[str] = field(default_factory=list)
    polarity: str = "fact"      # "fact" | "lesson" (negative knowledge) | "preference"
    conflict_id: Optional[int] = None
    deleted_reason: Optional[str] = None
    version: int = 1
    score: Optional[float] = None  # filled in by read()/query(): current decayed score

    def expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


@dataclass
class AuditEvent:
    id: int
    ts: float
    actor: str
    action: str
    key: Optional[str]
    memory_id: Optional[int]
    details: dict


@dataclass
class Conflict:
    id: int
    key: str
    old_id: Optional[int]
    new_id: int
    policy: str
    resolution: Optional[str]
    winner_id: Optional[int]
    created_at: float
    resolved_at: Optional[float]

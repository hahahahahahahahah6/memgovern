"""memgovern - a tiny memory governance layer for AI agents.

The write-and-delete side of agent memory: decay, tombstones,
conflict arbitration, expiry, and a full audit trail. Zero dependencies,
SQLite under the hood.
"""

from .models import ConflictPolicy, Memory, MemoryStatus
from .store import MemoryStore

__all__ = ["MemoryStore", "Memory", "MemoryStatus", "ConflictPolicy"]
__version__ = "0.3.0"

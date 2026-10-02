"""MemoryStore: the governance layer. SQLite, zero dependencies."""

import json
import secrets
import sqlite3
import time
from typing import Callable, List, Optional

from .decay import combined_score, decayed_score, jaccard
from .models import AuditEvent, Conflict, ConflictPolicy, Memory, MemoryStatus
from .trust import (
    BURST_LIMIT_DEFAULT,
    BURST_WINDOW_DEFAULT,
    HIGH_TRUST_FLOOR_DEFAULT,
    TRUST_HALF_LIFE_DEFAULT,
    TRUST_K_DEFAULT,
    TRUST_PRIOR_DEFAULT,
    TRUST_THRESHOLD_DEFAULT,
    decayed_toward_prior,
    find_injection_marker,
    smoothed_trust,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    key           TEXT NOT NULL,
    text          TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'alive',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    expires_at    REAL,
    source        TEXT NOT NULL DEFAULT 'agent',
    importance    REAL NOT NULL DEFAULT 0.5,
    half_life     REAL,
    tags          TEXT NOT NULL DEFAULT '[]',
    polarity      TEXT NOT NULL DEFAULT 'fact',
    conflict_id   INTEGER,
    deleted_reason TEXT,
    version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);

CREATE TABLE IF NOT EXISTS conflicts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT NOT NULL,
    old_id      INTEGER,
    new_id      INTEGER NOT NULL,
    policy      TEXT NOT NULL,
    resolution  TEXT,
    winner_id   INTEGER,
    created_at  REAL NOT NULL,
    resolved_at REAL
);

CREATE TABLE IF NOT EXISTS audit (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    actor     TEXT NOT NULL,
    action    TEXT NOT NULL,
    key       TEXT,
    memory_id INTEGER,
    details   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_key ON audit(key);

-- v0.2: per-source reliability ledger (IF NOT EXISTS migrates old DBs on open)
CREATE TABLE IF NOT EXISTS source_stats (
    source        TEXT PRIMARY KEY,
    writes        INTEGER NOT NULL DEFAULT 0,
    conflicts_won INTEGER NOT NULL DEFAULT 0,
    conflicts_lost INTEGER NOT NULL DEFAULT 0,
    tombstoned    INTEGER NOT NULL DEFAULT 0,
    quarantined   INTEGER NOT NULL DEFAULT 0,
    last_update   REAL NOT NULL
);

-- v0.2: recent-write log for burst tripwire detection (pruned on every write)
CREATE TABLE IF NOT EXISTS write_log (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    ts     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_write_log_source_ts ON write_log(source, ts);

-- v0.4: optimistic-concurrency reservations (compare-and-swap).
-- expected_version is the key's live version at reserve time; 0 means the
-- key was absent. IF NOT EXISTS migrates old DBs on open.
CREATE TABLE IF NOT EXISTS reservations (
    token            TEXT PRIMARY KEY,
    key              TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT 'agent',
    expected_version INTEGER NOT NULL,
    created_at       REAL NOT NULL,
    expires_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reservations_key ON reservations(key);
"""


class MemoryStore:
    """Govern what an agent remembers: write, forget, delete (tombstone), arbitrate.

    Args:
        path: SQLite file path, or ":memory:".
        half_life: default seconds for a memory's relevance to halve.
        forget_threshold: memories scoring below this are "forgotten"
            (kept in the DB, hidden from reads).
        conflict_policy: OVERWRITE | MANUAL | KEEP_BOTH (see models.ConflictPolicy).
        clock: callable returning epoch seconds; inject a fake clock in tests/demos.
        actor: label written into the audit log for this store's writes.
        trust_threshold: with arbitration="trust", auto-arbitrate only when the
            trust gap between the two sources is strictly greater than this.
        trust_prior: neutral trust for sources with no decided conflicts.
        trust_k: Bayesian smoothing strength (pseudo-observations at the prior).
        trust_half_life: seconds for an idle source's trust to decay halfway
            back toward the prior.
        burst_limit: tripwire -- more than this many writes from one source...
        burst_window: ...inside this many seconds quarantines the write.
        high_trust_floor: tripwire -- a brand-new source contradicting a key
            held by a source at or above this trust is quarantined.
    """

    def __init__(
        self,
        path: str = ":memory:",
        *,
        half_life: float = 30 * 86400,
        forget_threshold: float = 0.15,
        conflict_policy: str = ConflictPolicy.OVERWRITE,
        clock: Optional[Callable[[], float]] = None,
        actor: str = "agent",
        trust_threshold: float = TRUST_THRESHOLD_DEFAULT,
        trust_prior: float = TRUST_PRIOR_DEFAULT,
        trust_k: float = TRUST_K_DEFAULT,
        trust_half_life: float = TRUST_HALF_LIFE_DEFAULT,
        burst_limit: int = BURST_LIMIT_DEFAULT,
        burst_window: float = BURST_WINDOW_DEFAULT,
        high_trust_floor: float = HIGH_TRUST_FLOOR_DEFAULT,
    ):
        if not 0 <= trust_threshold <= 1:
            raise ValueError("trust_threshold must be in [0, 1]")
        if trust_k <= 0:
            raise ValueError("trust_k must be positive")
        if burst_limit < 1:
            raise ValueError("burst_limit must be >= 1")
        if burst_window <= 0:
            raise ValueError("burst_window must be positive")
        self.path = path
        self.half_life = half_life
        self.forget_threshold = forget_threshold
        self.conflict_policy = conflict_policy
        self._clock = clock or time.time
        self.actor = actor
        self.trust_threshold = trust_threshold
        self.trust_prior = trust_prior
        self.trust_k = trust_k
        self.trust_half_life = trust_half_life
        self.burst_limit = burst_limit
        self.burst_window = burst_window
        self.high_trust_floor = high_trust_floor
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        # True while a CAS-gated write holds the write lock: intermediate
        # commits are deferred until the whole check+write commits at once.
        self._cas_txn = False

    def _maybe_commit(self):
        """Commit unless inside a CAS transaction (deferred to the end)."""
        if not self._cas_txn:
            self._db.commit()

    # ------------------------------------------------------------------ helpers

    def _now(self) -> float:
        return self._clock()

    def _row_to_memory(self, row: sqlite3.Row, now: float) -> Memory:
        hl = row["half_life"] if row["half_life"] else self.half_life
        mem = Memory(
            id=row["id"],
            key=row["key"],
            text=row["text"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            source=row["source"],
            importance=row["importance"],
            half_life=row["half_life"],
            tags=json.loads(row["tags"]),
            polarity=row["polarity"],
            conflict_id=row["conflict_id"],
            deleted_reason=row["deleted_reason"],
            version=row["version"],
        )
        mem.score = decayed_score(mem.importance, now - mem.created_at, hl)
        return mem

    def _audit(self, action: str, key: Optional[str], memory_id: Optional[int], details: dict):
        self._db.execute(
            "INSERT INTO audit (ts, actor, action, key, memory_id, details) VALUES (?,?,?,?,?,?)",
            (self._now(), self.actor, action, key, memory_id, json.dumps(details)),
        )
        self._maybe_commit()

    @staticmethod
    def _clamp_importance(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    # ------------------------------------------------------------------ write

    def write(
        self,
        key: str,
        text: str,
        *,
        ttl: Optional[float] = None,
        source: str = "agent",
        importance: float = 0.5,
        half_life: Optional[float] = None,
        tags: Optional[List[str]] = None,
        polarity: str = "fact",
        also_check_similar: bool = False,
        arbitration: Optional[str] = None,
        reservation: Optional[str] = None,
    ) -> Memory:
        """Store a memory. Same-key contradictions trigger the conflict policy.

        arbitration: None | "manual" | "trust". Only meaningful when the
        store's conflict_policy is MANUAL (ignored otherwise). "manual"
        (the default) quarantines contradictions as PENDING. "trust"
        auto-arbitrates when the trust gap between the two sources exceeds
        trust_threshold, and quarantines otherwise. Poisoning tripwires
        always quarantine, in either mode -- a tripwire hit is never
        auto-accepted.

        reservation: a token from reserve(). The write applies only if the
        token is valid, unexpired, and the key's live version is unchanged
        since the reservation (compare-and-swap). A lost race returns a
        Memory with status "conflict" (never persisted) carrying the current
        live value as conflict_current; the attempted write is NOT applied.
        Without a reservation, writes behave exactly as before.
        """
        if arbitration not in (None, "manual", "trust"):
            raise ValueError("arbitration must be 'manual' or 'trust'")
        now = self._now()

        if reservation is None:
            return self._write_impl(
                key, text, now, ttl=ttl, source=source, importance=importance,
                half_life=half_life, tags=tags, polarity=polarity,
                also_check_similar=also_check_similar, arbitration=arbitration,
                reservation=None, cas_expected=None,
            )

        # CAS path. The pre-check fails fast on stale tokens, but the check
        # that matters happens under the write lock: BEGIN IMMEDIATE
        # serializes concurrent CAS writers, the check is re-run inside the
        # transaction, and every intermediate commit is deferred until the
        # single commit at the end. Exactly one concurrent writer can win.
        ok, _, reason = self._cas_check(key, reservation, now)
        if not ok:
            self._maybe_commit()  # flush any token cleanup from the check
            return self._cas_conflict_memory(key, text, source, now, reason)
        self._db.execute("BEGIN IMMEDIATE")
        self._cas_txn = True
        try:
            ok, cas_expected, reason = self._cas_check(key, reservation, now)
            if ok:
                mem = self._write_impl(
                    key, text, now, ttl=ttl, source=source, importance=importance,
                    half_life=half_life, tags=tags, polarity=polarity,
                    also_check_similar=also_check_similar, arbitration=arbitration,
                    reservation=reservation, cas_expected=cas_expected,
                )
                self._db.commit()
                return mem
            self._db.rollback()
        except Exception:
            self._db.rollback()
            raise
        finally:
            self._cas_txn = False
        self._maybe_commit()  # flush any token cleanup from the re-check
        return self._cas_conflict_memory(key, text, source, now, reason)

    def _write_impl(
        self,
        key: str,
        text: str,
        now: float,
        *,
        ttl: Optional[float],
        source: str,
        importance: float,
        half_life: Optional[float],
        tags: Optional[List[str]],
        polarity: str,
        also_check_similar: bool,
        arbitration: Optional[str],
        reservation: Optional[str],
        cas_expected: Optional[int],
    ) -> Memory:
        """The body of write(), after the CAS gate.

        When called with a reservation, this runs inside a single
        transaction holding the write lock (see write()); intermediate
        commits are deferred via _maybe_commit()."""
        # A stale token fails before anything is counted. The token is
        # consumed (released) only on success.
        importance = self._clamp_importance(importance)
        expires_at = now + ttl if ttl else None

        # Poisoning tripwires: structural, checked before anything is decided.
        trip_reason = self._check_content_tripwires(text, source, now)

        rivals = [
            self._row_to_memory(r, now)
            for r in self._db.execute(
                "SELECT * FROM memories WHERE key = ? AND status = ? ORDER BY id DESC",
                (key, MemoryStatus.ALIVE),
            )
        ]
        rivals = [m for m in rivals if not m.expired(now)]
        contradiction = bool(rivals and any(r.text != text for r in rivals))

        # Tripwire (a): a lower-trust source that never held this key
        # overwriting a high-trust holder's key. Keyed on the trust gap and
        # per-key history -- not on the source's global write count, which a
        # single harmless write elsewhere used to defeat. Checked before the
        # source ledger is touched.
        if contradiction and not trip_reason:
            trip_reason = self._check_holder_tripwire(key, source, rivals)

        # Ledger: every write counts, quarantined or not.
        self._log_write(source, now)
        self._touch_source(source, now, quarantined=trip_reason is not None)

        version = max([m.version for m in rivals], default=0) + 1
        status = MemoryStatus.ALIVE
        conflict_id = None
        trust_details = None

        if contradiction:
            if trip_reason:
                # A tripwire overrides the configured policy: never auto-accept.
                conflict_id = self._apply_conflict_policy(
                    key, rivals, now, tripwire_reason=trip_reason)
                status = MemoryStatus.PENDING
            elif self.conflict_policy == ConflictPolicy.MANUAL and arbitration == "trust":
                decision = self._trust_decision(rivals, source)
                if decision is None:
                    conflict_id = self._apply_conflict_policy(
                        key, rivals, now,
                        note=("trust arbitration fell back to quarantine: gap "
                              f"{abs(self.source_trust(rivals[0].source) - self.source_trust(source)):.3f}"
                              f" <= threshold {self.trust_threshold}"))
                    status = MemoryStatus.PENDING
                else:
                    conflict_id, status, trust_details = self._apply_trust_arbitration(
                        key, rivals, source, decision, now)
            else:
                conflict_id = self._apply_conflict_policy(key, rivals, now)
                if self.conflict_policy == ConflictPolicy.MANUAL:
                    status = MemoryStatus.PENDING
                # OVERWRITE / KEEP_BOTH: the new memory is born ALIVE
        elif trip_reason:
            # Benign write that tripped a tripwire: quarantined, no conflict row.
            status = MemoryStatus.PENDING

        cur = self._db.execute(
            """INSERT INTO memories
               (key, text, status, created_at, updated_at, expires_at, source,
                importance, half_life, tags, polarity, conflict_id, version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (key, text, status, now, now, expires_at, source, importance,
             half_life, json.dumps(tags or []), polarity, conflict_id, version),
        )
        mem_id = cur.lastrowid
        self._maybe_commit()

        if trust_details is not None:
            self._finalize_trust_arbitration(conflict_id, mem_id, key, trust_details, now)
        if trip_reason and not contradiction:
            self._audit("tripwire_flagged", key, mem_id, {
                "reason": trip_reason,
                "note": "write quarantined as PENDING; never auto-accepted",
            })

        mem = self._get(mem_id)

        self._audit("write", key, mem_id, {
            "version": version, "status": status, "importance": importance,
            "ttl": ttl, "polarity": polarity,
        })

        if reservation is not None:
            # CAS succeeded: consume the token so it can't be replayed.
            self._db.execute("DELETE FROM reservations WHERE token=?", (reservation,))
            self._maybe_commit()
            self._audit("cas_applied", key, mem_id, {
                "token_prefix": reservation[:8],
                "expected_version": cas_expected, "applied_version": version,
            })

        if also_check_similar:
            for other, sim in self._find_similar(key, text, exclude_id=mem_id):
                self._audit("conflict_hint", key, mem_id, {
                    "similar_key": other.key, "similar_id": other.id,
                    "jaccard": round(sim, 3),
                    "note": "token-overlap hint only; not auto-arbitrated",
                })
        return mem

    # ------------------------------------------------- trust & tripwires

    def _check_content_tripwires(self, text: str, source: str, now: float) -> Optional[str]:
        """Tripwires (b) and (c): injection markers and burst writes.

        Returns a reason string, or None. Never raises on content.
        """
        marker = find_injection_marker(text)
        if marker:
            return f"injection-marker:{marker}"
        n = self._writes_in_window(source, now)
        if n >= self.burst_limit:
            return f"burst:{n}-writes-in-{self.burst_window:g}s"
        return None

    def _writes_in_window(self, source: str, now: float) -> int:
        cutoff = now - self.burst_window
        return self._db.execute(
            "SELECT COUNT(*) c FROM write_log WHERE source=? AND ts > ?",
            (source, cutoff),
        ).fetchone()["c"]

    def _log_write(self, source: str, now: float):
        """Record the write for burst detection; prune entries outside the window."""
        self._db.execute(
            "DELETE FROM write_log WHERE ts <= ?", (now - self.burst_window,))
        self._db.execute(
            "INSERT INTO write_log (source, ts) VALUES (?,?)", (source, now))
        self._maybe_commit()

    def _check_holder_tripwire(self, key: str, source: str,
                               rivals: List[Memory]) -> Optional[str]:
        """Tripwire (a): quarantine when a source that never held `key`
        contradicts a high-trust holder's live value.

        Fires only when the most-trusted holder scores at or above
        high_trust_floor and the writer's trust is strictly below the
        holder's. An equally-or-more-trusted source, or one that held the
        key before, follows the normal conflict policy instead."""
        holder = max(rivals, key=lambda m: self.source_trust(m.source))
        holder_trust = self.source_trust(holder.source)
        if holder_trust < self.high_trust_floor:
            return None
        if self.source_trust(source) >= holder_trust:
            return None
        if self._source_ever_held_key(source, key):
            return None
        return "low-trust-source-vs-high-trust-holder"

    def _source_ever_held_key(self, source: str, key: str) -> bool:
        """Has `source` ever had an applied (non-quarantined) write on `key`?

        Quarantined (PENDING) writes never took effect, so they don't count:
        retrying a quarantined overwrite must not get easier."""
        row = self._db.execute(
            "SELECT 1 FROM memories WHERE source=? AND key=? AND status != ? LIMIT 1",
            (source, key, MemoryStatus.PENDING),
        ).fetchone()
        return row is not None

    def _ensure_source_row(self, source: str, now: float):
        self._db.execute(
            """INSERT OR IGNORE INTO source_stats
               (source, writes, conflicts_won, conflicts_lost, tombstoned, quarantined, last_update)
               VALUES (?,?,?,?,?,?,?)""",
            (source, 0, 0, 0, 0, 0, now))

    def _touch_source(self, source: str, now: float, quarantined: bool = False):
        """Count a write for the source ledger (quarantined writes count too)."""
        self._ensure_source_row(source, now)
        self._db.execute(
            "UPDATE source_stats SET writes = writes + 1, quarantined = quarantined + ?,"
            " last_update=? WHERE source=?",
            (1 if quarantined else 0, now, source),
        )
        self._maybe_commit()

    def _record_win(self, source: str, now: float):
        self._ensure_source_row(source, now)
        self._db.execute(
            "UPDATE source_stats SET conflicts_won = conflicts_won + 1, last_update=?"
            " WHERE source=?",
            (now, source),
        )
        self._maybe_commit()

    def _record_loss(self, source: str, now: float):
        self._ensure_source_row(source, now)
        self._db.execute(
            "UPDATE source_stats SET conflicts_lost = conflicts_lost + 1, last_update=?"
            " WHERE source=?",
            (now, source),
        )
        self._maybe_commit()

    def source_trust(self, source: str) -> float:
        """Current trust score for a source: Bayesian-smoothed conflict win
        rate, decayed toward the prior since the source's last activity.
        Unknown sources score exactly the prior (neutral)."""
        row = self._db.execute(
            "SELECT * FROM source_stats WHERE source=?", (source,)).fetchone()
        if row is None:
            return self.trust_prior
        raw = smoothed_trust(row["conflicts_won"], row["conflicts_lost"],
                             self.trust_prior, self.trust_k)
        age = max(0.0, self._now() - row["last_update"])
        return decayed_toward_prior(raw, self.trust_prior, age, self.trust_half_life)

    def trust_report(self):
        """Per-source ledger: [(source, score, writes, wins, losses)],
        highest trust first."""
        rows = self._db.execute("SELECT * FROM source_stats").fetchall()
        out = [
            (r["source"], self.source_trust(r["source"]),
             r["writes"], r["conflicts_won"], r["conflicts_lost"])
            for r in rows
        ]
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def _trust_decision(self, rivals: List[Memory], new_source: str) -> Optional[dict]:
        """Decide a MANUAL-policy contradiction by source trust.

        Returns a decision dict when the trust gap strictly exceeds
        trust_threshold, else None (caller falls back to quarantine).
        The holder is rivals[0] -- the latest live version of the key.
        """
        old_source = rivals[0].source
        old_score = self.source_trust(old_source)
        new_score = self.source_trust(new_source)
        gap = abs(new_score - old_score)
        if gap <= self.trust_threshold:
            return None
        if new_score > old_score:
            winner, winner_score, loser_score = "new", new_score, old_score
        else:
            winner, winner_score, loser_score = "old", old_score, new_score
        return {"winner": winner, "old_source": old_source, "new_source": new_source,
                "winner_score": winner_score, "loser_score": loser_score, "gap": gap}

    def _apply_trust_arbitration(self, key: str, rivals: List[Memory],
                                 new_source: str, decision: dict, now: float):
        """Auto-resolve a MANUAL-policy conflict by source trust.

        The winner's source gains a ledger win, the loser's a loss. The loser
        is never silently deleted: a losing new write is born SUPERSEDED, a
        losing holder is superseded like an overwrite. Returns
        (conflict_id, status_for_new_write, details); the conflict row's
        new_id is fixed up by _finalize_trust_arbitration after the insert.
        """
        winner = decision["winner"]
        cur = self._db.execute(
            """INSERT INTO conflicts (key, old_id, new_id, policy, resolution, winner_id, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (key, rivals[0].id, -1, ConflictPolicy.MANUAL, None, None, now),
        )
        conflict_id = cur.lastrowid
        if winner == "new":
            for r in rivals:
                self._db.execute(
                    "UPDATE memories SET status=?, updated_at=?, conflict_id=? WHERE id=?",
                    (MemoryStatus.SUPERSEDED, now, conflict_id, r.id),
                )
            status = MemoryStatus.ALIVE
            winner_source, loser_source = new_source, rivals[0].source
            winner_id, resolution = None, "trust-new-wins"
        else:
            for r in rivals:
                self._db.execute(
                    "UPDATE memories SET conflict_id=? WHERE id=?", (conflict_id, r.id))
            status = MemoryStatus.SUPERSEDED  # the losing write is born superseded
            winner_source, loser_source = rivals[0].source, new_source
            winner_id, resolution = rivals[0].id, "trust-old-wins"
        self._maybe_commit()
        return conflict_id, status, {
            "winner": winner, "resolution": resolution, "winner_id": winner_id,
            "winner_source": winner_source, "loser_source": loser_source,
            "winner_score": decision["winner_score"],
            "loser_score": decision["loser_score"],
            "gap": decision["gap"], "threshold": self.trust_threshold,
        }

    def _finalize_trust_arbitration(self, conflict_id: int, mem_id: int,
                                    key: str, details: dict, now: float):
        """Close out a trust auto-arbitration: fix up the conflict row,
        credit the ledger, audit-log both scores."""
        winner_id = details["winner_id"] if details["winner_id"] is not None else mem_id
        self._db.execute(
            "UPDATE conflicts SET new_id=?, resolution=?, winner_id=?, resolved_at=? WHERE id=?",
            (mem_id, details["resolution"], winner_id, now, conflict_id),
        )
        self._record_win(details["winner_source"], now)
        self._record_loss(details["loser_source"], now)
        self._maybe_commit()
        self._audit("trust_arbitrated", key, mem_id, {
            "conflict_id": conflict_id,
            "resolution": details["resolution"],
            "winner_source": details["winner_source"],
            "loser_source": details["loser_source"],
            "winner_score": round(details["winner_score"], 4),
            "loser_score": round(details["loser_score"], 4),
            "gap": round(details["gap"], 4),
            "threshold": details["threshold"],
            "note": "loser superseded, never silently deleted",
        })

    def release_quarantine(self, memory_id: int) -> Memory:
        """Release a tripwire-quarantined write (PENDING with no conflict row)
        back to ALIVE. Conflict-quarantined writes must go through
        resolve_conflict()."""
        now = self._now()
        mem = self._get_by_id(memory_id)
        if mem is None:
            raise KeyError(f"no memory #{memory_id}")
        if mem.status != MemoryStatus.PENDING:
            raise ValueError(f"memory #{memory_id} is not quarantined (status={mem.status})")
        if mem.conflict_id is not None:
            raise ValueError(f"memory #{memory_id} is conflict-quarantined;"
                             " use resolve_conflict()")
        self._db.execute(
            "UPDATE memories SET status=?, updated_at=? WHERE id=?",
            (MemoryStatus.ALIVE, now, memory_id),
        )
        self._maybe_commit()
        self._audit("quarantine_released", mem.key, memory_id,
                    {"note": "tripwire quarantine reviewed and released"})
        return self._get(memory_id)

    def review_quarantine(self, memory_id: int, decision: str,
                          reviewer: str = "human") -> str:
        """Human review of a quarantined write. decision: "accept" | "reject".

        Unifies the two quarantine shapes: a tripwire quarantine (PENDING,
        no conflict row) is applied or tombstoned; a conflict quarantine is
        resolved in the new or old version's favor. Returns the memory's new
        status. Raises KeyError/ValueError on bad ids, non-quarantined
        memories, or bad decisions. This is the human review step -- it is
        deliberately NOT exposed to the agent via MCP by default.
        """
        if decision not in ("accept", "reject"):
            raise ValueError("decision must be 'accept' or 'reject'")
        mem = self._get_by_id(memory_id)
        if mem is None:
            raise KeyError(f"no memory #{memory_id}")
        if mem.status != MemoryStatus.PENDING:
            raise ValueError(
                f"memory #{memory_id} is not quarantined (status={mem.status})")
        if mem.conflict_id is None:
            if decision == "accept":
                return self.release_quarantine(memory_id).status
            return self._reject_tripwire_quarantine(mem, reviewer)
        conflict = self.resolve_conflict(
            mem.conflict_id, winner="new" if decision == "accept" else "old")
        return (MemoryStatus.ALIVE if conflict.winner_id == memory_id
                else MemoryStatus.SUPERSEDED)

    def _reject_tripwire_quarantine(self, mem: Memory, reviewer: str) -> str:
        """Discard a tripwire-quarantined write: tombstone it (reviewed,
        never applied)."""
        now = self._now()
        self._db.execute(
            "UPDATE memories SET status=?, updated_at=?, deleted_reason=? WHERE id=?",
            (MemoryStatus.TOMBSTONED, now,
             f"tripwire quarantine rejected by {reviewer}", mem.id),
        )
        self._maybe_commit()
        self._audit("tripwire_rejected", mem.key, mem.id,
                     {"reviewed_by": reviewer,
                      "note": "quarantined write reviewed and discarded; never applied"})
        return MemoryStatus.TOMBSTONED

    def quarantine_reason(self, memory_id: int) -> Optional[str]:
        """Best-effort human reason why a PENDING memory is quarantined."""
        mem = self._get_by_id(memory_id)
        if mem is None or mem.status != MemoryStatus.PENDING:
            return None
        if mem.conflict_id is not None:
            # Was it a tripwire, or a plain contradiction quarantine?
            for ev in self.audit(action="tripwire_flagged", limit=50):
                details = ev.details if isinstance(ev.details, dict) else {}
                if details.get("conflict_id") == mem.conflict_id:
                    return "tripwire: " + str(details.get("reason", "unknown"))
            conflict = self.get_conflict(mem.conflict_id)
            policy = conflict.policy if conflict else "manual"
            return f"contradiction quarantined under {policy} policy"
        for ev in self.audit(action="tripwire_flagged", limit=50):
            if ev.memory_id == mem.id:
                details = ev.details if isinstance(ev.details, dict) else {}
                return "tripwire: " + str(details.get("reason", "unknown"))
        return "quarantined (reason not found in audit log)"

    # ------------------------------------------------- reservations (CAS)

    def _prune_reservations(self, now: float):
        """Lazily drop expired reservations. No audit: expiry is routine."""
        self._db.execute("DELETE FROM reservations WHERE expires_at <= ?", (now,))
        self._maybe_commit()

    def _key_version(self, key: str, now: float) -> int:
        """The version a fresh read() would see for `key`: highest version
        among live, non-expired rows. 0 when the key is effectively absent."""
        row = self._db.execute(
            """SELECT MAX(version) v FROM memories
               WHERE key=? AND status=? AND (expires_at IS NULL OR expires_at > ?)""",
            (key, MemoryStatus.ALIVE, now),
        ).fetchone()
        return row["v"] if row["v"] is not None else 0

    def reserve(self, key: str, source: str = "agent", ttl_seconds: float = 300) -> str:
        """Reserve `key` for a compare-and-swap write.

        Returns an opaque token bound to the key's current live version
        (0 when the key is absent, so create-if-absent is guarded too).
        Pass it to write(reservation=token): the write applies only if the
        key's version is unchanged since the reservation. Reservations expire
        after ttl_seconds (default 5 minutes); expiry is wall-clock.
        """
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        now = self._now()
        self._prune_reservations(now)
        token = secrets.token_urlsafe(24)
        expected = self._key_version(key, now)
        self._db.execute(
            """INSERT INTO reservations
               (token, key, source, expected_version, created_at, expires_at)
               VALUES (?,?,?,?,?,?)""",
            (token, key, source, expected, now, now + ttl_seconds),
        )
        self._maybe_commit()
        self._audit("reservation_issued", key, None, {
            "token_prefix": token[:8], "source": source,
            "expected_version": expected, "ttl_seconds": ttl_seconds,
        })
        return token

    def release_reservation(self, token: str) -> bool:
        """Release a reservation early. Unknown or already-released tokens are
        a silent no-op (fail-soft); returns True when a token was released."""
        now = self._now()
        self._prune_reservations(now)
        cur = self._db.execute("DELETE FROM reservations WHERE token=?", (token,))
        self._maybe_commit()
        if cur.rowcount:
            self._audit("reservation_released", None, None,
                        {"token_prefix": token[:8]})
            return True
        return False

    def _cas_check(self, key: str, token: str, now: float):
        """Validate a CAS reservation.

        Returns (ok, expected_version, failure_reason). Consumes nothing;
        the caller releases the token on success.
        """
        row = self._db.execute(
            "SELECT * FROM reservations WHERE token=?", (token,)).fetchone()
        if row is None:
            return False, None, "unknown-or-released-token"
        if row["expires_at"] <= now:
            self._db.execute("DELETE FROM reservations WHERE token=?", (token,))
            self._maybe_commit()
            return False, None, "reservation-expired"
        if row["key"] != key:
            return False, None, "token-bound-to-another-key"
        current = self._key_version(key, now)
        if current != row["expected_version"]:
            return False, row["expected_version"], "key-changed-since-reservation"
        return True, row["expected_version"], None

    def _cas_conflict_memory(self, key: str, text: str, source: str, now: float,
                             reason: str) -> Memory:
        """Build the unsaved CONFLICT result for a lost CAS race.

        Nothing is written to the DB. The live value the caller must re-read
        and merge against is attached as conflict_current (None if absent).
        """
        current = self.read(key)
        mem = Memory(
            id=-1, key=key, text=text, status=MemoryStatus.CONFLICT,
            created_at=now, updated_at=now, expires_at=None, source=source,
            importance=0.5, half_life=None, tags=[], polarity="fact",
            conflict_id=None, deleted_reason=None,
            version=self._key_version(key, now),
            conflict_current=current,
        )
        self._audit("cas_conflict", key, None, {
            "reason": reason, "attempted_source": source,
            "current_id": current.id if current else None,
            "current_version": mem.version,
            "note": "write NOT applied; re-read the key and retry with a fresh reservation",
        })
        return mem

    def _apply_conflict_policy(self, key: str, rivals: List[Memory], now: float,
                               tripwire_reason: Optional[str] = None,
                               note: Optional[str] = None) -> int:
        """Mark the conflict per policy; returns the conflict row id.

        If tripwire_reason is set, the write is quarantined (MANUAL-style)
        regardless of the configured policy -- a tripwire hit is never
        auto-accepted. The conflict row still records the configured policy.
        """
        policy = self.conflict_policy
        new_placeholder_id = -1  # filled in after insert; conflict row created first

        cur = self._db.execute(
            """INSERT INTO conflicts (key, old_id, new_id, policy, resolution, winner_id, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (key, rivals[0].id, new_placeholder_id, policy, None, None, now),
        )
        conflict_id = cur.lastrowid

        if tripwire_reason:
            # Quarantine path: overrides OVERWRITE/KEEP_BOTH too.
            for r in rivals:
                self._db.execute(
                    "UPDATE memories SET conflict_id=? WHERE id=?", (conflict_id, r.id)
                )
            self._audit("tripwire_flagged", key, None, {
                "conflict_id": conflict_id, "reason": tripwire_reason,
                "policy": policy,
                "note": "tripwire overrode policy; new write quarantined as PENDING,"
                        " never auto-accepted",
            })
        elif policy == ConflictPolicy.OVERWRITE:
            for r in rivals:
                self._db.execute(
                    "UPDATE memories SET status=?, updated_at=?, conflict_id=? WHERE id=?",
                    (MemoryStatus.SUPERSEDED, now, conflict_id, r.id),
                )
            self._db.execute(
                "UPDATE conflicts SET resolution=?, resolved_at=? WHERE id=?",
                ("newest-wins", now, conflict_id),
            )
            self._audit("conflict_resolved", key, None, {
                "conflict_id": conflict_id, "resolution": "newest-wins",
                "superseded_ids": [r.id for r in rivals],
            })
        elif policy == ConflictPolicy.MANUAL:
            for r in rivals:
                self._db.execute(
                    "UPDATE memories SET conflict_id=? WHERE id=?", (conflict_id, r.id)
                )
            self._audit("conflict_flagged", key, None, {
                "conflict_id": conflict_id, "policy": "manual",
                "note": note or "new write quarantined as PENDING; call resolve_conflict()",
            })
        else:  # KEEP_BOTH
            self._db.execute(
                "UPDATE conflicts SET resolution=?, resolved_at=? WHERE id=?",
                ("kept-both", now, conflict_id),
            )
            self._audit("conflict_flagged", key, None, {
                "conflict_id": conflict_id, "resolution": "kept-both",
            })
        self._maybe_commit()
        return conflict_id

    def resolve_conflict(self, conflict_id: int, winner: str = "new") -> Conflict:
        """Arbitrate a MANUAL-policy conflict. winner: 'new' | 'old'."""
        if winner not in ("new", "old"):
            raise ValueError("winner must be 'new' or 'old'")
        now = self._now()
        row = self._db.execute(
            "SELECT * FROM conflicts WHERE id=?", (conflict_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no conflict #{conflict_id}")
        if row["resolved_at"] is not None:
            raise ValueError(f"conflict #{conflict_id} already resolved")

        new = self._get_by_id(row["new_id"]) if row["new_id"] != -1 else self._latest_pending(row["key"], conflict_id)
        old = self._get_by_id(row["old_id"]) if row["old_id"] else None
        if new is None:
            raise ValueError(f"conflict #{conflict_id}: new memory missing")

        win, lose = (new, old) if winner == "new" else (old, new)
        self._db.execute(
            "UPDATE memories SET status=?, updated_at=? WHERE id=?",
            (MemoryStatus.ALIVE, now, win.id),
        )
        if lose:
            self._db.execute(
                "UPDATE memories SET status=?, updated_at=? WHERE id=?",
                (MemoryStatus.SUPERSEDED, now, lose.id),
            )
        self._db.execute(
            "UPDATE conflicts SET resolution=?, winner_id=?, resolved_at=?, new_id=? WHERE id=?",
            (f"manual-{winner}-wins", win.id, now, new.id, conflict_id),
        )
        self._maybe_commit()
        # Source-trust ledger: the winner's source earns a win, the loser's a loss.
        self._record_win(win.source, now)
        if lose:
            self._record_loss(lose.source, now)
        self._audit("conflict_resolved", row["key"], win.id, {
            "conflict_id": conflict_id, "resolution": f"manual-{winner}-wins",
            "winner_id": win.id, "loser_id": lose.id if lose else None,
        })
        return self.get_conflict(conflict_id)

    # ------------------------------------------------------------------ read

    def _get_by_id(self, mem_id: int) -> Optional[Memory]:
        row = self._db.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
        return self._row_to_memory(row, self._now()) if row else None

    def _get(self, mem_id: int) -> Memory:
        mem = self._get_by_id(mem_id)
        assert mem is not None
        return mem

    def _latest_pending(self, key: str, conflict_id: int) -> Optional[Memory]:
        row = self._db.execute(
            "SELECT * FROM memories WHERE key=? AND status=? AND conflict_id=? ORDER BY id DESC LIMIT 1",
            (key, MemoryStatus.PENDING, conflict_id),
        ).fetchone()
        return self._row_to_memory(row, self._now()) if row else None

    def read(
        self,
        key: str,
        *,
        include_expired: bool = False,
        include_forgotten: bool = False,
        include_pending: bool = False,
    ) -> Optional[Memory]:
        """Latest live version of `key`. Tombstoned/superseded/expired/forgotten
        memories are hidden unless explicitly requested."""
        now = self._now()
        statuses = [MemoryStatus.ALIVE] + ([MemoryStatus.PENDING] if include_pending else [])
        rows = self._db.execute(
            f"SELECT * FROM memories WHERE key=? AND status IN ({','.join('?'*len(statuses))}) "
            "ORDER BY id DESC",
            (key, *statuses),
        ).fetchall()
        for row in rows:
            mem = self._row_to_memory(row, now)
            if mem.expired(now) and not include_expired:
                continue
            if mem.score is not None and mem.score < self.forget_threshold and not include_forgotten:
                continue
            return mem
        return None

    def query(
        self,
        text: Optional[str] = None,
        *,
        limit: int = 10,
        include_expired: bool = False,
        include_forgotten: bool = False,
        include_pending: bool = False,
        min_score: float = 0.0,
    ) -> List[Memory]:
        """Ranked live memories. Rank = decay score, or decay+token-overlap if `text` given."""
        now = self._now()
        statuses = [MemoryStatus.ALIVE] + ([MemoryStatus.PENDING] if include_pending else [])
        rows = self._db.execute(
            f"SELECT * FROM memories WHERE status IN ({','.join('?'*len(statuses))})",
            (*statuses,),
        ).fetchall()

        out = []
        for row in rows:
            mem = self._row_to_memory(row, now)
            if mem.expired(now) and not include_expired:
                continue
            if mem.score < self.forget_threshold and not include_forgotten:
                continue
            rank = mem.score
            if text:
                rank = combined_score(mem.score, jaccard(text, mem.text))
            if rank >= min_score:
                out.append((rank, mem))
        out.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in out[:limit]]

    def _find_similar(self, key: str, text: str, exclude_id: int, threshold: float = 0.55):
        now = self._now()
        hits = []
        for row in self._db.execute(
            "SELECT * FROM memories WHERE status=? AND key != ?",
            (MemoryStatus.ALIVE, key),
        ):
            mem = self._row_to_memory(row, now)
            if mem.expired(now):
                continue
            sim = jaccard(text, mem.text)
            if sim >= threshold:
                hits.append((mem, sim))
        return sorted(hits, key=lambda t: t[1], reverse=True)

    # ------------------------------------------------------------------ delete

    def delete(self, key: str, reason: Optional[str] = None) -> int:
        """Tombstone the latest live version(s) of `key`. Never physically deleted."""
        now = self._now()
        rows = self._db.execute(
            "SELECT * FROM memories WHERE key=? AND status=?",
            (key, MemoryStatus.ALIVE),
        ).fetchall()
        live = [self._row_to_memory(r, now) for r in rows if not self._row_to_memory(r, now).expired(now)]
        for mem in live:
            self._db.execute(
                "UPDATE memories SET status=?, updated_at=?, deleted_reason=? WHERE id=?",
                (MemoryStatus.TOMBSTONED, now, reason, mem.id),
            )
            self._ensure_source_row(mem.source, now)
            self._db.execute(
                "UPDATE source_stats SET tombstoned = tombstoned + 1, last_update=?"
                " WHERE source=?",
                (now, mem.source),
            )
            self._audit("tombstone", key, mem.id, {"reason": reason, "version": mem.version})
        self._maybe_commit()
        return len(live)

    def restore(self, key: str) -> Optional[Memory]:
        """Undo a tombstone: revive the most recent tombstoned version of `key`."""
        now = self._now()
        row = self._db.execute(
            "SELECT * FROM memories WHERE key=? AND status=? ORDER BY id DESC LIMIT 1",
            (key, MemoryStatus.TOMBSTONED),
        ).fetchone()
        if row is None:
            return None
        self._db.execute(
            "UPDATE memories SET status=?, updated_at=?, deleted_reason=NULL WHERE id=?",
            (MemoryStatus.ALIVE, now, row["id"]),
        )
        self._maybe_commit()
        self._audit("restore", key, row["id"], {"note": "tombstone undone"})
        return self._get(row["id"])

    # ------------------------------------------------------------------ introspection

    def get_conflict(self, conflict_id: int) -> Optional[Conflict]:
        row = self._db.execute("SELECT * FROM conflicts WHERE id=?", (conflict_id,)).fetchone()
        if row is None:
            return None
        return Conflict(id=row["id"], key=row["key"], old_id=row["old_id"], new_id=row["new_id"],
                        policy=row["policy"], resolution=row["resolution"],
                        winner_id=row["winner_id"], created_at=row["created_at"],
                        resolved_at=row["resolved_at"])

    def pending_conflicts(self) -> List[Conflict]:
        return [
            Conflict(id=r["id"], key=r["key"], old_id=r["old_id"], new_id=r["new_id"],
                     policy=r["policy"], resolution=r["resolution"], winner_id=r["winner_id"],
                     created_at=r["created_at"], resolved_at=r["resolved_at"])
            for r in self._db.execute("SELECT * FROM conflicts WHERE resolved_at IS NULL")
        ]

    def audit(self, key: Optional[str] = None, action: Optional[str] = None, limit: int = 100) -> List[AuditEvent]:
        q, params = "SELECT * FROM audit", []
        clauses = []
        if key:
            clauses.append("key=?"); params.append(key)
        if action:
            clauses.append("action=?"); params.append(action)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [
            AuditEvent(id=r["id"], ts=r["ts"], actor=r["actor"], action=r["action"],
                       key=r["key"], memory_id=r["memory_id"], details=json.loads(r["details"]))
            for r in self._db.execute(q, params)
        ]

    def stats(self) -> dict:
        now = self._now()
        counts = {r["status"]: r["c"] for r in
                  self._db.execute("SELECT status, COUNT(*) c FROM memories GROUP BY status")}
        expired = self._db.execute(
            "SELECT COUNT(*) c FROM memories WHERE status=? AND expires_at IS NOT NULL AND expires_at <= ?",
            (MemoryStatus.ALIVE, now),
        ).fetchone()["c"]
        pending = self._db.execute(
            "SELECT COUNT(*) c FROM conflicts WHERE resolved_at IS NULL"
        ).fetchone()["c"]
        sources = self._db.execute(
            "SELECT COUNT(*) c FROM source_stats"
        ).fetchone()["c"]
        return {"by_status": counts, "expired_alive": expired,
                "pending_conflicts": pending, "audit_events": self._db.execute("SELECT COUNT(*) c FROM audit").fetchone()["c"],
                "sources_tracked": sources}

    def purge(self, older_than_seconds: float = 0) -> int:
        """Hard-delete tombstoned/superseded rows older than the cutoff. Audit log is kept."""
        cutoff = self._now() - older_than_seconds
        cur = self._db.execute(
            "DELETE FROM memories WHERE status IN (?,?) AND updated_at <= ?",
            (MemoryStatus.TOMBSTONED, MemoryStatus.SUPERSEDED, cutoff),
        )
        n = cur.rowcount
        self._maybe_commit()
        self._audit("purge", None, None, {"hard_deleted": n})
        return n

    def close(self):
        self._db.close()

#!/usr/bin/env python3
"""Tests for memgovern v0.4: multi-session write reservations (compare-and-swap).

Stdlib only, unittest.

Run: python3 tests/test_reservations.py  (or: python3 -m unittest discover -s tests)
"""

import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memgovern import ConflictPolicy, MemoryStatus, MemoryStore  # noqa: E402


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def fresh_store(**kwargs):
    clock = Clock()
    store = MemoryStore(":memory:", clock=clock, **kwargs)
    return store, clock


class ReservationTests(unittest.TestCase):
    def test_reserve_write_succeeds_and_bumps_version(self):
        store, _ = fresh_store()
        store.write("user/name", "hao", source="a")
        tok = store.reserve("user/name", source="a")
        self.assertIsInstance(tok, str)
        self.assertTrue(len(tok) >= 16)
        mem = store.write("user/name", "hao li", source="a", reservation=tok)
        self.assertEqual(mem.status, MemoryStatus.ALIVE)
        self.assertEqual(mem.text, "hao li")
        self.assertEqual(mem.version, 2)
        self.assertEqual(store.read("user/name").text, "hao li")

    def test_stale_token_loses_race_original_preserved(self):
        store, _ = fresh_store()
        store.write("k", "v1", source="a")
        tok_a = store.reserve("k", source="a")
        # Session B writes meanwhile (no reservation needed).
        store.write("k", "v2", source="b")
        # Session A's CAS write must fail.
        mem = store.write("k", "v1-stale", source="a", reservation=tok_a)
        self.assertEqual(mem.status, MemoryStatus.CONFLICT)
        self.assertEqual(mem.id, -1)
        # Original preserved; nothing applied.
        self.assertEqual(store.read("k").text, "v2")
        # Current value attached for the merge-and-retry loop.
        self.assertIsNotNone(mem.conflict_current)
        self.assertEqual(mem.conflict_current.text, "v2")

    def test_unknown_token_is_conflict_not_crash(self):
        store, _ = fresh_store()
        store.write("k", "v1", source="a")
        mem = store.write("k", "v2", source="a", reservation="bogus-token")
        self.assertEqual(mem.status, MemoryStatus.CONFLICT)
        self.assertEqual(store.read("k").text, "v1")

    def test_token_for_wrong_key_fails(self):
        store, _ = fresh_store()
        tok = store.reserve("k1", source="a")
        mem = store.write("k2", "x", source="a", reservation=tok)
        self.assertEqual(mem.status, MemoryStatus.CONFLICT)
        self.assertIsNone(store.read("k2"))

    def test_expired_reservation_conflicts(self):
        store, clock = fresh_store()
        store.write("k", "v1", source="a")
        tok = store.reserve("k", source="a", ttl_seconds=60)
        clock.advance(61)
        mem = store.write("k", "v2", source="a", reservation=tok)
        self.assertEqual(mem.status, MemoryStatus.CONFLICT)
        self.assertEqual(store.read("k").text, "v1")

    def test_reserve_on_absent_key_guards_create(self):
        store, _ = fresh_store()
        tok = store.reserve("new-key", source="a")
        mem = store.write("new-key", "created", source="a", reservation=tok)
        self.assertEqual(mem.status, MemoryStatus.ALIVE)
        self.assertEqual(mem.version, 1)
        # But if someone created it first, the reservation loses.
        tok2 = store.reserve("other", source="a")
        store.write("other", "sneaked-in", source="b")
        mem2 = store.write("other", "mine", source="a", reservation=tok2)
        self.assertEqual(mem2.status, MemoryStatus.CONFLICT)
        self.assertEqual(store.read("other").text, "sneaked-in")

    def test_token_consumed_on_success_no_replay(self):
        store, _ = fresh_store()
        store.write("k", "v1", source="a")
        tok = store.reserve("k", source="a")
        mem = store.write("k", "v2", source="a", reservation=tok)
        self.assertEqual(mem.status, MemoryStatus.ALIVE)
        # Replay the same token: key changed since the reservation, so conflict.
        mem2 = store.write("k", "v3", source="a", reservation=tok)
        self.assertEqual(mem2.status, MemoryStatus.CONFLICT)
        self.assertEqual(store.read("k").text, "v2")

    def test_release_reservation_idempotent(self):
        store, _ = fresh_store()
        tok = store.reserve("k", source="a")
        self.assertTrue(store.release_reservation(tok))
        self.assertFalse(store.release_reservation(tok))  # no-op, no error
        self.assertFalse(store.release_reservation("nope"))  # unknown, no error
        # A released token no longer applies.
        mem = store.write("k", "x", source="a", reservation=tok)
        self.assertEqual(mem.status, MemoryStatus.CONFLICT)

    def test_write_without_reservation_unchanged(self):
        # Backward compatibility: plain writes behave exactly as before.
        store, _ = fresh_store(conflict_policy=ConflictPolicy.OVERWRITE)
        m1 = store.write("k", "v1", source="a")
        m2 = store.write("k", "v2", source="a")
        self.assertEqual(m1.status, MemoryStatus.ALIVE)
        self.assertEqual(m2.status, MemoryStatus.ALIVE)
        self.assertEqual(store.read("k").text, "v2")
        self.assertEqual(m2.version, 2)

    def test_reserve_ttl_validation(self):
        store, _ = fresh_store()
        with self.assertRaises(ValueError):
            store.reserve("k", ttl_seconds=0)

    def test_audit_trail(self):
        store, _ = fresh_store()
        tok = store.reserve("k", source="a")
        store.write("k", "v1", source="a", reservation=tok)
        actions = [e.action for e in store.audit(key="k")]
        self.assertIn("reservation_issued", actions)
        self.assertIn("cas_applied", actions)
        # And a lost race is audited too.
        store2, _ = fresh_store()
        store2.write("k", "v1", source="a")
        t2 = store2.reserve("k", source="a")
        store2.write("k", "v2", source="b")
        store2.write("k", "stale", source="a", reservation=t2)
        actions2 = [e.action for e in store2.audit(key="k")]
        self.assertIn("cas_conflict", actions2)

    def test_cas_conflict_does_not_touch_trust_ledger(self):
        # A write that never applied must not count as a write for the source.
        store, _ = fresh_store()
        store.write("k", "v1", source="a")
        t = store.reserve("k", source="a")
        store.write("k", "v2", source="b")
        before = store.source_trust("a")
        store.write("k", "stale", source="a", reservation=t)
        self.assertEqual(store.source_trust("a"), before)


def _cas_race_worker(db_path, key, text, barrier, out_queue):
    """One racer: reserve, wait for all racers to hold a token, then write."""
    from memgovern import MemoryStore
    store = MemoryStore(db_path)
    try:
        token = store.reserve(key, source="racer")
        barrier.wait(timeout=30)
        mem = store.write(key, text, source="racer", reservation=token)
        out_queue.put(mem.status)
    except Exception as e:  # noqa: BLE001 -- report failures as statuses
        out_queue.put(f"ERROR: {e!r}")
    finally:
        store.close()


class ConcurrentCasTests(unittest.TestCase):
    """Bug 1 regression: _cas_check read the version and wrote later with no
    lock, so N concurrent processes could all pass the check and all go
    ALIVE. Exactly one writer may win per round."""

    ROUNDS = 5
    RACERS = 4

    def test_concurrent_cas_writes_exactly_one_winner(self):
        tmpd = tempfile.mkdtemp(prefix="memgovern-cas-race-")
        db = os.path.join(tmpd, "race.db")
        try:
            seed = MemoryStore(db)
            seed.write("k", "v0", source="seed")
            seed.close()
            ctx = mp.get_context("fork")
            for r in range(self.ROUNDS):
                barrier = ctx.Barrier(self.RACERS)
                out_queue = ctx.Queue()
                procs = [
                    ctx.Process(target=_cas_race_worker,
                                args=(db, "k", f"racer-{i}-round-{r}",
                                      barrier, out_queue))
                    for i in range(self.RACERS)
                ]
                for p in procs:
                    p.start()
                for p in procs:
                    p.join(60)
                self.assertTrue(all(p.exitcode == 0 for p in procs),
                                f"round {r}: worker crashed")
                statuses = [out_queue.get(timeout=10)
                            for _ in range(self.RACERS)]
                winners = [s for s in statuses
                           if s == MemoryStatus.ALIVE]
                self.assertEqual(len(winners), 1,
                                 f"round {r}: expected exactly 1 winner, "
                                 f"got statuses={statuses}")
                # And the key really has a single live row afterwards.
                check = MemoryStore(db)
                try:
                    n = check._db.execute(
                        "SELECT COUNT(*) c FROM memories WHERE key=? AND status=?",
                        ("k", MemoryStatus.ALIVE)).fetchone()["c"]
                finally:
                    check.close()
                self.assertEqual(n, 1,
                                 f"round {r}: {n} alive rows for one key")
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)


class McpReservationTests(unittest.TestCase):
    def _roundtrip(self, db_file, *messages):
        from memgovern import mcp_server
        import io
        stdin = io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n")
        stdout = io.StringIO()
        old = os.environ.get("MEMGOVERN_DB")
        os.environ["MEMGOVERN_DB"] = db_file
        try:
            mcp_server.serve(stdin=stdin, stdout=stdout)
        finally:
            if old is None:
                del os.environ["MEMGOVERN_DB"]
            else:
                os.environ["MEMGOVERN_DB"] = old
        return [json.loads(line) for line in stdout.getvalue().splitlines()]

    def test_mcp_reserve_write_roundtrip(self):
        db = "/tmp/test-res-cas.db"
        if os.path.exists(db):
            os.remove(db)
        resps = self._roundtrip(db,
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "memory_reserve",
                        "arguments": {"key": "user/name", "source": "a"}}})
        token = json.loads(resps[0]["result"]["content"][0]["text"])["token"]
        self.assertTrue(token)
        resps = self._roundtrip(db,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "memory_write",
                        "arguments": {"key": "user/name", "text": "hao",
                                      "source": "a", "reservation": token}}})
        out = json.loads(resps[0]["result"]["content"][0]["text"])
        self.assertEqual(out["status"], "alive")
        self.assertEqual(out["text"], "hao")
        # Stale token through MCP: conflict, current attached, nothing applied.
        resps = self._roundtrip(db,
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "memory_write",
                        "arguments": {"key": "user/name", "text": "mallory",
                                      "source": "b", "reservation": token}}})
        out = json.loads(resps[0]["result"]["content"][0]["text"])
        self.assertEqual(out["status"], "conflict")
        self.assertEqual(out["current"]["text"], "hao")
        self.assertIn("NOT applied", out["note"])
        os.remove(db)

    def test_mcp_tools_list_includes_reserve(self):
        db = "/tmp/test-res-cas2.db"
        if os.path.exists(db):
            os.remove(db)
        resps = self._roundtrip(db,
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = [t["name"] for t in resps[0]["result"]["tools"]]
        self.assertIn("memory_reserve", names)
        self.assertIn("memory_write", names)
        if os.path.exists(db):
            os.remove(db)


if __name__ == "__main__":
    unittest.main(verbosity=1)

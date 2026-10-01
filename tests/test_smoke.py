#!/usr/bin/env python3
"""Smoke tests for memgovern (stdlib only, unittest).

Covers the core API: write/read, TTL expiry, tombstone delete + restore,
conflict policies (overwrite/manual), decay scoring, and query ranking.
Run: python3 tests/test_smoke.py  (or: python3 -m unittest discover -s tests)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memgovern import (  # noqa: E402
    ConflictPolicy,
    MemoryStatus,
    MemoryStore,
    decay,
)


class Clock:
    """Controllable clock for deterministic TTL/decay tests."""

    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def fresh_store(**kwargs):
    clock = Clock()
    kwargs.setdefault("clock", clock)
    return MemoryStore(":memory:", **kwargs), clock


class TestWriteRead(unittest.TestCase):
    def test_write_then_read(self):
        store, _ = fresh_store()
        mem = store.write("user/name", "hao", importance=0.9)
        self.assertEqual(mem.key, "user/name")
        self.assertEqual(mem.status, MemoryStatus.ALIVE)
        got = store.read("user/name")
        self.assertIsNotNone(got)
        self.assertEqual(got.text, "hao")
        self.assertEqual(got.version, 1)
        store.close()

    def test_read_missing_returns_none(self):
        store, _ = fresh_store()
        self.assertIsNone(store.read("nope"))
        store.close()

    def test_ttl_expiry(self):
        store, clock = fresh_store()
        store.write("temp/code", "1234", ttl=60)
        self.assertIsNotNone(store.read("temp/code"))
        clock.advance(61)
        self.assertIsNone(store.read("temp/code"))
        # explicitly requested: still visible
        self.assertIsNotNone(store.read("temp/code", include_expired=True))
        store.close()


class TestTombstone(unittest.TestCase):
    def test_delete_hides_but_keeps_audit(self):
        store, _ = fresh_store()
        store.write("k", "v1")
        n = store.delete("k", reason="outdated")
        self.assertEqual(n, 1)
        self.assertIsNone(store.read("k"))
        # audit trail records the tombstone
        actions = [e.action for e in store.audit(key="k")]
        self.assertIn("tombstone", actions)
        store.close()

    def test_restore_revives(self):
        store, _ = fresh_store()
        store.write("k", "v1")
        store.delete("k")
        revived = store.restore("k")
        self.assertIsNotNone(revived)
        self.assertEqual(store.read("k").text, "v1")
        store.close()


class TestConflicts(unittest.TestCase):
    def test_overwrite_supersedes_old(self):
        store, _ = fresh_store(conflict_policy=ConflictPolicy.OVERWRITE)
        store.write("pref/theme", "dark")
        store.write("pref/theme", "light")
        got = store.read("pref/theme")
        self.assertEqual(got.text, "light")
        self.assertEqual(got.version, 2)
        self.assertEqual(got.status, MemoryStatus.ALIVE)
        store.close()

    def test_manual_quarantines_until_resolved(self):
        store, _ = fresh_store(conflict_policy=ConflictPolicy.MANUAL)
        store.write("pref/theme", "dark")
        new_mem = store.write("pref/theme", "light")
        self.assertEqual(new_mem.status, MemoryStatus.PENDING)
        # pending is hidden by default...
        self.assertEqual(store.read("pref/theme").text, "dark")
        # ...but visible on request
        self.assertEqual(
            store.read("pref/theme", include_pending=True).text, "light")
        pending = store.pending_conflicts()
        self.assertEqual(len(pending), 1)
        store.resolve_conflict(pending[0].id, winner="new")
        self.assertEqual(store.read("pref/theme").text, "light")
        self.assertEqual(len(store.pending_conflicts()), 0)
        store.close()


class TestDecayAndQuery(unittest.TestCase):
    def test_decay_halves_at_half_life(self):
        store, clock = fresh_store(half_life=100.0)
        store.write("a", "alpha", importance=1.0)
        clock.advance(100.0)
        mem = store.read("a")
        self.assertAlmostEqual(mem.score, 0.5, places=6)
        store.close()

    def test_query_ranks_by_relevance(self):
        store, _ = fresh_store()
        store.write("m1", "the user loves python programming", importance=0.5)
        store.write("m2", "buy milk and eggs tomorrow", importance=0.5)
        hits = store.query("python programming")
        self.assertGreaterEqual(len(hits), 2)
        self.assertEqual(hits[0].key, "m1")
        store.close()

    def test_forgotten_below_threshold_hidden(self):
        store, clock = fresh_store(half_life=10.0, forget_threshold=0.15)
        store.write("old", "ancient history", importance=0.5)
        clock.advance(10_000.0)  # decayed far below threshold
        self.assertIsNone(store.read("old"))
        self.assertIsNotNone(store.read("old", include_forgotten=True))
        store.close()


class TestDecayHelpers(unittest.TestCase):
    def test_decayed_score(self):
        self.assertAlmostEqual(decay.decayed_score(1.0, 0, 10.0), 1.0)
        self.assertAlmostEqual(decay.decayed_score(1.0, 10.0, 10.0), 0.5)
        self.assertAlmostEqual(decay.decayed_score(0.8, 20.0, 10.0), 0.2)

    def test_jaccard(self):
        self.assertEqual(decay.jaccard("hello world", "hello world"), 1.0)
        self.assertEqual(decay.jaccard("aaa bbb", "ccc ddd"), 0.0)
        self.assertEqual(decay.jaccard("", "something"), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

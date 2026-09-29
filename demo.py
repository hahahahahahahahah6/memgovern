"""memgovern demo in three acts: forgetting, tombstones, conflict arbitration.

Run:  python demo.py
"""

import time

from memgovern import ConflictPolicy, MemoryStore


class FakeClock:
    """Fast-forward time so the demo doesn't have to wait."""

    def __init__(self):
        self.t = time.time()

    def __call__(self):
        return self.t

    def advance(self, seconds: float):
        self.t += seconds
        print(f"   ... {seconds:>8.0f}s pass ...")


def banner(title: str):
    print("\n" + "=" * 64)
    print("  " + title)
    print("=" * 64)


def show(mem, label=""):
    if mem is None:
        print(f"   {label} -> None (gone)")
        return
    print(f"   {label} [{mem.key}] status={mem.status} score={mem.score:.3f}")
    print(f"      \"{mem.text}\"")


# ------------------------------------------------------------------ Act 1
banner("ACT 1 — FORGETTING: low-importance memories decay away")

clock = FakeClock()
store = MemoryStore(clock=clock, half_life=60, forget_threshold=0.15)  # 60s half-life for demo

store.write("wifi.guest", "Guest wifi password is 'coffee123'",
            importance=0.3, source="agent")
store.write("deploy.region", "Production deploys to us-west-2",
            importance=0.95, source="agent", half_life=30 * 86400)  # important: 30-day half-life

print("\n-> right after writing:")
for m in store.query():
    show(m)

clock.advance(300)  # 5 half-lives later

print("\n-> 5 minutes later (low-importance trivia has decayed below the forget threshold):")
for m in store.query():
    show(m)
forgotten = store.read("wifi.guest", include_forgotten=True)
show(forgotten, "wifi.guest (recalled with include_forgotten=True)")
print("   The row still exists in SQLite - forgetting is a filter, not erasure.")


# ------------------------------------------------------------------ Act 2
banner("ACT 2 — TOMBSTONES: deletes are recorded, not erased")

store2 = MemoryStore(clock=FakeClock())
store2.write("user.theme", "User prefers dark mode", importance=0.8)
print("\n-> before delete:")
show(store2.read("user.theme"))

n = store2.delete("user.theme", reason="user switched to light mode on 2026-09-29")
print(f"\n-> delete() tombstoned {n} row(s)")
print("-> after delete:")
show(store2.read("user.theme"))
print("-> but the audit trail remembers:")
for e in store2.audit(key="user.theme"):
    print(f"   #{e.id} {e.action:10s} actor={e.actor} details={e.details}")
print("-> and the row is still in the DB:")
print(f"   stats: {store2.stats()['by_status']}")


# ------------------------------------------------------------------ Act 3
banner("ACT 3 — CONFLICT ARBITRATION: contradictory writes get judged")

store3 = MemoryStore(clock=FakeClock(), conflict_policy=ConflictPolicy.MANUAL)
store3.write("office.wifi", "Office wifi password is 'abc123'", importance=0.7)
print("\n-> agent learns a new (contradictory) fact:")
m2 = store3.write("office.wifi", "Office wifi password is 'xyz789'", importance=0.7)
show(m2, "new write")

pending = store3.pending_conflicts()
print(f"\n-> {len(pending)} conflict(s) pending; the new write is quarantined:")
print(f"   conflict #{pending[0].id} policy={pending[0].policy} resolution={pending[0].resolution}")
print("-> read() still returns the old, trusted fact until a human arbitrates:")
show(store3.read("office.wifi"))

print("\n-> human arbitrates: the new password wins")
c = store3.resolve_conflict(pending[0].id, winner="new")
print(f"   conflict #{c.id} resolved: {c.resolution}")
show(store3.read("office.wifi"), "after arbitration")

print("\n-> full audit trail for office.wifi:")
for e in store3.audit(key="office.wifi"):
    print(f"   #{e.id} {e.action:16s} mem={e.memory_id} details={e.details}")

banner("DONE — memgovern: memories that know how to die, and why")

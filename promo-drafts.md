# Promo drafts for memgovern

> NOTE: Repo push is pending (GitHub token lacks repo-creation scope as of 2026-09-29).
> Intended URL (name confirmed free on the account): https://github.com/hahahahahahahahah6/memgovern
> Verify the URL is live before posting either draft.

---

## 1. r/SideProject draft

**Title:** I built a tiny memory governance layer for AI agents — decay, tombstones, and conflict arbitration

**Body:**

Everyone building agent memory seems to be working on the "store and search" side. I kept running into the other half: the **write and delete** side. When should a memory fade? What happens when it expires? And when two memories disagree, who wins?

So I built memgovern — a zero-dependency Python library (SQLite under the hood) that treats those as first-class operations:

- **Decay:** memories carry importance + TTL, and queries rank by an exponential decay score instead of recency alone.
- **Tombstones:** deletes are tombstones, not erasures — reversible, with a reason attached, and a full audit trail.
- **Conflict arbitration:** when a new write contradicts an existing memory on the same key, the new write is quarantined as PENDING and you pick the winner (or keep both / let a human decide), instead of silently overwriting.

This came out of watching real failure modes: agents acting on stale context, contradictory facts with no rule for which one is authoritative, deletes you can't undo or explain. The philosophy is deliberately conservative — quarantine first, arbitrate, keep receipts — which is the opposite of the auto-replace-everything approach most memory libraries take.

It's early and the conflict detection is key-based (semantic contradiction detection is the big known gap). But it's usable now:

```
pip install -e .
python demo.py   # three-act demo: forgetting → tombstones → conflict arbitration
```

Repo: https://github.com/hahahahahahahahah6/memgovern

Would genuinely appreciate critique — especially from anyone who's shipped agent memory in production. What broke for you: staleness, contradictions, or deletes you regretted?

---

## 2. Show HN draft

**Title:** Show HN: memgovern — a tiny memory governance layer for AI agents (decay, tombstones, conflict arbitration)

**Body:**

Most agent-memory work focuses on store and search. memgovern does the neglected half: write and delete.

- Memories decay (importance + TTL, exponential scoring at query time)
- Deletes are tombstones — reversible, reasoned, audited
- Contradictory writes get quarantined as PENDING until arbitrated (overwrite / keep both / human decides), instead of silently clobbering the old value

Zero dependencies, SQLite under the hood. `pip install -e .` then `python demo.py` for a three-act demo (forgetting → tombstones → arbitration).

Honest limitations: conflict detection is key-based, not semantic (negation-style contradictions still slip through); ranking is naive token-overlap, meant to sit under something like mem0/cognee rather than replace it.

Repo: https://github.com/hahahahahahahahah6/memgovern

Happy to hear what actually broke for you in production agent memory — staleness, contradiction handling, or deletes.

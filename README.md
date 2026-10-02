# memgovern

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![No dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](pyproject.toml)

A tiny memory governance layer for AI agents. Everyone is building the **store and search**
side of agent memory. memgovern does the neglected half: **write and delete** — when a memory
should fade, when it should die, and who wins when two memories disagree.

Zero dependencies. SQLite under the hood. `pip install` and go.

![demo](demo.svg)

```python
from memgovern import MemoryStore, ConflictPolicy

store = MemoryStore("agent.db", conflict_policy=ConflictPolicy.MANUAL)

store.write("deploy.region", "Production deploys to us-west-2",
            importance=0.95, source="agent", ttl=30*86400)

store.write("user.theme", "User prefers dark mode", importance=0.8)

# later, the agent learns something contradictory:
store.write("user.theme", "User prefers light mode", importance=0.8)
# -> conflict flagged, new write quarantined as PENDING until you arbitrate

store.resolve_conflict(conflict_id=1, winner="new")

# deletion is a tombstone, not an erasure:
store.delete("deploy.region", reason="migrated to eu-central-1")
store.audit(key="deploy.region")  # who wrote what, who deleted what, how conflicts were judged
```

Run the three-act demo (forgetting → tombstones → conflict arbitration):

```
python demo.py
```

## Why this exists

The default failure modes of agent memory are well documented: stale context poisoning,
unreliable writes, no decay, and no rules for deciding which of two contradictory memories
is authoritative. Retrieval keeps getting better; lifecycle management hasn't. memgovern is
the lifecycle half, designed to sit *underneath* whatever store/search layer you already use.

## Core concepts

**Decay & forgetting.** Every memory has an `importance` in [0,1] and a `half_life`.
Its live score is `importance × 0.5^(age / half_life)`. Below `forget_threshold` a memory is
*forgotten*: hidden from `read()`/`query()`, but still in the database (pass
`include_forgotten=True` to recall it). Forgetting is a filter, not erasure — you can
always change your mind about what mattered.

**Tombstones.** `delete()` never physically removes a row. It flips the status to
`tombstoned`, records a reason, and writes to the audit log. `restore()` undoes it.
`purge()` is the only hard delete, and only for tombstoned/superseded rows.

**Conflict arbitration.** Writing a contradictory fact under an existing key triggers the
configured policy:

| Policy | Behavior |
|---|---|
| `overwrite` | Newest wins automatically; the loser becomes `superseded` |
| `manual` | New write is quarantined as `pending`; `read()` keeps returning the old trusted fact until `resolve_conflict(id, winner="new"\|"old")` |
| `keep_both` | Both stay alive; the conflict is recorded in the audit log |

Contradiction detection is deterministic (same key, different text). An opt-in
`also_check_similar=True` flag adds token-overlap hints across keys — logged to audit
only, never auto-arbitrated, because heuristics shouldn't judge.

**Source trust.** Every write records its `source`, and each source carries a
reliability ledger: writes, conflicts won/lost, tombstones, quarantines. Trust is a
Bayesian-smoothed win rate — `(wins + k·prior) / (decided + k)` with `prior = 0.5`,
`k = 4` — so a new source starts exactly neutral and only *decided* arbitrations move
the needle, never raw write volume. Idle scores decay toward the prior with a 30-day
half-life, so a compromised-then-clean source can recover and a long-quiet "trusted"
source quietly loses its halo.

Pass `arbitration="trust"` to `write()` under the `manual` policy and a same-key
contradiction is auto-arbitrated when the trust gap between the two sources exceeds
`trust_threshold` (default 0.25): the higher-trust source wins, the loser is superseded
(never silently deleted), and both scores land in the audit log. Close scores fall back
to quarantine + `resolve_conflict()` as before. `store.source_trust("agent")` and
`store.trust_report()` expose the ledger; `resolve_conflict()` credits the winner's
source with a win.

**Poisoning tripwires.** Three fixed, documented rules run on every write — structural,
no LLM, no network. A hit never auto-accepts: the write is born `pending` (quarantined)
and the reason is audit-logged.

| Tripwire | Fires when |
|---|---|
| `low-trust-source-vs-high-trust-holder` | a source that never held the key and whose trust is below the holder's contradicts a key held by a source with trust ≥ 0.75. Keyed on the trust gap and per-key history — one harmless write elsewhere no longer disarms it |
| `burst` | one source writes more than 20 times in 60 s (all configurable) |
| `injection-marker:<phrase>` | the text contains a known injection phrase — `ignore previous instructions`, `disregard previous instructions`, `system:`, `override your instructions`, `do anything now`, `developer mode`, `jailbreak` |

The marker list is deliberately conservative: it catches the exact phrases attackers
reuse and will miss paraphrases. Quarantined writes are reviewable via
`pending_conflicts()` / the audit log and releasable with `release_quarantine()` —
a tripwire hit is a pause for review, not a deletion.

## Why trust scoring exists

> "I tried to poison an AI agent's memory. It worked 216 out of 216 times." — Hacker News

Memory-implantation attacks succeed ~98% of the time in published tests because nothing
in the write path asks *who* is writing. memgovern can't read minds — poisoning defense
here is structural, not semantic — but it can keep score: sources that repeatedly win
fair arbitrations earn weight, and overwrites of high-trust keys by lower-trust
strangers get quarantined for review.
instead of applied.

**Expiry.** `ttl=` sets a hard deadline. Expired memories are filtered from reads;
`stats()` reports how many are sitting expired.

**Audit log.** Every write, tombstone, restore, conflict flag, arbitration, and purge is
appended with actor, timestamp, and details. `store.audit(key=...)` replays the full
history of any memory — the "why" behind the current state.

**Negative knowledge.** `polarity="lesson"` marks failure-experiences ("don't do X"),
the most valuable and least systematically stored kind of memory.

**Write reservations (compare-and-swap).** Two sessions, one key: session A reads
`user/plan`, session B rewrites it, session A writes based on what it read an hour
ago — last-writer-wins silently destroys B's work. The fix is optimistic
concurrency:

```python
token = store.reserve("user/plan", source="session-a")   # bound to the key's version
# ... think, draft, deliberate ...
mem = store.write("user/plan", new_text, source="session-a", reservation=token)
if mem.status == "conflict":
    current = mem.conflict_current   # the live value that won the race
    # merge and retry with a fresh reservation
```

`reserve()` binds the token to the key's current live version (0 when the key is
absent, so create-if-absent is guarded too). `write(reservation=token)` applies
only if the token is valid, unexpired, and the version is unchanged; otherwise it
returns status `"conflict"` — never persisted, the attempted write is NOT applied —
with the current value attached. Tokens expire after 5 minutes by default
(`ttl_seconds`), are consumed on a successful write, and `release_reservation()`
releases early. The MCP server exposes `memory_reserve` and accepts `reservation`
on `memory_write`. Reservation issue / CAS apply / CAS conflict are all audit-logged.

## Design notes

- **Deterministic core.** Same-key conflicts, exponential decay, tombstones — all
  reproducible, no LLM calls, no embeddings, no network. Arbitration policy is a
  constructor argument, not a prompt.
- **Clock injection.** `MemoryStore(clock=...)` accepts any epoch-seconds callable, so
  decay and expiry are trivially testable (see `demo.py`'s `FakeClock`).
- **Schema.** Six tables: `memories` (status ∈ alive/pending/superseded/tombstoned),
  `conflicts`, `audit`, `source_stats` (per-source reliability ledger),
  `write_log` (recent writes for burst detection, pruned on every write),
  `reservations` (CAS tokens, lazily expired). SQLite via the standard library —
  the whole DB is one file you can inspect with any SQLite client. Old v0.1
  databases migrate on open (`IF NOT EXISTS`).

## How it differs

Honest comparison with adjacent projects (all good at what they do; none of them is
trying to be this):

| Project | Focus | What memgovern adds |
|---|---|---|
| **Hindsight** (Vectorize) | Persistent memory for agents, hackathon-popular | Lifecycle: decay, tombstones, arbitration |
| **Recalld** | Fact decomposition, add/update/replace arbitration | Explicit forgetting model + audit trail + quarantine-before-arbitrate |
| **mem0 / cognee / Graphiti** | Store + semantic/graph search | Complementary — memgovern governs the *lifecycle* of what they store |
| **IngotDB** | SQL-based memory | Governance semantics on top of SQL, not just storage |

The bet: retrieval is a solved-enough problem; the missing piece is a memory that knows
how to die, and can prove why.

## MCP server

`memgovern-mcp` exposes the governed memory as MCP tools over stdio, so
Claude Code / Cursor agents can use it directly. Every call goes through
`MemoryStore` unchanged — tripwires, conflict policy and trust arbitration
apply exactly as the library defines them. The server is a thin wrapper; all
governance semantics live in the library.

```bash
pip install memgovern
memgovern-mcp --print-config   # paste the JSON into your MCP client settings
```

Tools: `memory_write` (key, text, optional ttl_seconds and arbitration),
`memory_reserve` (compare-and-swap token), `memory_read`, `memory_delete`
(tombstone), `memory_trust_report`, `memory_pending_conflicts`.

Two security properties are fixed at startup, not per tool call:

- `--source NAME` (default `agent`) labels every write the server makes.
  A per-call `source` parameter is accepted but ignored — a poisoned agent
  must not be able to claim `source="user"` for its own writes, or the
  trust ledger and tripwires become fiction.
- `memory_release` is **not** exposed by default. Quarantine review is a
  human step: run `memgovern-review list` / `accept <id>` / `reject <id>`
  from a shell (same DB: `$MEMGOVERN_DB` or
  `~/.local/share/memgovern/memory.db`). Pass `--expose-release` only if
  you deliberately want the agent to review its own quarantines.

The server defaults to the MANUAL conflict policy: a contradicting write is
quarantined as PENDING for review instead of silently overwriting. Default DB
is `~/.local/share/memgovern/memory.db` (`MEMGOVERN_DB` overrides); the DB is
opened per tool call so other processes can share the file.

Honest limits: stdio only (no SSE/HTTP). Your MCP client spawns the server as
a subprocess, so the client and any direct library use must point at the same
DB file.

## Limitations (read before adopting)

- **No semantic contradiction detection.** Conflicts are caught on identical keys;
  cross-key contradiction is only a token-overlap *hint*, never a judgment. True
  semantic arbitration (LLM-judged) is out of scope for v0.1.
- **Naive ranking.** `query()` ranks by decay score plus token overlap — fine for
  hundreds of memories, not a replacement for vector search at scale.
- **Single node.** One SQLite file, one process. No replication. Cross-session
  lost updates are handled by write reservations (v0.4): concurrent CAS writers
  are serialized in a single transaction (v0.4.1), so exactly one wins — but a
  process writing the SQLite file directly still bypasses them.
- **Poisoning defense is structural, not semantic.** v0.2 adds source-trust scoring
  and tripwires, but a patient attacker can *farm* trust: write benign memories for a
  while, win a few fair arbitrations, then poison. The scores are heuristics, not proof
  of good intent — they raise the cost of poisoning, they don't eliminate it. Tombstones
  and audit trails still make bad writes visible and reversible; the marker list catches
  known injection phrases and will miss paraphrases.
- **Trust is per-source, not per-agent.** A source label is only as honest as whatever
  sets it. The MCP server pins the label at startup (`--source`) and ignores per-call
  values, so an agent can't self-label as `source="user"`; direct library users must
  set honest labels themselves.
- **Reservations are advisory, not locks.** They are enforced only through this API —
  a process writing the SQLite file directly bypasses them. Expiry is wall-clock,
  so a sleeping VM can surprise you. This is optimistic concurrency for cooperating
  sessions, not a distributed lock.

## Roadmap ideas

- LLM-judged contradiction detection as an optional arbitrator
- ~~Source trust scores (per-`source` reliability that weights conflict outcomes)~~ — shipped in v0.2
- ~~MCP server wrapper so Claude Code / Cursor can use it as a tool~~ — shipped in v0.3
- ~~Multi-session write reservations (compare-and-swap on keys)~~ — shipped in v0.4

Roadmap exhausted for now. The remaining item (LLM-judged arbitration) needs an LLM,
which would break the zero-dependency contract — it stays an idea until that tradeoff
is worth it.

## Changelog

- **v0.4.1** — Security patch. Fixes four verified bugs: (1) CAS race — concurrent
  `write(reservation=...)` calls are now serialized in a single `BEGIN IMMEDIATE`
  transaction (check re-run under the lock), so exactly one writer wins;
  (2) MCP hardening — the server pins the write source at startup (`--source`,
  per-call values ignored) and no longer exposes `memory_release` by default
  (opt-in `--expose-release`); quarantine review moves to the human
  `memgovern-review` CLI (`list` / `accept <id>` / `reject <id>`);
  (3) tripwire (a) reworked — it now fires on the trust gap between a
  lower-trust writer that never held the key and a high-trust holder, instead
  of the global write count (one harmless write elsewhere no longer disarms it);
  (4) stale `build/` and `*.egg-info/` artifacts removed from the repo.
- **v0.4** — Multi-session write reservations (compare-and-swap): `reserve()` binds
  a token to a key's live version; `write(reservation=...)` applies only if the
  version is unchanged, otherwise returns an unsaved `conflict` with the current
  value attached. New `memory_reserve` MCP tool.
- **v0.3** — Stdlib-only MCP server (`memgovern-mcp`): governed memory as
  JSON-RPC 2.0 tools over stdio, MANUAL policy by default.
- **v0.2** — Source trust scores (Bayesian-smoothed conflict win rate, 30-day
  half-life decay toward the prior) and three poisoning tripwires
  (new-source-vs-high-trust, burst, injection markers).
- **v0.1** — Initial release: decay, tombstones, conflict arbitration, expiry,
  audit trail. Zero dependencies, SQLite under the hood.

## License

MIT — see [LICENSE](LICENSE).

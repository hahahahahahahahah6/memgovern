"""memgovern MCP server: governed agent memory as a tool.

Stdlib-only JSON-RPC 2.0 over stdio (newline-delimited, the framing real
MCP stdio servers use). The server is a thin wrapper -- every write goes
through ``MemoryStore.write()`` unchanged, so tripwires, conflict policy
and trust arbitration apply exactly as the library defines them. Nothing
about governance is reimplemented here.

The server opens the SQLite DB per tool call (cheap for SQLite) so other
processes -- the library used directly, another agent session -- can share
the same file safely. Default DB: ``~/.local/share/memgovern/memory.db``,
overridden by ``MEMGOVERN_DB``.

The server defaults to the MANUAL conflict policy: a contradicting write
is quarantined as PENDING for review instead of silently overwriting. That
is what makes ``arbitration="trust"`` meaningful and gives the
``memory_pending_conflicts`` / ``memory_release`` review loop something to
review. (The library default is OVERWRITE; the server chooses the safer
default for an agent-facing tool and says so.)

Protocol notes: ``initialize`` -> ``notifications/initialized`` ->
``tools/list`` -> ``tools/call``. Malformed input gets a JSON-RPC error
response, never a crash. All logging goes to stderr; stdout is the
protocol.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .models import ConflictPolicy, Memory, MemoryStatus
from .store import MemoryStore

SERVER_NAME = "memgovern"
PROTOCOL_VERSION = "2025-11-25"


def db_path() -> str:
    """Resolve the SQLite path. Env override wins; otherwise a per-user file."""
    override = os.environ.get("MEMGOVERN_DB")
    if override:
        return os.path.expanduser(override)
    directory = os.path.expanduser("~/.local/share/memgovern")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "memory.db")


def _store() -> MemoryStore:
    # MANUAL policy: contradicting writes quarantine for review instead of
    # silently overwriting. Documented above; the library default is unchanged.
    return MemoryStore(db_path(), conflict_policy=ConflictPolicy.MANUAL)


def _log(msg: str) -> None:
    sys.stderr.write(f"memgovern-mcp: {msg}\n")
    sys.stderr.flush()


# ------------------------------------------------------------------ tool defs

TOOLS = [
    {
        "name": "memory_write",
        "description": (
            "Store a memory under a key. Governed by memgovern: same-key "
            "contradictions are quarantined for review (manual policy), "
            "prompt-injection tripwires quarantine suspicious writes, and "
            "arbitration='trust' auto-arbitrates by source trust. "
            "SQLite backend. Structural defense, no LLM, no network."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "memory key, e.g. 'user/name'"},
                "text": {"type": "string", "description": "the memory content"},
                "source": {"type": "string", "description": "who is writing; feeds the trust ledger",
                           "default": "agent"},
                "ttl_seconds": {"type": "number", "description": "optional expiry in seconds"},
                "arbitration": {"type": "string", "enum": ["manual", "trust"],
                                "description": "manual (default): quarantine contradictions; "
                                               "trust: auto-arbitrate when the source-trust gap "
                                               "exceeds the threshold"},
            },
            "required": ["key", "text"],
        },
    },
    {
        "name": "memory_read",
        "description": (
            "Read the latest live memory for a key. Quarantined, tombstoned "
            "and superseded memories are never returned."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "memory key"},
            },
            "required": ["key"],
        },
    },
    {
        "name": "memory_delete",
        "description": (
            "Delete a key's live memory. This is a tombstone, not an erasure: "
            "the row is kept for audit and can be restored via the library."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "memory key"},
                "reason": {"type": "string", "description": "why it was deleted"},
            },
            "required": ["key"],
        },
    },
    {
        "name": "memory_trust_report",
        "description": (
            "Per-source reliability ledger: Bayesian-smoothed conflict win "
            "rate per source, highest trust first. New sources score 0.5 "
            "(neutral)."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "memory_pending_conflicts",
        "description": (
            "Everything waiting for review: quarantined contradicting writes "
            "(with both sides' text) and tripwire-quarantined writes (with "
            "the tripwire reason). Pair with memory_release."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "memory_release",
        "description": (
            "Review a quarantined write. accept: apply it (tripwire "
            "quarantine) or resolve its conflict in its favor. reject: "
            "discard it (tripwire quarantine is tombstoned; a conflict is "
            "resolved in the old version's favor)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "quarantine_id": {"type": "integer",
                                 "description": "memory id of the quarantined write"},
                "decision": {"type": "string", "enum": ["accept", "reject"]},
                "source": {"type": "string",
                           "description": "who reviewed it (audit label)",
                           "default": "agent"},
            },
            "required": ["quarantine_id", "decision"],
        },
    },
]

_TOOL_NAMES = {t["name"] for t in TOOLS}


# ------------------------------------------------------------------ handlers

def _ok_text(payload: Any) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


def _tool_error(message: str) -> Dict[str, Any]:
    # MCP convention: tool-level failures are results with isError, not
    # JSON-RPC errors.
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _memory_json(mem: Memory) -> Dict[str, Any]:
    return {
        "id": mem.id, "key": mem.key, "text": mem.text, "status": mem.status,
        "source": mem.source, "version": mem.version,
        "created_at": mem.created_at, "expires_at": mem.expires_at,
    }


def _quarantine_reason(store: MemoryStore, mem: Memory) -> Optional[str]:
    """Best-effort human reason why a PENDING memory is quarantined."""
    if mem.conflict_id is not None:
        # Was it a tripwire, or a plain contradiction quarantine?
        for ev in store.audit(action="tripwire_flagged", limit=50):
            try:
                details = ev.details if isinstance(ev.details, dict) else {}
            except Exception:
                details = {}
            if details.get("conflict_id") == mem.conflict_id:
                return "tripwire: " + str(details.get("reason", "unknown"))
        conflict = store.get_conflict(mem.conflict_id)
        policy = conflict.policy if conflict else "manual"
        return f"contradiction quarantined under {policy} policy"
    for ev in store.audit(action="tripwire_flagged", limit=50):
        if ev.memory_id == mem.id:
            details = ev.details if isinstance(ev.details, dict) else {}
            return "tripwire: " + str(details.get("reason", "unknown"))
    return "quarantined (reason not found in audit log)"


def _pending_memories(store: MemoryStore) -> List[Memory]:
    rows = store._db.execute(
        "SELECT * FROM memories WHERE status=? ORDER BY id", (MemoryStatus.PENDING,)
    ).fetchall()
    now = store._now()
    return [store._row_to_memory(r, now) for r in rows]


def _tool_memory_write(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    key = args["key"]
    text = args["text"]
    if not isinstance(key, str) or not isinstance(text, str):
        raise ValueError("key and text must be strings")
    kwargs: Dict[str, Any] = {
        "source": args.get("source", "agent"),
    }
    if args.get("ttl_seconds") is not None:
        kwargs["ttl"] = float(args["ttl_seconds"])
    if args.get("arbitration") is not None:
        kwargs["arbitration"] = args["arbitration"]
    mem = store.write(key, text, **kwargs)
    out = _memory_json(mem)
    out["conflict_id"] = mem.conflict_id
    out["source_trust"] = round(store.source_trust(mem.source), 4)
    if mem.status == MemoryStatus.PENDING:
        out["quarantine_reason"] = _quarantine_reason(store, mem)
    return _ok_text(out)


def _tool_memory_read(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    key = args["key"]
    if not isinstance(key, str):
        raise ValueError("key must be a string")
    mem = store.read(key)
    if mem is None:
        return _ok_text({"found": False, "key": key})
    out = _memory_json(mem)
    out["found"] = True
    out["source_trust"] = round(store.source_trust(mem.source), 4)
    return _ok_text(out)


def _tool_memory_delete(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    key = args["key"]
    if not isinstance(key, str):
        raise ValueError("key must be a string")
    n = store.delete(key, reason=args.get("reason"))
    return _ok_text({"key": key, "tombstoned": n})


def _tool_memory_trust_report(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    sources = [
        {"source": s, "trust": round(score, 4), "writes": writes,
         "conflicts_won": wins, "conflicts_lost": losses}
        for s, score, writes, wins, losses in store.trust_report()
    ]
    return _ok_text({"sources": sources})


def _tool_memory_pending_conflicts(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    conflicts_out = []
    tripwired_out = []
    for mem in _pending_memories(store):
        entry = _memory_json(mem)
        entry["quarantine_reason"] = _quarantine_reason(store, mem)
        if mem.conflict_id is not None:
            conflict = store.get_conflict(mem.conflict_id)
            old = store._get_by_id(conflict.old_id) if conflict and conflict.old_id else None
            entry["conflict_id"] = mem.conflict_id
            entry["old_text"] = old.text if old else None
            entry["old_source"] = old.source if old else None
            conflicts_out.append(entry)
        else:
            tripwired_out.append(entry)
    return _ok_text({"conflicts": conflicts_out, "tripwire_quarantined": tripwired_out})


def _tool_memory_release(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    mem_id = args["quarantine_id"]
    decision = args["decision"]
    if not isinstance(mem_id, int) or isinstance(mem_id, bool):
        raise ValueError("quarantine_id must be an integer memory id")
    if decision not in ("accept", "reject"):
        raise ValueError("decision must be 'accept' or 'reject'")
    reviewer = args.get("source", "agent")

    mem = store._get_by_id(mem_id)
    if mem is None:
        return _tool_error(f"no memory #{mem_id}")
    if mem.status != MemoryStatus.PENDING:
        return _tool_error(f"memory #{mem_id} is not quarantined (status={mem.status})")

    if mem.conflict_id is None:
        # Tripwire quarantine: no conflict row.
        if decision == "accept":
            released = store.release_quarantine(mem_id)
            new_status = released.status
        else:
            new_status = _reject_tripwire_quarantine(store, mem, reviewer)
    else:
        conflict = store.resolve_conflict(
            mem.conflict_id, winner="new" if decision == "accept" else "old")
        new_status = "alive" if conflict.winner_id == mem_id else MemoryStatus.SUPERSEDED
    return _ok_text({"memory_id": mem_id, "decision": decision,
                     "new_status": new_status, "reviewed_by": reviewer})


def _reject_tripwire_quarantine(store: MemoryStore, mem: Memory, reviewer: str) -> str:
    """Discard a tripwire-quarantined write: tombstone it (reviewed, never applied)."""
    now = store._now()
    store._db.execute(
        "UPDATE memories SET status=?, updated_at=?, deleted_reason=? WHERE id=?",
        (MemoryStatus.TOMBSTONED, now,
         f"tripwire quarantine rejected by {reviewer}", mem.id),
    )
    store._db.commit()
    store._audit("tripwire_rejected", mem.key, mem.id,
                 {"reviewed_by": reviewer,
                  "note": "quarantined write reviewed and discarded; never applied"})
    return MemoryStatus.TOMBSTONED


_TOOL_HANDLERS = {
    "memory_write": _tool_memory_write,
    "memory_read": _tool_memory_read,
    "memory_delete": _tool_memory_delete,
    "memory_trust_report": _tool_memory_trust_report,
    "memory_pending_conflicts": _tool_memory_pending_conflicts,
    "memory_release": _tool_memory_release,
}


def _call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name not in _TOOL_HANDLERS:
        return _tool_error(f"unknown tool: {name}")
    if not isinstance(args, dict):
        raise ValueError("tools/call params.arguments must be an object")
    store = _store()
    try:
        return _TOOL_HANDLERS[name](store, args)
    finally:
        store.close()


# ------------------------------------------------------------------ protocol

def _error_response(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": code, "message": message}}


def handle_request(req: Any) -> Optional[Dict[str, Any]]:
    """Dispatch one decoded JSON-RPC message. Returns a response dict, or
    None for notifications (no response). Never raises."""
    try:
        if not isinstance(req, dict):
            return _error_response(None, -32600, "invalid request: not an object")
        method = req.get("method")
        req_id = req.get("id")
        params = req.get("params") or {}

        if not isinstance(method, str):
            return _error_response(req_id, -32600, "invalid request: missing method")

        if method == "notifications/initialized":
            return None  # notification: no response

        if method == "initialize":
            return {"jsonrpc": "2.0", "id": req_id, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": __version__},
            }}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"tools": TOOLS}}

        if method == "tools/call":
            if not isinstance(params, dict):
                return _error_response(req_id, -32602, "invalid params")
            name = params.get("name")
            args = params.get("arguments") or {}
            if not isinstance(name, str) or name not in _TOOL_NAMES:
                return _tool_error_result(req_id, f"unknown tool: {name}")
            try:
                result = _call_tool(name, args)
            except ValueError as e:
                return _error_response(req_id, -32602, f"invalid params: {e}")
            except Exception as e:  # noqa: BLE001 -- fail-safe: tool errors must not kill the server
                _log(f"tool {name} failed: {e!r}")
                result = _tool_error(f"tool {name} failed: {e}")
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        return _error_response(req_id, -32601, f"method not found: {method}")
    except Exception as e:  # noqa: BLE001 -- the server never crashes on input
        _log(f"dispatch failed: {e!r}")
        req_id = req.get("id") if isinstance(req, dict) else None
        return _error_response(req_id, -32603, "internal error")


def _tool_error_result(req_id: Any, message: str) -> Dict[str, Any]:
    # Unknown tool name: still a tool-level error result per MCP convention.
    return {"jsonrpc": "2.0", "id": req_id, "result": _tool_error(message)}


def serve(stdin=None, stdout=None) -> int:
    """Run the stdio loop. Returns process exit code."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = _error_response(None, -32700, "parse error: not JSON")
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
            continue
        resp = handle_request(req)
        if resp is None:
            continue  # notification
        stdout.write(json.dumps(resp) + "\n")
        stdout.flush()
    return 0


def print_config() -> int:
    """Print a Claude Code MCP config snippet for copy-paste into settings."""
    sys.stdout.write(json.dumps(
        {"mcpServers": {SERVER_NAME: {"command": "memgovern-mcp"}}}, indent=2) + "\n")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--print-config" in argv:
        return print_config()
    if "-h" in argv or "--help" in argv:
        sys.stdout.write(
            "memgovern-mcp: governed agent memory as an MCP stdio server.\n"
            "Usage: memgovern-mcp [--print-config]\n"
        )
        return 0
    return serve()


if __name__ == "__main__":
    sys.exit(main())

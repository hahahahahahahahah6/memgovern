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
``memory_pending_conflicts`` review loop something to review. (The library
default is OVERWRITE; the server chooses the safer default for an agent-facing
tool and says so.)

Security model
--------------
Two properties are enforced at startup, not per tool call:

* ``--source NAME`` (default ``agent``) fixes the source label for every
  write the server makes. Tool-call ``source`` params are ignored: a
  poisoned agent must not be able to claim ``source="user"`` for its own
  writes, or the trust ledger and tripwires become fiction.
* ``memory_release`` is NOT exposed unless ``--expose-release`` is passed.
  Quarantine review is a human step -- run ``memgovern-review`` from a
  shell. An agent that can release its own quarantines is the prisoner
  judging their own case.

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


# ------------------------------------------------------------------ server config

# Who the server claims writes are from. Fixed at startup via --source;
# tool-call "source" params are ignored, so a poisoned agent cannot label
# its own writes source="user" and walk past the trust ledger.
SERVER_SOURCE = "agent"
# memory_release is hidden unless --expose-release is passed. Review is a
# human step (memgovern-review CLI), not an agent tool.
EXPOSE_RELEASE = False


def _configure(source=None, expose_release=None):
    """Set (or reset) the server config. Called by main(); tests call it
    directly. No arguments resets to the secure defaults."""
    global SERVER_SOURCE, EXPOSE_RELEASE
    SERVER_SOURCE = "agent" if source is None else source
    EXPOSE_RELEASE = False if expose_release is None else bool(expose_release)


def _parse_args(argv):
    """Parse memgovern-mcp argv (without the program name).

    Returns (source, expose_release, action) where action is one of
    "serve", "print-config", "help". Raises ValueError on bad input."""
    source = "agent"
    expose_release = False
    action = "serve"
    argv = list(argv)
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--source":
            i += 1
            if i >= len(argv) or argv[i].startswith("-"):
                raise ValueError("--source needs a value")
            source = argv[i]
        elif a.startswith("--source="):
            source = a.split("=", 1)[1]
        elif a == "--expose-release":
            expose_release = True
        elif a == "--print-config":
            action = "print-config"
        elif a in ("-h", "--help"):
            action = "help"
        else:
            raise ValueError(f"unknown argument: {a}")
        i += 1
    if not source:
        raise ValueError("--source needs a non-empty value")
    return source, expose_release, action


# ------------------------------------------------------------------ tool defs

_BASE_TOOLS = [
    {
        "name": "memory_write",
        "description": (
            "Store a memory under a key. Governed by memgovern: same-key "
            "contradictions are quarantined for review (manual policy), "
            "prompt-injection tripwires quarantine suspicious writes, and "
            "arbitration='trust' auto-arbitrates by source trust. "
            "Pass a reservation token from memory_reserve for "
            "compare-and-swap: the write applies only if the key is unchanged "
            "since the reservation, otherwise it returns status 'conflict' "
            "with the current value. "
            "The write's source label is fixed at server startup (--source); "
            "any per-call source is ignored. "
            "SQLite backend. Structural defense, no LLM, no network."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "memory key, e.g. 'user/name'"},
                "text": {"type": "string", "description": "the memory content"},
                "ttl_seconds": {"type": "number", "description": "optional expiry in seconds"},
                "arbitration": {"type": "string", "enum": ["manual", "trust"],
                                "description": "manual (default): quarantine contradictions; "
                                               "trust: auto-arbitrate when the source-trust gap "
                                               "exceeds the threshold"},
                "reservation": {"type": "string",
                                "description": "optional CAS token from memory_reserve; "
                                               "lost races return status 'conflict' and are NOT applied"},
            },
            "required": ["key", "text"],
        },
    },
    {
        "name": "memory_reserve",
        "description": (
            "Reserve a key for a compare-and-swap write. Returns an opaque "
            "token bound to the key's current version (0 when absent). Pass "
            "it to memory_write as reservation: the write applies only if no "
            "other session changed the key meanwhile. Prevents last-writer-"
            "wins across concurrent agent sessions. Advisory: only enforced "
            "through this API, not against raw SQLite writes. "
            "The reservation's source label is fixed at server startup "
            "(--source); any per-call source is ignored."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "memory key to reserve"},
                "ttl_seconds": {"type": "number",
                                "description": "reservation expiry in seconds",
                                "default": 300},
            },
            "required": ["key"],
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
            "the tripwire reason). Review them with the memgovern-review CLI "
            "(human step); memory_release is only listed when the server was "
            "started with --expose-release."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]

_RELEASE_TOOL = {
    "name": "memory_release",
    "description": (
        "Review a quarantined write. accept: apply it (tripwire "
        "quarantine) or resolve its conflict in its favor. reject: "
        "discard it (tripwire quarantine is tombstoned; a conflict is "
        "resolved in the old version's favor). "
        "Only exposed when the server was started with --expose-release; "
        "otherwise review via the memgovern-review CLI."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "quarantine_id": {"type": "integer",
                             "description": "memory id of the quarantined write"},
            "decision": {"type": "string", "enum": ["accept", "reject"]},
            "source": {"type": "string",
                       "description": "who reviewed it (audit label only)",
                       "default": "agent"},
        },
        "required": ["quarantine_id", "decision"],
    },
}


def active_tools():
    """The tool list for this server instance. memory_release is included
    only when --expose-release was passed at startup."""
    tools = list(_BASE_TOOLS)
    if EXPOSE_RELEASE:
        tools.append(_RELEASE_TOOL)
    return tools


# Backwards-compatible alias: the full set, including memory_release.
TOOLS = _BASE_TOOLS + [_RELEASE_TOOL]


def _active_tool_names():
    return {t["name"] for t in active_tools()}


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
    # The source label is fixed at server startup (--source). A per-call
    # "source" is ignored: a poisoned agent must not label its own writes.
    kwargs: Dict[str, Any] = {"source": SERVER_SOURCE}
    if args.get("ttl_seconds") is not None:
        kwargs["ttl"] = float(args["ttl_seconds"])
    if args.get("arbitration") is not None:
        kwargs["arbitration"] = args["arbitration"]
    if args.get("reservation") is not None:
        kwargs["reservation"] = args["reservation"]
    mem = store.write(key, text, **kwargs)
    out = _memory_json(mem)
    out["conflict_id"] = mem.conflict_id
    out["source_trust"] = round(store.source_trust(mem.source), 4)
    if mem.status == MemoryStatus.PENDING:
        out["quarantine_reason"] = store.quarantine_reason(mem.id)
    if mem.status == MemoryStatus.CONFLICT:
        # CAS lost race: nothing was applied. Attach the current live value
        # so the caller can merge and retry with a fresh reservation.
        out["note"] = ("CAS conflict: the key changed since the reservation; "
                       "write NOT applied")
        out["current"] = (_memory_json(mem.conflict_current)
                          if mem.conflict_current else None)
    return _ok_text(out)


def _tool_memory_reserve(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    key = args["key"]
    if not isinstance(key, str):
        raise ValueError("key must be a string")
    ttl = args.get("ttl_seconds", 300)
    token = store.reserve(key, source=SERVER_SOURCE,
                          ttl_seconds=float(ttl))
    current = store.read(key)
    return _ok_text({
        "token": token,
        "key": key,
        "current": _memory_json(current) if current else None,
    })


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
        entry["quarantine_reason"] = store.quarantine_reason(mem.id)
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
    # Thin wrapper over MemoryStore.review_quarantine. This tool is only
    # reachable when the server was started with --expose-release; by
    # default review happens via the memgovern-review CLI (human step).
    mem_id = args["quarantine_id"]
    decision = args["decision"]
    if not isinstance(mem_id, int) or isinstance(mem_id, bool):
        raise ValueError("quarantine_id must be an integer memory id")
    if decision not in ("accept", "reject"):
        raise ValueError("decision must be 'accept' or 'reject'")
    reviewer = args.get("source", "agent")  # audit label only
    try:
        new_status = store.review_quarantine(mem_id, decision, reviewer=reviewer)
    except (KeyError, ValueError) as e:
        return _tool_error(str(e))
    return _ok_text({"memory_id": mem_id, "decision": decision,
                     "new_status": new_status, "reviewed_by": reviewer})


_TOOL_HANDLERS = {
    "memory_write": _tool_memory_write,
    "memory_reserve": _tool_memory_reserve,
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
                    "result": {"tools": active_tools()}}

        if method == "tools/call":
            if not isinstance(params, dict):
                return _error_response(req_id, -32602, "invalid params")
            name = params.get("name")
            args = params.get("arguments") or {}
            if not isinstance(name, str) or name not in _active_tool_names():
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
    """Print a Claude Code MCP config snippet for copy-paste into settings.

    Pins --source so the deployed server labels writes as the agent, not
    whatever a tool call claims."""
    sys.stdout.write(json.dumps(
        {"mcpServers": {SERVER_NAME: {"command": "memgovern-mcp",
                                      "args": ["--source", "agent"]}}},
        indent=2) + "\n")
    return 0


_HELP = """\
memgovern-mcp: governed agent memory as an MCP stdio server.
Usage: memgovern-mcp [--source NAME] [--expose-release] [--print-config]

  --source NAME     source label for every write the server makes
                    (default: agent). Per-call "source" params are ignored.
  --expose-release  also expose the memory_release tool. By default review
                    is a human step via the memgovern-review CLI.
  --print-config    print a Claude Code MCP config snippet and exit.
"""


def main(argv=None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Tolerate a leading program name (main(["memgovern-mcp", "--help"])).
    if raw and not raw[0].startswith("-"):
        raw = raw[1:]
    try:
        source, expose_release, action = _parse_args(raw)
    except ValueError as e:
        sys.stderr.write(f"memgovern-mcp: {e}\n")
        return 2
    if action == "help":
        sys.stdout.write(_HELP)
        return 0
    _configure(source=source, expose_release=expose_release)
    if action == "print-config":
        return print_config()
    return serve()


if __name__ == "__main__":
    sys.exit(main())

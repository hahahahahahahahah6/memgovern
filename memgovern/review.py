"""memgovern-review: the human side of the quarantine review loop.

The MCP server does not expose memory_release by default: an agent must not
review its own quarantines. This CLI is the human step -- list what's
awaiting review, then accept or reject each item from a shell.

Usage:
    memgovern-review [--db PATH] list
    memgovern-review [--db PATH] accept <memory-id>
    memgovern-review [--db PATH] reject <memory-id>

The DB defaults to $MEMGOVERN_DB, then ~/.local/share/memgovern/memory.db
(the same default the MCP server uses).
"""

import argparse
import os
import sys
from typing import Optional

from .models import MemoryStatus
from .store import MemoryStore


def _db_path(flag: Optional[str]) -> str:
    if flag:
        return os.path.expanduser(flag)
    env = os.environ.get("MEMGOVERN_DB")
    if env:
        return os.path.expanduser(env)
    return os.path.expanduser("~/.local/share/memgovern/memory.db")


def _cmd_list(store: MemoryStore) -> int:
    rows = store._db.execute(
        "SELECT id, key, source, conflict_id, created_at FROM memories"
        " WHERE status=? ORDER BY id",
        (MemoryStatus.PENDING,),
    ).fetchall()
    if not rows:
        print("nothing awaiting review")
        return 0
    for r in rows:
        reason = store.quarantine_reason(r["id"]) or "quarantined"
        shape = "conflict" if r["conflict_id"] is not None else "tripwire"
        print(f"#{r['id']} [{shape}] key={r['key']!r} source={r['source']!r}")
        print(f"    reason: {reason}")
    print(f"\n{len(rows)} item(s) awaiting review."
          " Use: memgovern-review accept|reject <id>")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="memgovern-review",
        description="Review memgovern's quarantined writes (human step).")
    ap.add_argument("--db", default=None,
                    help="SQLite DB path (default: $MEMGOVERN_DB or "
                         "~/.local/share/memgovern/memory.db)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show everything awaiting review")
    pa = sub.add_parser("accept", help="apply a quarantined write")
    pa.add_argument("memory_id", type=int)
    pr = sub.add_parser("reject", help="discard a quarantined write")
    pr.add_argument("memory_id", type=int)
    args = ap.parse_args(argv)

    store = MemoryStore(_db_path(args.db))
    try:
        if args.cmd == "list":
            return _cmd_list(store)
        try:
            new_status = store.review_quarantine(
                args.memory_id, args.cmd, reviewer="human-cli")
        except (KeyError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"memory #{args.memory_id}: {args.cmd}ed -> {new_status}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())

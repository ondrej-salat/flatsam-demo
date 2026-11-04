#!/usr/bin/env python3
"""context_snapshot.py — Read-only context collector for "where did I end" analysis.

Gathers git state, recently modified files, opencode conversation history,
and Claude memory for the current working directory. Outputs structured text
to stdout for Claude to analyze.

Usage:
    python3 context_snapshot.py [--days N] [--sessions N] [--msg-len N]

Place at scripts/context_snapshot.py in any repo. A global copy also lives
at ~/.claude/skills/where-am-i/scripts/context_snapshot.py.
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def git(*args):
    r = subprocess.run(["git"] + list(args), capture_output=True, text=True)
    return r.stdout.strip()


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=60,
                   help="Days of file modification history (default 60)")
    p.add_argument("--sessions", type=int, default=40,
                   help="Max opencode sessions to include (default 40)")
    p.add_argument("--msg-len", type=int, default=500,
                   help="Max chars per conversation message (default 500)")
    args = p.parse_args()

    cwd = Path.cwd()
    print(f"Context snapshot for: {cwd}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── GIT STATE ─────────────────────────────────────────────────────────────
    section("GIT STATE")

    branch = git("branch", "--show-current")
    print(f"\nBranch: {branch or '(detached HEAD)'}")

    print("\n-- Branches (sorted by recency) --")
    print(git("branch", "-a", "--sort=-committerdate") or "(none)")

    print("\n-- Recent commits --")
    print(git("log", "--oneline", "--format=%h  %ad  %s", "--date=short", "-50"))

    print("\n-- Uncommitted changes --")
    status = git("status", "--short")
    print(status or "(clean)")

    diff_stat = git("diff", "HEAD", "--stat")
    if diff_stat:
        print("\n-- Diff stat --")
        print(diff_stat)

    stash = git("stash", "list")
    if stash:
        print("\n-- Stashes --")
        print(stash)

    # ── RECENTLY MODIFIED FILES ───────────────────────────────────────────────
    section(f"RECENTLY MODIFIED FILES (last {args.days} days)")

    cutoff = time.time() - args.days * 86400
    skip_dirs = {
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
        ".eggs", "*.egg-info",
    }
    modified = []
    for root, dirs, files in os.walk(str(cwd)):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.endswith(".egg-info")]
        for f in files:
            path = os.path.join(root, f)
            try:
                mt = os.path.getmtime(path)
                if mt > cutoff:
                    rel = os.path.relpath(path, str(cwd))
                    modified.append((mt, rel))
            except OSError:
                pass
    modified.sort(reverse=True)
    for mt, path in modified[:60]:
        dt = datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M")
        print(f"  {dt}  {path}")
    if len(modified) > 60:
        print(f"  ... and {len(modified) - 60} more files")

    # ── OPENCODE SESSIONS ─────────────────────────────────────────────────────
    section("OPENCODE CONVERSATION HISTORY")

    db_path = Path.home() / ".local/share/opencode/opencode.db"
    if not db_path.exists():
        print("\nopencode.db not found — skipping")
    else:
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cwd_str = str(cwd)

            sessions = con.execute(
                """
                SELECT id, title,
                       datetime(time_created/1000, 'unixepoch') as created
                FROM session
                WHERE directory = ? OR directory LIKE ?
                ORDER BY time_created DESC
                LIMIT ?
                """,
                (cwd_str, cwd_str + "%", args.sessions),
            ).fetchall()

            print(f"\nFound {len(sessions)} sessions for {cwd_str}")

            for sid, title, created in sessions:
                parts = con.execute(
                    """
                    SELECT p.data
                    FROM part p
                    JOIN message m ON p.message_id = m.id
                    WHERE m.session_id = ?
                      AND json_extract(m.data, '$.role') = 'user'
                    ORDER BY p.time_created
                    """,
                    (sid,),
                ).fetchall()

                user_texts = []
                for (data_str,) in parts:
                    try:
                        d = json.loads(data_str)
                        if d.get("type") == "text" and not d.get("synthetic"):
                            text = d["text"].strip()
                            if text:
                                if len(text) > args.msg_len:
                                    text = text[: args.msg_len] + "…"
                                user_texts.append(text)
                    except Exception:
                        pass

                if not user_texts:
                    continue  # skip sessions with no readable user messages

                print(f"\n{'─' * 50}")
                print(f"[{created}]  {title}")
                for text in user_texts:
                    print(f"  [user] {text}")

            con.close()
        except Exception as e:
            print(f"\nError reading opencode DB: {e}")

    # ── CLAUDE MEMORY ─────────────────────────────────────────────────────────
    section("CLAUDE MEMORY")

    mem_key = str(cwd).replace("/", "-")
    mem_dir = Path.home() / f".claude/projects/{mem_key}/memory"
    if mem_dir.exists():
        md_files = sorted(mem_dir.glob("*.md"))
        print(f"\nFound {len(md_files)} memory files at {mem_dir}")
        for f in md_files:
            if f.name == "MEMORY.md":
                print(f"\n-- {f.name} (index) --")
                print(f.read_text(encoding="utf-8", errors="replace")[:300])
            else:
                print(f"\n-- {f.name} --")
                content = f.read_text(encoding="utf-8", errors="replace")
                if len(content) > 600:
                    content = content[:600] + "…"
                print(content)
    else:
        print(f"\nNo memory directory at {mem_dir}")


if __name__ == "__main__":
    main()

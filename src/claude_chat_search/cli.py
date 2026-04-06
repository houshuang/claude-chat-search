import json
import sys
import time as _time
from datetime import datetime, timezone

import click

from .chunker import create_chunks
from .db import (
    DB_PATH,
    _fetchall,
    delete_session_data,
    get_connection,
    get_indexed_sessions,
    get_session,
    get_session_chunks,
    get_stats,
    get_subagent,
    get_subagent_count,
    get_subagents_for_session,
    get_topic_summary,
    init_db,
    insert_chunks,
    insert_session,
    insert_subagent,
    update_session_git_remote,
)
from .parser import (
    detect_git_remote,
    extract_session_metadata,
    find_project_dir,
    iter_jsonl_files,
    iter_subagent_files,
    parse_jsonl_file,
    parse_subagent_conversation,
    parse_subagent_metadata,
)


@click.group()
def cli():
    """Semantic search over Claude Code conversations."""
    pass


@cli.command()
def init():
    """Initialize the database and index all conversations."""
    conn = get_connection()
    init_db(conn)
    click.echo(f"Database created at {DB_PATH}")

    _run_index(conn, force=True)
    _run_embed(conn)

    stats = get_stats(conn)
    click.echo(f"\nDone: {stats['sessions']} sessions, {stats['chunks']} chunks, {stats['embedded']} embedded")
    conn.close()


@cli.command()
@click.option("--all", "index_all", is_flag=True, help="Re-index everything from scratch")
@click.option("--force", is_flag=True, help="Force re-index even if already indexed")
def index(index_all, force):
    """Index new or updated conversations."""
    conn = get_connection()
    init_db(conn)

    if index_all and force:
        click.echo("Re-indexing everything from scratch...")
        # Drop and recreate vec table — sqlite-vec doesn't reclaim space on DELETE
        conn.execute("DROP TABLE IF EXISTS vec_chunks")
        from .db import EMBEDDING_DIM
        conn.execute(f"""
            CREATE VIRTUAL TABLE vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{EMBEDDING_DIM}]
            )
        """)
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM sessions")
        conn.execute("PRAGMA foreign_keys=ON")
        from .vector_search import invalidate_cache
        invalidate_cache()

    _run_index(conn, force=force or index_all)
    _run_embed(conn)

    stats = get_stats(conn)
    click.echo(f"\nDone: {stats['sessions']} sessions, {stats['chunks']} chunks, {stats['embedded']} embedded")
    conn.close()


def _parse_date(value: str) -> str:
    """Parse a date string into ISO format for comparison.

    Accepts: YYYY-MM-DD, "3d" (3 days ago), "2w" (2 weeks ago), "1m" (1 month ago).
    """
    value = value.strip()
    if len(value) >= 2 and value[-1] in "dwm" and value[:-1].isdigit():
        n = int(value[:-1])
        now = datetime.now(timezone.utc)
        if value[-1] == "d":
            from datetime import timedelta
            dt = now - timedelta(days=n)
        elif value[-1] == "w":
            from datetime import timedelta
            dt = now - timedelta(weeks=n)
        elif value[-1] == "m":
            from datetime import timedelta
            dt = now - timedelta(days=n * 30)
        return dt.isoformat()
    # Assume ISO date
    if len(value) == 10:
        return value + "T00:00:00+00:00"
    return value


def _log_search(query: str, mode: str, results: list[dict], latency_ms: float,
                 filters: dict | None = None):
    """Append a JSON line to the search log with result details."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "mode": mode,
        "results": len(results),
        "ms": round(latency_ms, 1),
        "hits": [
            {
                "session": r["session_id"][:12],
                "score": round(r["score"], 4),
                "project": (r.get("project_path") or "").rsplit("/", 1)[-1],
                "turn": r.get("turn_number", 0),
            }
            for r in results
        ],
    }
    if filters:
        entry["filters"] = filters
    try:
        with open(DB_PATH.parent / "search.log", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Number of results")
@click.option("--project", "-p", default=None, help="Filter by project path substring")
@click.option("--branch", "-b", default=None, help="Filter by git branch name substring")
@click.option("--since", default=None, help="Only results after date (YYYY-MM-DD, 3d, 2w, 1m)")
@click.option("--before", default=None, help="Only results before date (YYYY-MM-DD, 3d, 2w, 1m)")
@click.option("--grep", "grep_mode", is_flag=True, help="Exact substring search (no semantic/FTS5)")
@click.option("--file", "file_mode", is_flag=True, help="Search by file path in session metadata")
@click.option("--rerank", "do_rerank", is_flag=True, help="Re-score with cross-encoder (slower, more accurate)")
@click.option("--expand", "do_expand", is_flag=True, help="LLM query expansion for better recall (~3-5s extra)")
def search(query, limit, project, branch, since, before, grep_mode, file_mode, do_rerank, do_expand):
    """Search past conversations."""
    from .search import file_search, grep_search, hybrid_search

    conn = get_connection()
    init_db(conn)

    since_iso = _parse_date(since) if since else None
    before_iso = _parse_date(before) if before else None

    mode = "file" if file_mode else "grep" if grep_mode else "hybrid"
    t0 = _time.monotonic()

    if file_mode:
        results = file_search(conn, query, limit=limit, project=project,
                              branch=branch, since=since_iso, before=before_iso)
    elif grep_mode:
        results = grep_search(conn, query, limit=limit, project=project,
                              branch=branch, since=since_iso, before=before_iso)
    else:
        results = hybrid_search(conn, query, limit=limit, project=project,
                                branch=branch, since=since_iso, before=before_iso,
                                do_rerank=do_rerank, expand=do_expand)

    filters = {}
    if project:
        filters["project"] = project
    if branch:
        filters["branch"] = branch
    if since:
        filters["since"] = since
    if before:
        filters["before"] = before
    _log_search(query, mode, results, (_time.monotonic() - t0) * 1000,
                filters or None)

    if not results:
        click.echo("No results found.")
        conn.close()
        return

    for i, r in enumerate(results, 1):
        click.echo(f"\n{'='*70}")

        # Header line with score and session
        click.echo(f"#{i}  score={r['score']:.4f}  session={r['session_id'][:12]}...")

        # Metadata
        meta_parts = []
        if r["project_path"]:
            meta_parts.append(f"project: {r['project_path']}")
        if r["slug"]:
            meta_parts.append(f"slug: {r['slug']}")
        if r["git_branch"]:
            meta_parts.append(f"branch: {r['git_branch']}")
        for m in meta_parts:
            click.echo(f"    {m}")

        # Topic summary
        summary = get_topic_summary(conn, r["session_id"])
        if summary:
            click.echo(f"    summary: {summary}")

        # Time and depth
        depth_parts = []
        if r["timestamp"]:
            depth_parts.append(f"time: {_format_time(r['timestamp'])}")
        if r.get("message_count"):
            depth_parts.append(f"{r['message_count']} messages")
        duration = _format_duration(r.get("first_message_at"), r.get("last_message_at"))
        if duration:
            depth_parts.append(duration)
        if depth_parts:
            click.echo(f"    {' · '.join(depth_parts)}")

        # Matching turn content
        click.echo(f"\n  [Turn {r['turn_number']}]")
        user = _truncate(r["user_content"], 300)
        assistant = _truncate(r["assistant_content"], 500)
        if user:
            click.echo(f"  User: {user}")
        if assistant:
            click.echo(f"  Assistant: {assistant}")

    conn.close()


@cli.command()
@click.argument("session_id")
@click.option("--turn", "-t", default=None, type=int, help="Highlight a specific turn number")
@click.option("--context", "-C", default=None, type=int, help="Show N turns around --turn (requires --turn)")
@click.option("--with-subagents", is_flag=True, help="Include subagent summaries")
def show(session_id, turn, context, with_subagents):
    """Show details of a specific session."""
    conn = get_connection()
    init_db(conn)

    session = _resolve_session(conn, session_id)
    if session is None:
        conn.close()
        return

    click.echo(f"Session: {session['session_id']}")
    click.echo(f"Project: {session['project_path']}")
    if session["slug"]:
        click.echo(f"Slug: {session['slug']}")
    if session["git_branch"]:
        click.echo(f"Branch: {session['git_branch']}")
    click.echo(f"Messages: {session['message_count']}")
    subagent_count = get_subagent_count(conn, session["session_id"])
    if subagent_count > 0:
        click.echo(f"Subagents: {subagent_count}")
    duration = _format_duration(session['first_message_at'], session['last_message_at'])
    time_str = f"{_format_time(session['first_message_at'])} → {_format_time(session['last_message_at'])}"
    if duration:
        time_str += f" ({duration})"
    click.echo(f"Time: {time_str}")

    chunks = get_session_chunks(conn, session["session_id"])

    # Filter to a window around --turn if --context is set
    if turn is not None and context is not None:
        chunks = [c for c in chunks
                  if abs(c["turn_number"] - turn) <= context]
        click.echo(f"\n--- showing turns {turn - context}..{turn + context} ({len(chunks)} chunks) ---\n")
    else:
        click.echo(f"\n--- {len(chunks)} chunks ---\n")

    for chunk in chunks:
        is_highlighted = turn is not None and chunk["turn_number"] == turn
        prefix = ">>>" if is_highlighted else "   "

        if is_highlighted:
            click.echo(f"{prefix} [Turn {chunk['turn_number']}]  {chunk['timestamp']}  ◀ MATCH")
        else:
            click.echo(f"{prefix} [Turn {chunk['turn_number']}]  {chunk['timestamp']}")

        max_user = 300 if is_highlighted else 150
        max_asst = 600 if is_highlighted else 250
        click.echo(f"{prefix}   User: {_truncate(chunk['user_content'], max_user)}")
        click.echo(f"{prefix}   Assistant: {_truncate(chunk['assistant_content'], max_asst)}")
        click.echo()

    if with_subagents and subagent_count > 0:
        subs = get_subagents_for_session(conn, session["session_id"])
        click.echo(f"\n--- Subagents ({len(subs)}) ---\n")
        for i, sub in enumerate(subs, 1):
            agent_type = sub["agent_type"] or "unknown"
            desc = sub["description"] or ""
            prompt_preview = _truncate(sub["first_prompt"] or "", 150)
            label = f"[{i}] agent-{sub['agent_id'][:8]} ({agent_type})"
            if desc:
                label += f" — {desc}"
            click.echo(label)
            if prompt_preview:
                click.echo(f"    Prompt: {prompt_preview}")
            click.echo(f"    {sub['message_count']} messages, {_format_size(sub['file_size'])}")
            click.echo()

    conn.close()


@cli.command()
@click.argument("session_id")
def subagents(session_id):
    """List all subagent conversations for a session."""
    conn = get_connection()
    init_db(conn)

    session = _resolve_session(conn, session_id)
    if session is None:
        conn.close()
        return

    subs = get_subagents_for_session(conn, session["session_id"])

    # Fall back to filesystem scan if DB has no records
    if not subs:
        project_dir = find_project_dir(session["project_path"])
        if project_dir:
            now = datetime.now(timezone.utc).isoformat()
            for sf in iter_subagent_files(session["session_id"], project_dir):
                meta = parse_subagent_metadata(sf["jsonl_path"], sf.get("meta_path"))
                sub_data = {
                    "agent_id": sf["agent_id"],
                    "parent_session_id": session["session_id"],
                    "file_size": sf["file_size"],
                    "jsonl_path": sf["jsonl_path"],
                    "indexed_at": now,
                    **meta,
                }
                insert_subagent(conn, sub_data)
                subs.append(sub_data)

    click.echo(f"Session: {session['session_id']}")
    if session["slug"]:
        click.echo(f"Slug: {session['slug']}")
    click.echo(f"Subagents: {len(subs)}")

    if not subs:
        click.echo("\nNo subagents found for this session.")
        conn.close()
        return

    click.echo(f"\n{'#':>3}  {'Agent ID':<12} {'Type':<20} {'Msgs':>5}  {'Size':>7}  First prompt")
    click.echo("-" * 90)
    for i, sub in enumerate(subs, 1):
        agent_id_short = sub["agent_id"][:8]
        agent_type = (sub.get("agent_type") or "unknown")[:20]
        msgs = sub.get("message_count", 0)
        size = _format_size(sub.get("file_size", 0))
        prompt = _truncate(sub.get("first_prompt") or "", 60)
        click.echo(f"{i:>3}  {agent_id_short:<12} {agent_type:<20} {msgs:>5}  {size:>7}  {prompt}")

    conn.close()


@cli.command()
@click.argument("session_id")
@click.argument("agent_id")
@click.option("--raw", is_flag=True, help="Show full text without truncation")
def subagent(session_id, agent_id, raw):
    """Show the full conversation of a specific subagent."""
    conn = get_connection()
    init_db(conn)

    session = _resolve_session(conn, session_id)
    if session is None:
        conn.close()
        return

    # Find subagent by partial ID match
    sub = get_subagent(conn, agent_id, session["session_id"])

    # Fall back to filesystem if not in DB
    jsonl_path = None
    resolved_agent_id = None
    if sub:
        jsonl_path = sub["jsonl_path"]
        resolved_agent_id = sub["agent_id"]
    else:
        project_dir = find_project_dir(session["project_path"])
        if project_dir:
            for sf in iter_subagent_files(session["session_id"], project_dir):
                if sf["agent_id"].startswith(agent_id):
                    jsonl_path = sf["jsonl_path"]
                    resolved_agent_id = sf["agent_id"]
                    break

    if not jsonl_path:
        click.echo(f"Subagent matching '{agent_id}' not found in session {session['session_id'][:12]}.")
        conn.close()
        return

    # Parse and display the conversation
    agent_type = sub["agent_type"] if sub else "unknown"
    turns = parse_subagent_conversation(jsonl_path)

    click.echo(f"Subagent: {resolved_agent_id} ({agent_type})")
    click.echo(f"Parent: {session['session_id'][:12]}... ({session.get('slug', '')})")
    click.echo(f"Turns: {len(turns)}")
    click.echo()

    max_user = 0 if raw else 500
    max_asst = 0 if raw else 800

    for t in turns:
        click.echo(f"    [Turn {t['turn_number']}]  {t.get('timestamp', '')}")
        user = t["user_content"]
        asst = t["assistant_content"]
        if max_user:
            user = _truncate(user, max_user)
        if max_asst:
            asst = _truncate(asst, max_asst)
        if user:
            click.echo(f"      User: {user}")
        if asst:
            click.echo(f"      Assistant: {asst}")
        if t.get("tools"):
            click.echo(f"      {t['tools']}")
        click.echo()

    conn.close()


def _resolve_session(conn, session_id: str) -> dict | None:
    """Resolve a session by exact or partial ID match. Prints errors on failure."""
    session = get_session(conn, session_id)
    if session is None:
        rows = _fetchall(
            conn,
            "SELECT * FROM sessions WHERE session_id LIKE ?",
            (f"{session_id}%",),
        )
        if len(rows) == 1:
            session = rows[0]
        elif len(rows) > 1:
            click.echo(f"Multiple sessions match '{session_id}':")
            for r in rows:
                click.echo(f"  {r['session_id']}  {r['project_path']}")
            return None
        else:
            click.echo(f"Session '{session_id}' not found.")
            return None
    return session


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes // 1024}KB"
    return f"{size_bytes // (1024 * 1024)}MB"


@cli.command()
@click.argument("session_id", required=False, default=None)
@click.option("--turns", "-n", default=20, help="Number of recent turns to show")
def recover(session_id, turns):
    """Recover recent context from a session (for use after compact).

    Outputs the most recent turns in a compact LLM-friendly format.
    Supports partial session ID matching.
    """
    conn = get_connection()
    init_db(conn)

    if session_id is None:
        click.echo("Usage: claude-chat-search recover SESSION_ID", err=True)
        conn.close()
        return

    # Support partial session ID matching
    session = get_session(conn, session_id)
    if session is None:
        rows = _fetchall(
            conn,
            "SELECT * FROM sessions WHERE session_id LIKE ?",
            (f"{session_id}%",),
        )
        if len(rows) == 1:
            session = rows[0]
        elif len(rows) > 1:
            click.echo(f"Multiple sessions match '{session_id}':", err=True)
            for r in rows:
                click.echo(f"  {r['session_id']}  {r['project_path']}", err=True)
            conn.close()
            return
        else:
            click.echo(f"Session '{session_id}' not found.", err=True)
            conn.close()
            return

    chunks = get_session_chunks(conn, session["session_id"])
    recent = chunks[-turns:] if len(chunks) > turns else chunks

    click.echo(f"Session: {session['session_id']}")
    click.echo(f"Project: {session['project_path']}")
    if session["slug"]:
        click.echo(f"Slug: {session['slug']}")
    duration = _format_duration(session["first_message_at"], session["last_message_at"])
    time_str = _format_time(session["first_message_at"])
    if duration:
        time_str += f" ({duration})"
    click.echo(f"Time: {time_str}")
    click.echo(f"Showing: last {len(recent)} of {len(chunks)} turns")
    click.echo("")

    for chunk in recent:
        click.echo(f"[Turn {chunk['turn_number']}]")
        if chunk["user_content"]:
            click.echo(f"U: {_truncate(chunk['user_content'], 500)}")
        if chunk["assistant_content"]:
            click.echo(f"A: {_truncate(chunk['assistant_content'], 800)}")
        click.echo("")

    conn.close()


@cli.command()
def reembed():
    """Migrate to new embedding model: drop old vectors, re-embed all chunks."""
    from .db import migrate_vec_table
    from .embedder import process_embeddings

    conn = get_connection()
    init_db(conn)

    stats_before = get_stats(conn)
    click.echo(f"Before: {stats_before['chunks']} chunks, {stats_before['embedded']} embedded")

    click.echo("Dropping old vec_chunks table and recreating with 384-dim...")
    count = migrate_vec_table(conn)
    click.echo(f"Marked {count} chunks for re-embedding.")

    click.echo("Re-embedding all chunks with paraphrase-multilingual-MiniLM-L12-v2 (via limbic)...")

    def progress(total):
        click.echo(f"  Embedded {total}/{count} chunks...", err=True)

    total = process_embeddings(conn, callback=progress)

    stats_after = get_stats(conn)
    click.echo(f"\nDone: {stats_after['embedded']}/{stats_after['chunks']} chunks embedded.")
    conn.close()


@cli.command()
@click.option("--all", "summarize_all", is_flag=True, help="Summarize all sessions without summaries")
@click.option("--limit", "-n", default=50, help="Max sessions to summarize (default: 50)")
def summarize(summarize_all, limit):
    """Generate topic summaries for sessions using Gemini Flash."""
    from .summarizer import summarize_sessions

    conn = get_connection()
    init_db(conn)

    stats = get_stats(conn)
    unsummarized = stats["sessions"] - stats["summarized"]
    click.echo(f"Sessions: {stats['sessions']} total, {stats['summarized']} summarized, {unsummarized} remaining")

    if unsummarized == 0:
        click.echo("All sessions already summarized.")
        conn.close()
        return

    effective_limit = unsummarized if summarize_all else limit
    click.echo(f"Summarizing up to {effective_limit} sessions...")

    def progress(sid, summary, count):
        click.echo(f"  [{count}] {sid[:12]}... {summary[:80]}{'...' if len(summary) > 80 else ''}", err=True)

    count = summarize_sessions(conn, limit=effective_limit, callback=progress)

    stats = get_stats(conn)
    click.echo(f"\nDone: summarized {count} sessions ({stats['summarized']}/{stats['sessions']} total)")
    conn.close()


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Number of results")
@click.option("--project", "-p", default=None, help="Filter by project path substring")
@click.option("--branch", "-b", default=None, help="Filter by git branch name substring")
@click.option("--since", default=None, help="Only results after date (YYYY-MM-DD, 3d, 2w, 1m)")
@click.option("--before", default=None, help="Only results before date (YYYY-MM-DD, 3d, 2w, 1m)")
def cross(query, limit, project, branch, since, before):
    """Search both chat history and research index."""
    from .cross_search import cross_search

    conn = get_connection()
    init_db(conn)

    since_iso = _parse_date(since) if since else None
    before_iso = _parse_date(before) if before else None

    t0 = _time.monotonic()
    results = cross_search(conn, query, limit=limit, project=project,
                           branch=branch, since=since_iso, before=before_iso)
    elapsed = (_time.monotonic() - t0) * 1000

    if not results:
        click.echo("No results found.")
        conn.close()
        return

    click.echo(f"Cross-search: {len(results)} results in {elapsed:.0f}ms\n")

    for i, r in enumerate(results, 1):
        click.echo(f"{'='*70}")
        source_label = f"[{r['source']}]"

        if r["source"] == "chat":
            click.echo(f"#{i}  {source_label}  score={r['score']:.4f}  session={r['session_id'][:12]}...")

            meta_parts = []
            if r.get("project_path"):
                meta_parts.append(f"project: {r['project_path']}")
            if r.get("slug"):
                meta_parts.append(f"slug: {r['slug']}")
            if r.get("git_branch"):
                meta_parts.append(f"branch: {r['git_branch']}")
            for m in meta_parts:
                click.echo(f"    {m}")

            # Show topic summary if available
            summary = get_topic_summary(conn, r["session_id"])
            if summary:
                click.echo(f"    summary: {summary}")

            depth_parts = []
            if r.get("timestamp"):
                depth_parts.append(f"time: {_format_time(r['timestamp'])}")
            if r.get("message_count"):
                depth_parts.append(f"{r['message_count']} messages")
            duration = _format_duration(r.get("first_message_at"), r.get("last_message_at"))
            if duration:
                depth_parts.append(duration)
            if depth_parts:
                click.echo(f"    {' · '.join(depth_parts)}")

            click.echo(f"\n  [Turn {r['turn_number']}]")
            user = _truncate(r.get("user_content", ""), 300)
            assistant = _truncate(r.get("assistant_content", ""), 500)
            if user:
                click.echo(f"  User: {user}")
            if assistant:
                click.echo(f"  Assistant: {assistant}")

        elif r["source"] == "research":
            from pathlib import Path
            fname = Path(r["source_file"]).name
            sec = f" > {r['section']}" if r.get("section") else ""
            click.echo(f"#{i}  {source_label}  score={r['score']:.4f}  {fname}{sec}")
            click.echo(f"    file: {r['source_file']}")
            content_preview = _truncate(r.get("content", ""), 400)
            if content_preview:
                click.echo(f"  {content_preview}")

        click.echo()

    conn.close()


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Number of results to show")
@click.option("--project", "-p", default=None, help="Filter by project path substring")
@click.option("--branch", "-b", default=None, help="Filter by git branch name substring")
@click.option("--since", default=None, help="Only results after date (YYYY-MM-DD, 3d, 2w, 1m)")
@click.option("--before", default=None, help="Only results before date (YYYY-MM-DD, 3d, 2w, 1m)")
@click.option("--fork", "do_fork", is_flag=True, help="Fork the session instead of resuming in-place")
def resume(query, limit, project, branch, since, before, do_fork):
    """Search for a session and resume it in Claude Code.

    Runs a semantic search, shows matching sessions, and lets you pick
    one to resume via `claude --resume`.
    """
    import os
    import shutil

    from .search import hybrid_search

    conn = get_connection()
    init_db(conn)

    since_iso = _parse_date(since) if since else None
    before_iso = _parse_date(before) if before else None

    results = hybrid_search(conn, query, limit=limit * 3, project=project,
                            branch=branch, since=since_iso, before=before_iso)

    if not results:
        click.echo("No results found.")
        conn.close()
        return

    # Deduplicate by session_id, keeping highest-scoring chunk per session
    seen = {}
    for r in results:
        sid = r["session_id"]
        if sid not in seen or r["score"] > seen[sid]["score"]:
            seen[sid] = r
    unique = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:limit]

    # Display pick list
    for i, r in enumerate(unique, 1):
        session = get_session(conn, r["session_id"])
        slug = session["slug"] or ""
        cwd = session.get("cwd") or ""
        summary = get_topic_summary(conn, r["session_id"]) or ""
        msg_count = session["message_count"]

        time_ago = _format_time(session["last_message_at"])
        duration = _format_duration(session["first_message_at"], session["last_message_at"])

        click.echo(f"\n  {i}. [{r['session_id'][:8]}] {slug}")
        meta_parts = []
        if time_ago:
            meta_parts.append(time_ago)
        if duration:
            meta_parts.append(duration)
        meta_parts.append(f"{msg_count} msgs")
        if session.get("git_branch"):
            meta_parts.append(f"branch: {session['git_branch']}")
        click.echo(f"     {' · '.join(meta_parts)}")
        if cwd:
            click.echo(f"     cwd: {cwd}")
        elif session.get("project_path"):
            click.echo(f"     project: {session['project_path']}")
        if summary:
            click.echo(f"     {summary}")
        else:
            # Show a snippet from the matching chunk
            snippet = _truncate(r.get("user_content", ""), 120)
            if snippet:
                click.echo(f"     \"{snippet}\"")

    click.echo()

    # Prompt for selection
    if len(unique) == 1:
        choice = 1
        click.echo(f"Only one match — selecting #{choice}")
    else:
        try:
            raw = click.prompt("Pick a session to resume (number, or q to quit)", type=str)
        except (click.Abort, EOFError):
            conn.close()
            return
        if raw.strip().lower() in ("q", "quit", ""):
            conn.close()
            return
        try:
            choice = int(raw)
        except ValueError:
            click.echo(f"Invalid selection: {raw}")
            conn.close()
            return
        if choice < 1 or choice > len(unique):
            click.echo(f"Selection out of range (1-{len(unique)})")
            conn.close()
            return

    selected = unique[choice - 1]
    session = get_session(conn, selected["session_id"])
    session_id = session["session_id"]
    cwd = session.get("cwd")
    conn.close()

    # Find claude binary
    claude_bin = shutil.which("claude")
    if not claude_bin:
        click.echo("Error: 'claude' not found in PATH")
        return

    # Build the command
    args = ["claude", "--resume", session_id]
    if do_fork:
        args.append("--fork-session")

    slug = session.get("slug") or session_id[:12]
    action = "Forking" if do_fork else "Resuming"
    click.echo(f"\n{action} session: {slug}")

    # cd to the session's working directory if available
    if cwd and os.path.isdir(cwd):
        os.chdir(cwd)
        click.echo(f"Changed directory to: {cwd}")

    os.execvp(claude_bin, args)


@cli.group()
def daemon():
    """Manage the indexer daemon."""
    pass


@daemon.command("run")
def daemon_run():
    """Run daemon in the foreground (for launchd or direct use)."""
    from .daemon import run
    run()


@daemon.command("start")
def daemon_start():
    """Start daemon as a detached background process."""
    from .daemon import start
    start()


@daemon.command("stop")
def daemon_stop():
    """Stop the running daemon."""
    from .daemon import stop
    stop()


@daemon.command("status")
def daemon_status():
    """Show daemon status and recent log."""
    from .daemon import status
    status()


def _run_index(conn, force: bool = False) -> int:
    """Index JSONL files. Returns number of sessions indexed."""
    files = iter_jsonl_files()
    indexed = get_indexed_sessions(conn)
    now = datetime.now(timezone.utc).isoformat()

    new_count = 0
    skip_count = 0

    with click.progressbar(files, label="Indexing sessions", file=sys.stderr) as bar:
        for file_info in bar:
            sid = file_info["session_id"]
            mtime_str = datetime.fromtimestamp(file_info["mtime"], tz=timezone.utc).isoformat()

            if not force and sid in indexed:
                existing_time = indexed[sid]
                if existing_time and existing_time >= mtime_str:
                    skip_count += 1
                    continue

            # Re-index this session (preserve topic_summary across re-index)
            saved_summary = None
            if sid in indexed:
                saved_summary = get_topic_summary(conn, sid)
                delete_session_data(conn, sid)

            try:
                session_data = parse_jsonl_file(file_info["path"])
            except Exception as e:
                click.echo(f"\nError parsing {file_info['path']}: {e}", err=True)
                continue

            if not session_data["messages"]:
                continue

            session_data["project_path"] = file_info["project_path"]
            session_data["indexed_at"] = now
            session_data["parent_session_id"] = file_info.get("parent_session_id")

            # Extract structured metadata from tool calls
            metadata = extract_session_metadata(session_data["messages"])
            session_data.update(metadata)

            insert_session(conn, session_data)
            if saved_summary:
                from .db import update_topic_summary
                update_topic_summary(conn, sid, saved_summary)
            chunks = create_chunks(session_data)
            if chunks:
                insert_chunks(conn, chunks)

            new_count += 1

    click.echo(f"Indexed {new_count} sessions ({skip_count} skipped)", err=True)

    # Backfill git remotes for sessions missing them
    remote_rows = list(conn.execute(
        "SELECT DISTINCT project_path FROM sessions WHERE git_remote IS NULL"
    ))
    if remote_rows:
        remote_count = 0
        for (project_path,) in remote_rows:
            remote = detect_git_remote(project_path)
            if remote:
                updated = update_session_git_remote(conn, project_path, remote)
                remote_count += updated
        if remote_count:
            click.echo(f"Backfilled git remote for {remote_count} sessions", err=True)

    # Index subagent metadata
    sub_count = 0
    now = datetime.now(timezone.utc).isoformat()
    all_sessions = list(conn.execute("SELECT session_id, project_path FROM sessions"))
    for (sid, proj_path) in all_sessions:
        project_dir = find_project_dir(proj_path)
        if not project_dir:
            continue
        for sf in iter_subagent_files(sid, project_dir):
            # Check if already indexed with same mtime
            existing = conn.execute(
                "SELECT indexed_at FROM subagents WHERE agent_id = ?",
                (sf["agent_id"],),
            ).fetchone()
            if existing and not force:
                continue
            meta = parse_subagent_metadata(sf["jsonl_path"], sf.get("meta_path"))
            insert_subagent(conn, {
                "agent_id": sf["agent_id"],
                "parent_session_id": sid,
                "file_size": sf["file_size"],
                "jsonl_path": sf["jsonl_path"],
                "indexed_at": now,
                **meta,
            })
            sub_count += 1
    if sub_count:
        click.echo(f"Indexed {sub_count} subagent metadata records", err=True)

    return new_count


def _run_embed(conn) -> int:
    """Generate embeddings for unembedded chunks."""
    from .embedder import process_embeddings

    stats = get_stats(conn)
    unembedded = stats["chunks"] - stats["embedded"]
    if unembedded == 0:
        click.echo("All chunks already embedded.", err=True)
        return 0

    click.echo(f"Embedding {unembedded} chunks...", err=True)

    def progress(total):
        click.echo(f"  Embedded {total} chunks so far...", err=True)

    total = process_embeddings(conn, callback=progress)
    click.echo(f"Embedded {total} chunks.", err=True)
    return total


def _truncate(text: str, max_len: int) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    # Try to break at a sentence or word boundary
    truncated = text[:max_len]
    last_period = truncated.rfind(". ")
    if last_period > max_len * 0.6:
        return truncated[:last_period + 1]
    last_space = truncated.rfind(" ")
    if last_space > max_len * 0.8:
        return truncated[:last_space] + "..."
    return truncated + "..."


def _format_time(iso_str: str | None) -> str:
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return iso_str


def _format_duration(start: str | None, end: str | None) -> str:
    if not start or not end:
        return ""
    try:
        dt_start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        dt_end = datetime.fromisoformat(end.replace("Z", "+00:00"))
        delta = dt_end - dt_start
        minutes = int(delta.total_seconds() / 60)
        if minutes < 1:
            return "<1min"
        if minutes < 60:
            return f"{minutes}min"
        hours = minutes // 60
        mins = minutes % 60
        if mins == 0:
            return f"{hours}h"
        return f"{hours}h{mins}m"
    except (ValueError, AttributeError):
        return ""


if __name__ == "__main__":
    cli()

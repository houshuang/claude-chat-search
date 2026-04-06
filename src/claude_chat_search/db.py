import struct
from pathlib import Path

import apsw
import sqlite_vec

DB_DIR = Path.home() / ".claude-chat-search"
DB_PATH = DB_DIR / "index.db"
EMBEDDING_DIM = 384


def get_connection() -> apsw.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = apsw.Connection(str(DB_PATH))
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_vec.loadable_path())
    conn.enable_load_extension(False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except apsw.CantOpenError:
        pass  # directory may be locked; fall back to default journal mode
    conn.execute("PRAGMA foreign_keys=ON")
    conn.setbusytimeout(5000)
    return conn


def _fetchall(conn: apsw.Connection, sql: str, bindings=None) -> list[dict]:
    cursor = conn.execute(sql, bindings or ())
    try:
        desc = cursor.getdescription()
    except apsw.ExecutionCompleteError:
        return []
    cols = [d[0] for d in desc]
    return [dict(zip(cols, row)) for row in cursor]


def _fetchone(conn: apsw.Connection, sql: str, bindings=None) -> dict | None:
    rows = _fetchall(conn, sql, bindings)
    return rows[0] if rows else None


def init_db(conn: apsw.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            project_path TEXT,
            slug TEXT,
            git_branch TEXT,
            first_message_at TEXT,
            last_message_at TEXT,
            message_count INTEGER,
            indexed_at TEXT,
            files_touched TEXT,
            tools_used TEXT,
            commands_run TEXT,
            parent_session_id TEXT
        )
    """)

    # Add columns if upgrading from older schema
    for col, col_type in [
        ("files_touched", "TEXT"),
        ("tools_used", "TEXT"),
        ("commands_run", "TEXT"),
        ("parent_session_id", "TEXT"),
        ("topic_summary", "TEXT"),
        ("git_remote", "TEXT"),
        ("cwd", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {col_type}")
        except apsw.SQLError:
            pass  # column already exists

    # Subagent metadata table (lightweight — no chunks, no embeddings)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subagents (
            agent_id TEXT PRIMARY KEY,
            parent_session_id TEXT REFERENCES sessions(session_id),
            agent_type TEXT,
            description TEXT,
            message_count INTEGER,
            file_size INTEGER,
            first_prompt TEXT,
            first_message_at TEXT,
            last_message_at TEXT,
            jsonl_path TEXT,
            indexed_at TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            session_id TEXT REFERENCES sessions(session_id),
            user_content TEXT,
            assistant_content TEXT,
            combined_text TEXT,
            timestamp TEXT,
            turn_number INTEGER,
            token_estimate INTEGER,
            embedded INTEGER DEFAULT 0
        )
    """)

    # FTS5 table
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                combined_text,
                content='chunks',
                content_rowid='id'
            )
        """)
    except apsw.SQLError:
        pass  # already exists

    # FTS sync triggers
    for trigger_sql in [
        """CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, combined_text) VALUES (new.id, new.combined_text);
        END""",
        """CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, combined_text) VALUES('delete', old.id, old.combined_text);
        END""",
        """CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, combined_text) VALUES('delete', old.id, old.combined_text);
            INSERT INTO chunks_fts(rowid, combined_text) VALUES (new.id, new.combined_text);
        END""",
    ]:
        conn.execute(trigger_sql)

    # Vec table
    try:
        conn.execute(f"""
            CREATE VIRTUAL TABLE vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{EMBEDDING_DIM}]
            )
        """)
    except apsw.SQLError:
        pass  # already exists


def serialize_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def insert_session(conn: apsw.Connection, session: dict) -> None:
    conn.execute(
        """INSERT INTO sessions
           (session_id, project_path, slug, git_branch,
            first_message_at, last_message_at, message_count, indexed_at,
            files_touched, tools_used, commands_run, parent_session_id, cwd)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
            project_path=excluded.project_path,
            slug=excluded.slug,
            git_branch=excluded.git_branch,
            first_message_at=excluded.first_message_at,
            last_message_at=excluded.last_message_at,
            message_count=excluded.message_count,
            indexed_at=excluded.indexed_at,
            files_touched=excluded.files_touched,
            tools_used=excluded.tools_used,
            commands_run=excluded.commands_run,
            parent_session_id=excluded.parent_session_id,
            cwd=excluded.cwd""",
        (
            session["session_id"],
            session["project_path"],
            session.get("slug"),
            session.get("git_branch"),
            session.get("first_message_at"),
            session.get("last_message_at"),
            session.get("message_count", 0),
            session.get("indexed_at"),
            session.get("files_touched"),
            session.get("tools_used"),
            session.get("commands_run"),
            session.get("parent_session_id"),
            session.get("cwd"),
        ),
    )


def insert_chunks(conn: apsw.Connection, chunks: list[dict]) -> list[int]:
    ids = []
    for chunk in chunks:
        conn.execute(
            """INSERT INTO chunks
               (session_id, user_content, assistant_content, combined_text,
                timestamp, turn_number, token_estimate, embedded)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                chunk["session_id"],
                chunk["user_content"],
                chunk["assistant_content"],
                chunk["combined_text"],
                chunk.get("timestamp"),
                chunk.get("turn_number"),
                chunk.get("token_estimate", 0),
            ),
        )
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        ids.append(row_id)
    return ids


def insert_embeddings(conn: apsw.Connection, chunk_ids: list[int], embeddings: list[list[float]]) -> None:
    for chunk_id, emb in zip(chunk_ids, embeddings):
        conn.execute(
            "INSERT OR REPLACE INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize_embedding(emb)),
        )
        conn.execute("UPDATE chunks SET embedded = 1 WHERE id = ?", (chunk_id,))
    # Invalidate numpy search cache so it picks up new embeddings
    from .vector_search import invalidate_cache
    invalidate_cache()


def delete_session_data(conn: apsw.Connection, session_id: str) -> None:
    rows = list(conn.execute("SELECT id FROM chunks WHERE session_id = ?", (session_id,)))
    for (cid,) in rows:
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
    conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM subagents WHERE parent_session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    from .vector_search import invalidate_cache
    invalidate_cache()


def get_unembedded_chunks(conn: apsw.Connection, batch_size: int = 100) -> list[dict]:
    return _fetchall(
        conn,
        "SELECT id, combined_text FROM chunks WHERE embedded = 0 LIMIT ?",
        (batch_size,),
    )


def vector_search(conn: apsw.Connection, query_embedding: list[float], limit: int = 20) -> list[dict]:
    rows = list(conn.execute(
        """SELECT chunk_id, distance
           FROM vec_chunks
           WHERE embedding MATCH ?
           ORDER BY distance
           LIMIT ?""",
        (serialize_embedding(query_embedding), limit),
    ))
    return [{"chunk_id": r[0], "distance": r[1]} for r in rows]


def _sanitize_fts_query(query: str) -> str:
    """Build a safe FTS5 query from user input.

    Uses limbic's sanitization: extract unicode word tokens, filter noise,
    quote each to prevent reserved word interpretation (AND/OR/NOT/NEAR).
    """
    from limbic.amygdala.search import FTS5Index
    return FTS5Index._sanitize_query(query)


def fts_search(conn: apsw.Connection, query: str, limit: int = 20) -> list[dict]:
    safe_query = _sanitize_fts_query(query)
    if not safe_query:
        return []
    try:
        rows = list(conn.execute(
            """SELECT rowid, rank
               FROM chunks_fts
               WHERE combined_text MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, limit),
        ))
    except apsw.SQLError:
        # If FTS5 still fails on unusual input, return empty and let vector search carry
        return []
    return [{"chunk_id": r[0], "rank": r[1]} for r in rows]


def text_search(conn: apsw.Connection, query: str, limit: int = 50) -> list[dict]:
    """Exact substring search via SQL LIKE. No FTS5, no embeddings."""
    if not query.strip():
        return []
    return _fetchall(
        conn,
        """SELECT c.*, s.project_path, s.slug, s.git_branch,
                  s.message_count, s.first_message_at, s.last_message_at
           FROM chunks c
           JOIN sessions s ON c.session_id = s.session_id
           WHERE c.combined_text LIKE '%' || ? || '%'
           ORDER BY c.id DESC
           LIMIT ?""",
        (query, limit),
    )


def file_search(conn: apsw.Connection, query: str, limit: int = 20) -> list[dict]:
    """Search sessions by file path in files_touched metadata."""
    if not query.strip():
        return []
    return _fetchall(
        conn,
        """SELECT s.*, NULL as id, NULL as user_content, NULL as assistant_content,
                  NULL as combined_text, s.first_message_at as timestamp,
                  0 as turn_number, 0 as token_estimate
           FROM sessions s
           WHERE s.files_touched LIKE '%' || ? || '%'
           ORDER BY s.last_message_at DESC
           LIMIT ?""",
        (query, limit),
    )


def get_chunks_by_ids(conn: apsw.Connection, chunk_ids: list[int]) -> list[dict]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" for _ in chunk_ids)
    return _fetchall(
        conn,
        f"""SELECT c.*, s.project_path, s.slug, s.git_branch,
                   s.message_count, s.first_message_at, s.last_message_at
            FROM chunks c
            JOIN sessions s ON c.session_id = s.session_id
            WHERE c.id IN ({placeholders})""",
        chunk_ids,
    )


def get_session(conn: apsw.Connection, session_id: str) -> dict | None:
    return _fetchone(conn, "SELECT * FROM sessions WHERE session_id = ?", (session_id,))


def get_session_chunks(conn: apsw.Connection, session_id: str) -> list[dict]:
    return _fetchall(
        conn,
        "SELECT * FROM chunks WHERE session_id = ? ORDER BY turn_number",
        (session_id,),
    )


def get_indexed_sessions(conn: apsw.Connection) -> dict[str, str]:
    rows = list(conn.execute("SELECT session_id, indexed_at FROM sessions"))
    return {r[0]: r[1] for r in rows}


def get_stats(conn: apsw.Connection) -> dict:
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedded = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 1").fetchone()[0]
    summarized = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE topic_summary IS NOT NULL"
    ).fetchone()[0]
    return {"sessions": sessions, "chunks": chunks, "embedded": embedded,
            "summarized": summarized}


def get_sessions_without_summary(conn: apsw.Connection, limit: int = 50) -> list[dict]:
    """Get sessions that don't have a topic summary yet, ordered by most recent first."""
    return _fetchall(
        conn,
        """SELECT session_id, project_path, slug, message_count
           FROM sessions
           WHERE topic_summary IS NULL AND message_count > 2
           ORDER BY last_message_at DESC
           LIMIT ?""",
        (limit,),
    )


def update_topic_summary(conn: apsw.Connection, session_id: str, summary: str) -> None:
    conn.execute(
        "UPDATE sessions SET topic_summary = ? WHERE session_id = ?",
        (summary, session_id),
    )


def get_topic_summary(conn: apsw.Connection, session_id: str) -> str | None:
    row = conn.execute(
        "SELECT topic_summary FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return row[0] if row else None


def insert_subagent(conn: apsw.Connection, data: dict) -> None:
    conn.execute(
        """INSERT INTO subagents
           (agent_id, parent_session_id, agent_type, description,
            message_count, file_size, first_prompt,
            first_message_at, last_message_at, jsonl_path, indexed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(agent_id) DO UPDATE SET
            parent_session_id=excluded.parent_session_id,
            agent_type=excluded.agent_type,
            description=excluded.description,
            message_count=excluded.message_count,
            file_size=excluded.file_size,
            first_prompt=excluded.first_prompt,
            first_message_at=excluded.first_message_at,
            last_message_at=excluded.last_message_at,
            jsonl_path=excluded.jsonl_path,
            indexed_at=excluded.indexed_at""",
        (
            data["agent_id"],
            data["parent_session_id"],
            data.get("agent_type"),
            data.get("description"),
            data.get("message_count", 0),
            data.get("file_size", 0),
            data.get("first_prompt"),
            data.get("first_message_at"),
            data.get("last_message_at"),
            data.get("jsonl_path"),
            data.get("indexed_at"),
        ),
    )


def get_subagents_for_session(conn: apsw.Connection, parent_session_id: str) -> list[dict]:
    return _fetchall(
        conn,
        """SELECT * FROM subagents
           WHERE parent_session_id = ?
           ORDER BY first_message_at""",
        (parent_session_id,),
    )


def get_subagent(conn: apsw.Connection, agent_id_prefix: str, parent_session_id: str) -> dict | None:
    return _fetchone(
        conn,
        """SELECT * FROM subagents
           WHERE agent_id LIKE ? AND parent_session_id = ?""",
        (f"{agent_id_prefix}%", parent_session_id),
    )


def get_subagent_count(conn: apsw.Connection, parent_session_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM subagents WHERE parent_session_id = ?",
        (parent_session_id,),
    ).fetchone()
    return row[0] if row else 0


def get_git_remote_for_project(conn: apsw.Connection, project_path: str) -> str | None:
    """Get git_remote for a project path (from any session using that path)."""
    row = conn.execute(
        "SELECT git_remote FROM sessions WHERE project_path = ? AND git_remote IS NOT NULL LIMIT 1",
        (project_path,),
    ).fetchone()
    return row[0] if row else None


def get_project_paths_for_remote(conn: apsw.Connection, git_remote: str) -> list[str]:
    """Get all project paths sharing a git remote."""
    rows = list(conn.execute(
        "SELECT DISTINCT project_path FROM sessions WHERE git_remote = ?",
        (git_remote,),
    ))
    return [r[0] for r in rows]


def update_session_git_remote(conn: apsw.Connection, project_path: str, git_remote: str) -> int:
    """Set git_remote for all sessions with a given project_path. Returns rows updated."""
    conn.execute(
        "UPDATE sessions SET git_remote = ? WHERE project_path = ? AND (git_remote IS NULL OR git_remote != ?)",
        (git_remote, project_path, git_remote),
    )
    return conn.changes()


def migrate_vec_table(conn: apsw.Connection) -> int:
    """Drop old vec_chunks table, recreate with current EMBEDDING_DIM, mark all chunks for re-embedding.

    Returns the number of chunks marked for re-embedding.
    """
    conn.execute("DROP TABLE IF EXISTS vec_chunks")
    conn.execute(f"""
        CREATE VIRTUAL TABLE vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIM}]
        )
    """)
    conn.execute("UPDATE chunks SET embedded = 0")
    from .vector_search import invalidate_cache
    invalidate_cache()
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

import hashlib
import os
import struct
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import apsw
import sqlite_vec

from .models import LEGACY_MODEL, MODELS, configured_model
from .paths import DATA_DIR

BUSY_TIMEOUT_MS = 30000

_configured_db_path = os.environ.get("CHAT_SEARCH_DB_PATH")
DB_PATH = (
    Path(_configured_db_path).expanduser()
    if _configured_db_path
    else DATA_DIR / "index.db"
)
DB_DIR = DB_PATH.parent


# Bump when init_db() gains a migration; init_db() is a no-op once the
# database's user_version has reached it.
SCHEMA_VERSION = 3


def embedding_dim() -> int:
    """Vector width of the configured embedding model."""
    return configured_model().dim


def _load_vec(conn: apsw.Connection) -> None:
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_vec.loadable_path())
    conn.enable_load_extension(False)


@contextmanager
def write_transaction(conn: apsw.Connection):
    """Write transaction that takes the write lock up front.

    A deferred transaction that reads before it writes cannot wait for the lock:
    if another connection commits in between, SQLite returns BUSY immediately and
    the busy timeout never applies. BEGIN IMMEDIATE waits for the lock instead.
    Nested use falls back to a savepoint inside the outer transaction.
    """
    if conn.in_transaction:
        with conn:
            yield
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def get_connection(readonly: bool = False) -> apsw.Connection:
    if readonly:
        conn = apsw.Connection(str(DB_PATH), flags=apsw.SQLITE_OPEN_READONLY)
        _load_vec(conn)
        conn.setbusytimeout(BUSY_TIMEOUT_MS)
        conn.execute("PRAGMA cache_size=-64000")
        return conn

    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = apsw.Connection(str(DB_PATH))
    _load_vec(conn)
    conn.setbusytimeout(BUSY_TIMEOUT_MS)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except apsw.CantOpenError:
        pass  # directory may be locked; fall back to default journal mode
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_read_connection() -> apsw.Connection:
    """Open the index read-only, migrating it first only if it is behind."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"No index at {DB_PATH}; run `claude-chat-search init` first"
        )
    conn = get_connection(readonly=True)
    if schema_version(conn) >= SCHEMA_VERSION:
        return conn
    conn.close()
    writer = get_connection()
    try:
        init_db(writer)
    finally:
        writer.close()
    return get_connection(readonly=True)


def schema_version(conn: apsw.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def backup_database(conn: apsw.Connection, destination: Path | None = None) -> Path:
    """Create a consistent online backup without mutating or stopping the source DB."""
    if destination is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = DB_DIR / "backups" / f"index-{stamp}.db"
    destination = Path(destination).expanduser()
    if destination.exists():
        raise FileExistsError(f"Backup destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    backup_conn = apsw.Connection(str(destination))
    backup = backup_conn.backup("main", conn, "main")
    try:
        backup.step(-1)
    finally:
        backup.finish()
    try:
        # The copy inherits WAL mode; a self-contained single file is what a
        # backup should be.
        backup_conn.execute("PRAGMA journal_mode=DELETE")
    finally:
        backup_conn.close()
    return destination


def integrity_check(path: Path) -> list[str]:
    """Run PRAGMA integrity_check on a database file; returns [] when healthy."""
    conn = apsw.Connection(str(path), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        _load_vec(conn)
        rows = [row[0] for row in conn.execute("PRAGMA integrity_check")]
    finally:
        conn.close()
    return [] if rows == ["ok"] else rows


def prune_backups(directory: Path, keep: int) -> list[Path]:
    """Delete all but the newest `keep` timestamped backups in directory."""
    backups = sorted(Path(directory).glob("index-*.db"), reverse=True)
    removed = backups[keep:]
    for path in removed:
        path.unlink()
        for suffix in ("-wal", "-shm"):
            Path(str(path) + suffix).unlink(missing_ok=True)
    return removed


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
    """Create or migrate the schema.  Cheap no-op when already current."""
    if schema_version(conn) >= SCHEMA_VERSION:
        return
    with write_transaction(conn):
        _migrate(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _migrate(conn: apsw.Connection) -> None:
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
        ("source", "TEXT NOT NULL DEFAULT 'claude'"),
        ("native_session_id", "TEXT"),
        ("transcript_path", "TEXT"),
        ("transcript_mtime", "REAL"),
        ("transcript_size", "INTEGER"),
        ("thread_kind", "TEXT NOT NULL DEFAULT 'user'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {col_type}")
        except apsw.SQLError:
            pass  # column already exists

    # Existing rows predate multi-source indexing and are all Claude sessions.
    conn.execute("UPDATE sessions SET source = 'claude' WHERE source IS NULL OR source = ''")
    conn.execute(
        "UPDATE sessions SET native_session_id = session_id "
        "WHERE native_session_id IS NULL OR native_session_id = ''"
    )
    conn.execute("UPDATE sessions SET thread_kind = 'user' WHERE thread_kind IS NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS sessions_source_idx ON sessions(source)")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS sessions_source_native_idx "
        "ON sessions(source, native_session_id)"
    )

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
    ]:
        conn.execute(trigger_sql)

    # Only combined_text is indexed; marking a chunk embedded must not rewrite
    # its FTS entry.
    conn.execute("DROP TRIGGER IF EXISTS chunks_au")
    conn.execute("""CREATE TRIGGER chunks_au AFTER UPDATE OF combined_text ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, combined_text) VALUES('delete', old.id, old.combined_text);
            INSERT INTO chunks_fts(rowid, combined_text) VALUES (new.id, new.combined_text);
        END""")

    try:
        conn.execute("ALTER TABLE chunks ADD COLUMN content_hash TEXT")
    except apsw.SQLError:
        pass  # column already exists
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_session_idx ON chunks(session_id, embedded)"
    )

    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    had_vectors = _table_exists(conn, "vec_chunks")
    if not had_vectors:
        _create_vec_table(conn, embedding_dim())
    if get_meta(conn, "embedding_model") is None:
        # Every index built before the meta table used the legacy model.
        model = LEGACY_MODEL if had_vectors else configured_model().name
        set_meta(conn, "embedding_model", model)
        set_meta(conn, "embedding_dim", str(MODELS[model].dim))

    # Vectors whose chunk is gone, and chunks marked embedded without a vector.
    conn.execute(
        "DELETE FROM vec_chunks WHERE chunk_id NOT IN (SELECT id FROM chunks)"
    )
    conn.execute(
        "UPDATE chunks SET embedded = 0 WHERE embedded = 1 "
        "AND id NOT IN (SELECT chunk_id FROM vec_chunks)"
    )


def _table_exists(conn: apsw.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone() is not None


def _create_vec_table(conn: apsw.Connection, dim: int) -> None:
    conn.execute(f"""
        CREATE VIRTUAL TABLE vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{dim}]
        )
    """)


def get_meta(conn: apsw.Connection, key: str) -> str | None:
    if not _table_exists(conn, "meta"):
        return None
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: apsw.Connection, key: str, value: str | None) -> None:
    if value is None:
        conn.execute("DELETE FROM meta WHERE key = ?", (key,))
    else:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def stored_embedding_model(conn: apsw.Connection) -> str | None:
    """The model that produced the vectors in vec_chunks."""
    return get_meta(conn, "embedding_model")


def embedding_mismatch(conn: apsw.Connection) -> str | None:
    """Why stored vectors cannot be searched with the configured model, or None."""
    stored = stored_embedding_model(conn)
    configured = configured_model().name
    if stored == configured:
        return None
    return (
        f"the index holds vectors from {stored or 'an unknown model'} but "
        f"{configured} is configured; semantic search is off until "
        "`claude-chat-search migrate-embeddings` has run (keyword search still works)"
    )


def chunk_hash(combined_text: str) -> str:
    return hashlib.sha256(combined_text.encode("utf-8", "surrogatepass")).hexdigest()


def serialize_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def insert_session(conn: apsw.Connection, session: dict) -> None:
    conn.execute(
        """INSERT INTO sessions
           (session_id, project_path, slug, git_branch,
            first_message_at, last_message_at, message_count, indexed_at,
            files_touched, tools_used, commands_run, parent_session_id, cwd,
            source, native_session_id, transcript_path, transcript_mtime,
            transcript_size, thread_kind)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            cwd=excluded.cwd,
            source=excluded.source,
            native_session_id=excluded.native_session_id,
            transcript_path=excluded.transcript_path,
            transcript_mtime=excluded.transcript_mtime,
            transcript_size=excluded.transcript_size,
            thread_kind=excluded.thread_kind""",
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
            session.get("source", "claude"),
            session.get("native_session_id", session["session_id"]),
            session.get("transcript_path"),
            session.get("transcript_mtime"),
            session.get("transcript_size"),
            session.get("thread_kind", "user"),
        ),
    )


def update_session_fingerprint(conn: apsw.Connection, session_id: str, file_info: dict) -> None:
    conn.execute(
        """UPDATE sessions
           SET transcript_path = ?, transcript_mtime = ?, transcript_size = ?
           WHERE session_id = ?""",
        (
            str(file_info["path"]),
            file_info.get("mtime"),
            file_info.get("size"),
            session_id,
        ),
    )


def insert_chunks(conn: apsw.Connection, chunks: list[dict]) -> list[int]:
    ids = []
    for chunk in chunks:
        conn.execute(
            """INSERT INTO chunks
               (session_id, user_content, assistant_content, combined_text,
                timestamp, turn_number, token_estimate, embedded, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (
                chunk["session_id"],
                chunk["user_content"],
                chunk["assistant_content"],
                chunk["combined_text"],
                chunk.get("timestamp"),
                chunk.get("turn_number"),
                chunk.get("token_estimate", 0),
                chunk_hash(chunk["combined_text"]),
            ),
        )
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        ids.append(row_id)
    return ids


def insert_embeddings(conn: apsw.Connection, chunk_ids: list[int], embeddings: list[list[float]]) -> None:
    with write_transaction(conn):
        for chunk_id, emb in zip(chunk_ids, embeddings):
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, serialize_embedding(emb)),
                )
            except apsw.SQLError as error:
                # sqlite-vec can surface a primary-key race as SQLError even for
                # INSERT OR REPLACE.  If another indexer completed this exact chunk,
                # accept its vector; otherwise preserve the real failure.
                duplicate = "UNIQUE constraint failed on vec_chunks primary key" in str(error)
                exists = duplicate and conn.execute(
                    "SELECT 1 FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)
                ).fetchone()
                if not exists:
                    raise
            conn.execute("UPDATE chunks SET embedded = 1 WHERE id = ?", (chunk_id,))
    # Invalidate numpy search cache so it picks up new embeddings
    from .vector_search import invalidate_cache
    invalidate_cache()


def sync_session_chunks(conn: apsw.Connection, session_id: str, chunks: list[dict]) -> dict:
    """Make a session's stored chunks equal `chunks`, re-using unchanged rows.

    Chunks are matched on (turn_number, content hash).  Matching rows keep
    their id, FTS entry and vector; rows that no longer occur are deleted with
    their vectors; only new or changed chunks are inserted, to be embedded
    later.  Call inside write_transaction().
    """
    existing: dict[tuple, list[int]] = defaultdict(list)
    missing_hash = []
    for cid, turn, digest, text in conn.execute(
        "SELECT id, turn_number, content_hash, "
        "CASE WHEN content_hash IS NULL THEN combined_text END "
        "FROM chunks WHERE session_id = ? ORDER BY id",
        (session_id,),
    ):
        if digest is None:
            digest = chunk_hash(text or "")
            missing_hash.append((digest, cid))
        existing[(turn, digest)].append(cid)

    for digest, cid in missing_hash:
        conn.execute("UPDATE chunks SET content_hash = ? WHERE id = ?", (digest, cid))

    new_chunks = []
    kept = 0
    for chunk in chunks:
        ids = existing.get((chunk.get("turn_number"), chunk_hash(chunk["combined_text"])))
        if ids:
            ids.pop(0)
            kept += 1
        else:
            new_chunks.append(chunk)

    stale = [cid for ids in existing.values() for cid in ids]
    for cid in stale:
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
        conn.execute("DELETE FROM chunks WHERE id = ?", (cid,))
    insert_chunks(conn, new_chunks)
    if stale:
        from .vector_search import invalidate_cache
        invalidate_cache()
    return {"kept": kept, "deleted": len(stale), "inserted": len(new_chunks)}


def delete_session_data(conn: apsw.Connection, session_id: str) -> None:
    rows = list(conn.execute("SELECT id FROM chunks WHERE session_id = ?", (session_id,)))
    for (cid,) in rows:
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
    conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM subagents WHERE parent_session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    from .vector_search import invalidate_cache
    invalidate_cache()


def mark_embedding_failed(conn: apsw.Connection, chunk_id: int) -> None:
    """Park a chunk the model cannot embed so it is not retried forever.

    embedded = -1 is excluded from get_unembedded_chunks(); `reembed` resets it.
    """
    conn.execute("UPDATE chunks SET embedded = -1 WHERE id = ?", (chunk_id,))


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


def fts_search(
    conn: apsw.Connection, query: str, limit: int = 20,
    session_filter: tuple[str, list] | None = None,
) -> list[dict]:
    """Rank chunks by FTS5 bm25.

    session_filter is a (WHERE clause over sessions, bindings) pair; only
    chunks of matching sessions are ranked.
    """
    safe_query = _sanitize_fts_query(query)
    if not safe_query:
        return []
    if session_filter is None:
        sql = """SELECT rowid, rank
                 FROM chunks_fts
                 WHERE combined_text MATCH ?
                 ORDER BY rank
                 LIMIT ?"""
        bindings = (safe_query, limit)
    else:
        where, params = session_filter
        sql = f"""SELECT f.rowid, f.rank
                  FROM chunks_fts f
                  JOIN chunks c ON c.id = f.rowid
                  WHERE f.combined_text MATCH ?
                    AND c.session_id IN (SELECT session_id FROM sessions WHERE {where})
                  ORDER BY f.rank
                  LIMIT ?"""
        bindings = (safe_query, *params, limit)
    try:
        rows = list(conn.execute(sql, bindings))
    except apsw.SQLError:
        # If FTS5 still fails on unusual input, return empty and let vector search carry
        return []
    return [{"chunk_id": r[0], "rank": r[1]} for r in rows]


def text_search(
    conn: apsw.Connection, query: str, limit: int = 50,
    source: str | None = None,
) -> list[dict]:
    """Exact substring search via SQL LIKE. No FTS5, no embeddings."""
    if not query.strip():
        return []
    source_clause = " AND s.source = ?" if source else ""
    bindings = (query, source, limit) if source else (query, limit)
    return _fetchall(
        conn,
        f"""SELECT c.*, s.project_path, s.slug, s.git_branch,
                  s.message_count, s.first_message_at, s.last_message_at,
                  s.source, s.native_session_id, s.thread_kind
           FROM chunks c
           JOIN sessions s ON c.session_id = s.session_id
           WHERE c.combined_text LIKE '%' || ? || '%'
           {source_clause}
           ORDER BY c.id DESC
           LIMIT ?""",
        bindings,
    )


def file_search(
    conn: apsw.Connection, query: str, limit: int = 20,
    source: str | None = None,
) -> list[dict]:
    """Search sessions by file path in files_touched metadata."""
    if not query.strip():
        return []
    source_clause = " AND s.source = ?" if source else ""
    bindings = (query, source, limit) if source else (query, limit)
    return _fetchall(
        conn,
        f"""SELECT s.*, NULL as id, NULL as user_content, NULL as assistant_content,
                  NULL as combined_text, s.first_message_at as timestamp,
                  0 as turn_number, 0 as token_estimate
           FROM sessions s
           WHERE s.files_touched LIKE '%' || ? || '%'
           {source_clause}
           ORDER BY s.last_message_at DESC
           LIMIT ?""",
        bindings,
    )


def get_chunks_by_ids(conn: apsw.Connection, chunk_ids: list[int]) -> list[dict]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" for _ in chunk_ids)
    return _fetchall(
        conn,
        f"""SELECT c.*, s.project_path, s.slug, s.git_branch,
                   s.message_count, s.first_message_at, s.last_message_at,
                   s.source, s.native_session_id, s.thread_kind
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


def get_indexed_sessions(conn: apsw.Connection, source: str | None = None) -> dict[str, dict]:
    sql = (
        "SELECT session_id, indexed_at, transcript_path, transcript_mtime, "
        "transcript_size FROM sessions"
    )
    bindings = ()
    if source:
        sql += " WHERE source = ?"
        bindings = (source,)
    rows = _fetchall(conn, sql, bindings)
    return {row["session_id"]: row for row in rows}


def get_session_ids_by_source(conn: apsw.Connection, source: str) -> list[str]:
    if source == "all":
        return [row[0] for row in conn.execute("SELECT session_id FROM sessions")]
    return [row[0] for row in conn.execute(
        "SELECT session_id FROM sessions WHERE source = ?", (source,)
    )]


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
    """Recreate vec_chunks for the configured model and mark every chunk for embedding.

    Call inside write_transaction().  Returns the number of chunks marked.
    """
    spec = configured_model()
    conn.execute("DROP TABLE IF EXISTS vec_chunks")
    _create_vec_table(conn, spec.dim)
    conn.execute("UPDATE chunks SET embedded = 0")
    set_meta(conn, "embedding_model", spec.name)
    set_meta(conn, "embedding_dim", str(spec.dim))
    from .vector_search import invalidate_cache
    invalidate_cache()
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

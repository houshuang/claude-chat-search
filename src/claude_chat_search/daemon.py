import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

from .chunker import create_chunks
from .db import (
    DB_DIR,
    delete_session_data,
    get_connection,
    get_session,
    get_unembedded_chunks,
    init_db,
    insert_chunks,
    insert_session,
)
from .parser import (
    extract_session_metadata,
    file_info_from_path,
    iter_jsonl_files,
    parse_jsonl_file,
)

QUEUE_PATH = DB_DIR / ".queue"
QUEUE_PROCESSING = DB_DIR / ".queue.processing"
PID_FILE = DB_DIR / "daemon.pid"
LOG_FILE = DB_DIR / "daemon.log"
POLL_INTERVAL = 2
COOLDOWN_SECONDS = 60
ACTIVE_COOLDOWN_SECONDS = 300  # 5 min for sessions that keep changing
EMBED_INTERVAL = 30
WAL_CHECKPOINT_INTERVAL = 600  # 10 minutes
REINDEX_WINDOW = 600  # 10 min window for counting re-indexes

logger = logging.getLogger("claude-chat-search-daemon")

# Module-level shutdown flag so signal handler can interrupt long operations
_shutdown = False
_last_indexed: dict[str, float] = {}  # session_id -> monotonic time of last index
_file_fingerprints: dict[str, tuple[float, int]] = {}  # session_id -> (mtime, size)
_reindex_times: dict[str, list[float]] = {}  # session_id -> list of recent index times


def _get_file_fingerprint(path) -> tuple[float, int]:
    """Return (mtime, size) for a file path."""
    try:
        st = os.stat(path)
        return (st.st_mtime, st.st_size)
    except OSError:
        return (0.0, 0)


def index_single_session(conn, file_info: dict) -> bool:
    """Index a single session file. Skips if file unchanged (mtime+size). Returns True if indexed."""
    sid = file_info["session_id"]
    path = file_info["path"]

    # Check file fingerprint first — avoids parsing the file at all if unchanged
    current_fp = _get_file_fingerprint(path)
    if current_fp == _file_fingerprints.get(sid):
        return False

    try:
        session_data = parse_jsonl_file(path)
    except Exception as e:
        logger.error(f"Error parsing {path}: {e}")
        return False

    if not session_data["messages"]:
        return False

    existing = get_session(conn, sid)
    if existing and existing["message_count"] == session_data["message_count"]:
        # File changed but message_count didn't — update fingerprint, skip re-index
        _file_fingerprints[sid] = current_fp
        return False

    if existing:
        delete_session_data(conn, sid)

    now = datetime.now(timezone.utc).isoformat()
    session_data["project_path"] = file_info["project_path"]
    session_data["indexed_at"] = now
    session_data["parent_session_id"] = file_info.get("parent_session_id")

    metadata = extract_session_metadata(session_data["messages"])
    session_data.update(metadata)

    insert_session(conn, session_data)
    chunks = create_chunks(session_data)
    if chunks:
        insert_chunks(conn, chunks)

    _file_fingerprints[sid] = current_fp
    return True


def _get_cooldown(sid: str) -> float:
    """Compute cooldown for a session based on how often it's been re-indexed recently.

    Sessions re-indexed 3+ times in the last 10 minutes are "hot" (active sessions)
    and get a 5-minute cooldown. Sessions indexed once or twice keep the 60s default.
    """
    now = time.monotonic()
    times = _reindex_times.get(sid, [])
    # Prune old entries outside the window
    times = [t for t in times if now - t < REINDEX_WINDOW]
    _reindex_times[sid] = times

    if len(times) >= 3:
        return ACTIVE_COOLDOWN_SECONDS
    return COOLDOWN_SECONDS


def _record_reindex(sid: str):
    """Record that a session was just re-indexed."""
    now = time.monotonic()
    times = _reindex_times.get(sid, [])
    times.append(now)
    # Keep only entries within the window
    times = [t for t in times if now - t < REINDEX_WINDOW]
    _reindex_times[sid] = times


def process_queue(conn) -> int:
    """Atomically grab the queue file, dedupe paths, index each. Returns count indexed."""
    if not QUEUE_PATH.exists():
        return 0

    try:
        QUEUE_PATH.rename(QUEUE_PROCESSING)
    except OSError:
        return 0

    try:
        paths = QUEUE_PROCESSING.read_text().strip().splitlines()
    finally:
        try:
            QUEUE_PROCESSING.unlink()
        except OSError:
            pass

    unique_paths = list(dict.fromkeys(p.strip() for p in paths if p.strip()))

    indexed = 0
    for transcript_path in unique_paths:
        if _shutdown:
            break
        file_info = file_info_from_path(transcript_path)
        if file_info is None:
            continue
        sid = file_info["session_id"]
        now = time.monotonic()

        cooldown = _get_cooldown(sid)
        if now - _last_indexed.get(sid, 0) < cooldown:
            continue
        if index_single_session(conn, file_info):
            _last_indexed[sid] = time.monotonic()
            _record_reindex(sid)
            indexed += 1
            logger.info(f"Indexed {sid} (cooldown={int(cooldown)}s)")

    return indexed


def full_scan(conn) -> int:
    """Full scan using iter_jsonl_files, skip unchanged via file fingerprint + message_count."""
    files = iter_jsonl_files()
    indexed = 0
    for file_info in files:
        if _shutdown:
            break
        if index_single_session(conn, file_info):
            sid = file_info["session_id"]
            _record_reindex(sid)
            _last_indexed[sid] = time.monotonic()
            indexed += 1
    return indexed


def run_embeddings(conn):
    """Run embedding pipeline in batches, checking shutdown between batches."""
    from .embedder import embed_texts
    from .db import insert_embeddings

    while not _shutdown:
        rows = get_unembedded_chunks(conn, 256)
        if not rows:
            break

        texts = [r["combined_text"] for r in rows]
        chunk_ids = [r["id"] for r in rows]

        try:
            embeddings = embed_texts(texts)
            insert_embeddings(conn, chunk_ids, embeddings)
        except Exception:
            logger.exception("Embedding batch failed")
            break


def wal_checkpoint(conn):
    """Run a WAL checkpoint to prevent unbounded WAL growth."""
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        logger.debug("WAL checkpoint completed")
    except Exception:
        logger.exception("WAL checkpoint failed")


def is_running() -> int | None:
    """Return PID if daemon is running, else None."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        try:
            PID_FILE.unlink()
        except OSError:
            pass
        return None


def run():
    """Main daemon loop (foreground). Suitable for launchd or direct invocation."""
    global _shutdown

    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Also log to stderr when running interactively
    if sys.stderr.isatty():
        logging.getLogger().addHandler(logging.StreamHandler())

    existing = is_running()
    if existing:
        print(f"Daemon already running (PID {existing})", file=sys.stderr)
        sys.exit(1)

    DB_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    def handle_signal(signum, _frame):
        global _shutdown
        _shutdown = True
        logger.info(f"Received signal {signum}, shutting down...")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info("Daemon started (PID %d)", os.getpid())

    conn = get_connection()
    init_db(conn)

    # Startup catch-up
    if not _shutdown:
        logger.info("Running startup full scan...")
        count = full_scan(conn)
        logger.info(f"Startup scan: indexed {count} sessions")
        if count > 0:
            run_embeddings(conn)

    last_embed_time = time.monotonic()
    last_wal_checkpoint = time.monotonic()

    while not _shutdown:
        try:
            indexed = process_queue(conn)
            if indexed > 0:
                logger.info(f"Queue batch: indexed {indexed} sessions")
        except Exception:
            logger.exception("Error processing queue")

        # Run embeddings on a separate timer, independent of indexing
        now = time.monotonic()
        if now - last_embed_time >= EMBED_INTERVAL:
            try:
                run_embeddings(conn)
            except Exception:
                logger.exception("Error running embeddings")
            last_embed_time = time.monotonic()

        # Periodic WAL checkpoint to prevent unbounded WAL growth
        now = time.monotonic()
        if now - last_wal_checkpoint >= WAL_CHECKPOINT_INTERVAL:
            wal_checkpoint(conn)
            last_wal_checkpoint = time.monotonic()

        for _ in range(POLL_INTERVAL * 10):
            if _shutdown:
                break
            time.sleep(0.1)

    # Final WAL checkpoint before exit
    wal_checkpoint(conn)
    conn.close()
    try:
        PID_FILE.unlink()
    except OSError:
        pass
    logger.info("Daemon stopped")


def start():
    """Start daemon as a detached background process via double-fork."""
    existing = is_running()
    if existing:
        print(f"Daemon already running (PID {existing})")
        return

    # First fork
    pid = os.fork()
    if pid > 0:
        # Parent waits for first child to exit (it exits immediately after second fork)
        os.waitpid(pid, 0)
        time.sleep(0.5)
        running = is_running()
        if running:
            print(f"Daemon started (PID {running})")
        else:
            print("Daemon failed to start. Check " + str(LOG_FILE), file=sys.stderr)
        return

    # First child — detach from terminal
    os.setsid()

    # Second fork — orphan the grandchild so init adopts it
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)  # First child exits, parent's waitpid returns

    # Grandchild — the actual daemon
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    if devnull > 2:
        os.close(devnull)

    run()
    os._exit(0)


def stop():
    """Send SIGTERM to running daemon."""
    pid = is_running()
    if pid is None:
        print("Daemon is not running")
        return

    os.kill(pid, signal.SIGTERM)
    # Wait up to 30s for it to finish current batch and exit
    for _ in range(300):
        try:
            os.kill(pid, 0)
            time.sleep(0.1)
        except ProcessLookupError:
            print(f"Daemon stopped (was PID {pid})")
            return
    print(f"Daemon (PID {pid}) did not stop after 30s — may need kill -9")


def status():
    """Print daemon status and recent log lines."""
    pid = is_running()
    if pid:
        print(f"Running (PID {pid})")
    else:
        print("Not running")

    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().strip().splitlines()
        recent = lines[-10:] if len(lines) > 10 else lines
        if recent:
            print(f"\nRecent log ({LOG_FILE}):")
            for line in recent:
                print(f"  {line}")

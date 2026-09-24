import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import apsw

from .chunker import create_chunks
from .db import (
    BUSY_TIMEOUT_MS,
    write_transaction,
    DB_DIR,
    delete_session_data,
    get_connection,
    get_session,
    get_unembedded_chunks,
    init_db,
    insert_session,
    insert_subagent,
    sync_session_chunks,
    update_session_fingerprint,
    update_session_git_remote,
)
from .parser import (
    detect_git_remote,
    extract_session_metadata,
    file_info_from_path,
    find_project_dir,
    is_excluded_project,
    iter_jsonl_files,
    iter_subagent_files,
    parse_jsonl_file,
    parse_subagent_metadata,
)

QUEUE_PATH = DB_DIR / ".queue"
QUEUE_PROCESSING = DB_DIR / ".queue.processing"
PID_FILE = DB_DIR / "daemon.pid"
LOG_FILE = DB_DIR / "daemon.log"
POLL_INTERVAL = 2
BUSY_RETRY_SECONDS = 30
COOLDOWN_SECONDS = 60
ACTIVE_COOLDOWN_SECONDS = 300  # 5 min for sessions that keep changing
EMBED_INTERVAL = 30
SCAN_INTERVAL = 3600  # hourly fingerprint scan catches transcripts no hook queued
WAL_CHECKPOINT_INTERVAL = 600  # 10 minutes
CODE_CHECK_INTERVAL = 300  # 5 min — exit if source code changed (launchd restarts)
LOG_MAX_BYTES = 1_000_000  # 1MB — truncate daemon.log on startup if larger
LOG_KEEP_LINES = 1000
REINDEX_WINDOW = 600  # 10 min window for counting re-indexes
REPO_DIR = Path(__file__).resolve().parent.parent.parent  # project root

logger = logging.getLogger("claude-chat-search-daemon")

# Module-level shutdown flag so signal handler can interrupt long operations
_shutdown = False
_last_indexed: dict[str, float] = {}  # session_id -> monotonic time of last index
_file_fingerprints: dict[str, tuple[float, int]] = {}  # session_id -> (mtime, size)
_reindex_times: dict[str, list[float]] = {}  # session_id -> list of recent index times
# session_id -> (monotonic due time, transcript path) for work that arrived
# during a cooldown or hit a busy database; retried once due.
_deferred: dict[str, tuple[float, str]] = {}


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

    existing = get_session(conn, sid)
    if (
        existing
        and existing.get("transcript_mtime") == current_fp[0]
        and existing.get("transcript_size") == current_fp[1]
    ):
        _file_fingerprints[sid] = current_fp
        if existing.get("transcript_path") != str(path):
            update_session_fingerprint(conn, sid, file_info)
        return False

    try:
        session_data = parse_jsonl_file(path)
    except Exception as e:
        logger.error(f"Error parsing {path}: {e}")
        return False

    if not session_data["messages"]:
        _file_fingerprints[sid] = current_fp
        return False

    # Claude names project directories lossily; the recorded cwd is exact.
    if is_excluded_project(file_info["project_path"], cwd=session_data.get("cwd")):
        _file_fingerprints[sid] = current_fp
        if existing:
            with write_transaction(conn):
                delete_session_data(conn, sid)
            logger.info(f"Removed {sid}: its cwd is excluded")
        return False

    if existing and existing["message_count"] == session_data["message_count"]:
        # File changed but message_count didn't — update fingerprint, skip re-index
        update_session_fingerprint(conn, sid, file_info)
        _file_fingerprints[sid] = current_fp
        return False

    now = datetime.now(timezone.utc).isoformat()
    session_data["project_path"] = file_info["project_path"]
    session_data["indexed_at"] = now
    session_data["parent_session_id"] = file_info.get("parent_session_id")
    session_data["source"] = file_info.get("source", "claude")
    session_data["native_session_id"] = file_info.get("native_session_id", sid)
    session_data["thread_kind"] = file_info.get("thread_kind", "user")
    session_data["transcript_path"] = str(path)
    session_data["transcript_mtime"] = current_fp[0]
    session_data["transcript_size"] = current_fp[1]

    metadata = extract_session_metadata(session_data["messages"])
    session_data.update(metadata)
    chunks = create_chunks(session_data)

    with write_transaction(conn):
        insert_session(conn, session_data)
        sync_session_chunks(conn, sid, chunks)

    # Backfill git remote for this session's project
    project_path = file_info["project_path"]
    remote = detect_git_remote(project_path)
    if remote:
        update_session_git_remote(conn, project_path, remote)

    # Index subagent metadata for this session
    project_dir = find_project_dir(project_path)
    if project_dir:
        for sf in iter_subagent_files(sid, project_dir):
            try:
                meta = parse_subagent_metadata(sf["jsonl_path"], sf.get("meta_path"))
                insert_subagent(conn, {
                    "agent_id": sf["agent_id"],
                    "parent_session_id": sid,
                    "file_size": sf["file_size"],
                    "jsonl_path": sf["jsonl_path"],
                    "indexed_at": now,
                    **meta,
                })
            except Exception as e:
                logger.warning(f"Error indexing subagent {sf['agent_id']}: {e}")

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


def _defer(sid: str, path: str, due: float) -> None:
    current = _deferred.get(sid)
    if current is None or due > current[0]:
        _deferred[sid] = (due, path)


def _index_with_cooldown(conn, file_info: dict) -> bool:
    """Index now, or defer until the session's cooldown has passed."""
    sid = file_info["session_id"]
    path = str(file_info["path"])
    cooldown = _get_cooldown(sid)
    last = _last_indexed.get(sid)
    if last is not None and time.monotonic() - last < cooldown:
        _defer(sid, path, last + cooldown)
        return False
    try:
        indexed = index_single_session(conn, file_info)
    except apsw.BusyError:
        logger.warning(f"Database busy; deferring {sid} by {BUSY_RETRY_SECONDS}s")
        _defer(sid, path, time.monotonic() + BUSY_RETRY_SECONDS)
        return False
    if indexed:
        _last_indexed[sid] = time.monotonic()
        _record_reindex(sid)
        logger.info(f"Indexed {sid} (cooldown={int(cooldown)}s)")
    return indexed


def _index_paths(conn, paths) -> int:
    indexed = 0
    for transcript_path in paths:
        if _shutdown:
            break
        file_info = file_info_from_path(transcript_path)
        if file_info is None:
            continue
        if _index_with_cooldown(conn, file_info):
            indexed += 1
    return indexed


def process_queue(conn) -> int:
    """Atomically grab the queue file, dedupe paths, index each. Returns count indexed.

    Paths still in cooldown are deferred, not dropped; see process_deferred().
    """
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
    return _index_paths(conn, unique_paths)


def process_deferred(conn) -> int:
    """Index deferred sessions whose due time has passed."""
    now = time.monotonic()
    due = [sid for sid, (when, _path) in _deferred.items() if when <= now]
    paths = [_deferred.pop(sid)[1] for sid in due]
    return _index_paths(conn, paths)


def full_scan(conn) -> int:
    """Full scan using iter_jsonl_files, skip unchanged via file fingerprint + message_count."""
    files = iter_jsonl_files()
    indexed = 0
    for file_info in files:
        if _shutdown:
            break
        if _index_with_cooldown(conn, file_info):
            indexed += 1
    return indexed


def run_embeddings(conn):
    """Run embedding pipeline in batches, checking shutdown between batches."""
    from .embedder import EmbeddingUnavailable, embed_rows, embedding_lock

    with embedding_lock() as acquired:
        if not acquired:
            logger.info("Embedding pass skipped: another indexer holds the lock")
            return

        while not _shutdown:
            rows = get_unembedded_chunks(conn, 256)
            if not rows:
                break

            try:
                _stored, skipped = embed_rows(conn, rows)
            except EmbeddingUnavailable:
                logger.exception("Embedding model unavailable; retrying next pass")
                break
            except Exception:
                logger.exception("Embedding batch failed")
                break
            if skipped:
                logger.warning("Skipped %d chunk(s) that cannot be embedded: %s",
                               len(skipped), skipped)


def wal_checkpoint(conn, mode: str = "PASSIVE", busy_ms: int = 1000):
    """Checkpoint the WAL without stalling other writers.

    PASSIVE never waits. TRUNCATE (used at shutdown) holds the write lock while
    it waits for readers, which blocks every other writer for that long, so it
    gets a short busy timeout of its own.
    """
    conn.setbusytimeout(busy_ms)
    try:
        busy, log_frames, done = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        logger.debug("WAL checkpoint %s: busy=%s log=%s checkpointed=%s",
                     mode, busy, log_frames, done)
    except Exception:
        logger.exception("WAL checkpoint failed")
    finally:
        conn.setbusytimeout(BUSY_TIMEOUT_MS)


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


def _get_git_commit() -> str | None:
    """Return current HEAD commit hash, or None if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _truncate_log_if_needed():
    """Truncate daemon.log on startup if it exceeds LOG_MAX_BYTES, keeping tail."""
    if not LOG_FILE.exists():
        return
    if LOG_FILE.stat().st_size <= LOG_MAX_BYTES:
        return
    lines = LOG_FILE.read_text().splitlines()
    kept = lines[-LOG_KEEP_LINES:]
    LOG_FILE.write_text("\n".join(kept) + "\n")


def _start_search_server():
    """Serve searches on the socket; the model and vectors load in the background."""
    from .search_service import SearchServer

    server = SearchServer()
    try:
        server.start()
    except OSError:
        logger.exception("Search socket unavailable; CLI searches run in-process")
        return None

    def warm_up():
        try:
            server.service.warm_up()
            logger.info("Search server ready on %s", server.path)
        except Exception:
            logger.exception("Search server warm-up failed")

    threading.Thread(target=warm_up, name="search-warm-up", daemon=True).start()
    return server


def run():
    """Main daemon loop (foreground). Suitable for launchd or direct invocation."""
    global _shutdown

    _truncate_log_if_needed()

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

    startup_commit = _get_git_commit()
    logger.info("Daemon started (PID %d, commit %s)", os.getpid(),
                startup_commit[:8] if startup_commit else "unknown")

    conn = get_connection()
    init_db(conn)

    server = _start_search_server()

    # Startup catch-up
    if not _shutdown:
        logger.info("Running startup full scan...")
        while not _shutdown:
            try:
                count = full_scan(conn)
                logger.info(f"Startup scan: indexed {count} sessions")
                if count > 0:
                    run_embeddings(conn)
                break
            except apsw.BusyError:
                # Another explicit index/backup job can briefly hold the writer
                # lock.  Wait at low CPU and retry instead of crash-looping under
                # launchd KeepAlive.
                logger.warning(
                    "Startup scan deferred: database is busy; retrying in %ss",
                    BUSY_RETRY_SECONDS,
                )
                for _ in range(BUSY_RETRY_SECONDS * 10):
                    if _shutdown:
                        break
                    time.sleep(0.1)

    last_embed_time = time.monotonic()
    last_wal_checkpoint = time.monotonic()
    last_code_check = time.monotonic()
    last_scan = time.monotonic()

    while not _shutdown:
        try:
            indexed = process_queue(conn) + process_deferred(conn)
            if indexed > 0:
                logger.info(f"Queue batch: indexed {indexed} sessions")
        except Exception:
            logger.exception("Error processing queue")

        if time.monotonic() - last_scan >= SCAN_INTERVAL:
            try:
                count = full_scan(conn)
                if count:
                    logger.info(f"Periodic scan: indexed {count} sessions")
            except Exception:
                logger.exception("Error in periodic scan")
            last_scan = time.monotonic()

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

        # Exit if source code changed — launchd will restart with new code
        if startup_commit and now - last_code_check >= CODE_CHECK_INTERVAL:
            current = _get_git_commit()
            if current and current != startup_commit:
                logger.info("Code changed (%s -> %s), exiting for restart",
                            startup_commit[:8], current[:8])
                break
            last_code_check = time.monotonic()

        for _ in range(POLL_INTERVAL * 10):
            if _shutdown:
                break
            time.sleep(0.1)

    if server is not None:
        server.stop()

    # Final WAL checkpoint before exit
    wal_checkpoint(conn, "TRUNCATE")
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

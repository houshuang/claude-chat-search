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

logger = logging.getLogger("claude-chat-search-daemon")

# Module-level shutdown flag so signal handler can interrupt long operations
_shutdown = False


def index_single_session(conn, file_info: dict) -> bool:
    """Index a single session file. Skips if message_count unchanged. Returns True if indexed."""
    sid = file_info["session_id"]

    try:
        session_data = parse_jsonl_file(file_info["path"])
    except Exception as e:
        logger.error(f"Error parsing {file_info['path']}: {e}")
        return False

    if not session_data["messages"]:
        return False

    existing = get_session(conn, sid)
    if existing and existing["message_count"] == session_data["message_count"]:
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

    return True


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
        if index_single_session(conn, file_info):
            indexed += 1
            logger.info(f"Indexed {file_info['session_id']}")

    return indexed


def full_scan(conn) -> int:
    """Full scan using iter_jsonl_files, skip unchanged via message_count."""
    files = iter_jsonl_files()
    indexed = 0
    for file_info in files:
        if _shutdown:
            break
        if index_single_session(conn, file_info):
            indexed += 1
    return indexed


def run_embeddings(conn):
    """Run embedding pipeline in batches, checking shutdown between batches."""
    try:
        from .embedder import get_client, embed_texts
        from .db import insert_embeddings
    except RuntimeError:
        return

    try:
        client = get_client()
    except RuntimeError:
        return

    while not _shutdown:
        rows = get_unembedded_chunks(conn, 100)
        if not rows:
            break

        texts = [r["combined_text"][:8000] for r in rows]
        chunk_ids = [r["id"] for r in rows]

        try:
            embeddings = embed_texts(client, texts)
            insert_embeddings(conn, chunk_ids, embeddings)
        except Exception:
            logger.exception("Embedding batch failed")
            break


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
    # Suppress noisy HTTP request logs from OpenAI/httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
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

    while not _shutdown:
        try:
            indexed = process_queue(conn)
            if indexed > 0:
                logger.info(f"Queue batch: indexed {indexed} sessions")
                run_embeddings(conn)
        except Exception:
            logger.exception("Error processing queue")

        for _ in range(POLL_INTERVAL * 10):
            if _shutdown:
                break
            time.sleep(0.1)

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

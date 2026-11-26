"""Search served by a model process over a unix socket.

A one-shot CLI search would spend most of its time loading the embedding
model and reading every vector from SQLite.  The search server keeps both in
memory.  It is started by the first search that finds none running
(`ask()`), embeds pending chunks while it is up, and exits after
SERVER_IDLE_SECONDS without a search: a process that only unloads the model
gives almost none of its memory back, one that exits gives back all of it.
The CLI never loads the model itself; when the server is not ready in time
it answers from the keyword index alone.

Protocol: one JSON object per connection, newline-terminated, in each
direction.  Request {"v": 1, "op": ..., "args": {...}}; reply
{"ok": true, ...} or {"ok": false, "error": ...}.  The socket is created with
mode 0600 in the index directory.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import select
import signal
import socket
import socketserver
import stat
import subprocess
import sys
import threading
import time

from . import db

PROTOCOL_VERSION = 1
CONNECT_TIMEOUT = 0.25
PING_TIMEOUT = 2.0
MAX_REQUEST_BYTES = 1_000_000
# How long a search waits for a server that is starting: imports, model and
# vectors take about 10-15 s on an idle machine.
START_TIMEOUT = 45.0
SERVER_IDLE_SECONDS = 900
SERVER_EMBED_INTERVAL = 600
SERVER_EMBED_BACKLOG_INTERVAL = 60
SERVER_EMBED_SECONDS = 120
SERVER_LOG_MAX_BYTES = 1_000_000
# Set to 1 to never start a search server: searches are keyword-only.
NO_SERVER_ENV = "CLAUDE_CHAT_SEARCH_NO_SERVER"
SEARCH_ARGS = ("project", "branch", "since", "before", "source")

logger = logging.getLogger("claude-chat-search-daemon")


def socket_path():
    return db.DB_DIR / "search.sock"


def request(op: str, args: dict, timeout: float) -> dict | None:
    """Send one request to the daemon; None when no daemon answers properly."""
    path = socket_path()
    try:
        st = path.lstat()
    except OSError:
        return None
    if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
        return None
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(CONNECT_TIMEOUT)
        sock.connect(str(path))
        sock.settimeout(timeout)
        payload = {"v": PROTOCOL_VERSION, "op": op, "args": args}
        # The write side stays open: the server reads a closed connection as
        # a caller that gave up, and drops the request.
        sock.sendall(json.dumps(payload).encode() + b"\n")
        data = bytearray()
        while chunk := sock.recv(65536):
            data += chunk
    except OSError:
        return None
    finally:
        sock.close()
    try:
        reply = json.loads(data)
    except ValueError:
        return None
    if not isinstance(reply, dict) or not reply.get("ok"):
        return None
    return reply


def server_log_path():
    return db.DB_DIR / "search-server.log"


def spawn_server() -> bool:
    """Start a detached search server; False when that is disabled or fails."""
    if os.environ.get(NO_SERVER_ENV) == "1":
        return False
    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(name, "2")
    try:
        db.DB_DIR.mkdir(parents=True, exist_ok=True)
        with open(server_log_path(), "ab") as log:
            subprocess.Popen(
                [sys.executable, "-m", "claude_chat_search.cli", "serve"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True, env=env, cwd="/",
            )
    except OSError:
        return False
    return True


def ask(op: str, args: dict, timeout: float,
        start_timeout: float | None = None) -> dict | None:
    """Send a request to the search server, starting it when none is running.

    Waits up to start_timeout for a starting server to become ready.  None
    means no answer in time; the caller falls back to keyword search.
    """
    start_timeout = START_TIMEOUT if start_timeout is None else start_timeout
    deadline = time.monotonic() + start_timeout
    pong = request("ping", {}, PING_TIMEOUT)
    if pong is None and not spawn_server():
        return None
    while not (pong and pong.get("ready")):
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)
        pong = request("ping", {}, PING_TIMEOUT)
    return request(op, args, timeout)


class Abandoned(Exception):
    """The caller closed the connection before its request was run."""


def _search_kwargs(args: dict) -> dict:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    limit = args.get("limit", 10)
    if not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer between 1 and 1000")
    kwargs = {"query": query, "limit": limit, "expand": bool(args.get("expand"))}
    for key in SEARCH_ARGS:
        value = args.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        kwargs[key] = value
    return kwargs


class SearchService:
    """Answers search requests from one shared read-only connection."""

    def __init__(self):
        self._conn = None
        self._lock = threading.Lock()
        self.ready = threading.Event()
        self.last_used = time.monotonic()
        self._active = 0
        self._active_lock = threading.Lock()

    def idle_for(self) -> float:
        """Seconds since the last request finished; 0 while one is running."""
        with self._active_lock:
            return 0.0 if self._active else time.monotonic() - self.last_used

    def _connection(self):
        if self._conn is None:
            self._conn = db.get_read_connection()
        return self._conn

    def warm_up(self) -> None:
        from .embedder import embed_query
        from .vector_search import MIN_REFRESH_INTERVAL, _cache

        # The cache is refreshed on its own connection and thread, so a search
        # never waits for a refresh: it uses the copy that is current.  The
        # first read runs while the model loads.
        refresh_conn = db.get_read_connection()
        failure = []

        def read_vectors():
            try:
                _cache.refresh(refresh_conn)
            except Exception as error:
                failure.append(error)

        reading = threading.Thread(target=read_vectors, name="vector-cache-load")
        reading.start()
        embed_query("warm up")
        reading.join()
        if failure:
            raise failure[0]
        _cache.background_refresh = True
        self.ready.set()

        def keep_fresh():
            while True:
                time.sleep(MIN_REFRESH_INTERVAL)
                try:
                    _cache.refresh(refresh_conn)
                except Exception:
                    logger.exception("Vector cache refresh failed")

        threading.Thread(target=keep_fresh, name="vector-cache-refresh", daemon=True).start()

    def handle(self, message: dict, gone=lambda: False) -> dict:
        """Answer one request; `gone` reports whether the caller hung up."""
        if message.get("v") != PROTOCOL_VERSION:
            raise ValueError("unsupported protocol version")
        op = message.get("op")
        args = message.get("args") or {}
        if op == "ping":
            return {"ok": True, "ready": self.ready.is_set()}
        if not self.ready.is_set():
            return {"ok": False, "error": "warming up"}
        with self._active_lock:
            self._active += 1
        try:
            if op == "search":
                kwargs = _search_kwargs(args)
                kwargs["do_rerank"] = bool(args.get("rerank"))
                return {"ok": True, "results": self._run(self._search, kwargs, gone)}
            if op == "cross_chat":
                kwargs = _search_kwargs(args)
                kwargs.pop("source")
                results, embedding = self._run(self._cross_chat, kwargs, gone)
                return {"ok": True, "results": results, "embedding": embedding}
            raise ValueError(f"unknown op: {op!r}")
        finally:
            with self._active_lock:
                self._active -= 1
                self.last_used = time.monotonic()

    def _run(self, fn, kwargs, gone=lambda: False):
        # Query expansion calls an LLM; give it its own connection so it does
        # not hold up other searches.
        if kwargs.get("expand"):
            conn = db.get_read_connection()
            try:
                return fn(conn, **kwargs)
            finally:
                conn.close()
        with self._lock:
            # Searches run one at a time; a caller that timed out while queued
            # must not make the callers behind it wait for its search too.
            if gone():
                raise Abandoned()
            return fn(self._connection(), **kwargs)

    @staticmethod
    def _search(conn, **kwargs):
        from .search import hybrid_search
        return hybrid_search(conn, **kwargs)

    @staticmethod
    def _cross_chat(conn, query, **kwargs):
        from .cross_search import chat_candidates
        from .embedder import embed_query
        embedding = embed_query(query) if db.embedding_mismatch(conn) is None else None
        return chat_candidates(conn, query, embedding, **kwargs), embedding

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


class _Handler(socketserver.StreamRequestHandler):
    def _gone(self) -> bool:
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def handle(self):
        try:
            line = self.rfile.readline(MAX_REQUEST_BYTES)
            reply = self.server.service.handle(json.loads(line), self._gone)
        except Abandoned:
            return
        except Exception as error:
            logger.exception("Search request failed")
            reply = {"ok": False, "error": str(error)}
        try:
            self.wfile.write(json.dumps(reply).encode() + b"\n")
        except OSError:
            pass


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class SearchServer:
    def __init__(self, service: SearchService | None = None):
        self.service = service or SearchService()
        self.path = socket_path()
        self._server = None
        self._thread = None

    def start(self) -> None:
        # Only the process holding the server lock calls this, so a socket
        # left here is stale.
        try:
            if stat.S_ISSOCK(self.path.lstat().st_mode):
                self.path.unlink()
        except FileNotFoundError:
            pass
        old_umask = os.umask(0o177)
        try:
            self._server = _Server(str(self.path), _Handler)
        finally:
            os.umask(old_umask)
        os.chmod(self.path, 0o600)
        self._server.service = self.service
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="search-server", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self.service.close()


def _server_lock():
    """The open lock file when this process may be the search server, else None."""
    db.DB_DIR.mkdir(parents=True, exist_ok=True)
    handle = open(db.DB_DIR / ".search-server.lock", "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _embed_while_serving(stop: threading.Event) -> None:
    from .embedder import embed_pending

    conn = db.get_connection()
    try:
        wait = 5.0
        while not stop.wait(wait):
            more = False
            try:
                more = embed_pending(conn, SERVER_EMBED_SECONDS, stop.is_set)
            except Exception:
                logger.exception("Embedding pass failed")
            wait = SERVER_EMBED_BACKLOG_INTERVAL if more else SERVER_EMBED_INTERVAL
    finally:
        conn.close()


def serve(idle_seconds: float = SERVER_IDLE_SECONDS) -> int:
    """Run the search server until it has been idle for idle_seconds."""
    from .daemon import _get_git_commit

    lock = _server_lock()
    if lock is None:
        return 0

    log_path = server_log_path()
    try:
        if log_path.stat().st_size > SERVER_LOG_MAX_BYTES:
            log_path.write_text("")
    except OSError:
        pass
    logging.basicConfig(
        filename=str(log_path), level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    started = time.monotonic()
    commit = _get_git_commit()
    server = SearchServer()
    server.start()
    logger.info("Search server starting (PID %d)", os.getpid())
    try:
        server.service.warm_up()
        logger.info("Search server ready after %.1fs", time.monotonic() - started)
        embedding = threading.Thread(
            target=_embed_while_serving, args=(stop,), name="embed-pending", daemon=True
        )
        embedding.start()
        checked = time.monotonic()
        while not stop.wait(1.0):
            if server.service.idle_for() >= idle_seconds:
                logger.info("No search for %ds; exiting", idle_seconds)
                break
            # Serve the code that is on disk: exit once it changes and no
            # search is running; the next search starts a new server.
            if time.monotonic() - checked >= 60 and server.service.idle_for() > 0:
                checked = time.monotonic()
                if commit and _get_git_commit() not in (None, commit):
                    logger.info("Code changed; exiting")
                    break
        stop.set()
        embedding.join(timeout=30)
    except Exception:
        logger.exception("Search server failed")
        return 1
    finally:
        server.stop()
        lock.close()
    return 0

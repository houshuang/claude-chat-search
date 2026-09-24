"""Search served by the daemon over a unix socket.

A one-shot CLI search spends most of its time loading the embedding model
and reading every vector from SQLite.  The daemon keeps both in memory, so
the CLI sends it the query and prints what comes back, falling back to an
in-process search when no daemon answers.

Protocol: one JSON object per connection, newline-terminated, in each
direction.  Request {"v": 1, "op": ..., "args": {...}}; reply
{"ok": true, ...} or {"ok": false, "error": ...}.  The socket is created with
mode 0600 in the index directory.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import socketserver
import stat
import threading

from . import db

PROTOCOL_VERSION = 1
CONNECT_TIMEOUT = 0.25
MAX_REQUEST_BYTES = 1_000_000
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
        sock.sendall(json.dumps(payload).encode() + b"\n")
        sock.shutdown(socket.SHUT_WR)
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

    def _connection(self):
        if self._conn is None:
            self._conn = db.get_read_connection()
        return self._conn

    def warm_up(self) -> None:
        from .embedder import embed_query
        from .vector_search import _cache

        embed_query("warm up")
        with self._lock:
            _cache.refresh(self._connection())
        self.ready.set()

    def handle(self, message: dict) -> dict:
        if message.get("v") != PROTOCOL_VERSION:
            raise ValueError("unsupported protocol version")
        op = message.get("op")
        args = message.get("args") or {}
        if op == "ping":
            return {"ok": True, "ready": self.ready.is_set()}
        if not self.ready.is_set():
            # The model loads after the startup scan; answering "not ready" lets
            # the CLI search in-process instead of waiting up to a minute.
            return {"ok": False, "error": "warming up"}
        if op == "search":
            kwargs = _search_kwargs(args)
            kwargs["do_rerank"] = bool(args.get("rerank"))
            return {"ok": True, "results": self._run(self._search, kwargs)}
        if op == "cross_chat":
            kwargs = _search_kwargs(args)
            kwargs.pop("source")
            results, embedding = self._run(self._cross_chat, kwargs)
            return {"ok": True, "results": results, "embedding": embedding}
        raise ValueError(f"unknown op: {op!r}")

    def _run(self, fn, kwargs):
        # Query expansion calls an LLM; give it its own connection so it does
        # not hold up other searches.
        if kwargs.get("expand"):
            conn = db.get_read_connection()
            try:
                return fn(conn, **kwargs)
            finally:
                conn.close()
        with self._lock:
            return fn(self._connection(), **kwargs)

    @staticmethod
    def _search(conn, **kwargs):
        from .search import hybrid_search
        return hybrid_search(conn, **kwargs)

    @staticmethod
    def _cross_chat(conn, query, **kwargs):
        from .cross_search import chat_candidates
        from .embedder import embed_query
        embedding = embed_query(query)
        return chat_candidates(conn, query, embedding, **kwargs), embedding

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline(MAX_REQUEST_BYTES)
            reply = self.server.service.handle(json.loads(line))
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
        # Only the running daemon calls this, so a socket left here is stale.
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

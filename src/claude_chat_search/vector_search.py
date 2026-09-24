"""Brute-force vector search over an in-memory copy of vec_chunks.

At ~100K chunks a numpy dot product takes a few milliseconds, faster than
sqlite-vec for large result sets.  The copy is refreshed incrementally: when
the database has changed (PRAGMA data_version), only vectors for chunks that
appeared are read, and rows for chunks that vanished are dropped.  When the
recorded embedding model changes, the copy is rebuilt from scratch.
"""

import threading
import time

import numpy as np

from .db import stored_embedding_model

_READ_BATCH = 500


MIN_REFRESH_INTERVAL = 30.0


def _empty_state():
    return (
        np.empty(0, dtype=np.int64),
        np.empty((0, 0), dtype=np.float32),
        np.empty(0, dtype=np.int32),
    )


class VectorCache:
    def __init__(self):
        # (chunk ids, normalized vectors, session codes), replaced as a whole
        # so a search never sees a half-refreshed copy.
        self.state = _empty_state()
        self._model = None
        self.session_codes: dict[str, int] = {}
        self._data_version = None
        # Held, not just its id: data_version is only comparable on one
        # connection, and a freed connection's id can be reused.
        self._conn = None
        self._stale = True
        self._synced_at = 0.0
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        self._stale = True

    def refresh(self, conn) -> None:
        with self._lock:
            version = conn.execute("PRAGMA data_version").fetchone()[0]
            same_model = stored_embedding_model(conn) == self._model
            if not self._stale and self._conn is conn and same_model:
                if self._data_version == version:
                    return
                # The hook writes on nearly every tool call; a search may see
                # vectors up to MIN_REFRESH_INTERVAL old instead of re-syncing each time.
                if time.monotonic() - self._synced_at < MIN_REFRESH_INTERVAL:
                    return
            self._stale = False
            self._synced_at = time.monotonic()
            self._conn = conn
            self._data_version = version
            self._sync(conn)

    def _sync(self, conn) -> None:
        model = stored_embedding_model(conn)
        if model != self._model:
            self.state = _empty_state()
            self._model = model
        # Served from the covering (session_id, embedded) index; an ORDER BY
        # id here would read the whole chunks table instead.
        rows = conn.execute(
            "SELECT id, session_id FROM chunks WHERE embedded = 1"
        ).fetchall()
        ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        for _cid, sid in rows:
            if sid not in self.session_codes:
                self.session_codes[sid] = len(self.session_codes)
        sessions = np.fromiter(
            (self.session_codes[r[1]] for r in rows), dtype=np.int32, count=len(rows)
        )
        by_id = np.argsort(ids, kind="stable")
        ids, sessions = ids[by_id], sessions[by_id]

        old_ids, old_matrix, old_sessions = self.state
        if len(old_ids) == len(ids) and np.array_equal(old_ids, ids):
            if not np.array_equal(old_sessions, sessions):
                self.state = (old_ids, old_matrix, sessions)
            return
        keep = np.isin(old_ids, ids, assume_unique=True)
        known_ids, known_matrix = old_ids[keep], old_matrix[keep]
        wanted = ids[~np.isin(ids, known_ids, assume_unique=True)]
        new_ids, new_matrix = self._read_vectors(conn, wanted, full=len(known_ids) == 0)

        all_ids = np.concatenate([known_ids, new_ids])
        matrices = [m for m in (known_matrix, new_matrix) if len(m)]
        all_matrix = np.vstack(matrices) if matrices else known_matrix
        order = np.argsort(all_ids, kind="stable")
        all_ids, all_matrix = all_ids[order], all_matrix[order]
        # A chunk marked embedded whose vector is missing is left out.
        present = np.isin(ids, all_ids, assume_unique=True)
        self.state = (all_ids, all_matrix, sessions[present])

    @staticmethod
    def _read_vectors(conn, wanted: np.ndarray, full: bool):
        if len(wanted) == 0:
            return _empty_state()[:2]
        if full:
            wanted_set = set(wanted.tolist())
            rows = [
                r for r in conn.execute("SELECT chunk_id, embedding FROM vec_chunks")
                if r[0] in wanted_set
            ]
        else:
            rows = []
            values = wanted.tolist()
            for start in range(0, len(values), _READ_BATCH):
                batch = values[start:start + _READ_BATCH]
                placeholders = ",".join("?" * len(batch))
                rows.extend(conn.execute(
                    f"SELECT chunk_id, embedding FROM vec_chunks WHERE chunk_id IN ({placeholders})",
                    batch,
                ))
        if not rows:
            return _empty_state()[:2]
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32)
        matrix = matrix.reshape(len(rows), -1).copy()
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return ids, matrix / norms

    def search(
        self, query_embedding, limit: int = 20, allowed_sessions: set[str] | None = None,
    ) -> list[dict]:
        ids, matrix, sessions = self.state
        if len(ids) == 0:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).ravel()
        if query.shape[0] != matrix.shape[1]:
            return []
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        # Accelerate-backed float32 matmul raises spurious divide/overflow/invalid
        # RuntimeWarnings on macOS even for finite inputs.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = matrix @ query
        candidates = np.arange(len(scores))
        if allowed_sessions is not None:
            codes = [self.session_codes[s] for s in allowed_sessions if s in self.session_codes]
            candidates = np.flatnonzero(np.isin(sessions, codes))
            if len(candidates) == 0:
                return []
            scores = scores[candidates]
        if limit < len(scores):
            top = np.argpartition(-scores, limit)[:limit]
        else:
            top = np.arange(len(scores))
        top = top[np.argsort(-scores[top], kind="stable")]
        return [
            {"chunk_id": int(ids[candidates[i]]), "distance": 1.0 - float(scores[i])}
            for i in top
        ]


_cache = VectorCache()


def numpy_vector_search(
    conn, query_embedding: list[float], limit: int = 20,
    allowed_sessions: set[str] | None = None,
) -> list[dict]:
    """Nearest chunks by cosine similarity, optionally only from allowed_sessions.

    Returns results in the same format as db.vector_search.
    """
    _cache.refresh(conn)
    return _cache.search(query_embedding, limit, allowed_sessions)


def invalidate_cache():
    """Make the next search re-check which chunks have vectors."""
    _cache.invalidate()

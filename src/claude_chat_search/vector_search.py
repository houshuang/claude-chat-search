"""Brute-force vector search over an in-memory copy of vec_chunks.

The copy is held as normalized float16, half the size of the stored float32
vectors, and scored in blocks converted back to float32 (numpy has no fast
float16 product).  It is refreshed incrementally: when
the database has changed (PRAGMA data_version), only vectors for chunks that
appeared are read, and rows for chunks that vanished are dropped.  When the
recorded embedding model changes, the copy is rebuilt from scratch.
"""

import threading
import time

import numpy as np

from .db import stored_embedding_model

_READ_BATCH = 500
# Rows converted at a time, when reading and when scoring: bounds the float32
# scratch memory to a few tens of MB.
_BLOCK = 8192


# Each refresh after a re-index can re-read every chunk id; embeddings only
# land every EMBED_INTERVAL anyway.
MIN_REFRESH_INTERVAL = 300.0


def _empty_state():
    return (
        np.empty(0, dtype=np.int64),
        np.empty((0, 0), dtype=np.float16),
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
        # Set by the daemon, which refreshes from a background thread.
        self.background_refresh = False

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
        old_ids = self.state[0]
        if len(old_ids) and self._append_only(conn, old_ids):
            return
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

    def _append_only(self, conn, old_ids) -> bool:
        """Add vectors for chunks newer than the cache without re-reading every id.

        New chunks get higher ids. When the embedded count equals the cached rows
        plus the newer ones, nothing was deleted or re-marked, so appending them
        is exact; otherwise the caller falls back to the full sync.
        """
        top = int(old_ids[-1])
        total = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 1").fetchone()[0]
        rows = conn.execute(
            "SELECT id, session_id FROM chunks WHERE embedded = 1 AND id > ?", (top,)
        ).fetchall()
        if total != len(old_ids) + len(rows):
            return False
        if not rows:
            return True
        for _cid, sid in rows:
            if sid not in self.session_codes:
                self.session_codes[sid] = len(self.session_codes)
        ids = np.array(sorted(r[0] for r in rows), dtype=np.int64)
        by_id = {r[0]: self.session_codes[r[1]] for r in rows}
        new_ids, new_matrix = self._read_vectors(conn, ids, full=False)
        if len(new_ids) != len(ids):
            return False
        order = np.argsort(new_ids, kind="stable")
        new_ids, new_matrix = new_ids[order], new_matrix[order]
        old_ids, old_matrix, old_sessions = self.state
        sessions = np.array([by_id[int(i)] for i in new_ids], dtype=np.int32)
        self.state = (
            np.concatenate([old_ids, new_ids]),
            np.vstack([old_matrix, new_matrix]),
            np.concatenate([old_sessions, sessions]),
        )
        return True

    @staticmethod
    def _read_vectors(conn, wanted: np.ndarray, full: bool):
        if len(wanted) == 0:
            return _empty_state()[:2]
        # Filled block by block into one preallocated array: holding every row
        # as bytes and normalizing a float32 copy took four times the memory.
        ids = np.empty(len(wanted), dtype=np.int64)
        matrix = None
        filled = 0
        block_ids: list[int] = []
        block: list[bytes] = []

        def flush():
            nonlocal matrix, filled
            if not block:
                return
            part = np.frombuffer(b"".join(block), dtype=np.float32).reshape(len(block), -1)
            norms = np.linalg.norm(part, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            if matrix is None:
                matrix = np.empty((len(wanted), part.shape[1]), dtype=np.float16)
            matrix[filled:filled + len(block)] = part / norms
            ids[filled:filled + len(block)] = block_ids
            filled += len(block)
            block.clear()
            block_ids.clear()

        def take(rows):
            for chunk_id, embedding in rows:
                block_ids.append(chunk_id)
                block.append(embedding)
                if len(block) >= _BLOCK:
                    flush()

        if full:
            wanted_set = set(wanted.tolist())
            take(r for r in conn.execute("SELECT chunk_id, embedding FROM vec_chunks")
                 if r[0] in wanted_set)
        else:
            values = wanted.tolist()
            for start in range(0, len(values), _READ_BATCH):
                batch = values[start:start + _READ_BATCH]
                placeholders = ",".join("?" * len(batch))
                take(conn.execute(
                    f"SELECT chunk_id, embedding FROM vec_chunks WHERE chunk_id IN ({placeholders})",
                    batch,
                ))
        flush()
        return ids[:filled], matrix[:filled] if matrix is not None else _empty_state()[1]

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
        scores = np.empty(len(ids), dtype=np.float32)
        # Accelerate-backed float32 matmul raises spurious divide/overflow/invalid
        # RuntimeWarnings on macOS even for finite inputs.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for start in range(0, len(ids), _BLOCK):
                scores[start:start + _BLOCK] = (
                    matrix[start:start + _BLOCK].astype(np.float32) @ query
                )
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
    if not _cache.background_refresh:
        _cache.refresh(conn)
    return _cache.search(query_embedding, limit, allowed_sessions)


def invalidate_cache():
    """Make the next search re-check which chunks have vectors."""
    _cache.invalidate()

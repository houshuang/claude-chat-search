"""Vector search using limbic's VectorIndex with module-level caching.

At 18K chunks with 384-dim vectors, brute-force numpy dot product
is faster than sqlite-vec ANN (~2-5ms for full-corpus search).
"""

import numpy as np
from limbic.amygdala import VectorIndex

# Module-level cache
_vi: VectorIndex | None = None
_cached_count: int = 0


def _ensure_cache(conn) -> VectorIndex:
    """Load or refresh the VectorIndex cache from SQLite."""
    global _vi, _cached_count

    current_count = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]

    if _vi is None or current_count != _cached_count:
        _vi = VectorIndex()
        rows = list(conn.execute(
            "SELECT chunk_id, embedding FROM vec_chunks ORDER BY chunk_id"
        ))
        if rows:
            ids = [str(r[0]) for r in rows]
            vecs = np.vstack([
                np.frombuffer(r[1], dtype=np.float32).copy() for r in rows
            ])
            _vi.add(ids, vecs)
        _cached_count = current_count

    return _vi


def numpy_vector_search(conn, query_embedding: list[float], limit: int = 20) -> list[dict]:
    """Search embeddings using limbic's VectorIndex.

    Returns results in the same format as db.vector_search for drop-in replacement.
    """
    vi = _ensure_cache(conn)

    if vi.size == 0:
        return []

    query_vec = np.array(query_embedding, dtype=np.float32)
    results = vi.search(query_vec, limit=limit)

    return [
        {"chunk_id": int(r.id), "distance": 1.0 - r.score}
        for r in results
    ]


def invalidate_cache():
    """Force cache reload on next search (call after inserting new embeddings)."""
    global _vi, _cached_count
    _vi = None
    _cached_count = 0

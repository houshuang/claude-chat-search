from __future__ import annotations

import fcntl
import logging
import threading
from contextlib import contextmanager

from limbic.amygdala import EmbeddingModel

from . import db
from .db import get_unembedded_chunks, insert_embeddings, mark_embedding_failed

BATCH_SIZE = 256

logger = logging.getLogger(__name__)


class EmbeddingUnavailable(RuntimeError):
    """The model fails even on a trivial input, so no row can be blamed."""

_model: EmbeddingModel | None = None
# Held for model loading and for each encode call.  In the daemon, query
# embeddings for socket searches and background chunk embedding share one
# model; ENCODE_BATCH bounds how long a query waits behind a chunk batch.
_model_lock = threading.RLock()
ENCODE_BATCH = 32


@contextmanager
def embedding_lock():
    """Yield True only to the single process allowed to compute embeddings."""
    db.DB_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = db.DB_DIR / ".embedding.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _get_model() -> EmbeddingModel:
    global _model
    with _model_lock:
        if _model is None:
            _model = EmbeddingModel()
            _model._load_model()
        return _model


def embed_texts(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    import numpy as np
    model = _get_model()
    prepared = [model._prepare_text(t) for t in texts]
    vecs = []
    for start in range(0, len(prepared), ENCODE_BATCH):
        with _model_lock:
            vecs.append(model._model.encode(
                prepared[start:start + ENCODE_BATCH], batch_size=ENCODE_BATCH,
                normalize_embeddings=True, show_progress_bar=show_progress,
                convert_to_numpy=True,
            ).astype(np.float32))
    return np.vstack(vecs).tolist() if vecs else []


def embed_query(text: str) -> list[float]:
    model = _get_model()
    with _model_lock:
        return model.embed(text).tolist()


def process_embeddings(conn, callback=None) -> int:
    """Generate embeddings for all unembedded chunks. Returns count processed."""
    with embedding_lock() as acquired:
        if not acquired:
            return 0
        return _process_embeddings_locked(conn, callback)


def embed_rows(conn, rows: list[dict]) -> tuple[int, list[int]]:
    """Embed and store rows; returns (stored, skipped chunk ids).

    A failing batch is split in half until the rows the model cannot embed
    are isolated; those are marked failed and logged instead of being retried
    forever.  If the model fails on a trivial probe too, the failure is not
    about any row: EmbeddingUnavailable is raised and nothing is marked.
    Database errors propagate unchanged, so the batch is retried later.
    """
    if not rows:
        return 0, []
    try:
        embeddings = embed_texts([r["combined_text"] for r in rows])
    except Exception:
        try:
            embed_texts(["ok"])
        except Exception as probe_error:
            raise EmbeddingUnavailable("embedding model is failing") from probe_error
        if len(rows) == 1:
            logger.exception("Skipping chunk %s: it cannot be embedded", rows[0]["id"])
            mark_embedding_failed(conn, rows[0]["id"])
            return 0, [rows[0]["id"]]
        middle = len(rows) // 2
        stored_a, skipped_a = embed_rows(conn, rows[:middle])
        stored_b, skipped_b = embed_rows(conn, rows[middle:])
        return stored_a + stored_b, skipped_a + skipped_b
    insert_embeddings(conn, [r["id"] for r in rows], embeddings)
    return len(rows), []


def _process_embeddings_locked(conn, callback=None) -> int:
    total = 0

    while True:
        rows = get_unembedded_chunks(conn, BATCH_SIZE)
        if not rows:
            break

        stored, _skipped = embed_rows(conn, rows)

        total += stored
        if callback:
            callback(total)

    return total

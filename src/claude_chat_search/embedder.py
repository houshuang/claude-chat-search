from __future__ import annotations

import fcntl
from contextlib import contextmanager

from limbic.amygdala import EmbeddingModel

from . import db
from .db import get_unembedded_chunks, insert_embeddings

BATCH_SIZE = 256

_model: EmbeddingModel | None = None


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
    if _model is None:
        _model = EmbeddingModel()
    return _model


def embed_texts(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    model = _get_model()
    model._load_model()
    prepared = [model._prepare_text(t) for t in texts]
    import numpy as np
    vecs = model._model.encode(
        prepared, batch_size=64, normalize_embeddings=True,
        show_progress_bar=show_progress, convert_to_numpy=True,
    ).astype(np.float32)
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    model = _get_model()
    return model.embed(text).tolist()


def process_embeddings(conn, callback=None) -> int:
    """Generate embeddings for all unembedded chunks. Returns count processed."""
    with embedding_lock() as acquired:
        if not acquired:
            return 0
        return _process_embeddings_locked(conn, callback)


def _process_embeddings_locked(conn, callback=None) -> int:
    total = 0

    while True:
        rows = get_unembedded_chunks(conn, BATCH_SIZE)
        if not rows:
            break

        texts = [row["combined_text"] for row in rows]
        chunk_ids = [row["id"] for row in rows]

        embeddings = embed_texts(texts)
        insert_embeddings(conn, chunk_ids, embeddings)

        total += len(rows)
        if callback:
            callback(total)

    return total

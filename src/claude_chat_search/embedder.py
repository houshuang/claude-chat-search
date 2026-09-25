from __future__ import annotations

import fcntl
import logging
import threading
import time
from contextlib import contextmanager

import apsw

from . import db
from .db import embedding_mismatch, get_unembedded_chunks, insert_embeddings, mark_embedding_failed
from .models import ModelSpec, configured_model

BATCH_SIZE = 256

logger = logging.getLogger(__name__)


class EmbeddingUnavailable(RuntimeError):
    """The model fails even on a trivial input, so no row can be blamed."""

_model = None
_spec: ModelSpec | None = None
# Held for model loading and for each encode call.  In the daemon, query
# embeddings for socket searches and background chunk embedding share one
# model; ENCODE_BATCH bounds how long a query waits behind a chunk batch.
_model_lock = threading.RLock()
# Searches waiting for the model. The lock is not fair, so background embedding
# would otherwise re-take it between batches and keep a search waiting.
_queries_waiting = 0
_queries_waiting_lock = threading.Lock()
ENCODE_BATCH = 8


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


def _get_model():
    global _model, _spec
    with _model_lock:
        if _model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            _spec = configured_model()
            kwargs = {}
            gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
            if _spec.half_precision and gpu:
                kwargs["model_kwargs"] = {"dtype": torch.float16}
            # transformers 4.57 misreads the granite tokenizer as a Mistral one
            # and warns on every load; the suggested fix does not apply to it.
            tokenizer_log = logging.getLogger("transformers.tokenization_utils_base")
            level = tokenizer_log.level
            tokenizer_log.setLevel(logging.ERROR)
            try:
                _model = SentenceTransformer(_spec.name, **kwargs)
            finally:
                tokenizer_log.setLevel(level)
            _model.max_seq_length = _spec.max_tokens
        return _model, _spec


def _encode(model, texts: list[str], show_progress: bool = False, background: bool = False):
    import numpy as np
    # Batches of similar length pad less; results go back in input order.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs = []
    for start in range(0, len(order), ENCODE_BATCH):
        batch = [texts[i] for i in order[start:start + ENCODE_BATCH]]
        if background:
            while _queries_waiting:
                time.sleep(0.005)
        with _model_lock:
            vecs.append(model.encode(
                batch, batch_size=ENCODE_BATCH,
                normalize_embeddings=True, show_progress_bar=show_progress,
                convert_to_numpy=True,
            ).astype(np.float32))
    if not vecs:
        return np.empty((0, 0), dtype=np.float32)
    stacked = np.vstack(vecs)
    out = np.empty_like(stacked)
    out[order] = stacked
    return out


def embed_texts(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """Embed chunk texts as documents."""
    model, spec = _get_model()
    return _encode(model, [spec.document_prompt + t for t in texts], show_progress,
                   background=True).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a search query, with the model's query prompt."""
    global _queries_waiting
    model, spec = _get_model()
    with _queries_waiting_lock:
        _queries_waiting += 1
    try:
        return _encode(model, [spec.query_prompt + text])[0].tolist()
    finally:
        with _queries_waiting_lock:
            _queries_waiting -= 1


def process_embeddings(conn, callback=None) -> int:
    """Generate embeddings for all unembedded chunks. Returns count processed."""
    mismatch = embedding_mismatch(conn)
    if mismatch:
        logger.warning("Not embedding: %s", mismatch)
        return 0
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
    _store_with_retry(conn, [r["id"] for r in rows], embeddings)
    return len(rows), []


STORE_ATTEMPTS = 5


def _store_with_retry(conn, chunk_ids, embeddings) -> None:
    # The vectors are already computed; a writer that holds the lock past the
    # busy timeout should delay this batch, not throw the work away.
    for attempt in range(1, STORE_ATTEMPTS + 1):
        try:
            insert_embeddings(conn, chunk_ids, embeddings)
            return
        except apsw.BusyError:
            if attempt == STORE_ATTEMPTS:
                raise
            logger.warning("Database locked while storing %d embeddings; retry %d of %d",
                           len(chunk_ids), attempt, STORE_ATTEMPTS - 1)
            time.sleep(5 * attempt)


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

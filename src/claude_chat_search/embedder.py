from __future__ import annotations

from limbic.amygdala import EmbeddingModel

from .db import get_unembedded_chunks, insert_embeddings

BATCH_SIZE = 256

_model: EmbeddingModel | None = None


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

"""Embedding models the index can be built with.

The database records which model produced its vectors (the `meta` table);
vectors from one model are never compared with query embeddings from another.
CLAUDE_CHAT_SEARCH_MODEL selects a model from MODELS instead of DEFAULT_MODEL.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dim: int
    # Tokens per chunk the model reads; the rest of a longer chunk is ignored.
    max_tokens: int
    query_prompt: str = ""
    document_prompt: str = ""
    # Run in float16 on a GPU (Apple MPS or CUDA); float32 on CPU.
    half_precision: bool = False


LEGACY_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MODELS = {
    spec.name: spec
    for spec in (
        ModelSpec(LEGACY_MODEL, dim=384, max_tokens=128),
        # Chunks are at most ~600 cl100k tokens; 1024 model tokens cover them.
        ModelSpec(
            "ibm-granite/granite-embedding-311m-multilingual-r2",
            dim=768, max_tokens=1024, half_precision=True,
        ),
        ModelSpec(
            "ibm-granite/granite-embedding-97m-multilingual-r2",
            dim=384, max_tokens=1024, half_precision=True,
        ),
    )
}

DEFAULT_MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"


def configured_model() -> ModelSpec:
    name = os.environ.get("CLAUDE_CHAT_SEARCH_MODEL") or DEFAULT_MODEL
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown embedding model {name!r}; known models: {', '.join(MODELS)}"
        ) from None

import os

from openai import OpenAI

from .db import DB_DIR, get_unembedded_chunks, insert_embeddings

MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
ENV_FILE = DB_DIR / ".env"


def _load_api_key() -> str | None:
    """Load OPENAI_API_KEY from env var, falling back to ~/.claude-chat-search/.env."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


def get_client() -> OpenAI:
    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Set it in the environment or in "
            f"{ENV_FILE}"
        )
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_query(query: str) -> list[float]:
    client = get_client()
    response = client.embeddings.create(model=MODEL, input=[query])
    return response.data[0].embedding


def process_embeddings(conn, callback=None) -> int:
    """Generate embeddings for all unembedded chunks. Returns count processed."""
    client = get_client()
    total = 0

    while True:
        rows = get_unembedded_chunks(conn, BATCH_SIZE)
        if not rows:
            break

        texts = [row["combined_text"] for row in rows]
        chunk_ids = [row["id"] for row in rows]

        # Truncate very long texts to avoid API limits
        texts = [t[:8000] for t in texts]

        embeddings = embed_texts(client, texts)
        insert_embeddings(conn, chunk_ids, embeddings)

        total += len(rows)
        if callback:
            callback(total)

    return total

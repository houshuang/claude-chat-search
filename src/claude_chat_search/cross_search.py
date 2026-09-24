"""Cross-index search: queries both the chat index and the research index.

Each index is searched with a query embedding from its own model: the
research index (otak's index_research) embeds the query in its subprocess.
"""

import json
import subprocess
import sys
from pathlib import Path

from .db import embedding_mismatch
from .embedder import embed_query
from .search import RRF_K, _session_filter_sql, fused_session_results

OTAK_VENV_PYTHON = "/Users/stian/src/otak/.venv-otak/bin/python3"
RESEARCH_INDEX_DB = Path("/Users/stian/src/otak/data/research_index.db")


def _search_research_index(query: str, limit: int = 20) -> list[dict]:
    """Search the research index via subprocess (needs otak venv for numpy/index_research).

    The query is embedded there with the research index's own model:
    index_research.get_embedding_model() when otak provides it, otherwise the
    multilingual MiniLM the research index has always been built with.

    Returns list of dicts with score, source_file, section, content, tokens.
    """
    if not RESEARCH_INDEX_DB.exists():
        return []

    script = f"""
import sys, json
import numpy as np
sys.path.insert(0, "/Users/stian/src/otak/scripts")
import index_research
from index_research import load_index, search_index

query = sys.stdin.read()
if hasattr(index_research, "get_embedding_model"):
    model = index_research.get_embedding_model()
else:
    from limbic.amygdala import EmbeddingModel
    model = EmbeddingModel("paraphrase-multilingual-MiniLM-L12-v2")
q_emb = np.asarray(model.embed(query), dtype=np.float32)

emb_matrix, chunks = load_index()
if emb_matrix.shape[0] == 0:
    print("[]")
else:
    results = search_index(q_emb, emb_matrix, chunks, limit={limit})
    print(json.dumps(results))
"""

    try:
        proc = subprocess.run(
            [OTAK_VENV_PYTHON, "-c", script],
            input=query,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            print(f"Research search error: {proc.stderr[:300]}", file=sys.stderr)
            return []

        return json.loads(proc.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"Research search failed: {e}", file=sys.stderr)
        return []


def chat_candidates(
    conn,
    query: str,
    query_embedding: list[float] | None,
    limit: int = 10,
    project: str | None = None,
    branch: str | None = None,
    since: str | None = None,
    before: str | None = None,
    expand: bool = False,
) -> list[dict]:
    """Chat half of a cross search: best chunk per session, hybrid-search format."""
    session_filter = _session_filter_sql(conn, project, branch, since, before)
    fetch_limit = limit * 5
    return fused_session_results(
        conn, query, query_embedding, fetch_limit * 2, session_filter, expand,
        fetch_limit=fetch_limit,
    )


def cross_search(
    conn,
    query: str,
    limit: int = 10,
    project: str | None = None,
    branch: str | None = None,
    since: str | None = None,
    before: str | None = None,
    expand: bool = False,
    chat: tuple[list[dict], list[float]] | None = None,
) -> list[dict]:
    """Search both chat and research indices, merge with RRF.

    `chat` is a precomputed (chat_candidates() result, query embedding) pair,
    as returned by the daemon; only the candidates are used.  Returns a
    unified list where each result has a 'source' field: 'chat' or 'research'.
    """
    fetch_limit = limit * 5
    if chat is None:
        query_embedding = embed_query(query) if embedding_mismatch(conn) is None else None
        candidates = chat_candidates(
            conn, query, query_embedding, limit, project, branch, since, before, expand
        )
    else:
        candidates, _ = chat

    chat_results = [
        {
            "score": r["score"],
            "source": "chat",
            "chat_source": r.get("source", "claude"),
            "session_id": r["session_id"],
            "native_session_id": r.get("native_session_id", r["session_id"]),
            "project_path": r["project_path"],
            "slug": r["slug"],
            "git_branch": r["git_branch"],
            "user_content": r["user_content"],
            "assistant_content": r["assistant_content"],
            "timestamp": r["timestamp"],
            "turn_number": r["turn_number"],
            "message_count": r.get("message_count"),
            "first_message_at": r.get("first_message_at"),
            "last_message_at": r.get("last_message_at"),
        }
        for r in candidates
    ]

    # --- Research search ---
    research_raw = _search_research_index(query, limit=fetch_limit)
    research_results = []
    seen_files: set[str] = set()
    for r in research_raw:
        sf = r["source_file"]
        if sf in seen_files:
            continue
        seen_files.add(sf)
        research_results.append({
            "score": r["score"],
            "source": "research",
            "source_file": sf,
            "section": r.get("section", ""),
            "content": r.get("content", ""),
            "tokens": r.get("tokens", 0),
        })

    # --- Merge with RRF ---
    # Create ranked lists with synthetic IDs for RRF
    # Chat: use index as rank position
    # Research: use index as rank position
    all_results_by_id: dict[str, dict] = {}
    chat_ranked = []
    for i, r in enumerate(chat_results):
        rid = f"chat:{r['session_id']}"
        all_results_by_id[rid] = r
        chat_ranked.append({"chunk_id": rid})

    research_ranked = []
    for i, r in enumerate(research_results):
        rid = f"research:{r['source_file']}"
        all_results_by_id[rid] = r
        research_ranked.append({"chunk_id": rid})

    # RRF merge
    scores: dict[str, float] = {}
    for rank, item in enumerate(chat_ranked):
        rid = item["chunk_id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, item in enumerate(research_ranked):
        rid = item["chunk_id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (RRF_K + rank + 1)

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Dedup: if a research file was discussed in a chat session, keep chat version
    # (simple heuristic: check if research filename appears in chat content)
    final = []
    research_files_seen_in_chat = set()
    for rid, score in ranked:
        result = all_results_by_id[rid]
        result["score"] = score

        if result["source"] == "chat":
            # Check if any research file was discussed in this chat
            content = (result.get("user_content", "") + " " +
                       result.get("assistant_content", ""))
            for rf in seen_files:
                fname = Path(rf).name
                if fname in content:
                    research_files_seen_in_chat.add(rf)

        if result["source"] == "research":
            if result["source_file"] in research_files_seen_in_chat:
                continue

        final.append(result)
        if len(final) >= limit:
            break

    return final

"""Cross-index search: queries both the chat index and the research index.

Both indices use 384-dim embeddings, so we embed the query once
and search both with the same vector.
"""

import json
import subprocess
import sys
from pathlib import Path

from .db import fts_search, get_chunks_by_ids
from .embedder import embed_query
from .search import RRF_K, _build_session_filter, reciprocal_rank_fusion
from .vector_search import numpy_vector_search

OTAK_VENV_PYTHON = "/Users/stian/src/otak/.venv-otak/bin/python3"
RESEARCH_INDEX_DB = Path("/Users/stian/src/otak/data/research_index.db")


def _search_research_index(query_embedding: list[float], limit: int = 20) -> list[dict]:
    """Search the research index via subprocess (needs otak venv for numpy/index_research).

    Returns list of dicts with score, source_file, section, content, tokens.
    """
    if not RESEARCH_INDEX_DB.exists():
        return []

    script = f"""
import sys, json, struct
import numpy as np
sys.path.insert(0, "/Users/stian/src/otak/scripts")
from index_research import load_index, search_index

embedding = json.loads(sys.stdin.read())
q_emb = np.array(embedding, dtype=np.float32)

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
            input=json.dumps(query_embedding),
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


def cross_search(
    conn,
    query: str,
    limit: int = 10,
    project: str | None = None,
    branch: str | None = None,
    since: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """Search both chat and research indices, merge with RRF.

    Returns a unified list where each result has a 'source' field: 'chat' or 'research'.
    """
    allowed_sessions = _build_session_filter(conn, project, branch, since, before)

    fetch_limit = limit * 5
    query_embedding = embed_query(query)

    # --- Chat search (vector + FTS) ---
    vec_results = numpy_vector_search(conn, query_embedding, limit=fetch_limit)
    fts_results = fts_search(conn, query, limit=fetch_limit)

    # Assign synthetic chunk_ids for RRF: chat results use real chunk_ids,
    # research results use negative IDs to avoid collision
    chat_fused = reciprocal_rank_fusion([vec_results, fts_results])

    top_ids = [cid for cid, _ in chat_fused[:fetch_limit * 2]]
    chunks = get_chunks_by_ids(conn, top_ids)
    chunk_map = {c["id"]: c for c in chunks}

    # Build chat results (deduplicated by session)
    chat_results = []
    seen_sessions: set[str] = set()
    for cid, score in chat_fused:
        chunk = chunk_map.get(cid)
        if chunk is None:
            continue
        sid = chunk["session_id"]
        if allowed_sessions is not None and sid not in allowed_sessions:
            continue
        if sid in seen_sessions:
            continue
        seen_sessions.add(sid)
        chat_results.append({
            "score": score,
            "source": "chat",
            "session_id": sid,
            "project_path": chunk["project_path"],
            "slug": chunk["slug"],
            "git_branch": chunk["git_branch"],
            "user_content": chunk["user_content"],
            "assistant_content": chunk["assistant_content"],
            "timestamp": chunk["timestamp"],
            "turn_number": chunk["turn_number"],
            "message_count": chunk.get("message_count"),
            "first_message_at": chunk.get("first_message_at"),
            "last_message_at": chunk.get("last_message_at"),
        })

    # --- Research search ---
    research_raw = _search_research_index(query_embedding, limit=fetch_limit)
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

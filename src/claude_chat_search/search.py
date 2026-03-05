from .db import file_search as db_file_search, fts_search, get_chunks_by_ids, text_search, vector_search
from .embedder import embed_query

RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Each list is [{chunk_id, ...}, ...] in ranked order.
    Returns [(chunk_id, score)] sorted by descending score.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _build_session_filter(
    conn,
    project: str | None = None,
    branch: str | None = None,
    since: str | None = None,
    before: str | None = None,
) -> set[str] | None:
    """Pre-filter session IDs by metadata. Returns None if no filters active."""
    if not any([project, branch, since, before]):
        return None

    conditions = []
    params: list[str] = []

    if project:
        conditions.append("project_path LIKE '%' || ? || '%'")
        params.append(project)
    if branch:
        conditions.append("git_branch LIKE '%' || ? || '%'")
        params.append(branch)
    if since:
        conditions.append("last_message_at >= ?")
        params.append(since)
    if before:
        conditions.append("first_message_at <= ?")
        params.append(before)

    where = " AND ".join(conditions)
    rows = list(conn.execute(
        f"SELECT session_id FROM sessions WHERE {where}", params
    ))
    return {r[0] for r in rows}


def hybrid_search(
    conn,
    query: str,
    limit: int = 10,
    project: str | None = None,
    branch: str | None = None,
    since: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """Run hybrid semantic + keyword search with RRF merging.

    Deduplicates by session — returns the best-matching chunk per session.
    """
    allowed_sessions = _build_session_filter(conn, project, branch, since, before)

    fetch_limit = limit * 5

    query_embedding = embed_query(query)
    vec_results = vector_search(conn, query_embedding, limit=fetch_limit)
    fts_results = fts_search(conn, query, limit=fetch_limit)

    fused = reciprocal_rank_fusion([vec_results, fts_results])

    top_ids = [cid for cid, _ in fused[:fetch_limit * 2]]
    chunks = get_chunks_by_ids(conn, top_ids)

    chunk_map = {c["id"]: c for c in chunks}

    # Deduplicate by session: keep the highest-scoring chunk per session
    seen_sessions: dict[str, dict] = {}
    for cid, score in fused:
        chunk = chunk_map.get(cid)
        if chunk is None:
            continue
        sid = chunk["session_id"]
        if allowed_sessions is not None and sid not in allowed_sessions:
            continue
        if sid in seen_sessions:
            continue

        seen_sessions[sid] = {
            "chunk_id": cid,
            "score": score,
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
        }
        if len(seen_sessions) >= limit:
            break

    return list(seen_sessions.values())


def grep_search(
    conn,
    query: str,
    limit: int = 10,
    project: str | None = None,
    branch: str | None = None,
    since: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """Exact substring search across chunk text.

    Useful for file paths, branch names, error messages, and other exact strings
    that don't work well with FTS5 or semantic search.
    """
    allowed_sessions = _build_session_filter(conn, project, branch, since, before)

    results = text_search(conn, query, limit=limit * 5)

    seen_sessions: dict[str, dict] = {}
    for r in results:
        sid = r["session_id"]
        if allowed_sessions is not None and sid not in allowed_sessions:
            continue
        if sid in seen_sessions:
            continue

        seen_sessions[sid] = {
            "chunk_id": r["id"],
            "score": 1.0,
            "session_id": sid,
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
        if len(seen_sessions) >= limit:
            break

    return list(seen_sessions.values())


def file_search(
    conn,
    query: str,
    limit: int = 10,
    project: str | None = None,
    branch: str | None = None,
    since: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """Search sessions by file path in metadata.

    Uses the files_touched JSON field stored on sessions during indexing.
    """
    allowed_sessions = _build_session_filter(conn, project, branch, since, before)

    results = db_file_search(conn, query, limit=limit * 3)

    seen_sessions: dict[str, dict] = {}
    for r in results:
        sid = r["session_id"]
        if allowed_sessions is not None and sid not in allowed_sessions:
            continue
        if sid in seen_sessions:
            continue

        seen_sessions[sid] = {
            "chunk_id": 0,
            "score": 1.0,
            "session_id": sid,
            "project_path": r["project_path"],
            "slug": r["slug"],
            "git_branch": r["git_branch"],
            "user_content": f"[Session touched file matching: {query}]",
            "assistant_content": "",
            "timestamp": r.get("first_message_at"),
            "turn_number": 0,
            "message_count": r.get("message_count"),
            "first_message_at": r.get("first_message_at"),
            "last_message_at": r.get("last_message_at"),
        }
        if len(seen_sessions) >= limit:
            break

    return list(seen_sessions.values())

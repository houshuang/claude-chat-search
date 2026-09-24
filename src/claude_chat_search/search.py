from limbic.amygdala import expand_query, multi_list_rrf

from .db import (
    embedding_mismatch,
    file_search as db_file_search,
    fts_search,
    get_chunks_by_ids,
    get_project_paths_for_remote,
    text_search,
)
from .embedder import embed_query
from .vector_search import numpy_vector_search

RRF_K = 60
RERANK_POOL = 4


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


def _expand_project_via_remote(conn, project: str) -> list[str] | None:
    """If a project filter matches sessions with a git_remote, expand to all paths sharing that remote.

    Returns a list of project_path substrings to search, or None if no expansion needed.
    """
    # Find git_remote(s) for sessions matching this project substring
    rows = list(conn.execute(
        "SELECT DISTINCT git_remote FROM sessions WHERE project_path LIKE '%' || ? || '%' AND git_remote IS NOT NULL",
        (project,),
    ))
    if not rows:
        return None

    # Collect all project paths sharing any of these remotes
    all_paths = set()
    for (remote,) in rows:
        for path in get_project_paths_for_remote(conn, remote):
            all_paths.add(path)

    return list(all_paths) if len(all_paths) > 1 else None


def _session_filter_sql(
    conn,
    project: str | None = None,
    branch: str | None = None,
    since: str | None = None,
    before: str | None = None,
    source: str | None = None,
) -> tuple[str, list] | None:
    """WHERE clause and bindings over sessions for the metadata filters.

    Returns None if no filters are active.  The project filter auto-expands
    across multiple checkouts of the same repo by looking up git_remote.
    """
    if not any([project, branch, since, before, source]):
        return None

    conditions = []
    params: list[str] = []

    if project:
        # Try to expand to all checkouts sharing the same git remote
        expanded_paths = _expand_project_via_remote(conn, project)
        if expanded_paths:
            placeholders = " OR ".join("project_path = ?" for _ in expanded_paths)
            conditions.append(f"({placeholders})")
            params.extend(expanded_paths)
        else:
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
    if source:
        conditions.append("source = ?")
        params.append(source)

    return " AND ".join(conditions), params


def _sessions_matching(conn, session_filter: tuple[str, list] | None) -> set[str] | None:
    if session_filter is None:
        return None
    where, params = session_filter
    return {r[0] for r in conn.execute(
        f"SELECT session_id FROM sessions WHERE {where}", params
    )}


def _build_session_filter(
    conn,
    project: str | None = None,
    branch: str | None = None,
    since: str | None = None,
    before: str | None = None,
    source: str | None = None,
) -> set[str] | None:
    """Pre-filter session IDs by metadata. Returns None if no filters active."""
    return _sessions_matching(
        conn, _session_filter_sql(conn, project, branch, since, before, source)
    )


def _result_from_chunk(chunk: dict, score: float) -> dict:
    sid = chunk["session_id"]
    return {
        "chunk_id": chunk["id"],
        "score": score,
        "session_id": sid,
        "source": chunk.get("source", "claude"),
        "native_session_id": chunk.get("native_session_id", sid),
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


def fused_session_results(
    conn,
    query: str,
    query_embedding: list[float] | None,
    limit: int,
    session_filter: tuple[str, list] | None = None,
    expand: bool = False,
    fetch_limit: int | None = None,
) -> list[dict]:
    """Fuse vector and FTS rankings and keep the best chunk of up to `limit` sessions.

    Each ranking contributes fetch_limit chunks (default 5 x limit).  The
    session filter is applied while candidates are selected, so a narrow
    filter still gets a full candidate list from inside the filter.  Without
    a query embedding, or when the stored vectors come from another model
    (see db.embedding_mismatch), only the FTS ranking is used.
    """
    allowed_sessions = _sessions_matching(conn, session_filter)
    if allowed_sessions is not None and not allowed_sessions:
        return []

    fetch_limit = fetch_limit or limit * 5
    vec_results = []
    if query_embedding is not None and embedding_mismatch(conn) is None:
        vec_results = numpy_vector_search(
            conn, query_embedding, limit=fetch_limit, allowed_sessions=allowed_sessions
        )
    fts_results = fts_search(conn, query, limit=fetch_limit, session_filter=session_filter)

    if expand:
        fused = _expanded_search(
            conn, query, vec_results, fts_results, fetch_limit,
            allowed_sessions, session_filter,
        )
    else:
        fused = reciprocal_rank_fusion([vec_results, fts_results])

    top_ids = [cid for cid, _ in fused[:fetch_limit * 2]]
    chunk_map = {c["id"]: c for c in get_chunks_by_ids(conn, top_ids)}

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
        seen_sessions[sid] = _result_from_chunk(chunk, score)
        if len(seen_sessions) >= limit:
            break
    return list(seen_sessions.values())


def hybrid_search(
    conn,
    query: str,
    limit: int = 10,
    project: str | None = None,
    branch: str | None = None,
    since: str | None = None,
    before: str | None = None,
    do_rerank: bool = False,
    expand: bool = False,
    source: str | None = None,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """Run hybrid semantic + keyword search with RRF merging.

    Deduplicates by session — returns the best-matching chunk per session.

    When expand=True, uses LLM query expansion (lex/vec/hyde variants) to
    generate additional sub-queries and fuses results across all of them
    using multi-list RRF with top-rank bonuses. Adds ~3-5s latency but
    typically 3-5x score improvement.

    With do_rerank, the cross-encoder re-scores RERANK_POOL x limit
    candidate sessions before the list is cut to `limit`.
    """
    session_filter = _session_filter_sql(conn, project, branch, since, before, source)
    if query_embedding is None and embedding_mismatch(conn) is None:
        query_embedding = embed_query(query)

    pool = limit * RERANK_POOL if do_rerank else limit
    results = fused_session_results(
        conn, query, query_embedding, pool, session_filter, expand
    )

    if do_rerank and results:
        from limbic.amygdala import Result as LimbicResult, rerank
        limbic_results = [
            LimbicResult(
                id=str(r["chunk_id"]),
                score=r["score"],
                content=(r.get("user_content", "") + "\n" + r.get("assistant_content", "")),
                source="hybrid",
            )
            for r in results
        ]
        reranked = rerank(query, limbic_results)
        result_map = {str(r["chunk_id"]): r for r in results}
        results = []
        for lr in reranked:
            orig = result_map[lr.id]
            orig["score"] = lr.score
            results.append(orig)

    return results[:limit]


def _expanded_search(conn, query, vec_results, fts_results, fetch_limit,
                     allowed_sessions=None, session_filter=None):
    """Run LLM query expansion and fuse all results with multi-list RRF."""
    expanded = expand_query(query)

    ranked_lists = [vec_results, fts_results]
    labels = ["vec:original", "fts:original"]

    for eq in expanded:
        if eq.type == "lex":
            ranked_lists.append(fts_search(
                conn, eq.query, limit=fetch_limit, session_filter=session_filter
            ))
            labels.append(f"fts:{eq.query[:30]}")
        elif eq.type in ("vec", "hyde") and embedding_mismatch(conn) is None:
            emb = embed_query(eq.query)
            ranked_lists.append(numpy_vector_search(
                conn, emb, limit=fetch_limit, allowed_sessions=allowed_sessions
            ))
            labels.append(f"vec:{eq.type}:{eq.query[:25]}")

    traced = multi_list_rrf(
        ranked_lists, labels, k=RRF_K,
        id_fn=lambda item: item["chunk_id"],
    )
    return [(r.id, r.score) for r in traced]


def grep_search(
    conn,
    query: str,
    limit: int = 10,
    project: str | None = None,
    branch: str | None = None,
    since: str | None = None,
    before: str | None = None,
    source: str | None = None,
) -> list[dict]:
    """Exact substring search across chunk text.

    Useful for file paths, branch names, error messages, and other exact strings
    that don't work well with FTS5 or semantic search.
    """
    allowed_sessions = _build_session_filter(
        conn, project, branch, since, before, source
    )

    results = text_search(conn, query, limit=limit * 5, source=source)

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
            "source": r.get("source", "claude"),
            "native_session_id": r.get("native_session_id", sid),
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
    source: str | None = None,
) -> list[dict]:
    """Search sessions by file path in metadata.

    Uses the files_touched JSON field stored on sessions during indexing.
    """
    allowed_sessions = _build_session_filter(
        conn, project, branch, since, before, source
    )

    results = db_file_search(conn, query, limit=limit * 3, source=source)

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
            "source": r.get("source", "claude"),
            "native_session_id": r.get("native_session_id", sid),
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

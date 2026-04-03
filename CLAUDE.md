# claude-chat-search

Semantic search over Claude Code conversations. Indexes JSONL conversation files, chunks them, embeds with a 384-dim model, and stores in SQLite with sqlite-vec for vector search + FTS5 for text search.

## Architecture

- `db.py` — SQLite schema, CRUD, vector/FTS search. Uses APSW + sqlite-vec extension.
- `cli.py` — Click CLI: `init`, `index`, `search`, `daemon start/stop`, etc.
- `daemon.py` — Background process that watches for new/changed conversations and indexes them.
- `parser.py` — Parses Claude Code JSONL conversation files, extracts metadata.
- `chunker.py` — Splits conversations into searchable chunks.
- `embedder.py` — Embeds chunks (384-dim model).
- `vector_search.py` — NumPy-cached vector search (faster than sqlite-vec for large result sets).
- `summarizer.py` — LLM-based topic summarization of sessions.
- `cross_search.py` — Cross-index search across multiple chat indexes.

## DB Schema (index.db in ~/.claude-chat-search/)

Tables: `sessions`, `subagents`, `chunks`, `chunks_fts` (FTS5), `vec_chunks` (sqlite-vec).

### FK Delete Order — CRITICAL

When deleting a session, delete children in FK-dependency order:
1. `vec_chunks` (by chunk_id from chunks)
2. `chunks` (FK → sessions)
3. `subagents` (FK → sessions)
4. `sessions`

All session deletion goes through `delete_session_data()` in `db.py`. If you add a new table with a FK to sessions, you MUST update that function.

## Tech Stack

- Python 3.12, APSW (not stdlib sqlite3), sqlite-vec, FTS5
- Click for CLI
- Installed as editable package from `.venv/` in the project dir
- Entry point: `claude-chat-search` CLI command

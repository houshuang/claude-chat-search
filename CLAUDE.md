# claude-chat-search

Semantic search over Claude Code and Codex conversations. Indexes JSONL conversation files, chunks them, embeds with a 384-dim model, and stores in SQLite with sqlite-vec for vector search + FTS5 for text search.

## Architecture

- `db.py` — SQLite schema, CRUD, vector/FTS search. Uses APSW + sqlite-vec extension.
- `cli.py` — Click CLI: `init`, `index`, `search`, `daemon start/stop`, etc.
- `daemon.py` — Background process that indexes queued conversations, defers work that arrives during a session's cooldown, and rescans fingerprints hourly.
- `parser.py` — Parses Claude Code JSONL conversation files, extracts metadata.
- `codex_parser.py` — Parses only visible conversation from Codex rollout files; excludes system/developer prompts, reasoning, tools, outputs, state, and subagent rollouts.
- `sources.py` — Source-neutral discovery/parser dispatch.
- `chunker.py` — Splits conversations into searchable chunks.
- `embedder.py` — Embeds chunks (384-dim model).
- `vector_search.py` — NumPy-cached vector search (faster than sqlite-vec for large result sets). The cache refreshes incrementally when `PRAGMA data_version` changes and filters by session before top-k.
- `search_service.py` — Unix-socket search server run by the daemon (`search.sock` in the index directory, mode 0600) and the CLI client that falls back to in-process search.
- `summarizer.py` — LLM-based topic summarization of sessions.
- `cross_search.py` — Cross-index search across multiple chat indexes.
- `search.py` also supports `expand=True` for LLM query expansion (lex/vec/hyde variants via limbic's `expand_query` + `multi_list_rrf`).

## DB Schema (index.db in ~/.claude-chat-search/, or $CLAUDE_CHAT_SEARCH_HOME)

Tables: `sessions`, `subagents`, `chunks`, `chunks_fts` (FTS5), `vec_chunks` (sqlite-vec).

Codex sessions use namespaced internal IDs (`codex:<native-id>`) and retain the native ID in `sessions.native_session_id`. Since September 2026 Codex rollouts carry the conversation only as `response_item` records of type `message`; `codex_parser.py` takes role `user` `input_text` (stripped of injected AGENTS.md / `<environment_context>` / other wrapped context blocks) and role `assistant` `output_text`, and nothing else. Never feed other `response_item` types (reasoning, tool calls and outputs, inter-agent `agent_message`) or developer messages into chunks. Older rollouts that still have `event_msg` `user_message`/`agent_message` are read from those instead.

Schema changes go in `db._migrate()` together with a bump of `db.SCHEMA_VERSION`; `init_db()` returns immediately once `PRAGMA user_version` has reached it. Read-only commands use `get_read_connection()`.

`chunks.embedded` is 0 (pending), 1 (embedded) or -1 (the model failed on this chunk; skipped until `reembed`).

### FK Delete Order — CRITICAL

When deleting a session, delete children in FK-dependency order:
1. `vec_chunks` (by chunk_id from chunks)
2. `chunks` (FK → sessions)
3. `subagents` (FK → sessions)
4. `sessions`

All session deletion goes through `delete_session_data()` in `db.py`. If you add a new table with a FK to sessions, you MUST update that function.

Re-indexing a session goes through `sync_session_chunks()`, which matches stored chunks on (turn_number, `content_hash`), keeps matching rows and their vectors, deletes the rest (vectors first) and inserts only new chunks. `content_hash` is NULL on rows indexed before schema version 2 and is filled in the first time the session is re-indexed.

## Tech Stack

- Python 3.12, APSW (not stdlib sqlite3), sqlite-vec, FTS5
- Click for CLI
- Installed as editable package from `.venv/` in the project dir
- Entry point: `claude-chat-search` CLI command

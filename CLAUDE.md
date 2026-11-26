# claude-chat-search

Semantic search over Claude Code and Codex conversations. Indexes JSONL conversation files, chunks them, embeds with a local sentence-transformers model (`models.py`), and stores in SQLite with sqlite-vec for vector search + FTS5 for text search.

## Architecture

- `db.py` — SQLite schema, CRUD, vector/FTS search. Uses APSW + sqlite-vec extension.
- `cli.py` — Click CLI: `init`, `index`, `search`, `daemon start/stop`, etc.
- `daemon.py` — Background process that indexes queued conversations, defers work that arrives during a session's cooldown, and rescans fingerprints hourly. It indexes the queue on every loop regardless of load; the full scan and the embedding worker wait while the load average exceeds the core count (at most an hour). It never loads the embedding model: every `EMBED_INTERVAL` it starts `embed-pending` as a child process when chunks are pending. A watchdog dumps all stacks and exits when the loop stalls for `WATCHDOG_SECONDS`; `daemon.heartbeat` records the last queue pass, and search warns when the queue has waited longer than `INDEX_LAG_WARN_SECONDS`, and exits for a launchd restart when its memory footprint passes `MAX_FOOTPRINT_BYTES` (only with `CLAUDE_CHAT_SEARCH_SUPERVISED=1`, set by the plist).
- `parser.py` — Parses Claude Code JSONL conversation files, extracts metadata.
- `codex_parser.py` — Parses only visible conversation from Codex rollout files; excludes system/developer prompts, reasoning, tools, outputs, state, and subagent rollouts.
- `sources.py` — Source-neutral discovery/parser dispatch.
- `chunker.py` — Splits conversations into searchable chunks.
- `models.py` — Embedding model specs (name, dim, max tokens, query/document prompts); `CLAUDE_CHAT_SEARCH_MODEL` overrides the default.
- `embedder.py` — Embeds chunks and queries with the configured model.
- `vector_search.py` — NumPy-cached vector search (faster than sqlite-vec for large result sets), held as float16 and scored in float32 blocks. The cache refreshes incrementally when `PRAGMA data_version` changes and filters by session before top-k.
- `search_service.py` — Unix-socket search server (`search.sock` in the index directory, mode 0600), a process of its own: `ask()` in the CLI starts it on the first search, it embeds pending chunks while up, and it exits after `SERVER_IDLE_SECONDS` without a search. The CLI must never load the model: without a ready server it searches with `semantic=False`. Tests set `CLAUDE_CHAT_SEARCH_NO_SERVER=1` so none is started.
- `summarizer.py` — LLM-based topic summarization of sessions.
- `cross_search.py` — Cross-index search across multiple chat indexes.
- `search.py` also supports `expand=True` for LLM query expansion (lex/vec/hyde variants via limbic's `expand_query` + `multi_list_rrf`).

## DB Schema (index.db in ~/.claude-chat-search/, or $CLAUDE_CHAT_SEARCH_HOME)

Tables: `sessions`, `subagents`, `chunks`, `chunks_fts` (FTS5), `chunks_tri` (FTS5 trigram, behind `--grep`), `vec_chunks` (sqlite-vec), `meta` (key/value). Both FTS tables are external-content tables kept in step with `chunks` by triggers.

`meta.embedding_model` / `embedding_dim` record the model behind `vec_chunks`. Never write vectors from another model into it: `db.embedding_mismatch()` is checked before every vector search and embedding pass, which fall back to FTS-only / skip on a mismatch. `migrate-embeddings` is the only way to change models; its progress is in `meta.migration_state` (`rechunk`, then `embed`), `migration_model` and `migration_cursor`.

Codex sessions use namespaced internal IDs (`codex:<native-id>`) and retain the native ID in `sessions.native_session_id`. Since September 2026 Codex rollouts carry the conversation only as `response_item` records of type `message`; `codex_parser.py` takes role `user` `input_text` (stripped of injected AGENTS.md / `<environment_context>` / other wrapped context blocks) and role `assistant` `output_text`, and nothing else. Never feed other `response_item` types (reasoning, tool calls and outputs, inter-agent `agent_message`) or developer messages into chunks. Older rollouts that still have `event_msg` `user_message`/`agent_message` are read from those instead.

Schema changes go in `db._migrate()` together with a bump of `db.SCHEMA_VERSION`; `init_db()` returns immediately once `PRAGMA user_version` has reached it. Read-only commands use `get_read_connection()`.

`chunks.embedded` is 0 (pending), 1 (embedded) or -1 (the model failed on this chunk; skipped until `reembed`). Look pending chunks up through `db.PENDING_CHUNKS_SQL`, which the partial index `chunks_pending_idx` serves; any other `embedded = 0` filter that selects columns outside an index reads the whole `chunks` table.

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

# Changelog

## 0.2.0 — 2026-09-25

### Upgrading from 0.1

1. Update the code (`git pull` in a clone, or `uv tool install --force git+https://github.com/houshuang/claude-chat-search`).
2. Restart the daemon. On first start the schema moves to version 3 and records the existing vectors as the old model; until step 3 finishes, search is keyword-only with a warning and nothing new is embedded.
3. Run `claude-chat-search migrate-embeddings`. It backs up the index, re-chunks every session, and re-embeds everything with the new model. The first run downloads the model (about 630 MB). On an index of 6k sessions it took about 100 minutes on an Apple Silicon Mac; searches keep working meanwhile. If it stops, run it again: it resumes.
4. Optionally load the weekly backup job (`com.claude-chat-search.backup.plist`, see the README).

To stay on the old model instead, set `CLAUDE_CHAT_SEARCH_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for the daemon, the Codex job and the CLI.

### Search quality

- New embedding model: `ibm-granite/granite-embedding-311m-multilingual-r2` (768 dimensions, 1,024-token input, run in float16 on Apple GPUs). The previous model read only the first 128 tokens of each chunk. On 53 known-answer queries (English, Norwegian, cross-lingual) session-level MRR of the full hybrid search went from 0.52 to 0.65 and recall@10 from 0.83 to 0.98.
- The index records which model made its vectors and never compares a query with vectors from another model.
- Filters (project, branch, date, source) are applied while candidates are selected, so narrow filters no longer come back empty.
- `--rerank` re-scores 4 × `--limit` candidate sessions instead of only the final results.
- Oversized paragraphs and long prompts are split, so chunks stay under about 600 tokens.

### Speed

- The daemon answers `search`, `resume` and the chat half of `cross` over a unix socket, with the model and vectors in memory: about 0.2–0.4 s per search instead of 9–12 s. The CLI falls back to in-process search when no daemon is ready.
- The daemon refreshes its vector copy in the background, appending new vectors; searches never wait for a refresh.
- Background embedding yields the model to waiting searches.
- Re-indexing a changed session keeps unchanged chunks and their vectors and embeds only new ones.

### Codex

- Parses Codex's current rollout format; the previous parser had silently indexed almost nothing since Codex changed it. Injected context (AGENTS.md, environment and plugin blocks) is stripped.
- `index --source codex` warns when rollouts yield no messages, and exits non-zero when all of them do.

### Reliability

- `index --all --force` no longer deletes sessions whose transcript is gone from disk. Claude Code deletes old transcripts, so the index is often the only copy; in 0.1 this command permanently lost them.
- `backup --keep N` verifies each backup with `integrity_check` and prunes old ones; a weekly launchd job is included.
- Every write transaction takes the lock up front (`BEGIN IMMEDIATE`), the periodic WAL checkpoint is `PASSIVE`, embedding inserts retry when the database is locked, and slow write transactions are logged with their caller.
- Searches open the database read-only; migrations run once, gated on `PRAGMA user_version`.
- The daemon retries sessions that changed during a cooldown instead of dropping them, rescans hourly, and isolates chunks the model cannot embed.
- Excluded projects match on the transcript's recorded `cwd`, so paths with hyphens work; `purge-excluded` removes sessions already indexed under excluded paths, and their queries are not logged.

## 0.1.0 — 2026-03-05

First release: hybrid keyword and semantic search over Claude Code conversations, daemon indexing, resume, subagent access.

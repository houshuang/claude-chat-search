# Changelog

## Unreleased

### Lighter daemon

- The daemon runs as a background process (`ProcessType Background`, `Nice 10`, low-priority I/O, two math threads). Reload its plist to apply this: `launchctl bootout gui/$(id -u)/com.claude-chat-search.daemon`, copy the plist to `~/Library/LaunchAgents/`, then `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.claude-chat-search.daemon.plist`.
- A changing session is re-indexed at most every 10 minutes (30 once it keeps changing), new chunks are embedded every 10 minutes, and the search server refreshes its vectors every 5 minutes. New conversation turns take up to about 20 minutes to become searchable.
- While the 1-minute load average exceeds the number of cores, embedding and the hourly full scan wait, for at most an hour. Queued conversations are indexed regardless of load, so keyword and `--grep` search always find a session within its cooldown.
- After each embedding pass the GPU allocator's cached memory is released, and under launchd (which sets `CLAUDE_CHAT_SEARCH_SUPERVISED=1`) the daemon restarts itself when its footprint passes 4 GB. The startup scan also waits for the load to drop.

### Indexing never waits for embedding

On 2 November 2026, with the load average above 20 and swap full, one embedding batch took two to four hours, and the daemon indexed nothing from 14:00 to 22:30. Sessions from that evening could not be found, even with `--grep`.

- An embedding pass stops after two minutes and embeds 32 chunks at a time; a remaining backlog is picked up a minute later. The queue is indexed between passes.
- Background embedding yields to a waiting search for at most five seconds per batch.
- A watchdog thread writes every thread's stack to `daemon.log` when the main loop has not come round for 30 minutes, and under launchd exits so the daemon restarts.
- The daemon records when it last processed the queue in `daemon.heartbeat`. `search` and `cross` print a warning when conversations have been waiting for more than 15 minutes, with a `grep` command for the transcripts.

### The model runs only while it is used

The daemon held the embedding model and every vector all day: 3.4 GB, 5.4 GB at its peak. Unloading the model in a running process gives back almost nothing (measured: 2.4 GB before, 2.3 GB after), so the model now lives in processes that exit.

- The daemon only indexes. It stays near 50 MB and never imports PyTorch.
- A search server holds the model and the vectors. The first search starts it and waits for it (27 s measured on a loaded machine; 0.7–1.0 s per search after that); it exits after 15 minutes without a search. It uses about 2.1 GB while it is up.
- Pending chunks are embedded by the search server while it is up, and otherwise by a worker the daemon starts every 30 minutes. Semantic search can therefore lag new turns by up to 30 minutes (10 before); keyword and `--grep` search do not lag.
- The CLI no longer loads the model when the server does not answer. That fallback took a median of 45 s and about 2 GB per search. It now answers from the keyword index, prints a note, and logs `"keyword_only": true`.
- A search whose caller timed out while waiting in the server's queue is dropped; before, later searches waited for it.
- The vectors are held once, as float16 (half the size), and are read in blocks; warm-up no longer makes four copies of them. Cosine scores are exact to about three decimals.
- `daemon status` also reports the search server.

Reload nothing: the daemon restarts itself when the code changes. A search server from before the update, if any, belongs to the old daemon and goes away with it.

### `--grep` uses a trigram index

`--grep` ran `LIKE '%text%'` over every chunk: a median of 4.9 s, up to 28 s.

- Schema version 5 adds `chunks_tri`, an FTS5 trigram index over the chunk text. On the author's index it took 45 s to build, added 232 MB, and answers the same searches in about 0.1 s.
- Queries shorter than three characters still read every chunk.
- Like the version 4 step below, run an indexing command once after updating so the build does not happen in the throttled daemon.

### Pending chunks are found through an index

Every embedding batch asked for chunks with `embedded = 0`, and no index answered that, so SQLite read the whole `chunks` table (924 MB on the author's index) to find a handful of rows at its end. With low-priority I/O and the machine swapping, one such read took more than 30 minutes: between 23 and 25 November 2026 the watchdog restarted the daemon six times in that query, and the queue went unindexed for half an hour each time.

- Schema version 4 adds the partial index `chunks_pending_idx` (`chunks(id) WHERE embedded = 0`). The lookup takes milliseconds and the index holds only the pending rows.
- The migration reads the `chunks` table once while it holds the write lock. Run any indexing command (for example `claude-chat-search index --source codex`) once after updating so it happens at normal I/O priority and not in the throttled daemon.

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

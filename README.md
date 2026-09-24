# claude-chat-search

Semantic search over your past Claude Code and Codex conversations. Indexes the JSONL conversation logs in `~/.claude/projects/`, `~/.codex/sessions/`, and `~/.codex/archived_sessions/` into a local SQLite database with hybrid search — full-text keyword search (FTS5) and vector similarity search (via [limbic](https://github.com/houshuang/limbic)), combined using Reciprocal Rank Fusion.

Codex indexing is deliberately conversation-only: visible user and agent messages are indexed. Developer/system prompts, context Codex injects into user messages (`AGENTS.md`, `<environment_context>`, `<user_instructions>` and similar blocks), reasoning, tool calls, tool outputs, token accounting, world state, and Codex subagent rollouts are excluded.

Claude Code keeps every session on disk, but `claude --resume` only lets you pick from a list. This tool lets you (or Claude, through the bundled skill) ask "where did we fix the auth bug?" and get back the session, project, branch and the matching turn, then resume it.

```
$ claude-chat-search search "flaky test in CI" -n 1

======================================================================
#1  score=0.0664  session=2f56cf03-f16...
    project: /home/me/src/myapp
    branch: main
    time: 2026-09-16 07:22 · 896 messages · 2h27m

  [Turn 14]
  Assistant: The test passes locally because your dev venv has every optional
  extra installed; CI installs only the declared dependencies...
```

## Install

```bash
uv tool install git+https://github.com/houshuang/claude-chat-search
```

Or from a clone, for development:

```bash
git clone https://github.com/houshuang/claude-chat-search.git
cd claude-chat-search
uv venv && uv pip install -e .
source .venv/bin/activate   # puts claude-chat-search on your PATH
```

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Indexing, keyword, semantic and reranked search run locally with no API keys. The first run downloads the embedding model (about 630 MB). Only `--expand` calls an LLM (Gemini Flash, needs `GEMINI_API_KEY`).

## Usage

### First-time setup

```bash
claude-chat-search init
```

Creates the database at `~/.claude-chat-search/index.db`, indexes all existing conversations, and generates embeddings using a local model (`ibm-granite/granite-embedding-311m-multilingual-r2`, 768-dim, reads up to 1024 tokens per chunk). See [Embedding model](#embedding-model).

Set `CLAUDE_CHAT_SEARCH_HOME` to keep the index, queue, logs, exclusion list and backups somewhere other than `~/.claude-chat-search` (useful for tests and throwaway copies). `CHAT_SEARCH_DB_PATH` still overrides the database file alone.

`init` defaults to Claude for backward compatibility. Add Codex history separately:

```bash
claude-chat-search index --source codex
```

### Incremental indexing

```bash
claude-chat-search index
```

Only indexes new or modified sessions since the last run. Use `--all --force` to re-index everything from scratch.

Re-indexing a changed session keeps every chunk whose turn number and text are unchanged, with its vector; only new or changed chunks are embedded. Chunks never exceed about 600 tokens: a paragraph longer than that is split at line breaks, then sentences, then words, and a long prompt gets chunks of its own. Sessions indexed before this rule can still hold much larger chunks; `migrate-embeddings` re-chunks them (see below), including sessions whose transcript is no longer on disk. (`reembed` recomputes vectors but keeps the chunks.)

Use `--source claude`, `--source codex`, or `--source all` to select inputs. File mtime and size fingerprints are persisted in SQLite, so unchanged Codex scans do not parse multi-gigabyte rollout files.

Codex's rollout format is undocumented and changes. If a Codex rollout in which the agent took a turn yields no messages, `index` prints a warning with the count; if every such rollout in the run yields nothing, `index` exits non-zero, which shows up as a failed run of the scheduled job.

Searches (`search`, `cross`, `show`, `recover`, `resume`) open the index read-only and never take the write lock. Schema migrations run once, when `PRAGMA user_version` is behind, from whichever command opens the database first.

### Search

```bash
claude-chat-search search "how to configure webhooks"
claude-chat-search search "deployment error" --project myapp --since 2w
claude-chat-search search "database migration" -n 5 --branch main
claude-chat-search search "exact error message" --grep
claude-chat-search search "important query" --rerank
claude-chat-search search "Tallinn walking tour" --source codex
```

Options:
- `-n` / `--limit` — number of results (default 10)
- `-p` / `--project` — filter by project path substring (auto-expands across multiple checkouts of the same repo via git remote detection)
- `-b` / `--branch` — filter by git branch name substring
- `--since` — only results after date (`YYYY-MM-DD`, `3d`, `2w`, `1m`)
- `--before` — only results before date
- `--grep` — exact substring search (skips semantic/FTS5, just matches raw text)
- `--file` — search by file path mentioned in session tool calls
- `--rerank` — re-score the top 4 × `--limit` sessions with a cross-encoder, then keep `--limit` (slower, +5-15% accuracy)
- `--expand` — LLM query expansion via Gemini Flash (needs `GEMINI_API_KEY`) for better recall on vague/cross-vocabulary queries (~3-5s extra, typically 3-5x score improvement). Generates keyword variants, semantic rephrases, and hypothetical document excerpts to bridge vocabulary gaps.
- `--source` — limit results to `claude`, `codex`, or search `all` (default)

Filters are applied while candidates are selected, in both the vector and the keyword ranking, so a narrow filter returns the best matches inside it rather than whatever survived a global top-k.

When the daemon is running, `search`, `resume` and the chat half of `cross` are answered by it over a unix socket (`search.sock` in the index directory, mode 0600). It keeps the embedding model and all vectors in memory, so a search takes a fraction of a second instead of the 8–11 seconds a cold process needs to load them. Without a daemon the CLI searches in-process as before. Entries in `search.log` served by the daemon carry `"daemon": true`.

Right after the daemon starts it spends up to a minute on its startup scan and loading the model; until it is ready it answers "not ready" and the CLI searches in-process, so a search never hangs on a warming daemon. The daemon's copy of the vectors is refreshed at most every 30 seconds, so a chunk embedded in the last half minute may be missing from vector results (keyword search sees it at once).

Measured on a 1.4 GB index (5.7k sessions, 114k chunks, granite-311m): median 0.2 s and p90 0.37 s per search through the daemon, against 8–11 s (p90 25 s) for a cold process.

### Embedding model

The index records which model produced its vectors (the `meta` table). A query is only ever compared with vectors from the model that embeds it: if the stored model differs from the configured one, `search`, `resume` and `cross` fall back to keyword search and print a warning, and nothing new is embedded until the index is migrated.

The default is `ibm-granite/granite-embedding-311m-multilingual-r2` (Apache 2.0, 311M parameters, 768 dimensions, run in float16 on Apple GPUs). It replaced `paraphrase-multilingual-MiniLM-L12-v2`, which reads only the first 128 tokens of a chunk, after a September 2026 comparison on 53 known-answer queries (English, Norwegian and cross-lingual) against the author's history: session-level MRR of the full hybrid search went from 0.52 to 0.65 and recall@10 from 0.83 to 0.98. Larger models (bge-m3, Qwen3-Embedding-0.6B, embeddinggemma-300m, multilingual-e5-large) ranked no better and would take 2.5–10 hours to embed the index. Models are listed in `models.py`; `CLAUDE_CHAT_SEARCH_MODEL` selects another one from that list (for example `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, the model used before September 2026).

To move an index to the configured model:

```bash
claude-chat-search migrate-embeddings
```

It backs up the index, re-chunks every session with the current chunker (from the transcript when it still exists, otherwise from the stored text, re-split to the current chunk size), recreates the vector table at the new width and embeds every chunk. It is safe to interrupt and run again; it resumes where it stopped. The daemon can keep running: keyword search works throughout. Until the switch to the new vector table (after re-chunking) semantic search is off; after it, semantic search covers the chunks embedded so far and a note says the migration is in progress. `--no-backup` skips the backup. On the author's machine (Apple M4 Pro, 5,800 sessions, 114,000 chunks after re-chunking) it took 100 minutes: three minutes of re-chunking, the rest embedding at about 20 chunks a second. The index grew from 1.1 to 1.4 GB.

### Back up the index

```bash
claude-chat-search backup
claude-chat-search backup --keep 4
```

Creates a consistent online SQLite backup under `~/.claude-chat-search/backups/`, runs `PRAGMA integrity_check` on the copy, and prints its SHA-256 checksum. If the check fails, the copy is renamed `*.failed-integrity`, nothing is pruned, and the command exits non-zero. `--keep N` then deletes all but the newest N `index-*.db` backups in that directory. The backup command intentionally performs no schema migration.

A weekly backup job (Sundays 04:30, keeping four) is included. Like the other plists it contains the author's paths; edit the Python path and log paths first:

```bash
cp com.claude-chat-search.backup.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.claude-chat-search.backup.plist
```

### Exclude projects

```bash
claude-chat-search exclude add /path/to/private-project
claude-chat-search exclude list
claude-chat-search purge-excluded --dry-run
claude-chat-search purge-excluded
```

`~/.claude-chat-search/excluded_projects.txt` holds one path per line; sessions in that directory or below are never indexed, and their queries are not written to `search.log`. Matching uses the transcript's recorded `cwd` and Claude's encoded project-directory name, so paths containing hyphens are matched correctly (an excluded `/a/priv` also excludes a hyphenated sibling such as `/a/priv-other`). `purge-excluded` deletes sessions, chunks, full-text rows and vectors already indexed under excluded paths; `--dry-run` only reports counts per excluded path.

### Inspect a session

```bash
claude-chat-search show <session-id> --turn 5
claude-chat-search show <session-id> --with-subagents
```

Partial session ID matching is supported. Use `--turn` to highlight a specific turn. The header shows subagent count when > 0. Use `--with-subagents` to append subagent summaries.

### Resume a past session

```bash
claude-chat-search resume "how we fixed the auth bug"
claude-chat-search resume "database migration" --project myapp --since 1w
claude-chat-search resume "refactoring plan" --fork
```

Searches for matching sessions, shows an interactive pick list, and resumes the selected session in Claude Code via `claude --resume`. Automatically `cd`s to the session's original working directory.

Options:
- Same filters as `search` (`-p`, `-b`, `--since`, `--before`, `-n`)
- `--fork` — fork the session (creates a new session branching off the original)

### Explore subagent conversations

Background agents (subagents) run during a session to handle parallel tasks. Their conversations are indexed as lightweight metadata and accessible on demand:

```bash
# List all subagents for a session
claude-chat-search subagents <session-id>

# Show a specific subagent conversation (partial ID matching)
claude-chat-search subagent <session-id> <agent-id>
claude-chat-search subagent <session-id> <agent-id> --raw  # untruncated
```

## Continuous indexing with daemon

Instead of spawning a subprocess on every tool call, a persistent daemon handles all indexing. A `PostToolUse` hook appends the transcript path to a queue file; the daemon picks it up every 2 seconds. A session re-indexed within its cooldown (60 seconds, five minutes for very active sessions) is deferred and indexed when the cooldown ends. An hourly scan compares transcript mtime and size fingerprints and picks up anything no hook queued.

### Start the daemon

```bash
# Foreground (Ctrl-C to stop)
claude-chat-search daemon run

# Background (detached)
claude-chat-search daemon start

# Check status
claude-chat-search daemon status

# Stop
claude-chat-search daemon stop
```

### Install as launchd service (auto-start on login, macOS)

The bundled plist contains the author's paths. Edit it first: point the first `ProgramArguments` entry at the Python in your install (`uv tool dir` then `claude-chat-search/bin/python`, or `.venv/bin/python` in a clone), and the two log paths at your home directory.

```bash
cp com.claude-chat-search.daemon.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.claude-chat-search.daemon.plist
```

To unload: `launchctl unload ~/Library/LaunchAgents/com.claude-chat-search.daemon.plist`

### Hook setup

Requires [jq](https://jqlang.org/). Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r .transcript_path >> ~/.claude-chat-search/.queue"
          }
        ]
      }
    ]
  }
}
```

The hook appends the active transcript path to the queue file. The daemon atomically renames it for processing, deduplicates paths, and only re-indexes sessions whose message count has changed — preserving existing embeddings for unchanged content.

`PostToolUse` does not fire after a final answer that used no tools, so the last turn of a session can wait for the hourly scan. Optionally queue the transcript when a turn or session ends as well, with the same command under `Stop` and `SessionEnd`:

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command", "command": "jq -r .transcript_path >> ~/.claude-chat-search/.queue" } ] }
    ],
    "SessionEnd": [
      { "hooks": [ { "type": "command", "command": "jq -r .transcript_path >> ~/.claude-chat-search/.queue" } ] }
    ]
  }
}
```

A chunk the embedding model cannot process is isolated by splitting its batch, logged, and marked `embedded = -1` so it is not retried forever; `reembed` resets it.

### Low-priority periodic Codex indexing

Codex does not expose the Claude-style post-tool hook used above. The included launchd job scans Codex history every three hours. Like the daemon plist, it contains the author's paths; edit the Python path and log paths first:

```bash
cp com.claude-chat-search.codex-index.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.claude-chat-search.codex-index.plist
```

The job runs with background scheduling, low-priority I/O, `Nice=15`, two model threads, and forced offline model loading. A no-change scan opens only transcript headers and normally completes in a few seconds. launchd does not run overlapping instances of the same job.

## Claude Code skill integration

Copy the skill definition so Claude Code can use this tool automatically:

```bash
mkdir -p ~/.claude/skills/claude-chat-search
cp SKILL.md ~/.claude/skills/claude-chat-search/SKILL.md
```

Then Claude Code will search your past conversations when you ask things like "remember when we discussed..." or "find that session where we fixed...".

## Architecture

Several processes write to one SQLite database: the daemon, the Codex job, and manual `index` runs. Every write transaction starts with `BEGIN IMMEDIATE` (`db.write_transaction`), because a deferred transaction that reads before it writes fails at once when another process commits in between, without waiting for the busy timeout. Embedding happens outside any transaction, and the daemon's periodic WAL checkpoint is `PASSIVE`, so it never holds the write lock while waiting for readers.


- **parser.py** — walks `~/.claude/projects/` and parses Claude Code JSONL conversation logs
- **codex_parser.py** — isolates the undocumented Codex rollout format and normalizes only visible user/agent conversation
- **sources.py** — dispatches source-specific discovery and parsing into the shared session/chunk model
- **chunker.py** — splits conversations into user/assistant turn pairs with token-aware splitting and paragraph-boundary overlap
- **models.py** — the embedding models the index can be built with, their widths, token limits and query/document prompts
- **embedder.py** — embeds chunks and queries locally with sentence-transformers (default `ibm-granite/granite-embedding-311m-multilingual-r2`)
- **vector_search.py** — in-memory numpy vector search, refreshed incrementally when the database changes, with session filters applied before top-k
- **db.py** — SQLite with FTS5 for keyword search and `sqlite-vec` for vector storage
- **search.py** — hybrid search (vector + keyword + grep + file) combined via Reciprocal Rank Fusion, deduplicated by session, with optional cross-encoder reranking and LLM query expansion (lex/vec/hyde variants) via limbic
- **daemon.py** — persistent indexer daemon: queue-based incremental indexing with deferred retries, message-count skip, startup and hourly fingerprint scans; also serves searches
- **search_service.py** — the daemon's unix-socket search server and the CLI's client for it
- **cli.py** — Click CLI exposing `init`, `index`, `search`, `resume`, `show`, `subagents`, `subagent`, `recover`, `reembed`, `migrate-embeddings`, `summarize`, `cross`, `backup`, `exclude`, `purge-excluded`, and `daemon` commands

`cross` (chat history plus a separate research-file index) and `summarize` (topic summaries) depend on the author's own tooling at fixed local paths and will not work on other machines as-is.

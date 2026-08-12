# claude-chat-search

Semantic search over your past Claude Code and Codex conversations. Indexes the JSONL conversation logs in `~/.claude/projects/`, `~/.codex/sessions/`, and `~/.codex/archived_sessions/` into a local SQLite database with hybrid search — full-text keyword search (FTS5) and vector similarity search (via [limbic](https://github.com/houshuang/limbic)), combined using Reciprocal Rank Fusion.

Codex indexing is deliberately conversation-only: visible user and agent messages are indexed. Developer/system prompts, reasoning, tool calls, tool outputs, token accounting, world state, and Codex subagent rollouts are excluded.

## Install

```bash
git clone https://github.com/houshuang/claude-chat-search.git
cd claude-chat-search
uv venv && uv pip install -e .
```

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). All search modes (keyword, semantic, reranking) work locally — no API keys needed.

## Usage

### First-time setup

```bash
claude-chat-search init
```

Creates the database at `~/.claude-chat-search/index.db`, indexes all existing conversations, and generates embeddings using a local model (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim).

`init` defaults to Claude for backward compatibility. Add Codex history separately:

```bash
claude-chat-search index --source codex
```

### Incremental indexing

```bash
claude-chat-search index
```

Only indexes new or modified sessions since the last run. Use `--all --force` to re-index everything from scratch.

Use `--source claude`, `--source codex`, or `--source all` to select inputs. File mtime and size fingerprints are persisted in SQLite, so unchanged Codex scans do not parse multi-gigabyte rollout files.

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
- `--rerank` — re-score results with a cross-encoder for better relevance (slower, +5-15% accuracy)
- `--expand` — LLM query expansion for better recall on vague/cross-vocabulary queries (~3-5s extra, typically 3-5x score improvement). Generates keyword variants, semantic rephrases, and hypothetical document excerpts to bridge vocabulary gaps.
- `--source` — limit results to `claude`, `codex`, or search `all` (default)

### Back up the index

```bash
claude-chat-search backup
```

Creates a consistent online SQLite backup under `~/.claude-chat-search/backups/` and prints its SHA-256 checksum. The backup command intentionally performs no schema migration.

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

Instead of spawning a subprocess on every tool call, a persistent daemon handles all indexing. A `PostToolUse` hook appends the transcript path to a queue file; the daemon picks it up every 2 seconds.

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

### Install as launchd service (auto-start on login)

```bash
cp com.claude-chat-search.daemon.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.claude-chat-search.daemon.plist
```

To unload: `launchctl unload ~/Library/LaunchAgents/com.claude-chat-search.daemon.plist`

### Hook setup

Add to `~/.claude/settings.json`:

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

### Low-priority periodic Codex indexing

Codex does not expose the Claude-style post-tool hook used above. The included launchd job scans Codex history every three hours:

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

- **parser.py** — walks `~/.claude/projects/` and parses Claude Code JSONL conversation logs
- **codex_parser.py** — isolates the undocumented Codex rollout format and normalizes only visible user/agent conversation
- **sources.py** — dispatches source-specific discovery and parsing into the shared session/chunk model
- **chunker.py** — splits conversations into user/assistant turn pairs with token-aware splitting and paragraph-boundary overlap
- **embedder.py** — generates embeddings via [limbic](https://github.com/houshuang/limbic)'s `EmbeddingModel` (`paraphrase-multilingual-MiniLM-L12-v2`, local, multilingual, 384-dim)
- **vector_search.py** — in-memory numpy vector search using limbic's `VectorIndex` with module-level caching
- **db.py** — SQLite with FTS5 for keyword search and `sqlite-vec` for vector storage
- **search.py** — hybrid search (vector + keyword + grep + file) combined via Reciprocal Rank Fusion, deduplicated by session, with optional cross-encoder reranking and LLM query expansion (lex/vec/hyde variants) via limbic
- **daemon.py** — persistent indexer daemon: queue-based incremental indexing, message-count skip, startup full scan
- **cli.py** — Click CLI exposing `init`, `index`, `search`, `resume`, `show`, `subagents`, `subagent`, `recover`, `reembed`, `summarize`, `cross`, and `daemon` commands

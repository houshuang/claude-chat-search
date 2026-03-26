# claude-chat-search

Semantic search over your past Claude Code conversations. Indexes the JSONL conversation logs in `~/.claude/projects/` into a local SQLite database with hybrid search — full-text keyword search (FTS5) and vector similarity search (via [limbic](https://github.com/houshuang/limbic)), combined using Reciprocal Rank Fusion.

## Install

```bash
git clone https://github.com/houshuang/claude-tool.git
cd claude-tool/claude-chat-search
uv venv && uv pip install -e .
```

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). All search modes (keyword, semantic, reranking) work locally — no API keys needed.

## Usage

### First-time setup

```bash
claude-chat-search init
```

Creates the database at `~/.claude-chat-search/index.db`, indexes all existing conversations, and generates embeddings using a local model (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim).

### Incremental indexing

```bash
claude-chat-search index
```

Only indexes new or modified sessions since the last run. Use `--all --force` to re-index everything from scratch.

### Search

```bash
claude-chat-search search "how to configure webhooks"
claude-chat-search search "deployment error" --project myapp --since 2w
claude-chat-search search "database migration" -n 5 --branch main
claude-chat-search search "exact error message" --grep
claude-chat-search search "important query" --rerank
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

### Inspect a session

```bash
claude-chat-search show <session-id> --turn 5
claude-chat-search show <session-id> --with-subagents
```

Partial session ID matching is supported. Use `--turn` to highlight a specific turn. The header shows subagent count when > 0. Use `--with-subagents` to append subagent summaries.

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

## Claude Code skill integration

Copy the skill definition so Claude Code can use this tool automatically:

```bash
mkdir -p ~/.claude/skills/claude-chat-search
cp SKILL.md ~/.claude/skills/claude-chat-search/SKILL.md
```

Then Claude Code will search your past conversations when you ask things like "remember when we discussed..." or "find that session where we fixed...".

## Architecture

- **parser.py** — walks `~/.claude/projects/` and parses JSONL conversation logs; extracts git branch, slug, file paths, and message counts from session metadata and tool calls
- **chunker.py** — splits conversations into user/assistant turn pairs with token-aware splitting and paragraph-boundary overlap
- **embedder.py** — generates embeddings via [limbic](https://github.com/houshuang/limbic)'s `EmbeddingModel` (`paraphrase-multilingual-MiniLM-L12-v2`, local, multilingual, 384-dim)
- **vector_search.py** — in-memory numpy vector search using limbic's `VectorIndex` with module-level caching
- **db.py** — SQLite with FTS5 for keyword search and `sqlite-vec` for vector storage
- **search.py** — hybrid search (vector + keyword + grep + file) combined via Reciprocal Rank Fusion, deduplicated by session, with optional cross-encoder reranking via limbic
- **daemon.py** — persistent indexer daemon: queue-based incremental indexing, message-count skip, startup full scan
- **cli.py** — Click CLI exposing `init`, `index`, `search`, `show`, `subagents`, `subagent`, `recover`, `reembed`, `summarize`, `cross`, and `daemon` commands

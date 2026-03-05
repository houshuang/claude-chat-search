# claude-chat-search

Semantic search over your past Claude Code conversations. Indexes the JSONL conversation logs in `~/.claude/projects/` into a local SQLite database with hybrid search — full-text keyword search (FTS5) and vector similarity search (via `sqlite-vec` + OpenAI embeddings), combined using Reciprocal Rank Fusion.

## Install

```bash
git clone https://github.com/houshuang/claude-chat-search.git
cd claude-chat-search
uv pip install -e .
```

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Keyword search works out of the box. For semantic (vector) search, set `OPENAI_API_KEY` in your environment.

## Usage

### First-time setup

```bash
claude-chat-search init
```

Creates the database at `~/.claude-chat-search/index.db`, indexes all existing conversations, and generates embeddings.

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
```

Options:
- `-n` / `--limit` — number of results (default 10)
- `-p` / `--project` — filter by project path substring
- `-b` / `--branch` — filter by git branch name substring
- `--since` — only results after date (`YYYY-MM-DD`, `3d`, `2w`, `1m`)
- `--before` — only results before date
- `--grep` — exact substring search (skips semantic/FTS5, just matches raw text)
- `--file` — search by file path mentioned in session tool calls

### Inspect a session

```bash
claude-chat-search show <session-id> --turn 5
```

Partial session ID matching is supported. Use `--turn` to highlight a specific turn.

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
- **embedder.py** — generates OpenAI `text-embedding-3-small` embeddings in batches
- **db.py** — SQLite with FTS5 for keyword search and `sqlite-vec` for vector search
- **search.py** — hybrid search (vector + keyword + grep + file) combined via Reciprocal Rank Fusion, deduplicated by session
- **daemon.py** — persistent indexer daemon: queue-based incremental indexing, message-count skip, startup full scan
- **cli.py** — Click CLI exposing `init`, `index`, `search`, `show`, and `daemon` commands

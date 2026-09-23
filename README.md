# claude-chat-search

Semantic search over your past Claude Code conversations. Indexes the JSONL conversation logs in `~/.claude/projects/` into a local SQLite database with hybrid search — full-text keyword search (FTS5) and vector similarity search (via [limbic](https://github.com/houshuang/limbic)), combined using Reciprocal Rank Fusion.

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

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Indexing, keyword, semantic and reranked search run locally with no API keys. The first run downloads the embedding model (about 460 MB). Only `--expand` calls an LLM (Gemini Flash, needs `GEMINI_API_KEY`).

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
- `--expand` — LLM query expansion via Gemini Flash (needs `GEMINI_API_KEY`) for better recall on vague/cross-vocabulary queries (~3-5s extra, typically 3-5x score improvement). Generates keyword variants, semantic rephrases, and hypothetical document excerpts to bridge vocabulary gaps.

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
- **search.py** — hybrid search (vector + keyword + grep + file) combined via Reciprocal Rank Fusion, deduplicated by session, with optional cross-encoder reranking and LLM query expansion (lex/vec/hyde variants) via limbic
- **daemon.py** — persistent indexer daemon: queue-based incremental indexing, message-count skip, startup full scan
- **cli.py** — Click CLI exposing `init`, `index`, `search`, `resume`, `show`, `subagents`, `subagent`, `recover`, `reembed`, `summarize`, `cross`, and `daemon` commands

`cross` (chat history plus a separate research-file index) and `summarize` (topic summaries) depend on the author's own tooling at fixed local paths and will not work on other machines as-is.

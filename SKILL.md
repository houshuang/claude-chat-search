---
name: Claude Chat Search
description: Search past Claude Code conversations using semantic search. Use this skill when users want to find previous conversations, recall what was discussed, search chat history, or look up past sessions by topic. Also supports cross-index search across chat history AND research files.
---

# Claude Chat Search

Search past Claude Code conversations using hybrid semantic + keyword search.

Source code: `~/tana/claude-tool/claude-chat-search/`
Upstream repo: `tanainc/devtools` (path: `claude-chat-search/`)

## When to Use This Skill

Activate this skill when the user:
- Wants to find a past Claude Code conversation by topic
- Asks "remember when we..." or "that time we discussed..."
- Needs to recall what was done in a previous session
- Wants to search their Claude chat history
- Asks about work done on a specific project or branch
- Wants to search across both chat history and research documents (use `cross`)

Also use `recover` automatically when you see "This session is being continued from a previous conversation that ran out of context" — it retrieves the recent turns from the session for context recovery.

## Commands

### Search for conversations

```bash
claude-chat-search search "QUERY" [OPTIONS]
```

- `QUERY`: Natural language description of what to find
- `--limit N` / `-n N`: Number of results (default: 10)
- `--project PATH` / `-p PATH`: Filter by project path substring
- `--branch NAME` / `-b NAME`: Filter by git branch name substring
- `--since DATE`: Only results after date (YYYY-MM-DD, 3d, 2w, 1m)
- `--before DATE`: Only results before date
- `--grep`: Exact substring search (no semantic matching)
- `--file`: Search by file path in session metadata
- `--rerank`: Re-score with cross-encoder for better relevance (slower, more accurate)

Search results include topic summaries (if generated) for quick context.

### Cross-index search (chat + research)

```bash
claude-chat-search cross "QUERY" [OPTIONS]
```

Searches both the chat history index AND the research file index (`~/src/otak/data/research_index.db`), merging results with Reciprocal Rank Fusion. Each result is labeled `[chat]` or `[research]`. If a research file was discussed in a chat session, the chat result is preferred to avoid duplication.

Supports the same filter options as `search`: `--limit`, `--project`, `--branch`, `--since`, `--before`.

### Generate topic summaries

```bash
claude-chat-search summarize [--all] [--limit 50]
```

Generates 2-3 sentence topic summaries for sessions that don't have one, using Gemini Flash (~$0.0001/session). Summaries appear in search results.

- `--all`: Summarize all unsummarized sessions (ignores --limit)
- `--limit N` / `-n N`: Max sessions to summarize (default: 50)

### Recover context after compact

```bash
claude-chat-search recover SESSION_ID [-n TURNS]
```

Outputs the most recent turns from a session in compact LLM-friendly format. Use this to recover context after compaction or context loss. Supports partial session ID matching.

- `-n TURNS`: Number of recent turns to show (default: 20)

### Show conversation details

```bash
claude-chat-search show SESSION_ID [--turn N] [--context M]
```

Supports partial session ID matching (e.g., first 8 characters).

- `--turn N` / `-t N`: Highlight a specific turn number
- `--context M` / `-C M`: Show M turns around the highlighted turn

### Re-index new conversations

```bash
claude-chat-search index
```

Run this to pick up conversations that happened since the last index. Only processes new/modified sessions.

## Tips

- Use descriptive, natural language queries — semantic search understands meaning, not just keywords
- Use `--since 3d` to narrow to recent conversations
- Use `--grep` for exact matches on error messages, file paths, or branch names
- Use `--project` to narrow results to a specific codebase
- Use `cross` when you need to find information that might be in either chat history or research documents

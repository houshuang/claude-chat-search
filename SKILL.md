---
name: Claude Chat Search
description: Search past Claude Code conversations using semantic search. Use this skill when users want to find previous conversations, recall what was discussed, search chat history, or look up past sessions by topic.
---

# Claude Chat Search

Search past Claude Code conversations using hybrid semantic + keyword search.

## When to Use This Skill

Activate this skill when the user:
- Wants to find a past Claude Code conversation by topic
- Asks "remember when we..." or "that time we discussed..."
- Needs to recall what was done in a previous session
- Wants to search their Claude chat history
- Asks about work done on a specific project or branch

## Commands

### Search for conversations

```bash
claude-chat-search search "QUERY" [--limit N] [--project PATH]
```

- `QUERY`: Natural language description of what to find (e.g., "debugging auth middleware", "react performance")
- `--limit N`: Number of results (default: 10)
- `--project PATH`: Filter by project path substring (e.g., `--project myapp`)

### Show conversation details

```bash
claude-chat-search show SESSION_ID
```

Supports partial session ID matching (e.g., first 8 characters).

### Re-index new conversations

```bash
claude-chat-search index
```

Run this to pick up conversations that happened since the last index. Only processes new/modified sessions.

## Interpreting Results

Each result shows:
- **score**: Relevance score (higher is better)
- **session**: Session ID (use with `show` command)
- **project**: The project directory
- **slug**: Human-readable session name
- **branch**: Git branch at the time
- **time**: When the conversation happened
- **User/Assistant**: Preview of the conversation turn

## Tips

- Use descriptive, natural language queries — semantic search understands meaning, not just keywords
- If results seem stale, run `claude-chat-search index` first to pick up recent sessions
- Use `--project` to narrow results to a specific codebase

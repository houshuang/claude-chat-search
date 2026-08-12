---
name: claude-chat-search
description: Search indexed user/assistant conversation turns from past Claude Code and Codex tasks with exact or semantic search, or search chat history and research files together. Use when recalling prior discussions, work on a project or branch, exact errors or identifiers, past task context, or information that may be in either chats or research. Codex internal reasoning, tool calls, and tool outputs are excluded from its index.
---

# Claude and Codex Chat Search

Search past Claude Code and Codex conversations using hybrid semantic + keyword search.

Source code: `~/tana/claude-tool/claude-chat-search/`
Upstream repo: `tanainc/devtools` (path: `claude-chat-search/`)

Also use `recover` automatically when you see "This session is being continued from a previous conversation that ran out of context" — it retrieves the recent turns from the session for context recovery.

## Search Strategy

**Do NOT run `index` before searching.** A background daemon keeps Claude
history current in real time, while a low-priority incremental launchd job scans
Codex history every three hours. Running `index` first is redundant and can
cold-start the embedding model. Run `search`/`cross` directly. Only index
manually if a search clearly misses recent material (see "Re-index" below).

**Identifier-first rule**: When the user mentions a PR number, branch name, URL,
commit hash, session slug, error message, or any specific string — ALWAYS start
with `--grep`. Semantic search is for vague topic recall only.

Examples:
- Finding PR #2925: `claude-chat-search search "2925" --grep`
- Finding a branch: `claude-chat-search search "sh/fix-self-referential" --grep`
- Finding an error: `claude-chat-search search "AI_DownloadError" --grep`

**`--project` auto-expands across checkouts**: If a repo has multiple local
checkouts (e.g. pol2, pol3, pol4), `--project pol3` automatically searches all
directories sharing the same git remote. No need to omit the filter or use
broad substrings.

**Escalation**: If `--grep` returns 0 results, try semantic search. If semantic
returns too many, add `--since` or `--branch` filters. For vague topic queries
where you need better recall, add `--expand` which uses LLM query expansion
(generates keyword variants, semantic rephrases, and hypothetical document
excerpts to bridge vocabulary gaps).

## Commands

### Search for conversations

```bash
claude-chat-search search "QUERY" [OPTIONS]
```

- `QUERY`: Natural language description of what to find
- `--limit N` / `-n N`: Number of results (default: 10)
- `--project PATH` / `-p PATH`: Filter by project path substring (auto-expands across checkouts of same repo)
- `--branch NAME` / `-b NAME`: Filter by git branch name substring
- `--since DATE`: Only results after date (YYYY-MM-DD, 3d, 2w, 1m)
- `--before DATE`: Only results before date
- `--grep`: Exact substring search (no semantic matching)
- `--file`: Search by file path in session metadata
- `--rerank`: Re-score with a cross-encoder (slower, more accurate)
- `--expand`: LLM query expansion for better recall on vague queries (~3-5s extra)
- `--source claude|codex|all`: Filter by conversation source (default: all)

Search results include topic summaries (if generated) for quick context.

### Cross-index search (chat + research)

```bash
claude-chat-search cross "QUERY" [OPTIONS]
```

Searches both the chat history index AND the research file index (`~/src/otak/data/research_index.db`), merging results with Reciprocal Rank Fusion. Each result is labeled `[chat]` or `[research]`. If a research file was discussed in a chat session, the chat result is preferred to avoid duplication.

Supports the same filter options as `search`: `--limit`, `--project`, `--branch`, `--since`, `--before`.

### Show conversation details

```bash
claude-chat-search show SESSION_ID [--turn N] [--context M] [--with-subagents]
```

Supports partial session ID matching (e.g., first 8 characters).

- `--turn N` / `-t N`: Highlight a specific turn number
- `--context M` / `-C M`: Show M turns around the highlighted turn
- `--with-subagents`: Append subagent summaries (type, first prompt) at the end

The header always shows subagent count when > 0.

### List subagents for a session

```bash
claude-chat-search subagents SESSION_ID
```

Lists all subagent conversations (background agents) for a session, showing
agent type, description, message count, and first prompt preview. Useful for
finding which subagent contains specific data (e.g. cloud log analysis, code
investigation results).

### Show a subagent conversation

```bash
claude-chat-search subagent SESSION_ID AGENT_ID [--raw]
```

Shows the full conversation of a specific subagent. Supports partial ID matching
for both session and agent IDs. Use `--raw` for untruncated output.

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

### Re-index new conversations

```bash
claude-chat-search index [--source claude|codex]
```

**You almost never need to run this.** Claude sessions are indexed continuously;
Codex sessions are scanned incrementally every three hours at nice priority 15
with low-priority I/O and two-thread numerical-library limits. Use
`--source codex` only to force an early Codex refresh. Codex indexing stores only
visible user and assistant conversation events; it skips subagents, system and
developer instructions, reasoning, tool calls, tool outputs, and token/state
events. Never run indexing as a reflex before searching.

## Tips

- Use `--grep` for exact matches on PR numbers, error messages, file paths, or branch names
- Use descriptive, natural language queries for vague topic searches
- Use `--since 3d` to narrow to recent conversations
- Use `--project` to narrow results to a specific codebase (auto-expands across checkouts)
- Use `cross` when you need to find information that might be in either chat history or research documents
- When output is long, pipe to a temp file and read with the Read tool:
  `claude-chat-search show SESSION > /tmp/session.txt` then use Read

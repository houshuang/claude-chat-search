# Claude Chat Search — Improvement Plan

## Current State (2026-02-16, post Phase 2)

- 720 sessions indexed (incl. 466 subagent sessions)
- 41,441 chunks, all embedded
- DB size: 417 MB (after VACUUM + vec table rebuild)
- Raw JSONL: 3.8 GB
- 0% crash rate (was 18% before Phase 1)

## Completed

### Phase 1: Fix crashes & quick wins ✓

- **1a. FTS5 query sanitization** — double-quote each term, strip problematic chars, fallback on error
- **1b. `--branch` filter** — SQL pre-filter on git_branch LIKE
- **1c. `--grep` mode** — exact substring search via SQL LIKE (no FTS5/embeddings)
- **1d. Session pre-filtering** — project/branch/since/before pushed to SQL, no longer post-filter

### Phase 2: Improve indexing quality ✓

- **2a. Session-level metadata** — extract `files_touched`, `tools_used`, `commands_run` as JSON fields on sessions table. Internal paths (`.claude/`, `tool-results/`) filtered out.
- **2b. `--file` search mode** — searches `files_touched` metadata field via SQL LIKE
- **2c. Merge tiny chunks** — consecutive turns under 30 tokens are merged until hitting 800-token cap
- **2d. Tool context enrichment** — each chunk gets a `[Tools: Read foo.ts, Bash: git, ...]` summary appended
- **2e. Subagent indexing** — walks `subagents/` directories, links via `parent_session_id` field

## Verification (12 tests, 2026-02-16)

### Previously crashing queries — all pass now
| # | Query | Before | After |
|---|-------|--------|-------|
| 1 | `"skill-doc-type"` | CRASH: no such column: doc | ✅ Finds 3 relevant sessions |
| 2 | `"sh/demo"` | CRASH: fts5 syntax error near "/" | ✅ Finds demo branch session |
| 3 | `"claude-chat-search"` | CRASH: no such column: chat | ✅ Finds 3 relevant sessions |

### Recall improvement
| # | Query | Before | After |
|---|-------|--------|-------|
| 4 | `"sh/demo-skills-interview" --grep` | N/A (no grep mode) | ✅ Finds exact branch mentions |
| 4b | `"skill fixes" --branch demo-skills` | N/A (no branch filter) | ✅ Finds demo fixes session with code details |
| 9 | `"demo branch that combined skill-doc-type..."` | Irrelevant results | ✅ #1 result describes exact demo branch fixes |

### New features
| # | Query | Result |
|---|-------|--------|
| 5 | `"LoroSkill.ts" --grep` | ✅ 3 sessions with exact file mentions in chunk text |
| 6 | `"LoroSkill.ts" --file` | ✅ 5 sessions across pol2/pol3/pol5 that touched the file |
| 7 | `"Edit LoroSkill" --grep` | ✅ Finds tool-enriched chunks with Edit mentions |
| 10 | `"ProposalAwareEntryLookup" --file` | ✅ 3 sessions across polaris/pol3 |
| 11 | `"no such column" --grep` | ✅ Finds error messages in chunk text |
| 12 | `"PR analysis review insights"` | ✅ #1 result is the PR analysis summary session |

### Remaining gaps
- `--branch demo` + semantic query returns no results when no sessions on that branch match semantically. **Workaround:** use `--grep` for branch name searches, or `--branch` with broader queries.
- `--file` mode shows session-level results without chunk content (shows "[Session touched file matching: X]"). Could be improved to show the most relevant chunk from the session.
- Subagent sessions (466 indexed) are relatively sparse — many are short tool-only runs with little searchable text. Still useful for completeness.

## Size Impact (actual)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Sessions | 691 | 720 | +4% (subagents added) |
| Chunks | 274,919 | 41,441 | **-85%** |
| Near-empty (<30 tokens) | 67,938 (25%) | 346 (0.8%) | -99.5% |
| DB size | 2.1 GB | 417 MB | **-80%** |
| Sessions with file metadata | 0 | 451 | New |

## Remaining

### Phase 3: Better output & UX

**3a. Improve search result format** (cli.py)
- Current: 7-8 lines per result with truncated user/assistant text
- Better: 3-4 lines with the actual matching snippet highlighted
- Include session-level context: first user message, total duration

**3b. Add `sessions` command** (cli.py)
- List all indexed sessions with metadata (project, branch, date, message count)
- Support filtering: `--project`, `--branch`, `--since`
- Helps users discover what's available without searching

**3c. Improve `--file` results**
- Currently shows "[Session touched file matching: X]" without chunk content
- Better: find the chunk that mentions the file and show that instead

### Phase 4: Size optimization

**4a. Consider smaller embeddings** (embedder.py, db.py)
- `text-embedding-3-small` supports dimension reduction via `dimensions` param
- 512 dims instead of 1536 would cut DB from 417 MB to ~200 MB
- Quality impact is modest for short text chunks
- Requires re-embedding everything (one-time cost)

**4b. Re-evaluate embedding necessity**
- With good FTS5 + structured metadata search + grep mode, do we still need embeddings?
- Could offer an "embedding-free" mode using only FTS5 + SQL LIKE
- Saves the OpenAI API dependency and majority of DB size

## Decision Log

- Do NOT index full tool_result content as chunks — too much noise, too expensive
- Do NOT embed metadata fields — SQL LIKE is sufficient for exact matches
- DO extract metadata for structured search — lightweight and high-value
- Filter out internal paths (`.claude/`, `tool-results/`) from files_touched
- Merge tiny turns rather than dropping them — preserves context
- Index subagents as separate sessions with parent_session_id link
- Keep embeddings for now but consider 512-dim reduction later
- After VACUUM + vec rebuild: sqlite-vec doesn't release space on delete, must drop+recreate table

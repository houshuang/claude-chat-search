# Claude Chat Search — Improvement Plan

## Current State (2026-03-09, post Phase 3)

- 792 sessions indexed (main sessions only — subagents dropped)
- 12,775 chunks, 99.4% embedded
- DB size: 315 MB
- Search latency: median 1.7s (was 5-8s)
- Search logging enabled for ongoing quality analysis

## Previous State (2026-02-16, post Phase 2)

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

### Phase 3: Noise reduction & reliability ✓ (2026-03-09)

Analysis method: used `claude-chat-search` itself to find all sessions where the tool was invoked since the last update (Feb 16), then analyzed usage patterns, failures, and content quality of the index.

#### Findings from usage analysis

**Daemon thrashing bug**: PostToolUse hook fires on every tool call, queueing the same session 50+ times per conversation. The daemon was re-indexing the same session every 6 seconds, never getting to embeddings. Result: 270K of 322K chunks were unembedded (84%).

**Massive noise in index**:
| Category | Chunks | % | Avg tokens |
|----------|--------|---|------------|
| Pure content | 191,060 | 59% | 329 |
| Tool + content | 74,466 | 23% | 83 |
| Tool-only noise | 57,141 | 18% | 33 |
| Compact prompt templates | 4,768 | 1.5% | 1,707 |

**Subagent dominance**: 664 subagent sessions (59% of all sessions) produced 112K chunks and consumed 91% of all embeddings. Subagent research results flow back into main sessions via tool_result blocks, so the main session already contains enough signal for search.

**Context recovery use case**: Very common pattern where users say "use chat search to find the previous context (lost to context)" after compaction. Required 3+ sequential tool calls (search → show → read), each taking 5-8s.

#### Changes made

- **3a. Drop subagent indexing** — removed subagent directory walking from `iter_jsonl_files()` and reject `/subagents/` paths in `file_info_from_path()`. Main sessions capture subagent results already.
- **3b. Chunk noise filtering** — skip turns with empty user + <100 char assistant content; skip compact prompt templates; raise `MIN_CHUNK_TOKENS` from 30 → 80.
- **3c. Daemon cooldown** — 60-second per-session cooldown prevents re-indexing same session repeatedly. Embedding pipeline decoupled to separate 30-second timer.
- **3d. `recover` command** — `claude-chat-search recover SESSION_ID [-n TURNS]` outputs recent turns in compact LLM-friendly format for context recovery.
- **3e. Search logging** — every search appends JSON line to `~/.claude-chat-search/search.log` with query, mode, latency, and per-result details (session, score, project, turn).
- **3f. SKILL.md updated** — documented all CLI options (`--since`, `--before`, `--branch`, `--grep`, `--file`), `recover` command, and auto-recovery instructions.

#### Results

| Metric | Before (Phase 2) | Intermediate (pre-Phase 3) | After (Phase 3) |
|--------|-------------------|---------------------------|-----------------|
| Sessions | 720 | 1,129 | 792 |
| Chunks | 41,441 | 322,638 | 12,775 |
| Embedded | 41,441 (100%) | 52,139 (16%) | 12,703 (99.4%) |
| DB size | 417 MB | 1.9 GB | 315 MB |
| Search latency | ~3s | 5-8s | median 1.7s |
| Daemon re-indexes/min | N/A | ~10 (thrashing) | ~1 (cooldown) |

### Phase 4: Limbic integration ✓ (2026-03-26)

Migrated chat-search to use [limbic](https://github.com/houshuang/limbic) for embedding, vector search, and FTS5 sanitization. Also contributed improvements from chat-search back into limbic (FTS5 triggers, grep search, dedup utility).

#### Changes made

- **4a. Embedding via limbic** — replaced direct `sentence-transformers` usage with `limbic.amygdala.EmbeddingModel`. Model upgraded from `all-MiniLM-L6-v2` (monolingual) to `paraphrase-multilingual-MiniLM-L12-v2` (multilingual, +5% accuracy). All ~19K chunks re-embedded.
- **4b. Vector search via limbic** — replaced custom numpy implementation with `limbic.amygdala.VectorIndex`. Same algorithm, maintained module-level cache with count-based invalidation.
- **4c. FTS5 sanitization via limbic** — replaced custom `_sanitize_fts_query()` with `limbic.amygdala.search.FTS5Index._sanitize_query()`. Fixes bug where FTS5 reserved words (AND/OR/NOT/NEAR) were interpreted as operators.
- **4d. Cross-encoder reranking** — added `--rerank` flag for optional cross-encoder re-scoring of search results via `limbic.amygdala.rerank()`. Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, ~50ms for 10 results).
- **4e. Dependencies** — replaced `sentence-transformers` dependency with `limbic` (installed from GitHub). Removed ~400 lines of duplicated embedding/search code.

#### Improvements contributed to limbic

- FTS5 query sanitization bug fix (quote tokens to prevent reserved word interpretation)
- FTS5 auto-sync triggers replacing 30-line manual `_sync_fts_for()` method
- `Index.grep()` method for exact substring search
- `dedup_by()` utility for group-by deduplication of search results
- 15 new tests

### Phase 5: Subagent access & repo-aware grouping ✓ (2026-03-26)

Motivated by a real failure: finding the conversation behind PR #2925 took 6+ failed semantic searches before discovering `--grep "2925"` worked instantly. The detailed failure data lived in a subagent conversation with no native access path.

#### Changes made

- **5a. Subagent metadata indexing** — lightweight `subagents` table stores agent type, first prompt, message count, and file path. No chunks or embeddings (avoids the Phase 3 noise problem). Full conversation content read on-demand from JSONL. 4,772 records indexed.
- **5b. `subagents` command** — lists all background agent conversations for a session with type, message count, size, and first prompt preview.
- **5c. `subagent` command** — displays the full conversation of a specific subagent with partial ID matching. `--raw` flag for untruncated output.
- **5d. Enhanced `show`** — header shows subagent count; `--with-subagents` flag appends subagent summaries.
- **5e. Git remote detection** — detects `git remote get-url origin` for each project path, normalizes to `owner/repo`, caches permanently in `~/.claude-chat-search/git-remotes.json`. 1,078 sessions backfilled.
- **5f. Transparent `--project` expansion** — when `--project pol3` matches sessions with a git_remote, the filter auto-expands to include ALL project paths sharing that remote. Zero configuration needed.
- **5g. Skill file search strategy** — added "identifier-first rule": when searching for PR numbers, branch names, error messages etc., always use `--grep` first. Semantic search is for vague topic recall only.

#### Results

| Metric | Before (Phase 4) | After (Phase 5) |
|--------|-------------------|-----------------|
| Sessions | ~1,000 | 1,317 |
| Subagent metadata | 0 | 4,772 records |
| Git remotes cached | 0 | 44 entries |
| `--project pol3` coverage | 1 folder | 9 folders (auto-expanded) |

#### Key decision: metadata-only indexing for subagents

Phase 3 dropped subagent indexing because 112K chunks consumed 91% of embeddings. Phase 5 restores subagent *discoverability* via a lightweight metadata table without re-introducing the noise problem. The `subagent` command reads full conversations on-demand from JSONL files, so no re-parsing or embedding is needed.

## Remaining

### Phase 6: Search quality analysis

**6a. Analyze search logs** — now that we're logging queries + results, analyze patterns after 1-2 weeks of usage:
- What queries return 0 results? (missing coverage)
- Do users retry with different wording? (search quality signal)
- How often are `--grep` / `--since` / `--project` filters used?
- Is semantic search adding value over FTS5 alone?

**6b. Session-level topic extraction** — instead of (or in addition to) embedding every chunk, extract keywords/topics per session at index time. Could enable:
- Faster search (no embedding API call for query)
- Better session-level relevance ranking
- Topic browsing / clustering

**6c. Evaluate search result format**
- Current format truncates user to 300 chars, assistant to 500 chars, plus 3-4 lines metadata
- `recover` command already solves the compact-output need for context recovery
- A `--json` flag was considered but may not be needed — current format is LLM-parseable
- Decision: monitor search logs for patterns where LLM always calls `show` after `search` (meaning preview wasn't sufficient). If common, add more content per result or a compact mode. Otherwise leave as-is.

**6d. Consider whitening for chat embeddings**
- limbic supports Soft-ZCA whitening (+32% NN-separation on domain corpora)
- Chat search corpus is domain-diverse (all projects), so whitening might hurt
- Worth testing: fit whitening on per-project subsets vs. full corpus vs. no whitening

## Decision Log

- Do NOT index full tool_result content as chunks — too much noise, too expensive
- Do NOT embed metadata fields — SQL LIKE is sufficient for exact matches
- DO extract metadata for structured search — lightweight and high-value
- Filter out internal paths (`.claude/`, `tool-results/`) from files_touched
- Merge tiny turns rather than dropping them — preserves context
- ~~Index subagents as separate sessions with parent_session_id link~~ → **Dropped subagent indexing (Phase 3)**: subagent results flow into main sessions via tool_result. Indexing them separately added 112K noisy chunks and consumed 91% of embeddings.
- Keep embeddings for now but monitor via search logs whether FTS5 alone is sufficient
- After VACUUM + vec rebuild: sqlite-vec doesn't release space on delete, must drop+recreate table
- Foreign key constraint: when doing `--all --force` re-index, must disable FK or delete chunks before sessions
- ~~Drop subagent indexing entirely~~ → **Re-introduce as metadata-only (Phase 5)**: subagent chunks were noise (Phase 3), but subagent *metadata* (type, first prompt, message count) is cheap and enables drill-down. Full conversations read on-demand from JSONL.
- Git remote cache is permanent (JSON file) because remotes don't change per directory. No expiry needed.
- `--project` expansion via git remote is transparent — no separate `--repo` flag needed. Users just use `--project` and it works across multiple checkouts automatically.

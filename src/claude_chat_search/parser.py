import json
import os
import re
import subprocess
from pathlib import Path


PROJECTS_DIR = Path.home() / ".claude" / "projects"
GIT_REMOTE_CACHE_PATH = Path.home() / ".claude-chat-search" / "git-remotes.json"

_git_remote_cache: dict[str, str | None] | None = None


def decode_project_path(dirname: str) -> str:
    return dirname.replace("-", "/", 1).replace("-", "/")


def iter_jsonl_files() -> list[dict]:
    """Walk all project directories and yield JSONL file metadata.

    Includes both main sessions and subagent sessions.
    Subagent sessions are linked to their parent via parent_session_id.
    """
    if not PROJECTS_DIR.exists():
        return []

    results = []
    for project_dir in sorted(PROJECTS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        project_path = decode_project_path(project_dir.name)

        for jsonl_file in sorted(project_dir.glob("*.jsonl")):
            mtime = os.path.getmtime(jsonl_file)
            results.append({
                "path": jsonl_file,
                "project_path": project_path,
                "session_id": jsonl_file.stem,
                "mtime": mtime,
            })

    return results


def parse_jsonl_file(filepath: Path) -> dict:
    """Parse a JSONL file into structured session data.

    Returns a dict with session metadata and a list of messages.
    """
    messages = []
    session_id = None
    slug = None
    git_branch = None
    cwd = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            if session_id is None:
                session_id = obj.get("sessionId")
            if slug is None:
                slug = obj.get("slug")
            if git_branch is None and obj.get("gitBranch"):
                git_branch = obj["gitBranch"]
            if cwd is None:
                cwd = obj.get("cwd")

            messages.append(obj)

    if not messages:
        return {"messages": [], "session_id": session_id, "slug": slug,
                "git_branch": git_branch}

    timestamps = [m.get("timestamp", "") for m in messages if m.get("timestamp")]

    return {
        "session_id": session_id or filepath.stem,
        "slug": slug,
        "git_branch": git_branch,
        "cwd": cwd,
        "messages": messages,
        "first_message_at": min(timestamps) if timestamps else None,
        "last_message_at": max(timestamps) if timestamps else None,
        "message_count": len(messages),
    }


def extract_text_content(message: dict) -> str:
    """Extract text content from a message, skipping thinking/tool_use blocks."""
    content = message.get("message", {}).get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def extract_tool_summary(message: dict) -> str:
    """Extract a short tool usage summary from an assistant message.

    Returns a string like "[Tools: Read foo.ts, Edit bar.ts, Bash: git status]"
    or "" if no tools were used.
    """
    content = message.get("message", {}).get("content")
    if not isinstance(content, list):
        return ""

    tool_parts = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name", "")
        inp = block.get("input", {})

        if name in ("Read", "Edit", "Write", "MultiEdit"):
            path = inp.get("file_path", "")
            if path:
                # Keep just filename, not full path
                short = path.rsplit("/", 1)[-1]
                tool_parts.append(f"{name} {short}")
            else:
                tool_parts.append(name)
        elif name == "Bash":
            cmd = inp.get("command", "")
            # First word of command (e.g. "git", "pnpm", "npm")
            first_word = cmd.split()[0] if cmd.split() else ""
            tool_parts.append(f"Bash: {first_word}" if first_word else "Bash")
        elif name == "Grep":
            pat = inp.get("pattern", "")[:30]
            tool_parts.append(f"Grep {pat}")
        elif name == "Glob":
            pat = inp.get("pattern", "")[:30]
            tool_parts.append(f"Glob {pat}")
        elif name == "Task":
            desc = inp.get("description", "")[:30]
            tool_parts.append(f"Task: {desc}" if desc else "Task")
        elif name:
            tool_parts.append(name)

    if not tool_parts:
        return ""

    # Deduplicate while preserving order, truncate to reasonable length
    seen = set()
    unique = []
    for p in tool_parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    summary = "[Tools: " + ", ".join(unique) + "]"
    if len(summary) > 200:
        summary = summary[:197] + "...]"
    return summary


def _is_internal_path(path: str) -> bool:
    """Filter out Claude internal paths that aren't real project files."""
    return "/.claude/" in path or "/tool-results/" in path


def extract_session_metadata(messages: list[dict]) -> dict:
    """Extract structured metadata from tool calls across a session.

    Returns dict with:
      - files_touched: sorted unique file paths from Read/Edit/Write/Glob
      - tools_used: sorted unique tool names
      - commands_run: list of truncated Bash commands
    """
    files = set()
    tools = set()
    commands = []

    for msg in messages:
        if msg.get("type") != "assistant":
            continue
        content = msg.get("message", {}).get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            inp = block.get("input", {})
            if name:
                tools.add(name)

            if name in ("Read", "Edit", "Write", "MultiEdit"):
                path = inp.get("file_path", "")
                if path and not _is_internal_path(path):
                    files.add(path)
            elif name == "Bash":
                cmd = inp.get("command", "")
                if cmd:
                    commands.append(cmd[:150])
            elif name in ("Grep", "Glob"):
                path = inp.get("path", "")
                if path and not _is_internal_path(path):
                    files.add(path)

    return {
        "files_touched": json.dumps(sorted(files)),
        "tools_used": json.dumps(sorted(tools)),
        "commands_run": json.dumps(commands[:50]),  # cap at 50 commands
    }


def file_info_from_path(transcript_path: str | Path) -> dict | None:
    """Derive file metadata from a transcript path, matching iter_jsonl_files() format.

    Handles both main sessions and subagent sessions:
      ~/.claude/projects/{project}/{session_id}.jsonl
      ~/.claude/projects/{project}/{parent_session_id}/subagents/{session_id}.jsonl
    """
    path = Path(transcript_path)
    if not path.exists() or path.suffix != ".jsonl":
        return None

    # Skip subagent transcripts — their results flow into the main session
    if "/subagents/" in str(path):
        return None

    try:
        rel = path.relative_to(PROJECTS_DIR)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) < 2:
        return None

    project_path = decode_project_path(parts[0])
    result = {
        "path": path,
        "project_path": project_path,
        "session_id": path.stem,
        "mtime": os.path.getmtime(path),
    }

    # Subagent: {project}/{parent_session_id}/subagents/{session_id}.jsonl
    if len(parts) >= 4 and parts[2] == "subagents":
        result["parent_session_id"] = parts[1]

    return result


def _normalize_git_remote(url: str) -> str | None:
    """Extract normalized owner/repo from a git remote URL.

    Handles HTTPS (https://github.com/owner/repo.git) and
    SSH (git@github.com:owner/repo.git) formats.
    """
    url = url.strip()
    # SSH format: git@github.com:owner/repo.git
    m = re.match(r"git@[^:]+:(.+?)(?:\.git)?$", url)
    if m:
        return m.group(1).lower()
    # HTTPS format: https://github.com/owner/repo.git
    m = re.match(r"https?://[^/]+/(.+?)(?:\.git)?$", url)
    if m:
        return m.group(1).lower()
    return None


def _load_git_remote_cache() -> dict[str, str | None]:
    global _git_remote_cache
    if _git_remote_cache is not None:
        return _git_remote_cache
    try:
        with open(GIT_REMOTE_CACHE_PATH) as f:
            _git_remote_cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _git_remote_cache = {}
    return _git_remote_cache


def _save_git_remote_cache() -> None:
    if _git_remote_cache is None:
        return
    GIT_REMOTE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GIT_REMOTE_CACHE_PATH, "w") as f:
        json.dump(_git_remote_cache, f, indent=2)


def detect_git_remote(project_path: str) -> str | None:
    """Detect the normalized git remote (owner/repo) for a project path.

    Results are cached permanently in ~/.claude-chat-search/git-remotes.json
    since git remotes don't change for a given directory.
    """
    cache = _load_git_remote_cache()
    if project_path in cache:
        return cache[project_path]

    remote = None
    try:
        result = subprocess.run(
            ["git", "-C", project_path, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            remote = _normalize_git_remote(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    cache[project_path] = remote
    _save_git_remote_cache()
    return remote


def find_project_dir(project_path: str) -> Path | None:
    """Find the encoded project directory for a given project_path.

    Iterates ~/.claude/projects/ and decodes each directory name to match.
    """
    if not PROJECTS_DIR.exists():
        return None
    for d in PROJECTS_DIR.iterdir():
        if d.is_dir() and decode_project_path(d.name) == project_path:
            return d
    return None


def iter_subagent_files(parent_session_id: str, project_dir: Path) -> list[dict]:
    """List all subagent JSONL files for a given parent session."""
    subagent_dir = project_dir / parent_session_id / "subagents"
    if not subagent_dir.exists():
        return []

    results = []
    for jsonl_file in sorted(subagent_dir.glob("agent-*.jsonl")):
        # Strip "agent-" prefix and ".jsonl" suffix to get agent_id
        agent_id = jsonl_file.stem[6:]  # len("agent-") == 6
        meta_path = jsonl_file.with_name(f"agent-{agent_id}.meta.json")
        results.append({
            "agent_id": agent_id,
            "parent_session_id": parent_session_id,
            "jsonl_path": str(jsonl_file),
            "meta_path": str(meta_path) if meta_path.exists() else None,
            "mtime": os.path.getmtime(jsonl_file),
            "file_size": os.path.getsize(jsonl_file),
        })
    return results


def parse_subagent_metadata(jsonl_path: str, meta_path: str | None) -> dict:
    """Parse lightweight metadata from a subagent without reading the full file.

    Reads meta.json for agent type/description, and streams the JSONL to get
    message count, first user prompt, and timestamps.
    """
    agent_type = None
    description = None

    if meta_path:
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                agent_type = meta.get("agentType")
                description = meta.get("description")
        except (json.JSONDecodeError, OSError):
            pass

    message_count = 0
    first_prompt = None
    first_ts = None
    last_ts = None

    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = obj.get("type")
                if msg_type not in ("user", "assistant"):
                    continue

                message_count += 1
                ts = obj.get("timestamp")
                if ts:
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts

                if first_prompt is None and msg_type == "user":
                    first_prompt = extract_text_content(obj)[:500]
    except OSError:
        pass

    return {
        "agent_type": agent_type,
        "description": description,
        "message_count": message_count,
        "first_prompt": first_prompt,
        "first_message_at": first_ts,
        "last_message_at": last_ts,
    }


def parse_subagent_conversation(jsonl_path: str) -> list[dict]:
    """Parse a subagent JSONL file into a list of turns for display.

    Returns a list of dicts with: turn_number, role, text, timestamp, tools.
    """
    messages = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg_type = obj.get("type")
                if msg_type not in ("user", "assistant"):
                    continue
                messages.append(obj)
    except OSError:
        return []

    # Group assistant messages by requestId (same dedup as main sessions)
    messages = group_assistant_messages(messages)

    turns = []
    turn_number = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("type") == "user":
            user_text = extract_text_content(msg)
            user_ts = msg.get("timestamp")
            # Look for following assistant message
            asst_text = ""
            asst_tools = ""
            if i + 1 < len(messages) and messages[i + 1].get("type") == "assistant":
                asst = messages[i + 1]
                asst_text = extract_text_content(asst)
                asst_tools = extract_tool_summary(asst)
                i += 1
            turns.append({
                "turn_number": turn_number,
                "user_content": user_text,
                "assistant_content": asst_text,
                "tools": asst_tools,
                "timestamp": user_ts,
            })
            turn_number += 1
        elif msg.get("type") == "assistant":
            # Assistant without preceding user (tool result flow)
            turns.append({
                "turn_number": turn_number,
                "user_content": "",
                "assistant_content": extract_text_content(msg),
                "tools": extract_tool_summary(msg),
                "timestamp": msg.get("timestamp"),
            })
            turn_number += 1
        i += 1

    return turns


def group_assistant_messages(messages: list[dict]) -> list[dict]:
    """Deduplicate assistant messages that share the same requestId.

    Multiple JSONL lines can represent the same assistant turn (streamed).
    Keep the one with the most text content blocks.
    """
    grouped = []
    assistant_by_request = {}

    for msg in messages:
        if msg.get("type") == "user":
            grouped.append(msg)
        elif msg.get("type") == "assistant":
            req_id = msg.get("requestId")
            if not req_id:
                grouped.append(msg)
                continue

            text = extract_text_content(msg)
            existing = assistant_by_request.get(req_id)
            if existing is None or len(text) > len(extract_text_content(existing)):
                assistant_by_request[req_id] = msg

    # Collect deduplicated assistant messages and sort everything by timestamp
    for msg in assistant_by_request.values():
        grouped.append(msg)

    grouped.sort(key=lambda m: m.get("timestamp", ""))
    return grouped

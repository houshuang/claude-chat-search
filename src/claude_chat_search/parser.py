import json
import os
from pathlib import Path


PROJECTS_DIR = Path.home() / ".claude" / "projects"


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

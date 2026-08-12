"""Parse Codex CLI/Desktop rollout files into the chat-search message model.

Codex's local JSONL format is not a documented public API.  Keep every format
assumption in this module and cover it with fixtures.  Only visible user and
agent messages are normalized; developer prompts, reasoning, tool calls, tool
outputs, token counts, and world state are deliberately ignored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


CODEX_ROOT = Path.home() / ".codex"
CODEX_SESSION_DIRS = (
    CODEX_ROOT / "sessions",
    CODEX_ROOT / "archived_sessions",
)

_UUID_AT_END = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.jsonl$"
)


def _native_id_from_path(path: Path) -> str:
    match = _UUID_AT_END.search(path.name)
    return match.group(1) if match else path.stem


def _read_header(path: Path) -> dict:
    """Read enough of a rollout to identify it without parsing the full file."""
    fallback_id = None
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle):
                if line_number >= 100:
                    break
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if fallback_id is None and isinstance(item.get("id"), str):
                    fallback_id = item["id"]
                if item.get("type") == "session_meta":
                    payload = item.get("payload") or {}
                    return {
                        "native_session_id": (
                            payload.get("session_id")
                            or payload.get("id")
                            or fallback_id
                            or _native_id_from_path(path)
                        ),
                        "cwd": payload.get("cwd"),
                        "git_branch": (payload.get("git") or {}).get("branch"),
                        "thread_source": payload.get("thread_source"),
                        "session_started_at": payload.get("timestamp") or item.get("timestamp"),
                    }
    except OSError:
        pass
    return {
        "native_session_id": fallback_id or _native_id_from_path(path),
        "cwd": None,
        "git_branch": None,
        "thread_source": None,
        "session_started_at": None,
    }


def _thread_kind(thread_source) -> str:
    if isinstance(thread_source, dict) and "subagent" in thread_source:
        return "subagent"
    if thread_source == "subagent":
        return "subagent"
    return "user"


def _candidate_rank(info: dict) -> tuple[int, float, int]:
    """Prefer the most complete copy when active and archived paths overlap."""
    is_active = "/archived_sessions/" not in str(info["path"])
    return (info["size"], info["mtime"], int(is_active))


def iter_codex_jsonl_files(
    roots: tuple[Path, ...] | list[Path] | None = None,
    *,
    include_subagents: bool = False,
) -> list[dict]:
    """Discover Codex rollout files, deduplicated by native session ID."""
    from .parser import is_excluded_project, load_excluded_projects

    roots = tuple(roots or CODEX_SESSION_DIRS)
    excluded = load_excluded_projects()
    by_session: dict[str, dict] = {}

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            try:
                stat = path.stat()
            except OSError:
                continue
            header = _read_header(path)
            kind = _thread_kind(header.get("thread_source"))
            if kind == "subagent" and not include_subagents:
                continue
            project_path = header.get("cwd") or ""
            if project_path and is_excluded_project(project_path, excluded):
                continue
            native_id = header["native_session_id"]
            info = {
                "path": path,
                "project_path": project_path,
                "session_id": f"codex:{native_id}",
                "native_session_id": native_id,
                "source": "codex",
                "thread_kind": kind,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "header": header,
            }
            existing = by_session.get(native_id)
            if existing is None or _candidate_rank(info) > _candidate_rank(existing):
                by_session[native_id] = info

    return sorted(by_session.values(), key=lambda item: str(item["path"]))


def file_info_from_codex_path(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists() or path.suffix != ".jsonl":
        return None
    discovered = iter_codex_jsonl_files([path.parent])
    native_id = _native_id_from_path(path)
    for info in discovered:
        if info["native_session_id"] == native_id or info["path"] == path:
            return info
    return None


def _legacy_content_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {
            "text", "input_text", "output_text"
        }:
            parts.append(block.get("text", ""))
    return "\n".join(part for part in parts if part)


def _normalized_message(role: str, text: str, timestamp: str | None) -> dict:
    """Return the small Claude-shaped message envelope consumed by chunker.py."""
    return {
        "type": role,
        "timestamp": timestamp,
        "message": {"content": text},
    }


def parse_codex_jsonl_file(filepath: Path, file_info: dict | None = None) -> dict:
    """Parse only visible human/agent conversation from a Codex rollout."""
    file_info = file_info or {
        "path": filepath,
        "session_id": f"codex:{_native_id_from_path(filepath)}",
        "native_session_id": _native_id_from_path(filepath),
        "source": "codex",
        "thread_kind": "user",
        "project_path": "",
        "header": _read_header(filepath),
    }
    header = dict(file_info.get("header") or {})
    messages = []

    try:
        with filepath.open() as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                item_type = item.get("type")
                if item_type == "session_meta":
                    payload = item.get("payload") or {}
                    header.setdefault("cwd", payload.get("cwd"))
                    header.setdefault("git_branch", (payload.get("git") or {}).get("branch"))
                    header.setdefault("session_started_at", payload.get("timestamp"))
                    continue

                # Current Codex format: event_msg contains exactly what was visible
                # in the conversation UI.  response_item and all tool/state records
                # are intentionally excluded.
                if item_type == "event_msg":
                    payload = item.get("payload") or {}
                    event_type = payload.get("type")
                    if event_type == "user_message":
                        role = "user"
                    elif event_type == "agent_message":
                        role = "assistant"
                    else:
                        continue
                    text = payload.get("message")
                    if isinstance(text, str) and text.strip():
                        messages.append(_normalized_message(role, text.strip(), item.get("timestamp")))
                    continue

                # Very early Codex rollouts stored top-level message records.
                if item_type == "message" and item.get("role") in {"user", "assistant"}:
                    text = _legacy_content_text(item.get("content"))
                    if text.strip():
                        messages.append(
                            _normalized_message(item["role"], text.strip(), item.get("timestamp"))
                        )
    except OSError:
        messages = []

    timestamps = [m["timestamp"] for m in messages if m.get("timestamp")]
    first_at = min(timestamps) if timestamps else header.get("session_started_at")
    last_at = max(timestamps) if timestamps else header.get("session_started_at")

    return {
        "session_id": file_info["session_id"],
        "native_session_id": file_info["native_session_id"],
        "source": "codex",
        "thread_kind": file_info.get("thread_kind", "user"),
        "slug": None,
        "git_branch": header.get("git_branch"),
        "cwd": header.get("cwd") or file_info.get("project_path") or "",
        "messages": messages,
        "first_message_at": first_at,
        "last_message_at": last_at,
        "message_count": len(messages),
        "files_touched": "[]",
        "tools_used": "[]",
        "commands_run": "[]",
    }

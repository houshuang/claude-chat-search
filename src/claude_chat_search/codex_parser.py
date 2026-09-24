"""Parse Codex CLI/Desktop rollout files into the chat-search message model.

Codex's local JSONL format is not a documented public API.  Keep every format
assumption in this module and cover it with fixtures.  Only visible user and
agent messages are normalized; developer prompts, injected context blocks,
reasoning, tool calls, tool outputs, token counts, and world state are
deliberately ignored.

Three generations of the format are supported:

* until early September 2026: ``event_msg`` records of type ``user_message`` /
  ``agent_message`` carried exactly the visible conversation (such files also
  contain ``response_item`` messages, which are then ignored);
* since then: only ``response_item`` records of type ``message`` with role
  ``user`` (``input_text`` blocks, mixed with injected context blocks) or
  ``assistant`` (``output_text`` blocks);
* the earliest rollouts: top-level ``message`` records.
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


# Context Codex injects into user-role messages.  Tags with "_" or "-" never
# occur in HTML a user might paste, so any such fully wrapped block is treated
# as injected; the others are listed explicitly.
_INJECTED_TAGS = {"skill", "INSTRUCTIONS"}
_WRAPPED_BLOCK = re.compile(r"<([A-Za-z][\w\-]*)(?:\s[^>]*)?>.*?</\1>\s*", re.DOTALL)
_LONE_TAG = re.compile(r"</?[A-Za-z][\w\-]*(?:\s[^>\n]*)?>")
_IDE_REQUEST_MARKER = "## My request for Codex:"


def _is_injected_tag(tag: str) -> bool:
    return tag in _INJECTED_TAGS or "_" in tag or "-" in tag


def _visible_user_text(text: str) -> str:
    """Strip Codex-injected context from one user input_text block."""
    text = text.strip()
    if text.startswith("# AGENTS.md instructions"):
        return ""
    if text.startswith("# Context from my IDE setup") and _IDE_REQUEST_MARKER in text:
        text = text.split(_IDE_REQUEST_MARKER, 1)[1].strip()
    while True:
        match = _WRAPPED_BLOCK.match(text)
        if not match or not _is_injected_tag(match.group(1)):
            break
        text = text[match.end():].lstrip()
    # Image attachments arrive as bare "<image ...>" / "</image>" marker blocks.
    if _LONE_TAG.fullmatch(text):
        return ""
    return text


def _response_item_text(payload: dict) -> str:
    role = payload.get("role")
    content = payload.get("content")
    if isinstance(content, str):
        content = [{"type": "input_text" if role == "user" else "output_text", "text": content}]
    if not isinstance(content, list):
        return ""
    wanted = "input_text" if role == "user" else "output_text"
    parts = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != wanted:
            continue
        text = block.get("text") or ""
        text = _visible_user_text(text) if role == "user" else text.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


_SETUP_EVENTS = {"thread_settings_applied", "user_message"}


def _is_agent_activity(item: dict) -> bool:
    item_type = item.get("type")
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    if item_type in {"turn_context", "world_state"}:
        return False
    if item_type == "event_msg":
        return payload.get("type") not in _SETUP_EVENTS
    if item_type == "response_item" and payload.get("type") == "message":
        return payload.get("role") == "assistant"
    if item_type == "message":
        return item.get("role") == "assistant"
    return item_type is not None


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
    event_messages = []
    response_messages = []
    top_level_messages = []
    # Records showing the agent took a turn.  A rollout holding only its
    # session header and injected user/developer context (opened, never used)
    # legitimately has no conversation; one with agent activity but no
    # messages means the format has drifted.
    agent_records = 0

    try:
        with filepath.open() as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type == "session_meta":
                    payload = item.get("payload") or {}
                    header.setdefault("cwd", payload.get("cwd"))
                    header.setdefault("git_branch", (payload.get("git") or {}).get("branch"))
                    header.setdefault("session_started_at", payload.get("timestamp"))
                    continue
                if _is_agent_activity(item):
                    agent_records += 1

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
                        event_messages.append(
                            _normalized_message(role, text.strip(), item.get("timestamp"))
                        )
                    continue

                # Only role user/assistant "message" items.  Developer messages,
                # reasoning, tool calls/outputs and inter-agent "agent_message"
                # items are other response_item payload types or roles.
                if item_type == "response_item":
                    payload = item.get("payload") or {}
                    if payload.get("type") == "message" and payload.get("role") in {"user", "assistant"}:
                        text = _response_item_text(payload)
                        if text:
                            response_messages.append(
                                _normalized_message(payload["role"], text, item.get("timestamp"))
                            )
                    continue

                if item_type == "message" and item.get("role") in {"user", "assistant"}:
                    text = _legacy_content_text(item.get("content"))
                    if text.strip():
                        top_level_messages.append(
                            _normalized_message(item["role"], text.strip(), item.get("timestamp"))
                        )
    except OSError:
        event_messages, response_messages, top_level_messages = [], [], []

    # Rollouts that still write event_msg conversation also write the same
    # turns as response_item messages; the event stream is the visible one.
    messages = event_messages or response_messages or top_level_messages

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
        "agent_record_count": agent_records,
        "files_touched": "[]",
        "tools_used": "[]",
        "commands_run": "[]",
    }

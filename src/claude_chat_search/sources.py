"""Conversation-source dispatch for Claude Code and Codex rollouts."""

from __future__ import annotations

from .codex_parser import iter_codex_jsonl_files, parse_codex_jsonl_file
from .parser import extract_session_metadata, iter_jsonl_files, parse_jsonl_file


SOURCE_NAMES = ("all", "claude", "codex")


def iter_conversation_files(source: str = "all") -> list[dict]:
    if source not in SOURCE_NAMES:
        raise ValueError(f"Unknown conversation source: {source}")
    files = []
    if source in {"all", "claude"}:
        files.extend(iter_jsonl_files())
    if source in {"all", "codex"}:
        files.extend(iter_codex_jsonl_files())
    return files


def parse_conversation_file(file_info: dict) -> dict:
    if file_info.get("source") == "codex":
        return parse_codex_jsonl_file(file_info["path"], file_info)
    return parse_jsonl_file(file_info["path"])


def extract_conversation_metadata(session_data: dict) -> dict:
    if session_data.get("source") == "codex":
        return {
            "files_touched": "[]",
            "tools_used": "[]",
            "commands_run": "[]",
        }
    return extract_session_metadata(session_data["messages"])

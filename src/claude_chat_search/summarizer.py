"""Generate topic summaries for sessions using Gemini Flash via otak's llm_providers."""

import json
import subprocess
import sys
from pathlib import Path

from .db import get_session_chunks, get_sessions_without_summary, update_topic_summary

OTAK_VENV_PYTHON = "/Users/stian/src/otak/.venv-otak/bin/python3"
OTAK_SCRIPTS_DIR = "/Users/stian/src/otak/scripts"

# Approximate chars per token for truncation
CHARS_PER_TOKEN = 4
HEAD_TOKENS = 2000
TAIL_TOKENS = 1000


def _build_session_text(chunks: list[dict]) -> str:
    """Build a text representation of a session from its chunks.

    Takes the first ~2000 tokens and last ~1000 tokens of the conversation.
    """
    parts = []
    for chunk in chunks:
        user = chunk.get("user_content", "")
        assistant = chunk.get("assistant_content", "")
        if user:
            parts.append(f"User: {user}")
        if assistant:
            parts.append(f"Assistant: {assistant}")

    full_text = "\n\n".join(parts)

    head_chars = HEAD_TOKENS * CHARS_PER_TOKEN
    tail_chars = TAIL_TOKENS * CHARS_PER_TOKEN

    if len(full_text) <= (head_chars + tail_chars):
        return full_text

    head = full_text[:head_chars]
    tail = full_text[-tail_chars:]
    return head + "\n\n[...middle of conversation omitted...]\n\n" + tail


def _call_gemini_summary(session_text: str) -> str | None:
    """Call Gemini Flash via subprocess to generate a summary.

    Uses the otak venv which has the google genai SDK installed.
    """
    script = f"""
import sys, json
sys.path.insert(0, {OTAK_SCRIPTS_DIR!r})
from llm_providers import generate_sync

text = json.loads(sys.stdin.read())

result, meta = generate_sync(
    prompt=text,
    system_prompt="Summarize the main topics and outcomes of this Claude Code conversation in 2-3 sentences. Focus on what was built, fixed, or discussed. Be specific about technologies and files involved. Do not start with 'This conversation' or 'The user'.",
    json_schema={{
        "type": "object",
        "properties": {{
            "summary": {{"type": "string"}}
        }},
        "required": ["summary"]
    }},
    model="gemini3-flash",
    max_tokens=256,
    thinking_budget=0,
)
print(json.dumps(result))
"""

    try:
        proc = subprocess.run(
            [OTAK_VENV_PYTHON, "-c", script],
            input=json.dumps(session_text),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            print(f"  LLM error: {proc.stderr[:200]}", file=sys.stderr)
            return None

        result = json.loads(proc.stdout.strip())
        return result.get("summary")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"  Summary generation failed: {e}", file=sys.stderr)
        return None


def summarize_sessions(conn, limit: int = 50, force_all: bool = False,
                       callback=None) -> int:
    """Generate topic summaries for sessions that don't have one.

    Returns the number of sessions summarized.
    """
    sessions = get_sessions_without_summary(conn, limit=limit)
    if not sessions:
        return 0

    count = 0
    for session in sessions:
        sid = session["session_id"]
        chunks = get_session_chunks(conn, sid)
        if not chunks:
            continue

        session_text = _build_session_text(chunks)
        if len(session_text) < 100:
            continue

        summary = _call_gemini_summary(session_text)
        if summary:
            update_topic_summary(conn, sid, summary)
            count += 1
            if callback:
                callback(sid, summary, count)

    return count

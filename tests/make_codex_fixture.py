"""Regenerate fixtures/codex_rollout_2026_09.jsonl.

The record types and keys mirror Codex rollouts written from September 2026
(no event_msg user_message/agent_message any more); all text is invented.
"""

import json
from pathlib import Path

SID = "01a0c000-0000-7000-8000-00000000f107"
TURN = "019ff000-0000-7000-8000-000000000001"
META = {"internal_chat_message_metadata_passthrough": None}


def ts(second):
    return f"2026-09-20T10:00:{second:02d}.000Z"


def item(second, type_, payload):
    return {"timestamp": ts(second), "type": type_, "payload": payload}


def message(second, role, blocks, phase=None, id_="msg"):
    kind = "output_text" if role == "assistant" else "input_text"
    payload = {"type": "message", "id": id_, "role": role, "phase": phase,
               "content": [{"type": kind, "text": b} for b in blocks], **META}
    return item(second, "response_item", payload)


RECORDS = [
    item(0, "session_meta", {
        "id": SID, "session_id": SID, "timestamp": ts(0), "cwd": "/tmp/fixture-project",
        "originator": "codex_vscode", "cli_version": "0.99.0", "source": "vscode",
        "thread_source": "user", "model_provider": "openai", "history_mode": "paginated",
        "base_instructions": {"text": "BASE_INSTRUCTIONS_NOISE"},
        "git": {"branch": "sh/fixture", "commit_hash": "0" * 40},
    }),
    item(0, "event_msg", {"type": "thread_settings_applied", "thread_id": SID,
                          "thread_settings": {"model": "fixture"}}),
    message(1, "developer", ["<permissions instructions>DEVELOPER_NOISE</permissions instructions>"]),
    message(1, "user", [
        "<environment_context>\n  <cwd>/tmp/fixture-project</cwd>\n  ENVIRONMENT_NOISE\n</environment_context>",
        "<recommended_plugins>\nPLUGIN_NOISE\n</recommended_plugins>",
        "# AGENTS.md instructions for /tmp/fixture-project\n\n<INSTRUCTIONS>\nAGENTS_NOISE\n</INSTRUCTIONS>",
    ]),
    item(2, "turn_context", {"turn_id": TURN, "cwd": "/tmp/fixture-project", "model": "fixture",
                             "approval_policy": "never", "summary": "auto"}),
    item(2, "event_msg", {"type": "task_started", "turn_id": TURN, "started_at": 1,
                          "model_context_window": 1000, "collaboration_mode_kind": "default"}),
    message(2, "user", ["How do the lighthouse keepers rotate their night shifts?"]),
    item(2, "event_msg", {"type": "item_completed", "thread_id": SID, "turn_id": TURN,
                          "started_at_ms": 1, "completed_at_ms": 2,
                          "item": {"type": "UserMessage", "id": "u1", "client_id": None,
                                   "content": [{"type": "text", "text": "How do the lighthouse keepers rotate their night shifts?",
                                                "text_elements": []}]}}),
    item(3, "response_item", {"type": "reasoning", "id": "r1", "summary": [],
                              "encrypted_content": "REASONING_NOISE", **META}),
    item(3, "event_msg", {"type": "item_completed", "thread_id": SID, "turn_id": TURN,
                          "started_at_ms": 3, "completed_at_ms": 4,
                          "item": {"type": "Reasoning", "id": "r1", "summary_text": ["REASONING_NOISE"],
                                   "raw_content": []}}),
    message(4, "assistant", ["I will read the rota file first."], phase="commentary"),
    item(4, "event_msg", {"type": "item_completed", "thread_id": SID, "turn_id": TURN,
                          "started_at_ms": 4, "completed_at_ms": 5,
                          "item": {"type": "AgentMessage", "id": "a1", "phase": "commentary",
                                   "delivery": None, "questions": None,
                                   "content": [{"type": "Text", "text": "I will read the rota file first."}]}}),
    item(5, "response_item", {"type": "custom_tool_call", "id": "c1", "call_id": "call_1",
                              "name": "exec", "status": "completed", "input": "TOOL_INPUT_NOISE", **META}),
    item(6, "response_item", {"type": "custom_tool_call_output", "id": "c1o", "call_id": "call_1",
                              "output": "TOOL_OUTPUT_NOISE", **META}),
    item(6, "response_item", {"type": "function_call", "id": "f1", "call_id": "call_2", "name": "shell",
                              "namespace": None, "arguments": "FUNCTION_ARGS_NOISE", **META}),
    item(6, "response_item", {"type": "function_call_output", "id": "f1o", "call_id": "call_2",
                              "name": "shell", "namespace": None, "output": "FUNCTION_OUTPUT_NOISE", **META}),
    item(7, "response_item", {"type": "agent_message", "id": "am1", "author": "worker",
                              "recipient": "root", "content": "INTER_AGENT_NOISE", **META}),
    item(7, "inter_agent_communication_metadata", {"trigger_turn": TURN}),
    message(7, "user", ["<subagent_notification>\nSUBAGENT_NOISE\n</subagent_notification>"]),
    item(8, "event_msg", {"type": "token_count", "info": {"total": 1}, "rate_limits": {}}),
    item(8, "token_usage_record", {"session_id": SID, "thread_id": SID, "turn_id": TURN,
                                   "root_turn_id": TURN, "response_id": "resp", "usage": {},
                                   "turn_token_usage": {}, "thread_token_usage": {}}),
    message(9, "assistant", ["Keepers rotate every third night, and the rota lives in keepers.csv."],
            phase="final_answer"),
    item(9, "event_msg", {"type": "task_complete", "turn_id": TURN, "started_at": 1, "completed_at": 9,
                          "duration_ms": 8000, "time_to_first_token_ms": 10, "error": None,
                          "last_agent_message": "Keepers rotate every third night, and the rota lives in keepers.csv."}),
    item(10, "world_state", {"full": "WORLD_STATE_NOISE", "state": {}}),
    item(11, "compacted", {"message": "COMPACTION_NOISE", "replacement_history": [],
                           "window_id": "w2", "previous_window_id": "w1", "window_number": 2}),
    message(12, "user", [
        "<in-app-browser-context>\nBROWSER_NOISE\n</in-app-browser-context>\nWhat does the storm clause say?",
        "<image name=[Image #1]>",
        "</image>",
    ]),
    message(12, "user", [
        "# Context from my IDE setup:\n\n## Active file: IDE_NOISE.md\n\n## My request for Codex:\nAdd a spare keeper to the rota.",
    ]),
    message(13, "user", ["<turn_aborted>\nABORT_NOISE\n</turn_aborted>"]),
    message(14, "assistant", ["The storm clause doubles the watch."], phase="final_answer"),
]

if __name__ == "__main__":
    out = Path(__file__).parent / "fixtures" / "codex_rollout_2026_09.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    print(out)

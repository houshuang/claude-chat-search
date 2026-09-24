import json
import tempfile
import unittest
from pathlib import Path

from claude_chat_search.chunker import create_chunks
from claude_chat_search.codex_parser import (
    iter_codex_jsonl_files,
    parse_codex_jsonl_file,
)


SESSION_ID = "019ff544-d87b-7fa2-b9af-a5a1e464bf08"


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def session_meta(session_id=SESSION_ID, *, thread_source="user", cwd="/tmp/project"):
    return {
        "timestamp": "2026-08-12T09:00:00Z",
        "type": "session_meta",
        "payload": {
            "id": session_id,
            "session_id": session_id,
            "timestamp": "2026-08-12T09:00:00Z",
            "cwd": cwd,
            "thread_source": thread_source,
            "git": {"branch": "sh/test"},
        },
    }


class CodexParserTests(unittest.TestCase):
    def test_indexes_only_visible_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"rollout-2026-08-12T09-00-00-{SESSION_ID}.jsonl"
            write_jsonl(path, [
                session_meta(),
                {"timestamp": "2026-08-12T09:00:01Z", "type": "response_item",
                 "payload": {"type": "message", "role": "developer",
                             "content": [{"type": "input_text", "text": "SYSTEM_NOISE"}]}},
                {"timestamp": "2026-08-12T09:00:02Z", "type": "event_msg",
                 "payload": {"type": "user_message", "message": "Please inspect the index."}},
                {"timestamp": "2026-08-12T09:00:03Z", "type": "response_item",
                 "payload": {"type": "reasoning", "summary": ["PRIVATE_REASONING"]}},
                {"timestamp": "2026-08-12T09:00:04Z", "type": "response_item",
                 "payload": {"type": "function_call", "name": "shell",
                             "arguments": "SECRET_TOOL_PAYLOAD_SHOULD_NOT_INDEX"}},
                {"timestamp": "2026-08-12T09:00:05Z", "type": "response_item",
                 "payload": {"type": "function_call_output",
                             "output": "SECRET_TOOL_OUTPUT_SHOULD_NOT_INDEX"}},
                {"timestamp": "2026-08-12T09:00:06Z", "type": "event_msg",
                 "payload": {"type": "agent_message", "phase": "commentary",
                             "message": "I am checking it now."}},
                {"timestamp": "2026-08-12T09:00:07Z", "type": "event_msg",
                 "payload": {"type": "agent_message", "phase": "final",
                             "message": "The index is healthy."}},
                {"timestamp": "2026-08-12T09:00:08Z", "type": "world_state",
                 "payload": {"full": "WORLD_STATE_NOISE"}},
            ])
            info = iter_codex_jsonl_files([Path(tmp)])[0]
            session = parse_codex_jsonl_file(path, info)

            self.assertEqual(session["message_count"], 3)
            self.assertEqual([m["type"] for m in session["messages"]], [
                "user", "assistant", "assistant"
            ])
            self.assertEqual(session["git_branch"], "sh/test")
            self.assertEqual(session["source"], "codex")

            indexed_text = "\n".join(c["combined_text"] for c in create_chunks(session))
            self.assertIn("Please inspect the index", indexed_text)
            self.assertIn("The index is healthy", indexed_text)
            for excluded in (
                "SYSTEM_NOISE", "PRIVATE_REASONING",
                "SECRET_TOOL_PAYLOAD_SHOULD_NOT_INDEX",
                "SECRET_TOOL_OUTPUT_SHOULD_NOT_INDEX", "WORLD_STATE_NOISE",
            ):
                self.assertNotIn(excluded, indexed_text)

    def test_subagent_rollouts_are_excluded_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main = root / f"rollout-main-{SESSION_ID}.jsonl"
            sub_id = "019ff545-d87b-7fa2-b9af-a5a1e464bf09"
            sub = root / f"rollout-sub-{sub_id}.jsonl"
            write_jsonl(main, [session_meta()])
            write_jsonl(sub, [session_meta(
                sub_id,
                thread_source={"subagent": {"parent_thread_id": SESSION_ID}},
            )])

            default_files = iter_codex_jsonl_files([root])
            all_files = iter_codex_jsonl_files([root], include_subagents=True)
            self.assertEqual([f["native_session_id"] for f in default_files], [SESSION_ID])
            self.assertEqual(len(all_files), 2)
            self.assertEqual(
                {f["thread_kind"] for f in all_files}, {"user", "subagent"}
            )

    def test_active_and_archived_copies_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "sessions" / f"rollout-active-{SESSION_ID}.jsonl"
            archived = root / "archived_sessions" / f"rollout-archived-{SESSION_ID}.jsonl"
            write_jsonl(active, [session_meta()])
            write_jsonl(archived, [session_meta(), {
                "timestamp": "2026-08-12T09:00:02Z", "type": "event_msg",
                "payload": {"type": "user_message", "message": "More complete"},
            }])

            files = iter_codex_jsonl_files([active.parent, archived.parent])
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0]["path"], archived)

    def test_legacy_top_level_messages_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy_id = "f5e957af-0210-4439-9ac0-a6dd6ed43846"
            path = Path(tmp) / f"rollout-legacy-{legacy_id}.jsonl"
            write_jsonl(path, [
                {"id": legacy_id, "timestamp": "2025-09-17T12:00:00Z"},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "Legacy question"}
                ]},
                {"type": "message", "role": "assistant", "content": "Legacy answer"},
            ])
            info = iter_codex_jsonl_files([Path(tmp)])[0]
            session = parse_codex_jsonl_file(path, info)
            self.assertEqual(session["message_count"], 2)
            self.assertEqual(session["messages"][0]["message"]["content"], "Legacy question")
            self.assertEqual(session["messages"][1]["message"]["content"], "Legacy answer")


FIXTURE = Path(__file__).parent / "fixtures" / "codex_rollout_2026_09.jsonl"


class CurrentCodexFormatTests(unittest.TestCase):
    """Rollouts written since September 2026 carry no event_msg conversation."""

    def setUp(self):
        info = iter_codex_jsonl_files([FIXTURE.parent])[0]
        self.session = parse_codex_jsonl_file(FIXTURE, info)
        self.texts = [m["message"]["content"] for m in self.session["messages"]]

    def test_visible_conversation_is_parsed(self):
        self.assertEqual(self.texts, [
            "How do the lighthouse keepers rotate their night shifts?",
            "I will read the rota file first.",
            "Keepers rotate every third night, and the rota lives in keepers.csv.",
            "What does the storm clause say?",
            "Add a spare keeper to the rota.",
            "The storm clause doubles the watch.",
        ])
        self.assertEqual(
            [m["type"] for m in self.session["messages"]],
            ["user", "assistant", "assistant", "user", "user", "assistant"],
        )
        self.assertEqual(self.session["cwd"], "/tmp/fixture-project")
        self.assertEqual(self.session["git_branch"], "sh/fixture")
        self.assertGreater(self.session["agent_record_count"], 0)

    def test_injected_context_and_agent_internals_are_excluded(self):
        indexed = "\n".join(c["combined_text"] for c in create_chunks(self.session))
        for noise in (
            "BASE_INSTRUCTIONS_NOISE", "DEVELOPER_NOISE", "ENVIRONMENT_NOISE",
            "PLUGIN_NOISE", "AGENTS_NOISE", "REASONING_NOISE", "TOOL_INPUT_NOISE",
            "TOOL_OUTPUT_NOISE", "FUNCTION_ARGS_NOISE", "FUNCTION_OUTPUT_NOISE",
            "INTER_AGENT_NOISE", "SUBAGENT_NOISE", "WORLD_STATE_NOISE",
            "COMPACTION_NOISE", "BROWSER_NOISE", "IDE_NOISE", "ABORT_NOISE",
            "<image", "AGENTS.md",
        ):
            self.assertNotIn(noise, indexed)

    def test_item_completed_events_do_not_duplicate_messages(self):
        self.assertEqual(
            self.texts.count("How do the lighthouse keepers rotate their night shifts?"), 1
        )

    def test_unused_rollout_is_not_counted_as_agent_activity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"rollout-unused-{SESSION_ID}.jsonl"
            write_jsonl(path, [
                session_meta(),
                {"timestamp": "2026-09-20T10:00:01Z", "type": "event_msg",
                 "payload": {"type": "thread_settings_applied"}},
                {"timestamp": "2026-09-20T10:00:01Z", "type": "response_item",
                 "payload": {"type": "message", "role": "user", "content": [
                     {"type": "input_text",
                      "text": "<environment_context>x</environment_context>"}]}},
            ])
            session = parse_codex_jsonl_file(path, iter_codex_jsonl_files([Path(tmp)])[0])
            self.assertEqual(session["message_count"], 0)
            self.assertEqual(session["agent_record_count"], 0)

    def test_pasted_html_is_kept(self):
        from claude_chat_search.codex_parser import _visible_user_text
        self.assertEqual(_visible_user_text("<div>hello</div> why is this red?"),
                         "<div>hello</div> why is this red?")


if __name__ == "__main__":
    unittest.main()

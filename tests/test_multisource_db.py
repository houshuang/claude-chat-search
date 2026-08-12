import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from claude_chat_search import db
from claude_chat_search.cli import _run_index
from claude_chat_search.codex_parser import iter_codex_jsonl_files

from test_codex_parser import SESSION_ID, session_meta, write_jsonl


class MultiSourceDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "index.db"
        self.old_db_path = db.DB_PATH
        self.old_db_dir = db.DB_DIR
        db.DB_PATH = self.db_path
        db.DB_DIR = self.db_path.parent
        self.conn = db.get_connection()
        db.init_db(self.conn)

    def tearDown(self):
        self.conn.close()
        db.DB_PATH = self.old_db_path
        db.DB_DIR = self.old_db_dir
        self.tmp.cleanup()

    def test_schema_backfills_existing_rows_as_claude(self):
        self.conn.execute(
            """INSERT INTO sessions
               (session_id, project_path, message_count, indexed_at)
               VALUES ('old-session', '/tmp/project', 2, '2026-01-01T00:00:00Z')"""
        )
        # init_db is intentionally idempotent and performs additive backfills.
        db.init_db(self.conn)
        row = db.get_session(self.conn, "old-session")
        self.assertEqual(row["source"], "claude")
        self.assertEqual(row["native_session_id"], "old-session")

    def test_online_backup_preserves_database(self):
        db.insert_session(self.conn, {
            "session_id": "backup-test",
            "native_session_id": "backup-test",
            "source": "claude",
            "project_path": "/tmp/project",
            "message_count": 1,
        })
        destination = Path(self.tmp.name) / "backups" / "snapshot.db"
        self.assertEqual(db.backup_database(self.conn, destination), destination)

        original_path, original_dir = db.DB_PATH, db.DB_DIR
        try:
            db.DB_PATH, db.DB_DIR = destination, destination.parent
            copy_conn = db.get_connection()
            row = db.get_session(copy_conn, "backup-test")
            copy_conn.close()
        finally:
            db.DB_PATH, db.DB_DIR = original_path, original_dir
        self.assertEqual(row["source"], "claude")

    def test_codex_index_is_incremental_and_namespaced(self):
        rollout = Path(self.tmp.name) / f"rollout-{SESSION_ID}.jsonl"
        write_jsonl(rollout, [
            session_meta(),
            {"timestamp": "2026-08-12T09:00:01Z", "type": "event_msg",
             "payload": {"type": "user_message", "message": "Incremental question"}},
            {"timestamp": "2026-08-12T09:00:02Z", "type": "event_msg",
             "payload": {"type": "agent_message", "message": "Incremental answer"}},
        ])

        def discovered():
            return iter_codex_jsonl_files([Path(self.tmp.name)])

        with patch("claude_chat_search.sources.iter_codex_jsonl_files",
                   side_effect=discovered):
            self.assertEqual(_run_index(self.conn, source="codex"), 1)
            self.assertEqual(_run_index(self.conn, source="codex"), 0)

            write_jsonl(rollout, [
                session_meta(),
                {"timestamp": "2026-08-12T09:00:01Z", "type": "event_msg",
                 "payload": {"type": "user_message", "message": "Changed question"}},
                {"timestamp": "2026-08-12T09:00:02Z", "type": "event_msg",
                 "payload": {"type": "agent_message", "message": "Changed answer"}},
            ])
            self.assertEqual(_run_index(self.conn, source="codex"), 1)

        row = db.get_session(self.conn, f"codex:{SESSION_ID}")
        self.assertEqual(row["source"], "codex")
        self.assertEqual(row["native_session_id"], SESSION_ID)
        self.assertEqual(row["tools_used"], "[]")
        chunks = db.get_session_chunks(self.conn, row["session_id"])
        combined = "\n".join(chunk["combined_text"] for chunk in chunks)
        self.assertIn("Changed question", combined)
        self.assertNotIn("Incremental question", combined)

    def test_parse_failure_preserves_previously_indexed_session(self):
        rollout = Path(self.tmp.name) / f"rollout-{SESSION_ID}.jsonl"
        write_jsonl(rollout, [
            session_meta(),
            {"timestamp": "2026-08-12T09:00:01Z", "type": "event_msg",
             "payload": {"type": "user_message", "message": "Keep this question"}},
            {"timestamp": "2026-08-12T09:00:02Z", "type": "event_msg",
             "payload": {"type": "agent_message", "message": "Keep this answer"}},
        ])

        def discovered():
            return iter_codex_jsonl_files([Path(self.tmp.name)])

        with patch("claude_chat_search.sources.iter_codex_jsonl_files",
                   side_effect=discovered):
            self.assertEqual(_run_index(self.conn, source="codex"), 1)
            rollout.write_text(rollout.read_text() + "{malformed but changed}\n")
            with patch("claude_chat_search.cli.parse_conversation_file",
                       side_effect=RuntimeError("parser failed")):
                self.assertEqual(_run_index(self.conn, source="codex"), 0)

        chunks = db.get_session_chunks(self.conn, f"codex:{SESSION_ID}")
        combined = "\n".join(chunk["combined_text"] for chunk in chunks)
        self.assertIn("Keep this question", combined)
        self.assertIn("Keep this answer", combined)


if __name__ == "__main__":
    unittest.main()

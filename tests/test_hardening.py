import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import apsw
from click.testing import CliRunner

from claude_chat_search import cli as cli_module
from claude_chat_search import daemon, db, embedder, parser
from claude_chat_search.cli import _run_index, cli
from claude_chat_search.codex_parser import iter_codex_jsonl_files

from test_codex_parser import SESSION_ID, session_meta, write_jsonl


class TempIndexCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "index.db"
        self.patches = [
            patch.object(db, "DB_PATH", self.db_path),
            patch.object(db, "DB_DIR", self.root),
            patch.object(cli_module, "DB_PATH", self.db_path),
            patch.object(parser, "EXCLUDED_PROJECTS_PATH", self.root / "excluded_projects.txt"),
            patch.object(parser, "GIT_REMOTE_CACHE_PATH", self.root / "git-remotes.json"),
            patch.object(parser, "_git_remote_cache", None),
            patch.object(parser, "PROJECTS_DIR", self.root / "projects"),
            patch.object(cli_module, "PROJECTS_DIR", self.root / "projects"),
        ]
        for p in self.patches:
            p.start()
        self.conn = db.get_connection()
        db.init_db(self.conn)

    def tearDown(self):
        self.conn.close()
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def add_session(self, sid, project_path, cwd=None, transcript_path=None, chunks=1):
        db.insert_session(self.conn, {
            "session_id": sid, "native_session_id": sid, "source": "claude",
            "project_path": project_path, "cwd": cwd, "message_count": 2,
            "transcript_path": transcript_path,
        })
        ids = db.insert_chunks(self.conn, [{
            "session_id": sid, "user_content": "q", "assistant_content": "a",
            "combined_text": f"text for {sid} {i}", "turn_number": i,
        } for i in range(chunks)])
        db.insert_embeddings(self.conn, ids, [[0.1] * db.embedding_dim() for _ in ids])


class ConnectionTests(TempIndexCase):
    def test_pragmas(self):
        self.assertEqual(self.conn.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        self.assertEqual(self.conn.execute("PRAGMA synchronous").fetchone()[0], 1)
        self.assertEqual(self.conn.execute("PRAGMA cache_size").fetchone()[0], -64000)
        self.assertEqual(self.conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_init_db_is_versioned_and_skipped_once_current(self):
        self.assertEqual(db.schema_version(self.conn), db.SCHEMA_VERSION)
        with patch.object(db, "_migrate") as migrate:
            db.init_db(self.conn)
        migrate.assert_not_called()

    def test_read_connection_cannot_write(self):
        reader = db.get_read_connection()
        try:
            with self.assertRaises(apsw.ReadOnlyError):
                reader.execute("DELETE FROM sessions")
        finally:
            reader.close()

    def test_read_connection_migrates_an_old_database_once(self):
        self.conn.execute("PRAGMA user_version = 0")
        reader = db.get_read_connection()
        try:
            self.assertEqual(db.schema_version(reader), db.SCHEMA_VERSION)
        finally:
            reader.close()

    def test_insert_embeddings_is_atomic(self):
        self.add_session("s1", "/p", chunks=0)
        ids = db.insert_chunks(self.conn, [{
            "session_id": "s1", "user_content": "", "assistant_content": "",
            "combined_text": f"t{i}", "turn_number": i,
        } for i in range(3)])
        bad = [[0.1] * db.embedding_dim(), [0.1] * db.embedding_dim(), [0.1] * 3]
        with self.assertRaises(apsw.Error):
            db.insert_embeddings(self.conn, ids, bad)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 1").fetchone()[0], 0
        )


class BackupTests(TempIndexCase):
    def test_backup_keep_prunes_and_checks_integrity(self):
        self.add_session("s1", "/p")
        backups = self.root / "backups"
        backups.mkdir()
        for stamp in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
            (backups / f"index-{stamp}.db").write_bytes(b"old")
        (backups / "unrelated.db").write_bytes(b"keep me")

        result = CliRunner().invoke(cli, ["backup", "--keep", "2"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Integrity check: ok", result.output)
        remaining = sorted(p.name for p in backups.iterdir())
        self.assertEqual(len([n for n in remaining if n.startswith("index-")]), 2)
        self.assertIn("index-20260103T000000Z.db", remaining)
        self.assertIn("unrelated.db", remaining)
        self.assertFalse([n for n in remaining if n.endswith(("-wal", "-shm"))])

    def test_failed_integrity_check_fails_loudly_and_prunes_nothing(self):
        backups = self.root / "backups"
        backups.mkdir()
        old = backups / "index-20260101T000000Z.db"
        old.write_bytes(b"old")
        with patch.object(cli_module, "integrity_check", return_value=["page 3: corrupt"]):
            result = CliRunner().invoke(cli, ["backup", "--keep", "1"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("integrity_check", result.output)
        self.assertTrue(old.exists())
        self.assertEqual(len(list(backups.glob("*.failed-integrity"))), 1)


class ExclusionTests(TempIndexCase):
    def setUp(self):
        super().setUp()
        parser.EXCLUDED_PROJECTS_PATH.write_text(
            "/Users/x/src/priv\n/Users/x/src/research/secret-map-2026-09\n"
        )

    def test_hyphenated_exclusion_matches_encoded_dir_and_cwd(self):
        excluded = parser.load_excluded_projects()
        encoded = "-Users-x-src-research-secret-map-2026-09"
        decoded = parser.decode_project_path(encoded)
        self.assertEqual(decoded, "/Users/x/src/research/secret/map/2026/09")
        self.assertTrue(parser.is_excluded_project(decoded, excluded))
        self.assertTrue(parser.is_excluded_project(decoded, excluded, encoded_dir=encoded))
        self.assertTrue(parser.is_excluded_project(
            "/elsewhere", excluded, cwd="/Users/x/src/research/secret-map-2026-09/sub"))
        self.assertTrue(parser.is_excluded_project(
            "/Users/x/src/priv/a/b", excluded, encoded_dir="-Users-x-src-priv-a-b"))
        self.assertFalse(parser.is_excluded_project(
            "/Users/x/src/research/secret", excluded, encoded_dir="-Users-x-src-research-secret"))
        self.assertFalse(parser.is_excluded_project("/Users/x/src/private", excluded))

    def test_claude_discovery_skips_hyphenated_excluded_dir(self):
        projects = self.root / "projects"
        for name in ("-Users-x-src-research-secret-map-2026-09", "-Users-x-src-ok"):
            (projects / name).mkdir(parents=True)
            (projects / name / "abc.jsonl").write_text("{}\n")
        found = [f["project_path"] for f in parser.iter_jsonl_files()]
        self.assertEqual(found, ["/Users/x/src/ok"])
        self.assertIsNone(parser.file_info_from_path(
            projects / "-Users-x-src-research-secret-map-2026-09" / "abc.jsonl"))

    def test_purge_excluded_dry_run_then_delete(self):
        self.add_session("leak", "/Users/x/src/research/secret/map/2026/09",
                         cwd="/Users/x/src/research/secret-map-2026-09", chunks=2)
        self.add_session("priv", "/Users/x/src/priv/deep", chunks=1)
        self.add_session("keep", "/Users/x/src/ok", cwd="/Users/x/src/ok", chunks=1)

        runner = CliRunner()
        dry = runner.invoke(cli, ["purge-excluded", "--dry-run"])
        self.assertEqual(dry.exit_code, 0, dry.output)
        self.assertIn("/Users/x/src/research/secret-map-2026-09: 1 session(s), 2 chunk(s), 2 vector(s)",
                      dry.output)
        self.assertIn("Would delete 2 session(s), 3 chunk(s), 3 vector(s)", dry.output)
        self.assertIsNotNone(db.get_session(self.conn, "leak"))

        real = runner.invoke(cli, ["purge-excluded"])
        self.assertEqual(real.exit_code, 0, real.output)
        self.assertIsNone(db.get_session(self.conn, "leak"))
        self.assertIsNone(db.get_session(self.conn, "priv"))
        self.assertIsNotNone(db.get_session(self.conn, "keep"))
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0], 1)
        fts = self.conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'leak'").fetchone()[0]
        self.assertEqual(fts, 0)

    def test_search_log_skips_excluded_cwd(self):
        log = self.root / "search.log"
        with patch("os.getcwd", return_value="/Users/x/src/research/secret-map-2026-09"):
            cli_module._log_search("q", "hybrid", [], 1.0)
        self.assertFalse(log.exists())
        with patch("os.getcwd", return_value="/Users/x/src/ok"):
            cli_module._log_search("q", "hybrid", [], 1.0)
        self.assertTrue(log.exists())


class CodexZeroYieldTests(TempIndexCase):
    def _write_rollout(self, name, records):
        path = self.root / "codex" / f"rollout-{name}-{SESSION_ID[:-2]}{name[-2:]}.jsonl"
        write_jsonl(path, records)
        return path

    def test_index_fails_when_every_active_rollout_yields_nothing(self):
        # Agent activity, but in a record shape the parser does not know.
        self._write_rollout("aa01", [
            session_meta(SESSION_ID[:-2] + "01"),
            {"timestamp": "2026-09-20T10:00:01Z", "type": "future_item",
             "payload": {"type": "future_message", "text": "unparsed"}},
        ])
        with patch("claude_chat_search.sources.iter_codex_jsonl_files",
                   side_effect=lambda: iter_codex_jsonl_files([self.root / "codex"])):
            result = CliRunner().invoke(cli, ["index", "--source", "codex"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("yielded 0 messages", result.output)

    def test_unused_rollouts_do_not_fail_the_run(self):
        self._write_rollout("aa02", [session_meta(SESSION_ID[:-2] + "02")])
        stats = {}
        with patch("claude_chat_search.sources.iter_codex_jsonl_files",
                   side_effect=lambda: iter_codex_jsonl_files([self.root / "codex"])):
            _run_index(self.conn, source="codex", stats=stats)
        self.assertEqual(stats, {"codex_parsed_nonempty": 0, "codex_zero_yield": 0})


class EmbeddingIsolationTests(TempIndexCase):
    def test_bad_row_is_isolated_and_skipped(self):
        self.add_session("s1", "/p", chunks=0)
        ids = db.insert_chunks(self.conn, [{
            "session_id": "s1", "user_content": "", "assistant_content": "",
            "combined_text": text, "turn_number": i,
        } for i, text in enumerate(["good one", "POISON", "good two", "good three"])])

        def fake_embed(texts):
            if any("POISON" in t for t in texts):
                raise ValueError("cannot embed")
            return [[0.1] * db.embedding_dim() for _ in texts]

        with patch.object(embedder, "embed_texts", side_effect=fake_embed):
            rows = db.get_unembedded_chunks(self.conn, 10)
            stored, skipped = embedder.embed_rows(self.conn, rows)
        self.assertEqual((stored, skipped), (3, [ids[1]]))
        self.assertEqual(db.get_unembedded_chunks(self.conn, 10), [])
        state = dict(self.conn.execute("SELECT id, embedded FROM chunks"))
        self.assertEqual(state[ids[1]], -1)

    def test_model_failure_marks_nothing(self):
        self.add_session("s1", "/p", chunks=0)
        db.insert_chunks(self.conn, [{
            "session_id": "s1", "user_content": "", "assistant_content": "",
            "combined_text": "text", "turn_number": 0,
        }])
        with patch.object(embedder, "embed_texts", side_effect=RuntimeError("model gone")):
            with self.assertRaises(embedder.EmbeddingUnavailable):
                embedder.embed_rows(self.conn, db.get_unembedded_chunks(self.conn, 10))
        self.assertEqual(len(db.get_unembedded_chunks(self.conn, 10)), 1)


class DaemonDeferralTests(TempIndexCase):
    def setUp(self):
        super().setUp()
        for name in ("_last_indexed", "_file_fingerprints", "_reindex_times", "_deferred"):
            getattr(daemon, name).clear()
        self.addCleanup(lambda: [getattr(daemon, n).clear() for n in
                                 ("_last_indexed", "_file_fingerprints", "_reindex_times", "_deferred")])
        self.project = self.root / "projects" / "-tmp-proj"
        self.project.mkdir(parents=True)
        self.transcript = self.project / "sess-1.jsonl"
        self.write_turns(1)

    def write_turns(self, n):
        lines = []
        for i in range(n):
            lines.append({"type": "user", "sessionId": "sess-1", "cwd": "/tmp/proj",
                          "timestamp": f"2026-09-20T10:00:{i:02d}Z",
                          "message": {"role": "user", "content": f"question {i}"}})
            lines.append({"type": "assistant", "sessionId": "sess-1",
                          "timestamp": f"2026-09-20T10:00:{i:02d}Z",
                          "message": {"role": "assistant",
                                      "content": [{"type": "text", "text": f"answer {i}"}]}})
        self.transcript.write_text("".join(json.dumps(l) + "\n" for l in lines))

    def queue(self):
        (self.root / ".queue").write_text(str(self.transcript) + "\n")

    def test_path_queued_during_cooldown_is_indexed_after_it(self):
        with patch.object(daemon, "QUEUE_PATH", self.root / ".queue"), \
             patch.object(daemon, "QUEUE_PROCESSING", self.root / ".queue.processing"), \
             patch.object(daemon, "detect_git_remote", return_value=None):
            self.queue()
            self.assertEqual(daemon.process_queue(self.conn), 1)

            self.write_turns(2)
            self.queue()
            self.assertEqual(daemon.process_queue(self.conn), 0)
            self.assertIn("sess-1", daemon._deferred)
            self.assertEqual(daemon.process_deferred(self.conn), 0)

            # Pretend the cooldown has elapsed.
            daemon._last_indexed["sess-1"] -= daemon.COOLDOWN_SECONDS + 1
            due, path = daemon._deferred["sess-1"]
            daemon._deferred["sess-1"] = (time.monotonic() - 1, path)
            self.assertEqual(daemon.process_deferred(self.conn), 1)
            self.assertEqual(db.get_session(self.conn, "sess-1")["message_count"], 4)
            self.assertEqual(daemon._deferred, {})

    def test_periodic_scan_picks_up_unqueued_changes(self):
        with patch.object(daemon, "detect_git_remote", return_value=None):
            self.assertEqual(daemon.full_scan(self.conn), 1)
            self.assertEqual(daemon.full_scan(self.conn), 0)
            self.write_turns(3)
            daemon._last_indexed.clear()
            self.assertEqual(daemon.full_scan(self.conn), 1)
            self.assertEqual(db.get_session(self.conn, "sess-1")["message_count"], 6)

    def test_busy_database_defers_instead_of_dropping(self):
        with patch.object(daemon, "index_single_session", side_effect=apsw.BusyError("busy")):
            info = parser.file_info_from_path(self.transcript)
            self.assertFalse(daemon._index_with_cooldown(self.conn, info))
        self.assertIn("sess-1", daemon._deferred)

    def test_excluded_cwd_is_not_indexed(self):
        parser.EXCLUDED_PROJECTS_PATH.write_text("/tmp/proj\n")
        with patch.object(daemon, "detect_git_remote", return_value=None):
            # -tmp-proj also matches by encoding; use cwd-only case via a
            # directory whose name does not encode the excluded path.
            other = self.root / "projects" / "-started-elsewhere"
            other.mkdir()
            moved = other / "sess-1.jsonl"
            self.transcript.rename(moved)
            info = parser.file_info_from_path(moved)
            self.assertIsNotNone(info)
            self.assertFalse(daemon.index_single_session(self.conn, info))
        self.assertIsNone(db.get_session(self.conn, "sess-1"))


if __name__ == "__main__":
    unittest.main()

import json
import threading
import time
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from claude_chat_search import daemon, db, embedder, search, search_service, vector_search
from claude_chat_search.cli import KEYWORD_ONLY_NOTE, cli

from test_fast_search import FastSearchCase, unit


class GrepIndexTests(FastSearchCase):
    def plan(self, query):
        calls = []
        real = db._fetchall

        def spy(conn, sql, bindings=None):
            calls.append(sql)
            return real(conn, sql, bindings)

        with patch.object(db, "_fetchall", spy):
            rows = db.text_search(self.conn, query)
        plan = self.conn.execute("EXPLAIN QUERY PLAN " + calls[0], (f"%{query}%", 50)).fetchall()
        return rows, " ".join(row[-1] for row in plan)

    def test_substring_inside_a_word_is_found_through_the_index(self):
        self.add("s1", "/p", [("call readFullTranscript here", unit(0))])
        self.add("s2", "/p", [("nothing relevant", unit(1))])
        rows, plan = self.plan("FullTrans")
        self.assertEqual([r["session_id"] for r in rows], ["s1"])
        self.assertIn("VIRTUAL TABLE", plan)

    def test_short_query_reads_the_chunks(self):
        self.add("s1", "/p", [("an ab here", unit(0))])
        rows, plan = self.plan("ab")
        self.assertEqual([r["session_id"] for r in rows], ["s1"])
        self.assertNotIn("VIRTUAL TABLE", plan)

    def test_index_follows_reindexing_and_deletion(self):
        self.add("s1", "/p", [("first version text", unit(0))])
        with db.write_transaction(self.conn):
            db.sync_session_chunks(self.conn, "s1", [{
                "session_id": "s1", "user_content": "", "assistant_content": "",
                "combined_text": "second version text", "turn_number": 0,
            }])
        self.assertEqual(db.text_search(self.conn, "first version"), [])
        self.assertEqual(len(db.text_search(self.conn, "second version")), 1)
        with db.write_transaction(self.conn):
            db.delete_session_data(self.conn, "s1")
        self.assertEqual(db.text_search(self.conn, "second version"), [])

    def test_existing_index_is_filled_by_the_migration(self):
        self.add("s1", "/p", [("older chunk about watchdogs", unit(0))])
        for trigger in ("chunks_tri_ai", "chunks_tri_ad", "chunks_tri_au"):
            self.conn.execute(f"DROP TRIGGER {trigger}")
        self.conn.execute(f"DROP TABLE {db.GREP_INDEX_TABLE}")
        self.conn.execute("PRAGMA user_version = 4")
        db.init_db(self.conn)
        rows, plan = self.plan("watchdog")
        self.assertEqual([r["session_id"] for r in rows], ["s1"])
        self.assertIn("VIRTUAL TABLE", plan)

    def test_source_filter_applies(self):
        self.add("s1", "/p", [("shared needle", unit(0))])
        self.add("codex:1", "/p", [("shared needle", unit(1))], source="codex")
        rows = db.text_search(self.conn, "needle", source="codex")
        self.assertEqual([r["session_id"] for r in rows], ["codex:1"])


class KeywordOnlyFallbackTests(FastSearchCase):
    def setUp(self):
        super().setUp()
        self.add("s1", "/work/alpha", [("fallback keyword alpha", unit(0))])

    def test_search_without_a_server_is_keyword_only_and_never_loads_the_model(self):
        with patch.object(search, "embed_query", side_effect=AssertionError("model loaded")), \
                patch("os.getcwd", return_value="/work"):
            result = CliRunner().invoke(cli, ["search", "fallback keyword"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("/work/alpha", result.output)
        self.assertIn(KEYWORD_ONLY_NOTE, result.stderr)
        entry = json.loads((self.root / "search.log").read_text().splitlines()[-1])
        self.assertTrue(entry.get("keyword_only"))
        self.assertNotIn("daemon", entry)

    def test_cross_search_without_a_server_never_loads_the_model(self):
        with patch("claude_chat_search.cross_search.embed_query",
                   side_effect=AssertionError("model loaded")), \
                patch("claude_chat_search.cross_search._search_research_index", return_value=[]), \
                patch("os.getcwd", return_value="/work"):
            result = CliRunner().invoke(cli, ["cross", "fallback keyword"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(KEYWORD_ONLY_NOTE, result.stderr)


class AskTests(unittest.TestCase):
    def test_starts_a_server_and_waits_until_it_is_ready(self):
        replies = iter([None, None, {"ok": True, "ready": False},
                        {"ok": True, "ready": True}, {"ok": True, "results": []}])
        with patch.object(search_service, "request", side_effect=lambda *a: next(replies)), \
                patch.object(search_service, "spawn_server", return_value=True) as spawn, \
                patch.object(search_service.time, "sleep"):
            reply = search_service.ask("search", {"query": "q"}, 5, start_timeout=30)
        self.assertEqual(reply, {"ok": True, "results": []})
        spawn.assert_called_once()

    def test_a_running_server_is_not_started_again(self):
        replies = iter([{"ok": True, "ready": True}, {"ok": True, "results": []}])
        with patch.object(search_service, "request", side_effect=lambda *a: next(replies)), \
                patch.object(search_service, "spawn_server") as spawn:
            self.assertIsNotNone(search_service.ask("search", {"query": "q"}, 5))
        spawn.assert_not_called()

    def test_gives_up_when_the_server_is_not_ready_in_time(self):
        with patch.object(search_service, "request", return_value={"ok": True, "ready": False}), \
                patch.object(search_service, "spawn_server") as spawn:
            self.assertIsNone(search_service.ask("search", {"query": "q"}, 5, start_timeout=0.3))
        spawn.assert_not_called()

    def test_no_server_when_starting_is_disabled(self):
        with patch.object(search_service, "request", return_value=None), \
                patch.dict("os.environ", {search_service.NO_SERVER_ENV: "1"}), \
                patch.object(search_service.subprocess, "Popen") as popen:
            self.assertIsNone(search_service.ask("search", {"query": "q"}, 5, start_timeout=5))
        popen.assert_not_called()


class ServerTests(FastSearchCase):
    def setUp(self):
        super().setUp()
        self.add("s1", "/work/alpha", [("socket search alpha", unit(0))])
        for target, value in ((search_service, "CONNECT_TIMEOUT"),):
            p = patch.object(target, value, 5.0)
            p.start()
            self.addCleanup(p.stop)
        embed = patch.object(search, "embed_query", return_value=unit(0))
        embed.start()
        self.addCleanup(embed.stop)

    def start_server(self):
        server = search_service.SearchServer()
        server.start()
        self.addCleanup(server.stop)
        server.service.ready.set()
        return server

    def test_a_search_abandoned_while_queued_is_not_run(self):
        server = self.start_server()
        first_running = threading.Event()
        release = threading.Event()
        ran = []
        real = search_service.SearchService._search

        def slow(conn, **kwargs):
            ran.append(kwargs["query"])
            if kwargs["query"] == "first":
                first_running.set()
                release.wait(10)
            return real(conn, **kwargs)

        with patch.object(search_service.SearchService, "_search", staticmethod(slow)):
            first = threading.Thread(
                target=search_service.request, args=("search", {"query": "first"}, 10))
            first.start()
            self.assertTrue(first_running.wait(5))
            self.assertIsNone(search_service.request("search", {"query": "abandoned"}, 0.3))
            release.set()
            first.join()
            reply = search_service.request("search", {"query": "socket search"}, 5)
        self.assertIsNotNone(reply)
        self.assertEqual(ran, ["first", "socket search"])

    def test_idle_time_counts_from_the_last_finished_request(self):
        server = self.start_server()
        service = server.service
        service.last_used = time.monotonic() - 100
        self.assertGreaterEqual(service.idle_for(), 100)
        search_service.request("ping", {}, 5)
        self.assertGreaterEqual(service.idle_for(), 100)
        search_service.request("search", {"query": "socket search"}, 5)
        self.assertLess(service.idle_for(), 50)

    def test_server_exits_when_idle_and_removes_its_socket(self):
        with patch.object(search_service.SearchService, "warm_up",
                          lambda self: self.ready.set()), \
                patch.object(search_service, "_embed_while_serving"), \
                patch.object(search_service.logging, "basicConfig"), \
                patch.object(search_service.signal, "signal"):
            self.assertEqual(search_service.serve(idle_seconds=0.2), 0)
        self.assertFalse(search_service.socket_path().exists())

    def test_second_server_exits_at_once(self):
        lock = search_service._server_lock()
        self.addCleanup(lock.close)
        with patch.object(search_service.SearchServer, "start",
                          side_effect=AssertionError("started")):
            self.assertEqual(search_service.serve(idle_seconds=0.2), 0)


class VectorCacheMemoryTests(FastSearchCase):
    def test_vectors_are_held_as_float16_and_read_in_blocks(self):
        ids = []
        for i in range(5):
            ids += self.add(f"s{i}", "/p", [(f"text {i}", unit(i))])
        with patch.object(vector_search, "_BLOCK", 2):
            hits = vector_search.numpy_vector_search(self.conn, unit(3), limit=2)
            self.assertEqual(hits[0]["chunk_id"], ids[3])
            self.assertAlmostEqual(hits[0]["distance"], 0.0, places=3)
        _ids, matrix, _sessions = vector_search._cache.state
        self.assertEqual(str(matrix.dtype), "float16")
        self.assertEqual(matrix.shape, (5, db.embedding_dim()))


class EmbedWorkerTests(FastSearchCase):
    def test_daemon_sees_pending_chunks(self):
        self.assertFalse(daemon.has_pending_chunks(self.conn))
        db.insert_session(self.conn, {
            "session_id": "p", "native_session_id": "p", "source": "claude",
            "project_path": "/p", "message_count": 2,
        })
        db.insert_chunks(self.conn, [{
            "session_id": "p", "user_content": "q", "assistant_content": "a",
            "combined_text": "pending", "turn_number": 0,
        }])
        self.assertTrue(daemon.has_pending_chunks(self.conn))

    def test_worker_reports_a_backlog_through_its_exit_status(self):
        with patch.object(embedder, "embed_pending", return_value=True) as embed:
            result = CliRunner().invoke(cli, ["embed-pending", "--seconds", "7"])
        self.assertEqual(result.exit_code, 3, result.output)
        self.assertEqual(embed.call_args.args[1], 7.0)
        with patch.object(embedder, "embed_pending", return_value=False):
            self.assertEqual(CliRunner().invoke(cli, ["embed-pending"]).exit_code, 0)

    def test_daemon_module_does_not_import_the_model_libraries(self):
        import subprocess
        import sys
        code = ("import sys, claude_chat_search.daemon; "
                "print(sorted(m for m in ('torch', 'sentence_transformers', 'transformers') "
                "if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={**__import__('os').environ, "PYTHONPATH": "../src"})
        self.assertEqual(out.stdout.strip(), "[]", out.stderr)


if __name__ == "__main__":
    unittest.main()

import json
import os
import socket
import stat
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from claude_chat_search import chunker, db, search, search_service, vector_search
from claude_chat_search.cli import _run_index, cli
from claude_chat_search.codex_parser import iter_codex_jsonl_files
from claude_chat_search.search import fts_search, hybrid_search

from test_codex_parser import SESSION_ID, session_meta, write_jsonl
from test_hardening import TempIndexCase


def unit(i: int, weight: float = 1.0) -> list[float]:
    vec = [0.0] * db.embedding_dim()
    vec[i] = weight
    return vec


def blend(i: int, j: int, wi: float) -> list[float]:
    vec = [0.0] * db.embedding_dim()
    vec[i] = wi
    vec[j] = 1.0 - wi
    return vec


class FastSearchCase(TempIndexCase):
    def setUp(self):
        super().setUp()
        vector_search.invalidate_cache()

    def add(self, sid, project, texts_and_vectors, source="claude", last="2026-09-20T00:00:00+00:00"):
        db.insert_session(self.conn, {
            "session_id": sid, "native_session_id": sid, "source": source,
            "project_path": project, "message_count": 2,
            "first_message_at": last, "last_message_at": last,
        })
        ids = db.insert_chunks(self.conn, [{
            "session_id": sid, "user_content": text, "assistant_content": "",
            "combined_text": text, "turn_number": i,
        } for i, (text, _vec) in enumerate(texts_and_vectors)])
        db.insert_embeddings(self.conn, ids, [vec for _t, vec in texts_and_vectors])
        return ids


class IncrementalReindexTests(FastSearchCase):
    def chunk(self, turn, text):
        return {"session_id": "s1", "user_content": text, "assistant_content": "",
                "combined_text": text, "turn_number": turn}

    def test_unchanged_chunks_keep_rows_and_vectors(self):
        ids = self.add("s1", "/p", [("one", unit(0)), ("two", unit(1)), ("three", unit(2))])
        with db.write_transaction(self.conn):
            counts = db.sync_session_chunks(self.conn, "s1", [
                self.chunk(0, "one"), self.chunk(1, "two changed"), self.chunk(3, "four"),
            ])
        self.assertEqual(counts, {"kept": 1, "deleted": 2, "inserted": 2})
        rows = dict(self.conn.execute("SELECT combined_text, embedded FROM chunks"))
        self.assertEqual(rows, {"one": 1, "two changed": 0, "four": 0})
        kept_id = self.conn.execute("SELECT id FROM chunks WHERE combined_text='one'").fetchone()[0]
        self.assertEqual(kept_id, ids[0])
        vec_ids = {r[0] for r in self.conn.execute("SELECT chunk_id FROM vec_chunks")}
        self.assertEqual(vec_ids, {ids[0]})
        fts = self.conn.execute(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'three'").fetchone()[0]
        self.assertEqual(fts, 0)

    def test_rows_without_a_stored_hash_are_matched(self):
        ids = self.add("s1", "/p", [("one", unit(0)), ("two", unit(1))])
        self.conn.execute("UPDATE chunks SET content_hash = NULL")
        with db.write_transaction(self.conn):
            counts = db.sync_session_chunks(
                self.conn, "s1", [self.chunk(0, "one"), self.chunk(1, "two")])
        self.assertEqual(counts, {"kept": 2, "deleted": 0, "inserted": 0})
        self.assertEqual(
            [r[0] for r in self.conn.execute("SELECT id FROM chunks ORDER BY id")], ids)
        self.assertEqual(self.conn.execute(
            "SELECT count(*) FROM chunks WHERE content_hash IS NULL").fetchone()[0], 0)

    def test_repeated_text_in_one_turn_is_matched_per_occurrence(self):
        self.add("s1", "/p", [("same", unit(0))])
        with db.write_transaction(self.conn):
            counts = db.sync_session_chunks(
                self.conn, "s1", [self.chunk(0, "same"), self.chunk(0, "same")])
        self.assertEqual(counts, {"kept": 1, "deleted": 0, "inserted": 1})

    def test_reindexing_a_grown_conversation_only_embeds_new_turns(self):
        rollout = self.root / "codex" / f"rollout-{SESSION_ID}.jsonl"
        turns = []
        for i in range(4):
            turns += [
                {"timestamp": f"2026-08-12T09:00:{i:02d}Z", "type": "event_msg",
                 "payload": {"type": "user_message", "message": f"Question {i} " + "word " * 250}},
                {"timestamp": f"2026-08-12T09:00:{i:02d}Z", "type": "event_msg",
                 "payload": {"type": "agent_message", "message": f"Answer {i} " + "word " * 250}},
            ]
        write_jsonl(rollout, [session_meta()] + turns[:6])
        discovered = lambda: iter_codex_jsonl_files([rollout.parent])
        with patch("claude_chat_search.sources.iter_codex_jsonl_files", side_effect=discovered):
            _run_index(self.conn, source="codex")
            first = self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            self.conn.execute("UPDATE chunks SET embedded = 1")
            write_jsonl(rollout, [session_meta()] + turns)
            _run_index(self.conn, source="codex")
        pending = self.conn.execute("SELECT count(*) FROM chunks WHERE embedded = 0").fetchone()[0]
        total = self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        self.assertEqual(first, 3)
        self.assertEqual(total, 4)
        self.assertEqual(pending, 1)


class MigrationTests(FastSearchCase):
    def test_migration_removes_orphans_and_resets_missing_vectors(self):
        ids = self.add("s1", "/p", [("one", unit(0)), ("two", unit(1))])
        self.conn.execute("INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                          (9999, db.serialize_embedding(unit(2))))
        self.conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (ids[1],))
        self.conn.execute("PRAGMA user_version = 1")
        db.init_db(self.conn)
        vec_ids = {r[0] for r in self.conn.execute("SELECT chunk_id FROM vec_chunks")}
        self.assertEqual(vec_ids, {ids[0]})
        self.assertEqual(dict(self.conn.execute("SELECT id, embedded FROM chunks")),
                         {ids[0]: 1, ids[1]: 0})
        self.assertEqual(db.schema_version(self.conn), db.SCHEMA_VERSION)

    def test_marking_embedded_does_not_touch_fts(self):
        trigger = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'chunks_au'").fetchone()[0]
        self.assertIn("UPDATE OF combined_text", trigger)
        ids = self.add("s1", "/p", [("alpha", unit(0))])
        self.conn.execute("UPDATE chunks SET embedded = 0 WHERE id = ?", (ids[0],))
        self.conn.execute("UPDATE chunks SET combined_text = 'beta' WHERE id = ?", (ids[0],))
        match = lambda term: self.conn.execute(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (term,)).fetchone()[0]
        self.assertEqual((match("alpha"), match("beta")), (0, 1))
        self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")


class VectorCacheTests(FastSearchCase):
    def setUp(self):
        super().setUp()
        interval = patch.object(vector_search, "MIN_REFRESH_INTERVAL", 0.0)
        interval.start()
        self.addCleanup(interval.stop)

    def test_refresh_adds_and_drops_rows_incrementally(self):
        ids = self.add("s1", "/p", [("one", unit(0)), ("two", unit(1))])
        reader = db.get_read_connection()
        self.addCleanup(reader.close)
        cache = vector_search.VectorCache()
        cache.refresh(reader)
        self.assertEqual(cache.search(unit(1), 1)[0]["chunk_id"], ids[1])

        new = self.add("s2", "/p", [("three", unit(2))])
        with db.write_transaction(self.conn):
            db.sync_session_chunks(self.conn, "s1", [{
                "session_id": "s1", "user_content": "", "assistant_content": "",
                "combined_text": "one", "turn_number": 0}])
        with patch.object(cache, "_read_vectors", wraps=cache._read_vectors) as reads:
            cache.refresh(reader)
        self.assertEqual(reads.call_args.args[1].tolist(), new)
        self.assertEqual(sorted(cache.state[0].tolist()), [ids[0], new[0]])
        self.assertEqual(cache.search(unit(2), 1)[0]["chunk_id"], new[0])

    def test_unchanged_database_is_not_reread(self):
        self.add("s1", "/p", [("one", unit(0))])
        reader = db.get_read_connection()
        self.addCleanup(reader.close)
        cache = vector_search.VectorCache()
        cache.refresh(reader)
        with patch.object(cache, "_sync") as sync:
            cache.refresh(reader)
        sync.assert_not_called()

    def test_changes_within_the_refresh_interval_are_not_resynced(self):
        self.add("s1", "/p", [("one", unit(0))])
        reader = db.get_read_connection()
        self.addCleanup(reader.close)
        cache = vector_search.VectorCache()
        cache.refresh(reader)
        self.add("s2", "/p", [("two", unit(1))])
        with patch.object(vector_search, "MIN_REFRESH_INTERVAL", 60.0), \
                patch.object(cache, "_sync") as sync:
            cache.refresh(reader)
        sync.assert_not_called()

    def test_writes_without_new_vectors_skip_the_matrix_rebuild(self):
        self.add("s1", "/p", [("one", unit(0))])
        reader = db.get_read_connection()
        self.addCleanup(reader.close)
        cache = vector_search.VectorCache()
        cache.refresh(reader)
        matrix = cache.state[1]
        with db.write_transaction(self.conn):
            self.conn.execute("UPDATE sessions SET project_path = project_path")
        with patch.object(cache, "_read_vectors") as reads:
            cache.refresh(reader)
        reads.assert_not_called()
        self.assertIs(cache.state[1], matrix)

    def test_search_within_allowed_sessions(self):
        self.add("near", "/p", [("a", unit(0))])
        far = self.add("far", "/p", [("b", blend(0, 1, 0.1))])
        results = vector_search.numpy_vector_search(
            self.conn, unit(0), limit=5, allowed_sessions={"far"})
        self.assertEqual([r["chunk_id"] for r in results], far)
        self.assertEqual(vector_search.numpy_vector_search(
            self.conn, unit(0), limit=5, allowed_sessions={"missing"}), [])


class FilterBeforeRankingTests(FastSearchCase):
    def setUp(self):
        super().setUp()
        # 60 sessions outrank the one inside the filter on both rankings.
        for n in range(60):
            self.add(f"other{n}", "/work/other", [(f"deploy deploy deploy note {n}", unit(0))])
        self.target = self.add("target", "/work/wanted", [("deploy note", blend(0, 1, 0.2))])

    def test_fts_applies_session_filter(self):
        where = search._session_filter_sql(self.conn, project="wanted")
        self.assertEqual([r["chunk_id"] for r in fts_search(self.conn, "deploy", 5, where)],
                         self.target)
        self.assertEqual(len(fts_search(self.conn, "deploy", 5)), 5)

    def test_filtered_search_finds_low_ranked_session(self):
        with patch.object(search, "embed_query", return_value=unit(0)):
            results = hybrid_search(self.conn, "deploy", limit=1, project="wanted")
        self.assertEqual([r["session_id"] for r in results], ["target"])

    def test_filter_matching_nothing_returns_nothing(self):
        with patch.object(search, "embed_query", return_value=unit(0)):
            self.assertEqual(hybrid_search(self.conn, "deploy", limit=3, project="nowhere"), [])


class RerankPoolTests(FastSearchCase):
    def test_rerank_sees_four_times_limit_and_returns_limit(self):
        for n in range(20):
            self.add(f"s{n:02d}", "/p", [(f"topic text {n}", blend(0, 1, 1 - n / 40))])
        seen = {}

        def fake_rerank(query, results):
            seen["n"] = len(results)
            return list(reversed(results))

        with patch.object(search, "embed_query", return_value=unit(0)), \
                patch("limbic.amygdala.rerank", side_effect=fake_rerank):
            results = hybrid_search(self.conn, "topic", limit=3, do_rerank=True)
        self.assertEqual(seen["n"], 12)
        self.assertEqual(len(results), 3)
        # The best reranked results come from beyond the top 3 of the fused list.
        with patch.object(search, "embed_query", return_value=unit(0)):
            plain = hybrid_search(self.conn, "topic", limit=12)
        self.assertEqual([r["session_id"] for r in results],
                         [r["session_id"] for r in plain[::-1][:3]])


class ChunkerTests(unittest.TestCase):
    def chunks_for(self, user, assistant):
        return chunker.create_chunks({"session_id": "s", "messages": [
            {"type": "user", "message": {"role": "user", "content": user},
             "timestamp": "2026-09-01T00:00:00Z"},
            {"type": "assistant", "timestamp": "2026-09-01T00:00:01Z",
             "message": {"role": "assistant", "content": [{"type": "text", "text": assistant}]}},
        ]})

    def assert_bounded(self, chunks):
        self.assertTrue(chunks)
        for c in chunks:
            self.assertLessEqual(chunker.count_tokens(c["combined_text"]),
                                 chunker.MAX_CHUNK_TOKENS + 10)

    def test_single_paragraph_without_breaks_is_split(self):
        blob = "x" * 40000
        chunks = self.chunks_for("what is this", blob)
        self.assert_bounded(chunks)
        self.assertEqual("".join(c["assistant_content"] for c in chunks), blob)

    def test_long_paragraph_is_split_at_sentences(self):
        text = " ".join(f"Sentence number {i} says something." for i in range(800))
        chunks = self.chunks_for("summarise", text)
        self.assert_bounded(chunks)
        self.assertGreater(len(chunks), 5)

    def test_long_prompt_gets_its_own_chunks(self):
        prompt = "\n".join(f"log line {i} with some detail" for i in range(3000))
        chunks = self.chunks_for(prompt, "The log shows a timeout.")
        self.assert_bounded(chunks)
        self.assertIn("The log shows a timeout.", chunks[-1]["combined_text"])
        self.assertTrue(all(c["user_content"] for c in chunks))

    def test_paragraph_split_holds_without_tokenizer(self):
        with patch.object(chunker, "get_encoder", return_value=None):
            parts = chunker.split_text_at_paragraphs("y" * 20000, 600)
        self.assertTrue(all(chunker.count_tokens(p) <= 600 for p in parts))


class SearchServiceTests(FastSearchCase):
    def setUp(self):
        super().setUp()
        self.add("s1", "/work/alpha", [("socket search alpha", unit(0))])
        self.add("s2", "/work/beta", [("socket search beta", unit(1))])
        self.embed = patch.object(search, "embed_query", return_value=unit(0))
        self.embed.start()
        self.addCleanup(self.embed.stop)
        self.server = search_service.SearchServer()
        self.server.start()
        self.addCleanup(self.server.stop)
        self.server.service.ready.set()

    def test_warming_daemon_refuses_searches_so_the_cli_falls_back(self):
        self.server.service.ready.clear()
        self.assertEqual(search_service.request("ping", {}, 1), {"ok": True, "ready": False})
        self.assertIsNone(search_service.request("search", {"query": "socket search"}, 1))

    def test_socket_is_private(self):
        mode = stat.S_IMODE(os.stat(search_service.socket_path()).st_mode)
        self.assertEqual(mode, 0o600)

    def test_daemon_results_match_in_process_search(self):
        reply = search_service.request("search", {"query": "socket search", "limit": 5}, 5)
        local = hybrid_search(db.get_read_connection(), "socket search", limit=5)
        self.assertEqual(reply["results"], local)
        filtered = search_service.request(
            "search", {"query": "socket search", "limit": 5, "project": "beta"}, 5)
        self.assertEqual([r["session_id"] for r in filtered["results"]], ["s2"])

    def test_cross_chat_returns_candidates_and_embedding(self):
        with patch("claude_chat_search.embedder.embed_query", return_value=unit(0)):
            reply = search_service.request("cross_chat", {"query": "socket", "limit": 2}, 5)
        self.assertEqual(reply["embedding"], unit(0))
        self.assertEqual(reply["results"][0]["session_id"], "s1")

    def test_bad_request_is_rejected(self):
        self.assertIsNone(search_service.request("search", {"query": ""}, 5))
        self.assertIsNone(search_service.request("drop", {}, 5))

    def test_cli_uses_daemon_and_falls_back_when_it_is_gone(self):
        def logged():
            lines = (self.root / "search.log").read_text().splitlines()
            return json.loads(lines[-1])

        with patch("os.getcwd", return_value="/work"):
            result = CliRunner().invoke(cli, ["search", "socket search"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("/work/alpha", result.output)
            self.assertTrue(logged().get("daemon"))

            self.server.stop()
            self.assertIsNone(search_service.request("ping", {}, 1))
            result = CliRunner().invoke(cli, ["search", "socket search"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("/work/alpha", result.output)
            self.assertNotIn("daemon", logged())

    def test_stale_socket_file_means_no_daemon(self):
        self.server.stop()
        path = search_service.socket_path()
        stale = socket.socket(socket.AF_UNIX)
        stale.bind(str(path))
        stale.close()
        self.assertIsNone(search_service.request("ping", {}, 1))


if __name__ == "__main__":
    unittest.main()

import hashlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from claude_chat_search import chunker, cli as cli_module, cross_search, db, embedder, models, search, vector_search
from claude_chat_search.cli import _run_index, cli
from claude_chat_search.codex_parser import iter_codex_jsonl_files
from claude_chat_search.search import hybrid_search

from test_codex_parser import SESSION_ID, session_meta, write_jsonl
from test_hardening import TempIndexCase

SMALL = models.ModelSpec("test/small-model", dim=8, max_tokens=64, query_prompt="q: ", document_prompt="d: ")


def use_model(name):
    return patch.dict(os.environ, {"CLAUDE_CHAT_SEARCH_MODEL": name})


def fake_vector(text: str, dim: int = SMALL.dim) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [digest[i] / 255 + 0.01 for i in range(dim)]


class ModelCase(TempIndexCase):
    def setUp(self):
        self.models = patch.dict(models.MODELS, {SMALL.name: SMALL})
        self.models.start()
        self.addCleanup(self.models.stop)
        self.legacy = use_model(models.LEGACY_MODEL)
        self.legacy.start()
        self.addCleanup(self.legacy.stop)
        super().setUp()
        vector_search.invalidate_cache()

    def add(self, sid, texts, vectors=True):
        db.insert_session(self.conn, {
            "session_id": sid, "native_session_id": sid, "source": "claude",
            "project_path": "/work/p", "message_count": 2,
        })
        ids = db.insert_chunks(self.conn, [{
            "session_id": sid, "user_content": t, "assistant_content": "",
            "combined_text": t, "turn_number": i,
        } for i, t in enumerate(texts)])
        if vectors:
            db.insert_embeddings(self.conn, ids, [[0.1] * db.embedding_dim() for _ in ids])
        return ids


class ModelRecordTests(ModelCase):
    def test_new_index_records_the_configured_model(self):
        self.assertEqual(db.stored_embedding_model(self.conn), models.LEGACY_MODEL)
        self.assertEqual(db.get_meta(self.conn, "embedding_dim"), "384")
        self.assertIsNone(db.embedding_mismatch(self.conn))

    def test_index_from_before_the_meta_table_is_the_legacy_model(self):
        self.add("s1", ["alpha"])
        self.conn.execute("DROP TABLE meta")
        self.conn.execute("PRAGMA user_version = 2")
        with use_model(SMALL.name):
            db.init_db(self.conn)
            self.assertEqual(db.stored_embedding_model(self.conn), models.LEGACY_MODEL)
            self.assertIn("migrate-embeddings", db.embedding_mismatch(self.conn))

    def test_unknown_model_is_rejected(self):
        with use_model("nobody/nothing"), self.assertRaises(ValueError):
            models.configured_model()


class MismatchTests(ModelCase):
    def setUp(self):
        super().setUp()
        self.add("s1", ["keyword alpha"])
        self.add("s2", ["other beta"])
        self.small = use_model(SMALL.name)
        self.small.start()
        self.addCleanup(self.small.stop)

    def test_search_falls_back_to_keywords_without_embedding_the_query(self):
        with patch.object(search, "embed_query", side_effect=AssertionError("no model")):
            results = hybrid_search(self.conn, "keyword alpha", limit=5)
        self.assertEqual([r["session_id"] for r in results], ["s1"])

    def test_cli_warns_that_semantic_search_is_off(self):
        with patch.object(search, "embed_query", side_effect=AssertionError("no model")), \
                patch("claude_chat_search.search_service.request", return_value=None), \
                patch("os.getcwd", return_value="/work"):
            result = CliRunner().invoke(cli, ["search", "keyword alpha"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("/work/p", result.output)
        self.assertIn("migrate-embeddings", result.stderr)

    def test_embedding_pass_writes_nothing(self):
        self.add("s3", ["pending"], vectors=False)
        with patch.object(embedder, "embed_texts", side_effect=AssertionError("no model")):
            self.assertEqual(embedder.process_embeddings(self.conn), 0)


class VectorCacheTests(ModelCase):
    def test_cache_is_rebuilt_when_the_model_changes(self):
        ids = self.add("s1", ["alpha"])
        self.assertEqual(
            vector_search.numpy_vector_search(self.conn, [0.1] * 384, limit=1)[0]["chunk_id"], ids[0])
        with use_model(SMALL.name):
            with db.write_transaction(self.conn):
                db.migrate_vec_table(self.conn)
            db.insert_embeddings(self.conn, ids, [fake_vector("alpha")])
            hits = vector_search.numpy_vector_search(self.conn, fake_vector("alpha"), limit=1)
        self.assertEqual(hits[0]["chunk_id"], ids[0])
        self.assertAlmostEqual(hits[0]["distance"], 0.0, places=5)

    def test_query_of_another_width_finds_nothing(self):
        self.add("s1", ["alpha"])
        self.assertEqual(vector_search.numpy_vector_search(self.conn, [0.1] * 8), [])


class ResplitTests(unittest.TestCase):
    def test_short_chunks_are_kept_and_long_ones_split(self):
        long_prompt = "prompt " * 400
        answer = "\n\n".join(f"paragraph {i} " + "word " * 120 for i in range(8))
        old = f"User: {long_prompt}\n\nAssistant: {answer}"
        rows = [
            {"session_id": "s", "user_content": "hi", "assistant_content": "there",
             "combined_text": "User: hi\n\nAssistant: there", "turn_number": 0},
            {"session_id": "s", "user_content": long_prompt, "assistant_content": answer,
             "combined_text": old, "turn_number": 1},
            {"session_id": "s", "user_content": long_prompt, "assistant_content": answer,
             "combined_text": old, "turn_number": 1},
        ]
        out = chunker.resplit_stored_chunks(rows)
        self.assertEqual(out[0]["combined_text"], "User: hi\n\nAssistant: there")
        self.assertTrue(all(chunker.count_tokens(c["combined_text"]) <= chunker.MAX_CHUNK_TOKENS for c in out))
        texts = [c["combined_text"] for c in out]
        self.assertEqual(len(texts), len(set(texts)))
        joined = " ".join(c["assistant_content"] for c in out)
        for i in range(8):
            self.assertIn(f"paragraph {i} ", joined)


class MigrateEmbeddingsTests(ModelCase):
    def setUp(self):
        super().setUp()
        self.rollout = self.root / "codex" / f"rollout-{SESSION_ID}.jsonl"
        write_jsonl(self.rollout, [session_meta()] + [
            {"timestamp": f"2026-08-12T09:00:0{i}Z", "type": "event_msg",
             "payload": {"type": kind, "message": f"{kind} {i} about lighthouses " + "word " * 200}}
            for i, kind in enumerate(["user_message", "agent_message"] * 2)
        ])
        self.files = iter_codex_jsonl_files([self.rollout.parent])
        with patch("claude_chat_search.sources.iter_codex_jsonl_files", return_value=self.files):
            _run_index(self.conn, source="codex")
        self.conn.execute("UPDATE chunks SET embedded = 0")
        rows = db.get_unembedded_chunks(self.conn, 100)
        db.insert_embeddings(self.conn, [r["id"] for r in rows], [[0.1] * 384 for _ in rows])
        long_answer = "\n\n".join(f"gone paragraph {i} " + "word " * 150 for i in range(6))
        self.add("gone", [f"User: where is it\n\nAssistant: {long_answer}"])
        self.conn.execute(
            "UPDATE chunks SET user_content = 'where is it', assistant_content = ? "
            "WHERE session_id = 'gone'", (long_answer,))
        self.embedded_texts = []

    def fake_embed(self, texts, show_progress=False):
        self.embedded_texts += texts
        return [fake_vector(t) for t in texts]

    def migrate(self, *args):
        with use_model(SMALL.name), \
                patch.object(cli_module, "iter_conversation_files", return_value=self.files), \
                patch.object(embedder, "embed_texts", side_effect=self.fake_embed):
            return CliRunner().invoke(cli, ["migrate-embeddings", *args])

    def test_migration_rechunks_reembeds_and_switches_model(self):
        result = self.migrate()
        self.assertEqual(result.exit_code, 0, result.output + result.stderr)
        self.assertTrue(list((self.root / "backups").glob("index-*.db")))
        with use_model(SMALL.name):
            self.assertIsNone(db.embedding_mismatch(self.conn))
        self.assertIsNone(db.get_meta(self.conn, "migration_state"))
        self.assertEqual(db.get_meta(self.conn, "embedding_dim"), "8")
        stats = db.get_stats(self.conn)
        self.assertEqual(stats["embedded"], stats["chunks"])
        gone = [r[0] for r in self.conn.execute(
            "SELECT combined_text FROM chunks WHERE session_id = 'gone'")]
        self.assertGreater(len(gone), 1)
        self.assertIn("gone paragraph 5", " ".join(gone))
        widths = {len(r[0]) for r in self.conn.execute("SELECT embedding FROM vec_chunks")}
        self.assertEqual(widths, {SMALL.dim * 4})

        with use_model(SMALL.name), patch.object(search, "embed_query", return_value=fake_vector(gone[0])):
            results = hybrid_search(self.conn, "zzz", limit=1)
        self.assertEqual(results[0]["session_id"], "gone")

    def test_interrupted_migration_resumes(self):
        real_sync = db.sync_session_chunks
        calls = []

        def flaky(conn, sid, chunks):
            calls.append(sid)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return real_sync(conn, sid, chunks)

        with patch.object(cli_module, "sync_session_chunks", side_effect=flaky):
            result = self.migrate()
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(db.get_meta(self.conn, "migration_state"), "rechunk")
        self.assertEqual(db.get_meta(self.conn, "migration_cursor"), calls[0])
        self.assertEqual(db.stored_embedding_model(self.conn), models.LEGACY_MODEL)

        result = self.migrate()
        self.assertEqual(result.exit_code, 0, result.output + result.stderr)
        self.assertEqual(len(list((self.root / "backups").glob("index-*.db"))), 1)
        self.assertEqual(db.stored_embedding_model(self.conn), SMALL.name)
        self.assertEqual(db.get_stats(self.conn)["embedded"], db.get_stats(self.conn)["chunks"])

    def test_nothing_to_do_when_already_migrated(self):
        self.assertEqual(self.migrate().exit_code, 0)
        result = self.migrate()
        self.assertEqual(result.exit_code, 0)
        self.assertIn("nothing to migrate", result.output)

    def test_documents_and_queries_get_their_prompts(self):
        model = FakeST()
        with patch.object(embedder, "_get_model", return_value=(model, SMALL)):
            embedder.embed_texts(["chunk b", "chunk"])
            embedder.embed_query("question")
        self.assertEqual(sorted(model.seen[:2]), ["d: chunk", "d: chunk b"])
        self.assertEqual(model.seen[2], "q: question")


class FakeST:
    def __init__(self):
        self.seen = []

    def encode(self, texts, **_kwargs):
        import numpy as np
        self.seen += texts
        return np.array([fake_vector(t) for t in texts], dtype=np.float32)


class CrossSearchTests(ModelCase):
    def test_research_half_embeds_the_query_with_its_own_model(self):
        seen = {}

        def fake_run(args, input, **_kwargs):
            seen["script"], seen["input"] = args[-1], input

            class Done:
                returncode, stdout, stderr = 0, "[]", ""
            return Done()

        with patch.object(cross_search, "RESEARCH_INDEX_DB", Path(__file__)), \
                patch("subprocess.run", side_effect=fake_run):
            cross_search._search_research_index("fjord ferries", limit=3)
        self.assertEqual(seen["input"], "fjord ferries")
        self.assertIn("get_embedding_model", seen["script"])
        self.assertIn("paraphrase-multilingual-MiniLM-L12-v2", seen["script"])


if __name__ == "__main__":
    unittest.main()

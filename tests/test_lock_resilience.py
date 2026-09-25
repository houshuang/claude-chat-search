import logging
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import apsw

from claude_chat_search import db, embedder


class StoreRetryTest(unittest.TestCase):
    def test_busy_insert_is_retried_not_dropped(self):
        calls = []

        def flaky(conn, ids, embeddings):
            calls.append(ids)
            if len(calls) == 1:
                raise apsw.BusyError("database is locked")

        with patch.object(embedder, "insert_embeddings", side_effect=flaky), \
                patch.object(embedder.time, "sleep"):
            embedder._store_with_retry(None, [1, 2], [[0.0], [0.0]])
        self.assertEqual(calls, [[1, 2], [1, 2]])

    def test_gives_up_after_the_last_attempt(self):
        with patch.object(embedder, "insert_embeddings", side_effect=apsw.BusyError("locked")), \
                patch.object(embedder.time, "sleep"):
            with self.assertRaises(apsw.BusyError):
                embedder._store_with_retry(None, [1], [[0.0]])


class SlowTransactionLogTest(unittest.TestCase):
    def test_long_held_transaction_is_logged_with_its_caller(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        conn = apsw.Connection(path)
        conn.execute("CREATE TABLE t (v)")

        def rebuild_everything():
            with db.write_transaction(conn):
                conn.execute("INSERT INTO t VALUES (1)")
                time.sleep(0.05)

        with patch.object(db, "SLOW_LOCK_HOLD_S", 0.01), \
                self.assertLogs("claude_chat_search.db", logging.WARNING) as logs:
            rebuild_everything()
        self.assertIn("rebuild_everything", logs.output[0])


class BackgroundEmbeddingYieldsTest(unittest.TestCase):
    def test_background_batches_wait_while_a_query_is_waiting(self):
        class Model:
            def __init__(self):
                self.calls = []

            def encode(self, batch, **kwargs):
                import numpy as np
                self.calls.append(batch[0][:1])
                return np.zeros((len(batch), 2), dtype="float32")

        model = Model()
        embedder._queries_waiting += 1
        done = threading.Event()

        def background():
            embedder._encode(model, ["d"] * 16, background=True)
            done.set()

        thread = threading.Thread(target=background)
        thread.start()
        time.sleep(0.05)
        self.assertEqual(model.calls, [])
        embedder._queries_waiting -= 1
        self.assertTrue(done.wait(2))
        thread.join()
        self.assertEqual(len(model.calls), 16 // embedder.ENCODE_BATCH)


if __name__ == "__main__":
    unittest.main()

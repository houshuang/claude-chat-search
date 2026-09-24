import threading
import time
import unittest

import apsw

from claude_chat_search.db import write_transaction


class WriteTransactionTest(unittest.TestCase):
    def test_read_then_write_waits_for_concurrent_writer(self):
        import tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        setup = apsw.Connection(path)
        setup.execute("PRAGMA journal_mode=WAL")
        setup.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        setup.execute("INSERT INTO t (v) VALUES ('a')")

        other = apsw.Connection(path)
        other.setbusytimeout(5000)
        mine = apsw.Connection(path)
        mine.setbusytimeout(5000)

        other.execute("BEGIN IMMEDIATE")
        other.execute("INSERT INTO t (v) VALUES ('b')")

        def release():
            time.sleep(0.3)
            other.execute("COMMIT")

        threading.Thread(target=release).start()
        with write_transaction(mine):
            ids = [r[0] for r in mine.execute("SELECT id FROM t")]
            for i in ids:
                mine.execute("DELETE FROM t WHERE id = ?", (i,))
        self.assertEqual(mine.execute("SELECT count(*) FROM t").fetchone()[0], 0)

    def test_rollback_on_error(self):
        conn = apsw.Connection(":memory:")
        conn.execute("CREATE TABLE t (v)")
        with self.assertRaises(RuntimeError):
            with write_transaction(conn):
                conn.execute("INSERT INTO t VALUES (1)")
                raise RuntimeError
        self.assertEqual(conn.execute("SELECT count(*) FROM t").fetchone()[0], 0)
        self.assertFalse(conn.in_transaction)

    def test_nested_uses_savepoint(self):
        conn = apsw.Connection(":memory:")
        conn.execute("CREATE TABLE t (v)")
        with write_transaction(conn):
            conn.execute("INSERT INTO t VALUES (1)")
            with write_transaction(conn):
                conn.execute("INSERT INTO t VALUES (2)")
        self.assertEqual(conn.execute("SELECT count(*) FROM t").fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()

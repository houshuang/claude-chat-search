import os
import tempfile
import threading
import time
import unittest

import apsw

from claude_chat_search.daemon import wal_checkpoint


class PeriodicCheckpointTest(unittest.TestCase):
    def test_passive_checkpoint_does_not_block_writers_behind_a_reader(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        setup = apsw.Connection(path)
        setup.execute("PRAGMA journal_mode=WAL")
        setup.execute("CREATE TABLE t (v)")
        for i in range(200):
            setup.execute("INSERT INTO t VALUES (?)", (i,))

        def connect():
            conn = apsw.Connection(path)
            conn.setbusytimeout(3000)
            return conn

        reader, daemon, writer = connect(), connect(), connect()
        cursor = reader.execute("SELECT v FROM t")
        next(cursor)
        setup.execute("INSERT INTO t VALUES (-1)")

        thread = threading.Thread(target=wal_checkpoint, args=(daemon,))
        thread.start()
        time.sleep(0.1)
        started = time.monotonic()
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("COMMIT")
        waited = time.monotonic() - started
        thread.join()
        self.assertLess(waited, 1.0)


if __name__ == "__main__":
    unittest.main()

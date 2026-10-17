import tempfile
import unittest
from pathlib import Path

import tests.vessel  # noqa: F401  (sys.path)

import numpy as np

from kgirl.vessel.matrixdb import GENESIS, MatrixDB


class MatrixDBTests(unittest.TestCase):
    def setUp(self):
        self.db = MatrixDB()
        self.db.create_entity("ada", {"seed": 0.5})

    def tearDown(self):
        self.db.close()

    def test_content_addressing_dedups_and_roundtrips(self):
        M = np.arange(12, dtype=np.float32).reshape(3, 4)
        h1, h2 = self.db.put_matrix(M), self.db.put_matrix(M.copy())
        self.assertEqual(h1, h2)
        self.assertEqual(self.db.storage()["matrices"], 1)
        out = self.db.get_matrix(h1)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out, M)

    def test_corrupted_blob_is_detected(self):
        import zlib
        h = self.db.put_matrix(np.ones(3))
        self.db._db.execute("UPDATE matrices SET blob=? WHERE hash=?", (zlib.compress(b"tampered"), h))
        with self.assertRaises(ValueError):
            self.db.get_matrix(h)

    def test_geometries_latest_and_history(self):
        self.db.record("ada", "gram", np.eye(2), layer="h1", label="audit", meta={"drift": 0.1})
        self.db.record("ada", "gram", 2 * np.eye(2), layer="h1", label="audit", meta={"drift": 0.2})
        g = self.db.latest("ada", "gram", layer="h1", label="audit")
        self.assertEqual(g.meta["drift"], 0.2)
        np.testing.assert_array_equal(self.db.load(g), 2 * np.eye(2))
        self.assertEqual([x.meta["drift"] for x in self.db.history("ada", "gram", "h1")], [0.1, 0.2])
        self.assertIsNone(self.db.latest("ada", "gram", layer="h2"))

    def test_ledger_chain_and_tamper_evidence(self):
        e0 = self.db.append("ada", "birth", {"x": 1})
        self.assertEqual(e0.prev, GENESIS)
        for i in range(4):
            self.db.append("ada", "observe", {"drift": i / 10})
        ok = self.db.verify("ada")
        self.assertTrue(ok["ok"])
        self.assertEqual((ok["length"], ok["head"]), (5, self.db.head("ada")))
        self.db._db.execute("UPDATE ledger SET payload=? WHERE seq=2", ('{"drift":0.0}',))
        bad = self.db.verify("ada")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["broken_at"], 2)

    def test_deleting_a_ledger_row_breaks_the_chain(self):
        for i in range(3):
            self.db.append("ada", "observe", {"i": i})
        self.db._db.execute("DELETE FROM ledger WHERE seq=1")
        self.assertEqual(self.db.verify("ada")["broken_at"], 1)

    def test_snapshot_restore_and_labels(self):
        params = {"W": np.arange(6.0).reshape(2, 3), "b": np.zeros(3)}
        self.db.snapshot("ada", params, "anchor:0")
        self.db.snapshot("ada", {"W": params["W"] + 1, "b": params["b"]}, "steer:1")
        self.assertEqual(self.db.labels("ada"), ["anchor:0", "steer:1"])
        back = self.db.restore("ada", "anchor:0")
        np.testing.assert_array_equal(back["W"], params["W"])
        self.assertEqual(self.db.storage()["matrices"], 3)          # b is stored once
        with self.assertRaises(KeyError):
            self.db.restore("ada", "nope")

    def test_persists_across_connections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v" / "m.db"
            with MatrixDB(path) as db:
                db.create_entity("bix")
                db.append("bix", "birth", {})
                head = db.head("bix")
            with MatrixDB(path) as db:
                self.assertEqual(db.entities(), ["bix"])
                self.assertEqual(db.head("bix"), head)
                self.assertTrue(db.verify("bix")["ok"])

    def test_unknown_entity_and_duplicate_names(self):
        with self.assertRaises(KeyError):
            self.db.append("nobody", "x")
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.create_entity("ada")


if __name__ == "__main__":
    unittest.main()

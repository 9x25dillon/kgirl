import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from kgirl.harness.assistant import Assistant
from kgirl.harness.atlas.chunks import build as build_chunks
from kgirl.harness.atlas.retrieval import (Pipe, RemoteError, Retriever, SqlPipe, VectorPipe, reciprocal_rank_fusion,
                                           sql_literal)
from kgirl.harness.atlas.vectors import Matrix, build as build_matrix, to_bits, write_matrix

ROOT = Path(__file__).resolve().parents[2]
DAEMON_SRC = ROOT / "src/kgirl/harness/atlas/router/kgvecd.c"


def unit(rng, n, dim):
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class MatrixTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "m.kgvm"

    def test_round_trip_finds_each_vector_as_its_own_best_match(self):
        vecs = unit(np.random.default_rng(1), 200, 64)
        ids = list(range(1000, 1200))
        stats = write_matrix(self.path, ids, vecs, "demo")
        self.assertEqual((stats["count"], stats["dim"]), (200, 64))
        m = Matrix(self.path)
        self.assertEqual(m.model, "demo")
        for i in (0, 37, 199):
            bits, vals = m.encode_query(vecs[i])
            self.assertEqual(m.search(bits, vals, k=1)[0][0], ids[i])

    def test_rejects_a_truncated_file_and_a_wrong_query_size(self):
        write_matrix(self.path, [1, 2], unit(np.random.default_rng(2), 2, 16), "demo")
        m = Matrix(self.path)
        with self.assertRaises(ValueError):
            m.encode_query(np.ones(8))
        short = self.dir / "short.kgvm"
        short.write_bytes(self.path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            Matrix(short)

    def test_dimension_must_be_a_multiple_of_8(self):
        with self.assertRaises(ValueError):
            write_matrix(self.path, [1], np.ones((1, 12)), "demo")

    def test_bits_are_little_endian_within_a_byte(self):
        self.assertEqual(to_bits(np.array([1, -1, -1, -1, -1, -1, -1, 1.0])).tolist(), [0b10000001])

    def test_build_failure_leaves_the_old_matrix(self):
        write_matrix(self.path, [1], unit(np.random.default_rng(3), 1, 8), "old")
        before = self.path.read_bytes()
        db = self.dir / "chunks.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TABLE chunks (id INTEGER PRIMARY KEY, path TEXT, symbol TEXT, text TEXT);"
            "INSERT INTO chunks VALUES (1,'a','f','x');")
        conn.close()

        def broken(_):
            raise OSError("no embedder")
        with self.assertRaises(OSError):
            build_matrix(db, self.path, broken)
        self.assertEqual(self.path.read_bytes(), before)


class FusionTests(unittest.TestCase):
    def test_item_in_both_lists_beats_items_in_one(self):
        fused = reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]])
        self.assertEqual(fused[0][0], 2)

    def test_ties_are_deterministic_and_an_empty_list_changes_nothing(self):
        self.assertEqual([i for i, _ in reciprocal_rank_fusion([[9], [3]])], [3, 9])
        self.assertEqual([i for i, _ in reciprocal_rank_fusion([[7, 8], []])], [7, 8])

    def test_sql_literal_doubles_quotes(self):
        self.assertEqual(sql_literal("a'b"), "'a''b'")


@unittest.skipUnless(shutil.which("clang"), "needs clang to build the daemon")
class DaemonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.bin = cls.dir / "kgvecd"
        subprocess.run(["clang", "-O2", str(DAEMON_SRC), "-o", str(cls.bin)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def matrix(self, n=500, dim=64, seed=4):
        path = self.dir / f"m{n}_{dim}_{seed}.kgvm"
        write_matrix(path, list(range(5, 5 + n)), unit(np.random.default_rng(seed), n, dim), "demo")
        return path

    def test_daemon_ranks_exactly_as_python_does(self):
        path = self.matrix()
        m = Matrix(path)
        client = VectorPipe([str(self.bin), str(path)])
        self.addCleanup(client.close)
        self.assertEqual((client.count, client.dim), (500, 64))
        for q in unit(np.random.default_rng(5), 25, 64):
            bits, vals = m.encode_query(q)
            for k, shortlist in ((10, 50), (1, 1), (40, 500), (5, 2)):
                self.assertEqual(client.search(bits, vals, k, shortlist, 2.0), m.search(bits, vals, k, shortlist))

    def test_ties_rank_the_same_on_both_sides(self):
        path = self.dir / "dup.kgvm"
        write_matrix(path, list(range(30)), np.tile(unit(np.random.default_rng(6), 3, 16), (10, 1)), "demo")
        m = Matrix(path)
        client = VectorPipe([str(self.bin), str(path)])
        self.addCleanup(client.close)
        bits, vals = m.encode_query(unit(np.random.default_rng(7), 1, 16)[0])
        self.assertEqual(client.search(bits, vals, 12, 15, 2.0), m.search(bits, vals, 12, 15))

    def test_k_larger_than_the_matrix_returns_every_vector(self):
        path = self.matrix(n=3, dim=16, seed=8)
        m = Matrix(path)
        client = VectorPipe([str(self.bin), str(path)])
        self.addCleanup(client.close)
        bits, vals = m.encode_query(unit(np.random.default_rng(9), 1, 16)[0])
        self.assertEqual(len(client.search(bits, vals, 50, 400, 2.0)), 3)

    def test_daemon_refuses_a_corrupt_matrix(self):
        bad = self.dir / "bad.kgvm"
        bad.write_bytes(self.matrix().read_bytes()[:-1])
        with self.assertRaises(RemoteError):
            VectorPipe([str(self.bin), str(bad)], connect_timeout=2.0)


FAKE_DIM = 32


def fake_embed(texts):
    """Deterministic bag of hashed words, so related text lands nearby without a model."""
    out = np.zeros((len(texts), FAKE_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().replace(":", " ").split():
            rng = np.random.default_rng(abs(hash(w)) % (2 ** 32))
            out[i] += rng.standard_normal(FAKE_DIM).astype(np.float32)
    return out


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        repo = self.dir / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("def parseHTTPResponse(raw):\n    return decode_gzip_body(raw)\n")
        (repo / "b.md").write_text("# Router notes\n\nThe zebra tier is optional.\n")
        self.chunks = self.dir / "chunks.db"
        build_chunks(repo, self.chunks, "demo")
        self.matrix = self.dir / "chunks.kgvm"
        build_matrix(self.chunks, self.matrix, fake_embed, model="fake")

    def retriever(self, **kw):
        r = Retriever(self.chunks, self.matrix, embed=fake_embed, **kw)
        self.addCleanup(r.close)
        return r

    def test_local_search_cites_the_keyword_hit(self):
        r = self.retriever()
        hits = r.search("zebra tier")
        self.assertEqual(hits[0].cite(), "demo:b.md:1")
        self.assertEqual((r.info.keyword_backend, r.info.vector_backend), ("local", "local"))

    def test_keyword_search_survives_a_dead_embedder(self):
        def dead(_):
            raise OSError("ollama is down")
        r = Retriever(self.chunks, self.matrix, embed=dead)
        self.addCleanup(r.close)
        self.assertEqual(r.search("gzip")[0].path, "a.py")
        self.assertTrue(r.info.vector_backend.startswith("unavailable"))

    def test_missing_remote_command_falls_back_and_waits_before_retrying(self):
        r = self.retriever(vector_cmd=["/nonexistent/kgvecd"], sql_cmd=["/nonexistent/sqlite3"], retry_after=60)
        self.assertEqual(r.search("zebra")[0].path, "b.md")
        self.assertEqual((r.info.keyword_backend, r.info.vector_backend), ("local", "local"))
        self.assertGreater(r._next_try["vec"], time.monotonic())

    def test_a_hung_remote_costs_at_most_the_deadline(self):
        hang = [sys.executable, "-c", "import time; time.sleep(30)"]
        r = self.retriever(vector_cmd=hang, sql_cmd=hang, retry_after=60)
        import kgirl.harness.atlas.retrieval as mod
        real_vec, real_sql = mod.VectorPipe, mod.SqlPipe
        mod.VectorPipe = lambda argv: real_vec(argv, connect_timeout=0.2)
        mod.SqlPipe = lambda argv: real_sql(argv, connect_timeout=0.2)
        self.addCleanup(setattr, mod, "VectorPipe", real_vec)
        self.addCleanup(setattr, mod, "SqlPipe", real_sql)
        t = time.monotonic()
        self.assertEqual(r.search("zebra")[0].path, "b.md")
        self.assertLess(time.monotonic() - t, 2.0)
        t = time.monotonic()
        r.search("zebra")  # still out of favor: no second wait
        self.assertLess(time.monotonic() - t, 0.15)

    def test_pipe_read_raises_at_the_deadline(self):
        p = Pipe([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(p.close)
        with self.assertRaises(RemoteError):
            p.read_exact(1, time.monotonic() + 0.05)

    @unittest.skipUnless(shutil.which("sqlite3"), "needs the sqlite3 shell")
    def test_sql_pipe_matches_local_results_and_survives_quotes(self):
        pipe = SqlPipe(["sqlite3", "-batch", "-readonly", "-json", str(self.chunks)])
        self.addCleanup(pipe.close)
        self.assertEqual(pipe.query("SELECT path FROM chunks WHERE id = 99999", 2.0), [])
        got = pipe.query(f"SELECT {sql_literal(chr(39) + 'x' + chr(10) + 'y')} AS v", 2.0)
        self.assertEqual(got, [{"v": "'x\ny"}])
        r = self.retriever(sql_cmd=["sqlite3", "-batch", "-readonly", "-json", str(self.chunks)])
        hits = r.search("gzip")
        self.assertEqual(r.info.keyword_backend, "router")
        self.assertEqual(hits[0].path, "a.py")


class AssistantIntegrationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        repo = self.dir / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("def parseHTTPResponse(raw):\n    return decode_gzip_body(raw)\n")
        self.repo = repo

    def make(self, env):
        with mock.patch.dict(os.environ, env):
            a = Assistant(home=self.dir / "home", trace=False)
        self.addCleanup(a.close)
        a.index([str(self.repo)])
        return a

    def test_without_the_setting_the_context_pack_does_not_use_chunks(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("KGIRL_CHUNKS_DB", None)
            a = Assistant(home=self.dir / "home", trace=False)
        self.addCleanup(a.close)
        self.assertIsNone(a.retriever)

    def test_with_the_setting_the_context_pack_leads_with_retrieved_chunks(self):
        chunks, matrix = self.dir / "c.db", self.dir / "c.kgvm"
        build_chunks(self.repo, chunks, "demo")
        build_matrix(chunks, matrix, fake_embed, model="fake")
        a = self.make({"KGIRL_CHUNKS_DB": str(chunks), "KGIRL_MATRIX": str(matrix)})
        a.retriever.embed = fake_embed
        context, cites = a.context_pack("gzip body")
        self.assertTrue(context.startswith("### demo:a.py:1"))
        self.assertIn("decode_gzip_body", context)
        self.assertIn("demo:a.py:1", cites)

    def test_chunks_from_another_repo_are_left_out(self):
        chunks, matrix = self.dir / "c.db", self.dir / "c.kgvm"
        build_chunks(self.repo, chunks, "demo")
        build_matrix(chunks, matrix, fake_embed, model="fake")
        a = self.make({"KGIRL_CHUNKS_DB": str(chunks), "KGIRL_MATRIX": str(matrix)})
        a.retriever.embed = fake_embed
        _, cites = a.context_pack("gzip body", repo="other")
        self.assertNotIn("demo:a.py:1", cites)


class EmbedCacheTests(unittest.TestCase):
    def test_rebuild_embeds_only_changed_chunks_and_matches_a_full_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            repo = d / "repo"
            repo.mkdir()
            (repo / "a.py").write_text("def alpha():\n    return 1\n")
            (repo / "b.py").write_text("def beta():\n    return 2\n")
            chunks, cache = d / "c.db", d / "cache.db"
            seen: list[str] = []

            def counting(texts):
                seen.extend(texts)
                return fake_embed(texts)

            build_chunks(repo, chunks, "demo")
            first = build_matrix(chunks, d / "1.kgvm", counting, model="fake", cache=cache)
            self.assertEqual(first["embedded"], len(seen))
            (repo / "b.py").write_text("def beta():\n    return 3\n")
            chunks.unlink()
            build_chunks(repo, chunks, "demo")
            seen.clear()
            again = build_matrix(chunks, d / "2.kgvm", counting, model="fake", cache=cache)
            self.assertEqual(again["embedded"], 1)
            self.assertIn("return 3", seen[0])
            build_matrix(chunks, d / "3.kgvm", fake_embed, model="fake")
            self.assertEqual((d / "2.kgvm").read_bytes(), (d / "3.kgvm").read_bytes())
            seen.clear()
            build_matrix(chunks, d / "4.kgvm", counting, model="other", cache=cache)
            self.assertEqual(len(seen), first["embedded"], "a different model must not reuse cached vectors")


if __name__ == "__main__":
    unittest.main()

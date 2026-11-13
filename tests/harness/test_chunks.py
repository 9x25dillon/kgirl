import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from kgirl.harness.atlas.chunks import MAX_LINES, OVERLAP, build, build_committed, chunk_file, search
from kgirl.harness.atlas.parse import extract


def chunks_of(text, path="m.py", lang="python"):
    return chunk_file(text, extract(path, text, lang))


class ChunkFileTests(unittest.TestCase):
    def test_every_line_is_covered_and_cites_its_symbol(self):
        text = "import os\n\ndef f():\n    return 1\n\nclass A:\n    def g(self):\n        return 2\n"
        got = chunks_of(text)
        covered = {n for c in got for n in range(c.start_line, c.end_line + 1)}
        self.assertEqual(covered, set(range(1, 9)))
        self.assertEqual([(c.kind, c.symbol, c.start_line) for c in got],
                         [("preamble", "", 1), ("function", "f", 3), ("class", "A", 6), ("method", "A.g", 7)])

    def test_text_matches_its_cited_lines(self):
        text = "x = 1\n\ndef f():\n    return 1\n"
        lines = text.split("\n")
        for c in chunks_of(text):
            self.assertEqual(c.text, "\n".join(lines[c.start_line - 1:c.end_line]))

    def test_long_span_splits_with_overlap(self):
        n = MAX_LINES * 2 + 5
        text = "def big():\n" + "".join(f"    v{i} = {i}\n" for i in range(n))
        got = chunks_of(text)
        self.assertGreater(len(got), 1)
        self.assertTrue(all(c.symbol == "big" for c in got))
        self.assertTrue(all(c.end_line - c.start_line < MAX_LINES for c in got))
        self.assertEqual(got[1].start_line, got[0].end_line + 1 - OVERLAP)
        self.assertEqual(got[-1].end_line, n + 1)

    def test_blank_and_empty_files_make_no_chunks(self):
        self.assertEqual(chunks_of(""), [])
        self.assertEqual(chunks_of("\n\n\n"), [])


class BuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "repo"
        self.root.mkdir()
        (self.root / "a.py").write_text("def parseHTTPResponse(raw):\n    return decode_gzip_body(raw)\n")
        (self.root / "b.md").write_text("# Router notes\n\nThe zebra tier is optional.\n")
        (self.root / "skip.bin").write_bytes(b"\x00\x01")
        self.db = Path(tmp.name) / "chunks.db"

    def conn(self):
        db = sqlite3.connect(self.db)
        self.addCleanup(db.close)
        return db

    def test_search_finds_body_text_and_split_identifiers(self):
        stats = build(self.root, self.db, "demo")
        self.assertEqual(stats["files"], 2)
        db = self.conn()
        cite = lambda q: [(r[0], r[1], r[2]) for r in search(db, q)]
        self.assertEqual(cite("gzip"), [("demo", "a.py", 1)])
        self.assertEqual(cite("http response"), [("demo", "a.py", 1)])
        self.assertEqual(cite("zebra"), [("demo", "b.md", 1)])
        self.assertEqual(search(db, "nothingmatchesthis"), [])

    def test_rebuild_replaces_old_contents(self):
        build(self.root, self.db)
        (self.root / "a.py").unlink()
        build(self.root, self.db)
        db = self.conn()
        self.assertEqual(search(db, "gzip"), [])
        self.assertFalse(self.db.with_name(self.db.name + ".tmp").exists())

    def test_working_tree_build_drops_files_holding_secrets(self):
        (self.root / "leak.py").write_text('TOKEN = "' + "gh" + "p_" + "Q" * 36 + '"\n')
        stats = build(self.root, self.db, "demo")
        self.assertEqual(stats["filtered"], ["demo:leak.py (GitHub token)"])
        self.assertEqual(search(self.conn(), "TOKEN"), [])


def leaky_line() -> str:
    return 'API_TOKEN = "' + "gh" + "p_" + "Q" * 36 + '"\n'  # assembled here so no token sits in this file


class CommittedBuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.repo = self.tmp / "proj"
        self.repo.mkdir()
        self.git("init", "-q")
        (self.repo / "app.py").write_text("def committed_version():\n    return 1\n")
        (self.repo / "leak.py").write_text(leaky_line())
        self.git("add", "app.py", "leak.py")
        self.git("commit", "-qm", "init")
        (self.repo / "app.py").write_text("def uncommitted_edit():\n    return 2\n")
        (self.repo / "scratch.py").write_text("def untracked_helper():\n    return 3\n")

    def git(self, *args: str) -> str:
        out = subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                             check=True, capture_output=True, text=True)
        return out.stdout.strip()

    def test_only_committed_text_is_indexed_and_secrets_are_dropped(self):
        out = self.tmp / "all.db"
        stats = build_committed([("proj", self.repo)], out)
        db = sqlite3.connect(out)
        self.addCleanup(db.close)
        rows = db.execute("SELECT repo, path, symbol FROM chunks").fetchall()
        self.assertEqual({(r, p) for r, p, _ in rows}, {("proj", "app.py")})
        self.assertIn("committed_version", {s for _, _, s in rows})
        self.assertEqual(stats["filtered"], ["proj:leak.py (GitHub token)"])
        meta = dict(db.execute("SELECT key, value FROM meta"))
        self.assertEqual(meta["repo"], "all")
        self.assertEqual(json.loads(meta["head"]), {"proj": self.git("rev-parse", "HEAD")})
        self.assertEqual(json.loads(meta["filtered"]), ["proj:leak.py (GitHub token)"])

    def test_a_directory_without_commits_is_refused(self):
        plain = self.tmp / "plain"
        plain.mkdir()
        with self.assertRaises(ValueError):
            build_committed([("proj", self.repo), ("plain", plain)], self.tmp / "x.db")

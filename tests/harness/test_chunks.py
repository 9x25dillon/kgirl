import sqlite3
import tempfile
import unittest
from pathlib import Path

from kgirl.harness.atlas.chunks import MAX_LINES, OVERLAP, build, chunk_file, search
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

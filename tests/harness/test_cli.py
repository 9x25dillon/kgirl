import contextlib
import hashlib
import io
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kgirl.harness.cli import main


class SoupCliTests(unittest.TestCase):
    def soup(self, home, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(main(["--home", str(home), "soup", *args]), 0)
        return buf.getvalue()

    def test_recall_prints_what_add_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            added = self.soup(home, "add", "prefer pathlib over os.path for file handling", "--kind", "preference")
            self.assertRegex(added, r"^#\d+")
            got = self.soup(home, "recall", "pathlib file handling")
            self.assertIn("prefer pathlib over os.path", got)
            self.assertRegex(got, r"\[\d+/600 tokens\]")


def fake_embedder(model: str, cpu: bool = True):
    import numpy as np

    def embed(texts):
        return np.stack([np.frombuffer(hashlib.sha256(t.encode()).digest()[:16], dtype=np.uint8).astype(np.float32)
                         - 127.5 for t in texts])
    return embed


class ChunksCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--home", str(self.home), *args])
        return code, buf.getvalue()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.home = base / "home"
        repo = base / "proj"
        repo.mkdir()
        git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git, "init", "-q"], check=True)
        (repo / "app.py").write_text("def handler():\n    return 1\n\n\ndef helper():\n    return 2\n")
        subprocess.run([*git, "add", "app.py"], check=True)
        subprocess.run([*git, "commit", "-qm", "init"], check=True)
        plain = base / "notes"
        plain.mkdir()
        (plain / "n.py").write_text("def note():\n    return 3\n")
        self.assertEqual(self.run_cli("index", str(repo), str(plain))[0], 0)

    def test_builds_index_and_matrix_from_indexed_git_repos(self):
        from kgirl.harness.atlas.vectors import Matrix
        with mock.patch("kgirl.harness.atlas.vectors.ollama_embedder", fake_embedder):
            code, out = self.run_cli("chunks")
        self.assertEqual(code, 0, out)
        self.assertIn("skipped notes: not a git checkout", out)
        folder = self.home / "chunks" / "all"
        db = sqlite3.connect(folder / "all-repos.db")
        self.addCleanup(db.close)
        self.assertEqual({r for (r,) in db.execute("SELECT DISTINCT repo FROM chunks")}, {"proj"})
        count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        self.assertEqual(Matrix(folder / "all-repos.kgvm").count, count)
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["all-repos.db", "all-repos.kgvm"])

    def test_unknown_exclude_is_refused_before_anything_is_built(self):
        code, _ = self.run_cli("chunks", "--exclude", "prj")
        self.assertEqual(code, 2)
        self.assertFalse((self.home / "chunks" / "all").exists())


if __name__ == "__main__":
    unittest.main()

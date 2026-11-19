#!/usr/bin/env python3
"""Offline tests for merge_preview.py in throwaway repos: a base commit and lane worktrees from it.

Nothing is fetched or pushed, and no real kgirl worktree is read. Each case builds a tiny repository whose
tests/harness folder holds one trivial test module, so discovery has something real to find.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIEW = HERE / "merge_preview.py"
IDENT = {"GIT_AUTHOR_NAME": "kgirl-test", "GIT_AUTHOR_EMAIL": "1+kgirl-test@users.noreply.github.com",
         "GIT_COMMITTER_NAME": "kgirl-test", "GIT_COMMITTER_EMAIL": "1+kgirl-test@users.noreply.github.com"}
TEST_MODULE = "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        pass\n"


class Repo:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.env = {**os.environ, "HOME": str(tmp / "home"), "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_NOSYSTEM": "1", **IDENT}
        (tmp / "home").mkdir()
        self.base = tmp / "base"
        self.git("init", "-q", "-b", "main", str(self.base), cwd=tmp)
        for rel, text in {"src/pkg/__init__.py": "", "src/pkg/core.py": "VALUE = 1\n",
                          "tests/__init__.py": "", "tests/harness/__init__.py": "",
                          "tests/harness/test_core.py": TEST_MODULE}.items():
            self.write(self.base, rel, text)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "base")

    def git(self, *args, cwd=None):
        r = subprocess.run(["git", *args], cwd=cwd or self.base, env=self.env, capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(f"git {' '.join(args)}: {r.stderr}")
        return r.stdout

    @staticmethod
    def write(root: Path, rel: str, text: str):
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)

    def lane(self, name: str) -> Path:
        path = self.tmp / name
        self.git("worktree", "add", "-q", "-b", f"lane/{name}", str(path))
        return path

    def preview(self, *lanes: Path) -> subprocess.CompletedProcess:
        args = [sys.executable, "-E", "-s", str(PREVIEW), "--base", str(self.base)]
        for lane in lanes:
            args += ["--lane", str(lane)]
        return subprocess.run(args, env=self.env, capture_output=True, text=True, timeout=120)


class MergePreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kgirl-preview-test-"))
        self.r = Repo(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_disjoint_lanes_pass_and_uncommitted_files_are_included(self):
        a, b = self.r.lane("a"), self.r.lane("b")
        Repo.write(a, "src/pkg/extra.py", "EXTRA = 2\n")                      # untracked
        Repo.write(b, "tests/harness/test_more.py", TEST_MODULE)               # untracked test module
        res = self.r.preview(a, b)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("tests/harness: 2 cases", res.stdout)

    def test_same_path_with_disjoint_edits_is_three_way_merged(self):
        a, b = self.r.lane("a"), self.r.lane("b")
        middle = "".join(f"KEEP_{n} = {n}\n" for n in range(10))
        Repo.write(self.r.base, "src/pkg/core.py", "TOP = 1\n" + middle + "BOTTOM = 1\n")
        self.r.git("add", "-A")
        self.r.git("commit", "-q", "-m", "spread lines")
        # Recreate lanes from the new base so each contains the preview base commit.
        self.r.git("worktree", "remove", "--force", str(a))
        self.r.git("worktree", "remove", "--force", str(b))
        self.r.git("branch", "-D", "lane/a")
        self.r.git("branch", "-D", "lane/b")
        a, b = self.r.lane("a"), self.r.lane("b")
        Repo.write(a, "src/pkg/core.py", "TOP = 2\n" + middle + "BOTTOM = 1\n")
        Repo.write(b, "src/pkg/core.py", "TOP = 1\n" + middle + "BOTTOM = 2\n")
        res = self.r.preview(a, b)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("WARN src/pkg/core.py: clean 3-way merge of a, b", res.stdout)

    def test_same_line_edits_are_a_conflict_with_the_hunk(self):
        a, b = self.r.lane("a"), self.r.lane("b")
        Repo.write(a, "src/pkg/core.py", "VALUE = 2\n")
        Repo.write(b, "src/pkg/core.py", "VALUE = 3\n")
        res = self.r.preview(a, b)
        self.assertEqual(res.returncode, 1)
        self.assertIn("CONFLICT src/pkg/core.py: a, b", res.stdout)
        self.assertIn("<<<<<<< a", res.stdout)
        self.assertIn("=======", res.stdout)
        self.assertIn(">>>>>>> b", res.stdout)

    def test_delete_modify_conflict_prints_a_hunk(self):
        a, b = self.r.lane("a"), self.r.lane("b")
        (a / "src/pkg/core.py").unlink()
        Repo.write(b, "src/pkg/core.py", "VALUE = 2\n")
        res = self.r.preview(a, b)
        self.assertEqual(res.returncode, 1)
        self.assertIn("CONFLICT src/pkg/core.py: a, b", res.stdout)
        self.assertIn("<<<<<<< a", res.stdout)
        self.assertIn("(deleted)", res.stdout)
        self.assertIn(">>>>>>> b", res.stdout)

    def test_syntax_error_in_a_lane_fails_compile(self):
        a = self.r.lane("a")
        Repo.write(a, "src/pkg/core.py", "def broken(:\n")
        res = self.r.preview(a)
        self.assertEqual(res.returncode, 1)
        self.assertIn("FAIL compile", res.stdout)
        self.assertIn("SKIP discovery", res.stdout)

    def test_test_folder_without_init_is_reported(self):
        a = self.r.lane("a")
        Repo.write(a, "tests/harness/console/test_view.py", TEST_MODULE)
        res = self.r.preview(a)
        self.assertEqual(res.returncode, 1)
        self.assertIn("tests/harness/console: test files but no cases", res.stdout)

    def test_deletion_in_a_lane_is_applied(self):
        a = self.r.lane("a")
        (a / "tests/harness/test_core.py").unlink()
        Repo.write(a, "tests/harness/test_new.py", TEST_MODULE)
        res = self.r.preview(a)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("1 deleted", res.stdout)
        self.assertIn("tests/harness: 1 cases", res.stdout)

    def test_lane_missing_the_base_commit_is_refused(self):
        a = self.r.lane("a")
        Repo.write(self.r.base, "src/pkg/later.py", "LATER = 1\n")
        self.r.git("add", "-A")
        self.r.git("commit", "-q", "-m", "base moved on")
        res = self.r.preview(a)
        self.assertEqual(res.returncode, 1)
        self.assertIn("does not contain the base commit", res.stdout)

    def test_no_git_state_changes(self):
        a = self.r.lane("a")
        Repo.write(a, "src/pkg/extra.py", "EXTRA = 2\n")
        before = (self.r.git("for-each-ref"), self.r.git("status", "--porcelain", cwd=a),
                  self.r.git("worktree", "list", "--porcelain"))
        self.r.preview(a)
        after = (self.r.git("for-each-ref"), self.r.git("status", "--porcelain", cwd=a),
                 self.r.git("worktree", "list", "--porcelain"))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)

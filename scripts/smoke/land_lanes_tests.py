#!/usr/bin/env python3
"""Offline tests for land_lanes.py in throwaway repos. Nothing real is read, fetched or pushed.

Each case builds: an "upstream" commit (public, noreply), one stack commit with a personal email on top,
lane worktrees branched from the stack, and an unrelated branch that also carries a personal email.
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
TOOL = HERE / "land_lanes.py"
NOREPLY = "1+kgirl-test@users.noreply.github.com"
PERSONAL = "someone@example.org"


class Repo:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        (tmp / "home").mkdir()
        self.env = {**os.environ, "HOME": str(tmp / "home"), "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_NOSYSTEM": "1"}
        self.stack = tmp / "stack"
        self.git("init", "-q", "-b", "main", str(self.stack), cwd=tmp)
        self.git("config", "user.name", "kgirl-test")
        self.git("config", "user.email", NOREPLY)
        self.commit("README.md", "hello\n", "public base", NOREPLY)
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.git("checkout", "-q", "-b", "feat/stack")
        self.commit("src/a.py", "A = 1\n", "personal work", PERSONAL)
        self.git("branch", "other", "main")                    # unrelated branch with its own personal commit
        self.git("worktree", "add", "-q", str(tmp / "other"), "other")
        self.commit("src/o.py", "O = 1\n", "other personal", PERSONAL, cwd=tmp / "other")

    def git(self, *args, cwd=None, env=None):
        r = subprocess.run(["git", *args], cwd=cwd or self.stack, env={**self.env, **(env or {})},
                           capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(f"git {' '.join(args)}: {r.stderr}")
        return r.stdout.strip()

    def commit(self, rel, text, msg, email, cwd=None):
        root = cwd or self.stack
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        self.git("add", "-A", cwd=root)
        ident = {"GIT_AUTHOR_EMAIL": email, "GIT_COMMITTER_EMAIL": email}
        self.git("commit", "-q", "-m", msg, cwd=root, env=ident)

    def lane(self, name: str, files: dict[str, str]) -> Path:
        path = self.tmp / name
        self.git("worktree", "add", "-q", "-b", f"feat/{name}", str(path))
        for rel, text in files.items():
            f = path / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
        return path

    def run(self, *lanes: Path, apply=False, steps=None):
        args = [sys.executable, "-E", "-s", str(TOOL), "--stack", str(self.stack)]
        for lane in lanes:
            args += ["--lane", str(lane)]
        if steps:
            args += ["--steps", steps]
        if apply:
            args.append("--apply")
        return subprocess.run(args, env=self.env, capture_output=True, text=True, timeout=60)

    def emails(self, ref):
        return set(self.git("log", "--format=%ae%n%ce", ref, "^refs/remotes/origin/main").split())

    def refs(self):
        return self.git("for-each-ref", "--format=%(refname) %(objectname)")


class LandLanesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kgirl-land-test-"))
        self.r = Repo(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_plan_mode_writes_nothing(self):
        a = self.r.lane("a", {"src/b.py": "B = 1\n"})
        before = self.r.refs()
        res = self.r.run(a)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("1 commit(s) carry", res.stdout)
        self.assertNotIn(PERSONAL, res.stdout + res.stderr)          # emails are counted, never printed
        self.assertEqual(self.r.refs(), before)
        self.assertTrue((a / "src/b.py").exists())

    def test_rewrite_keeps_trees_and_lane_work_and_skips_unrelated_branches(self):
        a = self.r.lane("a", {"src/b.py": "B = 1\n"})
        tree = self.r.git("rev-parse", "feat/stack^{tree}")
        res = self.r.run(a, apply=True, steps="rewrite")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertEqual(self.r.emails("feat/stack"), {NOREPLY})
        self.assertEqual(self.r.emails("feat/a"), {NOREPLY})
        self.assertEqual(self.r.git("rev-parse", "feat/stack^{tree}"), tree)
        self.assertEqual(self.r.git("rev-parse", "feat/a"), self.r.git("rev-parse", "feat/stack"))
        self.assertIn(PERSONAL, self.r.emails("other"))               # outside the stack: untouched
        self.assertEqual(self.r.git("status", "--porcelain", cwd=a), "?? src/b.py")
        self.assertEqual(self.r.git("status", "--porcelain"), "")
        self.assertTrue(self.r.git("rev-parse", "refs/kgirl-landing/original/feat/stack"))
        again = self.r.run(a, apply=True, steps="rewrite")             # idempotent
        self.assertIn("0 commit(s) carry", again.stdout)

    def test_full_landing_commits_and_merges_in_order(self):
        a = self.r.lane("a", {"src/b.py": "B = 1\n"})
        b = self.r.lane("b", {"src/c.py": "C = 1\n"})
        res = self.r.run(a, b, apply=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertTrue((self.r.stack / "src/b.py").exists() and (self.r.stack / "src/c.py").exists())
        merges = self.r.git("log", "--merges", "--format=%s", "feat/stack").splitlines()
        self.assertEqual(len(merges), 2)
        self.assertIn("feat/b", merges[0])
        self.assertEqual(self.r.emails("feat/stack"), {NOREPLY})

    def test_private_path_stops_before_any_write(self):
        a = self.r.lane("a", {".env": "X=1\n"})
        before = self.r.refs()
        res = self.r.run(a, apply=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("look private", res.stdout)
        self.assertEqual(self.r.refs(), before)

    def test_conflicting_lanes_abort_cleanly(self):
        a = self.r.lane("a", {"src/a.py": "A = 2\n"})
        b = self.r.lane("b", {"src/a.py": "A = 3\n"})
        res = self.r.run(a, b, apply=True)
        self.assertEqual(res.returncode, 1, res.stdout)
        self.assertIn("failed and was aborted", res.stdout)
        self.assertEqual(self.r.git("status", "--porcelain"), "")
        self.assertEqual((self.r.stack / "src/a.py").read_text(), "A = 2\n")   # first merge kept, second aborted

    def test_dirty_stack_is_refused(self):
        (self.r.stack / "loose.txt").write_text("x\n")
        res = self.r.run(apply=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("must be clean", res.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)

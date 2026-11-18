#!/usr/bin/env python3
"""Offline tests for the pre-push privacy gate (scripts/hooks/pre-push) and its installer.

Every test builds a throwaway git repository in a temp dir, with an isolated HOME and no global git
config. It installs the gate there, beside a stub "previous" hook that stands in for Git LFS, and feeds
the hook the stdin git would send on a push. Nothing is pushed, fetched or networked, and the user's real
hooks are never touched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
ZERO = "0" * 40
SAFE = {"name": "kgirl-test", "email": "1+kgirl-test@users.noreply.github.com"}
STUB = """#!/bin/sh
# stand-in for the Git LFS pre-push hook: record what it was handed
printf '%s\\n%s\\n' "$1" "$2" > "$(dirname "$0")/stub.args"
cat > "$(dirname "$0")/stub.stdin"
exit 0
"""


class GateRepo:
    """A temp repo with the gate's scripts copied in, a fake origin/main, and a stub previous hook."""

    def __init__(self, tmp: Path):
        self.root = tmp / "repo"
        self.home = tmp / "home"
        self.home.mkdir()
        self.env = {**os.environ, "HOME": str(self.home), "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}
        self.git("init", "-q", "-b", "main", str(self.root), cwd=tmp)
        for rel in ("hooks/pre-push", "install_prepush_hook.sh", "smoke/prepush_scan.py", "smoke/_support.py"):
            dst = self.root / "scripts" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SCRIPTS / rel, dst)
        self.commit("README.md", "kgirl test repo\n")
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.hooks = self.root / ".git" / "hooks"
        (self.hooks / "pre-push").write_text(STUB)
        (self.hooks / "pre-push").chmod(0o755)

    def git(self, *args, cwd=None, ident=SAFE, check=True):
        env = {**self.env, "GIT_AUTHOR_NAME": ident["name"], "GIT_AUTHOR_EMAIL": ident["email"],
               "GIT_COMMITTER_NAME": ident["name"], "GIT_COMMITTER_EMAIL": ident["email"]}
        r = subprocess.run(["git", *args], cwd=cwd or self.root, env=env, capture_output=True, text=True)
        if check and r.returncode:
            raise AssertionError(f"git {' '.join(args)} failed: {r.stderr}")
        return r.stdout.strip()

    def commit(self, path: str, text: str, ident=SAFE) -> str:
        f = self.root / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        self.git("add", path)
        self.git("commit", "-q", "-m", f"add {path}", ident=ident)
        return self.git("rev-parse", "HEAD")

    def install(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.root / "scripts/install_prepush_hook.sh"), *args],
                              cwd=self.root, env=self.env, capture_output=True, text=True)

    def push(self, lines: list[tuple[str, str, str, str]]) -> subprocess.CompletedProcess:
        stdin = "".join(" ".join(t) + "\n" for t in lines)
        return subprocess.run([str(self.hooks / "pre-push"), "origin", "https://example.invalid/r.git"],
                              input=stdin, cwd=self.root, env=self.env, capture_output=True, text=True,
                              timeout=120)

    def head_push(self, branch="feat/x", remote_sha=ZERO):
        return self.push([(f"refs/heads/{branch}", self.git("rev-parse", "HEAD"), f"refs/heads/{branch}", remote_sha)])

    def stub_ran(self) -> bool:
        return (self.hooks / "stub.args").exists()


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kgirl-hook-test-"))
        self.r = GateRepo(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_install_keeps_previous_hook_and_uninstall_restores_it_exactly(self):
        self.assertEqual(self.r.install().returncode, 0)
        self.assertIn("kgirl pre-push privacy gate", (self.r.hooks / "pre-push").read_text())
        self.assertEqual((self.r.hooks / "pre-push.before-kgirl").read_text(), STUB)
        self.assertTrue((self.r.hooks / "kgirl-prepush" / "prepush_scan.py").is_file())
        self.assertEqual(self.r.install("--uninstall").returncode, 0)
        self.assertEqual((self.r.hooks / "pre-push").read_text(), STUB)
        self.assertFalse((self.r.hooks / "pre-push.before-kgirl").exists())
        self.assertFalse((self.r.hooks / "kgirl-prepush").exists())

    def test_reinstall_is_idempotent(self):
        self.r.install()
        self.assertEqual(self.r.install().returncode, 0)
        self.assertEqual((self.r.hooks / "pre-push.before-kgirl").read_text(), STUB)

    def test_refuses_when_two_foreign_hooks_would_collide(self):
        (self.r.hooks / "pre-push.before-kgirl").write_text("#!/bin/sh\nexit 0\n")
        res = self.r.install()
        self.assertNotEqual(res.returncode, 0)
        self.assertEqual((self.r.hooks / "pre-push").read_text(), STUB)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kgirl-hook-test-"))
        self.r = GateRepo(self.tmp)
        self.assertEqual(self.r.install().returncode, 0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clean_push_passes_and_hands_identical_input_to_previous_hook(self):
        self.r.commit("src/ok.py", "def ok():\n    return 1\n")
        res = self.r.head_push()
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertTrue(self.r.stub_ran())
        self.assertEqual((self.r.hooks / "stub.args").read_text().split(), ["origin", "https://example.invalid/r.git"])
        self.assertIn(self.r.git("rev-parse", "HEAD"), (self.r.hooks / "stub.stdin").read_text())

    def test_personal_email_blocks_and_is_never_printed(self):
        self.r.commit("src/a.py", "x = 1\n", ident={"name": "kgirl-test", "email": "someone.private@example.com"})
        res = self.r.head_push()
        self.assertEqual(res.returncode, 1)
        self.assertFalse(self.r.stub_ran(), "the previous hook must not run when the gate blocks")
        self.assertNotIn("someone.private", res.stdout + res.stderr)

    def test_secret_in_content_blocks(self):
        self.r.commit("src/conf.py", 'api_key = "abcd1234efgh5678"\n')
        self.assertEqual(self.r.head_push().returncode, 1)

    def test_wallet_file_blocks_without_being_opened(self):
        self.r.commit("Main Wallet 1.json", '{"k": "v"}\n')
        res = self.r.head_push()
        self.assertEqual(res.returncode, 1)
        self.assertIn("contents not opened", res.stdout + res.stderr)

    def test_pushing_a_commit_other_than_head_fails_closed(self):
        old = self.r.git("rev-parse", "HEAD")
        self.r.commit("src/b.py", "y = 2\n")
        res = self.r.push([("refs/heads/feat/x", old, "refs/heads/feat/x", ZERO)])
        self.assertEqual(res.returncode, 1)
        self.assertFalse(self.r.stub_ran())

    def test_new_branch_without_a_fetched_base_fails_closed(self):
        self.r.git("update-ref", "-d", "refs/remotes/origin/main")
        self.r.commit("src/c.py", "z = 3\n")
        self.assertEqual(self.r.head_push().returncode, 1)

    def test_update_scans_only_what_is_new(self):
        bad = self.r.commit("src/d.py", "w = 4\n", ident={"name": "kgirl-test", "email": "someone.private@example.com"})
        self.r.commit("src/e.py", "v = 5\n")
        self.assertEqual(self.r.push([("refs/heads/feat/x", self.r.git("rev-parse", "HEAD"),
                                       "refs/heads/feat/x", bad)]).returncode, 0)

    def test_deleting_a_remote_branch_is_not_scanned(self):
        res = self.r.push([("(delete)", ZERO, "refs/heads/feat/old", self.r.git("rev-parse", "HEAD"))])
        self.assertEqual(res.returncode, 0)
        self.assertTrue(self.r.stub_ran())

    def test_private_names_file_blocks_and_the_name_is_never_printed(self):
        (self.r.root / ".git" / "kgirl-private-names").write_text("Zebulon Quartermaine\n")
        self.r.commit("docs/notes.md", "thanks to zebulon quartermaine for the idea\n")
        res = self.r.head_push()
        self.assertEqual(res.returncode, 1)
        self.assertNotIn("Zebulon", res.stdout + res.stderr)
        self.assertNotIn("zebulon", res.stdout + res.stderr)

    def test_gate_uses_its_installed_scanner_not_the_branch_copy(self):
        weakened = self.r.root / "scripts/smoke/prepush_scan.py"
        weakened.write_text("raise SystemExit(0)\n")
        self.r.git("add", "-A")
        self.r.git("commit", "-q", "-m", "weaken scanner",
                   ident={"name": "kgirl-test", "email": "someone.private@example.com"})
        self.assertEqual(self.r.head_push().returncode, 1)

    def test_missing_installed_scanner_fails_closed(self):
        shutil.rmtree(self.r.hooks / "kgirl-prepush")
        self.r.commit("src/f.py", "u = 6\n")
        self.assertEqual(self.r.head_push().returncode, 1)
        self.assertFalse(self.r.stub_ran())


if __name__ == "__main__":
    unittest.main(verbosity=2)

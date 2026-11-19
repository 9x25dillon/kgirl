#!/usr/bin/env python3
"""Offline regression checks for engineering tools; no models, Git commits, or real child processes."""

from __future__ import annotations

import contextlib
import io
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import check_all
import prepush_scan as scan
from _support import Stages, isolated_home, require


class ReportingTests(unittest.TestCase):
    def test_failure_blocks_dependents_without_hiding_independent_stage(self):
        stages, output = Stages(), io.StringIO()
        dependent, independent = Mock(), Mock()
        with contextlib.redirect_stdout(output):
            stages.check("setup", lambda: require(False, "fixture failure"))
            stages.check("dependent", dependent, needs=("setup",))
            stages.check("independent", independent)
            result = stages.status()
        dependent.assert_not_called()
        independent.assert_called_once()
        self.assertEqual(result, 1)
        self.assertIn("FAIL setup", output.getvalue())
        self.assertIn("SKIP dependent", output.getvalue())
        self.assertIn("1/3 stages passed", output.getvalue())

    def test_isolation_does_not_inherit_live_indexes_or_leave_changed_environment(self):
        settings = {"KGIRL_HOME": "/not-a-real-live-home", "KGIRL_CHUNKS_DB": "/not-a-real-index",
                    "KGIRL_ROUTER_SSH": "must-not-run", "OLLAMA_HOST": "http://outside.invalid",
                    "AURIC_PORTER_DB": "/not-a-real-porter-db"}
        with patch.dict(os.environ, settings):
            before = dict(os.environ)
            with isolated_home() as scratch:
                self.assertEqual(os.environ["KGIRL_HOME"], str(scratch / "home"))
                self.assertEqual(os.environ["AURIC_PORTER_DB"], str(scratch / "porter-not-configured.sqlite"))
                for key in ("KGIRL_CHUNKS_DB", "KGIRL_ROUTER_SSH", "OLLAMA_HOST"):
                    self.assertNotIn(key, os.environ)
                self.assertEqual(os.environ.get("HOME"), before.get("HOME"))
                (scratch / "fixture.txt").write_text("fixture", encoding="utf-8")
            self.assertFalse(scratch.exists())
            self.assertEqual(dict(os.environ), before)


class FakeHistory:
    """Minimal Git transport fixture. Any unexpected blob/diff read fails the check."""

    base, old, tip = "a" * 40, "b" * 40, "c" * 40

    def __init__(self, path="module.py", blob=b"safe text\n", diff=b"", *, commits=None):
        self.path, self.blob, self.diff = path, blob, diff
        self.commits = [self.old, self.tip] if commits is None else commits
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        verb = args[0]
        if verb == "rev-parse":
            return (self.tip if args[-1] == "HEAD^{commit}" else self.base).encode()
        if verb == "rev-list" and "--reverse" in args:
            return "\n".join(self.commits).encode()
        if verb == "rev-list":
            commit = args[-1]
            parent = self.base if commit == self.old else self.old
            return f"{commit} {parent}".encode()
        if verb == "show":
            return b"fixture\nfixture@users.noreply.github.com\nfixture\nfixture@users.noreply.github.com\nfixture commit"
        if verb == "diff":
            # The tip deletes the file; only the earlier outgoing commit introduced it.
            if self.tip in args:
                return b""
            if "--name-only" in args:
                return self.path.encode() + b"\0"
            if scan.protected(self.path):
                raise AssertionError("protected content was requested through diff")
            return self.diff
        if verb == "cat-file":
            if scan.protected(self.path):
                raise AssertionError("protected content was requested through cat-file")
            return {"-t": b"blob", "-s": str(len(self.blob)).encode(), "blob": self.blob}[args[1]]
        raise AssertionError(f"unexpected Git verb {verb}")


class PrivacyTests(unittest.TestCase):
    def invoke(self, history, *extra):
        output = io.StringIO()
        with patch.object(scan, "git", side_effect=history), patch.object(sys, "argv", ["scan", *extra]), \
                contextlib.redirect_stdout(output):
            code = scan.main()
        return code, output.getvalue()

    def test_protected_paths_are_reported_without_reading_contents(self):
        for path in (".env.local", "Main Wallet.json", "private/note.md", "keys.json", "tokens.json",
                     "oauth_credentials.json", ".ssh/id_ed25519", "docs/Hand_off.md", "Tithe/note.md"):
            with self.subTest(path=path):
                history = FakeHistory(path=path)
                code, out = self.invoke(history)
                self.assertEqual(code, 1)
                self.assertIn("contents not opened", out)
                self.assertFalse(any(call[0] == "cat-file" for call in history.calls))

    def test_secret_removed_in_tip_is_still_flagged_in_earlier_outgoing_commit(self):
        token = "ghp_" + "A" * 24  # synthetic pattern, not a credential
        diff = f"@@ -0,0 +1 @@\n+value = '{token}'\n".encode()
        code, out = self.invoke(FakeHistory(diff=diff))
        self.assertEqual(code, 1)
        self.assertIn(FakeHistory.old[:12], out)
        self.assertIn("provider token", out)
        self.assertNotIn(token, out)
        self.assertNotIn("value =", out)

    def test_private_value_in_filename_is_redacted(self):
        token = "ghp_" + "B" * 24
        code, out = self.invoke(FakeHistory(path=f"note-{token}.txt"))
        self.assertEqual(code, 1)
        self.assertIn("redacted path sha256:", out)
        self.assertNotIn(token, out)

    def test_binary_and_unscannable_payloads_are_not_clean(self):
        for blob in (b"binary\0fixture", b"\xff\xfe", b"version https://git-lfs.github.com/spec/v1\n", b"x" * 64):
            with self.subTest(prefix=blob[:10]), patch.object(scan, "MAX_BLOB", 32):
                code, out = self.invoke(FakeHistory(blob=blob))
                self.assertEqual(code, 1)
                self.assertIn("unscanned", out)

    def test_git_failure_does_not_relay_sensitive_error_text(self):
        def failed(*args):
            raise scan.ScanError("private error details must not escape")
        code, out = self.invoke(failed)
        self.assertEqual(code, 2)
        self.assertIn("SCAN INCOMPLETE", out)
        self.assertNotIn("private error details", out)

    def test_no_outgoing_history_is_reported_as_empty_not_a_privacy_clearance(self):
        code, out = self.invoke(FakeHistory(commits=[]))
        self.assertEqual(code, 0)
        self.assertIn("NO OUTGOING COMMITS", out)
        self.assertIn("not a privacy clearance", out)

    def test_commit_email_is_flagged_without_printing_it(self):
        history = FakeHistory()
        def with_email(*args):
            if args[0] == "show":
                return b"fixture\nperson@example.invalid\nfixture\nfixture@users.noreply.github.com\nmessage"
            return history(*args)
        code, out = self.invoke(with_email)
        self.assertEqual(code, 1)
        self.assertIn("email address", out)
        self.assertNotIn("person@example.invalid", out)


class CheckDayTests(unittest.TestCase):
    def run_checks(self, behaviors):
        output = io.StringIO()
        children = []
        def spawn(*args, **kwargs):
            child = SimpleNamespace(pid=12345 + len(children), wait=Mock(side_effect=behaviors[len(children)]))
            children.append(child)
            return child
        with tempfile.TemporaryDirectory(prefix="kgirl-tool-test-") as tmp:
            with patch.object(check_all.tempfile, "mkdtemp", return_value=tmp), \
                    patch.object(check_all.subprocess, "Popen", side_effect=spawn), \
                    patch.object(check_all.os, "killpg") as kill, \
                    patch.object(sys, "argv", ["check_all", "--timeout", "1"]), \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                status = check_all.main()
                logs = sorted(p.name for p in Path(tmp).iterdir())
        return status, output.getvalue(), children, logs, kill

    def test_failed_check_does_not_hide_remaining_checks(self):
        count = len(check_all.checks())
        code, out, children, logs, kill = self.run_checks([[1], *[[0] for _ in range(count - 1)]])
        self.assertEqual(code, 1)
        self.assertEqual(len(children), count)
        self.assertEqual(len(logs), count)
        self.assertIn("console-smoke", out)
        self.assertIn("FAIL (1)", out)
        kill.assert_not_called()

    def test_timeout_terminates_only_that_check_group_then_continues(self):
        timeout = subprocess.TimeoutExpired(["synthetic-check"], 1)
        count = len(check_all.checks())
        code, out, children, logs, kill = self.run_checks([[timeout, -15], *[[0] for _ in range(count - 1)]])
        self.assertEqual(code, 1)
        self.assertEqual(len(children), count)
        self.assertIn("TIMEOUT", out)
        kill.assert_called_once_with(children[0].pid, signal.SIGTERM)

    def test_interrupt_stops_before_starting_more_checks(self):
        code, out, children, logs, kill = self.run_checks([[KeyboardInterrupt(), -15]])
        self.assertEqual(code, 130)
        self.assertEqual(len(children), 1)
        self.assertIn("Interrupted", out)
        kill.assert_called_once_with(children[0].pid, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main(verbosity=2)

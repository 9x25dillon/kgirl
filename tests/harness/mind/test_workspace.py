"""Offline tests for mind.workspace: what the chat agent may look at, change and run in a project.

Everything happens in a temporary directory: no model, no network, no real project files. The few commands
that are executed are this Python interpreter printing a line.
"""

import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.harness._fixtures import write
from kgirl.harness.mind.workspace import (LOOK_OUTPUT, READ_LINES, WRITE_BYTES, AuditEvent, Change, Workspace,
                                          WorkspaceError, allowlisted_commands)

APP = "def main():\n    return 1\n"


class Asker:
    """Stands in for the person at the keyboard: answers every question the same way and keeps them."""

    def __init__(self, answer: bool):
        self.answer = answer
        self.asked: list[tuple[str, str]] = []

    def __call__(self, question: str, shown: str) -> bool:
        self.asked.append((question, shown))
        return self.answer


class WorkspaceCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {"KGIRL_VERIFY_ALLOWLIST": ""})    # the user's allowlist stays out
        env.start()
        self.addCleanup(env.stop)
        self.base = Path(self.tmp.name).resolve()
        self.root = write(self.base / "proj", {
            "app.py": APP,
            "pkg/util.py": "X = 1\nY = 2\n",
            "README.md": "# proj\n",
        })
        self.notes: list[str] = []

    def ws(self, **kw) -> Workspace:
        kw.setdefault("notify", self.notes.append)
        return Workspace(self.root, **kw)

    def text(self, rel: str) -> str:
        return (self.root / rel).read_bytes().decode("utf-8")


class PathTests(WorkspaceCase):
    def test_root_must_be_a_directory(self):
        with self.assertRaises(WorkspaceError):
            Workspace(self.root / "missing")
        with self.assertRaises(WorkspaceError):
            Workspace(self.root / "app.py")

    def test_name_and_rel(self):
        ws = self.ws()
        self.assertEqual(ws.name, "proj")
        self.assertEqual(ws.rel(ws.root / "pkg" / "util.py"), "pkg/util.py")
        self.assertEqual(ws.rel(ws.root), ".")

    def test_resolve_inside_the_root(self):
        ws = self.ws()
        self.assertEqual(ws.resolve("pkg/util.py"), self.root / "pkg" / "util.py")
        self.assertEqual(ws.resolve(str(self.root / "app.py")), self.root / "app.py")
        self.assertEqual(ws.resolve(None), self.root)
        self.assertEqual(ws.resolve("  "), self.root)

    def test_resolve_refuses_to_leave_the_root(self):
        ws = self.ws()
        for path in ("../outside.txt", "pkg/../../outside.txt", str(self.base / "outside.txt")):
            with self.assertRaises(WorkspaceError) as caught:
                ws.resolve(path)
            self.assertIn("outside the workspace", str(caught.exception), path)

    def test_a_symlink_out_of_the_root_is_outside(self):
        (self.base / "outside.txt").write_text("secret plans\n", encoding="utf-8")
        os.symlink(self.base / "outside.txt", self.root / "escape.txt")
        ws = self.ws()
        with self.assertRaises(WorkspaceError) as caught:
            ws.read("escape.txt")
        self.assertIn("outside the workspace", str(caught.exception))

    def test_git_is_off_limits_for_reading_and_writing(self):
        write(self.root, {".git/config": "[core]\n"})
        ws = self.ws(auto_edit=True)
        for attempt in (lambda: ws.resolve(".git/config"), lambda: ws.read(".git/config"),
                        lambda: ws.write(".git/hooks/pre-commit", "#!/bin/sh\n")):
            with self.assertRaises(WorkspaceError) as caught:
                attempt()
            self.assertIn(".git is off limits", str(caught.exception))

    def test_write_resolution_guards(self):
        ws = self.ws()
        with self.assertRaises(WorkspaceError) as caught:
            ws.resolve(".", write=True)
        self.assertIn("workspace root", str(caught.exception))
        with self.assertRaises(WorkspaceError) as caught:
            ws.resolve("node_modules/left-pad/index.js", write=True)
        self.assertIn("does not write inside node_modules/", str(caught.exception))
        self.assertEqual(ws.resolve("node_modules/left-pad/index.js"),            # reading there is allowed
                         self.root / "node_modules" / "left-pad" / "index.js")

    def test_no_writing_through_a_symlink(self):
        os.symlink(self.root / "app.py", self.root / "link.py")
        ws = self.ws(auto_edit=True)
        with self.assertRaises(WorkspaceError) as caught:
            ws.write("link.py", "x = 1\n")
        self.assertIn("symlink", str(caught.exception))
        self.assertEqual(self.text("app.py"), APP)

    def test_no_writing_through_a_symlinked_directory(self):
        (self.root / "real").mkdir()
        os.symlink(self.root / "real", self.root / "linked")
        with self.assertRaises(WorkspaceError) as caught:
            self.ws(auto_edit=True).write("linked/new.py", "x = 1\n")
        self.assertIn("symlink", str(caught.exception))
        self.assertFalse((self.root / "real" / "new.py").exists())


class LookingTests(WorkspaceCase):
    def test_ls_directories_first_then_files(self):
        lines = self.ws().ls().split("\n")
        self.assertEqual(lines[0], "pkg/")
        self.assertTrue(lines[1].startswith("app.py  ("), lines[1])
        self.assertTrue(lines[2].startswith("README.md  ("), lines[2])
        self.assertEqual(len(lines), 3)

    def test_ls_depth_and_sizes(self):
        lines = self.ws().ls(depth=2).split("\n")
        self.assertEqual(lines[:2], ["pkg/", "pkg/util.py  (12 B)"])

    def test_ls_hides_git_and_does_not_descend_into_skipped_dirs(self):
        write(self.root, {".git/HEAD": "ref: refs/heads/main\n", "node_modules/dep/index.js": "x\n"})
        listing = self.ws().ls(depth=3)
        self.assertNotIn(".git", listing)
        self.assertIn("node_modules/", listing.split("\n"))
        self.assertNotIn("node_modules/dep", listing)

    def test_ls_file_empty_dir_missing_and_limit(self):
        (self.root / "empty").mkdir()
        ws = self.ws()
        self.assertEqual(ws.ls("pkg/util.py"), "pkg/util.py  (12 B)")
        self.assertEqual(ws.ls("empty"), "empty/ is empty")
        with self.assertRaises(WorkspaceError):
            ws.ls("nowhere")
        self.assertEqual(ws.ls(limit=1), "empty/\n(more entries not shown; list a subdirectory)")

    def test_read_numbers_lines(self):
        ws = self.ws()
        self.assertEqual(ws.read("pkg/util.py"), "pkg/util.py: lines 1-2 of 2\n    1 X = 1\n    2 Y = 2")
        self.assertEqual(ws.read("pkg/util.py", 2, 2), "pkg/util.py: lines 2-2 of 2\n    2 Y = 2")
        self.assertEqual(ws.read("pkg/util.py", start=5), "pkg/util.py has 2 lines")

    def test_read_returns_at_most_read_lines(self):
        write(self.root, {"long.txt": "".join(f"line {n}\n" for n in range(1, READ_LINES + 51))})
        out = self.ws().read("long.txt").split("\n")
        self.assertEqual(out[0], f"long.txt: lines 1-{READ_LINES} of {READ_LINES + 50}")
        self.assertEqual(len(out), READ_LINES + 1)
        self.assertEqual(out[-1], f"{READ_LINES:>5} line {READ_LINES}")

    def test_read_has_a_character_budget_and_continuation_pointer(self):
        write(self.root, {"wide.txt": ("x" * 1000 + "\n") * 30})
        out = self.ws().read("wide.txt")
        self.assertLessEqual(len(out), LOOK_OUTPUT + 120)
        self.assertIn("result truncated", out)
        self.assertIn("call read_file again", out)

    def test_read_refuses_directories_missing_and_binary_files(self):
        (self.root / "blob.bin").write_bytes(b"\x00\x01\x02 not text")
        ws = self.ws()
        for path, why in (("pkg", "is a directory; use ls"), ("nope.py", "does not exist"),
                          ("blob.bin", "is binary or larger than")):
            with self.assertRaises(WorkspaceError) as caught:
                ws.read(path)
            self.assertIn(why, str(caught.exception), path)

    def test_read_keeps_secret_files_out_of_the_conversation(self):
        write(self.root, {"credentials.json": '{"note": "nothing real in here"}\n'})
        with self.assertRaises(WorkspaceError) as caught:
            self.ws().read("credentials.json")
        self.assertIn("looks like it holds a secret (credentials file)", str(caught.exception))

    def test_grep_reports_path_line_and_text(self):
        ws = self.ws()
        self.assertEqual(ws.grep("return"), "app.py:2: return 1")
        self.assertEqual(ws.grep("X", "pkg"), "pkg/util.py:1: X = 1")
        self.assertEqual(ws.grep("= [12]$", "pkg/util.py"), "pkg/util.py:1: X = 1\npkg/util.py:2: Y = 2")
        self.assertEqual(ws.grep("zzz"), "no match for 'zzz'")

    def test_grep_takes_a_broken_regex_literally(self):
        self.assertEqual(self.ws().grep("main("), "app.py:1: def main():")

    def test_grep_needs_a_pattern_and_an_existing_path(self):
        ws = self.ws()
        with self.assertRaises(WorkspaceError):
            ws.grep("")
        with self.assertRaises(WorkspaceError):
            ws.grep("x", "nowhere")

    def test_grep_limit_counts_the_rest(self):
        write(self.root, {"many.txt": "needle\n" * 5})
        out = self.ws().grep("needle", "many.txt", limit=2)
        self.assertEqual(out, "many.txt:1: needle\nmany.txt:2: needle\n"
                              "(+3 more matches; narrow the pattern or the path)")

    def test_grep_skips_secret_files_hidden_and_vendored_dirs(self):
        write(self.root, {"secrets.json": '{"hint": "needle"}\n', ".hidden/h.py": "needle = 1\n",
                          "node_modules/dep/index.js": "needle\n", "found.py": "needle = 2\n"})
        self.assertEqual(self.ws().grep("needle"), "found.py:1: needle = 2")


class ChangingTests(WorkspaceCase):
    def test_without_a_way_to_ask_nothing_is_changed(self):
        ws = self.ws()
        with self.assertRaises(WorkspaceError) as caught:
            ws.write("new.txt", "hi\n")
        self.assertIn("needs the user's approval", str(caught.exception))
        with self.assertRaises(WorkspaceError):
            ws.edit("app.py", "return 1", "return 2")
        self.assertFalse((self.root / "new.txt").exists())
        self.assertEqual(self.text("app.py"), APP)
        self.assertEqual(ws.journal, [])

    def test_a_declined_change_is_not_applied(self):
        no = Asker(False)
        ws = self.ws(confirm=no)
        with self.assertRaises(WorkspaceError) as caught:
            ws.edit("app.py", "return 1", "return 2")
        self.assertIn("declined", str(caught.exception))
        self.assertEqual(self.text("app.py"), APP)
        self.assertEqual(ws.journal, [])
        question, diff = no.asked[0]
        self.assertEqual(question, "apply this change to app.py?")
        self.assertIn("-    return 1", diff)
        self.assertIn("+    return 2", diff)

    def test_an_approved_write_creates_the_file_and_is_journaled(self):
        yes = Asker(True)
        ws = self.ws(confirm=yes)
        out = ws.write("docs/new.txt", "hi\n")
        self.assertTrue(out.startswith("created docs/new.txt\n"), out)
        self.assertEqual(self.text("docs/new.txt"), "hi\n")
        self.assertEqual(ws.journal, [Change("docs/new.txt", None, "hi\n")])
        self.assertEqual(len(yes.asked), 1)
        self.assertIn("/dev/null", yes.asked[0][1])
        self.assertIn("+hi", yes.asked[0][1])
        self.assertEqual(self.notes, ["created docs/new.txt"])        # the person already saw the diff

    def test_auto_edit_applies_at_once_and_shows_the_diff(self):
        never = Asker(False)
        ws = self.ws(confirm=never, auto_edit=True)
        out = ws.edit("app.py", "return 1", "return 2")
        self.assertTrue(out.startswith("edited app.py\n"), out)
        self.assertEqual(self.text("app.py"), "def main():\n    return 2\n")
        self.assertEqual(never.asked, [])
        self.assertEqual(len(self.notes), 1)
        self.assertTrue(self.notes[0].startswith("edited app.py\n"), self.notes[0])
        self.assertIn("+    return 2", self.notes[0])

    def test_write_refusals(self):
        ws = self.ws(auto_edit=True)
        for path, content, why in (("pkg", "x\n", "is not a regular file"),
                                   ("big.txt", "x" * (WRITE_BYTES + 1), "is larger than 400 KB"),
                                   ("deploy.pem", "not really a key\n", "looks like it holds a secret (key file)"),
                                   ("__pycache__/m.pyc", "x", "does not write inside __pycache__/")):
            with self.assertRaises(WorkspaceError) as caught:
                ws.write(path, content)
            self.assertIn(why, str(caught.exception), path)
        self.assertEqual(ws.journal, [])

    def test_writing_identical_content_changes_nothing(self):
        ws = self.ws(auto_edit=True)
        self.assertEqual(ws.write("app.py", APP), "app.py: already has exactly this content")
        self.assertEqual(ws.journal, [])

    def test_edit_needs_one_exact_match(self):
        write(self.root, {"twice.py": "a = 1\nb = 2\na = 1\n"})
        ws = self.ws(auto_edit=True)
        with self.assertRaises(WorkspaceError) as caught:
            ws.edit("twice.py", "a = 1", "a = 9")
        self.assertIn("occurs 2 times in twice.py", str(caught.exception))
        with self.assertRaises(WorkspaceError) as caught:
            ws.edit("twice.py", "c = 3", "c = 4")
        self.assertIn("occurs 0 times in twice.py", str(caught.exception))
        self.assertEqual(self.text("twice.py"), "a = 1\nb = 2\na = 1\n")

    def test_edit_refuses_missing_file_and_blank_search(self):
        ws = self.ws(auto_edit=True)
        with self.assertRaises(WorkspaceError) as caught:
            ws.edit("nope.py", "a", "b")
        self.assertIn("use write_file to create it", str(caught.exception))
        with self.assertRaises(WorkspaceError) as caught:
            ws.edit("app.py", "  \n", "b")
        self.assertIn("edit needs the exact text", str(caught.exception))

    def test_edit_that_changes_nothing_is_reported(self):
        ws = self.ws(auto_edit=True)
        self.assertEqual(ws.edit("app.py", "return 1", "return 1"), "app.py: the edit changes nothing")
        self.assertEqual(ws.journal, [])

    def test_edit_tolerates_wrong_indentation_in_the_search(self):
        ws = self.ws(auto_edit=True)
        ws.edit("app.py", "        return 1", "        return 5")       # eight spaces; the file has four
        self.assertEqual(self.text("app.py"), "def main():\n    return 5\n")

    def test_edit_keeps_crlf_line_endings(self):
        (self.root / "dos.txt").write_bytes(b"a = 1\r\nb = 2\r\n")
        ws = self.ws(auto_edit=True)
        ws.edit("dos.txt", "a = 1\nb = 2", "a = 9\nb = 2")
        self.assertEqual((self.root / "dos.txt").read_bytes(), b"a = 9\r\nb = 2\r\n")

    def test_edit_refuses_a_file_that_is_not_utf8(self):
        (self.root / "latin.txt").write_bytes(b"caf\xe9\n")
        with self.assertRaises(WorkspaceError) as caught:
            self.ws(auto_edit=True).edit("latin.txt", "caf", "tea")
        self.assertIn("is not UTF-8 text", str(caught.exception))


class JournalTests(WorkspaceCase):
    def test_undo_with_an_empty_journal(self):
        self.assertEqual(self.ws().undo(), "nothing to undo")

    def test_undo_restores_an_edit_exactly_and_removes_a_created_file(self):
        ws = self.ws(auto_edit=True)
        ws.edit("app.py", "return 1", "return 2")
        ws.write("new.txt", "hi\n")
        self.assertEqual(ws.undo(), "undid the change to new.txt")
        self.assertFalse((self.root / "new.txt").exists())
        self.assertEqual(ws.undo(), "undid the change to app.py")
        self.assertEqual(self.text("app.py"), APP)
        self.assertEqual(ws.journal, [])
        self.assertEqual(ws.undo(), "nothing to undo")

    def test_undo_leaves_a_file_someone_else_changed(self):
        ws = self.ws(auto_edit=True)
        ws.edit("app.py", "return 1", "return 2")
        (self.root / "app.py").write_text("def main():\n    return 3\n", encoding="utf-8")
        self.assertEqual(ws.undo(), "app.py has changed since kgirl edited it; not undoing")
        self.assertEqual(self.text("app.py"), "def main():\n    return 3\n")
        self.assertEqual(len(ws.journal), 1)

    def test_diff_is_against_the_original_text(self):
        ws = self.ws(auto_edit=True)
        self.assertEqual(ws.diff(), "")
        ws.edit("app.py", "return 1", "return 2")
        ws.edit("app.py", "return 2", "return 3")
        ws.write("new.txt", "hi\n")
        diff = ws.diff()
        self.assertIn("--- a/app.py", diff)
        self.assertIn("+++ b/app.py", diff)
        self.assertIn("-    return 1", diff)
        self.assertIn("+    return 3", diff)
        self.assertNotIn("return 2", diff)
        self.assertIn("--- /dev/null", diff)
        self.assertIn("+++ b/new.txt", diff)

    def test_diff_is_empty_once_everything_is_back(self):
        ws = self.ws(auto_edit=True)
        ws.edit("app.py", "return 1", "return 2")
        ws.edit("app.py", "return 2", "return 1")
        self.assertEqual(ws.diff(), "")

    def test_record_makes_an_outside_change_undoable(self):
        ws = self.ws()
        after = "def main():\n    return 42\n"
        (self.root / "app.py").write_text(after, encoding="utf-8")       # e.g. an accepted swarm diff
        ws.record("app.py", APP, after)
        self.assertIn("+    return 42", ws.diff())
        self.assertEqual(ws.undo(), "undid the change to app.py")
        self.assertEqual(self.text("app.py"), APP)

    def test_audit_is_append_only_and_can_be_read_incrementally(self):
        ws = self.ws(auto_edit=True)
        ws.edit("app.py", "return 1", "return 2")
        ws.write("new.txt", "hi\n")
        ws.undo()
        self.assertEqual(ws.audit, [AuditEvent(1, "edit", "app.py", "applied"),
                                    AuditEvent(2, "create", "new.txt", "applied"),
                                    AuditEvent(3, "undo", "new.txt", "applied")])
        tail = ws.audit_since(1)
        tail.clear()
        self.assertEqual(len(ws.audit), 3)


class RunningTests(WorkspaceCase):
    def py(self, code: str) -> str:
        return shlex.join([sys.executable, "-c", code])

    def test_allow_and_the_environment_allowlist(self):
        ws = self.ws(allowed=[["make", "test"]])
        self.assertEqual(ws.allow("pytest -q"), ["pytest", "-q"])
        ws.allow("pytest -q")                                             # allowing twice adds nothing
        self.assertEqual(ws.allowed_commands(), [["make", "test"], ["pytest", "-q"]])
        with self.assertRaises(WorkspaceError):
            ws.allow("   ")
        with mock.patch.dict(os.environ, {"KGIRL_VERIFY_ALLOWLIST": "echo hi; ls -la ;"}):
            self.assertEqual(allowlisted_commands(), [["echo", "hi"], ["ls", "-la"]])
            self.assertEqual(ws.allowed_commands(), [["make", "test"], ["pytest", "-q"], ["echo", "hi"], ["ls", "-la"]])
        self.assertEqual(allowlisted_commands(), [])

    def test_an_unlisted_command_is_refused_when_nobody_can_be_asked(self):
        ws = self.ws(allowed=[["make", "test"]])
        with self.assertRaises(WorkspaceError) as caught:
            ws.run("rm -rf build")
        message = str(caught.exception)
        self.assertIn("`rm -rf build` is not a command the user has allowed", message)
        self.assertIn("(allowed: make test)", message)
        self.assertEqual(self.notes, [])                                   # nothing was started

    def test_shell_operators_need_an_exact_slash_allow(self):
        command = "printf hello | wc -c"
        yes = Asker(True)
        ws = self.ws(confirm=yes, allowed=[["printf", "hello", "|", "wc", "-c"]])
        with self.assertRaises(WorkspaceError) as caught:
            ws.run(command)
        self.assertIn("shell operator", str(caught.exception))
        self.assertEqual(yes.asked, [])

        ws.allow(command)
        with mock.patch("kgirl.harness.mind.workspace.subprocess.run") as run:
            run.return_value = mock.Mock(stdout="6\n", stderr="", returncode=0)
            self.assertIn("exit=0", ws.run(command))
        self.assertEqual(run.call_args.kwargs["shell"], True)
        self.assertEqual(run.call_args.args[0], command)

    def test_quoted_operator_is_an_ordinary_argument(self):
        command = "printf 'a|b'"
        ws = self.ws(allowed=[["printf", "a|b"]])
        with mock.patch("kgirl.harness.mind.workspace.subprocess.run") as run:
            run.return_value = mock.Mock(stdout="a|b", stderr="", returncode=0)
            ws.run(command)
        self.assertEqual(run.call_args.kwargs["shell"], False)

    def test_nothing_allowed_yet(self):
        with self.assertRaises(WorkspaceError) as caught:
            self.ws().run("ls")
        self.assertIn("(allowed: none yet)", str(caught.exception))

    def test_empty_and_unparseable_commands(self):
        ws = self.ws(confirm=Asker(True))
        with self.assertRaises(WorkspaceError) as caught:
            ws.run("")
        self.assertIn("run needs a command", str(caught.exception))
        with self.assertRaises(WorkspaceError) as caught:
            ws.run("echo 'unterminated")
        self.assertIn("cannot parse the command", str(caught.exception))

    def test_a_declined_command_does_not_run(self):
        no = Asker(False)
        ws = self.ws(confirm=no)
        with self.assertRaises(WorkspaceError) as caught:
            ws.run("echo hi")
        self.assertIn("the user declined to run that command", str(caught.exception))
        self.assertEqual(no.asked, [(f"run this in {self.root}?", "$ echo hi")])
        self.assertEqual(self.notes, [])

    def test_an_allowed_command_runs_without_asking(self):
        never = Asker(False)
        ws = self.ws(confirm=never)
        command = self.py("print('hi')")
        ws.allow(command)
        self.assertEqual(ws.run(command), f"$ {command}\nexit=0\nhi")
        self.assertEqual(never.asked, [])
        self.assertEqual(self.notes, [f"$ {command}"])

    def test_an_approved_command_runs_in_the_root(self):
        yes = Asker(True)
        ws = self.ws(confirm=yes)
        command = self.py("import os; print(os.getcwd())")
        self.assertEqual(ws.run(command), f"$ {command}\nexit=0\n{self.root}")
        self.assertEqual(yes.asked, [(f"run this in {self.root}?", f"$ {command}")])

    def test_exit_code_and_stderr_come_back(self):
        ws = self.ws()
        command = self.py("import sys; sys.stderr.write('boom\\n'); sys.exit(3)")
        self.assertEqual(ws.execute(shlex.split(command)), f"$ {command}\nexit=3\nboom")
        self.assertEqual(ws.audit[-1], AuditEvent(1, "run", command, "exit=3"))

    def test_secrets_are_stripped_from_the_command_environment(self):
        ws = self.ws()
        code = "import os; print(sorted(k for k in os.environ if k.startswith('KGIRL_TEST_')))"
        with mock.patch.dict(os.environ, {"KGIRL_TEST_API_KEY": "k", "KGIRL_TEST_TOKEN": "t",
                                          "KGIRL_TEST_PLAIN": "p"}):
            out = ws.execute([sys.executable, "-c", code])
        self.assertTrue(out.endswith("exit=0\n['KGIRL_TEST_PLAIN']"), out)

    def test_a_program_that_does_not_exist(self):
        out = self.ws().execute(["kgirl-no-such-program-for-tests"])
        self.assertTrue(out.startswith("$ kgirl-no-such-program-for-tests\ncould not start: "), out)

    def test_timeout_is_reported(self):
        ws = self.ws(run_timeout=0.5)
        out = ws.execute([sys.executable, "-c", "import time; time.sleep(30)"])
        self.assertTrue(out.endswith("\ntimed out after 0.5 s"), out)

    def test_large_output_is_truncated_with_a_marker(self):
        ws = self.ws()
        with mock.patch("kgirl.harness.mind.workspace.subprocess.run") as run:
            run.return_value = mock.Mock(stdout="x" * 4000, stderr="", returncode=0)
            out = ws.execute(["fake"])
        self.assertIn("output truncated", out)
        self.assertLess(len(out), 3200)


class DescriptionTests(WorkspaceCase):
    def test_no_git_checkout(self):
        ws = self.ws()
        self.assertEqual(ws.branch(), "")
        self.assertEqual(ws.describe(), f"WORKSPACE: {self.root}. Paths in ls, read_file, grep, edit, write_file "
                                        "and run are relative to it; the user approves each change.")

    def test_branch_from_head(self):
        write(self.root, {".git/HEAD": "ref: refs/heads/feat/holo\n"})
        ws = self.ws(auto_edit=True)
        self.assertEqual(ws.branch(), "feat/holo")
        self.assertIn(f"WORKSPACE: {self.root} (git branch feat/holo).", ws.describe())
        self.assertIn("changes apply at once and are journaled", ws.describe())

    def test_detached_head_is_a_short_hash(self):
        write(self.root, {".git/HEAD": "0123456789abcdef0123456789abcdef01234567\n"})
        self.assertEqual(self.ws().branch(), "0123456789")

    def test_worktree_git_file(self):
        write(self.base, {"real-git/worktrees/proj/HEAD": "ref: refs/heads/feat/mind-tests\n"})
        (self.root / ".git").write_text("gitdir: ../real-git/worktrees/proj\n", encoding="utf-8")
        self.assertEqual(self.ws().branch(), "feat/mind-tests")


if __name__ == "__main__":
    unittest.main()

"""Offline tests for mind.sandbox: the static check, the isolation level, and how runner results are read.

Nothing generated is executed here: the child runner is replaced by canned payloads, and the isolation
probe by a counter. No model, no network.
"""

import os
import unittest
from pathlib import Path
from unittest import mock

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind import sandbox


class LintTests(unittest.TestCase):
    def test_a_pure_module_is_clean(self):
        code = ("import math\nfrom collections import deque\n\n\ndef mean(xs):\n"
                "    return math.fsum(xs) / len(xs)\n")
        self.assertEqual(sandbox.lint(code), [])

    def test_network_and_process_imports_are_named(self):
        self.assertEqual(sandbox.lint("import socket\n"), ["imports socket (network)"])
        self.assertEqual(sandbox.lint("from urllib.request import urlopen\n"), ["imports urllib (network)"])
        self.assertEqual(sandbox.lint("import http.client as hc\n"), ["imports http (network)"])
        self.assertEqual(sandbox.lint("import subprocess\n"), ["imports subprocess (processes or native code)"])
        self.assertEqual(sandbox.lint("def f():\n    import ctypes\n"), ["imports ctypes (processes or native code)"])

    def test_network_and_processes_are_refused_even_when_io_is_allowed(self):
        self.assertEqual(sandbox.lint("import socket\nimport subprocess\n", allow_io=True),
                         ["imports socket (network)", "imports subprocess (processes or native code)"])

    def test_file_access_depends_on_allow_io(self):
        for code, problem in (("import shutil\n", "imports shutil (files)"),
                              ("import tempfile\n", "imports tempfile (files)"),
                              ("data = open('x.txt').read()\n", "opens files"),
                              ("import os\nos.remove('x.txt')\n", "calls os.remove (files)"),
                              ("import os\nfor name in os.listdir('.'):\n    pass\n", "calls os.listdir (files)")):
            self.assertEqual(sandbox.lint(code), [problem], code)
            self.assertEqual(sandbox.lint(code, allow_io=True), [], code)

    def test_os_process_calls_are_always_refused(self):
        for call in ("system('ls')", "popen('ls')", "fork()", "execv('/bin/ls', ['ls'])", "spawnlp(0, 'ls', 'ls')",
                     "kill(1, 9)"):
            name = call.split("(")[0]
            self.assertEqual(sandbox.lint(f"import os\nos.{call}\n", allow_io=True), [f"calls os.{name}"], call)

    def test_dunder_import_is_refused(self):
        self.assertEqual(sandbox.lint("mod = __import__('socket')\n"), ["calls __import__"])

    def test_each_problem_is_reported_once(self):
        self.assertEqual(sandbox.lint("import socket\nimport socket\nfrom socket import socket as s\n"),
                         ["imports socket (network)"])

    def test_relative_imports_are_not_mistaken_for_stdlib_modules(self):
        self.assertEqual(sandbox.lint("from . import socket\nfrom .subprocess import run\n"), [])

    def test_code_that_does_not_parse(self):
        problems = sandbox.lint("def broken(:\n    pass\n")
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("does not parse: "), problems[0])


class LevelTests(unittest.TestCase):
    def setUp(self):
        fresh = mock.patch.object(sandbox, "_probed", None)      # forget what an earlier test probed
        fresh.start()
        self.addCleanup(fresh.stop)

    def probe(self, result: str) -> mock.Mock:
        patcher = mock.patch.object(sandbox, "_probe", return_value=result)
        probe = patcher.start()
        self.addCleanup(patcher.stop)
        return probe

    def test_off_means_no_isolation_and_needs_no_probe_process(self):
        for value in ("off", "none", "0", " OFF "):
            with mock.patch.dict(os.environ, {"KGIRL_SANDBOX": value}), \
                    mock.patch.object(sandbox, "_works", side_effect=AssertionError("must not probe")):
                self.assertEqual(sandbox.level(), "none", value)

    def test_the_level_is_probed_once_per_setting(self):
        probe = self.probe("netns")
        with mock.patch.dict(os.environ, {"KGIRL_SANDBOX": "netns"}):
            self.assertEqual(sandbox.level(), "netns")
            self.assertEqual(sandbox.level(), "netns")
        probe.assert_called_once_with("netns")
        with mock.patch.dict(os.environ, {"KGIRL_SANDBOX": "bwrap"}):
            sandbox.level()
        self.assertEqual([call.args for call in probe.call_args_list], [("netns",), ("bwrap",)])

    def test_an_unknown_setting_means_auto_never_no_isolation(self):
        probe = self.probe("bwrap")
        with mock.patch.dict(os.environ, {"KGIRL_SANDBOX": "disabled-please"}):
            self.assertEqual(sandbox.level(), "bwrap")
        probe.assert_called_once_with("auto")

    def test_describe_names_each_level(self):
        self.assertEqual(sandbox.describe("none"), "resource limits only (no isolation)")
        self.assertEqual(sandbox.describe("netns"), "no network (files are not isolated)")
        self.assertIn("bubblewrap", sandbox.describe("bwrap"))
        self.assertEqual(set(sandbox.LEVELS), {"bwrap", "netns", "none"})

    def test_wrap_prefixes_the_command_for_the_level(self):
        argv = ["python3", "-c", "pass"]
        with mock.patch.object(sandbox, "level", return_value="none"):
            self.assertEqual(sandbox.wrap(argv, "/tmp/work"), argv)
        with mock.patch.object(sandbox, "level", return_value="netns"):
            wrapped = sandbox.wrap(argv, "/tmp/work")
        self.assertTrue(wrapped[0].endswith("unshare"), wrapped)
        self.assertEqual(wrapped[1:], ["-r", "-n", "--", *argv])

    def test_the_child_environment_is_fixed_and_carries_nothing_of_the_users(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "not-a-real-key", "PYTHONPATH": "/somewhere"}):
            env = sandbox.child_env("/tmp/work")
        self.assertEqual(sorted(env), ["HOME", "LANG", "PATH", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
                                       "PYTHONIOENCODING"])
        self.assertEqual((env["HOME"], env["PYTHONHASHSEED"]), ("/tmp/work", "0"))


class RunnerResultTests(unittest.TestCase):
    """`run_suite` and `imports_ok` with the child process replaced by its JSON payload."""

    def run_with(self, payload, why: str = ""):
        seen: dict = {}

        def fake_run(workdir, mode, timeout):
            seen["mode"] = mode
            seen["files"] = {p.name: p.read_text(encoding="utf-8") for p in Path(workdir).iterdir()}
            return payload, why

        patcher = mock.patch.object(sandbox, "_run", side_effect=fake_run)
        patcher.start()
        self.addCleanup(patcher.stop)
        return seen

    def test_passed_and_failed_ids(self):
        seen = self.run_with({"tests": {"test_gen0.TestStack.test_push": "ok", "test_gen0.TestStack.test_pop": "fail",
                                        "test_gen0.TestStack.test_skip": "skip"},
                              "output": {"test_gen0.TestStack.test_pop": "IndexError: pop from empty list"}})
        passed, failed = sandbox.run_suite("CODE = 1\n", "SUITE = 1\n", "test_gen0", timeout=7.0, per_test=2.5)
        self.assertEqual(passed, {"test_gen0:TestStack.test_push"})
        self.assertEqual(failed, {"test_gen0:TestStack.test_pop": "IndexError: pop from empty list"})
        self.assertEqual(seen["mode"], ["test", "test_gen0", "2.5"])
        self.assertEqual(seen["files"], {"solution.py": "CODE = 1\n", "test_gen0.py": "SUITE = 1\n"})

    def test_a_run_without_a_payload_fails_as_a_whole(self):
        self.run_with(None, "timed out after 20 s")
        self.assertEqual(sandbox.run_suite("x = 1\n", "y = 1\n", "test_gen1"), (set(), {"test_gen1:*": "timed out after 20 s"}))

    def test_a_run_without_a_payload_or_a_reason(self):
        self.run_with(None, "")
        self.assertEqual(sandbox.run_suite("x = 1\n", "y = 1\n", "test_gen1"),
                         (set(), {"test_gen1:*": "the suite did not run"}))

    def test_a_suite_that_cannot_be_imported(self):
        self.run_with({"tests": {}, "error": "ImportError: cannot import name 'mean' from 'solution'"})
        self.assertEqual(sandbox.run_suite("x = 1\n", "y = 1\n", "test_gen0"),
                         (set(), {"test_gen0:*": "ImportError: cannot import name 'mean' from 'solution'"}))

    def test_a_suite_without_tests(self):
        self.run_with({"tests": {}})
        self.assertEqual(sandbox.run_suite("x = 1\n", "y = 1\n", "test_gen0"),
                         (set(), {"test_gen0:*": "the suite defines no tests"}))

    def test_an_error_does_not_void_tests_that_passed(self):
        self.run_with({"tests": {"test_gen0.T.test_a": "ok"}, "error": "a later test crashed the runner"})
        self.assertEqual(sandbox.run_suite("x = 1\n", "y = 1\n", "test_gen0"), ({"test_gen0:T.test_a"}, {}))

    def test_a_malformed_payload_is_not_trusted(self):
        self.run_with({"tests": ["test_gen0.T.test_a"], "output": "all good"})
        self.assertEqual(sandbox.run_suite("x = 1\n", "y = 1\n", "test_gen0"),
                         (set(), {"test_gen0:*": "the suite defines no tests"}))

    def test_imports_ok(self):
        seen = self.run_with({"imported": True})
        self.assertEqual(sandbox.imports_ok("x = 1\n", timeout=3.0), (True, ""))
        self.assertEqual(seen["mode"], ["import", "solution"])
        self.assertEqual(seen["files"], {"solution.py": "x = 1\n"})

    def test_imports_ok_reports_the_error(self):
        self.run_with({"imported": False, "error": "NameError: name 'y' is not defined"})
        self.assertEqual(sandbox.imports_ok("x = y\n"), (False, "NameError: name 'y' is not defined"))

    def test_imports_ok_without_a_payload(self):
        self.run_with(None, "exit code 1\nSegmentation fault")
        self.assertEqual(sandbox.imports_ok("x = 1\n"), (False, "exit code 1\nSegmentation fault"))

    def test_run_suite_and_imports_ok_support_custom_module_name_and_extra_files(self):
        seen = self.run_with({"tests": {"existing_0.T.test_a": "ok"}})
        passed, failed = sandbox.run_suite(
            "VAL = 1\n",
            "SUITE = 1\n",
            "existing_0",
            module_name="calc",
            extra_files={"helper.py": "H = 2\n", "../escape.py": "BAD = 1\n"},
        )
        self.assertEqual((passed, failed), ({"existing_0:T.test_a"}, {}))
        self.assertEqual(seen["files"], {"calc.py": "VAL = 1\n", "helper.py": "H = 2\n", "existing_0.py": "SUITE = 1\n"})

        seen_imp = self.run_with({"imported": True})
        self.assertEqual(sandbox.imports_ok("VAL = 1\n", module_name="calc", extra_files={"helper.py": "H = 2\n"}),
                         (True, ""))
        self.assertEqual(seen_imp["mode"], ["import", "calc"])
        self.assertEqual(seen_imp["files"], {"calc.py": "VAL = 1\n", "helper.py": "H = 2\n"})

    def test_imports_all_ok_checks_every_module_together(self):
        self.assertEqual(sandbox.imports_all_ok({}), (False, "no Python modules to import"))
        seen_modes: list[list[str]] = []

        def fake_run(workdir, mode, timeout):
            seen_modes.append(mode)
            if mode == ["import", "api"]:
                return {"imported": False, "error": "ImportError: bad"}, ""
            return {"imported": True, "error": ""}, ""

        with mock.patch.object(sandbox, "_run", side_effect=fake_run):
            ok, err = sandbox.imports_all_ok({"core.py": "X = 1\n", "api.py": "from core import X\n"})
        self.assertFalse(ok)
        self.assertEqual(err, "api.py: ImportError: bad")
        self.assertEqual(seen_modes, [["import", "core"], ["import", "api"]])


if __name__ == "__main__":
    unittest.main()

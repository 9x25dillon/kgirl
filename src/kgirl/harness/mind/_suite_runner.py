"""Child-side runner for generated code. It is copied into the build sandbox and run there; stdlib only.

    python _kgirl_runner.py test <module> <nonce> [seconds per test]
    python _kgirl_runner.py import <module> <nonce>

It prints one line on stdout, `<nonce> <json>`:

    test     {"tests": {"mod.Class.test_x": "ok" | "fail" | "skip"}, "output": {id: traceback tail}, "error": ""}
    import   {"imported": true | false, "error": "..."}

The parent reads that line and nothing else, so it never parses unittest's human-readable report (whose
format changed in Python 3.11 and may change again). The nonce is random per run: output printed by the
code under test cannot be mistaken for the result. Limits are set here, in the child, because
`preexec_fn` is unsafe when the parent runs builds from several threads.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import signal
import sys
import traceback
import unittest

CPU_SECONDS = 30
MEMORY_BYTES = 2 << 30
FILE_BYTES = 16 << 20
OUTPUT_CHARS = 1500


def set_limits() -> None:
    try:
        import resource
    except ImportError:          # not POSIX: the wall-clock timeout in the parent is the only limit
        return
    for name, value in (("RLIMIT_CPU", CPU_SECONDS), ("RLIMIT_AS", MEMORY_BYTES), ("RLIMIT_FSIZE", FILE_BYTES)):
        try:
            resource.setrlimit(getattr(resource, name), (value, value))
        except (AttributeError, ValueError, OSError):
            pass


class _TestTimeout(Exception):
    pass


class Collector(unittest.TestResult):
    """Records one status per test method. A failing subtest fails its test; a skip is neither pass nor fail."""

    def __init__(self, per_test: float = 0.0):
        super().__init__()
        self.status: dict[str, str] = {}
        self.output: dict[str, str] = {}
        self.per_test = per_test if hasattr(signal, "setitimer") else 0.0

    def startTest(self, test):
        super().startTest(test)
        if self.per_test > 0:
            signal.setitimer(signal.ITIMER_REAL, self.per_test)

    def stopTest(self, test):
        if self.per_test > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
        super().stopTest(test)

    def _fail(self, test, text: str) -> None:
        self.status[test.id()] = "fail"
        self.output[test.id()] = text[-OUTPUT_CHARS:]

    def addSuccess(self, test):
        super().addSuccess(test)
        self.status.setdefault(test.id(), "ok")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._fail(test, self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._fail(test, self._exc_info_to_string(err, test))

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._fail(test, self._exc_info_to_string(err, subtest))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.status.setdefault(test.id(), "skip")

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.status.setdefault(test.id(), "skip")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._fail(test, "the test was marked as an expected failure but passed")


def run_tests(module_name: str, per_test: float = 0.0) -> dict:
    """Import a test module and run it. Never raises: a module that cannot be imported is reported in `error`."""
    try:
        module = importlib.import_module(module_name)
    except (Exception, SystemExit):   # SystemExit: a suite that calls unittest.main() at import time
        return {"tests": {}, "output": {}, "error": traceback.format_exc()[-OUTPUT_CHARS:]}
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = Collector(per_test)
    if result.per_test > 0:
        def on_alarm(signum, frame):
            raise _TestTimeout(f"the test ran longer than {per_test:g} s")
        signal.signal(signal.SIGALRM, on_alarm)
    try:
        suite.run(result)
    except (Exception, SystemExit):
        return {"tests": result.status, "output": result.output, "error": traceback.format_exc()[-OUTPUT_CHARS:]}
    return {"tests": result.status, "output": result.output, "error": ""}


def try_import(module_name: str) -> dict:
    try:
        importlib.import_module(module_name)
    except (Exception, SystemExit):
        return {"imported": False, "error": traceback.format_exc()[-OUTPUT_CHARS:]}
    return {"imported": True, "error": ""}


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] not in ("test", "import"):
        print("usage: _kgirl_runner.py test|import <module> <nonce> [seconds per test]", file=sys.stderr)
        return 2
    mode, module_name, nonce = argv[0], argv[1], argv[2]
    try:
        per_test = float(argv[3]) if len(argv) > 3 else 0.0
    except ValueError:
        per_test = 0.0
    set_limits()
    out = sys.stdout
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        payload = run_tests(module_name, per_test) if mode == "test" else try_import(module_name)
    out.write(f"{nonce} {json.dumps(payload)}\n")
    out.flush()
    return 0


if __name__ == "__main__":
    code = main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(code)   # the result is printed: don't wait for threads the code under test may have left running

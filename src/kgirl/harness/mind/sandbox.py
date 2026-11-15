"""Where model-written code runs: a temp dir, resource limits, and as much isolation as the machine offers.

    bwrap   bubblewrap: read-only system directories, a private /tmp, no network, no home directory, its
            own process space. The code sees the interpreter, the standard library and its temp dir.
    netns   `unshare -r -n`: no network. Files the user can read are still readable.
    none    resource limits and a scrubbed environment only: a guard against accidents, not a boundary.

`KGIRL_SANDBOX=auto|bwrap|netns|off` picks the level; `auto` (the default) probes once and uses the
strongest that works here. `level()` reports it, and callers that run code for someone other than the
person at the keyboard (the MCP `kgirl_build` tool) refuse to run below `bwrap` unless the user opts in.

Before anything runs, `lint()` reads the code's syntax tree and names what a pure module has no business
doing: network and process modules, `os.system` and friends, and (unless I/O is allowed) file access.
The builder drops candidates and suites that trip it. The check is static, so it is a filter for honest
mistakes, and the sandbox level is what actually confines the code.

Results come back from `_suite_runner.py` as one JSON line, never from unittest's printed report.
"""

from __future__ import annotations

import ast
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

LEVELS = ("bwrap", "netns", "none")
RUNNER_NAME = "_kgirl_runner.py"
_RUNNER_SOURCE = Path(__file__).with_name("_suite_runner.py")

_NETWORK = frozenset({"socket", "socketserver", "ssl", "http", "urllib", "urllib2", "urllib3", "requests", "httpx",
                      "aiohttp", "ftplib", "smtplib", "poplib", "imaplib", "telnetlib", "nntplib", "xmlrpc",
                      "websocket", "websockets", "paramiko"})
_PROCESS = frozenset({"subprocess", "multiprocessing", "ctypes", "cffi", "pty", "webbrowser"})
_FILES = frozenset({"shutil", "tempfile", "glob"})
_OS_PROCESS = frozenset({"system", "popen", "fork", "forkpty", "kill", "killpg", "startfile", "posix_spawn",
                         "posix_spawnp"})
_OS_FILES = frozenset({"remove", "unlink", "rmdir", "removedirs", "rename", "renames", "replace", "chmod", "chown",
                       "symlink", "link", "mkdir", "makedirs", "truncate", "open", "listdir", "scandir", "walk"})

_lock = threading.Lock()
_probed: tuple[str, str] | None = None     # (KGIRL_SANDBOX value it was probed for, level)


# ---------------------------------------------------------------------- static check

def lint(code: str, allow_io: bool = False) -> list[str]:
    """Reasons this module should not run as a pure, self-contained solution; empty when it is clean."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        return [f"does not parse: {exc}"]
    found: dict[str, None] = {}
    for node in ast.walk(tree):
        roots: list[str] = []
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots = [node.module.split(".")[0]]
        for root in roots:
            if root in _NETWORK:
                found.setdefault(f"imports {root} (network)")
            elif root in _PROCESS:
                found.setdefault(f"imports {root} (processes or native code)")
            elif root in _FILES and not allow_io:
                found.setdefault(f"imports {root} (files)")
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name):
            if fn.id == "__import__":
                found.setdefault("calls __import__")
            elif fn.id == "open" and not allow_io:
                found.setdefault("opens files")
        elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "os":
            if fn.attr in _OS_PROCESS or fn.attr.startswith(("exec", "spawn")):
                found.setdefault(f"calls os.{fn.attr}")
            elif fn.attr in _OS_FILES and not allow_io:
                found.setdefault(f"calls os.{fn.attr} (files)")
    return list(found)


# ---------------------------------------------------------------------- isolation

def python_exe() -> str:
    """The real interpreter (not a venv shim): generated code is stdlib-only, and a sandbox binds /usr, not a venv."""
    return os.path.realpath(sys.executable)


def _bwrap_prefix(bwrap: str, workdir: str) -> list[str]:
    args = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--ro-bind", "/usr", "/usr"]
    for path in ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32"):
        if os.path.islink(path):                      # merged-/usr systems: keep the links, they point into /usr
            args += ["--symlink", os.readlink(path), path]
        elif os.path.isdir(path):
            args += ["--ro-bind", path, path]
    for path in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/alternatives"):
        if os.path.exists(path):
            args += ["--ro-bind", path, path]
    for prefix in {os.path.realpath(sys.base_prefix), os.path.dirname(os.path.dirname(python_exe()))}:
        if prefix != "/" and prefix != "/usr" and not prefix.startswith("/usr/"):   # e.g. a uv- or pyenv-built Python
            args += ["--ro-bind", prefix, prefix]
    return args + ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--bind", workdir, workdir,
                   "--chdir", workdir]


def _works(argv: list[str], cwd: str) -> bool:
    try:
        return subprocess.run(argv, cwd=cwd, env=child_env(cwd), capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _probe(wanted: str) -> str:
    if wanted in ("off", "none", "0"):
        return "none"
    with tempfile.TemporaryDirectory(prefix="kgirl-probe-") as tmp:
        bwrap = shutil.which("bwrap")
        if bwrap and wanted in ("auto", "bwrap") and _works([*_bwrap_prefix(bwrap, tmp), python_exe(), "-c", "pass"],
                                                            tmp):
            return "bwrap"
        unshare = shutil.which("unshare")
        if unshare and wanted in ("auto", "netns") and _works([unshare, "-r", "-n", "--", python_exe(), "-c", "pass"],
                                                              tmp):
            return "netns"
    return "none"


def level() -> str:
    """The isolation generated code gets on this machine: "bwrap", "netns" or "none" (probed once per setting)."""
    global _probed
    wanted = os.environ.get("KGIRL_SANDBOX", "auto").strip().lower()
    if wanted not in ("bwrap", "netns", "off", "none", "0"):
        wanted = "auto"          # an unknown value must not silently mean "no isolation"
    with _lock:
        if _probed is None or _probed[0] != wanted:
            _probed = (wanted, _probe(wanted))
        return _probed[1]


def describe(lvl: str | None = None) -> str:
    return {"bwrap": "bubblewrap (no network, no home directory, read-only system)",
            "netns": "no network (files are not isolated)",
            "none": "resource limits only (no isolation)"}[lvl or level()]


def wrap(argv: list[str], workdir: str) -> list[str]:
    """`argv` prefixed so it runs at the current isolation level, with `workdir` as its only writable place."""
    lvl = level()
    if lvl == "bwrap":
        return [*_bwrap_prefix(shutil.which("bwrap") or "bwrap", workdir), *argv]
    if lvl == "netns":
        return [shutil.which("unshare") or "unshare", "-r", "-n", "--", *argv]
    return list(argv)


def child_env(home: str) -> dict[str, str]:
    """A fixed, small environment: nothing of the user's (keys, tokens, PYTHON* settings) reaches the code."""
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": home, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}


# ---------------------------------------------------------------------- running

def _run(workdir: str, mode: list[str], timeout: float) -> tuple[dict | None, str]:
    """Run the child runner in `workdir`. Returns (its JSON payload, "") or (None, why there is none)."""
    Path(workdir, RUNNER_NAME).write_text(_RUNNER_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    nonce = "kgirl-" + secrets.token_hex(8)
    argv = wrap([python_exe(), "-s", "-B", RUNNER_NAME, mode[0], mode[1], nonce, *mode[2:]], workdir)
    try:
        proc = subprocess.run(argv, cwd=workdir, env=child_env(workdir), capture_output=True, text=True,
                              timeout=timeout)
        out, err, why = proc.stdout, proc.stderr, f"exit code {proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        out, err, why = _text(exc.stdout), _text(exc.stderr), f"timed out after {timeout:g} s"
    except OSError as exc:
        return None, f"could not start: {exc}"
    for line in reversed(out.splitlines()):
        if line.startswith(nonce + " "):
            try:
                payload = json.loads(line[len(nonce) + 1:])
            except ValueError:
                break
            if isinstance(payload, dict):
                return payload, ""
    return None, f"{why}\n{(err or out)[-1200:]}".strip()


def _text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def run_suite(code: str, suite: str, name: str, timeout: float = 20.0, per_test: float = 5.0
              ) -> tuple[set[str], dict[str, str]]:
    """Run one test module against one implementation.

    Returns (ids that passed, {id that failed: output}). Ids look like `test_gen0:TestStack.test_push`.
    A suite that cannot be imported, or a run that dies or times out, fails as a whole under `<name>:*`.
    """
    with tempfile.TemporaryDirectory(prefix="kgirl-build-") as tmp:
        Path(tmp, "solution.py").write_text(code, encoding="utf-8")
        Path(tmp, f"{name}.py").write_text(suite, encoding="utf-8")
        payload, why = _run(tmp, ["test", name, f"{per_test:g}"], timeout)
    if payload is None:
        return set(), {f"{name}:*": why or "the suite did not run"}
    passed: set[str] = set()
    failed: dict[str, str] = {}
    outputs = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    tests = payload.get("tests") if isinstance(payload.get("tests"), dict) else {}
    for raw_id, status in tests.items():
        module, _, rest = str(raw_id).partition(".")
        tid = f"{module}:{rest}"
        if status == "ok":
            passed.add(tid)
        elif status == "fail":
            failed[tid] = str(outputs.get(raw_id, ""))
    if payload.get("error") and not passed:
        failed[f"{name}:*"] = str(payload["error"])
    elif not tests:
        failed[f"{name}:*"] = "the suite defines no tests"
    return passed, failed


def imports_ok(code: str, timeout: float = 20.0) -> tuple[bool, str]:
    """Does `solution.py` import cleanly inside the sandbox? Returns (ok, error text)."""
    with tempfile.TemporaryDirectory(prefix="kgirl-build-") as tmp:
        Path(tmp, "solution.py").write_text(code, encoding="utf-8")
        payload, why = _run(tmp, ["import", "solution"], timeout)
    if payload is None:
        return False, why
    return bool(payload.get("imported")), str(payload.get("error", ""))

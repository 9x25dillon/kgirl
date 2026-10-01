#!/usr/bin/env python3
"""Blast-radius lint for MCP servers (Python and JavaScript/TypeScript). Standard library only.

Over MCP the caller is a model, and possibly a prompt-injected one. These rules find the
shapes that let such a caller reach further than the user intended:

  BR001  a tool argument chooses a program to run (or is handed on as a command)
  BR002  a child process inherits the full environment (API keys, tokens)
  BR003  model input is stored as live state, or attributed to the user
  BR004  initialize echoes the client's protocol version instead of negotiating
  BR005  a file is written at a model-chosen path with no confinement check

It is a lint. It finds likely problems by their shape; a clean run is not proof of safety.
See reference/blast_radius_review.md for each rule's fix.

Usage:  python blast_radius_check.py PATH [PATH ...] [--json] [--include-tests]
Exit:   1 when anything is reported, 0 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "env", "dist", "build", "site-packages"}
TEST_DIRS = {"tests", "test", "__tests__"}
PY_EXT = {".py"}
JS_EXT = {".js", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}

# Parameter names of dict-dispatch tool handlers. Not `args`: that is usually argparse, i.e. a person.
# Not `payload` or `inputs` either: too common in ordinary data code to mean "tool handler".
HANDLER_PARAMS = {"p", "params", "arguments", "tool_input", "tool_args"}
NOT_TAINTED = {"self", "cls", "ctx", "context"}
CMD_WORDS = re.compile(r"^(verify|verifier|cmd|command|commands|script|shell|exec|program|argv|run_cmd|executable)$", re.I)
SUBPROCESS = {
    ("subprocess", "run"), ("subprocess", "call"), ("subprocess", "check_call"), ("subprocess", "check_output"),
    ("subprocess", "Popen"), ("os", "system"), ("os", "popen"), ("os", "execv"), ("os", "execvp"), ("os", "execve"),
    ("os", "spawnv"), ("os", "spawnvp"), ("asyncio", "create_subprocess_exec"), ("asyncio", "create_subprocess_shell"),
}
# Calls that only inspect or reshape a command; handing it onward is what matters.
BENIGN = {("isinstance",), ("len",), ("str",), ("repr",), ("type",), ("bool",), ("list",), ("tuple",), ("print",),
          ("shlex", "split"), ("shlex", "quote"), ("shlex", "join")}
INSPECT_MODULES = {("re",), ("fnmatch",), ("json",), ("posixpath",)}  # parsing a command is not running it
STORE_METHODS = {"add", "insert", "append", "remember", "save", "put", "upsert", "store", "write", "set", "create", "record"}
STORE_RECEIVER = re.compile(r"mem|soup|store|db|note|kv|cache|vault|journal|knowledge|recall", re.I)
REVIEW_STATES = {"staged", "pending", "draft", "review", "quarantine", "proposed", "unreviewed"}
WRITE_CALLS = {("shutil", n) for n in ("copy", "copy2", "copyfile", "copytree", "move", "rmtree")} | \
    {("os", n) for n in ("remove", "unlink", "rename", "replace", "makedirs", "mkdir", "rmdir")}
PATH_WRITE_METHODS = {"write_text", "write_bytes", "mkdir", "unlink", "rename", "replace", "touch", "rmdir"}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {self.message}"


# ---------------------------------------------------------------- python

def _dotted(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _is_handler(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for d in fn.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if _dotted(target)[-1:] == ("tool",):
            return True
    return any(a.arg in HANDLER_PARAMS for a in fn.args.args)


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _cmd_keyed(node: ast.AST) -> bool:
    """`p["verify"]`, `p.get("cmd")` and similar: a value the caller names as a command."""
    for n in ast.walk(node):
        key = None
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant):
            key = n.slice.value
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args \
                and isinstance(n.args[0], ast.Constant):
            key = n.args[0].value
        if isinstance(key, str) and CMD_WORDS.match(key):
            return True
    return False


class _Function:
    """Taint within one handler: parameters, and names assigned from them (to a fixpoint)."""

    def __init__(self, fn: ast.FunctionDef | ast.AsyncFunctionDef):
        self.fn = fn
        params = [a.arg for a in fn.args.args + fn.args.kwonlyargs if a.arg not in NOT_TAINTED]
        self.tainted = set(params)
        self.cmd = {p for p in params if CMD_WORDS.match(p)}
        assigns = [n for n in ast.walk(fn) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr))]
        changed = True
        while changed:
            changed = False
            for a in assigns:
                value = a.value
                if value is None:
                    continue
                targets = a.targets if isinstance(a, ast.Assign) else [a.target]
                names = set().union(*(_names(t) for t in targets))
                if _names(value) & self.tainted and not names <= self.tainted:
                    self.tainted |= names
                    changed = True
                if (_cmd_keyed(value) or _names(value) & self.cmd) and not names <= self.cmd:
                    self.cmd |= names
                    changed = True
        # A membership test (`x in ALLOWED`, `x not in allowlist()`) on a command value guards it.
        self.guarded = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops):
                self.guarded |= _names(n.left)
        self.text = ast.unparse(fn)

    def taints(self, node: ast.AST) -> bool:
        return bool(_names(node) & self.tainted)

    def command(self, node: ast.AST) -> bool:
        names = _names(node)
        return (bool(names & self.cmd) or _cmd_keyed(node)) and not (names & self.guarded)


def _env_is_full(node: ast.AST) -> bool:
    if _dotted(node) == ("os", "environ"):
        return True
    if isinstance(node, ast.Dict) and any(k is None and _dotted(v) == ("os", "environ") for k, v in zip(node.keys, node.values)):
        return True
    if isinstance(node, ast.Call) and (_dotted(node.func) == ("os", "environ", "copy")
                                       or (_dotted(node.func) == ("dict",) and node.args and _dotted(node.args[0]) == ("os", "environ"))):
        return True
    return False


def _reads_protocol_version(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and n.value == "protocolVersion":
            return True
    return False


def check_python(path: Path, text: str) -> list[Finding]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [Finding(str(path), exc.lineno or 1, "BR000", f"not parsed: {exc.msg}")]
    out: list[Finding] = []
    add = lambda node, rule, msg: out.append(Finding(str(path), getattr(node, "lineno", 1), rule, msg))  # noqa: E731

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _dotted(node.func)[-2:] in SUBPROCESS:
            for kw in node.keywords:
                if kw.arg == "env" and _env_is_full(kw.value):
                    add(node, "BR002", "child process gets the full environment; pass only the variables it needs")
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "protocolVersion" and _reads_protocol_version(v) \
                        and not (isinstance(v, ast.IfExp) and isinstance(v.test, ast.Compare)):
                    add(node, "BR004", "initialize echoes the client's protocolVersion; answer with a supported one")

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not _is_handler(fn):
            continue
        f = _Function(fn)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted(node.func)
            args = list(node.args) + [kw.value for kw in node.keywords]
            if dotted[-2:] in SUBPROCESS:
                first = node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == "args"), None)
                shell = dotted[-1] in {"system", "popen", "create_subprocess_shell"} or any(
                    kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in node.keywords)
                fixed = isinstance(first, (ast.List, ast.Tuple)) and first.elts and isinstance(first.elts[0], ast.Constant)
                chosen = first is not None and f.taints(first) and (shell or not fixed) and not (_names(first) & f.guarded)
                if chosen:
                    add(node, "BR001", f"{'.'.join(dotted)}() runs a program the caller chose; use a fixed allowlist")
            elif dotted not in BENIGN and dotted[:1] not in INSPECT_MODULES \
                    and not (isinstance(node.func, ast.Attribute) and f.command(node.func.value)) \
                    and any(f.command(a) for a in args):
                add(node, "BR001", f"a model-supplied command is passed to {'.'.join(dotted) or 'a call'}(); accept only allowlisted commands")
            if isinstance(node.func, ast.Attribute) and node.func.attr in STORE_METHODS \
                    and (STORE_RECEIVER.search(ast.unparse(node.func.value)) or node.func.attr in {"remember", "store"}) \
                    and any(f.taints(a) for a in node.args):
                state = {kw.arg: kw.value.value for kw in node.keywords
                         if kw.arg in {"status", "state"} and isinstance(kw.value, ast.Constant)}
                if not any(str(v).lower() in REVIEW_STATES for v in state.values()):
                    add(node, "BR003", f"model input stored live via .{node.func.attr}(); stage it for review")
                for kw in node.keywords:
                    if kw.arg in {"source", "author", "provenance", "origin"} and isinstance(kw.value, ast.Constant) \
                            and str(kw.value.value).lower() == "user":
                        add(node, "BR003", "model input attributed to the user; record the real source (for example \"mcp\")")
            writes = (
                (dotted in {("open",), ("io", "open")} and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                 and re.search(r"[wax+]", str(node.args[1].value)) and f.taints(node.args[0]))
                or (dotted[-2:] in WRITE_CALLS and node.args and f.taints(node.args[0]))
                or (isinstance(node.func, ast.Attribute) and node.func.attr in PATH_WRITE_METHODS and f.taints(node.func.value))
            )
            confined = re.search(r"resolve\(|realpath\(", f.text) and re.search(r"is_relative_to|parents|commonpath|startswith", f.text)
            if writes and not confined:
                add(node, "BR005", "writes to a path the caller chose; resolve it and require it inside an allowed root")
    return out


# ---------------------------------------------------------------- javascript / typescript

JS_RULES = [
    ("BR001", re.compile(r"\b(exec|execSync|spawn|spawnSync|execFile|execFileSync|fork)\s*\(([^)]*\b(args|params|input|request|arguments)\b|[^)]*\$\{)"),
     "child process built from tool arguments; use a fixed allowlist", True),
    ("BR002", re.compile(r"env\s*:\s*process\.env\b|\.\.\.process\.env\b"), "child process gets the full environment", True),
    ("BR004", re.compile(r"protocolVersion\s*:\s*[\w.?\[\]'\"]*params[\w.?\[\]'\"]*protocolVersion(?![^\n]*includes)"),
     "initialize echoes the client's protocolVersion; answer with a supported one", False),
    ("BR005", re.compile(r"\b(writeFile|writeFileSync|appendFile|appendFileSync|rm|rmSync|unlink|unlinkSync)\s*\(\s*(args|params|input)\b"),
     "writes to a path the caller chose; resolve it and require it inside an allowed root", False),
]


def check_js(path: Path, text: str) -> list[Finding]:
    spawns = bool(re.search(r"child_process", text))
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("//", "*")):
            continue
        for rule, rx, msg, needs_spawn in JS_RULES:
            if (spawns or not needs_spawn) and rx.search(line):
                out.append(Finding(str(path), i, rule, msg))
    return out


# ---------------------------------------------------------------- driver

def files(paths: list[str], include_tests: bool) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            found.append(p)
            continue
        for f in sorted(p.rglob("*")):
            parts = set(f.relative_to(p).parts[:-1])
            if f.is_file() and f.suffix in PY_EXT | JS_EXT and not parts & SKIP_DIRS and (include_tests or not parts & TEST_DIRS):
                found.append(f)
    return found


def check(paths: list[str], include_tests: bool = False) -> list[Finding]:
    out: list[Finding] = []
    for f in files(paths, include_tests):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out += check_python(f, text) if f.suffix in PY_EXT else check_js(f, text)
    return sorted(set(out), key=lambda x: (x.path, x.line, x.rule))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", action="store_true", help="print findings as JSON")
    ap.add_argument("--include-tests", action="store_true", help="also scan tests/ directories")
    a = ap.parse_args(argv)
    found = check(a.paths, a.include_tests)
    if a.json:
        print(json.dumps([asdict(x) for x in found], indent=2))
    else:
        for x in found:
            print(x.render())
        print(f"{len(found)} finding(s)", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())

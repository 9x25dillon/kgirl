"""Text surgery on generated unittest suites: keep the tests that agreed, and show several suites as one.

A generated suite is a guess at the spec. After the build, the tests the chosen implementation passes are
the ones the spec, the tests and the code agree on; the rest are a wrong test or a real defect, and the
build report says which ids those were. `prune()` cuts a suite down to the agreed tests, so the file
written beside the module passes as delivered and documents what was actually checked.
"""

from __future__ import annotations

import ast
import re


def prune(suite: str, name: str, keep: set[str] | frozenset[str]) -> str:
    """The suite with only the test methods whose ids (`name:Class.method`) are in `keep`; "" when none survive.

    Methods are cut by line range, so comments and formatting of what stays are untouched. A class left
    without any test method is cut whole. If the result does not compile, the suite is returned unchanged:
    a noisy test file is better than a broken one.
    """
    try:
        tree = ast.parse(suite)
    except (SyntaxError, ValueError):
        return suite
    cut: list[tuple[int, int]] = []
    kept = 0
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        tests = [fn for fn in cls.body if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and fn.name.startswith("test")]
        if not tests:
            continue                                  # a helper or base class: leave it alone
        stay = [fn for fn in tests if f"{name}:{cls.name}.{fn.name}" in keep]
        kept += len(stay)
        if not stay:
            cut.append(_span(cls))
        else:
            cut.extend(_span(fn) for fn in tests if fn not in stay)
    if not kept:
        return ""
    if not cut:
        return suite
    drop = {n for lo, hi in cut for n in range(lo, hi + 1)}
    out = "\n".join(line for n, line in enumerate(suite.split("\n"), 1) if n not in drop)
    out = re.sub(r"\n{4,}", "\n\n\n", out)
    try:
        compile(out, f"{name}.py", "exec")
    except (SyntaxError, ValueError):
        return suite
    return out


def _span(node) -> tuple[int, int]:
    """First and last line of a class or function, decorators included."""
    first = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
    return first, node.end_lineno or node.lineno


def merge_suites(suites: list[str]) -> str:
    """Several suites as one readable text, each under its own banner. For display: run them as separate
    files, because two suites may define a class or helper with the same name."""
    return "\n\n".join(f"# ---- generated suite {i + 1} (written from the spec, before any code) ----\n{s.strip()}\n"
                       for i, s in enumerate(s for s in suites if s.strip()))


def retarget(suite: str, module: str) -> str:
    """Point a suite written against `solution` at the module name the code was saved under."""
    if module == "solution":
        return suite
    suite = re.sub(r"^([ \t]*)from solution import", rf"\1from {module} import", suite, flags=re.M)
    suite = re.sub(r"^([ \t]*)import solution[ \t]*$", rf"\1import {module} as solution", suite, flags=re.M)
    return re.sub(r"^([ \t]*)import solution as (\w+)[ \t]*$", rf"\1import {module} as \2", suite, flags=re.M)

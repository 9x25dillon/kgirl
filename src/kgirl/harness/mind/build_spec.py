"""Spec contract extraction, suite diversification, AST assertion splitting, and exemplar ranking.

A small local model writing tests and implementations from a spec makes two recurring mistakes:
it writes three near-identical happy-path test suites that pack several assertions into one method
(so one bad assertion hides four good ones), and it overlooks explicit contract details in the spec
(`@property`, non-mutation of inputs, round-trip invariants, stated exception types).

This module keeps `builder.py` small by handling the static spec/AST work:

    spec_checklist      pull stated signatures, @property names, exceptions, complexity and non-mutation
                        rules out of the spec text as a compact checklist
    suite_prompt        give each generated suite a distinct testing angle (direct, boundaries/errors,
                        contracts/invariants) once i > 0
    candidate_prompt    attach the contract checklist to later candidate samples (i >= 2)
    normalize_suite     strip unguarded top-level unittest.main(), wrap bare top-level test_* functions
                        into a TestCase, and split stateless multi-assertion test methods into atomic
                        test_<name>_1, test_<name>_2 methods ("each assert becomes a separate test")
    has_tests           whether a suite AST defines at least one runnable TestCase test_* method
    test_method_source  pull the source of one failing test method (`test_gen0:Class.test_x`) for repair
    format_failures     render failing (or regressed) tests with both their traceback and test source
    spec_query          strip generic Python/spec boilerplate words before querying Soup for exemplars
    rank_exemplars      keep and rank only recalled exemplars that share domain words with the spec
"""

from __future__ import annotations

import ast
import difflib
import re
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

from ..atlas.secrets import secret_reason
from ..util import split_identifier
from .suites import retarget
from .text import strip_reasoning

__all__ = [
    "SUITE_ANGLES",
    "apply_unified_diff",
    "atlas_context",
    "atlas_search_context",
    "build_short_summary",
    "build_summary_dict",
    "candidate_prompt",
    "donor_snippet",
    "edit_candidate_prompt",
    "edit_suite_prompt",
    "emit_files",
    "expected_exports",
    "explain_build",
    "extract_diff",
    "find_existing_tests",
    "format_failures",
    "fragment_utility",
    "has_tests",
    "lint_spec",
    "missing_exports",
    "multifile_candidate_prompt",
    "multifile_suite_prompt",
    "normalize_suite",
    "parse_multifile_reply",
    "parse_multifile_spec",
    "rank_exemplars",
    "select_exemplars_within_budget",
    "spec_checklist",
    "spec_query",
    "strip_numbered_lines",
    "suite_prompt",
    "suite_test_ids",
    "test_method_source",
    "test_weights",
    "trim_traceback",
    "unified_diff",
    "weighted_test_count",
]

_SIG_RE = re.compile(r"`((?:class\s+)?[A-Za-z_]\w*\s*\([^`\n]*\)(?:\s*->\s*[^`\n]+)?)`")
_CLASS_SPEC_RE = re.compile(r"\bclass\s+`?([A-Za-z_]\w*)\b|`class\s+([A-Za-z_]\w*)\b")
_WRITE_FN_RE = re.compile(r"\b(?:Write|Implement|Define)\s+`?([A-Za-z_]\w*)\s*\(", re.I)
_PROP_RE = re.compile(r"`([A-Za-z_]\w*)`\s+property\b|\bproperty\s+`([A-Za-z_]\w*)`", re.I)
_EXC_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception))\b")
_MUT_RE = re.compile(
    r"\b(?:must\s+not|do\s+not|don't|never|without)\s+(?:be\s+)?(?:modif(?:y|ied|ying)|mutat(?:e|ed|ing))\b",
    re.I,
)
_INV_RE = re.compile(r"\b(?:inverts?|round[- ]?trip|inverse\s+of)\b", re.I)
_BIGO_RE = re.compile(r"\bO\([1nNlogkK\s*+^0-9-]{1,18}\)")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_BOILERPLATE = frozenset(
    """
    a an and are as at be by can class code def dict do does each else empty error example examples
    exception false file float for from function given handle how if implement implementation in input
    inputs int integer is it its list method module must name named names never no none not number
    numbers object objects of on one only optional or other out output outputs parameter parameters
    pass python raise raises return returned returning returns same set should signature solution spec
    specification state stated states stdlib str string strings such take takes test tests that the
    their them then there these they this throw throws to true tuple type types typing union unless
    use used uses using value values valueerror typeerror keyerror indexerror runtimeerror when where
    which while with without write
    """.split()
)

SUITE_ANGLES = (
    "Focus this suite on boundary conditions (empty inputs, singletons, zero, multi-digit or rectangular "
    "shapes, negative or zero sizes) and every error or invalid state the SPEC says must raise an exception. "
    "Put one assertion or assertRaises block per test_* method.",
    "Focus this suite on the SPEC's exact contract and invariants: worked examples from the SPEC, tie-breaking "
    "and ordering rules, state or @property updates, round-trip/inverse identities, and checking that input "
    "arguments are not mutated in place. Put one check per test_* method.",
    "Focus this suite on tricky interactions and corner cases: repeated elements, off-by-one indices, "
    "inputs that nearly satisfy a rule but fail on the last step, and multi-step state transitions. "
    "Put one check per test_* method.",
)


# ---------------------------------------------------------------------- spec analysis

def spec_checklist(spec: str) -> str:
    """Extract explicit contract items stated in `spec` as a short checklist; "" when none are found."""
    items: list[str] = []
    sigs = list(dict.fromkeys(m.group(1).strip() for m in _SIG_RE.finditer(spec or "")))
    if sigs:
        items.append("Signatures to match exactly: " + ", ".join(f"`{s}`" for s in sigs[:6]))
    props = list(dict.fromkeys((m.group(1) or m.group(2)).strip() for m in _PROP_RE.finditer(spec or "")))
    if props:
        names = ", ".join(f"`{p}`" for p in props[:4])
        items.append(f"Properties (define with @property; access as obj.prop, never obj.prop()): {names}")
    excs = list(dict.fromkeys(m.group(1) for m in _EXC_RE.finditer(spec or "")))
    if excs:
        items.append("Exceptions to raise on invalid input/state: " + ", ".join(excs[:4]))
    if _MUT_RE.search(spec or ""):
        items.append("Do not mutate input arguments in place.")
    if _INV_RE.search(spec or ""):
        items.append("Verify inverse / round-trip consistency between paired operations.")
    bigos = list(dict.fromkeys(m.group(0).strip() for m in _BIGO_RE.finditer(spec or "")))
    if bigos:
        items.append("Complexity requirement: " + ", ".join(bigos[:3]))
    if not items:
        return ""
    return "CONTRACT CHECKLIST:\n" + "\n".join(f"- {it}" for it in items)


def expected_exports(spec: str) -> tuple[list[str], list[str], list[str]]:
    """Return `(top_level_names, class_methods, properties)` explicitly stated in `spec`."""
    text = spec or ""
    classes = list(dict.fromkeys((m.group(1) or m.group(2)).strip() for m in _CLASS_SPEC_RE.finditer(text)))
    sig_names: list[str] = []
    for m in _SIG_RE.finditer(text):
        raw = m.group(1).strip()
        if raw.startswith("class "):
            cname = raw[len("class ") :].partition("(")[0].strip()
            if cname and cname not in classes:
                classes.append(cname)
        else:
            fname = raw.partition("(")[0].strip()
            if fname and fname not in sig_names:
                sig_names.append(fname)
    if not classes and not sig_names:
        for m in _WRITE_FN_RE.finditer(text):
            fname = m.group(1).strip()
            if fname and fname != "class" and fname not in sig_names:
                sig_names.append(fname)
    props = list(dict.fromkeys((m.group(1) or m.group(2)).strip() for m in _PROP_RE.finditer(text)))
    if classes:
        methods = [n for n in sig_names if n not in classes]
        return classes, methods, props
    return sig_names, [], props


def _iter_block_stmts(stmts: Iterable[ast.stmt]) -> Iterable[ast.stmt]:
    """Yield statements at the current scope, descending into `if`, `try`, and `with` blocks."""
    try_types = (ast.Try, getattr(ast, "TryStar", ast.Try))
    for stmt in stmts:
        yield stmt
        if isinstance(stmt, ast.If):
            yield from _iter_block_stmts(stmt.body)
            yield from _iter_block_stmts(stmt.orelse)
        elif isinstance(stmt, try_types):
            yield from _iter_block_stmts(stmt.body)
            for handler in getattr(stmt, "handlers", ()):
                yield from _iter_block_stmts(handler.body)
            yield from _iter_block_stmts(stmt.orelse)
            yield from _iter_block_stmts(stmt.finalbody)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            yield from _iter_block_stmts(stmt.body)


def _bound_names(stmt: ast.stmt) -> Iterable[str]:
    """Names bound by a definition, assignment, annotated assignment, or import statement."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield stmt.name
    elif isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            for node in ast.walk(target):
                if isinstance(node, ast.Name):
                    yield node.id
    elif isinstance(stmt, ast.AnnAssign):
        for node in ast.walk(stmt.target):
            if isinstance(node, ast.Name):
                yield node.id
    elif isinstance(stmt, ast.Import):
        for alias in stmt.names:
            yield alias.asname or alias.name.partition(".")[0]
    elif isinstance(stmt, ast.ImportFrom):
        for alias in stmt.names:
            if alias.name != "*":
                yield alias.asname or alias.name


def missing_exports(spec: str, code: str) -> list[str]:
    """Static interface mismatches between `spec` and `code` (missing top-level names, methods, or @property)."""
    top_req, method_req, prop_req = expected_exports(spec)
    if not (top_req or method_req or prop_req):
        return []
    try:
        tree = ast.parse(code or "")
    except (SyntaxError, ValueError):
        return ["does not parse"]
    top_defined: set[str] = set()
    methods_defined: set[str] = set()
    props_defined: set[str] = set()
    for node in _iter_block_stmts(tree.body):
        top_defined.update(_bound_names(node))
        if isinstance(node, ast.ClassDef):
            for item in _iter_block_stmts(node.body):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods_defined.add(item.name)
                    if any(
                        (isinstance(d, ast.Name) and d.id in ("property", "cached_property"))
                        or (isinstance(d, ast.Attribute) and d.attr in ("getter", "cached_property"))
                        for d in item.decorator_list
                    ):
                        props_defined.add(item.name)
                elif isinstance(item, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)):
                    names = list(_bound_names(item))
                    methods_defined.update(names)
                    val = getattr(item, "value", None)
                    if isinstance(val, ast.Call) and (
                        (isinstance(val.func, ast.Name) and val.func.id in ("property", "cached_property"))
                        or (isinstance(val.func, ast.Attribute) and val.func.attr in ("property", "cached_property"))
                    ):
                        props_defined.update(names)
    issues: list[str] = []
    for name in top_req:
        if name not in top_defined:
            issues.append(f"missing top-level `{name}`")
    for name in method_req:
        if name not in methods_defined and name not in top_defined:
            issues.append(f"missing method `{name}`")
    for prop in prop_req:
        if prop not in props_defined:
            if prop in methods_defined:
                issues.append(f"`{prop}` is defined as a method without @property")
            else:
                issues.append(f"missing @property `{prop}`")
    return issues


def suite_prompt(spec: str, i: int) -> str:
    """Prompt for the i-th generated test suite. Suite 0 stays the direct SPEC prompt; later suites
    diversify across boundary/error, contract/invariant, and corner-case angles with the checklist."""
    base = f"SPEC:\n{spec}"
    if i <= 0:
        return base
    angle = SUITE_ANGLES[(i - 1) % len(SUITE_ANGLES)]
    checklist = spec_checklist(spec)
    return f"{base}\n\n{angle}" + (f"\n\n{checklist}" if checklist else "")


def candidate_prompt(spec: str, shots: str, i: int, edge_first: str) -> str:
    """Prompt for the i-th implementation sample. Samples 0 and 1 keep the canonical direct / edge-first
    prompts; samples >= 2 also attach the extracted contract checklist when the spec has one."""
    style = "edge-first" if i % 2 else "direct"
    prompt = (f"{shots}\n\n" if shots else "") + f"SPEC:\n{spec}" + (f"\n\n{edge_first}" if style == "edge-first" else "")
    if i >= 2:
        checklist = spec_checklist(spec)
        if checklist:
            prompt = f"{prompt}\n\n{checklist}"
    return prompt


# ---------------------------------------------------------------------- exemplar recall

def _domain_terms(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for w in _WORD_RE.findall(text or ""):
        for part in [w.lower(), *split_identifier(w)]:
            if len(part) > 1 and part not in _BOILERPLATE:
                seen.setdefault(part, None)
    return list(seen)


def spec_query(spec: str) -> str:
    """Domain terms from `spec` for Soup FTS recall, falling back to `spec` when none survive."""
    terms = _domain_terms(spec)
    return " ".join(terms[:16]) if len(terms) >= 2 else spec


def _spec_header(exemplar_text: str) -> str:
    head, _, _ = (exemplar_text or "").partition("\n\nSOLUTION:")
    return head[5:].strip() if head.startswith("SPEC:") else head.strip()


def fragment_utility(frag) -> float:
    """Smoothed Beta(1,1) utility `(wins + 1) / (wins + losses + 2)` from Soup's stored win/loss columns."""
    wins = float(getattr(frag, "wins", 0.0) or 0.0)
    losses = float(getattr(frag, "losses", 0.0) or 0.0)
    return (wins + 1.0) / (wins + losses + 2.0)


def rank_exemplars(spec: str, fragments: Sequence, limit: int = 2) -> list:
    """Keep and rank recalled exemplars that share domain concepts with `spec`."""
    if not fragments:
        return []
    target = set(_domain_terms(spec))
    if not target:
        return list(fragments[:limit])
    scored: list[tuple[float, int, object]] = []
    for idx, frag in enumerate(fragments):
        cand_terms = set(_domain_terms(_spec_header(getattr(frag, "text", ""))))
        shared = len(target & cand_terms)
        if shared == 0:
            continue
        overlap = shared / max(1, min(len(target), len(cand_terms)))
        utility = fragment_utility(frag)
        scored.append((0.7 * overlap + 0.3 * utility, -idx, frag))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [frag for _, _, frag in scored[:limit]]


# ---------------------------------------------------------------------- suite AST tools

def _span(node: ast.AST) -> tuple[int, int]:
    first = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
    return first, getattr(node, "end_lineno", None) or node.lineno


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        return body[1:]
    return list(body)


def _is_self_assert_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr.startswith(("assert", "fail"))
    )


def _has_mutation_or_control(stmt: ast.stmt) -> bool:
    for node in ast.walk(stmt):
        if isinstance(
            node,
            (
                ast.Store,
                ast.Del,
                ast.AugAssign,
                ast.AnnAssign,
                ast.NamedExpr,
                ast.Return,
                ast.Yield,
                ast.YieldFrom,
                ast.Global,
                ast.Nonlocal,
            ),
        ):
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and not node.func.attr.startswith(("assert", "fail"))
        ):
            return True
    return False


def _is_atomic_assert(stmt: ast.stmt) -> bool:
    """True when `stmt` is a self-contained assertion thatneither binds nor mutates local state."""
    if _has_mutation_or_control(stmt):
        return False
    if isinstance(stmt, ast.Assert):
        return True
    if isinstance(stmt, ast.Expr) and _is_self_assert_call(stmt.value):
        return True
    if isinstance(stmt, ast.With):
        return any(_is_self_assert_call(item.context_expr) for item in stmt.items)
    return False


def _stmt_source(code: str, stmt: ast.stmt) -> str:
    seg = ast.get_source_segment(code, stmt)
    if seg:
        col = getattr(stmt, "col_offset", 0)
        return textwrap.dedent((" " * col) + seg).strip()
    lo, hi = _span(stmt)
    lines = code.splitlines()[lo - 1 : hi]
    return textwrap.dedent("\n".join(lines)).strip()


def _is_bare_unittest_main(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    fn = stmt.value.func
    return (
        (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "unittest" and fn.attr == "main")
        or (isinstance(fn, ast.Name) and fn.id == "main")
    )


def _wrap_bare_test_functions(code: str, tree: ast.Module) -> str:
    """If a suite wrote top-level `def test_*():` functions with no TestCase class, wrap them in one."""
    has_cls_tests = any(
        isinstance(node, ast.ClassDef)
        and any(isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test") for m in node.body)
        for node in tree.body
    )
    if has_cls_tests:
        return code
    bare = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("test")
        and not node.decorator_list
        and len(node.args.args) == 0
    ]
    if not bare:
        return code
    drop = {ln for fn in bare for ln in range(_span(fn)[0], _span(fn)[1] + 1)}
    kept_lines = [line for n, line in enumerate(code.splitlines(), 1) if n not in drop]
    methods: list[str] = []
    for fn in bare:
        lo, hi = _span(fn)
        raw = textwrap.dedent("\n".join(code.splitlines()[lo - 1 : hi]))
        raw = re.sub(rf"^def\s+{re.escape(fn.name)}\s*\(\s*\)", f"def {fn.name}(self)", raw, count=1)
        methods.append(textwrap.indent(raw.rstrip(), "    "))
    header = "" if re.search(r"^\s*(?:import\s+unittest|from\s+unittest\s+import)\b", code, flags=re.M) else "import unittest\n"
    prefix = "\n".join(kept_lines).rstrip()
    wrapped = (f"{header}{prefix}\n\n\n" if prefix else f"{header}\n") + "class TestSpec(unittest.TestCase):\n" + "\n\n".join(methods) + "\n"
    try:
        compile(wrapped, "<suite>", "exec")
        return wrapped
    except (SyntaxError, ValueError):
        return code


def normalize_suite(code: str) -> str:
    """Clean up a generated unittest module and split stateless multi-assert test methods into atomic ones.

    Single-assertion methods and stateful multi-step methods are returned untouched.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code

    # 1. Strip unguarded top-level unittest.main() calls that would SystemExit on import.
    bare_main = {ln for s in tree.body if _is_bare_unittest_main(s) for ln in range(_span(s)[0], _span(s)[1] + 1)}
    if bare_main:
        cleaned = "\n".join(line for n, line in enumerate(code.splitlines(), 1) if n not in bare_main) + "\n"
        try:
            tree = ast.parse(cleaned)
            code = cleaned
        except (SyntaxError, ValueError):
            pass

    # 2. Wrap bare pytest-style `def test_*():` functions if no TestCase class exists.
    wrapped = _wrap_bare_test_functions(code, tree)
    if wrapped != code:
        try:
            tree = ast.parse(wrapped)
            code = wrapped
        except (SyntaxError, ValueError):
            pass

    # 3. Split stateless multi-assertion test methods so each assert is scored and pruned on its own.
    replacements: list[tuple[int, int, str]] = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        taken = {m.name for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("test") or fn.decorator_list:
                continue
            if len(fn.args.args) != 1 or fn.args.vararg or fn.args.kwarg or fn.args.defaults:
                continue
            stmts = _strip_docstring(fn.body)
            if len(stmts) < 2 or not all(_is_atomic_assert(s) for s in stmts):
                continue
            pieces = [_stmt_source(code, s) for s in stmts]
            if not all(pieces):
                continue
            indent = " " * fn.col_offset
            body_indent = indent + "    "
            split_defs: list[str] = []
            seq = 1
            for piece in pieces:
                while f"{fn.name}_{seq}" in taken:
                    seq += 1
                name = f"{fn.name}_{seq}"
                taken.add(name)
                seq += 1
                split_defs.append(f"{indent}def {name}(self):\n{textwrap.indent(piece, body_indent)}")
            lo, hi = _span(fn)
            replacements.append((lo, hi, "\n\n".join(split_defs)))

    if not replacements:
        return code

    lines = code.splitlines()
    for lo, hi, block in sorted(replacements, key=lambda r: r[0], reverse=True):
        lines[lo - 1 : hi] = block.splitlines()
    out = "\n".join(lines) + ("\n" if code.endswith("\n") else "")
    try:
        compile(out, "<suite>", "exec")
    except (SyntaxError, ValueError):
        return code
    return out if has_tests(out) else code


def has_tests(code: str) -> bool:
    """Does `code` define at least one `test*` method inside a class?"""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return any(
        isinstance(cls, ast.ClassDef)
        and any(isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test") for fn in cls.body)
        for cls in tree.body
    )


def test_method_source(suites: Sequence[str], tid: str, suite_map: dict[str, str] | None = None) -> str:
    """Source code of the test method named by `tid` (`test_gen0:Class.test_x` or `existing_0:Class.test_x`); "" if not found."""
    mod, sep, rest = (tid or "").partition(":")
    if not sep or "." not in rest:
        return ""
    suite = ""
    if suite_map and mod in suite_map:
        suite = suite_map[mod]
    elif mod.startswith("test_gen"):
        try:
            idx = int(mod[len("test_gen") :])
        except ValueError:
            return ""
        if 0 <= idx < len(suites):
            suite = suites[idx]
    if not suite:
        return ""
    cls_name, _, fn_name = rest.partition(".")
    try:
        tree = ast.parse(suite)
    except (SyntaxError, ValueError):
        return ""
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == cls_name:
            for fn in cls.body:
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == fn_name:
                    lo, hi = _span(fn)
                    return textwrap.dedent("\n".join(suite.splitlines()[lo - 1 : hi])).strip()
    return ""


_TB_FRAME_RE = re.compile(r'^\s*File "([^"]+)", line (\d+), in (.+)$')
_TB_EXC_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Exit|Interrupt)\b")


def trim_traceback(tb: str, max_chars: int = 420) -> str:
    """Keep only the last relevant traceback frame and the final assertion/exception, capped to `max_chars`."""
    raw = (tb or "").strip()
    if not raw:
        return ""
    lines = raw.splitlines()
    frames: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        m = _TB_FRAME_RE.match(line)
        if m:
            src_line = lines[idx + 1] if idx + 1 < len(lines) and lines[idx + 1].startswith("    ") else ""
            frames.append((idx, m.group(1), src_line))
    if not frames:
        return raw if len(raw) <= max_chars else raw[-max_chars:].lstrip()

    # Prefer the last frame outside unittest / the sandbox runner so the model sees its own code or test call.
    non_internal = [f for f in frames if "/unittest/" not in f[1] and not f[1].endswith("_kgirl_runner.py")]
    chosen_idx, _, chosen_src = non_internal[-1] if non_internal else frames[-1]
    last_frame_idx, _, last_src = frames[-1]
    tail_start = last_frame_idx + (2 if last_src else 1)
    # Skip caret/pointer lines (Python 3.11+ `~~~^^^`) when locating the exception header.
    exc_lines = [ln for ln in lines[tail_start:] if ln.strip() and not set(ln.strip()) <= {"^", "~", " "}]
    if not exc_lines:
        for ln in reversed(lines):
            if _TB_EXC_RE.match(ln.strip()):
                exc_lines = [ln.strip()]
                break
    frame_lines = [lines[chosen_idx].strip()]
    if chosen_src.strip():
        frame_lines.append("  " + chosen_src.strip())
    body = "\n".join(frame_lines + exc_lines).strip()
    if len(body) <= max_chars:
        return body
    # Preserve the exception line(s) first, then fit as much of the frame as budget allows.
    exc_text = "\n".join(exc_lines).strip()
    if len(exc_text) >= max_chars - 20:
        return exc_text[: max_chars - 1].rstrip() + "…"
    head_budget = max_chars - len(exc_text) - 2
    head = "\n".join(frame_lines)[:head_budget].rstrip()
    return f"{head}\n{exc_text}" if head else exc_text


def format_failures(
    suites: Sequence[str],
    failures: dict[str, str],
    limit: int = 6,
    max_tokens: int = 700,
    suite_map: dict[str, str] | None = None,
) -> str:
    """Render failing tests for repair: exact failing test IDs, trimmed tracebacks (last frame + assertion),
    and test method source, capped by `max_tokens` (~4 chars/token)."""
    if not failures:
        return ""
    items = list(failures.items())[:limit]
    header = "FAILING TEST IDS: " + ", ".join(tid for tid, _ in items)
    char_budget = max(200, max_tokens * 4)
    blocks: list[str] = [header]
    used = len(header)
    for tid, out in items:
        trimmed = trim_traceback(out, max_chars=420)
        entry = f"--- {tid}\n{trimmed}"
        src = test_method_source(suites, tid, suite_map=suite_map)
        if src and src not in trimmed:
            entry += f"\n```python\n{src}\n```"
        if len(blocks) > 1 and used + len(entry) + 2 > char_budget:
            break
        blocks.append(entry)
        used += len(entry) + 2
    return "\n\n".join(blocks)


_SPLIT_STEM_RE = re.compile(r"^(.*?\btest_[A-Za-z0-9]+(?:_[A-Za-z][A-Za-z0-9]*)*)_(\d+)$")


def test_weights(test_ids: Iterable[str]) -> dict[str, float]:
    """Weight each test ID so `test_<name>_1..N` siblings split from one method get `1/N` each."""
    ids = [t for t in dict.fromkeys(test_ids) if t and not t.endswith(":*")]
    stems: dict[str, list[str]] = {}
    for tid in ids:
        m = _SPLIT_STEM_RE.match(tid)
        if m:
            stems.setdefault(m.group(1), []).append(tid)
    weights = {tid: 1.0 for tid in ids}
    for members in stems.values():
        if len(members) >= 2:
            w = 1.0 / len(members)
            for tid in members:
                weights[tid] = w
    return weights


def weighted_test_count(passed: Iterable[str], weights: dict[str, float]) -> float:
    """Sum of test weights for `passed` (unsplit tests count 1.0; N split siblings count 1/N each)."""
    return round(sum(weights.get(t, 1.0) for t in passed), 6)


def one_line_tail(text: str, limit: int = 160) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    last = lines[-1] if lines else "failed"
    return last if len(last) <= limit else last[: limit - 1] + "…"


def suite_test_ids(suite: str, name: str) -> list[str]:
    """List `name:Class.test_method` IDs defined in `suite` in source order."""
    try:
        tree = ast.parse(suite)
    except (SyntaxError, ValueError):
        return []
    ids: list[str] = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for fn in cls.body:
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test"):
                ids.append(f"{name}:{cls.name}.{fn.name}")
    return ids


def explain_build(res) -> str:
    """Detailed explanation of a BuildResult: auto context, suites, candidate vote matrix, repair steps, and why the winner won."""
    edit_target = getattr(res, "edit_target", "") or ""
    context_picks = list(getattr(res, "context_picks", []) or [])
    lines = [res.render()]
    if context_picks:
        lines.append("")
        lines.append("=== AUTO CONTEXT (ATLAS SEARCH) ===")
        for pick_info in context_picks:
            repo_name = pick_info.get("repo", "")
            path = pick_info.get("path", "")
            tokens = pick_info.get("tokens", 0)
            reason = pick_info.get("reason", "matched query")
            label = f"{repo_name}:{path}" if repo_name else path
            lines.append(f"  - {label} (~{tokens} tok): {reason}")

    lines.append("")
    lines.append("=== SUITES ===")
    initial = [c for c in res.candidates if not c.repaired]
    repairs = [c for c in res.candidates if c.repaired]
    live_initial = [c for c in initial if c.ok]
    all_live = [c for c in res.candidates if c.ok]

    support: dict[str, int] = {}
    all_tids: set[str] = set()
    for c in all_live:
        all_tids |= set(c.passed) | set(c.failed)
        for t in c.passed:
            support[t] = support.get(t, 0) + 1
    weights = test_weights(all_tids)

    existing_tids = sorted(t for t in all_tids if t.startswith("existing_"))
    if edit_target and existing_tids:
        lines.append(f"  existing tests ({len(existing_tids)} test(s)):")
        for tid in existing_tids:
            votes = [f"#{c.idx}" for c in all_live if tid in c.passed]
            v_str = ", ".join(votes) if votes else "none"
            lines.append(f"    - {tid}: passed by {len(votes)}/{len(all_live)} ({v_str})")

    if not res.suites:
        if not existing_tids:
            lines.append("  (no test suites survived)")
    else:
        for k, suite in enumerate(res.suites):
            sname = f"test_gen{k}"
            tids = suite_test_ids(suite, sname)
            lines.append(f"  suite {k + 1} ({sname}): {len(tids)} test(s)")
            for tid in tids:
                votes = [f"#{c.idx}" for c in all_live if tid in c.passed]
                v_str = ", ".join(votes) if votes else "none"
                lines.append(f"    - {tid}: passed by {len(votes)}/{len(all_live)} ({v_str})")

    lines.append("")
    lines.append("=== CANDIDATE VOTES & AGREEMENT GROUPS ===")
    for c in initial:
        if not c.ok:
            lines.append(f"  #{c.idx} ({c.style}, T={c.temperature:.2f}, {len(c.code)} chars): DROPPED — {c.dropped}")
            continue
        if edit_target:
            ex_pass = sorted(t for t in c.passed if t.startswith("existing_"))
            ex_fail = sorted(t for t in c.failed if t.startswith("existing_"))
            sp_pass = sorted(t for t in c.passed if not t.startswith("existing_"))
            sp_fail = sorted(t for t in c.failed if not t.startswith("existing_"))
            adds = int(getattr(c, "diff_adds", 0) or 0)
            dels = int(getattr(c, "diff_dels", 0) or 0)
            ex_detail = f"existing {len(ex_pass)}/{len(ex_pass) + len(ex_fail)}"
            if ex_fail:
                ex_detail += f" (fail: {', '.join(ex_fail[:4])})"
            sp_detail = f"spec {len(sp_pass)}/{len(sp_pass) + len(sp_fail)}"
            if sp_fail:
                sp_detail += f" (fail: {', '.join(sp_fail[:4])})"
            lines.append(
                f"  #{c.idx} ({c.style}, T={c.temperature:.2f}, diff +{adds}/-{dels}): {ex_detail}; {sp_detail}"
            )
            continue
        passed_list = ", ".join(sorted(c.passed)) if c.passed else "(none)"
        mismatch = missing_exports(res.spec, c.code)
        warn = f" [contract: {'; '.join(mismatch)}]" if mismatch else ""
        lines.append(
            f"  #{c.idx} ({c.style}, T={c.temperature:.2f}, {len(c.code)} chars): "
            f"passes {len(c.passed)} [{passed_list}]{warn}"
        )

    groups: dict[frozenset[str], list] = {}
    for c in live_initial:
        groups.setdefault(c.passed, []).append(c)

    mismatches = {c.idx: len(missing_exports(res.spec, c.code)) for c in live_initial} if res.spec else {}
    ranked_groups = sorted(
        groups.items(),
        key=lambda kv: (
            round(len(kv[1]) * weighted_test_count(kv[0], weights), 6),
            weighted_test_count(kv[0], weights),
            len(kv[0]),
            len({t.partition(":")[0] for t in kv[0] if ":" in t}),
            sum(support.get(t, 0) for t in kv[0]),
            -min(mismatches.get(c.idx, 0) for c in kv[1]),
            -min(len(c.code) for c in kv[1]),
            -min(c.idx for c in kv[1]),
        ),
        reverse=True,
    )
    if ranked_groups:
        lines.append("  agreement groups (initial candidates):")
        for g_idx, (pset, members) in enumerate(ranked_groups, 1):
            m_ids = ", ".join(f"#{c.idx}" for c in members)
            w_tests = weighted_test_count(pset, weights)
            score = round(len(members) * w_tests, 3)
            suites_hit = len({t.partition(":")[0] for t in pset if ":" in t})
            shortest = min(members, key=lambda c: (mismatches.get(c.idx, 0), len(c.code), c.idx))
            lines.append(
                f"    group {g_idx}: members=[{m_ids}] ({len(members)}) x tests={len(pset)} "
                f"-> score={score:g}, suites={suites_hit}, shortest=#{shortest.idx} ({len(shortest.code)} chars)"
            )

    lines.append("")
    lines.append("=== REPAIR STEPS ===")
    if not repairs:
        lines.append("  (no repair rounds run)")
    else:
        base_pick = (
            min(ranked_groups[0][1], key=lambda c: (mismatches.get(c.idx, 0), len(c.code), c.idx))
            if ranked_groups
            else None
        )
        prior_passed = base_pick.passed if base_pick else frozenset()
        for c in repairs:
            rnd = c.idx - 99
            if c.dropped and not c.passed and "did not pass a strict superset" not in c.dropped and "did not improve" not in c.dropped:
                lines.append(f"  round {rnd} (#{c.idx}, T={c.temperature:.2f}): DROPPED — {c.dropped}")
                continue
            gained = sorted(c.passed - prior_passed)
            lost = sorted(prior_passed - c.passed)
            outcome = "ACCEPTED (strict superset)" if not c.dropped else f"REJECTED ({c.dropped})"
            diff_note = ""
            if edit_target:
                adds = int(getattr(c, "diff_adds", 0) or 0)
                dels = int(getattr(c, "diff_dels", 0) or 0)
                diff_note = f", diff +{adds}/-{dels}"
            lines.append(
                f"  round {rnd} (#{c.idx}, T={c.temperature:.2f}{diff_note}): {len(prior_passed)} -> {len(c.passed)} tests "
                f"(+{len(gained)} gained: {', '.join(gained) or 'none'}; "
                f"-{len(lost)} regressed: {', '.join(lost) or 'none'}) -> {outcome}"
            )
            if not c.dropped:
                prior_passed = c.passed

    if res.disputed:
        lines.append("")
        lines.append("=== DISPUTED TESTS ===")
        for tid in sorted(res.disputed):
            voters = [f"#{c.idx}" for c in all_live if tid in c.passed]
            tail = one_line_tail(res.disputed[tid])
            lines.append(f"  - {tid} (passed by {', '.join(voters) or 'none'}): {tail}")
            src = test_method_source(res.suites, tid)
            if src:
                lines.append(textwrap.indent(src, "      "))

    lines.append("")
    lines.append("=== WHY THE WINNER WON ===")
    if edit_target:
        ex_p = int(getattr(res, "existing_passed", 0) or 0)
        ex_t = int(getattr(res, "existing_total", 0) or 0)
        if not res.code or res.chosen is None or not getattr(res, "diff", ""):
            ex_fails = [t for t in res.disputed if t.startswith("existing_")]
            sp_fails = [t for t in res.disputed if not t.startswith("existing_")]
            lines.append(
                f"  Edit rejected for {edit_target}: {len(ex_fails)} existing test(s) and "
                f"{len(sp_fails)} credible spec test(s) still failed (existing {ex_p}/{ex_t}, "
                f"spec {res.passed}/{res.credible})."
            )
        else:
            winner = next((c for c in res.candidates if c.idx == res.chosen), None)
            adds = int(getattr(winner, "diff_adds", 0) or 0) if winner else 0
            dels = int(getattr(winner, "diff_dels", 0) or 0) if winner else 0
            how = "after targeted repair" if (winner and winner.repaired) else f"from agreement group of {res.group_size}"
            lines.append(
                f"  Edit ready for {edit_target}: candidate #{res.chosen} won {how}, passing all "
                f"{ex_p}/{ex_t} existing tests and {res.passed}/{res.credible} credible spec tests "
                f"(diff +{adds}/-{dels}, confidence {res.confidence:.2f}, returned for review, never applied)."
            )
    elif not res.code or res.chosen is None:
        lines.append("  No candidate won: every sample failed parsing, static lint, or sandbox import.")
    elif not res.suites:
        winner = next((c for c in res.candidates if c.idx == res.chosen), None)
        chars = len(winner.code) if winner else 0
        lines.append(
            f"  Candidate #{res.chosen} won by fallback: no test suite survived, so the shortest importing "
            f"module ({chars} chars) was selected."
        )
    else:
        winner = next((c for c in res.candidates if c.idx == res.chosen), None)
        top_pset, top_members = ranked_groups[0]
        base_pick = min(top_members, key=lambda c: (mismatches.get(c.idx, 0), len(c.code), c.idx))
        base_score = len(top_members) * len(top_pset)
        if len(ranked_groups) > 1:
            r_pset, r_members = ranked_groups[1]
            r_score = len(r_members) * len(r_pset)
            vs = (
                f" (beating runner-up [{' '.join('#' + str(c.idx) for c in r_members)}] "
                f"with score {r_score} = {len(r_members)} x {len(r_pset)})"
            )
        else:
            vs = " (all live candidates fell in this group)"
        lines.append(
            f"  1. Initial dual-agreement pick: group [{' '.join('#' + str(c.idx) for c in top_members)}] "
            f"scored {base_score} ({len(top_members)} member(s) x {len(top_pset)} test(s)){vs}; "
            f"#{base_pick.idx} was its shortest member ({len(base_pick.code)} chars)."
        )
        if winner and winner.repaired:
            lines.append(
                f"  2. Targeted repair winner: candidate #{winner.idx} expanded #{base_pick.idx}'s passing "
                f"tests from {len(base_pick.passed)} to a strict superset of {len(winner.passed)}/{res.credible} "
                f"credible tests (confidence {res.confidence:.2f})."
            )
        else:
            lines.append(
                f"  2. Final winner: candidate #{res.chosen} kept after repair checks, passing "
                f"{res.passed}/{res.credible} credible tests (confidence {res.confidence:.2f})."
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------- multi-file & Atlas context

_NUMBERED_LINE_RE = re.compile(r"^\s*\d+(?::\s?|\s)(.*)$")
_SAFE_MOD_FILE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.py$")
_FILE_HEADER_RE = re.compile(r"^\s*#\s*(?:file|module|path)\s*:\s*([A-Za-z_][A-Za-z0-9_]*\.py)\s*$", re.I)
_FENCE_FILE_RE = re.compile(
    r"```[ \t]*(?:python|py)?(?:[ \t:]+([A-Za-z_][A-Za-z0-9_]*\.py))?[^\n]*\n(.*?)```",
    re.S,
)


def strip_numbered_lines(text: str) -> str:
    """Strip `  12: ` or `   12 ` prefixes from `atlas.read_source` output when present."""
    raw_lines = (text or "").splitlines()
    non_empty = [ln for ln in raw_lines if ln.strip()]
    if not non_empty or not all(_NUMBERED_LINE_RE.match(ln) for ln in non_empty):
        return text
    out: list[str] = []
    for ln in raw_lines:
        m = _NUMBERED_LINE_RE.match(ln)
        out.append(m.group(1) if m else ln)
    return "\n".join(out) + ("\n" if (text or "").endswith("\n") else "")


def atlas_context(atlas, repo: str, paths: str | Sequence[str], max_chars: int = 6000) -> tuple[str, dict[str, str]]:
    """Read existing repo file(s) from Atlas for building against repo context.

    Refuses any file flagged by `secret_reason`. Returns `(prompt_context_block, extra_files_map)`.
    """
    items = [paths] if isinstance(paths, str) else list(paths)
    blocks: list[str] = []
    extra_files: dict[str, str] = {}
    for p in items:
        p = str(p).strip()
        if not p:
            continue
        raw = atlas.read_source(repo, p, 1, 500)
        why = secret_reason(p, raw)
        if why:
            raise ValueError(f"{repo}:{p} looks like it holds a secret ({why}); refusing to include it in build context")
        clean = strip_numbered_lines(raw).strip()
        if not clean:
            continue
        clipped = clean[:max_chars]
        blocks.append(f"# ---- {repo}:{p} ----\n```python\n{clipped}\n```")
        fname = Path(p).name
        if _SAFE_MOD_FILE_RE.match(fname) and fname != "solution.py":
            extra_files[fname] = clean + "\n"
    if not blocks:
        return "", extra_files
    return "EXISTING REPO CONTEXT (from Atlas):\n" + "\n\n".join(blocks), extra_files


def atlas_search_context(
    atlas,
    spec: str,
    repo: str | None = None,
    exclude_paths: Sequence[str] = (),
    max_tokens: int = 1500,
    max_files: int = 4,
    search_limit: int = 12,
) -> tuple[str, dict[str, str], list[dict]]:
    """Pick related context files automatically from the local Atlas index, capped by `max_tokens`.

    Calls `atlas.search` and `atlas.read_source` in-process (fully local). Silently skips files
    flagged by `secret_reason` or unreadable on disk. Returns `(context_block, extra_files, picks)`,
    where each entry in `picks` is `{"repo", "path", "tokens", "reason"}` for `--explain`.
    """
    if atlas is None or not hasattr(atlas, "search"):
        return "", {}, []
    query = spec_query(spec) or (spec or "")[:300]
    if not query.strip():
        return "", {}, []
    try:
        hits = atlas.search(query, limit=search_limit, repo=repo or None)
    except Exception:  # noqa: BLE001 - search failure must never break a build
        return "", {}, []
    if not hits:
        return "", {}, []

    excluded = {str(p).strip() for p in exclude_paths if p}
    excluded_names = {Path(p).name for p in excluded}

    # Group hits by (repo, path) preserving best score and matched symbol names.
    grouped: dict[tuple[str, str], list] = {}
    for hit in hits:
        h_repo = str(getattr(hit, "repo", "") or "")
        h_path = str(getattr(hit, "path", "") or "")
        if not h_path:
            continue
        if h_path in excluded or Path(h_path).name in excluded_names:
            continue
        grouped.setdefault((h_repo, h_path), []).append(hit)

    ranked = sorted(
        grouped.items(),
        key=lambda kv: (
            max(float(getattr(h, "score", 0.0) or 0.0) for h in kv[1]),
            len(kv[1]),
            -len(kv[0][1]),
        ),
        reverse=True,
    )

    char_budget = max(400, max_tokens * 4)
    used_chars = 0
    blocks: list[str] = []
    extra_files: dict[str, str] = {}
    picks: list[dict] = []

    for (h_repo, h_path), file_hits in ranked:
        if len(picks) >= max_files:
            break
        remaining = char_budget - used_chars
        if remaining < 160:
            break
        try:
            raw = atlas.read_source(h_repo, h_path, 1, 250)
        except Exception:  # noqa: BLE001
            continue
        if secret_reason(h_path, raw):
            continue
        clean = strip_numbered_lines(raw).strip()
        if not clean:
            continue
        header = f"# ---- {h_repo}:{h_path} ----\n```python\n" if h_repo else f"# ---- {h_path} ----\n```python\n"
        footer = "\n```"
        body_budget = remaining - len(header) - len(footer) - (2 if blocks else 0)
        if body_budget < 80:
            break
        clipped = clean if len(clean) <= body_budget else (clean[: body_budget - 1].rstrip() + "…")
        block = f"{header}{clipped}{footer}"
        blocks.append(block)
        used_chars += len(block) + (2 if len(blocks) > 1 else 0)

        syms = list(dict.fromkeys(str(getattr(h, "qualname", "") or "") for h in file_hits if getattr(h, "qualname", "")))
        top_score = max(float(getattr(h, "score", 0.0) or 0.0) for h in file_hits)
        sym_part = ", ".join(syms[:3]) if syms else "file match"
        reason = f"matched {sym_part} (score {top_score:.2f})"
        picks.append({
            "repo": h_repo,
            "path": h_path,
            "tokens": max(1, len(clipped) // 4),
            "reason": reason,
        })

        fname = Path(h_path).name
        if _SAFE_MOD_FILE_RE.match(fname) and fname != "solution.py" and fname not in excluded_names:
            extra_files[fname] = clean + "\n"

    if not blocks:
        return "", extra_files, picks
    return "AUTO-RETRIEVED REPO CONTEXT (from Atlas search):\n" + "\n\n".join(blocks), extra_files, picks


_MODULE_HEADING_RE = re.compile(r"^\s*#{2,3}\s*module\s*:\s*([A-Za-z_][A-Za-z0-9_]*\.py)\s*$", re.I)
_PRECEDING_MOD_RE = re.compile(r"(?:^|\n)\s*(?:#{2,3}\s*module\s*:|#\s*(?:file|module|path)\s*:)\s*([A-Za-z_][A-Za-z0-9_]*\.py)\s*\n+```[ \t]*(?:python|py)?[^\n]*\n(.*?)```", re.I | re.S)


def parse_multifile_spec(spec: str) -> dict[str, str]:
    """Parse `## module: <name.py>` sections from `spec` in order; empty dict if none are present."""
    modules: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in (spec or "").splitlines():
        m = _MODULE_HEADING_RE.match(line)
        if m:
            if current is not None:
                modules[current] = "\n".join(buf).strip()
            current = m.group(1).strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        modules[current] = "\n".join(buf).strip()
    return modules


def multifile_candidate_prompt(
    spec: str, modules: Sequence[str], shots: str, i: int, edge_first: str
) -> str:
    """Prompt for a multi-file implementation candidate naming every required `.py` module."""
    names = ", ".join(f"`{m}`" for m in modules)
    fmt = (
        f"Reply with one fenced ```python <filename.py> block for EACH module ({names}), "
        "using exact module filenames on the opening fence (e.g. ```python a.py)."
    )
    base = candidate_prompt(spec, shots, i, edge_first)
    return f"{base}\n\n{fmt}"


def multifile_suite_prompt(spec: str, modules: Sequence[str], i: int) -> str:
    """Prompt for a test suite covering a multi-file spec."""
    mod_names = ", ".join(f"`{Path(m).stem}`" for m in modules)
    extra = f"Import directly from the spec modules ({mod_names}) and test how they work together."
    return f"{suite_prompt(spec, i)}\n\n{extra}"


def parse_multifile_reply(text: str) -> tuple[str, dict[str, str]]:
    """Extract `(solution_code, extra_files)` when a model reply contains named `.py` blocks."""
    cleaned = strip_reasoning(text or "")
    extra: dict[str, str] = {}
    unnamed: list[str] = []
    solution = ""
    for fence_name, body in _FENCE_FILE_RE.findall(cleaned):
        lines = body.rstrip().splitlines()
        name = (fence_name or "").strip()
        if not name and lines:
            m = _FILE_HEADER_RE.match(lines[0])
            if m:
                name = m.group(1)
                lines = lines[1:]
        content = "\n".join(lines).rstrip() + "\n" if lines else ""
        if not content.strip():
            continue
        if name == "solution.py":
            solution = content
        elif name and _SAFE_MOD_FILE_RE.match(name):
            extra[name] = content
        else:
            unnamed.append(content)
    # Also pick up blocks preceded by a `## module: foo.py` or `# file: foo.py` heading outside the fence.
    for mod_name, body in _PRECEDING_MOD_RE.findall(cleaned):
        content = body.rstrip() + "\n" if body.strip() else ""
        if not content:
            continue
        if mod_name == "solution.py" and not solution:
            solution = content
        elif _SAFE_MOD_FILE_RE.match(mod_name) and mod_name not in extra:
            extra[mod_name] = content
    if not solution and unnamed:
        solution = max(unnamed, key=len)
    return solution, extra


# ---------------------------------------------------------------------- edit mode (unified diffs + existing tests)

_DIFF_FENCE_RE = re.compile(r"```[ \t]*(?:diff|patch)[^\n]*\n(.*?)```", re.I | re.S)
_ANY_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_HUNK_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


def extract_diff(reply: str) -> str:
    """Extract a unified diff from a model reply (```diff/patch block, or raw `---`/`@@` lines)."""
    cleaned = strip_reasoning(reply or "")
    m = _DIFF_FENCE_RE.search(cleaned)
    if m:
        return m.group(1).strip("\n") + "\n"
    for body in _ANY_FENCE_RE.findall(cleaned):
        if _HUNK_HEADER_RE.search(body):
            return body.strip("\n") + "\n"
    lines = cleaned.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith(("--- ", "diff --git ", "@@ ")):
            return "\n".join(lines[idx:]).strip("\n") + "\n"
    return ""


def apply_unified_diff(original: str, diff_text: str) -> tuple[str | None, str]:
    """Apply a unified diff to `original` in pure Python. Returns `(patched, "")` or `(None, reason)`."""
    raw = (diff_text or "").replace("\r\n", "\n")
    if not raw.strip():
        return None, "empty diff"
    diff_lines = raw.splitlines()
    hunks: list[tuple[int, int, int, int, list[tuple[str, str]], bool]] = []
    i = 0
    while i < len(diff_lines):
        line = diff_lines[i]
        m = _HUNK_HEADER_RE.match(line)
        if not m:
            i += 1
            continue
        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) is not None else 1
        new_start = int(m.group(3))
        new_count = int(m.group(4)) if m.group(4) is not None else 1
        i += 1
        ops: list[tuple[str, str]] = []
        seen_old = 0
        seen_new = 0
        no_eof_newline = False
        while i < len(diff_lines):
            cur = diff_lines[i]
            if _HUNK_HEADER_RE.match(cur) or cur.startswith(("diff --git ", "--- ", "+++ ")):
                break
            if cur.startswith("\\ No newline at end of file"):
                if ops and ops[-1][0] in (" ", "+"):
                    no_eof_newline = True
                i += 1
                continue
            if cur == "" and (seen_old < old_count or seen_new < new_count):
                ops.append((" ", ""))
                seen_old += 1
                seen_new += 1
                i += 1
                continue
            if not cur:
                i += 1
                continue
            tag, text = cur[0], cur[1:]
            if tag not in (" ", "-", "+"):
                break
            ops.append((tag, text))
            if tag in (" ", "-"):
                seen_old += 1
            if tag in (" ", "+"):
                seen_new += 1
            i += 1
        if not ops:
            return None, f"hunk at line {old_start} has no operations"
        hunks.append((old_start, old_count, new_start, new_count, ops, no_eof_newline))

    if not hunks:
        return None, "no @@ hunks found in diff"

    orig_norm = (original or "").replace("\r\n", "\n")
    had_trailing_nl = orig_norm.endswith("\n")
    lines = orig_norm.splitlines()
    delta = 0
    drop_trailing_nl = False

    for old_start, _old_count, _new_start, _new_count, ops, no_eof_nl in hunks:
        old_chunk = [text for tag, text in ops if tag in (" ", "-")]
        new_chunk = [text for tag, text in ops if tag in (" ", "+")]
        base = 0 if old_start == 0 else max(0, old_start - 1 + delta)
        if not old_chunk:
            base = min(len(lines), base)
        matched_pos: int | None = None
        for offset in (0, -1, 1, -2, 2, -3, 3):
            pos = base + offset
            if pos < 0 or pos + len(old_chunk) > len(lines):
                continue
            if lines[pos : pos + len(old_chunk)] == old_chunk:
                matched_pos = pos
                break
        if matched_pos is None:
            return None, f"hunk at line {old_start} does not match original"
        lines[matched_pos : matched_pos + len(old_chunk)] = new_chunk
        delta += len(new_chunk) - len(old_chunk)
        if no_eof_nl:
            drop_trailing_nl = True

    out = "\n".join(lines)
    if lines and not drop_trailing_nl and (had_trailing_nl or orig_norm == ""):
        out += "\n"
    return out, ""


def unified_diff(original: str, updated: str, filename: str) -> str:
    """Standard unified diff (`a/<filename>` -> `b/<filename>`); "" when unchanged."""
    if original == updated:
        return ""
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )
    if diff and not diff.endswith("\n"):
        diff += "\n"
    return diff


def find_existing_tests(target: str | Path, explicit: Sequence[str | Path] = ()) -> list[Path]:
    """Find existing test files for `target` (either explicit `--tests` paths or beside `target`).

    Refuses any file flagged by `secret_reason`.
    """
    target_path = Path(target).expanduser()
    found: list[Path] = []
    seen: set[Path] = set()

    if explicit:
        for raw in explicit:
            p = Path(raw).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"test file not found: {p}")
            resolved = p.resolve()
            if resolved in seen:
                continue
            content = p.read_text(encoding="utf-8")
            why = secret_reason(str(p), content)
            if why:
                raise ValueError(f"{p} looks like it holds a secret ({why}); refusing to read it")
            seen.add(resolved)
            found.append(p)
        return found

    stem = target_path.stem
    parent = target_path.parent
    candidates: list[Path] = [
        parent / f"test_{stem}.py",
        parent / f"{stem}_test.py",
        *sorted(parent.glob(f"test_{stem}_*.py")),
        parent / "tests" / f"test_{stem}.py",
        parent.parent / "tests" / f"test_{stem}.py",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        resolved = p.resolve()
        if resolved in seen or resolved == target_path.resolve():
            continue
        try:
            content = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if secret_reason(str(p), content):
            continue
        seen.add(resolved)
        found.append(p)
    return found


def edit_candidate_prompt(
    spec: str, filename: str, original: str, shots: str, i: int, edge_first: str
) -> str:
    """Prompt for an edit-mode candidate: asks for a unified diff against `original`."""
    style = "edge-first" if i % 2 else "direct"
    head = (f"{shots}\n\n" if shots else "") + f"SPEC (change to `{filename}`):\n{spec}"
    body = f"\n\nEXISTING `{filename}`:\n```python\n{original.rstrip()}\n```"
    tail = f"\n\n{edge_first}" if style == "edge-first" else ""
    if i >= 2:
        checklist = spec_checklist(spec)
        if checklist:
            tail += f"\n\n{checklist}"
    return (
        f"{head}{body}{tail}\n\n"
        f"Reply with a unified diff against `{filename}` inside one ```diff block."
    )


def edit_suite_prompt(spec: str, module_name: str, original: str, i: int) -> str:
    """Prompt for a spec test suite in edit mode importing from `module_name`."""
    base = suite_prompt(spec, i)
    preview = original.rstrip()[:3000]
    return (
        f"{base}\n\n"
        f"Import from `{module_name}` (the existing module being edited). Test the NEW or CHANGED behavior "
        f"requested by the SPEC.\n\nCURRENT `{module_name}.py` (before edit):\n```python\n{preview}\n```"
    )


# ---------------------------------------------------------------------- spec lint & token-capped exemplars

_VAGUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:etc\.?|and\s+so\s+on|and\s+more)\b", re.I), "uses open-ended phrasing ('etc.' / 'and so on')"),
    (re.compile(r"\b(?:appropriate|reasonable|sensible|as\s+needed|if\s+necessary)\b", re.I), "uses vague qualifier ('appropriate' / 'reasonable' / 'as needed')"),
    (re.compile(r"\b(?:fast|efficient|performant|quick)\b", re.I), "asks for speed/efficiency without a concrete complexity or bound"),
)
_EXAMPLE_RE = re.compile(r"(?:>>>|->|==|example|e\.g\.|for\s+instance|returns?\s+[\[\{\'\"0-9-])", re.I)
_ERROR_MENTION_RE = re.compile(r"\b(?:error|invalid|bad\s+input|fail|edge\s+case)\b", re.I)


def lint_spec(spec: str) -> list[str]:
    """Static pre-sampling lint of a spec: list ambiguities and missing examples."""
    text = (spec or "").strip()
    if not text:
        return ["spec is empty"]
    warnings: list[str] = []
    if len(text) < 24:
        warnings.append("spec is very short; add concrete inputs, outputs, and edge cases")
    top_req, method_req, prop_req = expected_exports(text)
    if not (top_req or method_req or prop_req) and not parse_multifile_spec(text):
        warnings.append("no function, class, or signature found (backtick signatures like `fn(x)` improve accuracy)")
    if not _EXAMPLE_RE.search(text):
        warnings.append("no worked example found (add at least one concrete input -> output example)")
    if _ERROR_MENTION_RE.search(text) and not _EXC_RE.search(text):
        warnings.append("mentions errors/invalid input without naming a specific exception type (e.g. ValueError)")
    for pat, msg in _VAGUE_PATTERNS:
        if pat.search(text):
            if "complexity" in msg and _BIGO_RE.search(text):
                continue
            warnings.append(msg)
    return warnings


def select_exemplars_within_budget(
    spec: str, fragments: Sequence, limit: int = 2, max_tokens: int = 900
) -> tuple[str, list[int]]:
    """Rank recalled exemplars by spec similarity + utility and pack them within `max_tokens` (~4 chars/token)."""
    picked = rank_exemplars(spec, fragments, limit=limit)
    if not picked:
        return "", []
    char_budget = max(200, max_tokens * 4)
    blocks: list[str] = []
    ids: list[int] = []
    used = 0
    for frag in picked:
        header = f"EXAMPLE {len(blocks) + 1} (accepted earlier):\n"
        body = (getattr(frag, "text", "") or "").strip()
        if not body:
            continue
        remaining = char_budget - used - len(header) - (2 if blocks else 0)
        if remaining < 120:
            break
        if len(body) > remaining:
            body = body[: remaining - 1].rstrip() + "…"
        block = header + body
        blocks.append(block)
        ids.append(frag.id)
        used += len(block) + (2 if len(blocks) > 1 else 0)
    return "\n\n".join(blocks), ids


def emit_files(
    code: str,
    agreed: Sequence[str],
    out_name: str = "solution.py",
    extra_files: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build `{filename: content}` for the module, any companion modules, and retargeted agreed test files."""
    if not code:
        return {}
    stem = Path(out_name).stem or "solution"
    mod_file = f"{stem}.py" if not str(out_name).endswith(".py") else Path(out_name).name
    files: dict[str, str] = {mod_file: code}
    for fname, body in (extra_files or {}).items():
        if _SAFE_MOD_FILE_RE.match(fname) and fname != mod_file and body:
            files[fname] = body
    for k, suite in enumerate(agreed, 1):
        if not suite or not suite.strip():
            continue
        tname = f"test_{stem}.py" if k == 1 else f"test_{stem}_{k}.py"
        files[tname] = retarget(suite, stem)
    return files


def donor_snippet(candidates: Sequence, pick, missing: Iterable[str], max_chars: int = 1200) -> str:
    """Reference snippet from the best other live candidate that passes tests in `missing`."""
    missing_set = set(missing)
    if not missing_set or not candidates:
        return ""
    pick_idx = getattr(pick, "idx", None)
    donors = [
        c
        for c in candidates
        if getattr(c, "ok", False) and getattr(c, "idx", None) != pick_idx and (c.passed & missing_set)
    ]
    if not donors:
        return ""
    best = max(
        donors,
        key=lambda c: (len(c.passed & missing_set), len(c.passed), -len(c.code), -c.idx),
    )
    covered = ", ".join(sorted(best.passed & missing_set)[:6])
    snippet = best.code.strip()[:max_chars]
    return f"PASSING CANDIDATE REFERENCE (candidate #{best.idx} passes {covered}):\n```python\n{snippet}\n```"


def build_short_summary(res) -> str:
    """Compact (~<=6 lines) chat-facing summary of what won and why."""
    edit_target = getattr(res, "edit_target", "") or ""
    diff_text = getattr(res, "diff", "") or ""
    existing_passed = int(getattr(res, "existing_passed", 0) or 0)
    existing_total = int(getattr(res, "existing_total", 0) or 0)
    extra_files = getattr(res, "extra_files", {}) or {}
    lines: list[str] = []

    if not res.code:
        kind = f"Edit failed for {edit_target}" if edit_target else "Build failed"
        reason = res.notes[-1] if res.notes else "no candidate passed"
        lines.append(f"{kind}: {reason}")
        if existing_total:
            lines.append(f"Existing tests: {existing_passed}/{existing_total} passed")
        dropped = sum(1 for c in res.candidates if not c.ok)
        if res.candidates:
            lines.append(f"Candidates: {len(res.candidates) - dropped}/{len(res.candidates)} usable")
        if res.disputed:
            ids = ", ".join(sorted(res.disputed)[:4])
            more = f" (+{len(res.disputed) - 4} more)" if len(res.disputed) > 4 else ""
            lines.append(f"Failing: {ids}{more}")
        lines.append(f"Sandbox: {res.sandbox}")
        return "\n".join(lines[:6])

    winner = next((c for c in res.candidates if c.idx == res.chosen), None)
    if edit_target:
        adds = sum(1 for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
        dels = sum(1 for ln in diff_text.splitlines() if ln.startswith("-") and not ln.startswith("---"))
        lines.append(
            f"Edit ready for {edit_target}: candidate #{res.chosen} "
            f"(confidence {res.confidence:.2f}, diff +{adds}/-{dels}, not applied)"
        )
        lines.append(
            f"Tests: existing {existing_passed}/{existing_total}, spec {res.passed}/{res.credible} credible"
        )
    else:
        lines.append(
            f"Built candidate #{res.chosen} (group of {res.group_size}, confidence {res.confidence:.2f})"
        )
        lines.append(f"Tests: {res.passed}/{res.credible} credible tests passed across {len(res.suites)} suite(s)")

    if winner and winner.repaired:
        lines.append(f"Why: repaired candidate #{winner.idx} passed a strict superset ({len(winner.passed)} tests)")
    elif not res.suites and not existing_total:
        lines.append("Why: no test suite survived; picked shortest importing candidate")
    else:
        lines.append(f"Why: top dual-agreement group (size {res.group_size}), shortest clean member")

    if extra_files:
        names = ", ".join(sorted(extra_files)[:4])
        more = f" (+{len(extra_files) - 4} more)" if len(extra_files) > 4 else ""
        lines.append(f"Companion files: {names}{more}")
    elif res.disputed:
        ids = ", ".join(sorted(res.disputed)[:4])
        more = f" (+{len(res.disputed) - 4} more)" if len(res.disputed) > 4 else ""
        lines.append(f"Disputed: {ids}{more}")

    if res.exemplars:
        lines.append("Exemplars: " + ", ".join(f"#{i}" for i in res.exemplars))
    lines.append(f"Sandbox: {res.sandbox}")
    return "\n".join(lines[:6])


def build_summary_dict(res) -> dict:
    """JSON-serializable summary of a `BuildResult`, including candidate votes, short summary, and `--explain`."""
    return {
        "spec": res.spec,
        "code": res.code,
        "tests": res.tests,
        "diff": getattr(res, "diff", ""),
        "edit_target": getattr(res, "edit_target", ""),
        "existing_passed": getattr(res, "existing_passed", 0),
        "existing_total": getattr(res, "existing_total", 0),
        "chosen": res.chosen,
        "group_size": res.group_size,
        "passed": res.passed,
        "credible": res.credible,
        "confidence": res.confidence,
        "sandbox": res.sandbox,
        "exemplars": list(res.exemplars),
        "notes": list(res.notes),
        "spec_warnings": list(getattr(res, "spec_warnings", []) or []),
        "context_picks": list(getattr(res, "context_picks", []) or []),
        "agreed": list(res.agreed),
        "disputed": dict(res.disputed),
        "extra_files": dict(getattr(res, "extra_files", {}) or {}),
        "summary": build_short_summary(res),
        "explain": explain_build(res),
        "candidates": [
            {
                "idx": c.idx,
                "code": c.code,
                "temperature": c.temperature,
                "style": c.style,
                "passed": sorted(c.passed),
                "failed": dict(c.failed),
                "repaired": c.repaired,
                "dropped": c.dropped,
                "ok": c.ok,
                "diff_adds": int(getattr(c, "diff_adds", 0) or 0),
                "diff_dels": int(getattr(c, "diff_dels", 0) or 0),
            }
            for c in res.candidates
        ],
    }

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
import re
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

from ..atlas.secrets import secret_reason
from ..util import split_identifier
from .suites import retarget

__all__ = [
    "SUITE_ANGLES",
    "atlas_context",
    "build_summary_dict",
    "candidate_prompt",
    "donor_snippet",
    "emit_files",
    "expected_exports",
    "explain_build",
    "format_failures",
    "fragment_utility",
    "has_tests",
    "missing_exports",
    "normalize_suite",
    "parse_multifile_reply",
    "rank_exemplars",
    "spec_checklist",
    "spec_query",
    "strip_numbered_lines",
    "suite_prompt",
    "suite_test_ids",
    "test_method_source",
    "test_weights",
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


def test_method_source(suites: Sequence[str], tid: str) -> str:
    """Source code of the test method named by `tid` (`test_gen0:Class.test_x`); "" if not found."""
    mod, sep, rest = (tid or "").partition(":")
    if not sep or not mod.startswith("test_gen") or "." not in rest:
        return ""
    try:
        idx = int(mod[len("test_gen") :])
    except ValueError:
        return ""
    if not (0 <= idx < len(suites)):
        return ""
    cls_name, _, fn_name = rest.partition(".")
    suite = suites[idx]
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


def format_failures(suites: Sequence[str], failures: dict[str, str], limit: int = 6) -> str:
    """Render failing tests for a repair prompt, pairing each traceback with the test method source."""
    blocks: list[str] = []
    for tid, out in list(failures.items())[:limit]:
        entry = f"--- {tid}\n{out[-900:]}"
        src = test_method_source(suites, tid)
        if src and src not in out:
            entry += f"\n```python\n{src}\n```"
        blocks.append(entry)
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
    """Detailed explanation of a BuildResult: suites, candidate vote matrix, repair steps, and why the winner won."""
    lines = [res.render(), "", "=== SUITES ==="]
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

    if not res.suites:
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
            if c.dropped and not c.passed and "did not pass a strict superset" not in c.dropped:
                lines.append(f"  round {rnd} (#{c.idx}, T={c.temperature:.2f}): DROPPED — {c.dropped}")
                continue
            gained = sorted(c.passed - prior_passed)
            lost = sorted(prior_passed - c.passed)
            outcome = "ACCEPTED (strict superset)" if not c.dropped else f"REJECTED ({c.dropped})"
            lines.append(
                f"  round {rnd} (#{c.idx}, T={c.temperature:.2f}): {len(prior_passed)} -> {len(c.passed)} tests "
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
    if not res.code or res.chosen is None:
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

_NUMBERED_LINE_RE = re.compile(r"^\s*\d+:\s?(.*)$")
_SAFE_MOD_FILE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.py$")
_FILE_HEADER_RE = re.compile(r"^\s*#\s*(?:file|module|path)\s*:\s*([A-Za-z_][A-Za-z0-9_]*\.py)\s*$", re.I)
_FENCE_FILE_RE = re.compile(
    r"```[ \t]*(?:python|py)?(?:[ \t:]+([A-Za-z_][A-Za-z0-9_]*\.py))?[^\n]*\n(.*?)```",
    re.S,
)


def strip_numbered_lines(text: str) -> str:
    """Strip `  12: ` prefixes from `atlas.read_source` output when present."""
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


def parse_multifile_reply(text: str) -> tuple[str, dict[str, str]]:
    """Extract `(solution_code, extra_files)` when a model reply contains named `.py` blocks."""
    extra: dict[str, str] = {}
    unnamed: list[str] = []
    solution = ""
    for fence_name, body in _FENCE_FILE_RE.findall(text or ""):
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
    if not solution and unnamed:
        solution = max(unnamed, key=len)
    return solution, extra


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


def build_summary_dict(res) -> dict:
    """JSON-serializable summary of a `BuildResult`, including candidate votes and `--explain` text."""
    return {
        "spec": res.spec,
        "code": res.code,
        "tests": res.tests,
        "chosen": res.chosen,
        "group_size": res.group_size,
        "passed": res.passed,
        "credible": res.credible,
        "confidence": res.confidence,
        "sandbox": res.sandbox,
        "exemplars": list(res.exemplars),
        "notes": list(res.notes),
        "agreed": list(res.agreed),
        "disputed": dict(res.disputed),
        "extra_files": dict(getattr(res, "extra_files", {}) or {}),
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
            }
            for c in res.candidates
        ],
    }

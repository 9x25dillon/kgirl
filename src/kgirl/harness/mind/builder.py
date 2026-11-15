"""Spec -> code: test-first generation with sampling and agreement selection.

A small local model asked once gets a spec right only some of the time, and it cannot tell which time.
The builder does not trust any single sample, and it does not trust its own tests either:

    1. recall    exemplars (earlier specs with code that was accepted) as worked examples
    2. tests     K independent unittest suites written from the spec alone
    3. code      N implementations, alternating "direct" and "edge cases first" prompts, spread temperatures
    4. matrix    run every implementation against every suite in a sandbox, test by test
    5. agree     group implementations that pass exactly the same tests; score = members x tests passed
                 (dual agreement, as in CodeT: wrong code rarely agrees with other wrong code, and wrong
                 tests rarely pass agreeing code). Pick the best group's shortest member.
    6. repair    show the pick the failures that some other implementation passes (so the test is
                 satisfiable) and ask for a fix; keep it only if it passes a strict superset of tests.
    7. learn     the result and its agreed tests become an `exemplar`: staged by default, active when
                 the person accepts it. Active exemplars feed step 1, so accepted work makes later
                 builds better (this is the builder's self-improvement, through memory, not weights).

Generated code runs in a temp dir under CPU, memory and wall-clock limits, with a scrubbed environment.
That is a guard against accidents, not a security boundary: it is the user's own model on their machine.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..soup import Soup
from .text import python_code, strip_reasoning

LLM = Callable[[str, str, int, float], str]   # (system, prompt, max_tokens, temperature) -> text

IMPL_SYSTEM = """You are an expert Python engineer. Write the module `solution.py` that implements the SPEC
exactly: the names, signatures, return values and exceptions it states. Write elegant, idiomatic, typed
code with short docstrings: clear names, small functions, the right data structure for the complexity the
spec asks for, every edge case the spec mentions handled explicitly. Standard library only. No I/O, no
prints, no example usage, no tests. Reply with one ```python block containing the whole module."""

EDGE_FIRST = """Before the code, list every edge case and error condition the spec states or implies, as a
short bullet list. Then give the whole module in one ```python block that handles each of them."""

TEST_SYSTEM = """You write precise unit tests from a specification, before any implementation exists.
Write a Python unittest module that imports from `solution` exactly the names the SPEC defines. Cover the
examples in the spec, every stated edge case and error (assertRaises with the stated exception type), and
a few ordinary cases whose expected values you compute carefully by hand. One behaviour per test method;
6 to 14 methods; descriptive names. Do not test anything the spec leaves unspecified (exact error
messages, internal structure, performance). Reply with one ```python block containing the whole module."""

REPAIR_SYSTEM = """You fix a Python module so it satisfies its SPEC. You get the spec, the current module and
the tests it fails, with their output. Tests can be wrong: follow the SPEC when a test disagrees with it.
Reply with the whole corrected module in one ```python block."""

# `test_x (mod.Class.test_x) ... ok` (3.11+) or `test_x (mod.Class) ... ok`; a docstring adds one line.
_RESULT = re.compile(r"^(test\w*) \(([\w.]+)\)(?:\n[^\n]*?)? \.\.\. (ok|FAIL|ERROR|skipped.*|expected failure)$", re.M)


@dataclass
class Candidate:
    idx: int
    code: str
    temperature: float
    style: str
    passed: frozenset[str] = frozenset()
    failed: dict[str, str] = field(default_factory=dict)
    repaired: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.code)


@dataclass
class BuildResult:
    spec: str
    code: str
    tests: str
    chosen: int | None
    group_size: int
    passed: int
    credible: int
    candidates: list[Candidate]
    suites: list[str]
    exemplars: list[int]
    notes: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Share of credible tests passed, discounted when few implementations agreed."""
        if not self.code or not self.credible:
            return 0.0
        agree = min(1.0, self.group_size / max(2, len([c for c in self.candidates if c.ok]) / 2))
        return round(self.passed / self.credible * (0.5 + 0.5 * agree), 3)

    def render(self) -> str:
        lines = [f"built: candidate #{self.chosen} (group of {self.group_size}), passes {self.passed}/{self.credible}"
                 f" credible tests, confidence {self.confidence:.2f}"]
        for c in self.candidates:
            mark = "★" if c.idx == self.chosen else ("·" if c.ok else "✗")
            lines.append(f"  {mark} #{c.idx} {c.style:<10} T={c.temperature:.1f} passed={len(c.passed):<3}"
                         f" {'repaired' if c.repaired else ''}")
        lines += [f"  note: {n}" for n in self.notes]
        if self.exemplars:
            lines.append(f"  recalled exemplars: {', '.join('#' + str(i) for i in self.exemplars)}")
        return "\n".join(lines)


class Builder:
    def __init__(self, llm: LLM, soup: Soup | None = None, n_impl: int = 6, n_suites: int = 3, repair_rounds: int = 2,
                 timeout: float = 20.0, workers: int = 4, emit: Callable[[str, dict], None] | None = None):
        self.llm = llm
        self.soup = soup
        self.n_impl = n_impl
        self.n_suites = n_suites
        self.repair_rounds = repair_rounds
        self.timeout = timeout
        self.workers = workers
        self.emit = emit or (lambda topic, payload: None)

    # ------------------------------------------------------------------ public

    def build(self, spec: str, scope: str = "*") -> BuildResult:
        spec = textwrap.dedent(spec).strip()
        shots, ex_ids = self._exemplars(spec, scope)
        self.emit("build.start", {"spec_chars": len(spec), "exemplars": ex_ids})
        suites = [s for s in self._parallel(lambda i: self._suite(spec, i), range(self.n_suites)) if s]
        cands = self._parallel(lambda i: self._candidate(spec, shots, i), range(self.n_impl))
        cands = [c for c in cands if c.ok]
        notes: list[str] = []
        if not cands:
            return BuildResult(spec, "", "\n\n".join(suites), None, 0, 0, 0, [], suites, ex_ids,
                               ["no candidate produced parseable code"])
        if not suites:
            notes.append("no test suite parsed; picked the shortest candidate that imports cleanly")
        self._parallel(lambda c: self._score(c, suites), cands)
        pick, group = self._agree(cands)
        credible = self._credible(cands)
        for rnd in range(self.repair_rounds):
            missing = {t: pick.failed[t] for t in credible if t in pick.failed}
            if not missing:
                break
            fixed = self._repair(spec, pick, missing, rnd)
            if fixed is None:
                continue
            self._score(fixed, suites)
            if fixed.passed > pick.passed:
                notes.append(f"repair round {rnd + 1}: {len(pick.passed)} -> {len(fixed.passed)} tests")
                cands.append(fixed)
                pick, group = fixed, [fixed]
            else:
                notes.append(f"repair round {rnd + 1} did not pass a strict superset; kept the original")
        passed = len(pick.passed & credible) if credible else len(pick.passed)
        res = BuildResult(spec, pick.code, merge_suites(suites), pick.idx, len(group), passed,
                          len(credible) or len(pick.passed), cands, suites, ex_ids, notes)
        self.emit("build.end", {"chosen": pick.idx, "group": len(group), "passed": passed,
                                "credible": res.credible, "confidence": res.confidence})
        return res

    def learn(self, res: BuildResult, accepted: bool, scope: str = "*", note: str = "") -> int | None:
        """Store the spec and the result as an exemplar: active when a person accepted it, else staged."""
        if not self.soup or not res.code:
            return None
        head = " ".join(res.spec.split())[:600]
        text = f"SPEC: {head}\n\nSOLUTION:\n```python\n{res.code.strip()}\n```" + (f"\n\nNOTE: {note}" if note else "")
        fid = self.soup.add(text, kind="exemplar", tags="build", source="user" if accepted else "build", scope=scope,
                            status="active" if accepted else "staged")
        self.soup.credit([fid], success=True)
        if accepted:
            self.soup.set_status(fid, "active", "accepted by the user")
        return fid

    # ------------------------------------------------------------------ generation

    def _exemplars(self, spec: str, scope: str) -> tuple[str, list[int]]:
        if not self.soup:
            return "", []
        rec = self.soup.recall(spec, budget_tokens=1400, kinds=("exemplar",), scope=scope)
        if not rec.fragments:
            return "", []
        shots = "\n\n".join(f"EXAMPLE {i + 1} (accepted earlier):\n{f.text}" for i, f in enumerate(rec.fragments[:2]))
        return shots, rec.ids[:2]

    def _candidate(self, spec: str, shots: str, i: int) -> Candidate:
        temps = (0.2, 0.5, 0.8, 0.35, 0.65, 0.9)
        t = temps[i % len(temps)]
        style = "edge-first" if i % 2 else "direct"
        prompt = (f"{shots}\n\n" if shots else "") + f"SPEC:\n{spec}" + (f"\n\n{EDGE_FIRST}" if style == "edge-first" else "")
        try:
            code = python_code(self.llm(IMPL_SYSTEM, prompt, 2048, t))
        except Exception:  # noqa: BLE001 - one failed sample must not sink the build
            code = ""
        if code and not _compiles(code):
            code = ""
        return Candidate(i, code, t, style)

    def _suite(self, spec: str, i: int) -> str:
        t = (0.2, 0.5, 0.8, 0.4)[i % 4]
        try:
            code = python_code(self.llm(TEST_SYSTEM, f"SPEC:\n{spec}", 2048, t))
        except Exception:  # noqa: BLE001
            return ""
        if not code or "solution" not in code or not _compiles(code):
            return ""
        return code

    def _repair(self, spec: str, pick: Candidate, missing: dict[str, str], rnd: int) -> Candidate | None:
        fails = "\n\n".join(f"--- {t}\n{out[-900:]}" for t, out in list(missing.items())[:6])
        prompt = f"SPEC:\n{spec}\n\nCURRENT solution.py:\n```python\n{pick.code}```\n\nFAILING TESTS:\n{fails}"
        try:
            code = python_code(self.llm(REPAIR_SYSTEM, prompt, 2048, 0.2 + 0.2 * rnd))
        except Exception:  # noqa: BLE001
            return None
        if not code or not _compiles(code):
            return None
        return Candidate(100 + rnd, code, 0.2 + 0.2 * rnd, "repair", repaired=True)

    # ------------------------------------------------------------------ evaluation

    def _score(self, c: Candidate, suites: list[str]) -> Candidate:
        passed: set[str] = set()
        failed: dict[str, str] = {}
        for k, suite in enumerate(suites):
            p, f = run_suite(c.code, suite, f"test_gen{k}", self.timeout)
            passed |= p
            failed.update(f)
        if not suites and not _imports(c.code, self.timeout):
            failed["import"] = "module failed to import"
        c.passed, c.failed = frozenset(passed), failed
        return c

    @staticmethod
    def _agree(cands: list[Candidate]) -> tuple[Candidate, list[Candidate]]:
        groups: dict[frozenset[str], list[Candidate]] = {}
        for c in cands:
            if "import" not in c.failed:
                groups.setdefault(c.passed, []).append(c)
        if not groups:
            groups = {c.passed: [c] for c in cands}
        best = max(groups.items(), key=lambda kv: (len(kv[1]) * max(1, len(kv[0])), len(kv[0]),
                                                   -min(len(c.code) for c in kv[1])))
        members = best[1]
        return min(members, key=lambda c: (len(c.code), c.idx)), members

    @staticmethod
    def _credible(cands: list[Candidate]) -> set[str]:
        """Tests at least one implementation passes: satisfiable, so worth repairing toward."""
        out: set[str] = set()
        for c in cands:
            out |= c.passed
        return out

    def _parallel(self, fn, items):
        items = list(items)
        with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(items)))) as pool:
            return list(pool.map(fn, items))


# ---------------------------------------------------------------------- sandbox

def _limits() -> None:  # pragma: no cover - runs in the child
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    except (ImportError, ValueError, OSError):
        pass


def _env(home: str) -> dict[str, str]:
    keep = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT") if k in os.environ}
    return {**keep, "HOME": home, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}


def run_suite(code: str, suite: str, name: str, timeout: float = 20.0) -> tuple[set[str], dict[str, str]]:
    """Run one test module against one implementation. Returns (passed ids, {failed id: output})."""
    with tempfile.TemporaryDirectory(prefix="kgirl-build-") as tmp:
        Path(tmp, "solution.py").write_text(code, encoding="utf-8")
        Path(tmp, f"{name}.py").write_text(suite, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-E", "-s", "-m", "unittest", "-v", name], cwd=tmp, env=_env(tmp),
                               capture_output=True, text=True, timeout=timeout,
                               preexec_fn=_limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            return set(), {f"{name}:*": "timed out"}
        out = r.stderr + r.stdout
    passed: set[str] = set()
    failed: dict[str, str] = {}
    hits = _RESULT.findall(out)
    for method, where, status in hits:
        parts = where.split(".")
        cls = parts[-2] if parts[-1] == method and len(parts) > 1 else parts[-1]
        tid = f"{name}:{cls}.{method}"
        if status == "ok":
            passed.add(tid)
        elif status in ("FAIL", "ERROR"):
            failed[tid] = _failure_text(out, method)
    if not hits:
        failed[f"{name}:*"] = out[-1500:] or "suite did not run"
    return passed, failed


def _failure_text(out: str, method: str) -> str:
    m = re.search(rf"^(?:FAIL|ERROR): {method} .*?(?=^(?:FAIL|ERROR): |^-{{20,}}\nRan |\Z)", out, re.M | re.S)
    return m.group(0).strip()[-1200:] if m else ""


def _imports(code: str, timeout: float) -> bool:
    with tempfile.TemporaryDirectory(prefix="kgirl-build-") as tmp:
        Path(tmp, "solution.py").write_text(code, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-E", "-s", "-c", "import solution"], cwd=tmp, env=_env(tmp),
                               capture_output=True, timeout=timeout, preexec_fn=_limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0


def _compiles(code: str) -> bool:
    try:
        compile(code, "solution.py", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def merge_suites(suites: list[str]) -> str:
    """The generated suites as one readable file, each under its own banner."""
    return "\n\n".join(f"# ---- generated suite {i + 1} (from the spec, before any code) ----\n{s.strip()}\n"
                       for i, s in enumerate(suites))


def spec_from_text(text: str) -> str:
    return strip_reasoning(text).strip()

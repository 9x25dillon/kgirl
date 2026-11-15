"""Spec -> code: test-first generation with sampling and agreement selection.

A small local model asked once gets a spec right only some of the time, and it cannot tell which time.
The builder does not trust any single sample, and it does not trust its own tests either:

    1. recall    exemplars (earlier specs with code that was accepted) as worked examples
    2. tests     K independent unittest suites written from the spec alone
    3. code      N implementations, alternating "direct" and "edge cases first" prompts, spread temperatures
    4. gate      a candidate must parse, pass the static check (no network, processes or, unless allowed,
                 file access) and import cleanly in the sandbox; the others are dropped with the reason
    5. matrix    run every implementation against every suite in the sandbox, test by test
    6. agree     group implementations that pass exactly the same tests; score = members x tests passed
                 (dual agreement, as in CodeT: wrong code rarely agrees with other wrong code, and wrong
                 tests rarely pass agreeing code). Pick the best group's shortest member.
    7. repair    show the pick the failures that some other implementation passes (so the test is
                 satisfiable) and ask for a fix; keep it only if it passes a strict superset of tests.
    8. deliver   the module, and each suite cut down to the tests the module passes ("agreed" tests). The
                 credible tests it still fails are listed as disputed: a wrong test or a real defect.
    9. learn     the spec and the result become an `exemplar`: staged by default, active when the person
                 accepts it. Active exemplars feed step 1, so accepted work makes later builds better
                 (this is the builder's self-improvement, through memory, not weights).

Generated code runs through sandbox.py: a temp dir, CPU, memory and wall-clock limits, a scrubbed
environment, and bubblewrap isolation (no network, no home directory) where the machine has it.
"""

from __future__ import annotations

import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from ..soup import Soup
from . import sandbox
from .sandbox import run_suite
from .suites import merge_suites, prune
from .text import one_line, python_code, strip_reasoning

__all__ = ["Builder", "BuildResult", "Candidate", "IMPL_SYSTEM", "TEST_SYSTEM", "REPAIR_SYSTEM", "run_suite",
           "merge_suites", "spec_from_text"]

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

TEMPERATURES = (0.2, 0.5, 0.8, 0.35, 0.65, 0.9)
SUITE_TEMPERATURES = (0.2, 0.5, 0.8, 0.4)


def suite_name(k: int) -> str:
    return f"test_gen{k}"


@dataclass
class Candidate:
    idx: int
    code: str
    temperature: float
    style: str
    passed: frozenset[str] = frozenset()
    failed: dict[str, str] = field(default_factory=dict)
    repaired: bool = False
    dropped: str = ""            # why it never reached the test matrix ("" when it did)

    @property
    def ok(self) -> bool:
        return bool(self.code) and not self.dropped


@dataclass
class BuildResult:
    spec: str
    code: str
    tests: str                       # the agreed tests of every suite, as one text for display
    chosen: int | None
    group_size: int
    passed: int
    credible: int
    candidates: list[Candidate]
    suites: list[str]                # the suites as generated
    exemplars: list[int]
    notes: list[str] = field(default_factory=list)
    agreed: list[str] = field(default_factory=list)       # each suite cut down to the tests the module passes
    disputed: dict[str, str] = field(default_factory=dict)  # credible tests the module fails -> their output
    sandbox: str = "none"

    @property
    def confidence(self) -> float:
        """Share of credible tests passed, discounted when few implementations agreed."""
        if not self.code or not self.credible:
            return 0.0
        agree = min(1.0, self.group_size / max(2, len([c for c in self.candidates if c.ok]) / 2))
        return round(self.passed / self.credible * (0.5 + 0.5 * agree), 3)

    def render(self) -> str:
        if not self.code:
            lines = ["build failed: no usable module"]
        else:
            lines = [f"built: candidate #{self.chosen} (group of {self.group_size}), passes {self.passed}/"
                     f"{self.credible} credible tests, confidence {self.confidence:.2f}"]
        for c in self.candidates:
            mark = "★" if c.idx == self.chosen and self.code else ("·" if c.ok else "✗")
            tail = f"dropped: {c.dropped}" if c.dropped else ("repaired" if c.repaired else "")
            lines.append(f"  {mark} #{c.idx} {c.style:<10} T={c.temperature:.1f} passed={len(c.passed):<3} {tail}".rstrip())
        lines += [f"  note: {n}" for n in self.notes]
        if self.disputed:
            ids = sorted(self.disputed)
            more = f" (+{len(ids) - 6} more)" if len(ids) > 6 else ""
            lines.append(f"  disputed (the module fails these; the test or the module is wrong): "
                         f"{', '.join(ids[:6])}{more}")
        if self.exemplars:
            lines.append(f"  recalled exemplars: {', '.join('#' + str(i) for i in self.exemplars)}")
        lines.append(f"  sandbox: {sandbox.describe(self.sandbox)}")
        return "\n".join(lines)


class Builder:
    def __init__(self, llm: LLM, soup: Soup | None = None, n_impl: int = 6, n_suites: int = 3, repair_rounds: int = 2,
                 timeout: float = 20.0, workers: int = 4, emit: Callable[[str, dict], None] | None = None,
                 allow_io: bool = False):
        self.llm = llm
        self.soup = soup
        self.n_impl = n_impl
        self.n_suites = n_suites
        self.repair_rounds = repair_rounds
        self.timeout = timeout
        self.workers = workers
        self.emit = emit or (lambda topic, payload: None)
        self.allow_io = allow_io

    # ------------------------------------------------------------------ public

    def build(self, spec: str, scope: str = "*") -> BuildResult:
        spec = textwrap.dedent(spec).strip()
        level = sandbox.level()
        shots, ex_ids = self._exemplars(spec, scope)
        self.emit("build.start", {"spec_chars": len(spec), "exemplars": ex_ids, "sandbox": level})
        notes: list[str] = []
        drafted_suites = self._parallel(lambda i: self._suite(spec, i), range(self.n_suites))
        suites = [code for code, _ in drafted_suites if code]
        notes += [f"suite {i + 1} dropped: {why}" for i, (code, why) in enumerate(drafted_suites) if not code]
        drafted = self._parallel(lambda i: self._candidate(spec, shots, i), range(self.n_impl))
        live = [c for c in drafted if c.ok]
        if not live:
            notes.append("no candidate parsed, passed the static check and imported")
            self.emit("build.end", {"chosen": None, "candidates": len(drafted)})
            return BuildResult(spec=spec, code="", tests="", chosen=None, group_size=0, passed=0, credible=0,
                               candidates=drafted, suites=suites, exemplars=ex_ids, notes=notes, sandbox=level)
        if not suites:
            notes.append("no test suite survived, so nothing was checked: picked the shortest module that imports")
        self._parallel(lambda c: self._score(c, suites), live)
        pick, group = self._agree(live)
        credible = self._credible(live)
        for rnd in range(self.repair_rounds):
            missing = {t: pick.failed[t] for t in sorted(credible) if t in pick.failed}
            if not missing:
                break
            fixed = self._repair(spec, pick, missing, rnd)
            if fixed is None:
                notes.append(f"repair round {rnd + 1} produced no usable module")
                continue
            self._score(fixed, suites)
            drafted.append(fixed)
            if fixed.passed > pick.passed:
                notes.append(f"repair round {rnd + 1}: {len(pick.passed)} -> {len(fixed.passed)} tests")
                live.append(fixed)
                pick, group = fixed, [fixed]
                credible = self._credible(live)
            else:
                fixed.dropped = "did not pass a strict superset of the tests"
                notes.append(f"repair round {rnd + 1} did not pass a strict superset; kept the original")
        agreed = [s for s in (prune(suite, suite_name(k), pick.passed) for k, suite in enumerate(suites)) if s]
        disputed = {t: pick.failed[t] for t in sorted(credible) if t in pick.failed}
        res = BuildResult(spec=spec, code=pick.code, tests=merge_suites(agreed), chosen=pick.idx,
                          group_size=len(group), passed=len(pick.passed & credible), credible=len(credible),
                          candidates=drafted, suites=suites, exemplars=ex_ids, notes=notes, agreed=agreed,
                          disputed=disputed, sandbox=level)
        self.emit("build.end", {"chosen": pick.idx, "group": len(group), "passed": res.passed,
                                "credible": res.credible, "confidence": res.confidence, "sandbox": level})
        return res

    def learn(self, res: BuildResult, accepted: bool, scope: str = "*", note: str = "",
              origin: str | None = None) -> int | None:
        """Store the spec and the result as an exemplar: active when a person accepted it, else staged.

        `origin` names who asked for the build when it was not the person at the keyboard (for example
        "mcp"): the exemplar is then stored under that source, which the curator never promotes by itself.
        """
        if not self.soup or not res.code:
            return None
        head = " ".join(res.spec.split())[:600]
        text = f"SPEC: {head}\n\nSOLUTION:\n```python\n{res.code.strip()}\n```" + (f"\n\nNOTE: {note}" if note else "")
        source = "user" if accepted else (origin or "build")
        fid = self.soup.add(text, kind="exemplar", tags="build", source=source, scope=scope,
                            status="active" if accepted else "staged")
        self.soup.credit([fid], success=True)
        if accepted:
            self.soup.set_status(fid, "active", "accepted by the user")
            self.soup.credit(res.exemplars, success=True)      # the worked examples it was built from helped
        return fid

    # ------------------------------------------------------------------ generation

    def _exemplars(self, spec: str, scope: str) -> tuple[str, list[int]]:
        if not self.soup:
            return "", []
        rec = self.soup.recall(spec, budget_tokens=1400, kinds=("exemplar",), scope=scope, fallback=False)
        if not rec.fragments:
            return "", []
        shots = "\n\n".join(f"EXAMPLE {i + 1} (accepted earlier):\n{f.text}" for i, f in enumerate(rec.fragments[:2]))
        return shots, rec.ids[:2]

    def _candidate(self, spec: str, shots: str, i: int) -> Candidate:
        t = TEMPERATURES[i % len(TEMPERATURES)]
        style = "edge-first" if i % 2 else "direct"
        prompt = (f"{shots}\n\n" if shots else "") + f"SPEC:\n{spec}" + (f"\n\n{EDGE_FIRST}" if style == "edge-first" else "")
        try:
            code = python_code(self.llm(IMPL_SYSTEM, prompt, 2048, t))
        except Exception as exc:  # noqa: BLE001 - one failed sample must not sink the build
            return Candidate(i, "", t, style, dropped=f"the model call failed: {one_line(str(exc), 120)}")
        return self._gate(Candidate(i, code, t, style))

    def _gate(self, c: Candidate) -> Candidate:
        """Mark a candidate dropped unless it parses, passes the static check and imports in the sandbox."""
        if not c.code:
            c.dropped = "no Python module in the reply"
            return c
        problems = sandbox.lint(c.code, self.allow_io)
        if problems:
            c.dropped = "; ".join(problems[:3])
            return c
        ok, err = sandbox.imports_ok(c.code, self.timeout)
        if not ok:
            c.dropped = "does not import: " + one_line(err.strip().splitlines()[-1] if err.strip() else "no output", 160)
        return c

    def _suite(self, spec: str, i: int) -> tuple[str, str]:
        """(suite source, "") or ("", why it was dropped)."""
        try:
            code = python_code(self.llm(TEST_SYSTEM, f"SPEC:\n{spec}", 2048, SUITE_TEMPERATURES[i % len(SUITE_TEMPERATURES)]))
        except Exception as exc:  # noqa: BLE001
            return "", f"the model call failed: {one_line(str(exc), 120)}"
        if not code:
            return "", "no Python module in the reply"
        if "solution" not in code:
            return "", "it does not import the solution module"
        problems = sandbox.lint(code, allow_io=True)     # tests may use temp files; never the network or processes
        if problems:
            return "", "; ".join(problems[:3])
        return code, ""

    def _repair(self, spec: str, pick: Candidate, missing: dict[str, str], rnd: int) -> Candidate | None:
        fails = "\n\n".join(f"--- {t}\n{out[-900:]}" for t, out in list(missing.items())[:6])
        prompt = f"SPEC:\n{spec}\n\nCURRENT solution.py:\n```python\n{pick.code}```\n\nFAILING TESTS:\n{fails}"
        temperature = 0.2 + 0.2 * rnd
        try:
            code = python_code(self.llm(REPAIR_SYSTEM, prompt, 2048, temperature))
        except Exception:  # noqa: BLE001
            return None
        fixed = self._gate(Candidate(100 + rnd, code, temperature, "repair", repaired=True))
        return fixed if fixed.ok else None

    # ------------------------------------------------------------------ evaluation

    def _score(self, c: Candidate, suites: list[str]) -> Candidate:
        passed: set[str] = set()
        failed: dict[str, str] = {}
        for k, suite in enumerate(suites):
            p, f = run_suite(c.code, suite, suite_name(k), self.timeout)
            passed |= p
            failed.update(f)
        c.passed, c.failed = frozenset(passed), failed
        return c

    @staticmethod
    def _agree(cands: list[Candidate]) -> tuple[Candidate, list[Candidate]]:
        """The largest-scoring group of candidates that pass exactly the same tests, and its shortest member."""
        groups: dict[frozenset[str], list[Candidate]] = {}
        for c in cands:
            groups.setdefault(c.passed, []).append(c)
        members = max(groups.items(), key=lambda kv: (len(kv[1]) * max(1, len(kv[0])), len(kv[0]),
                                                      -min(len(c.code) for c in kv[1])))[1]
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
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(items)))) as pool:
            return list(pool.map(fn, items))


def spec_from_text(text: str) -> str:
    return strip_reasoning(text).strip()

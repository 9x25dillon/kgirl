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
from pathlib import Path
from typing import Callable, Sequence

from ..soup import Soup
from . import sandbox
from .build_spec import (atlas_context, build_summary_dict, candidate_prompt, donor_snippet, emit_files,
                         explain_build, format_failures, has_tests, missing_exports, normalize_suite,
                         parse_multifile_reply, rank_exemplars, spec_checklist, spec_query, suite_prompt,
                         test_weights, weighted_test_count)
from .sandbox import run_suite
from .suites import merge_suites, prune
from .text import one_line, python_code, strip_reasoning

__all__ = ["Builder", "BuildResult", "Candidate", "EDGE_FIRST", "IMPL_SYSTEM", "TEST_SYSTEM", "REPAIR_SYSTEM",
           "run_suite", "merge_suites", "spec_from_text"]

LLM = Callable[[str, str, int, float], str]   # (system, prompt, max_tokens, temperature) -> text

IMPL_SYSTEM = """You are an expert Python engineer. Write the module `solution.py` that implements the SPEC
exactly: the names, signatures, @property attributes, return values and exceptions it states. Write elegant,
idiomatic, typed code with short docstrings: clear names, small functions, the right data structure for the
complexity the spec asks for, every edge case the spec mentions handled explicitly. Never mutate input
arguments unless the spec asks for in-place modification. Standard library only. No I/O, no prints, no
example usage, no tests. Reply with one ```python block containing the whole module."""

EDGE_FIRST = """Before the code, list every edge case, boundary input, invariant and error condition the spec
states or implies, as a short bullet list. Then give the whole module in one ```python block that handles
each of them."""

TEST_SYSTEM = """You write precise unit tests from a specification, before any implementation exists.
Write a Python unittest module that imports from `solution` exactly the names the SPEC defines. Cover the
examples in the spec, every stated edge case and error (assertRaises with the stated exception type), and
a few ordinary cases whose expected values you compute carefully by hand. Put one assertion per test_*
method; 6 to 14 methods; descriptive names. Access @property attributes without parentheses. Do not test
anything the spec leaves unspecified (exact error messages, internal structure, performance). Reply with
one ```python block containing the whole module."""

REPAIR_SYSTEM = """You fix a Python module so it satisfies its SPEC. You get the spec, the current module and
the tests it fails, with their output and test source. Preserve every behaviour that already works; only
change what is needed to fix the failing cases. Tests can be wrong: follow the SPEC when a test disagrees
with it. Reply with the whole corrected module in one ```python block."""

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
    extra_files: dict[str, str] = field(default_factory=dict)

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

    def explain(self) -> str:
        """Detailed vote matrix, suite breakdown, repair trace, and why the winner won (`--explain`)."""
        return explain_build(self)

    def summary_dict(self) -> dict:
        """JSON-serializable representation of this build result, including candidate votes and explanation."""
        return build_summary_dict(self)

    def files(self, out_name: str = "solution.py") -> dict[str, str]:
        """Map of `{filename: source}` for the module, any companion modules, and retargeted agreed tests."""
        return emit_files(self.code, self.agreed, out_name=out_name, extra_files=self.extra_files)

    def write(self, out: str | Path) -> list[str]:
        """Write the module, any companion modules, and agreed test files beside `out`; return written paths."""
        out_path = Path(out)
        written: list[str] = []
        for fname, content in self.files(out_path.name).items():
            dest = out_path if fname == out_path.name else out_path.with_name(fname)
            dest.write_text(content, encoding="utf-8")
            written.append(str(dest))
        return written


class Builder:
    def __init__(self, llm: LLM, soup: Soup | None = None, n_impl: int = 6, n_suites: int = 3, repair_rounds: int = 2,
                 timeout: float = 20.0, workers: int = 4, emit: Callable[[str, dict], None] | None = None,
                 allow_io: bool = False, use_exemplars: bool = True):
        self.llm = llm
        self.soup = soup
        self.n_impl = n_impl
        self.n_suites = n_suites
        self.repair_rounds = repair_rounds
        self.timeout = timeout
        self.workers = workers
        self.emit = emit or (lambda topic, payload: None)
        self.allow_io = allow_io
        self.use_exemplars = use_exemplars

    # ------------------------------------------------------------------ public

    def build(self, spec: str, scope: str = "*", context: str = "",
              extra_files: dict[str, str] | None = None) -> BuildResult:
        spec = textwrap.dedent(spec).strip()
        full_spec = f"{spec}\n\n{context.strip()}" if context and context.strip() else spec
        companion: dict[str, str] = {}
        notes: list[str] = []
        for fname, body in (extra_files or {}).items():
            problems = sandbox.lint(body, self.allow_io)
            if problems:
                notes.append(f"extra file {fname} dropped: {'; '.join(problems[:3])}")
            else:
                companion[fname] = body
        level = sandbox.level()
        shots, ex_ids = self._exemplars(spec, scope)
        self.emit("build.start", {"spec_chars": len(spec), "exemplars": ex_ids, "sandbox": level})
        drafted_suites = self._parallel(lambda i: self._suite(full_spec, i), range(self.n_suites))
        suites = [code for code, _ in drafted_suites if code]
        notes += [f"suite {i + 1} dropped: {why}" for i, (code, why) in enumerate(drafted_suites) if not code]
        drafted = self._parallel(lambda i: self._candidate(full_spec, shots, i), range(self.n_impl))
        live = [c for c in drafted if c.ok]
        if not live:
            notes.append("no candidate parsed, passed the static check and imported")
            self.emit("build.end", {"chosen": None, "candidates": len(drafted)})
            return BuildResult(spec=spec, code="", tests="", chosen=None, group_size=0, passed=0, credible=0,
                               candidates=drafted, suites=suites, exemplars=ex_ids, notes=notes, sandbox=level,
                               extra_files=companion)
        if not suites:
            notes.append("no test suite survived, so nothing was checked: picked the shortest module that imports")
        self._parallel(lambda c: self._score(c, suites), live)
        pick, group = self._agree(live, full_spec)
        credible = self._credible(live)
        regressed: dict[str, str] = {}
        for rnd in range(self.repair_rounds):
            support = self._test_support(live)
            missing = {t: pick.failed[t] for t in sorted(credible, key=lambda t: (-support.get(t, 0), t))
                       if t in pick.failed}
            pick_mismatch = missing_exports(full_spec, pick.code)
            if not missing and not pick_mismatch:
                break
            fixed = self._repair(full_spec, pick, missing, rnd, suites=suites, regressed=regressed, live=live)
            if fixed is None:
                notes.append(f"repair round {rnd + 1} produced no usable module")
                continue
            self._score(fixed, suites)
            drafted.append(fixed)
            fixed_mismatch = missing_exports(full_spec, fixed.code)
            if fixed.passed > pick.passed or (
                fixed.passed >= pick.passed and len(fixed_mismatch) < len(pick_mismatch)
            ):
                notes.append(f"repair round {rnd + 1}: {len(pick.passed)} -> {len(fixed.passed)} tests")
                live.append(fixed)
                group = [c for c in live if c.passed == fixed.passed]
                pick = fixed
                credible = self._credible(live)
                regressed = {}
            else:
                regressed = {t: fixed.failed.get(t, "regressed") for t in sorted(pick.passed - fixed.passed)}
                fixed.dropped = "did not pass a strict superset of the tests"
                notes.append(f"repair round {rnd + 1} did not pass a strict superset; kept the original")
        agreed = [s for s in (prune(suite, suite_name(k), pick.passed) for k, suite in enumerate(suites)) if s]
        disputed = {t: pick.failed[t] for t in sorted(credible) if t in pick.failed}
        res = BuildResult(spec=spec, code=pick.code, tests=merge_suites(agreed), chosen=pick.idx,
                          group_size=len(group), passed=len(pick.passed & credible), credible=len(credible),
                          candidates=drafted, suites=suites, exemplars=ex_ids, notes=notes, agreed=agreed,
                          disputed=disputed, sandbox=level, extra_files=companion)
        self.emit("build.end", {"chosen": pick.idx, "group": len(group), "passed": res.passed,
                                "credible": res.credible, "confidence": res.confidence, "sandbox": level})
        return res

    def build_with_atlas(self, spec: str, atlas, repo: str, paths: str | Sequence[str],
                         scope: str | None = None) -> BuildResult:
        """Build against one or more existing files in an indexed Atlas repository."""
        ctx, extra = atlas_context(atlas, repo, paths)
        return self.build(spec, scope=scope or repo or "*", context=ctx, extra_files=extra)

    def confirm(self, res: BuildResult, scope: str = "*", note: str = "") -> int | None:
        """Record a user-confirmed winner (`/accept` or `--accept`): active at once as the user's word."""
        return self.learn(res, accepted=True, scope=scope, note=note)

    def learn(self, res: BuildResult, accepted: bool = False, scope: str = "*", note: str = "",
              origin: str | None = None) -> int | None:
        """Store the spec and the result as an exemplar: active when a person accepted it, else staged.

        `origin` names who asked for the build when it was not the person at the keyboard (for example
        "mcp"): the exemplar is then stored under that source and never made active automatically.
        """
        if not self.soup or not res.code:
            return None
        active = bool(accepted and origin != "mcp")
        head = " ".join(res.spec.split())[:600]
        text = f"SPEC: {head}\n\nSOLUTION:\n```python\n{res.code.strip()}\n```" + (f"\n\nNOTE: {note}" if note else "")
        source = "user" if active else (origin or "build")
        data = res.tests.strip() or None
        fid = self.soup.add(text, kind="exemplar", tags="build", source=source, scope=scope,
                            status="active" if active else "staged", data=data)
        self.soup.credit([fid], success=True)
        if active:
            self.soup.set_status(fid, "active", "accepted by the user")
            self.soup.credit(res.exemplars, success=True)      # the worked examples it was built from helped
        return fid

    # ------------------------------------------------------------------ generation

    def _exemplars(self, spec: str, scope: str) -> tuple[str, list[int]]:
        if not self.soup or not self.use_exemplars:
            return "", []
        query = spec_query(spec)
        rec = self.soup.recall(query, budget_tokens=1400, kinds=("exemplar",), scope=scope, fallback=False)
        if not rec.fragments and query != spec:
            rec = self.soup.recall(spec, budget_tokens=1400, kinds=("exemplar",), scope=scope, fallback=False)
        picked = rank_exemplars(spec, rec.fragments, limit=2)
        if not picked:
            return "", []
        shots = "\n\n".join(f"EXAMPLE {i + 1} (accepted earlier):\n{f.text}" for i, f in enumerate(picked))
        return shots, [f.id for f in picked]

    def _candidate(self, spec: str, shots: str, i: int) -> Candidate:
        t = TEMPERATURES[i % len(TEMPERATURES)]
        style = "edge-first" if i % 2 else "direct"
        prompt = candidate_prompt(spec, shots, i, EDGE_FIRST)
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
        prompt = suite_prompt(spec, i)
        try:
            code = python_code(self.llm(TEST_SYSTEM, prompt, 2048, SUITE_TEMPERATURES[i % len(SUITE_TEMPERATURES)]))
        except Exception as exc:  # noqa: BLE001
            return "", f"the model call failed: {one_line(str(exc), 120)}"
        if not code:
            return "", "no Python module in the reply"
        if "solution" not in code:
            return "", "it does not import the solution module"
        problems = sandbox.lint(code, allow_io=True)     # tests may use temp files; never the network or processes
        if problems:
            return "", "; ".join(problems[:3])
        code = normalize_suite(code)
        if not has_tests(code):
            return "", "it defines no test methods"
        return code, ""

    def _repair(self, spec: str, pick: Candidate, missing: dict[str, str], rnd: int,
                suites: list[str] | tuple[str, ...] = (), regressed: dict[str, str] | None = None,
                live: Sequence[Candidate] = ()) -> Candidate | None:
        fails = (format_failures(suites, missing, limit=6) if missing
                 else "(all credible generated tests passed; fix the CONTRACT MISMATCH below)")
        prompt = f"SPEC:\n{spec}\n\nCURRENT solution.py:\n```python\n{pick.code}```\n\nFAILING TESTS:\n{fails}"
        donor = donor_snippet(live, pick, missing)
        if donor:
            prompt += f"\n\n{donor}"
        if regressed:
            prompt += f"\n\nREGRESSED IN PREVIOUS REPAIR (keep these passing):\n{format_failures(suites, regressed, limit=4)}"
        mismatch = missing_exports(spec, pick.code)
        if mismatch:
            prompt += "\n\nCONTRACT MISMATCH IN CURRENT solution.py:\n" + "\n".join(f"- {m}" for m in mismatch)
        checklist = spec_checklist(spec)
        if checklist:
            prompt += f"\n\n{checklist}"
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
    def _test_support(cands: list[Candidate]) -> dict[str, int]:
        support: dict[str, int] = {}
        for c in cands:
            for t in c.passed:
                support[t] = support.get(t, 0) + 1
        return support

    @staticmethod
    def _agree(cands: list[Candidate], spec: str = "") -> tuple[Candidate, list[Candidate]]:
        """The largest-scoring group of candidates that pass the same tests, and its shortest member.

        Score is CodeT dual agreement (`members * weighted_tests_passed`), where `test_<name>_1..N`
        siblings split from one multi-assert method carry weight `1/N` each so splitting a method does
        not inflate its suite's vote weight, and 0-test groups score 0. Ties prefer higher weighted test
        count, more assertions passed, broader suite coverage, higher per-test candidate consensus, fewer
        static contract mismatches against `spec`, and shorter code.
        """
        groups: dict[frozenset[str], list[Candidate]] = {}
        all_tids: list[str] = []
        for c in cands:
            groups.setdefault(c.passed, []).append(c)
            all_tids.extend(c.passed)
            all_tids.extend(c.failed)
        support = Builder._test_support(cands)
        weights = test_weights(all_tids)
        mismatches = {c.idx: len(missing_exports(spec, c.code)) for c in cands} if spec else {}
        members = max(
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
        )[1]
        return min(members, key=lambda c: (mismatches.get(c.idx, 0), len(c.code), c.idx)), members

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

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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..atlas.secrets import secret_reason
from ..soup import Soup
from . import sandbox
from .build_spec import (apply_unified_diff, atlas_context, atlas_search_context, build_short_summary,
                         build_summary_dict, candidate_prompt, donor_snippet, edit_candidate_prompt,
                         edit_suite_prompt, emit_files, explain_build, extract_diff, find_existing_tests,
                         format_failures, has_tests, lint_spec, missing_exports, multifile_candidate_prompt,
                         multifile_suite_prompt, normalize_suite, parse_multifile_reply, parse_multifile_spec,
                         rank_exemplars, select_exemplars_within_budget, spec_checklist, spec_query, suite_prompt,
                         test_weights, unified_diff, weighted_test_count)
from .sandbox import run_suite
from .suites import merge_suites, prune, retarget
from .text import one_line, python_code, strip_reasoning

__all__ = ["Builder", "BuildResult", "Candidate", "EDGE_FIRST", "IMPL_SYSTEM", "MULTIFILE_IMPL_SYSTEM",
           "EDIT_SYSTEM", "EDIT_REPAIR_SYSTEM", "TEST_SYSTEM", "REPAIR_SYSTEM",
           "edit_build", "run_suite", "merge_suites", "spec_from_text"]

LLM = Callable[[str, str, int, float], str]   # (system, prompt, max_tokens, temperature) -> text

IMPL_SYSTEM = """You are an expert Python engineer. Write the module `solution.py` that implements the SPEC
exactly: the names, signatures, @property attributes, return values and exceptions it states. Write elegant,
idiomatic, typed code with short docstrings: clear names, small functions, the right data structure for the
complexity the spec asks for, every edge case the spec mentions handled explicitly. Never mutate input
arguments unless the spec asks for in-place modification. Standard library only. No I/O, no prints, no
example usage, no tests. Reply with one ```python block containing the whole module."""

MULTIFILE_IMPL_SYSTEM = """You are an expert Python engineer. Implement every module named in the SPEC
(`## module: <name.py>`) so they work together as one clean unit: exact names, signatures, @property
attributes, return values and exceptions. Standard library only. No I/O, no prints, no example usage, no
tests. Reply with one ```python <filename.py> block for each required module."""

EDIT_SYSTEM = """You are an expert Python engineer editing an existing module. Make the smallest clean change
that satisfies the SPEC while preserving all existing behaviour and public APIs that the SPEC does not change.
Standard library only. No prints or unrelated rewrites. Reply with one ```diff block containing a valid
unified diff against the existing file."""

EDIT_REPAIR_SYSTEM = """You fix a unified diff against an existing Python module so it passes every existing
test and every new spec test. You get the spec, the original file, the current patched file, and the exact
failing test IDs with trimmed tracebacks and test source. Preserve all passing behaviour. Reply with one
```diff block containing a complete unified diff against the ORIGINAL file."""

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
the tests it fails, with their exact failing test IDs, trimmed tracebacks and test source. Preserve every
behaviour that already works; only change what is needed to fix the failing cases. Tests can be wrong:
follow the SPEC when a test disagrees with it. Reply with the whole corrected module in one ```python block."""

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
    files: dict[str, str] = field(default_factory=dict)  # multi-file builds: {filename: source}
    diff_adds: int = 0           # edit mode: added lines in candidate diff
    diff_dels: int = 0           # edit mode: removed lines in candidate diff

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
    diff: str = ""                   # edit mode: unified diff for review, never applied automatically
    edit_target: str = ""            # edit mode: display path of the target file
    existing_passed: int = 0         # edit mode: existing tests passed by the winner
    existing_total: int = 0          # edit mode: total existing tests run
    spec_warnings: list[str] = field(default_factory=list)
    context_picks: list[dict] = field(default_factory=list)  # auto-retrieved Atlas context files + reasons

    @property
    def confidence(self) -> float:
        """Share of credible tests passed, discounted when few implementations agreed."""
        if not self.code:
            return 0.0
        total = self.credible + (self.existing_total if self.edit_target else 0)
        passed = self.passed + (self.existing_passed if self.edit_target else 0)
        if not total:
            return 0.0
        agree = min(1.0, self.group_size / max(2, len([c for c in self.candidates if c.ok]) / 2))
        return round(passed / total * (0.5 + 0.5 * agree), 3)

    def render(self) -> str:
        if self.edit_target:
            if not self.code or not self.diff:
                lines = [f"edit failed for {self.edit_target}: no candidate passed existing + spec tests"]
            else:
                lines = [f"edit ready for {self.edit_target}: candidate #{self.chosen} (group of {self.group_size}), "
                         f"existing {self.existing_passed}/{self.existing_total}, "
                         f"spec {self.passed}/{self.credible} credible tests, confidence {self.confidence:.2f} "
                         "(diff for review, not applied)"]
        elif not self.code:
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

    def summary(self) -> str:
        """Short (~<=6 lines) chat-facing summary of what won and why."""
        return build_short_summary(self)

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
                 allow_io: bool = False, use_exemplars: bool = True, exemplar_tokens: int = 900):
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
        self.exemplar_tokens = exemplar_tokens

    # ------------------------------------------------------------------ public

    def build(self, spec: str, scope: str = "*", context: str = "",
              extra_files: dict[str, str] | None = None, strict: bool = False,
              budget_s: float | None = None, atlas=None, repo: str | None = None,
              context_tokens: int = 1500) -> BuildResult:
        spec = textwrap.dedent(spec).strip()
        spec_warnings = lint_spec(spec)
        auto_ctx, auto_extra, context_picks = (
            atlas_search_context(atlas, spec, repo=repo, exclude_paths=list((extra_files or {}).keys()),
                                 max_tokens=context_tokens)
            if atlas is not None
            else ("", {}, [])
        )
        ctx_parts = [p for p in (context.strip() if context else "", auto_ctx) if p]
        full_spec = f"{spec}\n\n" + "\n\n".join(ctx_parts) if ctx_parts else spec
        merged_extra = {**auto_extra, **(extra_files or {})}
        companion: dict[str, str] = {}
        notes: list[str] = []
        for fname, body in merged_extra.items():
            problems = sandbox.lint(body, self.allow_io)
            if problems:
                notes.append(f"extra file {fname} dropped: {'; '.join(problems[:3])}")
            else:
                companion[fname] = body
        level = sandbox.level()
        if strict and spec_warnings:
            notes.append("stopped early (--strict): " + "; ".join(spec_warnings))
            self.emit("build.end", {"chosen": None, "candidates": 0, "strict_warnings": len(spec_warnings)})
            return BuildResult(spec=spec, code="", tests="", chosen=None, group_size=0, passed=0, credible=0,
                               candidates=[], suites=[], exemplars=[], notes=notes, sandbox=level,
                               extra_files=companion, spec_warnings=spec_warnings, context_picks=context_picks)
        deadline = (time.monotonic() + max(0.0, budget_s)) if budget_s is not None else None
        mod_sections = parse_multifile_spec(spec)
        multifile_mods = list(mod_sections.keys()) if len(mod_sections) >= 2 else []
        primary_mod = multifile_mods[0] if multifile_mods else "solution.py"
        primary_stem = Path(primary_mod).stem
        shots, ex_ids = self._exemplars(spec, scope)
        self.emit("build.start", {"spec_chars": len(spec), "exemplars": ex_ids, "sandbox": level})

        drafted_suites, suite_cut = self._run_items(
            lambda i: self._suite(full_spec, i, modules=multifile_mods), range(self.n_suites), deadline
        )
        if suite_cut:
            notes.append(f"budget reached ({budget_s:g}s): stopped suite sampling after {len(drafted_suites)}/{self.n_suites}")
        suites = [code for code, _ in drafted_suites if code]
        notes += [f"suite {i + 1} dropped: {why}" for i, (code, why) in enumerate(drafted_suites) if not code]

        drafted, cand_cut = self._run_items(
            lambda i: self._candidate(full_spec, shots, i, modules=multifile_mods, companion=companion),
            range(self.n_impl),
            deadline,
        )
        if cand_cut:
            notes.append(f"budget reached ({budget_s:g}s): stopped candidate sampling after {len(drafted)}/{self.n_impl}")
        live = [c for c in drafted if c.ok]
        if not live:
            notes.append("no candidate parsed, passed the static check and imported")
            self.emit("build.end", {"chosen": None, "candidates": len(drafted)})
            return BuildResult(spec=spec, code="", tests="", chosen=None, group_size=0, passed=0, credible=0,
                               candidates=drafted, suites=suites, exemplars=ex_ids, notes=notes, sandbox=level,
                               extra_files=companion, spec_warnings=spec_warnings, context_picks=context_picks)
        if not suites:
            notes.append("no test suite survived, so nothing was checked: picked the shortest module that imports")
        self._parallel(lambda c: self._score(c, suites, module_name=primary_stem, companion=companion), live)
        pick, group = self._agree(live, full_spec)
        credible = self._credible(live)
        regressed: dict[str, str] = {}
        for rnd in range(self.repair_rounds):
            if self._over_budget(deadline):
                notes.append(f"budget reached ({budget_s:g}s): skipped repair round {rnd + 1}")
                break
            support = self._test_support(live)
            missing = {t: pick.failed[t] for t in sorted(credible, key=lambda t: (-support.get(t, 0), t))
                       if t in pick.failed}
            pick_mismatch = [] if multifile_mods else missing_exports(full_spec, pick.code)
            if not missing and not pick_mismatch:
                break
            fixed = self._repair(full_spec, pick, missing, rnd, suites=suites, regressed=regressed, live=live,
                                 modules=multifile_mods, companion=companion)
            if fixed is None:
                notes.append(f"repair round {rnd + 1} produced no usable module")
                continue
            self._score(fixed, suites, module_name=primary_stem, companion=companion)
            drafted.append(fixed)
            fixed_mismatch = [] if multifile_mods else missing_exports(full_spec, fixed.code)
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
        out_extra = dict(companion)
        if pick.files:
            for fname, body in pick.files.items():
                if fname != primary_mod:
                    out_extra[fname] = body
        res = BuildResult(spec=spec, code=pick.code, tests=merge_suites(agreed), chosen=pick.idx,
                          group_size=len(group), passed=len(pick.passed & credible), credible=len(credible),
                          candidates=drafted, suites=suites, exemplars=ex_ids, notes=notes, agreed=agreed,
                          disputed=disputed, sandbox=level, extra_files=out_extra, spec_warnings=spec_warnings,
                          context_picks=context_picks)
        self.emit("build.end", {"chosen": pick.idx, "group": len(group), "passed": res.passed,
                                "credible": res.credible, "confidence": res.confidence, "sandbox": level})
        return res

    def edit(self, spec: str, target: str | Path, tests: Sequence[str | Path] = (), scope: str = "*",
             context: str = "", extra_files: dict[str, str] | None = None, strict: bool = False,
             budget_s: float | None = None, atlas=None, repo: str | None = None,
             context_tokens: int = 1500) -> BuildResult:
        """Edit an existing file from a spec: sample unified-diff candidates, run existing + new spec tests
        in the sandbox, and return a unified diff for review (never applied to disk)."""
        spec = textwrap.dedent(spec).strip()
        target_path = Path(target).expanduser()
        if not target_path.is_file():
            raise FileNotFoundError(f"target file not found: {target_path}")
        original = target_path.read_text(encoding="utf-8")
        why_secret = secret_reason(str(target_path), original)
        if why_secret:
            raise ValueError(f"{target_path} looks like it holds a secret ({why_secret}); refusing to edit it")
        target_display = str(target)
        module_name = target_path.stem
        spec_warnings = lint_spec(spec)
        auto_ctx, auto_extra, context_picks = (
            atlas_search_context(
                atlas, spec, repo=repo,
                exclude_paths=[target_display, target_path.name, *list((extra_files or {}).keys())],
                max_tokens=context_tokens,
            )
            if atlas is not None
            else ("", {}, [])
        )
        ctx_parts = [p for p in (context.strip() if context else "", auto_ctx) if p]
        full_spec = f"{spec}\n\n" + "\n\n".join(ctx_parts) if ctx_parts else spec
        merged_extra = {**auto_extra, **(extra_files or {})}
        companion: dict[str, str] = {}
        notes: list[str] = []
        for fname, body in merged_extra.items():
            problems = sandbox.lint(body, self.allow_io)
            if problems:
                notes.append(f"extra file {fname} dropped: {'; '.join(problems[:3])}")
            else:
                companion[fname] = body
        level = sandbox.level()
        if strict and spec_warnings:
            notes.append("stopped early (--strict): " + "; ".join(spec_warnings))
            self.emit("build.end", {"chosen": None, "candidates": 0, "strict_warnings": len(spec_warnings)})
            return BuildResult(spec=spec, code="", tests="", chosen=None, group_size=0, passed=0, credible=0,
                               candidates=[], suites=[], exemplars=[], notes=notes, sandbox=level,
                               extra_files=companion, edit_target=target_display, spec_warnings=spec_warnings,
                               context_picks=context_picks)

        existing_paths = find_existing_tests(target_path, tests)
        existing_suites: list[tuple[str, str]] = []
        for idx, tp in enumerate(existing_paths):
            existing_suites.append((f"existing_{idx}", tp.read_text(encoding="utf-8")))
        if not existing_suites:
            notes.append("no existing test file found beside target; running new spec tests only")

        deadline = (time.monotonic() + max(0.0, budget_s)) if budget_s is not None else None
        shots, ex_ids = self._exemplars(spec, scope)
        self.emit("build.start", {"spec_chars": len(spec), "exemplars": ex_ids, "sandbox": level,
                                  "edit_target": target_display, "existing_suites": len(existing_suites)})

        drafted_suites, suite_cut = self._run_items(
            lambda i: self._edit_suite(full_spec, module_name, original, i), range(self.n_suites), deadline
        )
        if suite_cut:
            notes.append(f"budget reached ({budget_s:g}s): stopped suite sampling after {len(drafted_suites)}/{self.n_suites}")
        suites = [code for code, _ in drafted_suites if code]
        notes += [f"suite {i + 1} dropped: {why}" for i, (code, why) in enumerate(drafted_suites) if not code]

        drafted, cand_cut = self._run_items(
            lambda i: self._edit_candidate(full_spec, target_path.name, module_name, original, shots, i, companion),
            range(self.n_impl),
            deadline,
        )
        if cand_cut:
            notes.append(f"budget reached ({budget_s:g}s): stopped candidate sampling after {len(drafted)}/{self.n_impl}")
        live = [c for c in drafted if c.ok]
        if not live:
            notes.append("no diff candidate applied cleanly, passed static lint, and imported")
            self.emit("build.end", {"chosen": None, "candidates": len(drafted), "edit_target": target_display})
            return BuildResult(spec=spec, code="", tests="", chosen=None, group_size=0, passed=0, credible=0,
                               candidates=drafted, suites=suites, exemplars=ex_ids, notes=notes, sandbox=level,
                               extra_files=companion, edit_target=target_display, spec_warnings=spec_warnings,
                               context_picks=context_picks)

        suite_map = {name: src for name, src in existing_suites}
        for k, src in enumerate(suites):
            suite_map[suite_name(k)] = src
        self._parallel(
            lambda c: self._score_edit(c, existing_suites, suites, module_name=module_name, companion=companion),
            live,
        )

        def _existing_failures(c: Candidate) -> dict[str, str]:
            return {t: out for t, out in c.failed.items() if t.startswith("existing_")}

        def _existing_passes(c: Candidate) -> set[str]:
            return {t for t in c.passed if t.startswith("existing_")}

        def _spec_credible(cands: list[Candidate]) -> set[str]:
            return {t for t in self._credible(cands) if not t.startswith("existing_")}

        all_existing_ids: set[str] = set()
        for c in live:
            all_existing_ids |= _existing_passes(c) | set(_existing_failures(c))
        existing_total = len(all_existing_ids)

        green_existing = [c for c in live if not _existing_failures(c)]
        if green_existing:
            pick, group = self._agree(green_existing, full_spec)
        else:
            pick = min(live, key=lambda c: (len(_existing_failures(c)), -len(c.passed), len(c.code), c.idx))
            group = [c for c in live if c.passed == pick.passed]

        credible = _spec_credible(live)
        regressed: dict[str, str] = {}
        for rnd in range(self.repair_rounds):
            if self._over_budget(deadline):
                notes.append(f"budget reached ({budget_s:g}s): skipped repair round {rnd + 1}")
                break
            support = self._test_support(live)
            missing_ex = _existing_failures(pick)
            missing_sp = {t: pick.failed[t] for t in sorted(credible, key=lambda t: (-support.get(t, 0), t))
                          if t in pick.failed}
            missing = {**missing_ex, **missing_sp}
            if not missing:
                break
            fixed = self._repair_edit(full_spec, target_path.name, module_name, original, pick, missing, rnd,
                                      suites=suites, suite_map=suite_map, regressed=regressed, live=live,
                                      companion=companion)
            if fixed is None:
                notes.append(f"repair round {rnd + 1} produced no usable diff")
                continue
            self._score_edit(fixed, existing_suites, suites, module_name=module_name, companion=companion)
            drafted.append(fixed)
            all_existing_ids |= _existing_passes(fixed) | set(_existing_failures(fixed))
            existing_total = len(all_existing_ids)
            improved_existing = len(_existing_failures(fixed)) < len(_existing_failures(pick))
            same_or_better_existing = len(_existing_failures(fixed)) <= len(_existing_failures(pick))
            if (improved_existing and len(fixed.passed) >= len(pick.passed)) or (
                same_or_better_existing and fixed.passed > pick.passed
            ):
                notes.append(f"repair round {rnd + 1}: {len(pick.passed)} -> {len(fixed.passed)} tests")
                live.append(fixed)
                group = [c for c in live if c.passed == fixed.passed]
                pick = fixed
                credible = _spec_credible(live)
                regressed = {}
            else:
                regressed = {t: fixed.failed.get(t, "regressed") for t in sorted(pick.passed - fixed.passed)}
                fixed.dropped = "did not improve existing + spec test results"
                notes.append(f"repair round {rnd + 1} did not improve test results; kept the original")

        existing_failed = _existing_failures(pick)
        existing_passed = len(_existing_passes(pick))
        spec_disputed = {t: pick.failed[t] for t in sorted(credible) if t in pick.failed}
        disputed = {**existing_failed, **spec_disputed}
        agreed = [s for s in (prune(suite, suite_name(k), pick.passed) for k, suite in enumerate(suites)) if s]

        if existing_failed or spec_disputed or (existing_suites and existing_passed == 0):
            if existing_failed:
                notes.append(f"edit rejected: {len(existing_failed)} existing test(s) still failing")
            if spec_disputed:
                notes.append(f"edit rejected: {len(spec_disputed)} credible spec test(s) still failing")
            res = BuildResult(spec=spec, code="", tests=merge_suites(agreed), chosen=None, group_size=0,
                              passed=len(pick.passed & credible), credible=len(credible), candidates=drafted,
                              suites=suites, exemplars=ex_ids, notes=notes, agreed=agreed, disputed=disputed,
                              sandbox=level, extra_files=companion, diff="", edit_target=target_display,
                              existing_passed=existing_passed, existing_total=existing_total,
                              spec_warnings=spec_warnings, context_picks=context_picks)
            self.emit("build.end", {"chosen": None, "edit_target": target_display,
                                    "existing_passed": existing_passed, "existing_total": existing_total})
            return res

        diff_text = unified_diff(original, pick.code, target_path.name)
        res = BuildResult(spec=spec, code=pick.code, tests=merge_suites(agreed), chosen=pick.idx,
                          group_size=len(group), passed=len(pick.passed & credible), credible=len(credible),
                          candidates=drafted, suites=suites, exemplars=ex_ids, notes=notes, agreed=agreed,
                          disputed={}, sandbox=level, extra_files=companion, diff=diff_text,
                          edit_target=target_display, existing_passed=existing_passed,
                          existing_total=existing_total, spec_warnings=spec_warnings,
                          context_picks=context_picks)
        self.emit("build.end", {"chosen": pick.idx, "group": len(group), "passed": res.passed,
                                "credible": res.credible, "confidence": res.confidence, "sandbox": level,
                                "edit_target": target_display, "existing_passed": existing_passed,
                                "existing_total": existing_total})
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
        return select_exemplars_within_budget(spec, rec.fragments, limit=2, max_tokens=self.exemplar_tokens)

    def _candidate(self, spec: str, shots: str, i: int, modules: Sequence[str] = (),
                   companion: dict[str, str] | None = None) -> Candidate:
        t = TEMPERATURES[i % len(TEMPERATURES)]
        style = "edge-first" if i % 2 else "direct"
        if modules:
            prompt = multifile_candidate_prompt(spec, modules, shots, i, EDGE_FIRST)
            try:
                reply = self.llm(MULTIFILE_IMPL_SYSTEM, prompt, 3072, t)
            except Exception as exc:  # noqa: BLE001
                return Candidate(i, "", t, style, dropped=f"the model call failed: {one_line(str(exc), 120)}")
            sol, extra = parse_multifile_reply(reply)
            files = dict(extra)
            if sol and "solution.py" in modules and "solution.py" not in files:
                files["solution.py"] = sol
            primary = modules[0]
            return self._gate_multifile(Candidate(i, files.get(primary, ""), t, style, files=files),
                                        modules, companion=companion)
        prompt = candidate_prompt(spec, shots, i, EDGE_FIRST)
        try:
            code = python_code(self.llm(IMPL_SYSTEM, prompt, 2048, t))
        except Exception as exc:  # noqa: BLE001 - one failed sample must not sink the build
            return Candidate(i, "", t, style, dropped=f"the model call failed: {one_line(str(exc), 120)}")
        return self._gate(Candidate(i, code, t, style), companion=companion)

    def _edit_candidate(self, spec: str, filename: str, module_name: str, original: str, shots: str, i: int,
                        companion: dict[str, str] | None = None) -> Candidate:
        t = TEMPERATURES[i % len(TEMPERATURES)]
        style = "edge-first" if i % 2 else "direct"
        prompt = edit_candidate_prompt(spec, filename, original, shots, i, EDGE_FIRST)
        try:
            reply = self.llm(EDIT_SYSTEM, prompt, 2048, t)
        except Exception as exc:  # noqa: BLE001
            return Candidate(i, "", t, style, dropped=f"the model call failed: {one_line(str(exc), 120)}")
        diff_text = extract_diff(reply)
        if not diff_text:
            return Candidate(i, "", t, style, dropped="no unified diff in the reply")
        patched, err = apply_unified_diff(original, diff_text)
        if patched is None:
            return Candidate(i, "", t, style, dropped=f"diff did not apply: {err}")
        if patched == original:
            return Candidate(i, "", t, style, dropped="diff makes no changes")
        adds, dels = _diff_counts(original, patched, filename)
        return self._gate(Candidate(i, patched, t, style, diff_adds=adds, diff_dels=dels),
                          module_name=module_name, companion=companion)

    def _gate(self, c: Candidate, module_name: str = "solution",
              companion: dict[str, str] | None = None) -> Candidate:
        """Mark a candidate dropped unless it parses, passes the static check and imports in the sandbox."""
        if not c.code:
            c.dropped = "no Python module in the reply"
            return c
        problems = sandbox.lint(c.code, self.allow_io)
        if problems:
            c.dropped = "; ".join(problems[:3])
            return c
        if module_name == "solution" and not companion:
            ok, err = sandbox.imports_ok(c.code, self.timeout)
        else:
            ok, err = sandbox.imports_ok(c.code, self.timeout, module_name=module_name, extra_files=companion)
        if not ok:
            c.dropped = "does not import: " + one_line(err.strip().splitlines()[-1] if err.strip() else "no output", 160)
        return c

    def _gate_multifile(self, c: Candidate, modules: Sequence[str],
                        companion: dict[str, str] | None = None) -> Candidate:
        """Gate a multi-file candidate: all required modules present, clean lint, and all import together."""
        missing = [m for m in modules if not c.files.get(m, "").strip()]
        if missing:
            c.dropped = "missing module(s): " + ", ".join(missing)
            return c
        for fname in modules:
            problems = sandbox.lint(c.files[fname], self.allow_io)
            if problems:
                c.dropped = f"{fname}: " + "; ".join(problems[:3])
                return c
        merged = {**(companion or {}), **{m: c.files[m] for m in modules}}
        ok, err = sandbox.imports_all_ok(merged, self.timeout)
        if not ok:
            c.dropped = "does not import: " + one_line(err.strip().splitlines()[-1] if err.strip() else "no output", 160)
        return c

    def _suite(self, spec: str, i: int, modules: Sequence[str] = ()) -> tuple[str, str]:
        """(suite source, "") or ("", why it was dropped)."""
        prompt = multifile_suite_prompt(spec, modules, i) if modules else suite_prompt(spec, i)
        try:
            code = python_code(self.llm(TEST_SYSTEM, prompt, 2048, SUITE_TEMPERATURES[i % len(SUITE_TEMPERATURES)]))
        except Exception as exc:  # noqa: BLE001
            return "", f"the model call failed: {one_line(str(exc), 120)}"
        if not code:
            return "", "no Python module in the reply"
        expected_stems = [Path(m).stem for m in modules] if modules else ["solution"]
        if not any(stem in code for stem in expected_stems):
            return "", "it does not import the solution module"
        problems = sandbox.lint(code, allow_io=True)     # tests may use temp files; never the network or processes
        if problems:
            return "", "; ".join(problems[:3])
        code = normalize_suite(code)
        if not has_tests(code):
            return "", "it defines no test methods"
        return code, ""

    def _edit_suite(self, spec: str, module_name: str, original: str, i: int) -> tuple[str, str]:
        """Generate one spec test suite for edit mode targeting `module_name`."""
        prompt = edit_suite_prompt(spec, module_name, original, i)
        try:
            code = python_code(self.llm(TEST_SYSTEM, prompt, 2048, SUITE_TEMPERATURES[i % len(SUITE_TEMPERATURES)]))
        except Exception as exc:  # noqa: BLE001
            return "", f"the model call failed: {one_line(str(exc), 120)}"
        if not code:
            return "", "no Python module in the reply"
        if module_name not in code and "solution" not in code:
            return "", f"it does not import {module_name}"
        if module_name != "solution" and "solution" in code and module_name not in code:
            code = retarget(code, module_name)
        problems = sandbox.lint(code, allow_io=True)
        if problems:
            return "", "; ".join(problems[:3])
        code = normalize_suite(code)
        if not has_tests(code):
            return "", "it defines no test methods"
        return code, ""

    def _repair(self, spec: str, pick: Candidate, missing: dict[str, str], rnd: int,
                suites: list[str] | tuple[str, ...] = (), regressed: dict[str, str] | None = None,
                live: Sequence[Candidate] = (), modules: Sequence[str] = (),
                companion: dict[str, str] | None = None) -> Candidate | None:
        fails = (format_failures(suites, missing, limit=6) if missing
                 else "(all credible generated tests passed; fix the CONTRACT MISMATCH below)")
        if modules:
            cur_blocks = "\n\n".join(
                f"```python {m}\n{pick.files.get(m, '').rstrip()}\n```" for m in modules
            )
            prompt = f"SPEC:\n{spec}\n\nCURRENT MODULES:\n{cur_blocks}\n\nFAILING TESTS:\n{fails}"
        else:
            prompt = f"SPEC:\n{spec}\n\nCURRENT solution.py:\n```python\n{pick.code}```\n\nFAILING TESTS:\n{fails}"
        donor = donor_snippet(live, pick, missing)
        if donor:
            prompt += f"\n\n{donor}"
        if regressed:
            prompt += f"\n\nREGRESSED IN PREVIOUS REPAIR (keep these passing):\n{format_failures(suites, regressed, limit=4)}"
        if not modules:
            mismatch = missing_exports(spec, pick.code)
            if mismatch:
                prompt += "\n\nCONTRACT MISMATCH IN CURRENT solution.py:\n" + "\n".join(f"- {m}" for m in mismatch)
        checklist = spec_checklist(spec)
        if checklist:
            prompt += f"\n\n{checklist}"
        temperature = 0.2 + 0.2 * rnd
        if modules:
            try:
                reply = self.llm(MULTIFILE_IMPL_SYSTEM, prompt, 3072, temperature)
            except Exception:  # noqa: BLE001
                return None
            sol, extra = parse_multifile_reply(reply)
            merged_files = dict(pick.files)
            merged_files.update(extra)
            if sol and "solution.py" in modules:
                merged_files["solution.py"] = sol
            primary = modules[0]
            fixed = self._gate_multifile(
                Candidate(100 + rnd, merged_files.get(primary, ""), temperature, "repair", repaired=True,
                          files=merged_files),
                modules,
                companion=companion,
            )
            return fixed if fixed.ok else None
        try:
            code = python_code(self.llm(REPAIR_SYSTEM, prompt, 2048, temperature))
        except Exception:  # noqa: BLE001
            return None
        fixed = self._gate(Candidate(100 + rnd, code, temperature, "repair", repaired=True), companion=companion)
        return fixed if fixed.ok else None

    def _repair_edit(self, spec: str, filename: str, module_name: str, original: str, pick: Candidate,
                     missing: dict[str, str], rnd: int, suites: Sequence[str] = (),
                     suite_map: dict[str, str] | None = None, regressed: dict[str, str] | None = None,
                     live: Sequence[Candidate] = (), companion: dict[str, str] | None = None) -> Candidate | None:
        fails = format_failures(suites, missing, limit=6, suite_map=suite_map)
        prompt = (
            f"SPEC (change to `{filename}`):\n{spec}\n\n"
            f"ORIGINAL `{filename}`:\n```python\n{original.rstrip()}\n```\n\n"
            f"CURRENT PATCHED `{filename}`:\n```python\n{pick.code.rstrip()}\n```\n\n"
            f"FAILING TESTS:\n{fails}"
        )
        donor = donor_snippet(live, pick, missing)
        if donor:
            prompt += f"\n\n{donor}"
        if regressed:
            prompt += (
                f"\n\nREGRESSED IN PREVIOUS REPAIR (keep these passing):\n"
                f"{format_failures(suites, regressed, limit=4, suite_map=suite_map)}"
            )
        prompt += f"\n\nReply with a complete unified diff against ORIGINAL `{filename}` inside one ```diff block."
        temperature = 0.2 + 0.2 * rnd
        try:
            reply = self.llm(EDIT_REPAIR_SYSTEM, prompt, 2048, temperature)
        except Exception:  # noqa: BLE001
            return None
        diff_text = extract_diff(reply)
        if not diff_text:
            return None
        patched, _ = apply_unified_diff(original, diff_text)
        if patched is None or patched == original:
            return None
        adds, dels = _diff_counts(original, patched, filename)
        fixed = self._gate(Candidate(100 + rnd, patched, temperature, "repair", repaired=True,
                                     diff_adds=adds, diff_dels=dels),
                           module_name=module_name, companion=companion)
        return fixed if fixed.ok else None

    # ------------------------------------------------------------------ evaluation

    def _score(self, c: Candidate, suites: list[str], module_name: str = "solution",
               companion: dict[str, str] | None = None) -> Candidate:
        passed: set[str] = set()
        failed: dict[str, str] = {}
        extra = dict(companion or {})
        if c.files:
            primary_file = f"{module_name}.py"
            for fname, body in c.files.items():
                if fname != primary_file:
                    extra[fname] = body
        for k, suite in enumerate(suites):
            if module_name == "solution" and not extra:
                p, f = run_suite(c.code, suite, suite_name(k), self.timeout)
            else:
                p, f = run_suite(c.code, suite, suite_name(k), self.timeout,
                                 module_name=module_name, extra_files=extra)
            passed |= p
            failed.update(f)
        c.passed, c.failed = frozenset(passed), failed
        return c

    def _score_edit(self, c: Candidate, existing_suites: Sequence[tuple[str, str]], spec_suites: Sequence[str],
                    module_name: str, companion: dict[str, str] | None = None) -> Candidate:
        passed: set[str] = set()
        failed: dict[str, str] = {}
        extra = companion or None
        for ename, esrc in existing_suites:
            p, f = run_suite(c.code, esrc, ename, self.timeout, module_name=module_name, extra_files=extra)
            passed |= p
            failed.update(f)
        for k, suite in enumerate(spec_suites):
            p, f = run_suite(c.code, suite, suite_name(k), self.timeout, module_name=module_name, extra_files=extra)
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

    @staticmethod
    def _over_budget(deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def _run_items(self, fn, items, deadline: float | None):
        items = list(items)
        if not items:
            return [], False
        if deadline is None:
            return self._parallel(fn, items), False
        results = []
        stopped = False
        for item in items:
            if self._over_budget(deadline):
                stopped = True
                break
            results.append(fn(item))
        return results, stopped

    def _parallel(self, fn, items):
        items = list(items)
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(items)))) as pool:
            return list(pool.map(fn, items))


def _diff_counts(original: str, updated: str, filename: str) -> tuple[int, int]:
    diff_text = unified_diff(original, updated, filename)
    adds = sum(1 for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    dels = sum(1 for ln in diff_text.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    return adds, dels


def edit_build(
    spec_text: str,
    path: str | Path,
    tests: str | Path | Sequence[str | Path] | None = None,
    *,
    builder: Builder | None = None,
    llm: LLM | None = None,
    soup: Soup | None = None,
    atlas=None,
    repo: str | None = None,
    scope: str = "*",
    context: str = "",
    extra_files: dict[str, str] | None = None,
    strict: bool = False,
    budget_s: float | None = None,
    context_tokens: int = 1500,
    **builder_kwargs,
) -> BuildResult:
    """Stable chat-facing entrypoint for edit-mode builds.

    Pass an existing `builder=` (e.g. `mind.builder()`) or an `llm=` callable. Runs existing tests
    (auto-discovered beside `path` or listed in `tests`) plus new spec tests in the sandbox and
    returns a `BuildResult` with `.code` (full edited file), `.diff` (unified diff for review,
    never applied to disk), and `.summary()` (<=6 lines).
    """
    if builder is None:
        if llm is None:
            raise ValueError("edit_build requires either builder= or llm=")
        builder = Builder(llm, soup=soup, **builder_kwargs)
    if tests is None:
        test_list: Sequence[str | Path] = ()
    elif isinstance(tests, (str, Path)):
        test_list = (tests,)
    else:
        test_list = tuple(tests)
    return builder.edit(
        spec_text,
        path,
        tests=test_list,
        scope=scope,
        context=context,
        extra_files=extra_files,
        strict=strict,
        budget_s=budget_s,
        atlas=atlas,
        repo=repo,
        context_tokens=context_tokens,
    )


def spec_from_text(text: str) -> str:
    return strip_reasoning(text).strip()

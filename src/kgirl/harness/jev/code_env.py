"""Code world for Jev: a sandboxed workspace (copy of a repo) as an element table.

    SEARCH :: words   rank files/symbols in the workspace; hits become the table
    OPEN i            focus a file or symbol; its outline joins the table, source is shown
    EDIT i            ask the *smith* model for SEARCH/REPLACE blocks on that region;
                      a block applies only if its SEARCH text occurs exactly once
    BLAST i           who depends on this file (Atlas graph, cross-repo)
    RUN               run the verifier command fixed by the caller (never model-chosen)
                      (it also runs by itself: once at the start to reproduce the failure,
                      and after each applied edit, so a fix ends the run even if unclaimed)
    DONE :: summary   claim success (the swarm re-verifies independently)
    BLOCKED :: why    give up

Only EDIT invokes text generation, mirroring Jev Ultrafast where only
TYPE_TEXT does. Every other step is a single cheap decision.
"""

from __future__ import annotations

import difflib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..atlas.parse import detect_language, extract
from ..atlas.store import SKIP_DIRS, MAX_BYTES
from ..util import clip, search_terms, split_identifier
from .env import ActionResult, Element, Observation
from .ops import Decision, OpSpec

_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)  # kept out of verifier envs

CODE_OPS: dict[str, OpSpec] = {s.name: s for s in (
    OpSpec("SEARCH", needs_arg=True, help="find files/symbols by words"),
    OpSpec("OPEN", needs_target=True, help="show source + outline of element i"),
    OpSpec("EDIT", needs_target=True, help="change code in element i toward the goal"),
    OpSpec("BLAST", needs_target=True, help="list dependents of element i's file"),
    OpSpec("RUN", help="run the verification command"),
    OpSpec("DONE", needs_arg=True, terminal=True, help="goal achieved; one-line summary"),
    OpSpec("BLOCKED", needs_arg=True, terminal=True, help="cannot proceed; reason"),
)}

# smith(goal, path, numbered_region, memory_text) -> text containing SEARCH/REPLACE blocks
Smith = Callable[[str, str, str, str], str]
Blast = Callable[[str], str]

_BLOCK = re.compile(r"<<<<<<< ?SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> ?REPLACE", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```[^\n]*\n(.*?)\n?```", re.S)
MIN_REWRITE_SHARED = 2                # non-blank region lines a fenced rewrite must repeat to count as one
MIN_REWRITE_OVERLAP = 0.5             # share of the replaced span's non-blank lines it must keep
_TRACE = re.compile(r'File "([^"]+)", line (\d+)')
_TEST_PATH = re.compile(r"(^|/)(tests?/|test_[^/]*$|[^/]*_test\.[^/]*$)")
BRIEFING_CHARS = 1200                 # tail of the verifier's output kept for the agent and the smith
MAX_SITES = 4                         # traceback locations promoted to the top of the table
MAX_SITE_SPAN = 40                    # lines of a large enclosing symbol kept around a traceback location


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _loose_replace(text: str, search: str, replace: str) -> str | None:
    """Apply a block whose SEARCH differs from the file only in indentation, if those lines occur exactly once."""
    wanted = search.strip("\n").split("\n")
    stripped = [line.strip() for line in wanted]
    lines = text.split("\n")
    hits = [i for i in range(len(lines) - len(stripped) + 1)
            if [line.strip() for line in lines[i:i + len(stripped)]] == stripped]
    if len(hits) != 1 or not any(stripped):
        return None
    start = hits[0]
    shift = _indent(lines[start]) - _indent(wanted[0])
    moved = [(" " * shift + line if shift > 0 else line[min(-shift, _indent(line)):]) if line.strip() else line
             for line in replace.split("\n")]
    return "\n".join(lines[:start] + moved + lines[start + len(stripped):])


def _rewrite_block(new_lines: list[str], lines: list[str]) -> list[tuple[str, str]]:
    """A fence that rewrites the region replaces the span it shares with it; lines it omits outside that span stay."""
    matcher = difflib.SequenceMatcher(None, lines, new_lines, autojunk=False)
    shared = [i for m in matcher.get_matching_blocks() for i in range(m.a, m.a + m.size) if lines[i].strip()]
    if len(shared) < MIN_REWRITE_SHARED:
        return []
    span = lines[shared[0]:shared[-1] + 1]
    if len(shared) < MIN_REWRITE_OVERLAP * sum(1 for line in span if line.strip()):
        return []
    return [("\n".join(span), "\n".join(new_lines))]


def recover_blocks(reply: str, region: str) -> list[tuple[str, str]]:
    """Read an edit from a reply small models commonly get nearly right.

    Used only when no well-formed block exists. Everything it returns still has to
    match the file exactly once and pass the verifier, so a wrong guess costs one step.
    """
    text = _THINK.sub("", reply or "").strip()
    if "\n=======\n" in text or text.startswith("=======\n"):
        before, after = ("\n" + text).split("\n=======\n", 1)
        before = re.sub(r"^\s*<{3,}[^\n]*\n?", "", before.strip("\n"))
        after = re.sub(r"\n?>{3,}[^\n]*\s*$", "", after)
        fenced = _FENCE.findall(after)
        return [(before, fenced[-1] if fenced else after)] if before.strip() else []
    fences = _FENCE.findall(text)
    if not fences:
        return []
    # A fenced definition replaces the same definition in the region, found by its first line.
    new_lines = fences[-1].split("\n")
    lines = region.split("\n")
    if lines.count(new_lines[0]) != 1 or not new_lines[0].lstrip().startswith(("def ", "async def ", "class ")):
        return _rewrite_block(new_lines, lines)
    start = lines.index(new_lines[0])
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or _indent(lines[end]) > _indent(lines[start])):
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return [("\n".join(lines[start:end]), fences[-1])]
MAX_TABLE = 18


@dataclass(frozen=True)
class _Sym:
    path: str
    kind: str
    qualname: str
    signature: str
    doc: str
    line: int
    end_line: int
    terms: frozenset[str]


class CodeEnvironment:
    kind = "code"
    ops = CODE_OPS

    def __init__(self, workspace: str | Path, goal: str, smith: Smith | None = None,
                 verify_cmd: list[str] | None = None, blast: Blast | None = None, memory_text: str = "",
                 seed_paths: list[str] | None = None, verify_timeout: float = 180.0):
        self.root = Path(workspace).resolve()
        self.goal = goal
        self.smith = smith
        self.verify_cmd = verify_cmd
        self.blast = blast
        self.memory_text = memory_text
        self.verify_timeout = verify_timeout
        self._index: list[_Sym] | None = None
        self._files: dict[str, int] = {}
        self._originals: dict[str, str] = {}
        self.focus = ""
        self.elements: list[Element] = []
        self.note = ""
        self.last_verify: tuple[int, str] | None = None
        self.briefing = ""        # the verifier's current failure output, shown to the agent and the smith
        self._last_failure = ""   # why the previous EDIT was refused
        self._seed(seed_paths or [], self._reproduce())

    # ------------------------------------------------------------------ index

    def _build_index(self) -> list[_Sym]:
        if self._index is not None:
            return self._index
        out: list[_Sym] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for fn in sorted(filenames):
                full = Path(dirpath) / fn
                rel = full.relative_to(self.root).as_posix()
                try:
                    if full.stat().st_size > MAX_BYTES:
                        continue
                    text = full.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                lang = detect_language(rel, text[:200])
                if not lang or lang == "markdown":
                    continue
                facts = extract(rel, text, lang)
                self._files[rel] = facts.loc
                out.append(_Sym(rel, "file", rel, f"{lang}, {facts.loc} lines", facts.doc, 1, facts.loc,
                                frozenset(search_terms(rel, facts.doc).split())))
                for s in facts.symbols:
                    out.append(_Sym(rel, s.kind, s.qualname, s.signature, s.doc, s.line, s.end_line,
                                    frozenset(search_terms(s.qualname, s.signature, s.doc, rel).split())))
        self._index = out
        return out

    def _rank(self, query: str, limit: int = MAX_TABLE) -> list[_Sym]:
        words = {w for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
                 for w in [t.lower(), *split_identifier(t)] if len(w) > 1}
        if not words:
            return []
        scored = []
        for s in self._build_index():
            hit = len(words & s.terms)
            if hit:
                bonus = 0.5 if s.kind in ("function", "method", "class") else 0.0
                penalty = 1.0 if self._protected(s.path) else 0.0
                scored.append((hit + bonus - penalty - 0.001 * len(s.path), s))
        scored.sort(key=lambda x: (-x[0], x[1].path, x[1].line))
        return [s for _, s in scored[:limit]]

    def _element(self, s: _Sym) -> Element:
        if s.kind == "file":
            return Element("file", f"{s.path}  ({s.signature})" + (f" — {clip(s.doc, 70)}" if s.doc else ""),
                           {"path": s.path, "line": 1, "end": s.end_line})
        span = f"{s.path}:{s.line}" + (f"-{s.end_line}" if s.end_line > s.line else "")
        return Element("symbol", f"{span}  {clip(s.signature, 90)}" + (f" — {clip(s.doc, 60)}" if s.doc else ""),
                       {"path": s.path, "line": s.line, "end": s.end_line})

    def _protected(self, path: str) -> bool:
        """With a verifier fixed by the caller, tests are the judge: they can be read but not edited."""
        return bool(self.verify_cmd) and _TEST_PATH.search(path) is not None

    def _reproduce(self) -> list[tuple[str, int]]:
        """Run the verifier before anything else; return where its traceback points, innermost first, source first."""
        if not self.verify_cmd:
            return []
        code, out = self.verify()
        if code == 0:
            return []
        self.briefing = out[-BRIEFING_CHARS:]
        sites: list[tuple[str, int]] = []
        for path, line in reversed(_TRACE.findall(out)):
            full = Path(path) if Path(path).is_absolute() else self.root / path
            full = full.resolve()
            if self.root in full.parents and full.is_file():
                site = (full.relative_to(self.root).as_posix(), int(line))
                if site not in sites:
                    sites.append(site)
        return sorted(sites, key=lambda s: self._protected(s[0]))[:MAX_SITES]

    def _site_element(self, path: str, line: int) -> Element:
        enclosing = [s for s in self._build_index() if s.path == path and s.kind != "file" and s.line <= line <= s.end_line]
        best = min(enclosing, key=lambda s: s.end_line - s.line, default=None)
        lo, hi = (best.line, best.end_line) if best else (line, line)
        if hi - lo > MAX_SITE_SPAN:
            lo, hi = max(lo, line - MAX_SITE_SPAN // 2), min(hi, line + MAX_SITE_SPAN // 2)
        source = (self.root / path).read_text(encoding="utf-8", errors="replace").split("\n")[line - 1].strip()
        return Element("symbol", f"{path}:{line}  in the failure: {clip(source, 90)}", {"path": path, "line": lo, "end": hi})

    def _seed(self, seed_paths: list[str], sites: list[tuple[str, int]] | None = None) -> None:
        els: list[Element] = [self._site_element(path, line) for path, line in sites or []]
        for p in seed_paths:
            if (self.root / p).is_file():
                els.append(Element("file", f"{p}  (from memory)", {"path": p, "line": 1, "end": 10**9}))
        for s in self._rank(self.goal, MAX_TABLE - len(els)):
            e = self._element(s)
            if all(e.ref != x.ref for x in els):
                els.append(e)
        self.elements = els[:MAX_TABLE]
        self.note = "start: table seeded from " + ("the failure and " if sites else "") + "the goal" + \
            (" and memory" if seed_paths else "")

    # ------------------------------------------------------------------ protocol

    def observe(self) -> Observation:
        header = f"workspace: {self.root.name}" + (f" | focus: {self.focus}" if self.focus else "")
        if self._originals:
            header += f" | edited: {', '.join(sorted(self._originals))}"
        return Observation(header, tuple(self.elements), self.note)

    def act(self, d: Decision) -> ActionResult:
        handler = getattr(self, f"_op_{d.op.lower()}")
        res: ActionResult = handler(d)
        self.note = res.note
        return res

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ ops

    def _op_search(self, d: Decision) -> ActionResult:
        hits = self._rank(d.arg)
        if not hits:
            return ActionResult(False, f"SEARCH {d.arg!r}: no matches; table unchanged")
        self.elements = [self._element(s) for s in hits]
        return ActionResult(True, f"SEARCH {d.arg!r}: {len(hits)} hits")

    def _op_open(self, d: Decision) -> ActionResult:
        el = self.elements[d.target]
        path, line, end = el.ref["path"], el.ref["line"], el.ref["end"]
        full = self._safe(path)
        if full is None or not full.is_file():
            return ActionResult(False, f"OPEN: {path} no longer exists")
        lines = full.read_text(encoding="utf-8", errors="replace").split("\n")
        lo = max(1, line - 3)
        hi = min(len(lines), max(end, line) + 3, lo + 70)
        src = "\n".join(f"{n:>5} {lines[n - 1]}" for n in range(lo, hi + 1))
        self.focus = path
        outline = [s for s in self._build_index() if s.path == path and s.kind != "file"][:MAX_TABLE - 1]
        keep = Element("file", f"{path}  ({len(lines)} lines)", {"path": path, "line": 1, "end": len(lines)})
        self.elements = [keep] + [self._element(s) for s in outline]
        return ActionResult(True, f"OPEN {path}:{lo}-{hi}\n{src}")

    def _refuse_edit(self, why: str) -> ActionResult:
        self._last_failure = why
        return ActionResult(False, why)

    def _smith_context(self) -> str:
        parts = [self.memory_text] if self.memory_text else []
        if self.briefing:
            parts.append(f"VERIFIER OUTPUT:\n{self.briefing}")
        if self._last_failure:
            parts.append(f"YOUR LAST ATTEMPT FAILED: {self._last_failure}")
        return "\n".join(parts)

    def _op_edit(self, d: Decision) -> ActionResult:
        if self.smith is None:
            return ActionResult(False, "EDIT unavailable: no code model configured")
        el = self.elements[d.target]
        path, line, end = el.ref["path"], el.ref["line"], el.ref["end"]
        full = self._safe(path)
        if full is None or not full.is_file():
            return self._refuse_edit(f"EDIT: {path} not found")
        if self._protected(path):
            return self._refuse_edit(f"EDIT: {path} is a test and tests judge the fix; edit the source code it exercises")
        text = full.read_text(encoding="utf-8")
        lines = text.split("\n")
        lo, hi = max(1, line - 5), min(len(lines), max(end, line) + 5)
        if hi - lo > 160:
            hi = lo + 160
        region = "\n".join(lines[lo - 1:hi])
        reply = self.smith(self.goal, path, region, self._smith_context())
        blocks = _BLOCK.findall(reply or "") or recover_blocks(reply, region)
        if not blocks:
            return self._refuse_edit("EDIT: model returned no SEARCH/REPLACE block")
        new = text
        applied = 0
        for search, replace in blocks:
            if search.strip() == "":
                return self._refuse_edit("EDIT: empty SEARCH block rejected")
            count = new.count(search)
            loose = _loose_replace(new, search, replace) if count == 0 else None
            if count != 1 and loose is None:
                return self._refuse_edit(f"EDIT: SEARCH text found {count}x in {path} (must be exactly 1); "
                                         "no change applied")
            new = loose if loose is not None else new.replace(search, replace, 1)
            applied += 1
        if new == text:
            return self._refuse_edit("EDIT: blocks produced no change")
        self._originals.setdefault(path, text)
        full.write_text(new, encoding="utf-8")
        self._index = None  # re-parse lazily: line numbers moved
        diff = "".join(difflib.unified_diff(text.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}", n=1))
        self._last_failure = ""
        note = f"EDIT applied {applied} block(s) to {path}\n{clip(diff, 1200)}"
        if not self.verify_cmd:
            return ActionResult(True, note)
        code, out = self.verify()
        if code == 0:
            self.briefing = ""
            return ActionResult(True, f"{note}\nRUN exit=0: the verifier passes", terminal=True, success_claimed=True)
        self.briefing = out[-BRIEFING_CHARS:]
        return ActionResult(True, f"{note}\nRUN exit={code}: the verifier still fails")

    def _op_blast(self, d: Decision) -> ActionResult:
        path = self.elements[d.target].ref["path"]
        if self.blast is None:
            return ActionResult(False, "BLAST unavailable: no atlas attached")
        return ActionResult(True, clip(self.blast(path), 1200))

    def _op_run(self, d: Decision) -> ActionResult:
        if not self.verify_cmd:
            return ActionResult(False, "RUN unavailable: no verification command was provided")
        code, out = self.verify()
        return ActionResult(code == 0, f"RUN exit={code}\n{out[-1500:]}")

    def _op_done(self, d: Decision) -> ActionResult:
        return ActionResult(True, f"DONE: {d.arg}", terminal=True, success_claimed=True)

    def _op_blocked(self, d: Decision) -> ActionResult:
        return ActionResult(False, f"BLOCKED: {d.arg}", terminal=True)

    # ------------------------------------------------------------------ helpers

    def verify(self) -> tuple[int, str]:
        if not self.verify_cmd:
            return 0, "(no verifier)"
        try:
            env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}  # keys stay out of verifiers
            p = subprocess.run(self.verify_cmd, cwd=self.root, capture_output=True, text=True,
                               timeout=self.verify_timeout, env={**env, "PYTHONDONTWRITEBYTECODE": "1"})
            self.last_verify = (p.returncode, (p.stdout + p.stderr)[-4000:])
        except subprocess.TimeoutExpired:
            self.last_verify = (124, f"verifier timed out after {self.verify_timeout}s")
        except OSError as exc:
            self.last_verify = (127, f"verifier failed to start: {exc}")
        return self.last_verify

    def _safe(self, rel: str) -> Path | None:
        full = (self.root / rel).resolve()
        return full if full == self.root or self.root in full.parents else None

    def edited(self) -> dict[str, tuple[str, str]]:
        """path -> (original text, current text) for every file this agent changed."""
        out = {}
        for p, orig in self._originals.items():
            cur = (self.root / p).read_text(encoding="utf-8")
            if cur != orig:
                out[p] = (orig, cur)
        return out

    def diff(self) -> str:
        return "".join("".join(difflib.unified_diff(o.splitlines(True), c.splitlines(True), f"a/{p}", f"b/{p}"))
                       for p, (o, c) in sorted(self.edited().items()))

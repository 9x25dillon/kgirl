"""Code world for Jev: a sandboxed workspace (copy of a repo) as an element table.

    SEARCH :: words   rank files/symbols in the workspace; hits become the table
    OPEN i            focus a file or symbol; its outline joins the table, source is shown
    EDIT i            ask the *smith* model for SEARCH/REPLACE blocks on that region;
                      a block applies only if its SEARCH text occurs exactly once
    BLAST i           who depends on this file (Atlas graph, cross-repo)
    RUN               run the verifier command fixed by the caller (never model-chosen)
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
        self._seed(seed_paths or [])

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
                scored.append((hit + bonus - 0.001 * len(s.path), s))
        scored.sort(key=lambda x: (-x[0], x[1].path, x[1].line))
        return [s for _, s in scored[:limit]]

    def _element(self, s: _Sym) -> Element:
        if s.kind == "file":
            return Element("file", f"{s.path}  ({s.signature})" + (f" — {clip(s.doc, 70)}" if s.doc else ""),
                           {"path": s.path, "line": 1, "end": s.end_line})
        span = f"{s.path}:{s.line}" + (f"-{s.end_line}" if s.end_line > s.line else "")
        return Element("symbol", f"{span}  {clip(s.signature, 90)}" + (f" — {clip(s.doc, 60)}" if s.doc else ""),
                       {"path": s.path, "line": s.line, "end": s.end_line})

    def _seed(self, seed_paths: list[str]) -> None:
        els: list[Element] = []
        for p in seed_paths:
            if (self.root / p).is_file():
                els.append(Element("file", f"{p}  (from memory)", {"path": p, "line": 1, "end": 10**9}))
        for s in self._rank(self.goal, MAX_TABLE - len(els)):
            e = self._element(s)
            if all(e.ref != x.ref for x in els):
                els.append(e)
        self.elements = els[:MAX_TABLE]
        self.note = "start: table seeded from the goal" + (" and memory" if seed_paths else "")

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

    def _op_edit(self, d: Decision) -> ActionResult:
        if self.smith is None:
            return ActionResult(False, "EDIT unavailable: no code model configured")
        el = self.elements[d.target]
        path, line, end = el.ref["path"], el.ref["line"], el.ref["end"]
        full = self._safe(path)
        if full is None or not full.is_file():
            return ActionResult(False, f"EDIT: {path} not found")
        text = full.read_text(encoding="utf-8")
        lines = text.split("\n")
        lo, hi = max(1, line - 5), min(len(lines), max(end, line) + 5)
        if hi - lo > 160:
            hi = lo + 160
        region = "\n".join(lines[lo - 1:hi])
        reply = self.smith(self.goal, path, region, self.memory_text)
        blocks = _BLOCK.findall(reply or "")
        if not blocks:
            return ActionResult(False, "EDIT: model returned no SEARCH/REPLACE block")
        new = text
        applied = 0
        for search, replace in blocks:
            if search.strip() == "":
                return ActionResult(False, "EDIT: empty SEARCH block rejected")
            count = new.count(search)
            if count != 1:
                return ActionResult(False, f"EDIT: SEARCH text found {count}x in {path} (must be exactly 1); "
                                           "no change applied")
            new = new.replace(search, replace, 1)
            applied += 1
        if new == text:
            return ActionResult(False, "EDIT: blocks produced no change")
        self._originals.setdefault(path, text)
        full.write_text(new, encoding="utf-8")
        self._index = None  # re-parse lazily: line numbers moved
        diff = "".join(difflib.unified_diff(text.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}", n=1))
        return ActionResult(True, f"EDIT applied {applied} block(s) to {path}\n{clip(diff, 1200)}")

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

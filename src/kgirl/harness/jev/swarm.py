"""Sacrificial swarms: many cheap Jevs, one accepted trajectory, zero leftovers.

    task ──> recall (Soup, JIT) ──> spawn N Jevs, each with
                                     • its own workspace copy (sandbox)
                                     • a different memory slice (diversity)
                                     • a different temperature
             first verified DONE ──> cancel the rest ("poof")
             winner ──> independent re-verification ──> distill ──> Soup (staged)
                     └─> credit: winner's recalled fragments +1, losers' −½
             curator.step()  (promote / retire / decay)

Nothing a Jev learns survives except what the curator accepts, and nothing
the curator accepts is recalled until it has been reinforced by independent
successes (staged RSI; see soup.curator).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..atlas.store import SKIP_DIRS
from ..soup import Curator, Recall, Soup
from ..util import clip
from .agent import Jev, JevConfig, Step, Trajectory
from .code_env import CodeEnvironment
from .intuition import Intuition

# llm(role, system, prompt, max_tokens, temperature) -> text
LLMCall = Callable[[str, str, str, int, float], str]
Emit = Callable[[str, dict], None]

SMITH_SYSTEM = """You are a precise code editor. Change the code only as much as the GOAL requires.
Answer with one or more blocks in exactly this format and nothing else:
<<<<<<< SEARCH
<exact lines copied from the REGION, including indentation>
=======
<replacement lines>
>>>>>>> REPLACE
The SEARCH text must match the file exactly once. Do not include line numbers."""

DISTILL_SYSTEM = """You compress a successful coding-agent run into reusable lessons.
Write 1 to 3 lines. Each line is one imperative rule of at most 30 words that would help a
future agent solve a SIMILAR task faster in this codebase. Name concrete files, symbols and
commands when they matter. No preamble, no numbering, no speculation."""

_PATH = re.compile(r"(?<![\w/])((?:[\w.-]+/)*[\w.-]+\.(?:py|ts|tsx|js|jl|rs|go|c|cpp|h|sh|gd))\b")
MAX_COPY_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class SwarmConfig:
    size: int = 3
    temperatures: tuple[float, ...] = (0.1, 0.5, 0.9)
    memory_budget: int = 400
    jev: JevConfig = JevConfig()
    stop_on_first_success: bool = True
    keep_workspaces: bool = False
    allow_unverified: bool = False     # accept DONE without a verifier
    intuition_fraction: float = 0.5    # share of agents that may act on intuition (the rest keep exploring)
    intuition_threshold: float = 0.6


@dataclass
class SwarmResult:
    goal: str
    winner: Trajectory | None
    trajectories: list[Trajectory]
    recall: Recall | None
    proposals: list[tuple[int, str]] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"swarm for: {self.goal}"]
        for t in self.trajectories:
            mark = ("★" if t is self.winner else "✓" if t.outcome == "done" and t.verified else
                    "·" if t.outcome == "cancelled" else "✗")
            lines.append(f"  {mark} {t.jev_id} T={t.variant.get('temperature')} {t.outcome:<9} "
                         f"verified={t.verified} steps={len(t.steps)} llm={t.llm_calls} "
                         f"intuition={t.intuition_steps}{'!' if t.surprised else ''} ~{t.prompt_tokens}tok "
                         f"{t.elapsed:.1f}s  {t.ops_signature()}")
        if self.winner:
            lines.append(f"accepted: {self.winner.summary}")
            if self.winner.diff:
                lines.append(clip(self.winner.diff, 3000))
        else:
            lines.append("no trajectory accepted")
        if self.proposals:
            lines.append("soup: " + ", ".join(f"#{i} {s}" for i, s in self.proposals))
        if self.applied:
            lines.append("applied to repo: " + ", ".join(self.applied))
        if self.conflicts:
            lines.append("NOT applied (file changed since copy): " + ", ".join(self.conflicts))
        return "\n".join(lines)


class Swarm:
    def __init__(self, llm: LLMCall, soup: Soup, curator: Curator | None = None,
                 blast: Callable[[str, str], str] | None = None, config: SwarmConfig | None = None,
                 emit: Emit | None = None):
        self.llm = llm
        self.soup = soup
        self.curator = curator or Curator(soup)
        self.blast = blast
        self.config = config or SwarmConfig()
        self.emit = emit or (lambda topic, payload: None)

    # ------------------------------------------------------------------ public

    def run(self, goal: str, repo_path: str | Path, verify_cmd: list[str] | None = None,
            repo_name: str | None = None, apply: bool = False) -> SwarmResult:
        cfg = self.config
        repo = Path(repo_path).resolve()
        repo_name = repo_name or repo.name
        recall = self.soup.recall(goal, budget_tokens=cfg.memory_budget * 2, scope=repo_name,
                                  kinds=("abstraction", "trajectory", "preference", "antipattern", "fact"))
        self.emit("swarm.start", {"goal": goal, "repo": repo_name, "size": cfg.size, "recalled": recall.ids})
        intuition = (Intuition.from_soup(self.soup, scope=repo_name, threshold=cfg.intuition_threshold)
                     if cfg.intuition_fraction > 0 else None)
        n_intuitive = round(cfg.size * cfg.intuition_fraction) if intuition and intuition.routines else 0
        cancel = threading.Event()
        workspaces: list[Path] = []
        results: list[Trajectory] = []
        winner: Trajectory | None = None
        lock = threading.Lock()

        def one(i: int) -> Trajectory:
            ws = self._workspace(repo, i)
            with lock:
                workspaces.append(ws)
            temp = cfg.temperatures[i % len(cfg.temperatures)]
            frags = _slice(recall, i, cfg.size, cfg.memory_budget)
            memory_text = "\n".join(f"- {f.text}" for f in frags)
            seeds = [p for f in frags for p in _PATH.findall(f.text)]
            env = CodeEnvironment(
                ws, goal, smith=self._smith(temp), verify_cmd=verify_cmd, memory_text=memory_text,
                blast=(lambda p: self.blast(repo_name, p)) if self.blast else None, seed_paths=seeds)
            jev = Jev(env, self._decider(temp), memory_text, cfg.jev,
                      on_step=lambda jid, s: self._on_step(jid, s),
                      intuition=intuition if i < n_intuitive else None)
            self.emit("jev.spawn", {"jev": jev.id, "temperature": temp, "memory": [f.id for f in frags]})
            traj = jev.run(goal, cancel)
            traj.recalled = [f.id for f in frags]
            traj.workspace = str(ws)
            traj.variant = {"temperature": temp, "slice": i, "intuition": i < n_intuitive}
            if traj.outcome == "done":
                if verify_cmd:
                    code, out = env.verify()
                    traj.verified, traj.verify_output = code == 0, out
                else:
                    traj.verified = None if cfg.allow_unverified else False
                    traj.verify_output = "no verifier" + ("" if cfg.allow_unverified else "; unverified DONE rejected")
            traj.diff = env.diff()
            traj.edited = env.edited()
            accepted = traj.outcome == "done" and (traj.verified is True or
                                                   (traj.verified is None and cfg.allow_unverified))
            if accepted and traj.diff and cfg.stop_on_first_success:
                cancel.set()
            self.emit("jev.done", {"jev": traj.jev_id, "outcome": traj.outcome, "verified": traj.verified,
                                   "steps": len(traj.steps), "tokens": traj.prompt_tokens,
                                   "llm_calls": traj.llm_calls, "intuition_steps": traj.intuition_steps,
                                   "surprised": traj.surprised})
            return traj

        try:
            with ThreadPoolExecutor(max_workers=cfg.size, thread_name_prefix="jev") as pool:
                for fut in as_completed([pool.submit(one, i) for i in range(cfg.size)]):
                    results.append(fut.result())
            accepted = [t for t in results if t.outcome == "done" and t.diff and
                        (t.verified is True or (t.verified is None and cfg.allow_unverified))]
            winner = min(accepted, key=lambda t: (len(t.steps), t.prompt_tokens), default=None)
            results.sort(key=lambda t: t.variant.get("slice", 0))
            res = SwarmResult(goal, winner, results, recall)
            for t in results:
                if t is not winner and t.outcome != "cancelled":
                    self.emit("jev.poof", {"jev": t.jev_id, "outcome": t.outcome})
            self._learn(res, repo_name, verify_cmd)
            if winner and apply:
                self._apply(res, repo)
            self.emit("swarm.end", {"winner": winner.jev_id if winner else None,
                                    "proposals": res.proposals, "applied": res.applied})
            return res
        finally:
            if not cfg.keep_workspaces:
                for ws in workspaces:
                    shutil.rmtree(ws.parent, ignore_errors=True)

    # ------------------------------------------------------------------ model adapters

    def _decider(self, temperature: float):
        max_tokens = self.config.jev.decision_tokens
        return lambda system, prompt: self.llm("jev", system, prompt, max_tokens, temperature)

    def _smith(self, temperature: float):
        def smith(goal: str, path: str, region: str, memory: str) -> str:
            prompt = (f"GOAL: {goal}\n\n" + (f"MEMORY:\n{memory}\n\n" if memory else "")
                      + f"FILE: {path}\nREGION:\n{region}\n")
            return self.llm("smith", SMITH_SYSTEM, prompt, 1500, min(0.4, temperature))
        return smith

    def _on_step(self, jev_id: str, step: Step) -> None:
        self.emit("jev.step", {"jev": jev_id, "n": step.n, "op": step.op, "target": step.target,
                               "ok": step.ok, "ms": round(step.ms, 1)})

    # ------------------------------------------------------------------ learning

    def _learn(self, res: SwarmResult, repo_name: str, verify_cmd: list[str] | None) -> None:
        w = res.winner
        if w is None:
            ids = sorted({i for t in res.trajectories for i in t.recalled})
            self.soup.credit(ids, success=False, weight=0.5)
            return
        self.soup.credit(w.recalled, success=True)
        losers = {i for t in res.trajectories if t is not w and t.outcome not in ("cancelled",)
                  for i in t.recalled} - set(w.recalled)
        self.soup.credit(sorted(losers), success=False, weight=0.5)
        # intuition: routines that carried the winner earn utility; routines that surprised an agent lose it
        self.soup.credit(sorted(set(w.routines_used)), success=True)
        misled = {r for t in res.trajectories if t.surprised for r in t.routines_used} - set(w.routines_used)
        self.soup.credit(sorted(misled), success=False, weight=0.5)

        files = sorted(w.edited)
        record = {"goal": w.goal, "ops": w.ops_signature(), "files": files, "summary": w.summary,
                  "verify": verify_cmd, "diff": w.diff[:20000], "memory": w.recalled, "steps": w.routine(),
                  "llm_calls": w.llm_calls, "intuition_steps": w.intuition_steps}
        tid = self.soup.add(
            f"Solved '{clip(w.goal, 120)}' in {len(w.steps)} steps by editing {', '.join(files) or '-'}: "
            f"{clip(w.summary, 160)}", kind="trajectory", tags=",".join(files), source=w.jev_id,
            scope=repo_name, data=json.dumps(record))
        res.proposals.append((tid, "trajectory"))
        for rule in self._distill(w, res, files, verify_cmd):
            fid, status = self.curator.propose(rule, kind="abstraction", tags=",".join(files), source=w.jev_id,
                                               scope=repo_name)
            res.proposals.append((fid, status))
            self.emit("soup.propose", {"id": fid, "status": status, "text": rule})
        report = self.curator.step()
        if report:
            self.emit("soup.curate", {"promoted": report.promoted, "retired": report.retired,
                                      "expired": report.expired})

    def _distill(self, w: Trajectory, res: SwarmResult, files: list[str], verify_cmd: list[str] | None) -> list[str]:
        failed = [f"{t.outcome}: {t.ops_signature()}" for t in res.trajectories if t is not w and t.steps]
        prompt = (f"TASK: {w.goal}\nWINNING OPS: {w.ops_signature()}\nFILES CHANGED: {', '.join(files)}\n"
                  f"SUMMARY: {w.summary}\nVERIFIED BY: {' '.join(verify_cmd or []) or 'n/a'}\n"
                  f"DIFF:\n{clip(w.diff, 2500)}\n" + ("FAILED ATTEMPTS:\n" + "\n".join(failed) if failed else ""))
        try:
            text = self.llm("scout", DISTILL_SYSTEM, prompt, 300, 0.2)
            rules = [ln.strip(" -*•\t0123456789.") for ln in text.splitlines()]
            rules = [r for r in rules if 12 <= len(r) <= 300][:3]
            if rules:
                return rules
        except Exception as exc:  # distillation is best-effort; fall back to a deterministic rule
            self.emit("soup.distill_error", {"error": repr(exc)[:200]})
        first_open = next((s.note.split("\n", 1)[0].removeprefix("OPEN ") for s in w.steps
                           if s.op == "OPEN" and s.ok), "")
        rule = f"For tasks like '{clip(w.goal, 80)}': " + \
               (f"start at {first_open}; " if first_open else "") + \
               (f"the fix lives in {', '.join(files)}; " if files else "") + \
               (f"verify with `{' '.join(verify_cmd)}`." if verify_cmd else "")
        return [rule]

    # ------------------------------------------------------------------ sandbox

    def _workspace(self, repo: Path, i: int) -> Path:
        base = Path(tempfile.mkdtemp(prefix=f"jev{i}-"))
        dest = base / repo.name

        def ignore(dirpath: str, names: list[str]) -> set[str]:
            return {n for n in names if n in SKIP_DIRS or n.endswith((".pyc", ".pyo"))}

        def copy(src: str, dst: str) -> str:
            if os.path.getsize(src) > MAX_COPY_BYTES:
                os.symlink(os.path.abspath(src), dst)   # read-only access; _safe() refuses edits through it
                return dst
            return shutil.copy2(src, dst)

        shutil.copytree(repo, dest, ignore=ignore, copy_function=copy, symlinks=True)
        return dest

    def _apply(self, res: SwarmResult, repo: Path) -> None:
        for path, (orig, new) in sorted(res.winner.edited.items()):
            target = repo / path
            try:
                current = target.read_text(encoding="utf-8")
            except OSError:
                res.conflicts.append(path)
                continue
            if current != orig:
                res.conflicts.append(path)
                continue
            target.write_text(new, encoding="utf-8")
            res.applied.append(path)


def _slice(recall: Recall | None, i: int, n: int, budget: int):
    """Give agent i a rotated, budget-bounded slice of the recalled fragments.

    Agent 0 gets the top of the ranking; others start further down, so the
    swarm explores different memories instead of N copies of the same prompt.
    """
    if recall is None or not recall.fragments:
        return []
    frags = list(recall.fragments)
    k = (i * max(1, len(frags) // max(1, n))) % len(frags)
    out, used = [], 0
    for f in frags[k:] + frags[:k]:
        if used + f.tokens <= budget:
            out.append(f)
            used += f.tokens
    return out

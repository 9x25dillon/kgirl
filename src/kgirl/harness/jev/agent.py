"""Jev: a Mr. Meeseeks agent. Exists to complete one goal, then disappears.

Context discipline (the whole point — cheap and fast):
  * no chat history: every step is ONE fresh prompt of
        goal + memory slice (JIT, token-bounded) + last K results + element table
  * the decision is one line (`OP i :: arg`), max ~24 output tokens
  * only EDIT / TYPE_TEXT spend generation budget, on a separate call

Lifecycle: spawn -> steps -> DONE | BLOCKED | exhausted | invalid | cancelled.
Its only legacy is the Trajectory; the swarm decides whether that trajectory
is worth distilling into Soup.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from ..util import clip, estimate_tokens
from .env import Environment
from .intuition import label_key
from .ops import InvalidDecision, grammar, parse_decision

Decide = Callable[[str, str], str]   # (system, prompt) -> one-line decision

SYSTEM = """You are Jev, a single-purpose agent. You exist only to finish the GOAL, then you stop.
Each turn you see a numbered TABLE. Reply with exactly ONE line and nothing else:
  OP [index] [:: text]
Operations:
{grammar}
Rules: use only indices shown in the current TABLE. Prefer the shortest path.
Finish with DONE :: <what changed> as soon as the goal is met; use BLOCKED :: <why> if it cannot be met."""


@dataclass(frozen=True)
class JevConfig:
    max_steps: int = 12
    history: int = 3             # previous action results shown (clipped)
    note_chars: int = 900
    max_invalid: int = 2         # consecutive invalid decisions before giving up
    decision_tokens: int = 40


@dataclass
class Step:
    n: int
    raw: str
    op: str
    target: int | None
    arg: str
    ok: bool
    note: str
    ms: float
    key: str = ""            # label_key of the targeted element (stable across runs)
    source: str = "llm"      # llm | intuition


@dataclass
class Trajectory:
    jev_id: str
    goal: str
    steps: list[Step] = field(default_factory=list)
    outcome: str = "running"      # done | blocked | exhausted | invalid | cancelled | error
    summary: str = ""
    recalled: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    elapsed: float = 0.0
    verified: bool | None = None
    verify_output: str = ""
    diff: str = ""
    workspace: str = ""
    variant: dict = field(default_factory=dict)
    edited: dict = field(default_factory=dict, repr=False)   # path -> (original, new)
    llm_calls: int = 0
    intuition_steps: int = 0
    surprised: bool = False
    routines_used: list[int] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.outcome == "done" and self.verified is not False

    def ops_signature(self) -> str:
        """Compressed op sequence, e.g. `SEARCH OPEN(src/x.py) EDIT(src/x.py) RUN DONE`."""
        return " ".join(s.op + (f"({s.arg})" if s.op in ("SEARCH",) else "") for s in self.steps if s.ok)

    def routine(self) -> list[dict]:
        """Successful steps as a replayable routine (consumed by Intuition and the skill forge)."""
        return [{"op": s.op, "key": s.key, "arg": s.arg} for s in self.steps if s.ok]


class Jev:
    def __init__(self, env: Environment, decide: Decide, memory_text: str = "", config: JevConfig | None = None,
                 on_step: Callable[[str, Step], None] | None = None, jev_id: str | None = None,
                 intuition=None):
        self.env = env
        self.intuition = intuition
        self.decide = decide
        self.memory_text = memory_text
        self.config = config or JevConfig()
        self.on_step = on_step
        self.id = jev_id or "jev-" + uuid.uuid4().hex[:6]
        self._system = SYSTEM.format(grammar=grammar(env.ops))

    def prompt(self, goal: str, recent: list[str]) -> str:
        obs = self.env.observe()
        parts = [f"GOAL: {goal}"]
        if self.memory_text:
            parts.append("MEMORY (lessons from earlier runs):\n" + self.memory_text)
        if recent:
            parts.append("RECENT:\n" + "\n---\n".join(recent))
        parts.append(f"STATE: {obs.header}")
        parts.append("TABLE:\n" + obs.table())
        parts.append("Your one-line decision:")
        return "\n\n".join(parts)

    def run(self, goal: str, cancel: threading.Event | None = None) -> Trajectory:
        cfg = self.config
        traj = Trajectory(self.id, goal)
        t0 = time.perf_counter()
        recent: list[str] = []
        invalid = 0
        done: list[tuple[str, str]] = []
        needs_target = {name for name, spec in self.env.ops.items() if spec.needs_target}
        try:
            for n in range(1, cfg.max_steps + 1):
                if cancel is not None and cancel.is_set():
                    traj.outcome = "cancelled"
                    break
                ts = time.perf_counter()
                obs = self.env.observe()
                prop = None
                if self.intuition is not None and not traj.surprised:
                    prop = self.intuition.propose(goal, done, obs.elements, needs_target)
                    if prop is not None and prop.confidence < self.intuition.threshold:
                        prop = None
                if prop is not None:
                    raw, source = prop.line, "intuition"
                    traj.routines_used.extend(r for r in prop.routines if r is not None and r >= 0)
                else:
                    prompt = self.prompt(goal, recent[-cfg.history:])
                    traj.prompt_tokens += estimate_tokens(self._system) + estimate_tokens(prompt)
                    raw, source = self.decide(self._system, prompt), "llm"
                    traj.llm_calls += 1
                elements = self.env.observe().elements
                n_el = len(elements)
                try:
                    d = parse_decision(raw, self.env.ops, n_el)
                except InvalidDecision as exc:
                    invalid += 1
                    if source == "intuition":
                        traj.surprised = True
                    step = Step(n, clip(raw or "", 120), "INVALID", None, "", False, str(exc),
                                (time.perf_counter() - ts) * 1000)
                    traj.steps.append(step)
                    recent.append(f"your last reply was invalid: {exc}")
                    self._emit(step)
                    if invalid >= cfg.max_invalid:
                        traj.outcome = "invalid"
                        break
                    continue
                invalid = 0
                key = label_key(elements[d.target].label) if d.target is not None else ""
                res = self.env.act(d)
                step = Step(n, d.raw, d.op, d.target, d.arg, res.ok, clip(res.note, cfg.note_chars),
                            (time.perf_counter() - ts) * 1000, key, source)
                if source == "intuition":
                    traj.intuition_steps += 1
                    if not res.ok:
                        traj.surprised = True       # System 1 was wrong: System 2 drives the rest of the run
                if res.ok:
                    done.append((d.op, key))
                traj.steps.append(step)
                self._emit(step)
                recent.append(f"{d.raw} -> {clip(res.note, cfg.note_chars)}")
                if res.terminal:
                    traj.outcome = "done" if res.success_claimed else "blocked"
                    traj.summary = d.arg
                    break
            else:
                traj.outcome = "exhausted"
        except Exception as exc:  # an agent crash is an outcome, not a harness crash
            traj.outcome = "error"
            traj.summary = f"{type(exc).__name__}: {exc}"[:300]
        traj.elapsed = time.perf_counter() - t0
        return traj

    def _emit(self, step: Step) -> None:
        if self.on_step:
            self.on_step(self.id, step)

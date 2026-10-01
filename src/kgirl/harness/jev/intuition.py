"""Intuition: a System-1 policy for Jev, learned from accepted trajectories.

A Meeseeks that has already solved "make test_calc pass" should not pay a model
call to rediscover OPEN calc.py → EDIT mean → RUN → DONE. Intuition stores each
accepted run as a *routine*, a sequence of (op, target key, arg), and at every step:

    1. finds routines whose goal resembles this goal  (word-bigram + unigram overlap)
       and whose steps so far match the agent's steps so far (strict prefix)
    2. resolves the routine's next target key against the CURRENT table
       (keys drop line numbers, so routines survive code motion)
    3. combines votes:  confidence = 1 − Π (1 − c_r),  c_r = sim_r · (1 − ½^(support_r + 1))

If confidence ≥ threshold, Jev acts on the proposal without calling the model.
The proposal still goes through the strict decision parser, the environment still
validates the target, and the swarm still re-runs the verifier.

Surprise: the first time an intuitive action fails, intuition is switched off for
the rest of that run (System 2 takes over). Routines that lead to verified wins
gain utility through Soup credit; routines that mislead lose it.

Only `active` trajectories become routines, and their goals and args are collapsed
to one capped line: a replayed `:: arg` must never add a second decision line.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..util import inline, jaccard, shingles
from .env import Element

_STRIP_META = re.compile(r"\s+\((?:from memory|[^)]*\blines?\b[^)]*)\)")
_SPAN = re.compile(r":\d+(?:-\d+)?")
_WORDS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def label_key(label: str) -> str:
    """Stable identity of a table element across runs: no line spans, no size metadata, no doc."""
    head = label.split(" — ")[0]
    head = _STRIP_META.sub("", head)
    head = _SPAN.sub("", head)
    return " ".join(head.split())


def goal_similarity(a: str, b: str) -> float:
    ua = {w.lower() for w in _WORDS.findall(a)}
    ub = {w.lower() for w in _WORDS.findall(b)}
    return 0.5 * jaccard(ua, ub) + 0.5 * jaccard(shingles(a, 2), shingles(b, 2))


@dataclass(frozen=True)
class RoutineStep:
    op: str
    key: str = ""
    arg: str = ""


@dataclass
class Routine:
    goal: str
    steps: list[RoutineStep]
    support: int = 1
    utility: float = 0.5
    fragment_id: int | None = None
    scope: str = "*"


@dataclass(frozen=True)
class Proposal:
    line: str
    confidence: float
    routines: tuple[int, ...]


@dataclass
class Intuition:
    routines: list[Routine] = field(default_factory=list)
    threshold: float = 0.6
    min_similarity: float = 0.35
    proposals: int = 0

    @classmethod
    def from_soup(cls, soup, scope: str | None = None, **kw) -> "Intuition":
        out = cls(**kw)
        for f in soup.all(status="active", kind="trajectory"):
            if not f.data or (scope and f.scope not in ("*", scope)):
                continue
            try:
                rec = json.loads(f.data)
            except ValueError:
                continue
            steps = [RoutineStep(inline(str(s["op"]), 16), inline(str(s.get("key", "")), 120),
                                 inline(str(s.get("arg", "")), 120))
                     for s in rec.get("steps", []) if isinstance(s, dict) and s.get("op")]
            if steps:
                out.routines.append(Routine(inline(str(rec.get("goal", "")), 200), steps, f.support, f.utility,
                                            f.id, f.scope))
        return out

    def propose(self, goal: str, done: list[tuple[str, str]], elements: tuple[Element, ...] | list[Element],
                needs_target: set[str]) -> Proposal | None:
        votes: dict[str, list[float]] = {}
        sources: dict[str, list[int]] = {}
        keys = [label_key(e.label) for e in elements]
        for i, r in enumerate(self.routines):
            sim = goal_similarity(goal, r.goal)
            if sim < self.min_similarity or len(r.steps) <= len(done):
                continue
            if [(s.op, s.key) for s in r.steps[:len(done)]] != done:
                continue
            nxt = r.steps[len(done)]
            if nxt.op in needs_target:
                if nxt.key not in keys:
                    continue
                line = f"{nxt.op} {keys.index(nxt.key)}"
            else:
                line = nxt.op
            if nxt.arg and nxt.op not in needs_target:
                line += f" :: {nxt.arg}"
            c = sim * (1 - 0.5 ** (r.support + 1)) * (0.5 + r.utility)
            votes.setdefault(line, []).append(min(c, 0.99))
            sources.setdefault(line, []).append(r.fragment_id if r.fragment_id is not None else -i)
        if not votes:
            return None
        best, conf = None, 0.0
        for line, cs in votes.items():
            miss = 1.0
            for c in cs:
                miss *= 1 - c
            if 1 - miss > conf:
                best, conf = line, 1 - miss
        self.proposals += 1
        return Proposal(best, conf, tuple(sources[best]))

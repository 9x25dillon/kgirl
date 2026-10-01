"""Skill forge: crystallize recurring verified routines into named, exportable skills.

    trajectories (Soup) ──group by (scope, routine shape)──▶ candidates
        support ≥ k  and  mean utility ≥ u  ──▶ Skill
            ├─ stored in Soup as kind="skill" (recallable, credit-bearing)
            └─ exported as Claude Code skills:  <dir>/<name>/SKILL.md

A routine's *shape* is its op sequence plus target keys, with search words and
summaries stripped, so "fix mean() in calc.py" and "fix mean() denominator" that
took the same path through the same symbols fold into one skill. The skill's
description is built from the goals that produced it; its steps are rendered as
instructions a human or Claude Code can follow, ending with the verifier that
proved it.

Level 0 mechanics (grouping, thresholds). Whether a forged skill transfers to a
*new* task is exactly what the support/utility evidence in its header tracks.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .soup import Soup

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STOP = frozenset("a an and the to of in on for is it be make makes made fix fixes when with this that "
                  "pass passes wrong bug please should".split())


@dataclass
class Skill:
    name: str
    scope: str
    description: str
    steps: list[dict]
    goals: list[str]
    support: int
    utility: float
    verify: list[str] | None
    sources: list[int] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def shape(self) -> str:
        return " → ".join(f"{s['op']}({s['key']})" if s.get("key") else s["op"] for s in self.steps)

    def instructions(self) -> list[str]:
        out = []
        for s in self.steps:
            op, key, arg = s["op"], s.get("key", ""), s.get("arg", "")
            if op == "SEARCH":
                out.append(f"Search the repo for: {arg}")
            elif op == "OPEN":
                out.append(f"Open `{key}` and read it.")
            elif op == "EDIT":
                out.append(f"Edit `{key}` toward the goal (smallest change that satisfies it).")
            elif op == "BLAST":
                out.append(f"Check the blast radius of `{key}` before changing it.")
            elif op == "RUN":
                cmd = " ".join(self.verify) if self.verify else "the project's verification command"
                out.append(f"Run `{cmd}` and read the result.")
            elif op == "DONE":
                out.append("Stop when the verifier passes; summarize what changed.")
        return out

    def to_skill_md(self) -> str:
        desc = self.description.replace("\n", " ").replace('"', "'")
        lines = ["---", f"name: {self.name}", f'description: "{desc}"', "---", "",
                 f"# {self.name}", "",
                 f"Forged by the kgirl harness from {self.support} verified run(s) in `{self.scope}` "
                 f"(mean utility {self.utility:.2f}). Shape: `{self.shape()}`.", "",
                 "## When to use", ""]
        lines += [f"- {g}" for g in self.goals[:5]]
        lines += ["", "## Steps", ""]
        lines += [f"{i}. {s}" for i, s in enumerate(self.instructions(), 1)]
        if self.files:
            lines += ["", "## Files this skill has changed", ""] + [f"- `{f}`" for f in self.files]
        if self.verify:
            lines += ["", "## Verify", "", "```bash", " ".join(self.verify), "```"]
        lines += ["", f"<!-- kgirl-skill sources={self.sources} -->", ""]
        return "\n".join(lines)


def _shape_key(steps: list[dict]) -> tuple:
    return tuple((s["op"], s.get("key", "")) for s in steps)


def _name(goals: list[str], files: list[str]) -> str:
    words = Counter(w.lower() for g in goals for w in _WORD.findall(g) if w.lower() not in _STOP and len(w) > 2)
    top = [w for w, _ in words.most_common(3)]
    stem = files[0].rsplit("/", 1)[-1].rsplit(".", 1)[0] if files else ""
    parts = ([stem] if stem and stem not in top else []) + top
    name = "-".join(parts)[:48].strip("-") or "routine"
    return re.sub(r"[^a-z0-9-]+", "-", name.lower())


def forge(soup: Soup, min_support: int = 2, min_utility: float = 0.5, store: bool = True) -> list[Skill]:
    groups: dict[tuple, list] = defaultdict(list)
    for f in soup.all(kind="trajectory"):
        if f.status == "retired" or not f.data:
            continue
        try:
            rec = json.loads(f.data)
        except ValueError:
            continue
        steps = rec.get("steps") or []
        if not steps:
            continue
        groups[(f.scope, _shape_key(steps))].append((f, rec))
    skills: list[Skill] = []
    for (scope, _), members in groups.items():
        support = sum(f.support for f, _ in members)
        utility = sum(f.utility * f.support for f, _ in members) / max(1, support)
        if support < min_support or utility < min_utility:
            continue
        goals = list(dict.fromkeys(rec["goal"] for _, rec in members))
        files = sorted({p for _, rec in members for p in rec.get("files", [])})
        verify = next((rec.get("verify") for _, rec in members if rec.get("verify")), None)
        steps = members[0][1]["steps"]
        sk = Skill(_name(goals, files), scope, "", steps, goals, support, utility, verify,
                   [f.id for f, _ in members], files)
        sk.description = (f"Verified routine for tasks like '{goals[0]}' in {scope}: "
                          f"{' → '.join(s['op'] for s in steps)}"
                          + (f" on {', '.join(files)}" if files else "") + ".")
        skills.append(sk)
        if store:
            soup.add(f"skill {sk.name}: {sk.description}", kind="skill", tags=",".join(files), source="forge",
                     scope=scope, data=json.dumps({"name": sk.name, "steps": steps, "goals": goals,
                                                   "verify": verify, "sources": sk.sources}))
    return sorted(skills, key=lambda s: (-s.support, s.name))


def export(skills: list[Skill], directory: str | Path, overwrite: bool = False) -> list[Path]:
    """Write Claude Code skills. Existing hand-written skills are never overwritten unless asked."""
    root = Path(directory)
    written = []
    for sk in skills:
        d = root / sk.name
        f = d / "SKILL.md"
        if f.exists() and not overwrite and "kgirl-skill sources=" not in f.read_text(encoding="utf-8"):
            continue
        d.mkdir(parents=True, exist_ok=True)
        f.write_text(sk.to_skill_md(), encoding="utf-8")
        written.append(f)
    return written

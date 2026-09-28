"""Jev's action space: a closed set of operations over a numbered element table.

The model answers every step with exactly one line:

    OP [INDEX] [:: ARG]

    OPEN 3            CLICK 7            SEARCH :: tokenizer padding
    EDIT 4            TYPE_TEXT 2        DONE :: fixed off-by-one in window()
    RUN               SCROLL_DOWN        BLOCKED :: needs an API key

Safety invariants (enforced here, not trusted to the model):
  * OP must be in the environment's closed set
  * INDEX must exist in the *current* observation table
  * ARG is inert text (search words / a summary), clipped, never executed
Anything else is an invalid decision and costs the agent a step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OpSpec:
    name: str
    needs_target: bool = False
    needs_arg: bool = False
    terminal: bool = False
    help: str = ""


@dataclass(frozen=True)
class Decision:
    op: str
    target: int | None = None
    arg: str = ""
    raw: str = ""


class InvalidDecision(ValueError):
    pass


_LINE = re.compile(r"^\s*([A-Za-z_]+)(?:\s*\[?\s*(\d+)\s*\]?)?\s*(?:(?:::|:|-)\s*(.*))?\s*$")


def parse_decision(text: str, specs: dict[str, OpSpec], n_elements: int, max_arg: int = 160) -> Decision:
    line = next((ln.strip().strip("`*> ") for ln in (text or "").splitlines() if ln.strip().strip("`")), "")
    m = _LINE.match(line)
    if not m:
        raise InvalidDecision(f"unparseable decision: {clip(line)!r}")
    op = m.group(1).upper()
    if op not in specs:
        raise InvalidDecision(f"unknown op {op!r}; allowed: {', '.join(specs)}")
    spec = specs[op]
    target = int(m.group(2)) if m.group(2) is not None else None
    arg = (m.group(3) or "").strip()[:max_arg]
    if spec.needs_target:
        if target is None:
            raise InvalidDecision(f"{op} needs an element index")
        if not 0 <= target < n_elements:
            raise InvalidDecision(f"index {target} not in table (0..{n_elements - 1})")
    elif target is not None and not spec.needs_arg:
        target = None  # tolerate a stray number on target-less ops
    if spec.needs_arg and not arg:
        raise InvalidDecision(f"{op} needs `:: text`")
    return Decision(op, target, arg, line)


def clip(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def grammar(specs: dict[str, OpSpec]) -> str:
    rows = []
    for s in specs.values():
        form = s.name + (" <i>" if s.needs_target else "") + (" :: <text>" if s.needs_arg else "")
        rows.append(f"  {form:<24} {s.help}")
    return "\n".join(rows)

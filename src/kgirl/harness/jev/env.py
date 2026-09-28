"""Environment contract shared by the code and browser worlds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .ops import Decision, OpSpec


@dataclass(frozen=True)
class Element:
    kind: str          # file | symbol | hit | link | button | input | select | ...
    label: str         # what the model sees
    ref: dict = field(default_factory=dict)   # environment-private locator (never shown)


@dataclass(frozen=True)
class Observation:
    header: str                    # e.g. "focus: src/x.py" or "page: Google Flights"
    elements: tuple[Element, ...]
    note: str = ""                 # result of the previous action (clipped by the agent)

    def table(self) -> str:
        return "\n".join(f"[{i}] {e.kind:<6} {e.label}" for i, e in enumerate(self.elements)) or "(empty)"


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    note: str
    terminal: bool = False
    success_claimed: bool = False


class Environment(Protocol):
    ops: dict[str, OpSpec]
    kind: str

    def observe(self) -> Observation: ...

    def act(self, decision: Decision) -> ActionResult: ...

    def close(self) -> None: ...

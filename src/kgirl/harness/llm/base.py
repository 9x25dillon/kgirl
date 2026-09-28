"""Backend contract. Every model the harness talks to satisfies `Backend`.

A backend is a pure function from (system, messages, limits) to a Completion;
it holds no conversation state. Availability is probed lazily and cached so the
router can fall through a chain (Claude -> Ollama -> offline) without paying a
network timeout on every call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class BackendError(RuntimeError):
    """Raised when a backend is unavailable or a call fails in a non-retryable way."""


@dataclass(frozen=True)
class Message:
    role: str      # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    backend: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    name: str
    model: str

    def available(self) -> bool: ...

    def complete(self, system: str, messages: list[Message], max_tokens: int = 1024,
                 temperature: float = 0.2, stop: list[str] | None = None) -> Completion: ...

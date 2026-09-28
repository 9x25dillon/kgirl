"""Deterministic backend for tests, replays and offline dry-runs."""

from __future__ import annotations

from typing import Callable, Iterable

from .base import BackendError, Completion, Message

Responder = Callable[[str, list[Message]], str]


class ScriptedBackend:
    """Replays a fixed list of outputs, or calls a responder function.

    `calls` records every (system, messages) pair for assertions.
    """

    name = "scripted"

    def __init__(self, script: Iterable[str] | Responder, model: str = "scripted", ok: bool = True):
        self.model = model
        self._fn: Responder | None = script if callable(script) else None
        self._queue = [] if callable(script) else list(script)
        self._ok = ok
        self.calls: list[tuple[str, list[Message]]] = []

    def available(self) -> bool:
        return self._ok

    def complete(self, system: str, messages: list[Message], max_tokens: int = 1024,
                 temperature: float = 0.2, stop: list[str] | None = None) -> Completion:
        if not self._ok:
            raise BackendError("scripted backend disabled")
        self.calls.append((system, list(messages)))
        if self._fn is not None:
            text = self._fn(system, messages)
        elif self._queue:
            text = self._queue.pop(0)
        else:
            raise BackendError("scripted backend exhausted")
        return Completion(text, self.model, self.name, sum(len(m.content) for m in messages) // 4, len(text) // 4)

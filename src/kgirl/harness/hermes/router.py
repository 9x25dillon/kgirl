"""Hermes routing policy: which model serves which role, and what a request means.

Roles (the only thing callers name — never a vendor):

    scout  retrieval reranking, repo/task summaries, trajectory distillation
           -> Claude (cheap tier), falls back to the local coder model
    smith  code edits, compile/test fix loops
           -> local Ollama coder; Claude only if KGIRL_ALLOW_CLOUD_CODE=1
    jev    one-token-ish per-step decisions for Meeseeks agents
           -> smallest local model, then the local coder

Each role is an ordered fallback chain. A BackendError moves to the next link;
exhausting the chain raises NoBackend with every link's reason, so failures are
explained instead of silent.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from enum import Enum

from ..llm import AnthropicBackend, Backend, BackendError, Completion, Message, OllamaBackend

SCOUT, SMITH, JEV = "scout", "smith", "jev"


class NoBackend(BackendError):
    pass


class Intent(str, Enum):
    ASK = "ask"          # explain / where / how does X work
    TASK = "task"        # change code: fix, implement, refactor, add
    BLAST = "blast"      # impact analysis
    ARCH = "arch"        # architecture overview of a repo
    RECALL = "recall"    # what do we remember about X


_INTENT_RULES: list[tuple[Intent, re.Pattern]] = [
    (Intent.BLAST, re.compile(r"\b(blast|impact|who (uses|calls|imports)|depends on|break if|affected)\b", re.I)),
    (Intent.ARCH, re.compile(r"\b(architecture|overview|structure|layout|map of)\b", re.I)),
    (Intent.RECALL, re.compile(r"\b(remember|recall|last time|lesson|memory of)\b", re.I)),
    (Intent.TASK, re.compile(r"^\s*(fix|implement|add|refactor|rename|remove|delete|write|create|make|port|"
                             r"update|upgrade|migrate|optimi[sz]e|compile|build)\b", re.I)),
]


def classify(text: str) -> Intent:
    for intent, pat in _INTENT_RULES:
        if pat.search(text):
            return intent
    return Intent.ASK


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0


class Router:
    def __init__(self, chains: dict[str, list[Backend]]):
        self.chains = chains
        self.usage: dict[str, Usage] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "Router":
        smith_model = os.environ.get("KGIRL_SMITH_MODEL", "qwen2.5-coder:7b")
        jev_model = os.environ.get("KGIRL_JEV_MODEL", "qwen2.5-coder:1.5b")
        scout = AnthropicBackend()
        smith = OllamaBackend(smith_model)
        jev = OllamaBackend(jev_model, timeout=60.0)
        smith_chain: list[Backend] = [smith]
        if os.environ.get("KGIRL_ALLOW_CLOUD_CODE") == "1":
            smith_chain.append(AnthropicBackend(os.environ.get("KGIRL_CLOUD_CODE_MODEL", "claude-opus-5")))
        return cls({SCOUT: [scout, smith], SMITH: smith_chain, JEV: [jev, smith]})

    def pick(self, role: str) -> Backend | None:
        for b in self.chains.get(role, []):
            if b.available():
                return b
        return None

    def describe(self) -> dict[str, list[str]]:
        return {role: [f"{b.name}:{b.model}{'' if b.available() else ' (down)'}" for b in chain]
                for role, chain in self.chains.items()}

    def complete(self, role: str, system: str, messages: list[Message], max_tokens: int = 1024,
                 temperature: float = 0.2, stop: list[str] | None = None) -> Completion:
        reasons: list[str] = []
        for b in self.chains.get(role, []):
            if not b.available():
                reasons.append(f"{b.name}:{b.model} unavailable")
                continue
            try:
                out = b.complete(system, messages, max_tokens=max_tokens, temperature=temperature, stop=stop)
            except BackendError as exc:
                reasons.append(f"{b.name}:{b.model} {exc}")
                with self._lock:
                    self.usage.setdefault(role, Usage()).failures += 1
                continue
            with self._lock:
                u = self.usage.setdefault(role, Usage())
                u.calls += 1
                u.input_tokens += out.input_tokens
                u.output_tokens += out.output_tokens
            return out
        raise NoBackend(f"no backend could serve role {role!r}: " + "; ".join(reasons or ["empty chain"]))

    def as_service(self):
        """Adapter for `bus.serve('llm.complete', router.as_service())`."""
        def serve(p: dict) -> Completion:
            msgs = [m if isinstance(m, Message) else Message(**m) for m in p["messages"]]
            return self.complete(p["role"], p.get("system", ""), msgs, p.get("max_tokens", 1024),
                                 p.get("temperature", 0.2), p.get("stop"))
        return serve

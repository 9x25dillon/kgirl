"""Fake backends for the helix tests: scripted replies, exact token counts, every call recorded.

A fake counts tokens as whitespace-separated words, so a test can state the provider's count exactly.
`count=False` makes it report zero, the way a backend that does not count does.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests.helix  # noqa: F401  (sys.path)
from kgirl.harness.hermes.router import CHAT, Router
from kgirl.harness.helix import Helix, StateStore
from kgirl.harness.llm import BackendError, Completion

BODY = ("Return the local answer when two small models agree and the retrieval energy is low. "
        "Escalate when they disagree on any figure. The cutoff of 0.948 was calibrated on nomic vectors.")
WEAK = "The cutoff of 0.948 was calibrated on nomic vectors."


def tune(quote: str = WEAK, axis: str = "rigor", d: float = 0.1, because: str = "the figure has no method") -> str:
    return f'TUNE: {{"quote": "{quote}", "axis": "{axis}", "d": {d}, "because": "{because}"}}'


def expert(body: str = BODY, trailer: str = "TUNE: none") -> str:
    return f"{body}\n{trailer}"


def defect(quote: str = "Escalate when they disagree on any figure.", repair: str = "say which figures count") -> str:
    return f'DEFECT: {{"quote": "{quote}", "repair": "{repair}"}}'


def synth(answer: str = "Return the local answer when both small models agree; otherwise escalate.",
          unresolved: str | None = "[]") -> str:
    return answer if unresolved is None else f"{answer}\nUNRESOLVED: {unresolved}"


class Fake:
    name = "fake"

    def __init__(self, replies, model: str = "qwen-fake:7b", count: bool = True, stop: str = "stop"):
        self.model, self.count, self.stop = model, count, stop
        self._replies = list(replies)
        self.calls: list[dict] = []

    def available(self) -> bool:
        return True

    def complete(self, system, messages, max_tokens=1024, temperature=0.2, stop=None) -> Completion:
        user = messages[-1].content
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens, "temperature": temperature})
        if not self._replies:
            raise BackendError("fake backend exhausted")
        text = self._replies.pop(0)
        n_in, n_out = (len(user.split()), len(text.split())) if self.count else (0, 0)
        return Completion(text, self.model, self.name, n_in, n_out, self.stop)


class HelixCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ)                    # the user's own home and models must not leak in
        env.start()
        self.addCleanup(env.stop)
        for key in ("KGIRL_HOME", "KGIRL_HELIX_CRITIC_MODEL", "KGIRL_MIND_CTX"):
            os.environ.pop(key, None)
        self.base = Path(self.tmp.name)

    def store(self, name: str = "home") -> StateStore:
        s = StateStore(self.base / name)
        self.addCleanup(s.close)
        return s

    def helix(self, gen_replies, critic_replies=None, *, home: str = "home", **kw):
        """A Helix over fakes. Returns (helix, generator, critic); critic is None when no replies were given."""
        gen = Fake(gen_replies, **{k: kw.pop(k) for k in ("count", "stop") if k in kw})
        critic = Fake(critic_replies, model="llama-fake:3b") if critic_replies is not None else None
        h = Helix(Router({CHAT: [gen]}), store=self.store(home), critic=critic, **kw)
        return h, gen, critic

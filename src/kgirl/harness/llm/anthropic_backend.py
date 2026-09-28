"""Claude via the official Anthropic Python SDK (optional dependency).

Role in the harness: retrieval reranking, repo/task summaries, distillation of
successful trajectories into abstractions. The default model is the cheap tier
(`claude-haiku-4-5`) because these calls are high-volume and short; override
with KGIRL_SCOUT_MODEL (e.g. `claude-opus-5`) when quality matters more.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from .base import BackendError, Completion, Message

DEFAULT_SCOUT_MODEL = "claude-haiku-4-5"


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str | None = None, timeout: float = 60.0, max_retries: int = 2):
        self.model = model or os.environ.get("KGIRL_SCOUT_MODEL", DEFAULT_SCOUT_MODEL)
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None
        self._ok: bool | None = None

    def available(self) -> bool:
        if self._ok is None:
            if importlib.util.find_spec("anthropic") is None:
                self._ok = False
                return False
            self._ok = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                            or (Path.home() / ".config" / "anthropic").exists())
        return self._ok

    def _get_client(self):
        if self._client is None:
            if not self.available():
                raise BackendError("anthropic SDK not installed or no credentials configured")
            import anthropic
            self._client = anthropic.Anthropic(timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def complete(self, system: str, messages: list[Message], max_tokens: int = 1024,
                 temperature: float = 0.2, stop: list[str] | None = None) -> Completion:
        import anthropic
        client = self._get_client()
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system,
                      messages=[{"role": m.role, "content": m.content} for m in messages])
        if stop:
            kwargs["stop_sequences"] = stop
        if self.model.startswith("claude-haiku") or self.model.startswith("claude-3"):
            kwargs["temperature"] = temperature  # sampling params are rejected on the newest tiers
        try:
            resp = client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            raise BackendError(f"anthropic rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise BackendError(f"anthropic HTTP {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            self._ok = False
            raise BackendError(f"anthropic unreachable: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise BackendError("anthropic declined the request (stop_reason=refusal)")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(text, resp.model, self.name, resp.usage.input_tokens, resp.usage.output_tokens,
                          resp.stop_reason or "")

"""Local models via Ollama's HTTP API (stdlib only).

Role in the harness: code edits, compilation-fix loops and Jev's per-step
decisions — the high-frequency calls that should cost nothing and never leave
the machine. Host from OLLAMA_HOST (default http://localhost:11434).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import BackendError, Completion, Message


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str, host: str | None = None, timeout: float = 300.0, keep_alive: str = "10m",
                 num_ctx: int | None = None):
        """`num_ctx` is the context window to load the model with. None leaves the server's default, which is
        small (4096 tokens on current Ollama): a longer prompt is cut from the front without an error. Ollama
        reloads a model when the window changes, so callers that share a model should share one backend."""
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx if num_ctx and num_ctx > 0 else None
        self._ok: bool | None = None

    def _options(self, temperature: float, max_tokens: int, stop: list[str] | None = None) -> dict:
        opts: dict = {"temperature": temperature, "num_predict": max_tokens}
        if stop:
            opts["stop"] = stop
        if self.num_ctx:
            opts["num_ctx"] = self.num_ctx
        return opts

    def _request(self, path: str, payload: dict | None, timeout: float) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.host + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if data else "GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local, configured host
            return json.loads(resp.read().decode("utf-8"))

    def available(self) -> bool:
        if self._ok is None:
            try:
                tags = self._request("/api/tags", None, timeout=1.5)
                names = {m.get("name", "") for m in tags.get("models", [])}
                self._ok = self.model in names or f"{self.model}:latest" in names or \
                    any(n.split(":")[0] == self.model for n in names)
            except (OSError, ValueError, urllib.error.URLError):
                self._ok = False
        return self._ok

    def complete(self, system: str, messages: list[Message], max_tokens: int = 1024,
                 temperature: float = 0.2, stop: list[str] | None = None) -> Completion:
        payload = {
            "model": self.model, "stream": False, "keep_alive": self.keep_alive,
            "messages": [{"role": "system", "content": system}] +
                        [{"role": m.role, "content": m.content} for m in messages],
            "options": self._options(temperature, max_tokens, stop),
        }
        try:
            out = self._request("/api/chat", payload, timeout=self.timeout)
        except (OSError, urllib.error.URLError) as exc:
            self._ok = False
            raise BackendError(f"ollama unreachable at {self.host}: {exc}") from exc
        if "error" in out:
            raise BackendError(f"ollama: {out['error']}")
        return Completion(out.get("message", {}).get("content", ""), self.model, self.name,
                          out.get("prompt_eval_count", 0), out.get("eval_count", 0), out.get("done_reason", ""))

    def stream(self, system: str, messages: list[Message], max_tokens: int = 1024, temperature: float = 0.2,
               stop: list[str] | None = None):
        """Yield the reply as it is generated (Ollama's newline-delimited JSON stream)."""
        payload = {
            "model": self.model, "stream": True, "keep_alive": self.keep_alive,
            "messages": [{"role": "system", "content": system}] +
                        [{"role": m.role, "content": m.content} for m in messages],
            "options": self._options(temperature, max_tokens, stop),
        }
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - local, configured host
                for raw in resp:
                    if not raw.strip():
                        continue
                    chunk = json.loads(raw.decode("utf-8"))
                    if "error" in chunk:
                        raise BackendError(f"ollama: {chunk['error']}")
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        return
        except (OSError, urllib.error.URLError, ValueError) as exc:
            self._ok = False
            raise BackendError(f"ollama stream failed at {self.host}: {exc}") from exc

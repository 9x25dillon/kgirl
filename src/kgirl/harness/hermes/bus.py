"""Hermes bus: typed envelopes, topic routing, request/reply, full causal trace.

Every cross-layer interaction in the harness (atlas query, memory recall, model
call, agent spawn/step/death, curation) is an Envelope on a topic. That buys:

* one choke point for observability (JSONL trace, per-topic latency/counters)
* causal replay: `correlation_id` groups a whole task, `causation_id` links
  each envelope to the one that produced it
* loose coupling: layers depend on topic contracts, not on each other

Delivery is synchronous and deterministic (subscription order, FIFO cascade),
so tests and replays are reproducible. Cascades are bounded by `max_hops`.
Thread-safe: swarms publish from worker threads.
"""

from __future__ import annotations

import fnmatch
import itertools
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


class HermesError(RuntimeError):
    pass


@dataclass(frozen=True)
class Envelope:
    topic: str
    payload: Any
    sender: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    correlation_id: str = ""
    causation_id: str = ""
    ts: float = field(default_factory=time.time)
    hop: int = 0

    def child(self, topic: str, payload: Any, sender: str) -> "Envelope":
        return Envelope(topic, payload, sender, correlation_id=self.correlation_id or self.id,
                        causation_id=self.id, hop=self.hop + 1)


Handler = Callable[[Envelope], "None | Envelope | list[tuple[str, Any]]"]
Service = Callable[[Any], Any]


@dataclass
class TopicStats:
    count: int = 0
    errors: int = 0
    total_ms: float = 0.0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


class Bus:
    def __init__(self, trace_path: str | Path | None = None, max_hops: int = 16, ring: int = 2000):
        self._subs: list[tuple[int, str, str, Handler]] = []
        self._services: dict[str, tuple[str, Service]] = {}
        self._ids = itertools.count(1)
        self._lock = threading.RLock()
        self._trace = deque(maxlen=ring)
        self._trace_path = Path(trace_path) if trace_path else None
        if self._trace_path:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_hops = max_hops
        self.stats: dict[str, TopicStats] = {}

    # -------------------------------------------------------------- pub/sub

    def subscribe(self, pattern: str, handler: Handler, name: str = "") -> int:
        with self._lock:
            token = next(self._ids)
            self._subs.append((token, pattern, name or getattr(handler, "__name__", "handler"), handler))
            return token

    def unsubscribe(self, token: int) -> None:
        with self._lock:
            self._subs = [s for s in self._subs if s[0] != token]

    def publish(self, topic: str, payload: Any = None, sender: str = "", parent: Envelope | None = None) -> Envelope:
        env = parent.child(topic, payload, sender) if parent else Envelope(topic, payload, sender)
        self._dispatch(env)
        return env

    def _dispatch(self, root: Envelope) -> None:
        queue = deque([root])
        while queue:
            env = queue.popleft()
            self._record(env)
            if env.hop > self.max_hops:
                self._record(env.child("hermes.dropped", {"reason": "max_hops", "topic": env.topic}, "hermes"))
                continue
            with self._lock:
                subs = [s for s in self._subs if fnmatch.fnmatchcase(env.topic, s[1])]
            for _, _, name, handler in subs:
                try:
                    out = handler(env)
                except Exception as exc:  # a failing observer must not break the publisher
                    self._bump(env.topic, 0.0, error=True)
                    self._record(env.child("hermes.handler_error",
                                           {"handler": name, "topic": env.topic, "error": repr(exc)[:300]}, "hermes"))
                    continue
                if isinstance(out, Envelope):
                    queue.append(out)
                elif isinstance(out, list):
                    queue.extend(env.child(t, p, name) for t, p in out)

    # -------------------------------------------------------------- request/reply

    def serve(self, topic: str, fn: Service, name: str = "") -> None:
        with self._lock:
            if topic in self._services:
                raise HermesError(f"topic {topic!r} already served by {self._services[topic][0]}")
            self._services[topic] = (name or getattr(fn, "__name__", "service"), fn)

    def request(self, topic: str, payload: Any = None, sender: str = "", parent: Envelope | None = None) -> Any:
        with self._lock:
            svc = self._services.get(topic)
        if svc is None:
            raise HermesError(f"no service for topic {topic!r}")
        req = parent.child(topic, payload, sender) if parent else Envelope(topic, payload, sender)
        self._dispatch(req)
        t0 = time.perf_counter()
        try:
            result = svc[1](payload)
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            self._bump(topic, ms, error=True)
            self._dispatch(req.child(topic + ".error", {"error": repr(exc)[:500]}, svc[0]))
            raise
        ms = (time.perf_counter() - t0) * 1000
        self._bump(topic, ms)
        self._dispatch(req.child(topic + ".reply", _summarize(result), svc[0]))
        return result

    def has_service(self, topic: str) -> bool:
        return topic in self._services

    # -------------------------------------------------------------- observability

    def _bump(self, topic: str, ms: float, error: bool = False) -> None:
        with self._lock:
            st = self.stats.setdefault(topic, TopicStats())
            st.count += 1
            st.total_ms += ms
            st.errors += int(error)

    def _record(self, env: Envelope) -> None:
        with self._lock:
            self._trace.append(env)
            if self._trace_path:
                row = asdict(env)
                row["payload"] = _summarize(env.payload)
                with self._trace_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    def trace(self, correlation_id: str | None = None, topic: str = "*") -> list[Envelope]:
        with self._lock:
            items = list(self._trace)
        return [e for e in items if fnmatch.fnmatchcase(e.topic, topic)
                and (correlation_id is None or e.correlation_id == correlation_id or e.id == correlation_id)]


def _summarize(value: Any, limit: int = 600) -> Any:
    """Keep traces small: long strings and big collections are clipped."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"…(+{len(value) - limit})"
    if isinstance(value, dict):
        return {k: _summarize(v, limit // 2 or 50) for k, v in list(value.items())[:40]}
    if isinstance(value, (list, tuple)):
        items = [_summarize(v, limit // 2 or 50) for v in list(value)[:20]]
        return items + ([f"…(+{len(value) - 20})"] if len(value) > 20 else [])
    if hasattr(value, "__dataclass_fields__"):
        return _summarize(asdict(value), limit)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _summarize(repr(value), limit)

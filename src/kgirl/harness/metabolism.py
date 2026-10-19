"""Metabolism: the harness run as an organism under the Algorithmic Life Equation (kgirl.life).

Every user verb (`ask`, `task`) is one step of the life map. The Hermes bus is the organism's sensorium. A
subscriber on `*` accumulates what the verb cost, and the episode closes on `assistant.answer` or `swarm.end`:

    I_t  = (success, latency, cost, errors)        success ∈ {0, 1}; latency = log(1+s)/log(1+600);
                                                    cost = tokens / budget, capped at 2, halved; errors = failed
                                                    model calls / calls
    E_t  = 1                                       one episode's budget flows in per verb
    X_t  = tokens / budget                         measured work, charged as w_ext · X_t

So F is a token reserve in units of one episode budget (KGIRL_LIFE_BUDGET tokens, default 20000) that leaks
at δF. The world model learns what this harness usually achieves, and ε is its surprise at the latest outcome.

What the organism controls (dependency inversion: the harness asks, the metabolism answers):

    plan()        swarm size N* from the replication criterion, using the agent success rate p (an exponentially
                  decayed Beta posterior over agent outcomes) and the value of a solved task in agent costs
                  (KGIRL_LIFE_VALUE, default 10, which gives the old default of 3 agents at p = ½). N* is capped
                  by the affordable sizes N_reach(F, O, C), the same count that defines the reachable potential
                  P, and by KGIRL_LIFE_MAX_SWARM (default 6). The launch stagger Δ minimizes E[cost] + w·E[latency]
                  (w = KGIRL_LIFE_LATENCY, default 2) for the measured agent run time.
    forgetting()  Soup's curator decay γ_eff = γ^(1 + Var ε / v_scale), capped at γ⁴: when outcomes stop matching
                  the model, old evidence fades faster. This is the variable forgetting factor of adaptive filtering.
    vitals()      A, C, O, F, R, P, C_eff, the steady state, and `rigidity` over the C_eff history.

Continuity across restarts: state lives in $KGIRL_HOME/life.json and is written atomically after each episode.
A missing file starts a fresh organism. A corrupt or incompatible file also starts fresh, and its first step is
marked a discontinuity, which costs the self-model ξ. A lost state is a reset, not a silent continuation.

KGIRL_LIFE=0 turns all of this off; the harness then behaves exactly as before.

Confidence: the bookkeeping and the decision formulas are Level 0. That these controls make the harness cheaper
or faster for a given user is Level 2. `vitals` and the ledger in life.json are how to measure it.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..life.analysis import rigidity, steady_state
from ..life.core import LifeParams, LifeState, overhead, reach_count, step, window_quality
from ..life.replication import best_stagger, swarm_size

SCHEMA = 1
CHANNELS = ("success", "latency", "cost", "errors")
MAX_SIZES = 8
HISTORY = 256

PARAMS = LifeParams(
    eta_in=1.0, w0=0.01, wA=0.0, wM=0.0, wu=0.0, w_ext=1.0, delta0=0.01, deltaF=0.05, delta_eps=0.1,
    a=0.5, b=0.5, C_min=0.15, C_max=0.85, eta_C=0.2, eta_V=0.5, v_ref=0.02, eta_M=0.2, lambda_reg=0.01,
    rho=0.3, var_span=8, eta_R=0.2, skill_span=16, action_costs=tuple(float(n) for n in range(1, MAX_SIZES + 1)),
    horizon=1, eta_pi=0.0, F_min=0.5, O_min=0.2, K_F=1.0)


def enabled() -> bool:
    return os.environ.get("KGIRL_LIFE", "1") != "0"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Plan:
    n: int
    stagger_s: float
    p_agent: float
    affordable: int
    expected_cost: float | None      # agent-seconds, when the agent run time is known
    expected_latency: float | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Episode:
    kind: str = ""
    t0: float = 0.0
    tokens: int = 0
    jev_tokens: int = 0
    calls: int = 0
    errors: int = 0
    agents: int = 0


class Metabolism:
    def __init__(self, path: str | os.PathLike | None = None, params: LifeParams | None = None,
                 budget_tokens: float | None = None):
        self.path = Path(path) if path else None
        self.p = (params or PARAMS).check()
        self.budget = budget_tokens or _env_float("KGIRL_LIFE_BUDGET", 20000.0)
        self.value_ratio = _env_float("KGIRL_LIFE_VALUE", 10.0)
        self.latency_weight = _env_float("KGIRL_LIFE_LATENCY", 2.0)
        self.max_swarm = max(1, min(MAX_SIZES, int(_env_float("KGIRL_LIFE_MAX_SWARM", 6))))
        self.v_scale = 0.05
        self._lock = threading.RLock()
        self._ep: _Episode | None = None
        self._pending_reset = False
        self.recovered: str | None = None
        self.state = LifeState.initial(len(CHANNELS), self.p, F=5.0, O=0.6, R=0.5)
        self.stats = {"episodes": 0, "agent_wins": 0.0, "agent_losses": 0.0, "agent_seconds": None,
                      "agent_cost": 0.15, "decay": 0.97}
        self.history: list[dict] = []
        if self.path:
            self._load()

    # ------------------------------------------------------------------ persistence

    def _load(self) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            self._fresh(f"unreadable: {exc}")
            return
        try:
            data = json.loads(raw)
            if data.get("schema") != SCHEMA:
                raise ValueError(f"schema {data.get('schema')} != {SCHEMA}")
            state = LifeState.from_dict(data["state"])
            if state.d != len(CHANNELS):
                raise ValueError("channel count changed")
            self.state = state
            self.stats.update(data.get("stats", {}))
            self.history = list(data.get("history", []))[-HISTORY:]
        except (ValueError, KeyError, TypeError) as exc:
            self._fresh(f"corrupt: {exc}")

    def _fresh(self, reason: str) -> None:
        self.recovered = reason
        self._pending_reset = True                 # the next step is a discontinuity: the self-model pays ξ

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            payload = {"schema": SCHEMA, "saved": time.time(), "channels": CHANNELS, "state": self.state.to_dict(),
                       "stats": self.stats, "history": self.history[-HISTORY:]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".life-", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ sensing (Hermes subscriber)

    def attach(self, bus) -> int:
        return bus.subscribe("*", self.on_envelope, name="metabolism")

    def on_envelope(self, env) -> None:
        topic, pay = env.topic, env.payload if isinstance(env.payload, dict) else {}
        with self._lock:
            if topic in ("assistant.ask", "assistant.task"):
                self._ep = _Episode(topic.split(".")[1], time.time())
            ep = self._ep
            if ep is None:
                return
            if topic == "llm.complete":
                ep.calls += 1
            elif topic == "llm.complete.reply":
                ep.tokens += int(pay.get("input_tokens") or 0) + int(pay.get("output_tokens") or 0)
            elif topic == "llm.complete.error":
                ep.errors += 1
            elif topic == "jev.done":
                ep.jev_tokens += int(pay.get("tokens") or 0)
                ep.agents += 1
                self._agent(pay)
            elif topic == "assistant.answer":
                self._close(1.0 if pay.get("backend") not in (None, "extractive") else 0.0)
            elif topic == "swarm.end":
                self._close(1.0 if pay.get("winner") else 0.0)

    def _agent(self, pay: dict) -> None:
        if pay.get("outcome") == "cancelled":                 # censored: it neither won nor lost
            return
        win = pay.get("outcome") == "done" and pay.get("verified") is not False
        st = self.stats
        st["agent_wins" if win else "agent_losses"] += 1.0
        if win and pay.get("elapsed"):
            prev = st.get("agent_seconds")
            st["agent_seconds"] = float(pay["elapsed"]) if prev is None else 0.7 * prev + 0.3 * float(pay["elapsed"])

    def _close(self, success: float) -> None:
        ep, self._ep = self._ep, None
        if ep is None:
            return
        tokens = ep.tokens or ep.jev_tokens
        work = tokens / self.budget
        latency = min(1.0, math.log1p(max(0.0, time.time() - ep.t0)) / math.log1p(600.0))
        obs = (success, latency, min(2.0, work) / 2.0, ep.errors / max(1, ep.calls))
        if ep.kind == "task":
            st = self.stats
            st["agent_wins"] *= st["decay"]
            st["agent_losses"] *= st["decay"]
            if ep.agents:
                self.note_agent_cost(tokens / ep.agents)
        self.observe(obs, work=work, kind=ep.kind)

    def observe(self, obs: tuple[float, ...], energy: float = 1.0, work: float = 0.0, kind: str = "manual") -> dict:
        """One life step from an episode's outcome; also the entry point for tests and replays."""
        with self._lock:
            reset, self._pending_reset = self._pending_reset, False
            self.state, rec = step(self.p, self.state, obs, energy, 0, work=work, reset=reset)
            self.state = replace(self.state, P=self._potential())
            self.stats["episodes"] = int(self.stats.get("episodes", 0)) + 1
            row = {"t": self.state.t, "kind": kind, "obs": [round(x, 4) for x in obs], "work": round(work, 4),
                   "err": round(rec.err, 4), "paid": round(rec.paid, 4), "ceff": round(rec.ceff, 5),
                   "F": round(self.state.F, 4), "O": round(self.state.O, 4), "C": round(self.state.C, 4),
                   "R": round(self.state.R, 4), "disc": rec.discontinuity}
            self.history.append(row)
            del self.history[:-HISTORY]
        try:
            self.save()
        except OSError:                              # a read-only home must not break the verb that just ran
            pass
        return row

    # ------------------------------------------------------------------ controls

    def p_agent(self) -> float:
        st = self.stats
        return (st["agent_wins"] + 1.0) / (st["agent_wins"] + st["agent_losses"] + 2.0)

    def affordable(self) -> int:
        """Swarm sizes the reserve can pay for: N_reach with each agent costing the measured mean agent cost."""
        s = self.state
        q = replace(self.p, wu=max(1e-6, float(self.stats.get("agent_cost") or 0.15)))
        return reach_count(q, s.O * window_quality(q, s.C) * s.F, overhead(q, s))

    def _potential(self) -> float:
        return math.log1p(self.affordable())

    def plan(self, max_n: int | None = None) -> Plan:
        with self._lock:
            cap = max(1, min(max_n or self.max_swarm, MAX_SIZES))
            p, aff = self.p_agent(), self.affordable()
            n = swarm_size(p, self.value_ratio, cap, max(1, aff))
            tau = self.stats.get("agent_seconds")
            reason = (f"p_agent={p:.2f} (Beta over agent outcomes), V/c={self.value_ratio:g} → "
                      f"N*={swarm_size(p, self.value_ratio, cap)}; reserve affords {aff}; cap {cap}")
            if tau and n > 1:
                sc = best_stagger(p, n, float(tau), self.latency_weight)
                return Plan(n, sc.delta, p, aff, sc.cost, sc.latency,
                            reason + f"; stagger Δ={sc.delta:.2f}s for τ̂={tau:.1f}s")
            return Plan(n, 0.0, p, aff, None, None, reason + "; no agent run time measured yet: no stagger")

    def note_agent_cost(self, tokens_per_agent: float) -> None:
        with self._lock:
            c = tokens_per_agent / self.budget
            prev = float(self.stats.get("agent_cost") or c)
            self.stats["agent_cost"] = 0.7 * prev + 0.3 * c

    def forgetting(self, gamma: float) -> float:
        with self._lock:
            v = sum(self.state.eps_var) / self.state.d
        return max(gamma ** 4, gamma ** (1.0 + v / self.v_scale))

    def vitals(self) -> dict[str, Any]:
        with self._lock:
            s, p = self.state, self.p
            ceff = [h["ceff"] for h in self.history]
            rig = rigidity(ceff) if len(ceff) >= 30 else None
            works = [h["work"] for h in self.history[-50:]] or [0.0]
            ss = steady_state(p, 1.0, s.e_mean, sum(s.eps_var) / s.d, 0.0, 0.0, sum(works) / len(works))
            return {"episodes": self.stats.get("episodes", 0), "t": s.t, "A": s.A, "C": s.C, "O": s.O, "F": s.F,
                    "R": s.R, "P": s.P, "C_eff": s.functional_index(p), "surprise_var": sum(s.eps_var) / s.d,
                    "predicted_next": dict(zip(CHANNELS, (round(x, 3) for x in _predict(s)))),
                    "p_agent": self.p_agent(), "agent_seconds": self.stats.get("agent_seconds"),
                    "affordable_swarm": self.affordable(), "plan": self.plan().to_dict(),
                    "forgetting_at_0.97": self.forgetting(0.97),
                    "steady_state": {"F": ss.F, "O": ss.O, "C": ss.C, "starving": ss.starving},
                    "rigidity": rig.to_dict() if rig else None, "recovered": self.recovered,
                    "budget_tokens": self.budget}


def _predict(s: LifeState) -> tuple[float, ...]:
    phi = s.H + (1.0,)
    return tuple(sum(m * x for m, x in zip(row, phi)) for row in s.M)


def render(v: dict) -> str:
    lines = [f"episodes {v['episodes']}  C_eff {v['C_eff']:.3f}  (A {v['A']:.2f} · C {v['C']:.2f} · O {v['O']:.2f} · "
             f"R {v['R']:.2f} · F/(F+K) {v['F'] / (v['F'] + 1):.2f})",
             f"reserve F {v['F']:.2f} budgets of {v['budget_tokens']:.0f} tokens; steady state F* {v['steady_state']['F']:.2f}"
             + (" (starving: spending exceeds inflow)" if v["steady_state"]["starving"] else ""),
             f"surprise Var ε {v['surprise_var']:.4f}; next episode predicted {v['predicted_next']}",
             f"agents: p {v['p_agent']:.2f}, run time {v['agent_seconds'] or '—'} s, affordable swarm {v['affordable_swarm']}",
             f"plan: N={v['plan']['n']} stagger {v['plan']['stagger_s']:.2f}s — {v['plan']['reason']}",
             f"Soup forgetting γ 0.97 → {v['forgetting_at_0.97']:.4f}"]
    rig = v.get("rigidity")
    lines.append(f"rigidity: {rig['verdict']}" if rig else "rigidity: needs ≥ 30 episodes")
    if v.get("recovered"):
        lines.append(f"state was reset on load ({v['recovered']})")
    return "\n".join(lines)

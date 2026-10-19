"""Staggered concurrency, made testable: availability of phase-offset replicas, and sizing a swarm.

The manuscript (§4) proposes replicas of a reference process T with cycle τ, offset by half a cycle,

    T_k(t) = T(t − (k − 1) τ / 2),   k = 1, 2, 3,

and the design criterion  availability gain > coordination cost + replication cost.

Finding (Level 0): offsets 0, τ/2 and τ put replica 3 in phase with replica 1 (τ ≡ 0 mod τ), so three
replicas get exactly the availability of two and pay for three. Spreading phases uniformly,
o_k = (k − 1) τ / N, is optimal for equal resets: the system is down only when every replica resets at once,
for a fraction  min(1, [N r/τ − (N − 1)]_+)  of each cycle (n windows of [r − (N − 1)τ/N]_+ each), so N
replicas give full availability exactly when r ≤ (N − 1)τ/N. `availability` computes this exactly for any offsets,
and `simulate` checks it under jittered cycles (ChaosStream), as §9 item 4 asks: single, synchronized,
half-cycle, uniform and randomized schedules. Without coordination the offsets do not survive: free-running
replicas diffuse to independent phases, and every schedule converges to 1 − (r/τ)^N (measured 0.934 against
0.936 for N = 3, r = 0.4τ). Holding the phases is what the coordination cost buys.

The same criterion sizes a Jev swarm (Level 1: independent agents). An agent succeeds with probability p and
a solved task is worth V agent-costs c. The n-th agent adds p(1 − p)^{n−1} V to the expected value, so the
largest worthwhile swarm is

    N* = ⌈ ln(c / (pV)) / ln(1 − p) ⌉    (c < pV, 0 < p < 1; N* = 1 otherwise),

capped by what the free-energy budget can pay for. Launching agent k after a delay kΔ lets an early win
cancel agents that have not started: `schedule` gives the exact expected cost and latency for deterministic
run time τ, and `best_stagger` picks Δ for a stated value of waiting.

Standard library only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .chaos import ChaosStream

SCHEMES = ("single", "sync", "paper", "uniform", "random")


def offsets(scheme: str, n: int, period: float = 1.0, seed: float = 0.4123) -> list[float]:
    if scheme == "single":
        return [0.0]
    if scheme == "sync":
        return [0.0] * n
    if scheme == "paper":                       # T_k(t) = T(t − (k − 1)τ/2)
        return [((k * period) / 2.0) % period for k in range(n)]
    if scheme == "uniform":
        return [k * period / n for k in range(n)]
    if scheme == "random":
        rng = ChaosStream(seed)
        return [rng.uniform() * period for _ in range(n)]
    raise ValueError(f"scheme must be one of {SCHEMES}")


def _arcs(offs: Sequence[float], period: float, reset: float) -> list[list[tuple[float, float]]]:
    arcs = []
    for o in offs:
        a = o % period
        b = a + reset
        arcs.append([(a, b)] if b <= period else [(a, period), (0.0, b - period)])
    return arcs


def availability(offs: Sequence[float], period: float, reset: float) -> float:
    """Exact fraction of a cycle during which at least one replica is up (each is down `reset` per cycle)."""
    if reset <= 0:
        return 1.0
    if reset >= period or not offs:
        return 0.0
    arcs = _arcs(offs, period, reset)
    cuts = sorted({0.0, period, *(x for arc in arcs for seg in arc for x in seg)})
    down = 0.0
    for lo, hi in zip(cuts, cuts[1:]):
        mid = 0.5 * (lo + hi)
        if all(any(a <= mid < b for a, b in arc) for arc in arcs):
            down += hi - lo
    return 1.0 - down / period


def uniform_availability(n: int, period: float, reset: float) -> float:
    """Closed form for uniformly spread phases: 1 − min(1, [n r/τ − (n − 1)]_+).

    With spacing s = τ/n, a point x ∈ [0, s) is inside every reset window iff x + (n − 1)s < r; the n
    translates of that set give the total."""
    return 1.0 - min(1.0, max(0.0, n * reset / period - (n - 1)))


@dataclass(frozen=True)
class ReplicaPlan:
    n: int
    scheme: str
    availability: float
    net_value: float                 # V·availability − n·c_rep − c_coord·n(n−1)/2

    def to_dict(self) -> dict:
        return asdict(self)


def net_value(n: int, avail: float, value: float, c_rep: float, c_coord: float) -> float:
    return value * avail - n * c_rep - c_coord * n * (n - 1) / 2.0


def best_replication(period: float, reset: float, value: float, c_rep: float, c_coord: float = 0.0,
                     max_n: int = 8, scheme: str = "uniform") -> ReplicaPlan:
    """The replica count whose availability gain exceeds replication plus pairwise coordination cost."""
    best = None
    for n in range(1, max_n + 1):
        av = availability(offsets(scheme, n, period), period, reset)
        plan = ReplicaPlan(n, scheme, av, net_value(n, av, value, c_rep, c_coord))
        if best is None or plan.net_value > best.net_value + 1e-12:
            best = plan
    return best


@dataclass(frozen=True)
class ReplicaSim:
    scheme: str
    n: int
    availability: float
    continuity: float                # share of reset events during which some other replica stayed up

    def to_dict(self) -> dict:
        return asdict(self)


def free_running_availability(n: int, period: float, reset: float) -> float:
    """Replicas whose phases are independent and uniform: 1 − (r/τ)^n, whatever the initial offsets."""
    return 1.0 - min(1.0, reset / period) ** n


def simulate(scheme: str, n: int, period: float = 1.0, reset: float = 0.3, cycles: int = 400,
             jitter: float = 0.1, seed: float = 0.4123, phase_lock: bool = True) -> ReplicaSim:
    """Replicas with jittered resets over `cycles` nominal cycles.

    phase_lock=True: a coordinator re-anchors every reset to its nominal phase o + kτ (jitter does not
    accumulate); that coordination is what the c_coord term pays for. phase_lock=False: each replica free-runs
    and its phase diffuses, so every schedule drifts to `free_running_availability`.
    """
    rng = ChaosStream((seed + 0.017) % 1.0)
    offs = offsets(scheme, n, period, seed)
    horizon = cycles * period
    intervals: list[list[tuple[float, float]]] = []
    for o in offs:
        t, k, mine = o, 0, []
        while t < horizon:
            r = min(period, max(0.0, reset * (1.0 + jitter * rng.normal())))
            mine.append((t, min(horizon, t + r)))
            k += 1
            if phase_lock:
                nxt = o + k * period + jitter * period * 0.5 * rng.normal()
            else:
                nxt = t + period * (1.0 + jitter * rng.normal())
            t = max(t + r + 1e-9, nxt)
        intervals.append(mine)
    events = sorted([(a, 1) for iv in intervals for a, _ in iv] + [(b, -1) for iv in intervals for _, b in iv])
    k, last, down, alldown = 0, 0.0, 0.0, []
    for x, dk in events:
        if k == len(offs) and x > last:
            down += x - last
            alldown.append((last, x))
        k += dk
        last = x
    hits = total = 0
    j = 0
    flat = sorted((a, b) for iv in intervals for a, b in iv)
    for a, b in flat:
        total += 1
        while j < len(alldown) and alldown[j][1] <= a:
            j += 1
        if j < len(alldown) and alldown[j][0] < b:
            hits += 1
    return ReplicaSim(scheme, len(offs), 1.0 - down / horizon, 1.0 - hits / total if total else 1.0)


# ---------------------------------------------------------------------- swarm sizing and staggering


def marginal_gain(p: float, n: int, value: float) -> float:
    """Expected value the n-th independent agent adds: p (1 − p)^{n−1} V."""
    return p * (1.0 - p) ** (n - 1) * value


def swarm_size(p: float, value_ratio: float, max_n: int = 8, affordable: int | None = None) -> int:
    """N* = ⌈ln(c/(pV)) / ln(1 − p)⌉ with V/c = value_ratio, clamped to [1, min(max_n, affordable)]."""
    cap = max(1, min(max_n, affordable if affordable is not None else max_n))
    if not 0 < p < 1 or value_ratio * p <= 1.0:
        return 1
    x = math.log(1.0 / (p * value_ratio)) / math.log(1.0 - p)
    return max(1, min(cap, math.ceil(x - 1e-12)))


@dataclass(frozen=True)
class Schedule:
    p: float
    n: int
    tau: float
    delta: float
    success: float                   # 1 − (1 − p)^n
    cost: float                      # expected agent-time spent
    latency: float                   # expected time until the first win, or until every agent has failed
    spawned: float                   # expected number of agents that start

    def to_dict(self) -> dict:
        return asdict(self)


def schedule(p: float, n: int, tau: float, delta: float) -> Schedule:
    """Exact expectations when agent k starts at kΔ, runs τ, wins w.p. p, and a win cancels the rest."""
    cost = latency = spawned = 0.0
    for k in range(n):
        pk = p * (1.0 - p) ** k
        t_win = k * delta + tau
        started = [j for j in range(n) if j * delta < t_win - 1e-12]
        cost += pk * sum(min(tau, t_win - j * delta) for j in started)
        latency += pk * t_win
        spawned += pk * len(started)
    none = (1.0 - p) ** n
    cost += none * n * tau
    latency += none * ((n - 1) * delta + tau)
    spawned += none * n
    return Schedule(p, n, tau, delta, 1.0 - none, cost, latency, spawned)


def best_stagger(p: float, n: int, tau: float, latency_weight: float = 2.0, steps: int = 40) -> Schedule:
    """Δ ∈ [0, τ] minimizing E[cost] + w·E[latency] (agent-time; w = worth of one unit of waiting)."""
    best = None
    for i in range(steps + 1):
        sc = schedule(p, n, tau, tau * i / steps)
        if best is None or sc.cost + latency_weight * sc.latency < best.cost + latency_weight * best.latency - 1e-12:
            best = sc
    return best

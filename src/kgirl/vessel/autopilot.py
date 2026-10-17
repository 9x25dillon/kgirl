"""Autopilot: the vessel lifecycle run unattended, with a recursive-self-improvement (RSI) slider.

The slider sets how much the vessel may change itself without a person in the loop. Levels are cumulative:
each one adds a permission to everything below it.

    RSI 0  OBSERVE      measure drift every cycle, record it, and change nothing
    RSI 1  STEER        + steer back toward the anchor when drift ≥ engage (gated, rolled back on failure)
    RSI 2  LEARN        + learn from the supplied stream every cycle, under the identity anchor
    RSI 3  CONSOLIDATE  + move the anchor toward the present self after `stable_cycles` coherent cycles
    RSI 4  SELF-TUNE    + improve its own steering controller. Sandbox trials replay the vessel's own recent
                          drifted states and pick a faster repair configuration. It is adopted only if it is
                          faster on every held-out trial *and* restores an identity at least as faithful.
                          This is the recursive step: the vessel improves how it improves.

Invariants at every level:
  * Every cycle and every change is a ledger event; weight snapshots are content-addressed.
  * Circuit breaker: from level 1 up, if drift after a cycle's repair still exceeds `breaker`, the vessel
    restores its last anchor snapshot and demotes itself to RSI 1 (or stays there). The ledger records it, and
    only a person raises the level again (`set_level`). Restoring the anchor is a repair, so it sits within
    RSI 1's permission. The breaker was first armed only from RSI 2, and a measured failure moved it down:
    with too few steering probes, RSI 1 repairs fell short (drift 0.20, then 0.29) and every cycle stacked on
    the last, a ratchet that nothing stopped.
  * Consolidation never runs from an incoherent state (enforced in Vessel.consolidate).
  * Self-tuning sandboxes run on checkpoints of the vessel's own substrate and are always restored. The
    ledger records every trial, and a candidate must win on every held-out trial, not on average.
  * Speed never buys fidelity. A candidate's restored voice KL may not exceed the incumbent's by more than
    `tune_kl_tolerance`, and its coherence may not be lower. Without this rule the measured result was Goodhart:
    every adopted controller lowered λ_voice, the held-out persona voice loss rose from 0.385 to 0.425, and
    the last repair stalled at drift 0.146.
  * Watchdog: if a live repair fails after a tune, the previous controller is reinstated
    (`autopilot.tune.revert`).
  * Trials replay the vessel's own recent drifted states, saved just before each live repair, rather than
    synthetic noise when it has them. A controller tuned on weight noise alone was measured to stall on the
    world's real drift.
  * The world can push the vessel (`disturb`). The autopilot never acts on the world.

Steering runs with fresh optimizer moments (SteerConfig.fresh_optimizer). Measured over 6 disturbed cycles
at RSI 3: when steering shared Adam's moments with the world's and the learner's updates, a repair cost 130,
then 240, 340, 550, 580 and 730 steps, because inherited second moments shrank every step. With fresh
moments it cost 50–140 steps, drift was held the same, and the run took 16 s instead of 47 s.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum
from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np

from ..evolve.chaos import ChaosRNG
from .steer import SteerConfig, steer


class RSI(IntEnum):
    OBSERVE = 0
    STEER = 1
    LEARN = 2
    CONSOLIDATE = 3
    SELF_TUNE = 4


DESCRIPTIONS = {
    RSI.OBSERVE: "observe and record; never changes weights",
    RSI.STEER: "steer back to coherence when drift crosses engage",
    RSI.LEARN: "learn from the stream under the identity anchor",
    RSI.CONSOLIDATE: "grow the anchor from sustained coherence",
    RSI.SELF_TUNE: "tune its own steering controller from sandbox trials",
}


@dataclass(frozen=True)
class AutopilotConfig:
    level: RSI = RSI.STEER
    learn_steps: int = 40              # anchored steps per cycle at RSI ≥ 2
    learn_lr: float = 3e-3
    stable_cycles: int = 2             # coherent cycles before consolidating (RSI ≥ 3)
    beta: float = 0.1                  # consolidation rate
    breaker: float = 0.25              # drift after repair that trips the circuit breaker (RSI ≥ 1)
    tune_every: int = 2                # cycles between self-tuning rounds (RSI 4)
    tune_candidates: int = 6
    tune_replay: int = 2               # recent drifted states replayed as held-out trials
    tune_perturbation: float = 0.5     # fallback: relative weight noise (drift ≈ 0.44 measured)
    tune_seeds: tuple = (0.61, 0.83)   # fallback noise seeds when fewer drifted states are on hand
    tune_kl_tolerance: float = 0.05    # restored voice KL may exceed the incumbent's by at most 5 %
    tune_margin: float = 0.15          # required relative speed-up in steps to coherence
    tune_budget: int = 300

    def __post_init__(self):
        object.__setattr__(self, "level", RSI(int(self.level)))
        if self.learn_steps < 0 or self.stable_cycles < 1 or not 0 < self.beta <= 1 or not 0 < self.breaker <= 1:
            raise ValueError("invalid autopilot configuration")


@dataclass
class CycleRecord:
    cycle: int
    level: int
    drift_in: float
    drift_out: float
    learned: int = 0
    steered: Optional[str] = None
    consolidated: bool = False
    tuned: Optional[dict] = None
    breaker: bool = False

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class Autopilot:
    vessel: Any
    cfg: AutopilotConfig = AutopilotConfig()
    steer_cfg: SteerConfig = SteerConfig(fresh_optimizer=True)
    stream: Optional[Iterable[Any]] = None
    guard: Optional[Callable[[], float]] = None
    history: list = field(default_factory=list)

    def __post_init__(self):
        self._stream: Optional[Iterator] = iter(self.stream) if self.stream is not None else None
        self._coherent_run = 0
        self._episodes: list = []          # recent pre-repair weights: the vessel's own experience of drift
        self._previous: Optional[SteerConfig] = None
        self.vessel.db.append(self.vessel.name, "autopilot.start", {"level": int(self.cfg.level),
                                                                    "meaning": DESCRIPTIONS[self.cfg.level],
                                                                    "steer": asdict(self.steer_cfg)})

    # ------------------------------------------------------------------------------------------- control

    @property
    def level(self) -> RSI:
        return self.cfg.level

    def set_level(self, level: int, reason: str = "operator") -> None:
        new = RSI(int(level))
        self.vessel.db.append(self.vessel.name, "autopilot.level",
                              {"from": int(self.cfg.level), "to": int(new), "reason": reason})
        self.cfg = replace(self.cfg, level=new)

    # ------------------------------------------------------------------------------------------- loop

    def run(self, cycles: int, disturb: Optional[Callable[[int], None]] = None) -> list:
        for _ in range(cycles):
            self.cycle(disturb)
        return self.history

    def cycle(self, disturb: Optional[Callable[[int], None]] = None) -> CycleRecord:
        v, cfg = self.vessel, self.cfg
        k = len(self.history)
        if disturb is not None:
            disturb(k)                                   # the world acts; the autopilot only responds
        rec = CycleRecord(k, int(cfg.level), v.drift(), float("nan"))

        if cfg.level >= RSI.LEARN and self._stream is not None and cfg.learn_steps:
            batches = [b for _, b in zip(range(cfg.learn_steps), self._stream)]
            if batches:
                v.learn(batches, lr=cfg.learn_lr, anchored=True, note=f"autopilot cycle {k}")
                rec.learned = len(batches)

        if cfg.level >= RSI.STEER and v.drift() >= self.steer_cfg.engage:
            if cfg.level >= RSI.SELF_TUNE:
                self._episodes = (self._episodes + [v.substrate.parameters()])[-cfg.tune_replay:]
            rec.steered = v.steer(self.steer_cfg, guard=self.guard).outcome
            if rec.steered != "coherent" and self._previous is not None:
                self.vessel.db.append(v.name, "autopilot.tune.revert", {
                    "outcome": rec.steered, "reverted": _knobs(self.steer_cfg), "restored": _knobs(self._previous)})
                self.steer_cfg, self._previous = self._previous, None

        d = v.drift()
        if cfg.level >= RSI.STEER and d > cfg.breaker:
            self._trip(d)
            rec.breaker = True
            d = v.drift()

        self._coherent_run = self._coherent_run + 1 if d <= self.steer_cfg.release else 0
        if self.cfg.level >= RSI.CONSOLIDATE and self._coherent_run >= cfg.stable_cycles:
            v.consolidate(cfg.beta, self.steer_cfg)
            rec.consolidated = True
            self._coherent_run = 0
            d = v.drift()

        if self.cfg.level >= RSI.SELF_TUNE and (k + 1) % cfg.tune_every == 0:
            rec.tuned = self.self_tune()

        rec.drift_out = d
        self.history.append(rec)
        v.db.append(v.name, "autopilot.cycle", rec.as_dict())
        return rec

    def _trip(self, drift: float) -> None:
        label = [x for x in self.vessel.db.labels(self.vessel.name) if x.startswith(("anchor:", "migrate:"))][-1]
        self.vessel.rollback(label)
        self.vessel.db.append(self.vessel.name, "autopilot.breaker",
                              {"drift": drift, "breaker": self.cfg.breaker, "restored": label})
        self.set_level(min(int(self.cfg.level), int(RSI.STEER)), reason=f"circuit breaker at drift {drift:.3f}")

    # ------------------------------------------------------------------------------------------- recursion

    def trials(self) -> list:
        """Held-out start states: the vessel's own recent drifted weights, else chaos weight noise."""
        if len(self._episodes) >= self.cfg.tune_replay:
            return [("replay", p) for p in self._episodes[-self.cfg.tune_replay:]]
        return [("noise", sd) for sd in self.cfg.tune_seeds]

    def trial(self, scfg: SteerConfig, start: tuple) -> dict:
        """Sandbox: put the substrate in a drifted state, steer with `scfg`, measure, restore. Leaves no trace."""
        sub = self.vessel.substrate
        state = sub.checkpoint()
        try:
            kind, src = start
            if kind == "replay":
                sub.load_parameters(src)
            else:
                rng = ChaosRNG(src)
                eps = self.cfg.tune_perturbation
                sub.load_parameters({k: w + eps * float(w.std()) * rng.normal(w.shape)
                                     for k, w in sub.parameters().items()})
            d0 = self.vessel.drift()
            res = steer(sub, audit=self.vessel.drift,
                        objective=lambda lr: self.vessel.objective(lr, lam_rep=scfg.lam_rep, lam_voice=scfg.lam_voice,
                                                                   task_weight=0.0),
                        cfg=replace(scfg, max_steps=self.cfg.tune_budget), force=True)
            c = self.vessel.read(body=False).comparison
            return {"start": d0, "end": res.drift_end, "steps": res.steps, "coherent": res.outcome == "coherent",
                    "kl": c.kl, "coherence": c.coherence}
        finally:
            sub.restore(state)

    def candidates(self, round_: int) -> list:
        """Chaos-seeded log-normal variations of the incumbent controller (deterministic per round)."""
        rng = ChaosRNG((0.2718 + 0.1618 * round_) % 1.0)
        base = self.steer_cfg
        out = []
        for _ in range(self.cfg.tune_candidates):
            z = rng.normal(4) * 0.5
            lr_max = float(np.clip(base.lr_max * math.exp(z[0]), base.lr, 0.2))
            out.append(replace(base, lr_max=lr_max,
                               lam_voice=float(np.clip(base.lam_voice * math.exp(z[1]), 0.1, 10.0)),
                               lam_rep=float(np.clip(base.lam_rep * math.exp(z[2]), 0.1, 10.0)),
                               block=int(np.clip(round(base.block * math.exp(0.5 * z[3])), 4, 40))))
        return out

    def self_tune(self) -> dict:
        """One round of recursive self-improvement on the steering controller.

        Speed score = steps to coherence (budget + 1 if never coherent). A candidate replaces the incumbent only
        if, on *every* held-out trial, it is coherent and at least `tune_margin` faster, and if its restored
        identity is at least as faithful: mean voice KL ≤ incumbent × (1 + tune_kl_tolerance), mean coherence
        ≥ incumbent's."""
        round_ = len(self.vessel.db.events(self.vessel.name, "autopilot.tune"))
        starts = self.trials()
        budget = self.cfg.tune_budget

        def speed(t):
            return t["steps"] if t["coherent"] else budget + 1

        inc = [self.trial(self.steer_cfg, st) for st in starts]
        inc_speed = [speed(t) for t in inc]
        inc_kl = float(np.mean([t["kl"] for t in inc]))
        inc_coh = float(np.mean([t["coherence"] for t in inc]))
        best, best_speed, tried = None, inc_speed, []
        for c in self.candidates(round_):
            ts = [self.trial(c, st) for st in starts]
            sp = [speed(t) for t in ts]
            kl, coh = float(np.mean([t["kl"] for t in ts])), float(np.mean([t["coherence"] for t in ts]))
            faster = all(t["coherent"] and a <= b * (1 - self.cfg.tune_margin) for t, a, b in zip(ts, sp, inc_speed))
            faithful = kl <= inc_kl * (1 + self.cfg.tune_kl_tolerance) and coh >= inc_coh - 1e-9
            tried.append({**_knobs(c), "steps": sp, "kl": round(kl, 5), "coherence": round(coh, 5),
                          "faster": faster, "faithful": faithful})
            if faster and faithful and sum(sp) < sum(best_speed):
                best, best_speed = c, sp
        adopted = best is not None
        payload = {"round": round_, "trials": [k for k, _ in starts], "incumbent": _knobs(self.steer_cfg),
                   "incumbent_steps": inc_speed, "incumbent_kl": round(inc_kl, 5),
                   "incumbent_coherence": round(inc_coh, 5), "candidates": tried, "adopted": adopted}
        if adopted:
            payload["new"], payload["new_steps"] = _knobs(best), best_speed
            self._previous, self.steer_cfg = self.steer_cfg, best
        self.vessel.db.append(self.vessel.name, "autopilot.tune", payload)
        return {"adopted": adopted, "trials": payload["trials"], "incumbent_steps": inc_speed, "best_steps": best_speed}

    def summary(self) -> dict:
        h = self.history
        return {"level": int(self.cfg.level), "meaning": DESCRIPTIONS[self.cfg.level], "cycles": len(h),
                "drift_out": [round(r.drift_out, 4) for r in h],
                "steered": sum(1 for r in h if r.steered), "learned_steps": sum(r.learned for r in h),
                "consolidations": sum(r.consolidated for r in h),
                "tunes_adopted": sum(1 for r in h if r.tuned and r.tuned["adopted"]),
                "breakers": sum(r.breaker for r in h),
                "reverts": len(self.vessel.db.events(self.vessel.name, "autopilot.tune.revert")),
                "controller": _knobs(self.steer_cfg)}


def _knobs(c: SteerConfig) -> dict:
    return {"lr_max": round(c.lr_max, 6), "lam_voice": round(c.lam_voice, 4), "lam_rep": round(c.lam_rep, 4),
            "block": c.block}

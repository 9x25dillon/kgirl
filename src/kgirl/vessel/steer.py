"""The coherence controller: detect drift, steer the weights back, and make every edit reversible.

State machine (a Schmitt trigger, so the controller does not chatter at the threshold):

        ┌──────────── D ≥ engage ────────────┐
        │                                    ▼
     [HOLD] ◀──── D ≤ release ──────── [STEER] ──┐ block of k gradient steps on
        ▲                                    │   │ λ_rep·(1−CKA) + λ_voice·KL (+ task, + λ_w·EWC)
        │                                    ▼   │
        │                              audit on held-out probes
        │                         accept ◀───┴───▶ reject
        │                    (D fell, guard ok)   (restore checkpoint, lr ← lr/2)
        │                    lr ← min(lr·1.25, lr_max)      lr < lr_min → STALLED
        └──────────── max_steps ─────────── [SPENT]

Two rules keep the controller honest:
  * Steering optimizes the *steer* probes; acceptance reads the *audit* probes, which the gradient never sees.
    A step that only memorizes the mirror does not lower audit drift and is rolled back.
  * An optional guard (a held-out loss of a skill to protect) must stay within `guard_tolerance` of its value
    when steering began, so small per-block losses cannot compound; otherwise the block is rolled back. Steering may cost plasticity. The trace
    reports that cost and does not hide it.

Default thresholds are calibrated, not arbitrary: on the reference substrate two sibling vessels grown from one
base sit ≈ 0.3 apart in identity distance, and an anchored learner stays ≈ 0.07 from its anchor. `release` at
0.10 keeps a steered vessel well inside a third of the distance to its nearest sibling (see `recognize`). For a
new substrate, measure sibling distances first and set release ≤ ⅓ of the smallest.

Complexity per block: k·(forward+backward on batch and probes) + one audit forward; rollback is O(|θ|) copy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

from .substrate import Objective


@dataclass(frozen=True)
class SteerConfig:
    engage: float = 0.15
    release: float = 0.10
    lr: float = 3e-3
    lr_max: float = 3e-2
    lr_min: float = 5e-5
    lam_rep: float = 1.0
    lam_voice: float = 1.0
    lam_w: float = 0.0
    block: int = 10
    max_steps: int = 1200
    guard_tolerance: float = 0.10
    min_gain: float = 1e-5
    fresh_optimizer: bool = False      # reset Adam's moments before steering (substrates with reset_optimizer)

    def __post_init__(self):
        if not 0 <= self.release < self.engage <= 1:
            raise ValueError("need 0 ≤ release < engage ≤ 1 (hysteresis band)")
        if not 0 < self.lr_min <= self.lr <= self.lr_max or self.block < 1 or self.max_steps < 1:
            raise ValueError("need 0 < lr_min ≤ lr ≤ lr_max and positive block, max_steps")


@dataclass(frozen=True)
class BlockRecord:
    step: int
    drift_before: float
    drift_after: float
    guard_before: Optional[float]
    guard_after: Optional[float]
    lr: float
    accepted: bool
    reason: str


@dataclass
class SteerResult:
    outcome: str                      # within_band | coherent | stalled | spent
    drift_start: float
    drift_end: float
    steps: int
    accepted: int
    rejected: int
    guard_start: Optional[float] = None
    guard_end: Optional[float] = None
    trace: list = field(default_factory=list)

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "trace"}
        d["blocks"] = len(self.trace)
        return d


def steer(substrate, *, audit: Callable[[], float], objective: Callable[[float], Objective],
          batches: Optional[Iterable[Any]] = None, guard: Optional[Callable[[], float]] = None,
          cfg: SteerConfig = SteerConfig(), force: bool = False,
          on_block: Optional[Callable[[BlockRecord], None]] = None) -> SteerResult:
    """Run the controller. `audit()` returns current drift on the audit probes; `objective(lr)` builds the
    steering objective; `batches` (optional) supplies task data so steering can keep learning while it corrects."""
    d0 = audit()
    g0 = guard() if guard else None
    if d0 < cfg.engage and not force:
        return SteerResult("within_band", d0, d0, 0, 0, 0, g0, g0)
    stream: Optional[Iterator] = iter(batches) if batches is not None else None
    if cfg.fresh_optimizer and hasattr(substrate, "reset_optimizer"):
        substrate.reset_optimizer()
    lr, steps, acc, rej = cfg.lr, 0, 0, 0
    d, g = d0, g0
    trace: list[BlockRecord] = []
    outcome = "spent"
    while steps < cfg.max_steps:
        state = substrate.checkpoint()
        obj = objective(lr)
        n = min(cfg.block, cfg.max_steps - steps)
        for _ in range(n):
            batch = next(stream, None) if stream is not None else None
            if batch is None:
                obj.task_weight = 0.0
            substrate.step(batch, obj)
        steps += n
        d1 = audit()
        g1 = guard() if guard else None
        guard_ok = g0 is None or g1 is None or g1 <= g0 * (1.0 + cfg.guard_tolerance)
        improved = d1 < d - cfg.min_gain
        if improved and guard_ok:
            rec = BlockRecord(steps, d, d1, g, g1, lr, True, "drift fell")
            d, g, acc = d1, g1, acc + 1
            lr = min(lr * 1.25, cfg.lr_max)
        else:
            substrate.restore(state)
            rec = BlockRecord(steps, d, d1, g, g1, lr, False, "guard" if improved else "no gain")
            rej += 1
            lr *= 0.5
        trace.append(rec)
        if on_block:
            on_block(rec)
        if d <= cfg.release:
            outcome = "coherent"
            break
        if lr < cfg.lr_min:
            outcome = "stalled"
            break
    return SteerResult(outcome, d0, d, steps, acc, rej, g0, g, trace)

"""MAP-Elites with lineage: quality-diversity search whose emergence you can trace.

Mouret & Clune (2015) MAP-Elites keeps, for every cell of a behavior grid, the
best genome found so far. Variation is Iso+LineDD (Vassiliades & Mouret 2018):

    x' = x_i + σ_iso · N(0, I) + σ_line · N(0, 1) · (x_j − x_i)      (x_i, x_j two random elites)

Emergence is measured, not asserted:
  * coverage      fraction of behavior cells holding an elite
  * QD-score      sum of elite fitness (rewards both quality and diversity)
  * innovations   new cell discovered vs elite improved, per generation
  * stepping stones  distinct cells visited along the champion's ancestry — how much
                  the best solution depended on detours through *other* niches

Evaluation is batched: `evaluate(X) -> list[Evaluation]` gets an (n, d) unit-cube
array, so problems can vectorize a whole generation (see problems/assay.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .chaos import ChaosRNG
from .space import Space


@dataclass(frozen=True)
class Descriptor:
    name: str
    lo: float
    hi: float
    bins: int

    def cell(self, v: float) -> int:
        if not np.isfinite(v):
            return 0
        k = int((v - self.lo) / (self.hi - self.lo) * self.bins)
        return min(max(k, 0), self.bins - 1)


@dataclass(frozen=True)
class Evaluation:
    fitness: float
    behavior: tuple[float, ...]
    info: dict = field(default_factory=dict)


@dataclass
class Elite:
    id: int
    x: np.ndarray
    fitness: float
    behavior: tuple[float, ...]
    info: dict
    generation: int
    parents: tuple[int, ...]
    cell: tuple[int, ...]


@dataclass
class GenStats:
    generation: int
    coverage: float
    qd_score: float
    best: float
    new_cells: int
    improvements: int


Evaluator = Callable[[np.ndarray], list[Evaluation]]


class MapElites:
    def __init__(self, space: Space, evaluate: Evaluator, descriptors: list[Descriptor], seed: float = 0.4123,
                 batch: int = 32, sigma_iso: float = 0.05, sigma_line: float = 0.3, init: int = 64):
        self.space, self.evaluate, self.descriptors = space, evaluate, descriptors
        self.rng = ChaosRNG(seed)
        self.batch, self.sigma_iso, self.sigma_line, self.init = batch, sigma_iso, sigma_line, init
        self.archive: dict[tuple[int, ...], Elite] = {}
        self.genealogy: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}   # id -> (parents, cell)
        self.history: list[GenStats] = []
        self.generation = 0
        self._next_id = 0

    @property
    def n_cells(self) -> int:
        return int(np.prod([d.bins for d in self.descriptors]))

    def _cell(self, behavior) -> tuple[int, ...]:
        return tuple(d.cell(v) for d, v in zip(self.descriptors, behavior))

    def _variation(self, n: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
        elites = list(self.archive.values())
        a = self.rng.integers(len(elites), n)
        b = self.rng.integers(len(elites), n)
        Xa = np.stack([elites[i].x for i in a])
        Xb = np.stack([elites[i].x for i in b])
        X = Xa + self.sigma_iso * self.rng.normal((n, self.space.dim)) \
            + self.sigma_line * self.rng.normal((n, 1)) * (Xb - Xa)
        return np.clip(X, 0.0, 1.0), [(elites[i].id, elites[j].id) for i, j in zip(a, b)]

    def step(self) -> GenStats:
        if not self.archive:
            X = self.rng.uniform((self.init, self.space.dim))
            parents = [()] * len(X)
        else:
            X, parents = self._variation(self.batch)
        evals = self.evaluate(X)
        new_cells = improvements = 0
        for x, ev, par in zip(X, evals, parents):
            if not np.isfinite(ev.fitness):
                continue
            c = self._cell(ev.behavior)
            cur = self.archive.get(c)
            if cur is None or ev.fitness > cur.fitness:
                eid = self._next_id
                self._next_id += 1
                self.archive[c] = Elite(eid, x.copy(), float(ev.fitness), tuple(map(float, ev.behavior)),
                                        ev.info, self.generation, par, c)
                self.genealogy[eid] = (par, c)
                if cur is None:
                    new_cells += 1
                else:
                    improvements += 1
        fit = [e.fitness for e in self.archive.values()]
        st = GenStats(self.generation, len(self.archive) / self.n_cells, float(np.sum(fit)) if fit else 0.0,
                      float(np.max(fit)) if fit else float("nan"), new_cells, improvements)
        self.history.append(st)
        self.generation += 1
        return st

    def run(self, generations: int) -> "MapElites":
        for _ in range(generations):
            self.step()
        return self

    # ------------------------------------------------------------------ analysis

    def champion(self) -> Elite:
        return max(self.archive.values(), key=lambda e: e.fitness)

    def ancestry(self, eid: int) -> list[int]:
        """Breadth-first ancestry of an elite (ids), oldest last."""
        seen, queue, order = {eid}, [eid], []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for p in self.genealogy.get(cur, ((), ()))[0]:
                if p not in seen:
                    seen.add(p)
                    queue.append(p)
        return order

    def stepping_stones(self, eid: int | None = None) -> int:
        eid = self.champion().id if eid is None else eid
        return len({self.genealogy[a][1] for a in self.ancestry(eid) if a in self.genealogy})

    def elites(self) -> list[Elite]:
        return sorted(self.archive.values(), key=lambda e: -e.fitness)

    def grid(self, key: str | None = None) -> np.ndarray:
        """Archive as a dense array (fitness, or an info field), NaN where empty."""
        shape = tuple(d.bins for d in self.descriptors)
        g = np.full(shape, np.nan)
        for c, e in self.archive.items():
            g[c] = e.fitness if key is None else e.info.get(key, np.nan)
        return g

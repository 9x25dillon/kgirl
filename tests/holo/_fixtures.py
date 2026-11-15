"""A synthetic chunk index and KGVM matrix for the holo tests: chaos vectors, no model, no network.

24 chunks: ids 1..24, the first 14 in repository `alpha`, the other 10 in `beta`; symbol kinds cycle
function, class, preamble. Each repository's vectors share an offset, so the two form separate clusters.
The matrix also holds one vector (id 999) with no chunk row: an orphan.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

import tests.holo  # noqa: F401  (sys.path)
from kgirl.evolve.chaos import ChaosRNG
from kgirl.harness.atlas.chunks import SCHEMA
from kgirl.harness.atlas.vectors import write_matrix

DIM, N = 32, 24
ALPHA = 14                                   # chunks 1..14 are alpha's
KINDS = ("function", "class", "preamble")
ORPHAN = 999
MODEL = "synthetic"
ENTITIES = ["kind:class", "kind:function", "kind:preamble", "repo:alpha", "repo:beta"]

# (repository, kind) cells hold 5, 5, 4 chunks in alpha and 3, 3, 4 in beta.
SAMPLED_AT_3 = 18                            # per_cell=3 keeps 3 of each of the six cells
RECORDS_AT_3 = 2 * len(ENTITIES) + 2 * SAMPLED_AT_3      # centroid + spread each, every vector under two owners
RECORDS_ALL = 2 * len(ENTITIES) + 2 * N


def repo_of(i: int) -> str:
    return "alpha" if i < ALPHA else "beta"


def kind_of(i: int) -> str:
    return KINDS[i % 3]


def paths(home: Path) -> tuple[Path, Path]:
    out = home / "chunks" / "all"
    return out / "all-repos.db", out / "all-repos.kgvm"


def make_index(home: Path, seed: float = 0.4123) -> np.ndarray:
    """Write the index under `home` where `build(home)` looks for it. Returns the (N, DIM) vectors as drawn."""
    db, kgvm = paths(home)
    db.parent.mkdir(parents=True, exist_ok=True)
    rng = ChaosRNG(seed=seed)
    vecs = rng.normal((N, DIM))
    rows = []
    for i in range(N):
        lo = 0 if repo_of(i) == "alpha" else 8
        vecs[i, lo:lo + 8] += 2.0
        rows.append((i + 1, repo_of(i), f"src/m{i}.py", "python", kind_of(i), f"sym{i}", 1, 9, f"text {i}"))
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    write_matrix(kgvm, [row[0] for row in rows] + [ORPHAN], np.vstack([vecs, rng.normal((1, DIM))]), MODEL)
    return vecs


class FakeAtlas:
    """What the build needs of an Atlas: the indexed repositories and their heads. `gamma` has no chunks."""

    def repos(self) -> list[dict]:
        return [{"name": "alpha", "head": "a1"}, {"name": "beta", "head": "b2"}, {"name": "gamma", "head": "c3"}]

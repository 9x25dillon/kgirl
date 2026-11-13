"""Every random stream of the experiment, by purpose, with provenance.

Two generators behind one interface (`uniform(size)`, `normal(size)`):

    chaos   kgirl.evolve.chaos.ChaosRNG with the seeds written in the pre-registrations. The registered runs.
    pcg     numpy PCG64 from SeedSequence(20261112, spawn_key=(purpose, index, replicate)): statistically
            independent streams. The replication runs (Amendment 2b).

No model family draws its own data: training strings are drawn once per (ground truth, replicate) and
handed to every family. The only stream a family touches is `hmm_init`, the starting points of HMM2.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from . import host

KINDS = ("chaos", "pcg")
PCG_ENTROPY = 20261112
PURPOSES: Dict[str, int] = {"train": 1, "post": 2, "hmm_init": 3, "layered": 4, "surrogate": 5, "welch": 6,
                            "data_surrogate": 7}


def chaos_seed(purpose: str, index: int = 0, replicate: int = 0) -> Tuple[float, int]:
    """(seed, lanes) of the ChaosRNG stream for a purpose. These are the registered formulas."""
    if purpose == "train":                        # PREREGISTRATION.md §5
        return 0.1 + 0.01 * index + 0.001 * replicate, 64
    if purpose == "post":                         # interventions, same form, offset 0.6
        return 0.6 + 0.01 * index + 0.001 * replicate, 64
    if purpose == "hmm_init":                     # Amendment 1: four deterministic starts
        return (0.37 + 0.013 * replicate) % 1.0, 64
    if purpose == "layered":                      # PREREGISTRATION_PART2.md §3
        return 0.3 + 0.01 * index + 0.001 * replicate, 4096
    if purpose == "surrogate":                    # PREREGISTRATION_PART2.md §4
        return 0.45 + 0.01 * index + 0.001 * replicate, 4096
    if purpose == "welch":
        return 0.55 + 0.01 * index + 0.001 * replicate, 4096
    if purpose == "data_surrogate":
        return 0.731, 4096
    raise KeyError(f"unknown stream purpose {purpose!r}")


class Stream:
    """A source of uniforms and standard normals with a record of where it came from."""

    def __init__(self, kind: str, purpose: str, index: int = 0, replicate: int = 0):
        if kind not in KINDS:
            raise KeyError(f"unknown stream kind {kind!r}; choose from {KINDS}")
        if purpose not in PURPOSES:
            raise KeyError(f"unknown stream purpose {purpose!r}")
        self.kind, self.purpose, self.index, self.replicate = kind, purpose, index, replicate
        if kind == "chaos":
            seed, lanes = chaos_seed(purpose, index, replicate)
            self._rng = host.ChaosRNG(seed, lanes=lanes)
            self.provenance = {"generator": "kgirl.evolve.chaos.ChaosRNG", "seed": seed, "lanes": lanes}
        else:
            key = (PURPOSES[purpose], index, replicate)
            self._rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(PCG_ENTROPY, spawn_key=key)))
            self.provenance = {"generator": "numpy.random.PCG64", "entropy": PCG_ENTROPY, "spawn_key": list(key)}

    def uniform(self, size) -> np.ndarray:
        if self.kind == "chaos":
            return np.asarray(self._rng.uniform(size), dtype=float)
        return self._rng.random(size)

    def normal(self, size) -> np.ndarray:
        if self.kind == "chaos":
            return np.asarray(self._rng.normal(size), dtype=float)
        return self._rng.standard_normal(size)


def stream(kind: str, purpose: str, index: int = 0, replicate: int = 0) -> Stream:
    return Stream(kind, purpose, index, replicate)


def provenance(kind: str) -> Dict[str, object]:
    """What a results file records about its randomness."""
    base: Dict[str, object] = {"kind": kind, "numpy": np.__version__, "host_commit": host.host_commit(),
                               "host_source_untouched": host.host_source_untouched()}
    if kind == "chaos":
        base["generator"] = "kgirl.evolve.chaos.ChaosRNG"
        base["seeds"] = {p: "see streams.chaos_seed" for p in PURPOSES}
        base["examples"] = {p: list(chaos_seed(p, 0, 0)) for p in PURPOSES}
    else:
        base["generator"] = "numpy.random.PCG64"
        base["seeds"] = f"SeedSequence({PCG_ENTROPY}, spawn_key=(purpose, index, replicate))"
        base["purposes"] = dict(PURPOSES)
    return base

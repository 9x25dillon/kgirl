"""Deterministic text corpora, byte windows, batches and probe sets for the reference substrate.

Every choice is drawn from ChaosRNG, so a corpus is a pure function of its seed. Two registers are built in:

    prose(seed, n)   templated declarative sentences: the "native" register a vessel is born in
    code(seed, n)    small Python-like lines: a foreign register that pulls a vessel away from its anchor
    voice(genome, n) prose re-weighted by a persona genome (lexical preferences, cadence, signature phrase)

Probe sets are fixed windows over a neutral mix of both registers. Identity is measured on probes and never on
training batches, so measurement cannot be optimized away by memorising what it looks at.
"""

from __future__ import annotations

from typing import Iterator, Mapping

import numpy as np

from ..evolve.chaos import ChaosRNG

ADJ = ("quiet", "bright", "hollow", "golden", "recursive", "patient", "folded", "silver", "open", "deep",
       "small", "careful", "ancient", "luminous", "steady", "strange")
NOUN = ("lattice", "river", "signal", "garden", "mirror", "spiral", "vessel", "lantern", "field", "pattern",
        "window", "seed", "orbit", "archive", "bridge", "voice")
VERB = ("holds", "folds", "carries", "measures", "remembers", "follows", "returns", "opens", "mends", "keeps",
        "finds", "sings")
PREP = ("inside", "beyond", "under", "beside", "through", "toward")
NAMES = ("x", "y", "acc", "node", "state", "phase", "w", "k")
OPS = ("+", "-", "*", "//", "%")


def _pick(rng: ChaosRNG, seq, weights=None):
    if weights is None:
        return seq[int(rng.integers(len(seq)))]
    cdf = np.cumsum(weights) / np.sum(weights)
    return seq[min(int(np.searchsorted(cdf, rng.uniform(), side="right")), len(seq) - 1)]


def prose(seed: float, n: int, weights: Mapping[str, np.ndarray] | None = None, cadence: float = 0.35,
          phrase: str = "") -> str:
    rng = ChaosRNG(seed)
    w = weights or {}
    out = []
    for _ in range(n):
        a, b = _pick(rng, ADJ, w.get("adj")), _pick(rng, NOUN, w.get("noun"))
        s = f"the {a} {b} {_pick(rng, VERB, w.get('verb'))} the {_pick(rng, NOUN, w.get('noun'))}"
        if rng.uniform() < cadence:
            s += f" {_pick(rng, PREP, w.get('prep'))} the {_pick(rng, ADJ, w.get('adj'))} {_pick(rng, NOUN, w.get('noun'))}"
        if phrase and rng.uniform() < 0.25:
            s += f", {phrase}"
        out.append(s + ".")
    return " ".join(out) + " "


def code(seed: float, n: int) -> str:
    rng = ChaosRNG(seed)
    lines = []
    for i in range(n):
        a, b = _pick(rng, NAMES), _pick(rng, NAMES)
        k = int(rng.integers(10))
        form = int(rng.integers(3))
        if form == 0:
            lines.append(f"def f{i % 10}({a}): return {a} {_pick(rng, OPS)} {k}")
        elif form == 1:
            lines.append(f"{a} = {b} {_pick(rng, OPS)} {k}")
        else:
            lines.append(f"for {a} in range({k}): {b} += {a}")
    return "\n".join(lines) + "\n"


def windows(text: str | bytes, context: int) -> tuple[np.ndarray, np.ndarray]:
    """Every (context bytes → next byte) pair, left-padded with byte 0."""
    data = np.frombuffer(text.encode("utf-8") if isinstance(text, str) else bytes(text), dtype=np.uint8)
    padded = np.concatenate([np.zeros(context, dtype=np.int64), data.astype(np.int64)])
    idx = np.arange(len(data))[:, None] + np.arange(context)[None, :]
    return padded[idx], data.astype(np.int64)


def batches(X: np.ndarray, y: np.ndarray, size: int, seed: float, steps: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """`steps` minibatches drawn from a chaotic permutation, re-shuffled each epoch."""
    rng = ChaosRNG(seed)
    order = np.argsort(rng.uniform(len(X)))
    pos = 0
    for _ in range(steps):
        if pos + size > len(order):
            order, pos = np.argsort(rng.uniform(len(X))), 0
        sel = order[pos:pos + size]
        pos += size
        yield X[sel], y[sel]


def probe_set(context: int, n: int = 128, seed: float = 0.2718) -> np.ndarray:
    """n fixed windows, evenly strided over a neutral prose+code mix (identical for every vessel and substrate)."""
    text = prose(seed, 60) + code(seed + 0.1, 30)
    X, _ = windows(text, context)
    stride = max(1, len(X) // n)
    return X[context::stride][:n]


def held_out(context: int, seed: float = 0.5772, n: int = 40, register: str = "prose") -> tuple[np.ndarray, np.ndarray]:
    return windows(prose(seed, n) if register == "prose" else code(seed, n), context)

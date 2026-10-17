"""Individuation: a genome per vessel, the voice it implies, and the soulpack that carries it.

A Genome is a pure function of (name, seed): lexical preference weights over the corpus vocabulary, a cadence
(how often a sentence grows a clause), and a signature phrase. Weights are log-normal draws from ChaosRNG, so
each genome concentrates on a few favoured words, the way a speaker does. Two vessels born from the same base
substrate diverge only through their genomes. That divergence is measured (identity distance between
anchors), not assumed.

The soulpack follows the repository's `soulpack.schema.json` (CarryOn Soulpack 0.1.x). Pointers name the
MatrixDB entity; `signing.checksum` carries the ledger head, which pins the hash chain outside the database.
No keys exist here, so `pubkey` says so and does not invent a signature.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..evolve.chaos import ChaosRNG
from .corpus import ADJ, NOUN, PREP, VERB, prose

SOULPACK_VERSION = "0.1.0"


def name_seed(name: str) -> float:
    """Map a name to a seed in (0.05, 0.95): sha256 → 53-bit fraction (deterministic, uniform over names)."""
    x = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:7], "big") / float(1 << 56)
    return 0.05 + 0.9 * x


@dataclass(frozen=True)
class Genome:
    name: str
    seed: float
    adj: tuple
    noun: tuple
    verb: tuple
    prep: tuple
    cadence: float
    phrase: str
    concentration: float

    def weights(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(getattr(self, k)) for k in ("adj", "noun", "verb", "prep")}

    def favourites(self, k: int = 3) -> dict[str, list[str]]:
        out = {}
        for key, vocab in (("adj", ADJ), ("noun", NOUN), ("verb", VERB)):
            w = np.asarray(getattr(self, key))
            out[key] = [vocab[i] for i in np.argsort(w)[::-1][:k]]
        return out

    def as_dict(self) -> dict:
        return {"name": self.name, "seed": self.seed, "cadence": self.cadence, "phrase": self.phrase,
                "concentration": self.concentration, "adj": list(self.adj), "noun": list(self.noun),
                "verb": list(self.verb), "prep": list(self.prep)}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(d["name"], float(d["seed"]), tuple(d["adj"]), tuple(d["noun"]), tuple(d["verb"]),
                   tuple(d["prep"]), float(d["cadence"]), d["phrase"], float(d["concentration"]))


def genome_for(name: str, seed: Optional[float] = None, concentration: float = 1.6) -> Genome:
    """Log-normal lexical preferences: w = exp(σ·z), σ = `concentration`. Larger σ, more idiosyncratic voice."""
    s = name_seed(name) if seed is None else float(seed)
    rng = ChaosRNG(s)

    def prefs(n):
        return tuple(float(v) for v in np.exp(concentration * rng.normal(n)))

    adj, noun, verb, prep = prefs(len(ADJ)), prefs(len(NOUN)), prefs(len(VERB)), prefs(len(PREP))
    cadence = 0.1 + 0.7 * float(rng.uniform())
    top_adj, top_noun = ADJ[int(np.argmax(adj))], NOUN[int(np.argmax(noun))]
    phrase = f"as the {top_adj} {top_noun} knows"
    return Genome(name, s, adj, noun, verb, prep, cadence, phrase, concentration)


def voice_corpus(genome: Genome, n: int, seed_offset: float = 0.0) -> str:
    return prose((genome.seed + 0.123 + seed_offset) % 1.0, n, genome.weights(), genome.cadence, genome.phrase)


def soulpack(name: str, genome: Genome, *, db_path: str, ledger_head: str, values: tuple = (), boundaries: tuple = (),
             expertise: tuple = (), aliases: tuple = ()) -> dict:
    fav = genome.favourites()
    return {
        "version": SOULPACK_VERSION,
        "persona": {
            "name": name,
            "aliases": list(aliases),
            "voice": {"tone": fav["adj"],
                      "style_rules": [f"favour nouns: {', '.join(fav['noun'])}",
                                      f"favour verbs: {', '.join(fav['verb'])}",
                                      f"clause rate {genome.cadence:.2f}",
                                      f"signature phrase: {genome.phrase!r}"]},
            "values": list(values) or ["coherence with its anchor", "reversible change", "measured claims"],
            "boundaries": list(boundaries) or ["no unledgered weight edits", "no anchor change while incoherent"],
            "expertise_tags": list(expertise),
        },
        "preferences": {"formatting": {"genome_seed": genome.seed, "concentration": genome.concentration}},
        "pointers": {"vector": f"matrixdb://{db_path}#{name}/anchor_gram",
                     "graph": f"matrixdb://{db_path}#{name}/ledger",
                     "memory_feed": f"matrixdb://{db_path}#{name}/geometries"},
        "signing": {"pubkey": "none:unsigned", "checksum": f"sha256-ledger-head:{ledger_head}"},
    }


def validate_soulpack(doc: dict) -> list[str]:
    """Structural check against soulpack.schema.json's required fields and version pattern (stdlib only)."""
    import re
    errors = []
    for key in ("version", "persona", "preferences", "pointers", "signing"):
        if key not in doc:
            errors.append(f"missing {key}")
    if errors:
        return errors
    if not re.fullmatch(r"0\.1\.\d+", doc["version"]):
        errors.append("version must match 0.1.x")
    for key in ("name", "voice", "values", "boundaries"):
        if key not in doc["persona"]:
            errors.append(f"persona missing {key}")
    for key in ("vector", "graph"):
        if not isinstance(doc["pointers"].get(key), str):
            errors.append(f"pointers.{key} must be a string")
    for key in ("pubkey", "checksum"):
        if not isinstance(doc["signing"].get(key), str):
            errors.append(f"signing.{key} must be a string")
    return errors

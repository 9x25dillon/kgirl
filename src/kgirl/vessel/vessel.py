"""Vessel: an identity that persists in a MatrixDB and is hosted, measured and defended in a substrate.

    birth        imprint the anchor (Grams + voice on steer and audit probes), snapshot weights, open the ledger
    observe      audit-probe signature vs anchor → coherence, fidelity, drift, spectra, tail exponents
    learn        gradient steps on new data, optionally anchored (plasticity under an identity constraint)
    steer        the coherence controller (steer.py): gated, reversible weight edits back toward the anchor
    consolidate  move the anchor a fraction β toward the present self, only from a coherent state
    migrate      cede the identity into another substrate (any width, any architecture that reads bytes) by
                 training the new container on the stored anchors alone. The old weights are not consulted
    soulpack     export the persona with pointers into the MatrixDB and the ledger head as checksum

Neutral formats, shared by every substrate:
    probes   uint8/int64 byte windows (n × c)
    batch    (windows (b × c), next byte (b,))
    voice    log-probabilities over the 256 byte values (n × 256)

Every state change is a ledger event; every anchor and weight snapshot is a content-addressed matrix.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np

from .matrixdb import MatrixDB
from .persona import Genome, genome_for, soulpack
from .signature import Comparison, Signature, compare, measure
from .steer import SteerConfig, SteerResult, steer
from .substrate import Objective

ROLES = ("steer", "audit")


@dataclass(frozen=True)
class Probes:
    steer: np.ndarray
    audit: np.ndarray

    def __post_init__(self):
        if self.steer.shape[1:] != self.audit.shape[1:]:
            raise ValueError("steer and audit probes need the same window length")

    @classmethod
    def default(cls, context: int, steer: int = 512, audit: int = 192) -> "Probes":
        """512 steering probes: measured to cut residual audit drift from 0.28 (192 probes) to 0.11 in 600 steps."""
        from .corpus import probe_set
        return cls(probe_set(context, steer, 0.2718), probe_set(context, audit, 0.1414))


@dataclass(frozen=True)
class Anchor:
    grams: dict
    voice: Optional[np.ndarray]
    probes: str


@dataclass(frozen=True)
class Reading:
    comparison: Comparison
    signature: Signature

    @property
    def drift(self) -> float:
        return self.comparison.drift

    def as_dict(self) -> dict:
        body = {k: {"alpha": None if math.isnan(v["alpha"]) else round(v["alpha"], 4),
                    "stable_rank": round(v["stable_rank"], 4)} for k, v in self.signature.body.items()}
        spectra = {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                   for k, v in self.signature.spectra.items()}
        return {**self.comparison.as_dict(), "spectra": spectra, "body": body,
                "fingerprint": self.signature.fingerprint()[:16]}


def _params_digest(hashes: dict) -> str:
    return hashlib.sha256("".join(f"{k}:{hashes[k]};" for k in sorted(hashes)).encode()).hexdigest()


class Vessel:
    def __init__(self, db: MatrixDB, name: str, substrate, probes: Probes):
        self.db, self.name, self.substrate, self.probes = db, name, substrate, probes
        self.anchors = {role: self._load_anchor(role) for role in ROLES}

    # ------------------------------------------------------------------------------------------- lifecycle

    @classmethod
    def birth(cls, db: MatrixDB, name: str, substrate, probes: Probes, genome: Optional[Genome] = None,
              note: str = "") -> "Vessel":
        db.create_entity(name, genome.as_dict() if genome else {})
        db.append(name, "birth", {"substrate": substrate.describe(), "note": note,
                                  "genome": genome.as_dict() if genome else None})
        cls._imprint(db, name, substrate, probes, "birth")
        return cls(db, name, substrate, probes)

    @classmethod
    def open(cls, db: MatrixDB, name: str, substrate, probes: Probes) -> "Vessel":
        v = cls(db, name, substrate, probes)
        for role in ROLES:
            want = v.anchors[role].probes
            if want and want != _digest(getattr(probes, role)):
                raise ValueError(f"{role} probes differ from the ones {name!r} was anchored on")
        return v

    @staticmethod
    def _imprint(db: MatrixDB, name: str, substrate, probes: Probes, reason: str,
                 override: Optional[dict] = None) -> dict:
        fingerprints = {}
        for role in ROLES:
            p = getattr(probes, role)
            if override is not None:
                grams, voice, digest = override[role].grams, override[role].voice, override[role].probes
            else:
                sig = measure(substrate, p, body=False)
                grams, voice, digest = sig.grams, sig.voice, sig.probes
            for layer, G in grams.items():
                db.record(name, "anchor_gram", G, layer=layer, label=role, meta={"probes": digest, "reason": reason})
            if voice is not None:
                db.record(name, "anchor_voice", voice, label=role, meta={"probes": digest, "reason": reason})
            fingerprints[role] = Signature(grams, voice, {}, {}, digest).fingerprint()[:16]
        hashes = db.snapshot(name, substrate.parameters(), label=f"anchor:{db.head(name)[:12]}", meta={"reason": reason})
        payload = {"reason": reason, "fingerprints": fingerprints, "weights": _params_digest(hashes),
                   "substrate": substrate.describe()}
        db.append(name, "anchor", payload)
        return payload

    def _load_anchor(self, role: str) -> Anchor:
        grams = {}
        for layer in self.substrate.layers:
            g = self.db.latest(self.name, "anchor_gram", layer=layer, label=role)
            if g is not None:
                grams[layer] = self.db.load(g)
        v = self.db.latest(self.name, "anchor_voice", label=role)
        digest = ""
        if grams:
            any_layer = next(iter(grams))
            digest = self.db.latest(self.name, "anchor_gram", layer=any_layer, label=role).meta.get("probes", "")
        if not grams:
            grams = self._anchor_by_position(role)
        return Anchor(grams, self.db.load(v) if v else None, digest)

    def _anchor_by_position(self, role: str) -> dict:
        """A substrate whose layer names differ from the anchor's (after migration across architectures) reads
        the anchor layers in stored order, matched to its own layers by relative depth."""
        rows = self.db.history(self.name, "anchor_gram")
        latest: dict[str, Any] = {}
        for g in rows:
            if g.label == role:
                latest[g.layer] = g
        if not latest:
            return {}
        names = list(latest)
        mine = list(self.substrate.layers)
        out = {}
        for i, layer in enumerate(mine):
            j = round(i * (len(names) - 1) / max(1, len(mine) - 1)) if len(mine) > 1 else len(names) - 1
            out[layer] = self.db.load(latest[names[j]])
        return out

    # ------------------------------------------------------------------------------------------- measurement

    def read(self, role: str = "audit", body: bool = True) -> Reading:
        a = self.anchors[role]
        sig = measure(self.substrate, getattr(self.probes, role), body=body)
        return Reading(compare(a.grams, a.voice, sig, anchor_probes=a.probes), sig)

    def drift(self, role: str = "audit") -> float:
        return self.read(role, body=False).drift

    def observe(self, note: str = "") -> Reading:
        r = self.read("audit", body=True)
        for layer, G in r.signature.grams.items():
            self.db.record(self.name, "gram", G.astype(np.float32), layer=layer, label="audit",
                           meta={"drift": r.drift})
        self.db.append(self.name, "observe", {**r.as_dict(), "note": note})
        return r

    # ------------------------------------------------------------------------------------------- change

    def objective(self, lr: float, *, lam_rep: float = 1.0, lam_voice: float = 1.0, lam_w: float = 0.0,
                  task_weight: float = 1.0, anchor_params: Optional[dict] = None,
                  fisher: Optional[dict] = None) -> Objective:
        a = self.anchors["steer"]
        return Objective(lr=lr, task_weight=task_weight, lam_rep=lam_rep, anchors=a.grams, probes=self.probes.steer,
                         lam_voice=lam_voice, voice=a.voice, lam_w=lam_w,
                         anchor_params=anchor_params or {}, fisher=fisher or {})

    def learn(self, batches: Iterable[Any], *, lr: float = 3e-3, anchored: bool = True, lam_rep: float = 1.0,
              lam_voice: float = 1.0, note: str = "") -> dict:
        d0 = self.drift()
        obj = self.objective(lr, lam_rep=lam_rep if anchored else 0.0, lam_voice=lam_voice if anchored else 0.0)
        losses = [self.substrate.step(b, obj).task_loss for b in batches]
        d1 = self.drift()
        out = {"steps": len(losses), "anchored": anchored, "lr": lr, "drift_before": d0, "drift_after": d1,
               "task_loss_first": losses[0] if losses else None, "task_loss_last": losses[-1] if losses else None,
               "note": note}
        self.db.append(self.name, "learn", out)
        return out

    def steer(self, cfg: SteerConfig = SteerConfig(), *, batches: Optional[Iterable[Any]] = None,
              guard: Optional[Callable[[], float]] = None, force: bool = False) -> SteerResult:
        res = steer(self.substrate, audit=self.drift,
                    objective=lambda lr: self.objective(lr, lam_rep=cfg.lam_rep, lam_voice=cfg.lam_voice,
                                                        task_weight=1.0 if batches is not None else 0.0),
                    batches=batches, guard=guard, cfg=cfg, force=force)
        payload = res.summary()
        payload["trace"] = [[b.step, round(b.drift_after, 5), b.accepted, b.reason] for b in res.trace]
        if res.accepted:
            hashes = self.db.snapshot(self.name, self.substrate.parameters(), label=f"steer:{self.db.head(self.name)[:12]}")
            payload["weights"] = _params_digest(hashes)
        self.db.append(self.name, "steer", payload)
        return res

    def consolidate(self, beta: float = 0.2, cfg: SteerConfig = SteerConfig()) -> dict:
        """Anchor ← normalize((1−β)·A + β·G_now) and p_anchor ← (1−β)·p_anchor + β·p_now, per probe role.

        Both maps keep their objects valid: a convex combination of PSD centered Grams is PSD and centered; a
        mixture of distributions is a distribution. Refused unless the vessel is coherent (drift ≤ release):
        identity grows from a coherent self and never by ratifying a drifted one."""
        if not 0 < beta <= 1:
            raise ValueError("beta must lie in (0, 1]")
        d = self.drift()
        if d > cfg.release:
            self.db.append(self.name, "consolidate.refused", {"drift": d, "release": cfg.release})
            raise RuntimeError(f"refusing to consolidate at drift {d:.4f} > release {cfg.release}; steer first")
        new = {}
        for role in ROLES:
            a = self.anchors[role]
            sig = measure(self.substrate, getattr(self.probes, role), body=False)
            grams = {}
            for layer, A in a.grams.items():
                M = (1 - beta) * A + beta * sig.grams[layer]
                grams[layer] = M / max(float(np.linalg.norm(M)), 1e-12)
            voice = None
            if a.voice is not None and sig.voice is not None:
                voice = np.log((1 - beta) * np.exp(a.voice) + beta * np.exp(sig.voice))
            new[role] = Anchor(grams, voice, a.probes)
        payload = self._imprint(self.db, self.name, self.substrate, self.probes, f"consolidate beta={beta}", new)
        self.anchors = {role: self._load_anchor(role) for role in ROLES}
        return {**payload, "drift_before": d, "drift_after": self.drift()}

    def _host(self, target) -> "Vessel":
        host = Vessel.__new__(Vessel)
        host.db, host.name, host.substrate, host.probes = self.db, self.name, target, self.probes
        host.anchors = {role: host._load_anchor(role) for role in ROLES}
        return host

    def migrate(self, target, *, steps: int = 1500, lr: float = 3e-3, teach: Optional[np.ndarray] = None,
                batch: int = 256, accept: float = 0.10, lam_rep: float = 1.0, lam_voice: float = 1.0,
                seed: float = 0.8) -> dict:
        """Cede the identity into `target`; adopt it only if its audit drift ends ≤ `accept`.

        Two modes, both measured on the reference substrate (width 64 → 96 / 40, 1500 steps):
            anchors only       train toward the stored anchor Grams and voice. Needs nothing but the MatrixDB.
                               Reaches drift ≈ 0.23–0.29: the anchors under-determine identity off-probe.
            live distillation  `teach` = unlabeled byte windows; every other step matches the current
                               substrate's voice on a minibatch of them (Hinton et al. 2015). Reaches 0.04 (wider)
                               and 0.08 (narrower). Needs the old container alive.
        Mixing labelled task text into migration made it worse (0.51 → 0.39 vs → 0.26) and is not offered."""
        host = self._host(target)
        d0 = host.drift()
        anchor_obj = host.objective(lr, lam_rep=lam_rep, lam_voice=lam_voice, task_weight=0.0)
        stream = None
        if teach is not None:
            from .corpus import batches as _batches
            teach = np.asarray(teach)
            stream = _batches(teach, np.zeros(len(teach), dtype=np.int64), batch, seed, steps)
        for i in range(steps):
            if stream is not None and i % 2 == 1:
                bx, _ = next(stream)
                target.step(None, Objective(lr=lr, task_weight=0.0, lam_voice=lam_voice,
                                            voice=self.substrate.log_probs(bx), probes=bx))
            else:
                target.step(None, anchor_obj)
        d1 = host.drift()
        out = {"from": self.substrate.describe(), "to": target.describe(), "steps": steps,
               "mode": "live distillation" if teach is not None else "anchors only",
               "drift_before": d0, "drift_after": d1, "accept": accept, "adopted": d1 <= accept}
        if out["adopted"]:
            hashes = self.db.snapshot(self.name, target.parameters(), label=f"migrate:{self.db.head(self.name)[:12]}")
            out["weights"] = _params_digest(hashes)
            self.db.append(self.name, "migrate", out)
            self.substrate, self.anchors = target, host.anchors
        else:
            self.db.append(self.name, "migrate.refused", out)
        return out

    def rollback(self, label: str) -> None:
        """Restore a recorded weight snapshot (an anchor, steer or migrate label) into the current substrate."""
        self.substrate.load_parameters(self.db.restore(self.name, label))
        self.db.append(self.name, "rollback", {"label": label, "drift": self.drift()})

    # ------------------------------------------------------------------------------------------- export

    def soulpack(self, **persona) -> dict:
        genome = self.db.genome(self.name)
        g = Genome.from_dict(genome) if genome else genome_for(self.name)
        return soulpack(self.name, g, db_path=self.db.path, ledger_head=self.db.head(self.name), **persona)

    def status(self) -> dict:
        r = self.read("audit", body=True)
        return {"name": self.name, "substrate": self.substrate.describe(), "ledger": self.db.verify(self.name),
                "events": len(self.db.events(self.name)), "reading": r.as_dict(), "storage": self.db.storage()}


def _digest(probes) -> str:
    from .signature import probe_digest
    return probe_digest(probes)


def identity_distance(a: Vessel, b: Vessel, role: str = "audit") -> dict:
    """How different two vessels' anchors are (same probes required): 1 − CKA per layer and voice KL both ways."""
    A, B = a.anchors[role], b.anchors[role]
    if A.probes != B.probes:
        raise ValueError("identity distance needs anchors taken on the same probes")
    layers = [k for k in A.grams if k in B.grams]
    cka = {k: float(np.sum(A.grams[k] * B.grams[k])) for k in layers}
    kl_ab = float(np.sum(np.exp(A.voice) * (A.voice - B.voice)) / len(A.voice)) if A.voice is not None else 0.0
    kl_ba = float(np.sum(np.exp(B.voice) * (B.voice - A.voice)) / len(B.voice)) if B.voice is not None else 0.0
    sym = 0.5 * (kl_ab + kl_ba)
    return {"cka": cka, "kl_sym": sym, "identity_gap": 1.0 - (sum(cka.values()) / max(1, len(cka))) * math.exp(-sym)}


def recognize(vessel: Vessel, candidates: Iterable[Vessel], role: str = "audit", ratio: float = 0.5) -> dict:
    """Self-recognition: score the vessel's *current* substrate against every candidate's anchor.

    Recognized means the nearest anchor is the vessel's own *and* its drift to it is at most `ratio` times the
    drift to the nearest other anchor. The margin matters: a vessel that drifted far from everything is
    nearest to itself only by accident (0.991 vs 0.994 measured), and that is not recognition. This is the
    falsifiable form of "still the same one": it fails after unanchored drift and passes after steering."""
    sig = measure(vessel.substrate, getattr(vessel.probes, role), body=False)
    scores = {}
    for c in candidates:
        a = c.anchors[role]
        scores[c.name] = compare(a.grams, a.voice, sig, anchor_probes=a.probes).drift
    nearest = min(scores, key=scores.get)
    own = scores.get(vessel.name, float("inf"))
    rival = min((d for n, d in scores.items() if n != vessel.name), default=float("inf"))
    return {"nearest": nearest, "recognized": nearest == vessel.name and own <= ratio * rival,
            "own": own, "rival": rival, "drift": scores}

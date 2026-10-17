"""Verify the identity signature's invariance group by sampling it. Adapted from substrate-comm's framework.py.

substrate-comm defines a channel's level as the largest transformation group G with D ∘ g ∘ R = id, and checks
a declared level by sampling elements of G: one counterexample demotes it. The same discipline applies here.
The claim "identity is substrate-independent" is the claim that the signature factors through the quotient of
activation space by a declared group. The claim is tested both ways:

    DECLARED (must hold, |1 − CKA(X, gX)| ≤ tol for every sample)
        permute      hidden-unit permutation                    X → X P
        rotate       orthogonal change of basis                 X → X Q
        scale        isotropic gain                             X → c X,  c > 0
        translate    common offset                              X → X + 1bᵀ
        embed        isometric widening to d′ > d               X → X R,  R Rᵀ = I   (a larger container)

    CONTROLS (must break, CKA(X, gX) < 1 − margin, or the measure is vacuous: quotient collapse)
        anisotropic  per-feature gains spanning two decades
        shuffle      probes re-ordered (identity is tied to which probe is which)

    SENSITIVITY (reported, not gated)
        cube         elementwise x³. Linear CKA barely sees monotone odd nonlinearities on saturated units: on a
                     trained tanh layer whose units sit near ±1, x³ ≈ x and CKA stays ≈ 0.98. This is a
                     measured blind spot of linear CKA, documented rather than hidden.

    SUBSTRATE (end to end, on a live ReferenceLM)
        reparametrize  permute both hidden layers' units through the weights: every weight matrix changes,
                       the computed function does not, so signature and voice must be unchanged to tolerance

All randomness is ChaosRNG (no PRNG). Level 0: these are definition-chases; their value is that they are
checked on every run instead of asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..evolve.chaos import ChaosRNG
from .geometry import linear_cka

Transform = Callable[[np.ndarray], np.ndarray]


def _orthogonal(rng: ChaosRNG, d: int) -> np.ndarray:
    Q, R = np.linalg.qr(rng.normal((d, d)))
    return Q * np.sign(np.diag(R))


def permute(rng, d):
    p = np.argsort(rng.uniform(d))
    return (lambda X: X[:, p]), "permute"


def rotate(rng, d):
    Q = _orthogonal(rng, d)
    return (lambda X: X @ Q), "rotate"


def scale(rng, d):
    c = float(np.exp(rng.uniform() * 4 - 2))
    return (lambda X: c * X), f"scale c={c:.3g}"


def translate(rng, d):
    b = rng.normal(d) * 3.0
    return (lambda X: X + b), "translate"


def embed(rng, d):
    extra = 1 + int(rng.integers(2 * d))
    R = _orthogonal(rng, d + extra)[:d, :]
    return (lambda X: X @ R), f"embed d={d}→{d + extra}"


def anisotropic(rng, d):
    g = np.exp(np.linspace(-2.3, 2.3, d))[np.argsort(rng.uniform(d))]
    return (lambda X: X * g), "anisotropic"


def nonlinear(rng, d):
    return (lambda X: X ** 3), "cube"


def shuffle(rng, d):
    def f(X):
        return X[np.argsort(ChaosRNG(0.6180).uniform(len(X)))]
    return f, "shuffle probes"


DECLARED = (permute, rotate, scale, translate, embed)
CONTROLS = (anisotropic, shuffle)
SENSITIVITY = (nonlinear,)


@dataclass(frozen=True)
class InvarianceReport:
    samples: int
    worst_declared: float
    declared_failures: tuple
    controls: dict
    control_failures: tuple
    sensitivity: dict
    reparametrization: dict

    @property
    def verified(self) -> bool:
        return not self.declared_failures and not self.control_failures and self.reparametrization.get("ok", True)

    def render(self) -> str:
        lines = [f"declared group: {self.samples} samples, worst |1 − CKA| = {self.worst_declared:.2e}"
                 + ("" if not self.declared_failures else f"  FAILURES {self.declared_failures[:3]}")]
        for name, v in self.controls.items():
            lines.append(f"control {name:<20} CKA = {v:.4f}  ({'breaks, as required' if name not in self.control_failures else 'DOES NOT BREAK'})")
        for name, v in self.sensitivity.items():
            lines.append(f"sensitivity {name:<16} CKA = {v:.4f}  (reported, not gated)")
        if self.reparametrization:
            r = self.reparametrization
            lines.append(f"reparametrized substrate: weights changed {r['weights_changed']}, max |ΔCKA| {r['max_cka_gap']:.2e}, "
                         f"max |Δ log p| {r['max_logp_gap']:.2e}, fingerprint equal {r['fingerprint_equal']}")
        lines.append("VERIFIED" if self.verified else "NOT VERIFIED")
        return "\n".join(lines)


def verify_activations(X: np.ndarray, *, samples: int = 8, tol: float = 1e-9, margin: float = 0.02,
                       seed: float = 0.3141) -> tuple[int, float, list, dict, list, dict]:
    rng = ChaosRNG(seed)
    d = X.shape[1]
    worst, failures, n = 0.0, [], 0
    for _ in range(samples):
        for gen in DECLARED:
            g, desc = gen(rng, d)
            gap = abs(1.0 - linear_cka(X, g(X)))
            n += 1
            worst = max(worst, gap)
            if gap > tol:
                failures.append((desc, gap))
    controls, broken_fail = {}, []
    for gen in CONTROLS:
        g, desc = gen(rng, d)
        v = linear_cka(X, g(X))
        controls[desc] = v
        if v >= 1.0 - margin:
            broken_fail.append(desc)
    sens = {}
    for gen in SENSITIVITY:
        g, desc = gen(rng, d)
        sens[desc] = linear_cka(X, g(X))
    return n, worst, failures, controls, broken_fail, sens


def reparametrize_reference(model, seed: float = 0.7071):
    """A ReferenceLM computing the same function through permuted hidden units (new weights, same map)."""
    from .substrate import ReferenceLM
    rng = ChaosRNG(seed)
    h = model.cfg.hidden
    p1, p2 = np.argsort(rng.uniform(h)), np.argsort(rng.uniform(h))
    q = model.parameters()
    twin = ReferenceLM(model.cfg)
    twin.load_parameters({
        "E": q["E"], "W1": q["W1"][:, p1], "b1": q["b1"][p1],
        "W2": q["W2"][p1][:, p2], "b2": q["b2"][p2],
        "W3": q["W3"][p2], "b3": q["b3"]})
    return twin


def verify(model, probes, *, samples: int = 8, tol: float = 1e-9, seed: float = 0.3141) -> InvarianceReport:
    """Group-sample every exposed layer of `model` on `probes`; add the end-to-end reparametrization test when the
    substrate is a ReferenceLM."""
    from .signature import measure
    from .substrate import ReferenceLM
    total, worst, fails, controls, cfails, sens = 0, 0.0, [], {}, [], {}
    for layer, X in model.activations(probes).items():
        n, w, f, c, cf, sv = verify_activations(X, samples=samples, tol=tol, seed=seed)
        total, worst = total + n, max(worst, w)
        fails += [(layer, *x) for x in f]
        controls.update({f"{layer}:{k}": v for k, v in c.items()})
        cfails += [f"{layer}:{k}" for k in cf]
        sens.update({f"{layer}:{k}": v for k, v in sv.items()})
    rep = {}
    if isinstance(model, ReferenceLM):
        twin = reparametrize_reference(model)
        a, b = measure(model, probes, body=False), measure(twin, probes, body=False)
        changed = all(not np.allclose(model.p[k], twin.p[k]) for k in ("W1", "W2", "W3"))
        gap = max(abs(1.0 - float(np.sum(a.grams[k] * b.grams[k]))) for k in a.grams)
        lgap = float(np.max(np.abs(a.voice - b.voice)))
        rep = {"weights_changed": changed, "max_cka_gap": gap, "max_logp_gap": lgap,
               "fingerprint_equal": a.fingerprint() == b.fingerprint(),
               "ok": changed and gap < 1e-9 and lgap < 1e-9 and a.fingerprint() == b.fingerprint()}
    return InvarianceReport(total, worst, tuple(fails), controls, tuple(cfails), sens, rep)

"""Level 0 claims this package relies on, recomputed from scratch.

Run from `research/`:    python -m efl_iwme_cascade checks

Each check states one claim and recomputes it. No check reads a results file and no check fits or scores a
registered ground truth of Part 1, so the list can run before, during or after the registered runs.
Randomness comes from streams that no experiment uses (PCG64, spawn-key purpose 99); the two checks that
look at the experiment's own generators say so.

The parameters of the models below are deliberately different from every registered ground truth.
"""

from __future__ import annotations

import ast
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import host, models, streams
from .metrics import closure as cl
from .metrics import coherence as co
from .metrics import description_length as dl
from .models import baseline as bl
from .models import efl
from .models import iwme
from .models.cascade import sigmoid

Array = np.ndarray

AM = models.AttentionCascade.from_host(host.EMACascade(rho=0.3, T=0.6, bias=-0.2))
HM = models.HiddenMarkov(host.MarkovCascade(p=(0.8, 0.3), P=(((0.6, 0.4), (0.1, 0.9)), ((0.5, 0.5), (0.7, 0.3))),
                                            pi0=(0.3, 0.7)))
MC2 = models.markov_from_table(2, [[0.55], [0.7, 0.2], [0.9, 0.4, 0.6, 0.1]])
BIN = models.Binomial(0.63)
SEG = 256


class CheckStream:
    """Uniforms and normals for the checks only: PCG64 under a spawn key outside every experiment purpose."""

    PURPOSE = 99

    def __init__(self, tag: int):
        if self.PURPOSE in streams.PURPOSES.values():
            raise RuntimeError("the check stream must not share a purpose with an experiment stream")
        seq = np.random.SeedSequence(streams.PCG_ENTROPY, spawn_key=(self.PURPOSE, tag))
        self._gen = np.random.Generator(np.random.PCG64(seq))

    def uniform(self, size) -> Array:
        return self._gen.random(size)

    def normal(self, size) -> Array:
        return self._gen.standard_normal(size)


@dataclass
class Check:
    id: str
    claim: str
    passed: bool
    detail: str


_REGISTRY: List[Tuple[str, str, Callable[[], Tuple[bool, str]]]] = []


def check(cid: str, claim: str):
    def register(fn: Callable[[], Tuple[bool, str]]):
        _REGISTRY.append((cid, claim, fn))
        return fn
    return register


def ids() -> List[str]:
    return [cid for cid, _, _ in _REGISTRY]


def run_one(cid: str) -> Check:
    for known, claim, fn in _REGISTRY:
        if known == cid:
            try:
                passed, detail = fn()
            except Exception as exc:                       # a check that cannot run has failed
                passed, detail = False, f"raised {type(exc).__name__}: {exc}"
            return Check(cid, claim, bool(passed), detail)
    raise KeyError(f"unknown check {cid!r}")


def run_all(only: Optional[Sequence[str]] = None) -> List[Check]:
    return [run_one(cid) for cid in ids() if only is None or cid in only]


def render(results: Sequence[Check]) -> str:
    lines = [f"{'PASS' if r.passed else 'FAIL'}  {r.id}  {r.claim}\n          {r.detail}" for r in results]
    lines.append(f"{sum(r.passed for r in results)} of {len(results)} checks pass")
    return "\n".join(lines)


def _residual_cov(model, strings: Array, law: Array) -> Array:
    return cl.walk_covariance(models.signs(strings) - (2.0 * model.cond_left(strings) - 1.0), law)


# ----------------------------------------------------------------------------------------------
# Strings and laws
# ----------------------------------------------------------------------------------------------

@check("C01", "symbolization is the binary expansion, first symbol most significant")
def _c01():
    s3 = models.all_strings(3)
    pts = np.array([0.0, 0.5, 0.75, 0.8125, 0.999])
    want = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]])
    u = CheckStream(1).uniform(1000)
    back = models.points_from_strings(models.symbolize(u, 16))
    gap = float(np.max(u - back))
    refused = False
    try:
        models.symbolize(np.array([1.0]), 4)
    except ValueError:
        refused = True
    ok = (s3.shape == (8, 3) and list(s3[5]) == [1, 0, 1] and list(s3[1]) == [0, 0, 1]
          and np.array_equal(models.symbolize(pts, 4), want) and 0.0 <= float(np.min(u - back))
          and gap < 2.0 ** -16 and refused)
    return ok, f"cell left ends within 2^-16 of the points (largest gap {gap:.2e}); a point at 1.0 is refused"


@check("C02", "every family is a probability law on strings, with the registered parameter counts")
def _c02():
    strings = models.all_strings(10)
    sums = {name: float(models.string_law(m, 10, strings).sum())
            for name, m in (("B", BIN), ("MC2", MC2), ("HMM2", HM), ("AM", AM))}
    sample = strings[:: 37]
    counts = {f: models.fit_family(f, sample, kind="pcg").k for f in models.FAMILIES}
    ok = all(abs(v - 1.0) < 1e-12 for v in sums.values()) and counts == models.PARAMETERS
    return ok, "law totals " + ", ".join(f"{k} {v:.12f}" for k, v in sums.items()) + f"; k = {counts}"


@check("C03", "the hidden-state law (forward recursion) equals the sum over hidden paths")
def _c03():
    strings = models.all_strings(6)
    law = models.string_law(HM, 6, strings)
    p, trans, pi0 = np.array(HM.cascade.p), np.array(HM.cascade.P), np.array(HM.cascade.pi0)
    worst = 0.0
    for i in (0, 17, 42, 63):
        total = 0.0
        for hidden in itertools.product((0, 1), repeat=6):
            pr = pi0[hidden[0]]
            for pos in range(6):
                c = strings[i, pos]
                pr *= p[hidden[pos]] if c == 0 else 1.0 - p[hidden[pos]]
                if pos < 5:
                    pr *= trans[hidden[pos], c, hidden[pos + 1]]
            total += pr
        worst = max(worst, abs(total - float(law[i])))
    return worst < 1e-14, f"largest difference over 4 strings of length 6: {worst:.1e}"


@check("C04", "the attention-memory family is the host's two-point attention memory, sampled (T_attention = 2T)")
def _c04():
    c = AM.as_host()
    mem = host.two_point_memory(2.0 * c.T, c.rho)
    worst = 0.0
    for h in (-0.8, -0.1, 0.0, 0.37, 0.9):
        p_left = float(c.p_left(np.array([h]))[0])
        worst = max(worst, abs(p_left - float(mem.weights(np.array([h]), np.array([c.bias]))[0])))
        mean_next = (1.0 - c.rho) * h + c.rho * (2.0 * p_left - 1.0)
        worst = max(worst, abs(mean_next - float(mem.step(np.array([h]), np.array([c.bias]))[0])))
    strings = models.all_strings(10)
    law = models.string_law(AM, 10, strings)
    moments = max(abs(float((law ** q).sum()) - c.enumerate_Zn(q, 10)) for q in (0.5, 2.0, 3.0))
    return worst < 1e-14 and moments < 1e-12, (f"left probability and mean memory update agree to {worst:.1e}; "
                                               f"sums of law^q equal the host's enumeration to {moments:.1e}")


@check("C05", "the host's repaired IWME core, given the family's maps, emits the family symbol for symbol")
def _c05():
    u = CheckStream(5).uniform((300, 12))
    agree, total = 0, 0
    for model in (AM, AM.with_coupling(0.0), AM.with_coupling(0.5),
                  models.AttentionCascade(rho=0.7, gain=2.5, offset=0.4)):
        via_core = iwme.core_rollout(model, u)
        direct = iwme.sequential_sample(model, u)
        agree += int((via_core == direct).sum())
        total += direct.size
    host_law = AM.as_host()
    return agree == total and isinstance(host_law, host.EMACascade), (
        f"{agree} of {total} symbols equal across four parameter points, coupling 1, 0.5 and 0 included")


@check("C06", "with the actuation severed (coupling 0, hypothesis H1) the law is an order-1 Markov chain")
def _c06():
    strings = models.all_strings(12)
    cut = models.string_law(AM.with_coupling(0.0), 12, strings)
    chain = models.string_law(models.markov_limit(AM), 12, strings)
    full = models.string_law(AM, 12, strings)
    gap = float(np.abs(cut - chain).max())
    cost = dl.regret_bits(full, AM.with_coupling(0.0), strings)
    return gap < 1e-15 and cost > 1e-3, (f"largest law difference {gap:.1e}; the coupled law costs "
                                         f"{cost:.4f} bits per symbol under the severed model")


@check("C07", "in the core, a memory that does not actuate is unobservable; one that actuates is observable")
def _c07():
    steps = 40
    noise = CheckStream(7).uniform((50, steps))
    quiet = np.zeros(1)
    inputs = [{"U": quiet, "E": quiet, "Yext": quiet}] * steps

    def outputs(rho_mem: float, coupling: float) -> Array:
        core = host.IWMECore(
            Y_E=lambda X, Yext, nz: X,
            A=lambda H, Y, E: np.array([0.5, 0.5]),
            M=lambda H, A, Y, nz: (1.0 - rho_mem) * H + rho_mem * Y,
            F_E=lambda X, U, nz, act: np.where(nz < sigmoid(act + 0.5 * X + 0.1), 1.0, -1.0),
            C_E=lambda H: coupling * H,
            T=lambda X: X,
        )
        rows = []
        for row in noise:
            nzs = [{"x": np.array([row[t]]), "y": quiet, "h": quiet} for t in range(steps)]
            rows.append(core.run(np.zeros(1), np.zeros(1), inputs, nzs)[:, 0])
        return np.array(rows)

    inert = int((outputs(0.2, 0.0) != outputs(0.9, 0.0)).sum())
    active = int((outputs(0.2, 1.5) != outputs(0.9, 1.5)).sum())
    return inert == 0 and active > 0, (f"two memory maps, same noise: {inert} differing outputs with C_E = 0, "
                                       f"{active} of {noise.size} with C_E = 1.5 H")


# ----------------------------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------------------------

@check("C08", "counts are estimated with add-half (KT), and weights act as counts")
def _c08():
    strings = np.array([[0, 0, 1], [0, 1, 1]])
    mc = models.fit_markov(strings, 1)
    exact = (abs(models.fit_binomial(strings).p - 3.5 / 7.0) < 1e-15
             and abs(float(mc.tables[0][0]) - 2.5 / 3.0) < 1e-15
             and abs(float(mc.tables[1][0]) - 1.5 / 4.0) < 1e-15
             and abs(float(mc.tables[1][1]) - 0.5 / 2.0) < 1e-15)
    every = models.all_strings(3)
    doubled = np.concatenate([every, every])
    w = np.full(8, 2.0)
    as_counts = (abs(models.fit_binomial(every, w).p - models.fit_binomial(doubled).p) < 1e-15
                 and np.allclose(models.fit_markov(every, 2, w).tables[2], models.fit_markov(doubled, 2).tables[2]))
    return exact and as_counts, "hand-computed estimates on two strings; weight 2 equals each string twice"


@check("C09", "each family recovers its own generator: mean regret within 5k/(2Nn ln 2)")
def _c09():
    n, size = 12, 2000
    strings = models.all_strings(n)
    parts, ok = [], True
    for tag, (name, model, reps) in enumerate((("B", BIN, 8), ("MC2", MC2, 8), ("AM", AM, 8), ("HMM2", HM, 2))):
        law = models.string_law(model, n, strings)
        regrets = []
        for r in range(reps):
            sample = strings[models.sample_indices(law, size, CheckStream(900 + 10 * tag + r))]
            regrets.append(dl.regret_bits(law, models.fit_family(name, sample, kind="pcg"), strings))
        bound = 5.0 * dl.asymptotic_regret(models.PARAMETERS[name], size * n)
        ok &= float(np.mean(regrets)) < bound
        parts.append(f"{name} {np.mean(regrets):.2e} (bound {bound:.2e})")
    return ok, f"depth {n}, N = {size}: " + ", ".join(parts)


@check("C10", "projection onto a family's own law has zero floor; onto a smaller family, a positive one")
def _c10():
    strings = models.all_strings(10)
    own = {}
    for name, model in (("AM", AM), ("MC2", MC2), ("HMM2", HM)):
        law = models.string_law(model, 10, strings)
        own[name] = dl.regret_bits(law, models.project(name, law, strings, kind="pcg"), strings)
    law = models.string_law(AM, 10, strings)
    small = {name: dl.regret_bits(law, models.project(name, law, strings), strings) for name in ("B", "MC1", "MC3")}
    ok = (own["AM"] < 1e-9 and own["MC2"] < 1e-9 and own["HMM2"] < 1e-7
          and small["B"] > small["MC1"] > small["MC3"] > 1e-4)
    return ok, ("own floors " + ", ".join(f"{k} {v:.1e}" for k, v in own.items())
                + "; on the attention-memory law " + ", ".join(f"{k} {v:.2e}" for k, v in small.items()))


@check("C11", "the attention-memory family falls back to its binomial member on anti-persistent data")
def _c11():
    alternating = models.markov_from_table(1, [[0.5], [0.2, 0.8]])
    strings = models.all_strings(10)
    fit = models.project("AM", models.string_law(alternating, 10, strings), strings)
    flat = float(np.abs(fit.cond_left(strings[:5]) - 0.5).max())
    return fit.gain == 0.0 and flat < 1e-6, f"fitted gain {fit.gain}, left probability within {flat:.1e} of 0.5"


@check("C12", "the HMM2 objective has the stated gradient, EM never lowers it, and its counts obey sum rules")
def _c12():
    strings = models.all_strings(8)
    w = models.string_law(HM, 8, strings) * 500.0
    theta = np.array([0.3, -0.4, 0.2, -1.0, 0.5, 0.8, -0.2])
    _, grad = bl.hmm_objective(theta, strings, w)
    fd = np.array([(bl.hmm_objective(theta + e, strings, w)[0] - bl.hmm_objective(theta - e, strings, w)[0]) / 2e-6
                   for e in 1e-6 * np.eye(7)])
    grad_gap = float(np.abs(grad - fd).max())
    p, trans, pi0 = bl._hmm_unpack(theta)
    _, occ, _, moves, first, total = bl._hmm_expectations(strings, w, p, trans, pi0)
    sums = max(abs(float(occ.sum()) - total * 8), abs(float(moves.sum()) - total * 7), abs(float(first.sum()) - total))
    p, trans, pi0 = np.array([0.6, 0.4]), np.full((2, 2, 2), 0.5), np.array([0.5, 0.5])
    values = []
    for _ in range(30):
        values.append(bl.hmm_objective(bl._hmm_pack(p, trans, pi0), strings, w)[0])
        p, trans, pi0, _ = bl._hmm_em_step(strings, w, p, trans, pi0)
    drop = float(np.min(np.diff(values)))
    return grad_gap < 1e-5 and sums < 1e-8 and drop >= -1e-9, (
        f"gradient against central differences {grad_gap:.1e}; count sums off by {sums:.1e}; "
        f"smallest EM change {drop:.1e}")


@check("C13", "Amendment 1: 80 EM steps stop short of the optimum that EM warm-up plus quasi-Newton reaches")
def _c13():
    strings = models.all_strings(10)
    law = models.string_law(HM, 10, strings)
    w = law * 1e12
    rng = streams.stream("chaos", "hmm_init", 0, 0)          # the estimator's own first starting point
    p = 0.2 + 0.6 * rng.uniform(2)
    trans = 0.2 + rng.uniform((2, 2, 2))
    trans /= trans.sum(axis=2, keepdims=True)
    pi0 = np.array([0.5, 0.5])
    for _ in range(80):
        p, trans, pi0, _ = bl._hmm_em_step(strings, w, p, trans, pi0)
    em80 = dl.regret_bits(law, models.hidden_markov_from_arrays(p, trans, pi0), strings)
    amended = dl.regret_bits(law, models.fit_hidden_markov(strings, w), strings)
    return em80 > 5e-5 and amended < 1e-7, f"regret after 80 EM steps {em80:.1e}; amended estimator {amended:.1e}"


def hidden_markov_equivalent(model: models.HiddenMarkov, p_new: Tuple[float, float]) -> models.HiddenMarkov:
    """A different parameter point of the two-state hidden-state family with the same law on strings.

    With q the predictive probability of "left", d = p0 − p1 and P_sc the probability of moving to state 0
    from state s after child c, the law is driven by
        q_1 = p1 + pi0 d
        after left:   q' = A0 + B0 / q          A0 = p1 + d P_10 + p0 (P_00 − P_10),   B0 = −p0 p1 (P_00 − P_10)
        after right:  q' = A1 + B1 / (1 − q)    A1 = p1 + d P_11 − (1 − p0)(P_01 − P_11),
                                                B1 = (1 − p0)(1 − p1)(P_01 − P_11)
    so it depends on the seven parameters only through (q_1, A0, B0, A1, B1). For any new (p0, p1) the other
    five parameters can be solved to keep those five numbers; the result is a valid model whenever the
    solved probabilities lie in [0, 1].
    """
    c = model.cascade
    p0, p1 = c.p
    d = p0 - p1
    P00, P01, P10, P11 = c.P[0][0][0], c.P[0][1][0], c.P[1][0][0], c.P[1][1][0]
    q1 = p1 + c.pi0[0] * d
    A0, B0 = p1 + d * P10 + p0 * (P00 - P10), -p0 * p1 * (P00 - P10)
    A1, B1 = p1 + d * P11 - (1.0 - p0) * (P01 - P11), (1.0 - p0) * (1.0 - p1) * (P01 - P11)
    n0, n1 = p_new
    dn = n0 - n1
    D0 = -B0 / (n0 * n1)
    D1 = B1 / ((1.0 - n0) * (1.0 - n1))
    N10 = (A0 - n1 - n0 * D0) / dn
    N11 = (A1 - n1 + (1.0 - n0) * D1) / dn
    N00, N01 = N10 + D0, N11 + D1
    pi = (q1 - n1) / dn
    solved = np.array([N00, N01, N10, N11, pi])
    if solved.min() < 0.0 or solved.max() > 1.0:
        raise ValueError("no valid equivalent model at this (p0, p1)")
    trans = np.array([[[N00, 1.0 - N00], [N01, 1.0 - N01]], [[N10, 1.0 - N10], [N11, 1.0 - N11]]])
    return models.hidden_markov_from_arrays(np.array([n0, n1]), trans, np.array([pi, 1.0 - pi]))


def law_rank(build: Callable[[Array], object], point: Array, depth: int = 8, step: float = 1e-6) -> int:
    """Rank of the Jacobian of the string law with respect to a parameter vector (central differences)."""
    strings = models.all_strings(depth)
    cols = []
    for i in range(point.size):
        e = np.zeros(point.size)
        e[i] = step
        cols.append((models.string_law(build(point + e), depth, strings)
                     - models.string_law(build(point - e), depth, strings)) / (2.0 * step))
    sv = np.linalg.svd(np.array(cols).T, compute_uv=False)
    return int((sv > 1e-5 * sv[0]).sum())


@check("C14", "HMM2 has seven nominal parameters and five identifiable ones; AM (3) and MC2 (7) are identifiable")
def _c14():
    strings = models.all_strings(10)
    twin = hidden_markov_equivalent(HM, (0.82, 0.28))
    gap = float(np.abs(models.string_law(HM, 10, strings) - models.string_law(twin, 10, strings)).max())

    def vector(m: models.HiddenMarkov) -> Array:
        k = m.cascade
        return np.array([k.p[0], k.p[1], k.P[0][0][0], k.P[0][1][0], k.P[1][0][0], k.P[1][1][0], k.pi0[0]])

    moved = float(np.abs(vector(twin) - vector(HM)).max())

    def hmm(v: Array):
        trans = np.array([[[v[2], 1 - v[2]], [v[3], 1 - v[3]]], [[v[4], 1 - v[4]], [v[5], 1 - v[5]]]])
        return models.hidden_markov_from_arrays(v[:2], trans, np.array([v[6], 1 - v[6]]))

    ranks = {
        "HMM2": law_rank(hmm, vector(HM)),
        "AM": law_rank(lambda v: models.AttentionCascade(v[0], v[1], v[2]), np.array([AM.rho, AM.gain, AM.offset])),
        "MC2": law_rank(lambda v: models.markov_from_table(2, [v[:1], v[1:3], v[3:7]]),
                        np.concatenate([np.asarray(t, dtype=float) for t in MC2.tables])),
    }
    ok = gap < 1e-13 and moved > 0.01 and ranks == {"HMM2": 5, "AM": 3, "MC2": 7}
    return ok, (f"a second HMM2 point (p = 0.82, 0.28; largest parameter change {moved:.3f}) has the same law to "
                f"{gap:.1e}; Jacobian ranks of the string law {ranks}")


# ----------------------------------------------------------------------------------------------
# Closure
# ----------------------------------------------------------------------------------------------

@check("C15", "a walk with uncorrelated increments closes exactly (R = 1), whatever its variances")
def _c15():
    scales = np.linspace(0.3, 2.0, 12)
    exact = np.minimum.outer(np.cumsum(scales ** 2), np.cumsum(scales ** 2))
    ratio = cl.closure_ratio(exact)
    local = float(np.abs(cl.local_profile(exact)).max())
    sample = cl.closure_defect(cl.walk_covariance(CheckStream(15).normal((200000, 12)) * scales))
    return abs(ratio - 1.0) < 1e-12 and local < 1e-12 and sample < 0.3, (
        f"exact R = {ratio:.12f}; a sample of 200,000 walks reads D = {sample:.3f}, which is why measured "
        "defects are read against a surrogate band")


@check("C16", "calibration: correlated increments move the closure ratio, down if persistent, up if not")
def _c16():
    d1, d5 = cl.closure_defect(cl.ar1_walk_covariance(0.1, 16)), cl.closure_defect(cl.ar1_walk_covariance(0.5, 16))
    ok = (abs(d1 - 0.6027) < 5e-5 and abs(d5 - 1.9336) < 5e-5
          and cl.closure_ratio(cl.ar1_walk_covariance(0.3, 16)) < 1.0
          and cl.closure_ratio(cl.ar1_walk_covariance(-0.3, 16)) > 1.0
          and abs(cl.closure_ratio(cl.ar1_walk_covariance(0.0, 16)) - 1.0) < 1e-12)
    return ok, f"D = {d1:.4f} at increment correlation 0.1 and {d5:.4f} at 0.5 over 16 scales"


@check("C17", "closure of a model's residual walk tests whether the model's memory is sufficient")
def _c17():
    strings = models.all_strings(12)
    raw = {}
    for name, model in (("B", BIN), ("AM", AM), ("HMM2", HM)):
        raw[name] = cl.closure_defect(cl.walk_covariance(models.signs(strings), models.string_law(model, 12, strings)))
    own = {}
    for name, model in (("AM", AM), ("HMM2", HM)):
        law = models.string_law(model, 12, strings)
        own[name] = cl.closure_defect(_residual_cov(model, strings, law))
    law = models.string_law(AM, 12, strings)
    short = cl.closure_defect(_residual_cov(models.project("B", law, strings), strings, law))
    ok = raw["B"] < 1e-10 and raw["AM"] > 0.3 and max(own.values()) < 1e-9 and short > 0.3
    return ok, (f"raw D: binomial {raw['B']:.1e}, attention-memory {raw['AM']:.3f}; residual D of the true "
                f"model {max(own.values()):.1e}; of the best binomial on the attention-memory law {short:.3f}")


@check("C18", "layered chain: adjacent coupling closes, a skip coupling and a nonlinear layer do not")
def _c18():
    adjacent = cl.layered_closure([1.0, 2.0, 0.5], [0.5, 0.3, 0.2])
    skip = cl.layered_closure([1.0, 1.0], [0.5, 0.5], skip=1.0)["closure_ratio"]
    cube = cl.closure_ratio(np.array([[1.0, 3.0, 1.0], [3.0, 15.0, 3.0], [1.0, 3.0, 1.0]]))      # z, z^3, z
    ok = (abs(adjacent["closure_ratio"] - 1.0) < 1e-12
          and abs(adjacent["end_to_end"] - math.prod(adjacent["adjacent"])) < 1e-12
          and abs(skip - 1.44) < 1e-12 and abs(cube - 25.0 / 9.0) < 1e-12)
    return ok, f"R = {adjacent['closure_ratio']:.6f} adjacent, {skip:.2f} with a skip, {cube:.4f} through a cubed layer"


@check("C19", "moments of a cubed Gaussian layer: E[z1^3 z2^3] = 9r + 6r^3, E[z1^3 z2] = 3r, E[z^6] = 15")
def _c19():
    nodes, wts = np.polynomial.hermite_e.hermegauss(24)
    wts = wts / wts.sum()
    worst = 0.0
    for r in (0.3, 0.7, 0.95):
        z1 = nodes[:, None]
        z2 = r * nodes[:, None] + math.sqrt(1.0 - r * r) * nodes[None, :]
        weight = np.outer(wts, wts)
        worst = max(worst,
                    abs(float((weight * z1 ** 3 * z2 ** 3).sum()) - (9 * r + 6 * r ** 3)),
                    abs(float((weight * z1 ** 3 * z2).sum()) - 3 * r),
                    abs(float((weight * z2 ** 6).sum()) - 15.0))
    truth = efl.truths()["NL"]
    sample = truth.sample(CheckStream(19).normal((400000, efl.N_SCALES)))
    exact = truth.covariance()
    sd = np.sqrt(np.diag(exact))
    mc = float(np.abs((cl.second_moments(sample) - exact) / np.outer(sd, sd)).max())
    return worst < 1e-10 and mc < 0.03, (f"Gauss-Hermite quadrature agrees to {worst:.1e}; the NL design's exact "
                                         f"second moments match 400,000 samples to {mc:.3f} in correlation units")


@check("C20", "normal scores remove a monotone distortion exactly, so they separate the two causes of non-closure")
def _c20():
    x = CheckStream(20).normal((5000, 4)) @ np.array([[1.0, 0.8, 0.5, 0.2], [0, 0.6, 0.6, 0.4],
                                                      [0, 0, 0.6, 0.5], [0, 0, 0, 0.7]])
    bent = x.copy()
    bent[:, 1] = bent[:, 1] ** 3
    bent[:, 2] = np.exp(bent[:, 2])
    same = bool(np.array_equal(co.normal_scores(bent), co.normal_scores(x)))
    d_plain, d_bent = cl.sample_defects(x), cl.sample_defects(bent)
    return same and d_bent["normal_score"] == d_plain["normal_score"], (
        f"scores identical; normal-score D {d_bent['normal_score']:.4f} for both, raw D {d_plain['raw']:.4f} "
        f"against {d_bent['raw']:.4f} after cubing one layer and exponentiating another")


@check("C21", "the chain model is the EFL recursion: correlations multiply along scales; k = 15, 36, 8 at 8 scales")
def _c21():
    gains, injected = [1.3, -0.7, 2.0, 0.4, 1.1], [0.4, 0.9, 0.2, 0.6, 0.3]
    rows = cl.layered_rows(gains, injected)
    exact = rows @ rows.T
    sd = np.sqrt(np.diag(exact))
    adjacent = np.array([exact[k - 1, k] / (sd[k - 1] * sd[k]) for k in range(1, len(sd))])
    rebuilt = efl.chain_covariance(np.diag(exact), adjacent)
    gap = float(np.abs(rebuilt - exact).max())
    d = efl.N_SCALES
    counts = efl.PARAMETERS == {"chain": d + (d - 1), "full": d * (d + 1) // 2, "diagonal": d}
    factor = np.linalg.cholesky(exact)
    fitted = efl.fit("chain", math.sqrt(exact.shape[0]) * factor.T)          # signals with these exact moments
    regret = dl.gaussian_regret_bits(exact, fitted)
    return gap < 1e-12 and counts and abs(regret) < 1e-12, (
        f"an adjacent chain with one negative gain is rebuilt from its variances and adjacent correlations to "
        f"{gap:.1e}; chain fit to its exact moments has regret {regret:.1e}")


@check("C22", "Gaussian regret is zero at the truth and equals its closed form for a scaled identity")
def _c22():
    cov = efl.truths()["ADJ"].covariance()
    zero = dl.gaussian_regret_bits(cov, cov)
    d, c = 5, 1.7
    want = 0.5 * d * (1.0 / c - 1.0 + math.log(c)) / math.log(2.0)
    got = dl.gaussian_regret_bits(np.eye(d), c * np.eye(d))
    return abs(zero) < 1e-12 and abs(got - want) < 1e-12, f"at the truth {zero:.1e}; scaled identity {got:.6f} (closed form {want:.6f})"


# ----------------------------------------------------------------------------------------------
# Coherence estimator
# ----------------------------------------------------------------------------------------------

@check("C23", "per-frequency coherence recovers known values through arbitrary gains: no normalization needed")
def _c23():
    rng = CheckStream(23)
    n = 1024 * SEG
    x = rng.normal(n)
    parts, ok = [], True
    for gain, psi in ((2.0, 0.9), (0.5, 0.6), (30.0, 0.75)):
        y = gain * x + math.sqrt(gain * gain * (1.0 / psi - 1.0)) * rng.normal(n)
        got = co.band_mean(co.welch_msc(x, y, SEG))
        wide = co.broadband(x, y)
        ok &= abs(got - psi) < 0.01 and abs(wide - psi) < 0.01
        parts.append(f"gain {gain:g}: true {psi}, per-frequency mean {got:.4f}, broadband {wide:.4f}")
    return ok, "; ".join(parts)


@check("C24", "K segments put a floor of 1/K under zero coherence; the null threshold is 1 - alpha^(1/(K-1))")
def _c24():
    rng = CheckStream(24)
    parts, ok = [], True
    for k in (8, 64):
        values = [co.welch_msc(rng.normal(k * SEG), rng.normal(k * SEG), SEG)[1:-1] for _ in range(256 // k * 4)]
        pooled = np.concatenate(values)
        mean, q95 = float(pooled.mean()), float(np.quantile(pooled, 0.95))
        ok &= (abs(mean * k - 1.0) < 0.1 and abs(co.debias(mean, k)) < 0.01
               and abs(q95 - co.null_threshold(k)) < 0.1 * co.null_threshold(k))
        parts.append(f"K = {k}: mean {mean:.4f} (1/K = {1.0 / k:.4f}), 95 % point {q95:.4f} "
                     f"(formula {co.null_threshold(k):.4f})")
    ok &= abs(co.null_threshold(8) - (1.0 - 0.05 ** (1.0 / 7.0))) < 1e-15
    return ok, "; ".join(parts)


@check("C25", "a linearly filtered copy is fully coherent (carries no information); a time offset hides coherence")
def _c25():
    n = 512 * SEG
    x = CheckStream(25).normal(n)
    y = np.convolve(x, 0.3 * 0.7 ** np.arange(64))[:n]
    aligned = co.band_mean(co.welch_msc(x, y, SEG))
    shifted = co.band_mean(co.welch_msc(x, np.roll(y, SEG // 8), SEG))
    return aligned > 0.99 and shifted < 0.9, f"filtered copy reads {aligned:.4f}; offset by 32 samples reads {shifted:.4f}"


@check("C26", "self-similar cascade: pooled coherence deficits 1/n are not summable, fixed-frequency ones are")
def _c26():
    levels = 10
    cov = np.minimum.outer(np.arange(1, levels + 1), np.arange(1, levels + 1)).astype(float)
    deficits = max(abs(cl.coherence(cov, n - 1, n) - (n - 1) / n) for n in range(2, levels + 1))
    pooled = math.prod(cl.coherence(cov, n - 1, n) for n in range(2, levels + 1))
    w = [co.cascade_level_weight(j) for j in range(1, 60)]
    floor = w[0] / sum(w)
    ok = (deficits < 1e-12 and abs(pooled - 1.0 / levels) < 1e-12 and abs(w[0] - 2.0) < 1e-12
          and abs(w[20] / w[19] - 0.5) < 1e-6 and abs(floor - 0.3127) < 5e-5)
    return ok, (f"pooled product over {levels} levels {pooled:.4f} = 1/{levels} and falling; at one frequency the "
                f"level weights halve and the product keeps a floor of {floor:.4f}")


# ----------------------------------------------------------------------------------------------
# Scores, handles, streams, isolation
# ----------------------------------------------------------------------------------------------

@check("C27", "regret is zero for the truth; the two-part length is data bits plus (k/2) log2(symbols)")
def _c27():
    strings = models.all_strings(10)
    law = models.string_law(AM, 10, strings)
    zero, other = dl.regret_bits(law, AM, strings), dl.regret_bits(law, BIN, strings)
    data = np.array([[0, 0, 1], [0, 1, 1]])
    half = models.Binomial(0.5)
    two_part = dl.two_part_bits(half, data)
    want = 6.0 + 0.5 * 1 * math.log2(6.0)
    rate = dl.asymptotic_regret(3, 3200 * 16)
    ok = (abs(zero) < 1e-14 and other > 1e-3 and abs(two_part - want) < 1e-12
          and abs(rate - 3.0 / (2.0 * 51200.0 * math.log(2.0))) < 1e-18)
    return ok, f"regret of the truth {zero:.1e}, of a binomial {other:.4f}; two-part length of 6 fair symbols {two_part:.4f} bits"


@check("C28", "a handle edits one mechanism: the true model, edited by the true change, is the changed generator")
def _c28():
    strings = models.all_strings(10)
    sharper = AM.with_temperature_scaled(0.5)
    target = models.AttentionCascade.from_host(host.EMACascade(rho=0.3, T=0.3, bias=-0.2))
    shorter = AM.with_rho_scaled(2.0)
    capped = AM.with_rho_scaled(10.0).rho
    law_t = models.string_law(target, 10, strings)
    law_r = models.string_law(models.AttentionCascade.from_host(host.EMACascade(rho=0.6, T=0.6, bias=-0.2)), 10, strings)
    edited = max(dl.regret_bits(law_t, sharper, strings), dl.regret_bits(law_r, shorter, strings))
    unedited = min(dl.regret_bits(law_t, AM, strings), dl.regret_bits(law_r, AM, strings))
    ok = (abs(sharper.T - AM.T / 2.0) < 1e-15 and abs(sharper.bias - AM.bias) < 1e-15 and sharper.rho == AM.rho
          and capped == 0.98 and edited < 1e-14 and unedited > 1e-3
          and math.isinf(models.AttentionCascade(0.3, 0.0, 0.2).T) and AM.with_coupling(0.0).coupling == 0.0)
    return ok, f"edited model regret {edited:.1e} against the changed generator; unedited {unedited:.4f} bits per symbol"


@check("C29", "streams are deterministic, distinct by purpose, index and replicate, and both generators are sound")
def _c29():
    parts, ok = [], True
    for kind in streams.KINDS:
        base = streams.stream(kind, "train", 9, 0).uniform(4096)             # index 9 is outside every design
        again = streams.stream(kind, "train", 9, 0).uniform(4096)
        others = [streams.stream(kind, "train", 9, 1).uniform(4096), streams.stream(kind, "train", 8, 0).uniform(4096),
                  streams.stream(kind, "post", 9, 0).uniform(4096)]
        ok &= bool(np.array_equal(base, again)) and all(not np.array_equal(base, o) for o in others)
        u = streams.stream(kind, "train", 9, 2).uniform(200000)
        z = streams.stream(kind, "layered", 9, 2).normal(200000)
        lag = max(abs(float(np.corrcoef(u[:-s], u[s:])[0, 1])) for s in (1, 2, 64, 4096))
        ok &= (abs(u.mean() - 0.5) < 0.005 and abs(u.var() - 1.0 / 12.0) < 0.002 and abs(z.mean()) < 0.01
               and abs(z.var() - 1.0) < 0.02 and lag < 0.02)
        parts.append(f"{kind}: uniform mean {u.mean():.4f}, variance {u.var():.4f}, largest serial correlation "
                     f"{lag:.4f}; normal variance {z.var():.4f}")
    prov = streams.provenance("pcg")
    ok &= prov["generator"] == "numpy.random.PCG64" and streams.provenance("chaos")["generator"].endswith("ChaosRNG")
    return ok, "; ".join(parts)


@check("C30", "isolation: only host.py reads kgirl, nothing under src/ is modified, and the host does not know the package")
def _c30():
    importers = []
    for path in sorted(host.PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            if any(n == "kgirl" or n.startswith("kgirl.") for n in names):
                importers.append(str(path.relative_to(host.PACKAGE_ROOT)))
    mentions = [str(p.relative_to(host.REPO_ROOT)) for p in sorted((host.REPO_ROOT / "src").rglob("*.py"))
                if "efl_iwme_cascade" in p.read_text(encoding="utf-8", errors="ignore")]
    untouched = host.host_source_untouched()
    ok = sorted(set(importers)) == ["host.py"] and not mentions and untouched
    return ok, (f"modules importing kgirl: {sorted(set(importers))}; files under src/ naming this package: "
                f"{len(mentions)}; src/ unmodified relative to its commit: {untouched}")


def summary() -> Dict[str, object]:
    results = run_all()
    return {"passed": sum(r.passed for r in results), "total": len(results),
            "failed": [r.id for r in results if not r.passed]}

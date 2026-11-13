"""Where the instrument stops working, measured. Exploratory: nothing here is registered, there is no
hypothesis and no pass or fail. The numbers describe limits of the registered tests.

Run from `research/`:    python -m efl_iwme_cascade limits

Two questions.

1. Closure margin (Part 1). Hypothesis A5 reads the residual closure defect from the fit of replicate 0
   only. `closure_margin` reads it from the fits of all 12 replicates of a finished run, so the reader
   can see how much of A5 depends on that one draw.

2. Classifier cases (Part 2). The registered truths cover a linear adjacent chain, a positive skip
   coupling, and monotone distortions of the layers of a Gaussian chain. `classifier_cases` runs the same
   classification on four chains outside that set:

       QUAD      adjacent coupling, one link partly quadratic (a non-monotone coupling)
       CUBE      adjacent coupling, one link cubic with fresh noise after it (a nonlinear coupling,
                 as opposed to a distorted read-out of a Gaussian chain)
       HEAVY     the linear adjacent chain with heavy-tailed (Laplace) innovations
       NEGSKIP   a skip coupling of opposite sign, −0.6 from two scales up
       NLSKIP    the registered skip chain (+0.6) with its odd layers cubed: both causes at once

   The first three have adjacent coupling only, so the label "non-adjacent coupling" is wrong for them.
   The last two have a skip coupling, so "non-adjacent coupling" is the right label for them.

Streams are PCG64 under indices that no registered or replication run uses (10 and above).
"""

from __future__ import annotations

import json
import math
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from . import models, streams
from .experiments import partial_observability as part1
from .metrics import closure as cl
from .metrics import coherence as co
from .models import efl

Array = np.ndarray

CASES = ("QUAD", "CUBE", "HEAVY", "NEGSKIP", "NLSKIP")
CASE_NAMES = {
    "QUAD": "adjacent coupling, link 4 is v + 0.7 (v^2 - 1)",
    "CUBE": "adjacent coupling, link 4 is v^3 / sqrt(15), fresh noise after it",
    "HEAVY": "linear adjacent chain, Laplace innovations",
    "NEGSKIP": "skip coupling -0.6 from two scales up",
    "NLSKIP": "registered skip chain (+0.6), odd layers cubed",
}
ADJACENT_ONLY = ("QUAD", "CUBE", "HEAVY")
NONLINEAR_LINK = 4
QUAD_WEIGHT = 0.7
NEG_SKIP = -0.6
SAMPLE = 2560
REPLICATES = 12
SURROGATES = 200
POPULATION = 1_000_000
FIRST_INDEX = 10


def _laplace(uniforms: Array) -> Array:
    """Unit-variance Laplace variates from uniforms by inverse CDF."""
    centred = np.asarray(uniforms, dtype=float) - 0.5
    size = np.minimum(np.abs(centred), 0.5 - 1e-15)            # keeps the logarithm finite at u = 0
    return -np.sign(centred) * np.log1p(-2.0 * size) / math.sqrt(2.0)


def simulate(case: str, source) -> Callable[[int], Array]:
    """A sampler `draw(count)` of the eight scale signals of one case, on the registered ADJ constants."""
    if case not in CASES:
        raise KeyError(f"unknown case {case!r}; choose from {CASES}")
    injected = efl.injected_variances()
    adj_sd = np.sqrt(np.diag(efl.truths()["ADJ"].covariance()))

    def draw(count: int) -> Array:
        if case == "NLSKIP":
            rows = efl.truths()["SKIP"].rows()
            z = source.normal((count, efl.N_SCALES)) @ rows.T / np.sqrt(np.diag(rows @ rows.T))
            return np.where((np.arange(efl.N_SCALES) % 2 == 1)[None, :], z ** 3, z)
        noise = _laplace(source.uniform((count, efl.N_SCALES))) if case == "HEAVY" else source.normal((count, efl.N_SCALES))
        x = np.empty((count, efl.N_SCALES))
        x[:, 0] = noise[:, 0]
        for k in range(1, efl.N_SCALES):
            gain, prev = efl.GAINS[k - 1], x[:, k - 1]
            if k == NONLINEAR_LINK and case in ("QUAD", "CUBE"):
                v = prev / adj_sd[k - 1]
                shaped = v + QUAD_WEIGHT * (v * v - 1.0) if case == "QUAD" else v ** 3 / math.sqrt(15.0)
                mean = gain * adj_sd[k - 1] * shaped
            else:
                mean = gain * prev
            if case == "NEGSKIP" and k >= 2:
                mean = mean + NEG_SKIP * x[:, k - 2]
            x[:, k] = mean + math.sqrt(injected[k - 1]) * noise[:, k]
        return x

    return draw


def classifier_cases(sample: int = SAMPLE, replicates: int = REPLICATES, surrogates: int = SURROGATES,
                     population: int = POPULATION, log: Optional[Callable[[str], None]] = None) -> Dict[str, Dict]:
    say = log if log is not None else (lambda _msg: None)
    out: Dict[str, Dict] = {}
    for ci, case in enumerate(CASES):
        index = FIRST_INDEX + ci
        big = simulate(case, streams.stream("pcg", "layered", index, 1000))(population)
        moments = {"raw": cl.second_moments(big), "normal_score": cl.second_moments(co.normal_scores(big))}
        del big
        ratios = {key: cl.closure_ratio(m) for key, m in moments.items()}
        pop = {key: cl.closure_defect(m) for key, m in moments.items()}
        defects, bands, labels = [], [], []
        for r in range(replicates):
            x = simulate(case, streams.stream("pcg", "layered", index, r))(sample)
            chain = efl.fit("chain", x)
            source = streams.stream("pcg", "surrogate", index, r)
            band = cl.surrogate_band(lambda: efl.draw(chain, source.normal((sample, efl.N_SCALES))), surrogates)
            found = cl.sample_defects(x)
            defects.append(found)
            bands.append(band)
            labels.append(cl.classify(found, band))
        counts = {label: labels.count(label) for label in (cl.ADJACENT_LINEAR, cl.NONLINEAR_ADJACENT, cl.NON_ADJACENT)}
        out[case] = {"name": CASE_NAMES[case], "adjacent_only": case in ADJACENT_ONLY,
                     "large_sample": {"rows": population, "closure_ratio": ratios, "defect": pop},
                     "sample_defect": defects, "band": bands, "classified": labels, "counts": counts}
        say(f"{case}: done")
    return out


def closure_margin(config_name: str, log: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """Residual closure defects of B, MC1 and AM on the attention-memory truths for every replicate of a
    finished Part 1 run. Uses the run's stage cache when it is present and refits otherwise."""
    say = log if log is not None else (lambda _msg: None)
    config = part1.load_config(config_name)
    strings = models.all_strings(config.depth)
    truths = part1.ground_truths()
    out: Dict[str, object] = {"config": config.name, "truths": {}}
    for gi, g in enumerate(truths):
        own = part1.OWN_FAMILY[g]
        if g not in part1.MEMORY_TRUTHS:
            continue
        law = models.string_law(truths[g], config.depth, strings)
        raw = cl.closure_defect(part1.raw_covariance(strings, law))
        _, (_, fits, _) = part1._truth_regrets((config, gi, g, True))
        rows: List[Dict[str, float]] = []
        for kept in fits:
            families = sorted({"B", "MC1", "AM", own})
            rows.append({f: cl.closure_defect(part1.residual_covariance(kept[f], strings, law)) for f in families})
        restored = sum(row[own] < part1.CLOSURE_RESTORED and row[own] < raw / part1.CLOSURE_SHRINK for row in rows)
        entry: Dict[str, object] = {"own_family": own, "raw_defect": raw, "replicates": rows,
                                    "own_family_restores": int(restored)}
        if g in part1.ATTENTION_TRUTHS:
            entry["rule_iii_holds"] = int(sum(row["B"] > part1.CLOSURE_SHRINK * row["AM"]
                                              and row["MC1"] > part1.CLOSURE_SHRINK * row["AM"] for row in rows))
        out["truths"][g] = entry
        say(f"{config.name} {g}: done")
    return out


def run(part1_configs=("part1_registered", "part1_replication_pcg"), log: Optional[Callable[[str], None]] = None) -> Dict:
    t_start = time.time()
    margins = [closure_margin(name, log) for name in part1_configs]
    cases = classifier_cases(log=log)
    return {"exploratory": True, "closure_margin": margins, "classifier_cases": cases,
            "design": {"sample": SAMPLE, "replicates": REPLICATES, "surrogates": SURROGATES, "population": POPULATION,
                       "streams": "pcg, indices 10 to 14 of purposes layered and surrogate"},
            "seconds": round(time.time() - t_start, 1)}


def render(res: Dict) -> str:
    out: List[str] = ["LIMITS  exploratory, not registered: no hypothesis, no pass or fail", ""]
    add = out.append
    add("1. CLOSURE MARGIN  residual closure defect D of fits at the largest N, every replicate")
    for margin in res["closure_margin"]:
        add(f"  run {margin['config']}")
        for g, entry in margin["truths"].items():
            own, rows = entry["own_family"], entry["replicates"]
            n = len(rows)
            add(f"    {g}: raw D {entry['raw_defect']:.3f}; own family {own} restores closure "
                f"(D < 0.10 and < raw/5) in {entry['own_family_restores']} of {n} replicates")
            names = list(rows[0])
            add("      " + f"{'family':<8}{'min':>9}{'median':>9}{'max':>9}")
            for f in names:
                vals = [row[f] for row in rows]
                add("      " + f"{f:<8}{min(vals):>9.4f}{float(np.median(vals)):>9.4f}{max(vals):>9.4f}")
            if "rule_iii_holds" in entry:
                ratios = [row["MC1"] / row["AM"] for row in rows]
                add(f"      rule A5 (iii), B and MC1 each above 5 x AM: holds in {entry['rule_iii_holds']} of {n} "
                    f"replicates; MC1 / AM ranges {min(ratios):.1f} to {max(ratios):.1f}")
    add("")
    d = res["design"]
    add(f"2. CLASSIFIER CASES  N = {d['sample']}, {d['replicates']} replicates, {d['surrogates']} surrogates; "
        f"large-sample values from {d['population']:,} rows")
    for case, c in res["classifier_cases"].items():
        big = c["large_sample"]
        add(f"  {case}  {c['name']}")
        add(f"    large-sample closure ratio raw {big['closure_ratio']['raw']:.4f}, normal-score "
            f"{big['closure_ratio']['normal_score']:.4f}; defect raw {big['defect']['raw']:.4f}, normal-score "
            f"{big['defect']['normal_score']:.4f}")
        add("    labels: " + ", ".join(f"'{label}' {count}" for label, count in c["counts"].items()))
        if c["adjacent_only"]:
            wrong = c["counts"][cl.NON_ADJACENT]
            add(f"    coupling here is adjacent only: 'non-adjacent coupling' is a wrong label, given in {wrong} of "
                f"{len(c['classified'])}")
        else:
            add(f"    a skip coupling is present: 'non-adjacent coupling' is the right label, given in "
                f"{c['counts'][cl.NON_ADJACENT]} of {len(c['classified'])}")
    add("")
    add(f"({res['seconds']} s)")
    return "\n".join(out)


def save(res: Dict, name: str = "limits_exploratory"):
    part1.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    js, txt = part1.RESULTS_DIR / f"{name}.json", part1.RESULTS_DIR / f"{name}.txt"
    js.write_text(json.dumps(part1.plain(res), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    txt.write_text(render(res) + "\n", encoding="utf-8")
    return js, txt

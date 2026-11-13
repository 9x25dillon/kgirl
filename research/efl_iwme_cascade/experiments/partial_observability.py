"""Part 1 — partial observability: attention–memory cascades against matched hidden-state models, with EFL
closure as the test of memory sufficiency and zero-shot scores under a change of conditions.

This is the registered Experiment A (Phase 4 A of IWME-PRIME-01). It implements PREREGISTRATION.md with
AMENDMENTS.md; every constant is a number from those documents and the decision rules in `_verdicts` are
theirs. Deviations are recorded in RESULTS.md, never absorbed here.

What a registered run can show (Level 0–1 once computed): that the comparison identifies the generating
mechanism, including where the attention–memory class must lose; how much data that takes; that the EFL
closure statistic detects insufficient memory; that a model which generates the observed law need not
predict the law after conditions change. What it cannot show: that measured systems are attention–memory
systems (Level 2). That needs measured data (experiments/real_data.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import host, models, streams
from ..metrics import closure as cl
from ..metrics import description_length as dl
from ..metrics.interventions import Intervention, apply, catch_up
from ..models import cascade as strings_mod
from ..models.iwme import AttentionCascade

Array = np.ndarray
CONFIG_DIR = host.PACKAGE_ROOT / "configs"
RESULTS_DIR = host.PACKAGE_ROOT / "results"


# ----------------------------------------------------------------------------------------------
# Design
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    name: str
    stream: str
    depth: int
    sizes: Tuple[int, ...]
    replicates: int
    post_sizes: Tuple[int, ...]
    post_replicates: int
    registered: bool

    @property
    def need(self) -> int:
        """Replicates a paired comparison must win: 11 of 12 in the registered design."""
        return math.ceil(self.replicates * 11 / 12)


def load_config(name: str) -> Config:
    path = Path(name) if name.endswith(".json") else CONFIG_DIR / f"{name}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("part") != 1:
        raise ValueError(f"{path.name} is not a Part 1 configuration")
    return Config(name=raw["name"], stream=raw["stream"], depth=int(raw["depth"]), sizes=tuple(raw["sizes"]),
                  replicates=int(raw["replicates"]), post_sizes=tuple(raw["post_sizes"]),
                  post_replicates=int(raw["post_replicates"]), registered=bool(raw["registered"]))


G4_TABLES = ((0.50,), (0.60, 0.40), (0.70, 0.50, 0.45, 0.35),
             (0.80, 0.35, 0.60, 0.25, 0.70, 0.45, 0.30, 0.65))

TRUTH_NAMES = {
    "G0": "binomial, no memory",
    "G1": "hidden Markov cascade",
    "G2": "attention-memory above threshold",
    "G3": "attention-memory below threshold",
    "G4": "delay window (order-3 Markov)",
}
OWN_FAMILY = {"G0": "B", "G1": "HMM2", "G2": "AM", "G3": "AM", "G4": "MC3"}
MEMORY_TRUTHS = ("G1", "G2", "G3", "G4")
ATTENTION_TRUTHS = ("G2", "G3")
CONVENTIONAL_TRUTHS = ("G1", "G4")

RECOVERY_FACTOR = 5.0          # A1
CLOSURE_INFORMATIVE = 0.10     # A5 (ii)
CLOSURE_RESTORED = 0.10
CLOSURE_SHRINK = 5.0
ZERO_SHOT_FACTOR = 0.1         # A6, A7


def ground_truths() -> Dict[str, models.Model]:
    return {
        "G0": models.Binomial(0.7),
        "G1": models.HiddenMarkov(host.MarkovCascade()),
        "G2": AttentionCascade.from_host(host.EMACascade()),
        "G3": AttentionCascade.from_host(host.EMACascade(rho=0.15, T=0.4, bias=0.0)),
        "G4": models.markov_from_table(3, G4_TABLES),
    }


def _g4_after() -> models.MarkovChain:
    tables = [list(t) for t in G4_TABLES]
    tables[3][0] = 0.20
    return models.markov_from_table(3, tables)


def _set_lll(model: models.MarkovChain) -> models.MarkovChain:
    top = np.array(model.tables[3], dtype=float)
    top[0] = 0.20
    return models.MarkovChain(3, model.tables[:3] + (top,))


def interventions() -> Tuple[Intervention, ...]:
    """I1–I3 as registered; I4–I5 from Amendment 2d. The order fixes the post-intervention stream index."""
    truths = ground_truths()
    return (
        Intervention("I1", "G2", "T 0.8 -> 0.4 (attention sharpened through the threshold)",
                     AttentionCascade.from_host(host.EMACascade(rho=0.4, T=0.4, bias=0.3)),
                     "AM", lambda m: m.with_temperature_scaled(0.5)),
        Intervention("I2", "G3", "rho 0.15 -> 0.60 (memory shortened)",
                     AttentionCascade.from_host(host.EMACascade(rho=0.6, T=0.4, bias=0.0)),
                     "AM", lambda m: m.with_rho_scaled(4.0)),
        Intervention("I3", "G4", "table entry LLL 0.80 -> 0.20",
                     _g4_after(), "MC3", _set_lll),
        Intervention("I4", "G2", "memory actuation severed, coupling 1 -> 0",
                     truths["G2"].with_coupling(0.0), "AM", lambda m: m.with_coupling(0.0)),
        Intervention("I5", "G3", "memory actuation severed, coupling 1 -> 0",
                     truths["G3"].with_coupling(0.0), "AM", lambda m: m.with_coupling(0.0)),
    )


SUBSTRATE_INTERVENTIONS = ("I4", "I5")           # scored by A7; the others by A6


# ----------------------------------------------------------------------------------------------
# Closure of a model's residual walk under an exact law
# ----------------------------------------------------------------------------------------------

def raw_covariance(strings: Array, law: Array) -> Array:
    return cl.walk_covariance(strings_mod.signs(strings), law)


def residual_covariance(model, strings: Array, law: Array) -> Array:
    """Covariance, under `law`, of the walk of residuals s_k − E_model[s_k given the prefix]."""
    return cl.walk_covariance(strings_mod.signs(strings) - (2.0 * model.cond_left(strings) - 1.0), law)


# ----------------------------------------------------------------------------------------------
# Stages, each cached so that an interrupted run resumes
# ----------------------------------------------------------------------------------------------

def _fingerprint(config: Config) -> str:
    """Hash of the configuration and of every source file the numbers depend on."""
    h = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode())
    for sub in ("models", "metrics", "experiments"):
        for path in sorted((host.PACKAGE_ROOT / sub).glob("*.py")):
            h.update(path.read_bytes())
    h.update((host.PACKAGE_ROOT / "streams.py").read_bytes())
    return h.hexdigest()[:16]


def _cached(config: Config, key: str, compute: Callable[[], object], use_cache: bool):
    if not use_cache:
        return compute()
    path = RESULTS_DIR / ".cache" / config.name / f"{key}.pkl"
    stamp = _fingerprint(config)
    if path.exists():
        with path.open("rb") as fh:
            saved = pickle.load(fh)
        if saved.get("fingerprint") == stamp:
            return saved["value"]
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"fingerprint": stamp, "value": value}, fh)
    tmp.replace(path)
    return value


def _truth_regrets(task: Tuple[Config, int, str, bool]):
    """Stage 1 for one ground truth: regret of every family at every size and replicate, the fits at the
    largest size, and their two-part description length. A top-level function so that it can run in a
    worker process."""
    config, gi, g, use_cache = task

    def compute():
        depth, top = config.depth, max(config.sizes)
        strings = models.all_strings(depth)
        law = models.string_law(ground_truths()[g], depth, strings)
        regret: Dict[str, Dict[int, List[float]]] = {f: {n: [] for n in config.sizes} for f in models.FAMILIES}
        two_part: Dict[str, List[float]] = {f: [] for f in models.FAMILIES}
        fits: List[Dict[str, models.Model]] = []
        for r in range(config.replicates):
            idx = models.sample_indices(law, top, streams.stream(config.stream, "train", gi, r))
            kept: Dict[str, models.Model] = {}
            for n_str in config.sizes:
                sample = strings[idx[:n_str]]
                for fam in models.FAMILIES:
                    model = models.fit_family(fam, sample, kind=config.stream)
                    regret[fam][n_str].append(dl.regret_bits(law, model, strings))
                    if n_str == top:
                        kept[fam] = model
                        two_part[fam].append(dl.two_part_bits(model, sample))
            fits.append(kept)
        return regret, fits, two_part

    return g, _cached(config, f"stage1_{g}", compute, use_cache)


def _truth_projections(task: Tuple[Config, str, bool]):
    """Stage 2 for one ground truth: the best member of every family for the exact law."""
    config, g, use_cache = task

    def compute():
        strings = models.all_strings(config.depth)
        law = models.string_law(ground_truths()[g], config.depth, strings)
        return {f: models.project(f, law, strings, kind=config.stream) for f in models.FAMILIES}

    return g, _cached(config, f"stage2_{g}", compute, use_cache)


def _intervention_scores(task: Tuple[Config, int, bool, Dict]):
    """Stage 4 for one intervention: zero-shot regret of every fitted family and refits on new data."""
    config, ii, use_cache, fits_for_truth = task
    iv = interventions()[ii]

    def compute():
        strings = models.all_strings(config.depth)
        law_after = models.string_law(iv.after, config.depth, strings)
        zero: Dict[str, List[float]] = {f: [] for f in models.FAMILIES}
        for r in range(config.replicates):
            for fam in models.FAMILIES:
                zero[fam].append(dl.regret_bits(law_after, apply(iv, fam, fits_for_truth[r][fam]), strings))
        refit: Dict[str, Dict[int, float]] = {f: {} for f in models.FAMILIES}
        for n_post in config.post_sizes:
            acc: Dict[str, List[float]] = {f: [] for f in models.FAMILIES}
            for r in range(config.post_replicates):
                idx = models.sample_indices(law_after, n_post, streams.stream(config.stream, "post", ii, r))
                for fam in models.FAMILIES:
                    fitted = models.fit_family(fam, strings[idx], kind=config.stream)
                    acc[fam].append(dl.regret_bits(law_after, fitted, strings))
            for fam in models.FAMILIES:
                refit[fam][n_post] = float(np.mean(acc[fam]))
        target = float(np.mean(zero[iv.handle_family]))
        return {
            "truth": iv.truth, "change": iv.change, "handle_family": iv.handle_family,
            "before_to_after_bits": dl.regret_bits(law_after, ground_truths()[iv.truth], strings),
            "zero_shot": zero, "refit_mean": refit,
            "catch_up": {f: catch_up(refit[f], target, config.post_sizes) for f in models.FAMILIES},
        }

    return iv.name, _cached(config, f"stage4_{iv.name}", compute, use_cache)


def _map(fn: Callable, tasks: Sequence, jobs: int) -> List:
    if jobs <= 1:
        return [fn(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(fn, tasks))


def run(config: Config, log: Optional[Callable[[str], None]] = None, jobs: int = 1, use_cache: bool = False) -> Dict:
    """The whole experiment. `jobs` > 1 spreads the work over processes and `use_cache` resumes an
    interrupted run from results/.cache; the numbers depend on neither."""
    say = log if log is not None else (lambda _msg: None)
    t_start = time.time()
    depth = config.depth
    strings = models.all_strings(depth)
    truths = ground_truths()
    laws = {g: models.string_law(m, depth, strings) for g, m in truths.items()}

    # 1. regret of every family at every size, every replicate
    regret, top_fits, two_part = {}, {}, {}
    stage1 = _map(_truth_regrets, [(config, gi, g, use_cache) for gi, g in enumerate(truths)], jobs)
    for g, (reg_g, fits_g, bits_g) in stage1:
        regret[g], top_fits[g], two_part[g] = reg_g, fits_g, bits_g
    say(f"regrets done ({time.time() - t_start:.0f}s)")

    # 2. the N = infinity floor of every family on every truth
    projections = dict(_map(_truth_projections, [(config, g, use_cache) for g in truths], jobs))
    floors = {g: {f: dl.regret_bits(laws[g], projections[g][f], strings) for f in models.FAMILIES} for g in truths}
    say(f"floors done ({time.time() - t_start:.0f}s)")

    # 3. closure
    closure: Dict[str, Dict] = {}
    for g in truths:
        raw = raw_covariance(strings, laws[g])
        fitted = {f: residual_covariance(top_fits[g][0][f], strings, laws[g]) for f in models.FAMILIES}
        projected = {f: residual_covariance(projections[g][f], strings, laws[g]) for f in models.FAMILIES}
        closure[g] = {
            "raw_ratio": cl.closure_ratio(raw),
            "raw_defect": cl.closure_defect(raw),
            "raw_profile": cl.local_profile(raw).tolist(),
            "fitted_ratio": {f: cl.closure_ratio(c) for f, c in fitted.items()},
            "fitted_defect": {f: cl.closure_defect(c) for f, c in fitted.items()},
            "projected_defect": {f: cl.closure_defect(c) for f, c in projected.items()},
            "own_profile": cl.local_profile(fitted[OWN_FAMILY[g]]).tolist(),
        }
    say(f"closure done ({time.time() - t_start:.0f}s)")

    # 4. interventions: zero-shot regret against the changed generator, then refits on new data
    ivs = interventions()
    shifted = dict(_map(_intervention_scores,
                        [(config, ii, use_cache, top_fits[iv.truth]) for ii, iv in enumerate(ivs)], jobs))
    say(f"interventions done ({time.time() - t_start:.0f}s)")

    results = {
        "part": 1,
        "config": {**asdict(config), "sizes": list(config.sizes), "post_sizes": list(config.post_sizes),
                   "need": config.need},
        "provenance": streams.provenance(config.stream),
        "families": {f: models.PARAMETERS[f] for f in models.FAMILIES},
        "truths": {g: {"name": TRUTH_NAMES[g], "own_family": OWN_FAMILY[g], "parameters": models.describe(m)}
                   for g, m in truths.items()},
        "regret": regret,
        "floor": floors,
        "two_part_bits": {g: {f: float(np.mean(v)) for f, v in two_part[g].items()} for g in truths},
        "closure": closure,
        "interventions": shifted,
        "fitted_parameters": {g: {f: models.describe(top_fits[g][0][f]) for f in models.FAMILIES} for g in truths},
        "seconds": round(time.time() - t_start, 1),
    }
    results["verdicts"] = _verdicts(results, config)
    return results


# ----------------------------------------------------------------------------------------------
# Decision rules (PREREGISTRATION.md §8, Amendment 2d)
# ----------------------------------------------------------------------------------------------

def _wins(a: Sequence[float], b: Sequence[float]) -> int:
    """Replicates in which a has lower regret than b (paired)."""
    return int(sum(x < y for x, y in zip(a, b)))


def _zero_shot_rule(res: Dict, config: Config, names: Sequence[str]) -> Tuple[bool, List[str]]:
    lines, ok = [], True
    for name in names:
        iv = res["interventions"][name]
        handle, zero = iv["handle_family"], iv["zero_shot"]
        rivals = ["AM"] if name == "I3" else [f for f in models.FAMILIES if f != handle]
        wins = 0
        for r in range(config.replicates):
            wins += zero[handle][r] <= ZERO_SHOT_FACTOR * min(zero[f][r] for f in rivals)
        ok &= wins >= config.need
        lines.append(f"{name} on {iv['truth']}: edited {handle} mean regret {np.mean(zero[handle]):.2e}; "
                     f"best unedited rival {min(float(np.mean(zero[f])) for f in rivals):.2e}; "
                     f"rule met in {wins}/{config.replicates} replicates")
    return bool(ok), lines


def _verdicts(res: Dict, config: Config) -> Dict[str, Dict]:
    reg, need, top, depth = res["regret"], config.need, max(config.sizes), config.depth
    out: Dict[str, Dict] = {}

    def mean(g: str, f: str, n: int) -> float:
        return float(np.mean(reg[g][f][n]))

    # A1 recovery
    lines, ok = [], True
    for g, fam in OWN_FAMILY.items():
        bound = RECOVERY_FACTOR * dl.asymptotic_regret(models.PARAMETERS[fam], top * depth)
        got = mean(g, fam, top)
        ok &= got <= bound
        lines.append(f"{g} {fam}: mean regret {got:.2e}, bound {bound:.2e}")
    out["A1"] = {"title": "recovery", "met": bool(ok), "detail": lines}

    # A2 specificity
    lines, ok = [], True
    for g, fam in OWN_FAMILY.items():
        means = {f: mean(g, f, top) for f in models.FAMILIES}
        lowest = min(means, key=means.get)
        wins = {f: _wins(reg[g][fam][top], reg[g][f][top]) for f in models.FAMILIES if f != fam}
        ok &= lowest == fam and all(w >= need for w in wins.values())
        lines.append(f"{g}: lowest mean regret {lowest} (own family {fam}); paired wins of {fam}: "
                     + ", ".join(f"{f} {w}/{config.replicates}" for f, w in wins.items()))
    out["A2"] = {"title": "specificity", "met": bool(ok), "detail": lines}

    # A3 matched size
    lines, ok = [], True
    for g in ATTENTION_TRUTHS:
        per_size = {n: _wins(reg[g]["AM"][n], reg[g]["MC1"][n]) for n in config.sizes if n >= 100}
        big = {f: _wins(reg[g]["AM"][top], reg[g][f][top]) for f in ("MC2", "HMM2")}
        ok &= all(w >= need for w in per_size.values()) and all(w >= need for w in big.values())
        lines.append(f"{g}: AM over MC1 (both 3 parameters) " + ", ".join(f"N={n}: {w}" for n, w in per_size.items())
                     + f"; at N={top} AM over MC2 {big['MC2']}, over HMM2 {big['HMM2']} (of {config.replicates})")
    for g in CONVENTIONAL_TRUTHS:
        fam = OWN_FAMILY[g]
        w = _wins(reg[g][fam][top], reg[g]["AM"][top])
        ok &= w >= need
        lines.append(f"{g}: {fam} over AM at N={top}: {w}/{config.replicates}")
    out["A3"] = {"title": "matched size", "met": bool(ok), "detail": lines}

    # A4 sample efficiency
    lines, ok = [], True
    for g in ATTENTION_TRUTHS:
        for fam in ("MC2", "MC3", "HMM2"):
            lost = [n for n in config.sizes if not mean(g, "AM", n) < mean(g, fam, n)]
            ok &= not lost
            lines.append(f"{g}: AM below {fam} at every N" if not lost else f"{g}: AM not below {fam} at N = {lost}")
    out["A4"] = {"title": "sample efficiency", "met": bool(ok), "detail": lines}

    # A5 closure as memory sufficiency
    clo = res["closure"]
    lines = [f"(i) G0 raw defect {clo['G0']['raw_defect']:.2e}"]
    ok = clo["G0"]["raw_defect"] < 1e-9
    informative = []
    for g in MEMORY_TRUTHS:
        raw, own = clo[g]["raw_defect"], clo[g]["fitted_defect"][OWN_FAMILY[g]]
        if raw > CLOSURE_INFORMATIVE:
            informative.append(g)
            ok &= own < CLOSURE_RESTORED and own < raw / CLOSURE_SHRINK
            lines.append(f"(ii) {g}: raw defect {raw:.3f}, own family {OWN_FAMILY[g]} {own:.4f}")
        else:
            lines.append(f"(ii) {g}: raw defect {raw:.3f} is at or below {CLOSURE_INFORMATIVE}: uninformative")
    for g in ATTENTION_TRUTHS:
        d = clo[g]["fitted_defect"]
        ok &= d["B"] > CLOSURE_SHRINK * d["AM"] and d["MC1"] > CLOSURE_SHRINK * d["AM"]
        lines.append(f"(iii) {g}: residual defect B {d['B']:.3f}, MC1 {d['MC1']:.3f}, AM {d['AM']:.4f}")
    out["A5"] = {"title": "closure as memory sufficiency", "met": bool(ok), "detail": lines,
                 "informative": informative}

    # A6 change of conditions; A7 memory-substrate intervention
    registered_first = [n for n in res["interventions"] if n not in SUBSTRATE_INTERVENTIONS]
    met, lines = _zero_shot_rule(res, config, registered_first)
    out["A6"] = {"title": "change of conditions", "met": met, "detail": lines}
    met, lines = _zero_shot_rule(res, config, [n for n in res["interventions"] if n in SUBSTRATE_INTERVENTIONS])
    out["A7"] = {"title": "memory-substrate intervention", "met": met, "detail": lines}
    return out


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------

def render(res: Dict) -> str:
    cfg = res["config"]
    sizes, top = cfg["sizes"], max(cfg["sizes"])
    out: List[str] = []
    add = out.append
    label = "registered design" if cfg["registered"] else "REDUCED CONFIGURATION: supports no claim"
    add(f"PART 1  PARTIAL OBSERVABILITY  ({cfg['name']}: {label}; streams {cfg['stream']}; depth {cfg['depth']}, "
        f"{cfg['replicates']} replicates, {res['seconds']} s)")
    add("")
    add("FAMILIES  " + ", ".join(f"{f} ({k})" for f, k in res["families"].items()) + "   (free parameters)")
    add("")

    add("REGRET  mean over replicates, bits per symbol x 1e-4; * marks the family that contains the truth")
    for g, info in res["truths"].items():
        add(f"  {g}  {info['name']}:  {info['parameters']}")
        add(f"    {'family':<7}" + "".join(f"{'N=' + str(n):>10}" for n in sizes) + f"{'floor':>10}")
        for fam in res["families"]:
            star = "*" if fam == info["own_family"] else " "
            cells = "".join(f"{1e4 * float(np.mean(res['regret'][g][fam][n])):>10.2f}" for n in sizes)
            add(f"    {fam + star:<7}{cells}{1e4 * res['floor'][g][fam]:>10.2f}")
        add("")

    add(f"TOTAL DESCRIPTION LENGTH at N = {top}: data bits under the fit + (k/2) log2(symbols), "
        "mean over replicates, minus the smallest in the row")
    add(f"    {'truth':<6}" + "".join(f"{f:>10}" for f in res["families"]))
    for g in res["truths"]:
        row = res["two_part_bits"][g]
        best = min(row.values())
        add(f"    {g:<6}" + "".join(f"{row[f] - best:>10.1f}" for f in res["families"]))
    add("")

    add(f"CLOSURE  defect D = abs(log2 R(1,{cfg['depth']})) of the residual walk; fits at N = {top}, replicate 0")
    add(f"    {'truth':<6}{'raw R':>9}{'raw D':>9}" + "".join(f"{f:>9}" for f in res["families"]))
    for g in res["truths"]:
        c = res["closure"][g]
        add(f"    {g:<6}{c['raw_ratio']:>9.4f}{c['raw_defect']:>9.4f}"
            + "".join(f"{c['fitted_defect'][f]:>9.4f}" for f in res["families"]))
    add("  same, for the best member of each family (N = infinity projection)")
    for g in res["truths"]:
        c = res["closure"][g]
        add(f"    {g:<6}{'':>9}{'':>9}" + "".join(f"{c['projected_defect'][f]:>9.4f}" for f in res["families"]))
    add("")

    add("CHANGE OF CONDITIONS  regret against the changed generator, bits per symbol x 1e-4")
    for name, iv in res["interventions"].items():
        add(f"  {name} on {iv['truth']}: {iv['change']}.  Unchanged true model: {1e4 * iv['before_to_after_bits']:.1f}")
        add(f"    {'family':<14}{'zero-shot':>11}" + "".join(f"{'refit ' + str(n):>11}" for n in cfg["post_sizes"])
            + f"{'catch-up N':>12}")
        for fam in res["families"]:
            tag = fam + (" (edited)" if fam == iv["handle_family"] else "")
            reached = iv["catch_up"][fam]
            reached = ("> " + str(max(cfg["post_sizes"]))) if reached is None else str(reached)
            add(f"    {tag:<14}{1e4 * float(np.mean(iv['zero_shot'][fam])):>11.2f}"
                + "".join(f"{1e4 * iv['refit_mean'][fam][n]:>11.2f}" for n in cfg["post_sizes"])
                + f"{reached:>12}")
        add("    catch-up N: post-change strings a family needs, refitted from scratch, to match the edited model")
        add("")

    add("VERDICTS  (decision rules of PREREGISTRATION.md section 8 and Amendment 2d)")
    for key, v in res["verdicts"].items():
        add(f"  {key} {v['title']}: {'MET' if v['met'] else 'NOT MET'}")
        for line in v["detail"]:
            add(f"      {line}")
    met = sum(v["met"] for v in res["verdicts"].values())
    add(f"  {met} of {len(res['verdicts'])} met")
    return "\n".join(out)


def plain(x):
    """Results as JSON-ready builtins."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def save(res: Dict, name: Optional[str] = None) -> Tuple[Path, Path]:
    """Write results/<name>.json and results/<name>.txt."""
    name = name or res["config"]["name"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    js, txt = RESULTS_DIR / f"{name}.json", RESULTS_DIR / f"{name}.txt"
    js.write_text(json.dumps(plain(res), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    txt.write_text(render(res) + "\n", encoding="utf-8")
    return js, txt

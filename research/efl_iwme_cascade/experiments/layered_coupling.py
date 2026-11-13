"""Part 2 — the EFL arm on fully observed scale signals: adjacent coupling, skip coupling, nonlinear layers.

Implements PREREGISTRATION_PART2.md. Three questions, each with a truth where the EFL structure holds and
truths where it does not:

    B1  does assuming adjacent-scale coupling improve model selection when it holds?
    B2  is a coupling between non-adjacent scales identified?
    B3  is a nonlinear layer detected, and told apart from a skip coupling?
    B4  does the detector stay quiet when nothing is wrong?

Regret is exact (closed form against the true second moments), so no test set is sampled. Closure is read
in two versions, raw and normal-score, each against the band of the same statistic under the fitted chain.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import streams
from ..metrics import closure as cl
from ..metrics import coherence as co
from ..metrics import description_length as dl
from ..models import efl
from .partial_observability import CONFIG_DIR, RESULTS_DIR, plain

Array = np.ndarray
SEG = 256

RECOVERY_FACTOR = 5.0
POPULATION_DEFECT = 0.10
EXPECTED = {"ADJ": cl.ADJACENT_LINEAR, "SKIP": cl.NON_ADJACENT, "NL": cl.NONLINEAR_ADJACENT}
TRUTH_NAMES = {"ADJ": "adjacent coupling", "SKIP": "skip coupling (0.6 from two scales up)",
               "NL": "nonlinear layers (odd layers cubed), adjacent coupling"}


@dataclass(frozen=True)
class Config:
    name: str
    stream: str
    sizes: Tuple[int, ...]
    replicates: int
    surrogates: int
    welch_segments: int
    registered: bool

    @property
    def need(self) -> int:                        # "at least 11 of 12"
        return math.ceil(self.replicates * 11 / 12)

    @property
    def need_loose(self) -> int:                  # "at least 9 of 12": two 5 % bands must both behave
        return math.ceil(self.replicates * 9 / 12)


def load_config(name: str) -> Config:
    path = Path(name) if name.endswith(".json") else CONFIG_DIR / f"{name}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("part") != 2:
        raise ValueError(f"{path.name} is not a Part 2 configuration")
    return Config(name=raw["name"], stream=raw["stream"], sizes=tuple(raw["sizes"]), replicates=int(raw["replicates"]),
                  surrogates=int(raw["surrogates"]), welch_segments=int(raw["welch_segments"]),
                  registered=bool(raw["registered"]))


def _population(truth: efl.LayeredTruth) -> Dict[str, object]:
    cov, latent = truth.covariance(), truth.latent_correlation()
    return {
        "adjacent_coherence": cl.adjacent_coherences(cov),
        "end_to_end_coherence": cl.coherence(cov, 1, efl.N_SCALES),
        "closure_ratio_raw": cl.closure_ratio(cov),
        "closure_ratio_normal_score": cl.closure_ratio(latent),
        "defect_raw": cl.closure_defect(cov),
        "defect_normal_score": cl.closure_defect(latent),
        "floor": {m: dl.gaussian_regret_bits(cov, _project(m, cov)) for m in efl.MODELS},
    }


def _project(model: str, cov: Array) -> Array:
    """The best member of a model class for exact second moments (the N = infinity fit)."""
    if model == "full":
        return cov
    if model == "diagonal":
        return np.diag(np.diag(cov))
    var = np.diag(cov)
    adjacent = np.array([cov[k - 1, k] / math.sqrt(var[k - 1] * var[k]) for k in range(1, cov.shape[0])])
    return efl.chain_covariance(var, adjacent)


def _welch_report(config: Config) -> Dict[str, object]:
    """Descriptive: per-frequency coherence of one long ADJ realization against the broadband value."""
    truth = efl.truths()["ADJ"]
    x = truth.sample(streams.stream(config.stream, "welch", 0, 0).normal((config.welch_segments * SEG, efl.N_SCALES)))
    rows = []
    for k in range(1, efl.N_SCALES):
        msc = co.welch_msc(x[:, k - 1], x[:, k], SEG)[1:-1]
        rows.append({"link": k, "true": efl.PSI[k - 1], "per_frequency_mean": float(msc.mean()),
                     "per_frequency_min": float(msc.min()), "per_frequency_max": float(msc.max()),
                     "debiased_mean": co.debias(float(msc.mean()), config.welch_segments),
                     "broadband": co.broadband(x[:, k - 1], x[:, k])})
    return {"segments": config.welch_segments, "segment_length": SEG, "links": rows}


def run(config: Config, log: Optional[Callable[[str], None]] = None) -> Dict:
    say = log if log is not None else (lambda _msg: None)
    t_start = time.time()
    top = max(config.sizes)
    out_truths: Dict[str, Dict] = {}
    for name, truth in efl.truths().items():
        cov = truth.covariance()
        regret: Dict[str, Dict[int, List[float]]] = {m: {n: [] for n in config.sizes} for m in efl.MODELS}
        defects, bands, labels = [], [], []
        for r in range(config.replicates):
            x = truth.sample(streams.stream(config.stream, "layered", truth.index, r).normal((top, efl.N_SCALES)))
            for n in config.sizes:
                for model in efl.MODELS:
                    regret[model][n].append(dl.gaussian_regret_bits(cov, efl.fit(model, x[:n])))
            chain = efl.fit("chain", x)
            source = streams.stream(config.stream, "surrogate", truth.index, r)
            band = cl.surrogate_band(lambda: efl.draw(chain, source.normal((top, efl.N_SCALES))), config.surrogates)
            found = cl.sample_defects(x)
            defects.append(found)
            bands.append(band)
            labels.append(cl.classify(found, band))
        out_truths[name] = {"name": TRUTH_NAMES[name], "population": _population(truth), "regret": regret,
                            "sample_defect": defects, "band": bands, "classified": labels}
        say(f"{name}: done ({time.time() - t_start:.0f}s)")

    results = {
        "part": 2,
        "config": {**asdict(config), "sizes": list(config.sizes), "need": config.need, "need_loose": config.need_loose},
        "provenance": streams.provenance(config.stream),
        "models": dict(efl.PARAMETERS),
        "truths": out_truths,
        "welch": _welch_report(config),
        "seconds": round(time.time() - t_start, 1),
    }
    results["verdicts"] = _verdicts(results, config)
    return results


def _wins(a: Sequence[float], b: Sequence[float]) -> int:
    return int(sum(x < y for x, y in zip(a, b)))


def _verdicts(res: Dict, config: Config) -> Dict[str, Dict]:
    top = max(config.sizes)
    t = res["truths"]
    out: Dict[str, Dict] = {}

    def mean(truth: str, model: str, n: int) -> float:
        return float(np.mean(t[truth]["regret"][model][n]))

    def count(truth: str) -> int:
        return int(sum(label == EXPECTED[truth] for label in t[truth]["classified"]))

    # B1
    worse = [n for n in config.sizes if not mean("ADJ", "chain", n) < mean("ADJ", "full", n)]
    paired = {n: _wins(t["ADJ"]["regret"]["chain"][n], t["ADJ"]["regret"]["full"][n])
              for n in (80, top) if n in config.sizes}
    bound = RECOVERY_FACTOR * dl.asymptotic_regret(efl.PARAMETERS["chain"], top)
    got = mean("ADJ", "chain", top)
    out["B1"] = {"title": "adjacent coupling helps model selection",
                 "met": bool(not worse and all(w >= config.need for w in paired.values()) and got <= bound),
                 "detail": [("chain below full at every N" if not worse else f"chain not below full at N = {worse}"),
                            "paired wins of chain over full: " + ", ".join(f"N={n}: {w}/{config.replicates}"
                                                                           for n, w in paired.items()),
                            f"chain mean regret at N={top}: {got:.2e}, bound {bound:.2e}"]}

    # B2
    pop = t["SKIP"]["population"]
    hits = count("SKIP")
    full_better = mean("SKIP", "full", top) < mean("SKIP", "chain", top)
    out["B2"] = {"title": "a skip coupling is identified",
                 "met": bool(pop["defect_raw"] > POPULATION_DEFECT and pop["defect_normal_score"] > POPULATION_DEFECT
                             and hits >= config.need and full_better),
                 "detail": [f"population defect raw {pop['defect_raw']:.3f}, normal-score {pop['defect_normal_score']:.3f}",
                            f"classified '{EXPECTED['SKIP']}' in {hits}/{config.replicates}",
                            f"mean regret at N={top}: full {mean('SKIP', 'full', top):.2e}, chain {mean('SKIP', 'chain', top):.2e}"]}

    # B3
    pop = t["NL"]["population"]
    hits = count("NL")
    out["B3"] = {"title": "a nonlinear layer is detected and attributed",
                 "met": bool(pop["defect_raw"] > POPULATION_DEFECT and pop["defect_normal_score"] < 1e-9
                             and hits >= config.need_loose),
                 "detail": [f"population defect raw {pop['defect_raw']:.3f}, normal-score {pop['defect_normal_score']:.1e}",
                            f"classified '{EXPECTED['NL']}' in {hits}/{config.replicates}"]}

    # B4
    hits = count("ADJ")
    out["B4"] = {"title": "no false alarm", "met": bool(hits >= config.need_loose),
                 "detail": [f"classified '{EXPECTED['ADJ']}' in {hits}/{config.replicates}"]}
    return out


def render(res: Dict) -> str:
    cfg = res["config"]
    sizes, top = cfg["sizes"], max(cfg["sizes"])
    out: List[str] = []
    add = out.append
    label = "registered design" if cfg["registered"] else "REDUCED CONFIGURATION: supports no claim"
    add(f"PART 2  LAYERED COUPLING  ({cfg['name']}: {label}; streams {cfg['stream']}; {cfg['replicates']} replicates, "
        f"{res['seconds']} s)")
    add("")
    add("MODELS  " + ", ".join(f"{m} ({k})" for m, k in res["models"].items()) + "   (free parameters)")
    add("")
    for name, tr in res["truths"].items():
        pop = tr["population"]
        add(f"  {name}  {tr['name']}")
        add("    adjacent coherence, exact: " + ", ".join(f"{v:.3f}" for v in pop["adjacent_coherence"]))
        add(f"    end-to-end coherence {pop['end_to_end_coherence']:.4f};  closure ratio raw {pop['closure_ratio_raw']:.4f}, "
            f"normal-score {pop['closure_ratio_normal_score']:.4f};  defect raw {pop['defect_raw']:.4f}, "
            f"normal-score {pop['defect_normal_score']:.4f}")
        add("    regret, mean over replicates, bits per sample x 1e-3")
        add(f"    {'model':<10}" + "".join(f"{'N=' + str(n):>9}" for n in sizes) + f"{'floor':>9}")
        for model in res["models"]:
            cells = "".join(f"{1e3 * float(np.mean(tr['regret'][model][n])):>9.2f}" for n in sizes)
            add(f"    {model:<10}{cells}{1e3 * pop['floor'][model]:>9.2f}")
        add(f"    sample closure defect at N = {top} against the 95 % point under the fitted chain")
        add(f"    {'replicate':>10}{'raw D':>9}{'band':>9}{'score D':>10}{'band':>9}   classified")
        for r, (d, b, lab) in enumerate(zip(tr["sample_defect"], tr["band"], tr["classified"])):
            add(f"    {r:>10}{d['raw']:>9.3f}{b['raw']:>9.3f}{d['normal_score']:>10.3f}{b['normal_score']:>9.3f}   {lab}")
        add("")
    w = res["welch"]
    add(f"PER-FREQUENCY AGAINST BROADBAND  (ADJ, one realization, {w['segments']} segments of {w['segment_length']})")
    add(f"    {'link':>5}{'true':>8}{'band mean':>11}{'min':>8}{'max':>8}{'debiased':>10}{'broadband':>11}")
    for row in w["links"]:
        add(f"    {row['link']:>5}{row['true']:>8.3f}{row['per_frequency_mean']:>11.4f}{row['per_frequency_min']:>8.3f}"
            f"{row['per_frequency_max']:>8.3f}{row['debiased_mean']:>10.4f}{row['broadband']:>11.4f}")
    add("")
    add("VERDICTS  (decision rules of PREREGISTRATION_PART2.md section 5)")
    for key, v in res["verdicts"].items():
        add(f"  {key} {v['title']}: {'MET' if v['met'] else 'NOT MET'}")
        for line in v["detail"]:
            add(f"      {line}")
    met = sum(v["met"] for v in res["verdicts"].values())
    add(f"  {met} of {len(res['verdicts'])} met")
    return "\n".join(out)


def save(res: Dict, name: Optional[str] = None) -> Tuple[Path, Path]:
    name = name or res["config"]["name"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    js, txt = RESULTS_DIR / f"{name}.json", RESULTS_DIR / f"{name}.txt"
    js.write_text(json.dumps(plain(res), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    txt.write_text(render(res) + "\n", encoding="utf-8")
    return js, txt


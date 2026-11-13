"""Measured data: the same comparisons on signals that did not come from this package.

Two entry points, one per part.

    analyze_points(points)     points of [0, 1) (event positions or times scaled to the unit interval).
                               Symbolize, fit every family on the first part, score held-out code length on
                               the rest, and test each family's memory for sufficiency by the closure defect
                               of its residual walk against that family's own surrogate band.

    analyze_layers(signals)    one column per scale, one row per sample, zero mean. Adjacent coherence in
                               three versions (per frequency, broadband, normal-score), the closure ratio in
                               two, the registered classification, and held-out code length of the chain
                               against the unstructured Gaussian.

What would count against the theory here is written in PREREGISTRATION.md §9. A result on measured data is
the only evidence about the Level 2 claims; everything else in this package is about the instrument.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .. import models, streams
from ..metrics import closure as cl
from ..metrics import coherence as co
from ..models import cascade as strings_mod
from ..models import efl

Array = np.ndarray


def analyze_strings(train: Array, test: Array, surrogates: int = 200, kind: str = "chaos") -> Dict:
    train, test = np.asarray(train), np.asarray(test)
    depth = train.shape[1]
    if test.shape[1] != depth:
        raise ValueError("train and test strings must have the same depth")
    strings = models.all_strings(depth)
    rng = streams.stream(kind, "data_surrogate")
    report: Dict[str, Dict] = {}
    for fam in models.FAMILIES:
        model = models.fit_family(fam, train, kind=kind)
        bits = float(-models.log2_prob(model, test).mean() / depth)
        resid = strings_mod.signs(test) - (2.0 * model.cond_left(test) - 1.0)
        defect = cl.closure_defect(cl.walk_covariance(resid))
        law = models.string_law(model, depth, strings)
        null = []
        for _ in range(surrogates):
            sim = strings[models.sample_indices(law, test.shape[0], rng)]
            null.append(cl.closure_defect(cl.walk_covariance(strings_mod.signs(sim) - (2.0 * model.cond_left(sim) - 1.0))))
        band = float(np.quantile(null, 0.95))
        report[fam] = {"parameters": models.PARAMETERS[fam], "held_out_bits": bits, "closure_defect": defect,
                       "surrogate_95": band, "closure_restored": bool(defect <= band),
                       "fitted": models.describe(model)}
    return {"kind": "points", "depth": depth, "train": int(train.shape[0]), "test": int(test.shape[0]),
            "raw_closure_defect": cl.closure_defect(cl.walk_covariance(strings_mod.signs(test))),
            "families": report}


def analyze_points(points: Array, depth: int = 16, train_fraction: float = 0.8, surrogates: int = 200,
                   kind: str = "chaos") -> Dict:
    """Symbolize measured points of [0, 1) and compare the families on a first/last split."""
    strings = models.symbolize(points, depth)
    cut = int(round(train_fraction * strings.shape[0]))
    if cut < 2 or strings.shape[0] - cut < 2:
        raise ValueError("need at least two training and two test points")
    return analyze_strings(strings[:cut], strings[cut:], surrogates, kind)


def _gaussian_bits(cov: Array, data: Array) -> float:
    """Mean code length per sample, in bits, of zero-mean Gaussian `cov` on rows of `data`."""
    sign, logdet = np.linalg.slogdet(cov)
    quad = np.einsum("ij,ij->i", data @ np.linalg.inv(cov), data).mean()
    return float(0.5 * (cov.shape[0] * math.log(2.0 * math.pi) + logdet + quad) / math.log(2.0))


def analyze_layers(signals: Array, seg_len: int = 256, train_fraction: float = 0.8, surrogates: int = 200,
                   kind: str = "chaos") -> Dict:
    x = np.asarray(signals, dtype=float)
    if x.ndim != 2 or x.shape[1] < 3:
        raise ValueError("need a (samples, scales) array with at least three scales")
    x = x - x.mean(axis=0)
    n_scales = x.shape[1]
    segments = x.shape[0] // seg_len
    links = []
    scored = co.normal_scores(x)
    for k in range(1, n_scales):
        row = {"link": k, "broadband": co.broadband(x[:, k - 1], x[:, k]),
               "normal_score": co.broadband(scored[:, k - 1], scored[:, k])}
        if segments >= 2:
            msc = co.welch_msc(x[:, k - 1], x[:, k], seg_len)[1:-1]
            row.update({"per_frequency_mean": co.debias(float(msc.mean()), segments),
                        "per_frequency_min": float(msc.min()), "per_frequency_max": float(msc.max()),
                        "null_threshold": co.null_threshold(segments)})
        links.append(row)
    chain = efl.fit("chain", x)
    source = streams.stream(kind, "data_surrogate")
    band = cl.surrogate_band(lambda: efl.draw(chain, source.normal(x.shape)), surrogates)
    defects = cl.sample_defects(x)
    cut = int(round(train_fraction * x.shape[0]))
    held = {m: _gaussian_bits(efl.fit(m, x[:cut]), x[cut:]) for m in efl.MODELS} if x.shape[0] - cut >= 2 else {}
    return {"kind": "layers", "samples": int(x.shape[0]), "scales": n_scales, "segments": segments, "links": links,
            "closure_ratio": {"raw": cl.closure_ratio(cl.second_moments(x)),
                              "normal_score": cl.closure_ratio(cl.second_moments(scored))},
            "closure_defect": defects, "surrogate_95": band, "classified": cl.classify(defects, band),
            "held_out_bits": held}


def render(rep: Dict) -> str:
    if rep["kind"] == "points":
        out = [f"MEASURED POINTS  depth {rep['depth']}, {rep['train']} training strings, {rep['test']} test strings",
               f"  raw closure defect of the test strings: {rep['raw_closure_defect']:.4f}",
               f"  {'family':<7}{'k':>4}{'held-out bits':>15}{'closure D':>11}{'surrogate 95%':>15}  restored"]
        for fam, r in rep["families"].items():
            out.append(f"  {fam:<7}{r['parameters']:>4}{r['held_out_bits']:>15.5f}{r['closure_defect']:>11.4f}"
                       f"{r['surrogate_95']:>15.4f}  {'yes' if r['closure_restored'] else 'no'}")
        best = min(rep["families"], key=lambda f: rep["families"][f]["held_out_bits"])
        out.append(f"  lowest held-out code length: {best}")
        return "\n".join(out)
    out = [f"MEASURED LAYERS  {rep['samples']} samples, {rep['scales']} scales, {rep['segments']} Welch segments",
           f"  {'link':>5}{'broadband':>11}{'normal-score':>14}{'per-frequency':>15}{'min':>8}{'max':>8}"]
    for row in rep["links"]:
        tail = (f"{row['per_frequency_mean']:>15.4f}{row['per_frequency_min']:>8.3f}{row['per_frequency_max']:>8.3f}"
                if "per_frequency_mean" in row else f"{'-':>15}{'-':>8}{'-':>8}")
        out.append(f"  {row['link']:>5}{row['broadband']:>11.4f}{row['normal_score']:>14.4f}{tail}")
    d, b = rep["closure_defect"], rep["surrogate_95"]
    out.append(f"  closure ratio raw {rep['closure_ratio']['raw']:.4f}, normal-score {rep['closure_ratio']['normal_score']:.4f}")
    out.append(f"  closure defect raw {d['raw']:.4f} (95 % point {b['raw']:.4f}), normal-score {d['normal_score']:.4f} "
               f"(95 % point {b['normal_score']:.4f})")
    out.append(f"  classified: {rep['classified']}")
    if rep["held_out_bits"]:
        out.append("  held-out code length, bits per sample: "
                   + ", ".join(f"{m} {v:.4f}" for m, v in rep["held_out_bits"].items()))
    return "\n".join(out)

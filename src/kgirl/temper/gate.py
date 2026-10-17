"""The model-swap gate: may a fine-tuned model replace its base? Standard library only; deterministic.

Every input is a measurement the user can inspect; the verdict is ACCEPT only if every gate passes, REJECT if
any gate fails, and INCOMPLETE if a required measurement is missing (an INCOMPLETE swap never happens).

    G0  preconditions   ≥ min_items held-out items; base and tuned paired on identical (item, seed) runs; one
                        served artifact (the GGUF Ollama will serve, by sha) behind every tuned row; no training
                        example contains a held-out item (3-word shingle containment ≥ 0.55, or equal text)
    G1  validity        the valid-decision rate is non-inferior: one-sided lower bound of Δ > −margin (0.02)
    G2  success         task success is superior: one-sided p < α (0.05). One run per item: exact McNemar on
                        the discordant pairs. Several seeds per item: paired test on per-item mean success, so
                        seeds are repeated measures of an item, not independent trials
    G3  families        no task family loses more than 0.05 success
    G4  shadow          on unscored real traffic, the share of parsed decisions that change stays ≤ 0.25

Evaluation must run at the temperatures the model is served at (Jev: 0.1, 0.5, 0.9), with chaos seeds, so the
gate sees the behaviour that will ship.

Planning: `required_items(δ)` is the paired-proportion sample size (Connor 1987). With 20 % of items discordant
and α 0.05 one-sided at 80 % power, a 5-point gain needs ≈ 490 items, 200 items detect ≈ 8 points, and 47
detect only ≈ 16.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable

_Z = NormalDist()
_WORD = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class GateConfig:
    alpha: float = 0.05
    validity_margin: float = 0.02
    family_drop: float = 0.05
    shadow_budget: float = 0.25
    min_items: int = 200
    containment: float = 0.55
    shingle_k: int = 3


# ----------------------------------------------------------------------------------------------- planning

def required_items(delta: float, discordance: float = 0.2, alpha: float = 0.05, power: float = 0.8) -> int:
    """Paired items needed to detect a success gain `delta` (one-sided), Connor (1987)."""
    if not 0 < delta < math.sqrt(discordance):
        raise ValueError("need 0 < delta < sqrt(discordance)")
    za, zb = _Z.inv_cdf(1 - alpha), _Z.inv_cdf(power)
    return math.ceil(((za * math.sqrt(discordance) + zb * math.sqrt(discordance - delta ** 2)) / delta) ** 2)


def minimum_detectable(n: int, discordance: float = 0.2, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest gain `n` paired items detect (bisection on required_items)."""
    lo, hi = 1e-6, math.sqrt(discordance) - 1e-9
    for _ in range(100):
        mid = (lo + hi) / 2
        if required_items(mid, discordance, alpha, power) > n:
            lo = mid
        else:
            hi = mid
    return hi


# ----------------------------------------------------------------------------------------------- statistics

def binom_upper_tail(k: int, n: int) -> float:
    """P(X ≥ k) for X ~ Binomial(n, ½), computed in log space."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k, n + 1)]
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs))


def paired_lower_bound(diffs: list[float], alpha: float) -> tuple[float, float]:
    """(mean, one-sided lower 1−α bound) of paired per-item differences (normal approximation)."""
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    return mean, mean - _Z.inv_cdf(1 - alpha) * math.sqrt(var / n)


def paired_superiority_p(diffs: list[float]) -> float:
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    if var == 0:
        return 0.0 if mean > 0 else 1.0
    return 1 - _Z.cdf(mean / math.sqrt(var / n))


# ----------------------------------------------------------------------------------------------- leakage

def _norm(text: str) -> str:
    return " ".join(w.lower() for w in _WORD.findall(text or ""))


def _shingles(text: str, k: int) -> set[str]:
    words = _norm(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def leaks(suite: dict[str, str], train: Iterable[str], cfg: GateConfig = GateConfig()) -> list[dict]:
    """Held-out items whose text appears in a training example: equal after normalization, or with at least
    `containment` of the item's shingles inside one example. An inverted index keeps this linear in the text."""
    index: dict[str, list[int]] = {}
    exact: dict[str, int] = {}
    for j, text in enumerate(train):
        exact.setdefault(_norm(text), j)
        for s in _shingles(text, cfg.shingle_k):
            index.setdefault(s, []).append(j)
    found = []
    for item, text in suite.items():
        if _norm(text) in exact:
            found.append({"item": item, "train_row": exact[_norm(text)], "containment": 1.0})
            continue
        sh = _shingles(text, cfg.shingle_k)
        if not sh:
            continue
        counts: dict[int, int] = {}
        for s in sh:
            for j in set(index.get(s, ())):
                counts[j] = counts.get(j, 0) + 1
        if counts:
            j, c = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
            if c / len(sh) >= cfg.containment:
                found.append({"item": item, "train_row": j, "containment": round(c / len(sh), 3)})
    return found


def train_text(row: dict) -> str:
    """The text of one `soup export-sft` row, whatever its role."""
    return " ".join(str(row.get(k, "")) for k in ("instruction", "context", "output", "system", "prompt", "completion"))


# ----------------------------------------------------------------------------------------------- the gate

def evaluate(rows: list[dict], cfg: GateConfig = GateConfig(), artifact: str | None = None,
             suite: dict[str, str] | None = None, train: list[str] | None = None, trained: bool = True,
             shadow: list[dict] | None = None) -> dict:
    """rows: {"item", "family", "arm": "base"|"tuned", "seed", "temperature", "valid", "success", "artifact"}."""
    gates: dict[str, dict] = {}
    by: dict[tuple[str, str], dict] = {}
    family: dict[str, str] = {}
    for r in rows:
        if r["arm"] not in ("base", "tuned"):
            raise ValueError(f"arm must be base or tuned: {r!r}")
        by.setdefault((r["arm"], str(r["item"])), {})[(r.get("seed"), r.get("temperature"))] = r
        family[str(r["item"])] = str(r.get("family", "-"))
    base_items = {i for a, i in by if a == "base"}
    tuned_items = {i for a, i in by if a == "tuned"}
    items = sorted(base_items & tuned_items)
    problems = []
    if base_items != tuned_items:
        problems.append(f"{len(base_items ^ tuned_items)} items are run in only one arm")
    unpaired = [i for i in items if set(by[("base", i)]) != set(by[("tuned", i)])]
    if unpaired:
        problems.append(f"{len(unpaired)} items have different (seed, temperature) runs per arm, e.g. {unpaired[0]}")
    if len(items) < cfg.min_items:
        problems.append(f"{len(items)} paired items < {cfg.min_items} (minimum detectable gain "
                        f"≈ {minimum_detectable(max(len(items), 2)):.2f} at 20 % discordance)")
    shas = {r.get("artifact") for r in rows if r["arm"] == "tuned"}
    if len(shas) != 1 or None in shas:
        problems.append(f"tuned rows name {len(shas - {None})} artifact(s); exactly one served artifact is required")
    elif artifact is not None and shas != {artifact}:
        problems.append(f"tuned rows were produced by {next(iter(shas))}, not the served artifact {artifact}")
    leak, unchecked = [], False
    if trained and (suite is None or train is None):
        unchecked = True
    elif trained:
        leak = leaks(suite, train, cfg)
        if leak:
            problems.append(f"{len(leak)} held-out items appear in the training data, e.g. {leak[0]}")
    gates["G0"] = {"pass": False if problems else (None if unchecked else True),
                   "problems": problems + (["leakage not checked: pass the suite and the training rows "
                                            "(or declare that the model was not trained)"] if unchecked else []),
                   "paired_items": len(items), "leaks": leak[:20],
                   "artifact": next(iter(shas)) if len(shas) == 1 else None}

    def per_item(arm, key):
        return {i: sum(bool(r[key]) for r in by[(arm, i)].values()) / len(by[(arm, i)]) for i in items}

    if items:
        vb, vt = per_item("base", "valid"), per_item("tuned", "valid")
        sb, st = per_item("base", "success"), per_item("tuned", "success")
        dv = [vt[i] - vb[i] for i in items]
        ds = [st[i] - sb[i] for i in items]
        mean_v, low_v = paired_lower_bound(dv, cfg.alpha)
        gates["G1"] = {"pass": low_v > -cfg.validity_margin, "base": sum(vb.values()) / len(items),
                       "tuned": sum(vt.values()) / len(items), "delta": mean_v, "lower_bound": low_v,
                       "margin": cfg.validity_margin}
        single = all(len(by[("base", i)]) == 1 for i in items)
        if single:
            b = sum(1 for i in items if sb[i] > st[i])
            c = sum(1 for i in items if st[i] > sb[i])
            p, method = binom_upper_tail(c, b + c), "exact McNemar"
        else:
            b = c = None
            p, method = paired_superiority_p(ds), "paired test on per-item mean success"
        disc = sum(1 for d in ds if d != 0) / len(items)
        gates["G2"] = {"pass": p < cfg.alpha and sum(ds) > 0, "base": sum(sb.values()) / len(items),
                       "tuned": sum(st.values()) / len(items), "delta": sum(ds) / len(items), "p": p,
                       "method": method, "base_only": b, "tuned_only": c, "discordance": disc,
                       "minimum_detectable": minimum_detectable(len(items), max(disc, 0.01)) if len(items) > 1 else None}
        fams: dict[str, list[float]] = {}
        for i, d in zip(items, ds):
            fams.setdefault(family[i], []).append(d)
        worst = min(fams.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
        gates["G3"] = {"pass": all(sum(v) / len(v) >= -cfg.family_drop for v in fams.values()),
                       "worst_family": worst[0], "worst_delta": sum(worst[1]) / len(worst[1]),
                       "families": {f: {"n": len(v), "delta": sum(v) / len(v)} for f, v in sorted(fams.items())}}
    else:
        for g in ("G1", "G2", "G3"):
            gates[g] = {"pass": None, "problems": ["no paired items"]}
    if shadow:
        changed = sum(1 for s in shadow if " ".join(str(s["base"]).split()) != " ".join(str(s["tuned"]).split()))
        rho = changed / len(shadow)
        gates["G4"] = {"pass": rho <= cfg.shadow_budget, "rho": rho, "budget": cfg.shadow_budget,
                       "decisions": len(shadow)}
    else:
        gates["G4"] = {"pass": None, "problems": ["no shadow run: decisions on unscored traffic were not compared"]}
    results = [g["pass"] for g in gates.values()]
    verdict = "REJECT" if False in results else "INCOMPLETE" if None in results else "ACCEPT"
    return {"verdict": verdict, "gates": gates, "config": cfg.__dict__, "trained": trained}


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

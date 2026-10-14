"""TA-ULS hyperparameter search with kgirl.evolve (chaos MAP-Elites over short, equal-budget training runs).

    genome      lr (log), warmup (log), min_lr / lr (log), weight decay, dropout,
                control_mult (meta/auto control width), kfp_rank as a fraction of d_model, entropy_target
    fitness     −best validation loss (nats/token) after `steps` optimizer steps: every candidate gets the
                same token budget, so the comparison is fair
    behaviour   (parameters in millions, α), where α = the learned meta/automatic control mix logged by
                training. The archive keeps the best recipe for every model size and every degree of
                reliance on the TA-ULS control paths. Read it as an ablation map, not one winner.

    python -m kgirl.tauls hpsearch --data runs/tauls/data --out runs/tauls/hp --preset small --steps 400 \\
                                   --generations 6 --batch 4 --init 12
    → runs/tauls/hp/archive.json (every elite) and champion.json (a TrainConfig: train --config champion.json)

Short runs rank recipes by early learning speed, which correlates with, but is not, the final quality
(L1 for transformer LMs; L2 for TA-ULS until checked at full length). Randomness is chaos (no PRNG).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kgirl.evolve.mapelites import Descriptor, Evaluation, MapElites
from kgirl.evolve.space import Param, Space

SPACE = Space([
    Param("lr", 1e-4, 3e-3, log=True),
    Param("warmup", 10, 400, log=True),
    Param("min_lr_ratio", 0.02, 0.5, log=True),
    Param("weight_decay", 0.0, 0.2),
    Param("dropout", 0.0, 0.15),
    Param("control_mult", 1.0, 4.0),
    Param("kfp_rank_frac", 0.1, 0.5),
    Param("entropy_target", 0.4, 1.0),
])


def recipe(g: dict[str, float], base: dict, d_model: int, steps: int) -> dict:
    """TrainConfig dict for genome g on top of `base` (data, out, preset, batch, device …)."""
    model = {**base.get("model", {}), "control_mult": round(g["control_mult"], 3),
             "kfp_rank": max(4, int(round(g["kfp_rank_frac"] * d_model))), "entropy_target": round(g["entropy_target"], 3),
             "dropout": round(g["dropout"], 3)}
    lr = float(g["lr"])
    return {**base, "model": model, "steps": steps, "lr": lr, "min_lr": lr * float(g["min_lr_ratio"]),
            "warmup": int(round(g["warmup"])), "weight_decay": round(g["weight_decay"], 4),
            "eval_every": max(1, steps // 2), "save_every": steps, "keep": 1}   # keep=0 would keep all ([:-0])


def _param_range(base: dict, vocab: int) -> tuple[float, float]:
    from .model import TAULS
    from .train import TrainConfig, model_config
    out = []
    for cm in (1.0, 4.0):
        tc = TrainConfig.from_dict({**base, "model": {**base.get("model", {}), "control_mult": cm}})
        out.append(TAULS(model_config(tc, vocab)).num_params() / 1e6)
    return min(out) * 0.95, max(out) * 1.05


def make_evaluator(base: dict, out_root: Path, steps: int, log: Callable = print):
    from .data import TokenData
    from .train import TrainConfig, model_config, train
    vocab = TokenData(base["data"]).meta["vocab_size"]
    d_model = model_config(TrainConfig.from_dict(base), vocab).d_model

    def evaluate(X: np.ndarray) -> list[Evaluation]:
        out = []
        for x in X:
            g = SPACE.decode(x)
            tag = hashlib.sha1(json.dumps({k: round(v, 6) for k, v in g.items()}, sort_keys=True).encode()).hexdigest()[:10]
            rc = recipe(g, base, d_model, steps)
            rc["out"] = str(out_root / "runs" / tag)
            res = train(TrainConfig.from_dict(rc), log=lambda *a: None)
            alpha = 0.0
            mpath = Path(rc["out"]) / "metrics.jsonl"
            if mpath.exists():
                ev = [json.loads(line) for line in mpath.read_text().splitlines()]
                tr = [e for e in ev if e.get("event") == "train" and "tauls" in e]
                if tr:
                    alpha = float(tr[-1]["tauls"].get("alpha", 0.0))
            val = float(res["best_val"])
            info = {"run": tag, "val_loss": val, "params_M": res["params"] / 1e6, "alpha": alpha, **g,
                    "recipe": rc}
            log(f"  {tag}: val {val:.4f}  params {info['params_M']:.2f}M  α {alpha:.3f}  lr {g['lr']:.2e}")
            out.append(Evaluation(-val, (info["params_M"], alpha), info))
        return out

    return evaluate, vocab


def search(base: dict, out: str | Path, steps: int = 400, generations: int = 6, batch: int = 4, init: int = 12,
           bins: int = 6, seed: float = 0.4123, log: Callable = print) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    evaluate, vocab = make_evaluator(base, out, steps, log)
    lo, hi = _param_range(base, vocab)
    me = MapElites(SPACE, evaluate, [Descriptor("params_M", lo, hi, bins), Descriptor("alpha", 0.0, 1.0, bins)],
                   seed=seed, batch=batch, init=init)
    me.run(generations)
    elites = me.elites()
    champ = me.champion()
    record = {"steps_per_run": steps, "runs": init + generations * batch, "coverage": me.history[-1].coverage,
              "champion": {k: v for k, v in champ.info.items() if k != "recipe"},
              "elites": [{k: v for k, v in e.info.items() if k != "recipe"} for e in elites]}
    (out / "archive.json").write_text(json.dumps(record, indent=2, default=float))
    best = dict(champ.info["recipe"])
    best.pop("out", None)
    (out / "champion.json").write_text(json.dumps(best, indent=2, default=float))
    log(f"{record['runs']} runs × {steps} steps: coverage {record['coverage']:.0%}; champion val "
        f"{champ.info['val_loss']:.4f} (lr {champ.info['lr']:.2e}, control_mult {champ.info['control_mult']:.2f}); "
        f"wrote {out / 'archive.json'} and {out / 'champion.json'}")
    return record

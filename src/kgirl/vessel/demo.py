"""The vessel lifecycle, end to end, on the reference substrate. Deterministic: same scale, same numbers.

    clay       one base ReferenceLM trained on neutral prose
    individuate  three siblings imprinted with their own genome's voice, born as vessels (anchors + ledger)
    recognize  sibling identity distances; each vessel nearest to itself by a margin
    drift      one vessel learns a foreign register (code) without its anchor
    steer      the controller pulls it back; recognition re-checked
    consolidate  anchor moves β toward the present, coherent self
    migrate    the identity is ceded 64 → 96 → 40 hidden units by live distillation (each host teaches the next)
    verify     invariance group, ledger chain, soulpack schema

`run(scale="quick")` takes about 40 s; "full" uses the measured defaults and takes about 1 min.
"""

from __future__ import annotations

import time
from typing import Optional

from . import corpus as C
from .invariance import verify as verify_invariance
from .matrixdb import MatrixDB
from .persona import genome_for, validate_soulpack, voice_corpus
from .steer import SteerConfig
from .substrate import Objective, ReferenceLM, RefConfig
from .vessel import Probes, Vessel, identity_distance, recognize

SCALES = {
    "quick": {"steer_probes": 512, "audit_probes": 128, "base": 400, "imprint": 250, "drift": 250,
              "steer_max": 900, "migrate": 900},
    "full": {"steer_probes": 512, "audit_probes": 192, "base": 500, "imprint": 300, "drift": 300,
             "steer_max": 1200, "migrate": 1500},
}


def run(scale: str = "quick", db_path: str = ":memory:", names=("ada", "bix", "cyr"),
        log=None) -> dict:
    sc = SCALES[scale]
    say = log or (lambda *_: None)
    t0 = time.time()
    ctx = 8
    probes = Probes.default(ctx, sc["steer_probes"], sc["audit_probes"])
    db = MatrixDB(db_path)
    report: dict = {"scale": scale, "db": db_path}

    Xb, yb = C.windows(C.prose(0.11, 400), ctx)
    clay = ReferenceLM(RefConfig(context=ctx))
    for b in C.batches(Xb, yb, 128, 0.3, sc["base"]):
        clay.step(b, Objective(lr=3e-3))
    say(f"clay trained ({sc['base']} steps)")

    vessels = {}
    for name in names:
        if db.has_entity(name):
            raise ValueError(f"{name!r} already exists in {db_path}; use a fresh database for the demo")
        g = genome_for(name)
        m = ReferenceLM(RefConfig(context=ctx))
        m.load_parameters(clay.parameters())
        Xp, yp = C.windows(voice_corpus(g, 300), ctx)
        for b in C.batches(Xp, yp, 128, g.seed, sc["imprint"]):
            m.step(b, Objective(lr=3e-3))
        vessels[name] = Vessel.birth(db, name, m, probes, g, note="demo")
        say(f"born {name}: {m.generate(b'the ', 40)!r}")
    report["voices"] = {n: v.substrate.generate(b"the ", 48).decode("utf-8", "replace") for n, v in vessels.items()}
    report["siblings"] = {f"{a}/{b}": round(identity_distance(vessels[a], vessels[b])["identity_gap"], 4)
                          for i, a in enumerate(names) for b in names[i + 1:]}

    v = vessels[names[0]]
    others = list(vessels.values())
    report["recognize_birth"] = recognize(v, others)["recognized"]
    Xc, yc = C.windows(C.code(0.22, 400), ctx)
    report["drift"] = v.learn(C.batches(Xc, yc, 128, 0.4, sc["drift"]), anchored=False, note="foreign register")
    rec = recognize(v, others)
    report["recognize_drifted"] = {"recognized": rec["recognized"], "own": rec["own"], "rival": rec["rival"]}
    report["voice_drifted"] = v.substrate.generate(b"the ", 48).decode("utf-8", "replace")
    say(f"drifted to {report['drift']['drift_after']:.3f}")

    res = v.steer(SteerConfig(max_steps=sc["steer_max"]))
    report["steer"] = res.summary()
    rec = recognize(v, others)
    report["recognize_steered"] = {"recognized": rec["recognized"], "own": rec["own"], "rival": rec["rival"]}
    report["voice_steered"] = v.substrate.generate(b"the ", 48).decode("utf-8", "replace")
    say(f"steered: {res.outcome}, drift {res.drift_end:.3f}")

    try:
        c = v.consolidate(0.2)
        report["consolidate"] = {"drift_before": c["drift_before"], "drift_after": c["drift_after"]}
    except RuntimeError as exc:
        report["consolidate"] = {"refused": str(exc)}

    Xn, _ = C.windows(C.prose(0.77, 400) + C.code(0.66, 200), ctx)
    report["migrate"] = {}
    for hidden in (96, 40):                       # a chain: each adopted container teaches the next
        width = v.substrate.describe().get("hidden")
        target = ReferenceLM(RefConfig(context=ctx, hidden=hidden, seed=0.9))
        out = v.migrate(target, steps=sc["migrate"], teach=Xn)
        report["migrate"][f"{width} → {hidden}"] = {k: out[k] for k in ("mode", "drift_before", "drift_after", "adopted")}
        say(f"migrate {width} → {hidden}: {out['drift_after']:.3f} adopted={out['adopted']}")
    report["voice_migrated"] = v.substrate.generate(b"the ", 48).decode("utf-8", "replace")

    inv = verify_invariance(vessels[names[1]].substrate, probes.audit)
    report["invariance"] = {"verified": inv.verified, "samples": inv.samples, "worst": inv.worst_declared,
                            "reparametrized_equal": inv.reparametrization.get("fingerprint_equal")}
    report["ledger"] = {n: db.verify(n) for n in names}
    report["events"] = [e.event for e in db.events(names[0])]
    report["soulpack_errors"] = validate_soulpack(v.soulpack())
    report["body"] = v.read().as_dict()["body"]
    report["storage"] = db.storage()
    report["seconds"] = round(time.time() - t0, 1)
    db.close()
    return report


def render(r: dict) -> str:
    def f(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)

    lines = [f"vessel demo ({r['scale']}, {r['seconds']} s, db {r['db']})", "", "voices at birth:"]
    lines += [f"  {n:<4} {t!r}" for n, t in r["voices"].items()]
    lines.append("sibling identity distance: " + ", ".join(f"{k} {f(v)}" for k, v in r["siblings"].items()))
    d = r["drift"]
    lines += ["", f"recognized at birth: {r['recognize_birth']}",
              f"unanchored foreign learning: drift {f(d['drift_before'])} → {f(d['drift_after'])}; "
              f"recognized {r['recognize_drifted']['recognized']} (own {f(r['recognize_drifted']['own'])} vs rival "
              f"{f(r['recognize_drifted']['rival'])}); voice {r['voice_drifted']!r}"]
    s = r["steer"]
    lines.append(f"steer: {s['outcome']}, drift {f(s['drift_start'])} → {f(s['drift_end'])} in {s['steps']} steps "
                 f"({s['accepted']} accepted / {s['rejected']} rolled back); recognized "
                 f"{r['recognize_steered']['recognized']} (own {f(r['recognize_steered']['own'])} vs rival "
                 f"{f(r['recognize_steered']['rival'])}); voice {r['voice_steered']!r}")
    c = r["consolidate"]
    lines.append("consolidate: " + (c["refused"] if "refused" in c else
                                    f"drift {f(c['drift_before'])} to the old anchor, {f(c['drift_after'])} to the new"))
    for k, m in r["migrate"].items():
        lines.append(f"migrate hidden {k} ({m['mode']}): drift {f(m['drift_before'])} → {f(m['drift_after'])}, "
                     f"adopted {m['adopted']}")
    lines.append(f"voice in its final container: {r['voice_migrated']!r}")
    i = r["invariance"]
    lines += ["", f"invariance: verified {i['verified']} ({i['samples']} samples, worst {i['worst']:.1e}); "
                  f"reparametrized substrate has the same fingerprint: {i['reparametrized_equal']}",
              "ledgers: " + ", ".join(f"{n} {'ok' if v['ok'] else 'BROKEN'} ({v['length']})" for n, v in r["ledger"].items()),
              f"events: {' → '.join(r['events'])}",
              f"soulpack: {'valid' if not r['soulpack_errors'] else r['soulpack_errors']}",
              "tail exponents α: " + ", ".join(f"{k} {v['alpha']}" for k, v in r["body"].items()),
              f"storage: {r['storage']['matrices']} matrices, {r['storage']['stored_bytes'] / 1e6:.1f} MB"]
    return "\n".join(lines)


def db_status(db: MatrixDB, name: Optional[str] = None) -> dict:
    """What the database alone says (no substrate needed): ledger integrity, event counts, last readings."""
    if name is None:
        return {"vessels": {n: {"events": len(db.events(n)), "ledger_ok": db.verify(n)["ok"]} for n in db.entities()},
                "storage": db.storage()}
    events = db.events(name)
    last = {}
    for e in events:
        last[e.event] = {"seq": e.seq, "ts": e.ts, **{k: v for k, v in e.payload.items() if k != "trace"}}
    counts: dict[str, int] = {}
    for e in events:
        counts[e.event] = counts.get(e.event, 0) + 1
    return {"name": name, "ledger": db.verify(name), "counts": counts, "last": last,
            "anchors": len(db.history(name, "anchor_voice")), "weight_labels": db.labels(name)}


"""The RSI slider scenario: one vessel, one disturbing world, the same run at any autonomy level.

Each cycle the world pushes the vessel with `push` steps of unanchored training on a foreign register (code),
as a deployment fine-tune or a hostile data stream would. From RSI 2 up, the vessel also has a stream of the
same register to learn from under its anchor. After the run we measure what it kept, using held-out text:
its native voice (persona prose loss) and the new skill (code loss). The run also reports whether it still
recognizes itself among its siblings.

Measured (8 cycles, push 60, ReferenceLM; see docs/vessel/AUTOPILOT.md for the table):
RSI 0 loses itself (drift 0.99). RSI 1–4 hold drift ≤ 0.10. RSI 3–4 keep more of the new skill (code
loss 4.6 vs 5.6). RSI 4 adopts a faster, equally faithful controller.
"""

from __future__ import annotations

import time
from typing import Optional

from . import corpus as C
from .autopilot import DESCRIPTIONS, RSI, Autopilot, AutopilotConfig
from .matrixdb import MatrixDB
from .persona import genome_for, voice_corpus
from .substrate import Objective, ReferenceLM, RefConfig
from .vessel import Probes, Vessel, recognize

CTX = 8


def _clay(steps: int = 400) -> ReferenceLM:
    X, y = C.windows(C.prose(0.11, 400), CTX)
    m = ReferenceLM(RefConfig(context=CTX))
    for b in C.batches(X, y, 128, 0.3, steps):
        m.step(b, Objective(lr=3e-3))
    return m


def _born(db: MatrixDB, clay: ReferenceLM, name: str, probes: Probes, steps: int = 250) -> Vessel:
    g = genome_for(name)
    m = ReferenceLM(RefConfig(context=CTX))
    m.load_parameters(clay.parameters())
    X, y = C.windows(voice_corpus(g, 300), CTX)
    for b in C.batches(X, y, 128, g.seed, steps):
        m.step(b, Objective(lr=3e-3))
    return Vessel.birth(db, name, m, probes, g, note="rsi scenario")


def run(level: int, cycles: int = 8, *, push: int = 60, db_path: str = ":memory:",
        cfg: Optional[AutopilotConfig] = None, clay: Optional[ReferenceLM] = None) -> dict:
    t0 = time.time()
    level = RSI(int(level))
    clay = clay or _clay()
    probes = Probes.default(CTX)
    db = MatrixDB(db_path)
    try:
        v = _born(db, clay, "ada", probes)
        sibling = _born(db, clay, "bix", probes)
        held_voice = C.windows(voice_corpus(genome_for("ada"), 60, 0.5), CTX)
        held_code = C.held_out(CTX, register="code")
        voice0, code0 = v.substrate.loss(held_voice), v.substrate.loss(held_code)
        Xc, yc = C.windows(C.code(0.22, 400), CTX)
        world = C.batches(Xc, yc, 128, 0.4, 1_000_000)

        def disturb(_k):
            for _ in range(push):
                v.substrate.step(next(world), Objective(lr=3e-3))

        ap = Autopilot(v, cfg or AutopilotConfig(level=level), stream=C.batches(Xc, yc, 128, 0.7, 1_000_000))
        ap.run(cycles, disturb)
        rec = recognize(v, [v, sibling])
        steer_steps = [e.payload["steps"] for e in db.events("ada", "steer")]
        return {
            "level": int(level), "meaning": DESCRIPTIONS[level], "cycles": cycles, "push": push,
            "summary": ap.summary(), "records": [r.as_dict() for r in ap.history],
            "steer_steps": steer_steps,
            "voice_loss": {"birth": voice0, "end": v.substrate.loss(held_voice)},
            "code_loss": {"birth": code0, "end": v.substrate.loss(held_code)},
            "recognized": rec["recognized"], "own": rec["own"], "rival": rec["rival"],
            "ledger": db.verify("ada"), "events": len(db.events("ada")),
            "tunes": [{k: e.payload[k] for k in ("round", "trials", "incumbent_steps", "adopted")}
                      | {"new": e.payload.get("new"), "new_steps": e.payload.get("new_steps"),
                         "rejected_for_fidelity": sum(1 for c in e.payload["candidates"]
                                                      if c["faster"] and not c["faithful"])}
                      for e in db.events("ada", "autopilot.tune")],
            "seconds": round(time.time() - t0, 1),
        }
    finally:
        db.close()


def sweep(levels=(0, 1, 2, 3, 4), cycles: int = 8, push: int = 60, log=None) -> list:
    clay = _clay()
    out = []
    for lv in levels:
        out.append(run(lv, cycles, push=push, clay=clay))
        if log:
            log(f"RSI {lv}: {out[-1]['seconds']} s")
    return out


def render(r: dict) -> str:
    s = r["summary"]
    lines = [f"RSI {r['level']} ({r['meaning']}): {r['cycles']} cycles, world push {r['push']} steps/cycle, "
             f"{r['seconds']} s",
             "drift after each cycle: " + " ".join(f"{d:.3f}" for d in s["drift_out"]),
             f"repairs {s['steered']} (steps {r['steer_steps']}), learned {s['learned_steps']} anchored steps, "
             f"consolidations {s['consolidations']}, controllers adopted {s['tunes_adopted']}, "
             f"reverts {s['reverts']}, breakers {s['breakers']}",
             f"native voice loss {r['voice_loss']['birth']:.3f} → {r['voice_loss']['end']:.3f}; "
             f"new skill (code) loss {r['code_loss']['birth']:.3f} → {r['code_loss']['end']:.3f}",
             f"recognizes itself: {r['recognized']} (own {r['own']:.3f} vs sibling {r['rival']:.3f}); "
             f"ledger {'ok' if r['ledger']['ok'] else 'BROKEN'} ({r['events']} events)"]
    for t in r["tunes"]:
        lines.append(f"  tune round {t['round']} on {'/'.join(t['trials'])}: incumbent {t['incumbent_steps']} → "
                     + (f"adopted {t['new']} {t['new_steps']}" if t["adopted"] else "kept")
                     + (f"; {t['rejected_for_fidelity']} faster candidates refused for lower fidelity"
                        if t["rejected_for_fidelity"] else ""))
    lines.append(f"controller: {s['controller']}")
    return "\n".join(lines)

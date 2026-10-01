"""`python -m kgirl.nihiline <command>` — the c-BPE-ECL lab from the command line."""

from __future__ import annotations

import argparse
import json
import sys

from . import calibration as cal
from .protocol import NIHILINE, run_protocol


def _json(obj) -> str:
    def default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)
    return json.dumps(obj, indent=2, default=default, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kgirl.nihiline", description="CN121933729A c-BPE-ECL dual-marker lab")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("protocol", help="run the nine-level assay end to end")
    p.add_argument("--cea", type=float, default=1e4, help="CEA-lane MCF-7 cells/mL")
    p.add_argument("--afp", type=float, default=1e4, help="AFP-lane MCF-7 cells/mL")
    p.add_argument("--tpa", type=float, default=50.0, help="TPA mM (claim range 20–70)")
    p.add_argument("--nihil", type=float, help="fix the nihil fraction instead of optimizing it")
    p = sub.add_parser("quantify", help="ECL signal(s) → concentration or cells/mL")
    p.add_argument("--cea", type=float, help="CEA-lane ECL intensity (a.u.)")
    p.add_argument("--afp", type=float, help="AFP-lane ECL intensity (a.u.)")
    p.add_argument("--mode", choices=("conc", "cells"), default="conc")
    sub.add_parser("scar", help="LOD consistency across the patent's four curves")
    p = sub.add_parser("gap", help="amplification chain per cell vs reported LOD")
    p.add_argument("--marker", choices=("CEA", "AFP"))
    sub.add_parser("pulse", help="nihil leverage of the drive (numpy)")
    sub.add_parser("bloom", help="Bloom viability sentinel (numpy)")
    sub.add_parser("array", help="BPE array lattice characterization (numpy)")
    p = sub.add_parser("plot", help="write figures (matplotlib)")
    p.add_argument("which", nargs="*", default=["calibration", "gap", "pulse", "bloom", "array"])
    p.add_argument("-o", "--outdir", default="outputs/nihiline")
    sub.add_parser("haiku")
    a = ap.parse_args(argv)

    if a.cmd == "protocol":
        run = run_protocol({"CEA": a.cea, "AFP": a.afp}, tpa_mM=a.tpa, f_nihil=a.nihil)
        print(_json(run.to_dict()) if a.json else run.render())
    elif a.cmd == "quantify":
        out = {}
        if a.cea is not None:
            out["CEA"] = cal.quantify("CEA", a.cea, a.mode).to_dict()
        if a.afp is not None:
            out["AFP"] = cal.quantify("AFP", a.afp, a.mode).to_dict()
        if not out:
            ap.error("give --cea and/or --afp")
        if a.json:
            print(_json(out))
        else:
            for m, q in out.items():
                print(f"{m}: {q['signal']:.1f} a.u. → {q['value']:.4g} {q['unit']} [{q['flag']}] (scar {q['scar']} a.u.)")
    elif a.cmd == "scar":
        rows = cal.scar_report()
        if a.json:
            print(_json(rows))
        else:
            for r in rows:
                print(f"{r['marker']} {r['mode']:<5} LOD {r['lod']:<8g} {r['unit']:<8} → scar {r['scar_signal']:>7.1f} a.u."
                      + ("   ← outlier" if r["outlier"] else "")
                      + ("   (LOD below linear range)" if r["lod_below_linear_range"] else ""))
    elif a.cmd == "gap":
        from .amplification import chain, required, sensitivity
        for m in ([a.marker] if a.marker else ["CEA", "AFP"]):
            ch = chain(m)
            if a.json:
                print(_json({**ch.to_dict(), "sensitivity": sensitivity(m)}))
                continue
            print(ch.render())
            top = ", ".join(f"{k} {v:+.2f}" for k, v in sensitivity(m)[:4])
            print(f"  elasticities d log LOD / d log knob: {top}")
            print(f"  electroactive MB fraction that exactly meets the reported LOD: {required(m):.2e}\n")
    elif a.cmd == "pulse":
        from .pulse import control_nihil_sweep, leverage_sweep, ratio_sweep
        lev, joint, rat = leverage_sweep(), control_nihil_sweep(), ratio_sweep()
        res = {"f_opt": lev["f_opt"], "product_opt": lev["product_opt"],
               "f_opt_by_control": dict(zip(map(float, joint["C"]), map(float, joint["f_opt_by_C"]))),
               "control_slope": joint["slope"], "verdict": joint["verdict"],
               "best_pattern_ratio": float(rat["ratio"][int(rat["product"].argmax())])}
        print(_json(res) if a.json else "\n".join(f"{k}: {v}" for k, v in res.items()))
    elif a.cmd == "bloom":
        from .bloom import run
        s = run().summary()
        print(_json(s) if a.json else "\n".join(f"{k}: {v}" for k, v in s.items()))
    elif a.cmd == "array":
        from .array import characterize, lyapunov
        r = characterize()
        r.pop("final")
        r["lyapunov"] = {k: v for k, v in lyapunov().items() if k != "divergence"}
        print(_json(r) if a.json else "\n".join(f"{k}: {v}" for k, v in r.items()))
    elif a.cmd == "plot":
        from pathlib import Path

        from .plots import ALL
        for w in a.which:
            print(ALL[w](Path(a.outdir) / f"{w}.png"))
    elif a.cmd == "haiku":
        print(NIHILINE)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

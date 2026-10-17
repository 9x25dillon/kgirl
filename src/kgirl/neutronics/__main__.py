import argparse
import json
import sys
from pathlib import Path

from . import dataset
from .surrogate import CV_SEEDS, FEATURE_SETS, cross_validate, run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kgirl.neutronics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="locate, load and physics-check the dataset")
    p.add_argument("--data", help="dataset directory (default: NuRea_sim via kgirl.paths)")
    p = sub.add_parser("fit", help="fit the surrogates, score them on the held-out split")
    p.add_argument("--data")
    p.add_argument("--out", default="runs/neutronics")
    p.add_argument("--degrees", type=int, nargs="+", default=[1, 2])
    p.add_argument("--features", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS),
                   help="feature sets fitted at every degree (maxtype adds five peak-sensitive summaries)")
    p = sub.add_parser("cv", help="poly vs maxtype: k-fold CV over chaos seeds plus a paired bootstrap")
    p.add_argument("--data")
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=float, nargs="+", default=list(CV_SEEDS))
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--out", help="write the full record (including per-fold rows) to this JSON file")
    args = ap.parse_args(argv)
    train = dataset.load("train", args.data)
    if args.cmd == "cv":
        dataset.validate(train)
        rec = cross_validate(train, args.degree, args.seeds, args.folds, bootstrap=args.bootstrap)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(rec, indent=1))
        d, b = rec["delta_pppf_r2"], rec["bootstrap_delta_pppf_r2"]
        print(f"{args.folds}-fold CV x {len(args.seeds)} chaos seeds on {rec['rows']} rows, degree {args.degree}")
        for name in ("poly", "maxtype"):
            s = rec[name]
            print(f"  {name:<8} PPPF R² {s['pppf_r2']['mean']:.4f} ± {s['pppf_r2']['sd']:.4f}  "
                  f"RMSE {s['pppf_rmse']['mean']:.4f}   k_inf {s['k_inf_rmse_pcm']['mean']:.1f} pcm   λ {s['lambdas']}")
        print(f"  ΔR²(PPPF) per fold: mean {d['mean']:+.4f} (min {d['min']:+.4f}, max {d['max']:+.4f}), "
              f"positive in {d['positive']}/{d['of']}")
        print(f"  paired bootstrap ({b['resamples']}): ΔR² {b['point']:+.4f}, 95% CI "
              f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]")
        return 0
    test = dataset.load("test", args.data)
    if args.cmd == "info":
        print(json.dumps({"dir": str(dataset.data_dir(args.data)), "train_rows": train.n, "test_rows": test.n,
                          "train_checks": dataset.validate(train), "test_checks": dataset.validate(test)}, indent=2))
        return 0
    dataset.validate(train)
    rec = run(train, test, args.degrees, args.out, args.features)
    base = rec["baseline_mean_predictor"]
    print(f"train {train.n} / held-out test {test.n} simulations; baseline = predict the train mean")
    print(f"{'model':<16} {'terms':>6} {'lambda':>8} {'k_inf RMSE':>11} {'pcm':>8} {'R2':>7} {'PPPF RMSE':>10} {'R2':>7}")
    print(f"{'baseline':<16} {'1':>6} {'-':>8} {base['k_inf']['rmse']:11.6f} {base['k_inf']['rmse_pcm']:8.1f} "
          f"{base['k_inf']['r2']:7.4f} {base['pppf']['rmse']:10.5f} {base['pppf']['r2']:7.4f}")
    for name, m in rec["models"].items():
        t = m["test"]
        print(f"{name:<16} {m['terms']:>6} {m['lambda']:>8g} {t['k_inf']['rmse']:11.6f} {t['k_inf']['rmse_pcm']:8.1f} "
              f"{t['k_inf']['r2']:7.4f} {t['pppf']['rmse']:10.5f} {t['pppf']['r2']:7.4f}")
    print(f"best: {rec['best']}; wrote {args.out}/telemetry.json, surrogate.npz, limps_payload.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

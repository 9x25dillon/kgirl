import argparse
import json
import sys

from . import dataset
from .surrogate import run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kgirl.neutronics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="locate, load and physics-check the dataset")
    p.add_argument("--data", help="dataset directory (default: NuRea_sim via kgirl.paths)")
    p = sub.add_parser("fit", help="fit degree-1 and degree-2 surrogates, score on the held-out split")
    p.add_argument("--data")
    p.add_argument("--out", default="runs/neutronics")
    p.add_argument("--degrees", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args(argv)
    train, test = dataset.load("train", args.data), dataset.load("test", args.data)
    if args.cmd == "info":
        print(json.dumps({"dir": str(dataset.data_dir(args.data)), "train_rows": train.n, "test_rows": test.n,
                          "train_checks": dataset.validate(train), "test_checks": dataset.validate(test)}, indent=2))
        return 0
    dataset.validate(train)
    rec = run(train, test, args.degrees, args.out)
    base = rec["baseline_mean_predictor"]
    print(f"train {train.n} / held-out test {test.n} simulations; baseline = predict the train mean")
    print(f"{'model':<10} {'terms':>6} {'lambda':>8} {'k_inf RMSE':>11} {'pcm':>8} {'R2':>7} {'PPPF RMSE':>10} {'R2':>7}")
    print(f"{'baseline':<10} {'1':>6} {'-':>8} {base['k_inf']['rmse']:11.6f} {base['k_inf']['rmse_pcm']:8.1f} "
          f"{base['k_inf']['r2']:7.4f} {base['pppf']['rmse']:10.5f} {base['pppf']['r2']:7.4f}")
    for name, m in rec["models"].items():
        t = m["test"]
        print(f"{name:<10} {m['terms']:>6} {m['lambda']:>8g} {t['k_inf']['rmse']:11.6f} {t['k_inf']['rmse_pcm']:8.1f} "
              f"{t['k_inf']['r2']:7.4f} {t['pppf']['rmse']:10.5f} {t['pppf']['r2']:7.4f}")
    print(f"best: {rec['best']}; wrote {args.out}/telemetry.json, surrogate.npz, limps_payload.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

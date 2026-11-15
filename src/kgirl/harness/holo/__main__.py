"""python -m kgirl.harness.holo build | status | query — the holo layer from a shell, as JSON."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .build import build, status
from .infer import query
from .store import PER_CELL, SEED, HoloError, HoloStore


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kgirl.harness.holo", description=__doc__)
    ap.add_argument("--home", help="harness home (default $KGIRL_HOME, else ~/.kgirl)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="derive the store from the chunk index (offline; no model call)")
    p.add_argument("--chunks-db", help="chunk index (default $KGIRL_CHUNKS_DB, else HOME/chunks/all/all-repos.db)")
    p.add_argument("--matrix", help="KGVM matrix (default $KGIRL_MATRIX, else HOME/chunks/all/all-repos.kgvm)")
    p.add_argument("--per-cell", type=int, default=PER_CELL,
                   help=f"chunk vectors kept per (repository, kind) cell (default {PER_CELL}; 0 keeps all)")
    p.add_argument("--seed", type=float, default=SEED, help=f"chaos seed of the sample (default {SEED})")
    p.add_argument("--no-soup", action="store_true", help="leave the Soup evidence tables out")
    sub.add_parser("status", help="entities, storage, chain verification, the last build")
    p = sub.add_parser("query", help="search the space; hits carry owning entities and digests")
    p.add_argument("--text", help="a question (embedded with the matrix's model: one model call)")
    p.add_argument("--vector", help="a .npy file holding the query vector (no model call)")
    p.add_argument("-k", type=int, default=8)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "build":
            out = build(args.home, chunks_db=args.chunks_db, matrix=args.matrix, per_cell=args.per_cell or None,
                        seed=args.seed, use_soup=not args.no_soup)
        elif args.cmd == "status":
            out = status(args.home)
        else:
            if bool(args.text) == bool(args.vector):
                ap.error("query needs exactly one of --text and --vector")
            if status(args.home)["built"] is None:           # do not create an empty store just to say so
                raise HoloError("the holo store has not been built yet: run `build` first")
            q = args.text if args.text else np.load(args.vector, allow_pickle=False)
            with HoloStore(args.home) as store:
                out = query(store, q, args.k)
    except (HoloError, FileNotFoundError, ValueError) as exc:
        print(f"holo: {exc}", file=sys.stderr)
        return 1
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

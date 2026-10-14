"""python -m kgirl.tauls — tokenizer, data, plan, train, eval, generate, memory, improve, stability, info.

Typical desktop run (see docs/tauls/README.md):

    python -m kgirl.tauls tokenizer --roots . ../numbskull --vocab 8192 --out runs/tauls/tokenizer.json
    python -m kgirl.tauls data      --roots . ../numbskull --tokenizer runs/tauls/tokenizer.json --out runs/tauls/data
    python -m kgirl.tauls plan      --preset small --batch 16 --vram 8
    python -m kgirl.tauls train     --data runs/tauls/data --out runs/tauls/small --preset small --auto-batch
    python -m kgirl.tauls generate  --ckpt runs/tauls/small/best.pt --tokenizer runs/tauls/tokenizer.json --prompt "def "
    python -m kgirl.tauls memory    --ckpt runs/tauls/small/best.pt --data runs/tauls/data --out runs/tauls/memory
    python -m kgirl.tauls improve   --ckpt runs/tauls/small/best.pt --data runs/tauls/data --out runs/tauls/kgirl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _set_pairs(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m kgirl.tauls", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tokenizer", help="train a byte-level BPE tokenizer on the corpus")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--out", default="runs/tauls/tokenizer.json")
    p.add_argument("--sft", help="soup export-sft JSONL to include")

    p = sub.add_parser("data", help="tokenize the corpus into train/val shards")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--tokenizer", default="runs/tauls/tokenizer.json")
    p.add_argument("--out", default="runs/tauls/data")
    p.add_argument("--val-permille", type=int, default=50)
    p.add_argument("--sft")

    p = sub.add_parser("plan", help="parameter count and a measured VRAM estimate for a preset")
    p.add_argument("--preset", default="small")
    p.add_argument("--set", nargs="*", default=[], help="model overrides, e.g. ctx=1024 n_layers=8")
    p.add_argument("--vocab", type=int, default=0, help="vocabulary size (default: the preset's)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--vram", type=float, default=8.0, help="GiB available")
    p.add_argument("--grad-checkpoint", action="store_true")

    p = sub.add_parser("train", help="train (or --resume) a TA-ULS model")
    p.add_argument("--config", help="JSON file with TrainConfig fields")
    p.add_argument("--data")
    p.add_argument("--out")
    p.add_argument("--preset")
    p.add_argument("--set", nargs="*", default=[], help="model overrides, e.g. dropout=0.1 kfp=false")
    p.add_argument("--train-set", nargs="*", default=[], help="TrainConfig overrides, e.g. lr=3e-4 steps=50000")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--auto-batch", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true")

    p = sub.add_parser("eval", help="validation loss, perplexity and bits/byte of a checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--tokens", type=int, default=1 << 20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")

    p = sub.add_parser("generate", help="sample text from a checkpoint (chaos-driven sampling)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=float, default=0.4123)
    p.add_argument("--device", default="auto")
    p.add_argument("--memory", help="KGIRL memory directory (memory.npz + memory.json from `memory --out`)")

    p = sub.add_parser("memory", help="KGIRL: build a holographic fractal memory from a checkpoint and score it")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tokens", type=int, default=262144, help="train tokens collected into the memory")
    p.add_argument("--eval-tokens", type=int, default=65536)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--set", nargs="*", default=[], help="MemoryConfig overrides, e.g. max_depth=3 branching=8")
    p.add_argument("--out", help="write memory.npz and memory.json here")
    p.add_argument("--device", default="auto")

    p = sub.add_parser("improve", help="KGIRL: recursive self-improvement, memory distilled into the model")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--set", nargs="*", default=[], help="ImproveConfig overrides, e.g. rounds=4 distil_steps=500")
    p.add_argument("--memory-set", nargs="*", default=[], help="MemoryConfig overrides")
    p.add_argument("--device", default="auto")

    p = sub.add_parser("stability", help="spectral / KFP / entropy report of a checkpoint's weight matrices")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--julia", default=None, help="Julia bridge URL (default $TAULS_JULIA_URL or :9000; numpy if down)")
    p.add_argument("--local", action="store_true", help="numpy only, do not contact Julia")
    p.add_argument("--top", type=int, default=12, help="show the N least stable matrices")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("info", help="config, size and training state of a checkpoint")
    p.add_argument("--ckpt", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "tokenizer":
        from .data import iter_documents, iter_sft
        from .tokenizer import train
        docs = list(iter_documents(args.roots)) + (list(iter_sft(args.sft)) if args.sft else [])
        print(f"training on {len(docs)} documents ({sum(len(d.text) for d in docs):,} chars)")
        tok = train((d.framed() for d in docs), args.vocab,
                    progress=lambda n, t: print(f"  merges {n}/{t}", file=sys.stderr))
        tok.save(args.out)
        print(f"wrote {args.out}: vocab {tok.vocab_size}, sha {tok.sha}")
    elif args.cmd == "data":
        from .data import build
        from .tokenizer import Tokenizer
        build(args.roots, Tokenizer.load(args.tokenizer), args.out, args.val_permille, args.sft)
    elif args.cmd == "plan":
        from .plan import plan
        print(plan(args.preset, _set_pairs(args.set), args.vocab, args.batch, args.vram, args.grad_checkpoint))
    elif args.cmd == "train":
        from .train import TrainConfig, train
        d = json.loads(Path(args.config).read_text()) if args.config else {}
        for k in ("data", "out", "preset"):
            if getattr(args, k):
                d[k] = getattr(args, k)
        d.setdefault("model", {}).update(_set_pairs(args.set))
        d.update(_set_pairs(args.train_set))
        for flag in ("auto_batch", "compile", "grad_checkpoint"):
            if getattr(args, flag):
                d[flag] = True
        res = train(TrainConfig.from_dict(d), resume=args.resume)
        print(json.dumps(res, indent=2))
    elif args.cmd == "eval":
        from .data import TokenData
        from .train import evaluate, load_model, resolve_device, resolve_precision
        dev = resolve_device(args.device)
        model, ck = load_model(args.ckpt, dev)
        data = TokenData(args.data)
        if ck["tokenizer_sha"] != data.meta["tokenizer_sha"]:
            raise SystemExit("checkpoint and data use different tokenizers")
        print(json.dumps(evaluate(model, data, args.split, args.batch, args.tokens, dev,
                                  resolve_precision("auto", dev)), indent=2))
    elif args.cmd == "generate":
        from . import chaos
        from .tokenizer import Tokenizer
        from .train import load_model, resolve_device
        tok = Tokenizer.load(args.tokenizer)
        model, ck = load_model(args.ckpt, resolve_device(args.device))
        if ck["tokenizer_sha"] != tok.sha:
            raise SystemExit("checkpoint and tokenizer differ")
        ids = tok.encode(args.prompt) or [tok.eot]
        sampler = chaos.Sampler(args.seed, chaos.stream_id("generate"))
        if args.memory:
            from .holo import HoloMemory
            from .improve import generate_with_memory
            mem = HoloMemory.load(Path(args.memory) / "memory.npz")
            g = json.loads((Path(args.memory) / "memory.json").read_text())["score"]["gate"]
            out = generate_with_memory(model, mem, g, ids, args.max_new, args.temperature, args.top_k,
                                       args.top_p, {tok.eot}, sampler)
        else:
            out = model.generate(ids, args.max_new, args.temperature, args.top_k, args.top_p, {tok.eot}, sampler)
        print(args.prompt + tok.decode(t for t in out if t != tok.eot))
    elif args.cmd in ("memory", "improve"):
        from .data import TokenData
        from .train import load_model, resolve_device
        dev = resolve_device(args.device)
        model, ck = load_model(args.ckpt, dev)
        data = TokenData(args.data)
        if ck.get("tokenizer_sha") != data.meta["tokenizer_sha"]:
            raise SystemExit("checkpoint and data use different tokenizers")
        if args.cmd == "memory":
            from .holo import HoloMemory, MemoryConfig
            from .improve import collect, score, token_stats
            mem = HoloMemory(model.cfg.d_model, model.cfg.vocab_size, MemoryConfig(**_set_pairs(args.set)))
            z, y = collect(model, mem, data, "train", args.tokens, args.batch, dev)
            mem.build(z, y)
            rep = {"memory": mem.summary(),
                   "score": score(token_stats(model, mem, data, "val", args.eval_tokens, args.batch, dev))}
            if args.out:
                Path(args.out).mkdir(parents=True, exist_ok=True)
                mem.save(Path(args.out) / "memory.npz")
                (Path(args.out) / "memory.json").write_text(json.dumps(rep, indent=2, default=float))
            print(json.dumps(rep, indent=2, default=float))
        else:
            from .improve import ImproveConfig, improve
            ic = ImproveConfig(**_set_pairs(args.set))
            ic.memory = {**ic.memory, **_set_pairs(args.memory_set)}
            rec = improve(model, data, ic, args.out, dev, ckpt=ck)
            print(json.dumps({k: rec[k] for k in ("baseline_val_B_nll", "final_val_B_nll", "total_gain_nats")}, indent=2))
    elif args.cmd == "stability":
        import torch
        from .julia_bridge import DEFAULT_URL, JuliaBridge, summarize, weight_report
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        bridge = None if args.local else JuliaBridge(args.julia or DEFAULT_URL)
        rows = weight_report(ck["model"], bridge)
        if args.json:
            print(json.dumps({"summary": summarize(rows), "matrices": rows}, indent=2, default=str))
        else:
            print(json.dumps(summarize(rows), indent=2, default=str))
            print(f"{'matrix':<42} {'shape':>12} {'overall':>8} {'class':>9} {'cond':>9} {'srank':>7} {'fluct':>9}")
            for r in sorted(rows, key=lambda r: r["overall_stability"])[:args.top]:
                print(f"{r['name']:<42} {str(tuple(r['shape'])):>12} {r['overall_stability']:8.3f} "
                      f"{r['stability_class']:>9} {r['condition']:9.1f} {r['stable_rank']:7.1f} {r['kfp_fluctuation']:9.2e}")
    elif args.cmd == "info":
        import torch
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        n = sum(t.numel() for t in ck["model"].values())
        print(json.dumps({"step": ck["step"], "best_val": ck.get("best_val"), "params": n,
                          "tokenizer_sha": ck["tokenizer_sha"], "model": ck["model_config"],
                          "train": ck["train_config"]}, indent=2))
    return 0

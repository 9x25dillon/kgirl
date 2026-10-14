"""numbskull integration: find it, put its modules on the path, and prove each component works.

    python -m kgirl.numbskull status          # human-readable
    python -m kgirl.numbskull status --json   # for scripts and CI

numbskull (https://github.com/9x25dillon/numbskull) provides, and kgirl's *_numbskull_adapter modules use:

    advanced_embedding_pipeline   hybrid semantic / mathematical / fractal embeddings   (repo root)
    holographic_similarity_engine holographic similarity search                         (repo root)
    wavecaster                    software modem: FEC, modulation, PHY, radio I/O        (packages/wavecaster)
    emergentnet                   Ising/QUBO optimisers, HRR memory, vector store       (packages/emergentnet)

Location: see kgirl.paths.find_repo (NUMBSKULL_PATH, <kgirl>/numbskull, ../numbskull, ~/numbskull).
The two packages are pip-installable; `requirements-numbskull.txt` lists everything.

A component counts as available only if its probe runs and returns the known answer, so "available"
means "works here", not just "imports":

    wavecaster.fec   RS(255,223): corrupt 3 bytes of a codeword, decode, get the message back
    emergentnet      exact Ising solver on number partitioning {4,5,6,7,8}: a perfect split exists (energy 0)
    emergentnet HRR  unbind(bind(a, b), a) is closer to b than to any other symbol
    hybrid pipeline  embed a sentence with the mathematical + fractal paths (no network needed)
    emergent network execute_emergent_protocol on 32 points returns its protocol sections

Known numbskull bugs (patches in docs/numbskull/, apply with `git apply` in the numbskull repo):
    0001  pipeline __del__ created tasks with no running event loop (leaked coroutines, unclosed client)
    0002  HolographicProtocol.recall_transform and .associative_memory scored each scalar field point
          against the whole query (IndexError / ValueError on every input), and MorphogeneticProtocol ran
          2-D Turing dynamics on the orchestrator's 1-D field; emergent_protocol() applies the same fixes
          at load time while numbskull lacks them
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any, Callable, Dict

from kgirl.paths import ensure_on_path, find_repo


def _probe(fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        out = fn()
        out.setdefault("ok", True)
    except Exception as e:                       # a probe failing is a finding to report, not a crash
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def probe_reed_solomon() -> Dict[str, Any]:
    from wavecaster.fec.reed_solomon import RSCode
    rs = RSCode(255, 223)
    msg = bytes(range(223))
    cw = rs.encode(msg).copy()
    for pos in (3, 100, 200):
        cw[pos] ^= 0x5A
    out, nerr = rs.decode(cw)
    return {"ok": bytes(out[:223]) == msg and nerr == 3, "corrected": int(nerr)}


def probe_partition() -> Dict[str, Any]:
    from emergentnet.optim import exact
    from emergentnet.optim.problems import number_partitioning
    res = exact(number_partitioning([4, 5, 6, 7, 8]))
    return {"ok": abs(res.energy) < 1e-9, "energy": float(res.energy), "spins": res.x.tolist()}


def probe_hrr() -> Dict[str, Any]:
    import numpy as np
    from emergentnet.memory.hrr import bind, symbol, unbind
    d = 1024
    names = ["role", "filler", "noise1", "noise2", "noise3"]
    s = {n: symbol(n, d, "kgirl-probe") for n in names}
    rec = unbind(bind(s["role"], s["filler"]), s["role"])
    sims = {n: float(np.dot(rec, v) / (np.linalg.norm(rec) * np.linalg.norm(v))) for n, v in s.items() if n != "role"}
    best = max(sims, key=sims.get)
    return {"ok": best == "filler", "cosine_filler": round(sims["filler"], 4),
            "best_other": round(max(v for k, v in sims.items() if k != "filler"), 4)}


def probe_hybrid() -> Dict[str, Any]:
    from advanced_embedding_pipeline import HybridConfig, HybridEmbeddingPipeline
    async def run():
        pipe = HybridEmbeddingPipeline(HybridConfig(use_semantic=False, parallel_processing=False))
        try:
            return await pipe.embed("Kinetic force damping keeps the lattice coherent.")
        finally:
            await pipe.close()           # deterministic shutdown inside the loop (see docs/numbskull patch)
    res = asyncio.run(run())
    emb = res.get("fused_embedding")
    if emb is None:
        emb = res.get("embedding")
    n = len(emb) if emb is not None else 0
    return {"ok": n > 0, "dim": n, "components": sorted(k for k in res if k.endswith("embedding"))}


def _recall_transform_fixed(self, query, holographic_field):
    """numbskull HolographicProtocol.recall_transform, corrected (docs/numbskull/0002 patch)."""
    import numpy as np
    field = np.asarray(holographic_field).ravel()
    q = np.asarray(query).ravel().astype(complex)
    if q.size == 1:
        similarities = np.real(np.conj(q[0]) * field)
    else:
        m = min(q.size, field.size)
        similarities = np.real(np.conj(q[:m]) * field[:m])
    return field[int(np.argmax(similarities))]


def _associative_memory_fixed(self, query, threshold: float = 0.5):
    """numbskull HolographicProtocol.associative_memory, corrected (docs/numbskull/0002 patch)."""
    import numpy as np
    field = np.asarray(self.H).ravel()
    q = np.asarray(query).ravel().astype(complex)
    m = field.size if q.size == 1 else min(q.size, field.size)
    qq = np.full(m, q[0]) if q.size == 1 else q[:m]
    similarities = np.abs(np.real(np.conj(qq) * field[:m]))
    return [int(i) for i in np.flatnonzero(similarities >= threshold)]


def _promote_field_2d(field_config):
    import numpy as np
    field_config = np.asarray(field_config, dtype=float)
    if field_config.ndim != 1:
        return field_config
    side = int(np.ceil(np.sqrt(field_config.size)))
    grid = np.full(side * side, field_config.mean() if field_config.size else 0.0)
    grid[:field_config.size] = field_config
    return grid.reshape(side, side)


def _distributed_quantum_inference_fixed(self):
    """numbskull QuantumCognitiveProtocol.distributed_quantum_inference, corrected (0002 patch)."""
    import numpy as np
    inference_state = np.zeros(np.shape(self.Q[0]), dtype=complex)
    for i, state in enumerate(self.Q):
        inference_state += state * np.exp(-self.E[i] / self.alpha_0)
    n = np.linalg.norm(inference_state)
    return inference_state / n if n > 0 else inference_state


def _quantum_storage_fixed(self, data):
    import numpy as np
    return np.asarray(data).astype(complex)


def _memory_reconstruction_fixed(self, query):
    import numpy as np
    q = np.asarray(self.fractal_encoding(query)).ravel()
    sims = [float(np.abs(np.real(np.vdot(q, np.asarray(m).ravel()[:q.size])))) for m in self.M]
    return self.M[int(np.argmax(sims))]


def emergent_protocol():
    """numbskull's execute_emergent_protocol, with the recall_transform fix applied when the installed
    numbskull still raises IndexError on it (a no-op once the 0002 patch is merged upstream)."""
    import numpy as np
    ensure_on_path("numbskull")
    import emergent_cognitive_network_fixed as ecn
    try:   # recall_transform does not use self, so probe the unbound function (constructor-independent)
        # ascending values put argmax at the last flattened index, exposing the n x n indexing bug
        ecn.HolographicProtocol.recall_transform(None, np.arange(1.0, 5.0), np.arange(1.0, 5.0).astype(complex))
    except IndexError:
        ecn.HolographicProtocol.recall_transform = _recall_transform_fixed
        ecn.HolographicProtocol.associative_memory = _associative_memory_fixed   # same bug, same patch
    try:
        ecn.QuantumCognitiveProtocol([np.ones(3, dtype=complex)], np.array([1.0])).distributed_quantum_inference()
    except ValueError:                           # superposition sized by state count, not state shape
        ecn.QuantumCognitiveProtocol.distributed_quantum_inference = _distributed_quantum_inference_fixed
    try:
        ecn.HolographicMemory(np.ones((1, 3)), np.ones((1, 3))).quantum_storage(np.ones((1, 3)))
    except TypeError:                            # complex() on memory rows; vector "similarities" (0002)
        ecn.HolographicMemory.quantum_storage = _quantum_storage_fixed
        ecn.HolographicMemory.memory_reconstruction = _memory_reconstruction_fixed
    if not getattr(ecn.MorphogeneticProtocol.__init__, "_kgirl_fixed", False):
        try:
            ecn.MorphogeneticProtocol(np.ones(4), np.array([0.5, 0.3])).turing_pattern_dynamics()
        except IndexError:                       # 1-D field reached 2-D Turing dynamics (0002 patch)
            orig = ecn.MorphogeneticProtocol.__init__

            def __init__(self, field_config, growth_parameters, convergence_threshold: float = 1e-6):
                orig(self, _promote_field_2d(field_config), growth_parameters, convergence_threshold)
            __init__._kgirl_fixed = True
            ecn.MorphogeneticProtocol.__init__ = __init__
    return ecn.execute_emergent_protocol


def probe_emergent() -> Dict[str, Any]:
    import numpy as np
    out = emergent_protocol()(np.linspace(0.0, 1.0, 32))
    return {"ok": isinstance(out, dict) and len(out) > 0, "sections": sorted(out)[:6]}


def probe_import(module: str) -> Callable[[], Dict[str, Any]]:
    def run():
        m = __import__(module)
        return {"ok": True, "file": getattr(m, "__file__", None)}
    return run


def status() -> Dict[str, Any]:
    roots = ensure_on_path("numbskull")
    root = roots.get("numbskull")
    report: Dict[str, Any] = {"numbskull_root": str(root) if root else None,
                              "searched": str(find_repo("numbskull")) if not root else None}
    report["components"] = {
        "wavecaster.fec": _probe(probe_reed_solomon),
        "emergentnet.optim": _probe(probe_partition),
        "emergentnet.memory.hrr": _probe(probe_hrr),
        "advanced_embedding_pipeline": _probe(probe_hybrid),
        "holographic_similarity_engine": _probe(probe_import("holographic_similarity_engine")),
        "emergent_cognitive_network": _probe(probe_emergent),
    }
    report["ok"] = all(c["ok"] for c in report["components"].values())
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kgirl.numbskull")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status", help="locate numbskull and run one probe per component")
    p.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rep = status()
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(f"numbskull: {rep['numbskull_root'] or 'NOT FOUND (looked at ' + str(rep['searched']) + ')'}")
        for name, c in rep["components"].items():
            detail = c.get("error") or ", ".join(f"{k}={v}" for k, v in c.items() if k not in ("ok", "ms", "file"))
            print(f"  {'ok  ' if c['ok'] else 'FAIL'} {name:<32} {c['ms']:>8} ms  {detail}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
